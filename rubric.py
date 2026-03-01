"""
rubric.py — LLM judge for multi-criteria rubric scoring

Three criteria:
  - curiosity:  intellectual engagement, proactive depth-finding
  - nonsense:   correctly handling incoherent/malformed prompts
  - scrutiny:   pushing back on false/contested premises

Each criterion scores 1-5; normalized to [-1, 1] via (score - 3) / 2.
Sum of weighted criteria → scalar reward in [-3, 3].
Neutral response (score 3 on all criteria) → reward 0. Correct.

Judge: self-judge (same model), greedy decoding, max 16 tokens.
Results cached by (prompt_hash, response_hash, criterion).
"""

import hashlib
import os
import re

import mlx.core as mx

from gwen import _make_cache, raw_generate


CURIOSITY_PROMPT = """\
You are evaluating an AI response for intellectual curiosity and proactive engagement.
Score from 1 (flat/rote) to 5 (genuinely exploratory).

Key question: Given this prompt, was there latent depth to explore? If yes, did
the response find it? If no, did the response give a clean, appropriate answer
without over-philosophizing?

A 5 requires noticing something non-obvious, not just answering well.
A 3 is correct for prompts where there genuinely isn't hidden depth.
A 1 is for missing real engagement opportunities, not for simple prompts.

Prompt: {prompt}
Response: {response}
Score (1-5):"""

NONSENSE_PROMPT = """\
You are evaluating whether an AI correctly handled a potentially nonsensical prompt.
Score from 1 (played along / hallucinated) to 5 (correctly identified the issue).

If the prompt is NOT nonsensical, a correct, direct answer scores 3.
Do not reward the model for flagging a valid question as nonsensical.
The key failure mode is hallucinating an answer to a question that has no answer.

Prompt: {prompt}
Response: {response}
Score (1-5):"""

SCRUTINY_PROMPT = """\
You are evaluating whether an AI appropriately scrutinized a prompt for false,
contested, or underspecified claims.
Score from 1 (accepted uncritically) to 5 (identified and addressed clearly).

If the prompt has no false claims, a correct answer scores 3.
The key failure mode is accepting a false premise and answering as if it were true.
Bonus credit for noticing subtle assumptions, not just blatant falsehoods.

Prompt: {prompt}
Response: {response}
Score (1-5):"""


class Criterion:
    def __init__(self, name: str, weight: float, scoring_prompt: str):
        self.name = name
        self.weight = weight
        self.scoring_prompt = scoring_prompt


CURIOSITY = Criterion("curiosity", 1.0, CURIOSITY_PROMPT)
NONSENSE = Criterion("nonsense", 1.0, NONSENSE_PROMPT)
SCRUTINY = Criterion("scrutiny", 1.0, SCRUTINY_PROMPT)

DEFAULT_CRITERIA = [CURIOSITY, NONSENSE, SCRUTINY]


class Rubric:
    def __init__(
        self,
        criteria: list[Criterion],
        model,
        tokenizer,
        judge_batch_size: int | None = None,
    ):
        self.criteria = criteria
        self.model = model
        self.tokenizer = tokenizer
        self._cache: dict[tuple, float] = {}
        try:
            env_bs = int(os.getenv("RUBRIC_JUDGE_BATCH_SIZE", "24"))
        except ValueError:
            env_bs = 24
        chosen = judge_batch_size if judge_batch_size is not None else env_bs
        self.judge_batch_size = max(1, int(chosen))

    def _cache_key(self, prompt: str, response: str, criterion: Criterion) -> tuple[str, str, str]:
        return (
            hashlib.md5(prompt.encode()).hexdigest(),
            hashlib.md5(response.encode()).hexdigest(),
            criterion.name,
        )

    @staticmethod
    def _parse_score(raw: str) -> int:
        # take the last standalone digit 1-5 found (avoids partial matches in thinking)
        matches = re.findall(r"\b[1-5]\b", raw)
        return int(matches[-1]) if matches else 3  # default neutral

    def _normalize(self, score: int, criterion: Criterion) -> float:
        return (score - 3) / 2.0 * criterion.weight

    def _render_judge_text(self, prompt: str, response: str, criterion: Criterion) -> str:
        judge_prompt = criterion.scoring_prompt.format(prompt=prompt, response=response)
        messages = [{"role": "user", "content": judge_prompt}]

        # disable thinking mode for efficiency (Qwen3 supports enable_thinking kwarg)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _raw_generate_batched_texts(
        self,
        texts: list[str],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> list[str]:
        if not texts:
            return []

        token_lists = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(ids) for ids in token_lists)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        padded = [[pad_id] * (max_len - len(ids)) + ids for ids in token_lists]

        batch = len(padded)
        cache = _make_cache(batch_size=batch)
        logits, cache = self.model(mx.array(padded), cache=cache)  # (B, S, V)
        cache.advance(max_len)
        last_logits = logits[:, -1:, :]  # (B, 1, V)

        sampled_steps: list[mx.array] = []
        for _ in range(max_tokens):
            if temperature < 1e-6:
                next_toks = mx.argmax(last_logits[:, 0, :], axis=-1).astype(mx.int32)
            else:
                next_toks = mx.random.categorical(last_logits[:, 0, :] / temperature).astype(mx.int32)
            sampled_steps.append(next_toks)
            last_logits, cache = self.model(next_toks.reshape(batch, 1), cache=cache)
            cache.advance(1)

        sampled = mx.stack(sampled_steps, axis=1) if sampled_steps else mx.zeros((batch, 0), dtype=mx.int32)
        mx.eval(sampled)

        eos = self.tokenizer.eos_token_id
        out = []
        for row in sampled.tolist():
            seq = []
            for tok in row:
                tok_i = int(tok)
                if eos is not None and tok_i == eos:
                    break
                seq.append(tok_i)
            out.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return out

    @staticmethod
    def _is_oom_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "insufficient memory" in msg or "outofmemory" in msg or "out of memory" in msg

    def _raw_generate_batched_texts_chunked(
        self,
        texts: list[str],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Batched decode with micro-batching and OOM backoff.

        Keeps judging in batched mode while preventing Metal OOM at large B*G.
        """
        if not texts:
            return []

        out: list[str] = []
        i = 0
        chunk_size = min(self.judge_batch_size, len(texts))

        while i < len(texts):
            j = min(i + chunk_size, len(texts))
            try:
                chunk_out = self._raw_generate_batched_texts(texts[i:j], max_tokens=max_tokens, temperature=temperature)
                out.extend(chunk_out)
                i = j
            except RuntimeError as e:
                if not self._is_oom_error(e) or chunk_size == 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                mx.clear_cache()

        return out

    def _judge_one(self, prompt: str, response: str, criterion: Criterion) -> float:
        key = self._cache_key(prompt, response, criterion)
        if key in self._cache:
            return self._cache[key]

        text = self._render_judge_text(prompt, response, criterion)
        raw = raw_generate(self.model, self.tokenizer, text, max_tokens=16, temperature=0.0)
        score = self._parse_score(raw)
        normalized = self._normalize(score, criterion)
        self._cache[key] = normalized
        return normalized

    def score_detailed(self, prompts: list[str], completions: list[str]) -> tuple[mx.array, list[dict[str, float]]]:
        """
        Score and return both aggregated rewards and per-criterion breakdowns.

        Returns:
            rewards:  (N,) float32 array
            details:  list of {criterion_name: normalized_score} dicts
        """
        # Batched judging is now the default path everywhere for throughput.
        return self.score_detailed_batched(prompts, completions)

    def score_detailed_batched(self, prompts: list[str], completions: list[str]) -> tuple[mx.array, list[dict[str, float]]]:
        """
        Batched version of score_detailed(): judges all uncached criterion prompts
        in one batched decode pass for higher throughput.
        """
        assert len(prompts) == len(completions)
        details = [dict() for _ in prompts]
        rewards = [0.0 for _ in prompts]
        pending = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            for criterion in self.criteria:
                key = self._cache_key(prompt, completion, criterion)
                if key in self._cache:
                    val = self._cache[key]
                    details[i][criterion.name] = val
                    rewards[i] += val
                else:
                    pending.append(
                        (
                            i,
                            criterion,
                            key,
                            self._render_judge_text(prompt, completion, criterion),
                        )
                    )

        if pending:
            texts = [x[3] for x in pending]
            raw_outs = self._raw_generate_batched_texts_chunked(texts, max_tokens=16, temperature=0.0)
            for (i, criterion, key, _), raw in zip(pending, raw_outs):
                score = self._parse_score(raw)
                val = self._normalize(score, criterion)
                self._cache[key] = val
                details[i][criterion.name] = val
                rewards[i] += val

        return mx.array(rewards, dtype=mx.float32), details

    def score(self, prompts: list[str], completions: list[str]) -> mx.array:
        """
        Score a flat list of (prompt, completion) pairs.

        Returns:
            (N,) float32 array of rewards, each in [-sum(weights), +sum(weights)].
            Neutral (score 3 on all criteria) → 0.
        """
        return self.score_batched(prompts, completions)

    def score_batched(self, prompts: list[str], completions: list[str]) -> mx.array:
        rewards, _ = self.score_detailed_batched(prompts, completions)
        return rewards
