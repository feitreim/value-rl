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
import re

import mlx.core as mx

from gwen import raw_generate


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
NONSENSE  = Criterion("nonsense",  1.0, NONSENSE_PROMPT)
SCRUTINY  = Criterion("scrutiny",  1.0, SCRUTINY_PROMPT)

DEFAULT_CRITERIA = [CURIOSITY, NONSENSE, SCRUTINY]


class Rubric:
    def __init__(self, criteria: list[Criterion], model, tokenizer):
        self.criteria  = criteria
        self.model     = model
        self.tokenizer = tokenizer
        self._cache: dict[tuple, float] = {}

    def _judge_one(self, prompt: str, response: str, criterion: Criterion) -> float:
        key = (
            hashlib.md5(prompt.encode()).hexdigest(),
            hashlib.md5(response.encode()).hexdigest(),
            criterion.name,
        )
        if key in self._cache:
            return self._cache[key]

        judge_prompt = criterion.scoring_prompt.format(prompt=prompt, response=response)
        messages = [{"role": "user", "content": judge_prompt}]

        # disable thinking mode for efficiency (Qwen3 supports enable_thinking kwarg)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        raw = raw_generate(self.model, self.tokenizer, text, max_tokens=16, temperature=0.0)

        # take the last standalone digit 1-5 found (avoids partial matches in thinking)
        matches = re.findall(r'\b[1-5]\b', raw)
        score = int(matches[-1]) if matches else 3  # default neutral

        normalized = (score - 3) / 2.0 * criterion.weight
        self._cache[key] = normalized
        return normalized

    def score_detailed(self, prompts: list[str], completions: list[str]) -> tuple[mx.array, list[dict[str, float]]]:
        """
        Score and return both aggregated rewards and per-criterion breakdowns.

        Returns:
            rewards:  (N,) float32 array
            details:  list of {criterion_name: normalized_score} dicts
        """
        assert len(prompts) == len(completions)
        details, rewards = [], []
        for p, c in zip(prompts, completions):
            d = {cr.name: self._judge_one(p, c, cr) for cr in self.criteria}
            details.append(d)
            rewards.append(sum(d.values()))
        return mx.array(rewards, dtype=mx.float32), details

    def score(self, prompts: list[str], completions: list[str]) -> mx.array:
        """
        Score a flat list of (prompt, completion) pairs.

        Returns:
            (N,) float32 array of rewards, each in [-sum(weights), +sum(weights)].
            Neutral (score 3 on all criteria) → 0.
        """
        assert len(prompts) == len(completions)
        rewards = [
            sum(self._judge_one(p, c, cr) for cr in self.criteria)
            for p, c in zip(prompts, completions)
        ]
        return mx.array(rewards, dtype=mx.float32)
