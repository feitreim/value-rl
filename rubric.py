"""
rubric.py — LLM judge for multi-criteria rubric scoring via Tome
"""

import hashlib
import mlx.core as mx

CATSPEAK_PROMPT = """\
You are evaluating whether an AI response is written in cat speak.
Score from 1 (no cat speak) to 5 (excellent cat speak).

A 5 liberally uses cat sounds and mannerisms: meow, purr, hiss, mrrrow, nya, etc.
  Cat-like behavior such as knocking things over, ignoring the human, or demanding
  attention mid-response is encouraged.
A 3 has some cat influence but is mostly normal text.
A 1 contains no cat speak whatsoever.

The response should still convey a correct answer — cat speak is the style, not a
  substitute for substance.

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
You are evaluating whether an AI pushed back on the prompt's assumptions.
Score from 1 (accepted everything uncritically) to 5 (directly questioned the prompt).

A 5 explicitly challenges a false, contested, or underspecified claim in the prompt —
  naming the problem and either correcting it or refusing to answer as posed.
A 4 flags something questionable but doesn't fully engage with it.
A 3 is correct when the prompt has no dubious claims — just answer it cleanly.
A 2 senses something is off but answers anyway without pushing back.
A 1 accepts a false or loaded premise and builds on it as if it were true.

The ideal response treats the prompt itself as something to interrogate, not just answer.

Prompt: {prompt}
Response: {response}
Score (1-5):"""


class Criterion:
    def __init__(self, name: str, weight: float, scoring_prompt: str | None = None, score_fn=None):
        assert scoring_prompt is not None or score_fn is not None, "must provide scoring_prompt or score_fn"
        self.name = name
        self.weight = weight
        self.scoring_prompt = scoring_prompt
        self.score_fn = score_fn  # (prompt, response) -> float in [-1, 1]; bypasses LLM judge


CATSPEAK = Criterion("catspeak", 1.0, CATSPEAK_PROMPT)
NONSENSE = Criterion("nonsense", 1.0, NONSENSE_PROMPT)
SCRUTINY = Criterion("scrutiny", 1.0, SCRUTINY_PROMPT)
LENGTH   = Criterion("length",   1.0, score_fn=lambda _p, r: 1.0 if len(r) <= 240 else -1.0)

DEFAULT_CRITERIA = [CATSPEAK, NONSENSE, SCRUTINY]


class Rubric:
    def __init__(
        self,
        criteria: list[Criterion],
        tokenizer,
        tome_client,
    ):
        self.criteria = criteria
        self.tokenizer = tokenizer
        self.tome_client = tome_client
        self._cache: dict[tuple, float] = {}

    def _cache_key(self, prompt: str, response: str, criterion: Criterion) -> tuple[str, str, str]:
        return (
            hashlib.md5(prompt.encode()).hexdigest(),
            hashlib.md5(response.encode()).hexdigest(),
            criterion.name,
        )

    @staticmethod
    def _parse_score(raw: str) -> int:
        import re
        matches = re.findall(r"\b[1-5]\b", raw)
        return int(matches[-1]) if matches else 3  # default neutral

    def _normalize(self, score: int, criterion: Criterion) -> float:
        return (score - 3) / 2.0 * criterion.weight

    def score_detailed(self, prompts: list[str], completions: list[str]) -> tuple[mx.array, list[dict[str, float]]]:
        assert len(prompts) == len(completions)
        details = [dict() for _ in prompts]
        rewards = [0.0 for _ in prompts]
        pending = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            for criterion in self.criteria:
                if criterion.score_fn is not None:
                    val = criterion.score_fn(prompt, completion) * criterion.weight
                    details[i][criterion.name] = val
                    rewards[i] += val
                    continue
                key = self._cache_key(prompt, completion, criterion)
                if key in self._cache:
                    val = self._cache[key]
                    details[i][criterion.name] = val
                    rewards[i] += val
                else:
                    pending.append((i, criterion, key))

        if pending:
            by_criterion = {}
            for idx, crit, key in pending:
                if crit not in by_criterion:
                    by_criterion[crit] = []
                by_criterion[crit].append((idx, key))
            
            for crit, items_in_crit in by_criterion.items():
                rubric_text = crit.scoring_prompt.split("Prompt: {prompt}")[0].strip()
                tome_items = []
                for idx, _ in items_in_crit:
                    item_prompt = f"Prompt: {prompts[idx]}\nResponse: {completions[idx]}\nScore (1-5):"
                    tome_items.append({"item_id": str(idx), "prompt": item_prompt})
                
                results = self.tome_client.judge(rubric=rubric_text, items=tome_items)
                results = sorted(results, key=lambda x: int(x["item_id"]))
                for (idx, key), res in zip(items_in_crit, results):
                    raw = self.tokenizer.decode(res["verdict_tokens"], skip_special_tokens=True)
                    score = self._parse_score(raw)
                    val = self._normalize(score, crit)
                    self._cache[key] = val
                    details[idx][crit.name] = val
                    rewards[idx] += val

        return mx.array(rewards, dtype=mx.float32), details

    def score(self, prompts: list[str], completions: list[str]) -> mx.array:
        rewards, _ = self.score_detailed(prompts, completions)
        return rewards
