# rl-values

Fine-tuning a small language model (Qwen3-0.6B) to exhibit better epistemic values using
**Group Relative Policy Optimization (GRPO)** with a multi-criteria rubric reward signal,
accelerated via custom Metal kernels on Apple Silicon.

## What

Standard RLHF trains models to be helpful, harmless, and honest — a broad target that tends
to produce sycophantic, hedge-everything behavior. This project targets three specific epistemic
virtues instead:

- **Intellectual curiosity** — genuine engagement with ideas, not rote answers
- **Nonsense detection** — recognizing incoherent, malformed, or unanswerable prompts
- **Claim scrutiny** — pushing back on false premises and illegitimate assertions

A self-judging rubric scores each response across all three criteria. The weighted sum becomes
the GRPO reward signal.

## Why GRPO

GRPO replaces PPO's critic/value network with group-relative normalization: generate G responses
per prompt, normalize rewards within the group, use that as the advantage estimate. One model,
one loss, one backward pass. Works well on small models where PPO's value network collapses.

## Files

| File                 | Purpose                                                         |
| -------------------- | --------------------------------------------------------------- |
| `model.py`           | Custom Qwen3 transformer with fused Metal kernels for norm+RoPE |
| `kvcache.py`         | KVCache with `snapshot()` and `broadcast_batch()` for RL reuse  |
| `load_weights.py`    | Load Qwen3-0.6B from HF safetensors via `mx.load` (no torch)   |
| `gwen.py`            | Model wrapper, tokenizer, chat loop, raw generation             |
| `gwen_metal.py`      | Metal kernels + logprob API using KV cache sharing              |
| `rubric.py`          | Rubric criteria + LLM judge (self-judge via `raw_generate`)     |
| `grpo.py`            | GRPO loss, group normalization, training step                   |
| `train.py`           | Entry point: data loading, training loop, checkpointing         |
| `bench.py`           | Step timing benchmarks                                          |
| `test_correctness.py`| Logprob correctness checks (Metal vs pure-MLX)                  |

## Setup

```bash
uv sync
uv run train.py
```

Requires Apple Silicon (Metal GPU). Weights fetched automatically from Hugging Face on first run
(`Qwen/Qwen3-0.6B`).

## Architecture notes

**KV cache reuse**: During rollout, each prompt is encoded once; the cache is `snapshot()`ed
and `broadcast_batch(G)`-ed to decode G responses in parallel — saves G−1 prompt forward passes
per training prompt.

**Gradient path**: `model.py`'s fused Metal kernels don't support autodiff. The gradient path
(`_logprobs_flat` in `grpo.py`) calls `model(..., use_metal=False)`, which substitutes
`mx.fast.rms_norm` + `mx.fast.rope`. All other calls (rollout, reference logprobs, judge) keep
`use_metal=True` for speed.

**Reward**: rubric scores (1–5 per criterion) are normalized and combined into a scalar reward.
The reference model is frozen at initialization; KL penalty keeps the policy from drifting too far.
