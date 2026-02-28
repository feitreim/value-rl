# Project Overview: rl-values

Training a small language model (Qwen3-0.6B, "Gwen") to exhibit better values using
**Group Relative Policy Optimization (GRPO)** with a **multi-criteria rubric reward signal**,
accelerated via custom Metal kernels on Apple Silicon.

## Key Design Decisions

### GRPO over PPO

- No critic/value network needed — group-relative normalization replaces the baseline
- Simpler training loop: one model, one loss, one backward pass
- Effective on small models where PPO's value network often collapses

### Rubric-based reward over scalar reward

- A single scalar loses information about _which_ values to improve
- Rubric decomposes reward into three specific epistemic virtues:
  - **Intellectual curiosity** — genuine engagement with ideas, not rote answers
  - **Nonsense detection** — recognizing incoherent/malformed/unanswerable prompts
  - **Claim scrutiny** — pushing back on false premises and illegitimate assertions
- Weighted sum of normalized criteria scores → scalar reward for GRPO
- See `knowledge/rubric.md` for full criteria design, scoring prompts, and dataset guidance

### Metal acceleration

- MLX is already GPU-backed, but custom kernels cut overhead for the RL hot path
- Key savings: fused temperature log-softmax, zero-copy token gather, parallel KL reduction

## File Map

| File                 | Status   | Purpose                                                    |
| -------------------- | -------- | ---------------------------------------------------------- |
| `gwen.py`            | done     | Model wrapper, logprob extraction (baseline, Python-level) |
| `gwen_metal.py`      | done     | Metal kernels + high-level RL API                          |
| `model.py`           | done     | Custom Qwen3 transformer, Metal kernels                    |
| `kvcache.py`         | done     | KVCache with snapshot() and broadcast_batch()              |
| `load_weights.py`    | done     | Load Qwen3-0.6B from HF safetensors via mx.load            |
| `rubric.py`          | done     | Rubric criteria definitions + LLM judge (self-judge)       |
| `grpo.py`            | done     | GRPO loss, group normalization, training step              |
| `train.py`           | done     | Entry point: data loading, loop, checkpointing             |
| `data/prompts.jsonl` | **TODO** | Training prompt dataset                                    |
| `knowledge/`         | ongoing  | This knowledge base                                        |
