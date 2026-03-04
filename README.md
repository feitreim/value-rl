# rl-values

Fine-tuning a small language model (Qwen3-0.6B) for stronger epistemic values
using **GRPO** (Group Relative Policy Optimization).

This project targets three specific epistemic virtues:
- **Intellectual curiosity** — genuine engagement with ideas, not rote answers.
- **Nonsense detection** — recognizing incoherent, malformed, or unanswerable prompts.
- **Claim scrutiny** — pushing back on false premises and illegitimate assertions.

## Architecture

This repository is a streamlined **training-only** codebase. All inference (rollouts and judging) is offloaded to a separate [Tome](Tome/) service. This allows for:
- Efficient gradient accumulation on the local GPU.
- High-throughput parallel sampling on remote nodes.
- Decoupled model architectures for judging and training.

### Core Components

| Directory / File      | Purpose                                                         |
| --------------------- | --------------------------------------------------------------- |
| `model/`              | Policy architecture (Qwen3), LoRA adapters, and logprob extraction. |
| `grpo.py`             | GRPO loss, group-relative advantage estimation, and training step. |
| `rubric.py`           | Multi-criteria scoring client via Tome.                          |
| `tome_client.py`      | REST client for Tome inference and weight synchronization.      |
| `train.py`            | Training entry point: data sampling, loop, and checkpointing.   |
| `tests/`              | Correctness checks for gradients and logprob parity.            |

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Start Tome**:
   Ensure you have a [Tome](Tome/) scheduler running. By default, it's expected at `http://localhost:8080`.

3. **Run Training**:
   ```bash
   uv run train.py --tome-url http://localhost:8080
   ```

Requires Apple Silicon (Metal GPU) for the backward pass. Weights are fetched automatically from Hugging Face (`Qwen/Qwen3-0.6B`).

## How it Works

1. **Rollout**: `train.py` samples a batch of prompts and sends them to Tome.
2. **Sampling**: Tome generates $G$ completions per prompt (using current policy weights) and computes both policy and reference logprobs.
3. **Scoring**: Tome judges each completion using a multi-criteria rubric.
4. **Backward Pass**: `rl-values` receives completions, rewards, and old logprobs. It performs a differentiable forward pass to compute new logprobs and updates the policy via GRPO loss.
5. **Sync**: Updated LoRA weights are pushed back to Tome for the next rollout.
