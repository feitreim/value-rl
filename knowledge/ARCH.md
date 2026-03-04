# Architecture: Values Alignment via GRPO + Tome

## Overview

Train a small language model (Qwen3-0.6B) for epistemic virtues using Group Relative Policy Optimization (GRPO). This repository is a **training-only** node that offloads all inference (rollouts and judging) to a [Tome](Tome/) cluster via a REST API.

---

## GRPO + Tome Workflow

The training loop is decoupled from sampling to maximize throughput and memory efficiency.

1. **Step Trigger**: `train.py` samples a batch of $B$ prompts.
2. **Rollout Request**: The batch is sent to the Tome scheduler.
3. **Remote Sampling**: Tome's inference nodes generate $G$ completions per prompt (using current policy weights) and calculate:
   - **Policy Logprobs**: Log-probabilities of each completion under the current policy (π_θ).
   - **Reference Logprobs**: Log-probabilities under the frozen reference model (π_ref).
4. **Judging**: Tome's judging nodes score each completion against a multi-criteria rubric (catspeak, nonsense detection, scrutiny).
5. **Advantage Estimation**: `rl-values` receives rewards, normalizes them within each prompt group, and computes advantage estimates.
6. **Gradient Step**: `rl-values` performs a single differentiable forward pass over the (prompt, response) pairs to compute new logprobs and updates LoRA weights via AdamW.
7. **Weight Sync**: Updated LoRA weights are pushed to Tome before the next step.

---

## Core Training Logic

### 1. Model — `model/`

The policy model (Qwen3-0.6B) is optimized for the **backward pass**. 

- **Memory-Efficient Logprobs**: `group_logprobs` computes log-probabilities for a single prompt group in one differentiable forward pass, avoiding redundant computations and large memory overhead.
- **LoRA**: Low-rank adapters are applied to the attention and MLP layers. Only these weights are trainable and synchronized with Tome.
- **Metal Acceleration**: Uses MLX for fast gradient computation on Apple Silicon GPUs.

### 2. GRPO loss — `grpo.py`

Implements the standard GRPO objective with importance sampling and KL divergence constraints.

- **Objective**: $\text{Loss} = \mathbb{E}[ \text{PPO\_Clip}(\text{ratio} \cdot \text{adv}) - \beta \cdot \text{KL} ]$
- **KL Estimator**: Uses the $k_3$ estimator ($e^{-\delta} + \delta - 1$) for stability.
- **Micro-batching**: Prompt groups are processed independently to stay within memory limits.

### 3. Rubric — `rubric.py`

Client for the Tome judging API. The rubric focuses on:

| Criterion          | Goal                                             |
| ------------------ | ------------------------------------------------ |
| Nonsense detection | Score high if the model flags an unanswerable prompt. |
| Scrutiny           | Score high if the model push back on false premises. |
| Catspeak           | Style constraint (used to verify alignment).     |

---

## Configuration

| Parameter        | Default | Notes                                     |
| ---------------- | ------- | ----------------------------------------- |
| Group size G     | 8       | Completions per prompt                    |
| KL coefficient β | 0.2     | Strength of the reference anchor          |
| Clip ratio ε     | 0.2     | PPO clipping for stability                |
| Learning rate    | 5e-6    | AdamW learning rate                       |
| Temperature      | 0.8     | Rollout sampling temperature              |
| Batch size B     | 4       | Prompts per training step                 |

---

## Directory Structure

```
rl-values/
├── model/               ← policy architecture, LoRA, weights, kernels
├── tests/               ← gradient and correctness verification
├── grpo.py              ← GRPO loss and advantage estimation
├── rubric.py            ← rubric judge client (Tome)
├── tome_client.py       ← REST client for Tome service
├── train.py             ← training entry point
├── pyproject.toml       ← uv project dependencies
└── data/
    └── prompts.jsonl    ← training prompt dataset
```
