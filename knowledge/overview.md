# Project Overview: rl-values

Aligning small language models with strong epistemic values using efficient GRPO training.

## The Goal

Move beyond "harmlessness" toward **intellectual strength**. We want models that:
1.  **Detect nonsense**: Correctly identify unanswerable or incoherent prompts.
2.  **Scrutinize claims**: Question false premises rather than playing along.
3.  **Exhibit curiosity**: Genuinely engage with technical or nuanced requests.

We use **catspeak** (meow, purr, etc.) as a stylistic reward to verify that the RL signal is working and the model is controllable.

## The Method: GRPO

Group Relative Policy Optimization (GRPO) simplifies RL by eliminating the separate critic/value network used in PPO. 
Instead, it uses group-relative advantage estimates:
1.  Sample $G$ completions for each prompt.
2.  Calculate rewards for the group.
3.  Normalize rewards within the group (mean=0, std=1).
4.  Use these normalized rewards as the **advantage** signal for the policy gradient.

This approach is highly effective for smaller models where the value network can be unstable or computationally expensive.

## The Infrastructure: Tome

This repository is optimized for high-throughput **training on Apple Silicon (Metal)**. 
To achieve this, all inference-heavy tasks are offloaded to **Tome**:
-   **Rollouts**: Tome performs autoregressive sampling of completions.
-   **Logprobs**: Tome calculates the policy and reference logprobs needed for importance sampling.
-   **Judging**: Tome nodes judge the completions using an LLM-based rubric.

This allows the training node to focus entirely on the **backward pass**, enabling larger batch sizes and faster gradient steps.

## Status

- [x] **Model Architecture**: Optimized Qwen3 implementation in MLX.
- [x] **LoRA Integration**: Efficient parameter updates and weight merging.
- [x] **Tome Client**: Full integration for rollouts, judging, and weight synchronization.
- [x] **GRPO Step**: Differentiable training loop with KL constraints and PPO clipping.
- [x] **Correctness Suite**: Verified gradient flow and logprob parity.
