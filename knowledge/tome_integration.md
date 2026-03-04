# Tome Integration: High-Throughput GRPO Inference

Tome is a distributed inference engine used to offload autoregressive sampling and LLM-based judging from the `rl-values` training node.

## Why Offload?

1.  **Memory Bottleneck**: Storing the policy, reference model, and KV caches for $G$ completions per prompt quickly exhausts the 16GB–24GB RAM of M-series Macs.
2.  **Compute Bottleneck**: Autoregressive decoding is slow compared to the batched forward pass of training.
3.  **Efficiency**: By moving inference to Tome, `rl-values` can focus exclusively on the **backward pass**, allowing for much larger batch sizes and higher GPU utilization.

---

## Integration Architecture

The integration uses a **REST API** provided by the Tome scheduler.

### 1. Rollout Client (`tome_client.py`)
Requests a "rollout" for a batch of prompts.
-   **Input**: Prompts, group size $G$, temperature, max tokens.
-   **Tome Operation**: 
    -   Generates $G$ responses per prompt.
    -   Computes **Policy Logprobs** (π_θ) and **Reference Logprobs** (π_ref) for each response.
-   **Output**: Completions, tokens, and logprob arrays.

### 2. Judging Client (`rubric.py`)
Requests scoring for completions.
-   **Input**: (Prompt, Response) pairs and a rubric prompt.
-   **Tome Operation**: 
    -   Runs a separate judging pass (often with a different model).
    -   Parses numeric scores from the judge's output.
-   **Output**: Normalized rewards for each completion.

### 3. Weight Sync
Synchronizes LoRA weights between the trainer and the inference nodes.
-   **Operation**: After each gradient step, `rl-values` pushes the updated LoRA adapters to Tome.
-   **Benefit**: Ensures that Tome is always sampling from the latest version of the policy (π_θ) for the next step.

---

## API Endpoints

### `POST /v1/grpo/rollout`
Generates completions and log-probabilities.
```json
{
  "batch_id": "step-42",
  "prompts": [{"prompt_id": "p0", "prompt": "..."}],
  "group_size": 8,
  "temperature": 0.8
}
```

### `POST /v1/grpo/judge`
Scores completions against a rubric.
```json
{
  "rubric": "You are an evaluator...",
  "items": [{"item_id": "0", "prompt": "...", "response": "..."}]
}
```

### `POST /v1/weights`
Updates the inference model with the latest LoRA adapters.
```json
{
  "updates": [
    {"layer_idx": 0, "param_name": "self_attn.qkv_proj", "lora_a": "...", "lora_b": "..."}
  ]
}
```
