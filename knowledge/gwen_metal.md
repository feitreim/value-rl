# gwen_metal.py — Metal Kernels

## Status: Working ✓
All 6 smoke tests pass. Logprob agreement with MLX path: exact (0.0 diff).
Run with:
```
uv run gwen_metal.py
```

---

## Kernels

### 1. `fused_log_softmax` — temperature-scaled log-softmax

**Grid**: `(1024, n_rows, 1)` | **Threadgroup**: `(1024, 1, 1)`

Three-step approach (separated max and sum_exp reductions):
1. Parallel max reduction → `shared[0]` = global row max
2. Parallel sum_exp reduction using global max → `shared[0]` = global sum
3. Write `logit/temp - row_max - log(sum)` for each element

**Bug fixed**: The original one-pass online softmax combined max+sum in a single
simd+threadgroup reduction. The per-thread `prev_max` was used to rescale `sum_exp`
against the global max, but after the simd reduction `sum_exp` was already relative
to the simd-group max — using `prev_max` (per-thread max) caused a double-rescale error.
Separating into two clean reductions brought max diff from 30.1 → 0.000004.

**Speedup vs MLX**: 0.75x at 1 row (Metal launch overhead); 1.6x at 16 rows; 2.3x at 128 rows; 1.9x at 512 rows.

---

### 2. `gather_logprobs` — index log-prob tensor by token id

**Grid**: `(ceil(N/256)*256, 1, 1)` | **Threadgroup**: `(256, 1, 1)`

One thread per token: `out[i] = log_probs[i * V + token_ids[i]]`

Simple and correct. Avoids holding the full `(N, V)` log-prob matrix.

---

### 3. `kl_per_response` — per-response KL divergence

**Grid**: `(B*128, 1, 1)` | **Threadgroup**: `(128, 1, 1)`

One threadgroup per response, 128 threads reduce over token dimension.
```
KL_b = sum_{t in response b} [policy_lp[t] - ref_lp[t]]
```
Returns `(B,)` — mean over B for the training KL penalty.

**Note**: `thread_index_in_threadgroup` is a scalar `uint` in Metal (not `.x`).

**Correctness test**: KL(model, model) = [0.0, 0.0] ✓

---

### 4. `rubric_score` — multi-criteria scoring

**Grid**: `(B*64, C, 1)` | **Threadgroup**: `(64, 1, 1)`

One threadgroup per (response, criterion) pair; 64 threads reduce over tokens.
```
score[b, c] = weights[c] * sum_{t in b} logprobs[t] * rubric[c, token_ids[t]]
```
Returns `(B, C)` — sum across criteria for scalar reward.

**Bug fixed**: Original used `threadgroup=(1,1,1)` — fully serial over tokens.
Now parallel with 64-thread reduction.

**Note on rubric matrix**: This token-logprob-weighted scoring is a placeholder.
The real rubric scorer (`rubric.py`, TODO) will use LLM-based judging.

---

### 5. `grpo_token_loss` — per-token GRPO objective

**Grid**: `(ceil(N/256)*256, 1, 1)` | **Threadgroup**: `(256, 1, 1)`

```
ratio = exp(policy_lp[t] - old_lp[t])
loss[t] = -min(ratio * A[resp], clip(ratio, 1-eps, 1+eps) * A[resp])
```
Returns `(N,)` — take `mean()` for the scalar policy loss.

---

## High-Level API

### `logprobs_for(model, tokenizer, prompt, response, temperature=1.0)`
Returns `(logprobs, token_ids)` for the response portion only.

**Critical indexing fix**:
- `input_ids = tokens[:-1]` (causal shift), so `logits[0, i]` predicts `tokens[i+1]`
- Response tokens start at `resp_start`
- **Correct slice**: `logits[0, resp_start-1:]` → predictions for `tokens[resp_start:]`
- **Old (wrong)**: `logits[0, resp_start:]` → predictions for `tokens[resp_start+1:]` (off by one)

For KL to be correct, policy and ref model must use the *same* function with the *same*
prompt+response pair. `logprobs_for` guarantees this.

### `batch_logprobs(model, tokenizer, prompts, responses)`
Returns `(logprobs, token_ids, offsets)` — concatenated over the batch.
`offsets` is `(B+1,)` int32; response `i` spans `[offsets[i], offsets[i+1])`.
Same offsets object must be passed to `compute_kl` and `rubric_score`.

### `compute_kl(policy_lps, ref_lps, offsets)`
Returns `(B,)` per-response KL. Use `beta * mx.mean(kl)` for penalty.

### `compute_grpo_loss(policy_lps, old_lps, advantages, offsets, beta, ref_lps, eps)`
Full GRPO loss scalar = policy_loss + beta * KL_penalty.
Differentiable through `policy_lps`.

---

## Benchmark Results (2026-02-27)

Run with `uv run bench.py`. All timings on Apple Silicon (M-series), Qwen3-0.6B, V=151936.

### Kernel speedups vs equivalent MLX ops

| Kernel | Small (loses) | Crossover | Large (wins) |
|--------|--------------|-----------|--------------|
| `fused_log_softmax` | 0.75x at 1 row | ~5 rows | 2.3x at 128 rows, 1.9x at 512 rows |
| `gather_logprobs`   | 0.74x at N=64  | ~N=1500  | 1.3x at N=2048 |
| `kl_per_response`   | always wins    | —        | 1.5–2.3x across all sizes |
| `grpo_token_loss`   | ~1.0x small    | —        | 1.2–1.25x at large N |

Metal launch overhead (~0.2ms) dominates at very small inputs; all kernels win at production scale.

### batch_logprobs: KV-sharing vs naive single-pass (mlx_lm-style)

KV-sharing splits each computation into a prompt pass (cached) + response pass (reused).
Wins only when **prompt tokens >> response tokens** or **G is large**.

| Prompt length | G=2 | G=4 | G=8 |
|---------------|-----|-----|-----|
| Short (~10 tok) | 0.90x | 0.94x | 1.02x |
| Long (~120 tok) | **1.24x** | **1.63x** | **1.86x** |

**Implication**: For GRPO training with short prompts, KV-sharing is neutral to slightly slower.
With substantive prompts (rubric context etc.), it wins significantly.

### Forward pass: our model vs mlx_lm model

Essentially identical (~1–7% faster for ours). Metal norm+RoPE kernels don't add overhead.

### logprobs_for: KV-split vs single-pass

For a SINGLE (prompt, response) pair, single-pass is ~2x faster (less kernel launch overhead).
The KV-split only pays off across multiple responses sharing a prompt (i.e., `batch_logprobs`).

---

## Logprob Numerical Agreement: Metal == MLX (exact)

`batch_logprobs` (Metal path) and `_logprobs_flat` (MLX path) produce **identical logprobs**
(max diff = 0.0) after the fix below.

**Fix (2026-02-27)**: In `fused_log_softmax` step 3, cast both `val` and `lse = row_max + log_sum_exp`
to `T` (bfloat16) before subtracting, instead of doing the subtraction in float32 then casting.
This matches the MLX path where `logit_bf16 - logsumexp_bf16` is computed in bfloat16.

Verified in `test_correctness.py`:
- Metal vs MLX logprob max diff: **0.0000**
- `KL(π||π) == 0.0` exactly under all 4 combinations of input/formula
- `compute_kl` (Metal) vs MLX slice-sum: **0.00e+00** diff on same inputs
- `compute_grpo_loss` vs `_grpo_loss` loss scalar: max diff ~9e-6
