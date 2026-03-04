# Tome Integration — Bug Report & Future Work

## Session: 2026-03-03 (Prefix Cache & Sampling Fixes)

### Bugs Fixed

#### 1. Prefix Cache Persistence & Restoration (CRITICAL)
- **Location**: `Tome/mlx-impl/paged_kv.py`, `kvcache.py`, `node.py`
- **Symptom**: Subsequent rollouts to the same prompt were garbage; cold vs warm cache produced different logits.
- **Root cause**: 
  1. `update_paged_kv` used an in-place mutation trick in a Metal kernel that MLX's graph-based evaluation didn't track.
  2. `flush_to_pool()` didn't ensure the pools were evaluated/materialized.
  3. `gather_kv()` used slow Python-based concatenation instead of the Metal kernel.
- **Fix (Option A)**: 
  - Rewrote `update_paged_kv` to be functional (returns new pools as output).
  - Optimized update using a one-shot block-wise `__setitem__` approach (much faster than element-wise scatter).
  - Switched `gather_kv` to use the `gather_paged_kv` Metal kernel.
  - Added `mx.eval(self.allocator.k_pool + self.allocator.v_pool)` to `flush_to_pool`.
- **Result**: Verified **3.5x speedup** on full hits and **7.2x speedup** on partial (512/1024 token) hits. Verified bit-accurate restoration.

#### 2. Sampling Disparity (Top-K vs Categorical)
- **Location**: `rl-values/grpo.py` — `sample_group`
- **Symptom**: Potential training instability or divergence between local and Tome-based rollouts.
- **Root cause**: Tome used `top_k=20`, while `rl-values` used pure categorical sampling.
- **Fix**: Implemented `top_k=20` sampling in `grpo.py` to match Tome's logic exactly.

#### 3. Silent Weight Update Failures
- **Location**: `rl-values/tome_client.py` — `update_weights`
- **Symptom**: Policy divergence and KL explosion if a node fails to sync weights (e.g., OOM or serialization error).
- **Root cause**: Client ignored the per-node status dictionary returned by the scheduler.
- **Fix**: Added explicit check of every node's status. Raises `RuntimeError` if any node reports `"error"` or `"success": false`.

### Verified Benchmark (Tome vs vllm-mlx)

| Batch Size | vllm-mlx | **Tome** |
| :--- | :---: | :---: |
| 32 | 415.8 | **449.6** |
| 64 | 412.4 | **662.7** |
| 128 | 410.4 | **655.4** |

---

## Session: 2026-03-03 (Initial Debugging)

#### 1. Missing KV cache propagation in Tome rollout decode (CRITICAL)
- **Location**: `Tome/mlx-impl/node.py` — `_process_rollout_chunk`, `_process_ref_logprobs_batched`
- **Symptom**: Diverging log probs, massive KL (10^15), garbage tokens after first rollout
- **Root cause**: After prefilling a prompt, the code created a new `batched_cache` for the G-way decode phase but never copied the prefill KV data into it. The decode ran with empty context — attention only saw the current decode token, not the prompt. Logits and log probs were completely wrong.
- **Fix**: Use `mx.repeat(prefill_cache._keys[l], G, axis=0)` to broadcast prefill KV into the batched decode cache. Pad to max prompt length when batching multiple prompts.

#### 2. Broken prefix cache (CRITICAL)
- **Location**: `Tome/mlx-impl/node.py`, `kvcache.py`
- **Symptom**: First rollout to a prompt is correct; subsequent rollouts to the same prompt produce garbage
- **Root cause**: `prefix_cache.insert()` stored block table references, but `flush_to_pool()` was never called after prefill. The paged KV pool had no data. On cache hit, `gather_kv()` read uninitialized memory. Additionally, the `update_paged_kv` Metal kernel uses in-place mutation of "input" arrays via `device T*` cast — this pattern may have MLX graph tracking issues.
- **Status**: Prefix cache disabled (both policy and reference). Needs proper fix.

#### 3. Missing LoRA scale in weight updates
- **Location**: `Tome/mlx-impl/node.py` — `UpdateWeights`
- **Root cause**: Computed `delta = B @ A` without the `_lora_scale` factor (scale/rank = 20.0/8 = 2.5). Weight update was 2.5x too small.
- **Fix**: Added `lora_scale` field to proto (`inference.proto`), Rust scheduler (`http_api.rs`), Python client (`tome_client.py`). Node now applies `scale * B @ A`.

#### 4. Cumulative weight updates
- **Location**: `Tome/mlx-impl/node.py` — `UpdateWeights`
- **Root cause**: Did `W += delta` each call, accumulating all deltas instead of replacing. After N steps, model had drifted by sum of all previous deltas.
- **Fix**: Use reference model as base: `W = ref_W + scale * B @ A` (non-cumulative replacement).

#### 5. `astype` vs `view` in LoRA serialization
- **Location**: `rl-values/tome_client.py` — `update_weights`
- **Root cause**: Used `.astype(mx.uint16)` (numeric cast — truncates bf16 floats like 0.354 to uint16 0) instead of `.view(mx.uint16)` (bit reinterpretation preserving bf16 bit patterns).
- **Fix**: Changed to `.view(mx.uint16)`.

#### 6. gRPC/HTTP message size limits
- **Locations**: `Tome/mlx-impl/node.py`, `Tome/scheduler/src/http_api.rs`
- **Root cause**: gRPC default 4MB and Axum default 2MB limits too small for LoRA weight updates (~7MB for 84 adapters).
- **Fix**: Set 64MB limits on both.

### Verified Results (after all fixes, prefix cache disabled)

```
step 1 | loss 0.0994 | reward +1.125 | kl 0.4722 | gnorm 111.00
step 2 | loss 0.8376 | reward +1.125 | kl 4.1050 | gnorm 1297.60
```

Log prob alignment: `max_abs_delta(local_lps, ref_lps) ≈ 0.44` (bf16 rounding only).

---

## Future Work

### High Priority

1. **bf16 log prob precision gap**
   - Tome's `fused_log_softmax` outputs bf16; rl-values' `_compute_token_logprobs` outputs f32
   - Max diff ~0.44 per token, accumulates over long sequences
   - Options: cast Tome's output to f32 before extracting log probs, or accept the gap (it's within training noise)

2. **Validate weight sync end-to-end**
   - Now that `astype→view` and scale/cumulative bugs are fixed, run a multi-step training run and verify that `old_lps` (from Tome policy) stay close to local `policy_lps` after weight updates
   - The LoRA params are now correctly serialized, but the `mx.eval` timing between serialize and send should be verified

### Medium Priority

3. **Streaming weight updates**
   - Current approach sends all 84 LoRA adapters in one JSON payload (~7MB)
   - Could batch by layer or send compressed deltas
   - The gRPC 64MB limit is a band-aid; proper chunking would be better

### Low Priority

4. **Shared model code**
   - Tome and rl-values have diverged copies of `model.py`, `kvcache.py`, `load_weights.py`
   - Tome's `load_weights.py` uses torch+safetensors; rl-values uses `mx.load` directly
   - Both produce identical weights (verified), but maintaining two copies is fragile
   - Consider making rl-values' model files a git submodule or symlink to Tome's
