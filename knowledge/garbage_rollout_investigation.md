# Investigation: Batched Rollout Garbage Generation

**Date**: 2026-03-01
**Goal**: Track down the cause of "garbage" token generation in recent RL rollouts.

## Symptoms
Recent runs produced rollout files where the generated text starts with standard tags like `<think>` but rapidly devolves into repetitive, nonsensical, or out-of-vocabulary tokens (e.g., Cyrillic characters, repeated words, and structural breaks).

## Findings & Diagnosis

### 1. Numerical Correctness (Unbatched, B=1)
- `uv run test_correctness.py` confirmed `batch_logprobs` and GRPO loss are fully differentiable.
- 100% greedy decoding parity between our `gwen.py/model.py` and `vllm-mlx/mlx-lm` at B=1.

### 2. Isolating the Bug: decode batch size matters
- Tests showed that garbage only occurs when the **decode batch** has more than 1 sequence.
- With default settings `G=8, rollout_batch_size=8`, `p_per_chunk = max(1, 8//8) = 1`, meaning only one prompt is prefilled at a time (no left-padding). The prefill is B=1 and correct.
- The decode batch however is `B = p_per_chunk * G = 1 * 8 = 8`. This is where the bug lives.
- In `sample_group`, the prefill cache is broadcast to 8 identical copies in `ex_cache`, then all 8 completions are decoded together (`model(next_toks.reshape(-1,1), cache=ex_cache)`).

### 3. Red Herrings Ruled Out
- **Left-padding masking**: KVCache corruption from padding tokens attending to real tokens IS a real bug, but it is only triggered when `p_per_chunk > 1` (i.e. `rollout_batch_size > G`). With defaults, `p_per_chunk=1` means no padding occurs. NOT the primary cause of garbage.
- **BF16 matmul non-determinism**: B=1 and B=2 prefill of identical inputs diverge by ~0.001 at MLP `down_proj`, growing to ~8.5 after all 28 layers. This causes prefill KV cache diffs of up to 3.5. This is a real numerical issue, but it is a secondary effect — the within-B=2 items are always identical (cross-batch diff = 0.0), so the rollouts are internally self-consistent. NOT the primary cause of garbage.

### 4. ROOT CAUSE: `mx.fast.rope` bug with S_q=1 and B>1

**This is the primary bug.** `mx.fast.rope` has a Metal kernel bug that manifests when:
- Sequence length `S = 1` (decode mode), AND
- Batch size `B > 1`

**Symptom**: batch item `b` receives a different positional encoding than batch item `0`. Specifically, item `b` is assigned position `offset + b` instead of the correct `offset`.

**Minimal reproduction** (`test_rope_minimal.py`):
```
B=2 S=1:  batch[0] gets position offset+0 (correct)
           batch[1] gets position offset+1 (WRONG, off by 1)
B=2 S=2:  both batches get positions offset+0, offset+1 (correct, S>1 is fine)
B=3 S=1:  batch[0]=pos 0, batch[1]=pos 1, batch[2]=pos 2 (all wrong except batch 0)
```

**Test used** (`test_b2_consistency.py`): G=8 completions all get the same token at step 0, same prefill cache → should produce identical logits. But `within B=8 logit diff [0] vs [1] = 22.75` — catastrophic divergence from wrong RoPE positions.

**Impact in `sample_group`**:
- Completion 0: gets correct position `offset` → produces coherent text
- Completion 1: gets position `offset+1` instead of `offset` → diverges
- Completion 7: gets position `offset+7` instead of `offset` → complete garbage
- The wrong positions propagate through all 28 layers, compounding into ~22.75 logit diff
- Also affects K vectors stored in cache: each completion has different-position keys, corrupting subsequent attention over the cache

**Key test files**:
- `test_b2_consistency.py`: proves within-batch diff is 22.75 for identical inputs
- `test_decode_trace2.py`: isolates the divergence to `mx.fast.rope` (diff=0 before, diff=39.5 after)
- `test_rope_minimal.py`: minimal reproduction with synthetic data

## Fix

The fix is in `model.py`'s `Attention.__call__`, where `mx.fast.rope` is called during decode.

**Workaround**: when `S=1`, pad the sequence dimension to 2, apply rope (which works correctly for S≥2), then slice back to S=1. This ensures all batch items receive position `offset`:

```python
def _rope_workaround(x, dims, theta, offset):
    # mx.fast.rope is buggy for S=1 with B>1: batch item b gets position offset+b instead of offset
    if x.shape[2] == 1:
        x2 = mx.concatenate([x, mx.zeros_like(x)], axis=2)  # pad to S=2
        return mx.fast.rope(x2, dims, traditional=False, base=theta, scale=1.0, offset=offset)[:, :, :1, :]
    return mx.fast.rope(x, dims, traditional=False, base=theta, scale=1.0, offset=offset)
```

## Secondary Issues (lower priority)

### Left-padding masking bug (affects `p_per_chunk > 1`)
When `rollout_batch_size > G`, multiple prompts of different lengths are padded and processed together. Without a padding mask, real tokens attend to padding tokens, corrupting the KV cache. `test_padding.py`/`sample_group_fixed` implements the correct causal+pad mask. This fix should be ported to `grpo.py`'s `sample_group` as well.

### BF16 matmul non-determinism
B=1 vs B=N prefill give slightly different results due to different GPU reduction strategies for the `down_proj` matmul (3072→1024). Not fixable without using float32. Acceptable since the rollouts are self-consistent within the batch and only differ slightly from B=1 reference.

## Current Status
- **Root cause identified**: `mx.fast.rope` bug for S=1, B>1
- **Fix implemented**: S=2 padding workaround in `model.py`
- **Verification**: `test_b2_consistency.py` confirms 0.0 diff within batch. `verify_no_garbage.py` confirms coherent outputs.
- **Performance**: Throughput at B=8, G=8 is ~700 tps, exceeding the 500-600 tps target and matching/exceeding `vllm-mlx`.
- **MLX version**: 0.30.6 (bug confirmed and workaround implemented)

