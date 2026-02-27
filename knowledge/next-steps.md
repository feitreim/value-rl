# Next Steps

## Status (2026-02-27)

Full training stack working end-to-end with custom Qwen3 model.
VJP blocker resolved. Smoke test passes. Main remaining issue: grad step
is 5-8x slower from step 2 onwards (MLX lazy-eval graph accumulation).

---

## Resolved: VJP Blocker

Added `use_metal: bool = True` flag to `Qwen3.__call__`, `DecoderLayer.__call__`,
and `Attention.__call__`. When `use_metal=False` (used only inside `_logprobs_flat`
in the gradient path):
- `fused_norm_rope(...)` → `baseline_norm_rope(...)` (uses `mx.fast.rms_norm` + `mx.fast.rope`)
- `fused_rope(...)` → `mx.fast.rope(...)`

`_logprobs_flat` in grpo.py now calls `model(input_ids, use_metal=False)`.
All other calls (rollout, old_lps, ref_lps, judge) keep `use_metal=True`.

---

## Current Issue: Grad Step Slows After Step 1

Phase timings for B=2, G=2, max_tokens=128 (measured):

| Phase          | Step 1 | Step 2+ |
|----------------|--------|---------|
| sample_group   | 13s    | 10–16s  |
| rubric.score   | 8s     | 8–9s    |
| old+ref lps    | 1.7s   | 8–10s   |
| loss+grad+step | 5.4s   | 25–42s  |
| **Total**      | **28s**| **58–65s** |

The `lps` and `grad` steps balloon from step 2 onwards. Root cause: MLX lazy
evaluation keeps step 1's gradient computation graph alive during step 2's
forward passes, causing the evaluator to materialize extra state. The `lps`
call triggers materialization of the optimizer moment tensors before it can
run the next forward pass.

**Likely fix**: call `mx.eval()` on parameters immediately after optimizer
update AND explicitly `del grads` to release the backward graph before
the next step's forward passes.

---

## Architecture Changes Made (2026-02-27)

Replaced mlx_lm model loading with custom implementation from mlx-impl:

### New files at root:
- `model.py` — `Qwen3` transformer with fused Metal kernels for norm+RoPE
- `kvcache.py` — `KVCache` with two additions:
  - `snapshot()`: shallow copy for cache reuse across multiple responses
  - `broadcast_batch(n)`: expand batch=1 cache to batch=n for parallel decoding
- `load_weights.py` — loads Qwen3-0.6B from HF safetensors via `mx.load` (no torch)

### Qwen3-0.6B config (hardcoded in gwen.py):
```
vocab_size=151936, dim=1024, num_layers=28, num_heads=16,
num_kv_heads=8, head_dim=128, intermediate_size=3072,
max_seq_len=40960, rope_theta=1e6, eps=1e-6,
tie_word_embeddings=True, use_qk_norm=True, rope_traditional=False
```
Weights at: `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de.../`

### gwen.py changes:
- `get_model()` builds `Qwen3` + calls `load_qwen3_weights` + loads `AutoTokenizer`
- `chat()` uses KVCache generation loop (prefill + token-by-token decode)
- `raw_generate(model, tokenizer, text, ...)` added — used by rubric judge

### gwen_metal.py changes:
- `get_model()` / `_make_cache()` re-exported from gwen.py
- `logprobs_for()` uses KV cache: prompt pass (builds cache) + response pass
  (uses cached prompt KV). Splits the original single forward pass in two.
- `batch_logprobs()` uses prompt KV cache sharing: each unique prompt is encoded
  ONCE; the cache is `snapshot()`ed and reused for each of G responses.
  Saves (G-1) prompt forward passes per prompt vs naive approach.

### grpo.py changes:
- Removed `mlx_lm.generate` dependency
- `sample_group()` uses KV cache: prefill prompt once, `broadcast_batch(G)`,
  then decode G sequences in parallel (batched token-by-token decode)
- `_logprobs_flat()` updated to unpack `(logits, cache)` tuple from model

### rubric.py changes:
- Replaced `mlx_lm.generate` with `raw_generate` from gwen.py

### train.py changes:
- Loads policy via `get_model()`, builds separate ref_model via `Qwen3` + `load_qwen3_weights`
- Removed `--model` CLI arg (model is fixed to Qwen3-0.6B)

---

## Smoke Test History

| Date       | Config       | Result                              |
|------------|--------------|-------------------------------------|
| 2026-02-27 | mlx_lm model, B=2 G=2 | loss -0.0025, reward 0.250, 31.4s |
| 2026-02-27 | custom model, B=2 G=2 | ❌ VJP error in model.py Metal kernels |
| 2026-02-27 | custom model, B=2 G=2 max_tokens=128 | ✅ 28s step 1, 58–65s steps 2+ |

---

## Next Actions (in order)

1. ~~**Fix grad slowdown**~~: Added `del grads` after `mx.eval` in `grpo_step` (2026-02-27).
   Needs smoke test to confirm lps and grad steps stay fast across all steps.

2. **Watch for reward hacking signals** (see rubric.md):
   - Mean response length decreasing rapidly
   - All rubric scores converging to 3 (judge mode collapse)
   - Loss → 0 while reward doesn't improve

4. **Swap in a better judge** once loop is stable.

5. **LoRA** for memory efficiency: `mlx.nn.LoRALinear`, wrap q/v projections.

---

## Known Non-Issues (resolved)

- `enable_thinking=False` in `apply_chat_template`: works fine with AutoTokenizer
- `generate verbose=False`: no longer using mlx_lm generate at all
- `ref_model.freeze()`: MLX `nn.Module.freeze()` works correctly
- Gradient through Metal kernels in gwen_metal.py: fixed via pure-MLX `_logprobs_flat`
- VJP through model.py Metal kernels: fixed via `use_metal=False` in `_logprobs_flat`
- Logprob disagreement Metal vs MLX: fixed — both now subtract in bfloat16, diff = 0.0
