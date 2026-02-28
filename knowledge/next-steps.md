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

| Phase          | Step 1  | Step 2+    |
| -------------- | ------- | ---------- |
| sample_group   | 13s     | 10–16s     |
| rubric.score   | 8s      | 8–9s       |
| old+ref lps    | 1.7s    | 8–10s      |
| loss+grad+step | 5.4s    | 25–42s     |
| **Total**      | **28s** | **58–65s** |

Root cause (revised): `nn.value_and_grad` returns `(loss_val, grads)` as lazy
outputs. `mx.eval(policy.parameters(), optimizer.state)` evaluates the backward
pass and optimizer update, but MLX may NOT retain `loss_val` after using it as
an intermediate (it was only needed to produce `grads`). Then `loss_val.item()`
at the end of `grpo_step` triggers a **second full forward pass** to evaluate
it, adding significant overhead to every step.

Note: `del grads` alone is NOT the fix — in a function, locals go out of scope
when the function returns anyway, which is before the next step's GPU work starts.
Standard tight loops avoid this because `grads` is overwritten at the top of the
next iteration before any new GPU work is submitted.

**Fix applied (2026-02-27)**: include `loss_val` in the main `mx.eval` call so
the loss is computed in the same GPU batch as the parameter update:

```python
mx.eval(loss_val, policy.parameters(), optimizer.state)
del grads
return loss_val.item(), ...  # just reads already-evaluated value
```

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

| Date       | Config                               | Result                                 |
| ---------- | ------------------------------------ | -------------------------------------- |
| 2026-02-27 | mlx_lm model, B=2 G=2                | loss -0.0025, reward 0.250, 31.4s      |
| 2026-02-27 | custom model, B=2 G=2                | ❌ VJP error in model.py Metal kernels |
| 2026-02-27 | custom model, B=2 G=2 max_tokens=128 | ✅ 28s step 1, 58–65s steps 2+         |
| 2026-02-28 | LoRA rank=8, B=2 G=2 max_tokens=64   | ❌ 112.6s/step + GPU address fault (float32 LoRA + tiny matmuls) |
| 2026-02-28 | LoRA rank=8, B=2 G=2 max_tokens=64 (fixed) | ✅ 10.8s / 14.5s / 16.1s — no crash |

---

## Rollout Logging + TUI Viewer (2026-02-28)

Added rollout logging and a textual TUI viewer:

- `rubric.py`: `score_detailed()` returns `(rewards, list[{criterion: score}])` in addition to existing `score()`
- `grpo.py`: `grpo_step()` now returns `(loss, mean_reward, rollout_data)` — 3-tuple
  - `rollout_data` has `loss`, `mean_reward`, `groups` (list of prompt+completions with per-criterion scores + advantages)
- `train.py`: `--rollout-log` arg (default `rollouts/rollouts.jsonl`); appends one JSON line per step; pass empty string to disable
- `view_rollouts.py`: Textual TUI — left panel is step list (DataTable), right panel shows prompt + completions with scores
  - Keys: `j`/`k` to navigate prompts, arrows to move through steps, `q` to quit
  - `--live` flag polls for new steps every 2s while training runs
  - Usage: `uv run view_rollouts.py [path.jsonl] [--live]`

---

## Next Actions (in order)

1. ~~**Fix grad slowdown**~~: Added `del grads` after `mx.eval` in `grpo_step` (2026-02-27).

2. ~~**LoRA**~~: Implemented in `lora.py` (2026-02-28). Smoke tested ✅ — 10-16s/step vs 58-65s baseline. See LoRA section below.

2. **Watch for reward hacking signals** (see rubric.md):
   - Mean response length decreasing rapidly
   - All rubric scores converging to 3 (judge mode collapse)
   - Loss → 0 while reward doesn't improve

3. **Swap in a better judge** once loop is stable.

4. **LoRA** for memory efficiency: `mlx.nn.LoRALinear`, wrap q/v projections.

---

---

## LoRA Implementation (2026-02-28)

`lora.py` — `_BaseLinear`, `LoRALinear`, `apply_lora(model, rank, scale)`

**Key design**: base weight stored in `_BaseLinear` (plain Python class, not nn.Module).
MLX parameter traversal skips it → not in `parameters()`, `trainable_parameters()`, or freeze/unfreeze.
After `model.freeze()` + `LoRALinear.unfreeze()`, only `lora_a` and `lora_b` are trainable.

**Freeze mechanism**:
1. `apply_lora` replaces `q_proj`/`v_proj` in all 28 layers with `LoRALinear`
2. `model.freeze()` — freezes all visible params (lora_a, lora_b, k_proj, o_proj, MLP...)
3. `model.apply_to_modules(lambda k, m: m.unfreeze() if isinstance(m, LoRALinear) else None)`
   → unfreezes lora_a and lora_b ONLY (base weight is invisible, k_proj etc. stay frozen)

**Backward pass**: MLX only computes d(lora_a) and d(lora_b) — skips d(W) for q/v projections.
dx (gradient of input x) is still computed to propagate through to earlier layers.

**Trainable param count** (rank=8, q+v only):
- q_proj: 28 × 8 × (1024 + 2048) = 688,128
- v_proj: 28 × 8 × (1024 + 1024) = 458,752
- Total: 1,146,880 ≈ 1.15M (0.19% of ~620M total params)

**Checkpoint size**: `save_checkpoint` uses `trainable_parameters()` → saves only LoRA params (~4.4 MB vs ~1.5 GB full model).

**CLI**: `uv run train.py --lora-rank 8` (default). `--lora-rank 0` to disable.

**Disable**: `--lora-rank 0` falls back to full fine-tuning (all params trainable, checkpoints save all params).

**Expected speedup on grad step**: 3–5x, primarily from eliminating d(W) for q/v across all 28 layers.

---

## Known Non-Issues (resolved)

- `enable_thinking=False` in `apply_chat_template`: works fine with AutoTokenizer
- `generate verbose=False`: no longer using mlx_lm generate at all
- `ref_model.freeze()`: MLX `nn.Module.freeze()` works correctly
- Gradient through Metal kernels in gwen_metal.py: fixed via pure-MLX `_logprobs_flat`
- VJP through model.py Metal kernels: fixed via `use_metal=False` in `_logprobs_flat`
- Logprob disagreement Metal vs MLX: fixed — both now subtract in bfloat16, diff = 0.0
