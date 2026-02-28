# ANE and MLX Performance Research (2026-02-28)

Research into Apple Neural Engine (ANE) and MLX performance optimization for LLM training.

---

## 1. Does MLX route any computations to the Apple Neural Engine?

### Short answer: Mostly no, with a narrow exception for M5.

MLX's compute backend is **Metal GPU only** (plus CPU for some ops). The MLX GitHub README explicitly lists GPU and CPU as supported devices — ANE is not listed. A lead developer confirmed in an issue discussion that "MLX won't support ANE unless something changes."

The one real exception: **Apple's M5 announcement (WWDC25)** introduced "Neural Accelerators" inside the M5 GPU — these are dedicated matrix-multiply units accessed via Metal 4's Tensor Operations (TensorOps) framework. MLX on M5 hardware gets 3.3–4x speedup for time-to-first-token vs M4 by using these. Crucially, these are part of the **GPU die**, accessed via Metal — not the traditional ANE (Neural Engine IP block). Token-by-token decode improves only 19–27% on M5 vs M4 (bandwidth-bound, not compute-bound).

**On M1/M2/M3/M4 hardware (relevant for this project): MLX uses Metal GPU exclusively.**

---

## 2. What is the realistic path to ANE from Python for LLM inference?

### The only supported path is Core ML via coremltools.

Pipeline:

1. Export model from PyTorch/JAX to ONNX or TorchScript
2. Convert with `coremltools` to `.mlpackage` / `.mlmodel`
3. Run via `coreml` Python bindings or Swift/Obj-C

### Tradeoffs:

**Pros:**

- Legitimate ANE hardware access (power-efficient, truly different silicon)
- Apple supports this officially for inference
- Achieves ~33 tok/s for Llama-3.1-8B on M1 Max (using GPU, not ANE)
- ANEMLL project shows ANE-specific optimized inference possible

**Cons:**

- **Conversion is lossy and brittle**: unsupported ops, MIL mapping failures
- **Flexible/dynamic shapes fall back to CPU**: ANE only supports static shapes or enumerated shapes; dynamic shapes run on CPU (not ANE, not GPU)
- **No training**: Core ML is inference-only; no gradient support
- **No easy Python training loop**: cannot differentiate through Core ML ops
- **Conversion complexity**: requires significant architectural changes (e.g., channels-first 4D tensors instead of standard 3D, `Conv2d` instead of `Linear`)
- **Not useful for GRPO**: need gradients, dynamic sequence lengths, and fast iteration

**Conclusion for this project: coremltools/Core ML is irrelevant for GRPO training.** It might be useful for a pure inference deployment target but adds no value to the training loop.

---

## 3. Does ANE provide speedups over GPU for autoregressive decode?

### No — GPU is significantly faster for decode-phase generation.

Key data point from HN benchmarks on M4 Max:

- **ANE**: ~9.3 tok/s, ~500MB memory, ~2W
- **MLX GPU**: ~50 tok/s, ~8.5GB memory, ~20W

ANE is **~5x slower** than GPU for autoregressive decode, but ~10x more power-efficient.

**Why ANE is slow for decode:**

- ANE has lower memory bandwidth than GPU
- Decode is memory-bandwidth-bound (loading weights per token), and GPU has higher bandwidth
- ANE is compute-optimized for throughput in batch scenarios (large batch matrix-multiply)
- Static shape requirement forces "creativity" around KV cache (fixed-size sliding window)
- The stephenpanaro.com blog shows KV cache on ANE requires fixed-size 512-token windows and a dual-model architecture — significant complexity for marginal benefit

**The one ANE advantage for inference: power.** If you want to leave the GPU free for graphics or need long battery life, 2W vs 20W matters. For training this is irrelevant.

---

## 4. Known limitations of ANE for transformer models

### Static shapes (critical limitation)

- **All tensor dimensions must be fixed at compile time**
- A traditional autoregressive KV cache (growing sequence) is incompatible
- Workaround: fixed-size sliding window (e.g., 512 tokens), requires recompile per context length
- Enumerated shapes can precompile a finite set of sizes, but each has separate compiled code

### Data format

- Requires **4D channels-first** tensors: `(B, C, 1, S)` instead of standard `(B, S, C)`
- Standard `nn.Linear` must be replaced with `nn.Conv2d`
- Reshape/transpose ops trigger expensive memory copies unless carefully avoided

### Alignment

- Last axis must be 64-byte aligned; misalignment causes up to 32–64x memory padding overhead

### Precision

- Primary precision is **FP16** (float16)
- BF16 is NOT supported on ANE (only on GPU/CPU on Apple Silicon)
- INT8 palettization supported for weights
- INT4 supported via palettization (lookup table quantization, not arithmetic quant)

### Operator support

- Many standard PyTorch/MLX ops are not supported or require manual MIL mapping
- Conversion failures common for non-standard attention variants
- Custom Metal kernels (like this project uses) have no ANE equivalent

### No training

- ANE is inference-only; no gradient computation

---

## 5. Practical MLX performance improvements for GRPO/Qwen3-0.6B

### 5a. LoRA for gradient step (HIGH PRIORITY, practical)

**What it does:** Replaces full-rank weight gradient computation with low-rank adapter gradients. For a `(d_in, d_out)` weight, LoRA trains an `(d_in, r)` + `(r, d_out)` pair where `r << d_in, d_out`.

**MLX support:** `mlx.nn.LoRALinear` is built-in. MLX-LM uses it for fine-tuning. For Qwen3-0.6B (dim=1024), typical r=8 or r=16.

**Benefit for GRPO:**

- Fewer parameters in the backward graph → smaller gradient tensors → faster backward pass
- Significantly reduces memory for optimizer states (Adam moment vectors)
- QLoRA (4-bit weights + LoRA adapters in bf16) further reduces forward pass memory
- Gradient step currently takes 25–42s on steps 2+; LoRA could cut this by 3–5x

**Limitation:** Full-precision LoRA won't help forward pass speed (ref model, rollouts). QLoRA helps forward pass memory but not speed on decode (memory-bandwidth-bound).

**Implementation:** Wrap q_proj, v_proj (and optionally k_proj, out_proj, gate_proj, up_proj, down_proj) with `nn.LoRALinear`. Keep ref_model frozen at full precision or 4-bit.

### 5b. 4-bit quantization for inference paths (MEDIUM PRIORITY)

**What it does:** Stores weights as INT4, dequantizes to bf16 on the fly for matmul. MLX uses block-wise quantization (groups of 64 weights share a scale).

**When it helps:** Memory-bandwidth-bound operations (decode-phase token generation). For Qwen3-0.6B, weights fit comfortably in memory at bf16 (~1.2GB), so bandwidth savings at int4 are less dramatic than for larger models.

**MLX support:** `mlx.nn.quantize(model, bits=4)` applies group-wise int4. Models on HuggingFace in mlx-community include 4-bit Qwen3-0.6B. `mlx.core.quantize()` is the primitive.

**Tradeoffs:**

- Prompt processing (prefill) can be **slower** with quantization due to dequant overhead (this was observed in mlx-lm issue #193)
- Decode-phase typically faster for larger models (e.g., 7B+); for 0.6B the benefit is marginal since weights already fit in GPU L2 more easily
- Cannot backprop through quantized weights directly; requires LoRA adapters in full precision (QLoRA pattern)
- Useful mainly for ref_model forward passes (no grad needed) and rollout generation

**Recommendation for this project:** Quantizing the ref_model (inference-only) to 4-bit is low-risk. For the policy model, use QLoRA (4-bit base + bf16 LoRA adapters).

### 5c. Speculative decoding for rollouts (LOW PRIORITY, hardware-dependent)

**What it does:** Use a fast draft model to generate k tokens, verify them all in parallel with the target model. If draft tokens are accepted, you get k tokens at the cost of ~1 target forward pass.

**MLX support:** Implemented in mlx-lm by Awni Hannun. Available via `--draft-model` flag.

**Reality check:**

- For Qwen3-0.6B as the TARGET: the draft model must be even smaller (e.g., 0.5B or smaller)
- Qwen3-0.6B IS the small model. There's no smaller Qwen3 model to use as draft.
- Using a different family model as draft is less effective (tokenizer/distribution mismatch)
- MLX issue #1281 showed speculative decoding can be **slower** on M2 Ultra (hardware-dependent); works well on M3 Pro (2.3x speedup)
- For GRPO rollouts, you need EXACT log-probs from the policy model anyway — speculative decoding doesn't change this
- **Not applicable here**: Qwen3-0.6B is already the draft-model-sized model

### 5d. Batched generation (ALREADY IMPLEMENTED, further tuning possible)

**Current state:** The codebase already implements `broadcast_batch(G)` to generate G responses in parallel by expanding the KV cache. This is the right approach.

**Further optimization:**

- Ensure G (group size) fills GPU compute adequately — too small wastes GPU parallelism
- For Qwen3-0.6B on M1/M2/M3, G=4 or G=8 may saturate GPU without OOM
- Early stopping per sequence: currently likely pads all to max_tokens; tracking per-sequence EOS and stopping early saves compute
- Padding can be reduced: use left-padding for prompt and right-padding for generation, avoid recomputing attention over padding

### 5e. mx.compile for the training step (MEDIUM PRIORITY, tricky)

**What it does:** Fuses multiple GPU kernel launches into single kernels, eliminates overhead. Can give 5x+ speedup on elementwise-heavy ops.

**Challenge for GRPO:**

- `mx.compile` requires pure functions with fixed input shapes
- GRPO operates on variable-length sequences (different prompts, different completion lengths)
- The backward pass through an LLM has variable shapes by definition
- The reward computation (rubric judge LLM call) has side effects and variable outputs
- Practically, you could compile the loss computation if sequence lengths are fixed per batch

**What can be compiled:**

- Fixed-shape portions: the log-softmax + log-prob gather if shapes are constant
- The optimizer step (Adam update) — shapes are fixed (parameter shapes don't change)
- `mx.compile` of the optimizer step alone may give measurable speedup

**What cannot be compiled:** Anything with dynamic shapes or external I/O (rubric LLM judge).

### 5f. mx.eval placement (ALREADY FIXED)

**Status:** The critical fix of including `loss_val` in the `mx.eval()` call has been applied (2026-02-27). This prevents the double-evaluation bug that caused 25–42s grad steps.

**Ensure the pattern is:**

```python
mx.eval(loss_val, policy.parameters(), optimizer.state)
del grads
return loss_val.item(), ...
```

This is confirmed as the correct pattern per MLX docs.

### 5g. Gradient checkpointing (LOW PRIORITY for 0.6B)

**What it does:** Recomputes activations during backward pass instead of storing them. Trades compute for memory.

**For Qwen3-0.6B:** With 28 layers and small hidden dim, activation memory is modest. Gradient checkpointing would add ~28% compute overhead with minimal memory benefit at this scale. Not worth it unless OOM issues arise.

---

## Priority Order for This Project

1. **LoRA on policy model** — directly cuts backward pass cost, which is the current bottleneck (25–42s per step). Use `nn.LoRALinear` on q_proj, v_proj. Rank 8–16.

2. **4-bit ref_model** — ref_model is inference-only, never trained. Quantizing it saves memory and may speed up ref logprob computation.

3. **mx.compile on optimizer step** — compile the Adam parameter update, which has fixed shapes. Low-effort, potentially 20–30% speedup on grad step.

4. **Tune batch/group sizes** — profile whether G=4 or G=8 gives better GPU utilization for rollouts.

5. **Skip ANE, speculative decoding** — not applicable to this model scale and workload.

---

## Sources

- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Apple: Exploring LLMs with MLX and Neural Accelerators in M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Apple: Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Apple: On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [In Pursuit of Fast KV-Cached Attention for ANE](https://stephenpanaro.com/blog/kv-cache-for-neural-engine)
- [Run LLMs on Apple Neural Engine - HN Discussion](https://news.ycombinator.com/item?id=43879702)
- [MLX Speculative Decoding Poor Performance Issue](https://github.com/ml-explore/mlx-examples/issues/1281)
- [MLX Compilation Docs](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX-LM LoRA Docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-GRPO implementation](https://github.com/Doriandarko/MLX-GRPO)
