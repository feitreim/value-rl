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
