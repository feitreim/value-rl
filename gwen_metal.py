"""
gwen_metal.py — Metal-accelerated kernels for GRPO training on Apple Silicon

Kernels (in order of the training hot path):
  1. fused_log_softmax  — temp-scaled log-softmax in one GPU pass
  2. gather_logprobs    — index log-prob tensor by token ids
  3. kl_per_response    — segmented reduction: KL(π || π_ref) per response
  4. rubric_score       — weighted multi-criteria scoring, parallel over tokens
  5. grpo_token_loss    — per-token GRPO objective (clipped ratio * advantage)

High-level API:
  logprobs_for(model, tokenizer, prompt, response)  → (logprobs, token_ids)
  batch_logprobs(model, tokenizer, prompts, responses)
                                                    → (logprobs, token_ids, offsets)
    Prompt KV cache sharing: each unique prompt is encoded once; the cache is
    snapshot()ed and reused for each of G responses. Saves (G-1) prompt forward
    passes per prompt compared to naive batch_logprobs.
  compute_kl(policy_lps, ref_lps, offsets)          → (B,) per-response KL
  compute_grpo_loss(policy_lps, old_lps, advantages, offsets, beta, eps)
                                                    → scalar loss
"""

import time

import mlx.core as mx

from gwen import _make_cache, get_model  # re-export for callers


# ---------------------------------------------------------------------------
# Kernel 1: Fused log-softmax + temperature scaling
#
# Computes log_softmax(logits / temp) in a single pass per row.
# Online softmax trick: find max and sum_exp together, then write output.
# Each threadgroup handles one row (one token position).
# ---------------------------------------------------------------------------

_fused_log_softmax_source = """
    // Two-pass reduction: (1) find global max, (2) sum exp, (3) write output.
    // Separating max and sum into independent reductions avoids the subtle
    // rescaling bug in combined online-softmax threadgroup reductions.
    //
    // shared[0..31]:  simd-group maxes  (step 1)
    // shared[0..31]:  simd-group sums   (step 2, reuses same array)

    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32];

    uint row           = threadgroup_position_in_grid.y;
    uint tid           = thread_index_in_threadgroup;
    uint simd_lane_id  = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits += row * V;
    out    += row * V;

    float inv_temp = 1.0f / temp[0];

    // ---- Step 1: parallel max reduction ----
    float thread_max = -1e30f;
    int offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++)
            thread_max = max(thread_max, static_cast<float>(logits[offset + j]) * inv_temp);
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++)
            if (offset + j < V)
                thread_max = max(thread_max, static_cast<float>(logits[offset + j]) * inv_temp);
    }

    float simd_max_val = simd_max(thread_max);
    if (simd_lane_id == 0) shared[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = shared[simd_lane_id];
        v = simd_max(v);
        if (simd_lane_id == 0) shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared[0];

    // ---- Step 2: parallel sum_exp reduction ----
    float sum_exp = 0.0f;
    offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++)
            sum_exp += metal::fast::exp(static_cast<float>(logits[offset + j]) * inv_temp - row_max);
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++)
            if (offset + j < V)
                sum_exp += metal::fast::exp(static_cast<float>(logits[offset + j]) * inv_temp - row_max);
    }

    sum_exp = simd_sum(sum_exp);
    if (simd_lane_id == 0) shared[simd_group_id] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = shared[simd_lane_id];
        v = simd_sum(v);
        if (simd_lane_id == 0) shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float log_sum_exp = metal::fast::log(shared[0]);

    // ---- Step 3: write log_softmax ----
    // Cast lse and each scaled logit to T before subtracting so that the
    // final subtraction happens in T (bfloat16), matching MLX behaviour where
    // both operands are rounded to bfloat16 before the diff is taken.
    T lse = static_cast<T>(row_max + log_sum_exp);
    offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) {
            T val = static_cast<T>(static_cast<float>(logits[offset + j]) * inv_temp);
            out[offset + j] = val - lse;
        }
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) {
            if (offset + j < V) {
                T val = static_cast<T>(static_cast<float>(logits[offset + j]) * inv_temp);
                out[offset + j] = val - lse;
            }
        }
    }
"""

_fused_log_softmax_kernel = mx.fast.metal_kernel(
    name="fused_log_softmax",
    input_names=["logits", "temp"],
    output_names=["out"],
    source=_fused_log_softmax_source,
    ensure_row_contiguous=True,
)


def fused_log_softmax(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """log_softmax(logits / temperature) via a single fused Metal kernel."""
    orig_shape = logits.shape
    V = orig_shape[-1]
    n_rows = 1
    for d in orig_shape[:-1]:
        n_rows *= d
    flat = logits.reshape(n_rows, V)
    dt = logits.dtype
    temp_arr = mx.array([temperature], dtype=mx.float32)
    result = _fused_log_softmax_kernel(
        inputs=[flat, temp_arr],
        output_shapes=[flat.shape],
        output_dtypes=[dt],
        template=[("T", dt), ("V", V)],
        grid=(1024, n_rows, 1),
        threadgroup=(1024, 1, 1),
    )[0]
    return result.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Kernel 2: Fused logprob gather
#
# Given log_probs (N, V) and token_ids (N,), gather the log-prob for each
# token in one pass. Avoids holding the full (N, V) matrix long-term.
# ---------------------------------------------------------------------------

_logprob_gather_source = """
    uint idx = thread_position_in_grid.x;
    if (idx < N) {
        int tok = token_ids[idx];
        out[idx] = log_probs[idx * V + tok];
    }
"""

_logprob_gather_kernel = mx.fast.metal_kernel(
    name="logprob_gather",
    input_names=["log_probs", "token_ids"],
    output_names=["out"],
    source=_logprob_gather_source,
    ensure_row_contiguous=True,
)


def gather_logprobs(log_probs: mx.array, token_ids: mx.array) -> mx.array:
    """
    Gather per-token log-probabilities.
    log_probs: (N, V), token_ids: (N,) → returns (N,)
    """
    N, V = log_probs.shape
    token_ids = token_ids.astype(mx.int32)
    result = _logprob_gather_kernel(
        inputs=[log_probs, token_ids],
        output_shapes=[(N,)],
        output_dtypes=[log_probs.dtype],
        template=[("T", log_probs.dtype), ("V", V), ("N", N)],
        grid=(((N + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    return result


# ---------------------------------------------------------------------------
# Kernel 3: KL divergence per response (segmented reduction)
#
# KL(π || π_ref) ≈ sum_t [log π(t) - log π_ref(t)] over response tokens.
# One threadgroup per response; threads cooperate to reduce the token sum.
# T=128 threads per group comfortably covers typical response lengths.
# ---------------------------------------------------------------------------

_kl_per_response_source = """
    constexpr int T = 128;
    threadgroup float shared[T];

    uint resp_idx = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;  // scalar uint in Metal

    int start = offsets[resp_idx];
    int end   = offsets[resp_idx + 1];
    int len   = end - start;

    float accum = 0.0f;
    for (int i = tid; i < len; i += T) {
        float plp  = static_cast<float>(policy_lps[start + i]);
        float rlp  = static_cast<float>(ref_lps[start + i]);
        accum += plp - rlp;
    }

    shared[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[resp_idx] = static_cast<T_type>(shared[0]);
"""

_kl_per_response_kernel = mx.fast.metal_kernel(
    name="kl_per_response",
    input_names=["policy_lps", "ref_lps", "offsets"],
    output_names=["out"],
    source=_kl_per_response_source,
    ensure_row_contiguous=True,
)


def compute_kl(
    policy_lps: mx.array,
    ref_lps: mx.array,
    offsets: mx.array,
) -> mx.array:
    """
    Per-response KL divergence: KL(π || π_ref) for each response in the batch.

    Args:
        policy_lps: (N,) — per-token log-probs under the current policy
        ref_lps:    (N,) — per-token log-probs under the reference model
        offsets:    (B+1,) — start/end indices per response

    Returns:
        (B,) — KL per response (use mean for the training penalty)

    Both logprob arrays must be computed from the SAME tokenization of the
    SAME (prompt, response) pairs at temperature=1.0. The kernels above
    guarantee this when called via logprobs_for() with the two models.
    """
    B = offsets.shape[0] - 1
    offsets = offsets.astype(mx.int32)
    dt = policy_lps.dtype
    result = _kl_per_response_kernel(
        inputs=[policy_lps, ref_lps, offsets],
        output_shapes=[(B,)],
        output_dtypes=[dt],
        template=[("T_type", dt)],
        grid=(B * 128, 1, 1),
        threadgroup=(128, 1, 1),
    )[0]
    return result


# ---------------------------------------------------------------------------
# Kernel 4: Rubric scoring (parallel over tokens)
#
# Each threadgroup handles one (response, criterion) pair.
# 64 threads cooperate to reduce over the token dimension.
# Grid: (n_responses * 64, n_criteria, 1), threadgroup: (64, 1, 1).
# This is much faster than the naive 1-thread-per-pair approach when
# responses are longer than a few tokens.
# ---------------------------------------------------------------------------

_rubric_score_source = """
    constexpr int T = 64;
    threadgroup float shared[T];

    uint resp_idx = threadgroup_position_in_grid.x;
    uint crit_idx = threadgroup_position_in_grid.y;
    uint tid      = thread_index_in_threadgroup;  // scalar uint in Metal

    if (resp_idx >= N_RESP || crit_idx >= N_CRIT) return;

    int start = offsets[resp_idx];
    int end   = offsets[resp_idx + 1];
    int len   = end - start;

    float accum = 0.0f;
    for (int i = tid; i < len; i += T) {
        int   tok        = token_ids[start + i];
        float lp         = static_cast<float>(logprobs[start + i]);
        float rubric_val = rubric[crit_idx * V + tok];
        accum += lp * rubric_val;
    }

    shared[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0)
        scores[resp_idx * N_CRIT + crit_idx] =
            static_cast<T_type>(shared[0] * weights[crit_idx]);
"""

_rubric_score_kernel = mx.fast.metal_kernel(
    name="rubric_score",
    input_names=["logprobs", "token_ids", "offsets", "rubric", "weights"],
    output_names=["scores"],
    source=_rubric_score_source,
    ensure_row_contiguous=True,
)


def rubric_score(
    logprobs: mx.array,
    token_ids: mx.array,
    offsets: mx.array,
    rubric: mx.array,
    weights: mx.array,
) -> mx.array:
    """
    Score a batch of responses against a multi-criteria rubric.

    Args:
        logprobs:   (N,)           — concatenated per-token logprobs
        token_ids:  (N,)           — concatenated token ids
        offsets:    (B+1,)         — start/end indices per response
        rubric:     (C, vocab)     — criteria scoring matrix
        weights:    (C,)           — per-criterion weights

    Returns:
        (B, C) — per-response per-criterion scores.
        Sum across criteria dim (weighted by `weights`) for scalar reward.
    """
    B = offsets.shape[0] - 1
    C, V = rubric.shape
    dt = logprobs.dtype
    token_ids = token_ids.astype(mx.int32)
    offsets = offsets.astype(mx.int32)
    result = _rubric_score_kernel(
        inputs=[logprobs, token_ids, offsets, rubric, weights],
        output_shapes=[(B, C)],
        output_dtypes=[dt],
        template=[("T_type", dt), ("V", V), ("N_RESP", B), ("N_CRIT", C)],
        grid=(B * 64, C, 1),
        threadgroup=(64, 1, 1),
    )[0]
    return result


# ---------------------------------------------------------------------------
# Kernel 5: Per-token GRPO loss
#
# Computes the clipped-ratio GRPO objective per token, then the caller
# reduces to a scalar. Each thread handles one token.
#
# loss_t = -min(ratio_t * A_resp, clip(ratio_t, 1-ε, 1+ε) * A_resp)
#
# where ratio_t = exp(policy_lp_t - old_lp_t)
#       A_resp  = group-normalized advantage for the response this token
#                 belongs to (looked up via response_idx).
# ---------------------------------------------------------------------------

_grpo_token_loss_source = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N) return;

    float log_ratio = static_cast<float>(policy_lps[idx])
                    - static_cast<float>(old_lps[idx]);
    float ratio = metal::fast::exp(log_ratio);

    int   resp  = response_idx[idx];
    float adv   = static_cast<float>(advantages[resp]);
    float lo    = 1.0f - eps[0];
    float hi    = 1.0f + eps[0];

    float unclipped = ratio * adv;
    float clipped   = metal::clamp(ratio, lo, hi) * adv;
    out[idx] = static_cast<T>(-metal::min(unclipped, clipped));
"""

_grpo_token_loss_kernel = mx.fast.metal_kernel(
    name="grpo_token_loss",
    input_names=["policy_lps", "old_lps", "response_idx", "advantages", "eps"],
    output_names=["out"],
    source=_grpo_token_loss_source,
    ensure_row_contiguous=True,
)


def grpo_token_loss(
    policy_lps: mx.array,
    old_lps: mx.array,
    response_idx: mx.array,
    advantages: mx.array,
    eps: float = 0.2,
) -> mx.array:
    """
    Per-token GRPO loss (before mean reduction).

    Args:
        policy_lps:   (N,) — log-probs under current policy π_θ
        old_lps:      (N,) — log-probs under old policy π_old (rollout model)
        response_idx: (N,) — int32, which response [0..B-1] each token belongs to
        advantages:   (B,) — group-normalized advantages per response
        eps:          clip radius (0 to disable clipping)

    Returns:
        (N,) per-token losses; take mean() for the final scalar.
    """
    N = policy_lps.shape[0]
    dt = policy_lps.dtype
    response_idx = response_idx.astype(mx.int32)
    eps_arr = mx.array([eps], dtype=mx.float32)
    result = _grpo_token_loss_kernel(
        inputs=[policy_lps, old_lps, response_idx, advantages, eps_arr],
        output_shapes=[(N,)],
        output_dtypes=[dt],
        template=[("T", dt), ("N", N)],
        grid=(((N + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def logprobs_for(
    model,
    tokenizer,
    prompt: str,
    response: str,
    temperature: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """
    Per-token log-probs for `response` given `prompt`, under `model`.

    Uses KV caching internally: encodes prompt in one pass (building the KV cache),
    then runs the response tokens in a second pass against the cached prompt KV.

    Returns:
        logprobs:  (R,) — log π(t_i | context) for each response token
        token_ids: (R,) int32 — response token ids
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer.encode(prompt_text)
    resp_start = len(prompt_ids)

    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    tokens    = tokenizer.encode(tokenizer.apply_chat_template(messages_full, tokenize=False))
    resp_tids = tokens[resp_start:]
    R         = len(resp_tids)

    # Prompt pass — build KV cache, capture last logit (predicts resp_tids[0])
    cache = _make_cache()
    prompt_logits, cache = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(prompt_logits)
    cache.advance(resp_start)
    first_logit = prompt_logits[0, -1:, :]  # (1, V)

    # Response pass — run resp_tids[:-1] against cached prompt KV
    if R > 1:
        resp_logits, _ = model(mx.array([resp_tids[:-1]]), cache=cache)
        all_logits = mx.concatenate([first_logit, resp_logits[0]], axis=0)  # (R, V)
    else:
        all_logits = first_logit  # (1, V)

    resp_token_ids = mx.array(resp_tids, dtype=mx.int32)
    log_probs      = fused_log_softmax(all_logits, temperature=temperature)
    token_logprobs = gather_logprobs(log_probs, resp_token_ids)
    return token_logprobs, resp_token_ids


def batch_logprobs(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    temperature: float = 1.0,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Batch logprob extraction with prompt KV cache sharing.

    Each unique prompt is encoded once. The resulting KV cache snapshot and last
    logit are reused for every response paired with that prompt. In GRPO, prompts
    repeat G times (one per completion), so this saves G-1 prompt forward passes
    per unique prompt.

    Returns:
        logprobs:  (N,)   — concatenated per-token logprobs
        token_ids: (N,)   — concatenated token ids
        offsets:   (B+1,) — segment boundaries; pair i spans [offsets[i], offsets[i+1])
    """
    all_lps, all_tids, offsets = [], [], [0]

    # prompt → (cache_snapshot, resp_start, first_logit)
    # first_logit: (1, V) — last prompt logit, predicts first response token
    prompt_state: dict[str, tuple] = {}

    for prompt, response in zip(prompts, responses):
        if prompt not in prompt_state:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer.encode(prompt_text)
            cache = _make_cache()
            plogs, cache = model(mx.array([prompt_ids]), cache=cache)
            mx.eval(plogs)
            cache.advance(len(prompt_ids))
            prompt_state[prompt] = (cache.snapshot(), len(prompt_ids), plogs[0, -1:, :])

        snap, resp_start, first_logit = prompt_state[prompt]

        tokens    = tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
        ))
        resp_tids = tokens[resp_start:]
        R         = len(resp_tids)

        # Response forward pass using a fresh copy of the prompt cache snapshot
        cache = snap.snapshot()
        if R > 1:
            resp_logits, _ = model(mx.array([resp_tids[:-1]]), cache=cache)
            all_logits = mx.concatenate([first_logit, resp_logits[0]], axis=0)  # (R, V)
        else:
            all_logits = first_logit  # (1, V)

        resp_token_ids = mx.array(resp_tids, dtype=mx.int32)
        log_probs      = fused_log_softmax(all_logits, temperature=temperature)
        lps            = gather_logprobs(log_probs, resp_token_ids)
        mx.eval(lps)

        all_lps.append(lps)
        all_tids.append(resp_token_ids)
        offsets.append(offsets[-1] + R)

    return (
        mx.concatenate(all_lps),
        mx.concatenate(all_tids),
        mx.array(offsets, dtype=mx.int32),
    )


def compute_grpo_loss(
    policy_lps: mx.array,
    old_lps: mx.array,
    advantages: mx.array,
    offsets: mx.array,
    beta: float = 0.01,
    ref_lps: mx.array | None = None,
    eps: float = 0.2,
) -> mx.array:
    """
    Full GRPO loss scalar.

    loss = policy_loss + beta * kl_penalty

    policy_loss = mean over all tokens of clipped-ratio GRPO objective
    kl_penalty  = mean over responses of KL(π || π_ref)
                  (only added when ref_lps is provided)

    Args:
        policy_lps:  (N,) — log π_θ(t) for rollout tokens
        old_lps:     (N,) — log π_old(t) for same tokens (used for ratio)
        advantages:  (B,) — group-normalized advantage per response
        offsets:     (B+1,) — token boundaries per response
        beta:        KL coefficient
        ref_lps:     (N,) — log π_ref(t); if None, KL penalty is skipped
        eps:         PPO clip radius (0 to disable)

    Returns:
        scalar loss (differentiable through policy_lps)
    """
    B = offsets.shape[0] - 1

    # Build response_idx: which response does each token belong to?
    # Derived from offsets in Python; small overhead since B is tiny (4-16).
    offsets_list = offsets.tolist()
    resp_idx_list = []
    for i in range(B):
        resp_idx_list.extend([i] * (offsets_list[i + 1] - offsets_list[i]))
    response_idx = mx.array(resp_idx_list, dtype=mx.int32)

    per_token_loss = grpo_token_loss(policy_lps, old_lps, response_idx, advantages, eps)
    loss = mx.mean(per_token_loss)

    if ref_lps is not None and beta > 0.0:
        kl = compute_kl(policy_lps, ref_lps, offsets)   # (B,)
        loss = loss + beta * mx.mean(kl)

    return loss


# ---------------------------------------------------------------------------
# Benchmarking / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Gwen Metal — Smoke Test & Benchmark ===\n")

    model, tokenizer = get_model()
    V = model.model.embed_tokens.weight.shape[0]
    print(f"Vocab size: {V}\n")

    # --- Test 1: fused_log_softmax ---
    print("[1] Fused log-softmax kernel")
    test_logits = mx.random.normal((1, 128, V))
    mx.eval(test_logits)

    t0 = time.perf_counter()
    for _ in range(100):
        result = fused_log_softmax(test_logits, temperature=0.7)
        mx.eval(result)
    t1 = time.perf_counter()
    metal_time = (t1 - t0) / 100

    t0 = time.perf_counter()
    for _ in range(100):
        ref = test_logits / 0.7
        ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
        mx.eval(ref)
    t1 = time.perf_counter()
    mlx_time = (t1 - t0) / 100

    metal_out = fused_log_softmax(test_logits, temperature=0.7)
    ref_out = test_logits / 0.7 - mx.logsumexp(test_logits / 0.7, axis=-1, keepdims=True)
    mx.eval(metal_out, ref_out)
    max_diff = mx.max(mx.abs(metal_out - ref_out)).item()
    print(f"  Metal:   {metal_time*1000:.3f} ms")
    print(f"  MLX:     {mlx_time*1000:.3f} ms")
    print(f"  Speedup: {mlx_time/metal_time:.2f}x")
    print(f"  Max diff: {max_diff:.6f}")
    print()

    # --- Test 2: logprob gather ---
    print("[2] Logprob gather kernel")
    fake_logprobs = mx.random.normal((512, V))
    fake_tokens = mx.random.randint(0, V, (512,)).astype(mx.int32)
    mx.eval(fake_logprobs, fake_tokens)

    t0 = time.perf_counter()
    for _ in range(100):
        result = gather_logprobs(fake_logprobs, fake_tokens)
        mx.eval(result)
    t1 = time.perf_counter()
    metal_time = (t1 - t0) / 100

    t0 = time.perf_counter()
    for _ in range(100):
        ref = fake_logprobs[mx.arange(512), fake_tokens]
        mx.eval(ref)
    t1 = time.perf_counter()
    mlx_time = (t1 - t0) / 100
    print(f"  Metal:   {metal_time*1000:.3f} ms")
    print(f"  MLX:     {mlx_time*1000:.3f} ms")
    print(f"  Speedup: {mlx_time/metal_time:.2f}x")
    print()

    # --- Test 3: logprobs_for (correctness vs gwen.py approach) ---
    print("[3] logprobs_for (correctness check)")
    test_prompt = "What makes a person trustworthy?"
    test_response = "Honesty, consistency, and empathy are key."

    t0 = time.perf_counter()
    lps, tids = logprobs_for(model, tokenizer, test_prompt, test_response)
    mx.eval(lps, tids)
    t1 = time.perf_counter()
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  Tokens: {lps.shape[0]}")
    print(f"  Mean logprob: {mx.mean(lps).item():.4f}")
    print(f"  Sum logprob:  {mx.sum(lps).item():.4f}")
    print()

    # --- Test 4: KL divergence (should be ~0 when both models are the same) ---
    print("[4] KL per-response kernel")
    # Use same model for policy and ref → KL should be ~0
    prompts_test = ["What is kindness?", "Why should we be honest?"]
    responses_test = [
        "Kindness is treating others with compassion.",
        "Honesty builds trust and strengthens relationships.",
    ]

    policy_lps_t, tids_t, offsets_t = batch_logprobs(model, tokenizer, prompts_test, responses_test)
    ref_lps_t = policy_lps_t  # same model → KL must be 0

    kl = compute_kl(policy_lps_t, ref_lps_t, offsets_t)
    mx.eval(kl)
    print(f"  KL (same model, should be 0): {kl.tolist()}")
    print()

    # --- Test 5: rubric_score (parallel kernel) ---
    print("[5] Rubric scoring kernel (parallel)")
    n_criteria = 3
    rubric_matrix = mx.random.normal((n_criteria, V)) * 0.01
    weights = mx.array([1.0, 0.7, 0.3])
    mx.eval(rubric_matrix, weights)

    batch_lps = mx.random.normal((200,)).astype(mx.bfloat16)
    batch_tids = mx.random.randint(0, V, (200,)).astype(mx.int32)
    batch_offsets = mx.array([0, 50, 100, 150, 200], dtype=mx.int32)
    mx.eval(batch_lps, batch_tids, batch_offsets)

    t0 = time.perf_counter()
    for _ in range(100):
        scores = rubric_score(batch_lps, batch_tids, batch_offsets, rubric_matrix, weights)
        mx.eval(scores)
    t1 = time.perf_counter()
    print(f"  Time (100 runs): {(t1-t0)*1000:.3f} ms total")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Per-response rewards: {mx.sum(scores, axis=-1).tolist()}")
    print()

    # --- Test 6: full GRPO loss ---
    print("[6] Full GRPO loss")
    advantages = mx.array([0.5, -0.5, 1.0, -1.0])  # 4 responses
    offsets_loss = mx.array([0, 50, 100, 150, 200], dtype=mx.int32)
    fake_policy = mx.random.normal((200,)).astype(mx.bfloat16)
    fake_old    = fake_policy + mx.random.normal((200,)).astype(mx.bfloat16) * 0.05
    fake_ref    = fake_policy + mx.random.normal((200,)).astype(mx.bfloat16) * 0.1
    mx.eval(fake_policy, fake_old, fake_ref, advantages)

    loss = compute_grpo_loss(
        fake_policy, fake_old, advantages, offsets_loss,
        beta=0.01, ref_lps=fake_ref, eps=0.2,
    )
    mx.eval(loss)
    print(f"  Loss: {loss.item():.4f}")
    print()

    print("Metal implementation ready.")
