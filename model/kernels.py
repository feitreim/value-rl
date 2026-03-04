import mlx.core as mx

# ---------------------------------------------------------------------------
# Metal Kernels
# ---------------------------------------------------------------------------

_fused_log_softmax_source = """
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;
    threadgroup float shared[32];
    uint row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    logits += row * V; out += row * V;
    float inv_temp = 1.0f / temp[0];
    float thread_max = -1e30f;
    int offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) thread_max = max(thread_max, static_cast<float>(logits[offset+j]) * inv_temp);
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) if (offset+j < V) thread_max = max(thread_max, static_cast<float>(logits[offset+j]) * inv_temp);
    }
    float simd_max_val = simd_max(thread_max);
    if (simd_lane_id == 0) shared[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) { float v = shared[simd_lane_id]; v = simd_max(v); if (simd_lane_id == 0) shared[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared[0];
    float sum_exp = 0.0f; offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) sum_exp += metal::fast::exp(static_cast<float>(logits[offset+j]) * inv_temp - row_max);
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) if (offset+j < V) sum_exp += metal::fast::exp(static_cast<float>(logits[offset+j]) * inv_temp - row_max);
    }
    sum_exp = simd_sum(sum_exp);
    if (simd_lane_id == 0) shared[simd_group_id] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) { float v = shared[simd_lane_id]; v = simd_sum(v); if (simd_lane_id == 0) shared[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float log_sum_exp = metal::fast::log(shared[0]);
    T lse = static_cast<T>(row_max + log_sum_exp);
    offset = tid * M;
    for (int i = 0; i < full_blocks; i++) {
        for (int j = 0; j < M; j++) out[offset+j] = static_cast<T>(static_cast<float>(logits[offset+j]) * inv_temp) - lse;
        offset += block;
    }
    if (extra > 0) {
        for (int j = 0; j < M; j++) if (offset+j < V) out[offset+j] = static_cast<T>(static_cast<float>(logits[offset+j]) * inv_temp) - lse;
    }
"""

_fused_log_softmax_kernel = mx.fast.metal_kernel(
    name="fused_log_softmax", input_names=["logits", "temp"], output_names=["out"],
    source=_fused_log_softmax_source, ensure_row_contiguous=True,
)

def fused_log_softmax(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """Fast non-differentiable log-softmax kernel for rollouts."""
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    res = _fused_log_softmax_kernel(
        inputs=[flat, mx.array([temperature], dtype=mx.float32)],
        output_shapes=[flat.shape], output_dtypes=[logits.dtype],
        template=[("T", logits.dtype), ("V", V)],
        grid=(1024, flat.shape[0], 1), threadgroup=(1024, 1, 1),
    )[0]
    return res.reshape(logits.shape)
