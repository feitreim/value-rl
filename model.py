from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from kvcache import KVCache


# ---------------------------------------------------------------------------
# Fused reshape + transpose + RMSNorm + RoPE Metal kernel
# ---------------------------------------------------------------------------
# Fuses four ops into a single GPU pass:
#   1. Reshape (B, S, NH*HD) -> (B, S, NH, HD)
#   2. Transpose -> (B, NH, S, HD)
#   3. Per-head RMSNorm
#   4. Non-traditional (NeoX-style) RoPE
# One threadgroup per (batch, head, seq_pos), HD threads per group.

_norm_rope_kernels: dict[tuple[int, float], Any] = {}


def _get_norm_rope_kernel(head_dim: int, eps: float = 1e-6, rope_base: float = 1000000.0) -> Any:
    cache_key = (head_dim, rope_base)
    if cache_key in _norm_rope_kernels:
        return _norm_rope_kernels[cache_key]

    n_simd = head_dim // 32
    half = head_dim // 2

    source = f"""
        threadgroup float partial_sums[{n_simd}];
        threadgroup float normed_vals[{head_dim}];

        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        uint S = seq_len[0];
        uint seq = gid % S;
        uint head = (gid / S) % NH;
        uint batch = gid / (S * NH);

        // Read from (B, S, NH*HD) contiguous layout
        uint in_idx = batch * (S * NH * {head_dim}) + seq * (NH * {head_dim}) + head * {head_dim} + tid;
        float val = float(inp[in_idx]);

        // RMSNorm: SIMD reduction + cross-group shared mem reduction
        float simd_sq = simd_sum(val * val);
        if (thread_index_in_simdgroup == 0) {{
            partial_sums[tid / threads_per_simdgroup] = simd_sq;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = 0.0f;
        for (uint i = 0; i < {n_simd}; i++) {{
            total_sq += partial_sums[i];
        }}
        float normed = val * metal::rsqrt(total_sq / {float(head_dim)}f + {eps}f) * float(norm_w[tid]);

        // Store normalised values for RoPE partner access
        normed_vals[tid] = normed;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RoPE (non-traditional / NeoX-style: pairs [i, i+d/2])
        uint pos = rope_offset[0] + seq;
        uint rope_dim = (tid < {half}) ? tid : (tid - {half});
        float freq = 1.0f / metal::pow({rope_base}f, float(2 * rope_dim) / {float(head_dim)}f);
        float angle = float(pos) * freq;
        float cos_val = metal::precise::cos(angle);
        float sin_val = metal::precise::sin(angle);

        float result;
        if (tid < {half}) {{
            result = normed_vals[tid] * cos_val - normed_vals[tid + {half}] * sin_val;
        }} else {{
            result = normed_vals[tid] * cos_val + normed_vals[tid - {half}] * sin_val;
        }}

        // Write to (B, NH, S, HD) transposed layout
        uint out_idx = batch * (NH * S * {head_dim}) + head * (S * {head_dim}) + seq * {head_dim} + tid;
        out[out_idx] = static_cast<T_type>(result);
    """

    kernel = mx.fast.metal_kernel(
        name="fused_norm_rope",
        input_names=["inp", "norm_w", "seq_len", "rope_offset"],
        output_names=["out"],
        source=source,
    )
    _norm_rope_kernels[cache_key] = kernel
    return kernel


def fused_norm_rope(
    proj_out: mx.array,
    norm_weight: mx.array,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
) -> mx.array:
    b, s, _ = proj_out.shape
    kernel = _get_norm_rope_kernel(head_dim, rope_base=rope_theta)
    return kernel(
        inputs=[proj_out, norm_weight, mx.array([s], dtype=mx.uint32), mx.array([offset], dtype=mx.uint32)],
        template=[("NH", num_heads), ("T_type", proj_out.dtype)],
        grid=(b * num_heads * s * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(b, num_heads, s, head_dim)],
        output_dtypes=[proj_out.dtype],
        stream=mx.gpu,
    )[0]


# ---------------------------------------------------------------------------
# Fused reshape + transpose + RoPE Metal kernel (no norm)
# ---------------------------------------------------------------------------
# Fuses three ops into a single GPU pass:
#   1. Reshape (B, S, NH*HD) -> (B, S, NH, HD)
#   2. Transpose -> (B, NH, S, HD)
#   3. RoPE (non-interleaved Llama-style: pairs [i, i+d/2])
# One threadgroup per (batch, head, seq_pos), HD threads per group.

_rope_kernels: dict[tuple[int, float], Any] = {}


def _get_rope_kernel(head_dim: int, rope_base: float = 1000000.0) -> Any:
    cache_key = (head_dim, rope_base)
    if cache_key in _rope_kernels:
        return _rope_kernels[cache_key]

    half = head_dim // 2

    source = f"""
        threadgroup float vals[{head_dim}];

        uint tid = thread_position_in_threadgroup.x;
        uint gid = threadgroup_position_in_grid.x;

        uint S = seq_len[0];
        uint seq = gid % S;
        uint head = (gid / S) % NH;
        uint batch = gid / (S * NH);

        // Read from (B, S, NH*HD) contiguous layout
        uint in_idx = batch * (S * NH * {head_dim}) + seq * (NH * {head_dim}) + head * {head_dim} + tid;
        float val = float(inp[in_idx]);

        // Store values for RoPE partner access
        vals[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RoPE (non-interleaved Llama-style: pairs [i, i+d/2])
        uint pos = rope_offset[0] + seq;
        uint rope_dim = (tid < {half}) ? tid : (tid - {half});
        float freq = 1.0f / metal::pow({rope_base}f, float(2 * rope_dim) / {float(head_dim)}f);
        float angle = float(pos) * freq;
        float cos_val = metal::precise::cos(angle);
        float sin_val = metal::precise::sin(angle);

        float result;
        if (tid < {half}) {{
            result = vals[tid] * cos_val - vals[tid + {half}] * sin_val;
        }} else {{
            result = vals[tid] * cos_val + vals[tid - {half}] * sin_val;
        }}

        // Write to (B, NH, S, HD) transposed layout
        uint out_idx = batch * (NH * S * {head_dim}) + head * (S * {head_dim}) + seq * {head_dim} + tid;
        out[out_idx] = static_cast<T_type>(result);
    """

    kernel = mx.fast.metal_kernel(
        name="fused_rope",
        input_names=["inp", "seq_len", "rope_offset"],
        output_names=["out"],
        source=source,
    )
    _rope_kernels[cache_key] = kernel
    return kernel


def fused_rope(
    proj_out: mx.array,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
) -> mx.array:
    b, s, _ = proj_out.shape
    kernel = _get_rope_kernel(head_dim, rope_base=rope_theta)
    return kernel(
        inputs=[proj_out, mx.array([s], dtype=mx.uint32), mx.array([offset], dtype=mx.uint32)],
        template=[("NH", num_heads), ("T_type", proj_out.dtype)],
        grid=(b * num_heads * s * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(b, num_heads, s, head_dim)],
        output_dtypes=[proj_out.dtype],
        stream=mx.gpu,
    )[0]


# ---------------------------------------------------------------------------
# Baseline (separate MLX ops) for benchmarking
# ---------------------------------------------------------------------------


def baseline_norm_rope(
    proj_out: mx.array,
    norm_weight: mx.array,
    eps: float,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
    rope_traditional: bool,
) -> mx.array:
    b, s, _ = proj_out.shape
    x = proj_out.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
    x = mx.fast.rms_norm(x, norm_weight, eps)
    return mx.fast.rope(x, head_dim, traditional=rope_traditional, base=rope_theta, scale=1.0, offset=offset)


def baseline_rope(
    proj_out: mx.array,
    num_heads: int,
    head_dim: int,
    offset: int,
    rope_theta: float,
    rope_traditional: bool,
) -> mx.array:
    b, s, _ = proj_out.shape
    x = proj_out.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
    return mx.fast.rope(x, head_dim, traditional=rope_traditional, base=rope_theta, scale=1.0, offset=offset)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        use_qk_norm: bool,
        eps: float,
        rope_theta: float,
        rope_traditional: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.eps = eps
        self.rope_theta = rope_theta
        self.rope_traditional = rope_traditional
        self.n_rep = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=eps)
            self.k_norm = RMSNorm(head_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(
        self,
        x: mx.array,
        mask: str | mx.array,
        layer_idx: int,
        cache: KVCache | None = None,
        use_metal: bool = True,
    ) -> tuple[mx.array, KVCache | None]:
        b, s, _d = x.shape

        offset = 0 if cache is None else cache.get_seq_len(layer_idx)
        if self.use_qk_norm:
            if use_metal:
                q = fused_norm_rope(self.q_proj(x), self.q_norm.weight, self.num_heads, self.head_dim, offset, self.rope_theta)
                k = fused_norm_rope(self.k_proj(x), self.k_norm.weight, self.num_kv_heads, self.head_dim, offset, self.rope_theta)
            else:
                q = baseline_norm_rope(self.q_proj(x), self.q_norm.weight, self.eps, self.num_heads, self.head_dim, offset, self.rope_theta, self.rope_traditional)
                k = baseline_norm_rope(self.k_proj(x), self.k_norm.weight, self.eps, self.num_kv_heads, self.head_dim, offset, self.rope_theta, self.rope_traditional)
        else:
            q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, self.head_dim, traditional=self.rope_traditional, base=self.rope_theta, scale=1.0, offset=offset)
            k = mx.fast.rope(k, self.head_dim, traditional=self.rope_traditional, base=self.rope_theta, scale=1.0, offset=offset)
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update(k, v, layer_idx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o_proj(out), cache


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float,
        *,
        use_qk_norm: bool,
        rope_theta: float,
        rope_traditional: bool,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = Attention(
            dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_qk_norm=use_qk_norm,
            eps=eps,
            rope_theta=rope_theta,
            rope_traditional=rope_traditional,
        )
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim, intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: str | mx.array,
        layer_idx: int,
        cache: KVCache | None = None,
        use_metal: bool = True,
    ) -> tuple[mx.array, KVCache | None]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), mask, layer_idx, cache, use_metal=use_metal)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


class Qwen3(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int,
        rope_theta: float = 1000000.0,
        eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        use_qk_norm: bool = True,
        rope_traditional: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.use_qk_norm = use_qk_norm
        self.rope_traditional = rope_traditional

        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [
            DecoderLayer(
                dim,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                eps,
                use_qk_norm=use_qk_norm,
                rope_theta=rope_theta,
                rope_traditional=rope_traditional,
            )
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(dim, eps=eps)
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(
        self,
        tokens: mx.array,
        cache: KVCache | None = None,
        use_metal: bool = True,
    ) -> tuple[mx.array, KVCache | None]:
        _b, s = tokens.shape
        x = self.embed_tokens(tokens)

        # Fast kernels use "causal" mask string for automatic causal masking
        mask = "causal"

        for i, layer in enumerate(self.layers):
            x, cache = layer(x, mask, i, cache, use_metal=use_metal)

        x = self.norm(x)

        logits = x @ self.embed_tokens.weight.T if self.tie_word_embeddings else self.lm_head(x)

        return logits, cache
