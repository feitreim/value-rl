from __future__ import annotations
from typing import TYPE_CHECKING
import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from kvcache import KVCache

def fused_norm_rope(proj_out, norm_weight, num_heads, head_dim, offset, rope_theta, eps=1e-6):
    b, s, _ = proj_out.shape
    x = proj_out.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
    x = mx.fast.rms_norm(x, norm_weight, eps)
    return mx.fast.rope(x, head_dim, traditional=False, base=rope_theta, scale=1.0, offset=offset)

def fused_rope(proj_out, num_heads, head_dim, offset, rope_theta):
    b, s, _ = proj_out.shape
    x = proj_out.reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
    return mx.fast.rope(x, head_dim, traditional=False, base=rope_theta, scale=1.0, offset=offset)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, mx.ones(dim)
    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj, self.down_proj = nn.Linear(dim, 2 * intermediate_size, bias=False), \
                                            nn.Linear(intermediate_size, dim, bias=False)
    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(nn.silu(gate_up[..., :gate_up.shape[-1]//2]) * gate_up[..., gate_up.shape[-1]//2:])

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, *,
                 use_qk_norm: bool, eps: float, rope_theta: float, rope_traditional: bool):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim, self.use_qk_norm, self.eps, \
            self.rope_theta, self.rope_traditional = num_heads, num_kv_heads, head_dim, \
            use_qk_norm, eps, rope_theta, rope_traditional
        self.scale, self.o_proj = head_dim**-0.5, nn.Linear(num_heads * head_dim, dim, bias=False)
        self.qkv_proj = nn.Linear(dim, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps) if use_qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=eps) if use_qk_norm else None
    def __call__(self, x: mx.array, mask: str | mx.array, layer_idx: int,
                 cache: KVCache | None = None) -> tuple[mx.array, KVCache | None]:
        b, s, _ = x.shape
        offset = 0 if cache is None else cache.get_seq_len(layer_idx)
        qkv = self.qkv_proj(x)
        q_end, k_end = self.num_heads * self.head_dim, (self.num_heads + self.num_kv_heads) * self.head_dim
        q_raw, k_raw, v_raw = qkv[..., :q_end], qkv[..., q_end:k_end], qkv[..., k_end:]
        if self.use_qk_norm:
            q = fused_norm_rope(q_raw, self.q_norm.weight, self.num_heads, self.head_dim, offset, self.rope_theta, self.eps)
            k = fused_norm_rope(k_raw, self.k_norm.weight, self.num_kv_heads, self.head_dim, offset, self.rope_theta, self.eps)
        else:
            q, k = fused_rope(q_raw, self.num_heads, self.head_dim, offset, self.rope_theta), \
                   fused_rope(k_raw, self.num_kv_heads, self.head_dim, offset, self.rope_theta)
        v = v_raw.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None: k, v = cache.update(k, v, layer_idx)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, s, -1)), cache

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int,
                 eps: float, *, use_qk_norm: bool, rope_theta: float, rope_traditional: bool):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = Attention(dim, num_heads, num_kv_heads, head_dim, use_qk_norm=use_qk_norm,
                                   eps=eps, rope_theta=rope_theta, rope_traditional=rope_traditional)
        self.post_attention_layernorm, self.mlp = RMSNorm(dim, eps=eps), SwiGLU(dim, intermediate_size)
    def __call__(self, x: mx.array, mask: str | mx.array, layer_idx: int,
                 cache: KVCache | None = None) -> tuple[mx.array, KVCache | None]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), mask, layer_idx, cache)
        x = x + attn_out
        return x + self.mlp(self.post_attention_layernorm(x)), cache

class Qwen3(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int, max_seq_len: int, rope_theta: float = 1000000.0,
                 eps: float = 1e-6, tie_word_embeddings: bool = True, use_qk_norm: bool = True,
                 rope_traditional: bool = False):
        super().__init__()
        self.num_layers, self.tie_word_embeddings = num_layers, tie_word_embeddings
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [DecoderLayer(dim, num_heads, num_kv_heads, head_dim, intermediate_size, eps,
                                    use_qk_norm=use_qk_norm, rope_theta=rope_theta,
                                    rope_traditional=rope_traditional) for _ in range(num_layers)]
        self.norm = RMSNorm(dim, eps=eps)
        if not tie_word_embeddings: self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    def __call__(self, tokens: mx.array, cache: KVCache | None = None, mask: mx.array | None = None) -> tuple[mx.array, KVCache | None]:
        x = self.embed_tokens(tokens)
        if mask is None: mask = ("causal" if tokens.shape[1] > 1 else None)
        for i, layer in enumerate(self.layers): x, cache = layer(x, mask, i, cache)
        x = self.norm(x)
        return (x @ self.embed_tokens.weight.T if self.tie_word_embeddings else self.lm_head(x)), cache
