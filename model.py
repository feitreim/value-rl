from __future__ import annotations
from typing import TYPE_CHECKING
import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from kvcache import KVCache

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, mx.ones(dim)
    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)

class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gu = self.gate_up_proj(x)
        gate, up = mx.split(gu, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)

def _rope_workaround(x, dims, theta, offset):
    # mx.fast.rope is buggy for S=1 with B>1: batch item b gets position offset+b instead of offset
    # Workaround: pass offset as a tensor of shape (B,) with identical values
    if x.shape[2] == 1 and x.shape[0] > 1:
        offsets = mx.full((x.shape[0],), offset, dtype=mx.int32)
        return mx.fast.rope(x, dims, traditional=False, base=theta, scale=1.0, offset=offsets)
    return mx.fast.rope(x, dims, traditional=False, base=theta, scale=1.0, offset=offset)

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, *,
                 use_qk_norm: bool, eps: float, rope_theta: float):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.scale = head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps) if use_qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=eps) if use_qk_norm else None
        self.rope_theta = rope_theta

    def __call__(self, x: mx.array, mask: str | mx.array, layer_idx: int,
                 cache: KVCache | None = None) -> tuple[mx.array, KVCache | None]:
        b, s, _ = x.shape
        offset = 0 if cache is None else cache.get_seq_len(layer_idx)

        # Fused projection for efficiency
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + k_size]
        v = qkv[..., q_size + k_size :]

        q = q.reshape(b, s, self.num_heads, self.head_dim)
        k = k.reshape(b, s, self.num_kv_heads, self.head_dim)
        v = v.reshape(b, s, self.num_kv_heads, self.head_dim)

        # QK norm before transpose — matches mlx_lm order
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = _rope_workaround(q, self.head_dim, self.rope_theta, offset)
        k = _rope_workaround(k, self.head_dim, self.rope_theta, offset)

        if cache is not None:
            k, v = cache.update(k, v, layer_idx)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, s, -1)), cache

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int,
                 eps: float, *, use_qk_norm: bool, rope_theta: float):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = Attention(dim, num_heads, num_kv_heads, head_dim, use_qk_norm=use_qk_norm,
                                   eps=eps, rope_theta=rope_theta)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = MLP(dim, intermediate_size)

    def __call__(self, x: mx.array, mask: str | mx.array, layer_idx: int,
                 cache: KVCache | None = None) -> tuple[mx.array, KVCache | None]:
        attn_out, cache = self.self_attn(self.input_layernorm(x), mask, layer_idx, cache)
        x = x + attn_out
        return x + self.mlp(self.post_attention_layernorm(x)), cache

class Qwen3(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int, max_seq_len: int, rope_theta: float = 1000000.0,
                 eps: float = 1e-6, tie_word_embeddings: bool = True, use_qk_norm: bool = True):
        super().__init__()
        self.num_layers, self.tie_word_embeddings = num_layers, tie_word_embeddings
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [DecoderLayer(dim, num_heads, num_kv_heads, head_dim, intermediate_size, eps,
                                    use_qk_norm=use_qk_norm, rope_theta=rope_theta) for _ in range(num_layers)]
        self.norm = RMSNorm(dim, eps=eps)
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, tokens: mx.array, cache: KVCache | None = None, mask: mx.array | None = None) -> tuple[mx.array, KVCache | None]:
        x = self.embed_tokens(tokens)
        if mask is None:
            mask = ("causal" if tokens.shape[1] > 1 else None)
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, mask, i, cache)
        x = self.norm(x)
        return (x @ self.embed_tokens.weight.T if self.tie_word_embeddings else self.lm_head(x)), cache
