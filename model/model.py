from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

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

    def __call__(self, x: mx.array, mask: str | mx.array, layer_idx: int) -> mx.array:
        b, s, _ = x.shape
        offset = 0 # No KV cache during training

        # Fused projection for efficiency
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + k_size]
        v = qkv[..., q_size + k_size :]

        q = q.reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = _rope_workaround(q, self.head_dim, self.rope_theta, offset)
        k = _rope_workaround(k, self.head_dim, self.rope_theta, offset)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o_proj(out)

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int,
                 eps: float, *, use_qk_norm: bool, rope_theta: float):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = Attention(dim, num_heads, num_kv_heads, head_dim, use_qk_norm=use_qk_norm,
                                   eps=eps, rope_theta=rope_theta)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = MLP(dim, intermediate_size)

    def __call__(self, x: mx.array, mask: str | mx.array, layer_idx: int) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

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

    def __call__(self, tokens: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.embed_tokens(tokens)
        if mask is None:
            mask = ("causal" if tokens.shape[1] > 1 else None)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, i)
        x = self.norm(x)
        return (x @ self.embed_tokens.weight.T if self.tie_word_embeddings else self.lm_head(x))
