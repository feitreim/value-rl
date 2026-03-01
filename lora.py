"""
lora.py — LoRA adapters for Qwen3 attention projections.

Replaces q_proj and v_proj in every attention layer with LoRALinear.
The base weight is stored in _BaseLinear (a plain Python class, invisible
to MLX parameter traversal), so only lora_a and lora_b are trainable.

After apply_lora():
  - model.freeze() marks all visible parameters frozen.
  - LoRALinear.unfreeze() makes lora_a / lora_b trainable again.
  - model.trainable_parameters() == {lora_a, lora_b} for each wrapped proj.
  - The backward pass computes d(lora_a) and d(lora_b) but skips d(W).
"""

import math

import mlx.core as mx
import mlx.nn as nn


class _BaseLinear:
    """
    Frozen weight wrapper invisible to MLX parameter traversal.

    MLX traverses nn.Module submodules, mx.array attributes, and dicts/lists
    thereof — but not arbitrary Python objects.  Storing the weight here keeps
    it out of parameters(), trainable_parameters(), and freeze/unfreeze.
    """

    __slots__ = ("w",)

    def __init__(self, w: mx.array):
        self.w = w

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.w.T


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank adapter.

    forward: y = x @ W.T  +  scale * (x @ A.T) @ B.T
    where W is frozen (via _BaseLinear), A and B are the only trainable params.

    lora_b = 0 at init → delta = 0 → exact match to base model at step 0.
    """

    def __init__(self, linear: nn.Linear, rank: int, scale: float = 20.0):
        super().__init__()
        out_dim, in_dim = linear.weight.shape
        dtype = linear.weight.dtype               # match base weight (bfloat16)
        self._base = _BaseLinear(linear.weight)   # hidden from MLX param tree
        self.lora_a = (mx.random.normal([rank, in_dim]) * (1 / math.sqrt(rank))).astype(dtype)
        self.lora_b = mx.zeros([out_dim, rank], dtype=dtype)
        self._lora_scale = scale / rank           # plain float, not a param

    def __call__(self, x: mx.array) -> mx.array:
        return self._base(x) + self._lora_scale * ((x @ self.lora_a.T) @ self.lora_b.T)


def apply_lora(model, rank: int = 8, scale: float = 20.0) -> int:
    """
    Replace qkv_proj, o_proj, and gate_up_proj in all layers with LoRA adapters.

    1. Wraps each projection in LoRALinear (base weight hidden in _BaseLinear).
    2. Freezes the entire model (lora_a / lora_b included).
    3. Unfreezes only the LoRALinear modules → lora_a, lora_b become trainable.

    Returns the number of trainable LoRA parameters.
    """
    for layer in model.layers:
        # Attention
        attn = layer.self_attn
        attn.qkv_proj = LoRALinear(attn.qkv_proj, rank, scale)
        attn.o_proj = LoRALinear(attn.o_proj, rank, scale)
        
        # MLP
        mlp = layer.mlp
        mlp.gate_up_proj = LoRALinear(mlp.gate_up_proj, rank, scale)
        mlp.down_proj = LoRALinear(mlp.down_proj, rank, scale)

    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze() if isinstance(m, LoRALinear) else None
    )

    n_params = 0
    for layer in model.layers:
        projs = [
            layer.self_attn.qkv_proj, 
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj
        ]
        for proj in projs:
            if isinstance(proj, LoRALinear):
                n_params += proj.lora_a.size + proj.lora_b.size
    
    return n_params


def merge_lora(model) -> list:
    """
    Merge LoRA adapters into base weights for fast inference.

    Computes W_eff = W + (scale/rank) * B @ A for each LoRALinear and
    stores it as _base.w.  Returns a list of (proj, original_w) pairs for
    restoration via restore_lora().
    """
    saved = []
    
    def _merge(_k, m):
        if isinstance(m, LoRALinear):
            orig_w = m._base.w
            saved.append((m, orig_w))
            delta = m._lora_scale * (m.lora_b @ m.lora_a)
            m._base.w = orig_w + delta.astype(orig_w.dtype)
            
    model.apply_to_modules(_merge)
    
    if saved:
        mx.eval(*[m._base.w for m, _ in saved])
    return saved


def restore_lora(saved: list) -> None:
    """
    Restore original (unmerged) base weights after inference.

    Call this before the gradient step so gradients flow through lora_a
    and lora_b in the LoRALinear forward pass.  No-op if saved is empty.
    """
    for proj, orig_w in saved:
        proj._base.w = orig_w
