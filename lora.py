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
    Replace q_proj and v_proj in all attention layers with LoRA adapters.

    1. Wraps each projection in LoRALinear (base weight hidden in _BaseLinear).
    2. Freezes the entire model (lora_a / lora_b included).
    3. Unfreezes only the LoRALinear modules → lora_a, lora_b become trainable.

    Returns the number of trainable LoRA parameters.
    """
    for layer in model.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear(attn.q_proj, rank, scale)
        attn.v_proj = LoRALinear(attn.v_proj, rank, scale)

    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze() if isinstance(m, LoRALinear) else None
    )

    return sum(
        p.size
        for layer in model.layers
        for proj in [layer.self_attn.q_proj, layer.self_attn.v_proj]
        if isinstance(proj, LoRALinear)
        for p in [proj.lora_a, proj.lora_b]
    )


def merge_lora(model) -> list:
    """
    Merge LoRA adapters into base weights for fast inference.

    Computes W_eff = W + (scale/rank) * B @ A for each LoRALinear and
    stores it as _base.w.  Returns a list of (proj, original_w) pairs for
    restoration via restore_lora().

    Why this matters:
      - Separate lora_a / lora_b matmuls are tiny (e.g. (G,1,8)) during
        token-by-token decode → many Metal kernel launches → very slow.
      - Merged W_eff is bfloat16, so all projections share the same dtype
        and mx.fast.scaled_dot_product_attention receives consistent inputs
        (mixed bfloat16/float32 causes GPU page faults).

    Returns empty list if no LoRA has been applied (no-op path).
    """
    saved = []
    for layer in model.layers:
        for proj_name in ("q_proj", "v_proj"):
            proj = getattr(layer.self_attn, proj_name)
            if isinstance(proj, LoRALinear):
                orig_w = proj._base.w
                saved.append((proj, orig_w))
                delta = proj._lora_scale * (proj.lora_b @ proj.lora_a)
                proj._base.w = orig_w + delta.astype(orig_w.dtype)
    if saved:
        mx.eval(*[proj._base.w for proj, _ in saved])
    return saved


def restore_lora(saved: list) -> None:
    """
    Restore original (unmerged) base weights after inference.

    Call this before the gradient step so gradients flow through lora_a
    and lora_b in the LoRALinear forward pass.  No-op if saved is empty.
    """
    for proj, orig_w in saved:
        proj._base.w = orig_w
