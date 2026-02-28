"""
test_correctness.py — Assert numerical agreement between Metal-accelerated
                      kernels (no VJP) and pure-MLX equivalents (VJP-enabled).

Key invariant for training correctness:
  KL(metal_kernel, same_inputs) == KL(mlx_ops, same_inputs)

The two logprob extraction paths differ slightly due to float32 vs bfloat16
accumulation in log-softmax (Metal kernel accumulates in fp32, which is more
precise). That difference is documented but not a bug.

What MUST agree:
  - compute_kl (Metal kernel) == MLX formula, given the SAME lp arrays
  - _grpo_loss scalar == metal compute_grpo_loss scalar, given same inputs

Run with:
    uv run test_correctness.py
"""

import mlx.core as mx

from gwen import WEIGHT_DTYPE, get_model
from gwen_metal import batch_logprobs, compute_kl, compute_grpo_loss
from grpo import _grpo_loss, _logprobs_flat

PROMPTS = [
    "What is intellectual curiosity?",
    "Why is honesty important in science?",
]
RESPONSES = [
    "It means genuinely engaging with ideas rather than just memorizing facts.",
    "Because scientific progress depends on accurate, unbiased reporting of results.",
]

if WEIGHT_DTYPE == mx.float16:
    # fp16 accumulation/rounding differences are larger than bf16 in this path.
    ATOL_KL = 5e-4
    ATOL_LOSS = 1e-3
else:
    # Metal kernel and MLX formula agree exactly on the same bf16 inputs.
    ATOL_KL = 1e-5
    ATOL_LOSS = 1e-4


def check(name, a, b, atol):
    a_list = a.tolist() if hasattr(a, "tolist") else [float(a)]
    b_list = b.tolist() if hasattr(b, "tolist") else [float(b)]
    if isinstance(a_list, float):
        a_list, b_list = [a_list], [b_list]
    max_diff = max(abs(x - y) for x, y in zip(a_list, b_list))
    status = "PASS" if max_diff <= atol else "FAIL"
    print(f"  [{status}] {name}: max_abs_diff = {max_diff:.2e}  (atol={atol:.0e})")
    assert max_diff <= atol, f"{name}: max diff {max_diff:.2e} > atol {atol:.0e}"


print("Loading model...")
model, tokenizer = get_model()
print("Ready.\n")


# ---------------------------------------------------------------------------
# Build shared logprob arrays via each path.
# ---------------------------------------------------------------------------

metal_lps, _, metal_offsets = batch_logprobs(model, tokenizer, PROMPTS, RESPONSES)
mlx_lps, mlx_offsets = _logprobs_flat(model, tokenizer, PROMPTS, RESPONSES)
mx.eval(metal_lps, metal_offsets, mlx_lps, mlx_offsets)

off_metal = metal_offsets.tolist()
off_mlx   = mlx_offsets.tolist()
assert off_metal == off_mlx, f"offsets mismatch: {off_metal} vs {off_mlx}"
B = len(off_metal) - 1


# ---------------------------------------------------------------------------
# Note: logprob values differ between Metal and MLX paths due to precision.
# Metal: log-softmax computed in float32 (more precise)
# MLX:   subtraction done in bfloat16 → high-prob tokens round to exactly 0.0
# This is expected. It does NOT affect KL correctness when both paths use the
# same inputs — that's what the tests below verify.
# ---------------------------------------------------------------------------

metal_diff = mx.max(mx.abs(metal_lps - mlx_lps)).item()
print(f"  [INFO] Metal vs MLX logprob max diff: {metal_diff:.4f}")
if WEIGHT_DTYPE == mx.bfloat16:
    print("         (Should be 0.0 — both paths subtract in bfloat16)")
else:
    print("         (Small non-zero diff is expected in float16 mode)")
print()


# ---------------------------------------------------------------------------
# 1. KL(same model) == 0: both implementations
#
# KL(π || π) must be identically zero. Tests both Metal kernel and MLX formula
# with logprobs from each extraction path.
# ---------------------------------------------------------------------------

print("=== 1. KL(π || π) == 0: Metal kernel and MLX formula ===")

# 1a. Metal lps → Metal kernel
kl_mm = compute_kl(metal_lps, metal_lps, metal_offsets)
mx.eval(kl_mm)
print(f"  Metal lps + Metal kernel:  {kl_mm.tolist()}")
check("KL(π||π) Metal+Metal == 0", kl_mm, mx.zeros_like(kl_mm), atol=1e-5)

# 1b. Metal lps → MLX formula
kl_mx_from_metal = mx.stack([
    mx.sum(metal_lps[off_metal[i]:off_metal[i+1]] - metal_lps[off_metal[i]:off_metal[i+1]])
    for i in range(B)
])
mx.eval(kl_mx_from_metal)
print(f"  Metal lps + MLX formula:   {kl_mx_from_metal.tolist()}")
check("KL(π||π) Metal+MLX == 0", kl_mx_from_metal, mx.zeros_like(kl_mx_from_metal), atol=1e-5)

# 1c. MLX lps → Metal kernel
kl_mlx_metal = compute_kl(mlx_lps, mlx_lps, mlx_offsets)
mx.eval(kl_mlx_metal)
print(f"  MLX lps + Metal kernel:    {kl_mlx_metal.tolist()}")
check("KL(π||π) MLX+Metal == 0", kl_mlx_metal, mx.zeros_like(kl_mlx_metal), atol=1e-5)

# 1d. MLX lps → MLX formula
kl_mm_mlx = mx.stack([
    mx.sum(mlx_lps[off_mlx[i]:off_mlx[i+1]] - mlx_lps[off_mlx[i]:off_mlx[i+1]])
    for i in range(B)
])
mx.eval(kl_mm_mlx)
print(f"  MLX  lps + MLX formula:    {kl_mm_mlx.tolist()}")
check("KL(π||π) MLX+MLX == 0", kl_mm_mlx, mx.zeros_like(kl_mm_mlx), atol=1e-5)

print()


# ---------------------------------------------------------------------------
# 2. KL agreement: Metal kernel == MLX formula, given the SAME input arrays
#
# This is the critical invariant: the two formulas (Metal vs MLX) must agree
# so that compute_kl (used for logging) and _grpo_loss (used for gradients)
# produce the same KL penalty value.
# ---------------------------------------------------------------------------

print("=== 2. KL agreement: Metal kernel == MLX formula (non-zero KL) ===")

mx.random.seed(42)
noise = mx.random.normal(metal_lps.shape, dtype=metal_lps.dtype) * 0.1
ref_lps = metal_lps + noise
mx.eval(ref_lps)

# Metal kernel
kl_metal = compute_kl(metal_lps, ref_lps, metal_offsets)

# MLX formula (same as in _grpo_loss)
kl_mlx = mx.stack([
    mx.sum(metal_lps[off_metal[i]:off_metal[i+1]] - ref_lps[off_metal[i]:off_metal[i+1]])
    for i in range(B)
])
mx.eval(kl_metal, kl_mlx)

print(f"  Metal kernel KL: {[f'{x:.5f}' for x in kl_metal.tolist()]}")
print(f"  MLX formula  KL: {[f'{x:.5f}' for x in kl_mlx.tolist()]}")
check("KL Metal == MLX (non-zero)", kl_metal, kl_mlx, atol=ATOL_KL)

# Repeat with MLX lps as the policy
noise_mlx = mx.random.normal(mlx_lps.shape, dtype=mlx_lps.dtype) * 0.1
ref_lps_mlx = mlx_lps + noise_mlx
mx.eval(ref_lps_mlx)

kl_metal_mlx_base = compute_kl(mlx_lps, ref_lps_mlx, mlx_offsets)
kl_mlx_mlx_base = mx.stack([
    mx.sum(mlx_lps[off_mlx[i]:off_mlx[i+1]] - ref_lps_mlx[off_mlx[i]:off_mlx[i+1]])
    for i in range(B)
])
mx.eval(kl_metal_mlx_base, kl_mlx_mlx_base)
check("KL Metal == MLX (MLX-base logprobs)", kl_metal_mlx_base, kl_mlx_mlx_base, atol=ATOL_KL)

print()


# ---------------------------------------------------------------------------
# 3. Full GRPO loss scalar: compute_grpo_loss (Metal) == _grpo_loss (MLX)
#    given the same policy/old/ref logprob arrays.
#
# In training: policy_lps from _logprobs_flat (MLX), old/ref from batch_logprobs
# (Metal). The loss formula must be equivalent.
# ---------------------------------------------------------------------------

print("=== 3. Full GRPO loss scalar: compute_grpo_loss (Metal) == _grpo_loss (MLX) ===")

advantages = mx.array([0.5, -0.5], dtype=mx.float32)
beta = 0.01
eps = 0.2

# Use metal_lps for all inputs — testing formula equivalence, not data path
loss_metal = compute_grpo_loss(
    metal_lps, metal_lps, advantages, metal_offsets,
    beta=beta, ref_lps=ref_lps, eps=eps,
)
loss_mlx = _grpo_loss(
    metal_lps, metal_lps, advantages, metal_offsets,
    beta=beta, ref_lps=ref_lps, eps=eps,
)
mx.eval(loss_metal, loss_mlx)
print(f"  compute_grpo_loss: {loss_metal.item():.6f}")
print(f"  _grpo_loss:        {loss_mlx.item():.6f}")
check("loss Metal == MLX (metal logprobs)", loss_metal, loss_mlx, atol=ATOL_LOSS)

# Also test with MLX lps
loss_metal2 = compute_grpo_loss(
    mlx_lps, mlx_lps, advantages, mlx_offsets,
    beta=beta, ref_lps=ref_lps_mlx, eps=eps,
)
loss_mlx2 = _grpo_loss(
    mlx_lps, mlx_lps, advantages, mlx_offsets,
    beta=beta, ref_lps=ref_lps_mlx, eps=eps,
)
mx.eval(loss_metal2, loss_mlx2)
check("loss Metal == MLX (MLX logprobs)", loss_metal2, loss_mlx2, atol=ATOL_LOSS)

print()
print("All correctness checks passed.")
