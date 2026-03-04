"""
gwen.py — Qwen3-0.6B wrapper.
"""

import os
import mlx.core as mx
from transformers import AutoTokenizer

from .load_weights import CHECKPOINT_PATH, load_qwen3_weights
from .model import Qwen3

# Architecture (Qwen3-0.6B)
VOCAB_SIZE        = 151936
DIM               = 1024
NUM_LAYERS        = 28
NUM_HEADS         = 16
NUM_KV_HEADS      = 8
HEAD_DIM          = 128
INTERMEDIATE_SIZE = 3072
MAX_SEQ_LEN       = 40960
ROPE_THETA        = 1_000_000.0
EPS               = 1e-6

_weight_dtype_name = os.getenv("GWEN_DTYPE", "bf16").strip().lower()
if _weight_dtype_name in {"fp16", "float16", "f16"}:
    WEIGHT_DTYPE = mx.float16
elif _weight_dtype_name in {"bf16", "bfloat16"}:
    WEIGHT_DTYPE = mx.bfloat16
else:
    raise ValueError(f"Unsupported GWEN_DTYPE={_weight_dtype_name!r}")

_model     = None
_tokenizer = None


def get_model() -> tuple[Qwen3, AutoTokenizer]:
    global _model, _tokenizer
    if _model is None:
        _model = Qwen3(
            vocab_size=VOCAB_SIZE, dim=DIM, num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            intermediate_size=INTERMEDIATE_SIZE, max_seq_len=MAX_SEQ_LEN,
            rope_theta=ROPE_THETA, eps=EPS,
        )
        load_qwen3_weights(_model, dtype=WEIGHT_DTYPE)
        mx.eval(_model.parameters())
        _tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_PATH))
    return _model, _tokenizer


from .kernels import fused_log_softmax

# ---------------------------------------------------------------------------
# Differentiable Logprob Extraction
# ---------------------------------------------------------------------------

@mx.compile
def _compute_token_logprobs(logits: mx.array, ids: mx.array, temperature: float) -> mx.array:
    """Memory-efficient logprob calculation: avoids keeping full log-softmax tensor."""
    logits = (logits.astype(mx.float32) / temperature)
    log_z = mx.logsumexp(logits, axis=-1, keepdims=True)
    token_logits = mx.take_along_axis(logits, ids[..., None], axis=-1).squeeze(-1)
    return token_logits - log_z.squeeze(-1)

def group_logprobs(model, p_ids_arr: mx.array, r_ids: mx.array, temperature: float = 1.0, actual_lengths: list[int] | None = None) -> mx.array:
    """Differentiable forward pass for a single prompt group."""
    G_p = r_ids.shape[0]
    max_r_len = r_ids.shape[1]
    p_len = p_ids_arr.shape[1]
    
    full_ids = mx.concatenate([mx.repeat(p_ids_arr, G_p, axis=0), r_ids], axis=1)
    logits = model(full_ids)
    resp_logits = logits[:, p_len - 1 : p_len - 1 + max_r_len, :]
    lps = _compute_token_logprobs(resp_logits, r_ids, temperature)
    
    if actual_lengths is not None:
        return mx.concatenate([lps[i, :actual_lengths[i]] for i in range(G_p)])
    return lps

@mx.compile
def compute_grpo_loss(policy_lps: mx.array, old_lps: mx.array, advantages: mx.array, offsets: mx.array, beta: float = 0.01, ref_lps: mx.array | None = None, eps: float = 0.2) -> mx.array:
    offsets_l = offsets.tolist()
    resp_idx = mx.array([i for i in range(len(offsets_l)-1) for _ in range(offsets_l[i+1]-offsets_l[i])], dtype=mx.int32)
    
    lr = mx.clip(policy_lps - old_lps, -6.0, 6.0)
    ratio = mx.exp(lr)
    adv = advantages[resp_idx]
    token_loss = -mx.minimum(ratio * adv, mx.clip(ratio, 1.0 - eps, 1.0 + eps) * adv)
    loss = mx.mean(mx.clip(token_loss, -100.0, 100.0))
    
    if ref_lps is not None and beta > 0.0:
        # k3 estimator: exp(ref-policy) - 1 - (ref-policy) = exp(-δ) + δ - 1 (always ≥ 0)
        # Gradient is (1 - exp(-δ)): pulls policy toward ref from both directions.
        delta = policy_lps - ref_lps
        kl_tokens = mx.exp(-delta) + delta - 1
        kl_per_resp = []
        for i in range(len(offsets_l)-1):
            kl_per_resp.append(mx.sum(kl_tokens[offsets_l[i]:offsets_l[i+1]]))
        loss = loss + beta * mx.mean(mx.stack(kl_per_resp))
    return loss
