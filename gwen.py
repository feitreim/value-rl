"""
gwen.py — Qwen3-0.6B wrapper.
"""

import os
import time
import mlx.core as mx
from transformers import AutoTokenizer

from kvcache import KVCache
from load_weights import CHECKPOINT_PATH, load_qwen3_weights
from model import Qwen3

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


def _make_cache(batch_size: int = 1) -> KVCache:
    return KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, batch_size=batch_size)


from kernels import fused_log_softmax

# ---------------------------------------------------------------------------
# Differentiable Logprob Extraction
# ---------------------------------------------------------------------------

def batch_logprobs(model, tokenizer, prompts: list[str], responses: list[str], temperature: float = 1.0) -> tuple[mx.array, mx.array, mx.array]:
    """
    Batched logprob calculation via single full-sequence forward pass.
    Groups responses by prompt for efficiency. Matches reference logprobs exactly.
    Uses standard (non-fused) MLP path for differentiability.
    """
    groups = {}
    for i, (p, r) in enumerate(zip(prompts, responses)):
        if p not in groups: groups[p] = []
        groups[p].append((i, r))

    all_lps_flat = [None] * len(prompts)
    all_tids_flat = [None] * len(prompts)
    lengths = [0] * len(prompts)

    for p, resps in groups.items():
        p_text = tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        p_len = len(tokenizer.encode(p_text))

        full_texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False
        ) for _, r in resps]
        full_token_lists = [tokenizer.encode(txt) for txt in full_texts]
        r_token_lists = [toks[p_len:] for toks in full_token_lists]

        G_p = len(resps)
        max_r_len = max(len(toks) for toks in r_token_lists)
        max_full_len = max(len(toks) for toks in full_token_lists)

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padded_full = [toks + [pad_id] * (max_full_len - len(toks)) for toks in full_token_lists]
        padded_r = [toks + [pad_id] * (max_r_len - len(toks)) for toks in r_token_lists]

        full_ids = mx.array(padded_full)  # (G_p, max_full_len)
        r_ids = mx.array(padded_r)  # (G_p, max_r_len)

        # Single full-sequence forward pass — no cache split, matches reference exactly
        all_logits, _ = model(full_ids)  # (G_p, max_full_len, V)

        # logit at position p_len-1 predicts response token 0, etc.
        resp_logits = all_logits[:, p_len - 1 : p_len - 1 + max_r_len, :]  # (G_p, max_r_len, V)

        # Compute log-softmax in fp32 for numerical stability across batch sizes
        resp_logits_f32 = resp_logits.astype(mx.float32) / temperature
        lps_all = resp_logits_f32 - mx.logsumexp(resp_logits_f32, axis=-1, keepdims=True)
        lps = mx.take_along_axis(lps_all, r_ids[..., None], axis=-1).squeeze(-1)  # (G_p, max_r_len)

        for idx_in_batch, (orig_idx, _) in enumerate(resps):
            actual_len = len(r_token_lists[idx_in_batch])
            all_lps_flat[orig_idx] = lps[idx_in_batch, :actual_len]
            all_tids_flat[orig_idx] = r_ids[idx_in_batch, :actual_len]
            lengths[orig_idx] = actual_len

    offsets = [0]
    for l in lengths: offsets.append(offsets[-1] + l)

    return mx.concatenate(all_lps_flat), mx.concatenate(all_tids_flat), mx.array(offsets, dtype=mx.int32)

def compute_grpo_loss(policy_lps: mx.array, old_lps: mx.array, advantages: mx.array, offsets: mx.array, beta: float = 0.01, ref_lps: mx.array | None = None, eps: float = 0.2) -> mx.array:
    offsets_l = offsets.tolist()
    resp_idx = mx.array([i for i in range(len(offsets_l)-1) for _ in range(offsets_l[i+1]-offsets_l[i])], dtype=mx.int32)
    
    lr = mx.clip(policy_lps - old_lps, -6.0, 6.0)
    ratio = mx.exp(lr)
    adv = advantages[resp_idx]
    token_loss = -mx.minimum(ratio * adv, mx.clip(ratio, 1.0 - eps, 1.0 + eps) * adv)
    loss = mx.mean(mx.clip(token_loss, -100.0, 100.0))
    
    if ref_lps is not None and beta > 0.0:
        kl_tokens = mx.clip(policy_lps - ref_lps, -30.0, 30.0)
        kl_per_resp = []
        for i in range(len(offsets_l)-1):
            kl_per_resp.append(mx.sum(kl_tokens[offsets_l[i]:offsets_l[i+1]]))
        loss = loss + beta * mx.mean(mx.stack(kl_per_resp))
    return loss

def chat(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    model, tokenizer = get_model()
    p_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    p_ids = tokenizer.encode(p_text)
    cache = _make_cache()
    logits, cache = model(mx.array([p_ids]), cache=cache)
    mx.eval(logits); cache.advance(len(p_ids))
    eos, generated = tokenizer.eos_token_id, []
    for _ in range(max_tokens):
        last = logits[0, -1, :]
        next_tok = int(mx.argmax(last).item() if temperature < 1e-6 else mx.random.categorical(last/temperature).item())
        if next_tok == eos: break
        generated.append(next_tok)
        logits, cache = model(mx.array([[next_tok]]), cache=cache)
        mx.eval(logits); cache.advance(1)
    return tokenizer.decode(generated, skip_special_tokens=True)

def raw_generate(model, tokenizer, text: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    p_ids = tokenizer.encode(text)
    cache = _make_cache()
    logits, cache = model(mx.array([p_ids]), cache=cache)
    mx.eval(logits); cache.advance(len(p_ids))
    eos, generated = tokenizer.eos_token_id, []
    for _ in range(max_tokens):
        last = logits[0, -1, :]
        next_tok = int(mx.argmax(last).item() if temperature < 1e-6 else mx.random.categorical(last/temperature).item())
        if next_tok == eos: break
        generated.append(next_tok)
        logits, cache = model(mx.array([[next_tok]]), cache=cache)
        mx.eval(logits); cache.advance(1)
    return tokenizer.decode(generated, skip_special_tokens=True)
