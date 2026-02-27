"""
gwen.py — Qwen3-0.6B wrapper using our custom implementation.

Uses model.py (Qwen3), kvcache.py (KVCache), and load_weights.py.
Tokenizer loaded via AutoTokenizer from the HF cache — no torch needed.
"""

import mlx.core as mx
from transformers import AutoTokenizer

from kvcache import KVCache
from load_weights import CHECKPOINT_PATH, load_qwen3_weights
from model import Qwen3

# Qwen3-0.6B architecture (from config.json)
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

_model     = None
_tokenizer = None


def get_model() -> tuple[Qwen3, AutoTokenizer]:
    global _model, _tokenizer
    if _model is None:
        print("Loading Qwen3-0.6B...")
        _model = Qwen3(
            vocab_size=VOCAB_SIZE,
            dim=DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            intermediate_size=INTERMEDIATE_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            rope_theta=ROPE_THETA,
            eps=EPS,
            tie_word_embeddings=True,
            use_qk_norm=True,
            rope_traditional=False,
        )
        load_qwen3_weights(_model)
        mx.eval(_model.parameters())

        _tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_PATH))
        print("Ready.")
    return _model, _tokenizer


def _make_cache() -> KVCache:
    return KVCache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN)


def chat(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a response using our Qwen3 model with KV cache decoding."""
    model, tokenizer = get_model()

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    prompt_ids = tokenizer.encode(text)
    cache = _make_cache()

    # Prefill
    logits, cache = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(logits)
    cache.advance(len(prompt_ids))

    # Decode
    eos = tokenizer.eos_token_id
    generated = []
    for _ in range(max_tokens):
        last = logits[0, -1, :]
        if temperature < 1e-6:
            next_tok = int(mx.argmax(last).item())
        else:
            next_tok = int(mx.random.categorical(last / temperature).item())

        if next_tok == eos:
            break
        generated.append(next_tok)

        logits, cache = model(mx.array([[next_tok]]), cache=cache)
        mx.eval(logits)
        cache.advance(1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def batch_generate(
    prompts: list[str], max_tokens: int = 256, temperature: float = 0.7
) -> list[str]:
    """Generate responses for a list of prompts (sequential)."""
    return [chat(p, max_tokens=max_tokens, temperature=temperature) for p in prompts]


def raw_generate(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Generate from pre-formatted text (no chat template applied).
    Used by the rubric judge which constructs its own prompt strings.
    """
    prompt_ids = tokenizer.encode(text)
    cache = _make_cache()

    logits, cache = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(logits)
    cache.advance(len(prompt_ids))

    eos = tokenizer.eos_token_id
    generated = []
    for _ in range(max_tokens):
        last = logits[0, -1, :]
        if temperature < 1e-6:
            next_tok = int(mx.argmax(last).item())
        else:
            next_tok = int(mx.random.categorical(last / temperature).item())
        if next_tok == eos:
            break
        generated.append(next_tok)
        logits, cache = model(mx.array([[next_tok]]), cache=cache)
        mx.eval(logits)
        cache.advance(1)

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    print("=== Gwen Smoke Test ===\n")
    response = chat("In one sentence, what makes someone a good person?")
    print(f"Response: {response}\n")
