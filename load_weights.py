"""
load_weights.py — Load Qwen3-0.6B weights from HuggingFace safetensors cache.

Uses mx.load() directly — no torch or safetensors dependency needed.
"""

from pathlib import Path

import mlx.core as mx

CHECKPOINT_PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B" / "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def load_qwen3_weights(
    model,
    checkpoint_path: str | Path | None = None,
    dtype=mx.bfloat16,
) -> None:
    """
    Load Qwen3 weights from HF safetensors into our model.

    Handles sharded checkpoints (multiple .safetensors files) transparently.
    Weight keys follow the standard HF Qwen3 layout.
    """
    path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    shards = sorted(path.glob("*.safetensors"))
    assert shards, f"No .safetensors files found in {path}"

    weights: dict[str, mx.array] = {}
    for shard in shards:
        weights.update(mx.load(str(shard)))

    def get(key: str) -> mx.array:
        return weights.pop(key).astype(dtype)

    model.embed_tokens.weight = get("model.embed_tokens.weight")
    model.norm.weight = get("model.norm.weight")

    if not model.tie_word_embeddings:
        model.lm_head.weight = get("lm_head.weight")
    else:
        weights.pop("lm_head.weight", None)

    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        attn = layer.self_attn

        q_w = get(f"{p}.self_attn.q_proj.weight")
        k_w = get(f"{p}.self_attn.k_proj.weight")
        v_w = get(f"{p}.self_attn.v_proj.weight")
        attn.qkv_proj.weight = mx.concatenate([q_w, k_w, v_w], axis=0)

        attn.o_proj.weight = get(f"{p}.self_attn.o_proj.weight")

        if attn.q_norm is not None:
            attn.q_norm.weight = get(f"{p}.self_attn.q_norm.weight")
            attn.k_norm.weight = get(f"{p}.self_attn.k_norm.weight")
        else:
            weights.pop(f"{p}.self_attn.q_norm.weight", None)
            weights.pop(f"{p}.self_attn.k_norm.weight", None)

        layer.input_layernorm.weight = get(f"{p}.input_layernorm.weight")
        layer.post_attention_layernorm.weight = get(f"{p}.post_attention_layernorm.weight")

        gate_w = get(f"{p}.mlp.gate_proj.weight")
        up_w = get(f"{p}.mlp.up_proj.weight")
        layer.mlp.gate_up_proj.weight = mx.concatenate([gate_w, up_w], axis=0)

        layer.mlp.down_proj.weight = get(f"{p}.mlp.down_proj.weight")

    if weights:
        print(f"Warning: {len(weights)} unused weight keys: {list(weights)[:5]}")
