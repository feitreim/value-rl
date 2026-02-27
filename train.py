"""
train.py — GRPO training entry point

Usage:
    uv run train.py
    uv run train.py --steps 200 --batch 4 --G 8 --lr 1e-5
"""

import argparse
import json
import os
import random
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from gwen import DIM, EPS, HEAD_DIM, INTERMEDIATE_SIZE, MAX_SEQ_LEN
from gwen import NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, ROPE_THETA, VOCAB_SIZE
from gwen import get_model
from grpo import grpo_step
from load_weights import load_qwen3_weights
from model import Qwen3
from rubric import DEFAULT_CRITERIA, Rubric


def load_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [json.loads(line)["prompt"] for line in f if line.strip()]


def save_checkpoint(model, step: int, path: str = "checkpoints") -> None:
    os.makedirs(path, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    out = f"{path}/step_{step:05d}.npz"
    mx.savez(out, **weights)
    print(f"  checkpoint → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--batch",      type=int,   default=4,    help="prompts per step")
    parser.add_argument("--G",          type=int,   default=8,    help="completions per prompt")
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--beta",       type=float, default=0.01, help="KL coefficient")
    parser.add_argument("--eps",        type=float, default=0.2,  help="PPO clip radius")
    parser.add_argument("--temp",       type=float, default=0.8,  help="rollout temperature")
    parser.add_argument("--max-tokens", type=int,   default=256,  help="max tokens per completion")
    parser.add_argument("--save-every", type=int,   default=50)
    parser.add_argument("--prompts",    type=str,   default="data/prompts.jsonl")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Policy: loaded via get_model() which caches the instance
    policy, tokenizer = get_model()

    # Ref model: separate instance with same weights, frozen
    ref_model = Qwen3(
        vocab_size=VOCAB_SIZE, dim=DIM, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE_SIZE, max_seq_len=MAX_SEQ_LEN,
        rope_theta=ROPE_THETA, eps=EPS,
        tie_word_embeddings=True, use_qk_norm=True, rope_traditional=False,
    )
    load_qwen3_weights(ref_model)
    mx.eval(ref_model.parameters())
    ref_model.freeze()
    print("Models loaded.")

    rubric  = Rubric(DEFAULT_CRITERIA, policy, tokenizer)
    dataset = load_prompts(args.prompts)
    assert len(dataset) >= args.batch, f"need at least {args.batch} prompts, got {len(dataset)}"
    optimizer = optim.AdamW(learning_rate=args.lr)

    print("\nTraining config:")
    print(f"  prompts: {len(dataset)}  |  steps: {args.steps}  |  B={args.batch}  G={args.G}")
    print(f"  lr={args.lr}  beta={args.beta}  eps={args.eps}  temp={args.temp}")
    print(f"  max_tokens={args.max_tokens}  save_every={args.save_every}")
    print()

    for step in range(args.steps):
        batch = random.sample(dataset, args.batch)

        t0 = time.perf_counter()
        loss, reward = grpo_step(
            policy, ref_model, tokenizer, batch, rubric, optimizer,
            G=args.G, beta=args.beta, eps=args.eps,
            temperature=args.temp, max_tokens=args.max_tokens,
        )
        dt = time.perf_counter() - t0

        print(f"step {step+1:4d}/{args.steps} | loss {loss:7.4f} | reward {reward:6.3f} | {dt:.1f}s")

        if (step + 1) % args.save_every == 0:
            save_checkpoint(policy, step + 1)

    save_checkpoint(policy, args.steps, path="checkpoints/final")
    print("\nDone.")


if __name__ == "__main__":
    main()
