"""
train.py — GRPO training entry point
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from gwen import DIM, EPS, HEAD_DIM, INTERMEDIATE_SIZE, MAX_SEQ_LEN
from gwen import NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, ROPE_THETA, VOCAB_SIZE
from gwen import get_model
from grpo import grpo_step
from load_weights import load_qwen3_weights
from lora import apply_lora
from model import Qwen3
from rubric import DEFAULT_CRITERIA, Rubric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    policy, tokenizer = get_model()
    if args.lora_rank > 0:
        print(f"LoRA rank={args.lora_rank}: {apply_lora(policy, rank=args.lora_rank):,} trainable params")

    ref_model = Qwen3(
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
    )
    load_qwen3_weights(ref_model)
    mx.eval(ref_model.parameters())
    ref_model.freeze()

    rubric = Rubric(DEFAULT_CRITERIA, policy, tokenizer)
    with open(args.prompts) as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    optimizer = optim.AdamW(learning_rate=args.lr)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"rollouts/{stamp}.jsonl"
    os.makedirs("rollouts", exist_ok=True)

    print(f"\nTraining: {len(dataset)} prompts, B={args.batch} G={args.G}, log={log_path}\n")

    with open(log_path, "w") as rollout_log:
        for step in range(args.steps):
            batch = random.sample(dataset, args.batch)
            loss, reward, data = grpo_step(
                policy,
                ref_model,
                tokenizer,
                [r["prompt"] for r in batch],
                rubric,
                optimizer,
                G=args.G,
                beta=args.beta,
                eps=args.eps,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                rollout_batch_size=args.rollout_batch_size,
            )

            t = data["times"]
            m = data["metrics"]
            print(f"step {step + 1:4d} | loss {loss:7.4f} | reward {reward:6.6f} | total {t['total']:5.1f}s | rollout {m['rollout_tps']:6.1f} tps")
            print(f"  timing: rollout {t['rollout']:4.1f}s | score {t['score']:4.1f}s | old_lp {t['old_lps']:4.1f}s | grad {t['grad_step']:4.1f}s")

            data.update({"step": step + 1, "timestamp": datetime.now(timezone.utc).isoformat()})
            rollout_log.write(json.dumps(data) + "\n")
            rollout_log.flush()

            if (step + 1) % args.save_every == 0:
                mx.savez(
                    f"checkpoints/step_{step + 1:05d}.npz",
                    **dict(tree_flatten(policy.trainable_parameters())),
                )

    mx.savez("checkpoints/final.npz", **dict(tree_flatten(policy.trainable_parameters())))
    print("\nDone.")


if __name__ == "__main__":
    main()
