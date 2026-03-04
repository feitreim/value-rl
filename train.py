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

from model.gwen import get_model
from grpo import grpo_step
from model.lora import apply_lora
from rubric import DEFAULT_CRITERIA, Rubric
from tome_client import TomeClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=str, default=None, help="Comma-separated list of layer indices to apply LoRA to (e.g. '24,25,26,27')")
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tome-url", type=str, required=True, help="Tome scheduler URL (e.g. http://localhost:8080)")
    args = parser.parse_args()

    random.seed(args.seed)
    policy, tokenizer = get_model()
    if args.lora_rank > 0:
        layers = [int(x) for x in args.lora_layers.split(",")] if args.lora_layers else None
        print(f"LoRA rank={args.lora_rank}, layers={layers}: {apply_lora(policy, rank=args.lora_rank, layers=layers):,} trainable params")

    tome_client = TomeClient(args.tome_url)
    print(f"Using Tome at {args.tome_url} for rollouts and judging")
    
    # Sync initial weights to Tome
    tome_client.update_weights(policy)

    rubric = Rubric(DEFAULT_CRITERIA, tokenizer, tome_client=tome_client)
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
                tokenizer,
                [r["prompt"] for r in batch],
                rubric,
                optimizer,
                G=args.G,
                beta=args.beta,
                eps=args.eps,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                tome_client=tome_client,
            )

            t = data["times"]
            m = data["metrics"]

            all_completions = [c for g in data["groups"] for c in g["completions"]]
            all_rewards = [c["reward"] for c in all_completions]
            mean_len = sum(len(c["text"]) for c in all_completions) / len(all_completions)
            reward_std = (sum((r - reward) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
            criterion_names = list(all_completions[0]["scores"].keys()) if all_completions else []
            criterion_means = {k: sum(c["scores"][k] for c in all_completions) / len(all_completions) for k in criterion_names}
            scores_str = " | ".join(f"{k} {v:+.2f}" for k, v in criterion_means.items())

            print(f"step {step + 1:4d} | loss {loss:7.4f} | reward {reward:+.3f} (std {reward_std:.3f}) | kl {data['mean_kl']:7.4f} | len {mean_len:.0f} | {scores_str} | gnorm {m['grad_norm']:.2f} | total {t['total']:5.1f}s")
            print(f"  timing: rollout {t['rollout']:4.1f}s | score {t['score']:4.1f}s | grad {t['grad_step']:4.1f}s | update {t['tome_weight_update']:4.1f}s")

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
