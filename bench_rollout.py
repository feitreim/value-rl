"""
bench_rollout.py — Benchmark rollout throughput:
  - Ours: grpo.sample_group (prompt prefill once + KV broadcast to G)
  - Baseline: mlx_lm.generate.batch_generate

Usage:
  uv run bench_rollout.py
  uv run bench_rollout.py --batch 4 --groups 8 --max-tokens 64 --runs 5
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler

from grpo import sample_group
from load_weights import CHECKPOINT_PATH
from gwen import get_model


def load_prompts(path: str, n: int) -> list[str]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line)["prompt"])
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise ValueError(f"Need at least {n} prompts in {path}, found {len(rows)}")
    return rows


def _chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def _stats(name: str, times_s: list[float], out_tokens: list[int]) -> None:
    mean_s = statistics.mean(times_s)
    std_s = statistics.pstdev(times_s) if len(times_s) > 1 else 0.0
    total_tokens = sum(out_tokens)
    tps = total_tokens / sum(times_s) if times_s else 0.0
    print(
        f"{name:<8} | mean {mean_s*1000:8.1f} ms | std {std_s*1000:6.1f} ms "
        f"| total_out_toks {total_tokens:5d} | throughput {tps:7.2f} tok/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    parser.add_argument("--batch", type=int, default=4, help="number of prompts (B)")
    parser.add_argument("--groups", type=int, default=8, help="completions per prompt (G)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.prompts).exists():
        raise FileNotFoundError(args.prompts)

    prompts = load_prompts(args.prompts, args.batch)
    total_pairs = args.batch * args.groups

    print("Loading local model...")
    our_model, our_tok = get_model()
    print("Loading mlx_lm model...")
    mlx_model, mlx_tok = mlx_lm.load(str(CHECKPOINT_PATH))

    print(
        f"\nConfig: B={args.batch} G={args.groups} pairs={total_pairs} "
        f"max_tokens={args.max_tokens} temp={args.temperature} "
        f"warmup={args.warmup} runs={args.runs}\n"
    )

    # Warmup to avoid counting first-run compilation/cache setup.
    for _ in range(args.warmup):
        sample_group(
            our_model, our_tok, prompts,
            G=args.groups, temperature=args.temperature, max_tokens=args.max_tokens
        )
        prompt_texts = [_chat_prompt(mlx_tok, p) for p in prompts for _ in range(args.groups)]
        token_prompts = [mlx_tok.encode(t) for t in prompt_texts]
        _ = batch_generate(
            mlx_model,
            mlx_tok,
            token_prompts,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=args.temperature),
            prefill_batch_size=max(1, min(8, total_pairs)),
            completion_batch_size=max(1, min(32, total_pairs)),
            verbose=False,
        )

    our_times, mlx_times = [], []
    our_toks, mlx_toks = [], []

    for i in range(args.runs):
        mx.random.seed(args.seed + i)
        t0 = time.perf_counter()
        _, ours = sample_group(
            our_model, our_tok, prompts,
            G=args.groups, temperature=args.temperature, max_tokens=args.max_tokens
        )
        dt = time.perf_counter() - t0
        our_times.append(dt)
        our_toks.append(sum(len(our_tok.encode(t)) for t in ours))

        mx.random.seed(args.seed + i)
        prompt_texts = [_chat_prompt(mlx_tok, p) for p in prompts for _ in range(args.groups)]
        token_prompts = [mlx_tok.encode(t) for t in prompt_texts]
        t0 = time.perf_counter()
        out = batch_generate(
            mlx_model,
            mlx_tok,
            token_prompts,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=args.temperature),
            prefill_batch_size=max(1, min(8, total_pairs)),
            completion_batch_size=max(1, min(32, total_pairs)),
            verbose=False,
        )
        dt = time.perf_counter() - t0
        mlx_times.append(dt)
        mlx_toks.append(sum(len(mlx_tok.encode(t)) for t in out.texts))

    _stats("ours", our_times, our_toks)
    _stats("mlx_lm", mlx_times, mlx_toks)

    mean_ours = statistics.mean(our_times)
    mean_mlx = statistics.mean(mlx_times)
    speedup = mean_mlx / mean_ours if mean_ours > 0 else float("inf")
    tag = "faster" if speedup >= 1.0 else "slower"
    print(f"\nResult: ours is {speedup:.2f}x {tag} than mlx_lm on this rollout workload.")


if __name__ == "__main__":
    main()
