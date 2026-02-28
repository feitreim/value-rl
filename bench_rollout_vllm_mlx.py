"""
bench_rollout_vllm_mlx.py — Benchmark rollout throughput:
  - Ours: grpo.sample_group
  - Baseline 1: mlx_lm.generate.batch_generate
  - Baseline 2: vllm-mlx EngineCore.generate_batch_sync

Usage:
  uv run bench_rollout_vllm_mlx.py
  uv run bench_rollout_vllm_mlx.py --batch 2 --groups 2 --max-tokens 64 --warmup 1 --runs 2
  uv run bench_rollout_vllm_mlx.py --batch 2 --groups 2 --max-tokens 64 --benchmark-judge
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
from rubric import DEFAULT_CRITERIA, Rubric

try:
    from vllm_mlx.engine_core import EngineConfig, EngineCore
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import SchedulerConfig
except ImportError as e:
    raise RuntimeError(
        "vllm-mlx is not installed. Install with:\n"
        "  uv pip install git+https://github.com/waybarrios/vllm-mlx.git"
    ) from e


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
        f"{name:<10} | mean {mean_s*1000:8.1f} ms | std {std_s*1000:6.1f} ms "
        f"| total_out_toks {total_tokens:5d} | throughput {tps:7.2f} tok/s"
    )


def _judge_stats(name: str, times_s: list[float], pair_counts: list[int], criteria_count: int) -> None:
    mean_s = statistics.mean(times_s)
    std_s = statistics.pstdev(times_s) if len(times_s) > 1 else 0.0
    total_pairs = sum(pair_counts)
    pairs_ps = total_pairs / sum(times_s) if times_s else 0.0
    evals_ps = (total_pairs * criteria_count) / sum(times_s) if times_s else 0.0
    print(
        f"{name:<12} | mean {mean_s*1000:8.1f} ms | std {std_s*1000:6.1f} ms "
        f"| pairs/s {pairs_ps:7.2f} | crit_eval/s {evals_ps:7.2f}"
    )


def _speed_tag(speedup: float) -> str:
    return "faster" if speedup >= 1.0 else "slower"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    parser.add_argument("--batch", type=int, default=2, help="number of prompts (B)")
    parser.add_argument("--groups", type=int, default=2, help="completions per prompt (G)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-model", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument(
        "--benchmark-judge",
        action="store_true",
        help="Also benchmark rubric judging throughput on sampled completions.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["sequential", "batched", "both"],
        default="both",
        help="Judge benchmark mode (default: both).",
    )
    parser.add_argument(
        "--judge-runs",
        type=int,
        default=None,
        help="Number of judge benchmark runs (default: --runs).",
    )
    args = parser.parse_args()

    if not Path(args.prompts).exists():
        raise FileNotFoundError(args.prompts)

    prompts = load_prompts(args.prompts, args.batch)
    total_pairs = args.batch * args.groups

    print("Loading local model...")
    our_model, our_tok = get_model()

    print("Loading mlx_lm model...")
    mlx_model, mlx_tok = mlx_lm.load(str(CHECKPOINT_PATH))
    mlx_prompt_texts = [_chat_prompt(mlx_tok, p) for p in prompts for _ in range(args.groups)]
    mlx_token_prompts = [mlx_tok.encode(t) for t in mlx_prompt_texts]

    print("Loading vllm-mlx model...")
    vllm_model, vllm_tok = mlx_lm.load(args.vllm_model)
    vllm_prompt_texts = [_chat_prompt(vllm_tok, p) for p in prompts for _ in range(args.groups)]
    vllm_token_prompts = [vllm_tok.encode(t) for t in vllm_prompt_texts]

    vllm_sched = SchedulerConfig(
        max_num_seqs=max(1, total_pairs),
        prefill_batch_size=max(1, min(32, total_pairs)),
        completion_batch_size=max(1, min(32, total_pairs)),
    )
    vllm_engine = EngineCore(
        vllm_model,
        vllm_tok,
        EngineConfig(model_name=args.vllm_model, scheduler_config=vllm_sched),
    )
    vllm_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
    )

    print(
        f"\nConfig: B={args.batch} G={args.groups} pairs={total_pairs} "
        f"max_tokens={args.max_tokens} temp={args.temperature} "
        f"warmup={args.warmup} runs={args.runs}\n"
    )

    try:
        # Warmup to avoid counting first-run compilation/cache setup.
        for _ in range(args.warmup):
            sample_group(
                our_model, our_tok, prompts,
                G=args.groups, temperature=args.temperature, max_tokens=args.max_tokens
            )
            _ = batch_generate(
                mlx_model,
                mlx_tok,
                mlx_token_prompts,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=args.temperature),
                prefill_batch_size=max(1, min(8, total_pairs)),
                completion_batch_size=max(1, min(32, total_pairs)),
                verbose=False,
            )
            _ = vllm_engine.generate_batch_sync(vllm_token_prompts, vllm_params)

        our_times, mlx_times, vllm_times = [], [], []
        our_toks, mlx_toks, vllm_toks = [], [], []
        rollout_batches: list[tuple[list[str], list[str]]] = []

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
            rollout_batches.append(([p for p in prompts for _ in range(args.groups)], ours))

            mx.random.seed(args.seed + i)
            t0 = time.perf_counter()
            out = batch_generate(
                mlx_model,
                mlx_tok,
                mlx_token_prompts,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=args.temperature),
                prefill_batch_size=max(1, min(8, total_pairs)),
                completion_batch_size=max(1, min(32, total_pairs)),
                verbose=False,
            )
            dt = time.perf_counter() - t0
            mlx_times.append(dt)
            mlx_toks.append(sum(len(mlx_tok.encode(t)) for t in out.texts))

            mx.random.seed(args.seed + i)
            t0 = time.perf_counter()
            vouts = vllm_engine.generate_batch_sync(vllm_token_prompts, vllm_params)
            dt = time.perf_counter() - t0
            vllm_times.append(dt)
            vllm_toks.append(sum(len(vllm_tok.encode(o.output_text)) for o in vouts))

    finally:
        vllm_engine.close()

    _stats("ours", our_times, our_toks)
    _stats("mlx_lm", mlx_times, mlx_toks)
    _stats("vllm_mlx", vllm_times, vllm_toks)

    mean_ours = statistics.mean(our_times)
    mean_mlx = statistics.mean(mlx_times)
    mean_vllm = statistics.mean(vllm_times)

    speedup_mlx = mean_mlx / mean_ours if mean_ours > 0 else float("inf")
    speedup_vllm = mean_vllm / mean_ours if mean_ours > 0 else float("inf")

    print(
        f"\nResult vs mlx_lm: ours is {speedup_mlx:.2f}x {_speed_tag(speedup_mlx)} "
        "on this rollout workload."
    )
    print(
        f"Result vs vllm_mlx: ours is {speedup_vllm:.2f}x {_speed_tag(speedup_vllm)} "
        "on this rollout workload."
    )

    if args.benchmark_judge:
        judge_runs = args.judge_runs if args.judge_runs is not None else args.runs

        while len(rollout_batches) < judge_runs:
            i = len(rollout_batches)
            mx.random.seed(args.seed + args.runs + i)
            rollout_prompts, rollout_completions = sample_group(
                our_model, our_tok, prompts,
                G=args.groups, temperature=args.temperature, max_tokens=args.max_tokens
            )
            rollout_batches.append((rollout_prompts, rollout_completions))

        def run_judge_bench(mode: str) -> tuple[list[float], list[int]]:
            # Warmup: compile kernels, then clear cache so timed runs are uncached.
            warm = Rubric(DEFAULT_CRITERIA, our_model, our_tok)
            for i in range(args.warmup):
                p, c = rollout_batches[i % len(rollout_batches)]
                if mode == "batched":
                    rewards, _ = warm.score_detailed_batched(p, c)
                else:
                    rewards, _ = warm.score_detailed(p, c)
                mx.eval(rewards)
                warm._cache.clear()

            times_s, pairs = [], []
            for i in range(judge_runs):
                p, c = rollout_batches[i]
                rubric = Rubric(DEFAULT_CRITERIA, our_model, our_tok)
                t0 = time.perf_counter()
                if mode == "batched":
                    rewards, _ = rubric.score_detailed_batched(p, c)
                else:
                    rewards, _ = rubric.score_detailed(p, c)
                mx.eval(rewards)
                times_s.append(time.perf_counter() - t0)
                pairs.append(len(p))
            return times_s, pairs

        print(
            f"\nJudge benchmark: pairs/run={total_pairs} criteria={len(DEFAULT_CRITERIA)} "
            f"warmup={args.warmup} runs={judge_runs}\n"
        )

        seq_times = bat_times = None
        if args.judge_mode in {"sequential", "both"}:
            seq_times, seq_pairs = run_judge_bench("sequential")
            _judge_stats("judge_seq", seq_times, seq_pairs, len(DEFAULT_CRITERIA))
        if args.judge_mode in {"batched", "both"}:
            bat_times, bat_pairs = run_judge_bench("batched")
            _judge_stats("judge_batch", bat_times, bat_pairs, len(DEFAULT_CRITERIA))

        if seq_times is not None and bat_times is not None:
            mean_seq = statistics.mean(seq_times)
            mean_bat = statistics.mean(bat_times)
            j_speedup = mean_seq / mean_bat if mean_bat > 0 else float("inf")
            j_tag = "faster" if j_speedup >= 1.0 else "slower"
            print(f"\nJudge result: batched is {j_speedup:.2f}x {j_tag} than sequential.")


if __name__ == "__main__":
    main()
