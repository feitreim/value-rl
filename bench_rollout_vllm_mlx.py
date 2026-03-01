"""
bench_rollout_vllm_mlx.py — Benchmark rollout + judge throughput:
  - Ours rollout: grpo.sample_group
  - vllm-mlx rollout: EngineCore.generate_batch_sync
  - Ours judge: Rubric batched scoring
  - vllm-mlx judge: batched judge prompts through EngineCore.generate_batch_sync

Usage:
  uv run bench_rollout_vllm_mlx.py
  uv run bench_rollout_vllm_mlx.py --batch 2 --groups 2 --max-tokens 64 --warmup 1 --runs 2
  uv run bench_rollout_vllm_mlx.py --batch 2 --groups 2 --max-tokens 64 --benchmark-judge
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm

from grpo import sample_group
from gwen import get_model
from load_weights import CHECKPOINT_PATH
from rubric import DEFAULT_CRITERIA, Rubric

try:
    from vllm_mlx.engine_core import EngineConfig, EngineCore
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import SchedulerConfig
except ImportError as e:
    raise RuntimeError("vllm-mlx is not installed. Install with:\n  uv pip install git+https://github.com/waybarrios/vllm-mlx.git") from e


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
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _stats(name: str, times_s: list[float], out_tokens: list[int]) -> None:
    mean_s = statistics.mean(times_s)
    std_s = statistics.pstdev(times_s) if len(times_s) > 1 else 0.0
    total_tokens = sum(out_tokens)
    tps = total_tokens / sum(times_s) if times_s else 0.0
    print(f"{name:<10} | mean {mean_s * 1000:8.1f} ms | std {std_s * 1000:6.1f} ms | total_out_toks {total_tokens:5d} | throughput {tps:7.2f} tok/s")


def _judge_stats(name: str, times_s: list[float], pair_counts: list[int], criteria_count: int) -> None:
    mean_s = statistics.mean(times_s)
    std_s = statistics.pstdev(times_s) if len(times_s) > 1 else 0.0
    total_pairs = sum(pair_counts)
    pairs_ps = total_pairs / sum(times_s) if times_s else 0.0
    evals_ps = (total_pairs * criteria_count) / sum(times_s) if times_s else 0.0
    print(f"{name:<12} | mean {mean_s * 1000:8.1f} ms | std {std_s * 1000:6.1f} ms | pairs/s {pairs_ps:7.2f} | crit_eval/s {evals_ps:7.2f}")


def _speed_tag(speedup: float) -> str:
    return "faster" if speedup >= 1.0 else "slower"


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "insufficient memory" in msg or "outofmemory" in msg or "out of memory" in msg


def _parse_score(raw: str) -> int:
    matches = re.findall(r"\b[1-5]\b", raw)
    return int(matches[-1]) if matches else 3


def _build_vllm_judge_token_prompts(
    tokenizer,
    prompts: list[str],
    completions: list[str],
) -> tuple[list[list[int]], int]:
    token_prompts: list[list[int]] = []
    for prompt, completion in zip(prompts, completions):
        for criterion in DEFAULT_CRITERIA:
            judge_prompt = criterion.scoring_prompt.format(prompt=prompt, response=completion)
            text = _chat_prompt(tokenizer, judge_prompt)
            token_prompts.append(tokenizer.encode(text))
    return token_prompts, len(DEFAULT_CRITERIA)


def _vllm_judge_rewards(
    engine: EngineCore,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    judge_chunk_size: int,
) -> list[float]:
    token_prompts, n_criteria = _build_vllm_judge_token_prompts(tokenizer, prompts, completions)
    judge_params = SamplingParams(max_tokens=16, temperature=0.0, top_p=1.0, top_k=0)
    outs = []
    i = 0
    chunk = min(max(1, judge_chunk_size), len(token_prompts)) if token_prompts else 1
    while i < len(token_prompts):
        j = min(i + chunk, len(token_prompts))
        try:
            outs.extend(engine.generate_batch_sync(token_prompts[i:j], judge_params))
            i = j
        except RuntimeError as e:
            if not _is_oom_error(e) or chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            mx.clear_cache()

    scores = [_parse_score(o.output_text) for o in outs]
    rewards = []
    idx = 0
    for _ in range(len(prompts)):
        total = 0.0
        for criterion in DEFAULT_CRITERIA:
            score = scores[idx]
            total += (score - 3) / 2.0 * criterion.weight
            idx += 1
        rewards.append(total)

    assert idx == len(prompts) * n_criteria
    return rewards


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
        help="Also benchmark judging throughput: ours-batched vs vllm_mlx.",
    )
    parser.add_argument(
        "--judge-runs",
        type=int,
        default=None,
        help="Number of judge benchmark runs (default: --runs).",
    )
    parser.add_argument(
        "--judge-chunk-size",
        type=int,
        default=24,
        help="Micro-batch size for judge requests to avoid Metal OOM.",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help="Micro-batch size for ours rollout sampling to avoid Metal OOM.",
    )
    args = parser.parse_args()

    if not Path(args.prompts).exists():
        raise FileNotFoundError(args.prompts)

    prompts = load_prompts(args.prompts, args.batch)
    total_pairs = args.batch * args.groups

    print("Loading local model...")
    our_model, our_tok = get_model()

    print("Loading vllm-mlx model...")
    vllm_model, vllm_tok = mlx_lm.load(args.vllm_model)
    vllm_prompt_texts = [_chat_prompt(vllm_tok, p) for p in prompts for _ in range(args.groups)]
    vllm_token_prompts = [vllm_tok.encode(t) for t in vllm_prompt_texts]

    vllm_sched = SchedulerConfig(
        max_num_seqs=max(1, total_pairs),
        prefill_batch_size=max(1, min(args.rollout_batch_size, total_pairs)),
        completion_batch_size=max(1, min(args.rollout_batch_size, total_pairs)),
    )
    vllm_engine = EngineCore(
        vllm_model,
        vllm_tok,
        EngineConfig(model_name=args.vllm_model, scheduler_config=vllm_sched),
    )
    vllm_rollout_params = SamplingParams(
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
        for i in range(args.warmup):
            print(f"Warmup {i + 1}/{args.warmup}...")
            sample_group(
                our_model,
                our_tok,
                prompts,
                G=args.groups,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                rollout_batch_size=args.rollout_batch_size,
            )
            print("\n  vllm warmup...")
            _ = vllm_engine.generate_batch_sync(vllm_token_prompts, vllm_rollout_params)
            print("  done.")

        our_times, vllm_times = [], []
        our_toks, vllm_toks = [], []
        rollout_batches: list[tuple[list[str], list[str]]] = []

        for i in range(args.runs):
            print(f"Run {i + 1}/{args.runs}...")
            mx.random.seed(args.seed + i)
            t0 = time.perf_counter()
            _, ours = sample_group(
                our_model,
                our_tok,
                prompts,
                G=args.groups,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                rollout_batch_size=args.rollout_batch_size,
            )
            dt = time.perf_counter() - t0
            print(f"  ours: {dt:.1f}s")
            our_times.append(dt)
            our_toks.append(sum(len(our_tok.encode(t)) for t in ours))
            rollout_batches.append(([p for p in prompts for _ in range(args.groups)], ours))

            mx.random.seed(args.seed + i)
            t0 = time.perf_counter()
            vouts = vllm_engine.generate_batch_sync(vllm_token_prompts, vllm_rollout_params)
            dt = time.perf_counter() - t0
            print(f"  vllm: {dt:.1f}s")
            vllm_times.append(dt)
            vllm_toks.append(sum(len(vllm_tok.encode(o.output_text)) for o in vouts))

        _stats("ours", our_times, our_toks)
        _stats("vllm_mlx", vllm_times, vllm_toks)

        mean_ours = statistics.mean(our_times)
        mean_vllm = statistics.mean(vllm_times)
        speedup_vllm = mean_vllm / mean_ours if mean_ours > 0 else float("inf")
        print(f"\nResult vs vllm_mlx: ours is {speedup_vllm:.2f}x {_speed_tag(speedup_vllm)} on this rollout workload.")

        if args.benchmark_judge:
            judge_runs = args.judge_runs if args.judge_runs is not None else args.runs

            while len(rollout_batches) < judge_runs:
                i = len(rollout_batches)
                mx.random.seed(args.seed + args.runs + i)
                rollout_prompts, rollout_completions = sample_group(
                    our_model,
                    our_tok,
                    prompts,
                    G=args.groups,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    rollout_batch_size=args.rollout_batch_size,
                )
                rollout_batches.append((rollout_prompts, rollout_completions))

            print(f"\nJudge benchmark: pairs/run={total_pairs} criteria={len(DEFAULT_CRITERIA)} warmup={args.warmup} runs={judge_runs}\n")

            # Warmups
            warm_rubric = Rubric(
                DEFAULT_CRITERIA,
                our_model,
                our_tok,
                judge_batch_size=args.judge_chunk_size,
            )
            for i in range(args.warmup):
                p, c = rollout_batches[i % len(rollout_batches)]
                rewards, _ = warm_rubric.score_detailed_batched(p, c)
                mx.eval(rewards)
                warm_rubric._cache.clear()
                _ = _vllm_judge_rewards(vllm_engine, vllm_tok, p, c, args.judge_chunk_size)

            # Timed runs
            our_j_times, our_j_pairs = [], []
            vllm_j_times, vllm_j_pairs = [], []
            for i in range(judge_runs):
                p, c = rollout_batches[i]

                rubric = Rubric(
                    DEFAULT_CRITERIA,
                    our_model,
                    our_tok,
                    judge_batch_size=args.judge_chunk_size,
                )
                t0 = time.perf_counter()
                rewards, _ = rubric.score_detailed_batched(p, c)
                mx.eval(rewards)
                our_j_times.append(time.perf_counter() - t0)
                our_j_pairs.append(len(p))

                t0 = time.perf_counter()
                _ = _vllm_judge_rewards(vllm_engine, vllm_tok, p, c, args.judge_chunk_size)
                vllm_j_times.append(time.perf_counter() - t0)
                vllm_j_pairs.append(len(p))

            _judge_stats("judge_ours", our_j_times, our_j_pairs, len(DEFAULT_CRITERIA))
            _judge_stats("judge_vllm", vllm_j_times, vllm_j_pairs, len(DEFAULT_CRITERIA))

            mean_ours_j = statistics.mean(our_j_times)
            mean_vllm_j = statistics.mean(vllm_j_times)
            j_speedup = mean_vllm_j / mean_ours_j if mean_ours_j > 0 else float("inf")
            print(f"\nJudge result vs vllm_mlx: ours is {j_speedup:.2f}x {_speed_tag(j_speedup)} on this judging workload.")
    finally:
        vllm_engine.close()


if __name__ == "__main__":
    main()
