"""
bench.py — Performance benchmarks comparing our Metal kernels and KV-sharing
           batch_logprobs against equivalent MLX ops and naive mlx_lm approach.

Sections:
  1. Kernel benchmarks (Metal vs MLX equivalents, multiple sizes)
  2. batch_logprobs: KV-sharing (ours) vs naive (mlx_lm-style)
  3. End-to-end forward pass: our model vs mlx_lm model

Run with:
    uv run bench.py
"""

import time

import mlx.core as mx
import mlx_lm

from gwen import _make_cache, get_model
from gwen_metal import (
    batch_logprobs,
    compute_kl,
    fused_log_softmax,
    gather_logprobs,
    grpo_token_loss,
    logprobs_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(fn, n_warmup=5, n_runs=50):
    """Time fn() with warmup. Returns mean wall-clock time in seconds."""
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) / n_runs


def _header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _row(label, metal_ms, mlx_ms, note=""):
    speedup = mlx_ms / metal_ms if metal_ms > 0 else float("inf")
    tag = "FASTER" if speedup >= 1.0 else "slower"
    print(f"  {label:<30}  Metal {metal_ms:7.3f} ms  MLX {mlx_ms:7.3f} ms  {speedup:.2f}x {tag}  {note}")


# ---------------------------------------------------------------------------
# Section 1: Kernel benchmarks
# ---------------------------------------------------------------------------

def bench_kernels(V: int = 151936):
    _header("1. Kernel Benchmarks (Metal vs MLX)")

    # 1a. fused_log_softmax at multiple row counts
    print(f"\n  1a. fused_log_softmax (V={V})")
    for n_rows in [1, 16, 128, 512]:
        logits = mx.random.normal((n_rows, V), dtype=mx.bfloat16)
        mx.eval(logits)

        t_metal = _bench(lambda: mx.eval(fused_log_softmax(logits, temperature=0.7))) * 1000
        t_mlx   = _bench(lambda: mx.eval(
            logits / 0.7 - mx.logsumexp(logits / 0.7, axis=-1, keepdims=True)
        )) * 1000
        _row(f"  {n_rows} rows", t_metal, t_mlx)

    # 1b. gather_logprobs at multiple sequence lengths
    print(f"\n  1b. gather_logprobs (V={V})")
    for N in [64, 256, 512, 2048]:
        lp = mx.random.normal((N, V), dtype=mx.bfloat16)
        tids = mx.random.randint(0, V, (N,)).astype(mx.int32)
        mx.eval(lp, tids)

        t_metal = _bench(lambda: mx.eval(gather_logprobs(lp, tids))) * 1000
        t_mlx   = _bench(lambda: mx.eval(lp[mx.arange(N), tids])) * 1000
        _row(f"  N={N:5d}", t_metal, t_mlx)

    # 1c. kl_per_response
    print(f"\n  1c. kl_per_response (segmented reduction)")
    for B, resp_len in [(4, 64), (8, 128), (16, 256)]:
        N = B * resp_len
        plps = mx.random.normal((N,), dtype=mx.bfloat16)
        rlps = mx.random.normal((N,), dtype=mx.bfloat16)
        offs = mx.array(list(range(0, N + 1, resp_len)), dtype=mx.int32)
        mx.eval(plps, rlps, offs)

        t_metal = _bench(lambda: mx.eval(compute_kl(plps, rlps, offs))) * 1000

        # MLX equivalent: segment sum via loop
        offs_list = offs.tolist()
        def _mlx_kl():
            parts = [mx.sum(plps[offs_list[i]:offs_list[i+1]] - rlps[offs_list[i]:offs_list[i+1]])
                     for i in range(B)]
            mx.eval(mx.stack(parts))
        t_mlx = _bench(_mlx_kl) * 1000
        _row(f"  B={B:2d} resp_len={resp_len:3d}", t_metal, t_mlx)

    # 1d. grpo_token_loss
    print(f"\n  1d. grpo_token_loss")
    for B, resp_len in [(4, 64), (8, 128), (16, 256)]:
        N = B * resp_len
        plps = mx.random.normal((N,), dtype=mx.bfloat16)
        olps = plps + mx.random.normal((N,), dtype=mx.bfloat16) * 0.05
        advs = mx.random.normal((B,))
        resp_idx = mx.repeat(mx.arange(B), resp_len).astype(mx.int32)
        mx.eval(plps, olps, advs, resp_idx)

        eps = 0.2

        t_metal = _bench(lambda: mx.eval(grpo_token_loss(plps, olps, resp_idx, advs, eps))) * 1000

        # MLX equivalent
        def _mlx_grpo():
            ratio = mx.exp(plps - olps)
            adv_per_tok = advs[resp_idx]
            unclipped = ratio * adv_per_tok
            clipped = mx.clip(ratio, 1 - eps, 1 + eps) * adv_per_tok
            loss = -mx.minimum(unclipped, clipped)
            mx.eval(loss)
        t_mlx = _bench(_mlx_grpo) * 1000
        _row(f"  B={B:2d} N={N:5d}", t_metal, t_mlx)


# ---------------------------------------------------------------------------
# Section 2: batch_logprobs — KV sharing vs naive (mlx_lm-style)
# ---------------------------------------------------------------------------

def _naive_batch_logprobs(model, tokenizer, prompts, responses, temperature=1.0):
    """
    Naive approach: one full forward pass per (prompt, response) pair.
    No KV cache sharing between responses to the same prompt.
    This is the mlx_lm-style baseline.
    """
    all_lps, all_tids, offsets = [], [], [0]
    for prompt, response in zip(prompts, responses):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer.encode(prompt_text)
        resp_start = len(prompt_ids)

        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
        )
        tokens = tokenizer.encode(full_text)
        resp_tids = tokens[resp_start:]
        R = len(resp_tids)

        # Single forward pass over full (prompt + response[:-1])
        input_ids = mx.array([tokens[:-1]])
        logits, _ = model(input_ids, cache=_make_cache())
        mx.eval(logits)

        # Slice response logits and gather
        resp_logits = logits[0, resp_start - 1:]  # (R, V)
        log_probs   = resp_logits / temperature - mx.logsumexp(resp_logits / temperature, axis=-1, keepdims=True)
        resp_tids_arr = mx.array(resp_tids, dtype=mx.int32)
        lps = log_probs[mx.arange(R), resp_tids_arr]
        mx.eval(lps)

        all_lps.append(lps)
        all_tids.append(resp_tids_arr)
        offsets.append(offsets[-1] + R)

    return mx.concatenate(all_lps), mx.concatenate(all_tids), mx.array(offsets, dtype=mx.int32)


def bench_batch_logprobs(model, tokenizer):
    _header("2. batch_logprobs: KV-sharing (ours) vs naive (mlx_lm-style)")
    print("  Each row: B prompts × G responses each (total B×G pairs)")
    print("  KV-sharing saves (G-1) prompt forward passes per unique prompt.")
    print("  Wins when: prompt is long OR G is large (savings > 2-pass overhead).\n")

    # Short prompts — overhead dominates at small G
    short_prompt = "Why is honesty important?"
    long_prompt = (
        "Consider a scenario in which a researcher has spent years developing a new methodology "
        "for evaluating scientific claims. The methodology relies on careful decomposition of "
        "arguments into sub-claims, each evaluated against empirical evidence independently. "
        "However, the researcher discovers that some colleagues are using the method incorrectly, "
        "applying it superficially without the rigorous decomposition step. "
        "What are the epistemic risks of this superficial application, and how should the "
        "researcher respond to preserve the integrity of the methodology?"
    )
    response_pool = [
        "It means being genuinely interested in how and why things work.",
        "Honesty builds trust and creates stable relationships over time.",
        "Look for unsupported assumptions and logical leaps in the reasoning.",
        "Critical thinking means questioning premises and evaluating evidence.",
        "Curiosity drives learning; without it, growth stagnates.",
        "Honesty is difficult but foundational to any real understanding.",
        "Flawed arguments often rely on emotional appeals or false dichotomies.",
        "Thinking critically requires stepping back from intuition and examining structure.",
    ]

    print("  Short prompt:")
    for G in [2, 4, 8]:
        B = 2
        prompts   = [short_prompt for _ in range(B) for _ in range(G)]
        responses = [response_pool[(i * G + j) % len(response_pool)]
                     for i in range(B) for j in range(G)]

        t_ours  = _bench(lambda: batch_logprobs(model, tokenizer, prompts, responses), n_warmup=2, n_runs=8) * 1000
        t_naive = _bench(lambda: _naive_batch_logprobs(model, tokenizer, prompts, responses), n_warmup=2, n_runs=8) * 1000
        speedup = t_naive / t_ours
        tag = "FASTER" if speedup >= 1.0 else "slower"
        print(f"  B={B} G={G:2d}  total={B*G:2d} pairs   "
              f"Ours {t_ours:7.1f} ms   Naive {t_naive:7.1f} ms   {speedup:.2f}x {tag}")

    print()
    print("  Long prompt (~120 tokens):")
    for G in [2, 4, 8]:
        B = 2
        prompts   = [long_prompt for _ in range(B) for _ in range(G)]
        responses = [response_pool[(i * G + j) % len(response_pool)]
                     for i in range(B) for j in range(G)]

        t_ours  = _bench(lambda: batch_logprobs(model, tokenizer, prompts, responses), n_warmup=2, n_runs=8) * 1000
        t_naive = _bench(lambda: _naive_batch_logprobs(model, tokenizer, prompts, responses), n_warmup=2, n_runs=8) * 1000
        speedup = t_naive / t_ours
        tag = "FASTER" if speedup >= 1.0 else "slower"
        print(f"  B={B} G={G:2d}  total={B*G:2d} pairs   "
              f"Ours {t_ours:7.1f} ms   Naive {t_naive:7.1f} ms   {speedup:.2f}x {tag}")


# ---------------------------------------------------------------------------
# Section 3: Forward pass — our model vs mlx_lm
# ---------------------------------------------------------------------------

def bench_forward_pass(our_model, tokenizer):
    _header("3. Forward Pass: our model vs mlx_lm")
    print("  Prefill timing at various sequence lengths.\n")

    # Load mlx_lm model for comparison
    print("  Loading mlx_lm model for comparison...")
    from load_weights import CHECKPOINT_PATH
    mlxlm_model, mlxlm_tok = mlx_lm.load(str(CHECKPOINT_PATH))
    print(f"  mlx_lm model loaded: {type(mlxlm_model).__name__}\n")

    for seq_len in [64, 256, 512, 1024]:
        input_ids = mx.random.randint(0, 151936, (1, seq_len))
        mx.eval(input_ids)

        # Our model
        our_cache = _make_cache()
        t_ours = _bench(lambda: mx.eval(our_model(input_ids, cache=_make_cache())[0]),
                        n_warmup=3, n_runs=20) * 1000

        # mlx_lm model — just a forward pass without KV cache
        t_mlxlm = _bench(lambda: mx.eval(mlxlm_model(input_ids)[0]),
                         n_warmup=3, n_runs=20) * 1000

        speedup = t_mlxlm / t_ours
        tag = "FASTER" if speedup >= 1.0 else "slower"
        print(f"  seq_len={seq_len:5d}   Ours {t_ours:7.2f} ms   mlx_lm {t_mlxlm:7.2f} ms   {speedup:.2f}x {tag}")


# ---------------------------------------------------------------------------
# Section 4: logprobs_for end-to-end — KV split (ours) vs single-pass
# ---------------------------------------------------------------------------

def bench_logprobs_for(model, tokenizer):
    _header("4. logprobs_for: KV-split (ours) vs single-pass")
    print("  Our approach splits into prompt pass + response pass (reusable cache).")
    print("  Single-pass: one forward over full prompt+response (mlx_lm-style).\n")

    prompt = "What is intellectual curiosity and why does it matter for learning?"
    responses = [
        "Curiosity is the engine of discovery.",
        "It means being genuinely interested in how and why things work, not just memorizing answers.",
        "Intellectual curiosity drives deeper engagement with ideas, questions, and evidence, "
        "making learning more effective and self-sustaining over time, rather than driven by external reward.",
    ]

    for response in responses:
        # Count response tokens
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
        )
        n_resp = len(tokenizer.encode(full_text)) - len(tokenizer.encode(prompt_text))

        t_ours = _bench(lambda: mx.eval(*logprobs_for(model, tokenizer, prompt, response)),
                        n_warmup=2, n_runs=15) * 1000

        # Single-pass baseline
        def _single_pass():
            prompt_text_ = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer.encode(prompt_text_)
            full_ids = tokenizer.encode(full_text)
            resp_start = len(prompt_ids)
            R = len(full_ids) - resp_start

            input_ids = mx.array([full_ids[:-1]])
            logits, _ = model(input_ids, cache=_make_cache())
            mx.eval(logits)
            resp_logits = logits[0, resp_start - 1:]
            lp_matrix = resp_logits - mx.logsumexp(resp_logits, axis=-1, keepdims=True)
            resp_tids = mx.array(full_ids[resp_start:], dtype=mx.int32)
            lps = lp_matrix[mx.arange(R), resp_tids]
            mx.eval(lps)

        t_single = _bench(_single_pass, n_warmup=2, n_runs=15) * 1000

        speedup = t_single / t_ours
        tag = "FASTER" if speedup >= 1.0 else "slower"
        print(f"  R={n_resp:3d} resp tokens   Ours {t_ours:7.2f} ms   Single-pass {t_single:7.2f} ms   {speedup:.2f}x {tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = get_model()
    V = model.embed_tokens.weight.shape[0]
    print(f"Vocab size: {V}\n")

    bench_kernels()
    bench_batch_logprobs(model, tokenizer)
    bench_forward_pass(model, tokenizer)
    bench_logprobs_for(model, tokenizer)

    print("\n\nAll benchmarks complete.")
