"""
grpo.py — GRPO training step

Flow per step:
  1. Rollout: sample G completions per prompt via the current policy
  2. Score: judge all B*G (prompt, completion) pairs with the rubric
  3. Group-normalize: compute advantages within each prompt's G completions
  4. Logprobs: compute policy/ref/old logprobs for the completion tokens
  5. Loss: clipped-ratio GRPO objective + KL penalty
  6. Update: gradient step via AdamW

On-policy by default (one gradient step per rollout batch).
old_lps ≈ policy_lps at rollout time → ratio ≈ 1 initially, eps=0.2 is a safeguard.
"""

import mlx.core as mx
import mlx.nn as nn

from gwen import _make_cache
from gwen_metal import batch_logprobs  # used for old_lps/ref_lps (no grad needed)
from rubric import Rubric


# ---------------------------------------------------------------------------
# Pure-MLX logprob extraction and loss — these support autograd.
# Custom Metal kernels (gwen_metal) don't implement VJP so can't be
# differentiated through. Use these inside loss_fn; use gwen_metal outside
# (for old_lps / ref_lps where no gradient is needed).
# ---------------------------------------------------------------------------

def _logprobs_flat(model, tokenizer, prompts: list[str], responses: list[str]):
    """
    Concatenated per-token logprobs for a list of (prompt, response) pairs.
    Pure MLX — differentiable.

    Returns:
        lps:     (N,) concatenated log-probs (grad-enabled)
        offsets: (B+1,) int32 segment boundaries
    """
    all_lps, offsets = [], [0]
    for prompt, response in zip(prompts, responses):
        messages_full = [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text  = tokenizer.apply_chat_template(messages_full, tokenize=False)
        tokens     = tokenizer.encode(full_text)

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        resp_start = len(tokenizer.encode(prompt_text))

        input_ids      = mx.array([tokens[:-1]])              # (1, N-1)
        logits_3d, _   = model(input_ids, use_metal=False)   # (1, N-1, V)  — VJP-safe
        logits         = logits_3d[0, resp_start - 1:]       # (resp_len, V)
        resp_tids = mx.array(tokens[resp_start:], dtype=mx.int32)

        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        lps = log_probs[mx.arange(logits.shape[0]), resp_tids]
        all_lps.append(lps)
        offsets.append(offsets[-1] + lps.shape[0])

    return mx.concatenate(all_lps), mx.array(offsets, dtype=mx.int32)


def _grpo_loss(
    policy_lps: mx.array,
    old_lps:    mx.array,
    advantages: mx.array,
    offsets:    mx.array,
    beta:       float = 0.01,
    ref_lps:    mx.array | None = None,
    eps:        float = 0.2,
) -> mx.array:
    """
    GRPO loss in pure MLX — differentiable through policy_lps.

    loss = mean_token[ -min(ratio*A, clip(ratio,1-ε,1+ε)*A) ]
         + beta * mean_response[ KL(π || π_ref) ]
    """
    offsets_list = offsets.tolist()
    B = len(offsets_list) - 1

    # build response index for each token
    resp_idx = mx.array(
        [i for i in range(B) for _ in range(offsets_list[i+1] - offsets_list[i])],
        dtype=mx.int32,
    )

    log_ratio = policy_lps - old_lps
    ratio     = mx.exp(log_ratio)
    adv       = advantages[resp_idx]

    unclipped = ratio * adv
    clipped   = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    loss      = mx.mean(-mx.minimum(unclipped, clipped))

    if ref_lps is not None and beta > 0.0:
        kl_tokens  = policy_lps - ref_lps
        kl_per_resp = mx.stack([
            mx.sum(kl_tokens[offsets_list[i]:offsets_list[i+1]])
            for i in range(B)
        ])
        loss = loss + beta * mx.mean(kl_per_resp)

    return loss


def _generate_one(model, tokenizer, prompt_ids: list[int], temperature: float, max_tokens: int) -> str:
    """Decode one completion from a prompt using KV cache."""
    cache = _make_cache()
    logits, cache = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(logits)
    cache.advance(len(prompt_ids))

    eos = tokenizer.eos_token_id
    generated = []
    for _ in range(max_tokens):
        last = logits[0, -1, :]
        next_tok = int(mx.random.categorical(last / temperature).item())
        if next_tok == eos:
            break
        generated.append(next_tok)
        logits, cache = model(mx.array([[next_tok]]), cache=cache)
        mx.eval(logits)
        cache.advance(1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def sample_group(
    model,
    tokenizer,
    prompts: list[str],
    G: int = 8,
    temperature: float = 0.8,
    max_tokens: int = 256,
) -> tuple[list[str], list[str]]:
    """
    Sample G completions per prompt using our KV-cache generation loop.

    The prompt is prefilled once per unique prompt; the KV cache is broadcast
    to batch=G for the decode phase (parallel batched decode).

    Returns:
        all_prompts:     (B*G,) — each prompt repeated G times
        all_completions: (B*G,) — the sampled completions
    """
    all_prompts, all_completions = [], []

    for prompt in prompts:
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

        # Prefill prompt once, then broadcast to G parallel sequences
        prompt_cache = _make_cache()
        prompt_logits, prompt_cache = model(mx.array([prompt_ids]), cache=prompt_cache)
        mx.eval(prompt_logits)
        prompt_cache.advance(len(prompt_ids))

        batch_cache = prompt_cache.broadcast_batch(G)  # (G, kv_heads, S, hd)
        # Decode G sequences in parallel
        last_logits = mx.repeat(prompt_logits[:, -1:, :], G, axis=0)  # (G, 1, V)
        mx.eval(last_logits)

        sequences = [[] for _ in range(G)]
        done = [False] * G
        eos = tokenizer.eos_token_id

        for _ in range(max_tokens):
            # Sample next token for each sequence
            next_toks = mx.random.categorical(last_logits[:, 0, :] / temperature)  # (G,)
            mx.eval(next_toks)
            next_toks_list = next_toks.tolist()

            for i, tok in enumerate(next_toks_list):
                if not done[i]:
                    if int(tok) == eos:
                        done[i] = True
                    else:
                        sequences[i].append(int(tok))

            if all(done):
                break

            # Step all G sequences forward together
            next_input = next_toks.reshape(G, 1)  # (G, 1)
            last_logits, batch_cache = model(next_input, cache=batch_cache)  # (G, 1, V)
            mx.eval(last_logits)
            batch_cache.advance(1)

        for seq in sequences:
            all_prompts.append(prompt)
            all_completions.append(tokenizer.decode(seq, skip_special_tokens=True))

    return all_prompts, all_completions


def grpo_step(
    policy,
    ref_model,
    tokenizer,
    prompts: list[str],
    rubric: Rubric,
    optimizer,
    G: int = 8,
    beta: float = 0.01,
    eps: float = 0.2,
    temperature: float = 0.8,
    max_tokens: int = 256,
) -> tuple[float, float]:
    """
    One GRPO training step.

    Args:
        policy:    the model being trained (MLX nn.Module)
        ref_model: frozen reference model for KL penalty
        tokenizer: shared tokenizer
        prompts:   (B,) list of prompt strings
        rubric:    Rubric instance for scoring
        optimizer: MLX optimizer (e.g. AdamW)
        G:         completions per prompt
        beta:      KL penalty coefficient
        eps:       PPO clip radius (0 to disable)
        temperature: rollout sampling temperature
        max_tokens:  max tokens per completion

    Returns:
        (loss_val, mean_reward) — both Python floats
    """
    B = len(prompts)

    # 1. Rollout
    all_prompts, all_completions = sample_group(
        policy, tokenizer, prompts, G=G, temperature=temperature, max_tokens=max_tokens
    )

    # 2. Score with rubric — (B*G,)
    rewards = rubric.score(all_prompts, all_completions)
    mx.eval(rewards)

    # 3. Group-normalize to get advantages — (B*G,)
    r = rewards.reshape(B, G)
    advantages = (r - r.mean(axis=-1, keepdims=True)) / (r.std(axis=-1, keepdims=True) + 1e-8)
    advantages = advantages.reshape(B * G)
    mx.eval(advantages)

    # 4. Old logprobs (policy at rollout time — treated as constants, no grad)
    old_lps, _, offsets = batch_logprobs(policy, tokenizer, all_prompts, all_completions)
    mx.eval(old_lps, offsets)

    # 5. Ref logprobs (frozen — no grad flows through ref_model)
    ref_lps, _, _ = batch_logprobs(ref_model, tokenizer, all_prompts, all_completions)
    mx.eval(ref_lps)

    # 6. Loss function — gradient flows through policy_lps only.
    #    Uses pure MLX ops (not Metal kernels) so VJP works.
    def loss_fn(model):
        policy_lps, _ = _logprobs_flat(model, tokenizer, all_prompts, all_completions)
        return _grpo_loss(policy_lps, old_lps, advantages, offsets,
                          beta=beta, ref_lps=ref_lps, eps=eps)

    # 7. Gradient step
    loss_and_grad = nn.value_and_grad(policy, loss_fn)
    loss_val, grads = loss_and_grad(policy)
    optimizer.update(policy, grads)
    mx.eval(loss_val, policy.parameters(), optimizer.state)  # one GPU sync, evaluates everything
    del grads  # free gradient memory before returning

    return loss_val.item(), float(rewards.mean().item())
