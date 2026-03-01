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

import time

import mlx.core as mx
import mlx.nn as nn

from gwen import _make_cache
from gwen_metal import batch_logprobs, fused_log_softmax  # used for old_lps/ref_lps (no grad needed)
from lora import merge_lora, restore_lora
from rubric import Rubric

_LOG_RATIO_CLAMP = 6.0
_TOKEN_LOSS_CLAMP = 100.0
_KL_TOKEN_CLAMP = 30.0
_ADV_STD_FLOOR = 1e-4
_ADV_CLAMP = 5.0


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
    if policy_lps.shape[0] == 0:
        return mx.array(0.0, dtype=mx.float32)

    offsets_list = offsets.tolist()
    B = len(offsets_list) - 1

    # build response index for each token
    resp_idx = mx.array(
        [i for i in range(B) for _ in range(offsets_list[i+1] - offsets_list[i])],
        dtype=mx.int32,
    )

    policy_lps = mx.where(mx.isfinite(policy_lps), policy_lps, mx.zeros_like(policy_lps))
    old_lps = mx.where(mx.isfinite(old_lps), old_lps, policy_lps)
    log_ratio = mx.clip(policy_lps - old_lps, -_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
    ratio     = mx.exp(log_ratio)
    adv       = advantages[resp_idx]

    unclipped = ratio * adv
    clipped   = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
    token_loss = -mx.minimum(unclipped, clipped)
    token_loss = mx.where(mx.isfinite(token_loss), token_loss, mx.zeros_like(token_loss))
    token_loss = mx.clip(token_loss, -_TOKEN_LOSS_CLAMP, _TOKEN_LOSS_CLAMP)
    loss = mx.mean(token_loss)

    if ref_lps is not None and beta > 0.0:
        ref_lps = mx.where(mx.isfinite(ref_lps), ref_lps, policy_lps)
        kl_tokens = mx.clip(policy_lps - ref_lps, -_KL_TOKEN_CLAMP, _KL_TOKEN_CLAMP)
        kl_per_resp = []
        for i in range(B):
            start, end = offsets_list[i], offsets_list[i + 1]
            if end > start:
                kl_per_resp.append(mx.sum(kl_tokens[start:end]))
            else:
                kl_per_resp.append(mx.array(0.0, dtype=policy_lps.dtype))
        kl_per_resp = mx.stack(kl_per_resp)
        loss = loss + beta * mx.mean(kl_per_resp)

    return loss


def _generate_one(model, tokenizer, prompt_ids: list[int], temperature: float, max_tokens: int) -> str:
    """Decode one completion from a prompt using KV cache."""
    cache = _make_cache(batch_size=1)
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
    rollout_batch_size: int = 8,
) -> tuple[list[str], list[str]]:
    """
    Sample G completions per prompt using a fully batched KV-cache decode loop.

    Optimized: Prefill each unique prompt once, then broadcast its KV cache
    G times for parallel decoding. This significantly speeds up rollout.
    Processes in chunks of rollout_batch_size to avoid Metal OOM.

    Returns:
        all_prompts:     (B*G,) — each prompt repeated G times
        all_completions: (B*G,) — the sampled completions
    """
    if not prompts or G <= 0:
        return [], []

    # 1. Tokenize each unique prompt once
    prompt_ids_list: list[list[int]] = []
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
        prompt_ids_list.append(tokenizer.encode(text))

    B = len(prompts)
    eos = tokenizer.eos_token_id
    all_completions_flat: list[str | None] = [None] * (B * G)

    # 2. Process prompts in batches to avoid OOM during prefill or broadcast
    prompts_per_chunk = max(1, rollout_batch_size // G)

    for b_start in range(0, B, prompts_per_chunk):
        b_end = min(b_start + prompts_per_chunk, B)
        chunk_prompts_ids = prompt_ids_list[b_start:b_end]
        
        # 3. Prefill unique prompts in this chunk together
        max_p_len = max(len(p) for p in chunk_prompts_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos if eos is not None else 0)
        padded_chunk_ids = [([pad_id] * (max_p_len - len(p)) + p) for p in chunk_prompts_ids]
        
        curr_num_prompts = len(chunk_prompts_ids)
        cache = _make_cache(batch_size=curr_num_prompts)
        logits, cache = model(mx.array(padded_chunk_ids), cache=cache)
        cache.advance(max_p_len)
        
        # 4. Broadcast each sequence in the prefilled batch G times for parallel decode
        curr_batch_size = curr_num_prompts * G
        
        expanded_logits = mx.concatenate([mx.repeat(logits[i:i+1, -1:, :], G, axis=0) for i in range(curr_num_prompts)], axis=0)
        expanded_cache = _make_cache(batch_size=curr_batch_size)
        for l in range(expanded_cache.num_layers):
            expanded_cache.keys[l] = mx.concatenate([mx.repeat(cache.keys[l][i:i+1], G, axis=0) for i in range(curr_num_prompts)], axis=0)
            expanded_cache.values[l] = mx.concatenate([mx.repeat(cache.values[l][i:i+1], G, axis=0) for i in range(curr_num_prompts)], axis=0)
        expanded_cache.offset = max_p_len
        
        # 5. Parallel Decode Loop for (curr_num_prompts * G) sequences
        sampled_steps: list[mx.array] = []
        last_logits = expanded_logits
        
        for _ in range(max_tokens):
            if temperature < 1e-6:
                next_toks = mx.argmax(last_logits[:, 0, :], axis=-1).astype(mx.int32)
            else:
                log_probs = fused_log_softmax(last_logits[:, 0, :], temperature=temperature)
                next_toks = mx.random.categorical(log_probs).astype(mx.int32)
            sampled_steps.append(next_toks)

            last_logits, expanded_cache = model(next_toks.reshape(curr_batch_size, 1), cache=expanded_cache)
            expanded_cache.advance(1)

        sampled = mx.stack(sampled_steps, axis=1) if sampled_steps else mx.zeros((curr_batch_size, 0), dtype=mx.int32)
        # Single GPU sync at the end of the chunk
        mx.eval(sampled)

        # Decode strings
        for seq_idx, row in enumerate(sampled.tolist()):
            seq = []
            for tok in row:
                if eos is not None and tok == eos:
                    break
                seq.append(tok)
            
            flat_idx = b_start * G + seq_idx
            all_completions_flat[flat_idx] = tokenizer.decode(seq, skip_special_tokens=True)
            
        del expanded_cache, cache, logits, last_logits, sampled, sampled_steps
        mx.clear_cache()

    expanded_prompts = [p for p in prompts for _ in range(G)]
    return expanded_prompts, all_completions_flat


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
    rollout_batch_size: int = 8,
) -> tuple[float, float, dict]:
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
        rollout_batch_size: micro-batch size for rollout sampling

    Returns:
        (loss_val, mean_reward, rollout_data)
        rollout_data: dict with groups, loss, mean_reward (step/timestamp added by caller)
    """
    B = len(prompts)

    # Merge LoRA into base weights for fast inference.
    # Fixes two issues vs separate lora_a/lora_b matmuls:
    #   1. Tiny LoRA matmuls per decode token → many Metal kernel launches → slow.
    #   2. lora_a/lora_b are float32; base weights are bfloat16 → mixed dtype in
    #      scaled_dot_product_attention → GPU address fault.
    # Merged W_eff = W + scale*B@A is bfloat16; restore before gradient step.
    _lora_saved = merge_lora(policy)

    # 1. Rollout
    t0_rollout = time.perf_counter()
    all_prompts, all_completions = sample_group(
        policy, tokenizer, prompts, G=G, temperature=temperature, max_tokens=max_tokens,
        rollout_batch_size=rollout_batch_size
    )
    dt_rollout = time.perf_counter() - t0_rollout
    n_rollout_toks = sum(len(tokenizer.encode(c)) for c in all_completions)

    # 2. Score with rubric — (B*G,) rewards + per-criterion breakdown
    t0_score = time.perf_counter()
    rewards, details = rubric.score_detailed(all_prompts, all_completions)
    mx.eval(rewards)
    dt_score = time.perf_counter() - t0_score

    # 3. Group-normalize to get advantages — (B*G,)
    r = rewards.reshape(B, G)
    r_mean = r.mean(axis=-1, keepdims=True)
    r_std = mx.maximum(r.std(axis=-1, keepdims=True), _ADV_STD_FLOOR)
    advantages = (r - r_mean) / r_std
    advantages = mx.clip(advantages, -_ADV_CLAMP, _ADV_CLAMP)
    advantages = advantages.reshape(B * G)
    mx.eval(advantages)

    # 4. Old logprobs (policy at rollout time — treated as constants, no grad)
    t0_old_lps = time.perf_counter()
    old_lps, _, offsets = batch_logprobs(policy, tokenizer, all_prompts, all_completions)
    mx.eval(old_lps, offsets)
    dt_old_lps = time.perf_counter() - t0_old_lps

    # 5. Ref logprobs (frozen — no grad flows through ref_model)
    t0_ref_lps = time.perf_counter()
    ref_lps, _, _ = batch_logprobs(ref_model, tokenizer, all_prompts, all_completions)
    mx.eval(ref_lps)
    dt_ref_lps = time.perf_counter() - t0_ref_lps
    
    n_train_toks = old_lps.shape[0]

    # Restore unmerged weights so gradients flow through lora_a / lora_b.
    restore_lora(_lora_saved)

    # 6. Loss function — gradient flows through policy_lps only.
    #    Uses pure MLX ops (not Metal kernels) so VJP works.
    def loss_fn(model):
        policy_lps, _ = _logprobs_flat(model, tokenizer, all_prompts, all_completions)
        return _grpo_loss(policy_lps, old_lps, advantages, offsets,
                          beta=beta, ref_lps=ref_lps, eps=eps)

    # 7. Gradient step
    t0_grad = time.perf_counter()
    loss_and_grad = nn.value_and_grad(policy, loss_fn)
    loss_val, grads = loss_and_grad(policy)
    optimizer.update(policy, grads)
    mx.eval(loss_val, policy.parameters(), optimizer.state)  # one GPU sync, evaluates everything
    dt_grad = time.perf_counter() - t0_grad
    del grads  # free gradient memory before returning

    loss_f = loss_val.item()
    mean_reward_f = float(rewards.mean().item())

    # Build rollout data for logging (step/timestamp added by caller)
    adv_list = advantages.tolist()
    reward_list = rewards.tolist()
    groups = []
    for b, prompt in enumerate(prompts):
        completions_data = []
        for g in range(G):
            flat_idx = b * G + g
            completions_data.append({
                "text": all_completions[flat_idx],
                "reward": reward_list[flat_idx],
                "advantage": adv_list[flat_idx],
                "scores": details[flat_idx],
            })
        groups.append({"prompt": prompt, "completions": completions_data})

    rollout_data = {
        "loss": loss_f, 
        "mean_reward": mean_reward_f, 
        "groups": groups,
        "metrics": {
            "rollout_tps": n_rollout_toks / dt_rollout if dt_rollout > 0 else 0,
            "train_tps": n_train_toks / (dt_old_lps + dt_ref_lps + dt_grad) if (dt_old_lps + dt_ref_lps + dt_grad) > 0 else 0,
            "dt_rollout": dt_rollout,
            "dt_score": dt_score,
            "dt_old_lps": dt_old_lps,
            "dt_ref_lps": dt_ref_lps,
            "dt_grad": dt_grad,
            "n_rollout_toks": n_rollout_toks,
            "n_train_toks": n_train_toks,
        }
    }
    return loss_f, mean_reward_f, rollout_data
