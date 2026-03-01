"""
grpo.py — GRPO training step
"""

import time
import mlx.core as mx
import mlx.nn as nn

from gwen import _make_cache, batch_logprobs, fused_log_softmax, compute_grpo_loss
from lora import merge_lora, restore_lora
from rubric import Rubric

_ADV_STD_FLOOR = 1e-4
_ADV_CLAMP     = 5.0

def sample_group(model, tokenizer, prompts: list[str], G: int = 8, temperature: float = 0.8, max_tokens: int = 256, 
                 rollout_batch_size: int = 8) -> tuple[list[str], list[str]]:
    """Sample G completions per prompt (batched)."""
    if not prompts or G <= 0: return [], []
    p_ids_list = [tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": p}], 
                  add_generation_prompt=True, tokenize=False)) for p in prompts]
    B, eos, all_completions = len(prompts), tokenizer.eos_token_id, [None] * (len(prompts) * G)
    p_per_chunk = max(1, rollout_batch_size // G)
    for b_s in range(0, B, p_per_chunk):
        b_e = min(b_s + p_per_chunk, B)
        chunk_ids = p_ids_list[b_s:b_e]
        max_p = max(len(p) for p in chunk_ids)
        pad = tokenizer.pad_token_id or eos or 0
        padded = [([pad] * (max_p - len(p)) + p) for p in chunk_ids]
        cache = _make_cache(batch_size=len(chunk_ids))
        logits, cache = model(mx.array(padded), cache=cache)
        cache.advance(max_p)
        curr_batch = len(chunk_ids) * G
        ex_cache = _make_cache(batch_size=curr_batch)
        for l in range(ex_cache.num_layers):
            ex_cache.keys[l] = mx.concatenate([mx.repeat(cache.keys[l][i:i+1], G, axis=0) 
                                               for i in range(len(chunk_ids))], axis=0)
            ex_cache.values[l] = mx.concatenate([mx.repeat(cache.values[l][i:i+1], G, axis=0) 
                                                 for i in range(len(chunk_ids))], axis=0)
        ex_cache.offset = max_p
        l_logits = mx.concatenate([mx.repeat(logits[i:i+1, -1:, :], G, axis=0) for i in range(len(chunk_ids))], axis=0)
        sampled_steps = []
        for _ in range(max_tokens):
            if temperature < 1e-6:
                next_toks = mx.argmax(l_logits[:, 0, :], axis=-1).astype(mx.int32)
            else:
                next_toks = mx.random.categorical(fused_log_softmax(l_logits[:, 0, :], temperature)).astype(mx.int32)
            sampled_steps.append(next_toks)
            l_logits, ex_cache = model(next_toks.reshape(-1, 1), cache=ex_cache)
            ex_cache.advance(1)
        sampled = mx.stack(sampled_steps, axis=1); mx.eval(sampled)
        for i, row in enumerate(sampled.tolist()):
            seq = []
            for t in row:
                if eos is not None and t == eos: break
                seq.append(t)
            all_completions[b_s * G + i] = tokenizer.decode(seq, skip_special_tokens=True)
        mx.clear_cache()
    return [p for p in prompts for _ in range(G)], all_completions

def grpo_step(policy, ref_model, tokenizer, prompts: list[str], rubric: Rubric, optimizer, G: int = 8, 
              beta: float = 0.01, eps: float = 0.2, temperature: float = 0.8, max_tokens: int = 256, 
              rollout_batch_size: int = 8) -> tuple[float, float, dict]:
    B, times = len(prompts), {}
    
    t_start = time.perf_counter()
    _l_saved = merge_lora(policy)
    times["merge_lora"] = time.perf_counter() - t_start
    
    t0 = time.perf_counter()
    all_p, all_c = sample_group(policy, tokenizer, prompts, G, temperature, max_tokens, rollout_batch_size)
    times["rollout"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rewards, details = rubric.score_detailed(all_p, all_c); mx.eval(rewards)
    times["score"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    r = rewards.reshape(B, G)
    adv_mx = ((r - r.mean(axis=-1, keepdims=True)) / mx.maximum(r.std(axis=-1, keepdims=True), _ADV_STD_FLOOR)).reshape(-1)
    adv_mx = mx.clip(adv_mx, -_ADV_CLAMP, _ADV_CLAMP); mx.eval(adv_mx)
    times["advantage"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    # For old and ref lps, we don't need gradients, so evaluate them
    old_lps, _, offsets = batch_logprobs(policy, tokenizer, all_p, all_c)
    mx.eval(old_lps, offsets)
    times["old_lps"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    ref_lps, _, _ = batch_logprobs(ref_model, tokenizer, all_p, all_c)
    mx.eval(ref_lps)
    times["ref_lps"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    restore_lora(_l_saved)
    times["restore_lora"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    
    def loss_fn(m):
        # Optimized differentiable batch_logprobs is already batched internally.
        # However, to save memory during backward, we could further chunk the differentiable pass.
        # For now, let's use the optimized batched version.
        p_lps, _, _ = batch_logprobs(m, tokenizer, all_p, all_c)
        return compute_grpo_loss(p_lps, old_lps, adv_mx, offsets, beta, ref_lps, eps)
    
    loss_val, grads = nn.value_and_grad(policy, loss_fn)(policy)
    optimizer.update(policy, grads); mx.eval(loss_val, policy.parameters())
    times["grad_step"] = time.perf_counter() - t0
    
    times["total"] = time.perf_counter() - t_start
    
    # Build rollout data for logging
    adv_list = adv_mx.tolist()
    reward_list = rewards.tolist()
    groups = []
    for b, prompt in enumerate(prompts):
        completions_data = []
        for g in range(G):
            idx = b * G + g
            completions_data.append({
                "text": all_c[idx], "reward": reward_list[idx], "advantage": adv_list[idx], "scores": details[idx]
            })
        groups.append({"prompt": prompt, "completions": completions_data})

    n_rollout_toks = sum(len(tokenizer.encode(c)) for c in all_c)
    rollout_data = {
        "loss": loss_val.item(), "mean_reward": rewards.mean().item(), "times": times, "groups": groups,
        "metrics": {"rollout_tps": n_rollout_toks / times["rollout"] if times["rollout"] > 0 else 0}
    }
    return rollout_data["loss"], rollout_data["mean_reward"], rollout_data
