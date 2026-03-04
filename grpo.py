"""
grpo.py — GRPO training step with gradient accumulation
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from model.gwen import group_logprobs, compute_grpo_loss
from model.lora import merge_lora, restore_lora
from rubric import Rubric

_ADV_STD_FLOOR = 1e-2
_ADV_CLAMP     = 5.0
_GRAD_CLIP_NORM = 1.0


def _clip_grad_norm(grads, max_norm: float):
    flat = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
    norm = mx.sqrt(sum(mx.sum(g * g) for g in flat))
    scale = mx.minimum(mx.array(max_norm, dtype=flat[0].dtype) / (norm.astype(flat[0].dtype) + 1e-6), mx.array(1.0, dtype=flat[0].dtype))
    return tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads), norm


def grpo_step(policy, tokenizer, prompts: list[str], rubric: Rubric, optimizer, G: int = 8,
              beta: float = 0.1, eps: float = 0.2, temperature: float = 0.8, max_tokens: int = 256,
              tome_client=None) -> tuple[float, float, dict]:
    """GRPO step using Tome for rollouts and judging."""
    assert tome_client is not None, "Tome client is required for rollouts and judging"
    B, times = len(prompts), {"tome_weight_update": 0.0}
    
    t_start = time.perf_counter()
    _l_saved = merge_lora(policy)
    times["merge_lora"] = time.perf_counter() - t_start
    
    t0 = time.perf_counter()
    prompt_texts = []
    for p in prompts:
        txt = p["prompt"] if isinstance(p, dict) else p
        templated = tokenizer.apply_chat_template([{"role": "user", "content": txt}], add_generation_prompt=True, tokenize=False)
        prompt_texts.append(templated)
            
    results = tome_client.rollout(prompt_texts, group_size=G, temperature=temperature, max_tokens=max_tokens)
    times["rollout"] = time.perf_counter() - t0
    
    # Sort results by prompt_id ("p0", "p1", ...) to ensure they match prompts order
    results = sorted(results, key=lambda x: int(x["prompt_id"][1:]))
    
    all_p = []
    all_c = []
    all_t = [] # Raw tokens from Tome
    old_lps_list = []
    ref_lps_list = []
    lengths = []
    for res in results:
        idx = int(res["prompt_id"][1:])
        prompt = prompt_texts[idx]
        for comp in res["completions"]:
            all_p.append(prompt)
            all_c.append(tokenizer.decode(comp["tokens"], skip_special_tokens=True))
            all_t.append(comp["tokens"])
            old_lps_list.append(mx.array(comp["log_probs"]))
            ref_lps_list.append(mx.array(comp["ref_log_probs"]))
            lengths.append(len(comp["tokens"]))
    
    old_lps = mx.stop_gradient(mx.concatenate(old_lps_list))
    ref_lps = mx.stop_gradient(mx.concatenate(ref_lps_list))
    offsets = mx.array([0] + [sum(lengths[:i+1]) for i in range(len(lengths))], dtype=mx.int32)
    
    t0 = time.perf_counter()
    rewards, details = rubric.score_detailed(all_p, all_c); mx.eval(rewards)
    times["score"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    r = rewards.reshape(B, G)
    adv_mx = ((r - r.mean(axis=-1, keepdims=True)) / mx.maximum(r.std(axis=-1, keepdims=True), _ADV_STD_FLOOR)).reshape(-1)
    adv_mx = mx.clip(adv_mx, -_ADV_CLAMP, _ADV_CLAMP); mx.eval(adv_mx)
    times["advantage"] = time.perf_counter() - t0
    
    restore_lora(_l_saved)
    times["restore_lora"] = time.perf_counter() - t0
    
    t_grad = time.perf_counter()
    total_loss = 0.0
    total_kl = 0.0
    accum_grads = None
    
    offsets_list = offsets.tolist()
    groups_data = []
    for i in range(B):
        start, end = i * G, (i + 1) * G
        p_text = prompt_texts[i]
        p_ids_arr = mx.array([tokenizer.encode(p_text)])
        
        r_ids_list = [all_t[j] for j in range(start, end)]
        max_r = max(len(toks) for toks in r_ids_list)
        
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padded_r = [toks + [pad_id] * (max_r - len(toks)) for toks in r_ids_list]
        r_ids = mx.array(padded_r)
        actual_lens = [len(toks) for toks in r_ids_list]
        
        g_old_lps = old_lps[offsets_list[start]:offsets_list[end]].reshape(-1)
        g_ref_lps = ref_lps[offsets_list[start]:offsets_list[end]].reshape(-1)
        g_adv = adv_mx[start:end]
        g_offsets = mx.array([o - offsets_list[start] for o in offsets_list[start:end+1]], dtype=mx.int32)
        
        groups_data.append((p_ids_arr, r_ids, actual_lens, g_old_lps, g_ref_lps, g_adv, g_offsets))

    for i, (p_ids_arr, r_ids, actual_lens, g_old_lps, g_ref_lps, g_adv, g_offsets) in enumerate(groups_data):
        def group_loss_fn(model):
            lps = group_logprobs(model, p_ids_arr, r_ids, temperature=temperature, actual_lengths=actual_lens)
            return compute_grpo_loss(lps, g_old_lps, g_adv, g_offsets, beta=beta, ref_lps=g_ref_lps, eps=eps)
        
        # We also want to track KL for reporting
        lps_no_grad = mx.stop_gradient(group_logprobs(policy, p_ids_arr, r_ids, temperature=temperature, actual_lengths=actual_lens))
        delta = lps_no_grad - g_ref_lps
        kl_tokens = mx.exp(-delta) + delta - 1
        kl = mx.mean(mx.stack([mx.sum(kl_tokens[int(g_offsets[j]):int(g_offsets[j+1])]) for j in range(len(g_offsets)-1)]))
        
        loss, grads = nn.value_and_grad(policy, group_loss_fn)(policy)
        
        loss = loss / B
        kl = kl / B
        grads = tree_map(lambda x: x / B, grads)
        
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda x, y: x + y, accum_grads, grads)
        
        total_loss += loss.item()
        total_kl += kl.item()
        
        mx.eval(total_loss, total_kl, accum_grads)
        mx.clear_cache()

    accum_grads, grad_norm = _clip_grad_norm(accum_grads, _GRAD_CLIP_NORM)
    mx.eval(accum_grads, grad_norm)
    grad_norm_val = grad_norm.item()

    times["grad_step"] = time.perf_counter() - t_grad

    optimizer.update(policy, accum_grads); mx.eval(policy.parameters())
    
    t_w = time.perf_counter()
    tome_client.update_weights(policy)
    times["tome_weight_update"] = time.perf_counter() - t_w
    
    times["total"] = time.perf_counter() - t_start
    
    # Build rollout data for logging
    adv_list = adv_mx.tolist()
    reward_list = rewards.tolist()
    groups_out = []
    for b, prompt in enumerate(prompts):
        completions_data = []
        for g in range(G):
            idx = b * G + g
            completions_data.append({
                "text": all_c[idx], "reward": reward_list[idx], "advantage": adv_list[idx], "scores": details[idx]
            })
        groups_out.append({"prompt": prompt, "completions": completions_data})

    n_rollout_toks = sum(len(tokenizer.encode(c)) for c in all_c)
    rollout_data = {
        "loss": total_loss, "mean_reward": rewards.mean().item(), "mean_kl": total_kl, "times": times, "groups": groups_out,
        "metrics": {"rollout_tps": n_rollout_toks / times["rollout"] if times["rollout"] > 0 else 0, "grad_norm": grad_norm_val}
    }
    return rollout_data["loss"], rollout_data["mean_reward"], rollout_data
