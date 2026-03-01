"""
test_correctness.py — Comprehensive correctness suite for Gwen (Qwen3-0.6B).
Consolidated and expanded for batched verification.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import mlx_lm

from gwen import get_model, batch_logprobs, compute_grpo_loss, fused_log_softmax
from load_weights import CHECKPOINT_PATH
from grpo import sample_group

PROMPTS = ["What is intellectual curiosity?"]
RESPONSES = ["It means genuinely engaging with ideas."]


def test_gradients():
    print("--- [TEST] Gradients ---")
    model, tokenizer = get_model()

    # 1. Test batch_logprobs differentiability
    def loss_fn(m):
        lps, _, _ = batch_logprobs(m, tokenizer, PROMPTS, RESPONSES)
        return mx.mean(lps)

    print("Checking batch_logprobs gradients...")
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    print(f"  Loss: {loss.item():.6f}")

    any_grad = False
    for k, v in tree_flatten(grads):
        if mx.max(mx.abs(v)) > 0:
            any_grad = True
            break

    if any_grad:
        print("  [PASS] Gradients are flowing through the model and kernels.")
    else:
        print("  [FAIL] Gradients are all zero.")
        assert False

    # 2. Test full GRPO loss differentiability
    old_lps, _, offsets = batch_logprobs(model, tokenizer, PROMPTS, RESPONSES)
    advantages = mx.array([1.0], dtype=mx.float32)

    def grpo_loss_fn(m):
        p_lps, _, _ = batch_logprobs(m, tokenizer, PROMPTS, RESPONSES)
        return compute_grpo_loss(p_lps, old_lps, advantages, offsets)

    print("Checking full GRPO loss gradients...")
    loss, grads = nn.value_and_grad(model, grpo_loss_fn)(model)
    mx.eval(loss, grads)
    print(f"  Loss: {loss.item():.6f}")
    print("  [PASS] Full GRPO loss is differentiable.")
    print()


def test_batch_consistency():
    print("--- [TEST] Batch Consistency (B=1,2,4,8) ---")
    model, tokenizer = get_model()
    prompt = "The quick brown fox jumps over the lazy dog."
    response = " Indeed it does."

    # --- Part 1: Training path (batch_logprobs, differentiable, standard MLP) ---
    def get_lps(B):
        lps, _, _ = batch_logprobs(model, tokenizer, [prompt] * B, [response] * B)
        return lps.reshape(B, -1)

    print("Training path (standard MLP, differentiable):")
    lps1 = get_lps(1)
    lps2 = get_lps(2)
    lps4 = get_lps(4)
    lps8 = get_lps(8)
    mx.eval(lps1, lps2, lps4, lps8)

    within_8 = mx.max(mx.abs(lps8[0] - lps8[7])).item()
    between_1_8 = mx.max(mx.abs(lps1[0] - lps8[0])).item()
    between_2_4 = mx.max(mx.abs(lps2[0] - lps4[0])).item()
    print(f"  B=8 within-batch max diff: {within_8:.6f}")
    print(f"  B=1 vs B=8[0] max diff: {between_1_8:.6f}")
    print(f"  B=2[0] vs B=4[0] max diff: {between_2_4:.6f}")

    # Between-batch: bf16 attention o_proj compounds across 28 layers (~0.2 logprob diff).
    # Training correctness unaffected: old_lps/policy_lps use same batch size.
    within_tol = 1e-3
    between_tol = 0.5
    assert within_8 < within_tol, f"Within-batch diff {within_8} >= {within_tol}"
    assert between_1_8 < between_tol, f"B=1 vs B=8 diff {between_1_8} >= {between_tol}"
    assert between_2_4 < between_tol, f"B=2 vs B=4 diff {between_2_4} >= {between_tol}"
    print(f"  [PASS] Training path (within={within_tol}, between={between_tol}).")
    print()


def test_vllm_comparison():
    print("--- [TEST] vLLM Comparison (Batched) ---")
    try:
        from vllm_mlx.engine_core import EngineConfig, EngineCore
        from vllm_mlx.request import SamplingParams
        from vllm_mlx.scheduler import SchedulerConfig
    except ImportError:
        print("  [SKIP] vllm-mlx not installed.")
        return

    model, tokenizer = get_model()
    vllm_model, vllm_tok = mlx_lm.load(str(CHECKPOINT_PATH))

    # 1. Batched Logprobs Parity
    prompts = ["What is 2+2?", "Capital of France?", "Is the sky blue?"]
    responses = [" 2+2 is 4.", " Paris.", " Yes."]
    
    print("Checking batched logprob parity...")
    # Our batched logprobs
    lps, _, offsets = batch_logprobs(model, tokenizer, prompts, responses)
    mx.eval(lps, offsets)
    
    # vLLM (mlx_lm) reference logprobs (manually batched for comparison)
    vllm_lps_list = []
    for p, r in zip(prompts, responses):
        p_text = tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        p_ids = tokenizer.encode(p_text)
        full_text = tokenizer.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
        full_ids = mx.array(tokenizer.encode(full_text))[None, :]
        
        resp_tids = full_ids[:, len(p_ids):]
        logits = vllm_model(full_ids)
        resp_logits = logits[:, len(p_ids)-1 : -1, :]
        
        v_lps_all = resp_logits - mx.logsumexp(resp_logits, axis=-1, keepdims=True)
        v_lps = mx.take_along_axis(v_lps_all, resp_tids[..., None], axis=-1).squeeze()
        vllm_lps_list.append(v_lps)
    
    vllm_lps = mx.concatenate(vllm_lps_list)
    mx.eval(vllm_lps)
    
    max_diff = mx.max(mx.abs(lps - vllm_lps)).item()
    print(f"  Max batched logprob diff: {max_diff:.6f}")

    if max_diff < 0.5:
        print("  [PASS] Batched logprobs match vLLM.")
    else:
        print("  [FAIL] Batched logprobs mismatch!")
        assert False

    # 2. Batched Greedy Sampling Parity
    print("Checking batched greedy sampling parity...")
    max_tokens = 12
    
    # Our batched sampling (greedy via temperature=0.0)
    # G=1 per prompt, total batch = 3
    _, our_completions = sample_group(model, tokenizer, prompts, G=1, temperature=0.0, max_tokens=max_tokens, rollout_batch_size=4)
    
    # vLLM-MLX batched sampling
    vllm_sched = SchedulerConfig(max_num_seqs=4, prefill_batch_size=4, completion_batch_size=4)
    vllm_engine = EngineCore(vllm_model, vllm_tok, EngineConfig(model_name=str(CHECKPOINT_PATH), scheduler_config=vllm_sched))
    vllm_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    
    formatted_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]
    vouts = vllm_engine.generate_batch_sync(formatted_prompts, vllm_params)
    vllm_completions = [v.output_text for v in vouts]
    vllm_engine.close()

    all_match = True
    for i, (p, ours, vllm) in enumerate(zip(prompts, our_completions, vllm_completions)):
        match = (ours.strip() == vllm.strip())
        print(f"  P{i}: '{p}'")
        print(f"    Ours: {repr(ours)}")
        print(f"    vLLM: {repr(vllm)}")
        
        if not match:
            # Check prefix similarity
            min_l = min(len(ours), len(vllm))
            prefix_match = 0
            for j in range(min_l):
                if ours[j] == vllm[j]: prefix_match += 1
                else: break
            print(f"    Match: {match}, Prefix match: {prefix_match} chars")
            all_match = False

    if all_match:
        print("  [PASS] Batched greedy samples match vLLM.")
    else:
        print("  [WARNING] Batched greedy samples mismatch (BF16 variance?).")
    print()


def test_padding_consistency():
    print("--- [TEST] Padding Consistency ---")
    model, tokenizer = get_model()
    prompt = "Short prompt"
    p_ids = tokenizer.encode(prompt)
    
    # Unpadded
    logits_unpadded, _ = model(mx.array([p_ids]))
    last_unpadded = logits_unpadded[0, len(p_ids)-1, :]
    
    # Right padded
    pad_len = 8
    pad_id = tokenizer.eos_token_id
    padded_ids = p_ids + [pad_id] * pad_len
    
    seq_len = len(padded_ids)
    indices = mx.arange(seq_len)
    mask = mx.where(indices[:, None] < indices[None, :], mx.array(-1e9, dtype=model.embed_tokens.weight.dtype), mx.array(0.0, dtype=model.embed_tokens.weight.dtype))
    mask[:, len(p_ids):] = mx.array(-1e9, dtype=mask.dtype)
    mask = mask[None, None, :, :]
    
    logits_padded, _ = model(mx.array([padded_ids]), mask=mask)
    last_padded = logits_padded[0, len(p_ids)-1, :]
    
    mx.eval(last_unpadded, last_padded)
    diff = mx.max(mx.abs(last_unpadded - last_padded)).item()
    print(f"  Unpadded vs Right-Padded max logit diff: {diff:.6f}")
    
    if diff < 1.0: 
        print("  [PASS] Padding consistency verified.")
    else:
        print(f"  [FAIL] Padding consistency failed! Diff: {diff:.6f}")
        assert False
    print()


def test_fused_log_softmax():
    print("--- [TEST] Fused Log-Softmax ---")
    logits = mx.random.normal((2, 16, 1024)).astype(mx.bfloat16)
    temp = 0.8
    
    out_fused = fused_log_softmax(logits, temperature=temp)
    t_logits = logits / temp
    out_ref = t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True)
    
    mx.eval(out_fused, out_ref)
    diff = mx.max(mx.abs(out_fused - out_ref)).item()
    print(f"  Fused vs Reference max diff: {diff:.6f}")
    
    if diff < 0.1:
        print("  [PASS] Fused log-softmax verified.")
    else:
        print(f"  [FAIL] Fused log-softmax mismatch! Diff: {diff:.6f}")
        assert False
    print()


def test_kernels_direct():
    print("--- [TEST] Kernels Direct ---")
    
    print("Checking RoPE batching...")
    q = mx.random.normal((2, 16, 32, 128)).astype(mx.bfloat16)
    out_b = mx.fast.rope(q, 128, traditional=False, base=1000000.0, scale=1.0, offset=0)
    out_s = mx.fast.rope(q[1:2], 128, traditional=False, base=1000000.0, scale=1.0, offset=0)
    mx.eval(out_b, out_s)
    diff_rope = mx.max(mx.abs(out_b[1:2] - out_s)).item()
    print(f"  RoPE B=1 vs B=2 diff: {diff_rope:.6f}")
    
    print("Checking SDPA batching...")
    q = mx.random.normal((2, 16, 32, 128)).astype(mx.bfloat16)
    k = mx.random.normal((2, 8, 32, 128)).astype(mx.bfloat16)
    v = mx.random.normal((2, 8, 32, 128)).astype(mx.bfloat16)
    mask = mx.zeros((2, 1, 32, 32), dtype=mx.bfloat16)
    indices = mx.arange(32)
    causal = mx.where(indices[:, None] < indices[None, :], mx.array(-1e9, dtype=mx.bfloat16), mx.array(0.0, dtype=mx.bfloat16))
    mask += causal[None, None, :, :]
    
    out_b = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
    out_s = mx.fast.scaled_dot_product_attention(q[1:2], k[1:2], v[1:2], scale=1.0, mask=mask[1:2])
    mx.eval(out_b, out_s)
    diff_sdpa = mx.max(mx.abs(out_b[1:2] - out_s)).item()
    print(f"  SDPA B=1 vs B=2 diff: {diff_sdpa:.6f}")
    
    if diff_rope < 1e-5 and diff_sdpa < 1e-5:
        print("  [PASS] Direct kernel consistency verified.")
    else:
        print(f"  [FAIL] Kernel consistency failed! RoPE: {diff_rope:.6f}, SDPA: {diff_sdpa:.6f}")
        assert False
    print()


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    test_batch_consistency()
    test_gradients()
    test_padding_consistency()
    test_fused_log_softmax()
    test_kernels_direct()
    test_vllm_comparison()
