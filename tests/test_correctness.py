"""
test_correctness.py — Comprehensive correctness suite for Gwen (Qwen3-0.6B).
Focuses on the backward pass and logprob extraction.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from model.gwen import get_model, group_logprobs, compute_grpo_loss, fused_log_softmax

PROMPTS = ["What is intellectual curiosity?"]
RESPONSES = ["It means genuinely engaging with ideas."]


def test_gradients():
    print("--- [TEST] Gradients ---")
    model, tokenizer = get_model()

    p_text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True, tokenize=False)
    p_ids = mx.array([tokenizer.encode(p_text)])
    
    r_text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[0]}, {"role": "assistant", "content": RESPONSES[0]}], tokenize=False)
    r_ids_full = tokenizer.encode(r_text)
    r_ids = mx.array([r_ids_full[p_ids.shape[1]:]])
    actual_lens = [r_ids.shape[1]]

    # 1. Test group_logprobs differentiability
    def loss_fn(m):
        lps = group_logprobs(m, p_ids, r_ids, actual_lengths=actual_lens)
        return mx.mean(lps)

    print("Checking group_logprobs gradients...")
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    print(f"  Loss: {loss.item():.6f}")

    any_grad = False
    for k, v in tree_flatten(grads):
        if mx.max(mx.abs(v)) > 0:
            any_grad = True
            break

    if any_grad:
        print("  [PASS] Gradients are flowing through the model.")
    else:
        print("  [FAIL] Gradients are all zero.")
        assert False

    # 2. Test full GRPO loss differentiability
    old_lps = mx.stop_gradient(group_logprobs(model, p_ids, r_ids, actual_lengths=actual_lens))
    advantages = mx.array([1.0], dtype=mx.float32)
    offsets = mx.array([0, len(old_lps)], dtype=mx.int32)

    def grpo_loss_fn(m):
        p_lps = group_logprobs(m, p_ids, r_ids, actual_lengths=actual_lens)
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

    p_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    p_ids = mx.array([tokenizer.encode(p_text)])
    
    r_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False)
    r_ids_full = tokenizer.encode(r_text)
    r_ids_single = r_ids_full[p_ids.shape[1]:]
    
    def get_lps(B):
        r_ids = mx.array([r_ids_single] * B)
        lps = group_logprobs(model, p_ids, r_ids, actual_lengths=[len(r_ids_single)] * B)
        return lps.reshape(B, -1)

    lps1 = get_lps(1)
    lps2 = get_lps(2)
    lps4 = get_lps(4)
    lps8 = get_lps(8)
    mx.eval(lps1, lps2, lps4, lps8)

    within_8 = mx.max(mx.abs(lps8[0] - lps8[7])).item()
    between_1_8 = mx.max(mx.abs(lps1[0] - lps8[0])).item()
    print(f"  B=8 within-batch max diff: {within_8:.6f}")
    print(f"  B=1 vs B=8[0] max diff: {between_1_8:.6f}")

    within_tol = 1e-3
    between_tol = 0.5
    assert within_8 < within_tol, f"Within-batch diff {within_8} >= {within_tol}"
    assert between_1_8 < between_tol, f"B=1 vs B=8 diff {between_1_8} >= {between_tol}"
    print(f"  [PASS] Batch consistency verified.")
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


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    test_batch_consistency()
    test_gradients()
    test_fused_log_softmax()
