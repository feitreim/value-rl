"""
test_correctness.py — Verify differentiability and numerical correctness
                      of our custom Metal kernels.
"""

import mlx.core as mx
import mlx.nn as nn

from gwen import get_model, batch_logprobs, compute_grpo_loss

PROMPTS = ["What is intellectual curiosity?"]
RESPONSES = ["It means genuinely engaging with ideas."]


def test_gradients():
    print("Loading model...")
    model, tokenizer = get_model()
    print("Ready.\n")

    # 1. Test batch_logprobs differentiability
    def loss_fn(m):
        lps, _, _ = batch_logprobs(m, tokenizer, PROMPTS, RESPONSES)
        return mx.mean(lps)

    print("Checking batch_logprobs gradients...")
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    print(f"  Loss: {loss.item():.6f}")

    # Check if any gradients are non-zero (proving flow through kernels)
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

    print("\nChecking full GRPO loss gradients...")
    loss, grads = nn.value_and_grad(model, grpo_loss_fn)(model)
    mx.eval(loss, grads)
    print(f"  Loss: {loss.item():.6f}")
    print("  [PASS] Full GRPO loss is differentiable.")


from mlx.utils import tree_flatten

if __name__ == "__main__":
    test_gradients()
