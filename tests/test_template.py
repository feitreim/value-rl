
import mlx.core as mx
from model.gwen import get_model, _compute_token_logprobs

def test_template_mismatch():
    model, tokenizer = get_model()
    prompt = "Why do minor keys sound sad?"
    completion = "Minor keys have a smaller interval between the root and the third."
    
    # Template 1 (Single)
    t1 = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    # Template 2 (Double - what happens if Tome re-templates)
    t2 = tokenizer.apply_chat_template([{"role": "user", "content": t1}], add_generation_prompt=True, tokenize=False)
    
    print(f"T1: {t1!r}")
    print(f"T2: {t2!r}")
    
    comp_ids = tokenizer.encode(completion)
    t1_ids = tokenizer.encode(t1)
    t2_ids = tokenizer.encode(t2)
    
    def get_lps(ctx_ids, target_ids):
        full_ids = mx.array([ctx_ids + target_ids])
        logits = model(full_ids)
        # logits[0, i, :] is prediction for full_ids[i+1]
        # prediction for target_ids[0] (at index len(ctx_ids)) is at logits[0, len(ctx_ids)-1, :]
        ctx_len = len(ctx_ids)
        resp_logits = logits[:, ctx_len - 1 : ctx_len - 1 + len(target_ids), :]
        lps = _compute_token_logprobs(resp_logits, mx.array([target_ids]), 1.0)
        return lps.sum().item()

    lp1 = get_lps(t1_ids, comp_ids)
    lp2 = get_lps(t2_ids, comp_ids)
    
    print(f"LP1 (Single): {lp1:.4f}")
    print(f"LP2 (Double): {lp2:.4f}")
    print(f"Diff (delta): {lp1 - lp2:.4f}")
    print(f"KL contribution exp(lp2-lp1) - 1 - (lp2-lp1): {mx.exp(lp1-lp2).item() + (lp2-lp1) - 1:.4f}")

if __name__ == "__main__":
    test_template_mismatch()
