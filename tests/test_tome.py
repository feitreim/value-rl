
import json
from tome_client import TomeClient
from transformers import AutoTokenizer
from model.load_weights import CHECKPOINT_PATH

def test_tome_output():
    client = TomeClient("http://localhost:8080")
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_PATH))
    
    prompt = "Why do minor keys sound sad?"
    # Use the same template as train.py
    templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    
    results = client.rollout([templated], group_size=1, max_tokens=10)
    res = results[0]
    comp = res["completions"][0]
    
    print(f"Tokens: {comp['tokens']}")
    print(f"Decoded: {tokenizer.decode(comp['tokens'])}")
    print(f"LogProbs: {comp['log_probs']}")
    print(f"RefLogProbs: {comp['ref_log_probs']}")
    
    # Compute local LPs for comparison
    from model.gwen import get_model, _compute_token_logprobs
    import mlx.core as mx
    model, _ = get_model()
    
    def get_local_lps(ctx_text, target_tokens):
        ctx_ids = tokenizer.encode(ctx_text)
        # Match group_logprobs logic: concatenate and take logits from the end of prompt
        full_ids = mx.array([ctx_ids + target_tokens])
        logits = model(full_ids)
        # response logits start at p_len - 1 to predict the first response token
        ctx_len = len(ctx_ids)
        resp_logits = logits[:, ctx_len - 1 : ctx_len - 1 + len(target_tokens), :]
        lps = _compute_token_logprobs(resp_logits, mx.array([target_tokens]), 1.0)
        return lps.tolist()[0]

    local_lps = get_local_lps(templated, comp['tokens'])
    print(f"LocalLPs: {local_lps}")

    # Check first token specifically
    ctx_ids = tokenizer.encode(templated)
    first_target = [comp['tokens'][0]]
    full_ids = mx.array([ctx_ids + first_target])
    logits = model(full_ids)
    first_token_logits = logits[0, len(ctx_ids)-1, :]
    print(f"First Token Local Logit for {first_target[0]} ({tokenizer.decode(first_target)}): {first_token_logits[first_target[0]].item()}")
    
    # Check max logit locally
    max_idx = mx.argmax(first_token_logits).item()
    print(f"Local Top-1: {max_idx} ({tokenizer.decode([max_idx])}) with logit {first_token_logits[max_idx].item()}")
    
    diffs = [abs(a - b) for a, b in zip(local_lps, comp['ref_log_probs'])]
    print(f"Max Diff: {max(diffs)}")
        
if __name__ == "__main__":
    test_tome_output()
