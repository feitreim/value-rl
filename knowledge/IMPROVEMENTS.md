# Pipeline Improvements

Analysis of `train.py`, `grpo.py`, `tome_client.py`, `Tome/mlx-impl/node.py`,
`model/gwen.py`, `model/lora.py`, `rubric.py`, `model/model.py`.

---

## Bugs

### 1. `restore_lora` timing uses wrong timestamp (grpo.py:82)

```python
t0 = time.perf_counter()          # line 75 — set for advantage timing
...
times["advantage"] = ...          # line 79
restore_lora(_l_saved)
times["restore_lora"] = time.perf_counter() - t0  # line 82 — uses stale t0!
```

`times["restore_lora"]` includes the advantage computation time. Should use a
fresh timestamp before `restore_lora()`.

### 2. `merge_lora` / `restore_lora` cycle is dead code (grpo.py:34,81)

The merge happens at line 34 before the rollout, and restore at line 81 after
scoring. But all inference between those lines is done remotely via Tome — the
local merged weights are never used. The merge also creates a correctness
hazard: during the merged window the local forward would double-count LoRA
(`base + delta + scale*(xA^T)B^T`), though this path is never hit.

Tome receives weights via `update_weights()` which sends raw LoRA A/B matrices
and reconstructs `base + delta` on its side. The merge/restore cycle is a
leftover from local-rollout days and wastes time computing `delta = scale*(B@A)`
for every LoRA layer then undoing it.

**Fix**: Remove `merge_lora` / `restore_lora` from `grpo_step`.

### 3. `compute_grpo_loss` may fight `@mx.compile` (gwen.py:~56)

`compute_grpo_loss` is decorated with `@mx.compile` but calls
`offsets.tolist()` inside the function body. `mx.compile` traces the compute
graph and replays it — calling `.tolist()` forces synchronous evaluation during
tracing and the resulting Python list becomes a compile-time constant. Since
`offsets` changes shape/values every call, the function is re-traced every time,
negating the compile benefit and adding tracing overhead.

**Fix**: Either remove `@mx.compile` or restructure to avoid `.tolist()` inside
the compiled region (pass offsets as a regular array and use `mx.where` / scatter
instead of Python list indexing).

### 4. Block allocator leak on weight update (node.py:155)

```python
self.prefix_cache = PrefixCache(self.allocator)
```

When the policy prefix cache is replaced after a weight update, the old
`PrefixCache` (and its `PrefixNode` tree) is discarded. But the blocks
referenced by those nodes are never explicitly released back to the
`BlockAllocator`. Since `BlockAllocator` uses refcounting, these blocks remain
allocated until their refcount drops to zero — which may never happen if the old
`PrefixCache` was the only holder. This leaks blocks on every weight update
(i.e. every training step).

**Fix**: Add a `clear()` method to `PrefixCache` that walks the trie and
releases all blocks, and call it before replacement.

---

## Performance — Critical

### 5. Double forward pass per group (grpo.py:117,122) — ~2x grad time

Every group triggers **two** full forward passes through the 28-layer model:

1. Line 117: `group_logprobs(policy, ...)` with `stop_gradient` — just for KL
   reporting
2. Line 122: `nn.value_and_grad(policy, group_loss_fn)(policy)` — calls
   `group_logprobs` again inside `group_loss_fn`

The KL is already computed inside `compute_grpo_loss` (the `beta * mean(kl)`
term). The reporting-only forward pass is redundant.

**Fix**: Return the KL from `compute_grpo_loss` as a second output, or compute
it from the logprobs that `group_loss_fn` already produces. This halves the
number of local forward passes from 2B to B (e.g. 8 → 4 for B=4).

### 6. Prompt tokens repeated G times in forward pass (gwen.py:group_logprobs)

```python
full_ids = mx.concatenate([mx.repeat(p_ids_arr, G_p, axis=0), r_ids], axis=1)
logits = model(full_ids)
```

For G=8, the same prompt prefix (~100-200 tokens) is processed 8 times in every
forward pass. Without KV caching in the training model, this is pure waste.

**Fix (easy)**: Run the prompt once, cache the hidden state, then run only the
response tokens with the cached prefix. This requires adding a simple prefix-
cache mechanism to the training model (or splitting the forward into prefix +
suffix). Reduces FLOPs roughly by `prompt_len / (prompt_len + max_resp_len)`,
which is ~30-50% for typical ratios.

**Fix (harder)**: Use activation checkpointing on the prefix to save memory.

---

## Performance — Medium

### 7. Sequential group processing (grpo.py:111)

Groups (B=4) are processed sequentially with `mx.eval` + `mx.clear_cache()`
between each. This prevents any cross-group GPU overlap and forces
synchronization.

The loop structure is intentional (gradient accumulation with memory clearing),
but the `mx.clear_cache()` is aggressive. MLX's allocator already reuses freed
buffers; `clear_cache` forces it to release memory back to the system pool.

**Consider**: Removing `mx.clear_cache()` and letting MLX manage memory. If OOM
is the concern, try processing 2 groups at a time instead of 1.

### 8. Token re-encoding for rollout TPS metric (grpo.py:166)

```python
n_rollout_toks = sum(len(tokenizer.encode(c)) for c in all_c)
```

This re-encodes every completion string back to tokens just to count them. The
token counts are already available from the Tome response (`lengths` list at
line 65, or `len(comp["tokens"])` from each completion).

**Fix**: `n_rollout_toks = sum(lengths)`.

### 9. Redundant `tokenizer.encode` per group (grpo.py:94)

```python
p_ids_arr = mx.array([tokenizer.encode(p_text)])
```

Every group re-encodes the prompt text to tokens. Tome already tokenized these
prompts and returned completions relative to them. The prompt token IDs could be
extracted from the Tome rollout response (prompt tokens are sent as part of the
gRPC request, and the response could include them), or cached from the initial
`apply_chat_template` call.

**Fix**: Tokenize once outside the group loop and cache the results.

### 10. `_sample_token` uses full argsort for top-k (node.py:949)

```python
top_indices = mx.argsort(scaled_logits)[-top_k:]
```

`mx.argsort` is O(V log V) for V=151936. For top-k=20, this is wasteful.

**Fix**: Use `mx.argpartition(scaled_logits, kth=V-top_k)[-top_k:]` which is
O(V) average case, or a single-pass custom kernel.

### 11. Reference logprobs processed per-prompt sequentially (node.py:420)

`_process_ref_logprobs_batched` iterates over prompts one at a time (line 420
`for p_idx, ...`), prefilling each prompt individually on the reference model.
For B=4 prompts, that's 4 sequential prefill passes.

**Fix**: Batch the reference prompt prefills like the rollout prefill does
(chunks of 8 prompts at a time). The completion logprob phase is already batched
per prompt (forks G copies), but the prompt-level loop is serial.

---

## Performance — Minor / Nice-to-Have

### 12. Score cache grows without bound (rubric.py:82)

`self._cache` is never evicted. For 500 steps × 32 completions × 3 criteria =
48k entries, memory is fine. For longer runs or larger batches, consider an LRU
cache. Not urgent.

### 13. Judge prefill hardcoded to chunks of 8 (node.py:263)

The judge's batched prefill uses `for start_idx in range(0, num_items, 8)`.
This magic number could be tuned based on `max_judge_batch_size` or sequence
lengths.

### 14. Synchronous weight sync every step (grpo.py:148)

`tome_client.update_weights(policy)` is a blocking HTTP POST after every
training step. For rank=8, the payload is small (~100KB), but it's still a
serial sync point.

**Consider**: Making this async — start the weight push while beginning the next
step's prompt sampling. The rollout doesn't start until after the POST returns
anyway, so the overlap opportunity is limited, but it would hide network
latency.

### 15. No gradient checkpointing (model/model.py)

The training model stores all intermediate activations for backprop through 28
layers. For sequences of ~400 tokens × 8 per group, peak memory is significant.
`mx.checkpoint` on every N-th layer would trade ~33% more compute for ~50% less
activation memory, allowing larger batches or longer sequences.

---

## Summary — Priority Order

| # | Type | Impact | Effort |
|---|------|--------|--------|
| 5 | Double forward pass | ~2x grad time | Low — return KL from loss fn |
| 2 | Dead merge/restore | Wasted compute + correctness risk | Trivial — delete 2 lines |
| 6 | Repeated prompt tokens | ~30-50% wasted FLOPs | Medium — split forward |
| 4 | Block allocator leak | Memory leak per step | Low — add PrefixCache.clear() |
| 3 | Bad mx.compile usage | Re-trace overhead every call | Low — remove decorator |
| 1 | Wrong timing | Incorrect profiling data | Trivial |
| 8 | Re-encoding tokens | Wasted CPU | Trivial |
| 9 | Redundant tokenizer.encode | Wasted CPU per group | Low |
| 10 | argsort for top-k | O(V log V) vs O(V) | Low |
| 7 | Sequential groups | No GPU overlap | Needs profiling |
| 11 | Serial ref logprob prefill | Serial GPU phases | Medium |
