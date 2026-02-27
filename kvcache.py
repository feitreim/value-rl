import mlx.core as mx


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, max_seq_len: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.keys:   list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        self.offset = 0

    def update(self, k: mx.array, v: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx]   = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx]   = mx.concatenate([self.keys[layer_idx],   k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx], v], axis=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def advance(self, num_tokens: int) -> None:
        self.offset += num_tokens

    def get_seq_len(self, layer_idx: int) -> int:
        if self.keys[layer_idx] is None:
            return 0
        return self.keys[layer_idx].shape[2]

    def snapshot(self) -> "KVCache":
        """
        Shallow-copy of the current cache state.

        Safe to use as a base for multiple independent decode passes: each call to
        cache.update() on the copy creates new tensors via mx.concatenate, leaving
        this snapshot's lists unchanged.
        """
        c = KVCache(self.num_layers, self.num_kv_heads, self.head_dim, self.max_seq_len)
        c.keys   = list(self.keys)    # new list, same array objects
        c.values = list(self.values)
        c.offset = self.offset
        return c

    def broadcast_batch(self, n: int) -> "KVCache":
        """
        Expand the batch dimension from 1 to n by repeating all cached tensors.

        Use this to fan out a single prompt cache into n independent completion
        caches for batched parallel decoding.

        Keys/values go from (1, num_kv_heads, S, head_dim)
                         to  (n, num_kv_heads, S, head_dim).
        """
        assert all(
            k is None or k.shape[0] == 1 for k in self.keys
        ), "broadcast_batch requires batch=1 cache"
        c = KVCache(self.num_layers, self.num_kv_heads, self.head_dim, self.max_seq_len)
        c.keys   = [mx.repeat(k, n, axis=0) if k is not None else None for k in self.keys]
        c.values = [mx.repeat(v, n, axis=0) if v is not None else None for v in self.values]
        c.offset = self.offset
        return c
