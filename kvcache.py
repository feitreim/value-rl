import mlx.core as mx


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, max_seq_len: int, batch_size: int = 1):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        self.keys:   list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        self.offset = 0

    def update(self, k: mx.array, v: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
        """
        Update the cache with new keys and values.
        For efficiency, we use pre-allocated buffers when in decode mode (S_new=1).
        """
        S_new = k.shape[2]
        
        if self.keys[layer_idx] is None:
            # Prefill: initialize with the provided tokens
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        elif S_new == 1:
            # Decode: Efficient update for single token
            # We use a fixed-size increment for the buffer to avoid constant reallocations
            curr_k = self.keys[layer_idx]
            curr_v = self.values[layer_idx]
            
            # current physical length
            B, H, S_phys, D = curr_k.shape
            
            if self.offset + 1 > S_phys:
                # Need to expand buffer. Expand by a decent chunk (e.g., 128)
                new_size = S_phys + 128
                new_k = mx.zeros((B, H, new_size, D), dtype=k.dtype)
                new_v = mx.zeros((B, H, new_size, D), dtype=v.dtype)
                new_k[:, :, :S_phys, :] = curr_k
                new_v[:, :, :S_phys, :] = curr_v
                curr_k = new_k
                curr_v = new_v
            
            # Update the next slot
            curr_k[:, :, self.offset:self.offset+1, :] = k
            curr_v[:, :, self.offset:self.offset+1, :] = v
            
            self.keys[layer_idx] = curr_k
            self.values[layer_idx] = curr_v
        else:
            # Multi-token update (unusual in decode but possible)
            self.keys[layer_idx] = mx.concatenate([self.keys[layer_idx][:, :, :self.offset, :], k], axis=2)
            self.values[layer_idx] = mx.concatenate([self.values[layer_idx][:, :, :self.offset, :], v], axis=2)
            
        return self.keys[layer_idx][:, :, :self.offset + S_new, :], \
               self.values[layer_idx][:, :, :self.offset + S_new, :]

    def advance(self, num_tokens: int) -> None:
        self.offset += num_tokens

    def get_seq_len(self, layer_idx: int) -> int:
        return self.offset

    def snapshot(self) -> "KVCache":
        c = KVCache(self.num_layers, self.num_kv_heads, self.head_dim, self.max_seq_len, self.batch_size)
        # We only snapshot up to the current offset to keep things clean
        c.keys   = [k[:, :, :self.offset, :] if k is not None else None for k in self.keys]
        c.values = [v[:, :, :self.offset, :] if v is not None else None for v in self.values]
        c.offset = self.offset
        return c

    def broadcast_batch(self, n: int) -> "KVCache":
        assert self.keys[0] is None or self.keys[0].shape[0] == 1
        c = KVCache(self.num_layers, self.num_kv_heads, self.head_dim, self.max_seq_len, n)
        # Again, only broadcast the valid portion
        c.keys   = [mx.repeat(k[:, :, :self.offset, :], n, axis=0) if k is not None else None for k in self.keys]
        c.values = [mx.repeat(v[:, :, :self.offset, :], n, axis=0) if v is not None else None for v in self.values]
        c.offset = self.offset
        return c
