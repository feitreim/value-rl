"""
tome_client.py — REST client for Tome inference scheduler
"""

import base64
import json
import time
import httpx
import mlx.core as mx
import numpy as np

class TomeClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=300.0)

    def rollout(self, prompts: list[str], group_size: int = 8, temperature: float = 0.8, max_tokens: int = 256):
        """
        Request rollouts from Tome.
        Returns a list of results, each containing completions with log_probs and ref_log_probs.
        """
        payload = {
            "batch_id": f"rollout-{int(time.time())}",
            "prompts": [{"prompt_id": f"p{i}", "prompt": p} for i, p in enumerate(prompts)],
            "group_size": group_size,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        resp = self.client.post(f"{self.base_url}/v1/grpo/rollout", json=payload)
        resp.raise_for_status()
        return resp.json()["results"]

    def judge(self, rubric: str, items: list[dict], temperature: float = 0.0, max_tokens: int = 16):
        """
        Request judging from Tome.
        items: list of {"item_id": str, "prompt": str}
        """
        payload = {
            "batch_id": f"judge-{int(time.time())}",
            "rubric": rubric,
            "items": items,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        resp = self.client.post(f"{self.base_url}/v1/grpo/judge", json=payload)
        resp.raise_for_status()
        return resp.json()["results"]

    def update_weights(self, model: mx.Module):
        """
        Send LoRA weight updates to Tome.
        """
        from lora import LoRALinear
        updates = []
        
        def _collect_updates(k, m):
            if isinstance(m, LoRALinear):
                # k is like "layers.0.self_attn.q_proj"
                parts = k.split(".")
                layer_idx = int(parts[1])
                param_name = ".".join(parts[2:])
                
                # Convert MLX to bfloat16 bytes
                # MLX bfloat16 doesn't directly map to a standard numpy dtype in some versions,
                # so we use uint16 to preserve the bit patterns.
                a_np = np.array(m.lora_a.astype(mx.uint16))
                b_np = np.array(m.lora_b.astype(mx.uint16))
                
                updates.append({
                    "layer_idx": layer_idx,
                    "param_name": param_name,
                    "lora_a": base64.b64encode(a_np.tobytes()).decode("utf-8"),
                    "lora_b": base64.b64encode(b_np.tobytes()).decode("utf-8"),
                    "shape_a": [int(s) for s in m.lora_a.shape],
                    "shape_b": [int(s) for s in m.lora_b.shape]
                })
        
        model.apply_to_modules(_collect_updates)
        
        if not updates:
            return True
            
        resp = self.client.post(f"{self.base_url}/v1/weights", json={"updates": updates})
        resp.raise_for_status()
        return resp.json()
