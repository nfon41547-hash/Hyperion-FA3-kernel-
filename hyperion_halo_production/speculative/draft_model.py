"""
draft_model.py – Draft model interface for speculative decoding (7B class).

The DraftModel wraps a small auto-regressive language model and exposes
``draft_tokens()``, which generates a configurable number of *draft* tokens
for each sequence in a batch.  These tokens are later verified by the target
model via the Verifier.

Design notes
------------
* The class is backend-agnostic: any callable ``forward(input_ids) → logits``
  can be used (real transformer, stub, or mock for testing).
* The ``DraftModelConfig`` dataclass mirrors the 7B LLaMA-style architecture
  defaults that pair well with a 70B target model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DraftModelConfig:
    """Configuration for the 7B draft model."""

    model_name: str = "meta-llama/Llama-2-7b-hf"
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    hidden_size: int = 4096
    vocab_size: int = 32000
    max_seq_len: int = 4096
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    # Speculative-specific
    draft_steps: int = 4        # tokens to draft per target step
    temperature: float = 1.0
    top_k: int = 0              # 0 = disabled
    top_p: float = 1.0


class DraftModel:
    """
    Lightweight wrapper around a 7B-class auto-regressive model for
    speculative decoding.

    Parameters
    ----------
    config  : DraftModelConfig
    forward_fn : callable ``(input_ids: Tensor[B, T]) → logits: Tensor[B, T, V]``
        If None a random-logit stub is used (useful for testing without a GPU
        or loaded weights).
    """

    def __init__(
        self,
        config: Optional[DraftModelConfig] = None,
        forward_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config or DraftModelConfig()
        self._forward_fn = forward_fn
        self._call_count = 0
        self._total_tokens_drafted = 0

    # ------------------------------------------------------------------
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one forward pass; falls back to random logits when no model."""
        if self._forward_fn is not None:
            return self._forward_fn(input_ids)
        # Stub: return uniform random logits
        B, T = input_ids.shape
        return torch.randn(
            B, T, self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_draft_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate *num_draft_steps* draft tokens for each sequence in the batch.

        Parameters
        ----------
        input_ids       : [B, T] int64  – current context token ids
        num_draft_steps : int or None   – overrides config.draft_steps

        Returns
        -------
        draft_ids   : [B, num_draft_steps]   int64  – sampled draft token ids
        draft_probs : [B, num_draft_steps, V] float32 – softmax probabilities
        """
        steps = num_draft_steps or self.config.draft_steps
        B = input_ids.shape[0]
        V = self.config.vocab_size
        device = input_ids.device

        all_ids = input_ids.clone()
        draft_ids_list: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []

        for _ in range(steps):
            logits = self._forward(all_ids)   # [B, T, V]
            next_logits = logits[:, -1, :]    # [B, V]

            probs = F.softmax(next_logits / max(self.config.temperature, 1e-8), dim=-1)

            if self.config.top_k > 0:
                probs = self._top_k_filter(probs, self.config.top_k)
            if self.config.top_p < 1.0:
                probs = self._top_p_filter(probs, self.config.top_p)

            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            draft_ids_list.append(next_token.squeeze(-1))          # [B]
            draft_probs_list.append(probs)                         # [B, V]

            all_ids = torch.cat([all_ids, next_token], dim=-1)

        self._call_count += 1
        self._total_tokens_drafted += B * steps

        draft_ids = torch.stack(draft_ids_list, dim=1)     # [B, steps]
        draft_probs = torch.stack(draft_probs_list, dim=1) # [B, steps, V]
        return draft_ids, draft_probs

    # ------------------------------------------------------------------
    @staticmethod
    def _top_k_filter(probs: torch.Tensor, k: int) -> torch.Tensor:
        top_k_vals, _ = torch.topk(probs, k, dim=-1)
        threshold = top_k_vals[:, -1:].expand_as(probs)
        filtered = probs.masked_fill(probs < threshold, 0.0)
        return filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = cumulative - sorted_probs > p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        restored = torch.zeros_like(probs)
        restored.scatter_(-1, sorted_idx, sorted_probs)
        return restored / restored.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens_drafted": self._total_tokens_drafted,
        }
