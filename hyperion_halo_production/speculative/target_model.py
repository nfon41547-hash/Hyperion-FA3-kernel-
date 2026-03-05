"""
target_model.py – Target model interface for speculative decoding (70B/405B).

The TargetModel wraps a large auto-regressive language model and exposes
``score_tokens()``, which runs a single forward pass over a context extended
by the draft tokens and returns per-position logit distributions used by the
Verifier to accept or reject draft tokens.

Design notes
------------
* Like DraftModel, this class is backend-agnostic.
* TargetModelConfig defaults mirror LLaMA-3 70B / 405B shapes.
* A ``tensor_parallel_size`` field is included for documentation purposes;
  actual TP is expected to be handled by the backend (vLLM, DeepSpeed, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TargetModelConfig:
    """Configuration for the 70B / 405B target model."""

    model_name: str = "meta-llama/Meta-Llama-3-70B"
    num_layers: int = 80
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 8192
    vocab_size: int = 128256
    max_seq_len: int = 8192
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    tensor_parallel_size: int = 1


class TargetModel:
    """
    Large-model wrapper for speculative decoding verification.

    Parameters
    ----------
    config     : TargetModelConfig
    forward_fn : callable ``(input_ids: Tensor[B, T]) → logits: Tensor[B, T, V]``
        If None a random-logit stub is used.
    """

    def __init__(
        self,
        config: Optional[TargetModelConfig] = None,
        forward_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config or TargetModelConfig()
        self._forward_fn = forward_fn
        self._call_count = 0
        self._total_tokens_scored = 0

    # ------------------------------------------------------------------
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._forward_fn is not None:
            return self._forward_fn(input_ids)
        B, T = input_ids.shape
        return torch.randn(
            B, T, self.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_tokens(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score a batch of (context + draft) sequences.

        Parameters
        ----------
        input_ids  : [B, T]            – original context
        draft_ids  : [B, draft_steps]  – draft tokens appended after context

        Returns
        -------
        target_probs : [B, draft_steps + 1, V]  float32
            Softmax probabilities at positions T..T+draft_steps (inclusive).
            The last position is the target's own prediction for the next token
            after all drafts have been accepted.
        target_logits : [B, draft_steps + 1, V]  float32
            Raw logits (useful for rejection-sampling with exact probabilities).
        """
        full_ids = torch.cat([input_ids, draft_ids], dim=-1)   # [B, T + steps]
        logits = self._forward(full_ids)                        # [B, T+steps, V]

        # We need positions T-1 .. T+steps-1  (shifted by 1: predicting next)
        T = input_ids.shape[1]
        steps = draft_ids.shape[1]
        # positions T-1 to T+steps-1 → slice [T-1 : T+steps]
        sliced_logits = logits[:, T - 1: T + steps, :]         # [B, steps+1, V]
        target_probs = F.softmax(sliced_logits, dim=-1)

        self._call_count += 1
        self._total_tokens_scored += full_ids.shape[0] * full_ids.shape[1]

        return target_probs, sliced_logits

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens_scored": self._total_tokens_scored,
        }
