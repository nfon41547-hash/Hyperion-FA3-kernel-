"""
verifier.py – Token verification for speculative decoding.

Implements the standard speculative-sampling acceptance criterion from
"Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.,
2023).

For each draft token at position i:
    r ~ Uniform(0, 1)
    if r < min(1, p_target[i] / p_draft[i]):
        accept token i
    else:
        reject and sample a correction token from
        max(0, p_target[i] - p_draft[i]) (re-normalised)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VerifierConfig:
    """Configuration for the speculative token verifier."""

    acceptance_threshold: float = 1.0  # multiplier on the acceptance ratio
    use_temperature_correction: bool = True


class Verifier:
    """
    Speculative-sampling token verifier.

    Usage
    -----
    verifier = Verifier()
    accepted_ids, correction_token = verifier.verify(
        draft_ids, draft_probs, target_probs
    )
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config or VerifierConfig()
        self._total_drafted = 0
        self._total_accepted = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify(
        self,
        draft_ids: torch.Tensor,          # [B, S]  int64
        draft_probs: torch.Tensor,        # [B, S, V] float32
        target_probs: torch.Tensor,       # [B, S+1, V] float32  (S draft + 1 correction pos)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Verify draft tokens and return accepted tokens + a correction.

        Parameters
        ----------
        draft_ids   : [B, S]
        draft_probs : [B, S, V]
        target_probs: [B, S+1, V]  – last slice is the correction position

        Returns
        -------
        accepted_ids   : [B, max_accepted]   int64    – verified draft tokens
        correction_ids : [B]                 int64    – correction token per batch
        num_accepted   : [B]                 int64    – per-sequence accept count
        """
        B, S = draft_ids.shape
        V = draft_probs.shape[-1]
        device = draft_ids.device

        # Gather draft probs for the actually-sampled token at each step
        # p_draft[b, s] = draft_probs[b, s, draft_ids[b, s]]
        p_draft = draft_probs.gather(
            -1, draft_ids.unsqueeze(-1)
        ).squeeze(-1)   # [B, S]

        # Gather target probs for the same tokens
        p_target = target_probs[:, :S, :].gather(
            -1, draft_ids.unsqueeze(-1)
        ).squeeze(-1)   # [B, S]

        # Acceptance ratio
        ratio = (p_target / p_draft.clamp(min=1e-8)).clamp(max=1.0)
        ratio *= self.config.acceptance_threshold

        # Stochastic acceptance
        r = torch.rand_like(ratio)           # [B, S]
        accept_mask = r < ratio              # [B, S]

        # Find first rejection per sequence (all tokens after first rejection are discarded)
        first_reject = torch.where(
            ~accept_mask,
            torch.arange(S, device=device).unsqueeze(0).expand(B, -1),
            torch.full((B, S), S, device=device),
        ).min(dim=-1).values                 # [B]

        num_accepted = first_reject          # [B]

        # Build accepted_ids: pad with -1 where not accepted
        accepted_ids = torch.full((B, S), -1, dtype=torch.long, device=device)
        for b in range(B):
            n = num_accepted[b].item()
            if n > 0:
                accepted_ids[b, :n] = draft_ids[b, :n]

        # Correction token: sample from re-normalised (target - draft) at rejection pos
        correction_ids = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            n = num_accepted[b].item()
            if n < S:
                # Resample from max(0, p_t - p_d)
                p_t = target_probs[b, n, :]
                p_d = draft_probs[b, n, :]
                residual = (p_t - p_d).clamp(min=0.0)
                residual_sum = residual.sum()
                if residual_sum > 1e-8:
                    correction = torch.multinomial(residual / residual_sum, 1)
                else:
                    correction = torch.multinomial(p_t, 1)
                correction_ids[b] = correction.item()
            else:
                # All accepted → sample from the last target position
                correction_ids[b] = torch.multinomial(target_probs[b, S, :], 1).item()

        self._total_drafted += B * S
        self._total_accepted += int(num_accepted.sum().item())

        return accepted_ids, correction_ids, num_accepted

    # ------------------------------------------------------------------
    def acceptance_rate(self) -> float:
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    def stats(self) -> dict:
        return {
            "total_drafted": self._total_drafted,
            "total_accepted": self._total_accepted,
            "acceptance_rate": self.acceptance_rate(),
        }
