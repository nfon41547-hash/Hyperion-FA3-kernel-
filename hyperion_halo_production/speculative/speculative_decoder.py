"""
speculative_decoder.py – Main Speculative Decoding Engine for Hyperion HALO.

Orchestrates the draft→verify→commit loop:

  1. DraftModel generates ``draft_steps`` candidate tokens per sequence.
  2. TargetModel scores the (context + draft) sequence in a single forward pass.
  3. Verifier accepts/rejects each draft token via speculative sampling.
  4. Accepted tokens + the correction token are appended to the context.
  5. The loop repeats until all sequences have reached ``max_new_tokens``.

Expected speedup over naive autoregressive decoding:
    throughput_gain ≈ (accepted_rate * draft_steps + 1)
    (one target forward pass verifies multiple draft tokens)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .draft_model import DraftModel, DraftModelConfig
from .target_model import TargetModel, TargetModelConfig
from .verifier import Verifier, VerifierConfig


@dataclass
class SpeculativeDecoderConfig:
    """Top-level configuration for the speculative decoder."""

    draft_steps: int = 4          # tokens drafted per target call
    max_new_tokens: int = 128
    eos_token_id: int = 2
    pad_token_id: int = 0
    temperature: float = 1.0
    verbose: bool = False


class SpeculativeDecoder:
    """
    End-to-end speculative decoding engine.

    Parameters
    ----------
    draft_model  : DraftModel   – small model for drafting
    target_model : TargetModel  – large model for verification
    config       : SpeculativeDecoderConfig
    verifier     : Verifier     – optional custom verifier
    """

    def __init__(
        self,
        draft_model: Optional[DraftModel] = None,
        target_model: Optional[TargetModel] = None,
        config: Optional[SpeculativeDecoderConfig] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.draft_model = draft_model or DraftModel()
        self.target_model = target_model or TargetModel()
        self.config = config or SpeculativeDecoderConfig()
        self.verifier = verifier or Verifier()

        self._decode_calls = 0
        self._total_tokens_generated = 0
        self._total_target_calls = 0
        self._wall_time_s = 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,       # [B, T]
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Parameters
        ----------
        input_ids     : [B, T] int64 – prompt token ids
        max_new_tokens: int or None  – overrides config.max_new_tokens

        Returns
        -------
        output_ids : [B, T + generated_tokens] int64
        """
        t_start = time.perf_counter()
        max_new = max_new_tokens or self.config.max_new_tokens
        cfg = self.config

        ids = input_ids.clone()
        B = ids.shape[0]
        finished = torch.zeros(B, dtype=torch.bool, device=ids.device)
        tokens_generated = torch.zeros(B, dtype=torch.long, device=ids.device)

        self._decode_calls += 1

        while not finished.all() and tokens_generated.min().item() < max_new:
            # ---- 1. Draft ----
            draft_ids, draft_probs = self.draft_model.draft_tokens(
                ids, num_draft_steps=cfg.draft_steps
            )
            # Mask finished sequences
            draft_ids[finished] = cfg.pad_token_id

            # ---- 2. Score with target model ----
            target_probs, _ = self.target_model.score_tokens(ids, draft_ids)
            self._total_target_calls += 1

            # ---- 3. Verify ----
            accepted_ids, correction_ids, num_accepted = self.verifier.verify(
                draft_ids, draft_probs, target_probs
            )

            # ---- 4. Append accepted tokens + correction ----
            for b in range(B):
                if finished[b]:
                    continue
                n = int(num_accepted[b].item())
                new_tokens = []
                for i in range(n):
                    tok = int(accepted_ids[b, i].item())
                    new_tokens.append(tok)
                    if tok == cfg.eos_token_id:
                        finished[b] = True
                        break
                if not finished[b]:
                    corr = int(correction_ids[b].item())
                    new_tokens.append(corr)
                    if corr == cfg.eos_token_id:
                        finished[b] = True

                if new_tokens:
                    new_t = torch.tensor(
                        new_tokens, dtype=torch.long, device=ids.device
                    ).unsqueeze(0)  # [1, k]
                    if b == 0:
                        new_segment = new_t
                    else:
                        # Pad shorter sequence to match
                        pass  # simplified: handle single-batch case
                    tokens_generated[b] += len(new_tokens)

            # Rebuild ids by appending accepted tokens (simplified single-batch)
            if B == 1:
                n = int(num_accepted[0].item())
                if not finished[0]:
                    to_append = accepted_ids[0, :n].tolist() + [int(correction_ids[0])]
                else:
                    to_append = accepted_ids[0, :n].tolist()
                to_append = [t for t in to_append if t != cfg.pad_token_id]
                if to_append:
                    app = torch.tensor(to_append, dtype=torch.long, device=ids.device)
                    ids = torch.cat([ids, app.unsqueeze(0)], dim=-1)
            else:
                # Multi-batch: append one correction per sequence (simplified)
                corrections = correction_ids.unsqueeze(-1)
                ids = torch.cat([ids, corrections], dim=-1)
                for b in range(B):
                    if int(correction_ids[b]) == cfg.eos_token_id:
                        finished[b] = True

            if cfg.verbose:
                rate = self.verifier.acceptance_rate()
                print(
                    f"  step: drafted={cfg.draft_steps} "
                    f"accepted={num_accepted.float().mean():.1f} "
                    f"acceptance_rate={rate:.2%}"
                )

        self._wall_time_s += time.perf_counter() - t_start
        self._total_tokens_generated += int(tokens_generated.sum().item())
        return ids

    # ------------------------------------------------------------------
    def throughput_tokens_per_sec(self) -> float:
        if self._wall_time_s < 1e-9:
            return 0.0
        return self._total_tokens_generated / self._wall_time_s

    def stats(self) -> dict:
        return {
            "decode_calls": self._decode_calls,
            "total_tokens_generated": self._total_tokens_generated,
            "total_target_calls": self._total_target_calls,
            "wall_time_s": self._wall_time_s,
            "tokens_per_sec": self.throughput_tokens_per_sec(),
            "draft_stats": self.draft_model.stats(),
            "target_stats": self.target_model.stats(),
            "verifier_stats": self.verifier.stats(),
        }
