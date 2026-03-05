"""
continuous_batch.py – Continuous batching engine for Hyperion HALO.

Extracted and extended from the ContinuousBatch class in
hyperion_fa3_production_final.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RequestState:
    """Tracks per-request generation progress."""

    req_id: str
    tokens_generated: int = 0
    max_tokens: int = 128
    is_finished: bool = False


class ContinuousBatch:
    """
    Iteration-level continuous batching (vLLM-style).

    Requests are added to an *active* pool and removed when they finish
    (i.e., ``tokens_generated >= max_tokens``).  The pool acts as the single
    source of truth: there is no separate ``current_batch`` list.

    Usage
    -----
    batch = ContinuousBatch(max_batch_size=32)
    batch.add(req)
    batch.update("req_0", tokens_generated=1)
    active = batch.active_request_ids()
    """

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size
        # req_id → (tokens_generated, max_tokens)
        self.active_requests: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    def can_add(self, req_id: str = "", max_tokens: int = 1) -> bool:
        """Return True if there is capacity for one more request."""
        return len(self.active_requests) < self.max_batch_size

    def add(self, req_id: str, max_tokens: int) -> None:
        """Add a request to the active pool."""
        self.active_requests[req_id] = (0, max_tokens)

    def update(self, req_id: str, tokens_generated: int = 1) -> bool:
        """
        Advance generation counter for *req_id*.

        Returns
        -------
        bool  True if the request is now finished and was removed.
        """
        if req_id not in self.active_requests:
            return False
        generated, max_tok = self.active_requests[req_id]
        generated += tokens_generated
        if generated >= max_tok:
            del self.active_requests[req_id]
            return True
        self.active_requests[req_id] = (generated, max_tok)
        return False

    def remove(self, req_id: str) -> None:
        """Forcibly remove a request (e.g. on error or cancellation)."""
        self.active_requests.pop(req_id, None)

    # ------------------------------------------------------------------
    def active_request_ids(self) -> list:
        return list(self.active_requests.keys())

    def size(self) -> int:
        return len(self.active_requests)

    def is_full(self) -> bool:
        return self.size() >= self.max_batch_size

    def is_empty(self) -> bool:
        return self.size() == 0

    def tokens_remaining(self, req_id: str) -> Optional[int]:
        if req_id not in self.active_requests:
            return None
        generated, max_tok = self.active_requests[req_id]
        return max_tok - generated

    def __repr__(self) -> str:
        return (
            f"ContinuousBatch(size={self.size()}/{self.max_batch_size}, "
            f"ids={list(self.active_requests)[:4]}{'...' if self.size() > 4 else ''})"
        )
