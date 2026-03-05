"""
kv_aware_scheduler.py – KV-Aware Scheduler with speculative-decoding support.

Extracted and extended from KVAwareSchedulerV2 in
hyperion_fa3_production_final.py.

New in HALO
-----------
* ``speculative_slots`` parameter: reserves batch capacity for speculative
  draft tokens so the scheduler can accept draft + target requests in the same
  batch without exceeding the physical limit.
* ``get_speculative_batch()`` returns a (draft_batch, target_batch) split.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .continuous_batch import ContinuousBatch


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------
class RequestPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(order=True)
class PrioritizedRequest:
    """A single inference request with scheduling metadata."""

    priority: int
    arrival_time: float
    req_id: str
    # ---- non-compare fields ----
    prompt: str = field(default="", compare=False)
    max_tokens: int = field(default=128, compare=False)
    kv_signature: Tuple = field(default_factory=tuple, compare=False)
    kv_head: int = field(default=0, compare=False)
    ctx_len: int = field(default=0, compare=False)
    block_table: List[int] = field(default_factory=list, compare=False)
    sla_ms: float = field(default=500.0, compare=False)
    is_draft: bool = field(default=False, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class KVAwareScheduler:
    """
    KV-locality-aware continuous-batching scheduler.

    Features
    --------
    * Priority queue with arrival-time tiebreaking
    * KV-block hotness scores drive locality clustering
    * Adaptive cluster window (shrinks when queue is deep)
    * Priority aging: long-waiting requests get boosted
    * Speculative-decoding support: ``speculative_slots`` reserves capacity
      for draft-token batches
    """

    def __init__(
        self,
        target_p95_ms: float = 100.0,
        target_p99_ms: float = 200.0,
        cluster_window_ms: float = 2.0,
        max_batch_size: int = 32,
        prefetch_top_n: int = 8,
        hotness_decay: float = 0.9,
        aging_boost_per_sec: float = 5.0,
        speculative_slots: int = 0,
    ) -> None:
        self.target_p95 = target_p95_ms
        self.target_p99 = target_p99_ms
        self.base_cluster_window = cluster_window_ms / 1000.0
        self.cluster_window = self.base_cluster_window
        self.max_batch_size = max_batch_size
        self.prefetch_top_n = prefetch_top_n
        self.hotness_decay = hotness_decay
        self.aging_boost_per_sec = aging_boost_per_sec
        self.speculative_slots = speculative_slots

        self.locality_queues: Dict[Tuple, List] = defaultdict(list)
        self.kv_hotness: Dict[int, float] = defaultdict(float)
        self.latencies: deque = deque(maxlen=2000)
        self.start_times: Dict[str, float] = {}
        self.completed: int = 0
        self.rejected: int = 0
        self.lock = threading.Lock()
        self.last_window: float = time.time()
        self.pending_batch: List[PrioritizedRequest] = []
        self.last_decay: float = time.time()
        self.continuous_batch = ContinuousBatch(max_batch_size)
        self.block_access_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Signature / cost helpers
    # ------------------------------------------------------------------
    def compute_signature(
        self, block_table: List[int], kv_head: int, ctx_len: int
    ) -> Tuple:
        blocks = tuple(block_table[:8])
        bucket = ctx_len // 256
        return (blocks, kv_head, bucket)

    def compute_cost(self, req: PrioritizedRequest) -> float:
        now = time.time()
        wait = now - req.arrival_time
        blocks = req.block_table[:8]
        hot = sum(self.kv_hotness.get(b, 0) for b in blocks) / max(len(blocks), 1)
        prio_factor = 1.0 / (req.priority + 1)
        aging = wait * self.aging_boost_per_sec
        return -(aging + prio_factor * 10 + 0.5 * hot)

    def _update_hotness(self, block_table: List[int]) -> None:
        for b in block_table[:16]:
            self.kv_hotness[b] += 1.0
            self.block_access_counts[b] += 1

    def decay_hotness(self) -> None:
        now = time.time()
        if now - self.last_decay > 0.05:
            for k in list(self.kv_hotness):
                self.kv_hotness[k] *= self.hotness_decay
                if self.kv_hotness[k] < 1e-3:
                    del self.kv_hotness[k]
            self.last_decay = now

    def _get_prefetch_blocks(self) -> List[int]:
        if not self.kv_hotness:
            return []
        return [
            b for b, _ in sorted(
                self.kv_hotness.items(), key=lambda x: x[1], reverse=True
            )[:self.prefetch_top_n]
        ]

    def _adapt_cluster_window(self) -> None:
        total = sum(len(q) for q in self.locality_queues.values())
        if total > self.max_batch_size * 2:
            self.cluster_window = self.base_cluster_window * 0.5
        elif total < self.max_batch_size // 2:
            self.cluster_window = self.base_cluster_window * 2.0
        else:
            self.cluster_window = self.base_cluster_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def submit(self, req: PrioritizedRequest) -> bool:
        """Enqueue a request.  Returns True if accepted."""
        with self.lock:
            sig = self.compute_signature(req.block_table, req.kv_head, req.ctx_len)
            req.kv_signature = sig
            cost = self.compute_cost(req)
            heapq.heappush(self.locality_queues[sig], (cost, req.arrival_time, req))
            self.start_times[req.req_id] = time.time()
            self._update_hotness(req.block_table)
            return True

    def get_next_batch(
        self,
    ) -> Tuple[List[PrioritizedRequest], List[int]]:
        """
        Drain the locality queues and return a batch of requests.

        Returns
        -------
        (batch, prefetch_blocks)
        """
        with self.lock:
            self.decay_hotness()
            self._adapt_cluster_window()

            now = time.time()
            if now - self.last_window < self.cluster_window and self.pending_batch:
                return [], []

            effective_max = self.max_batch_size - self.speculative_slots
            batch: List[PrioritizedRequest] = []

            for sig in list(self.locality_queues):
                q = self.locality_queues[sig]
                while q and len(batch) < effective_max:
                    _, _, req = heapq.heappop(q)
                    if self.continuous_batch.can_add(req.req_id):
                        batch.append(req)
                if not q:
                    del self.locality_queues[sig]
                if len(batch) >= effective_max:
                    break

            self.last_window = now
            self.pending_batch = batch

            for req in batch:
                self.continuous_batch.add(req.req_id, req.max_tokens)

            return batch, self._get_prefetch_blocks()

    def get_speculative_batch(
        self,
    ) -> Tuple[List[PrioritizedRequest], List[PrioritizedRequest], List[int]]:
        """
        Return a (draft_requests, target_requests, prefetch_blocks) split
        suitable for speculative decoding.

        Draft requests (``req.is_draft == True``) are extracted first from
        the speculative_slots capacity, then target requests fill the rest.
        """
        batch, prefetch = self.get_next_batch()
        draft = [r for r in batch if r.is_draft]
        target = [r for r in batch if not r.is_draft]
        return draft, target, prefetch

    def complete(self, req_id: str, tokens_generated: int = 1) -> None:
        with self.lock:
            if req_id in self.start_times:
                latency = (time.time() - self.start_times[req_id]) * 1000
                self.latencies.append(latency)
                self.completed += 1
                del self.start_times[req_id]
                self.continuous_batch.update(req_id, tokens_generated)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        lats = list(self.latencies)
        return {
            "completed": self.completed,
            "rejected": self.rejected,
            "pending": sum(len(q) for q in self.locality_queues.values()),
            "active": self.continuous_batch.size(),
            "avg_latency_ms": float(np.mean(lats)) if lats else 0.0,
            "p95_latency_ms": float(np.percentile(lats, 95)) if lats else 0.0,
            "p99_latency_ms": float(np.percentile(lats, 99)) if lats else 0.0,
        }
