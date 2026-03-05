"""
metrics_collector.py – Production metrics collection for Hyperion HALO.

Collects and aggregates:
  * Request latency (p50, p95, p99)
  * Throughput (tokens/s, requests/s)
  * KV-cache utilization
  * GPU memory usage
  * Speculative acceptance rate
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class RequestMetric:
    """Per-request metric snapshot."""

    req_id: str
    arrival_time: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    tokens_generated: int = 0
    draft_steps_used: int = 0
    tokens_accepted: int = 0

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time-to-first-token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        """End-to-end latency in milliseconds."""
        if self.completion_time is None:
            return None
        return (self.completion_time - self.arrival_time) * 1000

    @property
    def acceptance_rate(self) -> float:
        if self.draft_steps_used == 0:
            return 0.0
        return self.tokens_accepted / self.draft_steps_used


class MetricsCollector:
    """
    Thread-safe metrics collector.

    Usage
    -----
    collector = MetricsCollector(window_size=1000)
    collector.record_request_start("req_1")
    collector.record_first_token("req_1")
    collector.record_request_complete("req_1", tokens_generated=32)
    stats = collector.get_stats()
    """

    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._lock = Lock()
        self._in_flight: Dict[str, RequestMetric] = {}
        self._completed: Deque[RequestMetric] = deque(maxlen=window_size)

        # Counters
        self._total_requests = 0
        self._total_tokens = 0
        self._total_rejected = 0
        self._start_time = time.time()

        # GPU metrics (populated externally)
        self.gpu_utilization_pct: float = 0.0
        self.gpu_memory_used_gb: float = 0.0
        self.gpu_memory_total_gb: float = 0.0
        self.kv_cache_utilization: float = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_request_start(self, req_id: str) -> None:
        with self._lock:
            self._in_flight[req_id] = RequestMetric(
                req_id=req_id, arrival_time=time.time()
            )
            self._total_requests += 1

    def record_first_token(self, req_id: str) -> None:
        with self._lock:
            if req_id in self._in_flight:
                self._in_flight[req_id].first_token_time = time.time()

    def record_request_complete(
        self,
        req_id: str,
        tokens_generated: int = 0,
        draft_steps_used: int = 0,
        tokens_accepted: int = 0,
    ) -> None:
        with self._lock:
            metric = self._in_flight.pop(req_id, None)
            if metric is None:
                return
            metric.completion_time = time.time()
            metric.tokens_generated = tokens_generated
            metric.draft_steps_used = draft_steps_used
            metric.tokens_accepted = tokens_accepted
            self._completed.append(metric)
            self._total_tokens += tokens_generated

    def record_rejection(self) -> None:
        with self._lock:
            self._total_rejected += 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        with self._lock:
            completed = list(self._completed)

        if not completed:
            latencies: List[float] = []
            ttfts: List[float] = []
            acc_rates: List[float] = []
        else:
            latencies = [m.e2e_latency_ms for m in completed if m.e2e_latency_ms is not None]
            ttfts = [m.ttft_ms for m in completed if m.ttft_ms is not None]
            acc_rates = [m.acceptance_rate for m in completed if m.draft_steps_used > 0]

        elapsed = time.time() - self._start_time

        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_rejected": self._total_rejected,
            "in_flight": len(self._in_flight),
            "elapsed_s": elapsed,
            "throughput_req_per_s": self._total_requests / max(elapsed, 1e-9),
            "throughput_tok_per_s": self._total_tokens / max(elapsed, 1e-9),
            # Latency
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            # TTFT
            "avg_ttft_ms": float(np.mean(ttfts)) if ttfts else 0.0,
            "p95_ttft_ms": float(np.percentile(ttfts, 95)) if ttfts else 0.0,
            # Speculative
            "avg_acceptance_rate": float(np.mean(acc_rates)) if acc_rates else 0.0,
            # GPU
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "kv_cache_utilization": self.kv_cache_utilization,
        }

    def reset(self) -> None:
        with self._lock:
            self._in_flight.clear()
            self._completed.clear()
            self._total_requests = 0
            self._total_tokens = 0
            self._total_rejected = 0
            self._start_time = time.time()
