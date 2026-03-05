"""
dashboard.py – Real-time terminal dashboard for Hyperion HALO.

Prints a compact, auto-refreshing metrics table to stdout.  Designed to be
run in a background thread alongside the serving engine.

Usage
-----
from hyperion_halo_production.monitoring.dashboard import Dashboard
from hyperion_halo_production.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()
dashboard = Dashboard(collector, refresh_interval_s=2.0)
dashboard.start()
# ... run serving ...
dashboard.stop()
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

from .metrics_collector import MetricsCollector


_CLEAR = "\033[H\033[J"   # ANSI: move to top-left and clear screen


def _bar(fraction: float, width: int = 20) -> str:
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


class Dashboard:
    """
    Real-time ASCII metrics dashboard.

    Parameters
    ----------
    collector          : MetricsCollector
    refresh_interval_s : float  – seconds between screen refreshes
    clear_screen       : bool   – use ANSI clear (disable for CI/logging)
    """

    def __init__(
        self,
        collector: MetricsCollector,
        refresh_interval_s: float = 2.0,
        clear_screen: bool = True,
    ) -> None:
        self.collector = collector
        self.refresh_interval_s = refresh_interval_s
        self.clear_screen = clear_screen
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            self.render()
            time.sleep(self.refresh_interval_s)

    def render(self) -> None:
        stats = self.collector.get_stats()
        lines = self._build_lines(stats)
        output = "\n".join(lines)
        if self.clear_screen and sys.stdout.isatty():
            print(_CLEAR + output, flush=True)
        else:
            print(output, flush=True)

    @staticmethod
    def _build_lines(stats: dict) -> list:
        w = 60
        sep = "─" * w
        lines = [
            "╔" + "═" * w + "╗",
            "║" + " 🔥 HYPERION HALO – Live Dashboard".center(w) + "║",
            "╠" + "═" * w + "╣",
        ]

        def row(label: str, value: str) -> str:
            return "║  " + f"{label:<28}{value}".ljust(w - 2) + "║"

        sep_row = "║  " + sep[:w - 4] + "  ║"

        lines += [
            row("Requests completed:", f"{stats['total_requests']:,}"),
            row("Tokens generated:", f"{stats['total_tokens']:,}"),
            row("In-flight requests:", str(stats["in_flight"])),
            row("Throughput (req/s):", f"{stats['throughput_req_per_s']:.1f}"),
            row("Throughput (tok/s):", f"{stats['throughput_tok_per_s']:.1f}"),
            sep_row,
            row("Avg latency (ms):", f"{stats['avg_latency_ms']:.1f}"),
            row("P95 latency (ms):", f"{stats['p95_latency_ms']:.1f}"),
            row("P99 latency (ms):", f"{stats['p99_latency_ms']:.1f}"),
            row("Avg TTFT (ms):", f"{stats['avg_ttft_ms']:.1f}"),
            sep_row,
            row("Spec. accept rate:", f"{stats['avg_acceptance_rate']:.1%}"),
            sep_row,
        ]

        # GPU bar
        gpu_pct = stats.get("gpu_utilization_pct", 0.0)
        lines.append(row("GPU util:", f"[{_bar(gpu_pct / 100)}] {gpu_pct:.0f}%"))

        # KV-cache bar
        kv = stats.get("kv_cache_utilization", 0.0)
        lines.append(row("KV-cache util:", f"[{_bar(kv)}] {kv:.1%}"))

        mem_used = stats.get("gpu_memory_used_gb", 0.0)
        mem_total = stats.get("gpu_memory_total_gb", 0.0)
        lines.append(row("VRAM:", f"{mem_used:.1f} / {mem_total:.1f} GB"))
        lines.append("╚" + "═" * w + "╝")
        return lines
