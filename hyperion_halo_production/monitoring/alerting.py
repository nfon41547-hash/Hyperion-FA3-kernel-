"""
alerting.py – Alert system for Hyperion HALO production monitoring.

Evaluates metrics against configurable thresholds and fires callbacks
(or prints to stderr) when thresholds are exceeded.

Usage
-----
from hyperion_halo_production.monitoring.alerting import AlertManager, AlertRule

mgr = AlertManager()
mgr.add_rule(AlertRule("high_p99", metric="p99_latency_ms", threshold=500, op=">"))
mgr.evaluate(stats)  # call periodically with latest metrics dict
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class AlertRule:
    """A single threshold-based alert rule."""

    name: str
    metric: str                 # key in the metrics dict
    threshold: float
    op: str = ">"               # one of: ">", ">=", "<", "<=", "=="
    cooldown_s: float = 60.0    # minimum seconds between repeated alerts
    severity: str = "WARNING"   # WARNING | ERROR | CRITICAL
    message: Optional[str] = None

    def check(self, value: float) -> bool:
        ops = {
            ">":  lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<":  lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
        }
        fn = ops.get(self.op)
        return fn(value, self.threshold) if fn else False


class AlertManager:
    """
    Evaluates alert rules against a metrics snapshot and fires handlers.

    Default handler: print to stderr.
    Custom handlers can be registered via ``add_handler()``.
    """

    def __init__(self) -> None:
        self._rules: List[AlertRule] = []
        self._last_fired: Dict[str, float] = {}
        self._handlers: List[Callable[[AlertRule, float], None]] = []
        self._add_default_handler()
        self._fired_count: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _add_default_handler(self) -> None:
        def _stderr_handler(rule: AlertRule, value: float) -> None:
            msg = rule.message or (
                f"[{rule.severity}] {rule.name}: "
                f"{rule.metric}={value:.3f} {rule.op} {rule.threshold}"
            )
            print(f"🚨 ALERT {msg}", file=sys.stderr, flush=True)
        self._handlers.append(_stderr_handler)

    def add_handler(self, fn: Callable[[AlertRule, float], None]) -> None:
        """Register a custom alert handler."""
        self._handlers.append(fn)

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> None:
        self._rules = [r for r in self._rules if r.name != name]

    # ------------------------------------------------------------------
    def evaluate(self, metrics: dict) -> List[str]:
        """
        Evaluate all rules against *metrics*.

        Returns list of fired alert names (for testing / logging).
        """
        fired = []
        now = time.time()
        for rule in self._rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue
            if not rule.check(float(value)):
                continue
            # Cooldown check
            last = self._last_fired.get(rule.name, 0.0)
            if now - last < rule.cooldown_s:
                continue
            self._last_fired[rule.name] = now
            self._fired_count[rule.name] = self._fired_count.get(rule.name, 0) + 1
            for handler in self._handlers:
                try:
                    handler(rule, float(value))
                except Exception:
                    pass
            fired.append(rule.name)
        return fired

    # ------------------------------------------------------------------
    # Convenience: add the standard production ruleset
    # ------------------------------------------------------------------
    @classmethod
    def default_production_rules(cls) -> "AlertManager":
        """Return an AlertManager pre-loaded with sensible production rules."""
        mgr = cls()
        mgr.add_rule(AlertRule(
            name="high_p99_latency",
            metric="p99_latency_ms",
            threshold=500.0,
            op=">",
            severity="WARNING",
            cooldown_s=30.0,
        ))
        mgr.add_rule(AlertRule(
            name="very_high_p99_latency",
            metric="p99_latency_ms",
            threshold=1000.0,
            op=">",
            severity="CRITICAL",
            cooldown_s=10.0,
        ))
        mgr.add_rule(AlertRule(
            name="low_acceptance_rate",
            metric="avg_acceptance_rate",
            threshold=0.5,
            op="<",
            severity="WARNING",
            cooldown_s=60.0,
            message="Speculative acceptance rate < 50% – consider tuning draft_steps",
        ))
        mgr.add_rule(AlertRule(
            name="kv_cache_near_full",
            metric="kv_cache_utilization",
            threshold=0.90,
            op=">=",
            severity="ERROR",
            cooldown_s=10.0,
        ))
        mgr.add_rule(AlertRule(
            name="gpu_memory_high",
            metric="gpu_utilization_pct",
            threshold=95.0,
            op=">=",
            severity="WARNING",
            cooldown_s=30.0,
        ))
        return mgr

    def stats(self) -> dict:
        return {
            "rules": len(self._rules),
            "fired_counts": dict(self._fired_count),
        }
