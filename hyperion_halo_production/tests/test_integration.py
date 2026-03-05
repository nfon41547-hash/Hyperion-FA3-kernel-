"""
test_integration.py – Integration tests for Hyperion HALO.

Tests the interaction between the scheduler, continuous batcher, metrics
collector, alerting system, and dashboard.  All tests run on CPU (no GPU
or compiled CUDA kernels required).
"""

import time

import pytest
import torch

from hyperion_halo_production.scheduler.kv_aware_scheduler import (
    KVAwareScheduler,
    PrioritizedRequest,
    RequestPriority,
)
from hyperion_halo_production.scheduler.continuous_batch import ContinuousBatch
from hyperion_halo_production.monitoring.metrics_collector import MetricsCollector
from hyperion_halo_production.monitoring.alerting import AlertManager, AlertRule
from hyperion_halo_production.monitoring.dashboard import Dashboard
from hyperion_halo_production.speculative.speculative_decoder import (
    SpeculativeDecoder,
    SpeculativeDecoderConfig,
)


# ============================================================
# Helpers
# ============================================================
def _make_req(idx: int, max_tokens: int = 8) -> PrioritizedRequest:
    return PrioritizedRequest(
        priority=RequestPriority.MEDIUM.value,
        arrival_time=time.time(),
        req_id=f"req_{idx}",
        max_tokens=max_tokens,
        kv_head=0,
        ctx_len=16,
        block_table=list(range(8)),
    )


# ============================================================
# Scheduler + ContinuousBatch integration
# ============================================================
class TestSchedulerContinuousBatchIntegration:
    def test_submit_and_drain(self):
        sched = KVAwareScheduler(max_batch_size=8, cluster_window_ms=0.0)
        for i in range(5):
            sched.submit(_make_req(i))

        batch, _ = sched.get_next_batch()
        assert 1 <= len(batch) <= 5

    def test_complete_removes_from_active(self):
        sched = KVAwareScheduler(max_batch_size=8, cluster_window_ms=0.0)
        req = _make_req(0, max_tokens=2)
        sched.submit(req)
        batch, _ = sched.get_next_batch()
        assert batch
        sched.complete(batch[0].req_id, tokens_generated=2)
        assert sched.continuous_batch.is_empty()

    def test_stats_after_completion(self):
        sched = KVAwareScheduler(max_batch_size=8, cluster_window_ms=0.0)
        for i in range(3):
            sched.submit(_make_req(i))
        batch, _ = sched.get_next_batch()
        for r in batch:
            sched.complete(r.req_id, tokens_generated=r.max_tokens)
        stats = sched.get_stats()
        assert stats["completed"] == len(batch)
        assert stats["p95_latency_ms"] >= 0

    def test_priority_ordering(self):
        """CRITICAL priority requests should not be starved."""
        sched = KVAwareScheduler(max_batch_size=16, cluster_window_ms=0.0)
        for i in range(10):
            req = _make_req(i)
            req.priority = RequestPriority.LOW.value
            sched.submit(req)
        # Add one critical request
        critical = _make_req(99)
        critical.priority = RequestPriority.CRITICAL.value
        sched.submit(critical)
        batch, _ = sched.get_next_batch()
        req_ids = [r.req_id for r in batch]
        assert "req_99" in req_ids

    def test_speculative_slots_reserve_capacity(self):
        sched = KVAwareScheduler(
            max_batch_size=10, cluster_window_ms=0.0, speculative_slots=2
        )
        for i in range(10):
            sched.submit(_make_req(i))
        batch, _ = sched.get_next_batch()
        # Effective capacity = 10 - 2 = 8
        assert len(batch) <= 8


class TestContinuousBatch:
    def test_add_and_update(self):
        batch = ContinuousBatch(max_batch_size=4)
        batch.add("r0", max_tokens=4)
        batch.add("r1", max_tokens=2)
        assert batch.size() == 2
        done = batch.update("r1", tokens_generated=2)
        assert done is True
        assert batch.size() == 1

    def test_capacity_enforcement(self):
        batch = ContinuousBatch(max_batch_size=2)
        batch.add("r0", max_tokens=4)
        batch.add("r1", max_tokens=4)
        assert batch.is_full()
        assert not batch.can_add()

    def test_tokens_remaining(self):
        batch = ContinuousBatch()
        batch.add("r0", max_tokens=10)
        batch.update("r0", tokens_generated=3)
        assert batch.tokens_remaining("r0") == 7


# ============================================================
# MetricsCollector
# ============================================================
class TestMetricsCollector:
    def test_record_lifecycle(self):
        mc = MetricsCollector()
        mc.record_request_start("r0")
        time.sleep(0.01)
        mc.record_first_token("r0")
        mc.record_request_complete("r0", tokens_generated=10)
        stats = mc.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 10
        assert stats["avg_latency_ms"] > 0
        assert stats["avg_ttft_ms"] > 0

    def test_in_flight_counter(self):
        mc = MetricsCollector()
        mc.record_request_start("r0")
        mc.record_request_start("r1")
        assert mc.get_stats()["in_flight"] == 2
        mc.record_request_complete("r0")
        assert mc.get_stats()["in_flight"] == 1

    def test_reset(self):
        mc = MetricsCollector()
        mc.record_request_start("r0")
        mc.record_request_complete("r0", tokens_generated=5)
        mc.reset()
        stats = mc.get_stats()
        assert stats["total_requests"] == 0

    def test_multiple_requests_percentiles(self):
        mc = MetricsCollector()
        for i in range(20):
            mc.record_request_start(f"r{i}")
            time.sleep(0.001)
            mc.record_request_complete(f"r{i}", tokens_generated=1)
        stats = mc.get_stats()
        assert stats["p95_latency_ms"] >= stats["p50_latency_ms"]
        assert stats["p99_latency_ms"] >= stats["p95_latency_ms"]


# ============================================================
# Alerting
# ============================================================
class TestAlertManager:
    def test_fires_when_threshold_exceeded(self):
        mgr = AlertManager()
        fired = []
        mgr.add_handler(lambda rule, val: fired.append(rule.name))
        mgr.add_rule(AlertRule("test_alert", metric="p99_latency_ms", threshold=100, op=">", cooldown_s=0))
        mgr.evaluate({"p99_latency_ms": 200.0})
        assert "test_alert" in fired

    def test_does_not_fire_below_threshold(self):
        mgr = AlertManager()
        fired = []
        mgr.add_handler(lambda rule, val: fired.append(rule.name))
        mgr.add_rule(AlertRule("test_alert", metric="p99_latency_ms", threshold=100, op=">", cooldown_s=0))
        mgr.evaluate({"p99_latency_ms": 50.0})
        assert "test_alert" not in fired

    def test_cooldown_prevents_repeat_fire(self):
        mgr = AlertManager()
        fired = []
        mgr.add_handler(lambda rule, val: fired.append(rule.name))
        mgr.add_rule(AlertRule("cooldown_test", metric="val", threshold=0, op=">", cooldown_s=100))
        mgr.evaluate({"val": 1.0})
        mgr.evaluate({"val": 1.0})
        assert fired.count("cooldown_test") == 1

    def test_default_production_rules(self):
        mgr = AlertManager.default_production_rules()
        assert mgr._rules, "Should have at least one rule"

    def test_stats(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("r1", metric="x", threshold=0, op=">", cooldown_s=0))
        mgr.evaluate({"x": 1.0})
        stats = mgr.stats()
        assert stats["fired_counts"].get("r1", 0) == 1


# ============================================================
# Dashboard (no TTY – just make sure render() doesn't crash)
# ============================================================
class TestDashboard:
    def test_render_without_crash(self):
        mc = MetricsCollector()
        mc.record_request_start("r0")
        mc.record_request_complete("r0", tokens_generated=5)
        dash = Dashboard(mc, clear_screen=False)
        dash.render()  # should not raise

    def test_start_stop(self):
        mc = MetricsCollector()
        dash = Dashboard(mc, refresh_interval_s=0.05, clear_screen=False)
        dash.start()
        time.sleep(0.12)
        dash.stop()
        assert not dash._running


# ============================================================
# End-to-end: scheduler → speculative decoder → metrics
# ============================================================
class TestEndToEnd:
    def test_scheduler_feeds_speculative_decoder(self):
        V = 256
        sched = KVAwareScheduler(max_batch_size=4, cluster_window_ms=0.0)
        mc = MetricsCollector()
        cfg = SpeculativeDecoderConfig(draft_steps=2, max_new_tokens=4)
        from hyperion_halo_production.speculative.draft_model import DraftModel, DraftModelConfig
        from hyperion_halo_production.speculative.target_model import TargetModel, TargetModelConfig
        dm = DraftModel(DraftModelConfig(vocab_size=V, device="cpu"))
        tm = TargetModel(TargetModelConfig(vocab_size=V, device="cpu"))
        decoder = SpeculativeDecoder(draft_model=dm, target_model=tm, config=cfg)

        prompts = ["hello world", "test prompt"]
        for i, p in enumerate(prompts):
            req = _make_req(i, max_tokens=4)
            req.prompt = p
            sched.submit(req)
            mc.record_request_start(req.req_id)

        batch, _ = sched.get_next_batch()
        for req in batch:
            ids = torch.randint(3, V, (1, 5))
            out = decoder.generate(ids, max_new_tokens=4)
            tokens = out.shape[1] - ids.shape[1]
            mc.record_request_complete(req.req_id, tokens_generated=max(tokens, 1))
            sched.complete(req.req_id, tokens_generated=max(tokens, 1))

        stats = mc.get_stats()
        assert stats["total_requests"] == len(prompts)
        assert stats["total_tokens"] >= len(prompts)
