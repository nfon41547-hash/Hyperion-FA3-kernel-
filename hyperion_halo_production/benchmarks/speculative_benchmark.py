"""
speculative_benchmark.py – Speculative decoding performance benchmarks.

Measures:
  * Scheduler throughput (requests/s, latency percentiles)
  * Speculative decoder token throughput (with stub models)
  * Acceptance rate sensitivity to draft_steps

Run with:
    python -m hyperion_halo_production.benchmarks.speculative_benchmark
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from hyperion_halo_production.scheduler.kv_aware_scheduler import (
    KVAwareScheduler,
    PrioritizedRequest,
    RequestPriority,
)
from hyperion_halo_production.speculative.draft_model import DraftModel, DraftModelConfig
from hyperion_halo_production.speculative.target_model import TargetModel, TargetModelConfig
from hyperion_halo_production.speculative.verifier import Verifier
from hyperion_halo_production.speculative.speculative_decoder import (
    SpeculativeDecoder,
    SpeculativeDecoderConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_req(idx: int, num_blocks: int = 32, max_tokens: int = 32) -> PrioritizedRequest:
    return PrioritizedRequest(
        priority=RequestPriority.MEDIUM.value,
        arrival_time=time.time(),
        req_id=f"bench_{idx}",
        max_tokens=max_tokens,
        kv_head=idx % 8,
        ctx_len=64,
        block_table=[j % 1024 for j in range(num_blocks)],
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_scheduler(
    num_requests: int = 500,
    max_batch_size: int = 32,
    max_tokens: int = 8,
) -> Dict[str, float]:
    sched = KVAwareScheduler(
        max_batch_size=max_batch_size,
        cluster_window_ms=0.0,
    )

    t0 = time.perf_counter()
    for i in range(num_requests):
        sched.submit(_make_req(i, max_tokens=max_tokens))

    total_completed = 0
    batches = 0
    while total_completed < num_requests:
        batch, _ = sched.get_next_batch()
        if not batch:
            time.sleep(0.0001)
            continue
        for r in batch:
            sched.complete(r.req_id, tokens_generated=max_tokens)
            total_completed += 1
        batches += 1

    elapsed = time.perf_counter() - t0
    stats = sched.get_stats()
    return {
        "num_requests": num_requests,
        "elapsed_s": elapsed,
        "throughput_req_per_s": num_requests / elapsed,
        "avg_latency_ms": stats["avg_latency_ms"],
        "p95_latency_ms": stats["p95_latency_ms"],
        "p99_latency_ms": stats["p99_latency_ms"],
        "batches": batches,
    }


def bench_speculative_decoder(
    batch_size: int = 1,
    prompt_len: int = 32,
    max_new_tokens: int = 64,
    draft_steps: int = 4,
    vocab_size: int = 1024,
) -> Dict[str, float]:
    dm = DraftModel(DraftModelConfig(vocab_size=vocab_size, draft_steps=draft_steps, device="cpu"))
    tm = TargetModel(TargetModelConfig(vocab_size=vocab_size, device="cpu"))
    cfg = SpeculativeDecoderConfig(draft_steps=draft_steps, max_new_tokens=max_new_tokens)
    decoder = SpeculativeDecoder(draft_model=dm, target_model=tm, config=cfg)

    ids = torch.randint(3, vocab_size, (batch_size, prompt_len))

    t0 = time.perf_counter()
    out = decoder.generate(ids, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - t0

    tokens_gen = out.shape[1] - prompt_len
    s = decoder.stats()
    return {
        "batch_size": batch_size,
        "draft_steps": draft_steps,
        "tokens_generated": tokens_gen,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_gen / max(elapsed, 1e-9),
        "target_calls": s["total_target_calls"],
        "acceptance_rate": s["verifier_stats"]["acceptance_rate"],
    }


def bench_acceptance_rate_vs_draft_steps(
    vocab_size: int = 256,
    prompt_len: int = 16,
    max_new_tokens: int = 32,
) -> Dict[int, float]:
    results = {}
    for steps in [1, 2, 4, 8]:
        dm = DraftModel(DraftModelConfig(vocab_size=vocab_size, draft_steps=steps, device="cpu"))
        tm = TargetModel(TargetModelConfig(vocab_size=vocab_size, device="cpu"))
        cfg = SpeculativeDecoderConfig(draft_steps=steps, max_new_tokens=max_new_tokens)
        decoder = SpeculativeDecoder(draft_model=dm, target_model=tm, config=cfg)
        ids = torch.randint(3, vocab_size, (1, prompt_len))
        decoder.generate(ids)
        results[steps] = decoder.stats()["verifier_stats"]["acceptance_rate"]
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report() -> None:
    print("=" * 65)
    print("🔥 HYPERION HALO – Speculative Decoding Benchmark")
    print("=" * 65)

    print("\n  Scheduler (500 requests, batch_size=32):")
    r = bench_scheduler(num_requests=500)
    print(f"    Throughput  : {r['throughput_req_per_s']:,.0f} req/s")
    print(f"    Avg latency : {r['avg_latency_ms']:.2f} ms")
    print(f"    P95 latency : {r['p95_latency_ms']:.2f} ms")
    print(f"    P99 latency : {r['p99_latency_ms']:.2f} ms")
    print(f"    Batches     : {r['batches']}")

    print("\n  Speculative decoder (B=1, prompt=32, new_tokens=64, draft_steps=4):")
    r = bench_speculative_decoder()
    print(f"    Tokens/s    : {r['tokens_per_sec']:.1f}")
    print(f"    Target calls: {r['target_calls']}")
    print(f"    Accept rate : {r['acceptance_rate']:.1%}")

    print("\n  Acceptance rate vs draft_steps:")
    acc = bench_acceptance_rate_vs_draft_steps()
    for steps, rate in acc.items():
        print(f"    draft_steps={steps}: {rate:.1%}")

    print("=" * 65)


if __name__ == "__main__":
    print_report()
