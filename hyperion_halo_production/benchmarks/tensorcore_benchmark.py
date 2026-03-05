"""
tensorcore_benchmark.py – TensorCore / quantization engine performance benchmarks.

Measures:
  * INT4 quantization throughput (tokens/s)
  * FP8 E4M3 quantization throughput (tokens/s)
  * Round-trip quantization error (SNR dB)
  * KV-cache allocation throughput

Run with:
    python -m hyperion_halo_production.benchmarks.tensorcore_benchmark
"""

from __future__ import annotations

import time
from typing import Dict

import torch

from hyperion_halo_production.core.quantization_engine import HALOQuantizationEngine
from hyperion_halo_production.core.memory_manager import KVCacheAllocator, KVCacheConfig


def bench_int4_quantization(
    num_tokens: int = 65536,
    head_dim: int = 128,
    num_warmup: int = 5,
    num_iters: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    engine = HALOQuantizationEngine()
    x = torch.randn(num_tokens, head_dim, device=device)

    # Warmup
    for _ in range(num_warmup):
        engine.quantize_k(x)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_iters):
        engine.quantize_k(x)
    elapsed = time.perf_counter() - t0

    total_tokens = num_tokens * num_iters
    throughput = total_tokens / elapsed

    err = engine.compute_quantization_error(x[:256], mode="k")
    return {
        "mode": "INT4",
        "num_tokens": num_tokens,
        "head_dim": head_dim,
        "elapsed_s": elapsed,
        "throughput_tok_per_s": throughput,
        "snr_db": err["snr_db"],
        "rmse": err["rmse"],
    }


def bench_fp8_quantization(
    num_tokens: int = 65536,
    head_dim: int = 128,
    num_warmup: int = 5,
    num_iters: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    engine = HALOQuantizationEngine()
    x = torch.randn(num_tokens, head_dim, device=device)

    for _ in range(num_warmup):
        engine.quantize_v(x)

    t0 = time.perf_counter()
    for _ in range(num_iters):
        engine.quantize_v(x)
    elapsed = time.perf_counter() - t0

    total_tokens = num_tokens * num_iters
    throughput = total_tokens / elapsed

    err = engine.compute_quantization_error(x[:256], mode="v")
    return {
        "mode": "FP8-E4M3",
        "num_tokens": num_tokens,
        "head_dim": head_dim,
        "elapsed_s": elapsed,
        "throughput_tok_per_s": throughput,
        "snr_db": err["snr_db"],
        "rmse": err["rmse"],
    }


def bench_kv_cache_allocation(
    num_sequences: int = 1000,
    blocks_per_seq: int = 16,
    total_blocks: int = 32768,
) -> Dict[str, float]:
    cfg = KVCacheConfig(
        num_blocks=total_blocks,
        block_size=16,
        num_kv_heads=8,
        head_dim=128,
        device="cpu",
    )
    alloc = KVCacheAllocator(cfg)

    t0 = time.perf_counter()
    seq_ids = []
    for i in range(min(num_sequences, total_blocks // blocks_per_seq)):
        sid = f"seq_{i}"
        alloc.allocate_blocks(sid, num_blocks=blocks_per_seq)
        seq_ids.append(sid)
    alloc_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for sid in seq_ids:
        alloc.free_sequence(sid)
    free_time = time.perf_counter() - t1

    return {
        "num_sequences": len(seq_ids),
        "blocks_per_seq": blocks_per_seq,
        "alloc_time_ms": alloc_time * 1000,
        "free_time_ms": free_time * 1000,
        "alloc_throughput_seq_per_s": len(seq_ids) / max(alloc_time, 1e-9),
    }


def print_report() -> None:
    print("=" * 65)
    print("🔥 HYPERION HALO – TensorCore / Quantization Benchmark")
    print("=" * 65)

    print("\n  INT4 K-quantization:")
    r = bench_int4_quantization()
    print(f"    Tokens/s    : {r['throughput_tok_per_s']:,.0f}")
    print(f"    SNR         : {r['snr_db']:.1f} dB")
    print(f"    RMSE        : {r['rmse']:.6f}")

    print("\n  FP8 E4M3 V-quantization:")
    r = bench_fp8_quantization()
    print(f"    Tokens/s    : {r['throughput_tok_per_s']:,.0f}")
    print(f"    SNR         : {r['snr_db']:.1f} dB")
    print(f"    RMSE        : {r['rmse']:.6f}")

    print("\n  KV-cache allocation:")
    r = bench_kv_cache_allocation()
    print(f"    Alloc       : {r['alloc_time_ms']:.2f} ms "
          f"({r['alloc_throughput_seq_per_s']:,.0f} seq/s)")
    print(f"    Free        : {r['free_time_ms']:.2f} ms")

    print("=" * 65)


if __name__ == "__main__":
    print_report()
