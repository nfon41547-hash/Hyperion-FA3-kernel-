"""
main.py – Entry point for Hyperion HALO Production.

Commands
--------
  python main.py serve      – Start the inference server
  python main.py benchmark  – Run performance benchmarks
  python main.py demo       – Run a quick demo (CPU-only, no GPU needed)
  python main.py info       – Print system / configuration info
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from hyperion_halo_production.monitoring.metrics_collector import MetricsCollector
from hyperion_halo_production.monitoring.alerting import AlertManager
from hyperion_halo_production.monitoring.dashboard import Dashboard
from hyperion_halo_production.scheduler.kv_aware_scheduler import (
    KVAwareScheduler,
    PrioritizedRequest,
    RequestPriority,
)
from hyperion_halo_production.speculative.speculative_decoder import (
    SpeculativeDecoder,
    SpeculativeDecoderConfig,
)
from hyperion_halo_production.benchmarks.tensorcore_benchmark import (
    print_report as tc_report,
)
from hyperion_halo_production.benchmarks.speculative_benchmark import (
    print_report as spec_report,
)


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> None:
    import psutil
    import hyperion_halo_production

    print("=" * 65)
    print(f"  Hyperion HALO v{hyperion_halo_production.__version__}")
    print("=" * 65)
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA       : {'✅ ' + torch.version.cuda if torch.cuda.is_available() else '❌ not available'}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU        : {props.name}")
        print(f"  VRAM       : {props.total_memory / 1024**3:.1f} GB")
        print(f"  SM count   : {props.multi_processor_count}")
        print(f"  Capability : {props.major}.{props.minor}")
    print(f"  CPU cores  : {psutil.cpu_count()}")
    print(f"  RAM        : {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Demo command (CPU-only)
# ---------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> None:
    print("\n🚀 Hyperion HALO – Quick Demo (CPU mode)")
    print("=" * 55)

    # 1. Scheduler demo
    print("\n📋 Scheduler demo (20 requests)...")
    sched = KVAwareScheduler(max_batch_size=8, cluster_window_ms=0.0)
    mc = MetricsCollector()
    alerter = AlertManager.default_production_rules()

    for i in range(20):
        req = PrioritizedRequest(
            priority=RequestPriority.MEDIUM.value,
            arrival_time=time.time(),
            req_id=f"demo_{i}",
            max_tokens=4,
            kv_head=0,
            ctx_len=16,
            block_table=list(range(8)),
        )
        sched.submit(req)
        mc.record_request_start(req.req_id)

    completed = 0
    while completed < 20:
        batch, _ = sched.get_next_batch()
        if not batch:
            time.sleep(0.001)
            continue
        for r in batch:
            time.sleep(0.002)  # simulate compute
            mc.record_request_complete(r.req_id, tokens_generated=4)
            sched.complete(r.req_id, tokens_generated=4)
            completed += 1

    stats = mc.get_stats()
    print(f"  Completed  : {stats['total_requests']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f} ms")
    print(f"  P99 latency: {stats['p99_latency_ms']:.1f} ms")

    # 2. Speculative decoder demo
    print("\n⚡ Speculative decoder demo...")
    from hyperion_halo_production.speculative.draft_model import DraftModel, DraftModelConfig
    from hyperion_halo_production.speculative.target_model import TargetModel, TargetModelConfig
    _V = 1024  # shared stub vocab size
    _dm = DraftModel(DraftModelConfig(vocab_size=_V, draft_steps=4, device="cpu"))
    _tm = TargetModel(TargetModelConfig(vocab_size=_V, device="cpu"))
    cfg = SpeculativeDecoderConfig(draft_steps=4, max_new_tokens=16)
    decoder = SpeculativeDecoder(draft_model=_dm, target_model=_tm, config=cfg)
    ids = torch.randint(3, 1024, (1, 8))
    out = decoder.generate(ids)
    dec_stats = decoder.stats()
    print(f"  Input len  : {ids.shape[1]}")
    print(f"  Output len : {out.shape[1]}")
    print(f"  Accept rate: {dec_stats['verifier_stats']['acceptance_rate']:.1%}")

    # 3. Alert evaluation
    print("\n🔔 Alert evaluation...")
    fired = alerter.evaluate(stats)
    if fired:
        print(f"  Alerts fired: {fired}")
    else:
        print("  No alerts – all metrics within thresholds ✅")

    print("\n✅ Demo complete.")


# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------
def cmd_benchmark(args: argparse.Namespace) -> None:
    tc_report()
    print()
    spec_report()


# ---------------------------------------------------------------------------
# Serve command (minimal stub – wire in your HTTP framework)
# ---------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> None:
    print(f"Starting Hyperion HALO server on port {args.port}...")
    print("(Stub: wire in FastAPI / aiohttp / vLLM endpoint here)")
    cmd_info(args)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hyperion HALO – Production LLM Inference Engine"
    )
    sub = p.add_subparsers(dest="command", required=False)

    sub.add_parser("info", help="Print system / configuration info")
    sub.add_parser("demo", help="Run a quick CPU demo")
    sub.add_parser("benchmark", help="Run performance benchmarks")

    serve_p = sub.add_parser("serve", help="Start the inference server")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--config", type=str, default="config/hyperion_config.yaml")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "info": cmd_info,
        "demo": cmd_demo,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
        None: cmd_demo,   # default: demo
    }
    fn = dispatch.get(args.command, cmd_demo)
    fn(args)


if __name__ == "__main__":
    main()
