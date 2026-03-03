# Hyperion FA3 Kernel

A production-grade, from-scratch CUDA Flash-Attention–style kernel written entirely in Python (PyTorch + `load_inline`) and CUDA C++, optimized for NVIDIA Ampere GPUs (tested on RTX 3090 / sm_86).

---

## Overview

Hyperion FA3 re-implements the core compute path of a Flash-Attention 3–inspired attention kernel **without relying on any vendor-provided attention library**.  
The system couples the hand-written GPU kernel with a Python-level serving stack — scheduler, continuous batcher, memory estimator, and system profiler — to demonstrate end-to-end LLM inference infrastructure design.

---

## Key Technical Contributions

| Component | Details |
|---|---|
| **CUDA Persistent Kernel** | Persistent-kernel pattern with device-level `g_work_counter` and `atomicAdd`-based work-stealing across thread blocks |
| **4-Stage `cp.async` Pipeline** | Asynchronous global→shared memory copies (Ampere `cp.async.cg`) with 4 in-flight pipeline stages to fully overlap IO and compute |
| **Warp Specialisation** | 8 warps per block split into 4 *load warps* and 4 *compute warps*, eliminating synchronisation bottlenecks |
| **XOR Bank-Swizzle Layout** | Custom `xor_swizzle_bank` addressing eliminates shared-memory bank conflicts for both K and V tiles |
| **Binary / FP8 Mixed Precision** | Q and K stored as packed `uint32` bit-vectors (1-bit quantization); V stored as FP8 e4m3 with a software dequant path, Ampere-safe (no `cuda_fp8.h`) |
| **Online Softmax (Flash-Attention style)** | Numerically stable running `(m_i, l_i)` accumulation with exponential clamping (`EXP_CLAMP = -80`) |
| **KV-Aware Scheduler** | Priority queue + KV-locality clustering window; decaying KV hotness scores drive batch formation to maximize cache reuse |
| **Continuous Batching Engine** | `ContinuousBatch` + `KVAwareScheduler` implement iteration-level scheduling (vLLM-style), tracking per-request token budgets |
| **Memory Estimator** | Analytical `calculate_memory_requirements` covering weights, packed KV cache, activations, and overhead; reports fit/no-fit for 24 GB VRAM |
| **System Profiler** | Runtime GPU capability query (SM count, compute capability, max threads) driving dynamic grid/block/smem launch configuration |

---

## Project Goals

- Demonstrate deep, hands-on GPU programming knowledge beyond library calls
- Implement research-level attention optimizations (persistent kernels, async pipelines, warp specialization) in a self-contained, production-quality codebase
- Show full-stack ML systems thinking: from CUDA kernel authorship to serving-layer scheduler design

---

## Technology Stack

`Python 3` · `PyTorch` · `CUDA C++ (sm_86 / Ampere)` · `cp.async` · `atomicAdd` · `FP8 e4m3` · `Binary Quantisation` · `threading` · `heapq` · `psutil` · `NumPy`

---

## Showcasing This Project in a Resume or Portfolio

Below are ready-to-use descriptions at different lengths. Adapt them to your own voice.

---

### Resume bullet points (1–2 lines each)

```
• Authored a production CUDA Flash-Attention kernel (Hyperion FA3) from scratch in PyTorch + CUDA C++,
  implementing 4-stage cp.async pipelining, warp specialization, and XOR bank-swizzle for RTX 3090 (sm_86).

• Built an end-to-end LLM inference stack — KV-aware scheduler, continuous batching engine, FP8/binary
  mixed-precision attention, and analytical memory estimator — entirely without vendor attention libraries.
```

---

### Portfolio / GitHub README project card (3–5 sentences)

> **Hyperion FA3 Kernel** — *Original work*
>
> A fully hand-written, production-hardened CUDA attention kernel and Python serving stack for large language model inference on NVIDIA Ampere GPUs. Core contributions include a persistent-kernel work-stealing scheduler, 4-stage asynchronous shared-memory pipeline (`cp.async`), 8-warp specialization (load/compute split), XOR bank-conflict-free tile layout, and online Flash-Attention–style softmax with FP8 + binary quantization. The Python layer adds a KV-locality-aware continuous batcher and analytical VRAM estimator. Built from first principles without any third-party attention library — every line of CUDA and every scheduling decision is original.

---

### LinkedIn / Cover-letter paragraph

> I designed and implemented **Hyperion FA3**, an end-to-end GPU attention kernel and LLM inference runtime written from scratch. On the hardware side I authored a persistent CUDA kernel targeting NVIDIA Ampere (sm_86) that uses four-stage `cp.async` pipelining, warp specialization, and XOR swizzle addressing to maximize throughput on an RTX 3090. On the systems side I built a KV-cache-aware continuous batching scheduler, a mixed-precision (FP8 / 1-bit) compute path, and an analytical memory model that predicts VRAM requirements for 70 B–405 B parameter models. This project demonstrates my ability to work simultaneously at the CUDA instruction level and at the distributed-inference architecture level — skills directly applicable to AI/ML infrastructure, HPC, and production systems engineering roles.

---

### Applicable roles

This project is strong evidence for candidates targeting:

- **AI / ML Infrastructure Engineer** — LLM serving, attention kernel optimisation, KV-cache management
- **HPC / GPU Engineer** — CUDA kernel authorship, memory hierarchy optimisation, Ampere-specific ISA features
- **Senior Software Engineer (ML Systems)** — Production-quality Python + C++ co-design, scheduler design, profiling

---

## Original Work Statement

All code in this repository — including the CUDA kernel, Python serving stack, scheduler, memory estimator, and profiler — is original work authored by the repository owner. No vendor attention library (cuDNN, FlashAttention, xFormers, etc.) is used at runtime. The kernel and surrounding infrastructure were written from first principles and may be cited as an independent technical contribution in job applications, academic portfolios, or technical interviews.

---

## Quick Start

```bash
# Requires: Python ≥ 3.9, PyTorch with CUDA ≥ 11.8, NVIDIA Ampere GPU (sm_80+)
pip install torch psutil numpy
python hyperion_fa3_production_final.py
```

Expected output includes system info, a checklist of enabled optimizations, memory estimates for 70 B and 405 B models, and the CUDA kernel load status.

---

## License

See [LICENSE](LICENSE).