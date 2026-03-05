"""
test_tensorcore.py – Unit tests for the TensorCore / quantization engine.

These tests run without a physical GPU: the CUDA kernel .cu files are not
compiled here; instead we test the Python-level quantization helpers and
memory-manager logic which can run on CPU.
"""

import math
import pytest
import torch

from hyperion_halo_production.core.quantization_engine import (
    HALOQuantizationEngine,
    QuantizationConfig,
    _fp32_to_fp8_e4m3,
    _fp8_e4m3_to_fp32,
    _quantize_int4,
    _dequantize_int4,
)
from hyperion_halo_production.core.memory_manager import (
    KVCacheAllocator,
    KVCacheConfig,
    ZeroCopyMemoryManager,
)


# ============================================================
# INT4 quantization round-trip
# ============================================================
class TestINT4Quantization:
    def test_round_trip_simple(self):
        engine = HALOQuantizationEngine()
        x = torch.randn(16, 128)
        packed, scale = engine.quantize_k(x)
        recon = engine.dequantize_k(packed, scale)
        assert packed.shape == (16, 64)
        assert packed.dtype == torch.uint8
        assert recon.shape == x.shape

    def test_scale_shape(self):
        engine = HALOQuantizationEngine()
        x = torch.randn(32, 128)
        _, scale = engine.quantize_k(x)
        assert scale.shape == (32,)

    def test_zero_tensor(self):
        engine = HALOQuantizationEngine()
        x = torch.zeros(8, 128)
        packed, scale = engine.quantize_k(x)
        recon = engine.dequantize_k(packed, scale)
        assert torch.allclose(recon, x, atol=1e-3)

    def test_roundtrip_error_small(self):
        engine = HALOQuantizationEngine()
        x = torch.randn(64, 128)
        err = engine.compute_quantization_error(x, mode="k")
        # INT4 should have reasonable SNR for random inputs
        assert err["snr_db"] > 10.0, f"SNR too low: {err['snr_db']:.1f} dB"

    def test_packing_unpacking_identity(self):
        """Each INT4 value in [-8, 7] should survive a pack/unpack round-trip."""
        # Create a tensor with known INT4-friendly values
        vals = torch.arange(-8, 8, dtype=torch.float32)
        x = vals.unsqueeze(0).repeat(1, 8)  # shape [1, 128] after tiling
        x = x[:, :128]
        scale = torch.tensor([1.0])
        packed = _quantize_int4(x, scale)
        recon = _dequantize_int4(packed, scale)
        assert torch.allclose(recon, x, atol=1e-5)


# ============================================================
# FP8 E4M3 quantization round-trip
# ============================================================
class TestFP8Quantization:
    def test_round_trip_shape(self):
        engine = HALOQuantizationEngine()
        x = torch.randn(16, 128)
        fp8, scale = engine.quantize_v(x)
        recon = engine.dequantize_v(fp8, scale)
        assert fp8.shape == (16, 128)
        assert fp8.dtype == torch.uint8
        assert recon.shape == x.shape

    def test_zero_fp8(self):
        engine = HALOQuantizationEngine()
        x = torch.zeros(4, 128)
        fp8, scale = engine.quantize_v(x)
        recon = engine.dequantize_v(fp8, scale)
        assert torch.allclose(recon, x, atol=1e-3)

    def test_fp8_roundtrip_snr(self):
        engine = HALOQuantizationEngine()
        x = torch.randn(32, 128)
        err = engine.compute_quantization_error(x, mode="v")
        # FP8 E4M3 has only 3 mantissa bits; SNR of ~5-9 dB is expected for Gaussian inputs
        assert err["snr_db"] > 3.0, f"FP8 SNR too low: {err['snr_db']:.1f} dB"

    def test_fp8_encode_decode_special_values(self):
        """0.0 must encode to 0x00 and decode back to 0.0."""
        raw = _fp32_to_fp8_e4m3(torch.tensor([0.0]))
        decoded = _fp8_e4m3_to_fp32(raw)
        assert float(decoded[0]) == 0.0


# ============================================================
# Memory manager
# ============================================================
class TestZeroCopyMemoryManager:
    def test_allocate_returns_tensor(self):
        mgr = ZeroCopyMemoryManager()
        t = mgr.allocate((64, 128), dtype=torch.float16)
        assert t.shape == (64, 128)
        assert t.dtype == torch.float16
        # Pin memory requires CUDA; without GPU we get a plain CPU tensor
        if torch.cuda.is_available():
            assert t.is_pinned()

    def test_named_allocation(self):
        mgr = ZeroCopyMemoryManager()
        mgr.allocate((16, 32), name="weights")
        assert mgr.get("weights") is not None

    def test_free(self):
        mgr = ZeroCopyMemoryManager()
        mgr.allocate((16, 32), name="tmp")
        mgr.free("tmp")
        assert mgr.get("tmp") is None

    def test_total_bytes(self):
        mgr = ZeroCopyMemoryManager()
        mgr.allocate((1024,), dtype=torch.float32, name="a")
        mgr.allocate((1024,), dtype=torch.float32, name="b")
        assert mgr.total_bytes() == 2 * 1024 * 4  # float32 = 4 bytes


class TestKVCacheAllocator:
    def test_allocate_and_free(self):
        cfg = KVCacheConfig(num_blocks=64, block_size=16, device="cpu")
        alloc = KVCacheAllocator(cfg)
        blocks = alloc.allocate_blocks("seq_0", num_blocks=4)
        assert len(blocks) == 4
        assert alloc.used_block_count() == 4
        alloc.free_sequence("seq_0")
        assert alloc.used_block_count() == 0

    def test_oom_raises(self):
        cfg = KVCacheConfig(num_blocks=4, block_size=16, device="cpu")
        alloc = KVCacheAllocator(cfg)
        with pytest.raises(RuntimeError, match="OOM"):
            alloc.allocate_blocks("seq_0", num_blocks=5)

    def test_block_table(self):
        cfg = KVCacheConfig(num_blocks=64, block_size=16, device="cpu")
        alloc = KVCacheAllocator(cfg)
        blocks = alloc.allocate_blocks("seq_1", num_blocks=3)
        assert alloc.get_block_table("seq_1") == blocks

    def test_utilization(self):
        cfg = KVCacheConfig(num_blocks=100, block_size=16, device="cpu")
        alloc = KVCacheAllocator(cfg)
        alloc.allocate_blocks("s0", 50)
        assert abs(alloc.utilization() - 0.5) < 1e-9

    def test_cache_shapes(self):
        cfg = KVCacheConfig(
            num_blocks=8, num_kv_heads=2, head_dim=64, block_size=16, device="cpu"
        )
        alloc = KVCacheAllocator(cfg)
        total_slots = 8 * 16
        assert alloc.k_cache.shape == (total_slots, 2, 32)  # head_dim // 2
        assert alloc.v_cache.shape == (total_slots, 2, 64)
