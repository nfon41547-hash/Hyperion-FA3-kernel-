"""
quantization_engine.py – HALO Quantization Engine.

Provides runtime INT4 and FP8-E4M3 quantization / dequantization helpers for
preparing K and V tensors before they are written into the KV cache that the
Hyperion Apex kernel reads.

Classes
-------
QuantizationConfig   – per-tensor quantization settings
HALOQuantizationEngine – main quantization / dequantization interface
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class QuantizationConfig:
    """Per-tensor quantization settings."""

    k_bits: int = 4          # K-cache quantization bits (4 → INT4)
    v_bits: int = 8          # V-cache quantization bits (8 → FP8 E4M3 emulated)
    group_size: int = 128    # number of elements per quantization group
    symmetric: bool = True   # symmetric (zero-point = 0) or asymmetric
    clip_ratio: float = 1.0  # scale the absmax by this factor before clipping


# ---------------------------------------------------------------------------
# FP8 E4M3 helpers (software emulation – no cuda_fp8.h required)
# ---------------------------------------------------------------------------
_FP8_MAX = 448.0   # maximum representable value in E4M3


def _fp32_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize a float32 tensor to FP8 E4M3 stored as uint8.

    This mirrors the hardware E4M3 format:
      sign  : 1 bit
      exp   : 4 bits  (bias = 7)
      mant  : 3 bits
    """
    x = x.float().clamp(-_FP8_MAX, _FP8_MAX)

    sign = (x < 0).to(torch.uint8)
    abs_x = x.abs()
    zero_mask = abs_x == 0.0

    # Determine exponent (guard against log2(0) via clamp)
    safe_abs = abs_x.clamp(min=1e-30)
    exp_f = torch.floor(torch.log2(safe_abs)).clamp(-6, 8)  # bias=7 → stored exp 1..15

    # Subnormal path: exp_stored == 0
    exp_stored = (exp_f + 7).to(torch.int32)
    subnormal_mask = exp_stored <= 0

    # Normal mantissa: round to nearest 3-bit fraction
    scale_n = torch.pow(2.0, exp_f - 3)          # 2^(exp-3) for fractional bits
    mant_n = (abs_x / scale_n.clamp(min=1e-38)).round().to(torch.int32).clamp(0, 7)

    # Subnormal mantissa: value = mant * 2^(-9)
    mant_s = (abs_x / (2.0 ** -9)).round().to(torch.int32).clamp(0, 7)

    mant = torch.where(subnormal_mask, mant_s, mant_n)
    exp_out = torch.where(subnormal_mask, torch.zeros_like(exp_stored), exp_stored.clamp(1, 15))

    # Pack: [sign(1) | exp(4) | mant(3)]
    result = ((sign << 7) | (exp_out.to(torch.uint8) << 3) | mant.to(torch.uint8))
    # Exact zero → 0x00
    result = torch.where(zero_mask, torch.zeros_like(result), result)
    return result


def _fp8_e4m3_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """
    Dequantize a uint8 FP8-E4M3 tensor back to float32.
    Mirrors the kernel-side fp8_e4m3_to_fp32 device function.
    """
    x = x.to(torch.int32)
    sign = ((x >> 7) & 1).float()
    sign = 1.0 - 2.0 * sign          # +1 or -1
    exp = (x >> 3) & 0xF
    mant = (x & 0x7).float()

    # Subnormal (exp == 0): value = sign * mant * 2^(-9)
    subnorm = (exp == 0)
    val_subnorm = sign * mant * (2.0 ** -9)

    # Normal: value = sign * (1 + mant/8) * 2^(exp - 7)
    exp_f = exp.float()
    val_norm = sign * (1.0 + mant / 8.0) * torch.pow(2.0, exp_f - 7)

    return torch.where(subnorm, val_subnorm, val_norm)


# ---------------------------------------------------------------------------
# INT4 helpers
# ---------------------------------------------------------------------------
def _quantize_int4(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Quantize float32 → signed INT4 and pack two values per byte (uint8).

    Parameters
    ----------
    x     : [rows, head_dim] float32
    scale : [rows]           per-row absmax scale

    Returns
    -------
    packed : [rows, head_dim // 2] uint8
    """
    # Scale and clamp to [-8, 7]
    x_scaled = (x / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)

    rows, head_dim = x_scaled.shape
    assert head_dim % 2 == 0, "head_dim must be even for INT4 packing"

    lo = x_scaled[:, 0::2].to(torch.uint8) & 0x0F   # lower nibble
    hi = (x_scaled[:, 1::2].to(torch.uint8) & 0x0F) << 4  # upper nibble
    return (lo | hi)


def _dequantize_int4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 INT4 → float32.

    Parameters
    ----------
    packed : [rows, head_dim // 2] uint8
    scale  : [rows]

    Returns
    -------
    x : [rows, head_dim] float32
    """
    lo_u = (packed & 0x0F).to(torch.int32)
    hi_u = ((packed >> 4) & 0x0F).to(torch.int32)

    # Sign-extend from 4-bit
    lo = torch.where(lo_u >= 8, lo_u - 16, lo_u).float()
    hi = torch.where(hi_u >= 8, hi_u - 16, hi_u).float()

    rows = packed.shape[0]
    head_dim = packed.shape[1] * 2
    result = torch.empty(rows, head_dim, dtype=torch.float32, device=packed.device)
    result[:, 0::2] = lo
    result[:, 1::2] = hi
    return result * scale.unsqueeze(-1)


# ---------------------------------------------------------------------------
# HALOQuantizationEngine
# ---------------------------------------------------------------------------
class HALOQuantizationEngine:
    """
    Runtime quantization engine for the Hyperion HALO KV-cache.

    Supports:
      - INT4 per-row symmetric quantization for K tensors
      - FP8 E4M3 per-row symmetric quantization for V tensors

    Usage
    -----
    engine = HALOQuantizationEngine()
    k_packed, k_scale = engine.quantize_k(k_fp16)   # k_fp16: [tokens, head_dim]
    v_fp8,    v_scale = engine.quantize_v(v_fp16)
    k_reconstructed   = engine.dequantize_k(k_packed, k_scale)
    v_reconstructed   = engine.dequantize_v(v_fp8, v_scale)
    """

    def __init__(self, cfg: QuantizationConfig | None = None) -> None:
        self.cfg = cfg or QuantizationConfig()

    # ------------------------------------------------------------------
    # K : INT4
    # ------------------------------------------------------------------
    def quantize_k(
        self, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize K (float16/32) → INT4 packed uint8.

        Parameters
        ----------
        k : [tokens, head_dim]  float16 or float32

        Returns
        -------
        k_packed : [tokens, head_dim // 2] uint8
        k_scale  : [tokens]                float32
        """
        k_f32 = k.float()
        scale = k_f32.abs().max(dim=-1).values.clamp(min=1e-8) / 7.0
        k_packed = _quantize_int4(k_f32, scale)
        return k_packed, scale

    def dequantize_k(
        self, k_packed: torch.Tensor, k_scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize INT4 K → float32."""
        return _dequantize_int4(k_packed, k_scale)

    # ------------------------------------------------------------------
    # V : FP8 E4M3
    # ------------------------------------------------------------------
    def quantize_v(
        self, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize V (float16/32) → FP8 E4M3 stored as uint8.

        Parameters
        ----------
        v : [tokens, head_dim]  float16 or float32

        Returns
        -------
        v_fp8   : [tokens, head_dim] uint8
        v_scale : [tokens]           float32
        """
        v_f32 = v.float()
        scale = v_f32.abs().max(dim=-1).values.clamp(min=1e-8) / _FP8_MAX
        v_scaled = v_f32 / scale.unsqueeze(-1)
        v_fp8 = _fp32_to_fp8_e4m3(v_scaled)
        return v_fp8, scale

    def dequantize_v(
        self, v_fp8: torch.Tensor, v_scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize FP8 E4M3 V → float32."""
        v_f32 = _fp8_e4m3_to_fp32(v_fp8)
        return v_f32 * v_scale.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Round-trip error utility
    # ------------------------------------------------------------------
    def compute_quantization_error(
        self,
        original: torch.Tensor,
        mode: str = "k",
    ) -> dict:
        """
        Compute quantization round-trip error metrics.

        Parameters
        ----------
        original : [tokens, head_dim] tensor
        mode     : 'k' for INT4, 'v' for FP8

        Returns
        -------
        dict with keys: max_abs_err, mean_abs_err, rmse, snr_db
        """
        if mode == "k":
            packed, scale = self.quantize_k(original)
            reconstructed = self.dequantize_k(packed, scale)
        else:
            fp8, scale = self.quantize_v(original)
            reconstructed = self.dequantize_v(fp8, scale)

        orig_f = original.float()
        rec_f = reconstructed.float()
        err = (orig_f - rec_f).abs()

        signal_power = orig_f.pow(2).mean()
        noise_power = err.pow(2).mean().clamp(min=1e-30)
        snr_db = 10 * math.log10(float(signal_power) / float(noise_power) + 1e-10)

        return {
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "rmse": float(err.pow(2).mean().sqrt()),
            "snr_db": snr_db,
        }
