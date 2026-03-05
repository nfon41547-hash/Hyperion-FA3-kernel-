"""
memory_manager.py – Zero-Copy / pinned-memory management for Hyperion HALO.

Provides:
  * ZeroCopyMemoryManager  – allocates CUDA pinned + mapped host buffers so the
    GPU can access host memory directly without an explicit H2D copy.
  * KVCacheAllocator       – manages a paged KV-cache pool with INT4-K / FP8-V
    layout matching the Hyperion Apex kernel expectations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Helper: align to boundary
# ---------------------------------------------------------------------------
def _align(x: int, boundary: int = 128) -> int:
    return (x + boundary - 1) & ~(boundary - 1)


# ---------------------------------------------------------------------------
# ZeroCopyMemoryManager
# ---------------------------------------------------------------------------
class ZeroCopyMemoryManager:
    """
    Allocates CUDA page-locked (pinned) memory that is also mapped into the
    device address space, enabling zero-copy GPU reads from host RAM.

    This is useful for infrequently accessed tensors (e.g. draft-model weights
    during speculative decoding) where PCIe bandwidth is acceptable and VRAM
    headroom is tight.

    Usage
    -----
    mgr = ZeroCopyMemoryManager()
    t   = mgr.allocate((1024, 512), dtype=torch.float16)
    # t lives in CPU pinned memory; GPU kernels can read it via PCIe.
    """

    def __init__(self) -> None:
        self._allocations: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Allocate a pinned-memory tensor accessible from both CPU and GPU.

        When CUDA is unavailable (e.g. in a CPU-only test environment), falls
        back to a regular CPU tensor so callers still receive a valid tensor.

        Parameters
        ----------
        shape : tuple of int
        dtype : torch dtype (default float16)
        name  : optional key for later retrieval via :py:meth:`get`

        Returns
        -------
        torch.Tensor  – pinned CPU tensor when CUDA is available, else
                         regular CPU tensor
        """
        use_pin = torch.cuda.is_available()
        t = torch.zeros(shape, dtype=dtype, pin_memory=use_pin)
        if name is not None:
            self._allocations[name] = t
        return t

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self._allocations.get(name)

    def free(self, name: str) -> None:
        self._allocations.pop(name, None)

    def free_all(self) -> None:
        self._allocations.clear()

    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self._allocations.values())

    def __repr__(self) -> str:
        n = len(self._allocations)
        total_mb = self.total_bytes() / (1024 ** 2)
        return f"ZeroCopyMemoryManager(allocations={n}, total={total_mb:.1f} MB)"


# ---------------------------------------------------------------------------
# KVCacheAllocator
# ---------------------------------------------------------------------------
@dataclass
class KVCacheConfig:
    """Configuration for the paged KV-cache pool."""

    num_layers: int = 80
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 64      # tokens per block
    num_blocks: int = 1024    # total physical blocks in pool
    device: str = "cuda"


class KVCacheAllocator:
    """
    Manages a paged KV-cache pool compatible with the Hyperion Apex V3 kernel.

    Layout (per layer)
    ------------------
    K cache : uint8  [total_slots, num_kv_heads, head_dim // 2]  – INT4 packed
    K scale : float32[total_slots, num_kv_heads]
    V cache : uint8  [total_slots, num_kv_heads, head_dim]        – FP8 E4M3
    V scale : float32[total_slots, num_kv_heads]

    total_slots = num_blocks * block_size

    Block allocation uses a simple free-list.  Block tables are per-sequence
    integer tensors mapping logical block index → physical block id.
    """

    def __init__(self, cfg: Optional[KVCacheConfig] = None) -> None:
        self.cfg = cfg or KVCacheConfig()
        c = self.cfg
        total_slots = c.num_blocks * c.block_size
        k_packed = c.head_dim // 2  # INT4: 2 values per byte

        self.k_cache = torch.zeros(
            total_slots, c.num_kv_heads, k_packed,
            dtype=torch.uint8, device=c.device,
        )
        self.k_scale = torch.ones(
            total_slots, c.num_kv_heads,
            dtype=torch.float32, device=c.device,
        )
        self.v_cache = torch.zeros(
            total_slots, c.num_kv_heads, c.head_dim,
            dtype=torch.uint8, device=c.device,
        )
        self.v_scale = torch.ones(
            total_slots, c.num_kv_heads,
            dtype=torch.float32, device=c.device,
        )

        # Free-list of available block ids
        self._free_blocks: List[int] = list(range(c.num_blocks))
        self._seq_block_tables: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Block allocation / deallocation
    # ------------------------------------------------------------------
    def allocate_blocks(self, seq_id: str, num_blocks: int) -> List[int]:
        """Allocate *num_blocks* physical blocks for a sequence."""
        if len(self._free_blocks) < num_blocks:
            raise RuntimeError(
                f"KV-cache OOM: requested {num_blocks} blocks, "
                f"only {len(self._free_blocks)} free."
            )
        allocated = [self._free_blocks.pop() for _ in range(num_blocks)]
        self._seq_block_tables.setdefault(seq_id, []).extend(allocated)
        return allocated

    def free_sequence(self, seq_id: str) -> None:
        """Return all blocks belonging to *seq_id* to the free-list."""
        blocks = self._seq_block_tables.pop(seq_id, [])
        self._free_blocks.extend(blocks)

    def get_block_table(self, seq_id: str) -> List[int]:
        return self._seq_block_tables.get(seq_id, [])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    def used_block_count(self) -> int:
        return self.cfg.num_blocks - len(self._free_blocks)

    def utilization(self) -> float:
        return self.used_block_count() / self.cfg.num_blocks

    def memory_bytes(self) -> Dict[str, int]:
        return {
            "k_cache": self.k_cache.nbytes,
            "k_scale": self.k_scale.nbytes,
            "v_cache": self.v_cache.nbytes,
            "v_scale": self.v_scale.nbytes,
        }

    def total_memory_gb(self) -> float:
        return sum(self.memory_bytes().values()) / (1024 ** 3)

    def __repr__(self) -> str:
        return (
            f"KVCacheAllocator("
            f"blocks={self.used_block_count()}/{self.cfg.num_blocks}, "
            f"util={self.utilization():.1%}, "
            f"mem={self.total_memory_gb():.2f} GB)"
        )
