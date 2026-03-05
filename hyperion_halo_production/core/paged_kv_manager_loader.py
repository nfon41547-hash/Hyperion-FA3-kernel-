"""
paged_kv_manager_loader.py – Python loader for the lock-free C++ PagedKVManager.

Attempts to compile and load ``paged_kv_manager.cpp`` via
``torch.utils.cpp_extension.load``.  If compilation fails (e.g. on a CPU-only
build machine), a pure-Python fallback ``PyPagedKVManager`` is returned instead
so that the rest of the codebase continues to work without modification.

Usage
-----
from hyperion_halo_production.core.paged_kv_manager_loader import load_paged_kv_manager

PagedKVManager = load_paged_kv_manager()
mgr = PagedKVManager(num_blocks=1024)
blocks = mgr.allocate_blocks("seq_0", 4)
mgr.free_sequence("seq_0")
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading
from typing import List, Optional

_log = logging.getLogger(__name__)

# Cache the loaded class across repeated calls.
_manager_class: Optional[type] = None
_load_lock = threading.Lock()

_HERE = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Pure-Python fallback  (mirrors the C++ interface exactly)
# ---------------------------------------------------------------------------
class _PyPagedKVManager:
    """
    Pure-Python fallback for PagedKVManager.

    Identical public interface to the C++ extension but uses a plain list as
    the free-list (not lock-free).  Used when the C++ extension cannot be
    compiled (e.g. no compiler, CI environment).
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        self._num_blocks = num_blocks
        self._free_blocks: List[int] = list(range(num_blocks))
        self._seq_tables: dict = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def allocate_blocks(self, seq_id: str, num_blocks: int) -> List[int]:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        with self._lock:
            if len(self._free_blocks) < num_blocks:
                raise RuntimeError(
                    f"KV-cache OOM: requested {num_blocks} blocks, "
                    f"only {len(self._free_blocks)} free."
                )
            allocated = [self._free_blocks.pop() for _ in range(num_blocks)]
            self._seq_tables.setdefault(seq_id, []).extend(allocated)
            return allocated

    def free_sequence(self, seq_id: str) -> None:
        with self._lock:
            blocks = self._seq_tables.pop(seq_id, [])
            self._free_blocks.extend(blocks)

    def get_block_table(self, seq_id: str) -> List[int]:
        with self._lock:
            return list(self._seq_tables.get(seq_id, []))

    def free_block_count(self) -> int:
        with self._lock:
            return len(self._free_blocks)

    def used_block_count(self) -> int:
        with self._lock:
            return self._num_blocks - len(self._free_blocks)

    def utilization(self) -> float:
        return self.used_block_count() / self._num_blocks

    def num_blocks(self) -> int:
        return self._num_blocks

    def reset(self) -> None:
        with self._lock:
            self._free_blocks = list(range(self._num_blocks))
            self._seq_tables.clear()

    def __repr__(self) -> str:
        used = self.used_block_count()
        pct = int(self.utilization() * 100)
        return f"PyPagedKVManager(blocks={used}/{self._num_blocks}, util={pct}%)"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _try_compile_cpp_extension() -> Optional[type]:
    """
    Attempt to compile the C++ extension via torch.utils.cpp_extension.load.

    Returns the ``PagedKVManager`` class on success, or None on failure.
    """
    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        _log.debug("torch.utils.cpp_extension not available.")
        return None

    src = _HERE / "paged_kv_manager.cpp"
    if not src.exists():
        _log.warning("paged_kv_manager.cpp not found at %s", src)
        return None

    build_dir = pathlib.Path(os.environ.get(
        "HYPERION_BUILD_DIR",
        "/tmp/hyperion_build/paged_kv_manager",
    ))
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        ext = load(
            name="paged_kv_manager",
            sources=[str(src)],
            build_directory=str(build_dir),
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-DTORCH_EXTENSION_NAME=paged_kv_manager",
            ],
            verbose=False,
            with_cuda=False,
        )
        _log.info("paged_kv_manager C++ extension loaded successfully.")
        return ext.PagedKVManager  # type: ignore[attr-defined]
    except Exception as exc:
        _log.debug("C++ extension compile failed: %s – using Python fallback.", exc)
        return None


def load_paged_kv_manager() -> type:
    """
    Return the ``PagedKVManager`` class.

    Tries the C++ lock-free extension first; falls back to the pure-Python
    implementation if compilation is unavailable.

    Returns
    -------
    type  – ``PagedKVManager`` (C++) or ``_PyPagedKVManager`` (Python fallback)
    """
    global _manager_class
    if _manager_class is not None:
        return _manager_class

    with _load_lock:
        if _manager_class is not None:
            return _manager_class

        cls = _try_compile_cpp_extension()
        if cls is None:
            _log.info(
                "Using pure-Python PagedKVManager fallback "
                "(C++ extension not available)."
            )
            cls = _PyPagedKVManager

        _manager_class = cls
        return _manager_class


# ---------------------------------------------------------------------------
# Module-level convenience alias
# ---------------------------------------------------------------------------
# Expose at import time so users can do:
#   from hyperion_halo_production.core.paged_kv_manager_loader import PagedKVManager

def __getattr__(name: str):
    if name == "PagedKVManager":
        return load_paged_kv_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
