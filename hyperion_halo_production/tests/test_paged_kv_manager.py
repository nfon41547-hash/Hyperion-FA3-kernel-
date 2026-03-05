"""
test_paged_kv_manager.py – Tests for the lock-free PagedKVManager.

The test suite is backend-agnostic: it runs identically against the C++
extension (when available) and the pure-Python fallback.  Both implementations
must pass every test.

Run with:
    pytest hyperion_halo_production/tests/test_paged_kv_manager.py -v
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import List

import pytest

from hyperion_halo_production.core.paged_kv_manager_loader import (
    load_paged_kv_manager,
    _PyPagedKVManager,
)

# ---------------------------------------------------------------------------
# Parametrize over both backends so every test runs twice.
# ---------------------------------------------------------------------------
def _backends():
    impls = [pytest.param(_PyPagedKVManager, id="python_fallback")]
    cpp_cls = load_paged_kv_manager()
    if cpp_cls is not _PyPagedKVManager:
        impls.append(pytest.param(cpp_cls, id="cpp_lockfree"))
    return impls


@pytest.fixture(params=_backends())
def Mgr(request):
    """Fixture that yields a PagedKVManager *class* for the current backend."""
    return request.param


# ---------------------------------------------------------------------------
# Basic functional tests
# ---------------------------------------------------------------------------
class TestBasicAllocation:
    def test_initial_state(self, Mgr):
        mgr = Mgr(64)
        assert mgr.free_block_count() == 64
        assert mgr.used_block_count() == 0
        assert mgr.num_blocks() == 64
        assert abs(mgr.utilization()) < 1e-6

    def test_allocate_returns_correct_count(self, Mgr):
        mgr = Mgr(64)
        blocks = mgr.allocate_blocks("s0", 8)
        assert len(blocks) == 8

    def test_block_ids_in_valid_range(self, Mgr):
        mgr = Mgr(64)
        blocks = mgr.allocate_blocks("s0", 16)
        for b in blocks:
            assert 0 <= b < 64

    def test_no_duplicate_block_ids(self, Mgr):
        mgr = Mgr(128)
        a = mgr.allocate_blocks("s0", 64)
        b = mgr.allocate_blocks("s1", 64)
        all_blocks = a + b
        assert len(set(all_blocks)) == 128, "Every block id must be unique"

    def test_counters_after_alloc(self, Mgr):
        mgr = Mgr(32)
        mgr.allocate_blocks("s0", 10)
        assert mgr.used_block_count() == 10
        assert mgr.free_block_count() == 22

    def test_utilization(self, Mgr):
        mgr = Mgr(100)
        mgr.allocate_blocks("s0", 25)
        assert abs(mgr.utilization() - 0.25) < 1e-5

    def test_free_restores_count(self, Mgr):
        mgr = Mgr(64)
        mgr.allocate_blocks("s0", 16)
        mgr.free_sequence("s0")
        assert mgr.free_block_count() == 64
        assert mgr.used_block_count() == 0

    def test_freed_blocks_are_reusable(self, Mgr):
        mgr = Mgr(16)
        first = mgr.allocate_blocks("s0", 16)
        mgr.free_sequence("s0")
        second = mgr.allocate_blocks("s1", 16)
        assert set(first) == set(second)

    def test_block_table_matches_returned_ids(self, Mgr):
        mgr = Mgr(64)
        blocks = mgr.allocate_blocks("s0", 8)
        table = mgr.get_block_table("s0")
        assert sorted(blocks) == sorted(table)

    def test_get_block_table_unknown_seq(self, Mgr):
        mgr = Mgr(32)
        assert mgr.get_block_table("nonexistent") == []

    def test_free_unknown_seq_no_crash(self, Mgr):
        mgr = Mgr(32)
        mgr.free_sequence("unknown")  # should not raise

    def test_multiple_sequences(self, Mgr):
        mgr = Mgr(64)
        mgr.allocate_blocks("seq_A", 10)
        mgr.allocate_blocks("seq_B", 20)
        assert mgr.used_block_count() == 30
        mgr.free_sequence("seq_A")
        assert mgr.used_block_count() == 20
        mgr.free_sequence("seq_B")
        assert mgr.used_block_count() == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_oom_raises_runtime_error(self, Mgr):
        mgr = Mgr(4)
        with pytest.raises(RuntimeError, match="OOM|out of blocks"):
            mgr.allocate_blocks("s0", 5)

    def test_oom_leaves_state_consistent(self, Mgr):
        """After a failed allocation the manager must remain fully usable."""
        mgr = Mgr(4)
        try:
            mgr.allocate_blocks("s0", 5)
        except RuntimeError:
            pass
        # No blocks should have been consumed.
        assert mgr.free_block_count() == 4
        # Subsequent valid allocation should succeed.
        blocks = mgr.allocate_blocks("s1", 4)
        assert len(blocks) == 4

    def test_exact_capacity_allocation(self, Mgr):
        mgr = Mgr(8)
        blocks = mgr.allocate_blocks("s0", 8)
        assert len(blocks) == 8
        assert mgr.free_block_count() == 0

    def test_allocate_after_full_raises(self, Mgr):
        mgr = Mgr(4)
        mgr.allocate_blocks("s0", 4)
        with pytest.raises(RuntimeError):
            mgr.allocate_blocks("s1", 1)

    def test_invalid_num_blocks_constructor(self, Mgr):
        with pytest.raises((ValueError, Exception)):
            Mgr(0)

    def test_invalid_num_blocks_allocate(self, Mgr):
        mgr = Mgr(16)
        with pytest.raises((ValueError, Exception)):
            mgr.allocate_blocks("s0", 0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_clears_everything(self, Mgr):
        mgr = Mgr(64)
        mgr.allocate_blocks("s0", 32)
        mgr.allocate_blocks("s1", 16)
        mgr.reset()
        assert mgr.free_block_count() == 64
        assert mgr.used_block_count() == 0
        assert mgr.get_block_table("s0") == []
        assert mgr.get_block_table("s1") == []

    def test_allocate_after_reset(self, Mgr):
        mgr = Mgr(16)
        mgr.allocate_blocks("s0", 16)
        mgr.reset()
        blocks = mgr.allocate_blocks("s1", 16)
        assert len(blocks) == 16


# ---------------------------------------------------------------------------
# Thread-safety: concurrent allocations
# ---------------------------------------------------------------------------
class TestConcurrency:
    _NUM_THREADS = 8
    _BLOCKS_PER_THREAD = 16
    _TOTAL = _NUM_THREADS * _BLOCKS_PER_THREAD

    def test_concurrent_alloc_no_duplicates(self, Mgr):
        """All thread-level allocations must produce disjoint block sets."""
        mgr = Mgr(self._TOTAL)
        results: List[List[int]] = [[] for _ in range(self._NUM_THREADS)]
        errors: List[Exception] = []

        def worker(idx: int):
            try:
                results[idx] = mgr.allocate_blocks(
                    f"seq_{idx}", self._BLOCKS_PER_THREAD
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(self._NUM_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Worker errors: {errors}"
        all_ids = [b for sub in results for b in sub]
        assert len(all_ids) == self._TOTAL
        assert len(set(all_ids)) == self._TOTAL, "Duplicate block ids detected"

    def test_concurrent_alloc_free_no_leak(self, Mgr):
        """Interleaved alloc/free must leave the pool fully recovered."""
        num_blocks = 128
        mgr = Mgr(num_blocks)
        rounds = 10
        threads_per_round = 8
        blocks_per_thread = num_blocks // threads_per_round

        def worker(round_idx: int, t_idx: int):
            seq = f"r{round_idx}_t{t_idx}"
            mgr.allocate_blocks(seq, blocks_per_thread)
            # Hold briefly then release.
            time.sleep(0.001)
            mgr.free_sequence(seq)

        for r in range(rounds):
            ts = [
                threading.Thread(target=worker, args=(r, t))
                for t in range(threads_per_round)
            ]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

        # After all rounds every block should be free.
        assert mgr.free_block_count() == num_blocks
        assert mgr.used_block_count() == 0

    def test_concurrent_alloc_partial_pool(self, Mgr):
        """
        Multiple threads competing for a limited pool; some will succeed,
        some may get OOM – but there must be no block-id duplicates among
        the successful ones.
        """
        num_blocks = 32
        mgr = Mgr(num_blocks)
        num_threads = 10
        blocks_per = 4  # 10 × 4 = 40 > 32, so some threads will OOM

        results: List[List[int]] = []
        result_lock = threading.Lock()

        def worker():
            try:
                thread_ident = threading.current_thread().ident
                blks = mgr.allocate_blocks(f"seq_{thread_ident}", blocks_per)
                with result_lock:
                    results.append(blks)
            except RuntimeError:
                pass

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_ids = [b for r in results for b in r]
        assert len(set(all_ids)) == len(all_ids), "Duplicate block ids across threads"
        assert len(all_ids) <= num_blocks


# ---------------------------------------------------------------------------
# Repr / introspection
# ---------------------------------------------------------------------------
class TestRepr:
    def test_repr_contains_block_info(self, Mgr):
        mgr = Mgr(64)
        mgr.allocate_blocks("s0", 16)
        r = repr(mgr)
        # Must contain some numeric info; exact format may differ between backends.
        assert any(ch.isdigit() for ch in r)
