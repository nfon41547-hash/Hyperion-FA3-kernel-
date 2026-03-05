// =============================================================================
// paged_kv_manager.cpp – Lock-Free Paged KV-Cache Manager
// =============================================================================
//
// A production-grade, thread-safe, lock-free paged KV-cache block manager for
// the Hyperion HALO inference engine.
//
// Design overview
// ---------------
// Free-list (hot path)
//   Uses a Treiber stack with 128-bit ABA-safe tagged pointers.  The 64-bit
//   atomic head packs a 32-bit ABA generation counter in the upper word and a
//   32-bit block-id in the lower word.  Every push/pop is a single CAS loop —
//   no locks, no memory barriers beyond acquire/release.
//
// Block tables (per-sequence, cold path)
//   Stored in an std::unordered_map<string, vector<int32_t>> guarded by a
//   single std::shared_mutex.  Sequence starts/ends are rare compared to
//   token-level hot-path operations, so a single RW-lock is sufficient.
//
// Utilisation counters
//   Two std::atomic<int32_t> values track free and used block counts, updated
//   atomically by every alloc/free call.
//
// Interface (mirrors Python KVCacheAllocator in memory_manager.py)
// ----------------------------------------------------------------
//   PagedKVManager(num_blocks)
//   allocate_blocks(seq_id, n) → list[int]
//   free_sequence(seq_id)
//   get_block_table(seq_id) → list[int]
//   free_block_count()  → int
//   used_block_count()  → int
//   utilization()       → float
//   reset()
//
// Python binding
// --------------
// Exposed via pybind11.  Loaded through paged_kv_manager_loader.py at runtime.
//
// Compilation (standalone, no CUDA needed)
// -----------------------------------------
//   g++ -O3 -std=c++17 -shared -fPIC \
//       -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
//       -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
//       paged_kv_manager.cpp \
//       -o paged_kv_manager$(python3-config --extension-suffix)
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int32_t kInvalidBlock = -1;

// ---------------------------------------------------------------------------
// Packed tagged pointer helpers
// ---------------------------------------------------------------------------
//
// We pack (tag, block_id) into a single uint64_t:
//
//   bits [63:32]  →  ABA generation counter   (uint32_t tag)
//   bits [31: 0]  →  block id, or 0xFFFFFFFF for "null"
//
// ---------------------------------------------------------------------------
inline uint64_t pack(uint32_t tag, int32_t block_id) noexcept {
    return (static_cast<uint64_t>(tag) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(block_id));
}

inline uint32_t unpack_tag(uint64_t val) noexcept {
    return static_cast<uint32_t>(val >> 32);
}

inline int32_t unpack_id(uint64_t val) noexcept {
    return static_cast<int32_t>(static_cast<uint32_t>(val));
}

static constexpr uint64_t kNullPacked =
    (static_cast<uint64_t>(0u) << 32) |
    static_cast<uint64_t>(static_cast<uint32_t>(kInvalidBlock));

// ---------------------------------------------------------------------------
// BlockNode  – one node in the intrusive Treiber stack
// ---------------------------------------------------------------------------
struct BlockNode {
    int32_t block_id{kInvalidBlock};
    // Intrusive "next" packed as (tag, block_id) of the next node.
    // Initialised to kNullPacked by default.
    std::atomic<uint64_t> next{kNullPacked};

    BlockNode() = default;
    explicit BlockNode(int32_t id) : block_id(id) {}

    // Non-copyable / non-movable because atomic members are not copyable.
    BlockNode(const BlockNode&) = delete;
    BlockNode& operator=(const BlockNode&) = delete;
};

// ---------------------------------------------------------------------------
// LockFreeStack  – ABA-safe Treiber stack
// ---------------------------------------------------------------------------
//
// Operations: push / pop  (both O(1) amortised, lock-free)
//
// ABA prevention: each CAS increments the tag by 1.  The tag wraps around
// after 2^32 pops, which is safe in practice for KV-cache workloads.
//
// Node pool is a raw array owned by a unique_ptr to avoid vector-resize
// issues with non-movable std::atomic members.
//
class LockFreeStack {
public:
    explicit LockFreeStack(int num_nodes)
        : _capacity(num_nodes),
          _nodes(std::make_unique<BlockNode[]>(num_nodes)) {
        _head.store(kNullPacked, std::memory_order_relaxed);
    }

    // Initialise node pool: set block_ids and push all onto the free-list in
    // reverse order so block 0 is popped first (deterministic allocation).
    void init(int num_blocks) {
        // Reset head first.
        _head.store(kNullPacked, std::memory_order_relaxed);

        // Build the initial chain (no CAS needed – single-threaded init).
        for (int i = num_blocks - 1; i >= 0; --i) {
            _nodes[i].block_id = i;
            // Link node i → current top.
            _nodes[i].next.store(_head.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            _head.store(pack(0u, i), std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    // Push block_id back onto the stack (e.g. during free).
    // Returns true on success (always succeeds).
    bool push(int32_t block_id) noexcept {
        uint64_t old_head = _head.load(std::memory_order_relaxed);
        BlockNode& node = _nodes[block_id];

        while (true) {
            node.next.store(old_head, std::memory_order_relaxed);
            // New head: tag = old_tag + 1, id = block_id
            uint64_t new_head = pack(unpack_tag(old_head) + 1u, block_id);
            if (_head.compare_exchange_weak(old_head, new_head,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                return true;
            }
            // CAS failed – old_head is refreshed by compare_exchange_weak.
        }
    }

    // Pop one block off the stack.  Returns kInvalidBlock if empty.
    int32_t pop() noexcept {
        uint64_t old_head = _head.load(std::memory_order_acquire);

        while (true) {
            int32_t id = unpack_id(old_head);
            if (id == kInvalidBlock) {
                return kInvalidBlock;  // empty
            }
            // Read the next pointer from the node about to be removed.
            uint64_t next = _nodes[id].next.load(std::memory_order_relaxed);
            // New head: tag = old_tag + 1, id = next's block_id
            uint64_t new_head = pack(unpack_tag(old_head) + 1u,
                                     unpack_id(next));
            if (_head.compare_exchange_weak(old_head, new_head,
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
                return id;
            }
            // Retry with the refreshed old_head.
        }
    }

    // Drain all remaining entries into a vector (for reset).
    std::vector<int32_t> drain() {
        std::vector<int32_t> out;
        int32_t id;
        while ((id = pop()) != kInvalidBlock) {
            out.push_back(id);
        }
        return out;
    }

private:
    const int _capacity;
    std::atomic<uint64_t> _head{kNullPacked};
    std::unique_ptr<BlockNode[]> _nodes;
};

// ---------------------------------------------------------------------------
// PagedKVManager
// ---------------------------------------------------------------------------
class PagedKVManager {
public:
    // Construct a manager for `num_blocks` physical blocks.
    explicit PagedKVManager(int32_t num_blocks)
        : _num_blocks(num_blocks),
          _free_count(num_blocks),
          _used_count(0),
          _stack(num_blocks) {
        if (num_blocks <= 0) {
            throw std::invalid_argument("num_blocks must be > 0");
        }
        _stack.init(num_blocks);
    }

    // -----------------------------------------------------------------------
    // allocate_blocks
    // -----------------------------------------------------------------------
    // Atomically pop `n` blocks from the free-list and record them for seq_id.
    // Raises std::runtime_error on OOM.
    std::vector<int32_t> allocate_blocks(const std::string& seq_id,
                                          int32_t n) {
        if (n <= 0) {
            throw std::invalid_argument("num_blocks must be > 0");
        }

        // Speculatively check free count before doing any pops.
        int32_t current_free =
            _free_count.load(std::memory_order_acquire);
        if (current_free < n) {
            throw std::runtime_error(
                "KV-cache OOM: requested " + std::to_string(n) +
                " blocks, only " + std::to_string(current_free) + " free.");
        }

        // Pop n blocks from the lock-free stack.
        std::vector<int32_t> allocated;
        allocated.reserve(n);
        for (int32_t i = 0; i < n; ++i) {
            int32_t block_id = _stack.pop();
            if (block_id == kInvalidBlock) {
                // Race: another thread stole blocks between our check and pops.
                // Return already-popped blocks before throwing.
                for (int32_t b : allocated) {
                    _stack.push(b);
                }
                _free_count.fetch_add(
                    static_cast<int32_t>(allocated.size()),
                    std::memory_order_relaxed);
                _used_count.fetch_sub(
                    static_cast<int32_t>(allocated.size()),
                    std::memory_order_relaxed);
                throw std::runtime_error(
                    "KV-cache OOM: ran out of blocks during allocation.");
            }
            allocated.push_back(block_id);
            _free_count.fetch_sub(1, std::memory_order_relaxed);
            _used_count.fetch_add(1, std::memory_order_relaxed);
        }

        // Register in the sequence → block-table map (guarded by RW-lock).
        {
            std::unique_lock<std::shared_mutex> lk(_seq_mutex);
            auto& table = _seq_block_tables[seq_id];
            table.insert(table.end(), allocated.begin(), allocated.end());
        }

        return allocated;
    }

    // -----------------------------------------------------------------------
    // free_sequence
    // -----------------------------------------------------------------------
    // Return all blocks belonging to seq_id back to the free-list.
    void free_sequence(const std::string& seq_id) {
        std::vector<int32_t> blocks;
        {
            std::unique_lock<std::shared_mutex> lk(_seq_mutex);
            auto it = _seq_block_tables.find(seq_id);
            if (it == _seq_block_tables.end()) return;
            blocks = std::move(it->second);
            _seq_block_tables.erase(it);
        }

        for (int32_t b : blocks) {
            _stack.push(b);
            _free_count.fetch_add(1, std::memory_order_relaxed);
            _used_count.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    // -----------------------------------------------------------------------
    // get_block_table
    // -----------------------------------------------------------------------
    std::vector<int32_t> get_block_table(const std::string& seq_id) const {
        std::shared_lock<std::shared_mutex> lk(_seq_mutex);
        auto it = _seq_block_tables.find(seq_id);
        if (it == _seq_block_tables.end()) return {};
        return it->second;
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------
    int32_t free_block_count() const noexcept {
        return _free_count.load(std::memory_order_relaxed);
    }

    int32_t used_block_count() const noexcept {
        return _used_count.load(std::memory_order_relaxed);
    }

    float utilization() const noexcept {
        return static_cast<float>(used_block_count()) /
               static_cast<float>(_num_blocks);
    }

    int32_t num_blocks() const noexcept { return _num_blocks; }

    // -----------------------------------------------------------------------
    // reset
    // -----------------------------------------------------------------------
    // Reclaim all blocks and clear all sequence tables.  Not thread-safe –
    // caller must ensure no concurrent alloc/free during reset.
    void reset() {
        // Drain remaining free-list entries.
        _stack.drain();

        // Clear seq tables and return their blocks (silently).
        {
            std::unique_lock<std::shared_mutex> lk(_seq_mutex);
            _seq_block_tables.clear();
        }

        // Re-initialise.
        _free_count.store(_num_blocks, std::memory_order_relaxed);
        _used_count.store(0, std::memory_order_relaxed);
        _stack.init(_num_blocks);
    }

private:
    const int32_t _num_blocks;

    // Lock-free free-list (hot path).
    LockFreeStack _stack;

    // Utilisation counters – updated atomically on every alloc/free.
    std::atomic<int32_t> _free_count;
    std::atomic<int32_t> _used_count;

    // Per-sequence block tables (cold path, protected by RW-lock).
    mutable std::shared_mutex _seq_mutex;
    std::unordered_map<std::string, std::vector<int32_t>> _seq_block_tables;
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(paged_kv_manager, m) {
    m.doc() =
        "Lock-free paged KV-cache block manager for Hyperion HALO.\n\n"
        "Uses a Treiber stack with ABA-safe tagged pointers for the free-list\n"
        "and a shared_mutex-guarded unordered_map for per-sequence block tables.\n"
        "All allocation/deallocation hot-path operations are lock-free.";

    py::class_<PagedKVManager>(m, "PagedKVManager")
        .def(py::init<int32_t>(), py::arg("num_blocks"),
             "Construct a manager for `num_blocks` physical blocks.")

        .def("allocate_blocks",
             &PagedKVManager::allocate_blocks,
             py::arg("seq_id"), py::arg("num_blocks"),
             R"doc(
Atomically pop `num_blocks` entries from the lock-free free-list and
associate them with `seq_id`.

Parameters
----------
seq_id    : str   – unique sequence identifier
num_blocks : int  – number of blocks to allocate

Returns
-------
list[int]  – allocated physical block ids

Raises
------
RuntimeError  on OOM (fewer than `num_blocks` free blocks available)
)doc")

        .def("free_sequence",
             &PagedKVManager::free_sequence,
             py::arg("seq_id"),
             "Return all blocks belonging to `seq_id` back to the free-list.")

        .def("get_block_table",
             &PagedKVManager::get_block_table,
             py::arg("seq_id"),
             "Return a copy of the block table for `seq_id` (empty list if unknown).")

        .def("free_block_count",
             &PagedKVManager::free_block_count,
             "Number of currently unallocated blocks.")

        .def("used_block_count",
             &PagedKVManager::used_block_count,
             "Number of currently allocated blocks.")

        .def("utilization",
             &PagedKVManager::utilization,
             "Fraction of blocks currently allocated (0.0 – 1.0).")

        .def("num_blocks",
             &PagedKVManager::num_blocks,
             "Total number of physical blocks in the pool.")

        .def("reset",
             &PagedKVManager::reset,
             "Reclaim all blocks and clear all sequence tables.")

        .def("__repr__", [](const PagedKVManager& m) {
            return "PagedKVManager(blocks=" +
                   std::to_string(m.used_block_count()) + "/" +
                   std::to_string(m.num_blocks()) +
                   ", util=" +
                   std::to_string(static_cast<int>(m.utilization() * 100)) +
                   "%)";
        });
}
