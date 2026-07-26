#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

// Allocation counters + optional leak map (mem-06 stats slice).
//
// clay's system allocator stats used unsigned (new_size - old_size) and
// (new?1:0)-(old?1:0) on free → wrap to huge values (the 32/64 "underflow"
// in the asset list). HYLE never does unsigned free deltas: free is always a
// separate subtract of the known old size.
//
// Production hot path (system_allocator / SystemAllocator) is UNINSTRUMENTED.
// Stats are opt-in via StatsAllocator wrapping any Allocator — tests and
// debug tooling pay the cost; release default does not.

namespace hyle::mem {

// Plain POD copy of the atomic fields. During concurrent mutation each field is
// individually race-free but the fields need not describe one coherent instant.
// After callers quiesce the wrapper, the snapshot is exact and live_* should be
// zero when every allocation has been freed.
struct AllocStatsSnapshot {
  std::uint64_t live_count = 0;             // outstanding blocks
  std::uint64_t live_bytes = 0;             // outstanding bytes
  std::uint64_t total_allocs = 0;           // successful allocate calls
  std::uint64_t total_frees = 0;            // deallocate of non-null
  std::uint64_t total_bytes_allocated = 0;  // cumulative bytes at alloc time
  std::uint64_t peak_live_bytes = 0;        // high-water live_bytes
  std::uint64_t reallocs = 0;               // reallocate calls (any size change path)
};

// Thread-safe counters. All fields are 64-bit atomics (no 32-bit wrap).
class AllocStats {
 public:
  AllocStats() noexcept = default;
  HYLE_DISABLE_COPY(AllocStats);

  // Successful allocate of `bytes` (>0). size==0 allocate is a no-op for stats.
  void record_alloc(std::size_t bytes) noexcept {
    if (bytes == 0) return;
    live_count_.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t next =
        live_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    total_allocs_.fetch_add(1, std::memory_order_relaxed);
    total_bytes_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    bump_peak_(next);
  }

  // Deallocate of a live block of `bytes`. size==0 / null free: no-op.
  // Uses subtract-of-known-size (never unsigned (0 - old)).
  void record_free(std::size_t bytes) noexcept {
    if (bytes == 0) return;
    live_count_.fetch_sub(1, std::memory_order_relaxed);
    live_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    total_frees_.fetch_add(1, std::memory_order_relaxed);
  }

  // Realloc accounting. Matches Allocator contract:
  //   old==0, new>0  → new live block (does NOT bump total_allocs; that is for
  //                    allocate() only — reallocs_ counts this call)
  //   old>0,  new==0 → free
  //   both >0        → resize same live slot (count unchanged)
  // Never computes unsigned (new - old) when new < old (clay underflow).
  void record_realloc(std::size_t old_bytes, std::size_t new_bytes) noexcept {
    reallocs_.fetch_add(1, std::memory_order_relaxed);
    if (old_bytes == 0 && new_bytes == 0) return;
    if (old_bytes == 0) {
      if (new_bytes == 0) return;
      live_count_.fetch_add(1, std::memory_order_relaxed);
      const std::uint64_t next =
          live_bytes_.fetch_add(new_bytes, std::memory_order_relaxed) +
          new_bytes;
      total_bytes_allocated_.fetch_add(new_bytes, std::memory_order_relaxed);
      bump_peak_(next);
      return;
    }
    if (new_bytes == 0) {
      live_count_.fetch_sub(1, std::memory_order_relaxed);
      live_bytes_.fetch_sub(old_bytes, std::memory_order_relaxed);
      total_frees_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (new_bytes > old_bytes) {
      const std::size_t d = new_bytes - old_bytes;
      const std::uint64_t next =
          live_bytes_.fetch_add(d, std::memory_order_relaxed) + d;
      total_bytes_allocated_.fetch_add(d, std::memory_order_relaxed);
      bump_peak_(next);
    } else if (old_bytes > new_bytes) {
      live_bytes_.fetch_sub(old_bytes - new_bytes, std::memory_order_relaxed);
    }
  }

  AllocStatsSnapshot snapshot() const noexcept {
    AllocStatsSnapshot s;
    s.live_count = live_count_.load(std::memory_order_relaxed);
    s.live_bytes = live_bytes_.load(std::memory_order_relaxed);
    s.total_allocs = total_allocs_.load(std::memory_order_relaxed);
    s.total_frees = total_frees_.load(std::memory_order_relaxed);
    s.total_bytes_allocated = total_bytes_allocated_.load(std::memory_order_relaxed);
    s.peak_live_bytes = peak_live_bytes_.load(std::memory_order_relaxed);
    s.reallocs = reallocs_.load(std::memory_order_relaxed);
    return s;
  }

  void reset() noexcept {
    // A reset racing allocations cannot define a coherent new accounting
    // epoch. Fail fast on the observable lifetime violation; callers must also
    // provide external quiescence before entering this method.
    HYLE_FATAL_ASSERT(balanced());
    live_count_.store(0, std::memory_order_relaxed);
    live_bytes_.store(0, std::memory_order_relaxed);
    total_allocs_.store(0, std::memory_order_relaxed);
    total_frees_.store(0, std::memory_order_relaxed);
    total_bytes_allocated_.store(0, std::memory_order_relaxed);
    peak_live_bytes_.store(0, std::memory_order_relaxed);
    reallocs_.store(0, std::memory_order_relaxed);
  }

  bool balanced() const noexcept {
    return live_count_.load(std::memory_order_relaxed) == 0 &&
           live_bytes_.load(std::memory_order_relaxed) == 0;
  }

 private:
  void bump_peak_(std::uint64_t cur) noexcept {
    std::uint64_t peak = peak_live_bytes_.load(std::memory_order_relaxed);
    while (cur > peak &&
           !peak_live_bytes_.compare_exchange_weak(peak, cur,
                                                   std::memory_order_relaxed)) {
    }
  }

  std::atomic<std::uint64_t> live_count_{0};
  std::atomic<std::uint64_t> live_bytes_{0};
  std::atomic<std::uint64_t> total_allocs_{0};
  std::atomic<std::uint64_t> total_frees_{0};
  std::atomic<std::uint64_t> total_bytes_allocated_{0};
  std::atomic<std::uint64_t> peak_live_bytes_{0};
  std::atomic<std::uint64_t> reallocs_{0};
};

// One outstanding allocation for leak reports.
struct LeakEntry {
  void* ptr = nullptr;
  std::size_t size = 0;
};

// Optional ptr→size map. Debug/tooling only (lock + hash). Not for prod hot path.
class HYLE_MEM_API LeakTracker {
 public:
  // Metadata and every hash-table node use `metadata`. Construction failure is
  // an operational resource error, not an exception from the mem API.
  static Result<LeakTracker> create(
      Allocator metadata = system_allocator()) noexcept;
  ~LeakTracker();
  HYLE_DISABLE_COPY(LeakTracker);
  LeakTracker(LeakTracker&& o) noexcept;
  LeakTracker& operator=(LeakTracker&& o) noexcept;

  bool valid() const noexcept { return impl_ != nullptr; }

  void track_alloc(void* p, std::size_t size) noexcept;
  // Erase a tracked allocation. Returns false if ptr was not tracked
  // (double-free / foreign ptr). When non-null, *out_size receives the size
  // recorded at track_alloc time (untouched on a false return) — lets callers
  // account frees by the RECORDED size, robust to a wrong size passed to free.
  bool track_free(void* p, std::size_t* out_size = nullptr) noexcept;
  std::size_t outstanding() const noexcept;
  // Number of allocation records that could not be inserted because the
  // injected metadata allocator was exhausted. Nonzero means the inventory is
  // conservative/incomplete since the last clear(); counters remain exact.
  std::uint64_t dropped_records() const noexcept;
  bool complete() const noexcept { return dropped_records() == 0; }
  // Copy up to `cap` live entries into `out`. Returns total outstanding
  // known entries (may be > cap if truncated). Check complete() before
  // treating this as an exhaustive leak report.
  std::size_t copy_leaks(LeakEntry* out, std::size_t cap) const noexcept;
  // Clears known entries and the historical dropped-record count. Call only
  // when no allocation from the measured allocator remains live.
  void clear() noexcept;

 private:
  friend class StatsAllocator;
  LeakTracker() noexcept = default;
  void destroy() noexcept;

  struct Impl;
  Impl* impl_ = nullptr;
};

// Decorator: every allocate/free/realloc updates AllocStats (+ optional leaks).
// Owns the vtable state; Allocator from get() is valid only while *this lives.
class HYLE_MEM_API StatsAllocator {
 public:
  // track_leaks=true enables LeakTracker (slower; for leak tests). `backing`
  // remains the allocator being measured; stable wrapper state and optional
  // tracker nodes use the separate `metadata` allocator. This separation lets
  // StatsAllocator wrap a fixed-block/strict-quota allocator whose contract
  // cannot satisfy arbitrary control allocations. No global-new or exception
  // side channel is used for construction.
  static Result<StatsAllocator> create(Allocator backing,
                                       bool track_leaks = false,
                                       Allocator metadata =
                                           system_allocator()) noexcept;
  ~StatsAllocator();
  HYLE_DISABLE_COPY(StatsAllocator);
  StatsAllocator(StatsAllocator&& o) noexcept;
  StatsAllocator& operator=(StatsAllocator&& o) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  Allocator get() const noexcept;
  AllocStats& stats() noexcept;
  const AllocStats& stats() const noexcept;
  // Per-field atomic observation during mutation; coherent after callers
  // quiesce every allocator handle borrowing this wrapper.
  AllocStatsSnapshot snapshot() const noexcept { return stats().snapshot(); }
  bool balanced() const noexcept { return stats().balanced(); }
  // Null when constructed with track_leaks=false.
  LeakTracker* leaks() noexcept;
  const LeakTracker* leaks() const noexcept;

 private:
  StatsAllocator() noexcept = default;
  void destroy() noexcept;

  struct State;
  State* state_ = nullptr;
};

}  // namespace hyle::mem
