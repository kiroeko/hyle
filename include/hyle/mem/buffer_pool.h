#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/visibility.h"

namespace hyle::mem {

struct BufferPoolOptions {
  // Hard retention bounds. Zero disables caching while preserving the same
  // acquire/lease API.
  std::size_t max_cached_bytes = std::size_t{1} << 30;
  std::size_t max_cached_buffers = 256;

  // A cached buffer is reusable only when capacity-request <= this bound.
  // This prevents a tiny request from pinning a very large device allocation.
  std::size_t max_reuse_slack = std::size_t{4} << 20;

  // New allocations reserve this percentage of headroom, then round to the
  // power-of-two allocation_granularity. Five percent mirrors proven
  // inference-buffer growth without committing to power-of-two capacities.
  std::uint8_t growth_percent = 5;
  std::size_t allocation_granularity = 256;

  // Per size bucket, bound the best-fit/alignment inspection. A miss allocates
  // a new buffer instead of turning a long cache into an unbounded hot scan.
  std::uint8_t max_entries_scanned_per_bucket = 8;
};

struct BufferPoolStats {
  std::size_t cached_bytes = 0;
  std::size_t cached_buffers = 0;
  std::size_t live_bytes = 0;
  std::size_t live_buffers = 0;
  std::size_t peak_live_bytes = 0;
  std::uint64_t provider_allocations = 0;
  std::uint64_t cache_reuses = 0;
  std::uint64_t oom_trim_retries = 0;
  std::uint64_t discarded_on_return = 0;
};

class BufferPool;

// Move-only checkout from BufferPool. Destruction returns the underlying
// allocation to the pool. The caller must wait for all operations using the
// view before reset/destruction; Mem deliberately carries no Event/Stream.
class BufferLease {
 public:
  constexpr BufferLease() noexcept = default;
  BufferLease(const BufferLease&) = delete;
  BufferLease& operator=(const BufferLease&) = delete;
  HYLE_MEM_API BufferLease(BufferLease&& other) noexcept;
  HYLE_MEM_API BufferLease& operator=(BufferLease&& other) noexcept;
  HYLE_MEM_API ~BufferLease();

  constexpr bool valid() const noexcept { return entry_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr BufferView view() const noexcept { return view_; }
  std::size_t size() const noexcept { return view_.size(); }
  HYLE_MEM_API std::size_t capacity() const noexcept;

  // Return to the pool now. Idempotent; provider deallocation errors do not
  // exist in MemoryResource's contract.
  HYLE_MEM_API void reset() noexcept;

  // Remove this allocation from pool ownership and return an ordinary Buffer.
  // The caller then owns provider deallocation; the pool's live statistics are
  // updated exactly as for reset(), but the buffer is not cached.
  HYLE_MEM_API Buffer take() noexcept;

 private:
  friend class BufferPool;
  BufferLease(void* pool, void* entry, BufferView view) noexcept
      : pool_(pool), entry_(entry), view_(view) {}

  void* pool_ = nullptr;
  void* entry_ = nullptr;
  BufferView view_{};
};

// Thread-safe transient Buffer cache over one long-lived MemoryResourceRef.
// The pool must outlive every lease. Common same-shape reuse takes one mutex,
// O(1) bucket selection and a bounded list scan; no wrapper/provider allocation
// occurs. trim() and destruction are cold O(cached buffers).
class BufferPool {
 public:
  HYLE_MEM_API static Result<BufferPool> create(
      MemoryResourceRef resource, BufferPoolOptions options = {}) noexcept;

  constexpr BufferPool() noexcept = default;
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  HYLE_MEM_API BufferPool(BufferPool&& other) noexcept;
  HYLE_MEM_API BufferPool& operator=(BufferPool&& other) noexcept;
  HYLE_MEM_API ~BufferPool();

  bool valid() const noexcept { return impl_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  HYLE_MEM_API Result<BufferLease> acquire(
      std::size_t size,
      std::size_t alignment = alignof(std::max_align_t)) noexcept;
  HYLE_MEM_API BufferPoolStats stats() const noexcept;
  HYLE_MEM_API MemoryResourceRef resource() const noexcept;

  // Release cached buffers until cached_bytes <= target. target==0 clears the
  // cache. Live leases are unaffected and remain charged in live_* stats.
  HYLE_MEM_API void trim(std::size_t target_cached_bytes = 0) noexcept;

 private:
  explicit BufferPool(void* impl) noexcept : impl_(impl) {}
  void* impl_ = nullptr;
};

static_assert(sizeof(BufferPool) == sizeof(void*));

}  // namespace hyle::mem
