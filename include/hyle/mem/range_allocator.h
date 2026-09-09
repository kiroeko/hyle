#pragma once

#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/visibility.h"

namespace hyle::mem {

// Creation-time bounds for a payload-free logical range owner. The live
// allocation limit is deliberately required: metadata never grows on the
// allocate/free path, so callers can budget its exact upper bound.
struct RangeAllocatorOptions {
  std::uint32_t max_allocations = 0;
  AllocatorRef metadata_allocator = system_allocator();
};

struct RangeAllocatorStats {
  std::uint64_t capacity_bytes = 0;
  std::uint64_t allocated_bytes = 0;
  std::uint64_t free_bytes = 0;
  std::uint64_t peak_allocated_bytes = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t outstanding_handles = 0;
  std::uint32_t live_allocations = 0;
  std::uint32_t max_allocations = 0;
};

struct RangeAllocatorDetailedStats {
  RangeAllocatorStats summary{};
  std::uint64_t largest_free_range = 0;
  std::uint32_t free_range_count = 0;
};

class RangeAllocator;

// Move-only ownership of one [offset, offset + size) interval. Destruction
// returns the interval to its allocator. An allocator-wide reset invalidates
// old leases without making their later destruction release a newly reused
// interval. This object owns no bytes and performs no device synchronization.
class RangeAllocation {
 public:
  constexpr RangeAllocation() noexcept = default;
  RangeAllocation(const RangeAllocation&) = delete;
  RangeAllocation& operator=(const RangeAllocation&) = delete;
  RangeAllocation(RangeAllocation&& other) noexcept
      : impl_(other.impl_),
        generation_(other.generation_),
        offset_(other.offset_),
        node_(other.node_) {
    other.impl_ = nullptr;
  }
  RangeAllocation& operator=(RangeAllocation&& other) noexcept {
    if (this != &other) {
      if (impl_ != nullptr) reset();
      impl_ = other.impl_;
      generation_ = other.generation_;
      offset_ = other.offset_;
      node_ = other.node_;
      other.impl_ = nullptr;
    }
    return *this;
  }
  ~RangeAllocation() {
    if (impl_ != nullptr) reset();
  }

  HYLE_MEM_API bool valid() const noexcept;
  explicit operator bool() const noexcept { return valid(); }

  // These observers require valid()==true. reset() and owner-wide reset make
  // the old logical offset immediately unusable.
  HYLE_FORCE_INLINE std::uint64_t offset() const noexcept {
    HYLE_ASSUME(impl_ != nullptr);
    return offset_;
  }
  HYLE_MEM_API std::uint64_t size() const noexcept;

  // Return this interval now. Idempotent. If the owner already performed a
  // bulk reset or was destroyed, this only releases the retained control ref.
  HYLE_MEM_API void reset() noexcept;

 private:
  friend class RangeAllocator;
  RangeAllocation(void* impl, std::uint32_t node, std::uint64_t generation,
                  std::uint64_t offset) noexcept
      : impl_(impl),
        generation_(generation),
        offset_(offset),
        node_(node) {}

  void* impl_ = nullptr;
  std::uint64_t generation_ = 0;
  std::uint64_t offset_ = 0;
  std::uint32_t node_ = 0;
};

static_assert(sizeof(RangeAllocation) == 32);

// Caller-synchronized allocator for offsets inside one opaque backing range.
// It owns metadata only: no host/device payload is allocated, mapped or
// touched. The control block stays alive until the final outstanding
// RangeAllocation is destroyed, even if this owner is destroyed first.
class RangeAllocator {
 public:
  HYLE_MEM_API static Result<RangeAllocator> create(
      std::uint64_t capacity, RangeAllocatorOptions options) noexcept;

  constexpr RangeAllocator() noexcept = default;
  RangeAllocator(const RangeAllocator&) = delete;
  RangeAllocator& operator=(const RangeAllocator&) = delete;
  HYLE_MEM_API RangeAllocator(RangeAllocator&& other) noexcept;
  HYLE_MEM_API RangeAllocator& operator=(RangeAllocator&& other) noexcept;
  HYLE_MEM_API ~RangeAllocator();

  constexpr bool valid() const noexcept { return impl_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }

  HYLE_MEM_API Result<RangeAllocation> allocate(
      std::uint64_t size, std::uint64_t alignment = 1) noexcept;

  // Quiescent bulk invalidation. Existing RangeAllocation objects become
  // invalid but may safely be destroyed later. Peak accounting is retained.
  HYLE_MEM_API void reset() noexcept;

  HYLE_MEM_API RangeAllocatorStats stats() const noexcept;
  // Cold O(number of metadata nodes) fragmentation observation.
  HYLE_MEM_API RangeAllocatorDetailedStats detailed_stats() const noexcept;

 private:
  explicit RangeAllocator(void* impl) noexcept : impl_(impl) {}
  void* impl_ = nullptr;
};

static_assert(sizeof(RangeAllocator) == sizeof(void*));

}  // namespace hyle::mem
