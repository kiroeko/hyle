#pragma once

#include <cstddef>
#include <memory_resource>
#include <new>

#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"

// std::pmr::memory_resource backed by a HYLE Allocator. Lets any pmr-aware std
// container (std::pmr::vector / string / unordered_map, ...) route allocations
// through a HYLE Allocator. PMR erases the element type and is preferable when
// the consumer accepts memory_resource; StdAllocator<T> separately serves
// legacy and third-party allocator-template APIs that PMR cannot satisfy.
//
// Boundary note (conventions §0/§5 sanctioned exception): do_allocate THROWS
// std::bad_alloc on failure. That is the documented memory_resource contract --
// std::pmr containers depend on it to report OOM. This is an adapter boundary to
// std (which expects throwing), not HYLE's internal error path; the throw surfaces
// to the std container that requested the allocation, never back into HYLE code.

namespace hyle::mem {

// Process-wide PMR over the compile-time SystemAllocator path.  Prefer this
// for ordinary std::pmr containers that use HYLE's default host heap: it pays
// PMR's required virtual dispatch and the shared-library import, but does not
// stack an additional Allocator function-pointer dispatch on top.  The
// returned singleton is never null and has process lifetime; do not delete it.
// Use PmrResource below when runtime backing injection is the actual need.
HYLE_MEM_API std::pmr::memory_resource* system_memory_resource() noexcept;

namespace detail {
inline bool allocator_identity_equal(const Allocator& a,
                                     const Allocator& b) noexcept {
  return a.alloc_fn() == b.alloc_fn() && a.free_fn() == b.free_fn() &&
         a.realloc_fn() == b.realloc_fn() && a.state() == b.state();
}
}  // namespace detail

class PmrResource : public std::pmr::memory_resource {
 public:
  explicit PmrResource(Allocator alloc) noexcept : alloc_(alloc) {
    HYLE_FATAL_ASSERT(alloc_.valid());
  }

  Allocator allocator() const noexcept { return alloc_; }

 private:
  void* do_allocate(std::size_t bytes, std::size_t align) override {
    void* p = alloc_.allocate(bytes, align);
    if (p == nullptr) throw std::bad_alloc{};
    return p;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t align) override {
    alloc_.deallocate(p, bytes, align);
  }

  // Conservative identity for the complete exposed Allocator backing
  // {alloc,free,realloc,state}. PMR itself only exercises alloc/free, so ignoring realloc
  // could be deallocation-safe, but false is permitted and safer than declaring two
  // observably different allocator() values to be the same resource.
  bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    const auto* o = dynamic_cast<const PmrResource*>(&other);
    return o != nullptr && detail::allocator_identity_equal(o->alloc_, alloc_);
  }

  Allocator alloc_;
};

// PMR adapter that raises every allocation's BASE alignment to at least the
// caller-selected power of two. This is the std-container shape corresponding
// to oneTBB's cache_aligned_resource without forcing a cache policy into the
// general PmrResource path. It aligns the container buffer, not every element;
// callers that need per-element cache-line isolation must still pad the type.
class AlignedPmrResource : public std::pmr::memory_resource {
 public:
  AlignedPmrResource(Allocator alloc, std::size_t minimum_alignment) noexcept
      : alloc_(alloc), minimum_alignment_(minimum_alignment) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(is_pow2(minimum_alignment_));
  }

  Allocator allocator() const noexcept { return alloc_; }
  std::size_t minimum_alignment() const noexcept { return minimum_alignment_; }

 private:
  std::size_t effective(std::size_t requested) const noexcept {
    return requested < minimum_alignment_ ? minimum_alignment_ : requested;
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* p = alloc_.allocate(bytes, effective(alignment));
    if (p == nullptr) throw std::bad_alloc{};
    return p;
  }

  void do_deallocate(void* p, std::size_t bytes,
                     std::size_t alignment) override {
    alloc_.deallocate(p, bytes, effective(alignment));
  }

  bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    const auto* o = dynamic_cast<const AlignedPmrResource*>(&other);
    return o != nullptr && o->minimum_alignment_ == minimum_alignment_ &&
           detail::allocator_identity_equal(o->alloc_, alloc_);
  }

  Allocator alloc_;
  std::size_t minimum_alignment_;
};

}  // namespace hyle::mem
