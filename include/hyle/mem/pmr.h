#pragma once

#include <cstddef>
#include <memory_resource>
#include <new>

#include "hyle/mem/allocator.h"

// std::pmr::memory_resource backed by a HYLE Allocator. Lets any pmr-aware std
// container (std::pmr::vector / string / unordered_map, ...) route allocations
// through a HYLE Allocator. This is the modern C++17 replacement for two clay
// files at once:
//   * utils/stl_allocator<T> (196 lines, per-type) -> ONE resource serves every
//     T (pmr erases the element type), so no rebind boilerplate.
//   * utils/generic_allocator (62 lines) -> HYLE's Allocator value-handle ALREADY
//     is a type-erased {fn,state} forwarder, so that file has no HYLE counterpart.
//
// Boundary note (conventions §0/§5 sanctioned exception): do_allocate THROWS
// std::bad_alloc on failure. That is the documented memory_resource contract --
// std::pmr containers depend on it to report OOM. This is an adapter boundary to
// std (which expects throwing), not HYLE's internal error path; the throw surfaces
// to the std container that requested the allocation, never back into HYLE code.

namespace hyle::mem {

class PmrResource : public std::pmr::memory_resource {
 public:
  explicit PmrResource(Allocator alloc) noexcept : alloc_(alloc) {}

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

  // Two resources are interchangeable iff they wrap the same backing {fn,state}
  // -- then memory allocated by one can be freed by the other.
  bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    const auto* o = dynamic_cast<const PmrResource*>(&other);
    return o != nullptr && o->alloc_.fn() == alloc_.fn() &&
           o->alloc_.state() == alloc_.state();
  }

  Allocator alloc_;
};

}  // namespace hyle::mem
