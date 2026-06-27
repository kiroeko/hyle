#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {

// LIFO bump allocator over a single fixed slab. Allocations just advance a
// pointer; release is bulk via reset() (whole slab) or rewind(mark) (back to a
// saved point). Ideal for frame/temporary lifetimes (e.g. one taskgraph pass).
//
// Fixes clay bug #4 (stack_allocator.h:28,30,108): clay's copy ctor left
// allocator_ uninitialized -> a copied StackAllocator dereferenced a wild
// pointer on the heap-fallback path. Root-cause fix: this type OWNS a slab and
// is non-copyable/non-movable (an allocator holding a resource should never be
// value-copied). And there is NO heap-fallback path -- full returns nullptr.
//
// Single-threaded by design (stack/scratch lifetime is thread-local).
class StackAllocator {
 public:
  using Marker = std::size_t;  // an offset into the slab

  StackAllocator(Allocator alloc, std::size_t size)
      : alloc_(alloc), size_(size) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(size_ >= 1);
    base_ = static_cast<std::byte*>(
        alloc_.allocate(size_, alignof(std::max_align_t)));
    HYLE_FATAL_ASSERT(base_ != nullptr);
  }

  ~StackAllocator() {
    alloc_.deallocate(base_, size_, alignof(std::max_align_t));
  }

  HYLE_DISABLE_COPY_AND_MOVE(StackAllocator);

  // Bump `bytes` aligned to `alignment` (power of two). nullptr if it won't fit
  // (no heap fallback -- give a bigger slab instead).
  void* allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT((alignment & (alignment - 1)) == 0);
    // Align the REAL address, not just the offset: base_ is only max_align_t-
    // aligned, so for alignment > that, aligning top_ alone would not align the
    // returned pointer.
    auto base = reinterpret_cast<std::uintptr_t>(base_);
    std::uintptr_t cur = base + top_;
    std::uintptr_t aligned = (cur + alignment - 1) & ~(alignment - 1);
    std::size_t new_top = static_cast<std::size_t>(aligned - base);
    if (new_top > size_ || bytes > size_ - new_top) return nullptr;  // overflow-safe
    void* p = base_ + new_top;
    top_ = new_top + bytes;
    return p;
  }

  // Save the current top; rewind() returns the slab to exactly this point.
  Marker mark() const noexcept { return top_; }

  // Release everything allocated since `m` (LIFO). Must be <= current top.
  void rewind(Marker m) noexcept {
    HYLE_FATAL_ASSERT(m <= top_);
    top_ = m;
  }

  // Release everything.
  void reset() noexcept { top_ = 0; }

  std::size_t used() const noexcept { return top_; }
  std::size_t capacity() const noexcept { return size_; }

 private:
  Allocator alloc_;
  std::byte* base_ = nullptr;
  std::size_t size_;
  std::size_t top_ = 0;  // bytes consumed from base_
};

}  // namespace hyle::mem
