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
  using Marker = std::byte*;  // a saved cursor position within the slab

  StackAllocator(Allocator alloc, std::size_t size)
      : alloc_(alloc), size_(size) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(size_ >= 1);
    base_ = static_cast<std::byte*>(
        alloc_.allocate(size_, alignof(std::max_align_t)));
    HYLE_FATAL_ASSERT(base_ != nullptr);
    cur_ = base_;
    end_ = base_ + size_;
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
    // Pointer-style bump: align cur_ directly (the live address), no +base/-base
    // round-trip through an offset. base_ is max_align_t-aligned so aligning the
    // real cursor is what guarantees the returned pointer's alignment.
    auto cur = reinterpret_cast<std::uintptr_t>(cur_);
    std::uintptr_t aligned = (cur + alignment - 1) & ~(alignment - 1);
    auto e = reinterpret_cast<std::uintptr_t>(end_);
    if (aligned > e || bytes > e - aligned) return nullptr;  // overflow-safe
    cur_ = reinterpret_cast<std::byte*>(aligned + bytes);
    return reinterpret_cast<void*>(aligned);
  }

  // Save the current cursor; rewind() returns the slab to exactly this point.
  Marker mark() const noexcept { return cur_; }

  // Release everything allocated since `m` (LIFO). Must be within [base_, cur_].
  void rewind(Marker m) noexcept {
    HYLE_FATAL_ASSERT(m >= base_ && m <= cur_);
    cur_ = m;
  }

  // Release everything.
  void reset() noexcept { cur_ = base_; }

  std::size_t used() const noexcept { return static_cast<std::size_t>(cur_ - base_); }
  std::size_t capacity() const noexcept { return size_; }

 private:
  Allocator alloc_;
  std::byte* base_ = nullptr;
  std::size_t size_;
  std::byte* cur_ = nullptr;  // bump cursor (pointer-style: points at next free byte)
  std::byte* end_ = nullptr;  // one past the slab end (base_ + size_)
};

}  // namespace hyle::mem
