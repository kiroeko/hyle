#pragma once

#include <cstddef>
#include <cstring> // std::memcpy for as_allocator()'s realloc

#include "hyle/core/compiler.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/region.h"

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
  using Marker = std::byte *; // a saved cursor position within the slab

  StackAllocator(Allocator alloc, std::size_t size)
      : alloc_(alloc), size_(size) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(size_ >= 1);
    base_ = static_cast<std::byte *>(
        alloc_.allocate(size_, alignof(std::max_align_t)));
    // slab-OOM -> invalid (base_==nullptr); the caller checks valid(). Not
    // abort, per mem-audit-followup #2 arch B. allocate() naturally returns
    // nullptr while invalid (the shared bump cursor has no storage, so nothing
    // fits).
    cursor_.reset_storage(base_, base_ != nullptr ? size_ : 0);
  }

  // False iff the ctor's slab allocation hit OOM. An invalid StackAllocator
  // hands out nothing (allocate -> nullptr).
  bool valid() const noexcept { return base_ != nullptr; }

  ~StackAllocator() {
    if (base_ != nullptr)
      alloc_.deallocate(base_, size_, alignof(std::max_align_t));
  }

  StackAllocator(const StackAllocator&) = delete;
  StackAllocator& operator=(const StackAllocator&) = delete;
  StackAllocator(StackAllocator&&) = delete;
  StackAllocator& operator=(StackAllocator&&) = delete;

  // Bump `bytes` aligned to `alignment` (power of two). nullptr if it won't fit
  // (no heap fallback -- give a bigger slab instead).
  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(
        alignment)); // rejects 0 too (a&(a-1) alone lets 0 slip through)
    return cursor_.allocate(bytes, alignment);
  }

  // Save the current cursor; rewind() returns the slab to exactly this point.
  Marker mark() const noexcept { return cursor_.mark(); }

  // Release everything allocated since `m` (LIFO). Must be within [base_,
  // cur_].
  void rewind(Marker m) noexcept {
    HYLE_FATAL_ASSERT(valid());
    cursor_.rewind(m);
  }

  // Release everything.
  void reset() noexcept { cursor_.clear(); }

  std::size_t used() const noexcept { return cursor_.used(); }
  std::size_t capacity() const noexcept { return size_; }
  // Bytes still available from the current cursor (a max-aligned request of
  // this size or less is guaranteed to fit; a stronger alignment may consume
  // padding).
  std::size_t remaining() const noexcept { return cursor_.remaining(); }

  // Adapt to the Allocator vtable so a StackAllocator can be injected
  // downstream (like PooledRegion): allocate -> bump, deallocate -> no-op
  // (LIFO frees in bulk via reset()/rewind()), realloc -> bump a fresh slice +
  // copy (can't grow in place). One StackAllocator is single-owner, so the
  // vtable is single-threaded too.
  Allocator as_allocator() noexcept {
    return Allocator(&stack_alloc, &stack_free, &stack_realloc, this);
  }

private:
  static void *stack_alloc(void *state, std::size_t size,
                           std::size_t alignment) noexcept {
    return static_cast<StackAllocator *>(state)->allocate(size, alignment);
  }
  static void stack_free(void * /*state*/, void * /*ptr*/, std::size_t /*size*/,
                         std::size_t /*alignment*/) noexcept {
    // no-op: a stack allocator frees in bulk via reset()/rewind().
  }
  static void *stack_realloc(void *state, void *ptr, std::size_t old_size,
                             std::size_t new_size,
                             std::size_t alignment) noexcept {
    auto *self = static_cast<StackAllocator *>(state);
    if (new_size == 0)
      return nullptr;
    void *p = self->allocate(new_size, alignment);
    if (ptr != nullptr && p != nullptr && old_size != 0) {
      std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    }
    return p;
  }

  Allocator alloc_;
  std::byte *base_ = nullptr;
  std::size_t size_;
  detail::BumpCursor cursor_;
};

} // namespace hyle::mem
