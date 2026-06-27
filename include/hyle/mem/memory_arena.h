#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {

// Lock-free fixed-size block pool (clay's MemoryArena, modernized).
//
// A slab of `block_count` equal-size blocks; allocate()/deallocate() pop/push a
// Treiber stack of free blocks. Constant-time, lock-free, allocator-injected.
// Ideal for same-size churn (fiber stacks, pooled nodes) where a general malloc
// is overkill.
//
// Two deliberate changes from clay (see mem-design.md):
//  1. Free-list head is ONE 64-bit atomic packing (index<<32 | version) -- pure
//     64-bit CAS, lock-free everywhere. clay used a 16-byte atomic<{ptr,tag}>
//     needing cmpxchg16b/-mcx16 (else a silent lock fallback). The version tag
//     defeats ABA.
//  2. Block count is a ctor parameter (clay capped at 64 static slots -- bug #5).
//
// The free-list `next` links live in a PARALLEL index array, not intrusively in
// the block bytes: no aliasing of user storage, clean lifetimes, TSan-clean.
class MemoryArena {
 public:
  // `block_size` bytes per block, `block_count` blocks, each aligned to
  // `alignment` (power of two). Slab + link array come from `alloc`.
  MemoryArena(Allocator alloc, std::size_t block_size, std::uint32_t block_count,
              std::size_t alignment = alignof(std::max_align_t))
      : alloc_(alloc),
        block_size_(block_size),
        count_(block_count),
        alignment_(alignment < alignof(std::max_align_t) ? alignof(std::max_align_t)
                                                         : alignment) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(block_count >= 1 && block_count < kNull);
    HYLE_FATAL_ASSERT(block_size_ >= 1);
    HYLE_FATAL_ASSERT((alignment_ & (alignment_ - 1)) == 0);
    stride_ = (block_size_ + alignment_ - 1) & ~(alignment_ - 1);  // round up
    slab_ = static_cast<std::byte*>(alloc_.allocate(stride_ * count_, alignment_));
    HYLE_FATAL_ASSERT(slab_ != nullptr);
    next_ = static_cast<std::atomic<std::uint32_t>*>(
        alloc_.allocate(sizeof(std::atomic<std::uint32_t>) * count_,
                        alignof(std::atomic<std::uint32_t>)));
    HYLE_FATAL_ASSERT(next_ != nullptr);
    // Thread free list 0 -> 1 -> ... -> (count-1) -> null; head starts at 0.
    for (std::uint32_t i = 0; i < count_; ++i) {
      new (&next_[i]) std::atomic<std::uint32_t>(i + 1 < count_ ? i + 1 : kNull);
    }
    head_.store(pack(0, 0), std::memory_order_relaxed);
  }

  ~MemoryArena() {
    for (std::uint32_t i = 0; i < count_; ++i) next_[i].~atomic();
    alloc_.deallocate(next_, sizeof(std::atomic<std::uint32_t>) * count_,
                      alignof(std::atomic<std::uint32_t>));
    alloc_.deallocate(slab_, stride_ * count_, alignment_);
  }

  HYLE_DISABLE_COPY_AND_MOVE(MemoryArena);

  // Pop a free block. nullptr if the pool is exhausted.
  void* allocate() noexcept {
    std::uint64_t old = head_.load(std::memory_order_acquire);
    for (;;) {
      std::uint32_t idx = idx_of(old);
      if (idx == kNull) return nullptr;  // exhausted
      std::uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
      std::uint64_t desired = pack(nxt, ver_of(old) + 1);
      if (head_.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        return slab_ + static_cast<std::size_t>(idx) * stride_;
      }
      // old reloaded; retry.
    }
  }

  // Return a block previously obtained from allocate().
  void deallocate(void* p) noexcept {
    if (p == nullptr) return;
    auto off = static_cast<std::size_t>(static_cast<std::byte*>(p) - slab_);
    std::uint32_t idx = static_cast<std::uint32_t>(off / stride_);
    HYLE_FATAL_ASSERT(idx < count_ && off % stride_ == 0);  // belongs to this pool
    std::uint64_t old = head_.load(std::memory_order_acquire);
    for (;;) {
      next_[idx].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(idx, ver_of(old) + 1);
      if (head_.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        return;
      }
      // old reloaded; rewrite next_[idx] with the new head and retry.
    }
  }

  std::size_t block_size() const noexcept { return block_size_; }
  std::uint32_t capacity() const noexcept { return count_; }

 private:
  static constexpr std::uint32_t kNull = 0xFFFFFFFFu;
  static std::uint64_t pack(std::uint32_t idx, std::uint32_t ver) noexcept {
    return (static_cast<std::uint64_t>(ver) << 32) | idx;
  }
  static std::uint32_t idx_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h);
  }
  static std::uint32_t ver_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h >> 32);
  }

  Allocator alloc_;
  std::byte* slab_ = nullptr;
  std::atomic<std::uint32_t>* next_ = nullptr;  // parallel free-list links
  std::size_t block_size_;
  std::size_t stride_;
  std::uint32_t count_;
  std::size_t alignment_;
  alignas(HYLE_CACHE_LINE_SIZE) std::atomic<std::uint64_t> head_;  // index|version
};

// Bump (region) allocator over ONE MemoryArena block (clay's Arena). Borrows a
// block from the pool, hands out aligned bump slices from it (lock-free atomic
// bump -- concurrent allocate() is safe), and returns the whole block to the
// pool on destruction. There is NO per-allocation free: you release everything
// at once by destroying the Arena (or reset()). Ideal for a transient phase
// (one taskgraph pass, one request) backed by a reusable pool block.
//
// Difference vs StackAllocator: StackAllocator owns its own slab and is
// single-threaded with mark/rewind; Arena borrows a pooled block and supports
// concurrent bump. Both are "bump", different lifecycles.
class Arena {
 public:
  explicit Arena(MemoryArena* pool)
      : pool_(pool),
        block_(static_cast<std::byte*>(pool->allocate())),
        block_size_(pool->block_size()) {
    HYLE_FATAL_ASSERT(block_ != nullptr);  // pool exhausted
    offset_.store(0, std::memory_order_relaxed);
  }

  ~Arena() { pool_->deallocate(block_); }

  HYLE_DISABLE_COPY_AND_MOVE(Arena);

  // Bump `bytes` aligned to `alignment`. nullptr if the block is full.
  // Lock-free: safe to call concurrently from multiple threads.
  void* allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    auto base = reinterpret_cast<std::uintptr_t>(block_);
    std::size_t old = offset_.load(std::memory_order_acquire);
    for (;;) {
      std::uintptr_t aligned = (base + old + alignment - 1) & ~(alignment - 1);
      std::size_t start = static_cast<std::size_t>(aligned - base);
      if (start > block_size_ || bytes > block_size_ - start) return nullptr;
      std::size_t next = start + bytes;
      if (offset_.compare_exchange_weak(old, next, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        return block_ + start;
      }
      // old reloaded; retry.
    }
  }

  // Release everything (reuse the block from the start). HYLE addition over clay
  // (clay relied solely on dtor); only safe when no allocations are live.
  void reset() noexcept { offset_.store(0, std::memory_order_release); }

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t used() const noexcept {
    return offset_.load(std::memory_order_relaxed);
  }

  // Adapt this Arena to the Allocator vtable: allocate -> bump, deallocate ->
  // no-op (the arena frees in bulk on destruction). realloc bumps a fresh slice
  // and copies (arenas can't grow in place).
  Allocator as_allocator() noexcept { return Allocator(&arena_realloc, this); }

 private:
  static void* arena_realloc(void* state, void* ptr, std::size_t old_size,
                             std::size_t new_size, std::size_t alignment) noexcept {
    auto* self = static_cast<Arena*>(state);
    if (new_size == 0) return nullptr;  // free is a no-op for an arena
    void* p = self->allocate(new_size, alignment);
    if (ptr != nullptr && p != nullptr && old_size != 0) {
      std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    }
    return p;
  }

  MemoryArena* pool_;
  std::byte* block_;
  std::size_t block_size_;
  std::atomic<std::size_t> offset_{0};
};

}  // namespace hyle::mem
