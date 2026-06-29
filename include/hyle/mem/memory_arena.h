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
    // Shard count: cap at kMaxShards, never more than blocks (each shard needs >=1).
    num_shards_ = count_ < kMaxShards ? count_ : kMaxShards;
    per_ = count_ / num_shards_;  // >=1; the last shard absorbs the remainder
    heads_ = static_cast<ShardHead*>(
        alloc_.allocate(sizeof(ShardHead) * num_shards_, alignof(ShardHead)));
    HYLE_FATAL_ASSERT(heads_ != nullptr);
    for (std::uint32_t s = 0; s < num_shards_; ++s) new (&heads_[s]) ShardHead();
    // Build each shard's free list over its own contiguous index range [lo,hi): link
    // lo -> lo+1 -> ... -> hi-1 -> null, head = lo. So a block's free links live only
    // in its owning shard, and free (address-routed) returns it to the same shard.
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      std::uint32_t lo = s * per_;
      std::uint32_t hi = (s + 1 == num_shards_) ? count_ : lo + per_;
      for (std::uint32_t i = lo; i < hi; ++i) {
        new (&next_[i]) std::atomic<std::uint32_t>(i + 1 < hi ? i + 1 : kNull);
      }
      heads_[s].h.store(pack(lo, 0), std::memory_order_relaxed);
    }
  }

  ~MemoryArena() {
    for (std::uint32_t i = 0; i < count_; ++i) next_[i].~atomic();
    for (std::uint32_t s = 0; s < num_shards_; ++s) heads_[s].~ShardHead();
    alloc_.deallocate(heads_, sizeof(ShardHead) * num_shards_, alignof(ShardHead));
    alloc_.deallocate(next_, sizeof(std::atomic<std::uint32_t>) * count_,
                      alignof(std::atomic<std::uint32_t>));
    alloc_.deallocate(slab_, stride_ * count_, alignment_);
  }

  HYLE_DISABLE_COPY_AND_MOVE(MemoryArena);

  // Pop a free block. nullptr if the WHOLE pool is exhausted. Tries this thread's home
  // shard first (uncontended in the common per-thread case), then scans the others so a
  // block is handed out as long as ANY shard has one (preserves the single-pool "not
  // exhausted until truly empty" contract).
  void* allocate() noexcept {
    const std::uint32_t home = tls_thread_seq() % num_shards_;
    std::uint32_t idx = pop_shard(home);
    if (idx == kNull) {
      for (std::uint32_t k = 1; k < num_shards_; ++k) {
        idx = pop_shard((home + k) % num_shards_);
        if (idx != kNull) break;
      }
      if (idx == kNull) return nullptr;  // every shard empty
    }
    return slab_ + static_cast<std::size_t>(idx) * stride_;
  }

  // Return a block previously obtained from allocate(). Routes back to the block's
  // OWNING shard (by address), so each shard's free list only ever holds its own index
  // range -- no cross-shard next_ aliasing, per-shard ABA version stays sound.
  void deallocate(void* p) noexcept {
    if (p == nullptr) return;
    auto off = static_cast<std::size_t>(static_cast<std::byte*>(p) - slab_);
    std::uint32_t idx = static_cast<std::uint32_t>(off / stride_);
    HYLE_FATAL_ASSERT(idx < count_ && off % stride_ == 0);  // belongs to this pool
    push_shard(shard_of(idx), idx);
  }

  std::size_t block_size() const noexcept { return block_size_; }
  std::uint32_t capacity() const noexcept { return count_; }

  // Adapt this pool to the Allocator vtable (split alloc/free/realloc): allocate -> pop a
  // fixed block (the request must fit block_size_; the pool serves one size class),
  // deallocate -> push the block back. A pool block can't grow, so realloc is rejected
  // (nullptr). Ideal as a Fiber stack source: the 64 KB stack churns the lock-free free
  // list instead of malloc/mmap.
  Allocator as_allocator() noexcept {
    return Allocator(&pool_alloc, &pool_free, &pool_realloc, this);
  }

 private:
  static void* pool_alloc(void* state, std::size_t size, std::size_t alignment) noexcept {
    auto* self = static_cast<MemoryArena*>(state);
    if (size == 0) return nullptr;
    // One size class: the caller's request must fit a pool block and not need stronger
    // alignment than the slab was built with.
    HYLE_FATAL_ASSERT(size <= self->block_size_ && alignment <= self->alignment_);
    return self->allocate();
  }
  static void pool_free(void* state, void* ptr, std::size_t /*size*/,
                        std::size_t /*alignment*/) noexcept {
    if (ptr != nullptr) static_cast<MemoryArena*>(state)->deallocate(ptr);
  }
  static void* pool_realloc(void* /*state*/, void* /*ptr*/, std::size_t /*old_size*/,
                            std::size_t /*new_size*/, std::size_t /*alignment*/) noexcept {
    return nullptr;  // a fixed-size pool block cannot be reallocated in place
  }

  static constexpr std::uint32_t kNull = 0xFFFFFFFFu;
  // Shard the free list into independent Treiber stacks, one per (up to) this many
  // cache-line-isolated heads. clay inherited a SINGLE head -> every thread CAS'd the
  // same cache line, serializing (O(threads), measured 278x slower than mimalloc's
  // per-thread heaps under 8-way churn). Per-core/per-thread sharding is the SOTA fix
  // mem-design called a HARD constraint (the per-core pool + xcpu_freelist note) but
  // that the toy single-stack never implemented. A thread homes on one shard (tls
  // round-robin) so same-thread alloc/free hits an UNCONTENDED head; free routes by
  // ADDRESS back to the block's owning shard (so each shard's list holds only its own
  // index range -> per-shard ABA version stays valid, no cross-shard next_ aliasing).
  static constexpr std::uint32_t kMaxShards = 16;

  static std::uint64_t pack(std::uint32_t idx, std::uint32_t ver) noexcept {
    return (static_cast<std::uint64_t>(ver) << 32) | idx;
  }
  static std::uint32_t idx_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h);
  }
  static std::uint32_t ver_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h >> 32);
  }

  // Process-wide rotating thread sequence -> a stable per-thread home shard. Purely a
  // DISTRIBUTION hint: correctness never depends on it (free routes by address), so the
  // header-inline thread_local being per-DLL is fine.
  static std::uint32_t tls_thread_seq() noexcept {
    static std::atomic<std::uint32_t> g{0};
    thread_local std::uint32_t id = g.fetch_add(1, std::memory_order_relaxed);
    return id;
  }

  // Pop a free block index from shard s's Treiber stack. kNull if that shard is empty.
  std::uint32_t pop_shard(std::uint32_t s) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      std::uint32_t idx = idx_of(old);
      if (idx == kNull) return kNull;  // this shard empty
      std::uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
      std::uint64_t desired = pack(nxt, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return idx;
      }
    }
  }

  // Push block index back onto shard s's Treiber stack.
  void push_shard(std::uint32_t s, std::uint32_t idx) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      next_[idx].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(idx, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return;
      }
    }
  }

  // Shard owning block idx (contiguous index ranges -> O(1) address routing). The last
  // shard absorbs the remainder when count_ isn't a multiple of num_shards_.
  std::uint32_t shard_of(std::uint32_t idx) const noexcept {
    std::uint32_t s = idx / per_;
    return s < num_shards_ ? s : num_shards_ - 1;
  }

  // One Treiber-stack head per shard, each on its OWN cache line (kill false sharing
  // between shards -- the whole point of sharding).
  struct alignas(HYLE_CACHE_LINE_SIZE) ShardHead {
    std::atomic<std::uint64_t> h;  // index|version
  };

  Allocator alloc_;
  std::byte* slab_ = nullptr;
  std::atomic<std::uint32_t>* next_ = nullptr;  // parallel free-list links
  ShardHead* heads_ = nullptr;                  // num_shards_ cache-line-isolated heads
  std::size_t block_size_;
  std::size_t stride_;
  std::uint32_t count_;
  std::size_t alignment_;
  std::uint32_t num_shards_;
  std::uint32_t per_;  // blocks per shard (last shard takes the remainder)
};

// Bump (region) allocator over ONE MemoryArena block (clay's Arena). Borrows a
// block from the pool, hands out aligned bump slices from it (lock-free atomic
// bump -- concurrent allocate() is safe), and returns the whole block to the
// pool on destruction. There is NO per-allocation free: you release everything
// at once by destroying the Arena (or reset()). Ideal for a transient phase
// (one taskgraph pass, one request) backed by a reusable pool block.
//
// Difference vs StackAllocator: StackAllocator owns its own slab and has
// mark/rewind; Arena borrows a pooled block and returns it on destruction. Both
// are SINGLE-THREADED bump (no atomics -- a bump cursor is inherently one serial
// pointer; making it atomic only adds a per-alloc CAS and, shared across threads,
// the same single-point contention MemoryArena's head had). Concurrency = one
// Arena PER THREAD/fiber (each over its own pooled block) -- the SOTA model
// (per-core scratch arenas), not a contended shared cursor.
class Arena {
 public:
  explicit Arena(MemoryArena* pool)
      : pool_(pool),
        block_(static_cast<std::byte*>(pool->allocate())),
        block_size_(pool->block_size()) {
    HYLE_FATAL_ASSERT(block_ != nullptr);  // pool exhausted
  }

  ~Arena() { pool_->deallocate(block_); }

  HYLE_DISABLE_COPY_AND_MOVE(Arena);

  // Bump `bytes` aligned to `alignment`. nullptr if the block is full. Single-
  // threaded: a plain pointer bump (no atomic). One Arena is owned by one thread.
  void* allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    auto base = reinterpret_cast<std::uintptr_t>(block_);
    std::uintptr_t aligned = (base + offset_ + alignment - 1) & ~(alignment - 1);
    std::size_t start = static_cast<std::size_t>(aligned - base);
    if (start > block_size_ || bytes > block_size_ - start) return nullptr;
    offset_ = start + bytes;
    return block_ + start;
  }

  // Release everything (reuse the block from the start). HYLE addition over clay
  // (clay relied solely on dtor); only safe when no allocations are live.
  void reset() noexcept { offset_ = 0; }

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t used() const noexcept { return offset_; }

  // Adapt this Arena to the Allocator vtable (split): allocate -> bump, deallocate ->
  // no-op (the arena frees in bulk on destruction), realloc -> bump a fresh slice + copy
  // (arenas can't grow in place).
  Allocator as_allocator() noexcept {
    return Allocator(&arena_alloc, &arena_free, &arena_realloc, this);
  }

 private:
  static void* arena_alloc(void* state, std::size_t size, std::size_t alignment) noexcept {
    return static_cast<Arena*>(state)->allocate(size, alignment);
  }
  static void arena_free(void* /*state*/, void* /*ptr*/, std::size_t /*size*/,
                         std::size_t /*alignment*/) noexcept {
    // no-op: an arena frees everything in bulk on destruction / reset().
  }
  static void* arena_realloc(void* state, void* ptr, std::size_t old_size,
                             std::size_t new_size, std::size_t alignment) noexcept {
    auto* self = static_cast<Arena*>(state);
    if (new_size == 0) return nullptr;
    void* p = self->allocate(new_size, alignment);
    if (ptr != nullptr && p != nullptr && old_size != 0) {
      std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    }
    return p;
  }

  MemoryArena* pool_;
  std::byte* block_;
  std::size_t block_size_;
  std::size_t offset_ = 0;  // bump cursor (single-threaded; one Arena per thread)
};

}  // namespace hyle::mem
