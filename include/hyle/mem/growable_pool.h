#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"

// Growable single-threaded typed object pool (clay utils/memory_pool). Fast
// create/destroy of many T's with slot reuse, growing on demand. The niche it
// fills versus the other HYLE pools:
//   * mem::ObjectPool : FIXED capacity, LOCK-FREE (atomics). Use when the max is
//     known and threads share the pool.
//   * mem::GrowablePool : UNBOUNDED, SINGLE-THREADED (no atomics -> faster when
//     uncontended). Use for per-thread / per-fiber churn of unknown size.
//   * mem::Arena : bump, bulk free, no per-object reuse.
//
// Design: blocks of slots (geometric growth -> O(log n) allocations), an
// intrusive free-list threads recycled slots (a freed slot stores the next-free
// pointer in its own storage -- zero per-slot overhead). Blocks are linked so
// the destructor reclaims them; each remembers its byte size (blocks differ in
// size under geometric growth).
//
// Fixes a real clay bug: clay's move-assignment (memory_pool.h:61) swapped the
// block list but NEVER transferred allocator_, so a move-assigned pool freed
// blocks through a stale/null allocator. HYLE's move transfers every field.
//
// Lifetime contract: the pool tracks a live COUNT but not WHICH slots are live,
// so it cannot run leftover destructors on teardown. For non-trivially-
// destructible T you must destroy() everything you create(); a leak trips a
// fail-fast assert (skipped destructors are a bug). Trivial T may be left.

namespace hyle::mem {

template <class T>
class GrowablePool {
 public:
  explicit GrowablePool(Allocator alloc, std::uint32_t initial_slots = 64) noexcept
      : alloc_(alloc), next_block_slots_(initial_slots < 1 ? 1 : initial_slots) {
    HYLE_FATAL_ASSERT(alloc_.valid());
  }

  ~GrowablePool() { release_all(); }

  HYLE_DISABLE_COPY(GrowablePool);
  GrowablePool(GrowablePool&& o) noexcept { steal(o); }
  GrowablePool& operator=(GrowablePool&& o) noexcept {
    if (this != &o) {
      release_all();
      steal(o);
    }
    return *this;
  }

  // Construct a T in a pooled slot (reused if one is free, else bump/grow).
  template <class... Args>
  T* create(Args&&... args) {
    Slot* s;
    if (free_ != nullptr) {
      s = free_;
      free_ = free_->next_free;
    } else {
      if (cur_ == end_) grow();
      s = cur_++;
    }
    ++live_;
    return ::new (static_cast<void*>(s)) T(std::forward<Args>(args)...);
  }

  // Destroy a T from create() and return its slot to the free-list.
  void destroy(T* p) noexcept {
    if (p == nullptr) return;
    p->~T();
    Slot* s = reinterpret_cast<Slot*>(p);  // storage is at offset 0 of the union
    s->next_free = free_;
    free_ = s;
    --live_;
  }

  // Number of live (created, not yet destroyed) objects.
  std::size_t live() const noexcept { return live_; }

 private:
  // A slot is EITHER a free-list link OR (when live) holds a T. alignas(T) keeps
  // the storage suitably aligned; next_free overlaps it (offset 0).
  union Slot {
    Slot* next_free;
    alignas(T) unsigned char storage[sizeof(T)];
    Slot() noexcept {}
    ~Slot() noexcept {}
  };
  struct Block {
    Block* next;
    std::size_t bytes;  // size this block was allocated with (geometric growth)
  };

  static constexpr std::size_t kAlign =
      alignof(Slot) > alignof(Block) ? alignof(Slot) : alignof(Block);
  static std::size_t header_bytes() noexcept {
    return align_up(sizeof(Block), alignof(Slot));
  }

  void grow() {
    std::uint32_t n = next_block_slots_;
    std::size_t bytes = header_bytes() + static_cast<std::size_t>(n) * sizeof(Slot);
    void* chunk = alloc_.allocate(bytes, kAlign);
    HYLE_FATAL_ASSERT(chunk != nullptr);
    Block* b = static_cast<Block*>(chunk);
    b->next = blocks_;
    b->bytes = bytes;
    blocks_ = b;
    cur_ = reinterpret_cast<Slot*>(static_cast<unsigned char*>(chunk) +
                                   header_bytes());
    end_ = cur_ + n;
    // Geometric growth, guarding against overflow of the slot count.
    if (next_block_slots_ <= 0x7FFFFFFFu) next_block_slots_ = n * 2;
  }

  void release_all() noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      // Leftover live objects mean their destructors won't run -- a leak bug.
      HYLE_FATAL_ASSERT(live_ == 0);
    }
    Block* b = blocks_;
    while (b != nullptr) {
      Block* next = b->next;
      std::size_t bytes = b->bytes;
      alloc_.deallocate(b, bytes, kAlign);
      b = next;
    }
    blocks_ = nullptr;
    free_ = nullptr;
    cur_ = end_ = nullptr;
    live_ = 0;
  }

  void steal(GrowablePool& o) noexcept {
    alloc_ = o.alloc_;
    blocks_ = o.blocks_;
    free_ = o.free_;
    cur_ = o.cur_;
    end_ = o.end_;
    next_block_slots_ = o.next_block_slots_;
    live_ = o.live_;
    o.blocks_ = nullptr;
    o.free_ = nullptr;
    o.cur_ = o.end_ = nullptr;
    o.live_ = 0;
  }

  Allocator alloc_;
  Block* blocks_ = nullptr;  // all allocated blocks (for teardown)
  Slot* free_ = nullptr;     // recycled slots
  Slot* cur_ = nullptr;      // bump cursor within the current block
  Slot* end_ = nullptr;      // end of the current block's slots
  std::uint32_t next_block_slots_;  // slots for the next block (doubles)
  std::size_t live_ = 0;
};

}  // namespace hyle::mem
