#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"

// Growable single-threaded object pool (clay utils/memory_pool). Fast
// create/destroy of many objects with slot reuse, growing on demand. The niche
// it fills versus the other HYLE pools:
//   * mem::ObjectPool : FIXED capacity, LOCK-FREE (atomics). Use when the max is
//     known and threads share the pool.
//   * mem::GrowablePool : UNBOUNDED, SINGLE-THREADED (no atomics -> faster when
//     uncontended). Use for per-thread / per-fiber churn of unknown size.
//   * mem::Arena : bump, bulk free, no per-object reuse.
//
// Design: blocks of slots (geometric growth -> O(log n) allocations), an
// intrusive free-list threads recycled slots (a freed slot stores the next-free
// pointer in its own storage -- zero per-slot overhead). Blocks are linked so
// teardown reclaims them; each remembers its byte size (blocks differ in size
// under geometric growth).
//
// Two layers (mirrors MemoryArena <-> ObjectPool<T>):
//   * GrowablePoolCore : NON-TEMPLATE runtime core. Hands out RAW uninitialized
//     slots (acquire/release), no ctor/dtor. This is what the C ABI exports
//     (hyle_growable_pool_*): C has no ctor/dtor, so the C user placement-inits
//     the returned slot and cleans it up before release -- exactly the model of
//     the arena-backed object pool (HYLE_OBJECT_POOL_CREATE).
//   * GrowablePool<T> : thin TYPED wrapper. create()/destroy() run the ctor/dtor
//     around the core's raw slots. A C++-only convenience, not a C ABI gap.
//
// Fixes a real clay bug: clay's move-assignment (memory_pool.h:61) swapped the
// block list but NEVER transferred allocator_, so a move-assigned pool freed
// blocks through a stale/null allocator. HYLE's move transfers every field.
//
// Lifetime contract: the core tracks a live COUNT but not WHICH slots are live,
// so it cannot run leftover destructors on teardown. For non-trivially-
// destructible T you must destroy() everything you create(); a leak trips a
// fail-fast assert (skipped destructors are a bug). Trivial T may be left.

namespace hyle::mem {

namespace detail {

// Non-template runtime core of GrowablePool: raw-slot churn (acquire/release) +
// geometric block growth + intrusive free-list. No T, no ctor/dtor. The typed
// GrowablePool<T> wraps this to run ctor/dtor; the C ABI uses it directly.
class GrowablePoolCore {
 public:
  // `slot_size`/`slot_align` describe one raw slot. The stride is rounded up so a
  // slot can hold EITHER the user object OR the intrusive free-list link (a
  // void*), aligned to at least alignof(void*).
  GrowablePoolCore(Allocator alloc, std::size_t slot_size, std::size_t slot_align,
                   std::uint32_t initial_slots) noexcept
      : alloc_(alloc),
        slot_align_(slot_align < alignof(void*) ? alignof(void*) : slot_align),
        next_block_slots_(initial_slots < 1 ? 1 : initial_slots),
        initial_slots_(initial_slots < 1 ? 1 : initial_slots) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(slot_size >= 1);
    HYLE_FATAL_ASSERT(is_pow2(slot_align_));
    const std::size_t usable =
        slot_size < sizeof(void*) ? sizeof(void*) : slot_size;
    stride_ = align_up(usable, slot_align_);
  }

  ~GrowablePoolCore() { free_blocks(); }

  HYLE_DISABLE_COPY(GrowablePoolCore);
  GrowablePoolCore(GrowablePoolCore&& o) noexcept { steal(o); }
  GrowablePoolCore& operator=(GrowablePoolCore&& o) noexcept {
    if (this != &o) {
      free_blocks();
      steal(o);
    }
    return *this;
  }

  // Raw uninitialized slot: reuse a freed one, else bump within the current block,
  // else grow. `stride_` bytes, `slot_align_`-aligned. Returns nullptr if a needed
  // grow hits backing OOM (graceful, per mem-audit-followup #2 arch B).
  void* acquire() noexcept {
    void* s;
    if (free_ != nullptr) {
      s = free_;
      free_ = *static_cast<void**>(free_);  // next-free link stored at slot start
    } else {
      if (cur_ == end_ && !grow()) return nullptr;  // grow OOM -> graceful nullptr
      s = cur_;
      cur_ += stride_;
    }
    ++live_;
    return s;
  }

  // Return a slot from acquire() to the free-list. Raw: runs NO destructor. The
  // typed wrapper runs ~T before calling this. null is a no-op.
  void release(void* p) noexcept {
    if (p == nullptr) return;
    *static_cast<void**>(p) = free_;  // intrusive link back onto the free-list
    free_ = p;
    --live_;
  }

  // Hand every block back to the allocator (the pool's trim/shrink). Contract:
  // nothing may be live (a freed slot doesn't record its block, so a partial trim
  // can't tell which blocks are idle). After clear() the pool is empty and grows
  // again from the initial block size on the next acquire().
  void clear() noexcept {
    HYLE_FATAL_ASSERT(live_ == 0);  // trimming with live slots would UAF them
    free_blocks();
    free_ = nullptr;
    cur_ = end_ = nullptr;
    capacity_ = 0;
    next_block_slots_ = initial_slots_;
  }

  std::size_t live() const noexcept { return live_; }
  // Total slots across all blocks currently held (grows geometrically).
  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t slot_stride() const noexcept { return stride_; }

 private:
  struct Block {
    Block* next;
    std::size_t bytes;  // size this block was allocated with (geometric growth)
  };

  std::size_t block_align() const noexcept {
    return slot_align_ > alignof(Block) ? slot_align_ : alignof(Block);
  }
  std::size_t header_bytes() const noexcept {
    return align_up(sizeof(Block), slot_align_);
  }

  bool grow() noexcept {
    std::uint32_t n = next_block_slots_;
    std::size_t bytes = header_bytes() + static_cast<std::size_t>(n) * stride_;
    void* chunk = alloc_.allocate(bytes, block_align());
    if (chunk == nullptr) return false;  // grow OOM -> acquire returns nullptr (#2 B)
    Block* b = static_cast<Block*>(chunk);
    b->next = blocks_;
    b->bytes = bytes;
    blocks_ = b;
    cur_ = static_cast<unsigned char*>(chunk) + header_bytes();
    end_ = cur_ + static_cast<std::size_t>(n) * stride_;
    capacity_ += n;
    // Geometric growth, guarding against overflow of the slot count.
    if (next_block_slots_ <= 0x7FFFFFFFu) next_block_slots_ = n * 2;
    return true;
  }

  void free_blocks() noexcept {
    Block* b = blocks_;
    while (b != nullptr) {
      Block* next = b->next;
      alloc_.deallocate(b, b->bytes, block_align());
      b = next;
    }
    blocks_ = nullptr;
    live_ = 0;
  }

  void steal(GrowablePoolCore& o) noexcept {
    alloc_ = o.alloc_;
    stride_ = o.stride_;
    slot_align_ = o.slot_align_;
    blocks_ = o.blocks_;
    free_ = o.free_;
    cur_ = o.cur_;
    end_ = o.end_;
    next_block_slots_ = o.next_block_slots_;
    initial_slots_ = o.initial_slots_;
    live_ = o.live_;
    capacity_ = o.capacity_;
    o.blocks_ = nullptr;
    o.free_ = nullptr;
    o.cur_ = o.end_ = nullptr;
    o.live_ = 0;
    o.capacity_ = 0;
  }

  Allocator alloc_;
  std::size_t stride_ = 0;
  std::size_t slot_align_ = alignof(void*);
  Block* blocks_ = nullptr;          // all allocated blocks (for teardown)
  void* free_ = nullptr;             // recycled slots (intrusive free-list head)
  unsigned char* cur_ = nullptr;     // bump cursor within the current block
  unsigned char* end_ = nullptr;     // end of the current block's slots
  std::uint32_t next_block_slots_ = 1;  // slots for the next block (doubles)
  std::uint32_t initial_slots_ = 1;     // first-block size, restored by clear()
  std::size_t live_ = 0;
  std::size_t capacity_ = 0;
};

}  // namespace detail

template <class T>
class GrowablePool {
 public:
  explicit GrowablePool(Allocator alloc, std::uint32_t initial_slots = 64) noexcept
      : core_(alloc, sizeof(T), alignof(T), initial_slots) {}

  ~GrowablePool() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      // Leftover live objects mean their destructors won't run -- a leak bug.
      HYLE_FATAL_ASSERT(core_.live() == 0);
    }
  }

  HYLE_DISABLE_COPY(GrowablePool);
  GrowablePool(GrowablePool&& o) noexcept : core_(std::move(o.core_)) {}
  GrowablePool& operator=(GrowablePool&& o) noexcept {
    if (this != &o) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        // Discarding this pool's live objects (their dtors won't run) is a bug.
        HYLE_FATAL_ASSERT(core_.live() == 0);
      }
      core_ = std::move(o.core_);
    }
    return *this;
  }

  // Construct a T in a pooled slot (reused if one is free, else bump/grow).
  template <class... Args>
  T* create(Args&&... args) {
    void* slot = core_.acquire();
    if (slot == nullptr) return nullptr;  // pool grow hit OOM -> nullptr (like ObjectPool)
    return ::new (slot) T(std::forward<Args>(args)...);
  }

  // Destroy a T from create() and return its slot to the free-list.
  void destroy(T* p) noexcept {
    if (p == nullptr) return;
    p->~T();
    core_.release(p);
  }

  // Number of live (created, not yet destroyed) objects.
  std::size_t live() const noexcept { return core_.live(); }

  // Return ALL pooled memory to the allocator. Contract: nothing may be live.
  void clear() noexcept { core_.clear(); }

 private:
  detail::GrowablePoolCore core_;
};

}  // namespace hyle::mem
