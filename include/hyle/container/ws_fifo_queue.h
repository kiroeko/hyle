#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Bounded work-stealing FIFO queue (single owner, many thieves).
//
// Faithful port of Tokio's multi_thread scheduler local run-queue
// (tokio/src/runtime/scheduler/multi_thread/queue.rs). NOT a HYLE invention; per
// coding-conventions lock-free queues are ported, not self-written. Do not
// "simplify" the packed head or the memory orders.
//
// Layout: `head_` is ONE atomic packing two 32-bit indices --
//   high 32 = `steal` : where an in-progress steal has reserved up to
//   low  32 = `real`  : the true consumer head (owner pop advances this)
// `tail_` is a separate atomic the owner bumps on push. The owner pushes at the
// tail with NO cas (just a release store); owner pop and thief steal contend on
// `head_` via CAS. FIFO = steal the OLDEST (fair, no starvation), unlike the
// LIFO Chase-Lev deque in work_stealing_deque.h (that one is for fork-join).
//
// A thief steals roughly HALF of the victim's tasks into ITS OWN queue (amortizes
// steal cost, sched-design §2.3), returning one to run immediately.
//
// Slots are atomic to keep victim/thief accesses data-race-free. T must therefore
// be default-constructible, satisfy std::atomic<T>'s value requirements, and be
// always lock-free. Use a pointer or word-sized handle rather than embedding a
// wide task object.
// Alloc defaults to compile-time SystemAllocator ([[no_unique_address]] -> 0 bytes; buf is
// the only alloc, once at construction). Opt into mem::Allocator for injection (sched does).
template <class T, class Alloc = mem::SystemAllocator>
class WsFifoQueue {
  static_assert(std::is_trivially_copyable_v<T>,
                "WsFifoQueue<T>: T must be trivially copyable");
  static_assert(std::is_default_constructible_v<T> && std::is_copy_constructible_v<T> &&
                    std::is_move_constructible_v<T> && std::is_copy_assignable_v<T> &&
                    std::is_move_assignable_v<T>,
                "WsFifoQueue<T>: T must satisfy the preallocated std::atomic<T> value "
                "requirements");
  static_assert(std::atomic<T>::is_always_lock_free,
                "WsFifoQueue<T>: atomic payload must always be lock-free; use a pointer "
                "or word-sized handle");

 public:
  explicit WsFifoQueue(std::size_t capacity)
      requires std::is_default_constructible_v<Alloc>
      : mask_(static_cast<std::uint32_t>(capacity - 1)) {
    init(capacity);
  }
  WsFifoQueue(Alloc alloc, std::size_t capacity)
      : alloc_(alloc), mask_(static_cast<std::uint32_t>(capacity - 1)) {
    init(capacity);
  }

  ~WsFifoQueue() {
    if (buf_ == nullptr) return;
    std::size_t capacity = mask_ + 1;
    for (std::size_t i = 0; i < capacity; ++i) buf_[i].~Slot();
    alloc_.deallocate(buf_, sizeof(Slot) * capacity, alignof(Slot));
  }

  WsFifoQueue(const WsFifoQueue&) = delete;
  WsFifoQueue& operator=(const WsFifoQueue&) = delete;
  WsFifoQueue(WsFifoQueue&&) = delete;
  WsFifoQueue& operator=(WsFifoQueue&&) = delete;

  // Allocation failure leaves an invalid object; check once after construction.
  // Owner/thief hot paths assume valid() and pay no repeated branch.
  bool valid() const noexcept { return buf_ != nullptr; }

  // OWNER ONLY. Append at the tail. Returns false if full (caller spills to the
  // global queue). No CAS on the hot path -- just a release store of the tail.
  bool try_push_back(T x) noexcept {
    std::uint64_t h = head_.load(std::memory_order_acquire);
    std::uint32_t steal = unpack_steal(h);
    std::uint32_t tail = tail_.load(std::memory_order_relaxed);  // owner is sole writer
    // Capacity is measured against `steal` (not real): slots reserved by an
    // in-progress thief are not yet free for reuse.
    if (tail - steal >= mask_ + 1) return false;  // full
    buf_[tail & mask_].v.store(x, std::memory_order_relaxed);
    tail_.store(tail + 1, std::memory_order_release);  // publish slot then tail
    return true;
  }

  // OWNER ONLY. Pop from the front (FIFO). Returns false without changing
  // `out` if empty.
  bool try_pop_front(T& out) noexcept {
    std::uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
      std::uint32_t steal = unpack_steal(h);
      std::uint32_t real = unpack_real(h);
      std::uint32_t tail = tail_.load(std::memory_order_relaxed);
      if (real == tail) return false;
      std::uint32_t next = real + 1;
      // Advance real. If no steal is in progress (steal==real), advance both so
      // the two stay together; otherwise leave steal where the thief put it.
      std::uint64_t next_h = (steal == real) ? pack(next, next) : pack(steal, next);
      if (head_.compare_exchange_weak(h, next_h, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        const T value = buf_[real & mask_].v.load(std::memory_order_relaxed);
        copy_out(out, value);
        return true;
      }
      // h reloaded by compare_exchange; retry.
    }
  }

  // THIEF. Steal ~half of this (victim) queue's tasks INTO `dst` (the thief's own
  // queue), returning one task in `out` to run now. Returns false without
  // changing `out` if nothing was stolen (empty, another thief is mid-steal, or
  // dst lacks room). `dst` is owned by the caller.
  bool try_steal_into(WsFifoQueue& dst, T& out) noexcept {
    // Tokio local queues are identically sized. Supporting unlike capacities
    // requires a different reservation bound; reject that accidental shape
    // loudly instead of indexing dst with a source-sized batch.
    HYLE_FATAL_ASSERT(mask_ == dst.mask_ && this != &dst);
    std::uint32_t dst_tail = dst.tail_.load(std::memory_order_relaxed);  // thief owns dst
    // dst must have room for up to half a queue; if a concurrent steal already
    // reserved space in dst, bail (faithful to tokio's capacity guard).
    std::uint32_t dst_steal = unpack_steal(dst.head_.load(std::memory_order_acquire));
    if (dst_tail - dst_steal > (mask_ + 1) / 2) return false;

    std::uint32_t n = steal_into2(dst, dst_tail);
    if (n == 0) return false;

    n -= 1;  // we return one task directly; dst keeps the other n.
    std::uint32_t ret_pos = dst_tail + n;
    const T value = dst.buf_[ret_pos & dst.mask_].v.load(std::memory_order_relaxed);
    copy_out(out, value);
    if (n == 0) return true;  // only one stolen; dst stays empty
    // Publish the remaining n in dst (release pairs with a future steal/pop).
    dst.tail_.store(dst_tail + n, std::memory_order_release);
    return true;
  }

  std::size_t capacity() const noexcept { return valid() ? mask_ + 1 : 0; }

 private:
  static void copy_out(T& out, const T& value) noexcept {
    std::memcpy(static_cast<void*>(std::addressof(out)), std::addressof(value),
                sizeof(T));
  }

  // Reserve ~half, copy into dst, commit. Returns count stolen (incl. the one
  // the caller will return). Faithful to tokio steal_into2: reserve advances
  // `real` and leaves `steal` LAGGING (so the owner's pop continues past the
  // batch with no collision, while the lagging `steal` blocks push from reusing
  // the slots being copied -- capacity is measured against `steal`). Copy source
  // starts at the lagging `steal` (= the old head). Commit advances `steal` up
  // to the current `real` (the owner may have popped more meanwhile).
  std::uint32_t steal_into2(WsFifoQueue& dst, std::uint32_t dst_tail) noexcept {
    std::uint64_t prev = head_.load(std::memory_order_acquire);
    std::uint64_t next;
    std::uint32_t n;
    for (;;) {
      std::uint32_t src_steal = unpack_steal(prev);
      std::uint32_t src_real = unpack_real(prev);
      std::uint32_t src_tail = tail_.load(std::memory_order_acquire);
      if (src_steal != src_real) return 0;  // another thief is mid-steal
      n = src_tail - src_real;
      n = n - n / 2;  // ceil(half)
      if (n == 0) return 0;  // empty
      std::uint32_t steal_to = src_real + n;
      next = pack(src_steal, steal_to);  // advance REAL, leave steal lagging
      if (head_.compare_exchange_weak(prev, next, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        break;
      }
      // prev reloaded; retry.
    }
    // Copy [first, first+n) where first == lagging steal (the old head).
    std::uint32_t first = unpack_steal(next);
    for (std::uint32_t i = 0; i < n; ++i) {
      T v = buf_[(first + i) & mask_].v.load(std::memory_order_relaxed);
      dst.buf_[(dst_tail + i) & dst.mask_].v.store(v, std::memory_order_relaxed);
    }
    // Commit: set steal := real (real may have advanced past steal_to if the
    // owner popped during the copy), signalling the steal is complete.
    std::uint64_t prev2 = next;
    for (;;) {
      std::uint32_t head_real = unpack_real(prev2);
      std::uint64_t done = pack(head_real, head_real);
      if (head_.compare_exchange_weak(prev2, done, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        return n;
      }
      // prev2 reloaded; retry.
    }
  }

  struct Slot {
    std::atomic<T> v;
  };

  static std::uint64_t pack(std::uint32_t steal, std::uint32_t real) noexcept {
    return (static_cast<std::uint64_t>(steal) << 32) | real;
  }
  static std::uint32_t unpack_steal(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h >> 32);
  }
  static std::uint32_t unpack_real(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h & 0xFFFFFFFFu);
  }

  void init(std::size_t capacity) {
    HYLE_FATAL_ASSERT(capacity >= 2 && capacity <= (1u << 31) &&
                      (capacity & (capacity - 1)) == 0);
    HYLE_FATAL_ASSERT(capacity <= static_cast<std::size_t>(-1) / sizeof(Slot));
    void* mem = alloc_.allocate(sizeof(Slot) * capacity, alignof(Slot));
    if (mem == nullptr) return;
    buf_ = static_cast<Slot*>(mem);
    for (std::size_t i = 0; i < capacity; ++i) new (&buf_[i]) Slot();
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  Slot* buf_ = nullptr;
  std::uint32_t mask_;
  alignas(core::kMaxDestructiveInterferenceSize)
      std::atomic<std::uint64_t> head_; // packed steal|real
  alignas(core::kMaxDestructiveInterferenceSize)
      std::atomic<std::uint32_t> tail_;
};

}  // namespace hyle::container
