#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "hyle/core/cache.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Outcome of a steal(). Hoisted out of the class template (it does not depend on
// T) so every WorkStealingDeque<T> shares one enum -- callers compare against a
// single type, not a per-instantiation nested one.
enum class StealStatus : std::uint8_t {
  kOk,     // value valid
  kEmpty,  // deque was empty
  kAbort,  // lost the race to another thief / the owner; retry if you want
};

// Bounded Chase-Lev work-stealing deque.
//
// Faithful port of the C11 algorithm in Lê, Pop, Cohen & Zappa Nardelli,
// "Correct and Efficient Work-Stealing for Weak Memory Models" (PPoPP 2013),
// Figure 1 -- the authoritative memory-ordering reference (the part everyone
// gets wrong on ARM/weak models). NOT a HYLE invention; per coding-conventions
// lock-free queues are ported, not self-written. Do not "simplify" the orders
// or the standalone fences -- they are load-bearing.
//
// Roles (this is the whole point vs an MPMC ring): the OWNER thread pushes and
// pops at the bottom with almost no synchronization (relaxed + one fence; a CAS
// only when taking the very last element). THIEVES steal from the top with a
// single CAS. Owner's hot path never does an uncontended CAS.
//
// Bounded: fixed power-of-two capacity (like tokio's local queue, 256). Full
// pushes are rejected (sched-03 spills to the global queue); no dynamic array
// growth (the resize path is a separate hard problem HYLE doesn't need).
//
// T must be trivially copyable (fiber handles / pointers / PODs).
// Alloc defaults to compile-time SystemAllocator ([[no_unique_address]] -> 0 bytes; buf is
// allocated once at construction). Opt into mem::Allocator for injection.
template <class T, class Alloc = mem::SystemAllocator>
class WorkStealingDeque {
  static_assert(std::is_trivially_copyable_v<T>,
                "WorkStealingDeque<T>: T must be trivially copyable");

 public:
  struct PopResult {
    bool ok;
    T value;
  };
  struct StealResult {
    StealStatus status;
    T value;
  };

  explicit WorkStealingDeque(std::size_t capacity)
      requires std::is_default_constructible_v<Alloc>
      : mask_(capacity - 1) {
    init(capacity);
  }
  WorkStealingDeque(Alloc alloc, std::size_t capacity)
      : alloc_(alloc), mask_(capacity - 1) {
    init(capacity);
  }

  ~WorkStealingDeque() {
    if (buf_ == nullptr) return;
    std::size_t capacity = mask_ + 1;
    for (std::size_t i = 0; i < capacity; ++i) buf_[i].~Slot();
    alloc_.deallocate(buf_, sizeof(Slot) * capacity, alignof(Slot));
  }

  HYLE_DISABLE_COPY_AND_MOVE(WorkStealingDeque);

  // Allocation failure leaves an invalid object; check once after construction.
  // Owner/thief hot paths assume valid() and pay no repeated branch.
  bool valid() const noexcept { return buf_ != nullptr; }

  // OWNER ONLY. Push onto the bottom. Returns false if the deque is full.
  bool push_bottom(T x) noexcept {
    std::uint64_t b = bottom_.load(std::memory_order_relaxed);
    std::uint64_t t = top_.load(std::memory_order_acquire);
    if (b - t >= mask_ + 1) return false;  // full (b>=t always; b-t == size)
    buf_[b & mask_].v.store(x, std::memory_order_relaxed);
    // Publish the slot write before the bottom bump so a thief that observes the
    // new bottom also observes the value (release fence / acquire-load pairing).
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  // OWNER ONLY. Pop from the bottom (LIFO for the owner). Returns {false, _} if
  // empty or if a thief won the last element.
  PopResult pop_bottom() noexcept {
    std::uint64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    // Full barrier: orders the bottom write before the top read so owner/thief
    // agree on who gets the last element (Lê 2013 -- this seq_cst fence is the
    // crux; do not weaken).
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::uint64_t t = top_.load(std::memory_order_relaxed);

    // SIGNED difference: b was decremented and can wrap below t on an empty
    // deque (e.g. bottom==top==0 -> b==UINT64_MAX). A naive unsigned `t <= b`
    // would mis-read that as non-empty. Cast the wrapping difference to signed,
    // exactly as crossbeam-deque / st3 do.
    std::int64_t size = static_cast<std::int64_t>(b - t);
    if (size < 0) {
      // Empty: restore bottom.
      bottom_.store(b + 1, std::memory_order_relaxed);
      return {false, T{}};
    }
    T x = buf_[b & mask_].v.load(std::memory_order_relaxed);
    if (size > 0) {
      return {true, x};  // more than one element: no race with thieves
    }
    // Exactly one element (size == 0): race a thief for it via the top CAS.
    bool won = top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                            std::memory_order_relaxed);
    bottom_.store(b + 1, std::memory_order_relaxed);  // reset to empty either way
    return won ? PopResult{true, x} : PopResult{false, T{}};
  }

  // THIEF. Steal from the top (takes the oldest element). kEmpty if empty,
  // kAbort if another stealer/the owner won the race (caller may retry).
  StealResult steal() noexcept {
    std::uint64_t t = top_.load(std::memory_order_acquire);
    // Full barrier between the top and bottom reads (Lê 2013): ensures a
    // consistent snapshot vs a concurrent owner pop.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::uint64_t b = bottom_.load(std::memory_order_acquire);
    if (static_cast<std::int64_t>(b - t) <= 0) {
      return {StealStatus::kEmpty, T{}};  // empty
    }
    // Read the value BEFORE claiming the slot via CAS. The CAS both claims top
    // and validates the read wasn't overwritten by a wrapped owner push.
    T x = buf_[t & mask_].v.load(std::memory_order_relaxed);
    if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                      std::memory_order_relaxed)) {
      return {StealStatus::kAbort, T{}};  // lost the race
    }
    return {StealStatus::kOk, x};
  }

  std::size_t capacity() const noexcept { return valid() ? mask_ + 1 : 0; }

 private:
  // Atomic slots: owner writes and thieves read the same slot; making the slot
  // atomic (relaxed) keeps the access data-race-free under TSan while the
  // top/bottom atomics + fences carry the actual happens-before.
  struct Slot {
    std::atomic<T> v;
  };

  void init(std::size_t capacity) {
    HYLE_FATAL_ASSERT(capacity >= 2 && (capacity & (capacity - 1)) == 0);
    HYLE_FATAL_ASSERT(capacity <= static_cast<std::size_t>(-1) / sizeof(Slot));
    void* mem = alloc_.allocate(sizeof(Slot) * capacity, alignof(Slot));
    if (mem == nullptr) return;
    buf_ = static_cast<Slot*>(mem);
    for (std::size_t i = 0; i < capacity; ++i) new (&buf_[i]) Slot();
    top_.store(0, std::memory_order_relaxed);
    bottom_.store(0, std::memory_order_relaxed);
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  Slot* buf_ = nullptr;
  std::uint64_t mask_;
  // top (thief end) and bottom (owner end) are hammered by different threads;
  // keep them on separate cache lines to avoid false sharing.
  alignas(core::kDestructiveInterferenceSize) std::atomic<std::uint64_t> top_;
  alignas(core::kDestructiveInterferenceSize) std::atomic<std::uint64_t> bottom_;
};

}  // namespace hyle::container
