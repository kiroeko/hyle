#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "hyle/container/bounded_queue.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Sharded bounded MPMC queue: N independent Vyukov rings (BoundedQueue shards).
//
// WHY THIS EXISTS (vs the single-ring BoundedQueue): a single Vyukov ring has ONE
// contended enqueue_pos_ / dequeue_pos_; under many producers those atoms become
// a single-point serialization bottleneck (the CAS cache line ping-pongs between
// cores). Benchmarks showed the single ring losing ~1.5x to moodycamel at 4P/4C
// purely from this. Sharding gives each producer its OWN ring -> the producer hot
// path has ZERO inter-producer contention, and throughput stays flat as producers
// scale (measured: ~flat 2P->4P, vs the single ring degrading ~2x).
//
// THE TRADE (honest): this DROPS global FIFO order -- items are FIFO only WITHIN a
// shard, not across the whole queue. That is fine for the design's actual user
// (sched's global inject queue of runnable fibers): a work-stealing scheduler
// already abandons global order; it only needs "no starvation + roughly fair". If
// you need strict global FIFO, use BoundedQueue (one ring) instead. Boundedness is
// KEPT (each shard is bounded; total capacity = num_shards * cap_per_shard) -- the
// real requirement (backpressure, no unbounded growth) is preserved.
//
// NOT a lock-free reinvention: each shard IS the audited Vyukov BoundedQueue (same
// memory orders, TSan-clean). Sharding is pure composition -- it introduces no new
// atomic protocol, so correctness reduces to "each shard is correct" + "the shard
// pick / round-robin drains touch disjoint or independently-correct rings".
//
// CONTENTION MODEL (caller-supplied identity, no thread_local):
//   * try_push(producer_hint, x): `producer_hint` is the CALLER'S stable concurrency
//     identity (a worker index, a thread-id hash, etc). It picks the shard, so two
//     producers with distinct hints hit distinct rings. No hidden thread_local (which
//     is a footgun under fibers that migrate across worker threads).
//   * try_pop(cursor, out): `cursor` is a per-consumer round-robin cursor the caller
//     owns; the queue scans shards from it and updates it to where it found an item
//     (spreads consumers across shards, avoids all consumers hammering shard 0).
//
// T must be trivially copyable (same constraint as the underlying BoundedQueue).
// Alloc defaults to compile-time SystemAllocator ([[no_unique_address]] -> 0 bytes); each
// shard is a BoundedQueue<T,Alloc> so the choice propagates. Pass mem::Allocator to opt in
// (the scheduler does for its global inject queue).
template <class T, class Alloc = mem::SystemAllocator>
class ShardedQueue {
  static_assert(std::is_trivially_copyable_v<T>,
                "ShardedQueue<T>: T must be trivially copyable");

 public:
  // `num_shards` rings, each holding `cap_per_shard` (power of two >= 2) items.
  // For the contention win, size num_shards >= the expected producer count.
  ShardedQueue(std::size_t num_shards, std::size_t cap_per_shard)
      requires std::is_default_constructible_v<Alloc>
      : num_shards_(num_shards) {
    init(cap_per_shard);
  }
  ShardedQueue(Alloc alloc, std::size_t num_shards, std::size_t cap_per_shard)
      : alloc_(alloc), num_shards_(num_shards) {
    init(cap_per_shard);
  }

  ~ShardedQueue() {
    if (shards_ == nullptr) return;
    for (std::size_t i = 0; i < num_shards_; ++i) shards_[i].~BoundedQueue<T, Alloc>();
    alloc_.deallocate(shards_, sizeof(BoundedQueue<T, Alloc>) * num_shards_,
                      alignof(BoundedQueue<T, Alloc>));
  }

  HYLE_DISABLE_COPY_AND_MOVE(ShardedQueue);

  // Allocation failure leaves an invalid object; check once after construction.
  // Push/pop assume valid() so the hot path has no resource-state branch.
  bool valid() const noexcept { return shards_ != nullptr; }

  // Push `item` onto the shard chosen by `producer_hint`. Returns false if THAT
  // shard is full (a bounded-queue semantic: under skew one shard may fill while
  // others have room -- the caller retries / spills, same as the single ring).
  bool try_push(std::size_t producer_hint, T item) noexcept {
    return shards_[producer_hint % num_shards_].try_push(item);
  }

  // Pop the next item, scanning shards round-robin from `cursor`. On success
  // `cursor` advances PAST the shard that yielded, preserving round-robin
  // fairness even when one shard is continuously fed. Returns false only if
  // EVERY shard was empty at its check.
  bool try_pop(std::size_t& cursor, T& out) noexcept {
    for (std::size_t i = 0; i < num_shards_; ++i) {
      std::size_t s = (cursor + i) % num_shards_;
      if (shards_[s].try_pop(out)) {
        cursor = (s + 1) % num_shards_;
        return true;
      }
    }
    return false;
  }

  std::size_t num_shards() const noexcept { return num_shards_; }

  // Total capacity across all shards (sum of per-shard ring capacities).
  std::size_t capacity() const noexcept {
    if (shards_ == nullptr) return 0;
    std::size_t c = 0;
    for (std::size_t i = 0; i < num_shards_; ++i) c += shards_[i].capacity();
    return c;
  }

 private:
  void init(std::size_t cap_per_shard) {
    HYLE_FATAL_ASSERT(num_shards_ >= 1);
    HYLE_FATAL_ASSERT(
        num_shards_ <= static_cast<std::size_t>(-1) /
                           sizeof(BoundedQueue<T, Alloc>) &&
        cap_per_shard <= static_cast<std::size_t>(-1) / num_shards_);
    // Allocator-backed array of ring objects (NOT std::vector -- charter §1 forbids std
    // owning types in a header that may cross the ABI boundary). BoundedQueue is
    // non-movable, so placement-new each shard in situ (sharing this queue's Alloc).
    void* mem = alloc_.allocate(sizeof(BoundedQueue<T, Alloc>) * num_shards_,
                                alignof(BoundedQueue<T, Alloc>));
    if (mem == nullptr) return;
    shards_ = static_cast<BoundedQueue<T, Alloc>*>(mem);
    for (std::size_t i = 0; i < num_shards_; ++i) {
      new (&shards_[i]) BoundedQueue<T, Alloc>(alloc_, cap_per_shard);
      if (!shards_[i].valid()) {
        for (std::size_t j = 0; j <= i; ++j)
          shards_[j].~BoundedQueue<T, Alloc>();
        alloc_.deallocate(shards_,
                          sizeof(BoundedQueue<T, Alloc>) * num_shards_,
                          alignof(BoundedQueue<T, Alloc>));
        shards_ = nullptr;
        return;
      }
    }
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  BoundedQueue<T, Alloc>* shards_ = nullptr;  // allocator-backed array (placement-new'd)
  std::size_t num_shards_;
};

}  // namespace hyle::container
