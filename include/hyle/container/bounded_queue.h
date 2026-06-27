#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Bounded multi-producer / multi-consumer lock-free queue.
//
// This is a faithful port of Dmitry Vyukov's bounded MPMC queue algorithm
// (https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue),
// the same design used by rigtorp/MPMCQueue and folly. NOT a HYLE invention:
// per coding-conventions, lock-free queues are borrowed/ported (famously hard to
// get right; no C-ABI reason to self-write), while container/ self-builds only
// hashmap/carray/string. The per-slot sequence protocol and memory orders below
// match the canonical implementation exactly -- do not "simplify" them.
//
// T must be trivially copyable (skeleton scope: fiber handles / pointers / PODs)
// so the lock-free paths carry no construction/destruction semantics.
template <class T>
class BoundedQueue {
  static_assert(std::is_trivially_copyable_v<T>,
                "BoundedQueue<T>: T must be trivially copyable");

 public:
  // capacity must be a power of two >= 2. Buffer is taken from the injected
  // allocator (charter: explicit allocator injection, no hidden global heap).
  BoundedQueue(mem::Allocator alloc, std::size_t capacity)
      : alloc_(alloc), mask_(capacity - 1) {
    HYLE_FATAL_ASSERT(capacity >= 2 && (capacity & (capacity - 1)) == 0);
    HYLE_FATAL_ASSERT(alloc_.valid());
    buffer_ = static_cast<Cell*>(
        alloc_.allocate(sizeof(Cell) * capacity, alignof(Cell)));
    HYLE_FATAL_ASSERT(buffer_ != nullptr);
    for (std::size_t i = 0; i < capacity; ++i) {
      new (&buffer_[i]) Cell;
      buffer_[i].seq.store(i, std::memory_order_relaxed);
    }
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
  }

  ~BoundedQueue() {
    std::size_t capacity = mask_ + 1;
    for (std::size_t i = 0; i < capacity; ++i) buffer_[i].~Cell();
    alloc_.deallocate(buffer_, sizeof(Cell) * capacity, alignof(Cell));
  }

  HYLE_DISABLE_COPY_AND_MOVE(BoundedQueue);

  // Enqueue. Returns false if the queue is full. Wait-free per attempt.
  bool try_push(T item) noexcept {
    Cell* cell;
    std::uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & mask_];
      std::uint64_t seq = cell->seq.load(std::memory_order_acquire);
      std::int64_t diff =
          static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(pos);
      if (diff == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
          break;
        }
      } else if (diff < 0) {
        return false;  // full: producer lapped the consumer
      } else {
        pos = enqueue_pos_.load(std::memory_order_relaxed);  // raced, reload
      }
    }
    cell->data = item;
    cell->seq.store(pos + 1, std::memory_order_release);
    return true;
  }

  // Dequeue into `out`. Returns false if the queue is empty.
  bool try_pop(T& out) noexcept {
    Cell* cell;
    std::uint64_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & mask_];
      std::uint64_t seq = cell->seq.load(std::memory_order_acquire);
      std::int64_t diff = static_cast<std::int64_t>(seq) -
                          static_cast<std::int64_t>(pos + 1);
      if (diff == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
          break;
        }
      } else if (diff < 0) {
        return false;  // empty
      } else {
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      }
    }
    out = cell->data;
    // Release the slot for a producer one lap ahead.
    cell->seq.store(pos + mask_ + 1, std::memory_order_release);
    return true;
  }

  std::size_t capacity() const noexcept { return mask_ + 1; }

 private:
  struct Cell {
    std::atomic<std::uint64_t> seq;
    T data;
  };

  mem::Allocator alloc_;
  Cell* buffer_ = nullptr;
  std::size_t mask_;
  // Keep the two contended counters on separate cache lines (canonical Vyukov
  // layout) so producers hammering enqueue_pos_ don't false-share with
  // consumers hammering dequeue_pos_.
  alignas(HYLE_CACHE_LINE_SIZE) std::atomic<std::uint64_t> enqueue_pos_;
  alignas(HYLE_CACHE_LINE_SIZE) std::atomic<std::uint64_t> dequeue_pos_;
};

}  // namespace hyle::container
