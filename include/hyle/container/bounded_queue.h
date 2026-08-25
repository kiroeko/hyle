#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

#include "hyle/core/cache.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

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
//
// Alloc defaults to the compile-time SystemAllocator: [[no_unique_address]] makes it
// ZERO bytes (the buffer is the only allocation, done once at construction). Pass a
// runtime mem::Allocator (BoundedQueue<T,mem::Allocator>) to opt into injection -- the
// scheduler does this for its run queues.
template <class T, class Alloc = mem::SystemAllocator>
class BoundedQueue {
  static_assert(std::is_trivially_copyable_v<T>,
                "BoundedQueue<T>: T must be trivially copyable");

 public:
  // capacity must be a power of two >= 2. Buffer is taken from Alloc. Default-alloc
  // overload needs only the capacity; the explicit-alloc overload injects a runtime one.
  explicit BoundedQueue(std::size_t capacity)
      requires std::is_default_constructible_v<Alloc>
      : mask_(capacity - 1) {
    init(capacity);
  }
  BoundedQueue(Alloc alloc, std::size_t capacity) : alloc_(alloc), mask_(capacity - 1) {
    init(capacity);
  }

  ~BoundedQueue() {
    if (buffer_ == nullptr) return;
    std::size_t capacity = mask_ + 1;
    for (std::size_t i = 0; i < capacity; ++i) buffer_[i].~Cell();
    alloc_.deallocate(buffer_, sizeof(Cell) * capacity, alignof(Cell));
  }

  HYLE_DISABLE_COPY_AND_MOVE(BoundedQueue);

  // Allocation failure leaves an invalid object; check once after construction.
  // Hot queue operations deliberately do not repeat this cold-path check.
  bool valid() const noexcept { return buffer_ != nullptr; }

  // Enqueue. Returns false if the queue is full. Lock-free; a contended caller
  // may retry the CAS, so this is not a wait-free operation.
  bool try_push(const T& item) noexcept {
    Cell* cell;
    std::uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & mask_];
      std::uint64_t seq = cell->seq.load(std::memory_order_acquire);
      std::int64_t diff =
          static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(pos);
      if (diff == 0) {
        // Tight retry on a lost CAS (the contended path), matching rigtorp/folly:
        // CAS failure already reloads `pos` to the current head, so we re-probe the
        // new slot immediately. An earlier port added a cpu_relax() pause here, but
        // a ~40-cycle pause on EVERY lost CAS is net latency under 4P churn (measured
        // ~1.34x slower than rigtorp); the canonical queues do NOT back off here.
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
    std::memcpy(cell->data, std::addressof(item), sizeof(T));
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
    // T's public contract is trivially-copyable, not copy-assignable. Copy the
    // object representation so PODs with deleted assignment remain usable.
    std::memcpy(static_cast<void*>(std::addressof(out)), cell->data, sizeof(T));
    // Release the slot for a producer one lap ahead.
    cell->seq.store(pos + mask_ + 1, std::memory_order_release);
    return true;
  }

  std::size_t capacity() const noexcept { return valid() ? mask_ + 1 : 0; }

 private:
  // Each cell on its OWN destructive-interference span. The canonical
  // Vyukov/rigtorp layout pads slots to a cache line (rigtorp uses 64 bytes on
  // current x86), so
  // adjacent cells, produced/consumed by different threads concurrently, don't false-
  // share. HYLE's earlier port OMITTED this: an 8-byte T made Cell 16 bytes -> 4 cells
  // per line -> producers/consumers on neighbouring slots ping-pong the line (measured
  // ~1.33x slower than rigtorp at 4P/4C). alignas(cache line) restores parity. Cost:
  // HYLE uses core's stable 128-byte upper bound to also cover 128-byte target
  // families. This can cost twice rigtorp's x86 slot footprint and therefore
  // requires direct-consumer throughput/footprint regression evidence; it is
  // the explicit space-for-portable-false-sharing-isolation trade.
  static constexpr std::size_t kCellAlignment =
      alignof(T) > core::kDestructiveInterferenceSize
          ? alignof(T)
          : core::kDestructiveInterferenceSize;
  struct alignas(kCellAlignment) Cell {
    std::atomic<std::uint64_t> seq;
    // A representation buffer avoids default-constructing/assigning T in every
    // preallocated slot. Trivially-copyable values may round-trip through bytes.
    alignas(T) std::byte data[sizeof(T)];
  };

  void init(std::size_t capacity) {
    HYLE_FATAL_ASSERT(capacity >= 2 && (capacity & (capacity - 1)) == 0);
    HYLE_FATAL_ASSERT(capacity <= static_cast<std::size_t>(-1) / sizeof(Cell));
    buffer_ = static_cast<Cell*>(
        alloc_.allocate(sizeof(Cell) * capacity, alignof(Cell)));
    if (buffer_ == nullptr) return;
    for (std::size_t i = 0; i < capacity; ++i) {
      new (&buffer_[i]) Cell;
      buffer_[i].seq.store(i, std::memory_order_relaxed);
    }
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  Cell* buffer_ = nullptr;
  std::size_t mask_;
  // Keep the two contended counters on separate cache lines (canonical Vyukov
  // layout) so producers hammering enqueue_pos_ don't false-share with
  // consumers hammering dequeue_pos_.
  alignas(core::kDestructiveInterferenceSize) std::atomic<std::uint64_t> enqueue_pos_;
  alignas(core::kDestructiveInterferenceSize) std::atomic<std::uint64_t> dequeue_pos_;
};

}  // namespace hyle::container
