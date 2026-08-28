#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Bounded multi-producer / multi-consumer lock-free queue.
//
// This is a faithful port of Dmitry Vyukov's bounded MPMC queue algorithm
// (https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue),
// the same design used by rigtorp/MPMCQueue and folly. NOT a HYLE invention:
// per coding-conventions, lock-free queues are borrowed/ported (famously hard to
// get right), while container/ self-builds only
// hashmap/PodVector/string. The per-slot sequence protocol and memory orders below
// match the canonical implementation exactly -- do not "simplify" them.
//
// A producer constructs T before publishing the slot; the winning consumer
// destroys T before releasing it. Construction, destruction and callback/move
// extraction must therefore be noexcept: after a position CAS succeeds the
// lock-free sequence cannot roll back that reservation. Destruction requires
// all producers/consumers to have quiesced.
//
// Alloc defaults to the compile-time SystemAllocator: [[no_unique_address]] makes it
// ZERO bytes (the buffer is the only allocation, done once at construction). Pass a
// runtime mem::Allocator (BoundedQueue<T,mem::Allocator>) to opt into injection -- the
// scheduler does this for its run queues.
template <class T, class Alloc = mem::SystemAllocator>
class BoundedQueue {
  static_assert(std::is_nothrow_destructible_v<T>,
                "BoundedQueue<T>: T must be nothrow destructible");

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
    destroy_remaining();
    std::size_t capacity = mask_ + 1;
    for (std::size_t i = 0; i < capacity; ++i) buffer_[i].~Cell();
    alloc_.deallocate(buffer_, sizeof(Cell) * capacity, alignof(Cell));
  }

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;
  BoundedQueue(BoundedQueue&&) = delete;
  BoundedQueue& operator=(BoundedQueue&&) = delete;

  // Allocation failure leaves an invalid object; check once after construction.
  // Hot queue operations deliberately do not repeat this cold-path check.
  bool valid() const noexcept { return buffer_ != nullptr; }

  // Enqueue. Returns false if the queue is full. Lock-free; a contended caller
  // may retry the CAS, so this is not a wait-free operation. Trivially-copyable
  // T preserves the original memcpy code path, including deleted assignment or
  // copy constructors. Other T is placement-constructed after the slot claim.
  template <class... Args>
    requires std::is_nothrow_constructible_v<T, Args&&...>
  bool try_emplace(Args&&... args) noexcept {
    Cell* cell = nullptr;
    std::uint64_t pos = 0;
    if (!try_claim_enqueue(cell, pos)) return false;
    std::construct_at(raw_value(cell), std::forward<Args>(args)...);
    cell->seq.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool try_push(const T& item) noexcept
    requires(std::is_trivially_copyable_v<T> ||
             std::is_nothrow_copy_constructible_v<T>)
  {
    if constexpr (!std::is_trivially_copyable_v<T>) {
      return try_emplace(item);
    } else {
      Cell* cell = nullptr;
      std::uint64_t pos = 0;
      if (!try_claim_enqueue(cell, pos)) return false;
      std::memcpy(cell->data, std::addressof(item), sizeof(T));
      cell->seq.store(pos + 1, std::memory_order_release);
      return true;
    }
  }

  bool try_push(T&& item) noexcept
    requires(std::is_trivially_copyable_v<T> ||
             std::is_nothrow_move_constructible_v<T>)
  {
    if constexpr (std::is_trivially_copyable_v<T>)
      return try_push(static_cast<const T&>(item));
    else
      return try_emplace(std::move(item));
  }

  // Dequeue into `out`. Returns false if the queue is empty.
  bool try_pop(T& out) noexcept
    requires(std::is_trivially_copyable_v<T> ||
             std::is_nothrow_assignable_v<T&, T&&>)
  {
    Cell* cell = nullptr;
    std::uint64_t pos = 0;
    if (!try_claim_dequeue(cell, pos)) return false;
    if constexpr (std::is_trivially_copyable_v<T>) {
      // Preserve the byte path for PODs with deleted assignment.
      std::memcpy(static_cast<void*>(std::addressof(out)), cell->data,
                  sizeof(T));
    } else {
      T* item = value(cell);
      out = std::move(*item);
      std::destroy_at(item);
    }
    cell->seq.store(pos + mask_ + 1, std::memory_order_release);
    return true;
  }

  // Callback extraction supports values that are not assignable. The callback
  // receives the live slot and must not escape its address. It must be noexcept:
  // the dequeue position has already been claimed and cannot be rolled back.
  template <class Fn>
    requires std::is_nothrow_invocable_v<Fn&&, T&>
  bool try_consume(Fn&& consumer) noexcept {
    Cell* cell = nullptr;
    std::uint64_t pos = 0;
    if (!try_claim_dequeue(cell, pos)) return false;
    T* item = value(cell);
    std::invoke(std::forward<Fn>(consumer), *item);
    std::destroy_at(item);
    cell->seq.store(pos + mask_ + 1, std::memory_order_release);
    return true;
  }

  std::size_t capacity() const noexcept { return valid() ? mask_ + 1 : 0; }

 private:
  // Each cell uses the target-family destructive-interference span. The
  // canonical Vyukov/rigtorp layout pads slots to a cache line, so
  // adjacent cells, produced/consumed by different threads concurrently, don't false-
  // share. HYLE's earlier port OMITTED this: an 8-byte T made Cell 16 bytes -> 4 cells
  // per line -> producers/consumers on neighbouring slots ping-pong the line (measured
  // ~1.33x slower than rigtorp at 4P/4C). Target spacing restores parity without
  // forcing every x86 slot to pay the 128-byte cross-family maximum.
  static constexpr std::size_t kCellAlignment =
      alignof(T) > core::kDestructiveInterferenceSize
          ? alignof(T)
          : core::kDestructiveInterferenceSize;
  struct alignas(kCellAlignment) Cell {
    std::atomic<std::uint64_t> seq;
    // Raw storage avoids default-constructing every preallocated slot. The
    // sequence number is the publication/lifetime state; no extra flag is added.
    alignas(T) std::byte data[sizeof(T)];
  };

  static T* raw_value(Cell* cell) noexcept {
    return reinterpret_cast<T*>(cell->data);
  }

  static T* value(Cell* cell) noexcept {
    return std::launder(raw_value(cell));
  }

  HYLE_FORCE_INLINE bool try_claim_enqueue(Cell*& cell,
                                           std::uint64_t& position) noexcept {
    std::uint64_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & mask_];
      const std::uint64_t seq = cell->seq.load(std::memory_order_acquire);
      const std::int64_t diff =
          static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(pos);
      if (diff == 0) {
        // CAS failure already reloads pos. Match the canonical queue: do not
        // add a pause to every lost CAS on this contended path.
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
          position = pos;
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = enqueue_pos_.load(std::memory_order_relaxed);
      }
    }
  }

  HYLE_FORCE_INLINE bool try_claim_dequeue(Cell*& cell,
                                           std::uint64_t& position) noexcept {
    std::uint64_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &buffer_[pos & mask_];
      const std::uint64_t seq = cell->seq.load(std::memory_order_acquire);
      const std::int64_t diff = static_cast<std::int64_t>(seq) -
                                static_cast<std::int64_t>(pos + 1);
      if (diff == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed)) {
          position = pos;
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      }
    }
  }

  void destroy_remaining() noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      const std::uint64_t begin =
          dequeue_pos_.load(std::memory_order_relaxed);
      const std::uint64_t end = enqueue_pos_.load(std::memory_order_relaxed);
      const std::uint64_t count = end - begin;
      HYLE_FATAL_ASSERT(count <= mask_ + 1);
      for (std::uint64_t offset = 0; offset < count; ++offset) {
        const std::uint64_t pos = begin + offset;
        Cell* cell = &buffer_[pos & mask_];
        HYLE_FATAL_ASSERT(cell->seq.load(std::memory_order_relaxed) == pos + 1);
        std::destroy_at(value(cell));
      }
    }
  }

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
  alignas(core::kMaxDestructiveInterferenceSize)
      std::atomic<std::uint64_t> enqueue_pos_;
  alignas(core::kMaxDestructiveInterferenceSize)
      std::atomic<std::uint64_t> dequeue_pos_;
};

}  // namespace hyle::container
