#pragma once

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/macros.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Bounded single-producer / single-consumer queue.
//
// SpscQueue is deliberately separate from BoundedQueue: with exactly one
// producer and one consumer there is no reason to pay a CAS or a sequence
// atomic in every slot. The producer owns write_position, the consumer owns
// read_position. No compare/exchange or per-slot atomic is required.
//
// The requested capacity is exact; one private sentinel slot distinguishes
// full from empty without counters or per-slot state. Construction performs
// the only allocation; try_* never allocates. T may be non-trivial, unlike the
// MPMC queue: the producer constructs a slot before publishing it and the consumer
// destroys the slot before releasing it. Destruction requires both threads to
// have quiesced. Calling producer operations from multiple threads, consumer
// operations from multiple threads, or destroying while either side is active
// is undefined behavior.
template <class T, class Alloc = mem::SystemAllocator>
class SpscQueue {
  static_assert(std::is_nothrow_destructible_v<T>,
                "SpscQueue requires nothrow destruction");

 public:
  using value_type = T;
  using size_type = std::size_t;

  explicit SpscQueue(size_type capacity)
    requires std::is_default_constructible_v<Alloc>
      : capacity_(capacity) {
    init();
  }

  SpscQueue(Alloc alloc, size_type capacity)
      : alloc_(std::move(alloc)), capacity_(capacity) {
    init();
  }

  ~SpscQueue() {
    if (slots_ == nullptr) return;
    const size_type read = consumer_.read.load(std::memory_order_relaxed);
    const size_type write = producer_.write.load(std::memory_order_relaxed);
    for (size_type position = read; position != write;
         position = next(position))
      std::destroy_at(pointer(position));
    alloc_.deallocate(slots_, sizeof(Slot) * storage_size_, alignof(Slot));
  }

  HYLE_DISABLE_COPY_AND_MOVE(SpscQueue);

  // Allocation failure leaves an invalid object. Capacity preconditions are
  // programmer errors and are checked once during construction.
  bool valid() const noexcept { return slots_ != nullptr; }
  size_type capacity() const noexcept { return valid() ? capacity_ : 0; }

  // An instantaneous observation, not a synchronization barrier. It is exact
  // when called by the producer or consumer while the other side is paused;
  // otherwise the result may already be stale on return.
  size_type size_approx() const noexcept {
    const size_type write = producer_.write.load(std::memory_order_acquire);
    const size_type read = consumer_.read.load(std::memory_order_acquire);
    return write >= read ? write - read : storage_size_ - read + write;
  }

  bool empty() const noexcept {
    return consumer_.read.load(std::memory_order_acquire) ==
           producer_.write.load(std::memory_order_acquire);
  }

  bool full() const noexcept {
    const size_type write = producer_.write.load(std::memory_order_acquire);
    const size_type read = consumer_.read.load(std::memory_order_acquire);
    return next(write) == read;
  }

  // Producer-only.
  template <class... Args>
  bool try_emplace(Args&&... args) {
    const size_type write = producer_.write.load(std::memory_order_relaxed);
    const size_type next_write = next(write);
    const size_type read = consumer_.read.load(std::memory_order_acquire);
    if (next_write == read) return false;
    std::construct_at(raw_pointer(write), std::forward<Args>(args)...);
    producer_.write.store(next_write, std::memory_order_release);
    return true;
  }

  bool try_push(const T& value) { return try_emplace(value); }
  bool try_push(T&& value) { return try_emplace(std::move(value)); }

  // Consumer-only. The returned pointer remains valid until the consumer pops
  // this element or the queue is destroyed; the producer never touches a live
  // slot. nullptr means empty.
  T* try_front() noexcept {
    const size_type read = consumer_.read.load(std::memory_order_relaxed);
    if (!consumer_has_item(read)) return nullptr;
    return pointer(read);
  }

  // Consumer-only discard. Returns false if empty.
  bool try_pop_front() noexcept {
    const size_type read = consumer_.read.load(std::memory_order_relaxed);
    if (!consumer_has_item(read)) return false;
    std::destroy_at(pointer(read));
    consumer_.read.store(next(read), std::memory_order_release);
    return true;
  }

  // Consumer-only convenience for assignable values. If assignment throws,
  // the item remains at the front and may be retried.
  bool try_pop(T& out)
    noexcept(std::is_nothrow_move_assignable_v<T>)
    requires std::is_move_assignable_v<T>
  {
    const size_type read = consumer_.read.load(std::memory_order_relaxed);
    if (!consumer_has_item(read)) return false;
    T* item = pointer(read);
    out = std::move(*item);
    std::destroy_at(item);
    consumer_.read.store(next(read), std::memory_order_release);
    return true;
  }

  // Consumer-only callback path for values that are not move-assignable. The
  // callback receives T& while the slot is live. If it throws, the item stays.
  template <class Fn>
  bool try_consume(Fn&& consumer) {
    const size_type read = consumer_.read.load(std::memory_order_relaxed);
    if (!consumer_has_item(read)) return false;
    T* item = pointer(read);
    std::forward<Fn>(consumer)(*item);
    std::destroy_at(item);
    consumer_.read.store(next(read), std::memory_order_release);
    return true;
  }

 private:
  struct Slot {
    alignas(T) std::byte storage[sizeof(T)];
  };

  struct alignas(core::kDestructiveInterferenceSize) ProducerState {
    std::atomic<size_type> write{0};
  };

  struct alignas(core::kDestructiveInterferenceSize) ConsumerState {
    std::atomic<size_type> read{0};
  };

  void init() {
    HYLE_FATAL_ASSERT(capacity_ >= 1);
    HYLE_FATAL_ASSERT(capacity_ <
                      std::numeric_limits<size_type>::max() / sizeof(Slot));
    storage_size_ = capacity_ + 1;
    slots_ = static_cast<Slot*>(
        alloc_.allocate(sizeof(Slot) * storage_size_, alignof(Slot)));
  }

  size_type next(size_type position) const noexcept {
    ++position;
    return position == storage_size_ ? 0 : position;
  }

  bool consumer_has_item(size_type read) noexcept {
    return producer_.write.load(std::memory_order_acquire) != read;
  }

  T* pointer(size_type position) noexcept {
    return std::launder(raw_pointer(position));
  }

  const T* pointer(size_type position) const noexcept {
    return std::launder(reinterpret_cast<const T*>(
        slots_[position].storage));
  }

  T* raw_pointer(size_type position) noexcept {
    return reinterpret_cast<T*>(slots_[position].storage);
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  Slot* slots_ = nullptr;
  size_type capacity_;
  size_type storage_size_ = 0;
  ProducerState producer_;
  ConsumerState consumer_;
};

}  // namespace hyle::container
