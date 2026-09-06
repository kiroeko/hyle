#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Unbounded FIFO MPMC handoff storage. Producers and consumers have independent
// locks, so one producer and one consumer can make progress concurrently. Fixed
// size blocks amortize allocation and avoid a heap operation per item. This is
// deliberately a storage primitive: waiting, close, cancellation, deadlines
// and backpressure belong to Scheduler/IO.
//
// try_emplace() fails only when a new block cannot be allocated. try_pop()
// returns false when empty at its linearization point. approximate_size() is a
// monitoring hint derived from independently published producer and consumer
// counts; it can transiently overestimate while operations are in flight.
// Destruction requires producer/consumer quiescence.
//
// T construction, destruction and extraction are noexcept because a published
// FIFO position cannot be rolled back. Callback extraction supports
// non-assignable T; the rvalue reference is borrowed until callback return and
// the callback must not re-enter this queue. Alloc::allocate and deallocate may
// run concurrently on producer and consumer threads, so an injected allocator
// must support concurrent calls. The default SystemAllocator does.
template <class T, class Alloc = mem::SystemAllocator,
          std::size_t BlockCapacity = 64>
class UnboundedQueue {
  static_assert(BlockCapacity != 0,
                "UnboundedQueue block capacity must be nonzero");
  static_assert(std::is_nothrow_destructible_v<T>,
                "UnboundedQueue requires non-throwing destruction");

  struct Block {
    std::atomic<Block*> next{nullptr};
    std::atomic<std::size_t> published{0};
    alignas(T) std::byte storage[sizeof(T) * BlockCapacity];

    T* raw(std::size_t index) noexcept {
      return reinterpret_cast<T*>(storage + sizeof(T) * index);
    }
    T* value(std::size_t index) noexcept {
      return std::launder(raw(index));
    }
  };

 public:
  UnboundedQueue()
    requires std::is_default_constructible_v<Alloc>
  {
    init();
  }

  explicit UnboundedQueue(Alloc alloc) : alloc_(std::move(alloc)) { init(); }

  ~UnboundedQueue() { destroy_all(); }

  UnboundedQueue(const UnboundedQueue&) = delete;
  UnboundedQueue& operator=(const UnboundedQueue&) = delete;
  UnboundedQueue(UnboundedQueue&&) = delete;
  UnboundedQueue& operator=(UnboundedQueue&&) = delete;

  [[nodiscard]] bool valid() const noexcept { return head_ != nullptr; }

  template <class... Args>
    requires std::is_nothrow_constructible_v<T, Args&&...>
  bool try_emplace(Args&&... args) noexcept {
    std::lock_guard lock(tail_mutex_);
    if (HYLE_UNLIKELY(tail_ == nullptr)) return false;
    if (tail_offset_ == BlockCapacity) {
      Block* next = allocate_block();
      if (next == nullptr) return false;
      tail_->next.store(next, std::memory_order_release);
      tail_ = next;
      tail_offset_ = 0;
    }

    std::construct_at(tail_->raw(tail_offset_),
                      std::forward<Args>(args)...);
    const std::size_t published = ++tail_offset_;
    enqueued_.fetch_add(1, std::memory_order_relaxed);
    tail_->published.store(published, std::memory_order_release);
    return true;
  }

  bool try_push(const T& value) noexcept
    requires std::is_nothrow_copy_constructible_v<T>
  {
    return try_emplace(value);
  }

  bool try_push(T&& value) noexcept
    requires std::is_nothrow_move_constructible_v<T>
  {
    return try_emplace(std::move(value));
  }

  bool try_pop(T& output) noexcept
    requires std::is_nothrow_assignable_v<T&, T&&>
  {
    return consume_one([&](T&& value) noexcept { output = std::move(value); });
  }

  template <class Fn>
    requires std::is_nothrow_invocable_v<Fn&&, T&&>
  bool try_pop(Fn&& consumer) noexcept {
    return consume_one(std::forward<Fn>(consumer));
  }

  [[nodiscard]] std::size_t approximate_size() const noexcept {
    const std::size_t consumed = dequeued_.load(std::memory_order_relaxed);
    const std::size_t produced =
        enqueued_.load(std::memory_order_relaxed);
    return produced >= consumed ? produced - consumed : 0;
  }

  [[nodiscard]] bool empty() const noexcept {
    return approximate_size() == 0;
  }

 private:
  void init() noexcept {
    head_ = allocate_block();
    tail_ = head_;
  }

  Block* allocate_block() noexcept {
    void* storage = alloc_.allocate(sizeof(Block), alignof(Block));
    return storage == nullptr ? nullptr : ::new (storage) Block;
  }

  void deallocate_block(Block* block) noexcept {
    block->~Block();
    alloc_.deallocate(block, sizeof(Block), alignof(Block));
  }

  template <class Fn>
  bool consume_one(Fn&& consumer) noexcept {
    std::lock_guard lock(head_mutex_);
    if (HYLE_UNLIKELY(head_ == nullptr)) return false;

    for (;;) {
      const std::size_t published =
          head_->published.load(std::memory_order_acquire);
      if (head_offset_ < published) {
        T* value = head_->value(head_offset_++);
        std::invoke(std::forward<Fn>(consumer), std::move(*value));
        std::destroy_at(value);
        dequeued_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }

      if (head_offset_ != BlockCapacity) return false;
      Block* next = head_->next.load(std::memory_order_acquire);
      if (next == nullptr) return false;
      Block* retired = head_;
      head_ = next;
      head_offset_ = 0;
      deallocate_block(retired);
    }
  }

  void destroy_all() noexcept {
    Block* block = head_;
    std::size_t first_offset = head_offset_;
    while (block != nullptr) {
      const std::size_t published =
          block->published.load(std::memory_order_relaxed);
      for (std::size_t i = first_offset; i < published; ++i)
        std::destroy_at(block->value(i));
      Block* next = block->next.load(std::memory_order_relaxed);
      deallocate_block(block);
      block = next;
      first_offset = 0;
    }
    head_ = nullptr;
    tail_ = nullptr;
    enqueued_.store(0, std::memory_order_relaxed);
    dequeued_.store(0, std::memory_order_relaxed);
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  Block* head_ = nullptr;
  std::size_t head_offset_ = 0;
  Block* tail_ = nullptr;
  std::size_t tail_offset_ = 0;
  alignas(core::kMaxDestructiveInterferenceSize) std::mutex head_mutex_;
  alignas(core::kMaxDestructiveInterferenceSize) std::mutex tail_mutex_;
  // Producers and consumers already serialize on different locks. Keep their
  // monitoring publications on different cache lines as well; a single shared
  // size counter would add an unrelated all-worker coherence hotspot.
  alignas(core::kMaxDestructiveInterferenceSize)
      std::atomic<std::size_t> enqueued_{0};
  alignas(core::kMaxDestructiveInterferenceSize)
      std::atomic<std::size_t> dequeued_{0};
};

}  // namespace hyle::container
