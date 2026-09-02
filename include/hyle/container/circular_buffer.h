#pragma once

#include <cstddef>
#include <utility>

#include "hyle/container/ring_deque.h"

namespace hyle::container {

// Exact-capacity circular history. Unlike RingDeque, this container never grows
// when full: appending evicts the oldest element and prepending evicts the
// newest. This is the natural shape for token histories, rolling windows, and
// other "last N" state. RingDeque remains the grow-or-reject double-ended queue.
//
// The logical capacity is exactly the value supplied by the user even though
// the private RingDeque storage may round its allocation to a power of two.
// Once capacity has been established, steady push/pop/index operations are
// O(1) and allocate nothing. set_capacity() is O(n); shrinking keeps the newest
// min(size(), requested) elements.
template <class T>
class CircularBuffer {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using iterator = typename RingDeque<T>::iterator;
  using const_iterator = typename RingDeque<T>::const_iterator;
  using reverse_iterator = typename RingDeque<T>::reverse_iterator;
  using const_reverse_iterator = typename RingDeque<T>::const_reverse_iterator;
  using MutableSegments = typename RingDeque<T>::MutableSegments;
  using ConstSegments = typename RingDeque<T>::ConstSegments;

  CircularBuffer() noexcept = default;
  explicit CircularBuffer(mem::AllocatorRef alloc) noexcept : values_(alloc) {}
  explicit CircularBuffer(size_type capacity) { set_capacity(capacity); }
  CircularBuffer(mem::AllocatorRef alloc, size_type capacity) : values_(alloc) {
    set_capacity(capacity);
  }

  mem::AllocatorRef allocator() const noexcept { return values_.allocator(); }
  size_type size() const noexcept { return values_.size(); }
  size_type capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return values_.empty(); }
  bool full() const noexcept { return size() == capacity_; }

  reference operator[](size_type index) noexcept { return values_[index]; }
  const_reference operator[](size_type index) const noexcept {
    return values_[index];
  }
  reference front() noexcept { return values_.front(); }
  const_reference front() const noexcept { return values_.front(); }
  reference back() noexcept { return values_.back(); }
  const_reference back() const noexcept { return values_.back(); }

  iterator begin() noexcept { return values_.begin(); }
  iterator end() noexcept { return values_.end(); }
  const_iterator begin() const noexcept { return values_.begin(); }
  const_iterator end() const noexcept { return values_.end(); }
  const_iterator cbegin() const noexcept { return values_.cbegin(); }
  const_iterator cend() const noexcept { return values_.cend(); }
  reverse_iterator rbegin() noexcept { return values_.rbegin(); }
  reverse_iterator rend() noexcept { return values_.rend(); }
  const_reverse_iterator rbegin() const noexcept { return values_.rbegin(); }
  const_reverse_iterator rend() const noexcept { return values_.rend(); }

  MutableSegments segments() noexcept { return values_.segments(); }
  ConstSegments segments() const noexcept { return values_.segments(); }
  bool is_linearized() const noexcept { return values_.is_linearized(); }

  bool try_set_capacity(size_type requested) noexcept {
    if (requested == capacity_) return true;
    if (requested > capacity_) {
      if (!values_.try_reserve(requested)) return false;
      capacity_ = requested;
      return true;
    }

    RingDeque<T> replacement(values_.allocator());
    if (!replacement.try_reserve(requested)) return false;
    const size_type keep = size() < requested ? size() : requested;
    const size_type first = size() - keep;
    for (size_type i = first; i < size(); ++i) {
      const bool inserted =
          replacement.try_emplace_back_bounded(std::move(values_[i]));
      HYLE_FATAL_ASSERT(inserted);
    }
    values_.swap(replacement);
    capacity_ = requested;
    return true;
  }

  void set_capacity(size_type requested) {
    HYLE_FATAL_ASSERT(try_set_capacity(requested));
  }

  template <class... Args>
  bool try_emplace_back(Args&&... args) {
    if (capacity_ == 0) return false;
    if (!full()) {
      return values_.try_emplace_back_bounded(std::forward<Args>(args)...);
    }
    // Preserve inputs that alias an element which is about to be evicted.
    T saved(std::forward<Args>(args)...);
    values_.pop_front();
    return values_.try_emplace_back_bounded(std::move(saved));
  }

  template <class... Args>
  bool try_emplace_front(Args&&... args) {
    if (capacity_ == 0) return false;
    if (!full()) {
      return values_.try_emplace_front_bounded(std::forward<Args>(args)...);
    }
    T saved(std::forward<Args>(args)...);
    values_.pop_back();
    return values_.try_emplace_front_bounded(std::move(saved));
  }

  bool try_push_back(const T& value) { return try_emplace_back(value); }
  bool try_push_back(T&& value) { return try_emplace_back(std::move(value)); }
  bool try_push_front(const T& value) { return try_emplace_front(value); }
  bool try_push_front(T&& value) { return try_emplace_front(std::move(value)); }

  template <class... Args>
  reference emplace_back(Args&&... args) {
    HYLE_FATAL_ASSERT(try_emplace_back(std::forward<Args>(args)...));
    return back();
  }
  template <class... Args>
  reference emplace_front(Args&&... args) {
    HYLE_FATAL_ASSERT(try_emplace_front(std::forward<Args>(args)...));
    return front();
  }

  void push_back(const T& value) { emplace_back(value); }
  void push_back(T&& value) { emplace_back(std::move(value)); }
  void push_front(const T& value) { emplace_front(value); }
  void push_front(T&& value) { emplace_front(std::move(value)); }

  void pop_front() noexcept { values_.pop_front(); }
  void pop_back() noexcept { values_.pop_back(); }
  void clear() noexcept { values_.clear(); }

  void swap(CircularBuffer& other) noexcept {
    values_.swap(other.values_);
    using std::swap;
    swap(capacity_, other.capacity_);
  }

 private:
  RingDeque<T> values_;
  size_type capacity_ = 0;
};

template <class T>
void swap(CircularBuffer<T>& left, CircularBuffer<T>& right) noexcept {
  left.swap(right);
}

}  // namespace hyle::container
