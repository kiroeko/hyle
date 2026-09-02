#pragma once

#include <bit>
#include <compare>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// AllocatorRef-aware, single-threaded circular deque. The storage is a
// power-of-two ring so both-end mutation and random access are O(1). Normal
// try_emplace_* operations grow on demand; the *_bounded forms never allocate
// and provide fixed-capacity/backpressure semantics after reserve(). Growth or
// reserve invalidates every reference; mutation invalidates iterators/segments,
// while a no-growth push preserves references to existing elements and pop only
// invalidates references to erased elements.
template <class T> class RingDeque {
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "RingDeque requires nothrow move construction");
  static_assert(std::is_nothrow_destructible_v<T>,
                "RingDeque requires nothrow destruction");

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  struct MutableSegments {
    std::span<T> first;
    std::span<T> second;
  };
  struct ConstSegments {
    std::span<const T> first;
    std::span<const T> second;
  };

private:
  template <bool IsConst> class BasicIterator {
    template <bool> friend class BasicIterator;
    friend class RingDeque;
    using Owner = std::conditional_t<IsConst, const RingDeque, RingDeque>;

    BasicIterator(Owner *owner, size_type index) noexcept
        : owner_(owner), index_(index) {}

  public:
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = std::conditional_t<IsConst, const T &, T &>;
    using pointer = std::conditional_t<IsConst, const T *, T *>;

    BasicIterator() noexcept = default;
    template <bool OtherConst>
      requires(IsConst && !OtherConst)
    BasicIterator(const BasicIterator<OtherConst> &other) noexcept
        : owner_(other.owner_), index_(other.index_) {}

    reference operator*() const noexcept { return (*owner_)[index_]; }
    pointer operator->() const noexcept { return std::addressof(**this); }
    reference operator[](difference_type offset) const noexcept {
      return *(*this + offset);
    }
    BasicIterator &operator++() noexcept {
      ++index_;
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    BasicIterator &operator--() noexcept {
      --index_;
      return *this;
    }
    BasicIterator operator--(int) noexcept {
      BasicIterator copy = *this;
      --*this;
      return copy;
    }
    BasicIterator &operator+=(difference_type offset) noexcept {
      index_ =
          static_cast<size_type>(static_cast<difference_type>(index_) + offset);
      return *this;
    }
    BasicIterator &operator-=(difference_type offset) noexcept {
      return *this += -offset;
    }
    friend BasicIterator operator+(BasicIterator iterator,
                                   difference_type offset) noexcept {
      iterator += offset;
      return iterator;
    }
    friend BasicIterator operator+(difference_type offset,
                                   BasicIterator iterator) noexcept {
      return iterator + offset;
    }
    friend BasicIterator operator-(BasicIterator iterator,
                                   difference_type offset) noexcept {
      iterator -= offset;
      return iterator;
    }
    friend difference_type operator-(BasicIterator left,
                                     BasicIterator right) noexcept {
      return static_cast<difference_type>(left.index_) -
             static_cast<difference_type>(right.index_);
    }
    friend bool operator==(BasicIterator, BasicIterator) noexcept = default;
    friend auto operator<=>(BasicIterator left, BasicIterator right) noexcept {
      return left.index_ <=> right.index_;
    }

  private:
    Owner *owner_ = nullptr;
    size_type index_ = 0;
  };

public:
  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  RingDeque() noexcept : alloc_(mem::system_allocator()) {}
  explicit RingDeque(mem::AllocatorRef alloc) noexcept : alloc_(alloc) {}
  explicit RingDeque(size_type capacity) : RingDeque() { reserve(capacity); }
  RingDeque(mem::AllocatorRef alloc, size_type capacity) : RingDeque(alloc) {
    reserve(capacity);
  }

  RingDeque(const RingDeque &other) : RingDeque(other.alloc_) {
    copy_from(other);
  }
  RingDeque &operator=(const RingDeque &other) {
    if (this == &other)
      return *this;
    RingDeque replacement(alloc_);
    replacement.copy_from(other);
    swap(replacement);
    return *this;
  }
  RingDeque(RingDeque &&other) noexcept
      : alloc_(other.alloc_), storage_(std::exchange(other.storage_, nullptr)),
        capacity_(std::exchange(other.capacity_, 0)),
        head_(std::exchange(other.head_, 0)),
        tail_(std::exchange(other.tail_, 0)) {}
  RingDeque &operator=(RingDeque &&other) noexcept {
    if (this == &other)
      return *this;
    release();
    alloc_ = other.alloc_;
    storage_ = std::exchange(other.storage_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
    head_ = std::exchange(other.head_, 0);
    tail_ = std::exchange(other.tail_, 0);
    return *this;
  }
  ~RingDeque() { release(); }

  mem::AllocatorRef allocator() const noexcept { return alloc_; }
  size_type size() const noexcept { return tail_ - head_; }
  size_type capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return head_ == tail_; }
  bool full() const noexcept { return size() == capacity_; }

  reference operator[](size_type index) noexcept {
    return storage_[physical_index(index)];
  }
  const_reference operator[](size_type index) const noexcept {
    return storage_[physical_index(index)];
  }
  reference front() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return storage_[head_ & (capacity_ - 1)];
  }
  const_reference front() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return storage_[head_ & (capacity_ - 1)];
  }
  reference back() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return (*this)[size() - 1];
  }
  const_reference back() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return (*this)[size() - 1];
  }

  iterator begin() noexcept { return iterator(this, 0); }
  iterator end() noexcept { return iterator(this, size()); }
  const_iterator begin() const noexcept { return const_iterator(this, 0); }
  const_iterator end() const noexcept { return const_iterator(this, size()); }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }

  MutableSegments segments() noexcept {
    const size_type active = size();
    if (active == 0)
      return {};
    const size_type first = head_ & (capacity_ - 1);
    const size_type first_size =
        active < capacity_ - first ? active : capacity_ - first;
    return {{storage_ + first, first_size}, {storage_, active - first_size}};
  }
  ConstSegments segments() const noexcept {
    const size_type active = size();
    if (active == 0)
      return {};
    const size_type first = head_ & (capacity_ - 1);
    const size_type first_size =
        active < capacity_ - first ? active : capacity_ - first;
    return {{storage_ + first, first_size}, {storage_, active - first_size}};
  }
  bool is_linearized() const noexcept {
    if (empty())
      return true;
    const size_type first = head_ & (capacity_ - 1);
    return first + size() <= capacity_;
  }

  bool try_reserve(size_type requested) noexcept {
    if (requested <= capacity_)
      return true;
    const size_type allocation_limit = static_cast<size_type>(-1) / sizeof(T);
    const size_type iterator_limit =
        static_cast<size_type>(std::numeric_limits<difference_type>::max());
    const size_type maximum = std::bit_floor(
        allocation_limit < iterator_limit ? allocation_limit : iterator_limit);
    if (!alloc_.valid() || requested > maximum)
      return false;
    const size_type target = std::bit_ceil(requested);
    T *replacement = alloc_.allocate_n<T>(target);
    if (replacement == nullptr)
      return false;

    if constexpr (std::is_trivially_copyable_v<T>) {
      const ConstSegments source = std::as_const(*this).segments();
      if (!source.first.empty())
        std::memcpy(replacement, source.first.data(),
                    source.first.size_bytes());
      if (!source.second.empty())
        std::memcpy(replacement + source.first.size(), source.second.data(),
                    source.second.size_bytes());
    } else {
      for (size_type i = 0; i < size(); ++i)
        std::construct_at(replacement + i, std::move((*this)[i]));
    }
    const size_type active = size();
    destroy_elements();
    if (storage_ != nullptr)
      alloc_.deallocate_n(storage_, capacity_);
    storage_ = replacement;
    capacity_ = target;
    head_ = 0;
    tail_ = active;
    return true;
  }
  void reserve(size_type requested) {
    HYLE_FATAL_ASSERT(try_reserve(requested));
  }

  template <class... Args> bool try_emplace_back_bounded(Args &&...args) {
    if (full())
      return false;
    const size_type current_tail = tail_;
    std::construct_at(storage_ + (current_tail & (capacity_ - 1)),
                      std::forward<Args>(args)...);
    // The injected allocator may make storage_ opaque to the optimizer. Assign
    // from the saved counter so a slot store cannot force a tail_ reload.
    tail_ = current_tail + 1;
    return true;
  }
  template <class... Args> bool try_emplace_front_bounded(Args &&...args) {
    if (full())
      return false;
    const size_type new_head = head_ - 1;
    std::construct_at(storage_ + (new_head & (capacity_ - 1)),
                      std::forward<Args>(args)...);
    head_ = new_head;
    return true;
  }
  template <class... Args> bool try_emplace_back(Args &&...args) {
    if (!full())
      return try_emplace_back_bounded(std::forward<Args>(args)...);
    T saved(std::forward<Args>(args)...);
    if (!grow_for_one())
      return false;
    return try_emplace_back_bounded(std::move(saved));
  }
  template <class... Args> bool try_emplace_front(Args &&...args) {
    if (!full())
      return try_emplace_front_bounded(std::forward<Args>(args)...);
    T saved(std::forward<Args>(args)...);
    if (!grow_for_one())
      return false;
    return try_emplace_front_bounded(std::move(saved));
  }

  bool try_push_back(const T &value) { return try_emplace_back(value); }
  bool try_push_back(T &&value) { return try_emplace_back(std::move(value)); }
  bool try_push_front(const T &value) { return try_emplace_front(value); }
  bool try_push_front(T &&value) { return try_emplace_front(std::move(value)); }
  bool try_push_back_bounded(const T &value) {
    return try_emplace_back_bounded(value);
  }
  bool try_push_back_bounded(T &&value) {
    return try_emplace_back_bounded(std::move(value));
  }
  bool try_push_front_bounded(const T &value) {
    return try_emplace_front_bounded(value);
  }
  bool try_push_front_bounded(T &&value) {
    return try_emplace_front_bounded(std::move(value));
  }

  template <class... Args> reference emplace_back(Args &&...args) {
    HYLE_FATAL_ASSERT(try_emplace_back(std::forward<Args>(args)...));
    return back();
  }
  template <class... Args> reference emplace_front(Args &&...args) {
    HYLE_FATAL_ASSERT(try_emplace_front(std::forward<Args>(args)...));
    return front();
  }
  void push_back(const T &value) { emplace_back(value); }
  void push_back(T &&value) { emplace_back(std::move(value)); }
  void push_front(const T &value) { emplace_front(value); }
  void push_front(T &&value) { emplace_front(std::move(value)); }

  void pop_front() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    const size_type current_head = head_;
    std::destroy_at(storage_ + (current_head & (capacity_ - 1)));
    head_ = current_head + 1;
  }
  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    const size_type new_tail = tail_ - 1;
    std::destroy_at(storage_ + (new_tail & (capacity_ - 1)));
    tail_ = new_tail;
  }
  void clear() noexcept {
    destroy_elements();
    head_ = tail_ = 0;
  }

  void swap(RingDeque &other) noexcept {
    using std::swap;
    swap(alloc_, other.alloc_);
    swap(storage_, other.storage_);
    swap(capacity_, other.capacity_);
    swap(head_, other.head_);
    swap(tail_, other.tail_);
  }

private:
  size_type physical_index(size_type logical) const noexcept {
    return (head_ + logical) & (capacity_ - 1);
  }
  bool grow_for_one() noexcept {
    const size_type active = size();
    if (active == static_cast<size_type>(-1))
      return false;
    return try_reserve(active + 1);
  }
  void copy_from(const RingDeque &other) {
    reserve(other.size());
    try {
      for (const T &value : other)
        HYLE_FATAL_ASSERT(try_emplace_back_bounded(value));
    } catch (...) {
      release();
      throw;
    }
  }
  void destroy_elements() noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>)
      for (size_type i = 0; i < size(); ++i)
        std::destroy_at(storage_ + physical_index(i));
  }
  void release() noexcept {
    destroy_elements();
    if (storage_ != nullptr)
      alloc_.deallocate_n(storage_, capacity_);
    storage_ = nullptr;
    capacity_ = 0;
    head_ = tail_ = 0;
  }

  mem::AllocatorRef alloc_;
  T *storage_ = nullptr;
  size_type capacity_ = 0;
  size_type head_ = 0;
  size_type tail_ = 0;
};

template <class T> void swap(RingDeque<T> &left, RingDeque<T> &right) noexcept {
  left.swap(right);
}

} // namespace hyle::container
