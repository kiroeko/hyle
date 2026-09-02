#pragma once

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

namespace detail {
struct SmallVectorAccess;

// Keep a real inactive T[N] array boundary without constructing elements.
// The N==0 specialization deliberately has no dummy T-sized/aligned storage;
// a zero-inline vector starts with a null data pointer and allocates before the
// first element lifetime begins.
template <class T, std::size_t N>
union SmallVectorElements {
  constexpr SmallVectorElements() noexcept {}
  ~SmallVectorElements() noexcept {}

  T values[N];
};

template <class T, std::size_t N, bool = (N == 0)>
struct SmallVectorInlineStorage {
  SmallVectorElements<T, N> elements;
};

template <class T, std::size_t N>
struct SmallVectorInlineStorage<T, N, true> {};
}

// Vector with storage for the first N elements embedded in the object. It is a
// std::vector-like owning container for genuinely small sequences, not a POD
// benchmark fixture: non-trivial values are constructed, moved and destroyed
// normally. Trivially copyable values retain the memcpy relocation fast path.
//
// Growth must be recoverable without leaving the original vector half-moved,
// so T's move construction and destruction must not throw. This includes the
// normal systems types (strings, smart pointers, handles and PODs) while making
// the zero-exception contract explicit. Copy construction may throw; normal C++
// unwinding destroys the completed prefix.
//
// Alloc defaults to the zero-state SystemAllocator. Pass mem::Allocator for a
// runtime arena/mock/cross-DLL allocator. Allocation failure is available as a
// bool through try_reserve()/try_emplace_back(); the std-shaped convenience
// operations fail fast because they have no error return channel.
template <class T, std::size_t N, class Alloc = mem::SystemAllocator>
class SmallVector {
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "SmallVector requires nothrow move construction");
  static_assert(std::is_nothrow_destructible_v<T>,
                "SmallVector requires nothrow destruction");
  static_assert(std::is_nothrow_move_constructible_v<Alloc>,
                "SmallVector requires nothrow allocator move construction");
  static_assert(std::is_nothrow_move_assignable_v<Alloc>,
                "SmallVector requires nothrow allocator move assignment");

public:
  using value_type = T;
  using allocator_type = Alloc;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  SmallVector() noexcept(std::is_nothrow_default_constructible_v<Alloc>)
    requires std::is_default_constructible_v<Alloc>
      : cap_(N) {
    data_ = inline_ptr();
  }

  explicit SmallVector(Alloc alloc) noexcept(
      std::is_nothrow_move_constructible_v<Alloc>)
      : alloc_(std::move(alloc)), cap_(N) {
    data_ = inline_ptr();
  }

  explicit SmallVector(size_type count)
    requires std::is_default_constructible_v<Alloc> &&
             std::is_default_constructible_v<T>
      : SmallVector() {
    construct_with_cleanup([&] { resize(count); });
  }

  SmallVector(size_type count, const T &value)
    requires std::is_default_constructible_v<Alloc> &&
             std::is_copy_constructible_v<T>
      : SmallVector() {
    construct_with_cleanup([&] { resize(count, value); });
  }

  SmallVector(Alloc alloc, size_type count)
    requires std::is_default_constructible_v<T>
      : SmallVector(std::move(alloc)) {
    construct_with_cleanup([&] { resize(count); });
  }

  SmallVector(Alloc alloc, size_type count, const T &value)
    requires std::is_copy_constructible_v<T>
      : SmallVector(std::move(alloc)) {
    construct_with_cleanup([&] { resize(count, value); });
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  SmallVector(It first, Sentinel last)
    requires std::is_default_constructible_v<Alloc>
      : SmallVector() {
    construct_with_cleanup([&] { append_range(first, last); });
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  SmallVector(Alloc alloc, It first, Sentinel last)
      : SmallVector(std::move(alloc)) {
    construct_with_cleanup([&] { append_range(first, last); });
  }

  SmallVector(std::initializer_list<T> init)
    requires std::is_default_constructible_v<Alloc>
      : SmallVector() {
    construct_with_cleanup([&] { append_range(init.begin(), init.end()); });
  }

  SmallVector(Alloc alloc, std::initializer_list<T> init)
      : SmallVector(std::move(alloc)) {
    construct_with_cleanup([&] { append_range(init.begin(), init.end()); });
  }

  SmallVector(const SmallVector &other) : SmallVector(other.alloc_) {
    construct_with_cleanup(
        [&] { append_range(other.begin(), other.end()); });
  }

  SmallVector(SmallVector &&other) noexcept
      : SmallVector(std::move(other.alloc_)) {
    move_from(other);
  }

  SmallVector &operator=(const SmallVector &other) {
    if (this == &other)
      return *this;
    // Build first so a throwing T copy leaves *this unchanged.
    SmallVector replacement(alloc_);
    replacement.append_range(other.begin(), other.end());
    *this = std::move(replacement);
    return *this;
  }

  SmallVector &operator=(SmallVector &&other) noexcept {
    if (this == &other)
      return *this;
    reset_storage();
    alloc_ = std::move(other.alloc_);
    move_from(other);
    return *this;
  }

  ~SmallVector() { reset_storage(); }

  allocator_type &allocator() noexcept { return alloc_; }
  const allocator_type &allocator() const noexcept { return alloc_; }

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return cap_; }
  static constexpr size_type inline_capacity() noexcept { return N; }
  bool empty() const noexcept { return size_ == 0; }
  bool inline_storage() const noexcept { return data_ == inline_ptr(); }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }
  reference operator[](size_type i) noexcept { return data_[i]; }
  const_reference operator[](size_type i) const noexcept { return data_[i]; }
  reference front() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data_[0];
  }
  const_reference front() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data_[0];
  }
  reference back() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data_[size_ - 1];
  }
  const_reference back() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data_[size_ - 1];
  }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }
  iterator end() noexcept { return pointer_at(size_); }
  const_iterator end() const noexcept { return pointer_at(size_); }
  const_iterator cend() const noexcept { return pointer_at(size_); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

  bool try_reserve(size_type requested) noexcept {
    return requested <= cap_ || grow_to(requested);
  }

  void reserve(size_type requested) {
    HYLE_FATAL_ASSERT(try_reserve(requested));
  }

  template <class... Args> bool try_emplace_back(Args &&...args) {
    if (size_ == cap_) {
      if (size_ == static_cast<size_type>(-1) / sizeof(T))
        return false;
      // Construct first: args may refer to an existing element and grow_to()
      // relocates it. This is the same normal-use guarantee as std::vector and
      // Folly small_vector, paid only on the rare growth path.
      T saved(std::forward<Args>(args)...);
      if (!grow_to(next_capacity(size_ + 1)))
        return false;
      ::new (static_cast<void *>(data_ + size_)) T(std::move(saved));
      ++size_;
      return true;
    }
    ::new (static_cast<void *>(data_ + size_)) T(std::forward<Args>(args)...);
    ++size_;
    return true;
  }

  template <class... Args> reference emplace_back(Args &&...args) {
    HYLE_FATAL_ASSERT(try_emplace_back(std::forward<Args>(args)...));
    return back();
  }

  // Preserve the std::vector aliasing shape: push_back(v[0]) remains valid
  // when growth relocates the source element.
  bool try_push_back(const T &value) {
    if (size_ != cap_) {
      ::new (static_cast<void *>(data_ + size_)) T(value);
      ++size_;
      return true;
    }
    T saved(value);
    if (!grow_to(next_capacity(size_ + 1)))
      return false;
    ::new (static_cast<void *>(data_ + size_)) T(std::move(saved));
    ++size_;
    return true;
  }

  bool try_push_back(T &&value) {
    if (size_ != cap_) {
      ::new (static_cast<void *>(data_ + size_)) T(std::move(value));
      ++size_;
      return true;
    }
    T saved(std::move(value));
    if (!grow_to(next_capacity(size_ + 1)))
      return false;
    ::new (static_cast<void *>(data_ + size_)) T(std::move(saved));
    ++size_;
    return true;
  }

  void push_back(const T &value) { HYLE_FATAL_ASSERT(try_push_back(value)); }
  void push_back(T &&value) {
    HYLE_FATAL_ASSERT(try_push_back(std::move(value)));
  }

  template <class... Args>
  iterator emplace(const_iterator position, Args &&...args) {
    const size_type index = position_index(position, true);
    if (index == size_) {
      emplace_back(std::forward<Args>(args)...);
      return data_ + index;
    }

    // Construct before growth or shifting so arguments may alias any current
    // element. The saved value also gives the operation the strong guarantee
    // when its user-defined construction throws.
    T saved(std::forward<Args>(args)...);
    if (size_ == cap_)
      reserve(next_capacity(size_ + 1));

    open_gap(index, 1);
    ::new (static_cast<void *>(data_ + index)) T(std::move(saved));
    ++size_;
    return data_ + index;
  }

  iterator insert(const_iterator position, const T &value) {
    return emplace(position, value);
  }

  iterator insert(const_iterator position, T &&value) {
    return emplace(position, std::move(value));
  }

  iterator insert(const_iterator position, size_type count, const T &value) {
    const size_type index = position_index(position, true);
    SmallVector inserted(alloc_);
    inserted.resize(count, value);
    return insert_prepared(index, inserted);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  iterator insert(const_iterator position, It first, Sentinel last) {
    const size_type index = position_index(position, true);
    // Materialize first so self-ranges remain valid across reserve/shift and
    // input iterators are consumed exactly once.
    SmallVector inserted(alloc_);
    inserted.append_range(first, last);
    return insert_prepared(index, inserted);
  }

  iterator insert(const_iterator position, std::initializer_list<T> init) {
    return insert(position, init.begin(), init.end());
  }

  void assign(size_type count, const T &value) {
    SmallVector replacement(alloc_);
    replacement.resize(count, value);
    *this = std::move(replacement);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  void assign(It first, Sentinel last) {
    SmallVector replacement(alloc_);
    replacement.append_range(first, last);
    *this = std::move(replacement);
  }

  void assign(std::initializer_list<T> init) {
    assign(init.begin(), init.end());
  }

  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    --size_;
    data_[size_].~T();
  }

  void clear() noexcept {
    destroy_range(data_, size_);
    size_ = 0;
  }

  bool try_resize(size_type count) {
    if (count < size_) {
      destroy_range(data_ + count, size_ - count);
      size_ = count;
      return true;
    }
    if (count > cap_ && !grow_to(count))
      return false;
    while (size_ < count) {
      ::new (static_cast<void *>(data_ + size_)) T();
      ++size_;
    }
    return true;
  }

  bool try_resize(size_type count, const T &value) {
    if (count <= size_)
      return try_resize(count);
    // value may alias this vector and growth may relocate it.
    if (count > cap_) {
      T saved(value);
      if (!grow_to(count))
        return false;
      while (size_ < count) {
        ::new (static_cast<void *>(data_ + size_)) T(saved);
        ++size_;
      }
      return true;
    }
    while (size_ < count) {
      ::new (static_cast<void *>(data_ + size_)) T(value);
      ++size_;
    }
    return true;
  }

  void resize(size_type count) { HYLE_FATAL_ASSERT(try_resize(count)); }
  void resize(size_type count, const T &value) {
    HYLE_FATAL_ASSERT(try_resize(count, value));
  }

  bool try_shrink_to_fit() noexcept {
    if (inline_storage() || cap_ == size_)
      return true;
    if (size_ <= N) {
      relocate_to(inline_ptr(), N);
      return true;
    }
    pointer replacement = static_cast<pointer>(
        alloc_.allocate(size_ * sizeof(T), alignof(T)));
    if (replacement == nullptr)
      return false;
    relocate_to(replacement, size_);
    return true;
  }

  void shrink_to_fit() { HYLE_FATAL_ASSERT(try_shrink_to_fit()); }

  void swap(SmallVector &other) noexcept {
    if (this == &other)
      return;
    SmallVector temporary(std::move(*this));
    *this = std::move(other);
    other = std::move(temporary);
  }

  friend void swap(SmallVector &left, SmallVector &right) noexcept {
    left.swap(right);
  }

  friend bool operator==(const SmallVector &left, const SmallVector &right)
    requires std::equality_comparable<T>
  {
    return left.size_ == right.size_ &&
           (left.size_ == 0 ||
            std::equal(left.begin(), left.end(), right.begin()));
  }

  friend auto operator<=>(const SmallVector &left, const SmallVector &right)
    requires std::three_way_comparable<T>
  {
    return std::lexicographical_compare_three_way(
        left.begin(), left.end(), right.begin(), right.end());
  }

  iterator erase(const_iterator position) noexcept {
    const size_type index = position_index(position, false);
    return erase(pointer_at(index), pointer_at(index + 1));
  }

  iterator erase(const_iterator first, const_iterator last) noexcept {
    const size_type first_index = position_index(first, true);
    const size_type last_index = position_index(last, true);
    HYLE_FATAL_ASSERT(first_index <= last_index);
    const size_type count = last_index - first_index;
    if (count == 0)
      return pointer_at(first_index);

    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove(data_ + first_index, data_ + first_index + count,
                   (size_ - first_index - count) * sizeof(T));
    } else {
      destroy_range(data_ + first_index, count);
      for (size_type i = first_index; i + count < size_; ++i) {
        ::new (static_cast<void *>(data_ + i)) T(std::move(data_[i + count]));
        data_[i + count].~T();
      }
    }
    size_ -= count;
    return pointer_at(first_index);
  }

private:
  friend struct detail::SmallVectorAccess;

  pointer inline_ptr() noexcept {
    if constexpr (N == 0)
      return nullptr;
    else
      return inline_.elements.values;
  }
  const_pointer inline_ptr() const noexcept {
    if constexpr (N == 0)
      return nullptr;
    else
      return inline_.elements.values;
  }

  pointer pointer_at(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index <= size_);
    return data_ == nullptr ? nullptr : data_ + index;
  }
  const_pointer pointer_at(size_type index) const noexcept {
    HYLE_FATAL_ASSERT(index <= size_);
    return data_ == nullptr ? nullptr : data_ + index;
  }

  size_type position_index(const_iterator position,
                           bool allow_end) const noexcept {
    if (data_ == nullptr) {
      HYLE_FATAL_ASSERT(position == nullptr && allow_end);
      return 0;
    }
    HYLE_FATAL_ASSERT(position >= cbegin() &&
                      (allow_end ? position <= cend() : position < cend()));
    return static_cast<size_type>(position - cbegin());
  }

  static void destroy_range(pointer first, size_type count) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_type i = count; i != 0; --i)
        first[i - 1].~T();
    }
  }

  template <class Fn> void construct_with_cleanup(Fn &&fn) {
    try {
      std::forward<Fn>(fn)();
    } catch (...) {
      reset_storage();
      throw;
    }
  }

  iterator insert_prepared(size_type index, SmallVector &inserted) noexcept {
    const size_type count = inserted.size_;
    if (count == 0)
      return pointer_at(index);
    HYLE_FATAL_ASSERT(
        count <= static_cast<size_type>(-1) / sizeof(T) - size_);
    const size_type requested = size_ + count;
    if (requested > cap_)
      reserve(next_capacity(requested));
    open_gap(index, count);
    for (size_type i = 0; i < count; ++i)
      ::new (static_cast<void *>(data_ + index + i))
          T(std::move(inserted.data_[i]));
    size_ = requested;
    return pointer_at(index);
  }

  void open_gap(size_type index, size_type count) noexcept {
    HYLE_FATAL_ASSERT(index <= size_ && count <= cap_ - size_);
    if (count == 0)
      return;
    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove(data_ + index + count, data_ + index,
                   (size_ - index) * sizeof(T));
    } else {
      for (size_type i = size_; i > index; --i) {
        ::new (static_cast<void *>(data_ + i - 1 + count))
            T(std::move(data_[i - 1]));
        data_[i - 1].~T();
      }
    }
  }

  size_type next_capacity(size_type required) const noexcept {
    HYLE_FATAL_ASSERT(required <= static_cast<size_type>(-1) / sizeof(T));
    if (cap_ > static_cast<size_type>(-1) / 2)
      return required;
    const size_type doubled = cap_ * 2;
    return doubled < required ? required : doubled;
  }

  bool grow_to(size_type requested) noexcept {
    if (requested < size_ || requested > static_cast<size_type>(-1) / sizeof(T))
      return false;
    pointer replacement = static_cast<pointer>(
        alloc_.allocate(requested * sizeof(T), alignof(T)));
    if (replacement == nullptr)
      return false;

    relocate_to(replacement, requested);
    return true;
  }

  void relocate_to(pointer replacement,
                   size_type replacement_capacity) noexcept {
    pointer old_data = data_;
    const size_type old_capacity = cap_;
    const bool old_inline = inline_storage();
    if constexpr (std::is_trivially_copyable_v<T>) {
      if (size_ != 0)
        std::memcpy(replacement, old_data, size_ * sizeof(T));
    } else {
      for (size_type i = 0; i < size_; ++i)
        ::new (static_cast<void *>(replacement + i))
            T(std::move(old_data[i]));
      destroy_range(old_data, size_);
    }

    if (!old_inline)
      alloc_.deallocate(old_data, old_capacity * sizeof(T), alignof(T));
    data_ = replacement;
    cap_ = replacement_capacity;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  void append_range(It first, Sentinel last) {
    HYLE_FATAL_ASSERT(try_append_construct_unaliased(
        std::move(first), std::move(last),
        [](void *storage, auto &&value) noexcept(
            std::is_nothrow_constructible_v<T, decltype(value)>) {
          ::new (storage) T(std::forward<decltype(value)>(value));
        }));
  }

  // Internal bulk-construction seam shared by other owning container engines.
  // The input must not alias this vector: callers use it while constructing a
  // fresh replacement. A forward range reserves once and publishes size only
  // after the whole batch becomes live, avoiding one capacity branch and size
  // write per item while retaining rollback on allocation or construction
  // failure. Construct receives (void* destination, input-reference) and must
  // placement-construct exactly one T.
  template <std::input_iterator It, std::sentinel_for<It> Sentinel,
            class Construct>
  bool try_append_construct_unaliased(It first, Sentinel last,
                                      Construct construct) {
    const size_type original_size = size_;
    if constexpr (std::forward_iterator<It>) {
      size_type additional = 0;
      const size_type remaining =
          static_cast<size_type>(-1) / sizeof(T) - size_;
      if constexpr (std::sized_sentinel_for<Sentinel, It>) {
        const auto distance = last - first;
        if (distance < 0 ||
            static_cast<std::make_unsigned_t<decltype(distance)>>(distance) >
                remaining)
          return false;
        additional = static_cast<size_type>(distance);
      } else {
        for (It probe = first; probe != last; ++probe) {
          if (additional == remaining)
            return false;
          ++additional;
        }
      }
      if (!try_reserve(size_ + additional))
        return false;
      size_type constructed = 0;
      try {
        for (; first != last; ++first) {
          construct(static_cast<void *>(data_ + original_size + constructed),
                    *first);
          ++constructed;
        }
      } catch (...) {
        destroy_range(data_ + original_size, constructed);
        throw;
      }
      size_ = original_size + constructed;
      return true;
    } else {
      try {
        for (; first != last; ++first) {
          if (size_ == static_cast<size_type>(-1) / sizeof(T) ||
              (size_ == cap_ && !grow_to(next_capacity(size_ + 1)))) {
            destroy_range(data_ + original_size, size_ - original_size);
            size_ = original_size;
            return false;
          }
          construct(static_cast<void *>(data_ + size_), *first);
          ++size_;
        }
      } catch (...) {
        destroy_range(data_ + original_size, size_ - original_size);
        size_ = original_size;
        throw;
      }
      return true;
    }
  }

  void move_from(SmallVector &other) noexcept {
    if (!other.inline_storage()) {
      data_ = other.data_;
      size_ = other.size_;
      cap_ = other.cap_;
      other.data_ = other.inline_ptr();
      other.size_ = 0;
      other.cap_ = N;
      return;
    }
    if constexpr (N == 0) {
      // An inline N==0 vector is necessarily empty and both inline pointers are
      // null. Do not form memcpy(nullptr, nullptr, 0): besides being needless,
      // fortified libc headers correctly reject its pointer preconditions.
      size_ = 0;
      other.size_ = 0;
    } else if constexpr (std::is_trivially_copyable_v<T>) {
      // inline_storage() implies size_ <= N. GCC's interprocedural
      // -Warray-bounds analysis can lose that invariant when this path is
      // inlined through the heap/inline swap, then diagnose the valid copy as
      // if a spilled size could reach the fixed inline array. Keep the exact
      // zero-overhead move and suppress only that proven false positive.
#if HYLE_COMPILER_GCC
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Warray-bounds"
#  pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
      if (other.size_ != 0)
        std::memcpy(data_, other.data_, other.size_ * sizeof(T));
#if HYLE_COMPILER_GCC
#  pragma GCC diagnostic pop
#endif
      size_ = other.size_;
      other.size_ = 0;
    } else {
      for (size_type i = 0; i < other.size_; ++i) {
        ::new (static_cast<void *>(data_ + i)) T(std::move(other.data_[i]));
        ++size_;
      }
      other.clear();
    }
  }

  void reset_storage() noexcept {
    clear();
    if (!inline_storage())
      alloc_.deallocate(data_, cap_ * sizeof(T), alignof(T));
    data_ = inline_ptr();
    cap_ = N;
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  pointer data_ = nullptr;
  size_type size_ = 0;
  size_type cap_ = N;
  HYLE_NO_UNIQUE_ADDRESS detail::SmallVectorInlineStorage<T, N> inline_{};
};

namespace detail {

// Narrow internal seam: FlatSortedStorage can bulk-construct a fresh sequence
// from a projected range without exposing uninitialized storage or size
// mutation in SmallVector's public API.
struct SmallVectorAccess {
  template <class T, std::size_t N, class Alloc, std::input_iterator It,
            std::sentinel_for<It> Sentinel, class Construct>
  static bool try_append_construct_unaliased(
      SmallVector<T, N, Alloc> &vector, It first, Sentinel last,
      Construct construct) {
    return vector.try_append_construct_unaliased(
        std::move(first), std::move(last), std::move(construct));
  }

  template <class T, std::size_t N, class Alloc>
  static bool try_reserve_growth(SmallVector<T, N, Alloc> &vector,
                                 std::size_t required) noexcept {
    return required <= vector.cap_ ||
           vector.grow_to(vector.next_capacity(required));
  }
};

} // namespace detail

} // namespace hyle::container
