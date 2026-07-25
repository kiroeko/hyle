#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

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
  static_assert(N >= 1, "SmallVector requires non-zero inline capacity");
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "SmallVector requires nothrow move construction");
  static_assert(std::is_nothrow_destructible_v<T>,
                "SmallVector requires nothrow destruction");

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
      : data_(inline_ptr()), cap_(N) {}

  explicit SmallVector(Alloc alloc) noexcept(
      std::is_nothrow_move_constructible_v<Alloc>)
      : alloc_(std::move(alloc)), data_(inline_ptr()), cap_(N) {}

  SmallVector(std::initializer_list<T> init)
    requires std::is_default_constructible_v<Alloc>
      : SmallVector() {
    append_copy(init.begin(), init.end());
  }

  SmallVector(Alloc alloc, std::initializer_list<T> init)
      : SmallVector(std::move(alloc)) {
    append_copy(init.begin(), init.end());
  }

  SmallVector(const SmallVector &other) : SmallVector(other.alloc_) {
    append_copy(other.begin(), other.end());
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
    replacement.append_copy(other.begin(), other.end());
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
  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return data_ + size_; }
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
    HYLE_FATAL_ASSERT(position >= cbegin() && position <= cend());
    const size_type index = static_cast<size_type>(position - cbegin());
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

    open_gap(index);
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

  iterator erase(const_iterator position) noexcept {
    HYLE_FATAL_ASSERT(position >= cbegin() && position < cend());
    return erase(position, position + 1);
  }

  iterator erase(const_iterator first, const_iterator last) noexcept {
    HYLE_FATAL_ASSERT(first >= cbegin() && first <= last && last <= cend());
    const size_type first_index = static_cast<size_type>(first - cbegin());
    const size_type count = static_cast<size_type>(last - first);
    if (count == 0)
      return data_ + first_index;

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
    return data_ + first_index;
  }

private:
  pointer inline_ptr() noexcept { return reinterpret_cast<pointer>(inline_); }
  const_pointer inline_ptr() const noexcept {
    return reinterpret_cast<const_pointer>(inline_);
  }

  static void destroy_range(pointer first, size_type count) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_type i = count; i != 0; --i)
        first[i - 1].~T();
    }
  }

  void open_gap(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index < size_ && size_ < cap_);
    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove(data_ + index + 1, data_ + index,
                   (size_ - index) * sizeof(T));
      data_[index].~T();
    } else {
      for (size_type i = size_; i > index; --i) {
        ::new (static_cast<void *>(data_ + i)) T(std::move(data_[i - 1]));
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

    if constexpr (std::is_trivially_copyable_v<T>) {
      if (size_ != 0)
        std::memcpy(replacement, data_, size_ * sizeof(T));
    } else {
      for (size_type i = 0; i < size_; ++i)
        ::new (static_cast<void *>(replacement + i)) T(std::move(data_[i]));
      destroy_range(data_, size_);
    }

    if (!inline_storage())
      alloc_.deallocate(data_, cap_ * sizeof(T), alignof(T));
    data_ = replacement;
    cap_ = requested;
    return true;
  }

  template <class It> void append_copy(It first, It last) {
    const auto distance = std::distance(first, last);
    HYLE_FATAL_ASSERT(distance >= 0);
    reserve(static_cast<size_type>(distance));
    for (; first != last; ++first) {
      ::new (static_cast<void *>(data_ + size_)) T(*first);
      ++size_;
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
    if constexpr (std::is_trivially_copyable_v<T>) {
      if (other.size_ != 0)
        std::memcpy(data_, other.data_, other.size_ * sizeof(T));
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
  pointer data_;
  size_type size_ = 0;
  size_type cap_;
  alignas(T) std::byte inline_[N * sizeof(T)];
};

} // namespace hyle::container
