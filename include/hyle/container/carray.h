#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/container/hash.h"  // HYLE_CONTAINER_API
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Contiguous POD dynamic array with a header-before-data layout (from clay
// carray): the
// handle points AT the elements, and a CArrayHeader sits immediately before
// them. So `arr[i]` is a single load with zero indirection and the element
// pointer can be handed straight to a C API. A CArrayHeader-aware allocation
// also records its own allocator + alignment for self-contained growth/free.
//
// Heavy logic is type-erased (carray_set_capacity_internal(void*, item_size,
// ...)) and compiled ONCE into the DLL (conventions: type-erased core + typed
// shell -> no per-T code bloat, one allocator path). The typed PodVector<T>
// below is a thin inline shell.
//
// T must be trivially copyable (growth relocates via memcpy, like clay). Non-
// trivial T (move-construct on grow) is deferred.

struct CArrayHeader {
  std::uint64_t capacity;       // element slots allocated
  std::uint64_t size;           // elements in use
  std::uint32_t header_offset;  // bytes from block base to data (>= sizeof hdr)
  std::uint32_t align;          // block alignment (for precise deallocate)
  mem::Allocator alloc;         // value handle (fn+state)
};

// data - sizeof(header) lands on the header (it sits just before the elements,
// inside any leading alignment padding).
HYLE_FORCE_INLINE CArrayHeader* carray_header(void* data) noexcept {
  return reinterpret_cast<CArrayHeader*>(static_cast<std::byte*>(data) -
                                         sizeof(CArrayHeader));
}
HYLE_FORCE_INLINE const CArrayHeader* carray_header(const void* data) noexcept {
  return reinterpret_cast<const CArrayHeader*>(
      static_cast<const std::byte*>(data) - sizeof(CArrayHeader));
}
HYLE_FORCE_INLINE std::uint64_t carray_size(const void* d) noexcept {
  return d ? carray_header(d)->size : 0;
}
HYLE_FORCE_INLINE std::uint64_t carray_capacity(const void* d) noexcept {
  return d ? carray_header(d)->capacity : 0;
}

namespace detail {
// Ensure `data` has room for >= min_capacity elements of item_size/align. data
// may be null (fresh allocation). Returns the (possibly relocated) data ptr;
// header/size preserved. Type-erased: ONE compiled copy for all T.
HYLE_CONTAINER_API void* carray_set_capacity_internal(void* data,
                                                      std::uint64_t min_capacity,
                                                      std::size_t item_size,
                                                      std::size_t align,
                                                      mem::Allocator alloc);
// Free a carray's backing (no-op on null). Reads allocator from the header.
HYLE_CONTAINER_API void carray_free_internal(void* data,
                                             std::size_t item_size) noexcept;
}  // namespace detail

// ---- typed RAII shell -------------------------------------------------------
template <class T>
class PodVector {
  static_assert(std::is_trivially_copyable_v<T>,
                "PodVector<T>: T must be trivially copyable (memcpy growth)");

 public:
  using value_type = T;
  using size_type = std::uint64_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  PodVector() noexcept : alloc_(mem::system_allocator()) {}
  explicit PodVector(mem::Allocator alloc) noexcept : alloc_(alloc) {}
  PodVector(std::initializer_list<T> init) : PodVector() {
    append_copy(init.begin(), init.end());
  }
  PodVector(mem::Allocator alloc, std::initializer_list<T> init)
      : alloc_(alloc) {
    append_copy(init.begin(), init.end());
  }
  PodVector(const PodVector& other) : alloc_(other.alloc_) {
    append_copy(other.begin(), other.end());
  }
  PodVector& operator=(const PodVector& other) {
    if (this == &other) return *this;
    PodVector replacement(alloc_);
    replacement.append_copy(other.begin(), other.end());
    *this = std::move(replacement);
    return *this;
  }
  ~PodVector() { detail::carray_free_internal(data_, sizeof(T)); }

  PodVector(PodVector&& o) noexcept : data_(o.data_), alloc_(o.alloc_) {
    o.data_ = nullptr;
  }
  PodVector& operator=(PodVector&& o) noexcept {
    if (this != &o) {
      detail::carray_free_internal(data_, sizeof(T));
      data_ = o.data_;
      alloc_ = o.alloc_;
      o.data_ = nullptr;
    }
    return *this;
  }

  mem::Allocator allocator() const noexcept { return alloc_; }

  size_type size() const noexcept { return carray_size(data_); }
  size_type capacity() const noexcept { return carray_capacity(data_); }
  bool empty() const noexcept { return size() == 0; }
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
    return data_[size() - 1];
  }
  const_reference back() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data_[size() - 1];
  }

  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ ? data_ + size() : nullptr; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept {
    return data_ ? data_ + size() : nullptr;
  }
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
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

  void reserve(size_type n) {
    HYLE_FATAL_ASSERT(try_reserve(n));
  }
  bool try_reserve(size_type n) noexcept {
    return n <= capacity() || grow_to(n);
  }

  void push_back(const T& v) {
    HYLE_FATAL_ASSERT(try_push_back(v));
  }
  bool try_push_back(const T& v) noexcept {
    size_type s = size();
    if (s == capacity()) {
      // `v` may alias an element in this array. reallocate is allowed to move
      // the backing block, so preserve it before growth (std::vector supports
      // the same push_back(v[0]) shape).
      const T copy = v;
      if (s > UINT64_MAX / 2) return false;
      if (!grow_to(s == 0 ? 4 : s * 2)) return false;
      copy_one(data_ + s, copy);
    } else {
      copy_one(data_ + s, v);
    }
    carray_header(data_)->size = s + 1;
    return true;
  }

  iterator insert(const_iterator position, const T& value) {
    const size_type index = position_index(position, true);
    const T copy = value;
    if (size() == capacity()) {
      const size_type current = size();
      HYLE_FATAL_ASSERT(current <= UINT64_MAX / 2);
      reserve(current == 0 ? 4 : current * 2);
    }
    const size_type current = size();
    std::memmove(static_cast<void*>(data_ + index + 1),
                 static_cast<const void*>(data_ + index),
                 static_cast<std::size_t>(current - index) * sizeof(T));
    copy_one(data_ + index, copy);
    carray_header(data_)->size = current + 1;
    return data_ + index;
  }

  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(size() > 0);
    carray_header(data_)->size = size() - 1;
  }

  bool try_resize(size_type count)
    requires std::is_default_constructible_v<T>
  {
    if (count > capacity() && !grow_to(count)) return false;
    const T value{};
    for (size_type i = size(); i < count; ++i) copy_one(data_ + i, value);
    if (data_ != nullptr) carray_header(data_)->size = count;
    return true;
  }

  bool try_resize(size_type count, const T& value) noexcept {
    if (count > capacity()) {
      const T copy = value;
      if (!grow_to(count)) return false;
      for (size_type i = size(); i < count; ++i) copy_one(data_ + i, copy);
    } else {
      for (size_type i = size(); i < count; ++i) copy_one(data_ + i, value);
    }
    if (data_ != nullptr) carray_header(data_)->size = count;
    return true;
  }

  void resize(size_type count)
    requires std::is_default_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(try_resize(count));
  }
  void resize(size_type count, const T& value) {
    HYLE_FATAL_ASSERT(try_resize(count, value));
  }

  iterator erase(const_iterator position) noexcept {
    HYLE_FATAL_ASSERT(position >= cbegin() && position < cend());
    return erase(position, position + 1);
  }

  iterator erase(const_iterator first, const_iterator last) noexcept {
    const size_type index = position_index(first, true);
    const size_type last_index = position_index(last, true);
    HYLE_FATAL_ASSERT(index <= last_index);
    const size_type count = last_index - index;
    if (count == 0) return data_ == nullptr ? nullptr : data_ + index;
    const size_type current = size();
    std::memmove(static_cast<void*>(data_ + index),
                 static_cast<const void*>(data_ + index + count),
                 static_cast<std::size_t>(current - index - count) * sizeof(T));
    carray_header(data_)->size = current - count;
    return data_ + index;
  }

  void clear() noexcept {
    if (data_) carray_header(data_)->size = 0;
  }

  void swap(PodVector& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(alloc_, other.alloc_);
  }

 private:
  static void copy_one(pointer destination, const T& source) noexcept {
    std::memcpy(static_cast<void*>(destination),
                static_cast<const void*>(std::addressof(source)), sizeof(T));
  }

  size_type position_index(const_iterator position, bool allow_end) const
      noexcept {
    if (data_ == nullptr) {
      HYLE_FATAL_ASSERT(position == nullptr && allow_end);
      return 0;
    }
    HYLE_FATAL_ASSERT(position >= cbegin());
    HYLE_FATAL_ASSERT(allow_end ? position <= cend() : position < cend());
    return static_cast<size_type>(position - cbegin());
  }

  void append_copy(const_pointer first, const_pointer last) {
    if (first == last) return;
    const auto distance = std::distance(first, last);
    HYLE_FATAL_ASSERT(distance >= 0);
    reserve(static_cast<size_type>(distance));
    if (distance != 0) {
      std::memcpy(static_cast<void*>(data_),
                  static_cast<const void*>(std::addressof(*first)),
                  static_cast<std::size_t>(distance) * sizeof(T));
      carray_header(data_)->size = static_cast<size_type>(distance);
    }
  }

  bool grow_to(size_type cap) noexcept {
    void* replacement = detail::carray_set_capacity_internal(
        data_, cap, sizeof(T), alignof(T), alloc_);
    if (replacement == nullptr) return false;
    data_ = static_cast<T*>(replacement);
    return true;
  }

  T* data_ = nullptr;
  mem::Allocator alloc_;
};

template <class T>
void swap(PodVector<T>& left, PodVector<T>& right) noexcept {
  left.swap(right);
}

}  // namespace hyle::container
