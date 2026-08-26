#pragma once

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/container/hash.h"  // HYLE_CONTAINER_API
#include "hyle/core/compiler.h"
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
  explicit PodVector(size_type count)
    requires std::is_default_constructible_v<T>
      : PodVector() {
    resize(count);
  }
  PodVector(size_type count, const T& value) : PodVector() {
    resize(count, value);
  }
  PodVector(mem::Allocator alloc, size_type count)
    requires std::is_default_constructible_v<T>
      : PodVector(alloc) {
    resize(count);
  }
  PodVector(mem::Allocator alloc, size_type count, const T& value)
      : PodVector(alloc) {
    resize(count, value);
  }
  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires (std::constructible_from<T, std::iter_reference_t<It>> ||
              (std::contiguous_iterator<It> &&
               std::same_as<std::remove_cv_t<std::iter_value_t<It>>, T>))
  PodVector(It first, Sentinel last) : PodVector() {
    construct_with_cleanup([&] { append_range(first, last); });
  }
  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires (std::constructible_from<T, std::iter_reference_t<It>> ||
              (std::contiguous_iterator<It> &&
               std::same_as<std::remove_cv_t<std::iter_value_t<It>>, T>))
  PodVector(mem::Allocator alloc, It first, Sentinel last)
      : PodVector(alloc) {
    construct_with_cleanup([&] { append_range(first, last); });
  }
  PodVector(std::initializer_list<T> init) : PodVector() {
    append_range(init.begin(), init.end());
  }
  PodVector(mem::Allocator alloc, std::initializer_list<T> init)
      : alloc_(alloc) {
    append_range(init.begin(), init.end());
  }
  PodVector(const PodVector& other) : alloc_(other.alloc_) {
    append_range(other.begin(), other.end());
  }
  PodVector& operator=(const PodVector& other) {
    if (this == &other) return *this;
    PodVector replacement(alloc_);
    replacement.append_range(other.begin(), other.end());
    *this = std::move(replacement);
    return *this;
  }
  ~PodVector() { detail::carray_free_internal(data_, sizeof(T)); }

  PodVector(PodVector&& o) noexcept
      : data_(o.data_), alloc_(o.alloc_), size_(o.size_),
        capacity_(o.capacity_) {
    o.data_ = nullptr;
    o.size_ = 0;
    o.capacity_ = 0;
  }
  PodVector& operator=(PodVector&& o) noexcept {
    if (this != &o) {
      detail::carray_free_internal(data_, sizeof(T));
      data_ = o.data_;
      alloc_ = o.alloc_;
      size_ = o.size_;
      capacity_ = o.capacity_;
      o.data_ = nullptr;
      o.size_ = 0;
      o.capacity_ = 0;
    }
    return *this;
  }

  mem::Allocator allocator() const noexcept { return alloc_; }

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size() == 0; }
  pointer data() noexcept {
    sync_header_size();
    return data_;
  }
  const_pointer data() const noexcept {
    sync_header_size();
    return data_;
  }

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
      alignas(T) std::byte saved[sizeof(T)];
      copy_bytes(saved, std::addressof(v), sizeof(T));
      if (s > UINT64_MAX / 2) return false;
      if (!grow_to(s == 0 ? 4 : s * 2)) return false;
      copy_bytes(data_ + s, saved, sizeof(T));
    } else {
      copy_one(data_ + s, v);
    }
    set_size(s + 1);
    return true;
  }

  iterator insert(const_iterator position, const T& value) {
    return insert(position, 1, value);
  }

  iterator insert(const_iterator position, size_type count, const T& value) {
    const size_type index = position_index(position, true);
    const size_type current = size();
    if (count == 0) return data_ == nullptr ? nullptr : data_ + index;
    HYLE_FATAL_ASSERT(count <= UINT64_MAX - current);
    alignas(T) std::byte saved[sizeof(T)];
    copy_bytes(saved, std::addressof(value), sizeof(T));
    const size_type requested = current + count;
    if (requested > capacity()) reserve(next_capacity(requested));
    std::memmove(static_cast<void*>(data_ + index + count),
                 static_cast<const void*>(data_ + index),
                 static_cast<std::size_t>(current - index) * sizeof(T));
    fill_bytes(data_ + index, count, saved);
    set_size(requested);
    return data_ + index;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires (std::constructible_from<T, std::iter_reference_t<It>> ||
              (std::contiguous_iterator<It> &&
               std::same_as<std::remove_cv_t<std::iter_value_t<It>>, T>))
  iterator insert(const_iterator position, It first, Sentinel last) {
    const size_type index = position_index(position, true);
    // Materialize before reserve/memmove so self-ranges and single-pass input
    // iterators have the same predictable shape as std::vector.
    PodVector inserted(alloc_);
    inserted.append_range(first, last);
    return insert_prepared(index, inserted);
  }

  iterator insert(const_iterator position, std::initializer_list<T> init) {
    return insert(position, init.begin(), init.end());
  }

  void assign(size_type count, const T& value) {
    PodVector replacement(alloc_, count, value);
    *this = std::move(replacement);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires (std::constructible_from<T, std::iter_reference_t<It>> ||
              (std::contiguous_iterator<It> &&
               std::same_as<std::remove_cv_t<std::iter_value_t<It>>, T>))
  void assign(It first, Sentinel last) {
    PodVector replacement(alloc_);
    replacement.append_range(first, last);
    *this = std::move(replacement);
  }

  void assign(std::initializer_list<T> init) {
    assign(init.begin(), init.end());
  }

  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(size() > 0);
    set_size(size() - 1);
  }

  bool try_resize(size_type count)
    requires std::is_default_constructible_v<T>
  {
    if (count <= size()) {
      set_size(count);
      return true;
    }
    if (count > capacity() && !grow_to(count)) return false;
    const T value{};
    for (size_type i = size(); i < count; ++i) copy_one(data_ + i, value);
    set_size(count);
    return true;
  }

  bool try_resize(size_type count, const T& value) noexcept {
    if (count <= size()) {
      set_size(count);
      return true;
    }
    alignas(T) std::byte saved[sizeof(T)];
    copy_bytes(saved, std::addressof(value), sizeof(T));
    if (count > capacity()) {
      if (!grow_to(count)) return false;
      fill_bytes(data_ + size(), count - size(), saved);
    } else {
      fill_bytes(data_ + size(), count - size(), saved);
    }
    set_size(count);
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
    set_size(current - count);
    return data_ + index;
  }

  void clear() noexcept {
    set_size(0);
  }

  bool try_shrink_to_fit() noexcept {
    if (data_ == nullptr || capacity() == size()) return true;
    if (size() == 0) {
      detail::carray_free_internal(data_, sizeof(T));
      data_ = nullptr;
      capacity_ = 0;
      return true;
    }
    sync_header_size();
    void* replacement = detail::carray_set_capacity_internal(
        data_, size(), sizeof(T), alignof(T), alloc_);
    if (replacement == nullptr) return false;
    data_ = static_cast<T*>(replacement);
    capacity_ = size_;
    return true;
  }

  void shrink_to_fit() { HYLE_FATAL_ASSERT(try_shrink_to_fit()); }

  void swap(PodVector& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(alloc_, other.alloc_);
    swap(size_, other.size_);
    swap(capacity_, other.capacity_);
  }

  friend bool operator==(const PodVector& left, const PodVector& right)
    requires std::equality_comparable<T>
  {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
  }

  friend auto operator<=>(const PodVector& left, const PodVector& right)
    requires std::three_way_comparable<T>
  {
    return std::lexicographical_compare_three_way(
        left.begin(), left.end(), right.begin(), right.end());
  }

 private:
  static void copy_bytes(void* destination, const void* source,
                         std::size_t size) noexcept {
    std::memcpy(destination, source, size);
  }

  static void copy_one(pointer destination, const T& source) noexcept {
    copy_bytes(destination, std::addressof(source), sizeof(T));
  }

  static void fill_bytes(pointer destination, size_type count,
                         const void* value) noexcept {
    if (count == 0) return;
    copy_bytes(destination, value, sizeof(T));
    size_type completed = 1;
    while (completed < count) {
      const size_type batch = std::min(completed, count - completed);
      copy_bytes(destination + completed, destination,
                 static_cast<std::size_t>(batch) * sizeof(T));
      completed += batch;
    }
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

  template <class Fn>
  void construct_with_cleanup(Fn&& fn) {
    try {
      std::forward<Fn>(fn)();
    } catch (...) {
      detail::carray_free_internal(data_, sizeof(T));
      data_ = nullptr;
      throw;
    }
  }

  iterator insert_prepared(size_type index, const PodVector& inserted) {
    const size_type count = inserted.size();
    const size_type current = size();
    if (count == 0) return data_ == nullptr ? nullptr : data_ + index;
    HYLE_FATAL_ASSERT(count <= UINT64_MAX - current);
    const size_type requested = current + count;
    if (requested > capacity()) reserve(next_capacity(requested));
    std::memmove(static_cast<void*>(data_ + index + count),
                 static_cast<const void*>(data_ + index),
                 static_cast<std::size_t>(current - index) * sizeof(T));
    copy_bytes(data_ + index, inserted.data(),
               static_cast<std::size_t>(count) * sizeof(T));
    set_size(requested);
    return data_ + index;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires (std::constructible_from<T, std::iter_reference_t<It>> ||
              (std::contiguous_iterator<It> &&
               std::same_as<std::remove_cv_t<std::iter_value_t<It>>, T>))
  void append_range(It first, Sentinel last) {
    if constexpr (std::contiguous_iterator<It> &&
                  std::same_as<std::remove_cv_t<std::iter_value_t<It>>, T>) {
      size_type additional = 0;
      for (It probe = first; probe != last; ++probe) {
        HYLE_FATAL_ASSERT(additional != UINT64_MAX - size());
        ++additional;
      }
      reserve(size() + additional);
      if (additional != 0) {
        copy_bytes(data_ + size(), std::to_address(first),
                   static_cast<std::size_t>(additional) * sizeof(T));
        set_size(size_ + additional);
      }
    } else {
      if constexpr (std::forward_iterator<It>) {
        size_type additional = 0;
        for (It probe = first; probe != last; ++probe) {
          HYLE_FATAL_ASSERT(additional != UINT64_MAX - size());
          ++additional;
        }
        reserve(size() + additional);
      }
      for (; first != last; ++first) {
        T value(*first);
        push_back(value);
      }
    }
  }

  size_type next_capacity(size_type required) const noexcept {
    const size_type current = capacity();
    if (current == 0) return required < 4 ? 4 : required;
    if (current > UINT64_MAX / 2) return required;
    const size_type doubled = current * 2;
    return doubled < required ? required : doubled;
  }

  bool grow_to(size_type cap) noexcept {
    sync_header_size();
    void *replacement = detail::carray_set_capacity_internal(
        data_, cap, sizeof(T), alignof(T), alloc_);
    if (replacement == nullptr)
      return false;
    data_ = static_cast<T *>(replacement);
    capacity_ = carray_capacity(data_);
    return true;
  }

  void set_size(size_type size) noexcept {
    size_ = size;
  }

  void sync_header_size() const noexcept {
    if (data_ != nullptr)
      carray_header(data_)->size = size_;
  }

  T* data_ = nullptr;
  mem::Allocator alloc_;
  size_type size_ = 0;
  size_type capacity_ = 0;
};

template <class T>
void swap(PodVector<T>& left, PodVector<T>& right) noexcept {
  left.swap(right);
}

}  // namespace hyle::container
