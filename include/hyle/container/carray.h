#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include "hyle/container/hash.h"  // HYLE_CONTAINER_API
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Intrusive dynamic array with a header-before-data layout (clay carray): the
// handle points AT the elements, and a CArrayHeader sits immediately before
// them. So `arr[i]` is a single load with zero indirection and the element
// pointer can be handed straight to a C API. A CArrayHeader-aware allocation
// also records its own allocator + alignment for self-contained growth/free.
//
// Heavy logic is type-erased (carray_set_capacity_internal(void*, item_size,
// ...)) and compiled ONCE into the DLL (conventions: type-erased core + typed
// shell -> no per-T code bloat, one allocator path). The typed CArray<T> below
// is a thin inline shell.
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
class CArray {
  static_assert(std::is_trivially_copyable_v<T>,
                "CArray<T>: T must be trivially copyable (memcpy growth)");

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  explicit CArray(mem::Allocator alloc) noexcept : alloc_(alloc) {}
  CArray(mem::Allocator alloc, std::initializer_list<T> init) : alloc_(alloc) {
    reserve(init.size());
    for (const T& v : init) push_back(v);
  }
  ~CArray() { detail::carray_free_internal(data_, sizeof(T)); }

  CArray(CArray&& o) noexcept : data_(o.data_), alloc_(o.alloc_) {
    o.data_ = nullptr;
  }
  CArray& operator=(CArray&& o) noexcept {
    if (this != &o) {
      detail::carray_free_internal(data_, sizeof(T));
      data_ = o.data_;
      alloc_ = o.alloc_;
      o.data_ = nullptr;
    }
    return *this;
  }
  HYLE_DISABLE_COPY(CArray);

  std::uint64_t size() const noexcept { return carray_size(data_); }
  std::uint64_t capacity() const noexcept { return carray_capacity(data_); }
  bool empty() const noexcept { return size() == 0; }
  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  T& operator[](std::uint64_t i) noexcept { return data_[i]; }
  const T& operator[](std::uint64_t i) const noexcept { return data_[i]; }
  T& back() noexcept { return data_[size() - 1]; }

  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ ? data_ + size() : nullptr; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept { return data_ ? data_ + size() : nullptr; }

  void reserve(std::uint64_t n) {
    if (n > capacity()) grow_to(n);
  }

  void push_back(const T& v) {
    std::uint64_t s = size();
    if (s == capacity()) grow_to(s == 0 ? 4 : s * 2);
    data_[s] = v;
    carray_header(data_)->size = s + 1;
  }

  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(size() > 0);
    carray_header(data_)->size = size() - 1;
  }

  void clear() noexcept {
    if (data_) carray_header(data_)->size = 0;
  }

 private:
  void grow_to(std::uint64_t cap) {
    data_ = static_cast<T*>(detail::carray_set_capacity_internal(
        data_, cap, sizeof(T), alignof(T), alloc_));
    HYLE_FATAL_ASSERT(data_ != nullptr);
  }

  T* data_ = nullptr;
  mem::Allocator alloc_;
};

}  // namespace hyle::container
