#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Vector with inline storage for the first N elements: small collections never
// touch the heap; growth past N spills to an allocator-backed buffer. Design
// follows folly small_vector (inline buffer + spill), not the code.
//
// T must be trivially copyable (relocation on spill/grow is memcpy). Non-trivial
// T (move-relocate) is deferred -- the scheduler's uses (handles/pointers/PODs)
// are all trivial.
//
// Alloc defaults to the compile-time SystemAllocator (stateless, inlines to mi_malloc,
// zero seam tax -- and [[no_unique_address]] gives it ZERO bytes in the object). Pass a
// runtime mem::Allocator (SmallVector<T,N,mem::Allocator>) to opt into injection (arena,
// mock, cross-DLL). Either way the body is identical (alloc_.allocate(...)).
template <class T, std::size_t N, class Alloc = mem::SystemAllocator>
class SmallVector {
  static_assert(std::is_trivially_copyable_v<T>,
                "SmallVector<T,N>: T must be trivially copyable");
  static_assert(N >= 1, "N must be >= 1");

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  // Default: only when Alloc is default-constructible (SystemAllocator is) -- the common
  // zero-overhead path, no allocator argument needed.
  SmallVector() noexcept requires std::is_default_constructible_v<Alloc>
      : data_(inline_ptr()), cap_(N) {}
  // Explicit allocator (runtime mem::Allocator injection, or any stateful Alloc).
  explicit SmallVector(Alloc alloc) noexcept
      : alloc_(alloc), data_(inline_ptr()), cap_(N) {}
  SmallVector(Alloc alloc, std::initializer_list<T> init)
      : SmallVector(alloc) {
    reserve(init.size());
    for (const T& v : init) push_back(v);
  }
  explicit SmallVector(std::initializer_list<T> init)
      requires std::is_default_constructible_v<Alloc>
      : SmallVector() {
    reserve(init.size());
    for (const T& v : init) push_back(v);
  }
  ~SmallVector() { free_heap(); }

  SmallVector(SmallVector&& o) noexcept
      : alloc_(o.alloc_), size_(o.size_), cap_(o.cap_) {
    if (o.inline_storage()) {
      data_ = inline_ptr();
      std::memcpy(data_, o.data_, size_ * sizeof(T));  // copy inline elements
    } else {
      data_ = o.data_;       // steal heap buffer
      o.data_ = o.inline_ptr();
    }
    o.size_ = 0;
    o.cap_ = N;
  }
  SmallVector& operator=(SmallVector&& o) noexcept {
    if (this == &o) return *this;
    free_heap();
    alloc_ = o.alloc_;
    size_ = o.size_;
    cap_ = o.cap_;
    if (o.inline_storage()) {
      data_ = inline_ptr();
      std::memcpy(data_, o.data_, size_ * sizeof(T));
    } else {
      data_ = o.data_;
      o.data_ = o.inline_ptr();
    }
    o.size_ = 0;
    o.cap_ = N;
    return *this;
  }
  HYLE_DISABLE_COPY(SmallVector);

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return cap_; }
  bool empty() const noexcept { return size_ == 0; }
  bool inline_storage() const noexcept { return data_ == inline_ptr(); }
  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }
  T& back() noexcept { return data_[size_ - 1]; }

  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept { return data_ + size_; }

  void reserve(std::size_t n) {
    if (n > cap_) grow_to(n);
  }
  void push_back(const T& v) {
    if (size_ == cap_) {
      const T copy = v;  // preserve a possibly self-referential argument
      HYLE_FATAL_ASSERT(cap_ <= static_cast<std::size_t>(-1) / 2);
      grow_to(cap_ * 2);
      data_[size_++] = copy;
    } else {
      data_[size_++] = v;
    }
  }
  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(size_ > 0);
    --size_;
  }
  void clear() noexcept { size_ = 0; }

 private:
  T* inline_ptr() noexcept { return reinterpret_cast<T*>(inline_); }
  const T* inline_ptr() const noexcept {
    return reinterpret_cast<const T*>(inline_);
  }

  void grow_to(std::size_t n) {
    HYLE_FATAL_ASSERT(n >= size_ &&
                      n <= static_cast<std::size_t>(-1) / sizeof(T));
    T* nb = static_cast<T*>(alloc_.allocate(n * sizeof(T), alignof(T)));
    HYLE_FATAL_ASSERT(nb != nullptr);
    std::memcpy(nb, data_, size_ * sizeof(T));
    free_heap();
    data_ = nb;
    cap_ = n;
  }
  void free_heap() noexcept {
    if (!inline_storage()) alloc_.deallocate(data_, cap_ * sizeof(T), alignof(T));
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  T* data_;
  std::size_t size_ = 0;
  std::size_t cap_;
  alignas(T) std::byte inline_[N * sizeof(T)];
};

}  // namespace hyle::container
