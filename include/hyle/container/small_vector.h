#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Vector with inline storage for the first N elements: small collections never
// touch the heap; growth past N spills to an allocator-backed buffer. Design
// follows folly small_vector (inline buffer + spill), not the code.
//
// T must be trivially copyable (relocation on spill/grow is memcpy). Non-trivial
// T (move-relocate) is deferred -- the scheduler's uses (handles/pointers/PODs)
// are all trivial.
template <class T, std::size_t N>
class SmallVector {
  static_assert(std::is_trivially_copyable_v<T>,
                "SmallVector<T,N>: T must be trivially copyable");
  static_assert(N >= 1, "N must be >= 1");

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  explicit SmallVector(mem::Allocator alloc) noexcept
      : alloc_(alloc), data_(inline_ptr()), cap_(N) {}
  SmallVector(mem::Allocator alloc, std::initializer_list<T> init)
      : SmallVector(alloc) {
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
    if (size_ == cap_) grow_to(cap_ * 2);
    data_[size_++] = v;
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

  mem::Allocator alloc_;
  T* data_;
  std::size_t size_ = 0;
  std::size_t cap_;
  alignas(T) std::byte inline_[N * sizeof(T)];
};

}  // namespace hyle::container
