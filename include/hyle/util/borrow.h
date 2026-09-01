#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>

#include "hyle/core/compiler.h"

// Required, rebindable raw-pointer borrow. NotNull never owns or extends the
// pointee lifetime. Use T& when rebinding/pointer interop is unnecessary, T*
// when null is meaningful, and Result<T&> when acquisition can fail.

namespace hyle::util {

template <class Pointer>
class NotNull;

template <class T>
  requires(std::is_object_v<T> || std::is_function_v<T> || std::is_void_v<T>)
class NotNull<T*> final {
 public:
  using element_type = T;
  using pointer = T*;

  NotNull() = delete;
  NotNull(std::nullptr_t) = delete;

  template <class U>
    requires std::convertible_to<U*, T*>
  constexpr NotNull(U* pointer HYLE_LIFETIME_BOUND) noexcept
      : pointer_(pointer) {  // NOLINT(*-explicit)
    HYLE_FATAL_ASSERT(pointer_ != nullptr);
  }

  template <class U>
    requires std::convertible_to<U*, T*>
  constexpr NotNull(const NotNull<U*>& other) noexcept  // NOLINT(*-explicit)
      : pointer_(other.get()) {}

  template <class U>
    requires std::convertible_to<U*, T*>
  constexpr NotNull& operator=(const NotNull<U*>& other) noexcept {
    pointer_ = other.get();
    return *this;
  }

  NotNull& operator=(std::nullptr_t) = delete;

  [[nodiscard]] constexpr T* get() const noexcept { return pointer_; }
  template <class U = T>
    requires(!std::is_void_v<U>)
  [[nodiscard]] constexpr U& operator*() const noexcept {
    return *pointer_;
  }

  [[nodiscard]] constexpr T* operator->() const noexcept
    requires std::is_object_v<T>
  {
    return pointer_;
  }

  constexpr operator T*() const noexcept { return pointer_; }  // NOLINT

 private:
  T* pointer_;
};

template <class T, class U>
  requires requires(T* left, U* right) { left == right; }
[[nodiscard]] constexpr bool operator==(NotNull<T*> left,
                                        NotNull<U*> right) noexcept {
  return left.get() == right.get();
}

template <class T, class U>
  requires requires(T* left, U* right) { left == right; }
[[nodiscard]] constexpr bool operator==(NotNull<T*> left, U* right) noexcept {
  return left.get() == right;
}

}  // namespace hyle::util

namespace std {

template <class T>
struct hash<hyle::util::NotNull<T*>> {
  size_t operator()(hyle::util::NotNull<T*> pointer) const noexcept {
    return hash<T*>{}(pointer.get());
  }
};

}  // namespace std
