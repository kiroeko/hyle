#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"

// A non-owning callable view for synchronous, call-only APIs. FunctionRef is
// deliberately not nullable and never allocates or copies the callable. The
// callable must outlive every invocation; in particular, a callee accepting a
// FunctionRef by value must not store or return it.

namespace hyle::util {

namespace detail {

// Keep large/non-trivial by-value arguments on the same object while crossing
// the erased invoker. Small trivial values stay in their natural ABI registers.
// This is the same cost rule used by Abseil's mature FunctionRef.
template <class T>
inline constexpr bool kFunctionRefPassByValue =
    !std::is_lvalue_reference_v<T> &&
    std::is_trivially_copy_constructible_v<T> &&
    std::is_trivially_copy_assignable_v<std::remove_cv_t<T>> &&
    std::is_trivially_destructible_v<T> && sizeof(T) <= 2 * sizeof(void*);

template <class T>
using FunctionRefArg = std::conditional_t<kFunctionRefPassByValue<T>, T, T&&>;

template <class F>
HYLE_FORCE_INLINE void assert_function_ref_non_null(const F&) noexcept {}

template <class R, class... Args>
HYLE_FORCE_INLINE void assert_function_ref_non_null(
    const std::function<R(Args...)>& function) noexcept {
  HYLE_FATAL_ASSERT(static_cast<bool>(function));
}

#if defined(__cpp_lib_move_only_function) && \
    __cpp_lib_move_only_function >= 202110L
template <class Signature>
HYLE_FORCE_INLINE void assert_function_ref_non_null(
    const std::move_only_function<Signature>& function) noexcept {
  HYLE_FATAL_ASSERT(static_cast<bool>(function));
}
#endif

template <class Member, class Owner>
HYLE_FORCE_INLINE void assert_function_ref_non_null(
    Member Owner::* member) noexcept {
  HYLE_FATAL_ASSERT(member != nullptr);
}

}  // namespace detail

template <class Signature>
class FunctionRef;

template <class R, class... Args>
class FunctionRef<R(Args...)> final {
 private:
  union Storage {
    constexpr explicit Storage(const void* value) noexcept : object(value) {}
    constexpr explicit Storage(void (*value)()) noexcept : function(value) {}

    const void* object;
    void (*function)();
  };

  using Invoker = R (*)(Storage, detail::FunctionRefArg<Args>...);

  template <class Callable>
  static R invoke_object(Storage storage,
                         detail::FunctionRefArg<Args>... args) {
    auto* callable = static_cast<Callable*>(const_cast<void*>(storage.object));
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(*callable,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(*callable,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

  template <class Function>
  static R invoke_function(Storage storage,
                           detail::FunctionRefArg<Args>... args) {
    auto* function = reinterpret_cast<Function*>(storage.function);
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(function,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(function,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

 public:
  FunctionRef() = delete;
  FunctionRef(std::nullptr_t) = delete;

  // Function pointers are stored directly, avoiding an extra indirection
  // through the caller's pointer variable.
  template <class Function>
    requires(std::is_function_v<Function> &&
             std::is_invocable_r_v<R, Function*, Args...>)
  FunctionRef(Function* function HYLE_LIFETIME_BOUND) noexcept
      : storage_(reinterpret_cast<void (*)()>(function)),
        invoker_(&invoke_function<Function>) {
    HYLE_FATAL_ASSERT(function != nullptr);
  }

  template <class F>
    requires(!std::same_as<std::remove_cvref_t<F>, FunctionRef> &&
             !(std::is_pointer_v<std::remove_reference_t<F>> &&
               std::is_function_v<
                   std::remove_pointer_t<std::remove_reference_t<F>>>) &&
             std::is_object_v<std::remove_reference_t<F>> &&
             std::is_invocable_r_v<R, F&, Args...>)
  FunctionRef(F&& callable HYLE_LIFETIME_BOUND) noexcept  // NOLINT(*-explicit)
      : storage_(static_cast<const void*>(std::addressof(callable))),
        invoker_(&invoke_object<std::remove_reference_t<F>>) {
    detail::assert_function_ref_non_null(callable);
  }

  R operator()(Args... args) const {
    return invoker_(storage_, std::forward<Args>(args)...);
  }

 private:
  Storage storage_;
  Invoker invoker_;
};

template <class R, class... Args>
class FunctionRef<R(Args...) noexcept> final {
 private:
  union Storage {
    constexpr explicit Storage(const void* value) noexcept : object(value) {}
    constexpr explicit Storage(void (*value)()) noexcept : function(value) {}

    const void* object;
    void (*function)();
  };

  using Invoker = R (*)(Storage, detail::FunctionRefArg<Args>...) noexcept;

  template <class Callable>
  static R invoke_object(Storage storage,
                         detail::FunctionRefArg<Args>... args) noexcept {
    auto* callable = static_cast<Callable*>(const_cast<void*>(storage.object));
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(*callable,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(*callable,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

  template <class Function>
  static R invoke_function(Storage storage,
                           detail::FunctionRefArg<Args>... args) noexcept {
    auto* function = reinterpret_cast<Function*>(storage.function);
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(function,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(function,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

 public:
  FunctionRef() = delete;
  FunctionRef(std::nullptr_t) = delete;

  template <class Function>
    requires(std::is_function_v<Function> &&
             std::is_nothrow_invocable_r_v<R, Function*, Args...>)
  FunctionRef(Function* function HYLE_LIFETIME_BOUND) noexcept
      : storage_(reinterpret_cast<void (*)()>(function)),
        invoker_(&invoke_function<Function>) {
    HYLE_FATAL_ASSERT(function != nullptr);
  }

  template <class F>
    requires(!std::same_as<std::remove_cvref_t<F>, FunctionRef> &&
             !(std::is_pointer_v<std::remove_reference_t<F>> &&
               std::is_function_v<
                   std::remove_pointer_t<std::remove_reference_t<F>>>) &&
             std::is_object_v<std::remove_reference_t<F>> &&
             std::is_nothrow_invocable_r_v<R, F&, Args...>)
  FunctionRef(F&& callable HYLE_LIFETIME_BOUND) noexcept  // NOLINT(*-explicit)
      : storage_(static_cast<const void*>(std::addressof(callable))),
        invoker_(&invoke_object<std::remove_reference_t<F>>) {
    detail::assert_function_ref_non_null(callable);
  }

  R operator()(Args... args) const noexcept {
    return invoker_(storage_, std::forward<Args>(args)...);
  }

 private:
  Storage storage_;
  Invoker invoker_;
};

template <class R, class... Args>
FunctionRef(R (*)(Args...)) -> FunctionRef<R(Args...)>;

template <class R, class... Args>
FunctionRef(R (*)(Args...) noexcept) -> FunctionRef<R(Args...) noexcept>;

}  // namespace hyle::util
