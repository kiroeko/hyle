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

// Names a callable at compile time. A FunctionRef built from this tag stores no
// runtime function/member pointer; only a separately bound object, if any, is
// kept in the ordinary two-word view.
template <auto Callable>
struct NontypeTag final {
  explicit constexpr NontypeTag() noexcept = default;
};

template <auto Callable>
inline constexpr NontypeTag<Callable> nontype{};

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

template <auto Callable>
inline constexpr bool kValidNontypeCallable = [] {
  if constexpr (std::is_pointer_v<decltype(Callable)> ||
                std::is_member_pointer_v<decltype(Callable)>) {
    return Callable != nullptr;
  }
  return false;
}();

template <class Signature>
inline constexpr bool kSupportedFunctionRefSignature = false;

template <class R, class... Args>
inline constexpr bool kSupportedFunctionRefSignature<R(Args...)> = true;

template <class R, class... Args>
inline constexpr bool kSupportedFunctionRefSignature<R(Args...) noexcept> =
    true;

template <class R, class... Args>
inline constexpr bool kSupportedFunctionRefSignature<R(Args...) const> = true;

template <class R, class... Args>
inline constexpr bool
    kSupportedFunctionRefSignature<R(Args...) const noexcept> = true;

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

  template <auto Callable>
  static R invoke_nontype(Storage,
                          detail::FunctionRefArg<Args>... args) {
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(Callable,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(Callable,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

  template <auto Callable, class Object>
  static R invoke_bound_object(Storage storage,
                               detail::FunctionRefArg<Args>... args) {
    auto* object = static_cast<Object*>(const_cast<void*>(storage.object));
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(Callable, *object,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(Callable, *object,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

  template <auto Callable, class Object>
  static R invoke_bound_pointer(Storage storage,
                                detail::FunctionRefArg<Args>... args) {
    auto* object = static_cast<Object*>(const_cast<void*>(storage.object));
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(Callable, object,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(Callable, object,
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

  template <auto Callable>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_invocable_r_v<R, decltype(Callable), Args...>)
  FunctionRef(NontypeTag<Callable>) noexcept  // NOLINT(*-explicit)
      : storage_(static_cast<const void*>(nullptr)),
        invoker_(&invoke_nontype<Callable>) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             !std::is_pointer_v<std::remove_reference_t<Object>> &&
             !std::is_rvalue_reference_v<Object&&> &&
             std::is_invocable_r_v<R, decltype(Callable), Object&, Args...>)
  FunctionRef(NontypeTag<Callable>,
              Object&& object HYLE_LIFETIME_BOUND) noexcept
      : storage_(static_cast<const void*>(std::addressof(object))),
        invoker_(&invoke_bound_object<Callable,
                                      std::remove_reference_t<Object>>) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_invocable_r_v<R, decltype(Callable), Object*, Args...>)
  FunctionRef(NontypeTag<Callable>,
              Object* object HYLE_LIFETIME_BOUND) noexcept
      : storage_(static_cast<const void*>(object)),
        invoker_(&invoke_bound_pointer<Callable, Object>) {
    HYLE_FATAL_ASSERT(object != nullptr);
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

  template <auto Callable>
  static R invoke_nontype(Storage,
                          detail::FunctionRefArg<Args>... args) noexcept {
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(Callable,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(Callable,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

  template <auto Callable, class Object>
  static R invoke_bound_object(Storage storage,
                               detail::FunctionRefArg<Args>... args) noexcept {
    auto* object = static_cast<Object*>(const_cast<void*>(storage.object));
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(Callable, *object,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(Callable, *object,
                         std::forward<detail::FunctionRefArg<Args>>(args)...);
    }
  }

  template <auto Callable, class Object>
  static R invoke_bound_pointer(Storage storage,
                                detail::FunctionRefArg<Args>... args) noexcept {
    auto* object = static_cast<Object*>(const_cast<void*>(storage.object));
    if constexpr (std::is_void_v<R>) {
      (void)std::invoke(Callable, object,
                        std::forward<detail::FunctionRefArg<Args>>(args)...);
    } else {
      return std::invoke(Callable, object,
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

  template <auto Callable>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_nothrow_invocable_r_v<R, decltype(Callable), Args...>)
  FunctionRef(NontypeTag<Callable>) noexcept  // NOLINT(*-explicit)
      : storage_(static_cast<const void*>(nullptr)),
        invoker_(&invoke_nontype<Callable>) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             !std::is_pointer_v<std::remove_reference_t<Object>> &&
             !std::is_rvalue_reference_v<Object&&> &&
             std::is_nothrow_invocable_r_v<R, decltype(Callable), Object&,
                                           Args...>)
  FunctionRef(NontypeTag<Callable>,
              Object&& object HYLE_LIFETIME_BOUND) noexcept
      : storage_(static_cast<const void*>(std::addressof(object))),
        invoker_(&invoke_bound_object<Callable,
                                      std::remove_reference_t<Object>>) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_nothrow_invocable_r_v<R, decltype(Callable), Object*,
                                           Args...>)
  FunctionRef(NontypeTag<Callable>,
              Object* object HYLE_LIFETIME_BOUND) noexcept
      : storage_(static_cast<const void*>(object)),
        invoker_(&invoke_bound_pointer<Callable, Object>) {
    HYLE_FATAL_ASSERT(object != nullptr);
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

// A const-qualified signature borrows the target as const. The wrapper has no
// additional state, so layout and per-call cost stay identical to its base.
template <class R, class... Args>
class FunctionRef<R(Args...) const> final {
  using Base = FunctionRef<R(Args...)>;

 public:
  FunctionRef() = delete;
  FunctionRef(std::nullptr_t) = delete;

  template <class Function>
    requires(std::is_function_v<Function> &&
             std::is_invocable_r_v<R, Function*, Args...>)
  FunctionRef(Function* function HYLE_LIFETIME_BOUND) noexcept
      : base_(function) {}

  template <class F>
    requires(!std::same_as<std::remove_cvref_t<F>, FunctionRef> &&
             !(std::is_pointer_v<std::remove_reference_t<F>> &&
               std::is_function_v<
                   std::remove_pointer_t<std::remove_reference_t<F>>>) &&
             std::is_object_v<F> &&
             std::is_invocable_r_v<R, const F&, Args...>)
  FunctionRef(const F& callable HYLE_LIFETIME_BOUND) noexcept  // NOLINT(*-explicit)
      : base_(callable) {}

  template <auto Callable>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_invocable_r_v<R, decltype(Callable), Args...>)
  FunctionRef(NontypeTag<Callable> tag) noexcept  // NOLINT(*-explicit)
      : base_(tag) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_invocable_r_v<R, decltype(Callable), const Object&,
                                   Args...>)
  FunctionRef(NontypeTag<Callable> tag,
              const Object& object HYLE_LIFETIME_BOUND) noexcept
      : base_(tag, object) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_invocable_r_v<R, decltype(Callable), const Object*,
                                   Args...>)
  FunctionRef(NontypeTag<Callable> tag,
              const Object* object HYLE_LIFETIME_BOUND) noexcept
      : base_(tag, object) {}

  R operator()(Args... args) const {
    return base_(std::forward<Args>(args)...);
  }

 private:
  Base base_;
};

template <class R, class... Args>
class FunctionRef<R(Args...) const noexcept> final {
  using Base = FunctionRef<R(Args...) noexcept>;

 public:
  FunctionRef() = delete;
  FunctionRef(std::nullptr_t) = delete;

  template <class Function>
    requires(std::is_function_v<Function> &&
             std::is_nothrow_invocable_r_v<R, Function*, Args...>)
  FunctionRef(Function* function HYLE_LIFETIME_BOUND) noexcept
      : base_(function) {}

  template <class F>
    requires(!std::same_as<std::remove_cvref_t<F>, FunctionRef> &&
             !(std::is_pointer_v<std::remove_reference_t<F>> &&
               std::is_function_v<
                   std::remove_pointer_t<std::remove_reference_t<F>>>) &&
             std::is_object_v<F> &&
             std::is_nothrow_invocable_r_v<R, const F&, Args...>)
  FunctionRef(const F& callable HYLE_LIFETIME_BOUND) noexcept  // NOLINT(*-explicit)
      : base_(callable) {}

  template <auto Callable>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_nothrow_invocable_r_v<R, decltype(Callable), Args...>)
  FunctionRef(NontypeTag<Callable> tag) noexcept  // NOLINT(*-explicit)
      : base_(tag) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_nothrow_invocable_r_v<R, decltype(Callable),
                                           const Object&, Args...>)
  FunctionRef(NontypeTag<Callable> tag,
              const Object& object HYLE_LIFETIME_BOUND) noexcept
      : base_(tag, object) {}

  template <auto Callable, class Object>
    requires(detail::kValidNontypeCallable<Callable> &&
             std::is_nothrow_invocable_r_v<R, decltype(Callable),
                                           const Object*, Args...>)
  FunctionRef(NontypeTag<Callable> tag,
              const Object* object HYLE_LIFETIME_BOUND) noexcept
      : base_(tag, object) {}

  R operator()(Args... args) const noexcept {
    return base_(std::forward<Args>(args)...);
  }

 private:
  Base base_;
};

template <class R, class... Args>
FunctionRef(R (*)(Args...)) -> FunctionRef<R(Args...)>;

template <class R, class... Args>
FunctionRef(R (*)(Args...) noexcept) -> FunctionRef<R(Args...) noexcept>;

template <auto Function>
  requires(std::is_pointer_v<decltype(Function)> &&
           std::is_function_v<std::remove_pointer_t<decltype(Function)>>)
FunctionRef(NontypeTag<Function>)
    -> FunctionRef<std::remove_pointer_t<decltype(Function)>>;

// A member-function pointer already carries its complete result, argument,
// cv, and noexcept signature. Deduce only the four signatures implemented by
// FunctionRef; ref-qualified or volatile members remain explicit rejections.
template <class Member, class Owner, Member Owner::*Callable, class Object>
  requires(std::is_function_v<Member> &&
           detail::kSupportedFunctionRefSignature<Member>)
FunctionRef(NontypeTag<Callable>, Object&&) -> FunctionRef<Member>;

}  // namespace hyle::util
