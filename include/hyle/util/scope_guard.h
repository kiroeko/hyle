#pragma once

#include <concepts>
#include <exception>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/core/detail/preprocessor.h"

// Go-style `defer`: run a cleanup callable when the enclosing scope exits (clay
// scope_guard). C++23 still has no std defer; this stays. Unlike clay we need NO
// scope_guard()/finally() factory functions -- class template argument deduction
// (CTAD, C++17+) deduces the callable directly: `ScopeGuard g{[&]{ ... }};`.

namespace hyle::util {

template <class Fn>
class ScopeGuard;

namespace detail {

template <class Stored, class Source>
concept ScopeGuardSource =
    !std::same_as<std::remove_cvref_t<Source>, ScopeGuard<Stored>> &&
    std::constructible_from<Stored, Source> && std::invocable<Source&>;

template <class Fn, bool ExecuteOnFailure>
class ConditionalScopeGuard {
  static_assert(!std::is_reference_v<Fn>);
  static_assert(std::invocable<Fn&>);

 public:
#if HYLE_HAS_EXCEPTIONS
  template <class F>
    requires std::constructible_from<Fn, F> && std::invocable<F&>
  explicit ConditionalScopeGuard(F&& fn) noexcept(
      std::is_nothrow_constructible_v<Fn, F&>)
  try : fn_(fn) {
  } catch (...) {
    // A failure guard has already accepted cleanup responsibility when its
    // callable storage fails. A success action must never run on this path.
    if constexpr (ExecuteOnFailure) fn();
  }
#else
  template <class F>
    requires std::constructible_from<Fn, F> && std::invocable<F&>
  explicit ConditionalScopeGuard(F&& fn) noexcept(
      std::is_nothrow_constructible_v<Fn, F&>)
      : fn_(fn) {}
#endif

  template <class F>
    requires std::constructible_from<Fn, F> && std::invocable<F&> &&
             (!std::is_lvalue_reference_v<F>) &&
             std::is_nothrow_constructible_v<Fn, F>
  explicit ConditionalScopeGuard(F&& fn) noexcept
      : fn_(std::forward<F>(fn)) {}

  ConditionalScopeGuard(ConditionalScopeGuard&& other) noexcept
    requires std::is_nothrow_move_constructible_v<Fn>
      : fn_(std::move(other.fn_)),
        exception_count_(other.exception_count_),
        active_(std::exchange(other.active_, false)) {}

  ConditionalScopeGuard(ConditionalScopeGuard&& other) noexcept(
      std::is_nothrow_copy_constructible_v<Fn>)
    requires (!std::is_nothrow_move_constructible_v<Fn>) &&
             std::copy_constructible<Fn>
      : fn_(other.fn_),
        exception_count_(other.exception_count_),
        active_(std::exchange(other.active_, false)) {}

  ~ConditionalScopeGuard() noexcept {
    const bool new_exception =
        std::uncaught_exceptions() > exception_count_;
    if (active_ && new_exception == ExecuteOnFailure) fn_();
  }

  void dismiss() noexcept { active_ = false; }
  [[nodiscard]] bool active() const noexcept { return active_; }

  ConditionalScopeGuard(const ConditionalScopeGuard&) = delete;
  ConditionalScopeGuard& operator=(const ConditionalScopeGuard&) = delete;
  ConditionalScopeGuard& operator=(ConditionalScopeGuard&&) = delete;

 private:
  HYLE_NO_UNIQUE_ADDRESS Fn fn_;
  int exception_count_ = std::uncaught_exceptions();
  bool active_ = true;
};

}  // namespace detail

// Owns a callable and invokes it once at destruction unless dismissed. Move-only
// (a moved-from guard is dismissed so the action fires exactly once). The action
// must be noexcept-callable in practice -- it runs in a destructor.
template <class Fn>
class [[nodiscard("a temporary ScopeGuard runs immediately; bind it to a name")]]
    ScopeGuard {
  static_assert(!std::is_reference_v<Fn>,
                "ScopeGuard owns its callable by value (use CTAD on a temporary "
                "lambda, not an lvalue reference)");
  static_assert(std::invocable<Fn&>,
                "ScopeGuard callable must be invocable with no arguments");
 public:
#if HYLE_HAS_EXCEPTIONS
  template <class F>
    requires detail::ScopeGuardSource<Fn, F>
  explicit ScopeGuard(F&& fn) noexcept(
      std::is_nothrow_constructible_v<Fn, F&>)
  try : fn_(fn) {
  } catch (...) {
    // Match scope_exit/Folly: once resource acquisition has handed us a
    // cleanup, failure to store it must not silently leak the resource.
    fn();
  }
#else
  template <class F>
    requires detail::ScopeGuardSource<Fn, F>
  explicit ScopeGuard(F&& fn) noexcept(
      std::is_nothrow_constructible_v<Fn, F&>)
      : fn_(fn) {}
#endif

  template <class F>
    requires detail::ScopeGuardSource<Fn, F> &&
             (!std::is_lvalue_reference_v<F>) &&
             std::is_nothrow_constructible_v<Fn, F>
  explicit ScopeGuard(F&& fn) noexcept : fn_(std::forward<F>(fn)) {}

  ScopeGuard(ScopeGuard&& o) noexcept
    requires std::is_nothrow_move_constructible_v<Fn>
      : fn_(std::move(o.fn_)), active_(std::exchange(o.active_, false)) {}

  ScopeGuard(ScopeGuard&& o) noexcept(std::is_nothrow_copy_constructible_v<Fn>)
    requires (!std::is_nothrow_move_constructible_v<Fn>) &&
             std::copy_constructible<Fn>
      : fn_(o.fn_), active_(std::exchange(o.active_, false)) {}

  ~ScopeGuard() noexcept {
    if (active_) fn_();
  }

  // Cancel the pending action (it will not run at scope exit).
  void dismiss() noexcept { active_ = false; }

  // Run the action now, at most once. This is useful when cleanup is also the
  // successful commit step and avoids spelling `fn(); guard.dismiss()` outside
  // the guard even though the callable is intentionally encapsulated.
  void invoke() noexcept(std::is_nothrow_invocable_v<Fn&>) {
    if (!active_) return;
    active_ = false;
    fn_();
  }

  bool active() const noexcept { return active_; }

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;

 private:
  HYLE_NO_UNIQUE_ADDRESS Fn fn_;
  bool active_ = true;
};

template <class F>
ScopeGuard(F&&) -> ScopeGuard<std::decay_t<F>>;  // deduce stored type by value

// Execute only when a new exception leaves this lexical scope.
template <class Fn>
class [[nodiscard("a temporary ScopeFail cannot observe the later scope exit")]]
    ScopeFail final : private detail::ConditionalScopeGuard<Fn, true> {
  using Base = detail::ConditionalScopeGuard<Fn, true>;

 public:
  using Base::Base;
  using Base::active;
  using Base::dismiss;
};

template <class F>
ScopeFail(F&&) -> ScopeFail<std::decay_t<F>>;

// Execute only when no new exception leaves this lexical scope.
template <class Fn>
class [[nodiscard(
    "a temporary ScopeSuccess cannot observe the later scope exit")]]
    ScopeSuccess final : private detail::ConditionalScopeGuard<Fn, false> {
  using Base = detail::ConditionalScopeGuard<Fn, false>;

 public:
  using Base::Base;
  using Base::active;
  using Base::dismiss;
};

template <class F>
ScopeSuccess(F&&) -> ScopeSuccess<std::decay_t<F>>;

}  // namespace hyle::util

// HYLE_DEFER(stmt): run `stmt` at scope exit. The unique name (via __COUNTER__)
// lets multiple defers coexist in one scope. `[&]` captures by reference -- the
// statement runs before the captured locals leave scope.
#define HYLE_DEFER(...)                                                        \
  ::hyle::util::ScopeGuard HYLE_DETAIL_CONCAT(                                 \
      hyle_defer_, __COUNTER__) {                                              \
    [&]() noexcept { __VA_ARGS__; }                                           \
  }
