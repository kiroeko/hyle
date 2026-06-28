#pragma once

#include <type_traits>
#include <utility>

#include "hyle/macros.h"

// Go-style `defer`: run a cleanup callable when the enclosing scope exits (clay
// scope_guard). C++23 still has no std defer; this stays. Unlike clay we need NO
// scope_guard()/finally() factory functions -- class template argument deduction
// (CTAD, C++17+) deduces the callable directly: `ScopeGuard g{[&]{ ... }};`.

namespace hyle::util {

// Owns a callable and invokes it once at destruction unless dismissed. Move-only
// (a moved-from guard is dismissed so the action fires exactly once). The action
// must be noexcept-callable in practice -- it runs in a destructor.
template <class Fn>
class ScopeGuard {
  static_assert(!std::is_reference_v<Fn>,
                "ScopeGuard owns its callable by value (use CTAD on a temporary "
                "lambda, not an lvalue reference)");

 public:
  template <class F>
  explicit ScopeGuard(F&& fn) noexcept(
      std::is_nothrow_constructible_v<Fn, F>)
      : fn_(std::forward<F>(fn)) {}

  ScopeGuard(ScopeGuard&& o) noexcept(std::is_nothrow_move_constructible_v<Fn>)
      : fn_(std::move(o.fn_)), active_(o.active_) {
    o.active_ = false;
  }

  ~ScopeGuard() {
    if (active_) fn_();
  }

  // Cancel the pending action (it will not run at scope exit).
  void dismiss() noexcept { active_ = false; }

  HYLE_DISABLE_COPY(ScopeGuard);
  ScopeGuard& operator=(ScopeGuard&&) = delete;

 private:
  Fn fn_;
  bool active_ = true;
};

template <class F>
ScopeGuard(F&&) -> ScopeGuard<std::decay_t<F>>;  // deduce stored type by value

}  // namespace hyle::util

// HYLE_DEFER(stmt): run `stmt` at scope exit. The unique name (via __COUNTER__)
// lets multiple defers coexist in one scope. `[&]` captures by reference -- the
// statement runs before the captured locals leave scope.
#define HYLE_DEFER(...)                                                        \
  ::hyle::util::ScopeGuard HYLE_CONCAT(hyle_defer_, __COUNTER__) {             \
    [&]() noexcept { __VA_ARGS__; }                                           \
  }
