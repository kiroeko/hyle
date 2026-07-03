#pragma once

#include <expected>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/macros.h"

namespace hyle {

// Result<T> = a value of type T or a Status error. Thin alias over C++23
// std::expected so the monadic ops (and_then/transform/or_else) come for free.
template <class T>
using Result = std::expected<T, Status>;

namespace detail {

// Bridges an error Status into whatever the enclosing function returns: a bare
// Status, or a Result<U> for any U (via std::unexpected). This is what lets a
// single propagation macro serve both kinds of function without the caller
// spelling out std::unexpected. std::expected has no implicit ctor from E, so
// the conversion has to be carried explicitly -- that is this type's whole job.
class FailReturn {
 public:
  explicit FailReturn(Status status) : status_(std::move(status)) {}

  // -> a function returning Status
  operator Status() && { return std::move(status_); }  // NOLINT(*-explicit)
  // -> a function returning Result<U>
  template <class U>
  operator Result<U>() && {  // NOLINT(*-explicit)
    return std::unexpected<Status>(std::move(status_));
  }

 private:
  Status status_;
};

}  // namespace detail
}  // namespace hyle

// ---- propagation macros (Rust `?`-style early return) -----------------------
// The standard has no `?`; these are the real ergonomic increment. Both work in
// functions returning Status OR Result<T> thanks to detail::FailReturn.

// Evaluate a Status expression; if not OK, return it (converted) immediately.
#define HYLE_RETURN_NOT_OK(expr)                                               \
  do {                                                                         \
    ::hyle::Status hyle_status_ = (expr);                                      \
    if (HYLE_UNLIKELY(!hyle_status_.ok())) {                                   \
      return ::hyle::detail::FailReturn(std::move(hyle_status_));              \
    }                                                                          \
  } while (0)

// Assign the value of a Result<T> to `lhs`, or early-return its error.
// `lhs` may be a declaration ("auto x") or an existing lvalue. Uses a uniquely
// named temporary (via __COUNTER__) so multiple uses in one scope never clash.
#define HYLE_ASSIGN_OR_RAISE(lhs, expr)                                        \
  HYLE_ASSIGN_OR_RAISE_IMPL(HYLE_CONCAT(hyle_or_, __COUNTER__), lhs, expr)

#define HYLE_ASSIGN_OR_RAISE_IMPL(tmp, lhs, expr)                              \
  auto&& tmp = (expr);                                                         \
  if (HYLE_UNLIKELY(!tmp.has_value())) {                                       \
    return ::hyle::detail::FailReturn(std::move(tmp).error());                 \
  }                                                                            \
  lhs = *std::move(tmp)  /* operator*: has_value() already checked above, so   \
                            skip .value()'s redundant check + throw path */

// NOTE: HYLE_ASSERT (charter §5 soft assert) lives in error/error_handler.h --
// it routes failures through the installable diagnostic handler, not a bare
// fprintf. result.h is the lower Status/Result layer and intentionally does NOT
// define it (would clash + diverge by include order).
