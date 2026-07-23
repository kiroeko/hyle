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

// Construct the error arm of any Result<T> without repeating the standard
// carrier type at every return site:
//
//   return make_unexpected(Status::NotFound("model"));
//
// Keep this Status-specific: a generic clone of std::unexpected would add no
// HYLE value and could silently create the wrong error type.
[[nodiscard]] inline std::unexpected<Status> make_unexpected(
    Status status) noexcept {
  return std::unexpected<Status>(std::move(status));
}

namespace detail {

// Adaptor for the two value-less failure carriers accepted by
// HYLE_RETURN_IF_ERROR. The success path only observes the carrier by reference;
// it does not construct/copy a Status. On failure, lvalues copy their error and
// temporaries move it.
inline bool status_ok(const Status &status) noexcept {
  return status.ok();
}
inline bool status_ok(const Result<void> &result) noexcept {
  return result.has_value();
}

inline const Status &status_error(const Status &status) noexcept {
  return status;
}
inline Status &&status_error(Status &&status) noexcept {
  return std::move(status);
}
inline const Status &status_error(const Result<void> &result) noexcept {
  return result.error();
}
inline Status &&status_error(Result<void> &&result) noexcept {
  return std::move(result).error();
}

// Bridges an error Status into whatever the enclosing function returns: a bare
// Status, or a Result<U> for any U (via std::unexpected). This is what lets a
// single propagation macro serve both kinds of function without the caller
// spelling out std::unexpected. std::expected has no implicit ctor from E, so
// the conversion has to be carried explicitly -- that is this type's whole job.
class FailReturn {
 public:
  explicit FailReturn(Status status) noexcept : status_(std::move(status)) {}

  // -> a function returning Status
  operator Status() && noexcept {
    return std::move(status_);
  }  // NOLINT(*-explicit)
  // -> a function returning Result<U>
  template <class U>
  operator Result<U>() && noexcept {  // NOLINT(*-explicit)
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

// Evaluate a Status or Result<void> expression; if not OK, return its Status
// (converted) immediately.
#define HYLE_RETURN_IF_ERROR(expr)                                         \
  do {                                                                     \
    auto &&hyle_status_ = (expr);                                          \
    if (HYLE_UNLIKELY(!::hyle::detail::status_ok(hyle_status_))) {         \
      return ::hyle::detail::FailReturn(::hyle::detail::status_error(      \
          std::forward<decltype(hyle_status_)>(hyle_status_)));            \
    }                                                                      \
  } while (0)

// Assign the value of a Result<T> to `lhs`, or early-return its error.
// `lhs` may be a declaration ("auto x") or an existing lvalue. Uses a uniquely
// named temporary (via __COUNTER__) so multiple uses in one scope never clash.
// Like other declaration-capable propagation macros, this expands to multiple
// statements: use braces when it appears under if/else/for/while.
#define HYLE_ASSIGN_OR_RETURN(lhs, expr) \
  HYLE_ASSIGN_OR_RETURN_IMPL(HYLE_CONCAT(hyle_or_, __COUNTER__), lhs, expr)

#define HYLE_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr)                          \
  auto&& tmp = (expr);                                                      \
  if (HYLE_UNLIKELY(!tmp.has_value())) {                                    \
    return ::hyle::detail::FailReturn(std::move(tmp).error());              \
  }                                                                         \
  lhs = *std::move(tmp) /* operator*: has_value() already checked above, so \
                           skip .value()'s redundant check + throw path */
