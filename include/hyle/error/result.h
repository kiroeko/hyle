#pragma once

#include <expected>
#include <type_traits>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/macros.h"

namespace hyle {

namespace detail {

// Invalid Result construction is a caller bug. Keep its diagnostics out of
// every normal construction/propagation frame while preserving the loud trap.
[[noreturn]] HYLE_NOINLINE inline void result_from_ok_abort() noexcept {
  HYLE_FATAL_ASSERT(false && "Result error arm cannot contain OK Status");
  HYLE_UNREACHABLE();
}

HYLE_FORCE_INLINE void require_error(const Status &status) noexcept {
  if (HYLE_UNLIKELY(status.ok())) result_from_ok_abort();
}

// Keep Result's error arm physically identical to Status while enforcing the
// one semantic invariant that plain std::expected cannot express: an error arm
// must not contain OK. Public inheritance would let a caller upcast
// `result.error()` to mutable Status& and assign OK, bypassing every checked
// ResultError constructor/assignment. Composition prevents that escape hatch;
// the common read-only Status surface remains available directly, and explicit
// value conversions preserve the ordinary propagation/return shapes. A private
// Status base is not sufficient because the language prefers (then rejects)
// its inaccessible derived-to-base conversion before considering an operator
// that returns a Status value.
class ResultError final {
 public:
  ResultError(const Status &status) noexcept : status_(status) {
    require_error(status_);
  }
  ResultError(Status &&status) noexcept : status_(std::move(status)) {
    require_error(status_);
  }

  ResultError(const ResultError &) noexcept = default;
  ResultError(ResultError &&) noexcept = default;
  ResultError &operator=(const ResultError &) noexcept = default;
  ResultError &operator=(ResultError &&) noexcept = default;

  ResultError &operator=(const Status &status) noexcept {
    require_error(status);
    status_ = status;
    return *this;
  }
  ResultError &operator=(Status &&status) noexcept {
    require_error(status);
    status_ = std::move(status);
    return *this;
  }

  // Result observers should feel like Status without exposing a mutable base
  // reference that could violate the error-arm invariant.
  bool ok() const noexcept { return status_.ok(); }
  StatusCode code() const noexcept { return status_.code(); }
  bool has_message() const noexcept { return status_.has_message(); }
  std::string_view message() const noexcept { return status_.message(); }
  NativeError native_error() const noexcept { return status_.native_error(); }
  bool has_native_error() const noexcept {
    return status_.has_native_error();
  }
  std::size_t ToString(char *buffer, std::size_t capacity) const noexcept {
    return status_.ToString(buffer, capacity);
  }
  Status WithContext(std::string_view context) const & noexcept {
    return status_.WithContext(context);
  }
  Status WithContext(std::string_view context) && noexcept {
    return std::move(status_).WithContext(context);
  }
  void IgnoreError() const noexcept { status_.IgnoreError(); }

#define HYLE_RESULT_ERROR_PREDICATE(name, screaming, value)                  \
  bool Is##name() const noexcept { return status_.Is##name(); }
  HYLE_FOREACH_STATUS_CODE(HYLE_RESULT_ERROR_PREDICATE)
#undef HYLE_RESULT_ERROR_PREDICATE

  // Conversions deliberately produce a Status value. An lvalue copies; an
  // rvalue consumes and leaves this error as code-only Unknown. There is no
  // conversion to mutable Status&, so `Status& s = result.error()` is rejected.
  operator Status() const & noexcept { return status_; }
  operator Status() && noexcept { return std::move(status_); }

  const Status &status() const noexcept { return status_; }

  friend bool operator==(const ResultError &a,
                         const ResultError &b) noexcept {
    return a.status_ == b.status_;
  }
  friend bool operator==(const ResultError &a, const Status &b) noexcept {
    return a.status_ == b;
  }
  friend bool operator==(const Status &a, const ResultError &b) noexcept {
    return a == b.status_;
  }
  friend void swap(ResultError &a, ResultError &b) noexcept {
    using std::swap;
    swap(a.status_, b.status_);
  }

 private:
  Status status_;
};

static_assert(sizeof(ResultError) == sizeof(Status));
static_assert(!std::is_convertible_v<ResultError &, Status &>);
static_assert(std::is_convertible_v<const ResultError &, Status>);
static_assert(std::is_convertible_v<ResultError &&, Status>);
static_assert(std::is_nothrow_swappable_v<ResultError>);

}  // namespace detail

// Result<T> = a value or a non-OK Status error. It remains a thin C++23
// std::expected alias, so the standard observers and monadic operations come
// for free; the same-size checked error arm only enforces the invariant above.
template <class T>
using Result = std::expected<T, detail::ResultError>;

// Construct the error arm of any Result<T> without repeating the standard
// carrier type at every return site:
//
//   return make_unexpected(Status::NotFound("model"));
//
// Keep this Status-specific: a generic clone of std::unexpected would add no
// HYLE value and could silently create the wrong error type.
[[nodiscard]] inline std::unexpected<detail::ResultError> make_unexpected(
    Status status) noexcept {
  return std::unexpected<detail::ResultError>(std::move(status));
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
  return result.error().status();
}
inline Status status_error(Result<void> &&result) noexcept {
  return std::move(result).error();
}

// Bridges an error Status into whatever the enclosing function returns: a bare
// Status, or a Result<U> for any U (via std::unexpected). This is what lets a
// single propagation macro serve both kinds of function without the caller
// spelling out std::unexpected. std::expected has no implicit ctor from E, so
// the conversion has to be carried explicitly -- that is this type's whole job.
class FailReturn {
 public:
  explicit FailReturn(Status status) noexcept : status_(std::move(status)) {
    require_error(status_);
  }

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
#define HYLE_RETURN_IF_ERROR(expr)                                           \
  HYLE_RETURN_IF_ERROR_IMPL(HYLE_CONCAT(hyle_status_, __COUNTER__), expr)

#define HYLE_RETURN_IF_ERROR_IMPL(tmp, expr)                                 \
  do {                                                                       \
    auto &&tmp = (expr);                                                     \
    if (HYLE_UNLIKELY(!::hyle::detail::status_ok(tmp))) {                   \
      return ::hyle::detail::FailReturn(::hyle::detail::status_error(       \
          std::forward<decltype(tmp)>(tmp)));                                \
    }                                                                        \
  } while (0)

// Assign the value of a Result<T> to `lhs`, or early-return its error. An lvalue
// expression is copied; an rvalue expression is moved.
// `lhs` may be a declaration ("auto x") or an existing lvalue. Uses a uniquely
// named temporary (via __COUNTER__) so multiple uses in one scope never clash.
// Like other declaration-capable propagation macros, this expands to multiple
// statements: use braces when it appears under if/else/for/while.
#define HYLE_ASSIGN_OR_RETURN(lhs, expr) \
  HYLE_ASSIGN_OR_RETURN_IMPL(HYLE_CONCAT(hyle_or_, __COUNTER__), lhs, expr)

#define HYLE_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr)                          \
  auto&& tmp = (expr);                                                      \
  if (HYLE_UNLIKELY(!tmp.has_value())) {                                    \
    return ::hyle::detail::FailReturn(                                      \
        std::forward<decltype(tmp)>(tmp).error());                           \
  }                                                                         \
  lhs = *std::forward<decltype(tmp)>(tmp) /* operator*: has_value() already \
      checked above, so skip .value()'s redundant check + throw path */
