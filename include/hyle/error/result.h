#pragma once

#include <concepts>
#include <expected>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/core/detail/preprocessor.h"
#include "hyle/error/status.h"

namespace hyle {

template <class T>
class Result;

namespace detail {

// Invalid Result construction is a caller bug. Keep its diagnostics out of
// every normal construction/propagation frame while preserving the loud trap.
[[noreturn]] HYLE_NOINLINE inline void result_from_ok_abort() noexcept {
  HYLE_FATAL_ASSERT_MSG(false, "Result error arm cannot contain OK Status");
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

template <class T>
class BorrowedResult;

// std::expected deliberately does not accept reference value types. This
// adapter supplies the missing fallible non-null borrow without changing the
// representation or code generation of any owning Result<T>. The referent is
// always caller-owned; only the pointer or the ordinary ResultError is stored.
//
// Const is deep: const BorrowedResult<T> observes const T, and a mutable borrow
// cannot be copied back out through a const carrier. This prevents a const
// registry/result handle from silently granting mutable access. Callers that
// intentionally want shallow const should return a nullable pointer or an
// explicit std::reference_wrapper value instead.
template <class T>
class [[nodiscard]] BorrowedResult {
  static_assert(!std::is_reference_v<T>);

  template <class>
  friend class BorrowedResult;

  using Storage = std::expected<T*, ResultError>;
  using ConstT = std::add_const_t<T>;

 public:
  using value_type = T&;
  using error_type = ResultError;
  using unexpected_type = std::unexpected<ResultError>;

  template <class U>
  using rebind = ::hyle::Result<U>;

  // A reference cannot be value-initialized. Match mature reference-result
  // carriers with an explicit code-only Unknown state, and prevent `return {}`
  // from looking like a successful borrow.
  explicit BorrowedResult() noexcept
      : storage_(std::unexpect, Status::Unknown()) {}

  template <class U>
    requires std::is_convertible_v<U*, T*>
  BorrowedResult(U& value HYLE_LIFETIME_BOUND) noexcept  // NOLINT(*-explicit)
      : storage_(std::in_place, std::addressof(value)) {}

  // In particular, Result<const T&> must not bind a temporary.
  template <class U>
    requires(!std::is_lvalue_reference_v<U> &&
             std::is_convertible_v<std::remove_reference_t<U>*, T*>)
  BorrowedResult(U&&) = delete;

  template <class E>
    requires std::is_constructible_v<ResultError, const E&>
  BorrowedResult(const std::unexpected<E>& error) noexcept
      : storage_(std::unexpect, error.error()) {}

  template <class E>
    requires std::is_constructible_v<ResultError, E&&>
  BorrowedResult(std::unexpected<E>&& error) noexcept
      : storage_(std::unexpect, std::move(error).error()) {}

  // Deep-const copy semantics. A mutable carrier may copy another mutable
  // carrier; Result<const T&> remains ordinarily copyable.
  BorrowedResult(BorrowedResult&) noexcept = default;
  BorrowedResult(const BorrowedResult&) noexcept
    requires std::is_const_v<T> = default;
  BorrowedResult(const BorrowedResult&)
    requires(!std::is_const_v<T>) = delete;
  BorrowedResult(BorrowedResult&&) noexcept = default;

  BorrowedResult& operator=(BorrowedResult&) noexcept = default;
  BorrowedResult& operator=(const BorrowedResult&) noexcept
    requires std::is_const_v<T> = default;
  BorrowedResult& operator=(const BorrowedResult&)
    requires(!std::is_const_v<T>) = delete;
  BorrowedResult& operator=(BorrowedResult&&) noexcept = default;

  template <class U>
    requires(!std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
  BorrowedResult(BorrowedResult<U>& other) noexcept
      : storage_(copy_from(other)) {}

  template <class U>
    requires(!std::is_same_v<T, U> && std::is_const_v<T> &&
             std::is_convertible_v<std::add_const_t<U>*, T*>)
  BorrowedResult(const BorrowedResult<U>& other) noexcept
      : storage_(copy_from(other)) {}

  template <class U>
    requires(!std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
  BorrowedResult(BorrowedResult<U>&& other) noexcept
      : storage_(move_from(std::move(other))) {}

  template <class U>
    requires std::is_convertible_v<U*, T*>
  BorrowedResult& operator=(U& value HYLE_LIFETIME_BOUND) noexcept {
    storage_.emplace(std::addressof(value));
    return *this;
  }

  template <class E>
    requires std::is_constructible_v<ResultError, const E&>
  BorrowedResult& operator=(const std::unexpected<E>& error) noexcept {
    storage_ = std::unexpected<ResultError>(ResultError(error.error()));
    return *this;
  }

  template <class E>
    requires std::is_constructible_v<ResultError, E&&>
  BorrowedResult& operator=(std::unexpected<E>&& error) noexcept {
    storage_ = std::unexpected<ResultError>(
        ResultError(std::move(error).error()));
    return *this;
  }

  bool has_value() const noexcept { return storage_.has_value(); }
  explicit operator bool() const noexcept { return has_value(); }

  T& operator*() & noexcept { return **storage_; }
  ConstT& operator*() const& noexcept { return std::as_const(**storage_); }
  T& operator*() && noexcept { return **storage_; }
  ConstT& operator*() const&& noexcept { return std::as_const(**storage_); }

  T* operator->() noexcept { return std::addressof(**this); }
  const T* operator->() const noexcept { return std::addressof(**this); }

  T& value() & { return *storage_.value(); }
  ConstT& value() const& { return std::as_const(*storage_.value()); }
  T& value() && { return *storage_.value(); }
  ConstT& value() const&& { return std::as_const(*storage_.value()); }

  ResultError& error() & noexcept { return storage_.error(); }
  const ResultError& error() const& noexcept { return storage_.error(); }
  ResultError&& error() && noexcept { return std::move(storage_).error(); }
  const ResultError&& error() const&& noexcept {
    return std::move(storage_).error();
  }

  template <class U>
    requires std::is_convertible_v<U*, T*>
  T& emplace(U& value HYLE_LIFETIME_BOUND) noexcept {
    storage_.emplace(std::addressof(value));
    return value;
  }

  template <class U>
    requires std::is_convertible_v<U*, T*>
  T& value_or(U& fallback HYLE_LIFETIME_BOUND) & noexcept {
    return has_value() ? **this : static_cast<T&>(fallback);
  }

  template <class U>
    requires std::is_convertible_v<std::add_const_t<U>*, const T*>
  const T& value_or(U& fallback HYLE_LIFETIME_BOUND) const& noexcept {
    return has_value() ? **this : static_cast<const T&>(fallback);
  }

  template <class F>
  auto and_then(F&& function) & {
    using R = std::remove_cvref_t<std::invoke_result_t<F, T&>>;
    if (has_value()) return std::invoke(std::forward<F>(function), **this);
    return R(std::unexpected<ResultError>(error()));
  }

  template <class F>
  auto and_then(F&& function) const& {
    using R = std::remove_cvref_t<std::invoke_result_t<F, ConstT&>>;
    if (has_value()) return std::invoke(std::forward<F>(function), **this);
    return R(std::unexpected<ResultError>(error()));
  }

  template <class F>
  auto and_then(F&& function) && {
    using R = std::remove_cvref_t<std::invoke_result_t<F, T&>>;
    if (has_value()) return std::invoke(std::forward<F>(function), **this);
    return R(std::unexpected<ResultError>(std::move(*this).error()));
  }

  template <class F>
  auto and_then(F&& function) const&& {
    using R = std::remove_cvref_t<std::invoke_result_t<F, ConstT&>>;
    if (has_value()) return std::invoke(std::forward<F>(function), **this);
    return R(std::unexpected<ResultError>(error()));
  }

  template <class F>
  auto transform(F&& function) & {
    return transform_impl(*this, std::forward<F>(function));
  }

  template <class F>
  auto transform(F&& function) const& {
    return transform_impl(*this, std::forward<F>(function));
  }

  template <class F>
  auto transform(F&& function) && {
    return transform_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  auto transform(F&& function) const&& {
    return transform_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  auto or_else(F&& function) & {
    using R = std::remove_cvref_t<std::invoke_result_t<F, ResultError&>>;
    if (!has_value()) return std::invoke(std::forward<F>(function), error());
    return R(**this);
  }

  template <class F>
  auto or_else(F&& function) const& {
    using R =
        std::remove_cvref_t<std::invoke_result_t<F, const ResultError&>>;
    if (!has_value()) return std::invoke(std::forward<F>(function), error());
    return R(**this);
  }

  template <class F>
  auto or_else(F&& function) && {
    using R = std::remove_cvref_t<std::invoke_result_t<F, ResultError&&>>;
    if (!has_value()) {
      return std::invoke(std::forward<F>(function), std::move(*this).error());
    }
    return R(**this);
  }

  template <class F>
  auto or_else(F&& function) const&& {
    using R =
        std::remove_cvref_t<std::invoke_result_t<F, const ResultError&&>>;
    if (!has_value()) {
      return std::invoke(std::forward<F>(function), std::move(*this).error());
    }
    return R(**this);
  }

  void swap(BorrowedResult& other) noexcept { storage_.swap(other.storage_); }
  friend void swap(BorrowedResult& lhs, BorrowedResult& rhs) noexcept {
    lhs.swap(rhs);
  }

  template <class U>
  friend bool operator==(const BorrowedResult& lhs,
                         const BorrowedResult<U>& rhs)
    requires requires(const T& a, const U& b) {
      { a == b } -> std::convertible_to<bool>;
    }
  {
    if (lhs.has_value() != rhs.has_value()) return false;
    return lhs.has_value() ? static_cast<bool>(*lhs == *rhs)
                           : lhs.error() == rhs.error();
  }

  template <class E>
  friend bool operator==(const BorrowedResult& result,
                         const std::unexpected<E>& error)
    requires requires(const ResultError& a, const E& b) {
      { a == b } -> std::convertible_to<bool>;
    }
  {
    return !result.has_value() && result.error() == error.error();
  }

 private:
  template <class U>
  static Storage copy_from(U& other) noexcept {
    if (other.has_value()) {
      return Storage(std::in_place, std::addressof(*other));
    }
    return Storage(std::unexpect, other.error());
  }

  template <class U>
  static Storage move_from(U&& other) noexcept {
    if (other.has_value()) {
      return Storage(std::in_place, std::addressof(*other));
    }
    return Storage(std::unexpect, std::forward<U>(other).error());
  }

  template <class Self, class F>
  static auto transform_impl(Self&& self, F&& function) {
    using Arg = decltype(*std::forward<Self>(self));
    using U = std::invoke_result_t<F, Arg>;
    static_assert(!std::is_rvalue_reference_v<U>,
                  "Result<T&>::transform cannot create an rvalue borrow");
    using R = ::hyle::Result<U>;
    if (!self.has_value()) {
      if constexpr (std::is_rvalue_reference_v<Self&&> &&
                    !std::is_const_v<std::remove_reference_t<Self>>) {
        return R(std::unexpected<ResultError>(
            std::forward<Self>(self).error()));
      } else {
        return R(std::unexpected<ResultError>(self.error()));
      }
    }
    if constexpr (std::is_void_v<U>) {
      std::invoke(std::forward<F>(function), *std::forward<Self>(self));
      return R{};
    } else {
      return R(std::invoke(std::forward<F>(function),
                           *std::forward<Self>(self)));
    }
  }

  Storage storage_;
};

}  // namespace detail

// Result<T> = a value or a non-OK Status error. Owning/void results retain the
// std::expected representation and observer surface, while a public class
// template (rather than a selector alias) preserves ordinary deduction in
// generic code: `template<class T> f(const Result<T>&)` remains usable after
// admitting Result<T&>.
template <class T>
class [[nodiscard]] Result : public std::expected<T, detail::ResultError> {
  static_assert(!std::is_reference_v<T>);
  using Base = std::expected<T, detail::ResultError>;

 public:
  using Base::Base;
  using Base::operator=;
  using Base::transform_error;

  Result() = default;
  Result(const Result&) = default;
  Result(Result&&) noexcept(std::is_nothrow_move_constructible_v<Base>) =
      default;
  Result& operator=(const Result&) = default;
  Result& operator=(Result&&) noexcept(std::is_nothrow_move_assignable_v<Base>) =
      default;

  // Preserve direct interoperation with the standard carrier that backs this
  // specialization. Public inheritance already provides the reverse view.
  Result(const Base& other)  // NOLINT(*-explicit)
    requires std::is_copy_constructible_v<Base>
      : Base(other) {}
  Result(Base&& other) noexcept(std::is_nothrow_move_constructible_v<Base>)
      : Base(std::move(other)) {}  // NOLINT(*-explicit)

  template <class F>
  constexpr auto and_then(F&& function) & {
    return and_then_impl(*this, std::forward<F>(function));
  }

  template <class F>
  constexpr auto and_then(F&& function) const& {
    return and_then_impl(*this, std::forward<F>(function));
  }

  template <class F>
  constexpr auto and_then(F&& function) && {
    return and_then_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  constexpr auto and_then(F&& function) const&& {
    return and_then_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  constexpr auto transform(F&& function) & {
    return transform_impl(*this, std::forward<F>(function));
  }

  template <class F>
  constexpr auto transform(F&& function) const& {
    return transform_impl(*this, std::forward<F>(function));
  }

  template <class F>
  constexpr auto transform(F&& function) && {
    return transform_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  constexpr auto transform(F&& function) const&& {
    return transform_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  constexpr auto or_else(F&& function) & {
    return or_else_impl(*this, std::forward<F>(function));
  }

  template <class F>
  constexpr auto or_else(F&& function) const& {
    return or_else_impl(*this, std::forward<F>(function));
  }

  template <class F>
  constexpr auto or_else(F&& function) && {
    return or_else_impl(std::move(*this), std::forward<F>(function));
  }

  template <class F>
  constexpr auto or_else(F&& function) const&& {
    return or_else_impl(std::move(*this), std::forward<F>(function));
  }

  friend constexpr bool operator==(const Result& lhs, const Result& rhs)
    requires requires(const Base& a, const Base& b) {
      { a == b } -> std::convertible_to<bool>;
    }
  {
    return static_cast<const Base&>(lhs) == static_cast<const Base&>(rhs);
  }

  template <class U>
  friend constexpr bool operator==(const Result& result, const U& value)
    requires(!std::is_void_v<T> &&
             !std::is_same_v<std::remove_cvref_t<U>, Result> &&
             requires(const T& a, const U& b) {
               { a == b } -> std::convertible_to<bool>;
             })
  {
    return result.has_value() && static_cast<bool>(*result == value);
  }

  template <class E>
  friend constexpr bool operator==(const Result& result,
                                   const std::unexpected<E>& error)
    requires requires(const detail::ResultError& a, const E& b) {
      { a == b } -> std::convertible_to<bool>;
    }
  {
    return !result.has_value() && result.error() == error.error();
  }

  friend constexpr void swap(Result& lhs, Result& rhs)
      noexcept(noexcept(static_cast<Base&>(lhs).swap(
          static_cast<Base&>(rhs)))) {
    static_cast<Base&>(lhs).swap(static_cast<Base&>(rhs));
  }

 private:
  template <class Self>
  static constexpr decltype(auto) forwarded_error(Self&& self) noexcept {
    return std::forward<Self>(self).error();
  }

  template <class Self, class F>
  static constexpr auto and_then_impl(Self&& self, F&& function) {
    if constexpr (std::is_void_v<T>) {
      using R = std::remove_cvref_t<std::invoke_result_t<F>>;
      static_assert(
          std::is_same_v<typename R::error_type, detail::ResultError>,
          "Result<T>::and_then requires the HYLE canonical error type");
      if (self.has_value()) {
        return std::invoke(std::forward<F>(function));
      }
      return R(std::unexpected<detail::ResultError>(
          forwarded_error(std::forward<Self>(self))));
    } else {
      using Arg = decltype(*std::forward<Self>(self));
      using R = std::remove_cvref_t<std::invoke_result_t<F, Arg>>;
      static_assert(
          std::is_same_v<typename R::error_type, detail::ResultError>,
          "Result<T>::and_then requires the HYLE canonical error type");
      if (self.has_value()) {
        return std::invoke(std::forward<F>(function),
                           *std::forward<Self>(self));
      }
      return R(std::unexpected<detail::ResultError>(
          forwarded_error(std::forward<Self>(self))));
    }
  }

  template <class Self, class F>
  static constexpr auto transform_impl(Self&& self, F&& function) {
    if constexpr (std::is_void_v<T>) {
      using U = std::invoke_result_t<F>;
      static_assert(!std::is_rvalue_reference_v<U>,
                    "Result<T>::transform cannot create an rvalue borrow");
      using R = Result<std::remove_cv_t<U>>;
      if (!self.has_value()) {
        return R(std::unexpected<detail::ResultError>(
            forwarded_error(std::forward<Self>(self))));
      }
      if constexpr (std::is_void_v<U>) {
        std::invoke(std::forward<F>(function));
        return R{};
      } else {
        return R(std::invoke(std::forward<F>(function)));
      }
    } else {
      using Arg = decltype(*std::forward<Self>(self));
      using U = std::invoke_result_t<F, Arg>;
      static_assert(!std::is_rvalue_reference_v<U>,
                    "Result<T>::transform cannot create an rvalue borrow");
      using R = Result<std::remove_cv_t<U>>;
      if (!self.has_value()) {
        return R(std::unexpected<detail::ResultError>(
            forwarded_error(std::forward<Self>(self))));
      }
      if constexpr (std::is_void_v<U>) {
        std::invoke(std::forward<F>(function),
                    *std::forward<Self>(self));
        return R{};
      } else {
        return R(std::invoke(std::forward<F>(function),
                             *std::forward<Self>(self)));
      }
    }
  }

  template <class Self, class F>
  static constexpr auto or_else_impl(Self&& self, F&& function) {
    using ErrorArg = decltype(forwarded_error(std::forward<Self>(self)));
    using R = std::remove_cvref_t<std::invoke_result_t<F, ErrorArg>>;
    static_assert(std::is_same_v<typename R::value_type, T>,
                  "Result<T>::or_else must preserve the value type");
    if (!self.has_value()) {
      return std::invoke(std::forward<F>(function),
                         forwarded_error(std::forward<Self>(self)));
    }
    if constexpr (std::is_void_v<T>) {
      return R{};
    } else {
      return R(std::in_place, *std::forward<Self>(self));
    }
  }
};

// Reference results reuse the dedicated deep-const, non-null borrow carrier,
// but remain a specialization of the same deducible public class template.
template <class T>
class [[nodiscard]] Result<T&> final : public detail::BorrowedResult<T> {
  using Base = detail::BorrowedResult<T>;

 public:
  using Base::Base;
  using Base::operator=;

  explicit Result() noexcept = default;
  Result(Result&) noexcept = default;
  Result(const Result&) noexcept
    requires std::is_const_v<T> = default;
  Result(const Result&)
    requires(!std::is_const_v<T>) = delete;
  Result(Result&&) noexcept = default;

  Result& operator=(Result&) noexcept = default;
  Result& operator=(const Result&) noexcept
    requires std::is_const_v<T> = default;
  Result& operator=(const Result&)
    requires(!std::is_const_v<T>) = delete;
  Result& operator=(Result&&) noexcept = default;

  template <class U>
    requires(!std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
  Result(Result<U&>& other) noexcept
      : Base(static_cast<detail::BorrowedResult<U>&>(other)) {}

  template <class U>
    requires(!std::is_same_v<T, U> && std::is_const_v<T> &&
             std::is_convertible_v<std::add_const_t<U>*, T*>)
  Result(const Result<U&>& other) noexcept
      : Base(static_cast<const detail::BorrowedResult<U>&>(other)) {}

  template <class U>
    requires(!std::is_same_v<T, U> && std::is_convertible_v<U*, T*>)
  Result(Result<U&>&& other) noexcept
      : Base(static_cast<detail::BorrowedResult<U>&&>(other)) {}

  friend bool operator==(const Result& lhs, const Result& rhs)
    requires requires(const Base& a, const Base& b) {
      { a == b } -> std::convertible_to<bool>;
    }
  {
    return static_cast<const Base&>(lhs) == static_cast<const Base&>(rhs);
  }

  template <class U>
  friend bool operator==(const Result& result, const U& value)
    requires(!std::is_same_v<std::remove_cvref_t<U>, Result> &&
             requires(const T& a, const U& b) {
               { a == b } -> std::convertible_to<bool>;
             })
  {
    return result.has_value() && static_cast<bool>(*result == value);
  }

  template <class E>
  friend bool operator==(const Result& result,
                         const std::unexpected<E>& error)
    requires requires(const detail::ResultError& a, const E& b) {
      { a == b } -> std::convertible_to<bool>;
    }
  {
    return static_cast<const Base&>(result) == error;
  }

  friend void swap(Result& lhs, Result& rhs) noexcept {
    static_cast<Base&>(lhs).swap(static_cast<Base&>(rhs));
  }
};

static_assert(sizeof(Result<int>) ==
              sizeof(std::expected<int, detail::ResultError>));
static_assert(alignof(Result<int>) ==
              alignof(std::expected<int, detail::ResultError>));
static_assert(sizeof(Result<void>) ==
              sizeof(std::expected<void, detail::ResultError>));

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
  // -> a function returning a standard expected carrier.
  template <class U>
  operator std::expected<U, ResultError>() && noexcept {  // NOLINT(*-explicit)
    return std::unexpected<Status>(std::move(status_));
  }

  // -> a function returning any HYLE Result<U>, owning or borrowed.
  template <class U>
  operator ::hyle::Result<U>() && noexcept {  // NOLINT(*-explicit)
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
  HYLE_RETURN_IF_ERROR_IMPL(HYLE_DETAIL_CONCAT(hyle_status_, __COUNTER__), expr)

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
  HYLE_ASSIGN_OR_RETURN_IMPL(HYLE_DETAIL_CONCAT(hyle_or_, __COUNTER__), lhs, expr)

#define HYLE_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr)                          \
  auto&& tmp = (expr);                                                      \
  if (HYLE_UNLIKELY(!tmp.has_value())) {                                    \
    return ::hyle::detail::FailReturn(                                      \
        std::forward<decltype(tmp)>(tmp).error());                           \
  }                                                                         \
  lhs = *std::forward<decltype(tmp)>(tmp) /* operator*: has_value() already \
      checked above, so skip .value()'s redundant check + throw path */
