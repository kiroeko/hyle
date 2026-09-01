/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <type_traits>

#include "hyle/core/detail/compiler_attributes.h"
#include "hyle/visibility.h"

#if !defined(__cplusplus)
#  error "hyle/core/assert.h is a C++ header"
#endif

namespace hyle::core {

inline constexpr std::size_t kMaxFatalFields = 8;
inline constexpr std::size_t kMaxFatalTextBytes = 256;

enum class FatalKind : std::uint8_t {
  kInvariant,
  kDebugAssertion,
};

enum class FatalValueKind : std::uint8_t {
  kBoolean,
  kSigned,
  kUnsigned,
  kFloating,
  kPointer,
  kString,
};

// A failure-only, non-owning value. Keeping the alternatives explicit makes a
// host observer independent of printf format strings and guarantees that Core
// never allocates while describing an invariant failure.
struct FatalValue {
  FatalValueKind kind = FatalValueKind::kUnsigned;
  bool boolean_value = false;
  std::int64_t signed_value = 0;
  std::uint64_t unsigned_value = 0;
  double floating_value = 0.0;
  const void* pointer_value = nullptr;
  std::string_view string_value{};
};

struct FatalField {
  std::string_view name;
  FatalValue value;
};

struct FatalEvent {
  FatalKind kind;
  std::string_view expression;
  std::string_view message;
  std::string_view file;
  int line;
  std::span<const FatalField> fields;
};

// Observers run synchronously on the failing thread, may be exchanged
// concurrently, must not retain borrowed event views and cannot recover. The
// callback is an image-local header-only seam, not a cross-DSO crash manager.
using FatalObserver = void (*)(const FatalEvent& event) noexcept;

inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND, bool value) noexcept {
  FatalValue result;
  result.kind = FatalValueKind::kBoolean;
  result.boolean_value = value;
  return {name, result};
}

template <typename T>
  requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool> &&
           std::is_signed_v<T>)
inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND, T value) noexcept {
  static_assert(sizeof(T) <= sizeof(std::int64_t));
  FatalValue result;
  result.kind = FatalValueKind::kSigned;
  result.signed_value = static_cast<std::int64_t>(value);
  return {name, result};
}

template <typename T>
  requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool> &&
           std::is_unsigned_v<T>)
inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND, T value) noexcept {
  static_assert(sizeof(T) <= sizeof(std::uint64_t));
  FatalValue result;
  result.kind = FatalValueKind::kUnsigned;
  result.unsigned_value = static_cast<std::uint64_t>(value);
  return {name, result};
}

template <typename T>
  requires std::is_enum_v<T>
inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND, T value) noexcept {
  return fatal_field(name, static_cast<std::underlying_type_t<T>>(value));
}

template <typename T>
  requires std::is_floating_point_v<T>
inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND, T value) noexcept {
  FatalValue result;
  result.kind = FatalValueKind::kFloating;
  result.floating_value = static_cast<double>(value);
  return {name, result};
}

template <typename T>
  requires(std::is_object_v<T> &&
           !std::is_same_v<std::remove_cv_t<T>, char>)
inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND, T* value) noexcept {
  FatalValue result;
  result.kind = FatalValueKind::kPointer;
  result.pointer_value =
      const_cast<const void*>(static_cast<const volatile void*>(value));
  return {name, result};
}

inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND,
    std::nullptr_t) noexcept {
  FatalValue result;
  result.kind = FatalValueKind::kPointer;
  return {name, result};
}

inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND,
    std::string_view value HYLE_DETAIL_LIFETIME_BOUND) noexcept {
  FatalValue result;
  result.kind = FatalValueKind::kString;
  result.string_value = value;
  return {name, result};
}

inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND,
    const char* value HYLE_DETAIL_LIFETIME_BOUND) noexcept {
  return fatal_field(name, value == nullptr ? std::string_view{}
                                            : std::string_view(value));
}

inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND,
    char* value HYLE_DETAIL_LIFETIME_BOUND) noexcept {
  return fatal_field(name, static_cast<const char*>(value));
}

template <std::size_t N>
inline FatalField fatal_field(
    std::string_view name HYLE_DETAIL_LIFETIME_BOUND,
    const char (&value)[N]) noexcept {
  static_assert(N != 0);
  return fatal_field(name, std::string_view(value, N - 1));
}

namespace detail {

HYLE_HIDDEN inline std::string_view fatal_text(
    std::string_view text) noexcept {
  return text;
}

HYLE_HIDDEN inline std::string_view fatal_text(const char* text) noexcept {
  return text == nullptr ? std::string_view{} : std::string_view(text);
}

HYLE_HIDDEN inline std::string_view fatal_text(char* text) noexcept {
  return fatal_text(static_cast<const char*>(text));
}

template <std::size_t N>
HYLE_HIDDEN inline std::string_view fatal_text(
    const char (&text)[N]) noexcept {
  static_assert(N != 0);
  return std::string_view(text, N - 1);
}

HYLE_HIDDEN inline std::atomic<FatalObserver>& fatal_observer_slot() noexcept {
  static std::atomic<FatalObserver> observer{nullptr};
  return observer;
}

HYLE_HIDDEN inline bool& fatal_observer_active() noexcept {
  static thread_local bool active = false;
  return active;
}

HYLE_HIDDEN inline void write_bounded(
    std::FILE* stream, std::string_view text) noexcept {
  const std::size_t count =
      text.size() < kMaxFatalTextBytes ? text.size() : kMaxFatalTextBytes;
  if (count != 0) {
    static_cast<void>(std::fwrite(text.data(), 1, count, stream));
  }
  if (count != text.size()) {
    static_cast<void>(std::fwrite("...", 1, 3, stream));
  }
}

HYLE_HIDDEN inline void write_fatal_value(
    std::FILE* stream, const FatalValue& value) noexcept {
  switch (value.kind) {
    case FatalValueKind::kBoolean:
      static_cast<void>(std::fputs(value.boolean_value ? "true" : "false",
                                   stream));
      return;
    case FatalValueKind::kSigned:
      static_cast<void>(std::fprintf(
          stream, "%lld", static_cast<long long>(value.signed_value)));
      return;
    case FatalValueKind::kUnsigned:
      static_cast<void>(std::fprintf(
          stream, "%llu", static_cast<unsigned long long>(value.unsigned_value)));
      return;
    case FatalValueKind::kFloating:
      static_cast<void>(std::fprintf(stream, "%.17g", value.floating_value));
      return;
    case FatalValueKind::kPointer:
      static_cast<void>(std::fprintf(
          stream, "%p", const_cast<void*>(value.pointer_value)));
      return;
    case FatalValueKind::kString:
      write_bounded(stream, value.string_value);
      return;
  }
}

HYLE_HIDDEN inline void write_fatal_event(const FatalEvent& event) noexcept {
  static_cast<void>(std::fputs(
      event.kind == FatalKind::kInvariant ? "HYLE fatal: "
                                          : "HYLE debug assertion: ",
      stderr));
  write_bounded(stderr, event.expression);
  if (!event.message.empty()) {
    static_cast<void>(std::fputs("\n  ", stderr));
    write_bounded(stderr, event.message);
  }
  for (const FatalField& field : event.fields) {
    static_cast<void>(std::fputs("\n  ", stderr));
    write_bounded(stderr, field.name);
    static_cast<void>(std::fputs(" = ", stderr));
    write_fatal_value(stderr, field.value);
  }
  static_cast<void>(std::fputs("\n  at ", stderr));
  write_bounded(stderr, event.file);
  static_cast<void>(std::fprintf(stderr, ":%d\n", event.line));
}

// Keep invariant diagnostics out of every successful call site. noinline/cold
// controls code placement; no-tail-call preserves the failure frame where the
// compiler supports it. Observer state is touched only after failure.
[[noreturn]] HYLE_HIDDEN HYLE_DETAIL_NOINLINE HYLE_DETAIL_COLD
    HYLE_DETAIL_NO_TAIL_CALL inline void fatal_assert_failed(
        FatalKind kind, std::string_view expression, std::string_view message,
        std::string_view file, int line,
        std::span<const FatalField> fields) noexcept {
  const FatalEvent event{kind, expression, message, file, line, fields};
  FatalObserver observer =
      fatal_observer_slot().load(std::memory_order_acquire);
  bool& active = fatal_observer_active();
  if (observer != nullptr && !active) {
    active = true;
    observer(event);
    active = false;
  } else {
    write_fatal_event(event);
  }
  static_cast<void>(std::fflush(stderr));
  std::abort();
}

}  // namespace detail

inline FatalObserver set_fatal_observer(FatalObserver observer) noexcept {
  return detail::fatal_observer_slot().exchange(observer,
                                                 std::memory_order_acq_rel);
}

}  // namespace hyle::core

#define HYLE_FATAL_FIELD(value) \
  ::hyle::core::fatal_field(#value, (value))

#define HYLE_FATAL_NAMED_FIELD(name, value) \
  ::hyle::core::fatal_field((name), (value))

// Release-surviving program-invariant checks. Recoverable input, resource and
// system failures belong in hyle::Status/Result instead.
#define HYLE_FATAL_ASSERT(cond)                                             \
  do {                                                                      \
    if (!(cond)) [[unlikely]] {                                             \
      ::hyle::core::detail::fatal_assert_failed(                            \
          ::hyle::core::FatalKind::kInvariant, #cond, {}, __FILE__,         \
          __LINE__, {});                                                    \
    }                                                                       \
  } while (false)

// The message is evaluated exactly once and only after `cond` fails. It is
// borrowed synchronously and may be a C string or std::string_view.
#define HYLE_FATAL_ASSERT_MSG(cond, message)                                \
  do {                                                                      \
    if (!(cond)) [[unlikely]] {                                             \
      ::hyle::core::detail::fatal_assert_failed(                            \
          ::hyle::core::FatalKind::kInvariant, #cond,                       \
          ::hyle::core::detail::fatal_text((message)), __FILE__, __LINE__,  \
          {});                                                              \
    }                                                                       \
  } while (false)

// Context accepts one to eight FatalField expressions. Fields are evaluated
// only on failure and are borrowed until the synchronous observer returns.
#define HYLE_FATAL_ASSERT_CONTEXT(cond, message, ...)                       \
  do {                                                                      \
    if (!(cond)) [[unlikely]] {                                             \
      const ::hyle::core::FatalField hyle_fatal_fields[] = {__VA_ARGS__};   \
      static_assert(sizeof(hyle_fatal_fields) / sizeof(hyle_fatal_fields[0]) \
                        <= ::hyle::core::kMaxFatalFields,                    \
                    "HYLE fatal context supports at most eight fields");   \
      ::hyle::core::detail::fatal_assert_failed(                            \
          ::hyle::core::FatalKind::kInvariant, #cond,                       \
          ::hyle::core::detail::fatal_text((message)), __FILE__, __LINE__,  \
          std::span<const ::hyle::core::FatalField>(hyle_fatal_fields));    \
    }                                                                       \
  } while (false)

// Debug assertions are enabled in Debug, sanitizer and static-analyzer TUs.
// Disabled expressions remain type-checked but are never evaluated.
#define HYLE_DEBUG_ASSERT_ENABLED \
  (HYLE_BUILD_DEBUG || HYLE_SANITIZER_ANY || HYLE_STATIC_ANALYZER)

#if HYLE_DEBUG_ASSERT_ENABLED
#  define HYLE_DEBUG_ASSERT(cond)                                           \
    do {                                                                    \
      if (!(cond)) [[unlikely]] {                                           \
        ::hyle::core::detail::fatal_assert_failed(                          \
            ::hyle::core::FatalKind::kDebugAssertion, #cond, {}, __FILE__, \
            __LINE__, {});                                                  \
      }                                                                     \
    } while (false)
#  define HYLE_DEBUG_ASSERT_MSG(cond, message)                              \
    do {                                                                    \
      if (!(cond)) [[unlikely]] {                                           \
        ::hyle::core::detail::fatal_assert_failed(                          \
            ::hyle::core::FatalKind::kDebugAssertion, #cond,                \
            ::hyle::core::detail::fatal_text((message)), __FILE__,          \
            __LINE__, {});                                                  \
      }                                                                     \
    } while (false)
#  define HYLE_DEBUG_ASSERT_CONTEXT(cond, message, ...)                     \
    do {                                                                    \
      if (!(cond)) [[unlikely]] {                                           \
        const ::hyle::core::FatalField hyle_fatal_fields[] = {__VA_ARGS__}; \
        static_assert(                                                      \
            sizeof(hyle_fatal_fields) / sizeof(hyle_fatal_fields[0]) <=     \
                ::hyle::core::kMaxFatalFields,                              \
            "HYLE debug context supports at most eight fields");           \
        ::hyle::core::detail::fatal_assert_failed(                          \
            ::hyle::core::FatalKind::kDebugAssertion, #cond,                \
            ::hyle::core::detail::fatal_text((message)), __FILE__,          \
            __LINE__,                                                       \
            std::span<const ::hyle::core::FatalField>(hyle_fatal_fields));  \
      }                                                                     \
    } while (false)
#else
#  define HYLE_DEBUG_ASSERT(cond)                                           \
    do {                                                                    \
      if constexpr (false) {                                                \
        static_cast<void>(cond);                                            \
      }                                                                     \
    } while (false)
#  define HYLE_DEBUG_ASSERT_MSG(cond, message)                              \
    do {                                                                    \
      if constexpr (false) {                                                \
        static_cast<void>(cond);                                            \
        static_cast<void>(::hyle::core::detail::fatal_text((message)));     \
      }                                                                     \
    } while (false)
#  define HYLE_DEBUG_ASSERT_CONTEXT(cond, message, ...)                     \
    do {                                                                    \
      if constexpr (false) {                                                \
        static_cast<void>(cond);                                            \
        static_cast<void>(::hyle::core::detail::fatal_text((message)));     \
        const ::hyle::core::FatalField hyle_fatal_fields[] = {__VA_ARGS__}; \
        static_cast<void>(hyle_fatal_fields);                               \
      }                                                                     \
    } while (false)
#endif
