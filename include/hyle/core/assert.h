/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <cstdio>
#include <cstdlib>

#include "hyle/visibility.h"

#if !defined(__cplusplus)
#  error "hyle/core/assert.h is a C++ header"
#endif

namespace hyle::core::detail {

// Keep invariant diagnostics out of every successful call site. The function
// is inline for ODR-safe header-only delivery and hidden from DSO exports;
// noinline, plus cold where supported, controls its code placement.
#if defined(_MSC_VER)
#  define HYLE_DETAIL_ASSERT_NOINLINE __declspec(noinline)
#  define HYLE_DETAIL_ASSERT_COLD
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_ASSERT_NOINLINE __attribute__((noinline))
#  define HYLE_DETAIL_ASSERT_COLD __attribute__((cold))
#else
#  define HYLE_DETAIL_ASSERT_NOINLINE
#  define HYLE_DETAIL_ASSERT_COLD
#endif

[[noreturn]] HYLE_HIDDEN HYLE_DETAIL_ASSERT_NOINLINE HYLE_DETAIL_ASSERT_COLD
    inline void
fatal_assert_failed(const char* expression, const char* message,
                    const char* file, int line) noexcept {
  if (message == nullptr) {
    std::fprintf(stderr, "HYLE fatal: %s\n  at %s:%d\n", expression, file,
                 line);
  } else {
    std::fprintf(stderr, "HYLE fatal: %s\n  %s\n  at %s:%d\n", expression,
                 message, file, line);
  }
  std::abort();
}

#undef HYLE_DETAIL_ASSERT_COLD
#undef HYLE_DETAIL_ASSERT_NOINLINE

}  // namespace hyle::core::detail

// A release-surviving program-invariant check. Recoverable input, resource and
// system failures belong in hyle::Status/Result instead.
#define HYLE_FATAL_ASSERT(cond)                                             \
  do {                                                                      \
    if (!(cond)) [[unlikely]] {                                             \
      ::hyle::core::detail::fatal_assert_failed(#cond, nullptr, __FILE__,   \
                                                __LINE__);                   \
    }                                                                       \
  } while (false)

// The message must yield a borrowed null-terminated string. It is evaluated
// exactly once and only after `cond` fails; the failure sink consumes it before
// terminating, so no ownership or formatting machinery enters the hot path.
#define HYLE_FATAL_ASSERT_MSG(cond, message)            \
  do {                                                  \
    if (!(cond)) [[unlikely]] {                         \
      ::hyle::core::detail::fatal_assert_failed(        \
          #cond, (message), __FILE__, __LINE__);        \
    }                                                   \
  } while (false)
