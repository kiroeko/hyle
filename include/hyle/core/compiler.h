/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include "hyle/core/assert.h"

#if !defined(__cplusplus)
#  error "hyle/core/compiler.h is a C++ header"
#endif

// Cross-compiler source contracts for measured native hot paths. Unknown
// compilers degrade hints safely; ASSUME/RESTRICT/UNREACHABLE remain semantic
// promises and must never be used to validate untrusted input.

#if defined(_MSC_VER) && !defined(__clang__)
#  define HYLE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  if defined(__OPTIMIZE__)
#    define HYLE_FORCE_INLINE __attribute__((always_inline)) inline
#  else
#    define HYLE_FORCE_INLINE inline
#  endif
#else
#  define HYLE_FORCE_INLINE inline
#endif

#if defined(_MSC_VER)
#  define HYLE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_NOINLINE __attribute__((noinline))
#else
#  define HYLE_NOINLINE
#endif

#if defined(_MSC_VER)
#  define HYLE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_RESTRICT __restrict__
#else
#  define HYLE_RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_FLATTEN __attribute__((flatten))
#else
#  define HYLE_FLATTEN
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  define HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens) _Pragma(#tokens)
#  define HYLE_DETAIL_COMPILER_PRAGMA(tokens) \
    HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens)
#  define HYLE_UNROLL(factor) \
    HYLE_DETAIL_COMPILER_PRAGMA(GCC unroll factor)
#  define HYLE_DEFAULT_UNROLL HYLE_UNROLL(4)
#elif defined(__clang__)
#  if defined(__OPTIMIZE__) && !__has_feature(address_sanitizer) &&           \
      !__has_feature(thread_sanitizer) && !__has_feature(memory_sanitizer) && \
      !__has_feature(undefined_behavior_sanitizer)
#    define HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens) _Pragma(#tokens)
#    define HYLE_DETAIL_COMPILER_PRAGMA(tokens) \
      HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens)
#    define HYLE_UNROLL(factor) HYLE_DETAIL_COMPILER_PRAGMA(unroll factor)
#    define HYLE_DEFAULT_UNROLL HYLE_DETAIL_COMPILER_PRAGMA(unroll)
#  else
// An unhonored Clang loop-transform pragma emits -Wpass-failed. At -O0 and
// under sanitizers, keep this best-effort hint from breaking -Werror builds.
#    define HYLE_UNROLL(factor)
#    define HYLE_DEFAULT_UNROLL
#  endif
#else
#  define HYLE_UNROLL(factor)
#  define HYLE_DEFAULT_UNROLL
#endif

// MSVC silently ignores the standard spelling for ABI compatibility and uses
// its vendor attribute to provide the intended empty-member layout.
#if defined(_MSC_VER)
#  define HYLE_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#  define HYLE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_LIKELY(cond) (__builtin_expect(!!(cond), 1))
#  define HYLE_UNLIKELY(cond) (__builtin_expect(!!(cond), 0))
#else
#  define HYLE_LIKELY(cond) (!!(cond))
#  define HYLE_UNLIKELY(cond) (!!(cond))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) __assume(!!(cond))
#elif defined(__clang__)
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) __builtin_assume(!!(cond))
#elif defined(__GNUC__)
#  define HYLE_DETAIL_COMPILER_ASSUME(cond)  \
    do {                                       \
      if (!(cond)) __builtin_unreachable();   \
    } while (false)
#else
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) ((void)sizeof(!!(cond)))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() __builtin_unreachable()
#else
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() ((void)0)
#endif

// Debug diagnoses a broken optimizer promise. Release makes the promise
// available to code generation; a false expression is then undefined behavior.
#if !defined(NDEBUG)
#  define HYLE_ASSUME(cond) HYLE_FATAL_ASSERT(cond)
#  define HYLE_UNREACHABLE() HYLE_FATAL_ASSERT(false)
#else
#  define HYLE_ASSUME(cond) HYLE_DETAIL_COMPILER_ASSUME(cond)
#  define HYLE_UNREACHABLE() HYLE_DETAIL_COMPILER_UNREACHABLE()
#endif
