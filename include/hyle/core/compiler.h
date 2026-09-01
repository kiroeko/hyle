/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include "hyle/core/detail/compiler_attributes.h"
#include "hyle/core/assert.h"

#if !defined(__cplusplus)
#  error "hyle/core/compiler.h is a C++ header"
#endif

// Cross-compiler source contracts for measured native hot paths. Unknown
// compilers degrade hints safely; ASSUME/RESTRICT/UNREACHABLE remain semantic
// promises and must never be used to validate untrusted input.

#if HYLE_COMPILER_MSVC
#  define HYLE_FORCE_INLINE __forceinline
#elif HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  if defined(__OPTIMIZE__)
#    define HYLE_FORCE_INLINE __attribute__((always_inline)) inline
#  else
#    define HYLE_FORCE_INLINE inline
#  endif
#else
#  define HYLE_FORCE_INLINE inline
#endif

#define HYLE_NOINLINE HYLE_DETAIL_NOINLINE
#define HYLE_COLD HYLE_DETAIL_COLD
#define HYLE_NO_TAIL_CALL HYLE_DETAIL_NO_TAIL_CALL
#define HYLE_LIFETIME_BOUND HYLE_DETAIL_LIFETIME_BOUND
#define HYLE_ALLOCATOR_RETURN(size_index) \
  HYLE_DETAIL_ALLOCATOR_RETURN(size_index)

#if defined(_MSC_VER)
#  define HYLE_RESTRICT __restrict
#elif HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  define HYLE_RESTRICT __restrict__
#else
#  define HYLE_RESTRICT
#endif

#if HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  define HYLE_FLATTEN __attribute__((flatten))
#else
#  define HYLE_FLATTEN
#endif

#if HYLE_COMPILER_GCC
#  define HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens) _Pragma(#tokens)
#  define HYLE_DETAIL_COMPILER_PRAGMA(tokens) \
    HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens)
#  define HYLE_UNROLL(factor) \
    HYLE_DETAIL_COMPILER_PRAGMA(GCC unroll factor)
#  define HYLE_DEFAULT_UNROLL HYLE_UNROLL(4)
#elif HYLE_COMPILER_CLANG
#  if defined(__OPTIMIZE__) && !HYLE_SANITIZER_ANY
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

#if HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  define HYLE_LIKELY(cond) (__builtin_expect(!!(cond), 1))
#  define HYLE_UNLIKELY(cond) (__builtin_expect(!!(cond), 0))
#else
#  define HYLE_LIKELY(cond) (!!(cond))
#  define HYLE_UNLIKELY(cond) (!!(cond))
#endif

#if HYLE_COMPILER_MSVC
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) __assume(!!(cond))
#elif HYLE_COMPILER_CLANG
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) __builtin_assume(!!(cond))
#elif HYLE_COMPILER_GCC
#  define HYLE_DETAIL_COMPILER_ASSUME(cond)  \
    do {                                       \
      if (!(cond)) __builtin_unreachable();   \
    } while (false)
#else
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) ((void)sizeof(!!(cond)))
#endif

#if HYLE_COMPILER_MSVC
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() __assume(0)
#elif HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() __builtin_unreachable()
#else
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() ((void)0)
#endif

// Debug diagnoses a broken optimizer promise. Release makes the promise
// available to code generation; a false expression is then undefined behavior.
// `cond` must be free of side effects: compiler spellings are permitted to
// discard its evaluation, while the Debug diagnostic and GCC fallback evaluate
// it. This is an optimizer contract, never a way to perform required work.
#if HYLE_BUILD_DEBUG
#  define HYLE_ASSUME(cond) HYLE_FATAL_ASSERT(cond)
#  define HYLE_UNREACHABLE() HYLE_FATAL_ASSERT(false)
#else
#  define HYLE_ASSUME(cond) HYLE_DETAIL_COMPILER_ASSUME(cond)
#  define HYLE_UNREACHABLE() HYLE_DETAIL_COMPILER_UNREACHABLE()
#endif
