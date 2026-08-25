/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C source/compiler contracts.  These macros are header-only and do not
 * add symbols to HYLE's dynamic C ABI.
 */
#ifndef HYLE_C_COMPILER_H_
#define HYLE_C_COMPILER_H_

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__cplusplus)
#include <sal.h>
#endif

#include "hyle/core/compiler_defs.h"

#define HYLE_C_FORCE_INLINE HYLE_DETAIL_COMPILER_FORCE_INLINE
#define HYLE_C_NOINLINE HYLE_DETAIL_COMPILER_NOINLINE
#define HYLE_C_NODISCARD HYLE_DETAIL_COMPILER_NODISCARD
#define HYLE_C_RESTRICT HYLE_DETAIL_COMPILER_RESTRICT
#define HYLE_C_FLATTEN HYLE_DETAIL_COMPILER_FLATTEN
#define HYLE_C_UNROLL(factor) HYLE_DETAIL_COMPILER_UNROLL(factor)
#define HYLE_C_DEFAULT_UNROLL HYLE_DETAIL_COMPILER_DEFAULT_UNROLL
#define HYLE_C_LIKELY(cond) HYLE_DETAIL_COMPILER_LIKELY(cond)
#define HYLE_C_UNLIKELY(cond) HYLE_DETAIL_COMPILER_UNLIKELY(cond)

#include <stdio.h>
#include <stdlib.h>

/* Release-surviving invariant check. Use an error/status for bad user input. */
#define HYLE_C_FATAL_ASSERT(cond)                                           \
  do {                                                                      \
    if (HYLE_C_UNLIKELY(!(cond))) {                                         \
      fprintf(stderr, "HYLE C fatal: %s\n  at %s:%d\n", #cond, __FILE__,  \
              __LINE__);                                                    \
      abort();                                                              \
    }                                                                       \
  } while (0)

/* Preserve the optimizer contract in Release, but diagnose an accidental
 * trip in Debug instead of immediately entering undefined behavior. */
#if !defined(NDEBUG)
#  define HYLE_C_UNREACHABLE() HYLE_C_FATAL_ASSERT(0)
#else
#  define HYLE_C_UNREACHABLE() HYLE_DETAIL_COMPILER_UNREACHABLE()
#endif

/* cond must be side-effect free.  False is diagnosed in Debug and is
 * undefined behavior in optimized builds.  This is not input validation. */
#if !defined(NDEBUG)
#  define HYLE_C_ASSUME(cond) HYLE_C_FATAL_ASSERT(cond)
#else
#  define HYLE_C_ASSUME(cond) HYLE_DETAIL_COMPILER_ASSUME(cond)
#endif

#endif /* HYLE_C_COMPILER_H_ */
