/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Pure C source/compiler contracts.  These macros are header-only and do not
 * add symbols to HYLE's dynamic C ABI.
 */
#ifndef HYLE_C_COMPILER_H_
#define HYLE_C_COMPILER_H_

#include "hyle/core/compiler_defs.h"

#define HYLE_C_RESTRICT HYLE_DETAIL_COMPILER_RESTRICT
#define HYLE_C_FLATTEN HYLE_DETAIL_COMPILER_FLATTEN
#define HYLE_C_UNROLL(factor) HYLE_DETAIL_COMPILER_UNROLL(factor)
#define HYLE_C_DEFAULT_UNROLL HYLE_DETAIL_COMPILER_DEFAULT_UNROLL

/* cond must be side-effect free.  False is diagnosed in Debug and is
 * undefined behavior in optimized builds.  This is not input validation. */
#if !defined(NDEBUG)
#  include <stdio.h>
#  include <stdlib.h>
#  define HYLE_C_ASSUME(cond)                                              \
    do {                                                                  \
      if (!(cond)) {                                                      \
        fprintf(stderr, "HYLE C fatal: %s\n  at %s:%d\n", #cond,          \
                __FILE__, __LINE__);                                      \
        abort();                                                          \
      }                                                                   \
    } while (0)
#else
#  define HYLE_C_ASSUME(cond) HYLE_DETAIL_COMPILER_ASSUME(cond)
#endif

#endif /* HYLE_C_COMPILER_H_ */
