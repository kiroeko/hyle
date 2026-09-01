/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include "hyle/core/environment.h"

// One private spelling table is shared by assert.h and compiler.h. Public
// HYLE_* aliases live in compiler.h; these detail names are an installed
// implementation seam, not a second product API.
#if defined(_MSC_VER)
#  define HYLE_DETAIL_NOINLINE __declspec(noinline)
#elif HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  define HYLE_DETAIL_NOINLINE __attribute__((noinline))
#else
#  define HYLE_DETAIL_NOINLINE
#endif

#if HYLE_COMPILER_GCC || HYLE_COMPILER_CLANG
#  define HYLE_DETAIL_COLD __attribute__((cold))
#else
#  define HYLE_DETAIL_COLD
#endif

#if HYLE_HAS_ATTRIBUTE(disable_tail_calls)
#  define HYLE_DETAIL_NO_TAIL_CALL __attribute__((disable_tail_calls))
#elif HYLE_COMPILER_GCC
#  define HYLE_DETAIL_NO_TAIL_CALL \
    __attribute__((optimize("no-optimize-sibling-calls")))
#else
#  define HYLE_DETAIL_NO_TAIL_CALL
#endif

#if HYLE_HAS_CPP_ATTRIBUTE(clang::lifetimebound)
#  define HYLE_DETAIL_LIFETIME_BOUND [[clang::lifetimebound]]
#elif HYLE_HAS_CPP_ATTRIBUTE(msvc::lifetimebound)
#  define HYLE_DETAIL_LIFETIME_BOUND [[msvc::lifetimebound]]
#elif HYLE_HAS_ATTRIBUTE(lifetimebound)
#  define HYLE_DETAIL_LIFETIME_BOUND __attribute__((lifetimebound))
#else
#  define HYLE_DETAIL_LIFETIME_BOUND
#endif

#if defined(_MSC_VER) && !defined(__EDG__)
#  define HYLE_DETAIL_ALLOCATOR_RETURN(size_index) \
    __declspec(allocator) __declspec(restrict)
#elif defined(_MSC_VER)
#  define HYLE_DETAIL_ALLOCATOR_RETURN(size_index) __declspec(restrict)
#elif HYLE_HAS_ATTRIBUTE(malloc) && HYLE_HAS_ATTRIBUTE(alloc_size)
#  define HYLE_DETAIL_ALLOCATOR_RETURN(size_index) \
    __attribute__((malloc, alloc_size(size_index)))
#elif HYLE_HAS_ATTRIBUTE(malloc)
#  define HYLE_DETAIL_ALLOCATOR_RETURN(size_index) __attribute__((malloc))
#else
#  define HYLE_DETAIL_ALLOCATOR_RETURN(size_index)
#endif
