/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
 *
 * Language-neutral compiler spellings shared by HYLE's C and C++ source
 * surfaces.  These HYLE_DETAIL_* names are implementation details; users
 * include hyle/macros.h or hyle/c/compiler.h instead.
 */
#ifndef HYLE_CORE_COMPILER_DEFS_H_
#define HYLE_CORE_COMPILER_DEFS_H_

/* A source-level "the result carries information that must be inspected"
 * contract.  C++ has a standard spelling; GCC/Clang expose the equivalent in
 * C17, while native MSVC's C frontend records the contract for /analyze
 * through SAL. c/compiler.h includes sal.h before exposing that spelling. */
#if defined(__cplusplus)
#  define HYLE_DETAIL_COMPILER_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
#  define HYLE_DETAIL_COMPILER_NODISCARD _Check_return_
#else
#  define HYLE_DETAIL_COMPILER_NODISCARD
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#  define HYLE_DETAIL_COMPILER_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  if defined(__OPTIMIZE__)
#    define HYLE_DETAIL_COMPILER_FORCE_INLINE __attribute__((always_inline)) inline
#  else
/* At -O0, forcing thousands of SIMD/template leaves to inline dramatically
 * inflates compile time and can trigger compiler timeouts while providing no
 * useful optimized code.  Match Highway's debug-friendly source contract. */
#    define HYLE_DETAIL_COMPILER_FORCE_INLINE inline
#  endif
#else
#  define HYLE_DETAIL_COMPILER_FORCE_INLINE inline
#endif

#if defined(_MSC_VER)
#  define HYLE_DETAIL_COMPILER_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_NOINLINE __attribute__((noinline))
#else
#  define HYLE_DETAIL_COMPILER_NOINLINE
#endif

#if defined(_MSC_VER)
#  define HYLE_DETAIL_COMPILER_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_RESTRICT __restrict__
#else
#  define HYLE_DETAIL_COMPILER_RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_FLATTEN __attribute__((flatten))
#else
#  define HYLE_DETAIL_COMPILER_FLATTEN
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  define HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens) _Pragma(#tokens)
#  define HYLE_DETAIL_COMPILER_PRAGMA(tokens) HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens)
#  define HYLE_DETAIL_COMPILER_UNROLL(factor) \
    HYLE_DETAIL_COMPILER_PRAGMA(GCC unroll factor)
#  define HYLE_DETAIL_COMPILER_DEFAULT_UNROLL HYLE_DETAIL_COMPILER_UNROLL(4)
#elif defined(__clang__)
#  if defined(__OPTIMIZE__) && !__has_feature(address_sanitizer) &&           \
      !__has_feature(thread_sanitizer) && !__has_feature(memory_sanitizer) && \
      !__has_feature(undefined_behavior_sanitizer)
#  define HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens) _Pragma(#tokens)
#  define HYLE_DETAIL_COMPILER_PRAGMA(tokens) HYLE_DETAIL_COMPILER_PRAGMA_INNER(tokens)
#  define HYLE_DETAIL_COMPILER_UNROLL(factor) HYLE_DETAIL_COMPILER_PRAGMA(unroll factor)
#  define HYLE_DETAIL_COMPILER_DEFAULT_UNROLL HYLE_DETAIL_COMPILER_PRAGMA(unroll)
#  else
/* An explicit Clang loop-transform pragma that cannot be honored emits
 * -Wpass-failed. That turns a best-effort hint into a consumer build failure
 * under -Werror, notably at -O0 and under sanitizer instrumentation. Keep the
 * source contract a no-op in those configurations. */
#  define HYLE_DETAIL_COMPILER_UNROLL(factor)
#  define HYLE_DETAIL_COMPILER_DEFAULT_UNROLL
#  endif
#else
#  define HYLE_DETAIL_COMPILER_UNROLL(factor)
#  define HYLE_DETAIL_COMPILER_DEFAULT_UNROLL
#endif

#if defined(_MSC_VER)
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) __assume(!!(cond))
#elif defined(__clang__)
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) __builtin_assume(!!(cond))
#elif defined(__GNUC__)
#  define HYLE_DETAIL_COMPILER_ASSUME(cond)  \
    do {                                     \
      if (!(cond)) __builtin_unreachable(); \
    } while (0)
#else
#  define HYLE_DETAIL_COMPILER_ASSUME(cond) ((void)sizeof(!!(cond)))
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_LIKELY(cond) (__builtin_expect(!!(cond), 1))
#  define HYLE_DETAIL_COMPILER_UNLIKELY(cond) (__builtin_expect(!!(cond), 0))
#else
#  define HYLE_DETAIL_COMPILER_LIKELY(cond) (!!(cond))
#  define HYLE_DETAIL_COMPILER_UNLIKELY(cond) (!!(cond))
#endif

#if defined(_MSC_VER)
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() __builtin_unreachable()
#else
#  define HYLE_DETAIL_COMPILER_UNREACHABLE() ((void)0)
#endif

#endif /* HYLE_CORE_COMPILER_DEFS_H_ */
