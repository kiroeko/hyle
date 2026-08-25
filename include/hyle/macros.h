#pragma once

#include "hyle/core/compiler_defs.h"

// Core macro skeleton. Every attribute macro has an #else fallback so unknown
// compilers degrade to a safe empty/identity definition (never fail to compile).

// ---- token paste (two-level expansion so __COUNTER__/__LINE__ expand first) --
#define HYLE_CONCAT_INNER(a, b) a##b
#define HYLE_CONCAT(a, b) HYLE_CONCAT_INNER(a, b)

// ---- force inline ----
#define HYLE_FORCE_INLINE HYLE_DETAIL_COMPILER_FORCE_INLINE

// ---- no inline ----
#define HYLE_NOINLINE HYLE_DETAIL_COMPILER_NOINLINE

// ---- must inspect the returned value --------------------------------------
// Put before a function declaration. This is a diagnostic contract, not
// runtime validation.
#define HYLE_NODISCARD HYLE_DETAIL_COMPILER_NODISCARD

// ---- aliasing contract ----------------------------------------------------
// Apply to pointer declarations only when the pointed-to ranges do not alias
// for the lifetime of the pointer. Violating this promise is undefined.
#define HYLE_RESTRICT HYLE_DETAIL_COMPILER_RESTRICT

// ---- recursive inlining hint ---------------------------------------------
// Apply to measured, deep wrapper/template entry points. Unlike FORCE_INLINE,
// this asks GCC/Clang to inline eligible calls throughout the function body.
#define HYLE_FLATTEN HYLE_DETAIL_COMPILER_FLATTEN

// ---- measured loop-unroll hints ------------------------------------------
// These are best-effort compiler hints, not semantic contracts. Keep them on
// profiled kernels: forcing unrolling can increase code size and hurt the I-cache.
#define HYLE_UNROLL(factor) HYLE_DETAIL_COMPILER_UNROLL(factor)
#define HYLE_DEFAULT_UNROLL HYLE_DETAIL_COMPILER_DEFAULT_UNROLL

// ---- empty-member optimization (zero-size stateless members) ----
// [[no_unique_address]] lets a zero-size member (e.g. a stateless SystemAllocator) occupy
// NO storage. MSVC SILENTLY IGNORES the standard spelling (ABI freeze) and requires its
// vendor attribute [[msvc::no_unique_address]] -- using the wrong one on MSVC wastes a byte
// + breaks the "stateless allocator is free" guarantee. This macro picks the right one.
#if defined(_MSC_VER)
#  define HYLE_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#  define HYLE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

// ---- branch prediction ----
#define HYLE_LIKELY(x) HYLE_DETAIL_COMPILER_LIKELY(x)
#define HYLE_UNLIKELY(x) HYLE_DETAIL_COMPILER_UNLIKELY(x)

// ---- delete copy / copy+move ----
#define HYLE_DISABLE_COPY(T)        \
  T(const T&) = delete;             \
  T& operator=(const T&) = delete

#define HYLE_DISABLE_COPY_AND_MOVE(T) \
  HYLE_DISABLE_COPY(T);               \
  T(T&&) = delete;                    \
  T& operator=(T&&) = delete

// ---- fail-fast (charter: bugs abort loudly, never caught) ----
// release-surviving fatal assert; evaluates cond, aborts with diagnostics on false.
#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_TRAP() __builtin_trap()
#else
#  define HYLE_TRAP() (::std::abort())
#endif

#include <cstdio>
#include <cstdlib>

#define HYLE_FATAL_ASSERT(cond)                                              \
  do {                                                                       \
    if (HYLE_UNLIKELY(!(cond))) {                                            \
      ::std::fprintf(stderr, "HYLE fatal: %s\n  at %s:%d\n", #cond,          \
                     __FILE__, __LINE__);                                    \
      ::std::abort();                                                        \
    }                                                                        \
  } while (0)

// ---- unreachable ----------------------------------------------------------
// Preserve the optimizer contract in Release, but make an accidental trip
// diagnosable in Debug instead of immediately entering undefined behavior.
#if !defined(NDEBUG)
#  define HYLE_UNREACHABLE() HYLE_FATAL_ASSERT(false)
#else
#  define HYLE_UNREACHABLE() HYLE_DETAIL_COMPILER_UNREACHABLE()
#endif

// ---- optimizer assumption -------------------------------------------------
// The expression must be side-effect free. Debug builds diagnose a violated
// assumption; optimized builds make it undefined so the compiler can remove
// impossible branches. Use HYLE_FATAL_ASSERT instead when validation must
// survive release builds.
#if !defined(NDEBUG)
#  define HYLE_ASSUME(cond) HYLE_FATAL_ASSERT(cond)
#else
#  define HYLE_ASSUME(cond) HYLE_DETAIL_COMPILER_ASSUME(cond)
#endif
