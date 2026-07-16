#pragma once

// Platform / arch / compiler detection. Every branch has a fallback so unknown
// platforms degrade safe rather than fail to compile.

// ---- OS ----
#if defined(_WIN32)
#  define HYLE_OS_WINDOWS 1
#elif defined(__APPLE__)
#  define HYLE_OS_MACOS 1
#elif defined(__linux__)
#  define HYLE_OS_LINUX 1
#endif

// ---- arch ----
#if defined(__x86_64__) || defined(_M_X64)
#  define HYLE_ARCH_X64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define HYLE_ARCH_ARM64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#  define HYLE_ARCH_RISCV64 1
#endif

// ---- compiler ----
#if defined(_MSC_VER) && !defined(__clang__)
#  define HYLE_COMPILER_MSVC 1
#elif defined(__clang__)
#  define HYLE_COMPILER_CLANG 1
#elif defined(__GNUC__)
#  define HYLE_COMPILER_GCC 1
#endif

// ---- pointer width ----
#if INTPTR_MAX == INT64_MAX
#  define HYLE_BITNESS 64
#elif INTPTR_MAX == INT32_MAX
#  define HYLE_BITNESS 32
#endif

// ---- x86 SSSE3 availability (normalized across compilers) ----
// SSSE3 (the `_mm_sign_epi8` 1-insn SwissTable match_empty trick) is part of the
// x86-64-v2 baseline (universal since ~2009). The compilers expose it differently,
// so we normalize to ONE HYLE macro instead of probing the raw `__SSSE3__` (which is
// GCC/Clang-only -- MSVC never defines it, which would silently disable the fast path
// on Windows forever). Use HYLE_HAVE_SSSE3 everywhere, never `__SSSE3__` directly.
//   - MSVC x64: all SSE..SSE4 intrinsics are available UNCONDITIONALLY (no per-level
//     flag), so SSSE3 intrinsics always compile -> define it on x64.
//   - GCC/Clang: gate on their __SSSE3__ (set by -mssse3 / -march >= x86-64-v2). The
//     build gives the relevant TUs that baseline (see src/container/CMakeLists.txt).
#if defined(HYLE_ARCH_X64)
#  if defined(HYLE_COMPILER_MSVC)
#    define HYLE_HAVE_SSSE3 1
#  elif defined(__SSSE3__)
#    define HYLE_HAVE_SSSE3 1
#  endif
#endif
