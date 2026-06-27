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
