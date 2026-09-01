/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <cstdint>

#if !defined(__cplusplus)
#  error "hyle/core/environment.h is a C++ header"
#endif

// Generic current-translation-unit feature probes. These always expand to a
// preprocessor integer expression and safely return zero when unsupported.
#if defined(__has_builtin)
#  define HYLE_HAS_BUILTIN(name) (0 != __has_builtin(name))
#else
#  define HYLE_HAS_BUILTIN(name) 0
#endif

#if defined(__has_attribute)
#  define HYLE_HAS_ATTRIBUTE(name) (0 != __has_attribute(name))
#else
#  define HYLE_HAS_ATTRIBUTE(name) 0
#endif

#if defined(__has_cpp_attribute)
#  define HYLE_HAS_CPP_ATTRIBUTE(name) (0 != __has_cpp_attribute(name))
#else
#  define HYLE_HAS_CPP_ATTRIBUTE(name) 0
#endif

#if defined(__has_feature)
#  define HYLE_HAS_FEATURE(name) (0 != __has_feature(name))
#else
#  define HYLE_HAS_FEATURE(name) 0
#endif

// OS facts describe the target of this translation unit, never the runtime
// host. Exactly one known OS is true on supported HYLE targets.
#if defined(_WIN32)
#  define HYLE_OS_WINDOWS 1
#  define HYLE_OS_LINUX 0
#  define HYLE_OS_MACOS 0
#  define HYLE_OS_UNKNOWN 0
#elif defined(__APPLE__)
#  define HYLE_OS_WINDOWS 0
#  define HYLE_OS_LINUX 0
#  define HYLE_OS_MACOS 1
#  define HYLE_OS_UNKNOWN 0
#elif defined(__linux__)
#  define HYLE_OS_WINDOWS 0
#  define HYLE_OS_LINUX 1
#  define HYLE_OS_MACOS 0
#  define HYLE_OS_UNKNOWN 0
#else
#  define HYLE_OS_WINDOWS 0
#  define HYLE_OS_LINUX 0
#  define HYLE_OS_MACOS 0
#  define HYLE_OS_UNKNOWN 1
#endif

// Architecture facts likewise describe code being compiled, not executable
// runtime ISA support. SIMD owns runtime target dispatch.
#if defined(__x86_64__) || defined(_M_X64)
#  define HYLE_ARCH_X64 1
#  define HYLE_ARCH_ARM64 0
#  define HYLE_ARCH_RISCV64 0
#  define HYLE_ARCH_UNKNOWN 0
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#  define HYLE_ARCH_X64 0
#  define HYLE_ARCH_ARM64 1
#  define HYLE_ARCH_RISCV64 0
#  define HYLE_ARCH_UNKNOWN 0
#elif defined(__riscv) && defined(__riscv_xlen) && (__riscv_xlen == 64)
#  define HYLE_ARCH_X64 0
#  define HYLE_ARCH_ARM64 0
#  define HYLE_ARCH_RISCV64 1
#  define HYLE_ARCH_UNKNOWN 0
#else
#  define HYLE_ARCH_X64 0
#  define HYLE_ARCH_ARM64 0
#  define HYLE_ARCH_RISCV64 0
#  define HYLE_ARCH_UNKNOWN 1
#endif

// Compiler identity is mutually exclusive. clang-cl is Clang with the MSVC
// ABI and receives its own secondary fact where spelling depends on that mode.
#if defined(__clang__)
#  define HYLE_COMPILER_MSVC 0
#  define HYLE_COMPILER_CLANG 1
#  if defined(_MSC_VER)
#    define HYLE_COMPILER_CLANG_CL 1
#  else
#    define HYLE_COMPILER_CLANG_CL 0
#  endif
#  define HYLE_COMPILER_GCC 0
#  define HYLE_COMPILER_UNKNOWN 0
#elif defined(_MSC_VER)
#  define HYLE_COMPILER_MSVC 1
#  define HYLE_COMPILER_CLANG 0
#  define HYLE_COMPILER_CLANG_CL 0
#  define HYLE_COMPILER_GCC 0
#  define HYLE_COMPILER_UNKNOWN 0
#elif defined(__GNUC__)
#  define HYLE_COMPILER_MSVC 0
#  define HYLE_COMPILER_CLANG 0
#  define HYLE_COMPILER_CLANG_CL 0
#  define HYLE_COMPILER_GCC 1
#  define HYLE_COMPILER_UNKNOWN 0
#else
#  define HYLE_COMPILER_MSVC 0
#  define HYLE_COMPILER_CLANG 0
#  define HYLE_COMPILER_CLANG_CL 0
#  define HYLE_COMPILER_GCC 0
#  define HYLE_COMPILER_UNKNOWN 1
#endif

// Pointer-width facts are independent of architecture names. HYLE_BITNESS is
// 0 on an unknown representation so unsupported targets fail closed instead
// of accidentally selecting a 32- or 64-bit implementation.
#if defined(UINTPTR_MAX) && defined(UINT64_MAX) && \
    (UINTPTR_MAX == UINT64_MAX)
#  define HYLE_POINTER_64 1
#  define HYLE_POINTER_32 0
#  define HYLE_BITNESS 64
#elif defined(UINTPTR_MAX) && defined(UINT32_MAX) && \
    (UINTPTR_MAX == UINT32_MAX)
#  define HYLE_POINTER_64 0
#  define HYLE_POINTER_32 1
#  define HYLE_BITNESS 32
#else
#  define HYLE_POINTER_64 0
#  define HYLE_POINTER_32 0
#  define HYLE_BITNESS 0
#endif

// Endianness is a compile-target property. Windows targets are little-endian;
// GCC/Clang-compatible targets expose the standard byte-order macros.
#if defined(_WIN32)
#  define HYLE_ENDIAN_LITTLE 1
#  define HYLE_ENDIAN_BIG 0
#  define HYLE_ENDIAN_UNKNOWN 0
#elif defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#  define HYLE_ENDIAN_LITTLE 1
#  define HYLE_ENDIAN_BIG 0
#  define HYLE_ENDIAN_UNKNOWN 0
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#  define HYLE_ENDIAN_LITTLE 0
#  define HYLE_ENDIAN_BIG 1
#  define HYLE_ENDIAN_UNKNOWN 0
#else
#  define HYLE_ENDIAN_LITTLE 0
#  define HYLE_ENDIAN_BIG 0
#  define HYLE_ENDIAN_UNKNOWN 1
#endif

// Build-mode facts follow the standard assertion contract. They do not claim
// whether optimization or debug information is enabled.
#if defined(NDEBUG)
#  define HYLE_BUILD_DEBUG 0
#  define HYLE_BUILD_RELEASE 1
#else
#  define HYLE_BUILD_DEBUG 1
#  define HYLE_BUILD_RELEASE 0
#endif

// CMake supplies HYLE_DETAIL_SANITIZER_* to HYLE-owned targets so all supported
// compiler frontends are exact. Compiler-native facts keep source consumers
// correct when HYLE is embedded in a separately instrumented parent.
#if HYLE_HAS_FEATURE(address_sanitizer) || \
    defined(__SANITIZE_ADDRESS__) || defined(HYLE_DETAIL_SANITIZER_ADDRESS)
#  define HYLE_SANITIZER_ADDRESS 1
#else
#  define HYLE_SANITIZER_ADDRESS 0
#endif

#if HYLE_HAS_FEATURE(thread_sanitizer) || \
    defined(__SANITIZE_THREAD__) || defined(HYLE_DETAIL_SANITIZER_THREAD)
#  define HYLE_SANITIZER_THREAD 1
#else
#  define HYLE_SANITIZER_THREAD 0
#endif

#if HYLE_HAS_FEATURE(memory_sanitizer) || \
    defined(__SANITIZE_MEMORY__) || defined(HYLE_DETAIL_SANITIZER_MEMORY)
#  define HYLE_SANITIZER_MEMORY 1
#else
#  define HYLE_SANITIZER_MEMORY 0
#endif

#if HYLE_HAS_FEATURE(undefined_behavior_sanitizer) || \
    defined(__SANITIZE_UNDEFINED__) || defined(HYLE_DETAIL_SANITIZER_UNDEFINED)
#  define HYLE_SANITIZER_UNDEFINED 1
#else
#  define HYLE_SANITIZER_UNDEFINED 0
#endif

#if HYLE_SANITIZER_ADDRESS || HYLE_SANITIZER_THREAD || \
    HYLE_SANITIZER_MEMORY || HYLE_SANITIZER_UNDEFINED
#  define HYLE_SANITIZER_ANY 1
#else
#  define HYLE_SANITIZER_ANY 0
#endif

#if defined(__clang_analyzer__) || defined(_PREFAST_)
#  define HYLE_STATIC_ANALYZER 1
#else
#  define HYLE_STATIC_ANALYZER 0
#endif

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || \
    defined(_CPPUNWIND)
#  define HYLE_HAS_EXCEPTIONS 1
#else
#  define HYLE_HAS_EXCEPTIONS 0
#endif

#if defined(__cpp_rtti) || defined(__GXX_RTTI) || defined(_CPPRTTI)
#  define HYLE_HAS_RTTI 1
#else
#  define HYLE_HAS_RTTI 0
#endif
