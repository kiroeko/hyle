#pragma once

#include <cstddef>
#include <cstdint>

// Core macro skeleton. Every attribute macro has an #else fallback so unknown
// compilers degrade to a safe empty/identity definition (never fail to compile).

// ---- force inline ----
#if defined(_MSC_VER)
#  define HYLE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_FORCE_INLINE __attribute__((always_inline)) inline
#else
#  define HYLE_FORCE_INLINE inline
#endif

// ---- no inline ----
#if defined(_MSC_VER)
#  define HYLE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_NOINLINE __attribute__((noinline))
#else
#  define HYLE_NOINLINE
#endif

// ---- branch prediction ----
#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_LIKELY(x)   (__builtin_expect(!!(x), 1))
#  define HYLE_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#  define HYLE_LIKELY(x)   (x)
#  define HYLE_UNLIKELY(x) (x)
#endif

// ---- unreachable ----
#if defined(_MSC_VER)
#  define HYLE_UNREACHABLE() __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_UNREACHABLE() __builtin_unreachable()
#else
#  define HYLE_UNREACHABLE() ((void)0)
#endif

// ---- cache line + alignment ----
#define HYLE_CACHE_LINE_SIZE 64
#define HYLE_ALIGNED alignas(HYLE_CACHE_LINE_SIZE)

// ---- explicit struct padding (paired with -Wpadded) ----
#define HYLE_PAD(n) char hyle_pad_##__LINE__[n]

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
      HYLE_TRAP();                                                           \
    }                                                                        \
  } while (0)
