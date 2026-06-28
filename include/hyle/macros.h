#pragma once

#include <cstddef>
#include <cstdint>

// Core macro skeleton. Every attribute macro has an #else fallback so unknown
// compilers degrade to a safe empty/identity definition (never fail to compile).

// ---- token paste (two-level expansion so __COUNTER__/__LINE__ expand first) --
#define HYLE_CONCAT_INNER(a, b) a##b
#define HYLE_CONCAT(a, b) HYLE_CONCAT_INNER(a, b)

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

// ---- prefetch (read into L1/all caches, high temporal locality) ----
// Hides a load's latency by issuing it ahead of the dependent use. A pure hint:
// never faults (a bad address is dropped), so it's safe to prefetch a slot before
// confirming it's the match. Used in SwissTable::find to overlap the slot load
// with the control-byte SIMD match (abseil's find_large does the same).
#if defined(__GNUC__) || defined(__clang__)
#  define HYLE_PREFETCH(addr) __builtin_prefetch((addr), 0 /*read*/, 3 /*T0*/)
#elif defined(_MSC_VER)
#  define HYLE_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#  define HYLE_PREFETCH(addr) ((void)0)
#endif


// ---- cache line + alignment ----
#define HYLE_CACHE_LINE_SIZE 64
#define HYLE_ALIGNED alignas(HYLE_CACHE_LINE_SIZE)

// ---- explicit struct padding (paired with -Wpadded) ----
// HYLE_CONCAT (not ##) so __LINE__ expands to a number: two uses in one struct
// get distinct names (hyle_pad_42, hyle_pad_43), not a redefinition clash.
#define HYLE_PAD(n) char HYLE_CONCAT(hyle_pad_, __LINE__)[n]

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
