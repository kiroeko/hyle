/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <cstddef>

#include "hyle/core/compiler.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace hyle::core {

// Compile-time target-family layout recommendations. Use
// platform::cache_line_size() when an algorithm benefits from a best-effort
// runtime observation; it can fall back to 64 and is not a safe replacement
// for destructive-interference layout constants.
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
inline constexpr std::size_t kConstructiveInterferenceSize = 128;
inline constexpr std::size_t kDestructiveInterferenceSize = 128;
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
inline constexpr std::size_t kConstructiveInterferenceSize = 64;
inline constexpr std::size_t kDestructiveInterferenceSize = 64;
#elif defined(__x86_64__) || defined(_M_X64)
inline constexpr std::size_t kConstructiveInterferenceSize = 64;
inline constexpr std::size_t kDestructiveInterferenceSize = 64;
#else
inline constexpr std::size_t kConstructiveInterferenceSize = 64;
inline constexpr std::size_t kDestructiveInterferenceSize = 128;
#endif

// Cross-family conservative spacing for sparse, independently-written control
// words. Dense data structures normally use kDestructiveInterferenceSize to
// avoid paying 128-byte stride on x86.
inline constexpr std::size_t kMaxDestructiveInterferenceSize = 128;

// Best-effort read prefetch with high temporal locality. It is a performance
// hint only: correctness must never depend on the address being fetched.
HYLE_FORCE_INLINE void prefetch_read(const void* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 3);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x0); // PLD L1 KEEP
#else
  (void)address;
#endif
}

// Best-effort one-shot read prefetch. Prefer this when the line is unlikely to
// be reused; unlike prefetch_read(), capable targets may avoid retaining it in
// outer cache levels.
HYLE_FORCE_INLINE void
prefetch_read_non_temporal(const void* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 0);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch(static_cast<const char*>(address), _MM_HINT_NTA);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x1); // PLD L1 STRM
#else
  (void)address;
#endif
}

// Best-effort prefetch with intent to modify. This can request exclusive cache
// ownership and therefore should be used more selectively than read prefetch.
HYLE_FORCE_INLINE void prefetch_write(const void* address) noexcept {
#if defined(_MSC_VER) && !defined(__clang__) && \
    (defined(_M_IX86) || defined(_M_X64))
  // Modern MSVC exposes PREFETCHW directly. HYLE's minimum toolchain is newer
  // than the historical MSVC versions for which peer libraries disabled this
  // intrinsic.
  _m_prefetchw(address);
#elif (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(_M_X64)) && !defined(__PRFCHW__)
  // GCC/Clang otherwise lower write=1 to PREFETCHT0 on baseline x86-64.
  // PREFETCHW is a recognized no-op on older Intel CPUs and has existed on AMD
  // since K6-2, so this is safe for the distributable x86-64 baseline. This is
  // the same compatibility route used by Abseil's write-prefetch primitive.
  asm("prefetchw %0" : : "m"(*reinterpret_cast<const char*>(address)));
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 1, 3);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x10); // PST L1 KEEP
#else
  (void)address;
#endif
}

}  // namespace hyle::core
