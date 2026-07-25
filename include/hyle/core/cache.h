#pragma once

#include <cstddef>

#include "hyle/core/cache_defs.h"

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#  include <intrin.h>
#endif

namespace hyle::core {

// Compile-time layout constants. Use platform::cache_line_size() instead when
// an algorithm needs the exact runtime value rather than a stable object ABI.
inline constexpr std::size_t kCacheLineSize = HYLE_STATIC_CACHE_LINE_SIZE;
inline constexpr std::size_t kConstructiveInterferenceSize = kCacheLineSize;
inline constexpr std::size_t kDestructiveInterferenceSize = kCacheLineSize;

// Best-effort read prefetch with high temporal locality. It is a performance
// hint only: correctness must never depend on the address being fetched.
inline void prefetch_read(const void* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 3);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch(address);
#else
  (void)address;
#endif
}

}  // namespace hyle::core
