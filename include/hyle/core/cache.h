#pragma once

#include <cstddef>

#include "hyle/core/cache_defs.h"
#include "hyle/macros.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hyle::core {

// Compile-time layout bounds. Use platform::cache_line_size() instead when an
// algorithm needs the exact runtime value rather than a stable object ABI.
inline constexpr std::size_t kConstructiveInterferenceSize =
    HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE;
inline constexpr std::size_t kDestructiveInterferenceSize =
    HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE;

// Best-effort read prefetch with high temporal locality. It is a performance
// hint only: correctness must never depend on the address being fetched.
HYLE_FORCE_INLINE void prefetch_read(const void *address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 3);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch(static_cast<const char *>(address), _MM_HINT_T0);
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
prefetch_read_non_temporal(const void *address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(address, 0, 0);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_prefetch(static_cast<const char *>(address), _MM_HINT_NTA);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x1); // PLD L1 STRM
#else
  (void)address;
#endif
}

// Best-effort prefetch with intent to modify. This can request exclusive cache
// ownership and therefore should be used more selectively than read prefetch.
HYLE_FORCE_INLINE void prefetch_write(const void *address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  // Let the compiler choose an instruction valid for the translation unit's
  // target. On baseline x86 this may be a read-prefetch or no-op; spelling
  // PREFETCHW in inline assembly would bypass -march and could SIGILL on a
  // distributable binary running on an older CPU.
  __builtin_prefetch(address, 1, 3);
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#if defined(_MM_HINT_ET0)
  _mm_prefetch(static_cast<const char *>(address), _MM_HINT_ET0);
#else
  // Older MSVC headers cannot express exclusive intent; retain the useful
  // read hint instead of manufacturing a non-portable instruction sequence.
  _mm_prefetch(static_cast<const char *>(address), _MM_HINT_T0);
#endif
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __prefetch2(address, 0x10); // PST L1 KEEP
#else
  (void)address;
#endif
}

} // namespace hyle::core
