#include "hyle/container/string_ref.h"

#include "hyle/container/probe.h"  // hyle_clz32 (reverse-scan highest set lane)
#include "hyle/macros.h"
#include "hyle/platform.h"

#if defined(HYLE_ARCH_X64)
#  include <emmintrin.h>  // SSE2
#endif

// Out-of-line SIMD cores for string_ref's search family. Compiled ONCE here and
// exported (HYLE_CONTAINER_API) so the public header stays free of <emmintrin.h>
// and keeps container at Tier 0 (same self-contained SSE2 use as probe.h). The
// constexpr paths in the header remain scalar; only the runtime branch delegates
// here. The ~1ns call seam is negligible against the multi-KB scans these run.

namespace hyle::container::detail {

// Count occurrences of `c`. SSE2: _mm_cmpeq_epi8 marks matches as 0xFF bytes,
// then _mm_sad_epu8 vs zero horizontally sums each 8-byte half (a 0xFF match
// contributes 255) into two 64-bit lanes; accumulate across groups and divide
// the byte-sum by 255 at the end. This is the vectorization std::count uses --
// the scalar `n += (data[i]==c)` loop lost it ~25x. A short tail runs scalar.
std::uint32_t str_count(const char* data, std::uint32_t size, char c) noexcept {
#if defined(HYLE_ARCH_X64)
  std::uint32_t i = 0;
  __m128i acc = _mm_setzero_si128();
  const __m128i needle = _mm_set1_epi8(c);
  const __m128i zero = _mm_setzero_si128();
  for (; i + 16 <= size; i += 16) {
    __m128i chunk =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
    __m128i eq = _mm_cmpeq_epi8(chunk, needle);  // 0xFF per match, else 0x00
    acc = _mm_add_epi64(acc, _mm_sad_epu8(eq, zero));  // += 255*matches (2 lanes)
  }
  std::uint64_t byte_sum = static_cast<std::uint64_t>(_mm_cvtsi128_si64(acc)) +
                           static_cast<std::uint64_t>(
                               _mm_cvtsi128_si64(_mm_unpackhi_epi64(acc, acc)));
  std::uint32_t n = static_cast<std::uint32_t>(byte_sum / 255u);
  for (; i < size; ++i) n += (data[i] == c);  // scalar tail (< 16 bytes)
  return n;
#else
  std::uint32_t n = 0;
  for (std::uint32_t i = 0; i < size; ++i) n += (data[i] == c);
  return n;
#endif
}

// Index of the last `c` in data[0,size), or 0xFFFFFFFF if absent. SSE2 backward
// scan: process 16-byte blocks from the high end; in each, _mm_cmpeq_epi8 +
// movemask gives a per-lane match mask, and the HIGHEST set bit (31 - clz) is
// the rightmost match in the block. A sub-16 head runs scalar. No memrchr on
// Windows -- the scalar backward loop lost std::string_view::rfind ~20x.
std::uint32_t str_rfind_char(const char* data, std::uint32_t size,
                             char c) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
#if defined(HYLE_ARCH_X64)
  const __m128i needle = _mm_set1_epi8(c);
  std::uint32_t i = size;
  while (i >= 16) {
    i -= 16;
    __m128i chunk =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
    unsigned m = static_cast<unsigned>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, needle)));
    if (m != 0) return i + (31u - hyle_clz32(m));  // highest set lane
  }
  for (std::uint32_t j = i; j-- > 0;)  // scalar head (< 16 bytes)
    if (data[j] == c) return j;
  return kNpos;
#else
  for (std::uint32_t j = size; j-- > 0;)
    if (data[j] == c) return j;
  return kNpos;
#endif
}

}  // namespace hyle::container::detail
