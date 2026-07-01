#include <cstdint>

#include "hyle/macros.h"
#include "hyle/platform.h"

// AVX2 (256-bit) variants of string_ref's count/rfind cores. Compiled in its OWN TU
// at -mavx2 / MSVC /arch:AVX2 (see src/container/CMakeLists.txt), so the 256-bit
// codegen never leaks into the baseline dispatcher (string_ref.cc). These are ENTERED
// only when platform::cpu_features().avx2 is true (gated in string_ref.cc), so an AVX2
// instruction never runs on a CPU that lacks it -- no SIGILL. They double the SSE2
// (128-bit) throughput to match MSVC STL's AVX2 memchr/count on long scans.
//
// x86-only: guarded by HYLE_ARCH_X64. On non-x86 this TU compiles to nothing and the
// dispatcher stays on its SSE2/scalar path.

#if defined(HYLE_ARCH_X64)
#  include <immintrin.h>  // AVX2

#  if defined(_MSC_VER)
#    include <intrin.h>
static HYLE_FORCE_INLINE unsigned hyle_clz32_avx2(std::uint32_t x) noexcept {
  unsigned long i;
  _BitScanReverse(&i, x);
  return static_cast<unsigned>(31 - i);
}
#  else
static HYLE_FORCE_INLINE unsigned hyle_clz32_avx2(std::uint32_t x) noexcept {
  return static_cast<unsigned>(__builtin_clz(x));
}
#  endif

namespace hyle::container::detail {

// Count occurrences of `c`, AVX2. Same shape as the SSE2 core but 32 bytes/iter:
// _mm256_cmpeq_epi8 marks matches (0xFF), _mm256_sad_epu8 vs zero horizontally sums
// each 8-byte lane (a 0xFF match contributes 255) into four 64-bit lanes; accumulate
// across groups, sum the four lanes, divide by 255. A <32 tail runs on SSE2/scalar.
std::uint32_t str_count_avx2(const char* data, std::uint32_t size, char c) noexcept {
  std::uint32_t i = 0;
  const __m256i needle = _mm256_set1_epi8(c);
  const __m256i zero = _mm256_setzero_si256();
  // FOUR independent accumulators (128 bytes/iter) so the per-iteration psadbw+add
  // chains are independent -- the single-accumulator loop serialized on `acc` and lost
  // std::count ~1.7x. Each sad lane sums <=8 matches*255=2040 per iter; 64-bit lanes
  // never overflow for any realistic size, so no periodic drain needed.
  __m256i a0 = _mm256_setzero_si256();
  __m256i a1 = _mm256_setzero_si256();
  __m256i a2 = _mm256_setzero_si256();
  __m256i a3 = _mm256_setzero_si256();
  for (; i + 128 <= size; i += 128) {
    const auto* p = reinterpret_cast<const __m256i*>(data + i);
    a0 = _mm256_add_epi64(
        a0, _mm256_sad_epu8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p + 0), needle), zero));
    a1 = _mm256_add_epi64(
        a1, _mm256_sad_epu8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p + 1), needle), zero));
    a2 = _mm256_add_epi64(
        a2, _mm256_sad_epu8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p + 2), needle), zero));
    a3 = _mm256_add_epi64(
        a3, _mm256_sad_epu8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p + 3), needle), zero));
  }
  __m256i acc = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
  for (; i + 32 <= size; i += 32) {  // 32B remainder before the scalar tail
    __m256i eq = _mm256_cmpeq_epi8(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i)), needle);
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(eq, zero));
  }
  // Horizontal add of the four 64-bit lanes.
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i s = _mm_add_epi64(lo, hi);
  std::uint64_t byte_sum =
      static_cast<std::uint64_t>(_mm_cvtsi128_si64(s)) +
      static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(s, s)));
  std::uint32_t n = static_cast<std::uint32_t>(byte_sum / 255u);
  for (; i < size; ++i) n += (data[i] == c);  // scalar tail (< 32 bytes)
  return n;
}

// Index of the last `c` in data[0,size), AVX2, or 0xFFFFFFFF if absent. Backward
// 32-byte blocks from the high end; _mm256_cmpeq_epi8 + _mm256_movemask_epi8 gives a
// 32-bit per-lane mask, the HIGHEST set bit (31 - clz) is the rightmost match. A <32
// head runs scalar.
std::uint32_t str_rfind_char_avx2(const char* data, std::uint32_t size,
                                  char c) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  const __m256i needle = _mm256_set1_epi8(c);
  std::uint32_t i = size;
  while (i >= 32) {
    i -= 32;
    __m256i chunk =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
    std::uint32_t m = static_cast<std::uint32_t>(
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle)));
    if (m != 0) return i + (31u - hyle_clz32_avx2(m));  // highest set lane
  }
  for (std::uint32_t j = i; j-- > 0;)  // scalar head (< 32 bytes)
    if (data[j] == c) return j;
  return kNpos;
}

}  // namespace hyle::container::detail

#endif  // HYLE_ARCH_X64
