#include "hyle/container/string_ref.h"

#include <string_view>  // membership/substring family forwards to std's SOTA (see below)

#include "hyle/container/probe.h"  // hyle_clz32 (reverse-scan highest set lane)
#include "hyle/macros.h"
#include "hyle/platform.h"
#include "hyle/platform/cpu.h"  // cpu_features().avx2 -> opt into the AVX2 variant

#if defined(HYLE_ARCH_X64)
#  include <emmintrin.h>  // SSE2
#endif

// Out-of-line SIMD cores for string_ref's search family. Compiled ONCE here and
// exported (HYLE_CONTAINER_API) so the public header stays free of <emmintrin.h>
// AND of <string_view> -- string_ref.h pulls in no std string headers, this .cc does.
//
// SPLIT STRATEGY (measured, 2026-07): HYLE self-implements the ops where it matches or
// beats std (find(char)/rfind(char)/count via SSE2/AVX2, starts_with/split via memcmp),
// but for the ops where MSVC STL's hand-written vectorized assembly (vector_algorithms
// .cpp: two-way substring search, PCMPESTRI char-set scans) is genuinely ahead and hard
// to out-engineer, it FORWARDS to std::string_view here in the .cc. The forward is
// header-inline-free (stays out of the public header + DLL symbol table), so it neither
// leaks <string_view> to consumers nor breaks the C-ABI seam. Net: best-of-both -- our
// wins stay ours, std's wins are borrowed instead of lost. (clay forwarded EVERYTHING to
// std_view incl. <iostream> in its header; HYLE forwards only the ops std actually wins,
// and only from the .cc.)

namespace hyle::container::detail {

#if defined(HYLE_ARCH_X64)
// Defined in string_ref_avx2.cc (compiled at AVX2). Declared here so the baseline
// dispatcher can forward to them when the CPU supports AVX2.
std::uint32_t str_count_avx2(const char* data, std::uint32_t size, char c) noexcept;
std::uint32_t str_rfind_char_avx2(const char* data, std::uint32_t size,
                                  char c) noexcept;

// Cached once: does this CPU support the full AVX2 group? (Meyers singleton inside
// cpu_features(); the bool caches the single field read.)
static bool have_avx2() noexcept {
  static const bool v = ::hyle::platform::cpu_features().avx2;
  return v;
}
#endif

// Count occurrences of `c`. SSE2: _mm_cmpeq_epi8 marks matches as 0xFF bytes,
// then _mm_sad_epu8 vs zero horizontally sums each 8-byte half (a 0xFF match
// contributes 255) into two 64-bit lanes; accumulate across groups and divide
// the byte-sum by 255 at the end. This is the vectorization std::count uses --
// the scalar `n += (data[i]==c)` loop lost it ~25x. A short tail runs scalar.
// On an AVX2 CPU (and inputs long enough to amortize the call) forwards to the
// 256-bit variant instead.
std::uint32_t str_count(const char* data, std::uint32_t size, char c) noexcept {
#if defined(HYLE_ARCH_X64)
  if (size >= 32 && have_avx2()) return str_count_avx2(data, size, c);
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
// Windows -- the scalar backward loop lost std::string_view::rfind ~20x. Forwards
// to the 256-bit variant on an AVX2 CPU with a long-enough scan.
std::uint32_t str_rfind_char(const char* data, std::uint32_t size,
                             char c) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
#if defined(HYLE_ARCH_X64)
  if (size >= 32 && have_avx2()) return str_rfind_char_avx2(data, size, c);
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

// Forward char-set scan. `want`=true: first index IN set (find_first_of); false:
// first NOT in set (find_first_not_of). Forwards to std::string_view, whose MSVC
// vectorized find_first_of/not_of is the measured SOTA here (PCMPESTRI-class asm we
// couldn't out-engineer; our own PCMPESTRI attempt only reached ~0.78x). npos maps to
// our 0xFFFFFFFF.
std::uint32_t str_find_first_of(const char* data, std::uint32_t size,
                                const char* set, std::uint32_t set_size,
                                std::uint32_t pos, bool want) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  std::string_view hay(data, size);
  std::string_view s(set, set_size);
  std::size_t r = want ? hay.find_first_of(s, pos)
                       : hay.find_first_not_of(s, pos);
  return r == std::string_view::npos ? kNpos : static_cast<std::uint32_t>(r);
}

// Backward char-set scan. `pos` is the highest index to consider (inclusive).
// want=true => find_last_of; false => find_last_not_of. Forwards to std::string_view
// (same SOTA rationale as the forward variant).
std::uint32_t str_find_last_of(const char* data, std::uint32_t size,
                               const char* set, std::uint32_t set_size,
                               std::uint32_t pos, bool want) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  std::string_view hay(data, size);
  std::string_view s(set, set_size);
  std::size_t r = want ? hay.find_last_of(s, pos)
                       : hay.find_last_not_of(s, pos);
  return r == std::string_view::npos ? kNpos : static_cast<std::uint32_t>(r);
}

// Substring search, forwards to std::string_view (two-way vectorized SOTA). rev=false
// => find (first at/after pos); rev=true => rfind (last at/before pos).
std::uint32_t str_find_substr(const char* data, std::uint32_t size,
                              const char* needle, std::uint32_t needle_size,
                              std::uint32_t pos, bool rev) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  std::string_view hay(data, size);
  std::string_view n(needle, needle_size);
  std::size_t r = rev ? hay.rfind(n, pos) : hay.find(n, pos);
  return r == std::string_view::npos ? kNpos : static_cast<std::uint32_t>(r);
}

}  // namespace hyle::container::detail
