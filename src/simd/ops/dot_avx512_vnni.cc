#include "hyle/core/environment.h"

#if HYLE_ARCH_X64

#include <bit>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "src/simd/ops/kernels.h"

namespace hyle::simd::detail {
namespace {

inline __m512i signed_dot_quad(__m512i sum, __m512i a,
                               __m512i b) noexcept {
  const __m512i sign = _mm512_and_si512(
      _mm512_movm_epi8(_mm512_movepi8_mask(a)), _mm512_set1_epi8(1));
  const __m512i unsigned_sum = _mm512_dpbusd_epi32(sum, a, b);
  const __m512i correction = _mm512_slli_epi32(
      _mm512_dpbusd_epi32(_mm512_setzero_si512(), sign, b), 8);
  return _mm512_sub_epi32(unsigned_sum, correction);
}

inline void flush(__m512i& sum0, __m512i& sum1, __m512i& sum2,
                  __m512i& sum3, std::uint64_t& total) noexcept {
  alignas(64) std::int32_t lanes[16];
  const __m512i combined = _mm512_add_epi32(
      _mm512_add_epi32(sum0, sum1), _mm512_add_epi32(sum2, sum3));
  _mm512_store_si512(lanes, combined);
  for (std::int32_t lane : lanes)
    total += static_cast<std::uint64_t>(static_cast<std::int64_t>(lane));
  sum0 = sum1 = sum2 = sum3 = _mm512_setzero_si512();
}

}  // namespace

std::int64_t dot_i8_i8_avx512_vnni(const std::int8_t* x,
                                   const std::int8_t* y,
                                   std::size_t n) noexcept {
  constexpr std::size_t kWidth = 64;
  constexpr std::size_t kBlocksPerFlush = 1024;
  __m512i sum0 = _mm512_setzero_si512();
  __m512i sum1 = _mm512_setzero_si512();
  __m512i sum2 = _mm512_setzero_si512();
  __m512i sum3 = _mm512_setzero_si512();
  std::uint64_t total = 0;
  std::size_t blocks = 0;
  std::size_t i = 0;

  for (; i + 4 * kWidth <= n; i += 4 * kWidth) {
    sum0 = signed_dot_quad(
        sum0, _mm512_loadu_si512(x + i), _mm512_loadu_si512(y + i));
    sum1 = signed_dot_quad(sum1, _mm512_loadu_si512(x + i + kWidth),
                           _mm512_loadu_si512(y + i + kWidth));
    sum2 = signed_dot_quad(sum2, _mm512_loadu_si512(x + i + 2 * kWidth),
                           _mm512_loadu_si512(y + i + 2 * kWidth));
    sum3 = signed_dot_quad(sum3, _mm512_loadu_si512(x + i + 3 * kWidth),
                           _mm512_loadu_si512(y + i + 3 * kWidth));
    if (++blocks == kBlocksPerFlush) {
      flush(sum0, sum1, sum2, sum3, total);
      blocks = 0;
    }
  }
  for (; i + kWidth <= n; i += kWidth) {
    sum0 = signed_dot_quad(
        sum0, _mm512_loadu_si512(x + i), _mm512_loadu_si512(y + i));
  }
  flush(sum0, sum1, sum2, sum3, total);
  for (; i < n; ++i) {
    total += static_cast<std::uint64_t>(
        static_cast<std::int64_t>(x[i]) * static_cast<std::int64_t>(y[i]));
  }
  return std::bit_cast<std::int64_t>(total);
}

}  // namespace hyle::simd::detail

#endif
