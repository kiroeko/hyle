#include "hyle/core/environment.h"

#if HYLE_ARCH_X64

#include <bit>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "src/simd/ops/kernels.h"

namespace hyle::simd::detail {
namespace {

inline __m256i signed_dot_quad(__m256i sum, __m256i a,
                               __m256i b) noexcept {
  // AVX-VNNI provides u8*i8 VPDPBUSD, not signed*signed VPDPBSSD. Correct
  // a's signed interpretation in registers: a_signed = a_unsigned - 256*sign.
  const __m256i sign = _mm256_and_si256(
      _mm256_cmpgt_epi8(_mm256_setzero_si256(), a), _mm256_set1_epi8(1));
#if defined(_MSC_VER) && !defined(__clang__)
  const __m256i unsigned_sum = _mm256_dpbusd_avx_epi32(sum, a, b);
  const __m256i correction = _mm256_slli_epi32(
      _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), sign, b), 8);
#else
  const __m256i unsigned_sum = _mm256_dpbusd_epi32(sum, a, b);
  const __m256i correction = _mm256_slli_epi32(
      _mm256_dpbusd_epi32(_mm256_setzero_si256(), sign, b), 8);
#endif
  return _mm256_sub_epi32(unsigned_sum, correction);
}

inline void flush(__m256i& sum0, __m256i& sum1, __m256i& sum2,
                  __m256i& sum3, std::uint64_t& total) noexcept {
  alignas(32) std::int32_t lanes[8];
  const __m256i combined = _mm256_add_epi32(
      _mm256_add_epi32(sum0, sum1), _mm256_add_epi32(sum2, sum3));
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), combined);
  for (std::int32_t lane : lanes)
    total += static_cast<std::uint64_t>(static_cast<std::int64_t>(lane));
  sum0 = sum1 = sum2 = sum3 = _mm256_setzero_si256();
}

}  // namespace

// AVX-VNNI is orthogonal to the broad AVX2 tier. This TU is compiled with
// VEX-encoded VNNI enabled and is installed into the runtime table only after
// CPUID plus OS-YMM-state gating.
std::int64_t dot_i8_i8_avx_vnni(const std::int8_t* x,
                                const std::int8_t* y,
                                std::size_t n) noexcept {
  constexpr std::size_t kWidth = 32;
  constexpr std::size_t kBlocksPerFlush = 1024;
  __m256i sum0 = _mm256_setzero_si256();
  __m256i sum1 = _mm256_setzero_si256();
  __m256i sum2 = _mm256_setzero_si256();
  __m256i sum3 = _mm256_setzero_si256();
  std::uint64_t total = 0;
  std::size_t blocks = 0;
  std::size_t i = 0;

  for (; i + 4 * kWidth <= n; i += 4 * kWidth) {
    sum0 = signed_dot_quad(
        sum0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i)));
    sum1 = signed_dot_quad(
        sum1,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + kWidth)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i + kWidth)));
    sum2 = signed_dot_quad(
        sum2, _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(x + i + 2 * kWidth)),
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(y + i + 2 * kWidth)));
    sum3 = signed_dot_quad(
        sum3, _mm256_loadu_si256(
                  reinterpret_cast<const __m256i*>(x + i + 3 * kWidth)),
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(y + i + 3 * kWidth)));
    if (++blocks == kBlocksPerFlush) {
      flush(sum0, sum1, sum2, sum3, total);
      blocks = 0;
    }
  }
  for (; i + kWidth <= n; i += kWidth) {
    sum0 = signed_dot_quad(
        sum0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i)));
  }
  flush(sum0, sum1, sum2, sum3, total);
  for (; i < n; ++i) {
    total += static_cast<std::uint64_t>(
        static_cast<std::int64_t>(x[i]) * static_cast<std::int64_t>(y[i]));
  }
  return std::bit_cast<std::int64_t>(total);
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
