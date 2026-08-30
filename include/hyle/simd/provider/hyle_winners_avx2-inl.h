/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

// Finite HYLE AVX2 winners retained by simd-v3.  This is deliberately not a
// portable register facade: every spelling below is one qualified
// type/ISA/compiler exception from the generated route ledger and uses the
// locked Highway target's native vector/mask representation in-place.

#if !defined(HYLE_SIMD_TARGET_AVX2)
#error "hyle_winners_avx2-inl.h requires HYLE_SIMD_TARGET_AVX2"
#endif

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "hyle/core/assert.h"
#include "hyle/core/compiler.h"
#include "hwy/highway.h"

namespace hyle::simd::provider::hyle_winner::avx2 {

namespace hn = ::hwy::HWY_NAMESPACE;

using F32 = hn::Vec256<float>;
using I16 = hn::Vec256<std::int16_t>;
using I32 = hn::Vec256<std::int32_t>;
using U8 = hn::Vec256<std::uint8_t>;
using U16 = hn::Vec256<std::uint16_t>;
using U32 = hn::Vec256<std::uint32_t>;
using U64 = hn::Vec256<std::uint64_t>;
using MF32 = hn::Mask256<float>;
using MI32 = hn::Mask256<std::int32_t>;

HYLE_FORCE_INLINE I32 convert_to_i32(F32 v) noexcept {
  return I32{_mm256_cvttps_epi32(v.raw)};
}

HYLE_FORCE_INLINE I32 round_to_i32(F32 v) noexcept {
  return I32{_mm256_cvtps_epi32(v.raw)};
}

HYLE_FORCE_INLINE I32 ceil_to_i32(F32 v) noexcept {
  return I32{_mm256_cvttps_epi32(_mm256_ceil_ps(v.raw))};
}

HYLE_FORCE_INLINE I32 floor_to_i32(F32 v) noexcept {
  return I32{_mm256_cvttps_epi32(_mm256_floor_ps(v.raw))};
}

HYLE_FORCE_INLINE I16 demote_i32_to_i16(I32 a, I32 b) noexcept {
  const __m256i packed = _mm256_packs_epi32(a.raw, b.raw);
  return I16{
      _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0))};
}

HYLE_FORCE_INLINE I32 masked_convert_to_i32(MF32 mask, F32 v) noexcept {
  return I32{_mm256_and_si256(_mm256_cvttps_epi32(v.raw),
                              _mm256_castps_si256(mask.raw))};
}

HYLE_FORCE_INLINE F32 if_negative_then_else_zero(F32 sign, F32 yes) noexcept {
  const __m256 negative =
      _mm256_cmp_ps(sign.raw, _mm256_setzero_ps(), _CMP_LT_OQ);
  return F32{_mm256_and_ps(negative, yes.raw)};
}

HYLE_FORCE_INLINE F32 if_negative_then_zero_else(F32 sign, F32 no) noexcept {
  const __m256 negative =
      _mm256_cmp_ps(sign.raw, _mm256_setzero_ps(), _CMP_LT_OQ);
  return F32{_mm256_andnot_ps(negative, no.raw)};
}

HYLE_FORCE_INLINE F32 mul_complex(F32 a, F32 b) noexcept {
  const __m256 ar = _mm256_moveldup_ps(a.raw);
  const __m256 ai = _mm256_movehdup_ps(a.raw);
  const __m256 br = _mm256_moveldup_ps(b.raw);
  const __m256 bi = _mm256_movehdup_ps(b.raw);
  const __m256 real =
      _mm256_sub_ps(_mm256_mul_ps(ar, br), _mm256_mul_ps(ai, bi));
  const __m256 imag =
      _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));
  return F32{_mm256_blend_ps(real, imag, 0xAA)};
}

HYLE_FORCE_INLINE F32 mul_complex_conj(F32 a, F32 b) noexcept {
  const __m256 ar = _mm256_moveldup_ps(a.raw);
  const __m256 ai = _mm256_movehdup_ps(a.raw);
  const __m256 br = _mm256_moveldup_ps(b.raw);
  const __m256 bi = _mm256_movehdup_ps(b.raw);
  const __m256 real =
      _mm256_fmadd_ps(ar, br, _mm256_mul_ps(ai, bi));
  const __m256 imag =
      _mm256_sub_ps(_mm256_mul_ps(ai, br), _mm256_mul_ps(ar, bi));
  return F32{_mm256_blend_ps(real, imag, 0xAA)};
}

HYLE_FORCE_INLINE F32 masked_mul_complex(MF32 mask, F32 a, F32 b) noexcept {
  return F32{_mm256_and_ps(mask.raw, mul_complex(a, b).raw)};
}

HYLE_FORCE_INLINE F32 masked_mul_complex_conj(MF32 mask, F32 a,
                                               F32 b) noexcept {
  return F32{_mm256_and_ps(mask.raw, mul_complex_conj(a, b).raw)};
}

namespace detail {

HYLE_FORCE_INLINE __m128i partial_load_i32(const std::int32_t* p,
                                           int count) noexcept {
  switch (count) {
    case 0:
      return _mm_setzero_si128();
    case 1:
      return _mm_cvtsi32_si128(p[0]);
    case 2:
      return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    case 3:
      return _mm_insert_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)), p[2], 2);
    default:
      return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  }
}

HYLE_FORCE_INLINE void partial_store_i32(std::int32_t* p, __m128i v,
                                         int count) noexcept {
  switch (count) {
    case 0:
      return;
    case 1:
      p[0] = _mm_cvtsi128_si32(v);
      return;
    case 2:
      _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v);
      return;
    case 3:
      _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v);
      p[2] = _mm_extract_epi32(v, 2);
      return;
    default:
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
      return;
  }
}

HYLE_FORCE_INLINE __m256i shift_var_u16(__m256i values, __m256i amounts,
                                        bool left) noexcept {
  const __m128i value_lo = _mm256_castsi256_si128(values);
  const __m128i value_hi = _mm256_extracti128_si256(values, 1);
  const __m128i amount_lo = _mm256_castsi256_si128(amounts);
  const __m128i amount_hi = _mm256_extracti128_si256(amounts, 1);
  const __m256i a = _mm256_cvtepu16_epi32(value_lo);
  const __m256i b = _mm256_cvtepu16_epi32(value_hi);
  const __m256i sa = _mm256_cvtepu16_epi32(amount_lo);
  const __m256i sb = _mm256_cvtepu16_epi32(amount_hi);
  __m256i ra = left ? _mm256_sllv_epi32(a, sa) : _mm256_srlv_epi32(a, sa);
  __m256i rb = left ? _mm256_sllv_epi32(b, sb) : _mm256_srlv_epi32(b, sb);
  if (left) {
    const __m256i low16 = _mm256_set1_epi32(0xFFFF);
    ra = _mm256_and_si256(ra, low16);
    rb = _mm256_and_si256(rb, low16);
  }
  return _mm256_permute4x64_epi64(_mm256_packus_epi32(ra, rb),
                                  _MM_SHUFFLE(3, 1, 2, 0));
}

HYLE_FORCE_INLINE __m256i shift_var_u8(__m256i values, __m256i amounts,
                                       bool left) noexcept {
  const __m256i even_mask = _mm256_set1_epi16(0x00FF);
  const __m256i odd_mask = _mm256_set1_epi16(static_cast<short>(0xFF00));
  const __m256i even_values = _mm256_and_si256(values, even_mask);
  const __m256i odd_values = _mm256_and_si256(values, odd_mask);
  const __m256i even_amounts = _mm256_and_si256(amounts, even_mask);
  const __m256i odd_amounts = _mm256_srli_epi16(amounts, 8);
  const __m256i even = shift_var_u16(even_values, even_amounts, left);
  const __m256i odd = shift_var_u16(odd_values, odd_amounts, left);
  return _mm256_or_si256(_mm256_and_si256(even, even_mask),
                         _mm256_and_si256(odd, odd_mask));
}

HYLE_FORCE_INLINE __m256i mul_high_i32(__m256i a, __m256i b) noexcept {
  const __m256i even = _mm256_mul_epi32(a, b);
  const __m256i odd = _mm256_mul_epi32(
      _mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1)),
      _mm256_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 1)));
  const __m256i packed = _mm256_castps_si256(_mm256_shuffle_ps(
      _mm256_castsi256_ps(even), _mm256_castsi256_ps(odd),
      _MM_SHUFFLE(3, 1, 3, 1)));
  return _mm256_shuffle_epi32(packed, _MM_SHUFFLE(3, 1, 2, 0));
}

}  // namespace detail

HYLE_FORCE_INLINE I32 load_n_i32(const std::int32_t* p,
                                 std::size_t count) noexcept {
  const int n = static_cast<int>(count < 8 ? count : 8);
  __m128i lo;
  __m128i hi;
  if (n > 4) {
    lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    hi = detail::partial_load_i32(p + 4, n - 4);
  } else {
    lo = detail::partial_load_i32(p, n);
    hi = _mm_setzero_si128();
  }
  return I32{_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)};
}

HYLE_FORCE_INLINE void store_n_i32(I32 value, std::int32_t* p,
                                   std::size_t count) noexcept {
  const int n = static_cast<int>(count < 8 ? count : 8);
  const __m128i lo = _mm256_castsi256_si128(value.raw);
  if (n > 4) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), lo);
    detail::partial_store_i32(p + 4, _mm256_extracti128_si256(value.raw, 1),
                              n - 4);
  } else {
    detail::partial_store_i32(p, lo, n);
  }
}

HYLE_FORCE_INLINE I32 expand_i32(I32 packed, MI32 mask) noexcept {
  alignas(32) std::int32_t source[8];
  alignas(32) std::int32_t output[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(source), packed.raw);
  const unsigned bits = static_cast<unsigned>(
      _mm256_movemask_ps(_mm256_castsi256_ps(mask.raw)));
  std::size_t next = 0;
  for (std::size_t lane = 0; lane < 8; ++lane) {
    output[lane] = ((bits >> lane) & 1u) != 0 ? source[next++] : 0;
  }
  return I32{_mm256_load_si256(reinterpret_cast<const __m256i*>(output))};
}

HYLE_FORCE_INLINE I32 load_expand_i32(MI32 mask,
                                      const std::int32_t* packed) noexcept {
  return expand_i32(
      I32{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed))}, mask);
}

HYLE_FORCE_INLINE U32 population_count_u32(U32 value) noexcept {
  const __m256i lut = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i low = _mm256_and_si256(value.raw, low_mask);
  const __m256i high =
      _mm256_and_si256(_mm256_srli_epi16(value.raw, 4), low_mask);
  const __m256i bytes = _mm256_add_epi8(_mm256_shuffle_epi8(lut, low),
                                        _mm256_shuffle_epi8(lut, high));
  const __m256i pairs =
      _mm256_maddubs_epi16(bytes, _mm256_set1_epi8(1));
  return U32{_mm256_madd_epi16(pairs, _mm256_set1_epi16(1))};
}

HYLE_FORCE_INLINE I32 reverse_bits_i32(I32 value) noexcept {
  const __m256i lut = _mm256_setr_epi8(
      0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
      0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
      0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
      0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF);
  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i low = _mm256_and_si256(value.raw, low_mask);
  const __m256i high =
      _mm256_and_si256(_mm256_srli_epi16(value.raw, 4), low_mask);
  const __m256i reversed_bytes = _mm256_or_si256(
      _mm256_slli_epi16(_mm256_shuffle_epi8(lut, low), 4),
      _mm256_shuffle_epi8(lut, high));
  const __m256i indices = _mm256_setr_epi8(
      3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
      3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
  return I32{_mm256_shuffle_epi8(reversed_bytes, indices)};
}

HYLE_FORCE_INLINE I32 rounding_shr_i32(I32 value, I32 amounts) noexcept {
  const __m256i shifted_one =
      _mm256_sllv_epi32(_mm256_set1_epi32(1), amounts.raw);
  const __m256i bias = _mm256_srli_epi32(shifted_one, 1);
  return I32{_mm256_srav_epi32(_mm256_add_epi32(value.raw, bias),
                               amounts.raw)};
}

HYLE_FORCE_INLINE I32 pairwise_sub128_i32(I32 a, I32 b) noexcept {
  return I32{_mm256_hsub_epi32(a.raw, b.raw)};
}

HYLE_FORCE_INLINE I32 pairwise_sub_i32(I32 a, I32 b) noexcept {
  const __m256 av = _mm256_castsi256_ps(a.raw);
  const __m256 bv = _mm256_castsi256_ps(b.raw);
  const __m256i even = _mm256_castps_si256(
      _mm256_shuffle_ps(av, bv, _MM_SHUFFLE(2, 0, 2, 0)));
  const __m256i odd = _mm256_castps_si256(
      _mm256_shuffle_ps(av, bv, _MM_SHUFFLE(3, 1, 3, 1)));
  return I32{_mm256_permute4x64_epi64(_mm256_sub_epi32(even, odd),
                                      _MM_SHUFFLE(3, 1, 2, 0))};
}

HYLE_FORCE_INLINE void store_interleaved4_f32(F32 v0, F32 v1, F32 v2,
                                               F32 v3, float* out) noexcept {
  const __m256 v01_lo = _mm256_unpacklo_ps(v0.raw, v1.raw);
  const __m256 v01_hi = _mm256_unpackhi_ps(v0.raw, v1.raw);
  const __m256 v23_lo = _mm256_unpacklo_ps(v2.raw, v3.raw);
  const __m256 v23_hi = _mm256_unpackhi_ps(v2.raw, v3.raw);
  const __m256 a = _mm256_shuffle_ps(v01_lo, v23_lo, 0x44);
  const __m256 b = _mm256_shuffle_ps(v01_lo, v23_lo, 0xEE);
  const __m256 c = _mm256_shuffle_ps(v01_hi, v23_hi, 0x44);
  const __m256 d = _mm256_shuffle_ps(v01_hi, v23_hi, 0xEE);
  _mm256_storeu_ps(out + 0, _mm256_permute2f128_ps(a, b, 0x20));
  _mm256_storeu_ps(out + 8, _mm256_permute2f128_ps(c, d, 0x20));
  _mm256_storeu_ps(out + 16, _mm256_permute2f128_ps(a, b, 0x31));
  _mm256_storeu_ps(out + 24, _mm256_permute2f128_ps(c, d, 0x31));
}

template <int kI3, int kI2, int kI1, int kI0>
HYLE_FORCE_INLINE U16 sums_of_shuffled_quad_abs_diff(U8 a, U8 b) noexcept {
  static_assert(0 <= kI0 && kI0 <= 3 && 0 <= kI1 && kI1 <= 3 &&
                0 <= kI2 && kI2 <= 3 && 0 <= kI3 && kI3 <= 3);
  const __m256i shuffled =
      _mm256_shuffle_epi32(a.raw, _MM_SHUFFLE(kI3, kI2, kI1, kI0));
  const __m256i even_windows = _mm256_blend_epi32(
      shuffled, _mm256_slli_si256(shuffled, 2), 0xAA);
  const __m256i odd_windows = _mm256_blend_epi32(
      _mm256_srli_si256(shuffled, 1), _mm256_slli_si256(shuffled, 1),
      0xAA);
  const auto abs_diff = [](__m256i x, __m256i y) noexcept {
    return _mm256_sub_epi8(_mm256_max_epu8(x, y),
                           _mm256_min_epu8(x, y));
  };
  const __m256i ones8 = _mm256_set1_epi8(1);
  const __m256i ones16 = _mm256_set1_epi16(1);
  const auto sum4 = [&](const __m256i diff) noexcept {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(diff, ones8), ones16);
  };
  const __m256i even = sum4(abs_diff(even_windows, b.raw));
  const __m256i odd =
      _mm256_slli_epi32(sum4(abs_diff(odd_windows, b.raw)), 16);
  return U16{_mm256_blend_epi16(even, odd, 0xAA)};
}

HYLE_FORCE_INLINE U64 bit_shuffle_u64(U64 values, U8 indices) noexcept {
  const __m256i low3 = _mm256_set1_epi8(7);
  const __m256i index_div8 = _mm256_and_si256(
      _mm256_srli_epi16(indices.raw, 3), _mm256_set1_epi8(0x1F));
  const __m256i block_base = _mm256_set_epi64x(
      static_cast<long long>(0x0808080808080808ULL), 0,
      static_cast<long long>(0x0808080808080808ULL), 0);
  const __m256i byte_indices = _mm256_or_si256(
      _mm256_and_si256(index_div8, low3),
      _mm256_andnot_si256(low3, block_base));
  const __m256i iota = _mm256_set_epi8(
      7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0,
      7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
  const __m256i rotate = _mm256_sub_epi8(iota, indices.raw);
  const __m256i bytes = _mm256_shuffle_epi8(values.raw, byte_indices);
  const __m256i left = _mm256_and_si256(rotate, low3);
  const __m256i right = _mm256_and_si256(
      _mm256_sub_epi8(_mm256_setzero_si256(), rotate), low3);
  const __m256i rotated = _mm256_or_si256(
      detail::shift_var_u8(bytes, left, true),
      detail::shift_var_u8(bytes, right, false));
  const __m256i extracted = _mm256_and_si256(
      rotated,
      _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL)));
  return U64{_mm256_sad_epu8(extracted, _mm256_setzero_si256())};
}

class InvariantDivisorI32 final {
 public:
  explicit InvariantDivisorI32(std::int32_t divisor) noexcept
      : divisor_(divisor), negative_(divisor < 0) {
    HYLE_FATAL_ASSERT(divisor != 0);
    const std::uint32_t magnitude =
        divisor < 0 ? std::uint32_t{0} - static_cast<std::uint32_t>(divisor)
                    : static_cast<std::uint32_t>(divisor);
    power_of_two_ = std::has_single_bit(magnitude);
    if (power_of_two_) {
      shift_ = static_cast<int>(std::countr_zero(magnitude));
      multiplier_ = 1;
      return;
    }
    shift_ = static_cast<int>(31u - std::countl_zero(magnitude - 1u));
    const std::uint64_t multiplier =
        (std::uint64_t{0x100000000} << shift_) / magnitude + 1u;
    multiplier_ = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(multiplier));
  }

  [[nodiscard]] std::int32_t divisor() const noexcept { return divisor_; }

 private:
  friend HYLE_FORCE_INLINE I32 divide(I32, const InvariantDivisorI32&) noexcept;
  std::int32_t multiplier_ = 1;
  std::int32_t divisor_ = 1;
  int shift_ = 0;
  bool negative_ = false;
  bool power_of_two_ = true;
};

HYLE_FORCE_INLINE I32 divide(I32 dividend,
                             const InvariantDivisorI32& divisor) noexcept {
  if (divisor.power_of_two_) {
    if (divisor.shift_ == 0) {
      return divisor.negative_
                 ? I32{_mm256_sub_epi32(_mm256_setzero_si256(), dividend.raw)}
                 : dividend;
    }
    const std::int32_t mask = static_cast<std::int32_t>(
        (std::uint32_t{1} << static_cast<unsigned>(divisor.shift_)) - 1u);
    const __m256i bias = _mm256_and_si256(
        _mm256_srai_epi32(dividend.raw, 31), _mm256_set1_epi32(mask));
    __m256i quotient = _mm256_sra_epi32(
        _mm256_add_epi32(dividend.raw, bias),
        _mm_cvtsi32_si128(divisor.shift_));
    if (divisor.negative_) {
      quotient = _mm256_sub_epi32(_mm256_setzero_si256(), quotient);
    }
    return I32{quotient};
  }

  const __m256i multiplier = _mm256_set1_epi32(divisor.multiplier_);
  __m256i quotient = _mm256_add_epi32(
      dividend.raw, detail::mul_high_i32(dividend.raw, multiplier));
  quotient = _mm256_sra_epi32(quotient,
                              _mm_cvtsi32_si128(divisor.shift_));
  quotient = _mm256_sub_epi32(quotient,
                              _mm256_srai_epi32(dividend.raw, 31));
  if (divisor.negative_) {
    quotient = _mm256_sub_epi32(_mm256_setzero_si256(), quotient);
  }
  return I32{quotient};
}

HYLE_FORCE_INLINE I32 modulo(I32 dividend,
                             const InvariantDivisorI32& divisor) noexcept {
  const I32 quotient = divide(dividend, divisor);
  return I32{_mm256_sub_epi32(
      dividend.raw,
      _mm256_mullo_epi32(quotient.raw, _mm256_set1_epi32(divisor.divisor())))};
}

HYLE_FORCE_INLINE I32 masked_modulo_or(MI32 mask, I32 no, I32 dividend,
                                       const InvariantDivisorI32& divisor) noexcept {
  return I32{_mm256_blendv_epi8(no.raw, modulo(dividend, divisor).raw,
                                mask.raw)};
}

}  // namespace hyle::simd::provider::hyle_winner::avx2
