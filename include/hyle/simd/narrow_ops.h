#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/dtype.h"

// Runtime-dispatched whole-range operations for f16/bf16 storage. The public
// boundary contains only pointers, counts and scalar bit-carriers; ISA-specific
// D-tags remain inside the selected kernel TU.

namespace hyle::simd {

// Bijective IEEE-754 totalOrder key for HYLE's stable two-byte storage types.
// Unsigned key comparison distinguishes signed zero and every NaN encoding.
template <typename Narrow>
constexpr std::uint16_t narrow_total_order_key(Narrow value) noexcept
  requires(std::is_same_v<Narrow, float16_t> ||
           std::is_same_v<Narrow, bfloat16_t>)
{
  constexpr std::uint16_t kSign = 0x8000u;
  return static_cast<std::uint16_t>(
      value.bits ^ ((value.bits & kSign) != 0 ? 0xFFFFu : kSign));
}

template <typename Narrow>
constexpr Narrow narrow_from_total_order_key(std::uint16_t key) noexcept
  requires(std::is_same_v<Narrow, float16_t> ||
           std::is_same_v<Narrow, bfloat16_t>)
{
  constexpr std::uint16_t kSign = 0x8000u;
  const std::uint16_t bits = static_cast<std::uint16_t>(
      key ^ ((key & kSign) != 0 ? kSign : 0xFFFFu));
  return Narrow::from_bits(bits);
}

// IEEE numeric equality: NaN never matches and -0 compares equal to +0.
HYLE_SIMD_API std::size_t count_equal(const float16_t *data, std::size_t n,
                                      float16_t value) noexcept;
HYLE_SIMD_API std::size_t count_equal(const bfloat16_t *data, std::size_t n,
                                      bfloat16_t value) noexcept;
HYLE_SIMD_API std::size_t find_equal(const float16_t *data, std::size_t n,
                                     float16_t value) noexcept;
HYLE_SIMD_API std::size_t find_equal(const bfloat16_t *data, std::size_t n,
                                     bfloat16_t value) noexcept;

// Representation equality: signed zero and every NaN sign/payload remain
// distinct. find_equal_bits returns n when absent.
HYLE_SIMD_API std::size_t count_equal_bits(const float16_t *data, std::size_t n,
                                           float16_t value) noexcept;
HYLE_SIMD_API std::size_t count_equal_bits(const bfloat16_t *data,
                                           std::size_t n,
                                           bfloat16_t value) noexcept;
HYLE_SIMD_API std::size_t find_equal_bits(const float16_t *data, std::size_t n,
                                          float16_t value) noexcept;
HYLE_SIMD_API std::size_t find_equal_bits(const bfloat16_t *data, std::size_t n,
                                          bfloat16_t value) noexcept;

// Accumulates into f32. As with other SIMD reductions, association and the
// final rounded value may vary with the selected vector width.
HYLE_SIMD_API float sum(const float16_t *data, std::size_t n) noexcept;
HYLE_SIMD_API float sum(const bfloat16_t *data, std::size_t n) noexcept;

// IEEE-754 totalOrder over all 65,536 encodings, including signed zero and NaN
// sign/payload. descending reverses the complete order.
HYLE_SIMD_API bool is_sorted_total(const float16_t *data, std::size_t n,
                                   bool ascending = true) noexcept;
HYLE_SIMD_API bool is_sorted_total(const bfloat16_t *data, std::size_t n,
                                   bool ascending = true) noexcept;

} // namespace hyle::simd
