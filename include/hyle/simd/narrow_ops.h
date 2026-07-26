#pragma once

#include <cstddef>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/dtype.h"

// Runtime-dispatched whole-range operations for f16/bf16 storage. The public
// boundary contains only pointers, counts and scalar bit-carriers; ISA-specific
// D-tags remain inside the selected kernel TU.

namespace hyle::simd {

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
