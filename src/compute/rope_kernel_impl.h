#pragma once

#include <cmath>
#include <cstddef>

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include "rope_kernels.h"

#define HYLE_COMPUTE_ROPE_CAT_I(left, right) left##right
#define HYLE_COMPUTE_ROPE_CAT(left, right) HYLE_COMPUTE_ROPE_CAT_I(left, right)

namespace hyle::compute::detail {
namespace {

template <bool Interleaved>
HWY_INLINE void run_rope_row(const float *input, const float *frequencies,
                             float *output, std::size_t pairs,
                             float position) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = decltype(hn::Zero(d));
  const std::size_t lanes = hn::Lanes(d);
  const V position_v = hn::Set(d, position);
  std::size_t pair = 0;

  for (; pair + lanes <= pairs; pair += lanes) {
    V left;
    V right;
    if constexpr (Interleaved) {
      hn::LoadInterleaved2(d, input + pair * 2, left, right);
    } else {
      left = hn::LoadU(d, input + pair);
      right = hn::LoadU(d, input + pairs + pair);
    }

    const V angles = hn::Mul(position_v, hn::LoadU(d, frequencies + pair));
    V sine;
    V cosine;
    hn::SinCos(d, angles, sine, cosine);
    const V rotated_left =
        hn::Sub(hn::Mul(left, cosine), hn::Mul(right, sine));
    const V rotated_right =
        hn::Add(hn::Mul(left, sine), hn::Mul(right, cosine));
    if constexpr (Interleaved) {
      hn::StoreInterleaved2(rotated_left, rotated_right, d,
                            output + pair * 2);
    } else {
      hn::StoreU(rotated_left, d, output + pair);
      hn::StoreU(rotated_right, d, output + pairs + pair);
    }
  }

  for (; pair < pairs; ++pair) {
    const float angle = position * frequencies[pair];
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const std::size_t first = Interleaved ? pair * 2 : pair;
    const std::size_t second = Interleaved ? first + 1 : pairs + pair;
    const float left = input[first];
    const float right = input[second];
    output[first] = left * cosine - right * sine;
    output[second] = left * sine + right * cosine;
  }
}

} // namespace

void HYLE_COMPUTE_ROPE_CAT(run_rope_interleaved_,
                           HYLE_SIMD_VARIANT_SUFFIX)(
    const float *input, const float *frequencies, float *output,
    std::size_t pairs, float position) noexcept {
  run_rope_row<true>(input, frequencies, output, pairs, position);
}

void HYLE_COMPUTE_ROPE_CAT(run_rope_half_split_,
                           HYLE_SIMD_VARIANT_SUFFIX)(
    const float *input, const float *frequencies, float *output,
    std::size_t pairs, float position) noexcept {
  run_rope_row<false>(input, frequencies, output, pairs, position);
}

} // namespace hyle::compute::detail

#undef HYLE_COMPUTE_ROPE_CAT
#undef HYLE_COMPUTE_ROPE_CAT_I
