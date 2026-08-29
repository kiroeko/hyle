#pragma once

#include <cstddef>

#include "hwy/highway.h"

#define HYLE_SAMPLE_FAT_CAT_I(a, b) a##b
#define HYLE_SAMPLE_FAT_CAT(a, b) HYLE_SAMPLE_FAT_CAT_I(a, b)

void HYLE_SAMPLE_FAT_CAT(hyle_sample_bias_relu_, HYLE_SIMD_VARIANT_SUFFIX)(
    float* values, std::size_t count, float bias) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const auto vbias = hn::Set(d, bias);
  const auto zero = hn::Zero(d);
  const std::size_t lanes = hn::Lanes(d);

  std::size_t i = 0;
  for (; i + lanes <= count; i += lanes) {
    const auto value = hn::LoadU(d, values + i);
    hn::StoreU(hn::Max(hn::Add(value, vbias), zero), d, values + i);
  }
  for (; i < count; ++i) {
    const float shifted = values[i] + bias;
    values[i] = shifted > 0.0f ? shifted : 0.0f;
  }
}

#undef HYLE_SAMPLE_FAT_CAT
#undef HYLE_SAMPLE_FAT_CAT_I
