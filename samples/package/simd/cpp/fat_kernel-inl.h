#pragma once

#include <cstddef>

#include "hwy/highway.h"

#define HYLE_PACKAGE_FAT_CAT_I(a, b) a##b
#define HYLE_PACKAGE_FAT_CAT(a, b) HYLE_PACKAGE_FAT_CAT_I(a, b)

void HYLE_PACKAGE_FAT_CAT(hyle_package_bias_relu_,
                          HYLE_SIMD_VARIANT_SUFFIX)(
    float* values, std::size_t count, float bias) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const auto vbias = hn::Set(d, bias);
  const auto zero = hn::Zero(d);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t index = 0;
  for (; index + lanes <= count; index += lanes) {
    const auto value = hn::LoadU(d, values + index);
    hn::StoreU(hn::Max(hn::Add(value, vbias), zero), d, values + index);
  }
  for (; index < count; ++index) {
    const float shifted = values[index] + bias;
    values[index] = shifted > 0.0f ? shifted : 0.0f;
  }
}

#undef HYLE_PACKAGE_FAT_CAT
#undef HYLE_PACKAGE_FAT_CAT_I
