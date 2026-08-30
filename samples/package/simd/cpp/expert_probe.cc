#include <cstddef>

#include "hwy/highway.h"
#include "xsimd/xsimd.hpp"

float hyle_package_expert_sum(const float* values, std::size_t count) noexcept {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  auto sum = hn::Zero(d);
  std::size_t index = 0;
  for (; index + lanes <= count; index += lanes) {
    sum = hn::Add(sum, hn::LoadU(d, values + index));
  }
  float result = hn::ReduceSum(d, sum);
  for (; index < count; ++index) result += values[index];
  return result;
}

std::size_t hyle_package_xsimd_width() noexcept {
  return xsimd::batch<float>::size;
}
