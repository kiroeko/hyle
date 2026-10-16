#pragma once

namespace hyle::compute::apple::detail {
// Fixed Random123 Philox4x32-10 equations. Copyright 2010-2012 D. E. Shaw
// Research, BSD-3-Clause; the shared schema installs the original license.
// A source-only numerical test compiles this same header to check endpoints.
inline constexpr auto kSamplingMetalHeader = R"METAL(
  inline uint4 hyle_philox(uint4 value, uint2 key) {
    for (uint round = 0; round < 10; ++round) {
      uint low0 = 0xd2511f53u * value.x;
      uint low1 = 0xcd9e8d57u * value.z;
      uint high0 = mulhi(0xd2511f53u, value.x);
      uint high1 = mulhi(0xcd9e8d57u, value.z);
      value = uint4(high1 ^ value.y ^ key.x, low1,
                    high0 ^ value.w ^ key.y, low0);
      key += uint2(0x9e3779b9u, 0xbb67ae85u);
    }
    return value;
  }
  inline float hyle_gumbel(uint hi, uint lo) {
    bool upper = (hi & 0x80000000u) != 0;
    if (upper) { hi = ~hi; lo = ~lo; }
    float small = float(hi) * 0x1p-32f + (float(lo) + .5f) * 0x1p-64f;
    float exponential = upper ? -log1p(-small) : -log(small);
    return -log(exponential);
  }
)METAL";
} // namespace hyle::compute::apple::detail
