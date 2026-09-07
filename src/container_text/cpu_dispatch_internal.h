#pragma once

#include "hyle/platform/cpu.h"

namespace hyle::container::detail {

using platform::CpuFeatureSet;
using enum platform::CpuFeature;

inline constexpr CpuFeatureSet kTextHighwaySse4Requirements{
    kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42};
inline constexpr CpuFeatureSet kTextHighwayAvx2Requirements{
    kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
    kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt, kX86Bmi1, kX86Bmi2};
inline constexpr CpuFeatureSet kTextHighwayAvx512Requirements{
    kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
    kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt, kX86Bmi1, kX86Bmi2,
    kX86Avx512F, kX86Avx512Bw, kX86Avx512Vl, kX86Avx512Dq, kX86Avx512Cd};
inline constexpr CpuFeatureSet kTextHighwayNeonRequirements{kArmNeon};

[[nodiscard]] constexpr bool supports_highway_sse4(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kTextHighwaySse4Requirements);
}
[[nodiscard]] constexpr bool supports_highway_avx2(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kTextHighwayAvx2Requirements);
}
[[nodiscard]] constexpr bool supports_highway_avx512(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kTextHighwayAvx512Requirements);
}
[[nodiscard]] constexpr bool supports_highway_neon(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kTextHighwayNeonRequirements);
}

}  // namespace hyle::container::detail
