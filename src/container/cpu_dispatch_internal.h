#pragma once

#include "hyle/platform/cpu.h"

namespace hyle::container::detail {

using platform::CpuFeatureSet;
using enum platform::CpuFeature;

// These groups describe Container's actual generated direct-Highway TUs.
// HYLE disables Highway's optional AES/PCLMUL SSE4 requirement, but retains its
// full cumulative SSE/AVX groups.
inline constexpr CpuFeatureSet kHighwaySse4Requirements{
    kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42};
inline constexpr CpuFeatureSet kHighwayAvx2Requirements{
    kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
    kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt,
    kX86Bmi1, kX86Bmi2};
inline constexpr CpuFeatureSet kHighwayAvx512Requirements{
    kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
    kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt,
    kX86Bmi1, kX86Bmi2, kX86Avx512F, kX86Avx512Bw,
    kX86Avx512Vl, kX86Avx512Dq, kX86Avx512Cd};
inline constexpr CpuFeatureSet kHighwayNeonRequirements{kArmNeon};

// hash_xxh3_avx2.cc is not a direct-Highway target. Its ordinary AVX2 flags
// permit AVX2/FMA/F16C/BMI2 but do not require Highway's BMI1/LZCNT group.
inline constexpr CpuFeatureSet kXxh3Avx2Requirements{
    kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Bmi2};

[[nodiscard]] constexpr bool supports_highway_sse4(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kHighwaySse4Requirements);
}

[[nodiscard]] constexpr bool supports_highway_avx2(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kHighwayAvx2Requirements);
}

[[nodiscard]] constexpr bool supports_highway_avx512(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kHighwayAvx512Requirements);
}

[[nodiscard]] constexpr bool supports_highway_neon(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kHighwayNeonRequirements);
}

[[nodiscard]] constexpr bool supports_xxh3_avx2(
    const CpuFeatureSet& features) noexcept {
  return features.has_all(kXxh3Avx2Requirements);
}

}  // namespace hyle::container::detail
