#pragma once

#include <cstdint>

#include "hyle/platform/cpu.h"

namespace hyle::simd {

// A body HYLE actually compiled or selected. This is intentionally not an
// ordered CPU tier: targets from different architectures cannot be compared,
// and host capability facts remain in hyle::platform.
enum class KernelTarget : std::uint8_t {
  kScalar,
  kSse4,
  kAvx2,
  kAvx512,
  kNeon,
  kSve,
  kRvv,
};

[[nodiscard]] constexpr const char* kernel_target_name(
    KernelTarget target) noexcept {
  switch (target) {
    case KernelTarget::kScalar: return "scalar";
    case KernelTarget::kSse4: return "sse4";
    case KernelTarget::kAvx2: return "avx2";
    case KernelTarget::kAvx512: return "avx512";
    case KernelTarget::kNeon: return "neon";
    case KernelTarget::kSve: return "sve";
    case KernelTarget::kRvv: return "rvv";
  }
  return "unknown";
}

// SIMD owns these groups because they describe the exact flags of HYLE's
// canonical generated bodies, including Highway's full target prerequisites.
[[nodiscard]] constexpr platform::CpuFeatureSet kernel_target_requirements(
    KernelTarget target) noexcept {
  using enum platform::CpuFeature;
  switch (target) {
    case KernelTarget::kScalar: return {};
    case KernelTarget::kSse4:
      return platform::CpuFeatureSet{
          kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42};
    case KernelTarget::kAvx2:
      return platform::CpuFeatureSet{
          kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
          kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt,
          kX86Bmi1, kX86Bmi2};
    case KernelTarget::kAvx512:
      return platform::CpuFeatureSet{
          kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
          kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt,
          kX86Bmi1, kX86Bmi2, kX86Avx512F, kX86Avx512Bw,
          kX86Avx512Vl, kX86Avx512Dq, kX86Avx512Cd};
    case KernelTarget::kNeon:
      return platform::CpuFeatureSet{kArmNeon};
    case KernelTarget::kSve:
      return platform::CpuFeatureSet{kArmSve};
    case KernelTarget::kRvv:
      return platform::CpuFeatureSet{kRiscvV};
  }
  return {};
}

[[nodiscard]] constexpr bool kernel_target_supported(
    KernelTarget target, const platform::CpuFeatureSet& features) noexcept {
  const auto required = kernel_target_requirements(target);
  switch (target) {
    case KernelTarget::kScalar: return true;
    case KernelTarget::kSse4:
    case KernelTarget::kAvx2:
    case KernelTarget::kAvx512:
    case KernelTarget::kNeon:
    case KernelTarget::kSve:
    case KernelTarget::kRvv: return features.has_all(required);
  }
  return false;
}

[[nodiscard]] inline bool kernel_target_supported(
    KernelTarget target) noexcept {
  return kernel_target_supported(target, platform::cpu_features());
}

}  // namespace hyle::simd
