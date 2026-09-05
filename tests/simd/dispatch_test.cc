#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <initializer_list>

#include "hyle/core/environment.h"
#include "hyle/platform/cpu.h"
#include "hyle/simd/target.h"

namespace {

using hyle::platform::cpu_features;
using hyle::platform::CpuFeature;
using hyle::platform::CpuFeatureSet;
using hyle::simd::kernel_target_name;
using hyle::simd::kernel_target_supported;
using hyle::simd::KernelTarget;

void ExpectFeatureFromEnvironment(const char* name, CpuFeature feature) {
#if defined(_WIN32)
  char* expected = nullptr;
  std::size_t length = 0;
  ASSERT_EQ(_dupenv_s(&expected, &length, name), 0) << name;
#else
  const char* expected = std::getenv(name);
#endif
  if (expected == nullptr) return;
  const bool valid = (expected[0] == '0' || expected[0] == '1') &&
                     expected[1] == '\0';
  EXPECT_TRUE(valid) << name;
  if (valid) EXPECT_EQ(cpu_features().has(feature), expected[0] == '1') << name;
#if defined(_WIN32)
  std::free(expected);
#endif
}

void ExpectEveryRequirementIsNecessary(
    KernelTarget target, std::initializer_list<CpuFeature> required) {
  const CpuFeatureSet complete(required);
  ASSERT_TRUE(kernel_target_supported(target, complete))
      << kernel_target_name(target);
  for (CpuFeature feature : required) {
    auto masked = complete;
    masked.set(feature, false);
    EXPECT_FALSE(kernel_target_supported(target, masked))
        << kernel_target_name(target) << " accepted a missing "
        << hyle::platform::cpu_feature_name(feature);
  }
}

}  // namespace

TEST(Dispatch, PlatformProbeIsCachedAndSafeToConsume) {
  const CpuFeatureSet& first = cpu_features();
  EXPECT_EQ(&first, &cpu_features());
}

TEST(Dispatch, EnvironmentFeatureExpectations) {
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE", CpuFeature::kArmSve);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE2", CpuFeature::kArmSve2);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE_AES", CpuFeature::kArmSveAes);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE_PMULL", CpuFeature::kArmSvePmull);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV", CpuFeature::kRiscvV);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV_ZVBB", CpuFeature::kRiscvZvbb);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV_ZVBC", CpuFeature::kRiscvZvbc);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV_ZVKNED", CpuFeature::kRiscvZvkned);
}

TEST(Dispatch, KernelTargetNamesAndInvalidTarget) {
  for (KernelTarget target : {
           KernelTarget::kScalar, KernelTarget::kSse4,
           KernelTarget::kAvx2, KernelTarget::kAvx512,
           KernelTarget::kNeon, KernelTarget::kSve, KernelTarget::kRvv}) {
    EXPECT_STRNE(kernel_target_name(target), "unknown");
  }
  const auto invalid = static_cast<KernelTarget>(0xff);
  EXPECT_STREQ(kernel_target_name(invalid), "unknown");
  EXPECT_FALSE(kernel_target_supported(invalid, CpuFeatureSet{}));
}

TEST(Dispatch, ScalarNeedsNoHostFeature) {
  EXPECT_TRUE(kernel_target_supported(KernelTarget::kScalar,
                                      CpuFeatureSet{}));
}

TEST(Dispatch, CanonicalX86TargetsRequireEveryCompiledFeature) {
  using enum CpuFeature;
  ExpectEveryRequirementIsNecessary(
      KernelTarget::kSse4,
      {kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42});
  ExpectEveryRequirementIsNecessary(
      KernelTarget::kAvx2,
      {kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
       kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt,
       kX86Bmi1, kX86Bmi2});
  ExpectEveryRequirementIsNecessary(
      KernelTarget::kAvx512,
      {kX86Sse2, kX86Sse3, kX86Ssse3, kX86Sse41, kX86Sse42,
       kX86Avx, kX86Avx2, kX86Fma, kX86F16c, kX86Lzcnt,
       kX86Bmi1, kX86Bmi2, kX86Avx512F, kX86Avx512Bw,
       kX86Avx512Vl, kX86Avx512Dq, kX86Avx512Cd});
}

TEST(Dispatch, ArchitectureTargetsUseOnlyTheirCanonicalFact) {
  ExpectEveryRequirementIsNecessary(KernelTarget::kNeon,
                                    {CpuFeature::kArmNeon});
  ExpectEveryRequirementIsNecessary(KernelTarget::kSve,
                                    {CpuFeature::kArmSve});
  ExpectEveryRequirementIsNecessary(KernelTarget::kRvv,
                                    {CpuFeature::kRiscvV});
}

TEST(Dispatch, CurrentArchitectureCannotSelectForeignTarget) {
#if HYLE_ARCH_X64
  EXPECT_FALSE(kernel_target_supported(KernelTarget::kNeon));
  EXPECT_FALSE(kernel_target_supported(KernelTarget::kRvv));
#elif HYLE_ARCH_ARM64
  EXPECT_FALSE(kernel_target_supported(KernelTarget::kAvx2));
  EXPECT_FALSE(kernel_target_supported(KernelTarget::kRvv));
#elif HYLE_ARCH_RISCV64
  EXPECT_FALSE(kernel_target_supported(KernelTarget::kAvx2));
  EXPECT_FALSE(kernel_target_supported(KernelTarget::kNeon));
#endif
}
