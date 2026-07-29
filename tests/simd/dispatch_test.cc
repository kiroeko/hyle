#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"

using hyle::simd::best_isa;
using hyle::simd::cpu_features;
using hyle::simd::CpuFeatures;
using hyle::simd::Isa;
using hyle::simd::isa_name;
using hyle::simd::supports;

// detect() must not crash (the whole point of probe isolation: XGETBV/CPUID run here on
// the test machine's real CPU). Calling it proves the isolated TU executes safely.
TEST(Dispatch, DetectDoesNotCrash) {
  const CpuFeatures& f = cpu_features();
  (void)f;
  SUCCEED();
}

// The result is cached (Meyers singleton): two calls return the same object.
TEST(Dispatch, FeaturesAreCachedStableObject) {
  const CpuFeatures& a = cpu_features();
  const CpuFeatures& b = cpu_features();
  EXPECT_EQ(&a, &b);  // same instance -> detected once
}

namespace {

void ExpectFeatureFromEnvironment(const char* name, bool actual) {
#if defined(HYLE_COMPILER_MSVC)
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
  if (valid) {
    EXPECT_EQ(actual, expected[0] == '1') << name;
  }
#if defined(HYLE_COMPILER_MSVC)
  std::free(expected);
#endif
}

}  // namespace

// Cross-architecture QEMU gates set only the variables relevant to the selected
// virtual CPU. This turns feature probing into positive evidence: compiling an
// extension TU is not enough; the baseline detector must observe the extension
// without itself containing any advanced instruction.
TEST(Dispatch, EnvironmentFeatureExpectations) {
  const CpuFeatures& f = cpu_features();
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE", f.sve);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE2", f.sve2);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE_AES", f.sve_aes);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_SVE_PMULL", f.sve_pmull);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV", f.rvv);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV_ZVBB", f.rvv_zvbb);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV_ZVBC", f.rvv_zvbc);
  ExpectFeatureFromEnvironment("HYLE_EXPECT_RVV_ZVKNED", f.rvv_zvkned);
}

#if defined(HYLE_ARCH_X64)
// SSE2 is guaranteed on every x86-64 CPU, so the probe MUST report it and best_isa
// must be at least the SSE2 tier.
TEST(Dispatch, X64AlwaysHasSse2) {
  EXPECT_TRUE(cpu_features().sse2);
  EXPECT_GE(static_cast<std::uint32_t>(best_isa()),
            static_cast<std::uint32_t>(Isa::kSse2));
}

// Feature implication: the tier groups must be internally consistent and CUMULATIVE.
// If best_isa reports AVX2, the CPU must really have the whole AVX2 group we gate on
// (avx2+fma+f16c+lzcnt+bmi1+bmi2+sse42, mirroring Highway kGroupAVX2); if AVX-512, it
// must ALSO have that AVX2 base (kGroupAVX3 subsumes it). Guards against a tier being
// claimed without its prerequisites (which would SIGILL when that kernel runs) and
// against the enum's "higher tier implies all lower" contract being violated.
TEST(Dispatch, TierImpliesItsFeatureGroup) {
  const CpuFeatures& f = cpu_features();
  const Isa best = best_isa();
  // The pure-AVX tier (Sandy/Ivy Bridge: 256-bit float, no AVX2 int) implies f.avx, which
  // itself AND's in the OS-YMM-save check. AVX2 subsumes AVX, so this also holds above.
  if (static_cast<std::uint32_t>(best) >= static_cast<std::uint32_t>(Isa::kAvx)) {
    EXPECT_TRUE(f.avx);
  }
  if (static_cast<std::uint32_t>(best) >= static_cast<std::uint32_t>(Isa::kAvx2)) {
    EXPECT_TRUE(f.avx2);
    EXPECT_TRUE(f.avx);  // AVX2 tier subsumes pure-AVX
    EXPECT_TRUE(f.fma);
    EXPECT_TRUE(f.f16c);
    EXPECT_TRUE(f.lzcnt);
    EXPECT_TRUE(f.bmi1);
    EXPECT_TRUE(f.bmi2);
    EXPECT_TRUE(f.sse42);
  }
  if (static_cast<std::uint32_t>(best) >= static_cast<std::uint32_t>(Isa::kAvx512)) {
    EXPECT_TRUE(f.avx512f);
    EXPECT_TRUE(f.avx512bw);
    EXPECT_TRUE(f.avx512vl);
    EXPECT_TRUE(f.avx512dq);
    EXPECT_TRUE(f.avx512cd);
    EXPECT_TRUE(f.avx2);  // AVX-512 tier subsumes the AVX2 group
    EXPECT_TRUE(f.fma);
  }
}

// AVX-512 extensions are meaningless without the AVX-512 base: the probe must never
// leave vnni/bf16/fp16 set while the base is cleared (e.g. OS didn't enable ZMM save).
TEST(Dispatch, Avx512ExtensionsImplyBase) {
  const CpuFeatures& f = cpu_features();
  if (f.avx512vnni || f.avx512bf16 || f.avx512fp16) {
    EXPECT_TRUE(f.avx512f);
  }
}
#endif  // HYLE_ARCH_X64

#if defined(HYLE_ARCH_ARM64)
TEST(Dispatch, ArmScalableExtensionsImplyBase) {
  const CpuFeatures& f = cpu_features();
  if (f.sve2 || f.sve_aes || f.sve_pmull || f.sve_i8mm || f.sve_f32mm ||
      f.sve_f64mm || f.sve_bf16) {
    EXPECT_TRUE(f.sve);
  }
  if (f.sve_aes || f.sve_pmull) {
    EXPECT_TRUE(f.sve2);
  }
}
#endif

#if defined(HYLE_ARCH_RISCV64)
TEST(Dispatch, RvvExtensionsImplyBase) {
  const CpuFeatures& f = cpu_features();
  if (f.rvv_zvbb || f.rvv_zvbc || f.rvv_zvkned || f.rvv_zvfh ||
      f.rvv_zvfhmin) {
    EXPECT_TRUE(f.rvv);
  }
}
#endif

// supports() is monotone with the tier order: best_isa() is supported, and every tier
// strictly above it is not.
TEST(Dispatch, SupportsIsMonotone) {
  EXPECT_TRUE(supports(best_isa()));
  EXPECT_TRUE(supports(Isa::kScalar));  // floor: always supported
#if defined(HYLE_ARCH_X64)
  EXPECT_FALSE(supports(Isa::kNeon));
  EXPECT_FALSE(supports(Isa::kRvv));
#elif defined(HYLE_ARCH_ARM64)
  EXPECT_FALSE(supports(Isa::kAvx2));
  EXPECT_FALSE(supports(Isa::kRvv));
#elif defined(HYLE_ARCH_RISCV64)
  EXPECT_FALSE(supports(Isa::kAvx2));
  EXPECT_FALSE(supports(Isa::kNeon));
#endif
}

// isa_name covers every enumerator (no "unknown" for a real tier).
TEST(Dispatch, IsaNameCoversAllTiers) {
  for (Isa t : {Isa::kScalar, Isa::kSse2, Isa::kSsse3, Isa::kSse42, Isa::kAvx,
                Isa::kAvx2, Isa::kAvx512, Isa::kNeon, Isa::kSve, Isa::kRvv}) {
    EXPECT_STRNE(isa_name(t), "unknown");
  }
}
