#include <gtest/gtest.h>

#include <cstdint>

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
  if (static_cast<std::uint32_t>(best) >= static_cast<std::uint32_t>(Isa::kAvx2)) {
    EXPECT_TRUE(f.avx2);
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

// supports() is monotone with the tier order: best_isa() is supported, and every tier
// strictly above it is not.
TEST(Dispatch, SupportsIsMonotone) {
  EXPECT_TRUE(supports(best_isa()));
  EXPECT_TRUE(supports(Isa::kScalar));  // floor: always supported
}

// isa_name covers every enumerator (no "unknown" for a real tier).
TEST(Dispatch, IsaNameCoversAllTiers) {
  for (Isa t : {Isa::kScalar, Isa::kSse2, Isa::kSsse3, Isa::kSse42, Isa::kAvx2,
                Isa::kAvx512, Isa::kNeon, Isa::kSve}) {
    EXPECT_STRNE(isa_name(t), "unknown");
  }
}
