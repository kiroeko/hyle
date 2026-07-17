// Functional tests for runtime CPU-feature detection. These assert INVARIANTS
// that must hold on ANY machine (feature implications, tier ordering, cache
// idempotence) rather than "this box has AVX2" -- the suite runs on unknown
// hardware, so it checks the logic of the probe, not a specific CPU.
#include "hyle/platform/cpu.h"

#include <gtest/gtest.h>

#include <cstring>

using hyle::platform::best_isa;
using hyle::platform::cpu_features;
using hyle::platform::CpuFeatures;
using hyle::platform::Isa;
using hyle::platform::isa_name;
using hyle::platform::supports;

// cpu_features() is a cached Meyers singleton: repeated calls return the SAME
// object (same address, identical bits), never re-probe into a different answer.
TEST(Cpu, FeaturesCachedIdempotent) {
  const CpuFeatures& a = cpu_features();
  const CpuFeatures& b = cpu_features();
  EXPECT_EQ(&a, &b);  // same cached instance
  EXPECT_EQ(std::memcmp(&a, &b, sizeof(CpuFeatures)), 0);
}

// scalar is the portable floor: always supported, on every arch.
TEST(Cpu, ScalarAlwaysSupported) {
  EXPECT_TRUE(supports(Isa::kScalar));
}

// best_isa() must itself be supported, and be a fixed answer across calls.
TEST(Cpu, BestIsaSupportedAndStable) {
  Isa best = best_isa();
  EXPECT_EQ(best, best_isa());       // stable
  EXPECT_TRUE(supports(best));       // the best tier is, tautologically, supported
}

// isa_name covers every tier with a non-empty, distinct-ish string.
TEST(Cpu, IsaNameNonEmpty) {
  const Isa tiers[] = {Isa::kScalar, Isa::kSse2, Isa::kSsse3, Isa::kSse42,
                       Isa::kAvx,    Isa::kAvx2, Isa::kAvx512, Isa::kNeon,
                       Isa::kSve,    Isa::kRvv};
  for (Isa t : tiers) {
    const char* n = isa_name(t);
    ASSERT_NE(n, nullptr);
    EXPECT_GT(std::strlen(n), 0u);
  }
  EXPECT_STREQ(isa_name(Isa::kScalar), "scalar");
}

// Feature-implication invariants: a higher capability logically REQUIRES the
// lower ones the tiers/groups are built on. These hold regardless of the host.
TEST(Cpu, FeatureImplicationsHold) {
  const CpuFeatures& f = cpu_features();

#if defined(HYLE_ARCH_X64)
  // x86-64 baseline: SSE2 is mandatory.
  EXPECT_TRUE(f.sse2);

  // AVX-512 foundation implies the AVX2 group beneath it (Highway kGroupAVX3
  // subsumes kGroupAVX2). If the box has AVX-512 it must also report AVX2 + FMA.
  if (f.avx512f) {
    EXPECT_TRUE(f.avx2);
    EXPECT_TRUE(f.fma);
  }
  // AVX-512 sub-features ride on the foundation.
  if (f.avx512bw || f.avx512vl || f.avx512dq || f.avx512cd || f.avx512vnni ||
      f.avx512bf16 || f.avx512fp16) {
    EXPECT_TRUE(f.avx512f);
  }
  // AVX2 (256-bit int) implies AVX (256-bit float) is available.
  if (f.avx2) { EXPECT_TRUE(f.avx); }
  // SSE ladder is cumulative.
  if (f.sse42) { EXPECT_TRUE(f.sse41); }
  if (f.sse41) { EXPECT_TRUE(f.ssse3); }
  if (f.ssse3) { EXPECT_TRUE(f.sse2); }
#elif defined(HYLE_ARCH_ARM64)
  // aarch64 baseline: NEON is mandatory.
  EXPECT_TRUE(f.neon);
  // SVE2 implies SVE; SVE sub-features imply SVE2 / SVE.
  if (f.sve2) { EXPECT_TRUE(f.sve); }
  if (f.sve_aes || f.sve_pmull) { EXPECT_TRUE(f.sve2); }
  if (f.sve_i8mm || f.sve_f32mm || f.sve_f64mm || f.sve_bf16) {
    EXPECT_TRUE(f.sve);
  }
#elif defined(HYLE_ARCH_RISCV64)
  // Every vector sub-extension implies the V extension.
  if (f.rvv_zvbb || f.rvv_zvbc || f.rvv_zvkned || f.rvv_zvfh || f.rvv_zvfhmin) {
    EXPECT_TRUE(f.rvv);
  }
#endif
}

// best_isa() must AGREE with the feature bits: the reported tier's whole group
// must be present, and the next tier up must NOT be fully present.
TEST(Cpu, BestIsaAgreesWithFeatures) {
  const CpuFeatures& f = cpu_features();
  Isa best = best_isa();

#if defined(HYLE_ARCH_X64)
  const bool avx2_group =
      f.avx2 && f.fma && f.f16c && f.lzcnt && f.bmi1 && f.bmi2 && f.sse42;
  const bool avx512_group = avx2_group && f.avx512f && f.avx512bw &&
                            f.avx512vl && f.avx512dq && f.avx512cd;
  if (best == Isa::kAvx512) {
    EXPECT_TRUE(avx512_group);
  } else if (best == Isa::kAvx2) {
    EXPECT_TRUE(avx2_group);
    EXPECT_FALSE(avx512_group);  // else it would have scored higher
  }
  // Whatever the tier, scalar/sse2 remain supported on x86-64.
  EXPECT_TRUE(supports(Isa::kSse2));
#elif defined(HYLE_ARCH_ARM64)
  if (best == Isa::kSve) EXPECT_TRUE(f.sve);
  EXPECT_TRUE(supports(Isa::kNeon));  // mandatory on aarch64
#elif defined(HYLE_ARCH_RISCV64)
  if (best == Isa::kRvv) EXPECT_TRUE(f.rvv);
#endif
}

// supports() is monotone in the tier order within an arch: everything <= best
// is supported, everything > best is not.
TEST(Cpu, SupportsIsMonotone) {
  Isa best = best_isa();
  const std::uint32_t bv = static_cast<std::uint32_t>(best);

#if defined(HYLE_ARCH_X64)
  const Isa arch_tiers[] = {Isa::kSse2, Isa::kSsse3, Isa::kSse42,
                            Isa::kAvx2, Isa::kAvx512};
#elif defined(HYLE_ARCH_ARM64)
  const Isa arch_tiers[] = {Isa::kNeon, Isa::kSve};
#elif defined(HYLE_ARCH_RISCV64)
  const Isa arch_tiers[] = {Isa::kRvv};
#else
  const Isa arch_tiers[] = {Isa::kScalar};
#endif
  for (Isa t : arch_tiers) {
    const bool le = static_cast<std::uint32_t>(t) <= bv;
    EXPECT_EQ(supports(t), le)
        << "tier " << isa_name(t) << " vs best " << isa_name(best);
  }
}
