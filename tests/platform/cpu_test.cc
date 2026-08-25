// Functional tests for runtime CPU-feature detection. These assert INVARIANTS
// that must hold on ANY machine (feature implications, tier ordering, cache
// idempotence) rather than "this box has AVX2" -- the suite runs on unknown
// hardware, so it checks the logic of the probe, not a specific CPU.
#include "hyle/platform/cpu.h"

#include "hyle/core/cache.h"

#include "cpu_internal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#endif

using hyle::platform::available_cpu_ids;
using hyle::platform::available_cpu_count;
using hyle::platform::best_isa;
using hyle::platform::cache_line_size;
using hyle::platform::cache_info;
using hyle::platform::cpu_topology;
using hyle::platform::cpu_features;
using hyle::platform::CpuInfo;
using hyle::platform::CpuKind;
using hyle::platform::CpuOrder;
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

// cache_line_size(): a sane power-of-two in [16, 256] (commonly 64; Apple
// Silicon reports 128),
// stable across calls (cached). An invariant, not a specific-box assertion.
TEST(Cpu, CacheLineSizeSane) {
  const unsigned c = cache_line_size();
  EXPECT_GE(c, 16u);
  EXPECT_LE(c, 256u);
  EXPECT_EQ(c & (c - 1u), 0u);      // power of two
  EXPECT_EQ(c, cache_line_size());  // stable / cached
}

TEST(Cpu, RuntimeCacheLineFitsCertifiedStaticLayoutBounds) {
  const unsigned runtime = cache_line_size();
  EXPECT_GE(runtime, hyle::core::kConstructiveInterferenceSize);
  EXPECT_LE(runtime, hyle::core::kDestructiveInterferenceSize);
}

TEST(Cpu, CacheCapacityIsCachedAndHierarchicalWhenKnown) {
  const auto& a = cache_info();
  const auto& b = cache_info();
  EXPECT_EQ(&a, &b);
  if (a.l1d && a.l2) { EXPECT_LE(a.l1d, a.l2); }
  if (a.l1i && a.l2) { EXPECT_LE(a.l1i, a.l2); }
  if (a.l2 && a.l3) { EXPECT_LE(a.l2, a.l3); }
}

TEST(Cpu, TopologyHasConsistentLogicalCounts) {
  const auto& t = cpu_topology();
  EXPECT_GE(t.system_logical_cpus, 1u);
  EXPECT_EQ(t.performance_cpus + t.efficiency_cpus,
            t.system_logical_cpus);
  EXPECT_EQ(t.hybrid, t.efficiency_cpus != 0 && t.performance_cpus != 0);
}

TEST(Cpu, AvailableCpuIdsRespectBufferFillAndAreUnique) {
  auto count = available_cpu_ids(nullptr, 0);
  ASSERT_TRUE(count.has_value()) << count.error().message();
  ASSERT_GE(*count, 1u);
  auto convenient_count = available_cpu_count();
  ASSERT_TRUE(convenient_count.has_value())
      << convenient_count.error().message();
  EXPECT_EQ(*convenient_count, *count);
  EXPECT_LE(*count, cpu_topology().system_logical_cpus);

  unsigned first = ~0u;
  auto truncated = available_cpu_ids(&first, 1);
  ASSERT_TRUE(truncated.has_value()) << truncated.error().message();
  EXPECT_EQ(*truncated, *count);
  EXPECT_NE(first, ~0u);

  std::vector<unsigned> ids(*count);
  auto filled = available_cpu_ids(ids.data(), ids.size());
  ASSERT_TRUE(filled.has_value()) << filled.error().message();
  ASSERT_EQ(*filled, ids.size());
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
}

TEST(Cpu, AvailableCpuInfoMatchesIdsAndClassifiesKnownRanks) {
  auto count = hyle::platform::available_cpu_info(nullptr, 0);
  ASSERT_TRUE(count.has_value()) << count.error().message();
  std::vector<CpuInfo> info(*count);
  auto filled = hyle::platform::available_cpu_info(info.data(), info.size());
  ASSERT_TRUE(filled.has_value()) << filled.error().message();
  ASSERT_EQ(*filled, info.size());

  std::vector<unsigned> ids(*count);
  ASSERT_TRUE(available_cpu_ids(ids.data(), ids.size()).has_value());
  for (std::size_t i = 0; i < info.size(); ++i) EXPECT_EQ(info[i].id, ids[i]);

  unsigned max_rank = 0;
  for (const auto& cpu : info) max_rank = std::max(max_rank, cpu.performance_rank);
  for (const auto& cpu : info) {
    if (cpu.performance_rank == 0) {
      EXPECT_EQ(cpu.kind, CpuKind::kUnknown);
    } else if (cpu.performance_rank == max_rank) {
      EXPECT_EQ(cpu.kind, CpuKind::kPerformance);
    } else {
      EXPECT_EQ(cpu.kind, CpuKind::kEfficiency);
    }
  }
#if defined(__APPLE__)
  EXPECT_TRUE(std::all_of(info.begin(), info.end(), [](const CpuInfo& cpu) {
    return cpu.performance_rank == 0 && cpu.kind == CpuKind::kUnknown;
  }));
#endif
}

TEST(Cpu, OrderedAvailableIdsPreferKnownRanksWithoutDroppingCpus) {
  auto count = available_cpu_ids(nullptr, 0);
  ASSERT_TRUE(count.has_value()) << count.error().message();
  std::vector<CpuInfo> info(*count);
  ASSERT_TRUE(hyle::platform::available_cpu_info(info.data(), info.size()));
  std::vector<unsigned> natural(*count), performance(*count), efficiency(*count);
  ASSERT_TRUE(available_cpu_ids(natural.data(), natural.size()));
  ASSERT_TRUE(available_cpu_ids(CpuOrder::kPerformanceFirst,
                                performance.data(), performance.size()));
  ASSERT_TRUE(available_cpu_ids(CpuOrder::kEfficiencyFirst,
                                efficiency.data(), efficiency.size()));

  auto sorted_natural = natural;
  auto sorted_performance = performance;
  auto sorted_efficiency = efficiency;
  std::sort(sorted_natural.begin(), sorted_natural.end());
  std::sort(sorted_performance.begin(), sorted_performance.end());
  std::sort(sorted_efficiency.begin(), sorted_efficiency.end());
  EXPECT_EQ(sorted_performance, sorted_natural);
  EXPECT_EQ(sorted_efficiency, sorted_natural);

  auto rank_of = [&info](unsigned id) {
    auto found = std::find_if(info.begin(), info.end(),
                              [id](const CpuInfo& cpu) { return cpu.id == id; });
    return found == info.end() ? 0u : found->performance_rank;
  };
  bool unknown_seen = false;
  unsigned previous = std::numeric_limits<unsigned>::max();
  for (unsigned id : performance) {
    const unsigned rank = rank_of(id);
    if (rank == 0) unknown_seen = true;
    else {
      EXPECT_FALSE(unknown_seen);
      EXPECT_LE(rank, previous);
      previous = rank;
    }
  }
  unknown_seen = false;
  previous = 0;
  for (unsigned id : efficiency) {
    const unsigned rank = rank_of(id);
    if (rank == 0) unknown_seen = true;
    else {
      EXPECT_FALSE(unknown_seen);
      EXPECT_GE(rank, previous);
      previous = rank;
    }
  }

  const bool all_unknown = std::all_of(info.begin(), info.end(),
                                       [](const CpuInfo& cpu) {
                                         return cpu.performance_rank == 0;
                                       });
  if (all_unknown) {
    EXPECT_EQ(performance, natural);
    EXPECT_EQ(efficiency, natural);
  }
  auto invalid = available_cpu_ids(static_cast<CpuOrder>(255), nullptr, 0);
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
}

TEST(Cpu, AvailableCpuIdsTracksRuntimeAffinityChange) {
#if defined(_WIN32)
  if (::GetActiveProcessorGroupCount() != 1) GTEST_SKIP();
  DWORD_PTR original = 0, system = 0;
  if (!::GetProcessAffinityMask(::GetCurrentProcess(), &original, &system) ||
      original == 0) {
    GTEST_SKIP();
  }
  const DWORD_PTR single = original & (~original + 1);
  if (!::SetProcessAffinityMask(::GetCurrentProcess(), single)) GTEST_SKIP();
  auto restricted = available_cpu_ids(nullptr, 0);
  const BOOL restored =
      ::SetProcessAffinityMask(::GetCurrentProcess(), original);
  ASSERT_NE(restored, 0);
  ASSERT_TRUE(restricted.has_value()) << restricted.error().message();
  EXPECT_EQ(*restricted, 1u);
#elif defined(__linux__)
  cpu_set_t original;
  CPU_ZERO(&original);
  if (::sched_getaffinity(0, sizeof(original), &original) != 0) GTEST_SKIP();
  int first = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &original)) {
      first = cpu;
      break;
    }
  }
  if (first < 0) GTEST_SKIP();
  cpu_set_t single;
  CPU_ZERO(&single);
  CPU_SET(first, &single);
  if (::sched_setaffinity(0, sizeof(single), &single) != 0) GTEST_SKIP();
  auto restricted = available_cpu_ids(nullptr, 0);
  const int restored = ::sched_setaffinity(0, sizeof(original), &original);
  ASSERT_EQ(restored, 0);
  ASSERT_TRUE(restricted.has_value()) << restricted.error().message();
  EXPECT_EQ(*restricted, 1u);
#else
  GTEST_SKIP() << "platform has no process affinity API";
#endif
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
  const auto groups = hyle::platform::detail::x86_isa_groups(f);
  if (best == Isa::kAvx512) {
    EXPECT_TRUE(groups.avx512);
  } else if (best == Isa::kAvx2) {
    EXPECT_TRUE(groups.avx2);
    EXPECT_FALSE(groups.avx512);  // else it would have scored higher
  }
  // Whatever the tier, scalar/sse2 remain supported on x86-64.
  EXPECT_TRUE(supports(Isa::kSse2));
#elif defined(HYLE_ARCH_ARM64)
  if (best == Isa::kSve) {
    EXPECT_TRUE(f.sve);
  }
  EXPECT_TRUE(supports(Isa::kNeon));  // mandatory on aarch64
#elif defined(HYLE_ARCH_RISCV64)
  if (best == Isa::kRvv) {
    EXPECT_TRUE(f.rvv);
  }
#endif
}

#if defined(HYLE_ARCH_X64)
TEST(Cpu, SyntheticIsaGroupsRequireEveryCompiledTierFeature) {
  CpuFeatures f;
  f.avx = true;
  f.avx2 = true;
  f.fma = true;
  f.f16c = true;
  f.lzcnt = true;
  f.bmi1 = true;
  f.bmi2 = true;
  f.sse42 = true;

  EXPECT_TRUE(hyle::platform::detail::x86_isa_groups(f).avx2);
  for (bool CpuFeatures::*required : {
           &CpuFeatures::avx,
           &CpuFeatures::avx2,
           &CpuFeatures::fma,
           &CpuFeatures::f16c,
           &CpuFeatures::lzcnt,
           &CpuFeatures::bmi1,
           &CpuFeatures::bmi2,
           &CpuFeatures::sse42,
       }) {
    auto masked = f;
    masked.*required = false;
    EXPECT_FALSE(hyle::platform::detail::x86_isa_groups(masked).avx2);
  }

  f.avx512f = true;
  f.avx512bw = true;
  f.avx512vl = true;
  f.avx512dq = true;
  f.avx512cd = true;
  EXPECT_TRUE(hyle::platform::detail::x86_isa_groups(f).avx512);
  for (bool CpuFeatures::*required : {
           &CpuFeatures::avx,
           &CpuFeatures::avx2,
           &CpuFeatures::fma,
           &CpuFeatures::f16c,
           &CpuFeatures::lzcnt,
           &CpuFeatures::bmi1,
           &CpuFeatures::bmi2,
           &CpuFeatures::sse42,
           &CpuFeatures::avx512f,
           &CpuFeatures::avx512bw,
           &CpuFeatures::avx512vl,
           &CpuFeatures::avx512dq,
           &CpuFeatures::avx512cd,
       }) {
    auto masked = f;
    masked.*required = false;
    EXPECT_FALSE(hyle::platform::detail::x86_isa_groups(masked).avx512);
  }
}
#endif

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
