// Functional tests for runtime CPU-feature detection. These assert invariants
// that must hold on any machine (feature implications, OS-state gating, cache
// idempotence) rather than "this box has AVX2" -- the suite runs on unknown
// hardware, so it checks the logic of the probe, not a specific CPU.
#include "hyle/platform/cpu.h"

#include "cpu_internal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <thread>
#include <type_traits>
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
using hyle::platform::cache_line_size;
using hyle::platform::cache_info;
using hyle::platform::available_cpu_topology;
using hyle::platform::cpu_feature_name;
using hyle::platform::cpu_features;
using hyle::platform::CpuFeature;
using hyle::platform::CpuFeatureSet;
using hyle::platform::CpuInfo;
using hyle::platform::CpuKind;
using hyle::platform::CpuOrder;

// cpu_features() is a cached Meyers singleton: repeated calls return the SAME
// object (same address, identical bits), never re-probe into a different answer.
TEST(Cpu, FeaturesCachedIdempotent) {
  const CpuFeatureSet& a = cpu_features();
  const CpuFeatureSet& b = cpu_features();
  EXPECT_EQ(&a, &b);  // same cached instance
  EXPECT_EQ(a, b);
  static_assert(sizeof(CpuFeatureSet) == 2 * sizeof(std::uint64_t));
  static_assert(std::is_trivially_copyable_v<CpuFeatureSet>);
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

TEST(Cpu, CacheCapacityIsCachedAndHierarchicalWhenKnown) {
  const auto& a = cache_info();
  const auto& b = cache_info();
  EXPECT_EQ(&a, &b);
  if (a.l1d && a.l2) { EXPECT_LE(a.l1d, a.l2); }
  if (a.l1i && a.l2) { EXPECT_LE(a.l1i, a.l2); }
  if (a.l2 && a.l3) { EXPECT_LE(a.l2, a.l3); }
}

TEST(Cpu, TopologyHasConsistentLogicalCounts) {
  auto topology = available_cpu_topology();
  ASSERT_TRUE(topology.has_value()) << topology.error().message();
  const auto& t = *topology;
  EXPECT_GE(t.logical_cpus, 1u);
  EXPECT_LE(t.physical_cores, t.logical_cpus);
  EXPECT_LE(t.performance_cpus + t.efficiency_cpus, t.logical_cpus);
  EXPECT_EQ(t.hybrid, t.efficiency_cpus != 0 && t.performance_cpus != 0);
#if defined(_WIN32)
  // Windows relation records are variable-sized: a Group record can be
  // smaller than sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX). Rejecting
  // that valid prefix used to stop the scan before every core/package record.
  EXPECT_GE(t.physical_cores, 1u);
  EXPECT_GE(t.packages, 1u);
  EXPECT_GE(t.numa_nodes, 1u);
#endif
}

TEST(Cpu, AvailableCpuIdsRespectBufferFillAndAreUnique) {
  auto count = available_cpu_ids(nullptr, 0);
  ASSERT_TRUE(count.has_value()) << count.error().message();
  ASSERT_GE(*count, 1u);
  auto convenient_count = available_cpu_count();
  ASSERT_TRUE(convenient_count.has_value())
      << convenient_count.error().message();
  EXPECT_EQ(*convenient_count, *count);
  auto topology = available_cpu_topology();
  ASSERT_TRUE(topology.has_value()) << topology.error().message();
  EXPECT_EQ(*count, topology->logical_cpus);

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
  CpuInfo prefix{};
  auto truncated = hyle::platform::available_cpu_info(&prefix, 1);
  ASSERT_TRUE(truncated.has_value()) << truncated.error().message();
  ASSERT_EQ(*truncated, info.size());
  EXPECT_EQ(prefix.id, info.front().id);
  EXPECT_EQ(prefix.performance_rank, info.front().performance_rank);
  EXPECT_EQ(prefix.kind, info.front().kind);
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
  auto restricted_topology = available_cpu_topology();
  const BOOL restored =
      ::SetProcessAffinityMask(::GetCurrentProcess(), original);
  ASSERT_NE(restored, 0);
  ASSERT_TRUE(restricted.has_value()) << restricted.error().message();
  EXPECT_EQ(*restricted, 1u);
  ASSERT_TRUE(restricted_topology.has_value())
      << restricted_topology.error().message();
  EXPECT_EQ(restricted_topology->logical_cpus, 1u);
  EXPECT_EQ(restricted_topology->physical_cores, 1u);
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
  auto restricted_topology = available_cpu_topology();
  const int restored = ::sched_setaffinity(0, sizeof(original), &original);
  ASSERT_EQ(restored, 0);
  ASSERT_TRUE(restricted.has_value()) << restricted.error().message();
  EXPECT_EQ(*restricted, 1u);
  ASSERT_TRUE(restricted_topology.has_value())
      << restricted_topology.error().message();
  EXPECT_EQ(restricted_topology->logical_cpus, 1u);
#else
  GTEST_SKIP() << "platform has no process affinity API";
#endif
}

TEST(Cpu, FeatureSetAlgebraAndInvalidValues) {
  CpuFeatureSet features{CpuFeature::kX86Sse2, CpuFeature::kX86Avx};
  EXPECT_TRUE(features.has(CpuFeature::kX86Sse2));
  EXPECT_FALSE(features.has(CpuFeature::kX86Avx2));
  EXPECT_TRUE(features.has_all(CpuFeatureSet{CpuFeature::kX86Sse2}));
  EXPECT_FALSE(features.has_all(CpuFeatureSet{
      CpuFeature::kX86Sse2, CpuFeature::kX86Avx2}));
  EXPECT_TRUE(features.has_any(CpuFeatureSet{
      CpuFeature::kArmNeon, CpuFeature::kX86Avx}));
  features.set(CpuFeature::kX86Avx, false);
  EXPECT_FALSE(features.has(CpuFeature::kX86Avx));

  const auto invalid = static_cast<CpuFeature>(
      static_cast<std::uint16_t>(CpuFeature::kCount) + 7u);
  const auto before = features;
  EXPECT_FALSE(features.has(invalid));
  features.set(invalid);
  EXPECT_EQ(features, before);
}

TEST(Cpu, FeatureNamesCoverTheDeclaredSet) {
  for (std::uint16_t i = 0;
       i < static_cast<std::uint16_t>(CpuFeature::kCount); ++i) {
    const char* name = cpu_feature_name(static_cast<CpuFeature>(i));
    ASSERT_NE(name, nullptr);
    EXPECT_NE(name[0], '\0');
    EXPECT_STRNE(name, "unknown");
  }
  EXPECT_STREQ(cpu_feature_name(static_cast<CpuFeature>(0xffffu)), "unknown");
}

TEST(Cpu, FeatureCacheIsSafeForConcurrentReaders) {
  std::array<const CpuFeatureSet*, 8> observed{};
  std::array<std::thread, 8> readers;
  for (std::size_t i = 0; i < readers.size(); ++i) {
    readers[i] = std::thread([&, i] { observed[i] = &cpu_features(); });
  }
  for (auto& reader : readers) reader.join();
  for (const CpuFeatureSet* value : observed) EXPECT_EQ(value, &cpu_features());
}

TEST(Cpu, CurrentHostFeatureImplicationsHold) {
  const CpuFeatureSet& f = cpu_features();
#if HYLE_ARCH_X64
  EXPECT_TRUE(f.has(CpuFeature::kX86Sse2));
  if (f.has(CpuFeature::kX86Avx2)) {
    EXPECT_TRUE(f.has(CpuFeature::kX86Avx));
  }
  const CpuFeatureSet avx512_children{
      CpuFeature::kX86Avx512Bw, CpuFeature::kX86Avx512Vl,
      CpuFeature::kX86Avx512Dq, CpuFeature::kX86Avx512Cd,
      CpuFeature::kX86Avx512Vnni, CpuFeature::kX86Avx512Vbmi,
      CpuFeature::kX86Avx512Vbmi2, CpuFeature::kX86Avx512Bitalg,
      CpuFeature::kX86Avx512Vpopcntdq, CpuFeature::kX86Avx512Bf16,
      CpuFeature::kX86Avx512Fp16};
  if (f.has_any(avx512_children)) {
    EXPECT_TRUE(f.has(CpuFeature::kX86Avx512F));
  }
  if (f.has_any(CpuFeatureSet{
          CpuFeature::kX86AmxInt8Hardware,
          CpuFeature::kX86AmxBf16Hardware,
          CpuFeature::kX86AmxFp16Hardware})) {
    EXPECT_TRUE(f.has(CpuFeature::kX86AmxTileHardware));
  }
#elif HYLE_ARCH_ARM64
  EXPECT_TRUE(f.has(CpuFeature::kArmNeon));
  if (f.has(CpuFeature::kArmSve2)) {
    EXPECT_TRUE(f.has(CpuFeature::kArmSve));
  }
  if (f.has_any(CpuFeatureSet{CpuFeature::kArmSveAes,
                              CpuFeature::kArmSvePmull})) {
    EXPECT_TRUE(f.has(CpuFeature::kArmSve2));
  }
  if (f.has_any(CpuFeatureSet{
          CpuFeature::kArmSveI8mm, CpuFeature::kArmSveF32mm,
          CpuFeature::kArmSveF64mm, CpuFeature::kArmSveBf16})) {
    EXPECT_TRUE(f.has(CpuFeature::kArmSve));
  }
#elif HYLE_ARCH_RISCV64
  if (f.has_any(CpuFeatureSet{
          CpuFeature::kRiscvZvbb, CpuFeature::kRiscvZvbc,
          CpuFeature::kRiscvZvkned, CpuFeature::kRiscvZvfh,
          CpuFeature::kRiscvZvfhmin})) {
    EXPECT_TRUE(f.has(CpuFeature::kRiscvV));
  }
#endif
}

TEST(Cpu, SyntheticX86DecodeAppliesOsStateAndHardwareBoundaries) {
  using hyle::platform::detail::X86FeatureRegisters;
  auto b = [](unsigned bit) { return std::uint32_t{1} << bit; };
  X86FeatureRegisters r;
  r.leaf1_ecx = b(0) | b(1) | b(9) | b(12) | b(19) | b(20) |
                  b(25) | b(27) | b(28) | b(29);
  r.leaf7_0_ebx = b(3) | b(5) | b(8) | b(16) | b(17) | b(28) |
                  b(30) | b(31);
  r.leaf7_0_ecx = b(1) | b(6) | b(8) | b(9) | b(10) | b(11) |
                  b(12) | b(14);
  r.leaf7_0_edx = b(22) | b(23) | b(24) | b(25);
  r.leaf7_1_eax = b(4) | b(5) | b(21);
  r.leaf7_1_edx = b(19);
  r.extended1_ecx = b(5);
  r.leaf24_ebx = 2 | b(18);
  r.xcr0 = b(1) | b(2) | b(5) | b(6) | b(7);

  const auto all = hyle::platform::detail::decode_x86_features(r);
  EXPECT_TRUE(all.has_all(CpuFeatureSet{
      CpuFeature::kX86Sse3, CpuFeature::kX86Aes,
      CpuFeature::kX86Pclmulqdq, CpuFeature::kX86Avx2,
      CpuFeature::kX86Avx512Vbmi2, CpuFeature::kX86Gfni,
      CpuFeature::kX86Vaes, CpuFeature::kX86Vpclmulqdq,
       CpuFeature::kX86Avx10_2_512,
       CpuFeature::kX86AmxFp16Hardware}));

  r.xcr0 = b(1);  // OSXSAVE is advertised, but neither YMM nor ZMM is enabled.
  const auto narrow = hyle::platform::detail::decode_x86_features(r);
  EXPECT_TRUE(narrow.has_all(CpuFeatureSet{
      CpuFeature::kX86Sse3, CpuFeature::kX86Aes,
      CpuFeature::kX86AmxTileHardware}));
  EXPECT_FALSE(narrow.has_any(CpuFeatureSet{
      CpuFeature::kX86Avx, CpuFeature::kX86Fma, CpuFeature::kX86F16c,
      CpuFeature::kX86Avx2, CpuFeature::kX86Avx512F,
       CpuFeature::kX86Gfni, CpuFeature::kX86Avx10_2_512}));

  r.xcr0 = b(1) | b(2) | b(5) | b(6) | b(7);
  r.leaf7_1_edx = 0;
  EXPECT_FALSE(hyle::platform::detail::decode_x86_features(r).has(
      CpuFeature::kX86Avx10_2_512));
  r.leaf7_1_edx = b(19);
  r.leaf24_ebx = 2;  // AVX10.2, but no 512-bit width.
  EXPECT_FALSE(hyle::platform::detail::decode_x86_features(r).has(
      CpuFeature::kX86Avx10_2_512));
  r.leaf24_ebx = 2 | b(18);

  r.leaf7_0_edx &= ~b(24);
  const auto no_tile = hyle::platform::detail::decode_x86_features(r);
  EXPECT_FALSE(no_tile.has_any(CpuFeatureSet{
      CpuFeature::kX86AmxTileHardware,
      CpuFeature::kX86AmxInt8Hardware,
      CpuFeature::kX86AmxBf16Hardware,
      CpuFeature::kX86AmxFp16Hardware}));
}

TEST(Cpu, SyntheticArmLinuxDecodeRequiresParentFeatures) {
  namespace d = hyle::platform::detail;
  const std::uint64_t hw = d::kArmHwcapAes | d::kArmHwcapPmull |
                           d::kArmHwcapFphp | d::kArmHwcapAsimdhp |
                           d::kArmHwcapAsimddp | d::kArmHwcapSve;
  const std::uint64_t hw2 = d::kArmHwcap2Sve2 | d::kArmHwcap2SveAes |
                            d::kArmHwcap2SvePmull | d::kArmHwcap2SveI8mm |
                            d::kArmHwcap2SveF32mm | d::kArmHwcap2SveF64mm |
                            d::kArmHwcap2SveBf16 | d::kArmHwcap2I8mm |
                            d::kArmHwcap2Bf16 | d::kArmHwcap2Sme |
                            d::kArmHwcap2Sme2;
  const auto all = d::decode_arm_linux_features(hw, hw2);
  EXPECT_TRUE(all.has_all(CpuFeatureSet{
      CpuFeature::kArmNeon, CpuFeature::kArmAes, CpuFeature::kArmPmull,
      CpuFeature::kArmDotProd, CpuFeature::kArmFp16, CpuFeature::kArmSve2,
      CpuFeature::kArmSveAes, CpuFeature::kArmSvePmull,
      CpuFeature::kArmI8mm, CpuFeature::kArmBf16, CpuFeature::kArmSme2}));

  const auto malformed = d::decode_arm_linux_features(
      d::kArmHwcapFphp, d::kArmHwcap2Sve2 | d::kArmHwcap2SveAes |
                            d::kArmHwcap2Sme2);
  EXPECT_FALSE(malformed.has_any(CpuFeatureSet{
      CpuFeature::kArmFp16, CpuFeature::kArmSve,
      CpuFeature::kArmSve2, CpuFeature::kArmSveAes,
      CpuFeature::kArmSme2}));
}
