#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/core/environment.h"
#include "hyle/simd/fat_kernel.h"
#include "hyle/simd/random.h"
#include "hyle/simd/runtime.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/runtime_internal.h"

namespace hs = hyle::simd;

using FatAddOneFn = void (*)(const float *, float *, std::size_t,
                             std::uint32_t *) noexcept;
void hyle_test_fat_add_one_scalar(const float *, float *, std::size_t,
                                  std::uint32_t *) noexcept;
#if HYLE_ARCH_X64
void hyle_test_fat_add_one_sse4(const float *, float *, std::size_t,
                                std::uint32_t *) noexcept;
void hyle_test_fat_add_one_avx2(const float *, float *, std::size_t,
                                std::uint32_t *) noexcept;
void hyle_test_fat_add_one_avx512(const float *, float *, std::size_t,
                                  std::uint32_t *) noexcept;
#elif HYLE_ARCH_ARM64
void hyle_test_fat_add_one_neon(const float *, float *, std::size_t,
                                std::uint32_t *) noexcept;
void hyle_test_fat_add_one_sve(const float *, float *, std::size_t,
                               std::uint32_t *) noexcept;
#elif HYLE_ARCH_RISCV64
void hyle_test_fat_add_one_rvv(const float *, float *, std::size_t,
                               std::uint32_t *) noexcept;
#endif

namespace {
int fat_scalar() noexcept { return 0; }
#if HYLE_ARCH_X64
int fat_sse4() noexcept { return 1; }
int fat_avx2() noexcept { return 2; }
int fat_avx512() noexcept { return 3; }
#elif HYLE_ARCH_ARM64
int fat_neon() noexcept { return 1; }
int fat_sve() noexcept { return 2; }
#elif HYLE_ARCH_RISCV64
int fat_rvv() noexcept { return 1; }
#endif
} // namespace

TEST(SimdRuntime, TableMetadataAndExactTierLookup) {
  const hs::Runtime &active = hs::runtime();
  EXPECT_EQ(hs::selected_target().isa, active.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kOps).isa,
            active.ops_module.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kRandom).isa,
            active.random_module.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kNarrow).isa,
            active.narrow_module.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kSearch).isa,
            active.search_module.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kScan).isa, active.scan.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kMath).isa, active.math.isa);
  EXPECT_EQ(hs::selected_target(hs::KernelFamily::kDtype).features,
            active.dtype.features);
  EXPECT_EQ(active.table_version, hs::Runtime::kTableVersion);
  EXPECT_EQ(active.table_size, sizeof(hs::Runtime));
  EXPECT_NE(active.ops.dot_f32, nullptr);
  EXPECT_NE(active.ops.dot_bf16_bf16, nullptr);
  EXPECT_NE(active.ops.dot_f32_bf16, nullptr);
  EXPECT_NE(active.ops.dot_f16_f16, nullptr);
  EXPECT_NE(active.ops.dot_i16_i16, nullptr);
  EXPECT_NE(active.ops.dot_i8_i8, nullptr);
  EXPECT_NE(active.random.fill_u64, nullptr);
  EXPECT_NE(active.random.fill_f32, nullptr);
  EXPECT_NE(active.random.fill_f64, nullptr);
  EXPECT_NE(active.random.advance, nullptr);
  EXPECT_NE(active.search.count_equal_i32, nullptr);
  EXPECT_NE(active.search.index_of_min_f32, nullptr);
  EXPECT_NE(active.scan_ops.inclusive_scan_u32, nullptr);
  EXPECT_NE(active.scan_ops.exclusive_scan_f64, nullptr);
  EXPECT_NE(active.narrow.count_equal_f16, nullptr);
  EXPECT_NE(active.narrow.count_equal_bf16, nullptr);
  EXPECT_NE(active.narrow.sum_f16, nullptr);
  EXPECT_NE(active.narrow.sum_bf16, nullptr);
  EXPECT_NE(active.dtype_convert.f16_to_f32, nullptr);
  EXPECT_NE(active.dtype_convert.f32_to_f16, nullptr);
  EXPECT_NE(active.dtype_convert.bf16_to_f32, nullptr);
  EXPECT_NE(active.dtype_convert.f32_to_bf16, nullptr);
  EXPECT_NE(active.dtype_convert.f16_to_bf16, nullptr);
  EXPECT_NE(active.dtype_convert.bf16_to_f16, nullptr);
  EXPECT_NE(active.bitpack_ops.pack_u8, nullptr);
  EXPECT_NE(active.bitpack_ops.unpack_u8, nullptr);
  EXPECT_NE(active.bitpack_ops.pack_u16, nullptr);
  EXPECT_NE(active.bitpack_ops.unpack_u16, nullptr);
  EXPECT_NE(active.bitpack_ops.pack_u32, nullptr);
  EXPECT_NE(active.bitpack_ops.unpack_u32, nullptr);
  EXPECT_NE(active.math.features & hs::Runtime::kFeatureMathExactLibmScalar,
            0u);
  EXPECT_NE(active.dtype.features & hs::Runtime::kFeatureDtypeFp8Scalar, 0u);
  if (active.sort.isa == hs::Isa::kSve) {
    EXPECT_NE(active.sort.features & hs::Runtime::kFeatureSortCappedNetwork,
              0u);
  }
  EXPECT_NE(active.bitpack.features & hs::Runtime::kFeatureBitpackLogical128,
            0u);
#if HYLE_ARCH_X64
  if (active.isa == hs::Isa::kAvx512) {
    EXPECT_EQ(active.sort.isa,
              hs::detail::sort_isa_for_product_isa(hs::Isa::kAvx512));
    if (active.sort.isa != active.isa) {
      EXPECT_NE(active.sort.features & hs::Runtime::kFeaturePerOpLowerTier, 0u);
    }
  }
#endif
  const hs::Runtime *exact = hs::runtime_for(active.isa);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(active.ops.dot_f32, exact->ops.dot_f32);
  if ((active.ops_module.features &
       (hs::Runtime::kFeatureAvxVnni | hs::Runtime::kFeatureAvx512Vnni)) != 0) {
    EXPECT_NE(active.ops.dot_i8_i8, exact->ops.dot_i8_i8);
  } else {
    EXPECT_EQ(active.ops.dot_i8_i8, exact->ops.dot_i8_i8);
  }
  EXPECT_EQ(active.random.fill_u64, exact->random.fill_u64);
  EXPECT_EQ(active.random.advance, exact->random.advance);
  EXPECT_EQ(active.bitpack_ops.pack_u16, exact->bitpack_ops.pack_u16);
  const hs::Runtime *scalar = hs::runtime_for(hs::Isa::kScalar);
  if (scalar == nullptr) {
    // A non-scalar exact artifact intentionally contains one canonical tier
    // only.
    EXPECT_NE(active.isa, hs::Isa::kScalar);
    EXPECT_EQ(active.scan.isa, active.isa);
  } else {
    EXPECT_EQ(scalar->isa, hs::Isa::kScalar);
#if HYLE_ARCH_X64 && !defined(HYLE_SIMD_EXACT)
    const auto specialty =
        hs::detail::x86_specialty_support(hyle::platform::cpu_features());
    if (specialty.avx512_vnni) {
      EXPECT_NE(active.ops_module.features & hs::Runtime::kFeatureAvx512Vnni,
                0u);
      EXPECT_EQ(active.ops_module.features & hs::Runtime::kFeatureAvxVnni, 0u);
    } else if (specialty.avx_vnni) {
      EXPECT_NE(active.ops_module.features & hs::Runtime::kFeatureAvxVnni, 0u);
      EXPECT_EQ(active.ops_module.features & hs::Runtime::kFeatureAvx512Vnni,
                0u);
    }
#endif
    EXPECT_EQ(active.scan.isa, active.search_module.isa);
  }

#if HYLE_ARCH_X64
  EXPECT_EQ(hs::runtime_for(hs::Isa::kNeon), nullptr);
  if (const hs::Runtime *sse = hs::runtime_for(hs::Isa::kSse42)) {
    EXPECT_EQ(sse->math.isa, hs::Isa::kScalar);
    EXPECT_EQ(sse->dtype.isa, hs::Isa::kSse42);
  }
#elif HYLE_ARCH_ARM64
  EXPECT_EQ(hs::runtime_for(hs::Isa::kAvx2), nullptr);
#elif HYLE_ARCH_RISCV64
  EXPECT_EQ(hs::runtime_for(hs::Isa::kAvx2), nullptr);
  EXPECT_EQ(hs::runtime_for(hs::Isa::kNeon), nullptr);
#endif
}

#if HYLE_ARCH_X64
TEST(SimdRuntime, SortCompilerTierContract) {
  EXPECT_EQ(hs::detail::sort_isa_for_product_isa(hs::Isa::kAvx2),
            hs::Isa::kAvx2);
#if HYLE_COMPILER_MSVC
  EXPECT_EQ(hs::detail::sort_isa_for_product_isa(hs::Isa::kAvx512),
            hs::Isa::kAvx2);
#else
  // In particular, clang-cl defines _MSC_VER but is not subject to the cl.exe
  // fallback and must retain the supported AVX-512 vqsort implementation.
  EXPECT_EQ(hs::detail::sort_isa_for_product_isa(hs::Isa::kAvx512),
            hs::Isa::kAvx512);
#endif
}
#endif

TEST(SimdRuntime, PublicFatKernelSelectorMatchesRuntimeTier) {
#if HYLE_ARCH_X64
  const auto fn =
      hs::select_fat_kernel(&fat_scalar, &fat_sse4, &fat_avx2, &fat_avx512);
  const int expected = hs::runtime().isa == hs::Isa::kAvx512  ? 3
                       : hs::runtime().isa == hs::Isa::kAvx2  ? 2
                       : hs::runtime().isa == hs::Isa::kSse42 ? 1
                                                              : 0;
  EXPECT_EQ(fn(), expected);
#elif HYLE_ARCH_ARM64
  const auto fn = hs::select_fat_kernel(&fat_scalar, &fat_neon, &fat_sve);
  const int expected = hs::runtime().isa == hs::Isa::kSve    ? 2
                       : hs::runtime().isa == hs::Isa::kNeon ? 1
                                                             : 0;
  EXPECT_EQ(fn(), expected);
#elif HYLE_ARCH_RISCV64
  const auto fn = hs::select_fat_kernel(&fat_scalar, &fat_rvv);
  EXPECT_EQ(fn(), hs::runtime().isa == hs::Isa::kRvv ? 1 : 0);
#else
  EXPECT_EQ(hs::select_fat_kernel(&fat_scalar)(), 0);
#endif
}

TEST(SimdRuntime, GeneratedPublicFatKernelMatchesRuntimeTier) {
  FatAddOneFn fn = nullptr;
#if HYLE_ARCH_X64
  fn = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_test_fat_add_one_scalar, &hyle_test_fat_add_one_sse4,
      &hyle_test_fat_add_one_avx2, &hyle_test_fat_add_one_avx512);
#elif HYLE_ARCH_ARM64
  fn = HYLE_SIMD_SELECT_FAT_KERNEL(&hyle_test_fat_add_one_scalar,
                                   &hyle_test_fat_add_one_neon,
                                   &hyle_test_fat_add_one_sve);
#elif HYLE_ARCH_RISCV64
  fn = HYLE_SIMD_SELECT_FAT_KERNEL(&hyle_test_fat_add_one_scalar,
                                   &hyle_test_fat_add_one_rvv);
#else
  fn = HYLE_SIMD_SELECT_FAT_KERNEL(&hyle_test_fat_add_one_scalar);
#endif
  ASSERT_NE(fn, nullptr);

  std::array<float, 37> src{};
  std::array<float, 37> dst{};
  for (std::size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<float>(i) - 3.25f;
  std::uint32_t selected = ~std::uint32_t{0};
  fn(src.data(), dst.data(), src.size(), &selected);
  for (std::size_t i = 0; i < src.size(); ++i)
    EXPECT_FLOAT_EQ(dst[i], src[i] + 1.0f);
  auto expected_selected = static_cast<std::uint32_t>(hs::runtime().isa);
#if defined(HYLE_SIMD_FAT_AVX512_CAPPED)
  if (hs::runtime().isa == hs::Isa::kAvx512)
    expected_selected = static_cast<std::uint32_t>(hs::Isa::kAvx2);
#endif
  EXPECT_EQ(selected, expected_selected);
}

#if HYLE_ARCH_X64
TEST(SimdRuntime, SpecialtyDispatchRequiresEveryCompiledTuFeature) {
  hyle::platform::CpuFeatures f;
  f.avx2 = true;
  EXPECT_FALSE(hs::detail::x86_specialty_support(f).avx2_int);
  f.f16c = true;
  EXPECT_FALSE(hs::detail::x86_specialty_support(f).f16c);
  f.avx = true;
  EXPECT_TRUE(hs::detail::x86_specialty_support(f).avx2_int);
  EXPECT_TRUE(hs::detail::x86_specialty_support(f).f16c);

  f.avx_vnni = true;
  f.avx2 = false;
  EXPECT_FALSE(hs::detail::x86_specialty_support(f).avx_vnni);
  f.avx2 = true;
  EXPECT_TRUE(hs::detail::x86_specialty_support(f).avx_vnni);

  f.avx512vnni = true;
  f.avx512bf16 = true;
  f.fma = true;
  f.f16c = true;
  f.avx512f = true;
  f.avx512bw = true;
  f.avx512vl = true;
  f.avx512dq = true;
  auto specialty = hs::detail::x86_specialty_support(f);
  EXPECT_TRUE(specialty.avx512_vnni);
  EXPECT_TRUE(specialty.avx512_bf16);

  for (bool hyle::platform::CpuFeatures::*required : {
           &hyle::platform::CpuFeatures::avx,
           &hyle::platform::CpuFeatures::avx2,
           &hyle::platform::CpuFeatures::fma,
           &hyle::platform::CpuFeatures::f16c,
           &hyle::platform::CpuFeatures::avx512f,
           &hyle::platform::CpuFeatures::avx512bw,
           &hyle::platform::CpuFeatures::avx512vl,
           &hyle::platform::CpuFeatures::avx512dq,
       }) {
    auto masked = f;
    masked.*required = false;
    specialty = hs::detail::x86_specialty_support(masked);
    EXPECT_FALSE(specialty.avx512_vnni);
    EXPECT_FALSE(specialty.avx512_bf16);
  }

  // The detector folds AVX10.2 version + OS ZMM state into avx10_2. Current
  // AVX10 supports all vector widths; runtime still insists on the AVX2/YMM
  // prerequisite so a masked VM cannot select the -mavx10.2-512 TU.
  f.avx10_2 = true;
  EXPECT_TRUE(hs::detail::x86_specialty_support(f).avx10_2);
  for (bool hyle::platform::CpuFeatures::*required : {
           &hyle::platform::CpuFeatures::avx,
           &hyle::platform::CpuFeatures::avx2,
           &hyle::platform::CpuFeatures::fma,
           &hyle::platform::CpuFeatures::f16c,
       }) {
    auto masked = f;
    masked.*required = false;
    EXPECT_FALSE(hs::detail::x86_specialty_support(masked).avx10_2);
  }
}

TEST(SimdRuntime, DetectionOnlyAvxTierUsesSse4FatKernel) {
  EXPECT_EQ(hs::detail::select_x86_fat_kernel_for_isa(hs::Isa::kAvx,
                                                      &fat_scalar, &fat_sse4,
                                                      &fat_avx2, &fat_avx512)(),
            1);
  EXPECT_EQ(hs::detail::select_x86_fat_kernel_for_isa(hs::Isa::kSsse3,
                                                      &fat_scalar, &fat_sse4,
                                                      &fat_avx2, &fat_avx512)(),
            0);
}
#endif

TEST(SimdRuntime, RandomSequenceIgnoresIsaWidth) {
  constexpr std::size_t kCount = 257;
  hs::RandomState active_state(12345);
  hs::RandomState scalar_state(12345);
  std::array<std::uint64_t, kCount> active{};
  std::array<std::uint64_t, kCount> scalar{};

  hs::runtime().random.fill_u64(&active_state, active.data(), active.size());
  const hs::Runtime *scalar_runtime = hs::runtime_for(hs::Isa::kScalar);
  if (scalar_runtime != nullptr) {
    scalar_runtime->random.fill_u64(&scalar_state, scalar.data(),
                                    scalar.size());
  } else {
    // Single-leg exact artifacts have no scalar table; determinism remains
    // testable with a second state on the sole compiled tier.
    hs::runtime().random.fill_u64(&scalar_state, scalar.data(), scalar.size());
  }
  EXPECT_EQ(active, scalar);
}

TEST(SimdRuntime, EveryAvailableRandomTierPreservesSequenceAndAdvance) {
  constexpr std::size_t kCount = 257;
  hs::RandomState reference_u64(0x51A7Eull);
  hs::RandomState reference_f32(0x51A7Eull);
  hs::RandomState reference_f64(0x51A7Eull);
  std::array<std::uint64_t, kCount> expected_u64{};
  std::array<float, kCount> expected_f32{};
  std::array<double, kCount> expected_f64{};
  hs::random_fill(reference_u64, expected_u64.data(), expected_u64.size());
  hs::random_fill(reference_f32, expected_f32.data(), expected_f32.size());
  hs::random_fill(reference_f64, expected_f64.data(), expected_f64.size());

  for (hs::Isa isa :
       {hs::Isa::kScalar, hs::Isa::kSse42, hs::Isa::kAvx2, hs::Isa::kAvx512,
        hs::Isa::kNeon, hs::Isa::kSve, hs::Isa::kRvv}) {
    const hs::Runtime *runtime = hs::runtime_for(isa);
    if (runtime == nullptr)
      continue;
    hs::RandomState state_u64(0x51A7Eull);
    hs::RandomState state_f32(0x51A7Eull);
    hs::RandomState state_f64(0x51A7Eull);
    std::array<std::uint64_t, kCount> actual_u64{};
    std::array<float, kCount> actual_f32{};
    std::array<double, kCount> actual_f64{};
    runtime->random.fill_u64(&state_u64, actual_u64.data(), actual_u64.size());
    runtime->random.fill_f32(&state_f32, actual_f32.data(), actual_f32.size());
    runtime->random.fill_f64(&state_f64, actual_f64.data(), actual_f64.size());
    EXPECT_EQ(actual_u64, expected_u64)
        << "tier=" << static_cast<unsigned>(isa);
    EXPECT_EQ(actual_f32, expected_f32)
        << "tier=" << static_cast<unsigned>(isa);
    EXPECT_EQ(actual_f64, expected_f64)
        << "tier=" << static_cast<unsigned>(isa);

    hs::RandomState advanced(0x51A7Eull);
    runtime->random.advance(&advanced, 73);
    std::array<std::uint64_t, 41> after_advance{};
    runtime->random.fill_u64(&advanced, after_advance.data(),
                             after_advance.size());
    EXPECT_TRUE(std::equal(after_advance.begin(), after_advance.end(),
                           expected_u64.begin() + 73))
        << "advance tier=" << static_cast<unsigned>(isa);
  }
}

TEST(SimdRuntime, RandomFillIsChunkBoundaryIndependent) {
  constexpr std::size_t kCount = 79;
  hs::RandomState whole_state(9876);
  hs::RandomState chunked_state(9876);
  std::array<std::uint64_t, kCount> whole{};
  std::array<std::uint64_t, kCount> chunked{};

  hs::random_fill(whole_state, whole.data(), whole.size());
  std::size_t offset = 0;
  for (std::size_t chunk : {std::size_t{1}, std::size_t{7}, std::size_t{8},
                            std::size_t{9}, std::size_t{17}, std::size_t{37}}) {
    hs::random_fill(chunked_state, chunked.data() + offset, chunk);
    offset += chunk;
  }
  ASSERT_EQ(offset, kCount);
  EXPECT_EQ(whole, chunked);
}

TEST(SimdRuntime, RandomFloatAndDoubleAreUnitInterval) {
  hs::RandomState state(42);
  std::array<float, 65> f{};
  std::array<double, 65> d{};
  hs::random_fill(state, f.data(), f.size());
  hs::random_fill(state, d.data(), d.size());
  for (float value : f) {
    EXPECT_GE(value, 0.0f);
    EXPECT_LT(value, 1.0f);
  }
  for (double value : d) {
    EXPECT_GE(value, 0.0);
    EXPECT_LT(value, 1.0);
  }
}

TEST(SimdRuntime, RandomReseedCopyAndSpanPreserveSequence) {
  constexpr std::uint64_t kSeed = 0x123456789ABCDEF0ull;
  hs::RandomState original(kSeed);
  std::array<std::uint64_t, 19> prefix{};
  hs::random_fill(original, std::span(prefix));

  hs::RandomState copied = original;
  std::array<std::uint64_t, 67> from_original{};
  std::array<std::uint64_t, 67> from_copy{};
  hs::random_fill(original, std::span(from_original));
  hs::random_fill(copied, std::span(from_copy));
  EXPECT_EQ(from_original, from_copy);

  original.seed(kSeed);
  std::array<std::uint64_t, 19> restarted{};
  hs::random_fill(original, std::span(restarted));
  EXPECT_EQ(prefix, restarted);
}

TEST(SimdRuntime, RandomAdvanceMatchesDiscard) {
  constexpr std::uint64_t kSeed = 0x5EED1234ull;
  for (std::size_t skip :
       {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{8},
        std::size_t{9}, std::size_t{31}, std::size_t{64}, std::size_t{257}}) {
    hs::RandomState discarded(kSeed);
    hs::RandomState advanced(kSeed);
    std::vector<std::uint64_t> sink(skip);
    hs::random_fill(discarded, std::span(sink));
    advanced.advance(skip);

    std::array<std::uint64_t, 73> expected{};
    std::array<std::uint64_t, 73> actual{};
    hs::random_fill(discarded, std::span(expected));
    hs::random_fill(advanced, std::span(actual));
    EXPECT_EQ(expected, actual) << "skip=" << skip;
  }
}

TEST(SimdRuntime, RandomSplitProducesDeterministicDisjointSegments) {
  constexpr std::uint64_t kSeed = 0xA11CE5EEDull;
  hs::RandomState parent(kSeed);
  hs::RandomState untouched(kSeed);
  hs::RandomState child0 = parent.split();

  std::array<std::uint64_t, 65> child_values{};
  std::array<std::uint64_t, 65> untouched_values{};
  hs::random_fill(child0, std::span(child_values));
  hs::random_fill(untouched, std::span(untouched_values));
  EXPECT_EQ(child_values, untouched_values);

  hs::RandomState child1 = parent.split();
  std::array<std::uint64_t, 65> parent_values{};
  std::array<std::uint64_t, 65> child1_values{};
  hs::random_fill(parent, std::span(parent_values));
  hs::random_fill(child1, std::span(child1_values));
  EXPECT_NE(child_values, child1_values);
  EXPECT_NE(child1_values, parent_values);

  hs::RandomState replay_parent(kSeed);
  (void)replay_parent.split();
  hs::RandomState replay_child1 = replay_parent.split();
  std::array<std::uint64_t, 65> replay_values{};
  hs::random_fill(replay_child1, std::span(replay_values));
  EXPECT_EQ(child1_values, replay_values);
}

TEST(SimdRuntime, RandomFloatMappingMatchesLogicalU64Sequence) {
  constexpr std::size_t kCount = 71;
  hs::RandomState integers_f32(73);
  hs::RandomState floats(73);
  std::array<std::uint64_t, kCount> bits_f32{};
  std::array<float, kCount> values_f32{};
  hs::random_fill(integers_f32, std::span(bits_f32));
  hs::random_fill(floats, std::span(values_f32));
  for (std::size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(values_f32[i],
              static_cast<float>(bits_f32[i] >> 40) * 0x1.0p-24f);
  }

  hs::RandomState integers_f64(91);
  hs::RandomState doubles(91);
  std::array<std::uint64_t, kCount> bits_f64{};
  std::array<double, kCount> values_f64{};
  hs::random_fill(integers_f64, std::span(bits_f64));
  hs::random_fill(doubles, std::span(values_f64));
  for (std::size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(values_f64[i],
              static_cast<double>(bits_f64[i] >> 11) * 0x1.0p-53);
  }
}
