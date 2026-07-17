#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/platform.h"
#include "hyle/simd/random.h"
#include "hyle/simd/fat_kernel.h"
#include "hyle/simd/runtime.h"

namespace hs = hyle::simd;

namespace {
int fat_scalar() noexcept { return 0; }
#if defined(HYLE_ARCH_X64)
int fat_sse4() noexcept { return 1; }
int fat_avx2() noexcept { return 2; }
int fat_avx512() noexcept { return 3; }
#elif defined(HYLE_ARCH_ARM64)
int fat_neon() noexcept { return 1; }
int fat_sve() noexcept { return 2; }
#elif defined(HYLE_ARCH_RISCV64)
int fat_rvv() noexcept { return 1; }
#endif
}

TEST(SimdRuntime, TableMetadataAndExactTierLookup) {
  const hs::Runtime& active = hs::runtime();
  EXPECT_EQ(active.table_version, hs::Runtime::kTableVersion);
  EXPECT_EQ(active.table_size, sizeof(hs::Runtime));
  EXPECT_NE(active.ops.dot_f32, nullptr);
  EXPECT_NE(active.ops.dot_bf16_bf16, nullptr);
  EXPECT_NE(active.ops.dot_f32_bf16, nullptr);
  EXPECT_NE(active.ops.dot_f16_f16, nullptr);
  EXPECT_NE(active.ops.dot_i16_i16, nullptr);
  EXPECT_NE(active.random.fill_u64, nullptr);
  const hs::Runtime* exact = hs::runtime_for(active.isa);
  ASSERT_NE(exact, nullptr);
  EXPECT_EQ(active.ops.dot_f32, exact->ops.dot_f32);
  EXPECT_EQ(active.random.fill_u64, exact->random.fill_u64);
#if defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_X64)
  EXPECT_EQ(active.scan.isa, hs::Isa::kScalar);
#else
  EXPECT_EQ(active.scan.isa, active.isa);
#endif

  const hs::Runtime* scalar = hs::runtime_for(hs::Isa::kScalar);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->isa, hs::Isa::kScalar);

#if defined(HYLE_ARCH_X64)
  EXPECT_EQ(hs::runtime_for(hs::Isa::kNeon), nullptr);
#elif defined(HYLE_ARCH_ARM64)
  EXPECT_EQ(hs::runtime_for(hs::Isa::kAvx2), nullptr);
#elif defined(HYLE_ARCH_RISCV64)
  EXPECT_EQ(hs::runtime_for(hs::Isa::kAvx2), nullptr);
  EXPECT_EQ(hs::runtime_for(hs::Isa::kNeon), nullptr);
#endif
}

TEST(SimdRuntime, PublicFatKernelSelectorMatchesRuntimeTier) {
#if defined(HYLE_ARCH_X64)
  const auto fn = hs::select_fat_kernel(&fat_scalar, &fat_sse4, &fat_avx2, &fat_avx512);
  const int expected = hs::runtime().isa == hs::Isa::kAvx512 ? 3
                       : hs::runtime().isa == hs::Isa::kAvx2 ? 2
                       : hs::runtime().isa == hs::Isa::kSse42 ? 1 : 0;
  EXPECT_EQ(fn(), expected);
#elif defined(HYLE_ARCH_ARM64)
  const auto fn = hs::select_fat_kernel(&fat_scalar, &fat_neon, &fat_sve);
  const int expected = hs::runtime().isa == hs::Isa::kSve ? 2
                       : hs::runtime().isa == hs::Isa::kNeon ? 1 : 0;
  EXPECT_EQ(fn(), expected);
#elif defined(HYLE_ARCH_RISCV64)
  const auto fn = hs::select_fat_kernel(&fat_scalar, &fat_rvv);
  EXPECT_EQ(fn(), hs::runtime().isa == hs::Isa::kRvv ? 1 : 0);
#else
  EXPECT_EQ(hs::select_fat_kernel(&fat_scalar)(), 0);
#endif
}

TEST(SimdRuntime, RandomSequenceIgnoresIsaWidth) {
  constexpr std::size_t kCount = 257;
  hs::RandomState active_state(12345);
  hs::RandomState scalar_state(12345);
  std::array<std::uint64_t, kCount> active{};
  std::array<std::uint64_t, kCount> scalar{};

  hs::runtime().random.fill_u64(&active_state, active.data(), active.size());
  hs::runtime_for(hs::Isa::kScalar)->random.fill_u64(&scalar_state, scalar.data(),
                                                      scalar.size());
  EXPECT_EQ(active, scalar);
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
