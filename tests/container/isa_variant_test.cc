#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/platform/cpu.h"
#include "src/container/bitset_ops.h"
#include "src/container/cpu_dispatch_internal.h"

namespace {

namespace detail = hyle::container::detail;
using BinaryFn = void (*)(std::uint64_t*, const std::uint64_t*,
                         std::size_t) noexcept;
using UnaryFn = void (*)(std::uint64_t*, std::size_t) noexcept;
using CountFn = std::size_t (*)(const std::uint64_t*, std::size_t) noexcept;

struct Variant {
  const char* name;
  BinaryFn bit_and;
  BinaryFn bit_or;
  BinaryFn bit_xor;
  BinaryFn bit_and_not;
  UnaryFn bit_not;
  CountFn bit_count;
};

Variant scalar_variant() {
  return {"scalar",
          &detail::bitset_and_scalar,
          &detail::bitset_or_scalar,
          &detail::bitset_xor_scalar,
          &detail::bitset_and_not_scalar,
          &detail::bitset_not_scalar,
          &detail::bitset_count_scalar};
}

std::vector<Variant> supported_variants() {
  std::vector<Variant> result{scalar_variant()};
  const auto& features = hyle::platform::cpu_features();
#if HYLE_ARCH_X64
  if (detail::supports_highway_sse4(features)) {
    result.push_back({"sse4", &detail::bitset_and_sse4,
                      &detail::bitset_or_sse4, &detail::bitset_xor_sse4,
                      &detail::bitset_and_not_sse4, &detail::bitset_not_sse4,
                      &detail::bitset_count_sse4});
  }
  if (detail::supports_highway_avx2(features)) {
    result.push_back({"avx2", &detail::bitset_and_avx2,
                      &detail::bitset_or_avx2, &detail::bitset_xor_avx2,
                      &detail::bitset_and_not_avx2, &detail::bitset_not_avx2,
                      &detail::bitset_count_avx2});
  }
#elif HYLE_ARCH_ARM64
  if (detail::supports_highway_neon(features)) {
    result.push_back({"neon", &detail::bitset_and_neon,
                      &detail::bitset_or_neon, &detail::bitset_xor_neon,
                      &detail::bitset_and_not_neon, &detail::bitset_not_neon,
                      &detail::bitset_count_neon});
  }
#endif
  return result;
}

TEST(ContainerCpuDispatch, GeneratedTuRequirementsAreExplicit) {
  using hyle::platform::CpuFeature;
  EXPECT_TRUE(detail::supports_highway_sse4(
      detail::kHighwaySse4Requirements));
  for (CpuFeature required : {
           CpuFeature::kX86Sse2, CpuFeature::kX86Sse3,
           CpuFeature::kX86Ssse3, CpuFeature::kX86Sse41,
           CpuFeature::kX86Sse42}) {
    auto masked = detail::kHighwaySse4Requirements;
    masked.set(required, false);
    EXPECT_FALSE(detail::supports_highway_sse4(masked));
  }
  EXPECT_TRUE(detail::supports_highway_avx2(
      detail::kHighwayAvx2Requirements));
  EXPECT_TRUE(detail::supports_xxh3_avx2(
      detail::kXxh3Avx2Requirements));
}

TEST(ContainerIsaVariants, BitsetKernelsMatchScalar) {
  std::array<std::uint64_t, 65> left{};
  std::array<std::uint64_t, 65> right{};
  for (std::size_t i = 0; i < left.size(); ++i) {
    left[i] = 0x0123456789abcdefull ^ (i * 0x9e3779b97f4a7c15ull);
    right[i] = 0xfedcba9876543210ull ^ (i * 0xd6e8feb86659fd93ull);
  }

  const Variant scalar = scalar_variant();
  for (const Variant& variant : supported_variants()) {
    SCOPED_TRACE(variant.name);
    for (std::size_t size : {0u, 1u, 3u, 7u, 8u, 9u, 31u, 32u, 33u, 64u,
                             65u}) {
      EXPECT_EQ(variant.bit_count(left.data(), size),
                scalar.bit_count(left.data(), size));
      const auto compare_binary = [&](BinaryFn actual_fn,
                                      BinaryFn expected_fn) {
        auto actual = left;
        auto expected = left;
        actual_fn(actual.data(), right.data(), size);
        expected_fn(expected.data(), right.data(), size);
        EXPECT_EQ(actual, expected);
      };
      compare_binary(variant.bit_and, scalar.bit_and);
      compare_binary(variant.bit_or, scalar.bit_or);
      compare_binary(variant.bit_xor, scalar.bit_xor);
      compare_binary(variant.bit_and_not, scalar.bit_and_not);
      auto actual = left;
      auto expected = left;
      variant.bit_not(actual.data(), size);
      scalar.bit_not(expected.data(), size);
      EXPECT_EQ(actual, expected);
    }
  }
}

}  // namespace
