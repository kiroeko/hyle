#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hyle/platform/cpu.h"
#include "src/container/bitset_ops.h"
#include "src/container/str_search.h"

namespace {

namespace detail = hyle::container::detail;
using hyle::platform::Isa;
using hyle::platform::supports;

using StringFn = std::uint32_t (*)(const char*, std::uint32_t, char) noexcept;
using CaseFn = std::uint32_t (*)(const char*, std::uint32_t, char*,
                                std::uint32_t) noexcept;
using BinaryFn = void (*)(std::uint64_t*, const std::uint64_t*,
                         std::size_t) noexcept;
using UnaryFn = void (*)(std::uint64_t*, std::size_t) noexcept;
using CountFn = std::size_t (*)(const std::uint64_t*, std::size_t) noexcept;

struct Variant {
  const char* name;
  StringFn count_char;
  StringFn rfind_char;
  CaseFn lower;
  CaseFn upper;
  BinaryFn bit_and;
  BinaryFn bit_or;
  BinaryFn bit_xor;
  BinaryFn bit_and_not;
  UnaryFn bit_not;
  CountFn bit_count;
};

Variant scalar_variant() {
  return {"scalar",
          &detail::str_count_scalar,
          &detail::str_rfind_char_scalar,
          &detail::str_ascii_lower_scalar,
          &detail::str_ascii_upper_scalar,
          &detail::bitset_and_scalar,
          &detail::bitset_or_scalar,
          &detail::bitset_xor_scalar,
          &detail::bitset_and_not_scalar,
          &detail::bitset_not_scalar,
          &detail::bitset_count_scalar};
}

std::vector<Variant> supported_variants() {
  std::vector<Variant> result{scalar_variant()};
#if defined(HYLE_ARCH_X64)
  if (supports(Isa::kSse42)) {
    result.push_back({"sse4",
                      &detail::str_count_sse4,
                      &detail::str_rfind_char_sse4,
                      &detail::str_ascii_lower_sse4,
                      &detail::str_ascii_upper_sse4,
                      &detail::bitset_and_sse4,
                      &detail::bitset_or_sse4,
                      &detail::bitset_xor_sse4,
                      &detail::bitset_and_not_sse4,
                      &detail::bitset_not_sse4,
                      &detail::bitset_count_sse4});
  }
  if (supports(Isa::kAvx2)) {
    result.push_back({"avx2",
                      &detail::str_count_avx2,
                      &detail::str_rfind_char_avx2,
                      &detail::str_ascii_lower_avx2,
                      &detail::str_ascii_upper_avx2,
                      &detail::bitset_and_avx2,
                      &detail::bitset_or_avx2,
                      &detail::bitset_xor_avx2,
                      &detail::bitset_and_not_avx2,
                      &detail::bitset_not_avx2,
                      &detail::bitset_count_avx2});
  }
#elif defined(HYLE_ARCH_ARM64)
  if (supports(Isa::kNeon)) {
    result.push_back({"neon",
                      &detail::str_count_neon,
                      &detail::str_rfind_char_neon,
                      &detail::str_ascii_lower_neon,
                      &detail::str_ascii_upper_neon,
                      &detail::bitset_and_neon,
                      &detail::bitset_or_neon,
                      &detail::bitset_xor_neon,
                      &detail::bitset_and_not_neon,
                      &detail::bitset_not_neon,
                      &detail::bitset_count_neon});
  }
#endif
  return result;
}

TEST(ContainerIsaVariants, StringKernelsMatchScalar) {
  std::string input(513, 'a');
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<char>('A' + (i * 17 + i / 7) % 52);
  }
  input[0] = 'q';
  input[31] = 'q';
  input[256] = 'q';
  input.back() = 'q';

  const Variant scalar = scalar_variant();
  for (const Variant& variant : supported_variants()) {
    SCOPED_TRACE(variant.name);
    for (std::uint32_t size : {0u, 1u, 15u, 16u, 17u, 63u, 64u, 65u,
                               255u, 256u, 257u, 513u}) {
      EXPECT_EQ(variant.count_char(input.data(), size, 'q'),
                scalar.count_char(input.data(), size, 'q'));
      EXPECT_EQ(variant.rfind_char(input.data(), size, 'q'),
                scalar.rfind_char(input.data(), size, 'q'));

      std::vector<char> expected(size);
      std::vector<char> actual(size);
      EXPECT_EQ(variant.lower(input.data(), size, actual.data(), size),
                scalar.lower(input.data(), size, expected.data(), size));
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(variant.upper(input.data(), size, actual.data(), size),
                scalar.upper(input.data(), size, expected.data(), size));
      EXPECT_EQ(actual, expected);
    }
  }
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
