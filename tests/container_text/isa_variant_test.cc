#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "hyle/platform/cpu.h"
#include "src/container_text/cpu_dispatch_internal.h"
#include "src/container_text/str_search.h"

namespace {

namespace detail = hyle::container::detail;
using StringFn = std::uint32_t (*)(const char*, std::uint32_t, char) noexcept;
using CaseFn = std::uint32_t (*)(const char*, std::uint32_t, char*,
                                std::uint32_t) noexcept;

struct Variant {
  const char* name;
  StringFn count_char;
  StringFn rfind_char;
  CaseFn lower;
  CaseFn upper;
};

Variant scalar_variant() {
  return {"scalar", &detail::str_count_scalar, &detail::str_rfind_char_scalar,
          &detail::str_ascii_lower_scalar, &detail::str_ascii_upper_scalar};
}

std::vector<Variant> supported_variants() {
  std::vector<Variant> result{scalar_variant()};
  const auto& features = hyle::platform::cpu_features();
#if HYLE_ARCH_X64
  if (detail::supports_highway_sse4(features)) {
    result.push_back({"sse4", &detail::str_count_sse4,
                      &detail::str_rfind_char_sse4,
                      &detail::str_ascii_lower_sse4,
                      &detail::str_ascii_upper_sse4});
  }
  if (detail::supports_highway_avx2(features)) {
    result.push_back({"avx2", &detail::str_count_avx2,
                      &detail::str_rfind_char_avx2,
                      &detail::str_ascii_lower_avx2,
                      &detail::str_ascii_upper_avx2});
  }
#elif HYLE_ARCH_ARM64
  if (detail::supports_highway_neon(features)) {
    result.push_back({"neon", &detail::str_count_neon,
                      &detail::str_rfind_char_neon,
                      &detail::str_ascii_lower_neon,
                      &detail::str_ascii_upper_neon});
  }
#endif
  return result;
}

TEST(TextIsaVariants, StringKernelsMatchScalar) {
  std::string input(513, 'a');
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<char>('A' + (i * 17 + i / 7) % 52);
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

}  // namespace
