#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/native/algorithm.h"

namespace hyle::simd::test {

// Shared by the SVE and RVV multi-width binaries. Keeping n just over one
// runtime vector proves that the same template owns both a full vector and a
// partial tail for every VL/VLEN selected by QEMU or hardware.
inline void CheckNativeRangeAlgorithms(std::uint32_t* values,
                                       std::size_t capacity) {
  const ScalableTag<std::uint32_t> d;
  const std::size_t n = Lanes(d) + 3;
  ASSERT_LE(n, capacity);

  native::generate(values, n, [](auto tag, auto index) {
    return Add(index, Set(tag, 1u));
  });
  native::transform(values, n, [](auto tag, auto value) {
    return Add(value, Set(tag, 7u));
  });
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(values[i], i + 8);

  const auto even = [](auto tag, auto value) {
    return Eq(And(value, Set(tag, 1u)), Zero(tag));
  };
  std::size_t expected = 0;
  for (std::size_t i = 0; i < n; ++i) expected += (values[i] & 1u) == 0;
  EXPECT_EQ(native::count_if(values, n, even), expected);

  const std::size_t written = native::copy_if(values, values, n, even);
  EXPECT_EQ(written, expected);
  for (std::size_t i = 0; i < written; ++i) {
    EXPECT_EQ(values[i], static_cast<std::uint32_t>(8 + 2 * i));
  }
}

}  // namespace hyle::simd::test
