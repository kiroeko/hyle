#include <gtest/gtest.h>

#include "ggml-cpu/vec.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
float scalar_dot(ggml_bf16_t *a, ggml_bf16_t *b, int length) {
  double sum = 0;
  for (int i = 0; i < length; ++i)
    sum += static_cast<double>(GGML_BF16_TO_FP32(a[i]) * GGML_BF16_TO_FP32(b[i]));
  return static_cast<float>(sum);
}

float native_dot(ggml_bf16_t *a, ggml_bf16_t *b, int length) {
  float result = std::numeric_limits<float>::quiet_NaN();
  ggml_vec_dot_bf16(length, &result, 0, a, 0, b, 0, 1);
  return result;
}

TEST(ArmBf16Dot, OriginalProductsAndWideAccumulatorIncludeTailsAndOffsets) {
  std::array<ggml_bf16_t, 2051> a{}, b{};
  std::uint32_t random = 0x31415927;
  for (auto *values : {&a, &b})
    for (auto &value : *values) {
      random = random * 1664525u + 1013904223u;
      const float v = static_cast<float>(static_cast<int>(random >> 16) - 32768) / 128;
      value = GGML_FP32_TO_BF16(v);
    }
  for (int offset : {0, 1, 3})
    for (int length : {0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 2047, 2048}) {
      SCOPED_TRACE(offset);
      SCOPED_TRACE(length);
      EXPECT_FLOAT_EQ(native_dot(a.data() + offset, b.data() + offset, length),
                      scalar_dot(a.data() + offset, b.data() + offset, length));
    }

  // F32 summation would lose the small terms. Original F64 accumulation keeps
  // them; the new vector path must not replace it with a BF16 hardware dot.
  std::array<ggml_bf16_t, 17> cancellation{}, ones{};
  for (auto &value : ones) value = GGML_FP32_TO_BF16(1.0F);
  cancellation[0] = GGML_FP32_TO_BF16(16777216.0F);
  for (int i = 1; i < 16; ++i) cancellation[i] = GGML_FP32_TO_BF16(1.0F);
  cancellation[16] = GGML_FP32_TO_BF16(-16777216.0F);
  EXPECT_EQ(native_dot(cancellation.data(), ones.data(), 17), 15.0F);
}

TEST(ArmBf16Dot, ExceptionalValuesKeepOriginalFloatProductSemantics) {
  std::array<ggml_bf16_t, 17> a{}, b{};
  for (auto &value : b) value = GGML_FP32_TO_BF16(1.0F);
  for (int offset : {0, 7, 8, 16}) {
    SCOPED_TRACE(offset);
    a.fill({0});
    a[offset] = {0x7f80};
    EXPECT_EQ(native_dot(a.data(), b.data(), 17), std::numeric_limits<float>::infinity());
    a[offset] = {0xff80};
    EXPECT_EQ(native_dot(a.data(), b.data(), 17), -std::numeric_limits<float>::infinity());
    a[offset] = {0x7fc1};
    EXPECT_TRUE(std::isnan(native_dot(a.data(), b.data(), 17)));
    a[offset] = {0x0001};
    EXPECT_EQ(std::bit_cast<std::uint32_t>(native_dot(a.data(), b.data(), 17)),
              std::bit_cast<std::uint32_t>(scalar_dot(a.data(), b.data(), 17)));
    a[offset] = {0x0080};
    b[offset] = GGML_FP32_TO_BF16(0.5F);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(native_dot(a.data(), b.data(), 17)),
              std::bit_cast<std::uint32_t>(scalar_dot(a.data(), b.data(), 17)));
    a[offset] = {0x7f7f};
    b[offset] = GGML_FP32_TO_BF16(2.0F);
    EXPECT_EQ(native_dot(a.data(), b.data(), 17), std::numeric_limits<float>::infinity());
    b[offset] = GGML_FP32_TO_BF16(1.0F);
  }
}
} // namespace
