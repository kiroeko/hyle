#pragma once

#include <algorithm>
#include <limits>

#include "tensor_test_support.h"

namespace vulkan_scan_test {
using namespace vulkan_tensor_test;

inline std::vector<double> pattern(std::size_t count, int phase = 0) {
  std::vector<double> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] =
        static_cast<float>((static_cast<int>(i % 31) - 13 + phase) * 0.013F);
  return result;
}

inline std::vector<std::int32_t> integer_pattern(std::size_t count,
                                                 int phase = 0) {
  std::vector<std::int32_t> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = static_cast<std::int32_t>(i % 31) - 15 + phase;
  return result;
}

inline std::vector<std::byte>
integer_payload(const c::TensorDesc &desc,
                std::span<const std::int32_t> input) {
  std::vector<std::byte> result(*desc.required_bytes(), std::byte{0xcd});
  for (std::size_t i = 0; i < input.size(); ++i)
    std::memcpy(result.data() + offset(desc, i), &input[i], sizeof(input[i]));
  return result;
}

inline v::Tensor make_integer(const v::Context &context,
                              const c::TensorDesc &desc,
                              std::span<const std::int32_t> input) {
  return context.upload(desc, integer_payload(desc, input)).value();
}

inline std::vector<std::int32_t> integer_values(const v::Context &context,
                                                const v::Tensor &tensor) {
  std::vector<std::int32_t> result;
  for (const auto word : bits(context, tensor))
    result.push_back(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(word)));
  return result;
}

inline void expect_integer_prefix(const v::Context &context,
                                  const v::Tensor &actual,
                                  const c::TensorDesc &description,
                                  std::span<const std::int32_t> input,
                                  c::Scan scan) {
  const auto observed = integer_values(context, actual);
  ASSERT_EQ(observed.size(), input.size());
  if (observed.empty())
    return;
  const auto axis = static_cast<std::size_t>(
      scan.axis < 0 ? scan.axis + static_cast<std::int32_t>(description.rank())
                    : scan.axis);
  const auto width = static_cast<std::size_t>(description.dimensions()[axis]);
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < description.rank(); ++i)
    inner *= static_cast<std::size_t>(description.dimensions()[i]);
  for (std::size_t row = 0; row < input.size() / width / inner; ++row)
    for (std::size_t column = 0; column < inner; ++column) {
      std::int64_t sum = 0;
      for (std::size_t position = 0; position < width; ++position) {
        const auto index = (row * width + position) * inner + column;
        const auto expected = scan.exclusive ? sum : sum + input[index];
        sum += input[index];
        ASSERT_GE(sum, INT32_MIN) << index;
        ASSERT_LE(sum, INT32_MAX) << index;
        ASSERT_EQ(observed[index], expected) << index;
      }
    }
}

inline void expect_prefix(const v::Context &context, const v::Tensor &actual,
                          const c::TensorDesc &description,
                          std::span<const double> input, c::Scan scan) {
  const auto observed = values(context, actual);
  ASSERT_EQ(observed.size(), input.size());
  const auto axis = static_cast<std::size_t>(
      scan.axis < 0 ? scan.axis + static_cast<std::int32_t>(description.rank())
                    : scan.axis);
  const auto width = static_cast<std::size_t>(description.dimensions()[axis]);
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < description.rank(); ++i)
    inner *= static_cast<std::size_t>(description.dimensions()[i]);
  if (observed.empty())
    return;
  const auto outer = observed.size() / width / inner;
  for (std::size_t row = 0; row < outer; ++row)
    for (std::size_t column = 0; column < inner; ++column) {
      double sum = 0, absolute = 0;
      for (std::size_t position = 0; position < width; ++position) {
        const auto index = (row * width + position) * inner + column;
        const auto value =
            static_cast<double>(static_cast<float>(input[index]));
        const auto expected = scan.exclusive ? sum : sum + value;
        const auto included_absolute =
            scan.exclusive ? absolute : absolute + std::abs(value);
        sum += value;
        absolute += std::abs(value);
        if (std::isnan(expected))
          ASSERT_TRUE(std::isnan(observed[index])) << index;
        else if (std::isinf(expected))
          ASSERT_EQ(observed[index], expected) << index;
        else
          ASSERT_NEAR(observed[index], expected,
                      3e-6 * (std::max)(included_absolute, 1.0))
              << index;
      }
    }
}
} // namespace vulkan_scan_test
