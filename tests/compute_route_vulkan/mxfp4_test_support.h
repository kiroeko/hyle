#pragma once

#include "hyle/compute/low_precision.h"
#include "tensor_test_support.h"

namespace vulkan_mxfp4_test {
namespace c = hyle::compute;
namespace v = c::vulkan;

struct Weights final {
  int k, n;
  std::vector<std::uint8_t> packed;
  std::vector<double> logical;
  c::TensorDesc description;

  Weights(int inner, int columns, int seed)
      : k(inner), n(columns), packed(std::size_t(k / 32) * n * 17),
        logical(std::size_t(k) * n),
        description(c::mxfp4_weight_description(k, n).value()) {
    // Independent E2M1 logical values. Packing is test data construction;
    // the oracle consumes these values, never a product dequantizer.
    constexpr std::array levels{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    for (int column = 0; column < n; ++column)
      for (int group = 0; group < k / 32; ++group) {
        const auto block = std::size_t(column * (k / 32) + group) * 17;
        const auto exponent = (column + group + seed) % 5 - 2;
        packed[block] = static_cast<std::uint8_t>(127 + exponent);
        for (int lane = 0; lane < 32; ++lane) {
          const auto code = (column * 11 + group * 5 + lane * 3 + seed) % 16;
          const auto value =
              std::ldexp(levels[code % 8], exponent) * (code >= 8 ? -1 : 1);
          logical[std::size_t(group * 32 + lane) * n + column] = value;
          packed[block + 1 + lane % 16] |=
              static_cast<std::uint8_t>(code << (lane < 16 ? 0 : 4));
        }
      }
  }
  v::Tensor upload(const v::Context &context) const {
    return context.upload(description, std::as_bytes(std::span(packed)))
        .value();
  }
  std::vector<double> multiply(std::span<const double> input, int rows) const {
    std::vector<double> output(std::size_t(rows) * n);
    for (int row = 0; row < rows; ++row)
      for (int column = 0; column < n; ++column) {
        double sum = 0;
        for (int inner = 0; inner < k; ++inner)
          sum += input[std::size_t(row) * k + inner] *
                 logical[std::size_t(inner) * n + column];
        output[std::size_t(row) * n + column] =
            vulkan_tensor_test::rounded(sum, c::DataType::kFloat16);
      }
    return output;
  }
};

inline std::vector<double> activation(int rows, int k, int seed = 0) {
  std::vector<double> values(std::size_t(rows) * k);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = (static_cast<int>((i * 7 + seed) % 17) - 8) / 16.0;
  return values;
}
} // namespace vulkan_mxfp4_test
