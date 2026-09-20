#pragma once

#include "hyle/compute/low_precision.h"
#include "tensor_test_support.h"

namespace vulkan_iree_test {
namespace c = hyle::compute;
namespace v = c::vulkan;

// Independent logical values use four exactly representable E4M3 codes.
// Scale indexing follows the public format contract, not the provider recipe.
struct Float8Inputs final {
  c::TensorDesc activation, weight;
  std::vector<std::byte> a, w;
  std::vector<double> logical_a, logical_w;
  std::int64_t rows, depth, columns;

  Float8Inputs(int m, int k, int n, c::Float8ScaleMode mode, int seed = 0)
      : activation(c::float8_activation_description(m, k, mode).value()),
        weight(c::float8_weight_description(k, n, mode).value()),
        a(*activation.required_bytes()), w(*weight.required_bytes()),
        logical_a(static_cast<std::size_t>(m * k)),
        logical_w(static_cast<std::size_t>(k * n)), rows(m), depth(k),
        columns(n) {
    const bool block = mode == c::Float8ScaleMode::kFloat32Block128;
    const auto group = block ? 128 : 32;
    const std::array<std::uint8_t, 4> codes{0x30, 0x38, 0x40, 0xb8};
    const std::array<double, 4> values{0.5, 1, 2, -1};
    for (bool weights : {false, true}) {
      auto &bytes = weights ? w : a;
      const auto height = weights ? n : m;
      const auto scale_rows = weights && block ? height / 128 : height;
      for (int row = 0; row < scale_rows; ++row)
        for (int col = 0; col < k / group; ++col) {
          const auto index = static_cast<std::size_t>(row * (k / group) + col);
          const auto exponent = (row + 2 * col + seed) % 3 - 1;
          if (block) {
            const auto scale = std::ldexp(1.0F, exponent);
            std::memcpy(bytes.data() + height * k + index * 4, &scale, 4);
          } else {
            bytes[height * k + index] = static_cast<std::byte>(127 + exponent);
          }
        }
      for (int row = 0; row < height; ++row)
        for (int col = 0; col < k; ++col) {
          const auto selected =
              static_cast<std::size_t>((row * 3 + col + seed) % 4);
          bytes[row * k + col] = static_cast<std::byte>(codes[selected]);
          const auto scale_row = weights && block ? row / 128 : row;
          const auto exponent = (scale_row + 2 * (col / group) + seed) % 3 - 1;
          const auto value = std::ldexp(values[selected], exponent);
          if (weights)
            logical_w[col * n + row] = value;
          else
            logical_a[row * k + col] = value;
        }
    }
  }

  std::vector<double> multiply() const {
    std::vector<double> result(static_cast<std::size_t>(rows * columns));
    for (std::int64_t row = 0; row < rows; ++row)
      for (std::int64_t col = 0; col < columns; ++col)
        for (std::int64_t inner = 0; inner < depth; ++inner)
          result[row * columns + col] +=
              logical_a[row * depth + inner] * logical_w[inner * columns + col];
    return result;
  }
};
} // namespace vulkan_iree_test
