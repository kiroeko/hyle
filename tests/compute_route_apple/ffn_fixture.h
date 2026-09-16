#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <mlx/array.h>

namespace apple_test {
namespace mx = mlx::core;

// Independently packed words and scalar affine decoding. This intentionally
// does not use MLX quantize/dequantize/matmul as the numerical oracle.
struct Ffn final {
  static constexpr int k = 64;
  static constexpr int hidden = 64;
  static constexpr int output = 16;
  std::vector<mx::array> constants;
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> down;

  Ffn() : gate(k * hidden), up(k * hidden), down(hidden * output) {
    for (int projection = 0; projection < 2; ++projection) {
      std::vector<std::uint32_t> words(hidden * k / 8, 0);
      std::vector<float> scales(hidden);
      std::vector<float> biases(hidden);
      for (int row = 0; row < hidden; ++row) {
        scales[row] = 0.01f + 0.002f * float(row % 3);
        biases[row] = -0.07f - 0.003f * float(row % 5);
        for (int col = 0; col < k; ++col) {
          const auto q =
              std::uint32_t((row * 5 + col * 3 + projection * 7) % 16);
          words[(row * k + col) / 8] |= q << (4 * (col % 8));
          (projection == 0 ? gate : up)[row * k + col] =
              float(q) * scales[row] + biases[row];
        }
      }
      constants.emplace_back(words.data(), mx::Shape{hidden, k / 8},
                             mx::uint32);
      constants.emplace_back(scales.data(), mx::Shape{hidden, 1}, mx::float32);
      constants.emplace_back(biases.data(), mx::Shape{hidden, 1}, mx::float32);
    }
    for (int i = 0; i < hidden * output; ++i)
      down[i] = float(i % 17 - 8) * 0.015f;
    constants.emplace_back(down.data(), mx::Shape{hidden, output}, mx::float32);
  }

  static std::vector<float> input(int rows, int run) {
    std::vector<float> values(rows * k);
    for (int i = 0; i < rows * k; ++i)
      values[i] = float((i * 7 + run * 3) % 23 - 11) * 0.09f;
    return values;
  }

  std::vector<float> oracle(const std::vector<float> &input, int rows) const {
    std::vector<float> result(rows * output, 0.f);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < output; ++col) {
        double sum = 0;
        for (int h = 0; h < hidden; ++h) {
          double g = 0, u = 0;
          for (int i = 0; i < k; ++i) {
            g += double(input[row * k + i]) * gate[h * k + i];
            u += double(input[row * k + i]) * up[h * k + i];
          }
          sum += (g / (1.0 + std::exp(-g))) * u * down[h * output + col];
        }
        result[row * output + col] = static_cast<float>(sum);
      }
    }
    return result;
  }
};
} // namespace apple_test
