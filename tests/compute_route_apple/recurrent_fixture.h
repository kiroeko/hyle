#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "hyle/compute/apple.h"

namespace apple_test {
namespace mx = mlx::core;
struct Recurrent final {
  int tokens;
  int heads;
  int depth;
  bool per_channel_gate;
  std::array<std::vector<float>, 6> data;
  std::vector<mx::array> inputs;

  Recurrent(int t, int h, int d, int seed = 0, bool kda = false)
      : tokens(t), heads(h), depth(d), per_channel_gate(kda) {
    for (std::size_t i = 0; i < 6; ++i) {
      const mx::Shape shape =
          i == 5 ? mx::Shape{2, h, d, d}
                 : mx::Shape{2, t, h, i < 3 || (i == 3 && kda) ? d : 1};
      std::size_t size = 1;
      for (auto dimension : shape)
        size *= dimension;
      data[i].resize(size);
      for (std::size_t j = 0; j < size; ++j)
        data[i][j] =
            i == 3 ? -0.03F * float(1 + j % 7)
            : i == 4
                ? 0.1F + 0.02F * float(j % 5)
                : float(std::sin(double(j + 13 * i + seed) * 0.17) * 0.025);
      inputs.emplace_back(data[i].data(), shape);
    }
  }

  std::vector<hyle::compute::apple::InputSpec> specs() const {
    std::vector<hyle::compute::apple::InputSpec> result;
    for (const auto &input : inputs)
      result.push_back({input.shape(), input.dtype()});
    return result;
  }

  std::array<std::vector<double>, 2> oracle(float query_scale = 1) const {
    std::array<std::vector<double>, 2> result{
        std::vector<double>(data[2].size()),
        std::vector<double>(data[5].begin(), data[5].end())};
    auto &state = result[1];
    for (int b = 0; b < 2; ++b)
      for (int h = 0; h < heads; ++h)
        for (int t = 0; t < tokens; ++t) {
          const auto token = std::size_t((b * tokens + t) * heads + h);
          const auto matrix = std::size_t(b * heads + h) * depth * depth;
          for (int v = 0; v < depth; ++v) {
            const auto row = matrix + std::size_t(v * depth);
            double projected = 0;
            for (int k = 0; k < depth; ++k) {
              const auto gate = per_channel_gate ? data[3][token * depth + k]
                                                 : data[3][token];
              state[row + k] *= std::exp(double(gate));
              projected += state[row + k] * data[1][token * depth + k];
            }
            const double delta =
                data[4][token] * (data[2][token * depth + v] - projected);
            double score = 0;
            for (int k = 0; k < depth; ++k) {
              state[row + k] += delta * data[1][token * depth + k];
              score += state[row + k] *
                       double(data[0][token * depth + k] * query_scale);
            }
            result[0][token * depth + v] = score / std::sqrt(double(depth));
          }
        }
    return result;
  }
};
} // namespace apple_test
