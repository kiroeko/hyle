#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "fixture.h"

namespace vulkan_recurrent_test {
namespace c = hyle::compute;
namespace v = c::vulkan;

struct Fixture final {
  int tokens;
  int heads;
  int depth;
  std::array<std::vector<float>, 6> data;
  std::array<c::TensorDesc, 6> descriptions;

  Fixture(int t, int h, int d, int seed = 0)
      : tokens(t), heads(h), depth(d) {
    for (std::size_t i = 0; i < data.size(); ++i) {
      const std::array<std::int64_t, 4> shape =
          i == 5 ? std::array<std::int64_t, 4>{2, h, d, d}
                 : std::array<std::int64_t, 4>{2, t, h, i < 3 ? d : 1};
      descriptions[i] = c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
      data[i].resize(*descriptions[i].element_count());
      for (std::size_t j = 0; j < data[i].size(); ++j)
        data[i][j] = i == 3 ? -0.03F * float(1 + j % 7)
                     : i == 4 ? 0.1F + 0.02F * float(j % 5)
                              : float(std::sin(double(j + 13 * i + seed) * 0.17) * 0.025);
    }
  }

  std::vector<v::Argument> arguments() const {
    std::vector<v::Argument> result;
    for (const auto &description : descriptions)
      result.emplace_back(description);
    return result;
  }

  std::array<v::Tensor, 6> upload(const v::Context &context) const {
    std::array<v::Tensor, 6> result;
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = vulkan_test::upload(context, descriptions[i], data[i]).value();
    return result;
  }

  // Independent double recurrence, with the same explicit state convention
  // used by the Apple admission oracle. No provider decoder/math is called.
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
          for (int value = 0; value < depth; ++value) {
            const auto row = matrix + std::size_t(value * depth);
            double projected = 0;
            for (int key = 0; key < depth; ++key) {
              state[row + key] *= std::exp(double(data[3][token]));
              projected += state[row + key] * data[1][token * depth + key];
            }
            const double delta = data[4][token] *
                (data[2][token * depth + value] - projected);
            double score = 0;
            for (int key = 0; key < depth; ++key) {
              state[row + key] += delta * data[1][token * depth + key];
              score += state[row + key] *
                  double(data[0][token * depth + key] * query_scale);
            }
            result[0][token * depth + value] = score / std::sqrt(double(depth));
          }
        }
    return result;
  }
};

inline void expect(const v::Context &context, const v::Tensor &tensor,
                   std::span<const double> reference) {
  const auto actual = vulkan_test::read(context, tensor);
  ASSERT_TRUE(actual) << actual.error().message();
  ASSERT_EQ(actual->size(), reference.size());
  double max_error = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    ASSERT_TRUE(std::isfinite((*actual)[i]));
    max_error = (std::max)(max_error, std::abs(double((*actual)[i]) - reference[i]));
  }
  EXPECT_LT(max_error, 3e-6);
}
} // namespace vulkan_recurrent_test
