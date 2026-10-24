#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "hyle/graph/executable.h"

namespace derived_cast_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
inline constexpr std::int64_t rows = 2, width = 37, columns = 7;
inline auto description(c::DataType type, std::int64_t a, std::int64_t b) {
  return c::TensorDesc::dense(type, std::array{a, b}).value();
}
inline std::vector<double> weights(int phase) {
  std::vector<double> result(columns * width);
  for (std::size_t i = 0; i < result.size(); ++i) {
    auto bits = std::bit_cast<std::uint32_t>(float((int((i * 19 + phase * 13) % 47) - 23) * .01371));
    bits += 0x7fff + ((bits >> 16) & 1u);
    result[i] = std::bit_cast<float>(bits & 0xffff0000u);
  }
  return result;
}
inline std::vector<double> inputs(int phase) {
  std::vector<double> result(rows * width);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = float((int((i * 11 + phase * 7) % 43) - 21) * .01913);
  return result;
}
inline std::vector<double> reference(const std::vector<double> &input,
                                     const std::vector<double> &weight) {
  std::vector<double> result(rows * columns);
  for (std::int64_t r = 0; r < rows; ++r) {
    double squares = 0;
    for (std::int64_t k = 0; k < width; ++k) squares += input[r * width + k] * input[r * width + k];
    const double factor = 1 / std::sqrt(squares / width + 1e-5);
    for (std::int64_t n = 0; n < columns; ++n)
      for (std::int64_t k = 0; k < width; ++k)
        result[r * columns + n] += input[r * width + k] * factor * weight[n * width + k];
  }
  return result;
}
struct Fixture {
  g::Graph graph;
  explicit Fixture(int mode) {
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(description(c::DataType::kFloat32, rows, width)).value();
    const auto weight_desc = description(c::DataType::kBFloat16, columns, width);
    const auto weight = (mode == 1 ? builder.input(weight_desc) : builder.constant(weight_desc)).value();
    const auto scale_desc = c::TensorDesc::dense(c::DataType::kFloat32, std::array{width}).value();
    const auto scale = builder.constant(scale_desc).value();
    g::NodeId before;
    if (mode == 5)
      before = builder.add_node(c::Operation(c::TopK{.count = 3}), std::array{input}).value();
    const auto norm = builder.add(c::RmsNorm{}, input, scale).value();
    const auto cast = builder.add_node(c::Operation(c::Cast{c::DataType::kFloat32}), std::array{weight}).value();
    const auto wide = builder.result(cast).value();
    const auto view = builder.add(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(), wide).value();
    const auto output = builder.add(c::MatMul{}, norm, view).value();
    EXPECT_TRUE(builder.publish(output).ok());
    if (mode == 2) EXPECT_TRUE(builder.publish(wide).ok());
    if (mode == 3) EXPECT_TRUE(builder.publish(view).ok());
    if (mode == 4) EXPECT_TRUE(builder.publish(builder.add(c::Copy{}, weight).value()).ok());
    if (mode == 5) EXPECT_TRUE(builder.depends_on(cast, before).ok());
    graph = std::move(builder).finish().value();
  }
};
} // namespace derived_cast_test
