#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "../compute_route_apple/tensor_test_support.h"
#include "hyle/graph/apple.h"
#include "model_composition_consumer.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;
constexpr float kEpsilon = 1e-6F;

hyle::Result<g::Graph> model_graph(int rows, int width, mx::Dtype type) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(
      auto desc, c::TensorDesc::dense(
                     logical(type), std::array<std::int64_t, 3>{rows, 2, width}));
  HYLE_ASSIGN_OR_RETURN(auto projected_desc,
                        c::TensorDesc::dense(
                            logical(type),
                            std::array<std::int64_t, 3>{rows, 2, 2 * width}));
  HYLE_ASSIGN_OR_RETURN(
      auto weight_desc,
      c::TensorDesc::dense(c::DataType::kFloat32,
                          std::array<std::int64_t, 1>{width}));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto projection, builder.input(projected_desc));
  HYLE_ASSIGN_OR_RETURN(auto weight, builder.constant(weight_desc));
  const auto emit = [&](c::Operation operation, std::span<const g::ValueId> args)
      -> hyle::Result<g::ValueId> {
    HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(operation, args));
    return builder.result(node);
  };
  HYLE_ASSIGN_OR_RETURN(
      auto outputs,
      apple_model_consumer::compose(input, projection, weight, logical(type),
                                    width, kEpsilon, emit));
  for (auto output : outputs)
    HYLE_RETURN_IF_ERROR(builder.publish(output));
  return std::move(builder).finish();
}

hyle::Result<a::Program> direct_program(a::Context &context, int rows, int width,
                                        mx::Dtype type, mx::array weight) {
  return context.compile(
      [=](const auto &input,
          mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
        const auto emit = [&](c::Operation operation,
                              std::span<const mx::array> args)
            -> hyle::Result<mx::array> {
          HYLE_ASSIGN_OR_RETURN(auto output, a::lower(operation, args, stream));
          return std::move(output[0]);
        };
        HYLE_ASSIGN_OR_RETURN(
            auto output,
            apple_model_consumer::compose(input[0], input[1], weight,
                                          logical(type), width, kEpsilon, emit));
        return std::vector<mx::array>(output.begin(), output.end());
      },
      std::array{a::InputSpec{{rows, 2, width}, type},
                 a::InputSpec{{rows, 2, 2 * width}, type}});
}

std::array<std::vector<double>, 3>
reference(const mx::array &input, const mx::array &projection,
          const mx::array &weight) {
  const auto x = values(input), gates = values(projection), gamma = values(weight);
  const auto width = static_cast<std::size_t>(input.shape().back());
  std::array<std::vector<double>, 3> output;
  for (auto &part : output)
    part.resize(x.size());
  for (std::size_t row = 0; row < x.size() / width; ++row) {
    double square = 0;
    for (std::size_t col = 0; col < width; ++col)
      square += x[row * width + col] * x[row * width + col];
    const auto inverse_l2 = 1.0 / std::sqrt(square + double(kEpsilon));
    const auto inverse_rms = 1.0 / std::sqrt(square / width + double(kEpsilon));
    for (std::size_t col = 0; col < width; ++col) {
      const auto index = row * width + col;
      const auto gate = gates[row * width * 2 + width + col];
      const auto normalized =
          rounded(x[index] * inverse_rms * gamma[col], input.dtype());
      output[0][index] = x[index] * inverse_l2;
      output[1][index] = normalized * activation(c::UnaryKind::kSilu, gate);
      output[2][index] =
          x[index] * rounded(activation(c::UnaryKind::kSigmoid, gate), input.dtype());
    }
  }
  return output;
}

std::array<mx::array, 2> inputs(int rows, int width, mx::Dtype type, int phase) {
  auto data = pattern(static_cast<std::size_t>(rows * 2 * width), phase);
  // Near-zero rows distinguish epsilon inside the squared norm from a clamp
  // on the norm itself. Zero rows cover the finite empty-state limit.
  for (int row = 0; row < rows; ++row)
    for (int head = 0; head < 2; ++head)
      for (int col = 0; col < width; ++col) {
        const auto index = (head * width + col) * rows + row;
        if ((row + head + phase) % 3 == 0)
          data[index] = 0;
        else if ((row + head + phase) % 3 == 1)
          data[index] *= 1e-6;
      }
  auto x = ready(mx::transpose(make(data, {2, width, rows}, type), {2, 0, 1},
                                stream()));
  auto projection = ready(mx::transpose(
      make(pattern(static_cast<std::size_t>(rows * 4 * width), phase + 2),
           {2, 2 * width, rows}, type),
      {2, 0, 1}, stream()));
  return {std::move(x), std::move(projection)};
}

TEST(AppleGraph, ModelL2SwishAndAttentionGatesMatchIndependentStorageBoundaries) {
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    for (int width : {1, 17, 128, 129, 256}) {
      for (int rows : {0, 1, 17}) {
        SCOPED_TRACE(::testing::Message() << static_cast<int>(logical(type))
                                          << '/' << rows << '/' << width);
        auto context = a::Context::create().value();
        auto graph = model_graph(rows, width, type);
        ASSERT_TRUE(graph) << graph.error().message();
        auto gamma = pattern(width, 2);
        for (auto &value : gamma)
          value = 1.0 + 0.03 * value;
        const auto weight = make(gamma, {width}, mx::float32);
        const auto weight_bits = bits(weight);
        auto program = ga::compile(context, *graph, std::array{weight});
        ASSERT_TRUE(program) << program.error().message();
        graph = g::Graph{};
        auto direct = direct_program(context, rows, width, type, weight);
        ASSERT_TRUE(direct) << direct.error().message();
        auto initial = inputs(rows, width, type, 0);
        auto first = program->create_invocation().value();
        ASSERT_TRUE(first.bind(initial).ok());
        auto first_work = first.submit().value();
        auto old = first_work.wait().value();
        const auto expected_initial = reference(initial[0], initial[1], weight);
        std::array<std::vector<std::uint64_t>, 3> retained;
        ASSERT_EQ(old.get()->arrays().size(), 3U);
        for (std::size_t i = 0; i < retained.size(); ++i) {
          expect(old.get()->arrays()[i], expected_initial[i]);
          retained[i] = bits(old.get()->arrays()[i]);
        }
        auto second = program->create_invocation().value();
        auto direct_slot = direct->create_invocation().value();
        for (int phase : {1, 2}) {
          const auto arguments = inputs(rows, width, type, phase);
          const std::array original{bits(arguments[0]), bits(arguments[1])};
          const auto expected = reference(arguments[0], arguments[1], weight);
          ASSERT_TRUE(second.bind(arguments).ok());
          auto work = second.submit().value();
          auto output = work.wait().value();
          ASSERT_TRUE(direct_slot.bind(arguments).ok());
          auto direct_work = direct_slot.submit().value();
          auto direct_output = direct_work.wait().value();
          ASSERT_EQ(output.get()->arrays().size(), 3U);
          ASSERT_EQ(direct_output.get()->arrays().size(), 3U);
          for (std::size_t i = 0; i < expected.size(); ++i) {
            expect(output.get()->arrays()[i], expected[i]);
            expect(direct_output.get()->arrays()[i], expected[i]);
            EXPECT_EQ(bits(old.get()->arrays()[i]), retained[i]);
          }
          EXPECT_EQ(bits(arguments[0]), original[0]);
          EXPECT_EQ(bits(arguments[1]), original[1]);
          EXPECT_EQ(bits(weight), weight_bits);
        }
        *program = {};
        *direct = {};
        EXPECT_TRUE(context.close().ok());
        for (std::size_t i = 0; i < retained.size(); ++i)
          EXPECT_EQ(bits(old.get()->arrays()[i]), retained[i]);
        EXPECT_FALSE(second.submit());
      }
    }
  }
}

TEST(AppleGraph, RmsSwishKeepsTheRequiredIntermediateStorageRounding) {
  for (auto type : {mx::float16, mx::bfloat16}) {
    const auto gamma = type == mx::float16 ? 1.0003 : 1.003;
    const auto gate = type == mx::float16 ? 1.0 : 0.5;
    auto context = a::Context::create().value();
    auto graph = model_graph(1, 1, type).value();
    const auto weight = make(std::array{gamma}, {1}, mx::float32);
    const std::array arguments{
        make(std::array{1.0, 1.0}, {1, 2, 1}, type),
        make(std::array{0.0, gate, 0.0, gate}, {1, 2, 2}, type)};
    const auto expected = reference(arguments[0], arguments[1], weight);
    const auto without_storage_boundary = rounded(
        values(weight)[0] / std::sqrt(1.0 + double(kEpsilon)) *
            activation(c::UnaryKind::kSilu, gate),
        type);
    ASSERT_NE(rounded(expected[1][0], type), without_storage_boundary);
    auto program = ga::compile(context, graph, std::array{weight}).value();
    auto direct = direct_program(context, 1, 1, type, weight).value();
    for (auto *prepared : {&program, &direct}) {
      auto slot = prepared->create_invocation().value();
      ASSERT_TRUE(slot.bind(arguments).ok());
      auto work = slot.submit().value();
      auto output = work.wait().value();
      const auto observed = values(output.get()->arrays()[1]);
      for (double value : observed)
        EXPECT_EQ(value, rounded(expected[1][0], type));
    }
  }
}
} // namespace
