#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "../compute_route_apple/reduction_test_support.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;

TEST(AppleGraph,
     ProbabilitySelectionNormalizationUsesOrdinaryMultiOutputNodes) {
  for (auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    for (const auto &[rows, experts, count] : {std::tuple{1, 128, 6},
                                               {17, 256, 8},
                                               {23, 897, 8},
                                               {129, 129, 6},
                                               {0, 256, 8}}) {
      SCOPED_TRACE(::testing::Message() << static_cast<int>(logical(type))
                                        << '/' << rows << '/' << experts);
      auto context = a::Context::create().value();
      const c::TopK selection{.count = static_cast<std::size_t>(count)};
      const c::Reduce sum{.axes = c::Axes::from(std::array{-1}).value(),
                          .keep_dimensions = true,
                          .accumulation_type = c::DataType::kFloat32};
      auto builder = g::GraphBuilder::create().value();
      const auto input =
          builder
              .input(
                  c::TensorDesc::dense(
                      logical(type), std::array<std::int64_t, 2>{rows, experts})
                      .value())
              .value();
      const auto probability = builder.add(c::Softmax{}, input).value();
      const auto chosen =
          builder.add_node(c::Operation(selection), std::array{probability})
              .value();
      const auto score = builder.result(chosen, 0).value();
      const auto index = builder.result(chosen, 1).value();
      const auto total = builder.add(sum, score).value();
      const auto normalized =
          builder.add(c::Binary{.kind = c::BinaryKind::kDivide}, score, total)
              .value();
      ASSERT_TRUE(builder.publish(normalized).ok());
      ASSERT_TRUE(builder.publish(index).ok());
      ASSERT_TRUE(builder.publish(probability).ok());
      auto graph = std::move(builder).finish().value();
      ga::CompileReport report;
      auto program = ga::compile(context, graph, {}, {.report = &report});
      ASSERT_TRUE(program) << program.error().message();
      ASSERT_EQ(report.outputs.size(), 3U);
      EXPECT_EQ(report.outputs[1].data_type(), c::DataType::kInt32);
      EXPECT_FALSE(report.private_storage_bytes);
      graph = {};
      auto direct = context.compile(
          [selection, sum](const auto &inputs, mx::Stream stream)
              -> hyle::Result<std::vector<mx::array>> {
            HYLE_ASSIGN_OR_RETURN(
                auto p, a::lower(c::Operation(c::Softmax{}), inputs, stream));
            HYLE_ASSIGN_OR_RETURN(auto chosen,
                                  a::lower(c::Operation(selection), p, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto total,
                a::lower(c::Operation(sum), std::array{chosen[0]}, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto normalized,
                a::lower(
                    c::Operation(c::Binary{.kind = c::BinaryKind::kDivide}),
                    std::array{chosen[0], total[0]}, stream));
            return std::vector{normalized[0], chosen[1], p[0]};
          },
          std::array{a::InputSpec{{rows, experts}, type}});
      ASSERT_TRUE(direct) << direct.error().message();
      auto first = program->create_invocation().value();
      auto second = program->create_invocation().value();
      auto direct_slot = direct->create_invocation().value();
      const std::array initial{
          make(pattern(static_cast<std::size_t>(rows * experts)),
               {rows, experts}, type)};
      ASSERT_TRUE(first.bind(initial).ok());
      auto first_work = first.submit().value();
      auto old = first_work.wait().value();
      const auto old_values = bits(old.get()->arrays()[0]);
      const auto old_indices = bits(old.get()->arrays()[1]);
      for (int phase = 1; phase <= 3; ++phase) {
        const std::array arguments{
            make(pattern(static_cast<std::size_t>(rows * experts), phase),
                 {rows, experts}, type)};
        const auto original = bits(arguments[0]);
        ASSERT_TRUE(second.bind(arguments).ok());
        auto work = second.submit().value();
        auto result = work.wait().value();
        ASSERT_TRUE(direct_slot.bind(arguments).ok());
        auto direct_work = direct_slot.submit().value();
        auto direct_result = direct_work.wait().value();
        for (int i = 0; i < 3; ++i)
          EXPECT_EQ(bits(result.get()->arrays()[i]),
                    bits(direct_result.get()->arrays()[i]));
        auto expected = probability_oracle(values(arguments[0]),
                                           arguments[0].shape(), c::Softmax{});
        expect(result.get()->arrays()[2], expected);
        for (auto &x : expected)
          x = rounded(x, type);
        std::vector<double> weights(static_cast<std::size_t>(rows * count));
        const auto selected = values(result.get()->arrays()[1]);
        for (int row = 0; row < rows; ++row) {
          const auto scores =
              std::span<const double>(expected).subspan(row * experts, experts);
          const auto order = top_indices(scores, count, true);
          double total_score = 0;
          for (auto idx : order)
            total_score += scores[idx];
          total_score = rounded(total_score, type);
          for (int i = 0; i < count; ++i) {
            EXPECT_EQ(selected[row * count + i], order[i]);
            weights[row * count + i] = scores[order[i]] / total_score;
          }
        }
        expect(result.get()->arrays()[0], weights);
        EXPECT_EQ(bits(arguments[0]), original);
        EXPECT_EQ(bits(old.get()->arrays()[0]), old_values);
        EXPECT_EQ(bits(old.get()->arrays()[1]), old_indices);
      }
      ASSERT_TRUE(context.close().ok());
      EXPECT_EQ(bits(old.get()->arrays()[0]), old_values);
      EXPECT_EQ(bits(old.get()->arrays()[1]), old_indices);
      EXPECT_FALSE(second.submit());
    }
  }
}
} // namespace
