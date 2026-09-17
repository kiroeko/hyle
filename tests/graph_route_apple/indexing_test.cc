#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "../compute_route_apple/reduction_test_support.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;

TEST(AppleGraph, CheckedEmbeddingAndScanKeepStateAndRecoverInFreshSlots) {
  for (auto type : {mx::float16, mx::bfloat16}) {
    for (const auto &[rows, width, experts] : {std::tuple{1, 129, 17},
                                               {17, 2048, 128},
                                               {23, 129, 257},
                                               {128, 2048, 129}}) {
      auto context = a::Context::create().value();
      auto builder = g::GraphBuilder::create().value();
      const auto table_desc =
          c::TensorDesc::dense(logical(type),
                               std::array<std::int64_t, 2>{37, width})
              .value();
      const auto ids_desc =
          c::TensorDesc::dense(c::DataType::kInt64,
                               std::array<std::int64_t, 1>{rows})
              .value();
      const auto counts_desc =
          c::TensorDesc::dense(c::DataType::kInt32,
                               std::array<std::int64_t, 1>{experts})
              .value();
      const auto gate_desc =
          c::TensorDesc::dense(c::DataType::kFloat32,
                               std::array<std::int64_t, 1>{rows})
              .value();
      const auto weights_desc =
          c::TensorDesc::dense(c::DataType::kFloat32,
                               std::array<std::int64_t, 1>{width})
              .value();
      const auto ids = builder.input(ids_desc).value();
      const auto counts = builder.input(counts_desc).value();
      const auto gate = builder.input(gate_desc).value();
      const auto table = builder.constant(table_desc).value();
      const auto weight = builder.constant(weights_desc).value();
      const auto embedded = builder.add(c::Gather{}, table, ids).value();
      const auto normalized =
          builder.add(c::RmsNorm{}, embedded, weight).value();
      const auto residual =
          builder.add(c::Binary{}, normalized, embedded).value();
      const auto offsets =
          builder.add(c::Scan{.exclusive = true}, counts).value();
      const auto positive =
          builder.add(c::Unary{.kind = c::UnaryKind::kSoftplus}, gate).value();
      const auto prefix = builder.add(c::Scan{}, positive).value();
      ASSERT_TRUE(builder.publish(residual).ok());
      ASSERT_TRUE(builder.publish(offsets).ok());
      ASSERT_TRUE(builder.publish(prefix).ok());
      auto graph = std::move(builder).finish().value();
      const auto native_table = make(
          pattern(static_cast<std::size_t>(37 * width)), {37, width}, type);
      auto scale = pattern(width, 1);
      for (auto &x : scale)
        x = 1 + x * 0.03;
      const auto native_weight = make(scale, {width}, mx::float32);
      ga::CompileReport report;
      auto program =
          ga::compile(context, graph, std::array{native_table, native_weight},
                      {.report = &report});
      ASSERT_TRUE(program) << program.error().message();
      EXPECT_EQ(report.inputs[0].data_type(), c::DataType::kInt64);
      ASSERT_EQ(report.outputs.size(), 3U);
      graph = {};
      auto direct = context.compile(
          [=](const auto &input,
              mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
            HYLE_ASSIGN_OR_RETURN(
                auto e, a::lower(c::Operation(c::Gather{}),
                                 std::array{native_table, input[0]}, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto n, a::lower(c::Operation(c::RmsNorm{}),
                                 std::array{e[0], native_weight}, stream));
            HYLE_ASSIGN_OR_RETURN(auto r,
                                  a::lower(c::Operation(c::Binary{}),
                                           std::array{n[0], e[0]}, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto o, a::lower(c::Operation(c::Scan{.exclusive = true}),
                                 std::array{input[1]}, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto positive, a::lower(c::Operation(c::Unary{
                                            .kind = c::UnaryKind::kSoftplus}),
                                        std::array{input[2]}, stream));
            HYLE_ASSIGN_OR_RETURN(
                auto p, a::lower(c::Operation(c::Scan{}), positive, stream));
            return std::vector{r[0], o[0], p[0]};
          },
          std::array{a::InputSpec{{rows}, mx::int64},
                     a::InputSpec{{experts}, mx::int32},
                     a::InputSpec{{rows}, mx::float32}});
      ASSERT_TRUE(direct) << direct.error().message();
      const auto arguments = [&](int phase) {
        std::vector<std::int64_t> ids(rows);
        std::vector<std::int32_t> counts(experts);
        for (int i = 0; i < rows; ++i)
          ids[i] = (i * 7 + phase) % 37;
        for (int i = 0; i < experts; ++i)
          counts[i] = (i + phase) % 17;
        return std::array{mx::array(ids.data(), {rows}, mx::int64),
                          mx::array(counts.data(), {experts}, mx::int32),
                          make(pattern(rows, phase), {rows}, mx::float32)};
      };
      auto first = program->create_invocation().value();
      auto next = program->create_invocation().value();
      auto direct_slot = direct->create_invocation().value();
      ASSERT_TRUE(first.bind(arguments(0)).ok());
      auto first_work = first.submit().value();
      auto retained = first_work.wait().value();
      ASSERT_EQ(retained.get()->arrays().size(), 3U);
      const auto old = bits(retained.get()->arrays()[0]);
      const auto old_offsets = bits(retained.get()->arrays()[1]);
      const auto source = values(native_table), weights = values(native_weight);
      for (int phase = 1; phase <= 3; ++phase) {
        const auto input = arguments(phase);
        const auto ids_before = bits(input[0]), counts_before = bits(input[1]);
        ASSERT_TRUE(next.bind(input).ok());
        auto work = next.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        ASSERT_EQ(result->get()->arrays().size(), 3U);
        ASSERT_TRUE(direct_slot.bind(input).ok());
        auto dw = direct_slot.submit().value();
        auto dr = dw.wait().value();
        for (int i = 0; i < 3; ++i)
          EXPECT_EQ(bits(result->get()->arrays()[i]),
                    bits(dr.get()->arrays()[i]));
        std::vector<double> expected(static_cast<std::size_t>(rows * width));
        for (int row = 0; row < rows; ++row) {
          const auto index = (row * 7 + phase) % 37;
          double square = 0;
          for (int col = 0; col < width; ++col)
            square += source[index * width + col] * source[index * width + col];
          const auto inverse =
              1 / std::sqrt(square / width + static_cast<double>(1e-5F));
          for (int col = 0; col < width; ++col) {
            const auto x = source[index * width + col];
            expected[row * width + col] =
                rounded(x * inverse * weights[col], type) + x;
          }
        }
        expect(result->get()->arrays()[0], expected);
        std::vector<double> integer(experts), floating(rows);
        std::int64_t total = 0;
        for (int i = 0; i < experts; ++i) {
          integer[i] = static_cast<double>(total);
          total += (i + phase) % 17;
        }
        EXPECT_EQ(values(result->get()->arrays()[1]), integer);
        const auto gate_values = values(input[2]);
        double sum = 0;
        for (int i = 0; i < rows; ++i) {
          sum += static_cast<float>(
              activation(c::UnaryKind::kSoftplus, gate_values[i]));
          floating[i] = sum;
        }
        expect(result->get()->arrays()[2], floating);
        EXPECT_EQ(bits(input[0]), ids_before);
        EXPECT_EQ(bits(input[1]), counts_before);
        EXPECT_EQ(bits(retained.get()->arrays()[0]), old);
        EXPECT_EQ(bits(retained.get()->arrays()[1]), old_offsets);
      }
      for (bool bad_index : {true, false}) {
        auto input = arguments(1);
        if (bad_index) {
          std::vector<std::int64_t> ids(rows, 0);
          ids.back() = -1;
          input[0] = mx::array(ids.data(), {rows}, mx::int64);
        } else {
          std::vector<std::int32_t> counts(experts, 0);
          counts[0] = std::numeric_limits<std::int32_t>::max();
          counts[1] = 1;
          input[1] = mx::array(counts.data(), {experts}, mx::int32);
        }
        auto failed_slot = program->create_invocation().value();
        ASSERT_TRUE(failed_slot.bind(input).ok());
        auto fw = failed_slot.submit().value();
        auto failure = fw.wait();
        ASSERT_FALSE(failure);
        EXPECT_TRUE(failure.error().IsOutOfRange());
        EXPECT_FALSE(failed_slot.submit());
        EXPECT_EQ(bits(retained.get()->arrays()[0]), old);
        EXPECT_EQ(bits(retained.get()->arrays()[1]), old_offsets);
        auto fresh = program->create_invocation().value();
        ASSERT_TRUE(fresh.bind(arguments(0)).ok());
        auto recovered = fresh.submit().value();
        EXPECT_TRUE(recovered.wait());
      }
      EXPECT_TRUE(context.close().ok());
      EXPECT_EQ(bits(retained.get()->arrays()[0]), old);
    }
  }
}
} // namespace
