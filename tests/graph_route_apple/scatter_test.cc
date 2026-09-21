#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "../compute_route_apple/scatter_test_support.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ga = g::apple;

TEST(AppleGraph, ScatterStateFeedsLaterInvocationsAndKeepsOldResults) {
  for (auto type : {mx::float16, mx::bfloat16}) {
    for (const auto &[rows, width] :
         {std::pair{1, 128}, {17, 2048}, {23, 129}, {128, 2048}}) {
      auto context = a::Context::create().value();
      auto builder = g::GraphBuilder::create().value();
      const auto desc = [](c::DataType data_type, auto dimensions) {
        return c::TensorDesc::dense(data_type, dimensions).value();
      };
      const auto state =
          builder
              .input(
                  desc(logical(type), std::array<std::int64_t, 2>{37, width}))
              .value();
      const auto ids = builder
                           .input(desc(c::DataType::kInt64,
                                       std::array<std::int64_t, 1>{rows}))
                           .value();
      const auto updates =
          builder
              .input(
                  desc(logical(type), std::array<std::int64_t, 2>{rows, width}))
              .value();
      const auto scale = builder
                             .constant(desc(c::DataType::kFloat32,
                                            std::array<std::int64_t, 1>{width}))
                             .value();
      const auto changed = builder
                               .add(c::Scatter{.kind = c::ScatterKind::kAdd},
                                    state, ids, updates)
                               .value();
      const auto embedded = builder.add(c::Gather{}, changed, ids).value();
      const auto norm = builder.add(c::RmsNorm{}, embedded, scale).value();
      const auto residual = builder.add(c::Binary{}, norm, embedded).value();
      ASSERT_TRUE(builder.publish(changed).ok());
      ASSERT_TRUE(builder.publish(residual).ok());
      auto graph = std::move(builder).finish().value();
      auto weight_values = pattern(width, 1);
      for (auto &value : weight_values)
        value = 1.0 + value * 0.03;
      const auto weights = make(weight_values, {width}, mx::float32);
      ga::CompileReport report;
      auto program =
          ga::compile(context, graph, std::array{weights}, {.report = &report});
      ASSERT_TRUE(program) << program.error().message();
      ASSERT_EQ(report.outputs.size(), 2U);
      graph = {};
      auto direct = context.compile(
          [=](const auto &input,
              mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
            HYLE_ASSIGN_OR_RETURN(
                auto changed,
                a::lower(c::Operation(c::Scatter{.kind = c::ScatterKind::kAdd}),
                         input, stream));
            HYLE_ASSIGN_OR_RETURN(auto embedded,
                                  a::lower(c::Operation(c::Gather{}),
                                           std::array{changed[0], input[1]},
                                           stream));
            HYLE_ASSIGN_OR_RETURN(
                auto norm, a::lower(c::Operation(c::RmsNorm{}),
                                    std::array{embedded[0], weights}, stream));
            HYLE_ASSIGN_OR_RETURN(auto residual,
                                  a::lower(c::Operation(c::Binary{}),
                                           std::array{norm[0], embedded[0]},
                                           stream));
            return std::vector{changed[0], residual[0]};
          },
          std::array{a::InputSpec{{37, width}, type},
                     a::InputSpec{{rows}, mx::int64},
                     a::InputSpec{{rows, width}, type}});
      ASSERT_TRUE(direct) << direct.error().message();
      const auto make_ids = [&](int phase) {
        std::vector<std::int64_t> ids(rows);
        for (int row = 0; row < rows; ++row)
          ids[row] = (row * 7 + phase) % 37;
        return ids;
      };
      auto initial = make(pattern(37 * width), {37, width}, type);
      const auto initial_bits = bits(initial);
      auto first_ids = make_ids(0);
      const auto first_updates =
          make(pattern(rows * width, 1), {rows, width}, type);
      const auto first_indices = mx::array(first_ids.data(), {rows}, mx::int64);
      auto first = program->create_invocation().value();
      ASSERT_TRUE(
          first.bind(std::array{initial, first_indices, first_updates}).ok());
      auto first_work = first.submit().value();
      auto retained = first_work.wait().value();
      const auto old_state = bits(retained.get()->arrays()[0]);
      const auto old_result = bits(retained.get()->arrays()[1]);
      auto expected_state = scatter_sum_reference(
          values(initial), values(first_updates), first_ids, initial.shape(), 0,
          type);
      expect(retained.get()->arrays()[0], expected_state);
      auto current = retained.get()->arrays()[0];
      auto next = program->create_invocation().value();
      auto direct_slot = direct->create_invocation().value();
      const auto scale_values = values(weights);
      for (int phase = 1; phase <= 3; ++phase) {
        const auto logical_ids = make_ids(phase);
        const auto native_ids =
            mx::array(logical_ids.data(), {rows}, mx::int64);
        const auto native_updates =
            make(pattern(rows * width, phase + 1), {rows, width}, type);
        const std::array input{current, native_ids, native_updates};
        const auto input_bits = bits(current),
                   update_bits = bits(native_updates);
        ASSERT_TRUE(next.bind(input).ok());
        auto work = next.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        ASSERT_EQ(result->get()->arrays().size(), 2U);
        expected_state =
            scatter_sum_reference(values(current), values(native_updates),
                                  logical_ids, current.shape(), 0, type);
        ASSERT_TRUE(direct_slot.bind(input).ok());
        auto direct_work = direct_slot.submit().value();
        auto direct_result = direct_work.wait().value();
        const auto check_chain = [&](std::span<const mx::array> arrays) {
          // Both public paths must satisfy the independent reduction oracle.
          // Native parallel addition need not reproduce another invocation's
          // bits. Downstream operations consume this invocation's stored state.
          expect(arrays[0], expected_state);
          const auto observed_state = values(arrays[0]);
          std::vector<double> expected_residual(
              static_cast<std::size_t>(rows * width));
          for (int row = 0; row < rows; ++row) {
            const auto offset = logical_ids[row] * width;
            double square = 0;
            for (int col = 0; col < width; ++col)
              square +=
                  observed_state[offset + col] * observed_state[offset + col];
            const double inverse =
                1 / std::sqrt(square / width + double(1e-5F));
            for (int col = 0; col < width; ++col) {
              const auto x = observed_state[offset + col];
              expected_residual[row * width + col] =
                  rounded(x * inverse * scale_values[col], type) + x;
            }
          }
          expect(arrays[1], expected_residual);
        };
        check_chain(result->get()->arrays());
        check_chain(direct_result.get()->arrays());
        EXPECT_EQ(bits(current), input_bits);
        EXPECT_EQ(bits(native_updates), update_bits);
        current = result->get()->arrays()[0]; // Real native state feedback.
        EXPECT_EQ(bits(retained.get()->arrays()[0]), old_state);
        EXPECT_EQ(bits(retained.get()->arrays()[1]), old_result);
      }
      const auto current_bits = bits(current);
      auto bad_ids = make_ids(1);
      bad_ids.back() = 37;
      const auto bad_indices = mx::array(bad_ids.data(), {rows}, mx::int64);
      auto failed = program->create_invocation().value();
      ASSERT_TRUE(
          failed.bind(std::array{current, bad_indices, first_updates}).ok());
      auto bad_work = failed.submit().value();
      auto bad_result = bad_work.wait();
      ASSERT_FALSE(bad_result);
      EXPECT_TRUE(bad_result.error().IsOutOfRange());
      EXPECT_FALSE(bad_work.result());
      EXPECT_FALSE(failed.submit());
      EXPECT_EQ(bits(current), current_bits);
      auto recovered = program->create_invocation().value();
      ASSERT_TRUE(
          recovered.bind(std::array{current, first_indices, first_updates})
              .ok());
      auto recovery = recovered.submit().value();
      EXPECT_TRUE(recovery.wait());
      *program = {};
      *direct = {};
      EXPECT_TRUE(context.close().ok());
      EXPECT_EQ(bits(initial), initial_bits);
      EXPECT_EQ(bits(retained.get()->arrays()[0]), old_state);
      EXPECT_EQ(bits(retained.get()->arrays()[1]), old_result);
    }
  }
}
} // namespace
