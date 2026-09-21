#include <gtest/gtest.h>

#include "../compute_route_vulkan/tensor_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_tensor_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     TypedResidualMaskReplayMatchesDirectRecordingAndIndependentMath) {
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (const auto &[rows, width] :
         {std::pair{0, 129}, {1, 128}, {17, 2048}, {23, 129}}) {
      auto context = vulkan_test::context().value();
      const auto input_desc = dense(type, {rows, width});
      const auto full_desc = dense(c::DataType::kFloat32, {rows, width});
      const auto mask_desc = dense(c::DataType::kBool, {rows, 1});
      const auto bias_desc = dense(c::DataType::kFloat32, {width});
      std::vector<double> bias_values(width);
      for (int i = 0; i < width; ++i)
        bias_values[i] = (i % 19 - 9) * 0.17;
      const auto bias = make(context, bias_desc, bias_values);
      auto builder = g::GraphBuilder::create().value();
      const auto x = builder.input(input_desc).value();
      const auto mask = builder.input(mask_desc).value();
      const auto b = builder.constant(bias_desc).value();
      const auto f = builder.add(c::Cast{c::DataType::kFloat32}, x).value();
      const auto shifted = builder.add(c::Binary{}, f, b).value();
      const auto positive =
          builder.add(c::Unary{c::UnaryKind::kSoftplus}, shifted).value();
      const auto low = builder.add(c::Cast{type}, positive).value();
      const auto residual = builder.add(c::Binary{}, low, x).value();
      const auto selected = builder.add(c::Select{}, mask, residual, x).value();
      ASSERT_TRUE(builder.publish(selected).ok());
      ASSERT_TRUE(builder.publish(low).ok());
      auto graph = std::move(builder).finish().value();
      gv::CompileReport report;
      auto program =
          gv::compile(context, graph, std::array{bias}, {.report = &report});
      ASSERT_TRUE(program) << program.error().message();
      // F16 casts and arithmetic share one ggml region. BF16 retains two
      // ggml/IREE pairs. Select remains a separate storage operation.
      const bool half = type == c::DataType::kFloat16;
      EXPECT_EQ(report.native_dispatch_count, rows == 0 ? 0U : half ? 2U : 3U);
      EXPECT_EQ(report.managed_region_count, rows == 0 ? 0U : half ? 1U : 4U);
      ASSERT_EQ(report.outputs.size(), 2U);
      graph = {};
      // Equivalent ordinary Direct composition keeps fixed native buffers and
      // records the same six operations once. It is only a correctness oracle.
      const auto direct_input = context.allocate(input_desc).value();
      const auto direct_mask = context.allocate(mask_desc).value();
      const auto casted = context.allocate(full_desc).value();
      const auto sums = context.allocate(full_desc).value();
      const auto soft = context.allocate(full_desc).value();
      const auto rounded_low = context.allocate(input_desc).value();
      const auto added = context.allocate(input_desc).value();
      const auto direct_output = context.allocate(input_desc).value();
      auto cast_program =
          v::prepare(context, c::Cast{c::DataType::kFloat32}, input_desc)
              .value();
      auto sum_program =
          v::prepare(context, c::Binary{}, full_desc, bias).value();
      auto soft_program =
          v::prepare(context, c::Unary{c::UnaryKind::kSoftplus}, full_desc)
              .value();
      auto low_program = v::prepare(context, c::Cast{type}, full_desc).value();
      auto add_program =
          v::prepare(context, c::Binary{}, input_desc, input_desc).value();
      auto select_program =
          v::prepare(context, c::Select{}, mask_desc, input_desc, input_desc)
              .value();
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(recorder
                      .append(cast_program, std::array{direct_input},
                              std::array{casted})
                      .ok());
      ASSERT_TRUE(
          recorder.append(sum_program, std::array{casted}, std::array{sums})
              .ok());
      ASSERT_TRUE(
          recorder.append(soft_program, std::array{sums}, std::array{soft})
              .ok());
      ASSERT_TRUE(
          recorder
              .append(low_program, std::array{soft}, std::array{rounded_low})
              .ok());
      ASSERT_TRUE(recorder
                      .append(add_program,
                              std::array{rounded_low, direct_input},
                              std::array{added})
                      .ok());
      ASSERT_TRUE(recorder
                      .append(select_program,
                              std::array{direct_mask, added, direct_input},
                              std::array{direct_output})
                      .ok());
      auto direct = recorder.finish();
      ASSERT_TRUE(direct) << direct.error().message();
      const auto arguments = [&](int phase) {
        std::vector<double> values(static_cast<std::size_t>(rows * width));
        std::vector<std::uint64_t> flags(rows);
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = (static_cast<int>((i * 7 + phase) % 29) - 14) * 0.13;
        for (int row = 0; row < rows; ++row)
          flags[row] = (row + phase) % 2;
        return std::array{
            make(context, input_desc, values),
            context.upload(mask_desc, payload(mask_desc, flags)).value()};
      };
      auto first = program->create_invocation().value();
      auto next = program->create_invocation().value();
      auto original = arguments(0);
      EXPECT_FALSE(first.bind(original[1], original[0]).ok());
      ASSERT_TRUE(first.bind(original).ok());
      auto first_work = first.submit().value();
      auto old = first_work.wait().value();
      const auto old_bits = bits(context, old.get()->tensors()[0]);
      const auto old_low = bits(context, old.get()->tensors()[1]);
      for (int phase = 1; phase <= 3; ++phase) {
        const auto input = arguments(phase);
        const auto input_before = bytes(context, input[0]);
        ASSERT_TRUE(next.bind(input).ok());
        auto work = next.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        ASSERT_TRUE(context.upload(direct_input, input_before).ok());
        ASSERT_TRUE(context.upload(direct_mask, bytes(context, input[1])).ok());
        auto direct_work = direct->submit().value();
        ASSERT_TRUE(direct_work.wait().ok());
        EXPECT_EQ(bits(context, result->get()->tensors()[0]),
                  bits(context, direct_output));
        EXPECT_EQ(bits(context, result->get()->tensors()[1]),
                  bits(context, rounded_low));
        const auto xs = values(context, input[0]), bs = values(context, bias);
        const auto masks = bits(context, input[1]);
        std::vector<double> expected(xs.size()), expected_low(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) {
          const float shifted_value =
              static_cast<float>(xs[i]) + static_cast<float>(bs[i % width]);
          expected_low[i] =
              rounded(unary(c::UnaryKind::kSoftplus, shifted_value), type);
          expected[i] =
              masks[i / width] ? rounded(expected_low[i] + xs[i], type) : xs[i];
        }
        expect(context, result->get()->tensors()[0], expected);
        expect(context, result->get()->tensors()[1], expected_low);
        EXPECT_EQ(bytes(context, input[0]), input_before);
        EXPECT_EQ(bits(context, old.get()->tensors()[0]), old_bits);
        EXPECT_EQ(bits(context, old.get()->tensors()[1]), old_low);
      }
      {
        auto abandoned = program->create_invocation().value();
        ASSERT_TRUE(abandoned.bind(original).ok());
        auto accepted = abandoned.submit();
        ASSERT_TRUE(accepted);
        abandoned = {};
        *program = {};
      }
      EXPECT_TRUE(context.close().ok());
      EXPECT_EQ(bits(context, old.get()->tensors()[0]), old_bits);
      EXPECT_EQ(bits(context, old.get()->tensors()[1]), old_low);
    }
  }
}
} // namespace
