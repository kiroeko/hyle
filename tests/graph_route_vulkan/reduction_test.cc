#include <gtest/gtest.h>

#include "../compute_route_vulkan/reduction_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_reduction_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph, HalfL2AndMaskedProbabilityReplayAcrossIndependentSlots) {
  auto context = vulkan_test::context().value();
  for (int rows : {0, 1, 17, 23}) {
    const auto desc = dense(c::DataType::kBFloat16, {rows, 128});
    const auto mask_desc = dense(c::DataType::kBool, {1, 128});
    const auto scalar = dense(c::DataType::kFloat32, {});
    const auto epsilon = make(context, scalar, std::array<double, 1>{1e-6F});
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(desc).value(),
               mask = builder.input(mask_desc).value();
    const auto eps = builder.constant(scalar).value();
    const auto high = builder.add(c::Cast{c::DataType::kFloat32}, x).value();
    const auto squared =
        builder.add(c::Binary{c::BinaryKind::kMultiply}, high, high).value();
    const auto sum =
        builder
            .add(c::Reduce{.axes = c::Axes::from(std::array{-1}).value(),
                           .keep_dimensions = true,
                           .accumulation_type = c::DataType::kFloat32},
                 squared)
            .value();
    const auto shifted = builder.add(c::Binary{}, sum, eps).value();
    const auto inverse =
        builder.add(c::Unary{c::UnaryKind::kRsqrt}, shifted).value();
    const auto normalized =
        builder.add(c::Binary{c::BinaryKind::kMultiply}, high, inverse).value();
    const auto low =
        builder.add(c::Cast{c::DataType::kBFloat16}, normalized).value();
    const c::Softmax operation{.mask = c::MaskKind::kBoolean};
    const auto probability = builder.add(operation, x, mask).value();
    ASSERT_TRUE(builder.publish(low).ok());
    ASSERT_TRUE(builder.publish(probability).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program =
        gv::compile(context, graph, std::array{epsilon}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.native_dispatch_count, rows == 0 ? 0U : 5U);
    EXPECT_EQ(report.managed_region_count, rows == 0 ? 0U : 3U);
    EXPECT_EQ(report.retained_constant_owner_count, rows == 0 ? 0U : 1U);
    graph = {};
    auto first = program->create_invocation().value(),
         second = program->create_invocation().value();
    v::Tensor retained;
    std::vector<std::uint64_t> saved;
    for (int phase = 0; phase < 3; ++phase) {
      const auto data = pattern(*desc.element_count(), phase);
      std::vector<std::uint64_t> flags(128);
      for (std::size_t i = 0; i < flags.size(); ++i)
        flags[i] = (i + phase) % 3 != 1;
      const auto input = make(context, desc, data);
      const auto native_mask =
          context.upload(mask_desc, payload(mask_desc, flags)).value();
      auto &slot = phase == 0 ? first : second;
      ASSERT_TRUE(slot.bind(input, native_mask).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      std::vector<double> l2(data.size());
      for (int row = 0; row < rows; ++row) {
        double square_sum = 0;
        for (int col = 0; col < 128; ++col)
          square_sum += data[row * 128 + col] * data[row * 128 + col];
        for (int col = 0; col < 128; ++col)
          l2[row * 128 + col] =
              data[row * 128 + col] / std::sqrt(square_sum + 1e-6F);
      }
      expect(context, result.get()->tensors()[0], l2);
      expect(context, result.get()->tensors()[1],
             probability_reference(data, desc, operation, mask_values(flags)));
      auto direct = execute(context, c::Operation(operation),
                            std::array{input, native_mask});
      ASSERT_TRUE(direct) << direct.error().message();
      EXPECT_EQ(bits(context, *direct),
                bits(context, result.get()->tensors()[1]));
      if (phase == 0) {
        retained = result.get()->tensors()[0];
        saved = bits(context, retained);
      } else
        EXPECT_EQ(bits(context, retained), saved);
    }
    first = {};
    second = {};
    *program = {};
    EXPECT_EQ(bits(context, retained), saved);
  }
}

TEST(VulkanGraph,
     OddInnerProbabilityUsesBoundedStatisticsAndPreservesOldResults) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat16, {3, 129, 3});
  const auto mask_desc = dense(c::DataType::kBool, {1, 129, 1});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(desc).value(),
             mask = builder.input(mask_desc).value();
  const c::Softmax softmax{.axis = 1, .mask = c::MaskKind::kBoolean};
  const c::Softmax logsoftmax{.axis = 1,
                              .kind = c::SoftmaxKind::kLogSoftmax,
                              .mask = c::MaskKind::kBoolean};
  const auto probability = builder.add(softmax, x, mask).value();
  const auto logarithm = builder.add(logsoftmax, x, mask).value();
  ASSERT_TRUE(builder.publish(probability).ok());
  ASSERT_TRUE(builder.publish(logarithm).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.native_dispatch_count, 4U);
  EXPECT_EQ(report.invocation_storage_block_bytes.size(), 3U);
  const auto output_capacity = (*desc.required_bytes() + 3) & ~std::size_t{3};
  EXPECT_EQ(std::accumulate(report.invocation_storage_block_bytes.begin(),
                            report.invocation_storage_block_bytes.end(),
                            std::size_t{0}),
            2 * output_capacity + 9 * 8);
  graph = {};
  auto first = program->create_invocation().value(),
       second = program->create_invocation().value();
  std::array<v::Tensor, 2> retained;
  std::array<std::vector<std::uint64_t>, 2> saved;
  for (int phase = 0; phase < 3; ++phase) {
    const auto data = pattern(*desc.element_count(), phase);
    const auto input = make(context, desc, data);
    std::vector<std::uint64_t> flags(129);
    for (std::size_t i = 0; i < flags.size(); ++i)
      flags[i] = (i + phase) % 3 != 1;
    const auto native_mask =
        context.upload(mask_desc, payload(mask_desc, flags)).value();
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input, native_mask).ok());
    auto work = slot.submit().value();
    auto result = work.wait().value();
    expect(context, result.get()->tensors()[0],
           probability_reference(data, desc, softmax, mask_values(flags)));
    expect(context, result.get()->tensors()[1],
           probability_reference(data, desc, logsoftmax, mask_values(flags)));
    for (std::size_t i = 0; i < 2; ++i)
      if (phase == 0) {
        retained[i] = result.get()->tensors()[i];
        saved[i] = bits(context, retained[i]);
      } else
        EXPECT_EQ(bits(context, retained[i]), saved[i]);
  }
  first = {};
  second = {};
  *program = {};
  EXPECT_TRUE(context.close().ok());
  for (std::size_t i = 0; i < 2; ++i)
    EXPECT_EQ(bits(context, retained[i]), saved[i]);
}
} // namespace
