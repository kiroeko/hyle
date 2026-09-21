#include <gtest/gtest.h>

#include "../compute_route_vulkan/reduction_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_reduction_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     MatureHighRankL2RetainsPublishedReductionAndIndependentSlots) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  const auto description = dense(c::DataType::kFloat32, {2, 1, 3, 1, 2, 128});
  const auto scalar = dense(c::DataType::kFloat32, {});
  auto epsilon = make(context, scalar, std::array<double, 1>{1e-6F});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto eps = builder.constant(scalar).value();
  const auto copied = builder.add(c::Copy{}, x).value();
  const auto squared =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, copied, copied).value();
  const c::Reduce operation{.axes = c::Axes::from(std::array{-1}).value(),
                            .keep_dimensions = true,
                            .accumulation_type = c::DataType::kFloat32};
  const auto sum = builder.add(operation, squared).value();
  const auto shifted = builder.add(c::Binary{}, sum, eps).value();
  const auto inverse =
      builder.add(c::Unary{c::UnaryKind::kRsqrt}, shifted).value();
  const auto normalized =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, copied, inverse).value();
  ASSERT_TRUE(builder.publish(sum).ok());
  ASSERT_TRUE(builder.publish(normalized).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{epsilon}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_EQ(report.native_dispatch_count, 1U); // Input origin adapter only.
  EXPECT_FALSE(report.opaque_storage_bytes);
  auto first = program->create_invocation().value();
  auto second = program->create_invocation().value();
  graph = {};
  epsilon = {};
  v::Tensor retained_sum, retained_output;
  std::vector<double> saved_sum, saved_output;
  for (int phase = 0; phase != 3; ++phase) {
    const auto data = pattern(*description.element_count(), phase);
    auto input = make(context, description, data);
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    std::vector<double> squares(data.size()), output(data.size());
    for (std::size_t i = 0; i != data.size(); ++i)
      squares[i] = data[i] * data[i];
    const auto sums = reduce_reference(squares, description, operation);
    for (std::size_t i = 0; i != data.size(); ++i)
      output[i] = data[i] / std::sqrt(sums[i / 128] + 1e-6F);
    expect(context, result->get()->tensors()[0], sums);
    expect(context, result->get()->tensors()[1], output);
    if (phase == 0) {
      retained_sum = result->get()->tensors()[0];
      retained_output = result->get()->tensors()[1];
      saved_sum = sums;
      saved_output = output;
    } else {
      expect(context, retained_sum, saved_sum);
      expect(context, retained_output, saved_output);
    }
  }
  EXPECT_FALSE(first.submit());
  first = {};
  second = {};
  *program = {};
  ASSERT_TRUE(context.close().ok());
  expect(context, retained_sum, saved_sum);
  expect(context, retained_output, saved_output);
  retained_sum = {};
  retained_output = {};
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}
} // namespace
