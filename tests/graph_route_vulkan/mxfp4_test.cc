#include "../compute_route_vulkan/mxfp4_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_mxfp4_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph, OriginalMxfp4RegionChainsResidentWeightsAndPublishesForks) {
  auto context = vulkan_test::context().value();
  Weights first(128, 32, 0), second(32, 16, 2);
  const auto description =
      vulkan_tensor_test::dense(c::DataType::kFloat16, {7, 128});
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto w1 = builder.constant(first.description).value();
  const auto w2 = builder.constant(second.description).value();
  const auto middle = builder.add(c::MatMul{}, input, w1).value();
  const auto output = builder.add(c::MatMul{}, middle, w2).value();
  ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(middle).ok());
  auto graph = std::move(builder).finish().value();
  const std::array constants{first.upload(context), second.upload(context)};
  gv::CompileReport report;
  auto program = gv::compile(context, graph, constants, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 1U);
  EXPECT_NE(report.numerical_policy.find("MXFP4"), std::string::npos);
  graph = {};
  auto slot = program->create_invocation().value();
  const auto values = activation(7, 128);
  auto resident = vulkan_tensor_test::make(context, description, values);
  ASSERT_TRUE(slot.bind(std::array{resident}).ok());
  const auto expected_middle = first.multiply(values, 7);
  const auto expected_output = second.multiply(expected_middle, 7);
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    vulkan_tensor_test::expect(context, result->get()->tensors()[0],
                               expected_output, true);
    vulkan_tensor_test::expect(context, result->get()->tensors()[1],
                               expected_middle, true);
  }
}
} // namespace
