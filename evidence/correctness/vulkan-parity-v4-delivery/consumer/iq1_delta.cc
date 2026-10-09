#include <gtest/gtest.h>
#include "tests/compute_route_vulkan/iq1_public_support.h"
#include "hyle/graph/vulkan.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
hyle::Result<v::Program> prepare(const v::Context &context, c::Operation operation,
                                std::span<const c::TensorDesc> descriptions) {
#if defined(V4_GRAPH_IQ1)
  auto builder = hyle::graph::GraphBuilder::create().value();
  std::vector<hyle::graph::ValueId> inputs;
  for (const auto &description : descriptions)
    inputs.push_back(builder.input(description).value());
  const auto node = builder.add_node(operation, inputs).value();
  const auto result = builder.result(node).value();
  EXPECT_TRUE(builder.publish(result).ok());
  return hyle::graph::vulkan::compile(context, std::move(builder).finish().value());
#else
  std::vector<v::Argument> arguments;
  for (const auto &description : descriptions) arguments.emplace_back(description);
  return v::prepare(context, operation, std::span<const v::Argument>(arguments));
#endif
}
TEST(VulkanDelivery, V4OriginalIq1ShapesChangedInputsAndCheckedIds) {
  vulkan_iq1_test::batch(prepare);
}
TEST(VulkanDelivery, V4OriginalIq1OwnedLeaseCancelCloseAndOom) {
  vulkan_iq1_test::lifecycle(prepare);
}
}
