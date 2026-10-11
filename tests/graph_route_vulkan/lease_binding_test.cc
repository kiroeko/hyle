#include <gtest/gtest.h>

#include <array>
#include <span>

#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/vulkan.h"
#include "../compute_route_vulkan/fixture.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace t = vulkan_test;

TEST(VulkanLeaseBinding, OriginalOwnersDirectGraphAndReadonlyImport) {
  auto context = t::context().value();
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto desc = t::dense({2, 2});
  const std::array<float, 4> left_values{1, 2, 3, 4}, right_values{2, 0, 0, 3};
  const std::array<float, 4> expected{2, 6, 6, 12};
  auto left_native = t::upload(context, desc, left_values).value();
  auto right_native = t::upload(context, desc, right_values).value();
  auto left = memory.import_readonly(left_native).value();
  auto right = memory.import_readonly(right_native).value();
  EXPECT_FALSE(left.try_write().has_value());
  auto left_binding = memory.binding(left, desc).value();
  auto right_binding = memory.binding(right, desc).value();
  EXPECT_EQ(left_binding.native_buffer(), left_native.native_buffer());
  EXPECT_EQ(left_binding.native_offset(), left_native.native_offset());
  auto other = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  EXPECT_FALSE(other.binding(left, desc).has_value());
  EXPECT_FALSE(memory.binding(left, t::dense({3, 3})).has_value());
  EXPECT_FALSE(memory.import_readonly(left_native, {.allocator = {}}).has_value());
  auto direct = v::prepare(context, c::MatMul{}, desc, desc).value();
  auto direct_run = direct.create_invocation().value();
  ASSERT_TRUE(direct_run.bind(left_binding, right_binding).ok());
  auto direct_done = direct_run.submit().value();
  ASSERT_TRUE(direct_done.wait().has_value());
  auto direct_result = direct_done.result().value();
  EXPECT_EQ(t::read(context, direct_result->tensors()[0]).value(),
            (std::vector<float>(expected.begin(), expected.end())));

  auto builder = g::GraphBuilder::create().value();
  const auto a = builder.input(desc).value(), b = builder.input(desc).value();
  const auto product = builder.add(c::MatMul{}, a, b).value();
  ASSERT_TRUE(builder.publish(product).ok());
  const auto source_alias = builder.add(c::View::slice(0, 0, 2).value(), a).value();
  ASSERT_TRUE(builder.publish(source_alias).ok());
  auto definition = std::move(builder).finish().value();
  auto program = g::vulkan::compile(context, definition).value();
  auto run = program.create_invocation().value();
  ASSERT_TRUE(run.bind(left_binding, right_binding).ok());
  left_native = {}; right_native = {}; left.reset(); right.reset();
  left_binding = {}; right_binding = {};
  auto done = run.submit().value();
  ASSERT_TRUE(done.wait().has_value());
  auto result = done.result().value();
  EXPECT_EQ(t::read(context, result->tensors()[0]).value(),
            (std::vector<float>(expected.begin(), expected.end())));
  auto alias = result->tensors()[1];
  auto imported_alias = memory.import_readonly(alias).value();
  EXPECT_FALSE(imported_alias.try_write().has_value());
  auto retained = memory.binding(imported_alias, desc).value();
  done = {}; result = {}; run = {}; program = {}; alias = {}; imported_alias.reset();
  direct_done = {}; direct_result = {}; direct_run = {}; direct = {};
  EXPECT_EQ(t::read(context, retained).value(),
            (std::vector<float>(left_values.begin(), left_values.end())));
}
} // namespace
