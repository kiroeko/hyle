#include <gtest/gtest.h>

#include <array>
#include <span>

#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace mx = mlx::core;

TEST(AppleLeaseBinding, SharedLeaseDirectGraphAndImportedNativeAlias) {
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  auto pool = c::LeasePool::create(memory.ref(), 1024, {.max_blocks = 2}).value();
  const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  const std::array<float, 4> left_values{1, 2, 3, 4}, right_values{2, 0, 0, 3};
  const std::array<float, 4> expected{2, 6, 6, 12};
  auto left_write = pool.allocate(16).value(), right_write = pool.allocate(16).value();
  ASSERT_TRUE(left_write.view()->write(std::as_bytes(std::span(left_values))).ok());
  ASSERT_TRUE(right_write.view()->write(std::as_bytes(std::span(right_values))).ok());
  auto left = left_write.publish().value(), right = right_write.publish().value();
  auto left_binding = memory.binding(left, desc).value();
  auto right_binding = memory.binding(right, desc).value();
  EXPECT_FALSE(left.try_write().has_value());
  EXPECT_EQ(memory.storage_info(left_binding)->offset_bytes, left.view()->offset());
  EXPECT_FALSE(memory.binding(left, desc, {.allocator = {}}).has_value());
  const auto wrong = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{3, 3}).value();
  EXPECT_FALSE(memory.binding(left, wrong).has_value());
  std::array inputs{left_binding, right_binding};
  auto direct = context.prepare(c::MatMul{}, std::span<const mx::array>(inputs)).value();
  auto direct_run = direct.create_invocation().value();
  ASSERT_TRUE(direct_run.bind(inputs).ok());
  auto direct_done = direct_run.submit().value();
  ASSERT_TRUE(direct_done.wait().has_value());
  auto direct_result = direct_done.result().value();
  for (std::size_t index = 0; index < 4; ++index)
    EXPECT_FLOAT_EQ(direct_result->arrays()[0].data<float>()[index], expected[index]);

  auto builder = g::GraphBuilder::create().value();
  const auto first = builder.input(desc).value(), second = builder.input(desc).value();
  const auto product = builder.add(c::MatMul{}, first, second).value();
  ASSERT_TRUE(builder.publish(product).ok());
  const auto source_alias = builder.add(c::View::slice(0, 0, 2).value(), first).value();
  ASSERT_TRUE(builder.publish(source_alias).ok());
  auto definition = std::move(builder).finish().value();
  auto program = g::apple::compile(context, definition).value();
  auto run = program.create_invocation().value();
  ASSERT_TRUE(run.bind(inputs).ok());
  left.reset(); right.reset(); left_binding = mx::array(0); right_binding = mx::array(0);
  for (auto &input : inputs) input = mx::array(0);
  auto done = run.submit().value();
  ASSERT_TRUE(done.wait().has_value());
  auto result = done.result().value();
  for (std::size_t index = 0; index < 4; ++index)
    EXPECT_FLOAT_EQ(result->arrays()[0].data<float>()[index], expected[index]);
  auto native_alias = result->arrays()[1];
  auto imported = memory.import_readonly(native_alias, desc).value();
  EXPECT_FALSE(imported.try_write().has_value());
  auto retained = memory.binding(imported, desc).value();
  done = {}; result = {}; run = {}; program = {}; native_alias = mx::array(0); imported.reset();
  direct_done = {}; direct_result = {}; direct_run = {}; direct = {};
  for (std::size_t index = 0; index < 4; ++index)
    EXPECT_FLOAT_EQ(retained.data<float>()[index], left_values[index]);
  EXPECT_EQ(pool.stats().live_blocks, 1u);
}
} // namespace
