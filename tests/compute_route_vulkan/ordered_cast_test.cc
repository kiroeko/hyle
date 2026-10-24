#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "fixture.h"
#include "../../src/compute_route_vulkan/vulkan_carrier.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;

struct Boundary final {
  std::uint32_t input;
  std::uint16_t expected;
};
// Explicit IEEE encodings avoid host floating operations on signaling NaNs.
// Expected BF16 values are independent boundary facts, not the shader formula.
constexpr std::array<Boundary, 19> boundaries{{
    {0x00000000, 0x0000}, // +0
    {0x80000000, 0x8000}, // -0
    {0x00000001, 0x0000}, // F32 subnormal below BF16 range
    {0x80000001, 0x8000},
    {0x00008000, 0x0000}, // Halfway to the first BF16 subnormal: even zero
    {0x00008001, 0x0001},
    {0x00010000, 0x0001}, // Smallest positive BF16 subnormal
    {0x00018000, 0x0002}, // Odd BF16 subnormal rounds up at the tie
    {0x3f808000, 0x3f80}, // Positive even tie
    {0x3f818000, 0x3f82}, // Positive odd tie
    {0xbf808000, 0xbf80}, // Negative even tie
    {0xbf818000, 0xbf82}, // Negative odd tie
    {0x3f80ffff, 0x3f81},
    {0x7f7fffff, 0x7f80}, // Finite overflow rounds to infinity
    {0x7f800000, 0x7f80},
    {0xff800000, 0xff80},
    {0x7fc12345, 0x7fc1}, // Quiet NaN retains its high payload
    {0x7f812345, 0x7fc1}, // Signaling NaN is quieted
    {0xff812345, 0xffc1},
}};

d::Plan cast_plan(const c::TensorDesc &input) {
  const auto output = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(boundaries.size())}).value();
  d::Plan plan;
  plan.values = {{input, {}}, {output, {}}};
  plan.inputs = {0};
  plan.outputs = {1};
  plan.steps.push_back({c::Operation(c::Cast{c::DataType::kBFloat16}),
      {0}, {1}, {}, {1, 1, 1}, {}});
  return plan;
}

TEST(VulkanOrderedCast, RuntimeQueuePreservesBfloatBoundariesThroughStridedOffset) {
  auto context = vulkan_test::context().value();
  std::array<std::uint32_t, 1 + 2 * boundaries.size()> backing;
  backing.fill(0x7fc0dead);
  for (std::size_t i = 0; i < boundaries.size(); ++i)
    backing[1 + 2 * i] = boundaries[i].input;
  const auto backing_description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(backing.size())}).value();
  const auto root = context.upload(backing_description, std::as_bytes(std::span(backing))).value();
  const auto slice = c::View::slice(0, 1, static_cast<std::int64_t>(boundaries.size()), 2).value();
  const auto input = root.view(slice).value();
  ASSERT_EQ(input.native_buffer(), root.native_buffer());
  ASSERT_EQ(input.native_offset(), root.native_offset() + sizeof(std::uint32_t));
  ASSERT_EQ(input.description().byte_strides()[0], 2 * sizeof(std::uint32_t));

  auto program = d::prepare_plan(d::Access::runtime(context), cast_plan(input.description()),
      d::PlanQueuePolicy::kRuntimeQueueOnly);
  ASSERT_TRUE(program) << program.error().message();
  const auto &dispatches = d::Access::program(*program)->dispatches;
  ASSERT_FALSE(dispatches.empty());
  for (const auto &dispatch : dispatches) {
    EXPECT_FALSE(dispatch.iree.has_value());
    EXPECT_FALSE(dispatch.region);
    EXPECT_TRUE(dispatch.pipeline);
  }
  auto slot = program->create_invocation().value();
  const auto &run = *d::Access::invocation(slot)->run;
  EXPECT_EQ(run.runtime, d::Access::runtime(context));
  EXPECT_FALSE(run.async);
  EXPECT_FALSE(run.needs_drain);
  for (const auto &stage : run.stages) EXPECT_FALSE(stage.iree);
  ASSERT_TRUE(slot.bind(input).ok());
  auto work = slot.submit().value();
  auto result = work.wait();
  ASSERT_TRUE(result) << result.error().message();
  const auto &output = result->get()->tensors()[0];
  std::array<std::uint16_t, boundaries.size()> actual{};
  ASSERT_TRUE(context.download(output, std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(actual[i], boundaries[i].expected) << "boundary " << i;
  EXPECT_FALSE(run.async);
  EXPECT_FALSE(run.needs_drain);
}

TEST(VulkanOrderedCast, UnconstrainedPlanKeepsOriginalIreeCastDefault) {
  auto context = vulkan_test::context().value();
  const auto input = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(boundaries.size())}).value();
  // IREE's existing precision domain is dense. The operation, row count and
  // output dtype are unchanged; only the original unconstrained entry is used.
  auto program = d::prepare_plan(d::Access::runtime(context), cast_plan(input));
  ASSERT_TRUE(program) << program.error().message();
  const auto &dispatches = d::Access::program(*program)->dispatches;
  ASSERT_TRUE(std::ranges::any_of(dispatches,
      [](const auto &dispatch) { return dispatch.iree.has_value(); }));
  auto slot = program->create_invocation().value();
  const auto &run = *d::Access::invocation(slot)->run;
  EXPECT_TRUE(run.async);
  EXPECT_TRUE(std::ranges::any_of(run.stages,
      [](const auto &stage) { return bool(stage.iree); }));
}
} // namespace
