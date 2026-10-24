#include <gtest/gtest.h>

#include "tensor_test_support.h"
#include "../compute_models/gather_elements_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_tensor_test;
namespace g = hyle::graph;
namespace ge = gather_elements_test;

TEST(VulkanGatherElements, NarrowValuesKeepSubnormalsSignedZeroAndInfinity) {
  auto context = vulkan_test::context().value();
  const std::array<std::int32_t, 10> positions{9, 2, 7, 0, 5, 8, 1, 6, 3, 4};
  const auto ids_desc = dense(c::DataType::kInt32, {1, 10});
  const auto ids = context.upload(ids_desc, std::as_bytes(std::span(positions))).value();
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const std::uint64_t normal = type == c::DataType::kFloat16 ? 0x0400 : 0x0080;
    const std::uint64_t infinity = type == c::DataType::kFloat16 ? 0x7c00 : 0x7f80;
    const std::array<std::uint64_t, 10> source{
        0, 0x8000, 1, 0x8001, normal - 1, (normal - 1) | 0x8000,
        normal, normal | 0x8000, infinity, infinity | 0x8000};
    const auto desc = dense(type, {1, 10});
    const auto input = context.upload(desc, payload(desc, source)).value();
    const auto selected = execute(context, c::model_ops::gather_elements(),
                                  std::array{input, ids});
    ASSERT_TRUE(selected) << selected.error().message();
    const auto observed = bits(context, *selected);
    ASSERT_EQ(observed.size(), positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
      EXPECT_EQ(observed[i], source[positions[i]]) << i;
  }
}

TEST(VulkanGatherElements, IndependentRowsReplayAndPublishedGraphValues) {
  auto context = vulkan_test::context().value();
  for (const auto type : {c::DataType::kFloat32, c::DataType::kFloat16,
                          c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto data_desc = dense(type, {3, 5});
    const auto ids_desc = dense(c::DataType::kInt32, {3, 7});
    for (const auto bounds : {c::IndexBoundsPolicy::kChecked,
                              c::IndexBoundsPolicy::kUnchecked}) {
      auto builder = g::GraphBuilder::create().value();
      const auto data = builder.input(data_desc).value();
      const auto ids = builder.input(ids_desc).value();
      const auto node = builder.add_node(c::model_ops::gather_elements(-1, bounds),
                                         std::array{data, ids}).value();
      const auto selected = builder.result(node).value();
      const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, selected).value();
      ASSERT_TRUE(builder.publish(selected).ok());
      ASSERT_TRUE(builder.publish(negative).ok());
      const auto graph = std::move(builder).finish().value();
      auto prepared = g::vulkan::compile(context, graph);
      ASSERT_TRUE(prepared) << prepared.error().message();
      auto made = prepared->create_invocation();
      ASSERT_TRUE(made) << made.error().message();
      auto slot = std::move(*made);
      for (int seed = 0; seed < 2; ++seed) {
        const auto source = ge::data(seed);
        const auto positions = ge::indices(seed);
        const auto expected = ge::reference(source, positions);
        const auto input = make(context, data_desc, source);
        const auto index = context.upload(ids_desc, std::as_bytes(std::span(positions))).value();
        ASSERT_TRUE(slot.bind(std::array{input, index}).ok());
        auto submitted = slot.submit();
        ASSERT_TRUE(submitted) << submitted.error().message();
        auto work = std::move(*submitted);
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        ASSERT_EQ(result->get()->tensors().size(), 2U);
        const auto observed = values(context, result->get()->tensors()[0]);
        const auto negated = values(context, result->get()->tensors()[1]);
        ASSERT_EQ(observed.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(observed[i], expected[i]) << i;
          EXPECT_EQ(negated[i], -expected[i]) << i;
        }
      }
    }
  }
}

TEST(VulkanGatherElements, InvalidIdsPreserveRecordedOutputAndRetireOnlyFailedSlot) {
  auto context = vulkan_test::context().value();
  const auto ids_desc = dense(c::DataType::kInt32, {3, 7});
  for (const auto type : {c::DataType::kFloat32, c::DataType::kFloat16,
                          c::DataType::kBFloat16}) {
    const auto data_desc = dense(type, {3, 5});
    const auto source = make(context, data_desc, ge::data());
    auto prepared = v::prepare(context, c::model_ops::gather_elements(),
                               std::array{v::Argument(data_desc), v::Argument(ids_desc)});
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto program = std::move(*prepared);
    const auto output_desc = program.output_descriptions()[0];
    const std::vector<double> sentinel(21, -7.5);
    for (const auto invalid : {-1, 5}) {
      auto positions = ge::indices();
      positions.back() = invalid;
      const auto ids = context.upload(ids_desc, std::as_bytes(std::span(positions))).value();
      const auto output = make(context, output_desc, sentinel);
      const auto old_bits = bits(context, output);
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(recorder.append(program, std::array{source, ids}, std::array{output}).ok());
      auto built = recorder.finish();
      ASSERT_TRUE(built) << built.error().message();
      auto recorded = std::move(*built);
      auto recorded_work = recorded.submit();
      ASSERT_TRUE(recorded_work) << recorded_work.error().message();
      auto work = std::move(*recorded_work);
      EXPECT_TRUE(work.wait().IsOutOfRange());
      EXPECT_EQ(bits(context, output), old_bits);
      EXPECT_FALSE(recorded.submit());
      auto created = program.create_invocation();
      ASSERT_TRUE(created) << created.error().message();
      auto failed = std::move(*created);
      ASSERT_TRUE(failed.bind(std::array{source, ids}).ok());
      auto submitted = failed.submit().value();
      auto result = submitted.wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
      EXPECT_FALSE(failed.submit());
    }
    const auto positions = ge::indices();
    const auto ids = context.upload(ids_desc, std::as_bytes(std::span(positions))).value();
    const auto good = execute(context, c::model_ops::gather_elements(), std::array{source, ids});
    ASSERT_TRUE(good) << good.error().message();
    const auto expected = ge::reference(ge::data(), positions);
    expect(context, *good, expected, true);
  }
}
} // namespace
