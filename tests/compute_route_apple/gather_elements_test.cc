#include <gtest/gtest.h>

#include "tensor_test_support.h"
#include "../compute_models/gather_elements_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"

namespace {
using namespace apple_tensor_test;
namespace g = hyle::graph;
namespace ge = gather_elements_test;

TEST(AppleGatherElements, NarrowValuesKeepSubnormalsSignedZeroAndInfinity) {
  auto context = a::Context::create().value();
  const std::array<std::int32_t, 10> positions{9, 2, 7, 0, 5, 8, 1, 6, 3, 4};
  const auto ids = ready(mx::array(positions.data(), {1, 10}, mx::int32));
  for (const auto type : {mx::float16, mx::bfloat16}) {
    SCOPED_TRACE(static_cast<int>(logical(type)));
    const std::uint16_t normal = type == mx::float16 ? 0x0400 : 0x0080;
    const std::uint16_t infinity = type == mx::float16 ? 0x7c00 : 0x7f80;
    const std::array<std::uint16_t, 10> source{
        0, 0x8000, 1, 0x8001, static_cast<std::uint16_t>(normal - 1),
        static_cast<std::uint16_t>((normal - 1) | 0x8000), normal,
        static_cast<std::uint16_t>(normal | 0x8000), infinity,
        static_cast<std::uint16_t>(infinity | 0x8000)};
    const auto input = ready(mx::view(
        mx::array(source.data(), {1, 10}, mx::uint16), type, stream()));
    const auto selected = execute(context, c::model_ops::gather_elements(),
                                  std::array{input, ids});
    ASSERT_TRUE(selected) << selected.error().message();
    const auto observed = bits(*selected);
    ASSERT_EQ(observed.size(), positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
      EXPECT_EQ(observed[i], source[positions[i]]) << i;
  }
}

TEST(AppleGatherElements, IndependentRowsReplayAndPublishedGraphValues) {
  auto context = a::Context::create().value();
  for (const auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    SCOPED_TRACE(static_cast<int>(logical(type)));
    const auto data_desc = c::TensorDesc::dense(logical(type), std::array<std::int64_t, 2>{3, 5}).value();
    const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 2>{3, 7}).value();
    for (const auto bounds : {c::IndexBoundsPolicy::kChecked,
                              c::IndexBoundsPolicy::kUnchecked}) {
      const auto operation = c::model_ops::gather_elements(-1, bounds);
      auto builder = g::GraphBuilder::create().value();
      const auto data = builder.input(data_desc).value();
      const auto ids = builder.input(ids_desc).value();
      const auto node = builder.add_node(operation, std::array{data, ids}).value();
      const auto selected = builder.result(node).value();
      const auto negative = builder.add(c::Unary{c::UnaryKind::kNeg}, selected).value();
      ASSERT_TRUE(builder.publish(selected).ok());
      ASSERT_TRUE(builder.publish(negative).ok());
      const auto graph = std::move(builder).finish().value();
      auto prepared = g::apple::compile(context, graph);
      ASSERT_TRUE(prepared) << prepared.error().message();
      auto slot = prepared->create_invocation().value();
      for (int seed = 0; seed < 2; ++seed) {
        const auto source = ge::data(seed);
        const auto positions = ge::indices(seed);
        const auto expected = ge::reference(source, positions);
        const auto input = make(source, {3, 5}, type);
        const auto index = ready(mx::array(positions.data(), {3, 7}, mx::int32));
        ASSERT_TRUE(slot.bind(std::array{input, index}).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        ASSERT_EQ(result->get()->arrays().size(), 2U);
        const auto observed = values(result->get()->arrays()[0]);
        const auto negated = values(result->get()->arrays()[1]);
        ASSERT_EQ(observed.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(observed[i], expected[i]) << i;
          EXPECT_EQ(negated[i], -expected[i]) << i;
        }
      }
    }
  }
}

TEST(AppleGatherElements, InvalidIdsRetireOnlyFailedSlotAndKeepOldOutput) {
  auto context = a::Context::create().value();
  for (const auto type : {mx::float32, mx::float16, mx::bfloat16}) {
    const auto source = make(ge::data(), {3, 5}, type);
    const auto ids = ge::indices();
    const auto good = ready(mx::array(ids.data(), {3, 7}, mx::int32));
    const std::array specs{a::InputSpec{{3, 5}, type}, a::InputSpec{{3, 7}, mx::int32}};
    auto program = context.prepare(c::model_ops::gather_elements(), specs).value();
    auto retained = program.create_invocation().value();
    ASSERT_TRUE(retained.bind(std::array{source, good}).ok());
    auto old_work = retained.submit().value();
    auto old = old_work.wait().value();
    const auto old_bits = bits(old.get()->arrays()[0]);
    for (const auto invalid : {-1, 5}) {
      auto bad_ids = ids;
      bad_ids.back() = invalid;
      const auto bad = ready(mx::array(bad_ids.data(), {3, 7}, mx::int32));
      auto failed = program.create_invocation().value();
      ASSERT_TRUE(failed.bind(std::array{source, bad}).ok());
      auto work = failed.submit().value();
      auto result = work.wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
      EXPECT_FALSE(failed.submit());
      EXPECT_EQ(bits(old.get()->arrays()[0]), old_bits);
    }
    auto fresh = program.create_invocation().value();
    ASSERT_TRUE(fresh.bind(std::array{source, good}).ok());
    auto work = fresh.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(bits(result->get()->arrays()[0]), old_bits);
  }
}
} // namespace
