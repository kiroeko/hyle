#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "../compute_models/row_cache_indexer_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using row_cache_indexer_test::Fixture;

TEST(CpuRowCacheIndexer, SchemaKeepsPackedBankAndBfloatBoundariesExplicit) {
  const Fixture f;
  const auto op = c::model_ops::row_cache_indexer_scores();
  auto inferred = c::infer_single_output(op, f.descriptions);
  ASSERT_TRUE(inferred);
  EXPECT_EQ(inferred->data_type(), c::DataType::kFloat32);
  EXPECT_EQ(inferred->dimensions()[0], f.queries);
  EXPECT_EQ(inferred->dimensions()[1], f.rows);
  for (std::size_t port = 0; port < 4; ++port) {
    auto wrong = f.descriptions;
    wrong[port] = c::TensorDesc::dense(c::DataType::kFloat32,
        wrong[port].dimensions()).value();
    EXPECT_FALSE(c::infer_single_output(op, wrong));
  }
  auto wrong = f.descriptions;
  wrong[1] = c::TensorDesc::dense(c::DataType::kUInt8,
      std::array{f.rows, std::int64_t{584}}).value();
  EXPECT_FALSE(c::infer_single_output(op, wrong));
  EXPECT_TRUE(c::infer_single_output(op, Fixture(0).descriptions));
  EXPECT_TRUE(c::infer_single_output(op, Fixture(5, 0).descriptions));
}

TEST(CpuRowCacheIndexer, DefaultRouteTilesPackedHistoryAndRebindsIndependentSlots) {
#if !defined(HYLE_TEST_TORCH_CPU)
  GTEST_SKIP() << "optional ATen CPU carrier is unavailable";
#else
  Fixture f;
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const auto op = c::model_ops::row_cache_indexer_scores();
  const auto output_desc = c::infer_single_output(op, f.descriptions).value();
  auto prepared = c::prepare(context, op, f.descriptions,
      std::array{output_desc}, m::system_resource()).value();
  EXPECT_EQ(prepared.provider().id, c::native::kProviderId);
  EXPECT_EQ(prepared.runtime_properties().replay_class, c::ProviderReplayClass::kManaged);
  EXPECT_TRUE(prepared.runtime_properties().uses_private_threads);
  EXPECT_FALSE(prepared.capture_compatible());
  std::array lanes{prepared.create_invocation().value(), prepared.create_invocation().value()};
  auto q = half_test::buffer<std::uint16_t>(f.query);
  auto bank = half_test::buffer<std::uint8_t>(f.bank);
  auto weights = half_test::buffer<std::uint16_t>(f.weights);
  auto ranges = half_test::buffer<std::int32_t>(f.ranges);
  const std::array input{
      c::ConstTensorView::create(f.descriptions[0], q.view()).value(),
      c::ConstTensorView::create(f.descriptions[1], bank.view()).value(),
      c::ConstTensorView::create(f.descriptions[2], weights.view()).value(),
      c::ConstTensorView::create(f.descriptions[3], ranges.view()).value()};
  std::array outputs{
      half_test::buffer<float>(std::vector<float>(std::size_t(f.queries * f.rows))),
      half_test::buffer<float>(std::vector<float>(std::size_t(f.queries * f.rows)))};
  std::vector<float> previous;
  for (int phase : {0, 1}) {
    f.update(phase);
    ASSERT_TRUE(q.view().write(std::as_bytes(std::span(f.query))).ok());
    ASSERT_TRUE(bank.view().write(std::as_bytes(std::span(f.bank))).ok());
    ASSERT_TRUE(weights.view().write(std::as_bytes(std::span(f.weights))).ok());
    ASSERT_TRUE(ranges.view().write(std::as_bytes(std::span(f.ranges))).ok());
    const auto expected = f.expected();
    EXPECT_NE(expected, f.expected(false));
    auto output = c::TensorView::create(output_desc, outputs[phase].view()).value();
    auto done = lanes[phase].submit(queue, input, std::array{output});
    ASSERT_TRUE(done);
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(outputs[phase]);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_FLOAT_EQ(actual[i], expected[i]) << phase << ':' << i;
    if (phase == 0) previous = actual;
    else EXPECT_EQ(half_test::read<float>(outputs[0]), previous);
  }
  const auto sentinel = std::vector<float>(std::size_t(f.queries * f.rows), 91.0F);
  // Put the bad range in the final query: earlier queries must not publish.
  for (const auto bad : {std::array<std::int32_t, 2>{-1, 1},
                         std::array<std::int32_t, 2>{4, 3},
                         std::array<std::int32_t, 2>{0, std::int32_t(f.rows + 1)}}) {
    auto invalid = f.ranges;
    invalid[invalid.size() - 2] = bad[0]; invalid.back() = bad[1];
    ASSERT_TRUE(ranges.view().write(std::as_bytes(std::span(invalid))).ok());
    ASSERT_TRUE(outputs[1].view().write(std::as_bytes(std::span(sentinel))).ok());
    auto failed = prepared.create_invocation().value();
    auto done = failed.submit(queue, input,
        std::array{c::TensorView::create(output_desc, outputs[1].view()).value()});
    ASSERT_TRUE(done);
    EXPECT_TRUE(done->wait().IsOutOfRange());
    EXPECT_EQ(half_test::read<float>(outputs[1]), sentinel);
    EXPECT_EQ(half_test::read<float>(outputs[0]), previous);
  }
#endif
}

TEST(CpuRowCacheIndexer, EmptyQueriesAndEmptyHistoryExecuteWithoutPayload) {
#if !defined(HYLE_TEST_TORCH_CPU)
  GTEST_SKIP() << "optional ATen CPU carrier is unavailable";
#else
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  for (const auto shape : {std::array<std::int64_t, 2>{0, 3},
                           std::array<std::int64_t, 2>{5, 0}}) {
    const Fixture f(shape[0], shape[1]);
    const auto op = c::model_ops::row_cache_indexer_scores();
    const auto output_desc = c::infer_single_output(op, f.descriptions).value();
    auto prepared = c::prepare(context, op, f.descriptions,
        std::array{output_desc}, m::system_resource()).value();
    auto q = half_test::buffer<std::uint16_t>(f.query);
    auto bank = half_test::buffer<std::uint8_t>(f.bank);
    auto weights = half_test::buffer<std::uint16_t>(f.weights);
    auto ranges = half_test::buffer<std::int32_t>(f.ranges);
    auto output = half_test::buffer<float>(std::vector<float>{});
    const std::array inputs{
        c::ConstTensorView::create(f.descriptions[0], q.view()).value(),
        c::ConstTensorView::create(f.descriptions[1], bank.view()).value(),
        c::ConstTensorView::create(f.descriptions[2], weights.view()).value(),
        c::ConstTensorView::create(f.descriptions[3], ranges.view()).value()};
    auto invocation = prepared.create_invocation().value();
    auto done = invocation.submit(queue, inputs,
        std::array{c::TensorView::create(output_desc, output.view()).value()});
    ASSERT_TRUE(done);
    EXPECT_TRUE(done->wait().ok());
  }
#endif
}

TEST(CpuRowCacheIndexer, DisjointShortRangesSkipTheUnreferencedHistoryGap) {
#if !defined(HYLE_TEST_TORCH_CPU)
  GTEST_SKIP() << "optional ATen CPU carrier is unavailable";
#else
  Fixture f(8193, 2);
  f.ranges = {0, 1, 8192, 8193};
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const auto op = c::model_ops::row_cache_indexer_scores();
  const auto output_desc = c::infer_single_output(op, f.descriptions).value();
  auto prepared = c::prepare(context, op, f.descriptions,
      std::array{output_desc}, m::system_resource()).value();
  auto q = half_test::buffer<std::uint16_t>(f.query);
  auto bank = half_test::buffer<std::uint8_t>(f.bank);
  auto weights = half_test::buffer<std::uint16_t>(f.weights);
  auto ranges = half_test::buffer<std::int32_t>(f.ranges);
  auto output = half_test::buffer<float>(std::vector<float>(std::size_t(f.queries * f.rows)));
  const std::array inputs{
      c::ConstTensorView::create(f.descriptions[0], q.view()).value(),
      c::ConstTensorView::create(f.descriptions[1], bank.view()).value(),
      c::ConstTensorView::create(f.descriptions[2], weights.view()).value(),
      c::ConstTensorView::create(f.descriptions[3], ranges.view()).value()};
  auto invocation = prepared.create_invocation().value();
  auto done = invocation.submit(queue, inputs,
      std::array{c::TensorView::create(output_desc, output.view()).value()});
  ASSERT_TRUE(done);
  ASSERT_TRUE(done->wait().ok());
  const auto actual = half_test::read<float>(output), expected = f.expected();
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << i;
#endif
}
} // namespace
