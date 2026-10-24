#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "../compute_models/grouped_shared_activation_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#endif

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using Fixture = grouped_shared_test::Fixture;
enum class Mode { kPerSlot, kShared, kBroadcast };

void exercise(std::int64_t tokens, Mode mode, bool bfloat = false,
              std::int64_t slots = 2) {
  const Fixture data(tokens, slots);
  const bool shared = mode != Mode::kPerSlot;
  const auto wd = c::TensorDesc::dense(bfloat ? c::DataType::kBFloat16 : c::DataType::kFloat32,
      std::array{data.experts, data.rows, data.columns}).value();
  const auto compact = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array{tokens, shared ? std::int64_t{1} : data.slots, data.columns}).value();
  [[maybe_unused]] const auto ad = mode == Mode::kBroadcast ? c::TensorDesc::strided_bytes(c::DataType::kFloat32,
      std::array{tokens, data.slots, data.columns},
      std::array{data.columns * 4, std::int64_t{0}, std::int64_t{4}}).value() : compact;
  const auto id = c::TensorDesc::dense(c::DataType::kInt32, std::array{tokens, data.slots}).value();
  const auto yd = c::TensorDesc::dense(c::DataType::kFloat32, std::array{tokens, data.slots, data.rows}).value();
  const auto allocate = [&](const c::TensorDesc &description) {
    return m::Buffer::create(m::system_resource(), description.required_bytes().value(), 64).value();
  };
  auto weights = allocate(wd);
  if (bfloat) {
    std::vector<hyle::simd::bfloat16_t> values;
    for (const auto value : data.weights) values.push_back(half_test::encode<hyle::simd::bfloat16_t>(value));
    ASSERT_TRUE(weights.view().write(std::as_bytes(std::span(values))).ok());
  } else ASSERT_TRUE(weights.view().write(std::as_bytes(std::span(data.weights))).ok());
  const auto wv = c::ConstTensorView::create(wd, weights.view()).value();
  std::vector<m::Buffer> retained;
  std::vector<std::vector<float>> old_values;
  std::vector<m::Buffer> old_outputs;
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  namespace g = hyle::graph;
  auto context = g::cpu::create_context().value();
  auto builder = g::GraphBuilder::create().value();
  const auto w = (bfloat ? builder.constant(wd) : builder.input(wd)).value();
  const auto input = builder.input(compact).value(), indices = builder.input(id).value();
  auto activation = input;
  if (mode == Mode::kBroadcast) {
    const std::array shape{g::Dimension::fixed(tokens), g::Dimension::fixed(data.slots), g::Dimension::fixed(data.columns)};
    activation = builder.result(builder.add_node(g::ViewSpec::broadcast_to(shape).value(), input).value()).value();
  }
  const auto y = builder.result(builder.add_node(c::model_ops::grouped_matmul(),
      std::array{w, activation, indices}).value()).value();
  ASSERT_TRUE(builder.publish(y).ok());
  const std::array constants{g::ConstantBinding{w, wv}};
  auto compiled = g::compile(std::move(builder).finish().value(), context,
      {.constants = bfloat ? std::span<const g::ConstantBinding>(constants) : std::span<const g::ConstantBinding>{}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(compiled->report.step_count() - 1)->provider.id, c::ggml_cpu::kProviderId);
  if (bfloat) {
    EXPECT_EQ(compiled->report.packed_constant_count(), 1U);
    EXPECT_EQ(compiled->report.packed_constant_bytes(), data.weights.size() * sizeof(float));
  }
  auto queue = context.create_queue().value();
  std::array calls{compiled->executable.create_invocation().value(), compiled->executable.create_invocation().value()};
  const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
#else
  auto context = c::cpu::create_context().value();
  const std::array descriptions{wd, ad, id};
  const std::array outputs{yd};
  auto prepared = c::prepare(context, c::model_ops::grouped_matmul(), descriptions, outputs, m::system_resource());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cpu::kProviderId);
  auto queue = context.create_queue().value();
  std::array calls{prepared->create_invocation().value(), prepared->create_invocation().value()};
#endif
  for (int round = 0; round < 2; ++round)
    for (std::size_t lane = 0; lane < calls.size(); ++lane) {
      SCOPED_TRACE(::testing::Message() << tokens << '/' << int(mode) << '/' << bfloat << '/' << round << '/' << lane);
      auto values = data.activations(shared, round);
      if (bfloat) {
        auto first = std::ranges::find_if(values, [](float value) { return value != 0; });
        ASSERT_NE(first, values.end());
        *first = std::copysign(1.00390625F, *first);
      }
      auto ids = data.ids(lane != 0, round);
      const auto expected = data.expected(shared, values, ids);
      if (bfloat) {
        auto narrowed = values;
        for (auto &value : narrowed) value = static_cast<float>(half_test::decode(half_test::encode<hyle::simd::bfloat16_t>(value)));
        EXPECT_NE(data.expected(shared, narrowed, ids), expected);
      }
      auto x = allocate(compact), selected = allocate(id), result = allocate(yd);
      ASSERT_TRUE(x.view().write(std::as_bytes(std::span(values))).ok());
      ASSERT_TRUE(selected.view().write(std::as_bytes(std::span(ids))).ok());
      auto destination = c::TensorView::create(yd, result.view()).value();
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
      const auto bind = [&]() {
        if (!bfloat) EXPECT_TRUE(calls[lane].bind(w, wv).ok());
        EXPECT_TRUE(calls[lane].bind(input, c::ConstTensorView::create(compact, x.view()).value()).ok());
        EXPECT_TRUE(calls[lane].bind(indices, c::ConstTensorView::create(id, selected.view()).value()).ok());
        EXPECT_TRUE(calls[lane].bind(y, destination).ok());
        EXPECT_TRUE(calls[lane].prepare_replay(queues).ok());
      };
      bind();
      const auto run = [&]() -> hyle::Status {
        auto work = calls[lane].submit(queue); if (!work) return work.error(); return work->wait();
      };
#else
      const std::array inputs{wv, c::ConstTensorView::create(ad, x.view()).value(),
          c::ConstTensorView::create(id, selected.view()).value()};
      const auto run = [&]() -> hyle::Status {
        auto work = calls[lane].submit(queue, inputs, std::span(&destination, 1));
        if (!work) return work.error(); return work->wait();
      };
#endif
      ASSERT_TRUE(run().ok()); EXPECT_EQ(half_test::read<float>(result), expected);
      for (const auto bad : {std::int32_t{-1}, static_cast<std::int32_t>(data.experts)}) {
        auto invalid = ids; invalid.back() = bad;
        ASSERT_TRUE(selected.view().write(std::as_bytes(std::span(invalid))).ok());
        EXPECT_TRUE(run().IsOutOfRange());
        EXPECT_EQ(half_test::read<float>(result), expected);
        EXPECT_EQ(half_test::read<std::int32_t>(selected), invalid);
        ASSERT_TRUE(selected.view().write(std::as_bytes(std::span(ids))).ok());
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
        calls[lane] = compiled->executable.create_invocation().value(); bind();
#else
        calls[lane] = prepared->create_invocation().value();
#endif
        ASSERT_TRUE(run().ok()); EXPECT_EQ(half_test::read<float>(result), expected);
      }
      old_values.push_back(expected); old_outputs.push_back(std::move(result));
      retained.push_back(std::move(x)); retained.push_back(std::move(selected));
      for (std::size_t i = 0; i < old_outputs.size(); ++i)
        EXPECT_EQ(half_test::read<float>(old_outputs[i]), old_values[i]);
    }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#define GROUPED_CPU_SUITE CpuGroupedSharedGraph
#else
#define GROUPED_CPU_SUITE CpuGroupedSharedDirect
#endif
TEST(GROUPED_CPU_SUITE, DenseSharedAndPerSlotInputsPreserveEveryId) {
  for (const auto tokens : {1, 9})
    for (const auto mode : {Mode::kPerSlot, Mode::kShared}) exercise(tokens, mode);
}
TEST(GROUPED_CPU_SUITE, BroadcastViewKeepsCompactBackingAndCheckedRecovery) {
  for (const auto tokens : {1, 9}) exercise(tokens, Mode::kBroadcast);
  exercise(1, Mode::kBroadcast, false, 1);
}
TEST(GROUPED_CPU_SUITE, BfloatWeightsRetainNonBfloatActivationAndColdConstants) {
  exercise(1, Mode::kPerSlot, true);
  exercise(9, Mode::kBroadcast, true);
}
#undef GROUPED_CPU_SUITE
} // namespace
