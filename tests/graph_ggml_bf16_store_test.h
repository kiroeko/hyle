#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/graph/cpu.h"
#include "hyle/mem/buffer.h"
#include "hyle/simd/types.h"

namespace ggml_bf16_store_test {

inline void exercise() {
  namespace c = hyle::compute;
  namespace g = hyle::graph;
  using BF16 = hyle::simd::bfloat16_t;

  auto context_result = g::cpu::create_context({.native_threads = 4});
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(context_result).value();
  const auto description = c::TensorDesc::dense(
      c::DataType::kBFloat16, std::array<std::int64_t, 2>{1, 2}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto a = builder.input(description).value();
  const auto b = builder.input(description).value();
  const auto sum = builder.add(c::Binary{c::BinaryKind::kAdd}, a, b).value();
  const auto difference =
      builder.add(c::Binary{c::BinaryKind::kSubtract}, sum, a).value();
  // The intermediate is deliberately unpublished: its BF16 storage point
  // must survive grouping both arithmetic nodes into one native program.
  ASSERT_TRUE(builder.publish(difference).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = g::compile(graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1U);
  ASSERT_EQ(compiled->report.step_count(), 1U);
  auto step = compiled->report.step(0);
  ASSERT_TRUE(step) << step.error().message();
  EXPECT_EQ(step->provider.id, c::ggml_cpu::kProviderId);
  EXPECT_EQ(step->node_count, 2U);

  // All operands and both oracles are exact powers-of-two fractions. At the
  // first store, 1 + 1/256 rounds down to 1 and 129/128 + 1/256 rounds up to
  // 130/128 under ties-to-even. Omitting that store gives 1/256 in both lanes.
  const std::array<BF16, 2> a_values{
      BF16::from_bits(0x3f80), BF16::from_bits(0x3f81)};
  const std::array<BF16, 2> b_values{
      BF16::from_bits(0x3b80), BF16::from_bits(0x3b80)};
  constexpr std::array<std::uint16_t, 2> per_node_oracle{0x0000, 0x3c00};
  constexpr std::array<std::uint16_t, 2> omitted_store_oracle{0x3b80, 0x3b80};
  static_assert(per_node_oracle[0] != omitted_store_oracle[0] &&
                per_node_oracle[1] != omitted_store_oracle[1]);
  const auto resource = hyle::mem::system_resource();
  const auto bytes = description.required_bytes().value();
  auto a_buffer = hyle::mem::Buffer::create(resource, bytes, 64).value();
  auto b_buffer = hyle::mem::Buffer::create(resource, bytes, 64).value();
  auto output_buffer = hyle::mem::Buffer::create(resource, bytes, 64).value();
  ASSERT_TRUE(a_buffer.view().write(std::as_bytes(std::span(a_values))).ok());
  ASSERT_TRUE(b_buffer.view().write(std::as_bytes(std::span(b_values))).ok());
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(a, c::ConstTensorView::create(
      description, a_buffer.view()).value()).ok());
  ASSERT_TRUE(invocation.bind(b, c::ConstTensorView::create(
      description, b_buffer.view()).value()).ok());
  ASSERT_TRUE(invocation.bind(difference, c::TensorView::create(
      description, output_buffer.view()).value()).ok());
  auto queue = context.create_queue().value();
  const std::array queues{g::QueueBinding{
      .lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  auto done = invocation.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::array<std::uint16_t, 2> actual{};
  ASSERT_TRUE(output_buffer.view().read(
      std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, per_node_oracle);
  EXPECT_NE(actual, omitted_store_oracle);
}

} // namespace ggml_bf16_store_test
