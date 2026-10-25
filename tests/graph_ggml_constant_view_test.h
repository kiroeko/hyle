#pragma once

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"
#include "hyle/mem/buffer.h"

namespace ggml_constant_view_test {
// A late independent constant transpose must not divide two native math nodes.
// Published, controlled and dynamic aliases retain the original Graph route.
inline void exercise(hyle::compute::Context &context,
                     hyle::mem::MemoryResourceRef resource) {
  namespace c = hyle::compute;
  namespace g = hyle::graph;
  namespace m = hyle::mem;
  const auto a_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{1, 4}).value();
  const auto w_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{3, 4}).value();
  const auto y_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{1, 3}).value();
  const auto bits = [](int value) {
    // Every fixture integer/result is exactly representable in BF16.
    return static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(static_cast<float>(value)) >> 16);
  };
  const std::array<int, 12> weight{1, 2, 3, 4, 0, -1, 1, 2, -2, 3, -4, 1};
  std::array<std::uint16_t, 12> weight_bits{};
  for (std::size_t i = 0; i < weight.size(); ++i) weight_bits[i] = bits(weight[i]);
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode); // 0 bridge, 1 published, 2 control, 3 dynamic.
    const auto buffer = [&](const c::TensorDesc &desc) {
      return m::Buffer::create(resource, desc.required_bytes().value(), 256).value();
    };
    auto a = buffer(a_desc), w = buffer(w_desc);
    std::array y{buffer(y_desc), buffer(y_desc)};
    ASSERT_TRUE(w.view().write(std::as_bytes(std::span(weight_bits))).ok());
    c::ReadLease ready;
    if (mode != 3) ready = c::ReadLease::adopt(std::move(w)).value();
    const auto original = ready ? ready.tensor(w_desc).value()
        : c::ConstTensorView::create(w_desc, w.view()).value();
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(a_desc).value();
    const auto weights = (mode == 3 ? builder.input(w_desc)
                                    : builder.constant(w_desc)).value();
    const auto first = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
        std::array{input}).value();
    constexpr std::array<std::int32_t, 2> axes{1, 0};
    const auto view = builder.add_node(c::Operation(c::View::permute(axes).value()),
        std::array{weights}).value();
    if (mode == 2) ASSERT_TRUE(builder.depends_on(view, first).ok());
    const auto viewed = builder.result(view).value();
    const auto last = builder.add_node(c::Operation(c::MatMul{}),
        std::array{builder.result(first).value(), viewed}).value();
    const auto output = builder.result(last).value();
    ASSERT_TRUE(builder.publish(output).ok());
    if (mode == 1) ASSERT_TRUE(builder.publish(viewed).ok());
    const auto graph = std::move(builder).finish().value();
    const std::array constants{g::ConstantBinding{weights, original,
        g::ConstantPolicy::kBorrow, {}, ready}};
    const std::array storage{g::StorageDomain{.resource = resource}};
    const std::array targets{g::ExecutionTarget{
        .source = g::ExecutionSource::acquire(context).value(),
        .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
    auto compiled = g::compile(graph,
        {.storage_domains = storage, .targets = targets, .lanes = lanes},
        {.constants = mode == 3 ? std::span<const g::ConstantBinding>{}
                               : std::span<const g::ConstantBinding>(constants)});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.packed_constant_count(), 0U);
    EXPECT_EQ(compiled->report.packed_constant_bytes(), 0U);
    EXPECT_EQ(compiled->report.region_count(), mode == 0 ? 1U : 0U);
    EXPECT_EQ(compiled->report.step_count(), mode == 0 ? 2U : 3U);
    std::size_t metadata_views = 0;
    for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
      const auto step = compiled->report.step(i).value();
      if (step.kind == g::CompileStepKind::kView) ++metadata_views;
      if (step.kind != g::CompileStepKind::kProviderRegion) continue;
      ASSERT_EQ(step.node_count, 2U);
      EXPECT_EQ(compiled->report.step_node(i, 0).value(), first);
      EXPECT_EQ(compiled->report.step_node(i, 1).value(), last);
    }
    EXPECT_EQ(metadata_views, 1U);
    std::array invocations{compiled->executable.create_invocation().value(),
                           compiled->executable.create_invocation().value()};
    auto queue = context.create_queue().value();
    std::array<std::uint16_t, 3> previous{};
    for (std::size_t phase = 0; phase < invocations.size(); ++phase) {
      std::array<std::uint16_t, 4> input_bits{};
      std::array<int, 4> source{};
      for (std::size_t k = 0; k < source.size(); ++k) {
        source[k] = (k % 2 == 0 ? 1 : -1) * static_cast<int>(k + 1) +
                    static_cast<int>(phase);
        input_bits[k] = bits(source[k]);
      }
      ASSERT_TRUE(a.view().write(std::as_bytes(std::span(input_bits))).ok());
      auto &run = invocations[phase];
      ASSERT_TRUE(run.bind(input, c::ConstTensorView::create(a_desc, a.view()).value()).ok());
      if (mode == 3) ASSERT_TRUE(run.bind(weights, original).ok());
      ASSERT_TRUE(run.bind(output, c::TensorView::create(y_desc, y[phase].view()).value()).ok());
      const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
      ASSERT_TRUE(run.prepare_replay(queues).ok());
      auto done = run.submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      std::array<std::uint16_t, 3> actual{};
      ASSERT_TRUE(y[phase].view().read(std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t n = 0; n < actual.size(); ++n) {
        int expected = 0;
        for (std::size_t k = 0; k < source.size(); ++k)
          expected -= source[k] * weight[n * source.size() + k];
        EXPECT_EQ(actual[n], bits(expected));
      }
      if (phase == 0) previous = actual;
      else {
        std::array<std::uint16_t, 3> old{};
        ASSERT_TRUE(y[0].view().read(std::as_writable_bytes(std::span(old))).ok());
        EXPECT_EQ(old, previous);
      }
      std::array<std::uint16_t, 12> unchanged{};
      ASSERT_TRUE(original.storage().read(std::as_writable_bytes(std::span(unchanged))).ok());
      EXPECT_EQ(unchanged, weight_bits);
      if (mode == 1) {
        const auto published = run.result(viewed).value().tensor().value();
        EXPECT_EQ(published.description(), graph.value_description(viewed).value());
        EXPECT_EQ(published.storage().resource(), original.storage().resource());
        EXPECT_EQ(published.storage().native_handle(), original.storage().native_handle());
        EXPECT_EQ(published.storage().offset(), original.storage().offset());
      }
    }
  }
}
} // namespace ggml_constant_view_test
