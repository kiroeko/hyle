#pragma once

#include "gdn_snapshot_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/executable.h"

namespace gdn_region_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;

template <class Compile>
void exercise(c::Context &context, m::MemoryResourceRef memory,
              Compile compile) {
  constexpr std::uint32_t snapshots = 3;
  const gdn_snapshot_fixture::Fixture scalar;
  const gdn_snapshot_fixture::Fixture channel(7, 1, 5, 2, 6, 128, true);
  const gdn_snapshot_fixture::RecurrentFixture recurrent;
  for (int variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    const auto &fixture = variant == 2 ? recurrent.activated
                          : variant == 1 ? channel : scalar;
    const std::span<const c::TensorDesc> inputs = variant == 2
        ? std::span<const c::TensorDesc>(recurrent.inputs)
        : std::span<const c::TensorDesc>(fixture.inputs);
    const std::span<const std::vector<float>> data = variant == 2
        ? std::span<const std::vector<float>>(recurrent.data)
        : std::span<const std::vector<float>>(fixture.data);
    const auto operation = variant == 2
        ? c::model_ops::recurrent_gated_delta_net(snapshots)
        : c::Operation(c::GatedDeltaNet{.snapshot_count = snapshots});
    const auto outputs = fixture.outputs(snapshots);
    const auto expected = fixture.oracle(snapshots);
    auto builder = g::GraphBuilder::create().value();
    std::vector<g::ValueId> arguments;
    for (const auto &description : inputs)
      arguments.push_back(builder.input(description).value());
    const auto node = builder.add_node(operation, arguments).value();
    std::array<g::ValueId, 4> published;
    for (std::size_t port = 0; port < 2; ++port) {
      published[port] = builder.result(node, port).value();
      published[port + 2] = builder.add(
          c::Unary{c::UnaryKind::kNeg}, published[port]).value();
    }
    for (const auto id : published) ASSERT_TRUE(builder.publish(id).ok());
    const auto logical = std::move(builder).finish().value();
    auto compiled = compile(logical);
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.region_count(), 1U);
    ASSERT_EQ(compiled->report.step_count(), 1U);
    ASSERT_EQ(compiled->report.step(0)->node_count, 3U);
    auto invocation = compiled->executable.create_invocation().value();
    std::vector<m::Buffer> storage;
    storage.reserve(inputs.size() + published.size());
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      storage.push_back(m::Buffer::create(
          memory, inputs[port].required_bytes().value(), 256).value());
      ASSERT_TRUE(storage.back().view().write(
          std::as_bytes(std::span(data[port]))).ok());
      ASSERT_TRUE(invocation.bind(arguments[port], c::ConstTensorView::create(
          inputs[port], storage.back().view()).value()).ok());
    }
    for (std::size_t port = 0; port < published.size(); ++port) {
      const auto &description = outputs[port % 2];
      storage.push_back(m::Buffer::create(
          memory, description.required_bytes().value(), 256).value());
      ASSERT_TRUE(invocation.bind(published[port], c::TensorView::create(
          description, storage.back().view()).value()).ok());
    }
    auto queue = context.create_queue().value();
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto done = invocation.submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      for (std::size_t port = 0; port < published.size(); ++port) {
        std::vector<float> actual(expected[port % 2].size());
        ASSERT_TRUE(storage[inputs.size() + port].view().read(
            std::as_writable_bytes(std::span(actual))).ok());
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_NEAR(actual[i], (port < 2 ? 1.0 : -1.0) *
                                expected[port % 2][i], 3e-6) << port << ':' << i;
      }
    }
    if (variant == 1) {
      // Accept the middle retained full-matrix state (after token four),
      // then really compute token five from that device-resident subrange.
      // No payload readback or oracle value supplies the continued state.
      std::array<c::ConstTensorView, 6> feedback;
      for (std::size_t port = 0; port < 5; ++port) {
        const auto shape = fixture.inputs[port].dimensions();
        const std::array token_shape{shape[0], std::int64_t{1}, shape[2], shape[3]};
        const auto description = c::TensorDesc::dense(
            c::DataType::kFloat32, token_shape).value();
        const auto bytes = description.required_bytes().value();
        feedback[port] = c::ConstTensorView::create(description,
            storage[port].view().subview((fixture.tokens - 1) * bytes,
                                         bytes).value()).value();
      }
      const auto state_bytes = fixture.inputs[5].required_bytes().value();
      feedback[5] = c::ConstTensorView::create(fixture.inputs[5],
          storage[inputs.size() + 1].view().subview(state_bytes,
                                                   state_bytes).value()).value();
      const std::array next_descriptions{feedback[2].description(), fixture.inputs[5]};
      std::array<m::Buffer, 2> next_storage;
      std::array<c::TensorView, 2> next_outputs;
      for (std::size_t port = 0; port < 2; ++port) {
        next_storage[port] = m::Buffer::create(memory,
            next_descriptions[port].required_bytes().value(), 256).value();
        next_outputs[port] = c::TensorView::create(next_descriptions[port],
            next_storage[port].view()).value();
      }
      auto prepared = c::prepare(context, c::Operation(c::GatedDeltaNet{}),
                                  feedback, next_outputs);
      ASSERT_TRUE(prepared) << prepared.error().message();
      const auto requirement = prepared->workspace_requirement();
      auto workspace = requirement.size == 0
          ? m::Buffer{}
          : m::Buffer::create(memory, requirement.size, requirement.alignment)
                .value();
      auto continuation = prepared->create_invocation(workspace.view()).value();
      auto done = continuation.submit(queue, feedback, next_outputs);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto final = fixture.oracle(1);
      for (std::size_t port = 0; port < 2; ++port) {
        std::vector<float> actual(next_descriptions[port].element_count().value());
        ASSERT_TRUE(next_storage[port].view().read(
            std::as_writable_bytes(std::span(actual))).ok());
        const auto offset = port == 0 ? final[0].size() - actual.size() : 0;
        for (std::size_t i = 0; i < actual.size(); ++i)
          EXPECT_NEAR(actual[i], final[port][offset + i], 3e-6) << port << ':' << i;
      }
    }
  }
}
} // namespace gdn_region_test
