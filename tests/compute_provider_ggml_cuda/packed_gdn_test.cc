#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "../compute_models/gdn_snapshot_fixture.h"
#include "test_support.h"
#include "hyle/compute/sequence_ops.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
constexpr std::int64_t capacity = 128, key_heads = 2, value_heads = 4;

c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
void expect_bits(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected[i])) << i;
}

struct Environment final {
  // A declined GGML implementation cannot silently fall back to native CUDA.
  c::Context context = c::Context::create(c::cuda::device(0).value(), c::ggml_cuda::provider()).value();
  c::cuda::MemoryResource memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  c::Queue queue = context.create_queue().value();
};

struct Packed final {
  gdn_snapshot_fixture::Fixture fixture;
  std::array<c::TensorDesc, 7> descriptions;
  std::array<c::TensorDesc, 2> output_descriptions;
  // Original native invocation and its workspace die before borrowed buffers.
  std::array<m::Buffer, 7> storage;
  std::array<m::Buffer, 2> result_storage;
  std::array<c::ConstTensorView, 7> inputs;
  std::array<c::TensorView, 2> outputs;
  c::PreparedOperation prepared;
  m::Buffer workspace;
  c::OperationInvocation invocation;

  Packed(Environment &environment, std::int64_t depth,
         std::int64_t keys = key_heads, std::int64_t values = value_heads)
      : fixture(9, 1, capacity, keys, values, depth) {
    const auto f32 = c::DataType::kFloat32;
    descriptions = {dense(f32, {capacity, keys, depth}), dense(f32, {capacity, keys, depth}),
        dense(f32, {capacity, values, depth}), dense(f32, {capacity, values, 1}),
        dense(f32, {capacity, values, 1}), fixture.inputs[5], dense(c::DataType::kInt32, {2})};
    const auto operation = c::sequence_ops::packed_gated_delta_net();
    const auto inferred = c::infer_output_descriptions(operation, descriptions, output_descriptions);
    EXPECT_TRUE(inferred.ok()) << inferred.message();
    for (std::size_t i = 0; i < storage.size(); ++i) {
      storage[i] = m::Buffer::create(environment.memory.ref(), descriptions[i].required_bytes().value(), 256).value();
      inputs[i] = c::ConstTensorView::create(descriptions[i], storage[i].view()).value();
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto size = output_descriptions[i].required_bytes().value();
      result_storage[i] = m::Buffer::create(environment.memory.ref(), size + 32, 256).value();
      EXPECT_TRUE(result_storage[i].view().fill(std::byte{0xa5}).ok());
      outputs[i] = c::TensorView::create(output_descriptions[i], result_storage[i].view().subview(0, size).value()).value();
    }
    prepared = c::prepare(environment.context, operation, descriptions, output_descriptions, environment.memory.ref()).value();
    EXPECT_EQ(prepared.provider().id, c::ggml_cuda::kProviderId);
    workspace = ggml_cuda_test::workspace_buffer(prepared, environment.memory.ref());
    fresh(); upload(0);
  }

  void fresh() { invocation = prepared.create_invocation(workspace.view()).value(); }
  void offsets(std::array<std::int32_t, 2> values) {
    EXPECT_TRUE(storage[6].view().write(std::as_bytes(std::span(values))).ok());
  }
  void upload(std::size_t active) {
    for (std::size_t port = 0; port < 6; ++port) {
      auto data = fixture.data[port];
      if (port != 5) {
        const auto row_elements = data.size() / std::size_t(capacity);
        std::fill(data.begin() + active * row_elements, data.end(), std::bit_cast<float>(std::uint32_t{0x7fc12345}));
      }
      EXPECT_TRUE(storage[port].view().write(std::as_bytes(std::span(data))).ok());
    }
    offsets({0, static_cast<std::int32_t>(active)});
  }
  hyle::Status submit(c::Queue &queue) {
    auto done = invocation.submit(queue, inputs, outputs);
    if (!done) return done.error();
    return done->wait();
  }
  std::vector<float> read(std::size_t port) const {
    std::vector<float> values(output_descriptions[port].element_count().value());
    EXPECT_TRUE(outputs[port].storage().read(std::as_writable_bytes(std::span(values))).ok()); return values;
  }
  void poison() {
    for (auto &output : outputs) EXPECT_TRUE(output.storage().fill(std::byte{0x5a}).ok());
  }
  void expect_poison() const {
    for (const auto &output : outputs) {
      std::vector<std::byte> actual(output.storage().size());
      ASSERT_TRUE(output.storage().read(actual).ok());
      EXPECT_EQ(actual, std::vector<std::byte>(actual.size(), std::byte{0x5a}));
    }
    expect_guards();
  }
  void expect_guards() const {
    for (std::size_t port = 0; port < outputs.size(); ++port) {
      std::array<std::byte, 32> actual;
      std::array<std::byte, 32> expected;
      expected.fill(std::byte{0xa5});
      ASSERT_TRUE(result_storage[port].view().subview(output_descriptions[port].required_bytes().value(), 32).value().read(actual).ok());
      EXPECT_EQ(actual, expected);
    }
  }
};

std::array<std::vector<float>, 2> dense_reference(Environment &environment, const Packed &packed, std::size_t active) {
  const auto &fixture = packed.fixture;
  std::array<c::ConstTensorView, 6> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    auto shape = std::vector<std::int64_t>(fixture.inputs[port].dimensions().begin(), fixture.inputs[port].dimensions().end());
    if (port != 5) shape[1] = static_cast<std::int64_t>(active);
    const auto description = c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
    inputs[port] = c::ConstTensorView::create(description,
        packed.storage[port].view().subview(0, description.required_bytes().value()).value()).value();
  }
  const std::array descriptions{inputs[2].description(), inputs[5].description()};
  std::array<m::Buffer, 2> storage;
  std::array<c::TensorView, 2> outputs;
  for (std::size_t port = 0; port < outputs.size(); ++port) {
    storage[port] = m::Buffer::create(environment.memory.ref(), descriptions[port].required_bytes().value(), 256).value();
    outputs[port] = c::TensorView::create(descriptions[port], storage[port].view()).value();
  }
  auto prepared = c::prepare(environment.context, c::Operation(c::GatedDeltaNet{}), inputs, outputs).value();
  EXPECT_EQ(prepared.provider().id, c::ggml_cuda::kProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(prepared, environment.memory.ref());
  auto invocation = prepared.create_invocation(workspace.view()).value();
  auto done = invocation.submit(environment.queue, inputs, outputs).value();
  const auto status = done.wait(); EXPECT_TRUE(status.ok()) << status.message();
  std::array<std::vector<float>, 2> result;
  for (std::size_t port = 0; port < result.size(); ++port) {
    result[port].resize(descriptions[port].element_count().value());
    EXPECT_TRUE(storage[port].view().read(std::as_writable_bytes(std::span(result[port]))).ok());
  }
  return result;
}

void expect_result(Environment &environment, const Packed &packed, std::size_t active) {
  const auto actual_y = packed.read(0), state = packed.read(1);
  const auto active_elements = active * std::size_t(packed.fixture.value_heads * packed.fixture.depth);
  if (active) {
    const auto reference = dense_reference(environment, packed, active);
    expect_bits(std::span(actual_y).first(active_elements), reference[0]);
    expect_bits(state, reference[1]);
  } else expect_bits(state, packed.fixture.data[5]);
  for (std::size_t i = active_elements; i < actual_y.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_y[i]), 0U) << "inactive output " << i;
  std::vector<float> original(packed.fixture.data[5].size());
  ASSERT_TRUE(packed.storage[5].view().read(std::as_writable_bytes(std::span(original))).ok());
  expect_bits(original, packed.fixture.data[5]);
  packed.expect_guards();
}

TEST(GgmlCudaPackedGdn, ScalarGateDepthsAndActualLengthsMatchOrdinaryBits) {
  Environment environment;
  for (const auto depth : {16, 32, 64, 128}) {
    SCOPED_TRACE(depth);
    Packed packed(environment, depth);
    for (const auto active : {0u, 1u, 105u, 128u}) {
      SCOPED_TRACE(active);
      packed.upload(active); packed.poison();
      const auto status = packed.submit(environment.queue); ASSERT_TRUE(status.ok()) << status.message();
      expect_result(environment, packed, active);
    }
  }
}

TEST(GgmlCudaPackedGdn, EmptyPreservesInitialNaNPayloadAndNegativeZeroBits) {
  Environment environment;
  constexpr std::array<std::uint32_t, 5> special{0x80000000U, 0x7fc01234U, 0x7f800000U, 0xff800000U, 0x3f800000U};
  for (const auto depth : {16, 32, 64, 128}) {
    SCOPED_TRACE(depth);
    Packed packed(environment, depth);
    for (std::size_t i = 0; i < packed.fixture.data[5].size(); ++i)
      packed.fixture.data[5][i] = std::bit_cast<float>(special[i % special.size()]);
    packed.upload(0); packed.poison();
    const auto status = packed.submit(environment.queue); ASSERT_TRUE(status.ok()) << status.message();
    expect_result(environment, packed, 0);
  }
}

TEST(GgmlCudaPackedGdn, MmaDomainMatchesIndependentRecurrenceWithPoisonedPadding) {
  Environment environment;
  for (const auto heads : {16, 32}) {
    SCOPED_TRACE(heads);
    Packed packed(environment, 128, 16, heads);
    for (const auto active : {1u, 63u, 65u, 105u, 128u}) {
      SCOPED_TRACE(active);
      packed.upload(active); packed.poison();
      const auto status = packed.submit(environment.queue);
      ASSERT_TRUE(status.ok()) << status.message();
      gdn_snapshot_fixture::Fixture reference(9, 1, active, 16, heads, 128);
      for (std::size_t port = 0; port < reference.data.size(); ++port)
        std::copy_n(packed.fixture.data[port].begin(), reference.data[port].size(), reference.data[port].begin());
      const auto expected = reference.oracle(1);
      for (std::size_t port = 0; port < expected.size(); ++port) {
        const auto actual = packed.read(port);
        for (std::size_t i = 0; i < expected[port].size(); ++i)
          EXPECT_NEAR(actual[i], expected[port][i], 1e-5) << "port " << port << " element " << i;
        if (port == 0) {
          for (std::size_t i = expected[port].size(); i < actual.size(); ++i)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), 0U) << "inactive output " << i;
        }
      }
      // Both carriers use the same MMA schedule for these actual lengths.
      if (active >= 64) expect_result(environment, packed, active);
      packed.expect_guards();
    }
    // No recurrence may run for L=0, even though the physical shape selects MMA.
    constexpr std::array<std::uint32_t, 3> special{0x80000000U, 0x7fc01234U, 0x3f800000U};
    for (std::size_t i = 0; i < packed.fixture.data[5].size(); ++i)
      packed.fixture.data[5][i] = std::bit_cast<float>(special[i % special.size()]);
    packed.upload(0); packed.poison();
    ASSERT_TRUE(packed.submit(environment.queue).ok());
    expect_result(environment, packed, 0);
    packed.offsets({0, 129}); packed.poison();
    EXPECT_TRUE(packed.submit(environment.queue).IsInvalidArgument());
    packed.expect_poison();
  }
}

TEST(GgmlCudaPackedGdn, InvalidOffsetsPreserveBothOutputsAndRetiredPlanCanRetry) {
  Environment environment;
  Packed packed(environment, 32);
  for (const auto offsets : {std::array<std::int32_t, 2>{1, 105}, std::array<std::int32_t, 2>{-1, 105},
                             std::array<std::int32_t, 2>{0, -1}, std::array<std::int32_t, 2>{0, 129}}) {
    SCOPED_TRACE(offsets[0]);
    SCOPED_TRACE(offsets[1]);
    packed.upload(105); packed.offsets(offsets); packed.poison();
    const auto status = packed.submit(environment.queue);
    EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
    packed.expect_poison();
    EXPECT_FALSE(packed.invocation.in_flight());
    auto denied = packed.invocation.submit(environment.queue, packed.inputs, packed.outputs);
    EXPECT_FALSE(denied);
    if (!denied) EXPECT_TRUE(denied.error().IsFailedPrecondition()) << denied.error().message();
    // Failure is terminal for the old slot. Recreate only after real wait.
    packed.fresh(); packed.upload(105);
    const auto recovered = packed.submit(environment.queue); ASSERT_TRUE(recovered.ok()) << recovered.message();
    expect_result(environment, packed, 105);
  }
}

void captured_lengths(Environment &environment, Packed &packed, std::span<const unsigned> lengths) {
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 7> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port) inputs[port] = builder.input(packed.descriptions[port]).value();
  const auto node = builder.add_node(c::sequence_ops::packed_gated_delta_net(), inputs).value();
  const std::array outputs{builder.result(node, 0).value(), builder.result(node, 1).value()};
  for (const auto output : outputs) ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, environment.context, environment.memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto run = compiled->executable.create_invocation().value();
  for (std::size_t port = 0; port < inputs.size(); ++port) ASSERT_TRUE(run.bind(inputs[port], packed.inputs[port]).ok());
  for (std::size_t port = 0; port < outputs.size(); ++port) ASSERT_TRUE(run.bind(outputs[port], packed.outputs[port]).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &environment.queue}};
  std::size_t round = 0;
  for (const auto active : lengths) {
    SCOPED_TRACE(active);
    packed.upload(active); packed.poison();
    const auto prepared = run.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto replay = run.replay_report().value();
    EXPECT_EQ(replay.route, g::ReplayRoute::kCaptured);
    if (round++) {
      EXPECT_EQ(replay.recorded_capture_range_count, 0U);
      EXPECT_GE(replay.reused_capture_range_count, 1U);
    }
    auto done = run.submit(queues); ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait(); *done = {};
    ASSERT_TRUE(status.ok()) << status.message();
    expect_result(environment, packed, active);
  }
}
TEST(GgmlCudaPackedGdn, CapturedFixedCapacityReusesItsTupleWhenActualLengthChanges) {
  Environment environment;
  Packed packed(environment, 128);
  const std::array lengths{128u, 105u, 1u, 0u, 105u, 128u};
  captured_lengths(environment, packed, lengths);
}
TEST(GgmlCudaPackedGdn, CapturedMmaPreservesLengthChangesAndEmptyStateBits) {
  Environment environment;
  Packed packed(environment, 128, 16, 16);
  const std::array lengths{128u, 105u, 65u, 0u, 105u, 128u};
  captured_lengths(environment, packed, lengths);
}
} // namespace
