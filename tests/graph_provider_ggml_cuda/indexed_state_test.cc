#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "hyle/graph/cuda.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/slots.h"
#include "../compute_models/gdn_snapshot_fixture.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

constexpr std::size_t kBatches = 2, kSnapshots = 2;
constexpr std::size_t kElements = 2 * 16 * 16;
constexpr std::size_t kCapacity = 17, kStride = 4096;
constexpr std::size_t kAllocationAlignment = 256;
constexpr std::array<std::uint32_t, kBatches> kSources{13, 2};
// Batch-major, chronological last-K boundaries. Neither order is slot order.
constexpr std::array<std::uint32_t, kBatches * kSnapshots> kDestinations{7, 16, 4, 11};

void expect_bits(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
              std::bit_cast<std::uint32_t>(expected[i])) << "element " << i;
}

// One original public dense GDN, with no Gather/Store pattern to match. It
// keeps the original native arithmetic and public chronological snapshot order.
std::array<std::vector<float>, 2> dense_reference(c::Context &context,
    m::MemoryResourceRef resource, const gdn_snapshot_fixture::Fixture &fixture) {
  std::array<m::Buffer, 6> storage;
  std::array<c::ConstTensorView, 6> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    storage[port] = upload<float>(resource, fixture.data[port]);
    inputs[port] = c::ConstTensorView::create(fixture.inputs[port],
        storage[port].view()).value();
  }
  const auto descriptions = fixture.outputs(kSnapshots);
  std::array buffers{output_buffer(resource, descriptions[0]),
                    output_buffer(resource, descriptions[1])};
  const std::array outputs{
      c::TensorView::create(descriptions[0], buffers[0].view()).value(),
      c::TensorView::create(descriptions[1], buffers[1].view()).value()};
  auto prepared = c::prepare(context,
      c::Operation(c::GatedDeltaNet{.snapshot_count = kSnapshots}), inputs, outputs);
  if (!prepared) { ADD_FAILURE() << prepared.error().message(); return {}; }
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, resource);
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto queue = context.create_queue().value();
  auto done = invocation.submit(queue, inputs, outputs);
  if (!done) { ADD_FAILURE() << done.error().message(); return {}; }
  const auto waited = done->wait();
  if (!waited.ok()) { ADD_FAILURE() << waited.message(); return {}; }
  return {half_test::read<float>(buffers[0]), half_test::read<float>(buffers[1])};
}

void exercise(bool publish_activation, bool projected_fallback = false) {
  auto created = g::cuda::create_context();
  ASSERT_TRUE(created) << created.error().message();
  auto context = std::move(*created);
  auto memory_result = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice);
  ASSERT_TRUE(memory_result) << memory_result.error().message();
  auto memory = std::move(*memory_result);
  auto staging_result = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kPinnedHost);
  ASSERT_TRUE(staging_result) << staging_result.error().message();
  auto staging = std::move(*staging_result);
  // T>K is intentional: first-K stores would silently pass a T==K test.
  const gdn_snapshot_fixture::Fixture fixture(9, kBatches, 4, 2, 2, 16);
  const auto reference = dense_reference(context, memory.ref(), fixture);
  ASSERT_EQ(reference[0].size(), fixture.data[2].size());
  ASSERT_EQ(reference[1].size(), kBatches * kSnapshots * kElements);

  const auto item_result = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{2, 16, 16});
  ASSERT_TRUE(item_result) << item_result.error().message();
  const auto item = *item_result;
  const auto schema_result = s::Schema::create(item, kCapacity, kStride);
  ASSERT_TRUE(schema_result) << schema_result.error().message();
  const auto schema = *schema_result;
  std::vector<float> before(kCapacity * kStride / sizeof(float));
  for (std::size_t i = 0; i < before.size(); ++i)
    before[i] = 32.F + static_cast<float>(i % 127) / 128.F;
  for (std::size_t batch = 0; batch < kBatches; ++batch)
    std::copy_n(fixture.data[5].begin() + batch * kElements, kElements,
        before.begin() + kSources[batch] * kStride / sizeof(float));
  auto backing_result = m::Buffer::create(memory.ref(), before.size() * sizeof(float),
      kAllocationAlignment);
  ASSERT_TRUE(backing_result) << backing_result.error().message();
  auto backing = std::move(*backing_result);
  ASSERT_TRUE(backing.view().write(std::as_bytes(std::span(before))).ok());
  // Diagnostic read alias only, used after actual retirement. It is never
  // bound to Graph or represented as a fictional whole-pool ReadLease.
  const m::ConstBufferView backing_bytes = backing.view();
  auto pool_result = c::LeasePool::adopt(std::move(backing),
      {.max_blocks = 2 * kCapacity});
  ASSERT_TRUE(pool_result) << pool_result.error().message();
  auto pool = std::move(*pool_result);
  const auto layout_result = c::LeaseSlotLayout::create(pool, item, kCapacity, kStride);
  ASSERT_TRUE(layout_result) << layout_result.error().message();
  const auto layout = *layout_result;
  std::array<c::WriteLease, kCapacity> blocks;
  std::array<c::ReadLease, kCapacity> padding;
  for (std::size_t slot = 0; slot < kCapacity; ++slot) {
    auto block = pool.allocate(kElements * sizeof(float), kAllocationAlignment);
    ASSERT_TRUE(block) << "slot " << slot << ": " << block.error().message();
    blocks[slot] = std::move(*block);
    const auto address = layout.locate(blocks[slot]);
    ASSERT_TRUE(address) << address.error().message();
    ASSERT_EQ(address->index, slot);
    // CUDA's original allocation alignment is 256. Explicit ordinary gap
    // blocks preserve the larger real slot stride without claiming unsupported
    // allocation alignment or including padding in any Graph write lease.
    auto gap = pool.allocate(kStride - kElements * sizeof(float), kAllocationAlignment);
    ASSERT_TRUE(gap) << gap.error().message();
    auto ready_gap = gap->publish();
    ASSERT_TRUE(ready_gap) << ready_gap.error().message();
    padding[slot] = std::move(*ready_gap);
  }
  std::array<c::ReadLease, kBatches> sources;
  std::array<c::WriteLease, kBatches * kSnapshots> destinations;
  for (std::size_t batch = 0; batch < kBatches; ++batch) {
    auto source = blocks[kSources[batch]].publish();
    ASSERT_TRUE(source) << source.error().message();
    sources[batch] = std::move(*source);
  }
  for (std::size_t row = 0; row < destinations.size(); ++row)
    destinations[row] = std::move(blocks[kDestinations[row]]);
  std::array<c::ReadLease, kCapacity> unused;
  for (std::size_t slot = 0; slot < kCapacity; ++slot) {
    if (!blocks[slot].valid()) continue;
    auto ready = blocks[slot].publish();
    ASSERT_TRUE(ready) << ready.error().message();
    unused[slot] = std::move(*ready);
  }

  auto builder_result = g::GraphBuilder::create();
  ASSERT_TRUE(builder_result) << builder_result.error().message();
  auto builder = std::move(*builder_result);
  const auto capacity_description = schema.capacity_description();
  ASSERT_TRUE(capacity_description) << capacity_description.error().message();
  const auto pool_result_id = builder.input(*capacity_description);
  ASSERT_TRUE(pool_result_id) << pool_result_id.error().message();
  const auto pool_input = *pool_result_id;
  const auto source_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{kBatches});
  ASSERT_TRUE(source_description) << source_description.error().message();
  const auto source_result_id = builder.input(*source_description);
  ASSERT_TRUE(source_result_id) << source_result_id.error().message();
  const auto source_ids = *source_result_id;
  const auto destination_description = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{kBatches * kSnapshots});
  ASSERT_TRUE(destination_description) << destination_description.error().message();
  const auto destination_result_id = builder.input(*destination_description);
  ASSERT_TRUE(destination_result_id) << destination_result_id.error().message();
  const auto destination_ids = *destination_result_id;
  const auto gathered_result = builder.add(c::Gather{.axis = 0,
      .bounds = c::IndexBoundsPolicy::kChecked}, pool_input, source_ids);
  ASSERT_TRUE(gathered_result) << gathered_result.error().message();
  const auto gathered = *gathered_result;
  const auto initial_shape = c::TensorShape::create(fixture.inputs[5].dimensions());
  ASSERT_TRUE(initial_shape) << initial_shape.error().message();
  const auto initial_view = c::View::reshape(*initial_shape);
  ASSERT_TRUE(initial_view) << initial_view.error().message();
  const auto initial_result = builder.add(*initial_view, gathered);
  ASSERT_TRUE(initial_result) << initial_result.error().message();
  const auto initial = *initial_result;
  std::array<g::ValueId, 6> arguments;
  for (std::size_t port = 0; port < 5; ++port) {
    const auto input = builder.input(fixture.inputs[port]);
    ASSERT_TRUE(input) << input.error().message();
    arguments[port] = *input;
  }
  arguments[5] = initial;
  const auto gdn_result = builder.add_node(
      c::Operation(c::GatedDeltaNet{.snapshot_count = kSnapshots}), arguments);
  ASSERT_TRUE(gdn_result) << gdn_result.error().message();
  const auto gdn = *gdn_result;
  const auto activation_result = builder.result(gdn, 0);
  ASSERT_TRUE(activation_result) << activation_result.error().message();
  const auto activation = *activation_result;
  const auto state_result = builder.result(gdn, 1);
  ASSERT_TRUE(state_result) << state_result.error().message();
  const auto state = *state_result;
  const auto rows_shape = c::TensorShape::create(
      std::array<std::int64_t, 2>{kBatches * kSnapshots, kElements});
  ASSERT_TRUE(rows_shape) << rows_shape.error().message();
  const auto rows_view = c::View::reshape(*rows_shape);
  ASSERT_TRUE(rows_view) << rows_view.error().message();
  const auto rows_result = builder.add(*rows_view, state);
  ASSERT_TRUE(rows_result) << rows_result.error().message();
  const auto rows = *rows_result;
  const auto store_result = builder.add_node(c::Operation::custom(s::Store{schema}),
      std::array{rows, destination_ids});
  ASSERT_TRUE(store_result) << store_result.error().message();
  const auto store = *store_result;
  // Store has no mathematical result; its original external effect is the root.
  if (publish_activation) ASSERT_TRUE(builder.publish(activation).ok());
  // A legitimate escaping Gather result prevents the exact five-node indexed
  // fusion. It must remain usable through checked selected-row projection.
  if (projected_fallback) ASSERT_TRUE(builder.publish(gathered).ok());
  const auto graph_result = std::move(builder).finish();
  ASSERT_TRUE(graph_result) << graph_result.error().message();
  const auto &graph = *graph_result;
  auto compiled = g::cuda::compile(graph, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  if (projected_fallback) {
    ASSERT_GT(compiled->report.step_count(), 1u);
    for (std::size_t i = 0; i < compiled->report.step_count(); ++i)
      EXPECT_NE(compiled->report.step(i)->policy,
          (c::RoutePolicyId{0x48594c4547474355ULL, 6, 1}));
  } else {
    // Do not let a correct fallback Store/Gather masquerade as indexed fusion.
    ASSERT_EQ(compiled->report.step_count(), 1u);
    const auto step = compiled->report.step(0).value();
    EXPECT_EQ(step.kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(step.policy, (c::RoutePolicyId{0x48594c4547474355ULL, 6, 1}));
    EXPECT_EQ(step.implementation, c::ProviderImplementationId{7});
    EXPECT_EQ(step.dispatch_count, 1u);
    ASSERT_EQ(step.node_count, 5u);
    const std::array expected_nodes{graph.producer(gathered).value(),
        graph.producer(initial).value(), gdn, graph.producer(rows).value(), store};
    for (const auto expected : expected_nodes) {
      bool found = false;
      for (std::size_t i = 0; i < step.node_count; ++i)
        found |= compiled->report.step_node(0, i).value() == expected;
      EXPECT_TRUE(found);
    }
  }

  auto invocation = compiled->executable.create_invocation().value();
  auto queue = context.create_queue().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  std::array<m::Buffer, 5> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    inputs[port] = upload<float>(memory.ref(), fixture.data[port]);
    ASSERT_TRUE(invocation.bind(arguments[port], c::ConstTensorView::create(
        fixture.inputs[port], inputs[port].view()).value()).ok());
  }
  m::Buffer activation_buffer;
  if (publish_activation) {
    activation_buffer = output_buffer(memory.ref(), fixture.inputs[2]);
    ASSERT_TRUE(invocation.bind(activation, c::TensorView::create(fixture.inputs[2],
        activation_buffer.view()).value()).ok());
  }
  m::Buffer gathered_buffer;
  if (projected_fallback) {
    const auto gathered_description = graph.value_description(gathered).value();
    gathered_buffer = output_buffer(memory.ref(), gathered_description);
    ASSERT_TRUE(invocation.bind(gathered, c::TensorView::create(gathered_description,
        gathered_buffer.view()).value()).ok());
  }
  const std::array source_bindings{s::SourceBinding{store, graph.producer(gathered).value()}};
  const auto prepared = s::prepare(invocation, graph, queues, source_bindings,
      {.host_workspace = staging.ref()});
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  auto transaction = s::SlotBindings::create(layout, sources, destinations);
  ASSERT_TRUE(transaction) << transaction.error().message();
  for (std::size_t i = 0; i < kBatches; ++i)
    ASSERT_EQ(transaction->source_addresses()[i].index, kSources[i]);
  for (std::size_t i = 0; i < kBatches * kSnapshots; ++i)
    ASSERT_EQ(transaction->destination_addresses()[i].index, kDestinations[i]);
  g::CompileStepReport gather_step;
  bool found_gather_step = false;
  const auto original_gather = graph.producer(gathered).value();
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
    const auto candidate = compiled->report.step(i).value();
    for (std::size_t node = 0; node < candidate.node_count; ++node) {
      if (compiled->report.step_node(i, node).value() != original_gather) continue;
      gather_step = candidate;
      found_gather_step = true;
    }
  }
  ASSERT_TRUE(found_gather_step);
  const auto bound = s::bind(invocation, graph, store, std::move(*transaction));
  ASSERT_TRUE(bound.ok()) << bound.message() << "; Gather provider=" << gather_step.provider.name
      << ", step kind=" << static_cast<unsigned>(gather_step.kind);
  ASSERT_TRUE(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kCommands}).ok());
  auto done = invocation.submit(queues);
  ASSERT_TRUE(done) << done.error().message();
  const auto status = done->wait();
  ASSERT_TRUE(status.ok()) << status.message();
  *done = {};
  if (publish_activation) expect_bits(half_test::read<float>(activation_buffer), reference[0]);
  if (projected_fallback)
    expect_bits(half_test::read<float>(gathered_buffer), fixture.data[5]);
  auto returned = s::take(invocation, store);
  ASSERT_TRUE(returned) << returned.error().message();
  ASSERT_EQ(returned->source_count(), kBatches);
  ASSERT_EQ(returned->destination_count(), kBatches * kSnapshots);
  std::array<c::ReadLease, kBatches * kSnapshots> written;
  auto expected_backing = before;
  for (std::size_t row = 0; row < written.size(); ++row) {
    written[row] = returned->destinations()[row].publish().value();
    std::vector<float> actual(kElements);
    ASSERT_TRUE(written[row].view()->read(std::as_writable_bytes(std::span(actual))).ok());
    const auto expected = std::span(reference[1]).subspan(row * kElements, kElements);
    expect_bits(actual, expected);
    std::copy(expected.begin(), expected.end(), expected_backing.begin() +
        kDestinations[row] * kStride / sizeof(float));
  }
  // Includes every original source, unused slot, per-slot padding and trailing
  // padding, rather than only the selected destination ranges.
  std::vector<float> actual_backing(before.size());
  ASSERT_TRUE(backing_bytes.read(std::as_writable_bytes(std::span(actual_backing))).ok());
  expect_bits(actual_backing, expected_backing);
  for (std::size_t port = 0; port < inputs.size(); ++port)
    expect_bits(half_test::read<float>(inputs[port]), fixture.data[port]);

  if (!publish_activation) {
    // Bind a second real episode, then revoke its projected readers and
    // writers before submission. Reset must not trap their original owner or
    // let old native configuration revive the withdrawn permissions.
    std::array<c::WriteLease, kBatches * kSnapshots> next_writers;
    std::array<std::uint64_t, kBatches * kSnapshots> generations;
    for (std::size_t row = 0; row < next_writers.size(); ++row) {
      auto writer = written[row].try_write();
      ASSERT_TRUE(writer) << writer.error().message();
      next_writers[row] = std::move(*writer);
      generations[row] = next_writers[row].generation();
    }
    auto next = s::SlotBindings::create(layout, sources, next_writers);
    ASSERT_TRUE(next) << next.error().message();
    ASSERT_TRUE(s::bind(invocation, graph, store, std::move(*next)).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues,
        {.mode = g::ReplayMode::kCommands}).ok());
    ASSERT_TRUE(invocation.reset().ok());
    // Ordinary input bytes remain valid; they cannot re-admit reset's private
    // source projection or slot write episode.
    for (std::size_t port = 0; port < inputs.size(); ++port)
      ASSERT_TRUE(invocation.bind(arguments[port], c::ConstTensorView::create(
          fixture.inputs[port], inputs[port].view()).value()).ok());
    const auto stale_prepare = invocation.prepare_replay(queues,
        {.mode = g::ReplayMode::kCommands});
    ASSERT_FALSE(stale_prepare.ok());
    EXPECT_TRUE(stale_prepare.IsInvalidArgument() || stale_prepare.IsFailedPrecondition())
        << stale_prepare.message();
    auto stale_submit = invocation.submit(queues);
    ASSERT_FALSE(stale_submit);
    EXPECT_TRUE(stale_submit.error().IsInvalidArgument() ||
                stale_submit.error().IsFailedPrecondition()) << stale_submit.error().message();
    auto recovered = s::take(invocation, store);
    ASSERT_TRUE(recovered) << recovered.error().message();
    ASSERT_EQ(recovered->source_count(), kBatches);
    ASSERT_EQ(recovered->destination_count(), kBatches * kSnapshots);
    for (std::size_t row = 0; row < next_writers.size(); ++row) {
      EXPECT_EQ(recovered->destinations()[row].generation(), generations[row]);
      const auto address = layout.locate(recovered->destinations()[row]);
      ASSERT_TRUE(address) << address.error().message();
      EXPECT_EQ(address->index, kDestinations[row]);
    }
    ASSERT_TRUE(backing_bytes.read(std::as_writable_bytes(std::span(actual_backing))).ok());
    expect_bits(actual_backing, expected_backing);
  }
}

TEST(GgmlCudaIndexedStateGraph, PublishedActivationAndCandidateSlotsMatchDenseGdn) {
  exercise(true);
}
TEST(GgmlCudaIndexedStateGraph, EffectRootWritesCandidatesWithoutPublishedActivation) {
  exercise(false);
}
TEST(GgmlCudaIndexedStateGraph, EscapingGatherUsesCheckedProjectedFallback) {
  exercise(true, true);
}
} // namespace
