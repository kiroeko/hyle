#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../engine/inference/graph_bindings.h"
#include "../../engine/inference/submission_views.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace i = hyle::inference;
namespace m = hyle::mem;
namespace s = g::slots;

void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}

struct TwoDomains final {
  c::Context context;
  c::Queue queue;
  std::array<c::StorageOwner, 2> owners;
  std::array<c::LeasePool, 2> inputs, outputs;
  std::array<c::ReadLease, 2> input_ready;
  c::TensorDesc description;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  std::array<g::ValueId, 2> input_ids, output_ids;
  std::array<m::ConstBufferView, 2> output_storage;
  const std::array<std::array<float, 4>, 2> expected{{{1.F, -2.F, 3.F, 4.F},
      {11.F, 12.F, -13.F, 14.F}}};

  explicit TwoDomains(bool node_defaults = false) {
    context = take(g::cuda::create_context());
    queue = take(context.create_queue());
    check(queue.reserve_completions(4));
    description = take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 1>{4}));
    std::array<g::NodeId, 2> copies;
    auto builder = take(g::GraphBuilder::create());
    for (std::size_t domain = 0; domain != owners.size(); ++domain) {
      auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
      owners[domain] = take(c::StorageOwner::adopt(context, std::move(memory)));
      inputs[domain] = take(c::LeasePool::create(owners[domain].ref(), 256, {.max_blocks = 1}));
      outputs[domain] = take(c::LeasePool::create(owners[domain].ref(), 256, {.max_blocks = 1}));
      auto writer = take(inputs[domain].allocate(sizeof(expected[domain]), 256));
      check(take(writer.view()).write(std::as_bytes(std::span(expected[domain]))));
      input_ready[domain] = take(writer.publish());
      input_ids[domain] = take(builder.input(description));
      copies[domain] = take(builder.add_node(c::Operation(c::Copy{}), std::array{input_ids[domain]}));
      output_ids[domain] = take(builder.result(copies[domain]));
      check(builder.publish(output_ids[domain]));
    }
    graph = take(std::move(builder).finish());
    const std::array domains{
        g::StorageDomain{.resource = owners[0].ref(), .owner = owners[0]},
        g::StorageDomain{.resource = owners[1].ref(), .owner = owners[1]}};
    const std::array targets{g::ExecutionTarget{
        .source = take(g::ExecutionSource::acquire(context)), .default_storage = g::StorageDomainId{0}}};
    const std::array placements{
        g::ValuePlacement{output_ids[0], g::StorageDomainId{0}, g::PlacementStrength::kRequired},
        g::ValuePlacement{output_ids[1], g::StorageDomainId{1}, g::PlacementStrength::kRequired}};
    const std::array contracts{
        g::ExternalResourceContract{input_ids[0], g::StorageDomainId{0}},
        g::ExternalResourceContract{input_ids[1], g::StorageDomainId{1}},
        g::ExternalResourceContract{output_ids[0], g::StorageDomainId{0}},
        g::ExternalResourceContract{output_ids[1], g::StorageDomainId{1}}};
    if (node_defaults) {
      i::GraphBindings bindings(domains);
      for (std::size_t domain = 0; domain != owners.size(); ++domain)
        check(bindings.node(copies[domain], {g::ExecutionTargetId{0},
            g::StorageDomainId{static_cast<std::uint32_t>(domain)}}));
      for (const auto &contract : contracts) check(bindings.external(contract.value, contract.storage));
      compiled = take(g::cuda::compile(graph, {.storage_domains = domains, .targets = targets}, bindings.options()));
    } else compiled = take(g::cuda::compile(graph, {.storage_domains = domains, .targets = targets},
        {.value_placements = placements, .external_resources = contracts}));
    run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    for (std::size_t domain = 0; domain != owners.size(); ++domain) {
      check(run.bind(input_ids[domain], description, input_ready[domain]));
      auto writer = take(outputs[domain].allocate(sizeof(expected[domain]), 256));
      output_storage[domain] = take(writer.tensor(description)).as_const().storage();
      check(run.bind(output_ids[domain], description, std::move(writer)));
    }
    check(run.prepare_replay(queues(),
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject}));
  }
  ~TwoDomains() { context.drain().IgnoreError(); }

  std::array<g::QueueBinding, 1> queues() { return {{{g::ExecutionLaneId{0}, &queue}}}; }
  void prepare(i::SubmissionViews &views) {
    check(views.prepare(run, std::span<const i::SubmissionViews::StoreCapture>{},
        std::span<const i::SubmissionViews::PageCapture>{}, {}, 1));
  }
  std::array<c::ReadLease, 2> retain() {
    return {take(take(run.result(output_ids[0])).storage_lease()),
        take(take(run.result(output_ids[1])).storage_lease())};
  }
  void verify(const std::array<c::ReadLease, 2> &ready) {
    for (std::size_t domain = 0; domain != owners.size(); ++domain) {
      const auto view = take(ready[domain].view());
      EXPECT_EQ(view.resource(), owners[domain].ref());
      EXPECT_EQ(view.native_handle(), output_storage[domain].native_handle());
      EXPECT_EQ(view.offset(), output_storage[domain].offset());
      EXPECT_EQ(view.size(), output_storage[domain].size());
      std::array<float, 4> actual{};
      check(view.read(std::as_writable_bytes(std::span(actual))));
      EXPECT_EQ(actual, expected[domain]);
    }
  }
};

void outputs_capture_actual_owners(bool node_defaults = false) {
  TwoDomains fixture(node_defaults);
  ASSERT_NE(fixture.owners[0].ref(), fixture.owners[1].ref());
  EXPECT_EQ(fixture.owners[0].context_identity(), fixture.owners[1].context_identity());
  EXPECT_EQ(fixture.run.replay_route(), g::ReplayRoute::kCaptured);
  if (node_defaults) EXPECT_EQ(fixture.compiled.report.copy_bytes(), 0u);
  i::SubmissionViews views;
  fixture.prepare(views);
  auto completion = take(fixture.run.submit(fixture.queues()));
  const auto missing_outputs = views.capture_outputs(fixture.run, completion,
      std::span<const i::SubmissionViews::OutputCapture>{}, fixture.owners[0]);
  ASSERT_FALSE(missing_outputs);
  EXPECT_TRUE(missing_outputs.error().IsInvalidArgument());
  EXPECT_TRUE(completion.valid());
  EXPECT_FALSE(views.submission.valid());
  const std::array captures{
      i::SubmissionViews::OutputCapture{fixture.output_ids[0], fixture.owners[0], fixture.outputs[0]},
      i::SubmissionViews::OutputCapture{fixture.output_ids[1], fixture.owners[1], fixture.outputs[1]}};
  const auto captured = views.capture_outputs(fixture.run, completion, captures, fixture.owners[0]);
  ASSERT_TRUE(captured) << captured.error().message();
  EXPECT_TRUE(*captured);
  EXPECT_FALSE(completion.valid());
  ASSERT_TRUE(views.submission.valid()); ASSERT_TRUE(views.primary_output.valid());
  ASSERT_EQ(views.extra_outputs.size(), 1u); ASSERT_TRUE(views.extra_outputs[0].valid());
  const auto duplicate = views.capture_outputs(fixture.run, completion, captures, fixture.owners[0]);
  ASSERT_FALSE(duplicate);
  EXPECT_TRUE(duplicate.error().IsFailedPrecondition());
  EXPECT_TRUE(views.submission.valid());
  EXPECT_FALSE(completion.valid());
  EXPECT_FALSE(fixture.run.reset().ok());
  check(views.submission.wait());
  check(views.detach_outputs());
  auto ready = fixture.retain();
  fixture.verify(ready);
  check(views.submission.detach());
  views.clear_outputs(); views.submission = {};
  check(fixture.run.reset());
  fixture.verify(ready); // The same original ready permissions survive Graph withdrawal.
  ready = {};
  for (const auto &pool : fixture.outputs) EXPECT_EQ(pool.stats().live_blocks, 0u);
}

TEST(InferenceSubmissionViewsCuda, EachOutputCapturesItsActualOwnerInOneContextAndQueue) {
  outputs_capture_actual_owners();
}

TEST(InferenceSubmissionViewsCuda, NodeDefaultsPlaceBothCopiesWithoutDefaultDomainDetour) {
  outputs_capture_actual_owners(true);
}

void wrong_owner_retains_original_partial_bundle_for_recovery(std::size_t wrong_output) {
  SCOPED_TRACE(wrong_output);
  TwoDomains fixture;
  i::SubmissionViews views;
  fixture.prepare(views);
  auto completion = take(fixture.run.submit(fixture.queues()));
  const std::array captures{
      i::SubmissionViews::OutputCapture{fixture.output_ids[0], fixture.owners[wrong_output ? 0 : 1], fixture.outputs[0]},
      i::SubmissionViews::OutputCapture{fixture.output_ids[1], fixture.owners[wrong_output ? 0 : 1], fixture.outputs[1]}};
  const auto rejected = views.capture_outputs(fixture.run, completion, captures, fixture.owners[0]);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  // SubmissionSource already owns the accepted original Completion. This is
  // a partial view failure, so recovery uses that bundle without recapture.
  EXPECT_FALSE(completion.valid());
  ASSERT_TRUE(views.submission.valid()); EXPECT_EQ(views.primary_output.valid(), wrong_output != 0);
  ASSERT_EQ(views.extra_outputs.size(), 1u); EXPECT_FALSE(views.extra_outputs[0].valid());
  if (wrong_output) EXPECT_TRUE(views.submission.detach().IsFailedPrecondition());
  else {
    const auto detached = views.detach_outputs();
    EXPECT_TRUE(detached.IsFailedPrecondition());
    EXPECT_EQ(detached.message(), "output source is empty");
  }
  EXPECT_FALSE(fixture.run.reset().ok());
  for (const auto &pool : fixture.outputs) EXPECT_EQ(pool.stats().live_blocks, 1u);
  check(views.submission.wait());
  if (wrong_output) check(views.primary_output.detach());
  auto ready = fixture.retain();
  fixture.verify(ready);
  views.clear_outputs();
  check(views.submission.detach()); views.submission = {};
  check(fixture.run.reset());
  fixture.verify(ready);
  ready = {};
  for (const auto &pool : fixture.outputs) EXPECT_EQ(pool.stats().live_blocks, 0u);
}

TEST(InferenceSubmissionViewsCuda, LateWrongOwnerRetainsOriginalPartialBundleForRecovery) {
  wrong_owner_retains_original_partial_bundle_for_recovery(1);
}

TEST(InferenceSubmissionViewsCuda, FirstWrongOwnerDoesNotBecomeEffectOnlyCapture) {
  wrong_owner_retains_original_partial_bundle_for_recovery(0);
}

TEST(InferenceSubmissionViewsCuda, OriginalCompletionWithoutTypedExportsStillRequiresSuccessfulCapture) {
  TwoDomains fixture;
  i::SubmissionViews views;
  EXPECT_TRUE(views.detach_outputs().IsFailedPrecondition());
  check(views.prepare(fixture.run, std::span<const i::SubmissionViews::StoreCapture>{},
      std::span<const i::SubmissionViews::PageCapture>{}, {}));
  auto completion = take(fixture.run.submit(fixture.queues()));
  const auto captured = views.capture_outputs(fixture.run, completion,
      std::span<const i::SubmissionViews::OutputCapture>{}, fixture.owners[0]);
  ASSERT_TRUE(captured) << captured.error().message();
  EXPECT_TRUE(*captured);
  EXPECT_FALSE(completion.valid());
  ASSERT_TRUE(views.submission.valid());
  EXPECT_FALSE(views.primary_output.valid());
  EXPECT_TRUE(views.states.empty() && views.pages.empty() && views.extra_outputs.empty());
  const auto duplicate = views.capture_outputs(fixture.run, completion,
      std::span<const i::SubmissionViews::OutputCapture>{}, fixture.owners[0]);
  ASSERT_FALSE(duplicate);
  EXPECT_TRUE(duplicate.error().IsFailedPrecondition());
  check(views.submission.wait());
  check(views.detach_outputs());
  auto ready = fixture.retain();
  fixture.verify(ready);
  check(views.submission.detach());
  views.clear_outputs();
  EXPECT_TRUE(views.detach_outputs().IsFailedPrecondition());
  views.submission = {};
  check(fixture.run.reset());
  fixture.verify(ready);
  ready = {};
  for (const auto &pool : fixture.outputs) EXPECT_EQ(pool.stats().live_blocks, 0u);
}

// Two actual Store banks share one execution endpoint, but neither their
// private ID controls nor their writer permissions share an allocation domain.
struct TwoStores final {
  static constexpr std::size_t kStride = 256;
  c::Context context;
  c::Queue queue;
  std::array<c::StorageOwner, 2> owners;
  c::cuda::MemoryResource staging;
  std::array<c::LeasePool, 2> inputs, outputs;
  std::array<c::LeaseSlotLayout, 2> layouts;
  std::array<c::ReadLease, 2> input_ready, sentinels;
  std::array<m::ConstBufferView, 2> writer_storage;
  std::array<std::uint64_t, 2> writer_generations{};
  std::array<g::NodeId, 2> stores;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  const std::array<std::array<float, 4>, 2> expected{{{1.F, -2.F, 3.F, 4.F},
      {11.F, 12.F, -13.F, 14.F}}};
  const std::array<float, 4> sentinel_values{101.F, 103.F, 107.F, 109.F};

  TwoStores() {
    context = take(g::cuda::create_context());
    queue = take(context.create_queue());
    check(queue.reserve_completions(4));
    staging = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
    const auto item = take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 1>{4}));
    const auto rows = take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{1, 4}));
    const auto ids = take(c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{1}));
    const auto schema = take(s::Schema::create(item, 2, kStride));
    std::array<g::ValueId, 2> row_ids, destination_ids;
    std::array<std::array<c::WriteLease, 1>, 2> writers;
    auto builder = take(g::GraphBuilder::create());
    for (std::size_t domain = 0; domain != owners.size(); ++domain) {
      auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
      owners[domain] = take(c::StorageOwner::adopt(context, std::move(memory)));
      inputs[domain] = take(c::LeasePool::create(owners[domain].ref(), kStride, {.max_blocks = 1}));
      outputs[domain] = take(c::LeasePool::create(owners[domain].ref(), 2 * kStride, {.max_blocks = 2}));
      layouts[domain] = take(c::LeaseSlotLayout::create(outputs[domain], item, 2, kStride));
      auto input = take(inputs[domain].allocate(sizeof(expected[domain]), kStride));
      check(take(input.view()).write(std::as_bytes(std::span(expected[domain]))));
      input_ready[domain] = take(input.publish());
      auto &writer = writers[domain][0];
      writer = take(outputs[domain].allocate(sizeof(expected[domain]), kStride));
      writer_storage[domain] = take(writer.view());
      writer_generations[domain] = writer.generation();
      auto untouched = take(outputs[domain].allocate(sizeof(sentinel_values), kStride));
      check(take(untouched.view()).write(std::as_bytes(std::span(sentinel_values))));
      sentinels[domain] = take(untouched.publish());
      row_ids[domain] = take(builder.input(rows));
      destination_ids[domain] = take(builder.input(ids));
      stores[domain] = take(builder.add_node(c::Operation::custom(s::Store{schema}),
          std::array{row_ids[domain], destination_ids[domain]}));
    }
    check(builder.depends_on(stores[1], stores[0]));
    graph = take(std::move(builder).finish());
    const std::array domains{
        g::StorageDomain{.resource = owners[0].ref(), .owner = owners[0]},
        g::StorageDomain{.resource = owners[1].ref(), .owner = owners[1]}};
    const std::array targets{g::ExecutionTarget{
        .source = take(g::ExecutionSource::acquire(context)), .default_storage = g::StorageDomainId{0}}};
    const std::array contracts{
        g::ExternalResourceContract{row_ids[0], g::StorageDomainId{0}},
        g::ExternalResourceContract{destination_ids[0], g::StorageDomainId{0}},
        g::ExternalResourceContract{row_ids[1], g::StorageDomainId{1}},
        g::ExternalResourceContract{destination_ids[1], g::StorageDomainId{1}}};
    const std::array placements{
        g::NodePlacement{stores[0], g::ExecutionTargetId{0}, g::PlacementStrength::kRequired, g::StorageDomainId{0}},
        g::NodePlacement{stores[1], g::ExecutionTargetId{0}, g::PlacementStrength::kRequired, g::StorageDomainId{1}}};
    compiled = take(g::cuda::compile(graph, {.storage_domains = domains, .targets = targets},
        {.node_placements = placements, .external_resources = contracts}));
    run = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    for (std::size_t domain = 0; domain != owners.size(); ++domain)
      check(run.bind(row_ids[domain], rows, input_ready[domain]));
    check(s::prepare(run, queues(), {.host_workspace = staging.ref()}));
    std::array<s::SlotBindings, 2> transactions;
    for (std::size_t domain = 0; domain != owners.size(); ++domain)
      transactions[domain] = take(s::SlotBindings::create(layouts[domain], {}, writers[domain]));
    const std::array bindings{s::BindRequest{stores[0], &transactions[0]},
        s::BindRequest{stores[1], &transactions[1]}};
    check(s::bind(run, graph, bindings));
    check(run.prepare_replay(queues(),
        {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}));
  }
  ~TwoStores() { context.drain().IgnoreError(); }

  std::array<g::QueueBinding, 1> queues() { return {{{g::ExecutionLaneId{0}, &queue}}}; }
  hyle::Status prepare(i::SubmissionViews &views, const c::StorageOwner &second_owner) {
    const std::array captures{i::SubmissionViews::StoreCapture{stores[0], owners[0]},
        i::SubmissionViews::StoreCapture{stores[1], second_owner}};
    return views.prepare(run, captures, std::span<const i::SubmissionViews::PageCapture>{}, {});
  }
  void submit_and_verify(i::SubmissionViews &views) {
    ASSERT_EQ(views.states.size(), 2u);
    EXPECT_EQ(graph.output_count(), 0u);
    EXPECT_EQ(compiled.executable.required_queue_count(), 1u);
    auto completion = take(run.submit(queues()));
    EXPECT_TRUE(take(views.capture_outputs(run, completion,
        std::span<const i::SubmissionViews::OutputCapture>{}, owners[0])));
    EXPECT_FALSE(completion.valid());
    EXPECT_FALSE(views.primary_output.valid());
    EXPECT_TRUE(views.extra_outputs.empty());
    for (const auto &state : views.states) {
      EXPECT_TRUE(state.valid());
      EXPECT_EQ(state.destination_count(), 1u);
    }
    check(views.submission.wait());
    check(views.detach_outputs());
    check(views.submission.detach());
    std::array<c::ReadLease, 2> ready;
    for (std::size_t domain = 0; domain != owners.size(); ++domain) {
      auto returned = take(s::take(run, stores[domain]));
      ASSERT_EQ(returned.destination_count(), 1u);
      auto &writer = returned.destinations()[0];
      EXPECT_EQ(writer.generation(), writer_generations[domain]);
      ready[domain] = take(views.states[domain].publish(0, writer));
      const auto actual = take(ready[domain].view());
      EXPECT_EQ(actual.resource(), owners[domain].ref());
      EXPECT_EQ(actual.native_handle(), writer_storage[domain].native_handle());
      EXPECT_EQ(actual.offset(), writer_storage[domain].offset());
      EXPECT_EQ(actual.size(), writer_storage[domain].size());
    }
    views = {};
    check(run.reset());
    for (std::size_t domain = 0; domain != owners.size(); ++domain) {
      std::array<float, 4> actual{}, untouched{};
      check(take(ready[domain].view()).read(std::as_writable_bytes(std::span(actual))));
      check(take(sentinels[domain].view()).read(std::as_writable_bytes(std::span(untouched))));
      EXPECT_EQ(actual, expected[domain]);
      EXPECT_EQ(untouched, sentinel_values);
      ready[domain] = {}; sentinels[domain] = {};
      EXPECT_EQ(outputs[domain].stats().live_blocks, 0u);
    }
  }
};

TEST(InferenceSubmissionViewsCuda, EachStoreCapturesItsActualOwnerOnOneOriginalQueue) {
  TwoStores fixture;
  ASSERT_NE(fixture.owners[0].ref(), fixture.owners[1].ref());
  EXPECT_EQ(fixture.owners[0].context_identity(), fixture.owners[1].context_identity());
  i::SubmissionViews views;
  check(fixture.prepare(views, fixture.owners[1]));
  fixture.submit_and_verify(views);
}

TEST(InferenceSubmissionViewsCuda, WrongStoreOwnerPreservesPartialColdBundleAndOriginalWriterRetry) {
  TwoStores fixture;
  i::SubmissionViews views;
  const auto rejected = fixture.prepare(views, fixture.owners[0]);
  EXPECT_TRUE(rejected.IsFailedPrecondition());
  EXPECT_EQ(rejected.message(), "Store source Plan must retain its actual storage owner");
  ASSERT_EQ(views.states.size(), 1u);
  EXPECT_TRUE(views.states[0].prepared());
  EXPECT_FALSE(views.states[0].valid());
  EXPECT_FALSE(views.submission.valid());
  EXPECT_FALSE(fixture.run.in_flight());
  for (const auto &pool : fixture.outputs) EXPECT_EQ(pool.stats().live_blocks, 2u);
  const auto reused_partial = fixture.prepare(views, fixture.owners[1]);
  EXPECT_TRUE(reused_partial.IsFailedPrecondition());
  EXPECT_EQ(reused_partial.message(), "submission view preparation requires a fresh original bundle");
  views = {}; // No native submit occurred; return only the failed cold metadata.
  check(fixture.prepare(views, fixture.owners[1]));
  fixture.submit_and_verify(views);
}
} // namespace
