#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/recording_access.h"
#include "hyle/graph/source.h"
#include "hyle/graph/workspace.h"
#include "../../src/graph/plan_state.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;

struct WorkspaceGate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> open{false}, timed_out{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    static_cast<WorkspaceGate *>(opaque)->open.wait(false, std::memory_order_acquire);
  }
  void arm(cudaStream_t value) {
    stream = value;
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [this] { return open.load(); })) {
        timed_out.store(true);
        release();
      }
    });
  }
  void release() noexcept {
    open.store(true, std::memory_order_release);
    open.notify_all();
    changed.notify_all();
  }
  ~WorkspaceGate() {
    release();
    if (watchdog.joinable()) watchdog.join();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

struct WorkspaceRoute final {
  g::Graph graph;
  c::TensorDesc description;
  g::ValueId input, output;
};

WorkspaceRoute make_route(std::size_t rows, bool publish_input = false,
                          bool private_output = false) {
  auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(rows)}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto negated = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  const auto absolute = builder.add(c::Unary{c::UnaryKind::kAbs}, negated).value();
  const auto output = private_output
      ? builder.add(c::View::reshape(c::TensorShape::create(std::array<std::int64_t, 1>{
          static_cast<std::int64_t>(rows)}).value()).value(), absolute).value()
      : builder.add(c::Copy{}, absolute).value();
  EXPECT_TRUE(builder.publish(output).ok());
  if (publish_input) EXPECT_TRUE(builder.publish(input).ok());
  return {std::move(builder).finish().value(), description, input, output};
}

std::vector<float> values(std::size_t rows, float first) {
  std::vector<float> result(rows);
  for (std::size_t i = 0; i < rows; ++i)
    result[i] = (i % 2 ? -1.0F : 1.0F) * (first + static_cast<float>(i));
  return result;
}

void expect_absolute(c::ConstTensorView tensor, const std::vector<float> &input) {
  std::vector<float> actual(input.size());
  const auto status = tensor.storage().read(std::as_writable_bytes(std::span(actual)));
  ASSERT_TRUE(status.ok()) << status.message();
  for (std::size_t i = 0; i < input.size(); ++i) EXPECT_EQ(actual[i], std::abs(input[i]));
}

std::size_t external_bytes(const c::ResourceBudget &budget,
                           hyle::mem::MemoryResourceRef resource) {
  return budget.usage(resource).value().live_by_category[
      static_cast<std::size_t>(c::ResourceBudgetCategory::kExternal)];
}

struct ScratchExtent final {
  std::size_t bytes = 0, nonzero_backings = 0;
  hyle::mem::MemoryResourceRef resource{};
};

// Test oracle from the original private memory plan, not from the new shared
// workspace's implementation. Sum disjoint backings in plan order and insert
// each backing's required alignment gap; workspace_bytes() omits those gaps.
hyle::Result<ScratchExtent> scratch_extent(const g::Executable &route) {
  const auto *plan = g::detail::ExecutableAccess::state(route);
  if (!plan) return std::unexpected(hyle::Status::InvalidArgument("test route is empty"));
  ScratchExtent result;
  for (std::size_t slot = 0; slot < plan->workspace_plan.resource_count(); ++slot) {
    if (slot >= plan->workspace_resources.size())
      return std::unexpected(hyle::Status::Internal("test workspace resource table differs"));
    HYLE_ASSIGN_OR_RETURN(auto layout, plan->workspace_plan.resource_plan(slot));
    for (std::size_t index = 0; index < layout.backing_count; ++index) {
      HYLE_ASSIGN_OR_RETURN(auto backing, plan->workspace_plan.backing_plan(slot, index));
      if (!backing.size) continue;
      if (!backing.alignment || (backing.alignment & (backing.alignment - 1)))
        return std::unexpected(hyle::Status::Internal("test backing alignment is invalid"));
      const auto actual = plan->workspace_resources[slot];
      if (result.resource.valid() && result.resource != actual)
        return std::unexpected(hyle::Status::InvalidArgument("test route spans actual resources"));
      result.resource = actual;
      const auto remainder = result.bytes % backing.alignment;
      const auto gap = remainder ? backing.alignment - remainder : 0;
      if (gap > (std::numeric_limits<std::size_t>::max)() - result.bytes ||
          backing.size > (std::numeric_limits<std::size_t>::max)() - result.bytes - gap)
        return std::unexpected(hyle::Status::OutOfRange("test backing envelope overflows"));
      result.bytes += gap + backing.size;
      ++result.nonzero_backings;
    }
  }
  return result;
}

void shared_workspace(bool metadata_preparation) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(16).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto small = make_route(17), large = make_route(4097);
  const g::CompileOptions options{.region_selection = g::RegionSelectionPolicy::kDisabled};
  auto small_plan = g::cuda::compile(small.graph, context, storage, options);
  ASSERT_TRUE(small_plan) << small_plan.error().message();
  auto large_plan = g::cuda::compile(large.graph, context, storage, options);
  ASSERT_TRUE(large_plan) << large_plan.error().message();
  ASSERT_EQ(small_plan->executable.output_ownership().value(), g::OutputOwnership::kCaller);
  ASSERT_EQ(large_plan->executable.output_ownership().value(), g::OutputOwnership::kCaller);
  auto small_extent = scratch_extent(small_plan->executable);
  ASSERT_TRUE(small_extent) << small_extent.error().message();
  auto large_extent = scratch_extent(large_plan->executable);
  ASSERT_TRUE(large_extent) << large_extent.error().message();
  // Preserve the real multi-backing route that exposed the first contract's
  // limitation. Each route's backing chains must remain disjoint after packing.
  ASSERT_GE(small_extent->nonzero_backings, 2u);
  ASSERT_GE(large_extent->nonzero_backings, 2u);
  EXPECT_EQ(small_extent->resource, storage.ref());
  EXPECT_EQ(large_extent->resource, storage.ref());
  ASSERT_GT(large_extent->bytes, small_extent->bytes);
  const std::array limits{
      c::ResourceBudgetLimit{storage.ref(), 32u << 20},
      c::ResourceBudgetLimit{hyle::mem::system_resource(), 32u << 20}};
  auto budget = c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}).value();
  const std::array routes{small_plan->executable, large_plan->executable};
  auto shared_result = g::InvocationWorkspace::create(routes, {.resource_budget = budget});
  ASSERT_TRUE(shared_result) << shared_result.error().message();
  auto shared = std::move(*shared_result);
  ASSERT_TRUE(shared.valid());
  const auto scratch_bytes = (std::max)(small_extent->bytes, large_extent->bytes);
  EXPECT_EQ(shared.bytes(), scratch_bytes);
  // One managed LeasePool backing, charged in its existing External category.
  // This does not count all GPU storage or change its category to kWorkspace.
  EXPECT_EQ(external_bytes(budget, storage.ref()), scratch_bytes);
  auto shared_copy = shared;
  EXPECT_EQ(external_bytes(budget, storage.ref()), scratch_bytes);
  const g::InvocationOptions invocation_options{.resources = g::ResourceOptions{.resource_budget = budget}};
  auto a_result = small_plan->executable.create_invocation(invocation_options, shared);
  ASSERT_TRUE(a_result) << a_result.error().message();
  auto b_result = large_plan->executable.create_invocation(invocation_options, shared_copy);
  ASSERT_TRUE(b_result) << b_result.error().message();
  auto a = std::move(*a_result), b = std::move(*b_result);
  shared = {}; shared_copy = {};
  EXPECT_EQ(external_bytes(budget, storage.ref()), scratch_bytes);
  // Existing creation remains available and uses an independent ordinary slot.
  auto ordinary = small_plan->executable.create_invocation();
  ASSERT_TRUE(ordinary) << ordinary.error().message();
  *ordinary = {};
  auto inputs = c::LeasePool::create(storage.ref(), 1u << 20, {.max_blocks = 2}).value();
  auto outputs = c::LeasePool::create(storage.ref(), 1u << 20, {.max_blocks = 2}).value();
  const auto a_values = values(17, 1), b_values = values(4097, 1000);
  const auto upload = [&](const std::vector<float> &data) {
    auto writer = inputs.allocate(data.size() * sizeof(float), 256).value();
    EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(data))).ok());
    return writer.publish().value();
  };
  auto a_input = upload(a_values), b_input = upload(b_values);
  auto a_output = outputs.allocate(a_values.size() * sizeof(float), 256).value();
  auto b_output = outputs.allocate(b_values.size() * sizeof(float), 256).value();
  ASSERT_TRUE(a.bind(small.input, small.description, a_input).ok());
  ASSERT_TRUE(b.bind(large.input, large.description, b_input).ok());
  ASSERT_TRUE(a.bind(small.output, small.description, std::move(a_output)).ok());
  ASSERT_TRUE(b.bind(large.output, large.description, std::move(b_output)).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const g::ReplayOptions capture{.mode = g::ReplayMode::kRequireCapture,
                                 .host_wait = g::HostWaitPolicy::kReject};
  const auto prepared_a = a.prepare_replay(queues, capture);
  ASSERT_TRUE(prepared_a.ok()) << prepared_a.message();
  const auto prepared_b = b.prepare_replay(queues, capture);
  ASSERT_TRUE(prepared_b.ok()) << prepared_b.message();
  ASSERT_EQ(a.replay_route(), g::ReplayRoute::kCaptured);
  ASSERT_EQ(b.replay_route(), g::ReplayRoute::kCaptured);
  WorkspaceGate gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &WorkspaceGate::wait, &gate), cudaSuccess);
  auto pending = a.submit(queues);
  ASSERT_TRUE(pending) << pending.error().message();
  struct Retire final {
    WorkspaceGate &gate; c::Completion &completion;
    ~Retire() {
      gate.release();
      if (completion.valid()) {
        const auto status = completion.wait(); EXPECT_TRUE(status.ok()) << status.message();
        completion = {};
      }
    }
  } retire{gate, *pending};
  ASSERT_EQ(pending->poll().value(), c::CompletionPoll::kPending);
  auto collision = b.submit(queues);
  ASSERT_FALSE(collision);
  EXPECT_TRUE(collision.error().IsUnavailable()) << collision.error().message();
  EXPECT_FALSE(b.in_flight()); EXPECT_FALSE(b.output(large.output));
  if (metadata_preparation) {
    const auto prepared = b.prepare_replay(queues,
        {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    EXPECT_FALSE(gate.open.load());
    ASSERT_EQ(pending->poll().value(), c::CompletionPoll::kPending);
    const auto early = b.submit(queues);
    ASSERT_FALSE(early); EXPECT_TRUE(early.error().IsUnavailable());
    EXPECT_FALSE(b.in_flight()); EXPECT_FALSE(b.output(large.output));
  }
  ASSERT_TRUE(g::detail::RecordingAccess::begin(b).ok());
  const auto recording = b.prepare_replay(queues, capture);
  EXPECT_TRUE(recording.IsUnavailable()) << recording.message();
  EXPECT_FALSE(b.output(large.output)); EXPECT_FALSE(b.result(large.output));
  auto fake = c::Completion::completed();
  auto fake_source = g::SubmissionSource::create(b, std::move(fake), storage);
  EXPECT_FALSE(fake_source); EXPECT_TRUE(fake.valid());
  EXPECT_FALSE(b.submit(queues));
  ASSERT_TRUE(g::detail::RecordingAccess::end(b).ok());
  EXPECT_FALSE(gate.open.load());
  gate.release();
  ASSERT_TRUE(pending->wait().ok()); *pending = {};
  auto result_a = a.result(small.output);
  ASSERT_TRUE(result_a) << result_a.error().message();
  auto escaped = result_a->storage_lease();
  ASSERT_TRUE(escaped) << escaped.error().message();
  expect_absolute(result_a->tensor().value(), a_values);
  ASSERT_TRUE(b.bind(large.input, large.description, b_input).ok());
  const auto retried = b.prepare_replay(queues, capture);
  ASSERT_TRUE(retried.ok()) << retried.message();
  if (metadata_preparation) EXPECT_GT(b.replay_report()->recorded_capture_range_count, 0u);
  else {
    EXPECT_EQ(b.replay_report()->recorded_capture_range_count, 0u);
    EXPECT_GT(b.replay_report()->reused_capture_range_count, 0u);
  }
  auto completed = b.submit(queues);
  ASSERT_TRUE(completed) << completed.error().message();
  ASSERT_TRUE(completed->wait().ok()); *completed = {};
  expect_absolute(b.output(large.output).value(), b_values);
  expect_absolute(result_a->tensor().value(), a_values);
  expect_absolute(escaped->tensor(small.description).value(), a_values);
  EXPECT_EQ(external_bytes(budget, storage.ref()), scratch_bytes);
  result_a->reset(); a = {}; b = {};
  EXPECT_EQ(external_bytes(budget, storage.ref()), 0u);
  expect_absolute(escaped->tensor(small.description).value(), a_values);
  *escaped = {}; a_input = {}; b_input = {};
  EXPECT_EQ(inputs.stats().active_uses, 0u); EXPECT_EQ(inputs.stats().live_blocks, 0u);
  EXPECT_EQ(outputs.stats().active_uses, 0u); EXPECT_EQ(outputs.stats().live_blocks, 0u);
  EXPECT_FALSE(gate.timed_out.load());
}

TEST(InvocationWorkspaceCuda, ExactShapesSerializeNativeUsesAndKeepCallerResultsIndependent) { shared_workspace(false); }
TEST(InvocationWorkspaceCuda, CommandsMetadataCanPrepareWhilePriorScratchUseIsPending) { shared_workspace(true); }

TEST(InvocationWorkspaceCuda, CapturedSourceQueuesSharedScratchWithoutEarlyRetirement) {
  auto context = g::cuda::create_context().value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(16).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto route = make_route(17);
  auto compiled = g::cuda::compile(route.graph, context, storage,
      {.region_selection = g::RegionSelectionPolicy::kDisabled}).value();
  const std::array routes{compiled.executable};
  auto workspace = g::InvocationWorkspace::create(routes).value();
  auto a = compiled.executable.create_invocation({}, workspace).value();
  auto b = compiled.executable.create_invocation({}, workspace).value();
  auto inputs = c::LeasePool::create(storage.ref(), 1u << 20).value();
  auto outputs = c::LeasePool::create(storage.ref(), 1u << 20).value();
  const auto original = values(17, 10);
  auto input_writer = inputs.allocate(original.size() * sizeof(float), 256).value();
  ASSERT_TRUE(input_writer.view()->write(std::as_bytes(std::span(original))).ok());
  auto input = input_writer.publish().value();
  auto a_writer = outputs.allocate(original.size() * sizeof(float), 256).value();
  auto b_writer = outputs.allocate(original.size() * sizeof(float), 256).value();
  const auto a_view = a_writer.tensor(route.description).value().as_const();
  ASSERT_TRUE(a.bind(route.input, route.description, input).ok());
  ASSERT_TRUE(a.bind(route.output, route.description, std::move(a_writer)).ok());
  ASSERT_TRUE(b.bind(route.input, a_view).ok());
  ASSERT_TRUE(b.bind(route.output, route.description, std::move(b_writer)).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const g::ReplayOptions capture{.mode = g::ReplayMode::kRequireCapture,
      .host_wait = g::HostWaitPolicy::kReject};
  ASSERT_TRUE(a.prepare_replay(queues, capture).ok());
  ASSERT_TRUE(b.prepare_replay(queues, capture).ok());
  WorkspaceGate gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &WorkspaceGate::wait, &gate), cudaSuccess);
  auto first = a.submit(queues).value();
  ASSERT_EQ(first.poll().value(), c::CompletionPoll::kPending);
  // Same Queue alone cannot authorize an unrelated public submission.
  auto unrelated = b.submit(queues);
  ASSERT_FALSE(unrelated); EXPECT_TRUE(unrelated.error().IsUnavailable());
  auto producer = g::SubmissionSource::create(a, std::move(first), storage).value();
  auto output = g::OutputSource::create(producer, route.output, storage, outputs).value();
  const std::array bindings{output.reader_binding(route.input)};
  const auto bound = g::bind_sources(b, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto prepared = g::prepare_sources(b, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_EQ(b.replay_report()->recorded_capture_range_count, 0U);
  EXPECT_GT(b.replay_report()->reused_capture_range_count, 0U);
  auto second = g::submit_sources(b, queues);
  ASSERT_TRUE(second) << second.error().message();
  EXPECT_FALSE(gate.open.load());
  EXPECT_EQ(second->poll().value(), c::CompletionPoll::kPending);
  EXPECT_EQ(producer.poll().value(), c::CompletionPoll::kPending);
  EXPECT_GT(inputs.stats().active_uses, 0U);
  gate.release();
  ASSERT_TRUE(second->wait().ok()); *second = {};
  ASSERT_TRUE(producer.wait().ok());
  expect_absolute(b.output(route.output).value(), original);
  ASSERT_TRUE(g::release_sources(b).ok());
  output = {};
  ASSERT_TRUE(producer.detach().ok()); producer = {};
  a = {}; b = {}; input = {};
  EXPECT_EQ(inputs.stats().active_uses, 0U);
  EXPECT_EQ(outputs.stats().active_uses, 0U);
  EXPECT_FALSE(gate.timed_out.load());
}

TEST(InvocationWorkspaceCuda, RejectsNonCallerOutputsForeignResourcesAndUnlistedRoutes) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  auto other_memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto other_storage = c::StorageOwner::adopt(context, std::move(other_memory)).value();
  ASSERT_NE(storage.ref(), other_storage.ref());
  const g::CompileOptions options{.region_selection = g::RegionSelectionPolicy::kDisabled};
  const auto plain = make_route(17), alias = make_route(17, true), internal = make_route(17, false, true);
  auto compiled = g::cuda::compile(plain.graph, context, storage, options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto compiled_alias = g::cuda::compile(alias.graph, context, storage, options);
  ASSERT_TRUE(compiled_alias) << compiled_alias.error().message();
  ASSERT_EQ(compiled_alias->executable.output_ownership(1).value(), g::OutputOwnership::kSourceAlias);
  auto compiled_internal = g::cuda::compile(internal.graph, context, storage, options);
  ASSERT_TRUE(compiled_internal) << compiled_internal.error().message();
  ASSERT_EQ(compiled_internal->executable.output_ownership().value(), g::OutputOwnership::kInvocation);
  for (const auto &unsupported : {compiled_alias->executable, compiled_internal->executable}) {
    const std::array routes{unsupported};
    auto rejected = g::InvocationWorkspace::create(routes);
    ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsUnimplemented()) << rejected.error().message();
  }
  const std::array routes{compiled->executable};
  auto shared = g::InvocationWorkspace::create(routes); ASSERT_TRUE(shared) << shared.error().message();
  const auto unlisted_route = make_route(17);
  auto compatible = g::cuda::compile(unlisted_route.graph, context, storage, options);
  ASSERT_TRUE(compatible) << compatible.error().message();
  auto unlisted = compatible->executable.create_invocation({}, *shared);
  ASSERT_FALSE(unlisted); EXPECT_TRUE(unlisted.error().IsInvalidArgument()) << unlisted.error().message();
  auto foreign = g::cuda::compile(plain.graph, context, other_storage, options);
  ASSERT_TRUE(foreign) << foreign.error().message();
  const std::array incompatible{compiled->executable, foreign->executable};
  auto rejected = g::InvocationWorkspace::create(incompatible);
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument()) << rejected.error().message();
  EXPECT_FALSE(g::InvocationWorkspace::create({}));
  const std::array invalid{g::Executable{}};
  EXPECT_FALSE(g::InvocationWorkspace::create(invalid));
  EXPECT_FALSE(compiled->executable.create_invocation({}, g::InvocationWorkspace{}));
}

TEST(InvocationWorkspaceCuda, AutomaticBudgetTrackerChargesOnlyOneSharedRootAndKeepsSlotAccounting) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(16).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const std::array limits{
      c::ResourceBudgetLimit{storage.ref(), 256u << 20},
      c::ResourceBudgetLimit{hyle::mem::system_resource(), 256u << 20}};
  auto budget = c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}).value();
  auto other_budget = c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}).value();
  // ResourceBudget charges its own State + resource rows as metadata for its
  // complete lifetime. Record each scope before any Graph/workspace ownership.
  const std::array host_baselines{
      budget.usage(hyle::mem::system_resource()).value(),
      other_budget.usage(hyle::mem::system_resource()).value()};
  for (const auto &baseline : host_baselines) {
    ASSERT_GT(baseline.live_bytes, 0u);
    EXPECT_EQ(baseline.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)],
        baseline.live_bytes);
  }
  const auto category = [](const c::ResourceBudget &scope, hyle::mem::MemoryResourceRef resource,
                           c::ResourceBudgetCategory value) {
    return scope.usage(resource).value().live_by_category[static_cast<std::size_t>(value)];
  };
  struct PrivateFacts final { std::size_t bytes = 0, unknown = 0; };
  const auto private_facts = [](const g::Executable &route) {
    PrivateFacts result;
    const auto *plan = g::detail::ExecutableAccess::state(route);
    for (const auto &step : plan->steps) {
      if (!step.emits_command()) continue;
      std::optional<std::size_t> bytes;
      if (step.kind == g::detail::PlanStepKind::kOperation)
        bytes = step.operation.invocation_private_bytes();
      else if (step.region.valid()) bytes = step.region.properties().invocation_private_bytes;
      else continue;
      if (bytes) result.bytes += *bytes;
      else ++result.unknown;
    }
    return result;
  };
  {
    const auto small = make_route(17), large = make_route(4097);
    // Supplying only resource_budget must create the original automatic MAX
    // tracker. Do not replace it with an untracked or privately rewritten plan.
    const g::CompileOptions options{.resource_budget = budget,
        .region_selection = g::RegionSelectionPolicy::kDisabled};
    auto first = g::cuda::compile(small.graph, context, storage, options);
    ASSERT_TRUE(first) << first.error().message();
    auto second = g::cuda::compile(large.graph, context, storage, options);
    ASSERT_TRUE(second) << second.error().message();
    for (const auto &route : {first->executable, second->executable}) {
      auto tracker = route.invocation_storage_usage();
      ASSERT_TRUE(tracker);
      EXPECT_EQ(tracker->limit_bytes, (std::numeric_limits<std::size_t>::max)());
      EXPECT_EQ(tracker->live_bytes, 0u);
    }
    const auto first_private = private_facts(first->executable);
    const auto second_private = private_facts(second->executable);
    const std::array routes{first->executable, second->executable};
    auto missing_scope = g::InvocationWorkspace::create(routes);
    ASSERT_FALSE(missing_scope);
    EXPECT_TRUE(missing_scope.error().IsInvalidArgument()) << missing_scope.error().message();
    auto wrong_scope = g::InvocationWorkspace::create(routes, {.resource_budget = other_budget});
    ASSERT_FALSE(wrong_scope);
    EXPECT_TRUE(wrong_scope.error().IsInvalidArgument()) << wrong_scope.error().message();
    auto foreign_options = options;
    foreign_options.resource_budget = other_budget;
    auto foreign = g::cuda::compile(large.graph, context, storage, foreign_options);
    ASSERT_TRUE(foreign) << foreign.error().message();
    const std::array mixed_scopes{first->executable, foreign->executable};
    auto mixed = g::InvocationWorkspace::create(mixed_scopes, {.resource_budget = budget});
    ASSERT_FALSE(mixed);
    EXPECT_TRUE(mixed.error().IsInvalidArgument()) << mixed.error().message();
    auto finite_options = options;
    finite_options.invocation_storage = g::InvocationStorageLimits{
        .max_bytes = 64u << 20, .unknown = g::UnknownStoragePolicy::kAllow};
    auto finite = g::cuda::compile(small.graph, context, storage, finite_options);
    ASSERT_TRUE(finite) << finite.error().message();
    EXPECT_EQ(finite->executable.invocation_storage_usage()->limit_bytes, 64u << 20);
    const std::array finite_routes{finite->executable};
    auto finite_shared = g::InvocationWorkspace::create(finite_routes, {.resource_budget = budget});
    ASSERT_FALSE(finite_shared);
    EXPECT_TRUE(finite_shared.error().IsUnimplemented()) << finite_shared.error().message();
    auto same_scope_copy = budget;
    auto shared = g::InvocationWorkspace::create(routes, {.resource_budget = same_scope_copy});
    ASSERT_TRUE(shared) << shared.error().message();
    const auto root_bytes = shared->bytes();
    ASSERT_GT(root_bytes, 0u);
    EXPECT_EQ(external_bytes(budget, storage.ref()), root_bytes);
    EXPECT_EQ(category(budget, storage.ref(), c::ResourceBudgetCategory::kWorkspace), 0u);
    const auto metadata_before = category(budget, hyle::mem::system_resource(), c::ResourceBudgetCategory::kMetadata);
    const auto private_before = budget.unattributed_usage();
    const g::InvocationOptions invocation_options{.resources = g::ResourceOptions{.resource_budget = budget}};
    auto a = first->executable.create_invocation(invocation_options, *shared);
    ASSERT_TRUE(a) << a.error().message();
    auto b = second->executable.create_invocation(invocation_options, *shared);
    ASSERT_TRUE(b) << b.error().message();
    EXPECT_EQ(external_bytes(budget, storage.ref()), root_bytes);
    EXPECT_EQ(category(budget, storage.ref(), c::ResourceBudgetCategory::kWorkspace), 0u);
    EXPECT_GT(category(budget, hyle::mem::system_resource(), c::ResourceBudgetCategory::kMetadata), metadata_before);
    const auto private_live = budget.unattributed_usage();
    EXPECT_EQ(private_live.private_upper_bound_bytes,
        private_before.private_upper_bound_bytes + first_private.bytes + second_private.bytes);
    EXPECT_EQ(private_live.unknown_private_owners,
        private_before.unknown_private_owners + first_private.unknown + second_private.unknown);
    const auto first_live = first->executable.invocation_storage_usage().value();
    const auto second_live = second->executable.invocation_storage_usage().value();
    EXPECT_GT(first_live.live_bytes, 0u); EXPECT_GE(first_live.peak_bytes, first_live.live_bytes);
    EXPECT_GT(second_live.live_bytes, 0u); EXPECT_GE(second_live.peak_bytes, second_live.live_bytes);
    EXPECT_EQ(first_live.has_unknown_storage, first_private.unknown != 0);
    EXPECT_EQ(second_live.has_unknown_storage, second_private.unknown != 0);
    // Capture has unknown native storage under the existing partial-accounting
    // contract, even when every provider declares a finite slot-private bound.
    auto io = c::LeasePool::create(storage.ref(), 512, {.max_blocks = 2}).value();
    auto input_writer = io.allocate(17 * sizeof(float), 256).value();
    const auto input_values = values(17, 1);
    ASSERT_TRUE(input_writer.view()->write(std::as_bytes(std::span(input_values))).ok());
    auto input = input_writer.publish().value();
    auto output = io.allocate(17 * sizeof(float), 256).value();
    ASSERT_TRUE(a->bind(small.input, small.description, input).ok());
    ASSERT_TRUE(a->bind(small.output, small.description, std::move(output)).ok());
    ASSERT_TRUE(g::detail::RecordingAccess::begin(*a).ok());
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    const auto captured = a->prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(captured.ok()) << captured.message();
    EXPECT_TRUE(first->executable.invocation_storage_usage()->has_unknown_storage);
    EXPECT_FALSE(a->in_flight()); EXPECT_FALSE(a->output(small.output));
    ASSERT_TRUE(g::detail::RecordingAccess::end(*a).ok());
    *a = {}; *b = {};
    EXPECT_EQ(first->executable.invocation_storage_usage()->live_bytes, 0u);
    EXPECT_EQ(second->executable.invocation_storage_usage()->live_bytes, 0u);
    const auto private_after = budget.unattributed_usage();
    EXPECT_EQ(private_after.private_upper_bound_bytes, private_before.private_upper_bound_bytes);
    EXPECT_EQ(private_after.unknown_private_owners, private_before.unknown_private_owners);
    EXPECT_EQ(external_bytes(budget, storage.ref()), root_bytes);
    *shared = {};
    EXPECT_EQ(external_bytes(budget, storage.ref()), 0u);
    EXPECT_EQ(category(budget, storage.ref(), c::ResourceBudgetCategory::kWorkspace), 0u);
  }
  // Reports/plans and automatic trackers have released all additional charges.
  // The scopes are still alive, so their own control must match the exact
  // pre-Graph baseline, with no tolerance or ignored residual allocation.
  std::size_t scope_index = 0;
  for (const auto &scope : {budget, other_budget}) {
    EXPECT_EQ(scope.usage(storage.ref())->live_bytes, 0u);
    const auto host = scope.usage(hyle::mem::system_resource()).value();
    EXPECT_EQ(host.live_bytes, host_baselines[scope_index].live_bytes);
    EXPECT_EQ(host.live_by_category, host_baselines[scope_index].live_by_category);
    EXPECT_EQ(scope.unattributed_usage().private_upper_bound_bytes, 0u);
    EXPECT_EQ(scope.unattributed_usage().unknown_private_owners, 0u);
    ++scope_index;
  }
}
} // namespace
