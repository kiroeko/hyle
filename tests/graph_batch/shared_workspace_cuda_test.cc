#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/batch_graph_access.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/recording_access.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/workspace.h"
#include "../../src/graph/plan_state.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace g = hyle::graph;
namespace gb = g::batch;
namespace gd = g::detail;
namespace m = hyle::mem;
constexpr std::uint32_t depth = 64, max_rows = 128, kv_rows = 4096;
constexpr std::size_t float_cap = 64u << 20;

struct SharedBatchGate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> open{false}, timed_out{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    static_cast<SharedBatchGate *>(opaque)->open.wait(false, std::memory_order_acquire);
  }
  void arm(cudaStream_t value) {
    stream = value;
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10), [this] { return open.load(); })) {
        timed_out.store(true); release();
      }
    });
  }
  void release() noexcept {
    open.store(true, std::memory_order_release); open.notify_all(); changed.notify_all();
  }
  ~SharedBatchGate() {
    release();
    if (watchdog.joinable()) watchdog.join();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

struct Environment final {
  c::Context context = g::cuda::create_context().value();
  c::Queue queue = context.create_queue().value();
  c::StorageOwner storage = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  c::cuda::MemoryResource host = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kPinnedHost).value();
  c::BatchCapacity capacity;
  c::BatchSchema schema;
  c::ResourceBudget budget;
  Environment() {
    EXPECT_TRUE(queue.reserve_completions(32).ok());
    const auto plain = c::CacheStorage::plain(c::DataType::kFloat16).value();
    capacity = {.storage = c::BatchStorage::kRagged,
        .query_type = c::DataType::kFloat16, .output_type = c::DataType::kFloat16,
        .key_storage = plain, .value_storage = plain, .max_requests = 1,
        .max_query_rows = max_rows, .max_kv_rows = kv_rows,
        .max_pages = 0, .max_page_refs = 0, .query_heads = 1, .kv_heads = 1,
        .qk_head_dim = depth, .value_head_dim = depth, .page_size = 1};
    schema = c::BatchSchema::create(capacity).value();
    const std::array limits{c::ResourceBudgetLimit{storage.ref(), 256u << 20},
        c::ResourceBudgetLimit{m::system_resource(), 128u << 20},
        c::ResourceBudgetLimit{host.ref(), 128u << 20}};
    budget = c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}).value();
  }
  ~Environment() { context.drain().IgnoreError(); }
  gb::PrepareOptions options() const {
    return {.host_workspace = host.ref(), .max_float_workspace_bytes = float_cap,
        .max_int_workspace_bytes = 8u << 20, .resource_budget = budget};
  }
  c::BatchLayoutVersion layout(std::uint32_t active, std::uint32_t keys = kv_rows) const {
    const std::array<std::int32_t, 2> q{0, std::int32_t(active)}, k{0, std::int32_t(keys)};
    return schema.version({q, k, {}, {}}).value();
  }
};

struct Route final {
  g::Graph graph;
  g::CompiledGraph compiled;
  std::array<g::ValueId, 4> inputs;
  std::array<g::NodeId, 2> attention;
  g::ValueId output;
  c::TensorDesc query_description, output_description;
};
Route route(Environment &e, std::uint32_t rows) {
  Route result;
  result.query_description = c::batch_query_description(e.capacity, rows).value();
  result.output_description = c::batch_output_description(e.capacity, rows).value();
  auto builder = g::GraphBuilder::create().value();
  result.inputs[0] = builder.input(result.query_description).value();
  result.inputs[1] = builder.input(c::batch_cache_description(e.capacity, c::BatchCacheRole::kKey).value()).value();
  result.inputs[2] = builder.input(c::batch_cache_description(e.capacity, c::BatchCacheRole::kValue).value()).value();
  result.inputs[3] = builder.input(e.schema.metadata_description().value()).value();
  auto query = result.inputs[0];
  for (auto &node : result.attention) {
    node = builder.add_node(c::Operation::custom(c::RaggedAttention{
        .capacity = e.capacity, .scale = .125F, .causal = false}),
        std::array{query, result.inputs[1], result.inputs[2], result.inputs[3]}).value();
    query = builder.result(node).value();
  }
  // The original Q data edge orders the two native users of disposable scratch.
  EXPECT_TRUE(builder.depends_on(result.attention[1], result.attention[0]).ok());
  result.output = query; EXPECT_TRUE(builder.publish(query).ok());
  result.graph = std::move(builder).finish().value();
  result.compiled = g::cuda::compile(result.graph, e.context, e.storage,
      {.operation_fallback = g::OperationFallbackPolicy::kReject}).value();
  return result;
}

// Independent physical-envelope oracle from the original planner. Distinct
// backings remain disjoint, and each route inserts its original alignment gaps.
std::size_t extent(const g::Executable &route) {
  const auto *plan = gd::ExecutableAccess::state(route);
  std::size_t bytes = 0;
  for (std::size_t resource = 0; resource < plan->workspace_plan.resource_count(); ++resource) {
    const auto layout = plan->workspace_plan.resource_plan(resource).value();
    for (std::size_t index = 0; index < layout.backing_count; ++index) {
      const auto backing = plan->workspace_plan.backing_plan(resource, index).value();
      if (!backing.size) continue;
      const auto remainder = bytes % backing.alignment;
      const auto gap = remainder ? backing.alignment - remainder : 0;
      EXPECT_LE(gap, (std::numeric_limits<std::size_t>::max)() - bytes);
      bytes += gap;
      EXPECT_LE(backing.size, (std::numeric_limits<std::size_t>::max)() - bytes);
      bytes += backing.size;
    }
  }
  return bytes;
}

struct BoundRoute final {
  Environment &environment;
  const Route &route;
  m::Buffer query, keys, values, output;
  g::Invocation run;
  BoundRoute(Environment &e, const Route &r, const g::InvocationWorkspace &workspace)
      : environment(e), route(r) {
    query = m::Buffer::create(e.storage.ref(), r.query_description.required_bytes().value(), 256).value();
    keys = m::Buffer::create(e.storage.ref(), c::batch_cache_description(e.capacity, c::BatchCacheRole::kKey)->required_bytes().value(), 256).value();
    values = m::Buffer::create(e.storage.ref(), keys.size(), 256).value();
    output = m::Buffer::create(e.storage.ref(), r.output_description.required_bytes().value(), 256).value();
    EXPECT_TRUE(query.view().fill(std::byte{0}).ok()); EXPECT_TRUE(keys.view().fill(std::byte{0}).ok());
    // Uniform probabilities: the first half of the true 4096 V rows is 1,
    // the second half is 3. The full-cache oracle is exactly representable 2.
    std::vector<std::uint16_t> words(kv_rows * depth, 0x3c00u);
    std::fill(words.begin() + words.size() / 2, words.end(), std::uint16_t{0x4200});
    EXPECT_TRUE(values.view().write(std::as_bytes(std::span(words))).ok());
    EXPECT_TRUE(output.view().fill(std::byte{0x5a}).ok());
    run = r.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = e.budget}}, workspace).value();
    EXPECT_TRUE(run.bind(r.inputs[0], c::ConstTensorView::create(r.query_description, query.view()).value()).ok());
    EXPECT_TRUE(run.bind(r.inputs[1], c::ConstTensorView::create(
        c::batch_cache_description(e.capacity, c::BatchCacheRole::kKey).value(), keys.view()).value()).ok());
    EXPECT_TRUE(run.bind(r.inputs[2], c::ConstTensorView::create(
        c::batch_cache_description(e.capacity, c::BatchCacheRole::kValue).value(), values.view()).value()).ok());
    EXPECT_TRUE(run.bind(r.output, c::TensorView::create(r.output_description, output.view()).value()).ok());
  }
  // Raw caller IO remains alive even if a fatal assertion abandons a real run.
  ~BoundRoute() { environment.context.drain().IgnoreError(); }
  hyle::Status prepare(const c::BatchLayoutVersion &layout) {
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &environment.queue}};
    const std::array layouts{gb::NodeLayout{route.attention[0], layout}, gb::NodeLayout{route.attention[1], layout}};
    return gb::prepare(run, queues, layouts, environment.options());
  }
  m::BufferView scratch(std::size_t node) {
    auto *state = gd::InvocationAccess::state(run);
    const auto step = state->plan->node_steps[route.attention[node].ordinal()];
    const auto request = state->plan->step_workspace_requests[step];
    EXPECT_NE(request, gd::kNoWorkspaceRequest);
    EXPECT_GT(state->plan->steps[step].workspace_requirement().size, 0u);
    EXPECT_LE(state->plan->steps[step].workspace_requirement().size, float_cap);
    return state->workspace.view(request).value();
  }
  void check(std::uint32_t active, std::uint16_t expected = 0x4000u) const {
    std::vector<std::uint16_t> actual(output.size() / 2);
    ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], i < std::size_t(active) * depth ? expected : 0u) << i;
  }
};

g::InvocationWorkspace shared(Environment &e, const Route &a, const Route &b) {
  const std::array routes{a.compiled.executable, b.compiled.executable};
  auto result = g::InvocationWorkspace::create(routes, {.resource_budget = e.budget}).value();
  EXPECT_EQ(result.bytes(), (std::max)(extent(routes[0]), extent(routes[1])));
  // Two serial attention steps and two mutually exclusive capacities do not
  // reserve four configured float payloads. Other live tensor spans remain.
  EXPECT_GE(result.bytes(), float_cap); EXPECT_LT(result.bytes(), 2 * float_cap);
  EXPECT_EQ(e.budget.usage(e.storage.ref())->live_by_category[
      std::size_t(c::ResourceBudgetCategory::kExternal)], result.bytes());
  return result;
}
void same_scratch(BoundRoute &route) {
  const auto first = route.scratch(0), second = route.scratch(1);
  EXPECT_EQ(first.resource(), second.resource()); EXPECT_EQ(first.native_handle(), second.native_handle());
  EXPECT_EQ(first.offset(), second.offset()); EXPECT_EQ(first.size(), second.size());
}

TEST(GraphBatchSharedWorkspaceCuda, SerialAttentionAndExactRoutesBorrowOnePlannerRoot) {
  Environment e;
  const auto small = route(e, 16), large = route(e, 128);
  auto workspace = shared(e, small, large);
  BoundRoute a(e, small, workspace), b(e, large, workspace);
  same_scratch(a); same_scratch(b);
  EXPECT_EQ(a.scratch(0).native_handle(), b.scratch(0).native_handle());
  EXPECT_EQ(a.scratch(0).resource(), b.scratch(0).resource());
  ASSERT_TRUE(a.prepare(e.layout(3)).ok()); ASSERT_TRUE(b.prepare(e.layout(105)).ok());
  {
    // Float is shared; the actual uploaded integer metadata remains private
    // to each native slot. Drop these real readers before another prepare.
    const auto a_metadata = gb::query_offsets(a.run, small.attention[0]).value();
    const auto b_metadata = gb::query_offsets(b.run, large.attention[0]).value();
    const auto av = a_metadata.view().value(), bv = b_metadata.view().value();
    EXPECT_TRUE(av.native_handle() != bv.native_handle() || av.offset() != bv.offset());
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &e.queue}};
  const g::ReplayOptions commands{.mode = g::ReplayMode::kCommands};
  ASSERT_TRUE(a.run.prepare_replay(queues, commands).ok());
  auto first = a.run.submit(queues); ASSERT_TRUE(first) << first.error().message();
  ASSERT_TRUE(first->wait().ok()); *first = {}; a.check(3);
  auto retained = a.run.result(small.output).value();
  ASSERT_TRUE(b.run.prepare_replay(queues, commands).ok());
  auto second = b.run.submit(queues); ASSERT_TRUE(second) << second.error().message();
  ASSERT_TRUE(second->wait().ok()); *second = {}; b.check(105); a.check(3);
  EXPECT_TRUE(retained.tensor().has_value());
  for (const auto node : large.attention) {
    const auto report = gb::report(b.run, node).value();
    EXPECT_GT(report.float_workspace_bytes, 0u);
    EXPECT_GT(report.int_workspace_bytes, 0u);
    EXPECT_LE(report.float_workspace_bytes, float_cap);
  }
}

TEST(GraphBatchSharedWorkspaceCuda, DecodeVerifyAndPrefillReuseScratchAcrossConfigurationReset) {
  Environment e;
  const auto decode = route(e, 1), verify = route(e, 2), prefill = route(e, 128);
  const std::array routes{decode.compiled.executable, verify.compiled.executable, prefill.compiled.executable};
  auto workspace = g::InvocationWorkspace::create(routes, {.resource_budget = e.budget}).value();
  BoundRoute a(e, decode, workspace), b(e, verify, workspace), c(e, prefill, workspace);
  same_scratch(a); same_scratch(b); same_scratch(c);
  EXPECT_EQ(a.scratch(0).native_handle(), c.scratch(0).native_handle());
  EXPECT_EQ(b.scratch(0).native_handle(), c.scratch(0).native_handle());
  EXPECT_EQ(workspace.bytes(), (std::max)({extent(routes[0]), extent(routes[1]), extent(routes[2])}));
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &e.queue}};
  const g::ReplayOptions capture{.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kAllow};
  for (std::size_t cycle = 0; cycle != 2; ++cycle) {
    for (auto *bound : {&c, &a, &b}) {
      const auto active = bound == &a ? 1u : bound == &b ? 2u : 105u;
      if (cycle) ASSERT_TRUE(gd::RegionAccess::reset_configuration(bound->run).ok());
      ASSERT_TRUE(bound->prepare(e.layout(active)).ok());
      ASSERT_TRUE(bound->run.prepare_replay(queues, capture).ok());
      if (cycle) EXPECT_GT(bound->run.replay_report()->reused_capture_range_count, 0u);
      auto submitted = bound->run.submit(queues); ASSERT_TRUE(submitted) << submitted.error().message();
      ASSERT_TRUE(submitted->wait().ok()); *submitted = {}; bound->check(active);
    }
    // All public results retain their original caller IO after other shapes
    // overwrite the common partial buffer; none escapes through scratch.
    a.check(1); b.check(2); c.check(105);
    const auto a_plan = gb::query_offsets(a.run, decode.attention[0]).value();
    const auto b_plan = gb::query_offsets(b.run, verify.attention[0]).value();
    const auto av = a_plan.view().value(), bv = b_plan.view().value();
    EXPECT_TRUE(av.native_handle() != bv.native_handle() || av.offset() != bv.offset());
  }
}

TEST(GraphBatchSharedWorkspaceCuda, PendingRootAllowsAddressPlanningButRefusesRunAndNoSubmitRecording) {
  Environment e;
  const auto small = route(e, 16), large = route(e, 128);
  auto workspace = shared(e, small, large);
  BoundRoute a(e, small, workspace), b(e, large, workspace);
  const auto a_layout = e.layout(3), b_layout = e.layout(105);
  ASSERT_TRUE(a.prepare(a_layout).ok()); ASSERT_TRUE(b.prepare(b_layout).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &e.queue}};
  const g::ReplayOptions capture{.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject};
  ASSERT_TRUE(a.run.prepare_replay(queues, capture).ok());
  // Observe float only while idle. A successful NoSubmit capture must not
  // execute the native attention or modify its disposable float payload.
  auto scratch = b.scratch(0);
  ASSERT_TRUE(scratch.fill(std::byte{0x39}).ok());
  const auto b_before = std::vector<std::byte>(b.output.size(), std::byte{0x5a});
  ASSERT_TRUE(gd::RecordingAccess::begin(b.run).ok());
  ASSERT_TRUE(b.run.prepare_replay(queues, capture).ok());
  ASSERT_GT(b.run.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_FALSE(b.run.submit(queues));
  ASSERT_TRUE(gd::RecordingAccess::end(b.run).ok());
  std::vector<std::byte> observed(scratch.size());
  ASSERT_TRUE(scratch.read(observed).ok());
  EXPECT_TRUE(std::all_of(observed.begin(), observed.end(), [](auto x) { return x == std::byte{0x39}; }));
  std::vector<std::byte> output_before(b_before.size());
  ASSERT_TRUE(b.output.view().read(output_before).ok()); EXPECT_EQ(output_before, b_before);
  ASSERT_TRUE(b.run.prepare_replay(queues, capture).ok());
  SharedBatchGate gate; gate.arm(c::cuda::stream(e.queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &SharedBatchGate::wait, &gate), cudaSuccess);
  auto pending = a.run.submit(queues); ASSERT_TRUE(pending) << pending.error().message();
  struct Retire final {
    SharedBatchGate &gate; c::Completion &completion;
    ~Retire() { gate.release(); if (completion.valid()) { completion.wait().IgnoreError(); completion = {}; } }
  } retire{gate, *pending};
  ASSERT_EQ(pending->poll().value(), c::CompletionPoll::kPending);
  auto collision = b.run.submit(queues); ASSERT_FALSE(collision);
  EXPECT_TRUE(collision.error().IsUnavailable()) << collision.error().message();
  // Only private int/host tables may change: the provider promises that this
  // real planning/upload transaction never reads or writes borrowed float.
  const auto updated_layout = e.layout(106);
  const auto planned = b.prepare(updated_layout); ASSERT_TRUE(planned.ok()) << planned.message();
  EXPECT_EQ(gb::report(b.run, large.attention[0])->layout_generation, updated_layout.generation());
  EXPECT_FALSE(gate.open.load()); EXPECT_FALSE(gate.timed_out.load());
  ASSERT_TRUE(gd::RecordingAccess::begin(b.run).ok());
  const auto rejected = b.run.prepare_replay(queues, capture);
  EXPECT_TRUE(rejected.IsUnavailable()) << rejected.message();
  EXPECT_FALSE(b.run.submit(queues));
  ASSERT_TRUE(gd::RecordingAccess::end(b.run).ok());
  EXPECT_FALSE(gate.open.load());
  // No diagnostic read of scratch or output occurs while A is in flight.
  gate.release(); ASSERT_TRUE(pending->wait().ok()); *pending = {}; a.check(3);
  ASSERT_TRUE(b.run.prepare_replay(queues, capture).ok());
  auto finished = b.run.submit(queues); ASSERT_TRUE(finished) << finished.error().message();
  ASSERT_TRUE(finished->wait().ok()); *finished = {}; b.check(106);
  EXPECT_FALSE(gate.timed_out.load());
}

TEST(GraphBatchSharedWorkspaceCuda, BorrowedViewValidatesActualCapacityResourceAndAlignment) {
  Environment e;
  constexpr std::size_t cap = 1u << 20;
  auto program = c::BatchProgram::prepare(e.context, e.schema,
      {.resource = e.storage.ref(), .host_workspace = e.host.ref(),
       .max_float_workspace_bytes = cap, .max_int_workspace_bytes = 1u << 20}).value();
  auto metadata_queue = std::make_shared<c::Queue>(e.context.create_queue().value());
  auto backing = m::Buffer::create(e.storage.ref(), cap + 32, 256).value();
  const auto create = [&](m::BufferView view) {
    // Bounds qualification only: never submit through these trusted private
    // coordinates without the real outer Graph root admission.
    return cd::BatchGraphAccess::create(program, e.queue, metadata_queue, {view});
  };
  auto absent = create({}); ASSERT_FALSE(absent); EXPECT_TRUE(absent.error().IsInvalidArgument());
  {
    auto below_cap = create(backing.view().subview(0, cap - 16).value());
    ASSERT_TRUE(below_cap) << below_cap.error().message();
  }
  {
    auto tiny = create(backing.view().subview(0, 16).value());
    ASSERT_TRUE(tiny) << tiny.error().message();
    auto refused = cd::BatchGraphAccess::prepare(*tiny, e.layout(16), true, {});
    ASSERT_FALSE(refused); EXPECT_TRUE(refused.error().IsResourceExhausted()) << refused.error().message();
    EXPECT_FALSE(tiny->in_flight());
  }
  auto unaligned = create(backing.view().subview(1, cap).value());
  ASSERT_FALSE(unaligned); EXPECT_TRUE(unaligned.error().IsInvalidArgument());
  auto other_resource = c::cuda::MemoryResource::create(e.context, c::cuda::MemoryKind::kDevice).value();
  auto other = m::Buffer::create(other_resource.ref(), cap, 256).value();
  auto foreign = create(other.view()); ASSERT_FALSE(foreign); EXPECT_TRUE(foreign.error().IsInvalidArgument());
  auto accepted = create(backing.view().subview(16, cap).value());
  ASSERT_TRUE(accepted) << accepted.error().message(); EXPECT_FALSE(accepted->in_flight());
}
} // namespace
