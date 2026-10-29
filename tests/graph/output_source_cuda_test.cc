#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/output_source.h"
#include "hyle/graph/detail/submission_source.h"
#include "hyle/graph/detail/store_source.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/detail/replay_access.h"
#include "hyle/graph/detail/region_adapter.h"
#include "hyle/graph/region_provider.h"
#include "hyle/graph/slots.h"
#include "hyle/graph/graph.h"
#include "../../src/graph/slots_region.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gd = g::detail;

// Original Queue host gate, never a device spin kernel. The watchdog only
// releases this test gate, so a bad unexpected host wait cannot hang the GPU.
struct Gate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false}, timed_out{false};
  std::mutex lock;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<Gate *>(opaque);
    gate.released.wait(false, std::memory_order_acquire);
  }
  void arm(cudaStream_t value) {
    stream = value;
    watchdog = std::thread([this] {
      std::unique_lock guard(lock);
      if (!changed.wait_for(guard, std::chrono::seconds(10),
              [this] { return released.load(std::memory_order_acquire); })) {
        timed_out.store(true, std::memory_order_release);
        release();
      }
    });
  }
  void release() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all(); changed.notify_all();
  }
  ~Gate() {
    release();
    if (watchdog.joinable()) watchdog.join();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

void relay(bool bad_producer_index, bool reuse_producer = false, bool region_consumer = false) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(8).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  const auto table_i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{4}).value();
  const auto table_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4, 2}).value();
  const auto row_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 2}).value();
  auto inputs = c::LeasePool::create(storage.ref(), 3 * 256, {.max_blocks = 3}).value();
  const auto source_blocks = reuse_producer ? 2u : 1u;
  auto source_pool = c::LeasePool::create(storage.ref(), source_blocks * 256,
      {.max_blocks = source_blocks}).value();
  auto output_pool = c::LeasePool::create(storage.ref(), 256, {.max_blocks = 1}).value();
  c::cuda::MemoryResource staging;
  if (region_consumer) staging = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kPinnedHost).value();
  auto table = inputs.allocate(4 * sizeof(std::int32_t), 256).value();
  auto index = inputs.allocate(sizeof(std::int32_t), 256).value();
  auto embedding = inputs.allocate(8 * sizeof(float), 256).value();
  constexpr std::array<std::int32_t, 4> tokens{2, 1, 3, 0};
  const std::array<std::int32_t, 1> select{bad_producer_index ? 4 : 0};
  constexpr std::array<float, 8> rows{10, 11, 20, 21, 30, 31, 40, 41};
  ASSERT_TRUE(table.view()->write(std::as_bytes(std::span(tokens))).ok());
  ASSERT_TRUE(index.view()->write(std::as_bytes(std::span(select))).ok());
  ASSERT_TRUE(embedding.view()->write(std::as_bytes(std::span(rows))).ok());
  auto table_ready = table.publish().value(), index_ready = index.publish().value();
  auto embedding_ready = embedding.publish().value();
  // Prepare every new payload/allocation before any native host gate. The
  // reuse step itself only binds these existing genuine owners.
  c::LeasePool next_index_pool;
  c::ReadLease next_index;
  c::WriteLease next_token_writer;
  if (reuse_producer) {
    next_index_pool = c::LeasePool::create(storage.ref(), 256, {.max_blocks = 1}).value();
    auto prepared_index = next_index_pool.allocate(sizeof(std::int32_t), 256).value();
    constexpr std::array<std::int32_t, 1> other_index{1};
    ASSERT_TRUE(prepared_index.view()->write(std::as_bytes(std::span(other_index))).ok());
    next_index = prepared_index.publish().value();
  }

  auto first = g::GraphBuilder::create().value();
  const auto first_table = first.input(table_i32).value();
  const auto first_index = first.input(i32).value();
  const auto token = first.add(c::Gather{.axis = 0}, first_table, first_index).value();
  ASSERT_TRUE(first.publish(token).ok());
  auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage);
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(first_table, table_i32, table_ready).ok());
  ASSERT_TRUE(producer.bind(first_index, i32, index_ready).ok());
  auto token_writer = source_pool.allocate(sizeof(std::int32_t), 256).value();
  if (reuse_producer) {
    next_token_writer = source_pool.allocate(sizeof(std::int32_t), 256).value();
    const auto original_storage = token_writer.view().value();
    const auto next_storage = next_token_writer.view().value();
    EXPECT_EQ(original_storage.native_handle(), next_storage.native_handle());
    EXPECT_NE(original_storage.offset(), next_storage.offset());
  }
  ASSERT_TRUE(producer.bind(token, i32, std::move(token_writer)).ok());

  auto second = g::GraphBuilder::create().value();
  const auto second_table = second.input(table_f32).value();
  const auto second_index = second.input(i32).value();
  const auto selected = second.add(c::Gather{.axis = 0}, second_table, second_index).value();
  const auto item = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}).value();
  g::NodeId store;
  if (region_consumer) {
    const auto ids = second.input(i32).value();
    store = second.add_node(c::Operation::custom(g::slots::Store{
        g::slots::Schema::create(item, 1, 256).value()}), std::array{selected, ids}).value();
  } else {
    ASSERT_TRUE(second.publish(selected).ok());
  }
  auto second_graph = std::move(second).finish().value();
  auto consumer_plan = g::cuda::compile(second_graph, context, storage);
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  if (region_consumer) {
    ASSERT_EQ(consumer_plan->report.region_count(), 1u);
    ASSERT_EQ(consumer_plan->report.step_count(), 2u);
    bool store_region = false;
    for (std::size_t i = 0; i < consumer_plan->report.step_count(); ++i) {
      const auto step = consumer_plan->report.step(i).value();
      EXPECT_EQ(step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
      EXPECT_EQ(step.runtime.replay_class, c::ProviderReplayClass::kPrepared);
      EXPECT_FALSE(step.runtime.blocks_host);
      if (step.kind == g::CompileStepKind::kProviderRegion) {
        EXPECT_EQ(step.node_count, 1u); store_region = true;
      }
    }
    ASSERT_TRUE(store_region);
    bool hooks = false;
    const auto inspected = gd::RegionAccess::visit(consumer,
        [](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
          *static_cast<bool *>(opaque) = view.prepared_vtable &&
              view.prepared_vtable->prepare_bindings && view.prepared_vtable->seal_bindings;
          return hyle::Status::Ok();
        }, &hooks);
    ASSERT_TRUE(inspected.ok()) << inspected.message();
    ASSERT_TRUE(hooks); // A genuine selected region, never a leaf/proxy fallback.
  }
  ASSERT_TRUE(consumer.bind(second_table, table_f32, embedding_ready).ok());
  auto row_writer = output_pool.allocate(2 * sizeof(float), 256).value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  if (region_consumer) {
    const auto layout = c::LeaseSlotLayout::create(output_pool, item, 1, 256).value();
    std::array<c::WriteLease, 1> destination{std::move(row_writer)};
    auto transaction = g::slots::SlotBindings::create(layout, {}, destination).value();
    ASSERT_TRUE(g::slots::prepare(consumer, queues, {.host_workspace = staging.ref()}).ok());
    ASSERT_TRUE(g::slots::bind(consumer, second_graph, store, std::move(transaction)).ok());
  } else {
    ASSERT_TRUE(consumer.bind(selected, row_f32, std::move(row_writer)).ok());
  }
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  Gate gate;
  Gate reader_gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto foreign = c::Completion::completed();
  EXPECT_FALSE(gd::SubmissionSource::create(producer, std::move(foreign), storage));
  EXPECT_TRUE(foreign.valid());
  auto submission_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(submission_result) << submission_result.error().message();
  auto submission_source = std::move(*submission_result);
  auto source_result = gd::OutputSource::create(submission_source, token, storage, source_pool);
  ASSERT_TRUE(source_result) << source_result.error().message();
  auto source = std::move(*source_result);
  const std::array bindings{gd::OutputBinding{second_index, &source}};
  EXPECT_FALSE(submitted->valid());
  ASSERT_TRUE(source.poll());
  EXPECT_EQ(*source.poll(), c::CompletionPoll::kPending);
  if (bad_producer_index) {
    gd::SubmissionSource empty;
    EXPECT_TRUE(empty.request_cancel().IsFailedPrecondition());
    // The original CUDA work cannot be preempted. Its capability result is
    // independent of the still-pending whole verdict and the later bad ID.
    const auto cancellation = submission_source.request_cancel();
    EXPECT_TRUE(cancellation.IsUnimplemented()) << cancellation.message();
    ASSERT_TRUE(source.poll());
    EXPECT_EQ(*source.poll(), c::CompletionPoll::kPending);
    EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  }
  EXPECT_TRUE(source.detach().IsUnavailable());
  EXPECT_FALSE(producer.reset().ok());
  EXPECT_FALSE(gd::bind_outputs(producer, std::array{gd::OutputBinding{first_index, &source}}).ok());
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  EXPECT_FALSE(gd::bind_outputs(consumer, bindings).ok());
  EXPECT_FALSE(consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject}).ok());
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCommands);
  if (region_consumer) {
    EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
    ASSERT_TRUE(source.poll());
    EXPECT_EQ(*source.poll(), c::CompletionPoll::kPending);
  }
  const auto source_report = consumer.input_source_report().value();
  EXPECT_EQ(source_report.source_count, 1u);
  EXPECT_EQ(source_report.device_wait_count, 1u);
  EXPECT_EQ(source_report.host_wait_count, 0u);
  // The ordinary public entry cannot silently treat pending bytes as ready.
  EXPECT_FALSE(consumer.submit(queues));
  EXPECT_FALSE(consumer.in_flight());
  if (reuse_producer) {
    // The producer's original completion precedes this gate; B's actual read
    // follows it. Thus A can retire while B still owns a pending native read.
    reader_gate.arm(gate.stream);
    ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &Gate::wait, &reader_gate), cudaSuccess);
  }
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  auto pending = consumed->poll();
  ASSERT_TRUE(pending) << pending.error().message();
  EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  EXPECT_TRUE(producer.in_flight()); EXPECT_TRUE(consumer.in_flight());
  EXPECT_GT(source_pool.stats().active_uses, 0u);
  gate.release();
  c::Completion next_submission;
  if (reuse_producer) {
    const auto first_status = source.wait();
    ASSERT_TRUE(first_status.ok()) << first_status.message();
    ASSERT_FALSE(producer.in_flight());
    ASSERT_TRUE(consumer.in_flight());
    EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
    const auto detached = source.detach();
    ASSERT_TRUE(detached.ok()) << detached.message();
    EXPECT_TRUE(source.detach().ok());
    const auto closed_submission = submission_source.detach();
    ASSERT_TRUE(closed_submission.ok()) << closed_submission.message();
    ASSERT_TRUE(producer.bind(first_index, i32, next_index).ok());
    ASSERT_TRUE(producer.bind(token, i32, std::move(next_token_writer)).ok());
    const auto prepared_next = producer.prepare_replay(queues,
        {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(prepared_next.ok()) << prepared_next.message();
    auto next = producer.submit(queues);
    ASSERT_TRUE(next) << next.error().message();
    next_submission = std::move(*next);
    EXPECT_TRUE(producer.in_flight()); EXPECT_TRUE(consumer.in_flight());
    EXPECT_EQ(*next_submission.poll(), c::CompletionPoll::kPending);
    // The old source's terminal identity no longer names A's observation slot.
    EXPECT_EQ(*source.poll(), c::CompletionPoll::kComplete);
    EXPECT_TRUE(source.wait().ok());
    reader_gate.release();
  }
  const auto consumer_status = consumed->wait();
  const auto producer_status = source.wait();
  if (reuse_producer) {
    const auto next_status = next_submission.wait();
    ASSERT_TRUE(next_status.ok()) << next_status.message();
    std::array<std::int32_t, 1> actual_next{};
    ASSERT_TRUE(producer.output(token)->storage().read(
        std::as_writable_bytes(std::span(actual_next))).ok());
    EXPECT_EQ(actual_next, (std::array<std::int32_t, 1>{1}));
    EXPECT_TRUE(source.detach().ok());
  }
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  EXPECT_FALSE(reader_gate.timed_out.load(std::memory_order_acquire));
  EXPECT_FALSE(producer.in_flight()); EXPECT_FALSE(consumer.in_flight());
  EXPECT_EQ(source_pool.stats().active_uses, 0u);
  EXPECT_EQ(output_pool.stats().active_uses, 0u);
  if (bad_producer_index) {
    EXPECT_FALSE(producer_status.ok()); EXPECT_FALSE(consumer_status.ok());
    EXPECT_EQ(consumer_status.code(), producer_status.code());
    EXPECT_EQ(consumer_status.message(), producer_status.message());
    EXPECT_FALSE(consumer.output(selected));
    const auto detached = source.detach();
    EXPECT_FALSE(detached.ok());
    EXPECT_EQ(detached.code(), producer_status.code());
    EXPECT_EQ(detached.message(), producer_status.message());
  } else {
    ASSERT_TRUE(producer_status.ok()) << producer_status.message();
    ASSERT_TRUE(consumer_status.ok()) << consumer_status.message();
    std::array<float, 2> actual{};
    if (region_consumer) {
      auto returned = g::slots::take(consumer, store);
      ASSERT_TRUE(returned) << returned.error().message();
      ASSERT_EQ(returned->destination_count(), 1u);
      auto ready = returned->destinations()[0].publish();
      ASSERT_TRUE(ready) << ready.error().message();
      ASSERT_TRUE(ready->view()->read(std::as_writable_bytes(std::span(actual))).ok());
    } else {
      ASSERT_TRUE(consumer.output(selected)->storage().read(
          std::as_writable_bytes(std::span(actual))).ok());
    }
    EXPECT_EQ(actual, (std::array<float, 2>{30, 31}));
    // Successful observation does not drop the immutable producer pin.
    if (!reuse_producer) EXPECT_FALSE(producer.reset().ok());
  }
  *consumed = {};
  source = {};
  consumer = {};
  if (bad_producer_index) {
    EXPECT_FALSE(gd::RegionAccess::outcome_available(producer));
    const auto closed = submission_source.detach();
    ASSERT_TRUE(closed.ok()) << closed.message();
    EXPECT_TRUE(gd::RegionAccess::outcome_available(producer));
    const auto original_failure = submission_source.wait();
    EXPECT_EQ(original_failure.code(), producer_status.code());
    EXPECT_EQ(original_failure.message(), producer_status.message());
    EXPECT_TRUE(submission_source.request_cancel().ok()); // Original settled proxy.
    const auto after_cancellation = submission_source.wait();
    EXPECT_EQ(after_cancellation.code(), producer_status.code());
    EXPECT_EQ(after_cancellation.message(), producer_status.message());
  }
  submission_source = {}; producer = {};
  EXPECT_EQ(source_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}

TEST(GraphOutputSourceCuda, PendingI32UsesOriginalQueueAndLeasesForCheckedGather) { relay(false); }
TEST(GraphOutputSourceCuda, PendingI32PreparesRealStoreRegionBindingHooks) { relay(false, false, true); }
TEST(GraphOutputSourceCuda, ProducerCheckedFailureRejectsConsumerPublicationAfterRetirement) { relay(true); }
TEST(GraphOutputSourceCuda, RetiredSourceDetachesWhileReaderPendingAndProducerReusesNewWriter) { relay(false, true); }
TEST(GraphOutputSourceCuda, PendingI32ReusesRealNativeRegionCaptureWithoutRecording) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(8).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  const auto table_i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{4}).value();
  const auto table_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4, 2}).value();
  const auto row_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 2}).value();
  auto inputs = c::LeasePool::create(storage.ref(), 4 * 256, {.max_blocks = 4}).value();
  auto token_pool = c::LeasePool::create(storage.ref(), 256, {.max_blocks = 1}).value();
  auto output_pool = c::LeasePool::create(storage.ref(), 2 * 256, {.max_blocks = 2}).value();
  auto table = inputs.allocate(4 * sizeof(std::int32_t), 256).value();
  auto index = inputs.allocate(sizeof(std::int32_t), 256).value();
  auto embedding = inputs.allocate(8 * sizeof(float), 256).value();
  auto other_embedding = inputs.allocate(8 * sizeof(float), 256).value();
  auto token_writer = token_pool.allocate(sizeof(std::int32_t), 256).value();
  constexpr std::array<std::int32_t, 4> tokens{2, 1, 3, 0};
  constexpr std::array<std::int32_t, 1> select{0}, warm_token{1};
  constexpr std::array<float, 8> rows{10, 11, 20, 21, 30, 31, 40, 41};
  ASSERT_TRUE(table.view()->write(std::as_bytes(std::span(tokens))).ok());
  ASSERT_TRUE(index.view()->write(std::as_bytes(std::span(select))).ok());
  ASSERT_TRUE(embedding.view()->write(std::as_bytes(std::span(rows))).ok());
  ASSERT_TRUE(other_embedding.view()->write(std::as_bytes(std::span(rows))).ok());
  ASSERT_TRUE(token_writer.view()->write(std::as_bytes(std::span(warm_token))).ok());
  auto table_ready = table.publish().value(), index_ready = index.publish().value();
  auto embedding_ready = embedding.publish().value();
  auto other_embedding_ready = other_embedding.publish().value();
  auto token_ready = token_writer.publish().value();
  const auto token_generation = token_ready.generation();
  // A descriptor alone retains no extra read permission. The original ready
  // alias is explicitly withdrawn before handing its one block to A.
  const auto token_view = token_ready.tensor(i32).value();
  const auto token_storage = token_view.storage();

  auto first = g::GraphBuilder::create().value();
  const auto first_table = first.input(table_i32).value();
  const auto first_index = first.input(i32).value();
  const auto token = first.add(c::Gather{.axis = 0}, first_table, first_index).value();
  ASSERT_TRUE(first.publish(token).ok());
  const auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage);
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(first_table, table_i32, table_ready).ok());
  ASSERT_TRUE(producer.bind(first_index, i32, index_ready).ok());

  auto second = g::GraphBuilder::create().value();
  const auto second_table = second.input(table_f32).value();
  const auto second_index = second.input(i32).value();
  const auto gathered = second.add(c::Gather{.axis = 0}, second_table, second_index).value();
  const auto selected = second.add(c::Binary{.kind = c::BinaryKind::kAdd}, gathered, gathered).value();
  ASSERT_TRUE(second.publish(selected).ok());
  const auto second_graph = std::move(second).finish().value();
  auto consumer_plan = g::cuda::compile(second_graph, context, storage,
      {.operation_fallback = g::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  ASSERT_EQ(consumer_plan->report.region_count(), 1u);
  ASSERT_EQ(consumer_plan->report.step_count(), 1u);
  const auto compiled_step = consumer_plan->report.step(0).value();
  EXPECT_EQ(compiled_step.kind, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled_step.node_count, 2u);
  EXPECT_EQ(compiled_step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(compiled_step.runtime.replay_class, c::ProviderReplayClass::kManaged);
  EXPECT_TRUE(compiled_step.runtime.blocks_host); // Original properties are unchanged.
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  bool hooks = false;
  ASSERT_TRUE(gd::RegionAccess::visit(consumer,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
        *static_cast<bool *>(opaque) = view.prepared_vtable &&
            view.prepared_vtable->prepare_bindings && view.prepared_vtable->seal_bindings;
        return hyle::Status::Ok();
      }, &hooks).ok());
  ASSERT_TRUE(hooks);
  ASSERT_TRUE(consumer.bind(second_table, table_f32, embedding_ready).ok());
  ASSERT_TRUE(consumer.bind(second_index, i32, token_ready).ok());
  auto row_writer = output_pool.allocate(2 * sizeof(float), 256).value();
  ASSERT_TRUE(consumer.bind(selected, row_f32, std::move(row_writer)).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  // The same real Managed artifact is legal ordinary Commands work, but a
  // second Invocation has never recorded the capture needed for pending input.
  auto commands_consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(commands_consumer.prepare_resources().ok());
  ASSERT_TRUE(commands_consumer.bind(second_table, table_f32, embedding_ready).ok());
  ASSERT_TRUE(commands_consumer.bind(second_index, i32, token_ready).ok());
  auto commands_writer = output_pool.allocate(2 * sizeof(float), 256).value();
  ASSERT_TRUE(commands_consumer.bind(selected, row_f32, std::move(commands_writer)).ok());
  const auto commands = commands_consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow});
  ASSERT_TRUE(commands.ok()) << commands.message();
  ASSERT_EQ(commands_consumer.replay_route(), g::ReplayRoute::kCommands);
  EXPECT_EQ(commands_consumer.replay_report()->recorded_capture_range_count, 0u);
  ASSERT_TRUE(commands_consumer.unbind(second_index).ok());
  const auto warm = consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
  ASSERT_TRUE(warm.ok()) << warm.message();
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
  ASSERT_EQ(consumer.replay_report()->recorded_capture_range_count, 1u);
  ASSERT_EQ(consumer.replay_report()->reused_capture_range_count, 0u);

  // A different real embedding block invalidates the complete physical tuple.
  // Reuse-only preparation must reject rather than record it or choose Commands.
  ASSERT_TRUE(consumer.bind(second_table, table_f32, other_embedding_ready).ok());
  auto miss = gd::ReplayAccess::can_reuse_capture(consumer,
      std::array{gd::ReplayInputBinding{second_index, token_view}});
  ASSERT_TRUE(miss) << miss.error().message();
  EXPECT_FALSE(*miss);
  EXPECT_FALSE(gd::ReplayAccess::prepare_reuse(consumer, queues).ok());
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kUnprepared);
  EXPECT_FALSE(consumer.replay_report());
  ASSERT_TRUE(consumer.bind(second_table, table_f32, embedding_ready).ok());
  ASSERT_TRUE(consumer.unbind(second_index).ok());
  auto reclaimed = token_ready.try_write();
  ASSERT_TRUE(reclaimed) << reclaimed.error().message();
  EXPECT_FALSE(token_ready.valid());
  EXPECT_EQ(reclaimed->generation(), token_generation);
  const auto reclaimed_storage = reclaimed->view().value();
  EXPECT_EQ(reclaimed_storage.resource(), token_storage.resource());
  EXPECT_EQ(reclaimed_storage.native_handle(), token_storage.native_handle());
  EXPECT_EQ(reclaimed_storage.offset(), token_storage.offset());
  EXPECT_EQ(reclaimed_storage.size(), token_storage.size());
  ASSERT_TRUE(producer.bind(token, i32, std::move(*reclaimed)).ok());
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  Gate gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto submission_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(submission_result) << submission_result.error().message();
  auto submission_source = std::move(*submission_result);
  auto source_result = gd::OutputSource::create(submission_source, token, storage, token_pool);
  ASSERT_TRUE(source_result) << source_result.error().message();
  auto source = std::move(*source_result);
  const std::array bindings{gd::OutputBinding{second_index, &source}};
  ASSERT_TRUE(source.poll());
  ASSERT_EQ(*source.poll(), c::CompletionPoll::kPending);
  {
    // A typed view belongs to one reader bundle. Use another genuine view of
    // this submission so rejection cannot consume the successful reader's view.
    auto refused_view = gd::OutputSource::create(submission_source, token, storage, token_pool);
    if (!refused_view) gate.release();
    ASSERT_TRUE(refused_view) << refused_view.error().message();
    const std::array refused_bindings{gd::OutputBinding{second_index, &*refused_view}};
    const auto attached = gd::bind_outputs(commands_consumer, refused_bindings);
    if (!attached.ok()) gate.release();
    ASSERT_TRUE(attached.ok()) << attached.message();
    const auto refused = gd::prepare_outputs(commands_consumer, refused_bindings, queues);
    EXPECT_TRUE(refused.IsUnavailable()) << refused.message();
    auto refused_submission = gd::submit_outputs(commands_consumer, refused_bindings, queues);
    EXPECT_FALSE(refused_submission);
    if (refused_submission) {
      // A regression must still retire an accidentally accepted callback before
      // either its original resources or the borrowed Gate can be destroyed.
      gate.release();
      EXPECT_TRUE(refused_submission->wait().ok());
    } else {
      EXPECT_TRUE(refused_submission.error().IsUnavailable()) << refused_submission.error().message();
    }
    EXPECT_FALSE(commands_consumer.in_flight());
    EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
    const auto still_pending = source.poll();
    if (!still_pending) gate.release();
    ASSERT_TRUE(still_pending) << still_pending.error().message();
    EXPECT_EQ(*still_pending, c::CompletionPoll::kPending);
  }
  ASSERT_TRUE(consumer.bind(second_table, table_f32, other_embedding_ready).ok());
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  EXPECT_FALSE(gd::prepare_outputs(consumer, bindings, queues).ok());
  EXPECT_FALSE(gd::submit_outputs(consumer, bindings, queues));
  EXPECT_TRUE(source.valid());
  EXPECT_FALSE(consumer.in_flight());
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kUnprepared);
  EXPECT_FALSE(consumer.replay_report());
  ASSERT_TRUE(source.poll());
  EXPECT_EQ(*source.poll(), c::CompletionPoll::kPending);
  ASSERT_TRUE(consumer.bind(second_table, table_f32, embedding_ready).ok());
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  ASSERT_TRUE(source.poll());
  EXPECT_EQ(*source.poll(), c::CompletionPoll::kPending);
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, 1u);
  const auto dependencies = consumer.input_source_report().value();
  EXPECT_EQ(dependencies.source_count, 1u);
  EXPECT_EQ(dependencies.device_wait_count, 1u);
  EXPECT_EQ(dependencies.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues)); // The pending input still needs ordered admission.
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  ASSERT_TRUE(consumed->poll());
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  gate.release();
  const auto finished = consumed->wait();
  ASSERT_TRUE(finished.ok()) << finished.message();
  ASSERT_TRUE(source.wait().ok());
  commands_consumer = {};
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  std::array<float, 2> actual{};
  ASSERT_TRUE(consumer.output(selected)->storage().read(
      std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<float, 2>{60, 62}));
  EXPECT_FALSE(producer.in_flight()); EXPECT_FALSE(consumer.in_flight());
  EXPECT_EQ(token_pool.stats().active_uses, 0u);
  EXPECT_EQ(output_pool.stats().active_uses, 0u);
  *consumed = {}; source = {}; consumer = {}; submission_source = {}; producer = {};
  EXPECT_EQ(token_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}
TEST(GraphOutputSourceCuda, CapturedManagedI32ProducerRetainsItsOriginalSubmission) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(8).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  const auto table_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4, 2}).value();
  const auto row_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 2}).value();
  auto inputs = c::LeasePool::create(storage.ref(), 2 * 256, {.max_blocks = 2}).value();
  auto token_pool = c::LeasePool::create(storage.ref(), 256, {.max_blocks = 1}).value();
  auto output_pool = c::LeasePool::create(storage.ref(), 256, {.max_blocks = 1}).value();
  auto index = inputs.allocate(sizeof(std::int32_t), 256).value();
  auto embedding = inputs.allocate(8 * sizeof(float), 256).value();
  constexpr std::array<std::int32_t, 1> wanted{2};
  constexpr std::array<float, 8> rows{10, 11, 20, 21, 30, 31, 40, 41};
  ASSERT_TRUE(index.view()->write(std::as_bytes(std::span(wanted))).ok());
  ASSERT_TRUE(embedding.view()->write(std::as_bytes(std::span(rows))).ok());
  auto index_ready = index.publish().value(), embedding_ready = embedding.publish().value();

  auto first = g::GraphBuilder::create().value();
  const auto first_index = first.input(i32).value();
  // A nonempty one-input Concat is an owned integer copy. Two such ordinary
  // operations form a real ggml region without introducing a sampling kernel.
  const auto copy0 = first.add_node(c::Operation(c::Concat{.axis = 0}),
      std::array{first_index}).value();
  const auto intermediate = first.result(copy0).value();
  const auto copy1 = first.add_node(c::Operation(c::Concat{.axis = 0}),
      std::array{intermediate}).value();
  const auto token = first.result(copy1).value();
  ASSERT_TRUE(first.publish(token).ok());
  const auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage,
      {.operation_fallback = g::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  ASSERT_EQ(producer_plan->report.region_count(), 1u);
  ASSERT_EQ(producer_plan->report.step_count(), 1u);
  ASSERT_EQ(producer_plan->report.segment_count(), 1u);
  const auto producer_step = producer_plan->report.step(0).value();
  EXPECT_EQ(producer_step.kind, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(producer_step.node_count, 2u);
  EXPECT_EQ(producer_step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(producer_step.runtime.replay_class, c::ProviderReplayClass::kManaged);
  EXPECT_TRUE(producer_step.runtime.blocks_host);
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(first_index, i32, index_ready).ok());
  auto token_writer = token_pool.allocate(sizeof(std::int32_t), 256).value();
  ASSERT_TRUE(producer.bind(token, i32, std::move(token_writer)).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto captured = producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
  ASSERT_TRUE(captured.ok()) << captured.message();
  ASSERT_EQ(producer.replay_route(), g::ReplayRoute::kCaptured);
  const auto replay = producer.replay_report().value();
  EXPECT_EQ(replay.route, g::ReplayRoute::kCaptured);
  EXPECT_EQ(replay.recorded_capture_range_count, 1u);
  EXPECT_EQ(replay.reused_capture_range_count, 0u);
  EXPECT_EQ(replay.immediate_transfer_count, 0u);
  EXPECT_EQ(replay.host_wait_dependency_count, 0u);

  auto second = g::GraphBuilder::create().value();
  const auto second_table = second.input(table_f32).value();
  const auto second_index = second.input(i32).value();
  const auto selected = second.add(c::Gather{.axis = 0}, second_table, second_index).value();
  ASSERT_TRUE(second.publish(selected).ok());
  const auto second_graph = std::move(second).finish().value();
  auto consumer_plan = g::cuda::compile(second_graph, context, storage);
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  ASSERT_EQ(consumer_plan->report.step_count(), 1u);
  const auto consumer_step = consumer_plan->report.step(0).value();
  EXPECT_EQ(consumer_step.runtime.replay_class, c::ProviderReplayClass::kPrepared);
  EXPECT_FALSE(consumer_step.runtime.blocks_host);
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  ASSERT_TRUE(consumer.bind(second_table, table_f32, embedding_ready).ok());
  auto row_writer = output_pool.allocate(2 * sizeof(float), 256).value();
  ASSERT_TRUE(consumer.bind(selected, row_f32, std::move(row_writer)).ok());

  Gate gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  auto source_result = gd::OutputSource::create(common, token, storage, token_pool);
  ASSERT_TRUE(source_result) << source_result.error().message();
  auto source = std::move(*source_result);
  ASSERT_TRUE(common.poll());
  EXPECT_EQ(*common.poll(), c::CompletionPoll::kPending);
  EXPECT_FALSE(producer.reset().ok());
  const std::array bindings{gd::OutputBinding{second_index, &source}};
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCommands);
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  const auto dependency = consumer.input_source_report().value();
  EXPECT_EQ(dependency.source_count, 1u);
  EXPECT_EQ(dependency.device_wait_count, 1u);
  EXPECT_EQ(dependency.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  ASSERT_TRUE(consumed->poll());
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  gate.release();
  const auto terminal = consumed->wait();
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  ASSERT_TRUE(common.wait().ok());
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  std::array<std::int32_t, 1> actual_token{};
  ASSERT_TRUE(producer.output(token)->storage().read(
      std::as_writable_bytes(std::span(actual_token))).ok());
  EXPECT_EQ(actual_token, wanted);
  std::array<float, 2> actual{};
  ASSERT_TRUE(consumer.output(selected)->storage().read(
      std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<float, 2>{30, 31}));
  EXPECT_FALSE(producer.in_flight()); EXPECT_FALSE(consumer.in_flight());
  EXPECT_EQ(token_pool.stats().active_uses, 0u);
  EXPECT_EQ(output_pool.stats().active_uses, 0u);
  *consumed = {}; source = {}; consumer = {}; common = {}; producer = {};
  EXPECT_EQ(token_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}
TEST(GraphOutputSourceCuda, TwoOutputsShareOneOriginalSubmissionAndAtomicBundle) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(8).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  const auto table_i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{4}).value();
  const auto table_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4, 2}).value();
  const auto row_f32 = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 2}).value();
  auto inputs = c::LeasePool::create(storage.ref(), 6 * 256, {.max_blocks = 6}).value();
  auto tokens_pool = c::LeasePool::create(storage.ref(), 2 * 256, {.max_blocks = 2}).value();
  auto output_pool = c::LeasePool::create(storage.ref(), 2 * 256, {.max_blocks = 2}).value();
  auto table = inputs.allocate(4 * sizeof(std::int32_t), 256).value();
  auto embedding = inputs.allocate(8 * sizeof(float), 256).value();
  constexpr std::array<std::int32_t, 4> token_values{2, 1, 3, 0};
  constexpr std::array<float, 8> rows{10, 11, 20, 21, 30, 31, 40, 41};
  ASSERT_TRUE(table.view()->write(std::as_bytes(std::span(token_values))).ok());
  ASSERT_TRUE(embedding.view()->write(std::as_bytes(std::span(rows))).ok());
  auto table_ready = table.publish().value(), embedding_ready = embedding.publish().value();
  std::array<c::ReadLease, 2> selection, previous_indices;
  for (std::size_t i = 0; i < 2; ++i) {
    auto index = inputs.allocate(sizeof(std::int32_t), 256).value();
    const std::array<std::int32_t, 1> wanted{static_cast<std::int32_t>(i)};
    ASSERT_TRUE(index.view()->write(std::as_bytes(std::span(wanted))).ok());
    selection[i] = index.publish().value();
    auto previous = inputs.allocate(sizeof(std::int32_t), 256).value();
    const std::array<std::int32_t, 1> old{static_cast<std::int32_t>(i == 0 ? 3 : 0)};
    ASSERT_TRUE(previous.view()->write(std::as_bytes(std::span(old))).ok());
    previous_indices[i] = previous.publish().value();
  }

  auto first = g::GraphBuilder::create().value();
  const auto first_table = first.input(table_i32).value();
  std::array<g::ValueId, 2> first_indices, tokens;
  for (std::size_t i = 0; i < 2; ++i) {
    first_indices[i] = first.input(i32).value();
    tokens[i] = first.add(c::Gather{.axis = 0}, first_table, first_indices[i]).value();
    ASSERT_TRUE(first.publish(tokens[i]).ok());
  }
  const auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage);
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  ASSERT_EQ(producer_plan->report.step_count(), 2u);
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(first_table, table_i32, table_ready).ok());
  std::array<c::ConstTensorView, 2> original_tokens;
  for (std::size_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(producer.bind(first_indices[i], i32, selection[i]).ok());
    auto writer = tokens_pool.allocate(sizeof(std::int32_t), 256).value();
    original_tokens[i] = writer.tensor(i32)->as_const();
    ASSERT_TRUE(producer.bind(tokens[i], i32, std::move(writer)).ok());
    const auto step = producer_plan->report.step(i).value();
    EXPECT_EQ(step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
    EXPECT_EQ(step.runtime.replay_class, c::ProviderReplayClass::kPrepared);
    EXPECT_FALSE(step.runtime.blocks_host);
  }
  EXPECT_EQ(original_tokens[0].storage().resource(), original_tokens[1].storage().resource());
  EXPECT_EQ(original_tokens[0].storage().native_handle(), original_tokens[1].storage().native_handle());
  EXPECT_NE(original_tokens[0].storage().offset(), original_tokens[1].storage().offset());

  auto second = g::GraphBuilder::create().value();
  const auto second_table = second.input(table_f32).value();
  std::array<g::ValueId, 2> second_indices, selected;
  for (std::size_t i = 0; i < 2; ++i) {
    second_indices[i] = second.input(i32).value();
    selected[i] = second.add(c::Gather{.axis = 0}, second_table, second_indices[i]).value();
    ASSERT_TRUE(second.publish(selected[i]).ok());
  }
  const auto second_graph = std::move(second).finish().value();
  auto consumer_plan = g::cuda::compile(second_graph, context, storage);
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  ASSERT_EQ(consumer_plan->report.step_count(), 2u);
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  ASSERT_TRUE(consumer.bind(second_table, table_f32, embedding_ready).ok());
  for (std::size_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(consumer.bind(second_indices[i], i32, previous_indices[i]).ok());
    auto writer = output_pool.allocate(2 * sizeof(float), 256).value();
    ASSERT_TRUE(consumer.bind(selected[i], row_f32, std::move(writer)).ok());
    const auto step = consumer_plan->report.step(i).value();
    EXPECT_EQ(step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
    EXPECT_EQ(step.runtime.replay_class, c::ProviderReplayClass::kPrepared);
    EXPECT_FALSE(step.runtime.blocks_host);
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());
  Gate gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  EXPECT_FALSE(submitted->valid());
  auto left_result = gd::OutputSource::create(common, tokens[0], storage, tokens_pool);
  auto right_result = gd::OutputSource::create(common, tokens[1], storage, tokens_pool);
  ASSERT_TRUE(left_result) << left_result.error().message();
  ASSERT_TRUE(right_result) << right_result.error().message();
  std::array<gd::OutputSource, 2> views{std::move(*left_result), std::move(*right_result)};
  ASSERT_TRUE(common.poll());
  ASSERT_EQ(*common.poll(), c::CompletionPoll::kPending);
  // The second input is foreign. A late rejection must neither replace the
  // first original reader nor claim either typed view's one-shot input name.
  const std::array rejected{gd::OutputBinding{second_indices[0], &views[0]},
                            gd::OutputBinding{first_indices[1], &views[1]}};
  EXPECT_FALSE(gd::bind_outputs(consumer, rejected).ok());
  auto preserved = previous_indices[0].try_write();
  ASSERT_FALSE(preserved) << "rejected bundle partially replaced its first input";
  EXPECT_TRUE(previous_indices[0].valid());
  EXPECT_TRUE(views[0].valid()); EXPECT_TRUE(views[1].valid());
  EXPECT_FALSE(consumer.in_flight());
  const std::array bindings{gd::OutputBinding{second_indices[0], &views[0]},
                            gd::OutputBinding{second_indices[1], &views[1]}};
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  EXPECT_FALSE(gd::bind_outputs(consumer, bindings).ok());
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCommands);
  const auto report = consumer.input_source_report().value();
  EXPECT_EQ(report.source_count, 1u);
  EXPECT_EQ(report.device_wait_count, 1u);
  EXPECT_EQ(report.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  ASSERT_TRUE(consumed->poll());
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  gate.release();
  const auto terminal = consumed->wait();
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  ASSERT_TRUE(common.wait().ok());
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  constexpr std::array<std::array<float, 2>, 2> expected{{{30, 31}, {20, 21}}};
  for (std::size_t i = 0; i < 2; ++i) {
    std::array<float, 2> actual{};
    ASSERT_TRUE(consumer.output(selected[i])->storage().read(
        std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, expected[i]);
  }
  EXPECT_EQ(tokens_pool.stats().active_uses, 0u);
  EXPECT_EQ(output_pool.stats().active_uses, 0u);
  *consumed = {}; views = {}; consumer = {}; common = {}; producer = {};
  EXPECT_EQ(tokens_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}
void store_relay(bool failed_producer, bool selected_range = false) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(12).ok());
  auto device_memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(device_memory)).value();
  auto staging = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  const auto table_i32 = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{4}).value();
  const auto item = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}).value();
  const auto rows_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  const auto embedding_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{4, 2}).value();
  const auto row_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 2}).value();
  const auto read_ids_desc = selected_range ? i32 : ids_desc;
  const auto read_rows_desc = selected_range ? row_desc : rows_desc;
  constexpr std::size_t stride = 256, item_bytes = 2 * sizeof(float);
  auto backing = hyle::mem::Buffer::create(storage.ref(), 4 * stride, stride).value();
  ASSERT_TRUE(backing.view().fill(std::byte{0x5a}).ok());
  auto state_pool = c::LeasePool::adopt(std::move(backing), {.max_blocks = 8}).value();
  const auto layout = c::LeaseSlotLayout::create(state_pool, item, 4, stride).value();
  std::array<c::WriteLease, 4> initial;
  std::array<c::ReadLease, 4> padding;
  for (std::size_t i = 0; i < initial.size(); ++i) {
    initial[i] = state_pool.allocate(item_bytes, stride).value();
    EXPECT_EQ(layout.locate(initial[i])->index, i);
    const std::array<float, 2> old{100.0f + static_cast<float>(i), 200.0f + static_cast<float>(i)};
    ASSERT_TRUE(initial[i].view()->write(std::as_bytes(std::span(old))).ok());
    auto gap = state_pool.allocate(stride - item_bytes, 1).value();
    padding[i] = gap.publish().value();
  }
  std::array<c::ReadLease, 2> sentinels{
      initial[1].publish().value(), initial[3].publish().value()};
  std::array<c::WriteLease, 2> destinations{std::move(initial[0]), std::move(initial[2])};

  auto input_pool = c::LeasePool::create(storage.ref(), 6 * stride, {.max_blocks = 6}).value();
  auto table = input_pool.allocate(4 * sizeof(std::int32_t), stride).value();
  auto index = input_pool.allocate(sizeof(std::int32_t), stride).value();
  auto rows = input_pool.allocate(4 * sizeof(float), stride).value();
  auto embedding = input_pool.allocate(8 * sizeof(float), stride).value();
  auto old_ids = input_pool.allocate(2 * sizeof(std::int32_t), stride).value();
  auto decoy_ids = input_pool.allocate(2 * sizeof(std::int32_t), stride).value();
  constexpr std::array<std::int32_t, 4> tokens{2, 1, 3, 0};
  const std::array<std::int32_t, 1> selected_index{failed_producer ? 4 : 0};
  constexpr std::array<float, 4> incoming{3, 5, 7, 11};
  constexpr std::array<float, 8> embedding_values{10, 11, 20, 21, 30, 31, 40, 41};
  constexpr std::array<std::int32_t, 2> old_source_ids{1, 3}, decoy_values{0, 1};
  ASSERT_TRUE(table.view()->write(std::as_bytes(std::span(tokens))).ok());
  ASSERT_TRUE(index.view()->write(std::as_bytes(std::span(selected_index))).ok());
  ASSERT_TRUE(rows.view()->write(std::as_bytes(std::span(incoming))).ok());
  ASSERT_TRUE(embedding.view()->write(std::as_bytes(std::span(embedding_values))).ok());
  ASSERT_TRUE(old_ids.view()->write(std::as_bytes(std::span(old_source_ids))).ok());
  ASSERT_TRUE(decoy_ids.view()->write(std::as_bytes(std::span(decoy_values))).ok());
  auto table_ready = table.publish().value(), index_ready = index.publish().value();
  auto rows_ready = rows.publish().value(), embedding_ready = embedding.publish().value();
  auto old_ids_ready = old_ids.publish().value(), decoy_ids_ready = decoy_ids.publish().value();
  auto token_pool = c::LeasePool::create(storage.ref(), stride, {.max_blocks = 1}).value();
  auto output_pool = c::LeasePool::create(storage.ref(), 3 * stride, {.max_blocks = 3}).value();
  auto foreign_pool = c::LeasePool::create(storage.ref(), stride, {.max_blocks = 1}).value();
  auto foreign_writer = foreign_pool.allocate(item_bytes, stride).value();

  auto first = g::GraphBuilder::create().value();
  const auto table_input = first.input(table_i32).value();
  const auto index_input = first.input(i32).value();
  const auto row_input = first.input(rows_desc).value();
  const auto destination_ids = first.input(ids_desc).value();
  const auto token = first.add(c::Gather{.axis = 0}, table_input, index_input).value();
  ASSERT_TRUE(first.publish(token).ok());
  const auto store = first.add_node(c::Operation::custom(g::slots::Store{
      g::slots::Schema::create(item, 4, stride).value()}),
      std::array{row_input, destination_ids}).value();
  const auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage);
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  ASSERT_EQ(producer_plan->report.segment_count(), 1u);
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(table_input, table_i32, table_ready).ok());
  ASSERT_TRUE(producer.bind(index_input, i32, index_ready).ok());
  ASSERT_TRUE(producer.bind(row_input, rows_desc, rows_ready).ok());
  auto token_writer = token_pool.allocate(sizeof(std::int32_t), stride).value();
  c::ReadLease warm_token;
  if (selected_range) {
    constexpr std::array<std::int32_t, 1> warm_value{2};
    ASSERT_TRUE(token_writer.view()->write(std::as_bytes(std::span(warm_value))).ok());
    warm_token = token_writer.publish().value();
  } else {
    ASSERT_TRUE(producer.bind(token, i32, std::move(token_writer)).ok());
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(g::slots::prepare(producer, queues, {.host_workspace = staging.ref()}).ok());
  auto transaction = g::slots::SlotBindings::create(layout, {}, destinations).value();
  ASSERT_TRUE(g::slots::bind(producer, first_graph, store, std::move(transaction)).ok());
  auto second = g::GraphBuilder::create().value();
  const auto embedding_input = second.input(embedding_desc).value();
  const auto token_input = second.input(i32).value();
  const auto bank_input = second.input(
      g::slots::Schema::create(item, 4, stride)->capacity_description().value()).value();
  const auto source_ids = second.input(read_ids_desc).value();
  const auto unrelated_ids = second.input(ids_desc).value();
  const auto token_row = second.add(c::Gather{.axis = 0}, embedding_input, token_input).value();
  const auto gather = second.add_node(c::Operation(c::Gather{.axis = 0}),
      std::array{bank_input, source_ids}).value();
  const auto state_rows = second.result(gather).value();
  const auto decoy_rows = second.add(c::Gather{.axis = 0}, embedding_input, unrelated_ids).value();
  ASSERT_TRUE(second.publish(token_row).ok()); ASSERT_TRUE(second.publish(state_rows).ok());
  ASSERT_TRUE(second.publish(decoy_rows).ok());
  const auto second_graph = std::move(second).finish().value();
  auto consumer_plan = g::cuda::compile(second_graph, context, storage);
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  ASSERT_TRUE(consumer.bind(embedding_input, embedding_desc, embedding_ready).ok());
  auto previous_ids = selected_range ? old_ids_ready.subview(0, sizeof(std::int32_t)).value()
                                     : old_ids_ready;
  ASSERT_TRUE(consumer.bind(source_ids, read_ids_desc, previous_ids).ok());
  ASSERT_TRUE(consumer.bind(unrelated_ids, ids_desc, decoy_ids_ready).ok());
  auto token_row_writer = output_pool.allocate(item_bytes, stride).value();
  auto state_rows_writer = output_pool.allocate(selected_range ? item_bytes : 2 * item_bytes, stride).value();
  auto decoy_rows_writer = output_pool.allocate(2 * item_bytes, stride).value();
  ASSERT_TRUE(consumer.bind(token_row, row_desc, std::move(token_row_writer)).ok());
  ASSERT_TRUE(consumer.bind(state_rows, read_rows_desc, std::move(state_rows_writer)).ok());
  ASSERT_TRUE(consumer.bind(decoy_rows, rows_desc, std::move(decoy_rows_writer)).ok());
  g::Invocation guard_probe;
  if (selected_range) {
    guard_probe = consumer_plan->executable.create_invocation().value();
    ASSERT_TRUE(guard_probe.prepare_resources().ok());
  }
  std::size_t recorded_ranges = 0;
  if (selected_range) {
    // Prime the exact future physical tuple with genuinely initialized ready
    // state. The cold original Store has not run; both old payloads are real.
    struct ColdControl final { c::ConstTensorView capacity{}, indices{}; c::ReadLease ids; } control;
    ASSERT_TRUE(gd::RegionAccess::visit(producer, store,
        [](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
          auto &result = *static_cast<ColdControl *>(opaque);
          const auto *extension = static_cast<const g::slots::detail::StoreRegionVTable *>(
              view.prepared_vtable->query_extension(view.prepared, g::slots::detail::kStoreBindingExtensionId));
          if (!extension || !extension->binding) return hyle::Status::Unimplemented("Store export missing");
          HYLE_ASSIGN_OR_RETURN(auto exported, extension->binding(view.invocation)->export_pending());
          result.capacity = exported.capacity; result.indices = exported.indices;
          result.ids = std::move(exported.indices_guard);
          return hyle::Status::Ok();
        }, &control).ok());
    auto cold_returned = g::slots::take(producer, store);
    ASSERT_TRUE(cold_returned) << cold_returned.error().message();
    std::array<c::ReadLease, 2> warm_state;
    std::array<std::uint64_t, 2> generations{};
    std::array<hyle::mem::ConstBufferView, 2> old_storage;
    for (std::size_t i = 0; i < warm_state.size(); ++i) {
      warm_state[i] = cold_returned->destinations()[i].publish().value();
      generations[i] = warm_state[i].generation(); old_storage[i] = warm_state[i].view().value();
    }
    const auto token_generation = warm_token.generation();
    const auto old_token_storage = warm_token.view().value();
    auto selected_ids = control.ids.subview(sizeof(std::int32_t), sizeof(std::int32_t)).value();
    const auto selected_storage = selected_ids.view().value();
    EXPECT_EQ(selected_storage.native_handle(), control.indices.storage().native_handle());
    EXPECT_EQ(selected_storage.offset(), control.indices.storage().offset() + sizeof(std::int32_t));
    EXPECT_EQ(selected_storage.size(), sizeof(std::int32_t));
    ASSERT_TRUE(consumer.bind(token_input, i32, warm_token).ok());
    ASSERT_TRUE(consumer.bind(source_ids, read_ids_desc, selected_ids).ok());
    ASSERT_TRUE(gd::RegionAccess::reserve_reads(consumer, 1, {}).ok());
    const std::array access{gather};
    ASSERT_TRUE(gd::RegionAccess::bind_read_projection(consumer,
        {bank_input, {}, access}, control.capacity, std::span<const c::ReadLease>(warm_state).subspan(1, 1)).ok());
    const auto warm = consumer.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(warm.ok()) << warm.message();
    ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
    recorded_ranges = consumer.replay_report()->recorded_capture_range_count;
    ASSERT_GT(recorded_ranges, 0u);
    EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, 0u);
    ASSERT_TRUE(gd::RegionAccess::release_read_projection(consumer, bank_input).ok());
    ASSERT_TRUE(consumer.unbind(std::array{source_ids, token_input}).ok());
    selected_ids = {}; control.ids = {};
    for (std::size_t i = 0; i < warm_state.size(); ++i) {
      auto reclaimed = warm_state[i].try_write();
      ASSERT_TRUE(reclaimed) << reclaimed.error().message();
      EXPECT_EQ(reclaimed->generation(), generations[i]);
      const auto actual = reclaimed->view().value();
      EXPECT_EQ(actual.resource(), old_storage[i].resource());
      EXPECT_EQ(actual.native_handle(), old_storage[i].native_handle());
      EXPECT_EQ(actual.offset(), old_storage[i].offset()); EXPECT_EQ(actual.size(), old_storage[i].size());
      destinations[i] = std::move(*reclaimed);
    }
    auto reclaimed_token = warm_token.try_write();
    ASSERT_TRUE(reclaimed_token) << reclaimed_token.error().message();
    EXPECT_EQ(reclaimed_token->generation(), token_generation);
    EXPECT_EQ(reclaimed_token->view()->offset(), old_token_storage.offset());
    ASSERT_TRUE(producer.bind(token, i32, std::move(*reclaimed_token)).ok());
    auto actual = g::slots::SlotBindings::create(layout, {}, destinations).value();
    ASSERT_TRUE(g::slots::bind(producer, first_graph, store, std::move(actual)).ok());
    // Empty and out-of-bounds ranges must not affect the original transaction.
    for (const auto invalid : std::array{gd::StoreDestinationRange{0, 0},
             gd::StoreDestinationRange{2, 1}, gd::StoreDestinationRange{1, 2},
             gd::StoreDestinationRange{SIZE_MAX, 1}}) {
      auto rejected = gd::StoreSource::prepare(producer, store, storage, invalid, {});
      ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument());
    }
  }
  // Both overloads share one cold preparation path. Explicit {} retains its
  // original ResourceOptions meaning rather than becoming an ambiguous range.
  auto state_result = gd::StoreSource::prepare(producer, store, storage, {});
  ASSERT_TRUE(state_result) << state_result.error().message();
  auto state = std::move(*state_result);
  auto another_result = selected_range
      ? gd::StoreSource::prepare(producer, store, storage, gd::StoreDestinationRange{1, 1}, {})
      : gd::StoreSource::prepare(producer, store, storage);
  ASSERT_TRUE(another_result) << another_result.error().message();
  auto another_view = std::move(*another_result);
  gd::StoreSource proof_view;
  if (selected_range) {
    auto proof = gd::StoreSource::prepare(producer, store, storage, gd::StoreDestinationRange{1, 1}, {});
    ASSERT_TRUE(proof) << proof.error().message(); proof_view = std::move(*proof);
  }
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  Gate gate, reader_gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  const auto captured = state.capture(common);
  ASSERT_TRUE(captured.ok()) << captured.message();
  // A second independently prepared view must be able to export the same
  // immutable generation after its first ticket sealed further mutation.
  const auto captured_again = another_view.capture(common);
  ASSERT_TRUE(captured_again.ok()) << captured_again.message();
  EXPECT_EQ(state.destination_count(), 2u);
  EXPECT_EQ(another_view.destination_count(), selected_range ? 1u : 2u);
  if (selected_range) {
    const auto captured_proof = proof_view.capture(common);
    ASSERT_TRUE(captured_proof.ok()) << captured_proof.message();
  } else {
    another_view = {};
  }
  auto token_result = gd::OutputSource::create(common, token, storage, token_pool);
  ASSERT_TRUE(token_result) << token_result.error().message();
  auto token_source = std::move(*token_result);
  const std::array access{gather};
  const std::array bindings{
      gd::OutputBinding{token_input, &token_source},
      gd::OutputBinding{.store = selected_range ? &another_view : &state,
          .projection = {bank_input, source_ids, gather, access,
              c::detail::kProjectedInputReadExtensionId}, .graph = &second_graph}};
  auto bad_bindings = bindings;
  // Same shape/resource as the real IDs, but it controls another Gather.
  // Accepting it would let the state reader use the old {1,3} IDs while only
  // admitting tickets for {0,2}. Rejection must preserve this unrelated input.
  bad_bindings[1].projection.metadata = unrelated_ids;
  EXPECT_FALSE(gd::bind_outputs(consumer, bad_bindings).ok());
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  if (selected_range) {
    EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
    EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 0u);
    EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, recorded_ranges);
  }
  const auto report = consumer.input_source_report().value();
  EXPECT_EQ(report.source_count, 1u); EXPECT_EQ(report.device_wait_count, 1u);
  EXPECT_EQ(report.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  reader_gate.arm(gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &Gate::wait, &reader_gate), cudaSuccess);
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  gate.release();
  const auto original_status = common.wait();
  EXPECT_FALSE(producer.in_flight()); EXPECT_TRUE(consumer.in_flight());
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  std::array<c::ReadLease, 2> published;
  if (!failed_producer) {
    ASSERT_TRUE(original_status.ok()) << original_status.message();
    ASSERT_TRUE(token_source.detach().ok());
    ASSERT_TRUE(common.detach().ok());
    auto returned = g::slots::take(producer, store);
    ASSERT_TRUE(returned) << returned.error().message();
    ASSERT_EQ(returned->destination_count(), 2u);
    auto rejected = state.publish(0, foreign_writer);
    ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument());
    EXPECT_TRUE(foreign_writer.valid());
    if (selected_range) {
      // Original ordinals never become dense subset ordinals. Both rejected
      // calls leave their original returned writers intact.
      EXPECT_FALSE(another_view.publish(0, returned->destinations()[0]));
      EXPECT_TRUE(returned->destinations()[0].valid());
      EXPECT_FALSE(another_view.publish(1, returned->destinations()[0]));
      EXPECT_TRUE(returned->destinations()[0].valid());
    }
    for (std::size_t i = 0; i < published.size(); ++i) {
      EXPECT_FALSE(returned->destinations()[i].publish());
      auto value = selected_range && i == 1
          ? another_view.publish(i, returned->destinations()[i])
          : state.publish(i, returned->destinations()[i]);
      ASSERT_TRUE(value) << value.error().message();
      published[i] = std::move(*value);
      if (!selected_range || i == 1) EXPECT_FALSE(published[i].try_write());
    }
    if (selected_range) {
      // A complete producer succeeded and truly retired. These checks therefore
      // reach the exact guard validator, not a pending/result/output gate.
      auto guard_binding = bindings[1]; guard_binding.store = &proof_view;
      const auto rejects_ready = [&](std::span<const c::ReadLease> ready,
                                    const char *message) {
        auto bad = guard_binding; bad.store_ready = ready;
        const auto status = gd::bind_outputs(guard_probe, std::span(&bad, 1));
        EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
        EXPECT_EQ(status.message(), message);
      };
      rejects_ready(published, "ready Store source needs every original destination in order");
      rejects_ready(std::span<const c::ReadLease>(published).first(1),
          "ready Store destination differs from its original ticket root or generation");
      EXPECT_NE(sentinels[0].generation(), published[1].generation());
      rejects_ready(std::span<const c::ReadLease>(sentinels).first(1),
          "ready Store destination differs from its original ticket root or generation");
      const std::array partial{published[1].subview(0, sizeof(float)).value()};
      rejects_ready(partial, "slot requires the complete exact original block");
    }
    // Independent tickets and bounded control guards survive the original
    // transaction's take and reset while the successor still has a native use.
    EXPECT_TRUE(producer.reset().ok());
  } else {
    ASSERT_FALSE(original_status.ok());
    auto rejected = selected_range ? another_view.publish(1, foreign_writer) : state.publish(0, foreign_writer);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code(), original_status.code());
    EXPECT_EQ(rejected.error().message(), original_status.message());
  }
  reader_gate.release();
  const auto consumer_status = consumed->wait();
  if (failed_producer) {
    EXPECT_EQ(consumer_status.code(), original_status.code());
    EXPECT_EQ(consumer_status.message(), original_status.message());
    EXPECT_FALSE(consumer.output(token_row)); EXPECT_FALSE(consumer.output(state_rows));
    EXPECT_FALSE(consumer.output(decoy_rows));
  } else {
    ASSERT_TRUE(consumer_status.ok()) << consumer_status.message();
    std::array<float, 2> actual_token_row{};
    std::vector<float> actual_state_rows(selected_range ? 2u : 4u);
    std::array<float, 4> actual_decoy_rows{};
    ASSERT_TRUE(consumer.output(token_row)->storage().read(
        std::as_writable_bytes(std::span(actual_token_row))).ok());
    ASSERT_TRUE(consumer.output(state_rows)->storage().read(
        std::as_writable_bytes(std::span(actual_state_rows))).ok());
    ASSERT_TRUE(consumer.output(decoy_rows)->storage().read(
        std::as_writable_bytes(std::span(actual_decoy_rows))).ok());
    EXPECT_EQ(actual_token_row, (std::array<float, 2>{30, 31}));
    EXPECT_EQ(actual_state_rows, (selected_range ? std::vector<float>{7, 11}
                                               : std::vector<float>{3, 5, 7, 11}));
    EXPECT_EQ(actual_decoy_rows, (std::array<float, 4>{10, 11, 20, 21}));
  }
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  EXPECT_FALSE(reader_gate.timed_out.load(std::memory_order_acquire));
  for (std::size_t i = 0; i < sentinels.size(); ++i) {
    std::array<float, 2> actual{};
    ASSERT_TRUE(sentinels[i].view()->read(std::as_writable_bytes(std::span(actual))).ok());
    const float slot = i == 0 ? 1.0f : 3.0f;
    EXPECT_EQ(actual, (std::array<float, 2>{100.0f + slot, 200.0f + slot}));
  }
  for (const auto &guard : padding) {
    std::array<std::byte, stride - item_bytes> bytes{};
    ASSERT_TRUE(guard.view()->read(bytes).ok());
    for (const auto byte : bytes) EXPECT_EQ(byte, std::byte{0x5a});
  }
  EXPECT_EQ(state_pool.stats().active_uses, 0u);
  *consumed = {}; token_source = {}; consumer = {};
  if (failed_producer) {
    ASSERT_TRUE(common.detach().ok());
    auto returned = g::slots::take(producer, store);
    ASSERT_TRUE(returned) << returned.error().message();
    auto rejected = selected_range ? another_view.publish(1, returned->destinations()[1])
                                   : state.publish(0, returned->destinations()[0]);
    ASSERT_FALSE(rejected); EXPECT_EQ(rejected.error().code(), original_status.code());
    EXPECT_EQ(rejected.error().message(), original_status.message());
    EXPECT_FALSE(returned->destinations()[0].publish());
  }
  state = {}; another_view = {}; proof_view = {}; common = {}; producer = {};
  if (!failed_producer) {
    for (auto &ready : published) {
      auto writer = ready.try_write();
      ASSERT_TRUE(writer) << writer.error().message();
    }
  }
  EXPECT_EQ(token_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}

TEST(GraphOutputSourceCuda, StoreAndTokenShareOnePendingSubmissionAcrossTakeAndPublication) {
  store_relay(false);
}
TEST(GraphOutputSourceCuda, StorePendingReaderPreservesWholeProducerFailure) {
  store_relay(true);
}
TEST(GraphOutputSourceCuda, ContinuousStoreRangeReusesExactReadyCaptureAndOriginalOrdinals) {
  store_relay(false, true);
}
TEST(GraphOutputSourceCuda, ContinuousStoreRangePreservesWholeProducerFailure) {
  store_relay(true, true);
}
TEST(GraphOutputSourceCuda, HybridPendingTokenAndStoreReuseOnlyExistingRanges) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(12).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  auto staging = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto index_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}).value();
  const auto token_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{1, 1}).value();
  const auto table_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{4, 1}).value();
  const auto copied_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{2, 1}).value();
  const auto item = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{2}).value();
  const auto rows_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 2}).value();
  const auto ids_desc = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{2}).value();
  constexpr std::size_t stride = 256, item_bytes = 2 * sizeof(float);
  const auto schema = g::slots::Schema::create(item, 4, stride).value();
  const auto capacity_desc = schema.capacity_description().value();
  auto backing = hyle::mem::Buffer::create(storage.ref(), 4 * stride, stride).value();
  ASSERT_TRUE(backing.view().fill(std::byte{0x5a}).ok());
  auto state_pool = c::LeasePool::adopt(std::move(backing), {.max_blocks = 8}).value();
  const auto layout = c::LeaseSlotLayout::create(state_pool, item, 4, stride).value();
  std::array<c::WriteLease, 4> initial;
  std::array<c::ReadLease, 4> padding;
  for (std::size_t i = 0; i < initial.size(); ++i) {
    initial[i] = state_pool.allocate(item_bytes, stride).value();
    ASSERT_EQ(layout.locate(initial[i])->index, i);
    const std::array<float, 2> old{100.0f + static_cast<float>(i), 200.0f + static_cast<float>(i)};
    ASSERT_TRUE(initial[i].view()->write(std::as_bytes(std::span(old))).ok());
    auto gap = state_pool.allocate(stride - item_bytes, 1).value();
    padding[i] = gap.publish().value();
  }
  std::array<c::ReadLease, 2> sentinels{
      initial[1].publish().value(), initial[3].publish().value()};
  std::array<c::WriteLease, 2> destinations{std::move(initial[0]), std::move(initial[2])};
  auto inputs = c::LeasePool::create(storage.ref(), 5 * stride, {.max_blocks = 5}).value();
  auto table = inputs.allocate(4 * sizeof(std::int32_t), stride).value();
  auto index = inputs.allocate(sizeof(std::int32_t), stride).value();
  auto rows = inputs.allocate(4 * sizeof(float), stride).value();
  auto marker = inputs.allocate(sizeof(std::int32_t), stride).value();
  auto other_marker = inputs.allocate(sizeof(std::int32_t), stride).value();
  constexpr std::array<std::int32_t, 4> token_table{2, 1, 3, 0};
  constexpr std::array<std::int32_t, 1> select{0}, marker_value{7}, warm_value{1};
  constexpr std::array<float, 4> incoming{3, 5, 7, 11};
  ASSERT_TRUE(table.view()->write(std::as_bytes(std::span(token_table))).ok());
  ASSERT_TRUE(index.view()->write(std::as_bytes(std::span(select))).ok());
  ASSERT_TRUE(rows.view()->write(std::as_bytes(std::span(incoming))).ok());
  ASSERT_TRUE(marker.view()->write(std::as_bytes(std::span(marker_value))).ok());
  ASSERT_TRUE(other_marker.view()->write(std::as_bytes(std::span(marker_value))).ok());
  auto table_ready = table.publish().value(), index_ready = index.publish().value();
  auto rows_ready = rows.publish().value(), marker_ready = marker.publish().value();
  auto other_marker_ready = other_marker.publish().value();
  const auto marker_storage = marker_ready.view().value();
  const auto other_marker_storage = other_marker_ready.view().value();
  EXPECT_EQ(marker_storage.native_handle(), other_marker_storage.native_handle());
  ASSERT_NE(marker_storage.offset(), other_marker_storage.offset());
  auto token_pool = c::LeasePool::create(storage.ref(), stride, {.max_blocks = 1}).value();
  auto token_writer = token_pool.allocate(sizeof(std::int32_t), stride).value();
  ASSERT_TRUE(token_writer.view()->write(std::as_bytes(std::span(warm_value))).ok());
  auto warm_token = token_writer.publish().value();
  const auto token_generation = warm_token.generation();
  const auto token_view = warm_token.tensor(token_desc).value();
  auto output_pool = c::LeasePool::create(storage.ref(), stride, {.max_blocks = 1}).value();
  auto copied_writer = output_pool.allocate(2 * sizeof(std::int32_t), stride).value();
  auto destination_pool = c::LeasePool::create(storage.ref(), 2 * stride, {.max_blocks = 2}).value();
  const auto destination_layout = c::LeaseSlotLayout::create(destination_pool, item, 2, stride).value();
  std::array<c::WriteLease, 2> result_writers;
  for (auto &writer : result_writers) writer = destination_pool.allocate(item_bytes, stride).value();

  auto first = g::GraphBuilder::create().value();
  const auto table_input = first.input(table_desc).value();
  const auto index_input = first.input(index_desc).value();
  const auto row_input = first.input(rows_desc).value();
  const auto first_ids = first.input(ids_desc).value();
  const auto token = first.add(c::Gather{.axis = 0}, table_input, index_input).value();
  ASSERT_TRUE(first.publish(token).ok());
  const auto store = first.add_node(c::Operation::custom(g::slots::Store{schema}),
      std::array{row_input, first_ids}).value();
  const auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage);
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  ASSERT_EQ(producer_plan->report.segment_count(), 1u);
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(table_input, table_desc, table_ready).ok());
  ASSERT_TRUE(producer.bind(index_input, index_desc, index_ready).ok());
  ASSERT_TRUE(producer.bind(row_input, rows_desc, rows_ready).ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(g::slots::prepare(producer, queues, {.host_workspace = staging.ref()}).ok());
  auto transaction = g::slots::SlotBindings::create(layout, {}, destinations).value();
  ASSERT_TRUE(g::slots::bind(producer, first_graph, store, std::move(transaction)).ok());
  // The real A carrier is already initialized by cold binding. Keep only its
  // bounded IDs lease and capacity description; never retain a writer pointer
  // or the borrowed transaction across the following cold take.
  struct ColdExport final { c::ConstTensorView capacity{}, indices{}; c::ReadLease ids; } control;
  const auto exported = gd::RegionAccess::visit(producer, store,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
        auto &result = *static_cast<ColdExport *>(opaque);
        const auto *extension = view.prepared_vtable && view.prepared_vtable->query_extension
            ? static_cast<const g::slots::detail::StoreRegionVTable *>(
                view.prepared_vtable->query_extension(view.prepared,
                    g::slots::detail::kStoreBindingExtensionId)) : nullptr;
        if (!extension || !extension->binding)
          return hyle::Status::Unimplemented("cold Store has no original binding export");
        auto *binding = extension->binding(view.invocation);
        if (!binding) return hyle::Status::FailedPrecondition("cold Store binding is absent");
        HYLE_ASSIGN_OR_RETURN(auto source, binding->export_pending());
        result.capacity = source.capacity; result.indices = source.indices;
        result.ids = std::move(source.indices_guard);
        return hyle::Status::Ok();
      }, &control);
  ASSERT_TRUE(exported.ok()) << exported.message();
  ASSERT_TRUE(gd::RegionAccess::outcome_available(producer));
  auto cold_returned = g::slots::take(producer, store);
  ASSERT_TRUE(cold_returned) << cold_returned.error().message();
  std::array<c::ReadLease, 2> warm_sources;
  std::array<std::uint64_t, 2> generations{};
  std::array<hyle::mem::ConstBufferView, 2> source_storage;
  for (std::size_t i = 0; i < warm_sources.size(); ++i) {
    warm_sources[i] = cold_returned->destinations()[i].publish().value();
    generations[i] = warm_sources[i].generation();
    source_storage[i] = warm_sources[i].view().value();
  }

  auto second = g::GraphBuilder::create().value();
  const auto token_input = second.input(token_desc).value();
  const auto marker_input = second.input(token_desc).value();
  const auto copy0 = second.add_node(c::Operation(c::Concat{.axis = 0}),
      std::array{token_input, marker_input}).value();
  const auto intermediate = second.result(copy0).value();
  const auto copy1 = second.add_node(c::Operation(c::Concat{.axis = 0}),
      std::array{intermediate}).value();
  const auto copied_token = second.result(copy1).value();
  const auto bank_input = second.input(capacity_desc).value();
  const auto source_ids = second.input(ids_desc).value();
  const auto second_ids = second.input(ids_desc).value();
  const auto gather = second.add_node(c::Operation(c::Gather{.axis = 0}),
      std::array{bank_input, source_ids}).value();
  const auto gathered = second.result(gather).value();
  const auto result_schema = g::slots::Schema::create(item, 2, stride).value();
  const auto result_store = second.add_node(c::Operation::custom(g::slots::Store{result_schema}),
      std::array{gathered, second_ids}).value();
  ASSERT_TRUE(second.publish(copied_token).ok());
  const auto second_graph = std::move(second).finish().value();
  // Keep a genuine prepared, non-capturable Store gap. The normal CUDA
  // assembly now supplies an indexed-copy kernel that can capture this Store.
  // This fixture selects the original Store's ordered-copy fallback instead;
  // its native Context, Queue, permissions and math remain unchanged.
  auto execution_source = g::ExecutionSource::acquire(context).value();
  const std::array domains{g::StorageDomain{.resource = storage.ref(), .owner = storage}};
  const std::array targets{g::ExecutionTarget{
      .source = execution_source, .default_storage = g::StorageDomainId{0}}};
  const auto *store_provider = g::slots::detail::store_provider();
  auto region_source = gd::GraphRegionSource::shared_queue(execution_source, store_provider);
  ASSERT_TRUE(region_source) << region_source.error().message();
  g::slots::detail::StoreAdapter store_adapter{storage.ref(), context.device().device_class(), {}};
  const std::array adapters{gd::RegionAdapter{
      .source = std::move(*region_source), .provider_index = UINT32_MAX,
      .target = g::ExecutionTargetId{0}, .interface = store_provider, .context = &store_adapter}};
  auto consumer_plan = gd::compile_with_adapters(second_graph,
      {.storage_domains = domains, .targets = targets}, {}, adapters);
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  ASSERT_EQ(consumer_plan->report.segment_count(), 1u);
  bool captured_math = false, prepared_gap = false;
  for (std::size_t i = 0; i < consumer_plan->report.step_count(); ++i) {
    const auto step = consumer_plan->report.step(i).value();
    bool has_copy = false, has_store = false;
    for (std::size_t j = 0; j < step.node_count; ++j) {
      const auto node = consumer_plan->report.step_node(i, j).value();
      has_copy |= node == copy0 || node == copy1;
      has_store |= node == result_store;
    }
    if (has_copy) {
      EXPECT_EQ(step.kind, g::CompileStepKind::kProviderRegion);
      EXPECT_EQ(step.node_count, 2u);
      EXPECT_EQ(step.runtime.replay_class, c::ProviderReplayClass::kManaged);
      EXPECT_TRUE(step.runtime.blocks_host); captured_math = true;
    }
    if (has_store) {
      EXPECT_EQ(step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
      EXPECT_EQ(step.runtime.replay_class, c::ProviderReplayClass::kPrepared);
      EXPECT_FALSE(step.runtime.blocks_host);
      EXPECT_FALSE(step.runtime.capture_compatible); prepared_gap = true;
    }
  }
  ASSERT_TRUE(captured_math); ASSERT_TRUE(prepared_gap);
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  ASSERT_TRUE(consumer.bind(token_input, token_desc, warm_token).ok());
  ASSERT_TRUE(consumer.bind(marker_input, token_desc, marker_ready).ok());
  ASSERT_TRUE(consumer.bind(source_ids, ids_desc, control.ids).ok());
  ASSERT_TRUE(consumer.bind(copied_token, copied_desc, std::move(copied_writer)).ok());
  ASSERT_TRUE(g::slots::prepare(consumer, queues, {.host_workspace = staging.ref()}).ok());
  auto result_transaction = g::slots::SlotBindings::create(destination_layout, {}, result_writers).value();
  ASSERT_TRUE(g::slots::bind(consumer, second_graph, result_store, std::move(result_transaction)).ok());
  ASSERT_TRUE(gd::RegionAccess::reserve_reads(consumer, 2, {}).ok());
  const std::array access{gather};
  ASSERT_TRUE(gd::RegionAccess::bind_read_projection(consumer,
      {bank_input, {}, access}, control.capacity, warm_sources).ok());
  const auto warm = consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kPreferCapture, .host_wait = g::HostWaitPolicy::kReject});
  ASSERT_TRUE(warm.ok()) << warm.message();
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kHybrid);
  const auto recorded_ranges = consumer.replay_report()->recorded_capture_range_count;
  ASSERT_GT(recorded_ranges, 0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, 0u);
  EXPECT_EQ(consumer.replay_report()->immediate_transfer_count, 0u);
  EXPECT_EQ(consumer.replay_report()->host_wait_dependency_count, 0u);
  EXPECT_FALSE(consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject}).ok());
  const auto restored = gd::ReplayAccess::prepare_reuse(consumer, queues);
  ASSERT_TRUE(restored.ok()) << restored.message();
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kHybrid);
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, recorded_ranges);
  // A real captured input changes address while retaining the same value.
  // A miss cannot record this tuple or silently execute managed Commands.
  ASSERT_TRUE(consumer.bind(marker_input, token_desc, other_marker_ready).ok());
  const auto reusable = gd::ReplayAccess::can_reuse_capture(consumer,
      std::array{gd::ReplayInputBinding{token_input, token_view}});
  ASSERT_TRUE(reusable) << reusable.error().message();
  EXPECT_FALSE(*reusable);
  EXPECT_FALSE(gd::ReplayAccess::prepare_reuse(consumer, queues).ok());
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kUnprepared);
  EXPECT_FALSE(consumer.replay_report());
  ASSERT_TRUE(consumer.bind(marker_input, token_desc, marker_ready).ok());
  // Only the exact ready permissions are withdrawn. Both genuine blocks
  // must become writable again without changing generation or coordinates.
  const auto released = gd::RegionAccess::release_read_projection(consumer, bank_input);
  ASSERT_TRUE(released.ok()) << released.message();
  const std::array old_inputs{source_ids, token_input};
  ASSERT_TRUE(consumer.unbind(old_inputs).ok());
  control.ids = {};
  for (std::size_t i = 0; i < warm_sources.size(); ++i) {
    auto writer = warm_sources[i].try_write();
    ASSERT_TRUE(writer) << writer.error().message();
    EXPECT_EQ(writer->generation(), generations[i]);
    const auto actual = writer->view().value();
    EXPECT_EQ(actual.resource(), source_storage[i].resource());
    EXPECT_EQ(actual.native_handle(), source_storage[i].native_handle());
    EXPECT_EQ(actual.offset(), source_storage[i].offset());
    EXPECT_EQ(actual.size(), source_storage[i].size());
    destinations[i] = std::move(*writer);
  }
  auto reclaimed_token = warm_token.try_write();
  ASSERT_TRUE(reclaimed_token) << reclaimed_token.error().message();
  EXPECT_EQ(reclaimed_token->generation(), token_generation);
  const auto actual_token_storage = reclaimed_token->view().value();
  EXPECT_EQ(actual_token_storage.native_handle(), token_view.storage().native_handle());
  EXPECT_EQ(actual_token_storage.offset(), token_view.storage().offset());
  EXPECT_EQ(actual_token_storage.size(), token_view.storage().size());
  ASSERT_TRUE(producer.bind(token, token_desc, std::move(*reclaimed_token)).ok());
  auto actual_transaction = g::slots::SlotBindings::create(layout, {}, destinations).value();
  const auto rebound = g::slots::bind(producer, first_graph, store, std::move(actual_transaction));
  ASSERT_TRUE(rebound.ok()) << rebound.message();
  ColdExport current;
  const auto same_carrier = gd::RegionAccess::visit(producer, store,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> hyle::Status {
        auto &result = *static_cast<ColdExport *>(opaque);
        const auto *extension = static_cast<const g::slots::detail::StoreRegionVTable *>(
            view.prepared_vtable->query_extension(view.prepared, g::slots::detail::kStoreBindingExtensionId));
        if (!extension || !extension->binding) return hyle::Status::Internal("Store export disappeared");
        HYLE_ASSIGN_OR_RETURN(auto source, extension->binding(view.invocation)->export_pending());
        result.indices = source.indices;
        return hyle::Status::Ok();
      }, &current);
  ASSERT_TRUE(same_carrier.ok()) << same_carrier.message();
  EXPECT_EQ(current.indices.description(), control.indices.description());
  EXPECT_EQ(current.indices.storage().resource(), control.indices.storage().resource());
  EXPECT_EQ(current.indices.storage().native_handle(), control.indices.storage().native_handle());
  EXPECT_EQ(current.indices.storage().offset(), control.indices.storage().offset());
  EXPECT_EQ(current.indices.storage().size(), control.indices.storage().size());
  auto state_result = gd::StoreSource::prepare(producer, store, storage);
  ASSERT_TRUE(state_result) << state_result.error().message();
  auto state = std::move(*state_result);
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  Gate gate, reader_gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  ASSERT_TRUE(state.capture(common).ok());
  auto token_result = gd::OutputSource::create(common, token, storage, token_pool);
  ASSERT_TRUE(token_result) << token_result.error().message();
  auto token_source = std::move(*token_result);
  const std::array bindings{
      gd::OutputBinding{token_input, &token_source},
      gd::OutputBinding{.store = &state,
          .projection = {bank_input, source_ids, gather, access,
              c::detail::kProjectedInputReadExtensionId}, .graph = &second_graph}};
  // Source authority can be bound before an unrelated captured input exists.
  // Only final preparation/submission qualifies the complete physical tuple.
  ASSERT_TRUE(consumer.unbind(marker_input).ok());
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  EXPECT_FALSE(gd::prepare_outputs(consumer, bindings, queues).ok());
  ASSERT_TRUE(consumer.bind(marker_input, token_desc, other_marker_ready).ok());
  EXPECT_FALSE(gd::prepare_outputs(consumer, bindings, queues).ok());
  EXPECT_FALSE(gd::submit_outputs(consumer, bindings, queues));
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kUnprepared);
  EXPECT_FALSE(consumer.replay_report());
  ASSERT_TRUE(common.poll());
  EXPECT_EQ(*common.poll(), c::CompletionPoll::kPending);
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  ASSERT_TRUE(consumer.bind(marker_input, token_desc, marker_ready).ok());
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kHybrid);
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, recorded_ranges);
  EXPECT_EQ(consumer.replay_report()->immediate_transfer_count, 0u);
  EXPECT_EQ(consumer.replay_report()->host_wait_dependency_count, 0u);
  const auto dependency = consumer.input_source_report().value();
  EXPECT_EQ(dependency.source_count, 1u); EXPECT_EQ(dependency.device_wait_count, 1u);
  EXPECT_EQ(dependency.host_wait_count, 0u);
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  EXPECT_FALSE(consumer.submit(queues));
  reader_gate.arm(gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &Gate::wait, &reader_gate), cudaSuccess);
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  // The same Hybrid execution is also a producer of its original submission;
  // transfer that one exact proxy, without adding a view, graph or observer.
  auto successor_result = gd::SubmissionSource::create(consumer, std::move(*consumed), storage);
  ASSERT_TRUE(successor_result) << successor_result.error().message();
  auto successor = std::move(*successor_result);
  EXPECT_FALSE(consumed->valid());
  ASSERT_TRUE(successor.poll());
  EXPECT_EQ(*successor.poll(), c::CompletionPoll::kPending);
  gate.release();
  ASSERT_TRUE(common.wait().ok());
  EXPECT_TRUE(consumer.in_flight());
  EXPECT_EQ(*successor.poll(), c::CompletionPoll::kPending);
  ASSERT_TRUE(token_source.detach().ok()); ASSERT_TRUE(common.detach().ok());
  auto returned = g::slots::take(producer, store);
  ASSERT_TRUE(returned) << returned.error().message();
  std::array<c::ReadLease, 2> published;
  for (std::size_t i = 0; i < published.size(); ++i) {
    auto ready = state.publish(i, returned->destinations()[i]);
    ASSERT_TRUE(ready) << ready.error().message();
    published[i] = std::move(*ready);
    EXPECT_FALSE(published[i].try_write());
  }
  EXPECT_TRUE(producer.reset().ok());
  reader_gate.release();
  const auto terminal = successor.wait();
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  ASSERT_TRUE(successor.detach().ok());
  std::array<std::int32_t, 2> actual_token{};
  ASSERT_TRUE(consumer.output(copied_token)->storage().read(
      std::as_writable_bytes(std::span(actual_token))).ok());
  EXPECT_EQ(actual_token, (std::array<std::int32_t, 2>{2, 7}));
  auto result = g::slots::take(consumer, result_store);
  ASSERT_TRUE(result) << result.error().message();
  for (std::size_t i = 0; i < 2; ++i) {
    auto ready = result->destinations()[i].publish();
    ASSERT_TRUE(ready) << ready.error().message();
    std::array<float, 2> actual{};
    ASSERT_TRUE(ready->view()->read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, (std::array<float, 2>{incoming[2 * i], incoming[2 * i + 1]}));
  }
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  EXPECT_FALSE(reader_gate.timed_out.load(std::memory_order_acquire));
  EXPECT_EQ(state_pool.stats().active_uses, 0u);
  EXPECT_EQ(destination_pool.stats().active_uses, 0u);
  for (std::size_t i = 0; i < sentinels.size(); ++i) {
    std::array<float, 2> actual{};
    ASSERT_TRUE(sentinels[i].view()->read(std::as_writable_bytes(std::span(actual))).ok());
    const auto slot = i == 0 ? 1.0f : 3.0f;
    EXPECT_EQ(actual, (std::array<float, 2>{100.0f + slot, 200.0f + slot}));
  }
  for (const auto &guard : padding) {
    std::array<std::byte, stride - item_bytes> bytes{};
    ASSERT_TRUE(guard.view()->read(bytes).ok());
    for (const auto byte : bytes) EXPECT_EQ(byte, std::byte{0x5a});
  }
  successor = {}; token_source = {}; consumer = {}; state = {}; common = {}; producer = {};
  for (auto &ready : published) EXPECT_TRUE(ready.try_write());
  EXPECT_EQ(token_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}
// Raw exactly representable finite values make the half/bfloat checks test
// transport and real native reads, without an unrelated conversion oracle.
std::vector<std::byte> floating_rows(c::DataType type) {
  constexpr std::array<float, 8> floats{10.5F, -11.25F, 20.5F, -21.25F, 30.5F, -31.25F, 40.5F, -41.25F};
  std::vector<std::byte> result;
  if (type == c::DataType::kFloat32) {
    const auto bytes = std::as_bytes(std::span(floats)); result.assign(bytes.begin(), bytes.end());
  } else {
    std::array<std::uint16_t, 8> encoded{};
    if (type == c::DataType::kBFloat16) {
      for (std::size_t i = 0; i < floats.size(); ++i)
        encoded[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(floats[i]) >> 16);
    } else {
      EXPECT_EQ(type, c::DataType::kFloat16);
      encoded = {0x4940, 0xc9a0, 0x4d20, 0xcd50, 0x4fa0, 0xcfd0, 0x5110, 0xd128};
    }
    const auto bytes = std::as_bytes(std::span(encoded)); result.assign(bytes.begin(), bytes.end());
  }
  return result;
}

void mixed_full_writer_relay(c::DataType type, bool failed_producer = false) {
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(8).ok());
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(memory)).value();
  const auto i32 = c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{1}).value();
  const auto indices = c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{4}).value();
  const auto table_desc = c::TensorDesc::dense(type, std::array<std::int64_t, 2>{4, 2}).value();
  const auto row_desc = c::TensorDesc::dense(type, std::array<std::int64_t, 2>{1, 2}).value();
  const auto payload = floating_rows(type);
  const auto row_bytes = row_desc.required_bytes().value();
  auto inputs = c::LeasePool::create(storage.ref(), 8 * 256, {.max_blocks = 8}).value();
  auto sources = c::LeasePool::create(storage.ref(), 4 * 256, {.max_blocks = 4}).value();
  auto outputs = c::LeasePool::create(storage.ref(), 256, {.max_blocks = 1}).value();
  const auto upload = [&](std::span<const std::byte> bytes) {
    auto writer = inputs.allocate(bytes.size(), 256).value();
    const auto status = writer.view()->write(bytes);
    EXPECT_TRUE(status.ok()) << status.message();
    return writer.publish().value();
  };
  constexpr std::array<std::int32_t, 4> token_values{2, 1, 3, 0},
      permutation{3, 1, 0, 2}, next_permutation{0, 2, 3, 1};
  const std::array<std::int32_t, 1> wanted{failed_producer ? 4 : 0};
  constexpr std::array<std::int32_t, 1> next_wanted{1}, previous_wanted{3};
  auto tokens_ready = upload(std::as_bytes(std::span(token_values)));
  auto rows_ready = upload(payload);
  auto wanted_ready = upload(std::as_bytes(std::span(wanted)));
  auto permutation_ready = upload(std::as_bytes(std::span(permutation)));
  auto next_wanted_ready = upload(std::as_bytes(std::span(next_wanted)));
  auto next_permutation_ready = upload(std::as_bytes(std::span(next_permutation)));
  auto previous_ready = upload(std::as_bytes(std::span(previous_wanted)));
  // New-generation writers are prepared before either host gate is armed.
  auto token_writer = sources.allocate(sizeof(std::int32_t), 256).value();
  auto rows_writer = sources.allocate(payload.size(), 256).value();
  auto next_token_writer = sources.allocate(sizeof(std::int32_t), 256).value();
  auto next_rows_writer = sources.allocate(payload.size(), 256).value();

  auto first = g::GraphBuilder::create().value();
  const auto first_tokens = first.input(indices).value(), first_index = first.input(i32).value();
  const auto first_rows = first.input(table_desc).value(), first_permutation = first.input(indices).value();
  const auto token = first.add(c::Gather{.axis = 0}, first_tokens, first_index).value();
  const auto floats = first.add(c::Gather{.axis = 0}, first_rows, first_permutation).value();
  ASSERT_TRUE(first.publish(token).ok()); ASSERT_TRUE(first.publish(floats).ok());
  const auto first_graph = std::move(first).finish().value();
  auto producer_plan = g::cuda::compile(first_graph, context, storage);
  ASSERT_TRUE(producer_plan) << producer_plan.error().message();
  auto producer = producer_plan->executable.create_invocation().value();
  ASSERT_TRUE(producer.prepare_resources().ok());
  ASSERT_TRUE(producer.bind(first_tokens, indices, tokens_ready).ok());
  ASSERT_TRUE(producer.bind(first_rows, table_desc, rows_ready).ok());
  ASSERT_TRUE(producer.bind(first_index, i32, wanted_ready).ok());
  ASSERT_TRUE(producer.bind(first_permutation, indices, permutation_ready).ok());
  ASSERT_TRUE(producer.bind(token, i32, std::move(token_writer)).ok());
  ASSERT_TRUE(producer.bind(floats, table_desc, std::move(rows_writer)).ok());

  auto second = g::GraphBuilder::create().value();
  const auto second_index = second.input(i32).value(), second_rows = second.input(table_desc).value();
  const auto selected = second.add(c::Gather{.axis = 0}, second_rows, second_index).value();
  ASSERT_TRUE(second.publish(selected).ok());
  const auto second_graph = std::move(second).finish().value();
  auto consumer_plan = g::cuda::compile(second_graph, context, storage);
  ASSERT_TRUE(consumer_plan) << consumer_plan.error().message();
  auto consumer = consumer_plan->executable.create_invocation().value();
  ASSERT_TRUE(consumer.prepare_resources().ok());
  ASSERT_TRUE(consumer.bind(second_index, i32, previous_ready).ok());
  ASSERT_TRUE(consumer.bind(second_rows, table_desc, rows_ready).ok());
  ASSERT_TRUE(consumer.bind(selected, row_desc, outputs.allocate(row_bytes, 256).value()).ok());
  for (const auto *plan : {&*producer_plan, &*consumer_plan})
    for (std::size_t i = 0; i < plan->report.step_count(); ++i) {
      const auto step = plan->report.step(i).value();
      ASSERT_EQ(step.runtime.mode, c::ProviderRuntimeMode::kSharedQueue);
      ASSERT_EQ(step.runtime.replay_class, c::ProviderReplayClass::kPrepared);
      ASSERT_FALSE(step.runtime.blocks_host);
    }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());
  ASSERT_TRUE(consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());
  Gate gate, reader_gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  auto token_result = gd::OutputSource::create(common, token, storage, sources);
  auto float_result = gd::OutputSource::create(common, floats, storage, sources);
  ASSERT_TRUE(token_result) << token_result.error().message();
  ASSERT_TRUE(float_result) << float_result.error().message();
  std::array<gd::OutputSource, 2> views{std::move(*token_result), std::move(*float_result)};
  EXPECT_FALSE(gd::OutputSource::create(common, first_rows, storage, inputs)); // An input is not an owned output.
  EXPECT_TRUE(views[1].detach().IsUnavailable());
  const std::array rejected{gd::OutputBinding{second_index, &views[0]},
                            gd::OutputBinding{first_rows, &views[1]}};
  EXPECT_FALSE(gd::bind_outputs(consumer, rejected).ok());
  EXPECT_FALSE(previous_ready.try_write()); // The first old binding was not consumed by a late error.
  ASSERT_TRUE(consumer.replay_report()); // Its preexisting preparation also survived.
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCommands);
  const std::array bindings{gd::OutputBinding{second_index, &views[0]},
                            gd::OutputBinding{second_rows, &views[1]}};
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto prepared = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  ASSERT_TRUE(common.poll()); EXPECT_EQ(*common.poll(), c::CompletionPoll::kPending);
  const auto report = consumer.input_source_report().value();
  EXPECT_EQ(report.source_count, 1u); EXPECT_EQ(report.device_wait_count, 1u); EXPECT_EQ(report.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  if (!failed_producer) {
    // A can retire and be rebound while B still retains both old payloads.
    reader_gate.arm(gate.stream);
    ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &Gate::wait, &reader_gate), cudaSuccess);
  }
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  ASSERT_TRUE(consumed->poll()); EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  EXPECT_FALSE(gate.released.load(std::memory_order_acquire));
  EXPECT_TRUE(producer.in_flight()); EXPECT_TRUE(consumer.in_flight());
  gate.release();
  c::Completion next;
  const auto original_status = common.wait();
  if (!failed_producer) {
    ASSERT_TRUE(original_status.ok()) << original_status.message();
    ASSERT_TRUE(consumer.in_flight());
    ASSERT_TRUE(views[0].detach().ok()); ASSERT_TRUE(views[1].detach().ok());
    ASSERT_TRUE(common.detach().ok());
    ASSERT_TRUE(gd::RegionAccess::outcome_available(producer));
    ASSERT_TRUE(producer.bind(first_index, i32, next_wanted_ready).ok());
    ASSERT_TRUE(producer.bind(first_permutation, indices, next_permutation_ready).ok());
    ASSERT_TRUE(producer.bind(token, i32, std::move(next_token_writer)).ok());
    ASSERT_TRUE(producer.bind(floats, table_desc, std::move(next_rows_writer)).ok());
    ASSERT_TRUE(producer.prepare_replay(queues,
        {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());
    auto reused = producer.submit(queues);
    ASSERT_TRUE(reused) << reused.error().message(); next = std::move(*reused);
    ASSERT_TRUE(next.poll()); EXPECT_EQ(*next.poll(), c::CompletionPoll::kPending);
    EXPECT_EQ(*common.poll(), c::CompletionPoll::kComplete);
    reader_gate.release();
  }
  const auto status = consumed->wait();
  EXPECT_FALSE(consumer.in_flight());
  if (failed_producer) {
    EXPECT_FALSE(producer.in_flight());
    ASSERT_FALSE(original_status.ok()); ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), original_status.code()); EXPECT_EQ(status.message(), original_status.message());
    EXPECT_FALSE(consumer.output(selected));
    for (auto &view : views) {
      const auto detached = view.detach();
      EXPECT_EQ(detached.code(), original_status.code()); EXPECT_EQ(detached.message(), original_status.message());
    }
  } else {
    ASSERT_TRUE(status.ok()) << status.message(); ASSERT_TRUE(next.wait().ok());
    EXPECT_FALSE(producer.in_flight());
    std::vector<std::byte> actual(row_bytes);
    ASSERT_TRUE(consumer.output(selected)->storage().read(actual).ok());
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), payload.begin())); // A's old permutation row2 was original row0.
    std::array<std::int32_t, 1> next_token{};
    ASSERT_TRUE(producer.output(token)->storage().read(std::as_writable_bytes(std::span(next_token))).ok());
    EXPECT_EQ(next_token, (std::array<std::int32_t, 1>{1}));
    std::vector<std::byte> next_rows(payload.size()), expected;
    for (auto row : next_permutation)
      expected.insert(expected.end(), payload.begin() + row * row_bytes, payload.begin() + (row + 1) * row_bytes);
    ASSERT_TRUE(producer.output(floats)->storage().read(next_rows).ok()); EXPECT_EQ(next_rows, expected);
  }
  EXPECT_FALSE(gate.timed_out.load(std::memory_order_acquire));
  EXPECT_FALSE(reader_gate.timed_out.load(std::memory_order_acquire));
  EXPECT_EQ(sources.stats().active_uses, 0u); EXPECT_EQ(outputs.stats().active_uses, 0u);
  *consumed = {}; views = {}; consumer = {};
  if (failed_producer) {
    ASSERT_TRUE(common.detach().ok()); EXPECT_TRUE(gd::RegionAccess::outcome_available(producer));
    const auto retained = common.wait();
    EXPECT_EQ(retained.code(), original_status.code()); EXPECT_EQ(retained.message(), original_status.message());
  }
  common = {}; producer = {};
  next_token_writer = {}; next_rows_writer = {};
  EXPECT_EQ(sources.stats().live_blocks, 0u); EXPECT_EQ(outputs.stats().live_blocks, 0u);
}

TEST(GraphOutputSourceCuda, PendingF32FullWriterSharesIntegerSubmission) {
  mixed_full_writer_relay(c::DataType::kFloat32);
}
TEST(GraphOutputSourceCuda, PendingF16FullWriterSharesIntegerSubmission) {
  mixed_full_writer_relay(c::DataType::kFloat16);
}
TEST(GraphOutputSourceCuda, PendingBF16FullWriterSharesIntegerSubmission) {
  mixed_full_writer_relay(c::DataType::kBFloat16);
}
TEST(GraphOutputSourceCuda, MixedFullWritersPreserveOriginalProducerFailure) {
  mixed_full_writer_relay(c::DataType::kFloat32, true);
}
} // namespace
