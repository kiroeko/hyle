#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/output_source.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"
#include "hyle/graph/slots.h"
#include "hyle/graph/graph.h"

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
  EXPECT_FALSE(gd::OutputSource::create(producer, token,
      std::move(foreign), storage, source_pool));
  EXPECT_TRUE(foreign.valid());
  auto source_result = gd::OutputSource::create(producer, token,
      std::move(*submitted), storage, source_pool);
  ASSERT_TRUE(source_result) << source_result.error().message();
  auto source = std::move(*source_result);
  EXPECT_FALSE(submitted->valid());
  ASSERT_TRUE(source.poll());
  EXPECT_EQ(*source.poll(), c::CompletionPoll::kPending);
  EXPECT_TRUE(source.detach().IsUnavailable());
  EXPECT_FALSE(producer.reset().ok());
  EXPECT_FALSE(gd::bind_output(producer, first_index, source).ok());
  const auto bound = gd::bind_output(consumer, second_index, source);
  ASSERT_TRUE(bound.ok()) << bound.message();
  EXPECT_FALSE(gd::bind_output(consumer, second_index, source).ok());
  EXPECT_FALSE(consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject}).ok());
  const auto prepared = gd::prepare_output(consumer, second_index, source, queues);
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
  auto consumed = gd::submit_output(consumer, second_index, source, queues);
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
  consumer = {}; producer = {};
  EXPECT_EQ(source_pool.stats().live_blocks, 0u);
  EXPECT_EQ(output_pool.stats().live_blocks, 0u);
}

TEST(GraphOutputSourceCuda, PendingI32UsesOriginalQueueAndLeasesForCheckedGather) { relay(false); }
TEST(GraphOutputSourceCuda, PendingI32PreparesRealStoreRegionBindingHooks) { relay(false, false, true); }
TEST(GraphOutputSourceCuda, ProducerCheckedFailureRejectsConsumerPublicationAfterRetirement) { relay(true); }
TEST(GraphOutputSourceCuda, RetiredSourceDetachesWhileReaderPendingAndProducerReusesNewWriter) { relay(false, true); }
} // namespace
