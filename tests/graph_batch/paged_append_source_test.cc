#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/paged_frame.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/paged_append_source.h"
#include "hyle/graph/result.h"

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

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace g = hyle::graph;
namespace gd = g::detail;
namespace gb = g::batch;
namespace gbd = gb::detail;

// A bounded original-Queue host gate. Every assertion exit releases it before
// the Invocation/resource owners unwind; no device spin kernel is involved.
struct PageSourceGate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false}, timed_out{false};
  std::mutex lock;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<PageSourceGate *>(opaque);
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
  ~PageSourceGate() {
    release();
    if (watchdog.joinable()) watchdog.join();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

std::uint16_t bf16(float value) {
  // The chosen finite dyadic test values are exactly BF16-representable.
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}

void pending_original_pages_publish_while_attention_reads(bool owned_layout) {
  SCOPED_TRACE(owned_layout ? "owned layout" : "explicit metadata input");
  auto context_result = g::cuda::create_context();
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(8).ok());
  auto device = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto storage = c::StorageOwner::adopt(context, std::move(device)).value();
  auto staging = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto plain = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kPaged,
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 2, .max_query_rows = 4, .max_kv_rows = 10,
      .max_pages = 7, .max_page_refs = 6,
      .query_heads = 2, .kv_heads = 1,
      .qk_head_dim = 64, .value_head_dim = 64, .page_size = 2};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t, 3> old_q{0, 1, 2}, append_q{0, 2, 4};
  const std::array<std::int32_t, 3> old_ptr{0, 2, 4}, next_ptr{0, 3, 6};
  const std::array<std::int32_t, 4> old_ids{0, 1, 2, 3};
  const std::array<std::int32_t, 6> next_ids{0, 1, 4, 2, 5, 6};
  const std::array<std::int32_t, 2> tails{1, 1};
  const auto previous_layout = schema.version({old_q, old_ptr, old_ids, tails}).value();
  const auto produced_layout = schema.version({append_q, next_ptr, next_ids, tails}).value();
  // Attention consumes a different query layout, with exactly the same KV
  // prefix coverage. Equality with the append transaction's Q layout is wrong.
  const auto reader_layout = schema.version({old_q, next_ptr, next_ids, tails}).value();
  auto keys = c::PagedArena::create(schema, c::BatchCacheRole::kKey,
      storage.ref(), {.max_backing_bytes = 8192}).value();
  auto values = c::PagedArena::create(schema, c::BatchCacheRole::kValue,
      storage.ref(), {.max_backing_bytes = 8192}).value();
  std::array<c::ReadLease, 4> old_keys, old_values;
  std::array<c::PagedPageAppender, 1> key_tail, value_tail;
  std::array<std::array<std::uint16_t, 128>, 4> old_value_bytes{};
  constexpr std::array<float, 6> history{.125F, .25F, .375F, .5F, .625F, .75F};
  for (std::uint32_t page = 0; page < 4; ++page) {
    auto key = keys.acquire(page).value();
    auto value = values.acquire(page).value();
    ASSERT_TRUE(key.lease().view()->fill(std::byte{0}).ok());
    auto &words = old_value_bytes[page];
    const auto first = page == 0 ? history[0] : page == 1 ? history[2] : page == 2 ? history[3] : history[5];
    const auto last = page == 0 ? history[1] : history[4];
    std::fill_n(words.begin(), 64, bf16(first));
    std::fill(words.begin() + 64, words.end(), page % 2 ? std::uint16_t{0x7fc1} : bf16(last));
    ASSERT_TRUE(value.lease().view()->write(std::as_bytes(std::span(words))).ok());
    if (page == 1) {
      auto k = key.publish_appendable(1).value(), v = value.publish_appendable(1).value();
      old_keys[page] = std::move(k.prefix); key_tail[0] = std::move(k.appender);
      old_values[page] = std::move(v.prefix); value_tail[0] = std::move(v.appender);
    } else {
      // Page 3's physical second row is initialized NaN, although the old
      // layout authorizes only its first row. COW must not change either byte.
      old_keys[page] = key.publish(2).value();
      old_values[page] = value.publish(2).value();
    }
  }
  const auto previous_keys = keys.read(previous_layout, old_keys).value();
  const auto previous_values = values.read(previous_layout, old_values).value();
  auto working = c::PagedArenaFrame::fork(c::PagedArenaFootprint::from_read(previous_keys).value()).value();
  const auto expected = working.view().value();
  ASSERT_TRUE(key_tail[0].reserve_rows(1).ok());
  ASSERT_TRUE(value_tail[0].reserve_rows(1).ok());
  std::array<c::PagedPageWriter, 3> fresh_keys, fresh_values;
  constexpr std::array<std::uint32_t, 3> fresh_ids{4, 5, 6};
  for (std::size_t i = 0; i < fresh_ids.size(); ++i) {
    fresh_keys[i] = keys.acquire(fresh_ids[i]).value();
    fresh_values[i] = values.acquire(fresh_ids[i]).value();
    ASSERT_TRUE(fresh_keys[i].lease().view()->fill(std::byte{0x55}).ok());
    ASSERT_TRUE(fresh_values[i].lease().view()->fill(std::byte{0x55}).ok());
  }
  auto key_pages = gb::PagedAppendPages::create(keys, produced_layout, previous_keys,
      fresh_keys, key_tail).value();
  auto value_pages = gb::PagedAppendPages::create(values, produced_layout, previous_values,
      fresh_values, value_tail).value();
  const auto rows = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{4, 1, 64}).value();
  const auto query_desc = c::batch_query_description(capacity, 2).value();
  const auto output_desc = c::batch_output_description(capacity, 2).value();
  const auto key_desc = c::batch_cache_description(schema, c::BatchCacheRole::kKey).value();
  const auto value_desc = c::batch_cache_description(schema, c::BatchCacheRole::kValue).value();
  auto payload = c::LeasePool::create(storage.ref(), 8192, {.max_blocks = 4}).value();
  auto new_key = payload.allocate(rows.required_bytes().value(), 256).value();
  auto new_value = payload.allocate(rows.required_bytes().value(), 256).value();
  auto query = payload.allocate(query_desc.required_bytes().value(), 256).value();
  auto destination = payload.allocate(output_desc.required_bytes().value(), 256).value();
  ASSERT_TRUE(new_key.view()->fill(std::byte{0}).ok());
  ASSERT_TRUE(query.view()->fill(std::byte{0}).ok());
  std::array<std::uint16_t, 256> incoming{};
  constexpr std::array<float, 4> appended{.5F, 1.25F, 1.F, 2.125F};
  for (std::size_t row = 0; row < appended.size(); ++row)
    std::fill_n(incoming.begin() + row * 64, 64, bf16(appended[row]));
  ASSERT_TRUE(new_value.view()->write(std::as_bytes(std::span(incoming))).ok());
  const auto key_input = new_key.publish().value(), value_input = new_value.publish().value();
  const auto query_input = query.publish().value();

  auto first = g::GraphBuilder::create().value();
  const auto first_key = first.input(rows).value(), first_value = first.input(rows).value();
  const auto append_key = first.add_node(c::Operation::custom(
      gb::paged_append(schema, c::BatchCacheRole::kKey).value()), std::array{first_key}).value();
  const auto append_value = first.add_node(c::Operation::custom(
      gb::paged_append(schema, c::BatchCacheRole::kValue).value()), std::array{first_value}).value();
  ASSERT_TRUE(first.depends_on(append_value, append_key).ok());
  const auto first_graph = std::move(first).finish().value();
  auto first_plan = g::cuda::compile(first_graph, context, storage).value();
  auto producer = first_plan.executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const gb::PrepareOptions options{.host_workspace = staging.ref(),
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10};
  ASSERT_TRUE(gb::prepare(producer, queues, {}, options).ok());
  ASSERT_TRUE(gb::bind_append(producer, first_graph, append_key, std::move(key_pages)).ok());
  ASSERT_TRUE(gb::bind_append(producer, first_graph, append_value, std::move(value_pages)).ok());
  ASSERT_TRUE(producer.bind(first_key, rows, key_input).ok());
  ASSERT_TRUE(producer.bind(first_value, rows, value_input).ok());
  auto key_source_result = gbd::PagedAppendSource::prepare(producer, append_key, storage);
  ASSERT_TRUE(key_source_result) << key_source_result.error().message();
  auto value_source_result = gbd::PagedAppendSource::prepare(producer, append_value, storage);
  ASSERT_TRUE(value_source_result) << value_source_result.error().message();
  auto key_source = std::move(*key_source_result), value_source = std::move(*value_source_result);

  auto second = g::GraphBuilder::create().value();
  const auto second_query = second.input(query_desc).value();
  const auto second_key = second.input(key_desc).value(), second_value = second.input(value_desc).value();
  g::NodeId attention;
  if (owned_layout) {
    const auto operation = gb::paged_attention(reader_layout,
        {.scale = .125F, .causal = true}).value();
    attention = second.add_node(c::Operation::custom(operation),
        std::array{second_query, second_key, second_value}).value();
  } else {
    const auto metadata = second.input(schema.metadata_description().value()).value();
    attention = second.add_node(c::Operation::custom(c::PagedAttention{
        .capacity = capacity, .scale = .125F, .causal = true}),
        std::array{second_query, second_key, second_value, metadata}).value();
  }
  const auto output = second.result(attention).value();
  ASSERT_TRUE(second.publish(output).ok());
  const auto second_graph = std::move(second).finish().value();
  auto second_plan_result = g::cuda::compile(second_graph, context, storage);
  ASSERT_TRUE(second_plan_result) << second_plan_result.error().message();
  auto second_plan = std::move(*second_plan_result);
  ASSERT_EQ(second_plan.report.region_count(), 1u);
  ASSERT_EQ(second_plan.report.step_count(), 1u);
  auto consumer = second_plan.executable.create_invocation().value();
  const auto prepared = gb::prepare(consumer, queues, std::array{gb::NodeLayout{attention, reader_layout}}, options);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  ASSERT_TRUE(consumer.bind(second_query, query_desc, query_input).ok());
  ASSERT_TRUE(consumer.bind(output, output_desc, std::move(destination)).ok());
  // These original capacity addresses are borrowed for metadata/capture only.
  // No work is submitted and no initialized-page permission is invented.
  const auto key_capacity = cd::PagedArenaAccess::tensor(keys).value();
  const auto value_capacity = cd::PagedArenaAccess::tensor(values).value();
  ASSERT_TRUE(consumer.bind(second_key, key_capacity).ok());
  ASSERT_TRUE(consumer.bind(second_value, value_capacity).ok());
  const auto warm = consumer.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
  ASSERT_TRUE(warm.ok()) << warm.message();
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
  const auto warm_report = consumer.replay_report().value();
  ASSERT_EQ(warm_report.recorded_capture_range_count, 1u);
  auto key_binding = gbd::reader_binding(consumer, second_key, key_source, reader_layout);
  ASSERT_TRUE(key_binding) << key_binding.error().message();
  auto value_binding = gbd::reader_binding(consumer, second_value, value_source, reader_layout);
  ASSERT_TRUE(value_binding) << value_binding.error().message();
  if (owned_layout) {
    // This node really has only Q/K/V ports. The ready proof belongs to its
    // original private control, not an invented external metadata input.
    ASSERT_EQ(key_binding->ready_inputs.size(), 1u);
    ASSERT_EQ(value_binding->ready_inputs.size(), 1u);
    const auto &control = key_binding->ready_inputs.front();
    EXPECT_FALSE(control.input.valid());
    EXPECT_EQ(control.region, attention);
    EXPECT_TRUE(control.read.valid());
    EXPECT_EQ(control.tensor.description(), schema.metadata_description().value());
  }
  const std::array bindings{*key_binding, *value_binding};
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  PageSourceGate gate, reader_gate;
  gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &PageSourceGate::wait, &gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = gd::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  EXPECT_EQ(*common.poll(), c::CompletionPoll::kPending);
  const auto captured_keys = key_source.capture(common);
  ASSERT_TRUE(captured_keys.ok()) << captured_keys.message();
  const auto captured_values = value_source.capture(common);
  ASSERT_TRUE(captured_values.ok()) << captured_values.message();
  if (owned_layout) {
    auto invalid_control = value_binding->ready_inputs.front();
    invalid_control.ordinal = UINT32_MAX;
    const std::array invalid_ready{invalid_control};
    auto invalid_bindings = bindings;
    invalid_bindings.back().ready_inputs = invalid_ready;
    // Reject the final private control before committing any of this bundle.
    // The original complete bundle below must still bind both source views.
    EXPECT_FALSE(gd::bind_outputs(consumer, invalid_bindings).ok());
  }
  const auto bound = gd::bind_outputs(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto replay = gd::prepare_outputs(consumer, bindings, queues);
  ASSERT_TRUE(replay.ok()) << replay.message();
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
  const auto replay_report = consumer.replay_report().value();
  EXPECT_EQ(replay_report.recorded_capture_range_count, 0u);
  EXPECT_EQ(replay_report.reused_capture_range_count, 1u);
  const auto dependency = consumer.input_source_report().value();
  EXPECT_EQ(dependency.source_count, 1u);
  EXPECT_EQ(dependency.device_wait_count, 1u);
  EXPECT_EQ(dependency.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  reader_gate.arm(gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &PageSourceGate::wait, &reader_gate), cudaSuccess);
  auto consumed = gd::submit_outputs(consumer, bindings, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  gate.release();
  ASSERT_TRUE(common.wait().ok());
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  ASSERT_TRUE(common.detach().ok());
  std::array<gb::PagedAppendPages, 2> returned;
  const auto taken = gb::take_appends(producer, std::array{append_key, append_value}, returned);
  ASSERT_TRUE(taken.ok()) << taken.message();
  auto delta = key_source.ready_delta(returned[0], expected, produced_layout);
  ASSERT_TRUE(delta) << delta.error().message();
  auto commit = working.prepare_commit(std::move(*delta));
  ASSERT_TRUE(commit) << commit.error().message();
  ASSERT_TRUE(working.commit(std::move(*commit)).ok());
  auto next_values_result = value_source.ready(returned[1]);
  ASSERT_TRUE(next_values_result) << next_values_result.error().message();
  const auto next_values = std::move(*next_values_result);
  const auto retry_values = value_source.ready(returned[1]);
  ASSERT_TRUE(retry_values) << retry_values.error().message();
  EXPECT_EQ(next_values.layout(), produced_layout);
  EXPECT_EQ(next_values.page_count(), 6u);
  EXPECT_EQ(working.layout(), produced_layout);
  EXPECT_EQ(previous_keys.layout(), previous_layout);
  EXPECT_EQ(previous_values.layout(), previous_layout);
  EXPECT_FALSE(keys.acquire(4));
  EXPECT_FALSE(values.acquire(4));
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  // Returned tails remain original epoch owners, but an admitted reader
  // prevents a new append reservation even after successful publication.
  ASSERT_EQ(returned[1].appender_count(), 3u);
  ASSERT_TRUE(returned[1].appender(1)->valid());
  EXPECT_TRUE(returned[1].appender(1)->reserve_rows(1).IsUnavailable());
  reader_gate.release();
  ASSERT_TRUE(consumed->wait().ok());
  *consumed = {};
  {
    auto result = consumer.result(output).value();
    auto read = result.storage_lease().value();
    std::array<std::uint16_t, 256> actual{};
    ASSERT_TRUE(read.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], bf16(i < 128 ? .5F : 1.F)) << i;
  }
  constexpr std::array<std::uint32_t, 6> pages{0, 1, 2, 4, 5, 6};
  constexpr std::array<std::array<float, 2>, 6> expected_values{{
      {.125F, .25F}, {.375F, .5F}, {.5F, .625F}, {1.25F, 0}, {.75F, 1.F}, {2.125F, 0}}};
  for (std::size_t i = 0; i < pages.size(); ++i) {
    auto read = next_values.page_prefix(pages[i]).value();
    const auto rows_in_page = pages[i] == 4 || pages[i] == 6 ? 1u : 2u;
    std::array<std::uint16_t, 128> actual{};
    ASSERT_TRUE(read.view()->read(std::as_writable_bytes(std::span(actual).first(rows_in_page * 64))).ok());
    for (std::size_t element = 0; element < rows_in_page * 64; ++element)
      EXPECT_EQ(actual[element], bf16(expected_values[i][element / 64]));
  }
  for (std::size_t page = 0; page < old_values.size(); ++page) {
    const auto count = page == 1 ? 64u : 128u;
    std::array<std::uint16_t, 128> actual{};
    ASSERT_TRUE(old_values[page].view()->read(std::as_writable_bytes(std::span(actual).first(count))).ok());
    EXPECT_TRUE(std::equal(actual.begin(), actual.begin() + count, old_value_bytes[page].begin()));
  }
  if (owned_layout) {
    // No native work or GraphOutput is left to make this failure vacuous.
    // The same layout remains preparable. A different exact version with
    // changed page IDs must not overwrite the private mirror still retained
    // by these typed source views, even though its address/geometry is stable.
    const auto mirror_result = key_binding->ready_inputs.front().read.view();
    ASSERT_TRUE(mirror_result) << mirror_result.error().message();
    const auto mirror = *mirror_result;
    std::vector<std::byte> before(mirror.size()), after(mirror.size());
    ASSERT_TRUE(mirror.read(before).ok());
    const auto same = gb::prepare(consumer, queues,
        std::array{gb::NodeLayout{attention, reader_layout}}, options);
    ASSERT_TRUE(same.ok()) << same.message();
    const std::array<std::int32_t, 6> changed_ids{2, 5, 6, 0, 1, 4};
    const auto changed_layout_result = schema.version({old_q, next_ptr, changed_ids, tails});
    ASSERT_TRUE(changed_layout_result) << changed_layout_result.error().message();
    const auto rewrite = gb::prepare(consumer, queues,
        std::array{gb::NodeLayout{attention, *changed_layout_result}}, options);
    EXPECT_TRUE(rewrite.IsUnavailable()) << rewrite.message();
    ASSERT_TRUE(mirror.read(after).ok());
    EXPECT_EQ(after, before);
  }
  EXPECT_FALSE(gate.timed_out.load());
  EXPECT_FALSE(reader_gate.timed_out.load());
}

TEST(GraphPagedAppendSourceCuda, PendingOriginalPagesPublishWhileAttentionReads) {
  pending_original_pages_publish_while_attention_reads(false);
}

TEST(GraphPagedAppendSourceCuda, OwnedLayoutMirrorPreservesPendingPageSource) {
  pending_original_pages_publish_while_attention_reads(true);
}
} // namespace
