#include <gtest/gtest.h>

#include "../../engine/inference/submission_views.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/paged_frame.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/paged_append_source.h"
#include "hyle/graph/detail/source_binding_access.h"
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
namespace i = hyle::inference;

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
  i::SubmissionViews views;
  const std::array page_captures{i::SubmissionViews::PageCapture{append_key, storage},
      i::SubmissionViews::PageCapture{append_value, storage}};
  const auto prepared_views = views.prepare(producer,
      std::span<const i::SubmissionViews::StoreCapture>{}, page_captures, {});
  ASSERT_TRUE(prepared_views.ok()) << prepared_views.message();
  ASSERT_EQ(views.pages.size(), 2u);
  auto &key_source = views.pages[0], &value_source = views.pages[1];

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
  auto key_binding = gb::reader_binding(consumer, second_key, key_source, reader_layout);
  ASSERT_TRUE(key_binding) << key_binding.error().message();
  auto value_binding = gb::reader_binding(consumer, second_value, value_source, reader_layout);
  ASSERT_TRUE(value_binding) << value_binding.error().message();
  const auto &key_binding_data = gd::SourceBindingAccess::get(*key_binding);
  const auto &value_binding_data = gd::SourceBindingAccess::get(*value_binding);
  if (owned_layout) {
    // This node really has only Q/K/V ports. The ready proof belongs to its
    // original private control, not an invented external metadata input.
    ASSERT_EQ(key_binding_data.ready_inputs.size(), 1u);
    ASSERT_EQ(value_binding_data.ready_inputs.size(), 1u);
    const auto &control = key_binding_data.ready_inputs.front();
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
  const auto captured = views.capture_outputs(producer, *submitted,
      std::span<const i::SubmissionViews::OutputCapture>{}, storage);
  ASSERT_TRUE(captured) << captured.error().message();
  EXPECT_TRUE(*captured);
  EXPECT_FALSE(submitted->valid());
  EXPECT_FALSE(views.primary_output.valid());
  EXPECT_TRUE(views.extra_outputs.empty());
  auto &common = views.submission;
  EXPECT_EQ(*common.poll(), c::CompletionPoll::kPending);
  if (owned_layout) {
    auto invalid_control = gd::SourceBindingAccess::get(*value_binding).ready_inputs.front();
    invalid_control.ordinal = UINT32_MAX;
    const std::array invalid_ready{invalid_control};
    auto invalid_bindings = bindings;
    gd::SourceBindingAccess::get(invalid_bindings.back()).ready_inputs = invalid_ready;
    // Reject the final private control before committing any of this bundle.
    // The original complete bundle below must still bind both source views.
    EXPECT_FALSE(g::bind_sources(consumer, invalid_bindings).ok());
  }
  const auto bound = g::bind_sources(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto replay = g::prepare_sources(consumer, queues);
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
  auto consumed = g::submit_sources(consumer, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  gate.release();
  ASSERT_TRUE(common.wait().ok());
  EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  ASSERT_TRUE(views.detach_outputs().ok());
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
    const auto mirror_result = key_binding_data.ready_inputs.front().read.view();
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

void pending_append_inherits_published_pages(bool full_page_boundary, bool relay_to_attention = false,
    bool local_attention = false) {
  SCOPED_TRACE(full_page_boundary ? "full page boundary" : "partial page COW");
  SCOPED_TRACE(relay_to_attention ? "A to B to C" : "A to B");
  SCOPED_TRACE(local_attention ? "B appends and attends" : "B appends only");
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
      .max_requests = 1, .max_query_rows = 3, .max_kv_rows = 7,
      .max_pages = 4, .max_page_refs = 4,
      .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
      .value_head_dim = 64, .page_size = 2};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::int32_t a_count = full_page_boundary ? 3 : 2;
  const std::array<std::int32_t, 2> one_query{0, 1}, a_queries{0, a_count};
  const std::array<std::int32_t, 2> old_ptr{0, 2}, a_ptr{0, 3};
  const std::array<std::int32_t, 2> b_ptr{0, full_page_boundary ? 4 : 3};
  const std::array<std::int32_t, 2> old_ids{0, 1};
  const std::array<std::int32_t, 3> a_ids{0, 1, 2}, cow_ids{0, 1, 3};
  const std::array<std::int32_t, 4> boundary_ids{0, 1, 2, 3};
  const std::span<const std::int32_t> b_ids = full_page_boundary
      ? std::span<const std::int32_t>(boundary_ids) : std::span<const std::int32_t>(cow_ids);
  const std::array<std::int32_t, 1> one_tail{1}, a_tail_rows{full_page_boundary ? 2 : 1};
  const std::array<std::int32_t, 1> b_tail_rows{full_page_boundary ? 1 : 2};
  const auto old_layout = schema.version({one_query, old_ptr, old_ids, one_tail}).value();
  const auto a_layout = schema.version({a_queries, a_ptr, a_ids, a_tail_rows}).value();
  const auto b_layout = schema.version({one_query, b_ptr, b_ids, b_tail_rows}).value();
  auto arena = c::PagedArena::create(schema, c::BatchCacheRole::kValue,
      storage.ref(), {.max_backing_bytes = 4096}).value();
  auto old_full = arena.acquire(0).value(), old_tail = arena.acquire(1).value();
  std::array<std::uint16_t, 128> old_full_bytes{}, old_tail_bytes{};
  std::fill_n(old_full_bytes.begin(), 64, bf16(.125F));
  std::fill(old_full_bytes.begin() + 64, old_full_bytes.end(), bf16(.25F));
  std::fill_n(old_tail_bytes.begin(), 64, bf16(.375F));
  std::fill(old_tail_bytes.begin() + 64, old_tail_bytes.end(), std::uint16_t{0x7fc1});
  ASSERT_TRUE(old_full.lease().view()->write(std::as_bytes(std::span(old_full_bytes))).ok());
  ASSERT_TRUE(old_tail.lease().view()->write(std::as_bytes(std::span(old_tail_bytes))).ok());
  const auto old_full_ready = old_full.publish(2).value();
  auto tail_publication = old_tail.publish_appendable(1).value();
  const auto old_tail_ready = std::move(tail_publication.prefix);
  const std::array old_guards{old_full_ready, old_tail_ready};
  const auto old_frame = arena.read(old_layout, old_guards).value();
  std::array<c::PagedPageAppender, 1> tails{std::move(tail_publication.appender)};
  ASSERT_TRUE(tails.front().reserve_rows(1).ok());
  std::array<c::PagedPageWriter, 1> a_dirty{arena.acquire(2).value()};
  std::array<c::PagedPageWriter, 1> b_dirty{arena.acquire(3).value()};
  ASSERT_TRUE(a_dirty.front().lease().view()->fill(std::byte{0x55}).ok());
  ASSERT_TRUE(b_dirty.front().lease().view()->fill(std::byte{0x66}).ok());
  auto a_pages = gb::PagedAppendPages::create(arena, a_layout, old_frame, a_dirty, tails).value();
  const auto a_rows = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{a_count, 1, 64}).value();
  const auto b_rows = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{1, 1, 64}).value();
  auto payload = c::LeasePool::create(storage.ref(), 1024, {.max_blocks = 2}).value();
  auto a_input_writer = payload.allocate(a_rows.required_bytes().value(), 256).value();
  auto b_input_writer = payload.allocate(b_rows.required_bytes().value(), 256).value();
  std::array<std::uint16_t, 192> a_values{};
  std::array<std::uint16_t, 64> b_values{};
  std::fill_n(a_values.begin(), 64, bf16(.5F));
  std::fill_n(a_values.begin() + 64, 64, bf16(.625F));
  std::fill(a_values.begin() + 128, a_values.end(), bf16(.75F));
  std::fill(b_values.begin(), b_values.end(), bf16(full_page_boundary ? .875F : .75F));
  ASSERT_TRUE(a_input_writer.view()->write(
      std::as_bytes(std::span(a_values).first(std::size_t(a_count) * 64))).ok());
  ASSERT_TRUE(b_input_writer.view()->write(std::as_bytes(std::span(b_values))).ok());
  const auto a_input_ready = a_input_writer.publish().value();
  const auto b_input_ready = b_input_writer.publish().value();

  auto a_builder = g::GraphBuilder::create().value();
  const auto a_input = a_builder.input(a_rows).value();
  const auto a_append = a_builder.add_node(c::Operation::custom(
      gb::paged_append(schema, c::BatchCacheRole::kValue).value()), std::array{a_input}).value();
  const auto a_graph = std::move(a_builder).finish().value();
  auto a_plan = g::cuda::compile(a_graph, context, storage).value();
  auto producer = a_plan.executable.create_invocation().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const gb::PrepareOptions options{.host_workspace = staging.ref()};
  ASSERT_TRUE(gb::prepare(producer, queues, {}, options).ok());
  ASSERT_TRUE(gb::bind_append(producer, a_graph, a_append, std::move(a_pages)).ok());
  ASSERT_TRUE(producer.bind(a_input, a_rows, a_input_ready).ok());
  auto source_result = gb::PagedAppendSource::prepare(producer, a_append, storage);
  ASSERT_TRUE(source_result) << source_result.error().message();
  auto source = std::move(*source_result);
  // Cold geometry only: B owns a distinct fresh destination. It receives no
  // fake ready lease for A's pending tail and never borrows A's tail authority.
  auto b_pages_result = gb::make_append_from_source(arena, b_layout, source, b_dirty);
  ASSERT_TRUE(b_pages_result) << b_pages_result.error().message();
  auto b_builder = g::GraphBuilder::create().value();
  const auto b_input = b_builder.input(b_rows).value();
  const auto b_append = b_builder.add_node(c::Operation::custom(
      gb::paged_append(schema, c::BatchCacheRole::kValue).value()), std::array{b_input}).value();
  g::ValueId b_query{}, b_key{}, b_value{}, b_output{};
  g::NodeId b_attention{};
  if (local_attention) {
    b_query = b_builder.input(c::batch_query_description(capacity, 1).value()).value();
    b_key = b_builder.input(c::batch_cache_description(schema, c::BatchCacheRole::kKey).value()).value();
    b_value = b_builder.input(c::batch_cache_description(schema, c::BatchCacheRole::kValue).value()).value();
    const auto metadata = b_builder.input(schema.metadata_description().value()).value();
    b_attention = b_builder.add_node(c::Operation::custom(c::PagedAttention{
        .capacity = capacity, .scale = .125F, .causal = true}),
        std::array{b_query, b_key, b_value, metadata}).value();
    ASSERT_TRUE(b_builder.depends_on(b_attention, b_append).ok());
    b_output = b_builder.result(b_attention).value();
    ASSERT_TRUE(b_builder.publish(b_output).ok());
  }
  const auto b_graph = std::move(b_builder).finish().value();
  auto b_plan = g::cuda::compile(b_graph, context, storage).value();
  auto consumer = b_plan.executable.create_invocation().value();
  const std::array b_layouts{gb::NodeLayout{b_attention, b_layout}};
  ASSERT_TRUE(gb::prepare(consumer, queues,
      local_attention ? std::span<const gb::NodeLayout>(b_layouts) : std::span<const gb::NodeLayout>{}, options).ok());
  const auto bound_pages = local_attention
      ? gb::bind_append(consumer, b_graph, b_append, b_value, std::move(*b_pages_result))
      : gb::bind_append(consumer, b_graph, b_append, std::move(*b_pages_result));
  ASSERT_TRUE(bound_pages.ok()) << bound_pages.message();
  ASSERT_TRUE(consumer.bind(b_input, b_rows, b_input_ready).ok());
  c::PagedArena b_keys;
  c::LeasePool b_payload;
  auto b_route = g::ReplayRoute::kCommands;
  if (local_attention) {
    b_keys = c::PagedArena::create(schema, c::BatchCacheRole::kKey,
        storage.ref(), {.max_backing_bytes = 4096}).value();
    std::array<c::ReadLease, 4> guards;
    for (std::size_t page = 0; page < b_ids.size(); ++page) {
      auto writer = b_keys.acquire(std::uint32_t(b_ids[page])).value();
      ASSERT_TRUE(writer.lease().view()->fill(std::byte{0}).ok());
      guards[page] = writer.publish(page + 1 == b_ids.size() ? std::uint32_t(b_tail_rows.front()) : 2u).value();
    }
    auto frame = b_keys.read(b_layout, std::span(guards).first(b_ids.size())).value();
    const auto query_description = c::batch_query_description(capacity, 1).value();
    const auto output_description = c::batch_output_description(capacity, 1).value();
    b_payload = c::LeasePool::create(storage.ref(), 1024, {.max_blocks = 2}).value();
    auto query = b_payload.allocate(query_description.required_bytes().value(), 256).value();
    auto destination = b_payload.allocate(output_description.required_bytes().value(), 256).value();
    ASSERT_TRUE(query.view()->fill(std::byte{0}).ok());
    ASSERT_TRUE(gb::bind(consumer, b_key, std::move(frame)).ok());
    ASSERT_TRUE(consumer.bind(b_query, query_description, query.publish().value()).ok());
    ASSERT_TRUE(consumer.bind(b_output, output_description, std::move(destination)).ok());
    // Record only the prepared tuple before A is accepted. No kernel executes
    // and no ready value is invented for either A's or B's future page rows.
    const auto warm = consumer.prepare_replay(queues,
        {.mode = g::ReplayMode::kPreferCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(warm.ok()) << warm.message();
    b_route = consumer.replay_route();
    ASSERT_TRUE(b_route == g::ReplayRoute::kCaptured || b_route == g::ReplayRoute::kHybrid);
    ASSERT_GT(consumer.replay_report()->recorded_capture_range_count, 0u);
  }
  gb::PagedAppendSource b_source;
  c::PagedArena third_keys;
  c::LeasePool third_payload;
  std::array<c::ReadLease, 4> third_foreign_guards;
  g::Invocation third;
  g::ValueId third_value{}, third_output{};
  if (relay_to_attention) {
    auto b_source_result = gb::PagedAppendSource::prepare(consumer, b_append, storage);
    ASSERT_TRUE(b_source_result) << b_source_result.error().message();
    b_source = std::move(*b_source_result);
    third_keys = c::PagedArena::create(schema, c::BatchCacheRole::kKey,
        storage.ref(), {.max_backing_bytes = 4096}).value();
    for (std::size_t page = 0; page < b_ids.size(); ++page) {
      auto writer = third_keys.acquire(std::uint32_t(b_ids[page])).value();
      ASSERT_TRUE(writer.lease().view()->fill(std::byte{0}).ok());
      third_foreign_guards[page] = writer.publish(page + 1 == b_ids.size()
          ? std::uint32_t(b_tail_rows.front()) : 2u).value();
    }
    auto key_frame = third_keys.read(b_layout,
        std::span(third_foreign_guards).first(b_ids.size())).value();
    for (std::size_t page = 0; page < b_ids.size(); ++page)
      third_foreign_guards[page] = key_frame.page_prefix(std::uint32_t(b_ids[page])).value();
    const auto query_description = c::batch_query_description(capacity, 1).value();
    const auto output_description = c::batch_output_description(capacity, 1).value();
    third_payload = c::LeasePool::create(storage.ref(), 1024, {.max_blocks = 2}).value();
    auto query_writer = third_payload.allocate(query_description.required_bytes().value(), 256).value();
    auto destination = third_payload.allocate(output_description.required_bytes().value(), 256).value();
    ASSERT_TRUE(query_writer.view()->fill(std::byte{0}).ok());
    const auto query_ready = query_writer.publish().value();
    auto third_builder = g::GraphBuilder::create().value();
    const auto query = third_builder.input(query_description).value();
    const auto key = third_builder.input(c::batch_cache_description(schema, c::BatchCacheRole::kKey).value()).value();
    third_value = third_builder.input(c::batch_cache_description(schema, c::BatchCacheRole::kValue).value()).value();
    const auto metadata = third_builder.input(schema.metadata_description().value()).value();
    const auto attention = third_builder.add_node(c::Operation::custom(c::PagedAttention{
        .capacity = capacity, .scale = .125F, .causal = true}),
        std::array{query, key, third_value, metadata}).value();
    third_output = third_builder.result(attention).value();
    ASSERT_TRUE(third_builder.publish(third_output).ok());
    const auto third_graph = std::move(third_builder).finish().value();
    auto third_plan_result = g::cuda::compile(third_graph, context, storage);
    ASSERT_TRUE(third_plan_result) << third_plan_result.error().message();
    auto third_plan = std::move(*third_plan_result);
    ASSERT_EQ(third_plan.report.step_count(), 1u);
    third = third_plan.executable.create_invocation().value();
    const auto third_prepared = gb::prepare(third, queues,
        std::array{gb::NodeLayout{attention, b_layout}}, options);
    ASSERT_TRUE(third_prepared.ok()) << third_prepared.message();
    ASSERT_TRUE(gb::bind(third, key, std::move(key_frame)).ok());
    ASSERT_TRUE(third.bind(query, query_description, query_ready).ok());
    ASSERT_TRUE(third.bind(third_output, output_description, std::move(destination)).ok());
    // Record only against the real capacity address, without submitting it
    // or manufacturing ready values for any of B's future pages.
    ASSERT_TRUE(third.bind(third_value, cd::PagedArenaAccess::tensor(arena).value()).ok());
    const auto warm = third.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(warm.ok()) << warm.message();
    ASSERT_EQ(third.replay_report()->recorded_capture_range_count, 1u);
  }
  ASSERT_TRUE(producer.prepare_replay(queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());

  PageSourceGate producer_gate, consumer_gate, third_gate;
  producer_gate.arm(c::cuda::stream(queue).value());
  ASSERT_EQ(cudaLaunchHostFunc(producer_gate.stream, &PageSourceGate::wait, &producer_gate), cudaSuccess);
  auto submitted = producer.submit(queues);
  ASSERT_TRUE(submitted) << submitted.error().message();
  auto common_result = g::SubmissionSource::create(producer, std::move(*submitted), storage);
  ASSERT_TRUE(common_result) << common_result.error().message();
  auto common = std::move(*common_result);
  ASSERT_EQ(*common.poll(), c::CompletionPoll::kPending);
  const auto captured = source.capture(common);
  ASSERT_TRUE(captured.ok()) << captured.message();
  auto binding_result = gb::append_binding(consumer, b_append, source);
  ASSERT_TRUE(binding_result) << binding_result.error().message();
  const auto &binding_data = gd::SourceBindingAccess::get(*binding_result);
  // There is no pending tensor input in this bundle: the original append's
  // The selected effect consumes the same submission terminal even when a
  // full-page boundary requires no COW read of any pending payload.
  EXPECT_FALSE(binding_data.input.valid());
  const std::array bindings{*binding_result};
  ASSERT_NE(binding_data.resource, nullptr);
  if (local_attention) {
    // B's selected attention consumes every inherited prefix, including the
    // full-page/no-COW case. One exact range per physical predecessor page is
    // also one original ordered native read; new page 3 uses B's writer.
    ASSERT_EQ(binding_data.effect_reads.size(), 3u);
    const auto bank = cd::PagedArenaAccess::tensor(arena).value().storage();
    for (std::size_t page = 0; page < 3; ++page) {
      const auto range = binding_data.effect_reads[page];
      EXPECT_EQ(range.resource(), bank.resource());
      EXPECT_EQ(range.native_handle(), bank.native_handle());
      EXPECT_EQ(range.offset(), bank.offset() + page * arena.page_bytes());
      EXPECT_EQ(range.size(), page == 2 && !full_page_boundary ? 128u : 256u);
      EXPECT_TRUE(binding_data.resource->covers_range(range).ok());
    }
  } else if (full_page_boundary) {
    EXPECT_TRUE(binding_data.effect_reads.empty());
  } else {
    ASSERT_EQ(binding_data.effect_reads.size(), 1u);
    const auto prefix = binding_data.effect_reads.front();
    ASSERT_EQ(prefix.size(), 128u);
    ASSERT_LT(prefix.size(), arena.page_bytes());
    const auto capacity_view = cd::PagedArenaAccess::tensor(arena).value().storage();
    ASSERT_EQ(prefix.resource(), capacity_view.resource());
    ASSERT_EQ(prefix.native_handle(), capacity_view.native_handle());
    ASSERT_GE(prefix.offset(), capacity_view.offset());
    const auto overread = capacity_view.subview(prefix.offset() - capacity_view.offset(), prefix.size() + 1);
    ASSERT_TRUE(overread) << overread.error().message();
    const auto covered = binding_data.resource->covers_range(prefix);
    ASSERT_TRUE(covered.ok()) << covered.message();
    const auto uncovered = binding_data.resource->covers_range(*overread);
    EXPECT_TRUE(uncovered.IsInvalidArgument()) << uncovered.message();
    const std::array invalid_reads{*overread};
    auto invalid_binding = *binding_result;
    gd::SourceBindingAccess::get(invalid_binding).effect_reads = invalid_reads;
    const std::array invalid_bindings{invalid_binding};
    // This byte is inside A's original full page but outside its genuinely
    // written prefix. The source-range query above checks initialized coverage;
    // this separate registration rejection must leave the real bundle intact.
    const auto rejected = g::bind_sources(consumer, invalid_bindings);
    EXPECT_TRUE(rejected.IsInvalidArgument()) << rejected.message();
  }
  const auto bound = g::bind_sources(consumer, bindings);
  ASSERT_TRUE(bound.ok()) << bound.message();
  const auto prepared = g::prepare_sources(consumer, queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  EXPECT_EQ(consumer.replay_route(), b_route);
  if (local_attention) {
    EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 0u);
    EXPECT_GT(consumer.replay_report()->reused_capture_range_count, 0u);
  }
  const auto dependency = consumer.input_source_report().value();
  EXPECT_EQ(dependency.source_count, 1u);
  EXPECT_EQ(dependency.device_wait_count, 1u);
  EXPECT_EQ(dependency.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  consumer_gate.arm(producer_gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(consumer_gate.stream, &PageSourceGate::wait, &consumer_gate), cudaSuccess);
  auto consumed = g::submit_sources(consumer, queues);
  ASSERT_TRUE(consumed) << consumed.error().message();
  g::SubmissionSource b_common;
  std::array<g::SourceBinding, 1> third_bindings;
  if (relay_to_attention) {
    auto b_common_result = g::SubmissionSource::create(consumer, std::move(*consumed), storage);
    ASSERT_TRUE(b_common_result) << b_common_result.error().message();
    b_common = std::move(*b_common_result);
    ASSERT_EQ(*b_common.poll(), c::CompletionPoll::kPending);
    ASSERT_EQ(*common.poll(), c::CompletionPoll::kPending);
    const auto captured_b = b_source.capture(b_common);
    ASSERT_TRUE(captured_b.ok()) << captured_b.message();
    auto third_binding = gb::reader_binding(third, third_value, b_source, b_layout);
    ASSERT_TRUE(third_binding) << third_binding.error().message();
    const auto &third_data = gd::SourceBindingAccess::get(*third_binding);
    ASSERT_NE(third_data.resource, nullptr);
    const auto premature = third_data.resource->publish_ready({});
    EXPECT_TRUE(premature.IsUnavailable()) << premature.message();
    third_bindings.front() = *third_binding;
    const auto third_bound = g::bind_sources(third, third_bindings);
    ASSERT_TRUE(third_bound.ok()) << third_bound.message();
  }
  producer_gate.release();
  ASSERT_TRUE(common.wait().ok());
  if (relay_to_attention) ASSERT_EQ(*b_common.poll(), c::CompletionPoll::kPending);
  else ASSERT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  ASSERT_TRUE(common.detach().ok());
  auto returned_a_result = gb::take_append(producer, a_append);
  ASSERT_TRUE(returned_a_result) << returned_a_result.error().message();
  auto returned_a = std::move(*returned_a_result);
  auto a_frame_result = source.ready(returned_a);
  ASSERT_TRUE(a_frame_result) << a_frame_result.error().message();
  auto a_frame = std::move(*a_frame_result);
  ASSERT_EQ(a_frame.page_count(), 3u);
  c::PagedPageAppender a_tail;
  if (full_page_boundary) {
    ASSERT_EQ(returned_a.appender_count(), 1u);
    EXPECT_FALSE(returned_a.appender(0)->valid());
  } else {
    ASSERT_EQ(returned_a.appender_count(), 2u);
    a_tail = std::move(returned_a.appender(1).value());
    ASSERT_TRUE(a_tail.valid());
    EXPECT_EQ(a_tail.page(), 2u);
    EXPECT_EQ(a_tail.initialized_rows(), 1u);
    EXPECT_TRUE(a_tail.reserve_rows(1).IsUnavailable());
  }
  auto a_written_prefix = a_frame.page_prefix(2).value();
  // B's later untouched-page inheritance must use Source's actual ready
  // cache, not a pointer into the returned transaction or this local frame.
  returned_a = {};
  a_frame = {};
  c::Completion third_completion;
  if (relay_to_attention) {
    // A's original full page 2 has now become append-mode ready storage.
    // The boundary variant keeps that page in B, so C needs its origin's
    // actual ready guard rather than the now-ineligible full ticket. The
    // COW variant drops it and must retain only inherited pages 0 and 1.
    source = {};
    ASSERT_EQ(*b_common.poll(), c::CompletionPoll::kPending);
    const auto prepared_third = g::prepare_sources(third, queues);
    ASSERT_TRUE(prepared_third.ok()) << prepared_third.message();
    const auto replay = third.replay_report().value();
    EXPECT_EQ(replay.recorded_capture_range_count, 0u);
    EXPECT_EQ(replay.reused_capture_range_count, 1u);
    const auto third_dependency = third.input_source_report().value();
    EXPECT_EQ(third_dependency.source_count, 1u);
    EXPECT_EQ(third_dependency.device_wait_count, 1u);
    EXPECT_EQ(third_dependency.host_wait_count, 0u);
    EXPECT_FALSE(third.submit(queues));
    third_gate.arm(producer_gate.stream);
    ASSERT_EQ(cudaLaunchHostFunc(third_gate.stream, &PageSourceGate::wait, &third_gate), cudaSuccess);
    auto submitted_third = g::submit_sources(third, queues);
    ASSERT_TRUE(submitted_third) << submitted_third.error().message();
    third_completion = std::move(*submitted_third);
    ASSERT_EQ(*third_completion.poll(), c::CompletionPoll::kPending);
  } else EXPECT_EQ(*consumed->poll(), c::CompletionPoll::kPending);
  consumer_gate.release();
  if (relay_to_attention) {
    ASSERT_TRUE(b_common.wait().ok());
    ASSERT_TRUE(b_common.detach().ok());
    ASSERT_EQ(*third_completion.poll(), c::CompletionPoll::kPending);
    // Complete cardinality and each physical prefix length match B; only the
    // actual K roots/ranges are foreign to its V publication authority.
    const auto foreign = gd::SourceBindingAccess::get(third_bindings.front()).resource->publish_ready(
        std::span(third_foreign_guards).first(b_ids.size()));
    EXPECT_TRUE(foreign.IsInvalidArgument()) << foreign.message();
  } else {
    ASSERT_TRUE(consumed->wait().ok());
    *consumed = {};
  }
  if (local_attention) {
    auto result = consumer.result(b_output).value();
    auto ready = result.storage_lease().value();
    std::array<std::uint16_t, 64> actual{};
    ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    for (const auto word : actual) EXPECT_EQ(word, bf16(full_page_boundary ? .5F : .4375F));
  }
  auto returned_b = gb::take_append(consumer, b_append);
  ASSERT_TRUE(returned_b) << returned_b.error().message();
  auto final_result = relay_to_attention ? b_source.ready(*returned_b) : returned_b->ready();
  ASSERT_TRUE(final_result) << final_result.error().message();
  const auto final_frame = std::move(*final_result);
  EXPECT_EQ(final_frame.layout(), b_layout);
  ASSERT_EQ(final_frame.page_count(), b_ids.size());
  const std::size_t total_rows = full_page_boundary ? 7 : 6;
  constexpr std::array<float, 7> expected{.125F, .25F, .375F, .5F, .625F, .75F, .875F};
  if (relay_to_attention) {
    // Payload reads below may synchronize the resource's actual device.
    // Release the last Queue gate before inspecting any CUDA bytes on host.
    third_gate.release();
    ASSERT_TRUE(third_completion.wait().ok());
    third_completion = {};
    auto result = third.result(third_output).value();
    auto read = result.storage_lease().value();
    std::array<std::uint16_t, 64> actual{};
    ASSERT_TRUE(read.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    // Q/K are zero: C reads B's whole bank. The seven-row boundary variant
    // includes A's former full-writer page 2; the six-row COW variant does not.
    for (const auto word : actual)
      EXPECT_EQ(word, bf16(full_page_boundary ? .5F : .4375F));
  }
  for (std::size_t page = 0; page < b_ids.size(); ++page) {
    auto prefix = final_frame.page_prefix(std::uint32_t(b_ids[page])).value();
    std::array<std::uint16_t, 128> actual{};
    const auto words = 64 * std::min(std::size_t{2}, total_rows - 2 * page);
    ASSERT_TRUE(prefix.view()->read(std::as_writable_bytes(std::span(actual).first(words))).ok());
    for (std::size_t word = 0; word < words; ++word)
      EXPECT_EQ(actual[word], bf16(expected[2 * page + word / 64]));
  }
  EXPECT_EQ(final_frame.page_prefix(0)->generation(), old_full_ready.generation());
  std::array<std::uint16_t, 128> old_actual{};
  ASSERT_TRUE(old_full_ready.view()->read(std::as_writable_bytes(std::span(old_actual))).ok());
  EXPECT_EQ(old_actual, old_full_bytes);
  std::array<std::uint16_t, 64> old_prefix{};
  std::array<std::uint16_t, 128> a_prefix{};
  ASSERT_TRUE(old_tail_ready.view()->read(std::as_writable_bytes(std::span(old_prefix))).ok());
  const std::size_t a_words = full_page_boundary ? 128 : 64;
  ASSERT_TRUE(a_written_prefix.view()->read(std::as_writable_bytes(std::span(a_prefix).first(a_words))).ok());
  for (const auto word : old_prefix) EXPECT_EQ(word, bf16(.375F));
  for (std::size_t word = 0; word < a_words; ++word)
    EXPECT_EQ(a_prefix[word], bf16(word < 64 ? .625F : .75F));
  EXPECT_EQ(old_frame.layout(), old_layout);
  EXPECT_EQ(old_frame.page_prefix(1)->size(), 128u);
  if (!full_page_boundary) {
    ASSERT_TRUE(a_tail.reserve_rows(1).ok());
    ASSERT_TRUE(a_tail.rollback().ok());
    if (relay_to_attention) {
      // Both submissions and C's real read have retired. Drop every genuine
      // page-2 owner, the returned B transaction's old-prefix cache and B's
      // original InputSources. Keep B Source, C and final B snapshot alive:
      // shared origin metadata for pages 0/1 must not pin discarded page 2.
      a_tail = {};
      a_written_prefix.reset();
      *returned_b = {};
      const auto reset = consumer.reset();
      ASSERT_TRUE(reset.ok()) << reset.message();
      ASSERT_TRUE(b_source.valid());
      EXPECT_EQ(final_frame.layout(), b_layout);
      auto reclaimed = arena.acquire(2);
      ASSERT_TRUE(reclaimed) << reclaimed.error().message();
      EXPECT_EQ(reclaimed->page(), 2u);
      EXPECT_TRUE(reclaimed->lease().valid());
      for (const auto retained_page : std::array<std::uint32_t, 3>{0, 1, 3})
        EXPECT_FALSE(arena.acquire(retained_page)) << retained_page;
    }
  }
  EXPECT_FALSE(producer_gate.timed_out.load());
  EXPECT_FALSE(consumer_gate.timed_out.load());
  EXPECT_FALSE(third_gate.timed_out.load());
}

TEST(GraphPagedAppendSourceCuda, PendingAppendCopiesOriginalTailAndInheritsPublishedPages) {
  pending_append_inherits_published_pages(false);
}

TEST(GraphPagedAppendSourceCuda, PageBoundaryAppendRetainsSourceWithoutCowReads) {
  pending_append_inherits_published_pages(true);
}

TEST(GraphPagedAppendSourceCuda, ChainedAppendSourceReadsPublishedOriginAndPendingFreshPages) {
  pending_append_inherits_published_pages(true, true);
}

TEST(GraphPagedAppendSourceCuda, ChainedCowSourceReleasesUnrelatedPublishedPage) {
  pending_append_inherits_published_pages(false, true);
}

TEST(GraphPagedAppendSourceCuda, PendingPreviousAppendAndAttentionAdmitEveryInheritedPrefix) {
  pending_append_inherits_published_pages(true, false, true);
  pending_append_inherits_published_pages(false, false, true);
}
} // namespace
