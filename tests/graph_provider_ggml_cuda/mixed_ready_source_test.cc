#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/paged_append_source.h"
#include "hyle/graph/source.h"
#include "hyle/graph/detail/source_binding_access.h"
#include "hyle/graph/slots.h"
#include "hyle/util/scope_guard.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace g = hyle::graph;
namespace gd = g::detail;
namespace gb = g::batch;
namespace s = g::slots;

template<class T> T take(hyle::Result<T> value, const char *what) {
  if (!value) throw std::runtime_error(std::string(what) + ": " + std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status value, const char *what) {
  if (!value.ok()) throw std::runtime_error(std::string(what) + ": " + std::string(value.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())), "dense description");
}
struct Gate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false}, timed_out{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    static_cast<Gate *>(opaque)->released.wait(false, std::memory_order_acquire);
  }
  void arm(cudaStream_t value) {
    stream = value;
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10),
              [this] { return released.load(std::memory_order_acquire); })) {
        timed_out.store(true, std::memory_order_release); release();
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

// Actual Store, full I32 output and PagedAppend/Attention permissions share one
// terminal. The first iteration is real ready work; the second changes only
// authority, retaining the same token/ID/page/control/workspace addresses.
TEST(GgmlCudaMixedReadySource, RealTokenStoreAndKvWarmCaptureThenReusePending) {
  auto context = take(g::cuda::create_context(), "Context");
  auto queue = take(context.create_queue(), "Queue");
  check(queue.reserve_completions(12), "Completion capacity");
  auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice), "device");
  auto storage = take(c::StorageOwner::adopt(context, std::move(memory)), "StorageOwner");
  auto staging = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost), "staging");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const gb::PrepareOptions batch_options{.host_workspace = staging.ref()};

  const auto item = dense(c::DataType::kFloat32, {2});
  const auto rows = dense(c::DataType::kFloat32, {1, 2});
  const auto ids = dense(c::DataType::kInt32, {1});
  const auto token_shape = dense(c::DataType::kInt32, {1, 1});
  const auto table_shape = dense(c::DataType::kInt32, {4, 1});
  const auto embedding_shape = dense(c::DataType::kFloat32, {4, 2});
  constexpr std::size_t stride = 256;
  auto backing = take(hyle::mem::Buffer::create(storage.ref(), 3 * stride, stride), "state backing");
  auto state_pool = take(c::LeasePool::adopt(std::move(backing), {.max_blocks = 6}), "state pool");
  const auto layout = take(c::LeaseSlotLayout::create(state_pool, item, 3, stride), "state layout");
  std::array<c::WriteLease, 3> writers;
  std::array<c::ReadLease, 3> padding;
  for (std::size_t slot = 0; slot < writers.size(); ++slot) {
    writers[slot] = take(state_pool.allocate(8, stride), "state writer");
    auto gap = take(state_pool.allocate(stride - 8, 1), "padding");
    check(take(gap.view(), "padding view").fill(std::byte{0x5a}), "initialize padding");
    padding[slot] = take(gap.publish(), "publish padding");
  }
  constexpr std::array<float, 2> old_values{101, 103};
  check(take(writers[0].view(), "old state").write(std::as_bytes(std::span(old_values))), "write old state");
  const auto unrelated_ready = take(writers[0].publish(), "old state ready");
  const auto state_schema = take(s::Schema::create(item, 3, stride), "Store schema");

  auto inputs = take(c::LeasePool::create(storage.ref(), 8 * stride, {.max_blocks = 8}), "inputs");
  const auto upload = [&](const auto &values) {
    auto writer = take(inputs.allocate(std::as_bytes(std::span(values)).size(), stride), "input writer");
    check(take(writer.view(), "input view").write(std::as_bytes(std::span(values))), "upload input");
    return take(writer.publish(), "input ready");
  };
  const auto table_ready = upload(std::array<std::int32_t, 4>{2, 1, 3, 0});
  const auto index_ready = upload(std::array<std::int32_t, 1>{0});
  const auto state_input = upload(std::array<float, 2>{3, 5});
  const auto embedding_ready = upload(std::array<float, 8>{10, 11, 20, 21, 30, 31, 40, 41});
  std::array<std::uint16_t, 64> value_words{}, query_words{};
  value_words.fill(0x3f00); // BF16 .5 exactly.
  const auto value_ready = upload(value_words), query_ready = upload(query_words);
  auto token_pool = take(c::LeasePool::create(storage.ref(), stride, {.max_blocks = 1}), "token pool");
  auto outputs = take(c::LeasePool::create(storage.ref(), 3 * stride, {.max_blocks = 3}), "outputs");

  const auto plain = take(c::CacheStorage::plain(c::DataType::kBFloat16), "cache format");
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kPaged,
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 1, .max_kv_rows = 2,
      .max_pages = 1, .max_page_refs = 1,
      .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64,
      .value_head_dim = 64, .page_size = 2};
  const auto schema = take(c::BatchSchema::create(capacity), "KV schema");
  const std::array<std::int32_t, 2> pointers{0, 1};
  const std::array<std::int32_t, 1> pages{0}, tails{1};
  const auto kv_layout = take(schema.version({pointers, pointers, pages, tails}), "KV layout");
  auto keys = take(c::PagedArena::create(schema, c::BatchCacheRole::kKey,
      storage.ref(), {.max_backing_bytes = 4096}), "keys");
  auto values = take(c::PagedArena::create(schema, c::BatchCacheRole::kValue,
      storage.ref(), {.max_backing_bytes = 4096}), "values");
  const std::array<std::int32_t, 2> no_pages{0, 0};
  const std::array<std::int32_t, 1> no_tail{0};
  const auto empty_layout = take(schema.version({no_pages, no_pages, {}, no_tail}), "empty old layout");
  const auto empty_frame = take(values.read(empty_layout, {}), "original empty old frame");
  auto key_writer = take(keys.acquire(0), "key writer");
  check(take(key_writer.lease().view(), "key storage").fill(std::byte{}), "zero keys");
  const auto key_ready = take(key_writer.publish(1), "key ready");
  const auto key_frame = take(keys.read(kv_layout, std::array{key_ready}), "key frame");
  const auto kv_rows = take(c::batch_query_description(capacity, 1), "KV row");
  const auto attention_output = take(c::batch_output_description(capacity, 1), "attention output");

  auto first = take(g::GraphBuilder::create(), "A builder");
  const auto table = take(first.input(table_shape), "A table");
  const auto index = take(first.input(ids), "A index");
  const auto token = take(first.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, table, index), "A token");
  check(first.publish(token), "publish A token");
  const auto state_rows = take(first.input(rows), "A state rows");
  const auto a_ids = take(first.input(ids), "A private IDs");
  const auto a_store = take(first.add_node(c::Operation::custom(s::Store{state_schema}), std::array{state_rows, a_ids}), "A Store");
  const auto value_input = take(first.input(kv_rows), "A KV rows");
  const auto append = take(first.add_node(c::Operation::custom(take(gb::paged_append(schema, c::BatchCacheRole::kValue), "append operation")),
      std::array{value_input}), "A append");
  check(first.depends_on(append, a_store), "order original A effects");
  const auto a_graph = take(std::move(first).finish(), "A graph");
  auto a_plan = take(g::cuda::compile(a_graph, context, storage), "compile A");
  auto producer = take(a_plan.executable.create_invocation(), "A invocation");
  check(producer.prepare_resources(), "prepare original A owned bindings");
  check(producer.bind(table, table_shape, table_ready), "bind table");
  check(producer.bind(index, ids, index_ready), "bind index");
  check(producer.bind(state_rows, rows, state_input), "bind state rows");
  check(producer.bind(value_input, kv_rows, value_ready), "bind append rows");
  check(producer.bind(token, token_shape, take(token_pool.allocate(4, stride), "token writer")), "bind original token writer");
  check(s::prepare(producer, queues, {.host_workspace = staging.ref()}), "prepare A Store");

  auto second = take(g::GraphBuilder::create(), "B builder");
  const auto embedding = take(second.input(embedding_shape), "B embedding");
  const auto token_input = take(second.input(token_shape), "B token");
  const auto flat_token = take(second.add(take(c::View::reshape(take(c::TensorShape::create(std::array<std::int64_t, 1>{1}), "token shape")), "token reshape"), token_input), "B zero-copy token reshape");
  const auto row = take(second.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, embedding, flat_token), "B embedding Gather");
  const auto doubled = take(second.add(c::Binary{.kind = c::BinaryKind::kAdd}, row, row), "B managed embedding math");
  check(second.publish(doubled), "publish embedding");
  const auto bank = take(second.input(take(state_schema.capacity_description(), "state capacity")), "B state pool");
  const auto source_ids = take(second.input(ids), "B state IDs");
  const auto gather = take(second.add_node(c::Operation(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}),
      std::array{bank, source_ids}), "B state Gather");
  const auto gathered = take(second.result(gather), "B state rows");
  check(second.publish(gathered), "publish B state rows");
  const auto destination_ids = take(second.input(ids), "B private destination IDs");
  // The default CUDA indexed-copy Store participates in the full capture.
  const auto b_store = take(second.add_node(c::Operation::custom(s::Store{state_schema}), std::array{gathered, destination_ids}), "B capturable Store");
  const auto query = take(second.input(kv_rows), "B query");
  const auto key = take(second.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kKey), "key capacity")), "B keys");
  const auto value = take(second.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kValue), "value capacity")), "B values");
  const auto metadata = take(second.input(take(schema.metadata_description(), "layout description")), "B metadata");
  const auto attention = take(second.add_node(c::Operation::custom(c::PagedAttention{
      .capacity = capacity, .scale = .125F, .causal = true}), std::array{query, key, value, metadata}), "B attention");
  const auto attended = take(second.result(attention), "B attended");
  check(second.publish(attended), "publish attention");
  const auto b_graph = take(std::move(second).finish(), "B graph");
  auto b_plan = take(g::cuda::compile(b_graph, context, storage), "compile B");
  auto consumer = take(b_plan.executable.create_invocation(), "B invocation");
  check(consumer.prepare_resources(), "prepare original B owned bindings");
  check(consumer.bind(embedding, embedding_shape, embedding_ready), "bind embedding");
  check(consumer.bind(query, kv_rows, query_ready), "bind query");
  check(consumer.bind(doubled, rows, take(outputs.allocate(8, stride), "embedding output")), "bind embedding output");
  check(consumer.bind(gathered, rows, take(outputs.allocate(8, stride), "state output")), "bind state output");
  check(consumer.bind(attended, attention_output, take(outputs.allocate(128, stride), "attention destination")), "bind attention output");
  check(s::prepare(consumer, queues, {.host_workspace = staging.ref()}), "prepare B Store");

  Gate producer_gate, reader_gate;
  unsigned diagnostic_pass = 0;
  const char *diagnostic_stage = "before relay";
  const auto mark = [&](const char *stage) {
    diagnostic_stage = stage;
    std::fprintf(stderr, "mixed pass=%u stage=%s\n", diagnostic_pass, stage);
    std::fflush(stderr);
  };
  try {
  for (unsigned pass = 0; pass < 2; ++pass) {
    diagnostic_pass = pass;
    mark("prepare A");
    SCOPED_TRACE(pass == 0 ? "first real ready execution" : "same tuple with actual pending inputs");
    std::array<c::WriteLease, 1> a_destinations{std::move(writers[1])};
    auto page = take(values.acquire(0), "fresh original value page");
    check(take(page.lease().view(), "value backing").fill(std::byte{0x55}), "poison unwritten page padding");
    std::array dirty{std::move(page)};
    auto append_pages = take(gb::PagedAppendPages::create(values, kv_layout, empty_frame, dirty), "A append transaction");
    check(gb::prepare(producer, queues, {}, batch_options), "prepare A append");
    check(gb::bind_append(producer, a_graph, append, std::move(append_pages)), "bind A append");
    check(s::bind(producer, a_graph, a_store,
        take(s::SlotBindings::create(layout, {}, a_destinations), "A transaction")), "bind A Store");
    auto state_source = take(g::StoreSource::prepare(producer, a_store, storage), "A state Source");
    auto kv_source = take(gb::PagedAppendSource::prepare(producer, append, storage), "A KV Source");
    check(producer.prepare_replay(queues, {.mode = g::ReplayMode::kPreferCapture}), "prepare original A");
    if (pass) {
      producer_gate.arm(take(c::cuda::stream(queue), "stream"));
      ASSERT_EQ(cudaLaunchHostFunc(producer_gate.stream, &Gate::wait, &producer_gate), cudaSuccess);
    }
    auto done = take(producer.submit(queues), "submit A");
    g::SubmissionSource common;
    g::OutputSource token_source;
    c::Completion consumed;
    // Keep the original terminals and typed Source owners alive until their
    // actual work retires, including an exception or fatal test assertion.
    hyle::util::ScopeGuard retire_submissions{[&]() noexcept {
      if (!done.valid() && !common.valid() && !consumed.valid()) return;
      producer_gate.release(); reader_gate.release();
      const auto observe = [&](hyle::Status status, const char *which) noexcept {
        if (!status.ok()) {
          std::fprintf(stderr, "mixed cleanup pass=%u stage=%s %s: %.*s\n",
              diagnostic_pass, diagnostic_stage, which,
              static_cast<int>(status.message().size()), status.message().data());
          std::fflush(stderr);
        }
      };
      if (consumed.valid()) observe(consumed.wait(), "consumer");
      if (common.valid()) observe(common.wait(), "producer Source");
      else if (done.valid()) observe(done.wait(), "producer Completion");
      consumed = {}; done = {};
    }};
    common = take(g::SubmissionSource::create(producer, std::move(done), storage), "whole A terminal");
    mark("capture A views");
    token_source = take(g::OutputSource::create(common, token, storage, token_pool), "original I32 Source");
    check(state_source.capture(common), "capture A Store");
    check(kv_source.capture(common), "capture A KV");
    c::ReadLease state_ready;
    c::PagedArenaRead frame;
    gb::PagedAppendPages returned_pages;
    s::SlotBindings returned_state;
    const auto publish_a = [&] {
      mark("retire and detach A");
      check(common.wait(), "retire A");
      check(token_source.detach(), "publish exact I32 result");
      check(common.detach(), "detach successful A");
      // The original append success is attested before Store withdrawal
      // revokes the invocation's Graph-output-ready flag.
      mark("take A append");
      returned_pages = take(gb::take_append(producer, append), "take original A KV");
      mark("take A Store and publish");
      returned_state = take(s::take(producer, a_store), "take original A state");
      state_ready = take(state_source.publish(0, returned_state.destinations()[0]), "publish original A state");
      frame = take(kv_source.ready(returned_pages), "publish exact A KV prefixes");
    };
    if (!pass) publish_a();

    mark("prepare B metadata and bundle");
    check(gb::prepare(consumer, queues, std::array{gb::NodeLayout{attention, kv_layout}}, batch_options), "prepare B runtime metadata");
    check(gb::bind(consumer, key, key_frame), "bind original ready keys");
    const std::array access{gather};
    std::array real_ready{state_ready};
    std::array bindings{
        token_source.reader_binding(token_input),
        take(state_source.reader_binding(consumer, b_graph, access,
            pass ? std::span<const c::ReadLease>{} : std::span<const c::ReadLease>(real_ready)),
            "original Store reader proof"),
        take(gb::reader_binding(consumer, value, kv_source, kv_layout), "original KV reader proof")};
    if (!pass) {
      const std::array foreign_ready{unrelated_ready};
      auto bad = bindings;
      gd::SourceBindingAccess::get(bad[1]).store_ready = foreign_ready;
      EXPECT_TRUE(g::bind_sources(consumer, bad).IsInvalidArgument());
    }
    check(g::bind_sources(consumer, bindings), "bind complete token/Store/KV bundle");
    std::array<c::WriteLease, 1> b_destinations{std::move(writers[2])};
    check(s::bind(consumer, b_graph, b_store,
        take(s::SlotBindings::create(layout, {}, b_destinations), "B transaction")), "bind B Store");
    mark("prepare B replay");
    check(g::prepare_sources(consumer, queues), "prepare real mixed bundle");
    ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
    const auto replay = take(consumer.replay_report(), "actual replay report");
    if (!pass) EXPECT_GT(replay.recorded_capture_range_count, 0u);
    else {
      EXPECT_EQ(replay.recorded_capture_range_count, 0u);
      EXPECT_GT(replay.reused_capture_range_count, 0u);
      reader_gate.arm(take(c::cuda::stream(queue), "stream"));
      ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &Gate::wait, &reader_gate), cudaSuccess);
    }
    const auto sources = take(consumer.input_source_report(), "one original terminal");
    EXPECT_EQ(sources.source_count, 1u); EXPECT_EQ(sources.device_wait_count, 1u);
    if (pass) {
      auto ordinary = consumer.submit(queues);
      if (ordinary) {
        // A broken pending-admission gate must not turn this negative into
        // a second submission or destroy live native work during unwinding.
        producer_gate.release(); reader_gate.release();
        const auto terminal = ordinary->wait();
        *ordinary = {};
        check(common.wait(), "retire A after unexpected ordinary submission");
        FAIL() << "ordinary submission accepted pending inputs: " << terminal.message();
      }
    }
    mark("submit B");
    consumed = take(g::submit_sources(consumer, queues), "submit mixed consumer");
    if (pass) {
      EXPECT_EQ(take(common.poll(), "A pending"), c::CompletionPoll::kPending);
      producer_gate.release();
      publish_a();
      EXPECT_EQ(take(consumed.poll(), "B still pending"), c::CompletionPoll::kPending);
      reader_gate.release();
    }
    mark("retire B and read outputs");
    check(consumed.wait(), "retire B"); consumed = {};
    {
      std::array<float, 2> embedded{}, state_values{};
      std::array<std::uint16_t, 64> attended_values{};
      check(take(consumer.output(doubled), "embedding result").storage().read(std::as_writable_bytes(std::span(embedded))), "read embedding");
      check(take(consumer.output(gathered), "state result").storage().read(std::as_writable_bytes(std::span(state_values))), "read state");
      check(take(consumer.output(attended), "attention result").storage().read(std::as_writable_bytes(std::span(attended_values))), "read attention");
      EXPECT_EQ(embedded, (std::array<float, 2>{60, 62}));
      EXPECT_EQ(state_values, (std::array<float, 2>{3, 5}));
      EXPECT_EQ(attended_values, value_words);
    }
    mark("take B Store");
    auto b_returned = take(s::take(consumer, b_store), "take original B writer");
    mark("release B bundle");
    check(g::release_sources(consumer), "release retired exact mixed bundle without resetting capture");
    writers[2] = std::move(b_returned.destinations()[0]);
    frame = {}; returned_pages = {}; returned_state.clear();
    bindings = {}; real_ready = {}; token_source = {}; state_source = {}; kv_source = {}; common = {};
    // The same original caller-output allocation is reclaimed by Graph on
    // its next actual submission, after the detached Source read is dropped.
    mark("reclaim A state");
    writers[1] = take(state_ready.try_write(), "reuse original A state slot");
  }
  mark("relay complete");
  } catch (const std::exception &error) {
    // These original Graphs and their storage/staging owners still exist.
    // Print before their outer destructors can obscure the original error.
    std::fprintf(stderr, "mixed exception pass=%u stage=%s: %s\n",
        diagnostic_pass, diagnostic_stage, error.what());
    std::fflush(stderr);
    producer_gate.release(); reader_gate.release();
    throw;
  } catch (...) {
    std::fprintf(stderr, "mixed exception pass=%u stage=%s: non-standard exception\n",
        diagnostic_pass, diagnostic_stage);
    std::fflush(stderr);
    producer_gate.release(); reader_gate.release();
    throw;
  }
  std::array<float, 2> old{};
  check(take(unrelated_ready.view(), "old snapshot").read(std::as_writable_bytes(std::span(old))), "read old snapshot");
  EXPECT_EQ(old, old_values);
  std::array<std::byte, stride - 8> gap{};
  for (const auto &ready : padding) {
    check(take(ready.view(), "original padding").read(gap), "read padding");
    EXPECT_TRUE(std::ranges::all_of(gap, [](auto value) { return value == std::byte{0x5a}; }));
  }
  EXPECT_FALSE(producer_gate.timed_out.load()); EXPECT_FALSE(reader_gate.timed_out.load());
}
} // namespace
