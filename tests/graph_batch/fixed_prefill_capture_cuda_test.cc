#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "../compute_provider_native/half_test_support.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using T = hyle::simd::bfloat16_t;

// Zero Q/K gives a uniform causal average. Changing page order, valid tails and
// prefix length must change that average without changing the recorded graph.
void fixed_prefill_case(unsigned queries, unsigned heads, unsigned depth) {
  SCOPED_TRACE(::testing::Message() << "Q=" << queries << " H=" << heads << " D=" << depth);
  constexpr unsigned kv_heads = 2, page_size = 16, capacity = 32768;
  auto context = c::cuda::create_context().value();
  auto queue = context.create_queue().value();
  auto device = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto storage = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity cap{.storage = c::BatchStorage::kPaged,
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = storage, .value_storage = storage, .max_requests = 1,
      .max_query_rows = queries, .max_kv_rows = capacity, .max_pages = capacity / page_size,
      .max_page_refs = capacity / page_size, .query_heads = heads, .kv_heads = kv_heads,
      .qk_head_dim = depth, .value_head_dim = depth, .page_size = page_size};
  const auto schema = c::BatchSchema::create(cap).value();
  const auto qd = c::batch_query_description(cap).value();
  const auto kd = c::batch_cache_description(cap, c::BatchCacheRole::kKey).value();
  const auto vd = c::batch_cache_description(cap, c::BatchCacheRole::kValue).value();
  const auto od = c::batch_output_description(cap, queries).value();
  const std::array descriptions{qd, kd, vd, schema.metadata_description().value()};
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> inputs;
  for (unsigned i = 0; i < inputs.size(); ++i) inputs[i] = builder.input(descriptions[i]).value();
  const auto node = builder.add_node(c::Operation::custom(c::PagedAttention{cap, .0625F}), inputs).value();
  const auto output = builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, device.ref(),
      {.operation_fallback = g::OperationFallbackPolicy::kReject}).value();
  auto slot = compiled.executable.create_invocation().value();
  auto pool = c::LeasePool::create(device.ref(), 80u << 20).value();
  const auto load = [&](const std::vector<T> &words) {
    auto writer = pool.allocate(words.size() * sizeof(T)).value();
    EXPECT_TRUE(writer.view()->write(std::as_bytes(std::span(words))).ok());
    return writer.publish().value();
  };
  std::vector<T> q(std::size_t(queries) * heads * depth, half_test::encode<T>(0));
  std::vector<T> k(std::size_t(capacity) * kv_heads * depth, half_test::encode<T>(0));
  std::vector<T> v(k.size());
  for (unsigned row = 0; row < capacity; ++row)
    for (unsigned h = 0; h < kv_heads; ++h)
      for (unsigned d = 0; d < depth; ++d)
        v[(std::size_t(row) * kv_heads + h) * depth + d] = half_test::encode<T>(
            float(int((row / page_size * 17) % 31) - 15) / 32 + h * .0625F + d / 512.F);
  const std::array leases{load(q), load(k), load(v)};
  const auto bytes = od.required_bytes().value();
  auto output_buffer = hyle::mem::Buffer::create(device.ref(), bytes + 32, 256).value();
  ASSERT_TRUE(output_buffer.view().fill(std::byte{0xa5}).ok());
  const auto view = c::TensorView::create(od, output_buffer.view().subview(0, bytes).value()).value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  unsigned step = 0;
  for (const unsigned length : {512u, 1024u, 4096u, 8192u, 16383u, 32768u, 640u}) {
    SCOPED_TRACE(::testing::Message() << "KV=" << length);
    const auto pages = (length + page_size - 1) / page_size;
    std::vector<std::int32_t> ids(pages);
    for (unsigned i = 0; i < pages; ++i) ids[i] = int((capacity / page_size - 1 - i * 17 - step * 7) % (capacity / page_size));
    const std::array<std::int32_t, 2> qp{0, int(queries)}, kp{0, int(pages)};
    const std::array<std::int32_t, 1> tails{int((length - 1) % page_size + 1)};
    const auto layout = schema.version({qp, kp, ids, tails}).value();
    auto status = g::batch::prepare(slot, queues, std::array{g::batch::NodeLayout{node, layout}},
        {.host_workspace = host.ref(), .max_float_workspace_bytes = 64u << 20,
         .max_int_workspace_bytes = 64u << 10});
    ASSERT_TRUE(status.ok()) << status.message();
    for (unsigned i = 0; i < leases.size(); ++i) {
      status = slot.bind(inputs[i], descriptions[i], leases[i]);
      ASSERT_TRUE(status.ok()) << status.message();
    }
    ASSERT_TRUE(slot.bind(output, view).ok());
    status = slot.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture,
        .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(status.ok()) << status.message();
    const auto replay = slot.replay_report().value();
    EXPECT_EQ(replay.route, g::ReplayRoute::kCaptured);
    EXPECT_EQ(replay.recorded_capture_range_count, step ? 0U : 1U);
    if (step) EXPECT_GE(replay.reused_capture_range_count, 1U);
    const auto native = g::batch::report(slot, node).value();
    EXPECT_LE(native.float_workspace_bytes, 64u << 20);
    auto done = slot.submit(queues).value();
    ASSERT_TRUE(done.wait().ok());
    std::vector<T> actual(std::size_t(queries) * heads * depth);
    ASSERT_TRUE(view.storage().read(std::as_writable_bytes(std::span(actual))).ok());
    for (const auto row : {0u, 7u, queries / 2, queries - 1})
      for (const auto h : {0u, heads / 2, heads - 1})
        for (const auto d : {0u, depth / 2, depth - 1}) {
          const auto allowed = length - queries + row + 1;
          double sum = 0;
          for (unsigned at = 0; at < allowed; ++at) {
            const auto physical = unsigned(ids[at / page_size]) * page_size + at % page_size;
            sum += half_test::decode(v[(std::size_t(physical) * kv_heads + h / (heads / kv_heads)) * depth + d]);
          }
          EXPECT_NEAR(half_test::decode(actual[(std::size_t(row) * heads + h) * depth + d]), sum / allowed, .007);
        }
    std::array<std::byte, 32> guard;
    ASSERT_TRUE(output_buffer.view().subview(bytes, guard.size()).value().read(guard).ok());
    EXPECT_TRUE(std::all_of(guard.begin(), guard.end(), [](auto b) { return b == std::byte{0xa5}; }));
    ++step;
  }
  ASSERT_TRUE(slot.close().ok());
  ASSERT_TRUE(context.drain().ok());
}

TEST(GraphBatchCapture, FixedPrefillKeepsCaptureAcrossGrowingAndShrinkingPagedPrefixes) {
  ASSERT_NO_FATAL_FAILURE(fixed_prefill_case(128, 8, 128));
  ASSERT_NO_FATAL_FAILURE(fixed_prefill_case(512, 16, 256));
}
} // namespace
