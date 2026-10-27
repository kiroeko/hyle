#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/simd/dtype.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
namespace m = hyle::mem;

// Hold only the original compute stream, so changed metadata must finish on
// its own real Queue. Release on every assertion exit; never call CUDA here.
struct MetadataUploadGate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false}, timed_out{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    static_cast<MetadataUploadGate *>(opaque)->released.wait(false, std::memory_order_acquire);
  }
  void arm(cudaStream_t value) {
    stream = value;
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10),
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
  ~MetadataUploadGate() {
    release();
    if (watchdog.joinable()) watchdog.join();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

TEST(GraphBatchCapture, FreshMetadataPermissionsRetainOnlyUnchangedNativeCommands) {
  constexpr std::uint32_t depth = 64, page_size = 16;
  auto context = g::cuda::create_context().value();
  auto queue = context.create_queue().value();
  auto device = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto plain = c::CacheStorage::plain(c::DataType::kFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kPaged,
      .query_type = c::DataType::kFloat16, .output_type = c::DataType::kFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 1, .max_kv_rows = 48,
      .max_pages = 4, .max_page_refs = 3,
      .query_heads = 1, .kv_heads = 1,
      .qk_head_dim = depth, .value_head_dim = depth, .page_size = page_size};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t, 2> qptr{0, 1}, kvptr{0, 2};
  const std::array<std::int32_t, 2> first_ids{0, 1}, next_ids{2, 1};
  const std::array<std::int32_t, 1> tails{page_size};
  const std::array layouts{
      schema.version({qptr, kvptr, first_ids, tails}).value(),
      schema.version({qptr, kvptr, next_ids, tails}).value()};
  const auto query_desc = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 1, depth}).value();
  const auto bank_desc = c::batch_cache_description(
      capacity, c::BatchCacheRole::kKey).value();
  auto builder = g::GraphBuilder::create().value();
  const auto query = builder.input(query_desc).value();
  const auto key = builder.input(bank_desc).value();
  const auto value = builder.input(bank_desc).value();
  const auto narrowed = builder.add(c::Cast{c::DataType::kFloat16}, query).value();
  const auto attention = builder.add_node(c::Operation::custom(
      gb::paged_attention(layouts[0], {.scale = .125F, .causal = false}).value()),
      std::array{narrowed, key, value}).value();
  const auto attended = builder.result(attention).value();
  const auto widened = builder.add(c::Cast{c::DataType::kFloat32}, attended).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, widened).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, device.ref(),
      {.operation_fallback = g::OperationFallbackPolicy::kAllow});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto slot = compiled->executable.create_invocation(
      {.resources = g::ResourceOptions{}}).value();
  auto query_buffer = m::Buffer::create(device.ref(), query_desc.required_bytes().value(), 64).value();
  auto output_buffer = m::Buffer::create(device.ref(), query_desc.required_bytes().value(), 64).value();
  ASSERT_TRUE(query_buffer.view().fill(std::byte{0}).ok());
  auto keys = c::PagedArena::create(schema, c::BatchCacheRole::kKey,
      device.ref(), {.max_backing_bytes = 1u << 20}).value();
  auto values = c::PagedArena::create(schema, c::BatchCacheRole::kValue,
      device.ref(), {.max_backing_bytes = 1u << 20}).value();
  std::array<c::ReadLease, 4> ready_keys, ready_values;
  std::vector<std::uint16_t> words(page_size * depth);
  for (std::uint32_t page = 0; page < 4; ++page) {
    auto kw = keys.acquire(page).value();
    auto vw = values.acquire(page).value();
    ASSERT_TRUE(kw.lease().view()->fill(std::byte{0}).ok());
    std::fill(words.begin(), words.end(), hyle::simd::f32_to_f16(float(page + 1)).bits);
    ASSERT_TRUE(vw.lease().view()->write(std::as_bytes(std::span(words))).ok());
    ready_keys[page] = kw.publish(page_size).value();
    ready_values[page] = vw.publish(page_size).value();
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto upload = g::detail::RegionAccess::metadata_queue(slot, g::ExecutionLaneId{0}, {});
  ASSERT_TRUE(upload) << upload.error().message();
  EXPECT_NE(c::cuda::stream(**upload).value(), c::cuda::stream(queue).value());
  EXPECT_EQ(*upload, g::detail::RegionAccess::metadata_queue(slot, g::ExecutionLaneId{0}, {}).value());
  EXPECT_FALSE(g::detail::RegionAccess::metadata_queue(slot, g::ExecutionLaneId{99}, {}));
  gb::PrepareOptions options{.host_workspace = host.ref(),
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10};
  const auto bind_ordinary = [&]() {
    auto status = slot.bind(query, c::ConstTensorView::create(query_desc, query_buffer.view()).value());
    status.Update(slot.bind(output, c::TensorView::create(query_desc, output_buffer.view()).value()));
    return status;
  };
  std::array<c::ConstTensorView, 2> bank_views;
  for (unsigned episode = 0; episode < 4; ++episode) {
    SCOPED_TRACE(episode);
    const auto& layout = layouts[episode ? 1 : 0];
    const std::array versions{gb::NodeLayout{attention, layout}};
    // Pure page-ID contents change on episode 1. Episode 2 explicitly makes
    // a fresh native plan. Episode 3 replaces the original native owner even
    // if the device allocator later returns the same numerical addresses.
    options.update = episode != 2;
    if (episode == 3) options.max_float_workspace_bytes = 16u << 20;
    MetadataUploadGate gate;
    if (episode == 1) {
      gate.arm(c::cuda::stream(queue).value());
      ASSERT_EQ(cudaLaunchHostFunc(gate.stream,
          &MetadataUploadGate::wait, &gate), cudaSuccess);
    }
    const auto prepared = gb::prepare(slot, queues, versions, options);
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    if (episode == 1) {
      EXPECT_FALSE(gate.released.load());
      EXPECT_FALSE(gate.timed_out.load());
      gate.release();
      // The callback no longer references the local gate after this event.
      ASSERT_EQ(cudaStreamSynchronize(c::cuda::stream(queue).value()), cudaSuccess);
    }
    EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kUnprepared);
    EXPECT_FALSE(slot.submit(queues));
    if (episode == 0) ASSERT_TRUE(bind_ordinary().ok());
    // Stable query/output carriers survive layout changes. Old exact page
    // permissions must not: replay cannot prepare until fresh frames bind.
    EXPECT_FALSE(slot.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject}).ok());
    const auto& ids = episode ? next_ids : first_ids;
    const std::array selected_keys{ready_keys[std::size_t(ids[0])], ready_keys[std::size_t(ids[1])]};
    const std::array selected_values{ready_values[std::size_t(ids[0])], ready_values[std::size_t(ids[1])]};
    auto key_frame = keys.read(layout, selected_keys).value();
    auto value_frame = values.read(layout, selected_values).value();
    bank_views[0] = c::detail::PagedArenaAccess::tensor(key_frame).value();
    bank_views[1] = c::detail::PagedArenaAccess::tensor(value_frame).value();
    ASSERT_TRUE(gb::bind(slot, key, std::move(key_frame)).ok());
    ASSERT_TRUE(gb::bind(slot, value, std::move(value_frame)).ok());
    const auto replay = slot.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(replay.ok()) << replay.message();
    EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
    const auto report = slot.replay_report();
    ASSERT_TRUE(report) << report.error().message();
    EXPECT_EQ(report->recorded_capture_range_count, episode == 1 ? 0u : 1u);
    EXPECT_EQ(report->reused_capture_range_count, episode == 1 ? 1u : 0u);
    const auto native = gb::report(slot, attention).value();
    EXPECT_EQ(native.reused_native_plan, episode == 1);
    {
      auto done = slot.submit(queues);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      auto result = slot.result(output).value();
      std::array<float, depth> actual;
      auto tensor = result.tensor();
      ASSERT_TRUE(tensor) << tensor.error().message();
      ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
      // Q=K=0, so noncausal attention averages the two constant full pages.
      const float expected = episode ? -2.5F : -1.5F;
      for (float element : actual) EXPECT_FLOAT_EQ(element, expected);
      EXPECT_TRUE(gb::prepare(slot, queues, versions, options).IsUnavailable());
      EXPECT_TRUE(slot.reset().IsUnavailable());
    }
  }
  // All ordinary ports can be rebound after this rejected cold episode. An
  // owned-layout node has no external metadata port to make preparation fail
  // merely for an unbound tensor: the fresh native token itself must reject.
  EXPECT_TRUE(gb::prepare(slot, queues, {}).IsInvalidArgument());
  ASSERT_TRUE(bind_ordinary().ok());
  // These are the real original banks, retained by their PagedArenas. No typed
  // native preparation is smuggled in while completing generic bindings.
  ASSERT_TRUE(slot.bind(key, bank_views[0]).ok());
  ASSERT_TRUE(slot.bind(value, bank_views[1]).ok());
  EXPECT_TRUE(slot.prepare_replay(queues,
      {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject})
      .IsFailedPrecondition());
  EXPECT_FALSE(slot.submit(queues));
  ASSERT_TRUE(slot.close().ok());
  ASSERT_TRUE(context.drain().ok());
}

TEST(GraphBatchCapture, FiniteQueryExtentsUseOneCapacityAndIndependentSlots) {
  auto context = g::cuda::create_context().value();
  auto device = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  auto queue = context.create_queue().value();
  constexpr std::uint32_t depth = 64, capacity_rows = 128;
  const auto plain = c::CacheStorage::plain(c::DataType::kFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kRagged,
      .query_type = c::DataType::kFloat16, .output_type = c::DataType::kFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = capacity_rows, .max_kv_rows = capacity_rows,
      .max_pages = 0, .max_page_refs = 0,
      .query_heads = 1, .kv_heads = 1, .qk_head_dim = depth, .value_head_dim = depth,
      .page_size = 1};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t, 2> q128{0, 128}, q1{0, 1}, q2{0, 2}, kv128{0, 128};
  const std::array layouts{schema.version({q128, kv128, {}, {}}).value(),
                           schema.version({q1, kv128, {}, {}}).value()};
  const auto overflow_layout = schema.version({q2, kv128, {}, {}}).value();
  // Equal contents are not the same owned metadata version for two readers.
  const auto other_version = schema.version({q1, kv128, {}, {}}).value();
  ASSERT_NE(other_version, layouts[1]);
  const std::array fixed_descriptions{
      c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value(),
      c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value(),
      schema.metadata_description().value()};
  auto builder = g::GraphBuilder::create().value();
  const auto rows = builder.dimension({1, capacity_rows, 1}).value();
  const auto query_spec = g::TensorSpec::dense(c::DataType::kFloat16,
      std::array{g::Dimension(rows), g::Dimension::fixed(1),
                 g::Dimension::fixed(depth)}).value();
  std::array<g::ValueId, 4> inputs;
  inputs[0] = builder.input(query_spec).value();
  for (std::size_t port = 1; port < inputs.size(); ++port)
    inputs[port] = builder.input(fixed_descriptions[port - 1]).value();
  std::array<g::NodeId, 2> nodes;
  std::array<g::ValueId, 2> outputs;
  for (std::size_t branch = 0; branch < nodes.size(); ++branch) {
    nodes[branch] = builder.add_node(c::Operation::custom(c::RaggedAttention{
        .capacity = capacity, .scale = .125F, .causal = false}), inputs,
        std::array{query_spec}).value();
    outputs[branch] = builder.result(nodes[branch]).value();
    ASSERT_TRUE(builder.publish(outputs[branch]).ok());
  }
  const auto graph = std::move(builder).finish().value();
  const std::array large{g::ShapeBinding{rows, 128}}, small{g::ShapeBinding{rows, 1}};
  const std::array shapes{large, small};
  const std::array variants{g::ExactShapeVariant{large}, g::ExactShapeVariant{small}};
  auto compiled = g::cuda::compile(graph, context, device.ref(),
      {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
       .operation_fallback = g::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_FALSE(compiled->executable.has_execution_route());
  ASSERT_EQ(compiled->executable.exact_variant_count(), 2u);
  const std::array absent{g::ShapeBinding{rows, 2}};
  auto missing = compiled->executable.create_invocation({.shape = absent});
  ASSERT_FALSE(missing);
  EXPECT_TRUE(missing.error().IsUnimplemented());
  EXPECT_EQ(compiled->executable.exact_variant_count(), 2u); // No lazy variant.
  std::array<g::Invocation, 2> slots;
  for (std::size_t shape = 0; shape < slots.size(); ++shape)
    slots[shape] = compiled->executable.create_invocation({.shape = shapes[shape]}).value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  gb::PrepareOptions options{.host_workspace = host.ref(),
      .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10};
  auto keys = m::Buffer::create(device.ref(), fixed_descriptions[0].required_bytes().value(), 64).value();
  auto values = m::Buffer::create(device.ref(), fixed_descriptions[1].required_bytes().value(), 64).value();
  auto replacement_metadata = m::Buffer::create(
      device.ref(), fixed_descriptions[2].required_bytes().value(), 64).value();
  ASSERT_TRUE(keys.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(replacement_metadata.view().fill(std::byte{0}).ok());
  std::array<m::Buffer, 2> queries;
  std::array<std::array<m::Buffer, 2>, 2> destinations;
  std::array<g::GraphOutput, 2> old_results;
  const std::array<std::uint32_t, 2> extents{128, 1};
  for (std::size_t shape = 0; shape < slots.size(); ++shape) {
    SCOPED_TRACE(extents[shape]);
    auto &slot = slots[shape];
    const auto query_desc = c::batch_query_description(capacity, extents[shape]).value();
    const auto output_desc = c::batch_output_description(capacity, extents[shape]).value();
    ASSERT_EQ(query_desc.required_bytes().value(), std::size_t(extents[shape]) * depth * 2);
    ASSERT_EQ(output_desc.required_bytes().value(), query_desc.required_bytes().value());
    ASSERT_EQ(layouts[shape].facts().query_rows, extents[shape]); // A=R, no padding.
    EXPECT_EQ(layouts[shape].origin_schema().capacity().max_query_rows, capacity_rows);
    queries[shape] = m::Buffer::create(device.ref(), query_desc.required_bytes().value(), 64).value();
    ASSERT_TRUE(queries[shape].view().fill(std::byte{0}).ok());
    for (auto &destination : destinations[shape])
      destination = m::Buffer::create(device.ref(), output_desc.required_bytes().value(), 64).value();
    const std::array versions{gb::NodeLayout{nodes[0], layouts[shape]},
                              gb::NodeLayout{nodes[1], layouts[shape]}};
    if (shape == 1) {
      const std::array too_many{gb::NodeLayout{nodes[0], overflow_layout},
                               gb::NodeLayout{nodes[1], overflow_layout}};
      EXPECT_TRUE(gb::prepare(slot, queues, too_many, options).IsInvalidArgument());
      EXPECT_FALSE(slot.submit(queues)); // A2 cannot enter an R1 native plan.
      const std::array different_owners{versions[0], gb::NodeLayout{nodes[1], other_version}};
      EXPECT_TRUE(gb::prepare(slot, queues, different_owners, options).IsInvalidArgument());
      EXPECT_FALSE(slot.submit(queues));
    }
    const auto prepared = gb::prepare(slot, queues, versions, options);
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    for (const auto node : nodes)
      EXPECT_EQ(gb::report(slot, node)->layout_generation, layouts[shape].generation());
    ASSERT_TRUE(slot.bind(inputs[0], c::ConstTensorView::create(query_desc, queries[shape].view()).value()).ok());
    ASSERT_TRUE(slot.bind(inputs[1], c::ConstTensorView::create(fixed_descriptions[0], keys.view()).value()).ok());
    ASSERT_TRUE(slot.bind(inputs[2], c::ConstTensorView::create(fixed_descriptions[1], values.view()).value()).ok());
    for (std::size_t branch = 0; branch < outputs.size(); ++branch)
      ASSERT_TRUE(slot.bind(outputs[branch], c::TensorView::create(output_desc, destinations[shape][branch].view()).value()).ok());
    EXPECT_TRUE(slot.bind(inputs[3], c::ConstTensorView::create(
        fixed_descriptions[2], replacement_metadata.view()).value()).IsFailedPrecondition());
    // Q=K=0 gives exactly uniform probabilities. Half the 128 V rows are
    // 1/3 (then 4/8), so the independent oracle is exactly F16 2 (then 6).
    // Distinct results also expose accidental output sharing between slots.
    const auto low = shape ? std::uint16_t{0x4400} : std::uint16_t{0x3c00};
    const auto high = shape ? std::uint16_t{0x4800} : std::uint16_t{0x4200};
    std::vector<std::uint16_t> words(std::size_t(capacity_rows) * depth, low);
    std::fill(words.begin() + words.size() / 2, words.end(), high);
    ASSERT_TRUE(values.view().write(std::as_bytes(std::span(words))).ok());
    const auto replay = slot.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(replay.ok()) << replay.message();
    EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
    ASSERT_TRUE(slot.replay_report());
    EXPECT_GT(slot.replay_report()->recorded_capture_range_count, 0u);
    {
      auto completion = slot.submit(queues);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      for (std::size_t branch = 0; branch < outputs.size(); ++branch) {
        auto result = slot.result(outputs[branch]).value();
        EXPECT_EQ(result.description().value(), output_desc);
        auto tensor = result.tensor().value();
        ASSERT_EQ(tensor.storage().size(), output_desc.required_bytes().value());
        std::vector<std::uint16_t> actual(std::size_t(extents[shape]) * depth);
        ASSERT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(actual))).ok());
        for (const auto word : actual) EXPECT_EQ(word, shape ? 0x4600u : 0x4000u);
        if (shape == 0) old_results[branch] = std::move(result);
      }
    }
    if (shape == 0) {
      EXPECT_TRUE(gb::prepare(slot, queues, versions, options).IsUnavailable());
      EXPECT_TRUE(slot.reset().IsUnavailable());
    }
  }
  for (const auto &result : old_results) {
    std::vector<std::uint16_t> actual(std::size_t(capacity_rows) * depth);
    ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(actual))).ok());
    for (const auto word : actual) EXPECT_EQ(word, 0x4000u);
  }
  EXPECT_EQ(compiled->executable.exact_variant_count(), 2u);
  for (auto &result : old_results) result.reset();
  // Release all borrowed/captured bindings before their external buffers.
  for (auto &slot : slots) {
    ASSERT_TRUE(slot.close().ok());
    slot = {};
  }
}

TEST(GraphBatchCapture, InactiveOutputRowsAreClearedByCapturedCommands) {
  auto context = g::cuda::create_context().value();
  auto device = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto resource = device.ref(), workspace = host.ref();
  auto queue = context.create_queue().value();
  struct Configuration { c::DataType type; c::BatchStorage storage; };
  // Two existing native domains, including GQA and unequal Q/V head depths.
  const std::array configurations{
      Configuration{c::DataType::kFloat16, c::BatchStorage::kRagged},
      Configuration{c::DataType::kBFloat16, c::BatchStorage::kPaged}};
  for (const auto configuration : configurations) {
    SCOPED_TRACE(static_cast<int>(configuration.type));
    const bool paged = configuration.storage == c::BatchStorage::kPaged;
    const auto plain = c::CacheStorage::plain(configuration.type).value();
    constexpr std::uint32_t capacity_rows = 128, heads = 2, value_depth = 128;
    const c::BatchCapacity capacity{
        .storage = configuration.storage,
        .query_type = configuration.type, .output_type = configuration.type,
        .key_storage = plain, .value_storage = plain,
        .max_requests = 1, .max_query_rows = capacity_rows,
        .max_kv_rows = paged ? 16u : 2u,
        .max_pages = paged ? 1u : 0u, .max_page_refs = paged ? 1u : 0u,
        .query_heads = heads, .kv_heads = 1, .qk_head_dim = 64,
        .value_head_dim = value_depth, .page_size = paged ? 16u : 1u};
    auto schema = c::BatchSchema::create(capacity).value();
    struct Episode { std::uint32_t physical_rows, active_rows; };
    // The R1 episode must not write schema slack. The empty layout still
    // owns a nonempty R4 output and must initialize all four rows.
    for (const auto episode : std::array{Episode{4, 1}, Episode{1, 1}, Episode{4, 0}}) {
      SCOPED_TRACE(episode.physical_rows);
      SCOPED_TRACE(episode.active_rows);
      const std::array<std::int32_t, 2> qp{0, static_cast<std::int32_t>(episode.active_rows)};
      const std::array<std::int32_t, 2> kp{0, paged ? 1 : 2};
      const std::array<std::int32_t, 1> ids{0}, tails{16};
      const auto layout = schema.version({qp, kp,
          paged ? std::span<const std::int32_t>(ids) : std::span<const std::int32_t>{},
          paged ? std::span<const std::int32_t>(tails) : std::span<const std::int32_t>{}}).value();
      const std::array descriptions{
          c::batch_query_description(capacity, episode.physical_rows).value(),
          c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value(),
          c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value(),
          schema.metadata_description().value()};
      const auto output_desc = c::batch_output_description(capacity, episode.physical_rows).value();
      auto builder = g::GraphBuilder::create().value();
      std::array<g::ValueId, 4> inputs;
      for (std::size_t port = 0; port < inputs.size(); ++port)
        inputs[port] = builder.input(descriptions[port]).value();
      const auto operation = paged
          ? c::Operation::custom(c::PagedAttention{.capacity = capacity, .scale = .125F, .causal = false})
          : c::Operation::custom(c::RaggedAttention{.capacity = capacity, .scale = .125F, .causal = false});
      const auto added = builder.add_node(operation, inputs);
      ASSERT_TRUE(added) << added.error().message();
      const auto node = *added;
      const auto output = builder.result(node).value();
      ASSERT_TRUE(builder.publish(output).ok());
      const auto graph = std::move(builder).finish().value();
      const std::array domains{g::StorageDomain{resource}};
      const std::array targets{g::ExecutionTarget{
          g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
      auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets},
          {.operation_fallback = g::OperationFallbackPolicy::kReject});
      ASSERT_TRUE(compiled) << compiled.error().message();
      auto slot = compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value();
      const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
      const std::array layouts{gb::NodeLayout{node, layout}};
      std::array<m::Buffer, 3> payload;
      for (std::size_t port = 0; port < payload.size(); ++port) {
        payload[port] = m::Buffer::create(resource, descriptions[port].required_bytes().value(), 64).value();
        ASSERT_TRUE(payload[port].view().fill(std::byte{0}).ok());
      }
      const auto low = paged ? std::uint16_t{0x3f80} : std::uint16_t{0x3c00};
      const auto high = paged ? std::uint16_t{0x4040} : std::uint16_t{0x4200};
      std::vector<std::uint16_t> values(payload[2].size() / 2, low);
      std::fill(values.begin() + values.size() / 2, values.end(), high);
      ASSERT_TRUE(payload[2].view().write(std::as_bytes(std::span(values))).ok());
      constexpr std::size_t guard_words = 32, row_words = heads * value_depth;
      auto destination = m::Buffer::create(resource,
          (capacity_rows * row_words + 2 * guard_words) * 2, 64).value();
      const auto output_view = c::TensorView::create(output_desc,
          destination.view().subview(guard_words * 2).value()).value();
      ASSERT_EQ(output_view.storage().size(), output_desc.required_bytes().value());
      // Poison after prepare/capture and again before replay. A one-time cold
      // clear, wrong stream, or schema-wide clear cannot satisfy this oracle.
      for (unsigned replay_index = 0; replay_index < 2; ++replay_index) {
        const auto prepared = gb::prepare(slot, queues, layouts, {.host_workspace = workspace,
            .max_float_workspace_bytes = 8u << 20, .max_int_workspace_bytes = 64u << 10});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        for (std::size_t port = 0; port < payload.size(); ++port)
          ASSERT_TRUE(slot.bind(inputs[port], c::ConstTensorView::create(
              descriptions[port], payload[port].view()).value()).ok());
        ASSERT_TRUE(slot.bind(output, output_view).ok());
        const auto replay = slot.prepare_replay(queues,
            {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
        ASSERT_TRUE(replay.ok()) << replay.message();
        EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
        ASSERT_TRUE(destination.view().fill(std::byte{0x35}).ok());
        {
          auto done = slot.submit(queues);
          ASSERT_TRUE(done) << done.error().message();
          ASSERT_TRUE(done->wait().ok());
        }
        std::vector<std::uint16_t> actual(destination.size() / 2);
        ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(actual))).ok());
        const auto active = actual.begin() + guard_words;
        const auto tail = active + episode.active_rows * row_words;
        const auto end = active + episode.physical_rows * row_words;
        EXPECT_EQ(std::count(actual.begin(), active, std::uint16_t{0x3535}), guard_words);
        // Q=K=0: native attention averages equally many V=1 and V=3.
        // Both F16 and BF16 encode the independent exact result 2 as 0x4000.
        EXPECT_EQ(std::count(active, tail, std::uint16_t{0x4000}),
            std::size_t(episode.active_rows) * row_words);
        EXPECT_EQ(std::count(tail, end, std::uint16_t{0}),
            std::size_t(episode.physical_rows - episode.active_rows) * row_words);
        EXPECT_EQ(std::count(end, actual.end(), std::uint16_t{0x3535}),
            std::size_t(actual.end() - end));
      }
      ASSERT_TRUE(slot.close().ok());
      slot = {};
    }
  }
}
TEST(GraphBatchCapture, UniformDecodePlansKeepCaptureAcrossGrowingPagedPrefixes) {
  // Fixed physical ports exercise the real R/T layout transition independently
  // of Graph tensor variants. Every inactive output word must still be zeroed
  // by the captured provider, with the adjacent caller bytes untouched.
  constexpr std::uint32_t depth = 256, page_size = 16, heads = 4, kv_heads = 2;
  constexpr std::uint32_t physical_rows = 4, pages_per_request = 64;
  constexpr std::size_t row_words = heads * depth, guard_words = 32;
  constexpr float scale = .0625F;
  auto context = g::cuda::create_context().value();
  auto queue = context.create_queue().value();
  auto device = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  const auto resource = device.ref(), workspace = host.ref();
  const auto plain = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kPaged,
      .query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 2, .max_query_rows = physical_rows, .max_kv_rows = 2048,
      .max_pages = 128, .max_page_refs = 128, .query_heads = heads, .kv_heads = kv_heads,
      .qk_head_dim = depth, .value_head_dim = depth, .page_size = page_size};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array descriptions{
      c::batch_query_description(capacity, physical_rows).value(),
      c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value(),
      c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value(),
      schema.metadata_description().value()};
  const auto output_desc = c::batch_output_description(capacity, physical_rows).value();
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 4> inputs;
  for (std::size_t port = 0; port < inputs.size(); ++port)
    inputs[port] = builder.input(descriptions[port]).value();
  const auto node = builder.add_node(c::Operation::custom(c::PagedAttention{
      .capacity = capacity, .scale = scale, .causal = true, .window_left = -1}), inputs).value();
  const auto output = builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{resource}};
  const std::array targets{g::ExecutionTarget{
      g::ExecutionSource::acquire(context).value(), g::StorageDomainId{0}}};
  auto compiled = gb::compile(graph, {.storage_domains = domains, .targets = targets},
      {.operation_fallback = g::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  std::array<g::Invocation, 2> slots;
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  std::array<std::shared_ptr<c::Queue>, 2> metadata_queues;
  std::array<m::Buffer, 2> queries, destinations;
  std::array<std::vector<std::uint16_t>, 2> query_words;
  const auto word = [](float x) { return hyle::simd::f32_to_bf16(x).bits; };
  const auto number = [](std::uint16_t x) { return hyle::simd::bf16_to_f32({x}); };
  for (std::size_t index = 0; index < slots.size(); ++index) {
    slots[index] = compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    auto metadata = g::detail::RegionAccess::metadata_queue(slots[index], g::ExecutionLaneId{0}, {});
    ASSERT_TRUE(metadata) << metadata.error().message();
    metadata_queues[index] = *metadata;
    ASSERT_NE(metadata_queues[index].get(), &queue);
    const auto execution_stream = c::cuda::stream(queue), upload_stream = c::cuda::stream(**metadata);
    ASSERT_TRUE(execution_stream); ASSERT_TRUE(upload_stream);
    EXPECT_NE(*execution_stream, *upload_stream);
    query_words[index].resize(physical_rows * row_words);
    for (std::size_t at = 0; at < query_words[index].size(); ++at) {
      const auto row = at / row_words, head = at / depth % heads, d = at % depth;
      query_words[index][at] = word(float(int((index * 17 + row * 11 + head * 7 + d * 5) % 23) - 11) / 32.F);
    }
    queries[index] = m::Buffer::create(resource, descriptions[0].required_bytes().value(), 64).value();
    ASSERT_TRUE(queries[index].view().write(std::as_bytes(std::span(query_words[index]))).ok());
    destinations[index] = m::Buffer::create(resource,
        (physical_rows * row_words + 2 * guard_words) * sizeof(std::uint16_t), 64).value();
  }
  std::array<m::Buffer, 2> banks;
  std::array<std::vector<std::uint16_t>, 2> bank_words;
  for (std::size_t bank = 0; bank < banks.size(); ++bank) {
    bank_words[bank].resize(descriptions[bank + 1].required_bytes().value() / sizeof(std::uint16_t));
    for (std::size_t at = 0; at < bank_words[bank].size(); ++at) {
      const auto row = at / (kv_heads * depth), head = at / depth % kv_heads, d = at % depth;
      const auto numerator = bank == 0 ? int((row * 13 + head * 7 + d * 3) % 31) - 15
                                      : int((row * 11 + head * 5 + d) % 29) - 14;
      bank_words[bank][at] = word(float(numerator) / (bank == 0 ? 32.F : 16.F));
    }
    banks[bank] = m::Buffer::create(resource, descriptions[bank + 1].required_bytes().value(), 64).value();
    ASSERT_TRUE(banks[bank].view().write(std::as_bytes(std::span(bank_words[bank]))).ok());
  }
  struct Geometry { std::uint32_t requests, query_rows; };
  const std::array geometries{Geometry{1, 1}, Geometry{2, 1}, Geometry{1, 2}, Geometry{2, 2}};
  constexpr std::array<std::uint32_t, 7> prefixes{127, 128, 129, 383, 384, 385, 1023};
  const auto physical_row = [](std::uint32_t request, std::uint32_t row) {
    return (request * pages_per_request + ((row / page_size * 17 + 3) % pages_per_request)) * page_size + row % page_size;
  };
  const auto make_layout = [&](Geometry geometry, std::uint32_t prefix) {
    std::vector<std::int32_t> qp{0}, kp{0}, ids, tails;
    for (std::uint32_t request = 0; request < geometry.requests; ++request) {
      qp.push_back(static_cast<std::int32_t>((request + 1) * geometry.query_rows));
      const auto count = (prefix + page_size - 1) / page_size;
      for (std::uint32_t page = 0; page < count; ++page)
        ids.push_back(static_cast<std::int32_t>(physical_row(request, page * page_size) / page_size));
      kp.push_back(static_cast<std::int32_t>(ids.size()));
      tails.push_back(static_cast<std::int32_t>((prefix - 1) % page_size + 1));
    }
    return schema.version({qp, kp, ids, tails});
  };
  std::uint64_t recorded_total = 0, reused_total = 0, waits_total = 0;
  std::size_t graph_float_bytes_at_127 = 0;
  double maximum_error = 0;
  // A null update means replay the other Invocation unchanged after its peer
  // was replanned. This catches shared plan/mirror/output state between slots.
  const auto episode = [&](std::size_t index, Geometry geometry, std::uint32_t prefix,
                           const bool *update, bool must_reuse, bool must_record,
                           std::size_t float_cap = 64u << 20) {
    SCOPED_TRACE(index);
    SCOPED_TRACE(geometry.requests);
    SCOPED_TRACE(geometry.query_rows);
    SCOPED_TRACE(prefix);
    if (update) {
      auto layout = make_layout(geometry, prefix);
      ASSERT_TRUE(layout) << layout.error().message();
      const std::array layouts{gb::NodeLayout{node, *layout}};
      const auto status = gb::prepare(slots[index], queues, layouts, {.host_workspace = workspace,
          .max_float_workspace_bytes = float_cap, .max_int_workspace_bytes = 8u << 20, .update = *update});
      ASSERT_TRUE(status.ok()) << status.message();
    }
    auto &slot = slots[index];
    for (std::size_t port = 0; port < 3; ++port) {
      const auto storage = port == 0 ? queries[index].view() : banks[port - 1].view();
      const auto status = slot.bind(inputs[port], c::ConstTensorView::create(descriptions[port], storage).value());
      ASSERT_TRUE(status.ok()) << status.message();
    }
    const auto output_view = c::TensorView::create(output_desc,
        destinations[index].view().subview(guard_words * sizeof(std::uint16_t)).value()).value();
    ASSERT_TRUE(slot.bind(output, output_view).ok());
    const auto planned = gb::report(slot, node);
    ASSERT_TRUE(planned) << planned.error().message();
    EXPECT_LE(planned->float_workspace_bytes, float_cap);
    if (index == 0 && geometry.requests == 1 && geometry.query_rows == 1 && prefix == 127 && float_cap == (64u << 20))
      graph_float_bytes_at_127 = planned->float_workspace_bytes;
    const auto capture = slot.prepare_replay(queues,
        {.mode = g::ReplayMode::kRequireCapture, .host_wait = g::HostWaitPolicy::kReject});
    ASSERT_TRUE(capture.ok()) << capture.message();
    ASSERT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
    const auto replay = slot.replay_report();
    ASSERT_TRUE(replay) << replay.error().message();
    const auto recorded = replay->recorded_capture_range_count, reused = replay->reused_capture_range_count;
    ASSERT_GT(recorded + reused, 0u);
    if (must_reuse) { EXPECT_TRUE(planned->reused_native_plan); EXPECT_EQ(recorded, 0u); EXPECT_GT(reused, 0u); }
    if (must_record) { EXPECT_FALSE(planned->reused_native_plan); EXPECT_GT(recorded, 0u); EXPECT_EQ(reused, 0u); }
    if (!update) { EXPECT_EQ(recorded, 0u); EXPECT_GT(reused, 0u); }
    if (update && planned->reused_native_plan) { EXPECT_EQ(recorded, 0u); EXPECT_GT(reused, 0u); }
    if (update && !planned->reused_native_plan) EXPECT_GT(recorded, 0u);
    // Reuse is a topology fact. Changed metadata can still perform a real
    // scheduler upload and wait; preserve those independent observed costs.
    std::printf("stable_attention slot=%zu R=%u T=%u KV=%u update=%d native_reused=%u recorded=%u reused=%u plan_waits=%u upload_waits=%llu split=%u padded=%u tile=%u float_bytes=%zu float_cap=%zu int_bytes=%zu\n",
        index, geometry.requests, geometry.query_rows, prefix, update ? int(*update) : -1,
        unsigned(planned->reused_native_plan), unsigned(recorded), unsigned(reused), planned->host_waits,
        static_cast<unsigned long long>(planned->metadata_upload.host_waits), unsigned(planned->split_kv),
        planned->padded_batch_size, planned->cta_tile_q, planned->float_workspace_bytes,
        float_cap, planned->int_workspace_bytes);
    recorded_total += recorded; reused_total += reused;
    if (update) waits_total += planned->host_waits;
    ASSERT_TRUE(destinations[index].view().fill(std::byte{0x35}).ok());
    {
      auto done = slot.submit(queues);
      ASSERT_TRUE(done) << done.error().message();
      const auto terminal = done->wait();
      ASSERT_TRUE(terminal.ok()) << terminal.message();
    }
    std::vector<std::uint16_t> actual(destinations[index].size() / sizeof(std::uint16_t));
    ASSERT_TRUE(destinations[index].view().read(std::as_writable_bytes(std::span(actual))).ok());
    const auto begin = actual.begin() + guard_words;
    const auto active_end = begin + geometry.requests * geometry.query_rows * row_words;
    const auto end = begin + physical_rows * row_words;
    EXPECT_EQ(std::count(actual.begin(), begin, std::uint16_t{0x3535}), guard_words);
    EXPECT_EQ(std::count(active_end, end, std::uint16_t{0}), std::size_t(end - active_end));
    EXPECT_EQ(std::count(end, actual.end(), std::uint16_t{0x3535}), guard_words);
    for (std::uint32_t request = 0; request < geometry.requests; ++request) {
      for (std::uint32_t query_row = 0; query_row < geometry.query_rows; ++query_row) {
        const auto row = request * geometry.query_rows + query_row;
        // Bottom-right causal masking: the T queries end at this real prefix.
        const auto visible = prefix - geometry.query_rows + query_row + 1;
        for (std::uint32_t head = 0; head < heads; ++head) {
          const auto kv_head = head / (heads / kv_heads);
          std::vector<double> scores(visible);
          for (std::uint32_t key_row = 0; key_row < visible; ++key_row) {
            const auto physical = physical_row(request, key_row);
            double dot = 0;
            for (std::uint32_t d = 0; d < depth; ++d)
              dot += double(number(query_words[index][(row * heads + head) * depth + d])) *
                     double(number(bank_words[0][(physical * kv_heads + kv_head) * depth + d]));
            scores[key_row] = dot * scale;
          }
          const auto largest = *std::max_element(scores.begin(), scores.end());
          double denominator = 0;
          for (auto &score : scores) { score = std::exp(score - largest); denominator += score; }
          for (std::uint32_t d = 0; d < depth; ++d) {
            double expected = 0;
            for (std::uint32_t key_row = 0; key_row < visible; ++key_row)
              expected += scores[key_row] * double(number(bank_words[1][
                  (physical_row(request, key_row) * kv_heads + kv_head) * depth + d]));
            expected /= denominator;
            const auto observed = double(number(*(begin + (row * heads + head) * depth + d)));
            ASSERT_TRUE(std::isfinite(observed));
            const auto error = std::abs(observed - expected);
            maximum_error = std::max(maximum_error, error);
            ASSERT_LE(error, .007) << "request=" << request << " row=" << query_row
                << " head=" << head << " d=" << d << " expected=" << expected << " actual=" << observed;
          }
        }
      }
    }
    auto metadata = g::detail::RegionAccess::metadata_queue(slot, g::ExecutionLaneId{0}, {});
    ASSERT_TRUE(metadata) << metadata.error().message();
    EXPECT_EQ(metadata->get(), metadata_queues[index].get());
  };
  const bool fresh = false, update = true;
  for (std::size_t geometry = 0; geometry < geometries.size(); ++geometry) {
    for (std::size_t at = 0; at < prefixes.size(); ++at) {
      const bool same_page = at == 1 || at == 4;
      for (std::size_t index = 0; index < slots.size(); ++index)
        ASSERT_NO_FATAL_FAILURE(episode(index, geometries[geometry], prefixes[at], at ? &update : &fresh,
            same_page, at == 0));
      ASSERT_NO_FATAL_FAILURE(episode(0, geometries[geometry], prefixes[at], nullptr, false, false));
    }
    // Explicit fresh must withdraw the old commands even at identical geometry;
    // it must not mutate the peer's still-reusable recording or native state.
    ASSERT_NO_FATAL_FAILURE(episode(0, geometries[geometry], prefixes.back(), &fresh, false, true));
    ASSERT_NO_FATAL_FAILURE(episode(1, geometries[geometry], prefixes.back(), nullptr, false, false));
    // Includes R2/T1 -> R1/T2: total Q rows alone cannot qualify reuse.
    ASSERT_NO_FATAL_FAILURE(episode(0, geometries[(geometry + 1) % geometries.size()],
        prefixes.back(), &update, false, true));
  }
  // Padded graph scratch must not turn an executable ordinary native plan into
  // a cap rejection. Prove this cap excludes the actual large-cap graph plan;
  // then keep the same exact numerical/tail/guard oracle on the original
  // non-padded scheduler. A genuine fresh plan at these lengths is permitted.
  constexpr std::size_t small_float_cap = 1u << 20;
  ASSERT_GT(graph_float_bytes_at_127, small_float_cap);
  for (std::size_t at = 0; at < 3; ++at)
    ASSERT_NO_FATAL_FAILURE(episode(0, geometries[0], prefixes[at], &update,
        false, at == 0, small_float_cap));
  ASSERT_NO_FATAL_FAILURE(episode(0, geometries[0], prefixes[2], &fresh,
      false, true, small_float_cap));
  RecordProperty("stable_attention_large_cap_graph_float_bytes", std::to_string(graph_float_bytes_at_127));
  RecordProperty("stable_attention_small_float_cap", std::to_string(small_float_cap));
  std::printf("stable_attention totals recorded=%llu reused=%llu plan_waits=%llu max_abs_error=%.9g\n",
      static_cast<unsigned long long>(recorded_total), static_cast<unsigned long long>(reused_total),
      static_cast<unsigned long long>(waits_total), maximum_error);
  RecordProperty("stable_attention_recorded_ranges", std::to_string(recorded_total));
  RecordProperty("stable_attention_reused_ranges", std::to_string(reused_total));
  RecordProperty("stable_attention_plan_waits", std::to_string(waits_total));
  RecordProperty("stable_attention_max_abs_error", std::to_string(maximum_error));
  for (auto &slot : slots) { ASSERT_TRUE(slot.close().ok()); slot = {}; }
}
} // namespace
