#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/sequence_ops.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/recording_access.h"
#include "hyle/graph/detail/region_access.h"
#include "../../src/graph/plan_state.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gb = g::batch;
namespace gd = g::detail;
namespace m = hyle::mem;
constexpr std::uint32_t rows = 128, depth = 64;
#define ASSERT_ROWS_OK(expression) do { \
  const auto rows_status = (expression); \
  ASSERT_TRUE(rows_status.ok()) << #expression << ": " << rows_status.message(); \
} while (false)

struct PackedRowsFixture final {
  c::Context context = g::cuda::create_context().value();
  c::Queue queue = context.create_queue().value();
  c::StorageOwner storage = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  c::cuda::MemoryResource host = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kPinnedHost).value();
  c::BatchCapacity capacity;
  c::BatchSchema schema;
  g::Graph graph;
  g::CompiledGraph compiled;
  std::array<g::ValueId, 7> inputs;
  g::NodeId attention, matmul;
  g::ValueId attention_output, output;
  std::array<c::TensorDesc, 7> descriptions;
  c::TensorDesc attention_description, output_description;
  // Every borrowed caller buffer outlives the original Invocation/native use.
  std::array<m::Buffer, 6> payload;
  m::Buffer attention_buffer, output_buffer, raw_offsets;
  g::Invocation run;
  c::DataType source_type;

  explicit PackedRowsFixture(bool transposed = false,
      c::DataType type = c::DataType::kFloat16) : source_type(type) {
    EXPECT_TRUE(queue.reserve_completions(16).ok());
    const auto plain = c::CacheStorage::plain(c::DataType::kFloat16).value();
    capacity = {.storage = c::BatchStorage::kRagged,
        .query_type = c::DataType::kFloat16, .output_type = c::DataType::kFloat16,
        .key_storage = plain, .value_storage = plain, .max_requests = 1,
        .max_query_rows = rows, .max_kv_rows = rows,
        .max_pages = 0, .max_page_refs = 0, .query_heads = 1, .kv_heads = 1,
        .qk_head_dim = depth, .value_head_dim = depth, .page_size = 1};
    schema = c::BatchSchema::create(capacity).value();
    descriptions[0] = c::batch_query_description(capacity).value();
    descriptions[1] = c::batch_cache_description(capacity, c::BatchCacheRole::kKey).value();
    descriptions[2] = c::batch_cache_description(capacity, c::BatchCacheRole::kValue).value();
    descriptions[3] = schema.metadata_description().value();
    descriptions[4] = c::TensorDesc::dense(source_type,
        std::array<std::int64_t, 2>{rows, depth}).value();
    descriptions[5] = transposed
        ? c::TensorDesc::strided_bytes(source_type,
            std::array<std::int64_t, 2>{depth, depth},
            std::array<std::int64_t, 2>{2, depth * 2}).value()
        : c::TensorDesc::dense(source_type,
            std::array<std::int64_t, 2>{depth, depth}).value();
    descriptions[6] = c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{2}).value();
    auto builder = g::GraphBuilder::create().value();
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] = builder.input(descriptions[i]).value();
    attention = builder.add_node(c::Operation::custom(c::RaggedAttention{
        .capacity = capacity, .scale = .125F, .causal = false}),
        std::span(inputs).first(4)).value();
    attention_output = builder.result(attention).value();
    matmul = builder.add_node(c::sequence_ops::packed_row_matmul(c::DataType::kFloat32),
        std::array{inputs[4], inputs[5], inputs[6]}).value();
    // This is the real attention producer in the same original Graph episode.
    EXPECT_TRUE(builder.depends_on(matmul, attention).ok());
    output = builder.result(matmul).value();
    EXPECT_TRUE(builder.publish(attention_output).ok());
    EXPECT_TRUE(builder.publish(output).ok());
    graph = std::move(builder).finish().value();
    compiled = g::cuda::compile(graph, context, storage,
        {.operation_fallback = g::OperationFallbackPolicy::kAllow}).value();
    // The bounded row primitive must retain the original matrix provider;
    // mathematical correctness alone must not hide a backend replacement.
    const auto *plan = gd::ExecutableAccess::state(compiled.executable);
    const auto step = plan->node_steps[matmul.ordinal()];
    EXPECT_EQ(plan->steps[step].kind, gd::PlanStepKind::kOperation);
    EXPECT_EQ(plan->steps[step].operation.provider().id, c::cublas::kProviderId);
    attention_description = c::batch_output_description(capacity, rows).value();
    output_description = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{rows, depth}).value();
    run = compiled.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    for (std::size_t port : {0u, 1u, 2u, 4u, 5u}) {
      payload[port] = m::Buffer::create(storage.ref(),
          descriptions[port].required_bytes().value(), 256).value();
      EXPECT_TRUE(payload[port].view().fill(std::byte{0}).ok());
      EXPECT_TRUE(run.bind(inputs[port], c::ConstTensorView::create(
          descriptions[port], payload[port].view()).value()).ok());
    }
    std::vector<std::uint16_t> values(rows * depth, 0x4000u); // V = 2.
    EXPECT_TRUE(payload[2].view().write(std::as_bytes(std::span(values))).ok());
    std::vector<std::uint16_t> identity(depth * depth);
    for (std::size_t i = 0; i < depth; ++i)
      identity[i * depth + i] = source_type == c::DataType::kFloat16 ? 0x3c00u : 0x3f80u;
    EXPECT_TRUE(payload[5].view().write(std::as_bytes(std::span(identity))).ok());
    attention_buffer = m::Buffer::create(storage.ref(),
        attention_description.required_bytes().value(), 256).value();
    output_buffer = m::Buffer::create(storage.ref(),
        output_description.required_bytes().value(), 256).value();
    EXPECT_TRUE(run.bind(attention_output, c::TensorView::create(
        attention_description, attention_buffer.view()).value()).ok());
    EXPECT_TRUE(run.bind(output, c::TensorView::create(
        output_description, output_buffer.view()).value()).ok());
  }
  ~PackedRowsFixture() {
    const auto retired = context.drain();
    EXPECT_TRUE(retired.ok()) << retired.message();
  }
  auto queues() { return std::array{g::QueueBinding{g::ExecutionLaneId{0}, &queue}}; }
  c::BatchLayoutVersion layout(std::uint32_t active) {
    const std::array<std::int32_t, 2> q{0, std::int32_t(active)}, kv{0, rows};
    return schema.version({q, kv, {}, {}}).value();
  }
  hyle::Status prepare(const c::BatchLayoutVersion &layout) {
    const std::array layouts{gb::NodeLayout{attention, layout}};
    return gb::prepare(run, queues(), layouts, {.host_workspace = host.ref()});
  }
  hyle::Status bind_rows() { return gb::bind_query_rows(run, attention, inputs[6]); }
  void upload(std::uint32_t active) {
    // Finite prefix, NaN padding: multiplying padding even by zero is visible.
    std::vector<std::uint16_t> values(rows * depth,
        source_type == c::DataType::kFloat16 ? 0x7e01u : 0x7fc1u);
    std::fill_n(values.begin(), std::size_t(active) * depth, std::uint16_t{0x4000});
    ASSERT_TRUE(payload[4].view().write(std::as_bytes(std::span(values))).ok());
  }
  void poison() {
    ASSERT_TRUE(output_buffer.view().fill(std::byte{0x5a}).ok());
    ASSERT_TRUE(attention_buffer.view().fill(std::byte{0x5a}).ok());
  }
  void expect_poison() {
    for (auto view : {output_buffer.view(), attention_buffer.view()}) {
      std::vector<std::byte> actual(view.size());
      ASSERT_TRUE(view.read(actual).ok());
      EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](auto x) { return x == std::byte{0x5a}; }));
    }
  }
  void check(std::uint32_t active) {
    std::vector<float> actual(rows * depth);
    ASSERT_TRUE(output_buffer.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
          i < std::size_t(active) * depth ? 0x40000000u : 0u) << i;
    std::vector<std::uint16_t> attended(rows * depth);
    ASSERT_TRUE(attention_buffer.view().read(std::as_writable_bytes(std::span(attended))).ok());
    for (std::size_t i = 0; i < attended.size(); ++i)
      EXPECT_EQ(attended[i], i < std::size_t(active) * depth ? 0x4000u : 0u) << i;
  }
  hyle::Status execute(std::uint32_t active) {
    HYLE_RETURN_IF_ERROR(run.prepare_replay(queues(), {.mode = g::ReplayMode::kCommands}));
    HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queues()));
    HYLE_RETURN_IF_ERROR(done.wait()); done = {};
    check(active);
    return hyle::Status::Ok();
  }
};

TEST(GraphBatchPackedRowMatMulCuda, SameCarrierUsesOwnedActualRowsForDenseAndTransposedWeights) {
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
  SCOPED_TRACE(static_cast<int>(type));
  for (bool transposed : {false, true}) {
    SCOPED_TRACE(transposed);
    PackedRowsFixture f(transposed, type);
    m::ConstBufferView previous;
    std::uint64_t generation = 0;
    for (std::uint32_t active : {0u, 1u, 17u, 128u, 1u}) {
      SCOPED_TRACE(active);
      const auto layout = f.layout(active);
      ASSERT_NE(layout.generation(), generation); generation = layout.generation();
      ASSERT_ROWS_OK(f.prepare(layout)); f.upload(active); f.poison();
      ASSERT_ROWS_OK(f.bind_rows());
      {
        const auto offsets = gb::query_offsets(f.run, f.attention).value();
        const auto current = offsets.view().value();
        if (previous.valid()) {
          EXPECT_EQ(previous.resource(), current.resource());
          EXPECT_EQ(previous.native_handle(), current.native_handle());
          EXPECT_EQ(previous.offset(), current.offset()); EXPECT_EQ(previous.size(), current.size());
        }
        previous = current;
      }
      ASSERT_ROWS_OK(f.execute(active));
    }
  }
  }
}

TEST(GraphBatchPackedRowMatMulCuda, RawOffsetsAndResetCannotReuseAnOldHostRowsConfiguration) {
  PackedRowsFixture f;
  const auto layout = f.layout(17);
  ASSERT_ROWS_OK(f.prepare(layout)); f.upload(17); f.poison();
  EXPECT_FALSE(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kCommands}).ok());
  f.raw_offsets = m::Buffer::create(f.storage.ref(), 2 * sizeof(std::int32_t), 256).value();
  const std::array<std::int32_t, 2> wrong{0, 1};
  ASSERT_ROWS_OK(f.raw_offsets.view().write(std::as_bytes(std::span(wrong))));
  ASSERT_ROWS_OK(f.run.bind(f.inputs[6], c::ConstTensorView::create(f.descriptions[6], f.raw_offsets.view()).value()));
  EXPECT_FALSE(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kCommands}).ok());
  EXPECT_FALSE(f.run.submit(f.queues())); f.expect_poison();
  ASSERT_ROWS_OK(f.bind_rows()); ASSERT_ROWS_OK(f.execute(17));
  // Reuse the SAME immutable layout generation; the original Batch prepare
  // still creates a different real epoch and clears every old consumer.
  ASSERT_ROWS_OK(gd::RegionAccess::reset_configuration(f.run));
  EXPECT_FALSE(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kCommands}).ok());
  ASSERT_ROWS_OK(f.prepare(layout)); f.upload(17); f.poison();
  {
    auto offsets = gb::query_offsets(f.run, f.attention).value();
    ASSERT_ROWS_OK(f.run.bind(f.inputs[6], f.descriptions[6], offsets));
  }
  EXPECT_FALSE(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kCommands}).ok());
  EXPECT_FALSE(f.run.submit(f.queues())); f.expect_poison();
  ASSERT_ROWS_OK(f.bind_rows()); ASSERT_ROWS_OK(f.execute(17));
}

TEST(GraphBatchPackedRowMatMulCuda, RecordingOnlyNeverExecutesOrManufacturesRowsControl) {
  PackedRowsFixture f;
  ASSERT_ROWS_OK(f.prepare(f.layout(17))); f.upload(17); f.poison();
  ASSERT_ROWS_OK(gd::RecordingAccess::begin(f.run));
  const auto bound = f.bind_rows(); EXPECT_TRUE(bound.IsUnimplemented()) << bound.message();
  EXPECT_FALSE(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kCommands}).ok());
  EXPECT_FALSE(f.run.submit(f.queues()));
  ASSERT_ROWS_OK(gd::RecordingAccess::end(f.run)); f.expect_poison();
  ASSERT_ROWS_OK(f.bind_rows()); ASSERT_ROWS_OK(f.execute(17));
}

TEST(GraphBatchPackedRowMatMulCuda, ResetReusesCaptureOnlyAfterFreshRowsProofAndBindings) {
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    PackedRowsFixture f(true, type);
    const auto execute_capture = [&](std::uint32_t active) {
      const auto prepared = f.run.prepare_replay(f.queues(),
          {.mode = g::ReplayMode::kPreferCapture});
      EXPECT_TRUE(prepared.ok()) << prepared.message();
      if (!prepared.ok()) return g::ReplayReport{};
      const auto report = f.run.replay_report().value();
      auto submitted = f.run.submit(f.queues());
      EXPECT_TRUE(submitted.has_value());
      if (submitted) {
        const auto retired = submitted->wait();
        EXPECT_TRUE(retired.ok()) << retired.message();
      }
      f.check(active);
      return report;
    };
    ASSERT_ROWS_OK(f.prepare(f.layout(17))); f.upload(17); f.poison();
    ASSERT_ROWS_OK(f.bind_rows());
    const auto first = execute_capture(17);
    ASSERT_GT(first.recorded_capture_range_count, 0U);

    // The same row count has a NEW Batch epoch. Old native commands may stay
    // dormant, but a raw rebind of the genuine offsets cannot restore proof.
    ASSERT_ROWS_OK(f.prepare(f.layout(17))); f.upload(17); f.poison();
    {
      const auto offsets = gb::query_offsets(f.run, f.attention).value();
      ASSERT_ROWS_OK(f.run.bind(f.inputs[6], f.descriptions[6], offsets));
    }
    EXPECT_FALSE(f.run.prepare_replay(f.queues(),
        {.mode = g::ReplayMode::kPreferCapture}).ok());
    EXPECT_FALSE(f.run.submit(f.queues())); f.expect_poison();
    ASSERT_ROWS_OK(f.bind_rows());
    const auto second = execute_capture(17);
    EXPECT_EQ(second.recorded_capture_range_count, 0U);
    EXPECT_GT(second.reused_capture_range_count, 0U);

    // Changing the native M must discard its old capture before replacing Lt
    // descriptors. NaN padding detects accidental use of the old 17-row M.
    ASSERT_ROWS_OK(f.prepare(f.layout(1))); f.upload(1); f.poison();
    ASSERT_ROWS_OK(f.bind_rows());
    const auto changed = execute_capture(1);
    EXPECT_GT(changed.recorded_capture_range_count, 0U);
  }
}

TEST(GraphBatchPackedRowMatMulCuda, FixedFullRowsCanRecordWithoutExecutingOrAdvancingState) {
  PackedRowsFixture f;
  ASSERT_ROWS_OK(f.prepare(f.layout(rows))); f.upload(rows); f.poison();
  ASSERT_ROWS_OK(gd::RecordingAccess::begin(f.run));
  ASSERT_ROWS_OK(f.bind_rows());
  ASSERT_ROWS_OK(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kPreferCapture}));
  EXPECT_GT(f.run.replay_report()->recorded_capture_range_count, 0U);
  EXPECT_FALSE(f.run.submit(f.queues()));
  f.expect_poison();
  ASSERT_ROWS_OK(gd::RecordingAccess::end(f.run));
  ASSERT_ROWS_OK(f.prepare(f.layout(rows))); f.upload(rows); f.poison();
  ASSERT_ROWS_OK(f.bind_rows());
  ASSERT_ROWS_OK(f.run.prepare_replay(f.queues(), {.mode = g::ReplayMode::kPreferCapture}));
  EXPECT_EQ(f.run.replay_report()->recorded_capture_range_count, 0U);
  EXPECT_GT(f.run.replay_report()->reused_capture_range_count, 0U);
  auto submitted = f.run.submit(f.queues()).value();
  ASSERT_ROWS_OK(submitted.wait()); f.check(rows);
}
#undef ASSERT_ROWS_OK
} // namespace
