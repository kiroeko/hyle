#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen38_official_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using BF16 = hyle::simd::bfloat16_t;
using F16 = hyle::simd::float16_t;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

hyle::mem::Buffer bytes_buffer(std::span<const std::byte> bytes,
                               std::size_t alignment = 64) {
  auto result = hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                          bytes.size(), alignment)
                    .value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

hyle::mem::Buffer output_buffer(const c::TensorDesc &description) {
  return hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                   description.required_bytes().value(), 64)
      .value();
}

std::vector<std::byte> read_bytes(const hyle::mem::Buffer &buffer) {
  std::vector<std::byte> result(buffer.size());
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

template <class Compiled> void expect_mature(const Compiled &compiled) {
  ASSERT_GT(compiled.report.step_count(), 0U);
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step) {
    const auto report = compiled.report.step(step).value();
    const auto provider = report.provider.id;
    const auto node = compiled.report.node(report.node_begin).value();
    const bool qualified_norm = provider == c::native::kProviderId &&
        node.kind == c::OperationKind::kRmsNorm && report.node_count == 1 &&
        report.policy.namespace_id == 0x48594c4543505552ULL &&
        report.policy.version == 2 && report.policy.valid();
    EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
                provider == c::onednn::kProviderId ||
                provider == c::aten_cpu::kProviderId ||
                provider == g::onednn::kProviderId || qualified_norm)
        << "step=" << step << " node=" << report.node_begin
        << " operation=" << c::operation_name(node.kind)
        << " provider=" << report.provider.name;
  }
}

std::vector<BF16> decode_fixture(std::span<const std::byte> bytes) {
  EXPECT_EQ(bytes.size() % sizeof(BF16), 0U);
  std::vector<BF16> result(bytes.size() / sizeof(BF16));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

void run_quantized_embedding(
    const c::ModelArtifactProfileDescription &profile, std::int64_t hidden,
    std::int64_t rows, c::GgufQuantizedType type, std::uint64_t region,
    int seed) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t tokens = 2;
  const auto stored =
      c::gguf_weight_description(type, c::DataType::kFloat32, hidden, rows);
  ASSERT_TRUE(stored) << stored.error().message();
  const auto table_desc = stored->transposed().value();
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto output_desc = dense(c::DataType::kFloat32, {tokens, hidden});
  const auto physical = d5_gguf_test::values(rows, hidden, seed);
  const auto packed = d5_gguf_test::pack(
      type, physical, rows, hidden, table_desc.required_bytes().value());
  auto table_buffer = bytes_buffer(packed.bytes);
  auto ids_buffer = output_buffer(ids_desc);
  auto output_buffer_value = output_buffer(output_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto table = builder.constant(table_desc).value();
  const auto ids = builder.input(ids_desc).value();
  const auto embedded = builder.add(c::Gather{}, table, ids).value();
  ASSERT_TRUE(builder.publish(embedded).ok());
  auto logical = std::move(builder).finish().value();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      g::cpu::create_context({.metadata_allocator = stats.get()}).value();
  const std::array constants{g::ConstantBinding{
      table,
      c::ConstTensorView::create(table_desc, table_buffer.view()).value()}};
  const auto workload = d5_qwen35_fixture::workload(
      profile, c::WorkloadPhase::kDecode, region);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  ASSERT_EQ(compiled.report.step_count(), 1U);
  EXPECT_EQ(compiled.report.step(0)->provider.id, c::ggml_cpu::kProviderId);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(ids, c::ConstTensorView::create(ids_desc,
                                                       ids_buffer.view())
                                 .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(embedded,
                        c::TensorView::create(output_desc,
                                              output_buffer_value.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto before = stats.snapshot();
  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 5}}, {{6, 2}}}};
  for (const auto &selection : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selection))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output_buffer_value);
    const auto expected =
        d5_gguf_test::selected_rows(packed.decoded, hidden, selection);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1.0e-6F) << i;
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);

  const auto before_failure = read_bytes(output_buffer_value);
  const std::array<std::int32_t, tokens> invalid{
      0, static_cast<std::int32_t>(rows)};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(output_buffer_value), before_failure);
}

TEST(CpuDefaultGraph,
     D5Qwen35PrimaryGgufQuantizedEmbeddingUsesNativeRowsAndReplays) {
  run_quantized_embedding(
      d5_qwen35_fixture::kPrimaryGgufProfile, d5_qwen35_fixture::kHidden,
      d5_qwen35_fixture::kEmbeddingRows, c::GgufQuantizedType::kQ6K, 0x010f,
      17);
}

TEST(CpuDefaultGraph,
     D5Qwen38PrimaryGgufQuantizedEmbeddingUsesExactWidthAndReplays) {
  run_quantized_embedding(
      d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
      d5_qwen38_official_fixture::kHidden,
      d5_qwen38_official_fixture::kEmbeddingRows,
      c::GgufQuantizedType::kQ4K, 0x010f, 19);
}

TEST(CpuDefaultGraph,
     D5Qwen38FidelityGgufQuantizedEmbeddingUsesExactWidthAndReplays) {
  run_quantized_embedding(
      d5_qwen38_gdn_fixture::kFidelityGgufProfile,
      d5_qwen38_official_fixture::kHidden,
      d5_qwen38_official_fixture::kEmbeddingRows,
      c::GgufQuantizedType::kQ8_0, 0x0108, 23);
}

TEST(CpuDefaultGraph,
     D5Qwen35FidelityGgufF16EmbeddingUsesMatureRowsAndReplays) {
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto rows = d5_qwen35_fixture::kEmbeddingRows;
  constexpr std::int64_t tokens = 2;
  const auto table_desc = dense(c::DataType::kFloat16, {rows, hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto output_desc = dense(c::DataType::kFloat16, {tokens, hidden});
  const auto source = d5_gguf_test::values(rows, hidden, 61);
  std::vector<F16> table_values(source.size());
  std::vector<float> decoded(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    table_values[i] = half_test::encode<F16>(source[i]);
    decoded[i] = static_cast<float>(half_test::decode(table_values[i]));
  }
  auto table_buffer = half_test::buffer<F16>(table_values);
  auto ids_buffer = output_buffer(ids_desc);
  auto output_buffer_value = output_buffer(output_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto table = builder.constant(table_desc).value();
  const auto ids = builder.input(ids_desc).value();
  const auto embedded = builder.add(c::Gather{}, table, ids).value();
  ASSERT_TRUE(builder.publish(embedded).ok());
  auto logical = std::move(builder).finish().value();
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      g::cpu::create_context({.metadata_allocator = stats.get()}).value();
  const std::array constants{g::ConstantBinding{
      table,
      c::ConstTensorView::create(table_desc, table_buffer.view()).value()}};
  const auto workload =
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x0108);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(
      invocation
          .bind(ids,
                c::ConstTensorView::create(ids_desc, ids_buffer.view()).value())
          .ok());
  ASSERT_TRUE(invocation
                  .bind(embedded, c::TensorView::create(
                                      output_desc, output_buffer_value.view())
                                      .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto before = stats.snapshot();
  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 5}}, {{6, 2}}}};
  for (const auto &selection : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selection))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<F16>(output_buffer_value);
    const auto expected =
        d5_gguf_test::selected_rows(decoded, hidden, selection);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);

  const auto before_failure = read_bytes(output_buffer_value);
  constexpr std::array<std::int32_t, tokens> invalid{0, rows};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(output_buffer_value), before_failure);
}

TEST(CpuDefaultGraph,
     D5Qwen35OfficialEmbeddingNormHeadAndFailureUseOnePreparedGraph) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Embedding fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_embedding(*fixture_root));

  constexpr std::int64_t tokens = 2;
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto rows = d5_qwen35_fixture::kEmbeddingRows;
  const auto table_desc = dense(c::DataType::kBFloat16, {rows, hidden});
  const auto head_desc = table_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto token_desc = dense(c::DataType::kBFloat16, {tokens, hidden});
  const auto logits_desc = dense(c::DataType::kBFloat16, {tokens, rows});

  auto table_buffer = bytes_buffer(fixture.rows);
  auto scale_buffer = bytes_buffer(fixture.scale);
  auto ids_buffer = output_buffer(ids_desc);
  auto residual_buffer = output_buffer(token_desc);
  auto logits_buffer = output_buffer(logits_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto table = builder.constant(table_desc).value();
  const auto head = builder.constant(head_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const auto ids = builder.input(ids_desc).value();
  const auto embedded = builder.add(c::Gather{}, table, ids).value();
  const auto normalized =
      builder.add(c::RmsNorm{.epsilon = 1.0e-6F}, embedded, scale).value();
  const auto residual =
      builder.add(c::Binary{c::BinaryKind::kAdd}, embedded, normalized).value();
  const auto logits = builder.add(c::MatMul{}, residual, head).value();
  ASSERT_TRUE(builder.publish(residual).ok());
  ASSERT_TRUE(builder.publish(logits).ok());
  auto logical = std::move(builder).finish().value();

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      g::cpu::create_context({.metadata_allocator = stats.get()}).value();
  const std::array constants{
      g::ConstantBinding{table, c::ConstTensorView::create(
                                   table_desc, table_buffer.view())
                                   .value()},
      g::ConstantBinding{head, c::ConstTensorView::create(
                                  head_desc, table_buffer.view())
                                  .value()},
      g::ConstantBinding{scale, c::ConstTensorView::create(
                                   scale_desc, scale_buffer.view())
                                   .value()}};
  const auto workload = d5_qwen35_fixture::workload(
      c::WorkloadPhase::kDecode, 0x01020f);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(ids, c::ConstTensorView::create(ids_desc,
                                                       ids_buffer.view())
                                 .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(residual, c::TensorView::create(
                                      token_desc, residual_buffer.view())
                                      .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(logits, c::TensorView::create(logits_desc,
                                                       logits_buffer.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());

  const auto table_values = decode_fixture(fixture.rows);
  const auto scale_values = decode_fixture(fixture.scale);
  const std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 1}}, {{5, 6}}}};
  const auto before = stats.snapshot();
  std::vector<std::byte> last_residual;
  std::vector<std::byte> last_logits;
  for (const auto &selected : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_residual = half_test::read<BF16>(residual_buffer);
    const auto actual_logits = half_test::read<BF16>(logits_buffer);
    for (std::size_t token = 0; token < tokens; ++token) {
      const auto source_row = static_cast<std::size_t>(selected[token]);
      std::vector<double> expected_row(hidden);
      double squares = 0;
      for (std::size_t column = 0; column < hidden; ++column) {
        const auto value = half_test::decode(
            table_values[source_row * hidden + column]);
        squares += value * value;
      }
      const auto inverse = 1.0 / std::sqrt(squares / hidden + 1.0e-6);
      for (std::size_t column = 0; column < hidden; ++column) {
        const auto value = half_test::decode(
            table_values[source_row * hidden + column]);
        const auto norm = half_test::decode(half_test::encode<BF16>(
            value * inverse * half_test::decode(scale_values[column])));
        expected_row[column] = half_test::decode(
            half_test::encode<BF16>(value + norm));
        half_test::expect_near(
            actual_residual[token * hidden + column], expected_row[column]);
      }
      for (std::size_t row = 0; row < rows; ++row) {
        double expected = 0;
        for (std::size_t column = 0; column < hidden; ++column)
          expected += expected_row[column] *
                      half_test::decode(table_values[row * hidden + column]);
        half_test::expect_near(actual_logits[token * rows + row], expected);
      }
    }
    last_residual = read_bytes(residual_buffer);
    last_logits = read_bytes(logits_buffer);
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);

  const std::array<std::int32_t, tokens> invalid{0, 7};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(residual_buffer), last_residual);
  EXPECT_EQ(read_bytes(logits_buffer), last_logits);
}

TEST(CpuDefaultGraph,
     D5Qwen35OfficialBfloatFfnUsesMatureRegionsAndRealWeights) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Ffn fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_ffn(*fixture_root));

  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto ffn = d5_qwen35_fixture::kFfnHidden;
  const auto activation_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto stored_gate_desc =
      dense(c::DataType::kBFloat16, {ffn, hidden});
  const auto gate_weight_desc = stored_gate_desc.transposed().value();
  const auto stored_down_desc =
      dense(c::DataType::kBFloat16, {hidden, ffn});
  const auto down_weight_desc = stored_down_desc.transposed().value();

  auto gate_buffer = bytes_buffer(fixture.weights[0]);
  auto up_buffer = bytes_buffer(fixture.weights[1]);
  auto down_buffer = bytes_buffer(fixture.weights[2]);
  auto scale_buffer = bytes_buffer(fixture.norm_scale);
  auto activation_buffer = output_buffer(activation_desc);
  auto output = output_buffer(activation_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto activation = builder.input(activation_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const auto gate_weight = builder.constant(gate_weight_desc).value();
  const auto up_weight = builder.constant(gate_weight_desc).value();
  const auto down_weight = builder.constant(down_weight_desc).value();
  const auto normalized = builder.add(c::RmsNorm{}, activation, scale).value();
  const auto gate = builder.add(c::MatMul{}, normalized, gate_weight).value();
  const auto up = builder.add(c::MatMul{}, normalized, up_weight).value();
  const auto gated =
      builder
          .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate,
               up)
          .value();
  const auto result = builder.add(c::MatMul{}, gated, down_weight).value();
  ASSERT_TRUE(builder.publish(result).ok());
  auto logical = std::move(builder).finish().value();

  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{scale, c::ConstTensorView::create(
                                   scale_desc, scale_buffer.view())
                                   .value()},
      g::ConstantBinding{gate_weight,
                         c::ConstTensorView::create(
                             gate_weight_desc, gate_buffer.view())
                             .value()},
      g::ConstantBinding{up_weight,
                         c::ConstTensorView::create(
                             gate_weight_desc, up_buffer.view())
                             .value()},
      g::ConstantBinding{down_weight,
                         c::ConstTensorView::create(
                             down_weight_desc, down_buffer.view())
                             .value()}};
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 2);
  auto compiled_result =
      g::compile(logical, context, {.constants = constants,
                                    .workload = workload});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(activation,
                        c::ConstTensorView::create(activation_desc,
                                                   activation_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(result,
                        c::TensorView::create(activation_desc, output.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (std::size_t replay = 0; replay < fixture.activations.size(); ++replay) {
    ASSERT_TRUE(activation_buffer.view().write(fixture.activations[replay]).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<BF16>(output);
    ASSERT_EQ(actual.size(), fixture.direct_expected[replay].size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto expected = fixture.direct_expected[replay][i];
      EXPECT_NEAR(half_test::decode(actual[i]), expected,
                  std::abs(expected) * 0.008 + 1.0e-8)
          << replay << ':' << i;
    }
  }
}

TEST(CpuDefaultGraph,
     D5Qwen35OfficialGqaNormRopeAndAttentionUseExactProfileAndReplay) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::AttentionNorms fixture;
  ASSERT_NO_THROW(
      fixture = d5_qwen35_fixture::load_attention_norms(*fixture_root));
  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query_desc =
      dense(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv_desc = dense(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto scale_desc = dense(c::DataType::kBFloat16, {depth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc = dense(c::DataType::kFloat32, {depth / 2});

  auto builder = g::GraphBuilder::create().value();
  const auto query = builder.input(query_desc).value();
  const auto key = builder.input(kv_desc).value();
  const auto value = builder.input(kv_desc).value();
  const auto query_scale = builder.constant(scale_desc).value();
  const auto key_scale = builder.constant(scale_desc).value();
  const auto position = builder.constant(position_desc).value();
  const auto frequency = builder.constant(frequency_desc).value();
  const auto normalized_query =
      builder.add(c::RmsNorm{}, query, query_scale).value();
  const auto normalized_key =
      builder.add(c::RmsNorm{}, key, key_scale).value();
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  const auto rotated_query =
      builder.add(rope, normalized_query, position, frequency).value();
  const auto rotated_key =
      builder.add(rope, normalized_key, position, frequency).value();
  const auto output =
      builder
          .add(c::ScaledDotProductAttention{
                   .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                   .scale = 0.0625F,
                   .causal = true,
                   .accumulation_type = c::DataType::kFloat32},
               rotated_query, rotated_key, value)
          .value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();

  std::vector<float> frequencies(depth / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / static_cast<float>(i + 1);
  const std::array<std::int32_t, 1> positions{0};
  auto query_scale_buffer = bytes_buffer(fixture.query);
  auto key_scale_buffer = bytes_buffer(fixture.key);
  auto position_buffer = half_test::buffer<std::int32_t>(positions);
  auto frequency_buffer = half_test::buffer<float>(frequencies);
  std::vector<BF16> queries(query_desc.element_count().value());
  std::vector<BF16> keys(kv_desc.element_count().value());
  std::vector<BF16> values(kv_desc.element_count().value());
  auto query_buffer = half_test::buffer<BF16>(queries);
  auto key_buffer = half_test::buffer<BF16>(keys);
  auto value_buffer = half_test::buffer<BF16>(values);
  auto result_buffer = output_buffer(query_desc);

  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{query_scale,
                         c::ConstTensorView::create(
                             scale_desc, query_scale_buffer.view())
                             .value()},
      g::ConstantBinding{key_scale,
                         c::ConstTensorView::create(
                             scale_desc, key_scale_buffer.view())
                             .value()},
      g::ConstantBinding{position, c::ConstTensorView::create(
                                      position_desc, position_buffer.view())
                                      .value()},
      g::ConstantBinding{frequency, c::ConstTensorView::create(
                                       frequency_desc, frequency_buffer.view())
                                       .value()}};
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 4);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(query, c::ConstTensorView::create(
                                   query_desc, query_buffer.view())
                                   .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(key,
                        c::ConstTensorView::create(kv_desc, key_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(value, c::ConstTensorView::create(
                                   kv_desc, value_buffer.view())
                                   .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(output, c::TensorView::create(
                                    query_desc, result_buffer.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int seed : {1, 7}) {
    for (std::size_t i = 0; i < queries.size(); ++i)
      queries[i] = half_test::encode<BF16>(
          static_cast<double>(static_cast<int>((i + seed) % 31) - 15) / 64);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = half_test::encode<BF16>(
          static_cast<double>(static_cast<int>((i + seed) % 29) - 14) / 64);
      values[i] = half_test::encode<BF16>(
          static_cast<double>(seed * 100 + static_cast<int>(i)) / 256);
    }
    ASSERT_TRUE(query_buffer.view()
                    .write(std::as_bytes(std::span(queries)))
                    .ok());
    ASSERT_TRUE(
        key_buffer.view().write(std::as_bytes(std::span(keys))).ok());
    ASSERT_TRUE(
        value_buffer.view().write(std::as_bytes(std::span(values))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<BF16>(result_buffer);
    for (std::size_t head = 0; head < query_heads; ++head)
      for (std::size_t column = 0; column < depth; ++column)
        EXPECT_EQ(actual[head * depth + column].bits,
                  values[(head / (query_heads / kv_heads)) * depth + column]
                      .bits);
  }
}

TEST(CpuDefaultGraph,
     D5Qwen35OfficialCausalConvAndGdnUseExactProfileAndReplay) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Recurrent fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_recurrent(*fixture_root));

  constexpr auto heads = d5_qwen35_fixture::kGdnHeads;
  constexpr auto depth = d5_qwen35_fixture::kGdnDepth;
  constexpr auto channels = d5_qwen35_fixture::kConvChannels;
  constexpr auto kernel = d5_qwen35_fixture::kConvKernel;
  const auto conv_samples_desc =
      dense(c::DataType::kFloat32, {1, kernel, channels});
  const auto conv_weight_desc =
      dense(c::DataType::kFloat32, {channels, kernel});
  const auto conv_output_desc = dense(c::DataType::kFloat32, {1, 1, channels});
  const auto qkv_desc = dense(c::DataType::kFloat32, {1, 1, heads, depth});
  const auto gate_desc = dense(c::DataType::kFloat32, {1, 1, heads, 1});
  const auto state_desc =
      dense(c::DataType::kFloat32, {1, heads, depth, depth});

  const auto conv_weight_bf16 = decode_fixture(fixture.convolution_weight);
  std::vector<float> conv_weights(conv_weight_bf16.size());
  for (std::size_t i = 0; i < conv_weights.size(); ++i)
    conv_weights[i] =
        static_cast<float>(half_test::decode(conv_weight_bf16[i]));
  const auto time_bias_bf16 = decode_fixture(fixture.time_bias);
  std::vector<float> log_decay(heads), beta(heads);
  for (std::size_t head = 0; head < heads; ++head) {
    const auto bias = half_test::decode(time_bias_bf16[head]);
    const auto softplus =
        bias > 0 ? bias + std::log1p(std::exp(-bias))
                 : std::log1p(std::exp(bias));
    log_decay[head] = static_cast<float>(
        -std::exp(static_cast<double>(fixture.log_decay[head])) * softplus);
    beta[head] = static_cast<float>(1.0 / (1.0 + std::exp(-bias)));
  }

  auto builder = g::GraphBuilder::create().value();
  const auto conv_samples = builder.input(conv_samples_desc).value();
  const auto conv_weight = builder.constant(conv_weight_desc).value();
  const auto query = builder.input(qkv_desc).value();
  const auto key = builder.input(qkv_desc).value();
  const auto value = builder.input(qkv_desc).value();
  const auto decay = builder.constant(gate_desc).value();
  const auto update_gate = builder.constant(gate_desc).value();
  const auto old_state = builder.input(state_desc).value();
  const auto convolution =
      builder
          .add(c::model_ops::causal_conv1d(), conv_samples, conv_weight)
          .value();
  const std::array gdn_inputs{query, key, value, decay, update_gate, old_state};
  const auto gdn =
      builder.add_node(c::Operation(c::GatedDeltaNet{}), gdn_inputs).value();
  const std::array gdn_outputs{builder.result(gdn, 0).value(),
                               builder.result(gdn, 1).value()};
  ASSERT_TRUE(builder.publish(convolution).ok());
  ASSERT_TRUE(builder.publish(gdn_outputs[0]).ok());
  ASSERT_TRUE(builder.publish(gdn_outputs[1]).ok());
  auto logical = std::move(builder).finish().value();

  auto conv_weight_buffer = half_test::buffer<float>(conv_weights);
  auto decay_buffer = half_test::buffer<float>(log_decay);
  auto beta_buffer = half_test::buffer<float>(beta);
  auto conv_samples_buffer = output_buffer(conv_samples_desc);
  std::array qkv_buffers{output_buffer(qkv_desc), output_buffer(qkv_desc),
                         output_buffer(qkv_desc)};
  auto state_buffer = output_buffer(state_desc);
  auto conv_result_buffer = output_buffer(conv_output_desc);
  auto score_buffer = output_buffer(qkv_desc);
  auto new_state_buffer = output_buffer(state_desc);
  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{conv_weight,
                         c::ConstTensorView::create(
                             conv_weight_desc, conv_weight_buffer.view())
                             .value()},
      g::ConstantBinding{decay, c::ConstTensorView::create(
                                    gate_desc, decay_buffer.view())
                                    .value()},
      g::ConstantBinding{update_gate,
                         c::ConstTensorView::create(gate_desc,
                                                    beta_buffer.view())
                             .value()}};
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 5);
  auto compiled_result =
      g::compile(logical, context, {.constants = constants,
                                    .workload = workload});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(conv_samples,
                        c::ConstTensorView::create(
                            conv_samples_desc, conv_samples_buffer.view())
                            .value())
                  .ok());
  for (std::size_t port = 0; port < qkv_buffers.size(); ++port)
    ASSERT_TRUE(invocation
                    .bind(gdn_inputs[port],
                          c::ConstTensorView::create(
                              qkv_desc, qkv_buffers[port].view())
                              .value())
                    .ok());
  ASSERT_TRUE(invocation
                  .bind(old_state,
                        c::ConstTensorView::create(state_desc,
                                                   state_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(convolution,
                        c::TensorView::create(conv_output_desc,
                                              conv_result_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(gdn_outputs[0],
                        c::TensorView::create(qkv_desc, score_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(gdn_outputs[1],
                        c::TensorView::create(state_desc,
                                              new_state_buffer.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());

  for (int seed : {2, 9}) {
    std::vector<float> conv_samples_values(
        conv_samples_desc.element_count().value());
    std::array<std::vector<float>, 3> qkv_values;
    for (auto &values : qkv_values)
      values.resize(qkv_desc.element_count().value());
    std::vector<float> state_values(state_desc.element_count().value());
    for (std::size_t i = 0; i < conv_samples_values.size(); ++i)
      conv_samples_values[i] = static_cast<float>(
          static_cast<int>((i * 17 + seed * 5) % 61) - 30) /
          128.0F;
    for (std::size_t port = 0; port < qkv_values.size(); ++port)
      for (std::size_t i = 0; i < qkv_values[port].size(); ++i)
        qkv_values[port][i] = static_cast<float>(std::sin(
            static_cast<double>(i + port * 37 + seed) * 0.017)) *
                              0.025F;
    for (std::size_t i = 0; i < state_values.size(); ++i)
      state_values[i] = static_cast<float>(std::sin(
          static_cast<double>(i + seed * 19) * 0.009)) *
                        0.0005F;
    ASSERT_TRUE(conv_samples_buffer.view()
                    .write(std::as_bytes(std::span(conv_samples_values)))
                    .ok());
    for (std::size_t port = 0; port < qkv_buffers.size(); ++port)
      ASSERT_TRUE(qkv_buffers[port]
                      .view()
                      .write(std::as_bytes(std::span(qkv_values[port])))
                      .ok());
    ASSERT_TRUE(state_buffer.view()
                    .write(std::as_bytes(std::span(state_values)))
                    .ok());

    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_conv = half_test::read<float>(conv_result_buffer);
    const auto actual_scores = half_test::read<float>(score_buffer);
    const auto actual_state = half_test::read<float>(new_state_buffer);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      double expected = 0;
      for (std::size_t tap = 0; tap < kernel; ++tap)
        expected += conv_samples_values[tap * channels + channel] *
                    conv_weights[channel * kernel + tap];
      EXPECT_NEAR(actual_conv[channel], expected,
                  std::abs(expected) * 2.0e-5 + 1.0e-6)
          << channel;
    }

    std::vector<double> expected_state(state_values.begin(),
                                       state_values.end());
    std::vector<double> expected_scores(actual_scores.size());
    for (std::size_t head = 0; head < heads; ++head) {
      const auto vector = head * depth;
      const auto matrix = head * depth * depth;
      const auto retain = std::exp(static_cast<double>(log_decay[head]));
      for (std::size_t column = 0; column < depth; ++column) {
        const auto row = matrix + column * depth;
        double projected = 0;
        for (std::size_t key_column = 0; key_column < depth; ++key_column) {
          expected_state[row + key_column] *= retain;
          projected += expected_state[row + key_column] *
                       qkv_values[1][vector + key_column];
        }
        const auto delta =
            static_cast<double>(beta[head]) *
            (qkv_values[2][vector + column] - projected);
        double score = 0;
        for (std::size_t key_column = 0; key_column < depth; ++key_column) {
          expected_state[row + key_column] +=
              delta * qkv_values[1][vector + key_column];
          score += expected_state[row + key_column] *
                   qkv_values[0][vector + key_column];
        }
        expected_scores[vector + column] = score / std::sqrt(double(depth));
      }
    }
    for (std::size_t i = 0; i < actual_scores.size(); ++i)
      EXPECT_NEAR(actual_scores[i], expected_scores[i], 4.0e-6) << i;
    for (std::size_t i = 0; i < actual_state.size(); ++i)
      EXPECT_NEAR(actual_state[i], expected_state[i], 4.0e-6) << i;
  }
}

TEST(CpuDefaultGraph,
     D5Qwen35OfficialMtpAndBfloatBoundaryUseExactWeightsAndReplay) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::MtpProjection fixture;
  d5_qwen35_fixture::Embedding embedding;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_mtp(*fixture_root));
  ASSERT_NO_THROW(embedding =
                      d5_qwen35_fixture::load_embedding(*fixture_root));

  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto input = d5_qwen35_fixture::kMtpInput;
  const auto half_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto stored_weight_desc =
      dense(c::DataType::kBFloat16, {hidden, input});
  const auto weight_desc = stored_weight_desc.transposed().value();
  const auto output_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto output_f32_desc = dense(c::DataType::kFloat32, {1, hidden});

  auto builder = g::GraphBuilder::create().value();
  const auto first = builder.input(half_desc).value();
  const auto second = builder.input(half_desc).value();
  const auto weight = builder.constant(weight_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const std::array halves{first, second};
  const auto concat_node = builder.add_node(c::Operation(c::Concat{.axis = 1}),
                                             halves)
                               .value();
  const auto joined = builder.result(concat_node).value();
  const auto projected = builder.add(c::MatMul{}, joined, weight).value();
  const auto normalized = builder.add(c::RmsNorm{}, projected, scale).value();
  const auto converted =
      builder.add(c::Cast{c::DataType::kFloat32}, normalized).value();
  ASSERT_TRUE(builder.publish(projected).ok());
  ASSERT_TRUE(builder.publish(converted).ok());
  auto logical = std::move(builder).finish().value();

  auto weight_buffer = bytes_buffer(fixture.weight);
  auto scale_buffer = bytes_buffer(embedding.scale);
  auto first_buffer = output_buffer(half_desc);
  auto second_buffer = output_buffer(half_desc);
  auto projected_buffer = output_buffer(output_desc);
  auto converted_buffer = output_buffer(output_f32_desc);
  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{weight, c::ConstTensorView::create(
                                    weight_desc, weight_buffer.view())
                                    .value()},
      g::ConstantBinding{scale, c::ConstTensorView::create(
                                   scale_desc, scale_buffer.view())
                                   .value()}};
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0x0d0e);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(first, c::ConstTensorView::create(
                                   half_desc, first_buffer.view())
                                   .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(second, c::ConstTensorView::create(
                                    half_desc, second_buffer.view())
                                    .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(projected,
                        c::TensorView::create(output_desc,
                                              projected_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(converted,
                        c::TensorView::create(output_f32_desc,
                                              converted_buffer.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());

  const auto scales = decode_fixture(embedding.scale);
  for (std::size_t replay = 0; replay < fixture.activations.size(); ++replay) {
    const auto &activation = fixture.activations[replay];
    const auto split = static_cast<std::size_t>(hidden * sizeof(BF16));
    ASSERT_TRUE(first_buffer.view().write(
        std::span<const std::byte>(activation.data(), split)).ok());
    ASSERT_TRUE(second_buffer.view().write(
        std::span<const std::byte>(activation.data() + split,
                                   activation.size() - split)).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto projected_values = half_test::read<BF16>(projected_buffer);
    const auto converted_values = half_test::read<float>(converted_buffer);
    double squares = 0;
    for (std::size_t i = 0; i < projected_values.size(); ++i) {
      half_test::expect_near(projected_values[i], fixture.expected[replay][i]);
      const auto value = half_test::decode(projected_values[i]);
      squares += value * value;
    }
    const auto inverse = 1.0 / std::sqrt(squares / hidden + 1.0e-5);
    for (std::size_t i = 0; i < converted_values.size(); ++i) {
      const auto expected = half_test::decode(half_test::encode<BF16>(
          half_test::decode(projected_values[i]) * inverse *
          half_test::decode(scales[i])));
      EXPECT_NEAR(converted_values[i], expected,
                  std::abs(expected) * 0.008 + 1.0e-6);
    }
  }
}
} // namespace
