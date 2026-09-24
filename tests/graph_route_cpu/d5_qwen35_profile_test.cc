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

#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using BF16 = hyle::simd::bfloat16_t;

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
    EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
                provider == c::onednn::kProviderId)
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
     D5Qwen35OfficialGqaNormRopeAndAttentionUseExactProfileAndReplay) {
  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query_desc =
      dense(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv_desc = dense(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto scale_desc = dense(c::DataType::kFloat32, {depth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc = dense(c::DataType::kFloat32, {depth / 2});

  auto builder = g::GraphBuilder::create().value();
  const auto query = builder.input(query_desc).value();
  const auto key = builder.input(kv_desc).value();
  const auto value = builder.input(kv_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const auto position = builder.constant(position_desc).value();
  const auto frequency = builder.constant(frequency_desc).value();
  const auto normalized_query = builder.add(c::RmsNorm{}, query, scale).value();
  const auto normalized_key = builder.add(c::RmsNorm{}, key, scale).value();
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

  std::vector<float> scales(depth, 1.0F);
  std::vector<float> frequencies(depth / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / static_cast<float>(i + 1);
  const std::array<std::int32_t, 1> positions{0};
  auto scale_buffer = half_test::buffer<float>(scales);
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
      g::ConstantBinding{scale, c::ConstTensorView::create(
                                   scale_desc, scale_buffer.view())
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
  const auto joined_desc = dense(c::DataType::kBFloat16, {1, input});
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
