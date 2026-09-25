#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "../compute_models/d5_qwen38_24t_fixture.h"
#include "../compute_models/d5_qwen38_official_fixture.h"
#include "../compute_models/d5_qwen38_profile_cases.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace q = d5_qwen38_official_fixture;
namespace q24 = d5_qwen38_24t_fixture;
namespace qcase = d5_qwen38_profile_cases;
using BF16 = hyle::simd::bfloat16_t;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
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
    if (node.kind == c::OperationKind::kView && report.provider.name.empty())
      continue;
    EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
                provider == c::onednn::kProviderId ||
                provider == g::onednn::kProviderId)
        << "step=" << step << " node=" << report.node_begin
        << " operation=" << c::operation_name(node.kind)
        << " provider=" << report.provider.name;
  }
}

void expect_near(std::span<const BF16> actual,
                 std::span<const double> expected, double relative = 0.012) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(q::decode(actual[i]), expected[i],
                std::abs(expected[i]) * relative + 2.0e-5)
        << i;
}

class CpuGraphQwen38Profiles
    : public ::testing::TestWithParam<const qcase::ProfileCase *> {};

TEST_P(CpuGraphQwen38Profiles,
     D5Qwen38ProfileBoundaryProjectionUsesExactHiddenAndReplays) {
  const auto &shape = *GetParam();
  constexpr std::int64_t tokens = 2;
  constexpr int projection_seed = 31;
  const auto table_desc =
      dense(c::DataType::kBFloat16, {q::kEmbeddingRows, shape.hidden});
  const auto head_desc = table_desc.transposed().value();
  const auto stored_projection_desc =
      dense(c::DataType::kBFloat16, {shape.hidden, shape.hidden});
  const auto projection_desc = stored_projection_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {shape.hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto token_desc =
      dense(c::DataType::kBFloat16, {tokens, shape.hidden});
  const auto logits_desc =
      dense(c::DataType::kBFloat16, {tokens, q::kEmbeddingRows});

  auto table_values = q::values(q::kEmbeddingRows * shape.hidden, 5);
  auto head_values = q::values(q::kEmbeddingRows * shape.hidden, 7, 96.0);
  auto scale_values = q::scales(shape.hidden, 3);
  auto projection_values = q::sparse_output_major_weight(
      shape.hidden, shape.hidden, projection_seed);
  auto table_buffer = half_test::buffer<BF16>(table_values);
  auto head_buffer = half_test::buffer<BF16>(head_values);
  auto scale_buffer = half_test::buffer<BF16>(scale_values);
  auto projection_buffer = half_test::buffer<BF16>(projection_values);
  projection_values.clear();
  projection_values.shrink_to_fit();
  auto ids_buffer = output_buffer(ids_desc);
  auto residual_buffer = output_buffer(token_desc);
  auto projected_buffer = output_buffer(token_desc);
  auto logits_buffer = output_buffer(logits_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto table = builder.constant(table_desc).value();
  const auto head = builder.constant(head_desc).value();
  const auto projection_weight = builder.constant(projection_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const auto ids = builder.input(ids_desc).value();
  const auto embedded = builder.add(c::Gather{}, table, ids).value();
  const auto normalized =
      builder.add(c::RmsNorm{.epsilon = 1.0e-6F}, embedded, scale).value();
  const auto residual =
      builder.add(c::Binary{c::BinaryKind::kAdd}, embedded, normalized).value();
  const auto projected =
      builder.add(c::MatMul{}, residual, projection_weight).value();
  const auto logits = builder.add(c::MatMul{}, projected, head).value();
  ASSERT_TRUE(builder.publish(residual).ok());
  ASSERT_TRUE(builder.publish(projected).ok());
  ASSERT_TRUE(builder.publish(logits).ok());
  auto logical = std::move(builder).finish().value();

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      g::cpu::create_context({.metadata_allocator = stats.get()}).value();
  const std::array constants{
      g::ConstantBinding{
          table,
          c::ConstTensorView::create(table_desc, table_buffer.view()).value()},
      g::ConstantBinding{
          head, c::ConstTensorView::create(head_desc, head_buffer.view()).value()},
      g::ConstantBinding{projection_weight,
                         c::ConstTensorView::create(
                             projection_desc, projection_buffer.view())
                             .value()},
      g::ConstantBinding{
          scale,
          c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()}};
  const auto workload = q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x01030e0f);
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
                  .bind(projected, c::TensorView::create(
                                       token_desc, projected_buffer.view())
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

  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 4}}, {{6, 2}}}};
  const auto before = stats.snapshot();
  std::vector<std::byte> last_residual;
  std::vector<std::byte> last_projected;
  std::vector<std::byte> last_logits;
  for (const auto &selected : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_residual = half_test::read<BF16>(residual_buffer);
    const auto actual_projected = half_test::read<BF16>(projected_buffer);
    const auto actual_logits = half_test::read<BF16>(logits_buffer);
    for (std::size_t token = 0; token < tokens; ++token) {
      const auto row_index = static_cast<std::size_t>(selected[token]);
      const auto row = std::span<const BF16>(table_values)
                           .subspan(row_index * shape.hidden, shape.hidden);
      const auto expected_normalized =
          q::rms_norm(row, scale_values, 1.0e-6);
      std::vector<BF16> residual_values(shape.hidden);
      std::vector<double> residual_expected(shape.hidden);
      for (std::size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] =
            q::encode(q::decode(row[i]) + expected_normalized[i]);
        residual_expected[i] = q::decode(residual_values[i]);
      }
      expect_near(std::span(actual_residual).subspan(token * shape.hidden,
                                                     shape.hidden),
                  residual_expected);
      const auto expected_projected = q::sparse_projection(
          residual_values, shape.hidden, projection_seed);
      expect_near(std::span(actual_projected)
                      .subspan(token * shape.hidden, shape.hidden),
                  expected_projected);
      const auto projected_bf16 = q::as_bf16(expected_projected);
      const auto expected_logits =
          q::dense_head(projected_bf16, head_values, q::kEmbeddingRows);
      expect_near(std::span(actual_logits)
                      .subspan(token * q::kEmbeddingRows, q::kEmbeddingRows),
                  expected_logits, 0.016);
    }
    last_residual = read_bytes(residual_buffer);
    last_projected = read_bytes(projected_buffer);
    last_logits = read_bytes(logits_buffer);
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);

  constexpr std::array<std::int32_t, tokens> invalid{0, q::kEmbeddingRows};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(residual_buffer), last_residual);
  EXPECT_EQ(read_bytes(projected_buffer), last_projected);
  EXPECT_EQ(read_bytes(logits_buffer), last_logits);
}

TEST_P(CpuGraphQwen38Profiles, D5Qwen38ProfileFfnUsesExactInnerWidthAndReplays) {
  const auto &shape = *GetParam();
  constexpr int seed = 41;
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, shape.hidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {shape.hidden});
  const auto stored_gate_desc =
      dense(c::DataType::kBFloat16, {shape.ffn_hidden, shape.hidden});
  const auto gate_desc = stored_gate_desc.transposed().value();
  const auto stored_down_desc =
      dense(c::DataType::kBFloat16, {shape.hidden, shape.ffn_hidden});
  const auto down_desc = stored_down_desc.transposed().value();

  auto scale_values = q::scales(shape.hidden, 9);
  auto gate_values =
      q::sparse_output_major_weight(shape.ffn_hidden, shape.hidden, seed);
  auto up_values =
      q::sparse_output_major_weight(shape.ffn_hidden, shape.hidden, seed + 1);
  auto down_values =
      q::sparse_output_major_weight(shape.hidden, shape.ffn_hidden, seed + 2);
  auto scale_buffer = half_test::buffer<BF16>(scale_values);
  auto gate_buffer = half_test::buffer<BF16>(gate_values);
  gate_values.clear();
  gate_values.shrink_to_fit();
  auto up_buffer = half_test::buffer<BF16>(up_values);
  up_values.clear();
  up_values.shrink_to_fit();
  auto down_buffer = half_test::buffer<BF16>(down_values);
  down_values.clear();
  down_values.shrink_to_fit();
  auto activation_buffer = output_buffer(activation_desc);
  auto output = output_buffer(activation_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto activation = builder.input(activation_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const auto gate_weight = builder.constant(gate_desc).value();
  const auto up_weight = builder.constant(gate_desc).value();
  const auto down_weight = builder.constant(down_desc).value();
  const auto normalized = builder.add(c::RmsNorm{}, activation, scale).value();
  const auto gate = builder.add(c::MatMul{}, normalized, gate_weight).value();
  const auto up = builder.add(c::MatMul{}, normalized, up_weight).value();
  const auto gated =
      builder
          .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate,
               up)
          .value();
  const auto final = builder.add(c::MatMul{}, gated, down_weight).value();
  ASSERT_TRUE(builder.publish(final).ok());
  auto logical = std::move(builder).finish().value();
  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{
          scale,
          c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()},
      g::ConstantBinding{
          gate_weight,
          c::ConstTensorView::create(gate_desc, gate_buffer.view()).value()},
      g::ConstantBinding{
          up_weight,
          c::ConstTensorView::create(gate_desc, up_buffer.view()).value()},
      g::ConstantBinding{
          down_weight,
          c::ConstTensorView::create(down_desc, down_buffer.view()).value()}};
  const auto workload = q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x0203);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
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
                  .bind(final,
                        c::TensorView::create(activation_desc, output.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int replay_seed : {3, 17}) {
    const auto input = q::values(shape.hidden, replay_seed);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(input)))
                    .ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<BF16>(output);
    const auto expected = q::ffn(input, scale_values, seed, shape.hidden, shape.ffn_hidden);
    expect_near(actual, expected, 0.025);
  }
}

TEST(CpuGraphQwen38_24T,
     D5OfficialBfloatGroupedExpertKeepsExactWidthsTopKPortsAndReplay) {
  constexpr int seed = 71;
  const auto weight_desc = dense(
      c::DataType::kBFloat16,
      {q24::kActiveExperts, q24::kExpertHidden, q24::kHidden});
  const auto down_weight_desc = dense(
      c::DataType::kBFloat16,
      {q24::kActiveExperts, q24::kHidden, q24::kExpertHidden});
  const auto activation_desc = dense(
      c::DataType::kFloat32, {1, q24::kTopK, q24::kHidden});
  const auto router_activation_desc =
      dense(c::DataType::kBFloat16, {1, q24::kHidden});
  const auto router_weight_desc =
      dense(c::DataType::kBFloat16, {q24::kExperts, q24::kHidden})
          .transposed()
          .value();
  const auto replication_indices_desc =
      dense(c::DataType::kInt32, {q24::kTopK});
  const auto ids_desc = dense(c::DataType::kInt32, {1, q24::kTopK});
  const auto output_desc = activation_desc;
  const auto combined_desc =
      dense(c::DataType::kFloat32, {1, 1, q24::kHidden});
  std::vector<BF16> router_weight_values(q24::kExperts * q24::kHidden);
  for (std::int64_t expert = 0; expert < q24::kTopK; ++expert)
    router_weight_values[expert * q24::kHidden + expert] = q::encode(1.0);
  auto router_weight_buffer = half_test::buffer<BF16>(router_weight_values);
  std::array<std::int32_t, q24::kTopK> replication_indices_values{};
  auto replication_indices_buffer =
      half_test::buffer<std::int32_t>(replication_indices_values);
  auto gate_weight_values = q24::expert_weight(
      q24::kExpertHidden, q24::kHidden, seed);
  auto up_weight_values = q24::expert_weight(
      q24::kExpertHidden, q24::kHidden, seed + 1);
  auto down_weight_values = q24::expert_weight(
      q24::kHidden, q24::kExpertHidden, seed + 2);
  auto gate_weight_buffer = half_test::buffer<BF16>(gate_weight_values);
  gate_weight_values.clear();
  gate_weight_values.shrink_to_fit();
  auto up_weight_buffer = half_test::buffer<BF16>(up_weight_values);
  up_weight_values.clear();
  up_weight_values.shrink_to_fit();
  auto down_weight_buffer = half_test::buffer<BF16>(down_weight_values);
  down_weight_values.clear();
  down_weight_values.shrink_to_fit();
  auto ids_buffer = output_buffer(ids_desc);
  auto router_activation_buffer = output_buffer(router_activation_desc);
  auto result_buffer = output_buffer(output_desc);
  auto combined_buffer = output_buffer(combined_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto router_weight = builder.constant(router_weight_desc).value();
  const auto replication_indices =
      builder.constant(replication_indices_desc).value();
  const auto gate_weight = builder.constant(weight_desc).value();
  const auto up_weight = builder.constant(weight_desc).value();
  const auto down_weight = builder.constant(down_weight_desc).value();
  const auto router_activation = builder.input(router_activation_desc).value();
  const auto router =
      builder.add(c::MatMul{}, router_activation, router_weight).value();
  const auto token_fp32 =
      builder.add(c::Cast{c::DataType::kFloat32}, router_activation).value();
  const auto replicated =
      builder.add(c::Gather{}, token_fp32, replication_indices).value();
  const auto activation =
      builder
          .add(c::View::reshape(
                   c::TensorShape::create(std::array<std::int64_t, 3>{
                                              1, q24::kTopK, q24::kHidden})
                       .value())
                   .value(),
               replicated)
          .value();
  const auto fp32 =
      builder.add(c::Cast{c::DataType::kFloat32}, router).value();
  const auto probabilities = builder.add(c::Softmax{}, fp32).value();
  const std::array topk_inputs{probabilities};
  const auto topk =
      builder.add_node(c::Operation(c::TopK{.count = q24::kTopK}),
                       topk_inputs)
          .value();
  const auto selected = builder.result(topk, 0).value();
  const auto ids = builder.result(topk, 1).value();
  const auto selected_sum =
      builder
          .add(c::Reduce{.kind = c::ReductionKind::kSum,
                         .axes = c::Axes::from(std::array<std::int32_t, 1>{-1})
                                     .value(),
                         .keep_dimensions = true,
                         .accumulation_type = c::DataType::kFloat32},
               selected)
          .value();
  const auto normalized =
      builder.add(c::Binary{c::BinaryKind::kDivide}, selected, selected_sum)
          .value();
  const std::array gate_ports{gate_weight, activation, ids};
  const auto gate_node =
      builder.add_node(c::model_ops::grouped_matmul(), gate_ports).value();
  const auto gate = builder.result(gate_node).value();
  const std::array up_ports{up_weight, activation, ids};
  const auto up_node =
      builder.add_node(c::model_ops::grouped_matmul(), up_ports).value();
  const auto up = builder.result(up_node).value();
  const auto gated =
      builder
          .add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate,
               up)
          .value();
  const std::array down_ports{down_weight, gated, ids};
  const auto down_node =
      builder.add_node(c::model_ops::grouped_matmul(), down_ports).value();
  const auto final = builder.result(down_node).value();
  const auto slot_weights =
      builder
          .add(c::View::reshape(
                   c::TensorShape::create(std::array<std::int64_t, 3>{
                                              1, q24::kTopK, 1})
                       .value())
                   .value(),
               normalized)
          .value();
  const auto weighted =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, final, slot_weights)
          .value();
  const auto combined =
      builder
          .add(c::Reduce{.kind = c::ReductionKind::kSum,
                         .axes = c::Axes::from(std::array<std::int32_t, 1>{1})
                                     .value(),
                         .keep_dimensions = true,
                         .accumulation_type = c::DataType::kFloat32},
               weighted)
          .value();
  ASSERT_TRUE(builder.publish(ids).ok());
  ASSERT_TRUE(builder.publish(final).ok());
  ASSERT_TRUE(builder.publish(combined).ok());
  auto logical = std::move(builder).finish().value();
  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{
          router_weight,
          c::ConstTensorView::create(router_weight_desc,
                                     router_weight_buffer.view())
              .value()},
      g::ConstantBinding{
          replication_indices,
          c::ConstTensorView::create(replication_indices_desc,
                                     replication_indices_buffer.view())
              .value()},
      g::ConstantBinding{
          gate_weight,
          c::ConstTensorView::create(weight_desc, gate_weight_buffer.view())
              .value()},
      g::ConstantBinding{
          up_weight,
          c::ConstTensorView::create(weight_desc, up_weight_buffer.view())
              .value()},
      g::ConstantBinding{
          down_weight,
          c::ConstTensorView::create(down_weight_desc,
                                     down_weight_buffer.view())
              .value()}};
  const auto workload =
      q24::workload(c::WorkloadPhase::kDecode, 0x0a0b);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(router_activation,
                        c::ConstTensorView::create(
                            router_activation_desc,
                            router_activation_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(ids, c::TensorView::create(ids_desc,
                                                  ids_buffer.view())
                                 .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(final,
                        c::TensorView::create(output_desc, result_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(combined,
                        c::TensorView::create(combined_desc,
                                              combined_buffer.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int replay : {3, 17}) {
    const auto token_values = q24::expert_input(replay);
    std::vector<BF16> router_input(q24::kHidden);
    for (std::int64_t column = 0; column < q24::kHidden; ++column)
      router_input[column] = q::encode(token_values[column]);
    std::array<std::int32_t, q24::kTopK> expected_ids{};
    double selected_total = 0;
    for (std::int64_t expert = 0; expert < q24::kTopK; ++expert) {
      const double logit = replay == 3 ? 1.0 + expert / 8.0
                                       : 1.0 + (q24::kTopK - 1 - expert) / 8.0;
      router_input[expert] = q::encode(logit);
      selected_total += std::exp(q::decode(router_input[expert]));
      expected_ids[expert] = static_cast<std::int32_t>(
          replay == 3 ? q24::kTopK - 1 - expert : expert);
    }
    ASSERT_TRUE(router_activation_buffer.view()
                    .write(std::as_bytes(std::span(router_input)))
                    .ok());
    std::vector<float> input(q24::kTopK * q24::kHidden);
    for (std::int64_t slot = 0; slot < q24::kTopK; ++slot)
      for (std::int64_t column = 0; column < q24::kHidden; ++column)
        input[slot * q24::kHidden + column] =
            static_cast<float>(q::decode(router_input[column]));
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_ids = half_test::read<std::int32_t>(ids_buffer);
    EXPECT_TRUE(std::ranges::equal(actual_ids, expected_ids));
    const auto actual = half_test::read<float>(result_buffer);
    const auto expected = q24::expert_oracle(input, seed, expected_ids);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i],
                  std::abs(expected[i]) * q24::kExpertRelativeTolerance +
                      q24::kExpertAbsoluteTolerance)
          << i;
    const auto actual_combined = half_test::read<float>(combined_buffer);
    ASSERT_EQ(actual_combined.size(), q24::kHidden);
    for (std::int64_t column = 0; column < q24::kHidden; ++column) {
      double expected_combined = 0;
      for (std::int64_t slot = 0; slot < q24::kTopK; ++slot) {
        const auto id = expected_ids[slot];
        const double weight = std::exp(q::decode(router_input[id])) /
                              selected_total;
        expected_combined +=
            weight * expected[slot * q24::kHidden + column];
      }
      EXPECT_NEAR(actual_combined[column], expected_combined,
                  std::abs(expected_combined) *
                          q24::kExpertRelativeTolerance +
                      q24::kExpertAbsoluteTolerance)
          << column;
    }
  }
}

TEST(CpuGraphQwen38_24T,
     D5OfficialRouterScoresAll512ExpertsAndNormalizesStableTop10) {
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, q24::kHidden});
  const auto weight_desc =
      dense(c::DataType::kBFloat16, {q24::kExperts, q24::kHidden})
          .transposed()
          .value();
  const auto selected_desc =
      dense(c::DataType::kFloat32, {1, q24::kTopK});
  const auto ids_desc = dense(c::DataType::kInt32, {1, q24::kTopK});
  std::vector<BF16> router_weight(q24::kExperts * q24::kHidden);
  for (std::int64_t expert = 0; expert < q24::kTopK; ++expert)
    router_weight[expert * q24::kHidden + expert] = q::encode(1.0);
  auto weight_buffer = half_test::buffer<BF16>(router_weight);
  auto activation_buffer = output_buffer(activation_desc);
  auto ids_buffer = output_buffer(ids_desc);
  auto normalized_buffer = output_buffer(selected_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto activation = builder.input(activation_desc).value();
  const auto weight = builder.constant(weight_desc).value();
  const auto router = builder.add(c::MatMul{}, activation, weight).value();
  const auto fp32 =
      builder.add(c::Cast{c::DataType::kFloat32}, router).value();
  const auto probabilities = builder.add(c::Softmax{}, fp32).value();
  const std::array topk_inputs{probabilities};
  const auto topk =
      builder.add_node(c::Operation(c::TopK{.count = q24::kTopK}),
                       topk_inputs)
          .value();
  const auto selected = builder.result(topk, 0).value();
  const auto ids = builder.result(topk, 1).value();
  const auto sum =
      builder
          .add(c::Reduce{.kind = c::ReductionKind::kSum,
                         .axes = c::Axes::from(std::array<std::int32_t, 1>{-1})
                                     .value(),
                         .keep_dimensions = true,
                         .accumulation_type = c::DataType::kFloat32},
               selected)
          .value();
  const auto normalized =
      builder.add(c::Binary{c::BinaryKind::kDivide}, selected, sum).value();
  ASSERT_TRUE(builder.publish(ids).ok());
  ASSERT_TRUE(builder.publish(normalized).ok());
  auto logical = std::move(builder).finish().value();

  auto context = g::cpu::create_context().value();
  const std::array constants{g::ConstantBinding{
      weight, c::ConstTensorView::create(weight_desc, weight_buffer.view())
                  .value()}};
  const auto workload = q24::workload(c::WorkloadPhase::kDecode, 0x0a10);
  auto compiled_result = g::compile(
      logical, context, {.constants = constants, .workload = workload});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(activation,
                        c::ConstTensorView::create(activation_desc,
                                                   activation_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(ids,
                        c::TensorView::create(ids_desc, ids_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(normalized,
                        c::TensorView::create(selected_desc,
                                              normalized_buffer.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int replay = 0; replay < 3; ++replay) {
    std::vector<BF16> input(q24::kHidden);
    for (std::int64_t expert = 0; expert < q24::kTopK; ++expert) {
      const double logit = replay == 0 ? 1.0 + expert / 8.0
                           : replay == 1 ? 1.0 + (q24::kTopK - 1 - expert) / 8.0
                                         : 2.0;
      input[expert] = q::encode(logit);
    }
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(input)))
                    .ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_ids = half_test::read<std::int32_t>(ids_buffer);
    const auto weights = half_test::read<float>(normalized_buffer);
    double selected_sum = 0;
    for (std::int64_t expert = 0; expert < q24::kTopK; ++expert)
      selected_sum += std::exp(q::decode(input[expert]));
    for (std::int64_t slot = 0; slot < q24::kTopK; ++slot) {
      const auto expected_id = replay == 0 ? q24::kTopK - 1 - slot
                               : replay == 1 ? slot : slot;
      EXPECT_EQ(actual_ids[slot], expected_id);
      EXPECT_NEAR(weights[slot],
                  std::exp(q::decode(input[expected_id])) / selected_sum,
                  2.0e-6);
    }
  }
}

TEST_P(CpuGraphQwen38Profiles, D5Qwen38ProfilePartialRopeGqaAndOutputGateReplay) {
  const auto &shape = *GetParam();
  const auto query_desc = dense(c::DataType::kBFloat16,
                                {1, 1, shape.query_heads, q::kHeadDepth});
  const auto kv_desc = dense(c::DataType::kBFloat16,
                             {1, 1, shape.kv_heads, q::kHeadDepth});
  const auto scale_desc = dense(c::DataType::kBFloat16, {q::kHeadDepth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc =
      dense(c::DataType::kFloat32, {q::kRotaryDimensions / 2});
  auto query_scale = q::scales(q::kHeadDepth, 1);
  auto key_scale = q::scales(q::kHeadDepth, 2);
  std::array<std::int32_t, 1> position{11};
  std::vector<float> frequencies(q::kRotaryDimensions / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / static_cast<float>(i + 1);
  auto query_scale_buffer = half_test::buffer<BF16>(query_scale);
  auto key_scale_buffer = half_test::buffer<BF16>(key_scale);
  auto position_buffer = half_test::buffer<std::int32_t>(position);
  auto frequency_buffer = half_test::buffer<float>(frequencies);
  auto query_buffer = output_buffer(query_desc);
  auto key_buffer = output_buffer(kv_desc);
  auto value_buffer = output_buffer(kv_desc);
  auto gate_buffer = output_buffer(query_desc);
  auto result_buffer = output_buffer(query_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto query = builder.input(query_desc).value();
  const auto key = builder.input(kv_desc).value();
  const auto value = builder.input(kv_desc).value();
  const auto gate = builder.input(query_desc).value();
  const auto q_scale = builder.constant(scale_desc).value();
  const auto k_scale = builder.constant(scale_desc).value();
  const auto positions = builder.constant(position_desc).value();
  const auto frequency = builder.constant(frequency_desc).value();
  const auto normalized_query = builder.add(c::RmsNorm{}, query, q_scale).value();
  const auto normalized_key = builder.add(c::RmsNorm{}, key, k_scale).value();
  const c::RoPE rope{.rotary_dimensions = q::kRotaryDimensions,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kInterleaved};
  const auto rotated_query =
      builder.add(rope, normalized_query, positions, frequency).value();
  const auto rotated_key =
      builder.add(rope, normalized_key, positions, frequency).value();
  const auto attention =
      builder
          .add(c::ScaledDotProductAttention{
                   .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                   .scale = 0.0625F,
                   .causal = true,
                   .accumulation_type = c::DataType::kFloat32},
               rotated_query, rotated_key, value)
          .value();
  const auto sigmoid = builder.add(c::Unary{c::UnaryKind::kSigmoid}, gate).value();
  const auto output =
      builder.add(c::Binary{c::BinaryKind::kMultiply}, attention, sigmoid)
          .value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();
  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{q_scale,
                         c::ConstTensorView::create(
                             scale_desc, query_scale_buffer.view())
                             .value()},
      g::ConstantBinding{k_scale,
                         c::ConstTensorView::create(scale_desc,
                                                    key_scale_buffer.view())
                             .value()},
      g::ConstantBinding{positions,
                         c::ConstTensorView::create(position_desc,
                                                    position_buffer.view())
                             .value()},
      g::ConstantBinding{frequency,
                         c::ConstTensorView::create(frequency_desc,
                                                    frequency_buffer.view())
                             .value()}};
  const auto workload = q::workload(*shape.profile, c::WorkloadPhase::kDecode, 4);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
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
                  .bind(gate, c::ConstTensorView::create(
                                  query_desc, gate_buffer.view())
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
  for (int seed : {2, 13}) {
    const auto queries = q::values(*query_desc.element_count(), seed);
    const auto keys = q::values(*kv_desc.element_count(), seed + 1);
    const auto values = q::values(*kv_desc.element_count(), seed + 2, 128.0);
    const auto gates = q::values(*query_desc.element_count(), seed + 3, 32.0);
    ASSERT_TRUE(query_buffer.view().write(std::as_bytes(std::span(queries))).ok());
    ASSERT_TRUE(key_buffer.view().write(std::as_bytes(std::span(keys))).ok());
    ASSERT_TRUE(value_buffer.view().write(std::as_bytes(std::span(values))).ok());
    ASSERT_TRUE(gate_buffer.view().write(std::as_bytes(std::span(gates))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<BF16>(result_buffer);
    for (std::size_t head = 0; head < static_cast<std::size_t>(shape.query_heads); ++head) {
      const auto value_head = head / (shape.query_heads / shape.kv_heads);
      for (std::size_t column = 0; column < q::kHeadDepth; ++column) {
        const auto index = head * q::kHeadDepth + column;
        const auto sigmoid_value = q::decode(q::encode(
            1.0 / (1.0 + std::exp(-q::decode(gates[index])))));
        const auto expected = q::decode(q::encode(
            q::decode(values[value_head * q::kHeadDepth + column]) *
            sigmoid_value));
        EXPECT_NEAR(q::decode(actual[index]), expected,
                    std::abs(expected) * 0.012 + 2.0e-5)
            << index;
      }
    }
  }
}

TEST_P(CpuGraphQwen38Profiles, D5Qwen38ProfileMtpUsesExactConcatWidthAndReplays) {
  const auto &shape = *GetParam();
  constexpr int weight_seed = 67;
  const auto half_desc = dense(c::DataType::kBFloat16, {1, shape.hidden});
  const auto joined_desc = dense(c::DataType::kBFloat16, {1, (2 * shape.hidden)});
  const auto stored_weight_desc =
      dense(c::DataType::kBFloat16, {shape.hidden, (2 * shape.hidden)});
  const auto weight_desc = stored_weight_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {shape.hidden});
  const auto output_f32_desc = dense(c::DataType::kFloat32, {1, shape.hidden});
  auto weight_values =
      q::sparse_output_major_weight(shape.hidden, (2 * shape.hidden), weight_seed);
  auto scale_values = q::scales(shape.hidden, 11);
  auto weight_buffer = half_test::buffer<BF16>(weight_values);
  weight_values.clear();
  weight_values.shrink_to_fit();
  auto scale_buffer = half_test::buffer<BF16>(scale_values);
  auto first_buffer = output_buffer(half_desc);
  auto second_buffer = output_buffer(half_desc);
  auto projected_buffer = output_buffer(half_desc);
  auto converted_buffer = output_buffer(output_f32_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto first = builder.input(half_desc).value();
  const auto second = builder.input(half_desc).value();
  const auto weight = builder.constant(weight_desc).value();
  const auto scale = builder.constant(scale_desc).value();
  const std::array halves{first, second};
  const auto concat =
      builder.add_node(c::Operation(c::Concat{.axis = 1}), halves).value();
  const auto joined = builder.result(concat).value();
  const auto projected = builder.add(c::MatMul{}, joined, weight).value();
  const auto normalized = builder.add(c::RmsNorm{}, projected, scale).value();
  const auto converted =
      builder.add(c::Cast{c::DataType::kFloat32}, normalized).value();
  ASSERT_TRUE(builder.publish(projected).ok());
  ASSERT_TRUE(builder.publish(converted).ok());
  auto logical = std::move(builder).finish().value();
  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{
          weight,
          c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()},
      g::ConstantBinding{
          scale,
          c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()}};
  const auto workload = q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x0d0e);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
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
                        c::TensorView::create(half_desc,
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
  for (int seed : {5, 19}) {
    const auto input = q::values((2 * shape.hidden), seed);
    ASSERT_TRUE(first_buffer.view()
                    .write(std::as_bytes(std::span(input).first(shape.hidden)))
                    .ok());
    ASSERT_TRUE(second_buffer.view()
                    .write(std::as_bytes(std::span(input).last(shape.hidden)))
                    .ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_projected = half_test::read<BF16>(projected_buffer);
    const auto actual_converted = half_test::read<float>(converted_buffer);
    const auto expected_projected =
        q::sparse_projection(input, shape.hidden, weight_seed);
    expect_near(actual_projected, expected_projected);
    const auto projected_bf16 = q::as_bf16(expected_projected);
    const auto expected_normalized =
        q::rms_norm(projected_bf16, scale_values);
    ASSERT_EQ(actual_converted.size(), expected_normalized.size());
    for (std::size_t i = 0; i < actual_converted.size(); ++i)
      EXPECT_NEAR(actual_converted[i], expected_normalized[i],
                  std::abs(expected_normalized[i]) * 0.012 + 2.0e-5)
          << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, CpuGraphQwen38Profiles,
    ::testing::Values(&qcase::kOfficial27B, &qcase::kPrimaryGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, CpuGraphQwen38Profiles,
    ::testing::Values(&qcase::kFidelityGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5Official24T, CpuGraphQwen38Profiles,
    ::testing::Values(&qcase::kOfficial24T));

} // namespace
