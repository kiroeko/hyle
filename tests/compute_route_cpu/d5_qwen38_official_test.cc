#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../compute_models/d5_qwen38_official_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace q = d5_qwen38_official_fixture;
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

struct DirectCall final {
  c::PreparedOperation prepared;
  hyle::mem::Buffer workspace;
  c::OperationInvocation invocation;
};

DirectCall make_call(const c::Context &context, c::Operation operation,
                     std::span<const c::TensorDesc> inputs,
                     std::span<const c::TensorDesc> outputs,
                     const c::WorkloadContext &workload) {
  const auto kind = operation.kind();
  auto prepared = c::prepare(context, std::move(operation), inputs, outputs,
                             hyle::mem::system_resource(), workload);
  if (!prepared)
    throw std::runtime_error(std::string(c::operation_name(kind)) +
                             ": " +
                             std::string(prepared.error().message()));
  const auto requirement = prepared->workspace_requirement();
  auto workspace =
      requirement.size == 0
          ? hyle::mem::Buffer{}
          : hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      requirement.size, requirement.alignment)
                .value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  return {std::move(*prepared), std::move(workspace), std::move(invocation)};
}

void expect_mature(const DirectCall &call, const c::WorkloadContext &workload) {
  const auto provider = call.prepared.provider().id;
  EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
              provider == c::onednn::kProviderId)
      << c::operation_name(call.prepared.kind()) << ": "
      << call.prepared.provider().name;
  EXPECT_EQ(call.prepared.workload(), workload);
}

void run(DirectCall &call, c::Queue &queue,
         std::span<const c::ConstTensorView> inputs,
         std::span<const c::TensorView> outputs) {
  auto completion = call.invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
}

void expect_near(std::span<const BF16> actual,
                 std::span<const double> expected, double relative = 0.012) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(q::decode(actual[i]), expected[i],
                std::abs(expected[i]) * relative + 2.0e-5)
        << i;
}

class CpuQwen38Profiles
    : public ::testing::TestWithParam<
          const c::ModelArtifactProfileDescription *> {};

TEST_P(CpuQwen38Profiles,
     D5Qwen38ProfileBoundaryProjectionUsesExactHiddenAndReplays) {
  constexpr std::int64_t tokens = 2;
  constexpr int projection_seed = 31;
  const auto table_desc =
      dense(c::DataType::kBFloat16, {q::kEmbeddingRows, q::kHidden});
  const auto head_desc = table_desc.transposed().value();
  const auto stored_projection_desc =
      dense(c::DataType::kBFloat16, {q::kHidden, q::kHidden});
  const auto projection_desc = stored_projection_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {q::kHidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto token_desc =
      dense(c::DataType::kBFloat16, {tokens, q::kHidden});
  const auto logits_desc =
      dense(c::DataType::kBFloat16, {tokens, q::kEmbeddingRows});
  const auto workload = q::workload(*GetParam(), c::WorkloadPhase::kDecode, 0x01030e0f);

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      c::cpu::create_context({.metadata_allocator = stats.get()}).value();
  auto queue = context.create_queue().value();
  const std::array gather_inputs{table_desc, ids_desc};
  const std::array token_output{token_desc};
  auto gather = make_call(context, c::Operation(c::Gather{}), gather_inputs,
                          token_output, workload);
  const std::array norm_inputs{token_desc, scale_desc};
  auto norm = make_call(context, c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                        norm_inputs, token_output, workload);
  const std::array add_inputs{token_desc, token_desc};
  auto residual =
      make_call(context, c::Operation(c::Binary{c::BinaryKind::kAdd}),
                add_inputs, token_output, workload);
  const std::array projection_inputs{token_desc, projection_desc};
  auto projection = make_call(context, c::Operation(c::MatMul{}),
                              projection_inputs, token_output, workload);
  const std::array head_inputs{token_desc, head_desc};
  const std::array head_output{logits_desc};
  auto head = make_call(context, c::Operation(c::MatMul{}), head_inputs,
                        head_output, workload);
  for (const auto *call : {&gather, &norm, &residual, &projection, &head})
    expect_mature(*call, workload);

  auto table_values = q::values(q::kEmbeddingRows * q::kHidden, 5);
  auto head_values = q::values(q::kEmbeddingRows * q::kHidden, 7, 96.0);
  auto scale_values = q::scales(q::kHidden, 3);
  auto projection_values = q::sparse_output_major_weight(
      q::kHidden, q::kHidden, projection_seed);
  auto table_buffer = half_test::buffer<BF16>(table_values);
  auto head_buffer = half_test::buffer<BF16>(head_values);
  auto scale_buffer = half_test::buffer<BF16>(scale_values);
  auto projection_buffer = half_test::buffer<BF16>(projection_values);
  projection_values.clear();
  projection_values.shrink_to_fit();
  auto ids_buffer = output_buffer(ids_desc);
  auto embedded_buffer = output_buffer(token_desc);
  auto normalized_buffer = output_buffer(token_desc);
  auto residual_buffer = output_buffer(token_desc);
  auto projected_buffer = output_buffer(token_desc);
  auto logits_buffer = output_buffer(logits_desc);

  const std::array gather_views{
      c::ConstTensorView::create(table_desc, table_buffer.view()).value(),
      c::ConstTensorView::create(ids_desc, ids_buffer.view()).value()};
  const std::array embedded_view{
      c::TensorView::create(token_desc, embedded_buffer.view()).value()};
  const std::array norm_views{
      c::ConstTensorView::create(token_desc, embedded_buffer.view()).value(),
      c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()};
  const std::array normalized_view{
      c::TensorView::create(token_desc, normalized_buffer.view()).value()};
  const std::array add_views{
      c::ConstTensorView::create(token_desc, embedded_buffer.view()).value(),
      c::ConstTensorView::create(token_desc, normalized_buffer.view()).value()};
  const std::array residual_view{
      c::TensorView::create(token_desc, residual_buffer.view()).value()};
  const std::array projection_views{
      c::ConstTensorView::create(token_desc, residual_buffer.view()).value(),
      c::ConstTensorView::create(projection_desc, projection_buffer.view())
          .value()};
  const std::array projected_view{
      c::TensorView::create(token_desc, projected_buffer.view()).value()};
  const std::array head_views{
      c::ConstTensorView::create(token_desc, projected_buffer.view()).value(),
      c::ConstTensorView::create(head_desc, head_buffer.view()).value()};
  const std::array logits_view{
      c::TensorView::create(logits_desc, logits_buffer.view()).value()};

  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 4}}, {{6, 2}}}};
  const auto before = stats.snapshot();
  std::vector<std::byte> last_embedded;
  std::vector<std::byte> last_projected;
  std::vector<std::byte> last_logits;
  for (const auto &selected : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
    run(gather, queue, gather_views, embedded_view);
    run(norm, queue, norm_views, normalized_view);
    run(residual, queue, add_views, residual_view);
    run(projection, queue, projection_views, projected_view);
    run(head, queue, head_views, logits_view);
    const auto actual_residual = half_test::read<BF16>(residual_buffer);
    const auto actual_projected = half_test::read<BF16>(projected_buffer);
    const auto actual_logits = half_test::read<BF16>(logits_buffer);
    for (std::size_t token = 0; token < tokens; ++token) {
      const auto row_index = static_cast<std::size_t>(selected[token]);
      const auto row = std::span<const BF16>(table_values)
                           .subspan(row_index * q::kHidden, q::kHidden);
      const auto normalized = q::rms_norm(row, scale_values, 1.0e-6);
      std::vector<BF16> residual_values(q::kHidden);
      std::vector<double> residual_expected(q::kHidden);
      for (std::size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] = q::encode(q::decode(row[i]) + normalized[i]);
        residual_expected[i] = q::decode(residual_values[i]);
      }
      expect_near(std::span(actual_residual).subspan(token * q::kHidden,
                                                     q::kHidden),
                  residual_expected);
      const auto projected = q::sparse_projection(
          residual_values, q::kHidden, projection_seed);
      expect_near(std::span(actual_projected)
                      .subspan(token * q::kHidden, q::kHidden),
                  projected);
      const auto projected_bf16 = q::as_bf16(projected);
      const auto logits =
          q::dense_head(projected_bf16, head_values, q::kEmbeddingRows);
      expect_near(std::span(actual_logits)
                      .subspan(token * q::kEmbeddingRows, q::kEmbeddingRows),
                  logits, 0.016);
    }
    last_embedded = read_bytes(embedded_buffer);
    last_projected = read_bytes(projected_buffer);
    last_logits = read_bytes(logits_buffer);
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);

  constexpr std::array<std::int32_t, tokens> invalid{0, q::kEmbeddingRows};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = gather.invocation.submit(queue, gather_views, embedded_view);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(embedded_buffer), last_embedded);
  EXPECT_EQ(read_bytes(projected_buffer), last_projected);
  EXPECT_EQ(read_bytes(logits_buffer), last_logits);
}

TEST_P(CpuQwen38Profiles, D5Qwen38ProfileFfnUsesExactInnerWidthAndReplays) {
  constexpr int seed = 41;
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, q::kHidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {q::kHidden});
  const auto stored_gate_desc =
      dense(c::DataType::kBFloat16, {q::kFfnHidden, q::kHidden});
  const auto gate_desc = stored_gate_desc.transposed().value();
  const auto stored_down_desc =
      dense(c::DataType::kBFloat16, {q::kHidden, q::kFfnHidden});
  const auto down_desc = stored_down_desc.transposed().value();
  const auto expanded_desc =
      dense(c::DataType::kBFloat16, {1, q::kFfnHidden});
  const auto workload = q::workload(*GetParam(), c::WorkloadPhase::kDecode, 0x0203);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array norm_inputs{activation_desc, scale_desc};
  const std::array activation_output{activation_desc};
  auto norm = make_call(context, c::Operation(c::RmsNorm{}), norm_inputs,
                        activation_output, workload);
  const std::array gate_inputs{activation_desc, gate_desc};
  const std::array expanded_output{expanded_desc};
  auto gate = make_call(context, c::Operation(c::MatMul{}), gate_inputs,
                        expanded_output, workload);
  auto up = make_call(context, c::Operation(c::MatMul{}), gate_inputs,
                      expanded_output, workload);
  const std::array gated_inputs{expanded_desc, expanded_desc};
  auto gated = make_call(
      context,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      gated_inputs, expanded_output, workload);
  const std::array down_inputs{expanded_desc, down_desc};
  auto down = make_call(context, c::Operation(c::MatMul{}), down_inputs,
                        activation_output, workload);
  for (const auto *call : {&norm, &gate, &up, &gated, &down})
    expect_mature(*call, workload);

  auto scale_values = q::scales(q::kHidden, 9);
  auto gate_values =
      q::sparse_output_major_weight(q::kFfnHidden, q::kHidden, seed);
  auto up_values =
      q::sparse_output_major_weight(q::kFfnHidden, q::kHidden, seed + 1);
  auto down_values =
      q::sparse_output_major_weight(q::kHidden, q::kFfnHidden, seed + 2);
  auto scale_buffer = half_test::buffer<BF16>(scale_values);
  auto gate_weight_buffer = half_test::buffer<BF16>(gate_values);
  gate_values.clear();
  gate_values.shrink_to_fit();
  auto up_weight_buffer = half_test::buffer<BF16>(up_values);
  up_values.clear();
  up_values.shrink_to_fit();
  auto down_weight_buffer = half_test::buffer<BF16>(down_values);
  down_values.clear();
  down_values.shrink_to_fit();
  auto activation_buffer = output_buffer(activation_desc);
  auto normalized_buffer = output_buffer(activation_desc);
  auto gate_buffer = output_buffer(expanded_desc);
  auto up_buffer = output_buffer(expanded_desc);
  auto gated_buffer = output_buffer(expanded_desc);
  auto result_buffer = output_buffer(activation_desc);

  const std::array norm_views{
      c::ConstTensorView::create(activation_desc, activation_buffer.view())
          .value(),
      c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()};
  const std::array normalized_view{
      c::TensorView::create(activation_desc, normalized_buffer.view()).value()};
  const std::array gate_views{
      c::ConstTensorView::create(activation_desc, normalized_buffer.view())
          .value(),
      c::ConstTensorView::create(gate_desc, gate_weight_buffer.view()).value()};
  const std::array gate_view{
      c::TensorView::create(expanded_desc, gate_buffer.view()).value()};
  const std::array up_views{
      c::ConstTensorView::create(activation_desc, normalized_buffer.view())
          .value(),
      c::ConstTensorView::create(gate_desc, up_weight_buffer.view()).value()};
  const std::array up_view{
      c::TensorView::create(expanded_desc, up_buffer.view()).value()};
  const std::array gated_views{
      c::ConstTensorView::create(expanded_desc, gate_buffer.view()).value(),
      c::ConstTensorView::create(expanded_desc, up_buffer.view()).value()};
  const std::array gated_view{
      c::TensorView::create(expanded_desc, gated_buffer.view()).value()};
  const std::array down_views{
      c::ConstTensorView::create(expanded_desc, gated_buffer.view()).value(),
      c::ConstTensorView::create(down_desc, down_weight_buffer.view()).value()};
  const std::array result_view{
      c::TensorView::create(activation_desc, result_buffer.view()).value()};

  for (int replay_seed : {3, 17}) {
    const auto activation = q::values(q::kHidden, replay_seed);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(activation)))
                    .ok());
    run(norm, queue, norm_views, normalized_view);
    run(gate, queue, gate_views, gate_view);
    run(up, queue, up_views, up_view);
    run(gated, queue, gated_views, gated_view);
    run(down, queue, down_views, result_view);
    const auto actual = half_test::read<BF16>(result_buffer);
    const auto expected = q::ffn(activation, scale_values, seed);
    expect_near(actual, expected, 0.025);
  }
}

TEST_P(CpuQwen38Profiles,
     D5Qwen38ProfilePartialRopeGqaAndOutputGateReplay) {
  const auto query_desc = dense(c::DataType::kBFloat16,
                                {1, 1, q::kQueryHeads, q::kHeadDepth});
  const auto kv_desc = dense(c::DataType::kBFloat16,
                             {1, 1, q::kKvHeads, q::kHeadDepth});
  const auto scale_desc = dense(c::DataType::kBFloat16, {q::kHeadDepth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc =
      dense(c::DataType::kFloat32, {q::kRotaryDimensions / 2});
  const auto workload = q::workload(*GetParam(), c::WorkloadPhase::kDecode, 4);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array query_norm_inputs{query_desc, scale_desc};
  const std::array query_output{query_desc};
  auto query_norm = make_call(context, c::Operation(c::RmsNorm{}),
                              query_norm_inputs, query_output, workload);
  const std::array key_norm_inputs{kv_desc, scale_desc};
  const std::array key_output{kv_desc};
  auto key_norm = make_call(context, c::Operation(c::RmsNorm{}),
                            key_norm_inputs, key_output, workload);
  const c::RoPE rope{.rotary_dimensions = q::kRotaryDimensions,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kInterleaved};
  const std::array query_rope_inputs{query_desc, position_desc, frequency_desc};
  auto query_rope = make_call(context, c::Operation(rope), query_rope_inputs,
                              query_output, workload);
  const std::array key_rope_inputs{kv_desc, position_desc, frequency_desc};
  auto key_rope = make_call(context, c::Operation(rope), key_rope_inputs,
                            key_output, workload);
  const std::array attention_inputs{query_desc, kv_desc, kv_desc};
  auto attention = make_call(
      context,
      c::Operation(c::ScaledDotProductAttention{
          .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
          .scale = 0.0625F,
          .causal = true,
          .accumulation_type = c::DataType::kFloat32}),
      attention_inputs, query_output, workload);
  const std::array unary_input{query_desc};
  auto sigmoid = make_call(context, c::Operation(c::Unary{c::UnaryKind::kSigmoid}),
                           unary_input, query_output, workload);
  const std::array multiply_inputs{query_desc, query_desc};
  auto multiply = make_call(
      context, c::Operation(c::Binary{c::BinaryKind::kMultiply}),
      multiply_inputs, query_output, workload);
  for (const auto *call : {&query_norm, &key_norm, &query_rope, &key_rope,
                           &attention, &sigmoid, &multiply})
    expect_mature(*call, workload);

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
  auto normalized_query_buffer = output_buffer(query_desc);
  auto normalized_key_buffer = output_buffer(kv_desc);
  auto rotated_query_buffer = output_buffer(query_desc);
  auto rotated_key_buffer = output_buffer(kv_desc);
  auto attention_buffer = output_buffer(query_desc);
  auto sigmoid_buffer = output_buffer(query_desc);
  auto result_buffer = output_buffer(query_desc);
  const auto position_view =
      c::ConstTensorView::create(position_desc, position_buffer.view()).value();
  const auto frequency_view =
      c::ConstTensorView::create(frequency_desc, frequency_buffer.view()).value();
  const std::array query_norm_views{
      c::ConstTensorView::create(query_desc, query_buffer.view()).value(),
      c::ConstTensorView::create(scale_desc, query_scale_buffer.view()).value()};
  const std::array normalized_query_view{
      c::TensorView::create(query_desc, normalized_query_buffer.view()).value()};
  const std::array key_norm_views{
      c::ConstTensorView::create(kv_desc, key_buffer.view()).value(),
      c::ConstTensorView::create(scale_desc, key_scale_buffer.view()).value()};
  const std::array normalized_key_view{
      c::TensorView::create(kv_desc, normalized_key_buffer.view()).value()};
  const std::array query_rope_views{
      c::ConstTensorView::create(query_desc, normalized_query_buffer.view())
          .value(),
      position_view, frequency_view};
  const std::array rotated_query_view{
      c::TensorView::create(query_desc, rotated_query_buffer.view()).value()};
  const std::array key_rope_views{
      c::ConstTensorView::create(kv_desc, normalized_key_buffer.view()).value(),
      position_view, frequency_view};
  const std::array rotated_key_view{
      c::TensorView::create(kv_desc, rotated_key_buffer.view()).value()};
  const std::array attention_views{
      c::ConstTensorView::create(query_desc, rotated_query_buffer.view()).value(),
      c::ConstTensorView::create(kv_desc, rotated_key_buffer.view()).value(),
      c::ConstTensorView::create(kv_desc, value_buffer.view()).value()};
  const std::array attention_view{
      c::TensorView::create(query_desc, attention_buffer.view()).value()};
  const std::array sigmoid_views{
      c::ConstTensorView::create(query_desc, gate_buffer.view()).value()};
  const std::array sigmoid_view{
      c::TensorView::create(query_desc, sigmoid_buffer.view()).value()};
  const std::array multiply_views{
      c::ConstTensorView::create(query_desc, attention_buffer.view()).value(),
      c::ConstTensorView::create(query_desc, sigmoid_buffer.view()).value()};
  const std::array result_view{
      c::TensorView::create(query_desc, result_buffer.view()).value()};

  for (int seed : {2, 13}) {
    const auto queries = q::values(*query_desc.element_count(), seed);
    const auto keys = q::values(*kv_desc.element_count(), seed + 1);
    const auto values = q::values(*kv_desc.element_count(), seed + 2, 128.0);
    const auto gates = q::values(*query_desc.element_count(), seed + 3, 32.0);
    ASSERT_TRUE(query_buffer.view().write(std::as_bytes(std::span(queries))).ok());
    ASSERT_TRUE(key_buffer.view().write(std::as_bytes(std::span(keys))).ok());
    ASSERT_TRUE(value_buffer.view().write(std::as_bytes(std::span(values))).ok());
    ASSERT_TRUE(gate_buffer.view().write(std::as_bytes(std::span(gates))).ok());
    run(query_norm, queue, query_norm_views, normalized_query_view);
    run(key_norm, queue, key_norm_views, normalized_key_view);
    run(query_rope, queue, query_rope_views, rotated_query_view);
    run(key_rope, queue, key_rope_views, rotated_key_view);
    run(attention, queue, attention_views, attention_view);
    run(sigmoid, queue, sigmoid_views, sigmoid_view);
    run(multiply, queue, multiply_views, result_view);
    const auto actual = half_test::read<BF16>(result_buffer);
    for (std::size_t head = 0; head < q::kQueryHeads; ++head) {
      const auto value_head = head / (q::kQueryHeads / q::kKvHeads);
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

TEST_P(CpuQwen38Profiles, D5Qwen38ProfileMtpUsesExactConcatWidthAndReplays) {
  constexpr int weight_seed = 67;
  const auto half_desc = dense(c::DataType::kBFloat16, {1, q::kHidden});
  const auto joined_desc = dense(c::DataType::kBFloat16, {1, q::kMtpInput});
  const auto stored_weight_desc =
      dense(c::DataType::kBFloat16, {q::kHidden, q::kMtpInput});
  const auto weight_desc = stored_weight_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {q::kHidden});
  const auto output_f32_desc = dense(c::DataType::kFloat32, {1, q::kHidden});
  const auto workload = q::workload(*GetParam(), c::WorkloadPhase::kDecode, 0x0d0e);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array concat_inputs{half_desc, half_desc};
  const std::array joined_output{joined_desc};
  auto concat = make_call(context, c::Operation(c::Concat{.axis = 1}),
                          concat_inputs, joined_output, workload);
  const std::array projection_inputs{joined_desc, weight_desc};
  const std::array half_output{half_desc};
  auto projection = make_call(context, c::Operation(c::MatMul{}),
                              projection_inputs, half_output, workload);
  const std::array norm_inputs{half_desc, scale_desc};
  auto norm = make_call(context, c::Operation(c::RmsNorm{}), norm_inputs,
                        half_output, workload);
  const std::array cast_inputs{half_desc};
  const std::array cast_output{output_f32_desc};
  auto cast = make_call(context, c::Operation(c::Cast{c::DataType::kFloat32}),
                        cast_inputs, cast_output, workload);
  for (const auto *call : {&concat, &projection, &norm, &cast})
    expect_mature(*call, workload);

  auto weight_values =
      q::sparse_output_major_weight(q::kHidden, q::kMtpInput, weight_seed);
  auto scale_values = q::scales(q::kHidden, 11);
  auto weight_buffer = half_test::buffer<BF16>(weight_values);
  weight_values.clear();
  weight_values.shrink_to_fit();
  auto scale_buffer = half_test::buffer<BF16>(scale_values);
  auto first_buffer = output_buffer(half_desc);
  auto second_buffer = output_buffer(half_desc);
  auto joined_buffer = output_buffer(joined_desc);
  auto projected_buffer = output_buffer(half_desc);
  auto normalized_buffer = output_buffer(half_desc);
  auto converted_buffer = output_buffer(output_f32_desc);
  const std::array concat_views{
      c::ConstTensorView::create(half_desc, first_buffer.view()).value(),
      c::ConstTensorView::create(half_desc, second_buffer.view()).value()};
  const std::array joined_view{
      c::TensorView::create(joined_desc, joined_buffer.view()).value()};
  const std::array projection_views{
      c::ConstTensorView::create(joined_desc, joined_buffer.view()).value(),
      c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()};
  const std::array projected_view{
      c::TensorView::create(half_desc, projected_buffer.view()).value()};
  const std::array norm_views{
      c::ConstTensorView::create(half_desc, projected_buffer.view()).value(),
      c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()};
  const std::array normalized_view{
      c::TensorView::create(half_desc, normalized_buffer.view()).value()};
  const std::array cast_views{
      c::ConstTensorView::create(half_desc, normalized_buffer.view()).value()};
  const std::array converted_view{
      c::TensorView::create(output_f32_desc, converted_buffer.view()).value()};
  for (int seed : {5, 19}) {
    const auto joined = q::values(q::kMtpInput, seed);
    ASSERT_TRUE(first_buffer.view()
                    .write(std::as_bytes(std::span(joined).first(q::kHidden)))
                    .ok());
    ASSERT_TRUE(second_buffer.view()
                    .write(std::as_bytes(std::span(joined).last(q::kHidden)))
                    .ok());
    run(concat, queue, concat_views, joined_view);
    run(projection, queue, projection_views, projected_view);
    run(norm, queue, norm_views, normalized_view);
    run(cast, queue, cast_views, converted_view);
    const auto actual_projected = half_test::read<BF16>(projected_buffer);
    const auto actual_converted = half_test::read<float>(converted_buffer);
    const auto projected =
        q::sparse_projection(joined, q::kHidden, weight_seed);
    expect_near(actual_projected, projected);
    const auto projected_bf16 = q::as_bf16(projected);
    const auto normalized = q::rms_norm(projected_bf16, scale_values);
    ASSERT_EQ(actual_converted.size(), normalized.size());
    for (std::size_t i = 0; i < actual_converted.size(); ++i)
      EXPECT_NEAR(actual_converted[i], normalized[i],
                  std::abs(normalized[i]) * 0.012 + 2.0e-5)
          << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, CpuQwen38Profiles,
    ::testing::Values(&d5_qwen38_gdn_fixture::kOfficialProfile,
                      &d5_qwen38_gdn_fixture::kPrimaryGgufProfile));

} // namespace
