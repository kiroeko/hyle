#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen35_gguf_fragment_support.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;

c::TensorDesc dense(std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array{rows, columns})
      .value();
}

c::TensorDesc vector(std::int64_t elements) {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array{elements})
      .value();
}

c::TensorDesc tensor(std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(c::DataType::kFloat32,
                              {shape.begin(), shape.size()})
      .value();
}

hyle::mem::Buffer buffer(std::span<const std::byte> bytes) {
  auto result =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), bytes.size(), 64)
          .value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

template <class Value> hyle::mem::Buffer buffer(std::span<const Value> values) {
  return buffer(std::as_bytes(values));
}

hyle::mem::Buffer output(const c::TensorDesc &description) {
  return hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                   description.required_bytes().value(), 64)
      .value();
}

struct Call final {
  c::PreparedOperation prepared;
  hyle::mem::Buffer workspace;
  c::OperationInvocation invocation;
};

Call prepare(const c::Context &context, c::Operation operation,
             std::span<const c::TensorDesc> inputs,
             std::span<const c::TensorDesc> outputs,
             const c::WorkloadContext &workload) {
  auto prepared = c::prepare(context, std::move(operation), inputs, outputs,
                             hyle::mem::system_resource(), workload)
                      .value();
  const auto requirement = prepared.workspace_requirement();
  auto workspace =
      requirement.size == 0
          ? hyle::mem::Buffer{}
          : hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                      requirement.size, requirement.alignment)
                .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  return {std::move(prepared), std::move(workspace), std::move(invocation)};
}

void run(Call &call, c::Queue &queue,
         std::span<const c::ConstTensorView> inputs,
         std::span<const c::TensorView> outputs) {
  auto completion = call.invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
}

d5_gguf_test::Packed packed(c::GgufQuantizedType type, std::int64_t columns,
                            std::int64_t reduction,
                            const c::TensorDesc &description, int seed) {
  const auto source = d5_gguf_test::values(columns, reduction, seed);
  return d5_gguf_test::pack(type, source, columns, reduction,
                            description.required_bytes().value());
}

void run_qwen35_gguf_ffn(const c::ModelArtifactProfileDescription &profile,
                         c::GgufQuantizedType gate_type,
                         c::GgufQuantizedType down_type) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t hidden = d5_qwen35_fixture::kHidden;
  constexpr std::int64_t expanded = d5_qwen35_fixture::kFfnHidden;
  const auto activation_desc = dense(1, hidden);
  const auto scale_desc = vector(hidden);
  const auto gate_weight_desc =
      c::gguf_weight_description(gate_type, c::DataType::kFloat32, hidden,
                                 expanded)
          .value();
  const auto down_weight_desc =
      c::gguf_weight_description(down_type, c::DataType::kFloat32, expanded,
                                 hidden)
          .value();
  const auto expanded_desc = dense(1, expanded);
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x020f);

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      c::cpu::create_context({.metadata_allocator = stats.get()}).value();
  auto queue = context.create_queue().value();
  const std::array norm_inputs{activation_desc, scale_desc};
  const std::array activation_output{activation_desc};
  auto norm = prepare(context, c::Operation(c::RmsNorm{}), norm_inputs,
                      activation_output, workload);
  const std::array gate_inputs{activation_desc, gate_weight_desc};
  const std::array expanded_output{expanded_desc};
  auto gate = prepare(context, c::Operation(c::MatMul{}), gate_inputs,
                      expanded_output, workload);
  auto up = prepare(context, c::Operation(c::MatMul{}), gate_inputs,
                    expanded_output, workload);
  const std::array gated_inputs{expanded_desc, expanded_desc};
  auto gated = prepare(
      context,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      gated_inputs, expanded_output, workload);
  const std::array down_inputs{expanded_desc, down_weight_desc};
  auto down = prepare(context, c::Operation(c::MatMul{}), down_inputs,
                      activation_output, workload);

  EXPECT_TRUE(norm.prepared.provider().id == c::onednn::kProviderId ||
              norm.prepared.provider().id == c::ggml_cpu::kProviderId);
  EXPECT_EQ(gate.prepared.provider().id, c::ggml_cpu::kProviderId);
  EXPECT_EQ(up.prepared.provider().id, c::ggml_cpu::kProviderId);
  EXPECT_TRUE(gated.prepared.provider().id == c::onednn::kProviderId ||
              gated.prepared.provider().id == c::ggml_cpu::kProviderId);
  EXPECT_EQ(down.prepared.provider().id, c::ggml_cpu::kProviderId);
  for (const auto *call : {&norm, &gate, &up, &gated, &down})
    EXPECT_EQ(call->prepared.workload(), workload);

  const auto gate_packed =
      packed(gate_type, expanded, hidden, gate_weight_desc, 31);
  const auto up_packed =
      packed(gate_type, expanded, hidden, gate_weight_desc, 37);
  const auto down_packed =
      packed(down_type, hidden, expanded, down_weight_desc, 41);
  const auto scale_values = d5_qwen35_gguf_fragment::scale(hidden);
  auto activation_values = d5_gguf_test::activation(hidden, 43);

  auto activation_buffer = buffer<float>(activation_values);
  auto scale_buffer = buffer<float>(scale_values);
  auto gate_weight_buffer = buffer(gate_packed.bytes);
  auto up_weight_buffer = buffer(up_packed.bytes);
  auto down_weight_buffer = buffer(down_packed.bytes);
  auto normalized_buffer = output(activation_desc);
  auto gate_buffer = output(expanded_desc);
  auto up_buffer = output(expanded_desc);
  auto gated_buffer = output(expanded_desc);
  auto final_buffer = output(activation_desc);

  const auto activation =
      c::ConstTensorView::create(activation_desc, activation_buffer.view())
          .value();
  const auto scale =
      c::ConstTensorView::create(scale_desc, scale_buffer.view()).value();
  const auto gate_weight =
      c::ConstTensorView::create(gate_weight_desc, gate_weight_buffer.view())
          .value();
  const auto up_weight =
      c::ConstTensorView::create(gate_weight_desc, up_weight_buffer.view())
          .value();
  const auto down_weight =
      c::ConstTensorView::create(down_weight_desc, down_weight_buffer.view())
          .value();
  const std::array norm_views{activation, scale};
  const std::array normalized_output{
      c::TensorView::create(activation_desc, normalized_buffer.view()).value()};
  const auto normalized =
      c::ConstTensorView::create(activation_desc, normalized_buffer.view())
          .value();
  const std::array gate_views{normalized, gate_weight};
  const std::array up_views{normalized, up_weight};
  const std::array gate_output{
      c::TensorView::create(expanded_desc, gate_buffer.view()).value()};
  const std::array up_output{
      c::TensorView::create(expanded_desc, up_buffer.view()).value()};
  const std::array gated_views{
      c::ConstTensorView::create(expanded_desc, gate_buffer.view()).value(),
      c::ConstTensorView::create(expanded_desc, up_buffer.view()).value()};
  const std::array gated_output{
      c::TensorView::create(expanded_desc, gated_buffer.view()).value()};
  const std::array down_views{
      c::ConstTensorView::create(expanded_desc, gated_buffer.view()).value(),
      down_weight};
  const std::array final_output{
      c::TensorView::create(activation_desc, final_buffer.view()).value()};

  const auto before_replay = stats.snapshot();
  std::vector<float> previous_final;
  for (int replay = 0; replay < 2; ++replay) {
    if (replay != 0) {
      for (auto &value : activation_values)
        value = value * -0.625F + 0.00075F;
      ASSERT_TRUE(activation_buffer.view()
                      .write(std::as_bytes(std::span(activation_values)))
                      .ok());
    }

    run(norm, queue, norm_views, normalized_output);
    const auto normalized_actual = half_test::read<float>(normalized_buffer);
    const auto normalized_expected =
        d5_qwen35_gguf_fragment::rms_norm(activation_values, scale_values);
    d5_qwen35_gguf_fragment::expect_close(
        normalized_actual, normalized_expected, 3.0e-5F, 2.0e-5F);

    run(gate, queue, gate_views, gate_output);
    run(up, queue, up_views, up_output);
    const auto gate_actual = half_test::read<float>(gate_buffer);
    const auto up_actual = half_test::read<float>(up_buffer);
    const auto gate_expected = d5_gguf_test::matmul(
        normalized_actual, gate_packed.decoded, expanded, hidden);
    const auto up_expected = d5_gguf_test::matmul(
        normalized_actual, up_packed.decoded, expanded, hidden);
    d5_gguf_test::expect_quantized_chain_matmul_close(gate_actual,
                                                      gate_expected);
    d5_gguf_test::expect_quantized_chain_matmul_close(up_actual, up_expected);

    run(gated, queue, gated_views, gated_output);
    const auto gated_actual = half_test::read<float>(gated_buffer);
    const auto gated_expected =
        d5_qwen35_gguf_fragment::silu_gate(gate_actual, up_actual);
    d5_qwen35_gguf_fragment::expect_close(gated_actual, gated_expected, 2.0e-5F,
                                          2.0e-5F);

    run(down, queue, down_views, final_output);
    const auto final_actual = half_test::read<float>(final_buffer);
    const auto final_expected = d5_gguf_test::matmul(
        gated_actual, down_packed.decoded, hidden, expanded);
    d5_gguf_test::expect_quantized_chain_matmul_close(final_actual,
                                                      final_expected);
    if (!previous_final.empty()) {
      EXPECT_NE(final_actual, previous_final);
    }
    previous_final = final_actual;
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before_replay.total_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, before_replay.reallocs);
}

TEST(CpuDefaultDirect,
     D5Qwen35PrimaryGgufFfnUsesMixedNativeWeightsAndStableReplay) {
  run_qwen35_gguf_ffn(d5_qwen35_fixture::kPrimaryGgufProfile,
                      c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K);
}

TEST(CpuDefaultDirect,
     D5Qwen35FidelityGgufFfnUsesQ8NativeWeightsAndStableReplay) {
  run_qwen35_gguf_ffn(d5_qwen35_fixture::kFidelityGgufProfile,
                      c::GgufQuantizedType::kQ8_0, c::GgufQuantizedType::kQ8_0);
}

void run_qwen35_gguf_attention(
    const c::ModelArtifactProfileDescription &profile) {
  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query_desc = tensor({1, 1, query_heads, depth});
  const auto kv_desc = tensor({1, 1, kv_heads, depth});
  const auto scale_desc = vector(depth);
  const auto position_desc =
      c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{1})
          .value();
  const auto frequency_desc = vector(depth / 2);
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x040f);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array query_norm_inputs{query_desc, scale_desc};
  const std::array query_output{query_desc};
  auto query_norm = prepare(context, c::Operation(c::RmsNorm{}),
                            query_norm_inputs, query_output, workload);
  const std::array key_norm_inputs{kv_desc, scale_desc};
  const std::array key_output{kv_desc};
  auto key_norm = prepare(context, c::Operation(c::RmsNorm{}), key_norm_inputs,
                          key_output, workload);
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  const std::array query_rope_inputs{query_desc, position_desc, frequency_desc};
  auto query_rope = prepare(context, c::Operation(rope), query_rope_inputs,
                            query_output, workload);
  const std::array key_rope_inputs{kv_desc, position_desc, frequency_desc};
  auto key_rope = prepare(context, c::Operation(rope), key_rope_inputs,
                          key_output, workload);
  const std::array attention_inputs{query_desc, kv_desc, kv_desc};
  auto attention =
      prepare(context,
              c::Operation(c::ScaledDotProductAttention{
                  .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                  .scale = 0.0625F,
                  .causal = true,
                  .accumulation_type = c::DataType::kFloat32}),
              attention_inputs, query_output, workload);
  for (const auto *call :
       {&query_norm, &key_norm, &query_rope, &key_rope, &attention}) {
    EXPECT_TRUE(call->prepared.provider().id == c::ggml_cpu::kProviderId ||
                call->prepared.provider().id == c::onednn::kProviderId)
        << call->prepared.provider().name;
    EXPECT_EQ(call->prepared.workload(), workload);
  }

  const auto query_scale_values = d5_qwen35_gguf_fragment::scale(depth);
  auto key_scale_values = query_scale_values;
  std::reverse(key_scale_values.begin(), key_scale_values.end());
  std::vector<float> frequencies(depth / 2);
  for (std::size_t index = 0; index < frequencies.size(); ++index)
    frequencies[index] = 1.0F / static_cast<float>(index + 1);
  constexpr std::array<std::int32_t, 1> positions{3};
  auto query_scale_buffer = buffer<float>(query_scale_values);
  auto key_scale_buffer = buffer<float>(key_scale_values);
  auto position_buffer = buffer<std::int32_t>(positions);
  auto frequency_buffer = buffer<float>(frequencies);
  std::vector<float> queries(query_desc.element_count().value());
  std::vector<float> keys(kv_desc.element_count().value());
  std::vector<float> values(kv_desc.element_count().value());
  auto query_buffer = buffer<float>(queries);
  auto key_buffer = buffer<float>(keys);
  auto value_buffer = buffer<float>(values);
  auto normalized_query_buffer = output(query_desc);
  auto normalized_key_buffer = output(kv_desc);
  auto rotated_query_buffer = output(query_desc);
  auto rotated_key_buffer = output(kv_desc);
  auto attention_buffer = output(query_desc);

  const auto query =
      c::ConstTensorView::create(query_desc, query_buffer.view()).value();
  const auto key =
      c::ConstTensorView::create(kv_desc, key_buffer.view()).value();
  const auto value =
      c::ConstTensorView::create(kv_desc, value_buffer.view()).value();
  const auto query_scale =
      c::ConstTensorView::create(scale_desc, query_scale_buffer.view()).value();
  const auto key_scale =
      c::ConstTensorView::create(scale_desc, key_scale_buffer.view()).value();
  const auto position =
      c::ConstTensorView::create(position_desc, position_buffer.view()).value();
  const auto frequency =
      c::ConstTensorView::create(frequency_desc, frequency_buffer.view())
          .value();
  const std::array query_norm_views{query, query_scale};
  const std::array key_norm_views{key, key_scale};
  const std::array normalized_query_output{
      c::TensorView::create(query_desc, normalized_query_buffer.view())
          .value()};
  const std::array normalized_key_output{
      c::TensorView::create(kv_desc, normalized_key_buffer.view()).value()};
  const std::array query_rope_views{
      c::ConstTensorView::create(query_desc, normalized_query_buffer.view())
          .value(),
      position, frequency};
  const std::array key_rope_views{
      c::ConstTensorView::create(kv_desc, normalized_key_buffer.view()).value(),
      position, frequency};
  const std::array rotated_query_output{
      c::TensorView::create(query_desc, rotated_query_buffer.view()).value()};
  const std::array rotated_key_output{
      c::TensorView::create(kv_desc, rotated_key_buffer.view()).value()};
  const std::array attention_views{
      c::ConstTensorView::create(query_desc, rotated_query_buffer.view())
          .value(),
      c::ConstTensorView::create(kv_desc, rotated_key_buffer.view()).value(),
      value};
  const std::array attention_output{
      c::TensorView::create(query_desc, attention_buffer.view()).value()};

  std::vector<float> previous_attention;
  for (int seed : {1, 7}) {
    for (std::size_t index = 0; index < queries.size(); ++index)
      queries[index] =
          static_cast<float>(static_cast<int>((index + seed) % 31) - 15) /
          64.0F;
    for (std::size_t index = 0; index < keys.size(); ++index) {
      keys[index] =
          static_cast<float>(static_cast<int>((index + seed) % 29) - 14) /
          64.0F;
      values[index] =
          static_cast<float>(seed * 100 + static_cast<int>(index)) / 256.0F;
    }
    ASSERT_TRUE(
        query_buffer.view().write(std::as_bytes(std::span(queries))).ok());
    ASSERT_TRUE(key_buffer.view().write(std::as_bytes(std::span(keys))).ok());
    ASSERT_TRUE(
        value_buffer.view().write(std::as_bytes(std::span(values))).ok());
    run(query_norm, queue, query_norm_views, normalized_query_output);
    run(key_norm, queue, key_norm_views, normalized_key_output);
    const auto normalized_query =
        half_test::read<float>(normalized_query_buffer);
    const auto normalized_key = half_test::read<float>(normalized_key_buffer);
    const auto expected_query = d5_qwen35_gguf_fragment::rms_norm_rows(
        queries, query_scale_values, depth);
    const auto expected_key =
        d5_qwen35_gguf_fragment::rms_norm_rows(keys, key_scale_values, depth);
    d5_qwen35_gguf_fragment::expect_close(normalized_query, expected_query,
                                          3.0e-5F, 2.0e-5F);
    d5_qwen35_gguf_fragment::expect_close(normalized_key, expected_key, 3.0e-5F,
                                          2.0e-5F);
    run(query_rope, queue, query_rope_views, rotated_query_output);
    run(key_rope, queue, key_rope_views, rotated_key_output);
    const auto rotated_query = half_test::read<float>(rotated_query_buffer);
    const auto rotated_key = half_test::read<float>(rotated_key_buffer);
    const auto expected_rotated_query =
        d5_qwen35_gguf_fragment::rope_half_split(normalized_query, depth,
                                                 positions[0], frequencies);
    const auto expected_rotated_key = d5_qwen35_gguf_fragment::rope_half_split(
        normalized_key, depth, positions[0], frequencies);
    d5_qwen35_gguf_fragment::expect_close(rotated_query, expected_rotated_query,
                                          5.0e-5F, 2.0e-5F);
    d5_qwen35_gguf_fragment::expect_close(rotated_key, expected_rotated_key,
                                          5.0e-5F, 2.0e-5F);
    run(attention, queue, attention_views, attention_output);
    const auto actual = half_test::read<float>(attention_buffer);
    for (std::size_t head = 0; head < query_heads; ++head)
      for (std::size_t column = 0; column < depth; ++column)
        EXPECT_NEAR(actual[head * depth + column],
                    values[(head / (query_heads / kv_heads)) * depth + column],
                    2.0e-6F);
    if (!previous_attention.empty()) {
      EXPECT_NE(actual, previous_attention);
    }
    previous_attention = actual;
  }
}

TEST(CpuDefaultDirect,
     D5Qwen35PrimaryGgufAttentionSemanticsUseMaturePreparedCalls) {
  run_qwen35_gguf_attention(d5_qwen35_fixture::kPrimaryGgufProfile);
}

TEST(CpuDefaultDirect,
     D5Qwen35FidelityGgufAttentionSemanticsUseMaturePreparedCalls) {
  run_qwen35_gguf_attention(d5_qwen35_fixture::kFidelityGgufProfile);
}

void run_qwen35_gguf_gdn(const c::ModelArtifactProfileDescription &profile) {
  constexpr std::size_t heads = d5_qwen35_fixture::kGdnHeads;
  constexpr std::size_t depth = d5_qwen35_fixture::kGdnDepth;
  constexpr std::size_t channels = d5_qwen35_fixture::kConvChannels;
  constexpr std::size_t kernel = d5_qwen35_fixture::kConvKernel;
  const auto conv_samples_desc = tensor({1, static_cast<std::int64_t>(kernel),
                                         static_cast<std::int64_t>(channels)});
  const auto conv_weight_desc = tensor(
      {static_cast<std::int64_t>(channels), static_cast<std::int64_t>(kernel)});
  const auto conv_output_desc =
      tensor({1, 1, static_cast<std::int64_t>(channels)});
  const auto qkv_desc = tensor({1, 1, static_cast<std::int64_t>(heads),
                                static_cast<std::int64_t>(depth)});
  const auto gate_desc = tensor({1, 1, static_cast<std::int64_t>(heads), 1});
  const auto state_desc = tensor({1, static_cast<std::int64_t>(heads),
                                  static_cast<std::int64_t>(depth),
                                  static_cast<std::int64_t>(depth)});
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x050f);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array conv_inputs{conv_samples_desc, conv_weight_desc};
  const std::array conv_outputs{conv_output_desc};
  auto convolution = prepare(context, c::model_ops::causal_conv1d(),
                             conv_inputs, conv_outputs, workload);
  const std::array gdn_inputs{qkv_desc,  qkv_desc,  qkv_desc,
                              gate_desc, gate_desc, state_desc};
  const std::array gdn_outputs{qkv_desc, state_desc};
  auto gdn = prepare(context, c::Operation(c::GatedDeltaNet{}), gdn_inputs,
                     gdn_outputs, workload);
  for (const auto *call : {&convolution, &gdn}) {
    EXPECT_TRUE(call->prepared.provider().id == c::ggml_cpu::kProviderId ||
                call->prepared.provider().id == c::onednn::kProviderId)
        << call->prepared.provider().name;
    EXPECT_EQ(call->prepared.workload(), workload);
  }

  std::vector<float> conv_weights(channels * kernel);
  for (std::size_t index = 0; index < conv_weights.size(); ++index)
    conv_weights[index] =
        static_cast<float>(static_cast<int>((index * 7 + 3) % 23) - 11) /
        256.0F;
  std::vector<float> log_decay(heads), beta(heads);
  for (std::size_t head = 0; head < heads; ++head) {
    log_decay[head] = -0.015F - static_cast<float>(head) * 0.0005F;
    beta[head] = 0.2F + static_cast<float>(head) * 0.01F;
  }
  auto conv_weight_buffer = buffer<float>(conv_weights);
  auto decay_buffer = buffer<float>(log_decay);
  auto beta_buffer = buffer<float>(beta);
  auto conv_samples_buffer = output(conv_samples_desc);
  std::array qkv_buffers{output(qkv_desc), output(qkv_desc), output(qkv_desc)};
  auto state_buffer = output(state_desc);
  auto conv_result_buffer = output(conv_output_desc);
  auto score_buffer = output(qkv_desc);
  auto new_state_buffer = output(state_desc);
  const std::array conv_views{
      c::ConstTensorView::create(conv_samples_desc, conv_samples_buffer.view())
          .value(),
      c::ConstTensorView::create(conv_weight_desc, conv_weight_buffer.view())
          .value()};
  const std::array conv_result{
      c::TensorView::create(conv_output_desc, conv_result_buffer.view())
          .value()};
  const std::array gdn_views{
      c::ConstTensorView::create(qkv_desc, qkv_buffers[0].view()).value(),
      c::ConstTensorView::create(qkv_desc, qkv_buffers[1].view()).value(),
      c::ConstTensorView::create(qkv_desc, qkv_buffers[2].view()).value(),
      c::ConstTensorView::create(gate_desc, decay_buffer.view()).value(),
      c::ConstTensorView::create(gate_desc, beta_buffer.view()).value(),
      c::ConstTensorView::create(state_desc, state_buffer.view()).value()};
  const std::array gdn_result{
      c::TensorView::create(qkv_desc, score_buffer.view()).value(),
      c::TensorView::create(state_desc, new_state_buffer.view()).value()};

  std::vector<float> previous_scores;
  for (int seed : {2, 9}) {
    std::vector<float> conv_samples(channels * kernel);
    std::array<std::vector<float>, 3> qkv;
    for (auto &values : qkv)
      values.resize(heads * depth);
    std::vector<float> state(heads * depth * depth);
    for (std::size_t index = 0; index < conv_samples.size(); ++index)
      conv_samples[index] =
          static_cast<float>(static_cast<int>((index * 17 + seed * 5) % 61) -
                             30) /
          128.0F;
    for (std::size_t port = 0; port < qkv.size(); ++port)
      for (std::size_t index = 0; index < qkv[port].size(); ++index)
        qkv[port][index] =
            static_cast<float>(std::sin(
                static_cast<double>(index + port * 37 + seed) * 0.017)) *
            0.025F;
    for (std::size_t index = 0; index < state.size(); ++index)
      state[index] = static_cast<float>(std::sin(
                         static_cast<double>(index + seed * 19) * 0.009)) *
                     0.0005F;
    ASSERT_TRUE(conv_samples_buffer.view()
                    .write(std::as_bytes(std::span(conv_samples)))
                    .ok());
    for (std::size_t port = 0; port < qkv.size(); ++port)
      ASSERT_TRUE(qkv_buffers[port]
                      .view()
                      .write(std::as_bytes(std::span(qkv[port])))
                      .ok());
    ASSERT_TRUE(
        state_buffer.view().write(std::as_bytes(std::span(state))).ok());
    run(convolution, queue, conv_views, conv_result);
    run(gdn, queue, gdn_views, gdn_result);
    const auto actual_conv = half_test::read<float>(conv_result_buffer);
    const auto expected_conv = d5_qwen35_gguf_fragment::causal_conv(
        conv_samples, conv_weights, channels, kernel);
    d5_qwen35_gguf_fragment::expect_close(actual_conv, expected_conv, 1.0e-6F,
                                          2.0e-5F);
    const auto actual_scores = half_test::read<float>(score_buffer);
    const auto actual_state = half_test::read<float>(new_state_buffer);
    const auto expected = d5_qwen35_gguf_fragment::gated_delta_net(
        qkv[0], qkv[1], qkv[2], log_decay, beta, state, heads, depth);
    d5_qwen35_gguf_fragment::expect_close(actual_scores, expected.scores,
                                          4.0e-6F);
    d5_qwen35_gguf_fragment::expect_close(actual_state, expected.state,
                                          4.0e-6F);
    if (!previous_scores.empty()) {
      EXPECT_NE(actual_scores, previous_scores);
    }
    previous_scores = actual_scores;
  }
}

TEST(CpuDefaultDirect, D5Qwen35PrimaryGgufGdnStateUsesMaturePreparedCalls) {
  run_qwen35_gguf_gdn(d5_qwen35_fixture::kPrimaryGgufProfile);
}

TEST(CpuDefaultDirect, D5Qwen35FidelityGgufGdnStateUsesMaturePreparedCalls) {
  run_qwen35_gguf_gdn(d5_qwen35_fixture::kFidelityGgufProfile);
}

} // namespace
