#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen35_gguf_fragment_support.h"
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

d5_gguf_test::Packed packed(c::GgufQuantizedType type, std::int64_t columns,
                            std::int64_t reduction,
                            const c::TensorDesc &description, int seed) {
  const auto source = d5_gguf_test::values(columns, reduction, seed);
  return d5_gguf_test::pack(type, source, columns, reduction,
                            description.required_bytes().value());
}

template <class Compiled> void expect_mature(const Compiled &compiled) {
  ASSERT_GT(compiled.report.step_count(), 0U);
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step) {
    const auto report = compiled.report.step(step).value();
    const auto provider = report.provider.id;
    EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
                provider == c::onednn::kProviderId ||
                provider == c::aten_cpu::kProviderId ||
                provider == g::onednn::kProviderId)
        << report.provider.name;
  }
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
  const auto final = builder.add(c::MatMul{}, gated, down_weight).value();
  for (const auto value : {normalized, gate, up, gated, final})
    ASSERT_TRUE(builder.publish(value).ok());
  auto logical = std::move(builder).finish().value();

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      g::cpu::create_context({.metadata_allocator = stats.get()}).value();
  const std::array constants{
      g::ConstantBinding{
          scale,
          c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()},
      g::ConstantBinding{gate_weight,
                         c::ConstTensorView::create(gate_weight_desc,
                                                    gate_weight_buffer.view())
                             .value()},
      g::ConstantBinding{
          up_weight,
          c::ConstTensorView::create(gate_weight_desc, up_weight_buffer.view())
              .value()},
      g::ConstantBinding{down_weight,
                         c::ConstTensorView::create(down_weight_desc,
                                                    down_weight_buffer.view())
                             .value()}};
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x020f);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  ASSERT_GT(compiled.report.step_count(), 0U);
  std::size_t ggml_nodes = 0;
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step) {
    const auto report = compiled.report.step(step).value();
    const auto provider = report.provider.id;
    EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
                provider == c::onednn::kProviderId ||
                provider == c::aten_cpu::kProviderId ||
                provider == g::onednn::kProviderId)
        << report.provider.name;
    if (provider == c::ggml_cpu::kProviderId)
      ggml_nodes += report.node_count;
  }
  EXPECT_GE(ggml_nodes, 3U);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(
      invocation
          .bind(activation, c::ConstTensorView::create(activation_desc,
                                                       activation_buffer.view())
                                .value())
          .ok());
  ASSERT_TRUE(
      invocation
          .bind(normalized,
                c::TensorView::create(activation_desc, normalized_buffer.view())
                    .value())
          .ok());
  ASSERT_TRUE(
      invocation
          .bind(
              gate,
              c::TensorView::create(expanded_desc, gate_buffer.view()).value())
          .ok());
  ASSERT_TRUE(
      invocation
          .bind(up,
                c::TensorView::create(expanded_desc, up_buffer.view()).value())
          .ok());
  ASSERT_TRUE(
      invocation
          .bind(
              gated,
              c::TensorView::create(expanded_desc, gated_buffer.view()).value())
          .ok());
  ASSERT_TRUE(invocation
                  .bind(final, c::TensorView::create(activation_desc,
                                                     final_buffer.view())
                                   .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
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
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());

    const auto normalized_actual = half_test::read<float>(normalized_buffer);
    const auto normalized_expected =
        d5_qwen35_gguf_fragment::rms_norm(activation_values, scale_values);
    d5_qwen35_gguf_fragment::expect_close(
        normalized_actual, normalized_expected, 3.0e-5F, 2.0e-5F);
    const auto gate_actual = half_test::read<float>(gate_buffer);
    const auto up_actual = half_test::read<float>(up_buffer);
    const auto gate_expected = d5_gguf_test::matmul(
        normalized_actual, gate_packed.decoded, expanded, hidden);
    const auto up_expected = d5_gguf_test::matmul(
        normalized_actual, up_packed.decoded, expanded, hidden);
    d5_gguf_test::expect_quantized_chain_matmul_close(gate_actual,
                                                      gate_expected);
    d5_gguf_test::expect_quantized_chain_matmul_close(up_actual, up_expected);
    const auto gated_actual = half_test::read<float>(gated_buffer);
    const auto gated_expected =
        d5_qwen35_gguf_fragment::silu_gate(gate_actual, up_actual);
    d5_qwen35_gguf_fragment::expect_close(gated_actual, gated_expected, 2.0e-5F,
                                          2.0e-5F);
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

TEST(CpuDefaultGraph,
     D5Qwen35PrimaryGgufFfnUsesMixedMatureRegionsAndStableReplay) {
  run_qwen35_gguf_ffn(d5_qwen35_fixture::kPrimaryGgufProfile,
                      c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K);
}

TEST(CpuDefaultGraph,
     D5Qwen35FidelityGgufFfnUsesQ8MatureRegionsAndStableReplay) {
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
  const auto normalized_key = builder.add(c::RmsNorm{}, key, key_scale).value();
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  const auto rotated_query =
      builder.add(rope, normalized_query, position, frequency).value();
  const auto rotated_key =
      builder.add(rope, normalized_key, position, frequency).value();
  const auto attention =
      builder
          .add(
              c::ScaledDotProductAttention{
                  .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                  .scale = 0.0625F,
                  .causal = true,
                  .accumulation_type = c::DataType::kFloat32},
              rotated_query, rotated_key, value)
          .value();
  for (const auto result : {normalized_query, normalized_key, rotated_query,
                            rotated_key, attention})
    ASSERT_TRUE(builder.publish(result).ok());
  auto logical = std::move(builder).finish().value();

  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{query_scale, c::ConstTensorView::create(
                                          scale_desc, query_scale_buffer.view())
                                          .value()},
      g::ConstantBinding{key_scale, c::ConstTensorView::create(
                                        scale_desc, key_scale_buffer.view())
                                        .value()},
      g::ConstantBinding{position, c::ConstTensorView::create(
                                       position_desc, position_buffer.view())
                                       .value()},
      g::ConstantBinding{frequency, c::ConstTensorView::create(
                                        frequency_desc, frequency_buffer.view())
                                        .value()}};
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x040f);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);

  auto invocation = compiled.executable.create_invocation().value();
  const std::array input_bindings{std::pair{query, &query_buffer},
                                  std::pair{key, &key_buffer},
                                  std::pair{value, &value_buffer}};
  for (const auto &[id, storage] : input_bindings)
    ASSERT_TRUE(
        invocation
            .bind(id, c::ConstTensorView::create(
                          id == query ? query_desc : kv_desc, storage->view())
                          .value())
            .ok());
  const std::array output_bindings{
      std::pair{normalized_query, &normalized_query_buffer},
      std::pair{normalized_key, &normalized_key_buffer},
      std::pair{rotated_query, &rotated_query_buffer},
      std::pair{rotated_key, &rotated_key_buffer},
      std::pair{attention, &attention_buffer}};
  for (const auto &[id, storage] : output_bindings) {
    const auto description =
        id == normalized_query || id == rotated_query || id == attention
            ? query_desc
            : kv_desc;
    ASSERT_TRUE(
        invocation
            .bind(id,
                  c::TensorView::create(description, storage->view()).value())
            .ok());
  }
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
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
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto normalized_query_actual =
        half_test::read<float>(normalized_query_buffer);
    const auto normalized_key_actual =
        half_test::read<float>(normalized_key_buffer);
    const auto expected_query = d5_qwen35_gguf_fragment::rms_norm_rows(
        queries, query_scale_values, depth);
    const auto expected_key =
        d5_qwen35_gguf_fragment::rms_norm_rows(keys, key_scale_values, depth);
    d5_qwen35_gguf_fragment::expect_close(normalized_query_actual,
                                          expected_query, 3.0e-5F, 2.0e-5F);
    d5_qwen35_gguf_fragment::expect_close(normalized_key_actual, expected_key,
                                          3.0e-5F, 2.0e-5F);
    const auto rotated_query_actual =
        half_test::read<float>(rotated_query_buffer);
    const auto rotated_key_actual = half_test::read<float>(rotated_key_buffer);
    const auto expected_rotated_query =
        d5_qwen35_gguf_fragment::rope_half_split(normalized_query_actual, depth,
                                                 positions[0], frequencies);
    const auto expected_rotated_key = d5_qwen35_gguf_fragment::rope_half_split(
        normalized_key_actual, depth, positions[0], frequencies);
    d5_qwen35_gguf_fragment::expect_close(
        rotated_query_actual, expected_rotated_query, 5.0e-5F, 2.0e-5F);
    d5_qwen35_gguf_fragment::expect_close(
        rotated_key_actual, expected_rotated_key, 5.0e-5F, 2.0e-5F);
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

TEST(CpuDefaultGraph,
     D5Qwen35PrimaryGgufAttentionSemanticsUseMatureRegionsAndReplay) {
  run_qwen35_gguf_attention(d5_qwen35_fixture::kPrimaryGgufProfile);
}

TEST(CpuDefaultGraph,
     D5Qwen35FidelityGgufAttentionSemanticsUseMatureRegionsAndReplay) {
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
      builder.add(c::model_ops::causal_conv1d(), conv_samples, conv_weight)
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

  auto context = g::cpu::create_context().value();
  const std::array constants{
      g::ConstantBinding{conv_weight,
                         c::ConstTensorView::create(conv_weight_desc,
                                                    conv_weight_buffer.view())
                             .value()},
      g::ConstantBinding{
          decay,
          c::ConstTensorView::create(gate_desc, decay_buffer.view()).value()},
      g::ConstantBinding{
          update_gate,
          c::ConstTensorView::create(gate_desc, beta_buffer.view()).value()}};
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x050f);
  auto result = g::compile(logical, context,
                           {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.executable.workload(), workload);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(conv_samples,
                        c::ConstTensorView::create(conv_samples_desc,
                                                   conv_samples_buffer.view())
                            .value())
                  .ok());
  const std::array qkv_ids{query, key, value};
  for (std::size_t port = 0; port < qkv_ids.size(); ++port)
    ASSERT_TRUE(invocation
                    .bind(qkv_ids[port], c::ConstTensorView::create(
                                             qkv_desc, qkv_buffers[port].view())
                                             .value())
                    .ok());
  ASSERT_TRUE(invocation
                  .bind(old_state, c::ConstTensorView::create(
                                       state_desc, state_buffer.view())
                                       .value())
                  .ok());
  ASSERT_TRUE(
      invocation
          .bind(convolution, c::TensorView::create(conv_output_desc,
                                                   conv_result_buffer.view())
                                 .value())
          .ok());
  ASSERT_TRUE(
      invocation
          .bind(gdn_outputs[0],
                c::TensorView::create(qkv_desc, score_buffer.view()).value())
          .ok());
  ASSERT_TRUE(invocation
                  .bind(gdn_outputs[1], c::TensorView::create(
                                            state_desc, new_state_buffer.view())
                                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  std::vector<float> previous_scores;
  for (int seed : {2, 9}) {
    std::vector<float> conv_samples_values(channels * kernel);
    std::array<std::vector<float>, 3> qkv;
    for (auto &values : qkv)
      values.resize(heads * depth);
    std::vector<float> state(heads * depth * depth);
    for (std::size_t index = 0; index < conv_samples_values.size(); ++index)
      conv_samples_values[index] =
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
                    .write(std::as_bytes(std::span(conv_samples_values)))
                    .ok());
    for (std::size_t port = 0; port < qkv.size(); ++port)
      ASSERT_TRUE(qkv_buffers[port]
                      .view()
                      .write(std::as_bytes(std::span(qkv[port])))
                      .ok());
    ASSERT_TRUE(
        state_buffer.view().write(std::as_bytes(std::span(state))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual_conv = half_test::read<float>(conv_result_buffer);
    const auto expected_conv = d5_qwen35_gguf_fragment::causal_conv(
        conv_samples_values, conv_weights, channels, kernel);
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

TEST(CpuDefaultGraph, D5Qwen35PrimaryGgufGdnStateUsesMatureRegionsAndReplay) {
  run_qwen35_gguf_gdn(d5_qwen35_fixture::kPrimaryGgufProfile);
}

TEST(CpuDefaultGraph, D5Qwen35FidelityGgufGdnStateUsesMatureRegionsAndReplay) {
  run_qwen35_gguf_gdn(d5_qwen35_fixture::kFidelityGgufProfile);
}

} // namespace
