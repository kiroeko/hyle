#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen35_gguf_fragment_support.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/flashinfer.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;
using BF16 = hyle::simd::bfloat16_t;

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

c::TensorDesc tensor(c::DataType type,
                     std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

std::vector<BF16> to_bf16(std::span<const float> values) {
  std::vector<BF16> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(),
                 [](float value) { return half_test::encode<BF16>(value); });
  return result;
}

std::vector<float> to_float(std::span<const BF16> values) {
  std::vector<float> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(), [](BF16 value) {
    return static_cast<float>(half_test::decode(value));
  });
  return result;
}

hyle::mem::Buffer buffer(mem::MemoryResourceRef resource,
                         std::span<const std::byte> bytes) {
  auto result = hyle::mem::Buffer::create(resource, bytes.size(), 256).value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

template <class Value>
hyle::mem::Buffer buffer(mem::MemoryResourceRef resource,
                         std::span<const Value> values) {
  return buffer(resource, std::as_bytes(values));
}

hyle::mem::Buffer output(mem::MemoryResourceRef resource,
                         const c::TensorDesc &description) {
  return hyle::mem::Buffer::create(resource,
                                   description.required_bytes().value(), 256)
      .value();
}

struct Call final {
  c::PreparedOperation prepared;
  hyle::mem::Buffer workspace;
  c::OperationInvocation invocation;
};

Call prepare(const c::Context &context, mem::MemoryResourceRef resource,
             c::Operation operation, std::span<const c::TensorDesc> inputs,
             std::span<const c::TensorDesc> outputs,
             const c::WorkloadContext &workload) {
  const auto operation_name = std::string(c::operation_name(operation));
  auto result = c::prepare(context, std::move(operation), inputs, outputs,
                           resource, workload);
  if (!result)
    throw std::runtime_error(operation_name + ": " +
                             std::string(result.error().message()));
  auto prepared = std::move(*result);
  const auto requirement = prepared.workspace_requirement();
  auto workspace = requirement.size == 0
                       ? hyle::mem::Buffer{}
                       : hyle::mem::Buffer::create(resource, requirement.size,
                                                   requirement.alignment)
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

void expect_mature(const Call &call, const c::WorkloadContext &workload) {
  const auto provider = call.prepared.provider().id;
  EXPECT_TRUE(provider == c::ggml_cuda::kProviderId ||
              provider == c::cublas::kProviderId ||
              provider == c::cuda_native::kSglangProviderId ||
              provider == c::cuda_native::kCcclProviderId ||
              provider == c::flashinfer::kProviderId)
      << c::operation_name(call.prepared.kind()) << ": "
      << call.prepared.provider().name;
  EXPECT_EQ(call.prepared.workload(), workload);
}

d5_gguf_test::Packed packed(c::GgufQuantizedType type, std::int64_t columns,
                            std::int64_t reduction,
                            const c::TensorDesc &description, int seed) {
  const auto source = d5_gguf_test::values(columns, reduction, seed);
  return d5_gguf_test::pack(type, source, columns, reduction,
                            description.required_bytes().value());
}

TEST(CudaDefaultCompute,
     D5Qwen35PrimaryGgufFfnUsesMixedNativeWeightsAndStableReplay) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t hidden = d5_qwen35_fixture::kHidden;
  constexpr std::int64_t expanded = d5_qwen35_fixture::kFfnHidden;
  const auto activation_desc = dense(1, hidden);
  const auto scale_desc = vector(hidden);
  const auto gate_weight_desc =
      c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
                                 c::DataType::kFloat32, hidden, expanded)
          .value();
  const auto down_weight_desc =
      c::gguf_weight_description(c::GgufQuantizedType::kQ6K,
                                 c::DataType::kFloat32, expanded, hidden)
          .value();
  const auto expanded_desc = dense(1, expanded);
  const auto workload =
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kPrimaryGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x020f);

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      cuda::create_context({.metadata_allocator = stats.get()}).value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array norm_inputs{activation_desc, scale_desc};
  const std::array activation_output{activation_desc};
  auto norm = prepare(context, resource, c::Operation(c::RmsNorm{}),
                      norm_inputs, activation_output, workload);
  const std::array gate_inputs{activation_desc, gate_weight_desc};
  const std::array expanded_output{expanded_desc};
  auto gate = prepare(context, resource, c::Operation(c::MatMul{}), gate_inputs,
                      expanded_output, workload);
  auto up = prepare(context, resource, c::Operation(c::MatMul{}), gate_inputs,
                    expanded_output, workload);
  const std::array gated_inputs{expanded_desc, expanded_desc};
  auto gated = prepare(
      context, resource,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      gated_inputs, expanded_output, workload);
  const std::array down_inputs{expanded_desc, down_weight_desc};
  auto down = prepare(context, resource, c::Operation(c::MatMul{}), down_inputs,
                      activation_output, workload);

  expect_mature(norm, workload);
  EXPECT_EQ(gate.prepared.provider().id, c::ggml_cuda::kProviderId);
  EXPECT_EQ(up.prepared.provider().id, c::ggml_cuda::kProviderId);
  expect_mature(gated, workload);
  EXPECT_EQ(down.prepared.provider().id, c::ggml_cuda::kProviderId);
  for (const auto *call : {&norm, &gate, &up, &gated, &down})
    EXPECT_EQ(call->prepared.workload(), workload);

  const auto gate_packed = packed(c::GgufQuantizedType::kQ4K, expanded, hidden,
                                  gate_weight_desc, 31);
  const auto up_packed = packed(c::GgufQuantizedType::kQ4K, expanded, hidden,
                                gate_weight_desc, 37);
  const auto down_packed = packed(c::GgufQuantizedType::kQ6K, hidden, expanded,
                                  down_weight_desc, 41);
  const auto scale_values = d5_qwen35_gguf_fragment::scale(hidden);
  auto activation_values = d5_gguf_test::activation(hidden, 43);

  auto activation_buffer = buffer<float>(resource, activation_values);
  auto scale_buffer = buffer<float>(resource, scale_values);
  auto gate_weight_buffer = buffer(resource, gate_packed.bytes);
  auto up_weight_buffer = buffer(resource, up_packed.bytes);
  auto down_weight_buffer = buffer(resource, down_packed.bytes);
  auto normalized_buffer = output(resource, activation_desc);
  auto gate_buffer = output(resource, expanded_desc);
  auto up_buffer = output(resource, expanded_desc);
  auto gated_buffer = output(resource, expanded_desc);
  auto final_buffer = output(resource, activation_desc);

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

  std::uint64_t warmed_allocs = 0;
  std::uint64_t warmed_reallocs = 0;
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
    if (!previous_final.empty())
      EXPECT_NE(final_actual, previous_final);
    previous_final = final_actual;
    if (replay == 0) {
      const auto warmed = stats.snapshot();
      warmed_allocs = warmed.total_allocs;
      warmed_reallocs = warmed.reallocs;
    }
  }
  EXPECT_EQ(stats.snapshot().total_allocs, warmed_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, warmed_reallocs);
}

TEST(CudaDefaultCompute,
     D5Qwen35PrimaryGgufAttentionSemanticsUseMaturePreparedCalls) {
  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query_desc =
      tensor(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv_desc = tensor(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto scale_desc = tensor(c::DataType::kBFloat16, {depth});
  const auto position_desc =
      c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{1})
          .value();
  const auto frequency_desc = vector(depth / 2);
  const auto workload =
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kPrimaryGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x040f);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array query_norm_inputs{query_desc, scale_desc};
  const std::array query_output{query_desc};
  auto query_norm = prepare(context, resource, c::Operation(c::RmsNorm{}),
                            query_norm_inputs, query_output, workload);
  const std::array key_norm_inputs{kv_desc, scale_desc};
  const std::array key_output{kv_desc};
  auto key_norm = prepare(context, resource, c::Operation(c::RmsNorm{}),
                          key_norm_inputs, key_output, workload);
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  const std::array query_rope_inputs{query_desc, position_desc, frequency_desc};
  auto query_rope = prepare(context, resource, c::Operation(rope),
                            query_rope_inputs, query_output, workload);
  const std::array key_rope_inputs{kv_desc, position_desc, frequency_desc};
  auto key_rope = prepare(context, resource, c::Operation(rope),
                          key_rope_inputs, key_output, workload);
  const std::array attention_inputs{query_desc, kv_desc, kv_desc};
  auto attention =
      prepare(context, resource,
              c::Operation(c::ScaledDotProductAttention{
                  .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                  .scale = 0.0625F,
                  .causal = true,
                  .accumulation_type = c::DataType::kFloat32}),
              attention_inputs, query_output, workload);
  for (const auto *call :
       {&query_norm, &key_norm, &query_rope, &key_rope, &attention}) {
    expect_mature(*call, workload);
  }

  const auto query_scale_source = d5_qwen35_gguf_fragment::scale(depth);
  auto key_scale_source = query_scale_source;
  std::reverse(key_scale_source.begin(), key_scale_source.end());
  const auto query_scale_values = to_bf16(query_scale_source);
  const auto key_scale_values = to_bf16(key_scale_source);
  const auto query_scale_oracle = to_float(query_scale_values);
  const auto key_scale_oracle = to_float(key_scale_values);
  std::vector<float> frequencies(depth / 2);
  for (std::size_t index = 0; index < frequencies.size(); ++index)
    frequencies[index] = 1.0F / static_cast<float>(index + 1);
  constexpr std::array<std::int32_t, 1> positions{3};
  auto query_scale_buffer = buffer<BF16>(resource, query_scale_values);
  auto key_scale_buffer = buffer<BF16>(resource, key_scale_values);
  auto position_buffer = buffer<std::int32_t>(resource, positions);
  auto frequency_buffer = buffer<float>(resource, frequencies);
  std::vector<BF16> queries(query_desc.element_count().value());
  std::vector<BF16> keys(kv_desc.element_count().value());
  std::vector<BF16> values(kv_desc.element_count().value());
  auto query_buffer = buffer<BF16>(resource, queries);
  auto key_buffer = buffer<BF16>(resource, keys);
  auto value_buffer = buffer<BF16>(resource, values);
  auto normalized_query_buffer = output(resource, query_desc);
  auto normalized_key_buffer = output(resource, kv_desc);
  auto rotated_query_buffer = output(resource, query_desc);
  auto rotated_key_buffer = output(resource, kv_desc);
  auto attention_buffer = output(resource, query_desc);

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

  std::vector<BF16> previous_attention;
  for (int seed : {1, 7}) {
    for (std::size_t index = 0; index < queries.size(); ++index)
      queries[index] = half_test::encode<BF16>(
          static_cast<double>(static_cast<int>((index + seed) % 31) - 15) /
          64.0);
    for (std::size_t index = 0; index < keys.size(); ++index) {
      keys[index] = half_test::encode<BF16>(
          static_cast<double>(static_cast<int>((index + seed) % 29) - 14) /
          64.0);
      values[index] = half_test::encode<BF16>(
          static_cast<double>(seed * 100 + static_cast<int>(index)) / 256.0);
    }
    ASSERT_TRUE(
        query_buffer.view().write(std::as_bytes(std::span(queries))).ok());
    ASSERT_TRUE(key_buffer.view().write(std::as_bytes(std::span(keys))).ok());
    ASSERT_TRUE(
        value_buffer.view().write(std::as_bytes(std::span(values))).ok());
    run(query_norm, queue, query_norm_views, normalized_query_output);
    run(key_norm, queue, key_norm_views, normalized_key_output);
    const auto normalized_query =
        half_test::read<BF16>(normalized_query_buffer);
    const auto normalized_key = half_test::read<BF16>(normalized_key_buffer);
    const auto query_oracle = to_float(queries);
    const auto key_oracle = to_float(keys);
    const auto expected_query = d5_qwen35_gguf_fragment::rms_norm_rows(
        query_oracle, query_scale_oracle, depth);
    const auto expected_key = d5_qwen35_gguf_fragment::rms_norm_rows(
        key_oracle, key_scale_oracle, depth);
    for (std::size_t index = 0; index < normalized_query.size(); ++index)
      half_test::expect_near(normalized_query[index], expected_query[index]);
    for (std::size_t index = 0; index < normalized_key.size(); ++index)
      half_test::expect_near(normalized_key[index], expected_key[index]);
    run(query_rope, queue, query_rope_views, rotated_query_output);
    run(key_rope, queue, key_rope_views, rotated_key_output);
    const auto rotated_query = half_test::read<BF16>(rotated_query_buffer);
    const auto rotated_key = half_test::read<BF16>(rotated_key_buffer);
    const auto normalized_query_oracle = to_float(normalized_query);
    const auto normalized_key_oracle = to_float(normalized_key);
    const auto expected_rotated_query =
        d5_qwen35_gguf_fragment::rope_half_split(normalized_query_oracle, depth,
                                                 positions[0], frequencies);
    const auto expected_rotated_key = d5_qwen35_gguf_fragment::rope_half_split(
        normalized_key_oracle, depth, positions[0], frequencies);
    for (std::size_t index = 0; index < rotated_query.size(); ++index)
      half_test::expect_near(rotated_query[index],
                             expected_rotated_query[index]);
    for (std::size_t index = 0; index < rotated_key.size(); ++index)
      half_test::expect_near(rotated_key[index], expected_rotated_key[index]);
    run(attention, queue, attention_views, attention_output);
    const auto actual = half_test::read<BF16>(attention_buffer);
    for (std::size_t head = 0; head < query_heads; ++head)
      for (std::size_t column = 0; column < depth; ++column)
        EXPECT_EQ(
            actual[head * depth + column].bits,
            values[(head / (query_heads / kv_heads)) * depth + column].bits);
    if (!previous_attention.empty())
      EXPECT_NE(actual.front().bits, previous_attention.front().bits);
    previous_attention = actual;
  }
}

TEST(CudaDefaultCompute, D5Qwen35PrimaryGgufGdnStateUsesMaturePreparedCalls) {
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
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kPrimaryGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x050f);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array conv_inputs{conv_samples_desc, conv_weight_desc};
  const std::array conv_outputs{conv_output_desc};
  auto convolution = prepare(context, resource, c::model_ops::causal_conv1d(),
                             conv_inputs, conv_outputs, workload);
  const std::array gdn_inputs{qkv_desc,  qkv_desc,  qkv_desc,
                              gate_desc, gate_desc, state_desc};
  const std::array gdn_outputs{qkv_desc, state_desc};
  auto gdn = prepare(context, resource, c::Operation(c::GatedDeltaNet{}),
                     gdn_inputs, gdn_outputs, workload);
  for (const auto *call : {&convolution, &gdn}) {
    expect_mature(*call, workload);
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
  auto conv_weight_buffer = buffer<float>(resource, conv_weights);
  auto decay_buffer = buffer<float>(resource, log_decay);
  auto beta_buffer = buffer<float>(resource, beta);
  auto conv_samples_buffer = output(resource, conv_samples_desc);
  std::array qkv_buffers{output(resource, qkv_desc), output(resource, qkv_desc),
                         output(resource, qkv_desc)};
  auto state_buffer = output(resource, state_desc);
  auto conv_result_buffer = output(resource, conv_output_desc);
  auto score_buffer = output(resource, qkv_desc);
  auto new_state_buffer = output(resource, state_desc);
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
    if (!previous_scores.empty())
      EXPECT_NE(actual_scores, previous_scores);
    previous_scores = actual_scores;
  }
}

} // namespace
