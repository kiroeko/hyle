#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen38_official_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
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

std::vector<BF16> decode_fixture(std::span<const std::byte> bytes) {
  EXPECT_EQ(bytes.size() % sizeof(BF16), 0U);
  std::vector<BF16> result(bytes.size() / sizeof(BF16));
  std::memcpy(result.data(), bytes.data(), bytes.size());
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
  const std::array input_descs{table_desc, ids_desc};
  const std::array output_descs{output_desc};
  const auto workload = d5_qwen35_fixture::workload(
      profile, c::WorkloadPhase::kDecode, region);

  auto context = c::cpu::create_context().value();
  auto prepared =
      c::prepare(context, c::Operation(c::Gather{}), input_descs, output_descs,
                 hyle::mem::system_resource(), workload);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cpu::kProviderId);
  EXPECT_EQ(prepared->workload(), workload);

  const auto physical = d5_gguf_test::values(rows, hidden, seed);
  const auto packed = d5_gguf_test::pack(
      type, physical, rows, hidden, table_desc.required_bytes().value());
  auto table_buffer = bytes_buffer(packed.bytes);
  auto ids_buffer = output_buffer(ids_desc);
  auto result_buffer = output_buffer(output_desc);
  const auto table =
      c::ConstTensorView::create(table_desc, table_buffer.view()).value();
  const auto ids =
      c::ConstTensorView::create(ids_desc, ids_buffer.view()).value();
  const auto result =
      c::TensorView::create(output_desc, result_buffer.view()).value();
  const std::array inputs{table, ids};
  const std::array outputs{result};
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 5}}, {{6, 2}}}};
  for (const auto &selection : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selection))).ok());
    auto done = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result_buffer);
    const auto expected =
        d5_gguf_test::selected_rows(packed.decoded, hidden, selection);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1.0e-6F) << i;
  }

  const auto before_failure = read_bytes(result_buffer);
  const std::array<std::int32_t, tokens> invalid{
      0, static_cast<std::int32_t>(rows)};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(result_buffer), before_failure);
}

TEST(CpuDefaultDirect,
     D5Qwen35PrimaryGgufQuantizedEmbeddingUsesNativeRowsAndReplays) {
  run_quantized_embedding(
      d5_qwen35_fixture::kPrimaryGgufProfile, d5_qwen35_fixture::kHidden,
      d5_qwen35_fixture::kEmbeddingRows, c::GgufQuantizedType::kQ6K, 0x010f,
      17);
}

TEST(CpuDefaultDirect,
     D5Qwen38PrimaryGgufQuantizedEmbeddingUsesExactWidthAndReplays) {
  run_quantized_embedding(
      d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
      d5_qwen38_official_fixture::kHidden,
      d5_qwen38_official_fixture::kEmbeddingRows,
      c::GgufQuantizedType::kQ4K, 0x010f, 19);
}

TEST(CpuDefaultDirect,
     D5Qwen35FidelityGgufF16EmbeddingUsesMatureRowsAndReplays) {
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto rows = d5_qwen35_fixture::kEmbeddingRows;
  constexpr std::int64_t tokens = 2;
  const auto table_desc = dense(c::DataType::kFloat16, {rows, hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto output_desc = dense(c::DataType::kFloat16, {tokens, hidden});
  const std::array input_descs{table_desc, ids_desc};
  const std::array output_descs{output_desc};
  const auto workload =
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x0108);
  auto context = c::cpu::create_context().value();
  auto prepared =
      c::prepare(context, c::Operation(c::Gather{}), input_descs, output_descs,
                 hyle::mem::system_resource(), workload);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_TRUE(prepared->provider().id == c::ggml_cpu::kProviderId ||
              prepared->provider().id == c::onednn::kProviderId)
      << prepared->provider().name;
  EXPECT_EQ(prepared->workload(), workload);

  const auto source = d5_gguf_test::values(rows, hidden, 61);
  std::vector<F16> table_values(source.size());
  std::vector<float> decoded(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    table_values[i] = half_test::encode<F16>(source[i]);
    decoded[i] = static_cast<float>(half_test::decode(table_values[i]));
  }
  auto table_buffer = half_test::buffer<F16>(table_values);
  auto ids_buffer = output_buffer(ids_desc);
  auto result_buffer = output_buffer(output_desc);
  const std::array inputs{
      c::ConstTensorView::create(table_desc, table_buffer.view()).value(),
      c::ConstTensorView::create(ids_desc, ids_buffer.view()).value()};
  const std::array outputs{
      c::TensorView::create(output_desc, result_buffer.view()).value()};
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 5}}, {{6, 2}}}};
  for (const auto &selection : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selection))).ok());
    auto done = invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<F16>(result_buffer);
    const auto expected =
        d5_gguf_test::selected_rows(decoded, hidden, selection);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      half_test::expect_near(actual[i], expected[i]);
  }

  const auto before_failure = read_bytes(result_buffer);
  constexpr std::array<std::int32_t, tokens> invalid{0, rows};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue, inputs, outputs);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(result_buffer), before_failure);
}

TEST(CpuDefaultDirect,
     D5Qwen35OfficialEmbeddingNormHeadAndFailureUsePreparedCalls) {
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
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0x01020f);

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
  const std::array head_inputs{token_desc, head_desc};
  const std::array head_output{logits_desc};
  auto head = make_call(context, c::Operation(c::MatMul{}), head_inputs,
                        head_output, workload);
  for (const auto *call : {&gather, &norm, &residual, &head})
    expect_mature(*call, workload);

  auto table_buffer = bytes_buffer(fixture.rows);
  auto scale_buffer = bytes_buffer(fixture.scale);
  auto ids_buffer = output_buffer(ids_desc);
  auto embedded_buffer = output_buffer(token_desc);
  auto normalized_buffer = output_buffer(token_desc);
  auto residual_buffer = output_buffer(token_desc);
  auto logits_buffer = output_buffer(logits_desc);
  const auto table =
      c::ConstTensorView::create(table_desc, table_buffer.view()).value();
  const auto tied_head =
      c::ConstTensorView::create(head_desc, table_buffer.view()).value();
  const auto scale =
      c::ConstTensorView::create(scale_desc, scale_buffer.view()).value();
  const auto ids =
      c::ConstTensorView::create(ids_desc, ids_buffer.view()).value();
  const auto embedded =
      c::TensorView::create(token_desc, embedded_buffer.view()).value();
  const auto normalized =
      c::TensorView::create(token_desc, normalized_buffer.view()).value();
  const auto residual_view =
      c::TensorView::create(token_desc, residual_buffer.view()).value();
  const auto logits =
      c::TensorView::create(logits_desc, logits_buffer.view()).value();
  const std::array gather_views{table, ids};
  const std::array embedded_view{embedded};
  const std::array norm_views{
      c::ConstTensorView::create(token_desc, embedded_buffer.view()).value(),
      scale};
  const std::array normalized_view{normalized};
  const std::array add_views{
      c::ConstTensorView::create(token_desc, embedded_buffer.view()).value(),
      c::ConstTensorView::create(token_desc, normalized_buffer.view()).value()};
  const std::array residual_output{residual_view};
  const std::array head_views{
      c::ConstTensorView::create(token_desc, residual_buffer.view()).value(),
      tied_head};
  const std::array logits_output{logits};

  const auto table_values = decode_fixture(fixture.rows);
  const auto scale_values = decode_fixture(fixture.scale);
  const std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 1}}, {{5, 6}}}};
  const auto before = stats.snapshot();
  std::vector<std::byte> last_embedded;
  std::vector<std::byte> last_residual;
  std::vector<std::byte> last_logits;
  for (const auto &selected : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
    run(gather, queue, gather_views, embedded_view);
    run(norm, queue, norm_views, normalized_view);
    run(residual, queue, add_views, residual_output);
    run(head, queue, head_views, logits_output);
    const auto actual_residual = half_test::read<BF16>(residual_buffer);
    const auto actual_logits = half_test::read<BF16>(logits_buffer);
    for (std::size_t token = 0; token < tokens; ++token) {
      const auto source_row = static_cast<std::size_t>(selected[token]);
      std::vector<double> expected_row(hidden);
      double squares = 0;
      for (std::size_t column = 0; column < hidden; ++column) {
        const auto value =
            half_test::decode(table_values[source_row * hidden + column]);
        squares += value * value;
      }
      const auto inverse = 1.0 / std::sqrt(squares / hidden + 1.0e-6);
      for (std::size_t column = 0; column < hidden; ++column) {
        const auto value =
            half_test::decode(table_values[source_row * hidden + column]);
        const auto normalized_value = half_test::decode(half_test::encode<BF16>(
            value * inverse * half_test::decode(scale_values[column])));
        expected_row[column] = half_test::decode(
            half_test::encode<BF16>(value + normalized_value));
        half_test::expect_near(actual_residual[token * hidden + column],
                               expected_row[column]);
      }
      for (std::size_t row = 0; row < rows; ++row) {
        double expected = 0;
        for (std::size_t column = 0; column < hidden; ++column)
          expected += expected_row[column] *
                      half_test::decode(table_values[row * hidden + column]);
        half_test::expect_near(actual_logits[token * rows + row], expected);
      }
    }
    last_embedded = read_bytes(embedded_buffer);
    last_residual = read_bytes(residual_buffer);
    last_logits = read_bytes(logits_buffer);
  }
  EXPECT_EQ(stats.snapshot().total_allocs, before.total_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, before.reallocs);

  const std::array<std::int32_t, tokens> invalid{0, 7};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = gather.invocation.submit(queue, gather_views, embedded_view);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(embedded_buffer), last_embedded);
  EXPECT_EQ(read_bytes(residual_buffer), last_residual);
  EXPECT_EQ(read_bytes(logits_buffer), last_logits);
}

TEST(CpuDefaultDirect, D5Qwen35OfficialFfnUsesMaturePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::Ffn fixture;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_ffn(*fixture_root));

  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto ffn = d5_qwen35_fixture::kFfnHidden;
  const auto activation_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto stored_gate_desc = dense(c::DataType::kBFloat16, {ffn, hidden});
  const auto gate_weight_desc = stored_gate_desc.transposed().value();
  const auto stored_down_desc = dense(c::DataType::kBFloat16, {hidden, ffn});
  const auto down_weight_desc = stored_down_desc.transposed().value();
  const auto expanded_desc = dense(c::DataType::kBFloat16, {1, ffn});
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 2);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array norm_inputs{activation_desc, scale_desc};
  const std::array activation_output{activation_desc};
  auto norm = make_call(context, c::Operation(c::RmsNorm{}), norm_inputs,
                        activation_output, workload);
  const std::array gate_inputs{activation_desc, gate_weight_desc};
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
  const std::array down_inputs{expanded_desc, down_weight_desc};
  auto down = make_call(context, c::Operation(c::MatMul{}), down_inputs,
                        activation_output, workload);
  for (const auto *call : {&norm, &gate, &up, &gated, &down})
    expect_mature(*call, workload);

  auto activation_buffer = output_buffer(activation_desc);
  auto scale_buffer = bytes_buffer(fixture.norm_scale);
  auto gate_weight_buffer = bytes_buffer(fixture.weights[0]);
  auto up_weight_buffer = bytes_buffer(fixture.weights[1]);
  auto down_weight_buffer = bytes_buffer(fixture.weights[2]);
  auto normalized_buffer = output_buffer(activation_desc);
  auto gate_buffer = output_buffer(expanded_desc);
  auto up_buffer = output_buffer(expanded_desc);
  auto gated_buffer = output_buffer(expanded_desc);
  auto output = output_buffer(activation_desc);
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
  const std::array gate_output{
      c::TensorView::create(expanded_desc, gate_buffer.view()).value()};
  const std::array up_views{normalized, up_weight};
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
      c::TensorView::create(activation_desc, output.view()).value()};

  for (std::size_t replay = 0; replay < fixture.activations.size(); ++replay) {
    ASSERT_TRUE(
        activation_buffer.view().write(fixture.activations[replay]).ok());
    run(norm, queue, norm_views, normalized_output);
    run(gate, queue, gate_views, gate_output);
    run(up, queue, up_views, up_output);
    run(gated, queue, gated_views, gated_output);
    run(down, queue, down_views, final_output);
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

TEST(CpuDefaultDirect,
     D5Qwen35OfficialGqaNormRopeAndAttentionUsePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::AttentionNorms fixture;
  ASSERT_NO_THROW(fixture =
                      d5_qwen35_fixture::load_attention_norms(*fixture_root));
  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query_desc =
      dense(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv_desc = dense(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto scale_desc = dense(c::DataType::kBFloat16, {depth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc = dense(c::DataType::kFloat32, {depth / 2});
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 4);
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
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  const std::array query_rope_inputs{query_desc, position_desc, frequency_desc};
  auto query_rope = make_call(context, c::Operation(rope), query_rope_inputs,
                              query_output, workload);
  const std::array key_rope_inputs{kv_desc, position_desc, frequency_desc};
  auto key_rope = make_call(context, c::Operation(rope), key_rope_inputs,
                            key_output, workload);
  const std::array attention_inputs{query_desc, kv_desc, kv_desc};
  auto attention =
      make_call(context,
                c::Operation(c::ScaledDotProductAttention{
                    .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                    .scale = 0.0625F,
                    .causal = true,
                    .accumulation_type = c::DataType::kFloat32}),
                attention_inputs, query_output, workload);
  for (const auto *call :
       {&query_norm, &key_norm, &query_rope, &key_rope, &attention})
    expect_mature(*call, workload);

  std::vector<float> frequencies(depth / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / static_cast<float>(i + 1);
  const std::array<std::int32_t, 1> positions{3};
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
  auto normalized_query_buffer = output_buffer(query_desc);
  auto normalized_key_buffer = output_buffer(kv_desc);
  auto rotated_query_buffer = output_buffer(query_desc);
  auto rotated_key_buffer = output_buffer(kv_desc);
  auto result_buffer = output_buffer(query_desc);
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
  const std::array normalized_query_output{
      c::TensorView::create(query_desc, normalized_query_buffer.view())
          .value()};
  const std::array key_norm_views{key, key_scale};
  const std::array normalized_key_output{
      c::TensorView::create(kv_desc, normalized_key_buffer.view()).value()};
  const std::array query_rope_views{
      c::ConstTensorView::create(query_desc, normalized_query_buffer.view())
          .value(),
      position, frequency};
  const std::array rotated_query_output{
      c::TensorView::create(query_desc, rotated_query_buffer.view()).value()};
  const std::array key_rope_views{
      c::ConstTensorView::create(kv_desc, normalized_key_buffer.view()).value(),
      position, frequency};
  const std::array rotated_key_output{
      c::TensorView::create(kv_desc, rotated_key_buffer.view()).value()};
  const std::array attention_views{
      c::ConstTensorView::create(query_desc, rotated_query_buffer.view())
          .value(),
      c::ConstTensorView::create(kv_desc, rotated_key_buffer.view()).value(),
      value};
  const std::array attention_output{
      c::TensorView::create(query_desc, result_buffer.view()).value()};
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
    ASSERT_TRUE(
        query_buffer.view().write(std::as_bytes(std::span(queries))).ok());
    ASSERT_TRUE(key_buffer.view().write(std::as_bytes(std::span(keys))).ok());
    ASSERT_TRUE(
        value_buffer.view().write(std::as_bytes(std::span(values))).ok());
    run(query_norm, queue, query_norm_views, normalized_query_output);
    run(key_norm, queue, key_norm_views, normalized_key_output);
    run(query_rope, queue, query_rope_views, rotated_query_output);
    run(key_rope, queue, key_rope_views, rotated_key_output);
    run(attention, queue, attention_views, attention_output);
    const auto actual = half_test::read<BF16>(result_buffer);
    for (std::size_t head = 0; head < query_heads; ++head)
      for (std::size_t column = 0; column < depth; ++column)
        EXPECT_EQ(
            actual[head * depth + column].bits,
            values[(head / (query_heads / kv_heads)) * depth + column].bits);
  }
}

TEST(CpuDefaultDirect, D5Qwen35OfficialCausalConvAndGdnUsePreparedCalls) {
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
    const auto softplus = bias > 0 ? bias + std::log1p(std::exp(-bias))
                                   : std::log1p(std::exp(bias));
    log_decay[head] = static_cast<float>(
        -std::exp(static_cast<double>(fixture.log_decay[head])) * softplus);
    beta[head] = static_cast<float>(1.0 / (1.0 + std::exp(-bias)));
  }
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 5);
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array conv_inputs{conv_samples_desc, conv_weight_desc};
  const std::array conv_output{conv_output_desc};
  auto convolution = make_call(context, c::model_ops::causal_conv1d(),
                               conv_inputs, conv_output, workload);
  const std::array gdn_inputs{qkv_desc,  qkv_desc,  qkv_desc,
                              gate_desc, gate_desc, state_desc};
  const std::array gdn_outputs{qkv_desc, state_desc};
  auto gdn = make_call(context, c::Operation(c::GatedDeltaNet{}), gdn_inputs,
                       gdn_outputs, workload);
  expect_mature(convolution, workload);
  expect_mature(gdn, workload);

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
  for (int seed : {2, 9}) {
    std::vector<float> conv_samples_values(
        conv_samples_desc.element_count().value());
    std::array<std::vector<float>, 3> qkv_values;
    for (auto &values : qkv_values)
      values.resize(qkv_desc.element_count().value());
    std::vector<float> state_values(state_desc.element_count().value());
    for (std::size_t i = 0; i < conv_samples_values.size(); ++i)
      conv_samples_values[i] =
          static_cast<float>(static_cast<int>((i * 17 + seed * 5) % 61) - 30) /
          128.0F;
    for (std::size_t port = 0; port < qkv_values.size(); ++port)
      for (std::size_t i = 0; i < qkv_values[port].size(); ++i)
        qkv_values[port][i] =
            static_cast<float>(
                std::sin(static_cast<double>(i + port * 37 + seed) * 0.017)) *
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
    ASSERT_TRUE(
        state_buffer.view().write(std::as_bytes(std::span(state_values))).ok());
    run(convolution, queue, conv_views, conv_result);
    run(gdn, queue, gdn_views, gdn_result);
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
        const auto delta = static_cast<double>(beta[head]) *
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

TEST(CpuDefaultDirect, D5Qwen35OfficialMtpAndBfloatBoundaryUsePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  d5_qwen35_fixture::MtpProjection fixture;
  d5_qwen35_fixture::Embedding embedding;
  ASSERT_NO_THROW(fixture = d5_qwen35_fixture::load_mtp(*fixture_root));
  ASSERT_NO_THROW(embedding = d5_qwen35_fixture::load_embedding(*fixture_root));
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto input = d5_qwen35_fixture::kMtpInput;
  const auto half_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto joined_desc = dense(c::DataType::kBFloat16, {1, input});
  const auto stored_weight_desc =
      dense(c::DataType::kBFloat16, {hidden, input});
  const auto weight_desc = stored_weight_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto output_f32_desc = dense(c::DataType::kFloat32, {1, hidden});
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0x0d0e);
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

  auto weight_buffer = bytes_buffer(fixture.weight);
  auto scale_buffer = bytes_buffer(embedding.scale);
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
  const auto scales = decode_fixture(embedding.scale);
  for (std::size_t replay = 0; replay < fixture.activations.size(); ++replay) {
    const auto &activation = fixture.activations[replay];
    const auto split = static_cast<std::size_t>(hidden * sizeof(BF16));
    ASSERT_TRUE(first_buffer.view()
                    .write(std::span<const std::byte>(activation.data(), split))
                    .ok());
    ASSERT_TRUE(second_buffer.view()
                    .write(std::span<const std::byte>(
                        activation.data() + split, activation.size() - split))
                    .ok());
    run(concat, queue, concat_views, joined_view);
    run(projection, queue, projection_views, projected_view);
    run(norm, queue, norm_views, normalized_view);
    run(cast, queue, cast_views, converted_view);
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
      const auto expected = half_test::decode(
          half_test::encode<BF16>(half_test::decode(projected_values[i]) *
                                  inverse * half_test::decode(scales[i])));
      EXPECT_NEAR(converted_values[i], expected,
                  std::abs(expected) * 0.008 + 1.0e-6);
    }
  }
}
} // namespace
