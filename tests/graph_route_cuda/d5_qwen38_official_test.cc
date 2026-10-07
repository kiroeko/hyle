#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "../compute_models/d5_qwen38_24t_fixture.h"
#include "../compute_models/d5_qwen38_24t_grouped_gguf_fixture.h"
#include "../compute_models/d5_deepseek_remaining_fragments.h"
#include "../compute_models/d5_moe_execution_fragments.h"
#include "../compute_models/d5_official_fp8_fragments.h"
#include "../compute_models/d5_dynamic_fp8_fragments.h"
#include "../compute_models/d5_fp8_weight_preparation.h"
#include "../compute_models/d5_quantization_roundtrip.h"
#include "../compute_models/d5_deepseek_official_fragments.h"
#include "../compute_models/d5_deepseek_official_attention.h"
#include "../compute_models/d5_grouped_fp8_fragments.h"
#include "../compute_models/d5_kda_full_fragments.h"
#include "../compute_models/d5_kimi_remaining_fragments.h"
#include "../compute_models/d5_glm_remaining_fragments.h"
#include "../compute_models/d5_moe_semantic_fragments.h"
#include "../compute_models/d5_qwen38_24t_semantic_fragments.h"
#include "../compute_models/d5_qwen38_24t_full_router.h"
#include "../compute_models/d5_qwen38_24t_moe_profiles.h"
#include "../compute_models/d5_qwen38_official_fixture.h"
#include "../compute_models/d5_qwen38_profile_cases.h"
#include "../compute_models/d5_qwen_vocabulary_fixture.h"
#include "../compute_models/d5_qwen_vocabulary_graph_test.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "../compute_models/d5_optional_cuda_provider.h"
#include "../compute_models/cuda_iq1_model_contracts.h"
#include "../compute_models/cpu_c4_model_branches.h"
#include "../compute_models/cpu_c5_non_gguf_fixture.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace cuda = hyle::compute::cuda;
namespace g = hyle::graph;
namespace mem = hyle::mem;
namespace q = d5_qwen38_official_fixture;
namespace q24 = d5_qwen38_24t_fixture;
namespace qfull = d5_qwen38_24t_full_router;
namespace qcase = d5_qwen38_profile_cases;
namespace vocab = d5_qwen_vocabulary_fixture;
using BF16 = hyle::simd::bfloat16_t;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

hyle::mem::Buffer output_buffer(mem::MemoryResourceRef resource,
                                const c::TensorDesc &description) {
  return hyle::mem::Buffer::create(resource,
                                   description.required_bytes().value(), 256)
      .value();
}

template <class T>
hyle::mem::Buffer values_buffer(mem::MemoryResourceRef resource,
                                std::span<const T> values) {
  auto result = hyle::mem::Buffer::create(
                    resource, std::as_bytes(values).size(), 256)
                    .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());
  return result;
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
    if (node.kind == c::OperationKind::kView) {
      EXPECT_TRUE(report.provider.name.empty());
      continue;
    }
    // D4 retained the exact Copy/Bool Select storage boundaries. A residual
    // step cannot conceal an arithmetic node or a native multi-node region.
    const bool retained_storage = provider == c::cuda_native::kProviderId &&
        report.node_count == 1 &&
        (node.kind == c::OperationKind::kCopy || node.kind == c::OperationKind::kSelect);
    EXPECT_TRUE(provider == c::ggml_cuda::kProviderId ||
                provider == c::cublas::kProviderId ||
                provider == c::cuda_native::kSglangProviderId ||
              provider == c::cuda_native::kDeepSeekProviderId ||
                provider == c::cuda_native::kCcclProviderId ||
#if defined(HYLE_COMPUTE_TORCH_CUDA)
              provider == c::torch_cuda::kProviderId ||
#endif
                d5_optional_cuda_provider::is_flashinfer(provider) || retained_storage)
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

class CudaGraphQwen38Profiles
    : public ::testing::TestWithParam<const qcase::ProfileCase *> {};

TEST(CudaGraphQwen38_24T, D5ExactWidthCausalConvGraphReplay) {
  const d5_qwen38_gdn_fixture::ConvFixture fixture(
      q24::kGdnKeyHeads, q24::kGdnValueHeads);
  const auto workload = q24::workload(c::WorkloadPhase::kDecode, 5);
  auto builder = g::GraphBuilder::create().value();
  const auto samples_id = builder.input(fixture.samples_desc).value();
  const auto weights_id = builder.input(fixture.weights_desc).value();
  const auto output_id =
      builder.add(c::model_ops::causal_conv1d(), samples_id, weights_id)
          .value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  auto logical = std::move(builder).finish().value();
  auto context = g::cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice)
                    .value();
  const auto resource = memory.ref();
  auto compiled =
      g::cuda::compile(logical, context, resource, {.workload = workload})
          .value();
  expect_mature(compiled);
  EXPECT_EQ(compiled.executable.workload(), workload);
  auto samples = output_buffer(resource, fixture.samples_desc);
  auto weights = values_buffer<float>(resource, fixture.weights);
  auto output = output_buffer(resource, fixture.output_desc);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(samples_id,
                        c::ConstTensorView::create(fixture.samples_desc,
                                                   samples.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(weights_id,
                        c::ConstTensorView::create(fixture.weights_desc,
                                                   weights.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(output_id,
                        c::TensorView::create(fixture.output_desc,
                                              output.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
  for (const int seed : {3, 17}) {
    const auto input = fixture.samples(seed);
    ASSERT_TRUE(samples.view().write(std::as_bytes(std::span(input))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(output);
    const auto expected = fixture.oracle(input);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1.0e-6) << i;
  }
}

TEST_P(CudaGraphQwen38Profiles,
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
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      g::cuda::create_context({.metadata_allocator = stats.get()}).value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto table_buffer = values_buffer<BF16>(resource, table_values);
  auto head_buffer = values_buffer<BF16>(resource, head_values);
  auto scale_buffer = values_buffer<BF16>(resource, scale_values);
  auto projection_buffer = values_buffer<BF16>(resource, projection_values);
  projection_values.clear();
  projection_values.shrink_to_fit();
  auto ids_buffer = output_buffer(resource, ids_desc);
  auto residual_buffer = output_buffer(resource, token_desc);
  auto projected_buffer = output_buffer(resource, token_desc);
  auto logits_buffer = output_buffer(resource, logits_desc);

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
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x01030e0f);
  auto result = g::cuda::compile(
      logical, context, resource,
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
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());

  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 4}}, {{6, 2}}}};
  std::uint64_t warmed_allocs = 0;
  std::uint64_t warmed_reallocs = 0;
  std::size_t replay = 0;
  std::vector<std::byte> last_residual;
  std::vector<std::byte> last_projected;
  std::vector<std::byte> last_logits;
  for (const auto &selected : selections) {
    ASSERT_TRUE(
        ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait();
    ASSERT_TRUE(status.ok()) << status.message();
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
    if (replay++ == 0) {
      const auto warmed = stats.snapshot();
      warmed_allocs = warmed.total_allocs;
      warmed_reallocs = warmed.reallocs;
    }
  }
  EXPECT_EQ(stats.snapshot().total_allocs, warmed_allocs);
  EXPECT_EQ(stats.snapshot().reallocs, warmed_reallocs);

  constexpr std::array<std::int32_t, tokens> invalid{0, q::kEmbeddingRows};
  ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = invocation.submit(queue);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(residual_buffer), last_residual);
  EXPECT_EQ(read_bytes(projected_buffer), last_projected);
  EXPECT_EQ(read_bytes(logits_buffer), last_logits);
}

TEST_P(CudaGraphQwen38Profiles,
       D5Qwen38ProfileFfnUsesExactInnerWidthAndReplays) {
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
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto scale_buffer = values_buffer<BF16>(resource, scale_values);
  auto gate_buffer = values_buffer<BF16>(resource, gate_values);
  gate_values.clear();
  gate_values.shrink_to_fit();
  auto up_buffer = values_buffer<BF16>(resource, up_values);
  up_values.clear();
  up_values.shrink_to_fit();
  auto down_buffer = values_buffer<BF16>(resource, down_values);
  down_values.clear();
  down_values.shrink_to_fit();
  auto activation_buffer = output_buffer(resource, activation_desc);
  auto output = output_buffer(resource, activation_desc);

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
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x0203);
  auto result = g::cuda::compile(
      logical, context, resource,
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
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
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

TEST(CudaGraphQwen38_24T, D5GgufGroupedExpertKeepsNativeBlocksAndDynamicIds) {
  namespace gguf = d5_qwen38_24t_grouped_gguf_fixture;
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const auto activation_desc = dense(c::DataType::kFloat32,
                                     {1, gguf::kSlots, gguf::kReduction});
  const auto ids_desc = dense(c::DataType::kInt32, {1, gguf::kSlots});
  const auto output_desc = dense(c::DataType::kFloat32,
                                 {1, gguf::kSlots, gguf::kColumns});
  for (const auto &test : gguf::kCases) {
    SCOPED_TRACE(test.profile->variant);
    const auto weight_desc = c::gguf_grouped_weight_description(
        test.type, c::DataType::kFloat32, gguf::kResidentExperts,
        gguf::kColumns, gguf::kReduction).value();
    auto packed = gguf::pack_weight(test, weight_desc.required_bytes().value());
    auto weight_buffer = output_buffer(resource, weight_desc);
    ASSERT_TRUE(weight_buffer.view().write(packed.bytes).ok());
    auto activation_buffer = output_buffer(resource, activation_desc);
    auto ids_buffer = output_buffer(resource, ids_desc);
    auto result_buffer = output_buffer(resource, output_desc);
    auto builder = g::GraphBuilder::create().value();
    const auto weight = builder.constant(weight_desc).value();
    const auto activation = builder.input(activation_desc).value();
    const auto ids = builder.input(ids_desc).value();
    const std::array ports{weight, activation, ids};
    const auto node =
        builder.add_node(c::model_ops::grouped_matmul(), ports).value();
    const auto result = builder.result(node).value();
    ASSERT_TRUE(builder.publish(result).ok());
    auto logical = std::move(builder).finish().value();
    const auto workload = q::workload(*test.profile,
                                     c::WorkloadPhase::kDecode, 0x0b21);
    const std::array constants{g::ConstantBinding{
        weight,
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()}};
    auto compiled_result = g::cuda::compile(logical, context, resource,
                                            {.constants = constants,
                                             .workload = workload});
    ASSERT_TRUE(compiled_result) << compiled_result.error().message();
    auto compiled = std::move(*compiled_result);
    expect_mature(compiled);
    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(
        activation, c::ConstTensorView::create(
                        activation_desc, activation_buffer.view()).value()).ok());
    ASSERT_TRUE(invocation.bind(
        ids, c::ConstTensorView::create(ids_desc, ids_buffer.view()).value())
                    .ok());
    ASSERT_TRUE(invocation.bind(
        result, c::TensorView::create(output_desc, result_buffer.view()).value())
                    .ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
    for (int replay = 0; replay < 2; ++replay) {
      const auto values = q24::expert_input(replay);
      const auto selected = gguf::ids(replay);
      ASSERT_TRUE(activation_buffer.view()
                      .write(std::as_bytes(std::span(values))).ok());
      ASSERT_TRUE(ids_buffer.view()
                      .write(std::as_bytes(std::span(selected))).ok());
      auto completion = invocation.submit(queue);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<float>(result_buffer);
      const auto expected = gguf::expected(values, selected, packed.decoded);
      d5_gguf_test::expect_quantized_matmul_close(actual, expected);
    }

    const auto retained = half_test::read<float>(result_buffer);
    for (std::int32_t bad : {-1, static_cast<std::int32_t>(gguf::kResidentExperts)}) {
      auto selected = gguf::ids(1);
      selected.back() = bad;
      ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
      auto attempt = compiled.executable.create_invocation().value();
      ASSERT_TRUE(attempt.bind(activation,
          c::ConstTensorView::create(activation_desc, activation_buffer.view()).value()).ok());
      ASSERT_TRUE(attempt.bind(ids,
          c::ConstTensorView::create(ids_desc, ids_buffer.view()).value()).ok());
      ASSERT_TRUE(attempt.bind(result,
          c::TensorView::create(output_desc, result_buffer.view()).value()).ok());
      ASSERT_TRUE(g::cuda::prepare_replay(attempt, queue).ok());
      auto failed = attempt.submit(queue);
      ASSERT_TRUE(failed) << failed.error().message();
      EXPECT_TRUE(failed->wait().IsOutOfRange());
      EXPECT_EQ(half_test::read<float>(result_buffer), retained);
    }
    const auto valid = gguf::ids(1);
    ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(valid))).ok());
    auto recovered = invocation.submit(queue);
    ASSERT_TRUE(recovered) << recovered.error().message();
    EXPECT_TRUE(recovered->wait().ok());
    EXPECT_EQ(half_test::read<float>(result_buffer), retained);
  }
}

TEST(CudaGraphQwen38_24T,
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
  const auto ids_desc =
      dense(c::DataType::kInt32, {1, q24::kTopK});
  const auto output_desc = activation_desc;
  auto gate_weight_values = q24::expert_weight(
      q24::kExpertHidden, q24::kHidden, seed);
  auto up_weight_values = q24::expert_weight(
      q24::kExpertHidden, q24::kHidden, seed + 1);
  auto down_weight_values = q24::expert_weight(
      q24::kHidden, q24::kExpertHidden, seed + 2);
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto gate_weight_buffer =
      values_buffer<BF16>(resource, gate_weight_values);
  gate_weight_values.clear();
  gate_weight_values.shrink_to_fit();
  auto up_weight_buffer = values_buffer<BF16>(resource, up_weight_values);
  up_weight_values.clear();
  up_weight_values.shrink_to_fit();
  auto down_weight_buffer =
      values_buffer<BF16>(resource, down_weight_values);
  down_weight_values.clear();
  down_weight_values.shrink_to_fit();
  const auto ids_data = q24::active_expert_ids();
  auto ids_buffer = values_buffer<std::int32_t>(resource, ids_data);
  auto activation_buffer = output_buffer(resource, activation_desc);
  auto result_buffer = output_buffer(resource, output_desc);

  auto builder = g::GraphBuilder::create().value();
  const auto gate_weight = builder.constant(weight_desc).value();
  const auto up_weight = builder.constant(weight_desc).value();
  const auto down_weight = builder.constant(down_weight_desc).value();
  const auto activation = builder.input(activation_desc).value();
  const auto ids = builder.input(ids_desc).value();
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
  ASSERT_TRUE(builder.publish(final).ok());
  auto logical = std::move(builder).finish().value();
  const std::array constants{
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
  auto result = g::cuda::compile(
      logical, context, resource,
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
                  .bind(ids, c::ConstTensorView::create(ids_desc,
                                                       ids_buffer.view())
                                 .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(final,
                        c::TensorView::create(output_desc, result_buffer.view())
                            .value())
                  .ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
  for (int replay : {3, 17}) {
    const auto input = q24::expert_input(replay);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(input)))
                    .ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result_buffer);
    const auto expected = q24::expert_oracle(input, seed);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i],
                  std::abs(expected[i]) * q24::kExpertRelativeTolerance +
                      q24::kExpertAbsoluteTolerance)
          << i;
  }
}

TEST(CudaGraphQwen38_24T,
     D5OfficialBfloatRouterPreservesF32NearTieScores) {
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, q24::kHidden});
  const auto weight_desc =
      dense(c::DataType::kBFloat16, {q24::kExperts, q24::kHidden});
  const auto output_desc =
      dense(c::DataType::kFloat32, {1, q24::kExperts});
  std::vector<BF16> weights(q24::kExperts * q24::kHidden);
  weights[9 * q24::kHidden + 20] = q::encode(1.0);
  weights[10 * q24::kHidden + 20] = q::encode(1.0);
  weights[10 * q24::kHidden + 21] = q::encode(1.0);
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto weight_buffer = values_buffer<BF16>(resource, weights);
  auto activation_buffer = output_buffer(resource, activation_desc);
  auto result_buffer = output_buffer(resource, output_desc);
  auto builder = g::GraphBuilder::create().value();
  const auto activation = builder.input(activation_desc).value();
  const auto weight = builder.constant(weight_desc).value();
  const std::array ports{activation, weight};
  const auto node =
      builder.add_node(c::model_ops::bfloat_linear_f32(), ports).value();
  const auto scores = builder.result(node).value();
  ASSERT_TRUE(builder.publish(scores).ok());
  auto logical = std::move(builder).finish().value();
  const std::array constants{g::ConstantBinding{
      weight, c::ConstTensorView::create(weight_desc, weight_buffer.view())
                  .value()}};
  const auto workload = q24::workload(c::WorkloadPhase::kDecode, 0x0a10);
  auto result = g::cuda::compile(
      logical, context, resource,
      {.constants = constants, .workload = workload});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation
                  .bind(activation,
                        c::ConstTensorView::create(activation_desc,
                                                   activation_buffer.view())
                            .value())
                  .ok());
  ASSERT_TRUE(invocation
                  .bind(scores, c::TensorView::create(output_desc,
                                                      result_buffer.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
  for (const double fraction : {0.001953125, 0.0029296875}) {
    std::vector<BF16> input(q24::kHidden);
    input[20] = q::encode(1.0);
    input[21] = q::encode(fraction);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(input))).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result_buffer);
    ASSERT_EQ(actual.size(), q24::kExperts);
    EXPECT_NEAR(actual[9], 1.0, 1.0e-6);
    EXPECT_NEAR(actual[10], 1.0 + fraction, 1.0e-6);
  }
}

void prepare_model_constants(const d5_qwen38_24t_semantic_fragments::Fragment &cold,
    c::Context &context, mem::MemoryResourceRef resource,
    std::vector<mem::Buffer> &result) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  ASSERT_TRUE(cold.inputs.empty());
  ASSERT_FALSE(cold.preparation);
  auto graph = sem::make_graph(cold);
  std::vector<mem::Buffer> sources;
  std::vector<g::ConstantBinding> bindings;
  for (std::size_t i = 0; i < cold.constants.size(); ++i) {
    sources.push_back(values_buffer<std::byte>(resource, cold.constant_bytes[i]));
    bindings.push_back({graph.constants[i], c::ConstTensorView::create(
        cold.description(cold.constants[i]), sources.back().view()).value()});
  }
  const auto workload = q::workload(*cold.profile, c::WorkloadPhase::kDecode, cold.region);
  auto compiled_result = g::cuda::compile(graph.graph, context, resource,
      {.constants = bindings, .workload = workload});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  for (std::size_t i = 0; i < cold.outputs.size(); ++i) {
    result.push_back(output_buffer(resource, cold.description(cold.outputs[i])));
    ASSERT_TRUE(invocation.bind(graph.outputs[i], c::TensorView::create(
        cold.description(cold.outputs[i]), result.back().view()).value()).ok());
  }
  auto queue = context.create_queue().value();
  auto done = invocation.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  d5_fp8_weight_preparation::check_constants(cold, result);
  // The cold Graph and all source/intermediate storage are destroyed before
  // compiling the hot Graph. Only its independently owned outputs remain.
}

void run_q24_semantic_fragment(
    d5_qwen38_24t_semantic_fragments::Fragment fixture) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  auto full = sem::make_graph(fixture);
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  std::vector<hyle::mem::Buffer> constants(fixture.constants.size());
  std::vector<mem::Buffer> prepared_constants;
  if (fixture.preparation) {
    prepare_model_constants(*fixture.preparation, context, resource, prepared_constants);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(prepared_constants.size(), fixture.prepared_constants.size());
    for (const auto &step : fixture.nodes)
      EXPECT_FALSE(step.operation.custom_as<c::model_ops::Float8WeightDecode>());
  }
  for (std::size_t index = 0; index < constants.size(); ++index) {
    const auto found = std::ranges::find(fixture.prepared_constants, index);
    if (found != fixture.prepared_constants.end()) {
      ASSERT_TRUE(fixture.constant_bytes[index].empty());
      constants[index] = std::move(prepared_constants[found - fixture.prepared_constants.begin()]);
    } else constants[index] = values_buffer<std::byte>(resource, fixture.constant_bytes[index]);
  }
  fixture.constant_bytes = {};
  std::vector<g::ConstantBinding> bindings;
  for (std::size_t i = 0; i < constants.size(); ++i)
    bindings.push_back(
        {full.constants[i],
         c::ConstTensorView::create(fixture.description(fixture.constants[i]),
                                    constants[i].view())
             .value()});
  const auto workload =
      q::workload(*fixture.profile, c::WorkloadPhase::kDecode, fixture.region);
  auto compiled_result =
      g::cuda::compile(full.graph, context, resource,
                       {.constants = bindings, .workload = workload});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  std::vector<hyle::mem::Buffer> inputs, outputs;
  for (auto id : fixture.inputs)
    inputs.push_back(output_buffer(resource, fixture.description(id)));
  for (auto id : fixture.outputs)
    outputs.push_back(output_buffer(resource, fixture.description(id)));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    ASSERT_TRUE(
        invocation
            .bind(full.inputs[i],
                  c::ConstTensorView::create(
                      fixture.description(fixture.inputs[i]), inputs[i].view())
                      .value())
            .ok());
  for (std::size_t i = 0; i < outputs.size(); ++i)
    ASSERT_TRUE(
        invocation
            .bind(full.outputs[i],
                  c::TensorView::create(fixture.description(fixture.outputs[i]),
                                        outputs[i].view())
                      .value())
            .ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
  sem::StateFeedback feedback{fixture};
  std::size_t replay_index = 0;
  for (const auto &expected : fixture.replays) {
    SCOPED_TRACE(fixture.region);
    SCOPED_TRACE("replay " + std::to_string(replay_index++));
    for (std::size_t i = 0; i < inputs.size(); ++i)
      ASSERT_TRUE(inputs[i].view().write(feedback.input(expected, i)).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    for (std::size_t output = 0; output < outputs.size(); ++output) {
      SCOPED_TRACE("output " + std::to_string(output));
      const auto actual = sem::decode_output(fixture, output, half_test::read<float>(outputs[output]));
      ASSERT_EQ(actual.size(), expected.outputs[output].size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::isfinite(expected.outputs[output][i]))
          ASSERT_TRUE(std::isfinite(actual[i])) << "non-finite output at " << i;
      feedback.observe(output, actual);
      sem::expect_output(fixture, actual, expected.outputs[output], {},
                         fixture.exact_output(output), fixture.id_validation(output),
                         fixture.bfloat_chain_output(output));
    }
  }
}

TEST(CudaGraphMoeFragments, D5OriginalRouterNativeBoundaries) {
  for (auto experts : {256, 384})
    for (auto tokens : {1, 513})
      for (int mode = 0; mode < 6; ++mode) {
        SCOPED_TRACE(experts);
        SCOPED_TRACE(tokens);
        SCOPED_TRACE(mode);
        run_q24_semantic_fragment(d5_moe_execution_fragments::native_router_boundary(
            experts, tokens, mode));
      }
}

TEST(CudaGraphMoeFragments, D5MatureClampBoundaries) {
  run_q24_semantic_fragment(d5_moe_execution_fragments::clamp_boundaries());
}

TEST(CudaGraphMoeFragments, D5DeepSeekWeightedHyperConnectionAndHead) {
  for (std::size_t index : {3u, 4u, 5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    d5_deepseek_remaining_fragments::check_hyper_head(model, run_q24_semantic_fragment);
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekFullAttentionChain) {
  for (std::size_t index : {5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    for (std::int64_t ratio : {0, 4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_remaining_fragments::attention(model, ratio));
    }
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekStatefulCompression) {
  for (std::size_t index : {3u, 4u, 5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    for (std::int64_t ratio : {4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_remaining_fragments::compressor(model, ratio));
    }
    run_q24_semantic_fragment(d5_deepseek_remaining_fragments::compressor(model, 4, true));
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekNativeSinkAttention) {
  for (std::size_t index : {3u, 4u, 5u, 6u})
    for (int mode = 0; mode < 3; ++mode) {
      const auto &model = d5_moe_execution_cases::kCases[index];
      SCOPED_TRACE(model.profile->model_repository);
      SCOPED_TRACE(model.profile->variant);
      SCOPED_TRACE(mode);
      run_q24_semantic_fragment(d5_deepseek_remaining_fragments::sink_attention(model, mode));
    }
}

TEST(CudaGraphMoeFragments, D5DeepSeekTokenTableRouting) {
  for (std::size_t index : {3u, 4u, 5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::hash_router(model, true));
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekHashRoutingFeedsMixedWeightMoe) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (std::size_t index : {5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::hash_moe(model, true));
  }
}

TEST(CudaGraphMoeFragments, D5NativeHashRouterWarpTailsAndScoreBoundaries) {
  for (std::size_t index : {3u, 4u})
    for (auto rows : {1, 4, 5, 513})
      for (int mode = 1; mode < 5; ++mode) {
        SCOPED_TRACE(index);
        SCOPED_TRACE(rows);
        SCOPED_TRACE(mode);
        run_q24_semantic_fragment(d5_moe_execution_fragments::hash_router(
            d5_moe_execution_cases::kCases[index], true, rows, mode));
      }
}

TEST(CudaGraphMoeFragments, D5ThirteenProfilesModelRouters) {
  for (const auto &model : d5_moe_execution_cases::kCases) {
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::router(model, true));
  }
}

TEST(CudaGraphMoeFragments, D5ThirteenProfilesExpertActivation) {
  for (const auto &model : d5_moe_execution_cases::kCases) {
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::activation_core(model));
  }
}

TEST(CudaGraphMoeFragments, D5GroupedFp8Batch) {
  d5_grouped_fp8_fragments::projection_batch([](auto fragment) {
    run_q24_semantic_fragment(std::move(fragment));
  });
}

TEST(CudaGraphMoeFragments, D5GlmOfficialFp8FullMoe) {
  for (bool flash : {false, true}) {
    SCOPED_TRACE(flash);
    run_q24_semantic_fragment(d5_grouped_fp8_fragments::glm_moe(flash));
  }
}

TEST(CudaGraphMoeFragments, D5DynamicFp8DenseBatch) {
  namespace fp8 = d5_official_fp8_fragments;
  namespace dynamic = d5_dynamic_fp8_fragments;
  const auto check = [](const fp8::Case &shape) {
    SCOPED_TRACE(shape.profile->model_name);
    SCOPED_TRACE(shape.role);
    for (auto m : {1, 3}) for (bool bf16 : {false, true}) {
      SCOPED_TRACE(m);
      SCOPED_TRACE(bf16);
      run_q24_semantic_fragment(dynamic::projection(shape, m, bf16, false));
      if (bf16) run_q24_semantic_fragment(dynamic::projection(shape, m, true, true));
    }
  };
  for (const auto &shape : fp8::kCases) check(shape);
  check(fp8::kGlmTailCase);
}

TEST(CudaGraphMoeFragments, D5GlmOfficialDynamicFp8SharedFfn) {
  for (bool flash : {false, true}) {
    SCOPED_TRACE(flash);
    for (auto m : {1, 3}) {
      SCOPED_TRACE(m);
      run_q24_semantic_fragment(d5_dynamic_fp8_fragments::glm_shared_ffn(m, flash));
    }
  }
}

TEST(CudaGraphMoeFragments, D5OfficialFp8DenseNativeBatch) {
  namespace fp8 = d5_official_fp8_fragments;
  for (const auto &shape : fp8::kCases) {
    SCOPED_TRACE(shape.profile->model_name);
    SCOPED_TRACE(shape.role);
    for (auto m : {1, 3}) {
      SCOPED_TRACE(m);
      run_q24_semantic_fragment(fp8::projection(shape, m));
    }
  }
  for (auto m : {1, 3}) {
    SCOPED_TRACE("GLM partial N block");
    SCOPED_TRACE(m);
    run_q24_semantic_fragment(fp8::projection(fp8::kGlmTailCase, m));
  }
}

TEST(CudaGraphMoeFragments, D5OfficialFp4ExpertBatch) {
  ASSERT_TRUE(d5_official_fp4_storage::matches_original_converter());
  for (auto index : {0, 3, 4}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_name);
    for (bool down : {false, true}) {
      SCOPED_TRACE(down ? "down" : "gate/up");
      run_q24_semantic_fragment(
          d5_moe_execution_fragments::official_fp4_projection(model, down));
    }
  }
}

TEST(CudaGraphMoeFragments, D5OfficialKimiMixedStorageFullMoe) {
  run_q24_semantic_fragment(d5_moe_execution_fragments::official_kimi_moe());
}

TEST(CudaGraphMoeFragments, D5EightGgufProfilesFullMoe) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_moe_execution_cases::kCases) {
    if (!model.gguf)
      continue;
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::gguf_moe(model, true));
  }
}

TEST(CudaGraphMoeFragments, D5KdaProjectionRoleBatch) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &role : d5_kda_projection_cases::kCases) {
    SCOPED_TRACE(role.profile->model_repository);
    SCOPED_TRACE(role.profile->variant);
    SCOPED_TRACE(role.roles);
    run_q24_semantic_fragment(d5_moe_semantic_fragments::kda_projection(role));
  }
}

TEST(CudaGraphMoeFragments, D5GlmFlashWeightedHyperConnection) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (!model.flash) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::hyper_connection(model));
  }
}

TEST(CudaGraphMoeFragments, D5GlmDsaIndexerWithCache) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.flash || model.official) continue;
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::indexer(model));
    // The fidelity artifact has both F32 and Q8_0 indexer score projections.
    if (model.profile == &d5_moe_profile_cases::glm53_fidelity_gguf)
      run_q24_semantic_fragment(d5_glm_remaining_fragments::indexer(model, 0, true));
  }
}

TEST(CudaGraphMoeFragments, D5GlmFlashPooledIndexerWithTail) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (!model.flash) continue;
    SCOPED_TRACE(model.profile->variant);
    for (int tail = 0; tail < 4; ++tail) {
      SCOPED_TRACE(tail);
      run_q24_semantic_fragment(d5_glm_remaining_fragments::indexer(model, tail));
    }
  }
}

TEST(CudaGraphMoeFragments, D5GlmOfficialMixedRemaining) {
  namespace glm = d5_glm_remaining_fragments;
  run_q24_semantic_fragment(glm::official_flash_clamped_activation());
  for (const auto &model : glm::kCases) {
    if (!model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    run_q24_semantic_fragment(glm::dense_ffn(model));
    for (int tail : {0, 3}) {
      if (!model.flash && tail) continue;
      run_q24_semantic_fragment(glm::indexer(model, tail));
      run_q24_semantic_fragment(glm::sparse_attention(model, tail));
    }
    glm::check_index_carry(model, run_q24_semantic_fragment);
    if (model.flash) {
      run_q24_semantic_fragment(glm::mla(model));
      glm::check_mtp_head(model, run_q24_semantic_fragment);
    }
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekOfficialCompression) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    const auto &model = d5_moe_execution_fragments::cases::kCases[index];
    for (std::int64_t ratio : {4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::compressor(model, ratio));
    }
    run_q24_semantic_fragment(d5_deepseek_official_attention::compressor(model, 4, true));
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekOfficialAttention) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    const auto &model = d5_moe_execution_fragments::cases::kCases[index];
    for (std::int64_t ratio : {0, 4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::attention(model, ratio));
    }
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekOfficialMtpAndFullHead) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    d5_deepseek_official_attention::check_mtp_head(
        d5_moe_execution_fragments::cases::kCases[index], run_q24_semantic_fragment);
  }
}

TEST(CudaGraphMoeFragments, D5DeepSeekOriginalFp4Projections) {
  for (int index : {3, 4})
    for (bool down : {false, true})
      for (bool shared : {false, true})
        for (std::int64_t tokens : {1, 3}) {
          SCOPED_TRACE(index);
          SCOPED_TRACE(down);
          SCOPED_TRACE(shared);
          SCOPED_TRACE(tokens);
          run_q24_semantic_fragment(d5_deepseek_official_fragments::projection(
              d5_moe_execution_fragments::cases::kCases[index], down, shared, tokens));
        }
}

TEST(CudaGraphMoeFragments, D5DeepSeekOfficialMoe) {
  for (int index : {3, 4})
    for (bool hash : {false, true}) {
      SCOPED_TRACE(index);
      SCOPED_TRACE(hash);
      run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(
          d5_moe_execution_fragments::cases::kCases[index], hash));
    }
}

TEST(CudaGraphMoeFragments, D5DeepSeekExpertActivationRoundtrip) {
  for (std::int64_t width : {7168, 4096})
    for (std::int64_t rows : {1, 3, 33}) {
      SCOPED_TRACE(width);
      SCOPED_TRACE(rows);
      run_q24_semantic_fragment(d5_quantization_roundtrip::fragment(false, rows, width, 128));
    }
}

TEST(CudaGraphMoeFragments, D5DeepSeekQuantizationRoundtrip) {
  for (bool fp4 : {false, true})
    for (std::int64_t rows : {1, 3, 33}) {
      SCOPED_TRACE(fp4);
      SCOPED_TRACE(rows);
      run_q24_semantic_fragment(d5_quantization_roundtrip::fragment(fp4, rows, fp4 ? 128 : 448));
    }
}

TEST(CudaGraphMoeFragments, D5Fp8WeightPreparation) {
  for (bool bf16 : {false, true})
    for (const auto shape : {std::array<std::int64_t, 2>{129, 256}, {576, 6144}, {28672, 512}}) {
      SCOPED_TRACE(bf16);
      SCOPED_TRACE(shape[0]);
      run_q24_semantic_fragment(d5_fp8_weight_preparation::decode_weights(shape[0], shape[1], bf16));
    }
}

TEST(CudaGraphMoeFragments, D5GlmFp8AccumulationBoundaries) {
  run_q24_semantic_fragment(d5_dynamic_fp8_fragments::model_accumulation_boundaries());
}

TEST(CudaGraphMoeFragments, D5GlmDsaPreparedKvAndNextN) {
  namespace glm = d5_glm_remaining_fragments;
  const auto &model = glm::kCases[0];
  ASSERT_TRUE(model.official && !model.flash);
  run_q24_semantic_fragment(glm::mla(model));
  glm::check_mtp_head(model, run_q24_semantic_fragment);
}

TEST(CudaGraphMoeFragments, D5GlmGgufDenseFfn) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::dense_ffn(model));
  }
}

TEST(CudaGraphMoeFragments, D5GlmGgufNextNDecoderWithCarryAndFullHead) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    d5_glm_remaining_fragments::check_mtp_head(model, run_q24_semantic_fragment);
  }
}

TEST(CudaGraphMoeFragments, D5GlmGgufDsaSelectionConsumesMlaCache) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    for (int tail = 0; tail < (model.flash ? 4 : 1); ++tail) {
      SCOPED_TRACE(tail);
      run_q24_semantic_fragment(d5_glm_remaining_fragments::sparse_attention(model, tail));
    }
  }
}

TEST(CudaGraphMoeFragments, D5GlmLayerSpecificExpertFormats) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::mixed_expert_variants()) {
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.gate == d5_moe_execution_cases::Storage::kQ4K
                     ? "Q4_K/Q4_K/Q6_K" : "Q5_K/Q5_K/Q6_K");
    run_q24_semantic_fragment(d5_moe_execution_fragments::gguf_moe(model));
  }
}

TEST(CudaGraphMoeFragments, D5GlmGgufMtpIndexCarryReset) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    d5_glm_remaining_fragments::check_index_carry(model, run_q24_semantic_fragment);
  }
}

TEST(CudaGraphMoeFragments, D5GlmNextNInputBoundary) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::mtp_input(model));
  }
}

TEST(CudaGraphMoeFragments, D5GlmGgufAbsorbedMlaWithCache) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::mla(model));
  }
}

TEST(CudaGraphMoeFragments, D5KimiBfloatRoundingBoundaries) {
  run_q24_semantic_fragment(d5_kimi_remaining_fragments::bfloat_rounding_boundaries());
}

TEST(CudaGraphMoeFragments, D5KimiDenseFfn) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::dense_ffn(profile));
  }
}

TEST(CudaGraphMoeFragments, D5KimiAbsorbedMlaWithCache) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::mla(profile));
  }
}

TEST(CudaGraphMoeFragments, D5KimiAttentionResidualBanking) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::attention_residual(profile, true));
  }
}

TEST(CudaGraphMoeFragments, D5KimiAttentionResidualOrdinary) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::attention_residual(profile, false));
  }
}

TEST(CudaGraphMoeFragments, D5SixProfilesFullKdaChainAndTwoStates) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (auto profile : d5_moe_semantic_fragments::kKdaProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kda_full_fragments::complete(profile));
  }
}

TEST(CudaGraphMoeFragments, D5SixProfilesBoundedKdaActualStateFeedback) {
  namespace moe = d5_moe_semantic_fragments;
  for (auto profile : moe::kKdaProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(moe::kda(profile));
  }
}

TEST(CudaGraphMoeFragments, D5KimiThreeProfilesSituActivation) {
  namespace moe = d5_moe_semantic_fragments;
  for (std::size_t i = 0; i < 3; ++i) {
    auto profile = moe::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    for (auto width : moe::kSituWidths) {
      SCOPED_TRACE(width);
      run_q24_semantic_fragment(moe::situ(profile, width));
    }
  }
}

TEST(CudaGraphMoeFragments, D5SevenProfilesHyperConnectionCore) {
  namespace moe = d5_moe_semantic_fragments;
  for (const auto &profile : moe::kHyperProfiles) {
    SCOPED_TRACE(profile.profile->model_repository);
    SCOPED_TRACE(profile.profile->variant);
    run_q24_semantic_fragment(moe::hyper_connection(profile));
  }
}

TEST(CudaGraphQwenRecurrent, D5NineProfilesUseCompleteGateAndActualStateFeedback) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  if (!d5_qwen35_fixture::root())
    GTEST_SKIP() << "HYLE_D5_QWEN35_FIXTURE is required for actual 2B parameters";
  for (auto profile : sem::kRecurrentProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(sem::gdn(profile, 8));
  }
}

TEST(CudaGraphQwen38_24T, D5GgufAttentionGdnAndMtpUseRealRoleTypes) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  d5_gguf_test::QuantizerCleanup cleanup;
  for (auto profile : sem::kProfiles) {
    SCOPED_TRACE(profile->variant);
    {
      SCOPED_TRACE("attention");
      run_q24_semantic_fragment(sem::attention(profile));
    }
    {
      SCOPED_TRACE("gdn");
      run_q24_semantic_fragment(sem::gdn(profile));
    }
    {
      SCOPED_TRACE("mtp");
      run_q24_semantic_fragment(sem::mtp(profile));
    }
  }
}

TEST(CudaGraphQwen38_24T, D5GgufAttentionAndGdnProjectionRoleBatch) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  d5_gguf_test::QuantizerCleanup cleanup;
  for (auto profile : sem::kProfiles) {
    SCOPED_TRACE(profile->variant);
    for (const auto &role : sem::kProjections) {
      SCOPED_TRACE(role.tensor);
      run_q24_semantic_fragment(sem::projection(profile, role));
    }
  }
}

TEST(CudaGraphQwen38_24T, D5OfficialAttentionAndGdnProjectionRoleBatch) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  for (const auto &role : sem::kProjections) {
    SCOPED_TRACE(role.tensor);
    run_q24_semantic_fragment(sem::official_projection(role));
  }
  run_q24_semantic_fragment(
      sem::gdn(&d5_qwen38_24t_fixture::kOfficialProfile));
}

void run_full_moe_profile(const c::ModelArtifactProfileDescription *profile) {
  namespace moe = d5_qwen38_24t_moe_profiles;
  d5_gguf_test::QuantizerCleanup cleanup;
  auto fixture = moe::make(profile);
  auto full = qfull::make_graph(fixture.layout);
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  std::array<hyle::mem::Buffer, 10> constants;
  for (std::size_t index = 0; index < constants.size(); ++index)
    constants[index] = values_buffer<std::byte>(resource, fixture.constants[index]);
  fixture.constants = {};
  const std::array bindings{
      g::ConstantBinding{full.constants[0],
                         c::ConstTensorView::create(full.constant_descs[0],
                                                    constants[0].view()).value()},
      g::ConstantBinding{full.constants[1],
                         c::ConstTensorView::create(full.constant_descs[1],
                                                    constants[1].view()).value()},
      g::ConstantBinding{full.constants[2],
                         c::ConstTensorView::create(full.constant_descs[2],
                                                    constants[2].view()).value()},
      g::ConstantBinding{full.constants[3],
                         c::ConstTensorView::create(full.constant_descs[3],
                                                    constants[3].view()).value()},
      g::ConstantBinding{full.constants[4],
                         c::ConstTensorView::create(full.constant_descs[4],
                                                    constants[4].view()).value()},
      g::ConstantBinding{full.constants[5],
                         c::ConstTensorView::create(full.constant_descs[5],
                                                    constants[5].view()).value()},
      g::ConstantBinding{full.constants[6],
                         c::ConstTensorView::create(full.constant_descs[6],
                                                    constants[6].view()).value()},
      g::ConstantBinding{full.constants[7],
                         c::ConstTensorView::create(full.constant_descs[7],
                                                    constants[7].view()).value()},
      g::ConstantBinding{full.constants[8],
                         c::ConstTensorView::create(full.constant_descs[8],
                                                    constants[8].view()).value()},
      g::ConstantBinding{full.constants[9],
                         c::ConstTensorView::create(full.constant_descs[9],
                                                    constants[9].view()).value()}};
  const auto workload = q::workload(*profile, c::WorkloadPhase::kDecode, 0x0a12);
  auto compiled_result = g::cuda::compile(
      full.graph, context, resource,
      {.constants = bindings, .workload = workload});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  expect_mature(compiled);
  auto invocation = compiled.executable.create_invocation().value();
  auto activation_buffer = output_buffer(resource, full.activation_desc);
  auto outputs = std::array{
      output_buffer(resource, full.output_descs[0]),
      output_buffer(resource, full.output_descs[1]),
      output_buffer(resource, full.output_descs[2]),
      output_buffer(resource, full.output_descs[3]),
      output_buffer(resource, full.output_descs[4])};
  ASSERT_TRUE(invocation
                  .bind(full.activation,
                        c::ConstTensorView::create(full.activation_desc,
                                                   activation_buffer.view())
                            .value())
                  .ok());
  for (std::size_t i = 0; i < outputs.size(); ++i)
    ASSERT_TRUE(invocation
                    .bind(full.outputs[i],
                          c::TensorView::create(full.output_descs[i],
                                                outputs[i].view())
                              .value())
                    .ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
  for (const auto &expected : fixture.expected) {
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(expected.activation)))
                    .ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    const auto ids = half_test::read<std::int32_t>(outputs[0]);
    ASSERT_TRUE(std::ranges::equal(ids, expected.ids));
    for (std::size_t output = 1; output < outputs.size(); ++output) {
      const auto actual = half_test::read<float>(outputs[output]);
      const auto &oracle = output == 1 ? expected.routed
                           : output == 2 ? expected.combined
                           : output == 3 ? expected.shared
                                         : expected.full;
      moe::expect_output(profile, actual, oracle);
    }
  }
}

TEST(CudaGraphQwen38_24T, D5FullRouterSharedExpertAndNearTieReplay) {
  run_full_moe_profile(&q24::kOfficialProfile);
}

TEST(CudaGraphQwen38_24T, D5GgufFullRouterSharedExpertAndNearTieReplay) {
  for (const auto *profile : {&d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
                               &d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile}) {
    SCOPED_TRACE(profile->variant);
    run_full_moe_profile(profile);
  }
}

TEST_P(CudaGraphQwen38Profiles,
       D5Qwen38ProfilePartialRopeGqaAndOutputGateReplay) {
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
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto query_scale_buffer = values_buffer<BF16>(resource, query_scale);
  auto key_scale_buffer = values_buffer<BF16>(resource, key_scale);
  auto position_buffer = values_buffer<std::int32_t>(resource, position);
  auto frequency_buffer = values_buffer<float>(resource, frequencies);
  auto query_buffer = output_buffer(resource, query_desc);
  auto key_buffer = output_buffer(resource, kv_desc);
  auto value_buffer = output_buffer(resource, kv_desc);
  auto gate_buffer = output_buffer(resource, query_desc);
  auto result_buffer = output_buffer(resource, query_desc);

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
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 4);
  auto result = g::cuda::compile(
      logical, context, resource,
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
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
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

TEST_P(CudaGraphQwen38Profiles,
       D5Qwen38ProfileMtpUsesExactConcatWidthAndReplays) {
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
  auto context = g::cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto weight_buffer = values_buffer<BF16>(resource, weight_values);
  weight_values.clear();
  weight_values.shrink_to_fit();
  auto scale_buffer = values_buffer<BF16>(resource, scale_values);
  auto first_buffer = output_buffer(resource, half_desc);
  auto second_buffer = output_buffer(resource, half_desc);
  auto projected_buffer = output_buffer(resource, half_desc);
  auto converted_buffer = output_buffer(resource, output_f32_desc);

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
  const std::array constants{
      g::ConstantBinding{
          weight,
          c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()},
      g::ConstantBinding{
          scale,
          c::ConstTensorView::create(scale_desc, scale_buffer.view()).value()}};
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x0d0e);
  auto result = g::cuda::compile(
      logical, context, resource,
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
  ASSERT_TRUE(g::cuda::prepare_replay(invocation, queue).ok());
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
    D5OfficialAndPrimaryGguf, CudaGraphQwen38Profiles,
    ::testing::Values(&qcase::kOfficial27B, &qcase::kPrimaryGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, CudaGraphQwen38Profiles,
    ::testing::Values(&qcase::kFidelityGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5Official24T, CudaGraphQwen38Profiles,
    ::testing::Values(&qcase::kOfficial24T));

class CudaGraphQwenOfficialVocabulary
    : public ::testing::TestWithParam<const vocab::ProfileCase *> {};

TEST_P(CudaGraphQwenOfficialVocabulary,
       D5WholeVocabularyHeadStreamsAllRows) {
  const auto &shape = *GetParam();
  const auto activation_desc = dense(c::DataType::kBFloat16, {1, shape.hidden});
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 1);
  auto context = g::cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice)
                    .value();
  const auto resource = memory.ref();
  struct HeadGraph final {
    g::CompiledGraph compiled;
    g::ValueId activation;
    g::ValueId weight;
    g::ValueId logits;
  };
  const auto make_head = [&](std::int64_t rows) {
    const auto stored = dense(c::DataType::kBFloat16, {rows, shape.hidden});
    const auto weight_desc = stored.transposed().value();
    auto builder = g::GraphBuilder::create().value();
    const auto activation = builder.input(activation_desc).value();
    const auto weight = builder.input(weight_desc).value();
    const auto logits =
        builder.add(c::Operation(c::MatMul{}), activation, weight).value();
    EXPECT_TRUE(builder.publish(logits).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled =
        g::cuda::compile(logical, context, resource, {.workload = workload})
            .value();
    expect_mature(compiled);
    EXPECT_EQ(compiled.executable.workload(), workload);
    return HeadGraph{std::move(compiled), activation, weight, logits};
  };
  auto full = make_head(vocab::kShardRows);
  auto tail = make_head(vocab::kVocabularyRows % vocab::kShardRows);

  const auto max_weight = dense(c::DataType::kBFloat16,
                                {vocab::kShardRows, shape.hidden});
  const auto max_output =
      dense(c::DataType::kBFloat16, {1, vocab::kShardRows});
  auto weight_buffer = output_buffer(resource, max_weight);
  auto logits_buffer = output_buffer(resource, max_output);
  const std::array activations{vocab::activation(shape.hidden, 3),
                               vocab::activation(shape.hidden, 17)};
  auto activation_buffer = values_buffer<BF16>(resource, activations[0]);
  auto full_invocation = full.compiled.executable.create_invocation().value();
  auto tail_invocation = tail.compiled.executable.create_invocation().value();
  const auto bind = [&](g::Invocation &invocation, const HeadGraph &graph,
                        std::int64_t rows) {
    const auto stored = dense(c::DataType::kBFloat16, {rows, shape.hidden});
    const auto weight_desc = stored.transposed().value();
    const auto output_desc = dense(c::DataType::kBFloat16, {1, rows});
    const auto weight_view =
        weight_buffer.view().subview(0, stored.required_bytes().value()).value();
    const auto output_view =
        logits_buffer.view().subview(0, output_desc.required_bytes().value())
            .value();
    EXPECT_TRUE(invocation
                    .bind(graph.activation,
                          c::ConstTensorView::create(activation_desc,
                                                     activation_buffer.view())
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(graph.weight,
                          c::ConstTensorView::create(weight_desc, weight_view)
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(graph.logits,
                          c::TensorView::create(output_desc, output_view)
                              .value())
                    .ok());
  };
  bind(full_invocation, full, vocab::kShardRows);
  bind(tail_invocation, tail, vocab::kVocabularyRows % vocab::kShardRows);
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(full_invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(tail_invocation.prepare_replay(queues).ok());

  std::array<std::vector<BF16>, 2> logical_logits{
      std::vector<BF16>(vocab::kVocabularyRows),
      std::vector<BF16>(vocab::kVocabularyRows)};
  std::int64_t visited = 0;
  for (const auto shard : vocab::shards()) {
    const auto weights = vocab::head_weight(shard, shape.hidden);
    const auto weight_view = weight_buffer.view()
                                 .subview(0, weights.size() * sizeof(BF16))
                                 .value();
    ASSERT_TRUE(weight_view.write(std::as_bytes(std::span(weights))).ok());
    auto &invocation = shard.rows == vocab::kShardRows ? full_invocation
                                                      : tail_invocation;
    const auto output_bytes =
        static_cast<std::size_t>(shard.rows) * sizeof(BF16);
    const auto output_view = logits_buffer.view().subview(0, output_bytes).value();
    for (std::size_t replay = 0; replay < activations.size(); ++replay) {
      ASSERT_TRUE(activation_buffer.view()
                      .write(std::as_bytes(std::span(activations[replay])))
                      .ok());
      auto done = invocation.submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      std::vector<BF16> actual(static_cast<std::size_t>(shard.rows));
      ASSERT_TRUE(output_view.read(std::as_writable_bytes(std::span(actual))).ok());
      std::ranges::copy(actual, logical_logits[replay].begin() + shard.first);
    }
    visited += shard.rows;
  }
  ASSERT_EQ(visited, vocab::kVocabularyRows);
  for (std::size_t replay = 0; replay < activations.size(); ++replay)
    for (const auto shard : vocab::shards()) {
      const auto expected = vocab::expected_logits(activations[replay], shard);
      const auto actual = std::span(logical_logits[replay])
                              .subspan(shard.first, shard.rows);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto reference = vocab::decode(expected[i]);
        ASSERT_NEAR(vocab::decode(actual[i]), reference,
                    std::abs(reference) * 0.012 + 2.0e-5)
            << "global vocabulary row " << shard.first + i;
      }
    }
}

TEST_P(CudaGraphQwenOfficialVocabulary,
       D5WholeVocabularyGatherMapsGlobalRows) {
  const auto &shape = *GetParam();
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 1);
  auto context = g::cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice)
                    .value();
  const auto resource = memory.ref();
  d5_qwen_vocabulary_graph_test::check_gather(
      context, resource, workload, shape.hidden,
      [&](const g::Graph &logical, const c::WorkloadContext &identity) {
        return g::cuda::compile(logical, context, resource,
                                {.workload = identity})
            .value();
      },
      [&](const g::CompiledGraph &compiled) { expect_mature(compiled); });
}

INSTANTIATE_TEST_SUITE_P(
    D5Official2B27BAnd24T, CudaGraphQwenOfficialVocabulary,
    ::testing::Values(&vocab::kOfficial2B, &vocab::kOfficial27B,
                      &vocab::kOfficial24T));

TEST(CudaGraphIq1ModelContracts, EightCurrentExpertShapesUseOrdinaryGraphAndReplay) {
  std::size_t count = 0;
  for (const auto &item : cpu_c5_inputs::cases) {
    if (!cuda_iq1_model_contracts::required(item)) continue;
    SCOPED_TRACE(item.id);
    const auto profile = cuda_iq1_model_contracts::identity(item);
    run_q24_semantic_fragment(cuda_iq1_model_contracts::make(item, profile));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ++count;
  }
  EXPECT_EQ(count, 8U);
}
#define HYLE_TEST_GRAPH_ROUTE_CUDA
#include "../compute_models/cuda_n1_non_gguf_tests.h"
#undef HYLE_TEST_GRAPH_ROUTE_CUDA
} // namespace
