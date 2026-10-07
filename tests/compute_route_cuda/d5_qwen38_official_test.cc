#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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
#include "../compute_models/d5_qwen_vocabulary_compute_test.h"
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
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/stats.h"

namespace {
namespace c = hyle::compute;
namespace cuda = hyle::compute::cuda;
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

struct DirectCall final {
  c::PreparedOperation prepared;
  hyle::mem::Buffer workspace;
  c::OperationInvocation invocation;
};

DirectCall make_call(const c::Context &context,
                     mem::MemoryResourceRef resource,
                     c::Operation operation,
                     std::span<const c::TensorDesc> inputs,
                     std::span<const c::TensorDesc> outputs,
                     const c::WorkloadContext &workload) {
  const auto kind = operation.kind();
  auto prepared = c::prepare(context, std::move(operation), inputs, outputs,
                             resource, workload);
  if (!prepared)
    throw std::runtime_error(std::string(c::operation_name(kind)) +
                             ": " +
                             std::string(prepared.error().message()));
  const auto requirement = prepared->workspace_requirement();
  auto workspace =
      requirement.size == 0
          ? hyle::mem::Buffer{}
          : hyle::mem::Buffer::create(resource, requirement.size,
                                      requirement.alignment)
                .value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  return {std::move(*prepared), std::move(workspace), std::move(invocation)};
}

void expect_mature(const DirectCall &call, const c::WorkloadContext &workload) {
  const auto provider = call.prepared.provider().id;
  const auto kind = call.prepared.kind();
  // D4 retained these exact structural boundaries; no arithmetic fallback.
  bool retained_storage = false;
  if (provider == c::cuda_native::kProviderId && call.prepared.input_count() > 0) {
    const auto input_result = call.prepared.input_description();
    const auto &input = input_result.value();
    retained_storage =
        (kind == c::OperationKind::kCopy && input.data_type() == c::DataType::kInt32) ||
        (kind == c::OperationKind::kSelect && input.data_type() == c::DataType::kBool &&
         call.prepared.output_description()->data_type() == c::DataType::kFloat32);
  }
  EXPECT_TRUE(provider == c::ggml_cuda::kProviderId ||
              provider == c::cublas::kProviderId ||
              provider == c::cuda_native::kSglangProviderId ||
              provider == c::cuda_native::kDeepSeekProviderId ||
              provider == c::cuda_native::kCcclProviderId ||
              provider == c::torch_cuda::kProviderId ||
              d5_optional_cuda_provider::is_flashinfer(provider) || retained_storage)
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

class CudaQwen38Profiles
    : public ::testing::TestWithParam<const qcase::ProfileCase *> {};

TEST(CudaQwen38_24T, D5ExactWidthCausalConvDirectReplay) {
  const d5_qwen38_gdn_fixture::ConvFixture fixture(
      q24::kGdnKeyHeads, q24::kGdnValueHeads);
  const auto workload = q24::workload(c::WorkloadPhase::kDecode, 5);
  auto context = cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice)
                    .value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array inputs{fixture.samples_desc, fixture.weights_desc};
  const std::array outputs{fixture.output_desc};
  auto call = make_call(context, resource, c::model_ops::causal_conv1d(),
                        inputs, outputs, workload);
  expect_mature(call, workload);
  auto samples = output_buffer(resource, fixture.samples_desc);
  auto weights = values_buffer<float>(resource, fixture.weights);
  auto result = output_buffer(resource, fixture.output_desc);
  const std::array input_views{
      c::ConstTensorView::create(fixture.samples_desc, samples.view()).value(),
      c::ConstTensorView::create(fixture.weights_desc, weights.view()).value()};
  const std::array output_views{
      c::TensorView::create(fixture.output_desc, result.view()).value()};
  for (const int seed : {3, 17}) {
    const auto input = fixture.samples(seed);
    ASSERT_TRUE(samples.view().write(std::as_bytes(std::span(input))).ok());
    run(call, queue, input_views, output_views);
    const auto actual = half_test::read<float>(result);
    const auto expected = fixture.oracle(input);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1.0e-6) << i;
  }
}

TEST_P(CudaQwen38Profiles,
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
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x01030e0f);

  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context =
      cuda::create_context({.metadata_allocator = stats.get()}).value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array gather_inputs{table_desc, ids_desc};
  const std::array token_output{token_desc};
  auto gather = make_call(context, resource, c::Operation(c::Gather{}), gather_inputs,
                          token_output, workload);
  const std::array norm_inputs{token_desc, scale_desc};
  auto norm = make_call(context, resource, c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
                        norm_inputs, token_output, workload);
  const std::array add_inputs{token_desc, token_desc};
  auto residual =
      make_call(context, resource, c::Operation(c::Binary{c::BinaryKind::kAdd}),
                add_inputs, token_output, workload);
  const std::array projection_inputs{token_desc, projection_desc};
  auto projection = make_call(context, resource, c::Operation(c::MatMul{}),
                              projection_inputs, token_output, workload);
  const std::array head_inputs{token_desc, head_desc};
  const std::array head_output{logits_desc};
  auto head = make_call(context, resource, c::Operation(c::MatMul{}), head_inputs,
                        head_output, workload);
  for (const auto *call : {&gather, &norm, &residual, &projection, &head})
    expect_mature(*call, workload);

  auto table_values = q::values(q::kEmbeddingRows * shape.hidden, 5);
  auto head_values = q::values(q::kEmbeddingRows * shape.hidden, 7, 96.0);
  auto scale_values = q::scales(shape.hidden, 3);
  auto projection_values = q::sparse_output_major_weight(
      shape.hidden, shape.hidden, projection_seed);
  auto table_buffer = values_buffer<BF16>(resource, table_values);
  auto head_buffer = values_buffer<BF16>(resource, head_values);
  auto scale_buffer = values_buffer<BF16>(resource, scale_values);
  auto projection_buffer = values_buffer<BF16>(resource, projection_values);
  projection_values.clear();
  projection_values.shrink_to_fit();
  auto ids_buffer = output_buffer(resource, ids_desc);
  auto embedded_buffer = output_buffer(resource, token_desc);
  auto normalized_buffer = output_buffer(resource, token_desc);
  auto residual_buffer = output_buffer(resource, token_desc);
  auto projected_buffer = output_buffer(resource, token_desc);
  auto logits_buffer = output_buffer(resource, logits_desc);

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
  std::uint64_t warmed_allocs = 0;
  std::uint64_t warmed_reallocs = 0;
  std::size_t replay = 0;
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
                           .subspan(row_index * shape.hidden, shape.hidden);
      const auto normalized = q::rms_norm(row, scale_values, 1.0e-6);
      std::vector<BF16> residual_values(shape.hidden);
      std::vector<double> residual_expected(shape.hidden);
      for (std::size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] = q::encode(q::decode(row[i]) + normalized[i]);
        residual_expected[i] = q::decode(residual_values[i]);
      }
      expect_near(std::span(actual_residual).subspan(token * shape.hidden,
                                                     shape.hidden),
                  residual_expected);
      const auto projected = q::sparse_projection(
          residual_values, shape.hidden, projection_seed);
      expect_near(std::span(actual_projected)
                      .subspan(token * shape.hidden, shape.hidden),
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
  auto failed = gather.invocation.submit(queue, gather_views, embedded_view);
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().ok());
  EXPECT_EQ(read_bytes(embedded_buffer), last_embedded);
  EXPECT_EQ(read_bytes(projected_buffer), last_projected);
  EXPECT_EQ(read_bytes(logits_buffer), last_logits);
}

TEST_P(CudaQwen38Profiles,
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
  const auto expanded_desc =
      dense(c::DataType::kBFloat16, {1, shape.ffn_hidden});
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x0203);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array norm_inputs{activation_desc, scale_desc};
  const std::array activation_output{activation_desc};
  auto norm = make_call(context, resource, c::Operation(c::RmsNorm{}), norm_inputs,
                        activation_output, workload);
  const std::array gate_inputs{activation_desc, gate_desc};
  const std::array expanded_output{expanded_desc};
  auto gate = make_call(context, resource, c::Operation(c::MatMul{}), gate_inputs,
                        expanded_output, workload);
  auto up = make_call(context, resource, c::Operation(c::MatMul{}), gate_inputs,
                      expanded_output, workload);
  const std::array gated_inputs{expanded_desc, expanded_desc};
  auto gated = make_call(
      context, resource,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      gated_inputs, expanded_output, workload);
  const std::array down_inputs{expanded_desc, down_desc};
  auto down = make_call(context, resource, c::Operation(c::MatMul{}), down_inputs,
                        activation_output, workload);
  for (const auto *call : {&norm, &gate, &up, &gated, &down})
    expect_mature(*call, workload);

  auto scale_values = q::scales(shape.hidden, 9);
  auto gate_values =
      q::sparse_output_major_weight(shape.ffn_hidden, shape.hidden, seed);
  auto up_values =
      q::sparse_output_major_weight(shape.ffn_hidden, shape.hidden, seed + 1);
  auto down_values =
      q::sparse_output_major_weight(shape.hidden, shape.ffn_hidden, seed + 2);
  auto scale_buffer = values_buffer<BF16>(resource, scale_values);
  auto gate_weight_buffer = values_buffer<BF16>(resource, gate_values);
  gate_values.clear();
  gate_values.shrink_to_fit();
  auto up_weight_buffer = values_buffer<BF16>(resource, up_values);
  up_values.clear();
  up_values.shrink_to_fit();
  auto down_weight_buffer = values_buffer<BF16>(resource, down_values);
  down_values.clear();
  down_values.shrink_to_fit();
  auto activation_buffer = output_buffer(resource, activation_desc);
  auto normalized_buffer = output_buffer(resource, activation_desc);
  auto gate_buffer = output_buffer(resource, expanded_desc);
  auto up_buffer = output_buffer(resource, expanded_desc);
  auto gated_buffer = output_buffer(resource, expanded_desc);
  auto result_buffer = output_buffer(resource, activation_desc);

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
    const auto activation = q::values(shape.hidden, replay_seed);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(activation)))
                    .ok());
    run(norm, queue, norm_views, normalized_view);
    run(gate, queue, gate_views, gate_view);
    run(up, queue, up_views, up_view);
    run(gated, queue, gated_views, gated_view);
    run(down, queue, down_views, result_view);
    const auto actual = half_test::read<BF16>(result_buffer);
    const auto expected = q::ffn(activation, scale_values, seed, shape.hidden, shape.ffn_hidden);
    expect_near(actual, expected, 0.025);
  }
}

TEST(CudaQwen38_24T,
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
  const auto expanded_desc = dense(
      c::DataType::kFloat32, {1, q24::kTopK, q24::kExpertHidden});
  const auto output_desc = activation_desc;
  const auto workload =
      q24::workload(c::WorkloadPhase::kDecode, 0x0a0b);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array grouped_inputs{weight_desc, activation_desc, ids_desc};
  const std::array expanded_output{expanded_desc};
  auto gate = make_call(context, resource, c::model_ops::grouped_matmul(),
                        grouped_inputs, expanded_output, workload);
  auto up = make_call(context, resource, c::model_ops::grouped_matmul(),
                      grouped_inputs, expanded_output, workload);
  const std::array gated_inputs{expanded_desc, expanded_desc};
  auto gated = make_call(
      context, resource,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      gated_inputs, expanded_output, workload);
  const std::array down_inputs{down_weight_desc, expanded_desc, ids_desc};
  const std::array output{output_desc};
  auto down = make_call(context, resource, c::model_ops::grouped_matmul(),
                        down_inputs, output, workload);
  EXPECT_EQ(gate.prepared.provider().id, c::ggml_cuda::kProviderId);
  EXPECT_EQ(up.prepared.provider().id, c::ggml_cuda::kProviderId);
  EXPECT_EQ(down.prepared.provider().id, c::ggml_cuda::kProviderId);
  for (const auto *call : {&gate, &up, &gated, &down})
    expect_mature(*call, workload);

  auto gate_weight = q24::expert_weight(q24::kExpertHidden, q24::kHidden,
                                        seed);
  auto up_weight = q24::expert_weight(q24::kExpertHidden, q24::kHidden,
                                      seed + 1);
  auto down_weight = q24::expert_weight(q24::kHidden, q24::kExpertHidden,
                                        seed + 2);
  auto gate_weight_buffer = values_buffer<BF16>(resource, gate_weight);
  gate_weight.clear();
  gate_weight.shrink_to_fit();
  auto up_weight_buffer = values_buffer<BF16>(resource, up_weight);
  up_weight.clear();
  up_weight.shrink_to_fit();
  auto down_weight_buffer = values_buffer<BF16>(resource, down_weight);
  down_weight.clear();
  down_weight.shrink_to_fit();
  const auto ids = q24::active_expert_ids();
  auto ids_buffer = values_buffer<std::int32_t>(resource, ids);
  auto activation_buffer = output_buffer(resource, activation_desc);
  auto gate_buffer = output_buffer(resource, expanded_desc);
  auto up_buffer = output_buffer(resource, expanded_desc);
  auto gated_buffer = output_buffer(resource, expanded_desc);
  auto result_buffer = output_buffer(resource, output_desc);

  const auto id_view =
      c::ConstTensorView::create(ids_desc, ids_buffer.view()).value();
  const auto activation_view = c::ConstTensorView::create(
                                   activation_desc, activation_buffer.view())
                                   .value();
  const std::array gate_views{
      c::ConstTensorView::create(weight_desc, gate_weight_buffer.view()).value(),
      activation_view, id_view};
  const std::array up_views{
      c::ConstTensorView::create(weight_desc, up_weight_buffer.view()).value(),
      activation_view, id_view};
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
      c::ConstTensorView::create(down_weight_desc, down_weight_buffer.view())
          .value(),
      c::ConstTensorView::create(expanded_desc, gated_buffer.view()).value(),
      id_view};
  const std::array result_output{
      c::TensorView::create(output_desc, result_buffer.view()).value()};

  for (int replay : {3, 17}) {
    const auto input = q24::expert_input(replay);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(input)))
                    .ok());
    run(gate, queue, gate_views, gate_output);
    run(up, queue, up_views, up_output);
    run(gated, queue, gated_views, gated_output);
    run(down, queue, down_views, result_output);
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

TEST(CudaQwen38_24T, D5GgufGroupedExpertKeepsNativeBlocksAndDynamicIds) {
  namespace gguf = d5_qwen38_24t_grouped_gguf_fixture;
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
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
    const auto workload = q::workload(*test.profile,
                                     c::WorkloadPhase::kDecode, 0x0b21);
    auto call = make_call(context, resource, c::model_ops::grouped_matmul(),
                          std::array{weight_desc, activation_desc, ids_desc},
                          std::array{output_desc}, workload);
    EXPECT_EQ(call.prepared.provider().id, c::ggml_cuda::kProviderId);
    auto packed = gguf::pack_weight(test, weight_desc.required_bytes().value());
    auto weight_buffer = output_buffer(resource, weight_desc);
    ASSERT_TRUE(weight_buffer.view().write(packed.bytes).ok());
    auto activation_buffer = output_buffer(resource, activation_desc);
    auto ids_buffer = output_buffer(resource, ids_desc);
    auto result_buffer = output_buffer(resource, output_desc);
    const std::array inputs{
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value(),
        c::ConstTensorView::create(activation_desc, activation_buffer.view())
            .value(),
        c::ConstTensorView::create(ids_desc, ids_buffer.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, result_buffer.view()).value()};
    for (int replay = 0; replay < 2; ++replay) {
      const auto activation = q24::expert_input(replay);
      const auto selected = gguf::ids(replay);
      ASSERT_TRUE(activation_buffer.view()
                      .write(std::as_bytes(std::span(activation))).ok());
      ASSERT_TRUE(ids_buffer.view()
                      .write(std::as_bytes(std::span(selected))).ok());
      run(call, queue, inputs, outputs);
      const auto actual = half_test::read<float>(result_buffer);
      const auto expected = gguf::expected(activation, selected, packed.decoded);
      d5_gguf_test::expect_quantized_matmul_close(actual, expected);
    }

    const auto retained = half_test::read<float>(result_buffer);
    for (std::int32_t bad : {-1, static_cast<std::int32_t>(gguf::kResidentExperts)}) {
      auto selected = gguf::ids(1);
      selected.back() = bad;
      ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(selected))).ok());
      auto attempt = call.prepared.create_invocation(call.workspace.view()).value();
      auto failed = attempt.submit(queue, inputs, outputs);
      ASSERT_TRUE(failed) << failed.error().message();
      EXPECT_TRUE(failed->wait().IsOutOfRange());
      EXPECT_EQ(half_test::read<float>(result_buffer), retained);
    }
    const auto valid = gguf::ids(1);
    ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(valid))).ok());
    auto recovered = call.invocation.submit(queue, inputs, outputs);
    ASSERT_TRUE(recovered) << recovered.error().message();
    EXPECT_TRUE(recovered->wait().ok());
    EXPECT_EQ(half_test::read<float>(result_buffer), retained);
  }
}

TEST(CudaQwen38_24T, D5OfficialBfloatRouterPreservesF32NearTieScores) {
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, q24::kHidden});
  const auto weight_desc =
      dense(c::DataType::kBFloat16, {q24::kExperts, q24::kHidden});
  const auto output_desc =
      dense(c::DataType::kFloat32, {1, q24::kExperts});
  const auto workload = q24::workload(c::WorkloadPhase::kDecode, 0x0a10);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const std::array inputs_desc{activation_desc, weight_desc};
  const std::array outputs_desc{output_desc};
  auto call = make_call(context, resource,
                        c::model_ops::bfloat_linear_f32(), inputs_desc,
                        outputs_desc, workload);
  EXPECT_EQ(call.prepared.provider().id, c::ggml_cuda::kProviderId);
  std::vector<BF16> weights(q24::kExperts * q24::kHidden);
  weights[9 * q24::kHidden + 20] = q::encode(1.0);
  weights[10 * q24::kHidden + 20] = q::encode(1.0);
  weights[10 * q24::kHidden + 21] = q::encode(1.0);
  auto weight_buffer = values_buffer<BF16>(resource, weights);
  auto activation_buffer = output_buffer(resource, activation_desc);
  auto output = output_buffer(resource, output_desc);
  const std::array inputs{
      c::ConstTensorView::create(activation_desc,
                                 activation_buffer.view()).value(),
      c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()};
  const std::array outputs{
      c::TensorView::create(output_desc, output.view()).value()};
  auto queue = context.create_queue().value();
  for (const double fraction : {0.001953125, 0.0029296875}) {
    std::vector<BF16> activation(q24::kHidden);
    activation[20] = q::encode(1.0);
    activation[21] = q::encode(fraction);
    ASSERT_TRUE(activation_buffer.view()
                    .write(std::as_bytes(std::span(activation))).ok());
    run(call, queue, inputs, outputs);
    const auto scores = half_test::read<float>(output);
    ASSERT_EQ(scores.size(), q24::kExperts);
    EXPECT_NEAR(scores[9], 1.0, 1.0e-6);
    EXPECT_NEAR(scores[10], 1.0 + fraction, 1.0e-6);
  }
}

void prepare_model_constants(const d5_qwen38_24t_semantic_fragments::Fragment &cold,
    c::Context &context, mem::MemoryResourceRef resource,
    std::vector<mem::Buffer> &result) {
  ASSERT_TRUE(cold.inputs.empty());
  ASSERT_FALSE(cold.preparation);
  const auto workload = q::workload(*cold.profile, c::WorkloadPhase::kDecode, cold.region);
  auto queue = context.create_queue().value();
  std::vector<mem::Buffer> buffers(cold.descriptions.size());
  std::vector<mem::BufferView> views(cold.descriptions.size());
  const auto allocate = [&](qfull::ValueId id) {
    buffers[id.ordinal()] = output_buffer(resource, cold.description(id));
    views[id.ordinal()] = buffers[id.ordinal()].view();
  };
  for (std::size_t i = 0; i < cold.constants.size(); ++i) {
    auto id = cold.constants[i];
    allocate(id);
    ASSERT_TRUE(views[id.ordinal()].write(cold.constant_bytes[i]).ok());
  }
  for (const auto &step : cold.nodes) {
    if (const auto *view = step.operation.get_if<c::View>()) {
      const auto input = step.inputs[0], output = step.outputs[0];
      views[output.ordinal()] = views[input.ordinal()];
      if (view->kind() == c::ViewKind::kSlice) {
        const auto &desc = cold.description(input);
        const auto offset = std::size_t(view->start() * desc.byte_strides()[view->axis()]);
        views[output.ordinal()] = views[input.ordinal()].subview(offset,
            cold.description(output).required_bytes().value()).value();
      }
      continue;
    }
    std::vector<c::TensorDesc> in_desc, out_desc;
    std::vector<c::ConstTensorView> inputs;
    std::vector<c::TensorView> outputs;
    for (auto id : step.inputs) {
      in_desc.push_back(cold.description(id));
      inputs.push_back(c::ConstTensorView::create(cold.description(id), views[id.ordinal()]).value());
    }
    for (auto id : step.outputs) {
      allocate(id);
      out_desc.push_back(cold.description(id));
      outputs.push_back(c::TensorView::create(cold.description(id), views[id.ordinal()]).value());
    }
    auto call = make_call(context, resource, step.operation, in_desc, out_desc, workload);
    expect_mature(call, workload);
    run(call, queue, inputs, outputs);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
  for (auto id : cold.outputs) result.push_back(std::move(buffers[id.ordinal()]));
  d5_fp8_weight_preparation::check_constants(cold, result);
  // Encoded weights, decoded full matrix and cold programs die here; only
  // the two real device caches survive into the repeated model program.
}

void run_q24_semantic_fragment(
    d5_qwen38_24t_semantic_fragments::Fragment fixture) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const auto workload =
      q::workload(*fixture.profile, fixture.phase, fixture.region);
  std::vector<hyle::mem::Buffer> buffers(fixture.descriptions.size());
  std::vector<hyle::mem::BufferView> views(fixture.descriptions.size());
  const auto allocate = [&](qfull::ValueId id) {
    const auto &description = fixture.description(id);
    buffers[id.ordinal()] =
        hyle::mem::Buffer::create(resource,
                                  description.required_bytes().value(), 256)
            .value();
    views[id.ordinal()] = buffers[id.ordinal()].view();
  };
  for (auto id : fixture.inputs)
    allocate(id);
  std::vector<mem::Buffer> prepared_constants;
  if (fixture.preparation) {
    prepare_model_constants(*fixture.preparation, context, resource, prepared_constants);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(prepared_constants.size(), fixture.prepared_constants.size());
    for (const auto &step : fixture.nodes)
      EXPECT_FALSE(step.operation.custom_as<c::model_ops::Float8WeightDecode>());
  }
  const auto &constant_bytes = fixture.constant_bytes;
  for (std::size_t index = 0; index < fixture.constants.size(); ++index) {
    const auto id = fixture.constants[index];
    const auto found = std::ranges::find(fixture.prepared_constants, index);
    if (found != fixture.prepared_constants.end()) {
      ASSERT_TRUE(constant_bytes[index].empty());
      buffers[id.ordinal()] = std::move(prepared_constants[found - fixture.prepared_constants.begin()]);
      views[id.ordinal()] = buffers[id.ordinal()].view();
      continue;
    }
    allocate(id);
    ASSERT_TRUE(views[id.ordinal()].write(constant_bytes[index]).ok());
  }
  fixture.constant_bytes = {};

  std::vector<DirectCall> calls;
  calls.reserve(fixture.nodes.size());
  for (const auto &step : fixture.nodes) {
    if (step.operation.kind() == c::OperationKind::kView) {
      ASSERT_EQ(step.inputs.size(), 1U);
      ASSERT_EQ(step.outputs.size(), 1U);
      const auto &view = *step.operation.get_if<c::View>();
      ASSERT_TRUE(view.kind() == c::ViewKind::kReshape ||
                  view.kind() == c::ViewKind::kPermute || view.kind() == c::ViewKind::kSlice);
      views[step.outputs[0].ordinal()] = views[step.inputs[0].ordinal()];
      if (view.kind() == c::ViewKind::kSlice) {
        const auto &input = fixture.description(step.inputs[0]);
        const auto axis = view.axis() < 0 ? view.axis() + std::int32_t(input.rank()) : view.axis();
        const auto offset = std::size_t(view.start()) * std::size_t(input.byte_strides()[axis]);
        const auto length = fixture.description(step.outputs[0]).required_bytes().value();
        auto sliced = views[step.inputs[0].ordinal()].subview(offset, length);
        ASSERT_TRUE(sliced) << sliced.error().message();
        views[step.outputs[0].ordinal()] = *sliced;
      }
      continue;
    }
    std::vector<c::TensorDesc> inputs, outputs;
    for (const auto id : step.inputs)
      inputs.push_back(fixture.description(id));
    for (const auto id : step.outputs) {
      outputs.push_back(fixture.description(id));
      allocate(id);
    }
    calls.push_back(make_call(context, resource, step.operation, inputs,
                              outputs, workload));
    expect_mature(calls.back(), workload);
  }

  sem::StateFeedback feedback{fixture};
  std::size_t replay_index = 0;
  for (const auto &expected : fixture.replays) {
    SCOPED_TRACE(fixture.region);
    SCOPED_TRACE("replay " + std::to_string(replay_index++));
    for (std::size_t i = 0; i < fixture.inputs.size(); ++i)
      ASSERT_TRUE(
          views[fixture.inputs[i].ordinal()].write(feedback.input(expected, i)).ok());
    std::size_t call_index = 0;
    for (const auto &step : fixture.nodes) {
      if (step.operation.kind() == c::OperationKind::kView)
        continue;
      std::vector<c::ConstTensorView> inputs;
      std::vector<c::TensorView> outputs;
      for (const auto id : step.inputs)
        inputs.push_back(c::ConstTensorView::create(fixture.description(id),
                                                    views[id.ordinal()])
                             .value());
      for (const auto id : step.outputs)
        outputs.push_back(
            c::TensorView::create(fixture.description(id), views[id.ordinal()])
                .value());
      run(calls[call_index++], queue, inputs, outputs);
    }
    for (std::size_t output = 0; output < fixture.outputs.size(); ++output) {
      SCOPED_TRACE("output " + std::to_string(output));
      const auto actual = sem::decode_output(fixture, output, half_test::read<float>(buffers[fixture.outputs[output].ordinal()]));
      ASSERT_EQ(actual.size(), expected.outputs[output].size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::isfinite(expected.outputs[output][i])) {
          ASSERT_TRUE(std::isfinite(actual[i])) << "non-finite output at " << i;
        }
      feedback.observe(output, actual);
      sem::expect_output(fixture, actual, expected.outputs[output], {},
                         fixture.exact_output(output), fixture.id_validation(output),
                         fixture.bfloat_chain_output(output),output);
    }
  }
}

TEST(CudaMoeFragments, D5OriginalRouterNativeBoundaries) {
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

TEST(CudaMoeFragments, D5OriginalRouterRejectsBindingsAndReusesIndependentLanes) {
  auto context = cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const auto logits_desc = dense(c::DataType::kFloat32, {1, 256});
  const auto bias_desc = dense(c::DataType::kFloat32, {256});
  const auto weights_desc = dense(c::DataType::kFloat32, {1, 6});
  const auto ids_desc = dense(c::DataType::kInt32, {1, 6});
  const std::array descriptions{logits_desc, bias_desc};
  const std::array results{weights_desc, ids_desc};
  const auto operation = c::model_ops::sqrt_softplus_routing(1.5F);
  auto host = c::prepare(context, operation, descriptions, results, mem::system_resource());
  EXPECT_FALSE(host);
  auto call = make_call(context, resource, operation, descriptions, results, {});
  EXPECT_EQ(call.prepared.provider().id, c::cuda_native::kSglangProviderId);
  EXPECT_EQ(call.prepared.workspace_requirement().size, 0U);
  std::vector<float> zeros(256, 0.0F);
  auto logits = values_buffer<float>(resource, zeros);
  auto bias = values_buffer<float>(resource, zeros);
  auto weights = output_buffer(resource, weights_desc);
  auto ids = output_buffer(resource, ids_desc);
  const std::array inputs{
      c::ConstTensorView::create(logits_desc, logits.view()).value(),
      c::ConstTensorView::create(bias_desc, bias.view()).value()};
  const std::array outputs{
      c::TensorView::create(weights_desc, weights.view()).value(),
      c::TensorView::create(ids_desc, ids.view()).value()};
  auto queue = context.create_queue().value();
  const auto reject = [&](std::span<const c::ConstTensorView> in,
                          std::span<const c::TensorView> out) {
    auto completion = call.invocation.submit(queue, in, out);
    EXPECT_FALSE(completion);
    EXPECT_FALSE(call.invocation.in_flight());
  };
  reject(std::span(inputs).first(1), outputs);
  auto aliased = outputs;
  aliased[0] = c::TensorView::create(
      weights_desc, logits.view().subview(0, weights.size()).value()).value();
  reject(inputs, aliased);
  aliased = outputs;
  aliased[1] = c::TensorView::create(ids_desc, weights.view()).value();
  reject(inputs, aliased);
  auto host_bias = values_buffer<float>(mem::system_resource(), zeros);
  auto wrong_resource = inputs;
  wrong_resource[1] = c::ConstTensorView::create(bias_desc, host_bias.view()).value();
  reject(wrong_resource, outputs);

  // Rejection leaves the original lane reusable. A second lane shares only
  // immutable program state and read-only inputs, with independent outputs.
  auto second_lane = call.prepared.create_invocation().value();
  auto second_queue = context.create_queue().value();
  auto second_weights = output_buffer(resource, weights_desc);
  auto second_ids = output_buffer(resource, ids_desc);
  const std::array second_outputs{
      c::TensorView::create(weights_desc, second_weights.view()).value(),
      c::TensorView::create(ids_desc, second_ids.view()).value()};
  for (int replay = 0; replay < 3; ++replay) {
    auto first = call.invocation.submit(queue, inputs, outputs);
    auto second = second_lane.submit(second_queue, inputs, second_outputs);
    ASSERT_TRUE(first) << first.error().message();
    ASSERT_TRUE(second) << second.error().message();
    ASSERT_TRUE(first->wait().ok());
    ASSERT_TRUE(second->wait().ok());
    for (const auto *buffer : {&weights, &second_weights})
      for (const auto value : half_test::read<float>(*buffer))
        EXPECT_NEAR(value, 0.25F, 1.0e-6F);
    for (const auto *buffer : {&ids, &second_ids}) {
      auto selected = half_test::read<std::int32_t>(*buffer);
      std::ranges::sort(selected);
      EXPECT_GE(selected.front(), 0);
      EXPECT_LT(selected.back(), 256);
      EXPECT_EQ(std::adjacent_find(selected.begin(), selected.end()), selected.end());
    }
  }
}

struct RouterFaultAllocator {
  std::size_t attempts = 0, fail = SIZE_MAX, live = 0;
  mem::AllocatorRef ref() {
    return {[](void *self, std::size_t bytes, std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<RouterFaultAllocator *>(self);
              if (s.attempts++ == s.fail) return nullptr;
              auto *p = mem::system_allocator().allocate(bytes, alignment);
              if (p) ++s.live;
              return p;
            },
            [](void *self, void *p, std::size_t bytes, std::size_t alignment) noexcept {
              auto &s = *static_cast<RouterFaultAllocator *>(self);
              if (p) --s.live;
              mem::system_allocator().deallocate(p, bytes, alignment);
            },
            [](void *self, void *p, std::size_t old_bytes, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<RouterFaultAllocator *>(self);
              if (bytes && s.attempts++ == s.fail) return nullptr;
              auto *next = mem::system_allocator().reallocate(p, old_bytes, bytes, alignment);
              if (bytes == 0 || next) {
                if (p) --s.live;
                if (next) ++s.live;
              }
              return next;
            }, this};
  }
};

TEST(CudaMoeFragments, D5OriginalRouterPreparationOomReleasesPartialState) {
  RouterFaultAllocator allocator;
  {
    auto context = cuda::create_context({.metadata_allocator = allocator.ref()}).value();
    auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
    const std::array inputs{dense(c::DataType::kFloat32, {513, 384}),
                            dense(c::DataType::kFloat32, {384})};
    const std::array outputs{dense(c::DataType::kFloat32, {513, 6}),
                             dense(c::DataType::kInt32, {513, 6})};
    const auto prepare = [&] {
      return c::prepare(context, c::model_ops::sqrt_softplus_routing(2.5F),
                         inputs, outputs, memory.ref());
    };
    const auto live = allocator.live, start = allocator.attempts;
    { auto prepared = prepare(); ASSERT_TRUE(prepared) << prepared.error().message(); }
    const auto count = allocator.attempts - start;
    ASSERT_GT(count, 0U);
    ASSERT_EQ(allocator.live, live);
    for (std::size_t allocation = 0; allocation < count; ++allocation) {
      SCOPED_TRACE(allocation);
      allocator.fail = allocator.attempts + allocation;
      { auto failed = prepare(); EXPECT_FALSE(failed); }
      allocator.fail = SIZE_MAX;
      EXPECT_EQ(allocator.live, live);
    }
    { auto retry = prepare(); ASSERT_TRUE(retry) << retry.error().message(); }
    EXPECT_EQ(allocator.live, live);
  }
  EXPECT_EQ(allocator.live, 0U);
}

TEST(CudaMoeFragments, D5MatureClampBoundaries) {
  run_q24_semantic_fragment(d5_moe_execution_fragments::clamp_boundaries());
}

TEST(CudaMoeFragments, D5DeepSeekWeightedHyperConnectionAndHead) {
  for (std::size_t index : {3u, 4u, 5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    d5_deepseek_remaining_fragments::check_hyper_head(model, run_q24_semantic_fragment);
  }
}

TEST(CudaMoeFragments, D5DeepSeekFullAttentionChain) {
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

TEST(CudaMoeFragments, D5DeepSeekStatefulCompression) {
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

TEST(CudaMoeFragments, D5DeepSeekNativeSinkAttention) {
  for (std::size_t index : {3u, 4u, 5u, 6u})
    for (int mode = 0; mode < 3; ++mode) {
      const auto &model = d5_moe_execution_cases::kCases[index];
      SCOPED_TRACE(model.profile->model_repository);
      SCOPED_TRACE(model.profile->variant);
      SCOPED_TRACE(mode);
      run_q24_semantic_fragment(d5_deepseek_remaining_fragments::sink_attention(model, mode));
    }
}

TEST(CudaMoeFragments, D5DeepSeekTokenTableRouting) {
  for (std::size_t index : {3u, 4u, 5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::hash_router(model, true));
  }
}

TEST(CudaMoeFragments, D5DeepSeekHashRoutingFeedsMixedWeightMoe) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (std::size_t index : {5u, 6u}) {
    const auto &model = d5_moe_execution_cases::kCases[index];
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::hash_moe(model, true));
  }
}

TEST(CudaMoeFragments, D5NativeHashRouterWarpTailsAndScoreBoundaries) {
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

TEST(CudaMoeFragments, D5ThirteenProfilesModelRouters) {
  for (const auto &model : d5_moe_execution_cases::kCases) {
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::router(model, true));
  }
}

TEST(CudaMoeFragments, D5ThirteenProfilesExpertActivation) {
  for (const auto &model : d5_moe_execution_cases::kCases) {
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::activation_core(model));
  }
}

TEST(CudaMoeFragments, D5GroupedFp8Batch) {
  d5_grouped_fp8_fragments::projection_batch([](auto fragment) {
    run_q24_semantic_fragment(std::move(fragment));
  });
}

TEST(CudaMoeFragments, D5GlmOfficialFp8FullMoe) {
  for (bool flash : {false, true}) {
    SCOPED_TRACE(flash);
    run_q24_semantic_fragment(d5_grouped_fp8_fragments::glm_moe(flash));
  }
}

TEST(CudaMoeFragments, D5DynamicFp8DenseBatch) {
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

TEST(CudaMoeFragments, D5GlmOfficialDynamicFp8SharedFfn) {
  for (bool flash : {false, true}) {
    SCOPED_TRACE(flash);
    for (auto m : {1, 3}) {
      SCOPED_TRACE(m);
      run_q24_semantic_fragment(d5_dynamic_fp8_fragments::glm_shared_ffn(m, flash));
    }
  }
}

TEST(CudaMoeFragments, D5OfficialFp8DenseNativeBatch) {
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

TEST(CudaMoeFragments, D5OfficialFp4ExpertBatch) {
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

TEST(CudaMoeFragments, D5OfficialKimiMixedStorageFullMoe) {
  run_q24_semantic_fragment(d5_moe_execution_fragments::official_kimi_moe());
}

TEST(CudaMoeFragments, D5EightGgufProfilesFullMoe) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_moe_execution_cases::kCases) {
    if (!model.gguf)
      continue;
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::gguf_moe(model, true));
  }
}

TEST(CudaMoeFragments, D5KdaProjectionRoleBatch) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &role : d5_kda_projection_cases::kCases) {
    SCOPED_TRACE(role.profile->model_repository);
    SCOPED_TRACE(role.profile->variant);
    SCOPED_TRACE(role.roles);
    run_q24_semantic_fragment(d5_moe_semantic_fragments::kda_projection(role));
  }
}

TEST(CudaMoeFragments, D5GlmFlashWeightedHyperConnection) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (!model.flash) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::hyper_connection(model));
  }
}

TEST(CudaMoeFragments, D5GlmDsaIndexerWithCache) {
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

TEST(CudaMoeFragments, D5GlmFlashPooledIndexerWithTail) {
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

TEST(CudaMoeFragments, D5GlmOfficialMixedRemaining) {
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

TEST(CudaMoeFragments, D5DeepSeekOfficialCompression) {
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

TEST(CudaMoeFragments, D5DeepSeekOfficialAttention) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    const auto &model = d5_moe_execution_fragments::cases::kCases[index];
    for (std::int64_t ratio : {0, 4, 128}) {
      SCOPED_TRACE(ratio);
      run_q24_semantic_fragment(d5_deepseek_official_attention::attention(model, ratio));
    }
  }
}

TEST(CudaMoeFragments, D5DeepSeekOfficialMtpAndFullHead) {
  for (int index : {3, 4}) {
    SCOPED_TRACE(index);
    d5_deepseek_official_attention::check_mtp_head(
        d5_moe_execution_fragments::cases::kCases[index], run_q24_semantic_fragment);
  }
}

TEST(CudaMoeFragments, D5DeepSeekOriginalFp4Projections) {
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

TEST(CudaMoeFragments, D5DeepSeekOfficialMoe) {
  for (int index : {3, 4})
    for (bool hash : {false, true}) {
      SCOPED_TRACE(index);
      SCOPED_TRACE(hash);
      run_q24_semantic_fragment(d5_deepseek_official_fragments::moe_chain(
          d5_moe_execution_fragments::cases::kCases[index], hash));
    }
}

TEST(CudaMoeFragments, D5DeepSeekExpertActivationRoundtrip) {
  for (std::int64_t width : {7168, 4096})
    for (std::int64_t rows : {1, 3, 33}) {
      SCOPED_TRACE(width);
      SCOPED_TRACE(rows);
      run_q24_semantic_fragment(d5_quantization_roundtrip::fragment(false, rows, width, 128));
    }
}

TEST(CudaMoeFragments, D5DeepSeekQuantizationRoundtrip) {
  for (bool fp4 : {false, true})
    for (std::int64_t rows : {1, 3, 33}) {
      SCOPED_TRACE(fp4);
      SCOPED_TRACE(rows);
      run_q24_semantic_fragment(d5_quantization_roundtrip::fragment(fp4, rows, fp4 ? 128 : 448));
    }
}

TEST(CudaMoeFragments, D5Fp8WeightPreparation) {
  for (bool bf16 : {false, true})
    for (const auto shape : {std::array<std::int64_t, 2>{129, 256}, {576, 6144}, {28672, 512}}) {
      SCOPED_TRACE(bf16);
      SCOPED_TRACE(shape[0]);
      run_q24_semantic_fragment(d5_fp8_weight_preparation::decode_weights(shape[0], shape[1], bf16));
    }
}

TEST(CudaMoeFragments, D5GlmFp8AccumulationBoundaries) {
  run_q24_semantic_fragment(d5_dynamic_fp8_fragments::model_accumulation_boundaries());
}

TEST(CudaMoeFragments, D5GlmDsaPreparedKvAndNextN) {
  namespace glm = d5_glm_remaining_fragments;
  const auto &model = glm::kCases[0];
  ASSERT_TRUE(model.official && !model.flash);
  run_q24_semantic_fragment(glm::mla(model));
  glm::check_mtp_head(model, run_q24_semantic_fragment);
}

TEST(CudaMoeFragments, D5GlmGgufDenseFfn) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::dense_ffn(model));
  }
}

TEST(CudaMoeFragments, D5GlmGgufNextNDecoderWithCarryAndFullHead) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    d5_glm_remaining_fragments::check_mtp_head(model, run_q24_semantic_fragment);
  }
}

TEST(CudaMoeFragments, D5GlmGgufDsaSelectionConsumesMlaCache) {
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

TEST(CudaMoeFragments, D5GlmLayerSpecificExpertFormats) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::mixed_expert_variants()) {
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.gate == d5_moe_execution_cases::Storage::kQ4K
                     ? "Q4_K/Q4_K/Q6_K" : "Q5_K/Q5_K/Q6_K");
    run_q24_semantic_fragment(d5_moe_execution_fragments::gguf_moe(model));
  }
}

TEST(CudaMoeFragments, D5GlmGgufMtpIndexCarryReset) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    d5_glm_remaining_fragments::check_index_carry(model, run_q24_semantic_fragment);
  }
}

TEST(CudaMoeFragments, D5GlmNextNInputBoundary) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::mtp_input(model));
  }
}

TEST(CudaMoeFragments, D5GlmGgufAbsorbedMlaWithCache) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_glm_remaining_fragments::kCases) {
    if (model.official) continue;
    SCOPED_TRACE(model.profile->model_name);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_glm_remaining_fragments::mla(model));
  }
}

TEST(CudaMoeFragments, D5KimiBfloatRoundingBoundaries) {
  run_q24_semantic_fragment(d5_kimi_remaining_fragments::bfloat_rounding_boundaries());
}

TEST(CudaMoeFragments, D5KimiDenseFfn) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::dense_ffn(profile));
  }
}

TEST(CudaMoeFragments, D5KimiAbsorbedMlaWithCache) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::mla(profile));
  }
}

TEST(CudaMoeFragments, D5KimiAttentionResidualBanking) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::attention_residual(profile, true));
  }
}

TEST(CudaMoeFragments, D5KimiAttentionResidualOrdinary) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (int i = 0; i < 3; ++i) {
    auto profile = d5_moe_semantic_fragments::kKdaProfiles[i];
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kimi_remaining_fragments::attention_residual(profile, false));
  }
}

TEST(CudaMoeFragments, D5SixProfilesFullKdaChainAndTwoStates) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (auto profile : d5_moe_semantic_fragments::kKdaProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(d5_kda_full_fragments::complete(profile));
  }
}

TEST(CudaMoeFragments, D5SixProfilesBoundedKdaActualStateFeedback) {
  namespace moe = d5_moe_semantic_fragments;
  for (auto profile : moe::kKdaProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(moe::kda(profile));
  }
}

TEST(CudaMoeFragments, D5KimiThreeProfilesSituActivation) {
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

TEST(CudaMoeFragments, D5SevenProfilesHyperConnectionCore) {
  namespace moe = d5_moe_semantic_fragments;
  for (const auto &profile : moe::kHyperProfiles) {
    SCOPED_TRACE(profile.profile->model_repository);
    SCOPED_TRACE(profile.profile->variant);
    run_q24_semantic_fragment(moe::hyper_connection(profile));
  }
}

TEST(CudaQwenRecurrent, D5NineProfilesUseCompleteGateAndActualStateFeedback) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  if (!d5_qwen35_fixture::root())
    GTEST_SKIP() << "HYLE_D5_QWEN35_FIXTURE is required for actual 2B parameters";
  for (auto profile : sem::kRecurrentProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(sem::gdn(profile, 8));
  }
}

TEST(CudaQwen38_24T, D5GgufAttentionGdnAndMtpUseRealRoleTypes) {
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

TEST(CudaQwen38_24T, D5GgufAttentionAndGdnProjectionRoleBatch) {
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

TEST(CudaQwen38_24T, D5OfficialAttentionAndGdnProjectionRoleBatch) {
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
  auto direct = qfull::make_direct_case(fixture.layout);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const auto workload = q::workload(*profile, c::WorkloadPhase::kDecode, 0x0a12);
  std::vector<hyle::mem::Buffer> buffers(direct.descriptions.size());
  std::vector<hyle::mem::BufferView> views(direct.descriptions.size());
  const auto allocate = [&](qfull::ValueId id) {
    const auto &description = direct.description(id);
    buffers[id.ordinal()] =
        hyle::mem::Buffer::create(resource,
                                  description.required_bytes().value(), 256)
            .value();
    views[id.ordinal()] = buffers[id.ordinal()].view();
  };
  allocate(direct.full.activation);
  const auto &constant_bytes = fixture.constants;
  for (std::size_t index = 0; index < direct.full.constants.size(); ++index) {
    const auto id = direct.full.constants[index];
    allocate(id);
    ASSERT_TRUE(views[id.ordinal()].write(constant_bytes[index]).ok());
  }
  fixture.constants = {};

  std::vector<DirectCall> calls;
  calls.reserve(direct.nodes.size());
  for (const auto &step : direct.nodes) {
    if (step.operation.kind() == c::OperationKind::kView) {
      ASSERT_EQ(step.inputs.size(), 1U);
      ASSERT_EQ(step.outputs.size(), 1U);
      ASSERT_TRUE(step.operation.get_if<c::View>()->kind() == c::ViewKind::kReshape ||
                step.operation.get_if<c::View>()->kind() == c::ViewKind::kPermute);
      views[step.outputs[0].ordinal()] = views[step.inputs[0].ordinal()];
      continue;
    }
    std::vector<c::TensorDesc> inputs, outputs;
    for (const auto id : step.inputs)
      inputs.push_back(direct.description(id));
    for (const auto id : step.outputs) {
      outputs.push_back(direct.description(id));
      allocate(id);
    }
    calls.push_back(make_call(context, resource, step.operation, inputs,
                              outputs, workload));
    expect_mature(calls.back(), workload);
  }

  for (const auto &expected : fixture.expected) {
    ASSERT_TRUE(views[direct.full.activation.ordinal()]
                    .write(std::as_bytes(std::span(expected.activation)))
                    .ok());
    std::size_t call_index = 0;
    for (const auto &step : direct.nodes) {
      if (step.operation.kind() == c::OperationKind::kView)
        continue;
      std::vector<c::ConstTensorView> inputs;
      std::vector<c::TensorView> outputs;
      for (const auto id : step.inputs)
        inputs.push_back(c::ConstTensorView::create(
                             direct.description(id), views[id.ordinal()])
                             .value());
      for (const auto id : step.outputs)
        outputs.push_back(c::TensorView::create(
                              direct.description(id), views[id.ordinal()])
                              .value());
      run(calls[call_index++], queue, inputs, outputs);
    }
    const auto ids = half_test::read<std::int32_t>(
        buffers[direct.full.outputs[0].ordinal()]);
    ASSERT_TRUE(std::ranges::equal(ids, expected.ids));
    for (std::size_t output = 1; output < direct.full.outputs.size(); ++output) {
      const auto actual = half_test::read<float>(
          buffers[direct.full.outputs[output].ordinal()]);
      const auto &oracle = output == 1 ? expected.routed
                           : output == 2 ? expected.combined
                           : output == 3 ? expected.shared
                                         : expected.full;
      moe::expect_output(profile, actual, oracle);
    }
  }
}

TEST(CudaQwen38_24T, D5FullRouterSharedExpertAndNearTieDirectReplay) {
  run_full_moe_profile(&q24::kOfficialProfile);
}

TEST(CudaQwen38_24T, D5GgufFullRouterSharedExpertAndNearTieReplay) {
  for (const auto *profile : {&d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
                               &d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile}) {
    SCOPED_TRACE(profile->variant);
    run_full_moe_profile(profile);
  }
}

TEST_P(CudaQwen38Profiles,
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
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 4);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array query_norm_inputs{query_desc, scale_desc};
  const std::array query_output{query_desc};
  auto query_norm = make_call(context, resource, c::Operation(c::RmsNorm{}),
                              query_norm_inputs, query_output, workload);
  const std::array key_norm_inputs{kv_desc, scale_desc};
  const std::array key_output{kv_desc};
  auto key_norm = make_call(context, resource, c::Operation(c::RmsNorm{}),
                            key_norm_inputs, key_output, workload);
  const c::RoPE rope{.rotary_dimensions = q::kRotaryDimensions,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kInterleaved};
  const std::array query_rope_inputs{query_desc, position_desc, frequency_desc};
  auto query_rope = make_call(context, resource, c::Operation(rope), query_rope_inputs,
                              query_output, workload);
  const std::array key_rope_inputs{kv_desc, position_desc, frequency_desc};
  auto key_rope = make_call(context, resource, c::Operation(rope), key_rope_inputs,
                            key_output, workload);
  const std::array attention_inputs{query_desc, kv_desc, kv_desc};
  auto attention = make_call(
      context, resource,
      c::Operation(c::ScaledDotProductAttention{
          .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
          .scale = 0.0625F,
          .causal = true,
          .accumulation_type = c::DataType::kFloat32}),
      attention_inputs, query_output, workload);
  const std::array unary_input{query_desc};
  auto sigmoid = make_call(context, resource, c::Operation(c::Unary{c::UnaryKind::kSigmoid}),
                           unary_input, query_output, workload);
  const std::array multiply_inputs{query_desc, query_desc};
  auto multiply = make_call(
      context, resource, c::Operation(c::Binary{c::BinaryKind::kMultiply}),
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
  auto query_scale_buffer = values_buffer<BF16>(resource, query_scale);
  auto key_scale_buffer = values_buffer<BF16>(resource, key_scale);
  auto position_buffer = values_buffer<std::int32_t>(resource, position);
  auto frequency_buffer = values_buffer<float>(resource, frequencies);
  auto query_buffer = output_buffer(resource, query_desc);
  auto key_buffer = output_buffer(resource, kv_desc);
  auto value_buffer = output_buffer(resource, kv_desc);
  auto gate_buffer = output_buffer(resource, query_desc);
  auto normalized_query_buffer = output_buffer(resource, query_desc);
  auto normalized_key_buffer = output_buffer(resource, kv_desc);
  auto rotated_query_buffer = output_buffer(resource, query_desc);
  auto rotated_key_buffer = output_buffer(resource, kv_desc);
  auto attention_buffer = output_buffer(resource, query_desc);
  auto sigmoid_buffer = output_buffer(resource, query_desc);
  auto result_buffer = output_buffer(resource, query_desc);
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

TEST_P(CudaQwen38Profiles,
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
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 0x0d0e);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const std::array concat_inputs{half_desc, half_desc};
  const std::array joined_output{joined_desc};
  auto concat = make_call(context, resource, c::Operation(c::Concat{.axis = 1}),
                          concat_inputs, joined_output, workload);
  const std::array projection_inputs{joined_desc, weight_desc};
  const std::array half_output{half_desc};
  auto projection = make_call(context, resource, c::Operation(c::MatMul{}),
                              projection_inputs, half_output, workload);
  const std::array norm_inputs{half_desc, scale_desc};
  auto norm = make_call(context, resource, c::Operation(c::RmsNorm{}), norm_inputs,
                        half_output, workload);
  const std::array cast_inputs{half_desc};
  const std::array cast_output{output_f32_desc};
  auto cast = make_call(context, resource, c::Operation(c::Cast{c::DataType::kFloat32}),
                        cast_inputs, cast_output, workload);
  for (const auto *call : {&concat, &projection, &norm, &cast})
    expect_mature(*call, workload);

  auto weight_values =
      q::sparse_output_major_weight(shape.hidden, (2 * shape.hidden), weight_seed);
  auto scale_values = q::scales(shape.hidden, 11);
  auto weight_buffer = values_buffer<BF16>(resource, weight_values);
  weight_values.clear();
  weight_values.shrink_to_fit();
  auto scale_buffer = values_buffer<BF16>(resource, scale_values);
  auto first_buffer = output_buffer(resource, half_desc);
  auto second_buffer = output_buffer(resource, half_desc);
  auto joined_buffer = output_buffer(resource, joined_desc);
  auto projected_buffer = output_buffer(resource, half_desc);
  auto normalized_buffer = output_buffer(resource, half_desc);
  auto converted_buffer = output_buffer(resource, output_f32_desc);
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
    const auto joined = q::values((2 * shape.hidden), seed);
    ASSERT_TRUE(first_buffer.view()
                    .write(std::as_bytes(std::span(joined).first(shape.hidden)))
                    .ok());
    ASSERT_TRUE(second_buffer.view()
                    .write(std::as_bytes(std::span(joined).last(shape.hidden)))
                    .ok());
    run(concat, queue, concat_views, joined_view);
    run(projection, queue, projection_views, projected_view);
    run(norm, queue, norm_views, normalized_view);
    run(cast, queue, cast_views, converted_view);
    const auto actual_projected = half_test::read<BF16>(projected_buffer);
    const auto actual_converted = half_test::read<float>(converted_buffer);
    const auto projected =
        q::sparse_projection(joined, shape.hidden, weight_seed);
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
    D5OfficialAndPrimaryGguf, CudaQwen38Profiles,
    ::testing::Values(&qcase::kOfficial27B, &qcase::kPrimaryGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, CudaQwen38Profiles,
    ::testing::Values(&qcase::kFidelityGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5Official24T, CudaQwen38Profiles,
    ::testing::Values(&qcase::kOfficial24T));

class CudaQwenOfficialVocabulary
    : public ::testing::TestWithParam<const vocab::ProfileCase *> {};

TEST_P(CudaQwenOfficialVocabulary, D5WholeVocabularyHeadStreamsAllRows) {
  const auto &shape = *GetParam();
  const auto activation_desc = dense(c::DataType::kBFloat16, {1, shape.hidden});
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 1);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  auto queue = context.create_queue().value();
  const auto make_head = [&](std::int64_t rows) {
    const auto stored = dense(c::DataType::kBFloat16, {rows, shape.hidden});
    const auto weight_desc = stored.transposed().value();
    const auto logits_desc = dense(c::DataType::kBFloat16, {1, rows});
    const std::array inputs{activation_desc, weight_desc};
    const std::array outputs{logits_desc};
    return make_call(context, resource, c::Operation(c::MatMul{}), inputs,
                     outputs, workload);
  };
  auto full = make_head(vocab::kShardRows);
  auto tail = make_head(vocab::kVocabularyRows % vocab::kShardRows);
  expect_mature(full, workload);
  expect_mature(tail, workload);

  const auto max_weight = dense(c::DataType::kBFloat16,
                                {vocab::kShardRows, shape.hidden});
  const auto max_output =
      dense(c::DataType::kBFloat16, {1, vocab::kShardRows});
  auto weight_buffer = output_buffer(resource, max_weight);
  auto logits_buffer = output_buffer(resource, max_output);
  const std::array activations{vocab::activation(shape.hidden, 3),
                               vocab::activation(shape.hidden, 17)};
  auto activation_buffer = values_buffer<BF16>(resource, activations[0]);
  std::array<std::vector<BF16>, 2> logical_logits{
      std::vector<BF16>(vocab::kVocabularyRows),
      std::vector<BF16>(vocab::kVocabularyRows)};

  std::int64_t visited = 0;
  for (const auto shard : vocab::shards()) {
    const auto stored = dense(c::DataType::kBFloat16,
                              {shard.rows, shape.hidden});
    const auto weight_desc = stored.transposed().value();
    const auto output_desc = dense(c::DataType::kBFloat16, {1, shard.rows});
    const auto weights = vocab::head_weight(shard, shape.hidden);
    const auto weight_bytes = stored.required_bytes().value();
    const auto output_bytes = output_desc.required_bytes().value();
    const auto weight_view = weight_buffer.view().subview(0, weight_bytes).value();
    const auto output_view = logits_buffer.view().subview(0, output_bytes).value();
    ASSERT_TRUE(weight_view.write(std::as_bytes(std::span(weights))).ok());
    const std::array inputs{
        c::ConstTensorView::create(activation_desc, activation_buffer.view())
            .value(),
        c::ConstTensorView::create(weight_desc, weight_view).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, output_view).value()};
    auto &call = shard.rows == vocab::kShardRows ? full : tail;
    for (std::size_t replay = 0; replay < activations.size(); ++replay) {
      ASSERT_TRUE(activation_buffer.view()
                      .write(std::as_bytes(std::span(activations[replay])))
                      .ok());
      run(call, queue, inputs, outputs);
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

TEST_P(CudaQwenOfficialVocabulary, D5WholeVocabularyGatherMapsGlobalRows) {
  const auto &shape = *GetParam();
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 1);
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  d5_qwen_vocabulary_compute_test::check_gather(
      context, memory.ref(), workload, shape.hidden,
      [&](const c::PreparedOperation &prepared) {
        const auto provider = prepared.provider().id;
        EXPECT_TRUE(provider == c::ggml_cuda::kProviderId ||
                    provider == c::cublas::kProviderId ||
                    provider == c::cuda_native::kSglangProviderId ||
                    provider == c::cuda_native::kCcclProviderId ||
                    d5_optional_cuda_provider::is_flashinfer(provider));
      });
}

INSTANTIATE_TEST_SUITE_P(
    D5Official2B27BAnd24T, CudaQwenOfficialVocabulary,
    ::testing::Values(&vocab::kOfficial2B, &vocab::kOfficial27B,
                      &vocab::kOfficial24T));

TEST(CudaIq1ModelContracts, EightCurrentExpertShapesUseOrdinaryDirectAndReplay) {
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
#include "../compute_models/cuda_n1_non_gguf_tests.h"
#include "../compute_models/cuda_n2_mlx_tests.h"
#include "../compute_models/cuda_r1_current_region_tests.h"
} // namespace
