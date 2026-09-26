#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_models/d5_qwen_gguf_vocabulary_cases.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;

c::TensorDesc dense(std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array{rows, columns})
      .value();
}

hyle::mem::Buffer buffer(std::span<const std::byte> bytes) {
  auto result =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), bytes.size(), 64)
          .value();
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

hyle::mem::Buffer output(const c::TensorDesc &description) {
  return hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                   description.required_bytes().value(), 64)
      .value();
}

template <std::size_t N>
void run_quantized_projections(
    const c::ModelArtifactProfileDescription &profile,
    const std::array<d5_gguf_test::Qwen35Projection, N> &cases) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = g::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};

  for (const auto &test : cases) {
    SCOPED_TRACE(test.tensor);
    const auto input_desc = dense(1, test.reduction);
    const auto weight_desc =
        c::gguf_weight_description(test.type, c::DataType::kFloat32,
                                   test.reduction, test.columns)
            .value();
    const auto output_desc = dense(1, test.columns);
    const auto physical =
        d5_gguf_test::values(test.columns, test.reduction, test.seed);
    const auto packed =
        d5_gguf_test::pack(test.type, physical, test.columns, test.reduction,
                           weight_desc.required_bytes().value());
    auto activation_values =
        d5_gguf_test::activation(test.reduction, test.seed);
    auto activation_buffer = half_test::buffer<float>(activation_values);
    auto weight_buffer = buffer(packed.bytes);
    auto result_buffer = output(output_desc);

    auto builder = g::GraphBuilder::create().value();
    const auto activation = builder.input(input_desc).value();
    const auto weight = builder.constant(weight_desc).value();
    const auto product = builder.add(c::MatMul{}, activation, weight).value();
    ASSERT_TRUE(builder.publish(product).ok());
    auto logical = std::move(builder).finish().value();
    const std::array constants{g::ConstantBinding{
        weight,
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()}};
    const auto workload = d5_qwen35_fixture::workload(
        profile, c::WorkloadPhase::kDecode, test.region);
    auto result = g::compile(std::move(logical), context,
                             {.constants = constants, .workload = workload});
    ASSERT_TRUE(result) << result.error().message();
    auto compiled = std::move(*result);
    EXPECT_EQ(compiled.executable.workload(), workload);
    ASSERT_EQ(compiled.report.step_count(), 1U);
    EXPECT_EQ(compiled.report.step(0)->provider.id, c::ggml_cpu::kProviderId);

    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation
                    .bind(activation, c::ConstTensorView::create(
                                          input_desc, activation_buffer.view())
                                          .value())
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(product, c::TensorView::create(output_desc,
                                                         result_buffer.view())
                                       .value())
                    .ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());

    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        for (auto &value : activation_values)
          value = value * -0.75F + 0.001F;
        ASSERT_TRUE(activation_buffer.view()
                        .write(std::as_bytes(std::span(activation_values)))
                        .ok());
      }
      auto completion = invocation.submit(queue);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<float>(result_buffer);
      const auto expected = d5_gguf_test::matmul(
          activation_values, packed.decoded, test.columns, test.reduction);
      d5_gguf_test::expect_quantized_matmul_close(actual, expected);
    }
  }
}

TEST(CpuDefaultGraph,
     D5Qwen35PrimaryGgufExactMixedProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(d5_qwen35_fixture::kPrimaryGgufProfile,
                            d5_gguf_test::kQwen35PrimaryProjectionCases);
}

TEST(CpuDefaultGraph,
     D5Qwen38PrimaryGgufEveryStoredProjectionTypeUsesNativeRegionsAndReplay) {
  run_quantized_projections(d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
                            d5_gguf_test::kQwen38PrimaryProjectionCases);
}

TEST(CpuDefaultGraph,
     D5Qwen38FidelityGgufQ8ProjectionRolesUseNativeRegionsAndReplay) {
  run_quantized_projections(
      d5_qwen38_gdn_fixture::kFidelityGgufProfile,
      d5_gguf_test::kQwen38FidelityQuantizedProjectionCases);
}

TEST(CpuDefaultGraph, D5Qwen38_24TGgufMoeWeightRolesUseNativeRegions) {
  run_quantized_projections(
      d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
      d5_gguf_test::kQwen38_24TPrimaryProjectionCases);
  run_quantized_projections(
      d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile,
      d5_gguf_test::kQwen38_24TFidelityProjectionCases);
}

TEST(CpuDefaultGraph,
     D5Qwen35FidelityGgufExactQ8ProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(
      d5_qwen35_fixture::kFidelityGgufProfile,
      d5_gguf_test::kQwen35FidelityQuantizedProjectionCases);
}

TEST(CpuDefaultGraph,
     D5Qwen35FidelityGgufF16ProjectionsUseMatureRegionsAndReplay) {
  using F16 = hyle::simd::float16_t;
  auto context = g::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};

  for (const auto &test : d5_gguf_test::kQwen35FidelityF16ProjectionCases) {
    SCOPED_TRACE(test.tensor);
    const auto input_desc =
        c::TensorDesc::dense(c::DataType::kFloat16,
                             std::array<std::int64_t, 2>{1, test.reduction})
            .value();
    const auto stored_weight_desc =
        c::TensorDesc::dense(
            c::DataType::kFloat16,
            std::array<std::int64_t, 2>{test.columns, test.reduction})
            .value();
    const auto weight_desc = stored_weight_desc.transposed().value();
    const auto output_desc =
        c::TensorDesc::dense(c::DataType::kFloat16,
                             std::array<std::int64_t, 2>{1, test.columns})
            .value();
    const auto weight_source =
        d5_gguf_test::values(test.columns, test.reduction, test.seed);
    std::vector<F16> weight_values(weight_source.size());
    std::vector<float> decoded_weights(weight_source.size());
    for (std::size_t i = 0; i < weight_values.size(); ++i) {
      weight_values[i] = half_test::encode<F16>(weight_source[i]);
      decoded_weights[i] =
          static_cast<float>(half_test::decode(weight_values[i]));
    }
    auto activation_source =
        d5_gguf_test::activation(test.reduction, test.seed);
    std::vector<F16> activation_values(activation_source.size());
    std::vector<float> decoded_activation(activation_source.size());
    const auto encode_activation = [&] {
      for (std::size_t i = 0; i < activation_values.size(); ++i) {
        activation_values[i] = half_test::encode<F16>(activation_source[i]);
        decoded_activation[i] =
            static_cast<float>(half_test::decode(activation_values[i]));
      }
    };
    encode_activation();
    auto activation_buffer = half_test::buffer<F16>(activation_values);
    auto weight_buffer = half_test::buffer<F16>(weight_values);
    auto result_buffer = output(output_desc);

    auto builder = g::GraphBuilder::create().value();
    const auto activation = builder.input(input_desc).value();
    const auto weight = builder.constant(weight_desc).value();
    const auto product = builder.add(c::MatMul{}, activation, weight).value();
    ASSERT_TRUE(builder.publish(product).ok());
    auto logical = std::move(builder).finish().value();
    const std::array constants{g::ConstantBinding{
        weight,
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()}};
    const auto workload =
        d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                    c::WorkloadPhase::kDecode, test.region);
    auto result = g::compile(std::move(logical), context,
                             {.constants = constants, .workload = workload});
    ASSERT_TRUE(result) << result.error().message();
    auto compiled = std::move(*result);
    EXPECT_EQ(compiled.executable.workload(), workload);
    ASSERT_EQ(compiled.report.step_count(), 1U);
    const auto provider = compiled.report.step(0)->provider.id;
    EXPECT_TRUE(provider == c::ggml_cpu::kProviderId ||
                provider == c::onednn::kProviderId)
        << compiled.report.step(0)->provider.name;
#if defined(__aarch64__) || defined(_M_ARM64)
    const bool native_half_accumulation =
        provider == c::ggml_cpu::kProviderId;
#else
    const bool native_half_accumulation = false;
#endif

    auto invocation = compiled.executable.create_invocation().value();
    ASSERT_TRUE(invocation
                    .bind(activation, c::ConstTensorView::create(
                                          input_desc, activation_buffer.view())
                                          .value())
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(product, c::TensorView::create(output_desc,
                                                         result_buffer.view())
                                       .value())
                    .ok());
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());

    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        for (auto &value : activation_source)
          value = value * -0.75F + 0.001F;
        encode_activation();
        ASSERT_TRUE(activation_buffer.view()
                        .write(std::as_bytes(std::span(activation_values)))
                        .ok());
      }
      auto completion = invocation.submit(queue);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<F16>(result_buffer);
      const auto expected = d5_gguf_test::matmul(
          decoded_activation, decoded_weights, test.columns, test.reduction);
      ASSERT_EQ(actual.size(), expected.size());
      if (native_half_accumulation) {
        double squared_error = 0.0;
        double squared_reference = 0.0;
        double max_error = 0.0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
          const double rounded = half_test::decode(
              half_test::encode<F16>(expected[i]));
          const double error = half_test::decode(actual[i]) - rounded;
          squared_error += error * error;
          squared_reference += rounded * rounded;
          max_error = std::max(max_error, std::abs(error));
        }
        EXPECT_LE(max_error, 2.5e-3);
        ASSERT_GT(squared_reference, 0.0);
        EXPECT_LE(std::sqrt(squared_error / squared_reference), 1.0e-3);
      } else {
        for (std::size_t i = 0; i < actual.size(); ++i)
          half_test::expect_near(actual[i], expected[i]);
      }
    }
  }
}

} // namespace
