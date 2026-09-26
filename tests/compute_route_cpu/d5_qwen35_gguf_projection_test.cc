#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_models/d5_qwen_gguf_vocabulary_cases.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/onednn.h"

namespace {
namespace c = hyle::compute;

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
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();

  for (const auto &test : cases) {
    SCOPED_TRACE(test.tensor);
    const auto input_desc = dense(1, test.reduction);
    const auto weight_desc =
        c::gguf_weight_description(test.type, c::DataType::kFloat32,
                                   test.reduction, test.columns)
            .value();
    const auto output_desc = dense(1, test.columns);
    const std::array input_descs{input_desc, weight_desc};
    const std::array output_descs{output_desc};
    const auto workload = d5_qwen35_fixture::workload(
        profile, c::WorkloadPhase::kDecode, test.region);
    auto prepared =
        c::prepare(context, c::Operation(c::MatMul{}), input_descs,
                   output_descs, hyle::mem::system_resource(), workload);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, c::ggml_cpu::kProviderId);
    EXPECT_EQ(prepared->workload(), workload);

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
    const std::array inputs{
        c::ConstTensorView::create(input_desc, activation_buffer.view())
            .value(),
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, result_buffer.view()).value()};
    auto invocation = prepared->create_invocation().value();

    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        for (auto &value : activation_values)
          value = value * -0.75F + 0.001F;
        ASSERT_TRUE(activation_buffer.view()
                        .write(std::as_bytes(std::span(activation_values)))
                        .ok());
      }
      auto completion = invocation.submit(queue, inputs, outputs);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<float>(result_buffer);
      const auto expected = d5_gguf_test::matmul(
          activation_values, packed.decoded, test.columns, test.reduction);
      d5_gguf_test::expect_quantized_matmul_close(actual, expected);
    }
  }
}

TEST(CpuDefaultDirect,
     D5Qwen35PrimaryGgufExactMixedProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(d5_qwen35_fixture::kPrimaryGgufProfile,
                            d5_gguf_test::kQwen35PrimaryProjectionCases);
}

TEST(CpuDefaultDirect,
     D5Qwen38PrimaryGgufEveryStoredProjectionTypeUsesNativeRegionsAndReplay) {
  run_quantized_projections(d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
                            d5_gguf_test::kQwen38PrimaryProjectionCases);
}

TEST(CpuDefaultDirect,
     D5Qwen38FidelityGgufQ8ProjectionRolesUseNativeRegionsAndReplay) {
  run_quantized_projections(
      d5_qwen38_gdn_fixture::kFidelityGgufProfile,
      d5_gguf_test::kQwen38FidelityQuantizedProjectionCases);
}

TEST(CpuDefaultDirect, D5Qwen38_24TGgufMoeWeightRolesUseNativeRegions) {
  run_quantized_projections(
      d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
      d5_gguf_test::kQwen38_24TPrimaryProjectionCases);
  run_quantized_projections(
      d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile,
      d5_gguf_test::kQwen38_24TFidelityProjectionCases);
}

TEST(CpuDefaultDirect,
     D5Qwen35FidelityGgufExactQ8ProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(
      d5_qwen35_fixture::kFidelityGgufProfile,
      d5_gguf_test::kQwen35FidelityQuantizedProjectionCases);
}

TEST(CpuDefaultDirect,
     D5Qwen35FidelityGgufF16ProjectionsUseMatureRegionsAndReplay) {
  using F16 = hyle::simd::float16_t;
  auto context = c::cpu::create_context().value();
  auto queue = context.create_queue().value();

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
    const std::array input_descs{input_desc, weight_desc};
    const std::array output_descs{output_desc};
    const auto workload =
        d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                    c::WorkloadPhase::kDecode, test.region);
    auto prepared =
        c::prepare(context, c::Operation(c::MatMul{}), input_descs,
                   output_descs, hyle::mem::system_resource(), workload);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_TRUE(prepared->provider().id == c::ggml_cpu::kProviderId ||
                prepared->provider().id == c::onednn::kProviderId)
        << prepared->provider().name;
    EXPECT_EQ(prepared->workload(), workload);
#if defined(__aarch64__) || defined(_M_ARM64)
    const bool native_half_accumulation =
        prepared->provider().id == c::ggml_cpu::kProviderId;
#else
    const bool native_half_accumulation = false;
#endif

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
    const std::array inputs{
        c::ConstTensorView::create(input_desc, activation_buffer.view())
            .value(),
        c::ConstTensorView::create(weight_desc, weight_buffer.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, result_buffer.view()).value()};
    auto invocation = prepared->create_invocation().value();

    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        for (auto &value : activation_source)
          value = value * -0.75F + 0.001F;
        encode_activation();
        ASSERT_TRUE(activation_buffer.view()
                        .write(std::as_bytes(std::span(activation_values)))
                        .ok());
      }
      auto completion = invocation.submit(queue, inputs, outputs);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      const auto actual = half_test::read<F16>(result_buffer);
      const auto expected = d5_gguf_test::matmul(
          decoded_activation, decoded_weights, test.columns, test.reduction);
      ASSERT_EQ(actual.size(), expected.size());
      if (native_half_accumulation) {
        // ggml's ARM NEON path accumulates these dot products in FP16. A
        // per-element relative check against FP32 accumulation misclassifies
        // cancellation near zero; bound both absolute and aggregate error.
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
