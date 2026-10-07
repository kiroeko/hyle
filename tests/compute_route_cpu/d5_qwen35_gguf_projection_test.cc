#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
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
#include "hyle/compute/providers/native.h"

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
                prepared->provider().id == c::onednn::kProviderId ||
                prepared->provider().id == c::aten_cpu::kProviderId)
        << prepared->provider().name;
    EXPECT_EQ(prepared->workload(), workload);


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
      // MatMul declares F32 accumulation; all accepted providers must retain
      // that precision. A double-reference output-relative test is invalid
      // near cancellation. Use the existing expanded-GGUF F32 roundoff gate,
      // scaled by the actual row L1 norm and activation, plus one storage ULP.
      // The withdrawn ARM F16-accumulation exemption is deliberately absent.
      float maximum_activation = 0.0F;
      for (float value : decoded_activation)
        maximum_activation = std::max(maximum_activation, std::abs(value));
      for (std::size_t i = 0; i < actual.size(); ++i) {
        double row_l1 = 0.0;
        const auto offset = i * static_cast<std::size_t>(test.reduction);
        for (std::int64_t k = 0; k < test.reduction; ++k)
          row_l1 += std::abs(decoded_weights[offset + static_cast<std::size_t>(k)]);
        const auto encoded = half_test::encode<F16>(expected[i]);
        const double rounded = half_test::decode(encoded);
        const auto magnitude = static_cast<std::uint16_t>(encoded.bits & 0x7fffU);
        const double spacing = half_test::decode_bits<F16>(static_cast<std::uint16_t>(magnitude + 1U)) -
                               half_test::decode_bits<F16>(magnitude);
        const double roundoff = 16.0 * std::numeric_limits<float>::epsilon() *
                                maximum_activation * row_l1;
        EXPECT_NEAR(half_test::decode(actual[i]), rounded, roundoff + spacing);
      }
    }
  }
}

} // namespace
