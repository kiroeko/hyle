#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mlx/ops.h>

#include "../compute_models/d5_moe_profile_cases.h"
#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "../compute_models/d5_qwen_gguf_vocabulary_cases.h"
#include "../compute_models/d5_qwen35_gguf_fragment_support.h"
#include "../compute_models/d5_qwen_gguf_vocabulary_apple_test.h"
#include "../compute_models/d5_qwen38_24t_fixture.h"
#include "../compute_models/d5_qwen38_24t_grouped_gguf_fixture.h"
#include "../compute_models/d5_moe_execution_fragments.h"
#include "../compute_models/d5_moe_semantic_fragments.h"
#include "../compute_models/d5_qwen38_24t_semantic_fragments.h"
#include "../compute_models/d5_qwen38_24t_full_router.h"
#include "../compute_models/d5_qwen38_24t_moe_profiles.h"
#include "../compute_models/d5_qwen38_profile_cases.h"
#include "../compute_models/d5_qwen_vocabulary_test.h"
#include "../compute_models/d5_qwen38_official_fixture.h"
#include "hyle/compute/apple.h"
#include "hyle/compute/model_ops.h"
#include "tensor_test_support.h"

namespace {
using namespace apple_tensor_test;
namespace a = c::apple;
namespace mx = mlx::core;
namespace q = d5_qwen38_official_fixture;
namespace q24 = d5_qwen38_24t_fixture;
namespace qfull = d5_qwen38_24t_full_router;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

std::vector<B16> decode_bfloat(std::span<const std::byte> bytes) {
  EXPECT_EQ(bytes.size() % sizeof(B16), 0U);
  std::vector<B16> result(bytes.size() / sizeof(B16));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

mx::Shape native_shape(const c::TensorDesc &description, bool stored) {
  mx::Shape result;
  const auto dimensions = description.dimensions();
  result.reserve(dimensions.size());
  if (stored && dimensions.size() == 2) {
    result.push_back(static_cast<int>(dimensions[1]));
    result.push_back(static_cast<int>(dimensions[0]));
  } else {
    for (const auto dimension : dimensions)
      result.push_back(static_cast<int>(dimension));
  }
  return result;
}

mx::array upload(const a::Context &, const c::TensorDesc &description,
                 std::span<const std::byte> bytes) {
  const bool transposed = !description.is_dense() && description.rank() == 2;
  const auto shape = native_shape(description, transposed);
  mx::array result = [&]() -> mx::array {
    switch (description.data_type()) {
    case c::DataType::kFloat16: {
      std::vector<std::uint16_t> data(bytes.size() / sizeof(std::uint16_t));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::view(mx::array(data.data(), shape, mx::uint16), mx::float16,
                      stream());
    }
    case c::DataType::kBFloat16: {
      std::vector<std::uint16_t> data(bytes.size() / sizeof(std::uint16_t));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::view(mx::array(data.data(), shape, mx::uint16), mx::bfloat16,
                      stream());
    }
    case c::DataType::kFloat32: {
      std::vector<float> data(bytes.size() / sizeof(float));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::array(data.data(), shape, mx::float32);
    }
    case c::DataType::kInt32: {
      std::vector<std::int32_t> data(bytes.size() / sizeof(std::int32_t));
      std::memcpy(data.data(), bytes.data(), bytes.size());
      return mx::array(data.data(), shape, mx::int32);
    }
    default:
      throw std::runtime_error("unsupported Qwen3.5 Apple fixture type");
    }
  }();
  if (transposed)
    result = mx::transpose(result, stream());
  return ready(std::move(result));
}

template <class T, std::size_t Extent>
mx::array upload(const a::Context &context, const c::TensorDesc &description,
                 std::span<T, Extent> values) {
  return upload(context, description,
                std::span<const std::byte>(std::as_bytes(values)));
}

mx::array example(const a::Context &context, const c::TensorDesc &description) {
  const std::vector<std::byte> zeros(*description.required_bytes());
  return upload(context, description, std::span<const std::byte>(zeros));
}

mx::array example(const a::Context &, const mx::array &value) { return value; }

mx::array packed_array(std::span<const std::byte> bytes) {
  std::vector<std::uint8_t> storage(bytes.size());
  std::memcpy(storage.data(), bytes.data(), bytes.size());
  return ready(mx::array(
      storage.data(), mx::Shape{static_cast<int>(storage.size())}, mx::uint8));
}

d5_gguf_test::Packed packed(c::GgufQuantizedType type, std::int64_t columns,
                            std::int64_t reduction,
                            const c::TensorDesc &description, int seed) {
  const auto source = d5_gguf_test::values(columns, reduction, seed);
  return d5_gguf_test::pack(type, source, columns, reduction,
                            description.required_bytes().value());
}

std::vector<float> float_values(const mx::array &array) {
  const auto source = values(array);
  std::vector<float> result(source.size());
  std::ranges::transform(source, result.begin(), [](double value) {
    return static_cast<float>(value);
  });
  return result;
}

struct Call final {
  a::Program program;
};

Call prepare(const a::Context &context, const c::WorkloadContext &workload,
             c::Operation operation, std::span<const mx::array> arguments) {
  auto program = context.prepare(operation, arguments, workload).value();
  EXPECT_EQ(program.workload(), workload);
  return {std::move(program)};
}

template <class... Arguments>
Call prepare(const a::Context &context, const c::WorkloadContext &workload,
             c::Operation operation, Arguments &&...arguments) {
  const std::array<mx::array, sizeof...(Arguments)> values{
      example(context, std::forward<Arguments>(arguments))...};
  return prepare(context, workload, std::move(operation),
                 std::span<const mx::array>(values));
}

std::vector<mx::array> run(Call &call, std::span<const mx::array> inputs) {
  auto invocation = call.program.create_invocation().value();
  const auto bound = invocation.bind(inputs);
  EXPECT_TRUE(bound.ok()) << bound.message();
  auto work = invocation.submit();
  EXPECT_TRUE(work) << work.error().message();
  if (!work)
    return {};
  auto result = work->wait();
  EXPECT_TRUE(result) << result.error().message();
  if (!result)
    return {};
  const auto outputs = result->get()->arrays();
  return {outputs.begin(), outputs.end()};
}

template <class... Inputs>
std::vector<mx::array> run(Call &call, const Inputs &...inputs) {
  const std::array<mx::array, sizeof...(Inputs)> values{inputs...};
  return run(call, std::span<const mx::array>(values));
}

void expect_qwen38_near(std::span<const double> actual,
                        std::span<const double> expected,
                        double relative = 0.012) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i],
                std::abs(expected[i]) * relative + 2.0e-5)
        << i;
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
  const std::array specs{a::input_spec(table_desc).value(),
                         a::input_spec(ids_desc).value()};
  const auto workload = d5_qwen35_fixture::workload(
      profile, c::WorkloadPhase::kDecode, region);
  auto context = a::Context::create().value();
  auto program =
      context.prepare(c::Operation(c::Gather{}), specs, workload).value();
  EXPECT_EQ(program.workload(), workload);

  const auto physical = d5_gguf_test::values(rows, hidden, seed);
  const auto packed = d5_gguf_test::pack(
      type, physical, rows, hidden, table_desc.required_bytes().value());
  const auto table = packed_array(packed.bytes);
  auto invocation = program.create_invocation().value();
  // ggml Metal's Q4_K get_rows path evaluates the high-nibble scale division
  // in half precision before publishing F32. The independent oracle decodes
  // the same packed bytes in F32, so bound that one provider conversion step.
  const auto tolerance =
      type == c::GgufQuantizedType::kQ4K ? 5.0e-4 : 1.0e-6;
  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 5}}, {{6, 2}}}};
  std::optional<mx::array> retained;
  for (const auto &selection : selections) {
    const auto ids = upload(context, ids_desc, std::span(selection));
    const std::array inputs{table, ids};
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_EQ(result->get()->arrays().size(), 1U);
    const auto &output = result->get()->arrays()[0];
    const auto actual = values(output);
    const auto expected =
        d5_gguf_test::selected_rows(packed.decoded, hidden, selection);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], tolerance) << i;
    retained = output;
  }

  ASSERT_TRUE(retained.has_value());
  const auto before_failure = bits(*retained);
  const std::array<std::int32_t, tokens> invalid{
      0, static_cast<std::int32_t>(rows)};
  const auto bad_ids = upload(context, ids_desc, std::span(invalid));
  const std::array bad_inputs{table, bad_ids};
  ASSERT_TRUE(invocation.bind(bad_inputs).ok());
  auto failed = invocation.submit();
  ASSERT_TRUE(failed) << failed.error().message();
  auto failure = failed->wait();
  ASSERT_FALSE(failure);
  EXPECT_TRUE(failure.error().IsOutOfRange());
  EXPECT_EQ(bits(*retained), before_failure);
}

TEST(AppleD5, D5Qwen35PrimaryGgufQuantizedEmbeddingUsesNativeRowsAndReplays) {
  run_quantized_embedding(
      d5_qwen35_fixture::kPrimaryGgufProfile, d5_qwen35_fixture::kHidden,
      d5_qwen35_fixture::kEmbeddingRows, c::GgufQuantizedType::kQ6K, 0x010f,
      37);
}

TEST(AppleD5, D5Qwen38PrimaryGgufQuantizedEmbeddingUsesExactWidthAndReplays) {
  run_quantized_embedding(
      d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
      d5_qwen38_official_fixture::kHidden,
      d5_qwen38_official_fixture::kEmbeddingRows,
      c::GgufQuantizedType::kQ4K, 0x010f, 19);
}

TEST(AppleD5,
     D5Qwen38FidelityGgufQuantizedEmbeddingUsesExactWidthAndReplays) {
  run_quantized_embedding(
      d5_qwen38_gdn_fixture::kFidelityGgufProfile,
      d5_qwen38_official_fixture::kHidden,
      d5_qwen38_official_fixture::kEmbeddingRows,
      c::GgufQuantizedType::kQ8_0, 0x0108, 23);
}

TEST(AppleD5, D5Qwen35FidelityGgufF16EmbeddingUsesMatureRowsAndReplays) {
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto rows = d5_qwen35_fixture::kEmbeddingRows;
  constexpr std::int64_t tokens = 2;
  const auto table_desc = dense(c::DataType::kFloat16, {rows, hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const std::array specs{a::input_spec(table_desc).value(),
                         a::input_spec(ids_desc).value()};
  const auto workload =
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x0108);
  auto context = a::Context::create().value();
  auto program =
      context.prepare(c::Operation(c::Gather{}), specs, workload).value();
  EXPECT_EQ(program.workload(), workload);

  const auto source = d5_gguf_test::values(rows, hidden, 61);
  std::vector<F16> table_values(source.size());
  std::vector<float> decoded(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    table_values[i] = half_test::encode<F16>(source[i]);
    decoded[i] = static_cast<float>(half_test::decode(table_values[i]));
  }
  const auto table = upload(context, table_desc, std::span(table_values));
  auto invocation = program.create_invocation().value();
  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 5}}, {{6, 2}}}};
  std::optional<mx::array> retained;
  for (const auto &selection : selections) {
    const auto ids = upload(context, ids_desc, std::span(selection));
    const std::array inputs{table, ids};
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_EQ(result->get()->arrays().size(), 1U);
    const auto &output = result->get()->arrays()[0];
    const auto actual = values(output);
    const auto expected =
        d5_gguf_test::selected_rows(decoded, hidden, selection);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1.0e-6) << i;
    retained = output;
  }

  ASSERT_TRUE(retained.has_value());
  const auto before_failure = bits(*retained);
  constexpr std::array<std::int32_t, tokens> invalid{0, rows};
  const auto bad_ids = upload(context, ids_desc, std::span(invalid));
  const std::array bad_inputs{table, bad_ids};
  ASSERT_TRUE(invocation.bind(bad_inputs).ok());
  auto failed = invocation.submit();
  ASSERT_TRUE(failed) << failed.error().message();
  auto failure = failed->wait();
  ASSERT_FALSE(failure);
  EXPECT_TRUE(failure.error().IsOutOfRange());
  EXPECT_EQ(bits(*retained), before_failure);
}

template <std::size_t N>
void run_quantized_projections(
    const c::ModelArtifactProfileDescription &profile,
    const std::array<d5_gguf_test::Qwen35Projection, N> &cases) {
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = a::Context::create().value();

  for (const auto &test : cases) {
    SCOPED_TRACE(test.tensor);
    const auto input_desc = dense(c::DataType::kFloat32, {1, test.reduction});
    const auto weight_desc =
        c::gguf_weight_description(test.type, c::DataType::kFloat32,
                                   test.reduction, test.columns)
            .value();
    const auto physical =
        d5_gguf_test::values(test.columns, test.reduction, test.seed);
    const auto packed =
        d5_gguf_test::pack(test.type, physical, test.columns, test.reduction,
                           weight_desc.required_bytes().value());
    const auto weight = packed_array(packed.bytes);
    const std::array specs{a::input_spec(input_desc).value(),
                           a::input_spec(weight_desc).value()};
    const auto workload =
        d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode,
                                    test.region);
    auto program =
        context.prepare(c::Operation(c::MatMul{}), specs, workload).value();
    EXPECT_EQ(program.workload(), workload);

    auto invocation = program.create_invocation().value();
    auto activation_values =
        d5_gguf_test::activation(test.reduction, test.seed);
    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        for (auto &value : activation_values)
          value = value * -0.75F + 0.001F;
      }
      const auto activation =
          upload(context, input_desc, std::span(activation_values));
      const std::array inputs{activation, weight};
      ASSERT_TRUE(invocation.bind(inputs).ok());
      auto result = invocation.submit().value().wait();
      ASSERT_TRUE(result) << result.error().message();
      ASSERT_EQ(result->get()->arrays().size(), 1U);
      const auto actual = values(result->get()->arrays()[0]);
      std::vector<float> actual_f32(actual.size());
      std::ranges::transform(actual, actual_f32.begin(), [](double value) {
        return static_cast<float>(value);
      });
      const auto expected = d5_gguf_test::matmul(
          activation_values, packed.decoded, test.columns, test.reduction);
      d5_gguf_test::expect_quantized_matmul_close(actual_f32, expected);
    }
  }
}

TEST(AppleD5,
     D5Qwen35PrimaryGgufExactMixedProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(d5_qwen35_fixture::kPrimaryGgufProfile,
                            d5_gguf_test::kQwen35PrimaryProjectionCases);
}

TEST(AppleD5,
     D5Qwen38PrimaryGgufExactMixedProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(d5_qwen38_gdn_fixture::kPrimaryGgufProfile,
                            d5_gguf_test::kQwen38PrimaryProjectionCases);
}

TEST(AppleD5,
     D5Qwen38FidelityGgufQ8ProjectionRolesUseNativeRegionsAndReplay) {
  run_quantized_projections(
      d5_qwen38_gdn_fixture::kFidelityGgufProfile,
      d5_gguf_test::kQwen38FidelityQuantizedProjectionCases);
}

TEST(AppleD5, D5Qwen38_24TGgufMoeWeightRolesUseNativeRegions) {
  run_quantized_projections(
      d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
      d5_gguf_test::kQwen38_24TPrimaryProjectionCases);
  run_quantized_projections(
      d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile,
      d5_gguf_test::kQwen38_24TFidelityProjectionCases);
}

TEST(AppleD5,
     D5Qwen35FidelityGgufExactQ8ProjectionsUseNativeRegionsAndReplay) {
  run_quantized_projections(
      d5_qwen35_fixture::kFidelityGgufProfile,
      d5_gguf_test::kQwen35FidelityQuantizedProjectionCases);
}

TEST(AppleD5, D5Qwen35FidelityGgufF16ProjectionsUseMatureRegionsAndReplay) {
  auto context = a::Context::create().value();
  for (const auto &test : d5_gguf_test::kQwen35FidelityF16ProjectionCases) {
    SCOPED_TRACE(test.tensor);
    const auto input_desc = dense(c::DataType::kFloat16, {1, test.reduction});
    const auto weight_desc =
        dense(c::DataType::kFloat16, {test.columns, test.reduction})
            .transposed()
            .value();
    const std::array specs{a::input_spec(input_desc).value(),
                           a::input_spec(weight_desc).value()};
    const auto workload =
        d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                    c::WorkloadPhase::kDecode, test.region);
    auto program =
        context.prepare(c::Operation(c::MatMul{}), specs, workload).value();
    EXPECT_EQ(program.workload(), workload);

    const auto weight_source =
        d5_gguf_test::values(test.columns, test.reduction, test.seed);
    std::vector<F16> weight_values(weight_source.size());
    std::vector<float> decoded_weights(weight_source.size());
    for (std::size_t i = 0; i < weight_values.size(); ++i) {
      weight_values[i] = half_test::encode<F16>(weight_source[i]);
      decoded_weights[i] =
          static_cast<float>(half_test::decode(weight_values[i]));
    }
    const auto weight = upload(context, weight_desc, std::span(weight_values));
    auto activation_source =
        d5_gguf_test::activation(test.reduction, test.seed);
    auto invocation = program.create_invocation().value();
    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0)
        for (auto &value : activation_source)
          value = value * -0.75F + 0.001F;
      std::vector<F16> activation_values(activation_source.size());
      std::vector<float> decoded_activation(activation_source.size());
      for (std::size_t i = 0; i < activation_values.size(); ++i) {
        activation_values[i] = half_test::encode<F16>(activation_source[i]);
        decoded_activation[i] =
            static_cast<float>(half_test::decode(activation_values[i]));
      }
      const auto activation =
          upload(context, input_desc, std::span(activation_values));
      const std::array inputs{activation, weight};
      ASSERT_TRUE(invocation.bind(inputs).ok());
      auto result = invocation.submit().value().wait();
      ASSERT_TRUE(result) << result.error().message();
      ASSERT_EQ(result->get()->arrays().size(), 1U);
      const auto expected_f32 = d5_gguf_test::matmul(
          decoded_activation, decoded_weights, test.columns, test.reduction);
      const std::vector<double> expected(expected_f32.begin(),
                                         expected_f32.end());
      expect(result->get()->arrays()[0], expected);
    }
  }
}

void run_qwen35_gguf_ffn(const c::ModelArtifactProfileDescription &profile,
                         c::GgufQuantizedType gate_type,
                         c::GgufQuantizedType down_type) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr std::int64_t hidden = d5_qwen35_fixture::kHidden;
  constexpr std::int64_t expanded = d5_qwen35_fixture::kFfnHidden;
  const auto activation_desc = dense(c::DataType::kFloat32, {1, hidden});
  const auto scale_desc = dense(c::DataType::kFloat32, {hidden});
  const auto gate_weight_desc =
      c::gguf_weight_description(gate_type, c::DataType::kFloat32, hidden,
                                 expanded)
          .value();
  const auto down_weight_desc =
      c::gguf_weight_description(down_type, c::DataType::kFloat32, expanded,
                                 hidden)
          .value();
  const auto expanded_desc = dense(c::DataType::kFloat32, {1, expanded});
  const auto workload =
      d5_qwen35_fixture::workload(profile, c::WorkloadPhase::kDecode, 0x020f);
  auto context = a::Context::create().value();

  const std::array norm_specs{a::input_spec(activation_desc).value(),
                              a::input_spec(scale_desc).value()};
  Call normalize{
      context.prepare(c::Operation(c::RmsNorm{}), norm_specs, workload)
          .value()};
  const std::array gate_specs{a::input_spec(activation_desc).value(),
                              a::input_spec(gate_weight_desc).value()};
  Call gate{
      context.prepare(c::Operation(c::MatMul{}), gate_specs, workload).value()};
  Call up{
      context.prepare(c::Operation(c::MatMul{}), gate_specs, workload).value()};
  const std::array gated_specs{a::input_spec(expanded_desc).value(),
                               a::input_spec(expanded_desc).value()};
  Call gated{context
                 .prepare(c::Operation(c::GatedActivation{
                              .inputs = c::GatedInputKind::kSeparate}),
                          gated_specs, workload)
                 .value()};
  const std::array down_specs{a::input_spec(expanded_desc).value(),
                              a::input_spec(down_weight_desc).value()};
  Call down{
      context.prepare(c::Operation(c::MatMul{}), down_specs, workload).value()};
  for (const auto *call : {&normalize, &gate, &up, &gated, &down})
    EXPECT_EQ(call->program.workload(), workload);

  const auto gate_packed =
      packed(gate_type, expanded, hidden, gate_weight_desc, 31);
  const auto up_packed =
      packed(gate_type, expanded, hidden, gate_weight_desc, 37);
  const auto down_packed =
      packed(down_type, hidden, expanded, down_weight_desc, 41);
  const auto gate_weight = packed_array(gate_packed.bytes);
  const auto up_weight = packed_array(up_packed.bytes);
  const auto down_weight = packed_array(down_packed.bytes);
  const auto scale_values = d5_qwen35_gguf_fragment::scale(hidden);
  const auto scale = upload(context, scale_desc, std::span(scale_values));
  auto activation_values = d5_gguf_test::activation(hidden, 43);
  std::vector<float> previous_final;
  for (int replay = 0; replay < 2; ++replay) {
    if (replay != 0)
      for (auto &value : activation_values)
        value = value * -0.625F + 0.00075F;
    const auto activation =
        upload(context, activation_desc, std::span(activation_values));
    const auto normalized = run(normalize, activation, scale);
    const auto normalized_actual = float_values(normalized[0]);
    const auto normalized_expected =
        d5_qwen35_gguf_fragment::rms_norm(activation_values, scale_values);
    d5_qwen35_gguf_fragment::expect_close(
        normalized_actual, normalized_expected, 3.0e-5F, 2.0e-5F);

    const auto gate_value = run(gate, normalized[0], gate_weight);
    const auto up_value = run(up, normalized[0], up_weight);
    const auto gate_actual = float_values(gate_value[0]);
    const auto up_actual = float_values(up_value[0]);
    const auto gate_expected = d5_gguf_test::matmul(
        normalized_actual, gate_packed.decoded, expanded, hidden);
    const auto up_expected = d5_gguf_test::matmul(
        normalized_actual, up_packed.decoded, expanded, hidden);
    d5_gguf_test::expect_quantized_chain_matmul_close(gate_actual,
                                                      gate_expected);
    d5_gguf_test::expect_quantized_chain_matmul_close(up_actual, up_expected);

    const auto product = run(gated, gate_value[0], up_value[0]);
    const auto gated_actual = float_values(product[0]);
    const auto gated_expected =
        d5_qwen35_gguf_fragment::silu_gate(gate_actual, up_actual);
    d5_qwen35_gguf_fragment::expect_close(gated_actual, gated_expected, 2.0e-5F,
                                          2.0e-5F);

    const auto output = run(down, product[0], down_weight);
    const auto final_actual = float_values(output[0]);
    const auto final_expected = d5_gguf_test::matmul(
        gated_actual, down_packed.decoded, hidden, expanded);
    d5_gguf_test::expect_quantized_chain_matmul_close(final_actual,
                                                      final_expected);
    if (!previous_final.empty())
      EXPECT_NE(final_actual, previous_final);
    previous_final = final_actual;
  }
}

TEST(AppleD5, D5Qwen35PrimaryGgufFfnUsesMixedNativeWeightsAndStableReplay) {
  run_qwen35_gguf_ffn(d5_qwen35_fixture::kPrimaryGgufProfile,
                      c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ6K);
}

TEST(AppleD5, D5Qwen35FidelityGgufFfnUsesQ8NativeWeightsAndStableReplay) {
  run_qwen35_gguf_ffn(d5_qwen35_fixture::kFidelityGgufProfile,
                      c::GgufQuantizedType::kQ8_0,
                      c::GgufQuantizedType::kQ8_0);
}

TEST(AppleD5, D5Qwen35OfficialEmbeddingNormHeadAndFailureUsePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  const auto fixture = d5_qwen35_fixture::load_embedding(*fixture_root);
  constexpr std::int64_t tokens = 2;
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto rows = d5_qwen35_fixture::kEmbeddingRows;
  const auto table_desc = dense(c::DataType::kBFloat16, {rows, hidden});
  const auto head_desc = table_desc.transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto token_desc = dense(c::DataType::kBFloat16, {tokens, hidden});
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0x01020f);
  auto context = a::Context::create().value();
  const auto table = upload(context, table_desc, fixture.rows);
  const auto head = upload(context, head_desc, fixture.rows);
  const auto scale = upload(context, scale_desc, fixture.scale);
  auto gather =
      prepare(context, workload, c::Operation(c::Gather{}), table, ids_desc);
  auto normalize =
      prepare(context, workload, c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
              token_desc, scale);
  auto residual =
      prepare(context, workload, c::Operation(c::Binary{c::BinaryKind::kAdd}),
              token_desc, token_desc);
  auto lm_head =
      prepare(context, workload, c::Operation(c::MatMul{}), token_desc, head);

  const auto table_values = decode_bfloat(fixture.rows);
  const auto scale_values = decode_bfloat(fixture.scale);
  const std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 1}}, {{5, 6}}}};
  std::optional<mx::array> retained_embedding;
  for (const auto &selected : selections) {
    const auto ids = upload(context, ids_desc, std::span(selected));
    auto embedded = run(gather, table, ids);
    ASSERT_EQ(embedded.size(), 1U);
    auto normalized = run(normalize, embedded[0], scale);
    ASSERT_EQ(normalized.size(), 1U);
    auto combined = run(residual, embedded[0], normalized[0]);
    ASSERT_EQ(combined.size(), 1U);
    auto logits = run(lm_head, combined[0], head);
    ASSERT_EQ(logits.size(), 1U);
    const auto actual_residual = values(combined[0]);
    const auto actual_logits = values(logits[0]);
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
        const auto normalized_value =
            rounded(value * inverse * half_test::decode(scale_values[column]),
                    mx::bfloat16);
        expected_row[column] = rounded(value + normalized_value, mx::bfloat16);
        EXPECT_EQ(actual_residual[token * hidden + column],
                  expected_row[column]);
      }
      for (std::size_t row = 0; row < rows; ++row) {
        double expected = 0;
        for (std::size_t column = 0; column < hidden; ++column)
          expected += expected_row[column] *
                      half_test::decode(table_values[row * hidden + column]);
        EXPECT_NEAR(actual_logits[token * rows + row], expected,
                    std::abs(expected) * 0.008 + 2.0e-7);
      }
    }
    retained_embedding = embedded[0];
  }

  ASSERT_TRUE(retained_embedding.has_value());
  const auto before_failure = bits(*retained_embedding);
  const std::array<std::int32_t, tokens> invalid{0, 7};
  const auto bad_ids = upload(context, ids_desc, std::span(invalid));
  auto failed_invocation = gather.program.create_invocation().value();
  const std::array failed_inputs{table, bad_ids};
  ASSERT_TRUE(failed_invocation.bind(failed_inputs).ok());
  auto failed = failed_invocation.submit();
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().has_value());
  EXPECT_EQ(bits(*retained_embedding), before_failure);
}

TEST(AppleD5, D5Qwen35OfficialFfnUsesMaturePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  const auto fixture = d5_qwen35_fixture::load_ffn(*fixture_root);
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto ffn = d5_qwen35_fixture::kFfnHidden;
  const auto activation_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto gate_desc =
      dense(c::DataType::kBFloat16, {ffn, hidden}).transposed().value();
  const auto down_desc =
      dense(c::DataType::kBFloat16, {hidden, ffn}).transposed().value();
  const auto expanded_desc = dense(c::DataType::kBFloat16, {1, ffn});
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 2);
  auto context = a::Context::create().value();
  const auto scale = upload(context, scale_desc, fixture.norm_scale);
  const auto gate_weight = upload(context, gate_desc, fixture.weights[0]);
  const auto up_weight = upload(context, gate_desc, fixture.weights[1]);
  const auto down_weight = upload(context, down_desc, fixture.weights[2]);
  auto normalize = prepare(context, workload, c::Operation(c::RmsNorm{}),
                           activation_desc, scale);
  auto gate = prepare(context, workload, c::Operation(c::MatMul{}),
                      activation_desc, gate_weight);
  auto up = prepare(context, workload, c::Operation(c::MatMul{}),
                    activation_desc, up_weight);
  auto gated = prepare(
      context, workload,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      expanded_desc, expanded_desc);
  auto down = prepare(context, workload, c::Operation(c::MatMul{}),
                      expanded_desc, down_weight);
  for (std::size_t replay = 0; replay < fixture.activations.size(); ++replay) {
    const auto activation =
        upload(context, activation_desc, fixture.activations[replay]);
    const auto normalized = run(normalize, activation, scale);
    const auto gate_value = run(gate, normalized[0], gate_weight);
    const auto up_value = run(up, normalized[0], up_weight);
    const auto product = run(gated, gate_value[0], up_value[0]);
    const auto output = run(down, product[0], down_weight);
    const auto actual = values(output[0]);
    ASSERT_EQ(actual.size(), fixture.direct_expected[replay].size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto expected = fixture.direct_expected[replay][i];
      EXPECT_NEAR(actual[i], expected, std::abs(expected) * 0.008 + 3.0e-8)
          << replay << ':' << i;
    }
  }
}

TEST(AppleD5, D5Qwen35OfficialGqaNormRopeAndAttentionUsePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  const auto fixture = d5_qwen35_fixture::load_attention_norms(*fixture_root);
  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query_desc =
      dense(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv_desc = dense(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto scale_desc = dense(c::DataType::kBFloat16, {depth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc = dense(c::DataType::kFloat32, {depth / 2});
  const std::array workloads{
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 4),
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kPrimaryGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x040f),
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x040f)};
  auto context = a::Context::create().value();
  const auto query_scale = upload(context, scale_desc, fixture.query);
  const auto key_scale = upload(context, scale_desc, fixture.key);
  const std::array<std::int32_t, 1> positions{3};
  std::vector<float> frequencies(depth / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / static_cast<float>(i + 1);
  const auto position = upload(context, position_desc, std::span(positions));
  const auto frequency =
      upload(context, frequency_desc, std::span<const float>(frequencies));
  for (const auto &workload : workloads) {
    auto query_norm = prepare(context, workload, c::Operation(c::RmsNorm{}),
                              query_desc, query_scale);
    auto key_norm = prepare(context, workload, c::Operation(c::RmsNorm{}),
                            kv_desc, key_scale);
    const c::RoPE rope{.rotary_dimensions = depth,
                       .sequence_axis = 1,
                       .layout = c::RopeLayout::kHalfSplit};
    auto query_rope = prepare(context, workload, c::Operation(rope), query_desc,
                              position, frequency);
    auto key_rope = prepare(context, workload, c::Operation(rope), kv_desc,
                            position, frequency);
    auto attention =
        prepare(context, workload,
                c::Operation(c::ScaledDotProductAttention{
                    .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
                    .scale = 0.0625F,
                    .causal = true,
                    .accumulation_type = c::DataType::kFloat32}),
                query_desc, kv_desc, kv_desc);
    std::vector<B16> queries(query_desc.element_count().value());
    std::vector<B16> keys(kv_desc.element_count().value());
    std::vector<B16> value_data(kv_desc.element_count().value());
    for (int seed : {1, 7}) {
      for (std::size_t i = 0; i < queries.size(); ++i)
        queries[i] = half_test::encode<B16>(
            static_cast<double>(static_cast<int>((i + seed) % 31) - 15) / 64);
      for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = half_test::encode<B16>(
            static_cast<double>(static_cast<int>((i + seed) % 29) - 14) / 64);
        value_data[i] = half_test::encode<B16>(
            static_cast<double>(seed * 100 + static_cast<int>(i)) / 256);
      }
      const auto query = upload(context, query_desc, std::span(queries));
      const auto key = upload(context, kv_desc, std::span(keys));
      const auto value = upload(context, kv_desc, std::span(value_data));
      const auto normalized_query = run(query_norm, query, query_scale);
      const auto normalized_key = run(key_norm, key, key_scale);
      const auto rotated_query =
          run(query_rope, normalized_query[0], position, frequency);
      const auto rotated_key =
          run(key_rope, normalized_key[0], position, frequency);
      const auto output =
          run(attention, rotated_query[0], rotated_key[0], value);
      const auto actual = bits(output[0]);
      for (std::size_t head = 0; head < query_heads; ++head)
        for (std::size_t column = 0; column < depth; ++column)
          EXPECT_EQ(
              actual[head * depth + column],
              value_data[(head / (query_heads / kv_heads)) * depth + column]
                  .bits);
    }
  }
}

TEST(AppleD5, D5Qwen35OfficialCausalConvAndGdnUsePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  const auto fixture = d5_qwen35_fixture::load_recurrent(*fixture_root);
  constexpr auto heads = d5_qwen35_fixture::kGdnHeads;
  constexpr auto depth = d5_qwen35_fixture::kGdnDepth;
  constexpr auto channels = d5_qwen35_fixture::kConvChannels;
  constexpr auto kernel = d5_qwen35_fixture::kConvKernel;
  const auto conv_samples_desc =
      dense(c::DataType::kFloat32, {1, kernel, channels});
  const auto conv_weight_desc =
      dense(c::DataType::kFloat32, {channels, kernel});
  const auto qkv_desc = dense(c::DataType::kFloat32, {1, 1, heads, depth});
  const auto gate_desc = dense(c::DataType::kFloat32, {1, 1, heads, 1});
  const auto state_desc =
      dense(c::DataType::kFloat32, {1, heads, depth, depth});
  const auto conv_weight_bf16 = decode_bfloat(fixture.convolution_weight);
  std::vector<float> conv_weights(conv_weight_bf16.size());
  for (std::size_t i = 0; i < conv_weights.size(); ++i)
    conv_weights[i] =
        static_cast<float>(half_test::decode(conv_weight_bf16[i]));
  const auto time_bias_bf16 = decode_bfloat(fixture.time_bias);
  std::vector<float> log_decay(heads), beta(heads);
  for (std::size_t head = 0; head < heads; ++head) {
    const auto bias = half_test::decode(time_bias_bf16[head]);
    const auto softplus = bias > 0 ? bias + std::log1p(std::exp(-bias))
                                   : std::log1p(std::exp(bias));
    log_decay[head] = static_cast<float>(
        -std::exp(static_cast<double>(fixture.log_decay[head])) * softplus);
    beta[head] = static_cast<float>(1.0 / (1.0 + std::exp(-bias)));
  }
  const std::array workloads{
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 5),
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kPrimaryGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x050f),
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kFidelityGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x050f)};
  auto context = a::Context::create().value();
  const auto conv_weight =
      upload(context, conv_weight_desc, std::span(conv_weights));
  const auto decay = upload(context, gate_desc, std::span(log_decay));
  const auto update = upload(context, gate_desc, std::span(beta));
  for (const auto &workload : workloads) {
    auto convolution = prepare(context, workload, c::model_ops::causal_conv1d(),
                               conv_samples_desc, conv_weight);
    auto gdn = prepare(context, workload, c::Operation(c::GatedDeltaNet{}),
                       qkv_desc, qkv_desc, qkv_desc, decay, update, state_desc);
    for (int seed : {2, 9}) {
      std::vector<float> conv_samples(
          conv_samples_desc.element_count().value());
      std::array<std::vector<float>, 3> qkv_values;
      for (auto &port : qkv_values)
        port.resize(qkv_desc.element_count().value());
      std::vector<float> state_values(state_desc.element_count().value());
      for (std::size_t i = 0; i < conv_samples.size(); ++i)
        conv_samples[i] = static_cast<float>(
                              static_cast<int>((i * 17 + seed * 5) % 61) - 30) /
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
      const auto samples =
          upload(context, conv_samples_desc, std::span(conv_samples));
      std::array qkv{
          upload(context, qkv_desc, std::span(qkv_values[0])),
          upload(context, qkv_desc, std::span(qkv_values[1])),
          upload(context, qkv_desc, std::span(qkv_values[2])),
      };
      const auto state = upload(context, state_desc, std::span(state_values));
      const auto conv_output = run(convolution, samples, conv_weight);
      const auto gdn_output =
          run(gdn, qkv[0], qkv[1], qkv[2], decay, update, state);
      ASSERT_EQ(gdn_output.size(), 2U);
      const auto actual_conv = values(conv_output[0]);
      const auto actual_scores = values(gdn_output[0]);
      const auto actual_state = values(gdn_output[1]);
      for (std::size_t channel = 0; channel < channels; ++channel) {
        double expected = 0;
        for (std::size_t tap = 0; tap < kernel; ++tap)
          expected += conv_samples[tap * channels + channel] *
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
}

TEST(AppleD5, D5Qwen35OfficialMtpAndBfloatBoundaryUsePreparedCalls) {
  const auto fixture_root = d5_qwen35_fixture::root();
  if (!fixture_root)
    GTEST_SKIP() << "set HYLE_D5_QWEN35_FIXTURE to the prepared local fixture";
  const auto fixture = d5_qwen35_fixture::load_mtp(*fixture_root);
  const auto embedding = d5_qwen35_fixture::load_embedding(*fixture_root);
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto input = d5_qwen35_fixture::kMtpInput;
  const auto half_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto joined_desc = dense(c::DataType::kBFloat16, {1, input});
  const auto weight_desc =
      dense(c::DataType::kBFloat16, {hidden, input}).transposed().value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {hidden});
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0x0d0e);
  auto context = a::Context::create().value();
  const auto weight = upload(context, weight_desc, fixture.weight);
  const auto scale = upload(context, scale_desc, embedding.scale);
  auto concat = prepare(context, workload, c::Operation(c::Concat{.axis = 1}),
                        half_desc, half_desc);
  auto projection = prepare(context, workload, c::Operation(c::MatMul{}),
                            joined_desc, weight);
  auto normalize =
      prepare(context, workload, c::Operation(c::RmsNorm{}), half_desc, scale);
  auto cast = prepare(context, workload,
                      c::Operation(c::Cast{c::DataType::kFloat32}), half_desc);
  const auto scales = decode_bfloat(embedding.scale);
  for (std::size_t replay = 0; replay < fixture.activations.size(); ++replay) {
    const auto &activation = fixture.activations[replay];
    const auto split = static_cast<std::size_t>(hidden * sizeof(B16));
    const auto first =
        upload(context, half_desc,
               std::span<const std::byte>(activation.data(), split));
    const auto second =
        upload(context, half_desc,
               std::span<const std::byte>(activation.data() + split,
                                          activation.size() - split));
    const auto joined = run(concat, first, second);
    const auto projected = run(projection, joined[0], weight);
    const auto normalized = run(normalize, projected[0], scale);
    const auto converted = run(cast, normalized[0]);
    const auto projected_values = values(projected[0]);
    const auto converted_values = values(converted[0]);
    double squares = 0;
    for (std::size_t i = 0; i < projected_values.size(); ++i) {
      EXPECT_NEAR(projected_values[i], fixture.expected[replay][i],
                  std::abs(fixture.expected[replay][i]) * 0.008 + 2.0e-7);
      squares += projected_values[i] * projected_values[i];
    }
    const auto inverse = 1.0 / std::sqrt(squares / hidden + 1.0e-5);
    for (std::size_t i = 0; i < converted_values.size(); ++i) {
      const auto expected =
          rounded(projected_values[i] * inverse * half_test::decode(scales[i]),
                  mx::bfloat16);
      EXPECT_NEAR(converted_values[i], expected, 1.0e-6) << i;
    }
  }
}

class AppleQwen38Profiles
    : public ::testing::TestWithParam<
          const d5_qwen38_profile_cases::ProfileCase *> {};

TEST_P(AppleQwen38Profiles,
       D5Qwen38ProfileBoundaryProjectionUsesExactHiddenAndReplays) {
  constexpr std::int64_t tokens = 2;
  constexpr int projection_seed = 31;
  const auto table_desc =
      dense(c::DataType::kBFloat16, {q::kEmbeddingRows, GetParam()->hidden});
  const auto head_desc = table_desc.transposed().value();
  const auto projection_desc =
      dense(c::DataType::kBFloat16, {GetParam()->hidden, GetParam()->hidden})
          .transposed()
          .value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {GetParam()->hidden});
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const auto token_desc =
      dense(c::DataType::kBFloat16, {tokens, GetParam()->hidden});
  const auto workload =
      q::workload(*GetParam()->profile, c::WorkloadPhase::kDecode, 0x01030e0f);
  auto context = a::Context::create().value();
  const auto table_values = q::values(q::kEmbeddingRows * GetParam()->hidden, 5);
  const auto head_values =
      q::values(q::kEmbeddingRows * GetParam()->hidden, 7, 96.0);
  const auto scale_values = q::scales(GetParam()->hidden, 3);
  auto projection_values = q::sparse_output_major_weight(
      GetParam()->hidden, GetParam()->hidden, projection_seed);
  const auto table = upload(context, table_desc, std::span(table_values));
  const auto head = upload(context, head_desc, std::span(head_values));
  const auto scale = upload(context, scale_desc, std::span(scale_values));
  const auto projection_weight =
      upload(context, projection_desc, std::span(projection_values));
  projection_values.clear();
  projection_values.shrink_to_fit();
  auto gather =
      prepare(context, workload, c::Operation(c::Gather{}), table, ids_desc);
  auto normalize =
      prepare(context, workload, c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
              token_desc, scale);
  auto residual =
      prepare(context, workload, c::Operation(c::Binary{c::BinaryKind::kAdd}),
              token_desc, token_desc);
  auto projection = prepare(context, workload, c::Operation(c::MatMul{}),
                            token_desc, projection_weight);
  auto lm_head = prepare(context, workload, c::Operation(c::MatMul{}),
                         token_desc, head);

  constexpr std::array<std::array<std::int32_t, tokens>, 2> selections{
      {{{0, 4}}, {{6, 2}}}};
  std::optional<mx::array> retained_embedding;
  for (const auto &selected : selections) {
    const auto ids = upload(context, ids_desc, std::span(selected));
    const auto embedded = run(gather, table, ids);
    const auto normalized = run(normalize, embedded[0], scale);
    const auto combined = run(residual, embedded[0], normalized[0]);
    const auto projected = run(projection, combined[0], projection_weight);
    const auto logits = run(lm_head, projected[0], head);
    const auto actual_residual = values(combined[0]);
    const auto actual_projected = values(projected[0]);
    const auto actual_logits = values(logits[0]);
    for (std::size_t token = 0; token < tokens; ++token) {
      const auto row_index = static_cast<std::size_t>(selected[token]);
      const auto row = std::span<const B16>(table_values)
                           .subspan(row_index * GetParam()->hidden, GetParam()->hidden);
      const auto normalized_expected =
          q::rms_norm(row, scale_values, 1.0e-6);
      std::vector<B16> residual_values(GetParam()->hidden);
      std::vector<double> residual_expected(GetParam()->hidden);
      for (std::size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] =
            q::encode(q::decode(row[i]) + normalized_expected[i]);
        residual_expected[i] = q::decode(residual_values[i]);
      }
      expect_qwen38_near(
          std::span(actual_residual).subspan(token * GetParam()->hidden, GetParam()->hidden),
          residual_expected);
      const auto projected_expected = q::sparse_projection(
          residual_values, GetParam()->hidden, projection_seed);
      expect_qwen38_near(
          std::span(actual_projected)
              .subspan(token * GetParam()->hidden, GetParam()->hidden),
          projected_expected);
      const auto logits_expected = q::dense_head(
          q::as_bf16(projected_expected), head_values, q::kEmbeddingRows);
      expect_qwen38_near(
          std::span(actual_logits)
              .subspan(token * q::kEmbeddingRows, q::kEmbeddingRows),
          logits_expected, 0.016);
    }
    retained_embedding = embedded[0];
  }

  ASSERT_TRUE(retained_embedding.has_value());
  const auto before_failure = bits(*retained_embedding);
  const std::array<std::int32_t, tokens> invalid{0, q::kEmbeddingRows};
  const auto bad_ids = upload(context, ids_desc, std::span(invalid));
  auto failed_invocation = gather.program.create_invocation().value();
  const std::array failed_inputs{table, bad_ids};
  ASSERT_TRUE(failed_invocation.bind(failed_inputs).ok());
  auto failed = failed_invocation.submit();
  ASSERT_TRUE(failed) << failed.error().message();
  EXPECT_FALSE(failed->wait().has_value());
  EXPECT_EQ(bits(*retained_embedding), before_failure);
}

TEST_P(AppleQwen38Profiles,
       D5Qwen38ProfileFfnUsesExactInnerWidthAndReplays) {
  constexpr int seed = 41;
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, GetParam()->hidden});
  const auto scale_desc = dense(c::DataType::kBFloat16, {GetParam()->hidden});
  const auto gate_desc =
      dense(c::DataType::kBFloat16, {GetParam()->ffn_hidden, GetParam()->hidden})
          .transposed()
          .value();
  const auto down_desc =
      dense(c::DataType::kBFloat16, {GetParam()->hidden, GetParam()->ffn_hidden})
          .transposed()
          .value();
  const auto expanded_desc =
      dense(c::DataType::kBFloat16, {1, GetParam()->ffn_hidden});
  const auto workload =
      q::workload(*GetParam()->profile, c::WorkloadPhase::kDecode, 0x0203);
  auto context = a::Context::create().value();
  const auto scale_values = q::scales(GetParam()->hidden, 9);
  const auto scale = upload(context, scale_desc, std::span(scale_values));
  auto gate_values =
      q::sparse_output_major_weight(GetParam()->ffn_hidden, GetParam()->hidden, seed);
  const auto gate_weight = upload(context, gate_desc, std::span(gate_values));
  gate_values.clear();
  gate_values.shrink_to_fit();
  auto up_values =
      q::sparse_output_major_weight(GetParam()->ffn_hidden, GetParam()->hidden, seed + 1);
  const auto up_weight = upload(context, gate_desc, std::span(up_values));
  up_values.clear();
  up_values.shrink_to_fit();
  auto down_values =
      q::sparse_output_major_weight(GetParam()->hidden, GetParam()->ffn_hidden, seed + 2);
  const auto down_weight = upload(context, down_desc, std::span(down_values));
  down_values.clear();
  down_values.shrink_to_fit();
  auto normalize = prepare(context, workload, c::Operation(c::RmsNorm{}),
                           activation_desc, scale);
  auto gate = prepare(context, workload, c::Operation(c::MatMul{}),
                      activation_desc, gate_weight);
  auto up = prepare(context, workload, c::Operation(c::MatMul{}),
                    activation_desc, up_weight);
  auto gated = prepare(
      context, workload,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      expanded_desc, expanded_desc);
  auto down = prepare(context, workload, c::Operation(c::MatMul{}),
                      expanded_desc, down_weight);
  for (int replay_seed : {3, 17}) {
    const auto input = q::values(GetParam()->hidden, replay_seed);
    const auto activation = upload(context, activation_desc, std::span(input));
    const auto normalized = run(normalize, activation, scale);
    const auto gate_value = run(gate, normalized[0], gate_weight);
    const auto up_value = run(up, normalized[0], up_weight);
    const auto product = run(gated, gate_value[0], up_value[0]);
    const auto output = run(down, product[0], down_weight);
    expect_qwen38_near(values(output[0]), q::ffn(input, scale_values, seed,
                                                    GetParam()->hidden,
                                                    GetParam()->ffn_hidden),
                       0.025);
  }
}

TEST(AppleQwen38_24T, D5GgufGroupedExpertKeepsNativeBlocksAndDynamicIds) {
  namespace grouped = d5_qwen38_24t_grouped_gguf_fixture;
  d5_gguf_test::QuantizerCleanup cleanup;
  auto context = a::Context::create().value();
  const auto activation_desc = dense(c::DataType::kFloat32,
      {1, grouped::kSlots, grouped::kReduction});
  const auto ids_desc = dense(c::DataType::kInt32, {1, grouped::kSlots});
  for (const auto &test : grouped::kCases) {
    SCOPED_TRACE(test.seed);
    const auto weight_desc = c::gguf_grouped_weight_description(
        test.type, c::DataType::kFloat32, grouped::kResidentExperts,
        grouped::kColumns, grouped::kReduction).value();
    const auto packed = grouped::pack_weight(
        test, weight_desc.required_bytes().value());
    const auto workload = q::workload(*test.profile,
                                      c::WorkloadPhase::kDecode, 0x0a0b);
    const auto weight = packed_array(packed.bytes);
    const std::array specs{a::input_spec(weight_desc).value(),
                           a::input_spec(activation_desc).value(),
                           a::input_spec(ids_desc).value()};
    Call call{context.prepare(c::model_ops::grouped_matmul(), specs,
                              workload).value()};
    EXPECT_EQ(call.program.workload(), workload);
    std::optional<mx::array> retained;
    for (int replay = 0; replay < 2; ++replay) {
      const auto input = q24::expert_input(3 + replay * 14);
      const auto selected = grouped::ids(replay);
      const auto activation = upload(context, activation_desc, std::span(input));
      const auto ids = upload(context, ids_desc, std::span(selected));
      const auto output = run(call, weight, activation, ids);
      ASSERT_EQ(output.size(), 1U);
      retained = output[0];
      const auto actual = float_values(output[0]);
      const auto expected = grouped::expected(input, selected, packed.decoded);
      d5_gguf_test::expect_quantized_matmul_close(actual, expected);
    }

    const auto before_failure = bits(*retained);
    const auto input = q24::expert_input(17);
    const auto activation = upload(context, activation_desc, std::span(input));
    for (std::int32_t bad : {-1, static_cast<std::int32_t>(grouped::kResidentExperts)}) {
      auto selected = grouped::ids(1);
      selected.back() = bad;
      const auto bad_ids = upload(context, ids_desc, std::span(selected));
      const auto bad_inputs = std::array{weight, activation, bad_ids};
      auto invocation = call.program.create_invocation().value();
      ASSERT_TRUE(invocation.bind(bad_inputs).ok());
      auto failed = invocation.submit();
      ASSERT_TRUE(failed) << failed.error().message();
      auto failure = failed->wait();
      ASSERT_FALSE(failure);
      EXPECT_TRUE(failure.error().IsOutOfRange());
      EXPECT_EQ(bits(*retained), before_failure);
    }
    const auto valid = grouped::ids(1);
    const auto valid_ids = upload(context, ids_desc, std::span(valid));
    const auto recovered = run(call, weight, activation, valid_ids);
    ASSERT_EQ(recovered.size(), 1U);
    d5_gguf_test::expect_quantized_matmul_close(
        float_values(recovered[0]), grouped::expected(input, valid, packed.decoded));
  }
}

TEST(AppleQwen38_24T,
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
  const auto workload =
      q24::workload(c::WorkloadPhase::kDecode, 0x0a0b);
  auto context = a::Context::create().value();
  auto gate_weight_values = q24::expert_weight(
      q24::kExpertHidden, q24::kHidden, seed);
  const auto gate_weight =
      upload(context, weight_desc, std::span(gate_weight_values));
  gate_weight_values.clear();
  gate_weight_values.shrink_to_fit();
  auto up_weight_values = q24::expert_weight(
      q24::kExpertHidden, q24::kHidden, seed + 1);
  const auto up_weight =
      upload(context, weight_desc, std::span(up_weight_values));
  up_weight_values.clear();
  up_weight_values.shrink_to_fit();
  auto down_weight_values = q24::expert_weight(
      q24::kHidden, q24::kExpertHidden, seed + 2);
  const auto down_weight =
      upload(context, down_weight_desc, std::span(down_weight_values));
  down_weight_values.clear();
  down_weight_values.shrink_to_fit();
  const auto ids_data = q24::active_expert_ids();
  const auto ids = upload(context, ids_desc, std::span(ids_data));
  auto gate = prepare(context, workload, c::model_ops::grouped_matmul(),
                      weight_desc, activation_desc, ids_desc);
  auto up = prepare(context, workload, c::model_ops::grouped_matmul(),
                    weight_desc, activation_desc, ids_desc);
  auto gated = prepare(
      context, workload,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      expanded_desc, expanded_desc);
  auto down = prepare(context, workload, c::model_ops::grouped_matmul(),
                      down_weight_desc, expanded_desc, ids_desc);
  for (int replay : {3, 17}) {
    const auto input = q24::expert_input(replay);
    const auto activation =
        upload(context, activation_desc, std::span(input));
    const auto gate_value = run(gate, gate_weight, activation, ids);
    const auto up_value = run(up, up_weight, activation, ids);
    const auto product = run(gated, gate_value[0], up_value[0]);
    const auto output = run(down, down_weight, product[0], ids);
    const auto actual = values(output[0]);
    const auto expected = q24::expert_oracle(input, seed);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i],
                  std::abs(expected[i]) * q24::kExpertRelativeTolerance +
                      q24::kExpertAbsoluteTolerance)
          << i;
  }
}

TEST(AppleQwen38_24T, D5OfficialBfloatRouterPreservesF32NearTieScores) {
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, q24::kHidden});
  const auto weight_desc =
      dense(c::DataType::kBFloat16, {q24::kExperts, q24::kHidden});
  std::vector<B16> weights(q24::kExperts * q24::kHidden);
  weights[9 * q24::kHidden + 20] = q::encode(1.0);
  weights[10 * q24::kHidden + 20] = q::encode(1.0);
  weights[10 * q24::kHidden + 21] = q::encode(1.0);
  auto context = a::Context::create().value();
  const auto weight = upload(context, weight_desc, std::span(weights));
  auto call = prepare(context,
                      q24::workload(c::WorkloadPhase::kDecode, 0x0a10),
                      c::model_ops::bfloat_linear_f32(), activation_desc,
                      weight_desc);
  for (const double fraction : {0.001953125, 0.0029296875}) {
    std::vector<B16> input(q24::kHidden);
    input[20] = q::encode(1.0);
    input[21] = q::encode(fraction);
    const auto activation = upload(context, activation_desc, std::span(input));
    const auto output = run(call, activation, weight);
    ASSERT_EQ(output.size(), 1U);
    const auto scores = values(output[0]);
    ASSERT_EQ(scores.size(), q24::kExperts);
    EXPECT_NEAR(scores[9], 1.0, 1.0e-6);
    EXPECT_NEAR(scores[10], 1.0 + fraction, 1.0e-6);
  }
}

void run_q24_semantic_fragment(
    d5_qwen38_24t_semantic_fragments::Fragment fixture) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  auto context = a::Context::create().value();
  const auto workload =
      q::workload(*fixture.profile, c::WorkloadPhase::kDecode, fixture.region);
  std::vector<std::optional<mx::array>> tensors(fixture.descriptions.size());
  const auto &bytes = fixture.constant_bytes;
  for (std::size_t index = 0; index < fixture.constants.size(); ++index) {
    const auto id = fixture.constants[index];
    tensors[id.ordinal()] =
        fixture.description(id).is_plain()
            ? upload(context, fixture.description(id),
                     std::span<const std::byte>(bytes[index]))
            : packed_array(bytes[index]);
  }
  fixture.constant_bytes = {};

  std::vector<Call> calls;
  calls.reserve(fixture.nodes.size());
  for (const auto &step : fixture.nodes) {
    if (step.operation.kind() == c::OperationKind::kView)
      continue;
    std::vector<a::InputSpec> specs;
    for (const auto id : step.inputs)
      specs.push_back(a::input_spec(fixture.description(id)).value());
    calls.push_back({context.prepare(step.operation, specs, workload).value()});
  }

  sem::StateFeedback feedback{fixture};
  for (const auto &expected : fixture.replays) {
    // The previous outputs have been checked; release their leases before
    // replay.
    for (const auto &step : fixture.nodes)
      for (auto id : step.outputs)
        tensors[id.ordinal()] = std::nullopt;
    for (std::size_t i = 0; i < fixture.inputs.size(); ++i) {
      auto id = fixture.inputs[i];
      tensors[id.ordinal()] = upload(context, fixture.description(id),
                                     feedback.input(expected, i));
    }
    std::size_t call_index = 0;
    for (const auto &step : fixture.nodes) {
      SCOPED_TRACE(c::operation_name(step.operation.kind()));
      if (step.operation.kind() == c::OperationKind::kView) {
        ASSERT_EQ(step.inputs.size(), 1U);
        ASSERT_EQ(step.outputs.size(), 1U);
        ASSERT_EQ(step.operation.get_if<c::View>()->kind(),
                  c::ViewKind::kReshape);
        tensors[step.outputs[0].ordinal()] = ready(mx::reshape(
            *tensors[step.inputs[0].ordinal()],
            native_shape(fixture.description(step.outputs[0]), false),
            stream()));
        continue;
      }
      std::vector<mx::array> inputs;
      for (const auto id : step.inputs)
        inputs.push_back(*tensors[id.ordinal()]);
      const auto outputs =
          run(calls[call_index++], std::span<const mx::array>(inputs));
      ASSERT_EQ(outputs.size(), step.outputs.size());
      for (std::size_t index = 0; index < outputs.size(); ++index)
        tensors[step.outputs[index].ordinal()] = outputs[index];
    }
    for (std::size_t output = 0; output < fixture.outputs.size(); ++output) {
      const auto actual =
          float_values(*tensors[fixture.outputs[output].ordinal()]);
      feedback.observe(output, actual);
      sem::expect_output(fixture, actual, expected.outputs[output], {},
                         fixture.exact_output(output));
    }
  }
}

TEST(AppleMoeFragments, D5MatureClampBoundaries) {
  run_q24_semantic_fragment(d5_moe_execution_fragments::clamp_boundaries());
}

TEST(AppleMoeFragments, D5ThirteenProfilesModelRouters) {
  for (const auto &model : d5_moe_execution_cases::kCases) {
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::router(model));
  }
}

TEST(AppleMoeFragments, D5ThirteenProfilesExpertActivation) {
  for (const auto &model : d5_moe_execution_cases::kCases) {
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::activation_core(model));
  }
}

TEST(AppleMoeFragments, D5EightGgufProfilesFullMoe) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &model : d5_moe_execution_cases::kCases) {
    if (!model.gguf)
      continue;
    SCOPED_TRACE(model.profile->model_repository);
    SCOPED_TRACE(model.profile->variant);
    run_q24_semantic_fragment(d5_moe_execution_fragments::gguf_moe(model));
  }
}

TEST(AppleMoeFragments, D5KdaProjectionRoleBatch) {
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto &role : d5_kda_projection_cases::kCases) {
    SCOPED_TRACE(role.profile->model_repository);
    SCOPED_TRACE(role.profile->variant);
    SCOPED_TRACE(role.roles);
    run_q24_semantic_fragment(d5_moe_semantic_fragments::kda_projection(role));
  }
}

TEST(AppleMoeFragments, D5SixProfilesBoundedKdaActualStateFeedback) {
  namespace moe = d5_moe_semantic_fragments;
  for (auto profile : moe::kKdaProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(moe::kda(profile));
  }
}

TEST(AppleMoeFragments, D5KimiThreeProfilesSituActivation) {
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

TEST(AppleMoeFragments, D5SevenProfilesHyperConnectionCore) {
  namespace moe = d5_moe_semantic_fragments;
  for (const auto &profile : moe::kHyperProfiles) {
    SCOPED_TRACE(profile.profile->model_repository);
    SCOPED_TRACE(profile.profile->variant);
    run_q24_semantic_fragment(moe::hyper_connection(profile));
  }
}

TEST(AppleQwenRecurrent, D5NineProfilesUseCompleteGateAndActualStateFeedback) {
  namespace sem = d5_qwen38_24t_semantic_fragments;
  if (!d5_qwen35_fixture::root())
    GTEST_SKIP() << "HYLE_D5_QWEN35_FIXTURE is required for actual 2B parameters";
  for (auto profile : sem::kRecurrentProfiles) {
    SCOPED_TRACE(profile->model_repository);
    SCOPED_TRACE(profile->variant);
    run_q24_semantic_fragment(sem::gdn(profile, 8));
  }
}

TEST(AppleQwen38_24T, D5GgufAttentionGdnAndMtpUseRealRoleTypes) {
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

TEST(AppleQwen38_24T, D5GgufAttentionAndGdnProjectionRoleBatch) {
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

TEST(AppleQwen38_24T, D5OfficialAttentionAndGdnProjectionRoleBatch) {
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
  auto context = a::Context::create().value();
  const auto workload = q::workload(*profile, c::WorkloadPhase::kDecode, 0x0a12);
  std::vector<std::optional<mx::array>> tensors(direct.descriptions.size());
  const auto &bytes = fixture.constants;
  for (std::size_t index = 0; index < direct.full.constants.size(); ++index) {
    const auto id = direct.full.constants[index];
    tensors[id.ordinal()] = direct.description(id).is_plain()
        ? upload(context, direct.description(id), std::span<const std::byte>(bytes[index]))
        : packed_array(bytes[index]);
  }
  fixture.constants = {};

  std::vector<Call> calls;
  calls.reserve(direct.nodes.size());
  for (const auto &step : direct.nodes) {
    if (step.operation.kind() == c::OperationKind::kView)
      continue;
    std::vector<a::InputSpec> specs;
    for (const auto id : step.inputs)
      specs.push_back(a::input_spec(direct.description(id)).value());
    calls.push_back({context.prepare(step.operation, specs, workload).value()});
  }

  for (const auto &expected : fixture.expected) {
    tensors[direct.full.activation.ordinal()] =
        upload(context, direct.full.activation_desc,
               std::span(expected.activation));
    std::size_t call_index = 0;
    for (const auto &step : direct.nodes) {
      SCOPED_TRACE(c::operation_name(step.operation.kind()));
      if (step.operation.kind() == c::OperationKind::kView) {
        ASSERT_EQ(step.inputs.size(), 1U);
        ASSERT_EQ(step.outputs.size(), 1U);
        ASSERT_EQ(step.operation.get_if<c::View>()->kind(),
                  c::ViewKind::kReshape);
        tensors[step.outputs[0].ordinal()] = ready(mx::reshape(
            *tensors[step.inputs[0].ordinal()],
            native_shape(direct.description(step.outputs[0]), false), stream()));
        continue;
      }
      std::vector<mx::array> inputs;
      for (const auto id : step.inputs)
        inputs.push_back(*tensors[id.ordinal()]);
      const auto outputs =
          run(calls[call_index++], std::span<const mx::array>(inputs));
      ASSERT_EQ(outputs.size(), step.outputs.size());
      for (std::size_t index = 0; index < outputs.size(); ++index)
        tensors[step.outputs[index].ordinal()] = outputs[index];
    }
    const auto ids = values(*tensors[direct.full.outputs[0].ordinal()]);
    ASSERT_EQ(ids.size(), expected.ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
      EXPECT_EQ(ids[i], expected.ids[i]) << "slot=" << i;
    for (std::size_t output = 1; output < direct.full.outputs.size(); ++output) {
      const auto actual =
          values(*tensors[direct.full.outputs[output].ordinal()]);
      const auto &oracle = output == 1 ? expected.routed
                           : output == 2 ? expected.combined
                           : output == 3 ? expected.shared
                                         : expected.full;
      moe::expect_output(profile, actual, oracle);
    }
  }
}

TEST(AppleQwen38_24T, D5FullRouterSharedExpertAndNearTieDirectReplay) {
  run_full_moe_profile(&q24::kOfficialProfile);
}

TEST(AppleQwen38_24T, D5GgufFullRouterSharedExpertAndNearTieReplay) {
  for (const auto *profile : {&d5_qwen_gguf_vocabulary_cases::k24TPrimaryProfile,
                               &d5_qwen_gguf_vocabulary_cases::k24TFidelityProfile}) {
    SCOPED_TRACE(profile->variant);
    run_full_moe_profile(profile);
  }
}

TEST_P(AppleQwen38Profiles,
       D5Qwen38ProfilePartialRopeGqaAndOutputGateReplay) {
  const auto query_desc = dense(c::DataType::kBFloat16,
                                {1, 1, GetParam()->query_heads, q::kHeadDepth});
  const auto kv_desc = dense(c::DataType::kBFloat16,
                             {1, 1, GetParam()->kv_heads, q::kHeadDepth});
  const auto scale_desc = dense(c::DataType::kBFloat16, {q::kHeadDepth});
  const auto position_desc = dense(c::DataType::kInt32, {1});
  const auto frequency_desc =
      dense(c::DataType::kFloat32, {q::kRotaryDimensions / 2});
  const auto workload =
      q::workload(*GetParam()->profile, c::WorkloadPhase::kDecode, 4);
  auto context = a::Context::create().value();
  const auto query_scale_values = q::scales(q::kHeadDepth, 1);
  const auto key_scale_values = q::scales(q::kHeadDepth, 2);
  const auto query_scale =
      upload(context, scale_desc, std::span(query_scale_values));
  const auto key_scale =
      upload(context, scale_desc, std::span(key_scale_values));
  const std::array<std::int32_t, 1> positions{11};
  std::vector<float> frequencies(q::kRotaryDimensions / 2);
  for (std::size_t i = 0; i < frequencies.size(); ++i)
    frequencies[i] = 1.0F / static_cast<float>(i + 1);
  const auto position = upload(context, position_desc, std::span(positions));
  const auto frequency =
      upload(context, frequency_desc, std::span(frequencies));
  auto query_norm = prepare(context, workload, c::Operation(c::RmsNorm{}),
                            query_desc, query_scale);
  auto key_norm = prepare(context, workload, c::Operation(c::RmsNorm{}),
                          kv_desc, key_scale);
  const c::RoPE rope{.rotary_dimensions = q::kRotaryDimensions,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kInterleaved};
  auto query_rope = prepare(context, workload, c::Operation(rope), query_desc,
                            position, frequency);
  auto key_rope = prepare(context, workload, c::Operation(rope), kv_desc,
                          position, frequency);
  auto attention = prepare(
      context, workload,
      c::Operation(c::ScaledDotProductAttention{
          .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
          .scale = 0.0625F,
          .causal = true,
          .accumulation_type = c::DataType::kFloat32}),
      query_desc, kv_desc, kv_desc);
  auto sigmoid = prepare(context, workload,
                         c::Operation(c::Unary{c::UnaryKind::kSigmoid}),
                         query_desc);
  auto multiply =
      prepare(context, workload,
              c::Operation(c::Binary{c::BinaryKind::kMultiply}), query_desc,
              query_desc);
  for (int seed : {2, 13}) {
    const auto queries = q::values(*query_desc.element_count(), seed);
    const auto keys = q::values(*kv_desc.element_count(), seed + 1);
    const auto value_data =
        q::values(*kv_desc.element_count(), seed + 2, 128.0);
    const auto gate_data =
        q::values(*query_desc.element_count(), seed + 3, 32.0);
    const auto query = upload(context, query_desc, std::span(queries));
    const auto key = upload(context, kv_desc, std::span(keys));
    const auto value = upload(context, kv_desc, std::span(value_data));
    const auto gate = upload(context, query_desc, std::span(gate_data));
    const auto normalized_query = run(query_norm, query, query_scale);
    const auto normalized_key = run(key_norm, key, key_scale);
    const auto rotated_query =
        run(query_rope, normalized_query[0], position, frequency);
    const auto rotated_key =
        run(key_rope, normalized_key[0], position, frequency);
    const auto attended =
        run(attention, rotated_query[0], rotated_key[0], value);
    const auto gate_value = run(sigmoid, gate);
    const auto output = run(multiply, attended[0], gate_value[0]);
    const auto actual = values(output[0]);
    for (std::size_t head = 0; head < static_cast<std::size_t>(GetParam()->query_heads); ++head) {
      const auto value_head = head / (GetParam()->query_heads / GetParam()->kv_heads);
      for (std::size_t column = 0; column < q::kHeadDepth; ++column) {
        const auto index = head * q::kHeadDepth + column;
        const auto gate_expected = q::decode(q::encode(
            1.0 / (1.0 + std::exp(-q::decode(gate_data[index])))));
        const auto expected = q::decode(q::encode(
            q::decode(value_data[value_head * q::kHeadDepth + column]) *
            gate_expected));
        EXPECT_NEAR(actual[index], expected,
                    std::abs(expected) * 0.012 + 2.0e-5)
            << index;
      }
    }
  }
}

TEST_P(AppleQwen38Profiles,
       D5Qwen38ProfileMtpUsesExactConcatWidthAndReplays) {
  constexpr int weight_seed = 67;
  const auto half_desc = dense(c::DataType::kBFloat16, {1, GetParam()->hidden});
  const auto joined_desc = dense(c::DataType::kBFloat16, {1, (2 * GetParam()->hidden)});
  const auto weight_desc =
      dense(c::DataType::kBFloat16, {GetParam()->hidden, (2 * GetParam()->hidden)})
          .transposed()
          .value();
  const auto scale_desc = dense(c::DataType::kBFloat16, {GetParam()->hidden});
  const auto workload =
      q::workload(*GetParam()->profile, c::WorkloadPhase::kDecode, 0x0d0e);
  auto context = a::Context::create().value();
  auto weight_values =
      q::sparse_output_major_weight(GetParam()->hidden, (2 * GetParam()->hidden), weight_seed);
  const auto weight = upload(context, weight_desc, std::span(weight_values));
  weight_values.clear();
  weight_values.shrink_to_fit();
  const auto scale_values = q::scales(GetParam()->hidden, 11);
  const auto scale = upload(context, scale_desc, std::span(scale_values));
  auto concat = prepare(context, workload,
                        c::Operation(c::Concat{.axis = 1}), half_desc,
                        half_desc);
  auto projection = prepare(context, workload, c::Operation(c::MatMul{}),
                            joined_desc, weight);
  auto normalize = prepare(context, workload, c::Operation(c::RmsNorm{}),
                           half_desc, scale);
  auto cast = prepare(context, workload,
                      c::Operation(c::Cast{c::DataType::kFloat32}), half_desc);
  for (int seed : {5, 19}) {
    const auto input = q::values((2 * GetParam()->hidden), seed);
    const auto first = upload(
        context, half_desc,
        std::span<const B16>(input).first(static_cast<std::size_t>(GetParam()->hidden)));
    const auto second = upload(
        context, half_desc,
        std::span<const B16>(input).last(static_cast<std::size_t>(GetParam()->hidden)));
    const auto joined = run(concat, first, second);
    const auto projected = run(projection, joined[0], weight);
    const auto normalized = run(normalize, projected[0], scale);
    const auto converted = run(cast, normalized[0]);
    const auto projected_expected =
        q::sparse_projection(input, GetParam()->hidden, weight_seed);
    expect_qwen38_near(values(projected[0]), projected_expected);
    const auto normalized_expected =
        q::rms_norm(q::as_bf16(projected_expected), scale_values);
    expect_qwen38_near(values(converted[0]), normalized_expected);
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, AppleQwen38Profiles,
    ::testing::Values(&d5_qwen38_profile_cases::kOfficial27B,
                      &d5_qwen38_profile_cases::kPrimaryGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, AppleQwen38Profiles,
    ::testing::Values(&d5_qwen38_profile_cases::kFidelityGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5Official24T, AppleQwen38Profiles,
    ::testing::Values(&d5_qwen38_profile_cases::kOfficial24T));

class AppleQwenOfficialVocabulary
    : public ::testing::TestWithParam<
          const d5_qwen_vocabulary_fixture::ProfileCase *> {};

TEST_P(AppleQwenOfficialVocabulary,
       D5WholeVocabularyHeadStreamsAllRows) {
  namespace vocab = d5_qwen_vocabulary_fixture;
  const auto &shape = *GetParam();
  const auto activation_desc =
      dense(c::DataType::kBFloat16, {1, shape.hidden});
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 1);
  auto context = a::Context::create().value();
  const auto prepare_head = [&](std::int64_t rows) {
    const auto stored = dense(c::DataType::kBFloat16, {rows, shape.hidden});
    return prepare(context, workload, c::Operation(c::MatMul{}),
                   activation_desc, stored.transposed().value());
  };
  const auto upload_weight = [&](vocab::Shard shard,
                                 std::span<const vocab::BF16> values) {
    const auto stored =
        dense(c::DataType::kBFloat16, {shard.rows, shape.hidden});
    return upload(context, stored.transposed().value(), values);
  };
  const auto execute = [&](Call &call, std::span<const vocab::BF16> input,
                           const mx::array &weight) {
    const auto activation = upload(context, activation_desc, input);
    const auto outputs = run(call, activation, weight);
    EXPECT_EQ(outputs.size(), 1U);
    return values(outputs.at(0));
  };
  d5_qwen_vocabulary_test::check_head(shape.hidden, prepare_head,
                                       upload_weight, execute);
}

TEST_P(AppleQwenOfficialVocabulary,
       D5WholeVocabularyGatherMapsGlobalRows) {
  namespace vocab = d5_qwen_vocabulary_fixture;
  const auto &shape = *GetParam();
  const auto ids_desc = dense(c::DataType::kInt32, {1});
  const auto workload =
      q::workload(*shape.profile, c::WorkloadPhase::kDecode, 1);
  auto context = a::Context::create().value();
  const auto prepare_gather = [&](std::int64_t rows) {
    const auto table_desc =
        dense(c::DataType::kBFloat16, {rows, shape.hidden});
    return prepare(context, workload, c::Operation(c::Gather{}), table_desc,
                   ids_desc);
  };
  const auto upload_table = [&](vocab::Shard shard,
                                std::span<const vocab::BF16> values) {
    const auto table_desc =
        dense(c::DataType::kBFloat16, {shard.rows, shape.hidden});
    return upload(context, table_desc, values);
  };
  std::optional<mx::array> retained;
  const auto execute = [&](Call &call, std::int64_t local_id,
                           const mx::array &table) {
    const std::array index{static_cast<std::int32_t>(local_id)};
    const auto ids = upload(context, ids_desc, std::span(index));
    const auto outputs = run(call, table, ids);
    EXPECT_EQ(outputs.size(), 1U);
    retained = outputs.at(0);
    return values(outputs.at(0));
  };
  const auto reject = [&](Call &call, std::int64_t local_id,
                          const mx::array &table) {
    ASSERT_TRUE(retained.has_value());
    const auto before = bits(*retained);
    const std::array index{static_cast<std::int32_t>(local_id)};
    const auto ids = upload(context, ids_desc, std::span(index));
    auto invocation = call.program.create_invocation().value();
    const std::array inputs{table, ids};
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto failed = invocation.submit();
    if (failed) {
      const auto result = failed->wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
    } else {
      EXPECT_TRUE(failed.error().IsOutOfRange());
    }
    EXPECT_EQ(bits(*retained), before);
  };
  d5_qwen_vocabulary_test::check_gather(shape.hidden, prepare_gather,
                                         upload_table, execute, reject);
}

INSTANTIATE_TEST_SUITE_P(
    D5Official2B27BAnd24T, AppleQwenOfficialVocabulary,
    ::testing::Values(&d5_qwen_vocabulary_fixture::kOfficial2B,
                      &d5_qwen_vocabulary_fixture::kOfficial27B,
                      &d5_qwen_vocabulary_fixture::kOfficial24T));

class AppleQwenGgufVocabulary
    : public ::testing::TestWithParam<
          const d5_qwen_gguf_vocabulary_cases::Case *> {};

TEST_P(AppleQwenGgufVocabulary, HeadRunsEntireLogicalVocabulary) {
  namespace gguf = d5_qwen_gguf_vocabulary_apple_test;
  auto context = a::Context::create().value();
  const auto prepare_native = [&](const c::WorkloadContext &workload,
                                  c::Operation operation,
                                  const c::TensorDesc &left,
                                  const c::TensorDesc &right) {
    const std::array specs{a::input_spec(left).value(),
                           a::input_spec(right).value()};
    return Call{context.prepare(operation, specs, workload).value()};
  };
  const auto upload_dense = [&](const c::TensorDesc &description,
                                std::span<const std::byte> bytes) {
    return upload(context, description, bytes);
  };
  gguf::check_head(*GetParam(), prepare_native, upload_dense,
                   [](std::span<const std::byte> bytes) {
                     return packed_array(bytes);
                   },
                   [](Call &call, const mx::array &left,
                      const mx::array &right) { return run(call, left, right); });
}

TEST_P(AppleQwenGgufVocabulary, GatherMapsGlobalRowsAndRejectsOutOfBounds) {
  namespace gguf = d5_qwen_gguf_vocabulary_apple_test;
  auto context = a::Context::create().value();
  const auto prepare_native = [&](const c::WorkloadContext &workload,
                                  c::Operation operation,
                                  const c::TensorDesc &left,
                                  const c::TensorDesc &right) {
    const std::array specs{a::input_spec(left).value(),
                           a::input_spec(right).value()};
    return Call{context.prepare(operation, specs, workload).value()};
  };
  const auto upload_dense = [&](const c::TensorDesc &description,
                                std::span<const std::byte> bytes) {
    return upload(context, description, bytes);
  };
  gguf::check_gather(*GetParam(), prepare_native, upload_dense,
                     [](std::span<const std::byte> bytes) {
                       return packed_array(bytes);
                     },
                     [](Call &call, const mx::array &left,
                        const mx::array &right) { return run(call, left, right); });
}

INSTANTIATE_TEST_SUITE_P(
    D5ExactQwenGguf, AppleQwenGgufVocabulary,
    ::testing::Values(&d5_qwen_gguf_vocabulary_cases::kCases[0],
                      &d5_qwen_gguf_vocabulary_cases::kCases[1],
                      &d5_qwen_gguf_vocabulary_cases::kCases[2],
                      &d5_qwen_gguf_vocabulary_cases::kCases[3],
                      &d5_qwen_gguf_vocabulary_cases::kCases[4],
                      &d5_qwen_gguf_vocabulary_cases::kCases[5]));

INSTANTIATE_TEST_SUITE_P(
    D5ExactMoeVocabulary, AppleQwenGgufVocabulary,
    ::testing::ValuesIn(d5_moe_profile_cases::kVocabularyPointers));

} // namespace
