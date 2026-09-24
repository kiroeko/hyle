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

#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/d5_qwen35_fixture.h"
#include "hyle/compute/apple.h"
#include "hyle/compute/model_ops.h"
#include "tensor_test_support.h"

namespace {
using namespace apple_tensor_test;
namespace a = c::apple;
namespace mx = mlx::core;

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
  return ready(mx::array(storage.data(),
                         mx::Shape{static_cast<int>(storage.size())},
                         mx::uint8));
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
  EXPECT_TRUE(invocation.bind(inputs).ok());
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

TEST(AppleD5,
     D5Qwen35PrimaryGgufQuantizedEmbeddingUsesNativeRowsAndReplays) {
  d5_gguf_test::QuantizerCleanup cleanup;
  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto rows = d5_qwen35_fixture::kEmbeddingRows;
  constexpr std::int64_t tokens = 2;
  const auto stored = c::gguf_weight_description(
      c::GgufQuantizedType::kQ6K, c::DataType::kFloat32, hidden, rows);
  ASSERT_TRUE(stored) << stored.error().message();
  const auto table_desc = stored->transposed().value();
  const auto ids_desc = dense(c::DataType::kInt32, {tokens});
  const std::array specs{a::input_spec(table_desc).value(),
                         a::input_spec(ids_desc).value()};
  const auto workload =
      d5_qwen35_fixture::workload(d5_qwen35_fixture::kPrimaryGgufProfile,
                                  c::WorkloadPhase::kDecode, 0x010f);
  auto context = a::Context::create().value();
  auto program =
      context.prepare(c::Operation(c::Gather{}), specs, workload).value();
  EXPECT_EQ(program.workload(), workload);

  const auto physical = d5_gguf_test::values(rows, hidden, 37);
  const auto packed =
      d5_gguf_test::pack(c::GgufQuantizedType::kQ6K, physical, rows, hidden,
                         table_desc.required_bytes().value());
  const auto table = packed_array(packed.bytes);
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
        d5_gguf_test::selected_rows(packed.decoded, hidden, selection);
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
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 4);
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
    const auto output = run(attention, rotated_query[0], rotated_key[0], value);
    const auto actual = bits(output[0]);
    for (std::size_t head = 0; head < query_heads; ++head)
      for (std::size_t column = 0; column < depth; ++column)
        EXPECT_EQ(actual[head * depth + column],
                  value_data[(head / (query_heads / kv_heads)) * depth + column]
                      .bits);
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
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 5);
  auto context = a::Context::create().value();
  const auto conv_weight =
      upload(context, conv_weight_desc, std::span(conv_weights));
  const auto decay = upload(context, gate_desc, std::span(log_decay));
  const auto update = upload(context, gate_desc, std::span(beta));
  auto convolution = prepare(context, workload, c::model_ops::causal_conv1d(),
                             conv_samples_desc, conv_weight);
  auto gdn = prepare(context, workload, c::Operation(c::GatedDeltaNet{}),
                     qkv_desc, qkv_desc, qkv_desc, decay, update, state_desc);
  for (int seed : {2, 9}) {
    std::vector<float> conv_samples(conv_samples_desc.element_count().value());
    std::array<std::vector<float>, 3> qkv_values;
    for (auto &port : qkv_values)
      port.resize(qkv_desc.element_count().value());
    std::vector<float> state_values(state_desc.element_count().value());
    for (std::size_t i = 0; i < conv_samples.size(); ++i)
      conv_samples[i] =
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
} // namespace
