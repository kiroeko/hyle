#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "d5_gguf_test_support.h"
#include "d5_qwen38_24t_full_router.h"
#include "d5_qwen_gguf_vocabulary_cases.h"

namespace d5_qwen38_24t_moe_profiles {
namespace c = hyle::compute;
namespace q = d5_qwen38_24t_fixture;
namespace full = d5_qwen38_24t_full_router;
namespace profiles = d5_qwen_gguf_vocabulary_cases;

inline constexpr std::array kProfiles{
    &q::kOfficialProfile, &profiles::k24TPrimaryProfile,
    &profiles::k24TFidelityProfile};
inline constexpr std::array kReplays{3, 17, 29};

template <class T>
std::vector<std::byte> bytes(std::span<const T> values) {
  const auto view = std::as_bytes(values);
  return {view.begin(), view.end()};
}

struct Expected final {
  std::vector<std::byte> activation;
  std::array<std::int32_t, q::kTopK> ids{};
  std::vector<float> routed, combined, shared, full;
};

struct Fixture final {
  const c::ModelArtifactProfileDescription *profile;
  full::Layout layout;
  std::array<std::vector<std::byte>, 10> constants;
  std::array<Expected, 3> expected;
};

using Replays = std::array<std::vector<float>, 3>;
using ExpertIds = std::array<std::vector<std::int32_t>, 3>;

template <class Actual>
void expect_output(const c::ModelArtifactProfileDescription *profile,
                   const Actual &actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  if (profile == &q::kOfficialProfile) {
    for (std::size_t index = 0; index < actual.size(); ++index)
      EXPECT_NEAR(actual[index], expected[index],
                  std::abs(expected[index]) * q::kExpertRelativeTolerance +
                      q::kExpertAbsoluteTolerance) << index;
    return;
  }
  // Use the established quantized-chain gate. Normalize to the reference's
  // largest value so its absolute floor cannot hide a tiny all-zero expert
  // result. The oracle retains F32 activations; native ggml dots may quantize
  // those activations again between gate/up and down projections.
  float scale = 0.0F;
  for (const auto value : expected)
    scale = std::max(scale, std::abs(value));
  ASSERT_GT(scale, 0.0F);
  std::vector<float> normalized_actual(actual.size()), normalized_expected(actual.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    normalized_actual[index] = static_cast<float>(actual[index]) / scale;
    normalized_expected[index] = expected[index] / scale;
  }
  d5_gguf_test::expect_quantized_chain_matmul_close(normalized_actual,
                                                   normalized_expected);
}

// Independent F64 dot products over the decoded native blocks. Skip exact
// zeros in this sparse-content fixture, retaining all nonzero quantizer error.
// Scan each weight bank once, then release it before preparing the next bank.
inline Replays project(std::span<const float> decoded, std::int64_t outputs,
                       std::int64_t inputs, const Replays &activation,
                       const ExpertIds &ids) {
  std::array<std::vector<double>, 3> sums;
  for (std::size_t replay = 0; replay < sums.size(); ++replay)
    sums[replay].resize(ids[replay].size() * outputs);
  const auto rows = decoded.size() / inputs;
  for (std::size_t row = 0; row < rows; ++row) {
    const auto expert = row / outputs;
    const auto output = row % outputs;
    for (std::int64_t column = 0; column < inputs; ++column) {
      const auto weight = decoded[row * inputs + column];
      if (weight == 0.0F)
        continue;
      for (std::size_t replay = 0; replay < sums.size(); ++replay)
        for (std::size_t slot = 0; slot < ids[replay].size(); ++slot)
          if (static_cast<std::size_t>(ids[replay][slot]) == expert)
            sums[replay][slot * outputs + output] +=
                static_cast<double>(weight) *
                activation[replay][slot * inputs + column];
    }
  }
  Replays result;
  for (std::size_t replay = 0; replay < result.size(); ++replay) {
    result[replay].resize(sums[replay].size());
    std::ranges::transform(sums[replay], result[replay].begin(),
                           [](double value) { return static_cast<float>(value); });
  }
  return result;
}

inline Fixture make(const c::ModelArtifactProfileDescription *profile) {
  Fixture result{profile, full::official_layout(), {}, {}};
  std::array<full::Expected, 3> baseline;
  for (std::size_t replay = 0; replay < baseline.size(); ++replay) {
    baseline[replay] = full::expected(kReplays[replay]);
    const auto &source = baseline[replay];
    result.expected[replay] = {
        bytes<q::BF16>(source.activation), source.ids, source.routed,
        source.combined, source.shared, source.full};
  }
  if (profile == &q::kOfficialProfile) {
    const auto weights = full::make_weights();
    const auto views = full::constant_bytes(weights);
    for (std::size_t index = 0; index < views.size(); ++index)
      result.constants[index].assign(views[index].begin(), views[index].end());
    return result;
  }

  const bool fidelity = profile == &profiles::k24TFidelityProfile;
  result.layout.activation_desc = full::dense(
      c::DataType::kFloat32, std::array<std::int64_t, 2>{1, q::kHidden});
  result.layout.linear = c::Operation(c::MatMul{});
  // GGUF router and shared gating weights are F32, independently of the
  // expert quantization. MatMul receives their transposed logical views.
  result.layout.constant_descs[0] = full::dense(
      c::DataType::kFloat32,
      std::array<std::int64_t, 2>{q::kExperts, q::kHidden}).transposed().value();
  result.layout.constant_descs[8] = full::dense(
      c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, q::kHidden}).transposed().value();
  std::vector<float> router(q::kExperts * q::kHidden);
  for (std::int64_t expert = 0; expert < q::kTopK; ++expert)
    router[expert * q::kHidden + expert] = 1.0F;
  router[q::kTopK * q::kHidden + 20] = 1.0F;
  router[q::kTopK * q::kHidden + 21] = 1.0F;
  result.constants[0] = bytes<float>(router);
  const std::array<std::int32_t, q::kTopK> replication{};
  const std::array<std::int32_t, 1> shared_ids{0};
  result.constants[1] = bytes<std::int32_t>(replication);
  result.constants[9] = bytes<std::int32_t>(shared_ids);
  std::vector<float> shared_score_weight(q::kHidden);
  shared_score_weight[q::source_index(0, q::kHidden, full::kSharedSeed + 3)] =
      static_cast<float>(q::decode(q::coefficient(0, full::kSharedSeed + 3)));
  result.constants[8] = bytes<float>(shared_score_weight);

  Replays tokens, routed_input;
  ExpertIds routed_ids, shared_experts;
  for (std::size_t replay = 0; replay < tokens.size(); ++replay) {
    for (const auto value : baseline[replay].activation)
      tokens[replay].push_back(static_cast<float>(q::decode(value)));
    result.expected[replay].activation = bytes<float>(tokens[replay]);
    routed_ids[replay].assign(baseline[replay].ids.begin(),
                              baseline[replay].ids.end());
    shared_experts[replay] = {0};
    for (std::int64_t slot = 0; slot < q::kTopK; ++slot)
      routed_input[replay].insert(routed_input[replay].end(),
                                   tokens[replay].begin(), tokens[replay].end());
  }

  const auto projection = [&](std::size_t port, c::GgufQuantizedType type,
                               std::int64_t experts, std::int64_t outputs,
                               std::int64_t inputs, int seed,
                               const Replays &activation, const ExpertIds &ids) {
    auto &description = result.layout.constant_descs[port];
    description = c::gguf_grouped_weight_description(
        type, c::DataType::kFloat32, experts, outputs, inputs).value();
    std::vector<float> physical(experts * outputs * inputs);
    for (std::int64_t expert = 0; expert < experts; ++expert) {
      const auto expert_seed = seed + static_cast<int>(expert) * q::kExpertSeedStride;
      for (std::int64_t output = 0; output < outputs; ++output)
        physical[(expert * outputs + output) * inputs +
                 q::source_index(output, inputs, expert_seed)] =
            static_cast<float>(q::decode(q::coefficient(output, expert_seed)));
    }
    auto packed = d5_gguf_test::pack(type, physical, experts * outputs, inputs,
                                     description.required_bytes().value());
    result.constants[port] = std::move(packed.bytes);
    physical.clear();
    physical.shrink_to_fit();
    return project(packed.decoded, outputs, inputs, activation, ids);
  };
  const auto gated = [](Replays gate, const Replays &up) {
    for (std::size_t replay = 0; replay < gate.size(); ++replay)
      for (std::size_t i = 0; i < gate[replay].size(); ++i)
        gate[replay][i] = gate[replay][i] /
            (1.0F + std::exp(-gate[replay][i])) * up[replay][i];
    return gate;
  };
  const auto routed_type = fidelity ? c::GgufQuantizedType::kQ8_0
                                   : c::GgufQuantizedType::kIq4Xs;
  const auto gate = projection(2, routed_type, q::kRouterResidentExperts,
      q::kExpertHidden, q::kHidden, full::kRoutedSeed, routed_input, routed_ids);
  const auto up = projection(3, routed_type, q::kRouterResidentExperts,
      q::kExpertHidden, q::kHidden, full::kRoutedSeed + 1, routed_input, routed_ids);
  const auto routed = projection(4, routed_type, q::kRouterResidentExperts,
      q::kHidden, q::kExpertHidden, full::kRoutedSeed + 2, gated(gate, up), routed_ids);
  const auto shared_type = fidelity ? c::GgufQuantizedType::kQ8_0
                                   : c::GgufQuantizedType::kQ5K;
  const auto shared_gate = projection(5, shared_type, 1, q::kSharedExpertHidden,
      q::kHidden, full::kSharedSeed, tokens, shared_experts);
  const auto shared_up = projection(6, shared_type, 1, q::kSharedExpertHidden,
      q::kHidden, full::kSharedSeed + 1, tokens, shared_experts);
  const auto shared = projection(7, fidelity ? c::GgufQuantizedType::kQ8_0
                                            : c::GgufQuantizedType::kQ6K,
      1, q::kHidden, q::kSharedExpertHidden, full::kSharedSeed + 2,
      gated(shared_gate, shared_up), shared_experts);
  for (std::size_t replay = 0; replay < result.expected.size(); ++replay) {
    auto &out = result.expected[replay];
    out.routed = routed[replay];
    const auto score = [&](std::int32_t id) {
      return id == q::kTopK ? tokens[replay][20] + tokens[replay][21]
                           : tokens[replay][id];
    };
    double sum = 0.0;
    for (const auto id : out.ids)
      sum += std::exp(static_cast<double>(score(id)));
    double shared_score = 0.0;
    for (std::int64_t column = 0; column < q::kHidden; ++column)
      shared_score += static_cast<double>(shared_score_weight[column]) *
                      tokens[replay][column];
    const auto probability = 1.0 / (1.0 + std::exp(-shared_score));
    for (std::int64_t column = 0; column < q::kHidden; ++column) {
      double combined = 0.0;
      for (std::int64_t slot = 0; slot < q::kTopK; ++slot)
        combined += std::exp(static_cast<double>(score(out.ids[slot]))) / sum *
                    out.routed[slot * q::kHidden + column];
      out.combined[column] = static_cast<float>(combined);
      out.shared[column] = static_cast<float>(shared[replay][column] * probability);
      out.full[column] = out.combined[column] + out.shared[column];
    }
  }
  return result;
}
} // namespace d5_qwen38_24t_moe_profiles
