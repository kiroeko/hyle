#pragma once

#include "original_role_input.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

// Fixed trained DSpark head roles. This test input helper has no routing or
// execution policy. The complete decoder/stage and LM weight are separate.
namespace dspark_original_roles {
inline constexpr std::size_t vocabulary = 129280, rank = 256, hidden = 4096;
inline constexpr auto w1_sha256 =
    "966bd0507046347754d0f4bf3addd6df6c179ff16243d63998258b3987e4b2f5";
inline constexpr auto w2_sha256 =
    "40ac7e99651c5c6aab8d2555ff65d247931f318414cf94baedc6da2d3bf7c175";
inline constexpr auto confidence_sha256 =
    "15c5d09c7842d0f947ce0afdc154691ead377e6263af71bd2f7cb34bd592f30f";

// unsloth/DeepSeek-V4-Flash-0731-GGUF
// @fbbb5b93fb787c21338159b0af3318bb3f4d9768,
// dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf. Each role is actually BF16.
struct Roles final {
  std::vector<std::uint16_t> w1, w2, confidence;
};
inline Roles load(const std::filesystem::path &markov_directory,
                  const std::filesystem::path &confidence_path) {
  return {original_role_input::read_bfloat(
              markov_directory / "markov_w1.weight.bf16", vocabulary * rank, w1_sha256),
          original_role_input::read_bfloat(
              markov_directory / "markov_w2.weight.bf16", vocabulary * rank, w2_sha256),
          original_role_input::read_bfloat(
              confidence_path, hidden + rank, confidence_sha256)};
}

// Independent FP64 Neumaier oracle on exact BF16 coefficients. The terms and
// reduction are never obtained from a provider or its quantization helper.
struct Dot final {
  double value = 0, sum_absolute_products = 0;
  bool exact_f32_products_with_normal_quantum = true;
};
inline Dot dot(std::span<const float> input,
               std::span<const std::uint16_t> weight) {
  if (input.size() != weight.size())
    throw std::runtime_error("original DSpark dot dimensions differ");
  double sum = 0, correction = 0, absolute = 0;
  bool normal_quantum = true;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const double product = double(input[i]) * original_role_input::widen(weight[i]);
    // This sufficient range gives every F32 product a quantum >=2^-126;
    // cancellation of these products cannot create an FTZ-sensitive term.
    normal_quantum = normal_quantum && std::isfinite(product) &&
        double(float(product)) == product &&
        (product == 0 || std::abs(product) >= 0x1p-103);
    const double next = sum + product;
    correction += std::abs(sum) >= std::abs(product)
        ? (sum - next) + product : (product - next) + sum;
    sum = next;
    absolute += std::abs(product);
  }
  return {sum + correction, absolute, normal_quantum};
}

struct HeadReference final {
  std::vector<std::int32_t> tokens;
  std::vector<std::uint16_t> markov;
  std::vector<float> logits, confidence;
  std::vector<double> bias_absolute_products, confidence_absolute_products;
  std::vector<double> exact_logits, logit_absolute_terms, exact_bias;
};

// LM logits are a declared external boundary, not invented trained LM weights.
// Raw collapsed hidden is distinct from the already-normalized LM input. Only
// each actual preceding selected token supplies the next row's Markov index.
inline HeadReference reference(const Roles &roles,
    std::span<const float> base_logits, std::span<const float> collapsed_hidden,
    std::int32_t first_token) {
  if (base_logits.empty() || base_logits.size() % vocabulary ||
      roles.w1.size() != vocabulary * rank || roles.w2.size() != vocabulary * rank ||
      roles.confidence.size() != hidden + rank || first_token < 0 ||
      std::size_t(first_token) >= vocabulary)
    throw std::runtime_error("original DSpark head dimensions or first token invalid");
  const auto rows = base_logits.size() / vocabulary;
  if (collapsed_hidden.size() != rows * hidden)
    throw std::runtime_error("original DSpark raw hidden dimensions differ");
  HeadReference result;
  result.tokens.push_back(first_token);
  result.markov.reserve(rows * rank);
  result.logits.reserve(rows * vocabulary);
  result.bias_absolute_products.reserve(rows * vocabulary);
  std::array<float, rank> embedding{};
  std::array<float, hidden + rank> confidence_input{};
  for (std::size_t row = 0; row < rows; ++row) {
    const auto selected = std::span(roles.w1).subspan(
        std::size_t(result.tokens.back()) * rank, rank);
    result.markov.insert(result.markov.end(), selected.begin(), selected.end());
    for (std::size_t column = 0; column < rank; ++column)
      embedding[column] = original_role_input::widen(selected[column]);
    std::copy_n(collapsed_hidden.begin() + row * hidden, hidden, confidence_input.begin());
    std::copy(embedding.begin(), embedding.end(), confidence_input.begin() + hidden);
    const auto score = dot(confidence_input, roles.confidence);
    result.confidence.push_back(float(score.value));
    result.confidence_absolute_products.push_back(score.sum_absolute_products);
    float maximum = -std::numeric_limits<float>::infinity();
    std::int32_t next_token = 0;
    for (std::size_t token = 0; token < vocabulary; ++token) {
      const auto bias = dot(embedding,
          std::span(roles.w2).subspan(token * rank, rank));
      const auto lm = base_logits[row * vocabulary + token];
      if (!bias.exact_f32_products_with_normal_quantum ||
          (lm != 0 && std::abs(double(lm)) < 0x1p-103) ||
          bias.sum_absolute_products + std::abs(double(lm)) >
              double(std::numeric_limits<float>::max()) / 2)
        throw std::runtime_error("DSpark logit oracle requires a separate underflow/overflow analysis");
      const float value = lm + float(bias.value);
      if (!std::isfinite(value))
        throw std::runtime_error("original DSpark reference logit is nonfinite");
      result.logits.push_back(value);
      result.bias_absolute_products.push_back(bias.sum_absolute_products);
      result.exact_logits.push_back(double(lm) + bias.value);
      result.logit_absolute_terms.push_back(std::abs(double(lm)) + bias.sum_absolute_products);
      result.exact_bias.push_back(bias.value);
      if (value > maximum) {
        maximum = value;
        next_token = static_cast<std::int32_t>(token);
      }
    }
    result.tokens.push_back(next_token);
  }
  return result;
}
} // namespace dspark_original_roles
