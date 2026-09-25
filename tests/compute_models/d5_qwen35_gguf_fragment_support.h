#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace d5_qwen35_gguf_fragment {

inline std::vector<float> scale(std::size_t width) {
  std::vector<float> result(width);
  for (std::size_t index = 0; index < width; ++index)
    result[index] = 0.875F + static_cast<float>((index * 13 + 5) % 29) / 128.0F;
  return result;
}

inline std::vector<float> rms_norm(std::span<const float> input,
                                   std::span<const float> weights,
                                   float epsilon = 1.0e-5F) {
  EXPECT_EQ(input.size(), weights.size());
  double square_sum = 0.0;
  for (const auto value : input)
    square_sum += static_cast<double>(value) * value;
  const auto inverse =
      1.0 / std::sqrt(square_sum / static_cast<double>(input.size()) +
                      static_cast<double>(epsilon));
  std::vector<float> result(input.size());
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] =
        static_cast<float>(static_cast<double>(input[index]) * inverse *
                           static_cast<double>(weights[index]));
  return result;
}

inline std::vector<float> rms_norm_rows(std::span<const float> input,
                                        std::span<const float> weights,
                                        std::size_t width,
                                        float epsilon = 1.0e-5F) {
  EXPECT_EQ(input.size() % width, 0U);
  EXPECT_EQ(weights.size(), width);
  std::vector<float> result(input.size());
  for (std::size_t offset = 0; offset < input.size(); offset += width) {
    const auto row = rms_norm(input.subspan(offset, width), weights, epsilon);
    std::copy(row.begin(), row.end(),
              result.begin() + static_cast<std::ptrdiff_t>(offset));
  }
  return result;
}

inline std::vector<float> rope_half_split(std::span<const float> input,
                                          std::size_t depth,
                                          std::int32_t position,
                                          std::span<const float> frequencies) {
  EXPECT_EQ(input.size() % depth, 0U);
  EXPECT_EQ(frequencies.size(), depth / 2);
  std::vector<float> result(input.begin(), input.end());
  for (std::size_t offset = 0; offset < input.size(); offset += depth) {
    for (std::size_t pair = 0; pair < depth / 2; ++pair) {
      const auto first = offset + pair;
      const auto second = first + depth / 2;
      const auto angle = static_cast<float>(position) * frequencies[pair];
      const auto cosine = std::cos(angle);
      const auto sine = std::sin(angle);
      result[first] = input[first] * cosine - input[second] * sine;
      result[second] = input[first] * sine + input[second] * cosine;
    }
  }
  return result;
}

inline std::vector<float> causal_conv(std::span<const float> samples,
                                      std::span<const float> weights,
                                      std::size_t channels,
                                      std::size_t kernel) {
  EXPECT_EQ(samples.size(), channels * kernel);
  EXPECT_EQ(weights.size(), channels * kernel);
  std::vector<float> result(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    double sum = 0.0;
    for (std::size_t tap = 0; tap < kernel; ++tap)
      sum += static_cast<double>(samples[tap * channels + channel]) *
             weights[channel * kernel + tap];
    result[channel] = static_cast<float>(sum);
  }
  return result;
}

struct GdnResult final {
  std::vector<float> scores;
  std::vector<float> state;
};

inline GdnResult
gated_delta_net(std::span<const float> query, std::span<const float> key,
                std::span<const float> value, std::span<const float> log_decay,
                std::span<const float> beta, std::span<const float> state,
                std::size_t heads, std::size_t depth) {
  EXPECT_EQ(query.size(), heads * depth);
  EXPECT_EQ(key.size(), heads * depth);
  EXPECT_EQ(value.size(), heads * depth);
  EXPECT_EQ(log_decay.size(), heads);
  EXPECT_EQ(beta.size(), heads);
  EXPECT_EQ(state.size(), heads * depth * depth);
  std::vector<double> next_state(state.begin(), state.end());
  std::vector<double> scores(query.size());
  for (std::size_t head = 0; head < heads; ++head) {
    const auto vector = head * depth;
    const auto matrix = head * depth * depth;
    const auto retain = std::exp(static_cast<double>(log_decay[head]));
    for (std::size_t column = 0; column < depth; ++column) {
      const auto row = matrix + column * depth;
      double projected = 0.0;
      for (std::size_t key_column = 0; key_column < depth; ++key_column) {
        next_state[row + key_column] *= retain;
        projected += next_state[row + key_column] * key[vector + key_column];
      }
      const auto delta = static_cast<double>(beta[head]) *
                         (value[vector + column] - projected);
      double score = 0.0;
      for (std::size_t key_column = 0; key_column < depth; ++key_column) {
        next_state[row + key_column] += delta * key[vector + key_column];
        score += next_state[row + key_column] * query[vector + key_column];
      }
      scores[vector + column] = score / std::sqrt(double(depth));
    }
  }
  GdnResult result{std::vector<float>(scores.size()),
                   std::vector<float>(next_state.size())};
  std::transform(scores.begin(), scores.end(), result.scores.begin(),
                 [](double value) { return static_cast<float>(value); });
  std::transform(next_state.begin(), next_state.end(), result.state.begin(),
                 [](double value) { return static_cast<float>(value); });
  return result;
}

inline std::vector<float> silu_gate(std::span<const float> gate,
                                    std::span<const float> up) {
  EXPECT_EQ(gate.size(), up.size());
  std::vector<float> result(gate.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto value = static_cast<double>(gate[index]);
    result[index] =
        static_cast<float>((value / (1.0 + std::exp(-value))) * up[index]);
  }
  return result;
}

inline void expect_close(std::span<const float> actual,
                         std::span<const float> expected, float absolute,
                         float relative = 0.0F) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_NEAR(actual[index], expected[index],
                absolute + relative * std::abs(expected[index]))
        << index;
}

} // namespace d5_qwen35_gguf_fragment
