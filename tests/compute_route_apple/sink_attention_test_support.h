#pragma once

#include "tensor_test_support.h"
#include "hyle/compute/model_ops.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <limits>

namespace apple_tensor_test {
// Independent FP64 oracle for the existing five-port F32 Sink contract.
// The two nonzero Q/K channels and separable dyadic V keep full-output checks
// practical without deriving expected values from a provider result.
inline constexpr int kSinkDepth = 512;
inline constexpr float kSinkScale = 0.125F;

inline float sink_query(int token, int head, int channel, int seed) {
  return channel == 0
      ? float((token + seed) % 7 - 3) / 8 + float(head) / 256
      : float((head + seed) % 9 - 4) / 16;
}
inline float sink_key(int row, int channel, int seed) {
  return channel == 0 ? float((row + seed) % 11 - 5) / 16
                      : float((row + 2 * seed) % 13 - 6) / 32;
}
inline float sink_mask(int token, int row, int queries, int keys, int seed) {
  return token == 15 || token == 16 || token == queries - 1 ||
                 (token + seed) % 97 == 0 || row > keys - queries + token ||
                 (row + token + seed) % 5 == 0
      ? -std::numeric_limits<float>::infinity()
      : float((row + token + seed) % 7 - 3) / 16;
}
inline float sink_logit(int head, int seed) {
  return float((head + seed) % 9 - 4) / 4;
}

inline std::array<mx::array, 5> sink_ports(int queries, int keys, int heads, int seed) {
  std::vector<float> q(std::size_t(queries) * heads * kSinkDepth),
      k(std::size_t(keys) * kSinkDepth), v(k.size()),
      mask(std::size_t(queries) * keys), sinks(heads);
  for (int t = 0; t < queries; ++t)
    for (int h = 0; h < heads; ++h) {
      const auto offset = std::size_t(t * heads + h) * kSinkDepth;
      q[offset] = sink_query(t, h, 0, seed);
      q[offset + 1] = sink_query(t, h, 1, seed);
    }
  for (int s = 0; s < keys; ++s) {
    k[std::size_t(s) * kSinkDepth] = sink_key(s, 0, seed);
    k[std::size_t(s) * kSinkDepth + 1] = sink_key(s, 1, seed);
    const float factor = float((s + seed) % 17 - 8) / 16;
    for (int d = 0; d < kSinkDepth; ++d)
      v[std::size_t(s) * kSinkDepth + d] =
          factor * float(d % 7 - 3) / 32 + float(d % 11 - 5) / 64;
  }
  for (int t = 0; t < queries; ++t)
    for (int s = 0; s < keys; ++s)
      mask[std::size_t(t) * keys + s] = sink_mask(t, s, queries, keys, seed);
  for (int h = 0; h < heads; ++h) sinks[h] = sink_logit(h, seed);
  return {mx::array(q.data(), {1, queries, heads, kSinkDepth}, mx::float32),
      mx::array(k.data(), {1, keys, 1, kSinkDepth}, mx::float32),
      mx::array(v.data(), {1, keys, 1, kSinkDepth}, mx::float32),
      mx::array(mask.data(), {queries, keys}, mx::float32),
      mx::array(sinks.data(), {heads}, mx::float32)};
}

struct SinkExpectedRow {
  double weighted = 0;
  double mass = 0;
};
inline std::vector<SinkExpectedRow> sink_reference(int queries, int keys, int heads,
                                           int seed) {
  std::vector<SinkExpectedRow> result(std::size_t(queries) * heads);
  std::vector<double> scores(keys);
  for (int t = 0; t < queries; ++t)
    for (int h = 0; h < heads; ++h) {
      double maximum = sink_logit(h, seed);
      for (int s = 0; s < keys; ++s) {
        scores[s] = (double(sink_query(t, h, 0, seed)) * sink_key(s, 0, seed) +
                      double(sink_query(t, h, 1, seed)) * sink_key(s, 1, seed)) *
                kSinkScale + sink_mask(t, s, queries, keys, seed);
        maximum = std::max(maximum, scores[s]);
      }
      double denominator = std::exp(double(sink_logit(h, seed)) - maximum);
      double mass = 0, weighted = 0;
      for (int s = 0; s < keys; ++s) {
        const double weight = std::exp(scores[s] - maximum);
        denominator += weight;
        mass += weight;
        weighted += weight * (double((s + seed) % 17 - 8) / 16);
      }
      result[std::size_t(t) * heads + h] = {weighted / denominator,
                                                   mass / denominator};
    }
  return result;
}

inline std::uint64_t sink_output_hash(const mx::array &output) {
  const float *data = output.data<float>();
  std::uint64_t hash = 1469598103934665603ULL;
  for (int t = 0; t < output.shape(1); ++t)
    for (int h = 0; h < output.shape(2); ++h)
      for (int d = 0; d < kSinkDepth; ++d) {
        const auto offset = std::size_t(t) * output.strides(1) +
            std::size_t(h) * output.strides(2) +
            std::size_t(d) * output.strides(3);
        hash = (hash ^ std::bit_cast<std::uint32_t>(data[offset])) * 1099511628211ULL;
      }
  return hash;
}

inline void check_sink(const mx::array &output,
                 const std::vector<SinkExpectedRow> &expected) {
  ASSERT_EQ(output.dtype(), mx::float32);
  ASSERT_EQ(output.shape(3), kSinkDepth);
  ASSERT_EQ(expected.size(), std::size_t(output.shape(1)) * output.shape(2));
  const float *data = output.data<float>();
  bool finite = true;
  double maximum_error = 0, maximum_budget_fraction = 0;
  for (int t = 0; t < output.shape(1); ++t)
    for (int h = 0; h < output.shape(2); ++h) {
      const auto row = expected[std::size_t(t) * output.shape(2) + h];
      const auto offset = std::size_t(t) * output.strides(1) +
          std::size_t(h) * output.strides(2);
      for (int d = 0; d < kSinkDepth; ++d) {
        // All source values and both separable V terms are exactly dyadic F32.
        const double value = row.weighted * (double(d % 7 - 3) / 32) +
            row.mass * (double(d % 11 - 5) / 64);
        const double actual = data[offset + std::size_t(d) * output.strides(3)];
        finite &= std::isfinite(actual);
        const double error = std::abs(actual - value);
        maximum_error = std::max(maximum_error, error);
        maximum_budget_fraction = std::max(maximum_budget_fraction,
            error / (6e-6 * std::abs(value) + 2e-6));
      }
    }
  EXPECT_TRUE(finite);
  EXPECT_LE(maximum_budget_fraction, 1);
  std::printf("{\"kind\":\"sink_numeric\",\"Q\":%d,\"H\":%d,\"maximum_error\":%.17g,\"maximum_budget_fraction\":%.17g,\"finite\":%s}\n",
      output.shape(1), output.shape(2), maximum_error, maximum_budget_fraction,
      finite ? "true" : "false");
}

} // namespace apple_tensor_test
