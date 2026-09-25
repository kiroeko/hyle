#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "d5_qwen38_gdn_fixture.h"
#include "../compute_provider_native/half_test_support.h"

namespace d5_qwen38_official_fixture {
namespace c = hyle::compute;
using BF16 = hyle::simd::bfloat16_t;

inline constexpr std::int64_t kHidden = 5120;
inline constexpr std::int64_t kFfnHidden = 17408;
inline constexpr std::int64_t kEmbeddingRows = 7;
inline constexpr std::int64_t kMtpInput = 2 * kHidden;
inline constexpr std::int64_t kQueryHeads = 24;
inline constexpr std::int64_t kKvHeads = 4;
inline constexpr std::int64_t kHeadDepth = 256;
inline constexpr std::int64_t kRotaryDimensions = 64;

inline c::WorkloadContext workload(c::WorkloadPhase phase,
                                   std::uint64_t region) {
  return {c::ModelArtifactProfile::create(
              d5_qwen38_gdn_fixture::kOfficialProfile)
              .value(),
          phase,
          {0x48594c454d4f444cULL, region, 1}};
}

inline BF16 encode(double value) { return half_test::encode<BF16>(value); }

inline double decode(BF16 value) { return half_test::decode(value); }

inline std::vector<BF16> values(std::size_t count, int seed,
                                double denominator = 64.0) {
  std::vector<BF16> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto value = static_cast<int>((i * 17 + seed * 13) % 61) - 30;
    result[i] = encode(static_cast<double>(value) / denominator);
  }
  return result;
}

inline std::vector<BF16> scales(std::size_t count, int seed = 0) {
  std::vector<BF16> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = encode(0.75 + static_cast<double>((i + seed) % 7) / 32.0);
  return result;
}

inline std::size_t source_index(std::size_t output, std::size_t input_count,
                                int seed) {
  return (output * 131 + static_cast<std::size_t>(seed) * 17) % input_count;
}

inline BF16 coefficient(std::size_t output, int seed) {
  const auto magnitude = 1 + static_cast<int>((output + seed) % 5);
  const auto sign = ((output + static_cast<std::size_t>(seed)) & 1U) ? -1 : 1;
  return encode(static_cast<double>(sign * magnitude) / 16.0);
}

// Safetensors stores these linear weights output-major. Tests expose a
// transposed TensorDesc, matching the real [input, output] MatMul view while
// keeping the full model inner widths. One deterministic non-zero per output
// makes the independent oracle linear without reducing the backend workload.
inline std::vector<BF16> sparse_output_major_weight(std::size_t outputs,
                                                    std::size_t inputs,
                                                    int seed) {
  std::vector<BF16> result(outputs * inputs);
  for (std::size_t output = 0; output < outputs; ++output)
    result[output * inputs + source_index(output, inputs, seed)] =
        coefficient(output, seed);
  return result;
}

inline std::vector<double>
sparse_projection(std::span<const BF16> input, std::size_t outputs, int seed,
                  bool round_to_bf16 = true) {
  std::vector<double> result(outputs);
  for (std::size_t output = 0; output < outputs; ++output) {
    const auto value =
        decode(input[source_index(output, input.size(), seed)]) *
        decode(coefficient(output, seed));
    result[output] = round_to_bf16 ? decode(encode(value)) : value;
  }
  return result;
}

inline std::vector<BF16> as_bf16(std::span<const double> values) {
  std::vector<BF16> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    result[i] = encode(values[i]);
  return result;
}

inline std::vector<double> rms_norm(std::span<const BF16> input,
                                    std::span<const BF16> scale,
                                    double epsilon = 1.0e-5) {
  double squares = 0;
  for (const auto value : input) {
    const auto decoded = decode(value);
    squares += decoded * decoded;
  }
  const auto inverse = 1.0 / std::sqrt(squares / input.size() + epsilon);
  std::vector<double> result(input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    result[i] = decode(encode(decode(input[i]) * inverse * decode(scale[i])));
  return result;
}

inline std::vector<double> ffn(std::span<const BF16> activation,
                               std::span<const BF16> scale, int seed) {
  const auto normalized = rms_norm(activation, scale);
  const auto normalized_bf16 = as_bf16(normalized);
  const auto gate = sparse_projection(normalized_bf16, kFfnHidden, seed);
  const auto up = sparse_projection(normalized_bf16, kFfnHidden, seed + 1);
  std::vector<BF16> gated(kFfnHidden);
  for (std::size_t i = 0; i < gated.size(); ++i) {
    const auto silu = gate[i] / (1.0 + std::exp(-gate[i]));
    gated[i] = encode(silu * up[i]);
  }
  return sparse_projection(gated, kHidden, seed + 2);
}

inline std::vector<double> dense_head(std::span<const BF16> input,
                                      std::span<const BF16> output_major,
                                      std::size_t rows) {
  std::vector<double> result(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0;
    for (std::size_t column = 0; column < input.size(); ++column)
      sum += decode(input[column]) *
             decode(output_major[row * input.size() + column]);
    result[row] = decode(encode(sum));
  }
  return result;
}

} // namespace d5_qwen38_official_fixture
