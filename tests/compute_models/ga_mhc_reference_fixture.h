#pragma once

#include "nvfp4_reference_fixture.h"

// Independent scalar oracle for the formal DeepSeek V4 GA hyper-connection
// boundary. Formula: model.py:680-716 and kernel.py:372-438 in the fixed
// deepseek-v4-ga-2026-10-19 research archive. This file calls no mHC operator,
// tensor library, backend kernel or product reduction implementation.
namespace ga_mhc_reference_test {
namespace scalar = nvfp4_reference_test;

inline constexpr std::int64_t batches = 2, streams = 4, depth = 128;
inline constexpr std::int64_t flattened_depth = streams * depth;
inline constexpr std::int64_t mix_width = (2 + streams) * streams;
inline constexpr std::int32_t sinkhorn_iterations = 20;
inline constexpr float norm_epsilon = 1.e-6F, hc_epsilon = 1.e-6F;
// Fixed before backend runs. Named scalar F32 operations need not reproduce a
// vendor's FMA/reduction/transcendental tree bit for bit. BF16 endpoints permit
// one adjacent finite representable value; no relative-to-zero BF16 shortcut.
inline constexpr float absolute_tolerance = 2.e-6F;
inline constexpr float relative_tolerance = 2.e-6F;
inline constexpr std::uint32_t bfloat_ulp_tolerance = 1;

inline float add(float a, float b) {
  return scalar::f32(double(a) + double(b));
}
inline float subtract(float a, float b) {
  return scalar::f32(double(a) - double(b));
}
inline float divide(float a, float b) {
  return scalar::f32(double(a) / double(b));
}
inline float sigmoid(float value) {
  const auto exponential = scalar::f32(std::exp(-double(value)));
  return divide(1.F, add(1.F, exponential));
}

struct Inputs final {
  std::int64_t tokens = 0;
  std::vector<std::uint16_t> stream_bf16; // [B,S,4,128], original residual.
  std::vector<std::uint16_t> branch_bf16; // [B,S,128], independent branch.
  std::vector<float> fn, scale, base;    // [24,512], [3], [24], all F32.
  std::vector<float> head_fn, head_scale, head_base; // [4,512], [1], [4].
};

inline Inputs make_inputs(std::int64_t tokens, unsigned phase = 0) {
  if (tokens != 1 && tokens != 3)
    throw std::invalid_argument("GA mHC oracle covers S1 and S3");
  Inputs result;
  result.tokens = tokens;
  const auto rows = batches * tokens;
  result.stream_bf16.resize(std::size_t(rows * flattened_depth));
  result.branch_bf16.resize(std::size_t(rows * depth));
  result.fn.assign(std::size_t(mix_width * flattened_depth), 0.F);
  result.head_fn.assign(std::size_t(streams * flattened_depth), 0.F);
  result.scale = {1.173828125F, -.919921875F, 1.083984375F};
  result.base.resize(std::size_t(mix_width));
  result.head_scale = {1.04296875F};
  result.head_base = {-.31298828125F, .17626953125F,
                      .42724609375F, -.10888671875F};
  // Positive asymmetric streams keep post/head reductions away from zero;
  // variation across batch, token, stream, feature and generation detects
  // wrong flattening and stale state. Values are stored once as actual BF16.
  const auto generation = static_cast<std::int64_t>(phase % 7u);
  constexpr std::int64_t payload_seed = 12;
  for (std::int64_t row = 0; row != rows; ++row) {
    for (std::int64_t stream = 0; stream != streams; ++stream) {
      for (std::int64_t d = 0; d != depth; ++d) {
        const auto local = d % 17;
        const auto numerator = 89 +
            (row * 31 + stream * 67 + local * 13 + (local / 7) * 11 +
             generation * 43 + payload_seed * 17) % 173;
        result.stream_bf16[std::size_t(row * flattened_depth +
                                      stream * depth + d)] =
            scalar::bf16(divide(float(numerator), 128.F));
      }
    }
    for (std::int64_t d = 0; d != depth; ++d) {
      const auto numerator =
          (row * 23 + (d % 17) * 19 + generation * 47 + 17 +
           payload_seed * 13) % 97 - 38;
      result.branch_bf16[std::size_t(row * depth + d)] =
          scalar::bf16(divide(float(numerator), 256.F));
    }
  }
  // Twelve distinct cross-stream taps per projection make the independent
  // scalar reduction small without pretending the projection input was
  // normalized first. F32 parameter words are not BF16 checkpoint surrogates.
  for (std::int64_t output = 0; output != mix_width; ++output) {
    result.base[std::size_t(output)] =
        divide(float((output * 37 + 11) % 101 - 49), 128.F);
    for (std::int64_t tap = 0; tap != 12; ++tap) {
      const auto lane = (output * 29 + tap * 71 + tap * tap * 7) %
                        flattened_depth;
      const auto numerator = (output * 19 + tap * 23 + 5) % 61 - 29;
      result.fn[std::size_t(output * flattened_depth + lane)] =
          divide(float(numerator), 1024.F);
    }
  }
  for (std::int64_t output = 0; output != streams; ++output) {
    for (std::int64_t tap = 0; tap != 12; ++tap) {
      const auto lane = (output * 43 + tap * 59 + tap * tap * 3 + 17) %
                        flattened_depth;
      const auto numerator = (output * 31 + tap * 17 + 7) % 67 - 31;
      result.head_fn[std::size_t(output * flattened_depth + lane)] =
          divide(float(numerator), 1024.F);
    }
  }
  return result;
}

struct Snapshot final {
  std::vector<float> norm_factor;  // [B,S,1].
  std::vector<float> linear_mixes, mixes; // [B,S,24], before/after rsqrt.
  std::vector<float> pre, post;   // [B,S,4].
  std::vector<float> comb;        // [B,S,source_stream,destination_stream].
  std::vector<float> collapsed_f32; // Source F32 weighted stream reduction.
  std::vector<std::uint16_t> collapsed_bf16; // [B,S,128].
  std::vector<float> post_raw_f32; // [B,S,4,128], before source BF16 boundary.
  std::vector<std::uint16_t> expanded_bf16; // [B,S,4,128].
  std::vector<float> head_norm_factor; // [B,S,1].
  std::vector<float> head_linear_mixes, head_mixes, head_pre; // [B,S,4].
  std::vector<float> head_raw_f32; // [B,S,128].
  std::vector<std::uint16_t> head_bf16; // [B,S,128].
};

inline void validate(const Inputs &input,
                     std::span<const std::uint16_t> stream,
                     std::span<const std::uint16_t> branch) {
  const auto rows = batches * input.tokens;
  if ((input.tokens != 1 && input.tokens != 3) ||
      stream.size() != std::size_t(rows * flattened_depth) ||
      branch.size() != std::size_t(rows * depth) ||
      input.fn.size() != std::size_t(mix_width * flattened_depth) ||
      input.scale.size() != 3 || input.base.size() != std::size_t(mix_width) ||
      input.head_fn.size() != std::size_t(streams * flattened_depth) ||
      input.head_scale.size() != 1 ||
      input.head_base.size() != std::size_t(streams))
    throw std::invalid_argument("GA mHC oracle input shape mismatch");
}

// The two projection sites deliberately compute F.linear(x,F32_fn) first,
// then multiply that F32 result by rsqrt(mean(square(x))+epsilon). Moving the
// norm factor into each x lane changes the named rounding boundaries.
inline void project(std::span<const float> x, std::span<const float> weights,
                    std::int64_t rows, std::int64_t outputs,
                    std::vector<float> &factors,
                    std::vector<float> &linear,
                    std::vector<float> &normalized) {
  factors.resize(std::size_t(rows));
  linear.resize(std::size_t(rows * outputs));
  normalized.resize(linear.size());
  for (std::int64_t row = 0; row != rows; ++row) {
    float squares = 0.F;
    for (std::int64_t lane = 0; lane != flattened_depth; ++lane) {
      const auto value = x[std::size_t(row * flattened_depth + lane)];
      squares = add(squares, scalar::multiply(value, value));
    }
    const auto mean = divide(squares, float(flattened_depth));
    const auto factor = scalar::f32(
        1. / std::sqrt(double(add(mean, norm_epsilon))));
    factors[std::size_t(row)] = factor;
    for (std::int64_t output = 0; output != outputs; ++output) {
      float sum = 0.F;
      for (std::int64_t lane = 0; lane != flattened_depth; ++lane)
        sum = add(sum, scalar::multiply(
            x[std::size_t(row * flattened_depth + lane)],
            weights[std::size_t(output * flattened_depth + lane)]));
      const auto index = std::size_t(row * outputs + output);
      linear[index] = sum;
      normalized[index] = scalar::multiply(sum, factor);
    }
  }
}

inline std::array<float, 16> sinkhorn(std::span<const float> logits) {
  std::array<float, 16> result{};
  // One row softmax + epsilon, then one column normalization. The initial
  // softmax denominator has NO epsilon in the formal TileLang source.
  for (std::int64_t src = 0; src != streams; ++src) {
    const auto offset = std::size_t(src * streams);
    auto maximum = logits[offset];
    for (std::int64_t dst = 1; dst != streams; ++dst)
      maximum = std::max(maximum, logits[offset + std::size_t(dst)]);
    float denominator = 0.F;
    for (std::int64_t dst = 0; dst != streams; ++dst) {
      const auto index = offset + std::size_t(dst);
      result[index] = scalar::f32(
          std::exp(double(subtract(logits[index], maximum))));
      denominator = add(denominator, result[index]);
    }
    for (std::int64_t dst = 0; dst != streams; ++dst) {
      const auto index = offset + std::size_t(dst);
      result[index] = add(divide(result[index], denominator), hc_epsilon);
    }
  }
  const auto columns = [&]() {
    for (std::int64_t dst = 0; dst != streams; ++dst) {
      float denominator = 0.F;
      for (std::int64_t src = 0; src != streams; ++src)
        denominator = add(denominator,
                          result[std::size_t(src * streams + dst)]);
      denominator = add(denominator, hc_epsilon);
      for (std::int64_t src = 0; src != streams; ++src) {
        const auto index = std::size_t(src * streams + dst);
        result[index] = divide(result[index], denominator);
      }
    }
  };
  columns();
  for (std::int32_t iteration = 1; iteration != sinkhorn_iterations;
       ++iteration) {
    for (std::int64_t src = 0; src != streams; ++src) {
      float denominator = 0.F;
      for (std::int64_t dst = 0; dst != streams; ++dst)
        denominator = add(denominator,
                          result[std::size_t(src * streams + dst)]);
      denominator = add(denominator, hc_epsilon);
      for (std::int64_t dst = 0; dst != streams; ++dst) {
        const auto index = std::size_t(src * streams + dst);
        result[index] = divide(result[index], denominator);
      }
    }
    columns();
  }
  return result;
}

inline void head(const Inputs &input, std::span<const float> expanded,
                 Snapshot &result) {
  const auto rows = batches * input.tokens;
  project(expanded, input.head_fn, rows, streams, result.head_norm_factor,
          result.head_linear_mixes, result.head_mixes);
  result.head_pre.resize(std::size_t(rows * streams));
  result.head_raw_f32.resize(std::size_t(rows * depth));
  result.head_bf16.resize(result.head_raw_f32.size());
  for (std::int64_t row = 0; row != rows; ++row) {
    for (std::int64_t stream = 0; stream != streams; ++stream) {
      const auto index = std::size_t(row * streams + stream);
      result.head_pre[index] = add(sigmoid(add(
          scalar::multiply(result.head_mixes[index], input.head_scale[0]),
          input.head_base[std::size_t(stream)])), hc_epsilon);
    }
    for (std::int64_t d = 0; d != depth; ++d) {
      float sum = 0.F;
      for (std::int64_t stream = 0; stream != streams; ++stream)
        sum = add(sum, scalar::multiply(
            result.head_pre[std::size_t(row * streams + stream)],
            expanded[std::size_t(row * flattened_depth + stream * depth + d)]));
      const auto index = std::size_t(row * depth + d);
      result.head_raw_f32[index] = sum;
      result.head_bf16[index] = scalar::bf16(sum);
    }
  }
}

inline Snapshot reference(
    const Inputs &input,
    std::span<const std::uint16_t> stream_bf16 = {},
    std::span<const std::uint16_t> branch_bf16 = {}) {
  if (stream_bf16.empty()) stream_bf16 = input.stream_bf16;
  if (branch_bf16.empty()) branch_bf16 = input.branch_bf16;
  validate(input, stream_bf16, branch_bf16);
  const auto rows = batches * input.tokens;
  Snapshot result;
  std::vector<float> stream(stream_bf16.size());
  for (std::size_t i = 0; i != stream.size(); ++i)
    stream[i] = scalar::wide(stream_bf16[i]);
  project(stream, input.fn, rows, mix_width, result.norm_factor,
          result.linear_mixes, result.mixes);
  result.pre.resize(std::size_t(rows * streams));
  result.post.resize(result.pre.size());
  result.comb.resize(std::size_t(rows * streams * streams));
  result.collapsed_f32.resize(std::size_t(rows * depth));
  result.collapsed_bf16.resize(result.collapsed_f32.size());
  result.post_raw_f32.resize(stream.size());
  result.expanded_bf16.resize(stream.size());
  for (std::int64_t row = 0; row != rows; ++row) {
    const auto mix_offset = std::size_t(row * mix_width);
    for (std::int64_t lane = 0; lane != streams; ++lane) {
      const auto index = std::size_t(row * streams + lane);
      result.pre[index] = add(sigmoid(add(
          scalar::multiply(result.mixes[mix_offset + std::size_t(lane)],
                           input.scale[0]),
          input.base[std::size_t(lane)])), hc_epsilon);
      result.post[index] = scalar::multiply(2.F, sigmoid(add(
          scalar::multiply(
              result.mixes[mix_offset + std::size_t(streams + lane)],
              input.scale[1]),
          input.base[std::size_t(streams + lane)])));
    }
    std::array<float, 16> logits{};
    for (std::int64_t lane = 0; lane != streams * streams; ++lane) {
      const auto source_lane = std::size_t(2 * streams + lane);
      logits[std::size_t(lane)] = add(scalar::multiply(
          result.mixes[mix_offset + source_lane], input.scale[2]),
          input.base[source_lane]);
    }
    const auto comb = sinkhorn(logits);
    for (std::size_t lane = 0; lane != comb.size(); ++lane)
      result.comb[std::size_t(row * streams * streams) + lane] = comb[lane];
    for (std::int64_t d = 0; d != depth; ++d) {
      float collapsed = 0.F;
      for (std::int64_t src = 0; src != streams; ++src)
        collapsed = add(collapsed, scalar::multiply(
            result.pre[std::size_t(row * streams + src)],
            stream[std::size_t(row * flattened_depth + src * depth + d)]));
      const auto output_index = std::size_t(row * depth + d);
      result.collapsed_f32[output_index] = collapsed;
      result.collapsed_bf16[output_index] = scalar::bf16(collapsed);
      for (std::int64_t dst = 0; dst != streams; ++dst) {
        // Formal source: reduce residual terms first, then add the separate
        // post*branch term. comb's FIRST index is the source stream.
        float residual_sum = 0.F;
        for (std::int64_t src = 0; src != streams; ++src)
          residual_sum = add(residual_sum, scalar::multiply(
              comb[std::size_t(src * streams + dst)],
              stream[std::size_t(row * flattened_depth + src * depth + d)]));
        const auto value = add(scalar::multiply(
            result.post[std::size_t(row * streams + dst)],
            scalar::wide(branch_bf16[output_index])), residual_sum);
        const auto index = std::size_t(row * flattened_depth + dst * depth + d);
        result.post_raw_f32[index] = value;
        result.expanded_bf16[index] = scalar::bf16(value);
      }
    }
  }
  std::vector<float> expanded(result.expanded_bf16.size());
  for (std::size_t i = 0; i != expanded.size(); ++i)
    expanded[i] = scalar::wide(result.expanded_bf16[i]);
  head(input, expanded, result);
  return result;
}

struct Counterfactuals final {
  std::size_t transposed_comb_words = 0;
  std::size_t early_bfloat_pre_words = 0;
  std::size_t omitted_post_boundary_head_words = 0;
  float transposed_comb_maximum_error = 0.F;
};

// Payload witnesses, independent of actual backend output. They guard against
// accidentally selecting a symmetric payload which cannot detect wrong stream
// direction or silently deleting a required BF16 storage boundary.
inline Counterfactuals counterfactuals(
    const Inputs &input, const Snapshot &expected,
    std::span<const std::uint16_t> stream_bf16 = {},
    std::span<const std::uint16_t> branch_bf16 = {}) {
  if (stream_bf16.empty()) stream_bf16 = input.stream_bf16;
  if (branch_bf16.empty()) branch_bf16 = input.branch_bf16;
  validate(input, stream_bf16, branch_bf16);
  Counterfactuals result;
  const auto rows = batches * input.tokens;
  for (std::int64_t row = 0; row != rows; ++row) {
    for (std::int64_t d = 0; d != depth; ++d) {
      float early = 0.F;
      for (std::int64_t src = 0; src != streams; ++src) {
        const auto term = scalar::rounded_bf16(scalar::multiply(
            scalar::rounded_bf16(expected.pre[std::size_t(row * streams + src)]),
            scalar::wide(stream_bf16[std::size_t(
                row * flattened_depth + src * depth + d)])));
        early = scalar::rounded_bf16(add(early, term));
      }
      const auto output_index = std::size_t(row * depth + d);
      if (scalar::bf16(early) != expected.collapsed_bf16[output_index])
        ++result.early_bfloat_pre_words;
      for (std::int64_t dst = 0; dst != streams; ++dst) {
        float residual_sum = 0.F;
        for (std::int64_t src = 0; src != streams; ++src)
          residual_sum = add(residual_sum, scalar::multiply(
              expected.comb[std::size_t(row * streams * streams +
                                        dst * streams + src)],
              scalar::wide(stream_bf16[std::size_t(
                  row * flattened_depth + src * depth + d)])));
        const auto wrong = add(scalar::multiply(
            expected.post[std::size_t(row * streams + dst)],
            scalar::wide(branch_bf16[output_index])), residual_sum);
        const auto index = std::size_t(row * flattened_depth + dst * depth + d);
        if (scalar::bf16(wrong) != expected.expanded_bf16[index])
          ++result.transposed_comb_words;
        result.transposed_comb_maximum_error = std::max(
            result.transposed_comb_maximum_error,
            std::abs(subtract(wrong, expected.post_raw_f32[index])));
      }
    }
  }
  Snapshot omitted_boundary;
  head(input, expected.post_raw_f32, omitted_boundary);
  for (std::size_t i = 0; i != expected.head_bf16.size(); ++i)
    if (omitted_boundary.head_bf16[i] != expected.head_bf16[i])
      ++result.omitted_post_boundary_head_words;
  return result;
}

} // namespace ga_mhc_reference_test
