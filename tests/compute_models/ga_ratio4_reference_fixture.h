#pragma once

#include "nvfp4_reference_fixture.h"

#include <numbers>

// Independent scalar reference for the formal GA ratio4 indexer compressor.
// Starts at projected F32 wkv/wgate, not at a packed FP4 or F32 cache carrier.
// Formula source: the fixed formal GA model.py:285-383 and kernel.py:129-199
// in docs/evidence/research/deepseek-v4-ga-2026-10-19.zip. No product operator,
// quantization codec, tensor library or backend kernel is called by this file.
namespace ga_ratio4_reference_test {
namespace scalar = nvfp4_reference_test;

inline constexpr std::int64_t batches = 2, tokens = 8, depth = 128,
    rope_depth = 64, channels = 256, ratio = 4, state_rows = 8,
    cache_capacity = 3;
inline constexpr float norm_epsilon = 1.e-6F;
inline constexpr float pool_absolute_tolerance = 2.e-6F;
inline constexpr float pool_relative_tolerance = 2.e-6F;
inline constexpr std::uint32_t intermediate_bfloat_ulp_tolerance = 1;
inline constexpr float required_fp4_midpoint_margin = .125F;
inline constexpr std::array<float, 8> e2m1_magnitudes{
    0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
inline constexpr std::array<float, 7> e2m1_midpoints{
    .25F, .75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.F};

struct Inputs final {
  std::vector<float> kv, gate; // [B,8,256], original F32 projections.
  std::vector<float> ape;     // [4,256], original F32 position parameter.
  std::vector<float> gamma;   // [128], BF16 checkpoint values widened to F32.
  std::vector<float> frequencies; // [32], formal compressed YaRN frequencies.
};

inline float add(float a, float b) {
  return scalar::f32(double(a) + double(b));
}
inline float subtract(float a, float b) {
  return scalar::f32(double(a) - double(b));
}
inline float divide(float a, float b) {
  return scalar::f32(double(a) / double(b));
}

// Exact source configuration, not the ModelArgs default theta40000:
// theta160000, original_seq_len65536, factor16, beta_fast32/beta_slow1.
inline std::vector<float> formal_frequencies() {
  constexpr double base = 160000., original_length = 65536.;
  const auto correction = [](double rotations) {
    return double(rope_depth) *
        std::log(original_length /
                 (rotations * 2. * std::numbers::pi_v<double>)) /
        (2. * std::log(base));
  };
  const float low = float(std::max(0., std::floor(correction(32.))));
  const float high = float(std::min(double(rope_depth - 1),
                                  std::ceil(correction(1.))));
  const float upper = high == low ? add(high, .001F) : high;
  std::vector<float> result(std::size_t(rope_depth / 2));
  for (std::int64_t pair = 0; pair < rope_depth / 2; ++pair) {
    const auto exponent = divide(float(pair * 2), float(rope_depth));
    const auto power = scalar::f32(std::pow(base, double(exponent)));
    const auto original = divide(1.F, power);
    const auto ramp = std::clamp(divide(subtract(float(pair), low),
                                        subtract(upper, low)), 0.F, 1.F);
    const auto smooth = subtract(1.F, ramp);
    result[std::size_t(pair)] = add(
        scalar::multiply(divide(original, 16.F), subtract(1.F, smooth)),
        scalar::multiply(original, smooth));
  }
  return result;
}

inline Inputs make_inputs() {
  Inputs input;
  input.kv.resize(std::size_t(batches * tokens * channels));
  input.gate.resize(input.kv.size());
  input.ape.resize(std::size_t(ratio * channels));
  input.gamma.assign(std::size_t(depth), 1.125F);
  input.frequencies = formal_frequencies();
  // Fixed before backend execution, using the independent host planning script.
  // Distinct halves and non-BF16-representable F32 projections detect premature
  // narrowing. Low-rank channels keep every FP4 midpoint comfortably distant;
  // the nonzero suffix pair still exercises real RoPE at positions0 and4.
  constexpr std::int64_t seed = 15;
  for (std::int64_t phase = 0; phase < ratio; ++phase)
    for (std::int64_t half = 0; half < 2; ++half)
      for (std::int64_t d = 0; d < depth; ++d) {
        const auto local = d < 64 ? 0 : d - 63;
        const auto numerator =
            (phase * 149 + half * 211 + local * 13 + seed * 17) % 1031 - 515;
        input.ape[std::size_t(phase * channels + half * depth + d)] =
            divide(float(numerator), 1024.F);
      }
  for (std::int64_t batch = 0; batch < batches; ++batch)
    for (std::int64_t position = 0; position < tokens; ++position)
      for (std::int64_t half = 0; half < 2; ++half) {
        const auto alpha = divide(float(1049 + batch * 211 + position * 37 +
                                        half * 503 + seed * 13), 1024.F);
        const auto beta = divide(float(73 + batch * 19 + position * 11 -
                                       half * 23 + seed * 3), 1024.F);
        const auto delta = divide(float(-91 + batch * 13 - position * 7 +
                                        half * 17 - seed * 2), 1024.F);
        for (std::int64_t d = 0; d < depth; ++d) {
          const auto at = std::size_t((batch * tokens + position) * channels +
                                      half * depth + d);
          input.kv[at] = d < 64 ? alpha : d == 64 ? beta : d == 65 ? delta : 0.F;
          const auto local = d < 64 ? 0 : d - 63;
          const auto numerator =
              (position * 73 + batch * 31 + half * 97 + local * 7 + seed * 5) %
                  733 - 366;
          input.gate[at] = divide(float(numerator), 1024.F);
        }
      }
  return input;
}

inline std::vector<float> projected_slice(std::span<const float> all,
                                         std::int64_t start,
                                         std::int64_t count) {
  if (all.size() != std::size_t(batches * tokens * channels) ||
      start < 0 || count <= 0 || start > tokens - count)
    throw std::runtime_error("ratio4 projected slice is outside fixed input");
  std::vector<float> result(std::size_t(batches * count * channels));
  for (std::int64_t batch = 0; batch < batches; ++batch)
    std::copy_n(all.begin() + (batch * tokens + start) * channels,
                std::size_t(count * channels),
                result.begin() + batch * count * channels);
  return result;
}

inline std::vector<std::uint16_t> initial_cache() {
  return std::vector<std::uint16_t>(
      std::size_t(batches * cache_capacity * depth), scalar::bf16(-77.F));
}

struct Snapshot final {
  std::vector<float> kv_state, score_state; // [B,8,256], including unused rows.
  std::vector<std::uint16_t> cache;         // [B,3,128], including sentinel tail.
  std::vector<float> pooled_f32;            // Emitting steps only: [B,1,128].
  std::vector<std::uint16_t> pooled_bf16, norm_bf16, rope_bf16,
      hadamard_bf16, qat_bf16;              // Each [B,1,128], original dtype.
  std::int64_t start_pos = 0, token_count = 0, active_rows = 0;
  bool emitted = false;
  float minimum_fp4_midpoint_distance = std::numeric_limits<float>::infinity();
  std::size_t norm_early_rounding_differences = 0;
};

inline float fp4_scale(std::span<const std::uint16_t> group) {
  float maximum = std::ldexp(6.F, -126);
  for (const auto value : group)
    maximum = std::max(maximum, std::abs(scalar::wide(value)));
  // Source fast_round_scale(amax, F32(1/6)), including the F32 product before
  // inspecting the mantissa. No product quantization helper is used here.
  const auto scaled = scalar::multiply(maximum, scalar::f32(1. / 6.));
  const auto bits = std::bit_cast<std::uint32_t>(scaled);
  const int exponent = int((bits >> 23) & 255U) - 127 +
      ((bits & 0x7fffffU) != 0 ? 1 : 0);
  return std::ldexp(1.F, exponent);
}

inline std::uint16_t fp4_roundtrip(std::uint16_t input, float scale) {
  const auto value = scalar::wide(input);
  const auto magnitude = std::min(std::abs(divide(value, scale)), 6.F);
  std::size_t selected = 0;
  float distance = std::numeric_limits<float>::infinity();
  for (std::size_t code = 0; code < e2m1_magnitudes.size(); ++code) {
    const auto error = std::abs(magnitude - e2m1_magnitudes[code]);
    if (error < distance ||
        (error == distance && (code & 1U) == 0 && (selected & 1U) != 0)) {
      selected = code;
      distance = error;
    }
  }
  const auto code = (input & 0x8000U) != 0
      ? -e2m1_magnitudes[selected] : e2m1_magnitudes[selected];
  return scalar::bf16(scalar::multiply(code, scale));
}

// Value semantics are intentional: copy after p3 and independently replay
// p4..7 from the accepted oracle prefix. No backend result is fed back here.
class Reference final {
public:
  explicit Reference(Inputs input)
      : input_(std::move(input)),
        kv_state_(std::size_t(batches * state_rows * channels), 0.F),
        score_state_(kv_state_.size(),
                     -std::numeric_limits<float>::infinity()),
        cache_(initial_cache()) {
    if (input_.kv.size() != std::size_t(batches * tokens * channels) ||
        input_.gate.size() != input_.kv.size() ||
        input_.ape.size() != std::size_t(ratio * channels) ||
        input_.gamma.size() != std::size_t(depth) ||
        input_.frequencies.size() != std::size_t(rope_depth / 2))
      throw std::runtime_error("ratio4 reference input shape differs");
  }

  Snapshot advance(std::int64_t start, std::int64_t count) {
    if (start != next_position_ || count <= 0 || start > tokens - count ||
        !((start == 0 && count == 3) || (start >= 3 && count == 1)))
      throw std::runtime_error("ratio4 reference expects prefill3 then p3..7");
    Snapshot out;
    out.start_pos = start;
    out.token_count = count;
    for (auto position = start; position < start + count; ++position) {
      const auto phase = position % ratio;
      for (std::int64_t batch = 0; batch < batches; ++batch)
        for (std::int64_t channel = 0; channel < channels; ++channel) {
          const auto source = std::size_t((batch * tokens + position) * channels + channel);
          const auto target = state_offset(batch, ratio + phase, channel);
          kv_state_[target] = input_.kv[source];
          score_state_[target] = add(input_.gate[source],
              input_.ape[std::size_t(phase * channels + channel)]);
        }
      if (phase == ratio - 1) {
        emit(position, out);
        for (std::int64_t batch = 0; batch < batches; ++batch)
          for (std::int64_t row = 0; row < ratio; ++row)
            for (std::int64_t channel = 0; channel < channels; ++channel) {
              kv_state_[state_offset(batch, row, channel)] =
                  kv_state_[state_offset(batch, row + ratio, channel)];
              score_state_[state_offset(batch, row, channel)] =
                  score_state_[state_offset(batch, row + ratio, channel)];
            }
        active_rows_ = (position + 1) / ratio;
      }
    }
    next_position_ = start + count;
    out.kv_state = kv_state_;
    out.score_state = score_state_;
    out.cache = cache_;
    out.active_rows = active_rows_;
    return out;
  }

private:
  static std::size_t state_offset(std::int64_t batch, std::int64_t row,
                                  std::int64_t channel) {
    return std::size_t((batch * state_rows + row) * channels + channel);
  }

  float pool(std::int64_t batch, std::int64_t channel) const {
    std::array<float, state_rows> values{}, logits{}, exponential{};
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::int64_t row = 0; row < state_rows; ++row) {
      const auto component = channel + (row < ratio ? 0 : depth);
      values[std::size_t(row)] = kv_state_[state_offset(batch, row, component)];
      logits[std::size_t(row)] = score_state_[state_offset(batch, row, component)];
      maximum = std::max(maximum, logits[std::size_t(row)]);
    }
    float denominator = 0.F;
    for (std::size_t row = 0; row < exponential.size(); ++row) {
      exponential[row] = scalar::f32(
          std::exp(double(subtract(logits[row], maximum))));
      denominator = add(denominator, exponential[row]);
    }
    float result = 0.F;
    for (std::size_t row = 0; row < exponential.size(); ++row)
      result = add(result, scalar::multiply(values[row],
                                            divide(exponential[row], denominator)));
    return result;
  }

  void emit(std::int64_t position, Snapshot &out) {
    out.emitted = true;
    const auto count = std::size_t(batches * depth);
    out.pooled_f32.resize(count);
    out.pooled_bf16.resize(count);
    out.norm_bf16.resize(count);
    out.rope_bf16.resize(count);
    out.hadamard_bf16.resize(count);
    out.qat_bf16.resize(count);
    for (std::int64_t batch = 0; batch < batches; ++batch) {
      const auto base = std::size_t(batch * depth);
      float square_sum = 0.F;
      for (std::int64_t d = 0; d < depth; ++d) {
        const auto at = base + std::size_t(d);
        out.pooled_f32[at] = pool(batch, d);
        out.pooled_bf16[at] = scalar::bf16(out.pooled_f32[at]);
        const auto x = scalar::wide(out.pooled_bf16[at]);
        square_sum = add(square_sum, scalar::multiply(x, x));
      }
      const auto mean = divide(square_sum, float(depth));
      const auto inverse = scalar::f32(
          1. / std::sqrt(double(add(mean, norm_epsilon))));
      for (std::int64_t d = 0; d < depth; ++d) {
        const auto at = base + std::size_t(d);
        const auto normalized = scalar::multiply(
            scalar::wide(out.pooled_bf16[at]), inverse);
        out.norm_bf16[at] = scalar::bf16(
            scalar::multiply(input_.gamma[std::size_t(d)], normalized));
        out.norm_early_rounding_differences += out.norm_bf16[at] != scalar::bf16(
            scalar::multiply(input_.gamma[std::size_t(d)],
                             scalar::rounded_bf16(normalized)));
      }
      std::copy_n(out.norm_bf16.begin() + std::ptrdiff_t(base),
                  std::size_t(depth), out.rope_bf16.begin() + std::ptrdiff_t(base));
      const auto block_start = position + 1 - ratio;
      for (std::int64_t pair = 0; pair < rope_depth / 2; ++pair) {
        const auto at = base + std::size_t(depth - rope_depth + pair * 2);
        const auto angle = scalar::multiply(float(block_start),
            input_.frequencies[std::size_t(pair)]);
        const auto cosine = scalar::f32(std::cos(double(angle)));
        const auto sine = scalar::f32(std::sin(double(angle)));
        const auto a = scalar::wide(out.norm_bf16[at]);
        const auto b = scalar::wide(out.norm_bf16[at + 1]);
        out.rope_bf16[at] = scalar::bf16(subtract(
            scalar::multiply(a, cosine), scalar::multiply(b, sine)));
        out.rope_bf16[at + 1] = scalar::bf16(add(
            scalar::multiply(a, sine), scalar::multiply(b, cosine)));
      }
      std::array<float, depth> transform{};
      for (std::int64_t d = 0; d < depth; ++d)
        transform[std::size_t(d)] = scalar::wide(out.rope_bf16[base + std::size_t(d)]);
      for (std::size_t width = 1; width < transform.size(); width *= 2)
        for (std::size_t group = 0; group < transform.size(); group += width * 2)
          for (std::size_t lane = 0; lane < width; ++lane) {
            const auto a = transform[group + lane];
            const auto b = transform[group + lane + width];
            transform[group + lane] = add(a, b);
            transform[group + lane + width] = subtract(a, b);
          }
      const auto normalization = scalar::f32(1. / std::sqrt(double(depth)));
      for (std::int64_t d = 0; d < depth; ++d)
        out.hadamard_bf16[base + std::size_t(d)] = scalar::bf16(
            scalar::multiply(transform[std::size_t(d)], normalization));
      for (std::size_t group = 0; group < std::size_t(depth); group += 32) {
        const auto values = std::span(out.hadamard_bf16).subspan(base + group, 32);
        const auto scale = fp4_scale(values);
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
          const auto magnitude = std::abs(divide(scalar::wide(values[lane]), scale));
          for (const auto midpoint : e2m1_midpoints)
            out.minimum_fp4_midpoint_distance = std::min(
                out.minimum_fp4_midpoint_distance, std::abs(magnitude - midpoint));
          out.qat_bf16[base + group + lane] = fp4_roundtrip(values[lane], scale);
        }
      }
      const auto destination = std::size_t((batch * cache_capacity + position / ratio) * depth);
      std::copy_n(out.qat_bf16.begin() + std::ptrdiff_t(base),
                  std::size_t(depth), cache_.begin() + std::ptrdiff_t(destination));
    }
  }

  Inputs input_;
  std::vector<float> kv_state_, score_state_;
  std::vector<std::uint16_t> cache_;
  std::int64_t next_position_ = 0, active_rows_ = 0;
};

} // namespace ga_ratio4_reference_test
