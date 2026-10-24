#pragma once

#include <ggml.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen27_original_nextn_reference.h"

// Independent original-byte local equations for MAIN Q24. Every nonlinear
// or projection reference begins with its actual observed predecessor. The
// fixed one-token GGML CUDA dispatch is MMVQ, including ten selected slots;
// its Q4/Q5 minimum uses the integer Q8 code sum, not the MMQ half-sum path.
namespace qwen24_original_nextn_reference {
namespace base = qwen27_original_nextn_reference;
namespace scalar = dspark_original_shared_reference::detail;
using ActivationCodec = base::ActivationCodec;
inline constexpr std::size_t kExperts = 512, kTopK = 10;
inline constexpr float kMinimumDenominator = 0x1p-14F;

namespace detail {
inline std::size_t block_bytes(ggml_type type) {
  if (type == GGML_TYPE_IQ4_XS) return 136;
  return type == GGML_TYPE_Q5_K ? 176 : base::detail::block_bytes(type);
}
inline base::detail::WeightGroup weight(std::span<const std::byte> row,
    ggml_type type, std::size_t group) {
  if (type == GGML_TYPE_IQ4_XS) {
    // Independent byte decoding: F16 d, LE16 scale high bits, four packed
    // scale-low bytes, then eight groups of sixteen packed nonlinear codes.
    // Every group is low-nibble columns0..15/high-nibble columns16..31.
    static constexpr std::array<int, 16> values{
        -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const auto block = row.subspan(group / 8 * 136, 136);
    const auto local = group % 8;
    const auto high = base::detail::byte(block, 2) | (base::detail::byte(block, 3) << 8U);
    const int scale = int(((base::detail::byte(block, 4 + local / 2) >> (4 * (local % 2))) & 15U) |
        (((high >> (2 * local)) & 3U) << 4U)) - 32;
    base::detail::WeightGroup result;
    result.scale = scalar::scale(block, 0);
    for (std::size_t i = 0; i < 32; ++i) {
      const auto code = (base::detail::byte(block, 8 + local * 16 + i % 16) >>
          (i >= 16 ? 4U : 0U)) & 15U;
      result.codes[i] = scale * values[code];
    }
    return result;
  }
  if (type != GGML_TYPE_Q5_K) return base::detail::weight(row, type, group);
  const auto block = row.subspan(group / 8 * 176, 176);
  const auto local = group % 8;
  const auto [scale, minimum] = base::detail::q4_scale(block, local);
  base::detail::WeightGroup result;
  result.scale = scalar::scale(block, 0); result.minimum = scalar::scale(block, 2);
  for (std::size_t i = 0; i < 32; ++i) {
    const auto low = (base::detail::byte(block, 48 + local / 2 * 32 + i) >>
        (local % 2 ? 4U : 0U)) & 15U;
    const auto high = (base::detail::byte(block, 16 + i) >> local) & 1U;
    result.codes[i] = int(low | (high << 4U)) * scale;
    result.offsets[i] = -minimum;
  }
  return result;
}
inline bool exact(float actual, float expected, const char *role, std::size_t index) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
    ADD_FAILURE() << role << '[' << index << "] actual=" << actual << " expected=" << expected;
    return false;
  }
  return true;
}
// The existing semantic local nonlinear gate, distinct from every analytic
// native-codec projection/reduction certificate. It is never data-fitted.
inline bool local(float actual, float expected, const char *role, std::size_t index) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(double(actual) - expected) > 8.0e-5 + 4.0e-5 * std::abs(double(expected))) {
    ADD_FAILURE() << role << '[' << index << "] actual=" << actual << " local expected=" << expected;
    return false;
  }
  return true;
}
inline double sigmoid(double value) {
  if (value >= 0) return 1.0 / (1.0 + std::exp(-value));
  const auto exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}
inline float f32(std::span<const std::byte> bytes, std::size_t index) {
  std::uint32_t raw = 0;
  for (std::size_t i = 0; i < 4; ++i)
    raw |= std::to_integer<std::uint32_t>(bytes[index * 4 + i]) << (i * 8U);
  return std::bit_cast<float>(raw);
}
struct Reduced final { double center, bound; };
template<class Term>
Reduced sum(std::size_t count, Term &&term) {
  if (!count) throw std::runtime_error("original Q24 empty reduction");
  scalar::Sum reference; double absolute = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = term(i);
    if (!std::isfinite(value) || (value != 0 && std::abs(value) < 0x1p-103))
      throw std::runtime_error("original Q24 reduction needs separate underflow analysis");
    reference.add(value); absolute = scalar::add_upper(absolute, std::abs(value));
  }
  if (!std::isfinite(absolute) || absolute > double(std::numeric_limits<float>::max()) / 4)
    throw std::runtime_error("original Q24 reduction needs separate overflow analysis");
  const auto center = reference.value();
  const auto oracle = scalar::add_upper(base::detail::distance(center, reference.plain),
      scalar::multiply_upper(scalar::gamma(count, 0x1p-53), absolute));
  const auto native = scalar::multiply_upper(scalar::gamma(count, 0x1p-24), absolute);
  return {center, scalar::add_upper(oracle, native)};
}
inline bool reduced(float actual, const Reduced &reference, const char *role, std::size_t index) {
  const auto error = base::detail::distance(actual, reference.center);
  if (!std::isfinite(actual) || error > reference.bound) {
    ADD_FAILURE() << role << '[' << index << "] actual=" << actual << " F64=" << reference.center
        << " error=" << error << " analytic F32 bound=" << reference.bound;
    return false;
  }
  return true;
}
} // namespace detail

inline bool expect_projection(std::span<const float> actual, std::span<const std::byte> packed,
    std::size_t outputs, std::size_t width, ggml_type type, std::span<const float> actual_input,
    ActivationCodec codec, const char *role, bool full, std::span<const std::size_t> extra_rows = {}) {
  if (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_NVFP4)
    return base::expect_projection(actual, packed, outputs, width, type, actual_input, codec, role, full, extra_rows);
  if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K && type != GGML_TYPE_IQ4_XS) || !outputs || !width || width % 256 ||
      actual.size() != outputs || actual_input.size() != width ||
      outputs > std::numeric_limits<std::size_t>::max() / (width / 256) / detail::block_bytes(type) ||
      packed.size() != outputs * (width / 256) * detail::block_bytes(type) || std::fegetround() != FE_TONEAREST) {
    ADD_FAILURE() << role << " original Q24 projection extent/codec/rounding"; return false;
  }
  // Q4/Q5/IQ4_XS CPU use AQ8_K; CUDA MMVQ uses AQ8_1. Reuse precisely the
  // BQ actual-F32 quantizer and its conditional strict/fast division envelope.
  const auto activation = base::detail::activation(actual_input, GGML_TYPE_Q6_K, codec);
  const auto row_bytes = width / 256 * detail::block_bytes(type), groups = width / 32;
  const auto selected = base::detail::rows(outputs, full, extra_rows);
  const double g32 = scalar::gamma(width / 4 + 65, 0x1p-24);
  const double g64 = scalar::gamma(2 * groups + 2, 0x1p-53);
  double maximum_error = 0, maximum_ratio = 0, maximum_ambiguity = 0;
  for (const auto output : selected) {
    const auto row = packed.subspan(output * row_bytes, row_bytes);
    scalar::Sum sum; double absolute = 0, ambiguity = 0;
    for (std::size_t group = 0; group < groups; ++group) {
      const auto w = detail::weight(row, type, group); const auto &a = activation[group];
      const double largest_scale = std::max(std::abs(double(a.scale_low)), std::abs(double(a.scale_high)));
      const double scale_error = std::max(std::abs(double(a.scale_low) - a.scale), std::abs(double(a.scale_high) - a.scale));
      const auto component = [&](double scale, const std::array<int, 32> &codes) {
        if (scale == 0) return;
        base::detail::range(scale, a);
        int dot = 0, uncertain = 0, maximum_absolute = 0;
        for (std::size_t i = 0; i < 32; ++i) {
          dot += codes[i] * a.codes[i];
          uncertain += std::abs(codes[i]) * std::max(std::abs(a.low[i] - a.codes[i]), std::abs(a.high[i] - a.codes[i]));
          maximum_absolute += std::abs(codes[i]) * std::max(std::abs(a.low[i]), std::abs(a.high[i]));
        }
        sum.add(scale * double(a.scale) * dot);
        absolute = scalar::add_upper(absolute, scalar::multiply_upper(std::abs(scale),
            scalar::multiply_upper(largest_scale, double(maximum_absolute))));
        ambiguity = scalar::add_upper(ambiguity, scalar::multiply_upper(std::abs(scale),
            scalar::add_upper(scalar::multiply_upper(scale_error, std::abs(double(dot))),
                scalar::multiply_upper(largest_scale, double(uncertain)))));
      };
      // Native CPU/CUDA keep d*scale*q and -dmin*min components separate in
      // F32. Their cancellation must not erase the arithmetic sumabs bound.
      // CUDA MMVQ's minimum is d8*sum(integer codes), unlike MMQ's ds.y.
      // IQ4_XS has no minimum. Its signed6-bit scale multiplies the signed
      // nonlinear LUT integer before F32 conversion, with no bsums term.
      // CPU AVX2's eight lanes each gather at most32 products, and CUDA one
      // group gathers32: 32*127*127*32=16,516,096 <2^24. Both the integer
      // reduction and its conversion are exact; gamma(K/4+65) covers the
      // actual scale multiplication/FMA/tree reductions above that boundary.
      component(w.scale, w.codes); component(w.minimum, w.offsets);
    }
    if (!std::isfinite(absolute) || absolute > double(std::numeric_limits<float>::max()) / 4)
      throw std::runtime_error("original Q24 packed dot needs separate overflow analysis");
    const auto center = sum.value();
    const auto oracle = scalar::add_upper(base::detail::distance(center, sum.plain), scalar::multiply_upper(g64, absolute));
    const auto arithmetic = scalar::multiply_upper(g32, absolute);
    const auto bound = scalar::add_upper(ambiguity, scalar::add_upper(oracle, arithmetic));
    const auto error = base::detail::distance(actual[output], center);
    if (!std::isfinite(actual[output]) || error > bound) {
      ADD_FAILURE() << role << '[' << output << "] actual=" << actual[output] << " F64=" << center
          << " error=" << error << " code/scale ambiguity=" << ambiguity << " nativeF32=" << arithmetic
          << " separate-component sumabs=" << absolute; return false;
    }
    maximum_error = std::max(maximum_error, error);
    maximum_ratio = std::max(maximum_ratio, bound ? error / bound : 0.0);
    maximum_ambiguity = std::max(maximum_ambiguity, ambiguity);
  }
  std::printf("Q24 original %s codec=%d A=%d full=%d checked=%zu K=%zu maxAbs=%.17g maxRatio=%.17g maxAmbiguity=%.17g\n",
      role, int(type), int(codec), int(full), selected.size(), width, maximum_error, maximum_ratio, maximum_ambiguity);
  return true;
}

struct PackedMatrix final {
  std::span<const std::byte> bytes;
  std::size_t outputs, width;
  ggml_type type;
};
template<class Source>
bool expect_selected_projection(std::span<const float> actual, std::span<const float> actual_input,
    std::span<const std::int32_t> global_ids, Source &&source, ActivationCodec codec, const char *role, bool full) {
  if (global_ids.size() != kTopK) { ADD_FAILURE() << role << " actual selected count"; return false; }
  for (const auto id : global_ids)
    if (id < 0 || std::size_t(id) >= kExperts) { ADD_FAILURE() << role << " original global expert ID"; return false; }
  const auto first = source(global_ids.front());
  if (!first.outputs || !first.width || first.outputs > std::numeric_limits<std::size_t>::max() / kTopK ||
      first.width > std::numeric_limits<std::size_t>::max() / kTopK || actual.size() != kTopK * first.outputs ||
      (actual_input.size() != first.width && actual_input.size() != kTopK * first.width)) {
    ADD_FAILURE() << role << " original selected projection extent"; return false;
  }
  for (std::size_t slot = 0; slot < kTopK; ++slot) {
    const auto matrix = source(global_ids[slot]);
    if (matrix.outputs != first.outputs || matrix.width != first.width || matrix.type != first.type) {
      ADD_FAILURE() << role << " inconsistent original expert geometry/codec"; return false;
    }
    const auto input_row = actual_input.size() == first.width ? 0 : slot;
    const auto label = std::string(role) + " slot=" + std::to_string(slot) + " global=" + std::to_string(global_ids[slot]);
    if (!qwen24_original_nextn_reference::expect_projection(actual.subspan(slot * first.outputs, first.outputs), matrix.bytes,
        matrix.outputs, matrix.width, matrix.type, actual_input.subspan(input_row * first.width, first.width),
        codec, label.c_str(), full)) return false;
  }
  return true;
}

// Original F32 router/shared-gate coefficients multiplied by actual F32 input.
// Products are exact in F64 (<=48 significant bits), not assumed exact F32.
inline bool expect_f32_projection(std::span<const float> actual, std::span<const std::byte> original,
    std::size_t outputs, std::span<const float> actual_input, const char *role) {
  const auto width = actual_input.size();
  if (!width || !outputs || actual.size() != outputs || outputs > std::numeric_limits<std::size_t>::max() / width / 4 ||
      original.size() != outputs * width * 4) { ADD_FAILURE() << role << " original F32 extent"; return false; }
  const auto g32 = scalar::gamma(width, 0x1p-24), g64 = scalar::gamma(width, 0x1p-53);
  for (std::size_t row = 0; row < outputs; ++row) {
    scalar::Sum sum; double absolute = 0;
    for (std::size_t column = 0; column < width; ++column) {
      const float left = actual_input[column], right = detail::f32(original, row * width + column);
      const double product = double(left) * right;
      if (!std::isfinite(left) || !std::isfinite(right) || std::fpclassify(left) == FP_SUBNORMAL ||
          std::fpclassify(right) == FP_SUBNORMAL || !std::isfinite(product) ||
          (product != 0 && std::abs(product) < 0x1p-103))
        throw std::runtime_error("original Q24 F32 dot needs separate finite/underflow analysis");
      sum.add(product); absolute = scalar::add_upper(absolute, std::abs(product));
    }
    if (!std::isfinite(absolute) || absolute > double(std::numeric_limits<float>::max()) / 4)
      throw std::runtime_error("original Q24 F32 dot needs separate overflow analysis");
    const auto center = sum.value();
    const auto product_error = scalar::multiply_upper(0x1p-24, absolute);
    const auto native = scalar::add_upper(product_error, scalar::multiply_upper(g32,
        scalar::multiply_upper(1 + 0x1p-24, absolute)));
    const auto oracle = scalar::add_upper(base::detail::distance(center, sum.plain), scalar::multiply_upper(g64, absolute));
    if (!detail::reduced(actual[row], {center, scalar::add_upper(native, oracle)}, role, row)) return false;
  }
  return true;
}

struct RoutingValues final {
  std::span<const float> logits, probabilities, selected_scores, sum, denominator, weights;
  std::span<const std::int32_t> ids;
};
inline bool expect_routing(const RoutingValues &actual) {
  if (actual.logits.size() != kExperts || actual.probabilities.size() != kExperts || actual.ids.size() != kTopK ||
      actual.selected_scores.size() != kTopK || actual.weights.size() != kTopK || actual.sum.size() != 1 ||
      actual.denominator.size() != 1) { ADD_FAILURE() << "original Q24 routing extent"; return false; }
  const auto maximum = *std::max_element(actual.logits.begin(), actual.logits.end());
  std::vector<double> exponentials(kExperts); double total = 0;
  for (std::size_t i = 0; i < kExperts; ++i) {
    if (!std::isfinite(actual.logits[i])) { ADD_FAILURE() << "original Q24 nonfinite router"; return false; }
    total += exponentials[i] = std::exp(double(actual.logits[i]) - maximum);
  }
  std::array<std::size_t, kExperts> order{}; std::iota(order.begin(), order.end(), 0);
  for (std::size_t i = 0; i < kExperts; ++i)
    if (actual.probabilities[i] < 0 || actual.probabilities[i] > 1 ||
        !detail::local(actual.probabilities[i], float(exponentials[i] / total), "original Q24 actual-logit softmax", i)) return false;
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    return actual.probabilities[left] > actual.probabilities[right];
  });
  for (std::size_t slot = 0; slot < kTopK; ++slot) {
    if (actual.ids[slot] < 0 || std::size_t(actual.ids[slot]) != order[slot]) {
      ADD_FAILURE() << "original Q24 stable full512 top10 slot=" << slot; return false;
    }
    if (!detail::exact(actual.selected_scores[slot], actual.probabilities[order[slot]], "original Q24 selected probability", slot)) return false;
  }
  if (!detail::reduced(actual.sum[0], detail::sum(kTopK, [&](auto i) { return actual.selected_scores[i]; }),
      "original Q24 actual selected-score sum", 0) ||
      !detail::exact(actual.denominator[0], std::max(actual.sum[0], kMinimumDenominator), "original Q24 clamp2^-14", 0)) return false;
  for (std::size_t slot = 0; slot < kTopK; ++slot)
    if (actual.weights[slot] < 0 || actual.weights[slot] > 1 ||
        !detail::local(actual.weights[slot], float(double(actual.selected_scores[slot]) / actual.denominator[0]),
        "original Q24 normalized scale1 route weight", slot)) return false;
  return true;
}
inline bool expect_map(std::span<const std::int32_t> global_ids, std::span<const std::int32_t> local_ids,
    std::span<const std::int32_t> mapping) {
  if (global_ids.size() != kTopK || local_ids.size() != kTopK || mapping.size() != kExperts) {
    ADD_FAILURE() << "original Q24 expert-map extent"; return false;
  }
  std::array<bool, kTopK> seen{};
  std::array<bool, kExperts> selected{};
  for (std::size_t slot = 0; slot < kTopK; ++slot) {
    const auto global = global_ids[slot], local = local_ids[slot];
    if (global < 0 || std::size_t(global) >= kExperts || local < 0 || std::size_t(local) >= kTopK ||
        mapping[std::size_t(global)] != local || seen[std::size_t(local)]) {
      ADD_FAILURE() << "original Q24 actual global/local map slot=" << slot; return false;
    }
    seen[std::size_t(local)] = true;
    selected[std::size_t(global)] = true;
  }
  for (std::size_t global = 0; global < kExperts; ++global)
    if (!selected[global] && mapping[global] != -1) {
      ADD_FAILURE() << "original Q24 unselected expert must have no resident slot"; return false;
    }
  return true;
}
inline bool expect_swiglu(std::span<const float> actual, std::span<const float> gate,
    std::span<const float> up, const char *role) {
  if (actual.size() != gate.size() || actual.size() != up.size()) { ADD_FAILURE() << role << " SwiGLU extent"; return false; }
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!detail::local(actual[i], float(double(gate[i]) * detail::sigmoid(gate[i]) * up[i]), role, i)) return false;
  return true;
}
inline bool expect_product(std::span<const float> actual, std::span<const float> left,
    std::span<const float> row_weights, std::size_t width, const char *role) {
  if (!width || actual.empty() || actual.size() != left.size() || actual.size() % width ||
      row_weights.size() != actual.size() / width) { ADD_FAILURE() << role << " actual product extent"; return false; }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double product = double(left[i]) * row_weights[i / width];
    if (!std::isfinite(product) || (product != 0 && std::abs(product) < std::numeric_limits<float>::min()))
      throw std::runtime_error("original Q24 product needs separate range analysis");
    if (!detail::exact(actual[i], float(product), role, i)) return false;
  }
  return true;
}
inline bool expect_sum(std::span<const float> actual, std::span<const float> terms,
    std::size_t width, const char *role) {
  if (!width || actual.size() != width || terms.empty() || terms.size() % width) { ADD_FAILURE() << role << " actual sum extent"; return false; }
  for (std::size_t i = 0; i < width; ++i)
    if (!detail::reduced(actual[i], detail::sum(terms.size() / width,
        [&](auto row) { return terms[row * width + i]; }), role, i)) return false;
  return true;
}
using base::expect_embedding;
using base::expect_greedy;
} // namespace qwen24_original_nextn_reference
