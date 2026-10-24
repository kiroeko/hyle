#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Independent local references for the actual F32 HC/norm roles. These do not
// execute a model or propagate an oracle activation through a BF16 boundary.
namespace dspark_original_hc_reference {
namespace detail {
inline double upper(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}
inline double add_upper(double a, double b) { return upper(a + b); }
inline double multiply_upper(double a, double b) { return upper(a * b); }
inline double distance(double a, double b) {
  return a == b ? 0.0 : upper(std::abs(a - b));
}
inline double gamma(std::size_t count, double unit) {
  const double product = double(count) * unit;
  if (product >= 1.0) throw std::runtime_error("original HC reduction bound is undefined");
  return upper(product / (1.0 - product));
}
struct Sum final {
  double plain = 0.0, correction = 0.0;
  void add(double term) {
    const double next = plain + term;
    correction += std::abs(plain) >= std::abs(term)
        ? (plain - next) + term : (term - next) + plain;
    plain = next;
  }
  double value() const { return plain + correction; }
};
inline void finite(std::span<const float> values) {
  for (const float value : values)
    if (!std::isfinite(value)) throw std::runtime_error("original HC reference operand is nonfinite");
}
inline void bfloat(std::span<const float> values) {
  finite(values);
  for (const float value : values)
    if (std::bit_cast<std::uint32_t>(value) & 0xffffU)
      throw std::runtime_error("original HC reference needs actual widened BF16 predecessors");
}
inline float widen(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t(value) << 16U);
}
inline std::uint16_t rounded(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return std::uint16_t((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
inline double sigmoid(double value) {
  if (value >= 0.0) return 1.0 / (1.0 + std::exp(-value));
  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}
inline void rows(std::span<const float> input, std::size_t width) {
  if (!width || input.empty() || input.size() % width)
    throw std::runtime_error("original HC local row extent");
}
} // namespace detail

struct DenseReference final {
  std::vector<double> center, bound, sumabs;
  std::size_t width = 0;
};

// input[M,K] is the actual BF16 state widened to F32. The original HC fn is
// row-major F32[N,K], so its products are NOT assumed exact F32. BF16*F32 is
// exact FP64 (at most32 significant bits), which supplies the independent dot.
inline DenseReference linear(std::span<const float> actual_bfloat_input,
    std::span<const float> original_f32_weight, std::size_t output_rows) {
  if (!output_rows || original_f32_weight.empty() || original_f32_weight.size() % output_rows)
    throw std::runtime_error("original HC dense weight extent");
  const auto width = original_f32_weight.size() / output_rows;
  detail::rows(actual_bfloat_input, width);
  detail::bfloat(actual_bfloat_input); detail::finite(original_f32_weight);
  for (const float value : actual_bfloat_input)
    if (std::fpclassify(value) == FP_SUBNORMAL)
      throw std::runtime_error("original HC dot needs separate subnormal-source analysis");
  for (const float value : original_f32_weight)
    if (std::fpclassify(value) == FP_SUBNORMAL)
      throw std::runtime_error("original HC dot needs separate subnormal-weight analysis");
  const auto input_rows = actual_bfloat_input.size() / width;
  if (input_rows > std::numeric_limits<std::size_t>::max() / output_rows)
    throw std::runtime_error("original HC dense output extent overflow");
  DenseReference result;
  result.width = width; result.center.resize(input_rows * output_rows);
  result.bound.resize(result.center.size()); result.sumabs.resize(result.center.size());
  constexpr double u32 = 0x1p-24;
  const double g32 = detail::gamma(width, u32), g64 = detail::gamma(width, 0x1p-53);
  for (std::size_t input_row = 0; input_row < input_rows; ++input_row)
    for (std::size_t output_row = 0; output_row < output_rows; ++output_row) {
      detail::Sum sum; double absolute = 0.0;
      for (std::size_t column = 0; column < width; ++column) {
        const double product = double(actual_bfloat_input[input_row * width + column]) *
            double(original_f32_weight[output_row * width + column]);
        // A magnitude>=2^-103 makes every rounded F32 product's quantum at
        // least2^-126. Sums of those rounded products cannot be FTZ-sensitive.
        if (!std::isfinite(product) || (product != 0.0 && std::abs(product) < 0x1p-103))
          throw std::runtime_error("original HC dot needs separate underflow analysis");
        sum.add(product); absolute += std::abs(product);
      }
      if (!std::isfinite(absolute) || absolute > double(std::numeric_limits<float>::max()) / 4.0)
        throw std::runtime_error("original HC dot needs separate overflow analysis");
      const auto offset = input_row * output_rows + output_row;
      const double center = sum.value();
      const double upper_sum = absolute == 0.0 ? 0.0 : detail::upper(absolute / (1.0 - g64));
      // Separate F32 multiplication error u*S from accumulation error
      // gamma_K*(1+u)*S. This also admits FMA implementations, without
      // claiming that TF32/BF16-rounded F32 coefficients satisfy the contract.
      const double product_error = detail::multiply_upper(u32, upper_sum);
      const double rounded_sum = detail::multiply_upper(1.0 + u32, upper_sum);
      const double native_error = detail::add_upper(product_error,
          detail::multiply_upper(g32, rounded_sum));
      // Ordinary FP64 summation has a standard gamma_K bound. Add its actual
      // distance to the compensated center; never assume Neumaier is exact.
      const double oracle_error = detail::add_upper(detail::distance(center, sum.plain),
          detail::multiply_upper(g64, upper_sum));
      result.center[offset] = center; result.sumabs[offset] = absolute;
      result.bound[offset] = absolute == 0.0 ? 0.0 : detail::add_upper(native_error, oracle_error);
    }
  return result;
}
inline bool expect_linear(std::span<const float> actual, const DenseReference &reference,
                           std::string_view role) {
  if (actual.size() != reference.center.size() || actual.size() != reference.bound.size() ||
      actual.size() != reference.sumabs.size()) {
    ADD_FAILURE() << role << " original HC dot extent"; return false;
  }
  double maximum_error = 0.0, maximum_ratio = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = detail::distance(double(actual[i]), reference.center[i]);
    const double bound = reference.bound[i];
    if (!std::isfinite(actual[i]) || !std::isfinite(reference.center[i]) ||
        !std::isfinite(bound) || error > bound) {
      ADD_FAILURE() << role << '[' << i << "] actual=" << actual[i] << " F64=" << reference.center[i]
          << " error=" << error << " sumabs=" << reference.sumabs[i] << " F32 product+sum bound=" << bound;
      return false;
    }
    maximum_error = std::max(maximum_error, error);
    maximum_ratio = std::max(maximum_ratio, bound ? error / bound : 0.0);
  }
  std::printf("DSpark original HC %.*s K=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
      int(role.size()), role.data(), reference.width, maximum_error, maximum_ratio);
  return true;
}

inline bool expect_bfloat_store(std::span<const std::uint16_t> stored,
    std::span<const float> actual_f32, std::string_view role) {
  if (stored.size() != actual_f32.size()) {
    ADD_FAILURE() << role << " original HC BF16 extent"; return false;
  }
  for (std::size_t i = 0; i < stored.size(); ++i)
    if (!std::isfinite(actual_f32[i]) || !std::isfinite(detail::widen(stored[i])) ||
        stored[i] != detail::rounded(actual_f32[i])) {
      ADD_FAILURE() << role << '[' << i << "] differs from actual F32 nearest-even BF16 store"; return false;
    }
  return true;
}
inline bool expect_bfloat_store(std::span<const float> stored_widened,
    std::span<const float> actual_f32, std::string_view role) {
  if (stored_widened.size() != actual_f32.size()) {
    ADD_FAILURE() << role << " original HC widened BF16 extent"; return false;
  }
  for (std::size_t i = 0; i < stored_widened.size(); ++i)
    if (!std::isfinite(actual_f32[i]) || !std::isfinite(stored_widened[i]) ||
        std::bit_cast<std::uint32_t>(stored_widened[i]) !=
            (std::uint32_t(detail::rounded(actual_f32[i])) << 16U)) {
      ADD_FAILURE() << role << '[' << i << "] differs from actual F32 stored-and-widened BF16"; return false;
    }
  return true;
}

inline std::vector<float> inverse_rms(std::span<const float> actual_bfloat_input,
    std::size_t width, float epsilon = 1.0e-6F) {
  detail::rows(actual_bfloat_input, width); detail::bfloat(actual_bfloat_input);
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) throw std::runtime_error("original HC RMS epsilon");
  std::vector<float> result(actual_bfloat_input.size() / width);
  for (std::size_t row = 0; row < result.size(); ++row) {
    detail::Sum square;
    for (std::size_t column = 0; column < width; ++column) {
      const double value = actual_bfloat_input[row * width + column]; square.add(value * value);
    }
    result[row] = float(1.0 / std::sqrt(square.value() / double(width) + double(epsilon)));
  }
  return result;
}
inline std::vector<float> rms(std::span<const float> actual_bfloat_input,
    std::span<const float> original_f32_norm, float epsilon = 1.0e-6F) {
  detail::finite(original_f32_norm);
  detail::rows(actual_bfloat_input, original_f32_norm.size()); detail::bfloat(actual_bfloat_input);
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) throw std::runtime_error("original HC RMS epsilon");
  const auto width = original_f32_norm.size(); std::vector<float> result(actual_bfloat_input.size());
  // Evaluate the local normalization at FP64 accuracy; its input is the
  // actual prior BF16 store. The consumer checks raw F32 math before its
  // separate exact BF16 store, using the existing local norm gate.
  for (std::size_t row = 0; row < result.size() / width; ++row) {
    detail::Sum square;
    for (std::size_t column = 0; column < width; ++column) {
      const double value = actual_bfloat_input[row * width + column]; square.add(value * value);
    }
    const double scale = 1.0 / std::sqrt(square.value() / double(width) + double(epsilon));
    for (std::size_t column = 0; column < width; ++column)
      result[row * width + column] = float(double(actual_bfloat_input[row * width + column]) *
          scale * double(original_f32_norm[column]));
  }
  return result;
}

struct Split final { std::vector<float> pre, post, comb; };
// Fixed official kernel: rowSoftmax+eps, first column normalization, then
// nineteen row/column normalizations. Matrix[src,dst] is used by hc_post.
inline Split split(std::span<const float> actual_mixes, std::span<const float> scales,
                   std::span<const float> bases, float epsilon = 1.0e-6F) {
  detail::rows(actual_mixes, 24); detail::finite(actual_mixes);
  if (scales.size() != 3 || bases.size() != 24 || !std::isfinite(epsilon) || epsilon <= 0.0F)
    throw std::runtime_error("original HC split parameter extent");
  detail::finite(scales); detail::finite(bases);
  const auto rows = actual_mixes.size() / 24; Split result;
  result.pre.resize(rows * 4); result.post.resize(rows * 4); result.comb.resize(rows * 16);
  for (std::size_t row = 0; row < rows; ++row) {
    std::array<double, 16> matrix{};
    for (std::size_t stream = 0; stream < 4; ++stream) {
      result.pre[row * 4 + stream] = float(detail::sigmoid(double(actual_mixes[row * 24 + stream]) *
          scales[0] + bases[stream]) + double(epsilon));
      result.post[row * 4 + stream] = float(2.0 * detail::sigmoid(double(actual_mixes[row * 24 + 4 + stream]) *
          scales[1] + bases[4 + stream]));
    }
    for (std::size_t source = 0; source < 4; ++source) {
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t destination = 0; destination < 4; ++destination) {
        const auto index = source * 4 + destination;
        matrix[index] = double(actual_mixes[row * 24 + 8 + index]) * scales[2] + bases[8 + index];
        maximum = std::max(maximum, matrix[index]);
      }
      detail::Sum total;
      for (std::size_t destination = 0; destination < 4; ++destination) {
        auto &value = matrix[source * 4 + destination]; value = std::exp(value - maximum); total.add(value);
      }
      for (std::size_t destination = 0; destination < 4; ++destination)
        matrix[source * 4 + destination] = matrix[source * 4 + destination] / total.value() + double(epsilon);
    }
    const auto normalize = [&](bool columns) {
      for (std::size_t outer = 0; outer < 4; ++outer) {
        detail::Sum total; total.add(double(epsilon));
        for (std::size_t inner = 0; inner < 4; ++inner)
          total.add(matrix[columns ? inner * 4 + outer : outer * 4 + inner]);
        for (std::size_t inner = 0; inner < 4; ++inner)
          matrix[columns ? inner * 4 + outer : outer * 4 + inner] /= total.value();
      }
    };
    normalize(true);
    for (std::size_t iteration = 1; iteration < 20; ++iteration) { normalize(false); normalize(true); }
    for (std::size_t index = 0; index < matrix.size(); ++index) result.comb[row * 16 + index] = float(matrix[index]);
  }
  return result;
}
// The final output head is four independent sigmoid gates, not a24-way
// attention/FFN HC split and not a Sinkhorn matrix.
inline std::vector<float> head_gate(std::span<const float> actual_mixes,
    std::span<const float> scale, std::span<const float> bases, float epsilon = 1.0e-6F) {
  detail::rows(actual_mixes, 4); detail::finite(actual_mixes);
  if (scale.size() != 1 || bases.size() != 4 || !std::isfinite(epsilon) || epsilon <= 0.0F)
    throw std::runtime_error("original HC head gate extent");
  detail::finite(scale); detail::finite(bases);
  std::vector<float> result(actual_mixes.size());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = float(detail::sigmoid(double(actual_mixes[i]) * scale[0] + bases[i % 4]) + double(epsilon));
  return result;
}
inline std::vector<float> collapse(std::span<const float> actual_bfloat_state,
                                  std::span<const float> actual_pre) {
  detail::rows(actual_pre, 4); detail::finite(actual_pre); detail::bfloat(actual_bfloat_state);
  if (actual_bfloat_state.empty() || actual_bfloat_state.size() % actual_pre.size())
    throw std::runtime_error("original HC collapse extent");
  const auto hidden = actual_bfloat_state.size() / actual_pre.size(), rows = actual_pre.size() / 4;
  std::vector<float> result(rows * hidden);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < hidden; ++column) {
      detail::Sum sum;
      for (std::size_t stream = 0; stream < 4; ++stream)
        sum.add(double(actual_bfloat_state[(row * 4 + stream) * hidden + column]) * actual_pre[row * 4 + stream]);
      result[row * hidden + column] = float(sum.value());
    }
  return result;
}
inline std::vector<float> expand(std::span<const float> actual_bfloat_branch,
    std::span<const float> actual_bfloat_residual, std::span<const float> actual_post,
    std::span<const float> actual_combination) {
  detail::rows(actual_post, 4); detail::finite(actual_post); detail::finite(actual_combination);
  detail::bfloat(actual_bfloat_branch); detail::bfloat(actual_bfloat_residual);
  const auto rows = actual_post.size() / 4;
  if (actual_bfloat_branch.empty() || actual_bfloat_branch.size() % rows ||
      actual_bfloat_residual.size() != 4 * actual_bfloat_branch.size() || actual_combination.size() != rows * 16)
    throw std::runtime_error("original HC expansion extent");
  const auto hidden = actual_bfloat_branch.size() / rows; std::vector<float> result(actual_bfloat_residual.size());
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t destination = 0; destination < 4; ++destination)
      for (std::size_t column = 0; column < hidden; ++column) {
        detail::Sum sum;
        sum.add(double(actual_bfloat_branch[row * hidden + column]) * actual_post[row * 4 + destination]);
        for (std::size_t source = 0; source < 4; ++source)
          sum.add(double(actual_bfloat_residual[(row * 4 + source) * hidden + column]) *
              actual_combination[row * 16 + source * 4 + destination]);
        result[(row * 4 + destination) * hidden + column] = float(sum.value());
      }
  return result;
}
} // namespace dspark_original_hc_reference
