#pragma once

#include <gtest/gtest.h>

#include <algorithm>
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

// Test-only independent references for the original GLM NextN indexer.
// Every linear starts at the ACTUAL checked BF16 producer. A reference result
// is never propagated through a BF16/E4 store into the next reference stage.
namespace glm_original_indexer_reference {
namespace detail {
inline float widen(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t(bits) << 16U);
}
inline double upper(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}
inline double upper_add(double a, double b) { return upper(a + b); }
inline double upper_multiply(double a, double b) { return upper(a * b); }
inline double distance(double a, double b) {
  return a == b ? 0.0 : upper(std::abs(a - b));
}
inline double gamma(std::size_t count, double unit) {
  const double product = double(count) * unit;
  if (product >= 1.0) throw std::runtime_error("original indexer reduction bound is undefined");
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
inline void checked_bfloat(std::span<const float> values) {
  for (const float value : values)
    if (!std::isfinite(value) || (std::bit_cast<std::uint32_t>(value) & 0xffffU))
      throw std::runtime_error("original indexer needs actual widened BF16 operands");
}
inline void checked_bfloat(std::span<const std::uint16_t> values) {
  for (const auto value : values)
    if (!std::isfinite(widen(value)))
      throw std::runtime_error("original indexer coefficient or ring is nonfinite");
}
inline std::uint16_t bfloat(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return std::uint16_t((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
} // namespace detail

struct Reference final {
  std::vector<double> center, bound, sumabs;
  std::size_t input_width = 0;
};

// Original weight banks are row-major [N,K]. The widened input is [M,K].
// BF16 products are exact F32, so at most K F32 accumulation roundings bound
// any reduction tree or FMA sequence. The conservative normal-quantum guard
// excludes product underflow, FTZ-sensitive cancellation and overflow.
inline Reference linear(std::span<const float> actual_widened_bfloat,
    std::span<const std::uint16_t> rowmajor_weight, std::size_t output_rows) {
  if (!output_rows || rowmajor_weight.empty() || rowmajor_weight.size() % output_rows)
    throw std::runtime_error("original indexer linear weight extent");
  const auto width = rowmajor_weight.size() / output_rows;
  if (actual_widened_bfloat.empty() || actual_widened_bfloat.size() % width)
    throw std::runtime_error("original indexer linear actual input extent");
  detail::checked_bfloat(actual_widened_bfloat);
  detail::checked_bfloat(rowmajor_weight);
  const auto input_rows = actual_widened_bfloat.size() / width;
  if (input_rows > std::numeric_limits<std::size_t>::max() / output_rows)
    throw std::runtime_error("original indexer linear output extent overflow");
  Reference result;
  result.input_width = width;
  result.center.resize(input_rows * output_rows);
  result.bound.resize(result.center.size()); result.sumabs.resize(result.center.size());
  const double g32 = detail::gamma(width, 0x1p-24), g64 = detail::gamma(width, 0x1p-53);
  for (std::size_t input_row = 0; input_row < input_rows; ++input_row)
    for (std::size_t output_row = 0; output_row < output_rows; ++output_row) {
      detail::Sum sum; double absolute = 0.0;
      for (std::size_t column = 0; column < width; ++column) {
        const double product = double(actual_widened_bfloat[input_row * width + column]) *
            detail::widen(rowmajor_weight[output_row * width + column]);
        // A representable F32 term of magnitude >=2^-103 has quantum >=2^-126.
        // All partial sums of these terms therefore avoid subnormal quanta.
        if (!std::isfinite(product) || double(float(product)) != product ||
            (product != 0.0 && std::abs(product) < 0x1p-103))
          throw std::runtime_error("original indexer dot needs separate product/underflow analysis");
        sum.add(product); absolute += std::abs(product);
      }
      if (!std::isfinite(absolute) || absolute > double(std::numeric_limits<float>::max()) / 4.0)
        throw std::runtime_error("original indexer dot needs separate overflow analysis");
      const auto offset = input_row * output_rows + output_row;
      const double center = sum.value();
      const double absolute_upper = absolute == 0.0 ? 0.0 : detail::upper(absolute / (1.0 - g64));
      // Do not assume a compensated FP64 result is exact. The ordinary FP64
      // sum has the standard gamma_K bound; triangle inequality adds its
      // ACTUAL distance to the compensated center. This also covers the
      // rounding in the compensated reduction and its final addition.
      const double oracle_bound = detail::upper_add(
          detail::distance(center, sum.plain), detail::upper_multiply(g64, absolute_upper));
      result.center[offset] = center; result.sumabs[offset] = absolute;
      result.bound[offset] = absolute == 0.0 ? 0.0 : detail::upper_add(
          detail::upper_multiply(g32, absolute_upper), oracle_bound);
    }
  return result;
}

inline bool expect_accum(std::span<const float> actual, const Reference &reference,
                          std::string_view role) {
  if (actual.size() != reference.center.size() || actual.size() != reference.bound.size() ||
      actual.size() != reference.sumabs.size()) {
    ADD_FAILURE() << role << " original indexer accumulation extent"; return false;
  }
  double maximum_error = 0.0, maximum_ratio = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = detail::distance(double(actual[i]), reference.center[i]);
    const double bound = reference.bound[i];
    if (!std::isfinite(actual[i]) || !std::isfinite(reference.center[i]) ||
        !std::isfinite(bound) || error > bound) {
      ADD_FAILURE() << role << '[' << i << "] actual=" << actual[i]
          << " F64=" << reference.center[i] << " error=" << error
          << " sumabs=" << reference.sumabs[i] << " analytic F32 bound=" << bound;
      return false;
    }
    maximum_error = std::max(maximum_error, error);
    maximum_ratio = std::max(maximum_ratio, bound ? error / bound : 0.0);
  }
  std::printf("GLM original indexer %.*s: K=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
      int(role.size()), role.data(), reference.input_width, maximum_error, maximum_ratio);
  return true;
}

// The exact store is compared with its actual F32 predecessor, independently
// of whether the dot/norm above rounded to the same side of a BF16 midpoint.
inline bool expect_bfloat_store(std::span<const std::uint16_t> actual,
    std::span<const float> actual_f32, std::string_view role) {
  if (actual.size() != actual_f32.size()) {
    ADD_FAILURE() << role << " original indexer BF16 store extent"; return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!std::isfinite(actual_f32[i]) || !std::isfinite(detail::widen(actual[i])) ||
        actual[i] != detail::bfloat(actual_f32[i])) {
      ADD_FAILURE() << role << '[' << i << "] differs from actual F32 nearest-even BF16 store";
      return false;
    }
  return true;
}

// Local LayerNorm math starts at the actual rounded WK, not an independent
// projection carried through this nonlinear and subsequently discrete stage.
inline std::vector<float> layer_norm(std::span<const float> actual_rounded_wk,
    std::span<const std::uint16_t> norm, std::span<const std::uint16_t> bias,
    float epsilon = 1.0e-6F) {
  if (norm.empty() || norm.size() != bias.size() || actual_rounded_wk.empty() ||
      actual_rounded_wk.size() % norm.size() || !std::isfinite(epsilon) || epsilon <= 0.0F)
    throw std::runtime_error("original indexer LayerNorm extent/epsilon");
  detail::checked_bfloat(actual_rounded_wk);
  detail::checked_bfloat(norm); detail::checked_bfloat(bias);
  const auto width = norm.size(); std::vector<float> result(actual_rounded_wk.size());
  for (std::size_t first = 0; first < result.size(); first += width) {
    detail::Sum total;
    for (std::size_t column = 0; column < width; ++column) total.add(actual_rounded_wk[first + column]);
    const double mean = total.value() / double(width);
    detail::Sum variance;
    for (std::size_t column = 0; column < width; ++column) {
      const double delta = double(actual_rounded_wk[first + column]) - mean;
      variance.add(delta * delta);
    }
    const double inverse = 1.0 / std::sqrt(variance.value() / double(width) + double(epsilon));
    for (std::size_t column = 0; column < width; ++column)
      result[first + column] = float((double(actual_rounded_wk[first + column]) - mean) *
          inverse * detail::widen(norm[column]) + detail::widen(bias[column]));
  }
  return result;
}

// One four-token pool, in its explicitly supplied physical ring order.
// APE is the original BF16 checkpoint role exactly widened at load. Gate+APE
// is an F32 operation; the softmax/weighted-sum reference itself uses FP64.
inline std::vector<float> pool_reference(std::span<const std::uint16_t> keys,
    std::span<const std::uint16_t> gates, std::span<const std::uint16_t> ape,
    std::span<const std::int32_t> ringrows) {
  constexpr std::size_t depth = 128, pool = 4;
  if (keys.empty() || keys.size() % depth || gates.size() != keys.size() ||
      ape.size() != pool * depth || ringrows.size() != pool)
    throw std::runtime_error("original indexer pooled reference extent");
  detail::checked_bfloat(ape);
  for (std::size_t slot = 0; slot < pool; ++slot) {
    if (ringrows[slot] < 0 || std::size_t(ringrows[slot]) >= keys.size() / depth)
      throw std::runtime_error("original indexer pool physical ring index");
    for (std::size_t previous = 0; previous < slot; ++previous)
      if (ringrows[slot] == ringrows[previous])
        throw std::runtime_error("original indexer pool repeats a physical ring row");
  }
  std::vector<float> result(depth);
  for (std::size_t column = 0; column < depth; ++column) {
    double logits[pool], maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t slot = 0; slot < pool; ++slot) {
      const auto offset = std::size_t(ringrows[slot]) * depth + column;
      if (!std::isfinite(detail::widen(keys[offset])) || !std::isfinite(detail::widen(gates[offset])))
        throw std::runtime_error("original indexer pool selected a nonfinite ring value");
      volatile float added = float(double(detail::widen(gates[offset])) +
          double(detail::widen(ape[slot * depth + column])));
      logits[slot] = added; maximum = std::max(maximum, logits[slot]);
    }
    detail::Sum numerator, denominator;
    for (std::size_t slot = 0; slot < pool; ++slot) {
      const double probability = std::exp(logits[slot] - maximum);
      denominator.add(probability);
      numerator.add(probability * detail::widen(keys[std::size_t(ringrows[slot]) * depth + column]));
    }
    result[column] = float(numerator.value() / denominator.value());
  }
  return result;
}
} // namespace glm_original_indexer_reference
