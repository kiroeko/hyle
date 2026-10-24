#pragma once

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
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Independent local Q8_0 references, with explicit native activation codecs.
// No device reference, expanded weight bank, or model executor lives here.
namespace dspark_original_shared_reference {
enum class ActivationCodec { kCpuAvxQ8_0, kCpuScalarQ8_0, kCudaMmvqQ8_1Fast };
namespace detail {
inline double upper(double term) { return std::nextafter(term, INFINITY); }
inline double add_upper(double left, double right) { return upper(left + right); }
inline double multiply_upper(double left, double right) { return upper(left * right); }
inline double gamma(std::size_t count, double unit) {
  const double product = double(count) * unit;
  if (product >= 1.0) throw std::runtime_error("shared Q8 reduction bound is undefined");
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
inline float half(std::uint16_t bits) {
  const auto sign = std::uint32_t(bits & 0x8000U) << 16U;
  const auto exponent = unsigned((bits >> 10U) & 31U);
  const auto fraction = unsigned(bits & 1023U);
  if (exponent == 31U) throw std::runtime_error("shared Q8 scale is nonfinite");
  if (!exponent) return std::copysign(std::ldexp(float(fraction), -24), sign ? -1.0F : 1.0F);
  return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (fraction << 13U));
}
inline std::uint16_t half_bits(float term) {
  if (!std::isfinite(term)) throw std::runtime_error("shared Q8 half input is nonfinite");
  const auto raw = std::bit_cast<std::uint32_t>(term);
  const auto sign = std::uint16_t((raw >> 16U) & 0x8000U);
  const int exponent = int((raw >> 23U) & 255U) - 127;
  const auto significand = (raw & 0x7fffffU) | 0x800000U;
  if (exponent < -25) return sign;
  if (exponent > 15) throw std::runtime_error("shared Q8 activation half scale overflows");
  if (exponent < -14) {
    const auto shift = unsigned(-exponent - 1);
    const auto main = significand >> shift;
    const auto remainder = significand & ((std::uint32_t(1) << shift) - 1U);
    const auto midpoint = std::uint32_t(1) << (shift - 1U);
    return std::uint16_t(sign | std::uint16_t(main +
        unsigned(remainder > midpoint || (remainder == midpoint && (main & 1U)))));
  }
  const auto main = (raw >> 13U) & 0x3ffU;
  const auto remainder = raw & 0x1fffU;
  const auto rounded = main + unsigned(remainder > 0x1000U ||
      (remainder == 0x1000U && (main & 1U)));
  const auto result = (unsigned(exponent + 15) << 10U) + rounded;
  if (result >= 0x7c00U) throw std::runtime_error("shared Q8 rounded scale overflows");
  return std::uint16_t(sign | std::uint16_t(result));
}
inline float scale(std::span<const std::byte> packed, std::size_t offset) {
  return half(std::uint16_t(std::to_integer<unsigned>(packed[offset]) |
      (std::to_integer<unsigned>(packed[offset + 1]) << 8U)));
}
inline int code(std::byte raw) { return int(std::bit_cast<std::int8_t>(raw)); }
inline void extent(std::span<const std::byte> packed, std::size_t outputs,
    std::size_t width, std::span<const float> input) {
  if (!outputs || !width || width % 32 || input.empty() || input.size() % width ||
      outputs > std::numeric_limits<std::size_t>::max() / (width / 32) / 34 ||
      input.size() / width > std::numeric_limits<std::size_t>::max() / outputs ||
      packed.size() != outputs * (width / 32) * 34)
    throw std::runtime_error("shared Q8 reference extent");
  if (std::fegetround() != FE_TONEAREST)
    throw std::runtime_error("shared Q8 reference requires host round-to-nearest-even");
  for (const float term : input)
    if (!std::isfinite(term)) throw std::runtime_error("shared Q8 reference input is nonfinite");
}
inline float rn_divide(float numerator, float denominator) {
  // Binary32 division through binary64 has enough precision to decide the
  // binary32 rounding boundary for these finite normal binary32 operands.
  return float(double(numerator) / double(denominator));
}
inline float rn_multiply(float left, float right) { return float(double(left) * double(right)); }
inline int rounded_code(float term, bool nearest_even) {
  const double rounded = nearest_even ? std::nearbyint(double(term)) : std::round(double(term));
  if (rounded < -127.0 || rounded > 127.0)
    throw std::runtime_error("shared Q8 native code interval exceeds signed Q8 range");
  return int(rounded);
}
struct Range final { float low, high; };
inline Range fast_divide(float numerator, float denominator) {
  if (!std::isnormal(denominator) || std::abs(denominator) > 0x1p126F)
    throw std::runtime_error("shared Q8 fast division denominator outside published envelope");
  if (numerator == 0.0F) return {numerator, numerator};
  if (!std::isnormal(numerator))
    throw std::runtime_error("shared Q8 fast division needs separate FTZ-input analysis");
  const float center = rn_divide(numerator, denominator);
  if (!std::isnormal(center))
    throw std::runtime_error("shared Q8 fast division needs separate under/overflow analysis");
  float low = center, high = center;
  // CUDA 13.3 publishes a 2-ULP __fdividef envelope, explicitly observed rather
  // than guaranteed. Four binary32 neighbours cover both spacings at a binade
  // edge and the correctly-rounded center's half-ULP. This is a conditional
  // source-based certificate, not a formal guarantee for every CUDA input.
  // Correctly-rounded strict division is a subset of the same interval: the
  // archived quantizer's precise compiler flags need not be guessed.
  for (int step = 0; step < 4; ++step) {
    low = std::nextafter(low, -INFINITY); high = std::nextafter(high, INFINITY);
  }
  if (!std::isnormal(low) || !std::isnormal(high))
    throw std::runtime_error("shared Q8 fast division interval crosses FTZ/overflow boundary");
  return {low, high};
}
struct ActivationBlock final {
  float scale = 0, scale_low = 0, scale_high = 0;
  std::array<int, 32> codes{}, low{}, high{};
};
inline ActivationBlock quantize(std::span<const float, 32> input, ActivationCodec codec,
    bool require_bfloat = true) {
  ActivationBlock result; float maximum = 0.0F;
  for (const float term : input) {
    if ((require_bfloat && (std::bit_cast<std::uint32_t>(term) & 0xffffU)) ||
        (term != 0.0F && !std::isnormal(term)))
      throw std::runtime_error("shared Q8 local certificate needs actual normal widened BF16 inputs");
    maximum = std::max(maximum, std::abs(term));
  }
  if (maximum == 0.0F) return result;
  const float divisor = rn_divide(maximum, 127.0F);
  if (!std::isnormal(divisor))
    throw std::runtime_error("shared Q8 activation divisor needs separate underflow analysis");
  result.scale = half(half_bits(divisor));
  result.scale_low = result.scale_high = result.scale;
  if (codec == ActivationCodec::kCudaMmvqQ8_1Fast) {
    const auto divisor_range = fast_divide(maximum, 127.0F);
    result.scale_low = half(half_bits(divisor_range.low));
    result.scale_high = half(half_bits(divisor_range.high));
    for (std::size_t column = 0; column < 32; ++column) {
      const float term = input[column];
      result.codes[column] = rounded_code(rn_divide(term, divisor), false);
      const auto first = fast_divide(term, divisor_range.low);
      const auto last = fast_divide(term, divisor_range.high);
      result.low[column] = rounded_code(std::min(first.low, last.low), false);
      result.high[column] = rounded_code(std::max(first.high, last.high), false);
    }
  } else {
    const bool avx = codec == ActivationCodec::kCpuAvxQ8_0;
    const float reciprocal = avx ? rn_divide(127.0F, maximum) : rn_divide(1.0F, divisor);
    if (!std::isnormal(reciprocal))
      throw std::runtime_error("shared Q8 CPU reciprocal needs separate under/overflow analysis");
    for (std::size_t column = 0; column < 32; ++column)
      result.codes[column] = result.low[column] = result.high[column] =
          rounded_code(rn_multiply(input[column], reciprocal), avx);
  }
  return result;
}
} // namespace detail

// Cold whole-chain reference callback only. This is the decoded raw-input
// mathematical projection, not the native-activation-quantized local oracle.
// Its caller owns the explicit BF16 store; this function returns raw F32.
inline std::vector<float> decoded_reference(std::span<const std::byte> packed,
    std::size_t outputs, std::size_t width, std::span<const float> input) {
  detail::extent(packed, outputs, width, input);
  std::vector<float> result(input.size() / width * outputs);
  const auto blocks = width / 32;
  for (std::size_t row = 0; row < input.size() / width; ++row)
    for (std::size_t output_row = 0; output_row < outputs; ++output_row) {
      detail::Sum sum;
      for (std::size_t block = 0; block < blocks; ++block) {
        const auto offset = (output_row * blocks + block) * 34;
        const double weight_scale = detail::scale(packed, offset);
        for (std::size_t column = 0; column < 32; ++column)
          sum.add(weight_scale * detail::code(packed[offset + 2 + column]) *
              double(input[row * width + block * 32 + column]));
      }
      result[row * outputs + output_row] = float(sum.value());
    }
  return result;
}

struct ProjectionReference final {
  std::vector<double> center, bound, quantization_ambiguity, arithmetic_bound, sumabs;
  std::size_t ambiguous_codes = 0, ambiguous_scales = 0;
  ActivationCodec codec = ActivationCodec::kCpuScalarQ8_0;
};
inline ProjectionReference evaluate(std::span<const std::byte> packed,
    std::size_t outputs, std::size_t width, std::span<const float> actual_bfloat_input,
    ActivationCodec codec) {
  detail::extent(packed, outputs, width, actual_bfloat_input);
  const auto blocks = width / 32, rows = actual_bfloat_input.size() / width;
  std::vector<detail::ActivationBlock> activation(rows * blocks);
  ProjectionReference result; result.codec = codec;
  for (std::size_t block = 0; block < activation.size(); ++block) {
    activation[block] = detail::quantize(
        std::span<const float, 32>(actual_bfloat_input.data() + block * 32, 32), codec);
    const auto &source = activation[block];
    result.ambiguous_scales += source.scale_low != source.scale_high;
    for (std::size_t column = 0; column < 32; ++column)
      result.ambiguous_codes += source.low[column] != source.high[column];
  }
  result.center.resize(rows * outputs); result.bound.resize(rows * outputs);
  result.quantization_ambiguity.resize(rows * outputs);
  result.arithmetic_bound.resize(rows * outputs); result.sumabs.resize(rows * outputs);
  // AVX dots use four-code integer partials; CUDA MMVQ uses eight-code partials.
  // Integers and half*half scales are exact F32. At most K/4 scaled partials
  // plus64 reduction/product operations on a path bounds both implementations
  // (including scalar CPU) without relying on their parallel reduction order.
  const double g32 = detail::gamma(width / 4 + 65, 0x1p-24);
  const double g64 = detail::gamma(blocks, 0x1p-53);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t output_row = 0; output_row < outputs; ++output_row) {
      detail::Sum sum; double ambiguity = 0.0, absolute = 0.0;
      for (std::size_t block = 0; block < blocks; ++block) {
        const auto offset = (output_row * blocks + block) * 34;
        const auto &source = activation[row * blocks + block];
        const double weight_scale = detail::scale(packed, offset);
        int dot = 0, uncertain = 0, maximum_absolute = 0;
        for (std::size_t column = 0; column < 32; ++column) {
          const int coefficient = detail::code(packed[offset + 2 + column]);
          dot += coefficient * source.codes[column];
          uncertain += std::abs(coefficient) * std::max(
              std::abs(source.low[column] - source.codes[column]),
              std::abs(source.high[column] - source.codes[column]));
          maximum_absolute += std::abs(coefficient) * std::max(
              std::abs(source.low[column]), std::abs(source.high[column]));
        }
        // All integer block dots fit int32, their conversions fit F32; finite
        // half*half*int products fit F64 exactly. Minimum nonzero scale-product
        // quantum is2^-48, so neither native products nor cancellation can FTZ.
        sum.add(weight_scale * double(source.scale) * double(dot));
        const double scale_error = std::max(std::abs(double(source.scale_low) - source.scale),
            std::abs(double(source.scale_high) - source.scale));
        const double scale_maximum = std::max(std::abs(double(source.scale_low)),
            std::abs(double(source.scale_high)));
        ambiguity = detail::add_upper(ambiguity, detail::multiply_upper(std::abs(weight_scale),
            detail::add_upper(detail::multiply_upper(scale_error, std::abs(double(dot))),
                detail::multiply_upper(scale_maximum, double(uncertain)))));
        absolute = detail::add_upper(absolute, detail::multiply_upper(std::abs(weight_scale),
            detail::multiply_upper(scale_maximum, double(maximum_absolute))));
      }
      if (!std::isfinite(absolute) || absolute > double(std::numeric_limits<float>::max()) / 4.0)
        throw std::runtime_error("shared Q8 native reduction needs separate overflow analysis");
      const auto index = row * outputs + output_row;
      result.center[index] = sum.value(); result.sumabs[index] = absolute;
      const double oracle = detail::add_upper(detail::upper(std::abs(sum.value() - sum.plain)),
          detail::multiply_upper(g64, absolute));
      result.quantization_ambiguity[index] = ambiguity;
      result.arithmetic_bound[index] = detail::multiply_upper(g32, absolute);
      result.bound[index] = detail::add_upper(ambiguity,
          detail::add_upper(result.arithmetic_bound[index], oracle));
    }
  return result;
}
inline void expect_projection(std::span<const float> actual,
    std::span<const std::byte> packed, std::size_t outputs, std::size_t width,
    std::span<const float> actual_bfloat_input, ActivationCodec codec, std::string_view role) {
  const auto reference = evaluate(packed, outputs, width, actual_bfloat_input, codec);
  ASSERT_EQ(actual.size(), reference.center.size()) << role;
  double maximum_ratio = 0.0, maximum_error = 0.0, maximum_ambiguity = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ASSERT_TRUE(std::isfinite(actual[index])) << role << " output=" << index;
    const double error = double(actual[index]) == reference.center[index] ? 0.0 :
        detail::upper(std::abs(double(actual[index]) - reference.center[index]));
    maximum_error = std::max(maximum_error, error);
    maximum_ambiguity = std::max(maximum_ambiguity, reference.quantization_ambiguity[index]);
    if (reference.bound[index] != 0.0)
      maximum_ratio = std::max(maximum_ratio, error / reference.bound[index]);
    ASSERT_LE(error, reference.bound[index]) << role << " output=" << index
        << " center=" << reference.center[index] << " ambiguity="
        << reference.quantization_ambiguity[index] << " F32_bound="
        << reference.arithmetic_bound[index] << " sumabs=" << reference.sumabs[index];
  }
  std::fprintf(stderr, "shared-Q8-local role=%.*s codec=%d outputs=%zu max_error=%.17g "
      "max_ratio=%.17g max_quantization_ambiguity=%.17g ambiguous_codes=%zu ambiguous_scales=%zu\n",
      int(role.size()), role.data(), int(codec), actual.size(), maximum_error, maximum_ratio,
      maximum_ambiguity, reference.ambiguous_codes, reference.ambiguous_scales);
}
} // namespace dspark_original_shared_reference
