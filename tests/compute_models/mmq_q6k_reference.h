#pragma once

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "qwen27_original_nextn_reference.h"

// One independent MMQ Q6_K/D4 reference. The only reused packed primitive is
// original W decoding; no MMVQ activation quantizer or native dot is called.
// This deliberately small normal-input domain does not certify every CUDA
// fast-division input, another MMQ codec, another architecture or real weights.
namespace ggml_cuda_mmq_q6k_reference {
namespace fp = dspark_original_shared_reference::detail;
namespace packed = qwen27_original_nextn_reference::detail;
inline constexpr std::size_t kRows = 16, kWidth = 512, kOutputs = 3;
inline constexpr std::size_t kGroups = kWidth / 32;

struct Activation final {
  float scale = 0, low_scale = 0, high_scale = 0;
  std::array<int, 32> codes{}, low{}, high{};
};

inline Activation quantize_d4(std::span<const float, 32> input) {
  float maximum = 0.0F;
  for (const float value : input) {
    if (!std::isfinite(value) || (value != 0.0F &&
        (!std::isnormal(value) || std::abs(value) < 0x1p-16F)) ||
        std::abs(value) > 0x1p8F)
      throw std::runtime_error("MMQ D4 fixture requires finite normal bounded F32 A");
    maximum = std::max(maximum, std::abs(value));
  }
  if (maximum < 0x1p-8F || maximum > 0x1p8F)
    throw std::runtime_error("MMQ D4 fixture excludes zero/small-maximum blocks");

  // quantize.cu: d_inv=127/amax; q=roundf(RN32(A*d_inv)); d=1/d_inv.
  // D4 stores d as F32. These are two divisions, not A/(amax/127).
  // fast_divide retains the existing conditional CUDA published-envelope
  // assumption (four F32 neighbours); it is not a universal CUDA guarantee.
  const float reciprocal = fp::rn_divide(127.0F, maximum);
  const auto inverse_range = fp::fast_divide(127.0F, maximum);
  Activation result;
  result.scale = fp::rn_divide(1.0F, reciprocal);
  result.low_scale = fp::fast_divide(1.0F, inverse_range.high).low;
  result.high_scale = fp::fast_divide(1.0F, inverse_range.low).high;
  if (!(result.low_scale >= 0x1p-17F && result.high_scale <= 0x1p3F))
    throw std::runtime_error("MMQ D4 scale leaves the proved normal domain");
  for (std::size_t lane = 0; lane < 32; ++lane) {
    const float first = fp::rn_multiply(input[lane], inverse_range.low);
    const float last = fp::rn_multiply(input[lane], inverse_range.high);
    result.codes[lane] = fp::rounded_code(fp::rn_multiply(input[lane], reciprocal), false);
    // roundf is monotone, including its ties-away-from-zero discontinuities.
    result.low[lane] = fp::rounded_code(std::min(first, last), false);
    result.high[lane] = fp::rounded_code(std::max(first, last), false);
  }
  return result;
}

struct Projection final {
  std::array<double, kRows * kOutputs> center{}, bound{}, ambiguity{}, arithmetic{}, sumabs{};
  std::size_t ambiguous_codes = 0, non_half_scales = 0;
};

namespace detail {
inline Projection evaluate(std::span<const std::byte> original,
    std::span<const Activation> activation) {
  if (original.size() != kOutputs * (kWidth / 256) * 210 ||
      activation.size() != kRows * kGroups || std::fegetround() != FE_TONEAREST)
    throw std::runtime_error("MMQ Q6_K reference extent/host rounding");
  Projection result;
  for (const auto &a : activation) {
    result.non_half_scales += fp::half(fp::half_bits(a.scale)) != a.scale;
    for (std::size_t lane = 0; lane < 32; ++lane)
      result.ambiguous_codes += a.low[lane] != a.high[lane];
  }

  // Q6_K native integer dot combines two signed-scale16 partials BEFORE
  // conversion: |I32| <=32*32*128*127=16,646,144 <2^24. Thus that
  // conversion is exact, including the Turing integer-MMA implementation.
  //
  // Count actual scalar F32 operations conservatively over the entire dot:
  // G products d8*I +G additions; at most H=K/64 products by d6 and H
  // outer additions (DP4A subdot64; Turing MMA subdot128 is a subset).
  // Stream-K partitions at original QK256 block boundaries: B=K/256
  // nonempty pieces at most. B+1 additional fixup/final adds suffices.
  // This gives 2G+2H+B+1=51, independent of the chosen J or SM count.
  // FMA contraction reduces the count. This is not the MMVQ bound.
  constexpr std::size_t operations = 2 * kGroups + 2 * (kWidth / 64) + kWidth / 256 + 1;
  constexpr double unit32 = 0x1p-24, unit64 = 0x1p-53;
  const double native_gamma = fp::gamma(operations, unit32);
  const double oracle_gamma = fp::gamma(2 * kGroups + 4, unit64);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t output = 0; output < kOutputs; ++output) {
      const auto weights = original.subspan(output * 420, 420);
      fp::Sum sum;
      double ambiguity = 0.0, absolute = 0.0;
      for (std::size_t group = 0; group < kGroups; ++group) {
        const auto w = packed::weight(weights, GGML_TYPE_Q6_K, group);
        const auto &a = activation[row * kGroups + group];
        if (w.scale != 0.0 && (std::abs(w.scale) < 0x1p-8 || std::abs(w.scale) > 0x1p8))
          throw std::runtime_error("MMQ Q6_K W scale leaves the proved normal domain");
        std::int64_t integer = 0, uncertainty = 0;
        for (std::size_t lane = 0; lane < 32; ++lane) {
          integer += std::int64_t(w.codes[lane]) * a.codes[lane];
          uncertainty += std::int64_t(std::abs(w.codes[lane])) * std::max(
              std::abs(a.low[lane] - a.codes[lane]), std::abs(a.high[lane] - a.codes[lane]));
        }
        const double integer_maximum = double(std::abs(integer) + uncertainty);
        if (integer_maximum > 16646144.0)
          throw std::runtime_error("MMQ Q6_K centered integer exceeds exact F32 range");
        const double maximum_scale = std::max(double(a.low_scale), double(a.high_scale));
        const double scale_error = std::max(std::abs(double(a.low_scale) - a.scale),
            std::abs(double(a.high_scale) - a.scale));
        sum.add(w.scale * double(a.scale) * double(integer));
        ambiguity = fp::add_upper(ambiguity, fp::multiply_upper(std::abs(w.scale),
            fp::add_upper(fp::multiply_upper(scale_error, std::abs(double(integer))),
                fp::multiply_upper(maximum_scale, double(uncertainty)))));
        absolute = fp::add_upper(absolute, fp::multiply_upper(std::abs(w.scale),
            fp::multiply_upper(maximum_scale, integer_maximum)));
      }
      // Every F32 D4 scale is on a >=2^-40 grid in this domain; every
      // original finite half d6 is on a >=2^-18 grid. Integer products,
      // additions, d6 products and fixups therefore stay on a >=2^-58 grid
      // (or coarser after rounding); a nonzero intermediate cannot FTZ.
      // The same explicit maxima imply sumabs<2^40, hence no F32 overflow.
      if (!std::isfinite(absolute) || absolute >= 0x1p40)
        throw std::runtime_error("MMQ Q6_K dot leaves the proved finite domain");
      const auto index = row * kOutputs + output;
      result.center[index] = sum.value();
      result.sumabs[index] = absolute;
      result.ambiguity[index] = ambiguity;
      result.arithmetic[index] = fp::multiply_upper(native_gamma, absolute);
      // The plain F64 evaluation has at most two products per group and G
      // additions. Neumaier/plain distance plus gamma_(2G+4)*C encloses the
      // center's independent F64 rounding; no native result enters the gate.
      const double oracle = fp::add_upper(fp::upper(std::abs(sum.value() - sum.plain)),
          fp::multiply_upper(oracle_gamma, absolute));
      result.bound[index] = fp::add_upper(ambiguity,
          fp::add_upper(result.arithmetic[index], oracle));
    }
  }
  return result;
}

inline std::vector<Activation> activations(std::span<const float> input) {
  if (input.size() != kRows * kWidth)
    throw std::runtime_error("MMQ Q6_K activation extent");
  std::vector<Activation> result(kRows * kGroups);
  for (std::size_t group = 0; group < result.size(); ++group)
    result[group] = quantize_d4(std::span<const float, 32>(input.data() + group * 32, 32));
  return result;
}
} // namespace detail

inline Projection evaluate(std::span<const std::byte> original, std::span<const float> input) {
  return detail::evaluate(original, detail::activations(input));
}

// Fixture discrimination only: same MMQ reciprocal/multiply/codes, but an
// intentionally wrong F16 store of D4 d. It is neither a MMVQ oracle nor an
// alternate accepted contract. The test requires disjoint correct/wrong
// intervals for at least one output, before any native submission.
inline Projection wrong_half_scale_for_fixture(std::span<const std::byte> original,
    std::span<const float> input) {
  auto activation = detail::activations(input);
  for (auto &a : activation) {
    a.scale = fp::half(fp::half_bits(a.scale));
    a.low_scale = fp::half(fp::half_bits(a.low_scale));
    a.high_scale = fp::half(fp::half_bits(a.high_scale));
  }
  return detail::evaluate(original, activation);
}

inline void expect_projection(std::span<const float> actual, const Projection &reference) {
  ASSERT_EQ(actual.size(), reference.center.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ASSERT_TRUE(std::isfinite(actual[index])) << "MMQ Q6_K output=" << index;
    const double error = double(actual[index]) == reference.center[index] ? 0.0 :
        fp::upper(std::abs(double(actual[index]) - reference.center[index]));
    ASSERT_LE(error, reference.bound[index]) << "MMQ Q6_K output=" << index
        << " center=" << reference.center[index] << " AQ_ambiguity=" << reference.ambiguity[index]
        << " F32_bound=" << reference.arithmetic[index] << " sumabs=" << reference.sumabs[index];
  }
}
} // namespace ggml_cuda_mmq_q6k_reference
