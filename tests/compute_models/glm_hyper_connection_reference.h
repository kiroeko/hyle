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
#include <string>
#include <string_view>
#include <vector>

#include "dspark_original_hc_reference.h"
#include "glm_hyper_connection_fragment.h"
#include "glm_shared_vocabulary_reference.h"

// The exact former target44 HC local checks, shared by ordinary43/44 model
// fragments. Raw fn remains original BF16 bytes; its runtimeF32 widening is
// exact. This helper follows fixed06992 torch/unfused semantics only.
namespace glm_hyper_connection_reference {
namespace graph = glm_hyper_connection_fragment;
namespace lm = glm_shared_vocabulary_reference;
namespace hc = dspark_original_hc_reference;
namespace scalar = glm_original_indexer_reference::detail;
using F32 = std::span<const float>;
using BF16 = std::span<const std::uint16_t>;
using Bytes = std::span<const std::byte>;
inline constexpr std::size_t kHidden = 4096, kStreams = 4;

namespace detail {
inline std::uint32_t unsigned_le(Bytes raw, std::size_t offset, std::size_t count) {
  std::uint32_t bits = 0;
  for (std::size_t byte = 0; byte < count; ++byte)
    bits |= std::to_integer<std::uint32_t>(raw[offset + byte]) << (8U * unsigned(byte));
  return bits;
}
inline std::vector<float> floats(Bytes raw) {
  if (raw.size() % 4) throw std::runtime_error("original44 F32 observation byte extent");
  std::vector<float> result(raw.size() / 4);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = std::bit_cast<float>(unsigned_le(raw, i * 4, 4));
    if (!std::isfinite(result[i])) throw std::runtime_error("original44 F32 observation is nonfinite");
  }
  return result;
}
inline std::vector<std::uint16_t> bfloats(Bytes raw) {
  if (raw.size() % 2) throw std::runtime_error("original44 BF16 observation byte extent");
  std::vector<std::uint16_t> result(raw.size() / 2);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = std::uint16_t(unsigned_le(raw, i * 2, 2));
    lm::detail::finite_bfloat(result[i]);
  }
  return result;
}
inline std::vector<float> widen(BF16 source) {
  std::vector<float> result(source.size());
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = lm::detail::finite_bfloat(source[i]);
  return result;
}
inline bool same_bfloat(BF16 actual, BF16 expected, std::string_view role) {
  if (actual.size() != expected.size() || !std::equal(actual.begin(), actual.end(), expected.begin())) {
    ADD_FAILURE() << role << " differs from its actual BF16 producer"; return false;
  }
  return true;
}
inline bool local(F32 actual, F32 expected, std::string_view role) {
  if (actual.size() != expected.size()) { ADD_FAILURE() << role << " local extent"; return false; }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double bound = 8.0e-5 + 4.0e-5 * std::abs(double(expected[i]));
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        std::abs(double(actual[i]) - expected[i]) > bound) {
      ADD_FAILURE() << role << '[' << i << "] actual=" << actual[i] << " local F64=" << expected[i]
          << " established local gate=" << bound; return false;
    }
  }
  return true;
}
inline bool rounded(float actual, double exact, std::string_view role, std::size_t index) {
  const float expected = float(exact);
  if (!std::isfinite(exact) || !std::isfinite(expected) || !std::isfinite(actual) ||
      (expected != 0 && std::fpclassify(expected) == FP_SUBNORMAL))
    throw std::runtime_error("original44 scalar operation needs separate underflow/overflow analysis");
  if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
    ADD_FAILURE() << role << '[' << index << "] differs from actual F32 operation"; return false;
  }
  return true;
}
inline bool reduction(float actual, F32 terms, double divisor, std::string_view role) {
  if (terms.empty() || !(divisor > 0)) throw std::runtime_error("original44 reduction geometry");
  scalar::Sum sum; double absolute = 0;
  for (const float term : terms) {
    if (!std::isfinite(term) || (term != 0 && std::abs(double(term)) < 0x1p-103))
      throw std::runtime_error("original44 reduction needs separate subnormal analysis");
    sum.add(term); absolute += std::abs(double(term));
  }
  const double g64 = scalar::gamma(terms.size(), 0x1p-53);
  const double upper_sum = absolute == 0 ? 0 : scalar::upper(absolute / (1 - g64));
  if (upper_sum > double(std::numeric_limits<float>::max()) / 4)
    throw std::runtime_error("original44 reduction needs separate overflow analysis");
  const double center = sum.value() / divisor;
  // An extra F32 rounding admits mean's final division, including a native
  // multiply by an exactly represented reciprocal for the4/128 power-of-two.
  const double native = scalar::upper_multiply(scalar::gamma(terms.size() + 1, 0x1p-24), upper_sum);
  const double oracle = scalar::upper_add(scalar::distance(sum.value(), sum.plain),
      scalar::upper_multiply(g64, upper_sum));
  const double bound = absolute == 0 ? 0 : scalar::upper(scalar::upper_add(native, oracle) / divisor);
  if (!std::isfinite(actual) || scalar::distance(actual, center) > bound) {
    ADD_FAILURE() << role << " actual=" << actual << " F64=" << center << " bound=" << bound; return false;
  }
  return true;
}
inline std::vector<float> sigmoid(F32 input) {
  std::vector<float> result(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) result[i] = float(hc::detail::sigmoid(input[i]));
  return result;
}
} // namespace detail

template<class ReadValue>
inline bool expect_norm(const graph::NormPorts &part, BF16 gamma, ReadValue &&actual) {
  const auto f32 = [&](graph::Value value) { return detail::floats(actual(value)); };
  const auto bf16 = [&](graph::Value value) { return detail::bfloats(actual(value)); };
    const auto input = f32(part.input), raw = f32(part.raw); const auto stored = bf16(part.stored);
    return lm::expect_norm(raw, stored, input, gamma, gamma.size(), graph::kNormEpsilon) &&
        lm::expect_widen(f32(part.visible), stored);
  }

template<class ReadValue>
inline bool expect_pre(const graph::HcPorts &part, Bytes fn, F32 scales, F32 bases,
                       BF16 gamma, ReadValue &&actual) {
  const auto f32 = [&](graph::Value value) { return detail::floats(actual(value)); };
  const auto bf16 = [&](graph::Value value) { return detail::bfloats(actual(value)); };
    const auto state = bf16(part.residual); const auto flat = f32(part.flat);
    const auto projection = f32(part.projection), square = f32(part.square), mean = f32(part.mean);
    const auto inverse = f32(part.inverse), mixes = f32(part.mixes);
    if (state.size() != kStreams * kHidden || !lm::expect_widen(flat, state) ||
        !lm::expect_logits(projection, state, fn, 24, kStreams * kHidden, true)) return false;
    if (square.size() != flat.size() || mean.size() != 1 || inverse.size() != 1 || mixes.size() != 24) {
      ADD_FAILURE() << "original44 HC normalization extent"; return false;
    }
    for (std::size_t i = 0; i < flat.size(); ++i)
      if (!detail::rounded(square[i], double(flat[i]) * flat[i], "HC actual square", i)) return false;
    if (!detail::reduction(mean[0], square, double(square.size()), "HC actual square mean")) return false;
    const float rms_denominator = float(double(mean[0]) + double(graph::kNormEpsilon));
    const std::array expected_inverse{float(1.0 / std::sqrt(double(rms_denominator)))};
    if (!detail::local(inverse, expected_inverse, "HC inverse RMS1e-5 from actual F32 mean")) return false;
    for (std::size_t i = 0; i < mixes.size(); ++i)
      if (!detail::rounded(mixes[i], double(projection[i]) * inverse[0], "HC actual dot times inverse", i)) return false;
    const auto expected = hc::split(mixes, scales, bases, graph::kHcEpsilon);
    const auto pre = f32(part.pre);
    if (!detail::local(pre, expected.pre, "HC actual pre") || !detail::local(f32(part.post), expected.post, "HC actual post") ||
        !detail::local(f32(part.combination), expected.comb, "HC actual Sinkhorn20")) return false;
    const auto collapsed = f32(part.collapsed_raw); const auto stored = bf16(part.collapsed_stored);
    return detail::local(collapsed, hc::collapse(flat, pre), "HC actual four-stream collapse") &&
        glm_original_indexer_reference::expect_bfloat_store(stored, collapsed, "HC collapsed BF16 before learned RMS") &&
        lm::expect_widen(f32(part.norm.input), stored) && expect_norm(part.norm, gamma, actual);
  }

template<class ReadValue>
inline bool expect_post(const graph::PostPorts &part, const graph::HcPorts &pre, ReadValue &&actual) {
  const auto f32 = [&](graph::Value value) { return detail::floats(actual(value)); };
  const auto bf16 = [&](graph::Value value) { return detail::bfloats(actual(value)); };
    const auto branch = detail::widen(bf16(part.branch)), residual = detail::widen(bf16(part.residual));
    const auto raw = f32(part.raw); const auto stored = bf16(part.stored);
    return detail::local(raw, hc::expand(branch, residual, f32(pre.post), f32(pre.combination)), "HC actual post expansion") &&
        glm_original_indexer_reference::expect_bfloat_store(stored, raw, "HC actual post BF16") && lm::expect_widen(f32(part.visible), stored);
  }
} // namespace glm_hyper_connection_reference
