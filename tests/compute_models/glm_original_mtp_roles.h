#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"
#include "hyle/mem/buffer.h"
#include "d5_qwen38_24t_semantic_fragments.h"
#include "original_role_input.h"
#include "glm_fp8_arithmetic.h"

// Original GLM-5.3-Flash NextN layer45, fixed official revision
// eb9eb208eb0d988989d07a6a12d0fdeb5f52574a. Test input and pure Fragment
// construction only: no Graph/Invocation, backend macros, executor or loader
// policy. The seven original bodies become six immutable runtime banks because
// q_a keeps its unchanged E4 codes and F32 block scales in one packed carrier.
// Independent oracle remains test-owned; model construction is compiled privately.
#include "../../engine/models/glm/glm_original_mtp_roles.h"

namespace glm_original_mtp_role_oracle {
using namespace glm_original_mtp_roles;
namespace sem=d5_qwen38_24t_semantic_fragments;
inline std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}
// Every named F32 stage remains a rounding boundary in the independent oracle.
inline float f32(double value) { volatile float rounded = static_cast<float>(value); return rounded; }
inline float multiply(float a, float b) { return f32(double(a) * double(b)); }

struct Dot final { double sum = 0, correction = 0; };
inline void add(Dot &dot, double value) {
  const auto next = dot.sum + value;
  dot.correction += std::abs(dot.sum) >= std::abs(value)
      ? (dot.sum - next) + value : (value - next) + dot.sum;
  dot.sum = next;
}
inline std::vector<float> norm_reference(std::span<const std::uint16_t> values,
                                std::span<const std::uint16_t> weights) {
  Dot squares;
  for (auto value : values) { const double x = original::widen(value); add(squares, x * x); }
  const double factor = 1.0 / std::sqrt((squares.sum + squares.correction) / double(values.size()) + kEpsilon);
  std::vector<float> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    result[i] = original::widen(bf16(float(double(original::widen(values[i])) * factor * double(original::widen(weights[i])))));
  return result;
}
inline std::vector<float> eh_reference(std::span<const std::uint16_t> joined, const Data &roles) {
  std::vector<float> result(kHidden);
  for (std::int64_t row = 0; row < kHidden; ++row) {
    Dot dot;
    for (std::int64_t col = 0; col < kJoined; ++col)
      add(dot, double(original::widen(joined[std::size_t(col)])) * original::widen(roles.eh[std::size_t(row * kJoined + col)]));
    result[std::size_t(row)] = original::widen(bf16(float(dot.sum + dot.correction)));
  }
  return result;
}
// Independent E4 code enumeration, with saturated nearest-even conversion.
inline float e4(std::uint8_t code) {
  const unsigned magnitude = code & 0x7fu, exponent = magnitude >> 3u, fraction = magnitude & 7u;
  const float value = exponent == 0 ? std::ldexp(float(fraction), -9)
      : std::ldexp(1.0F + float(fraction) / 8.0F, int(exponent) - 7);
  return (code & 0x80u) ? -value : value;
}
inline float rounded_e4(float value) {
  const double magnitude = std::min(std::abs(double(value)), 448.0);
  unsigned best = 0; double distance = magnitude;
  for (unsigned code = 1; code <= 0x7e; ++code) {
    const auto delta = std::abs(magnitude - double(e4(std::uint8_t(code))));
    if (delta < distance || (delta == distance && (code & 1u) == 0)) { best = code; distance = delta; }
  }
  return std::copysign(e4(std::uint8_t(best)), value);
}
inline std::vector<float> qa_reference(std::span<const std::uint16_t> values, const Data &roles, bool cuda,
    glm_fp8_arithmetic::Arithmetic arithmetic = {}) {
  arithmetic.require();
  constexpr std::size_t groups = kHidden / 128;
  std::array<float, groups> activation_scale{};
  std::vector<float> codes(values.size());
  for (std::size_t group = 0; group < groups; ++group) {
    float maximum = arithmetic.absmax_floor;
    for (std::size_t col = 0; col < 128; ++col)
      maximum = std::max(maximum, std::abs(original::widen(values[group * 128 + col])));
    using Policy=c::model_ops::Float8ScalePolicy;
    auto scale=arithmetic.policy==Policy::kDivide?f32(double(maximum)/448.0):
        multiply(maximum,1.0F/448.0F);
    if(arithmetic.policy==Policy::kPowerOfTwo) {
      int exponent=0;const auto fraction=std::frexp(double(scale),&exponent);
      scale=std::ldexp(1.0F,exponent-(fraction==0.5?1:0));
    }
    activation_scale[group]=scale;
    for (std::size_t col = 0; col < 128; ++col) {
      const auto value=original::widen(values[group*128+col]);
      const auto normalized=arithmetic.policy==Policy::kDivide?f32(double(value)/double(scale)):
          multiply(value,f32(1.0/double(scale)));
      codes[group*128+col]=rounded_e4(normalized);
    }
  }
  std::vector<float> result(kQuery);
  for (std::size_t row = 0; row < std::size_t(kQuery); ++row) {
    float accumulated = 0;
    for (std::size_t group = 0; group < groups; ++group) {
      Dot dot;
      for (std::size_t col = 0; col < 128; ++col) {
        const auto k = group * 128 + col;
        add(dot, double(codes[k]) * e4(std::to_integer<std::uint8_t>(roles.qa[row * kHidden + k])));
      }
      const auto scaled = multiply(f32(dot.sum + dot.correction), activation_scale[group]);
      const auto weight_scale = roles.scales[(row / 128) * groups + group];
      // Native CPU ATen has distinct mul/add boundaries; CUDA uses an FMA.
      // This is selected by the declared backend before examining any result.
      accumulated = cuda ? std::fma(scaled, weight_scale, accumulated)
          : f32(double(accumulated) + multiply(scaled, weight_scale));
    }
    result[row] = original::widen(bf16(accumulated));
  }
  return result;
}
inline bool close_bfloat(std::span<const std::uint16_t> actual,
                  std::span<const float> expected, std::string_view role) {
  if (actual.size() != expected.size()) { ADD_FAILURE() << role << " extent"; return false; }
  double maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double value = original::widen(actual[i]);
    const double error = std::abs(value - expected[i]); maximum = std::max(maximum, error);
    // Unchanged original BF16 cuBLAS gate used by the predecessor input test.
    // Dense original q_a has cancellation; this does not inherit a sparse
    // two-term fixture's exact-code-domain one-ULP assumption.
    const double tolerance = 0.008 * std::abs(double(expected[i])) + 0.0001;
    if (!std::isfinite(value) || error > tolerance) {
      ADD_FAILURE() << role << '[' << i << "] actual=" << value << " independent=" << expected[i]
                    << " tolerance=" << tolerance;
      return false;
    }
  }
  std::printf("GLM_ORIGINAL_CHAIN boundary=%.*s count=%zu max_abs=%.9g\n", int(role.size()), role.data(), actual.size(), maximum);
  return true;
}


// Host observations only, never new model inputs. Each independent local
// reference starts from the ACTUAL rounded predecessor, so error is not fed
// through several BF16 discontinuities or mistaken for a state defect.
inline bool observe_prefix(const Data &data,std::span<const std::uint16_t> hidden,
    std::span<const std::uint16_t> embedding,
    const std::array<std::vector<std::uint16_t>,7> &actual,bool cuda,
    glm_fp8_arithmetic::Arithmetic arithmetic = {}) {
  if(hidden.size()!=std::size_t(kHidden) || embedding.size()!=std::size_t(kHidden) ||
      actual[0].size()!=std::size_t(kHidden) || actual[1].size()!=std::size_t(kHidden) ||
      actual[2].size()!=std::size_t(kJoined) || actual[3].size()!=std::size_t(kHidden) ||
      actual[4].size()!=std::size_t(kHidden) || actual[5].size()!=std::size_t(kQuery) ||
      actual[6].size()!=std::size_t(kQuery)) {
    ADD_FAILURE()<<"original GLM prefix observation extent";return false;
  }
  if(!close_bfloat(actual[0],norm_reference(hidden,data.hnorm),"hnorm") ||
      !close_bfloat(actual[1],norm_reference(embedding,data.enorm),"enorm"))return false;
  auto joined=actual[1];joined.insert(joined.end(),actual[0].begin(),actual[0].end());
  if(actual[2]!=joined) {
    ADD_FAILURE()<<"embedding-before-hidden original EH order";return false;
  }
  return close_bfloat(actual[3],eh_reference(actual[2],data),"original_EH") &&
      close_bfloat(actual[4],norm_reference(actual[3],data.input_norm),"original_decoder_input_norm") &&
      close_bfloat(actual[5],qa_reference(actual[4],data,cuda,arithmetic),"original_dynamic128_q_a") &&
      close_bfloat(actual[6],norm_reference(actual[5],data.query_norm),"original_q_a_norm");
}
}

namespace glm_original_mtp_roles {
using glm_original_mtp_role_oracle::bf16;
using glm_original_mtp_role_oracle::f32;
using glm_original_mtp_role_oracle::multiply;
using glm_original_mtp_role_oracle::Dot;
using glm_original_mtp_role_oracle::add;
using glm_original_mtp_role_oracle::norm_reference;
using glm_original_mtp_role_oracle::eh_reference;
using glm_original_mtp_role_oracle::e4;
using glm_original_mtp_role_oracle::rounded_e4;
using glm_original_mtp_role_oracle::qa_reference;
using glm_original_mtp_role_oracle::close_bfloat;
using glm_original_mtp_role_oracle::observe_prefix;
} // namespace glm_original_mtp_roles
