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
#include <vector>

#include "glm_original_attention_roles.h"
#include "glm_original_mtp_roles.h"
#include "glm_original_indexer_reference.h"
#include "glm_shared_vocabulary_reference.h"

// Local original-byte references for the Flash NoPE MLA. No reference value
// is fed into a Graph or propagated across a BF16 boundary. Native BF16 BMM
// returns BF16 directly: its unobservable F32 accumulator is checked through
// an analytically derived rounding interval, never an invented raw port.
namespace glm_original_attention_reference {
namespace original = glm_original_attention_roles;
namespace prefix = glm_original_mtp_roles;
namespace scalar = glm_original_indexer_reference::detail;
namespace bytes = glm_shared_vocabulary_reference::detail;
namespace detail {
inline float read_f32(std::span<const std::byte> source, std::size_t element) {
  std::uint32_t bits = 0;
  for (std::size_t i = 0; i < 4; ++i)
    bits |= std::to_integer<std::uint32_t>(source[element * 4 + i]) << (i * 8);
  return std::bit_cast<float>(bits);
}
inline std::vector<std::size_t> rows(std::size_t count, bool full) {
  if (!count) throw std::runtime_error("original attention empty projection");
  std::vector<std::size_t> result;
  if (full) {
    result.resize(count);
    for (std::size_t i = 0; i < count; ++i) result[i] = i;
  } else {
    result = {0, std::min(std::size_t{1}, count - 1), count / 4, count / 2,
        count / 4 * 3, count > 1 ? count - 2 : 0, count - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
  }
  return result;
}
inline float finite(std::uint16_t value) { return bytes::finite_bfloat(value); }
inline void exact_product(double value) {
  if (!std::isfinite(value) || double(float(value)) != value ||
      (value != 0 && std::abs(value) < 0x1p-103))
    throw std::runtime_error("original attention BF16 dot needs separate product/underflow analysis");
}
struct Dot final { double center = 0, bound = 0; };
template<class Term> Dot bfloat_dot(std::size_t count, Term &&term) {
  scalar::Sum sum;
  double absolute = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = term(i); exact_product(value);
    sum.add(value); absolute += std::abs(value);
  }
  const double g64 = scalar::gamma(count, 0x1p-53);
  const double upper_sum = absolute == 0 ? 0 : scalar::upper(absolute / (1 - g64));
  if (!std::isfinite(upper_sum) || upper_sum > double(std::numeric_limits<float>::max()) / 4)
    throw std::runtime_error("original attention BF16 dot needs separate overflow analysis");
  const double center = sum.value();
  const double oracle = scalar::upper_add(scalar::distance(center, sum.plain), scalar::upper_multiply(g64, upper_sum));
  const double native = scalar::upper_multiply(scalar::gamma(count, 0x1p-24), upper_sum);
  return {center, absolute == 0 ? 0 : scalar::upper_add(native, oracle)};
}
inline bool bfloat_interval(std::uint16_t actual, const Dot &reference,
                            const char *role, std::size_t head, std::size_t row) {
  const double lower = std::nextafter(reference.center - reference.bound, -std::numeric_limits<double>::infinity());
  const double upper = std::nextafter(reference.center + reference.bound, std::numeric_limits<double>::infinity());
  // Both conversions are monotone. These outward endpoint stores therefore
  // include every RNE BF16 return of any admitted F32 accumulation result.
  const float low = finite(scalar::bfloat(float(lower))), high = finite(scalar::bfloat(float(upper)));
  const float value = finite(actual);
  if (value < low || value > high) {
    ADD_FAILURE() << role << " head=" << head << " row=" << row << " actual=" << value
        << " F64=" << reference.center << " F32 bound=" << reference.bound
        << " admitted BF16 interval=[" << low << ',' << high << ']';
    return false;
  }
  return true;
}
struct Fp8Shape final { original::RawRole codes, scales; std::size_t n, k; const char *name; };
inline Fp8Shape shape(original::Parameter parameter) {
  using P = original::Parameter; using R = original::RawRole;
  switch (parameter) {
  case P::kQueryProjection: return {R::kQueryCodes,R::kQueryScale,16384,1536,"q_b"};
  case P::kKvProjection: return {R::kKvCodes,R::kKvScale,512,4096,"kv_a"};
  case P::kOutput: return {R::kOutputCodes,R::kOutputScale,4096,16384,"o"};
  default: throw std::runtime_error("original attention reference needs an E4 projection role");
  }
}
} // namespace detail

inline bool expect_fp8_matrix(std::span<const float> actual,
    std::span<const std::uint16_t> actual_input, std::span<const std::byte> weight,
    std::span<const std::byte> scales, std::size_t outputs, std::size_t width,
    const char *role, bool full,
    hyle::compute::model_ops::Float8ScalePolicy policy=hyle::compute::model_ops::Float8ScalePolicy::kDivide,
    float absmax_floor=1.0e-10F,bool bfloat_output=false) {
  using Policy=hyle::compute::model_ops::Float8ScalePolicy;
  if((policy!=Policy::kDivide&&policy!=Policy::kPowerOfTwo&&policy!=Policy::kMultiplyReciprocal)||
      !std::isfinite(absmax_floor)||!(absmax_floor>0)||!std::isnormal(absmax_floor/448.0F)) {
    ADD_FAILURE()<<role<<" original E4 activation scale contract";return false;
  }
  if (!outputs || !width || width % 128 ||
      outputs > std::numeric_limits<std::size_t>::max() / width) {
    ADD_FAILURE() << "original E4 projection shape"; return false;
  }
  const auto groups = width / 128;
  if (actual.size() != outputs || actual_input.size() != width ||
      weight.size() != outputs * width || scales.size() != (outputs / 128 + (outputs % 128 != 0)) * groups * 4) {
    ADD_FAILURE() << "original E4 attention projection extent"; return false;
  }
  std::vector<float> codes(width), activation_scale(groups);
  for (std::size_t group = 0; group < groups; ++group) {
    float maximum = absmax_floor;
    for (std::size_t column = 0; column < 128; ++column)
      maximum = std::max(maximum, std::abs(detail::finite(actual_input[group * 128 + column])));
    float scale = policy==Policy::kDivide?float(double(maximum) / 448.0):
        float(double(maximum)*double(1.0F/448.0F));
    if(!std::isnormal(scale)||!(scale>0))
      throw std::runtime_error("original E4 raw activation scale outside finite normal domain");
    if(policy==Policy::kPowerOfTwo) {
      int exponent=0;const double fraction=std::frexp(double(scale),&exponent);
      // Definition-based upward power2, independent of native exponent-bit
      // manipulation/log2 implementation. The raw scale is already F32.
      scale=std::ldexp(1.0F,exponent-(fraction==0.5?1:0));
    }
    if (!std::isnormal(scale) || !(scale > 0))
      throw std::runtime_error("original E4 attention activation scale outside finite normal domain");
    activation_scale[group] = scale;
    for (std::size_t column = 0; column < 128; ++column) {
      const auto i = group * 128 + column;
      const auto value=detail::finite(actual_input[i]);
      const float normalized=policy==Policy::kDivide?float(double(value)/double(scale)):
          float(double(value)*double(float(1.0/double(scale))));
      codes[i] = prefix::rounded_e4(normalized);
    }
  }
  const auto selected = detail::rows(outputs, full);
  const double g64 = scalar::gamma(groups + 2, 0x1p-53);
  const auto native_count = 128 + groups + 2;
  const double g32 = scalar::gamma(native_count, 0x1p-24);
  double maximum_error = 0, maximum_ratio = 0;
  for (const auto row : selected) {
    scalar::Sum sum;
    double absolute = 0;
    for (std::size_t group = 0; group < groups; ++group) {
      double code_dot = 0, code_abs = 0;
      for (std::size_t column = 0; column < 128; ++column) {
        const auto i = group * 128 + column;
        const auto byte = std::to_integer<std::uint8_t>(weight[row * width + i]);
        if ((byte & 0x7fU) == 0x7fU) throw std::runtime_error("original attention E4 weight is nonfinite");
        const double product = double(codes[i]) * prefix::e4(byte);
        code_dot += product; code_abs += std::abs(product);
      }
      // E4 products lie on a2^-18 grid and the128-product absolute sum is
      // below2^25, so these independent F64 code-domain sums are exact.
      const double as = activation_scale[group];
      const float weight_scale = detail::read_f32(scales, (row / 128) * groups + group);
      const double ws = weight_scale;
      if (!std::isnormal(weight_scale) || !(ws > 0) || code_abs * as > double(std::numeric_limits<float>::max()) / 4)
        throw std::runtime_error("original attention FP8 scale/intermediate needs separate analysis");
      sum.add(code_dot * as * ws); absolute += code_abs * as * ws;
    }
    const double upper_sum = absolute == 0 ? 0 : scalar::upper(absolute / (1 - g64));
    if (!std::isfinite(upper_sum) || upper_sum > double(std::numeric_limits<float>::max()) / 4)
      throw std::runtime_error("original attention FP8 sum needs separate overflow analysis");
    const double center = sum.value();
    const double oracle = scalar::upper_add(scalar::distance(center, sum.plain), scalar::upper_multiply(g64, upper_sum));
    // At most128 F32 code-dot roundings, two scale products and G outer
    // accumulation steps on each term's path. This covers fused and separate
    // scale/add order. A-scale>=floor/448 and the code grid preclude first
    // scale underflow. Allow at most2G*minNormal absolute error for optional
    // FTZ at weight scaling/outer adds, propagated through the same roundings.
    const double underflow = scalar::upper(double(2 * groups) * 0x1p-126 /
        (1 - double(native_count) * 0x1p-24));
    const double bound = absolute == 0 ? 0 : scalar::upper_add(
        scalar::upper_add(scalar::upper_multiply(g32, upper_sum), oracle), underflow);
    const double error = scalar::distance(double(actual[row]), center);
    double admitted=bound;
    bool accepted=std::isfinite(actual[row])&&error<=bound;
    if(bfloat_output) {
      const double lower=std::nextafter(center-bound,-std::numeric_limits<double>::infinity());
      const double upper=std::nextafter(center+bound,std::numeric_limits<double>::infinity());
      const float lo=detail::finite(scalar::bfloat(float(lower))),hi=detail::finite(scalar::bfloat(float(upper)));
      admitted=std::max(std::abs(double(lo)-center),std::abs(double(hi)-center));
      accepted=std::isfinite(actual[row])&&(std::bit_cast<std::uint32_t>(actual[row])&0xffffU)==0&&actual[row]>=lo&&actual[row]<=hi;
    }
    if (!accepted) {
      ADD_FAILURE() << "original attention " << role << '[' << row << "] actual=" << actual[row]
          << " F64=" << center << " error=" << error << " code/scales F32 bound=" << bound;
      return false;
    }
    maximum_error = std::max(maximum_error, error);
    maximum_ratio = std::max(maximum_ratio, admitted ? error / admitted : 0.0);
  }
  std::printf("GLM original attention %s full=%d outputs=%zu K=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
      role,int(full),selected.size(),width,maximum_error,maximum_ratio);
  return true;
}

// Keep the original MLA entry point and the same original-byte numerical gate.
// The shared primitive also serves the selected original MoE expert rows.
inline bool expect_fp8_projection(std::span<const float> actual,
    std::span<const std::uint16_t> actual_input, const original::Data &data,
    original::Parameter parameter, bool full,glm_fp8_arithmetic::Arithmetic arithmetic = {}) {
  arithmetic.require();
  const auto geometry = detail::shape(parameter);
  return expect_fp8_matrix(actual, actual_input, data.bytes(geometry.codes),
      data.bytes(geometry.scales), geometry.n, geometry.k, geometry.name, full,arithmetic.policy,arithmetic.absmax_floor);
}

inline bool expect_absorbed_query(std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> query, const original::Data &data, bool full) {
  constexpr std::size_t heads = 64, width = 256, latent = 512;
  const auto source = data.bytes(original::RawRole::kKvB);
  if (actual.size() != heads * latent || query.size() != heads * width || source.size() != heads * (width + 256) * latent * 2) {
    ADD_FAILURE() << "original query absorption extent"; return false;
  }
  const auto selected = detail::rows(latent, full);
  for (std::size_t head = 0; head < heads; ++head)
    for (const auto row : selected) {
      const auto reference = detail::bfloat_dot(width,[&](std::size_t column) {
        const auto q = query[head * width + column];
        const auto w = bytes::read_bfloat(source, (head * (width + 256) + column) * latent + row);
        bytes::normal_bfloat(q); bytes::normal_bfloat(w);
        return double(detail::finite(q)) * detail::finite(w);
      });
      if (!detail::bfloat_interval(actual[head * latent + row],reference,"original absorbed query",head,row)) return false;
    }
  std::printf("GLM original BF16 absorption full=%d checked=%zu\n",int(full),heads * selected.size());
  return true;
}

inline bool expect_value_projection(std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> attention_output, const original::Data &data, bool full) {
  constexpr std::size_t heads = 64, width = 256, latent = 512;
  const auto source = data.bytes(original::RawRole::kKvB);
  if (actual.size() != heads * width || attention_output.size() != heads * latent || source.size() != heads * (width + 256) * latent * 2) {
    ADD_FAILURE() << "original value BMM extent"; return false;
  }
  const auto selected = detail::rows(width, full);
  for (std::size_t head = 0; head < heads; ++head)
    for (const auto row : selected) {
      const auto reference = detail::bfloat_dot(latent,[&](std::size_t column) {
        const auto a = attention_output[head * latent + column];
        const auto w = bytes::read_bfloat(source, (head * (width + 256) + width + row) * latent + column);
        bytes::normal_bfloat(a); bytes::normal_bfloat(w);
        return double(detail::finite(a)) * detail::finite(w);
      });
      if (!detail::bfloat_interval(actual[head * width + row],reference,"original value projection",head,row)) return false;
    }
  std::printf("GLM original BF16 value BMM full=%d checked=%zu\n",int(full),heads * selected.size());
  return true;
}

inline std::vector<float> norm(std::span<const float> actual_input,
    std::span<const std::uint16_t> original_weight, std::size_t hidden) {
  if (!hidden || actual_input.empty() || actual_input.size() % hidden || original_weight.size() != hidden)
    throw std::runtime_error("original attention local RMS extent");
  std::vector<float> result(actual_input.size());
  for (std::size_t first = 0; first < actual_input.size(); first += hidden) {
    scalar::Sum squares;
    for (std::size_t i = 0; i < hidden; ++i) {
      const double value = actual_input[first + i];
      if (!std::isfinite(value)) throw std::runtime_error("original attention RMS input is nonfinite");
      squares.add(value * value);
    }
    const double factor = 1 / std::sqrt(squares.value() / double(hidden) + double(original::kEpsilon));
    for (std::size_t i = 0; i < hidden; ++i)
      result[first + i] = float(double(actual_input[first + i]) * factor * detail::finite(original_weight[i]));
  }
  return result;
}

inline std::vector<float> attention(std::span<const std::uint16_t> absorbed_query,
    std::span<const float> actual_chosen, std::size_t selected) {
  constexpr std::size_t heads = 64, latent = 512;
  if (!selected || selected > std::numeric_limits<std::size_t>::max() / latent ||
      absorbed_query.size() != heads * latent || actual_chosen.size() != selected * latent)
    throw std::runtime_error("original NoPE MQA reference extent");
  scalar::checked_bfloat(actual_chosen);
  std::vector<float> query(absorbed_query.size());
  for (std::size_t i = 0; i < query.size(); ++i) query[i] = detail::finite(absorbed_query[i]);
  std::vector<float> result(heads * latent);
  std::vector<double> scores(selected), probability(selected);
  for (std::size_t head = 0; head < heads; ++head) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < selected; ++token) {
      scalar::Sum dot;
      for (std::size_t column = 0; column < latent; ++column)
        dot.add(double(query[head * latent + column]) * actual_chosen[token * latent + column]);
      scores[token] = dot.value() * double(original::kAttentionScale);
      maximum = std::max(maximum,scores[token]);
    }
    scalar::Sum denominator;
    for (std::size_t token = 0; token < selected; ++token) {
      probability[token] = std::exp(scores[token] - maximum); denominator.add(probability[token]);
    }
    if (!(denominator.value() > 0) || !std::isfinite(denominator.value()))
      throw std::runtime_error("original attention softmax denominator is invalid");
    for (auto &value : probability) value /= denominator.value();
    for (std::size_t column = 0; column < latent; ++column) {
      scalar::Sum value;
      for (std::size_t token = 0; token < selected; ++token)
        value.add(probability[token] * actual_chosen[token * latent + column]);
      result[head * latent + column] = float(value.value());
    }
  }
  return result;
}
} // namespace glm_original_attention_reference
