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
namespace glm_original_mtp_roles {
namespace c=hyle::compute;
namespace m=hyle::mem;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace original=original_role_input;
using Bytes=std::vector<std::byte>;
using Value=sem::ValueId;

template<class T> inline T take(hyle::Result<T> result,std::string_view where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status,std::string_view where) {
  if(!status.ok()) throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
template<class T> inline Bytes bytes(std::span<const T> values) {
  const auto raw=std::as_bytes(values);return {raw.begin(),raw.end()};
}
inline c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions,
                           c::DataType type=c::DataType::kBFloat16) {
  return take(c::TensorDesc::dense(type,{dimensions.begin(),dimensions.size()}),"typed original description");
}

inline constexpr std::int64_t kHidden = 4096, kJoined = 8192, kQuery = 1536;
inline constexpr float kEpsilon = 1.0e-5F;
inline constexpr auto kEhSha = "17af8402a6b674a462f2847612b2477e76c0de7dfc5ff9fba50ead1e5c88d649";
inline constexpr auto kEnSha = "187f2b4efe991b8c69a1a0365ded195b7fd463d50a7facd0d7196402a66dcc54";
inline constexpr auto kHnSha = "3fbc01762fea938ae24e94ece075b03dd5cb44cbe218884712b35d865b820501";
inline constexpr auto kQaSha = "615e317d19a83114b756e4634599ebba333c95241f3e775c77cc1633802d8be6";
inline constexpr auto kScaleSha = "c93393c24ed2c12f5458fbf88a257df91e22f586acc58436b861465c56f4d3fb";
inline constexpr auto kInputNormSha = "34ebd6b797e89011997e3f8fa674e4d0b9bc322d88bee17a91fe0a5421263e0c";
inline constexpr auto kQueryNormSha = "33fec4645dc799a2a96b870b48012fcd096e66c2003c528881d98cd397afa96a";

using Pins = std::array<std::string_view, 7>;
inline constexpr Pins kDefaultPins{kEhSha,kEnSha,kHnSha,kQaSha,kScaleSha,kInputNormSha,kQueryNormSha};

inline std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}
// Every named F32 stage remains a rounding boundary in the independent oracle.
inline float f32(double value) { volatile float rounded = static_cast<float>(value); return rounded; }
inline float multiply(float a, float b) { return f32(double(a) * double(b)); }

struct Data final {
  std::vector<std::uint16_t> eh, enorm, hnorm, input_norm, query_norm;
  Bytes qa, canonical;
  std::vector<float> scales;
  static Data load(const std::filesystem::path &input, const std::filesystem::path &query,
                   const Pins &pins = kDefaultPins) {
    for (const auto pin : pins)
      if (pin.size()!=64 || !std::ranges::all_of(pin,[](char ch) {
            return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f'); }))
        throw std::runtime_error("GLM prefix requires every actual original pin");
    Data result;
    result.eh = original::read_bfloat(input / "eh_proj.weight.bf16", kHidden * kJoined, pins[0]);
    result.enorm = original::read_bfloat(input / "enorm.weight.bf16", kHidden, pins[1]);
    result.hnorm = original::read_bfloat(input / "hnorm.weight.bf16", kHidden, pins[2]);
    result.input_norm = original::read_bfloat(query / "input_layernorm.weight.bf16", kHidden, pins[5]);
    result.query_norm = original::read_bfloat(query / "q_a_layernorm.weight.bf16", kQuery, pins[6]);
    result.qa = original::read_bytes(query / "q_a.weight.e4m3", kQuery * kHidden, pins[3]);
    result.scales = original::read_float(query / "q_a.weight_scale.f32", (kQuery / 128) * (kHidden / 128), pins[4]);
    for (auto value : result.scales)
      if (!(value > 0)) throw std::runtime_error("original q_a requires positive finite stored multiplier");
    for (auto value : result.qa)
      if ((std::to_integer<unsigned>(value) & 0x7fu) == 0x7fu)
        throw std::runtime_error("original q_a has nonfinite E4 code");

    const auto code_desc = dense({kQuery, kHidden}, c::DataType::kFloat8E4M3FN);
    const auto scale_desc = dense({kQuery / 128, kHidden / 128}, c::DataType::kFloat32);
    const auto carrier_desc = take(c::float8_weight_description(kHidden, kQuery,
        c::Float8ScaleMode::kFloat32Block128), "original block128 carrier");
    auto codes = take(m::Buffer::create(m::system_resource(), result.qa.size(), 64), "cold original E4 storage");
    auto scales = take(m::Buffer::create(m::system_resource(), result.scales.size() * 4, 64), "cold original scale storage");
    auto carrier = take(m::Buffer::create(m::system_resource(), take(carrier_desc.required_bytes(), "carrier bytes"), 64), "cold canonical storage");
    check(codes.view().write(result.qa), "unchanged original E4 bytes");
    check(scales.view().write(std::as_bytes(std::span(result.scales))), "unchanged original F32 multiplier bits");
    check(c::import_float8_block128_weight(
        take(c::ConstTensorView::create(code_desc, codes.view()), "source E4 tensor"),
        take(c::ConstTensorView::create(scale_desc, scales.view()), "source F32 scale tensor"),
        take(c::TensorView::create(carrier_desc, carrier.view()), "destination canonical tensor")), "cold original block128 import");
    result.canonical.resize(carrier.size());
    check(carrier.view().read(result.canonical), "read cold imported carrier");
    auto independent = result.qa;
    const auto scale_bytes = bytes<float>(result.scales);
    independent.insert(independent.end(), scale_bytes.begin(), scale_bytes.end());
    if (result.canonical != independent) throw std::runtime_error("original importer changed codes or scale order/bits");
    return result;
  }
};

enum class Parameter : std::size_t {
  kEmbeddingNorm,kHiddenNorm,kEhProjection,kInputNorm,kQueryProjection,kQueryNorm,kCount
};
inline c::TensorDesc description(Parameter parameter) {
  switch(parameter) {
  case Parameter::kEmbeddingNorm:
  case Parameter::kHiddenNorm:
  case Parameter::kInputNorm:return dense({kHidden});
  case Parameter::kEhProjection:return dense({kHidden,kJoined});
  case Parameter::kQueryProjection:return take(c::float8_weight_description(kHidden,kQuery,
      c::Float8ScaleMode::kFloat32Block128),"original q_a packed type");
  case Parameter::kQueryNorm:return dense({kQuery});
  default:throw std::runtime_error("unknown original GLM prefix parameter");
  }
}
inline std::span<const std::byte> payload(const Data &data,Parameter parameter) {
  switch(parameter) {
  case Parameter::kEmbeddingNorm:return std::as_bytes(std::span(data.enorm));
  case Parameter::kHiddenNorm:return std::as_bytes(std::span(data.hnorm));
  case Parameter::kEhProjection:return std::as_bytes(std::span(data.eh));
  case Parameter::kInputNorm:return std::as_bytes(std::span(data.input_norm));
  case Parameter::kQueryProjection:return data.canonical;
  case Parameter::kQueryNorm:return std::as_bytes(std::span(data.query_norm));
  default:throw std::runtime_error("unknown original GLM prefix parameter");
  }
}
struct Prefix final {
  Value hn,en,joined,eh,nx,qa,qr;
  std::array<Value,7> values() const {return {hn,en,joined,eh,nx,qa,qr};}
};
// Factory returns an ordinary constant Value in this SAME Fragment. The full
// state consumer can bind an already-uploaded readonly owner/lease here;
// neither parameter bytes nor runtime permissions are captured by this helper.
// M1 is the existing finite NextN path; inputs must already be actual BF16.
// Do not infer or insert a narrowing cast from an arbitrary F32 input.
template<class ConstantFactory>
inline Prefix append_with_constants(sem::Fragment &f,Value hidden,Value embedding,
                                      ConstantFactory &&factory,
                                      glm_fp8_arithmetic::Arithmetic arithmetic = {}) {
  arithmetic.require();
  const auto row=dense({1,kHidden});
  if(hidden.ordinal()>=f.descriptions.size() || embedding.ordinal()>=f.descriptions.size() ||
      f.description(hidden)!=row || f.description(embedding)!=row)
    throw std::runtime_error("original GLM prefix needs actual BF16[1,4096] hidden and embedding");
  const auto constant=[&](Parameter parameter) {
    const auto value=factory(parameter);
    if(value.ordinal()>=f.descriptions.size() || f.description(value)!=description(parameter) ||
        std::none_of(f.constants.begin(),f.constants.end(),
            [&](Value entry){return entry.ordinal()==value.ordinal();}))
      throw std::runtime_error("original GLM prefix parameter description differs");
    return value;
  };
  // Preserve the original chain's cold-constant and operation ordering.
  const auto ew=constant(Parameter::kEmbeddingNorm);
  const auto hw=constant(Parameter::kHiddenNorm);
  const auto ehw=constant(Parameter::kEhProjection);
  const auto iw=constant(Parameter::kInputNorm);
  const auto qw=constant(Parameter::kQueryProjection);
  const auto qnw=constant(Parameter::kQueryNorm);
  const auto hn=f.one(c::Operation(c::RmsNorm{.epsilon=kEpsilon}),{hidden,hw});
  const auto en=f.one(c::Operation(c::RmsNorm{.epsilon=kEpsilon}),{embedding,ew});
  const auto joined=f.one(c::Operation(c::Concat{.axis=1}),{en,hn});
  const std::array<std::int32_t,2> axes{1,0};
  const auto transposed=f.one(c::Operation(take(c::View::permute(axes),"original EH transpose")),{ehw});
  const auto eh=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{joined,transposed});
  const auto nx=f.one(c::Operation(c::RmsNorm{.epsilon=kEpsilon}),{eh,iw});
  const auto qa=f.one(c::model_ops::float8_block_linear(
      arithmetic.policy,arithmetic.absmax_floor),{nx,qw});
  const auto qr=f.one(c::Operation(c::RmsNorm{.epsilon=kEpsilon}),{qa,qnw});
  return {hn,en,joined,eh,nx,qa,qr};
}
inline Prefix append(sem::Fragment &f,Value hidden,Value embedding,const Data &data) {
  return append_with_constants(f,hidden,embedding,[&](Parameter parameter) {
    const auto raw=payload(data,parameter);
    return f.constant(description(parameter),Bytes(raw.begin(),raw.end()));
  });
}

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
} // namespace glm_original_mtp_roles
