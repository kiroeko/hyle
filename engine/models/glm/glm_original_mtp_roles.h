#pragma once


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
#include "../detail/component_definition.h"
#include "../detail/original_role_input.h"
#include "glm_fp8_arithmetic.h"

// Original GLM-5.3-Flash NextN layer45, fixed official revision
// eb9eb208eb0d988989d07a6a12d0fdeb5f52574a. Test input and pure Fragment
// construction only: no Graph/Invocation, backend macros, executor or loader
// policy. The seven original bodies become six immutable runtime banks because
// q_a keeps its unchanged E4 codes and F32 block scales in one packed carrier.
namespace glm_original_mtp_roles {
namespace c=hyle::compute;
namespace m=hyle::mem;
namespace sem=hyle::inference::model_definition;
namespace original=original_role_input;
using Bytes=std::vector<std::byte>;
using Value=sem::ValueId;

template<class T> inline T take(hyle::Result<T> result,std::string_view where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status,std::string_view where);
template<class T> inline Bytes bytes(std::span<const T> values) {
  const auto raw=std::as_bytes(values);return {raw.begin(),raw.end()};
}
c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions,
                           c::DataType type=c::DataType::kBFloat16);

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
c::TensorDesc description(Parameter parameter);
std::span<const std::byte> payload(const Data &data,Parameter parameter);
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
Prefix append(sem::Fragment &f,Value hidden,Value embedding,const Data &data);

} // namespace glm_original_mtp_roles
