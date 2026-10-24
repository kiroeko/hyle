#pragma once

#include <numeric>

#include "glm_original_attention_reference.h"
#include "qwen_bfloat_original_reference.h"

// Independent numerical MXFP4 oracle. The threshold/binade derivation is the
// existing mxfp4_reference_fixture scalar proof, not product bit-mask code.
// Only one decoded original W row is temporary; no floating expert bank.
namespace glm_mxfp4_moe_reference {
namespace bf=qwen_bfloat_original_reference;
namespace attention=glm_original_attention_reference;
struct Matrix final {
  std::span<const std::byte> codes,scales;
  std::size_t outputs,width;
};
inline float f32(double value) {volatile float result=static_cast<float>(value);return result;}
inline float scale(float maximum) {
  if(maximum==0)return std::ldexp(1.F,-127);
  if(double(maximum)>=std::ldexp(1.75,127))return std::ldexp(1.F,127);
  int exponent=0;const auto fraction=std::frexp(double(maximum),&exponent);
  return std::ldexp(1.F,std::clamp(exponent-(fraction>=.875?2:3),-127,127));
}
inline float nearest(float value) {
  constexpr std::array<float,8> magnitudes{0.F,.5F,1.F,1.5F,2.F,3.F,4.F,6.F};
  unsigned best=0;double distance=std::abs(double(value));
  for(unsigned code=1;code<magnitudes.size();++code) {
    const auto next=std::abs(std::abs(double(value))-double(magnitudes[code]));
    if(next<distance||(next==distance&&!(code&1u))){best=code;distance=next;}
  }
  return std::copysign(magnitudes[best],value);
}
inline std::vector<std::uint16_t> quantize(std::span<const std::uint16_t> actual) {
  if(actual.empty()||actual.size()%32)throw std::runtime_error("MXFP4 actual A needs complete groups32");
  std::vector<std::uint16_t> result(actual.size());
  for(std::size_t group=0;group<actual.size()/32;++group) {
    float maximum=0;
    for(std::size_t lane=0;lane<32;++lane)
      maximum=std::max(maximum,std::abs(bf::detail::normal(actual[group*32+lane])));
    const auto s=scale(maximum);
    if(maximum!=0&&!std::isnormal(s))
      throw std::runtime_error("MXFP4 A scale needs separate subnormal quantization analysis");
    for(std::size_t lane=0;lane<32;++lane) {
      const auto value=bf::widen(actual[group*32+lane]);
      const auto normalized=bf::widen(bf::round(f32(double(value)/double(s))));
      result[group*32+lane]=bf::round(f32(double(nearest(normalized))*double(s)));
    }
  }
  return result;
}
inline bool expect_projection(std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> actual_input,Matrix original,const char *role,bool full) {
  const auto n=original.outputs,k=original.width;
  if(!n||!k||k%32||n>std::numeric_limits<std::size_t>::max()/k||
      actual.size()!=n||actual_input.size()!=k||original.codes.size()!=n*(k/2)||original.scales.size()!=n*(k/32)) {
    ADD_FAILURE()<<role<<" original MXFP4 projection extent";return false;
  }
  const auto activation=quantize(actual_input);
  constexpr std::array<float,8> magnitudes{0.F,.5F,1.F,1.5F,2.F,3.F,4.F,6.F};
  std::vector<std::uint16_t> weight(k);
  std::vector<std::size_t> rows;
  if(full) {rows.resize(n);std::iota(rows.begin(),rows.end(),0);}
  else for(const auto row:std::array<std::size_t,7>{0,1,n/4,n/2,3*n/4,n-2,n-1})
    if(row<n&&std::find(rows.begin(),rows.end(),row)==rows.end())rows.push_back(row);
  for(const auto row:rows) {
    for(std::size_t col=0;col<k;++col) {
      const auto packed=std::to_integer<unsigned>(original.codes[row*(k/2)+col/2]);
      const auto code=(packed>>(4*(col%2)))&15U;
      const auto sf=std::to_integer<unsigned>(original.scales[row*(k/32)+col/32]);
      if(sf==255)throw std::runtime_error("original MXFP4 SF255 is nonfinite");
      if(sf==0&&(code&7U)!=0)
        throw std::runtime_error("MXFP4 nonzero SF0 weight needs separate decode/FTZ analysis");
      const auto multiplier=bf::widen(bf::round(std::ldexp(1.F,int(sf)-127)));
      const auto value=code&8U?-magnitudes[code&7U]:magnitudes[code&7U];
      weight[col]=bf::round(f32(double(value)*double(multiplier)));
    }
    // Same parsed F32 accumulation/FTZ certificate as the existing original
    // BF16 dot. Nonfinite/overflow and BF16-subnormal operands remain explicit
    // unsupported reference domains, never empirical output tolerances.
    const auto dot=bf::detail::dot(activation,std::as_bytes(std::span(weight)));
    if(!attention::detail::bfloat_interval(actual[row],{dot.center,dot.bound},role,0,row))return false;
  }
  return true;
}
template<class Source>
bool expect_selected_projection(std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> actual_input,std::span<const std::int32_t> ids,
    Source &&source,const char *role,bool full) {
  if(ids.empty()){ADD_FAILURE()<<role<<" actual MXFP4 IDs missing";return false;}
  const auto first=source(ids.front());
  if(!first.outputs||!first.width||first.outputs>std::numeric_limits<std::size_t>::max()/ids.size()||
      first.width>std::numeric_limits<std::size_t>::max()/ids.size()||actual.size()!=ids.size()*first.outputs||
      (actual_input.size()!=first.width&&actual_input.size()!=ids.size()*first.width)) {
    ADD_FAILURE()<<role<<" selected MXFP4 actual extent";return false;
  }
  for(std::size_t slot=0;slot<ids.size();++slot) {
    const auto matrix=source(ids[slot]);
    if(matrix.outputs!=first.outputs||matrix.width!=first.width)
      throw std::runtime_error("selected original MXFP4 expert geometry differs");
    const auto input_row=actual_input.size()==first.width?0:slot;
    if(!expect_projection(actual.subspan(slot*first.outputs,first.outputs),
        actual_input.subspan(input_row*first.width,first.width),matrix,role,full))return false;
  }
  return true;
}
} // namespace glm_mxfp4_moe_reference
