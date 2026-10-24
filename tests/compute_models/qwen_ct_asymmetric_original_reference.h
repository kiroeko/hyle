#pragma once

#include "qwen_bfloat_original_reference.h"

// Actual BF16 A and original CT bytes. Integer codes and zero points unpack
// signed with the same -8 offset, which cancels. Product is stored BF16 BEFORE
// the dot; there is no GPTQ +1, F16 product, g_idx, activation QDQ or full W.
namespace qwen_ct_asymmetric_original_reference {
namespace bf=qwen_bfloat_original_reference;
inline std::uint32_t u32(std::span<const std::byte> bytes,std::size_t index) {
  if(index>=bytes.size()/4)throw std::runtime_error("original CT word address exceeds its role");
  std::uint32_t value=0;
  for(std::size_t lane=0;lane<4;++lane)value|=std::to_integer<std::uint32_t>(bytes[4*index+lane])<<(8*lane);
  return value;
}
inline std::uint16_t u16(std::span<const std::byte> bytes,std::size_t index) {
  if(index>=bytes.size()/2)throw std::runtime_error("original CT scale address exceeds its role");
  return std::uint16_t(std::to_integer<unsigned>(bytes[2*index])|(std::to_integer<unsigned>(bytes[2*index+1])<<8U));
}
inline bool expect_projection(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> input,
    std::span<const std::byte> codes,std::span<const std::byte> zeros,std::span<const std::byte> scales,
    std::size_t outputs,std::size_t width,std::size_t group,std::string_view role,bool full) {
  if(!outputs||!width||!group||width%8||width%group||
      outputs>std::numeric_limits<std::size_t>::max()/width/4||input.empty()||input.size()%width||
      input.size()/width>std::numeric_limits<std::size_t>::max()/outputs||actual.size()!=input.size()/width*outputs) {
    ADD_FAILURE()<<role<<" original CT dimensions";return false;
  }
  const auto groups=width/group,zero_rows=outputs/8+(outputs%8!=0),rows=input.size()/width;
  if(codes.size()!=outputs*(width/8)*4||zeros.size()!=zero_rows*groups*4||scales.size()!=outputs*groups*2) {
    ADD_FAILURE()<<role<<" original CT codes/Z/BF16SF extents";return false;
  }
  std::vector<std::uint16_t> decoded(width);
  for(std::size_t row=0;row<rows;++row) {
    const auto values=actual.subspan(row*outputs,outputs);
    const auto candidates=bf::detail::candidates(bf::widen(values));
    const auto check=[&](std::size_t output) {
      for(std::size_t column=0;column<width;++column) {
        const auto g=column/group;
        const auto code=(u32(codes,output*(width/8)+column/8)>>(4*(column%8)))&15U;
        const auto zero=(u32(zeros,(output/8)*groups+g)>>(4*(output%8)))&15U;
        const auto scale_bits=u16(scales,output*groups+g);
        const auto coefficient=bf::widen(scale_bits);
        if(!std::isfinite(coefficient)||((scale_bits&0x7f80U)==0&&(scale_bits&0x007fU)!=0))
          throw std::runtime_error("original CT SF needs finite normal-or-zero BF16 source or separate DAZ analysis");
        // At most 12 significant bits: this original BF16 scale times an
        // exact [-15,15] integer is exact F32 before its actual BF16 product.
        const auto product=float(int(code)-int(zero))*coefficient;
        decoded[column]=bf::round(product);
        if(!std::isfinite(bf::widen(decoded[column])))throw std::runtime_error("original CT product requires separate overflow analysis");
      }
      const auto dot=bf::detail::dot(input.subspan(row*width,width),std::as_bytes(std::span(decoded)));
      const auto low=std::nextafter(float(dot.center-dot.bound),-std::numeric_limits<float>::infinity());
      const auto high=std::nextafter(float(dot.center+dot.bound),std::numeric_limits<float>::infinity());
      if(!std::isfinite(low)||!std::isfinite(high))throw std::runtime_error("CT BF16 return needs separate overflow analysis");
      const auto observed=bf::widen(values[output]),lower=bf::widen(bf::round(low)),upper=bf::widen(bf::round(high));
      if(!std::isfinite(observed)||observed<lower||observed>upper) {
        ADD_FAILURE()<<role<<" ["<<row<<','<<output<<"] actual="<<observed<<" original CT F64="<<dot.center
            <<" F32 bound="<<dot.bound<<" BF16 interval=["<<lower<<','<<upper<<']';return false;
      }
      return true;
    };
    if(full){for(std::size_t output=0;output<outputs;++output)if(!check(output))return false;}
    else {for(const auto output:candidates)if(!check(output))return false;}
  }
  return true;
}
} // namespace qwen_ct_asymmetric_original_reference
