#pragma once

#include "../compute_provider_native/half_test_support.h"
#include "qwen_bfloat_original_reference.h"

// Original GPTQ-v1 bytes and actual BF16 input. There is NO activation QDQ.
// Integer centering -> original F16 scale/product store -> BF16 W operand ->
// F32 dot -> BF16 return mirrors the explicit public operation chain. One W
// row is decoded at a time; no complete dense weight bank or native oracle.
namespace qwen_gptq_original_reference {
namespace bf=qwen_bfloat_original_reference;
using Half=hyle::simd::float16_t;
inline std::uint32_t u32(std::span<const std::byte> bytes,std::size_t index) {
  if(index>=bytes.size()/4)throw std::runtime_error("original GPTQ word address exceeds its role");
  std::uint32_t result=0;
  for(std::size_t lane=0;lane<4;++lane)result|=std::to_integer<std::uint32_t>(bytes[4*index+lane])<<(8*lane);
  return result;
}
inline float scale(std::span<const std::byte> bytes,std::size_t index) {
  if(index>=bytes.size()/2)throw std::runtime_error("original GPTQ scale address exceeds its role");
  const auto bits=std::uint16_t(std::to_integer<unsigned>(bytes[2*index])|
      (std::to_integer<unsigned>(bytes[2*index+1])<<8U));
  const auto value=float(half_test::decode_bits<Half>(bits));
  if(!std::isfinite(value))throw std::runtime_error("original GPTQ scale needs separate nonfinite analysis");
  return value;
}
inline bool interval(std::uint16_t actual,const bf::detail::Dot &dot,std::string_view role,
    std::size_t row,std::size_t column) {
  const auto low=std::nextafter(float(dot.center-dot.bound),-std::numeric_limits<float>::infinity());
  const auto high=std::nextafter(float(dot.center+dot.bound),std::numeric_limits<float>::infinity());
  if(!std::isfinite(low)||!std::isfinite(high))throw std::runtime_error("GPTQ BF16 return needs separate overflow analysis");
  const auto value=bf::widen(actual),lower=bf::widen(bf::round(low)),upper=bf::widen(bf::round(high));
  if(!std::isfinite(value)||value<lower||value>upper) {
    ADD_FAILURE()<<role<<" ["<<row<<','<<column<<"] actual="<<value<<" original GPTQ F64="<<dot.center
        <<" F32 bound="<<dot.bound<<" BF16 interval=["<<lower<<','<<upper<<']';return false;
  }
  return true;
}
inline bool expect_projection(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> input,
    std::span<const std::byte> qweight,std::span<const std::byte> qzeros,std::span<const std::byte> scales,
    std::span<const std::byte> group_index,std::size_t outputs,std::size_t width,std::size_t group,
    std::string_view role,bool full) {
  if(!outputs||!width||!group||outputs%8||width%8||width%group||
      outputs>std::numeric_limits<std::size_t>::max()/width/4||input.empty()||input.size()%width||
      input.size()/width>std::numeric_limits<std::size_t>::max()/outputs||actual.size()!=input.size()/width*outputs||
      qweight.size()!=(width/8)*outputs*4||qzeros.size()!=(width/group)*(outputs/8)*4||
      scales.size()!=(width/group)*outputs*2||group_index.size()!=width*4) {
    ADD_FAILURE()<<role<<" original GPTQ Q/Z/SF/g_idx extents";return false;
  }
  const auto groups=width/group,rows=input.size()/width;
  std::vector<std::size_t> selected_groups(width);
  for(std::size_t column=0;column<width;++column) {
    const auto selected=std::bit_cast<std::int32_t>(u32(group_index,column));
    if(selected<0||std::size_t(selected)>=groups)
      throw std::runtime_error("original GPTQ g_idx is outside the original group bank");
    selected_groups[column]=std::size_t(selected);
  }
  std::vector<std::uint16_t> decoded(width);
  for(std::size_t row=0;row<rows;++row) {
    const auto values=actual.subspan(row*outputs,outputs);
    const auto widened=bf::widen(values);
    const auto candidates=bf::detail::candidates(widened);
    const auto check=[&](std::size_t output) {
      for(std::size_t column=0;column<width;++column) {
        const auto selected=selected_groups[column];
        const auto code=(u32(qweight,(column/8)*outputs+output)>>(4*(column%8)))&15U;
        // GPTQ stores zero-1. In particular raw15 means zero16, not zero0.
        const auto zero=((u32(qzeros,selected*(outputs/8)+output/8)>>(4*(output%8)))&15U)+1U;
        const auto coefficient=scale(scales,selected*outputs+output);
        const auto product=half_test::encode<Half>(double(int(code)-int(zero))*double(coefficient));
        const auto source=float(half_test::decode(product));
        if(!std::isfinite(source))throw std::runtime_error("original GPTQ F16 product needs separate overflow analysis");
        decoded[column]=bf::round(source);
      }
      const auto dot=bf::detail::dot(input.subspan(row*width,width),std::as_bytes(std::span(decoded)));
      return interval(values[output],dot,role,row,output);
    };
    if(full){for(std::size_t output=0;output<outputs;++output)if(!check(output))return false;}
    else {for(const auto output:candidates)if(!check(output))return false;}
  }
  return true;
}
} // namespace qwen_gptq_original_reference
