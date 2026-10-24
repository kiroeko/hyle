#pragma once

#include "nvfp4_reference_fixture.h"
#include "qwen_bfloat_original_reference.h"

// Independent original-byte ModelOpt QDQ followed by a F32-dot certificate.
// Only one decoded W row and the current A rows are held; never a full dense
// shadow. The actual operation returns BF16, so the final gate is an admitted
// BF16 rounding interval, not equality against an invented observable rawF32.
namespace qwen_nvfp4_original_reference {
namespace scalar=nvfp4_reference_test;
namespace bf=qwen_bfloat_original_reference;
inline float global(std::span<const std::byte> raw,std::string_view role) {
  if(raw.size()!=4)throw std::runtime_error("NVFP4 original global is not one F32 scalar");
  std::uint32_t bits=0;
  for(std::size_t i=0;i<4;++i)bits|=std::to_integer<std::uint32_t>(raw[i])<<(8*i);
  const float value=std::bit_cast<float>(bits);
  // A positive F32 subnormal global needs its own pointwise-DAZ analysis.
  // Do not assume the normal-global certificate covers that separate domain.
  if(!(value>0.F)||!std::isnormal(value))
    throw std::runtime_error(std::string(role)+" needs separate nonnormal global-scale analysis");
  return value;
}
inline bool interval(std::uint16_t actual,const bf::detail::Dot &expected,std::string_view role,
    std::size_t row,std::size_t output) {
  const float low=std::nextafter(float(expected.center-expected.bound),-std::numeric_limits<float>::infinity());
  const float high=std::nextafter(float(expected.center+expected.bound),std::numeric_limits<float>::infinity());
  if(!std::isfinite(low)||!std::isfinite(high))
    throw std::runtime_error("NVFP4 BF16 result interval needs separate overflow analysis");
  const float value=bf::widen(actual),lower=bf::widen(bf::round(low)),upper=bf::widen(bf::round(high));
  if(!std::isfinite(value)||value<lower||value>upper) {
    ADD_FAILURE()<<role<<" ["<<row<<','<<output<<"] actual="<<value<<" F64="<<expected.center
        <<" F32 bound="<<expected.bound<<" BF16 interval=["<<lower<<','<<upper<<']';return false;
  }
  return true;
}
inline bool expect_projection(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> input,
    std::span<const std::byte> codes,std::span<const std::byte> scales,
    std::span<const std::byte> weight_global,std::span<const std::byte> input_global,
    std::size_t outputs,std::size_t width,std::string_view role,bool full) {
  if(!outputs||!width||width%16||outputs>std::numeric_limits<std::size_t>::max()/width||
      input.empty()||input.size()%width||input.size()/width>std::numeric_limits<std::size_t>::max()/outputs||
      codes.size()!=outputs*(width/2)||scales.size()!=outputs*(width/16)||actual.size()!=input.size()/width*outputs) {
    ADD_FAILURE()<<role<<" original NVFP4 projection extent";return false;
  }
  const float wg=global(weight_global,role),ig=global(input_global,role);
  const auto rows=input.size()/width;
  std::vector<std::uint16_t> activation(input.size()),decoded(width);
  for(std::size_t row=0;row<rows;++row)for(std::size_t first=0;first<width;first+=16) {
    float maximum=0.F;
    for(std::size_t lane=0;lane<16;++lane)
      maximum=std::max(maximum,std::abs(bf::detail::normal(input[row*width+first+lane])));
    // Every named pointwise F32 operation, E4 and BF16 store is taken from
    // the existing independent ModelOpt scalar oracle, not product helpers.
    for(std::size_t lane=0;lane<16;++lane) {
      const auto offset=row*width+first+lane;
      const float value=scalar::activation_value(bf::widen(input[offset]),maximum,ig);
      if(!std::isfinite(value))throw std::runtime_error("NVFP4 actual A QDQ requires finite stored operands");
      activation[offset]=bf::round(value);
    }
  }
  // Validate every original scale, even when later invocations only check
  // candidate rows. A successful Completion must not conceal an invalid SF.
  for(const auto raw:scales) {
    const float value=scalar::e4m3(std::to_integer<std::uint8_t>(raw));
    if(!std::isfinite(value)||value<0.F)throw std::runtime_error("invalid original NVFP4 block scale");
    const float effective=scalar::multiply(value,wg);
    if(!std::isfinite(effective)||(effective!=0.F&&!std::isnormal(effective)))
      throw std::runtime_error("NVFP4 W scale needs separate overflow/pointwise-DAZ analysis");
  }
  for(std::size_t row=0;row<rows;++row) {
    const auto values=actual.subspan(row*outputs,outputs);const auto widened=bf::widen(values);
    const auto chosen=bf::detail::candidates(widened);
    const auto check=[&](std::size_t output) {
      for(std::size_t column=0;column<width;++column) {
        const auto packed=std::to_integer<unsigned>(codes[output*(width/2)+column/2]);
        const auto code=(packed>>(4*(column%2)))&15U;
        const float value=scalar::weight_value(code,
            std::to_integer<std::uint8_t>(scales[output*(width/16)+column/16]),wg);
        if(!std::isfinite(value))throw std::runtime_error("NVFP4 decoded original weight is not finite");
        decoded[column]=bf::round(value);
      }
      // Reuse the existing normal-source BF16 product/FTZ proof after the
      // separate pointwise QDQ. Nonzero BF16 subnormals remain explicit hard
      // analysis failures; no unsupported source domain is silently skipped.
      const auto dot=bf::detail::dot(std::span(activation).subspan(row*width,width),std::as_bytes(std::span(decoded)));
      return interval(values[output],dot,role,row,output);
    };
    if(full){for(std::size_t output=0;output<outputs;++output)if(!check(output))return false;}
    else {for(const auto output:chosen)if(!check(output))return false;}
  }
  return true;
}
} // namespace qwen_nvfp4_original_reference
