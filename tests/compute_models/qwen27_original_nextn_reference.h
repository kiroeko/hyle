#pragma once

#include <ggml.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dspark_original_shared_reference.h"
#include "packed_projection_failure_capture.h"

// Local, original-byte packed references for the selected main Q27 NextN.
// Native activation codecs are part of the equation, not a tolerance against
// an ideal raw-input dot. No generated prototypes or native reference submit.
namespace qwen27_original_nextn_reference {
namespace q8 = dspark_original_shared_reference;
enum class ActivationCodec { kCpuAvx2, kCudaMmvqQ8_1 };
namespace detail {
inline std::size_t block_bytes(ggml_type type) {
  switch(type) {
  case GGML_TYPE_Q3_K:return 110;
  case GGML_TYPE_Q4_K:return 144;
  case GGML_TYPE_Q6_K:return 210;
  case GGML_TYPE_Q8_0:return 34;
  case GGML_TYPE_NVFP4:return 36;
  default:throw std::runtime_error("original Q27 reference unsupported packed codec");
  }
}
inline std::size_t block_width(ggml_type type) {
  (void)block_bytes(type);
  return type==GGML_TYPE_Q8_0?32:type==GGML_TYPE_NVFP4?64:256;
}
inline unsigned byte(std::span<const std::byte> data,std::size_t offset) {
  return std::to_integer<unsigned>(data[offset]);
}
inline int signed_byte(std::span<const std::byte> data,std::size_t offset) { return q8::detail::code(data[offset]); }
// GGUF NVFP4 has four inline UE4M3 scales per64 values. The integer table
// stores twice E2M1, so the scale includes1/2. This is weight-only GGUF,
// independently of ModelOpt's separate activation/global-scale contract.
inline double nvfp4_scale(unsigned code) {
  if(code==0||code==0x7f)return 0;
  if(code>0x7f)throw std::runtime_error("original NVFP4 scale exceeds UE4M3");
  const int exponent=int(code>>3),mantissa=int(code&7U);
  return exponent?std::ldexp(double(8+mantissa),exponent-11):std::ldexp(double(mantissa),-10);
}
inline double distance(double left,double right) {return left==right?0:q8::detail::upper(std::abs(left-right));}

// Q3_K: hmask[32], qs[64], packed signed-six-bit scales[12], F16 d.
// Decode each scale from its own low nibble and two high bits, independently
// of the native word/shuffle implementations; scale16 is in [-32,31].
inline int q3_scale(std::span<const std::byte> block,std::size_t scale16) {
  const auto low=(byte(block,96+scale16%8)>>(4*(scale16/8)))&15U;
  const auto high=(byte(block,104+scale16%4)>>(2*(scale16/4)))&3U;
  return int(low|(high<<4U))-32;
}
inline int q3_code(std::span<const std::byte> block,std::size_t column) {
  const auto within=column%128;
  const auto low=(byte(block,32+(column/128)*32+within%32)>>(2*(within/32)))&3U;
  const bool positive=(byte(block,column%32)&(1U<<(column/32)))!=0;
  return int(low)-(positive?0:4);
}

// Independently decode the original six-bit scale/min packing. This is the
// checkpoint's output-row layout, not a native dot/dequantize function call.
inline std::pair<int,int> q4_scale(std::span<const std::byte> block,std::size_t group) {
  if(group<4)return {int(byte(block,4+group)&63U),int(byte(block,8+group)&63U)};
  return {int((byte(block,8+group)&15U)|((byte(block,group)>>6U)<<4U)),
      int((byte(block,8+group)>>4U)|((byte(block,4+group)>>6U)<<4U))};
}
inline int q6_code(std::span<const std::byte> block,std::size_t column) {
  const auto half=column/128,within=column%128,lane=within%32,quarter=within/32;
  const auto low=byte(block,half*64+lane+(quarter%2)*32);
  const auto high=byte(block,128+half*32+lane);
  return int(((low>>(quarter>=2?4U:0U))&15U)|(((high>>(quarter*2))&3U)<<4U))-32;
}
struct WeightGroup final {
  double scale=0,minimum=0;
  std::array<int,32> codes{},offsets{};
};
inline WeightGroup weight(std::span<const std::byte> row,ggml_type type,std::size_t group) {
  WeightGroup result;
  if(type==GGML_TYPE_NVFP4) {
    static constexpr std::array<int,16> doubled{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    const auto block=row.subspan(group/2*36,36);
    const auto first_subgroup=(group%2)*2;
    result.scale=nvfp4_scale(byte(block,first_subgroup));
    result.minimum=nvfp4_scale(byte(block,first_subgroup+1));
    // Two16-value independently scaled components in one activation32.
    // Retaining both prevents cancellation from hiding native rounding.
    for(std::size_t i=0;i<16;++i) {
      const auto shift=i<8?0U:4U;
      result.codes[i]=doubled[(byte(block,4+first_subgroup*8+i%8)>>shift)&15U];
      result.offsets[16+i]=doubled[(byte(block,4+(first_subgroup+1)*8+i%8)>>shift)&15U];
    }
  } else if(type==GGML_TYPE_Q8_0) {
    const auto block=row.subspan(group*34,34);result.scale=q8::detail::scale(block,0);
    for(std::size_t i=0;i<32;++i)result.codes[i]=signed_byte(block,2+i);
  } else {
    const auto block=row.subspan((group/8)*block_bytes(type),block_bytes(type));
    const auto local=group%8;
    if(type==GGML_TYPE_Q3_K) {
      result.scale=q8::detail::scale(block,108);
      for(std::size_t i=0;i<32;++i) {
        const auto column=local*32+i;
        result.codes[i]=q3_code(block,column)*q3_scale(block,column/16);
      }
    } else if(type==GGML_TYPE_Q6_K) {
      result.scale=q8::detail::scale(block,208);
      for(std::size_t i=0;i<32;++i) {
        const auto column=local*32+i;
        result.codes[i]=q6_code(block,column)*signed_byte(block,192+column/16);
      }
    } else {
      result.scale=q8::detail::scale(block,0);result.minimum=q8::detail::scale(block,2);
      const auto [scale,minimum]=q4_scale(block,local);
      for(std::size_t i=0;i<32;++i) {
        result.codes[i]=int((byte(block,16+(local/2)*32+i)>>(local%2?4U:0U))&15U)*scale;
        result.offsets[i]=-minimum;
      }
    }
  }
  return result;
}
inline std::vector<q8::detail::ActivationBlock> activation(std::span<const float> input,
    ggml_type type,ActivationCodec codec) {
  std::vector<q8::detail::ActivationBlock> result(input.size()/32);
  if(codec==ActivationCodec::kCudaMmvqQ8_1||type==GGML_TYPE_Q8_0||type==GGML_TYPE_NVFP4) {
    for(std::size_t group=0;group<result.size();++group)
      result[group]=q8::detail::quantize(std::span<const float,32>(input.data()+group*32,32),
          codec==ActivationCodec::kCudaMmvqQ8_1?q8::ActivationCodec::kCudaMmvqQ8_1Fast:q8::ActivationCodec::kCpuAvxQ8_0,false);
    return result;
  }
  // Actual x86 quantize_row_q8_K delegates to the scalar reference: preserve
  // FIRST signed absmax, RN(-127/max), RN(x*iscale), nearest-even integer,
  // and a distinct RN(1/iscale) F32 stored scale (no F16 narrowing).
  for(std::size_t first=0;first<input.size();first+=256) {
    float maximum=0,absolute=0;
    for(std::size_t i=0;i<256;++i) {
      const auto value=input[first+i];
      if(!std::isfinite(value)||(value!=0&&!std::isnormal(value)))
        throw std::runtime_error("original Q27 AQ8_K needs finite normal F32 inputs");
      if(std::abs(value)>absolute){absolute=std::abs(value);maximum=value;}
    }
    if(absolute==0)continue;
    const float inverse=q8::detail::rn_divide(-127.0F,maximum);
    const float stored=q8::detail::rn_divide(1.0F,inverse);
    if(!std::isnormal(inverse)||!std::isnormal(stored))
      throw std::runtime_error("original Q27 AQ8_K scale needs separate range analysis");
    for(std::size_t i=0;i<256;++i) {
      auto &group=result[(first+i)/32];const auto lane=i%32;
      const auto scaled=q8::detail::rn_multiply(input[first+i],inverse);
      const int code=std::min(127,q8::detail::rounded_code(scaled,true));
      group.scale=group.scale_low=group.scale_high=stored;
      group.codes[lane]=group.low[lane]=group.high[lane]=code;
    }
  }
  return result;
}
inline std::vector<std::size_t> rows(std::size_t count,bool full,std::span<const std::size_t> extra) {
  std::vector<std::size_t> result;
  if(full){result.resize(count);for(std::size_t i=0;i<count;++i)result[i]=i;}
  else {
    result={0,std::min(std::size_t{1},count-1),count/4,count/2,count/4*3,count>1?count-2:0,count-1};
    result.insert(result.end(),extra.begin(),extra.end());
    std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());
  }
  for(const auto row:result)if(row>=count)throw std::runtime_error("original Q27 extra reference row is out of range");
  return result;
}
inline void range(double weight_scale,const q8::detail::ActivationBlock &source) {
  for(const double active:{double(source.scale_low),double(source.scale_high)}) {
    const double combined=weight_scale*active;
    if(!std::isfinite(combined)||(combined!=0&&std::abs(combined)<0x1p-103))
      throw std::runtime_error("original Q27 scale product needs separate FTZ/overflow analysis");
  }
}
} // namespace detail

inline bool expect_projection(std::span<const float> actual,std::span<const std::byte> packed,
    std::size_t outputs,std::size_t width,ggml_type type,std::span<const float> actual_input,
    ActivationCodec codec,const char *role,bool full,std::span<const std::size_t> extra_rows={}) {
  // Q3_K supports the selected sidecar head; Q6_K/Q8_0 retain their original
  // contract. Q4_K is decoded below for Gather, not admitted for projection.
  if(type!=GGML_TYPE_Q3_K&&type!=GGML_TYPE_Q6_K&&type!=GGML_TYPE_Q8_0&&type!=GGML_TYPE_NVFP4) {
    ADD_FAILURE()<<role<<" original Q27 projection requires Q3_K, Q6_K, Q8_0 or NVFP4";return false;
  }
  const auto block=detail::block_width(type),bytes=detail::block_bytes(type);
  if(!outputs||!width||width%block||actual.size()!=outputs||actual_input.size()!=width||
      outputs>std::numeric_limits<std::size_t>::max()/(width/block)/bytes||
      packed.size()!=outputs*(width/block)*bytes||std::fegetround()!=FE_TONEAREST) {
    ADD_FAILURE()<<role<<" original packed projection extent/rounding";return false;
  }
  const auto activation=detail::activation(actual_input,type,codec);
  const auto row_bytes=(width/block)*bytes,groups=width/32;
  // In the fixed AVX2 Q6_K path the -32 compensation is merged in INT32
  // BEFORE conversion, and each centered lane contains32 terms bounded by
  // 32*128*127: 16,646,144<2^24. CUDA centers each four-code DP4A before
  // its F32 scale product. Thus |centered coefficient|, without a separate
  // uncentered correction, bounds the actual floating arithmetic. K/4+65
  // rounds dominates products and every fixed AVX2/MMVQ reduction path.
  // Q3_K also centers low2bits-(hmask?0:4) and multiplies its signed6-bit
  // scale in integers before F32. CPU AVX2 each lane holds at most32 terms:
  // 32*4*32*127=520192<2^19; CUDA each four-code partial is <2^16. No
  // separate floating correction or Q8_1.ds.y term exists. CUDA first scales
  // its four integer partials by d8, adds them, then multiplies by F16 d3;
  // the same conservative round-count bound covers this reassociation.
  const double g32=q8::detail::gamma(width/4+65,0x1p-24);
  const double g64=q8::detail::gamma(2*groups+2,0x1p-53);
  double maximum_error=0,maximum_ratio=0,maximum_ambiguity=0;
  const auto selected=detail::rows(outputs,full,extra_rows);
  for(const auto output:selected) {
    const auto row=packed.subspan(output*row_bytes,row_bytes);
    q8::detail::Sum sum;double absolute=0,ambiguity=0;
    for(std::size_t group=0;group<groups;++group) {
      const auto w=detail::weight(row,type,group);const auto &a=activation[group];
      const double maximum_scale=std::max(std::abs(double(a.scale_low)),std::abs(double(a.scale_high)));
      const double scale_error=std::max(std::abs(double(a.scale_low)-a.scale),std::abs(double(a.scale_high)-a.scale));
      const auto component=[&](double scale,const std::array<int,32> &codes) {
        if(scale==0)return;
        detail::range(scale,a);
        int dot=0,uncertain=0,maximum_absolute=0;
        for(std::size_t i=0;i<32;++i) {
          dot+=codes[i]*a.codes[i];
          uncertain+=std::abs(codes[i])*std::max(std::abs(a.low[i]-a.codes[i]),std::abs(a.high[i]-a.codes[i]));
          maximum_absolute+=std::abs(codes[i])*std::max(std::abs(a.low[i]),std::abs(a.high[i]));
        }
        sum.add(scale*double(a.scale)*double(dot));
        // Keep Q4_K d and dmin separate: collapsed decoded-weight cancellation
        // must not erase the source's two-component native arithmetic error.
        absolute=q8::detail::add_upper(absolute,q8::detail::multiply_upper(std::abs(scale),
            q8::detail::multiply_upper(maximum_scale,double(maximum_absolute))));
        ambiguity=q8::detail::add_upper(ambiguity,q8::detail::multiply_upper(std::abs(scale),
            q8::detail::add_upper(q8::detail::multiply_upper(scale_error,std::abs(double(dot))),
                q8::detail::multiply_upper(maximum_scale,double(uncertain)))));
      };
      component(w.scale,w.codes);component(w.minimum,w.offsets);
    }
    if(!std::isfinite(absolute)||absolute>double(std::numeric_limits<float>::max())/4)
      throw std::runtime_error("original Q27 dot needs separate overflow analysis");
    const double center=sum.value();
    const double oracle=q8::detail::add_upper(detail::distance(center,sum.plain),q8::detail::multiply_upper(g64,absolute));
    const double arithmetic=q8::detail::multiply_upper(g32,absolute);
    const double bound=q8::detail::add_upper(ambiguity,q8::detail::add_upper(arithmetic,oracle));
    const auto error=detail::distance(actual[output],center);
    if(!std::isfinite(actual[output])||error>bound) {
      if(type==GGML_TYPE_Q8_0)packed_projection_failure_capture::q8_failure(
          role,actual_input,row,output,actual[output],center,bound);
      ADD_FAILURE()<<role<<'['<<output<<"] actual="<<actual[output]<<" F64="<<center<<" error="<<error
          <<" code/scale ambiguity="<<ambiguity<<" nativeF32="<<arithmetic<<" sumabs="<<absolute;return false;
    }
    maximum_error=std::max(maximum_error,error);maximum_ratio=std::max(maximum_ratio,bound?error/bound:0.0);
    maximum_ambiguity=std::max(maximum_ambiguity,ambiguity);
  }
  std::printf("Q27 original %s codec=%d A=%d full=%d checked=%zu K=%zu maxAbs=%.17g maxRatio=%.17g maxAmbiguity=%.17g\n",
      role,int(type),int(codec),int(full),selected.size(),width,maximum_error,maximum_ratio,maximum_ambiguity);
  return true;
}

inline bool expect_embedding(std::span<const float> actual,std::span<const std::byte> original,
    std::size_t hidden,std::size_t vocabulary,std::int32_t actual_id,ggml_type type) {
  if(type!=GGML_TYPE_Q3_K&&type!=GGML_TYPE_Q4_K&&type!=GGML_TYPE_Q6_K&&type!=GGML_TYPE_Q8_0) {
    ADD_FAILURE()<<"Q27 original embedding requires Q3_K, Q4_K, Q6_K or Q8_0";return false;
  }
  const auto block_width=detail::block_width(type),bytes=detail::block_bytes(type);
  if(!hidden||hidden%block_width||!vocabulary||actual.size()!=hidden||actual_id<0||std::size_t(actual_id)>=vocabulary||
      vocabulary>std::numeric_limits<std::size_t>::max()/(hidden/block_width)/bytes||
      original.size()!=vocabulary*(hidden/block_width)*bytes) {
    ADD_FAILURE()<<"Q27 actual original embedding row extent/ID";return false;
  }
  const auto row=original.subspan(std::size_t(actual_id)*(hidden/block_width)*bytes,(hidden/block_width)*bytes);
  for(std::size_t group=0;group<hidden/32;++group) {
    const auto w=detail::weight(row,type,group);
    for(std::size_t i=0;i<32;++i) {
      float expected;
      if(type==GGML_TYPE_Q8_0) {
        // Original block_q8_0 is LE F16 d followed by 32 signed i8 codes.
        // detail::weight reuses the independent F16 decoder. The one source
        // F32 d*q multiply is exact (<=18 significant bits), including signed
        // zero; do not add a synthetic zero min term that can change its sign.
        expected=q8::detail::rn_multiply(float(w.scale),float(w.codes[i]));
      } else if(type==GGML_TYPE_Q6_K) {
        const auto block=row.subspan(group/8*210,210);
        const auto column=(group%8)*32+i;
        const auto scaled=q8::detail::rn_multiply(float(w.scale),float(detail::signed_byte(block,192+column/16)));
        expected=q8::detail::rn_multiply(scaled,float(detail::q6_code(block,column)));
      } else if(type==GGML_TYPE_Q3_K) {
        const auto block=row.subspan(group/8*110,110);
        const auto column=(group%8)*32+i;
        // Match the two source dequantization multiplications, including the
        // sign of zero. F16 d * scale16 * q3 needs <=18 significant bits,
        // never underflows/overflows F32. Combining scale16*q3 in an integer
        // first would lose a negative scale's sign when q3 is zero.
        const auto scaled=q8::detail::rn_multiply(float(w.scale),float(detail::q3_scale(block,column/16)));
        expected=q8::detail::rn_multiply(scaled,float(detail::q3_code(block,column)));
      } else {
        expected=float(w.scale*w.codes[i]+w.minimum*w.offsets[i]);
      }
      if(!std::isfinite(expected)||std::bit_cast<std::uint32_t>(actual[group*32+i])!=std::bit_cast<std::uint32_t>(expected)) {
        ADD_FAILURE()<<"Q27 exact original embedding global="<<actual_id<<" column="<<group*32+i;return false;
      }
    }
  }
  return true;
}

inline bool expect_greedy(std::span<const float> logits,float actual_score,std::int32_t actual_id) {
  if(logits.empty()) {ADD_FAILURE()<<"Q27 empty original full-V logits";return false;}
  std::size_t best=0;
  for(std::size_t i=0;i<logits.size();++i) {
    if(!std::isfinite(logits[i])) {ADD_FAILURE()<<"Q27 nonfinite actual original logit";return false;}
    if(logits[i]>logits[best])best=i;
  }
  if(actual_id<0||std::size_t(actual_id)!=best||
      std::bit_cast<std::uint32_t>(actual_score)!=std::bit_cast<std::uint32_t>(logits[best])) {
    ADD_FAILURE()<<"Q27 stable full-V Top1 actual="<<actual_id<<" expected="<<best;return false;
  }
  return true;
}
} // namespace qwen27_original_nextn_reference
