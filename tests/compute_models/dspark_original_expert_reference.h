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
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "dspark_original_shared_reference.h"

namespace dspark_original_expert_reference {
namespace q8 = dspark_original_shared_reference;
using ActivationCodec = q8::ActivationCodec;
namespace detail {
inline constexpr std::array<int,16> kDoubledE2M1{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
inline double scale(std::byte raw) {
  const auto exponent=std::to_integer<unsigned>(raw);
  if(exponent==255)throw std::runtime_error("original MXFP4 E8M0 scale is NaN");
  return std::ldexp(1.0,int(exponent)-128);
}
inline int code(std::span<const std::byte> packed,std::size_t block,std::size_t column) {
  const auto pair=std::to_integer<unsigned>(packed[block+1+column%16]);
  return kDoubledE2M1[(pair>>(column<16?0:4))&15U];
}
inline void bfloat(std::span<const float> input) {
  for(const float term:input)
    if(!std::isfinite(term)||(std::bit_cast<std::uint32_t>(term)&0xffffU))
      throw std::runtime_error("original expert reference requires actual widened BF16");
}
struct Layout final {std::size_t bank_bytes,rows,input_slots;};
inline Layout layout(std::span<const std::byte> packed,std::size_t experts,
    std::size_t outputs,std::size_t width,std::span<const float> input,
    std::span<const std::int32_t> ids,std::size_t slots) {
  if(!experts||!outputs||!width||width%32||!slots||ids.empty()||ids.size()%slots||
      outputs>std::numeric_limits<std::size_t>::max()/(width/32)/17||
      ids.size()>std::numeric_limits<std::size_t>::max()/outputs)
    throw std::runtime_error("original expert reference extent");
  const auto rows=ids.size()/slots,bank_bytes=outputs*(width/32)*17;
  if(experts>std::numeric_limits<std::size_t>::max()/bank_bytes||
      packed.size()!=experts*bank_bytes||input.empty()||input.size()%width||
      (input.size()/width!=rows&&input.size()/width!=ids.size()))
    throw std::runtime_error("original expert reference bank or input-slot extent");
  for(const auto id:ids)if(id<0||std::uint64_t(id)>=experts)
    throw std::runtime_error("original expert global ID is out of range");
  return {bank_bytes,rows,input.size()/width/rows};
}
inline float widen(std::uint16_t bits) {return std::bit_cast<float>(std::uint32_t(bits)<<16U);}
inline double distance(double left,double right) {return left==right?0:q8::detail::upper(std::abs(left-right));}
} // namespace detail

// Independent raw decoded mathematical callback only, never the local native
// activation-quantized numerical gate. Each selected original expert is read
// directly from its authenticated mmap byte subspan, with no F32 weight shadow.
inline std::vector<float> decoded_selected_reference(std::span<const std::byte> packed,
    std::size_t experts,std::size_t outputs,std::size_t width,std::span<const float> input,
    std::span<const std::int32_t> global_ids,std::size_t slots=6) {
  const auto shape=detail::layout(packed,experts,outputs,width,input,global_ids,slots);
  std::vector<float> result(global_ids.size()*outputs);
  for(std::size_t selected=0;selected<global_ids.size();++selected) {
    const auto source_row=shape.input_slots==1?selected/slots:selected;
    const auto expert=packed.subspan(std::size_t(global_ids[selected])*shape.bank_bytes,shape.bank_bytes);
    for(std::size_t output_row=0;output_row<outputs;++output_row) {
      q8::detail::Sum sum;
      for(std::size_t block=0;block<width/32;++block) {
        const auto offset=(output_row*(width/32)+block)*17;
        const double weight_scale=detail::scale(expert[offset]);
        for(std::size_t column=0;column<32;++column)
          sum.add(weight_scale*detail::code(expert,offset,column)*double(input[source_row*width+block*32+column]));
      }
      result[selected*outputs+output_row]=float(sum.value());
    }
  }
  return result;
}

inline void expect_selected_projection(std::span<const float> actual,
    std::span<const std::byte> original_packed,std::size_t experts,std::size_t outputs,
    std::size_t width,std::span<const float> actual_bfloat_input,
    std::span<const std::int32_t> global_ids,ActivationCodec codec,std::string_view role,
    std::size_t slots=6) {
  const auto shape=detail::layout(original_packed,experts,outputs,width,actual_bfloat_input,global_ids,slots);
  ASSERT_EQ(actual.size(),global_ids.size()*outputs)<<role;
  const auto blocks=width/32;
  std::vector<q8::detail::ActivationBlock> activation(actual_bfloat_input.size()/32);
  std::size_t ambiguous_codes=0,ambiguous_scales=0;
  for(std::size_t block=0;block<activation.size();++block) {
    auto &source=activation[block];
    source=q8::detail::quantize(std::span<const float,32>(actual_bfloat_input.data()+block*32,32),codec);
    ambiguous_scales+=source.scale_low!=source.scale_high;
    for(std::size_t column=0;column<32;++column)ambiguous_codes+=source.low[column]!=source.high[column];
  }
  const double g32=q8::detail::gamma(width/4+65,0x1p-24),g64=q8::detail::gamma(blocks,0x1p-53);
  double maximum_error=0,maximum_ratio=0,maximum_ambiguity=0;
  for(std::size_t selected=0;selected<global_ids.size();++selected) {
    const auto source_row=shape.input_slots==1?selected/slots:selected;
    const auto expert=original_packed.subspan(std::size_t(global_ids[selected])*shape.bank_bytes,shape.bank_bytes);
    for(std::size_t output_row=0;output_row<outputs;++output_row) {
      q8::detail::Sum sum;double ambiguity=0,absolute=0;
      for(std::size_t block=0;block<blocks;++block) {
        const auto offset=(output_row*blocks+block)*17;
        const double weight_scale=detail::scale(expert[offset]);
        // The E8 power-of-two times the stored activation F16 scale is exact
        // F32 under these explicit range conditions. Tiny E8 encodings need
        // separate FTZ analysis; they are not silently treated as ordinary Q8.
        if(weight_scale<double(std::numeric_limits<float>::min()))
          throw std::runtime_error("original MXFP4 scale needs separate FTZ analysis");
        const auto &source=activation[source_row*blocks+block];
        const double scale_maximum=std::max(double(source.scale_low),double(source.scale_high));
        const double smallest=source.scale_low==0?double(source.scale_high):double(source.scale_low);
        for(const double active_scale:{smallest,scale_maximum}) {
          const double combined=weight_scale*active_scale;
          if(!std::isfinite(combined)||double(float(combined))!=combined||
              (combined!=0&&combined<0x1p-103))
            throw std::runtime_error("original MXFP4 combined scale needs separate range analysis");
        }
        int dot=0,uncertain=0,maximum_absolute=0;
        for(std::size_t column=0;column<32;++column) {
          const int coefficient=detail::code(expert,offset,column);
          dot+=coefficient*source.codes[column];
          uncertain+=std::abs(coefficient)*std::max(std::abs(source.low[column]-source.codes[column]),
              std::abs(source.high[column]-source.codes[column]));
          maximum_absolute+=std::abs(coefficient)*std::max(std::abs(source.low[column]),std::abs(source.high[column]));
        }
        sum.add(weight_scale*double(source.scale)*double(dot));
        const double scale_error=std::max(std::abs(double(source.scale_low)-source.scale),
            std::abs(double(source.scale_high)-source.scale));
        ambiguity=q8::detail::add_upper(ambiguity,q8::detail::multiply_upper(weight_scale,
            q8::detail::add_upper(q8::detail::multiply_upper(scale_error,std::abs(double(dot))),
                q8::detail::multiply_upper(scale_maximum,double(uncertain)))));
        absolute=q8::detail::add_upper(absolute,q8::detail::multiply_upper(weight_scale,
            q8::detail::multiply_upper(scale_maximum,double(maximum_absolute))));
      }
      if(!std::isfinite(absolute)||absolute>double(std::numeric_limits<float>::max())/4)
        throw std::runtime_error("original MXFP4 reduction needs separate overflow analysis");
      const double center=sum.value();
      const double oracle=q8::detail::add_upper(detail::distance(center,sum.plain),q8::detail::multiply_upper(g64,absolute));
      const double arithmetic=q8::detail::multiply_upper(g32,absolute);
      const double bound=q8::detail::add_upper(ambiguity,q8::detail::add_upper(arithmetic,oracle));
      const auto index=selected*outputs+output_row;
      ASSERT_TRUE(std::isfinite(actual[index]))<<role<<" slot="<<selected<<" output="<<output_row;
      const double error=detail::distance(actual[index],center);
      ASSERT_LE(error,bound)<<role<<" global="<<global_ids[selected]<<" slot="<<selected
          <<" output="<<output_row<<" F64="<<center<<" ambiguity="<<ambiguity
          <<" F32_bound="<<arithmetic<<" sumabs="<<absolute;
      maximum_error=std::max(maximum_error,error);maximum_ambiguity=std::max(maximum_ambiguity,ambiguity);
      if(bound)maximum_ratio=std::max(maximum_ratio,error/bound);
    }
  }
  std::fprintf(stderr,"original-MXFP4 role=%.*s outputs=%zu max_error=%.17g max_ratio=%.17g "
      "max_AQ_ambiguity=%.17g ambiguous_codes=%zu ambiguous_scales=%zu\n",int(role.size()),role.data(),
      actual.size(),maximum_error,maximum_ratio,maximum_ambiguity,ambiguous_codes,ambiguous_scales);
}

inline void expect_router(std::span<const float> actual,std::span<const float> actual_bfloat_input,
    std::span<const std::uint16_t> original_bfloat_weight,std::string_view role,std::size_t experts=256) {
  if(!experts||original_bfloat_weight.empty()||original_bfloat_weight.size()%experts)
    throw std::runtime_error("original router weight extent");
  const auto width=original_bfloat_weight.size()/experts;
  if(actual_bfloat_input.empty()||actual_bfloat_input.size()%width)
    throw std::runtime_error("original router input extent");
  detail::bfloat(actual_bfloat_input);
  ASSERT_EQ(actual.size(),actual_bfloat_input.size()/width*experts)<<role;
  const double g32=q8::detail::gamma(width,0x1p-24),g64=q8::detail::gamma(width,0x1p-53);
  double maximum_error=0,maximum_ratio=0;
  for(std::size_t row=0;row<actual_bfloat_input.size()/width;++row)
    for(std::size_t expert=0;expert<experts;++expert) {
      q8::detail::Sum sum;double absolute=0;
      for(std::size_t column=0;column<width;++column) {
        const double product=double(actual_bfloat_input[row*width+column])*
            double(detail::widen(original_bfloat_weight[expert*width+column]));
        if(!std::isfinite(product)||double(float(product))!=product||
            (product!=0&&std::abs(product)<0x1p-103))
          throw std::runtime_error("original router BF16 product needs separate range analysis");
        sum.add(product);absolute=q8::detail::add_upper(absolute,std::abs(product));
      }
      if(!std::isfinite(absolute)||absolute>double(std::numeric_limits<float>::max())/4)
        throw std::runtime_error("original router reduction needs separate overflow analysis");
      const double center=sum.value();
      const double bound=q8::detail::add_upper(q8::detail::multiply_upper(g32,absolute),
          q8::detail::add_upper(detail::distance(center,sum.plain),q8::detail::multiply_upper(g64,absolute)));
      const auto index=row*experts+expert;const double error=detail::distance(actual[index],center);
      ASSERT_TRUE(std::isfinite(actual[index]))<<role<<" output="<<index;
      ASSERT_LE(error,bound)<<role<<" output="<<index<<" F64="<<center<<" sumabs="<<absolute;
      maximum_error=std::max(maximum_error,error);if(bound)maximum_ratio=std::max(maximum_ratio,error/bound);
    }
  std::fprintf(stderr,"original-router role=%.*s outputs=%zu max_error=%.17g max_ratio=%.17g\n",
      int(role.size()),role.data(),actual.size(),maximum_error,maximum_ratio);
}

inline void expect_mapping(std::span<const std::int32_t> global_ids,std::span<const std::int32_t> local_ids,
    std::span<const std::int32_t> global_to_local,std::span<const std::int32_t> resident_global_ids) {
  ASSERT_EQ(global_ids.size(),local_ids.size());ASSERT_EQ(global_to_local.size(),256U);
  std::array<bool,256> selected{};
  for(std::size_t index=0;index<global_ids.size();++index) {
    ASSERT_GE(global_ids[index],0);ASSERT_LT(global_ids[index],256);
    const auto global=std::size_t(global_ids[index]);selected[global]=true;
    ASSERT_EQ(local_ids[index],global_to_local[global]);ASSERT_GE(local_ids[index],0);
    ASSERT_LT(std::size_t(local_ids[index]),resident_global_ids.size());
    EXPECT_EQ(resident_global_ids[std::size_t(local_ids[index])],global_ids[index]);
  }
  for(std::size_t global=0;global<selected.size();++global)
    if(!selected[global])EXPECT_EQ(global_to_local[global],-1)<<global;
}

struct RoutingValues final {
  std::span<const float> logits,softplus,scores,selection,selected_scores,sum,normalized,weights;
  std::span<const std::int32_t> ids;
};
// Nonlinear gates use the existing semantic local8e-5/4e-5 policy. Discrete
// TopK is instead checked exactly against its ACTUAL F32 selection predecessor;
// no ideal router dot is propagated through softplus/sqrt into an ID verdict.
inline void expect_routing(const RoutingValues &actual,std::span<const float> original_bias,
    float scaling,std::string_view role,std::size_t experts=256,std::size_t slots=6) {
  if(!experts||!slots||slots>experts||actual.logits.empty()||actual.logits.size()%experts||
      original_bias.size()!=experts||!std::isfinite(scaling))
    throw std::runtime_error("original router local boundary extent");
  const auto rows=actual.logits.size()/experts;
  ASSERT_EQ(actual.softplus.size(),actual.logits.size())<<role;
  ASSERT_EQ(actual.scores.size(),actual.logits.size())<<role;
  ASSERT_EQ(actual.selection.size(),actual.logits.size())<<role;
  ASSERT_EQ(actual.ids.size(),rows*slots)<<role;
  ASSERT_EQ(actual.selected_scores.size(),actual.ids.size())<<role;
  ASSERT_EQ(actual.sum.size(),rows)<<role;
  ASSERT_EQ(actual.normalized.size(),actual.ids.size())<<role;
  ASSERT_EQ(actual.weights.size(),actual.ids.size())<<role;
  const auto exact=[](float observed,float expected) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(observed),std::bit_cast<std::uint32_t>(expected));
  };
  const auto local=[](float observed,float expected) {
    ASSERT_TRUE(std::isfinite(observed));ASSERT_TRUE(std::isfinite(expected));
    EXPECT_NEAR(observed,expected,8.0e-5+4.0e-5*std::abs(double(expected)));
  };
  for(std::size_t index=0;index<actual.logits.size();++index) {
    ASSERT_TRUE(std::isfinite(actual.logits[index]))<<role;
    ASSERT_TRUE(std::isfinite(original_bias[index%experts]))<<role;
    const double wide=actual.logits[index];
    const float logarithm=float(std::log1p(std::exp(-std::abs(wide))));
    const float softplus=float(std::max(wide,0.0)+double(logarithm));
    local(actual.softplus[index],softplus);
    ASSERT_GE(actual.softplus[index],0.0F)<<role;
    local(actual.scores[index],float(std::sqrt(double(actual.softplus[index]))));
    ASSERT_TRUE(std::isfinite(actual.selection[index]))<<role;
    exact(actual.selection[index],float(double(actual.scores[index])+double(original_bias[index%experts])));
  }
  const double g32=q8::detail::gamma(slots,0x1p-24),g64=q8::detail::gamma(slots,0x1p-53);
  for(std::size_t row=0;row<rows;++row) {
    std::vector<std::int32_t> order(experts);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto left,auto right) {
      return actual.selection[row*experts+std::size_t(left)]>actual.selection[row*experts+std::size_t(right)];
    });
    q8::detail::Sum sum;double absolute=0;
    for(std::size_t slot=0;slot<slots;++slot) {
      const auto index=row*slots+slot;
      ASSERT_EQ(actual.ids[index],order[slot])<<role<<" row="<<row<<" slot="<<slot;
      const float score=actual.scores[row*experts+std::size_t(actual.ids[index])];
      exact(actual.selected_scores[index],score);
      if(!std::isfinite(score)||(score!=0&&std::abs(double(score))<0x1p-103))
        throw std::runtime_error("original router denominator needs separate range analysis");
      sum.add(double(score));absolute=q8::detail::add_upper(absolute,std::abs(double(score)));
    }
    if(!std::isfinite(absolute)||absolute>double(std::numeric_limits<float>::max())/4)
      throw std::runtime_error("original router denominator needs separate overflow analysis");
    const double denominator_bound=q8::detail::add_upper(q8::detail::multiply_upper(g32,absolute),
        q8::detail::add_upper(detail::distance(sum.value(),sum.plain),q8::detail::multiply_upper(g64,absolute)));
    ASSERT_TRUE(std::isfinite(actual.sum[row]))<<role;ASSERT_GT(actual.sum[row],0.0F)<<role;
    ASSERT_LE(detail::distance(actual.sum[row],sum.value()),denominator_bound)<<role;
    for(std::size_t slot=0;slot<slots;++slot) {
      const auto index=row*slots+slot;
      local(actual.normalized[index],float(double(actual.selected_scores[index])/double(actual.sum[row])));
      exact(actual.weights[index],float(double(actual.normalized[index])*double(scaling)));
    }
  }
}
} // namespace dspark_original_expert_reference
