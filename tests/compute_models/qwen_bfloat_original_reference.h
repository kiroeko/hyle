#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Original BF16 local references. Only actual checked predecessors feed the
// next check. Original matrices remain LE byte spans; no expanded bank or
// native reference executor is constructed. Nonlinear F32 local gates are
// the existing 8e-5/4e-5 gates, followed by a separate BF16 rounding interval.
namespace qwen_bfloat_original_reference {
inline float widen(std::uint16_t value) {return std::bit_cast<float>(std::uint32_t(value)<<16U);}
inline std::uint16_t round(float value) {
  if(!std::isfinite(value))throw std::runtime_error("BF16 reference needs finite arithmetic");
  const auto bits=std::bit_cast<std::uint32_t>(value);
  return std::uint16_t((bits+0x7fffU+((bits>>16U)&1U))>>16U);
}
inline std::uint16_t read(std::span<const std::byte> raw,std::size_t index) {
  if(index>=raw.size()/2)throw std::runtime_error("BF16 original byte address exceeds its role");
  return std::uint16_t(std::to_integer<unsigned>(raw[2*index])|(std::to_integer<unsigned>(raw[2*index+1])<<8U));
}
inline std::vector<float> widen(std::span<const std::uint16_t> input) {
  std::vector<float> result(input.size());
  for(std::size_t i=0;i<input.size();++i) {
    result[i]=widen(input[i]);if(!std::isfinite(result[i]))throw std::runtime_error("nonfinite actual BF16 operand");
  }
  return result;
}
inline std::vector<std::uint16_t> store(std::span<const float> input) {
  std::vector<std::uint16_t> result(input.size());
  for(std::size_t i=0;i<input.size();++i)result[i]=round(input[i]);return result;
}
namespace detail {
inline double upper(double value){return std::nextafter(value,std::numeric_limits<double>::infinity());}
inline double add(double a,double b){return upper(a+b);}
inline double multiply(double a,double b){return upper(a*b);}
inline double distance(double a,double b){return a==b?0.:upper(std::abs(a-b));}
inline double gamma(std::size_t count,double unit) {
  const double product=double(count)*unit;
  if(product>=1)throw std::runtime_error("BF16 dot reduction bound is undefined");
  return upper(product/(1-product));
}
struct Sum final {
  double plain=0,correction=0;
  void add(double value) {
    const double next=plain+value;
    correction+=std::abs(plain)>=std::abs(value)?(plain-next)+value:(value-next)+plain;plain=next;
  }
  double value()const{return plain+correction;}
};
inline float normal(std::uint16_t bits) {
  const auto value=widen(bits);
  if(!std::isfinite(value)||((bits&0x7f80U)==0&&(bits&0x7fU)!=0))
    throw std::runtime_error("BF16 dot needs separate nonfinite/subnormal source analysis");
  return value;
}
struct Dot final {double center,bound;};
inline Dot dot(std::span<const std::uint16_t> input,std::span<const std::byte> weight) {
  if(input.empty()||input.size()>std::numeric_limits<std::size_t>::max()/2||weight.size()!=input.size()*2)
    throw std::runtime_error("BF16 original dot extent differs");
  constexpr std::size_t block=128;Sum blocks;double absolute=0;
  std::size_t tiny_products=0,subnormal_products=0;
  double minimum_product=std::numeric_limits<double>::infinity();
  for(std::size_t first=0;first<input.size();first+=block) {
    double partial=0;
    for(std::size_t column=first;column<std::min(input.size(),first+block);++column) {
      const double product=double(normal(input[column]))*normal(read(weight,column));
      // Normal BF16 operands have <=16-bit exact F64 products, including
      // products below the F32 normal range. Source subnormals stay rejected.
      if(!std::isfinite(product))throw std::runtime_error("BF16 dot needs separate nonfinite product analysis");
      if(product!=0) {
        const double magnitude=std::abs(product);
        minimum_product=std::min(minimum_product,magnitude);
        tiny_products+=std::size_t(magnitude<0x1p-103);
        subnormal_products+=std::size_t(magnitude<0x1p-126);
      }
      partial+=product;absolute+=std::abs(product);
    }
    blocks.add(partial);
  }
  const auto width=input.size();const auto g64=gamma(width,0x1p-53);
  const double upper_sum=absolute==0?0:upper(absolute/(1-g64));
  if(!std::isfinite(upper_sum)||upper_sum>double(std::numeric_limits<float>::max())/4)
    throw std::runtime_error("BF16 dot needs separate overflow analysis");
  const auto count=width/block+std::size_t(width%block!=0);
  const auto inner=gamma(std::min(width,block),0x1p-53),outer=gamma(count,0x1p-53);
  const auto blocked=add(add(inner,outer),multiply(inner,outer));
  const auto oracle=add(distance(blocks.value(),blocks.plain),multiply(blocked,upper_sum));
  double bound=absolute==0?0:add(multiply(gamma(width,0x1p-24),upper_sum),oracle);
  if(tiny_products!=0) {
    // For F32 round-to-nearest with optional result FTZ,
    // |fl(z)-z| <= u*|z|+eta, u=2^-24, eta=minNormal=2^-126.
    // At most K products and K-1 sums (or fewer FMAs) each contribute eta;
    // a term crosses at most K roundings, so (1+u)^K <= 1/(1-K*u).
    // Hence 2*K*eta/(1-K*u) covers product and cancellation-result FTZ.
    // If all products are >=2^-103, their <=16-bit significands give a
    // >=2^-118 quantum: every F32 intermediate is normal or zero and the
    // original no-FTZ bound above remains exactly unchanged.
    const double ftz=upper(multiply(multiply(2.,double(width)),0x1p-126)/(1-double(width)*0x1p-24));
    bound=add(bound,ftz);
    // Bound the diagnostic volume; these are actual operand statistics,
    // never observed-output differences used to select a numerical bound.
    static std::atomic<unsigned> diagnostics{0};
    if(diagnostics.load(std::memory_order_relaxed)<4&&diagnostics.fetch_add(1,std::memory_order_relaxed)<4)
      std::fprintf(stderr,"BF16 dot tiny-domain K=%zu below_2^-103=%zu below_F32_minNormal=%zu min_abs_product=%.17g FTZ_bound=%.17g total_bound=%.17g\n",
          width,tiny_products,subnormal_products,minimum_product,ftz,bound);
  }
  return {blocks.value(),bound};
}
inline std::vector<std::size_t> candidates(std::span<const float> row) {
  std::vector<std::size_t> result;
  for(std::size_t index=0;index<row.size();++index) {
    if(!std::isfinite(row[index]))throw std::runtime_error("BF16 projection has nonfinite actual output");
    auto where=result.begin();while(where!=result.end()&&(row[*where]>row[index]||(row[*where]==row[index]&&*where<index)))++where;
    if(where!=result.end()||result.size()<4)result.insert(where,index);if(result.size()>4)result.pop_back();
  }
  return result;
}
} // namespace detail
inline bool expect_store(std::span<const std::uint16_t> actual,std::span<const float> raw,std::string_view role) {
  if(actual.size()!=raw.size()){ADD_FAILURE()<<role<<" BF16 store extent";return false;}
  for(std::size_t i=0;i<actual.size();++i)if(actual[i]!=round(raw[i])) {
    ADD_FAILURE()<<role<<" actual BF16 store["<<i<<']';return false;
  }
  return true;
}
inline bool expect_local(std::uint16_t actual,double center,std::string_view role) {
  const double error=8.e-5+4.e-5*std::abs(center);
  const auto lower=round(std::nextafter(float(center-error),-std::numeric_limits<float>::infinity()));
  const auto upper=round(std::nextafter(float(center+error),std::numeric_limits<float>::infinity()));
  const float value=widen(actual);
  if(!std::isfinite(value)||value<widen(lower)||value>widen(upper)) {
    ADD_FAILURE()<<role<<" BF16 value="<<value<<" outside original local F32 gate then BF16 interval, center="<<center;return false;
  }
  return true;
}
inline bool expect_projection(std::span<const float> actual,std::span<const std::uint16_t> input,
    std::span<const std::byte> weight,std::size_t outputs,std::size_t width,std::string_view role,bool full,
    std::span<const std::int32_t> selected={}) {
  if(!width||!outputs||outputs>std::numeric_limits<std::size_t>::max()/width/2||
      weight.size()!=outputs*width*2||input.empty()||input.size()%width||
      input.size()/width>std::numeric_limits<std::size_t>::max()/outputs||actual.size()!=input.size()/width*outputs||
      (!selected.empty()&&selected.size()!=input.size()/width)) {
    ADD_FAILURE()<<role<<" BF16 projection extent";return false;
  }
  for(std::size_t row=0;row<input.size()/width;++row) {
    const auto values=actual.subspan(row*outputs,outputs);auto chosen=detail::candidates(values);
    if(!selected.empty()) {
      if(selected[row]<0||std::size_t(selected[row])>=outputs){ADD_FAILURE()<<role<<" selected output outside domain";return false;}
      chosen.push_back(std::size_t(selected[row]));
    }
    const auto check=[&](std::size_t index) {
      const auto expected=detail::dot(input.subspan(row*width,width),weight.subspan(index*width*2,width*2));
      if(detail::distance(values[index],expected.center)>expected.bound) {
        ADD_FAILURE()<<role<<" actual["<<row<<','<<index<<"]="<<values[index]<<" F64="<<expected.center<<" bound="<<expected.bound;return false;
      }
      return true;
    };
    if(full){for(std::size_t index=0;index<outputs;++index)if(!check(index))return false;}
    else {for(auto index:chosen)if(!check(index))return false;}
  }
  return true;
}
inline bool expect_gamma(std::span<const float> actual,std::span<const std::uint16_t> raw,bool delta,std::string_view role) {
  if(actual.size()!=raw.size()){ADD_FAILURE()<<role<<" gamma extent";return false;}
  for(std::size_t i=0;i<raw.size();++i) {
    const float wanted=delta?float(widen(raw[i])+1.F):widen(raw[i]);
    if(!std::isfinite(wanted)||std::bit_cast<std::uint32_t>(actual[i])!=std::bit_cast<std::uint32_t>(wanted)) {
      ADD_FAILURE()<<role<<" actual source gamma["<<i<<']';return false;
    }
  }
  return true;
}
inline bool expect_norm(std::span<const float> raw,std::span<const std::uint16_t> stored,
    std::span<const std::uint16_t> input,std::span<const float> gamma,std::string_view role,float epsilon=1.e-6F) {
  if(gamma.empty()||input.size()%gamma.size()||input.size()!=raw.size()||stored.size()!=raw.size()||!std::isfinite(epsilon)||epsilon<=0) {
    ADD_FAILURE()<<role<<" actual norm extent";return false;
  }
  for(std::size_t row=0;row<input.size()/gamma.size();++row) {
    detail::Sum squares;
    for(std::size_t column=0;column<gamma.size();++column) {
      const auto value=detail::normal(input[row*gamma.size()+column]);const double square=double(value)*value;
      if(square!=0&&square<0x1p-103)throw std::runtime_error("BF16 norm needs separate product/FTZ analysis");
      squares.add(square);
    }
    if(!std::isfinite(squares.value())||squares.value()>double(std::numeric_limits<float>::max())/4)
      throw std::runtime_error("BF16 norm needs separate overflow analysis");
    const double inverse=1/std::sqrt(squares.value()/double(gamma.size())+epsilon);
    for(std::size_t column=0;column<gamma.size();++column) {
      const auto index=row*gamma.size()+column;const double center=double(widen(input[index]))*inverse*gamma[column];
      if(!std::isfinite(center)||!std::isfinite(raw[index])||std::abs(raw[index]-center)>3.e-5+2.e-5*std::abs(center)) {
        ADD_FAILURE()<<role<<" actual F32 norm["<<index<<']';return false;
      }
    }
  }
  return expect_store(stored,raw,role);
}
inline bool expect_binary(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> left,
    std::span<const std::uint16_t> right,bool multiply,std::string_view role) {
  if(actual.size()!=left.size()||left.size()!=right.size()){ADD_FAILURE()<<role<<" binary extent";return false;}
  for(std::size_t i=0;i<actual.size();++i) {
    const float value=multiply?float(double(widen(left[i]))*widen(right[i])):float(double(widen(left[i]))+widen(right[i]));
    if(actual[i]!=round(value)){ADD_FAILURE()<<role<<" BF16 binary["<<i<<']';return false;}
  }
  return true;
}
inline bool expect_unary(std::span<const std::uint16_t> actual,std::span<const std::uint16_t> input,bool silu,std::string_view role) {
  if(actual.size()!=input.size()){ADD_FAILURE()<<role<<" unary extent";return false;}
  for(std::size_t i=0;i<actual.size();++i) {
    const double value=widen(input[i]),sigmoid=1/(1+std::exp(-value));
    if(!expect_local(actual[i],silu?value*sigmoid:sigmoid,role))return false;
  }
  return true;
}
inline bool expect_embedding(std::span<const std::uint16_t> actual,std::span<const std::byte> bank,
    std::span<const std::int32_t> ids,std::size_t vocabulary,std::size_t hidden) {
  if(!hidden||!vocabulary||vocabulary>std::numeric_limits<std::size_t>::max()/hidden/2||
      bank.size()!=vocabulary*hidden*2||ids.size()>std::numeric_limits<std::size_t>::max()/hidden||actual.size()!=ids.size()*hidden) {
    ADD_FAILURE()<<"BF16 embedding extent";return false;
  }
  for(std::size_t row=0;row<ids.size();++row) {
    if(ids[row]<0||std::size_t(ids[row])>=vocabulary){ADD_FAILURE()<<"BF16 pending outside actual vocabulary";return false;}
    for(std::size_t column=0;column<hidden;++column)if(actual[row*hidden+column]!=read(bank,std::size_t(ids[row])*hidden+column)) {
      ADD_FAILURE()<<"original BF16 embedding bit mismatch";return false;
    }
  }
  return true;
}
inline bool expect_greedy(std::span<const float> logits,std::span<const std::int32_t> ids,
    std::span<const float> scores,std::size_t vocabulary) {
  if(!vocabulary||ids.empty()||ids.size()>std::numeric_limits<std::size_t>::max()/vocabulary||
      logits.size()!=ids.size()*vocabulary||scores.size()!=ids.size()){ADD_FAILURE()<<"BF16 stable Top1 extent";return false;}
  for(std::size_t row=0;row<ids.size();++row) {
    const auto values=logits.subspan(row*vocabulary,vocabulary);const auto selected=detail::candidates(values).front();
    if(ids[row]!=std::int32_t(selected)||scores[row]!=values[selected]){ADD_FAILURE()<<"actual BF16 head stable Top1";return false;}
  }
  return true;
}
} // namespace qwen_bfloat_original_reference
