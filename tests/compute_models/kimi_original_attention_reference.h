#pragma once

#include "kimi_original_attention_roles.h"
#include "glm_original_indexer_reference.h"

// Private source-only oracle. Reads original file rows, never the uploaded
// packed convolution or derived K/V operands. No full F32 weight bank exists.
// DenseLinearF32 uses BF16-RNE A; CUDA grouped BF16-W/F32-A keeps actual F32 A.
// The chain centers below are not a substitute for checking actual typed
// producer boundaries: local projection certificates accept those producers.
namespace kimi_original_attention_reference {
namespace raw=kimi_original_attention_roles;
namespace scalar=glm_original_indexer_reference::detail;
using raw::Data;
using raw::Role;
inline constexpr std::size_t hidden=7168,heads=96,depth=128,width=heads*depth;

inline float coefficient(std::span<const std::byte> bytes,raw::c::DataType type,std::size_t index) {
  const auto size=type==raw::c::DataType::kBFloat16?2u:4u;
  if(index>=bytes.size()/size)throw std::runtime_error("original Kimi oracle coefficient extent");
  std::uint32_t bits=0;
  for(std::size_t i=0;i<size;++i)bits|=std::to_integer<std::uint32_t>(bytes[index*size+i])<<(8*i);
  if(size==2)bits<<=16;
  const auto value=std::bit_cast<float>(bits);
  if(!std::isfinite(value))throw std::runtime_error("original Kimi oracle used nonfinite coefficient");
  return value;
}
inline float coefficient(const Data &data,Role role,std::size_t index) {
  return coefficient(data.raw(role),raw::specs.at(std::size_t(role)).type,index);
}
inline float stored_bfloat(float value) {
  if(!std::isfinite(value))throw std::runtime_error("original Kimi oracle BF16 producer is nonfinite");
  const auto stored=scalar::widen(scalar::bfloat(value));
  if(!std::isfinite(stored))throw std::runtime_error("original Kimi oracle BF16 store overflow");
  return stored;
}
inline double sigmoid(double value) {
  if(value>=0)return 1/(1+std::exp(-value));
  const auto exponential=std::exp(value);return exponential/(1+exponential);
}

struct Dot final {double center,bound,sumabs;};
// Any F32 accumulation tree/FMA is bounded conservatively by gamma_(2K).
// Add the measured compensated/plain FP64 distance and its ordinary gamma_K
// bound. The absolute term also covers worst-case FTZ at every native step;
// it is not an empirical tolerance or a claim that CUDA uses gradual underflow.
template<class Weight>
Dot dot(std::span<const float> actual_input,Weight &&weight,bool bfloat_input) {
  if(actual_input.empty())throw std::runtime_error("original Kimi oracle dot is empty");
  scalar::Sum sum;double absolute=0;
  for(std::size_t i=0;i<actual_input.size();++i) {
    const auto a=bfloat_input?stored_bfloat(actual_input[i]):actual_input[i];
    const auto b=weight(i);
    if(!std::isfinite(a)||!std::isfinite(b)||std::fpclassify(a)==FP_SUBNORMAL||
        std::fpclassify(b)==FP_SUBNORMAL)
      throw std::runtime_error("original Kimi oracle needs separate subnormal-source analysis");
    const auto product=double(a)*double(b);
    sum.add(product);absolute+=std::abs(product);
  }
  const auto count=actual_input.size();
  const auto g64=scalar::gamma(count,0x1p-53),g32=scalar::gamma(2*count,0x1p-24);
  const auto upper_sum=absolute==0?0:scalar::upper(absolute/(1-g64));
  if(!std::isfinite(absolute)||upper_sum>double(std::numeric_limits<float>::max())/4)
    throw std::runtime_error("original Kimi oracle dot needs separate overflow analysis");
  const auto oracle=scalar::upper_add(scalar::distance(sum.value(),sum.plain),
      scalar::upper_multiply(g64,upper_sum));
  const auto ftz=scalar::upper(double(2*count)*double(std::numeric_limits<float>::min())*(1+g32));
  return {sum.value(),scalar::upper_add(scalar::upper_add(
      scalar::upper_multiply(g32,upper_sum),oracle),ftz),absolute};
}
inline Dot original_row(const Data &data,Role role,std::size_t row,
                        std::span<const float> actual_input) {
  const auto &spec=raw::specs.at(std::size_t(role));
  if(spec.rank!=2||spec.type!=raw::c::DataType::kBFloat16||row>=std::size_t(spec.shape[0])||
      actual_input.size()!=std::size_t(spec.shape[1]))
    throw std::runtime_error("original Kimi oracle raw projection shape");
  const auto bank=data.raw(role);
  return dot(actual_input,[&](std::size_t column){return coefficient(bank,spec.type,row*actual_input.size()+column);},true);
}
inline std::vector<float> project(const Data &data,Role role,std::span<const float> input) {
  const auto &spec=raw::specs.at(std::size_t(role));
  std::vector<float> result(std::size_t(spec.shape[0]));
  for(std::size_t row=0;row<result.size();++row)result[row]=float(original_row(data,role,row,input).center);
  return result;
}
inline std::vector<float> normalize(const Data &data,Role role,
    std::span<const float> input,std::size_t row_width,double epsilon=1.e-5F) {
  if(!row_width||input.empty()||input.size()%row_width||
      data.raw(role).size()/std::size_t(raw::specs[std::size_t(role)].type==raw::c::DataType::kBFloat16?2:4)!=row_width)
    throw std::runtime_error("original Kimi oracle norm shape");
  std::vector<float> result(input.size());
  for(std::size_t first=0;first<input.size();first+=row_width) {
    scalar::Sum square;
    for(std::size_t i=0;i<row_width;++i)square.add(double(input[first+i])*input[first+i]);
    const auto scale=1/std::sqrt(square.value()/double(row_width)+epsilon);
    for(std::size_t i=0;i<row_width;++i)
      result[first+i]=float(double(input[first+i])*scale*coefficient(data,role,i));
  }
  return result;
}

inline std::vector<std::vector<float>> kda(const Data &data,std::span<const float> input,
    std::span<const float> history,std::span<const float> previous_state) {
  constexpr auto channels=3*width;
  if(data.component()!=raw::Component::kKda||input.size()!=hidden||history.size()!=3*channels||
      previous_state.size()!=heads*depth*depth)
    throw std::runtime_error("original Kimi KDA oracle input extent");
  auto q=project(data,Role::kKdaQ,input),k=project(data,Role::kKdaK,input),v=project(data,Role::kKdaV,input);
  auto samples=std::vector<float>(history.begin(),history.end());
  samples.insert(samples.end(),q.begin(),q.end());samples.insert(samples.end(),k.begin(),k.end());
  samples.insert(samples.end(),v.begin(),v.end());
  std::vector<float> next_history(samples.begin()+channels,samples.end()),convolution(channels),activation(channels);
  constexpr std::array conv{Role::kKdaQConv,Role::kKdaKConv,Role::kKdaVConv};
  for(std::size_t channel=0;channel<channels;++channel) {
    scalar::Sum sum;
    for(std::size_t tap=0;tap<4;++tap)
      sum.add(double(samples[tap*channels+channel])*coefficient(data,conv[channel/width],(channel%width)*4+tap));
    convolution[channel]=float(sum.value());
    activation[channel]=float(double(convolution[channel])*sigmoid(convolution[channel]));
  }
  const auto fa=project(data,Role::kKdaFa,input),fb=project(data,Role::kKdaFb,fa);
  const auto beta=project(data,Role::kKdaBeta,input),gate=project(data,Role::kKdaGate,input);
  std::array<std::vector<float>,2> qk{std::vector<float>(width),std::vector<float>(width)};
  for(std::size_t part=0;part<2;++part)for(std::size_t head=0;head<heads;++head) {
    scalar::Sum squared;
    for(std::size_t d=0;d<depth;++d) {
      const auto value=activation[part*width+head*depth+d];squared.add(double(value)*value);
    }
    const auto scale=1/std::sqrt(squared.value()+double(1.e-5F));
    for(std::size_t d=0;d<depth;++d)qk[part][head*depth+d]=float(activation[part*width+head*depth+d]*scale);
  }
  auto state=std::vector<float>(previous_state.begin(),previous_state.end());std::vector<float> recurrent(width);
  for(std::size_t head=0;head<heads;++head) {
    std::array<double,depth> decay;
    const auto alpha=std::exp(coefficient(data,Role::kKdaA,head));
    for(std::size_t d=0;d<depth;++d)
      decay[d]=std::exp(-5*sigmoid(double(alpha)*(double(fb[head*depth+d])+coefficient(data,Role::kKdaDt,head*depth+d))));
    for(std::size_t vi=0;vi<depth;++vi) {
      const auto row=(head*depth+vi)*depth;std::array<double,depth> decayed;scalar::Sum prediction;
      for(std::size_t d=0;d<depth;++d) {
        decayed[d]=double(state[row+d])*decay[d];prediction.add(decayed[d]*qk[1][head*depth+d]);
      }
      const auto delta=sigmoid(beta[head])*(activation[2*width+head*depth+vi]-prediction.value());
      scalar::Sum score;
      for(std::size_t d=0;d<depth;++d) {
        const auto next=decayed[d]+delta*qk[1][head*depth+d];state[row+d]=float(next);
        score.add(next*qk[0][head*depth+d]);
      }
      recurrent[head*depth+vi]=float(score.value()/std::sqrt(double(depth)));
    }
  }
  auto gated=normalize(data,Role::kKdaNorm,recurrent,depth);
  for(std::size_t i=0;i<gated.size();++i)gated[i]*=float(sigmoid(gate[i]));
  auto output=project(data,Role::kKdaOutput,gated);
  return {std::move(next_history),std::move(convolution),std::move(state),std::move(recurrent),std::move(gated),std::move(output)};
}

inline Dot original_absorbed_row(const Data &data,std::size_t head,std::size_t row,
    std::span<const float> actual_input,bool key) {
  if(head>=heads||row>=(key?512u:depth)||actual_input.size()!=(key?depth:512u))
    throw std::runtime_error("original Kimi oracle absorbed row shape");
  const auto bank=data.raw(Role::kMlaKvB);
  return dot(actual_input,[&](std::size_t column) {
    // Raw KV-b is [H,256,512], NOT the prepared [H,512,128] K bank.
    const auto raw_index=key?(head*256+column)*512+row:(head*256+128+row)*512+column;
    return coefficient(bank,raw::c::DataType::kBFloat16,raw_index);
  },false);
}
inline std::vector<std::vector<float>> mla(const Data &data,std::span<const float> input,
    std::span<const float> previous_cache,std::size_t capacity,std::size_t position) {
  if(data.component()!=raw::Component::kMla||input.size()!=hidden||!capacity||position>=capacity||
      previous_cache.size()!=capacity*576)
    throw std::runtime_error("original Kimi MLA oracle input extent");
  auto normalized=normalize(data,Role::kMlaInputNorm,input,hidden);
  auto qa=project(data,Role::kMlaQa,normalized);
  qa=normalize(data,Role::kMlaQNorm,qa,1536);
  auto query=project(data,Role::kMlaQb,qa),current=project(data,Role::kMlaKvA,normalized);
  auto latent=normalize(data,Role::kMlaKvNorm,std::span(current).first(512),512);
  std::copy(latent.begin(),latent.end(),current.begin());
  auto cache=std::vector<float>(previous_cache.begin(),previous_cache.end());
  std::copy(current.begin(),current.end(),cache.begin()+position*576);
  std::vector<float> absorbed(heads*512),attended(heads*512),projected(width);
  for(std::size_t head=0;head<heads;++head) {
    const auto no_pe=std::span(query).subspan(head*192,depth);
    for(std::size_t row=0;row<512;++row)
      absorbed[head*512+row]=float(original_absorbed_row(data,head,row,no_pe,true).center);
    std::vector<double> scores(position+1);
    for(std::size_t t=0;t<scores.size();++t) {
      scalar::Sum score;
      for(std::size_t d=0;d<512;++d)score.add(double(absorbed[head*512+d])*cache[t*576+d]);
      for(std::size_t d=0;d<64;++d)score.add(double(query[head*192+128+d])*cache[t*576+512+d]);
      scores[t]=score.value()*double(1.F/std::sqrt(192.F));
    }
    const auto maximum=*std::max_element(scores.begin(),scores.end());scalar::Sum total;
    for(auto &value:scores){value=std::exp(value-maximum);total.add(value);}
    for(std::size_t d=0;d<512;++d) {
      scalar::Sum sum;
      for(std::size_t t=0;t<scores.size();++t)sum.add((scores[t]/total.value())*cache[t*576+d]);
      attended[head*512+d]=float(sum.value());
    }
    for(std::size_t row=0;row<depth;++row)
      projected[head*depth+row]=float(original_absorbed_row(data,head,row,
          std::span(attended).subspan(head*512,512),false).center);
  }
  auto gate=project(data,Role::kMlaGate,normalized),gated=projected;
  for(std::size_t i=0;i<gated.size();++i)gated[i]*=float(sigmoid(gate[i]));
  auto output=project(data,Role::kMlaOutput,gated);
  return {std::move(cache),std::move(query),std::move(absorbed),std::move(attended),std::move(projected),
      std::move(gated),std::move(output),std::move(normalized),std::move(gate)};
}
} // namespace kimi_original_attention_reference
