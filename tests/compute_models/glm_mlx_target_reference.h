#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "glm_hyper_connection_reference.h"
#include "glm_mlx_original_reference.h"
#include "glm_mlx_target_fragment.h"

// Actual-predecessor checks for the fixed MLX F32 HC/KDA path. Source is a
// typed original-byte view with bytes(name) and coefficients(name); it is not
// a tested Graph constant or a floating copy of a large weight matrix.
// MLA, routing and MoE projection observers remain the caller's shared ones.
namespace glm_mlx_target_reference {
namespace graph=glm_mlx_target_fragment;
namespace kda=glm_original_target44_fragment;
namespace roles=glm_original_target44_roles;
namespace hc=dspark_original_hc_reference;
namespace d=glm_hyper_connection_reference::detail;
namespace dense=glm_mlx_original_reference;
using Value=graph::Value;
using F32=std::span<const float>;
inline constexpr std::size_t H=4096,S=4,Heads=64,D=128,W=Heads*D,Channels=3*W;

inline bool same(F32 actual,F32 expected,std::string_view label) {
  if(actual.size()!=expected.size()){ADD_FAILURE()<<label<<" actual predecessor extent";return false;}
  for(std::size_t i=0;i<actual.size();++i)
    if(!std::isfinite(actual[i])||!std::isfinite(expected[i])||
        std::bit_cast<std::uint32_t>(actual[i])!=std::bit_cast<std::uint32_t>(expected[i])) {
      ADD_FAILURE()<<label<<'['<<i<<"] differs from its actual F32 predecessor";return false;
    }
  return true;
}
inline std::vector<float> rms(F32 input,F32 gamma,std::size_t width,float epsilon) {
  if(!width||input.empty()||input.size()%width||(!gamma.empty()&&gamma.size()!=width)||
      !std::isfinite(epsilon)||!(epsilon>0))throw std::runtime_error("MLX F32 RMS reference geometry");
  hc::detail::finite(input);hc::detail::finite(gamma);
  std::vector<float> result(input.size());
  for(std::size_t row=0;row<input.size()/width;++row) {
    hc::detail::Sum square;
    for(std::size_t col=0;col<width;++col) {
      const auto x=input[row*width+col];
      if(x!=0&&!std::isnormal(x))throw std::runtime_error("MLX RMS needs separate subnormal-source analysis");
      square.add(double(x)*x);
    }
    if(square.value()>double(std::numeric_limits<float>::max())/4)
      throw std::runtime_error("MLX RMS needs separate square-overflow analysis");
    const double inverse=1/std::sqrt(square.value()/double(width)+double(epsilon));
    for(std::size_t col=0;col<width;++col)
      result[row*width+col]=float(double(input[row*width+col])*inverse*(gamma.empty()?1.:double(gamma[col])));
  }
  return result;
}
inline std::vector<float> collapse(F32 streams,F32 pre) {
  if(streams.size()!=S*H||pre.size()!=S)throw std::runtime_error("MLX F32 HC collapse extent");
  hc::detail::finite(streams);hc::detail::finite(pre);std::vector<float> result(H);
  for(std::size_t col=0;col<H;++col) {
    hc::detail::Sum sum;for(std::size_t s=0;s<S;++s)sum.add(double(streams[s*H+col])*pre[s]);
    result[col]=float(sum.value());
  }
  return result;
}
inline std::vector<float> expand(F32 branch,F32 residual,F32 post,F32 comb) {
  if(branch.size()!=H||residual.size()!=S*H||post.size()!=S||comb.size()!=S*S)
    throw std::runtime_error("MLX F32 HC expansion extent");
  hc::detail::finite(branch);hc::detail::finite(residual);hc::detail::finite(post);hc::detail::finite(comb);
  std::vector<float> result(S*H);
  for(std::size_t dst=0;dst<S;++dst)for(std::size_t col=0;col<H;++col) {
    hc::detail::Sum sum;sum.add(double(branch[col])*post[dst]);
    for(std::size_t src=0;src<S;++src)sum.add(double(residual[src*H+col])*comb[src*S+dst]);
    result[dst*H+col]=float(sum.value());
  }
  return result;
}
template<class Actual>
inline bool expect_norm(const glm_hyper_connection_fragment::NormPorts &p,F32 gamma,Actual &&actual) {
  const auto input=actual(p.input),raw=actual(p.raw);
  return d::local(raw,rms(input,gamma,gamma.size(),kda::kNormEpsilon),"MLX actual F32 learned RMS")&&
      same(actual(p.stored),raw,"MLX RMS retains F32")&&same(actual(p.visible),raw,"MLX RMS F32 visible");
}
template<class Source,class Actual>
inline bool expect_pre(const Source &source,const glm_hyper_connection_fragment::HcPorts &p,
    std::string_view prefix,bool attention,Actual &&actual) {
  const auto name=[&](std::string_view suffix){return std::string(prefix)+std::string(suffix);};
  const auto stem=attention?"hc_attn_":"hc_ffn_";
  if(p.projection_order!=glm_hyper_connection_fragment::ProjectionOrder::kNormalizeInput||!p.normalized_input) {
    ADD_FAILURE()<<"MLX HC must expose its actual normalize-before-projection boundary";return false;
  }
  const auto residual=actual(p.residual),flat=actual(p.flat),normalized=actual(*p.normalized_input);
  const auto projection=actual(p.projection),mixes=actual(p.mixes);
  if(residual.size()!=S*H||!same(flat,residual,"MLX HC actual flat streams")||
      !d::local(normalized,rms(flat,{},S*H,kda::kNormEpsilon),"MLX HC actual normalized projection input")||
      !dense::expect_projection(projection,normalized,1,24,S*H,
          dense::BFloat{source.bytes(name(std::string(stem)+"fn")),24,S*H},true)||
      !same(mixes,projection,"MLX HC mixes are the actual normalized-input dot"))return false;
  const auto expected=hc::split(mixes,source.coefficients(name(std::string(stem)+"scale")),
      source.coefficients(name(std::string(stem)+"base")),kda::kHcEpsilon);
  const auto pre=actual(p.pre),collapsed=actual(p.collapsed_raw);
  const auto gamma=source.coefficients(name(attention?"input_layernorm.weight":"post_attention_layernorm.weight"));
  return d::local(pre,expected.pre,"MLX HC actual pre sigmoid")&&
      d::local(actual(p.post),expected.post,"MLX HC actual post sigmoid")&&
      d::local(actual(p.combination),expected.comb,"MLX HC actual Sinkhorn20")&&
      d::local(collapsed,collapse(flat,pre),"MLX actual F32 four-stream collapse")&&
      same(actual(p.collapsed_stored),collapsed,"MLX HC F32 collapse store")&&
      same(actual(p.norm.input),collapsed,"MLX HC learned RMS actual collapse")&&expect_norm(p.norm,gamma,actual);
}
template<class Actual>
inline bool expect_post(const glm_hyper_connection_fragment::PostPorts &p,
    const glm_hyper_connection_fragment::HcPorts &pre,Actual &&actual) {
  const auto raw=actual(p.raw);
  return d::local(raw,expand(actual(p.branch),actual(p.residual),actual(pre.post),actual(pre.combination)),
      "MLX actual F32 HC expansion")&&same(actual(p.stored),raw,"MLX HC F32 post store")&&
      same(actual(p.visible),raw,"MLX HC F32 post visible");
}
template<class Source,class MoeResult,class Actual>
inline bool observe43(const Source &source,const graph::Result43<MoeResult> &model,
    Actual &&actual,std::string_view prefix) {
  const auto &p=model.ports;
  return expect_pre(source,p.attention_hc,prefix,true,actual)&&expect_post(p.attention_post,p.attention_hc,actual)&&
      same(actual(p.ffn_hc.residual),actual(p.attention_post.stored),"MLX43 FFN actual streams")&&
      expect_pre(source,p.ffn_hc,prefix,false,actual)&&
      same(actual(p.ffn_post.branch),actual(model.moe.output),"MLX43 actual MoE branch")&&
      same(actual(p.ffn_post.residual),actual(p.attention_post.stored),"MLX43 post actual residual")&&
      expect_post(p.ffn_post,p.ffn_hc,actual)&&same(actual(model.mhc),actual(p.ffn_post.visible),"MLX43 F32 output streams");
}
template<class Actual>
inline bool expect_query_key(const kda::L2Ports &p,F32 predecessor,bool query,Actual &&actual) {
  if(p.normalization!=kda::QueryKeyNormalization::kMlxRmsPreScaled||!p.normalized_raw) {
    ADD_FAILURE()<<"MLX KDA must expose actual F32 RMS and separate q/k scale";return false;
  }
  const auto input=actual(p.input),normalized=actual(*p.normalized_raw),output=actual(p.output);
  if(input.size()!=W||output.size()!=W||!same(input,predecessor,"MLX KDA actual convolution Q/K")||
      !d::local(normalized,rms(input,{},D,kda::kL2Epsilon/float(D)),"MLX KDA actual RMS eps1e-6/128"))return false;
  const float scale=query?1.0F/float(D):static_cast<float>(1.0/std::sqrt(double(D)));
  for(std::size_t i=0;i<W;++i)
    if(!d::rounded(output[i],double(normalized[i])*scale,"MLX KDA explicit F32 q/k scalar",i))return false;
  return true;
}
template<class Source,class MoeResult,class Actual>
inline bool observe44(const Source &source,const kda::LayerResult<MoeResult> &model,
                     Actual &&actual,bool full) {
  const auto &p=model.ports;const auto &a=p.kda;
  constexpr std::string_view prefix="model.language_model.layers.44.";
  if(!expect_pre(source,p.attention_hc,prefix,true,actual))return false;
  constexpr std::array projection_roles{roles::Role::kQuery,roles::Role::kKey,roles::Role::kValue,
      roles::Role::kForgetA,roles::Role::kForgetB,roles::Role::kBeta,roles::Role::kGateA,
      roles::Role::kGateB,roles::Role::kOutput};
  for(std::size_t i=0;i<a.projections.size();++i) {
    const auto &part=a.projections[i];const auto &spec=roles::spec(projection_roles[i]);
    Value predecessor=p.attention_hc.norm.stored;
    if(i==std::size_t(kda::Projection::kForgetB))predecessor=a.projections[std::size_t(kda::Projection::kForgetA)].stored;
    if(i==std::size_t(kda::Projection::kGateB))predecessor=a.projections[std::size_t(kda::Projection::kGateA)].stored;
    if(i==std::size_t(kda::Projection::kOutput))predecessor=a.gated_stored;
    const auto input=actual(part.input),raw=actual(part.raw);
    if(!same(input,actual(predecessor),"MLX44 actual dense predecessor")||
        !dense::expect_projection(raw,input,1,std::size_t(spec.shape[0]),std::size_t(spec.shape[1]),
            dense::BFloat{source.bytes(spec.name),std::size_t(spec.shape[0]),std::size_t(spec.shape[1])},full)||
        !same(actual(part.stored),raw,"MLX44 dense retains F32 output"))return false;
  }
  const auto previous=actual(a.history_input),history=actual(a.history_stored),window=actual(a.window),next=actual(a.next_history);
  if(history.size()!=3*Channels||window.size()!=4*Channels||next.size()!=3*Channels||
      !same(history,previous,"MLX44 actual F32 history")||
      !same(F32(window).first(history.size()),history,"MLX44 three chronological history rows")||
      !same(next,F32(window).subspan(Channels),"MLX44 next history rows1..3")||
      !same(actual(model.history),next,"MLX44 published F32 history"))return false;
  constexpr std::array conv_roles{roles::Role::kQueryConv,roles::Role::kKeyConv,roles::Role::kValueConv};
  for(std::size_t leg=0;leg<3;++leg) {
    const auto &part=a.conv[leg];const auto samples=actual(part.samples),projected=actual(a.projections[leg].stored);
    const auto raw=actual(part.raw),activated=actual(part.activated_raw),stored=actual(part.stored);
    if(samples.size()!=4*W||projected.size()!=W||raw.size()!=W||activated.size()!=W||stored.size()!=W) {
      ADD_FAILURE()<<"MLX44 actual convolution extent";return false;
    }
    const auto bytes=source.bytes(roles::spec(conv_roles[leg]).name);
    const dense::BFloat original{bytes,W,4};
    std::vector<float> expected_activation(W);
    for(std::size_t channel=0;channel<W;++channel) {
      if(std::bit_cast<std::uint32_t>(window[3*Channels+leg*W+channel])!=std::bit_cast<std::uint32_t>(projected[channel])) {
        ADD_FAILURE()<<"MLX44 newest history row is not actual Q/K/V projection";return false;
      }
      std::array<float,4> values{};
      for(std::size_t tap=0;tap<4;++tap) {
        values[tap]=samples[tap*W+channel];
        if(std::bit_cast<std::uint32_t>(values[tap])!=std::bit_cast<std::uint32_t>(window[tap*Channels+leg*W+channel])) {
          ADD_FAILURE()<<"MLX44 convolution is not the actual chronological window";return false;
        }
      }
      const auto weight=[&](std::size_t,std::size_t tap){return original(channel,tap);};
      if(!dense::expect_projection(F32(raw).subspan(channel,1),values,1,1,4,weight,true))return false;
      expected_activation[channel]=float(double(raw[channel])*hc::detail::sigmoid(raw[channel]));
    }
    if(!d::local(activated,expected_activation,"MLX44 actual convolution SiLU")||
        !same(stored,activated,"MLX44 convolution F32 activation store"))return false;
  }
  if(!expect_query_key(a.query,actual(a.conv[0].stored),true,actual)||
      !expect_query_key(a.key,actual(a.conv[1].stored),false,actual)||
      !same(actual(a.value),actual(a.conv[2].stored),"MLX44 actual F32 V"))return false;
  const auto alpha=actual(a.alpha),add=actual(a.forget_add),product=actual(a.forget_product);
  const auto sigmoid=actual(a.forget_sigmoid),decay=actual(a.decay),beta=actual(a.beta);
  const auto forget=actual(a.projections[std::size_t(kda::Projection::kForgetB)].stored);
  const auto beta_input=actual(a.projections[std::size_t(kda::Projection::kBeta)].stored);
  const auto log=source.coefficients(roles::spec(roles::Role::kALog).name);
  const auto dt=source.coefficients(roles::spec(roles::Role::kDtBias).name);
  if(alpha.size()!=Heads||log.size()!=Heads||dt.size()!=W||forget.size()!=W||add.size()!=W||product.size()!=W||
      sigmoid.size()!=W||decay.size()!=W||beta.size()!=Heads||beta_input.size()!=Heads) {
    ADD_FAILURE()<<"MLX44 actual safe gate extent";return false;
  }
  std::vector<float> expected_alpha(Heads);
  for(std::size_t h=0;h<Heads;++h)expected_alpha[h]=float(std::exp(double(log[h])));
  if(!d::local(alpha,expected_alpha,"MLX44 exp(original A_log)")||
      !d::local(sigmoid,d::sigmoid(product),"MLX44 actual safe decay sigmoid")||
      !d::local(beta,d::sigmoid(beta_input),"MLX44 sigmoid actual b"))return false;
  for(std::size_t i=0;i<W;++i)
    if(!d::rounded(add[i],double(forget[i])+dt[i],"MLX44 actual f plus dt",i)||
        !d::rounded(product[i],double(add[i])*alpha[i/D],"MLX44 actual exp(A_log) product",i)||
        !d::rounded(decay[i],double(sigmoid[i])*-5.,"MLX44 actual log decay",i))return false;
  const auto query=actual(a.query.output),key=actual(a.key.output),value=actual(a.value);
  const auto initial=actual(a.state_input),state=actual(a.state_output),attended=actual(a.recurrent_raw);
  if(initial.size()!=Heads*D*D||state.size()!=initial.size()||attended.size()!=W) {
    ADD_FAILURE()<<"MLX44 actual recurrent [H,V,K] state extent";return false;
  }
  std::vector<float> expected_state(state.size()),expected_output(W);
  for(std::size_t h=0;h<Heads;++h)for(std::size_t v=0;v<D;++v) {
    const auto row=(h*D+v)*D;std::array<double,D> decayed{};hc::detail::Sum prediction;
    for(std::size_t k=0;k<D;++k) {
      decayed[k]=double(initial[row+k])*std::exp(double(decay[h*D+k]));
      prediction.add(decayed[k]*key[h*D+k]);
    }
    const double delta=double(beta[h])*(double(value[h*D+v])-prediction.value());hc::detail::Sum output;
    for(std::size_t k=0;k<D;++k) {
      expected_state[row+k]=float(decayed[k]+delta*key[h*D+k]);
      output.add(double(state[row+k])*query[h*D+k]);
    }
    expected_output[h*D+v]=float(output.value()); // Actual q is already scaled; no extra sqrt(D).
  }
  if(!d::local(state,expected_state,"MLX44 actual F32 recurrent state")||
      !d::local(attended,expected_output,"MLX44 actual-state unscaled output")||
      !same(actual(a.recurrent_stored),attended,"MLX44 recurrent F32 store")||
      !same(actual(a.output_norm_input),attended,"MLX44 output RMS actual recurrent predecessor")||
      !same(actual(model.state),state,"MLX44 published actual recurrent state"))return false;
  const auto norm_input=actual(a.output_norm_input),norm_raw=actual(a.output_norm_raw);
  const auto gate=actual(a.gate_sigmoid),gated=actual(a.gated_raw),gate_input=actual(a.projections[std::size_t(kda::Projection::kGateB)].stored);
  const auto gamma=source.coefficients(roles::spec(roles::Role::kOutputNorm).name);
  if(gate.size()!=W||gated.size()!=W||norm_raw.size()!=W||gate_input.size()!=W||gamma.size()!=D||
      !d::local(norm_raw,rms(norm_input,gamma,D,kda::kNormEpsilon),"MLX44 actual F32 output RMS")||
      !d::local(gate,d::sigmoid(gate_input),"MLX44 actual g sigmoid"))return false;
  for(std::size_t i=0;i<W;++i)
    if(!d::rounded(gated[i],double(norm_raw[i])*gate[i],"MLX44 actual RMS times sigmoid",i))return false;
  if(!same(actual(a.gated_stored),gated,"MLX44 gated F32 store")||
      !same(actual(p.attention_post.branch),actual(a.projections[std::size_t(kda::Projection::kOutput)].stored),"MLX44 actual O branch")||
      !expect_post(p.attention_post,p.attention_hc,actual)||
      !same(actual(p.ffn_hc.residual),actual(p.attention_post.stored),"MLX44 FFN actual streams")||
      !expect_pre(source,p.ffn_hc,prefix,false,actual)||
      !same(actual(p.ffn_post.branch),actual(model.moe.output),"MLX44 actual MoE branch")||
      !same(actual(p.ffn_post.residual),actual(p.attention_post.stored),"MLX44 post actual residual")||
      !expect_post(p.ffn_post,p.ffn_hc,actual))return false;
  const auto expanded=actual(p.ffn_post.visible),mean=actual(p.contract_raw);
  if(expanded.size()!=S*H||mean.size()!=H){ADD_FAILURE()<<"MLX44 actual final mean extent";return false;}
  for(std::size_t col=0;col<H;++col) {
    std::array<float,S> values{};for(std::size_t s=0;s<S;++s)values[s]=expanded[s*H+col];
    if(!d::reduction(mean[col],values,double(S),"MLX44 actual four-stream mean"))return false;
  }
  const auto model_gamma=source.coefficients(roles::spec(roles::Role::kModelNorm).name);
  return same(actual(p.contract_stored),mean,"MLX44 final mean F32 store")&&
      same(actual(p.model_norm.input),mean,"MLX44 model.norm actual F32 mean")&&
      expect_norm(p.model_norm,model_gamma,actual)&&same(actual(model.mhc),expanded,"MLX44 F32 four-stream output")&&
      same(actual(model.tap),actual(p.model_norm.visible),"MLX44 actual F32 final target tap");
}
} // namespace glm_mlx_target_reference
