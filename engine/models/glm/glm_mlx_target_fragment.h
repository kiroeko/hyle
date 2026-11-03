#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "glm_original_target43_fragment.h"
#include "glm_original_target44_fragment.h"

// Fixed MLX-VLM f1a9b7 GLM Flash target shell. All streams and state inputs are
// actual F32; original BF16 dense/HC weights remain immutable BF16 banks.
// The caller appends its real MLA/MoE and owns the ordinary Graph execution.
// There is no role loader, cache owner, route choice or executor here.
namespace glm_mlx_target_fragment {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
namespace hc=glm_hyper_connection_fragment;
namespace target43=glm_original_target43_fragment;
namespace target44=glm_original_target44_fragment;
namespace roles44=glm_original_target44_roles;
using Value=sem::ValueId;
using HcParameters=target43::HcParameters;
using Parameters43=target43::Parameters;
using Ports43=target43::Ports;
using Parameters44=target44::ParameterValues;

void require(sem::Fragment &f,Value value,const c::TensorDesc &expected);
void require_constant(sem::Fragment &f,Value value,const c::TensorDesc &expected);
void require_hc(sem::Fragment &f,const HcParameters &p);
template<class Lookup>
inline Parameters43 parameters43(sem::Fragment &f,Lookup &&lookup,std::string_view layer_prefix) {
  const auto get=[&](std::string_view suffix){return lookup(std::string(layer_prefix)+std::string(suffix));};
  Parameters43 result{
      {get("hc_attn_fn"),get("hc_attn_scale"),get("hc_attn_base"),get("input_layernorm.weight")},
      {get("hc_ffn_fn"),get("hc_ffn_scale"),get("hc_ffn_base"),get("post_attention_layernorm.weight")}};
  require_hc(f,result.attention);require_hc(f,result.ffn);return result;
}
struct Prepared43 final {
  hc::HcPorts attention_hc;
  HcParameters ffn_parameters;
  Value input; // Actual F32 collapsed/learned-RMS row, sole MLA source.
};
Prepared43 prepare43(sem::Fragment &f,Value actual_four_stream_f32,const Parameters43 &p);
template<class MoeResult> struct Result43 final {
  Value mhc; // Actual F32 four streams, no intermediate model.norm.
  Ports43 ports;
  MoeResult moe;
};
template<class MoeBuilder>
inline auto finish43(sem::Fragment &f,const Prepared43 &prepared,Value actual_mla_output_f32,
                     MoeBuilder &&moe_builder) {
  require(f,prepared.attention_hc.residual,sem::dense({1,hc::kStreams,hc::kHidden}));
  require(f,prepared.input,sem::dense({1,hc::kHidden}));
  require(f,actual_mla_output_f32,sem::dense({1,hc::kHidden}));
  require_hc(f,prepared.ffn_parameters);
  using MoeResult=decltype(moe_builder(f,prepared.input));
  Result43<MoeResult> result;auto &ports=result.ports;
  ports.attention_hc=prepared.attention_hc;
  ports.attention_post=hc::hc_post(f,ports.attention_hc,ports.attention_hc.residual,
      actual_mla_output_f32,c::DataType::kFloat32);
  const auto &p=prepared.ffn_parameters;
  ports.ffn_hc=hc::hc_pre(f,ports.attention_post.stored,p.fn,p.scale,p.base,p.norm,
      c::DataType::kFloat32,hc::ProjectionOrder::kNormalizeInput);
  result.moe=moe_builder(f,ports.ffn_hc.norm.stored);
  require(f,result.moe.output,sem::dense({1,hc::kHidden}));
  ports.ffn_post=hc::hc_post(f,ports.ffn_hc,ports.attention_post.stored,
      result.moe.output,c::DataType::kFloat32);
  result.mhc=ports.ffn_post.visible;return result;
}

// Only three 8192x4 convolution banks are widened. These are cold Graph
// outputs, not host assumptions or shadows of the nine large BF16 matrices.
struct ConvolutionPreparation final {
  std::array<Value,3> inputs,outputs; // Q,K,V in explicit source role order.
};
ConvolutionPreparation prepare_convolutions(sem::Fragment &f,const std::array<Value,3> &raw);
template<class Lookup>
inline Parameters44 parameters44(sem::Fragment &f,Lookup &&lookup,const std::array<Value,3> &prepared_convolutions) {
  Parameters44 result;
  constexpr std::array conv_roles{roles44::Role::kQueryConv,roles44::Role::kKeyConv,roles44::Role::kValueConv};
  for(std::size_t i=0;i<prepared_convolutions.size();++i)
    require_constant(f,prepared_convolutions[i],sem::dense({target44::kWidth,4}));
  for(std::size_t i=0;i<result.size();++i) {
    const auto role=static_cast<roles44::Role>(i);
    if(roles44::is_conv(role)) {
      for(std::size_t leg=0;leg<conv_roles.size();++leg)
        if(role==conv_roles[leg])result[i]=prepared_convolutions[leg];
    } else {
      // Names are shared schema suffixes, not official pins or official data.
      // lookup supplies this identity's distinct raw role and model.norm.
      result[i]=lookup(roles44::spec(role).name);
      require_constant(f,result[i],roles44::raw_description(role));
    }
  }
  return result;
}
template<class UnscaledRecurrent,class MoeBuilder>
inline auto append44(sem::Fragment &f,Value actual_four_stream_f32,Value actual_history_f32,
    Value actual_matrix_f32,const Parameters44 &parameters,
    UnscaledRecurrent &&unscaled_recurrent,MoeBuilder &&moe_builder) {
  // MLX q/k RMS and scalar stores are explicit inside kda_with_recurrent.
  // unscaled_recurrent must return {actual F32 y=S*q,actual F32 next_state}.
  // Public GatedDeltaNet{} currently includes 1/sqrt(Dk) and cannot be passed
  // here unchanged. The caller must bind a genuinely unscaled implementation;
  // no implicit fallback or reciprocal input/output compensation exists.
  return target44::append_with_moe(f,actual_four_stream_f32,actual_history_f32,
      actual_matrix_f32,parameters,c::DataType::kFloat32,hc::ProjectionOrder::kNormalizeInput,
      target44::QueryKeyNormalization::kMlxRmsPreScaled,
      std::forward<UnscaledRecurrent>(unscaled_recurrent),std::forward<MoeBuilder>(moe_builder));
}
} // namespace glm_mlx_target_fragment
