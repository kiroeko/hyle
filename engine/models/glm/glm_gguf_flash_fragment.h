#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "glm_gguf_mtp_fragment.h"
#include "glm_named_parameters.h"
#include "glm_mlx_mtp_fragment.h"
#include "glm_original_target43_fragment.h"
#include "glm_original_target44_fragment.h"

// Converted Flash weights, explicit F32 runtime. Only named-constant binding
// and pure composition live here. The pooled MLA/history mathematics and its
// 16 entries are shared with the existing plain-cache path; the caller still
// owns Program, selected-expert Supply, K1, StateHead and accepted repair.
namespace glm_gguf_flash_fragment {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
namespace core=hyle::inference::model_definition::ops;
namespace named=glm_gguf_named_parameters;
namespace types=glm_gguf_model_types;
namespace fp=glm_gguf_compute_fragment;
namespace leaf=glm_gguf_projection_fragment;
namespace pooled=glm_mlx_mtp_fragment;
namespace moe=glm_gguf_mtp_fragment;
namespace hc=glm_hyper_connection_fragment;
namespace target43=glm_original_target43_fragment;
namespace target44=glm_original_target44_fragment;
namespace roles44=glm_original_target44_roles;
using Value=sem::ValueId;
using State=pooled::State;
using HcParameters=target43::HcParameters;
inline constexpr pooled::IndexPolicy kIndexPolicy{
    pooled::IndexWeightScale::kCombined,pooled::PoolOrder::kDescendingScore};
inline constexpr std::size_t kStateEntries=16;

// These are algorithm/shape capabilities, never a repository/model selector.
// The eventual fixture must derive these values from its own original header.
void require_geometry(const types::Geometry &g);
void require(sem::Fragment &f,Value value,const c::TensorDesc &expected);
std::array<Value,kStateEntries> state_values(const State &s);
State bind_state(sem::Fragment &f,const std::array<Value,kStateEntries> &v);

struct PoolParameters final {
  types::NextnParameters nextn;
  Value ape,index_gate;
  leaf::PackedProjections project;
};
struct VocabularyNames final {std::string_view embedding,head,target_norm;};
struct Policy final {
  types::Geometry geometry;
  std::array<glm_moe_geometry::Geometry,3> moe_geometry; // target,target tail,draft
  std::array<std::optional<float>,3> shared_swiglu_limit;
  float gate_lower_bound,hc_epsilon;
  std::int64_t hc_streams,hc_iterations,conv_kernel,kda_depth;
  bool index_select_tail,sigmoid_gating,normalize_expert_weights;
};
struct Parameters final {
  PoolParameters nextn,target43;
  std::array<HcParameters,2> target43_hc;
  target44::ParameterValues target44;
  types::Block target44_moe;
  types::Geometry geometry;
  std::array<glm_moe_geometry::Geometry,3> moe_geometry; // target43,target44,draft
  std::array<std::optional<float>,3> shared_swiglu_limit;
  float gate_lower_bound;
};
template<class Lookup>
inline types::Block moe_parameters(const Lookup &value,std::string_view prefix) {
  const auto get=[&](std::string_view suffix){return value(std::string(prefix)+std::string(suffix));};
  types::Block b;
  b.post_norm=get("ffn_norm.weight");b.router=get("ffn_gate_inp.weight");b.bias=get("exp_probs_b.bias");
  b.shared_gate=get("ffn_gate_shexp.weight");b.shared_up=get("ffn_up_shexp.weight");b.shared_down=get("ffn_down_shexp.weight");
  return b;
}
template<class Lookup>
inline PoolParameters block_parameters(const Lookup &value,std::string_view prefix) {
  const auto get=[&](std::string_view suffix){return value(std::string(prefix)+std::string(suffix));};
  PoolParameters p;p.nextn.block=moe_parameters(value,prefix);auto &b=p.nextn.block;
  b.input_norm=get("attn_norm.weight");b.query_a=get("attn_q_a.weight");b.query_norm=get("attn_q_a_norm.weight");
  b.query_b=get("attn_q_b.weight");b.kv_a=get("attn_kv_a_mqa.weight");b.kv_norm=get("attn_kv_a_norm.weight");
  b.key=get("attn_k_b.weight");b.value=get("attn_v_b.weight");b.output=get("attn_output.weight");
  b.index_key=get("indexer.attn_k.weight");b.index_query=get("indexer.attn_q_b.weight");
  b.index_norm=get("indexer.k_norm.weight");b.index_bias=get("indexer.k_norm.bias");b.index_weights=get("indexer.proj.weight");
  p.ape=get("indexer_compressor_ape.weight");p.index_gate=get("indexer_compressor_gate.weight");return p;
}
template<class Lookup>
inline HcParameters hc_parameters(const Lookup &value,std::string_view prefix,bool ffn) {
  const auto get=[&](std::string_view suffix){return value(std::string(prefix)+std::string(suffix));};
  return ffn?HcParameters{get("hc_ffn_fn.weight"),get("hc_ffn_scale.weight"),get("hc_ffn_base.weight"),get("ffn_norm.weight")}:
      HcParameters{get("hc_attn_fn.weight"),get("hc_attn_scale.weight"),get("hc_attn_base.weight"),get("attn_norm.weight")};
}
hc::HcPorts hc_pre(sem::Fragment &f,Value residual,const HcParameters &p);
target44::ProjectionPorts project(sem::Fragment &f,Value actual,Value weight,c::DataType storage);
template<class Lookup>
inline Parameters parameters(sem::Fragment &f,const Lookup &value,std::int64_t first_target,
    std::int64_t last_target,std::int64_t draft,const VocabularyNames &vocab,
    const Policy &policy) {
  const auto &geometry=policy.geometry;const auto &moe_geometry=policy.moe_geometry;
  const auto gate_lower_bound=policy.gate_lower_bound;
  glm_gguf_flash_fragment::require_geometry(geometry);
  if(first_target<0||last_target!=first_target+1||draft!=last_target+1||
      !std::isfinite(gate_lower_bound)||!(gate_lower_bound<0.F)||
      vocab.embedding.empty()||vocab.head.empty()||vocab.target_norm.empty())
    throw std::runtime_error("converted Flash needs explicit original layer/vocabulary/gate contracts");
  if(policy.hc_epsilon!=hc::kHcEpsilon||policy.hc_streams!=4||policy.hc_iterations!=20||
      policy.conv_kernel!=4||policy.kda_depth!=128||!policy.index_select_tail||
      !policy.sigmoid_gating||!policy.normalize_expert_weights)
    throw std::runtime_error("actual GGUF HC/KDA/index/route policy is outside the current shared pure implementation");
  for(const auto &g:moe_geometry) {
    glm_moe_geometry::require(g);
    if(g.hidden!=4096||g.intermediate!=2048||g.experts!=288||g.slots!=8)
      throw std::runtime_error("converted Flash actual expert geometry differs from shared pure route");
    if(g.swiglu_limit&&!(*g.swiglu_limit>1.e-6F))
      throw std::runtime_error("active routed clamp must match the original reader's epsilon predicate");
  }
  for(const auto limit:policy.shared_swiglu_limit)
    if(limit&&(!std::isfinite(*limit)||!(*limit>1.e-6F)))
      throw std::runtime_error("active shared clamp must match the original reader's epsilon predicate");
  Parameters p;p.geometry=geometry;p.moe_geometry=moe_geometry;p.gate_lower_bound=gate_lower_bound;
  p.shared_swiglu_limit=policy.shared_swiglu_limit;
  const auto first="blk."+std::to_string(first_target)+".",last="blk."+std::to_string(last_target)+".",next="blk."+std::to_string(draft)+".";
  p.target43=block_parameters(value,first);p.nextn=block_parameters(value,next);
  p.target43_hc={hc_parameters(value,first,false),hc_parameters(value,first,true)};
  p.target44_moe=moe_parameters(value,last);
  auto &n=p.nextn.nextn;n.eh=value(next+"nextn.eh_proj.weight");n.embedding_norm=value(next+"nextn.enorm.weight");
  n.hidden_norm=value(next+"nextn.hnorm.weight");n.shared_norm=value(next+"nextn.shared_head_norm.weight");
  n.embedding=value(vocab.embedding);n.head=value(vocab.head);
  // Role order is only a shared mathematical slot order. Every value below
  // is this artifact's own actual GGUF role, not an official BF16 parameter.
  using Role=roles44::Role;
  constexpr std::array<std::pair<Role,std::string_view>,roles44::kRoles> suffix{{
      {Role::kHcAttnBase,"hc_attn_base.weight"},{Role::kHcAttnFn,"hc_attn_fn.weight"},{Role::kHcAttnScale,"hc_attn_scale.weight"},
      {Role::kHcFfnBase,"hc_ffn_base.weight"},{Role::kHcFfnFn,"hc_ffn_fn.weight"},{Role::kHcFfnScale,"hc_ffn_scale.weight"},
      {Role::kInputNorm,"attn_norm.weight"},{Role::kPostNorm,"ffn_norm.weight"},{Role::kALog,"ssm_a"},
      {Role::kBeta,"ssm_beta.weight"},{Role::kDtBias,"ssm_dt.bias"},{Role::kForgetA,"ssm_f_a.weight"},{Role::kForgetB,"ssm_f_b.weight"},
      {Role::kGateA,"ssm_g_a.weight"},{Role::kGateB,"ssm_g_b.weight"},{Role::kKeyConv,"ssm_conv1d_k.weight"},{Role::kKey,"attn_k.weight"},
      {Role::kOutputNorm,"ssm_norm.weight"},{Role::kOutput,"attn_output.weight"},{Role::kQueryConv,"ssm_conv1d_q.weight"},{Role::kQuery,"attn_q.weight"},
      {Role::kValueConv,"ssm_conv1d_v.weight"},{Role::kValue,"attn_v.weight"},{Role::kModelNorm,""}}};
  for(const auto &[role,suffix_name]:suffix) {
    auto actual=value(role==Role::kModelNorm?std::string(vocab.target_norm):last+std::string(suffix_name));
    if(roles44::is_conv(role)) {
      require(f,actual,sem::dense({8192,1,4}));actual=fp::reshape(f,actual,{8192,4});
    }
    p.target44[std::size_t(role)]=actual;
  }
  return p;
}
template<class Bound>
inline auto connect(sem::Fragment &f,Bound bound,std::int64_t first_target,std::int64_t last_target,
    std::int64_t draft,const VocabularyNames &vocab,const Policy &policy) {
  const auto build_parameters=[&](const auto &lookup) {
    return parameters(f,lookup,first_target,last_target,draft,vocab,policy);
  };
  return named::connect_with<Parameters>(f,std::move(bound),build_parameters);
}

using NextnResult=pooled::NextnResult<moe::Moe>;
moe::Moe feed_forward(sem::Fragment &f,Value actual,const types::Block &block,
    const Parameters &p,std::size_t stage,const std::array<Value,3> &banks,Value map);
NextnResult append_nextn(sem::Fragment &f,State state,Value actual_hidden,const Parameters &p,
    const std::array<Value,3> &banks,Value map,bool fresh,std::int32_t position,std::int32_t carried_tail);
pooled::History append_history(sem::Fragment &f,State &state,Value actual_normalized,const Parameters &p,std::int32_t position);
struct Target43Result final {
  pooled::Mla mla;types::Projection query;fp::Norm query_norm;target43::Ports ports;moe::Moe moe;Value mhc;
  template<class Visitor>void visit_values(Visitor &&v) {
    mla.visit_values(v);query.visit_values(v);query_norm.visit_values(v);ports.visit_values(v);moe.visit_values(v);v(mhc);
  }
};
Target43Result append_target43(sem::Fragment &f,State state,Value actual_four_streams,const Parameters &p,
    const std::array<Value,3> &banks,Value map,std::int32_t position);
target44::LayerResult<moe::Moe> append_target44(sem::Fragment &f,Value actual_four_streams,
    Value history,Value matrix,const Parameters &p,const std::array<Value,3> &banks,Value map);
} // namespace glm_gguf_flash_fragment
