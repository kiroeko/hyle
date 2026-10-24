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
#include "glm_gguf_named_parameters.h"
#include "glm_mlx_mtp_fragment.h"
#include "glm_original_target43_fragment.h"
#include "glm_original_target44_fragment.h"

// Converted Flash weights, explicit F32 runtime. Only named-constant binding
// and pure composition live here. The pooled MLA/history mathematics and its
// 16 entries are shared with the existing plain-cache path; the caller still
// owns Program, selected-expert Supply, K1, StateHead and accepted repair.
namespace glm_gguf_flash_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
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
inline void require_geometry(const types::Geometry &g) {
  types::require_geometry(g);
  if(g.hidden!=4096||g.query_rank!=1536||g.heads!=64||g.nope!=256||g.rope!=0||
      g.latent!=512||g.value!=256||g.index_depth!=128||g.index_heads!=32||
      g.index_pool!=4||g.index_topk!=2048||g.rms_epsilon!=1.e-5F||
      g.index_epsilon!=1.e-6F||g.attention_scale!=1.F/16.F)
    throw std::runtime_error("converted pooled MLA geometry is outside the current shared pure algorithm");
}
inline void require(sem::Fragment &f,Value value,const c::TensorDesc &expected) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=expected)
    throw std::runtime_error("converted Flash actual predecessor/role descriptor differs");
}
inline std::array<Value,kStateEntries> state_values(const State &s) {
  return {s.main,s.raw_key,s.raw_gate,s.pool,s.hidden,s.query,s.pending,s.mla_cursor,
      s.index_cursor,s.pool_count,s.ring_offset,s.tail_valid,s.tail_start,s.selection,s.journal,s.lag};
}
inline State bind_state(sem::Fragment &f,const std::array<Value,kStateEntries> &v) {
  for(const auto value:v)if(value.ordinal()>=f.descriptions.size())
    throw std::runtime_error("converted pooled state contains a value from another fragment");
  const auto &main=f.description(v[0]);
  if(main.rank()!=2||main.dimensions()[0]<2048||main.dimensions()[0]>8192)
    throw std::runtime_error("converted pooled cache capacity is outside its explicit cursor domain");
  require(f,v[0],sem::dense({main.dimensions()[0],512}));
  require(f,v[1],sem::dense({6,128}));require(f,v[2],sem::dense({6,128}));
  require(f,v[3],sem::dense({main.dimensions()[0]/4,128}));
  require(f,v[4],sem::dense({1,4096}));require(f,v[5],sem::dense({1,1536}));
  for(const auto slot:{6u,7u,8u,9u,10u,11u,12u,15u})require(f,v[slot],sem::dense({1},c::DataType::kInt32));
  require(f,v[13],sem::dense({2051},c::DataType::kInt32));
  const auto &journal=f.description(v[14]);
  if(journal.rank()!=2||journal.dimensions()[0]<1)throw std::runtime_error("converted skip journal has no explicit rows");
  require(f,v[14],sem::dense({journal.dimensions()[0],4096}));
  return {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]};
}

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
inline hc::HcPorts hc_pre(sem::Fragment &f,Value residual,const HcParameters &p) {
  const auto packed=[](sem::Fragment &graph,Value actual,Value weight){return leaf::linear(graph,actual,weight).output;};
  return hc::hc_pre_with_projection(f,residual,p.fn,p.scale,p.base,p.norm,c::DataType::kFloat32,
      hc::ProjectionOrder::kNormalizeInput,packed);
}
inline target44::ProjectionPorts project(sem::Fragment &f,Value actual,Value weight,c::DataType storage) {
  if(storage!=c::DataType::kFloat32)throw std::runtime_error("converted packed projection has an explicit F32 result/store policy");
  const auto p=leaf::linear(f,actual,weight);return {p.input,p.output,p.output};
}
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
inline moe::Moe feed_forward(sem::Fragment &f,Value actual,const types::Block &block,
    const Parameters &p,std::size_t stage,const std::array<Value,3> &banks,Value map) {
  if(stage>=p.moe_geometry.size())throw std::runtime_error("converted original MoE stage is outside the explicit contract");
  return moe::append_moe_with_limits(f,actual,block,banks,map,p.moe_geometry[stage],
      p.shared_swiglu_limit[stage],moe::ExpertFold::kSelectedOrder);
}
inline NextnResult append_nextn(sem::Fragment &f,State state,Value actual_hidden,const Parameters &p,
    const std::array<Value,3> &banks,Value map,bool fresh,std::int32_t position,std::int32_t carried_tail) {
  const auto moe_builder=[&](sem::Fragment &graph,Value actual) {
    return feed_forward(graph,actual,p.nextn.nextn.block,p,2,banks,map);
  };
  return pooled::append_with_moe(f,state,actual_hidden,p.nextn,fresh,position,carried_tail,p.geometry,kIndexPolicy,moe_builder);
}
inline pooled::History append_history(sem::Fragment &f,State &state,Value actual_normalized,const Parameters &p,std::int32_t position) {
  return pooled::append_history(f,state,actual_normalized,p.nextn,position);
}
struct Target43Result final {
  pooled::Mla mla;types::Projection query;fp::Norm query_norm;target43::Ports ports;moe::Moe moe;Value mhc;
  template<class Visitor>void visit_values(Visitor &&v) {
    mla.visit_values(v);query.visit_values(v);query_norm.visit_values(v);ports.visit_values(v);moe.visit_values(v);v(mhc);
  }
};
inline Target43Result append_target43(sem::Fragment &f,State state,Value actual_four_streams,const Parameters &p,
    const std::array<Value,3> &banks,Value map,std::int32_t position) {
  Target43Result r;r.ports.attention_hc=hc_pre(f,actual_four_streams,p.target43_hc[0]);
  const auto input=r.ports.attention_hc.norm.stored;const auto &b=p.target43.nextn.block;
  r.query=leaf::linear(f,input,b.query_a);r.query_norm=fp::norm(f,r.query.output,b.query_norm,p.geometry.rms_epsilon);
  r.mla=pooled::append_mla(f,state,input,r.query_norm.output,p.target43,true,position,0,kIndexPolicy);
  r.ports.attention_post=hc::hc_post(f,r.ports.attention_hc,actual_four_streams,r.mla.output.output,c::DataType::kFloat32);
  r.ports.ffn_hc=hc_pre(f,r.ports.attention_post.stored,p.target43_hc[1]);
  r.moe=feed_forward(f,r.ports.ffn_hc.norm.stored,b,p,0,banks,map);
  r.ports.ffn_post=hc::hc_post(f,r.ports.ffn_hc,r.ports.attention_post.stored,r.moe.output,c::DataType::kFloat32);
  r.mhc=r.ports.ffn_post.visible;return r;
}
inline target44::LayerResult<moe::Moe> append_target44(sem::Fragment &f,Value actual_four_streams,
    Value history,Value matrix,const Parameters &p,const std::array<Value,3> &banks,Value map) {
  require(f,history,sem::dense({3,24576}));require(f,matrix,sem::dense({1,64,128,128}));
  target44::LayerResult<moe::Moe> r;auto &a=r.ports;const auto &v=p.target44;
  const auto role=[&](roles44::Role id){return v[std::size_t(id)];};
  const HcParameters attention{role(roles44::Role::kHcAttnFn),role(roles44::Role::kHcAttnScale),role(roles44::Role::kHcAttnBase),role(roles44::Role::kInputNorm)};
  const HcParameters ffn{role(roles44::Role::kHcFfnFn),role(roles44::Role::kHcFfnScale),role(roles44::Role::kHcFfnBase),role(roles44::Role::kPostNorm)};
  a.attention_hc=hc_pre(f,actual_four_streams,attention);
  a.kda=target44::kda_with_projection(f,a.attention_hc.norm.stored,history,matrix,v,c::DataType::kFloat32,
      target44::QueryKeyNormalization::kL2ForScaledGdn,target44::scaled_recurrent,project,
      target44::DecayParameter::kNegativeExpRate,p.gate_lower_bound);
  a.attention_post=hc::hc_post(f,a.attention_hc,actual_four_streams,a.kda.projections[std::size_t(target44::Projection::kOutput)].stored,c::DataType::kFloat32);
  a.ffn_hc=hc_pre(f,a.attention_post.stored,ffn);r.moe=feed_forward(f,a.ffn_hc.norm.stored,p.target44_moe,p,1,banks,map);
  a.ffn_post=hc::hc_post(f,a.ffn_hc,a.attention_post.stored,r.moe.output,c::DataType::kFloat32);
  a.contract_raw=hc::reduce(f,a.ffn_post.visible,1,c::ReductionKind::kMean,false);a.contract_stored=a.contract_raw;
  a.model_norm=hc::norm(f,a.contract_stored,role(roles44::Role::kModelNorm),c::DataType::kFloat32);
  r.history=a.kda.next_history;r.state=a.kda.state_output;r.mhc=a.ffn_post.visible;r.tap=a.model_norm.visible;return r;
}
} // namespace glm_gguf_flash_fragment
