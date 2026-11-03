#include "glm_gguf_flash_fragment.h"

namespace glm_gguf_flash_fragment {
void require_geometry(const types::Geometry &g) {
  types::require_geometry(g);
  if(g.hidden!=4096||g.query_rank!=1536||g.heads!=64||g.nope!=256||g.rope!=0||
      g.latent!=512||g.value!=256||g.index_depth!=128||g.index_heads!=32||
      g.index_pool!=4||g.index_topk!=2048||g.rms_epsilon!=1.e-5F||
      g.index_epsilon!=1.e-6F||g.attention_scale!=1.F/16.F)
    throw std::runtime_error("converted pooled MLA geometry is outside the current shared pure algorithm");
}
}

namespace glm_gguf_flash_fragment {
void require(sem::Fragment &f,Value value,const c::TensorDesc &expected) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=expected)
    throw std::runtime_error("converted Flash actual predecessor/role descriptor differs");
}
}

namespace glm_gguf_flash_fragment {
std::array<Value,kStateEntries> state_values(const State &s) {
  return {s.main,s.raw_key,s.raw_gate,s.pool,s.hidden,s.query,s.pending,s.mla_cursor,
      s.index_cursor,s.pool_count,s.ring_offset,s.tail_valid,s.tail_start,s.selection,s.journal,s.lag};
}
}

namespace glm_gguf_flash_fragment {
State bind_state(sem::Fragment &f,const std::array<Value,kStateEntries> &v) {
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
}

namespace glm_gguf_flash_fragment {
hc::HcPorts hc_pre(sem::Fragment &f,Value residual,const HcParameters &p) {
  const auto packed=[](sem::Fragment &graph,Value actual,Value weight){return leaf::linear(graph,actual,weight).output;};
  return hc::hc_pre_with_projection(f,residual,p.fn,p.scale,p.base,p.norm,c::DataType::kFloat32,
      hc::ProjectionOrder::kNormalizeInput,packed);
}
}

namespace glm_gguf_flash_fragment {
target44::ProjectionPorts project(sem::Fragment &f,Value actual,Value weight,c::DataType storage) {
  if(storage!=c::DataType::kFloat32)throw std::runtime_error("converted packed projection has an explicit F32 result/store policy");
  const auto p=leaf::linear(f,actual,weight);return {p.input,p.output,p.output};
}
}

namespace glm_gguf_flash_fragment {
moe::Moe feed_forward(sem::Fragment &f,Value actual,const types::Block &block,
    const Parameters &p,std::size_t stage,const std::array<Value,3> &banks,Value map) {
  if(stage>=p.moe_geometry.size())throw std::runtime_error("converted original MoE stage is outside the explicit contract");
  return moe::append_moe_with_limits(f,actual,block,banks,map,p.moe_geometry[stage],
      p.shared_swiglu_limit[stage],moe::ExpertFold::kSelectedOrder);
}
}

namespace glm_gguf_flash_fragment {
NextnResult append_nextn(sem::Fragment &f,State state,Value actual_hidden,const Parameters &p,
    const std::array<Value,3> &banks,Value map,bool fresh,std::int32_t position,std::int32_t carried_tail) {
  const auto moe_builder=[&](sem::Fragment &graph,Value actual) {
    return feed_forward(graph,actual,p.nextn.nextn.block,p,2,banks,map);
  };
  return pooled::append_with_moe(f,state,actual_hidden,p.nextn,fresh,position,carried_tail,p.geometry,kIndexPolicy,moe_builder);
}
}

namespace glm_gguf_flash_fragment {
pooled::History append_history(sem::Fragment &f,State &state,Value actual_normalized,const Parameters &p,std::int32_t position) {
  return pooled::append_history(f,state,actual_normalized,p.nextn,position);
}
}

namespace glm_gguf_flash_fragment {
Target43Result append_target43(sem::Fragment &f,State state,Value actual_four_streams,const Parameters &p,
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
}

namespace glm_gguf_flash_fragment {
target44::LayerResult<moe::Moe> append_target44(sem::Fragment &f,Value actual_four_streams,
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
}
