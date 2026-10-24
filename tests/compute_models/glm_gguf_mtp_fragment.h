#pragma once

#include "glm_gguf_f32_cache_fragment.h"
#include "glm_original_moe_fragment.h"

// Converted packed weights with the explicit all-F32 runtime chosen for this
// consumer. These are pure ordinary Graph leaves; index algorithm, absolute
// IDs and accepted-journal semantics do not change with the cache encoding.
namespace glm_gguf_mtp_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
namespace core=d5_moe_semantic_fragments;
namespace types=glm_gguf_model_types;
namespace compute=glm_gguf_compute_fragment;
namespace cache=glm_gguf_f32_cache_fragment;
namespace leaf=glm_gguf_projection_fragment;
using Value=sem::ValueId;
using Boundary=glm_original_moe_fragment::Boundary;
struct Moe final {
  Value input,logits,scores,selection,ids,selected_scores,sum,denominator,weights,local_ids,map;
  std::array<Value,3> banks,shared_weights;
  std::array<Value,4> routed,shared; // gate, up, SiLU(gate)*up, down
  Value weighted,routed_sum,output;
  Boundary boundary;
  template<class Visitor> void visit_values(Visitor &&v) {
    for(auto *x:{&input,&logits,&scores,&selection,&ids,&selected_scores,&sum,&denominator,&weights,&local_ids,&map,
        &weighted,&routed_sum,&output})v(*x);
    for(auto &x:routed)v(x);for(auto &x:shared)v(x);
  }
};
enum class ExpertFold {kReduction,kSelectedOrder};
inline Moe append_moe_with_limits(sem::Fragment &f,Value input,const types::Block &p,
    const std::array<Value,3> &banks,Value map,const glm_moe_geometry::Geometry &g,
    std::optional<float> shared_limit,ExpertFold fold) {
  glm_moe_geometry::require(g);
  if((shared_limit&&(!std::isfinite(*shared_limit)||!(*shared_limit>0.F)))||
      (fold!=ExpertFold::kReduction&&fold!=ExpertFold::kSelectedOrder))
    throw std::runtime_error("converted MoE shared clamp/fold contract is unsupported");
  cache::require(f,input,sem::dense({1,g.hidden}));
  cache::require(f,map,sem::dense({g.experts},c::DataType::kInt32));
  for(std::size_t leg=0;leg<3;++leg) {
    const auto &d=f.description(banks[leg]);const auto codec=c::gguf_quantized_type(d.storage_format());
    if(!codec||d!=c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,
        std::int64_t(g.slots),leg==2?g.hidden:g.intermediate,leg==2?g.intermediate:g.hidden).value())
      throw std::runtime_error("converted GLM needs the actual packed E8 expert descriptors");
  }
  Moe r;r.input=input;r.map=map;r.banks=banks;r.shared_weights={p.shared_gate,p.shared_up,p.shared_down};
  r.logits=leaf::f32_linear(f,input,p.router).output;
  r.scores=core::unary(f,c::UnaryKind::kSigmoid,r.logits);
  r.selection=core::binary(f,c::BinaryKind::kAdd,r.scores,p.bias);
  const auto top=f.append(c::Operation(c::TopK{.count=g.slots,.stable=true}),{r.selection},2);
  r.ids=top[1];r.selected_scores=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),{r.scores,r.ids});
  r.sum=glm_original_moe_fragment::sum(f,r.selected_scores,-1);
  // Fixed converted GGUF norm_w policy. This is deliberately not the
  // official safetensor path's sum+1e-20 normalization.
  r.denominator=core::binary(f,c::BinaryKind::kMaximum,r.sum,core::filled(f,sem::dense({1,1}),0x1p-14F));
  const auto normalized=core::binary(f,c::BinaryKind::kDivide,r.selected_scores,r.denominator);
  r.weights=core::binary(f,c::BinaryKind::kMultiply,normalized,
      core::filled(f,sem::dense({1,std::int64_t(g.slots)}),g.route_scale));
  r.boundary={input,r.ids,r.weights,banks,map};
  r.local_ids=f.one(c::Operation(c::Gather{}),{map,r.ids});
  const auto grouped=[&](Value actual,std::size_t leg) {
    return f.one(c::model_ops::grouped_matmul(),{banks[leg],actual,r.local_ids});
  };
  const auto activate=[&](Value gate,Value up,std::optional<float> limit) {
    if(limit) {
      gate=f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(),*limit),{gate});
      up=f.one(c::model_ops::clamp(-*limit,*limit),{up});
    }
    return core::binary(f,c::BinaryKind::kMultiply,core::unary(f,c::UnaryKind::kSilu,gate),up);
  };
  const auto row=compute::reshape(f,input,{1,1,g.hidden});
  r.routed[0]=grouped(row,0);r.routed[1]=grouped(row,1);
  r.routed[2]=activate(r.routed[0],r.routed[1],g.swiglu_limit);r.routed[3]=grouped(r.routed[2],2);
  r.weighted=core::binary(f,c::BinaryKind::kMultiply,r.routed[3],
      compute::reshape(f,r.weights,{1,std::int64_t(g.slots),1}));
  if(fold==ExpertFold::kReduction) {
    r.routed_sum=compute::reshape(f,glm_original_moe_fragment::sum(f,r.weighted,1),{1,g.hidden});
  } else {
    // Fixed GGUF build_moe_ffn adds actual selected-slot views in order.
    const auto selected_row=[&](std::int64_t slot){return compute::reshape(f,compute::slice(f,r.weighted,1,slot,1),{1,g.hidden});};
    r.routed_sum=selected_row(0);
    for(std::size_t slot=1;slot<g.slots;++slot)
      r.routed_sum=core::binary(f,c::BinaryKind::kAdd,r.routed_sum,selected_row(std::int64_t(slot)));
  }
  r.shared[0]=leaf::linear(f,input,p.shared_gate).output;r.shared[1]=leaf::linear(f,input,p.shared_up).output;
  r.shared[2]=activate(r.shared[0],r.shared[1],shared_limit);r.shared[3]=leaf::linear(f,r.shared[2],p.shared_down).output;
  r.output=core::binary(f,c::BinaryKind::kAdd,r.routed_sum,r.shared[3]);return r;
}
inline Moe append_moe(sem::Fragment &f,Value input,const types::Block &p,
    const std::array<Value,3> &banks,Value map,const glm_moe_geometry::Geometry &g) {
  return append_moe_with_limits(f,input,p,banks,map,g,g.swiglu_limit,ExpertFold::kReduction);
}
struct Attention final {
  compute::Absorption query;
  types::Projection kv;
  compute::Norm kv_norm,post_norm;
  std::optional<Value> query_rotated,key_rotated;
  Value query_full,row_stored,selected_full,selected_value,attention_raw,residual_sum;
  compute::Output output;
  template<class Visitor> void visit_values(Visitor &&v) {
    query.visit_values(v);kv.visit_values(v);kv_norm.visit_values(v);post_norm.visit_values(v);output.visit_values(v);
    if(query_rotated)v(*query_rotated);if(key_rotated)v(*key_rotated);
    for(auto *x:{&query_full,&row_stored,&selected_full,&selected_value,&attention_raw,&residual_sum})v(*x);
  }
};
struct State final {Value main,keys,selection,positions;};
struct Ports final {
  compute::Prefix prefix;Attention attention;
  std::optional<cache::IndexHistory> history;
  std::optional<cache::Selection> index;
  std::optional<types::Projection> index_query,index_weights;
  std::optional<Value> index_rotated;
  compute::Norm shared_norm;
  Value positions,combined,logits,top_values,pending;
  template<class Visitor> void visit_values(Visitor &&v) {
    prefix.visit_values(v);attention.visit_values(v);if(history)history->visit_values(v);
    if(index)index->visit_values(v);if(index_query)index_query->visit_values(v);
    if(index_weights)index_weights->visit_values(v);if(index_rotated)v(*index_rotated);shared_norm.visit_values(v);
    for(auto *x:{&positions,&combined,&logits,&top_values,&pending})v(*x);
  }
};
struct Result final {State state;Value hidden,pending;Ports ports;Moe feed_forward;};
struct Mla final {Value main;Attention ports;};
template<class Project>
Mla append_mla_prefix(sem::Fragment &f,Value normalized,Value query_norm,Value old_main,Value positions,
    const types::Block &p,const types::Geometry &g,Value frequencies,const Project &project) {
  Mla r;auto &a=r.ports;
  a.query=compute::append_f32_absorption(f,query_norm,p,g,project);
  if(g.rope) {
    if(!a.query.rope_suffix)throw std::runtime_error("rotary MLA needs its actual KPE suffix");
    a.query_rotated=cache::rotate(f,*a.query.rope_suffix,positions,frequencies,g);
    a.query_full=compute::reshape(f,f.one(c::Operation(c::Concat{.axis=2}),
        {a.query.absorbed.output,*a.query_rotated}),{1,1,g.heads,g.latent+g.rope});
  } else a.query_full=compute::reshape(f,a.query.absorbed.output,{1,1,g.heads,g.latent});
  a.kv=project.linear(f,normalized,p.kv_a);
  const auto latent=compute::copy(f,compute::slice(f,a.kv.output,1,0,g.latent));
  a.kv_norm=compute::norm(f,latent,p.kv_norm,g.rms_epsilon);
  if(g.rope) {
    const auto suffix=compute::reshape(f,compute::copy(f,compute::slice(f,a.kv.output,1,g.latent,g.rope)),{1,1,g.rope});
    a.key_rotated=compute::reshape(f,cache::rotate(f,suffix,positions,frequencies,g),{1,g.rope});
  }
  const auto write=cache::append_main_row(f,old_main,a.kv_norm.output,a.key_rotated,positions,g);
  a.row_stored=write.row;r.main=write.main;return r;
}
template<class Project>
void append_mla_suffix(sem::Fragment &f,Mla &r,Value ids,Value residual,
    const types::Block &p,const types::Geometry &g,const Project &project) {
  auto &a=r.ports;const auto selected=f.description(ids).dimensions()[0];
  if(selected<=0||selected>g.index_topk)throw std::runtime_error("converted MLA selected extent differs");
  cache::require(f,ids,sem::dense({selected},c::DataType::kInt32));
  a.selected_full=f.one(c::Operation(c::Gather{}),{r.main,ids});
  a.selected_value=compute::copy(f,compute::slice(f,a.selected_full,1,0,g.latent));
  a.attention_raw=f.one(c::Operation(c::ScaledDotProductAttention{.scale=g.attention_scale,.causal=false,
      .accumulation_type=c::DataType::kFloat32}),{a.query_full,
      compute::reshape(f,a.selected_full,{1,selected,1,g.latent+g.rope}),
      compute::reshape(f,a.selected_value,{1,selected,1,g.latent})});
  a.output=compute::append_f32_output(f,compute::reshape(f,a.attention_raw,{1,g.heads,g.latent}),p,g,project);
  a.residual_sum=core::binary(f,c::BinaryKind::kAdd,residual,a.output.output.output);
  a.post_norm=compute::norm(f,a.residual_sum,p.post_norm,g.rms_epsilon);
}
template<class Project>
Result append_step(sem::Fragment &f,const compute::Prefix &prefix,State state,const types::NextnParameters &p,
    const types::Geometry &g,const glm_moe_geometry::Geometry &moe_geometry,const std::array<Value,3> &banks,
    Value map,Value frequencies,std::int64_t visible,std::int64_t selected,bool fresh,const Project &project) {
  Result r;r.state=state;r.ports.prefix=prefix;r.ports.positions=state.positions;
  auto mla=append_mla_prefix(f,prefix.input_norm.output,prefix.query_norm.output,state.main,state.positions,p.block,g,frequencies,project);
  r.state.main=mla.main;
  if(fresh) {
    r.ports.history=cache::append_index_history(f,prefix.input_norm.output,state.positions,state.keys,p.block,g,frequencies,project);
    r.state.keys=r.ports.history->keys;
    r.ports.index_query=project.linear(f,prefix.query_norm.output,p.block.index_query);
    r.ports.index_rotated=compute::reshape(f,cache::rotate(f,
        compute::reshape(f,r.ports.index_query->output,{1,g.index_heads,g.index_depth}),state.positions,frequencies,g),{g.index_heads,g.index_depth});
    r.ports.index_weights=project.linear(f,prefix.input_norm.output,p.block.index_weights);
    r.ports.index=cache::select(f,*r.ports.index_rotated,r.state.keys,r.ports.index_weights->output,g,visible,selected);
    r.state.selection=r.ports.index->ids;
  }
  append_mla_suffix(f,mla,r.state.selection,prefix.eh.output,p.block,g,project);r.ports.attention=mla.ports;
  r.feed_forward=append_moe(f,mla.ports.post_norm.output,p.block,banks,map,moe_geometry);
  r.ports.combined=core::binary(f,c::BinaryKind::kAdd,mla.ports.residual_sum,r.feed_forward.output);
  r.ports.shared_norm=compute::norm(f,r.ports.combined,p.shared_norm,g.rms_epsilon);r.hidden=r.ports.shared_norm.output;
  r.ports.logits=project.linear(f,r.hidden,p.head).output;
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{r.ports.logits},2);
  r.ports.top_values=top[0];r.pending=compute::copy(f,compute::reshape(f,top[1],{1}));r.ports.pending=r.pending;return r;
}
struct TargetResult final {
  Value hidden,residual,position,selection,input_sum,query_input,post_ffn,main,cursor,normalized_tap;
  compute::Norm input_norm,query_norm,final_norm;types::Projection query;Attention attention;Moe feed_forward;
  template<class Visitor> void visit_values(Visitor &&v) {
    for(auto *x:{&hidden,&residual,&position,&selection,&input_sum,&query_input,&post_ffn,&main,&cursor,&normalized_tap})v(*x);
    input_norm.visit_values(v);query_norm.visit_values(v);final_norm.visit_values(v);query.visit_values(v);
    attention.visit_values(v);feed_forward.visit_values(v);
  }
};
template<class Project>
TargetResult append_target(sem::Fragment &f,Value hidden,Value residual,Value old_main,Value ids,Value cursor,
    const types::Block &p,Value final_gamma,const types::Geometry &g,const glm_moe_geometry::Geometry &moe_geometry,
    const std::array<Value,3> &banks,Value map,Value frequencies,const Project &project) {
  TargetResult r;r.hidden=hidden;r.residual=residual;r.position=cursor;r.selection=ids;
  r.input_sum=core::binary(f,c::BinaryKind::kAdd,hidden,residual);
  r.input_norm=compute::norm(f,r.input_sum,p.input_norm,g.rms_epsilon);
  r.query_input=r.input_norm.output;r.query=project.linear(f,r.query_input,p.query_a);
  r.query_norm=compute::norm(f,r.query.output,p.query_norm,g.rms_epsilon);
  auto mla=append_mla_prefix(f,r.input_norm.output,r.query_norm.output,old_main,cursor,p,g,frequencies,project);
  append_mla_suffix(f,mla,ids,r.input_sum,p,g,project);r.main=mla.main;r.attention=mla.ports;
  r.feed_forward=append_moe(f,r.attention.post_norm.output,p,banks,map,moe_geometry);
  r.post_ffn=core::binary(f,c::BinaryKind::kAdd,r.attention.residual_sum,r.feed_forward.output);
  r.final_norm=compute::norm(f,r.post_ffn,final_gamma,g.rms_epsilon);r.normalized_tap=r.final_norm.output;
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto scan=f.one(c::Operation(c::Scan{.axis=0}),{f.one(c::Operation(c::Concat{.axis=0}),{cursor,one})});
  r.cursor=f.one(c::Operation(c::Gather{}),{scan,one});return r;
}
} // namespace glm_gguf_mtp_fragment
