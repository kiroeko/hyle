#pragma once
#include <numeric>
#include "glm_gguf_compute_fragment.h"
#include "glm_mlx_original_roles.h"
#include "glm_mlx_affine_projection_fragment.h"
#include "glm_original_moe_fragment.h"

// Original MLX affine/F16 metadata path, with an explicit F32 activation and
// plain F32 pooled-state contract. No quantizer, BF16 intermediate store,
// provider choice, loader or executor. Ordinary public mixed DenseLinearF32
// rank3 support is a prerequisite for the two original BF16 KV_b head banks.
namespace glm_mlx_mtp_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;namespace c=hyle::compute;
namespace core=d5_moe_semantic_fragments;namespace fp=glm_gguf_compute_fragment;
namespace types=glm_gguf_model_types;namespace affine=glm_mlx_affine_projection_fragment;
namespace own=glm_mlx_original_roles;
using Value=sem::ValueId;
inline constexpr types::Geometry kGeometry{4096,1536,64,256,0,512,256,154880,128,32,4,2048,1.e-5F,1.e-6F,1.F/16.F};
inline Value cast(sem::Fragment &f,Value x,c::DataType type) {
  return f.description(x).data_type()==type?x:f.one(c::Operation(c::Cast{.output_type=type}),{x});
}
inline Value permute(sem::Fragment &f,Value x,std::initializer_list<std::int32_t> axes) {
  const auto p=c::View::permute({axes.begin(),axes.size()});if(!p)throw std::runtime_error("MLX pure permutation differs");return f.one(c::Operation(*p),{x});
}
inline Value inc(sem::Fragment &f,Value x,std::int32_t amount=1) {
  if(f.description(x)!=sem::dense({1},c::DataType::kInt32))throw std::runtime_error("MLX cursor must be actual I32[1]");
  std::vector<std::int32_t> table(8192);
  for(std::size_t i=0;i<table.size();++i)table[i]=std::int32_t(i)+amount;
  return f.one(c::Operation(c::Gather{}),{f.constant(sem::dense({std::int64_t(table.size())},c::DataType::kInt32),sem::bytes(table)),x});
}
struct Projects final {
  std::vector<affine::Parameters> quantized;
  types::Projection linear(sem::Fragment &f,Value actual,Value weight)const {
    for(const auto &p:quantized)if(p.codes.ordinal()==weight.ordinal()) {
      const auto y=affine::linear(f,actual,p);return {actual,weight,y.output};
    }
    if(f.description(weight).data_type()==c::DataType::kBFloat16)
      return {actual,weight,f.one(c::model_ops::dense_linear_f32(),{actual,weight})};
    if(f.description(weight).data_type()==c::DataType::kFloat32)return glm_gguf_projection_fragment::f32_linear(f,actual,weight);
    throw std::runtime_error("MLX source projection has neither original BF16 nor a complete U32/F16/F16 triple");
  }
  types::Projection per_head(sem::Fragment &f,Value input,Value weight)const {
    const auto &a=f.description(input),&w=f.description(weight);
    if(a.rank()!=3||a.dimensions()[1]!=64||a.data_type()!=c::DataType::kFloat32||
        w.rank()!=3||w.dimensions()[0]!=64||w.data_type()!=c::DataType::kBFloat16||
        a.dimensions()[2]!=w.dimensions()[2])throw std::runtime_error("MLX mixed KV_b requires actual F32 A and original64-head BF16 W");
    const auto batches=fp::copy(f,permute(f,input,{1,0,2}));
    const auto raw=f.one(c::model_ops::dense_linear_f32(),{batches,weight});
    return {input,weight,fp::copy(f,permute(f,raw,{1,0,2}))};
  }
};
struct Parameters final {types::NextnParameters nextn;Value ape,index_gate;std::array<affine::Parameters,3> shared;Projects project;};
template<class Lookup>
inline Parameters moe_parameters(sem::Fragment &f,const Lookup &value,std::string_view prefix) {
  (void)f;Parameters r;const auto raw=[&](std::string_view suffix){return value(std::string(prefix)+std::string(suffix));};
  r.nextn.block.router=raw("mlp.gate.weight");r.nextn.block.bias=raw("mlp.gate.e_score_correction_bias");
  for(std::size_t leg=0;leg<3;++leg){const auto stem="mlp.shared_experts."+std::string(own::kLegs[leg])+"_proj.";
    r.shared[leg]={raw(stem+"weight"),raw(stem+"scales"),raw(stem+"biases"),64};r.project.quantized.push_back(r.shared[leg]);}
  return r;
}
template<class Lookup>
inline Parameters block_parameters(sem::Fragment &f,const Lookup &value,std::string_view prefix,
    Value key_bf16,Value value_bf16) {
  auto r=moe_parameters(f,value,prefix);const auto name=[&](std::string_view suffix){return std::string(prefix)+std::string(suffix);};
  const auto raw=[&](std::string_view suffix){return value(name(suffix));};
  const auto gamma=[&](std::string_view suffix){return cast(f,raw(suffix),c::DataType::kFloat32);};
  const auto affine_role=[&](std::string_view stem) {
    affine::Parameters p{raw(std::string(stem)+"weight"),raw(std::string(stem)+"scales"),raw(std::string(stem)+"biases"),64};
    r.project.quantized.push_back(p);return p;
  };
  auto &b=r.nextn.block;
  b.input_norm=gamma("input_layernorm.weight");b.query_norm=gamma("self_attn.q_a_layernorm.weight");
  b.kv_norm=gamma("self_attn.kv_a_layernorm.weight");b.post_norm=gamma("post_attention_layernorm.weight");
  b.query_a=affine_role("self_attn.q_a_proj.").codes;b.query_b=affine_role("self_attn.q_b_proj.").codes;
  b.kv_a=affine_role("self_attn.kv_a_proj_with_mqa.").codes;b.output=affine_role("self_attn.o_proj.").codes;
  b.key=key_bf16;b.value=value_bf16;b.router=raw("mlp.gate.weight");b.bias=raw("mlp.gate.e_score_correction_bias");
  b.index_key=raw("self_attn.indexer.wk.weight");b.index_query=raw("self_attn.indexer.wq_b.weight");
  b.index_weights=raw("self_attn.indexer.weights_proj.weight");b.index_norm=gamma("self_attn.indexer.k_norm.weight");b.index_bias=gamma("self_attn.indexer.k_norm.bias");
  r.ape=cast(f,raw("self_attn.indexer.index_kpool_compress_ape"),c::DataType::kFloat32);
  r.index_gate=raw("self_attn.indexer.index_kpool_compress_gate");
  return r;
}
template<class Lookup>
inline Parameters parameters(sem::Fragment &f,const Lookup &value,Value key_bf16,Value value_bf16) {
  auto r=block_parameters(f,value,own::kPrefix,key_bf16,value_bf16);
  const auto raw=[&](std::string_view suffix){return value(std::string(own::kPrefix)+std::string(suffix));};
  auto &p=r.nextn;p.eh=raw("eh_proj.weight");
  p.hidden_norm=cast(f,raw("hnorm.weight"),c::DataType::kFloat32);
  p.embedding_norm=cast(f,raw("enorm.weight"),c::DataType::kFloat32);
  p.shared_norm=cast(f,raw("shared_head.norm.weight"),c::DataType::kFloat32);
  p.embedding=value("model.language_model.embed_tokens.weight");p.head=value("lm_head.weight");
  return r;
}
// One ordinary cold Graph. Original BF16 KV_b remains BF16; only its two
// compact reordered views are retained by caller leases after this Graph.
inline std::array<Value,2> prepare_kv_b(sem::Fragment &f,Value original) {
  if(f.description(original)!=sem::dense({32768,512},c::DataType::kBFloat16))throw std::runtime_error("Q32 original KV_b physical shape differs");
  const auto heads=fp::reshape(f,original,{64,512,512});
  return {fp::copy(f,permute(f,fp::slice(f,heads,1,0,256),{0,2,1})),
          fp::copy(f,fp::slice(f,heads,1,256,256))};
}
struct Boundary final {Value input,ids,weights;std::vector<Value> banks;Value map;};
struct Moe final {Value output;glm_original_moe_fragment::Ports ports;Boundary boundary;
  template<class Visitor>void visit_values(Visitor &&visitor){ports.visit_values(visitor);}
};
inline Moe append_moe(sem::Fragment &f,Value actual,const Parameters &p,std::span<const Value> banks,Value map) {
  if(f.description(actual)!=sem::dense({1,4096})||banks.size()!=9||f.description(map)!=sem::dense({288},c::DataType::kInt32))
    throw std::runtime_error("MLX MoE requires F32 actual predecessor, original9 planes and actual288-map");
  for(std::size_t i=0;i<9;++i)if(f.description(banks[i])!=own::resident_description(own::Leg(i/3),own::Plane(i%3)))
    throw std::runtime_error("MLX compact original plane type/shape differs");
  Moe r;auto &a=r.ports;a.input=actual;a.map=map;
  a.logits=p.project.linear(f,actual,p.nextn.block.router).output;
  a.scores=core::unary(f,c::UnaryKind::kSigmoid,a.logits);a.selection=core::binary(f,c::BinaryKind::kAdd,a.scores,p.nextn.block.bias);
  const auto top=f.append(c::Operation(c::TopK{.count=8,.stable=true}),{a.selection},2);a.ids=top[1];
  a.selected_scores=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),{a.scores,a.ids});
  a.sum=glm_original_moe_fragment::sum(f,a.selected_scores,-1);
  a.denominator=core::binary(f,c::BinaryKind::kAdd,a.sum,core::filled(f,sem::dense({1,1}),1.e-20F));
  a.normalized=core::binary(f,c::BinaryKind::kDivide,a.selected_scores,a.denominator);
  a.weights=core::binary(f,c::BinaryKind::kMultiply,a.normalized,core::filled(f,sem::dense({1,8}),2.5F));
  r.boundary={actual,a.ids,a.weights,std::vector<Value>(banks.begin(),banks.end()),map};
  a.local_ids=f.one(c::Operation(c::Gather{}),{map,a.ids});
  const auto activation=[&](auto &branch) {
    branch.gate_clipped=f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(),10.F),{branch.gate_stored});
    branch.up_clipped=f.one(c::model_ops::clamp(-10.F,10.F),{branch.up_stored});
    branch.activation_raw=core::binary(f,c::BinaryKind::kMultiply,core::unary(f,c::UnaryKind::kSilu,branch.gate_clipped),branch.up_clipped);
    branch.activation_stored=branch.activation_raw; // Actual F32 store, not BF16.
  };
  const auto grouped=[&](Value input,std::size_t leg) {
    return affine::grouped(f,input,{banks[3*leg],banks[3*leg+1],banks[3*leg+2],64},a.local_ids).output;
  };
  const auto row=fp::reshape(f,actual,{1,1,4096});
  const auto gate=grouped(row,0),up=grouped(row,1);
  a.routed.gate_raw=gate;a.routed.gate_stored=gate;a.routed.up_raw=up;a.routed.up_stored=up;activation(a.routed);
  const auto down=grouped(a.routed.activation_stored,2);a.routed.down_raw=down;
  a.weighted_raw=core::binary(f,c::BinaryKind::kMultiply,down,fp::reshape(f,a.weights,{1,8,1}));a.weighted_stored=a.weighted_raw;
  a.routed_sum_raw=fp::reshape(f,glm_original_moe_fragment::sum(f,a.weighted_stored,1),{1,4096});a.routed_sum_stored=a.routed_sum_raw;
  const auto sg=affine::linear(f,actual,p.shared[0]).output,su=affine::linear(f,actual,p.shared[1]).output;
  a.shared.gate_raw=sg;a.shared.gate_stored=sg;a.shared.up_raw=su;a.shared.up_stored=su;activation(a.shared);
  const auto sd=affine::linear(f,a.shared.activation_stored,p.shared[2]).output;a.shared.down_raw=sd;a.shared_stored=sd;
  a.merge_raw=core::binary(f,c::BinaryKind::kAdd,a.routed_sum_stored,a.shared_stored);a.merge_stored=a.merge_raw;r.output=a.merge_stored;return r;
}
// The common Flash controller keeps these logical entries. A cold layout must
// OMIT pool-scale physically for this plain F32 representation. No unit scale,
// fake scale resource or Scatter1 is permitted.
struct State final {Value main,raw_key,raw_gate,pool,hidden,query,pending,mla_cursor,index_cursor,pool_count,ring_offset,tail_valid,tail_start,selection,journal,lag;};
struct History final {
  types::Projection key,gate;Value normalized,keys,gates,pool;std::optional<Value> logits,probabilities,weights,weighted,pooled;
  template<class Visitor>void visit_values(Visitor &&v) {
    key.visit_values(v);gate.visit_values(v);for(auto *p:{&normalized,&keys,&gates,&pool})v(*p);for(auto *p:{&logits,&probabilities,&weights,&weighted,&pooled})if(*p)v(**p);
  }
};
template<class ParameterSet>
inline History append_history(sem::Fragment &f,State &s,Value nx,const ParameterSet &p,std::int32_t position) {
  History r;r.key=p.project.linear(f,nx,p.nextn.block.index_key);r.gate=p.project.linear(f,nx,p.index_gate);
  r.normalized=f.one(c::Operation(c::LayerNorm{.epsilon=1.e-6F}),{r.key.output,p.nextn.block.index_norm,p.nextn.block.index_bias});
  r.keys=f.one(c::Operation(c::Scatter{}),{s.raw_key,s.ring_offset,r.normalized});r.gates=f.one(c::Operation(c::Scatter{}),{s.raw_gate,s.ring_offset,r.gate.output});
  r.pool=s.pool;
  if(position%4==3) {
    std::vector<std::int32_t> rows(4);for(std::int32_t i=0;i<4;++i)rows[std::size_t(i)]=(position-3+i)%6;
    const auto ids=f.constant(sem::dense({1,4},c::DataType::kInt32),sem::bytes(rows));
    const auto keys=f.one(c::Operation(c::Gather{}),{r.keys,ids}),gates=f.one(c::Operation(c::Gather{}),{r.gates,ids});
    r.logits=core::binary(f,c::BinaryKind::kAdd,gates,fp::reshape(f,p.ape,{1,4,128}));
    const auto flat=fp::reshape(f,fp::copy(f,permute(f,*r.logits,{0,2,1})),{128,4});
    r.probabilities=f.one(c::Operation(c::Softmax{}),{flat});
    r.weights=fp::copy(f,permute(f,fp::reshape(f,*r.probabilities,{1,128,4}),{0,2,1}));
    r.weighted=core::binary(f,c::BinaryKind::kMultiply,*r.weights,keys);
    r.pooled=fp::reshape(f,glm_original_moe_fragment::sum(f,*r.weighted,1),{1,128});
    r.pool=f.one(c::Operation(c::Scatter{}),{s.pool,s.pool_count,*r.pooled});s.pool_count=inc(f,s.pool_count);s.tail_start=inc(f,s.tail_start,4);
  }
  s.raw_key=r.keys;s.raw_gate=r.gates;s.pool=r.pool;s.index_cursor=inc(f,s.index_cursor);
  std::vector<std::int32_t> ring(6),tail(4);for(std::int32_t i=0;i<6;++i)ring[std::size_t(i)]=(i+1)%6;
  for(std::int32_t i=0;i<4;++i)tail[std::size_t(i)]=(i+1)%4;
  s.ring_offset=f.one(c::Operation(c::Gather{}),{f.constant(sem::dense({6},c::DataType::kInt32),sem::bytes(ring)),s.ring_offset});
  s.tail_valid=f.one(c::Operation(c::Gather{}),{f.constant(sem::dense({4},c::DataType::kInt32),sem::bytes(tail)),s.tail_valid});return r;
}
enum class IndexWeightScale {kHeadThenDepth,kCombined};
enum class PoolOrder {kChronological,kDescendingScore};
struct IndexPolicy final {
  IndexWeightScale weight_scale=IndexWeightScale::kHeadThenDepth;
  PoolOrder pool_order=PoolOrder::kChronological;
};
struct Index final {
  IndexPolicy policy;

  types::Projection query,weights;Value weights_scaled,head_weights,dots,relu,scores,pool_ids,ids;
  template<class Visitor>void visit_values(Visitor &&v) {
    query.visit_values(v);weights.visit_values(v);for(auto *p:{&weights_scaled,&head_weights,&dots,&relu,&scores,&pool_ids,&ids})v(*p);
  }
};
template<class ParameterSet>
inline Index select(sem::Fragment &f,const State &s,Value nx,Value query_norm,const ParameterSet &p,std::int32_t position,std::int32_t tail,IndexPolicy policy={}) {
  const auto pools=(position+1)/4;if(pools<512||tail<0||tail>3)throw std::runtime_error("MLX finite query requires complete512-pool plus declared tail domain");
  Index r;r.policy=policy;r.query=p.project.linear(f,query_norm,p.nextn.block.index_query);r.weights=p.project.linear(f,nx,p.nextn.block.index_weights);
  const auto keys=fp::copy(f,fp::slice(f,s.pool,0,0,pools));
  r.dots=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{fp::reshape(f,r.query.output,{32,128}),permute(f,keys,{1,0})});
  r.relu=core::binary(f,c::BinaryKind::kMaximum,r.dots,core::filled(f,sem::dense({32,pools}),0.F));
  // Preserve the two actual F32 multiplications from _project_queries and
  // _score_index_keys; their constants must not be algebraically folded.
  if(policy.weight_scale==IndexWeightScale::kHeadThenDepth) {
    r.weights_scaled=core::binary(f,c::BinaryKind::kMultiply,r.weights.output,core::filled(f,sem::dense({1,32}),float(1./std::sqrt(32.))));
    r.head_weights=core::binary(f,c::BinaryKind::kMultiply,fp::reshape(f,r.weights_scaled,{32,1}),core::filled(f,sem::dense({32,1}),float(1./std::sqrt(128.))));
  } else if(policy.weight_scale==IndexWeightScale::kCombined) {
    // Fixed GGUF glm5-next.cpp: one ggml_scale after weights projection.
    // sqrtf(32*128)=64 exactly; no earlier per-head scalar store exists.
    r.weights_scaled=core::binary(f,c::BinaryKind::kMultiply,r.weights.output,core::filled(f,sem::dense({1,32}),1.F/64.F));
    r.head_weights=fp::reshape(f,r.weights_scaled,{32,1});
  } else throw std::runtime_error("pooled index weight scaling policy is unsupported");
  r.scores=fp::reshape(f,glm_original_moe_fragment::sum(f,core::binary(f,c::BinaryKind::kMultiply,r.relu,r.head_weights),0),{pools});
  const auto top=f.append(c::Operation(c::TopK{.count=512,.stable=true}),{r.scores},2);
  // Pinned MLX restores chronological pool order before attention. Float32
  // exactly represents these finite pool IDs; a second ordinary TopK orders
  // that actual chosen set without host-created replacement IDs.
  if(policy.pool_order==PoolOrder::kChronological) {
    const auto actual=cast(f,top[1],c::DataType::kFloat32);
    const auto ordered=f.append(c::Operation(c::TopK{.count=512,.largest=false,.stable=true}),{actual},2);
    // TopK indices address the preceding selected-ID vector. Reorder those
    // original I32 IDs directly; no unsupported F32-to-I32 Cast is required.
    r.pool_ids=f.one(c::Operation(c::Gather{}),{top[1],ordered[1]});
  } else if(policy.pool_order==PoolOrder::kDescendingScore) {
    // Public stable TopK already emits the chosen set by descending actual
    // score. This replaces the peer's unordered-topk + selected-score argsort,
    // never the MLX chronological-ID reorder. Equal-score ties use this
    // runtime's explicit stable order, not a claimed peer bit identity.
    r.pool_ids=top[1];
  } else throw std::runtime_error("pooled index chosen-set order is unsupported");
  std::vector<std::int32_t> expanded(std::size_t(pools)*4);std::iota(expanded.begin(),expanded.end(),0);
  const auto table=f.constant(sem::dense({pools,4},c::DataType::kInt32),sem::bytes(expanded));
  const auto selected=fp::reshape(f,f.one(c::Operation(c::Gather{}),{table,r.pool_ids}),{2048});
  std::vector<Value> suffix;
  for(std::int32_t i=0;i<tail;++i)suffix.push_back(inc(f,s.tail_start,i));
  if(tail<3)suffix.push_back(f.constant(sem::dense({3-tail},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>(std::size_t(3-tail),-1))));
  Value end=suffix[0];for(std::size_t i=1;i<suffix.size();++i)end=f.one(c::Operation(c::Concat{.axis=0}),{end,suffix[i]});
  r.ids=f.one(c::Operation(c::Concat{.axis=0}),{selected,end});return r;
}
struct Mla final {
  State state;std::optional<History> history;std::optional<Index> index;
  types::Projection query,absorbed,kv,value,output;fp::Norm latent_norm;
  Value latent,attention;
  template<class Visitor>void visit_values(Visitor &&v) {
    if(history)history->visit_values(v);if(index)index->visit_values(v);
    query.visit_values(v);absorbed.visit_values(v);kv.visit_values(v);value.visit_values(v);output.visit_values(v);
    latent_norm.visit_values(v);v(latent);v(attention);v(state.main);v(state.pool);v(state.selection);
  }
};
template<class ParameterSet>
inline Mla append_mla(sem::Fragment &f,State state,Value actual_input_norm,Value actual_query_norm,
    const ParameterSet &p,bool fresh,std::int32_t position,std::int32_t carried_tail,IndexPolicy policy={}) {
  Mla r;r.state=state;r.state.query=actual_query_norm;
  r.query=p.project.linear(f,actual_query_norm,p.nextn.block.query_b);
  r.absorbed=p.project.per_head(f,fp::reshape(f,r.query.output,{1,64,256}),p.nextn.block.key);
  r.kv=p.project.linear(f,actual_input_norm,p.nextn.block.kv_a);
  r.latent_norm=fp::norm(f,r.kv.output,p.nextn.block.kv_norm,1.e-5F);r.latent=r.latent_norm.output;
  r.state.main=f.one(c::Operation(c::Scatter{}),{state.main,state.mla_cursor,r.latent});r.state.mla_cursor=inc(f,state.mla_cursor);
  const auto tail=fresh?(position+1)%4:carried_tail;
  if(fresh) {
    r.history=append_history(f,r.state,actual_input_norm,p,position);
    r.index=select(f,r.state,actual_input_norm,actual_query_norm,p,position,tail,policy);r.state.selection=r.index->ids;
  } else {
    r.state.selection=fp::copy(f,state.selection);
    r.state.journal=f.one(c::Operation(c::Scatter{}),{state.journal,state.lag,actual_input_norm});r.state.lag=inc(f,state.lag);
  }
  const auto ids=fp::copy(f,fp::slice(f,r.state.selection,0,0,2048+tail));
  const auto selected=f.one(c::Operation(c::Gather{}),{r.state.main,ids});
  r.attention=f.one(c::Operation(c::ScaledDotProductAttention{.scale=1.F/16.F,.causal=false,.accumulation_type=c::DataType::kFloat32}),
      {fp::reshape(f,r.absorbed.output,{1,1,64,512}),fp::reshape(f,selected,{1,2048+tail,1,512}),fp::reshape(f,selected,{1,2048+tail,1,512})});
  r.value=p.project.per_head(f,fp::reshape(f,r.attention,{1,64,512}),p.nextn.block.value);
  r.output=p.project.linear(f,fp::reshape(f,r.value.output,{1,16384}),p.nextn.block.output);return r;
}
template<class FeedForward> struct NextnResult final {
  Mla mla;fp::Prefix prefix;fp::Norm post_norm,shared_norm;
  Value residual,combined,logits,top_values,pending;FeedForward moe;
  State &state(){return mla.state;}const State &state()const{return mla.state;}
  template<class Visitor>void visit_values(Visitor &&v) {
    prefix.visit_values(v);mla.visit_values(v);post_norm.visit_values(v);shared_norm.visit_values(v);moe.visit_values(v);
    for(auto *p:{&residual,&combined,&logits,&top_values,&pending})v(*p);
  }
};
using Result=NextnResult<Moe>;
template<class ParameterSet,class MoeBuilder>
inline auto append_with_moe(sem::Fragment &f,State state,Value actual_hidden,const ParameterSet &p,
    bool fresh,std::int32_t position,std::int32_t carried_tail,const types::Geometry &geometry,
    IndexPolicy policy,MoeBuilder &&moe_builder) {
  using FeedForward=decltype(moe_builder(f,actual_hidden));
  NextnResult<FeedForward> r;
  const auto embedded=f.one(c::Operation(c::Gather{}),{p.nextn.embedding,state.pending});
  r.prefix=fp::append_f32_prefix(f,actual_hidden,cast(f,embedded,c::DataType::kFloat32),p.nextn,geometry,p.project);
  r.mla=append_mla(f,state,r.prefix.input_norm.output,r.prefix.query_norm.output,p,fresh,position,carried_tail,policy);
  r.residual=core::binary(f,c::BinaryKind::kAdd,r.prefix.eh.output,r.mla.output.output);
  r.post_norm=fp::norm(f,r.residual,p.nextn.block.post_norm,geometry.rms_epsilon);
  r.moe=moe_builder(f,r.post_norm.output);
  r.combined=core::binary(f,c::BinaryKind::kAdd,r.residual,r.moe.output);
  r.shared_norm=fp::norm(f,r.combined,p.nextn.shared_norm,geometry.rms_epsilon);r.mla.state.hidden=r.shared_norm.output;
  r.logits=p.project.linear(f,r.mla.state.hidden,p.nextn.head).output;
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{r.logits},2);r.top_values=top[0];
  r.pending=fp::copy(f,fp::reshape(f,top[1],{1}));r.mla.state.pending=r.pending;return r;
}
inline Result append(sem::Fragment &f,State state,Value actual_hidden,const Parameters &p,std::span<const Value> banks,
    Value map,bool fresh,std::int32_t position,std::int32_t carried_tail) {
  const auto moe_builder=[&](sem::Fragment &graph,Value actual){return append_moe(graph,actual,p,banks,map);};
  return append_with_moe(f,state,actual_hidden,p,fresh,position,carried_tail,kGeometry,{},moe_builder);
}
} // namespace glm_mlx_mtp_fragment
