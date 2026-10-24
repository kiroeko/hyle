#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "glm_official_index_cache.h"
#include "glm_original_moe_fragment.h"

// Model-local original GLM Pro NextN construction. Values name operands in
// this same ordinary Fragment; no loading, execution, provider or acceptance
// policy is hidden here. The selected official non-fused indexer keeps its
// Hadamard/UE8M0 boundary. Main cache rows contain BOTH CKV512 and rotated KPE64.
namespace glm_pro_mtp_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace codec=glm_official_index_cache;
namespace moe=glm_original_moe_fragment;
using Value=sem::ValueId;
inline constexpr std::int64_t kHidden=6144,kQueryRank=2048,kHeads=64;
inline constexpr std::int64_t kNope=192,kRope=64,kLatent=512,kValue=256;
inline constexpr std::int64_t kCacheDepth=kLatent+kRope,kIndexDepth=128,kIndexHeads=32;
inline constexpr std::int64_t kVocabulary=154880;
inline constexpr float kEpsilon=1.0e-5F,kAttentionScale=1.0F/16.0F;

struct Parameters final {
  Value eh,embedding_norm,hidden_norm,input_norm,post_norm;
  Value index_bias,index_norm,index_weights,index_key,index_query;
  Value kv_norm,kv_a,output,query_norm,query_a,query_b;
  Value shared_norm,head,prepared_key,prepared_value;
};
// Shared MLA operands. Target and NextN callers supply their own trained
// parameters and explicit residual; this layer knows neither EH nor LM policy.
struct MlaParameters final {
  Value kv_norm,kv_a,output,post_norm,query_b,prepared_key,prepared_value;
};
inline MlaParameters mla_parameters(const Parameters &p) {
  return {p.kv_norm,p.kv_a,p.output,p.post_norm,p.query_b,p.prepared_key,p.prepared_value};
}
struct NormPorts final {
  Value input,raw,stored,visible;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored,&visible})visitor(*value);
  }
};
struct ProjectionPorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored})visitor(*value);
  }
};
struct RopePorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored})visitor(*value);
  }
};
struct Prefix final {
  Value hidden,embedding,joined;
  NormPorts hidden_norm,embedding_norm,input_norm,query_norm;
  ProjectionPorts eh,query;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&hidden,&embedding,&joined})visitor(*value);
    hidden_norm.visit_values(visitor);embedding_norm.visit_values(visitor);
    input_norm.visit_values(visitor);query_norm.visit_values(visitor);
    eh.visit_values(visitor);query.visit_values(visitor);
  }
};
struct IndexHistory final {
  Value input,positions,codes,scales;
  ProjectionPorts key;
  NormPorts key_norm;
  RopePorts rope;
  codec::Quantized encoded;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&positions,&codes,&scales})visitor(*value);
    key.visit_values(visitor);key_norm.visit_values(visitor);rope.visit_values(visitor);
    for(auto *value:{&encoded.codes,&encoded.scale,&encoded.rotated,&encoded.transformed,&encoded.input})visitor(*value);
  }
};
struct IndexPorts final {
  IndexHistory history;
  ProjectionPorts query;
  RopePorts rope;
  codec::Quantized encoded;
  Value weights_input,weights,scores,ids;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    history.visit_values(visitor);query.visit_values(visitor);rope.visit_values(visitor);
    for(auto *value:{&encoded.codes,&encoded.scale,&encoded.rotated,&encoded.transformed,&encoded.input,
        &weights_input,&weights,&scores,&ids})visitor(*value);
  }
};
struct AttentionPorts final {
  ProjectionPorts query,kv,output;
  NormPorts kv_norm,post_norm;
  RopePorts query_rope,key_rope;
  Value absorbed_input,absorbed_stored,query_full,row_stored;
  Value selected_full,selected_value,attention_raw,attention_stored,value_stored;
  Value residual_sum,residual_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    query.visit_values(visitor);kv.visit_values(visitor);output.visit_values(visitor);
    kv_norm.visit_values(visitor);post_norm.visit_values(visitor);
    query_rope.visit_values(visitor);key_rope.visit_values(visitor);
    for(auto *value:{&absorbed_input,&absorbed_stored,&query_full,&row_stored,
        &selected_full,&selected_value,&attention_raw,&attention_stored,&value_stored,
        &residual_sum,&residual_stored})visitor(*value);
  }
};
struct State final {
  Value main,codes,scales,selection,positions;
};
struct Options final {
  std::int64_t visible=0,selected=2048;
  bool fresh=true;
};
struct Ports final {
  Prefix prefix;
  AttentionPorts attention;
  std::optional<IndexPorts> index;
  Value positions,combined,logits,top_values,pending;
  NormPorts shared_norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    prefix.visit_values(visitor);attention.visit_values(visitor);
    if(index)index->visit_values(visitor);
    shared_norm.visit_values(visitor);
    for(auto *value:{&positions,&combined,&logits,&top_values,&pending})visitor(*value);
  }
};
struct Result final {
  State state;
  Value hidden,pending;
  Ports ports;
  moe::Result feed_forward;
};
struct PreparedKv final {Value decoded,key,value;};

inline void require(sem::Fragment &f,Value value,const c::TensorDesc &description) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=description)
    throw std::runtime_error("original Pro actual operand/parameter description differs");
}
inline Value cast(sem::Fragment &f,Value value,c::DataType type) {
  return f.one(c::Operation(c::Cast{.output_type=type}),{value});
}
inline Value reshape(sem::Fragment &f,Value value,std::initializer_list<std::int64_t> dimensions) {
  return core::reshape(f,value,dimensions);
}
inline Value permute(sem::Fragment &f,Value value,std::initializer_list<std::int32_t> axes) {
  return f.one(c::Operation(c::View::permute(std::span(axes.begin(),axes.size())).value()),{value});
}
inline Value slice(sem::Fragment &f,Value value,std::int32_t axis,std::int64_t first,std::int64_t count) {
  return f.one(c::Operation(c::View::slice(axis,first,count).value()),{value});
}
inline Value contiguous(sem::Fragment &f,Value value) {
  return f.one(c::Operation(c::Copy{}),{value});
}
inline NormPorts norm(sem::Fragment &f,Value actual_f32,Value gamma,
                      std::int64_t width,bool layer=false,Value bias={}) {
  require(f,actual_f32,sem::dense({1,width}));
  if(gamma.ordinal()>=f.descriptions.size())
    throw std::runtime_error("original Pro normalization parameter is outside this Fragment");
  Value scale=gamma;
  const auto &description=f.description(gamma);
  if(description==sem::dense({width},c::DataType::kBFloat16))scale=cast(f,gamma,c::DataType::kFloat32);
  else require(f,gamma,sem::dense({width}));
  Value raw;
  if(layer) {
    require(f,bias,sem::dense({width}));
    raw=f.one(c::Operation(c::LayerNorm{.epsilon=1.0e-6F}),{actual_f32,scale,bias});
  } else raw=f.one(c::Operation(c::RmsNorm{.epsilon=kEpsilon}),{actual_f32,scale});
  const auto stored=cast(f,raw,c::DataType::kBFloat16);
  return {actual_f32,raw,stored,cast(f,stored,c::DataType::kFloat32)};
}
inline ProjectionPorts fp8(sem::Fragment &f,Value actual_bf16,Value weight) {
  if(actual_bf16.ordinal()>=f.descriptions.size())
    throw std::runtime_error("original Pro FP8 source is outside this Fragment");
  const auto &description=f.description(actual_bf16);
  if(!description.is_plain()||!description.is_dense()||description.rank()!=2||
      description.data_type()!=c::DataType::kBFloat16)
    throw std::runtime_error("original Pro FP8 projection needs actual dense BF16 rows");
  const auto input=cast(f,actual_bf16,c::DataType::kFloat32);
  const auto raw=f.one(c::model_ops::float8_block_linear(
      c::model_ops::Float8ScalePolicy::kDivide,1.0e-10F),{input,weight});
  return {actual_bf16,raw,cast(f,raw,c::DataType::kBFloat16)};
}
inline RopePorts rotate(sem::Fragment &f,Value actual_bf16,Value positions) {
  // Only the first64 channels rotate. Indexer puts its rotary prefix first;
  // MLA calls this on its separate64-wide suffix. Absolute positions are an
  // explicit I32 input, never the cache slot or a per-head position counter.
  std::vector<float> frequencies(std::size_t(kRope/2));
  for(std::size_t pair=0;pair<frequencies.size();++pair)
    frequencies[pair]=float(std::pow(8000000.0,-2.0*double(pair)/double(kRope)));
  const auto frequency=f.constant(sem::dense({kRope/2}),sem::bytes(frequencies));
  const auto &description=f.description(actual_bf16);
  if(description.rank()!=3||description.dimensions()[0]!=1||
      !description.is_dense()||description.data_type()!=c::DataType::kBFloat16)
    throw std::runtime_error("original Pro rotary source needs actual dense BF16[1,H,D]");
  const auto heads=description.dimensions()[1],width=description.dimensions()[2];
  // Use the ordinary BSHD native admission, not a different rank3 provider
  // assumption. These Views preserve the single explicit absolute position.
  const auto raw4=f.one(c::Operation(c::RoPE{.rotary_dimensions=std::size_t(kRope),
      .sequence_axis=1,.layout=c::RopeLayout::kInterleaved}),
      {reshape(f,cast(f,actual_bf16,c::DataType::kFloat32),{1,1,heads,width}),positions,frequency});
  const auto raw=reshape(f,raw4,{1,heads,width});
  return {actual_bf16,raw,cast(f,raw,c::DataType::kBFloat16)};
}

// Cold only. Decode the ONE original macroblock grid before the per-head
// split:448 rows/head is not divisible by128, so scales must not reset at a
// head boundary. Outputs are compact BF16, never a persistent F32 weight bank.
inline PreparedKv append_prepare_kv(sem::Fragment &f,Value encoded) {
  require(f,encoded,c::float8_weight_description(kLatent,kHeads*(kNope+kValue),
      c::Float8ScaleMode::kFloat32Block128).value());
  const auto decoded=f.one(c::model_ops::float8_weight_decode(),{encoded});
  const auto heads=reshape(f,decoded,{kHeads,kNope+kValue,kLatent});
  const auto key=contiguous(f,permute(f,slice(f,heads,1,0,kNope),{0,2,1}));
  const auto value=contiguous(f,slice(f,heads,1,kNope,kValue));
  return {decoded,key,value};
}

inline Prefix append_prefix(sem::Fragment &f,Value hidden,Value embedding,const Parameters &p) {
  require(f,hidden,sem::dense({1,kHidden},c::DataType::kBFloat16));
  require(f,embedding,sem::dense({1,kHidden},c::DataType::kBFloat16));
  require(f,p.eh,sem::dense({kHidden,2*kHidden},c::DataType::kBFloat16));
  Prefix result;result.hidden=hidden;result.embedding=embedding;
  result.hidden_norm=norm(f,cast(f,hidden,c::DataType::kFloat32),p.hidden_norm,kHidden);
  result.embedding_norm=norm(f,cast(f,embedding,c::DataType::kFloat32),p.embedding_norm,kHidden);
  result.joined=f.one(c::Operation(c::Concat{.axis=1}),
      {result.embedding_norm.stored,result.hidden_norm.stored});
  const auto raw=f.one(c::model_ops::dense_linear_f32(),{result.joined,p.eh});
  result.eh={result.joined,raw,cast(f,raw,c::DataType::kBFloat16)};
  result.input_norm=norm(f,cast(f,result.eh.stored,c::DataType::kFloat32),p.input_norm,kHidden);
  result.query=fp8(f,result.input_norm.stored,p.query_a);
  result.query_norm=norm(f,cast(f,result.query.stored,c::DataType::kFloat32),p.query_norm,kQueryRank);
  return result;
}

// Fresh selection and accepted-feature catchup share this actual unpooled
// history producer. A skip step never calls it. Caller ensures positions are
// the corresponding absolute append rows of its finite declared cache.
inline IndexHistory append_index_history(sem::Fragment &f,Value input,Value positions,
    Value old_codes,Value old_scales,const Parameters &p) {
  require(f,input,sem::dense({1,kHidden},c::DataType::kBFloat16));
  require(f,positions,sem::dense({1},c::DataType::kInt32));
  const auto &codes=f.description(old_codes);
  if(codes.rank()!=2||codes.dimensions()[0]<=0)
    throw std::runtime_error("original Pro index cache requires a positive finite row capacity");
  const auto capacity=codes.dimensions()[0];
  require(f,old_codes,sem::dense({capacity,kIndexDepth},c::DataType::kFloat8E4M3FN));
  require(f,old_scales,sem::dense({capacity,1}));
  IndexHistory result;result.input=input;result.positions=positions;
  result.key=fp8(f,input,p.index_key);
  result.key_norm=norm(f,cast(f,result.key.stored,c::DataType::kFloat32),
      p.index_norm,kIndexDepth,true,p.index_bias);
  result.rope=rotate(f,reshape(f,result.key_norm.stored,{1,1,kIndexDepth}),positions);
  result.encoded=codec::quantize(f,reshape(f,cast(f,result.rope.stored,c::DataType::kFloat32),{1,kIndexDepth}),1);
  result.codes=f.one(c::Operation(c::Scatter{}),{old_codes,positions,result.encoded.codes});
  result.scales=f.one(c::Operation(c::Scatter{}),{old_scales,positions,result.encoded.scale});
  return result;
}

// Prefix and suffix are split at the optional learned-indexer frontier.
// The shared-index target77 legitimately supplies upstream selection directly.
inline Value append_mla_prefix(sem::Fragment &f,Value input_normalized,Value query_normalized,
    Value old_main,Value positions,const MlaParameters &p,AttentionPorts &a) {
  require(f,input_normalized,sem::dense({1,kHidden},c::DataType::kBFloat16));
  require(f,query_normalized,sem::dense({1,kQueryRank},c::DataType::kBFloat16));
  const auto &main=f.description(old_main);
  if(main.rank()!=2||main.dimensions()[0]<=0)
    throw std::runtime_error("Pro MLA requires a finite declared main cache");
  require(f,old_main,sem::dense({main.dimensions()[0],kCacheDepth}));
  require(f,positions,sem::dense({1},c::DataType::kInt32));
  require(f,p.prepared_key,sem::dense({kHeads,kLatent,kNope},c::DataType::kBFloat16));
  require(f,p.prepared_value,sem::dense({kHeads,kValue,kLatent},c::DataType::kBFloat16));
  a.query=fp8(f,query_normalized,p.query_b);
  const auto query_heads=reshape(f,a.query.stored,{1,kHeads,kNope+kRope});
  a.absorbed_input=contiguous(f,slice(f,query_heads,2,0,kNope));
  a.absorbed_stored=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),
      {reshape(f,a.absorbed_input,{kHeads,1,kNope}),permute(f,p.prepared_key,{0,2,1})});
  a.query_rope=rotate(f,contiguous(f,slice(f,query_heads,2,kNope,kRope)),positions);
  a.query_full=reshape(f,cast(f,f.one(c::Operation(c::Concat{.axis=2}),
      {reshape(f,a.absorbed_stored,{1,kHeads,kLatent}),a.query_rope.stored}),c::DataType::kFloat32),
      {1,1,kHeads,kCacheDepth});
  a.kv=fp8(f,input_normalized,p.kv_a);
  const auto latent=contiguous(f,slice(f,a.kv.stored,1,0,kLatent));
  a.kv_norm=norm(f,cast(f,latent,c::DataType::kFloat32),p.kv_norm,kLatent);
  a.key_rope=rotate(f,reshape(f,contiguous(f,slice(f,a.kv.stored,1,kLatent,kRope)),{1,1,kRope}),positions);
  a.row_stored=f.one(c::Operation(c::Concat{.axis=1}),
      {a.kv_norm.stored,reshape(f,a.key_rope.stored,{1,kRope})});
  const auto next=f.one(c::Operation(c::Scatter{}),
      {old_main,positions,cast(f,a.row_stored,c::DataType::kFloat32)});
  return next;
}
inline void append_mla_suffix(sem::Fragment &f,Value main,Value selection,Value residual,
    const MlaParameters &p,AttentionPorts &a) {
  require(f,residual,sem::dense({1,kHidden},c::DataType::kBFloat16));
  const auto &ids=f.description(selection);
  if(ids.rank()!=1||ids.dimensions()[0]<=0||ids.dimensions()[0]>2048)
    throw std::runtime_error("Pro MLA requires positive bounded upstream row IDs");
  const auto selected=ids.dimensions()[0];
  require(f,selection,sem::dense({selected},c::DataType::kInt32));
  a.selected_full=f.one(c::Operation(c::Gather{}),{main,selection});
  a.selected_value=contiguous(f,slice(f,a.selected_full,1,0,kLatent));
  a.attention_raw=f.one(c::Operation(c::ScaledDotProductAttention{
      .scale=kAttentionScale,.causal=false,.accumulation_type=c::DataType::kFloat32}),
      {a.query_full,reshape(f,a.selected_full,{1,selected,1,kCacheDepth}),
       reshape(f,a.selected_value,{1,selected,1,kLatent})});
  a.attention_stored=cast(f,a.attention_raw,c::DataType::kBFloat16);
  a.value_stored=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),
      {reshape(f,a.attention_stored,{kHeads,1,kLatent}),permute(f,p.prepared_value,{0,2,1})});
  a.output=fp8(f,reshape(f,a.value_stored,{1,kHeads*kValue}),p.output);
  a.residual_sum=core::binary(f,c::BinaryKind::kAdd,
      cast(f,residual,c::DataType::kFloat32),cast(f,a.output.stored,c::DataType::kFloat32));
  a.residual_stored=cast(f,a.residual_sum,c::DataType::kBFloat16);
  a.post_norm=norm(f,a.residual_sum,p.post_norm,kHidden);
}

inline Result append_step(sem::Fragment &f,const Prefix &prefix,const State &state,
    const Parameters &p,const moe::Parameters &moe_parameters,const Options &options) {
  const auto &main=f.description(state.main);
  if(main.rank()!=2||main.dimensions()[0]<=0)
    throw std::runtime_error("original Pro main cache needs positive declared capacity");
  const auto capacity=main.dimensions()[0];
  if(options.visible<=0||options.visible>capacity||options.selected<=0||
      options.selected>2048||options.selected>options.visible)
    throw std::runtime_error("original Pro visible/selected cache extent differs");
  require(f,state.main,sem::dense({capacity,kCacheDepth}));
  require(f,state.codes,sem::dense({capacity,kIndexDepth},c::DataType::kFloat8E4M3FN));
  require(f,state.scales,sem::dense({capacity,1}));
  require(f,state.selection,sem::dense({options.selected},c::DataType::kInt32));
  require(f,state.positions,sem::dense({1},c::DataType::kInt32));
  require(f,p.prepared_key,sem::dense({kHeads,kLatent,kNope},c::DataType::kBFloat16));
  require(f,p.prepared_value,sem::dense({kHeads,kValue,kLatent},c::DataType::kBFloat16));
  Result result;result.state=state;result.ports.prefix=prefix;result.ports.positions=state.positions;
  auto &a=result.ports.attention;
  result.state.main=append_mla_prefix(f,prefix.input_norm.stored,prefix.query_norm.stored,
      state.main,state.positions,mla_parameters(p),a);
  if(options.fresh) {
    result.ports.index.emplace();auto &index=*result.ports.index;
    index.history=append_index_history(f,prefix.input_norm.stored,state.positions,state.codes,state.scales,p);
    result.state.codes=index.history.codes;result.state.scales=index.history.scales;
    index.query=fp8(f,prefix.query_norm.stored,p.index_query);
    index.rope=rotate(f,reshape(f,index.query.stored,{1,kIndexHeads,kIndexDepth}),state.positions);
    index.encoded=codec::quantize(f,reshape(f,cast(f,index.rope.stored,c::DataType::kFloat32),{kIndexHeads,kIndexDepth}),kIndexHeads);
    index.weights_input=prefix.input_norm.stored;
    index.weights=f.one(c::model_ops::dense_linear_f32(),{index.weights_input,p.index_weights});
    index.scores=codec::score_encoded(f,result.state.codes,result.state.scales,index.encoded,index.weights,options.visible);
    const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(options.selected),.stable=true}),{index.scores},2);
    index.ids=contiguous(f,reshape(f,top[1],{options.selected}));result.state.selection=index.ids;
  }
  // The SAME selected absolute rows supply the full576-wide K and the first
  //512 V channels. Copy makes the value view compact without a persistent
  // duplicate cache. Future/unwritten rows must never appear in selection.
  append_mla_suffix(f,result.state.main,result.state.selection,prefix.eh.stored,mla_parameters(p),a);
  result.feed_forward=moe::append(f,a.post_norm.stored,moe_parameters,glm_moe_geometry::kPro);
  result.ports.combined=core::binary(f,c::BinaryKind::kAdd,
      cast(f,a.residual_stored,c::DataType::kFloat32),cast(f,result.feed_forward.output,c::DataType::kFloat32));
  result.ports.shared_norm=norm(f,result.ports.combined,p.shared_norm,kHidden);
  result.hidden=result.ports.shared_norm.stored;
  result.ports.logits=f.one(c::model_ops::dense_linear_f32(),{result.hidden,p.head});
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{result.ports.logits},2);
  result.ports.top_values=top[0];result.pending=contiguous(f,reshape(f,top[1],{1}));
  result.ports.pending=result.pending;
  return result;
}
} // namespace glm_pro_mtp_fragment
