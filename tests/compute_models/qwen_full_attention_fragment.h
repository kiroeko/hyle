#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_moe_semantic_fragments.h"

// Pure Qwen full-attention math shared by target and NextN consumers. The
// caller owns immutable parameter Values, actual KV state and execution.
// Codecs come from those actual tensor descriptions, never a model switch.
namespace qwen_full_attention_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
using Value=sem::ValueId;
struct Geometry final {
  std::int64_t hidden,heads,kv_heads,depth,rotary;
  float epsilon=1.0e-6F;
  float scale=0.0F; // Metadata zero means the ordinary 1/sqrt(head depth).
  c::DataType storage=c::DataType::kFloat32;
};
inline void require_geometry(const Geometry &geometry) {
  constexpr auto limit=std::numeric_limits<std::int64_t>::max();
  if(geometry.hidden<=0||geometry.heads<=0||geometry.kv_heads<=0||geometry.depth<=0||
      geometry.heads%geometry.kv_heads||geometry.heads>limit/geometry.depth/2||
      geometry.kv_heads>limit/geometry.depth||geometry.rotary<=0||
      geometry.rotary>geometry.depth||geometry.rotary%2||
      !std::isfinite(geometry.epsilon)||geometry.epsilon<=0||
      !std::isfinite(geometry.scale)||geometry.scale<0||
      (geometry.storage!=c::DataType::kFloat32&&geometry.storage!=c::DataType::kBFloat16))
    throw std::runtime_error("invalid Qwen full-attention geometry");
}
inline void require(sem::Fragment &fragment,Value value,const c::TensorDesc &description) {
  if(value.ordinal()>=fragment.descriptions.size()||fragment.description(value)!=description)
    throw std::runtime_error("Qwen full-attention actual operand description differs");
}
inline Value cast(sem::Fragment &fragment,Value value,c::DataType type) {
  return fragment.description(value).data_type()==type?value:
      fragment.one(c::Operation(c::Cast{.output_type=type}),{value});
}
struct NormTrace final {Value input,gamma,raw,stored;};
struct ProjectionTrace final {Value input,weight,raw,stored;std::optional<Value> precursor{};};
struct LocalTrace final {
  std::vector<NormTrace> norms;
  std::vector<ProjectionTrace> projections;
  std::optional<Value> query_rotary_input,key_rotary_input,sdpa,gate_sigmoid,ffn_silu;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto &norm:norms)for(auto *value:{&norm.input,&norm.gamma,&norm.raw,&norm.stored})visitor(*value);
    // Immutable weight metadata is deliberately not an activation output.
    for(auto &projection:projections) {
      for(auto *value:{&projection.input,&projection.raw,&projection.stored})visitor(*value);
      if(projection.precursor)visitor(*projection.precursor);
    }
    for(auto *value:{&query_rotary_input,&key_rotary_input,&sdpa,&gate_sigmoid,&ffn_silu})if(*value)visitor(**value);
  }
};
// Gamma is already the effective F32 coefficient. Raw HF delta parameters
// are explicitly converted by the source adapter, never inferred here.
inline Value normalize(sem::Fragment &fragment,Value input,Value gamma,std::int64_t width,const Geometry &geometry,LocalTrace *trace=nullptr) {
  require(fragment,gamma,sem::dense({width}));
  const auto actual=geometry.storage==c::DataType::kBFloat16?cast(fragment,input,c::DataType::kFloat32):input;
  const auto raw=fragment.one(c::Operation(c::RmsNorm{.epsilon=geometry.epsilon}),{actual,gamma});
  const auto stored=geometry.storage==c::DataType::kBFloat16?cast(fragment,raw,c::DataType::kBFloat16):raw;
  if(trace)trace->norms.push_back({input,gamma,raw,stored});return stored;
}
inline bool bfloat_weight(const c::TensorDesc &weight) {
  return weight.is_plain()&&weight.data_type()==c::DataType::kBFloat16;
}
inline Value project(sem::Fragment &fragment,Value input,Value weight,const Geometry &geometry,LocalTrace *trace=nullptr) {
  const auto a=fragment.description(input),w=fragment.description(weight);
  if(bfloat_weight(w)) {
    if(a.rank()!=2||w.rank()!=2||!a.is_plain()||!a.is_dense()||!w.is_dense()||
        (a.data_type()!=c::DataType::kBFloat16&&a.data_type()!=c::DataType::kFloat32)||
        (geometry.storage==c::DataType::kBFloat16&&a.data_type()!=c::DataType::kBFloat16)||
        a.dimensions()[1]!=w.dimensions()[1])
      throw std::runtime_error("original Qwen projection needs original BF16 [N,K] and dense actual rows");
    const auto actual=cast(fragment,input,c::DataType::kBFloat16);
    const auto raw=fragment.one(c::model_ops::bfloat_linear_f32(),{actual,weight});
    // File W encoding does not change the caller's activation/state storage.
    const auto stored=geometry.storage==c::DataType::kBFloat16?cast(fragment,raw,c::DataType::kBFloat16):raw;
    if(trace)trace->projections.push_back({actual,weight,raw,stored,actual.ordinal()==input.ordinal()?std::nullopt:std::optional<Value>{input}});
    return stored;
  }
  if(geometry.storage==c::DataType::kBFloat16)
    throw std::runtime_error("whole-BF16 Qwen policy requires original BF16 projection weights");
  return fragment.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{input,weight});
}
inline Value slice(sem::Fragment &fragment,Value input,std::int32_t axis,
                   std::int64_t first,std::int64_t count) {
  return fragment.one(c::Operation(c::View::slice(axis,first,count).value()),{input});
}
inline std::vector<float> text_frequencies(std::int64_t rotary,float theta) {
  if(rotary<=0||rotary%2||!std::isfinite(theta)||theta<=0)
    throw std::runtime_error("invalid Qwen text rotary metadata");
  std::vector<float> result(std::size_t(rotary/2));
  for(std::size_t pair=0;pair<result.size();++pair)
    result[pair]=1.F/std::pow(theta,float(pair)/float(rotary/2));
  return result;
}
struct Projected final {Value query,gate,key,value;};
inline Projected append_qkv(sem::Fragment &fragment,Value qg,Value key,Value value,
    Value query_scale,Value key_scale,Value position,Value frequency,const Geometry &geometry,LocalTrace *trace=nullptr) {
  require_geometry(geometry);
  require(fragment,qg,sem::dense({1,2*geometry.heads*geometry.depth},geometry.storage));
  require(fragment,key,sem::dense({1,geometry.kv_heads*geometry.depth},geometry.storage));
  require(fragment,value,sem::dense({1,geometry.kv_heads*geometry.depth},geometry.storage));
  require(fragment,query_scale,sem::dense({geometry.depth}));
  require(fragment,key_scale,sem::dense({geometry.depth}));
  require(fragment,position,sem::dense({1},c::DataType::kInt32));
  require(fragment,frequency,sem::dense({geometry.rotary/2}));
  const auto qg_heads=core::reshape(fragment,qg,{1,1,geometry.heads,2*geometry.depth});
  const auto query=normalize(fragment,slice(fragment,qg_heads,3,0,geometry.depth),query_scale,geometry.depth,geometry,trace);
  const auto current_key=normalize(fragment,core::reshape(fragment,key,{1,1,geometry.kv_heads,geometry.depth}),
      key_scale,geometry.depth,geometry,trace);
  if(trace){trace->query_rotary_input=query;trace->key_rotary_input=current_key;}
  const c::Operation rope(c::RoPE{.rotary_dimensions=std::size_t(geometry.rotary),
      .sequence_axis=1,.layout=c::RopeLayout::kHalfSplit});
  return {fragment.one(rope,{query,position,frequency}),
      slice(fragment,qg_heads,3,geometry.depth,geometry.depth),
      fragment.one(rope,{current_key,position,frequency}),
      core::reshape(fragment,value,{1,1,geometry.kv_heads,geometry.depth})};
}
struct Cache final {Value key,value;};
inline std::int64_t cache_capacity(sem::Fragment &fragment,const Cache &cache,const Geometry &geometry) {
  require_geometry(geometry);
  if(cache.key.ordinal()>=fragment.descriptions.size())throw std::runtime_error("Qwen KV cache value is outside this Fragment");
  const auto &description=fragment.description(cache.key);
  if(description.rank()!=3||description.dimensions()[0]<=0)
    throw std::runtime_error("Qwen full-attention needs a positive finite KV cache");
  const auto capacity=description.dimensions()[0];
  const auto expected=sem::dense({capacity,geometry.kv_heads,geometry.depth},geometry.storage);
  require(fragment,cache.key,expected);require(fragment,cache.value,expected);
  return capacity;
}
inline Cache append_update(sem::Fragment &fragment,const Projected &projected,
    const Cache &old,Value cursor,const Geometry &geometry) {
  (void)cache_capacity(fragment,old,geometry);
  require(fragment,cursor,sem::dense({1},c::DataType::kInt32));
  return {fragment.one(c::Operation(c::Scatter{}),{old.key,cursor,
              core::reshape(fragment,projected.key,{1,geometry.kv_heads,geometry.depth})}),
      fragment.one(c::Operation(c::Scatter{}),{old.value,cursor,
              core::reshape(fragment,projected.value,{1,geometry.kv_heads,geometry.depth})})};
}
struct History final {Value keys,values,mask;};
inline History append_finite_history(sem::Fragment &fragment,const Cache &cache,
    Value cursor,const Geometry &geometry) {
  const auto capacity=cache_capacity(fragment,cache,geometry);
  require(fragment,cursor,sem::dense({1},c::DataType::kInt32));
  // This small finite-window policy belongs to these test consumers. It is
  // not a production cache allocator. Absolute position remains independent.
  constexpr auto limit=std::int64_t{(std::numeric_limits<std::int32_t>::max()-1024)/sizeof(float)};
  if(capacity>limit/capacity)throw std::runtime_error("finite Qwen history metadata is too large");
  std::vector<std::int32_t> mapping(std::size_t(capacity*capacity));
  std::vector<float> masks(mapping.size());
  for(std::int64_t index=0;index<capacity;++index)for(std::int64_t row=0;row<capacity;++row) {
    const auto offset=std::size_t(index*capacity+row);
    mapping[offset]=std::int32_t(row<=index?row:0);
    masks[offset]=row<=index?0.F:-std::numeric_limits<float>::infinity();
  }
  const auto row_table=fragment.constant(sem::dense({capacity,capacity},c::DataType::kInt32),sem::bytes(mapping));
  const auto mask_table=fragment.constant(sem::dense({capacity,capacity}),sem::bytes(masks));
  const auto rows=fragment.one(c::Operation(c::Gather{}),{row_table,cursor});
  const auto mask=core::reshape(fragment,fragment.one(c::Operation(c::Gather{}),{mask_table,cursor}),{1,1,1,capacity});
  // First gather only valid real rows. Repeating row0 for masked padding
  // prevents a poisoned unwritten KV tail from entering attention operands.
  return {fragment.one(c::Operation(c::Gather{}),{cache.key,rows}),
      fragment.one(c::Operation(c::Gather{}),{cache.value,rows}),mask};
}
inline Value append_attention(sem::Fragment &fragment,const Projected &projected,
    Value keys,Value values,const Geometry &geometry,std::optional<Value> additive_mask={},LocalTrace *trace=nullptr) {
  require_geometry(geometry);
  const float scale=geometry.scale==0?1.F/std::sqrt(float(geometry.depth)):geometry.scale;
  const c::Operation attention(c::ScaledDotProductAttention{
      .layout=c::AttentionLayout::kBatchSequenceHeadDepth,
      .mask=additive_mask?c::MaskKind::kAdditive:c::MaskKind::kNone,
      .scale=scale,.causal=!additive_mask,.accumulation_type=c::DataType::kFloat32});
  const auto context=additive_mask?fragment.one(attention,{projected.query,keys,values,*additive_mask}):
      fragment.one(attention,{projected.query,keys,values});
  const auto sigmoid=core::unary(fragment,c::UnaryKind::kSigmoid,projected.gate);
  if(trace){trace->sdpa=context;trace->gate_sigmoid=sigmoid;}
  return core::binary(fragment,c::BinaryKind::kMultiply,context,sigmoid);
}

struct TargetParameters final {
  Value input_norm,query,key,value,query_norm,key_norm,output;
  Value post_norm,ffn_gate,ffn_up,ffn_down,target_norm;
};
struct TargetAttentionParameters final {
  Value input_norm,query,key,value,query_norm,key_norm,output,post_norm;
};
struct TargetAttentionPorts {
  Value input,nx,qg,key,value,query,gate,rotated_key,attended,oproj,residual;
  Value ffn_input;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&nx,&qg,&key,&value,&query,&gate,&rotated_key,&attended,&oproj,&residual,&ffn_input})visitor(*port);
  }
};
struct TargetPorts final : TargetAttentionPorts {
  Value ffn_gate,ffn_up,ffn_activation,ffn_down,post_ffn,tap;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    TargetAttentionPorts::visit_values(visitor);
    for(auto *port:{&ffn_gate,&ffn_up,&ffn_activation,&ffn_down,&post_ffn,&tap})visitor(*port);
  }
};
struct TargetAttentionResult final {Value key,value;TargetAttentionPorts ports;std::optional<LocalTrace> trace;};
struct TargetResult final {
  Value key,value,raw_post_ffn,normalized_tap;
  TargetPorts ports;
  std::optional<LocalTrace> trace;
};
template<class ProjectionLeaf>
inline TargetAttentionResult append_target_attention_with_projection(sem::Fragment &fragment,Value input,Value old_key,Value old_value,
    Value cursor,Value position,Value frequency,const TargetAttentionParameters &parameters,const Geometry &geometry,
    std::optional<Value> normalized_input,ProjectionLeaf &&projection) {
  require_geometry(geometry);require(fragment,input,sem::dense({1,geometry.hidden},geometry.storage));
  TargetAttentionResult result;
  if(geometry.storage==c::DataType::kBFloat16||bfloat_weight(fragment.description(parameters.query))||
      bfloat_weight(fragment.description(parameters.key))||bfloat_weight(fragment.description(parameters.value))||
      bfloat_weight(fragment.description(parameters.output)))result.trace.emplace();
  auto *trace=result.trace?&*result.trace:nullptr;
  const auto norm=[&](Value actual,Value gamma,std::int64_t width) {
    return normalize(fragment,actual,gamma,width,geometry,trace);
  };
  const auto linear=[&](Value actual,Value weight) {
    // Public inference checks actual logical dimensions and storage codec.
    // No packed-format or target-model whitelist belongs in this builder.
    return projection(fragment,actual,weight,geometry,trace);
  };
  auto &ports=result.ports;ports.input=input;
  if(normalized_input) {
    require(fragment,*normalized_input,sem::dense({1,geometry.hidden},geometry.storage));
    ports.nx=*normalized_input;
  } else ports.nx=norm(input,parameters.input_norm,geometry.hidden);
  ports.qg=linear(ports.nx,parameters.query);ports.key=linear(ports.nx,parameters.key);ports.value=linear(ports.nx,parameters.value);
  const auto projected=append_qkv(fragment,ports.qg,ports.key,ports.value,parameters.query_norm,
      parameters.key_norm,position,frequency,geometry,trace);
  ports.query=projected.query;ports.gate=projected.gate;ports.rotated_key=projected.key;
  const auto cache=append_update(fragment,projected,{old_key,old_value},cursor,geometry);
  result.key=cache.key;result.value=cache.value;
  const auto history=append_finite_history(fragment,cache,cursor,geometry);
  ports.attended=append_attention(fragment,projected,history.keys,history.values,geometry,history.mask,trace);
  ports.oproj=linear(core::reshape(fragment,ports.attended,{1,geometry.heads*geometry.depth}),parameters.output);
  require(fragment,ports.oproj,sem::dense({1,geometry.hidden},geometry.storage));
  ports.residual=core::binary(fragment,c::BinaryKind::kAdd,input,ports.oproj);
  ports.ffn_input=norm(ports.residual,parameters.post_norm,geometry.hidden);
  return result;
}
template<class ProjectionLeaf>
inline TargetResult append_target_with_projection(sem::Fragment &fragment,Value input,Value old_key,Value old_value,
    Value cursor,Value position,Value frequency,const TargetParameters &parameters,const Geometry &geometry,
    std::optional<Value> normalized_input,ProjectionLeaf &&projection) {
  const auto attention=append_target_attention_with_projection(fragment,input,old_key,old_value,cursor,position,frequency,
      {parameters.input_norm,parameters.query,parameters.key,parameters.value,parameters.query_norm,parameters.key_norm,
       parameters.output,parameters.post_norm},geometry,normalized_input,projection);
  TargetResult result;result.key=attention.key;result.value=attention.value;auto &ports=result.ports;
  result.trace=attention.trace;auto *trace=result.trace?&*result.trace:nullptr;
  static_cast<TargetAttentionPorts&>(ports)=attention.ports;
  const auto linear=[&](Value actual,Value weight) {
    return projection(fragment,actual,weight,geometry,trace);
  };
  ports.ffn_gate=linear(ports.ffn_input,parameters.ffn_gate);ports.ffn_up=linear(ports.ffn_input,parameters.ffn_up);
  if(geometry.storage==c::DataType::kBFloat16) {
    const auto silu=core::unary(fragment,c::UnaryKind::kSilu,ports.ffn_gate);
    if(trace)trace->ffn_silu=silu;
    ports.ffn_activation=core::binary(fragment,c::BinaryKind::kMultiply,silu,ports.ffn_up);
  }
  else ports.ffn_activation=fragment.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{ports.ffn_gate,ports.ffn_up});
  ports.ffn_down=linear(ports.ffn_activation,parameters.ffn_down);
  require(fragment,ports.ffn_down,sem::dense({1,geometry.hidden},geometry.storage));
  ports.post_ffn=core::binary(fragment,c::BinaryKind::kAdd,ports.residual,ports.ffn_down);
  // Actual Qwen target graph publishes t_h_nextn AFTER global output_norm and
  // before output cropping. NextN applies its own distinct hnorm to this tap.
  require(fragment,parameters.target_norm,sem::dense({geometry.hidden}));
  ports.tap=normalize(fragment,ports.post_ffn,parameters.target_norm,geometry.hidden,geometry,trace);
  result.raw_post_ffn=ports.post_ffn;result.normalized_tap=ports.tap;return result;
}
// Existing BF16/packed callers use the same projection leaf and DAG. The
// template seam only supplies ordinary Graph operations during construction;
// it owns no execution, provider policy or model-state transition.
struct OrdinaryProjection final {
  Value operator()(sem::Fragment &fragment,Value input,Value weight,
      const Geometry &geometry,LocalTrace *trace) const {
    return project(fragment,input,weight,geometry,trace);
  }
};
inline TargetAttentionResult append_target_attention(sem::Fragment &fragment,Value input,Value old_key,Value old_value,
    Value cursor,Value position,Value frequency,const TargetAttentionParameters &parameters,const Geometry &geometry,
    std::optional<Value> normalized_input={}) {
  return append_target_attention_with_projection(fragment,input,old_key,old_value,cursor,position,frequency,
      parameters,geometry,normalized_input,OrdinaryProjection{});
}
inline TargetResult append_target(sem::Fragment &fragment,Value input,Value old_key,Value old_value,
    Value cursor,Value position,Value frequency,const TargetParameters &parameters,const Geometry &geometry,
    std::optional<Value> normalized_input={}) {
  return append_target_with_projection(fragment,input,old_key,old_value,cursor,position,frequency,
      parameters,geometry,normalized_input,OrdinaryProjection{});
}
} // namespace qwen_full_attention_fragment
