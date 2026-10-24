#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_deepseek_official_attention.h"
#include "dspark_target_geometry.h"
#include "hyle/compute/low_precision.h"

// Formal DeepSeek projection, HC-head and MoE mathematics shared by trained
// target layers and their original DSpark stages. All weights/caches are
// caller-owned Values. No loader, host coefficients, executor or provider
// decision participates in construction.
namespace dspark_formal_fragment {
namespace ds=d5_deepseek_official_attention;
namespace core=d5_moe_semantic_fragments;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
namespace geometry=dspark_target_geometry;
using Geometry=geometry::Geometry;
using Value=sem::ValueId;
inline constexpr auto kSlots=geometry::kSlots;

inline void require(const sem::Fragment &f,Value value,const c::TensorDesc &description) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=description)
    throw std::runtime_error("original DeepSeek target Value has a different typed layout");
}
inline Value cast(sem::Fragment &f,Value value,c::DataType type) {
  return f.description(value).data_type()==type?value:f.one(c::Operation(c::Cast{type}),{value});
}
inline Value stored(sem::Fragment &f,Value value) {return cast(f,value,c::DataType::kBFloat16);}
inline Value visible(sem::Fragment &f,Value value) {return cast(f,value,c::DataType::kFloat32);}
inline Value copy(sem::Fragment &f,Value value) {return f.one(c::Operation(c::Copy{}),{value});}
inline Value reshape(sem::Fragment &f,Value value,std::initializer_list<std::int64_t> shape) {
  return core::reshape(f,value,shape);
}
inline Value indices(sem::Fragment &f,std::span<const std::int32_t> values) {
  return f.constant(sem::dense({std::int64_t(values.size())},c::DataType::kInt32),sem::bytes(values));
}
inline Value index(sem::Fragment &f,std::int64_t value) {
  if(value<0||value>std::numeric_limits<std::int32_t>::max())throw std::runtime_error("target index exceeds I32");
  return indices(f,std::array{std::int32_t(value)});
}
inline Value range(sem::Fragment &f,std::int64_t count,std::int64_t first=0) {
  if(count<=0||first<0||first>std::numeric_limits<std::int32_t>::max()-count)
    throw std::runtime_error("target finite range exceeds I32");
  std::vector<std::int32_t> values(std::size_t(count),0);std::iota(values.begin(),values.end(),std::int32_t(first));
  return indices(f,values);
}
inline Value scatter(sem::Fragment &f,Value bank,Value rows,Value update) {
  return f.one(c::Operation(c::Scatter{}),{bank,rows,update});
}
inline Value gather(sem::Fragment &f,Value bank,Value rows) {
  return f.one(c::Operation(c::Gather{}),{bank,rows});
}
inline Value decode(sem::Fragment &f,Value bank,Value rows,c::RowCacheEncoding encoding) {
  return f.one(c::model_ops::row_cache_gather_decode(encoding),{bank,rows});
}
inline Value encode(sem::Fragment &f,Value rows,c::RowCacheEncoding encoding) {
  return f.one(c::model_ops::row_cache_encode(encoding),{stored(f,rows)});
}
inline Value binary(sem::Fragment &f,c::BinaryKind kind,Value a,Value b) {return core::binary(f,kind,a,b);}
inline Value sum(sem::Fragment &f,Value input,std::int32_t axis,bool keep=true) {
  return f.one(c::Operation(c::Reduce{.kind=c::ReductionKind::kSum,
      .axes=c::Axes::from(std::array{axis}).value(),.keep_dimensions=keep,
      .accumulation_type=c::DataType::kFloat32}),{input});
}


struct NormPorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);visitor(raw);visitor(stored);}
};
inline NormPorts norm(sem::Fragment &f,Value input,Value gamma) {
  const auto width=f.description(input).dimensions().back();
  const auto type=f.description(gamma).data_type();
  if(type!=c::DataType::kBFloat16&&type!=c::DataType::kFloat32)throw std::runtime_error("target norm gamma storage");
  require(f,gamma,sem::dense({width},type));
  const auto actual=visible(f,input);
  const auto raw=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-6F}),{actual,visible(f,gamma)});
  return {actual,raw,stored(f,raw)};
}
struct ProjectionPorts final {
  Value input;
  std::optional<Value> raw;
  Value stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);if(raw)visitor(*raw);visitor(stored);}
};
inline ProjectionPorts fp8(sem::Fragment &f,Value input,Value weight,std::int64_t outputs,std::int64_t width) {
  require(f,weight,c::float8_weight_description(width,outputs,c::Float8ScaleMode::kFloat32Block128).value());
  if(input.ordinal()>=f.descriptions.size()||f.description(input).rank()!=2)
    throw std::runtime_error("formal FP8 projection requires a typed matrix input");
  const auto rows=f.description(input).dimensions()[0];
  const auto actual=stored(f,input);require(f,actual,sem::dense({rows,width},c::DataType::kBFloat16));
  const auto output=f.one(c::model_ops::float8_block_linear(c::model_ops::Float8ScalePolicy::kPowerOfTwo,1.0e-4F),{actual,weight});
  // The original operation returns BF16 directly. Its internal F32 dot is
  // not a public port; the independent reference checks an RNE interval.
  return {actual,{},output};
}
struct OutputAPorts final {
  Value input,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);visitor(stored);}
};
// Original wo_a is cold-decoded once to BF16 [G,4096,1024]. Only the small
// activation/result tiles change order when a draft has several token rows.
inline OutputAPorts output_a(sem::Fragment &f,Value input,Value weight,
    std::int64_t rows,const Geometry &model) {
  model.validate();
  if(rows<=0)throw std::runtime_error("formal outputA needs nonempty token rows");
  require(f,input,sem::dense({1,rows,model.heads,geometry::kDepth},c::DataType::kBFloat16));
  require(f,weight,sem::dense({model.groups,model.group_width(),geometry::kOutputRank},c::DataType::kBFloat16));
  Value lhs;
  if(rows==1)lhs=reshape(f,input,{model.groups,1,model.group_width()});
  else {
    const auto grouped=reshape(f,input,{rows,model.groups,model.group_width()});
    lhs=copy(f,f.one(c::Operation(c::View::permute(std::array<std::int32_t,3>{1,0,2}).value()),{grouped}));
  }
  auto output=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{lhs,weight});
  if(rows!=1)output=copy(f,f.one(c::Operation(c::View::permute(std::array<std::int32_t,3>{1,0,2}).value()),{output}));
  return {lhs,output};
}

struct HcParameters final {ds::HyperValues mixing;Value norm;};
struct MoeParameters final {
  Value router,bias;
  std::array<Value,3> shared,banks; // gate/up/down
  Value map;
};
struct Boundary final {Value input,ids,weights;std::array<Value,3> banks;Value map;};
struct MoeProjectionPorts final {
  Value gate_stored,up_stored,gate_clipped,up_clipped,activation_raw,down_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&gate_stored,&up_stored,&gate_clipped,&up_clipped,&activation_raw,&down_stored})visitor(*port);
  }
};
struct MoePorts final {
  Value input,logits,softplus,scores,selection,ids,selected,sum,normalized,weights,local_ids,map;
  MoeProjectionPorts routed,shared;
  Value weighted_raw,weighted_stored,shared_activation_stored,routed_sum,merged_raw,merged_stored;
  ProjectionPorts shared_gate,shared_up,shared_down;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&logits,&softplus,&scores,&selection,&ids,&selected,&sum,&normalized,&weights,&local_ids,&map})visitor(*port);
    routed.visit_values(visitor);shared.visit_values(visitor);
    for(auto *port:{&weighted_raw,&weighted_stored,&shared_activation_stored,&routed_sum,&merged_raw,&merged_stored})visitor(*port);
    shared_gate.visit_values(visitor);shared_up.visit_values(visitor);shared_down.visit_values(visitor);
  }
};
struct MoeResult final {Value output;MoePorts ports;Boundary boundary;};
inline MoeResult append_moe(sem::Fragment &f,Value input,const MoeParameters &parameters,
    const Geometry &model=geometry::kFlash) {
  model.validate();
  if(input.ordinal()>=f.descriptions.size()||f.description(input).rank()!=2)
    throw std::runtime_error("formal MoE requires a typed matrix input");
  const auto rows=f.description(input).dimensions()[0];
  if(rows<=0||rows>std::numeric_limits<std::int32_t>::max()/kSlots)
    throw std::runtime_error("formal MoE row/slot extent exceeds its I32 domain");
  require(f,input,sem::dense({rows,model.hidden},c::DataType::kBFloat16));
  require(f,parameters.router,sem::dense({model.experts,model.hidden},c::DataType::kBFloat16));
  require(f,parameters.bias,sem::dense({model.experts}));require(f,parameters.map,sem::dense({model.experts},c::DataType::kInt32));
  for(std::size_t leg=0;leg<3;++leg) {
    const auto n=leg==2?model.hidden:model.intermediate,k=leg==2?model.intermediate:model.hidden;
    if(parameters.banks[leg].ordinal()>=f.descriptions.size()||f.description(parameters.banks[leg]).rank()!=3)
      throw std::runtime_error("formal MoE bank needs a typed original FP4 descriptor");
    const auto capacity=f.description(parameters.banks[leg]).dimensions()[0];
    if(capacity<std::min(model.experts,rows*kSlots)||capacity>model.experts)
      throw std::runtime_error("formal MoE resident capacity cannot hold the actual row union");
    require(f,parameters.banks[leg],c::float4_grouped_weight_description(capacity,n,k).value());
    if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.banks[leg].ordinal();}))
      throw std::runtime_error("target selected FP4 banks must be explicit ordinary inputs");
  }
  if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.map.ordinal();}))
    throw std::runtime_error("target exact global map must be an ordinary input");
  MoePorts ports;ports.input=input;ports.map=parameters.map;
  ports.logits=f.one(c::model_ops::dense_linear_f32(),{input,parameters.router});
  ports.softplus=core::unary(f,c::UnaryKind::kSoftplus,ports.logits);ports.scores=core::unary(f,c::UnaryKind::kSqrt,ports.softplus);
  ports.selection=binary(f,c::BinaryKind::kAdd,ports.scores,parameters.bias);
  const auto selected=f.append(c::Operation(c::TopK{.count=std::size_t(kSlots),.stable=true}),{ports.selection},2);
  ports.ids=selected[1];ports.selected=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),{ports.scores,ports.ids});
  ports.sum=sum(f,ports.selected,-1);ports.normalized=binary(f,c::BinaryKind::kDivide,ports.selected,ports.sum);
  ports.weights=binary(f,c::BinaryKind::kMultiply,ports.normalized,core::filled(f,sem::dense({1,kSlots}),model.route_scale));
  const Boundary boundary{input,ports.ids,ports.weights,parameters.banks,parameters.map};
  ports.local_ids=gather(f,parameters.map,ports.ids);
  const auto grouped=[&](Value actual,std::size_t leg) {
    return f.one(c::model_ops::grouped_float4_linear(),{parameters.banks[leg],actual,ports.local_ids});
  };
  const auto activate=[&](MoeProjectionPorts &branch) {
    branch.gate_clipped=f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(),10.F),{visible(f,branch.gate_stored)});
    branch.up_clipped=f.one(c::model_ops::clamp(-10.F,10.F),{visible(f,branch.up_stored)});
    branch.activation_raw=binary(f,c::BinaryKind::kMultiply,core::unary(f,c::UnaryKind::kSilu,branch.gate_clipped),branch.up_clipped);
  };
  const auto slots=reshape(f,input,{rows,1,model.hidden});
  ports.routed.gate_stored=grouped(slots,0);ports.routed.up_stored=grouped(slots,1);activate(ports.routed);
  // Original Expert.forward multiplies F32 activation by actual route weight
  // BEFORE its sole BF16 input store/down A128 quantization boundary.
  ports.weighted_raw=binary(f,c::BinaryKind::kMultiply,ports.routed.activation_raw,reshape(f,ports.weights,{rows,kSlots,1}));
  ports.weighted_stored=stored(f,ports.weighted_raw);ports.routed.down_stored=grouped(ports.weighted_stored,2);
  ports.routed_sum=reshape(f,sum(f,visible(f,ports.routed.down_stored),1),{rows,model.hidden});
  ports.shared_gate=fp8(f,input,parameters.shared[0],model.intermediate,model.hidden);
  ports.shared_up=fp8(f,input,parameters.shared[1],model.intermediate,model.hidden);
  ports.shared.gate_stored=ports.shared_gate.stored;ports.shared.up_stored=ports.shared_up.stored;activate(ports.shared);
  ports.shared_activation_stored=stored(f,ports.shared.activation_raw);
  ports.shared_down=fp8(f,ports.shared_activation_stored,parameters.shared[2],model.hidden,model.intermediate);
  ports.shared.down_stored=ports.shared_down.stored;
  // The source y accumulator is F32 through sum6 AND shared addition. There
  // is no intermediate BF16 routed-sum store or extra shared route scaling.
  ports.merged_raw=binary(f,c::BinaryKind::kAdd,ports.routed_sum,visible(f,ports.shared.down_stored));
  ports.merged_stored=stored(f,ports.merged_raw);return {ports.merged_stored,ports,boundary};
}


struct HeadParameters final {Value fn,scale,base,norm;};
struct HeadPorts final {
  Value input,flat,projection,inverse,mixes,gate,collapsed_raw,collapsed_stored;
  NormPorts norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&flat,&projection,&inverse,&mixes,&gate,&collapsed_raw,&collapsed_stored})visitor(*port);
    norm.visit_values(visitor);
  }
};
struct HeadResult final {Value normalized;HeadPorts ports;};
inline HeadResult append_head(sem::Fragment &f,Value input,const HeadParameters &parameters) {
  if(input.ordinal()>=f.descriptions.size()||f.description(input).rank()!=3)
    throw std::runtime_error("original target head needs a typed four-stream input");
  const auto hidden=f.description(input).dimensions().back();
  const auto rows=f.description(input).dimensions()[0];
  require(f,input,sem::dense({rows,4,hidden},c::DataType::kBFloat16));
  require(f,parameters.fn,sem::dense({4,4*hidden}));require(f,parameters.scale,sem::dense({1}));require(f,parameters.base,sem::dense({4}));
  HeadResult result;auto &ports=result.ports;ports.input=input;ports.flat=reshape(f,visible(f,input),{rows,4*hidden});
  const auto transpose=f.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{parameters.fn});
  ports.projection=f.one(c::Operation(c::MatMul{}),{ports.flat,transpose});ports.inverse=ds::inverse_rms(f,ports.flat);
  ports.mixes=binary(f,c::BinaryKind::kMultiply,ports.projection,ports.inverse);
  ports.gate=binary(f,c::BinaryKind::kAdd,core::unary(f,c::UnaryKind::kSigmoid,
      binary(f,c::BinaryKind::kAdd,binary(f,c::BinaryKind::kMultiply,ports.mixes,parameters.scale),reshape(f,parameters.base,{1,4}))),
      core::filled(f,sem::dense({1}),1.0e-6F));
  ports.collapsed_raw=f.one(c::model_ops::hyper_connection_pre(),{visible(f,input),ports.gate});
  ports.collapsed_stored=stored(f,ports.collapsed_raw);ports.norm=norm(f,ports.collapsed_stored,parameters.norm);
  result.normalized=ports.norm.stored;return result;
}


} // namespace dspark_formal_fragment
