#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "d5_moe_semantic_fragments.h"
#include "hyle/compute/model_ops.h"

// Model-local GLM Flash hyper-connections shared by ordinary target layers.
// Fixed06992 torch/unfused contract: raw F32 mixing projection, model-RMS
// epsilon1e-5, HC/Sinkhorn epsilon1e-6 and20 iterations. Default storage has
// BF16 collapse/learned RMS/post boundaries; explicit F32 storage retains the
// original mixed-type F32 path. No parameter supplier or execution.
namespace glm_hyper_connection_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
using Value=sem::ValueId;
inline constexpr std::int64_t kHidden=4096,kStreams=4;
inline constexpr float kNormEpsilon=1.0e-5F,kHcEpsilon=1.0e-6F;
enum class ProjectionOrder {kScaleRawProjection,kNormalizeInput};

struct NormPorts final {
  Value input,raw,stored,visible;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored,&visible})visitor(*value);
  }
};
struct HcPorts final {
  Value residual,flat,projection,square,mean,inverse,mixes,pre,post,combination;
  ProjectionOrder projection_order=ProjectionOrder::kScaleRawProjection;
  std::optional<Value> normalized_input;
  Value collapsed_raw,collapsed_stored;
  NormPorts norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visitor(residual);visitor(flat);
    if(normalized_input)visitor(*normalized_input);
    visitor(projection);
    if(projection_order==ProjectionOrder::kScaleRawProjection)
      for(auto *value:{&square,&mean,&inverse})visitor(*value);
    for(auto *value:{&mixes,&pre,&post,&combination,&collapsed_raw,&collapsed_stored})visitor(*value);
    norm.visit_values(visitor);
  }
};
struct PostPorts final {
  Value branch,residual,raw,stored,visible;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&branch,&residual,&raw,&stored,&visible})visitor(*value);
  }
};
inline Value cast(sem::Fragment &f,Value input,c::DataType type) {
  return f.one(c::Operation(c::Cast{.output_type=type}),{input});
}
inline Value slice(sem::Fragment &f,Value input,std::int32_t axis,std::int64_t begin,std::int64_t count) {
  return f.one(c::Operation(c::View::slice(axis,begin,count).value()),{input});
}
inline Value compact(sem::Fragment &f,Value input) {return f.one(c::Operation(c::Copy{}),{input});}
inline Value reduce(sem::Fragment &f,Value input,std::int32_t axis,c::ReductionKind kind,bool keep=true) {
  return f.one(c::Operation(c::Reduce{.kind=kind,.axes=c::Axes::from(std::array{axis}).value(),
      .keep_dimensions=keep,.accumulation_type=c::DataType::kFloat32}),{input});
}
inline void require_storage(c::DataType type) {
  if(type!=c::DataType::kBFloat16&&type!=c::DataType::kFloat32)
    throw std::runtime_error("GLM HC requires its explicit BF16 or F32 activation storage");
}
inline Value as_f32(sem::Fragment &f,Value input) {
  const auto type=f.description(input).data_type();require_storage(type);
  return type==c::DataType::kFloat32?input:cast(f,input,c::DataType::kFloat32);
}
inline Value store(sem::Fragment &f,Value input,c::DataType type) {
  require_storage(type);
  if(f.description(input).data_type()!=c::DataType::kFloat32)
    throw std::runtime_error("GLM explicit activation store needs its real F32 predecessor");
  return type==c::DataType::kFloat32?input:cast(f,input,type);
}
inline NormPorts norm(sem::Fragment &f,Value actual,Value gamma,
                      c::DataType storage_type=c::DataType::kBFloat16) {
  require_storage(storage_type);
  if(f.description(actual).data_type()!=storage_type)
    throw std::runtime_error("GLM learned RMS input differs from its explicit activation storage");
  NormPorts result;result.input=as_f32(f,actual);
  result.raw=f.one(c::Operation(c::RmsNorm{.epsilon=kNormEpsilon}),{result.input,as_f32(f,gamma)});
  result.stored=store(f,result.raw,storage_type);
  result.visible=as_f32(f,result.stored);return result;
}
template<class Projection>
inline HcPorts hc_pre_with_projection(sem::Fragment &f,Value residual,Value fn,Value scale,Value base,Value norm_weight,
                      c::DataType storage_type,
                      ProjectionOrder order,Projection &&project) {
  require_storage(storage_type);
  if(f.description(residual)!=sem::dense({1,kStreams,kHidden},storage_type))
    throw std::runtime_error("GLM HC residual differs from its four-stream explicit storage");
  HcPorts result;result.residual=residual;result.projection_order=order;
  const auto visible=as_f32(f,residual);
  result.flat=core::reshape(f,visible,{1,kStreams*kHidden});
  // The fixed06992 default scales its actual raw projection by inverse RMS.
  // Fixed MLX-VLM f1a9b7 instead normalizes the actual F32 input first. Keep
  // these different finite-precision boundaries explicit at cold construction.
  Value projection_input=result.flat;
  if(order==ProjectionOrder::kNormalizeInput) {
    result.normalized_input=f.one(c::Operation(c::RmsNorm{.epsilon=kNormEpsilon}),{result.flat});
    projection_input=*result.normalized_input;
  } else if(order!=ProjectionOrder::kScaleRawProjection) {
    throw std::runtime_error("GLM HC projection order is not supported");
  }
  result.projection=project(f,projection_input,fn);
  if(f.description(result.projection)!=sem::dense({1,24}))
    throw std::runtime_error("GLM HC projection must expose its real F32 [1,24] result");
  if(order==ProjectionOrder::kScaleRawProjection) {
    result.square=core::binary(f,c::BinaryKind::kMultiply,result.flat,result.flat);
    result.mean=reduce(f,result.square,-1,c::ReductionKind::kMean);
    result.inverse=core::unary(f,c::UnaryKind::kRsqrt,
        core::binary(f,c::BinaryKind::kAdd,result.mean,core::filled(f,sem::dense({1}),kNormEpsilon)));
    result.mixes=core::binary(f,c::BinaryKind::kMultiply,result.projection,result.inverse);
  } else result.mixes=result.projection;
  result.combination=f.one(c::model_ops::hyper_connection_combine(kHcEpsilon,20),{result.mixes,scale,base});
  const auto gate=[&](std::int64_t first,std::int64_t scale_index) {
    const auto affine=core::binary(f,c::BinaryKind::kAdd,
        core::binary(f,c::BinaryKind::kMultiply,slice(f,result.mixes,1,first,kStreams),slice(f,scale,0,scale_index,1)),
        core::reshape(f,slice(f,base,0,first,kStreams),{1,kStreams}));
    return core::unary(f,c::UnaryKind::kSigmoid,affine);
  };
  result.pre=core::binary(f,c::BinaryKind::kAdd,gate(0,0),core::filled(f,sem::dense({1}),kHcEpsilon));
  result.post=core::binary(f,c::BinaryKind::kMultiply,gate(kStreams,1),core::filled(f,sem::dense({1}),2.0F));
  result.collapsed_raw=f.one(c::model_ops::hyper_connection_pre(),{visible,result.pre});
  result.collapsed_stored=store(f,result.collapsed_raw,storage_type);
  // The default retains fixed06992 _mhc_pre_torch + separate learned RMS.
  // The normalized-input branch instead keeps the actual F32 collapse before
  // learned RMS, as in the fixed MLX single-token op path.
  result.norm=norm(f,result.collapsed_stored,norm_weight,storage_type);return result;
}
inline HcPorts hc_pre(sem::Fragment &f,Value residual,Value fn,Value scale,Value base,Value norm_weight,
                      c::DataType storage_type=c::DataType::kBFloat16,
                      ProjectionOrder order=ProjectionOrder::kScaleRawProjection) {
  const auto project=[](sem::Fragment &graph,Value actual_input,Value mixing) {
  if(graph.description(mixing).data_type()==c::DataType::kBFloat16) {
    return graph.one(c::model_ops::dense_linear_f32(),{actual_input,mixing});
  } else {
    if(graph.description(mixing).data_type()!=c::DataType::kFloat32)
      throw std::runtime_error("GLM HC mixing coefficients must retain actual BF16 or F32 storage");
    const auto transpose=graph.one(c::Operation(c::View::permute(std::array<std::int32_t,2>{1,0}).value()),{mixing});
    return graph.one(c::Operation(c::MatMul{}),{actual_input,transpose});
  }
  };
  return hc_pre_with_projection(f,residual,fn,scale,base,norm_weight,storage_type,order,project);
}
inline PostPorts hc_post(sem::Fragment &f,const HcPorts &pre,Value residual,Value branch,
                        c::DataType storage_type=c::DataType::kBFloat16) {
  require_storage(storage_type);
  if(f.description(residual).data_type()!=storage_type||f.description(branch).data_type()!=storage_type)
    throw std::runtime_error("GLM HC post operands differ from explicit activation storage");
  PostPorts result;result.branch=branch;result.residual=residual;
  result.raw=f.one(c::model_ops::hyper_connection_post(),
      {as_f32(f,branch),as_f32(f,residual),pre.post,pre.combination});
  result.stored=store(f,result.raw,storage_type);
  result.visible=as_f32(f,result.stored);return result;
}
} // namespace glm_hyper_connection_fragment
