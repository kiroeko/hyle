#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include "glm_hyper_connection_fragment.h"
#include "glm_original_moe_fragment.h"
#include "glm_original_target44_roles.h"

// Pure original GLM Flash layer44 construction. The caller binds distinct
// authenticated44 constants, compact selected expert banks and actual state.
// HC uses fixed06992's explicit torch/unfused path, including BF16 collapse
// before learned RMS; it does not claim TileLang fused-norm bit equivalence.
// KDA normalization is F32 inside the recurrent boundary. The original wrapper
// retains BF16 convolution/recurrent/output-gate stores; the explicit MLX
// constructor retains F32 stores and its own q/k and recurrence boundaries.
// There is no loader/provider/executor here.
namespace glm_original_target44_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace roles=glm_original_target44_roles;
namespace moe_fragment=glm_original_moe_fragment;
namespace hc=glm_hyper_connection_fragment;
using Value=sem::ValueId;
using Role=roles::Role;
using ParameterValues=std::array<Value,static_cast<std::size_t>(Role::kCount)>;
inline constexpr std::int64_t kHidden=4096,kStreams=4,kHeads=64,kDepth=128;
inline constexpr std::int64_t kWidth=kHeads*kDepth,kChannels=3*kWidth;
using hc::kNormEpsilon;
using hc::kHcEpsilon;
inline constexpr float kL2Epsilon=1.0e-6F;

using hc::NormPorts;
using hc::HcPorts;
using hc::PostPorts;
struct ProjectionPorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored})visitor(*value);
  }
};
struct ConvPorts final {
  Value samples,raw,activated_raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&samples,&raw,&activated_raw,&stored})visitor(*value);
  }
};
enum class QueryKeyNormalization {kL2ForScaledGdn,kMlxRmsPreScaled};
// Actual source representation, not a conversion back through log/exp.
enum class DecayParameter {kLogPositiveRate,kNegativeExpRate};
struct L2Ports final {
  Value input,square,sum,denominator,inverse,output;
  QueryKeyNormalization normalization=QueryKeyNormalization::kL2ForScaledGdn;
  std::optional<Value> normalized_raw;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visitor(input);
    if(normalization==QueryKeyNormalization::kL2ForScaledGdn)
      for(auto *value:{&square,&sum,&denominator,&inverse})visitor(*value);
    if(normalized_raw)visitor(*normalized_raw);
    visitor(output);
  }
};
enum class Projection : std::size_t {kQuery,kKey,kValue,kForgetA,kForgetB,kBeta,kGateA,kGateB,kOutput,kCount};
struct KdaPorts final {
  std::array<ProjectionPorts,static_cast<std::size_t>(Projection::kCount)> projections;
  std::array<ConvPorts,3> conv; // Q,K,V in original checkpoint role order.
  Value history_input,history_stored,window,next_history;
  L2Ports query,key;
  Value value,alpha,forget_add,forget_product,forget_sigmoid,decay,beta;
  DecayParameter decay_parameter=DecayParameter::kLogPositiveRate;
  std::optional<Value> negative_rate_sigmoid_input;
  Value state_input,recurrent_raw,recurrent_stored,state_output;
  Value output_norm_input,output_norm_raw,gate_sigmoid,gated_raw,gated_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto &projection:projections)projection.visit_values(visitor);
    for(auto &part:conv)part.visit_values(visitor);
    for(auto *v:{&history_input,&history_stored,&window,&next_history})visitor(*v);
    query.visit_values(visitor);key.visit_values(visitor);
    if(negative_rate_sigmoid_input)visitor(*negative_rate_sigmoid_input);
    for(auto *v:{&value,&alpha,&forget_add,&forget_product,&forget_sigmoid,&decay,&beta,
        &state_input,&recurrent_raw,&recurrent_stored,&state_output,&output_norm_input,&output_norm_raw,&gate_sigmoid,&gated_raw,&gated_stored})visitor(*v);
  }
};
struct Ports final {
  HcPorts attention_hc,ffn_hc;
  PostPorts attention_post,ffn_post;
  KdaPorts kda;
  Value contract_raw,contract_stored;
  NormPorts model_norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    attention_hc.visit_values(visitor);ffn_hc.visit_values(visitor);
    attention_post.visit_values(visitor);ffn_post.visit_values(visitor);kda.visit_values(visitor);
    visitor(contract_raw);visitor(contract_stored);model_norm.visit_values(visitor);
  }
};
template<class MoeResult> struct LayerResult final {
  Value history,state,mhc,tap; // F32 ABI carriers; state itself is recurrentF32.
  Ports ports;
  MoeResult moe;
};
using Result=LayerResult<moe_fragment::Result>;
using hc::cast;
using hc::slice;
using hc::compact;
using hc::reduce;
using hc::norm;
using hc::hc_pre;
using hc::hc_post;
inline ProjectionPorts project(sem::Fragment &f,Value input,Value weight,
                               c::DataType storage_type=c::DataType::kBFloat16) {
  hc::require_storage(storage_type);
  if(f.description(input).data_type()!=storage_type||f.description(weight).data_type()!=c::DataType::kBFloat16)
    throw std::runtime_error("GLM KDA projection requires actual BF16 weights and declared activation storage");
  ProjectionPorts result;result.input=input;
  result.raw=f.one(c::model_ops::bfloat_linear_f32(),{input,weight});
  result.stored=hc::store(f,result.raw,storage_type);return result;
}
inline L2Ports l2(sem::Fragment &f,Value actual) {
  L2Ports result;result.input=hc::as_f32(f,actual);
  result.square=core::binary(f,c::BinaryKind::kMultiply,result.input,result.input);
  result.sum=reduce(f,result.square,-1,c::ReductionKind::kSum);
  result.denominator=core::binary(f,c::BinaryKind::kAdd,result.sum,core::filled(f,sem::dense({1}),kL2Epsilon));
  result.inverse=core::unary(f,c::UnaryKind::kRsqrt,result.denominator);
  result.output=core::binary(f,c::BinaryKind::kMultiply,result.input,result.inverse);
  return result; // F32 in-kernel Q/K, deliberately no extra BF16 store.
}
// MLX stores the actual F32 RMS result before its explicit q/k scalar multiply.
// Its recurrent primitive then has no additional query scaling. The callback
// must implement that exact contract; the default public scaled GDN is not a
// substitute and no reciprocal compensation is inserted here.
inline L2Ports mlx_query_key(sem::Fragment &f,Value actual,bool query) {
  L2Ports result;result.normalization=QueryKeyNormalization::kMlxRmsPreScaled;
  result.input=hc::as_f32(f,actual);
  result.normalized_raw=f.one(c::Operation(c::RmsNorm{.epsilon=kL2Epsilon/float(kDepth)}),{result.input});
  const float scale=query?1.0F/float(kDepth):static_cast<float>(1.0/std::sqrt(double(kDepth)));
  result.output=core::binary(f,c::BinaryKind::kMultiply,*result.normalized_raw,core::filled(f,sem::dense({1}),scale));
  return result;
}
template<class Recurrent,class Project>
inline KdaPorts kda_with_projection(sem::Fragment &f,Value input,Value history,Value state,
    const ParameterValues &parameters,c::DataType storage_type,
    QueryKeyNormalization normalization,Recurrent &&recurrent_builder,Project &&project_builder,
    DecayParameter decay_parameter,float lower_bound) {
  const auto parameter=[&](Role role){return parameters[static_cast<std::size_t>(role)];};
  hc::require_storage(storage_type);
  if(!std::isfinite(lower_bound)||!(lower_bound<0.F))
    throw std::runtime_error("GLM KDA requires its declared finite negative gate lower bound");
  if(decay_parameter!=DecayParameter::kLogPositiveRate&&decay_parameter!=DecayParameter::kNegativeExpRate)
    throw std::runtime_error("GLM KDA decay parameter representation is unsupported");
  KdaPorts result;result.history_input=history;result.state_input=state;result.decay_parameter=decay_parameter;
  const auto projection=[&](Projection slot,Value source,Role role) {
    auto &port=result.projections[static_cast<std::size_t>(slot)];port=project_builder(f,source,parameter(role),storage_type);return port.stored;
  };
  const std::array qkv{projection(Projection::kQuery,input,Role::kQuery),projection(Projection::kKey,input,Role::kKey),projection(Projection::kValue,input,Role::kValue)};
  const auto forget_a=projection(Projection::kForgetA,input,Role::kForgetA);
  const auto forget_b=projection(Projection::kForgetB,forget_a,Role::kForgetB);
  const auto beta_raw=projection(Projection::kBeta,input,Role::kBeta);
  const auto gate_a=projection(Projection::kGateA,input,Role::kGateA);
  const auto gate_b=projection(Projection::kGateB,gate_a,Role::kGateB);
  result.history_stored=hc::store(f,history,storage_type);
  const auto joined=f.one(c::Operation(c::Concat{.axis=1}),{qkv[0],qkv[1],qkv[2]});
  result.window=f.one(c::Operation(c::Concat{.axis=0}),{result.history_stored,joined});
  result.next_history=compact(f,slice(f,result.window,0,1,3));
  const std::array convolution_roles{Role::kQueryConv,Role::kKeyConv,Role::kValueConv};
  for(std::size_t leg=0;leg<result.conv.size();++leg) {
    auto &conv=result.conv[leg];
    conv.samples=core::reshape(f,compact(f,slice(f,result.window,1,std::int64_t(leg)*kWidth,kWidth)),{1,4,kWidth});
    conv.raw=f.one(c::model_ops::causal_conv1d(),{hc::as_f32(f,conv.samples),parameter(convolution_roles[leg])});
    conv.activated_raw=core::unary(f,c::UnaryKind::kSilu,conv.raw);
    conv.stored=core::reshape(f,hc::store(f,conv.activated_raw,storage_type),{1,1,kHeads,kDepth});
  }
  if(normalization==QueryKeyNormalization::kL2ForScaledGdn) {
    result.query=l2(f,result.conv[0].stored);result.key=l2(f,result.conv[1].stored);
  } else if(normalization==QueryKeyNormalization::kMlxRmsPreScaled) {
    result.query=mlx_query_key(f,result.conv[0].stored,true);
    result.key=mlx_query_key(f,result.conv[1].stored,false);
  } else throw std::runtime_error("GLM KDA query/key normalization contract is not supported");
  result.value=hc::as_f32(f,result.conv[2].stored);
  const auto rate=decay_parameter==DecayParameter::kLogPositiveRate?
      core::unary(f,c::UnaryKind::kExp,parameter(Role::kALog)):parameter(Role::kALog);
  result.alpha=core::reshape(f,rate,{1,1,kHeads,1});
  const auto forget=core::reshape(f,hc::as_f32(f,forget_b),{1,1,kHeads,kDepth});
  result.forget_add=core::binary(f,c::BinaryKind::kAdd,forget,core::reshape(f,parameter(Role::kDtBias),{1,1,kHeads,kDepth}));
  result.forget_product=core::binary(f,c::BinaryKind::kMultiply,result.forget_add,result.alpha);
  Value sigmoid_input=result.forget_product;
  if(decay_parameter==DecayParameter::kNegativeExpRate) {
    result.negative_rate_sigmoid_input=core::binary(f,c::BinaryKind::kMultiply,
        result.forget_product,core::filled(f,sem::dense({1}),-1.F));
    sigmoid_input=*result.negative_rate_sigmoid_input;
  }
  result.forget_sigmoid=core::unary(f,c::UnaryKind::kSigmoid,sigmoid_input);
  result.decay=core::binary(f,c::BinaryKind::kMultiply,result.forget_sigmoid,core::filled(f,sem::dense({1}),lower_bound));
  result.beta=core::unary(f,c::UnaryKind::kSigmoid,core::reshape(f,hc::as_f32(f,beta_raw),{1,1,kHeads,1}));
  // Public GDN and the selected fixed06992 kernel store [H,V,K]. Keep that
  // orientation across accepted states; an external format needs a transpose
  // only if it explicitly declares a different [H,K,V] layout. Q/K and the
  // recurrent accumulation remain F32.
  const auto recurrent=recurrent_builder(f,result.query.output,result.key.output,
      result.value,result.decay,result.beta,state);
  if(f.description(recurrent[0])!=sem::dense({1,1,kHeads,kDepth})||
      f.description(recurrent[1])!=sem::dense({1,kHeads,kDepth,kDepth}))
    throw std::runtime_error("GLM KDA recurrent builder must expose real F32 output and state");
  result.recurrent_raw=recurrent[0];result.state_output=recurrent[1];
  result.recurrent_stored=hc::store(f,result.recurrent_raw,storage_type);
  result.output_norm_input=hc::as_f32(f,result.recurrent_stored);
  result.output_norm_raw=f.one(c::Operation(c::RmsNorm{.epsilon=kNormEpsilon}),
      {result.output_norm_input,hc::as_f32(f,parameter(Role::kOutputNorm))});
  result.gate_sigmoid=core::unary(f,c::UnaryKind::kSigmoid,core::reshape(f,hc::as_f32(f,gate_b),{1,1,kHeads,kDepth}));
  // Fused source output RMS/sigmoid gate has one final BF16 store, not a
  // separately rounded intermediate RMS output.
  result.gated_raw=core::binary(f,c::BinaryKind::kMultiply,result.output_norm_raw,result.gate_sigmoid);
  result.gated_stored=hc::store(f,result.gated_raw,storage_type);
  projection(Projection::kOutput,core::reshape(f,result.gated_stored,{1,kWidth}),Role::kOutput);
  return result;
}
template<class Recurrent>
inline KdaPorts kda_with_recurrent(sem::Fragment &f,Value input,Value history,Value state,
    const ParameterValues &parameters,c::DataType storage_type,
    QueryKeyNormalization normalization,Recurrent &&recurrent_builder) {
  const auto ordinary=[](sem::Fragment &graph,Value source,Value weight,c::DataType storage) {
    return project(graph,source,weight,storage);
  };
  return kda_with_projection(f,input,history,state,parameters,storage_type,normalization,
      std::forward<Recurrent>(recurrent_builder),ordinary,DecayParameter::kLogPositiveRate,-5.F);
}
inline auto scaled_recurrent(sem::Fragment &f,Value query,Value key,Value value,
                             Value decay,Value beta,Value state) {
  const auto result=f.append(c::Operation(c::GatedDeltaNet{}),{query,key,value,decay,beta,state},2);
  return std::array{result[0],result[1]};
}
inline KdaPorts kda(sem::Fragment &f,Value input,Value history,Value state,const ParameterValues &parameters) {
  return kda_with_recurrent(f,input,history,state,parameters,c::DataType::kBFloat16,
      QueryKeyNormalization::kL2ForScaledGdn,scaled_recurrent);
}
template<class Recurrent,class MoeBuilder>
inline auto append_with_moe(sem::Fragment &f,Value actual_four_streams,Value history_f32,Value matrix_f32,
    const ParameterValues &parameters,c::DataType storage_type,hc::ProjectionOrder hc_order,
    QueryKeyNormalization normalization,Recurrent &&recurrent_builder,MoeBuilder &&moe_builder) {
  hc::require_storage(storage_type);
  const auto require=[&](Value value,const c::TensorDesc &expected) {
    if(value.ordinal()>=f.descriptions.size()||f.description(value)!=expected)
      throw std::runtime_error("GLM44 actual parameter/activation/state geometry differs");
  };
  require(actual_four_streams,sem::dense({1,kStreams,kHidden},storage_type));
  require(history_f32,sem::dense({3,kChannels}));require(matrix_f32,sem::dense({1,kHeads,kDepth,kDepth}));
  for(std::size_t slot=0;slot<parameters.size();++slot) {
    const auto role=static_cast<Role>(slot);
    auto expected=roles::description(role);
    // MLX retains original BF16 HC banks instead of a permanent F32 shadow.
    if(storage_type==c::DataType::kFloat32&&roles::is_hc(role)&&
        f.description(parameters[slot]).data_type()==c::DataType::kBFloat16)
      expected=roles::raw_description(role);
    require(parameters[slot],expected);
    bool immutable=false;for(auto value:f.constants)if(value.ordinal()==parameters[slot].ordinal())immutable=true;
    if(!immutable)throw std::runtime_error("GLM44 parameters must be immutable already-bound Graph constants");
  }
  const auto parameter=[&](Role role){return parameters[static_cast<std::size_t>(role)];};
  using MoeResult=decltype(moe_builder(f,actual_four_streams));
  LayerResult<MoeResult> result;auto &ports=result.ports;
  ports.attention_hc=hc_pre(f,actual_four_streams,parameter(Role::kHcAttnFn),parameter(Role::kHcAttnScale),
      parameter(Role::kHcAttnBase),parameter(Role::kInputNorm),storage_type,hc_order);
  ports.kda=kda_with_recurrent(f,ports.attention_hc.norm.stored,history_f32,matrix_f32,parameters,
      storage_type,normalization,std::forward<Recurrent>(recurrent_builder));
  const auto branch=ports.kda.projections[static_cast<std::size_t>(Projection::kOutput)].stored;
  ports.attention_post=hc_post(f,ports.attention_hc,actual_four_streams,branch,storage_type);
  ports.ffn_hc=hc_pre(f,ports.attention_post.stored,parameter(Role::kHcFfnFn),parameter(Role::kHcFfnScale),
      parameter(Role::kHcFfnBase),parameter(Role::kPostNorm),storage_type,hc_order);
  result.moe=moe_builder(f,ports.ffn_hc.norm.stored);
  ports.ffn_post=hc_post(f,ports.ffn_hc,ports.attention_post.stored,result.moe.output,storage_type);
  ports.contract_raw=reduce(f,ports.ffn_post.visible,1,c::ReductionKind::kMean,false);
  ports.contract_stored=hc::store(f,ports.contract_raw,storage_type);
  ports.model_norm=norm(f,ports.contract_stored,parameter(Role::kModelNorm),storage_type);
  result.history=hc::as_f32(f,ports.kda.next_history);
  result.state=ports.kda.state_output;result.mhc=ports.ffn_post.visible;result.tap=ports.model_norm.visible;
  return result;
}
inline Result append(sem::Fragment &f,Value actual_four_stream_bf16,Value history_f32,Value matrix_f32,
                      const ParameterValues &parameters,const moe_fragment::Parameters &moe_parameters,
                      glm_fp8_arithmetic::Arithmetic arithmetic={},
                      moe_fragment::WeightStorage storage=moe_fragment::WeightStorage::kFloat8Block128) {
  const auto moe_builder=[&](sem::Fragment &graph,Value actual) {
    return moe_fragment::append(graph,actual,moe_parameters,moe_fragment::kFlashGeometry,arithmetic,storage);
  };
  return append_with_moe(f,actual_four_stream_bf16,history_f32,matrix_f32,parameters,
      c::DataType::kBFloat16,hc::ProjectionOrder::kScaleRawProjection,
      QueryKeyNormalization::kL2ForScaledGdn,scaled_recurrent,moe_builder);
}
} // namespace glm_original_target44_fragment
