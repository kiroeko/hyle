#include "glm_original_target44_fragment.h"

namespace glm_original_target44_fragment {
ProjectionPorts project(sem::Fragment &f,Value input,Value weight,
                               c::DataType storage_type) {
  hc::require_storage(storage_type);
  if(f.description(input).data_type()!=storage_type||f.description(weight).data_type()!=c::DataType::kBFloat16)
    throw std::runtime_error("GLM KDA projection requires actual BF16 weights and declared activation storage");
  ProjectionPorts result;result.input=input;
  result.raw=f.one(c::model_ops::dense_linear_f32(),{input,weight});
  result.stored=hc::store(f,result.raw,storage_type);return result;
}
}

namespace glm_original_target44_fragment {
L2Ports l2(sem::Fragment &f,Value actual) {
  L2Ports result;result.input=hc::as_f32(f,actual);
  result.square=core::binary(f,c::BinaryKind::kMultiply,result.input,result.input);
  result.sum=reduce(f,result.square,-1,c::ReductionKind::kSum);
  result.denominator=core::binary(f,c::BinaryKind::kAdd,result.sum,core::filled(f,sem::dense({1}),kL2Epsilon));
  result.inverse=core::unary(f,c::UnaryKind::kRsqrt,result.denominator);
  result.output=core::binary(f,c::BinaryKind::kMultiply,result.input,result.inverse);
  return result; // F32 in-kernel Q/K, deliberately no extra BF16 store.
}
}

namespace glm_original_target44_fragment {
L2Ports mlx_query_key(sem::Fragment &f,Value actual,bool query) {
  L2Ports result;result.normalization=QueryKeyNormalization::kMlxRmsPreScaled;
  result.input=hc::as_f32(f,actual);
  result.normalized_raw=f.one(c::Operation(c::RmsNorm{.epsilon=kL2Epsilon/float(kDepth)}),{result.input});
  const float scale=query?1.0F/float(kDepth):static_cast<float>(1.0/std::sqrt(double(kDepth)));
  result.output=core::binary(f,c::BinaryKind::kMultiply,*result.normalized_raw,core::filled(f,sem::dense({1}),scale));
  return result;
}
}

namespace glm_original_target44_fragment {
KdaPorts kda(sem::Fragment &f,Value input,Value history,Value state,const ParameterValues &parameters) {
  return kda_with_recurrent(f,input,history,state,parameters,c::DataType::kBFloat16,
      QueryKeyNormalization::kL2ForScaledGdn,scaled_recurrent);
}
}

namespace glm_original_target44_fragment {
Result append(sem::Fragment &f,Value actual_four_stream_bf16,Value history_f32,Value matrix_f32,
                      const ParameterValues &parameters,const moe_fragment::Parameters &moe_parameters,
                      glm_fp8_arithmetic::Arithmetic arithmetic,
                      moe_fragment::WeightStorage storage) {
  const auto moe_builder=[&](sem::Fragment &graph,Value actual) {
    return moe_fragment::append(graph,actual,moe_parameters,moe_fragment::kFlashGeometry,arithmetic,storage);
  };
  return append_with_moe(f,actual_four_stream_bf16,history_f32,matrix_f32,parameters,
      c::DataType::kBFloat16,hc::ProjectionOrder::kScaleRawProjection,
      QueryKeyNormalization::kL2ForScaledGdn,scaled_recurrent,moe_builder);
}
}
