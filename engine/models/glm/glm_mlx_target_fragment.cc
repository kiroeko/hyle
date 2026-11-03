#include "glm_mlx_target_fragment.h"

namespace glm_mlx_target_fragment {
void require(sem::Fragment &f,Value value,const c::TensorDesc &expected) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=expected)
    throw std::runtime_error("MLX GLM target actual role or state descriptor differs");
}
}

namespace glm_mlx_target_fragment {
void require_constant(sem::Fragment &f,Value value,const c::TensorDesc &expected) {
  require(f,value,expected);
  for(const auto constant:f.constants)if(constant.ordinal()==value.ordinal())return;
  throw std::runtime_error("MLX GLM target parameters must be already-bound immutable constants");
}
}

namespace glm_mlx_target_fragment {
void require_hc(sem::Fragment &f,const HcParameters &p) {
  require_constant(f,p.fn,sem::dense({24,hc::kStreams*hc::kHidden},c::DataType::kBFloat16));
  require_constant(f,p.scale,sem::dense({3}));require_constant(f,p.base,sem::dense({24}));
  require_constant(f,p.norm,sem::dense({hc::kHidden},c::DataType::kBFloat16));
}
}

namespace glm_mlx_target_fragment {
Prepared43 prepare43(sem::Fragment &f,Value actual_four_stream_f32,const Parameters43 &p) {
  require(f,actual_four_stream_f32,sem::dense({1,hc::kStreams,hc::kHidden}));
  require_hc(f,p.attention);require_hc(f,p.ffn);
  Prepared43 result;
  result.attention_hc=hc::hc_pre(f,actual_four_stream_f32,p.attention.fn,p.attention.scale,
      p.attention.base,p.attention.norm,c::DataType::kFloat32,hc::ProjectionOrder::kNormalizeInput);
  result.ffn_parameters=p.ffn;result.input=result.attention_hc.norm.stored;return result;
}
}

namespace glm_mlx_target_fragment {
ConvolutionPreparation prepare_convolutions(sem::Fragment &f,const std::array<Value,3> &raw) {
  ConvolutionPreparation result;result.inputs=raw;
  for(std::size_t i=0;i<raw.size();++i) {
    require_constant(f,raw[i],sem::dense({target44::kWidth,1,4},c::DataType::kBFloat16));
    result.outputs[i]=hc::compact(f,hyle::inference::model_definition::ops::reshape(
        f,hc::as_f32(f,raw[i]),{target44::kWidth,4}));
  }
  return result;
}
}
