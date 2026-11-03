#include "glm_original_target43_fragment.h"

namespace glm_original_target43_fragment {
void require(sem::Fragment &f,Value value,const c::TensorDesc &expected) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=expected)
    throw std::runtime_error("original GLM43 actual activation/parameter geometry differs");
}
}

namespace glm_original_target43_fragment {
void require_parameters(sem::Fragment &f,const HcParameters &parameters) {
  const std::array values{parameters.fn,parameters.scale,parameters.base,parameters.norm};
  const std::array descriptions{
      sem::dense({24,kStreams*kHidden}),sem::dense({3}),sem::dense({24}),
      sem::dense({kHidden},c::DataType::kBFloat16)};
  for(std::size_t slot=0;slot<values.size();++slot) {
    require(f,values[slot],descriptions[slot]);
    bool immutable=false;
    for(const auto value:f.constants)if(value.ordinal()==values[slot].ordinal())immutable=true;
    if(!immutable)throw std::runtime_error("original GLM43 HC parameters must be already-bound immutable constants");
  }
}
}

namespace glm_original_target43_fragment {
Prepared prepare(sem::Fragment &f,Value actual_four_stream_bf16,const Parameters &parameters) {
  require(f,actual_four_stream_bf16,sem::dense({1,kStreams,kHidden},c::DataType::kBFloat16));
  require_parameters(f,parameters.attention);require_parameters(f,parameters.ffn);
  Prepared result;
  const auto &attention=parameters.attention;
  result.attention_hc=hc::hc_pre(f,actual_four_stream_bf16,
      attention.fn,attention.scale,attention.base,attention.norm);
  result.ffn_parameters=parameters.ffn;
  result.input=result.attention_hc.norm.stored;
  return result;
}
}

namespace glm_original_target43_fragment {
Result finish(sem::Fragment &f,const Prepared &prepared,Value actual_mla_output_bf16,
                     const moe_fragment::Parameters &moe_parameters,
                     glm_fp8_arithmetic::Arithmetic arithmetic,
                     moe_fragment::WeightStorage storage) {
  require(f,prepared.attention_hc.residual,sem::dense({1,kStreams,kHidden},c::DataType::kBFloat16));
  require(f,prepared.input,sem::dense({1,kHidden},c::DataType::kBFloat16));
  require(f,actual_mla_output_bf16,sem::dense({1,kHidden},c::DataType::kBFloat16));
  require_parameters(f,prepared.ffn_parameters);
  Result result;auto &ports=result.ports;
  ports.attention_hc=prepared.attention_hc;
  ports.attention_post=hc::hc_post(f,ports.attention_hc,
      ports.attention_hc.residual,actual_mla_output_bf16);
  const auto &ffn=prepared.ffn_parameters;
  ports.ffn_hc=hc::hc_pre(f,ports.attention_post.stored,ffn.fn,ffn.scale,ffn.base,ffn.norm);
  result.moe=moe_fragment::append(f,ports.ffn_hc.norm.stored,moe_parameters,moe_fragment::kFlashGeometry,arithmetic,storage);
  ports.ffn_post=hc::hc_post(f,ports.ffn_hc,ports.attention_post.stored,result.moe.output);
  result.stored=ports.ffn_post.stored;result.mhc=ports.ffn_post.visible;
  return result;
}
}
