#pragma once

#include <array>
#include <stdexcept>

#include "glm_hyper_connection_fragment.h"
#include "glm_original_moe_fragment.h"

// Pure original GLM Flash layer43 shell. The caller appends this layer's
// actual MLA/indexer between prepare and finish in the same Fragment. This
// non-final target layer forwards four BF16 streams; it has no stream mean,
// model.norm, cache owner, parameter loader, provider choice or executor.
namespace glm_original_target43_fragment {
namespace c=hyle::compute;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace hc=glm_hyper_connection_fragment;
namespace moe_fragment=glm_original_moe_fragment;
using Value=sem::ValueId;
inline constexpr auto kHidden=hc::kHidden,kStreams=hc::kStreams;

struct HcParameters final {
  Value fn,scale,base,norm;
};
struct Parameters final {
  HcParameters attention,ffn;
};
struct Prepared final {
  hc::HcPorts attention_hc;
  HcParameters ffn_parameters;
  Value input; // Actual collapsed/learned-RMS BF16[1,4096], sole MLA source.
};
struct Ports final {
  hc::HcPorts attention_hc,ffn_hc;
  hc::PostPorts attention_post,ffn_post;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    attention_hc.visit_values(visitor);ffn_hc.visit_values(visitor);
    attention_post.visit_values(visitor);ffn_post.visit_values(visitor);
  }
};
struct Result final {
  Value mhc,stored; // Exact F32 ABI carrier and its actual BF16 four streams.
  Ports ports;
  moe_fragment::Result moe;
};
inline void require(sem::Fragment &f,Value value,const c::TensorDesc &expected) {
  if(value.ordinal()>=f.descriptions.size()||f.description(value)!=expected)
    throw std::runtime_error("original GLM43 actual activation/parameter geometry differs");
}
inline void require_parameters(sem::Fragment &f,const HcParameters &parameters) {
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
inline Prepared prepare(sem::Fragment &f,Value actual_four_stream_bf16,const Parameters &parameters) {
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
inline Result finish(sem::Fragment &f,const Prepared &prepared,Value actual_mla_output_bf16,
                     const moe_fragment::Parameters &moe_parameters,
                     glm_fp8_arithmetic::Arithmetic arithmetic={},
                     moe_fragment::WeightStorage storage=moe_fragment::WeightStorage::kFloat8Block128) {
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
} // namespace glm_original_target43_fragment
