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
namespace sem=hyle::inference::model_definition;
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
void require(sem::Fragment &f,Value value,const c::TensorDesc &expected);
void require_parameters(sem::Fragment &f,const HcParameters &parameters);
Prepared prepare(sem::Fragment &f,Value actual_four_stream_bf16,const Parameters &parameters);
Result finish(sem::Fragment &f,const Prepared &prepared,Value actual_mla_output_bf16,
                     const moe_fragment::Parameters &moe_parameters,
                     glm_fp8_arithmetic::Arithmetic arithmetic={},
                     moe_fragment::WeightStorage storage=moe_fragment::WeightStorage::kFloat8Block128);
} // namespace glm_original_target43_fragment
