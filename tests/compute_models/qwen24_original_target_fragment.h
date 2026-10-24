#pragma once

#include "qwen_full_attention_fragment.h"
#include "qwen24_original_moe_fragment.h"

// Pure final main-model MoE block. The exact original attention/static banks
// and compact expert generations are ordinary Values owned by the caller.
namespace qwen24_original_target_fragment {
namespace attention=qwen_full_attention_fragment;
namespace moe=qwen24_original_moe_fragment;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
struct Parameters final {attention::TargetAttentionParameters attention;moe::Parameters feed_forward;Value target_norm;};
struct Result final {
  attention::TargetAttentionResult attention;
  moe::Result feed_forward;
  Value raw_post_ffn,normalized_tap;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    attention.ports.visit_values(visitor);feed_forward.ports.visit_values(visitor);
    visitor(raw_post_ffn);visitor(normalized_tap);
  }
};
inline Result append(sem::Fragment &fragment,Value input,Value key,Value value,Value cursor,Value position,
    Value frequency,const Parameters &parameters,const attention::Geometry &geometry) {
  Result result;result.attention=attention::append_target_attention(fragment,input,key,value,cursor,position,frequency,parameters.attention,geometry);
  result.feed_forward=moe::append(fragment,result.attention.ports.ffn_input,parameters.feed_forward);
  result.raw_post_ffn=core::binary(fragment,c::BinaryKind::kAdd,result.attention.ports.residual,result.feed_forward.output);
  attention::require(fragment,parameters.target_norm,sem::dense({geometry.hidden}));
  result.normalized_tap=fragment.one(c::Operation(c::RmsNorm{.epsilon=geometry.epsilon}),{result.raw_post_ffn,parameters.target_norm});
  return result;
}
} // namespace qwen24_original_target_fragment
