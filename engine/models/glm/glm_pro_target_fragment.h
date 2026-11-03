#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "glm_pro_mtp_fragment.h"

// Original last main layer77. Its shared DSA selection is an explicit input
// from layer74, carried through75/76; it has NO trained local indexer. Inputs
// hidden/residual name the separate BF16 outputs at the layer76 boundary.
// No loading, execution, provider selection or acceptance policy lives here.
namespace glm_pro_target_fragment {
namespace graph=glm_pro_mtp_fragment;
namespace sem=graph::sem;
namespace c=hyle::compute;
using Value=graph::Value;
struct Parameters final {
  graph::MlaParameters attention;
  Value input_norm,query_a,query_norm,target_norm;
};
struct Ports final {
  Value hidden,residual,main_input,selection,position,input_sum,input_residual_stored;
  graph::NormPorts input_norm,query_norm,final_norm;
  graph::ProjectionPorts query;
  graph::AttentionPorts attention;
  Value post_ffn;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&hidden,&residual,&main_input,&selection,&position,&input_sum,&input_residual_stored,&post_ffn})visitor(*value);
    input_norm.visit_values(visitor);query_norm.visit_values(visitor);final_norm.visit_values(visitor);
    query.visit_values(visitor);attention.visit_values(visitor);
  }
};
struct Result final {
  Value main,cursor,normalized_tap,raw_post_ffn;
  Ports ports;
  graph::moe::Result feed_forward;
};
Result append(sem::Fragment &f,Value hidden,Value residual,Value old_main,
    Value upstream_selection,Value cursor,const Parameters &p,
    const graph::moe::Parameters &moe);
} // namespace glm_pro_target_fragment
