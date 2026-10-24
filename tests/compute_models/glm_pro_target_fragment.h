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
inline Result append(sem::Fragment &f,Value hidden,Value residual,Value old_main,
    Value upstream_selection,Value cursor,const Parameters &p,
    const graph::moe::Parameters &moe) {
  graph::require(f,hidden,sem::dense({1,graph::kHidden},c::DataType::kBFloat16));
  graph::require(f,residual,sem::dense({1,graph::kHidden},c::DataType::kBFloat16));
  graph::require(f,upstream_selection,sem::dense({2048},c::DataType::kInt32));
  graph::require(f,cursor,sem::dense({1},c::DataType::kInt32));
  Result result;auto &ports=result.ports;
  ports.hidden=hidden;ports.residual=residual;ports.main_input=old_main;
  ports.selection=upstream_selection;ports.position=cursor;
  // Match RMSNorm.forward_native: normalize the raw F32 residual sum, and
  // independently return its BF16 residual. Do not round before the RMS.
  ports.input_sum=graph::core::binary(f,c::BinaryKind::kAdd,
      graph::cast(f,hidden,c::DataType::kFloat32),graph::cast(f,residual,c::DataType::kFloat32));
  ports.input_residual_stored=graph::cast(f,ports.input_sum,c::DataType::kBFloat16);
  ports.input_norm=graph::norm(f,ports.input_sum,p.input_norm,graph::kHidden);
  ports.query=graph::fp8(f,ports.input_norm.stored,p.query_a);
  ports.query_norm=graph::norm(f,graph::cast(f,ports.query.stored,c::DataType::kFloat32),p.query_norm,graph::kQueryRank);
  result.main=graph::append_mla_prefix(f,ports.input_norm.stored,ports.query_norm.stored,
      old_main,cursor,p.attention,ports.attention);
  graph::append_mla_suffix(f,result.main,upstream_selection,ports.input_residual_stored,p.attention,ports.attention);
  result.feed_forward=graph::moe::append(f,ports.attention.post_norm.stored,moe,glm_moe_geometry::kPro);
  ports.post_ffn=graph::core::binary(f,c::BinaryKind::kAdd,
      graph::cast(f,ports.attention.residual_stored,c::DataType::kFloat32),
      graph::cast(f,result.feed_forward.output,c::DataType::kFloat32));
  ports.final_norm=graph::norm(f,ports.post_ffn,p.target_norm,graph::kHidden);
  result.raw_post_ffn=ports.post_ffn;result.normalized_tap=ports.final_norm.stored;
  // Integer Scan is the same public increment used by the NextN controller.
  const auto one=f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{1}));
  const auto scanned=f.one(c::Operation(c::Scan{.axis=0}),
      {f.one(c::Operation(c::Concat{.axis=0}),{cursor,one})});
  result.cursor=f.one(c::Operation(c::Gather{}),{scanned,one});
  return result;
}
} // namespace glm_pro_target_fragment
