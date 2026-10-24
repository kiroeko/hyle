#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "qwen_artifact_projection_fragment.h"
#include "hyle/graph/executable.h"

// A cold public-operation leaf for original GPTQ-v1, shared by any matching
// source geometry. It plugs into append_target_with_projection; it does not
// own a loader, execution loop, model state or artifact identity.
namespace qwen_gptq_projection_fragment {
namespace attention=qwen_full_attention_fragment;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace sem=attention::sem;
using Value=attention::Value;
struct Parameters final {Value qweight,qzeros,scales,group_index;std::uint32_t group_size;};
struct Names final {std::string_view qweight,qzeros,scales,group_index;std::uint32_t group_size;};
struct Ports final {
  Value input;
  Parameters parameters;
  Value stored; // Actual BF16 MatMul return. No observable raw-F32 result.
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);visitor(stored);}
};
inline Value project(sem::Fragment &f,Value input,const Parameters &parameters,
    const attention::Geometry &geometry,std::vector<Ports> &observations) {
  const auto stored=qwen_artifact_projection_fragment::project(f,input,
      {parameters.qweight,parameters.qzeros,parameters.scales,parameters.group_index},geometry,
      {c::model_ops::WeightArtifactFormat::kGptqV1Int4,parameters.group_size,c::DataType::kFloat16});
  // Do NOT expose decoded/operand/rhs as diagnostic outputs. Their closed
  // Decode/Cast/View/MatMul chain is the existing packed-region contract.
  observations.push_back({input,parameters,stored});return stored;
}
struct ProjectionLeaf final {
  std::span<const Parameters> parameters;
  std::vector<Ports> *observations;
  Value operator()(sem::Fragment &f,Value input,Value code,const attention::Geometry &geometry,
      attention::LocalTrace *) const {
    if(!observations)throw std::runtime_error("GPTQ target needs its explicit stored-result observations");
    const auto matches=std::count_if(parameters.begin(),parameters.end(),[&](const auto &p){return p.qweight.ordinal()==code.ordinal();});
    if(matches!=1)throw std::runtime_error("GPTQ target needs exactly one original quartet for its actual qweight");
    const auto found=std::find_if(parameters.begin(),parameters.end(),[&](const auto &p){return p.qweight.ordinal()==code.ordinal();});
    return project(f,input,*found,geometry,*observations);
  }
};
inline constexpr std::size_t kProjections=7;
struct Result final {attention::TargetResult block;std::vector<Ports> projections;};
inline Result append_target(sem::Fragment &f,Value input,Value old_key,Value old_value,Value cursor,
    Value position,Value frequency,const attention::TargetParameters &supplied,const attention::Geometry &geometry,
    std::span<const Parameters> parameters) {
  if(parameters.size()!=kProjections)throw std::runtime_error("GPTQ target needs all seven actual original quartets");
  for(std::size_t i=0;i<parameters.size();++i)for(std::size_t j=0;j<i;++j)
    if(parameters[i].qweight.ordinal()==parameters[j].qweight.ordinal())
      throw std::runtime_error("GPTQ target needs seven distinct original qweight banks");
  Result result;result.projections.reserve(kProjections);
  result.block=attention::append_target_with_projection(f,input,old_key,old_value,cursor,position,frequency,
      supplied,geometry,std::nullopt,ProjectionLeaf{parameters,&result.projections});
  if(result.projections.size()!=kProjections)throw std::runtime_error("GPTQ target did not build seven original packed projections");
  return result;
}
inline constexpr auto kPackedPolicy=qwen_artifact_projection_fragment::kPackedPolicy;
// GPTQ's real F16-to-BF16 Cast remains mandatory: exactly four public nodes.
inline void require_packed(const g::CompileReport &report,std::span<const g::NodeId> matrices) {
  qwen_artifact_projection_fragment::require_packed(report,matrices,4);
}
} // namespace qwen_gptq_projection_fragment
