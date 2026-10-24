#pragma once

#include <algorithm>
#include <span>
#include <string_view>
#include <vector>
#include "qwen_artifact_projection_fragment.h"

// Original CompressedTensors asymmetric4, not AWQ-GEMM4 or GPTQ packing.
// Code/Z words have distinct axes, BF16 SF, natural K groups and no g_idx.
namespace qwen_ct_asymmetric_projection_fragment {
namespace shared=qwen_artifact_projection_fragment;
namespace attention=shared::attention;
namespace c=shared::c;
namespace g=shared::g;
namespace sem=shared::sem;
using Value=shared::Value;
struct Parameters final {Value codes,zeros,scales;std::uint32_t group_size;};
// Original I64 shape is authenticated and compared before Graph construction;
// it is not an activation, gamma, code bank or fourth decode/kernel input.
struct Names final {std::string_view codes,zeros,scales,shape;std::uint32_t group_size;};
struct Ports final {
  Value input;Parameters parameters;Value stored;
  template<class Visitor> void visit_values(Visitor &&visitor){visitor(input);visitor(stored);}
};
struct ProjectionLeaf final {
  std::span<const Parameters> parameters;std::vector<Ports> *observations;
  Value operator()(sem::Fragment &f,Value input,Value code,const attention::Geometry &geometry,
      attention::LocalTrace *) const {
    if(!observations)throw std::runtime_error("CT original projection needs its actual stored observations");
    const auto matches=std::count_if(parameters.begin(),parameters.end(),[&](const auto &p){return p.codes.ordinal()==code.ordinal();});
    if(matches!=1)throw std::runtime_error("CT original projection needs exactly one codes/Z/SF parameter set");
    const auto &p=*std::find_if(parameters.begin(),parameters.end(),[&](const auto &entry){return entry.codes.ordinal()==code.ordinal();});
    if(f.description(p.scales).data_type()!=c::DataType::kBFloat16)
      throw std::runtime_error("this CT projection declares the original BF16 scale/product boundary");
    const auto stored=shared::project(f,input,{p.codes,p.zeros,p.scales},geometry,
        {c::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4,p.group_size,c::DataType::kBFloat16});
    observations->push_back({input,p,stored});return stored;
  }
};
inline constexpr std::size_t kProjections=7;
struct Result final {attention::TargetResult block;std::vector<Ports> projections;};
inline Result append_target(sem::Fragment &f,Value input,Value old_key,Value old_value,Value cursor,
    Value position,Value frequency,const attention::TargetParameters &supplied,const attention::Geometry &geometry,
    std::span<const Parameters> parameters) {
  if(parameters.size()!=kProjections)throw std::runtime_error("CT target needs all seven original parameter triples");
  for(std::size_t i=0;i<parameters.size();++i)for(std::size_t j=0;j<i;++j)
    if(parameters[i].codes.ordinal()==parameters[j].codes.ordinal())
      throw std::runtime_error("CT target needs seven distinct original code banks");
  Result result;result.projections.reserve(kProjections);
  result.block=attention::append_target_with_projection(f,input,old_key,old_value,cursor,position,frequency,
      supplied,geometry,std::nullopt,ProjectionLeaf{parameters,&result.projections});
  if(result.projections.size()!=kProjections)throw std::runtime_error("CT target did not build seven original packed projections");
  return result;
}
inline void require_packed(const g::CompileReport &report,std::span<const g::NodeId> matrices) {
  shared::require_packed(report,matrices,3);
}
} // namespace qwen_ct_asymmetric_projection_fragment
