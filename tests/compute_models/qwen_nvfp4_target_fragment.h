#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <string_view>
#include <vector>

#include "qwen_full_attention_fragment.h"

// Typed original ModelOpt projection leaves inside the SAME Qwen target math.
// No provider callback, weight decode Graph, state machine or executor.
namespace qwen_nvfp4_target_fragment {
namespace attention=qwen_full_attention_fragment;
namespace c=hyle::compute;
namespace sem=attention::sem;
using Value=attention::Value;
enum class Projection : std::size_t {kQuery,kKey,kValue,kOutput,kGate,kUp,kDown,kCount};
inline constexpr std::size_t kProjections=std::size_t(Projection::kCount);
struct ProjectionParameters final {Value codes,scales,weight_global,input_global;};
struct ProjectionNames final {
  std::string_view codes,scales,weight_global,input_global;
};
struct ProjectionPorts final {
  Value input;
  ProjectionParameters parameters;
  Value stored; // Actual Nvfp4Linear BF16 result; no raw-F32 port exists.
  template<class Visitor> void visit_values(Visitor &&visitor) {
    visitor(input);visitor(stored);
  }
};
struct Parameters final {
  Value input_norm,query_norm,key_norm,post_norm,target_norm;
  std::array<ProjectionParameters,kProjections> projections;
};
struct Result final {
  attention::TargetResult block;
  std::vector<ProjectionPorts> projections;
};
inline Result append_target(sem::Fragment &fragment,Value input,Value old_key,Value old_value,
    Value cursor,Value position,Value frequency,const Parameters &parameters,const attention::Geometry &geometry,
    std::optional<Value> normalized_input={}) {
  if(geometry.storage!=c::DataType::kBFloat16)
    throw std::runtime_error("NVFP4 Qwen target keeps the original BF16 activation/store contract");
  for(std::size_t i=0;i<kProjections;++i)for(std::size_t j=0;j<i;++j)
    if(parameters.projections[i].codes.ordinal()==parameters.projections[j].codes.ordinal())
      throw std::runtime_error("NVFP4 target projections need distinct actual original code banks");
  Result result;result.projections.reserve(kProjections);
  const auto codes=[&](Projection role){return parameters.projections[std::size_t(role)].codes;};
  const attention::TargetParameters shared{parameters.input_norm,codes(Projection::kQuery),codes(Projection::kKey),
      codes(Projection::kValue),parameters.query_norm,parameters.key_norm,codes(Projection::kOutput),
      parameters.post_norm,codes(Projection::kGate),codes(Projection::kUp),codes(Projection::kDown),parameters.target_norm};
  const auto leaf=[&](sem::Fragment &f,Value actual,Value weight,const attention::Geometry &actual_geometry,
      attention::LocalTrace *) {
    if(actual_geometry.storage!=c::DataType::kBFloat16)
      throw std::runtime_error("NVFP4 target cannot silently convert an F32-state projection policy");
    const auto found=std::find_if(parameters.projections.begin(),parameters.projections.end(),
        [&](const auto &entry){return entry.codes.ordinal()==weight.ordinal();});
    if(found==parameters.projections.end())
      throw std::runtime_error("NVFP4 target is missing an explicit original calibration quartet");
    // Public inference enforces BF16 A[M,K], adjacent low-first U8 W[N,K/2],
    // E4 SF[N,K/16], independent F32 scalars and a BF16 output. QDQ remains
    // inside this actual operation on every invocation, before its F32 dot.
    const auto stored=f.one(c::model_ops::nvfp4_linear(),
        {actual,found->codes,found->scales,found->weight_global,found->input_global});
    result.projections.push_back({actual,*found,stored});return stored;
  };
  result.block=attention::append_target_with_projection(fragment,input,old_key,old_value,cursor,position,
      frequency,shared,geometry,normalized_input,leaf);
  if(result.projections.size()!=kProjections)
    throw std::runtime_error("NVFP4 target did not construct all seven original projection leaves");
  return result;
}
} // namespace qwen_nvfp4_target_fragment
