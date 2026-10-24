#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_moe_semantic_fragments.h"
#include "glm_moe_geometry.h"
#include "glm_moe_storage.h"
#include "glm_fp8_arithmetic.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Pure original GLM MoE construction, with explicit model-local geometry. All parameters are actual immutable
// Graph constants or explicit compact expert inputs supplied by the caller.
// No weight loading, provider routing, host callback or execution lives here.
namespace glm_original_moe_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
using Geometry=glm_moe_geometry::Geometry;
using WeightStorage=glm_moe_storage::Kind;
inline constexpr auto kFlashGeometry=glm_moe_geometry::kFlash;
inline constexpr auto kProGeometry=glm_moe_geometry::kPro;
inline constexpr std::int64_t kHidden=kFlashGeometry.hidden,kIntermediate=kFlashGeometry.intermediate;
inline constexpr std::int64_t kExperts=kFlashGeometry.experts,kSlots=std::int64_t(kFlashGeometry.slots);
struct Parameters final {
  Value router,bias;
  std::array<Value,3> shared,banks; // gate, up, down
  Value map;
};
struct Boundary final {
  Value input,ids,weights;
  std::array<Value,3> banks;
  Value map;
};
struct ProjectionPorts final {
  std::optional<Value> gate_raw,up_raw,down_raw,down_stored;
  Value gate_stored,up_stored,gate_clipped,up_clipped;
  Value activation_raw,activation_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    if(gate_raw)visitor(*gate_raw);visitor(gate_stored);
    if(up_raw)visitor(*up_raw);visitor(up_stored);
    for(auto *value:{&gate_clipped,&up_clipped,&activation_raw,&activation_stored})visitor(*value);
    if(down_raw)visitor(*down_raw);if(down_stored)visitor(*down_stored);
  }
};
struct Ports final {
  Value input,logits,scores,selection,ids,selected_scores,sum,denominator,normalized,weights,local_ids,map;
  ProjectionPorts routed,shared;
  Value weighted_raw,weighted_stored,routed_sum_raw,routed_sum_stored,shared_stored,merge_raw,merge_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&logits,&scores,&selection,&ids,&selected_scores,&sum,&denominator,&normalized,&weights,&local_ids,&map})visitor(*value);
    routed.visit_values(visitor);shared.visit_values(visitor);
    for(auto *value:{&weighted_raw,&weighted_stored,&routed_sum_raw,&routed_sum_stored,&shared_stored,&merge_raw,&merge_stored})visitor(*value);
  }
};
struct Result final {Value output;Ports ports;Boundary boundary;};
inline Value cast(sem::Fragment &f,Value input,c::DataType type) {
  return f.one(c::Operation(c::Cast{.output_type=type}),{input});
}
inline Value sum(sem::Fragment &f,Value input,std::int32_t axis) {
  return f.one(c::Operation(c::Reduce{.kind=c::ReductionKind::kSum,
      .axes=c::Axes::from(std::array{axis}).value(),.keep_dimensions=true,
      .accumulation_type=c::DataType::kFloat32}),{input});
}
inline Result append(sem::Fragment &f,Value input,const Parameters &parameters,const Geometry &model=kFlashGeometry,
                     glm_fp8_arithmetic::Arithmetic arithmetic = {},
                     WeightStorage storage=WeightStorage::kFloat8Block128) {
  glm_moe_storage::require(storage,model);arithmetic.require();
  const bool mxfp4=storage==WeightStorage::kMxfp4Group32;
  const auto slots_count=std::int64_t(model.slots);
  if(f.description(input)!=sem::dense({1,model.hidden},c::DataType::kBFloat16)||
      f.description(parameters.router)!=sem::dense({model.experts,model.hidden},c::DataType::kBFloat16)||
      f.description(parameters.bias)!=sem::dense({model.experts})||
      f.description(parameters.map)!=sem::dense({model.experts},c::DataType::kInt32))
    throw std::runtime_error("original GLM MoE requires full original router domain and actual BF16 input");
  for(std::size_t leg=0;leg<parameters.banks.size();++leg) {
    const auto n=leg==2?model.hidden:model.intermediate,k=leg==2?model.intermediate:model.hidden;
    if(f.description(parameters.banks[leg])!=glm_moe_storage::packed_description(n,k,model.slots,storage,true)||
        f.description(parameters.shared[leg])!=glm_moe_storage::packed_description(n,k,1,storage,false))
      throw std::runtime_error("original GLM MoE requires the actual compact storage and original shared banks");
    if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.banks[leg].ordinal();}))
      throw std::runtime_error("compact expert bank must be an explicit ordinary Graph input");
  }
  if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.map.ordinal();}))
    throw std::runtime_error("expert global-to-local map must be an explicit Graph input");
  Ports ports;ports.input=input;ports.map=parameters.map;
  // The fixed CUDA router path consumes BF16 operands and returns F32
  // logits. The same declared contract is used on every Hyle backend.
  ports.logits=f.one(c::model_ops::dense_linear_f32(),{input,parameters.router});
  ports.scores=core::unary(f,c::UnaryKind::kSigmoid,ports.logits);
  ports.selection=core::binary(f,c::BinaryKind::kAdd,ports.scores,parameters.bias);
  const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(slots_count),.stable=true}),{ports.selection},2);
  ports.ids=top[1];
  ports.selected_scores=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),{ports.scores,ports.ids});
  ports.sum=sum(f,ports.selected_scores,-1);
  ports.denominator=core::binary(f,c::BinaryKind::kAdd,ports.sum,core::filled(f,sem::dense({1,1}),1.0e-20F));
  ports.normalized=core::binary(f,c::BinaryKind::kDivide,ports.selected_scores,ports.denominator);
  ports.weights=core::binary(f,c::BinaryKind::kMultiply,ports.normalized,core::filled(f,sem::dense({1,slots_count}),model.route_scale));
  const Boundary boundary{input,ports.ids,ports.weights,parameters.banks,parameters.map};
  // The full global IDs remain visible; only the explicit checked map selects
  // compact local slots. No modulo, clamp or restriction of the expert domain.
  ports.local_ids=f.one(c::Operation(c::Gather{}),{parameters.map,ports.ids});
  const auto activate=[&](ProjectionPorts &branch) {
    const auto gate=cast(f,branch.gate_stored,c::DataType::kFloat32);
    const auto up=cast(f,branch.up_stored,c::DataType::kFloat32);
    if(model.swiglu_limit) {
      const auto limit=*model.swiglu_limit;
      branch.gate_clipped=f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(),limit),{gate});
      branch.up_clipped=f.one(c::model_ops::clamp(-limit,limit),{up});
    } else {branch.gate_clipped=gate;branch.up_clipped=up;}
    branch.activation_raw=core::binary(f,c::BinaryKind::kMultiply,core::unary(f,c::UnaryKind::kSilu,branch.gate_clipped),branch.up_clipped);
    branch.activation_stored=cast(f,branch.activation_raw,c::DataType::kBFloat16);
  };
  const auto planes=[&](Value bank,std::size_t leg,bool grouped) {
    const auto n=leg==2?model.hidden:model.intermediate,k=leg==2?model.intermediate:model.hidden;
    const auto count=grouped?slots_count:1;
    const auto code_bytes=count*n*(k/2),scale_bytes=count*n*(k/32);
    const auto code_slice=c::View::slice(0,0,code_bytes),scale_slice=c::View::slice(0,code_bytes,scale_bytes);
    if(!code_slice||!scale_slice)throw std::runtime_error("original MXFP4 carrier plane view is invalid");
    const auto codes=f.one(c::Operation(*code_slice),{bank});
    const auto scales=f.one(c::Operation(*scale_slice),{bank});
    if(grouped)return std::array{core::reshape(f,codes,{slots_count,n,k/2}),core::reshape(f,scales,{slots_count,n,k/32})};
    return std::array{core::reshape(f,codes,{n,k/2}),core::reshape(f,scales,{n,k/32})};
  };
  const auto grouped=[&](Value activation,std::size_t leg) {
    if(mxfp4) {
      const auto original=planes(parameters.banks[leg],leg,true);
      return f.one(c::model_ops::grouped_mxfp4_linear(),{activation,original[0],original[1],ports.local_ids});
    }
    return f.one(c::model_ops::grouped_float8_linear(arithmetic.policy,arithmetic.absmax_floor),
        {parameters.banks[leg],activation,ports.local_ids});
  };
  const auto shared=[&](Value activation,std::size_t leg) {
    if(mxfp4) {
      const auto original=planes(parameters.shared[leg],leg,false);
      return f.one(c::model_ops::mxfp4_linear(),{activation,original[0],original[1]});
    }
    return f.one(c::model_ops::float8_block_linear(arithmetic.policy,arithmetic.absmax_floor),
        {activation,parameters.shared[leg]});
  };
  const auto widened=mxfp4?input:cast(f,input,c::DataType::kFloat32);
  const auto slots=core::reshape(f,widened,{1,1,model.hidden});
  const auto routed_gate=grouped(slots,0),routed_up=grouped(slots,1);
  if(mxfp4) {
    ports.routed.gate_stored=routed_gate;ports.routed.up_stored=routed_up;
  } else {
    ports.routed.gate_raw=routed_gate;ports.routed.up_raw=routed_up;
    ports.routed.gate_stored=cast(f,routed_gate,c::DataType::kBFloat16);
    ports.routed.up_stored=cast(f,routed_up,c::DataType::kBFloat16);
  }
  activate(ports.routed);
  const auto routed_down=grouped(mxfp4?ports.routed.activation_stored:
      cast(f,ports.routed.activation_stored,c::DataType::kFloat32),2);
  if(mxfp4)ports.routed.down_stored=routed_down;else ports.routed.down_raw=routed_down;
  // E4 retains F32 down accumulation before weighting. Mxfp4Linear's actual
  // BF16 return is a distinct boundary before weighting; it is widened here,
  // never relabelled as an unrounded F32 down result. Neither weights A.
  ports.weighted_raw=core::binary(f,c::BinaryKind::kMultiply,
      mxfp4?cast(f,routed_down,c::DataType::kFloat32):routed_down,core::reshape(f,ports.weights,{1,slots_count,1}));
  ports.weighted_stored=cast(f,ports.weighted_raw,c::DataType::kBFloat16);
  ports.routed_sum_raw=core::reshape(f,sum(f,cast(f,ports.weighted_stored,c::DataType::kFloat32),1),{1,model.hidden});
  ports.routed_sum_stored=cast(f,ports.routed_sum_raw,c::DataType::kBFloat16);
  const auto shared_gate=shared(widened,0),shared_up=shared(widened,1);
  if(mxfp4) {
    ports.shared.gate_stored=shared_gate;ports.shared.up_stored=shared_up;
  } else {
    ports.shared.gate_raw=shared_gate;ports.shared.up_raw=shared_up;
    ports.shared.gate_stored=cast(f,shared_gate,c::DataType::kBFloat16);
    ports.shared.up_stored=cast(f,shared_up,c::DataType::kBFloat16);
  }
  activate(ports.shared);
  const auto shared_down=shared(mxfp4?ports.shared.activation_stored:
      cast(f,ports.shared.activation_stored,c::DataType::kFloat32),2);
  if(mxfp4) {ports.shared.down_stored=shared_down;ports.shared_stored=shared_down;}
  else {ports.shared.down_raw=shared_down;ports.shared_stored=cast(f,shared_down,c::DataType::kBFloat16);}
  ports.merge_raw=core::binary(f,c::BinaryKind::kAdd,cast(f,ports.routed_sum_stored,c::DataType::kFloat32),cast(f,ports.shared_stored,c::DataType::kFloat32));
  ports.merge_stored=cast(f,ports.merge_raw,c::DataType::kBFloat16);
  return {ports.merge_stored,ports,boundary};
}
} // namespace glm_original_moe_fragment
