#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../detail/math_helpers.h"
#include "glm_moe_geometry.h"
#include "glm_moe_storage.h"
#include "glm_fp8_arithmetic.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Pure original GLM MoE construction, with explicit model-local geometry. All parameters are actual immutable
// Graph constants or explicit compact expert inputs supplied by the caller.
// No weight loading, provider routing, host callback or execution lives here.
namespace glm_original_moe_fragment {
namespace sem=hyle::inference::model_definition;
namespace core=hyle::inference::model_definition::ops;
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
Value cast(sem::Fragment &f,Value input,c::DataType type);
Value sum(sem::Fragment &f,Value input,std::int32_t axis);
Result append(sem::Fragment &f,Value input,const Parameters &parameters,const Geometry &model=kFlashGeometry,
                     glm_fp8_arithmetic::Arithmetic arithmetic = {},
                     WeightStorage storage=WeightStorage::kFloat8Block128);
} // namespace glm_original_moe_fragment
