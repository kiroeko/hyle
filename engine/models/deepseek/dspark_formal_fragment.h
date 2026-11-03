#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "dspark_component_math.h"
#include "dspark_target_geometry.h"
#include "hyle/compute/low_precision.h"

// Formal DeepSeek projection, HC-head and MoE mathematics shared by trained
// target layers and their original DSpark stages. All weights/caches are
// caller-owned Values. No loader, host coefficients, executor or provider
// decision participates in construction.
namespace dspark_formal_fragment {
namespace ds=dspark_component_math;
namespace core=hyle::inference::model_definition::ops;
namespace sem=hyle::inference::model_definition;
namespace c=hyle::compute;
namespace geometry=dspark_target_geometry;
using Geometry=geometry::Geometry;
using Value=sem::ValueId;
inline constexpr auto kSlots=geometry::kSlots;

void require(const sem::Fragment &f,Value value,const c::TensorDesc &description);
Value cast(sem::Fragment &f,Value value,c::DataType type);
Value stored(sem::Fragment &f,Value value);
Value visible(sem::Fragment &f,Value value);
Value copy(sem::Fragment &f,Value value);
Value reshape(sem::Fragment &f,Value value,std::initializer_list<std::int64_t> shape);
Value indices(sem::Fragment &f,std::span<const std::int32_t> values);
Value index(sem::Fragment &f,std::int64_t value);
Value range(sem::Fragment &f,std::int64_t count,std::int64_t first=0);
Value scatter(sem::Fragment &f,Value bank,Value rows,Value update);
Value gather(sem::Fragment &f,Value bank,Value rows);
Value decode(sem::Fragment &f,Value bank,Value rows,c::RowCacheEncoding encoding);
Value encode(sem::Fragment &f,Value rows,c::RowCacheEncoding encoding);
Value binary(sem::Fragment &f,c::BinaryKind kind,Value a,Value b);
Value sum(sem::Fragment &f,Value input,std::int32_t axis,bool keep=true);


struct NormPorts final {
  Value input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);visitor(raw);visitor(stored);}
};
NormPorts norm(sem::Fragment &f,Value input,Value gamma);
struct ProjectionPorts final {
  Value input;
  std::optional<Value> raw;
  Value stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);if(raw)visitor(*raw);visitor(stored);}
};
ProjectionPorts fp8(sem::Fragment &f,Value input,Value weight,std::int64_t outputs,std::int64_t width);
struct OutputAPorts final {
  Value input,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {visitor(input);visitor(stored);}
};
// Original wo_a is cold-decoded once to BF16 [G,4096,1024]. Only the small
// activation/result tiles change order when a draft has several token rows.
OutputAPorts output_a(sem::Fragment &f,Value input,Value weight,
    std::int64_t rows,const Geometry &model);

struct HcParameters final {ds::HyperValues mixing;Value norm;};
struct MoeParameters final {
  Value router,bias;
  std::array<Value,3> shared,banks; // gate/up/down
  Value map;
};
struct Boundary final {Value input,ids,weights;std::array<Value,3> banks;Value map;};
struct MoeProjectionPorts final {
  Value gate_stored,up_stored,gate_clipped,up_clipped,activation_raw,down_stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&gate_stored,&up_stored,&gate_clipped,&up_clipped,&activation_raw,&down_stored})visitor(*port);
  }
};
struct MoePorts final {
  Value input,logits,softplus,scores,selection,ids,selected,sum,normalized,weights,local_ids,map;
  MoeProjectionPorts routed,shared;
  Value weighted_raw,weighted_stored,shared_activation_stored,routed_sum,merged_raw,merged_stored;
  ProjectionPorts shared_gate,shared_up,shared_down;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&logits,&softplus,&scores,&selection,&ids,&selected,&sum,&normalized,&weights,&local_ids,&map})visitor(*port);
    routed.visit_values(visitor);shared.visit_values(visitor);
    for(auto *port:{&weighted_raw,&weighted_stored,&shared_activation_stored,&routed_sum,&merged_raw,&merged_stored})visitor(*port);
    shared_gate.visit_values(visitor);shared_up.visit_values(visitor);shared_down.visit_values(visitor);
  }
};
struct MoeResult final {Value output;MoePorts ports;Boundary boundary;};
MoeResult append_moe(sem::Fragment &f,Value input,const MoeParameters &parameters,
    const Geometry &model=geometry::kFlash);


struct HeadParameters final {Value fn,scale,base,norm;};
struct HeadPorts final {
  Value input,flat,projection,inverse,mixes,gate,collapsed_raw,collapsed_stored;
  NormPorts norm;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *port:{&input,&flat,&projection,&inverse,&mixes,&gate,&collapsed_raw,&collapsed_stored})visitor(*port);
    norm.visit_values(visitor);
  }
};
struct HeadResult final {Value normalized;HeadPorts ports;};
HeadResult append_head(sem::Fragment &f,Value input,const HeadParameters &parameters);


} // namespace dspark_formal_fragment
