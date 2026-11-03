#pragma once
#include <array>
#include <cmath>
#include <vector>
#include "../detail/math_helpers.h"
#include "hyle/compute/model_ops.h"

// Actual-value DeepSeek construction. No synthetic weights or reference math.
namespace dspark_component_math {
namespace c=hyle::compute;
namespace sem=hyle::inference::model_definition;
using sem::Fragment;
using sem::ValueId;
} // namespace dspark_component_math
namespace dspark_component_layout {
using ValueId=hyle::inference::model_definition::ValueId;
struct QueryNormPorts final {
  ValueId input,square_raw,square,mean_raw,mean,variance_raw,variance,inverse_raw,inverse,output_raw,output;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&square_raw,&square,&mean_raw,&mean,&variance_raw,&variance,
        &inverse_raw,&inverse,&output_raw,&output}) visitor(*value);
  }
};
struct RopePorts final {
  ValueId input,raw,stored;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&raw,&stored})visitor(*value);
  }
};
struct HyperPrePorts final {
  ValueId state, flat, projection, inverse, mixes, pre, post, combination, collapsed_raw, collapsed;
};
struct HyperPostPorts final { ValueId branch, residual, expanded, stored; };
struct HyperValues final { ValueId fn, scale, base; };
} // namespace dspark_component_layout
namespace dspark_component_math {
using dspark_component_layout::QueryNormPorts;
using dspark_component_layout::RopePorts;
using dspark_component_layout::HyperPrePorts;
using dspark_component_layout::HyperPostPorts;
using dspark_component_layout::HyperValues;
struct Projection final {ValueId output;};
struct Hyper final {
  ValueId collapsed,post,combination;
  Projection fn;
  std::int64_t hidden;
  bool bfloat_outputs=true;
  HyperPrePorts ports{};
};
ValueId bfloat_boundary(Fragment &,ValueId);
using sem::ops::slice;
using sem::ops::permute;
using sem::ops::concat;
ValueId mean(Fragment &f, ValueId x);
ValueId query_norm(Fragment &f, ValueId x, QueryNormPorts *ports=nullptr);
std::vector<float> rope_frequencies(bool compressed, bool inverse);
ValueId rope(Fragment &f, ValueId x, ValueId positions, bool compressed, bool inverse = false,
                    RopePorts *ports=nullptr);
ValueId inverse_rms(Fragment &f, ValueId input);
Hyper hyper_pre(Fragment &f, ValueId state, std::int64_t hidden,
                       const HyperValues &parameters, bool bfloat = true);
ValueId hyper_post(Fragment &f, const Hyper &mix, ValueId state, ValueId branch,
                          HyperPostPorts *ports = nullptr);
}
