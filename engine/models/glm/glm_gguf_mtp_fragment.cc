#include "glm_gguf_mtp_fragment.h"

namespace glm_gguf_mtp_fragment {
Moe append_moe_with_limits(sem::Fragment &f,Value input,const types::Block &p,
    const std::array<Value,3> &banks,Value map,const glm_moe_geometry::Geometry &g,
    std::optional<float> shared_limit,ExpertFold fold) {
  glm_moe_geometry::require(g);
  if((shared_limit&&(!std::isfinite(*shared_limit)||!(*shared_limit>0.F)))||
      (fold!=ExpertFold::kReduction&&fold!=ExpertFold::kSelectedOrder))
    throw std::runtime_error("converted MoE shared clamp/fold contract is unsupported");
  cache::require(f,input,sem::dense({1,g.hidden}));
  cache::require(f,map,sem::dense({g.experts},c::DataType::kInt32));
  for(std::size_t leg=0;leg<3;++leg) {
    const auto &d=f.description(banks[leg]);const auto codec=c::gguf_quantized_type(d.storage_format());
    if(!codec||d!=c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,
        std::int64_t(g.slots),leg==2?g.hidden:g.intermediate,leg==2?g.intermediate:g.hidden).value())
      throw std::runtime_error("converted GLM needs the actual packed E8 expert descriptors");
  }
  Moe r;r.input=input;r.map=map;r.banks=banks;r.shared_weights={p.shared_gate,p.shared_up,p.shared_down};
  r.logits=leaf::f32_linear(f,input,p.router).output;
  r.scores=core::unary(f,c::UnaryKind::kSigmoid,r.logits);
  r.selection=core::binary(f,c::BinaryKind::kAdd,r.scores,p.bias);
  const auto top=f.append(c::Operation(c::TopK{.count=g.slots,.stable=true}),{r.selection},2);
  r.ids=top[1];r.selected_scores=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),{r.scores,r.ids});
  r.sum=glm_original_moe_fragment::sum(f,r.selected_scores,-1);
  // Fixed converted GGUF norm_w policy. This is deliberately not the
  // official safetensor path's sum+1e-20 normalization.
  r.denominator=core::binary(f,c::BinaryKind::kMaximum,r.sum,core::filled(f,sem::dense({1,1}),0x1p-14F));
  const auto normalized=core::binary(f,c::BinaryKind::kDivide,r.selected_scores,r.denominator);
  r.weights=core::binary(f,c::BinaryKind::kMultiply,normalized,
      core::filled(f,sem::dense({1,std::int64_t(g.slots)}),g.route_scale));
  r.boundary={input,r.ids,r.weights,banks,map};
  r.local_ids=f.one(c::Operation(c::Gather{}),{map,r.ids});
  const auto grouped=[&](Value actual,std::size_t leg) {
    return f.one(c::model_ops::grouped_matmul(),{banks[leg],actual,r.local_ids});
  };
  const auto activate=[&](Value gate,Value up,std::optional<float> limit) {
    if(limit) {
      gate=f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(),*limit),{gate});
      up=f.one(c::model_ops::clamp(-*limit,*limit),{up});
    }
    return core::binary(f,c::BinaryKind::kMultiply,core::unary(f,c::UnaryKind::kSilu,gate),up);
  };
  const auto row=compute::reshape(f,input,{1,1,g.hidden});
  r.routed[0]=grouped(row,0);r.routed[1]=grouped(row,1);
  r.routed[2]=activate(r.routed[0],r.routed[1],g.swiglu_limit);r.routed[3]=grouped(r.routed[2],2);
  r.weighted=core::binary(f,c::BinaryKind::kMultiply,r.routed[3],
      compute::reshape(f,r.weights,{1,std::int64_t(g.slots),1}));
  if(fold==ExpertFold::kReduction) {
    r.routed_sum=compute::reshape(f,glm_original_moe_fragment::sum(f,r.weighted,1),{1,g.hidden});
  } else {
    // Fixed GGUF build_moe_ffn adds actual selected-slot views in order.
    const auto selected_row=[&](std::int64_t slot){return compute::reshape(f,compute::slice(f,r.weighted,1,slot,1),{1,g.hidden});};
    r.routed_sum=selected_row(0);
    for(std::size_t slot=1;slot<g.slots;++slot)
      r.routed_sum=core::binary(f,c::BinaryKind::kAdd,r.routed_sum,selected_row(std::int64_t(slot)));
  }
  r.shared[0]=leaf::linear(f,input,p.shared_gate).output;r.shared[1]=leaf::linear(f,input,p.shared_up).output;
  r.shared[2]=activate(r.shared[0],r.shared[1],shared_limit);r.shared[3]=leaf::linear(f,r.shared[2],p.shared_down).output;
  r.output=core::binary(f,c::BinaryKind::kAdd,r.routed_sum,r.shared[3]);return r;
}
}

namespace glm_gguf_mtp_fragment {
Moe append_moe(sem::Fragment &f,Value input,const types::Block &p,
    const std::array<Value,3> &banks,Value map,const glm_moe_geometry::Geometry &g) {
  return append_moe_with_limits(f,input,p,banks,map,g,g.swiglu_limit,ExpertFold::kReduction);
}
}
