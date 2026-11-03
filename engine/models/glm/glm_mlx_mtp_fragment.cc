#include "glm_mlx_mtp_fragment.h"

namespace glm_mlx_mtp_fragment {
Value cast(sem::Fragment &f,Value x,c::DataType type) {
  return f.description(x).data_type()==type?x:f.one(c::Operation(c::Cast{.output_type=type}),{x});
}
}

namespace glm_mlx_mtp_fragment {
Value permute(sem::Fragment &f,Value x,std::initializer_list<std::int32_t> axes) {
  const auto p=c::View::permute({axes.begin(),axes.size()});if(!p)throw std::runtime_error("MLX pure permutation differs");return f.one(c::Operation(*p),{x});
}
}

namespace glm_mlx_mtp_fragment {
Value inc(sem::Fragment &f,Value x,std::int32_t amount) {
  if(f.description(x)!=sem::dense({1},c::DataType::kInt32))throw std::runtime_error("MLX cursor must be actual I32[1]");
  std::vector<std::int32_t> table(8192);
  for(std::size_t i=0;i<table.size();++i)table[i]=std::int32_t(i)+amount;
  return f.one(c::Operation(c::Gather{}),{f.constant(sem::dense({std::int64_t(table.size())},c::DataType::kInt32),sem::bytes(table)),x});
}
}

namespace glm_mlx_mtp_fragment {
std::array<Value,2> prepare_kv_b(sem::Fragment &f,Value original) {
  if(f.description(original)!=sem::dense({32768,512},c::DataType::kBFloat16))throw std::runtime_error("Q32 original KV_b physical shape differs");
  const auto heads=fp::reshape(f,original,{64,512,512});
  return {fp::copy(f,permute(f,fp::slice(f,heads,1,0,256),{0,2,1})),
          fp::copy(f,fp::slice(f,heads,1,256,256))};
}
}

namespace glm_mlx_mtp_fragment {
Moe append_moe(sem::Fragment &f,Value actual,const Parameters &p,std::span<const Value> banks,Value map) {
  if(f.description(actual)!=sem::dense({1,4096})||banks.size()!=9||f.description(map)!=sem::dense({288},c::DataType::kInt32))
    throw std::runtime_error("MLX MoE requires F32 actual predecessor, original9 planes and actual288-map");
  for(std::size_t i=0;i<9;++i)if(f.description(banks[i])!=own::resident_description(own::Leg(i/3),own::Plane(i%3)))
    throw std::runtime_error("MLX compact original plane type/shape differs");
  Moe r;auto &a=r.ports;a.input=actual;a.map=map;
  a.logits=p.project.linear(f,actual,p.nextn.block.router).output;
  a.scores=core::unary(f,c::UnaryKind::kSigmoid,a.logits);a.selection=core::binary(f,c::BinaryKind::kAdd,a.scores,p.nextn.block.bias);
  const auto top=f.append(c::Operation(c::TopK{.count=8,.stable=true}),{a.selection},2);a.ids=top[1];
  a.selected_scores=f.one(c::model_ops::gather_elements(-1,c::IndexBoundsPolicy::kChecked),{a.scores,a.ids});
  a.sum=glm_original_moe_fragment::sum(f,a.selected_scores,-1);
  a.denominator=core::binary(f,c::BinaryKind::kAdd,a.sum,core::filled(f,sem::dense({1,1}),1.e-20F));
  a.normalized=core::binary(f,c::BinaryKind::kDivide,a.selected_scores,a.denominator);
  a.weights=core::binary(f,c::BinaryKind::kMultiply,a.normalized,core::filled(f,sem::dense({1,8}),2.5F));
  r.boundary={actual,a.ids,a.weights,std::vector<Value>(banks.begin(),banks.end()),map};
  a.local_ids=f.one(c::Operation(c::Gather{}),{map,a.ids});
  const auto activation=[&](auto &branch) {
    branch.gate_clipped=f.one(c::model_ops::clamp(-std::numeric_limits<float>::infinity(),10.F),{branch.gate_stored});
    branch.up_clipped=f.one(c::model_ops::clamp(-10.F,10.F),{branch.up_stored});
    branch.activation_raw=core::binary(f,c::BinaryKind::kMultiply,core::unary(f,c::UnaryKind::kSilu,branch.gate_clipped),branch.up_clipped);
    branch.activation_stored=branch.activation_raw; // Actual F32 store, not BF16.
  };
  const auto grouped=[&](Value input,std::size_t leg) {
    return affine::grouped(f,input,{banks[3*leg],banks[3*leg+1],banks[3*leg+2],64},a.local_ids).output;
  };
  const auto row=fp::reshape(f,actual,{1,1,4096});
  const auto gate=grouped(row,0),up=grouped(row,1);
  a.routed.gate_raw=gate;a.routed.gate_stored=gate;a.routed.up_raw=up;a.routed.up_stored=up;activation(a.routed);
  const auto down=grouped(a.routed.activation_stored,2);a.routed.down_raw=down;
  a.weighted_raw=core::binary(f,c::BinaryKind::kMultiply,down,fp::reshape(f,a.weights,{1,8,1}));a.weighted_stored=a.weighted_raw;
  a.routed_sum_raw=fp::reshape(f,glm_original_moe_fragment::sum(f,a.weighted_stored,1),{1,4096});a.routed_sum_stored=a.routed_sum_raw;
  const auto sg=affine::linear(f,actual,p.shared[0]).output,su=affine::linear(f,actual,p.shared[1]).output;
  a.shared.gate_raw=sg;a.shared.gate_stored=sg;a.shared.up_raw=su;a.shared.up_stored=su;activation(a.shared);
  const auto sd=affine::linear(f,a.shared.activation_stored,p.shared[2]).output;a.shared.down_raw=sd;a.shared_stored=sd;
  a.merge_raw=core::binary(f,c::BinaryKind::kAdd,a.routed_sum_stored,a.shared_stored);a.merge_stored=a.merge_raw;r.output=a.merge_stored;return r;
}
}

namespace glm_mlx_mtp_fragment {
Result append(sem::Fragment &f,State state,Value actual_hidden,const Parameters &p,std::span<const Value> banks,
    Value map,bool fresh,std::int32_t position,std::int32_t carried_tail) {
  const auto moe_builder=[&](sem::Fragment &graph,Value actual){return append_moe(graph,actual,p,banks,map);};
  return append_with_moe(f,state,actual_hidden,p,fresh,position,carried_tail,kGeometry,{},moe_builder);
}
}
