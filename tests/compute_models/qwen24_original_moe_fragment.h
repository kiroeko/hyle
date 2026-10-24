#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "d5_moe_semantic_fragments.h"
#include "hyle/compute/low_precision.h"

// Pure original MAIN Q24 GGUF MoE math. The caller supplies immutable original
// parameters and ordinary compact expert inputs; no loading or execution here.
namespace qwen24_original_moe_fragment {
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace core=d5_moe_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
inline constexpr std::int64_t kHidden=8192,kIntermediate=2048,kExperts=512,kSlots=10;
struct Parameters final {
  Value router,shared_gate;
  std::array<Value,3> shared,banks; // gate,up,down
  Value map;
};
struct Boundary final {
  Value input,ids,weights;
  std::array<Value,3> banks;
  Value map;
};
struct ProjectionPorts final {
  Value gate,up,activation,down;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&gate,&up,&activation,&down})visitor(*value);
  }
};
struct Ports final {
  Value input,logits,probabilities,ids,selected_scores,sum,denominator,weights,local_ids,map;
  ProjectionPorts routed,shared;
  Value weighted,routed_sum,shared_gate_raw,shared_gate_sigmoid,shared_gated,merged;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    for(auto *value:{&input,&logits,&probabilities,&ids,&selected_scores,&sum,&denominator,&weights,&local_ids,&map})visitor(*value);
    routed.visit_values(visitor);shared.visit_values(visitor);
    for(auto *value:{&weighted,&routed_sum,&shared_gate_raw,&shared_gate_sigmoid,&shared_gated,&merged})visitor(*value);
  }
};
struct Result final {Value output;Ports ports;Boundary boundary;};
inline Value sum(sem::Fragment &f,Value input,std::int32_t axis) {
  return f.one(c::Operation(c::Reduce{.kind=c::ReductionKind::kSum,
      .axes=c::Axes::from(std::array{axis}).value(),.keep_dimensions=true,
      .accumulation_type=c::DataType::kFloat32}),{input});
}
inline Result append(sem::Fragment &f,Value input,const Parameters &parameters) {
  if(f.description(input)!=sem::dense({1,kHidden})||
      f.description(parameters.router)!=sem::dense({kExperts,kHidden}).transposed().value()||
      f.description(parameters.shared_gate)!=sem::dense({1,kHidden}).transposed().value()||
      f.description(parameters.map)!=sem::dense({kExperts},c::DataType::kInt32))
    throw std::runtime_error("original Q24 MoE needs the full original F32 router and shared gate");
  for(std::size_t leg=0;leg<3;++leg) {
    const auto n=leg==2?kHidden:kIntermediate,k=leg==2?kIntermediate:kHidden;
    const auto &bank=f.description(parameters.banks[leg]);
    const auto codec=c::gguf_quantized_type(bank.storage_format());
    const auto shared_codec=c::gguf_quantized_type(f.description(parameters.shared[leg]).storage_format());
    if(!codec||!shared_codec||
        bank!=c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,kSlots,n,k).value()||
        f.description(parameters.shared[leg])!=c::gguf_weight_description(*shared_codec,c::DataType::kFloat32,k,n).value())
      throw std::runtime_error("original Q24 MoE requires canonical actual GGUF expert and shared banks");
    if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.banks[leg].ordinal();}))
      throw std::runtime_error("compact Q24 experts must be ordinary runtime inputs");
  }
  if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.map.ordinal();}))
    throw std::runtime_error("Q24 resident mapping must be an ordinary runtime input");
  Ports ports;ports.input=input;ports.map=parameters.map;
  ports.logits=f.one(c::Operation(c::MatMul{}),{input,parameters.router});
  ports.probabilities=f.one(c::Operation(c::Softmax{}),{ports.logits});
  const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(kSlots),.stable=true}),{ports.probabilities},2);
  ports.ids=top[1];ports.selected_scores=top[0];
  ports.sum=sum(f,ports.selected_scores,-1);
  // Exact fixed llama.cpp norm_w contract. No DeepSeek bias/epsilon/scale.
  ports.denominator=core::binary(f,c::BinaryKind::kMaximum,ports.sum,core::filled(f,sem::dense({1,1}),0x1p-14F));
  ports.weights=core::binary(f,c::BinaryKind::kDivide,ports.selected_scores,ports.denominator);
  const Boundary boundary{input,ports.ids,ports.weights,parameters.banks,parameters.map};
  ports.local_ids=f.one(c::Operation(c::Gather{}),{parameters.map,ports.ids});
  const auto grouped=[&](Value actual,std::size_t leg) {
    return f.one(c::model_ops::grouped_matmul(),{parameters.banks[leg],actual,ports.local_ids});
  };
  const auto slots=core::reshape(f,input,{1,1,kHidden});
  ports.routed.gate=grouped(slots,0);ports.routed.up=grouped(slots,1);
  ports.routed.activation=f.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{ports.routed.gate,ports.routed.up});
  ports.routed.down=grouped(ports.routed.activation,2);
  // Qwen is not the Llama4 input-weighting branch: multiply actual down
  // outputs, then reduce selected slots, preserving F32 GGUF boundaries.
  ports.weighted=core::binary(f,c::BinaryKind::kMultiply,ports.routed.down,core::reshape(f,ports.weights,{1,kSlots,1}));
  ports.routed_sum=core::reshape(f,sum(f,ports.weighted,1),{1,kHidden});
  ports.shared.gate=f.one(c::Operation(c::MatMul{}),{input,parameters.shared[0]});
  ports.shared.up=f.one(c::Operation(c::MatMul{}),{input,parameters.shared[1]});
  ports.shared.activation=f.one(c::Operation(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate}),{ports.shared.gate,ports.shared.up});
  ports.shared.down=f.one(c::Operation(c::MatMul{}),{ports.shared.activation,parameters.shared[2]});
  ports.shared_gate_raw=f.one(c::Operation(c::MatMul{}),{input,parameters.shared_gate});
  ports.shared_gate_sigmoid=core::unary(f,c::UnaryKind::kSigmoid,ports.shared_gate_raw);
  ports.shared_gated=core::binary(f,c::BinaryKind::kMultiply,ports.shared.down,ports.shared_gate_sigmoid);
  ports.merged=core::binary(f,c::BinaryKind::kAdd,ports.routed_sum,ports.shared_gated);
  return {ports.merged,ports,boundary};
}

// Original BF16 experts use a different storage/normalization contract from
// converted GGUF. Keep the selected banks BF16 and use ordinary batched
// MatMul; GroupedMatMul's F32 output is not a BF16 store boundary.
struct BFloatPorts final {
  Ports values;
  Value router_raw,sorted_weights,routed_silu,shared_silu,weighted_raw;
  std::vector<Value> sums;
  template<class Visitor> void visit_values(Visitor &&visitor) {
    values.visit_values(visitor);
    for(auto *value:{&router_raw,&sorted_weights,&routed_silu,&shared_silu,&weighted_raw})visitor(*value);
    for(auto &value:sums)visitor(value);
  }
};
struct BFloatResult final {Value output;BFloatPorts ports;Boundary boundary;};
inline BFloatResult append_bfloat(sem::Fragment &f,Value input,const Parameters &parameters) {
  const auto &activation=f.description(input),&router=f.description(parameters.router);
  if(activation.rank()!=2||activation.dimensions()[0]!=1||!activation.is_plain()||!activation.is_dense()||
      activation.data_type()!=c::DataType::kBFloat16||router.rank()!=2||!router.is_plain()||!router.is_dense()||
      router.data_type()!=c::DataType::kBFloat16||router.dimensions()[0]<=0||
      router.dimensions()[1]!=activation.dimensions()[1])
    throw std::runtime_error("original BF16 MoE needs one BF16 input row and its actual BF16 router");
  const auto hidden=activation.dimensions()[1],experts=router.dimensions()[0];
  const auto &gate_bank=f.description(parameters.banks[0]);
  if(hidden<=0||gate_bank.rank()!=3||gate_bank.dimensions()[0]<=0||gate_bank.dimensions()[0]>experts||
      gate_bank.dimensions()[1]<=0||gate_bank.dimensions()[2]!=hidden)
    throw std::runtime_error("invalid original BF16 selected expert geometry");
  const auto slots=gate_bank.dimensions()[0],intermediate=gate_bank.dimensions()[1];
  if(f.description(parameters.map)!=sem::dense({experts},c::DataType::kInt32)||
      f.description(parameters.shared_gate)!=sem::dense({1,hidden},c::DataType::kBFloat16))
    throw std::runtime_error("original BF16 MoE needs the full global mapping and BF16 shared gate");
  const auto &shared_gate=f.description(parameters.shared[0]);
  if(shared_gate.rank()!=2||shared_gate.dimensions()[0]<=0||shared_gate.dimensions()[1]!=hidden)
    throw std::runtime_error("invalid original BF16 shared expert geometry");
  const auto shared_intermediate=shared_gate.dimensions()[0];
  for(std::size_t leg=0;leg<3;++leg) {
    const auto n=leg==2?hidden:intermediate,k=leg==2?intermediate:hidden;
    const auto shared_n=leg==2?hidden:shared_intermediate,shared_k=leg==2?shared_intermediate:hidden;
    if(f.description(parameters.banks[leg])!=sem::dense({slots,n,k},c::DataType::kBFloat16)||
        f.description(parameters.shared[leg])!=sem::dense({shared_n,shared_k},c::DataType::kBFloat16)||
        std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.banks[leg].ordinal();}))
      throw std::runtime_error("original BF16 experts need compact dynamic banks and actual BF16 shared matrices");
  }
  if(std::none_of(f.inputs.begin(),f.inputs.end(),[&](Value value){return value.ordinal()==parameters.map.ordinal();}))
    throw std::runtime_error("BF16 expert mapping must be an ordinary runtime input");
  const auto cast=[&](Value value,c::DataType type) {
    return f.description(value).data_type()==type?value:f.one(c::Operation(c::Cast{.output_type=type}),{value});
  };
  const auto linear=[&](Value actual,Value weight) {
    return cast(f.one(c::model_ops::bfloat_linear_f32(),{actual,weight}),c::DataType::kBFloat16);
  };
  BFloatResult result;auto &ports=result.ports;auto &values=ports.values;
  values.input=input;values.map=parameters.map;
  ports.router_raw=f.one(c::model_ops::bfloat_linear_f32(),{input,parameters.router});
  values.logits=cast(ports.router_raw,c::DataType::kBFloat16);
  values.probabilities=f.one(c::Operation(c::Softmax{}),{cast(values.logits,c::DataType::kFloat32)});
  const auto top=f.append(c::Operation(c::TopK{.count=std::size_t(slots),.stable=true}),{values.probabilities},2);
  values.selected_scores=top[0];values.ids=top[1];values.sum=sum(f,values.selected_scores,-1);
  // HF normalizes selected F32 probabilities directly. The GGUF norm_w
  // floor in append() is not part of this original BF16 contract.
  values.denominator=values.sum;
  values.weights=core::binary(f,c::BinaryKind::kDivide,values.selected_scores,values.sum);
  result.boundary={input,values.ids,values.weights,parameters.banks,parameters.map};
  values.local_ids=f.one(c::Operation(c::Gather{}),{parameters.map,values.ids});
  // Supply's sorted unique globals define compact order. Move weights, not
  // the large banks, from TopK score order into that ascending global order.
  ports.sorted_weights=f.one(c::Operation(c::Scatter{}),{
      core::filled(f,sem::dense({slots}),0.F),core::reshape(f,values.local_ids,{slots}),
      core::reshape(f,values.weights,{slots})});
  const auto grouped=[&](Value actual,std::size_t leg) {
    const auto rhs=f.one(c::Operation(c::View::permute(std::array<std::int32_t,3>{0,2,1}).value()),{parameters.banks[leg]});
    // The mature CUDA route broadcasts RHS batches over activation rows,
    // not the reverse. Materialize only the tiny repeated input, never W.
    if(f.description(actual).dimensions()[0]!=slots) {
      const auto width=f.description(actual).dimensions()[2];
      actual=f.one(c::Operation(c::View::broadcast_to(c::TensorShape::create(std::array{slots,std::int64_t{1},width}).value()).value()),{actual});
      actual=f.one(c::Operation(c::Copy{}),{actual});
    }
    return f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),{actual,rhs});
  };
  const auto row=core::reshape(f,input,{1,1,hidden});
  values.routed.gate=grouped(row,0);values.routed.up=grouped(row,1);
  ports.routed_silu=core::unary(f,c::UnaryKind::kSilu,values.routed.gate);
  values.routed.activation=core::binary(f,c::BinaryKind::kMultiply,ports.routed_silu,values.routed.up);
  values.routed.down=grouped(values.routed.activation,2);
  ports.weighted_raw=core::binary(f,c::BinaryKind::kMultiply,cast(values.routed.down,c::DataType::kFloat32),
      core::reshape(f,ports.sorted_weights,{slots,1,1}));
  values.weighted=cast(ports.weighted_raw,c::DataType::kBFloat16);
  auto total=f.constant(sem::dense({1,hidden},c::DataType::kBFloat16),sem::bytes(std::vector<std::uint16_t>(std::size_t(hidden),0)));
  for(std::int64_t local=0;local<slots;++local) {
    const auto selected=f.one(c::Operation(c::View::slice(0,local,1).value()),{values.weighted});
    total=core::binary(f,c::BinaryKind::kAdd,total,core::reshape(f,selected,{1,hidden}));
    ports.sums.push_back(total); // One BF16 store per ascending expert add.
  }
  values.routed_sum=total;
  values.shared.gate=linear(input,parameters.shared[0]);values.shared.up=linear(input,parameters.shared[1]);
  ports.shared_silu=core::unary(f,c::UnaryKind::kSilu,values.shared.gate);
  values.shared.activation=core::binary(f,c::BinaryKind::kMultiply,ports.shared_silu,values.shared.up);
  values.shared.down=linear(values.shared.activation,parameters.shared[2]);
  values.shared_gate_raw=linear(input,parameters.shared_gate);
  values.shared_gate_sigmoid=core::unary(f,c::UnaryKind::kSigmoid,values.shared_gate_raw);
  values.shared_gated=core::binary(f,c::BinaryKind::kMultiply,values.shared.down,values.shared_gate_sigmoid);
  values.merged=core::binary(f,c::BinaryKind::kAdd,values.routed_sum,values.shared_gated);
  result.output=values.merged;return result;
}
} // namespace qwen24_original_moe_fragment
