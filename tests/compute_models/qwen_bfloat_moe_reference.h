#pragma once

#include <array>
#include <cstring>
#include <numeric>
#include <string>

#include "qwen24_original_moe_fragment.h"
#include "qwen24_original_nextn_roles.h"
#include "qwen_bfloat_original_reference.h"

// Actual-predecessor reference for the original BF16 selected-expert branch.
// Source supplies only independently authenticated selected original planes;
// no whole expert decode, substitute weight bank or native oracle is used.
namespace qwen_bfloat_moe_reference {
namespace model=qwen24_original_moe_fragment;
namespace bf=qwen_bfloat_original_reference;
namespace sem=d5_qwen38_24t_semantic_fragments;
namespace c=hyle::compute;
using Value=sem::ValueId;
using Leg=qwen24_original_nextn_roles::Leg;
struct Matrix final {std::span<const std::byte> bytes;std::size_t outputs,width;};
struct Parameters final {Matrix router,shared_gate;std::array<Matrix,3> shared;};
template<class OriginalData> Matrix original_matrix(const OriginalData &source,std::string_view name) {
  const auto &description=source.description(name);
  if(description.rank()!=2||!description.is_plain()||!description.is_dense()||
      description.data_type()!=c::DataType::kBFloat16||
      description.dimensions()[0]<=0||description.dimensions()[1]<=0||
      std::uint64_t(description.dimensions()[0])>std::numeric_limits<std::size_t>::max()||
      std::uint64_t(description.dimensions()[1])>std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("original BF16 MoE static role is not its actual positive BF16 matrix");
  return {source.bytes(name),std::size_t(description.dimensions()[0]),std::size_t(description.dimensions()[1])};
}
template<class OriginalData> Parameters from_source(const OriginalData &source,std::string_view router,
    std::string_view shared_gate,const std::array<std::string_view,3> &shared) {
  return {original_matrix(source,router),original_matrix(source,shared_gate),
      {original_matrix(source,shared[0]),original_matrix(source,shared[1]),original_matrix(source,shared[2])}};
}
// append_bfloat already creates these four F32 linear results followed by
// explicit BF16 Casts. Expose the existing producers alongside normal ports;
// batched BF16 MatMul has only its actual BF16 result, so it is checked against
// the F32-accumulation interval followed by BF16 rounding instead.
struct LinearStores final {
  std::array<Value,4> raw; // shared gate projection,up,down,shared scalar gate.
  template<class Visitor> void visit_values(Visitor &&visitor){for(auto &value:raw)visitor(value);}
};
inline LinearStores linear_stores(const sem::Fragment &fragment,const model::BFloatResult &result) {
  LinearStores ports;
  const std::array stored{result.ports.values.shared.gate,result.ports.values.shared.up,
      result.ports.values.shared.down,result.ports.values.shared_gate_raw};
  for(std::size_t slot=0;slot<stored.size();++slot) {
    const auto producer=std::find_if(fragment.nodes.begin(),fragment.nodes.end(),[&](const auto &node) {
      return node.outputs.size()==1&&node.outputs[0].ordinal()==stored[slot].ordinal();
    });
    if(producer==fragment.nodes.end()||producer->inputs.size()!=1||
        !producer->operation.get_if<c::Cast>()||
        producer->operation.get_if<c::Cast>()->output_type!=c::DataType::kBFloat16||
        fragment.description(producer->inputs[0]).data_type()!=c::DataType::kFloat32)
      throw std::runtime_error("original BF16 shared projection lost its actual F32/BF16 boundary");
    ports.raw[slot]=producer->inputs[0];
  }
  return ports;
}
namespace detail {
template<class T,class Actual> std::vector<T> values(const sem::Fragment &fragment,Value value,
    c::DataType type,Actual &actual) {
  if(value.ordinal()>=fragment.descriptions.size()||fragment.description(value).data_type()!=type)
    throw std::runtime_error("BF16 MoE observed value has a different actual dtype");
  const auto raw=actual(value);
  const auto extent=fragment.description(value).required_bytes();
  if(!extent||raw.size()!=*extent||raw.size()%sizeof(T))throw std::runtime_error("BF16 MoE observed value extent differs");
  std::vector<T> result(raw.size()/sizeof(T));if(!raw.empty())std::memcpy(result.data(),raw.data(),raw.size());return result;
}
inline bool exact(float actual,float expected,std::string_view role) {
  if(!std::isfinite(actual)||!std::isfinite(expected)||std::bit_cast<std::uint32_t>(actual)!=std::bit_cast<std::uint32_t>(expected)) {
    ADD_FAILURE()<<role<<" F32 source/copy/operation differs";return false;
  }
  return true;
}
inline bool local(float actual,double center,std::string_view role) {
  if(!std::isfinite(actual)||!std::isfinite(center)||std::abs(double(actual)-center)>8.e-5+4.e-5*std::abs(center)) {
    ADD_FAILURE()<<role<<" original local F32 gate";return false;
  }
  return true;
}
inline bool stored_projection(std::span<const std::uint16_t> actual,
    std::span<const std::uint16_t> input,const Matrix &matrix,std::string_view role,bool full) {
  if(!matrix.outputs||!matrix.width||matrix.outputs>std::numeric_limits<std::size_t>::max()/matrix.width/2||
      matrix.bytes.size()!=matrix.outputs*matrix.width*2||input.size()!=matrix.width||actual.size()!=matrix.outputs) {
    ADD_FAILURE()<<role<<" original BF16 batched projection extent";return false;
  }
  const auto widened=bf::widen(actual);const auto chosen=bf::detail::candidates(widened);
  const auto check=[&](std::size_t index) {
    const auto expected=bf::detail::dot(input,matrix.bytes.subspan(index*matrix.width*2,matrix.width*2));
    const auto lower=bf::round(std::nextafter(float(expected.center-expected.bound),-std::numeric_limits<float>::infinity()));
    const auto upper=bf::round(std::nextafter(float(expected.center+expected.bound),std::numeric_limits<float>::infinity()));
    if(widened[index]<bf::widen(lower)||widened[index]>bf::widen(upper)) {
      ADD_FAILURE()<<role<<" output["<<index<<"]="<<widened[index]<<" outside F32 accumulation then BF16 interval; F64="<<expected.center<<" bound="<<expected.bound;return false;
    }
    return true;
  };
  if(full){for(std::size_t index=0;index<actual.size();++index)if(!check(index))return false;}
  else for(auto index:chosen)if(!check(index))return false;
  return true;
}
inline bool routing(std::span<const std::uint16_t> logits,std::span<const float> probabilities,
    std::span<const std::int32_t> ids,std::span<const float> selected,std::span<const float> total,
    std::span<const float> denominator,std::span<const float> weights) {
  if(logits.empty()||probabilities.size()!=logits.size()||ids.empty()||ids.size()>logits.size()||
      selected.size()!=ids.size()||weights.size()!=ids.size()||total.size()!=1||denominator.size()!=1) {
    ADD_FAILURE()<<"original BF16 MoE routing extent";return false;
  }
  const auto actual=bf::widen(logits);const auto maximum=*std::max_element(actual.begin(),actual.end());
  std::vector<double> exponentials(actual.size());bf::detail::Sum sum;
  for(std::size_t i=0;i<actual.size();++i){exponentials[i]=std::exp(double(actual[i])-maximum);sum.add(exponentials[i]);}
  for(std::size_t i=0;i<actual.size();++i) {
    if(probabilities[i]<0||probabilities[i]>1||!local(probabilities[i],exponentials[i]/sum.value(),"BF16-logit F32 softmax"))return false;
  }
  std::vector<std::size_t> order(actual.size());std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto left,auto right){return probabilities[left]>probabilities[right];});
  bf::detail::Sum selected_sum;double absolute=0;
  for(std::size_t slot=0;slot<ids.size();++slot) {
    if(ids[slot]<0||std::size_t(ids[slot])!=order[slot]||
        !exact(selected[slot],probabilities[order[slot]],"BF16 actual-probability stable TopK"))return false;
    selected_sum.add(selected[slot]);absolute+=std::abs(double(selected[slot]));
  }
  const auto g64=bf::detail::gamma(ids.size(),0x1p-53);
  const auto upper=bf::detail::upper(absolute/(1-g64));
  const auto oracle=bf::detail::add(bf::detail::distance(selected_sum.value(),selected_sum.plain),bf::detail::multiply(g64,upper));
  const auto bound=bf::detail::add(bf::detail::multiply(bf::detail::gamma(ids.size(),0x1p-24),upper),oracle);
  if(!std::isfinite(total[0])||total[0]<=0||bf::detail::distance(total[0],selected_sum.value())>bound||
      !exact(denominator[0],total[0],"BF16 direct selected sum (no GGUF floor)")) {
    ADD_FAILURE()<<"original BF16 selected F32 sum/denominator";return false;
  }
  for(std::size_t slot=0;slot<ids.size();++slot)
    if(weights[slot]<0||weights[slot]>1||!local(weights[slot],double(selected[slot])/total[0],"original BF16 F32 route weight"))return false;
  return true;
}
} // namespace detail

// Source is the existing finite supply protocol: expert_description(Leg) and
// expert_bytes(Leg,actualGlobalID). Missing actual expert bytes must throw; this
// observer neither downloads nor substitutes an unselected/zero expert.
template<class Source,class Actual>
void observe(const sem::Fragment &fragment,const Parameters &parameters,const model::BFloatResult &result,
    const LinearStores &linear,const Source &source,Actual &&actual,bool full,std::size_t &full_values) {
  const auto bits=[&](Value value){return detail::values<std::uint16_t>(fragment,value,c::DataType::kBFloat16,actual);};
  const auto floats=[&](Value value){return detail::values<float>(fragment,value,c::DataType::kFloat32,actual);};
  const auto integers=[&](Value value){return detail::values<std::int32_t>(fragment,value,c::DataType::kInt32,actual);};
  const auto &ports=result.ports;const auto &values=ports.values;
  const auto input=bits(values.input),logits=bits(values.logits);
  const auto router_raw=floats(ports.router_raw);
  const auto probabilities=floats(values.probabilities),selected=floats(values.selected_scores),total=floats(values.sum);
  const auto denominator=floats(values.denominator),weights=floats(values.weights),sorted_weights=floats(ports.sorted_weights);
  const auto ids=integers(values.ids),local=integers(values.local_ids),mapping=integers(values.map);
  ASSERT_EQ(parameters.router.width,input.size());ASSERT_EQ(parameters.router.outputs,logits.size());
  ASSERT_TRUE(bf::expect_projection(router_raw,input,parameters.router.bytes,parameters.router.outputs,parameters.router.width,"original BF16 full router",true));
  ASSERT_TRUE(bf::expect_store(logits,router_raw,"original BF16 router store before F32 softmax"));
  ASSERT_TRUE(detail::routing(logits,probabilities,ids,selected,total,denominator,weights));
  if(full)full_values+=router_raw.size();
  ASSERT_EQ(mapping.size(),parameters.router.outputs);ASSERT_EQ(local.size(),ids.size());ASSERT_EQ(sorted_weights.size(),ids.size());
  auto globals=ids;std::sort(globals.begin(),globals.end());
  ASSERT_EQ(std::unique(globals.begin(),globals.end()),globals.end());
  std::vector<std::int32_t> expected_map(mapping.size(),-1);
  for(std::size_t slot=0;slot<globals.size();++slot)expected_map[std::size_t(globals[slot])]=std::int32_t(slot);
  ASSERT_EQ(mapping,expected_map);
  for(std::size_t slot=0;slot<ids.size();++slot) {
    ASSERT_EQ(local[slot],mapping[std::size_t(ids[slot])]);
    ASSERT_TRUE(detail::exact(sorted_weights[std::size_t(local[slot])],weights[slot],"BF16 route weights scatter to ascending globals"));
  }
  const auto gate=bits(values.routed.gate),up=bits(values.routed.up),activation=bits(values.routed.activation),down=bits(values.routed.down);
  const auto grouped=[&](Leg leg,std::span<const std::uint16_t> projected,std::span<const std::uint16_t> previous,bool broadcast,std::string_view label) {
    const auto description=source.expert_description(leg);
    if(description.rank()!=3||description.data_type()!=c::DataType::kBFloat16||!description.is_plain()||
        description.dimensions()[0]!=std::int64_t(mapping.size())||description.dimensions()[1]<=0||description.dimensions()[2]<=0) {
      ADD_FAILURE()<<label<<" source is not its actual full-domain BF16 descriptor";return false;
    }
    const auto outputs=std::size_t(description.dimensions()[1]),width=std::size_t(description.dimensions()[2]);
    if(width>std::size_t(std::numeric_limits<std::int64_t>::max()/2)||
        outputs>std::numeric_limits<std::size_t>::max()/width/2||
        globals.size()>std::numeric_limits<std::size_t>::max()/outputs||
        globals.size()>std::numeric_limits<std::size_t>::max()/width||projected.size()!=globals.size()*outputs||
        previous.size()!=(broadcast?width:globals.size()*width)) {
      ADD_FAILURE()<<label<<" actual selected projection shape differs";return false;
    }
    const auto plane=outputs*width*2;
    const auto strides=description.byte_strides();
    if(strides[2]!=2||strides[1]!=std::int64_t(width*2)||strides[0]<=0||std::uint64_t(strides[0])<plane) {
      ADD_FAILURE()<<label<<" source expert planes are not the exact original BF16 rows";return false;
    }
    for(std::size_t slot=0;slot<globals.size();++slot) {
      const auto original=source.expert_bytes(leg,globals[slot]);
      if(!detail::stored_projection(projected.subspan(slot*outputs,outputs),
          previous.subspan(broadcast?0:slot*width,width),{original,outputs,width},label,full))return false;
    }
    if(full)full_values+=projected.size();return true;
  };
  ASSERT_TRUE(grouped(Leg::kGate,gate,input,true,"original BF16 selected gate"));
  ASSERT_TRUE(grouped(Leg::kUp,up,input,true,"original BF16 selected up"));
  const auto silu=bits(ports.routed_silu);
  ASSERT_TRUE(bf::expect_unary(silu,gate,true,"actual routed BF16 SiLU"));
  ASSERT_TRUE(bf::expect_binary(activation,silu,up,true,"actual routed BF16 activation product"));
  ASSERT_TRUE(grouped(Leg::kDown,down,activation,false,"original BF16 selected down"));
  const auto weighted_raw=floats(ports.weighted_raw);
  const auto weighted=bits(values.weighted);
  ASSERT_EQ(down.size(),globals.size()*input.size());ASSERT_EQ(weighted_raw.size(),down.size());
  for(std::size_t slot=0;slot<globals.size();++slot)for(std::size_t column=0;column<input.size();++column) {
    const auto index=slot*input.size()+column;
    ASSERT_TRUE(detail::exact(weighted_raw[index],float(double(bf::widen(down[index]))*sorted_weights[slot]),"actual down-after F32 route product"));
  }
  ASSERT_TRUE(bf::expect_store(weighted,weighted_raw,"actual weighted expert BF16 store"));
  ASSERT_EQ(ports.sums.size(),globals.size());std::vector<std::uint16_t> previous(input.size(),0);
  for(std::size_t slot=0;slot<globals.size();++slot) {
    const auto folded=bits(ports.sums[slot]);
    ASSERT_TRUE(bf::expect_binary(folded,previous,std::span(weighted).subspan(slot*input.size(),input.size()),false,"ascending-global BF16 expert fold"));previous=folded;
  }
  const auto routed_sum=bits(values.routed_sum);ASSERT_EQ(routed_sum,previous);
  const auto shared_gate=bits(values.shared.gate),shared_up=bits(values.shared.up),shared_activation=bits(values.shared.activation),shared_down=bits(values.shared.down);
  const std::array shared_values{std::span<const std::uint16_t>(shared_gate),std::span<const std::uint16_t>(shared_up),std::span<const std::uint16_t>(shared_down)};
  for(std::size_t leg=0;leg<parameters.shared.size();++leg) {
    const auto raw=floats(linear.raw[leg]);const auto predecessor=leg==2?std::span<const std::uint16_t>(shared_activation):std::span<const std::uint16_t>(input);
    const auto &matrix=parameters.shared[leg];
    ASSERT_TRUE(bf::expect_projection(raw,predecessor,matrix.bytes,matrix.outputs,matrix.width,"original BF16 shared linear",full));
    ASSERT_TRUE(bf::expect_store(shared_values[leg],raw,"actual shared linear BF16 store"));if(full)full_values+=raw.size();
  }
  const auto shared_silu=bits(ports.shared_silu);
  ASSERT_TRUE(bf::expect_unary(shared_silu,shared_gate,true,"actual shared BF16 SiLU"));
  ASSERT_TRUE(bf::expect_binary(shared_activation,shared_silu,shared_up,true,"actual shared BF16 activation product"));
  const auto shared_scalar_raw=floats(linear.raw[3]);
  const auto shared_scalar=bits(values.shared_gate_raw),sigmoid=bits(values.shared_gate_sigmoid);
  ASSERT_EQ(parameters.shared_gate.outputs,1u);
  ASSERT_TRUE(bf::expect_projection(shared_scalar_raw,input,parameters.shared_gate.bytes,1,parameters.shared_gate.width,"original BF16 shared scalar gate",true));
  ASSERT_TRUE(bf::expect_store(shared_scalar,shared_scalar_raw,"actual shared scalar BF16 store"));
  ASSERT_TRUE(bf::expect_unary(sigmoid,shared_scalar,false,"actual shared BF16 sigmoid"));
  if(full)full_values+=shared_scalar_raw.size();ASSERT_EQ(sigmoid.size(),1u);
  const std::vector<std::uint16_t> expanded_gate(shared_down.size(),sigmoid[0]);const auto shared_gated=bits(values.shared_gated);
  ASSERT_TRUE(bf::expect_binary(shared_gated,shared_down,expanded_gate,true,"actual shared gated BF16 product"));
  const auto merged=bits(values.merged);
  ASSERT_TRUE(bf::expect_binary(merged,routed_sum,shared_gated,false,"actual routed plus shared BF16 merge"));
  ASSERT_EQ(bits(result.output),merged);
}
} // namespace qwen_bfloat_moe_reference
