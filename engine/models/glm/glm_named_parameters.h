#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "glm_gguf_model_types.h"
#include "../detail/component_definition.h"

// Existing immutable named-role wiring, after caller-owned authentication.
// This constructs ordinary graph constants; it does not load an artifact.
namespace glm_gguf_named_parameters {
namespace sem=hyle::inference::model_definition;
using Value=sem::ValueId;
using Block=glm_gguf_model_types::Block;
struct Parameters final {
  std::array<Block,2> block; // 0: original target77, 1: original NextN78.
  Value eh,embedding_norm,hidden_norm,shared_norm,embedding,head,target_norm;
  glm_gguf_model_types::NextnParameters nextn() const {
    return {block[1],eh,embedding_norm,hidden_norm,shared_norm,embedding,head};
  }
};
template<class Lookup>
Parameters parameter_values(const Lookup &value,std::int64_t target_layer,std::int64_t nextn_layer) {
  if(target_layer<0||nextn_layer<=target_layer)throw std::runtime_error("converted source requires its actual target/NextN layer names");
  Parameters result;
  for(std::size_t index=0;index<2;++index) {
    const auto name="blk."+std::to_string(index==0?target_layer:nextn_layer)+".";
    result.block[index]={value(name+"attn_norm.weight"),value(name+"attn_q_a.weight"),
      value(name+"attn_q_a_norm.weight"),value(name+"attn_q_b.weight"),value(name+"attn_kv_a_mqa.weight"),
      value(name+"attn_kv_a_norm.weight"),value(name+"attn_k_b.weight"),value(name+"attn_v_b.weight"),
      value(name+"attn_output.weight"),value(name+"ffn_norm.weight"),value(name+"ffn_gate_inp.weight"),
      value(name+"exp_probs_b.bias"),value(name+"ffn_gate_shexp.weight"),value(name+"ffn_up_shexp.weight"),
      value(name+"ffn_down_shexp.weight"),value(name+"indexer.attn_k.weight"),value(name+"indexer.attn_q_b.weight"),
      value(name+"indexer.k_norm.weight"),value(name+"indexer.k_norm.bias"),value(name+"indexer.proj.weight")};
  }
  auto &p=result;
  const auto nextn="blk."+std::to_string(nextn_layer)+".nextn.";
  p.eh=value(nextn+"eh_proj.weight");p.embedding_norm=value(nextn+"enorm.weight");
  p.hidden_norm=value(nextn+"hnorm.weight");p.shared_norm=value(nextn+"shared_head_norm.weight");
  p.embedding=value("token_embd.weight");p.head=value("output.weight");p.target_norm=value("output_norm.weight");
  return result;
}
template<class Role> struct Constant final {Value value;Role role;};
template<class Bound,class ParameterSet> struct NamedConnection final {
  using Role=std::remove_cvref_t<decltype(std::declval<const Bound&>().at(std::string_view{}))>;
  // Strong source and immutable role leases outlive every borrowed name/span.
  // Reverse destruction retires constants before their authenticated source.
  Bound bound;
  ParameterSet parameters;
  std::vector<Constant<Role>> constants;
};
template<class Bound> using Connected=NamedConnection<Bound,Parameters>;
template<class ParameterSet,class Bound,class BuildParameters>
NamedConnection<Bound,ParameterSet> connect_with(sem::Fragment &f,Bound bound,const BuildParameters &build_parameters) {
  if(!bound.source)throw std::runtime_error("converted actual named source is absent");
  // Data::load has already authenticated the explicit identity and every
  // native/descriptor/name/file/SHA. Do not reselect an artifact by model ID.
  NamedConnection<Bound,ParameterSet> result{std::move(bound),{}, {}};
  const auto count=result.bound.source->size();result.constants.reserve(count);
  std::unordered_map<std::string_view,Value> values;values.reserve(count);
  for(std::size_t slot=0;slot<count;++slot) {
    const auto &s=result.bound.source->spec(result.bound.source->name(slot));
    const auto &d=s.description;const auto &role=result.bound.at(s.name);
    const auto extent=d.required_bytes();
    if(!extent||*extent!=s.bytes||role.name!=s.name||role.description!=d||!role.owner||!role.lease||
        role.identity!=&result.bound.source->identity()||
        result.bound.source->description(s.name)!=d)
      throw std::runtime_error("B09 original named bank descriptor/lease differs");
    const auto tensor=role.lease.tensor(d);
    if(!tensor||tensor->storage().resource()!=role.owner.ref())
      throw std::runtime_error("B09 original named bank belongs to another resource");
    const auto value=f.constant(d,{});
    if(!values.emplace(s.name,value).second)throw std::runtime_error("duplicate B09 source name");
    result.constants.push_back({value,role});
  }
  const auto value=[&](std::string_view name) {
    const auto it=values.find(name);if(it==values.end())throw std::runtime_error("B09 model role is not connected");
    return it->second;
  };
  result.parameters=build_parameters(value);
  return result;
}
template<class Bound>
Connected<Bound> connect(sem::Fragment &f,Bound bound,std::int64_t target_layer,std::int64_t nextn_layer) {
  const auto build_parameters=[&](const auto &lookup) {
    return parameter_values(lookup,target_layer,nextn_layer);
  };
  return connect_with<Parameters>(f,std::move(bound),build_parameters);
}
} // namespace glm_gguf_named_parameters
