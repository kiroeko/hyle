#pragma once

#include <limits>

#include "dspark_original_draft_program.h"
#include "model_sampling_fragment.h"
#include "../../engine/inference/mtp_acceptance.h"

// Source-specific Markov head wiring only. All distribution, draws and
// speculative acceptance mathematics are shared with the inference engine.
namespace dspark_stochastic_head {
namespace own = dspark_original_formal_roles;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace c = hyle::compute;
namespace s = hyle::inference::sampling;
using Value = sem::ValueId;
using s::fixed;
struct Proposal final {
  sem::Fragment fragment;
  std::vector<dspark_original_draft_program::Constant> constants;
  std::size_t ids = 0;
  std::vector<std::size_t> q, valid, adjusted, markov, confidence;
};
// Physical role storage and visible/projection activation types are separate.
// The source adapter selects its real store boundary, never a provider label.
struct HeadSource final {
  std::int64_t hidden=0,vocabulary=0;
  c::DataType hidden_type=c::DataType::kUnknown;
  c::DataType projection_input_type=c::DataType::kUnknown;
  std::array<own::ReadonlyRole,3> roles; // W1,W2,confidence: original BF16
};
inline Proposal proposal(const HeadSource &source,std::size_t count,s::Policy policy,
    const c::ModelArtifactProfileDescription *profile) {
  if(!count||count>5)throw std::runtime_error("DSpark stochastic head needs1..5 trained draft rows");
  const auto floating=[](auto type){return type==c::DataType::kBFloat16||type==c::DataType::kFloat32;};
  if(!profile||source.hidden<=0||source.vocabulary!=dspark_original_draft_fragment::kVocabulary||
      !floating(source.hidden_type)||!floating(source.projection_input_type))
    throw std::runtime_error("DSpark stochastic source has no actual fixed head/storage contract");
  Proposal result;result.fragment.profile=profile;result.fragment.region=0x0ca34000+count;
  result.fragment.phase=c::WorkloadPhase::kDecode;auto &f=result.fragment;
  model_sampling_fragment::Builder r(f);
  const auto base=r.input(sem::dense({5,source.vocabulary}));
  auto previous=r.input(sem::dense({1},c::DataType::kInt32));
  const auto keys=r.input(sem::dense({std::int64_t(count),4},c::DataType::kUInt32));
  const auto hidden=r.input(sem::dense({5,source.hidden},source.hidden_type));
  const auto &first=source.roles[0];
  if(first.description.rank()!=2)throw std::runtime_error("DSpark Markov source must be a rank2 BF16 vocabulary table");
  const auto rank=first.description.dimensions()[1];
  if(rank<=0||source.hidden>(std::numeric_limits<std::int32_t>::max)()||
      rank>(std::numeric_limits<std::int32_t>::max)()-source.hidden)
    throw std::runtime_error("DSpark Markov/confidence source geometry exceeds I32");
  const auto parameter=[&](std::size_t index,const c::TensorDesc &expected) {
    return dspark_original_draft_program::immutable(f,result.constants,source.roles[index],expected);
  };
  const auto markov_shape=sem::dense({source.vocabulary,rank},c::DataType::kBFloat16);
  const auto w1=parameter(0,markov_shape),w2=parameter(1,markov_shape);
  const auto confidence=parameter(2,sem::dense({1,source.hidden+rank},c::DataType::kBFloat16));
  const auto cast=[&](Value value,c::DataType type){return f.description(value).data_type()==type?value:r.cast(value,type);};
  auto ids=previous;
  const auto publish=[&](Value value) {return r.publish(r.copy(value));};
  for(std::size_t row=0;row<count;++row) {
    // This is the sampled preceding ID, never the old greedy head's ID.
    const auto markov=r.one(c::Gather{}, {w1,previous});
    const auto visible_markov=cast(markov,source.hidden_type);
    const auto bias=r.one(c::model_ops::dense_linear_f32(), {cast(visible_markov,source.projection_input_type),w2});
    const auto logits=r.binary(c::BinaryKind::kAdd,
        r.copy(r.slice(base,0,fixed(std::int64_t(row)),fixed(1))),bias);
    const auto distribution=s::distribution(r,logits,policy);
    const auto identity=r.copy(r.slice(keys,0,fixed(std::int64_t(row)),fixed(1)));
    previous=r.one(c::sampling_ops::categorical(), {distribution.log_probabilities,identity});
    ids=r.one(c::Concat{.axis=0}, {ids,previous});
    const auto confidence_input=r.one(c::Concat{.axis=1},
        {r.copy(r.slice(hidden,0,fixed(std::int64_t(row)),fixed(1))),visible_markov});
    result.confidence.push_back(publish(r.one(c::model_ops::dense_linear_f32(), {cast(confidence_input,source.projection_input_type),confidence})));
    result.q.push_back(publish(distribution.log_probabilities));result.valid.push_back(publish(distribution.valid));
    result.adjusted.push_back(publish(logits));result.markov.push_back(publish(markov));
  }
  result.ids=publish(ids);
  return result;
}
inline Proposal proposal(const own::Bound &bound,std::size_t count,s::Policy policy,
    const c::ModelArtifactProfileDescription *profile) {
  if(!bound.source)throw std::runtime_error("DSpark stochastic head needs its authenticated original source");
  bound.source->geometry().validate();
  const HeadSource source{bound.source->geometry().hidden,dspark_original_draft_fragment::kVocabulary,
      c::DataType::kBFloat16,c::DataType::kBFloat16,
      {bound.at(own::Section::kDraft,2,"markov_head.markov_w1.weight"),
       bound.at(own::Section::kDraft,2,"markov_head.markov_w2.weight"),
       bound.at(own::Section::kDraft,2,"confidence_head.proj.weight")}};
  return proposal(source,count,policy,dspark_original_draft_program::checked_profile(*bound.source,profile));
}

struct Verification final {
  sem::Fragment fragment;
  std::vector<dspark_original_draft_program::Constant> constants;
  std::vector<std::size_t> accepted, final_ids, valid;
};
inline Verification verification(std::size_t count,s::Policy policy,
    const c::ModelArtifactProfileDescription *profile) {
  if(count>5) throw std::runtime_error("DSpark stochastic verifier exceeds trained block5");
  Verification result;auto &f=result.fragment;f.profile=profile;f.region=0x0ca35000+count;
  f.phase=c::WorkloadPhase::kDecode;model_sampling_fragment::Builder r(f);
  const auto logits=r.input(sem::dense({std::int64_t(count+1),dspark_original_draft_fragment::kVocabulary}));
  const auto keys=r.input(sem::dense({std::int64_t(count+1),4},c::DataType::kUInt32));
  const auto ids=count?r.input(sem::dense({std::int64_t(count)},c::DataType::kInt32)):Value{};
  const auto decision=s::speculative_prefix(r,logits,keys,ids,policy);
  for(const auto value:decision.accepted)result.accepted.push_back(r.publish(r.copy(value)));
  for(const auto value:decision.final_ids)result.final_ids.push_back(r.publish(r.copy(value)));
  for(const auto value:decision.valid)result.valid.push_back(r.publish(r.copy(value)));
  return result;
}
} // namespace dspark_stochastic_head
