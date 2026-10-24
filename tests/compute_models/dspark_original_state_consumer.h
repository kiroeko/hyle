#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "dspark_main_history.h"
#include "dspark_complete_graph_fixture.h"
#include "dspark_shared_target_roles.h"
#include "dspark_original_hc_roles.h"
#include "dspark_original_shared_roles.h"
#include "dspark_original_shared_reference.h"
#include "dspark_original_attention_roles.h"
#include "dspark_original_attention_reference.h"
#include "dspark_original_expert_roles.h"

// Cold model assembly only. The caller may supply the matching target's
// shared BF16 vocabulary. Target captures and other stage roles stay explicit.
namespace dspark_original_state_consumer {
namespace b = dspark_target_binding;
namespace h = dspark_main_history;
namespace d = dspark_complete_graph;
namespace vocabulary = dspark_shared_target_roles;
namespace parameters = dspark_original_hc_roles;
namespace shared_ffn = dspark_original_shared_roles;
namespace attention = dspark_original_attention_roles;
namespace expert = dspark_original_expert_roles;
namespace base = cpu_c4_model_branches;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::Result;
using hyle::Status;

using ExternalConstant=vocabulary::Constant;
struct Draft final {
  d::CompleteProgram program;
  std::vector<ExternalConstant> constants;
  std::shared_ptr<const parameters::Data> parameter_source;
  std::shared_ptr<const shared_ffn::Data> shared_source;
  std::shared_ptr<const attention::Data> attention_source;
  std::shared_ptr<const expert::Data> expert_source;
  struct StageChecks final {
    sem::ValueId activation,normalized,rotated,entry,encoded,decoded;
    std::optional<sem::ValueId> raw_projection;
  };
  std::array<StageChecks,3> checkpoints;
};
inline std::size_t parameter_index(std::size_t stage,base::DraftParameter parameter) {
  using P=base::DraftParameter;
  using S=parameters::StageRole;
  using H=parameters::HeadRole;
  switch(parameter) {
  case P::kAttentionNorm:return parameters::index(stage,S::kAttentionNorm);
  case P::kFfnNorm:return parameters::index(stage,S::kFfnNorm);
  case P::kQueryNorm:return parameters::index(stage,S::kQueryNorm);
  case P::kSinks:return parameters::index(stage,S::kSinks);
  case P::kAttentionFn:return parameters::index(stage,S::kAttentionFn);
  case P::kAttentionScale:return parameters::index(stage,S::kAttentionScale);
  case P::kAttentionBase:return parameters::index(stage,S::kAttentionBase);
  case P::kFfnFn:return parameters::index(stage,S::kFfnFn);
  case P::kFfnScale:return parameters::index(stage,S::kFfnScale);
  case P::kFfnBase:return parameters::index(stage,S::kFfnBase);
  default:break;
  }
  if(stage!=3)throw std::runtime_error("original DSpark output role has wrong stage");
  switch(parameter) {
  case P::kOutputNorm:return parameters::index(H::kOutputNorm);
  case P::kHeadFn:return parameters::index(H::kFn);
  case P::kHeadScale:return parameters::index(H::kScale);
  case P::kHeadBase:return parameters::index(H::kBase);
  default:throw std::runtime_error("original DSpark parameter is absent");
  }
}
inline Result<Draft> make_draft(const std::array<b::StageKvRoles,3> &roles,
                                const b::ReadonlyRole &frequencies,
                                d::HeadParameters heads,bool observe_boundaries=false,
                                const vocabulary::Roles *shared_roles=nullptr,
                                const parameters::Roles *parameter_roles=nullptr,
                                const shared_ffn::Roles *shared_ffn_roles=nullptr,
                                const attention::Roles *attention_roles=nullptr,
                                const expert::Roles *expert_roles=nullptr,
                                const base::DraftParameters *complete_parameters=nullptr,
                                std::vector<ExternalConstant> *complete_constants=nullptr) {
  // Reuse actual role qualification and the same codec/position definition.
  HYLE_ASSIGN_OR_RETURN(auto qualified, h::make_history(roles,frequencies,1));
  (void)qualified;
  Draft result;
  if(complete_constants&&!complete_parameters)
    return std::unexpected(Status::InvalidArgument("complete constant directory requires its explicit typed supplier"));
  if(complete_parameters && (!complete_constants||!shared_roles||!heads.readonly_parameter||
      parameter_roles||shared_ffn_roles||attention_roles||expert_roles||
      !complete_parameters->parameter||!complete_parameters->shared_ffn||
      !complete_parameters->attention_projection||!complete_parameters->routed_experts))
    return std::unexpected(Status::InvalidArgument("complete original typed suppliers cannot mix identities or omit banks"));
  if(shared_ffn_roles&&!parameter_roles)
    return std::unexpected(Status::InvalidArgument("original shared FFN journey needs its complete HC/norm parameters"));
  if(attention_roles&&!parameter_roles)
    return std::unexpected(Status::InvalidArgument("original attention journey needs its complete HC/norm parameters"));
  if(expert_roles&&(!attention_roles||!shared_ffn_roles))
    return std::unexpected(Status::InvalidArgument("original experts need complete original attention/shared boundaries"));
  base::DraftParameters supplied_parameters;
  if(complete_parameters)supplied_parameters=*complete_parameters;
  if(parameter_roles) {
    HYLE_RETURN_IF_ERROR(parameters::qualify(*parameter_roles,roles[0].target));
    result.parameter_source=parameter_roles->source;
    supplied_parameters.parameter=[&](sem::Fragment &f,std::size_t stage,base::DraftParameter parameter) {
      const auto index=parameter_index(stage,parameter);
      const auto &role=parameter_roles->at(index);
      const auto value=f.constant(role.description,{});
      result.constants.push_back({value,role});
      return base::ds::ImmutableF32Parameter{value,result.parameter_source->values(index)};
    };
  }
  if(shared_ffn_roles) {
    HYLE_RETURN_IF_ERROR(shared_ffn::qualify(*shared_ffn_roles,roles[0].target));
    result.shared_source=shared_ffn_roles->source;
    supplied_parameters.shared_ffn=[&](sem::Fragment &f,std::size_t stage,
        base::ds::official::SharedFfnRole parameter) {
      shared_ffn::StageRole mapped;
      switch(parameter) {
      case base::ds::official::SharedFfnRole::kGate:mapped=shared_ffn::StageRole::kGate;break;
      case base::ds::official::SharedFfnRole::kUp:mapped=shared_ffn::StageRole::kUp;break;
      case base::ds::official::SharedFfnRole::kDown:mapped=shared_ffn::StageRole::kDown;break;
      default:throw std::runtime_error("original shared FFN parameter is absent");
      }
      const auto index=shared_ffn::index(stage,mapped);
      const auto &role=shared_ffn_roles->at(index);
      const auto value=f.constant(role.description,{});
      result.constants.push_back({value,role});
      const auto source=result.shared_source;
      return base::ds::official::PackedProjectionParameter{value,source->bytes(index),
          [source,index](std::span<const float> actual) {
            const auto &specification=shared_ffn::spec(index);
            return dspark_original_shared_reference::decoded_reference(source->bytes(index),
                std::size_t(specification.outputs),std::size_t(specification.reduction),actual);
          }};
    };
  }
  if(attention_roles) {
    HYLE_RETURN_IF_ERROR(attention::qualify(*attention_roles,roles[0].target));
    result.attention_source=attention_roles->source;
    supplied_parameters.attention_projection=[&](sem::Fragment &f,std::size_t stage,
        base::AttentionProjectionRole parameter) {
      attention::StageRole mapped;
      switch(parameter) {
      case base::AttentionProjectionRole::kQueryA:mapped=attention::StageRole::kQueryA;break;
      case base::AttentionProjectionRole::kQueryB:mapped=attention::StageRole::kQueryB;break;
      case base::AttentionProjectionRole::kOutputA:mapped=attention::StageRole::kOutputA;break;
      case base::AttentionProjectionRole::kOutputB:mapped=attention::StageRole::kOutputB;break;
      default:throw std::runtime_error("original attention projection is absent");
      }
      const auto index=attention::index(stage,mapped);
      const auto &role=attention_roles->at(index);
      const auto value=f.constant(role.description,{});
      result.constants.push_back({value,role});
      const auto source=result.attention_source;
      return base::ds::official::PackedProjectionParameter{value,source->bytes(index),
          [source,index](std::span<const float> actual) {
            const auto &specification=attention::spec(index);
            if(specification.groups==1)return dspark_original_shared_reference::decoded_reference(source->bytes(index),
                std::size_t(specification.outputs),std::size_t(specification.reduction),actual);
            const auto width=std::size_t(specification.groups*specification.reduction);
            if(actual.empty()||actual.size()%width)throw std::runtime_error("original output-a group reference extent");
            std::vector<std::int32_t> ids(actual.size()/std::size_t(specification.reduction));
            for(std::size_t row=0;row<actual.size()/width;++row)
              std::iota(ids.begin()+std::ptrdiff_t(row*std::size_t(specification.groups)),
                  ids.begin()+std::ptrdiff_t((row+1)*std::size_t(specification.groups)),0);
            return dspark_original_attention_reference::decoded_grouped_reference(source->bytes(index),
                std::size_t(specification.groups),std::size_t(specification.outputs),std::size_t(specification.reduction),actual,ids);
          }};
    };
  }
  if(expert_roles) {
    HYLE_RETURN_IF_ERROR(expert::qualify(*expert_roles,roles[0].target));
    if(!expert_roles->source)return std::unexpected(Status::InvalidArgument("original routed expert source missing"));
    result.expert_source=expert_roles->source;
    supplied_parameters.routed_experts=[&](sem::Fragment &f,std::size_t stage) {
      if(stage>=3)throw std::runtime_error("original routed stage outside source");
      const auto router=f.constant(expert_roles->routers[stage].description,{});
      const auto bias=f.constant(expert_roles->biases[stage].description,{});
      result.constants.push_back({router,expert_roles->routers[stage]});
      result.constants.push_back({bias,expert_roles->biases[stage]});
      std::array<sem::ValueId,3> weights;
      for(std::size_t leg=0;leg<3;++leg)weights[leg]=f.input(expert::resident_description(static_cast<expert::Leg>(leg)));
      const auto map=f.input(c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,1>{256}).value());
      return base::ds::official::RoutedExpertParameters{router,bias,weights,map,30};
    };
  }
  d::SharedParameters shared;
  if(shared_roles) {
    HYLE_ASSIGN_OR_RETURN(auto supplied,vocabulary::parameters(*shared_roles,
        roles[0].target,result.constants));
    shared=std::move(supplied);
  }
  base::DraftHistoryInputs histories;
  for(std::size_t stage=0;stage<3;++stage) {
    HYLE_ASSIGN_OR_RETURN(histories[stage].description,
        c::row_cache_carrier_description(h::kEncoding,h::kWindow));
    histories[stage].append_keys=[&,stage](sem::Fragment &f,sem::ValueId activation,
        sem::ValueId history,sem::ValueId positions) {
      const auto weight=f.constant(roles[stage].projection.description,{});
      const auto norm=f.constant(roles[stage].norm.description,{});
      const auto frequency=f.constant(frequencies.description,{});
      result.constants.push_back({weight,roles[stage].projection});
      result.constants.push_back({norm,roles[stage].norm});
      result.constants.push_back({frequency,frequencies});
      auto projected=dspark_stored_projection_fragment::linear(f,activation,weight);
      auto normalized=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-6F}),{projected,norm});
      // Same selected Q8 main-history boundary: BF16 -> last64 RoPE -> BF16.
      auto rounded=base::ds::bfloat_boundary(f,normalized);
      auto shaped=base::core::reshape(f,rounded,{1,base::block,1,b::kKv});
      auto head=base::ds::slice(f,shaped,3,0,448);
      auto tail=base::ds::slice(f,shaped,3,448,64);
      auto rotated=f.one(c::Operation(c::RoPE{.rotary_dimensions=64,.sequence_axis=1,
          .layout=c::RopeLayout::kInterleaved}),{tail,positions,frequency});
      auto flat=base::core::reshape(f,base::ds::concat(f,head,rotated,3),{base::block,b::kKv});
      auto entry=f.one(c::Operation(c::Cast{c::DataType::kBFloat16}),{flat});
      auto proposed=f.one(c::model_ops::row_cache_encode(h::kEncoding),{entry});
      // Only the actual committed main rows are persistent. Proposal rows are
      // encoded and consumed inside this episode, never copied into that ring.
      auto combined=base::ds::concat(f,history,proposed,0);
      std::vector<std::int32_t> ids(std::size_t(h::kWindow+base::block));
      std::iota(ids.begin(),ids.end(),0);
      auto indices=f.constant(sem::dense({h::kWindow+base::block},c::DataType::kInt32),sem::bytes(ids));
      auto decoded=f.one(c::model_ops::row_cache_gather_decode(h::kEncoding),{combined,indices});
      auto fp32=f.one(c::Operation(c::Cast{c::DataType::kFloat32}),{decoded});
      result.checkpoints[stage]={activation,normalized,flat,entry,proposed,fp32};
      if(roles[stage].projection.description.is_plain())result.checkpoints[stage].raw_projection=projected;
      return base::core::reshape(f,fp32,{1,h::kWindow+base::block,1,b::kKv});
    };
  }
  result.program=d::build_flash(heads,&histories,shared_roles?&shared:nullptr,
      (parameter_roles||complete_parameters)?&supplied_parameters:nullptr);
  if(!result.program.encoded || result.program.fragment.inputs.size()!=((expert_roles||complete_parameters)?18u:6u) ||
      result.program.fragment.outputs.size()!=7 || !result.program.fragment.state_ports.empty())
    return std::unexpected(Status::Internal("DSpark encoded draft ports differ from actual read-only history boundary"));
  if(observe_boundaries) {
    // Optional test observations of existing arithmetic, never model state or
    // next-cache outputs. The semantic component still has seven named ports.
    // Observations must survive Invocation reset alongside prior proposals.
    // Materialize required views explicitly rather than binding writers to
    // an alias or borrowing a private workspace generation.
    std::vector<std::pair<std::size_t,sem::ValueId>> observed;
    for(const auto value:result.program.fragment.outputs)observed.push_back({value.ordinal(),value});
    const auto observe=[&](sem::ValueId &value) {
      const auto known=std::find_if(observed.begin(),observed.end(),
          [&](const auto &entry){return entry.first==value.ordinal();});
      if(known!=observed.end()){value=known->second;return;}
      const auto original=value.ordinal();
      value=result.program.fragment.one(c::Operation(c::Copy{}),{value});
      result.program.fragment.outputs.push_back(value);
      observed.push_back({original,value});
    };
    for(auto &stage:result.checkpoints)
      for(auto *value:{&stage.activation,&stage.normalized,&stage.rotated,&stage.entry,&stage.encoded,&stage.decoded}) {
        observe(*value);
      }
    for(auto &stage:result.checkpoints)if(stage.raw_projection)observe(*stage.raw_projection);
    if(result.program.observations)result.program.observations->visit_values(observe);
  }
  if(complete_constants)result.constants.insert(result.constants.end(),complete_constants->begin(),complete_constants->end());
  return result;
}

struct Boundary final {
  std::uint64_t session=0,branch=0,epoch=0;
  std::int64_t next_position=0;
  std::int32_t pending=0;
  std::uint32_t consumed=0;
};
struct VerifiedPrefix final {
  std::uint32_t accepted_drafts=0,consumed=1;
  std::size_t pending_row=0;
};
// Fixed DSpark forward_head emits [current, proposal1, ..., proposal5].
// Target row i consumes that row's token and predicts the next token. This is
// only the consumer's greedy policy; it neither rewrites IDs nor computes a
// replacement pending tensor on the host.
inline Result<VerifiedPrefix> decide_block5(std::span<const std::int32_t> proposed,
    std::span<const std::int32_t> target_ids) {
  if(proposed.size()!=6 || target_ids.size()!=6)
    return std::unexpected(Status::InvalidArgument("DSpark block5 verification needs current+five proposals and six target rows"));
  for(const auto token:proposed)if(token<0 || token>=d::vocabulary)
    return std::unexpected(Status::InvalidArgument("DSpark proposal token is outside the shared vocabulary"));
  for(const auto token:target_ids)if(token<0 || token>=d::vocabulary)
    return std::unexpected(Status::InvalidArgument("DSpark target token is outside the shared vocabulary"));
  VerifiedPrefix result;
  while(result.accepted_drafts<5 &&
      proposed[result.accepted_drafts+1]==target_ids[result.accepted_drafts])
    ++result.accepted_drafts;
  result.consumed=1+result.accepted_drafts;
  result.pending_row=result.accepted_drafts;
  return result;
}
struct TargetVerifier final {
  sem::Fragment fragment{.profile=nullptr,.region=0x4453545645524946ULL};
  sem::ValueId normalized,head,logits,values;
  std::vector<sem::ValueId> inputs,row_ids;
};
inline Result<TargetVerifier> target_verifier(const b::ReadonlyRole &head,
    std::int64_t rows=6) {
  if(rows<1 || rows>6)
    return std::unexpected(Status::InvalidArgument("DSpark target verifier supports one through six actual target rows"));
  if(!dspark_vocabulary_fragment::bfloat_table(head.description,d::vocabulary,b::kHidden)&&
      !dspark_vocabulary_fragment::packed_head(head.description,d::vocabulary,b::kHidden))
    return std::unexpected(Status::InvalidArgument("DSpark verifier needs the actual complete typed head"));
  HYLE_RETURN_IF_ERROR(b::qualify_role(head,"head.weight",head.description));
  TargetVerifier result;auto &f=result.fragment;
  for(std::int64_t row=0;row<rows;++row) {
    const auto input=f.input(sem::dense({1,b::kHidden},c::DataType::kBFloat16));
    result.inputs.push_back(input);
    result.normalized=row?base::ds::concat(f,result.normalized,input,0):input;
  }
  result.head=f.constant(head.description,{});
  result.logits=dspark_vocabulary_fragment::append_head(f,result.normalized,result.head,d::vocabulary,b::kHidden);
  const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{result.logits},2);
  result.values=top[0];
  // Each row's ID has its own ordinary output storage. A selected correction
  // or bonus token can become the next pending State resource directly.
  for(std::int64_t row=0;row<rows;++row) {
    auto id=base::ds::slice(f,top[1],0,row,1);
    id=base::core::reshape(f,id,{1});
    id=f.one(c::Operation(c::Copy{}),{id});f.outputs.push_back(id);result.row_ids.push_back(id);
  }
  for(auto *value:{&result.logits,&result.values}) {
    *value=f.one(c::Operation(c::Copy{}),{*value});f.outputs.push_back(*value);
  }
  return result;
}
struct State final {
  b::TargetIdentity target;
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId,8> entries;
  // The trained target tail owns its complete cache tuple. These ordinary
  // resources are optional for the earlier captured-prefix component tests.
  std::vector<c::StateEntryId> target_entries;
  std::optional<c::StateEntryId> pending_entry;
  static Result<State> create(b::TargetIdentity target,
      std::span<const c::TensorDesc> target_state={},std::size_t backing_bytes=64u<<20) {
    HYLE_RETURN_IF_ERROR(b::qualify_identity(target,{b::kSidecarRepository,b::kSidecarRevision,b::kSidecarFile}));
    State result{.target=std::move(target)};
    HYLE_ASSIGN_OR_RETURN(auto ring,c::row_cache_carrier_description(h::kEncoding,h::kWindow));
    HYLE_ASSIGN_OR_RETURN(auto carry,b::plain(std::array<std::int64_t,2>{1,b::kHidden}));
    HYLE_ASSIGN_OR_RETURN(auto capture,b::plain(std::array<std::int64_t,3>{1,4,b::kHidden}));
    std::vector<c::StateEntrySpec> specs{c::StateEntrySpec{.description=ring},c::StateEntrySpec{.description=ring},
      c::StateEntrySpec{.description=ring},c::StateEntrySpec{.description=carry},
      c::StateEntrySpec{.description=capture},c::StateEntrySpec{.description=capture},
      c::StateEntrySpec{.description=capture},
      c::StateEntrySpec{.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Boundary)}};
    for(const auto &description:target_state)specs.push_back({.description=description});
    if(!target_state.empty()) {
      HYLE_ASSIGN_OR_RETURN(auto pending,c::TensorDesc::dense(c::DataType::kInt32,
          std::array<std::int64_t,1>{1}));
      specs.push_back({.description=pending});
    }
    HYLE_ASSIGN_OR_RETURN(result.schema,c::StateSchema::create(specs));
    HYLE_ASSIGN_OR_RETURN(result.store,c::StateStore::create(result.schema,
        {.max_versions=32,.max_drafts=4,.max_retained_blocks=256,.max_backing_bytes=backing_bytes}));
    for(std::size_t i=0;i<result.entries.size();++i) {
      HYLE_ASSIGN_OR_RETURN(result.entries[i],result.schema.entry(i));
    }
    for(std::size_t i=0;i<target_state.size();++i) {
      HYLE_ASSIGN_OR_RETURN(auto entry,result.schema.entry(result.entries.size()+i));
      result.target_entries.push_back(entry);
    }
    if(!target_state.empty()) {
      HYLE_ASSIGN_OR_RETURN(result.pending_entry,result.schema.entry(result.entries.size()+target_state.size()));
    }
    return result;
  }
  Result<Boundary> boundary(const c::StateVersion &version) const {
    HYLE_ASSIGN_OR_RETURN(auto member,version.member(entries[7]));
    HYLE_ASSIGN_OR_RETURN(auto view,member.view());Boundary result;
    HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(&result,1))));return result;
  }
  Result<c::StateVersion> select(const c::StateVersion &from,
      std::span<const c::ReadLease> history,std::span<const c::ReadLease> captures,
      const b::CaptureFrame &frame,std::uint32_t accepted,std::int32_t pending) {
    HYLE_ASSIGN_OR_RETURN(auto facts,boundary(from));
    HYLE_ASSIGN_OR_RETURN(auto locations,h::placement(frame));(void)locations;
    if(frame.target!=target || facts.session!=frame.session || facts.branch!=frame.branch ||
        facts.epoch>=frame.epoch || facts.next_position!=frame.first_position ||
        history.size()!=std::size_t(frame.rows*4) || captures.size()!=3 ||
        accepted>std::uint32_t(frame.rows) || pending<0 ||
        accepted>std::numeric_limits<std::uint32_t>::max()-facts.consumed)
      return std::unexpected(Status::InvalidArgument("DSpark complete prefix does not belong to this captured frontier"));
    if(!accepted) return from;
    HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(from));
    for(std::size_t i=0;i<4;++i)
      HYLE_RETURN_IF_ERROR(draft.replace(entries[i],history[(accepted-1)*4+i]));
    constexpr auto row_bytes=std::size_t(4*b::kHidden*sizeof(float));
    for(std::size_t i=0;i<3;++i) {
      if(captures[i].size()!=std::size_t(frame.rows)*row_bytes)
        return std::unexpected(Status::InvalidArgument("DSpark captured target prefix row extent differs"));
      HYLE_ASSIGN_OR_RETURN(auto last,captures[i].subview(std::size_t(accepted-1)*row_bytes,row_bytes));
      HYLE_RETURN_IF_ERROR(draft.replace(entries[4+i],last));
    }
    facts.epoch=frame.epoch;facts.next_position+=accepted;facts.pending=pending;facts.consumed+=accepted;
    HYLE_RETURN_IF_ERROR(draft.replace_metadata(entries[7],std::as_bytes(std::span(&facts,1))));
    return draft.finalize();
  }
  // Unlike select()'s raw captured-row prefix, an actual speculative target
  // verification always consumes its mandatory current token. `consumed` is
  // therefore 1+accepted_drafts, and cannot select the pre-current base.
  // The three history rings are repaired only from actual target captures;
  // the parallel five proposal KV rows are temporary and never published.
  Result<c::StateVersion> select_verified(const c::StateVersion &from,
      std::span<const c::ReadLease> history,std::span<const c::ReadLease> captures,
      std::span<const c::ReadLease> target_state,const c::ReadLease &pending,
      const b::CaptureFrame &frame,std::uint32_t consumed) {
    HYLE_ASSIGN_OR_RETURN(auto facts,boundary(from));
    if(!pending_entry || target_entries.empty() || target_state.size()!=target_entries.size() ||
        history.size()!=4 || captures.size()!=3 || frame.target!=target ||
        facts.session!=frame.session || facts.branch!=frame.branch || facts.epoch>=frame.epoch ||
        facts.next_position!=frame.first_position || !consumed || consumed>6 ||
        frame.rows!=std::int64_t(consumed) ||
        frame.first_position<0 || frame.first_position>
            std::numeric_limits<std::int32_t>::max()-std::int64_t(consumed) ||
        consumed>std::numeric_limits<std::uint32_t>::max()-facts.consumed ||
        pending.size()!=sizeof(std::int32_t))
      return std::unexpected(Status::InvalidArgument("DSpark verified target tuple does not belong to this consumed prefix"));
    std::int32_t token=-1;
    HYLE_ASSIGN_OR_RETURN(auto pending_view,pending.view());
    HYLE_RETURN_IF_ERROR(pending_view.read(std::as_writable_bytes(std::span(&token,1))));
    if(token<0 || token>=d::vocabulary)
      return std::unexpected(Status::InvalidArgument("DSpark verified pending lease is outside the original vocabulary"));
    HYLE_ASSIGN_OR_RETURN(auto draft,store.begin(from));
    for(std::size_t i=0;i<4;++i)HYLE_RETURN_IF_ERROR(draft.replace(entries[i],history[i]));
    constexpr auto capture_bytes=std::size_t(4*b::kHidden*sizeof(float));
    for(std::size_t i=0;i<3;++i) {
      if(captures[i].size()!=capture_bytes)
        return std::unexpected(Status::InvalidArgument("DSpark verified last target capture has a different row extent"));
      HYLE_RETURN_IF_ERROR(draft.replace(entries[4+i],captures[i]));
    }
    for(std::size_t i=0;i<target_state.size();++i)
      HYLE_RETURN_IF_ERROR(draft.replace(target_entries[i],target_state[i]));
    HYLE_RETURN_IF_ERROR(draft.replace(*pending_entry,pending));
    facts.epoch=frame.epoch;facts.next_position+=consumed;facts.pending=token;facts.consumed+=consumed;
    HYLE_RETURN_IF_ERROR(draft.replace_metadata(entries[7],std::as_bytes(std::span(&facts,1))));
    return draft.finalize();
  }
};
} // namespace dspark_original_state_consumer
