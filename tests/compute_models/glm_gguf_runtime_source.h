#pragma once

#include "glm_gguf_named_parameters.h"
#include "glm_original_moe_roles.h"
#include "glm_original_moe_resident_weights.h"
#include "glm_shared_vocabulary.h"
#include "model_packed_expert_layout.h"
#include "original_named_role_source.h"
#include "dspark_original_role_source.h"
#include <unordered_map>
#include <functional>
#include <optional>
#include "hyle/compute/workload.h"

namespace glm_gguf_runtime_source {
namespace c=hyle::compute;
namespace m=hyle::mem;
namespace named=original_named_role_source;
namespace data=glm_gguf_named_parameters;
namespace expert=glm_original_moe_roles;
using Leg=expert::Leg;
using Readonly=glm_shared_vocabulary::ReadonlyRole;
inline Readonly readonly(const named::ReadonlyRole &r) {return {r.name,r.description,r.owner,r.lease};}

enum class Section {kDraft,kTarget,kNextn,kTarget43,kTarget44};
inline std::string_view label(Section s) {
  switch(s){case Section::kDraft:return "draft";case Section::kTarget:return "target";
    case Section::kNextn:return "nextn";case Section::kTarget43:return "target43";case Section::kTarget44:return "target44";}
  throw std::runtime_error("unknown converted GLM selected expert section");
}
struct Bank final {
  std::string name;ggml_type native;
  c::TensorDesc description; // Logical full canonical metadata; no full W.
};
struct Contract final {
  Section section;
  std::string plan_sha256,repository,revision,variant;
  std::array<Bank,3> banks;
};
inline c::TensorDesc expert_bank_description(const Bank &bank) {
  const auto &d=bank.description;const auto codec=c::gguf_quantized_type(d.storage_format());
  if(!codec||d.rank()!=3||d.data_type()!=c::DataType::kFloat32||
      named::native_codec(*codec)!=bank.native||std::ranges::any_of(d.dimensions(),[](auto n){return n<=0;}))
    throw std::runtime_error("converted expert bank native/canonical descriptor differs");
  const auto canonical=c::gguf_grouped_weight_description(*codec,c::DataType::kFloat32,
      d.dimensions()[0],d.dimensions()[1],d.dimensions()[2]);
  if(!canonical||*canonical!=d||!d.required_bytes())
    throw std::runtime_error("converted original expert bank layout differs");return d;
}
inline Contract contract(Section section,std::string plan_sha256,const named::Identity &identity,
                         std::span<const data::Spec> actual_banks,std::int64_t layer) {
  (void)label(section);
  if(!data::actual_pin(plan_sha256)||identity.repository.empty()||identity.variant.empty()||identity.file.empty()||
      !named::pinned(identity.revision,40)||layer<0)
    throw std::runtime_error("converted selected source needs explicit authenticated identity/layer/range-plan");
  Contract r{section,std::move(plan_sha256),std::string(identity.repository),std::string(identity.revision),std::string(identity.variant),{}};
  constexpr std::array<std::string_view,3> suffix{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
  for(std::size_t leg=0;leg<3;++leg) {
    const auto name="blk."+std::to_string(layer)+"."+std::string(suffix[leg]);
    const data::Spec *found=nullptr;
    for(const auto &spec:actual_banks)if(spec.name==name) {
      if(found)throw std::runtime_error("duplicate converted original expert bank metadata");found=&spec;
    }
    if(!found||found->rank!=3)throw std::runtime_error("converted original expert bank metadata is absent");
    const auto d=data::description(*found);
    if(!d.required_bytes()||*d.required_bytes()!=found->bytes)
      throw std::runtime_error("converted original expert bank byte extent differs");
    r.banks[leg]={name,static_cast<ggml_type>(found->native),d};
    (void)expert_bank_description(r.banks[leg]);
  }
  const auto domain=r.banks[0].description.dimensions()[0];
  if(domain>std::numeric_limits<std::int32_t>::max()||r.banks[1].description.dimensions()[0]!=domain||
      r.banks[2].description.dimensions()[0]!=domain)
    throw std::runtime_error("converted expert legs disagree about their original domain");return r;
}
class ExpertSet final {
  Contract contract_;
  std::int32_t global_;
  dspark_original_role_source::Bundle body_;
  ExpertSet(Contract c,std::int32_t global,dspark_original_role_source::Bundle body):contract_(std::move(c)),global_(global),body_(std::move(body)){}
 public:
  ExpertSet(const ExpertSet&)=delete;ExpertSet& operator=(const ExpertSet&)=delete;
  static std::shared_ptr<const ExpertSet> load(const std::filesystem::path &directory,std::string_view actual_catalog_sha,
      const Contract &c,std::int32_t global) {
    if(!data::actual_pin(actual_catalog_sha)||!data::actual_pin(c.plan_sha256)||global<0||global>=expert_bank_description(c.banks[0]).dimensions()[0])
      throw std::runtime_error("B09 per-ID actual catalogue pin/source/global is invalid");
    std::array<dspark_original_role_source::Spec,3> specs;
    std::size_t total=0;
    for(std::size_t leg=0;leg<3;++leg) {
      const auto d=expert_bank_description(c.banks[leg]);const auto stride=d.byte_strides()[0];
      if(stride<=0||std::uint64_t(stride)>std::uint64_t(std::numeric_limits<std::size_t>::max()-total))
        throw std::runtime_error("B09 selected expert extent overflow");
      specs[leg]={c.banks[leg].name,c::DataType::kUInt8,stride};total+=std::size_t(stride);
    }
    // Exactly three original native packed slices. Reuse the authenticated
    // HYDSPT01 reader (catalog112B); never mmap or pin a complete expert bank.
    auto body=dspark_original_role_source::Bundle::load(directory,actual_catalog_sha,specs,total);
    return std::shared_ptr<const ExpertSet>(new ExpertSet(c,global,std::move(body)));
  }
  const Contract &source_contract()const noexcept{return contract_;}
  std::int32_t global()const noexcept{return global_;}
  std::span<const std::byte> bytes(Leg leg)const{return body_.raw(contract_.banks[std::size_t(leg)].name);}
};
// Explicit attachment happens only at the external artifact frontier. Lookup
// has no I/O/download and cannot substitute a different layer's same globalID.
class Experts final {
  Contract contract_;
  std::unordered_map<std::int32_t,std::shared_ptr<const ExpertSet>> selected_;
 public:
  glm_moe_geometry::Geometry geometry;
  Experts(Contract c,const glm_moe_geometry::Geometry &g):contract_(std::move(c)),geometry(g){
    glm_moe_geometry::require(g);
    for(std::size_t leg=0;leg<3;++leg) {
      const auto d=expert_bank_description(contract_.banks[leg]);
      if(d.dimensions()[0]!=g.experts||d.dimensions()[1]!=(leg==2?g.hidden:g.intermediate)||
          d.dimensions()[2]!=(leg==2?g.intermediate:g.hidden))
        throw std::runtime_error("converted source geometry differs from the three actual expert descriptors");
    }
  }
  const auto &model()const noexcept{return geometry;}
  const Contract &source_contract()const noexcept{return contract_;}
  void attach(std::shared_ptr<const ExpertSet> set) {
    if(!set||set->source_contract().section!=contract_.section||set->source_contract().plan_sha256!=contract_.plan_sha256||
        set->source_contract().repository!=contract_.repository||set->source_contract().revision!=contract_.revision||
        set->source_contract().variant!=contract_.variant)
      throw std::runtime_error("B09 per-ID source identity/section does not match this frontier");
    for(std::size_t leg=0;leg<3;++leg)if(set->source_contract().banks[leg].name!=contract_.banks[leg].name||
        expert_bank_description(set->source_contract().banks[leg])!=expert_bank_description(contract_.banks[leg]))
      throw std::runtime_error("B09 attached original expert physical metadata differs");
    if(!selected_.emplace(set->global(),std::move(set)).second)throw std::runtime_error("B09 original per-ID set cannot be rebound");
  }
  std::vector<std::int32_t> missing(std::span<const std::int32_t> actual_ids)const {
    std::vector<std::int32_t> result;
    for(const auto id:actual_ids) {
      if(id<0||id>=geometry.experts)throw std::runtime_error("B09 actual global expert outside original domain");
      if(!selected_.contains(id))result.push_back(id);
    }
    std::ranges::sort(result);result.erase(std::unique(result.begin(),result.end()),result.end());return result;
  }
  c::TensorDesc expert_description(Leg leg)const{return expert_bank_description(contract_.banks[std::size_t(leg)]);}
  ggml_type native(Leg leg)const{return static_cast<ggml_type>(contract_.banks[std::size_t(leg)].native);}
  std::span<const std::byte> expert_bytes(Leg leg,std::int32_t global)const {
    const auto found=selected_.find(global);if(found==selected_.end())throw std::runtime_error("B09 actual selected expert has not been explicitly attached");
    return found->second->bytes(leg);
  }
};
using Layout=model_packed_expert_layout::Layout;
struct Batch final {c::StorageOwner owner;std::array<c::ReadLease,4> banks;};
// Same ordinary input lease path on CPU/device: bounded two-generation native
// storage with original packed bytes. Buffer::write performs the actual copy
// into its declared owner; no full-domain upload, F32 W expansion or executor.
class Supply final {
  c::StorageOwner owner_;Layout layout_;c::LeasePool pool_;
 public:
  static hyle::Result<Supply> create(const c::StorageOwner &owner,const Experts &source) {
    if(!owner)return std::unexpected(hyle::Status::InvalidArgument("missing converted GLM resident owner"));
    Supply r;r.owner_=owner;HYLE_ASSIGN_OR_RETURN(r.layout_,Layout::create({source.expert_description(Leg::kGate),source.expert_description(Leg::kUp),source.expert_description(Leg::kDown)},source.geometry.slots));
    if(r.layout_.slot_bytes()>std::numeric_limits<std::size_t>::max()/2)
      return std::unexpected(hyle::Status::InvalidArgument("converted GLM resident capacity overflow"));
    HYLE_ASSIGN_OR_RETURN(r.pool_,c::LeasePool::create(owner.ref(),2*r.layout_.slot_bytes(),
        {.max_blocks=8,.backing_alignment=256}));return r;
  }
  hyle::Result<Batch> upload(const expert::Selection &selection,const Experts &source) {
    HYLE_RETURN_IF_ERROR(glm_original_moe_resident_weights::validate(selection,source.geometry));
    HYLE_RETURN_IF_ERROR(layout_.validate_descriptions({source.expert_description(Leg::kGate),source.expert_description(Leg::kUp),source.expert_description(Leg::kDown)}));
    Batch r{owner_,{}};std::array<c::WriteLease,4> writers;
    for(std::size_t leg=0;leg<3;++leg) {
      HYLE_ASSIGN_OR_RETURN(writers[leg],pool_.allocate(layout_.bank_bytes(leg)));
      HYLE_ASSIGN_OR_RETURN(auto out,writers[leg].view());
      for(std::size_t slot=0;slot<source.geometry.slots;++slot) {
        const auto global=slot<selection.globals.size()?selection.globals[slot]:selection.globals.front();
        const auto bytes=source.expert_bytes(Leg(leg),global);
        HYLE_ASSIGN_OR_RETURN(auto row,out.subview(slot*layout_.expert_bytes(leg),layout_.expert_bytes(leg)));
        HYLE_RETURN_IF_ERROR(row.write(bytes));
      }
    }
    HYLE_ASSIGN_OR_RETURN(writers[3],pool_.allocate(layout_.map_bytes()));
    HYLE_ASSIGN_OR_RETURN(auto map,writers[3].view());HYLE_RETURN_IF_ERROR(map.write(std::as_bytes(std::span(selection.map))));
    for(std::size_t i=0;i<writers.size();++i){HYLE_ASSIGN_OR_RETURN(r.banks[i],writers[i].publish());}
    return r;
  }
  const Layout &layout()const noexcept{return layout_;}
  c::LeasePoolStats stats()const noexcept{return pool_.stats();}
};
struct ResidentSource {
  static constexpr bool kPacked=true;
  using CpuBatch=Batch;
  const c::ModelArtifactProfileDescription *profile=nullptr; // Cold actual artifact, never selected in execution.
  struct ExpertOwner {std::shared_ptr<Experts> source;} experts;
  std::unique_ptr<Supply> cpu_supply;
  std::function<void(std::span<const std::int32_t>)> before_upload;
  bool moe_full_checked=false,full_checked=false,cuda=false;
  std::size_t episodes=0,cold_submissions=0,checked_steps=0;
  std::vector<expert::Selection> selections;
};
struct Words final {
  std::shared_ptr<const named::Data> source;
  std::array<Readonly,3> roles;
  const Readonly &at(glm_shared_vocabulary::Role role)const{return roles[std::size_t(role)];}
};
struct Source final:ResidentSource {
  named::Roles parameters;Words words;std::array<Readonly,2> prepared;
  std::optional<Contract> target_contract;
  std::filesystem::path expert_directory,frontier_file;
  bool wait_for_experts=false;
};
struct TargetSource final:ResidentSource {named::Roles parameters;};
} // namespace glm_gguf_runtime_source
