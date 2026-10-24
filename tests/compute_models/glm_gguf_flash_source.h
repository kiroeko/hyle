#pragma once
#include "glm_gguf_flash_fragment.h"
#include "glm_gguf_runtime_source.h"
#include "glm_gguf_expert_frontier.h"

// Cold artifact composition only. The existing named reader authenticates the
// loose files and existing packed Supply owns two bounded generations. No
// model executor, provider routing, download or full expert bank lives here.
namespace glm_gguf_flash_source {
namespace named=original_named_role_source;
namespace data=glm_gguf_named_parameters;
namespace graph=glm_gguf_flash_fragment;
namespace resident=glm_gguf_runtime_source;
namespace c=hyle::compute;
struct Fixture final {
  named::Identity identity;
  std::span<const data::Spec> static_specs,expert_specs;
  std::span<const named::PinnedFile> static_pins;
  std::uint64_t static_bytes;
  std::string_view static_plan_sha256,expert_plan_sha256;
  c::ModelArtifactProfileDescription profile;
  graph::Policy policy;
};
inline void require(const Fixture &a) {
  if(a.static_specs.size()!=85||a.expert_specs.size()!=9||a.static_pins.size()!=85||
      !data::actual_pin(a.static_plan_sha256)||!data::actual_pin(a.expert_plan_sha256)||
      a.profile.artifact_repository!=a.identity.repository||a.profile.artifact_revision!=a.identity.revision||
      a.profile.variant!=a.identity.variant||a.profile.manifest_digest!=a.static_plan_sha256)
    throw std::runtime_error("converted Flash needs its own complete actual85-role pins/profile and nine bank range metadata");
  graph::require_geometry(a.policy.geometry);
}
inline std::shared_ptr<const named::Data> load(const std::filesystem::path &directory,const Fixture &a) {
  require(a);
  const auto manifest=data::manifest(a.static_specs,a.static_pins,a.static_bytes);
  return named::Data::load(directory,a.identity,manifest);
}
struct Source final : resident::ResidentSource {
  // Fixture has static storage supplied by an actual generated pins header.
  const Fixture *artifact=nullptr;
  named::Roles parameters;
  std::size_t stage=0,head_calls=0,pending_checks=0,projection_values=0;
  std::vector<std::size_t> union_sizes;
  void ensure(std::span<const std::int32_t> ids)const {
    if(experts.source->missing(ids).empty())return;
    if(!before_upload)throw std::runtime_error("converted Flash actual selected experts require explicit external supply");
    before_upload(ids);
    if(!experts.source->missing(ids).empty())throw std::runtime_error("converted Flash frontier did not attach every actual ID");
  }
  const named::Spec &spec(std::string_view name)const{return parameters.source->spec(name);}
  std::span<const std::byte> bytes(std::string_view name)const{return parameters.source->bytes(name);}
};
inline std::shared_ptr<Source> prepare(named::Roles bound,const Fixture &a,std::size_t stage,
    const c::StorageOwner &owner,const std::filesystem::path &expert_root,const std::filesystem::path &request,bool wait) {
  require(a);if(stage>=3||!bound.source)throw std::runtime_error("converted Flash stage/source missing");
  const auto &identity=bound.source->identity();
  if(identity.repository!=a.identity.repository||identity.revision!=a.identity.revision||identity.variant!=a.identity.variant)
    throw std::runtime_error("converted Flash Bound belongs to another original artifact");
  constexpr std::array sections{resident::Section::kTarget43,resident::Section::kTarget44,resident::Section::kNextn};
  auto result=std::make_shared<Source>();result->artifact=&a;result->parameters=std::move(bound);result->stage=stage;result->profile=&a.profile;
  result->experts.source=std::make_shared<resident::Experts>(resident::contract(sections[stage],std::string(a.expert_plan_sha256),identity,a.expert_specs,43+std::int64_t(stage)),a.policy.moe_geometry[stage]);
  result->cpu_supply=std::make_unique<resident::Supply>(named::detail::take(resident::Supply::create(owner,*result->experts.source),"actual converted Flash packed supply"));
  auto frontier=std::make_shared<glm_gguf_expert_frontier::Frontier>();frontier->selected=result->experts.source;frontier->root=expert_root;frontier->request=request;frontier->wait=wait;
  result->before_upload=[frontier](std::span<const std::int32_t> ids){frontier->ensure(ids);};return result;
}
} // namespace glm_gguf_flash_source
