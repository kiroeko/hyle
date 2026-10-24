#pragma once

#include "glm_original_artifact_source.h"
#include "glm_original_target43_roles.h"
#include "glm_original_target44_roles.h"

// Cold original-name binding only. This shares the existing43/44 suppliers;
// no alternate Data, Graph executor, decoded weight bank or default identity.
namespace glm_original_target_source {
namespace source=glm_original_artifact_source;
namespace moe=glm_original_moe_roles;
namespace target43=glm_original_target43_roles;
namespace target44=glm_original_target44_roles;
namespace c=hyle::compute;
using Manifest=source::Manifest;
using FilePin=moe::FilePin;
struct Input43 final {target43::Pins parameters;std::array<FilePin,8> moe;};
struct Input44 final {target44::Pins parameters;std::array<FilePin,8> moe;};

inline std::array<FilePin,8> mxfp4_moe(const Manifest &manifest) {
  std::array<FilePin,8> result;
  result[0]=source::detail::file(manifest,manifest.identity.layer_prefix+"mlp.gate.weight",moe::router_description());
  result[1]=source::detail::file(manifest,manifest.identity.layer_prefix+"mlp.gate.e_score_correction_bias",moe::bias_description());
  for(std::size_t leg=0;leg<3;++leg)for(std::size_t plane=0;plane<2;++plane) {
    const auto kind=static_cast<moe::Leg>(leg);const bool scales=plane!=0;
    const auto [n,k]=moe::geometry(kind);
    result[2+2*leg+plane]=source::detail::file(manifest,
        moe::source_name(manifest.identity,kind,scales),
        source::detail::dense({n,k/(scales?32:2)},c::DataType::kUInt8));
  }
  return result;
}
inline void require(const Manifest &manifest,const moe::SourceIdentity &nextn,
    std::string_view prefix,std::size_t count) {
  moe::require_identity(manifest.identity);manifest.arithmetic.require();
  if(manifest.identity.repository!=nextn.repository||manifest.identity.revision!=nextn.revision||
      manifest.identity.scale_member!=nextn.scale_member||manifest.identity.layer_prefix!=prefix||
      manifest.files.size()!=count)
    throw std::runtime_error("original target needs the same actual artifact as NextN/vocabulary and its own layer pins");
  std::unordered_set<std::string_view> names,files;
  for(const auto &pin:manifest.files) {
    const auto path=std::filesystem::path(std::string(pin.file));
    if(!names.insert(pin.name).second||!files.insert(pin.file).second||
        path.empty()||path.filename()!=path||pin.sha256.size()!=64||
        !std::ranges::all_of(pin.sha256,[](char ch){return (ch>='0'&&ch<='9')||(ch>='a'&&ch<='f');}))
      throw std::runtime_error("original target requires unique typed names/files and every real pin");
  }
}
inline Input43 input43(const Manifest &manifest,const moe::SourceIdentity &nextn) {
  require(manifest,nextn,"model.language_model.layers.43.",34);
  Input43 result;
  for(std::size_t i=0;i<target43::kRawRoles;++i) {
    const auto role=static_cast<target43::RawRole>(i);const auto &spec=target43::kSpecs[i];
    std::string name(spec.name);
    constexpr std::string_view old="weight_scale_inv";
    if(name.ends_with(old))name.replace(name.size()-old.size(),old.size(),manifest.identity.scale_member);
    result.parameters.own[i]=source::detail::file(manifest,name,target43::raw_description(role),spec.file).sha256;
  }
  result.parameters.attention=source::detail::attention_pins(manifest);
  result.parameters.indexer=source::detail::indexer_pins(manifest);
  result.moe=mxfp4_moe(manifest);
  return result;
}
inline Input44 input44(const Manifest &manifest,const moe::SourceIdentity &nextn) {
  require(manifest,nextn,"model.language_model.layers.44.",32);
  Input44 result;
  for(std::size_t i=0;i<target44::kRoles;++i) {
    const auto role=static_cast<target44::Role>(i);const auto &spec=target44::kSpecs[i];
    // model.language_model.norm is an independent original target role;
    // shared_head.norm from NextN is deliberately not accepted here.
    result.parameters[i]=source::detail::file(manifest,spec.name,target44::raw_description(role),spec.file).sha256;
  }
  result.moe=mxfp4_moe(manifest);
  return result;
}
} // namespace glm_original_target_source
