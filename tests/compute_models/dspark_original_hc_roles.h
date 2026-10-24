#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dspark_target_binding.h"
#include "original_role_input.h"

// The exact34 original F32 roles from the formal Flash0731 Q8 DSpark sidecar.
// This is test-only data supply plus ordinary readonly parameter binding. It
// has no network, Graph executor, mutable cache, hidden precision conversion
// or original-router-bias injection. Runtime roles retain their host source
// because cold model reference callbacks borrow the original coefficients.
namespace dspark_original_hc_roles {
namespace b=dspark_target_binding;
namespace c=hyle::compute;
namespace m=hyle::mem;
inline constexpr std::string_view kRepository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr std::string_view kRevision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr std::string_view kFile="dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
inline constexpr std::size_t kStageRoles=10,kHeadRoles=4,kRoles=34,kOriginalBytes=9827740;
enum class StageRole:std::size_t {
  kAttentionNorm,kFfnNorm,kQueryNorm,kSinks,kAttentionFn,kAttentionScale,kAttentionBase,
  kFfnFn,kFfnScale,kFfnBase
};
enum class HeadRole:std::size_t {kOutputNorm,kFn,kScale,kBase};
struct Spec final {
  std::string_view name;
  std::array<std::int64_t,2> dimensions;
  std::size_t rank;
  std::string_view sha256;
};
inline constexpr std::array<Spec,kRoles> kSpecs{
    Spec{"blk.0.attn_norm.weight",{4096,0},1,"396a12e43a9f52abd1559782a8ff9e56ecc64be27d8ca5adfbb306a0c3bf8383"},
    Spec{"blk.0.ffn_norm.weight",{4096,0},1,"4a9ba6b7f09bc524eff7bdefccdd15ce017d01dde5c12a30deccc744345d5353"},
    Spec{"blk.0.attn_q_a_norm.weight",{1024,0},1,"7f0b93f9e600ee455e8d92e19eb3096f7d6515a2c7d9ea946fae445c3eb27d98"},
    Spec{"blk.0.attn_sinks.weight",{64,0},1,"dc60662d5641f55fe255aeca254e1c9833c45031a3b2298f5f25adb1e29bb60c"},
    Spec{"blk.0.hc_attn_fn.weight",{24,16384},2,"1e8c65edb07d24ea1bd64fc6ae5ca7a664f7a51c7b1f51176089c9099e747e1b"},
    Spec{"blk.0.hc_attn_scale.weight",{3,0},1,"0e74710a3eb16a4cb2bef752dec1eaf5ce8ecc8b87ae69a2f6d11bb8bacfe2c9"},
    Spec{"blk.0.hc_attn_base.weight",{24,0},1,"385b7df6b621441668980435b404bb88afe7fa0cfbb5633a8cf422ba3636b33a"},
    Spec{"blk.0.hc_ffn_fn.weight",{24,16384},2,"bcfbd07fef36881bd8545af66c10d178cd7bd413fbd9ca004d480ff37b52e71d"},
    Spec{"blk.0.hc_ffn_scale.weight",{3,0},1,"aa79a2ca028960d269a0157a53f154838ca479874563334dbef2850d6d87a3fc"},
    Spec{"blk.0.hc_ffn_base.weight",{24,0},1,"989b3ee507d35c56717d9b1a1326e519e05cbd227a4d7a8e095f2f83d6c97874"},
    Spec{"blk.1.attn_norm.weight",{4096,0},1,"48ebda5d30a703e9fb9f09356841b1215058bba1453f8a7b425f798f9f645b8f"},
    Spec{"blk.1.ffn_norm.weight",{4096,0},1,"086662abac6dff19d070ebc05df0c3031044d49a6987a170b14885fed1a17075"},
    Spec{"blk.1.attn_q_a_norm.weight",{1024,0},1,"73f77ae8a5126548d7ca1381b31367c391739a50e2cfcdb4ccd9f0450820e734"},
    Spec{"blk.1.attn_sinks.weight",{64,0},1,"45cfa7493ee060bad19775720b1cbff6fdcfa2e4badd10192a4948bae7faf56f"},
    Spec{"blk.1.hc_attn_fn.weight",{24,16384},2,"36db97bf157625436cde8ef5d664cc75a5200ad0703b1e45e3f1cb014c8ee4d9"},
    Spec{"blk.1.hc_attn_scale.weight",{3,0},1,"0d005a43144192ae70ab0f5c76f920a83e90d50e119fe1f5e44e4f80709ef539"},
    Spec{"blk.1.hc_attn_base.weight",{24,0},1,"f0f0a1bd42b1e477d1ca9350516cb122171c973bd9d909d593a7a051b5c14f7a"},
    Spec{"blk.1.hc_ffn_fn.weight",{24,16384},2,"5b3c567304f9d7c265eaae38269402eda8ceb33cdf26dea1937cfc7bc48b6531"},
    Spec{"blk.1.hc_ffn_scale.weight",{3,0},1,"7563f05bc705670f285c9b51742b695ab18456421a0f587dc3d17ffbdd12fb2b"},
    Spec{"blk.1.hc_ffn_base.weight",{24,0},1,"c60c0d4de5e5f2120f78c0e2b4d8f24c40e79eab67a2a97860f11a2b4d10b4dd"},
    Spec{"blk.2.attn_norm.weight",{4096,0},1,"d97f7b34ce8be7a6beb6250cc9625d217e2068d12da45fed856d2742cf0c50f3"},
    Spec{"blk.2.ffn_norm.weight",{4096,0},1,"fdfd11f38ab60126ad31e0a05b38ddfdfe36cd50aa74f72911573853ee8b31a7"},
    Spec{"blk.2.attn_q_a_norm.weight",{1024,0},1,"47b11cc973464af7f43922817a1ded49d6327a5efe3b16ffe6e97b0b811cec5e"},
    Spec{"blk.2.attn_sinks.weight",{64,0},1,"6969251b319b8ee73981eb3065741216f0b76a32428bdd3c90cfb84d07008bb8"},
    Spec{"blk.2.hc_attn_fn.weight",{24,16384},2,"23570d806e81e2e9686d68a79c118666cc0e8aec6b03b5260c4daa66454c7553"},
    Spec{"blk.2.hc_attn_scale.weight",{3,0},1,"b7c0ff61287e13c717a836032bc538b55afab27a2ff133c582412ffaef219f0b"},
    Spec{"blk.2.hc_attn_base.weight",{24,0},1,"fd1d75a55d767ec1dce0871a11b374886ac51d85efff280e0ca135d31ed654d2"},
    Spec{"blk.2.hc_ffn_fn.weight",{24,16384},2,"5924e62360949028b71892d86054d36c3c90ea78864711ebdc70b281db38ab3c"},
    Spec{"blk.2.hc_ffn_scale.weight",{3,0},1,"58d30d528c8f1a7a2e32eb9e0beea282e02b2417ebac6f44d5b09512a06487ab"},
    Spec{"blk.2.hc_ffn_base.weight",{24,0},1,"0dd7c6e80b26d6995747380d75b931b33eb0c9c57483599be1ca60ca22feb97c"},
    Spec{"output_norm.weight",{4096,0},1,"014657de94dfad9981299e368ae326c9c02948499f17f427ee9c4b2743d23b87"},
    Spec{"output_hc_fn.weight",{4,16384},2,"2f6395f102ce551931d54e46bc85ac54dff63f00ce4e90266fc70ccdb5795e9b"},
    Spec{"output_hc_scale.weight",{1,0},1,"47243371b69eabf3b1251a81710fbf577c470f7331d98bef369877defb16e0c5"},
    Spec{"output_hc_base.weight",{4,0},1,"feae4c4530df67ba8b01555ad528ad0b5452463828ebb6dbd44b99fae81048fb"}
};
inline std::size_t index(std::size_t stage,StageRole role) {
  const auto parameter=static_cast<std::size_t>(role);
  if(stage>=3 || parameter>=kStageRoles)throw std::runtime_error("original DSpark stage parameter is absent");
  return stage*kStageRoles+parameter;
}
inline std::size_t index(HeadRole role) {
  const auto parameter=static_cast<std::size_t>(role);
  if(parameter>=kHeadRoles)throw std::runtime_error("original DSpark output parameter is absent");
  return 3*kStageRoles+parameter;
}
inline const Spec &spec(std::size_t parameter) {
  if(parameter>=kSpecs.size())throw std::runtime_error("original DSpark parameter index is absent");
  return kSpecs[parameter];
}
inline c::TensorDesc description(std::size_t parameter) {
  const auto &s=spec(parameter);
  auto result=c::TensorDesc::dense(c::DataType::kFloat32,std::span(s.dimensions).first(s.rank));
  if(!result)throw std::runtime_error(std::string(result.error().message()));
  return *result;
}
struct Data final {
  std::array<std::vector<float>,kRoles> parameters;
  static Data load(const std::filesystem::path &directory) {
    Data result;
    std::size_t bytes=0;
    for(std::size_t i=0;i<kRoles;++i) {
      const auto &s=spec(i);const auto extent=description(i).required_bytes();
      if(!extent)throw std::runtime_error(std::string(extent.error().message()));
      result.parameters[i]=original_role_input::read_float(directory/(std::string(s.name)+".f32"),*extent/sizeof(float),s.sha256);
      bytes+=*extent;
    }
    if(bytes!=kOriginalBytes)throw std::runtime_error("original DSpark parameter total differs");
    return result;
  }
  std::span<const float> values(std::size_t parameter) const {return parameters.at(parameter);}
  std::span<const float> values(std::size_t stage,StageRole role) const {return values(index(stage,role));}
  std::span<const float> values(HeadRole role) const {return values(index(role));}
};
struct Roles final {
  std::shared_ptr<const Data> source;
  std::array<b::ReadonlyRole,kRoles> parameters;
  const b::ReadonlyRole &at(std::size_t parameter) const {return parameters.at(parameter);}
};
inline Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source)throw std::runtime_error("original DSpark coefficient source is absent");
  Roles result{.source=std::move(source)};
  for(std::size_t i=0;i<kRoles;++i) {
    const auto raw=std::as_bytes(result.source->values(i));
    const auto desc=description(i);
    const auto extent=desc.required_bytes();
    if(!extent || raw.size()!=*extent)throw std::runtime_error("original DSpark coefficient extent differs");
    auto allocated=m::Buffer::create(owner.ref(),raw.size(),256);
    if(!allocated)throw std::runtime_error(std::string(allocated.error().message()));
    auto buffer=std::move(*allocated);const auto written=buffer.view().write(raw);
    if(!written.ok())throw std::runtime_error(std::string(written.message()));
    auto lease=owner.adopt_buffer(std::move(buffer));
    if(!lease)throw std::runtime_error(std::string(lease.error().message()));
    result.parameters[i]={std::string(spec(i).name),desc,owner,std::move(*lease)};
  }
  return result;
}
inline hyle::Status qualify(const Roles &roles,const b::TargetIdentity &target) {
  if(!roles.source || target.model_repository!=b::kModelRepository || target.model_revision!=b::kModelRevision)
    return hyle::Status::InvalidArgument("original DSpark parameters require matching formal Flash0731");
  for(std::size_t i=0;i<kRoles;++i) {
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.at(i),spec(i).name,description(i)));
  }
  return hyle::Status::Ok();
}
} // namespace dspark_original_hc_roles
