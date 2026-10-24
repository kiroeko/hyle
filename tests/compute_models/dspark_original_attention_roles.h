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

#include "dspark_target_binding.h"
#include "original_role_input.h"

// Exact twelve original Q8_0 attention banks, supplied once as ordinary
// readonly Graph operands. The output-a role's original8192 contiguous rows
// naturally describe8groups of1024 rows; no cross-group dense product or
// weight conversion is needed. This helper owns no executor or model cache.
namespace dspark_original_attention_roles {
namespace b=dspark_target_binding;
namespace c=hyle::compute;
namespace m=hyle::mem;
inline constexpr std::string_view kRepository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr std::string_view kRevision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr std::string_view kFile="dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
inline constexpr std::size_t kStages=3,kStageRoles=4,kRoles=12,kOriginalBytes=334233600;
enum class StageRole:std::size_t {kQueryA,kQueryB,kOutputA,kOutputB};
struct Spec final {
  std::string_view name;
  std::int64_t groups,outputs,reduction;
  std::string_view sha256;
};
inline constexpr std::array<Spec,kRoles> kSpecs{
  Spec{"blk.0.attn_q_a.weight",1,1024,4096,"23070c5c754035345e13a6ab83d328c42d0aa5c29db8df4f2520752274b2f696"},
  Spec{"blk.0.attn_q_b.weight",1,32768,1024,"a7a59a2cdb98e6dac7366ace3b7bb55cf3bf65b0bba8580236e0877e283df58c"},
  Spec{"blk.0.attn_output_a.weight",8,1024,4096,"6183e80a9224432ca9660183c6cb032d5b8a29d3bfb22ecb7017fd7ab34e8431"},
  Spec{"blk.0.attn_output_b.weight",1,4096,8192,"03b72cf551e074276174466d79b03aa0520af9510bbcff015ba0b3a5afdcb905"},
  Spec{"blk.1.attn_q_a.weight",1,1024,4096,"397fda2e7a484aedad8212fad93f267f4e8fb8673b485b64bf48a94ec54842ce"},
  Spec{"blk.1.attn_q_b.weight",1,32768,1024,"9826d5cafc695083bc4b2cd21fde04236a956de8fb74e0dbacda76f242f4e273"},
  Spec{"blk.1.attn_output_a.weight",8,1024,4096,"f05f92eabfcd5ec29a4772e858cdd375bbc0b5d0748d6d91836a1df7be14bda5"},
  Spec{"blk.1.attn_output_b.weight",1,4096,8192,"17a800e0281ee4b6f240cc02090bde492149dc022ae64a86193faf360dd99c13"},
  Spec{"blk.2.attn_q_a.weight",1,1024,4096,"8075cd34904ab843caf4d7988bcad60bf0388bbf6b0a8c2a5d4fde4e0f7ba8e4"},
  Spec{"blk.2.attn_q_b.weight",1,32768,1024,"24358aa3b4e598a19136bf167db570bc7040b10459be0d0492bed5b82786fa7f"},
  Spec{"blk.2.attn_output_a.weight",8,1024,4096,"6e31c8ea1ad49734e502f5b80d556c67842dc140a6948b64d4a22987e5f80707"},
  Spec{"blk.2.attn_output_b.weight",1,4096,8192,"e87988e7a0e5986396dbba0cdceeba7fe8ff171efde949dfb1d23cad95381dbd"}
};
inline std::size_t index(std::size_t stage,StageRole role) {
  const auto parameter=static_cast<std::size_t>(role);
  if(stage>=kStages||parameter>=kStageRoles)throw std::runtime_error("original attention role is absent");
  return stage*kStageRoles+parameter;
}
inline const Spec &spec(std::size_t parameter) {
  if(parameter>=kRoles)throw std::runtime_error("original attention parameter index is absent");
  return kSpecs[parameter];
}
inline c::TensorDesc description(std::size_t parameter) {
  const auto &s=spec(parameter);
  auto result=s.groups==1?c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32,s.reduction,s.outputs):c::gguf_grouped_weight_description(
      c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,s.groups,s.outputs,s.reduction);
  if(!result)throw std::runtime_error(std::string(result.error().message()));
  return *result;
}
struct Data final {
  std::array<original_role_input::Bytes,kRoles> parameters;
  static Data load(const std::filesystem::path &directory) {
    Data result;std::size_t total_bytes=0;
    for(std::size_t i=0;i<kRoles;++i) {
      const auto &s=spec(i);const auto extent=description(i).required_bytes();
      if(!extent)throw std::runtime_error(std::string(extent.error().message()));
      auto &payload=result.parameters[i];
      payload=original_role_input::read_bytes(directory/(std::string(s.name)+".q8_0"),*extent,s.sha256);
      if(payload.size()%34)throw std::runtime_error("original attention Q8_0 block extent");
      for(std::size_t offset=0;offset<payload.size();offset+=34) {
        const auto scale=std::uint16_t(std::to_integer<unsigned>(payload[offset])|
            (std::to_integer<unsigned>(payload[offset+1])<<8u));
        if((scale&0x7c00u)==0x7c00u)throw std::runtime_error("original attention Q8_0 scale is nonfinite");
      }
      total_bytes+=*extent;
    }
    if(total_bytes!=kOriginalBytes)throw std::runtime_error("original attention byte total differs");
    return result;
  }
  std::span<const std::byte> bytes(std::size_t parameter) const {return parameters.at(parameter);}
  std::span<const std::byte> bytes(std::size_t stage,StageRole role) const {return bytes(index(stage,role));}
};
struct Roles final {
  std::shared_ptr<const Data> source;
  std::array<b::ReadonlyRole,kRoles> parameters;
  const b::ReadonlyRole &at(std::size_t parameter) const {return parameters.at(parameter);}
};
inline Roles bind(std::shared_ptr<const Data> source,const c::StorageOwner &owner) {
  if(!source)throw std::runtime_error("original attention source is absent");
  Roles result{.source=std::move(source)};
  for(std::size_t i=0;i<kRoles;++i) {
    const auto raw=result.source->bytes(i);const auto desc=description(i);const auto extent=desc.required_bytes();
    if(!extent||raw.size()!=*extent)throw std::runtime_error("original attention parameter extent differs");
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
  if(!roles.source||target.model_repository!=b::kModelRepository||target.model_revision!=b::kModelRevision)
    return hyle::Status::InvalidArgument("original attention needs the formal Flash0731 identity");
  for(std::size_t i=0;i<kRoles;++i) {
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.at(i),spec(i).name,description(i)));
  }
  return hyle::Status::Ok();
}
} // namespace dspark_original_attention_roles
