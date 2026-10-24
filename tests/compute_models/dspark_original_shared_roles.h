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

// Test-only supply of the exact nine shared FFN Q8_0 banks. They are ordinary
// immutable MatMul operands, with no decoder/loader service or executor. The
// physical original[N,K] rows are described as logical MatMul RHS[K,N].
namespace dspark_original_shared_roles {
namespace b=dspark_target_binding;
namespace c=hyle::compute;
namespace m=hyle::mem;
inline constexpr std::string_view kRepository="unsloth/DeepSeek-V4-Flash-0731-GGUF";
inline constexpr std::string_view kRevision="fbbb5b93fb787c21338159b0af3318bb3f4d9768";
inline constexpr std::string_view kFile="dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
inline constexpr std::size_t kStages=3,kStageRoles=3,kRoles=9,kOriginalBytes=80216064;
enum class StageRole:std::size_t {kGate,kUp,kDown};
struct Spec final {std::string_view name;std::int64_t outputs,reduction;std::string_view sha256;};
inline constexpr std::array<Spec,kRoles> kSpecs{
  Spec{"blk.0.ffn_gate_shexp.weight",2048,4096,"4dea85a5c73f1f3d8556436306b386e6db1656cf41482143f027b632f5a864ad"},
  Spec{"blk.0.ffn_up_shexp.weight",2048,4096,"dde85012f13ca2c96d7e877c1c6d141db0b2d2cf91ba087a45458cc1a4dec6b8"},
  Spec{"blk.0.ffn_down_shexp.weight",4096,2048,"a2fedccd9c8c75d4b953cb3af6368a52683372293fbf23909bb451aecaed4939"},
  Spec{"blk.1.ffn_gate_shexp.weight",2048,4096,"bd674477c52462120a2092e0cadbdc3470f358221b76ee90f874dafcca1b9309"},
  Spec{"blk.1.ffn_up_shexp.weight",2048,4096,"26a1a4e1040d82892b0abdb9cac2e80bfb2a476b3717dbd8765c0b589472675e"},
  Spec{"blk.1.ffn_down_shexp.weight",4096,2048,"68d3bb6b03fad6d0cff461031334365e1fd6476764f495eda20b74eccd9777a0"},
  Spec{"blk.2.ffn_gate_shexp.weight",2048,4096,"26d4547520661f6e16c13dd5f166ee493ce6a92d4417d6bbade73a94298a2b22"},
  Spec{"blk.2.ffn_up_shexp.weight",2048,4096,"cae64bbaf6a701f3b1c49582527710a018e44513a907ec762f033e88fdb2f82a"},
  Spec{"blk.2.ffn_down_shexp.weight",4096,2048,"cdd77201bdb49a8eb559cd3fc4c032a835d4de1e7b7f7682fa44b9830607de37"}
};
inline std::size_t index(std::size_t stage,StageRole role) {
  const auto parameter=static_cast<std::size_t>(role);
  if(stage>=kStages||parameter>=kStageRoles)throw std::runtime_error("original DSpark shared FFN role is absent");
  return stage*kStageRoles+parameter;
}
inline const Spec &spec(std::size_t parameter) {
  if(parameter>=kRoles)throw std::runtime_error("original DSpark shared FFN index is absent");
  return kSpecs[parameter];
}
inline c::TensorDesc description(std::size_t parameter) {
  const auto &s=spec(parameter);
  auto result=c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,s.reduction,s.outputs);
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
      // Each original Q8_0 block is one little-endian F16 scale followed by
      //32 signed bytes. Authentication and this finite-scale precondition do
      //not transform the bytes or assume a particular activation quantizer.
      if(payload.size()%34)throw std::runtime_error("original shared Q8_0 block extent");
      for(std::size_t offset=0;offset<payload.size();offset+=34) {
        const auto scale=std::uint16_t(std::to_integer<unsigned>(payload[offset])|
            (std::to_integer<unsigned>(payload[offset+1])<<8u));
        if((scale&0x7c00u)==0x7c00u)throw std::runtime_error("original shared Q8_0 scale is nonfinite");
      }
      total_bytes+=*extent;
    }
    if(total_bytes!=kOriginalBytes)throw std::runtime_error("original shared FFN byte total differs");
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
  if(!source)throw std::runtime_error("original shared FFN source is absent");
  Roles result{.source=std::move(source)};
  for(std::size_t i=0;i<kRoles;++i) {
    const auto raw=result.source->bytes(i);const auto desc=description(i);const auto extent=desc.required_bytes();
    if(!extent||raw.size()!=*extent)throw std::runtime_error("original shared FFN parameter extent differs");
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
    return hyle::Status::InvalidArgument("original shared FFN requires the formal Flash0731 identity");
  for(std::size_t i=0;i<kRoles;++i) {
    HYLE_RETURN_IF_ERROR(b::qualify_role(roles.at(i),spec(i).name,description(i)));
  }
  return hyle::Status::Ok();
}
} // namespace dspark_original_shared_roles
