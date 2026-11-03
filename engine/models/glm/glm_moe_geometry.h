#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

// Immutable-by-use, model-local MoE math/storage geometry. Artifact identity
// and its authenticated catalogue stay in the original parameter supplier.
namespace glm_moe_geometry {
struct Geometry final {
  std::int64_t hidden,intermediate;
  std::int32_t experts;
  std::size_t slots;
  float route_scale;
  std::optional<float> swiglu_limit;
  friend constexpr bool operator==(const Geometry&,const Geometry&)=default;
  constexpr bool valid() const noexcept {
    constexpr auto max=std::numeric_limits<std::size_t>::max();
    if(hidden<=0||intermediate<=0||hidden%128||intermediate%128||experts<=0||
        !slots||slots>std::size_t(experts)||!(route_scale>0)||
        route_scale>std::numeric_limits<float>::max()||
        (swiglu_limit&&(!(*swiglu_limit>0)||*swiglu_limit>std::numeric_limits<float>::max())))return false;
    if(std::uint64_t(hidden)>max||std::uint64_t(intermediate)>max)return false;
    const auto h=std::size_t(hidden),i=std::size_t(intermediate),e=std::size_t(experts);
    if(h>max/i||e>(std::numeric_limits<std::uint32_t>::max()-8u)/6u||
        slots>(std::numeric_limits<std::uint32_t>::max()-1u)/6u)return false;
    const auto codes=h*i,scales=codes/4096;
    if(codes>max-scales)return false;
    const auto matrix=codes+scales;
    if(matrix>max/3||e+1>max/(3*matrix)||h>max/2||e>max/(2*h))return false;
    const auto router=e*h*2;
    if(e>(max-router)/4||(e+1)*3*matrix>max-router-e*4)return false;
    if(slots>max/matrix||slots*matrix>max-255||e>max/4)return false;
    const auto bank=(slots*matrix+255)/256*256;
    const auto map=(e*4+255)/256*256;
    return map<=max/2&&bank<=(max/2-map)/3;
  }
  constexpr std::size_t code_bytes() const noexcept {return std::size_t(hidden)*std::size_t(intermediate);}
  constexpr std::size_t scale_bytes() const noexcept {return code_bytes()/4096;}
  constexpr std::size_t raw_roles() const noexcept {return 8+6*std::size_t(experts);}
  constexpr std::size_t original_bytes() const noexcept {
    return std::size_t(experts)*std::size_t(hidden)*2+std::size_t(experts)*4+
        (std::size_t(experts)+1)*3*(code_bytes()+scale_bytes());
  }
  constexpr std::size_t index_bytes() const noexcept {return 64+16*raw_roles();}
};
inline constexpr Geometry kFlash{4096,2048,288,8,2.5F,10.0F};
inline constexpr Geometry kPro{6144,2048,256,8,2.5F,std::nullopt};
inline constexpr void require(const Geometry &model) {
  if(!model.valid())throw std::runtime_error("invalid or overflowing original GLM MoE geometry");
}
struct Layout final {
  std::size_t code_bytes,scale_bytes,bank_bytes,map_bytes,logical_bytes,slot_bytes,fragments;
};
inline constexpr Layout layout(const Geometry &model) {
  require(model);
  const auto codes=model.code_bytes(),scales=model.scale_bytes();
  const auto bank=model.slots*(codes+scales),map=std::size_t(model.experts)*4;
  return {codes,scales,bank,map,3*bank+map,
      3*((bank+255)/256*256)+(map+255)/256*256,6*model.slots+1};
}
static_assert(kFlash.valid()&&kFlash.original_bytes()==7277059200ULL&&kFlash.raw_roles()==1736);
static_assert(kPro.valid()&&kPro.original_bytes()==9706940416ULL&&kPro.raw_roles()==1544);
static_assert(layout(kFlash).slot_bytes==201377024&&layout(kPro).slot_bytes==302064640);
} // namespace glm_moe_geometry
