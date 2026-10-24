#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "glm_moe_geometry.h"
#include "glm_shared_vocabulary.h"

// Two actual source encodings, independent of repository identity. MXFP4
// carriers concatenate unchanged U8 codes then U8 UE8 scales; they are not
// GGUF blocks, an FP8-A128 carrier, or an expanded floating weight bank.
namespace glm_moe_storage {
namespace c=hyle::compute;
using Geometry=glm_moe_geometry::Geometry;
using Layout=glm_moe_geometry::Layout;
enum class Kind {kFloat8Block128,kMxfp4Group32};
inline void require(Kind kind,const Geometry &model) {
  glm_moe_geometry::require(model);
  if(kind!=Kind::kFloat8Block128&&kind!=Kind::kMxfp4Group32)
    throw std::runtime_error("unknown original GLM MoE storage");
}
inline std::size_t code_bytes(Kind kind,const Geometry &model) {
  require(kind,model);return model.code_bytes()/(kind==Kind::kMxfp4Group32?2:1);
}
inline std::size_t scale_bytes(Kind kind,const Geometry &model) {
  require(kind,model);return kind==Kind::kMxfp4Group32?model.code_bytes()/32:model.scale_bytes();
}
inline Layout layout(Kind kind,const Geometry &model) {
  require(kind,model);
  const auto codes=code_bytes(kind,model),scales=scale_bytes(kind,model);
  const auto bank=model.slots*(codes+scales),map=std::size_t(model.experts)*4;
  return {codes,scales,bank,map,3*bank+map,
      3*((bank+255)/256*256)+(map+255)/256*256,6*model.slots+1};
}
inline c::TensorDesc packed_description(std::int64_t n,std::int64_t k,std::size_t slots,Kind kind,bool grouped) {
  if(n<=0||k<=0||n%128||k%128||!slots||(!grouped&&slots!=1))
    throw std::runtime_error("original MoE carrier geometry differs");
  if(kind==Kind::kFloat8Block128) {
    return !grouped?glm_shared_vocabulary::take(c::float8_weight_description(k,n,
        c::Float8ScaleMode::kFloat32Block128),"original shared E4 carrier"):
        glm_shared_vocabulary::take(c::float8_grouped_weight_description(std::int64_t(slots),n,k),"original resident E4 carrier");
  }
  if(kind!=Kind::kMxfp4Group32||std::uint64_t(n)>std::uint64_t(std::numeric_limits<std::int64_t>::max())/std::uint64_t(k))
    throw std::runtime_error("original MXFP4 carrier extent overflows");
  const auto elements=std::uint64_t(n)*std::uint64_t(k),bytes=elements/2+elements/32;
  if(slots>std::uint64_t(std::numeric_limits<std::int64_t>::max())/bytes)
    throw std::runtime_error("original MXFP4 resident extent overflows");
  return glm_shared_vocabulary::take(c::TensorDesc::dense(c::DataType::kUInt8,
      std::array<std::int64_t,1>{std::int64_t(slots*bytes)}),"original MXFP4 unchanged U8 carrier");
}
} // namespace glm_moe_storage
