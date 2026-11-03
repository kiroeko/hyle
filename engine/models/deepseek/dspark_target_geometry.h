#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// The two pinned GA configs use the same target mathematics. Only these
// immutable cold dimensions/routing facts vary; codecs, state ownership and
// execution stay in their existing layers. This is not a model registry.
namespace dspark_target_geometry {
inline constexpr std::int64_t kDepth=512,kWindow=128,kIndexDepth=128,kIndexHeads=64;
inline constexpr std::int64_t kOutputRank=1024,kSlots=6,kStreams=4;
struct Geometry final {
  std::int64_t hidden,heads,query_rank,groups,experts,intermediate,index_top;
  float route_scale;
  std::int64_t first_target_layer;
  constexpr std::int64_t group_width() const noexcept {return heads*kDepth/groups;}
  void validate() const {
    constexpr auto limit=std::numeric_limits<std::int32_t>::max();
    if(hidden<=0||hidden>limit/kStreams||hidden%128||heads<=0||heads>limit/kDepth||
        query_rank<=0||query_rank>limit||query_rank%128||groups<=0||groups>heads||heads%groups||
        experts<kSlots||experts>limit||intermediate<=0||intermediate>limit||intermediate%128||
        index_top<=0||index_top>limit||!std::isfinite(route_scale)||route_scale<=0.F||
        first_target_layer<0||first_target_layer>limit-2)
      throw std::runtime_error("original DeepSeek target geometry exceeds its typed domain");
  }
};
inline constexpr Geometry kFlash{4096,64,1024,8,256,2048,512,1.5F,40};
inline constexpr Geometry kPro{7168,128,1536,16,384,3072,1024,2.5F,58};
} // namespace dspark_target_geometry
