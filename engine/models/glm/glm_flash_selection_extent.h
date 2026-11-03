#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Cold geometry only. The physical selection carrier stays2051 entries;
// readers consume the actual selected pools immediately followed by live tail.
namespace glm_flash_selection_extent {
inline constexpr std::int64_t kPool=4,kMaxPools=512,kMaxTokens=2048,kStorage=2051;
struct Extent final {
  std::int64_t visible_pools,selected_pools,tail;
  constexpr std::int64_t pool_tokens()const noexcept{return selected_pools*kPool;}
  constexpr std::int64_t live_tokens()const noexcept{return pool_tokens()+tail;}
  constexpr std::int64_t padding()const noexcept{return kStorage-live_tokens();}
};
Extent fresh(std::int64_t position);
// Existing same-epoch MTP skip calls use position0 as a placeholder. Their
// carried selection is mature; never infer its provenance from that placeholder.
Extent carried(std::int32_t tail);
} // namespace glm_flash_selection_extent
