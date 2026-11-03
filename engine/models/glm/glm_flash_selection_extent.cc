#include "glm_flash_selection_extent.h"

namespace glm_flash_selection_extent {
Extent fresh(std::int64_t position) {
  if(position<0||position>=std::numeric_limits<std::int32_t>::max())
    throw std::runtime_error("Flash selection requires a nonnegative representable next cursor");
  const auto rows=position+1,pools=rows/kPool;
  return {pools,std::min(pools,kMaxPools),rows%kPool};
}
Extent carried(std::int32_t tail) {
  if(tail<0||tail>=kPool)throw std::runtime_error("Flash carried tail is outside0..3");
  return {kMaxPools,kMaxPools,tail};
}
}
