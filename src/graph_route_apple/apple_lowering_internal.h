#pragma once

#include "hyle/graph/apple.h"
#include "hyle/graph/region_provider.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/mem/std_allocator.h"

#include <vector>

namespace hyle::graph::apple::detail {
struct PreparedRowAccess final {
  std::uint32_t node = 0;
  std::uint32_t input_port = UINT32_MAX;
  compute::apple::detail::RowCacheAccessFacts native;
};
using PreparedRowAccesses = std::vector<PreparedRowAccess,
    mem::StdAllocator<PreparedRowAccess>>;
struct PreparedConstantStorage final {
  std::size_t count = 0;
  std::size_t bytes = 0;
};
// The original typed lowering and native Program, restricted to the selected
// Graph boundary. Constants follow request.constants order and are ready real
// native array views; no second graph or per-submit lowering is constructed.
Result<compute::apple::Program> prepare_region(
    const std::shared_ptr<compute::apple::detail::Runtime> &,
    const RegionCompileRequest &, std::span<const mlx::core::array>,
    CustomLowering, bool *custom_failed, PreparedRowAccesses *,
    PreparedConstantStorage *) noexcept;
}
