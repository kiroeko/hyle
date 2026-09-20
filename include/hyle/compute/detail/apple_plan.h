#pragma once

#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include <cstddef>

namespace hyle::compute::apple::detail {

// Compute-owned, cold region boundary used by the native Graph compiler.
// It deliberately carries neither Graph types nor a per-kernel executor.
enum class RegionKind { kMlx, kGatedDeltaNet };

struct Region final {
  RegionKind kind = RegionKind::kMlx;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  RegionFunction mlx;
  std::vector<TensorDesc> recurrent_inputs;
};

struct RegionPlan final {
  WorkloadContext workload;
  std::vector<InputSpec> values;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> constants;
  std::vector<mlx::core::array> constant_values;
  std::vector<std::size_t> outputs;
  std::vector<Region> regions;
};

struct ContextAccess final {
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<Program>
  compile(const Context &context, RegionPlan plan) noexcept;
};

} // namespace hyle::compute::apple::detail
#endif
