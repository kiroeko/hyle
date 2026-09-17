#pragma once

#include "hyle/compute/vulkan.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hyle::compute::vulkan::detail {

// Cold cross-factor lowering seam. Graph owns DAG legality/topological order;
// Compute owns native implementations and resource/command construction. This
// recipe is discarded after preparation and is never a hot graph executor.
struct PlanValue final {
  TensorDesc description;
  Tensor constant;
};
struct PlanStep final {
  Operation operation;
  std::vector<std::uint32_t> inputs;
  std::uint32_t output = 0;
  // Pure custom schemas lower cold to a native pipeline. Bindings are the
  // schema inputs followed by its single output; no callback survives prepare.
  Kernel custom;
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  std::vector<std::byte> push_constants;
};
struct Plan final {
  std::vector<PlanValue> values;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::vector<PlanStep> steps;
};

HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare_plan(const Context &context, const Plan &plan) noexcept;

} // namespace hyle::compute::vulkan::detail
#endif
