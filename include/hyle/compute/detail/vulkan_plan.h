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
  std::vector<std::uint32_t> outputs;
  // Pure custom schemas lower cold to a native pipeline. Bindings are the
  // schema inputs followed by its single output; no callback survives prepare.
  Kernel custom;
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  std::vector<std::byte> push_constants;
};
struct Plan final {
  WorkloadContext workload;
  std::vector<PlanValue> values;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::vector<PlanStep> steps;
};

// Optional cold snapshot for Graph diagnostics. It owns only capacities and
// counts from the final native plan, with no native or tensor handles.
struct PlanFacts final {
  std::size_t native_dispatch_count = 0;
  std::size_t managed_region_count = 0;
  std::vector<std::size_t> invocation_storage_block_bytes;
  std::size_t retained_constant_owner_count = 0;
  std::size_t retained_constant_capacity_bytes = 0;
  std::size_t owned_constant_allocation_bytes = 0;
  bool has_borrowed_constants = false;
};

HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare_plan(const Context &context, const Plan &plan,
             PlanFacts *facts = nullptr) noexcept;

} // namespace hyle::compute::vulkan::detail
#endif
