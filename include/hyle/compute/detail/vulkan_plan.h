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
  // Cold Graph control endpoints. Keep managed fusion on each side separate
  // and materialize a control-only predecessor through the original recipe.
  bool control_boundary = false;
  bool preserve_outputs = false;
};
struct Plan final {
  WorkloadContext workload;
  std::vector<PlanValue> values;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::vector<PlanStep> steps;
};

// Sparse facts emitted by the actual row-cache lowering. A checked row reader is
// retained here only if the final program uses its boundary bank exclusively
// through that reader's original native dispatches, without a bank alias or
// managed import. These are cold facts, not another native plan.
enum class RowCacheAccessKind : std::uint8_t {
  kNone, kMap, kPaddedMap, kCheckedDecode, kPaddedCheckedDecode,
  kPagedIndexerScores
};
struct RowCacheAccessFacts final {
  RowCacheAccessKind kind = RowCacheAccessKind::kNone;
  std::size_t step = 0;
  std::uint32_t bank_value = UINT32_MAX;
  // Actual logical input and native descriptor binding of this bank. They
  // need not be zero or equal to the selected Graph boundary input port.
  std::uint32_t bank_input = 0, bank_binding = 0;
  std::uint32_t query_rows = 0;
  std::uint32_t max_requests = 0;
  std::uint32_t max_refs = 0;
  std::uint32_t page_size = 0;
  std::uint32_t score_columns = 0;
  std::size_t dispatch_begin = 0, dispatch_end = 0;
};

// Optional cold snapshot for Graph diagnostics and selected access proofs.
// It owns only facts from the final native plan, with no native/tensor handles.
struct PlanFacts final {
  std::size_t native_dispatch_count = 0;
  std::size_t managed_region_count = 0;
  std::vector<std::size_t> invocation_storage_block_bytes;
  std::size_t retained_constant_owner_count = 0;
  std::size_t retained_constant_capacity_bytes = 0;
  std::size_t owned_constant_allocation_bytes = 0;
  bool has_borrowed_constants = false;
  std::vector<RowCacheAccessFacts> row_cache_accesses;
};

HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare_plan(const Context &context, const Plan &plan,
             PlanFacts *facts = nullptr) noexcept;

} // namespace hyle::compute::vulkan::detail
#endif
