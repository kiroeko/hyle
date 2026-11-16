#pragma once

#include "hyle/compute/apple.h"
#include "hyle/compute/batch_layout.h"

namespace hyle::compute::apple::detail {
HYLE_COMPUTE_ROUTE_APPLE_API Status qualify_device_lengths_paged_schema(
    const BatchSchema &, float scale, bool causal, std::int32_t window_left) noexcept;
// Cold native builder. The original schema/version bounds every page and
// request; device lengths can only shorten that envelope. Input ports are
// physical Q/K/V, I32 query/KV lengths, and canonical readonly layout metadata.
// The prepared program freezes schema geometry, not one version's bounds;
// another owned capacity version only replaces that original metadata binding.
// Page permission/import and publication remain with the typed Graph owner.
HYLE_COMPUTE_ROUTE_APPLE_API Result<Program> prepare_device_lengths_paged_graph(
    std::shared_ptr<Runtime>, BatchSchema,
    std::uint32_t physical_query_rows, float scale, bool causal,
    std::int32_t window_left, const WorkloadContext & = {}, bool guard_writes = false) noexcept;
} // namespace hyle::compute::apple::detail
