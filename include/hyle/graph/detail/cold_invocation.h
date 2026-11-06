#pragma once

#include "hyle/compute/provider.h"
#include "hyle/error/status.h"

namespace hyle::graph {
struct ProviderRegionPrepared;
namespace detail {

// Exact selected-region proof. Creation and any generated-output projection
// during creation (including cleanup of a partial result) do not read or write
// caller workspace bytes. Creating independent Invocations of the artifact
// concurrently is supported. Private owned allocations remain subject to their
// original budget and lifetime.
inline constexpr compute::ProviderExtensionId kRegionColdInvocationWorkspaceExtensionId{
    0x48594c45434f4c44ULL, 0x524547494f4e5753ULL, 1};
struct RegionColdInvocationWorkspaceVTable final {
  Status (*inspect)(const ProviderRegionPrepared &, bool *independent) noexcept = nullptr;
};
} // namespace detail
} // namespace hyle::graph
