#pragma once

#include "hyle/compute/provider.h"
#include "hyle/error/status.h"

namespace hyle::compute::detail {
struct ProviderPrepared;

// Exact-artifact proof for creating an Invocation without accessing the bytes
// of its caller-supplied workspace, including destruction of a partial result
// after a failure. The callback may still allocate and write independently
// owned private resources. It must be safe to call concurrently with other
// Invocations of the same prepared artifact. This proof grants no access to
// workspace contents, no submission, and no retirement authority.
inline constexpr ProviderExtensionId kColdInvocationWorkspaceExtensionId{
    0x48594c45434f4c44ULL, 0x574f524b53504143ULL, 1};
struct ColdInvocationWorkspaceVTable final {
  Status (*inspect)(const ProviderPrepared &, bool *independent) noexcept = nullptr;
};
} // namespace hyle::compute::detail
