#pragma once

#include <cstddef>
#include <span>

#include "hyle/error/result.h"
#include "hyle/graph/resources.h"
#include "hyle/graph/visibility.h"

namespace hyle::graph {
class Executable;
namespace detail { struct InvocationWorkspaceState; struct WorkspaceAccess; }

// One fixed scratch allocation shared by mutually exclusive exact routes.
// No payload view, Source or result permission is exported. Each invocation
// retains this owner; its original native retirement guards exclude reuse.
// Nonempty planner backings must use one native resource. Each remains a
// disjoint aligned span in the shared allocation. Public outputs must be
// caller-owned. Provider-private storage and caller IO stay separate.
// Cold creation/preparation and submission use the same exclusive write
// admission. Unknown native retirement keeps that use alive. A non-OK accepted
// native terminal, including cancellation, makes the root writer unusable:
// recreate the workspace and its borrowing invocations before retrying.
// Existing Invocation control/Completion-observation serialization and native
// resource/Context/allocator lifetime requirements still apply.
class InvocationWorkspace final {
 public:
  InvocationWorkspace() noexcept = default;
  HYLE_GRAPH_API InvocationWorkspace(const InvocationWorkspace &) noexcept;
  HYLE_GRAPH_API InvocationWorkspace &operator=(const InvocationWorkspace &) noexcept;
  HYLE_GRAPH_API InvocationWorkspace(InvocationWorkspace &&) noexcept;
  HYLE_GRAPH_API InvocationWorkspace &operator=(InvocationWorkspace &&) noexcept;
  HYLE_GRAPH_API ~InvocationWorkspace();
  // Only these original exact route identities may borrow the allocation.
  // Finite family InvocationStorageLimits are currently unsupported. For an
  // unlimited family tracker, shared backing is excluded from that family's
  // InvocationStorageUsage; per-invocation metadata/private/unknown accounting
  // remains intact. This root and its owner metadata are charged once in the
  // supplied ResourceBudget, which must be the original scope of every
  // budgeted route. bytes() reports that root's physical backing request.
  HYLE_GRAPH_API static Result<InvocationWorkspace> create(
      std::span<const Executable> exact_routes, ResourceOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  // Exact shared backing extent, charged once through ResourceOptions' budget.
  HYLE_GRAPH_API std::size_t bytes() const noexcept;
 private:
  friend struct detail::WorkspaceAccess;
  explicit InvocationWorkspace(detail::InvocationWorkspaceState *state) noexcept : state_(state) {}
  detail::InvocationWorkspaceState *state_ = nullptr;
};
static_assert(sizeof(InvocationWorkspace) == sizeof(void *));
} // namespace hyle::graph
