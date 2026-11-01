#pragma once
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/prepared_operation.h"

namespace hyle::compute::detail {
// Separate opt-in; the existing BatchPlanning v7/MLA v5 ABI is unchanged.
inline constexpr ProviderExtensionId kBatchFloatWorkspaceExtensionId{
    0x48594c4542415443ULL, 0x464c4f4154534352ULL, 1};
struct BatchFloatWorkspaceVTable final {
  // Cold, host-only qualification for this concrete Q carrier and original
  // schema. The returned capacity is configured, never inferred from future
  // workload layouts. The provider may decline unprofitable small carriers.
  //
  // On success float storage is disposable run scratch: no contents survive a
  // native run. All cold query/update/plan/replan/setup paths, including their
  // failure paths, neither read nor write that payload; they may only compute
  // offsets from its fixed coordinates/capacity. Int plan tables stay private.
  // Recording executes no math. Actual run/replay accesses float only inside
  // the outer Graph region's admitted Queue sequence and real retirement use.
  // An absent/declined capability keeps the original owned workspace path.
  Status (*workspace)(void *, BatchCapacity, std::uint32_t physical_query_rows,
      const BatchProgramOptions &, WorkspaceRequirement *) noexcept = nullptr;
};
} // namespace hyle::compute::detail
