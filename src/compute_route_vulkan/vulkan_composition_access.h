#pragma once

#include "hyle/compute/vulkan.h"

namespace hyle::compute::vulkan::detail {
// Cold composition seam only. Outputs are uninitialized storage, not ready
// results. A coarse plan binds its internal references before any execution,
// then freezes the exact reference baseline. Ordinary native users keep the
// original exported-owner guard; this header is not installed as public API.
struct CompositionAccess final {
  // Retired Graph-only cold edits revoke execution without discounting old
  // readers or beginning a result writer. Direct callers keep reuse_gate.
  // prepare_outputs, full bind and final seal must follow before submission.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status
  prepare_bindings(Invocation &invocation) noexcept;
  // Ordinary Graph generated results only. Preserve the original Run and
  // endpoint; rotate complete owned blocks whose retired readonly outputs
  // escape. On success refresh is true until caller exports and seal finish.
  // No public ResultLease/native-use/observer is discounted by this seam.
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status
  prepare_outputs(Invocation &invocation, bool *refresh) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  output(Invocation &invocation, std::size_t index) noexcept;
  static HYLE_COMPUTE_ROUTE_VULKAN_API Status seal(Invocation &invocation) noexcept;
};
} // namespace hyle::compute::vulkan::detail
