#pragma once

#include "hyle/compute/resource_leases.h"

namespace hyle::compute::detail {
// Typed native adapters transfer their actual resource owner only after
// backing adoption succeeds. No payload/queue erasure or execution callback.
struct LeaseAccess final {
  using ReleaseOwnerFn = void (*)(void *owner) noexcept;
  static HYLE_COMPUTE_API void retain_resource_owner(ReadLease &lease, void *owner,
                                                     ReleaseOwnerFn release) noexcept;
  static HYLE_COMPUTE_API void retain_resource_owner(LeasePool &pool, void *owner,
                                                     ReleaseOwnerFn release) noexcept;
};
} // namespace hyle::compute::detail
