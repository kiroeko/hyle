#pragma once

#include "hyle/compute/resource_leases.h"

namespace hyle::compute::detail {
// Typed native adapters transfer their actual resource owner only after
// backing adoption succeeds. No payload/queue erasure or execution callback.
struct LeaseAccess final {
  static const void *identity(const ReadLease &lease) noexcept { return lease.block_; }
  static const void *identity(const WriteLease &lease) noexcept { return lease.block_; }
  using ReleaseOwnerFn = void (*)(void *owner) noexcept;
  static HYLE_COMPUTE_API void retain_resource_owner(ReadLease &lease, void *owner,
                                                     ReleaseOwnerFn release) noexcept;
  static HYLE_COMPUTE_API void retain_resource_owner(LeasePool &pool, void *owner,
                                                     ReleaseOwnerFn release) noexcept;
  // Native route calls this only after its original actual-retirement proof.
  static HYLE_COMPUTE_API void retire_native(LeaseUseBatch &uses, Status terminal) noexcept;
  // F2 only: cold binding has deduplicated roots. The owning copy controller
  // proves native ordering for pending mutable sources/writers, retains all
  // producer state and never exposes an untracked mutable alias. These are
  // permission/lifetime operations, not execution or dependency validation.
  static HYLE_COMPUTE_API Status read_unique(LeaseUseBatch &uses, const ReadLease &lease) noexcept;
  static HYLE_COMPUTE_API Status ordered_read_unique(LeaseUseBatch &uses, const WriteLease &lease) noexcept;
  static HYLE_COMPUTE_API Status write_unique(LeaseUseBatch &uses, const WriteLease &lease,
                                             bool ordered_after_prior) noexcept;
  // A group may publish after its last successful actual write while ordered
  // readonly successors still retain storage. No subsequent writer is allowed.
  // Public WriteLease::publish keeps its original all-use uniqueness rule.
  static HYLE_COMPUTE_API Result<ReadLease> publish_after_writes(WriteLease &lease) noexcept;
};
} // namespace hyle::compute::detail
