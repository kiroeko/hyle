#pragma once
#include "hyle/compute/state_versions.h"

namespace hyle::compute::detail {
// C++ same-toolchain lifecycle bridge over original typed F2 objects. No
// operation, scheduling, Queue, native completion or mathematical callback.
struct StateCopyDriver final {
  void *object = nullptr;
  std::uint32_t slots = 0;
  Result<std::uint32_t> (*start)(void *, const ReadLease &, WriteLease &&, std::size_t) noexcept = nullptr;
  Result<WriteLease> (*take)(void *, std::uint32_t, bool) noexcept = nullptr;
  Status (*wait)(void *, std::uint32_t) noexcept = nullptr;
  void (*observe)(void *, std::uint32_t, bool) noexcept = nullptr;
  Status (*cancel)(void *, std::uint32_t) noexcept = nullptr;
  Status (*collect)(void *) noexcept = nullptr;
  Status (*close)(void *) noexcept = nullptr;
  Status (*drain)(void *) noexcept = nullptr;
  Status (*rearm)(void *) noexcept = nullptr;
  std::uint32_t (*active)(void *) noexcept = nullptr;
  Result<CopyBatchReport> (*report)(void *) noexcept = nullptr;
  void (*destroy)(void *) noexcept = nullptr;
};
struct StateCloneAccess final {
  HYLE_COMPUTE_STATE_API static Result<mem::AllocatorRef> allocator(const StateStore &) noexcept;
  HYLE_COMPUTE_STATE_API static Status preflight(const StateStore &, std::uint32_t,
                                                std::size_t) noexcept;
  // Consumes driver only on successful registration. Its target LeasePool is
  // store-owned; original native resource/Context/factory borrows stay explicit.
  HYLE_COMPUTE_STATE_API static Result<StateCloneRoute> register_driver(const StateStore &,
      StateCopyDriver &, mem::MemoryResourceRef, std::size_t, std::size_t) noexcept;
};
} // namespace hyle::compute::detail
