#pragma once
#include "hyle/compute/paged_arena.h"
namespace hyle::compute::detail {
// Trusted batch adapters only. Full span is an addressable capacity, not read
// permission for unused pages. Retain/admit every returned page guard, and only
// run a qualified operation whose actual reads follow this immutable layout.
struct PagedArenaAccess final {
  HYLE_COMPUTE_BATCH_API static Result<ConstTensorView> tensor(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const ReadLease> pages(const PagedArenaRead&) noexcept;
};
} // namespace hyle::compute::detail
