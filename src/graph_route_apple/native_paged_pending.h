#pragma once

#include "../graph_batch/native_pending_regions.h"
#include "hyle/compute/detail/apple_copy_stream_access.h"
#include "hyle/compute/detail/copy_outcome_source.h"

namespace hyle::graph::apple::detail {
// BatchResources owns the original native F2 Update. Selected row/page slots
// only borrow it for one episode; the one Graph DependentCompletion maps it.
struct NativePagePendingSource final {
  explicit NativePagePendingSource(mem::AllocatorRef selected) noexcept
      : allocator(selected) {}
  mem::AllocatorRef allocator;
  std::shared_ptr<compute::apple::detail::Runtime> runtime;
  compute::detail::CopyOutcomeSource<compute::apple::CopyBatchUpdate,
      compute::apple::detail::CopyStreamAccess> source;
};
extern const batch::detail::NativePendingSourceVTable native_pending_source_table;
Result<void *> create_native_pending_source(
    const std::shared_ptr<compute::apple::detail::Runtime> &, std::size_t,
    mem::AllocatorRef, compute::ResourceBudget) noexcept;
Result<NativePagePendingSource *> native_pending_source(
    const batch::detail::NativePendingSourceOwner &,
    const std::shared_ptr<compute::apple::detail::Runtime> &) noexcept;
Status qualify_native_pending_source(NativePagePendingSource &,
    const compute::apple::CopyBatchUpdate &) noexcept;
} // namespace hyle::graph::apple::detail
