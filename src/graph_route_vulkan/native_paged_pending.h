#pragma once

#include "../graph_batch/native_pending_regions.h"
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#include "hyle/compute/detail/copy_outcome_source.h"

namespace hyle::graph::vulkan::detail {
// Actual native F2 source controls only. The original Graph resource payload
// owns this object; native page slots borrow its stable Update during one
// episode. The sole Graph DependentCompletion maps source and accepted suffix.
struct NativePagePendingSource final {
  explicit NativePagePendingSource(mem::AllocatorRef a) noexcept : allocator(a) {}
  mem::AllocatorRef allocator;
  std::shared_ptr<compute::vulkan::detail::Runtime> runtime;
  compute::detail::CopyOutcomeSource<compute::vulkan::CopyBatchUpdate,
      compute::vulkan::detail::CopyStreamAccess> source;
};
extern const batch::detail::NativePendingSourceVTable native_pending_source_table;
Result<void *> create_native_pending_source(
    const std::shared_ptr<compute::vulkan::detail::Runtime> &, std::size_t,
    mem::AllocatorRef, compute::ResourceBudget) noexcept;
Result<NativePagePendingSource *> native_pending_source(
    const batch::detail::NativePendingSourceOwner &,
    const std::shared_ptr<compute::vulkan::detail::Runtime> &) noexcept;
Status qualify_native_pending_source(NativePagePendingSource &,
    const compute::vulkan::CopyBatchUpdate &) noexcept;
} // namespace hyle::graph::vulkan::detail
