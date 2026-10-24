#include "native_paged_pending.h"

#include "hyle/graph/vulkan.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "../compute_route_vulkan/vulkan_internal.h"

namespace hyle::graph::vulkan {
namespace bd = batch::detail;
namespace cd = compute::detail;
namespace vn = compute::vulkan::detail;
namespace detail {
const bd::NativePendingSourceVTable native_pending_source_table{
    [](void *opaque) noexcept {
      auto *source = static_cast<NativePagePendingSource *>(opaque);
      const auto allocator = source->allocator;
      allocator.destroy(source);
    },
    [](void *opaque) noexcept -> Status {
      auto &source = static_cast<NativePagePendingSource *>(opaque)->source;
      return source.available() && !source.update().valid() ? Status::Ok() :
          Status::Unavailable("native Vulkan Graph still owns its copy source controls");
    }};

Result<void *> create_native_pending_source(const std::shared_ptr<vn::Runtime> &runtime,
    std::size_t capacity, mem::AllocatorRef allocator, compute::ResourceBudget budget) noexcept {
  if (!runtime || !capacity || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("native Vulkan source capacity/owner is empty"));
  auto *source = allocator.construct<NativePagePendingSource>(allocator);
  if (!source) return std::unexpected(Status::ResourceExhausted("native Vulkan copy source metadata"));
  auto created = cd::CopyOutcomeSource<compute::vulkan::CopyBatchUpdate, vn::CopyStreamAccess>::create(
      capacity, {.allocator = allocator, .resource_budget = std::move(budget)});
  if (!created) {
    allocator.destroy(source);
    return std::unexpected(created.error());
  }
  source->runtime = runtime;
  source->source = std::move(*created);
  return source;
}
Result<NativePagePendingSource *> native_pending_source(const bd::NativePendingSourceOwner &owner,
    const std::shared_ptr<vn::Runtime> &runtime) noexcept {
  if (owner.table != &native_pending_source_table || !owner.state || owner.identity != runtime.get())
    return std::unexpected(Status::InvalidArgument("native Vulkan page source implementation/Runtime differs"));
  auto *source = static_cast<NativePagePendingSource *>(owner.state);
  if (source->runtime != runtime)
    return std::unexpected(Status::InvalidArgument("native Vulkan page source owner differs"));
  return source;
}
Status qualify_native_pending_source(NativePagePendingSource &source,
    const compute::vulkan::CopyBatchUpdate &update) noexcept {
  {
    std::lock_guard lock(source.runtime->mutex);
    if (source.runtime->closed)
      return Status::FailedPrecondition("native Vulkan page source is closed");
    HYLE_RETURN_IF_ERROR(source.runtime->failure);
  }
  HYLE_ASSIGN_OR_RETURN(auto prefix, vn::CopyStreamAccess::dependency(update));
  // A qualified Update with all native phases retired has no pending prefix.
  // Its original controls still carry checked status and page coverage.
  if (prefix.pending()) return vn::qualify_copy_dependency(source.runtime, prefix);
  return prefix.terminal_hint ? *prefix.terminal_hint : Status::Ok();
}
} // namespace detail

Result<compute::Completion> submit(Invocation &invocation,
    std::span<const QueueBinding> queues, compute::vulkan::CopyBatchUpdate &&update) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto access, bd::native_pending_access(invocation,
      &detail::native_pending_source_table));
  auto &source = *static_cast<detail::NativePagePendingSource *>(access.source);
  for (const auto &input : access.inputs)
    HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::validate<compute::vulkan::CopyBatchUpdate,
        vn::CopyStreamAccess>(input.frame, update)));
  HYLE_RETURN_IF_ERROR(detail::qualify_native_pending_source(source, update));
  HYLE_ASSIGN_OR_RETURN(auto sources, source.source.take(std::move(update)));
  auto submitted = bd::submit_native_pending(invocation, queues,
      &detail::native_pending_source_table, sources);
  if (!submitted) {
    update = std::move(source.source.update());
    sources.release(sources.state);
  }
  return submitted;
}
Result<compute::vulkan::CopyBatchUpdate> take_update(Invocation &invocation) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto opaque, bd::native_pending_recovery(invocation,
      &detail::native_pending_source_table));
  auto &source = static_cast<detail::NativePagePendingSource *>(opaque)->source;
  if (!source.available() || !source.update().valid() || source.update().in_flight())
    return std::unexpected(Status::Unavailable("native Vulkan source controls have not retired"));
  auto result = std::move(source.update());
  bd::finish_native_pending_recovery(invocation);
  return result;
}
} // namespace hyle::graph::vulkan
