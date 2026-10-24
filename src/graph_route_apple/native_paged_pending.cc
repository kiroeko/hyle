#include "native_paged_pending.h"

#include "../compute_route_apple/apple_copy_internal.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/graph/apple.h"

namespace hyle::graph::apple::detail {
namespace ca = compute::apple;
namespace an = ca::detail;
namespace cd = compute::detail;
namespace bd = batch::detail;
// Only the original BatchResources payload owns the actual native Update.
// All selected page slots borrow it until the one DependentCompletion retires.
const bd::NativePendingSourceVTable native_pending_source_table{
    [](void *opaque) noexcept {
      auto *source = static_cast<NativePagePendingSource *>(opaque);
      const auto allocator = source->allocator;
      allocator.destroy(source);
    },
    [](void *opaque) noexcept -> Status {
      auto &source = static_cast<NativePagePendingSource *>(opaque)->source;
      return source.available() && !source.update().valid() ? Status::Ok() :
          Status::Unavailable("Apple Graph still owns its copy source controls");
    }};
Result<void *> create_native_pending_source(const std::shared_ptr<an::Runtime> &owner,
    std::size_t capacity, mem::AllocatorRef allocator,
    compute::ResourceBudget budget) noexcept {
  if (!owner || !capacity || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("Apple page source needs its actual Runtime and reserved capacity"));
  auto *source = allocator.construct<NativePagePendingSource>(allocator);
  if (!source) return std::unexpected(Status::ResourceExhausted("Apple copy source metadata"));
  auto created = cd::CopyOutcomeSource<ca::CopyBatchUpdate, an::CopyStreamAccess>::create(
      capacity, {.allocator = allocator, .resource_budget = std::move(budget)});
  if (!created) {
    allocator.destroy(source);
    return std::unexpected(created.error());
  }
  source->runtime = owner;
  source->source = std::move(*created);
  return source;
}
Result<NativePagePendingSource *> native_pending_source(
    const bd::NativePendingSourceOwner &source,
    const std::shared_ptr<an::Runtime> &owner) noexcept {
  if (source.table != &native_pending_source_table || !source.state ||
      source.identity != owner.get())
    return std::unexpected(Status::InvalidArgument("Apple page source implementation/Runtime differs"));
  auto *actual = static_cast<NativePagePendingSource *>(source.state);
  if (actual->runtime != owner)
    return std::unexpected(Status::InvalidArgument("Apple page source actual owner differs"));
  return actual;
}
Status qualify_native_pending_source(NativePagePendingSource &source,
    const ca::CopyBatchUpdate &update) noexcept {
  {
    std::lock_guard lock(source.runtime->queue_mutex);
    if (!source.runtime->accepting)
      return Status::FailedPrecondition("Apple page source is closed");
    HYLE_RETURN_IF_ERROR(source.runtime->failure);
  }
  HYLE_ASSIGN_OR_RETURN(auto prefix, an::CopyStreamAccess::dependency(update));
  // Fully retired native phases have no pending prefix. Their original Update
  // still proves checked status, actual writers and complete page coverage.
  if (prefix.pending()) return an::qualify_recording_dependency(source.runtime, prefix);
  return prefix.terminal_hint ? *prefix.terminal_hint : Status::Ok();
}
} // namespace hyle::graph::apple::detail

namespace hyle::graph::apple {
Result<compute::Completion> submit(Invocation &invocation,
    std::span<const QueueBinding> queues, compute::apple::CopyBatchUpdate &&update) noexcept {
  namespace bd = batch::detail;
  namespace an = compute::apple::detail;
  namespace cd = compute::detail;
  HYLE_ASSIGN_OR_RETURN(auto access, bd::native_pending_access(invocation,
      &detail::native_pending_source_table));
  auto &source = *static_cast<detail::NativePagePendingSource *>(access.source);
  for (const auto &input : access.inputs)
    HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::validate<compute::apple::CopyBatchUpdate,
        an::CopyStreamAccess>(input.frame, update)));
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
Result<compute::apple::CopyBatchUpdate> take_update(Invocation &invocation) noexcept {
  namespace bd = batch::detail;
  HYLE_ASSIGN_OR_RETURN(auto opaque, bd::native_pending_recovery(invocation,
      &detail::native_pending_source_table));
  auto &source = static_cast<detail::NativePagePendingSource *>(opaque)->source;
  if (!source.available() || !source.update().valid() || source.update().in_flight())
    return std::unexpected(Status::Unavailable("Apple copy source controls have not retired"));
  auto result = std::move(source.update());
  bd::finish_native_pending_recovery(invocation);
  return result;
}
} // namespace hyle::graph::apple
