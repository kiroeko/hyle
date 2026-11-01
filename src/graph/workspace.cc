#include "hyle/graph/workspace.h"

#include <algorithm>
#include <bit>
#include <memory>
#include <utility>

#include "hyle/mem/alignment.h"
#include "plan_state.h"
#include "workspace_internal.h"

namespace hyle::graph {
namespace {
struct ScratchBacking final {
  mem::MemoryResourceRef resource;
  std::size_t offset = 0, size = 0, alignment = alignof(std::max_align_t);
};
struct ScratchLayout final {
  explicit ScratchLayout(mem::AllocatorRef allocator) noexcept
      : resource_begins(allocator), backings(allocator) {}
  container::SmallVector<std::size_t, 0, mem::AllocatorRef> resource_begins;
  container::SmallVector<ScratchBacking, 0, mem::AllocatorRef> backings;
  mem::MemoryResourceRef resource;
  std::size_t bytes = 0, alignment = alignof(std::max_align_t);
};
Result<ScratchLayout> qualify(const Executable &route, mem::AllocatorRef allocator) noexcept {
  const auto *plan = detail::ExecutableAccess::state(route);
  if (!plan || !route.has_execution_route() || route.is_shape_domain())
    return std::unexpected(Status::InvalidArgument("shared scratch needs concrete exact routes"));
  if (plan->invocation_storage && plan->invocation_storage->usage().limit_bytes != SIZE_MAX)
    return std::unexpected(Status::Unimplemented("shared scratch does not support finite family invocation storage quotas"));
  if (plan->workspace_resources.size() != plan->workspace_plan.resource_count())
    return std::unexpected(Status::Internal("shared scratch resource inventory differs"));
  for (std::size_t index = 0; index < route.output_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto ownership, route.output_ownership(index));
    if (ownership != OutputOwnership::kCaller)
      return std::unexpected(Status::Unimplemented("shared scratch requires independent caller-owned public outputs"));
    HYLE_ASSIGN_OR_RETURN(auto value, route.output_id(index));
    auto root = plan->external_instances[value.ordinal()];
    while (plan->alias_sources[root] != detail::kNoInPlaceSource) root = plan->alias_sources[root];
    if (plan->workspace_requests[root] != detail::kNoWorkspaceRequest ||
        (plan->instance_roles[root] & detail::kGeneratedOutput))
      return std::unexpected(Status::Unimplemented("public output can escape shared scratch"));
  }
  ScratchLayout layout(allocator);
  if (plan->workspace_plan.resource_count() == SIZE_MAX ||
      !layout.resource_begins.try_reserve(plan->workspace_plan.resource_count() + 1) ||
      !layout.backings.try_reserve(plan->workspace_backing_count))
    return std::unexpected(Status::ResourceExhausted("shared scratch backing map metadata"));
  // Planner resource slots are disjoint happens-before chains, not distinct
  // native resources. Preserve every original physical backing as a disjoint
  // aligned span; never overlap chains merely because their resource matches.
  for (std::size_t slot = 0; slot < plan->workspace_plan.resource_count(); ++slot) {
    if (!layout.resource_begins.try_push_back(layout.backings.size()))
      return std::unexpected(Status::ResourceExhausted("shared scratch resource map metadata"));
    HYLE_ASSIGN_OR_RETURN(auto resource_plan, plan->workspace_plan.resource_plan(slot));
    for (std::size_t index = 0; index < resource_plan.backing_count; ++index) {
      HYLE_ASSIGN_OR_RETURN(auto backing, plan->workspace_plan.backing_plan(slot, index));
      if (!std::has_single_bit(backing.alignment))
        return std::unexpected(Status::InvalidArgument("shared scratch backing alignment"));
      const auto actual = plan->workspace_resources[slot];
      std::size_t offset = 0;
      if (backing.size) {
        if (layout.resource.valid() && layout.resource != actual)
          return std::unexpected(Status::Unimplemented("shared scratch requires one nonempty native resource"));
        layout.resource = actual;
        if (!mem::try_align_up(layout.bytes, backing.alignment, &offset) || backing.size > SIZE_MAX - offset)
          return std::unexpected(Status::OutOfRange("shared scratch aligned backing extent overflow"));
        layout.bytes = offset + backing.size;
        layout.alignment = std::max(layout.alignment, backing.alignment);
      }
      if (!layout.backings.try_push_back({actual, offset, backing.size, backing.alignment}))
        return std::unexpected(Status::ResourceExhausted("shared scratch backing map metadata"));
    }
  }
  if (!layout.resource_begins.try_push_back(layout.backings.size()))
    return std::unexpected(Status::ResourceExhausted("shared scratch resource map metadata"));
  for (std::size_t index = 0; index < plan->workspace_plan.request_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto placement, plan->workspace_plan.placement(index));
    if (placement.resource_slot >= plan->workspace_plan.resource_count())
      return std::unexpected(Status::InvalidArgument("shared scratch placement resource"));
    const auto begin = layout.resource_begins[placement.resource_slot];
    const auto end = layout.resource_begins[placement.resource_slot + 1];
    if (placement.backing_index >= end - begin)
      return std::unexpected(Status::InvalidArgument("shared scratch placement backing"));
    HYLE_ASSIGN_OR_RETURN(auto backing, plan->workspace_plan.backing_plan(
        placement.resource_slot, placement.backing_index));
    if (!std::has_single_bit(placement.alignment) || placement.alignment > backing.alignment ||
        placement.offset % placement.alignment || placement.offset > backing.size ||
        placement.size > backing.size - placement.offset)
      return std::unexpected(Status::InvalidArgument("shared scratch placement is outside its exact backing"));
  }
  return layout;
}
} // namespace

void detail::InvocationWorkspaceState::destroy() noexcept {
  const auto owner = allocator;
  this->~InvocationWorkspaceState();
  owner.deallocate(this, sizeof(InvocationWorkspaceState), alignof(InvocationWorkspaceState));
}
InvocationWorkspace::InvocationWorkspace(const InvocationWorkspace &other) noexcept : state_(other.state_) {
  if (state_) state_->incref();
}
InvocationWorkspace &InvocationWorkspace::operator=(const InvocationWorkspace &other) noexcept {
  if (this != &other) { InvocationWorkspace copy(other); *this = std::move(copy); }
  return *this;
}
InvocationWorkspace::InvocationWorkspace(InvocationWorkspace &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
InvocationWorkspace &InvocationWorkspace::operator=(InvocationWorkspace &&other) noexcept {
  if (this != &other) { if (state_) state_->decref(); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
InvocationWorkspace::~InvocationWorkspace() { if (state_) state_->decref(); }
std::size_t InvocationWorkspace::bytes() const noexcept { return state_ ? state_->coordinates.size() : 0; }

Result<InvocationWorkspace> InvocationWorkspace::create(std::span<const Executable> routes,
                                                       ResourceOptions options) noexcept {
  if (routes.empty() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("shared scratch requires routes and an allocator"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  mem::MemoryResourceRef resource;
  mem::LifetimeBackingPlan maximum{};
  for (const auto &route : routes) {
    HYLE_ASSIGN_OR_RETURN(auto layout, qualify(route, allocator));
    const auto *plan = detail::ExecutableAccess::state(route);
    if (plan->invocation_storage) {
      const auto &required = plan->invocation_storage->resource_budget();
      if (required) {
        if (!options.resource_budget)
          return std::unexpected(Status::InvalidArgument("shared scratch needs the route's original resource budget"));
        HYLE_ASSIGN_OR_RETURN(auto original, required.metadata_allocator());
        // These are the original self-retaining budget adapter callbacks and
        // that scope's fixed metadata row, not its possibly shared backing
        // allocator. Equal backing allocators cannot prove equal budgets.
        if (original.alloc_fn() != allocator.alloc_fn() || original.free_fn() != allocator.free_fn() ||
            original.realloc_fn() != allocator.realloc_fn() || original.state() != allocator.state())
          return std::unexpected(Status::InvalidArgument("shared scratch resource budget differs from its route"));
      }
    }
    if (layout.resource.valid()) {
      if (resource.valid() && resource != layout.resource)
        return std::unexpected(Status::InvalidArgument("shared scratch routes use different resources"));
      resource = layout.resource;
    }
    maximum.size = std::max(maximum.size, layout.bytes);
    maximum.alignment = std::max(maximum.alignment, layout.alignment);
  }
  if (!resource.valid() || !maximum.size)
    return std::unexpected(Status::Unimplemented("shared scratch requires a nonempty backing"));
  if (maximum.size > resource.properties().maximum_allocation)
    return std::unexpected(Status::OutOfRange("shared scratch exceeds resource allocation capacity"));
  auto *state = allocator.construct<detail::InvocationWorkspaceState>(allocator);
  if (!state) return std::unexpected(Status::ResourceExhausted("shared scratch owner metadata"));
  auto result = detail::WorkspaceAccess::adopt(state);
  if (!state->routes.try_reserve(routes.size()))
    return std::unexpected(Status::ResourceExhausted("shared scratch route metadata"));
  for (const auto &route : routes)
    if (!state->routes.try_push_back(route))
      return std::unexpected(Status::ResourceExhausted("shared scratch route metadata"));
  HYLE_ASSIGN_OR_RETURN(state->pool, compute::LeasePool::create(resource, maximum.size,
      {.max_blocks = 1, .backing_alignment = maximum.alignment,
       .allocator = allocator, .resource_budget = options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(state->writer, state->pool.allocate(maximum.size, maximum.alignment));
  HYLE_ASSIGN_OR_RETURN(state->coordinates, state->writer.view());
  HYLE_ASSIGN_OR_RETURN(state->preparation_uses, compute::LeaseUsePool::create(
      {.max_batches = 1, .max_resources_per_batch = 1, .allocator = allocator,
       .resource_budget = options.resource_budget}));
  return result;
}

Result<detail::WorkspaceInstance> detail::WorkspaceInstance::borrow(
    const InvocationWorkspace &owner, const PlanState &plan, mem::AllocatorRef allocator) noexcept {
  auto *state = WorkspaceAccess::state(owner);
  if (!state)
    return std::unexpected(Status::InvalidArgument("shared scratch is empty"));
  const auto route = std::find_if(state->routes.begin(), state->routes.end(), [&](const auto &candidate) {
        return ExecutableAccess::state(candidate) == &plan;
      });
  if (route == state->routes.end())
    return std::unexpected(Status::InvalidArgument("route was not admitted to this shared scratch"));
  HYLE_ASSIGN_OR_RETURN(auto layout, qualify(*route, allocator));
  if (layout.bytes > state->coordinates.size() ||
      (layout.resource.valid() && layout.resource != state->coordinates.resource()))
    return std::unexpected(Status::Internal("shared scratch no longer covers its admitted route"));
  WorkspaceInstance result;
  result.shared_ = util::retain_ref(state);
  result.views_ = decltype(result.views_)(allocator);
  result.empty_backings_ = decltype(result.empty_backings_)(allocator);
  container::SmallVector<mem::BufferView, 0, mem::AllocatorRef> backings(allocator);
  if (!backings.try_reserve(layout.backings.size()) ||
      !result.empty_backings_.try_reserve(layout.backings.size()))
    return std::unexpected(Status::ResourceExhausted("shared scratch empty backing metadata"));
  for (const auto &backing : layout.backings) {
    mem::BufferView view;
    if (backing.size) {
      HYLE_ASSIGN_OR_RETURN(view, state->coordinates.subview(backing.offset, backing.size));
    } else {
      // Zero-only planner slots preserve their real resource identity. They
      // carry no bytes and grant no access to the nonempty shared allocation.
      HYLE_ASSIGN_OR_RETURN(auto empty, mem::Buffer::create(backing.resource, 0, backing.alignment));
      view = empty.view();
      if (!result.empty_backings_.try_push_back(std::move(empty)))
        return std::unexpected(Status::ResourceExhausted("shared scratch empty backing metadata"));
    }
    if (!backings.try_push_back(view))
      return std::unexpected(Status::ResourceExhausted("shared scratch backing view metadata"));
  }
  if (!result.views_.try_reserve(plan.workspace_plan.request_count()))
    return std::unexpected(Status::ResourceExhausted("shared scratch placement metadata"));
  for (std::size_t index = 0; index < plan.workspace_plan.request_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto placement, plan.workspace_plan.placement(index));
    const auto backing = layout.resource_begins[placement.resource_slot] + placement.backing_index;
    HYLE_ASSIGN_OR_RETURN(auto view, backings[backing].subview(placement.offset, placement.size));
    if (!result.views_.try_push_back(view))
      return std::unexpected(Status::ResourceExhausted("shared scratch placement metadata"));
  }
  return result;
}
Result<compute::LeaseUseBatch> detail::WorkspaceInstance::begin_preparation() const noexcept {
  if (!shared_) return compute::LeaseUseBatch{};
  HYLE_ASSIGN_OR_RETURN(auto use, shared_->preparation_uses.begin());
  HYLE_RETURN_IF_ERROR(use.write(shared_->writer));
  return use;
}
} // namespace hyle::graph
