#include "hyle/graph/executable.h"

#include <algorithm>
#include <functional>

#include "plan_state.h"

namespace hyle::graph {
Result<InvocationResourceReport> Invocation::resource_report() const noexcept {
  if (!state_ || state_->closed)
    return std::unexpected(Status::FailedPrecondition("invalid or closed graph invocation"));
  const auto &plan = *state_->plan;
  if (!plan.invocation_storage || !plan.invocation_storage->resource_budget())
    return std::unexpected(Status::FailedPrecondition("graph has no shared resource budget"));
  const auto &budget = plan.invocation_storage->resource_budget();
  InvocationResourceReport result;
  try { result.storage.resize(plan.storage_domains.size()); }
  catch (...) { return std::unexpected(Status::ResourceExhausted("resource report metadata")); }
  {
    // Temporary metadata dies before the usage snapshot. Merge disjoint and
    // overlapping aliases of the same allocation without counting gaps twice.
    container::SmallVector<mem::ConstBufferView, 16, mem::AllocatorRef> ranges(state_->allocator);
    for (std::size_t i = 0; i < plan.instance_roles.size(); ++i) {
      if (!(plan.instance_roles[i] & (detail::kExternalInput | detail::kExternalOutput | detail::kConstant)))
        continue;
      bool owned = false;
      for (std::size_t j = 0; j < plan.constant_ids.size(); ++j)
        if (plan.instance_values[i] == plan.constant_ids[j].ordinal() && plan.owned_constant(j))
          owned = true;
      if (!owned && state_->readable[i].valid() && !ranges.try_push_back(state_->readable[i].storage()))
        return std::unexpected(Status::ResourceExhausted("resource report scratch"));
    }
    const std::less<const void *> less;
    std::sort(ranges.begin(), ranges.end(), [&](auto a, auto b) noexcept {
      if (a.resource() != b.resource()) return less(&a.resource().resource(), &b.resource().resource());
      if (a.native_handle() != b.native_handle()) return less(a.native_handle(), b.native_handle());
      return a.offset() < b.offset();
    });
    for (std::size_t i = 0; i < ranges.size();) {
      const auto first = ranges[i++];
      if (first.size() > SIZE_MAX - first.offset())
        return std::unexpected(Status::ResourceExhausted("boundary range end overflows"));
      std::size_t covered = first.size(), end = first.offset() + first.size();
      while (i < ranges.size() && ranges[i].resource() == first.resource() &&
             ranges[i].native_handle() == first.native_handle()) {
        const auto next = ranges[i++];
        if (next.size() > SIZE_MAX - next.offset())
          return std::unexpected(Status::ResourceExhausted("boundary range end overflows"));
        const auto next_end = next.offset() + next.size();
        if (next_end > end) {
          const auto extra = next_end - (std::max)(end, next.offset());
          if (extra > SIZE_MAX - covered)
            return std::unexpected(Status::ResourceExhausted("boundary range sum overflows"));
          covered += extra;
          end = next_end;
        }
      }
      for (std::size_t domain = 0; domain < result.storage.size(); ++domain) {
        if (plan.storage_domains[domain].resource != first.resource()) continue;
        auto &bytes = result.storage[domain].boundary_view_bytes;
        if (covered > SIZE_MAX - bytes)
          return std::unexpected(Status::ResourceExhausted("boundary report sum overflows"));
        bytes += covered;
      }
    }
  }
  for (std::size_t i = 0; i < result.storage.size(); ++i) {
    result.storage[i].resource = plan.storage_domains[i].resource;
    HYLE_ASSIGN_OR_RETURN(result.storage[i].usage, budget.usage(result.storage[i].resource));
  }
  for (std::size_t i = 0; i < budget.resource_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto resource, budget.resource(i));
    if (std::any_of(result.storage.begin(), result.storage.end(),
        [&](const auto &domain) noexcept { return domain.resource == resource; })) continue;
    HYLE_ASSIGN_OR_RETURN(auto usage, budget.usage(resource));
    if (usage.live_bytes > SIZE_MAX - result.other_live_bytes)
      return std::unexpected(Status::ResourceExhausted("resource report sum overflows"));
    ++result.other_resource_count;
    result.other_live_bytes += usage.live_bytes;
  }
  result.unattributed = budget.unattributed_usage();
  result.has_unknown_graph_storage_history = plan.invocation_storage->usage().has_unknown_storage ||
      (plan.compilation_storage && plan.compilation_storage->usage().has_unknown_storage);
  return result;
}
} // namespace hyle::graph
