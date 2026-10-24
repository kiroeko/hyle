#include "native_pending_regions.h"

#include <algorithm>

namespace hyle::graph::batch::detail {
namespace gd = graph::detail;
namespace {
Status validate_facts(const NativePendingRegionFacts &facts) noexcept {
  if (facts.inputs.empty() || facts.inputs.size() != facts.input_ports.size() ||
      !facts.source.table || !facts.source.table->destroy || !facts.source.table->reusable ||
      !facts.source.identity || !facts.source.create)
    return Status::FailedPrecondition("native pending reader/source capability is incomplete");
  for (std::size_t i = 0; i < facts.inputs.size(); ++i) {
    if (!facts.inputs[i].valid() || facts.inputs[i].kind() != ValueKind::kInput)
      return Status::InvalidArgument("native pending reader needs an external physical bank");
    for (std::size_t j = 0; j < i; ++j)
      if (facts.input_ports[i] == facts.input_ports[j])
        return Status::InvalidArgument("native pending reader repeats a physical input port");
  }
  return Status::Ok();
}
Status inspect(const NativePendingRegionRef &region) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto facts, region.table->inspect(region.prepared, region.invocation));
  HYLE_RETURN_IF_ERROR(validate_facts(facts));
  if (facts.inputs.size() != region.inputs.size() ||
      !std::ranges::equal(facts.inputs, region.inputs) ||
      !std::ranges::equal(facts.input_ports, region.input_ports) ||
      facts.source.table != region.source.table || facts.source.identity != region.source.identity ||
      facts.source.create != region.source.create)
    return Status::InvalidArgument("native pending selected port/source identity changed");
  return Status::Ok();
}
} // namespace

Status collect_native_pending_regions(Invocation &invocation, NativePendingRegions &regions) noexcept {
  regions.clear();
  return gd::RegionAccess::visit(invocation,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> Status {
        if (!view.prepared_vtable || !view.prepared_vtable->query_extension) return Status::Ok();
        const auto *table = static_cast<const NativePendingRegionVTable *>(
            view.prepared_vtable->query_extension(view.prepared, kNativePendingRegionExtensionId));
        if (!table) return Status::Ok();
        if (!view.invocation || !table->inspect || !table->validate_frame ||
            !table->commit_frame || !table->reset)
          return Status::FailedPrecondition("native pending selected slot capability is incomplete");
        HYLE_ASSIGN_OR_RETURN(auto facts, table->inspect(view.prepared, view.invocation));
        HYLE_RETURN_IF_ERROR(validate_facts(facts));
        auto &regions = *static_cast<NativePendingRegions *>(opaque);
        try {
          regions.push_back({table, view.prepared_vtable, view.prepared, view.invocation,
              view.step, view.lane, facts.inputs, facts.input_ports, facts.source});
        } catch (...) {
          return Status::ResourceExhausted("native pending original slot references");
        }
        return Status::Ok();
      }, &regions);
}
void reset_native_pending_regions(std::span<const NativePendingRegionRef> regions) noexcept {
  for (const auto &region : regions) region.table->reset(region.prepared, region.invocation);
}
Result<bool> native_pending_frame(std::span<const NativePendingRegionRef> regions, ValueId value,
    const compute::PagedArenaPendingRead &frame, const NativePendingSourceOwner &source,
    std::vector<gd::ProjectedReader, mem::StdAllocator<gd::ProjectedReader>> &readers,
    std::vector<ExecutionLaneId, mem::StdAllocator<ExecutionLaneId>> &lanes) noexcept {
  bool found = false;
  for (const auto &region : regions) {
    if (std::ranges::find(region.inputs, value) == region.inputs.end()) continue;
    HYLE_RETURN_IF_ERROR(inspect(region));
    if (!source.state || source.table != region.source.table || source.identity != region.source.identity)
      return std::unexpected(Status::InvalidArgument("native pending reader has a different source owner"));
    HYLE_RETURN_IF_ERROR(region.table->validate_frame(region.prepared, region.invocation, value, frame, source));
    try {
      for (std::size_t i = 0; i < region.inputs.size(); ++i)
        if (region.inputs[i] == value)
          readers.push_back({region.step, region.implementation, region.prepared, region.input_ports[i]});
      if (std::ranges::find(lanes, region.lane) == lanes.end()) lanes.push_back(region.lane);
    } catch (...) {
      return std::unexpected(Status::ResourceExhausted("native pending exact reader metadata"));
    }
    found = true;
  }
  return found;
}
void commit_native_pending_frame(std::span<const NativePendingRegionRef> regions, ValueId value,
    compute::PagedArenaPendingRead frame, const NativePendingSourceOwner &source) noexcept {
  for (const auto &region : regions)
    if (std::ranges::find(region.inputs, value) != region.inputs.end())
      region.table->commit_frame(region.prepared, region.invocation, value, frame, source);
}
} // namespace hyle::graph::batch::detail
