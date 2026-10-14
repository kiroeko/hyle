#include "hyle/compute/detail/copy_control.h"

#include <algorithm>
#include <functional>

#include "hyle/compute/detail/lease_access.h"

namespace hyle::compute::detail {
mem::AllocatorRef CopyMetadata::allocator() noexcept {
  return {[](void *opaque, std::size_t bytes, std::size_t alignment) noexcept -> void * {
    auto &self = *static_cast<CopyMetadata *>(opaque);
    auto *storage = self.base.allocate(bytes, alignment);
    if (storage) self.live.fetch_add(bytes, std::memory_order_relaxed);
    return storage;
  }, [](void *opaque, void *storage, std::size_t bytes, std::size_t alignment) noexcept {
    auto &self = *static_cast<CopyMetadata *>(opaque);
    if (storage) self.live.fetch_sub(bytes, std::memory_order_relaxed);
    self.base.deallocate(storage, bytes, alignment);
  }, [](void *opaque, void *storage, std::size_t old_bytes, std::size_t bytes, std::size_t alignment) noexcept -> void * {
    auto &self = *static_cast<CopyMetadata *>(opaque);
    auto *updated = self.base.reallocate(storage, old_bytes, bytes, alignment);
    if (updated || bytes == 0) {
      self.live.fetch_add(bytes, std::memory_order_relaxed);
      if (storage) self.live.fetch_sub(old_bytes, std::memory_order_relaxed);
    }
    return updated;
  }, this};
}
Status prepare_copy_control(std::span<const CopyBinding> bindings, std::span<const CopyPhase> phases,
    std::span<const ResolvedCopyRange> copies, CopyControlScratch scratch,
    std::span<const CopyResourceId> native_sources) noexcept {
  const auto n = bindings.size();
  const auto owned=[&](std::size_t id){return std::any_of(native_sources.begin(),native_sources.end(),
      [&](CopyResourceId source){return source.value==id;});};
  for(const auto id:native_sources)if(id.value>=n || !bindings[id.value].root ||
      bindings[id.value].read.valid() || bindings[id.value].write.valid())
    return Status::InvalidArgument("native result source requires its retained readonly owner");
  if (phases.size() > 64 || n > SIZE_MAX / (std::max)(phases.size(), std::size_t{1}) ||
      scratch.root_order.size() < n || scratch.root_leader.size() < n || scratch.last_write.size() < n ||
      scratch.roles.size() < n * phases.size() || scratch.native_ranges.size() < copies.size() ||
      scratch.written.size() < copies.size() || scratch.root_uses.size() / 2 < copies.size() ||
      scratch.producer_order.size() / 2 < copies.size() || scratch.phases.size() < phases.size())
    return Status::ResourceExhausted("copy control metadata capacity");
  auto order = scratch.root_order.first(n);
  for (std::size_t id = 0; id < n; ++id) order[id] = static_cast<std::uint32_t>(id);
  std::sort(order.begin(), order.end(), [bindings](auto left, auto right) noexcept {
    return std::less<const void *>{}(bindings[left].root, bindings[right].root);
  });
  std::uint32_t leader = UINT32_MAX;
  for (auto id : order) {
    if (leader == UINT32_MAX || bindings[id].root != bindings[leader].root) leader = id;
    scratch.root_leader[id] = leader;
  }
  std::fill(scratch.roles.begin(), scratch.roles.end(), static_cast<unsigned char>(0));
  std::fill(scratch.last_write.begin(), scratch.last_write.end(), UINT32_MAX);
  for (std::size_t index = 0; index < copies.size(); ++index) {
    const auto &copy = copies[index];
    if (copy.source >= n || copy.destination >= n || copy.phase >= phases.size())
      return Status::InvalidArgument("copy control range is not validated");
    const auto &source = bindings[copy.source], &destination = bindings[copy.destination];
    if (!source.root || !destination.root || !destination.write.valid())
      return Status::FailedPrecondition("copy control permission is missing");
    mem::ConstBufferView source_view;
    if(!owned(copy.source)){HYLE_ASSIGN_OR_RETURN(source_view,source.source.subview(copy.source_offset,copy.physical_bytes));}
    HYLE_ASSIGN_OR_RETURN(auto destination_view, destination.destination.subview(copy.destination_offset, copy.physical_bytes));
    scratch.native_ranges[index] = {destination_view, source_view};
    scratch.roles[copy.phase * n + scratch.root_leader[copy.source]] |= 1;
    scratch.roles[copy.phase * n + scratch.root_leader[copy.destination]] |= 2;
    if (source.producer.valid()) scratch.roles[copy.phase * n + copy.source] |= 4;
    else if(owned(copy.source)){
      // Several outputs of one move-only native result share one completion.
      // Every referencing phase checks that original observer, including a
      // phase which reads only a different output binding.
      bool producer=false;
      for(const auto id:native_sources)if(bindings[id.value].root==source.root && bindings[id.value].producer.valid()){
        scratch.roles[copy.phase*n+id.value]|=4;producer=true;break;
      }
      if(!producer)return Status::FailedPrecondition("native result source has no original producer");
    }
    if (destination.producer.valid()) scratch.roles[copy.phase * n + copy.destination] |= 4;
    scratch.last_write[scratch.root_leader[copy.destination]] = copy.phase;
    scratch.written[index] = {copy.destination, copy.phase, copy.destination_offset, copy.destination_offset + copy.logical_bytes};
  }
  auto written = scratch.written.first(copies.size());
  std::sort(written.begin(), written.end(), [](const auto &left, const auto &right) noexcept {
    if (left.binding != right.binding) return left.binding < right.binding;
    if (left.begin != right.begin) return left.begin < right.begin;
    return left.phase < right.phase;
  });
  std::size_t cursor = 0, use_count = 0, producer_count = 0;
  for (std::size_t phase = 0; phase < phases.size(); ++phase) {
    auto &plan = scratch.phases[phase];
    plan.first = cursor; plan.use_first = use_count; plan.producer_first = producer_count;
    plan.dependency_mask = phases[phase].dependency_mask;
    while (cursor < copies.size() && copies[cursor].phase == phase) ++cursor;
    plan.copies = cursor - plan.first;
    for (std::size_t id = 0; id < n; ++id) {
      const auto role = scratch.roles[phase * n + id];
      if (role & 4) scratch.producer_order[producer_count++] = static_cast<std::uint32_t>(id);
      if (!(role & 3)) continue;
      bool earlier = false;
      for (std::size_t preceding = 0; preceding < phase; ++preceding)
        earlier |= (scratch.roles[preceding * n + id] & 2) != 0;
      scratch.root_uses[use_count++] = {static_cast<std::uint32_t>(id), (role & 2) != 0,
                                       earlier || bindings[id].producer.valid()};
    }
    plan.use_count = use_count - plan.use_first; plan.producer_count = producer_count - plan.producer_first;
  }
  return Status::Ok();
}
Status admit_copy_control(LeaseUsePool &pool, std::span<const CopyBinding> bindings,
    std::span<const CopyPhasePlan> phases, std::span<const CopyRootUse> uses,
    std::span<LeaseUseBatch> tickets,std::span<const CopyResourceId> native_sources) noexcept {
  if (tickets.size() < phases.size()) return Status::ResourceExhausted("copy use ticket capacity");
  Status admission;
  for (std::size_t phase = 0; phase < phases.size() && admission.ok(); ++phase) {
    auto ticket = pool.begin();
    if (!ticket) { admission = ticket.error(); break; }
    tickets[phase] = std::move(*ticket);
    const auto &plan = phases[phase];
    if (plan.use_first > uses.size() || plan.use_count > uses.size() - plan.use_first) {
      admission = Status::InvalidArgument("copy use plan exceeds metadata"); break;
    }
    for (const auto &use : uses.subspan(plan.use_first, plan.use_count)) {
      if (use.binding >= bindings.size()) { admission = Status::InvalidArgument("copy use binding is invalid"); break; }
      const auto &binding = bindings[use.binding];
      if(!use.write && std::any_of(native_sources.begin(),native_sources.end(),
          [&](CopyResourceId id){return id.value==use.binding;})){
        if(!binding.root || binding.read.valid() || binding.write.valid()){
          admission=Status::InvalidArgument("native result source owner is invalid");break;
        }
        continue; // Original native phase retains immutable result storage.
      }
      admission = use.write ? LeaseAccess::write_unique(tickets[phase], binding.write, use.ordered)
          : binding.write.valid() ? LeaseAccess::ordered_read_unique(tickets[phase], binding.write)
                                  : LeaseAccess::read_unique(tickets[phase], binding.read);
      if (!admission.ok()) break;
    }
  }
  if (!admission.ok()) for (auto &ticket : tickets) ticket = {};
  return admission;
}
Status copy_output_coverage(std::span<const CopyWrittenSpan> written, CopyResourceId resource,
    CopyPhaseId phase, std::size_t offset, std::size_t bytes) noexcept {
  if (bytes > SIZE_MAX - offset) return Status::OutOfRange("copy output coordinate overflows");
  const auto end = offset + bytes;
  auto covered = offset;
  for (const auto &span : written) {
    if (covered >= end) break;
    if (span.binding != resource.value || span.phase > phase.value || span.end <= covered) continue;
    if (span.begin > covered) break;
    covered = (std::max)(covered, span.end);
  }
  return covered < end ? Status::OutOfRange("copy output includes uncopied logical bytes") : Status::Ok();
}
} // namespace hyle::compute::detail
