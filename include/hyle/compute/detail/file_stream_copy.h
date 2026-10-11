#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "hyle/compute/detail/file_stream_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::detail {

// Cold typed native adapter. Only the file-copy episode is erased at the
// composite boundary; Program/Invocation/Submission remain the original route
// types and the native implementation owns every actual accepted operation.
template<class Program, class Invocation, class Submission, class Access>
struct FileStreamCopyOwner final {
  struct Slot final { Invocation invocation; Submission submission; };
  FileStreamCopyOwner(mem::AllocatorRef selected, Program prepared)
      : allocator(selected), program(std::move(prepared)),
        specs(mem::StdAllocator<CopyResourceSpec>(selected)), slots(mem::StdAllocator<Slot>(selected)) {}
  mem::AllocatorRef allocator;
  Program program;
  std::vector<CopyResourceSpec, mem::StdAllocator<CopyResourceSpec>> specs;
  std::vector<Slot, mem::StdAllocator<Slot>> slots;
  static void destroy(void *opaque) noexcept {
    auto *self = static_cast<FileStreamCopyOwner *>(opaque); const auto allocator = self->allocator;
    std::destroy_at(self); allocator.deallocate(self, sizeof(FileStreamCopyOwner), alignof(FileStreamCopyOwner));
  }
  static Result<PhysicalCopyResource> physical(void *opaque, mem::ConstBufferView view, bool writable) noexcept {
    auto &self = *static_cast<FileStreamCopyOwner *>(opaque);
    return Access::physical(self.slots[0].invocation, view, writable);
  }
  CopyResourceId resource(mem::MemoryResourceRef resource, std::size_t bytes) const noexcept {
    // Zero is always staging. Target identity may be the same system resource
    // on CPU, but source/destination bindings are still distinct permissions.
    for (std::uint32_t index = 1; index < specs.size(); ++index)
      if (specs[index].resource == resource && bytes <= specs[index].capacity_bytes) return {index};
    return {};
  }
  static StreamCopyResult copy(void *opaque, std::uint32_t index, ReadLease &source,
      WriteLease *mutable_source, WriteLease &destination, Completion *producer,
      std::size_t source_offset, std::size_t destination_offset, std::size_t bytes,
      std::size_t physical_bytes, bool source_stage) noexcept {
    auto &self = *static_cast<FileStreamCopyOwner *>(opaque); auto &slot = self.slots[index];
    StreamCopyResult result;
    bool prepared = false;
    const auto perform = [&]() noexcept -> Status {
      HYLE_RETURN_IF_ERROR(slot.invocation.reset());
      mem::ConstBufferView source_view;
      if (mutable_source) { HYLE_ASSIGN_OR_RETURN(source_view, LeaseAccess::ordered_view(*mutable_source)); }
      else { HYLE_ASSIGN_OR_RETURN(source_view, source.view()); }
      HYLE_ASSIGN_OR_RETURN(auto destination_view, LeaseAccess::ordered_view(destination));
      CopyResourceId source_id = source_stage ? CopyResourceId{0} : self.resource(source_view.resource(), source_view.size());
      CopyResourceId destination_id = source_stage ? self.resource(destination_view.resource(), destination_view.size()) : CopyResourceId{0};
      if (!source_id.valid() || !destination_id.valid()) return Status::InvalidArgument("stream copy has no prepared resource");
      if (mutable_source) HYLE_RETURN_IF_ERROR(slot.invocation.bind_mutable(source_id, std::move(*mutable_source)));
      else HYLE_RETURN_IF_ERROR(slot.invocation.bind_source(source_id, source));
      HYLE_RETURN_IF_ERROR(slot.invocation.bind_mutable(destination_id, std::move(destination)));
      if (producer) HYLE_RETURN_IF_ERROR(slot.invocation.bind_producer(source_id, std::move(*producer)));
      const std::array phases{CopyPhase{}};
      const std::array ranges{CopyRange{source_id, destination_id, {0}, source_offset, destination_offset, bytes, physical_bytes}};
      HYLE_RETURN_IF_ERROR(slot.invocation.prepare_replay(phases, ranges));
      prepared = true;
      HYLE_ASSIGN_OR_RETURN(slot.submission, slot.invocation.submit());
      HYLE_RETURN_IF_ERROR(slot.submission.wait());
      slot.submission = {};
      HYLE_ASSIGN_OR_RETURN(destination, Access::take_mutable(slot.invocation, destination_id));
      if (mutable_source) {
        HYLE_ASSIGN_OR_RETURN(*mutable_source, Access::take_mutable(slot.invocation, source_id));
        HYLE_ASSIGN_OR_RETURN(source, mutable_source->publish());
      }
      HYLE_RETURN_IF_ERROR(slot.invocation.reset()); return Status::Ok();
    };
    result.status = perform(); if (prepared) result.report = slot.invocation.report();
    result.retirement_proven = !slot.invocation.in_flight();
    if (!result.status.ok() && result.retirement_proven) {
      slot.submission = {}; slot.invocation.reset().IgnoreError();
    }
    return result;
  }
  static Status observe(void *opaque, std::uint32_t index) noexcept {
    return Access::observe(static_cast<FileStreamCopyOwner *>(opaque)->slots[index].invocation);
  }
  static bool in_flight(void *opaque, std::uint32_t index) noexcept {
    return static_cast<FileStreamCopyOwner *>(opaque)->slots[index].invocation.in_flight();
  }
  static Status reset(void *opaque, std::uint32_t index) noexcept {
    auto &slot = static_cast<FileStreamCopyOwner *>(opaque)->slots[index];
    if (slot.invocation.in_flight()) return Status::Unavailable("stream copy native use is pending");
    slot.submission = {}; return slot.invocation.reset();
  }
  static CopyBatchReport report(void *opaque) noexcept {
    auto &self = *static_cast<FileStreamCopyOwner *>(opaque); auto result = self.slots[0].invocation.report();
    result.metadata_bytes += sizeof(self) + self.slots.capacity() * sizeof(Slot) + self.specs.capacity() * sizeof(CopyResourceSpec);
    return result;
  }
};

template<class Program, class Invocation, class Submission, class Access, class Create>
Result<StreamCopyEndpoint> make_file_copy_endpoint(Program program,
    std::span<const CopyResourceSpec> resources, FileStreamOptions options, Create create) noexcept {
  using Owner = FileStreamCopyOwner<Program, Invocation, Submission, Access>;
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) { HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator()); }
  auto *storage = allocator.allocate(sizeof(Owner), alignof(Owner));
  if (!storage) return std::unexpected(Status::ResourceExhausted("stream copy adapter metadata"));
  Owner *owner = nullptr;
  try { owner = std::construct_at(static_cast<Owner *>(storage), allocator, std::move(program)); }
  catch (...) { allocator.deallocate(storage, sizeof(Owner), alignof(Owner)); return std::unexpected(Status::ResourceExhausted("stream copy adapter metadata")); }
  StreamCopyEndpoint result; result.owner = owner; result.destroy = &Owner::destroy;
  try { owner->specs.assign(resources.begin(), resources.end()); owner->slots.resize(options.in_flight_slots); }
  catch (...) { return std::unexpected(Status::ResourceExhausted("stream copy adapter fixed slots")); }
  for (auto &slot : owner->slots) { HYLE_ASSIGN_OR_RETURN(slot.invocation, create(owner->program)); }
  result.physical = &Owner::physical; result.copy = &Owner::copy; result.observe = &Owner::observe;
  result.in_flight = &Owner::in_flight; result.reset = &Owner::reset; result.report = &Owner::report;
  return result;
}
} // namespace hyle::compute::detail
