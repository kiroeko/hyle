#include "hyle/compute/detail/file_stream_access.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/compute/detail/lease_access.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/join.h"
#include "hyle/sched/sync.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"
#include "../sched/fiber_access.h"

namespace hyle::compute::detail {
template<class T> using StreamArray = std::vector<T, mem::StdAllocator<T>>;
enum class GroupPhase : unsigned char { kQueued, kWorking, kReady, kFailed };
struct StreamRangeCell final {
  WriteLease writer;
  ReadLease reader, output;
  Completion producer;
  std::uint64_t file_offset = 0;
  std::size_t offset = 0, bytes = 0, physical_bytes = 0;
  LeaseBackingInfo backing;
  PhysicalCopyResource physical;
};
struct StreamGroupCell final {
  GroupPhase phase = GroupPhase::kQueued;
  std::uint64_t tag = 0;
  std::uint32_t count = 0, handles = 0;
  bool delivered = false;
};
struct StreamRequestCell final {
  StreamSource source;
  StreamFile sink;
  sched::CancelToken cancel;
  io::OperationOptions io_options;
  StreamResult result;
  std::uint64_t sequence = 0;
  std::uint32_t remaining = 0;
  bool live = false, ticket = false, write = false;
};
struct StreamSlotCell final {
  FileStreamState *owner = nullptr;
  std::uint32_t index = 0, group = 0;
  mem::FiberStack prepared_stack;
  std::optional<sched::Fiber> fiber;
  WriteLease writer;
  void *address = nullptr;
  PhysicalCopyResource physical;
  bool working = false, quarantined = false;
  std::uint32_t request = 0;
  sched::detail::BlockingJob job;
  StreamCopyResult copy_result;
  const mem::FileMapping *job_mapping = nullptr;
  mem::BufferView job_stage;
  ReadLease *job_reader = nullptr;
  WriteLease *job_source = nullptr, *job_destination = nullptr;
  Completion *job_producer = nullptr;
  std::uint64_t job_file_offset = 0;
  std::size_t job_source_offset = 0, job_destination_offset = 0, job_bytes = 0, job_physical = 0;
  bool job_upload = false, job_done = true;
};
struct StreamInterval final {
  std::uintptr_t space = 0;
  std::uint64_t backing = 0, begin = 0, end = 0;
  bool write = false;
};
struct StreamAdmission final {
  LeaseBackingInfo backing;
  PhysicalCopyResource physical;
};

struct FileStreamState final : util::RefCounted<FileStreamState> {
  FileStreamState(io::IoContext &context, FileStreamOptions selected, mem::AllocatorRef metadata)
      : io(context), options(std::move(selected)), allocator(metadata),
        stacks(sched::kDefaultStackSize, options.in_flight_slots),
        specs(mem::StdAllocator<CopyResourceSpec>(metadata)),
        requests(mem::StdAllocator<StreamRequestCell>(metadata)),
        groups(mem::StdAllocator<StreamGroupCell>(metadata)),
        ranges(mem::StdAllocator<StreamRangeCell>(metadata)),
        slots(mem::StdAllocator<StreamSlotCell>(metadata)),
        admissions(mem::StdAllocator<StreamAdmission>(metadata)),
        intervals(mem::StdAllocator<StreamInterval>(metadata)),
        backings(mem::StdAllocator<LeaseBackingInfo>(metadata)) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this); selected.deallocate(this, sizeof(FileStreamState), alignof(FileStreamState));
  }
  StreamGroupCell &group(std::uint32_t request, std::uint32_t index) noexcept {
    return groups[static_cast<std::size_t>(request) * options.max_groups_per_request + index];
  }
  StreamRangeCell &range(std::uint32_t request, std::uint32_t group, std::uint32_t index) noexcept {
    return ranges[(static_cast<std::size_t>(request) * options.max_groups_per_request + group)
                  * options.max_ranges_per_group + index];
  }
  io::IoContext &io;
  FileStreamOptions options;
  mem::AllocatorRef allocator;
  StreamCopyEndpoint endpoint;
  LeasePool staging;
  mem::FiberStackPool stacks;
  LeaseBackingInfo stage_backing;
  StreamArray<CopyResourceSpec> specs;
  StreamArray<StreamRequestCell> requests;
  StreamArray<StreamGroupCell> groups;
  StreamArray<StreamRangeCell> ranges;
  StreamArray<StreamSlotCell> slots;
  StreamArray<StreamAdmission> admissions;
  StreamArray<StreamInterval> intervals;
  StreamArray<LeaseBackingInfo> backings;
  util::SpinLock lock;
  sched::ConditionVariable changed;
  FileStreamReport report;
  std::uint64_t sequence = 0;
  std::uint32_t worker_count = 0;
  bool initialized = false, admission_closed = false, closing = false;
};

void release_state(FileStreamState *state) noexcept {
  bool last;
  { std::lock_guard guard(state->lock); last = state->is_unique(); if (!last) state->decref(); }
  if (last) state->decref();
}

StreamCopyEndpoint::StreamCopyEndpoint(StreamCopyEndpoint &&other) noexcept { *this = std::move(other); }
StreamCopyEndpoint &StreamCopyEndpoint::operator=(StreamCopyEndpoint &&other) noexcept {
  if (this != &other) {
    if (owner) destroy(owner);
    owner = std::exchange(other.owner, nullptr); destroy = other.destroy; physical = other.physical;
    copy = other.copy; observe = other.observe; in_flight = other.in_flight;
    reset = other.reset; report = other.report;
  } return *this;
}
StreamCopyEndpoint::~StreamCopyEndpoint() { if (owner) destroy(owner); }

namespace {
bool multiply(std::size_t left, std::size_t right, std::size_t &result) noexcept {
  if (right && left > SIZE_MAX / right) return false;
  result = left * right; return true;
}
Status check_options(const FileStreamOptions &options) noexcept {
  if (!options.allocator.valid() || !options.max_requests || !options.max_groups_per_request ||
      !options.max_ranges_per_group || !options.in_flight_slots ||
      options.in_flight_slots > mem::FiberStackPool::kMaxCacheCap || !options.staging_bytes_per_slot ||
      !options.staging_alignment || (options.staging_alignment & (options.staging_alignment - 1)) ||
      !options.max_active_backing_bytes || options.staging_bytes_per_slot % options.staging_alignment)
    return Status::InvalidArgument("invalid file stream capacities/allocator/alignment");
  std::size_t groups, ranges, staging;
  if (!multiply(options.max_requests, options.max_groups_per_request, groups) ||
      !multiply(groups, options.max_ranges_per_group, ranges) ||
      !multiply(options.in_flight_slots, options.staging_bytes_per_slot, staging) ||
      ranges > (SIZE_MAX - 1) / 2 || options.in_flight_slots == UINT32_MAX)
    return Status::OutOfRange("file stream capacity overflow");
  if (staging > options.max_active_backing_bytes)
    return Status::ResourceExhausted("file stream stage exceeds backing budget");
  return Status::Ok();
}
Status check_file(FileStreamState &state, io::File *file) noexcept {
  if (!file || !file->valid() || io::detail::io_context_identity(*file) != &state.io)
    return Status::InvalidArgument("stream file belongs to another or invalid IoContext");
  return Status::Ok();
}
Status check_range(FileStreamState &state, mem::ConstBufferView view, bool writable,
    std::uint64_t file_offset, std::uint64_t file_capacity, std::size_t offset,
    std::size_t bytes, std::size_t physical_bytes, LeaseBackingInfo backing,
    io::File *file, const mem::FileMapping *mapping, StreamAdmission &admission) noexcept {
  const auto physical = physical_bytes ? physical_bytes : bytes;
  if (physical < bytes || (!bytes && physical) || offset > view.size() || physical > view.size() - offset ||
      file_offset > file_capacity || bytes > file_capacity - file_offset)
    return Status::OutOfRange("stream file/resource range or explicit padding exceeds capacity");
  bool known = false;
  for (const auto &spec : state.specs)
    if (view.resource() == spec.resource && view.size() <= spec.capacity_bytes) { known = true; break; }
  if (!known) return Status::InvalidArgument("stream resource has no prepared route/capacity");
  HYLE_ASSIGN_OR_RETURN(admission.physical, state.endpoint.physical(state.endpoint.owner, view, writable));
  const auto &geometry = admission.physical;
  const auto alignment = geometry.copy_alignment;
  if (!alignment || offset > geometry.capacity || physical > geometry.capacity - offset ||
      geometry.origin > SIZE_MAX - offset || geometry.origin + offset > SIZE_MAX - physical ||
      (bytes && (offset % alignment || physical % alignment || state.options.staging_bytes_per_slot % alignment)))
    return Status::InvalidArgument("stream native range/chunk alignment is unsupported");
  if (physical > bytes && (physical - bytes >= state.options.staging_bytes_per_slot ||
      (bytes ? (bytes - 1) % state.options.staging_bytes_per_slot + 1 : 0) + physical - bytes > state.options.staging_bytes_per_slot))
    return Status::InvalidArgument("stream tail padding does not fit its staging slot");
  for (const auto &slot : state.slots) {
    const auto &stage = slot.physical;
    if (geometry.address_space == stage.address_space && geometry.backing == stage.backing &&
        geometry.origin + offset < stage.origin + stage.capacity &&
        stage.origin < geometry.origin + offset + physical && physical)
      return Status::InvalidArgument("stream target aliases its staging storage");
    if (file && file->cache_mode() == io::FileCacheMode::kDirect && bytes) {
      const auto io_alignment = file->direct_alignment();
      const auto first = (std::min)(bytes, state.options.staging_bytes_per_slot);
      if (!io_alignment.accepts(slot.address, file_offset, first) ||
          bytes % io_alignment.length_alignment ||
          state.options.staging_bytes_per_slot % io_alignment.offset_alignment)
        return Status::InvalidArgument("stream direct IO chunk/tail alignment is unsupported");
    }
  }
  if (mapping && !mapping->contains(static_cast<std::size_t>(file_offset), bytes))
    return Status::OutOfRange("stream mapping range crosses an unmapped hole");
  admission.backing = backing;
  return Status::Ok();
}
void add_interval(FileStreamState &state, const PhysicalCopyResource &physical,
                  std::size_t offset, std::size_t bytes, bool write) {
  if (!bytes) return;
  state.intervals.push_back({physical.address_space, physical.backing,
                            physical.origin + offset, physical.origin + offset + bytes, write});
}
const char kFileCoordinates = 0;
void add_file_interval(FileStreamState &state, io::File *file, std::uint64_t offset,
                       std::size_t bytes) {
  if (!bytes) return;
  state.intervals.push_back({reinterpret_cast<std::uintptr_t>(&kFileCoordinates),
      static_cast<std::uint64_t>(file->native_handle()), offset, offset + bytes, true});
}
Status interval_hazards(FileStreamState &state) noexcept {
  std::sort(state.intervals.begin(), state.intervals.end(), [](const auto &a, const auto &b) {
    if (a.space != b.space) return a.space < b.space;
    if (a.backing != b.backing) return a.backing < b.backing;
    return a.begin < b.begin;
  });
  std::uintptr_t space = 0; std::uint64_t backing = 0, read_end = 0, write_end = 0;
  bool first = true;
  for (const auto &span : state.intervals) {
    if (first || span.space != space || span.backing != backing) {
      first = false; space = span.space; backing = span.backing; read_end = write_end = 0;
    }
    if (span.begin < write_end || (span.write && span.begin < read_end))
      return Status::InvalidArgument("stream ranges overlap a live write/file range");
    auto &end = span.write ? write_end : read_end; end = (std::max)(end, span.end);
  }
  return Status::Ok();
}
std::size_t backing_usage(FileStreamState &state) noexcept {
  std::sort(state.backings.begin(), state.backings.end(), [](const auto &a, const auto &b) {
    return std::less<const void *>{}(a.identity, b.identity);
  });
  std::size_t bytes = 0; const void *prior = nullptr;
  for (const auto &backing : state.backings) {
    if (backing.identity == prior) continue;
    prior = backing.identity;
    if (backing.bytes > SIZE_MAX - bytes) return SIZE_MAX;
    bytes += backing.bytes;
  }
  return bytes;
}
void existing_accounting(FileStreamState &state, bool hazards) {
  state.backings.clear(); state.intervals.clear();
  state.backings.push_back(state.stage_backing);
  for (std::uint32_t r = 0; r < state.requests.size(); ++r) {
    const auto &request = state.requests[r]; if (!request.live) continue;
    for (std::uint32_t g = 0; g < request.result.total_groups; ++g) {
      const auto &group = state.group(r, g);
      for (std::uint32_t i = 0; i < group.count; ++i) {
        const auto &range = state.range(r, g, i);
        if (range.writer.valid() || range.reader.valid() || range.output.valid() || !request.result.retirement_proven)
          state.backings.push_back(range.backing);
        if (hazards) {
          add_interval(state, range.physical, range.offset, range.physical_bytes,
                       !request.write && group.phase != GroupPhase::kReady);
          if (request.write) add_file_interval(state, FileStreamAccess::file(request.sink), range.file_offset, range.bytes);
        }
      }
    }
  }
}
bool readers_returned(FileStreamState &state, std::uint32_t r) noexcept {
  const auto &request = state.requests[r];
  for (std::uint32_t g = 0; g < request.result.total_groups; ++g) {
    const auto &group = state.group(r, g);
    if (group.handles) return false;
    for (std::uint32_t i = 0; i < group.count; ++i) {
      const auto &range = state.range(r, g, i);
      // Write-back sources were caller-owned before admission. Their external
      // references do not prevent release of this request's read permission.
      if (!request.write && range.output.valid() && !LeaseAccess::unique_read_owner(range.output)) return false;
    }
  }
  return true;
}
void reclaim_locked(FileStreamState &state) noexcept {
  for (std::uint32_t r = 0; r < state.requests.size(); ++r) {
    auto &request = state.requests[r];
    if (!request.live || request.ticket || !request.result.complete || !request.result.retirement_proven ||
        !readers_returned(state, r)) continue;
    for (std::uint32_t g = 0; g < request.result.total_groups; ++g) {
      auto &group = state.group(r, g);
      for (std::uint32_t i = 0; i < group.count; ++i) state.range(r, g, i) = {};
      group = {};
    }
    request = {}; --state.report.live_requests;
  }
  existing_accounting(state, false);
  state.report.active_backing_bytes = backing_usage(state);
  state.report.external_backing_bytes = state.report.active_backing_bytes - state.stage_backing.bytes;
}
bool choose_group(FileStreamState &state, std::uint32_t &r, std::uint32_t &g) noexcept {
  auto sequence = UINT64_MAX; bool found = false;
  for (std::uint32_t index = 0; index < state.requests.size(); ++index) {
    const auto &request = state.requests[index]; if (!request.live || request.sequence >= sequence) continue;
    for (std::uint32_t group = 0; group < request.result.total_groups; ++group) {
      if (state.group(index, group).phase == GroupPhase::kQueued) {
        sequence = request.sequence; r = index; g = group; found = true; break;
      }
    }
  }
  return found;
}
Status checkpoint(const StreamRequestCell &request) noexcept {
  if (request.cancel.is_cancelled()) return Status::Cancelled("file stream request cancelled");
  if (request.io_options.deadline && time::remaining(*request.io_options.deadline) <= time::Duration::zero())
    return Status::DeadlineExceeded("file stream request deadline expired");
  return Status::Ok();
}
void count_io(FileStreamState &state, StreamRequestCell &request, std::size_t bytes, bool write) noexcept {
  std::lock_guard guard(state.lock); ++state.report.io_calls;
  if (write) request.result.written_bytes += bytes; else request.result.read_bytes += bytes;
}

// Original BlockingPool jobs are destruction-owned, so publish after its
// destroy boundary instead of using a JoinHandle's earlier result boundary.
// One embedded cell per fixed slot avoids per-chunk heap jobs and survives OOM.
void run_slot_job(void *opaque) noexcept {
  auto &slot = *static_cast<StreamSlotCell *>(opaque); auto &state = *slot.owner;
  if (slot.job_mapping) {
    const auto fill = [&]() noexcept -> Status {
      HYLE_ASSIGN_OR_RETURN(auto source, slot.job_mapping->bytes(static_cast<std::size_t>(slot.job_file_offset), slot.job_bytes));
      HYLE_ASSIGN_OR_RETURN(auto destination, slot.job_stage.subview(0, slot.job_bytes)); return destination.write(source);
    };
    slot.copy_result = {}; slot.copy_result.status = fill();
  } else {
    slot.copy_result = state.endpoint.copy(state.endpoint.owner, slot.index, *slot.job_reader,
        slot.job_source, *slot.job_destination, slot.job_producer, slot.job_source_offset,
        slot.job_destination_offset, slot.job_bytes, slot.job_physical, slot.job_upload);
  }
}
void cancel_slot_job(void *opaque) noexcept {
  auto &slot = *static_cast<StreamSlotCell *>(opaque); slot.copy_result = {};
  slot.copy_result.status = Status::Cancelled("file stream blocking runtime is shutting down");
}
void finish_slot_job(void *opaque) noexcept {
  auto &slot = *static_cast<StreamSlotCell *>(opaque); auto &state = *slot.owner;
  std::lock_guard guard(state.lock); slot.job_done = true; state.changed.notify_all();
}
void await_slot_job(FileStreamState &state, StreamSlotCell &slot) noexcept {
  slot.job_done = false;
  auto &pool = io::detail::IoContextAccess::blocking_pool(state.io);
  if (!pool.submit(&slot.job)) { slot.job.cancel(); slot.job.destroy(); }
  state.lock.lock(); state.changed.wait(state.lock, [&] { return slot.job_done; }); state.lock.unlock();
}

// IO waits suspend the existing fiber. Only native waits and mapping page
// faults leave it through IoContext's original BlockingPool.
sched::Task<Status> process_group(mem::AllocatorRef allocator, FileStreamState *state,
    std::uint32_t slot_index, std::uint32_t r, std::uint32_t g) noexcept {
  (void)allocator;
  auto &slot = state->slots[slot_index]; auto &request = state->requests[r];
  auto &group = state->group(r, g);
  for (std::uint32_t i = 0; i < group.count; ++i) {
    auto &range = state->range(r, g, i);
    for (std::size_t offset = 0; offset < range.bytes;) {
      auto checked = checkpoint(request); if (!checked.ok()) co_return checked;
      const auto bytes = (std::min)(range.bytes - offset, state->options.staging_bytes_per_slot);
      const auto physical = bytes + (offset + bytes == range.bytes ? range.physical_bytes - range.bytes : 0);
      if (!slot.writer.valid()) {
        auto allocated = state->staging.allocate(state->options.staging_bytes_per_slot, state->options.staging_alignment);
        if (!allocated) co_return allocated.error();
        slot.writer = std::move(*allocated);
      }
      auto stage_view = slot.writer.view(); if (!stage_view) co_return stage_view.error();
      if (!request.write) {
        if (auto *mapping = FileStreamAccess::mapping(request.source)) {
          slot.job_mapping = mapping; slot.job_stage = *stage_view; slot.job_bytes = bytes;
          slot.job_file_offset = range.file_offset + offset; await_slot_job(*state, slot);
          if (!slot.copy_result.status.ok()) co_return slot.copy_result.status;
          { std::lock_guard guard(state->lock); request.result.read_bytes += bytes; state->report.mapping_copy_bytes += bytes; }
        } else {
          auto mapped = stage_view->map_host(); if (!mapped) co_return mapped.error();
          auto buffer = mapped->writable_bytes(); if (!buffer) co_return buffer.error();
          auto *file = FileStreamAccess::file(request.source);
          // File may use the original blocking fallback (kqueue/epoll or an
          // adopted handle). Cancellation/deadline stops later chunks here;
          // an accepted chunk retires before its stage/source can be released.
          // Do not ask IO to pretend an in-kernel read is natively cancellable.
          auto task = io::read_exact_at(state->allocator, *file, range.file_offset + offset, buffer->first(bytes));
          if (!task.valid()) co_return Status::ResourceExhausted("file stream read coroutine");
          auto transferred = co_await task; count_io(*state, request, transferred.bytes, false);
          if (!transferred.status.ok()) co_return transferred.status;
          if (transferred.bytes != bytes) co_return Status::OutOfRange("file stream source reached EOF");
          auto closed = mapped->close(); if (!closed.ok()) co_return closed;
        }
        if (physical > bytes) {
          auto padding = stage_view->subview(bytes, physical - bytes); if (!padding) co_return padding.error();
          auto filled = padding->fill(std::byte{0}); if (!filled.ok()) co_return filled;
        }
      }
      checked = checkpoint(request); if (!checked.ok()) co_return checked;
      ReadLease stage;
      if (!request.write) {
        auto ready = slot.writer.publish(); if (!ready) co_return ready.error(); stage = std::move(*ready);
      }
      slot.job_mapping = nullptr; slot.job_reader = request.write ? &range.reader : &stage;
      slot.job_source = request.write && range.writer.valid() ? &range.writer : nullptr;
      slot.job_destination = request.write ? &slot.writer : &range.writer;
      slot.job_producer = request.write && range.producer.valid() ? &range.producer : nullptr;
      slot.job_source_offset = request.write ? range.offset + offset : 0;
      slot.job_destination_offset = request.write ? 0 : range.offset + offset;
      slot.job_bytes = bytes; slot.job_physical = physical; slot.job_upload = !request.write;
      await_slot_job(*state, slot); const auto *copied = &slot.copy_result;
      {
        std::lock_guard guard(state->lock);
        state->report.native_calls += copied->report.native_calls; state->report.native_waits += copied->report.native_waits;
        state->report.phase_completions += copied->report.phase_completions; state->report.host_waits += copied->report.host_waits;
        if (!copied->retirement_proven) { slot.quarantined = true; slot.request = r; request.result.retirement_proven = false; }
        if (copied->status.ok()) {
          request.result.copied_bytes += bytes;
          state->report.copied_physical_bytes += copied->report.physical_bytes;
        }
      }
      if (!copied->status.ok()) co_return copied->status;
      if (!request.write) {
        auto mutable_stage = stage.try_write(); if (!mutable_stage) co_return mutable_stage.error();
        slot.writer = std::move(*mutable_stage);
      } else {
        auto mapped = slot.writer.view()->map_host(mem::HostAccess::kRead); if (!mapped) co_return mapped.error();
        auto *file = FileStreamAccess::file(request.sink);
        auto task = io::write_all_at(state->allocator, *file, range.file_offset + offset, mapped->bytes().first(bytes));
        if (!task.valid()) co_return Status::ResourceExhausted("file stream write coroutine");
        auto transferred = co_await task; count_io(*state, request, transferred.bytes, true);
        if (!transferred.status.ok()) co_return transferred.status;
        auto closed = mapped->close(); if (!closed.ok()) co_return closed;
      }
      offset += bytes;
    }
  }
  // Publish the complete group together. Every range's accepted writes already
  // retired; a partial/failed group never exposes a prefix as a ready tensor.
  for (std::uint32_t i = 0; i < group.count; ++i) {
    auto &range = state->range(r, g, i);
    if (range.writer.valid()) {
      auto ready = range.writer.publish(); if (!ready) co_return ready.error(); range.reader = std::move(*ready);
    }
    if (!range.reader.valid() && !range.bytes) continue;
    auto output = range.reader.subview(range.offset, range.bytes); if (!output) co_return output.error();
    range.output = std::move(*output); range.reader.reset();
  }
  co_return Status::Ok();
}

void kick_locked(FileStreamState &state) noexcept;
void stream_worker(void *opaque) noexcept {
  auto &slot = *static_cast<StreamSlotCell *>(opaque);
  auto &state = *slot.owner;
  const auto r = slot.request, g = slot.group;
  auto work = process_group(state.allocator, &state, slot.index, r, g);
    Status terminal = Status::ResourceExhausted("file stream group coroutine metadata");
    if (work.valid()) {
      while (!work.done()) work.resume();
      terminal = work.result();
    }
    work = {}; // no child frame/borrow survives publication
    state.lock.lock();
    auto &request = state.requests[r]; auto &group = state.group(r, g);
    group.phase = terminal.ok() ? GroupPhase::kReady : GroupPhase::kFailed;
    if (terminal.ok()) ++request.result.ready_groups;
    else {
      request.result.status.Update(terminal);
      // Accepted independent groups finish on their own. Only unstarted work
      // is withdrawn; earlier successful results keep their own permissions.
      for (std::uint32_t index = 0; index < request.result.total_groups; ++index)
        if (state.group(r, index).phase == GroupPhase::kQueued) {
          state.group(r, index).phase = GroupPhase::kFailed; --request.remaining;
        }
    }
    --request.remaining; request.result.complete = request.remaining == 0;
    state.changed.notify_all(); state.lock.unlock();
}
// Original Scheduler finish hook, called after leaving the Fiber stack. Destroy
// the Fiber/stack before acknowledging drain. All state reference drops share
// its monitor, so a last worker can destroy an abandoned owner outside the lock.
void stream_worker_finished(void *opaque) noexcept {
  auto &slot = *static_cast<StreamSlotCell *>(opaque); auto *state = slot.owner;
  slot.fiber.reset(); state->lock.lock(); --state->worker_count;
  slot.working = false; --state->report.active_slots;
  kick_locked(*state);
  const bool last = state->is_unique();
  if (!last) state->decref();
  state->changed.notify_all(); state->lock.unlock();
  if (last) state->decref();
}
// Each File is affined to its original IO worker. A fresh Fiber for a group
// uses the same fixed slot and preallocated stack; no started Fiber is moved or
// re-affined, and mapping groups need no file-ring identity.
void kick_locked(FileStreamState &state) noexcept {
  if (!state.initialized) return;
  auto &scheduler = io::detail::IoContextAccess::scheduler(state.io);
  for (auto &slot : state.slots) {
    if (slot.working || slot.quarantined) continue;
    std::uint32_t r = 0, g = 0;
    if (!choose_group(state, r, g)) break;
    auto &request = state.requests[r]; auto &group = state.group(r, g);
    auto *file = request.write ? FileStreamAccess::file(request.sink) : FileStreamAccess::file(request.source);
    const auto owner = file ? file->owner_worker() : 0u;
    auto stack = slot.prepared_stack.valid() ? Result<mem::FiberStack>(std::move(slot.prepared_stack)) : state.stacks.acquire();
    Status started;
    if (!stack) started = stack.error();
    else {
      slot.request = r; slot.group = g;
      slot.fiber.emplace(*std::move(stack), &stream_worker, &slot, &state.stacks);
      sched::detail::FiberAccess::set_on_finish(&*slot.fiber, &stream_worker_finished, &slot);
      slot.working = true; group.phase = GroupPhase::kWorking;
      ++state.worker_count; ++state.report.active_slots; state.incref();
      state.report.peak_slots = (std::max)(state.report.peak_slots, state.report.active_slots);
      started = scheduler.spawn_on(owner, &*slot.fiber, sched::ExecutionOptions{});
      if (!started.ok()) {
        slot.fiber.reset(); slot.working = false; --state.worker_count; --state.report.active_slots;
        state.decref(); // caller/finishing-worker reference still owns this locked state
      }
    }
    if (!started.ok()) {
      group.phase = GroupPhase::kFailed; request.result.status.Update(started);
      --request.remaining;
      for (std::uint32_t index = 0; index < request.result.total_groups; ++index)
        if (state.group(r, index).phase == GroupPhase::kQueued) {
          state.group(r, index).phase = GroupPhase::kFailed; --request.remaining;
        }
      request.result.complete = request.remaining == 0; state.changed.notify_all();
    }
  }
}

std::size_t cold_metadata_bytes(const FileStreamState &state) noexcept {
  return sizeof(state) + state.specs.capacity() * sizeof(CopyResourceSpec) +
      state.requests.capacity() * sizeof(StreamRequestCell) + state.groups.capacity() * sizeof(StreamGroupCell) +
      state.ranges.capacity() * sizeof(StreamRangeCell) + state.slots.capacity() * sizeof(StreamSlotCell) +
      state.admissions.capacity() * sizeof(StreamAdmission) +
      state.intervals.capacity() * sizeof(StreamInterval) + state.backings.capacity() * sizeof(LeaseBackingInfo);
}
} // namespace

Result<FileResourcePipeline> FileStreamAccess::create(io::IoContext &io, mem::MemoryResourceRef stage,
    std::span<const CopyResourceSpec> specs, FileStreamOptions options, StreamCopyEndpoint &&endpoint) noexcept {
  HYLE_RETURN_IF_ERROR(check_options(options));
  if (!io.valid() || !stage.valid() || !stage.properties().has(mem::MemoryProperty::kHostAccessible) ||
      !endpoint.owner || !endpoint.destroy || !endpoint.physical || !endpoint.copy || !endpoint.observe ||
      !endpoint.in_flight || !endpoint.reset || !endpoint.report)
    return std::unexpected(Status::InvalidArgument("invalid file stream IO/staging/native endpoint"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) { HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator()); }
  auto *storage = allocator.allocate(sizeof(FileStreamState), alignof(FileStreamState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("file stream state metadata"));
  FileStreamState *state = nullptr;
  try { state = std::construct_at(static_cast<FileStreamState *>(storage), io, std::move(options), allocator); }
  catch (...) { allocator.deallocate(storage, sizeof(FileStreamState), alignof(FileStreamState)); return std::unexpected(Status::ResourceExhausted("file stream state metadata")); }
  FileResourcePipeline result(state);
  state->endpoint = std::move(endpoint);
  const auto &limits = state->options;
  const auto group_count = static_cast<std::size_t>(limits.max_requests) * limits.max_groups_per_request;
  const auto range_count = group_count * limits.max_ranges_per_group;
  try {
    state->specs.assign(specs.begin(), specs.end()); state->requests.resize(limits.max_requests);
    state->groups.resize(group_count); state->ranges.resize(range_count); state->slots.resize(limits.in_flight_slots);
    state->admissions.reserve(static_cast<std::size_t>(limits.max_groups_per_request) * limits.max_ranges_per_group);
    state->intervals.reserve(range_count * 2); state->backings.reserve(range_count + 1);
  } catch (...) { return std::unexpected(Status::ResourceExhausted("file stream fixed arrays")); }
  HYLE_ASSIGN_OR_RETURN(state->staging, LeasePool::create(stage, limits.staging_bytes_per_slot * limits.in_flight_slots,
      {.max_blocks = limits.in_flight_slots, .backing_alignment = limits.staging_alignment,
       .allocator = allocator, .resource_budget = limits.resource_budget}));
  for (auto &slot : state->slots) {
    HYLE_ASSIGN_OR_RETURN(slot.writer, state->staging.allocate(limits.staging_bytes_per_slot, limits.staging_alignment));
    HYLE_ASSIGN_OR_RETURN(auto view, slot.writer.view());
    HYLE_ASSIGN_OR_RETURN(slot.physical, state->endpoint.physical(state->endpoint.owner, view, true));
    HYLE_ASSIGN_OR_RETURN(auto mapped, view.map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes, mapped.writable_bytes()); slot.address = bytes.data();
    HYLE_RETURN_IF_ERROR(mapped.close());
  }
  HYLE_ASSIGN_OR_RETURN(state->stage_backing, LeaseAccess::backing(state->slots[0].writer));
  state->report.staging_bytes = state->stage_backing.bytes;
  state->report.active_backing_bytes = state->report.peak_backing_bytes = state->stage_backing.bytes;
  auto native_report = state->endpoint.report(state->endpoint.owner);
  state->report.metadata_bytes = cold_metadata_bytes(*state) + native_report.metadata_bytes;
  state->report.native_metadata_unknown = native_report.native_metadata_unknown;
  for (std::uint32_t index = 0; index < limits.in_flight_slots; ++index) {
    auto &slot = state->slots[index]; slot.owner = state; slot.index = index;
    HYLE_ASSIGN_OR_RETURN(slot.prepared_stack, state->stacks.acquire());
    slot.job.ctx = &slot; slot.job.run_fn = &run_slot_job; slot.job.cancel_fn = &cancel_slot_job; slot.job.destroy_fn = &finish_slot_job;
  }
  { std::lock_guard guard(state->lock); state->initialized = true; state->changed.notify_all(); }
  return result;
}

} // namespace hyle::compute::detail

namespace hyle::compute {
namespace {
using detail::FileStreamState;
template<class Group> Result<std::uint32_t> admit(FileStreamState &state, StreamSource source,
    StreamFile sink, std::span<Group> groups, io::OperationOptions options) noexcept {
  constexpr bool write = std::is_same_v<Group, StreamWriteGroup>;
  std::lock_guard guard(state.lock);
  if (state.admission_closed) return std::unexpected(Status::FailedPrecondition("file stream admission is closed"));
  if (!io::detail::IoContextAccess::scheduler(state.io).accepting())
    return std::unexpected(Status::Cancelled("file stream scheduler is shutting down"));
  detail::reclaim_locked(state);
  std::uint32_t selected = UINT32_MAX;
  for (std::uint32_t index = 0; index < state.requests.size(); ++index)
    if (!state.requests[index].live) { selected = index; break; }
  if (selected == UINT32_MAX) return std::unexpected(Status::Unavailable("file stream request/result capacity is full"));
  if (groups.size() > state.options.max_groups_per_request || state.sequence == UINT64_MAX)
    return std::unexpected(Status::ResourceExhausted("file stream group/sequence capacity exceeded"));
  bool data = false;
  for (const auto &group : groups) for (const auto &range : group.ranges) data |= range.bytes != 0;
  auto *file = write ? detail::FileStreamAccess::file(sink) : detail::FileStreamAccess::file(source);
  auto *mapping = write ? nullptr : detail::FileStreamAccess::mapping(source);
  if (data) {
    if (file) HYLE_RETURN_IF_ERROR(detail::check_file(state, file));
    else if (!mapping) return std::unexpected(Status::InvalidArgument("file stream source/sink is empty"));
  }
  const auto capacity = write ? sink.capacity_bytes() : source.capacity_bytes();
  detail::existing_accounting(state, true); state.admissions.clear();
  std::uint64_t total = 0;
  for (const auto &group : groups) {
    if (group.ranges.size() > state.options.max_ranges_per_group)
      return std::unexpected(Status::ResourceExhausted("file stream range capacity exceeded"));
    for (const auto &range : group.ranges) {
      if (range.bytes > UINT64_MAX - total) return std::unexpected(Status::OutOfRange("file stream byte total overflow"));
      total += range.bytes;
      mem::ConstBufferView view;
      detail::LeaseBackingInfo backing;
      bool writable = !write;
      if constexpr (write) {
        if (range.source.valid() == range.pending_source.valid()) {
          if (range.bytes || range.source.valid()) return std::unexpected(Status::InvalidArgument("stream write requires one source permission"));
        }
        if (range.pending_source.valid()) {
          writable = true;
          if (!range.producer.valid()) return std::unexpected(Status::InvalidArgument("stream mutable source requires producer"));
          HYLE_ASSIGN_OR_RETURN(view, detail::LeaseAccess::ordered_view(range.pending_source));
          HYLE_ASSIGN_OR_RETURN(backing, detail::LeaseAccess::backing(range.pending_source));
        } else if (range.source.valid()) {
          if (range.producer.valid()) return std::unexpected(Status::InvalidArgument("ready stream source has an unexpected producer"));
          HYLE_ASSIGN_OR_RETURN(view, range.source.view());
          HYLE_ASSIGN_OR_RETURN(backing, detail::LeaseAccess::backing(range.source));
        }
      } else {
        if (range.destination.valid()) {
          HYLE_ASSIGN_OR_RETURN(view, detail::LeaseAccess::ordered_view(range.destination));
          HYLE_ASSIGN_OR_RETURN(backing, detail::LeaseAccess::backing(range.destination));
        }
        if (range.description) {
          HYLE_ASSIGN_OR_RETURN(auto required, range.description->required_bytes());
          if (required != range.bytes) return std::unexpected(Status::InvalidArgument("stream tensor description does not cover exact logical range"));
          HYLE_ASSIGN_OR_RETURN(auto storage, view.subview(range.resource_offset, range.bytes));
          HYLE_ASSIGN_OR_RETURN(auto tensor, ConstTensorView::create(*range.description, storage)); (void)tensor;
        }
      }
      detail::StreamAdmission admission;
      if (view.valid()) {
        HYLE_RETURN_IF_ERROR(detail::check_range(state, view, writable,
            range.file_offset, capacity, range.resource_offset, range.bytes, range.physical_bytes,
            backing, file, mapping, admission));
        state.backings.push_back(backing);
        detail::add_interval(state, admission.physical, range.resource_offset,
                            range.physical_bytes ? range.physical_bytes : range.bytes, !write);
      } else if (range.bytes || range.physical_bytes || range.resource_offset || range.file_offset) {
        return std::unexpected(Status::InvalidArgument("stream range has no lease"));
      }
      if constexpr (write) if (range.bytes) detail::add_file_interval(state, file, range.file_offset, range.bytes);
      state.admissions.push_back(admission);
    }
  }
  HYLE_RETURN_IF_ERROR(detail::interval_hazards(state));
  const auto active_bytes = detail::backing_usage(state);
  if (active_bytes > state.options.max_active_backing_bytes)
    return std::unexpected(Status::Unavailable("file stream backing byte capacity is full"));
  Result<sched::CancelToken> cancel = options.cancel && options.cancel->valid()
      ? options.cancel->try_derive() : sched::CancelToken::try_create_root(state.allocator);
  if (!cancel) return std::unexpected(cancel.error());
  // Everything that can reject admission has completed. No input permission is
  // moved before this transaction commits to pre-existing worker/record slots.
  auto &request = state.requests[selected];
  request.source = std::move(source); request.sink = std::move(sink); request.cancel = std::move(*cancel);
  request.io_options = options; request.io_options.cancel = &request.cancel;
  request.live = request.ticket = true; request.write = write; request.sequence = state.sequence++;
  request.result.total_groups = static_cast<std::uint32_t>(groups.size());
  request.remaining = request.result.total_groups; request.result.complete = groups.empty();
  std::size_t admitted = 0;
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    auto &cell = state.group(selected, g); cell.tag = groups[g].tag;
    cell.count = static_cast<std::uint32_t>(groups[g].ranges.size());
    for (std::uint32_t i = 0; i < cell.count; ++i) {
      auto &range = state.range(selected, g, i); auto &input = groups[g].ranges[i];
      range.file_offset = input.file_offset; range.offset = input.resource_offset;
      range.bytes = input.bytes; range.physical_bytes = input.physical_bytes ? input.physical_bytes : input.bytes;
      range.backing = state.admissions[admitted].backing; range.physical = state.admissions[admitted++].physical;
      if constexpr (write) {
        range.reader = std::move(input.source); range.writer = std::move(input.pending_source);
        range.producer = std::move(input.producer);
      } else { range.writer = std::move(input.destination); }
    }
  }
  ++state.report.live_requests; state.report.active_backing_bytes = active_bytes;
  state.report.external_backing_bytes = active_bytes - state.stage_backing.bytes;
  state.report.peak_backing_bytes = (std::max)(state.report.peak_backing_bytes, active_bytes);
  detail::kick_locked(state); state.changed.notify_all(); return selected;
}
void cancel_request(FileStreamState &state, std::uint32_t request) noexcept {
  sched::CancelToken token;
  { std::lock_guard guard(state.lock); token = state.requests[request].cancel; }
  token.cancel();
}
} // namespace

StreamReadyGroup::StreamReadyGroup(detail::FileStreamState *state, std::uint32_t request, std::uint32_t group) noexcept
    : state_(state), request_(request), group_(group) { state_->incref(); }
StreamReadyGroup::StreamReadyGroup(const StreamReadyGroup &other) noexcept
    : state_(other.state_), request_(other.request_), group_(other.group_) {
  if (state_) { state_->incref(); std::lock_guard guard(state_->lock); ++state_->group(request_, group_).handles; }
}
StreamReadyGroup &StreamReadyGroup::operator=(const StreamReadyGroup &other) noexcept {
  if (this != &other) { StreamReadyGroup next(other); *this = std::move(next); } return *this;
}
StreamReadyGroup::StreamReadyGroup(StreamReadyGroup &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)), request_(other.request_), group_(other.group_) {}
StreamReadyGroup &StreamReadyGroup::operator=(StreamReadyGroup &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); request_ = other.request_; group_ = other.group_; } return *this;
}
StreamReadyGroup::~StreamReadyGroup() { reset(); }
void StreamReadyGroup::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) {
    { std::lock_guard guard(state->lock); --state->group(request_, group_).handles; }
    detail::release_state(state);
  }
}
std::uint64_t StreamReadyGroup::tag() const noexcept {
  if (!state_) return 0; std::lock_guard guard(state_->lock); return state_->group(request_, group_).tag;
}
std::uint32_t StreamReadyGroup::range_count() const noexcept {
  if (!state_) return 0; std::lock_guard guard(state_->lock); return state_->group(request_, group_).count;
}
Result<ReadLease> StreamReadyGroup::output(std::uint32_t index) const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("stream ready group is empty"));
  std::lock_guard guard(state_->lock);
  if (index >= state_->group(request_, group_).count) return std::unexpected(Status::OutOfRange("stream output index exceeds group"));
  return state_->range(request_, group_, index).output;
}
StreamTicket::StreamTicket(detail::FileStreamState *state, std::uint32_t request) noexcept
    : state_(state), request_(request) { state_->incref(); }
StreamTicket::StreamTicket(StreamTicket &&other) noexcept : state_(std::exchange(other.state_, nullptr)), request_(other.request_) {}
StreamTicket &StreamTicket::operator=(StreamTicket &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); request_ = other.request_; } return *this;
}
StreamTicket::~StreamTicket() { reset(); }
void StreamTicket::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) {
    cancel_request(*state, request_);
    { std::lock_guard guard(state->lock); state->requests[request_].ticket = false; }
    detail::release_state(state);
  }
}
Result<StreamReadyGroup> StreamTicket::next_ready() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("stream ticket is empty"));
  std::lock_guard guard(state_->lock); auto &request = state_->requests[request_];
  for (std::uint32_t g = 0; g < request.result.total_groups; ++g) {
    auto &group = state_->group(request_, g);
    if (group.phase == detail::GroupPhase::kReady && !group.delivered) {
      group.delivered = true; ++group.handles; return StreamReadyGroup(state_, request_, g);
    }
  }
  if (request.result.complete) return std::unexpected(request.result.status.ok()
      ? Status::OutOfRange("stream request has no more ready groups") : request.result.status);
  return std::unexpected(Status::Unavailable("stream ready group is pending"));
}
Result<StreamReadyGroup> StreamTicket::wait_next_ready() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("stream ticket is empty"));
  state_->lock.lock();
  state_->changed.wait(state_->lock, [&] {
    const auto &request = state_->requests[request_];
    if (request.result.complete) return true;
    for (std::uint32_t g = 0; g < request.result.total_groups; ++g) {
      const auto &group = state_->group(request_, g);
      if (group.phase == detail::GroupPhase::kReady && !group.delivered) return true;
    } return false;
  });
  state_->lock.unlock(); return next_ready();
}
StreamResult StreamTicket::result() const noexcept {
  if (!state_) return {Status::FailedPrecondition("stream ticket is empty")};
  std::lock_guard guard(state_->lock); return state_->requests[request_].result;
}
StreamResult StreamTicket::wait() noexcept {
  if (!state_) return {Status::FailedPrecondition("stream ticket is empty")};
  state_->lock.lock(); state_->changed.wait(state_->lock, [&] { return state_->requests[request_].result.complete; });
  auto result = state_->requests[request_].result; state_->lock.unlock(); return result;
}
void StreamTicket::request_cancel() noexcept { if (state_) cancel_request(*state_, request_); }
FileResourcePipeline::FileResourcePipeline(FileResourcePipeline &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
FileResourcePipeline &FileResourcePipeline::operator=(FileResourcePipeline &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
FileResourcePipeline::~FileResourcePipeline() { reset(); }
void FileResourcePipeline::reset() noexcept {
  if (auto *state = state_) {
    stop_admission();
    for (std::uint32_t r = 0; r < state->requests.size(); ++r) {
      bool live; { std::lock_guard guard(state->lock); live = state->requests[r].live; }
      if (live) cancel_request(*state, r);
    }
    close().IgnoreError(); state_ = nullptr; detail::release_state(state);
  }
}
Result<StreamTicket> FileResourcePipeline::read(StreamSource source, std::span<StreamReadGroup> groups, io::OperationOptions options) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("file stream pipeline is empty"));
  HYLE_ASSIGN_OR_RETURN(auto index, admit(*state_, std::move(source), {}, groups, options)); return StreamTicket(state_, index);
}
Result<StreamTicket> FileResourcePipeline::write(StreamFile sink, std::span<StreamWriteGroup> groups, io::OperationOptions options) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("file stream pipeline is empty"));
  HYLE_ASSIGN_OR_RETURN(auto index, admit(*state_, {}, std::move(sink), groups, options)); return StreamTicket(state_, index);
}
void FileResourcePipeline::stop_admission() noexcept {
  if (state_) { std::lock_guard guard(state_->lock); state_->admission_closed = true; }
}
Status FileResourcePipeline::collect() noexcept {
  if (!state_) return Status::Ok();
  std::lock_guard guard(state_->lock); Status result;
  for (std::uint32_t i = 0; i < state_->slots.size(); ++i) {
    auto &slot = state_->slots[i]; if (!slot.quarantined) continue;
    result.Update(state_->endpoint.observe(state_->endpoint.owner, i));
    if (state_->endpoint.in_flight(state_->endpoint.owner, i)) continue;
    auto reset = state_->endpoint.reset(state_->endpoint.owner, i);
    result.Update(reset); if (!reset.ok()) continue;
    slot.quarantined = false;
    bool other = false;
    for (const auto &pending : state_->slots) other |= pending.quarantined && pending.request == slot.request;
    if (!other) state_->requests[slot.request].result.retirement_proven = true;
  }
  detail::reclaim_locked(*state_); detail::kick_locked(*state_); state_->changed.notify_all(); return result;
}
Status FileResourcePipeline::close() noexcept {
  if (!state_) return Status::Ok();
  std::lock_guard guard(state_->lock); state_->closing = state_->admission_closed = true; state_->changed.notify_all();
  for (const auto &request : state_->requests)
    if (request.live && (!request.result.complete || !request.result.retirement_proven))
      return Status::Unavailable("file stream still has accepted work");
  return state_->worker_count ? Status::Unavailable("file stream workers are retiring") : Status::Ok();
}
Status FileResourcePipeline::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError(); state_->lock.lock();
  state_->changed.wait(state_->lock, [&] {
    // An unknown observation may quarantine every slot. There is no safe
    // progress source for queued work until collect qualifies that observation;
    // do not wait forever or turn that queued work into a fabricated completion.
    if (state_->worker_count == 0) return true;
    for (const auto &request : state_->requests) if (request.live && !request.result.complete) return false;
    return state_->worker_count == 0;
  });
  state_->lock.unlock();
  auto result = collect();
  std::lock_guard guard(state_->lock);
  for (const auto &request : state_->requests) if (request.live) {
    result.Update(request.result.status);
    if (!request.result.complete) result.Update(Status::Unavailable("file stream has queued work behind an unproven native slot"));
    if (!request.result.retirement_proven) result.Update(Status::Unavailable("file stream native retirement is unproven"));
  }
  return result;
}
FileStreamReport FileResourcePipeline::report() const noexcept {
  if (!state_) return {};
  std::lock_guard guard(state_->lock); return state_->report;
}
} // namespace hyle::compute
