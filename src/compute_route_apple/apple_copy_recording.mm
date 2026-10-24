#include "apple_copy_internal.h"

#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <mutex>
#include <vector>
#include <mlx/backend/metal/device.h>

#include "apple_internal.h"
#include "completion_internal.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::apple::detail {
namespace cd = compute::detail;
template<class T> using CopyArray = std::vector<T, mem::StdAllocator<T>>;
struct CopyCommand final {
  void *source = nullptr, *destination = nullptr;
  std::size_t source_offset = 0, destination_offset = 0, bytes = 0;
};
struct CopyRecording final : Job {
  explicit CopyRecording(mem::AllocatorRef allocator)
      : physical(mem::StdAllocator<cd::PhysicalCopyResource>(allocator)),
        inputs(mem::StdAllocator<CopyRange>(allocator)), intervals(mem::StdAllocator<cd::CopyInterval>(allocator)),
        resolved(mem::StdAllocator<cd::ResolvedCopyRange>(allocator)), commands(mem::StdAllocator<CopyCommand>(allocator)) {}
  std::shared_ptr<Runtime> runtime;
  // Both the original worker and public Completion hold a preallocated slot.
  // The worker owner is moved locally before done becomes visible.
  std::shared_ptr<CopyRecording> admitted_owner, completion_owner;
  std::weak_ptr<CopyRecording> self;
  mutable std::mutex completion_mutex;
  std::atomic<std::uint32_t> done{1};
  std::atomic<bool> cancel{false};
  std::atomic<std::uint64_t> calls{0}, waits{0};
  cd::CompletionLifetime abandoned;
  Status terminal, hint;
  CopyBatchOptions limits;
  CopyBatchReport report;
  std::uintptr_t device = 0;
  CopyRecordingMode mode = CopyRecordingMode::kOnce;
  bool prepared = false;
  bool prefix_marker = false;
  CopyPreflight preflight = nullptr;
  void *preflight_state = nullptr;
  bool host_copy = false;
  InputSource host_source;
  ReadLease host_owner;
  InputSpec host_spec;
  mem::HostMapping host_input, host_output;
  std::span<std::byte> host_output_bytes;
  std::size_t host_bytes = 0;
  std::uint64_t consumed_epoch = 0, expected_epoch = 0;
  bool source_pinned = false;
  CopyArray<cd::PhysicalCopyResource> physical;
  CopyArray<CopyRange> inputs;
  CopyArray<cd::CopyInterval> intervals;
  CopyArray<cd::ResolvedCopyRange> resolved;
  CopyArray<CopyCommand> commands;
  std::array<std::uint64_t, 1> closure{};
};
namespace {
void deliver(cd::CompletionLifetime lifetime, Status terminal, const Status &hint) noexcept {
  if (!lifetime.valid()) return;
  terminal.Update(hint); lifetime.terminal(lifetime.state, terminal); lifetime.release(lifetime.state);
}
void finish(CopyRecording &state, Status terminal) noexcept {
  cd::CompletionLifetime abandoned; Status hint;
  {
    std::lock_guard lock(state.completion_mutex);
    if (state.source_pinned) {
      InvocationAccess::release_source(state.host_source);
      state.source_pinned = false;
    }
    state.terminal = terminal; abandoned = std::exchange(state.abandoned, {}); hint = std::move(state.hint);
    state.done.store(1, std::memory_order_release);
  }
  platform::detail::wake_all(&state.done);
  deliver(abandoned, std::move(terminal), hint);
}
Status execute_host_copy(CopyRecording &state) noexcept {
  const void *source = nullptr;
  std::optional<mlx::core::array> output;
  if (state.host_source.valid()) {
    HYLE_ASSIGN_OR_RETURN(output, InvocationAccess::source_output(
        state.host_source, state.runtime, state.expected_epoch));
    if (output->shape() != state.host_spec.shape || output->dtype() != state.host_spec.dtype ||
        !output->flags().row_contiguous || output->nbytes() != state.host_bytes)
      return Status::InvalidArgument("Apple copy requires its exact dense native output carrier");
    HYLE_ASSIGN_OR_RETURN(auto storage, MemoryAccess::storage_info(*output));
    if (storage.view_bytes != state.host_bytes)
      return Status::InvalidArgument("Apple copy output byte span differs");
    if (state.host_bytes) {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
      if (buffer == nil || buffer.contents == nullptr ||
          reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device) != state.device)
        return Status::InvalidArgument("Apple copy output belongs to a foreign Metal Device");
      source = static_cast<const std::byte *>(buffer.contents) + storage.offset_bytes;
    }
  } else {
    HYLE_RETURN_IF_ERROR(state.host_input.invalidate());
    if (state.host_input.bytes().size() != state.host_bytes)
      return Status::Internal("Apple fixed copy source byte span changed");
    source = state.host_input.bytes().data();
  }
  // All producer/epoch/layout/device/range checks finish before touching the
  // cold destination mapping. The original producer has already drained on
  // this same FIFO; no new eval, host wait, allocation or projection is used.
  if (state.host_bytes) std::memmove(state.host_output_bytes.data(), source, state.host_bytes);
  return state.host_output.flush();
}
void execute_copy(Job *job) noexcept {
  auto &state = *static_cast<CopyRecording *>(job);
  std::shared_ptr<CopyRecording> owner;
  { std::lock_guard lock(state.completion_mutex); owner = std::move(state.admitted_owner); }
  Status terminal;
  {
    std::lock_guard lock(state.runtime->queue_mutex); terminal = state.runtime->failure;
  }
  if (state.cancel.load(std::memory_order_acquire)) terminal.Update(Status::Cancelled("Apple copy cancelled before native recording"));
  if (terminal.ok() && state.preflight) terminal.Update(state.preflight(state.preflight_state));
  if (terminal.ok() && state.host_copy) terminal = execute_host_copy(state);
  else if (terminal.ok() && state.report.active_copies != 0) {
    @autoreleasepool {
      try {
        auto &encoder = mlx::core::metal::get_command_encoder(*state.runtime->stream);
        // Prior original Jobs synchronize before their retirement, so these
        // untracked buffers have no unfinished earlier encoder when this FIFO
        // Job starts. End any encoder before the single original native blit.
        encoder.end_encoding();
        auto native_command = NS::RetainPtr(encoder.get_command_buffer());
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)native_command.get();
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (blit == nil) {
          terminal = Status::ResourceExhausted("Metal blit command encoder allocation failed");
        } else {
          for (std::size_t index = 0; index < state.report.active_copies; ++index) {
            const auto &copy = state.commands[index];
            [blit copyFromBuffer:(__bridge id<MTLBuffer>)copy.source sourceOffset:copy.source_offset
                toBuffer:(__bridge id<MTLBuffer>)copy.destination destinationOffset:copy.destination_offset size:copy.bytes];
            state.calls.fetch_add(1, std::memory_order_relaxed);
          }
          [blit endEncoding];
          state.waits.fetch_add(1, std::memory_order_relaxed); // original owner-thread boundary
          encoder.synchronize(); // original stream commit, actual MCB wait and native error check
          if (command.status == MTLCommandBufferStatusError) {
            const auto *description = command.error.localizedDescription.UTF8String;
            terminal = Status::Internal(description ? description : "Metal blit command failed");
          }
        }
      } catch (...) {
        terminal = native_exception();
        // The original synchronization waits an accepted prefix before its
        // error check. Drain this same owner/stream if recording threw earlier.
        try { state.waits.fetch_add(1, std::memory_order_relaxed); mlx::core::metal::get_command_encoder(*state.runtime->stream).synchronize(); }
        catch (...) {}
        state.runtime->poison(terminal);
      }
    }
    if (state.cancel.load(std::memory_order_acquire)) terminal.Update(Status::Cancelled("Apple copy cancelled after native retirement"));
  }
  if (state.host_copy && state.cancel.load(std::memory_order_acquire))
    terminal.Update(Status::Cancelled("Apple host copy cancelled after original FIFO retirement"));
  finish(state, std::move(terminal));
}
const cd::CompletionVTable copy_completion_vtable{
    .destroy = [](void *opaque, cd::CompletionLifetime lifetime, Status hint) noexcept {
      auto &state = *static_cast<CopyRecording *>(opaque); std::shared_ptr<CopyRecording> owner;
      Status terminal; bool ready;
      {
        std::lock_guard lock(state.completion_mutex);
        ready = state.done.load(std::memory_order_acquire) != 0;
        if (ready) terminal = state.terminal;
        else { state.abandoned = lifetime; state.hint = hint; }
        owner = std::move(state.completion_owner);
      }
      if (ready) deliver(lifetime, std::move(terminal), hint);
    },
    .poll = [](void *opaque) noexcept -> Result<CompletionPoll> {
      const auto &state = *static_cast<CopyRecording *>(opaque);
      if (state.done.load(std::memory_order_acquire) == 0) return CompletionPoll::kPending;
      if (!state.terminal.ok()) return std::unexpected(state.terminal);
      return CompletionPoll::kComplete;
    },
    .wait = [](void *opaque) noexcept -> Status {
      const auto &state = *static_cast<CopyRecording *>(opaque);
      while (state.done.load(std::memory_order_acquire) == 0) platform::detail::wait_on_address(&state.done, 0);
      return state.terminal;
    },
    .request_cancel = [](void *opaque) noexcept -> Status {
      static_cast<CopyRecording *>(opaque)->cancel.store(true, std::memory_order_release); return Status::Ok();
    }};
} // namespace

Result<std::shared_ptr<CopyRecording>> prepare_copy_recording(const Context &context,
    std::uint32_t capacity, mem::AllocatorRef allocator) noexcept {
  if (!context.valid()) return std::unexpected(Status::InvalidArgument("Apple copy Context is empty"));
  return prepare_copy_recording_runtime(ContextAccess::runtime(context), capacity, allocator, 64u * 1024u * 1024u);
}
Result<std::shared_ptr<CopyRecording>> prepare_copy_recording_runtime(const std::shared_ptr<Runtime> &runtime,
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes, CopyRecordingMode mode) noexcept {
  if (!runtime || !allocator.valid() || capacity == 0 || capacity > UINT32_MAX / 2 || max_bytes == 0)
    return std::unexpected(Status::InvalidArgument("Apple copy limits are invalid"));
  { std::lock_guard lock(runtime->queue_mutex);
    if (!runtime->accepting) return std::unexpected(Status::FailedPrecondition("Apple copy Context is closed"));
    if (!runtime->failure.ok()) return std::unexpected(runtime->failure); }
  try {
    auto state = std::allocate_shared<CopyRecording>(mem::StdAllocator<CopyRecording>(allocator), allocator);
    state->self = state;
    state->runtime = runtime; state->execute = &execute_copy;
    state->mode = mode;
    state->device = reinterpret_cast<std::uintptr_t>(mlx::core::metal::device(runtime->stream->device).mtl_device());
    state->limits.max_resources = capacity * 2; state->limits.max_copies = capacity;
    state->limits.max_phases = 1; state->limits.max_bytes = max_bytes;
    state->physical.resize(static_cast<std::size_t>(capacity) * 2); state->inputs.resize(capacity);
    state->intervals.resize(static_cast<std::size_t>(capacity) * 2); state->resolved.resize(capacity); state->commands.resize(capacity);
    return state;
  } catch (...) { return std::unexpected(native_exception()); }
}
namespace {
Status mutable_recording(const CopyRecording &state) noexcept {
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple copy phase is retained");
  return state.terminal;
}
Status clear_host_copy(CopyRecording &state) noexcept {
  Status result = state.host_input.close();
  result.Update(state.host_output.close());
  state.host_source.reset(); state.host_copy = false; state.host_output_bytes = {};
  state.host_owner.reset();
  state.host_bytes = 0; state.consumed_epoch = 0; state.expected_epoch = 0;
  return result;
}
Status bind_host_destination(CopyRecording &state, mem::BufferView destination,
    std::size_t bytes, mem::HostMapping *mapping, std::span<std::byte> *output) noexcept {
  if (!destination.valid() || !destination.resource().properties().has(mem::MemoryProperty::kHostAccessible) ||
      destination.size() < bytes || bytes > state.limits.max_bytes)
    return Status::InvalidArgument("Apple host copy destination/range differs");
  HYLE_ASSIGN_OR_RETURN(auto range, destination.subview(0, bytes));
  HYLE_ASSIGN_OR_RETURN(*mapping, range.map_host(mem::HostAccess::kWrite));
  HYLE_ASSIGN_OR_RETURN(*output, mapping->writable_bytes());
  return Status::Ok();
}
void host_copy_report(CopyRecording &state, std::size_t bytes) noexcept {
  state.report = {};
  state.report.logical_bytes = bytes; state.report.physical_bytes = bytes;
  state.report.active_copies = bytes ? 1U : 0U;
  state.report.no_op_copies = bytes ? 0U : 1U;
  state.report.native_metadata_unknown = true;
  state.prefix_marker = true; state.prepared = true; state.host_copy = true;
  state.calls.store(0, std::memory_order_relaxed); state.waits.store(0, std::memory_order_relaxed);
}
}
Status bind_host_copy_recording(CopyRecording &state, InputSource source,
    const TensorDesc &description, mem::BufferView destination) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  state.prepared = false;
  if (!source.valid() || (description.is_plain() && !description.is_dense()))
    return Status::InvalidArgument("Apple host copy requires its dense native producer");
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto epoch, InvocationAccess::source_epoch(source, state.runtime));
  mem::HostMapping mapping;
  std::span<std::byte> output;
  HYLE_RETURN_IF_ERROR(bind_host_destination(state, destination, bytes, &mapping, &output));
  HYLE_RETURN_IF_ERROR(clear_host_copy(state));
  state.host_source = std::move(source); state.host_spec = std::move(spec);
  state.host_output = std::move(mapping); state.host_output_bytes = output;
  state.host_bytes = bytes; state.consumed_epoch = epoch;
  host_copy_report(state, bytes);
  return Status::Ok();
}
Status bind_host_copy_recording(CopyRecording &state, ConstTensorView source,
    mem::BufferView destination, ReadLease source_owner) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  state.prepared = false;
  if (!source.valid() || (source.description().is_plain() && !source.description().is_dense()))
    return Status::InvalidArgument("Apple fixed host copy requires a dense tensor");
  HYLE_ASSIGN_OR_RETURN(auto bytes, source.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto range, source.storage().subview(0, bytes));
  if (bytes) {
    HYLE_ASSIGN_OR_RETURN(auto span, copy_span(range, mem::HostAccess::kRead));
    if (span.physical.address_space != state.device)
      return Status::InvalidArgument("Apple fixed host copy source Device differs");
  } else if (!is_copy_resource(range.resource())) {
    return Status::InvalidArgument("Apple empty host copy requires its actual native resource");
  }
  HYLE_ASSIGN_OR_RETURN(auto input, range.map_host());
  mem::HostMapping mapping;
  std::span<std::byte> output;
  HYLE_RETURN_IF_ERROR(bind_host_destination(state, destination, bytes, &mapping, &output));
  HYLE_RETURN_IF_ERROR(clear_host_copy(state));
  state.host_input = std::move(input); state.host_output = std::move(mapping);
  state.host_owner = std::move(source_owner);
  state.host_output_bytes = output; state.host_bytes = bytes;
  host_copy_report(state, bytes);
  return Status::Ok();
}
Status copy_recording_reuse_gate(const CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  return mutable_recording(state);
}
Status copy_recording_retired_gate(const CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  return state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner ?
      Status::Unavailable("Apple copy phase is retained") : Status::Ok();
}
Status bind_native_copy_source(CopyRecording &state, InputSource source,
    const TensorDesc &description) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  if (!state.prepared || !source.valid() || !description.is_plain() || !description.is_dense())
    return Status::InvalidArgument("Apple native copy needs its prepared dense producer");
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto epoch, InvocationAccess::source_epoch(source, state.runtime));
  state.host_source = std::move(source); state.host_spec = std::move(spec);
  state.host_bytes = bytes; state.consumed_epoch = epoch;
  return Status::Ok();
}
Result<CopySpan> native_copy_source(CopyRecording &state) noexcept {
  if (!state.host_source.valid() || state.host_copy || !state.source_pinned)
    return std::unexpected(Status::FailedPrecondition("Apple native copy producer is not pinned"));
  HYLE_ASSIGN_OR_RETURN(auto output, InvocationAccess::source_output(
      state.host_source, state.runtime, state.expected_epoch));
  if (output.shape() != state.host_spec.shape || output.dtype() != state.host_spec.dtype ||
      !output.flags().row_contiguous || output.nbytes() != state.host_bytes)
    return std::unexpected(Status::InvalidArgument("Apple native copy producer layout changed"));
  HYLE_ASSIGN_OR_RETURN(auto storage, MemoryAccess::storage_info(output));
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
  if (buffer == nil || buffer.storageMode != MTLStorageModeShared ||
      storage.offset_bytes > buffer.length || storage.view_bytes != state.host_bytes ||
      storage.view_bytes > buffer.length - storage.offset_bytes ||
      reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device) != state.device)
    return std::unexpected(Status::InvalidArgument("Apple native copy producer backing/Device differs"));
  return CopySpan{const_cast<void *>(storage.buffer), {state.device,
      reinterpret_cast<std::uintptr_t>(storage.buffer), storage.offset_bytes, storage.view_bytes, 4}};
}
Status bind_copy_recording(CopyRecording &state, std::span<const cd::CopyBackendRange> ranges, bool prefix_marker) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple copy phase is retained");
  if (!state.terminal.ok()) return state.terminal;
  state.prepared = false;
  HYLE_RETURN_IF_ERROR(clear_host_copy(state));
  state.prefix_marker = prefix_marker;
  if (ranges.size() > state.limits.max_copies) return Status::ResourceExhausted("Apple copy range capacity exhausted");
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    if (ranges[index].source.size() != ranges[index].destination.size()) return Status::InvalidArgument("Apple copy sizes differ");
    HYLE_ASSIGN_OR_RETURN(auto source, copy_span(ranges[index].source, mem::HostAccess::kRead));
    HYLE_ASSIGN_OR_RETURN(auto destination, copy_span(ranges[index].destination, mem::HostAccess::kWrite));
    if (source.physical.address_space != state.device || destination.physical.address_space != state.device)
      return Status::InvalidArgument("Apple copy buffer belongs to a foreign Metal Device");
    state.physical[2 * index] = source.physical; state.physical[2 * index + 1] = destination.physical;
    const auto id = static_cast<std::uint32_t>(index);
    state.inputs[index] = {{2 * id}, {2 * id + 1}, {0}, 0, 0, ranges[index].source.size()};
  }
  const std::array<CopyPhase, 1> phases{{{}}};
  HYLE_ASSIGN_OR_RETURN(auto report, cd::validate_copy_ranges(std::span(state.physical).first(ranges.size() * 2),
      phases, std::span(state.inputs).first(ranges.size()), state.limits, {state.intervals, state.resolved, state.closure}));
  for (std::size_t index = 0; index < report.active_copies; ++index) {
    const auto &range = ranges[state.resolved[index].input_index];
    state.commands[index] = {range.source.native_handle(), range.destination.native_handle(),
        range.source.offset(), range.destination.offset(), range.source.size()};
  }
  report.native_metadata_unknown = true; state.report = report; state.prepared = true;
  state.calls.store(0, std::memory_order_relaxed); state.waits.store(0, std::memory_order_relaxed);
  return Status::Ok();
}
Result<Completion> submit_copy_recording(CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (!state.prepared || state.completion_owner || state.done.load(std::memory_order_acquire) == 0)
    return std::unexpected(Status::FailedPrecondition("Apple copy phase is not prepared/idle"));
  if (!state.terminal.ok()) return std::unexpected(state.terminal);
  if (state.host_source.valid()) {
    HYLE_ASSIGN_OR_RETURN(state.expected_epoch, InvocationAccess::pin_source(
        state.host_source, state.runtime, state.consumed_epoch));
    state.source_pinned = true;
  }
  if (state.mode == CopyRecordingMode::kOnce) state.prepared = false;
  if (state.report.active_copies == 0 && !state.prefix_marker) return Completion::completed();
  auto owner = state.self.lock(); HYLE_FATAL_ASSERT(owner != nullptr);
  state.cancel.store(false, std::memory_order_relaxed); state.terminal = Status::Ok(); state.hint = Status::Ok();
  state.admitted_owner = owner; state.completion_owner = owner;
  state.done.store(0, std::memory_order_release);
  const auto admitted = state.runtime->enqueue(state);
  if (!admitted.ok()) {
    if (state.source_pinned) {
      InvocationAccess::release_source(state.host_source); state.source_pinned = false;
    }
    state.done.store(1, std::memory_order_release); state.admitted_owner.reset(); state.completion_owner.reset();
    return std::unexpected(admitted);
  }
  if (state.source_pinned) state.consumed_epoch = state.expected_epoch;
  state.report.phase_completions = 1;
  return cd::CompletionAccess::make(&copy_completion_vtable, &state);
}
Status bind_result_copy_recording(CopyRecording&state,std::span<const cd::CopyBackendRange>ranges)noexcept{
  std::lock_guard lock(state.completion_mutex);
  if(state.done.load(std::memory_order_acquire)==0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple result-copy phase is retained");
  if(!state.terminal.ok())return state.terminal;
  state.prepared=false;
  HYLE_RETURN_IF_ERROR(clear_host_copy(state));
  if(ranges.size()>state.limits.max_copies)return Status::ResourceExhausted("Apple result-copy capacity");
  std::size_t bytes=0;
  for(std::size_t index=0;index<ranges.size();++index){
    HYLE_ASSIGN_OR_RETURN(auto destination,copy_span(ranges[index].destination,mem::HostAccess::kWrite));
    if(destination.physical.address_space!=state.device)return Status::InvalidArgument("Apple result-copy destination Device differs");
    if(destination.physical.capacity>state.limits.max_bytes-bytes)return Status::ResourceExhausted("Apple result-copy byte capacity");
    bytes+=destination.physical.capacity;
    state.commands[index]={nullptr,destination.buffer,0,destination.physical.origin,destination.physical.capacity};
  }
  state.report={};state.report.active_copies=static_cast<std::uint32_t>(ranges.size());
  state.report.logical_bytes=bytes;state.report.physical_bytes=bytes;state.report.native_metadata_unknown=true;
  state.prefix_marker=true;state.prepared=true;
  state.calls.store(0,std::memory_order_relaxed);state.waits.store(0,std::memory_order_relaxed);return Status::Ok();
}
Status resolve_result_copy_recording(CopyRecording&state,std::span<const cd::CopyBackendRange>ranges,
    ResultCopySourceFn source,void*owner)noexcept{
  if(!source || state.runtime->owner_id.load(std::memory_order_acquire)!=platform::this_thread::get_id())
    return Status::FailedPrecondition("Apple result-copy resolution requires the original FIFO owner");
  if(ranges.size()!=state.report.active_copies)return Status::Internal("Apple result-copy descriptor set changed");
  for(std::size_t index=0;index<ranges.size();++index){
    HYLE_ASSIGN_OR_RETURN(auto span,source(owner,index));
    if(span.physical.address_space!=state.device || span.physical.capacity!=ranges[index].destination.size())
      return Status::InvalidArgument("Apple result-copy source Device/range differs");
    state.commands[index].source=span.buffer;state.commands[index].source_offset=span.physical.origin;
  }return Status::Ok();
}
CopyBatchReport copy_recording_report(const CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  auto report = state.report;
  report.native_calls = state.calls.load(std::memory_order_relaxed);
  report.host_waits = state.waits.load(std::memory_order_relaxed);
  return report;
}
bool copy_recording_matches_source(const CopyRecording &state, const InputSource &source) noexcept {
  std::lock_guard lock(state.completion_mutex);
  return state.host_source.same_output(source);
}
Status clear_copy_recording(CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple copy phase is retained");
  state.prepared = false; return clear_host_copy(state);
}
Status set_copy_preflight(CopyRecording &state, CopyPreflight predicate, void *opaque) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple copy predicate slot is retained");
  state.preflight = predicate; state.preflight_state = opaque; return Status::Ok();
}
Result<CompletionPoll> poll_copy_recording(const CopyRecording &state) noexcept {
  if (state.done.load(std::memory_order_acquire) == 0) return CompletionPoll::kPending;
  if (!state.terminal.ok()) return std::unexpected(state.terminal);
  return CompletionPoll::kComplete;
}
Status wait_copy_recording(const CopyRecording &state) noexcept {
  while (state.done.load(std::memory_order_acquire) == 0) platform::detail::wait_on_address(&state.done, 0);
  return state.terminal;
}
Status qualify_recording_dependency(const std::shared_ptr<Runtime> &runtime, cd::CompletionDependency source) noexcept {
  if (source.vtable != &copy_completion_vtable) return qualify_copy_dependency(runtime, source);
  const auto &state = *static_cast<CopyRecording *>(source.state);
  if (state.runtime != runtime) return Status::InvalidArgument("Apple copy producer Runtime differs");
  if (!source.terminal_hint || !source.terminal_hint->ok()) return source.terminal_hint ? *source.terminal_hint : Status::FailedPrecondition("Apple copy producer is empty");
  if (state.done.load(std::memory_order_acquire) != 0 && !state.terminal.ok()) return state.terminal;
  if (state.done.load(std::memory_order_acquire) == 0 &&
      state.cancel.load(std::memory_order_acquire))
    return Status::Cancelled("Apple copy producer cancelled before dependency admission");
  return Status::Ok();
}
} // namespace hyle::compute::apple::detail
