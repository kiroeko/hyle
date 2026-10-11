#include "apple_copy_internal.h"

#import <Metal/Metal.h>

#include <array>
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
  bool prepared = false;
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
    state.terminal = terminal; abandoned = std::exchange(state.abandoned, {}); hint = std::move(state.hint);
    state.done.store(1, std::memory_order_release);
  }
  platform::detail::wake_all(&state.done);
  deliver(abandoned, std::move(terminal), hint);
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
  if (terminal.ok()) {
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
    std::uint32_t capacity, mem::AllocatorRef allocator, std::size_t max_bytes) noexcept {
  if (!runtime || !allocator.valid() || capacity == 0 || capacity > UINT32_MAX / 2 || max_bytes == 0)
    return std::unexpected(Status::InvalidArgument("Apple copy limits are invalid"));
  { std::lock_guard lock(runtime->queue_mutex);
    if (!runtime->accepting) return std::unexpected(Status::FailedPrecondition("Apple copy Context is closed"));
    if (!runtime->failure.ok()) return std::unexpected(runtime->failure); }
  try {
    auto state = std::allocate_shared<CopyRecording>(mem::StdAllocator<CopyRecording>(allocator), allocator);
    state->self = state;
    state->runtime = runtime; state->execute = &execute_copy;
    state->device = reinterpret_cast<std::uintptr_t>(mlx::core::metal::device(runtime->stream->device).mtl_device());
    state->limits.max_resources = capacity * 2; state->limits.max_copies = capacity;
    state->limits.max_phases = 1; state->limits.max_bytes = max_bytes;
    state->physical.resize(static_cast<std::size_t>(capacity) * 2); state->inputs.resize(capacity);
    state->intervals.resize(static_cast<std::size_t>(capacity) * 2); state->resolved.resize(capacity); state->commands.resize(capacity);
    return state;
  } catch (...) { return std::unexpected(native_exception()); }
}
Status bind_copy_recording(CopyRecording &state, std::span<const cd::CopyBackendRange> ranges) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple copy phase is retained");
  if (!state.terminal.ok()) return state.terminal;
  state.prepared = false;
  if (ranges.size() > state.limits.max_copies) return Status::ResourceExhausted("Apple copy range capacity exhausted");
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    if (ranges[index].source.size() != ranges[index].destination.size()) return Status::InvalidArgument("Apple copy sizes differ");
    HYLE_ASSIGN_OR_RETURN(auto source, copy_span(ranges[index].source));
    HYLE_ASSIGN_OR_RETURN(auto destination, copy_span(ranges[index].destination));
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
  state.prepared = false;
  if (state.report.active_copies == 0) return Completion::completed();
  auto owner = state.self.lock(); HYLE_FATAL_ASSERT(owner != nullptr);
  state.cancel.store(false, std::memory_order_relaxed); state.terminal = Status::Ok(); state.hint = Status::Ok();
  state.admitted_owner = owner; state.completion_owner = owner;
  state.done.store(0, std::memory_order_release);
  const auto admitted = state.runtime->enqueue(state);
  if (!admitted.ok()) {
    state.done.store(1, std::memory_order_release); state.admitted_owner.reset(); state.completion_owner.reset();
    return std::unexpected(admitted);
  }
  state.report.phase_completions = 1;
  return cd::CompletionAccess::make(&copy_completion_vtable, &state);
}
CopyBatchReport copy_recording_report(const CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  auto report = state.report;
  report.native_calls = state.calls.load(std::memory_order_relaxed);
  report.host_waits = state.waits.load(std::memory_order_relaxed);
  return report;
}
Status clear_copy_recording(CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner)
    return Status::Unavailable("Apple copy phase is retained");
  state.prepared = false; return Status::Ok();
}
Status qualify_recording_dependency(const std::shared_ptr<Runtime> &runtime, cd::CompletionDependency source) noexcept {
  if (source.vtable != &copy_completion_vtable) return qualify_copy_dependency(runtime, source);
  const auto &state = *static_cast<CopyRecording *>(source.state);
  if (state.runtime != runtime) return Status::InvalidArgument("Apple copy producer Runtime differs");
  if (!source.terminal_hint || !source.terminal_hint->ok()) return source.terminal_hint ? *source.terminal_hint : Status::FailedPrecondition("Apple copy producer is empty");
  if (state.done.load(std::memory_order_acquire) != 0 && !state.terminal.ok()) return state.terminal;
  return Status::Ok();
}
} // namespace hyle::compute::apple::detail
