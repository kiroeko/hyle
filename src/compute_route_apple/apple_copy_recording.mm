#include "apple_copy_internal.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <vector>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/event.h>

#include "apple_internal.h"
#include "apple_dependencies.h"
#include "completion_internal.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "apple_device_append_metal.h"
#include "apple_deferred_copy_metal.h"

namespace mlx::core::metal {
MLX_API bool hyle_store_output_fences(Stream, const void *const *, size_t, const Event *, size_t,
    NS::SharedPtr<MTL::Fence> *, size_t, size_t *) noexcept;
}

namespace hyle::compute::apple::detail {
void execute_native_tail(Job *) noexcept;
bool is_store_producer_frontier(const InputSource &, const std::shared_ptr<Runtime> &,
    std::uint64_t, const void *, const NativeStorage &, std::span<const mlx::core::Event>) noexcept;
namespace cd = compute::detail;
struct CopyFence final {
  id<MTLFence> native = nil;
  id<MTLComputePipelineState> checked_copy = nil;
  std::array<DeviceCheckVisit,4096> check_visits{};
  bool encoded = false;
  ~CopyFence() { [native release]; [checked_copy release]; }
};
std::span<DeviceCheckVisit> borrow_copy_check_visits(Runtime &runtime) noexcept {
  HYLE_FATAL_ASSERT(runtime.owner_id.load(std::memory_order_acquire)==platform::this_thread::get_id());
  HYLE_FATAL_ASSERT(runtime.copy_fence);
  auto &visits=runtime.copy_fence->check_visits;visits.fill({});return visits;
}
template<class T> using CopyArray = std::vector<T, mem::StdAllocator<T>>;
struct HostCopyTarget final {
  mem::HostMapping mapping;
  std::span<std::byte> bytes;
};
struct CopyRecording final : Job {
  struct NativeTail final : Job {
    CopyRecording *owner;
    explicit NativeTail(CopyRecording &recording) noexcept : owner(&recording) {}
  } native_tail;
  mem::AllocatorRef allocator;
  explicit CopyRecording(mem::AllocatorRef allocator)
      : native_tail(*this), allocator(allocator), physical(mem::StdAllocator<cd::PhysicalCopyResource>(allocator)),
        inputs(mem::StdAllocator<CopyRange>(allocator)), intervals(mem::StdAllocator<cd::CopyInterval>(allocator)),
        resolved(mem::StdAllocator<cd::ResolvedCopyRange>(allocator)), commands(mem::StdAllocator<CopyCommand>(allocator)),
        host_targets(mem::StdAllocator<HostCopyTarget>(allocator)) {}
  std::shared_ptr<Runtime> runtime;
  NativeDependencies dependencies;
  std::size_t dependency_capacity=0;
  mlx::core::Event completion_event;
  NS::SharedPtr<MTL::CommandBuffer> native_command;
  bool native_encoded = false, native_retired = false;
  Status native_terminal;
  std::size_t dependency_readers = 0;
  std::uint64_t submitted_epoch = 0;
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
  bool static_append_preflight = false;
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
  CopyArray<HostCopyTarget> host_targets;
  std::size_t host_range_count = 0;
  std::array<std::uint64_t, 1> closure{};
  std::shared_ptr<DeviceAppendState> device_append;
  bool device_append_active = false;
  std::shared_ptr<DeferredCopyState> deferred_copy;
  bool deferred_copy_active = false;
};
namespace {
void deliver(cd::CompletionLifetime lifetime, Status terminal, const Status &hint) noexcept {
  if (!lifetime.valid()) return;
  terminal.Update(hint); lifetime.terminal(lifetime.state, terminal); lifetime.release(lifetime.state);
}
void finish(CopyRecording &state, Status terminal) noexcept {
  state.dependencies.clear();
  if(state.deferred_copy)state.deferred_copy->clear();
  cd::CompletionLifetime abandoned; Status hint;
  {
    std::lock_guard lock(state.completion_mutex);
    if (state.source_pinned) {
      InvocationAccess::release_source(state.host_source);
      state.source_pinned = false;
    }
    if (state.device_append) state.device_append->release();
    state.terminal = terminal; abandoned = std::exchange(state.abandoned, {}); hint = state.hint;
    state.done.store(1, std::memory_order_release);
  }
  platform::detail::wake_all(&state.done);
  deliver(abandoned, std::move(terminal), hint);
}
Status observe_copy_prefix(CopyRecording &state) noexcept {
  if (!state.native_encoded || !state.native_command)
    return Status::FailedPrecondition("Apple copy has no original encoded prefix");
  if (state.native_retired) return state.native_terminal;
  id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)state.native_command.get();
  if (command.status != MTLCommandBufferStatusCompleted && command.status != MTLCommandBufferStatusError)
    state.waits.fetch_add(1, std::memory_order_relaxed);
  // Join all original handlers even if status already says Completed: MLX
  // installs native/event error metadata in its own completion handler.
  // This runs at observation/retirement, never at ordinary blit submission.
  [command waitUntilCompleted];
  auto terminal = state.dependencies.observe();
  if(state.deferred_copy_active){
    auto source=InvocationAccess::source_output(state.host_source,state.runtime,state.expected_epoch);
    if(!source)terminal.Update(source.error());
  }
  if (command.status == MTLCommandBufferStatusError) {
    const auto *message = command.error.localizedDescription.UTF8String;
    terminal.Update(Status::Internal(message ? message : "Metal blit command failed"));
  }
  try {state.completion_event.check_error();}
  catch(...) {auto failure=native_exception();state.runtime->poison(failure);terminal.Update(failure);}
  // An event proves device order only. Preserve the original upstream
  // checked verdict before this copy or any dependent can publish success.
  terminal.Update(state.native_terminal);
  if (state.device_append_active) terminal.Update(state.device_append->observe());
  state.native_terminal = std::move(terminal);
  state.native_retired = true;
  return state.native_terminal;
}
void execute_copy_tail(Job *job) noexcept {
  auto &state = *static_cast<CopyRecording::NativeTail *>(job)->owner;
  std::shared_ptr<CopyRecording> owner;
  { std::lock_guard lock(state.completion_mutex); owner = std::move(state.admitted_owner); }
  auto terminal = observe_copy_prefix(state);
  if (state.cancel.load(std::memory_order_acquire))
    terminal.Update(Status::Cancelled("Apple copy cancelled after native retirement"));
  state.native_terminal = terminal;
  state.native_command.reset();
  state.runtime->release_native_tail();
  finish(state, std::move(terminal));
}
// Only the original FIFO owner may resolve a pinned producer. The pin keeps
// its successfully produced array and actual backing alive through finish().
Result<CopySpan> resolve_native_source(CopyRecording &state,DeviceCheckInputs *checks=nullptr) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto output, checks
      ?InvocationAccess::device_source_output(state.host_source,state.runtime,state.expected_epoch,*checks)
      :InvocationAccess::source_output(state.host_source, state.runtime, state.expected_epoch));
  if (output.shape() != state.host_spec.shape || output.dtype() != state.host_spec.dtype ||
      !output.flags().row_contiguous || output.nbytes() != state.host_bytes)
    return std::unexpected(Status::InvalidArgument("Apple native copy producer layout changed"));
  HYLE_ASSIGN_OR_RETURN(auto storage, checks?MemoryAccess::submitted_storage_info(output):MemoryAccess::storage_info(output));
  if (storage.view_bytes != state.host_bytes)
    return std::unexpected(Status::InvalidArgument("Apple native copy producer byte span differs"));
  // The original single-destination API accepts an empty producer without
  // manufacturing a backing or dereferencing its nil Metal buffer.
  if (!state.host_bytes)
    return CopySpan{const_cast<void *>(storage.buffer), {state.device,
        reinterpret_cast<std::uintptr_t>(storage.buffer), storage.offset_bytes, 0, 4}};
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
  if (buffer == nil || buffer.storageMode != MTLStorageModeShared ||
      (state.host_bytes && buffer.contents == nullptr) || storage.offset_bytes > buffer.length ||
      storage.view_bytes != state.host_bytes || storage.view_bytes > buffer.length - storage.offset_bytes ||
      reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device) != state.device)
    return std::unexpected(Status::InvalidArgument("Apple native copy producer backing/Device differs"));
  return CopySpan{const_cast<void *>(storage.buffer), {state.device,
      reinterpret_cast<std::uintptr_t>(storage.buffer), storage.offset_bytes, storage.view_bytes, 4}};
}
bool overlaps(const cd::PhysicalCopyResource &a, const cd::PhysicalCopyResource &b) noexcept {
  if (a.address_space != b.address_space || a.backing != b.backing || !a.capacity || !b.capacity)
    return false;
  return a.origin <= b.origin ? b.origin - a.origin < a.capacity : a.origin - b.origin < b.capacity;
}
Result<CopyBatchReport> validate_host_ranges(CopyRecording &state,
    cd::PhysicalCopyResource source, bool unresolved_source = false) noexcept {
  // This byte alignment describes host access only. copy_span and every
  // existing native blit path retain their original 4-byte contract.
  source.copy_alignment = 1;
  state.physical[0] = source;
  if (!unresolved_source) {
    for (std::size_t index = 0; index < state.host_range_count; ++index)
      if (overlaps(source, state.physical[index + 1]))
        return std::unexpected(Status::InvalidArgument("Apple host ranges overlap their logical source"));
  }
  const std::array<CopyPhase, 1> phases{{{}}};
  const std::array deferred{CopyResourceId{0}};
  return cd::validate_copy_ranges(std::span(state.physical).first(state.host_range_count + 1),
      phases, std::span(state.inputs).first(state.host_range_count), state.limits,
      {state.intervals, state.resolved, state.closure},
      unresolved_source ? std::span<const CopyResourceId>(deferred) : std::span<const CopyResourceId>{});
}
Status execute_host_copy(CopyRecording &state, bool *use_blit) noexcept {
  *use_blit = false;
  const std::byte *source = nullptr;
  CopySpan physical;
  if (state.host_source.valid()) {
    HYLE_ASSIGN_OR_RETURN(physical, resolve_native_source(state));
    if (state.host_bytes) {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)physical.buffer;
      source = static_cast<const std::byte *>(buffer.contents) + physical.physical.origin;
    }
  } else {
    HYLE_RETURN_IF_ERROR(state.host_input.invalidate());
    if (state.host_input.bytes().size() != state.host_bytes)
      return Status::Internal("Apple fixed copy source byte span changed");
    source = state.host_input.bytes().data();
    if (state.host_range_count) {
      physical.physical = state.physical[0];
      physical.buffer = reinterpret_cast<void *>(static_cast<std::uintptr_t>(physical.physical.backing));
    }
  }
  if (state.host_range_count) {
    HYLE_ASSIGN_OR_RETURN(auto report, validate_host_ranges(state, physical.physical));
    // Positive, disjoint exact destinations cannot become no-op copies after
    // resolution. Keep the cold report stable while the public slot is active.
    if (report.active_copies != state.host_range_count)
      return Status::Internal("Apple host range descriptor set changed");
    bool aligned = true;
    for (std::size_t index = 0; index < state.host_range_count; ++index) {
      const auto &range = state.resolved[index];
      const auto &target = state.physical[range.destination];
      auto &command = state.commands[index];
      command = {physical.buffer,
          reinterpret_cast<void *>(static_cast<std::uintptr_t>(target.backing)),
          physical.physical.origin + range.source_offset, target.origin, range.logical_bytes};
      aligned = aligned && command.source_offset % 4 == 0 &&
          command.destination_offset % 4 == 0 && command.bytes % 4 == 0;
      if (state.host_targets[range.input_index].bytes.size() != command.bytes)
        return Status::Internal("Apple host destination mapping changed");
    }
    if (aligned) {
      *use_blit = true;
      return Status::Ok();
    }
    // Every producer/epoch/layout/device/range/alias check above completes
    // before the first byte is changed. This is a legal byte-copy choice,
    // never a recovery from a failed native command or a model recomputation.
    for (std::size_t index = 0; index < state.host_range_count; ++index) {
      const auto &range = state.resolved[index];
      const auto output = state.host_targets[range.input_index].bytes;
      std::memmove(output.data(), source + range.source_offset, output.size());
    }
    Status terminal;
    for (std::size_t index = 0; index < state.host_range_count; ++index)
      terminal.Update(state.host_targets[index].mapping.flush());
    return terminal;
  }
  // Preserve the existing single CPU destination's memmove semantics. Its
  // original producer has already drained on this FIFO; no eval/wait is added.
  if (state.host_bytes) std::memmove(state.host_output_bytes.data(), source, state.host_bytes);
  return state.host_output.flush();
}
// Native shared destinations do not require a CPU observation. Resolve only
// original encoded addresses/error words and preserve the exact cold spans.
// Exact byte spans retain their original nonalias contract.
Status resolve_deferred_copy(CopyRecording &state,bool *selected) noexcept {
  *selected=false;
  auto &copy=*state.deferred_copy;copy.clear();
  auto checks=copy.inputs(borrow_copy_check_visits(*state.runtime));
  HYLE_RETURN_IF_ERROR(state.dependencies.device_checks(checks));
  HYLE_ASSIGN_OR_RETURN(auto source,resolve_native_source(state,&checks));
  copy.check_count=checks.count;
  if(state.host_range_count){
    HYLE_ASSIGN_OR_RETURN(auto report,validate_host_ranges(state,source.physical));
    if(report.active_copies!=state.host_range_count)return Status::Internal("Apple deferred copy descriptor set changed");
    for(std::size_t i=0;i<state.host_range_count;++i){
      const auto &range=state.resolved[i];const auto &target=state.physical[range.destination];
      state.commands[i]={source.buffer,reinterpret_cast<void *>(static_cast<std::uintptr_t>(target.backing)),
          source.physical.origin+range.source_offset,target.origin,range.logical_bytes};
    }
  }else{
    const auto &target=state.physical[1];
    if(overlaps(source.physical,target))return Status::InvalidArgument("Apple device copy source overlaps its destination");
    state.commands[0].source=source.buffer;state.commands[0].source_offset=source.physical.origin;
  }
  *selected=true;return Status::Ok();
}
bool aligned_copy_commands(const CopyRecording &state) noexcept {
  return std::ranges::all_of(std::span(state.commands).first(state.report.active_copies),[](const auto &copy){
    return copy.source_offset%4==0 && copy.destination_offset%4==0 && copy.bytes%4==0;
  });
}
// A cohort borrows earlier accepted copies only while resolving this original
// FIFO's first ready group. Public events are installed after actual encoding.
thread_local std::span<CopyRecording *const> resolving_cohort;

bool cohort_predecessor(const CopyRecording &state) noexcept {
  return std::ranges::find(resolving_cohort, &state) != resolving_cohort.end();
}

class ResolveCohort final {
public:
  explicit ResolveCohort(std::span<CopyRecording *const> prefix) noexcept
      : previous_(std::exchange(resolving_cohort, prefix)) {}
  ~ResolveCohort() { resolving_cohort = previous_; }
private:
  std::span<CopyRecording *const> previous_;
};

bool eligible_store_copy(const CopyRecording &state) noexcept {
  return state.host_copy && state.host_range_count && state.deferred_copy_active &&
      state.source_pinned && !state.device_append_active && !state.preflight &&
      !state.cancel.load(std::memory_order_acquire);
}

struct CohortRange final {
  std::uintptr_t host = 0;
  std::uint64_t gpu = 0;
  std::size_t bytes = 0;
};

bool intersect_ranges(const CohortRange &a, const CohortRange &b) noexcept {
  const auto intersect = [](std::uint64_t x, std::size_t x_size, std::uint64_t y, std::size_t y_size) {
    return x <= y ? y - x < x_size : x - y < y_size;
  };
  return intersect(a.gpu, a.bytes, b.gpu, b.bytes) ||
      intersect(a.host, a.bytes, b.host, b.bytes);
}

bool inspect_cohort_range(void *opaque, std::size_t offset, std::size_t bytes,
                          std::uintptr_t device, CohortRange &range) noexcept {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)opaque;
  if (buffer == nil || !buffer.contents || !buffer.gpuAddress || !bytes ||
      reinterpret_cast<std::uintptr_t>((__bridge void *)buffer.device) != device ||
      offset > buffer.length || bytes > buffer.length - offset)
    return false;
  range = {reinterpret_cast<std::uintptr_t>(buffer.contents) + offset,
           buffer.gpuAddress + offset, bytes};
  return true;
}

bool eligible_static_copy_pair(const CopyRecording &state) noexcept {
  return state.static_append_preflight && state.preflight && !state.host_copy && !state.device_append_active &&
      state.source_pinned && state.host_source.valid() && state.report.active_copies &&
      !state.cancel.load(std::memory_order_acquire);
}

bool completed_copy_or_math_tail(const Job &job) noexcept {
  return job.execute == &execute_copy_tail || job.execute == &execute_native_tail;
}

bool record_original_blits(id<MTLCommandBuffer> command, CopyFence &frontier,
                          std::span<const CopyCommand> copies) {
  id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
  if (blit == nil) return false;
  if (frontier.encoded) [blit waitForFence:frontier.native];
  for (const auto &copy : copies)
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)copy.source sourceOffset:copy.source_offset
        toBuffer:(__bridge id<MTLBuffer>)copy.destination destinationOffset:copy.destination_offset size:copy.bytes];
  [blit updateFence:frontier.native];
  [blit endEncoding];
  return true;
}

struct StaticPageLoan {
  Runtime *runtime;
  MTL::CommandBuffer *command;
  std::array<CopyRecording *, 2> writers;
  unsigned seen = 0;
  bool open = true;
};

thread_local StaticPageLoan *static_page_loan = nullptr;
Status flush_static_page(StaticPageLoan &loan) noexcept {
  if (!std::exchange(loan.open, false)) return Status::Ok();
  try {
    auto &encoder = mlx::core::metal::get_command_encoder(*loan.runtime->stream);
    if (encoder.get_command_buffer() == loan.command) {
      encoder.end_encoding();
      encoder.commit();
    }
    return Status::Ok();
  } catch (...) {
    const auto error = native_exception();
    for (auto *writer : loan.writers) writer->native_terminal.Update(error);
    try { mlx::core::metal::get_command_encoder(*loan.runtime->stream).synchronize(); } catch (...) {}
    loan.runtime->poison(error);
    return error;
  }
}

bool append_static_page_reader(Runtime &runtime, MTL::CommandBuffer *command,
                               CopyRecording &first, CopyRecording &second) noexcept {
  if (!command || static_page_loan) return false;
  Job *reader = nullptr;
  {
    std::lock_guard lock(runtime.queue_mutex);
    if (!runtime.failure.ok()) return false;
    reader = runtime.head;
    if (!reader || !is_static_page_followup(*reader, first, first.submitted_epoch,
                                              second, second.submitted_epoch))
      return false;
    runtime.head = reader->next;
    if (!runtime.head) runtime.tail = nullptr;
  }
  auto loan = StaticPageLoan{&runtime, command, {&first, &second}};
  static_page_loan = &loan;
  reader->execute(reader);
  (void)flush_static_page(loan);
  static_page_loan = nullptr;
  return true;
}

void execute_copy(Job *job) noexcept;
void execute_copy_single(Job *, const Status *resolved = nullptr, bool preflight_done = false) noexcept;

class CopyCohort final {
public:
  static constexpr std::size_t job_capacity = 8;
  static constexpr std::size_t span_capacity = 64;

  explicit CopyCohort(CopyRecording &first) noexcept
      : runtime_(*first.runtime), first_(&first) {}

  bool prepare() noexcept {
    std::array<CopyRecording *, job_capacity> candidates{first_};
    std::size_t candidate_count = 1;
    {
      std::lock_guard lock(runtime_.queue_mutex);
      if (!runtime_.failure.ok()) return false;
      for (auto *next = runtime_.head; next && candidate_count < candidates.size(); next = next->next) {
        if (next->execute != &execute_copy) break;
        auto *copy = static_cast<CopyRecording *>(next);
        if (!eligible_store_copy(*copy)) break;
        candidates[candidate_count++] = copy;
      }
    }
    if (candidate_count < 2) return false;
    for (auto *candidate : std::span(candidates).first(candidate_count)) {
      const ResolveCohort proof{members()};
      if (!append(*candidate)) break;
    }
    if (count_ < 2) return false;

    // A real completed callback may have joined the FIFO during the read-only
    // proof. In that case leave the original order and all jobs intact.
    std::lock_guard lock(runtime_.queue_mutex);
    auto *next = runtime_.head;
    for (std::size_t i = 1; i < count_; ++i) {
      if (next != jobs_[i]) return false;
      next = next->next;
    }
    runtime_.head = next;
    if (!next) runtime_.tail = nullptr;
    return true;
  }

  bool prepare_static() noexcept {
    std::array<Job *, 12> tails{};
    std::size_t tail_count = 0;
    CopyRecording *second = nullptr;
    {
      std::lock_guard lock(runtime_.queue_mutex);
      if (!runtime_.failure.ok()) return false;
      auto *cursor = runtime_.head;
      while (cursor && completed_copy_or_math_tail(*cursor) && tail_count < tails.size()) {
        tails[tail_count++] = cursor;
        cursor = cursor->next;
      }
      if (!cursor || cursor->execute != &execute_copy) return false;
      second = static_cast<CopyRecording *>(cursor);
      if (!eligible_static_copy_pair(*second) || first_->preflight != second->preflight ||
          first_->runtime != second->runtime || first_->expected_epoch != second->expected_epoch ||
          first_->host_source.same_output(second->host_source))
        return false;
    }
    auto first_source = InvocationAccess::source_dependency(first_->host_source, first_->runtime);
    auto second_source = InvocationAccess::source_dependency(second->host_source, second->runtime);
    if (!first_source || !second_source || first_source->state != second_source->state ||
        first_source->vtable != second_source->vtable)
      return false;
    // Both ports live in the same original resolved producer. The exact
    // static-recording preflight resolves only this writer's already pinned
    // source and immutable COW spans. No command has been recorded here.
    if (!second->preflight(second->preflight_state).ok() ||
        !append_disjoint(*first_, {}, nullptr) || !append_disjoint(*second, {}, nullptr))
      return false;
    {
      std::lock_guard lock(runtime_.queue_mutex);
      if (!runtime_.failure.ok()) return false;
      auto *cursor = runtime_.head;
      for (const auto *tail : std::span(tails).first(tail_count)) {
        if (cursor != tail) return false;
        cursor = cursor->next;
      }
      if (cursor != second) return false;
      runtime_.head = second->next;
      if (!runtime_.head) runtime_.tail = nullptr;
    }
    // Execute every original accepted callback in the same order, before a
    // native command is opened for either writer.
    for (auto *tail : std::span(tails).first(tail_count)) tail->execute(tail);
    static_pair_ = true;
    return true;
  }


  void run(bool first_resolved = false) noexcept {
    std::array<Status, job_capacity> resolved;
    bool ready = !jobs_[0]->cancel.load(std::memory_order_acquire);
    if (static_pair_) {
      std::lock_guard lock(runtime_.queue_mutex);
      ready &= runtime_.failure.ok();
    }
    for (std::size_t i = first_resolved ? 1 : 0; i < count_; ++i) {
      const ResolveCohort proof{members().first(i)};
      resolved[i] = jobs_[i]->dependencies.resolve_copy();
      ready &= resolved[i].ok() && !jobs_[i]->cancel.load(std::memory_order_acquire);
    }
    if (!ready || !record()) {
      // The checked dispatch did not start. Original groups retain every
      // captured dependency. The selected FIFO precedes later jobs like the
      // old per-copy paths, including cancellation or a failed source.
      for (std::size_t i = 0; i < count_; ++i) execute_copy_single(jobs_[i], &resolved[i], static_pair_);
    }
  }

private:
  auto members() const noexcept -> std::span<CopyRecording *const> {
    return std::span(jobs_).first(count_);
  }

  bool append(CopyRecording &state) noexcept {
    bool selected = false;
    if (!resolve_deferred_copy(state, &selected).ok() || !selected) return false;
    const auto &guard = *state.deferred_copy;
    CohortRange check{};
    if (guard.check_count > 1 ||
        (guard.check_count && !inspect_cohort_range(const_cast<void *>(guard.words[0].buffer),
            guard.words[0].offset_bytes, 4, state.device, check)) ||
        (count_ && (check.gpu != check_.gpu || check.host != check_.host ||
                    (check.bytes && guard.words[0].buffer != guard_buffer_))))
      return false;
    return append_disjoint(state, check, guard.check_count ? guard.words[0].buffer : nullptr);
  }

  bool append_disjoint(CopyRecording &state, CohortRange check, const void *guard_buffer) noexcept {
    const auto copies = std::span(state.commands).first(state.report.active_copies);
    if (copies.empty() || copies.size() > span_capacity - span_count_ || !aligned_copy_commands(state))
      return false;

    std::size_t next = span_count_;
    for (const auto &copy : copies) {
      auto &source = reads_[next];
      auto &destination = writes_[next];
      if (!inspect_cohort_range(copy.source, copy.source_offset, copy.bytes, state.device, source) ||
          !inspect_cohort_range(copy.destination, copy.destination_offset, copy.bytes, state.device, destination))
        return false;
      bool alias = intersect_ranges(source, destination) ||
          (check.bytes && intersect_ranges(destination, check));
      for (std::size_t previous = 0; previous < next; ++previous)
        alias |= intersect_ranges(destination, reads_[previous]) ||
            intersect_ranges(destination, writes_[previous]) || intersect_ranges(source, writes_[previous]);
      if (alias) return false;
      copies_[next] = copy;
      ++next;
    }
    check_ = check;
    guard_buffer_ = guard_buffer;
    span_count_ = next;
    jobs_[count_++] = &state;
    return true;
  }

  using ProducerFences = std::array<NS::SharedPtr<MTL::Fence>, 8>;
  std::size_t prepare_producer_fences(MTL::CommandBuffer *command, ProducerFences &fences) const noexcept {
    if (!command || !guard_buffer_) return 0;
    const auto &guard = first_->deferred_copy->words[0];
    const auto queue = command->commandQueue();
    const mlx::core::Event *frontier = nullptr;
    const MTL::SharedEvent *physical = nullptr;
    for (auto *job : members()) {
      const auto waits = job->dependencies.events();
      if (waits.empty()) continue;
      if (!frontier) {
        if (!is_store_producer_frontier(job->host_source, job->runtime, job->expected_epoch, queue, guard, waits))
          return 0;
        frontier = &waits.front();
        physical = frontier->cast<mlx::core::metal::EventImpl>().mtl_event();
      }
      if (std::ranges::any_of(waits, [&](const auto &event) {
            return event.value() != frontier->value() ||
                event.template cast<mlx::core::metal::EventImpl>().mtl_event() != physical;
          })) return 0;
    }
    if (!frontier) return 0;
    std::array<const void *, 1 + span_capacity> resources{guard_buffer_};
    std::size_t count = 1;
    for (const auto &copy : std::span(copies_).first(span_count_)) {
      const auto end = resources.begin() + count;
      if (std::find(resources.begin(), end, copy.source) == end)
        resources[count++] = copy.source;
    }
    std::size_t used = 0;
    if (!mlx::core::metal::hyle_store_output_fences(*runtime_.stream, resources.data(), count,
          frontier, 1, fences.data(), fences.size(), &used))
      return 0;
    return used;
  }

  bool record() noexcept {
    std::size_t reserved = 0;
    for (; reserved < count_; ++reserved) {
      if (runtime_.reserve_native_tail().ok()) continue;
      for (std::size_t i = 0; i < reserved; ++i) runtime_.release_native_tail();
      return false;
    }
    bool attached = false;
    @autoreleasepool {
      try {
        auto &stream = *runtime_.stream;
        auto &encoder = mlx::core::metal::get_command_encoder(stream);
        ProducerFences fences;
        encoder.end_encoding();
        const auto fence_count = prepare_producer_fences(encoder.get_command_buffer(), fences);
        if (!fence_count)
          for (auto *job : members()) job->dependencies.encode(stream);
        encoder.end_encoding();
        const auto native = NS::RetainPtr(encoder.get_command_buffer());
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)native.get();
        auto &frontier = *runtime_.copy_fence;
        const auto copies = std::span(copies_).first(span_count_);
        const bool recorded = [&] {
          if (static_pair_) return record_original_blits(command, frontier, copies);
          auto &original = *first_->deferred_copy;
          return original.arguments.length >= copies.size() * 4 * sizeof(std::uint64_t) &&
              original.encode(command, frontier.checked_copy, frontier.native, frontier.encoded, copies,
                              std::span(fences).first(fence_count)).ok();
        }();
        if (!recorded) {
          for (std::size_t i = 0; i < reserved; ++i) runtime_.release_native_tail();
          return false;
        }
        const auto originals = jobs_;
        const auto count = count_;
        [command addCompletedHandler:^(id<MTLCommandBuffer>) {
          // Retain the actual producer fences even when the original command
          // buffer does not retain its resource references.
          (void)fences;
          auto &runtime = *originals[0]->runtime;
          {
            std::lock_guard lock(runtime.queue_mutex);
            HYLE_FATAL_ASSERT(runtime.native_tails >= count);
            for (auto i = count; i != 0; --i) {
              auto &tail = originals[i - 1]->native_tail;
              tail.next = runtime.head;
              runtime.head = &tail;
              if (!runtime.tail) runtime.tail = &tail;
            }
          }
          runtime.wake.unpark();
        }];
        attached = true;
        for (auto *job : members()) {
          job->native_command = native;
          job->native_encoded = true;
        }
        frontier.encoded = true;
        for (auto *job : members()) {
          job->completion_event.set_value(job->submitted_epoch);
          job->completion_event.signal(stream);
        }
        // Charge the physical dispatch once. The remaining original writers
        // still report their own logical spans and completion.
        if (static_pair_)
          for (auto *job : members()) job->calls.fetch_add(job->report.active_copies, std::memory_order_relaxed);
        else
          first_->calls.fetch_add(1, std::memory_order_relaxed);
        if (!static_pair_ || !append_static_page_reader(runtime_, native.get(), *jobs_[0], *jobs_[1]))
          encoder.commit();
        return true;
      } catch (...) {
        auto error = native_exception();
        runtime_.poison(error);
        for (auto *job : members()) job->native_terminal.Update(error);
        try { mlx::core::metal::get_command_encoder(*runtime_.stream).synchronize(); } catch (...) {}
        if (!attached) {
          for (auto *job : members()) {
            std::shared_ptr<CopyRecording> owner;
            { std::lock_guard lock(job->completion_mutex); owner = std::move(job->admitted_owner); }
            runtime_.release_native_tail();
            auto terminal = job->dependencies.observe();
            terminal.Update(error);
            finish(*job, std::move(terminal));
          }
        }
        return true;
      }
    }
  }

  Runtime &runtime_;
  CopyRecording *first_;
  std::array<CopyRecording *, job_capacity> jobs_{};
  std::array<CopyCommand, span_capacity> copies_{};
  std::array<CohortRange, span_capacity> reads_{}, writes_{};
  CohortRange check_{};
  const void *guard_buffer_ = nullptr;
  std::size_t count_ = 0, span_count_ = 0;
  bool static_pair_ = false;
};

void execute_copy(Job *job) noexcept {
  auto &copy = *static_cast<CopyRecording *>(job);
  if (eligible_store_copy(copy)) {
    CopyCohort group(copy);
    if (group.prepare()) {
      group.run();
      return;
    }
  }
  execute_copy_single(job);
}

void execute_copy_single(Job *job, const Status *resolved, bool preflight_done) noexcept {
  auto &state = *static_cast<CopyRecording *>(job);
  std::shared_ptr<CopyRecording> owner;
  { std::lock_guard lock(state.completion_mutex); owner = std::move(state.admitted_owner); }
  Status terminal;
  {
    std::lock_guard lock(state.runtime->queue_mutex); terminal = state.runtime->failure;
  }
  if (state.cancel.load(std::memory_order_acquire)) terminal.Update(Status::Cancelled("Apple copy cancelled before native recording"));
  if (terminal.ok()) terminal.Update(resolved ? *resolved : state.dependencies.resolve_copy());
  bool deferred=false;
  if(terminal.ok() && state.host_copy && state.deferred_copy_active)
    terminal.Update(resolve_deferred_copy(state,&deferred));
  if (terminal.ok() && state.host_copy && !deferred) terminal.Update(state.dependencies.observe());
  if (terminal.ok() && state.device_append_active) terminal.Update(state.device_append->resolve(state.runtime,state.dependencies));
  if (terminal.ok() && state.preflight && !preflight_done)
    terminal.Update(state.preflight(state.preflight_state));
  if (terminal.ok() && !resolved && eligible_static_copy_pair(state)) {
    CopyCohort pair(state);
    if (pair.prepare_static()) {
      { std::lock_guard lock(state.completion_mutex); state.admitted_owner = std::move(owner); }
      pair.run(true);
      return;
    }
  }
  bool use_blit = !state.host_copy || deferred;
  if (terminal.ok() && state.host_copy && !deferred) terminal = execute_host_copy(state, &use_blit);
  // An empty phase with pending order inputs still needs their real native
  // frontier; complete it through the same event/tail without an owner drain.
  if (terminal.ok() && use_blit && (state.device_append_active || state.report.active_copies != 0 || !state.dependencies.empty())) {
    @autoreleasepool {
      bool callback_installed = false;
      bool reserved_tail = false;
      try {
        terminal = state.runtime->reserve_native_tail();
        if (!terminal.ok()) {
          auto prerequisite = state.dependencies.observe(); prerequisite.Update(terminal);
          finish(state, std::move(prerequisite)); return;
        }
        reserved_tail = true;
        auto &encoder = mlx::core::metal::get_command_encoder(*state.runtime->stream);
        state.dependencies.encode(*state.runtime->stream);
        // Event imports order real pending writers before this original blit.
        // A FIFO/host-visible buffer alone does not prove GPU readiness.
        encoder.end_encoding();
        auto native_command = NS::RetainPtr(encoder.get_command_buffer());
        id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)native_command.get();
        auto &frontier = *state.runtime->copy_fence;
        if (state.device_append_active) {
          terminal = state.device_append->encode(command, frontier.native, frontier.encoded);
          if (terminal.ok()) { frontier.encoded = true; state.calls.fetch_add(1, std::memory_order_relaxed); }
        } else if(deferred && (state.deferred_copy->check_count || !aligned_copy_commands(state))){
          terminal=state.deferred_copy->encode(command,frontier.checked_copy,frontier.native,frontier.encoded,
              std::span(state.commands).first(state.report.active_copies));
          if(terminal.ok()){frontier.encoded=true;state.calls.fetch_add(1,std::memory_order_relaxed);}
        } else {
          const auto copies = std::span(state.commands).first(state.report.active_copies);
          if (!record_original_blits(command, frontier, copies))
            terminal = Status::ResourceExhausted("Metal blit command encoder allocation failed");
          else {
            frontier.encoded = true;
            state.calls.fetch_add(copies.size(), std::memory_order_relaxed);
          }
        }
        if (terminal.ok()) {
          state.completion_event.set_value(state.submitted_epoch);
          state.completion_event.signal(*state.runtime->stream);
          state.native_command = std::move(native_command);
          state.native_encoded = true;
          { std::lock_guard lock(state.completion_mutex); state.admitted_owner = std::move(owner); }
          auto *original = &state;
          [command addCompletedHandler:^(id<MTLCommandBuffer>) {
            original->runtime->enqueue_retirement(original->native_tail);
          }];
          callback_installed = true;
          encoder.commit();
          // The original native tail owns the accepted copy and its leases.
          // No main/owner wait, extra queue or second completion is created.
          return;
        }
      } catch (...) {
        terminal = native_exception();
        // The original synchronization waits an accepted prefix before its
        // error check. Drain this same owner/stream if recording threw earlier.
        try { state.waits.fetch_add(1, std::memory_order_relaxed); mlx::core::metal::get_command_encoder(*state.runtime->stream).synchronize(); }
        catch (...) {}
        state.runtime->poison(terminal);
        if (callback_installed) {
          state.native_terminal.Update(terminal);
          // The registered original callback owns retirement of this accepted
          // prefix, including a failure while committing its stream.
          return;
        }
      }
      if (reserved_tail) state.runtime->release_native_tail();
      if (!owner) {
        std::lock_guard lock(state.completion_mutex); owner = std::move(state.admitted_owner);
      }
      state.native_command.reset();
    }
    if (state.cancel.load(std::memory_order_acquire)) terminal.Update(Status::Cancelled("Apple copy cancelled after native retirement"));
  }
  if (state.host_copy && state.cancel.load(std::memory_order_acquire))
    terminal.Update(Status::Cancelled("Apple host copy cancelled after original FIFO retirement"));
  auto prerequisite = state.dependencies.observe(); prerequisite.Update(terminal);
  finish(state, std::move(prerequisite));
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
    state->native_tail.execute = &execute_copy_tail;
    state->completion_event = mlx::core::Event(*runtime->stream);
    state->mode = mode;
    state->device = reinterpret_cast<std::uintptr_t>(mlx::core::metal::device(runtime->stream->device).mtl_device());
    {
      std::lock_guard lock(runtime->queue_mutex);
      if (!runtime->copy_fence) {
        auto frontier = std::make_shared<CopyFence>();
        @autoreleasepool {
          auto device = (__bridge id<MTLDevice>)reinterpret_cast<void *>(state->device);
          frontier->native = [device newFence];
        }
        if (frontier->native == nil)
          return std::unexpected(Status::ResourceExhausted("Metal copy memory fence allocation failed"));
        runtime->copy_fence = std::move(frontier);
      }
    }
    state->limits.max_resources = capacity * 2; state->limits.max_copies = capacity;
    state->limits.max_phases = 1; state->limits.max_bytes = max_bytes;
    state->physical.resize(static_cast<std::size_t>(capacity) * 2); state->inputs.resize(capacity);
    state->intervals.resize(static_cast<std::size_t>(capacity) * 2); state->resolved.resize(capacity); state->commands.resize(capacity);
    state->host_targets.resize(capacity);
    return state;
  } catch (...) { return std::unexpected(native_exception()); }
}
namespace {
Status mutable_recording(const CopyRecording &state) noexcept {
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner || state.dependency_readers)
    return Status::Unavailable("Apple copy phase is retained");
  return state.terminal;
}
Status clear_host_copy(CopyRecording &state) noexcept {
  Status result = state.host_input.close();
  result.Update(state.host_output.close());
  for (std::size_t index = 0; index < state.host_range_count; ++index) {
    result.Update(state.host_targets[index].mapping.close());
    state.host_targets[index].bytes = {};
  }
  state.host_range_count = 0;
  state.host_source.reset(); state.host_copy = false; state.host_output_bytes = {};
  state.static_append_preflight = false;
  state.deferred_copy_active=false;
  if(state.deferred_copy)state.deferred_copy->clear();
  state.device_append_active = false;
  if (state.device_append) { state.device_append->inputs = {}; state.device_append->bound = false; }
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
Status prepare_deferred_copy(CopyRecording &state) noexcept {
  state.prepared=false;
  state.deferred_copy_active=false;
  try {
  if(!state.deferred_copy)
    state.deferred_copy=std::allocate_shared<DeferredCopyState>(mem::StdAllocator<DeferredCopyState>(state.allocator));
  if(state.dependency_capacity==SIZE_MAX)return Status::OutOfRange("Apple device copy source capacity overflows");
  HYLE_RETURN_IF_ERROR(state.dependencies.reserve(state.dependency_capacity+1));
  std::lock_guard lock(state.runtime->queue_mutex);
  @autoreleasepool {
    auto device=(__bridge id<MTLDevice>)reinterpret_cast<void *>(state.device);
    // The first original recording supplies the group's descriptor storage.
    // Its own capacity still suffices if a wider buffer cannot be prepared.
    auto &pipeline = state.runtime->copy_fence->checked_copy;
    const auto capacity = std::size_t(state.limits.max_copies);
    const auto group_capacity = (std::max)(capacity, CopyCohort::span_capacity);
    auto prepared = state.deferred_copy->prepare(device, group_capacity, pipeline);
    if (!prepared && prepared.error().IsResourceExhausted() && group_capacity != capacity)
      prepared = state.deferred_copy->prepare(device, capacity, pipeline);
    HYLE_ASSIGN_OR_RETURN(state.deferred_copy_active, std::move(prepared));
  }
  state.prepared=true;
  return Status::Ok();
  }catch(...){return native_exception();}
}
}
Result<bool> prepare_device_copy_recording(CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  if(state.prepared || state.host_copy || state.device_append)
    return std::unexpected(Status::FailedPrecondition("Apple device-copy preparation needs an unbound original recording"));
  const auto status=prepare_deferred_copy(state);
  const auto available=state.deferred_copy_active;
  state.prepared=false;state.deferred_copy_active=false;
  HYLE_RETURN_IF_ERROR(status);
  return available;
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
Status bind_device_copy_recording(CopyRecording &state,InputSource source,
    const TensorDesc &description,mem::BufferView destination) noexcept {
  HYLE_RETURN_IF_ERROR(bind_host_copy_recording(state,std::move(source),description,destination));
  std::lock_guard lock(state.completion_mutex);state.prepared=false;
  const auto bytes=state.host_bytes;
  if(!bytes || !is_copy_resource(destination.resource()))
    return Status::Unimplemented("Apple device copy requires a positive native destination");
  HYLE_ASSIGN_OR_RETURN(auto range,destination.subview(0,bytes));
  HYLE_ASSIGN_OR_RETURN(auto target,copy_span(range,mem::HostAccess::kWrite));
  if(target.physical.address_space!=state.device)return Status::InvalidArgument("Apple device copy destination Device differs");
  state.physical[1]=target.physical;
  state.commands[0]={nullptr,target.buffer,0,target.physical.origin,bytes};
  HYLE_RETURN_IF_ERROR(prepare_deferred_copy(state));
  if(!state.deferred_copy_active){state.prepared=false;return Status::Unimplemented("Apple device copy requires argument-buffer tier2");}
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
namespace {
Status bind_host_ranges(CopyRecording &state, std::size_t source_bytes,
    const cd::PhysicalCopyResource *source, std::span<const HostCopyRange> ranges) noexcept {
  state.prepared = false;
  HYLE_RETURN_IF_ERROR(clear_host_copy(state));
  if (ranges.empty()) return Status::InvalidArgument("Apple host ranges are empty");
  if (ranges.size() > state.limits.max_copies)
    return Status::ResourceExhausted("Apple host range capacity exhausted");
  state.host_range_count = ranges.size();
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const auto &range = ranges[index];
    if (!range.destination.size()) return Status::InvalidArgument("Apple host range is empty");
    HYLE_ASSIGN_OR_RETURN(auto destination, copy_span(range.destination, mem::HostAccess::kWrite));
    if (destination.physical.address_space != state.device)
      return Status::InvalidArgument("Apple host range destination Device differs");
    destination.physical.copy_alignment = 1;
    state.physical[index + 1] = destination.physical;
    state.inputs[index] = {{0}, {static_cast<std::uint32_t>(index + 1)}, {0},
        range.source_offset, 0, range.destination.size()};
  }
  const auto physical = source ? *source : cd::PhysicalCopyResource{.capacity = source_bytes};
  HYLE_ASSIGN_OR_RETURN(auto report, validate_host_ranges(state, physical, source == nullptr));
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    auto &target = state.host_targets[index];
    HYLE_ASSIGN_OR_RETURN(target.mapping, ranges[index].destination.map_host(mem::HostAccess::kWrite));
    HYLE_ASSIGN_OR_RETURN(target.bytes, target.mapping.writable_bytes());
    if (target.bytes.size() != ranges[index].destination.size())
      return Status::Internal("Apple host range mapping byte span differs");
  }
  state.host_bytes = source_bytes;
  state.report = report; state.report.native_metadata_unknown = true;
  state.prefix_marker = true; state.prepared = true; state.host_copy = true;
  state.calls.store(0, std::memory_order_relaxed); state.waits.store(0, std::memory_order_relaxed);
  return Status::Ok();
}
Status discard_failed_host_ranges(CopyRecording &state, Status status) noexcept {
  if (!status.ok()) {
    state.prepared = false;
    status.Update(clear_host_copy(state));
  }
  return status;
}
} // namespace
Status bind_host_copy_ranges(CopyRecording &state, InputSource source,
    const TensorDesc &description, std::span<const HostCopyRange> ranges) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  state.prepared = false;
  if (!source.valid() || (description.is_plain() && !description.is_dense()))
    return Status::InvalidArgument("Apple host ranges require their dense native producer");
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto epoch, InvocationAccess::source_epoch(source, state.runtime));
  HYLE_RETURN_IF_ERROR(discard_failed_host_ranges(state, bind_host_ranges(state, bytes, nullptr, ranges)));
  state.host_source = std::move(source); state.host_spec = std::move(spec);
  state.consumed_epoch = epoch;
  return Status::Ok();
}
Status bind_device_copy_ranges(CopyRecording &state,InputSource source,
    const TensorDesc &description,std::span<const HostCopyRange> ranges) noexcept {
  HYLE_RETURN_IF_ERROR(bind_host_copy_ranges(state,std::move(source),description,ranges));
  std::lock_guard lock(state.completion_mutex);state.prepared=false;
  HYLE_RETURN_IF_ERROR(prepare_deferred_copy(state));
  if(!state.deferred_copy_active){state.prepared=false;return Status::Unimplemented("Apple device copy requires argument-buffer tier2");}
  return Status::Ok();
}
Status bind_host_copy_ranges(CopyRecording &state, ConstTensorView source,
    std::span<const HostCopyRange> ranges, ReadLease source_owner) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  state.prepared = false;
  if (!source.valid() || (source.description().is_plain() && !source.description().is_dense()))
    return Status::InvalidArgument("Apple fixed host ranges require a dense tensor");
  HYLE_ASSIGN_OR_RETURN(auto bytes, source.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto range, source.storage().subview(0, bytes));
  HYLE_ASSIGN_OR_RETURN(auto physical, copy_span(range, mem::HostAccess::kRead));
  if (physical.physical.address_space != state.device)
    return Status::InvalidArgument("Apple fixed host ranges source Device differs");
  HYLE_ASSIGN_OR_RETURN(auto input, range.map_host());
  HYLE_RETURN_IF_ERROR(discard_failed_host_ranges(state, bind_host_ranges(state, bytes, &physical.physical, ranges)));
  state.host_input = std::move(input); state.host_owner = std::move(source_owner);
  return Status::Ok();
}

Status copy_recording_reuse_gate(const CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  return mutable_recording(state);
}
Status copy_recording_retired_gate(const CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  // Downstream readers retain the completed episode's epoch/verdict, not
  // these already-consumed input mappings. Rebinding/submission still uses
  // mutable_recording and cannot overwrite that episode while readers exist.
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
  return resolve_native_source(state);
}
Status bind_copy_recording(CopyRecording &state, std::span<const cd::CopyBackendRange> ranges, bool prefix_marker) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner || state.admitted_owner || state.dependency_readers)
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
  return submit_copy_recording(state, {});
}
Status reserve_copy_dependencies(CopyRecording &state, std::size_t capacity) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  if(capacity>SIZE_MAX-3)return Status::OutOfRange("Apple copy dependency capacity overflows");
  HYLE_RETURN_IF_ERROR(state.dependencies.reserve(capacity+(state.device_append?3u:state.deferred_copy?1u:0u)));
  state.dependency_capacity=capacity;return Status::Ok();
}
Result<Completion> submit_copy_recording(CopyRecording &state,
    std::span<const cd::CompletionDependency> dependencies) noexcept {
  struct Admission final {
    CopyRecording &state;
    bool accepted = false, capturing = false;
    ~Admission() {
      if (accepted || !capturing) return;
      if (state.source_pinned) {
        InvocationAccess::release_source(state.host_source); state.source_pinned = false;
      }
      if (state.device_append) state.device_append->release();
      state.dependencies.clear();
    }
  } admission{state};
  std::lock_guard lock(state.completion_mutex);
  if (!state.prepared || state.completion_owner || state.dependency_readers || state.done.load(std::memory_order_acquire) == 0)
    return std::unexpected(Status::FailedPrecondition("Apple copy phase is not prepared/idle"));
  if (!state.terminal.ok()) return std::unexpected(state.terminal);
  if (state.submitted_epoch == UINT64_MAX)
    return std::unexpected(Status::ResourceExhausted("Apple copy epoch exhausted"));
  admission.capturing = true;
  HYLE_RETURN_IF_ERROR(state.dependencies.capture(state.runtime, dependencies, nullptr, &state));
  if (state.host_source.valid()) {
    HYLE_ASSIGN_OR_RETURN(state.expected_epoch, InvocationAccess::pin_source(
        state.host_source, state.runtime, state.consumed_epoch));
    state.source_pinned = true;
    if(state.deferred_copy_active){
      HYLE_ASSIGN_OR_RETURN(auto dependency,InvocationAccess::source_dependency(state.host_source,state.runtime));
      HYLE_RETURN_IF_ERROR(state.dependencies.append(state.runtime,std::span(&dependency,1),nullptr,&state));
    }
  }
  if (state.device_append_active) {
    HYLE_RETURN_IF_ERROR(state.device_append->pin(state.runtime));
    for(const auto &input:state.device_append->inputs)if(input.source.valid()){
      HYLE_ASSIGN_OR_RETURN(auto dependency,InvocationAccess::source_dependency(input.source,state.runtime));
      HYLE_RETURN_IF_ERROR(state.dependencies.append(state.runtime,std::span(&dependency,1),nullptr,&state));
    }
  }
  if (state.mode == CopyRecordingMode::kOnce) state.prepared = false;
  if (state.report.active_copies == 0 && !state.prefix_marker && state.dependencies.empty())
    return Completion::completed();
  auto owner = state.self.lock(); HYLE_FATAL_ASSERT(owner != nullptr);
  state.cancel.store(false, std::memory_order_relaxed); state.terminal = Status::Ok(); state.hint = Status::Ok();
  state.native_encoded = state.native_retired = false; state.native_terminal = Status::Ok();
  state.admitted_owner = owner; state.completion_owner = owner;
  const auto previous_epoch = state.submitted_epoch++;
  state.done.store(0, std::memory_order_release);
  const auto admitted = state.runtime->enqueue(state);
  if (!admitted.ok()) {
    if (state.source_pinned) {
      InvocationAccess::release_source(state.host_source); state.source_pinned = false;
    }
    if (state.device_append) state.device_append->release();
    state.done.store(1, std::memory_order_release); state.admitted_owner.reset(); state.completion_owner.reset();
    state.submitted_epoch = previous_epoch;
    return std::unexpected(admitted);
  }
  if (state.source_pinned) state.consumed_epoch = state.expected_epoch;
  if (state.device_append_active) state.device_append->accepted();
  state.report.phase_completions = 1;
  admission.accepted = true;
  return cd::CompletionAccess::make(&copy_completion_vtable, &state);
}
Status bind_result_copy_recording(CopyRecording&state,std::span<const cd::CopyBackendRange>ranges)noexcept{
  std::lock_guard lock(state.completion_mutex);
  if(state.done.load(std::memory_order_acquire)==0 || state.completion_owner || state.admitted_owner || state.dependency_readers)
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
  // Keep submitted_epoch, terminal and reader ownership intact. This only
  // releases inputs after real retirement; mutable_recording still gates
  // every replacement or subsequent submission against the last reader.
  state.prepared = false; return clear_host_copy(state);
}
namespace {
Status set_recording_preflight(CopyRecording &state, CopyPreflight predicate, void *opaque,
                               bool static_append) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if (state.done.load(std::memory_order_acquire) == 0 || state.completion_owner ||
      state.admitted_owner || state.dependency_readers)
    return Status::Unavailable("Apple copy predicate slot is retained");
  if (static_append && (!predicate || !opaque || !state.host_source.valid() || state.host_copy ||
                        state.device_append_active))
    return Status::InvalidArgument("Apple static append requires its bound pinned rows preflight");
  state.preflight = predicate;
  state.preflight_state = opaque;
  state.static_append_preflight = static_append;
  return Status::Ok();
}
} // namespace

Status set_copy_preflight(CopyRecording &state, CopyPreflight predicate, void *opaque) noexcept {
  return set_recording_preflight(state, predicate, opaque, false);
}

Status set_static_append_preflight(CopyRecording &state, CopyPreflight predicate, void *opaque) noexcept {
  return set_recording_preflight(state, predicate, opaque, true);
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
Result<std::shared_ptr<CopyRecording>> pin_copy_dependency(const std::shared_ptr<Runtime> &runtime,
    cd::CompletionDependency source, std::uint64_t *epoch) noexcept {
  if (!epoch || source.vtable != &copy_completion_vtable || !source.state)
    return std::unexpected(Status::Unimplemented("Apple dependency is not an original copy episode"));
  auto &state = *static_cast<CopyRecording *>(source.state);
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(qualify_recording_dependency(runtime, source));
  if (!state.submitted_epoch || !state.completion_owner)
    return std::unexpected(Status::FailedPrecondition("Apple copy dependency has no retained accepted episode"));
  if (state.dependency_readers == SIZE_MAX)
    return std::unexpected(Status::ResourceExhausted("Apple copy dependency reader capacity"));
  auto owner = state.self.lock();
  if (!owner) return std::unexpected(Status::FailedPrecondition("Apple copy dependency owner expired"));
  ++state.dependency_readers; *epoch = state.submitted_epoch;
  return owner;
}
Status observe_copy_dependency(const CopyRecording &state, std::uint64_t epoch) noexcept {
  if (state.runtime->owner_id.load(std::memory_order_acquire) != platform::this_thread::get_id())
    return Status::FailedPrecondition("Apple copy dependency observation requires its original FIFO");
  {
    std::lock_guard lock(state.completion_mutex);
    if (state.submitted_epoch != epoch || !state.dependency_readers)
      return Status::FailedPrecondition("Apple copy dependency episode was replaced");
    if (state.done.load(std::memory_order_acquire)) return state.terminal;
  }
  return observe_copy_prefix(const_cast<CopyRecording &>(state));
}
Status require_static_append_for_page(CopyRecording &state) noexcept {
  if (auto *loan = static_page_loan; loan && loan->open && loan->runtime == state.runtime.get()) {
    for (std::size_t i = 0; i < loan->writers.size(); ++i) {
      if (loan->writers[i] != &state) continue;
      // This is the actual page preflight of the admitted reader only. The
      // reader validates its original events before any native evaluation.
      if (state.static_append_preflight && state.native_encoded && state.native_command.get() == loan->command &&
          state.native_terminal.ok() && !state.cancel.load(std::memory_order_acquire)) {
        loan->seen |= 1u << i;
        return Status::Ok();
      }
    }
    HYLE_RETURN_IF_ERROR(flush_static_page(*loan));
  }
  return observe_copy_recording_on_owner(state);
}

Status observe_copy_recording_on_owner(CopyRecording &state) noexcept {
  if (auto *loan = static_page_loan; loan && loan->open && loan->runtime == state.runtime.get())
    HYLE_RETURN_IF_ERROR(flush_static_page(*loan));
  {
    std::lock_guard lock(state.completion_mutex);
    if(state.done.load(std::memory_order_acquire)) return state.terminal;
  }
  if (state.runtime->owner_id.load(std::memory_order_acquire) != platform::this_thread::get_id())
    return Status::Unavailable("Apple copy prefix still requires its original FIFO observer");
  return observe_copy_prefix(state);
}
Status verify_static_page_loan(Runtime &runtime, std::span<mlx::core::Event> events) noexcept {
  auto *loan = static_page_loan;
  if (!loan || loan->runtime != &runtime) return Status::Ok();
  const auto original = [&](const CopyRecording *writer) {
    return !writer->cancel.load(std::memory_order_acquire) &&
        std::ranges::any_of(events, [&](const auto &event) {
          return event.valid() && writer->completion_event.valid() &&
              event.value() == writer->submitted_epoch &&
              event.template cast<mlx::core::metal::EventImpl>().mtl_event() ==
                  writer->completion_event.cast<mlx::core::metal::EventImpl>().mtl_event();
        });
  };
  if (loan->open && loan->seen == 3 && std::ranges::all_of(loan->writers, original))
    return Status::Ok();
  HYLE_RETURN_IF_ERROR(flush_static_page(*loan));
  for (std::size_t i = 0; i < loan->writers.size(); ++i) {
    auto &writer = *loan->writers[i];
    if (writer.cancel.load(std::memory_order_acquire))
      return Status::Cancelled("Apple page writer cancelled before combined read");
    if (loan->seen & (1u << i)) HYLE_RETURN_IF_ERROR(observe_copy_prefix(writer));
  }
  return Status::Ok();
}

Status finish_static_page_loan(Runtime &runtime) noexcept {
  auto *loan = static_page_loan;
  if (!loan || loan->runtime != &runtime) return Status::Ok();
  return flush_static_page(*loan);
}

Status observe_copy_inputs_on_owner(CopyRecording &state) noexcept {
  if (state.runtime->owner_id.load(std::memory_order_acquire) != platform::this_thread::get_id())
    return Status::FailedPrecondition("Apple copy input observation requires its original FIFO");
  return state.dependencies.observe();
}
Result<mlx::core::Event> copy_dependency_event(const CopyRecording &state, std::uint64_t epoch) noexcept {
  if (state.runtime->owner_id.load(std::memory_order_acquire) != platform::this_thread::get_id())
    return std::unexpected(Status::FailedPrecondition("Apple copy event requires its original FIFO"));
  std::lock_guard lock(state.completion_mutex);
  if (state.submitted_epoch != epoch || !state.dependency_readers)
    return std::unexpected(Status::FailedPrecondition("Apple copy event episode was replaced"));
  if (state.done.load(std::memory_order_acquire)) {
    if (!state.terminal.ok()) return std::unexpected(state.terminal);
    return mlx::core::Event{};
  }
  if (cohort_predecessor(state)) return mlx::core::Event{};
  if (!state.native_encoded || !state.native_command || state.completion_event.value() != epoch)
    return std::unexpected(Status::FailedPrecondition("Apple copy has no original pending event"));
  if (state.native_retired && !state.native_terminal.ok()) return std::unexpected(state.native_terminal);
  return state.completion_event;
}
void release_copy_dependency(CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_FATAL_ASSERT(state.dependency_readers); --state.dependency_readers;
}
Status prepare_device_append_recording(CopyRecording &state,
    cd::DeviceLengthsPagedAppendSpec geometry,const TensorDesc &rows) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  if (std::uint64_t(geometry.storage.instruction_capacity) > std::uint64_t(state.limits.max_copies)+geometry.requests+1)
    return Status::ResourceExhausted("Apple device append exceeds the original recording's metadata capacity");
  try {
    auto prepared=std::allocate_shared<DeviceAppendState>(mem::StdAllocator<DeviceAppendState>(state.allocator),state.allocator);
    HYLE_RETURN_IF_ERROR(prepared->prepare(state.runtime,geometry,rows));
    if(state.dependency_capacity>SIZE_MAX-3)return Status::OutOfRange("Apple append dependency capacity overflows");
    HYLE_RETURN_IF_ERROR(state.dependencies.reserve(state.dependency_capacity+3));
    HYLE_RETURN_IF_ERROR(clear_host_copy(state));
    state.prepared=false;state.device_append=std::move(prepared);
    return Status::Ok();
  }catch(...){return native_exception();}
}
Status configure_device_append_recording(CopyRecording &state,mem::ConstBufferView arena,
    std::span<const cd::DeviceAppendRequest> requests,std::span<const DeviceAppendPageBinding> pages) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));
  if(!state.device_append)return Status::FailedPrecondition("Apple device append original recording is not prepared");
  if (pages.size()>state.limits.max_copies)
    return Status::ResourceExhausted("Apple device append exceeds the original page capacity");
  std::size_t bytes=0;
  for (const auto &page:pages) {
    if (page.destination.size()>state.limits.max_bytes-bytes)
      return Status::ResourceExhausted("Apple device append exceeds the original byte capacity");
    bytes+=page.destination.size();
  }
  state.prepared=false;HYLE_RETURN_IF_ERROR(clear_host_copy(state));
  HYLE_RETURN_IF_ERROR(state.device_append->configure(state.device,arena,requests,pages));
  state.device_append_active=true;state.host_copy=false;state.prefix_marker=true;
  state.report={};state.report.active_copies=static_cast<std::uint32_t>(pages.size());
  for(const auto &page:pages)state.report.logical_bytes+=page.destination.size();
  state.report.physical_bytes=state.report.logical_bytes;state.report.native_metadata_unknown=true;
  state.calls.store(0,std::memory_order_relaxed);state.waits.store(0,std::memory_order_relaxed);
  return Status::Ok();
}
Status bind_device_append_inputs(CopyRecording &state,std::span<const InputBinding> inputs) noexcept {
  std::lock_guard lock(state.completion_mutex);
  HYLE_RETURN_IF_ERROR(mutable_recording(state));state.prepared=false;
  if(!state.device_append_active)return Status::FailedPrecondition("Apple device append has no original capacity episode");
  HYLE_RETURN_IF_ERROR(state.device_append->bind(state.runtime,inputs));state.prepared=true;return Status::Ok();
}
Result<ConstTensorView> device_append_status(CopyRecording &state) noexcept {
  std::lock_guard lock(state.completion_mutex);
  if(!state.device_append_active)return std::unexpected(Status::FailedPrecondition("Apple device append is not selected"));
  return state.device_append->status_view();
}
Status copy_device_checks(CopyRecording &state,DeviceCheckInputs &checks) noexcept {
  if(state.done.load(std::memory_order_acquire))return state.terminal;
  HYLE_RETURN_IF_ERROR(state.native_terminal);
  if(state.native_retired)return Status::Ok();
  if(!state.native_encoded&&!cohort_predecessor(state))return Status::FailedPrecondition("Apple copy check has no original encoded prefix");
  HYLE_ASSIGN_OR_RETURN(auto fresh,visit_device_check(checks,&state,state.submitted_epoch));
  if(!fresh)return Status::Ok();
  HYLE_RETURN_IF_ERROR(state.dependencies.device_checks(checks));
  if(state.host_source.valid()){
    auto source=InvocationAccess::device_source_output(state.host_source,state.runtime,state.expected_epoch,checks);
    if(!source)return source.error();
  }
  if(state.device_append_active){
    HYLE_ASSIGN_OR_RETURN(auto verdict,state.device_append->status_view());
    HYLE_ASSIGN_OR_RETURN(auto storage,copy_span(verdict.storage(),mem::HostAccess::kRead));
    HYLE_RETURN_IF_ERROR(append_device_check(checks,{storage.buffer,storage.physical.origin,sizeof(std::int32_t),0}));
  }
  return Status::Ok();
}
} // namespace hyle::compute::apple::detail
