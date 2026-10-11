#include "hyle/compute/batch_transfer.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "context_state.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/compute/detail/copy_backend.h"
#include "hyle/compute/detail/copy_control.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/compute/transfer.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::detail {
template<class T> using CopyArray = std::vector<T, mem::StdAllocator<T>>;

struct CopyProgramState final : util::RefCounted<CopyProgramState> {
  explicit CopyProgramState(mem::AllocatorRef base, CopyBatchOptions selected) noexcept
      : metadata{base}, options(std::move(selected)), specs(mem::StdAllocator<CopyResourceSpec>(metadata.allocator())) {}
  void destroy() noexcept {
    const auto base = metadata.base;
    if (native) table->destroy_program(native, metadata.allocator());
    release(context);
    std::destroy_at(this);
    base.deallocate(this, sizeof(CopyProgramState), alignof(CopyProgramState));
  }
  CopyMetadata metadata;
  CopyBatchOptions options;
  ContextState *context = nullptr;
  BackendExtensionRef extension;
  ResourceBudget::Reservation opaque_native;
  const CopyBackendVTable *table = nullptr;
  void *native = nullptr;
  CopyArray<CopyResourceSpec> specs;
};

struct CopyPhaseState final {
  CopyInvocationState *owner = nullptr;
  Completion completion;
  CompletionRetention retention;
  Status terminal;
  std::atomic<bool> retired{false};
  std::size_t first = 0, copies = 0, use_first = 0, use_count = 0, producer_first = 0, producer_count = 0;
  std::uint64_t dependency_mask = 0;
};
struct CopyInvocationState final : util::RefCounted<CopyInvocationState> {
  explicit CopyInvocationState(CopyProgramState *selected, Queue *endpoint) noexcept
      : program(selected), queue(endpoint), bindings(mem::StdAllocator<CopyBinding>(allocator())),
        physical(mem::StdAllocator<PhysicalCopyResource>(allocator())),
        intervals(mem::StdAllocator<CopyInterval>(allocator())), copies(mem::StdAllocator<ResolvedCopyRange>(allocator())),
        closure(mem::StdAllocator<std::uint64_t>(allocator())), phases(mem::StdAllocator<CopyPhaseState *>(allocator())),
        plans(mem::StdAllocator<CopyPhasePlan>(allocator())), root_order(mem::StdAllocator<std::uint32_t>(allocator())), root_leader(mem::StdAllocator<std::uint32_t>(allocator())),
        last_write(mem::StdAllocator<std::uint32_t>(allocator())), producer_order(mem::StdAllocator<std::uint32_t>(allocator())),
        roles(mem::StdAllocator<unsigned char>(allocator())),
        root_uses(mem::StdAllocator<CopyRootUse>(allocator())), written(mem::StdAllocator<CopyWrittenSpan>(allocator())),
        native_ranges(mem::StdAllocator<CopyBackendRange>(allocator())), tickets(mem::StdAllocator<LeaseUseBatch>(allocator())) {
    program->incref();
  }
  mem::AllocatorRef allocator() const noexcept { return program->metadata.allocator(); }
  void destroy() noexcept {
    auto *retained = program;
    const auto selected = allocator();
    if (native) retained->table->destroy_slot(native, selected);
    for (auto *phase : phases) {
      std::destroy_at(phase);
      selected.deallocate(phase, sizeof(CopyPhaseState), alignof(CopyPhaseState));
    }
    std::destroy_at(this);
    selected.deallocate(this, sizeof(CopyInvocationState), alignof(CopyInvocationState));
    retained->decref();
  }
  bool busy() const noexcept { return pending.load(std::memory_order_acquire) != 0; }
  Status mutable_slot() const noexcept {
    if (closed) return Status::FailedPrecondition("copy invocation is closed");
    if (users != 1 || busy()) return Status::Unavailable("copy submission or native work retains the slot");
    if (failed) return Status::FailedPrecondition("failed copy slot requires explicit reset");
    return Status::Ok();
  }
  void release_user() noexcept {
    HYLE_FATAL_ASSERT(users != 0);
    if (--users == 0) {
      // Break observation cycles while this user reference still protects the
      // descriptor owner. Original native destroy retains late attachments.
      for (auto *phase : phases) phase->completion = {};
    }
    decref();
  }
  CopyProgramState *program;
  Queue *queue;
  void *native = nullptr;
  LeaseUsePool uses;
  ResourceBudget::Reservation opaque_native;
  std::atomic<std::uint32_t> pending{0};
  std::uint32_t users = 1, phase_count = 0, submitted = 0;
  bool prepared = false, closed = false, failed = false;
  Status tail;
  CopyBatchReport report;
  CopyArray<CopyBinding> bindings;
  CopyArray<PhysicalCopyResource> physical;
  CopyArray<CopyInterval> intervals;
  CopyArray<ResolvedCopyRange> copies;
  CopyArray<std::uint64_t> closure;
  CopyArray<CopyPhaseState *> phases;
  CopyArray<CopyPhasePlan> plans;
  CopyArray<std::uint32_t> root_order, root_leader, last_write, producer_order;
  CopyArray<unsigned char> roles;
  CopyArray<CopyRootUse> root_uses;
  CopyArray<CopyWrittenSpan> written;
  CopyArray<CopyBackendRange> native_ranges;
  CopyArray<LeaseUseBatch> tickets;
};

namespace {
void phase_terminal(void *opaque, const Status &terminal) noexcept {
  auto &phase = *static_cast<CopyPhaseState *>(opaque);
  phase.terminal = terminal;
  phase.retired.store(true, std::memory_order_release);
}
void phase_release(void *opaque) noexcept {
  auto *owner = static_cast<CopyPhaseState *>(opaque)->owner;
  owner->pending.fetch_sub(1, std::memory_order_release);
  owner->decref(); // No access to the phase/node after this drop.
}
Status valid_binding(CopyInvocationState *state, CopyResourceId id) noexcept {
  if (!state) return Status::FailedPrecondition("copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state->mutable_slot());
  if (id.value >= state->bindings.size()) return Status::InvalidArgument("copy resource ID is invalid");
  return Status::Ok();
}
Result<PhysicalCopyResource> bind_physical(CopyInvocationState &state, mem::ConstBufferView view,
                                          bool writable, mem::HostMapping &mapping) noexcept {
  if (state.program->table) {
    struct Call { mem::ConstBufferView view; PhysicalCopyResource result; } call{view, {}};
    const auto status = BackendExtensionAccess::call(state.program->extension,
        [](void *context, const void *table, void *opaque) noexcept -> Status {
      auto &call = *static_cast<Call *>(opaque);
      HYLE_ASSIGN_OR_RETURN(call.result, static_cast<const CopyBackendVTable *>(table)->physical(context, call.view));
      return Status::Ok();
    }, &call);
    if (!status.ok()) return std::unexpected(status);
    return call.result;
  }
  // CPU writable storage needs a proven persistent address. Arbitrary opaque
  // writable resource carriers are rejected, not identified by wrapper pointer.
  if (writable && view.resource() != mem::system_resource())
    return std::unexpected(Status::Unimplemented("CPU copy mutable carrier has no proven physical identity"));
  HYLE_ASSIGN_OR_RETURN(mapping, view.map_host());
  const auto address = reinterpret_cast<std::uintptr_t>(mapping.bytes().data());
  if (view.size() > SIZE_MAX - address)
    return std::unexpected(Status::OutOfRange("CPU copy physical coordinate overflow"));
  return PhysicalCopyResource{1, 0, address, view.size(), 1};
}
CopyBatchReport current_report(const CopyInvocationState *state) noexcept {
  if (!state) return {};
  auto result = state->report;
  result.metadata_bytes = state->program->metadata.live.load(std::memory_order_relaxed);
  return result;
}
Status wait_submission(CopyInvocationState *state) noexcept {
  if (!state) return Status::Ok();
  Status result;
  for (std::uint32_t index = 0; index < state->submitted; ++index) {
    auto &phase = *state->phases[index];
    if (phase.completion.valid()) {
      if (CompletionAccess::dependency(phase.completion).pending()) ++state->report.host_waits;
      result.Update(phase.completion.wait());
    }
    if (phase.retired.load(std::memory_order_acquire)) result.Update(phase.terminal);
    else if (result.ok()) result.Update(Status::Unavailable("copy phase retirement is unproven"));
  }
  result.Update(state->tail);
  if (!result.ok()) state->failed = true;
  return result;
}
} // namespace
} // namespace hyle::compute::detail

namespace hyle::compute {
using detail::CopyInvocationState;
using detail::CopyProgramState;

CopyBatchProgram::CopyBatchProgram(const CopyBatchProgram &other) noexcept : state_(other.state_) {
  if (state_) state_->incref();
}
CopyBatchProgram &CopyBatchProgram::operator=(const CopyBatchProgram &other) noexcept {
  if (this != &other) { auto *next = other.state_; if (next) next->incref(); reset(); state_ = next; }
  return *this;
}
CopyBatchProgram::CopyBatchProgram(CopyBatchProgram &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
CopyBatchProgram &CopyBatchProgram::operator=(CopyBatchProgram &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
CopyBatchProgram::~CopyBatchProgram() { reset(); }
void CopyBatchProgram::reset() noexcept { if (auto *state = std::exchange(state_, nullptr)) state->decref(); }

Result<CopyBatchProgram> CopyBatchProgram::prepare(const Context &context,
    std::span<const CopyResourceSpec> resources, CopyBatchOptions options) noexcept {
  auto *owner = detail::ContextAccess::state(context);
  if (!owner || !context.is_open()) return std::unexpected(Status::FailedPrecondition("copy context is closed or empty"));
  if (!options.allocator.valid() || options.max_resources == 0 || options.max_copies == 0 ||
      options.max_phases == 0 || options.max_phases > 64 || options.max_bytes == 0 ||
      resources.size() > options.max_resources || options.max_copies > UINT32_MAX / 2)
    return std::unexpected(Status::InvalidArgument("copy program capacities are invalid"));
  for (const auto &spec : resources) {
    if (!spec.resource.valid()) return std::unexpected(Status::InvalidArgument("copy resource is invalid"));
    HYLE_RETURN_IF_ERROR(owner->backend_vtable->validate_resource(owner->device, spec.resource));
  }
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *storage = allocator.allocate(sizeof(CopyProgramState), alignof(CopyProgramState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("copy program metadata"));
  auto *state = std::construct_at(static_cast<CopyProgramState *>(storage), allocator, std::move(options));
  CopyBatchProgram result(state);
  state->metadata.live.store(sizeof(CopyProgramState), std::memory_order_relaxed);
  state->context = owner; detail::retain(owner);
  try { state->specs.assign(resources.begin(), resources.end()); }
  catch (...) { return std::unexpected(Status::ResourceExhausted("copy program resource metadata")); }
  auto extension = detail::BackendExtensionAccess::query(context, detail::kCopyBatchExtensionId);
  if (extension) {
    if (state->options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(state->opaque_native, state->options.resource_budget.reserve_unattributed(std::nullopt));
    }
    state->extension = std::move(*extension);
    HYLE_RETURN_IF_ERROR(detail::BackendExtensionAccess::call(state->extension,
        [](void *context, const void *interface, void *opaque) noexcept -> Status {
      auto &state = *static_cast<CopyProgramState *>(opaque);
      state.table = static_cast<const detail::CopyBackendVTable *>(interface);
      std::size_t charged = 0;
      return state.table->prepare(context, state.metadata.allocator(), &state.native, &charged);
    }, state));
  } else if (!extension.error().IsUnimplemented() || owner->device.device_class() != DeviceClass::kCpu) {
    return std::unexpected(extension.error());
  }
  return result;
}

CopyBatchInvocation::CopyBatchInvocation(CopyBatchInvocation &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
CopyBatchInvocation &CopyBatchInvocation::operator=(CopyBatchInvocation &&other) noexcept {
  if (this != &other) { if (state_) state_->release_user(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
CopyBatchInvocation::~CopyBatchInvocation() { if (state_) state_->release_user(); }
Result<CopyBatchInvocation> CopyBatchInvocation::create(const CopyBatchProgram &prepared, Queue &queue) noexcept {
  if (!prepared.state_ || !queue.is_open() || detail::ContextAccess::state(queue)->context != prepared.state_->context)
    return std::unexpected(Status::InvalidArgument("copy queue and program contexts differ"));
  auto *program = prepared.state_;
  const auto allocator = program->metadata.allocator();
  auto *storage = allocator.allocate(sizeof(CopyInvocationState), alignof(CopyInvocationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("copy invocation metadata"));
  auto *state = std::construct_at(static_cast<CopyInvocationState *>(storage), program, &queue);
  CopyBatchInvocation result(state);
  const auto n = program->specs.size();
  const auto r = program->options.max_copies, g = program->options.max_phases;
  if (n > SIZE_MAX / g) return std::unexpected(Status::ResourceExhausted("copy role metadata overflows"));
  try {
    state->bindings.resize(n); state->physical.resize(n);
    state->intervals.resize(static_cast<std::size_t>(r) * 2); state->copies.resize(r);
    state->plans.resize(g); state->closure.resize(g); state->phases.reserve(g); state->tickets.resize(g);
    state->root_order.resize(n); state->root_leader.resize(n); state->last_write.resize(n);
    state->producer_order.resize(static_cast<std::size_t>(r) * 2);
    state->roles.resize(n * g); state->root_uses.resize(static_cast<std::size_t>(r) * 2);
    state->written.resize(r); state->native_ranges.resize(r);
  } catch (...) { return std::unexpected(Status::ResourceExhausted("copy fixed replay metadata")); }
  for (std::uint32_t index = 0; index < g; ++index) {
    auto *phase_storage = allocator.allocate(sizeof(detail::CopyPhaseState), alignof(detail::CopyPhaseState));
    if (!phase_storage) return std::unexpected(Status::ResourceExhausted("copy phase metadata"));
    auto *phase = std::construct_at(static_cast<detail::CopyPhaseState *>(phase_storage));
    phase->owner = state; state->phases.push_back(phase);
  }
  HYLE_ASSIGN_OR_RETURN(state->uses, LeaseUsePool::create({.max_batches = g,
      .max_resources_per_batch = static_cast<std::uint32_t>((std::max)(n, std::size_t{1})), .allocator = allocator}));
  HYLE_RETURN_IF_ERROR(queue.reserve_completions(g));
  if (program->table) {
    if (program->options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(state->opaque_native, program->options.resource_budget.reserve_unattributed(std::nullopt));
    }
    HYLE_RETURN_IF_ERROR(detail::BackendExtensionAccess::call(program->extension,
        [](void *context, const void *, void *opaque) noexcept -> Status {
      auto &state = *static_cast<CopyInvocationState *>(opaque);
      std::size_t charged = 0;
      return state.program->table->create_slot(context, detail::ContextAccess::backend_queue(*state.queue),
          state.program->native, state.program->options.max_copies, state.program->options.max_phases,
          state.allocator(), &state.native, &charged);
    }, state));
  }
  state->report.native_metadata_unknown = program->table != nullptr;
  return result;
}

Status CopyBatchInvocation::bind_source(CopyResourceId id, const ReadLease &source) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  HYLE_ASSIGN_OR_RETURN(auto view, source.view());
  const auto &spec = state_->program->specs[id.value];
  if (view.resource() != spec.resource || view.size() > spec.capacity_bytes)
    return Status::InvalidArgument("copy source resource/capacity mismatch");
  mem::HostMapping mapping;
  HYLE_ASSIGN_OR_RETURN(auto physical, detail::bind_physical(*state_, view, false, mapping));
  auto &binding = state_->bindings[id.value];
  binding = {}; binding.read = source; binding.source = view; binding.mapping = std::move(mapping);
  binding.root = detail::LeaseAccess::identity(source);
  state_->physical[id.value] = physical; state_->prepared = false;
  return Status::Ok();
}
Status CopyBatchInvocation::bind_mutable(CopyResourceId id, WriteLease &&destination) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  HYLE_ASSIGN_OR_RETURN(auto view, detail::LeaseAccess::ordered_view(destination));
  const auto &spec = state_->program->specs[id.value];
  if (view.resource() != spec.resource || view.size() > spec.capacity_bytes)
    return Status::InvalidArgument("copy mutable resource/capacity mismatch");
  mem::HostMapping mapping;
  HYLE_ASSIGN_OR_RETURN(auto physical, detail::bind_physical(*state_, view, true, mapping));
  auto &binding = state_->bindings[id.value];
  binding = {}; binding.root = detail::LeaseAccess::identity(destination);
  binding.write = std::move(destination); binding.source = view; binding.destination = view;
  binding.mapping = std::move(mapping); state_->physical[id.value] = physical; state_->prepared = false;
  return Status::Ok();
}
Status CopyBatchInvocation::bind_producer(CopyResourceId id, Completion &&producer) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  if (!state_->bindings[id.value].root) return Status::FailedPrecondition("copy producer has no resource binding");
  HYLE_ASSIGN_OR_RETURN(auto dependency, detail::RetainedCompletion::native_dependency(producer));
  if (!dependency.terminal_hint->ok()) return *dependency.terminal_hint;
  state_->bindings[id.value].producer = std::move(producer); state_->prepared = false;
  return Status::Ok();
}

Status CopyBatchInvocation::prepare_replay(std::span<const CopyPhase> phases, std::span<const CopyRange> ranges) noexcept {
  if (!state_) return Status::FailedPrecondition("copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state_->mutable_slot());
  state_->prepared = false;
  for (const auto &range : ranges) {
    if (range.source.value >= state_->bindings.size() || range.destination.value >= state_->bindings.size() ||
        !state_->bindings[range.source.value].root || !state_->bindings[range.destination.value].root)
      return Status::InvalidArgument("copy replay uses an unbound resource");
  }
  HYLE_ASSIGN_OR_RETURN(auto report, detail::validate_copy_ranges(state_->physical, phases, ranges,
      state_->program->options, {state_->intervals, state_->copies, state_->closure}));
  HYLE_RETURN_IF_ERROR(detail::prepare_copy_control(state_->bindings, phases,
      std::span(state_->copies).first(report.active_copies),
      {state_->root_order, state_->root_leader, state_->last_write, state_->producer_order,
       state_->roles, state_->root_uses, state_->written, state_->native_ranges, state_->plans}));
  for (std::size_t index = 0; index < report.active_copies; ++index) {
    const auto &range = state_->native_ranges[index];
    HYLE_ASSIGN_OR_RETURN(auto route, transfer_properties(*state_->queue, range.destination, range.source));
    (void)route;
  }
  for (std::size_t index = 0; index < phases.size(); ++index) {
    auto &slot = *state_->phases[index]; const auto &plan = state_->plans[index];
    slot.completion = {}; slot.retired.store(false, std::memory_order_relaxed); slot.terminal = Status::Ok();
    slot.first = plan.first; slot.copies = plan.copies; slot.use_first = plan.use_first; slot.use_count = plan.use_count;
    slot.producer_first = plan.producer_first; slot.producer_count = plan.producer_count;
    slot.dependency_mask = plan.dependency_mask;
  }
  if (state_->program->table) {
    struct Call { CopyInvocationState *state; std::size_t count; } call{state_, report.active_copies};
    HYLE_RETURN_IF_ERROR(detail::BackendExtensionAccess::call(state_->program->extension,
        [](void *context, const void *, void *opaque) noexcept -> Status {
      auto &call = *static_cast<Call *>(opaque);
      return call.state->program->table->bind_slot(context, call.state->program->native, call.state->native,
          std::span(call.state->native_ranges).first(call.count));
    }, &call));
  }
  report.native_metadata_unknown = state_->program->table != nullptr;
  state_->report = report; state_->tail = Status::Ok(); state_->submitted = 0;
  state_->phase_count = static_cast<std::uint32_t>(phases.size()); state_->prepared = true;
  return Status::Ok();
}

Result<CopyBatchSubmission> CopyBatchInvocation::submit() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("copy invocation is empty"));
  HYLE_RETURN_IF_ERROR(state_->mutable_slot());
  if (!state_->prepared) return std::unexpected(Status::FailedPrecondition("copy replay is not prepared"));
  // Reserve every root's use before the first native acceptance. Rejection
  // rolls all tickets back; each phase consumes only its own reserved episode.
  HYLE_RETURN_IF_ERROR(detail::admit_copy_control(state_->uses, state_->bindings,
      std::span(state_->plans).first(state_->phase_count), state_->root_uses, state_->tickets));
  state_->prepared = false;
  for (std::uint32_t phase = 0; phase < state_->phase_count; ++phase) {
    auto &slot = *state_->phases[phase];
    auto sequence = detail::QueueSequence::begin(*state_->queue);
    if (!sequence) { state_->tail = sequence.error(); break; }
    auto dependencies = slot.dependency_mask;
    while (dependencies != 0 && sequence->terminal().ok()) {
      const auto preceding = static_cast<std::size_t>(std::countr_zero(dependencies));
      auto waited = sequence->enqueue_resolved_wait(detail::CompletionAccess::dependency(state_->phases[preceding]->completion));
      if (waited.accepted) ++state_->report.native_waits;
      sequence->record(std::move(waited));
      dependencies &= dependencies - 1;
    }
    for (std::size_t index = slot.producer_first; index < slot.producer_first + slot.producer_count && sequence->terminal().ok(); ++index) {
      auto &producer = state_->bindings[state_->producer_order[index]].producer;
      auto dependency = detail::RetainedCompletion::native_dependency(producer);
      auto waited = dependency ? sequence->enqueue_resolved_wait(*dependency)
                               : detail::EnqueueResult::rejected(dependency.error());
      if (waited.accepted) ++state_->report.native_waits;
      sequence->record(std::move(waited));
    }
    if (sequence->terminal().ok()) {
      if (state_->program->table) {
        struct Call { CopyInvocationState *state; detail::CopyPhaseState *phase; } call{state_, &slot};
        sequence->record(detail::BackendExtensionAccess::enqueue_admitted(state_->program->extension, *sequence,
            [](void *context, const void *, void *queue, void *opaque) noexcept {
          auto &call = *static_cast<Call *>(opaque);
          return call.state->program->table->enqueue(context, queue, call.state->program->native, call.state->native,
              call.phase->first, call.phase->copies, &call.state->report.native_calls);
        }, &call));
      } else {
        for (std::size_t index = slot.first; index < slot.first + slot.copies && sequence->terminal().ok(); ++index) {
          ++state_->report.native_calls;
          const auto &range = state_->native_ranges[index];
          sequence->record(detail::TransferAccess::enqueue_prevalidated_copy(*sequence, range.destination, range.source));
        }
      }
    }
    const auto terminal = sequence->terminal();
    auto completion = sequence->finish();
    if (!completion) { state_->tail = completion.error(); break; }
    HYLE_FATAL_ASSERT(state_->tickets[phase].attach(*completion).ok());
    state_->incref(); state_->pending.fetch_add(1, std::memory_order_relaxed);
    slot.retention.own = {&slot, &detail::phase_terminal, &detail::phase_release};
    HYLE_FATAL_ASSERT(detail::CompletionAccess::attach_retention(*completion, slot.retention).ok());
    slot.completion = std::move(*completion); ++state_->submitted; ++state_->report.phase_completions;
    if (!terminal.ok()) { state_->tail = terminal; break; }
  }
  for (auto &ticket : state_->tickets) ticket = {};
  if (!state_->tail.ok()) state_->failed = true;
  if (state_->submitted == 0 && !state_->tail.ok()) return std::unexpected(state_->tail);
  ++state_->users; state_->incref();
  return CopyBatchSubmission(state_);
}
Status CopyBatchInvocation::reset() noexcept {
  if (!state_) return Status::Ok();
  if (state_->users != 1 || state_->busy()) return Status::Unavailable("copy slot still has a submission or native work");
  for (auto *phase : state_->phases) phase->completion = {};
  for (auto &binding : state_->bindings) binding = {};
  std::fill(state_->physical.begin(), state_->physical.end(), detail::PhysicalCopyResource{});
  state_->prepared = false; state_->failed = false; state_->tail = Status::Ok(); state_->submitted = 0; state_->phase_count = 0;
  return Status::Ok();
}
Status CopyBatchInvocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  return state_->busy() ? Status::Unavailable("copy invocation has native work") : Status::Ok();
}
Status CopyBatchInvocation::drain() noexcept { close().IgnoreError(); return detail::wait_submission(state_); }
bool CopyBatchInvocation::in_flight() const noexcept { return state_ && state_->busy(); }
CopyBatchReport CopyBatchInvocation::report() const noexcept { return detail::current_report(state_); }

CopyBatchSubmission::CopyBatchSubmission(CopyBatchSubmission &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
CopyBatchSubmission &CopyBatchSubmission::operator=(CopyBatchSubmission &&other) noexcept {
  if (this != &other) { if (state_) state_->release_user(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
CopyBatchSubmission::~CopyBatchSubmission() { if (state_) state_->release_user(); }
Result<ReadLease> CopyBatchSubmission::output(CopyPhaseId phase_id, CopyResourceId resource,
                                             std::size_t offset, std::size_t bytes) noexcept {
  if (!state_ || phase_id.value >= state_->phase_count || resource.value >= state_->bindings.size())
    return std::unexpected(Status::InvalidArgument("copy output phase/resource is invalid"));
  if (phase_id.value >= state_->submitted) return std::unexpected(state_->tail.ok()
      ? Status::Unavailable("copy output phase was not submitted") : state_->tail);
  auto &phase = *state_->phases[phase_id.value];
  auto observed = phase.completion.poll();
  if (!observed) { state_->failed = true; return std::unexpected(observed.error()); }
  if (*observed == CompletionPoll::kPending || !phase.retired.load(std::memory_order_acquire))
    return std::unexpected(Status::Unavailable("copy output has not actually retired"));
  if (!phase.terminal.ok()) return std::unexpected(phase.terminal);
  auto &binding = state_->bindings[resource.value];
  if (state_->last_write[state_->root_leader[resource.value]] != phase_id.value)
    return std::unexpected(Status::Unavailable("copy block has a later write or no output"));
  if (offset > binding.source.size() || bytes > binding.source.size() - offset)
    return std::unexpected(Status::OutOfRange("copy output exceeds binding capacity"));
  HYLE_RETURN_IF_ERROR(detail::copy_output_coverage(std::span(state_->written).first(state_->report.active_copies),
      resource, phase_id, offset, bytes));
  // Every contributing write must have real successful retirement before the
  // shared geometry/permission helper can publish its last-write block.
  for (const auto &span : std::span(state_->written).first(state_->report.active_copies)) {
    if (span.binding != resource.value || span.phase > phase_id.value ||
        span.end <= offset || span.begin >= offset + bytes) continue;
    auto &prefix = *state_->phases[span.phase];
    if (!prefix.retired.load(std::memory_order_acquire)) {
      auto observed_prefix = prefix.completion.poll();
      if (!observed_prefix) return std::unexpected(observed_prefix.error());
      if (*observed_prefix == CompletionPoll::kPending || !prefix.retired.load(std::memory_order_acquire))
        return std::unexpected(Status::Unavailable("copy output prefix is pending or unproven"));
    }
    if (!prefix.terminal.ok()) return std::unexpected(prefix.terminal);
  }
  if (binding.write.valid()) {
    HYLE_ASSIGN_OR_RETURN(binding.read, detail::LeaseAccess::publish_after_writes(binding.write));
  }
  return binding.read.subview(offset, bytes);
}
Status CopyBatchSubmission::wait() noexcept { return detail::wait_submission(state_); }
Status CopyBatchSubmission::request_cancel() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  Status result;
  for (std::uint32_t phase = 0; phase < state_->submitted; ++phase)
    result.Update(state_->phases[phase]->completion.request_cancel());
  return result;
}
bool CopyBatchSubmission::in_flight() const noexcept { return state_ && state_->busy(); }
CopyBatchReport CopyBatchSubmission::report() const noexcept { return detail::current_report(state_); }
} // namespace hyle::compute
