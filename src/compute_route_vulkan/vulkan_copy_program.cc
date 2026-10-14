#include "hyle/compute/vulkan_memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <memory>
#include <vector>

#include "vulkan_copy_internal.h"
#include "vulkan_file_stream_access.h"
#include "hyle/compute/detail/copy_control.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::vulkan::detail {
template<class T> using CopyArray = std::vector<T, mem::StdAllocator<T>>;
struct CopyProgramState final : util::RefCounted<CopyProgramState> {
  CopyProgramState(ResourceBudget scope, mem::AllocatorRef selected, CopyBatchOptions limits)
      : accounting(std::move(scope)), allocator(selected), options(std::move(limits)),
        specs(mem::StdAllocator<CopyResourceSpec>(selected)) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this); selected.deallocate(this, sizeof(CopyProgramState), alignof(CopyProgramState));
  }
  ResourceBudget accounting;
  mem::AllocatorRef allocator;
  CopyBatchOptions options;
  std::shared_ptr<Runtime> runtime;
  ResourceBudget::Reservation opaque;
  CopyArray<CopyResourceSpec> specs;
};
struct CopyNativePhase final {
  CopyInvocationState *owner = nullptr;
  std::uint32_t index = 0;
  std::shared_ptr<CopyRecording> recording;
  Completion completion;
  LeaseUseBatch use;
  cd::CompletionRetention retention;
  Status terminal;
  std::atomic<bool> retired{false};
};
struct CopyInvocationState final : util::RefCounted<CopyInvocationState> {
  explicit CopyInvocationState(CopyProgramState *selected)
      : program(selected), bindings(mem::StdAllocator<cd::CopyBinding>(allocator())),
        physical(mem::StdAllocator<cd::PhysicalCopyResource>(allocator())),
        intervals(mem::StdAllocator<cd::CopyInterval>(allocator())), copies(mem::StdAllocator<cd::ResolvedCopyRange>(allocator())),
        closure(mem::StdAllocator<std::uint64_t>(allocator())), phases(mem::StdAllocator<CopyNativePhase *>(allocator())),
        plans(mem::StdAllocator<cd::CopyPhasePlan>(allocator())), order(mem::StdAllocator<std::uint32_t>(allocator())),
        leader(mem::StdAllocator<std::uint32_t>(allocator())), last_write(mem::StdAllocator<std::uint32_t>(allocator())),
        producers(mem::StdAllocator<std::uint32_t>(allocator())), roles(mem::StdAllocator<unsigned char>(allocator())),
        root_uses(mem::StdAllocator<cd::CopyRootUse>(allocator())), written(mem::StdAllocator<cd::CopyWrittenSpan>(allocator())),
        ranges(mem::StdAllocator<cd::CopyBackendRange>(allocator())), tickets(mem::StdAllocator<LeaseUseBatch>(allocator())) { program->incref(); }
  mem::AllocatorRef allocator() const noexcept { return program->allocator; }
  bool busy() const noexcept { return pending.load(std::memory_order_acquire) != 0; }
  Status mutable_slot() const noexcept {
    if (closed) return Status::FailedPrecondition("Vulkan copy invocation is closed");
    if (users != 1 || busy()) return Status::Unavailable("Vulkan copy slot is retained");
    if (failed) return Status::FailedPrecondition("Vulkan failed copy slot needs reset");
    return Status::Ok();
  }
  void release_user() noexcept {
    HYLE_FATAL_ASSERT(users != 0);
    if (--users == 0) for (auto *phase : phases) phase->completion = {};
    decref();
  }
  void destroy() noexcept {
    auto *retained = program; const auto selected = allocator();
    for (auto *phase : phases) {
      std::destroy_at(phase); selected.deallocate(phase, sizeof(CopyNativePhase), alignof(CopyNativePhase));
    }
    std::destroy_at(this); selected.deallocate(this, sizeof(CopyInvocationState), alignof(CopyInvocationState));
    retained->decref();
  }
  CopyProgramState *program;
  LeaseUsePool uses;
  ResourceBudget::Reservation opaque;
  std::atomic<std::uint32_t> pending{0};
  std::uint32_t users = 1, phase_count = 0, submitted = 0;
  bool prepared = false, closed = false, failed = false;
  Status tail;
  CopyBatchReport report;
  CopyArray<cd::CopyBinding> bindings;
  CopyArray<cd::PhysicalCopyResource> physical;
  CopyArray<cd::CopyInterval> intervals;
  CopyArray<cd::ResolvedCopyRange> copies;
  CopyArray<std::uint64_t> closure;
  CopyArray<CopyNativePhase *> phases;
  CopyArray<cd::CopyPhasePlan> plans;
  CopyArray<std::uint32_t> order, leader, last_write, producers;
  CopyArray<unsigned char> roles;
  CopyArray<cd::CopyRootUse> root_uses;
  CopyArray<cd::CopyWrittenSpan> written;
  CopyArray<cd::CopyBackendRange> ranges;
  CopyArray<LeaseUseBatch> tickets;
};
namespace {
void phase_terminal(void *opaque, const Status &native) noexcept {
  auto &phase = *static_cast<CopyNativePhase *>(opaque); auto &state = *phase.owner;
  Status terminal = native;
  const auto &plan = state.plans[phase.index];
  // Child native completion proves this original queue's earlier fixed
  // prefixes have executed. Poll preserves checked errors without host waits.
  auto dependencies = plan.dependency_mask;
  while (dependencies != 0) {
    const auto prior = static_cast<std::size_t>(std::countr_zero(dependencies));
    auto &source = *state.phases[prior];
    auto observed = source.completion.valid() ? source.completion.poll() : poll_copy_recording(*source.recording);
    if (!observed) terminal.Update(observed.error());
    else if (*observed == CompletionPoll::kPending || !source.retired.load(std::memory_order_acquire))
      terminal.Update(Status::Unavailable("Vulkan copy predecessor retirement is unproven"));
    else terminal.Update(source.terminal);
    dependencies &= dependencies - 1;
  }
  for (std::size_t index = plan.producer_first; index < plan.producer_first + plan.producer_count; ++index) {
    auto observed = state.bindings[state.producers[index]].producer.poll();
    if (!observed) terminal.Update(observed.error());
    else if (*observed == CompletionPoll::kPending)
      terminal.Update(Status::Unavailable("Vulkan copy producer retirement is unproven"));
  }
  cd::LeaseAccess::retire_native(phase.use, terminal);
  phase.terminal = terminal; phase.retired.store(true, std::memory_order_release);
}
void phase_release(void *opaque) noexcept {
  auto *owner = static_cast<CopyNativePhase *>(opaque)->owner;
  owner->pending.fetch_sub(1, std::memory_order_release); owner->decref();
}
Status valid_binding(CopyInvocationState *state, CopyResourceId id) noexcept {
  if (!state) return Status::FailedPrecondition("Vulkan copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state->mutable_slot());
  return id.value >= state->bindings.size() ? Status::InvalidArgument("Vulkan copy resource ID is invalid") : Status::Ok();
}
CopyBatchReport report(const CopyInvocationState *state) noexcept {
  if (!state) return {};
  auto result = state->report;
  auto usage = state->program->accounting.usage(mem::system_resource());
  HYLE_FATAL_ASSERT(usage.has_value()); result.metadata_bytes = usage->live_bytes;
  return result;
}
Status wait(CopyInvocationState *state) noexcept {
  if (!state) return Status::Ok();
  Status result;
  for (std::uint32_t id = 0; id < state->submitted; ++id) {
    auto &phase = *state->phases[id];
    if (cd::CompletionAccess::dependency(phase.completion).pending()) ++state->report.host_waits;
    result.Update(phase.completion.wait());
    if (phase.retired.load(std::memory_order_acquire)) result.Update(phase.terminal);
    else result.Update(Status::Unavailable("Vulkan copy actual retirement is unproven"));
  }
  result.Update(state->tail); if (!result.ok()) state->failed = true;
  return result;
}
} // namespace
} // namespace hyle::compute::vulkan::detail

namespace hyle::compute::vulkan {
CopyBatchProgram::CopyBatchProgram(const CopyBatchProgram &other) noexcept : state_(other.state_) { if (state_) state_->incref(); }
CopyBatchProgram &CopyBatchProgram::operator=(const CopyBatchProgram &other) noexcept {
  if (this != &other) { auto *next = other.state_; if (next) next->incref(); reset(); state_ = next; } return *this;
}
CopyBatchProgram::CopyBatchProgram(CopyBatchProgram &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
CopyBatchProgram &CopyBatchProgram::operator=(CopyBatchProgram &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
CopyBatchProgram::~CopyBatchProgram() { reset(); }
void CopyBatchProgram::reset() noexcept { if (auto *state = std::exchange(state_, nullptr)) state->decref(); }
Result<CopyBatchProgram> CopyBatchProgram::prepare(const Context &context,
    std::span<const CopyResourceSpec> specs, CopyBatchOptions options) noexcept {
  if (!context.valid() || !options.allocator.valid() || options.max_resources == 0 ||
      options.max_copies == 0 || options.max_copies > UINT32_MAX / 2 || options.max_phases == 0 ||
      options.max_phases > 64 || options.max_bytes == 0 || specs.size() > options.max_resources)
    return std::unexpected(Status::InvalidArgument("Vulkan copy program limits are invalid"));
  const auto &runtime = detail::Access::runtime(context);
  { std::lock_guard lock(runtime->mutex);
    if (runtime->closed) return std::unexpected(Status::FailedPrecondition("Vulkan copy Context is closed"));
    if (!runtime->failure.ok()) return std::unexpected(runtime->failure); }
  for (const auto &spec : specs) if (!spec.resource.valid() || spec.resource.domain().authority != runtime->device.get())
    return std::unexpected(Status::InvalidArgument("Vulkan copy program resource Device differs"));
  auto base = options.allocator;
  if (options.resource_budget.valid()) { HYLE_ASSIGN_OR_RETURN(base, options.resource_budget.metadata_allocator()); }
  const std::array<ResourceBudgetLimit, 1> limits{{{mem::system_resource(), SIZE_MAX}}};
  // Original ResourceBudget retains its accounting owner through every actual
  // allocation, including the Run control block freed after its last callback.
  HYLE_ASSIGN_OR_RETURN(auto accounting, ResourceBudget::create({.limits = limits, .allocator = base}));
  HYLE_ASSIGN_OR_RETURN(auto allocator, accounting.metadata_allocator());
  auto *storage = allocator.allocate(sizeof(detail::CopyProgramState), alignof(detail::CopyProgramState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Vulkan copy program metadata"));
  auto *state = std::construct_at(static_cast<detail::CopyProgramState *>(storage), std::move(accounting), allocator, std::move(options));
  CopyBatchProgram result(state); state->runtime = runtime;
  try { state->specs.assign(specs.begin(), specs.end()); }
  catch (...) { return std::unexpected(Status::ResourceExhausted("Vulkan copy endpoint metadata")); }
  if (state->options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(state->opaque, state->options.resource_budget.reserve_unattributed(std::nullopt));
  }
  return result;
}
CopyBatchInvocation::CopyBatchInvocation(CopyBatchInvocation &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
CopyBatchInvocation &CopyBatchInvocation::operator=(CopyBatchInvocation &&other) noexcept {
  if (this != &other) { if (state_) state_->release_user(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
CopyBatchInvocation::~CopyBatchInvocation() { if (state_) state_->release_user(); }
Result<CopyBatchInvocation> CopyBatchInvocation::create(const CopyBatchProgram &prepared) noexcept {
  if (!prepared.state_) return std::unexpected(Status::InvalidArgument("Vulkan copy program is empty"));
  auto *program = prepared.state_; const auto allocator = program->allocator;
  auto *storage = allocator.allocate(sizeof(detail::CopyInvocationState), alignof(detail::CopyInvocationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Vulkan copy invocation metadata"));
  auto *state = std::construct_at(static_cast<detail::CopyInvocationState *>(storage), program);
  CopyBatchInvocation result(state);
  const auto n = program->specs.size(); const auto r = program->options.max_copies, g = program->options.max_phases;
  if (n > SIZE_MAX / g) return std::unexpected(Status::ResourceExhausted("Vulkan copy role capacity overflows"));
  try {
    state->bindings.resize(n); state->physical.resize(n); state->intervals.resize(static_cast<std::size_t>(r) * 2);
    state->copies.resize(r); state->closure.resize(g); state->phases.reserve(g); state->plans.resize(g);
    state->order.resize(n); state->leader.resize(n); state->last_write.resize(n); state->producers.resize(static_cast<std::size_t>(r) * 2);
    state->roles.resize(n * g); state->root_uses.resize(static_cast<std::size_t>(r) * 2);
    state->written.resize(r); state->ranges.resize(r); state->tickets.resize(g);
  } catch (...) { return std::unexpected(Status::ResourceExhausted("Vulkan copy fixed metadata")); }
  for (std::uint32_t id = 0; id < g; ++id) {
    auto *memory = allocator.allocate(sizeof(detail::CopyNativePhase), alignof(detail::CopyNativePhase));
    if (!memory) return std::unexpected(Status::ResourceExhausted("Vulkan copy phase metadata"));
    auto *phase = std::construct_at(static_cast<detail::CopyNativePhase *>(memory));
    phase->owner = state; phase->index = id; state->phases.push_back(phase);
    HYLE_ASSIGN_OR_RETURN(phase->recording, detail::prepare_copy_recording_runtime(program->runtime, r, allocator, program->options.max_bytes));
  }
  HYLE_ASSIGN_OR_RETURN(state->uses, LeaseUsePool::create({.max_batches = g,
      .max_resources_per_batch = static_cast<std::uint32_t>((std::max)(n, std::size_t{1})), .allocator = allocator}));
  if (program->options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(state->opaque, program->options.resource_budget.reserve_unattributed(std::nullopt));
  }
  state->report.native_metadata_unknown = true; return result;
}
Result<WriteLease> CopyBatchInvocation::unbind_mutable(CopyResourceId id) noexcept {
  HYLE_RETURN_IF_ERROR(detail::CopyStreamAccess::observe(*this));
  return detail::CopyStreamAccess::take_mutable(*this,id);
}
Result<WriteLease> detail::CopyStreamAccess::take_mutable(CopyBatchInvocation &invocation, CopyResourceId id) noexcept {
  auto *state = invocation.state_; HYLE_RETURN_IF_ERROR(detail::valid_binding(state, id));
  for (std::uint32_t index = 0; index < state->submitted; ++index) {
    const auto &phase = *state->phases[index];
    if (!phase.retired.load(std::memory_order_acquire))
      return std::unexpected(Status::Unavailable("Vulkan copy writer retirement is unproven"));
    if (!phase.terminal.ok()) return std::unexpected(phase.terminal);
  }
  auto &binding = state->bindings[id.value];
  if (!binding.write.valid()) return std::unexpected(Status::FailedPrecondition("Vulkan copy binding has no writer"));
  HYLE_ASSIGN_OR_RETURN(auto unique, binding.write.view());(void)unique;
  auto writer = std::move(binding.write); binding = {}; state->physical[id.value] = {}; state->prepared = false;
  return writer;
}
Result<compute::detail::PhysicalCopyResource> detail::CopyStreamAccess::physical(
    CopyBatchInvocation &invocation, mem::ConstBufferView view, bool) noexcept {
  if (!invocation.state_) return std::unexpected(Status::FailedPrecondition("Vulkan copy invocation is empty"));
  HYLE_ASSIGN_OR_RETURN(auto span, detail::MemoryAccess::copy_span(invocation.state_->program->runtime, view));
  return span.physical;
}
Status detail::CopyStreamAccess::observe(CopyBatchInvocation &invocation) noexcept {
  auto *state = invocation.state_;
  if (!state) return Status::FailedPrecondition("Vulkan copy invocation is empty");
  Status result = state->tail;
  for (std::uint32_t index = 0; index < state->submitted; ++index) {
    auto &phase = *state->phases[index]; auto observed = phase.completion.poll();
    if (!observed) result.Update(observed.error());
    else if (*observed == CompletionPoll::kPending || !phase.retired.load(std::memory_order_acquire))
      result.Update(Status::Unavailable("Vulkan copy retirement is pending"));
    else result.Update(phase.terminal);
  }
  return result;
}
Status CopyBatchInvocation::bind_source(CopyResourceId id, const ReadLease &source) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id)); HYLE_ASSIGN_OR_RETURN(auto view, source.view());
  const auto &spec = state_->program->specs[id.value];
  if (view.resource() != spec.resource || view.size() > spec.capacity_bytes)
    return Status::InvalidArgument("Vulkan copy source endpoint/capacity differs");
  HYLE_ASSIGN_OR_RETURN(auto span, detail::MemoryAccess::copy_span(state_->program->runtime, view));
  auto &binding = state_->bindings[id.value]; binding = {}; binding.read = source; binding.source = view;
  binding.root = compute::detail::LeaseAccess::identity(source); state_->physical[id.value] = span.physical;
  state_->prepared = false; return Status::Ok();
}
Status CopyBatchInvocation::bind_mutable(CopyResourceId id, WriteLease &&destination) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  HYLE_ASSIGN_OR_RETURN(auto view, compute::detail::LeaseAccess::ordered_view(destination));
  const auto &spec = state_->program->specs[id.value];
  if (view.resource() != spec.resource || view.size() > spec.capacity_bytes)
    return Status::InvalidArgument("Vulkan copy mutable endpoint/capacity differs");
  HYLE_ASSIGN_OR_RETURN(auto span, detail::MemoryAccess::copy_span(state_->program->runtime, view));
  auto &binding = state_->bindings[id.value]; binding = {}; binding.root = compute::detail::LeaseAccess::identity(destination);
  binding.write = std::move(destination); binding.source = view; binding.destination = view;
  state_->physical[id.value] = span.physical; state_->prepared = false; return Status::Ok();
}
Status CopyBatchInvocation::bind_producer(CopyResourceId id, Completion &&producer) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  if (!state_->bindings[id.value].root) return Status::FailedPrecondition("Vulkan copy producer resource is unbound");
  HYLE_ASSIGN_OR_RETURN(auto dependency, compute::detail::RetainedCompletion::native_dependency(producer));
  HYLE_RETURN_IF_ERROR(detail::qualify_copy_dependency(state_->program->runtime, dependency));
  state_->bindings[id.value].producer = std::move(producer); state_->prepared = false; return Status::Ok();
}
Status CopyBatchInvocation::prepare_replay(std::span<const CopyPhase> phases, std::span<const CopyRange> ranges) noexcept {
  if (!state_) return Status::FailedPrecondition("Vulkan copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state_->mutable_slot()); state_->prepared = false;
  for (const auto &range : ranges) if (range.source.value >= state_->bindings.size() || range.destination.value >= state_->bindings.size() ||
      !state_->bindings[range.source.value].root || !state_->bindings[range.destination.value].root)
    return Status::InvalidArgument("Vulkan copy resource is unbound");
  HYLE_ASSIGN_OR_RETURN(auto report, compute::detail::validate_copy_ranges(state_->physical, phases, ranges,
      state_->program->options, {state_->intervals, state_->copies, state_->closure}));
  HYLE_RETURN_IF_ERROR(compute::detail::prepare_copy_control(state_->bindings, phases,
      std::span(state_->copies).first(report.active_copies), {state_->order, state_->leader, state_->last_write,
      state_->producers, state_->roles, state_->root_uses, state_->written, state_->ranges, state_->plans}));
  for (std::size_t id = 0; id < phases.size(); ++id) {
    const auto &plan = state_->plans[id]; bool host_ready = plan.dependency_mask == 0 && plan.producer_count == 0;
    for (std::size_t index = plan.first; index < plan.first + plan.copies; ++index) {
      const auto source = state_->copies[index].source;
      host_ready = host_ready && state_->last_write[state_->leader[source]] == UINT32_MAX;
    }
    HYLE_RETURN_IF_ERROR(detail::bind_copy_recording(*state_->phases[id]->recording,
        std::span(state_->ranges).subspan(plan.first, plan.copies), host_ready,
        plan.dependency_mask != 0 || plan.producer_count != 0));
    state_->phases[id]->completion = {}; state_->phases[id]->terminal = Status::Ok();
    state_->phases[id]->retired.store(false, std::memory_order_relaxed);
  }
  state_->phase_count = static_cast<std::uint32_t>(phases.size()); state_->submitted = 0;
  report.native_metadata_unknown = true; state_->report = report; state_->tail = Status::Ok(); state_->prepared = true;
  return Status::Ok();
}
Result<CopyBatchSubmission> CopyBatchInvocation::submit() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("Vulkan copy invocation is empty"));
  HYLE_RETURN_IF_ERROR(state_->mutable_slot());
  if (!state_->prepared) return std::unexpected(Status::FailedPrecondition("Vulkan copy replay is not prepared"));
  // Revalidate every borrowed producer before accepting any phase, without a
  // host wait. Same original compute queue ordering is the qualified route.
  for (std::uint32_t id = 0; id < state_->phase_count; ++id) {
    const auto &plan = state_->plans[id];
    for (std::size_t index = plan.producer_first; index < plan.producer_first + plan.producer_count; ++index) {
      HYLE_ASSIGN_OR_RETURN(auto dependency, compute::detail::RetainedCompletion::native_dependency(state_->bindings[state_->producers[index]].producer));
      HYLE_RETURN_IF_ERROR(detail::qualify_copy_dependency(state_->program->runtime, dependency));
    }
  }
  HYLE_RETURN_IF_ERROR(compute::detail::admit_copy_control(state_->uses, state_->bindings,
      std::span(state_->plans).first(state_->phase_count), state_->root_uses, state_->tickets));
  state_->prepared = false;
  for (std::uint32_t id = 0; id < state_->phase_count; ++id) {
    auto &phase = *state_->phases[id];
    auto completion = detail::submit_copy_recording(*phase.recording);
    if (!completion) { state_->tail = completion.error(); break; }
    phase.use = std::move(state_->tickets[id]);
    state_->incref(); state_->pending.fetch_add(1, std::memory_order_relaxed);
    phase.retention.own = {&phase, &detail::phase_terminal, &detail::phase_release};
    HYLE_FATAL_ASSERT(compute::detail::CompletionAccess::attach_retention(*completion, phase.retention).ok());
    phase.completion = std::move(*completion); ++state_->submitted;
    const auto native = detail::copy_recording_report(*phase.recording);
    state_->report.native_calls += native.native_calls; state_->report.phase_completions += native.phase_completions;
    const auto hint = detail::copy_recording_terminal_hint(*phase.recording);
    if (!hint.ok()) { state_->tail = hint; break; }
  }
  for (auto &ticket : state_->tickets) ticket = {};
  if (!state_->tail.ok()) state_->failed = true;
  if (state_->submitted == 0 && !state_->tail.ok()) return std::unexpected(state_->tail);
  ++state_->users; state_->incref(); return CopyBatchSubmission(state_);
}
Status CopyBatchInvocation::reset() noexcept {
  if (!state_) return Status::Ok();
  if (state_->users != 1 || state_->busy()) return Status::Unavailable("Vulkan copy slot is retained");
  for (auto *phase : state_->phases) phase->completion = {};
  for (auto *phase : state_->phases) HYLE_RETURN_IF_ERROR(detail::clear_copy_recording(*phase->recording));
  for (auto &binding : state_->bindings) binding = {};
  std::fill(state_->physical.begin(), state_->physical.end(), compute::detail::PhysicalCopyResource{});
  state_->prepared = false; state_->failed = false; state_->submitted = 0; state_->phase_count = 0; state_->tail = Status::Ok();
  return Status::Ok();
}
Status CopyBatchInvocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  return state_->busy() ? Status::Unavailable("Vulkan copy has native work") : Status::Ok();
}
Status CopyBatchInvocation::drain() noexcept { close().IgnoreError(); return detail::wait(state_); }
bool CopyBatchInvocation::in_flight() const noexcept { return state_ && state_->busy(); }
CopyBatchReport CopyBatchInvocation::report() const noexcept { return detail::report(state_); }
CopyBatchSubmission::CopyBatchSubmission(CopyBatchSubmission &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
CopyBatchSubmission &CopyBatchSubmission::operator=(CopyBatchSubmission &&other) noexcept {
  if (this != &other) { if (state_) state_->release_user(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
CopyBatchSubmission::~CopyBatchSubmission() { if (state_) state_->release_user(); }
Result<ReadLease> CopyBatchSubmission::output(CopyPhaseId phase, CopyResourceId id, std::size_t offset, std::size_t bytes) noexcept {
  if (!state_ || phase.value >= state_->phase_count || id.value >= state_->bindings.size())
    return std::unexpected(Status::InvalidArgument("Vulkan copy output IDs are invalid"));
  if (phase.value >= state_->submitted) return std::unexpected(state_->tail.ok()
      ? Status::Unavailable("Vulkan copy phase was not submitted") : state_->tail);
  auto &observed = *state_->phases[phase.value]; auto poll = observed.completion.poll();
  if (!poll) { state_->failed = true; return std::unexpected(poll.error()); }
  if (*poll == CompletionPoll::kPending || !observed.retired.load(std::memory_order_acquire))
    return std::unexpected(Status::Unavailable("Vulkan copy output is pending"));
  if (!observed.terminal.ok()) return std::unexpected(observed.terminal);
  if (state_->last_write[state_->leader[id.value]] != phase.value)
    return std::unexpected(Status::Unavailable("Vulkan copy block has a later write or no output"));
  auto &binding = state_->bindings[id.value];
  if (offset > binding.source.size() || bytes > binding.source.size() - offset)
    return std::unexpected(Status::OutOfRange("Vulkan copy output exceeds binding"));
  HYLE_RETURN_IF_ERROR(compute::detail::copy_output_coverage(std::span(state_->written).first(state_->report.active_copies), id, phase, offset, bytes));
  for (const auto &span : std::span(state_->written).first(state_->report.active_copies)) {
    if (span.binding != id.value || span.phase > phase.value || span.end <= offset || span.begin >= offset + bytes) continue;
    auto &prior = *state_->phases[span.phase]; auto result = prior.completion.poll();
    if (!result) return std::unexpected(result.error());
    if (*result == CompletionPoll::kPending || !prior.retired.load(std::memory_order_acquire))
      return std::unexpected(Status::Unavailable("Vulkan copy output write is pending"));
    if (!prior.terminal.ok()) return std::unexpected(prior.terminal);
  }
  if (binding.write.valid()) { HYLE_ASSIGN_OR_RETURN(binding.read, compute::detail::LeaseAccess::publish_after_writes(binding.write)); }
  return binding.read.subview(offset, bytes);
}
Status CopyBatchSubmission::wait() noexcept { return detail::wait(state_); }
Status CopyBatchSubmission::request_cancel() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  Status result; for (std::uint32_t id = 0; id < state_->submitted; ++id) result.Update(state_->phases[id]->completion.request_cancel());
  return result;
}
bool CopyBatchSubmission::in_flight() const noexcept { return state_ && state_->busy(); }
CopyBatchReport CopyBatchSubmission::report() const noexcept { return detail::report(state_); }
Status detail::CopyUpdateAccess::qualify(const CopyBatchInvocation &invocation,
                                         const CopyBatchSubmission &submission) noexcept {
  auto *state = invocation.state_;
  if (!state || state != submission.state_)
    return Status::InvalidArgument("Vulkan copy update controls do not own the same accepted slot");
  if (state->users != 2 || state->submitted == 0)
    return Status::FailedPrecondition("Vulkan copy update needs its sole original accepted submission");
  return Status::Ok();
}
Status detail::CopyUpdateAccess::observe(CopyBatchInvocation &invocation) noexcept {
  return detail::CopyStreamAccess::observe(invocation);
}

Result<compute::detail::CopyUpdateInput> detail::CopyStreamAccess::input(
    const CopyBatchUpdate &owner, CopyResourceId id) noexcept {
  const auto &invocation = detail::CopyUpdateAccess::invocation(owner);
  const auto &submission = detail::CopyUpdateAccess::submission(owner);
  HYLE_RETURN_IF_ERROR(detail::CopyUpdateAccess::qualify(invocation, submission));
  auto *state = invocation.state_;
  if (!state->tail.ok()) return std::unexpected(state->tail);
  if (state->failed) return std::unexpected(Status::FailedPrecondition("copy update source failed"));
  if (id.value >= state->bindings.size())
    return std::unexpected(Status::InvalidArgument("copy update resource ID is invalid"));
  const auto &binding = state->bindings[id.value];
  if (!binding.write.valid())
    return std::unexpected(Status::FailedPrecondition("copy update has no original mutable destination"));
  bool written = false;
  for (const auto &span : std::span(state->written).first(state->report.active_copies))
    written |= span.binding == id.value;
  if (!written || state->leader[id.value] >= state->last_write.size())
    return std::unexpected(Status::FailedPrecondition("copy update destination has no recorded write"));
  const auto last = state->last_write[state->leader[id.value]];
  if (last >= state->submitted)
    return std::unexpected(Status::FailedPrecondition("copy update destination has no accepted write"));
  for (std::uint32_t phase = 0; phase < state->submitted; ++phase)
    if (state->phases[phase]->retired.load(std::memory_order_acquire) &&
        !state->phases[phase]->terminal.ok())
      return std::unexpected(state->phases[phase]->terminal);
  HYLE_ASSIGN_OR_RETURN(auto view, compute::detail::LeaseAccess::ordered_view(binding.write));
  return compute::detail::CopyUpdateInput{&binding.write, view, state->physical[id.value],
      binding.root, {last}};
}
Status detail::CopyStreamAccess::covers(const CopyBatchUpdate &owner, CopyResourceId id,
                                      std::size_t offset, std::size_t bytes) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto original, input(owner, id));
  if (offset > original.view.size() || bytes > original.view.size() - offset)
    return Status::OutOfRange("copy update coverage exceeds original destination");
  const auto *state = detail::CopyUpdateAccess::invocation(owner).state_;
  return compute::detail::copy_output_coverage(
      std::span(state->written).first(state->report.active_copies), id,
      original.last_write, offset, bytes);
}
Result<compute::detail::CompletionDependency> detail::CopyStreamAccess::dependency(
    const CopyBatchUpdate &owner) noexcept {
  const auto &invocation = detail::CopyUpdateAccess::invocation(owner);
  const auto &submission = detail::CopyUpdateAccess::submission(owner);
  HYLE_RETURN_IF_ERROR(detail::CopyUpdateAccess::qualify(invocation, submission));
  auto *state = invocation.state_;
  if (!state->tail.ok()) return std::unexpected(state->tail);
  if (state->failed) return std::unexpected(Status::FailedPrecondition("copy update source failed"));
  compute::detail::CompletionDependency latest{};
  for (std::uint32_t phase = 0; phase < state->submitted; ++phase) {
    HYLE_ASSIGN_OR_RETURN(auto native,
        compute::detail::RetainedCompletion::native_dependency(state->phases[phase]->completion));
    if (native.terminal_hint && !native.terminal_hint->ok())
      return std::unexpected(*native.terminal_hint);
    if (state->phases[phase]->retired.load(std::memory_order_acquire) &&
        !state->phases[phase]->terminal.ok())
      return std::unexpected(state->phases[phase]->terminal);
    if (native.pending()) latest = native;
  }
  return latest;
}
Result<std::size_t> detail::CopyStreamAccess::retirement_sources(
    CopyBatchUpdate &owner, std::span<Completion *> output) noexcept {
  auto &invocation=detail::CopyUpdateAccess::invocation(owner);
  auto &submission=detail::CopyUpdateAccess::submission(owner);
  HYLE_RETURN_IF_ERROR(detail::CopyUpdateAccess::qualify(invocation,submission));
  auto *state=invocation.state_;
  const auto used=[&](std::size_t binding) noexcept {
    for(std::uint32_t phase=0;phase<state->submitted;++phase) {
      const auto &plan=state->plans[phase];
      for(std::size_t i=plan.producer_first;i<plan.producer_first+plan.producer_count;++i)
        if(state->producers[i]==binding)return true;
    }
    return false;
  };
  std::size_t count=state->submitted;
  for(std::size_t i=0;i<state->bindings.size();++i)if(used(i))++count;
  if(output.empty())return count;
  if(output.size()<count)return std::unexpected(Status::ResourceExhausted("Vulkan retirement source capacity"));
  std::size_t at=0;
  for(std::size_t i=0;i<state->bindings.size();++i)
    if(used(i))output[at++]=&state->bindings[i].producer;
  for(std::uint32_t i=0;i<state->submitted;++i)output[at++]=&state->phases[i]->completion;
  return count;
}
Status detail::CopyStreamAccess::admit_read(LeaseUseBatch &uses,
    const CopyBatchUpdate &owner, CopyResourceId id) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto original, input(owner, id));
  return compute::detail::LeaseAccess::ordered_read_unique(uses, *original.writer);
}

} // namespace hyle::compute::vulkan
