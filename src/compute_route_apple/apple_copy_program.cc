#include "hyle/compute/apple_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "apple_copy_internal.h"
#include "hyle/compute/detail/apple_copy_stream_access.h"
#include "hyle/compute/detail/copy_control.h"
#include "hyle/compute/detail/native_lease_inputs.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::apple::detail {
namespace cd = compute::detail;
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
struct ResultSources final {
  explicit ResultSources(mem::AllocatorRef a):results(a),spans(mem::StdAllocator<CopySpan>(a)),phase_inputs(mem::StdAllocator<CopyPhase>(a)),range_inputs(mem::StdAllocator<CopyRange>(a)),real_copies(mem::StdAllocator<cd::ResolvedCopyRange>(a)),real_intervals(mem::StdAllocator<cd::CopyInterval>(a)),real_closure(mem::StdAllocator<std::uint64_t>(a)){}
  cd::NativeCopyResults<Results> results;
  CopyArray<CopySpan> spans;
  CopyArray<CopyPhase> phase_inputs;
  CopyArray<CopyRange> range_inputs;
  CopyArray<cd::ResolvedCopyRange> real_copies;
  CopyArray<cd::CopyInterval> real_intervals;
  CopyArray<std::uint64_t> real_closure;
  bool resolved=false;
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
  Status mutable_slot(bool recover_writer = false) const noexcept {
    if (closed && !recover_writer) return Status::FailedPrecondition("Apple copy invocation is closed");
    if (users != 1 || busy()) return Status::Unavailable("Apple copy slot is retained");
    if (failed) return Status::FailedPrecondition("Apple failed copy slot needs reset");
    return Status::Ok();
  }
  void release_user() noexcept {
    {
      std::lock_guard lock(observations);
      HYLE_FATAL_ASSERT(users != 0);
      if (--users == 0) for (auto *phase : phases) phase->completion = {};
    }
    decref();
  }
  void destroy() noexcept {
    auto *retained = program; const auto selected = allocator();
    for (auto *phase : phases) {
      std::destroy_at(phase); selected.deallocate(phase, sizeof(CopyNativePhase), alignof(CopyNativePhase));
    }
    auto*extra=result_sources;
    std::destroy_at(this);if(extra)selected.destroy(extra);
    selected.deallocate(this, sizeof(CopyInvocationState), alignof(CopyInvocationState));
    retained->decref();
  }
  mutable std::recursive_mutex observations;
  CopyProgramState *program;
  LeaseUsePool uses;
  ResourceBudget::Reservation opaque;
  std::atomic<std::uint32_t> pending{0};
  std::uint32_t users = 1, phase_count = 0, submitted = 0;
  bool prepared = false, closed = false, failed = false;
  Status tail;
  CopyBatchReport report;
  ResultSources*result_sources=nullptr; // Explicit opt-in cold ownership only.
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
Result<CopySpan> result_source_range(void*opaque,std::size_t index)noexcept {
  auto&phase=*static_cast<CopyNativePhase*>(opaque);auto&state=*phase.owner;
  const auto&plan=state.plans[phase.index];
  if(index>=plan.copies)return std::unexpected(Status::OutOfRange("native result source range index"));
  const auto&copy=state.copies[plan.first+index];
  if(!state.result_sources || !state.result_sources->results.owns(copy.source))
    return copy_span(state.ranges[plan.first+index].source, mem::HostAccess::kRead);
  auto source=state.result_sources->spans[copy.source];
  if(copy.source_offset>source.physical.capacity || copy.physical_bytes>source.physical.capacity-copy.source_offset ||
      copy.source_offset>SIZE_MAX-source.physical.origin)
    return std::unexpected(Status::OutOfRange("native result source subrange exceeds output"));
  source.physical.origin+=copy.source_offset;source.physical.capacity=copy.physical_bytes;
  return source;
}
void phase_terminal(void *opaque, const Status &native) noexcept {
  auto &phase = *static_cast<CopyNativePhase *>(opaque); auto &state = *phase.owner;
  std::lock_guard lock(state.observations);
  Status terminal = native;
  const auto &plan = state.plans[phase.index];
  // Original FIFO Jobs have retired their native work before this predicate
  // or retirement callback. Serialize internal and public observations of
  // the same Completion, never race worker polling with output()/wait().
  auto dependencies = plan.dependency_mask;
  while (dependencies != 0) {
    const auto prior = static_cast<std::size_t>(std::countr_zero(dependencies));
    auto &source = *state.phases[prior];
    auto observed = source.completion.valid() ? source.completion.poll() : poll_copy_recording(*source.recording);
    if (!observed) terminal.Update(observed.error());
    else if (*observed == CompletionPoll::kPending || !source.retired.load(std::memory_order_acquire))
      terminal.Update(Status::Unavailable("Apple copy predecessor retirement is unproven"));
    else terminal.Update(source.terminal);
    dependencies &= dependencies - 1;
  }
  for (std::size_t index = plan.producer_first; index < plan.producer_first + plan.producer_count; ++index) {
    auto observed = state.bindings[state.producers[index]].producer.poll();
    if (!observed) terminal.Update(observed.error());
    else if (*observed == CompletionPoll::kPending)
      terminal.Update(Status::Unavailable("Apple copy producer retirement is unproven"));
  }
  cd::LeaseAccess::retire_native(phase.use, terminal);
  phase.terminal = terminal; phase.retired.store(true, std::memory_order_release);
}
Status phase_preflight(void *opaque) noexcept {
  auto &phase = *static_cast<CopyNativePhase *>(opaque); auto &state = *phase.owner;
  std::lock_guard lock(state.observations);
  Status terminal;
  const auto &plan = state.plans[phase.index];
  auto dependencies = plan.dependency_mask;
  while (dependencies != 0) {
    auto &prior = *state.phases[static_cast<std::size_t>(std::countr_zero(dependencies))];
    auto observed = prior.completion.valid() ? prior.completion.poll() : poll_copy_recording(*prior.recording);
    if (!observed) terminal.Update(observed.error());
    else if (*observed == CompletionPoll::kPending || !prior.retired.load(std::memory_order_acquire))
      terminal.Update(Status::Unavailable("Apple copy preceding FIFO Job is not retired"));
    else terminal.Update(prior.terminal);
    dependencies &= dependencies - 1;
  }
  for (std::size_t index = plan.producer_first; index < plan.producer_first + plan.producer_count; ++index) {
    auto observed = state.bindings[state.producers[index]].producer.poll();
    if (!observed) terminal.Update(observed.error());
    else if (*observed == CompletionPoll::kPending)
      terminal.Update(Status::Unavailable("Apple copy producer FIFO Job is not retired"));
  }
  if(terminal.ok() && state.result_sources && !state.result_sources->results.ids().empty()){
    auto&extra=*state.result_sources;
    if(!extra.resolved){
      for(const auto id:extra.results.ids()){
        const bool used=std::any_of(state.copies.begin(),state.copies.begin()+state.report.active_copies,
            [&](const auto&copy){return copy.source==id.value;});
        if(!used)continue;
        const auto producer=std::find_if(state.bindings.begin(),state.bindings.end(),
            [&](const auto&binding){return binding.root==extra.results.root(id.value) && binding.producer.valid();});
        if(producer==state.bindings.end())return Status::Internal("Apple result source lost its original producer");
        auto ready=producer->producer.poll();if(!ready)return ready.error();
        if(*ready!=CompletionPoll::kComplete)return Status::Unavailable("Apple result source FIFO producer is not retired");
        HYLE_ASSIGN_OR_RETURN(extra.spans[id.value],result_copy_span(extra.results.payload(id.value),
            extra.results.output(id.value),extra.results.description(id.value)));
        state.physical[id.value]=extra.spans[id.value].physical;
      }
      HYLE_ASSIGN_OR_RETURN(auto real,cd::validate_copy_ranges(state.physical,extra.phase_inputs,
          extra.range_inputs,state.program->options,{extra.real_intervals,extra.real_copies,extra.real_closure}));
      // Keep the admitted coverage and phase/retirement descriptors immutable.
      // An unexpected physical no-op cannot become a fresh page write proof.
      if(real.active_copies!=state.report.active_copies)
        return Status::FailedPrecondition("Apple late physical mapping changes the admitted copy set");
      for(std::size_t i=0;i<real.active_copies;++i){const auto&a=extra.real_copies[i];const auto&b=state.copies[i];
        if(a.source!=b.source || a.destination!=b.destination || a.phase!=b.phase || a.input_index!=b.input_index ||
            a.source_offset!=b.source_offset || a.destination_offset!=b.destination_offset ||
            a.logical_bytes!=b.logical_bytes || a.physical_bytes!=b.physical_bytes)
          return Status::FailedPrecondition("Apple late physical copy footprint changed");
      }
      extra.resolved=true;
    }
    HYLE_RETURN_IF_ERROR(resolve_result_copy_recording(*phase.recording,
        std::span(state.ranges).subspan(plan.first,plan.copies),&result_source_range,&phase));
  }
  return terminal;
}
void phase_release(void *opaque) noexcept {
  auto *owner = static_cast<CopyNativePhase *>(opaque)->owner;
  owner->pending.fetch_sub(1, std::memory_order_release); owner->decref();
}
Status valid_binding(CopyInvocationState *state, CopyResourceId id,
                     bool recover_writer = false) noexcept {
  if (!state) return Status::FailedPrecondition("Apple copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state->mutable_slot(recover_writer));
  return id.value >= state->bindings.size() ? Status::InvalidArgument("Apple copy resource ID is invalid") : Status::Ok();
}
CopyBatchReport report(const CopyInvocationState *state) noexcept {
  if (!state) return {};
  std::lock_guard lock(state->observations);
  auto result = state->report;
  for (std::uint32_t id = 0; id < state->submitted; ++id) {
    const auto native = copy_recording_report(*state->phases[id]->recording);
    result.native_calls += native.native_calls; result.phase_completions += native.phase_completions;
    result.host_waits += native.host_waits;
  }
  auto usage = state->program->accounting.usage(mem::system_resource());
  HYLE_FATAL_ASSERT(usage.has_value()); result.metadata_bytes = usage->live_bytes;
  return result;
}
Status wait(CopyInvocationState *state) noexcept {
  if (!state) return Status::Ok();
  Status result;
  for (std::uint32_t id = 0; id < state->submitted; ++id) {
    auto &phase = *state->phases[id];
    // Block only on the original native recording's own observer, leaving
    // the shared public Completion free for serialized worker preflight.
    auto observed = poll_copy_recording(*phase.recording);
    if (observed && *observed == CompletionPoll::kPending) ++state->report.host_waits;
    result.Update(wait_copy_recording(*phase.recording));
    { std::lock_guard lock(state->observations); result.Update(phase.completion.wait()); }
    if (phase.retired.load(std::memory_order_acquire)) result.Update(phase.terminal);
    else result.Update(Status::Unavailable("Apple copy actual retirement is unproven"));
  }
  result.Update(state->tail); if (!result.ok()) state->failed = true;
  return result;
}
} // namespace
} // namespace hyle::compute::apple::detail

namespace hyle::compute::apple {
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
  return detail::CopyStreamAccess::prepare_program(detail::ContextAccess::runtime(context),specs,std::move(options));
}
Result<CopyBatchProgram> detail::CopyStreamAccess::prepare_program(std::shared_ptr<Runtime> runtime,
    std::span<const CopyResourceSpec> specs, CopyBatchOptions options) noexcept {
  if (!runtime || !options.allocator.valid() || options.max_resources == 0 ||
      options.max_copies == 0 || options.max_copies > UINT32_MAX / 2 || options.max_phases == 0 ||
      options.max_phases > 64 || options.max_bytes == 0 || specs.size() > options.max_resources)
    return std::unexpected(Status::InvalidArgument("Apple copy program limits are invalid"));
  { std::lock_guard lock(runtime->queue_mutex);
    if (!runtime->accepting) return std::unexpected(Status::FailedPrecondition("Apple copy Context is closed"));
    if (!runtime->failure.ok()) return std::unexpected(runtime->failure); }
  for (const auto &spec : specs) if (!detail::is_copy_resource(spec.resource))
    return std::unexpected(Status::InvalidArgument("Apple copy program resource Device differs"));
  auto base = options.allocator;
  if (options.resource_budget.valid()) { HYLE_ASSIGN_OR_RETURN(base, options.resource_budget.metadata_allocator()); }
  const std::array<ResourceBudgetLimit, 1> limits{{{mem::system_resource(), SIZE_MAX}}};
  // Original ResourceBudget retains its accounting owner through every actual
  // allocation, including the Run control block freed after its last callback.
  HYLE_ASSIGN_OR_RETURN(auto accounting, ResourceBudget::create({.limits = limits, .allocator = base}));
  HYLE_ASSIGN_OR_RETURN(auto allocator, accounting.metadata_allocator());
  auto *storage = allocator.allocate(sizeof(detail::CopyProgramState), alignof(detail::CopyProgramState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Apple copy program metadata"));
  auto *state = std::construct_at(static_cast<detail::CopyProgramState *>(storage), std::move(accounting), allocator, std::move(options));
  CopyBatchProgram result(state); state->runtime = runtime;
  try { state->specs.assign(specs.begin(), specs.end()); }
  catch (...) { return std::unexpected(Status::ResourceExhausted("Apple copy endpoint metadata")); }
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
  if (!prepared.state_) return std::unexpected(Status::InvalidArgument("Apple copy program is empty"));
  auto *program = prepared.state_; const auto allocator = program->allocator;
  auto *storage = allocator.allocate(sizeof(detail::CopyInvocationState), alignof(detail::CopyInvocationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Apple copy invocation metadata"));
  auto *state = std::construct_at(static_cast<detail::CopyInvocationState *>(storage), program);
  CopyBatchInvocation result(state);
  const auto n = program->specs.size(); const auto r = program->options.max_copies, g = program->options.max_phases;
  if (n > SIZE_MAX / g) return std::unexpected(Status::ResourceExhausted("Apple copy role capacity overflows"));
  try {
    state->bindings.resize(n); state->physical.resize(n); state->intervals.resize(static_cast<std::size_t>(r) * 2);
    state->copies.resize(r); state->closure.resize(g); state->phases.reserve(g); state->plans.resize(g);
    state->order.resize(n); state->leader.resize(n); state->last_write.resize(n); state->producers.resize(static_cast<std::size_t>(r) * 2);
    state->roles.resize(n * g); state->root_uses.resize(static_cast<std::size_t>(r) * 2);
    state->written.resize(r); state->ranges.resize(r); state->tickets.resize(g);
  } catch (...) { return std::unexpected(Status::ResourceExhausted("Apple copy fixed metadata")); }
  for (std::uint32_t id = 0; id < g; ++id) {
    auto *memory = allocator.allocate(sizeof(detail::CopyNativePhase), alignof(detail::CopyNativePhase));
    if (!memory) return std::unexpected(Status::ResourceExhausted("Apple copy phase metadata"));
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
  auto *state = invocation.state_;
  if (!state) return std::unexpected(Status::FailedPrecondition("Apple copy invocation is empty"));
  std::lock_guard lock(state->observations);
  // Recover only original successful retired content; admission stays closed.
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state, id, true));
  for (std::uint32_t index = 0; index < state->submitted; ++index) {
    const auto &phase = *state->phases[index];
    if (!phase.retired.load(std::memory_order_acquire))
      return std::unexpected(Status::Unavailable("Apple copy writer retirement is unproven"));
    if (!phase.terminal.ok()) return std::unexpected(phase.terminal);
  }
  auto &binding = state->bindings[id.value];
  if (!binding.write.valid()) return std::unexpected(Status::FailedPrecondition("Apple copy binding has no writer"));
  HYLE_ASSIGN_OR_RETURN(auto unique, binding.write.view());(void)unique;
  auto writer = std::move(binding.write); binding = {}; state->physical[id.value] = {}; state->prepared = false;
  return writer;
}
Result<compute::detail::PhysicalCopyResource> detail::CopyStreamAccess::physical(
    CopyBatchInvocation &invocation, mem::ConstBufferView view, bool writable) noexcept {
  if (!invocation.state_) return std::unexpected(Status::FailedPrecondition("Apple copy invocation is empty"));
  HYLE_ASSIGN_OR_RETURN(auto span, detail::copy_span(view,
      writable ? mem::HostAccess::kWrite : mem::HostAccess::kRead)); return span.physical;
}
Status detail::CopyStreamAccess::observe(CopyBatchInvocation &invocation) noexcept {
  auto *state = invocation.state_;
  if (!state) return Status::FailedPrecondition("Apple copy invocation is empty");
  std::lock_guard lock(state->observations); Status result = state->tail;
  for (std::uint32_t index = 0; index < state->submitted; ++index) {
    auto &phase = *state->phases[index]; auto observed = phase.completion.poll();
    if (!observed) result.Update(observed.error());
    else if (*observed == CompletionPoll::kPending || !phase.retired.load(std::memory_order_acquire))
      result.Update(Status::Unavailable("Apple copy retirement is pending"));
    else result.Update(phase.terminal);
  }
  return result;
}
Status CopyBatchInvocation::bind_source(CopyResourceId id, const ReadLease &source) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  if(state_->result_sources && state_->result_sources->results.owns(id.value))return Status::FailedPrecondition("native result binding needs reset before replacement");
  HYLE_ASSIGN_OR_RETURN(auto view, source.view());
  const auto &spec = state_->program->specs[id.value];
  if (!detail::copy_source_resource_matches(spec.resource, view.resource()) ||
      view.size() > spec.capacity_bytes)
    return Status::InvalidArgument("Apple copy source endpoint/capacity differs");
  HYLE_ASSIGN_OR_RETURN(auto span, detail::copy_span(view, mem::HostAccess::kRead));
  auto &binding = state_->bindings[id.value]; binding = {}; binding.read = source; binding.source = view;
  binding.root = compute::detail::LeaseAccess::identity(source); state_->physical[id.value] = span.physical;
  state_->prepared = false; return Status::Ok();
}
Status CopyBatchInvocation::bind_source(std::span<const ResultCopyBinding>rows,OwnedResult<Results>&&result)noexcept {
  if(!state_)return Status::FailedPrecondition("apple result-copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state_->mutable_slot());
  HYLE_ASSIGN_OR_RETURN(auto prefix,compute::detail::RetainedCompletion::native_dependency(result.completion()));
  HYLE_RETURN_IF_ERROR(detail::qualify_recording_dependency(state_->program->runtime,prefix));
  if(!state_->result_sources){
    auto*extra=state_->allocator().construct<detail::ResultSources>(state_->allocator());
    if(!extra)return Status::ResourceExhausted("native result-copy extension metadata");
    const auto fail=[&](Status error){state_->allocator().destroy(extra);return error;};
    auto reserved=extra->results.reserve(state_->bindings.size());if(!reserved.ok())return fail(reserved);
    try{extra->spans.resize(state_->bindings.size());extra->phase_inputs.reserve(state_->program->options.max_phases);extra->range_inputs.reserve(state_->program->options.max_copies);
      extra->real_copies.resize(state_->program->options.max_copies);extra->real_intervals.resize(std::size_t(state_->program->options.max_copies)*2);extra->real_closure.resize(state_->program->options.max_phases);}
    catch(...){return fail(Status::ResourceExhausted("native result-copy fixed descriptors"));}
    state_->result_sources=extra;
  }
  auto*extra=state_->result_sources;
  HYLE_RETURN_IF_ERROR(extra->results.validate(rows,state_->program->specs,result));
  auto producer=extra->results.take(rows,std::move(result));
  for(const auto&row:rows){auto&binding=state_->bindings[row.resource.value];binding={};
    binding.root=extra->results.root(row.resource.value);
    const auto bytes=row.description.required_bytes();HYLE_FATAL_ASSERT(bytes.has_value());
    // Logical capacity only. No native address or successful data permission
    // is asserted here; the original FIFO Job proves every actual coordinate.
    state_->physical[row.resource.value]={0,0,0,*bytes,4};
  }
  state_->bindings[rows.front().resource.value].producer=std::move(producer);
  state_->prepared=false;return Status::Ok();
}
Status CopyBatchInvocation::bind_mutable(CopyResourceId id, WriteLease &&destination) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  if(state_->result_sources && state_->result_sources->results.owns(id.value))return Status::FailedPrecondition("native result binding needs reset before replacement");
  HYLE_ASSIGN_OR_RETURN(auto view, compute::detail::LeaseAccess::ordered_view(destination));
  const auto &spec = state_->program->specs[id.value];
  if (view.resource() != spec.resource || view.size() > spec.capacity_bytes)
    return Status::InvalidArgument("Apple copy mutable endpoint/capacity differs");
  HYLE_ASSIGN_OR_RETURN(auto span, detail::copy_span(view, mem::HostAccess::kWrite));
  auto &binding = state_->bindings[id.value]; binding = {}; binding.root = compute::detail::LeaseAccess::identity(destination);
  binding.write = std::move(destination); binding.source = view; binding.destination = view;
  state_->physical[id.value] = span.physical; state_->prepared = false; return Status::Ok();
}
Status CopyBatchInvocation::bind_producer(CopyResourceId id, Completion &&producer) noexcept {
  HYLE_RETURN_IF_ERROR(detail::valid_binding(state_, id));
  if(state_->result_sources && state_->result_sources->results.owns(id.value))return Status::FailedPrecondition("native result binding needs reset before replacement");
  if (!state_->bindings[id.value].root) return Status::FailedPrecondition("Apple copy producer resource is unbound");
  HYLE_ASSIGN_OR_RETURN(auto dependency, compute::detail::RetainedCompletion::native_dependency(producer));
  HYLE_RETURN_IF_ERROR(detail::qualify_recording_dependency(state_->program->runtime, dependency));
  state_->bindings[id.value].producer = std::move(producer); state_->prepared = false; return Status::Ok();
}
Status CopyBatchInvocation::prepare_replay(std::span<const CopyPhase> phases, std::span<const CopyRange> ranges) noexcept {
  if (!state_) return Status::FailedPrecondition("Apple copy invocation is empty");
  HYLE_RETURN_IF_ERROR(state_->mutable_slot()); state_->prepared = false;
  if(state_->result_sources)for(const auto id:state_->result_sources->results.ids()){
    const auto bytes=state_->result_sources->results.description(id.value).required_bytes();
    HYLE_FATAL_ASSERT(bytes.has_value());state_->physical[id.value]={0,0,0,*bytes,4};
  }
  for (const auto &range : ranges) if (range.source.value >= state_->bindings.size() || range.destination.value >= state_->bindings.size() ||
      !state_->bindings[range.source.value].root || !state_->bindings[range.destination.value].root)
    return Status::InvalidArgument("Apple copy resource is unbound");
  HYLE_ASSIGN_OR_RETURN(auto report, compute::detail::validate_copy_ranges(state_->physical, phases, ranges,
      state_->program->options, {state_->intervals, state_->copies, state_->closure},
      state_->result_sources?state_->result_sources->results.ids():std::span<const CopyResourceId>{}));
  HYLE_RETURN_IF_ERROR(compute::detail::prepare_copy_control(state_->bindings, phases,
      std::span(state_->copies).first(report.active_copies), {state_->order, state_->leader, state_->last_write,
      state_->producers, state_->roles, state_->root_uses, state_->written, state_->ranges, state_->plans},
      state_->result_sources?state_->result_sources->results.ids():std::span<const CopyResourceId>{}));
  if(state_->result_sources){auto&extra=*state_->result_sources;
    extra.phase_inputs.assign(phases.begin(),phases.end());extra.range_inputs.assign(ranges.begin(),ranges.end());extra.resolved=false;
  }
  for (std::size_t id = 0; id < phases.size(); ++id) {
    const auto &plan = state_->plans[id];
    const auto native_ranges=std::span(state_->ranges).subspan(plan.first,plan.copies);
    if(state_->result_sources && !state_->result_sources->results.ids().empty()){
      HYLE_RETURN_IF_ERROR(detail::bind_result_copy_recording(*state_->phases[id]->recording,native_ranges));
    }else{
      HYLE_RETURN_IF_ERROR(detail::bind_copy_recording(*state_->phases[id]->recording,native_ranges,
          plan.dependency_mask!=0 || plan.producer_count!=0));
    }
    HYLE_RETURN_IF_ERROR(detail::set_copy_preflight(*state_->phases[id]->recording,
        &detail::phase_preflight, state_->phases[id]));
    state_->phases[id]->completion = {}; state_->phases[id]->terminal = Status::Ok();
    state_->phases[id]->retired.store(false, std::memory_order_relaxed);
  }
  state_->phase_count = static_cast<std::uint32_t>(phases.size()); state_->submitted = 0;
  report.native_metadata_unknown = true; state_->report = report; state_->tail = Status::Ok(); state_->prepared = true;
  return Status::Ok();
}
Result<CopyBatchSubmission> CopyBatchInvocation::submit() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("Apple copy invocation is empty"));
  HYLE_RETURN_IF_ERROR(state_->mutable_slot());
  if (!state_->prepared) return std::unexpected(Status::FailedPrecondition("Apple copy replay is not prepared"));
  // Revalidate every borrowed producer before accepting any phase, without a
  // host wait. The original native owner FIFO is the qualified route.
  for (std::uint32_t id = 0; id < state_->phase_count; ++id) {
    const auto &plan = state_->plans[id];
    for (std::size_t index = plan.producer_first; index < plan.producer_first + plan.producer_count; ++index) {
      HYLE_ASSIGN_OR_RETURN(auto dependency, compute::detail::RetainedCompletion::native_dependency(state_->bindings[state_->producers[index]].producer));
      HYLE_RETURN_IF_ERROR(detail::qualify_recording_dependency(state_->program->runtime, dependency));
    }
  }
  HYLE_RETURN_IF_ERROR(compute::detail::admit_copy_control(state_->uses, state_->bindings,
      std::span(state_->plans).first(state_->phase_count), state_->root_uses, state_->tickets,
      state_->result_sources?state_->result_sources->results.ids():std::span<const CopyResourceId>{}));
  std::lock_guard observation_lock(state_->observations);
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

  }
  for (auto &ticket : state_->tickets) ticket = {};
  if (!state_->tail.ok()) state_->failed = true;
  if (state_->submitted == 0 && !state_->tail.ok()) return std::unexpected(state_->tail);
  ++state_->users; state_->incref(); return CopyBatchSubmission(state_);
}
Status CopyBatchInvocation::reset() noexcept {
  if (!state_) return Status::Ok();
  if (state_->users != 1 || state_->busy()) return Status::Unavailable("Apple copy slot is retained");
  for (auto *phase : state_->phases) phase->completion = {};
  for (auto *phase : state_->phases) HYLE_RETURN_IF_ERROR(detail::clear_copy_recording(*phase->recording));
  for (auto &binding : state_->bindings) binding = {};
  if(state_->result_sources){state_->result_sources->results.reset();for(auto&span:state_->result_sources->spans)span={};}
  std::fill(state_->physical.begin(), state_->physical.end(), compute::detail::PhysicalCopyResource{});
  state_->prepared = false; state_->failed = false; state_->submitted = 0; state_->phase_count = 0; state_->tail = Status::Ok();
  return Status::Ok();
}
Status CopyBatchInvocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  return state_->busy() ? Status::Unavailable("Apple copy has native work") : Status::Ok();
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
    return std::unexpected(Status::InvalidArgument("Apple copy output IDs are invalid"));
  if (phase.value >= state_->submitted) return std::unexpected(state_->tail.ok()
      ? Status::Unavailable("Apple copy phase was not submitted") : state_->tail);
  std::lock_guard lock(state_->observations);
  auto &observed = *state_->phases[phase.value]; auto poll = observed.completion.poll();
  if (!poll) { state_->failed = true; return std::unexpected(poll.error()); }
  if (*poll == CompletionPoll::kPending || !observed.retired.load(std::memory_order_acquire))
    return std::unexpected(Status::Unavailable("Apple copy output is pending"));
  if (!observed.terminal.ok()) return std::unexpected(observed.terminal);
  if (state_->last_write[state_->leader[id.value]] != phase.value)
    return std::unexpected(Status::Unavailable("Apple copy block has a later write or no output"));
  auto &binding = state_->bindings[id.value];
  if (offset > binding.source.size() || bytes > binding.source.size() - offset)
    return std::unexpected(Status::OutOfRange("Apple copy output exceeds binding"));
  HYLE_RETURN_IF_ERROR(compute::detail::copy_output_coverage(std::span(state_->written).first(state_->report.active_copies), id, phase, offset, bytes));
  for (const auto &span : std::span(state_->written).first(state_->report.active_copies)) {
    if (span.binding != id.value || span.phase > phase.value || span.end <= offset || span.begin >= offset + bytes) continue;
    auto &prior = *state_->phases[span.phase]; auto result = prior.completion.poll();
    if (!result) return std::unexpected(result.error());
    if (*result == CompletionPoll::kPending || !prior.retired.load(std::memory_order_acquire))
      return std::unexpected(Status::Unavailable("Apple copy output write is pending"));
    if (!prior.terminal.ok()) return std::unexpected(prior.terminal);
  }
  if (binding.write.valid()) { HYLE_ASSIGN_OR_RETURN(binding.read, compute::detail::LeaseAccess::publish_after_writes(binding.write)); }
  return binding.read.subview(offset, bytes);
}
Status CopyBatchSubmission::wait() noexcept { return detail::wait(state_); }
Status CopyBatchSubmission::request_cancel() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  std::lock_guard lock(state_->observations);
  Status result; for (std::uint32_t id = 0; id < state_->submitted; ++id) result.Update(state_->phases[id]->completion.request_cancel());
  return result;
}
bool CopyBatchSubmission::in_flight() const noexcept { return state_ && state_->busy(); }
CopyBatchReport CopyBatchSubmission::report() const noexcept { return detail::report(state_); }
Status detail::CopyUpdateAccess::qualify(const CopyBatchInvocation &invocation,
                                         const CopyBatchSubmission &submission) noexcept {
  auto *state = invocation.state_;
  if (!state || state != submission.state_)
    return Status::InvalidArgument("Apple copy update controls do not own the same accepted slot");
  if (state->users != 2 || state->submitted == 0)
    return Status::FailedPrecondition("Apple copy update needs its sole original accepted submission");
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
  std::lock_guard lock(state->observations);
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
  std::lock_guard lock(state->observations);
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
Status detail::CopyStreamAccess::preflight(const CopyBatchUpdate &owner) noexcept {
  const auto &invocation = detail::CopyUpdateAccess::invocation(owner);
  const auto &submission = detail::CopyUpdateAccess::submission(owner);
  HYLE_RETURN_IF_ERROR(detail::CopyUpdateAccess::qualify(invocation, submission));
  auto *state = invocation.state_;
  std::lock_guard lock(state->observations);
  HYLE_RETURN_IF_ERROR(state->tail);
  for (std::uint32_t index = 0; index < state->submitted; ++index) {
    const auto &phase = *state->phases[index];
    // Read cached/native identity hints without settling its public observer.
    HYLE_ASSIGN_OR_RETURN(auto native,
        compute::detail::RetainedCompletion::native_dependency(phase.completion));
    if (native.terminal_hint) HYLE_RETURN_IF_ERROR(*native.terminal_hint);
    HYLE_ASSIGN_OR_RETURN(auto ready, detail::poll_copy_recording(*phase.recording));
    if (ready != CompletionPoll::kComplete)
      return Status::Unavailable("Apple pending source FIFO recording is not retired");
    if (phase.retired.load(std::memory_order_acquire)) HYLE_RETURN_IF_ERROR(phase.terminal);
  }
  if (state->failed) return Status::FailedPrecondition("Apple pending copy source failed");
  return Status::Ok();
}
Result<std::size_t> detail::CopyStreamAccess::retirement_sources(
    CopyBatchUpdate &owner, std::span<Completion *> output) noexcept {
  auto &invocation=detail::CopyUpdateAccess::invocation(owner);
  auto &submission=detail::CopyUpdateAccess::submission(owner);
  HYLE_RETURN_IF_ERROR(detail::CopyUpdateAccess::qualify(invocation,submission));
  auto *state=invocation.state_;
  std::lock_guard lock(state->observations);
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
  if(output.size()<count)return std::unexpected(Status::ResourceExhausted("Apple retirement source capacity"));
  std::size_t at=0;
  for(std::size_t i=0;i<state->bindings.size();++i)
    if(used(i))output[at++]=&state->bindings[i].producer;
  for(std::uint32_t i=0;i<state->submitted;++i)output[at++]=&state->phases[i]->completion;
  return count;
}
void detail::CopyStreamAccess::serialize(CopyBatchUpdate &owner,
    void (*action)(void *) noexcept, void *context) noexcept {
  auto *state=detail::CopyUpdateAccess::invocation(owner).state_;
  HYLE_FATAL_ASSERT(state && action);
  std::lock_guard lock(state->observations);
  action(context);
}
Status detail::CopyStreamAccess::admit_read(LeaseUseBatch &uses,
    const CopyBatchUpdate &owner, CopyResourceId id) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto original, input(owner, id));
  return compute::detail::LeaseAccess::ordered_read_unique(uses, *original.writer);
}

} // namespace hyle::compute::apple
