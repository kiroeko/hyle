#include "plan.h"
#include "paged.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <mutex>
#include <new>
#include <utility>

#include "../compute/completion_internal.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/graph/detail/lease_submission.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::cpu_apple {
namespace ca = compute::apple;
namespace mx = mlx::core;
namespace cd = compute::detail;
namespace detail {

struct ValueRun final {
  compute::ResourceBudget::Reservation host_charge;
  mem::Buffer host;
  compute::ConstTensorView host_input, host_value;
  std::optional<mx::array> native_input, native_value, boundary;
  std::optional<ca::WeightBatch> weight;
  std::uint64_t weight_id = 0;
  graph::Invocation repack;
};
struct FragmentRun final {
  graph::Invocation cpu;
  ca::Invocation apple;
  compute::ResultLease<ca::Results> lease;
  std::vector<mx::array> arguments; // Vendor bind carrier, sized cold.
};
struct RunState final : util::RefCounted<RunState> {
  explicit RunState(std::shared_ptr<const Plan> source)
      : plan(std::move(source)), allocator(plan->allocator),
        values(mem::StdAllocator<ValueRun>(allocator)),
        fragments(mem::StdAllocator<FragmentRun>(allocator)),
        pages(nullptr,PageSlotsDelete{allocator}), frontier(allocator) {}
  std::shared_ptr<const Plan> plan;
  mem::AllocatorRef allocator;
  graph::detail::LeaseSubmission leases;
  Vector<ValueRun> values;
  Vector<FragmentRun> fragments;
  std::unique_ptr<PageSlots,PageSlotsDelete> pages;
  sched::Scheduler *scheduler = nullptr;
  compute::Queue *cpu_queue = nullptr;
  sched::Event progress, finished;
  graph::detail::JointPipelineFrontier frontier;
  std::mutex pipeline_mutex, lease_mutex;
  sched::ConditionVariable pipeline_ready;
  Status pipeline_error;
  bool cpu_stopped = false, pipeline_accepted = false;
  std::mutex retirement_mutex;
  cd::CompletionLifetime abandoned{};
  Status hint, terminal;
  std::atomic<bool> running{false}, cancel{false};
  bool done = false, retired = false, prepared = false;
  std::size_t host_bytes = 0, boundary_bytes = 0;
  ReplayReport report;
  void destroy() noexcept {
    const auto owner = allocator;
    std::destroy_at(this);
    owner.deallocate(this, sizeof(RunState), alignof(RunState));
  }
};
struct InvocationState final {
  mem::AllocatorRef allocator;
  RunState *run = nullptr;
  cd::ResultSlot<Results> result;
  cd::ResultWrite<Results> write;
  bool closed = false;
  ~InvocationState() { if (run != nullptr) run->decref(); }
};
} // namespace detail
namespace {
using detail::RunState;
using detail::InvocationState;

void release_invocation(InvocationState *state) noexcept {
  if (state == nullptr) return;
  const auto allocator = state->allocator;
  std::destroy_at(state);
  allocator.deallocate(state, sizeof(InvocationState), alignof(InvocationState));
}
Status admit(InvocationState *state,bool accepting_source=false) noexcept {
  if (state == nullptr || state->closed)
    return Status::FailedPrecondition("CPU+Apple invocation is empty or closed");
  if (state->run->running.load(std::memory_order_acquire))
    return Status::FailedPrecondition("CPU+Apple invocation is running");
  if (state->run->retired)
    return Status::FailedPrecondition("CPU+Apple failed invocation is retired");
  if(!accepting_source && state->run->pages)HYLE_RETURN_IF_ERROR(state->run->pages->reusable());
  if (!state->write.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto write, state->result.begin());
    state->write = std::move(write);
  }
  return Status::Ok();
}
Result<std::size_t> input_index(const RunState &run, ValueId id) noexcept {
  if (!id.valid() || id.ordinal() >= run.plan->values.size() ||
      run.plan->values[id.ordinal()].id != id || id.kind() != ValueKind::kInput)
    return std::unexpected(Status::InvalidArgument("foreign or non-input CPU+Apple value"));
  return id.ordinal();
}
Result<compute::TensorView> host_tensor(RunState &run, std::size_t id) noexcept {
  return compute::TensorView::create(run.plan->values[id].carrier, run.values[id].host.view());
}
Status allocate_host(RunState &run, std::size_t id) noexcept {
  auto &value = run.values[id];
  if (value.host.valid()) return Status::Ok();
  const auto &description = run.plan->values[id].carrier;
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if (bytes > run.plan->environment.host_workspace_budget_bytes - run.host_bytes)
    return Status::ResourceExhausted("CPU+Apple host boundary exceeds workspace limit");
  compute::ResourceBudget::Reservation charge;
  if (run.plan->budget) {
    HYLE_ASSIGN_OR_RETURN(charge, run.plan->budget.reserve(run.plan->environment.host, bytes,
        compute::ResourceBudgetCategory::kWorkspace));
  }
  HYLE_ASSIGN_OR_RETURN(auto buffer, mem::Buffer::create(run.plan->environment.host, bytes,
      std::max(alignof(std::max_align_t), description.required_alignment())));
  value.host = std::move(buffer);
  value.host_charge = std::move(charge);
  run.host_bytes += bytes;
  return Status::Ok();
}
Status allocate_boundary(RunState &run, std::size_t id) noexcept {
  auto &value = run.values[id];
  if (value.boundary) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto bytes, run.plan->values[id].carrier.required_bytes());
  if (bytes > run.plan->environment.native_boundary_budget_bytes - run.boundary_bytes)
    return Status::ResourceExhausted("CPU+Apple native boundary exceeds limit");
  HYLE_ASSIGN_OR_RETURN(auto array, run.plan->environment.native->allocate(
      run.plan->values[id].carrier, run.plan->budget));
  value.boundary = std::move(array);
  run.boundary_bytes += bytes;
  return Status::Ok();
}

Status native_input(const detail::Value &spec, const mx::array &array) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto carrier, ca::input_spec(spec.carrier));
  if (array.status() != mx::array::available || array.shape() != carrier.shape ||
      array.dtype() != carrier.dtype || !array.flags().row_contiguous)
    return Status::InvalidArgument("CPU+Apple input must be a ready exact dense native carrier");
  if (spec.cpu_use && spec.description != spec.carrier)
    return Status::Unimplemented("strided CPU input requires its original host tensor binding");
  return Status::Ok();
}
Status ensure_leases(RunState& run) noexcept {
  if (run.leases.valid())return Status::Ok();
  {
    const auto inputs = run.plan->inputs.size(), fragments = run.plan->fragments.size();
    if (inputs > (SIZE_MAX - fragments) / 2) return Status::ResourceExhausted("CPU+Apple lease prefix capacity");
    auto prefixes = 2 * inputs + fragments;
    for (const auto &fragment : run.plan->fragments) {
      if (fragment.outputs.size() > SIZE_MAX - prefixes)
        return Status::ResourceExhausted("CPU+Apple lease prefix capacity");
      prefixes += fragment.outputs.size();
    }
    std::size_t pages=0;
    for(const auto& fragment:run.plan->paged){
      const auto refs=fragment.binding.program.schema().capacity().max_page_refs;
      if(refs>(SIZE_MAX-pages)/2)return Status::ResourceExhausted("native page guard capacity overflow");
      pages+=2*std::size_t(refs);
    }
    HYLE_ASSIGN_OR_RETURN(run.leases, graph::detail::LeaseSubmission::create(inputs, prefixes,
        {.allocator = run.allocator, .resource_budget = run.plan->budget, .max_page_reads = pages}));
  }
  return Status::Ok();
}

Status add_copy(ReplayReport &report, std::size_t bytes) noexcept {
  if (bytes > SIZE_MAX - report.host_copy_bytes)
    return Status::ResourceExhausted("CPU+Apple copy report overflows");
  ++report.host_copy_count;
  report.host_copy_bytes += bytes;
  return Status::Ok();
}

Status await_completion(RunState &run, const compute::Completion &completion) noexcept {
  if (run.leases.valid()) {
    std::lock_guard lock(run.lease_mutex);
    HYLE_RETURN_IF_ERROR(run.leases.protect(completion));
  }
  for (;;) {
    const auto ready = completion.poll();
    if (!ready) return ready.error();
    if (*ready == compute::CompletionPoll::kComplete) return completion.wait();
    // The current Apple owner drains/checks native work. A pending native wait
    // never strands the Sched worker that hosts this coarse progress fiber.
    run.progress.wait_for(std::chrono::milliseconds(1));
  }
}
void publish_terminal(RunState &run, Status status, bool accepted) noexcept {
  cd::CompletionLifetime abandoned;
  Status hint;
  {
    std::lock_guard lock(run.retirement_mutex);
    run.terminal = status;
    run.done = true;
    run.retired = accepted && !status.ok();
    abandoned = std::exchange(run.abandoned, {});
    hint = run.hint;
  }
  run.finished.signal();
  if (abandoned.valid()) {
    if (status.ok() && !hint.ok()) status = hint;
    abandoned.terminal(abandoned.state, status);
    abandoned.release(abandoned.state);
  }
  run.running.store(false, std::memory_order_release);
}
const cd::CompletionVTable completion_vtable{
    .destroy = [](void *opaque, cd::CompletionLifetime lifetime, Status hint) noexcept {
      auto &run = *static_cast<RunState *>(opaque);
      Status terminal;
      bool done;
      {
        std::lock_guard lock(run.retirement_mutex);
        done = run.done;
        if (done) terminal = run.terminal;
        else { run.abandoned = lifetime; run.hint = hint; }
      }
      if (done && lifetime.valid()) {
        if (terminal.ok() && !hint.ok()) terminal = hint;
        lifetime.terminal(lifetime.state, terminal);
        lifetime.release(lifetime.state);
      }
      run.decref();
    },
    .poll = [](void *opaque) noexcept -> Result<compute::CompletionPoll> {
      auto &run = *static_cast<RunState *>(opaque);
      if (run.running.load(std::memory_order_acquire)) {
        const auto error = run.leases.observed_error();
        if (!error.ok()) return std::unexpected(error);
        return compute::CompletionPoll::kPending;
      }
      std::lock_guard lock(run.retirement_mutex);
      if (!run.done) return compute::CompletionPoll::kPending;
      if (!run.terminal.ok()) return std::unexpected(run.terminal);
      return compute::CompletionPoll::kComplete;
    },
    .wait = [](void *opaque) noexcept {
      auto &run = *static_cast<RunState *>(opaque);
      while (run.running.load(std::memory_order_acquire)) {
        const auto error = run.leases.observed_error();
        if (!error.ok()) return error;
        run.finished.wait_for(std::chrono::milliseconds(1));
      }
      std::lock_guard lock(run.retirement_mutex);
      return run.terminal;
    },
    .request_cancel = [](void *opaque) noexcept {
      auto &run = *static_cast<RunState *>(opaque);
      {
        std::lock_guard lock(run.pipeline_mutex);
        run.cancel.store(true, std::memory_order_release);
      }
      run.pipeline_ready.notify_all();
      return Status::Ok();
    }};

Status prepare_inputs(RunState &run, bool &accepted) noexcept {
  // Bind metadata/owners are fixed. No Provider selection, tracing, graph
  // traversal, capacity growth or CPU weight upload occurs in this loop.
  for (auto id : run.plan->inputs) {
    if (run.cancel.load(std::memory_order_acquire))
      return Status::Cancelled("CPU+Apple cancelled before boundary preparation");
    auto &value = run.values[id];
    const auto &spec = run.plan->values[id];
    if(run.pages && run.pages->page_input(id) && !value.native_input)continue;
    if (value.native_input) {
      value.native_value = value.native_input;
      if (spec.cpu_use) {
        HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
        HYLE_RETURN_IF_ERROR(run.plan->environment.native->copy_to_host(*value.native_input, destination));
        value.host_value = destination.as_const();
      }
    } else {
      value.host_value = value.host_input;
      if (spec.apple_use) {
        auto source = value.host_input;
        if (value.repack.valid()) {
          HYLE_ASSIGN_OR_RETURN(auto completion, value.repack.submit(*run.cpu_queue));
          accepted = true;
          HYLE_RETURN_IF_ERROR(await_completion(run, completion));
          HYLE_ASSIGN_OR_RETURN(source, value.repack.output(spec.repack_output));
        }
        HYLE_RETURN_IF_ERROR(run.plan->environment.native->copy_from_host(*value.boundary, source));
        value.native_value = value.boundary;
      }
    }
  }
  return Status::Ok();
}
Status drive_fragment(RunState &run, std::size_t index, bool &accepted) noexcept {
  if (run.cancel.load(std::memory_order_acquire))
    return Status::Cancelled("CPU+Apple cancelled before the next coarse fragment");
  const auto &fragment = run.plan->fragments[index];
  auto &slot = run.fragments[index];
  if (fragment.target == kCpuTarget) {
    HYLE_ASSIGN_OR_RETURN(auto completion, slot.cpu.submit(*run.cpu_queue));
    accepted = true;
    HYLE_RETURN_IF_ERROR(await_completion(run, completion));
    for (std::size_t port = 0; port < fragment.outputs.size(); ++port) {
      const auto id = fragment.outputs[port];
      auto &value = run.values[id];
      HYLE_ASSIGN_OR_RETURN(auto output, slot.cpu.output(fragment.cpu_outputs[port]));
      if (output.description() != run.plan->values[id].carrier)
        return Status::Internal("CPU fragment output violates its fixed boundary description");
      if (output.storage().native_handle() != value.host.native_handle() ||
          output.storage().resource() != value.host.resource() || output.storage().offset() != 0)
        HYLE_RETURN_IF_ERROR(value.host.view().copy_from(output.storage()));
      HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
      value.host_value = destination.as_const();
      if (run.plan->values[id].apple_use) {
        HYLE_RETURN_IF_ERROR(run.plan->environment.native->copy_from_host(*value.boundary, value.host_value));
        value.native_value = value.boundary;
      }
    }
  } else {
    slot.lease = {};
    auto submitted_result=[&]()->Result<compute::OwnedResult<ca::Results>>{
      if(auto* page=run.pages?run.pages->fragment(index):nullptr){
        const auto qid=fragment.inputs[0];
        if(!run.values[qid].native_value)return std::unexpected(Status::Internal("page query producer did not complete"));
        HYLE_ASSIGN_OR_RETURN(auto query,run.plan->environment.native->import_readonly(
            *run.values[qid].native_value,run.plan->values[qid].carrier,
            {.allocator=run.allocator,.resource_budget=run.plan->budget}));
        return run.pages->submit_fragment(*page,std::move(query));
      }
      slot.arguments.clear();
      for(auto id:fragment.inputs){
        if(!run.values[id].native_value)return std::unexpected(Status::Internal("Apple fragment input was not produced"));
        slot.arguments.push_back(*run.values[id].native_value);
      }
      HYLE_RETURN_IF_ERROR(slot.apple.bind(slot.arguments));return slot.apple.submit();
    }();
    HYLE_ASSIGN_OR_RETURN(auto submitted,std::move(submitted_result));
    accepted = true;
    // Cancellation stops successors, while the original Apple owner drains
    // the already-accepted prefix before publishing its terminal result.
    HYLE_RETURN_IF_ERROR(await_completion(run, submitted.completion()));
    HYLE_ASSIGN_OR_RETURN(slot.lease, submitted.wait());
    const auto arrays = slot.lease->arrays();
    if (arrays.size() != fragment.outputs.size())
      return Status::Internal("Apple fragment result arity differs");
    for (std::size_t port = 0; port < arrays.size(); ++port) {
      const auto id = fragment.outputs[port];
      run.values[id].native_value = arrays[port];
      if (run.plan->values[id].cpu_use) {
        HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
        HYLE_RETURN_IF_ERROR(run.plan->environment.native->copy_to_host(arrays[port], destination));
        run.values[id].host_value = destination.as_const();
      }
    }
  }
  return Status::Ok();
}

using PipelineLane = graph::detail::JointPipelinePlan::Lane;

void drive_lane(RunState &run, PipelineLane lane) noexcept {
  for (;;) {
    std::size_t index;
    {
      std::unique_lock lock(run.pipeline_mutex);
      run.pipeline_ready.wait(lock, [&] {
        return !run.pipeline_error.ok() || run.cancel.load(std::memory_order_acquire) ||
               run.frontier.done() || run.frontier.ready(lane);
      });
      if (run.cancel.load(std::memory_order_acquire))
        run.pipeline_error.Update(Status::Cancelled("joint pipeline cancelled before admission"));
      if (!run.pipeline_error.ok() || run.frontier.done()) break;
      index = run.frontier.take(lane);
    }
    bool accepted = false;
    const auto status = drive_fragment(run, index, accepted);
    {
      std::lock_guard lock(run.pipeline_mutex);
      run.pipeline_accepted = run.pipeline_accepted || accepted;
      run.pipeline_error.Update(status);
      if (status.ok()) run.frontier.complete(run.plan->pipeline, index);
    }
    run.pipeline_ready.notify_all();
  }
}

void drive_cpu_lane(void *opaque) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  drive_lane(run, PipelineLane::kCpu);
  {
    std::lock_guard lock(run.pipeline_mutex);
    run.cpu_stopped = true;
  }
  run.pipeline_ready.notify_all();
  run.decref(); // no resettable per-run state is accessed after publication
}

Status drive_graph(RunState &run, bool &accepted) noexcept {
  HYLE_RETURN_IF_ERROR(prepare_inputs(run, accepted));
  if (!run.plan->pipeline.has_parallel_lanes) {
    for (std::size_t index = 0; index < run.fragments.size(); ++index)
      HYLE_RETURN_IF_ERROR(drive_fragment(run, index, accepted));
  } else {
    run.frontier.reset(run.plan->pipeline);
    run.pipeline_error = Status::Ok();
    run.pipeline_accepted = false;
    run.cpu_stopped = false;
    run.incref();
    const auto launched = run.scheduler->spawn_detached(&drive_cpu_lane, &run, mem::kFiberStackLarge);
    if (!launched.ok()) { run.decref(); return launched; }
    drive_lane(run, PipelineLane::kNative);
    std::unique_lock lock(run.pipeline_mutex);
    run.pipeline_ready.wait(lock, [&] { return run.cpu_stopped; });
    accepted = accepted || run.pipeline_accepted;
    HYLE_RETURN_IF_ERROR(run.pipeline_error);
  }
  return run.cancel.load(std::memory_order_acquire)
      ? Status::Cancelled("joint pipeline cancelled after native drain") : Status::Ok();
}

void drive(void *opaque) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  bool accepted = false;
  const auto status = drive_graph(run, accepted);
  if (run.leases.valid()) run.leases.finish(status, accepted);
  else publish_terminal(run, status, accepted);
  run.decref(); // existing Sched worker's admitted reference
}

Result<ResourceReport> resource_report(const RunState &run) noexcept {
  ResourceReport report;
  if (run.running.load(std::memory_order_acquire))
    return std::unexpected(Status::FailedPrecondition("CPU+Apple resource observation requires an idle slot"));
  report.has_shared_budget = run.plan->budget.valid();
  report.managed_host_workspace_bytes = run.host_bytes;
  report.managed_native_boundary_bytes = run.boundary_bytes;
  report.has_unknown_native_storage = std::ranges::any_of(run.plan->fragments,
      [](const auto &fragment) { return fragment.target == kAppleTarget; });
  if (run.plan->budget) {
    const std::array resources{run.plan->environment.host, run.plan->environment.native->ref()};
    for (std::size_t i = 0; i < resources.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(report.storage[i], run.plan->budget.usage(resources[i]));
    }
    for (std::size_t i = 0; i < run.plan->budget.resource_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto resource, run.plan->budget.resource(i));
      if (resource == resources[0] || resource == resources[1]) continue;
      HYLE_ASSIGN_OR_RETURN(auto usage, run.plan->budget.usage(resource));
      if (usage.live_bytes > SIZE_MAX - report.other_live_bytes)
        return std::unexpected(Status::ResourceExhausted("CPU+Apple resource sum overflows"));
      ++report.other_resource_count;
      report.other_live_bytes += usage.live_bytes;
    }
    report.unattributed = run.plan->budget.unattributed_usage();
  }
  try {
    detail::Vector<ca::NativeStorage> backings{mem::StdAllocator<ca::NativeStorage>(run.allocator)};
    const auto observe = [&](const std::optional<mx::array> &array) -> Status {
      if (!array) return Status::Ok();
      HYLE_ASSIGN_OR_RETURN(auto storage, run.plan->environment.native->storage_info(*array));
      if (storage.buffer != nullptr) backings.push_back(storage);
      return Status::Ok();
    };
    for (std::size_t id = 0; id < run.values.size(); ++id) {
      HYLE_RETURN_IF_ERROR(observe(run.values[id].native_input));
      HYLE_RETURN_IF_ERROR(observe(run.values[id].native_value));
      HYLE_RETURN_IF_ERROR(observe(run.values[id].boundary));
      HYLE_RETURN_IF_ERROR(observe(run.plan->values[id].native_constant));
    }
    std::sort(backings.begin(), backings.end(), [](const auto &a, const auto &b) {
      return std::less<const void *>{}(a.buffer, b.buffer);
    });
    const void *previous = nullptr;
    for (const auto &backing : backings) {
      if (backing.buffer == previous) continue;
      previous = backing.buffer;
      if (backing.backing_bytes > SIZE_MAX - report.observed_native_backing_bytes)
        return std::unexpected(Status::ResourceExhausted("CPU+Apple backing observation overflows"));
      ++report.observed_native_backing_count;
      report.observed_native_backing_bytes += backing.backing_bytes;
    }
    return report;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("CPU+Apple resource report metadata"));
  }
}
} // namespace

Result<Invocation> Executable::create_invocation(sched::Scheduler &scheduler, compute::Queue &cpu_queue) const noexcept {
  if (!state_ || !scheduler.valid() || !state_->cpu_context.owns(cpu_queue))
    return std::unexpected(Status::InvalidArgument("CPU+Apple requires its exact CPU Queue and a live Scheduler"));
  const auto allocator = state_->allocator;
  auto *memory = allocator.allocate(sizeof(InvocationState), alignof(InvocationState));
  if (memory == nullptr) return std::unexpected(Status::ResourceExhausted("CPU+Apple invocation metadata"));
  auto *state = std::construct_at(static_cast<InvocationState *>(memory));
  state->allocator = allocator;
  const auto cleanup = [](InvocationState *pointer) noexcept { release_invocation(pointer); };
  std::unique_ptr<InvocationState, decltype(cleanup)> owner(state, cleanup);
  memory = allocator.allocate(sizeof(RunState), alignof(RunState));
  if (memory == nullptr) return std::unexpected(Status::ResourceExhausted("CPU+Apple run metadata"));
  state->run = std::construct_at(static_cast<RunState *>(memory), state_);
  auto &run = *state->run;
  run.scheduler = &scheduler;
  run.cpu_queue = &cpu_queue;
  run.host_bytes = state_->host_workspace_bytes;
  HYLE_ASSIGN_OR_RETURN(state->result, cd::ResultSlot<Results>::create(allocator));
  try {
    run.values.resize(state_->values.size());
    run.fragments.resize(state_->fragments.size());
    if (state_->pipeline.has_parallel_lanes) {
      HYLE_RETURN_IF_ERROR(run.frontier.prepare(state_->pipeline));
      // Even ordinary borrowed bindings need every independent accepted
      // prefix to retain this run after an early observation/other-lane error.
      HYLE_RETURN_IF_ERROR(ensure_leases(run));
    }
    for (std::size_t id = 0; id < run.values.size(); ++id) {
      const auto &spec = state_->values[id];
      if (!spec.id.valid()) continue;
      if (spec.fragment_output && (spec.producer == kCpuTarget || spec.cpu_use))
        HYLE_RETURN_IF_ERROR(allocate_host(run, id));
      if (spec.producer == kCpuTarget && spec.apple_use && spec.fragment_output)
        HYLE_RETURN_IF_ERROR(allocate_boundary(run, id));
      if (spec.repack.valid()) {
        HYLE_ASSIGN_OR_RETURN(run.values[id].repack, spec.repack.create_invocation());
      }
      if (spec.id.kind() == ValueKind::kConstant) run.values[id].host_value = spec.constant;
    }
    if(!state_->paged.empty()){
      auto* pointer=allocator.construct<detail::PageSlots>(allocator);
      if(!pointer)return std::unexpected(Status::ResourceExhausted("joint page slot owner"));
      run.pages.reset(pointer);
      for(const auto& page:state_->paged){
        const auto& fragment=state_->fragments[page.fragment];
        HYLE_RETURN_IF_ERROR(run.pages->add({state_->environment.apple,&page.initial_program,page.binding.layout},
            page.binding,page.fragment,fragment.inputs,*state_->environment.native,
            *state_->environment.native,state_->workload));
        const auto query=fragment.inputs[0];
        if(state_->values[query].id.kind()==ValueKind::kConstant)
          run.values[query].native_value=state_->values[query].native_constant;
      }
      HYLE_RETURN_IF_ERROR(ensure_leases(run));
    }
    for (std::size_t i = 0; i < run.fragments.size(); ++i) {
      const auto &fragment = state_->fragments[i];
      if (fragment.target == kCpuTarget) {
        HYLE_ASSIGN_OR_RETURN(run.fragments[i].cpu, fragment.cpu.create_invocation());
      } else {
        if(run.pages && run.pages->fragment(i))continue;
        HYLE_ASSIGN_OR_RETURN(run.fragments[i].apple, fragment.apple.create_invocation(allocator));
        run.fragments[i].arguments.reserve(fragment.inputs.size());
      }
    }
    owner.release();
    return Invocation(state);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("CPU+Apple slot metadata allocation"));
  }
}

Results::Results(detail::RunState *state) noexcept : state_(state) { state_->incref(); }
Results::Results(Results &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
Results &Results::operator=(Results &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
Results::~Results() { if (state_ != nullptr) state_->decref(); }
Result<compute::ConstTensorView> Results::output(ValueId id) const noexcept {
  if (state_ == nullptr || !id.valid() || id.ordinal() >= state_->values.size() ||
      state_->plan->values[id.ordinal()].id != id || !state_->plan->values[id.ordinal()].published)
    return std::unexpected(Status::InvalidArgument("foreign or unpublished CPU+Apple output"));
  const auto tensor = state_->values[id.ordinal()].host_value;
  if (!tensor.valid()) return std::unexpected(Status::FailedPrecondition("CPU+Apple output is native; use array()"));
  return tensor;
}
Result<const mx::array &> Results::array(ValueId id) const noexcept {
  if (state_ == nullptr || !id.valid() || id.ordinal() >= state_->values.size() ||
      state_->plan->values[id.ordinal()].id != id || !state_->plan->values[id.ordinal()].published)
    return std::unexpected(Status::InvalidArgument("foreign or unpublished CPU+Apple array output"));
  const auto &value = state_->values[id.ordinal()].native_value;
  if (!value) return std::unexpected(Status::FailedPrecondition("CPU+Apple output is host; use output()"));
  return *value;
}

Invocation::Invocation(Invocation &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
Invocation &Invocation::operator=(Invocation &&other) noexcept {
  if (this != &other) { release_invocation(state_); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
Invocation::~Invocation() { release_invocation(state_); }
Status Invocation::bind(ValueId id, compute::ConstTensorView tensor) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index, input_index(run, id));
  if(run.pages && run.pages->page_input(index))
    return Status::InvalidArgument("native page input requires an exact prepared frame");
  if (!tensor.valid() || tensor.description() != run.plan->values[index].description ||
      !tensor.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return Status::InvalidArgument("CPU+Apple host input description/storage differs");
  HYLE_RETURN_IF_ERROR(graph::detail::validate_effect_tensor_alias(
      run.plan->effect_bindings, tensor.storage()));
  auto &value = run.values[index];
  value.host_input = tensor;
  value.native_input.reset(); value.weight.reset();
  run.prepared = false;
  const auto slot = std::find(run.plan->inputs.begin(), run.plan->inputs.end(), index) - run.plan->inputs.begin();
  run.leases.clear(static_cast<std::size_t>(slot));
  return Status::Ok();
}
Status Invocation::bind(ValueId id, const mx::array &array) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index, input_index(run, id));
  if(run.pages && run.pages->page_input(index))
    return Status::InvalidArgument("native page input requires an exact prepared frame");
  if (!run.plan->effect_bindings.empty())
    return Status::Unimplemented("effect composition requires a checked host footprint for native input alias admission");
  HYLE_RETURN_IF_ERROR(native_input(run.plan->values[index], array));
  auto &value = run.values[index];
  value.native_input = array;
  value.host_input = {}; value.weight.reset();
  run.prepared = false;
  const auto slot = std::find(run.plan->inputs.begin(), run.plan->inputs.end(), index) - run.plan->inputs.begin();
  run.leases.clear(static_cast<std::size_t>(slot));
  return Status::Ok();
}
Status Invocation::bind(ValueId id, const ca::WeightBatch &batch, std::uint64_t weight_id) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index, input_index(run, id));
  if(run.pages && run.pages->page_input(index))
    return Status::InvalidArgument("native page input requires an exact prepared frame");
  if (!run.plan->effect_bindings.empty())
    return Status::Unimplemented("effect composition requires a checked host footprint for native weight alias admission");
  HYLE_ASSIGN_OR_RETURN(auto description, batch.description(weight_id));
  if (description != run.plan->values[index].carrier ||
      (run.plan->values[index].cpu_use && description != run.plan->values[index].description))
    return Status::InvalidArgument("CPU+Apple weight encoding/description differs");
  auto &value = run.values[index];
  value.weight = batch; value.weight_id = weight_id;
  value.host_input = {}; value.native_input.reset();
  run.prepared = false;
  const auto slot = std::find(run.plan->inputs.begin(), run.plan->inputs.end(), index) - run.plan->inputs.begin();
  run.leases.clear(static_cast<std::size_t>(slot));
  return Status::Ok();
}
Status Invocation::bind(ValueId id, compute::TensorDesc description, compute::ReadLease source) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index, input_index(run, id));
  if(run.pages && run.pages->page_input(index))
    return Status::InvalidArgument("native page input requires an exact prepared frame");
  HYLE_ASSIGN_OR_RETURN(auto checked, source.view());
  HYLE_RETURN_IF_ERROR(graph::detail::validate_effect_tensor_alias(
      run.plan->effect_bindings, checked));
  if (!run.plan->effect_bindings.empty() &&
      !checked.resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return Status::Unimplemented("effect composition requires checked host lease footprint");
  HYLE_RETURN_IF_ERROR(ensure_leases(run));
  HYLE_ASSIGN_OR_RETURN(auto storage, source.view());
  if (storage.resource() == run.plan->environment.native->ref()) {
    HYLE_ASSIGN_OR_RETURN(auto native, run.plan->environment.native->binding(source, description,
        {.allocator = run.allocator, .resource_budget = run.plan->budget}));
    HYLE_RETURN_IF_ERROR(bind(id, native));
  } else {
    HYLE_ASSIGN_OR_RETURN(auto host, source.tensor(description));
    HYLE_RETURN_IF_ERROR(bind(id, host));
  }
  const auto slot = std::find(run.plan->inputs.begin(), run.plan->inputs.end(), index) - run.plan->inputs.begin();
  run.leases.bind(static_cast<std::size_t>(slot), std::move(source));
  return Status::Ok();
}
Status Invocation::reserve_pending_sources(std::size_t capacity)noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));auto&run=*state_->run;
  if(!run.pages)return Status::InvalidArgument("graph has no native page nodes");
  return run.pages->reserve_pending(capacity,run.plan->budget);
}
Status Invocation::bind(ValueId id,compute::PagedArenaPendingRead frame)noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));auto&run=*state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index,input_index(run,id));
  if(!run.pages || !run.pages->pending)return Status::FailedPrecondition("reserve native Graph pending source capacity first");
  HYLE_RETURN_IF_ERROR(run.pages->validate_frame(index,frame));
  if(!frame.valid())return Status::InvalidArgument("pending Graph needs an actual page frame");
  if(!run.plan->effect_bindings.empty())return Status::Unimplemented("pending page capacity has no whole-arena effect footprint");
  HYLE_ASSIGN_OR_RETURN(auto native,ca::detail::PagedBatchAccess::binding(
      *run.plan->environment.native,frame,{.allocator=run.allocator,.resource_budget=run.plan->budget}));
  HYLE_RETURN_IF_ERROR(native_input(run.plan->values[index],native));
  HYLE_RETURN_IF_ERROR(ensure_leases(run));
  HYLE_RETURN_IF_ERROR(run.pages->commit_pending(index,std::move(frame)));
  const auto slot=std::find(run.plan->inputs.begin(),run.plan->inputs.end(),index)-run.plan->inputs.begin();
  run.leases.clear(std::size_t(slot));
  auto&value=run.values[index];value.native_input=std::move(native);value.host_input={};value.weight.reset();
  run.prepared=false;return Status::Ok();
}
Status Invocation::bind(ValueId id,compute::PagedArenaRead frame) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));auto& run=*state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index,input_index(run,id));
  if(!run.pages)return Status::InvalidArgument("graph has no native page inputs");
  if(run.pages->pending_input(index))return Status::FailedPrecondition("pending Graph input needs reset before ready binding");
  HYLE_RETURN_IF_ERROR(run.pages->validate_frame(index,frame));
  if(!run.plan->effect_bindings.empty())
    return Status::Unimplemented("page capacity has no whole-arena effect footprint");
  std::optional<mx::array> native;
  if(frame.valid()){
    HYLE_ASSIGN_OR_RETURN(native,ca::detail::PagedBatchAccess::binding(
        *run.plan->environment.native,frame,{.allocator=run.allocator,.resource_budget=run.plan->budget}));
    HYLE_RETURN_IF_ERROR(native_input(run.plan->values[index],*native));
  }
  HYLE_RETURN_IF_ERROR(ensure_leases(run));
  const auto slot=std::find(run.plan->inputs.begin(),run.plan->inputs.end(),index)-run.plan->inputs.begin();
  HYLE_RETURN_IF_ERROR(run.leases.bind_pages(std::size_t(slot),cd::PagedArenaAccess::pages(frame)));
  auto& value=run.values[index];value.native_input=std::move(native);value.host_input={};value.weight.reset();
  run.pages->commit_frame(index,std::move(frame));run.prepared=false;return Status::Ok();
}

Status Invocation::reset() noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  for (std::size_t id = 0; id < run.values.size(); ++id) {
    if (run.plan->values[id].id.kind() != ValueKind::kInput) continue;
    auto &value = run.values[id];
    value.host_input = {}; value.host_value = {};
    value.native_input.reset(); value.native_value.reset(); value.weight.reset();
  }
  run.prepared = false;
  run.leases.reset();
  if(run.pages)run.pages->reset();
  return Status::Ok();
}
Status Invocation::prepare_batch(std::span<const batch::NodeLayout> layouts) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));auto& run=*state_->run;
  if(!run.pages)return Status::InvalidArgument("graph has no native page nodes");
  run.prepared=false;run.pages->prepared=false;
  HYLE_RETURN_IF_ERROR(run.pages->validate_layouts(layouts));
  HYLE_RETURN_IF_ERROR(reset());
  for(const auto& page:run.pages->slots)run.fragments[page->fragment].lease={};
  HYLE_RETURN_IF_ERROR(run.pages->prepare(layouts));
  return Status::Ok();
}

Status Invocation::prepare_replay(ReplayOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  run.prepared = false;
  ReplayReport report;
  if(run.pages){
    if(!run.pages->prepared)return Status::FailedPrecondition("joint page nodes need explicit prepare_batch");
    for(const auto& page:run.pages->slots){
      const auto& cost=page->controller.report;
      ++report.paged_fragments;
      if(cost.uploaded_bytes>SIZE_MAX-report.paged_metadata_uploaded_bytes ||
         cost.backing_bytes>SIZE_MAX-report.paged_metadata_backing_bytes ||
         cost.upload_host_waits>SIZE_MAX-report.paged_metadata_host_waits)
        return Status::ResourceExhausted("native page cost report overflow");
      report.paged_metadata_uploaded_bytes+=cost.uploaded_bytes;
      report.paged_metadata_backing_bytes+=cost.backing_bytes;
      report.paged_metadata_host_waits+=cost.upload_host_waits;
    }
  }
  if (options.mode == ReplayMode::kRequireCapture)
    return Status::Unimplemented("CPU+Apple coarse composition uses managed native replay");
  for (const auto &fragment : run.plan->fragments) {
    if (fragment.target == kCpuTarget) ++report.cpu_fragments;
    else { ++report.apple_fragments; ++report.native_completion_wait_count; }
  }
  for (const auto &value : run.values) if (value.weight) ++report.weight_preparation_wait_count;
  if (options.host_wait == HostWaitPolicy::kReject &&
      (report.native_completion_wait_count != 0 || report.weight_preparation_wait_count != 0))
    return Status::Unimplemented("CPU+Apple composition requires checked host completion at native boundaries");
  for (std::size_t id = 0; id < run.values.size(); ++id) {
    const auto &spec = run.plan->values[id];
    auto &value = run.values[id];
    if (!spec.id.valid()) continue;
    HYLE_ASSIGN_OR_RETURN(auto bytes, spec.carrier.required_bytes());
    if (spec.id.kind() == ValueKind::kInput) {
      if (!spec.cpu_use && !spec.apple_use && !spec.published) continue;
      if (value.weight) {
        for (;;) {
          HYLE_ASSIGN_OR_RETURN(auto ready, value.weight->poll());
          if (ready == compute::CompletionPoll::kComplete) break;
          run.progress.wait_for(std::chrono::milliseconds(1));
        }
        HYLE_RETURN_IF_ERROR(value.weight->wait()); // Explicit cold barrier, reported above.
        HYLE_ASSIGN_OR_RETURN(auto array, value.weight->binding(value.weight_id));
        HYLE_RETURN_IF_ERROR(native_input(spec, array));
        value.native_input = std::move(array);
      }
      if(run.pages && run.pages->page_input(id)){
        HYLE_RETURN_IF_ERROR(run.pages->validate_frame(id,compute::PagedArenaRead{}).ok() || value.native_input
            ? Status::Ok():Status::FailedPrecondition("native page input frame is not bound"));
        value.native_value=value.native_input;continue;
      }
      if (!value.native_input && !value.host_input.valid())
        return Status::FailedPrecondition("CPU+Apple input is not bound");
      if (value.native_input && spec.cpu_use) {
        HYLE_RETURN_IF_ERROR(allocate_host(run, id));
        HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
        value.host_value = destination.as_const();
        HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
      } else if (value.host_input.valid()) {
        value.host_value = value.host_input;
        if (spec.apple_use) {
          HYLE_RETURN_IF_ERROR(allocate_boundary(run, id));
          HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
          if (value.repack.valid()) {
            HYLE_RETURN_IF_ERROR(allocate_host(run, id));
            HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
            HYLE_RETURN_IF_ERROR(value.repack.bind(spec.repack_input, value.host_input));
            HYLE_RETURN_IF_ERROR(value.repack.bind(spec.repack_output, destination));
            const std::array queues{QueueBinding{ExecutionLaneId{0}, run.cpu_queue}};
            HYLE_RETURN_IF_ERROR(value.repack.prepare_replay(queues));
            ++report.host_repack_count;
          }
        }
      }
    } else if (spec.fragment_output) {
      if (spec.producer == kCpuTarget && spec.apple_use)
        HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
      if (spec.producer == kAppleTarget && spec.cpu_use)
        HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
    }
  }
  for (std::size_t i = 0; i < run.fragments.size(); ++i) {
    const auto &fragment = run.plan->fragments[i];
    if (fragment.target != kCpuTarget) continue;
    auto &invocation = run.fragments[i].cpu;
    for (std::size_t port = 0; port < fragment.inputs.size(); ++port) {
      const auto id = fragment.inputs[port];
      auto input = run.values[id].host_value;
      if (run.plan->values[id].id.kind() != ValueKind::kInput) {
        HYLE_ASSIGN_OR_RETURN(auto tensor, host_tensor(run, id));
        input = tensor.as_const();
      }
      HYLE_RETURN_IF_ERROR(invocation.bind(fragment.cpu_inputs[port], input));
    }
    for (std::size_t port = 0; port < fragment.outputs.size(); ++port) {
      HYLE_ASSIGN_OR_RETURN(auto ownership, fragment.cpu.output_ownership(port));
      const auto id = fragment.outputs[port];
      if (ownership == OutputOwnership::kCaller) {
        HYLE_ASSIGN_OR_RETURN(auto tensor, host_tensor(run, id));
        HYLE_RETURN_IF_ERROR(invocation.bind(fragment.cpu_outputs[port], tensor));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto bytes, run.plan->values[id].carrier.required_bytes());
        HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
      }
    }
    const std::array queues{QueueBinding{ExecutionLaneId{0}, run.cpu_queue}};
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
  }
  run.report = report;
  run.prepared = true;
  return Status::Ok();
}
Result<ReplayReport> Invocation::replay_report() const noexcept {
  if (state_ == nullptr || state_->closed || !state_->run->prepared)
    return std::unexpected(Status::FailedPrecondition("CPU+Apple replay is not prepared"));
  return state_->run->report;
}
Result<ResourceReport> Invocation::resource_report() const noexcept {
  if (state_ == nullptr || state_->closed)
    return std::unexpected(Status::FailedPrecondition("empty or closed CPU+Apple invocation"));
  return cpu_apple::resource_report(*state_->run);
}
Result<compute::OwnedResult<Results>> Invocation::submit(ca::CopyBatchUpdate&&update)noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));auto&run=*state_->run;
  if(!run.prepared || !run.pages)return std::unexpected(Status::FailedPrecondition("pending Graph replay is not prepared"));
  HYLE_RETURN_IF_ERROR(run.pages->take_source(std::move(update)));
  auto result=submit();
  if(!result){run.pages->restore_source(update);return std::unexpected(result.error());}
  run.incref();
  run.pages->map_result(*result,&run,[](void*owner)noexcept{static_cast<RunState*>(owner)->decref();});
  return std::move(*result);
}
Result<ca::CopyBatchUpdate> Invocation::take_update()noexcept {
  if(!state_ || !state_->run->pages)return std::unexpected(Status::FailedPrecondition("native Graph has no pending source owner"));
  return state_->run->pages->take_update();
}
Result<compute::OwnedResult<Results>> Invocation::submit() noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_,true));
  auto &run = *state_->run;
  if(run.pages && run.pages->has_pending() && !run.pages->pending->source.update().valid())
    return std::unexpected(Status::FailedPrecondition("pending Graph requires submit with its original Update"));
  if (!run.prepared) return std::unexpected(Status::FailedPrecondition("CPU+Apple replay is not prepared"));
  if(run.pages){
    if(!run.pages->prepared)return std::unexpected(Status::FailedPrecondition("page episode already consumed"));
    run.pages->prepared=false;
  }
  if (run.leases.valid()) {
    HYLE_RETURN_IF_ERROR(run.leases.begin(&run,
        [](void *owner) noexcept { static_cast<RunState *>(owner)->incref(); },
        [](void *owner) noexcept { static_cast<RunState *>(owner)->decref(); },
        [](void *owner, Status terminal, bool accepted) noexcept {
          publish_terminal(*static_cast<RunState *>(owner), std::move(terminal), accepted);
        }));
  }
  const auto emplaced = state_->write.emplace(Results(&run));
  if (!emplaced.ok()) {
    if (run.leases.valid()) run.leases.finish(emplaced, false);
    state_->write = {};
    return std::unexpected(emplaced);
  }
  run.finished.reset(); run.cancel.store(false, std::memory_order_relaxed);
  {
    std::lock_guard lock(run.retirement_mutex);
    run.done = false; run.terminal = Status::Ok(); run.abandoned = {}; run.hint = Status::Ok();
  }
  run.running.store(true, std::memory_order_release);
  run.incref(); // public Completion
  auto completion = cd::CompletionAccess::make(&completion_vtable, &run);
  auto result = state_->write.publish(std::move(completion));
  if (!result) {
    if (run.leases.valid()) run.leases.finish(result.error(), false);
    else publish_terminal(run, result.error(), false);
    state_->write = {};
    return std::unexpected(result.error());
  }
  run.incref(); // Sched admission may start the driver immediately.
  const auto scheduled = run.scheduler->spawn_detached(&drive, &run, mem::kFiberStackLarge);
  if (!scheduled.ok()) {
    if (run.leases.valid()) run.leases.finish(scheduled, false);
    else publish_terminal(run, scheduled, false);
    run.decref();
    return std::unexpected(scheduled);
  }
  return result;
}
Result<compute::ResultLease<Results>> Invocation::run() noexcept {
  HYLE_ASSIGN_OR_RETURN(auto result, submit());
  return result.wait();
}
Status Invocation::close() noexcept {
  if (state_ == nullptr) return Status::FailedPrecondition("empty CPU+Apple invocation");
  state_->closed = true;
  auto &run = *state_->run;
  {
    std::lock_guard lock(run.pipeline_mutex);
    run.cancel.store(true, std::memory_order_release);
  }
  run.pipeline_ready.notify_all();
  while (run.running.load(std::memory_order_acquire))
    run.finished.wait_for(std::chrono::milliseconds(1));
  state_->write = {};
  if(run.pages)for(const auto& page:run.pages->slots){
    page->controller.close().IgnoreError();
    HYLE_RETURN_IF_ERROR(page->controller.drain());
  }
  std::lock_guard lock(run.retirement_mutex);
  return run.terminal;
}
} // namespace hyle::graph::cpu_apple
