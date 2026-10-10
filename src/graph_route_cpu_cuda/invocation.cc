#include "hyle/graph/cpu_cuda.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <functional>
#include <mutex>
#include <utility>

#include "../compute/completion_internal.h"
#include "../graph/plan_state.h"
#include "../graph/submission_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/container/small_vector.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/ref_counted.h"
#include "progress.h"
#include "../compute_route_cuda/weights_internal.h"

namespace hyle::graph::cpu_cuda {
namespace detail {

using WeightState = compute::cuda::detail::WeightState;

struct WeightUse final {
  ValueId value{};
  WeightState *state = nullptr;
  WeightUse() noexcept = default;
  WeightUse(const WeightUse &) = delete;
  WeightUse &operator=(const WeightUse &) = delete;
  WeightUse(WeightUse &&other) noexcept
      : value(other.value), state(std::exchange(other.state, nullptr)) {}
  WeightUse &operator=(WeightUse &&other) noexcept {
    if (this != &other) {
      compute::cuda::detail::release_weight(state);
      value = other.value;
      state = std::exchange(other.state, nullptr);
    }
    return *this;
  }
  ~WeightUse() { compute::cuda::detail::release_weight(state); }
  void reset(ValueId next_value = {}, WeightState *next_state = nullptr) noexcept {
    compute::cuda::detail::retain_weight(next_state);
    compute::cuda::detail::release_weight(state);
    value = next_value;
    state = next_state;
  }
};

// Separate backing from the result slot: a result retains this backing, never
// the owner that contains its own ResultSlot. This avoids a reference cycle.
struct RunState final : util::RefCounted<RunState> {
  explicit RunState(mem::AllocatorRef owner) noexcept
      : allocator(owner), weights(owner), weight_prefixes(owner) {}
  mem::AllocatorRef allocator{};
  // Bound batches also cover native Invocation destruction: provider state
  // may borrow its tensor backing until its destructor finishes.
  container::SmallVector<WeightUse, 2, mem::AllocatorRef> weights;
  container::SmallVector<compute::Completion, 2, mem::AllocatorRef> weight_prefixes;
  graph::Invocation invocation{};
  sched::Scheduler *scheduler = nullptr;
  std::array<QueueBinding, 2> queues{};
  std::size_t queue_count = 0;
  compute::Queue *cuda_queue = nullptr;
  graph::detail::SubmissionCursor cursor{};
  sched::Event cuda_ready{};
  sched::Event finished{};
  std::mutex retirement_mutex{};
  compute::detail::CompletionLifetime abandoned{};
  Status abandoned_hint{};
  Status terminal{};
  bool done = false; // protected by retirement_mutex
  std::atomic<bool> running{false};
  std::atomic<bool> cancel_requested{false};
  ReplayReport replay_report{};
  bool prepared = false;

  std::span<const QueueBinding> bindings() const noexcept {
    return {queues.data(), queue_count};
  }
  void destroy() noexcept {
    const auto owner = allocator;
    std::destroy_at(this);
    owner.deallocate(this, sizeof(RunState), alignof(RunState));
  }
};

struct InvocationState final {
  mem::AllocatorRef allocator{};
  RunState *run = nullptr;
  compute::detail::ResultSlot<Results> result{};
  compute::detail::ResultWrite<Results> write{};
  bool closed = false;

  ~InvocationState() {
    if (run != nullptr)
      run->decref();
  }
};

} // namespace detail
namespace {

using detail::InvocationState;
using detail::RunState;

void release_invocation(InvocationState *state) noexcept {
  if (state == nullptr)
    return;
  const auto owner = state->allocator;
  std::destroy_at(state);
  owner.deallocate(state, sizeof(InvocationState), alignof(InvocationState));
}

Status admit_write(InvocationState *state) noexcept {
  if (state == nullptr || state->closed)
    return Status::FailedPrecondition("CPU+CUDA invocation is invalid or closed");
  if (state->run->running.load(std::memory_order_acquire))
    return Status::FailedPrecondition("CPU+CUDA slot has work in flight");
  if (!state->write.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto write, state->result.begin());
    state->write = std::move(write);
  }
  return Status::Ok();
}

Status terminal_status(RunState &run) noexcept {
  std::lock_guard lock(run.retirement_mutex);
  return run.terminal;
}

Result<ResourceReport> report_resources(const RunState &run) noexcept {
  const auto *native = graph::detail::InvocationAccess::state(run.invocation);
  const auto &plan = *native->plan;
  if (!plan.invocation_storage || !plan.invocation_storage->resource_budget())
    return std::unexpected(Status::FailedPrecondition(
        "CPU+CUDA graph has no shared resource budget"));
  const auto &budget = plan.invocation_storage->resource_budget();
  ResourceReport result;
  {
    // Merge canonical allocation ranges once. Cold scratch dies before the
    // live-usage snapshot; it cannot inflate reported resident metadata.
    container::SmallVector<mem::ConstBufferView, 16, mem::AllocatorRef> ranges(run.allocator);
    for (std::size_t i = 0; i < plan.instance_roles.size(); ++i) {
      if ((plan.instance_roles[i] & (graph::detail::kExternalInput |
                                    graph::detail::kExternalOutput |
                                    graph::detail::kConstant)) == 0)
        continue;
      bool owned = false;
      for (std::size_t j = 0; j < plan.constant_ids.size(); ++j)
        if (plan.instance_values[i] == plan.constant_ids[j].ordinal() &&
            plan.owned_constant(j) != nullptr)
          owned = true;
      if (owned || !native->readable[i].valid()) continue;
      if (!ranges.try_push_back(native->readable[i].storage()))
        return std::unexpected(Status::ResourceExhausted("resource report scratch"));
    }
    const std::less<const void *> less;
    std::sort(ranges.begin(), ranges.end(), [&](auto a, auto b) noexcept {
      if (a.resource() != b.resource())
        return less(&a.resource().resource(), &b.resource().resource());
      if (a.native_handle() != b.native_handle())
        return less(a.native_handle(), b.native_handle());
      return a.offset() < b.offset();
    });
    for (std::size_t i = 0; i < ranges.size();) {
      auto first = ranges[i++];
      std::size_t covered = first.size(), end = first.offset() + first.size();
      while (i < ranges.size() && ranges[i].resource() == first.resource() &&
             ranges[i].native_handle() == first.native_handle()) {
        const auto next = ranges[i++];
        const auto next_end = next.offset() + next.size();
        if (next_end > end) {
          const auto extra = next_end - (std::max)(end, next.offset());
          if (extra > SIZE_MAX - covered)
            return std::unexpected(Status::ResourceExhausted("boundary range sum overflows"));
          covered += extra;
          end = next_end;
        }
      }
      for (std::size_t storage = 0; storage < result.storage.size(); ++storage) {
        if (plan.storage_domains[storage].resource != first.resource()) continue;
        if (covered > SIZE_MAX - result.boundary_view_bytes[storage])
          return std::unexpected(Status::ResourceExhausted("boundary report sum overflows"));
        result.boundary_view_bytes[storage] += covered;
      }
    }
  }
  for (std::size_t i = 0; i < result.storage.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(result.storage[i], budget.usage(plan.storage_domains[i].resource));
  }
  for (std::size_t i = 0; i < budget.resource_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto resource, budget.resource(i));
    bool included = false;
    for (std::size_t j = 0; j < result.storage.size(); ++j)
      included |= resource == plan.storage_domains[j].resource;
    if (included) continue;
    HYLE_ASSIGN_OR_RETURN(auto usage, budget.usage(resource));
    if (usage.live_bytes > SIZE_MAX - result.other_live_bytes)
      return std::unexpected(Status::ResourceExhausted("resource report sum overflows"));
    ++result.other_resource_count;
    result.other_live_bytes += usage.live_bytes;
  }
  result.unattributed = budget.unattributed_usage();
  result.has_unknown_graph_storage_history = plan.invocation_storage->usage().has_unknown_storage ||
      (plan.compilation_storage && plan.compilation_storage->usage().has_unknown_storage);
  return result;
}

void destroy_completion(void *opaque,
                        compute::detail::CompletionLifetime lifetime,
                        Status hint) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  Status terminal;
  bool done;
  {
    std::lock_guard lock(run.retirement_mutex);
    done = run.done;
    if (done) {
      terminal = run.terminal;
    } else {
      // Ownership is handed to the existing worker, which already holds its
      // own backing reference. Public destruction never waits for that worker.
      run.abandoned = lifetime;
      run.abandoned_hint = std::move(hint);
    }
  }
  if (done && lifetime.valid()) {
    if (terminal.ok() && !hint.ok())
      terminal = std::move(hint);
    lifetime.terminal(lifetime.state, terminal);
    lifetime.release(lifetime.state);
  }
  run.decref(); // this public Completion's reference
}

Result<compute::CompletionPoll> poll_completion(void *opaque) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  if (run.running.load(std::memory_order_acquire))
    return compute::CompletionPoll::kPending;
  std::lock_guard lock(run.retirement_mutex);
  if (!run.done)
    return compute::CompletionPoll::kPending;
  if (!run.terminal.ok())
    return std::unexpected(run.terminal);
  return compute::CompletionPoll::kComplete;
}

Status wait_completion(void *opaque) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  while (run.running.load(std::memory_order_acquire))
    run.finished.wait_for(std::chrono::milliseconds(1));
  return terminal_status(run);
}

Status cancel_completion(void *opaque) noexcept {
  static_cast<RunState *>(opaque)->cancel_requested.store(
      true, std::memory_order_release);
  return Status::Ok();
}

const compute::detail::CompletionVTable kCompletion{
    &destroy_completion, &poll_completion, &wait_completion, &cancel_completion};

void publish_terminal(RunState &run, Status terminal) noexcept {
  compute::detail::CompletionLifetime abandoned;
  Status hint;
  {
    std::lock_guard lock(run.retirement_mutex);
    run.terminal = std::move(terminal);
    run.done = true;
    abandoned = std::exchange(run.abandoned, {});
    hint = std::move(run.abandoned_hint);
  }
  // The worker reference survives signal's notification tail and all lifetime
  // callbacks. A settled public result may not yet reuse this running slot.
  run.finished.signal();
  if (abandoned.valid()) {
    auto result = terminal_status(run);
    if (result.ok() && !hint.ok())
      result = std::move(hint);
    abandoned.terminal(abandoned.state, result);
    abandoned.release(abandoned.state);
  }
  run.running.store(false, std::memory_order_release);
}

void drive(void *opaque) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  // Fixed selections keep both device backing and upload prefix alive. Queue
  // waits precede every shared CUDA segment on this exact native endpoint.
  for (std::size_t i = 0; i < run.weights.size(); ++i) {
    auto *weight = run.weights[i].state;
    if (weight == nullptr)
      continue;
    if (run.cancel_requested.load(std::memory_order_acquire)) {
      run.cursor.request_cancel();
      break;
    }
    if (run.cuda_queue != nullptr) {
      auto ordered = compute::cuda::detail::order_weight(weight, *run.cuda_queue);
      if (!ordered) {
        run.cursor.reject_successors(std::move(ordered).error());
        break;
      }
      run.weight_prefixes[i] = std::move(*ordered);
    } else {
      const auto ready = compute::cuda::detail::wait_weight(weight);
      if (!ready.ok()) {
        run.cursor.reject_successors(ready);
        break;
      }
    }
  }
  while (!run.cursor.done()) {
    if (run.cancel_requested.load(std::memory_order_acquire)) {
      run.cursor.request_cancel();
      break;
    }
    auto advanced = run.cursor.advance(false);
    if (!advanced) {
      run.cursor.reject_successors(std::move(advanced).error());
      break;
    }
    if (*advanced)
      continue;
    auto *source_queue = run.cursor.pending_queue();
    Status dependency;
    if (source_queue != nullptr &&
        source_queue->device().backend_id() == compute::cuda::kBackendId) {
      dependency = detail::wait_cuda_prefix(*source_queue, run.cuda_ready);
    } else {
      // A delegated provider owns its episode, not a HYLE stream. Preserve its
      // declared host-wait cost instead of pretending it is a CUDA event.
      const auto *source = run.cursor.pending_dependency();
      dependency = source != nullptr
                       ? source->wait()
                       : Status::Internal("CPU+CUDA dependency disappeared");
    }
    if (!dependency.ok())
      run.cursor.reject_successors(std::move(dependency));
  }
  auto prefix = run.cursor.finish();
  Status terminal;
  if (!prefix) {
    terminal = std::move(prefix).error();
  } else {
    auto state = prefix->poll();
    if (state && *state == compute::CompletionPoll::kPending &&
        run.cuda_queue != nullptr) {
      terminal = detail::wait_cuda_prefix(*run.cuda_queue, run.cuda_ready);
    }
    // Retire the actual accepted prefix even if its notification failed. The
    // original Graph lifetime attachment makes outputs ready only on success.
    const auto native = prefix->wait();
    if (terminal.ok())
      terminal = native;
  }
  // An early graph rejection may still follow an accepted dependency prefix.
  // Retire it and observe late errors before publishing this run's terminal.
  for (auto &dependency : run.weight_prefixes) {
    if (dependency.valid()) {
      const auto status = dependency.wait();
      if (terminal.ok())
        terminal = status;
      dependency = {};
    }
  }
  run.cursor = {};
  publish_terminal(run, std::move(terminal));
  run.decref(); // existing Sched worker's reference; no access follows
}

} // namespace

Results::Results(detail::RunState *state) noexcept : state_(state) {
  state_->incref();
}
Results::Results(Results &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
Results &Results::operator=(Results &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr)
      state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
Results::~Results() {
  if (state_ != nullptr)
    state_->decref();
}
Result<compute::ConstTensorView> Results::output(ValueId value) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid CPU+CUDA result"));
  return state_->invocation.output(value);
}

Invocation::Invocation(Invocation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
Invocation &Invocation::operator=(Invocation &&other) noexcept {
  if (this != &other) {
    release_invocation(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
Invocation::~Invocation() { release_invocation(state_); }

Result<Invocation> Invocation::create(graph::Invocation &&invocation,
                                     sched::Scheduler &scheduler,
                                     std::span<const QueueBinding> queues,
                                     mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid() || !invocation.valid() || !scheduler.valid() ||
      invocation.in_flight())
    return std::unexpected(Status::InvalidArgument("invalid CPU+CUDA owners"));
  const auto *native = graph::detail::InvocationAccess::state(invocation);
  const auto &plan = *native->plan;
  if (plan.targets.size() != 2 || plan.lanes.size() != 2 ||
      plan.targets[0].context.device().backend_id() != compute::cuda::kBackendId ||
      plan.targets[1].context.device().backend_id() != compute::cpu::kBackendId ||
      plan.lanes[0].target != kCudaTarget || plan.lanes[1].target != kCpuTarget ||
      queues.size() > 2) {
    return std::unexpected(Status::InvalidArgument(
        "invocation was not compiled for CPU+CUDA composition"));
  }
  for (std::size_t i = 0; i < queues.size(); ++i) {
    const auto &binding = queues[i];
    if (!binding.lane.valid() || binding.lane.ordinal() >= 2 ||
        binding.queue == nullptr ||
        !plan.targets[binding.lane.ordinal()].context.owns(*binding.queue)) {
      return std::unexpected(Status::InvalidArgument(
          "CPU+CUDA queue has a different lane or Context"));
    }
    if (i != 0 && binding.lane == queues[0].lane)
      return std::unexpected(Status::InvalidArgument("duplicate CPU+CUDA lane"));
  }
  if (plan.invocation_storage && plan.invocation_storage->resource_budget()) {
    HYLE_ASSIGN_OR_RETURN(allocator,
        plan.invocation_storage->resource_budget().metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto result,
                       compute::detail::ResultSlot<Results>::create(allocator));
  auto *run_memory = allocator.allocate(sizeof(RunState), alignof(RunState));
  if (run_memory == nullptr)
    return std::unexpected(Status::ResourceExhausted("CPU+CUDA run allocation"));
  auto *state_memory =
      allocator.allocate(sizeof(InvocationState), alignof(InvocationState));
  if (state_memory == nullptr) {
    allocator.deallocate(run_memory, sizeof(RunState), alignof(RunState));
    return std::unexpected(Status::ResourceExhausted("CPU+CUDA slot allocation"));
  }
  auto *run = std::construct_at(static_cast<RunState *>(run_memory), allocator);
  if (!run->weights.try_resize(plan.input_ids.size()) ||
      !run->weight_prefixes.try_resize(plan.input_ids.size())) {
    run->decref();
    allocator.deallocate(state_memory, sizeof(InvocationState), alignof(InvocationState));
    return std::unexpected(Status::ResourceExhausted("CPU+CUDA weight bindings"));
  }
  run->invocation = std::move(invocation);
  run->scheduler = &scheduler;
  run->queue_count = queues.size();
  for (std::size_t i = 0; i < queues.size(); ++i) {
    run->queues[i] = queues[i];
    if (queues[i].lane.ordinal() == 0)
      run->cuda_queue = queues[i].queue;
  }
  auto *state = std::construct_at(static_cast<InvocationState *>(state_memory));
  state->allocator = allocator;
  state->run = run;
  state->result = std::move(result);
  return Invocation(state);
}

Status Invocation::bind(ValueId value, compute::ConstTensorView tensor) noexcept {
  HYLE_RETURN_IF_ERROR(admit_write(state_));
  HYLE_RETURN_IF_ERROR(state_->run->invocation.bind(value, tensor));
  state_->run->prepared = false;
  for (auto &weight : state_->run->weights)
    if (weight.value == value)
      weight.reset();
  return Status::Ok();
}
Status Invocation::bind(ValueId value, compute::TensorView tensor) noexcept {
  HYLE_RETURN_IF_ERROR(admit_write(state_));
  HYLE_RETURN_IF_ERROR(state_->run->invocation.bind(value, tensor));
  state_->run->prepared = false;
  for (auto &weight : state_->run->weights)
    if (weight.value == value)
      weight.reset();
  return Status::Ok();
}
Status Invocation::reset() noexcept {
  HYLE_RETURN_IF_ERROR(admit_write(state_));
  HYLE_RETURN_IF_ERROR(state_->run->invocation.reset());
  state_->run->prepared = false;
  for (auto &weight : state_->run->weights)
    weight.reset();
  return Status::Ok();
}
Status Invocation::bind(ValueId value, const compute::cuda::WeightBatch &batch,
                        std::uint64_t id) noexcept {
  HYLE_RETURN_IF_ERROR(admit_write(state_));
  HYLE_ASSIGN_OR_RETURN(auto tensor, batch.binding(id));
  auto &run = *state_->run;
  auto *slot = static_cast<detail::WeightUse *>(nullptr);
  for (auto &weight : run.weights) {
    if (weight.value == value) {
      slot = &weight;
      break;
    }
    if (slot == nullptr && weight.state == nullptr)
      slot = &weight;
  }
  if (slot == nullptr)
    return Status::ResourceExhausted("CPU+CUDA weight binding capacity");
  HYLE_RETURN_IF_ERROR(run.invocation.bind(value, tensor));
  run.prepared = false;
  slot->reset(value, compute::cuda::detail::WeightAccess::state(batch));
  return Status::Ok();
}
Status Invocation::prepare_replay(ReplayOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(admit_write(state_));
  auto &run = *state_->run;
  run.prepared = false;
  ReplayReport report;
  for (const auto &weight : run.weights) {
    if (weight.state == nullptr)
      continue;
    if (run.cuda_queue != nullptr)
      ++report.weight_device_wait_count;
    else
      ++report.weight_host_wait_count;
  }
  if (report.weight_host_wait_count != 0 &&
      options.host_wait == HostWaitPolicy::kReject)
    return Status::Unimplemented("CPU+CUDA weight binding requires a host wait");
  if (run.cuda_queue != nullptr) {
    const auto segments = run.invocation.executable().segment_count();
    if (segments > SIZE_MAX - 4 ||
        run.weights.size() > (SIZE_MAX - segments - 4) / 2)
      return Status::ResourceExhausted("CPU+CUDA completion capacity overflows");
    HYLE_RETURN_IF_ERROR(run.cuda_queue->reserve_completions(
        segments + run.weights.size() * 2 + 4));
  }
  HYLE_RETURN_IF_ERROR(run.invocation.prepare_replay(run.bindings(), options));
  HYLE_ASSIGN_OR_RETURN(report.graph, run.invocation.replay_report());
  const auto *native = graph::detail::InvocationAccess::state(run.invocation);
  for (const auto &step : native->plan->steps) {
    if (step.kind != graph::detail::PlanStepKind::kCopy ||
        (step.source_storage != kStagingStorage &&
         step.destination_storage != kStagingStorage))
      continue;
    if (step.copy_bytes > SIZE_MAX - report.staging_copy_bytes)
      return Status::ResourceExhausted("CPU+CUDA staging report overflows");
    ++report.staging_copy_count;
    report.staging_copy_bytes += step.copy_bytes;
  }
  run.replay_report = report;
  run.prepared = true;
  return Status::Ok();
}

Result<ReplayReport> Invocation::replay_report() const noexcept {
  if (state_ == nullptr || state_->closed || !state_->run->prepared)
    return std::unexpected(Status::FailedPrecondition(
        "CPU+CUDA replay is not prepared"));
  return state_->run->replay_report;
}

Result<ResourceReport> Invocation::resource_report() const noexcept {
  if (state_ == nullptr || state_->closed)
    return std::unexpected(Status::FailedPrecondition("invalid CPU+CUDA invocation"));
  return report_resources(*state_->run);
}

Result<ReplayDependencyRoute>
Invocation::dependency_route(std::size_t segment,
                             std::size_t dependency) const noexcept {
  if (state_ == nullptr || state_->closed || !state_->run->prepared)
    return std::unexpected(Status::FailedPrecondition(
        "CPU+CUDA replay is not prepared"));
  return state_->run->invocation.dependency_route(segment, dependency);
}

Result<compute::OwnedResult<Results>> Invocation::submit() noexcept {
  if (state_ == nullptr || !state_->run->prepared)
    return std::unexpected(Status::FailedPrecondition(
        "CPU+CUDA replay is not prepared"));
  HYLE_RETURN_IF_ERROR(admit_write(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(
      auto cursor,
      graph::detail::SubmissionCursor::begin(run.invocation, run.bindings()));
  HYLE_RETURN_IF_ERROR(state_->write.emplace(Results(&run)));
  run.cursor = std::move(cursor);
  run.finished.reset();
  run.cancel_requested.store(false, std::memory_order_relaxed);
  {
    std::lock_guard lock(run.retirement_mutex);
    run.done = false;
    run.terminal = Status::Ok();
    run.abandoned = {};
    run.abandoned_hint = Status::Ok();
  }
  run.running.store(true, std::memory_order_release);
  run.incref(); // the public Completion
  auto completion = compute::detail::CompletionAccess::make(&kCompletion, &run);
  auto result = state_->write.publish(std::move(completion));
  if (!result) {
    // No native work was accepted yet. Finish the admitted empty cursor and
    // retire this local Completion before rolling back the result writer.
    run.cursor.request_cancel();
    auto empty = run.cursor.finish();
    if (empty)
      empty->wait().IgnoreError();
    run.cursor = {};
    publish_terminal(run, std::move(result).error());
    state_->write = {};
    return std::unexpected(terminal_status(run));
  }
  run.incref(); // retained before the existing scheduler can start the callback
  // Mature CUDA host dispatch (notably cuBLASLt) uses deeper stacks than the
  // scheduler's 64 KiB lightweight-task default. Use its existing 256 KiB
  // recycled large-stack class for this adapter; ordinary Sched is unchanged.
  const auto scheduled =
      run.scheduler->spawn_detached(&drive, &run, mem::kFiberStackLarge);
  if (!scheduled.ok()) {
    run.cursor.reject_successors(scheduled);
    auto empty = run.cursor.finish();
    if (empty)
      empty->wait().IgnoreError();
    run.cursor = {};
    publish_terminal(run, scheduled);
    run.decref();
    return std::unexpected(scheduled);
  }
  return result;
}

Result<compute::ResultLease<Results>> Invocation::run() noexcept {
  HYLE_ASSIGN_OR_RETURN(auto submitted, submit());
  return submitted.wait();
}

Status Invocation::close() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("invalid CPU+CUDA invocation");
  state_->closed = true;
  auto &run = *state_->run;
  run.cancel_requested.store(true, std::memory_order_release);
  while (run.running.load(std::memory_order_acquire))
    run.finished.wait_for(std::chrono::milliseconds(1));
  state_->write = {};
  return terminal_status(run);
}

} // namespace hyle::graph::cpu_cuda
