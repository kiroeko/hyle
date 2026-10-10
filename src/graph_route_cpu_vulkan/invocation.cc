#include "plan.h"

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
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/util/ref_counted.h"
#include "../compute_route_vulkan/vulkan_composition_access.h"

namespace hyle::graph::cpu_vulkan {
namespace ca = compute::vulkan;
namespace cv = compute::vulkan;
namespace cd = compute::detail;
namespace detail {

struct ValueRun final {
  compute::ResourceBudget::Reservation host_charge;
  mem::Buffer host;
  compute::ConstTensorView host_input, host_value;
  std::optional<cv::Tensor> native_input, native_value, boundary;
  std::optional<ca::WeightBatch> weight;
  std::uint64_t weight_id = 0;
  graph::Invocation repack;
  ca::TransferSlot transfer;
};
struct FragmentRun final {
  graph::Invocation cpu;
  ca::Invocation vulkan;
  compute::ResultLease<ca::Results> lease;
  std::vector<cv::Tensor> arguments; // Vendor bind carrier, sized cold.
};
struct RunState final : util::RefCounted<RunState> {
  explicit RunState(std::shared_ptr<const Plan> source)
      : plan(std::move(source)), allocator(plan->allocator),
        values(mem::StdAllocator<ValueRun>(allocator)),
        fragments(mem::StdAllocator<FragmentRun>(allocator)) {}
  std::shared_ptr<const Plan> plan;
  mem::AllocatorRef allocator;
  Vector<ValueRun> values;
  Vector<FragmentRun> fragments;
  sched::Scheduler *scheduler = nullptr;
  compute::Queue *cpu_queue = nullptr;
  sched::Event progress, finished;
  std::mutex retirement_mutex;
  cd::CompletionLifetime abandoned{};
  Status hint, terminal;
  std::atomic<bool> running{false}, cancel{false};
  bool done = false, retired = false, prepared = false, native_sealed = false;
  std::size_t host_bytes = 0, boundary_bytes = 0, staging_bytes = 0;
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
Status admit(InvocationState *state) noexcept {
  if (state == nullptr || state->closed)
    return Status::FailedPrecondition("CPU+Vulkan invocation is empty or closed");
  if (state->run->running.load(std::memory_order_acquire))
    return Status::FailedPrecondition("CPU+Vulkan invocation is running");
  if (state->run->retired)
    return Status::FailedPrecondition("CPU+Vulkan failed invocation is retired");
  if (!state->write.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto write, state->result.begin());
    state->write = std::move(write);
  }
  return Status::Ok();
}
Result<std::size_t> input_index(const RunState &run, ValueId id) noexcept {
  if (!id.valid() || id.ordinal() >= run.plan->values.size() ||
      run.plan->values[id.ordinal()].id != id || id.kind() != ValueKind::kInput)
    return std::unexpected(Status::InvalidArgument("foreign or non-input CPU+Vulkan value"));
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
    return Status::ResourceExhausted("CPU+Vulkan host boundary exceeds workspace limit");
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
    return Status::ResourceExhausted("CPU+Vulkan native boundary exceeds limit");
  HYLE_ASSIGN_OR_RETURN(auto array, run.plan->environment.device->allocate(
      run.plan->values[id].carrier, {.resource_budget = run.plan->budget}));
  value.boundary = std::move(array);
  run.boundary_bytes += bytes;
  return Status::Ok();
}

Status native_input(const detail::Value &spec, const cv::Tensor &tensor) noexcept {
  if (!tensor.valid() || tensor.description() != spec.carrier)
    return Status::InvalidArgument("CPU+Vulkan native input encoding/description differs");
  if (spec.cpu_use && spec.description != spec.carrier)
    return Status::Unimplemented("strided CPU input requires its original host tensor binding");
  return Status::Ok();
}
Status allocate_transfer(RunState &run, std::size_t id) noexcept {
  if (run.values[id].transfer.valid()) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto bytes, run.plan->values[id].carrier.required_bytes());
  const auto capacity = std::max<std::size_t>(1, bytes);
  if (capacity > run.plan->environment.staging_budget_bytes - run.staging_bytes)
    return Status::ResourceExhausted("CPU+Vulkan staging capacity exceeds limit");
  HYLE_ASSIGN_OR_RETURN(run.values[id].transfer, ca::TransferSlot::create(
      *run.plan->environment.vulkan, *run.plan->environment.staging, capacity,
      {.resource_budget = run.plan->budget}));
  run.staging_bytes += capacity;
  return Status::Ok();
}
Status add_copy(ReplayReport &report, std::size_t bytes) noexcept {
  if (bytes > SIZE_MAX - report.host_copy_bytes)
    return Status::ResourceExhausted("CPU+Vulkan copy report overflows");
  ++report.host_copy_count;
  report.host_copy_bytes += bytes;
  return Status::Ok();
}

Status await_completion(RunState &run, const compute::Completion &completion) noexcept {
  for (;;) {
    const auto ready = completion.poll();
    if (!ready) return ready.error();
    if (*ready == compute::CompletionPoll::kComplete) return completion.wait();
    // The original Vulkan completion stays pending until native work and all
    // provider tails retire, including observation errors. This timed wait
    // yields the coarse progress fiber without occupying its Sched worker.
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
      if (run.running.load(std::memory_order_acquire)) return compute::CompletionPoll::kPending;
      std::lock_guard lock(run.retirement_mutex);
      if (!run.done) return compute::CompletionPoll::kPending;
      if (!run.terminal.ok()) return std::unexpected(run.terminal);
      return compute::CompletionPoll::kComplete;
    },
    .wait = [](void *opaque) noexcept {
      auto &run = *static_cast<RunState *>(opaque);
      while (run.running.load(std::memory_order_acquire))
        run.finished.wait_for(std::chrono::milliseconds(1));
      std::lock_guard lock(run.retirement_mutex);
      return run.terminal;
    },
    .request_cancel = [](void *opaque) noexcept {
      static_cast<RunState *>(opaque)->cancel.store(true, std::memory_order_release);
      return Status::Ok();
    }};

Status drive_graph(RunState &run, bool &accepted) noexcept {
  // Bind metadata/owners are fixed. No Provider selection, tracing, graph
  // traversal, capacity growth or CPU weight upload occurs in this loop.
  for (auto id : run.plan->inputs) {
    if (run.cancel.load(std::memory_order_acquire))
      return Status::Cancelled("CPU+Vulkan cancelled before boundary preparation");
    auto &value = run.values[id];
    const auto &spec = run.plan->values[id];
    if (value.native_input) {
      value.native_value = value.native_input;
      if (spec.cpu_use) {
        HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
        HYLE_ASSIGN_OR_RETURN(auto copy, value.transfer.download(*value.native_input));
        accepted = true;
        HYLE_RETURN_IF_ERROR(await_completion(run, copy));
        HYLE_RETURN_IF_ERROR(value.transfer.read(destination));
        value.host_value = destination.as_const();
      }
    } else {
      value.host_value = value.host_input;
      if (spec.vulkan_use) {
        auto source = value.host_input;
        if (value.repack.valid()) {
          HYLE_ASSIGN_OR_RETURN(auto completion, value.repack.submit(*run.cpu_queue));
          accepted = true;
          HYLE_RETURN_IF_ERROR(await_completion(run, completion));
          HYLE_ASSIGN_OR_RETURN(source, value.repack.output(spec.repack_output));
        }
        HYLE_ASSIGN_OR_RETURN(auto copy, value.transfer.upload(*value.boundary, source));
        accepted = true;
        HYLE_RETURN_IF_ERROR(await_completion(run, copy));
        // Fixed boundary native owner was committed by cold prepare.
      }
    }
  }
  for (std::size_t index = 0; index < run.fragments.size(); ++index) {
    if (run.cancel.load(std::memory_order_acquire))
      return Status::Cancelled("CPU+Vulkan cancelled before the next coarse fragment");
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
        if (run.plan->values[id].vulkan_use) {
          HYLE_ASSIGN_OR_RETURN(auto copy, value.transfer.upload(*value.boundary, value.host_value));
          HYLE_RETURN_IF_ERROR(await_completion(run, copy));
          // Fixed boundary native owner was committed by cold prepare.
        }
      }
    } else {
      slot.lease = {};
      HYLE_ASSIGN_OR_RETURN(auto submitted, slot.vulkan.submit());
      accepted = true;
      // Cancellation stops successors, while the original Vulkan owner drains
      // the already-accepted prefix before publishing its terminal result.
      HYLE_RETURN_IF_ERROR(await_completion(run, submitted.completion()));
      HYLE_ASSIGN_OR_RETURN(slot.lease, submitted.wait());
      const auto arrays = slot.lease->tensors();
      if (arrays.size() != fragment.outputs.size())
        return Status::Internal("Vulkan fragment result arity differs");
      for (std::size_t port = 0; port < arrays.size(); ++port) {
        const auto id = fragment.outputs[port];
        // The same original native owners were bound cold. Do not replace
        // their reference topology or construct an external ready import here.
        if (arrays[port].native_buffer() != run.values[id].native_value->native_buffer())
          return Status::Internal("Vulkan original output backing changed during replay");
        if (run.plan->values[id].cpu_use) {
          HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
          HYLE_ASSIGN_OR_RETURN(auto copy, run.values[id].transfer.download(arrays[port]));
          HYLE_RETURN_IF_ERROR(await_completion(run, copy));
          HYLE_RETURN_IF_ERROR(run.values[id].transfer.read(destination));
          run.values[id].host_value = destination.as_const();
        }
      }
    }
  }
  return run.cancel.load(std::memory_order_acquire)
      ? Status::Cancelled("CPU+Vulkan cancelled after native drain") : Status::Ok();
}
void drive(void *opaque) noexcept {
  auto &run = *static_cast<RunState *>(opaque);
  bool accepted = false;
  const auto status = drive_graph(run, accepted);
  publish_terminal(run, status, accepted);
  run.decref(); // existing Sched worker's admitted reference
}

Result<ResourceReport> resource_report(const RunState &run) noexcept {
  ResourceReport report;
  if (run.running.load(std::memory_order_acquire))
    return std::unexpected(Status::FailedPrecondition("CPU+Vulkan resource observation requires an idle slot"));
  report.has_shared_budget = run.plan->budget.valid();
  report.managed_host_workspace_bytes = run.host_bytes;
  report.managed_native_boundary_bytes = run.boundary_bytes;
  report.managed_staging_capacity_bytes = run.staging_bytes;
  report.has_unknown_native_storage = std::ranges::any_of(run.plan->fragments,
      [](const auto &fragment) { return fragment.target == kVulkanTarget; });
  if (run.plan->budget) {
    const std::array resources{run.plan->environment.host, run.plan->environment.device->ref(), run.plan->environment.staging->ref()};
    for (std::size_t i = 0; i < resources.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(report.storage[i], run.plan->budget.usage(resources[i]));
    }
    for (std::size_t i = 0; i < run.plan->budget.resource_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto resource, run.plan->budget.resource(i));
      if (resource == resources[0] || resource == resources[1] || resource == resources[2]) continue;
      HYLE_ASSIGN_OR_RETURN(auto usage, run.plan->budget.usage(resource));
      if (usage.live_bytes > SIZE_MAX - report.other_live_bytes)
        return std::unexpected(Status::ResourceExhausted("CPU+Vulkan resource sum overflows"));
      ++report.other_resource_count;
      report.other_live_bytes += usage.live_bytes;
    }
    report.unattributed = run.plan->budget.unattributed_usage();
  }
  try {
    detail::Vector<ca::NativeStorage> backings{mem::StdAllocator<ca::NativeStorage>(run.allocator)};
    const auto observe = [&](const std::optional<cv::Tensor> &array) -> Status {
      if (!array) return Status::Ok();
      HYLE_ASSIGN_OR_RETURN(auto storage, run.plan->environment.device->storage_info(*array));
      if (storage.buffer != VK_NULL_HANDLE) backings.push_back(storage);
      return Status::Ok();
    };
    for (std::size_t id = 0; id < run.values.size(); ++id) {
      HYLE_RETURN_IF_ERROR(observe(run.values[id].native_input));
      HYLE_RETURN_IF_ERROR(observe(run.values[id].native_value));
      HYLE_RETURN_IF_ERROR(observe(run.values[id].boundary));
      HYLE_RETURN_IF_ERROR(observe(run.plan->values[id].native_constant));
    }
    std::sort(backings.begin(), backings.end(), [](const auto &a, const auto &b) {
      return std::less<VkBuffer>{}(a.buffer, b.buffer);
    });
    VkBuffer previous = VK_NULL_HANDLE;
    for (const auto &backing : backings) {
      if (backing.buffer == previous) continue;
      previous = backing.buffer;
      const auto observed = backing.allocation_bytes.value_or(static_cast<std::size_t>(backing.capacity_bytes));
      if (observed > SIZE_MAX - report.observed_native_backing_bytes)
        return std::unexpected(Status::ResourceExhausted("CPU+Vulkan backing observation overflows"));
      ++report.observed_native_backing_count;
      report.observed_native_backing_bytes += observed;
    }
    return report;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("CPU+Vulkan resource report metadata"));
  }
}
} // namespace

Result<Invocation> Executable::create_invocation(sched::Scheduler &scheduler, compute::Queue &cpu_queue) const noexcept {
  if (!state_ || !scheduler.valid() || !state_->cpu_context.owns(cpu_queue))
    return std::unexpected(Status::InvalidArgument("CPU+Vulkan requires its exact CPU Queue and a live Scheduler"));
  const auto allocator = state_->allocator;
  auto *memory = allocator.allocate(sizeof(InvocationState), alignof(InvocationState));
  if (memory == nullptr) return std::unexpected(Status::ResourceExhausted("CPU+Vulkan invocation metadata"));
  auto *state = std::construct_at(static_cast<InvocationState *>(memory));
  state->allocator = allocator;
  const auto cleanup = [](InvocationState *pointer) noexcept { release_invocation(pointer); };
  std::unique_ptr<InvocationState, decltype(cleanup)> owner(state, cleanup);
  memory = allocator.allocate(sizeof(RunState), alignof(RunState));
  if (memory == nullptr) return std::unexpected(Status::ResourceExhausted("CPU+Vulkan run metadata"));
  state->run = std::construct_at(static_cast<RunState *>(memory), state_);
  auto &run = *state->run;
  run.scheduler = &scheduler;
  run.cpu_queue = &cpu_queue;
  run.host_bytes = state_->host_workspace_bytes;
  HYLE_ASSIGN_OR_RETURN(state->result, cd::ResultSlot<Results>::create(allocator));
  try {
    run.values.resize(state_->values.size());
    run.fragments.resize(state_->fragments.size());
    for (std::size_t id = 0; id < run.values.size(); ++id) {
      const auto &spec = state_->values[id];
      if (!spec.id.valid()) continue;
      if (spec.fragment_output && (spec.producer == kCpuTarget || spec.cpu_use))
        HYLE_RETURN_IF_ERROR(allocate_host(run, id));
      if (spec.producer == kCpuTarget && spec.vulkan_use && spec.fragment_output)
        HYLE_RETURN_IF_ERROR(allocate_boundary(run, id));
      if (spec.fragment_output && ((spec.producer == kCpuTarget && spec.vulkan_use) ||
                                  (spec.producer == kVulkanTarget && spec.cpu_use))) {
        HYLE_RETURN_IF_ERROR(allocate_transfer(run, id));
      }
      if (spec.repack.valid()) {
        HYLE_ASSIGN_OR_RETURN(run.values[id].repack, spec.repack.create_invocation());
      }
      if (spec.id.kind() == ValueKind::kConstant) run.values[id].host_value = spec.constant;
    }
    for (std::size_t i = 0; i < run.fragments.size(); ++i) {
      const auto &fragment = state_->fragments[i];
      if (fragment.target == kCpuTarget) {
        HYLE_ASSIGN_OR_RETURN(run.fragments[i].cpu, fragment.cpu.create_invocation());
      } else {
        HYLE_ASSIGN_OR_RETURN(run.fragments[i].vulkan, fragment.vulkan.create_invocation(allocator,
            {.resource_budget = state_->budget, .device = state_->environment.device, .staging = state_->environment.staging}));
        run.fragments[i].arguments.reserve(fragment.inputs.size());
        for (std::size_t port = 0; port < fragment.outputs.size(); ++port) {
          HYLE_ASSIGN_OR_RETURN(auto output, ca::detail::CompositionAccess::output(run.fragments[i].vulkan, port));
          run.values[fragment.outputs[port]].native_value = std::move(output);
        }
      }
    }
    owner.release();
    return Invocation(state);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("CPU+Vulkan slot metadata allocation"));
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
    return std::unexpected(Status::InvalidArgument("foreign or unpublished CPU+Vulkan output"));
  const auto tensor = state_->values[id.ordinal()].host_value;
  if (!tensor.valid()) return std::unexpected(Status::FailedPrecondition("CPU+Vulkan output is native; use tensor()"));
  return tensor;
}
Result<const cv::Tensor &> Results::tensor(ValueId id) const noexcept {
  if (state_ == nullptr || !id.valid() || id.ordinal() >= state_->values.size() ||
      state_->plan->values[id.ordinal()].id != id || !state_->plan->values[id.ordinal()].published)
    return std::unexpected(Status::InvalidArgument("foreign or unpublished CPU+Vulkan array output"));
  const auto &value = state_->values[id.ordinal()].native_value;
  if (!value) return std::unexpected(Status::FailedPrecondition("CPU+Vulkan output is host; use output()"));
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
  if (!tensor.valid() || tensor.description() != run.plan->values[index].description ||
      !tensor.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return Status::InvalidArgument("CPU+Vulkan host input description/storage differs");
  auto &value = run.values[index];
  value.host_input = tensor;
  value.native_input.reset(); value.weight.reset();
  run.prepared = false;
  return Status::Ok();
}
Status Invocation::bind(ValueId id, const cv::Tensor &array) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index, input_index(run, id));
  HYLE_RETURN_IF_ERROR(native_input(run.plan->values[index], array));
  HYLE_ASSIGN_OR_RETURN(auto storage, run.plan->environment.device->storage_info(array));
  (void)storage;
  auto &value = run.values[index];
  value.native_input = array;
  value.host_input = {}; value.weight.reset();
  run.prepared = false;
  return Status::Ok();
}
Status Invocation::bind(ValueId id, const ca::WeightBatch &batch, std::uint64_t weight_id) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  HYLE_ASSIGN_OR_RETURN(auto index, input_index(run, id));
  HYLE_ASSIGN_OR_RETURN(auto description, batch.description(weight_id));
  if (description != run.plan->values[index].carrier ||
      (run.plan->values[index].cpu_use && description != run.plan->values[index].description))
    return Status::InvalidArgument("CPU+Vulkan weight encoding/description differs");
  auto &value = run.values[index];
  value.weight = batch; value.weight_id = weight_id;
  value.host_input = {}; value.native_input.reset();
  run.prepared = false;
  return Status::Ok();
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
  return Status::Ok();
}
Status Invocation::prepare_replay(ReplayOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  run.prepared = false;
  ReplayReport report;
  if (options.mode == ReplayMode::kRequireCapture)
    return Status::Unimplemented("CPU+Vulkan coarse composition uses managed native replay");
  for (const auto &fragment : run.plan->fragments) {
    if (fragment.target == kCpuTarget) ++report.cpu_fragments;
    else { ++report.vulkan_fragments; ++report.native_completion_wait_count; }
  }
  for (const auto &value : run.values) if (value.weight) ++report.weight_preparation_wait_count;
  if (options.host_wait == HostWaitPolicy::kReject &&
      (report.native_completion_wait_count != 0 || report.weight_preparation_wait_count != 0))
    return Status::Unimplemented("CPU+Vulkan composition requires checked host completion at native boundaries");
  for (std::size_t id = 0; id < run.values.size(); ++id) {
    const auto &spec = run.plan->values[id];
    auto &value = run.values[id];
    if (!spec.id.valid()) continue;
    HYLE_ASSIGN_OR_RETURN(auto bytes, spec.carrier.required_bytes());
    if (spec.id.kind() == ValueKind::kInput) {
      if (!spec.cpu_use && !spec.vulkan_use && !spec.published) continue;
      if (value.weight) {
        for (;;) {
          HYLE_ASSIGN_OR_RETURN(auto ready, value.weight->poll());
          if (ready == compute::CompletionPoll::kComplete) break;
          run.progress.wait_for(std::chrono::milliseconds(1));
        }
        HYLE_RETURN_IF_ERROR(value.weight->wait()); // Explicit cold barrier, reported above.
        HYLE_ASSIGN_OR_RETURN(auto array, value.weight->binding(value.weight_id));
        HYLE_RETURN_IF_ERROR(native_input(spec, array));
        if (const auto storage = run.plan->environment.device->storage_info(array); !storage)
          return storage.error();
        value.native_input = std::move(array);
      }
      if (!value.native_input && !value.host_input.valid())
        return Status::FailedPrecondition("CPU+Vulkan input is not bound");
      if (value.native_input) value.native_value = value.native_input;
      if (value.native_input && spec.cpu_use) {
        HYLE_RETURN_IF_ERROR(allocate_transfer(run, id));
        HYLE_RETURN_IF_ERROR(allocate_host(run, id));
        HYLE_ASSIGN_OR_RETURN(auto destination, host_tensor(run, id));
        value.host_value = destination.as_const();
        HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
      } else if (value.host_input.valid()) {
        value.host_value = value.host_input;
        if (spec.vulkan_use) {
          HYLE_RETURN_IF_ERROR(allocate_boundary(run, id));
          HYLE_RETURN_IF_ERROR(allocate_transfer(run, id));
          value.native_value = value.boundary;
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
      if (spec.producer == kCpuTarget && spec.vulkan_use) {
        value.native_value = value.boundary;
        HYLE_RETURN_IF_ERROR(add_copy(report, bytes));
      }
      if (spec.producer == kVulkanTarget && spec.cpu_use)
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
  for (std::size_t i = 0; i < run.fragments.size(); ++i) {
    const auto &fragment = run.plan->fragments[i];
    if (fragment.target != kVulkanTarget) continue;
    auto &slot = run.fragments[i];
    slot.lease = {};
    slot.arguments.clear();
    for (auto id : fragment.inputs) {
      if (!run.values[id].native_value)
        return Status::FailedPrecondition("Vulkan cold input owner is not bound");
      slot.arguments.push_back(*run.values[id].native_value);
    }
    HYLE_RETURN_IF_ERROR(slot.vulkan.bind(slot.arguments));
  }
  if (!run.native_sealed) {
    for (auto &fragment : run.fragments)
      if (fragment.vulkan.valid()) HYLE_RETURN_IF_ERROR(ca::detail::CompositionAccess::seal(fragment.vulkan));
    run.native_sealed = true;
  }
  run.report = report;
  run.prepared = true;
  return Status::Ok();
}
Result<ReplayReport> Invocation::replay_report() const noexcept {
  if (state_ == nullptr || state_->closed || !state_->run->prepared)
    return std::unexpected(Status::FailedPrecondition("CPU+Vulkan replay is not prepared"));
  return state_->run->report;
}
Result<ResourceReport> Invocation::resource_report() const noexcept {
  if (state_ == nullptr || state_->closed)
    return std::unexpected(Status::FailedPrecondition("empty or closed CPU+Vulkan invocation"));
  return cpu_vulkan::resource_report(*state_->run);
}
Result<compute::OwnedResult<Results>> Invocation::submit() noexcept {
  HYLE_RETURN_IF_ERROR(admit(state_));
  auto &run = *state_->run;
  if (!run.prepared) return std::unexpected(Status::FailedPrecondition("CPU+Vulkan replay is not prepared"));
  HYLE_RETURN_IF_ERROR(state_->write.emplace(Results(&run)));
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
    publish_terminal(run, result.error(), false);
    state_->write = {};
    return std::unexpected(result.error());
  }
  run.incref(); // Sched admission may start the driver immediately.
  const auto scheduled = run.scheduler->spawn_detached(&drive, &run, mem::kFiberStackLarge);
  if (!scheduled.ok()) {
    publish_terminal(run, scheduled, false);
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
  if (state_ == nullptr) return Status::FailedPrecondition("empty CPU+Vulkan invocation");
  state_->closed = true;
  auto &run = *state_->run;
  run.cancel.store(true, std::memory_order_release);
  while (run.running.load(std::memory_order_acquire))
    run.finished.wait_for(std::chrono::milliseconds(1));
  state_->write = {};
  std::lock_guard lock(run.retirement_mutex);
  return run.terminal;
}
} // namespace hyle::graph::cpu_vulkan
