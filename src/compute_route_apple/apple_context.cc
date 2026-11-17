#include "hyle/compute/apple.h"

#include "hyle/obs/trace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

#include <mlx/compile.h>
#include <mlx/backend/metal/device.h>
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/transforms.h>

#include "apple_checks.h"
#include "apple_carrier.h"
#include "apple_copy_internal.h"
#include "apple_dependencies.h"
#include "apple_completion_internal.h"
#include "apple_internal.h"
#include "apple_runtime_internal.h"
#include "apple_recurrent.h"
#include "apple_native_event.h"
#include "apple_tensor.h"
#include "completion_internal.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/container/flat_hash_set.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace cd = compute::detail;

namespace detail {
Environment read_environment();

struct RegionFailure {
  Status status;
};

struct DeviceCheck {
  std::uintptr_t result;
  mx::array status;
};

// HYLE only builds this list during native tracing. Each Context has its own
// native owner thread; RAII also restores it if a region throws while tracing.
thread_local std::vector<DeviceCheck> *active_checks = nullptr;
struct CheckTrace {
  std::vector<DeviceCheck> checks;
  std::vector<DeviceCheck> *previous = active_checks;
  CheckTrace() { active_checks = &checks; }
  ~CheckTrace() { active_checks = previous; }
  CheckTrace(const CheckTrace &) = delete;
  CheckTrace &operator=(const CheckTrace &) = delete;
};

Result<mx::array> with_device_status(mx::array output, mx::array status,
                                     mx::Stream stream) {
  if (active_checks == nullptr)
    return std::unexpected(Status::FailedPrecondition(
        "Apple checked lowering requires managed Context tracing"));
  // Public depends inherits the output primitive's stream. Do not fall back
  // to MLX's default stream, or instantiate its non-exported Depends class.
  if (!output.has_primitive() || output.primitive().stream() != stream)
    return std::unexpected(Status::InvalidArgument(
        "Apple checked output requires the managed stream"));
  auto result = mx::depends({output}, {status}).front();
  active_checks->push_back({result.id(), std::move(status)});
  return result;
}

Status algorithm_status(std::int32_t code) {
  switch (code) {
  case 0:
    return Status::Ok();
  case kIndexOutOfRange:
    return Status::OutOfRange("Apple index is outside the selected axis");
  case kScanOverflow:
    return Status::OutOfRange("Apple integer Scan prefix exceeds I32");
  case kInvalidNumericInput:
    return Status::InvalidArgument("Apple numeric input is outside the operation domain");
  default:
    return Status::Internal("Apple region returned an unknown device status");
  }
}

Status native_exception() noexcept {
  try {
    throw;
  } catch (const RegionFailure &failure) {
    return failure.status;
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("MLX/native metadata allocation failed");
  } catch (const std::invalid_argument &error) {
    return Status::InvalidArgument(error.what());
  } catch (const std::exception &error) {
    return Status::Internal(error.what());
  } catch (...) {
    return Status::Internal("MLX native execution failed");
  }
}
} // namespace detail

namespace {
using detail::native_exception;

// Cold only. Native primitives remain in one MLX GPU stream; the complete
// compiled region is one hot invocation. Reuse the existing fallible hash set.
bool validate_region(std::vector<mx::array> &outputs, mx::Stream stream,
                     const detail::CheckTrace &trace) {
  if (outputs.empty())
    throw std::invalid_argument("Apple region must return at least one array");
  container::FlatHashSet<std::uintptr_t> visited;
  std::vector<mx::array> pending(outputs);
  while (!pending.empty()) {
    auto value = std::move(pending.back());
    pending.pop_back();
    const auto inserted = visited.try_insert(value.id());
    if (inserted == decltype(visited)::InsertResult::kOutOfMemory)
      throw std::bad_alloc();
    if (inserted == decltype(visited)::InsertResult::kAlreadyPresent)
      continue;
    if (value.has_primitive()) {
      if (value.primitive().stream() != stream)
        throw std::invalid_argument("Apple region uses a foreign/CPU stream");
      for (const auto &input : value.inputs())
        pending.push_back(input);
    } else if (!value.is_tracer() && value.status() != mx::array::available) {
      throw std::invalid_argument("Apple region captures a pending array");
    }
  }
  std::optional<mx::array> status;
  for (const auto &check : trace.checks) {
    if (!visited.contains(check.result))
      continue; // Dead pure expressions do not add failures to the program.
    status = status ? mx::where(mx::equal(*status, mx::array(0), stream),
                                check.status, *status, stream)
                    : check.status;
  }
  if (status)
    outputs.push_back(std::move(*status));
  return status.has_value();
}

void deliver(cd::CompletionLifetime lifetime, Status terminal,
             const Status &hint) noexcept {
  if (!lifetime.valid())
    return;
  if (terminal.ok() && !hint.ok())
    terminal = hint;
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}
} // namespace

namespace detail {
Status Runtime::enqueue(Job &job) noexcept {
    {
      std::lock_guard lock(queue_mutex);
      if (!accepting)
        return Status::FailedPrecondition("Apple Context is closed");
      if (!failure.ok())
        return failure;
      job.next = nullptr;
      if (tail != nullptr)
        tail->next = &job;
      else
        head = &job;
      tail = &job;
    }
    wake.unpark();
    return Status::Ok();
  }

void Runtime::poison(Status terminal) noexcept {
    std::lock_guard lock(queue_mutex);
    if (failure.ok())
      failure = std::move(terminal);
  }

Status Runtime::reserve_native_tail() noexcept {
  std::lock_guard lock(queue_mutex);
  if (native_tails == SIZE_MAX)
    return Status::ResourceExhausted("Apple native retirement capacity");
  ++native_tails;
  native_tail_peak = (std::max)(native_tail_peak, native_tails);
  return Status::Ok();
}
void Runtime::release_native_tail() noexcept {
  {
    std::lock_guard lock(queue_mutex);
    HYLE_FATAL_ASSERT(native_tails);
    --native_tails;
  }
  wake.unpark();
}
void Runtime::enqueue_retirement(Job &job) noexcept {
  {
    std::lock_guard lock(queue_mutex);
    HYLE_FATAL_ASSERT(native_tails);
    // These are observers of already accepted native work, not new public
    // admission. Closing/poisoning cannot discard their original owners.
    job.next = head;
    head = &job;
    if (!tail) tail = &job;
  }
  wake.unpark();
}

void Runtime::run() noexcept {
    owner_id.store(platform::this_thread::get_id(), std::memory_order_release);
    for (;;) {
      Job *job;
      {
        std::lock_guard lock(queue_mutex);
        job = head;
        if (job != nullptr) {
          head = job->next;
          if (head == nullptr)
            tail = nullptr;
        } else if (!accepting && native_tails == 0) {
          break;
        }
      }
      if (job != nullptr)
        job->execute(job);
      else
        wake.park();
    }
  }

Status Runtime::close() noexcept {
    if (owner_id.load(std::memory_order_acquire) ==
        platform::this_thread::get_id())
      return Status::FailedPrecondition(
          "Apple Context cannot close on its native owner");
    std::lock_guard join_lock(close_mutex);
    {
      std::lock_guard lock(queue_mutex);
      accepting = false;
    }
    wake.unpark();
    if (thread.joinable())
      thread.join();
    return failure;
  }

struct ColdJob : Job {
  // The owner thread still needs this word while waking the caller. Keep a
  // separate admitted reference so the caller may return/destroy its handle
  // as soon as it observes done, without racing the final notification.
  std::shared_ptr<ColdJob> admitted_owner;
  std::function<Status()> body;
  Status terminal;
  std::atomic<std::uint32_t> done{0};
  explicit ColdJob(std::function<Status()> fn) : body(std::move(fn)) {
    execute = [](Job *job) noexcept {
      auto &self = *static_cast<ColdJob *>(job);
      auto owner = std::move(self.admitted_owner);
      try {
        self.terminal = self.body();
      } catch (...) {
        self.terminal = native_exception();
      }
      self.done.store(1, std::memory_order_release);
      platform::detail::wake_all(&self.done);
    };
  }
};

Status run_cold(Runtime &runtime,
                const std::shared_ptr<ColdJob> &job) noexcept {
  if (runtime.owner_id.load(std::memory_order_acquire) ==
      platform::this_thread::get_id())
    return Status::FailedPrecondition(
        "Apple cold preparation cannot reenter its native owner");
  job->admitted_owner = job;
  const Status admitted = runtime.enqueue(*job);
  if (!admitted.ok()) {
    job->admitted_owner.reset();
    return admitted;
  }
  while (job->done.load(std::memory_order_acquire) == 0)
    platform::detail::wait_on_address(&job->done, 0);
  return job->terminal;
}

struct MlxProgram {
  // Shared with the retained MLX tracing callable, never a ProgramState cycle.
  struct CheckContract {
    bool present = false;
  };
  std::shared_ptr<CheckContract> checks = std::make_shared<CheckContract>();
  std::function<std::vector<mx::array>(const std::vector<mx::array> &)>
      compiled;
};

// Immutable Program-owned recipe. The current version belongs to each native batch
// invocation; sharing a Program never shares mutable attention lengths.
struct RuntimeBatchFunction {
  BatchLayoutVersion initial;
  PagedBatchAccess::RuntimeFunction build;
  bool causal=true;
  std::int32_t window_left=-1;
  std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled;
  auto operator()(const std::vector<mx::array>& inputs) const {
    return compiled(inputs);
  }
};

// Compile each maximal MLX region once. Native region boundaries preserve its
// private fusion and do not expose its internal node/value table to replay.
std::vector<mx::array> prepare_mlx(MlxProgram &program, RegionFunction region,
                                  std::span<const InputSpec> inputs,
                                  mx::Stream stream,
                                  std::span<const std::size_t> descriptor_inputs = {}) {
  program.compiled = mx::compile(
      [fn = std::move(region), checks = program.checks,
       stream](const std::vector<mx::array> &arguments) {
        CheckTrace trace;
        auto outputs = fn(arguments, stream);
        if (!outputs)
          throw RegionFailure{std::move(outputs).error()};
        checks->present = validate_region(*outputs, stream, trace);
        return std::move(*outputs);
      }, false);
  std::vector<mx::array> isolated;
  isolated.reserve(inputs.size());
  for (const auto &spec : inputs) {
    const auto storage = spec.logical_description.storage_format().id();
    if (spec.dtype == mx::uint8 &&
        (std::ranges::find(descriptor_inputs, isolated.size()) != descriptor_inputs.end() ||
         storage == kFloat8GroupedWeightFloat32Block128 ||
         storage == kFloat4GroupedWeightFloat32Group32)) {
      // Canonical banks may exceed the native flat-index range. The traced
      // program is shape/dtype specialized; real bindings retain their full
      // capacity and strides. Declared row-bank ports also need only
      // one zero byte, including fused regions with other boundary ports.
      // Ordinary U8/opaque regions retain their original warmup allocation.
      isolated.push_back(mx::broadcast_to(mx::array(std::uint8_t{0}),
                                          spec.shape, stream));
    } else {
      isolated.push_back(mx::zeros(spec.shape, spec.dtype, stream));
    }
  }
  mx::async_eval(isolated);
  mx::synchronize(stream);
  for (auto &input : isolated)
    input.wait();
  auto outputs = program.compiled(isolated);
  mx::async_eval(outputs);
  mx::synchronize(stream);
  for (auto &output : outputs)
    output.wait();
  if (program.checks->present)
    outputs.pop_back(); // Isolated warmup data does not decide user validity.
  return outputs;
}

struct InvocationState;
Status evaluate_device_prefix(InvocationState &, mx::Stream,
                             const std::vector<mx::array> &);
Status evaluate_mlx(const MlxProgram &program,
                    const std::vector<mx::array> &inputs,
                    std::vector<mx::array> &outputs, mx::Stream stream,
                    std::vector<mx::Event> *pending = nullptr,
                    std::optional<mx::array> *check = nullptr,
                    InvocationState *device_prefix = nullptr) {
  encode_native_input_waits(inputs, stream);
  outputs = program.compiled(inputs);
  if (pending) {
    // GPU eval encodes buffers/events before returning. Consumers import the
    // original events; the same Invocation drains their errors at retirement.
    if (device_prefix)
      HYLE_RETURN_IF_ERROR(evaluate_device_prefix(*device_prefix, stream, outputs));
    else
      mx::async_eval(outputs);
    for (const auto &output : outputs) {
      if (output.event().valid()) {
        HYLE_FATAL_ASSERT(pending->size() < pending->capacity());
        pending->push_back(output.event());
      }
    }
    if (program.checks->present) {
      HYLE_FATAL_ASSERT(check && !check->has_value());
      check->emplace(outputs.back());
      outputs.pop_back();
    }
    return Status::Ok();
  }
  // MLX eval waits the original graph's synchronizer event and checks its
  // native error. A second stream synchronize commits an otherwise empty
  // command buffer after that graph and adds another host/device boundary.
  mx::eval(outputs);
  // Synchronous eval has retired the GPU work; retain the public ready-array
  // contract before a following native region borrows its physical buffer.
  for (auto &output : outputs)
    output.wait();
  if (!program.checks->present)
    return Status::Ok();
  const auto code = outputs.back().data<std::int32_t>()[0];
  outputs.pop_back();
  return algorithm_status(code);
}

struct AttentionBlocks {
  MlxProgram full;
  std::optional<MlxProgram> tail;
  int rows = 0;
  int query_axis = 1;
  bool ordinary = false;
  MaskKind mask = MaskKind::kNone;
  bool causal = false;
};

struct PreparedPlan {
  RegionPlan description;
  std::vector<std::optional<MlxProgram>> mlx;
  std::vector<std::optional<AttentionBlocks>> attention_blocks;
  std::vector<std::size_t> last_use;
};

struct ProgramState : MlxProgram {
  std::shared_ptr<Runtime> runtime;
  WorkloadContext workload;
  std::vector<InputSpec> inputs;
  std::vector<InputSpec> outputs;
  std::unique_ptr<PreparedPlan> native;
  bool device_prefix = false;
};

struct ResultValues {
  std::vector<mx::array> arrays;
};

struct InvocationState;
void retain(InvocationState *) noexcept;
void release(InvocationState *) noexcept;
void execute_native_tail(Job *) noexcept;
void native_prefix_completed(void *) noexcept;
Status observe_native_prefix(InvocationState &) noexcept;
Status observe_native_inputs(InvocationState &) noexcept;
Status drain_plan_regions(InvocationState &) noexcept;

// The locked MLX evaluator can commit several command buffers within one
// async_eval. Install the original Invocation callback after ALL actual output
// dependencies, rather than on whichever buffer existed before evaluation.
// This borrowed alias adds no Metal kernel or buffer and is never compiled
// into the model's mathematical recipe.
class DevicePrefixTail final : public mx::Primitive {
public:
  DevicePrefixTail(mx::Stream stream, InvocationState *owner)
      : Primitive(stream), owner_(owner) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
    throw std::logic_error("Apple device prefix tail requires its native GPU stream");
  }
  void eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) override;
  const char *name() const override { return "HyleOriginalDevicePrefixTail"; }
private:
  InvocationState *owner_;
};

struct InvocationState : Job {
  struct NativeTail final : Job { InvocationState *owner = nullptr; } native_tail;
  struct PreparedInput final {
    InputBinding binding;
    std::uint64_t expected_epoch = 0;
    std::uint64_t consumed_epoch = 0;
    bool pinned = false;
  };
  std::atomic<std::uint32_t> refs{1};
  std::shared_ptr<ProgramState> program;
  std::shared_ptr<ResultValues> values;
  cd::ResultSlot<Results> result_slot;
  LeaseUseBatch native_uses;
  // Only this accepted Job borrows the original receiver's stable Update.
  // Its existing source/suffix retirement owner keeps the payload alive.
  Status (*source_preflight)(const void *) noexcept = nullptr;
  const void *pending_source = nullptr;
  std::vector<mx::array> inputs;
  std::vector<PreparedInput> prepared_inputs;
  NativeDependencies dependencies;
  std::vector<std::unique_ptr<Recurrent>> recurrent;
  std::vector<std::optional<mx::array>> native_values;
  std::vector<mx::array> arguments;
  std::vector<mx::array> region_outputs;
  std::vector<mx::Event> mlx_events;
  std::vector<std::size_t> mlx_event_ends;
  std::vector<std::optional<mx::array>> mlx_checks;
  bool defer_mlx = false;
  std::vector<mx::array> block_arguments, block_outputs, output_blocks;
  std::mutex binding_mutex;
  std::mutex completion_mutex;
  std::atomic<std::uint32_t> done{1};
  std::atomic<bool> cancel{false};
  std::atomic<bool> retired{false};
  bool bound = false;
  bool source_bindings = false;
  // Cold source-edge preparation may sample the baseline without acquiring
  // another slot's binding lock while holding the Runtime DAG monitor.
  std::atomic<std::uint64_t> submitted_epoch{0};
  std::size_t source_readers = 0; // Protected by binding_mutex.
  bool outputs_detached = false; // Retired bindings withdrawn; readers still own outputs.
  Status terminal;
  cd::CompletionLifetime abandoned;
  CompletionNotification notification;
  Status terminal_hint;
  bool native_async = false; // Selected once from this actual prepared Plan.
  std::shared_ptr<DevicePrefixTail> device_prefix_tail;
  std::optional<mx::array> device_prefix_check;
  NS::SharedPtr<MTL::CommandBuffer> device_prefix_command;
  mx::Event device_prefix_event;
  std::size_t native_tail_region = SIZE_MAX;
  bool native_deferred = false, native_retired = false;
  bool native_values_ready = false; // Published under binding_mutex.
  std::uint64_t native_retired_epoch = 0;
  Status native_terminal;
  Status observe_inputs() noexcept;
  void encode_dependencies(mx::Stream stream) { dependencies.encode(stream); }
  // Caller holds this producer's binding_mutex. Returning the last real
  // reader also returns any retired array's original input-lease alias.
  void release_source_reader() noexcept {
    HYLE_FATAL_ASSERT(source_readers != 0);
    if (--source_readers == 0 && outputs_detached) values->arrays.clear();
  }

  InvocationState(std::shared_ptr<ProgramState> p, cd::ResultSlot<Results> slot)
      : program(std::move(p)), values(std::make_shared<ResultValues>()),
        result_slot(std::move(slot)) {
    inputs.reserve(program->inputs.size());
    native_tail.owner = this;
    native_tail.execute = &execute_native_tail;
  }

  // Source snapshots are taken at native submission, not at cold binding.
  // Counts belong to each producer; unrelated slots need not run equally often.
  Status snapshot_sources() noexcept {
    for (auto &input : prepared_inputs) {
      auto *producer = input.binding.source.state_;
      if (!producer) continue;
      std::lock_guard lock(producer->binding_mutex);
      const auto epoch = producer->submitted_epoch.load(std::memory_order_acquire);
      if (epoch == 0 || epoch <= input.consumed_epoch)
        return Status::FailedPrecondition("Apple late input has no new producer submission");
      if (producer->done.load(std::memory_order_acquire) != 0 &&
          !producer->terminal.ok())
        return producer->terminal;
      if (producer->source_readers == SIZE_MAX)
        return Status::ResourceExhausted("Apple late input reader capacity");
      input.expected_epoch = epoch;
      ++producer->source_readers;
      input.pinned = true;
    }
    return Status::Ok();
  }

  void release_source_pins() noexcept {
    for (auto &input : prepared_inputs) {
      if (!input.pinned) continue;
      auto &producer = *input.binding.source.state_;
      // Never hold two producer monitors, and never wait for native work.
      std::lock_guard lock(producer.binding_mutex);
      producer.release_source_reader();
      input.pinned = false;
    }
    dependencies.clear();
  }

  Status resolve_inputs() {
    if (!source_bindings) return Status::Ok();
    inputs.clear();
    for (const auto &input : prepared_inputs) {
      const auto &binding = input.binding;
      if (binding.value) {
        inputs.push_back(*binding.value);
        continue;
      }
      auto &producer = *binding.source.state_;
      std::lock_guard lock(producer.binding_mutex);
      const bool complete = producer.done.load(std::memory_order_acquire) != 0;
      if (!complete && producer.native_deferred && !producer.native_terminal.ok())
        return producer.native_terminal;
      if (producer.submitted_epoch.load(std::memory_order_acquire) != input.expected_epoch ||
          (!complete && !producer.native_values_ready))
        return Status::FailedPrecondition("Apple late input producer epoch is not ready");
      if (complete && !producer.terminal.ok()) return producer.terminal;
      if (producer.native_retired && !producer.native_terminal.ok()) return producer.native_terminal;
      if (binding.source.port_ >= producer.values->arrays.size())
        return Status::Internal("Apple late input producer output is absent");
      const auto &value = producer.values->arrays[binding.source.port_];
      const auto &spec = program->inputs[inputs.size()];
      const bool native_pending = !complete && producer.native_deferred &&
          value.status() == mx::array::evaluated && value.event().valid();
      // MLX can detach an already signaled event while leaving the raw
      // status evaluated. Its nonblocking availability query normalizes that
      // legitimate ready state; Source success/epoch and carrier remain exact.
      if ((!native_pending && !value.is_available()) || value.shape() != spec.shape ||
          value.dtype() != spec.dtype)
        return Status::Internal("Apple late input producer output differs from its prepared carrier");
      inputs.push_back(value); // Original owner/event stays pinned through retirement.
    }
    return Status::Ok();
  }
};

struct RuntimeBatchInvocationState final : InvocationState {
  BatchLayoutVersion layout;
  RuntimeBatchInvocationState(std::shared_ptr<ProgramState> p, cd::ResultSlot<Results> slot,
                       BatchLayoutVersion initial)
      : InvocationState(std::move(p),std::move(slot)),layout(std::move(initial)) {}
};
void execute_batch_invocation(Job*) noexcept;

Status evaluate_batch(RuntimeBatchInvocationState& state, mx::Stream stream) {
  const auto& fn=*state.program->compiled.target<RuntimeBatchFunction>();
  const auto a=fn.initial.host(),b=state.layout.host();
  const auto equal=[](auto x,auto y){return x.size()==y.size() && std::equal(x.begin(),x.end(),y.begin());};
  if(equal(a.kv_indptr,b.kv_indptr) && equal(a.last_page_lengths,b.last_page_lengths))
    return evaluate_mlx(*state.program,state.inputs,state.values->arrays,stream);
  CheckTrace trace;
  HYLE_ASSIGN_OR_RETURN(auto outputs,fn.build(state.layout,state.inputs,stream));
  // This private builder has the same cold-validated, selected-stream topology.
  // Preserve reachable checked statuses if a checked primitive is ever used.
  const bool checked=!trace.checks.empty() && validate_region(outputs,stream,trace);
  state.values->arrays=std::move(outputs);
  mx::async_eval(state.values->arrays);mx::synchronize(stream);
  for(auto& output:state.values->arrays)output.wait();
  if(!checked)return Status::Ok();
  const auto code=state.values->arrays.back().data<std::int32_t>()[0];
  state.values->arrays.pop_back();return algorithm_status(code);
}

Status evaluate_attention_blocks(const AttentionBlocks &program, InvocationState &state,
                                 mx::Stream stream) {
  encode_native_input_waits(state.arguments, stream);
  const auto &ports=state.arguments;
  auto &args=state.block_arguments;
  auto &outputs=state.block_outputs;
  auto &blocks=state.output_blocks;
  blocks.clear();
  const int keys = ports[1].shape(program.query_axis);
  std::optional<mx::array> broadcast_mask, key_indices;
  if (program.ordinary && program.mask != MaskKind::kNone) {
    // Match native mask broadcasting before slicing global query rows; do not
    // request a contiguous expansion of scalar/per-head/per-key masks.
    broadcast_mask = mx::broadcast_to(ports[3],
        {ports[0].shape(0), ports[0].shape(3 - program.query_axis),
         program.rows, keys}, stream);
  } else if (program.ordinary && program.causal) {
    key_indices = mx::arange(0, keys, mx::int32, stream);
  }
  for (int begin=0;begin<program.rows;) {
    const int end=begin+std::min(128,program.rows-begin);
    args.clear();
    auto first=mx::Shape(4,0),last=ports[0].shape();
    first[program.query_axis]=begin;last[program.query_axis]=end;
    args.push_back(mx::slice(ports[0],first,last,stream));
    args.push_back(ports[1]);args.push_back(ports[2]);
    if (broadcast_mask) {
      auto stop = broadcast_mask->shape();
      stop[2] = end;
      args.push_back(mx::slice(*broadcast_mask, {0,0,begin,0}, stop, stream));
    } else if (key_indices) {
      // Bottom-right causality uses the original full Q, including Q > KV.
      auto query_indices = mx::arange(keys - program.rows + begin,
          keys - program.rows + end, mx::int32, stream);
      args.push_back(mx::greater_equal(mx::expand_dims(query_indices, 1, stream),
          mx::expand_dims(*key_indices, 0, stream), stream));
    } else if (!program.ordinary) {
      auto mask_begin = mx::Shape(ports[3].ndim(), 0);
      auto mask_end = ports[3].shape();
      const int mask_query_axis = ports[3].ndim() - 2;
      mask_begin[mask_query_axis] = begin;
      mask_end[mask_query_axis] = end;
      args.push_back(mx::slice(ports[3], mask_begin, mask_end, stream));
      args.push_back(ports[4]);
    }
    const auto &part=end-begin==128 ? program.full : *program.tail;
    HYLE_RETURN_IF_ERROR(evaluate_mlx(part,args,outputs,stream));
    blocks.push_back(std::move(outputs[0]));
    outputs.clear();args.clear();
    begin=end;
  }
  state.region_outputs.clear();
  state.region_outputs.push_back(mx::concatenate(blocks,program.query_axis,stream));
  mx::async_eval(state.region_outputs);
  mx::synchronize(stream);
  for(auto &output:state.region_outputs)output.wait();
  blocks.clear();
  return Status::Ok();
}

Status execute_plan(InvocationState &state, mx::Stream stream) {
  const auto &prepared = *state.program->native;
  const auto &plan = prepared.description;
  auto &values = state.native_values;
  for (std::size_t i = 0; i < plan.inputs.size(); ++i)
    values[plan.inputs[i]] = state.inputs[i];
  for (std::size_t i = 0; i < plan.constants.size(); ++i)
    values[plan.constants[i]] = plan.constant_values[i];
  for (std::size_t i = 0; i < plan.regions.size(); ++i) {
    const auto &region = plan.regions[i];
    state.arguments.clear();
    for (auto id : region.inputs)
      state.arguments.push_back(*values[id]);
    if (region.kind == RegionKind::kMlx) {
      state.encode_dependencies(stream);
      HYLE_RETURN_IF_ERROR(evaluate_mlx(*prepared.mlx[i], state.arguments,
          state.region_outputs, stream, state.defer_mlx ? &state.mlx_events : nullptr,
          state.defer_mlx ? &state.mlx_checks[i] : nullptr));
      state.mlx_event_ends[i] = state.mlx_events.size();
    } else if (region.kind == RegionKind::kAttentionBlocks) {
      state.encode_dependencies(stream);
      HYLE_RETURN_IF_ERROR(evaluate_attention_blocks(*prepared.attention_blocks[i],state,stream));
    } else {
      if (state.recurrent[i]->host_indices)
        HYLE_RETURN_IF_ERROR(drain_plan_regions(state));
      if (state.recurrent[i]->host_indices)
        HYLE_RETURN_IF_ERROR(observe_native_inputs(state));
      if (state.recurrent[i]->completion_event.valid()) {
        auto &runtime = *state.program->runtime;
        auto &program = *state.recurrent[i]->program;
        const bool tail = state.native_async && i == state.native_tail_region;
        HYLE_RETURN_IF_ERROR(runtime.reserve_native_tail());
        struct PrefixAdmission final {
          Runtime &runtime;
          ggml_metal::NativeProgram &program;
          InvocationState &state;
          bool owns_tail = false;
          ~PrefixAdmission() {
            if (program.pending()) return; // Accepted prefix owns its reservation.
            runtime.release_native_tail();
            if (owns_tail) {
              state.native_deferred = false;
              runtime.release_native_tail();
            }
          }
        } admission{runtime, program, state};
        if (tail) {
          // Separate the actual prefix from its callback's original tail Job.
          // A dependent may observe the prefix before that callback is drained.
          HYLE_RETURN_IF_ERROR(runtime.reserve_native_tail());
          admission.owns_tail = true;
          state.native_deferred = true;
        }
        HYLE_ASSIGN_OR_RETURN(state.region_outputs,
            state.recurrent[i]->submit(state.arguments, stream,
                tail ? &native_prefix_completed : nullptr, tail ? &state : nullptr,
                state.recurrent[i]->host_indices ? std::span<const ggml_metal::NativeWait>{} : state.dependencies.waits(),
                state.recurrent[i]->host_indices ? std::span<mx::Event>{} : state.dependencies.events()));
        state.dependencies.native_submitted();
        HYLE_RETURN_IF_ERROR(program.submission_status());
      } else {
        HYLE_ASSIGN_OR_RETURN(state.region_outputs,
                              state.recurrent[i]->run(state.arguments, stream));
      }
    }
    for (std::size_t port = 0; port < region.outputs.size(); ++port)
      values[region.outputs[port]] = std::move(state.region_outputs[port]);
    state.region_outputs.clear();
    state.arguments.clear();
    for (auto id : region.inputs)
      if (prepared.last_use[id] == i)
        values[id].reset();
    for (auto id : region.outputs)
      if (prepared.last_use[id] == i)
        values[id].reset();
  }
  for (auto id : plan.outputs)
    state.values->arrays.push_back(*values[id]);
  for (auto &value : values)
    value.reset();
  if (state.native_deferred) {
    std::lock_guard lock(state.binding_mutex);
    state.native_values_ready = state.native_terminal.ok();
  }
  return Status::Ok();
}

void retain(InvocationState *state) noexcept {
  state->refs.fetch_add(1, std::memory_order_relaxed);
}
void release(InvocationState *state) noexcept {
  if (state != nullptr &&
      state->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    if(state->execute==execute_batch_invocation)
      delete static_cast<RuntimeBatchInvocationState*>(state);
    else delete state;
  }
}

InputSource::InputSource(InvocationState *state, std::size_t port) noexcept
    : state_(state), port_(port) { retain(state_); }
InputSource::InputSource(const InputSource &other) noexcept
    : state_(other.state_), port_(other.port_) { if (state_) retain(state_); }
InputSource &InputSource::operator=(const InputSource &other) noexcept {
  if (this != &other) { InputSource copy(other); *this = std::move(copy); }
  return *this;
}
InputSource::InputSource(InputSource &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)), port_(other.port_) {}
InputSource &InputSource::operator=(InputSource &&other) noexcept {
  if (this != &other) {
    reset(); state_ = std::exchange(other.state_, nullptr); port_ = other.port_;
  }
  return *this;
}
InputSource::~InputSource() { reset(); }
void InputSource::reset() noexcept { release(std::exchange(state_, nullptr)); port_ = 0; }

void finish(InvocationState &state, Status terminal) noexcept {
  cd::CompletionLifetime abandoned;
  CompletionNotification notification;
  LeaseUseBatch uses;
  Status hint;
  {
    std::lock_guard lock(state.completion_mutex);
    state.terminal = terminal;
    if (!terminal.ok())
      state.retired.store(true, std::memory_order_release);
    uses = std::move(state.native_uses);
  }
  cd::LeaseAccess::retire_native(uses, terminal);
  {
    std::lock_guard lock(state.completion_mutex);
    // A destroy during retire_native still leaves its attachment here. Publish
    // readiness only after permissions retire, and take every late callback in
    // this same critical section before another episode can reuse the slot.
    abandoned = std::exchange(state.abandoned, {});
    notification = std::exchange(state.notification, {});
    // Source dependencies retain a pointer to this epoch's hint. Moving it
    // would turn a successful retired producer into a code-only Unknown.
    hint = state.terminal_hint;
    state.done.store(1, std::memory_order_release);
  }
  platform::detail::wake_all(&state.done);
  deliver(abandoned, std::move(terminal), hint);
  deliver_notification(notification);
}

Status drain_plan_regions(InvocationState &state) noexcept {
  Status terminal, device_failure;
  if(state.device_prefix_command){
    try{state.device_prefix_command->waitUntilCompleted();}
    catch(...){device_failure=native_exception();terminal.Update(device_failure);}
    state.device_prefix_command.reset();
  }
  if(state.program->device_prefix){
    try{
      for(auto &event:state.mlx_events){event.wait();event.check_error();}
      if (state.device_prefix_event.valid()) {
        state.device_prefix_event.wait();
        state.device_prefix_event.check_error();
      }
      if(state.device_prefix_check){
        state.device_prefix_check->wait();
        terminal.Update(algorithm_status(state.device_prefix_check->data<std::int32_t>()[0]));
      }
    }catch(...){device_failure=native_exception();terminal.Update(device_failure);}
    state.mlx_events.clear();state.device_prefix_check.reset();
    if(!device_failure.ok())state.program->runtime->poison(device_failure);
    return terminal;
  }
  std::size_t cursor = 0;
  // Observe the original region order, even if later encoding also failed.
  // A deferred checked-input error must not become a later OOM/shape error.
  for (std::size_t i = 0; i < state.recurrent.size(); ++i) {
    const auto end = std::exchange(state.mlx_event_ends[i], 0);
    for (; cursor < end; ++cursor) {
      try { state.mlx_events[cursor].wait();state.mlx_events[cursor].check_error(); }
      catch (...) {
        auto failure = native_exception();
        device_failure.Update(failure); terminal.Update(failure);
      }
    }
    if (auto &check = state.mlx_checks[i]; check) {
      try {
        check->wait();
        terminal.Update(algorithm_status(check->data<std::int32_t>()[0]));
      } catch (...) {
        auto failure = native_exception();
        device_failure.Update(failure); terminal.Update(failure);
      }
      check.reset();
    }
    auto &region = state.recurrent[i];
    if (region && region->program->pending()) {
      terminal.Update(region->finish());
      state.program->runtime->release_native_tail();
    }
  }
  HYLE_FATAL_ASSERT(cursor == state.mlx_events.size());
  state.mlx_events.clear();
  if (!device_failure.ok()) state.program->runtime->poison(device_failure);
  return terminal;
}

// This proves native readiness on the original owner; public Completion and
// slot reuse still wait for the original completion callback's tail Job.
Status observe_native_prefix(InvocationState &state) noexcept {
  if (!state.native_deferred || state.native_retired) return state.native_terminal;
  auto terminal = drain_plan_regions(state);
  terminal.Update(state.native_terminal); // Earlier device checks precede a later encoding failure.
  state.native_terminal = std::move(terminal);
  state.native_terminal.Update(observe_native_inputs(state));
  if (state.native_terminal.ok()) {
    try {
      for (auto &output : state.values->arrays) output.wait();
    } catch (...) {
      state.native_terminal.Update(native_exception());
      state.program->runtime->poison(state.native_terminal);
    }
  }
  {
    std::lock_guard lock(state.completion_mutex);
    if (state.native_terminal.ok() && state.cancel.load(std::memory_order_acquire))
      state.native_terminal = Status::Cancelled("Apple submission cancelled after native drain");
    state.native_retired_epoch = state.submitted_epoch.load(std::memory_order_acquire);
    state.native_retired = true;
  }
  return state.native_terminal;
}
Status InvocationState::observe_inputs() noexcept {
  Status result;
  for (const auto &input : prepared_inputs) {
    if (!input.pinned) continue;
    auto &producer = *input.binding.source.state_;
    if (producer.native_deferred) result.Update(observe_native_prefix(producer));
    if (producer.done.load(std::memory_order_acquire)) result.Update(producer.terminal);
  }
  result.Update(dependencies.observe());
  return result;
}
Status NativeDependencies::reserve(std::size_t capacity) noexcept {
  if (!sources_.empty()) return Status::Unavailable("Apple dependencies are still pinned");
  if (capacity > (SIZE_MAX - 8) / 2) return Status::OutOfRange("Apple dependency capacity overflows");
  try {
    // A copy encoder imports both the native and MLX memory frontiers of
    // each original producer. Math consumers still use at most one of each.
    sources_.reserve(capacity); events_.reserve(2*capacity);
    native_events_.reserve(capacity); waits_.reserve(capacity);
    return Status::Ok();
  } catch (...) { return native_exception(); }
}
NativeDependencies::~NativeDependencies() { clear(); }
void NativeDependencies::clear() noexcept {
  for (const auto &dependency : sources_) {
    if (dependency.copy) { release_copy_dependency(*dependency.copy); continue; }
    auto &producer = *dependency.native;
    { std::lock_guard lock(producer.binding_mutex);
      producer.release_source_reader(); }
    release(&producer);
  }
  sources_.clear(); events_.clear(); native_events_.clear(); waits_.clear();
  encoded_ = native_encoded_ = false;
}
void NativeDependencies::encode(mx::Stream stream) {
  if (encoded_) return;
  for (auto &event : events_) event.wait(stream);
  encoded_ = true;
}
Status NativeDependencies::observe() noexcept {
  Status status;
  for (const auto &dependency : sources_) {
    if (dependency.copy) { status.Update(observe_copy_dependency(*dependency.copy, dependency.epoch)); continue; }
    auto &producer = *dependency.native;
    if (producer.native_deferred) status.Update(observe_native_prefix(producer));
    if (producer.done.load(std::memory_order_acquire)) status.Update(producer.terminal);
  }
  return status;
}
Status NativeDependencies::resolve() noexcept {
  return resolve_impl(false);
}
Status NativeDependencies::resolve_copy() noexcept {
  return resolve_impl(true);
}
Status NativeDependencies::resolve_impl(bool blit_stream) noexcept {
  encoded_ = native_encoded_ = false;
  HYLE_FATAL_ASSERT(events_.empty() && native_events_.empty() && waits_.empty());
  const auto append = [&](const mx::Event &event, bool from_native) -> Status {
    if (!event.valid() || event.is_signaled()) return Status::Ok();
    auto &owners = from_native ? events_ : native_events_;
    if (owners.size() == owners.capacity() ||
        (!from_native && waits_.size() == waits_.capacity()))
      return Status::Internal("Apple dependency frontier exceeds cold capacity");
    owners.push_back(event);
    if (!from_native)
      waits_.push_back({event.cast<mx::metal::EventImpl>().mtl_event(), event.value()});
    return Status::Ok();
  };
  // Runs only on the original FIFO owner. Each pinned producer's complete
  // encoding precedes this Job, including its last real native and MLX events.
  // The locked ggml device queue orders all native regions of this Runtime;
  // the original MLX stream orders its own regions. Import only the opposite
  // real queue's frontier. Raw blits are outside MLX's resource fence table,
  // so their memory frontier also needs an explicit MLX event import.
  for (const auto &dependency : sources_) {
    if (dependency.copy) {
      HYLE_ASSIGN_OR_RETURN(auto event, copy_dependency_event(*dependency.copy, dependency.epoch));
      HYLE_RETURN_IF_ERROR(append(event, false));
      if (!blit_stream) HYLE_RETURN_IF_ERROR(append(event, true));
      continue;
    }
    auto &producer = *dependency.native;
    if (producer.submitted_epoch.load(std::memory_order_acquire) != dependency.epoch)
      return Status::FailedPrecondition("Apple native dependency epoch was replaced");
    if (producer.done.load(std::memory_order_acquire)) {
      HYLE_RETURN_IF_ERROR(producer.terminal);
      continue;
    }
    if (!producer.native_deferred)
      return Status::FailedPrecondition("Apple dependency has no original encoded prefix");
    if (producer.native_retired || !producer.native_values_ready || !producer.native_terminal.ok()) {
      HYLE_RETURN_IF_ERROR(observe_native_prefix(producer));
      continue;
    }
    if(producer.program->device_prefix){
      HYLE_RETURN_IF_ERROR(append(producer.device_prefix_event,true));
    }else{
      HYLE_RETURN_IF_ERROR(append(producer.recurrent[producer.native_tail_region]->completion_event, true));
    }
    if (!producer.mlx_events.empty()) {
      HYLE_RETURN_IF_ERROR(append(producer.mlx_events.back(), false));
      // Native append/blit encoders are outside MLX's resource fence table.
      // They need its real memory frontier even on this same stream.
      if(blit_stream)HYLE_RETURN_IF_ERROR(append(producer.mlx_events.back(), true));
    }
  }
  return Status::Ok();
}
Status observe_native_inputs(InvocationState &state) noexcept { return state.observe_inputs(); }
void retire_invocation(InvocationState &state, Status terminal) noexcept {
  if (terminal.ok() && state.cancel.load(std::memory_order_acquire))
    terminal = Status::Cancelled("Apple submission cancelled after native drain");
  state.block_arguments.clear(); state.block_outputs.clear(); state.output_blocks.clear();
  if (state.source_bindings) state.inputs.clear();
  state.release_source_pins();
  finish(state, std::move(terminal));
  release(&state);
}
void native_prefix_completed(void *opaque) noexcept {
  auto &state = *static_cast<InvocationState *>(opaque);
  state.program->runtime->enqueue_retirement(state.native_tail);
}
Status arm_device_prefix(InvocationState &state, mx::Stream stream) {
  if (state.native_deferred)
    return Status::Ok();
  auto &runtime=*state.program->runtime;
  state.device_prefix_command=NS::RetainPtr(mx::metal::get_command_encoder(stream).get_command_buffer());
  if(!state.device_prefix_command)return Status::ResourceExhausted("Apple original MLX command buffer allocation failed");
  const auto admitted=runtime.reserve_native_tail();
  if(!admitted.ok()){state.device_prefix_command.reset();return admitted;}
  try{
    auto *original=&state;
    state.device_prefix_command->addCompletedHandler([original](MTL::CommandBuffer *){native_prefix_completed(original);});
    state.native_deferred=true;
  }catch(...){state.device_prefix_command.reset();runtime.release_native_tail();throw;}
  return Status::Ok();
}
void DevicePrefixTail::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  HYLE_FATAL_ASSERT(!inputs.empty() && outputs.size() == 1);
  outputs[0].copy_shared_buffer(inputs[0]);
  const auto status = arm_device_prefix(*owner_, stream());
  if (!status.ok()) throw RegionFailure{status};
  // This event belongs to this whole async_eval, even if every mathematical
  // output is only a view retaining an earlier producer's different event.
  HYLE_FATAL_ASSERT(outputs[0].event().valid());
  owner_->device_prefix_event = outputs[0].event();
}
Status evaluate_device_prefix(InvocationState &state, mx::Stream stream,
                             const std::vector<mx::array> &outputs) {
  if (outputs.empty() || std::ranges::all_of(outputs,
          [](const auto &value) { return value.is_available(); })) {
    mx::async_eval(outputs);
    return Status::Ok();
  }
  HYLE_FATAL_ASSERT(state.device_prefix_tail &&
                    state.device_prefix_tail->stream() == stream);
  // This fresh array descriptor must stay outside MLX compile/CSE. All real
  // outputs, including the checked status, precede it. MLX finishes its own
  // normal event/finalize epilogue, including for a pending view-only recipe;
  // HYLE creates no separate completion-only command buffer.
  auto marker = mx::array(outputs.front().shape(), outputs.front().dtype(),
                         state.device_prefix_tail, outputs);
  mx::async_eval({std::move(marker)});
  return Status::Ok();
}
void execute_native_tail(Job *job) noexcept {
  auto &state = *static_cast<InvocationState::NativeTail *>(job)->owner;
  const auto terminal = observe_native_prefix(state);
  state.program->runtime->release_native_tail();
  retire_invocation(state, terminal);
}

template<class Evaluate>
void execute_with(InvocationState& state, Evaluate evaluate) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.apple.execute");
  auto &runtime = *state.program->runtime;
  HYLE_FATAL_ASSERT(state.mlx_events.empty());
  state.native_deferred = false;
  if (state.program->device_prefix) state.device_prefix_event = {};
  {
    std::lock_guard lock(state.completion_mutex);
    state.native_retired = false;
  }
  state.native_terminal = {};
  {
    std::lock_guard lock(state.binding_mutex);
    state.native_values_ready = false;
  }
  // Clear borrowed fields before any finish/retirement callback can return the
  // source controls. Locals are used only before resolve/eval in this Job.
  const auto preflight = std::exchange(state.source_preflight, nullptr);
  const auto *source = std::exchange(state.pending_source, nullptr);
  Status terminal;
  {
    std::lock_guard lock(runtime.queue_mutex);
    terminal = runtime.failure;
  }
  if (terminal.ok() && state.cancel.load(std::memory_order_acquire))
    terminal =
        Status::Cancelled("Apple submission cancelled before native execution");
  if (terminal.ok()) {
    try {
      terminal = state.dependencies.resolve();
      if (terminal.ok() && preflight) terminal = preflight(source);
      if (terminal.ok()) terminal = state.resolve_inputs();
      if (terminal.ok()) terminal = evaluate(*runtime.stream);
    } catch (...) {
      terminal = native_exception();
      // Native construction/eval may already have accepted a prefix. Keep
      // every owner until its stream is checked and drained, even on failure.
      try {
        mx::synchronize(*runtime.stream);
      } catch (...) {
        // Preserve the first failure; synchronize already waited the buffer.
      }
      runtime.poison(terminal);
    }
  }
  if (state.native_deferred) {
    state.native_terminal.Update(terminal);
    return; // The accepted original prefix callback owns this worker reference.
  }
  auto native_terminal = drain_plan_regions(state);
  native_terminal.Update(terminal);
  terminal = std::move(native_terminal);
  if (terminal.ok()) {
    try {
      for (auto &output : state.values->arrays) output.wait();
    } catch (...) {
      terminal = native_exception();
      runtime.poison(terminal);
    }
  }
  terminal.Update(observe_native_inputs(state));
  // evaluate_mlx/execute_with have drained every accepted prefix before
  // temporary block owners are released. Whole-Job ports/uses retire in finish.
  retire_invocation(state, std::move(terminal));
}

void execute_invocation(Job* job) noexcept {
  auto& state=*static_cast<InvocationState*>(job);
  execute_with(state,[&](mx::Stream stream)->Status{
    if (!state.program->native) state.encode_dependencies(stream);
    if(!state.program->device_prefix)return state.program->native ? execute_plan(state,stream) :
        evaluate_mlx(*state.program,state.inputs,state.values->arrays,stream);
    HYLE_RETURN_IF_ERROR(evaluate_mlx(*state.program,state.inputs,state.values->arrays,stream,
        &state.mlx_events,&state.device_prefix_check,&state));
    if(state.native_deferred){
      std::lock_guard lock(state.binding_mutex);
      HYLE_FATAL_ASSERT(state.device_prefix_event.valid());
      state.native_values_ready=true;
    }
    return Status::Ok();
  });
}
void execute_batch_invocation(Job* job) noexcept {
  auto& state=*static_cast<RuntimeBatchInvocationState*>(job);
  execute_with(state,[&](mx::Stream stream){state.encode_dependencies(stream);return evaluate_batch(state,stream);});
}

const cd::CompletionVTable completion_vtable{
    .destroy =
        [](void *pointer, cd::CompletionLifetime lifetime,
           Status hint) noexcept {
          auto *state = static_cast<InvocationState *>(pointer);
          Status terminal;
          bool ready;
          {
            std::lock_guard lock(state->completion_mutex);
            ready = state->done.load(std::memory_order_acquire) != 0;
            if (ready)
              terminal = state->terminal;
            else {
              state->abandoned = lifetime;
              state->terminal_hint = hint;
            }
          }
          if (ready)
            deliver(lifetime, std::move(terminal), hint);
          release(state);
        },
    .poll = [](void *pointer) noexcept -> Result<CompletionPoll> {
      const auto &state = *static_cast<InvocationState *>(pointer);
      if (state.done.load(std::memory_order_acquire) == 0)
        return CompletionPoll::kPending;
      if (!state.terminal.ok())
        return std::unexpected(state.terminal);
      return CompletionPoll::kComplete;
    },
    .wait =
        [](void *pointer) noexcept {
          const auto &state = *static_cast<InvocationState *>(pointer);
          while (state.done.load(std::memory_order_acquire) == 0)
            platform::detail::wait_on_address(&state.done, 0);
          return state.terminal;
        },
    .request_cancel =
        [](void *pointer) noexcept {
          auto &state = *static_cast<InvocationState *>(pointer);
          std::lock_guard lock(state.completion_mutex);
          // An original native terminal observation is immutable. Public
          // readiness still waits for permission retirement in its tail Job.
          if (!state.native_retired || state.native_retired_epoch !=
              state.submitted_epoch.load(std::memory_order_acquire))
            state.cancel.store(true, std::memory_order_release);
          return Status::Ok();
        }};
Status NativeDependencies::capture(const std::shared_ptr<Runtime> &runtime,
    std::span<const cd::CompletionDependency> dependencies, InvocationState *consumer, CopyRecording *copy_consumer) noexcept {
  HYLE_FATAL_ASSERT(sources_.empty());
  return capture_impl(runtime,dependencies,consumer,copy_consumer,false);
}
Status NativeDependencies::append(const std::shared_ptr<Runtime> &runtime,
    std::span<const cd::CompletionDependency> dependencies, InvocationState *consumer, CopyRecording *copy_consumer) noexcept {
  return capture_impl(runtime,dependencies,consumer,copy_consumer,true);
}
Status NativeDependencies::capture_impl(const std::shared_ptr<Runtime> &runtime,
    std::span<const cd::CompletionDependency> dependencies,InvocationState *consumer,CopyRecording *copy_consumer,bool deduplicate) noexcept {
  for (const auto &source : dependencies) {
    if (!source.terminal_hint) return Status::FailedPrecondition("Apple dependency is empty");
    HYLE_RETURN_IF_ERROR(*source.terminal_hint);
    if (!source.pending()) continue;
    if(deduplicate && std::ranges::any_of(sources_,[&](const Source &captured){
      return source.vtable==&completion_vtable?captured.native==source.state:captured.copy.get()==source.state;
    }))continue;
    if (sources_.size() == sources_.capacity())
      return Status::ResourceExhausted("Apple dependency count exceeds cold preparation");
    if (source.vtable != &completion_vtable) {
      if (source.state == copy_consumer) return Status::InvalidArgument("Apple copy cannot depend on itself");
      std::uint64_t epoch = 0;
      HYLE_ASSIGN_OR_RETURN(auto copy, pin_copy_dependency(runtime, source, &epoch));
      sources_.push_back({nullptr, std::move(copy), epoch});
      continue;
    }
    auto *producer = static_cast<InvocationState *>(source.state);
    if (producer == consumer || producer->program->runtime != runtime)
      return Status::InvalidArgument("Apple dependency has another Runtime or names its consumer");
    if (sources_.size() == sources_.capacity())
      return Status::ResourceExhausted("Apple dependency count exceeds cold preparation");
    std::lock_guard lock(producer->binding_mutex);
    const auto epoch = producer->submitted_epoch.load(std::memory_order_acquire);
    if (!epoch) return Status::FailedPrecondition("Apple dependency has no accepted producer epoch");
    if (producer->done.load(std::memory_order_acquire)) HYLE_RETURN_IF_ERROR(producer->terminal);
    if (producer->cancel.load(std::memory_order_acquire))
      return Status::Cancelled("Apple producer cancelled before dependency admission");
    if (producer->source_readers == SIZE_MAX)
      return Status::ResourceExhausted("Apple dependency reader capacity");
    ++producer->source_readers;
    retain(producer);
    sources_.push_back({producer, {}, epoch});
  }
  return Status::Ok();
}
Status append_device_check(DeviceCheckInputs &inputs,NativeStorage word,const std::optional<mx::array> &owner) noexcept {
  if(inputs.count>inputs.words.size() || inputs.owners.size()!=inputs.words.size())
    return Status::InvalidArgument("Apple producer checks need their original bounded owner storage");
  if(!word.buffer || word.view_bytes!=sizeof(std::int32_t) || word.offset_bytes%alignof(std::int32_t))
    return Status::InvalidArgument("Apple producer check is not its original I32 word");
  for(std::size_t i=0;i<inputs.count;++i){const auto &existing=inputs.words[i];
    if(existing.buffer==word.buffer && existing.offset_bytes==word.offset_bytes){
      if(owner && !inputs.owners[i])inputs.owners[i]=owner;
      return Status::Ok();
    }
  }
  if(inputs.count==inputs.words.size())return Status::ResourceExhausted("Apple source checks exceed cold reserved capacity");
  inputs.words[inputs.count]=word;inputs.owners[inputs.count++]=owner;return Status::Ok();
}
Status invocation_device_checks(InvocationState &state,DeviceCheckInputs &inputs) noexcept {
  if(state.done.load(std::memory_order_acquire))return state.terminal;
  HYLE_RETURN_IF_ERROR(state.native_terminal);
  if(state.native_retired)return Status::Ok();
  if(!state.native_deferred || !state.native_values_ready)
    return Status::FailedPrecondition("Apple source checks require its original encoded producer");
  HYLE_ASSIGN_OR_RETURN(auto fresh,visit_device_check(inputs,&state,state.submitted_epoch.load(std::memory_order_acquire)));
  if(!fresh)return Status::Ok();
  for(const auto &check:state.mlx_checks)if(check){
    if(check->dtype()!=mx::int32 || check->size()!=1)
      return Status::Internal("Apple producer check differs from its native I32 contract");
    HYLE_ASSIGN_OR_RETURN(auto word,MemoryAccess::submitted_storage_info(*check));
    HYLE_RETURN_IF_ERROR(append_device_check(inputs,word,check));
  }
  if(const auto &check=state.device_prefix_check;check){
    if(check->dtype()!=mx::int32 || check->size()!=1)
      return Status::Internal("Apple device prefix check differs from its original I32 contract");
    HYLE_ASSIGN_OR_RETURN(auto word,MemoryAccess::submitted_storage_info(*check));
    HYLE_RETURN_IF_ERROR(append_device_check(inputs,word,check));
  }
  for(const auto &input:state.prepared_inputs)if(input.pinned){
    // The original cold DAG and epoch pins retain these exact ancestors.
    // Resolve their metadata, never their payload or public Completion.
    auto source=InvocationAccess::device_source_output(input.binding.source,state.program->runtime,input.expected_epoch,inputs);
    if(!source)return source.error();
  }
  return state.dependencies.device_checks(inputs);
}
Status NativeDependencies::device_checks(DeviceCheckInputs &inputs) noexcept {
  for(const auto &source:sources_){
    if(source.copy)HYLE_RETURN_IF_ERROR(copy_device_checks(*source.copy,inputs));
    else HYLE_RETURN_IF_ERROR(invocation_device_checks(*source.native,inputs));
  }
  return Status::Ok();
}
Status subscribe_completion(cd::CompletionDependency source,
                            CompletionNotification notification) noexcept {
  if (!notification.valid())
    return Status::InvalidArgument("Apple completion notification is invalid");
  if (!source.pending()) {
    if (source.vtable || source.state || !source.terminal_hint)
      return Status::FailedPrecondition("Apple completion source is empty");
    deliver_notification(notification);
    return Status::Ok();
  }
  if (source.vtable != &completion_vtable)
    return Status::Unimplemented("Apple completion owner has no notification bridge");
  auto &state = *static_cast<InvocationState *>(source.state);
  bool ready;
  {
    std::lock_guard lock(state.completion_mutex);
    if (state.notification.valid())
      return Status::Unavailable("Apple completion already has a subscriber");
    ready = state.done.load(std::memory_order_acquire) != 0;
    if (!ready) state.notification = notification;
  }
  if (ready) deliver_notification(notification);
  return Status::Ok();
}
Status qualify_copy_dependency(const std::shared_ptr<Runtime> &runtime, cd::CompletionDependency source) noexcept {
  if (!source.terminal_hint) return Status::FailedPrecondition("Apple copy producer is empty");
  if (!source.terminal_hint->ok()) return *source.terminal_hint;
  if (!source.pending()) return Status::Ok();
  if (source.vtable != &completion_vtable)
    return Status::Unimplemented("Apple copy producer is not an original Runtime invocation");
  const auto &state = *static_cast<InvocationState *>(source.state);
  if (state.program->runtime != runtime) return Status::InvalidArgument("Apple copy producer Runtime differs");
  if (state.done.load(std::memory_order_acquire) != 0 && !state.terminal.ok()) return state.terminal;
  if (state.done.load(std::memory_order_acquire) == 0 &&
      state.cancel.load(std::memory_order_acquire))
    return Status::Cancelled("Apple native producer cancelled before dependency admission");
  return Status::Ok();
}
} // namespace detail

std::span<const mx::array> Results::arrays() const noexcept {
  return values_ != nullptr ? std::span<const mx::array>(values_->arrays)
                            : std::span<const mx::array>{};
}

Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    close().IgnoreError();
    state_ = std::move(other.state_);
  }
  return *this;
}
Context::~Context() { close().IgnoreError(); }

Result<Context> Context::create() noexcept {
  try {
    auto runtime = std::make_shared<detail::Runtime>();
    auto thread = platform::Thread::spawn(
        [](void *pointer) { static_cast<detail::Runtime *>(pointer)->run(); },
        runtime.get(), {.name = "hyle-mlx"});
    if (!thread)
      return std::unexpected(std::move(thread).error());
    runtime->thread = std::move(*thread);
    Context context(runtime);
    auto initialize = std::make_shared<detail::ColdJob>([&]() -> Status {
      // The native availability count catches initialization exceptions and
      // turns a missing metallib into "no GPU". Initialize through the same
      // environment query we already need, so the original error reaches the
      // cold job's fallible boundary on this owner.
      runtime->environment = detail::read_environment();
      runtime->stream = mx::new_stream(mx::Device::gpu);
      return Status::Ok();
    });
    const Status status = detail::run_cold(*runtime, initialize);
    if (!status.ok())
      return std::unexpected(status);
    return context;
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Status Context::close() noexcept {
  return state_ != nullptr ? state_->close() : Status::Ok();
}

Result<const Environment &> Context::environment() const noexcept {
  if (!state_)
    return std::unexpected(Status::FailedPrecondition("invalid Apple Context"));
  return state_->environment;
}

Result<Program>
Context::compile(RegionFunction region,
                 std::span<const mx::array> examples,
                 const WorkloadContext &workload) const noexcept {
  try {
    std::vector<InputSpec> specs;
    specs.reserve(examples.size());
    for (const auto &array : examples) {
      if (array.status() != mx::array::available)
        return std::unexpected(
            Status::FailedPrecondition("Apple examples must be ready arrays"));
      specs.push_back({array.shape(), array.dtype()});
    }
    return compile(std::move(region), specs, workload);
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Result<Program>
Context::compile(RegionFunction region,
                 std::span<const InputSpec> inputs,
                 const WorkloadContext &workload) const noexcept {
  return detail::ContextAccess::compile(state_, std::move(region), inputs, workload);
}

Result<Program> detail::ContextAccess::compile(std::shared_ptr<Runtime> runtime,
                 RegionFunction region, std::span<const InputSpec> inputs,
                 const WorkloadContext &workload,
                 std::span<const std::size_t> descriptor_inputs) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.apple.prepare");
  if (!runtime || !region)
    return std::unexpected(
        Status::InvalidArgument("invalid Apple Context or region"));
  if (!workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple workload"));
  for (const auto port : descriptor_inputs)
    if (port >= inputs.size() || inputs[port].dtype != mx::uint8)
      return std::unexpected(Status::InvalidArgument("Apple descriptor-only warmup port differs"));
  try {
    auto program = std::make_shared<detail::ProgramState>();
    program->runtime = runtime;
    program->workload = workload;
    program->inputs.assign(inputs.begin(), inputs.end());
    auto prepare = std::make_shared<detail::ColdJob>([&]() -> Status {
      const auto stream = *runtime->stream;
      try {
        auto outputs = detail::prepare_mlx(*program, std::move(region),
                                           program->inputs, stream, descriptor_inputs);
        program->outputs.reserve(outputs.size());
        for (const auto &output : outputs)
          program->outputs.push_back({output.shape(), output.dtype()});
        return Status::Ok();
      } catch (...) {
        const Status failure = native_exception();
        try {
          mx::synchronize(stream);
        } catch (...) {
          runtime->poison(native_exception());
        }
        return failure;
      }
    });
    const Status status = detail::run_cold(*runtime, prepare);
    if (!status.ok())
      return std::unexpected(status);
    return Program(std::move(program));
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

WorkloadContext Program::workload() const noexcept {
  return state_ ? state_->workload : WorkloadContext{};
}

Result<Program> detail::ContextAccess::compile_device_prefix(std::shared_ptr<Runtime> runtime,
    RegionFunction region,std::span<const InputSpec> inputs,const WorkloadContext &workload,
    std::span<const std::size_t> descriptor_inputs) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto program,compile(std::move(runtime),std::move(region),inputs,workload,descriptor_inputs));
  program.state_->device_prefix=true;
  return program;
}

Result<Program> detail::ContextAccess::compile(const Context &context,
                                               RegionPlan plan) noexcept {
  return compile(context.state_, std::move(plan));
}

Result<Program> detail::ContextAccess::compile(std::shared_ptr<Runtime> runtime,
                                               RegionPlan plan,
                                               std::span<const std::size_t> descriptor_values) noexcept {
  if (!runtime || !plan.workload.valid() || plan.regions.empty() ||
      plan.outputs.empty() || plan.constants.size() != plan.constant_values.size())
    return std::unexpected(Status::InvalidArgument("invalid Apple native region plan"));
  for (const auto value : descriptor_values)
    if (value >= plan.values.size() || plan.values[value].dtype != mx::uint8 ||
        (std::ranges::find(plan.inputs, value) == plan.inputs.end() &&
         std::ranges::find(plan.constants, value) == plan.constants.end()))
      return std::unexpected(Status::InvalidArgument("Apple descriptor-only warmup value differs"));
  try {
    auto program = std::make_shared<ProgramState>();
    program->runtime = runtime;
    program->workload = plan.workload;
    program->native = std::make_unique<PreparedPlan>();
    auto &prepared = *program->native;
    prepared.last_use.resize(plan.values.size());
    std::vector<bool> defined(plan.values.size(), false);
    auto define = [&](std::size_t id) -> Status {
      if (id >= defined.size() || defined[id])
        return Status::InvalidArgument("Apple native plan has an invalid definition");
      defined[id] = true;
      return Status::Ok();
    };
    for (auto id : plan.inputs) {
      HYLE_RETURN_IF_ERROR(define(id));
      program->inputs.push_back(plan.values[id]);
    }
    for (std::size_t i = 0; i < plan.constants.size(); ++i) {
      const auto id = plan.constants[i];
      HYLE_RETURN_IF_ERROR(define(id));
      const auto &value = plan.constant_values[i];
      if (value.status() != mx::array::available ||
          value.shape() != plan.values[id].shape ||
          value.dtype() != plan.values[id].dtype)
        return std::unexpected(Status::InvalidArgument("Apple native constant is not ready or has the wrong schema"));
    }
    for (std::size_t i = 0; i < plan.regions.size(); ++i) {
      const auto &region = plan.regions[i];
      if ((region.proven_index_port != static_cast<std::size_t>(-1)) !=
              (region.proven_index_extent > 0) ||
          (region.proven_index_extent > 0 &&
           (region.kind != RegionKind::kGgml ||
            !region.operation.custom_as<model_ops::GroupedMatMul>() ||
            region.inputs.size() != 3 || region.native_inputs.size() != 3 ||
            region.native_inputs[0].rank() != 3 ||
            region.proven_index_port != 2 ||
            region.proven_index_extent != region.native_inputs[0].dimensions()[0])))
        return std::unexpected(Status::InvalidArgument("Apple native index proof has the wrong port/extent"));
      if (region.outputs.empty())
        return std::unexpected(Status::InvalidArgument("Apple region has no outputs"));
      for (auto id : region.inputs) {
        if (id >= defined.size() || !defined[id])
          return std::unexpected(Status::InvalidArgument("Apple region input precedes its definition"));
        prepared.last_use[id] = i;
      }
      for (auto id : region.outputs) {
        HYLE_RETURN_IF_ERROR(define(id));
        prepared.last_use[id] = i;
      }
      if (region.kind == RegionKind::kMlx) {
        if (!region.mlx)
          return std::unexpected(Status::InvalidArgument("Apple MLX region has no native callable"));
      } else if (region.kind == RegionKind::kGgml || region.kind == RegionKind::kAttentionBlocks) {
        const auto traits = operation_traits(region.operation);
        if (!traits.inputs.accepts(region.inputs.size()) ||
            region.outputs.size() != traits.outputs ||
            region.native_inputs.size() != region.inputs.size())
          return std::unexpected(
              Status::InvalidArgument("Apple native region arity mismatch"));
        if(region.kind==RegionKind::kGgml) {
          HYLE_RETURN_IF_ERROR(ggml_metal::NativeProgram::validate(
              region.operation, region.native_inputs));
        } else if(!is_mlx_attention_blocks(region.operation,region.native_inputs)) {
          return std::unexpected(Status::InvalidArgument("invalid Apple attention block region"));
        }
        std::vector<TensorDesc> outputs(traits.outputs);
        HYLE_RETURN_IF_ERROR(infer_output_descriptions(
            region.operation, region.native_inputs, outputs));
        auto match = [&](const TensorDesc &desc, std::size_t id) {
          const auto &spec = plan.values[id];
          const auto expected = input_spec(desc);
          return expected.has_value() && expected->shape == spec.shape &&
                 expected->dtype == spec.dtype &&
                 (!spec.logical_description.valid() ||
                  spec.logical_description == desc);
        };
        for (std::size_t port = 0; port < region.inputs.size(); ++port)
          if (!match(region.native_inputs[port], region.inputs[port]))
            return std::unexpected(
                Status::InvalidArgument("Apple native input schema mismatch"));
        for (std::size_t port = 0; port < outputs.size(); ++port)
          if (!match(outputs[port], region.outputs[port]))
            return std::unexpected(
                Status::InvalidArgument("Apple native output schema mismatch"));
      } else {
        return std::unexpected(Status::InvalidArgument("unknown Apple native region kind"));
      }
    }
    for (auto id : plan.outputs) {
      if (id >= defined.size() || !defined[id])
        return std::unexpected(Status::InvalidArgument("Apple native output has no definition"));
      prepared.last_use[id] = plan.regions.size();
      program->outputs.push_back(plan.values[id]);
    }
    prepared.description = std::move(plan);
    prepared.mlx.resize(prepared.description.regions.size());
    prepared.attention_blocks.resize(prepared.description.regions.size());
    auto prepare = std::make_shared<ColdJob>([&]() -> Status {
      const auto stream = *runtime->stream;
      try {
        for (std::size_t i = 0; i < prepared.description.regions.size(); ++i) {
          auto &region = prepared.description.regions[i];
          if (region.kind == RegionKind::kGgml) {
            // Qualify original pipeline/resources in preparation. Each actual
            // invocation subsequently owns an independent backend/metadata.
            HYLE_ASSIGN_OR_RETURN(
                auto probe,
                Recurrent::create(region.operation, region.native_inputs,
                    region.proven_index_port, region.proven_index_extent));
            continue;
          }
          std::vector<InputSpec> specs;
          for (auto id : region.inputs)
            specs.push_back(prepared.description.values[id]);
          if (region.kind == RegionKind::kAttentionBlocks) {
            auto &blocks = prepared.attention_blocks[i].emplace();
            const auto *attention = region.operation.get_if<ScaledDotProductAttention>();
            blocks.ordinary = attention != nullptr;
            blocks.query_axis = attention &&
                attention->layout == AttentionLayout::kBatchHeadSequenceDepth ? 2 : 1;
            blocks.rows = specs[0].shape[blocks.query_axis];
            blocks.mask = attention ? attention->mask : MaskKind::kNone;
            blocks.causal = attention && attention->causal;
            auto build = [operation = region.operation, workload = program->workload,
                          ordinary = blocks.ordinary](const auto &args,
                                                       mx::Stream native_stream) {
              return ordinary ? lower_attention_math_block(operation, args, native_stream)
                              : lower(operation, args, native_stream, workload);
            };
            const auto prepare_part = [&](MlxProgram &part, int rows) -> Status {
              auto part_specs = specs;
              part_specs[0].shape[blocks.query_axis] = rows;
              part_specs[0].logical_description = {};
              if (blocks.ordinary && blocks.mask != MaskKind::kNone) {
                part_specs[3].shape = {specs[0].shape[0],
                    specs[0].shape[3 - blocks.query_axis], rows,
                    specs[1].shape[blocks.query_axis]};
                part_specs[3].logical_description = {};
              } else if (blocks.ordinary && blocks.causal) {
                part_specs.push_back({{rows, specs[1].shape[blocks.query_axis]}, mx::bool_});
              } else if (!blocks.ordinary) {
                part_specs[3].shape[part_specs[3].shape.size() - 2] = rows;
                part_specs[3].logical_description = {};
              }
              auto outputs = prepare_mlx(part, build, part_specs, stream);
              auto expected_shape = part_specs[0].shape;
              expected_shape.back() = specs[2].shape[3];
              if (outputs.size() != 1 || outputs[0].shape() != expected_shape ||
                  outputs[0].dtype() != specs[0].dtype)
                return Status::InvalidArgument("Apple attention block schema mismatch");
              return Status::Ok();
            };
            HYLE_RETURN_IF_ERROR(prepare_part(blocks.full, 128));
            if (const int tail = blocks.rows % 128; tail != 0)
              HYLE_RETURN_IF_ERROR(prepare_part(blocks.tail.emplace(), tail));
            continue;
          }
          auto &mlx = prepared.mlx[i].emplace();
          std::vector<std::size_t> descriptor_inputs;
          for (std::size_t port = 0; port < region.inputs.size(); ++port)
            if (std::ranges::find(descriptor_values, region.inputs[port]) != descriptor_values.end())
              descriptor_inputs.push_back(port);
          auto outputs = prepare_mlx(mlx, std::move(region.mlx), specs, stream,
              descriptor_inputs);
          if (outputs.size() != region.outputs.size())
            return Status::InvalidArgument("Apple MLX region output count mismatch");
          for (std::size_t port = 0; port < outputs.size(); ++port) {
            const auto &spec = prepared.description.values[region.outputs[port]];
            if (spec.shape != outputs[port].shape() || spec.dtype != outputs[port].dtype())
              return Status::InvalidArgument("Apple MLX region output schema mismatch");
          }
        }
        return Status::Ok();
      } catch (...) {
        const auto failure = native_exception();
        try {
          mx::synchronize(stream);
        } catch (...) {
          runtime->poison(native_exception());
        }
        return failure;
      }
    });
    HYLE_RETURN_IF_ERROR(run_cold(*runtime, prepare));
    return Program(std::move(program));
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Result<Invocation>
Program::create_invocation(mem::AllocatorRef allocator) const noexcept {
  if (!state_)
    return std::unexpected(Status::FailedPrecondition("invalid Apple Program"));
  auto slot = cd::ResultSlot<Results>::create(allocator);
  if (!slot)
    return std::unexpected(std::move(slot).error());
  try {
    if(const auto* fn=state_->compiled.target<detail::RuntimeBatchFunction>()) {
      auto state=std::make_unique<detail::RuntimeBatchInvocationState>(state_,std::move(*slot),fn->initial);
      state->execute=detail::execute_batch_invocation;
      return Invocation(state.release());
    }
    auto state = std::make_unique<detail::InvocationState>(state_, std::move(*slot));
    state->execute = detail::execute_invocation;
    if (state_->device_prefix) {
      state->mlx_events.reserve(state_->outputs.size()+1);
      state->device_prefix_tail = std::make_shared<detail::DevicePrefixTail>(
          *state_->runtime->stream, state.get());
    }
    if (state_->native) {
      const auto &plan = state_->native->description;
      state->native_values.resize(plan.values.size());
      state->recurrent.resize(plan.regions.size());
      state->mlx_event_ends.resize(plan.regions.size());
      state->mlx_checks.resize(plan.regions.size());
      state->values->arrays.reserve(plan.outputs.size());
      std::size_t max_inputs = 0;
      std::size_t mlx_outputs = 0;
      for (const auto &region : plan.regions) {
        max_inputs = (std::max)(max_inputs, region.inputs.size());
        if (region.kind == detail::RegionKind::kMlx) {
          if (region.outputs.size() >= SIZE_MAX - mlx_outputs)
            return std::unexpected(Status::ResourceExhausted("Apple region event capacity"));
          mlx_outputs += region.outputs.size() + 1; // Optional device check result.
        }
      }
      state->mlx_events.reserve(mlx_outputs);
      state->arguments.reserve(max_inputs);
      state->block_arguments.reserve(5);state->block_outputs.reserve(1);
      std::size_t max_output_blocks=0;
      for (const auto &blocks : state_->native->attention_blocks)
        if (blocks) max_output_blocks = std::max(max_output_blocks,
            static_cast<std::size_t>((blocks->rows - 1) / 128 + 1));
      state->output_blocks.reserve(max_output_blocks);
      auto prepare = std::make_shared<detail::ColdJob>([&]() -> Status {
        bool has_native = false, all_signal = true;
        for (std::size_t i = 0; i < plan.regions.size(); ++i)
          if (plan.regions[i].kind == detail::RegionKind::kGgml) {
            HYLE_ASSIGN_OR_RETURN(state->recurrent[i],
                detail::Recurrent::create(plan.regions[i].operation,
                                          plan.regions[i].native_inputs,
                                          plan.regions[i].proven_index_port,
                                          plan.regions[i].proven_index_extent));
            auto signal = state->recurrent[i]->prepare_signal(*state_->runtime->stream);
            if (!signal) return std::move(signal).error();
            has_native = true;
            all_signal &= *signal;
            state->native_tail_region = i;
          }
        state->native_async = has_native && all_signal;
        state->defer_mlx = state->native_async && mlx_outputs != 0;
        return Status::Ok();
      });
      HYLE_RETURN_IF_ERROR(detail::run_cold(*state_->runtime, prepare));
    }
    return Invocation(state.release());
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

std::size_t Program::input_count() const noexcept { return state_ ? state_->inputs.size() : 0; }

Invocation::Invocation(Invocation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
Invocation &Invocation::operator=(Invocation &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
Invocation::~Invocation() { detail::release(state_); }

Status Invocation::bind(std::span<const mx::array> inputs) noexcept {
  if (!state_)
    return Status::FailedPrecondition("invalid Apple Invocation");
  std::lock_guard lock(state_->binding_mutex);
  if (state_->done.load(std::memory_order_acquire) == 0 ||
      state_->retired.load(std::memory_order_acquire))
    return Status::FailedPrecondition(
        "Apple Invocation is in flight or failed");
  if (state_->source_readers != 0)
    return Status::Unavailable("Apple outputs are retained by a pending native consumer");
  if (inputs.size() != state_->program->inputs.size())
    return Status::InvalidArgument(
        "Apple input count differs from prepared region");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &spec = state_->program->inputs[i];
    if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
      return Status::InvalidArgument(
          "Apple input shape/dtype differs from prepared region");
    if (inputs[i].status() != mx::array::available)
      return Status::FailedPrecondition(
          "Apple bind requires successfully evaluated inputs");
  }
  // Capacity was reserved at slot creation. array copies retain native owners.
  // All source-edge changes use this original Runtime's cold monitor. It lets
  // prepare_inputs prove a DAG without taking locks along arbitrary edges.
  std::lock_guard source_lock(state_->program->runtime->queue_mutex);
  state_->prepared_inputs.clear();
  state_->source_bindings = false;
  state_->inputs.assign(inputs.begin(), inputs.end());
  state_->bound = true;
  return Status::Ok();
}

Result<detail::InputSource> detail::InvocationAccess::source(
    const Invocation &invocation, std::size_t port) noexcept {
  auto *state = invocation.state_;
  if (!state) return std::unexpected(Status::FailedPrecondition("empty Apple source invocation"));
  if (port >= state->program->outputs.size())
    return std::unexpected(Status::OutOfRange("Apple source output port"));
  return InputSource(state, port);
}

Result<std::uint64_t> detail::InvocationAccess::source_epoch(
    const InputSource &source, const std::shared_ptr<Runtime> &runtime) noexcept {
  auto *state = source.state_;
  if (!state || state->program->runtime != runtime || source.port_ >= state->program->outputs.size())
    return std::unexpected(Status::InvalidArgument("Apple source Runtime/output port differs"));
  return state->submitted_epoch.load(std::memory_order_acquire);
}
Result<std::uint64_t> detail::InvocationAccess::pin_source(const InputSource &source,
    const std::shared_ptr<Runtime> &runtime, std::uint64_t consumed_epoch) noexcept {
  auto qualified = source_epoch(source, runtime);
  if (!qualified) return std::unexpected(std::move(qualified).error());
  auto &state = *source.state_;
  std::lock_guard lock(state.binding_mutex);
  const auto epoch = state.submitted_epoch.load(std::memory_order_acquire);
  if (!epoch || epoch <= consumed_epoch)
    return std::unexpected(Status::FailedPrecondition("Apple copy has no new producer submission"));
  if (state.done.load(std::memory_order_acquire) != 0 && !state.terminal.ok())
    return std::unexpected(state.terminal);
  if (state.source_readers == SIZE_MAX)
    return std::unexpected(Status::ResourceExhausted("Apple copy producer reader capacity"));
  ++state.source_readers;
  return epoch;
}
Result<mx::array> detail::InvocationAccess::source_output(const InputSource &source,
    const std::shared_ptr<Runtime> &runtime, std::uint64_t expected_epoch) noexcept {
  auto qualified = source_epoch(source, runtime);
  if (!qualified) return std::unexpected(std::move(qualified).error());
  if (runtime->owner_id.load(std::memory_order_acquire) != platform::this_thread::get_id())
    return std::unexpected(Status::FailedPrecondition("Apple source resolution requires its original FIFO owner"));
  auto &state = *source.state_;
  if (state.native_deferred && !state.done.load(std::memory_order_acquire))
    HYLE_RETURN_IF_ERROR(detail::observe_native_prefix(state));
  std::lock_guard lock(state.binding_mutex);
  if (!expected_epoch || state.submitted_epoch.load(std::memory_order_acquire) != expected_epoch ||
      (state.done.load(std::memory_order_acquire) == 0 && !state.native_retired) || !state.source_readers)
    return std::unexpected(Status::FailedPrecondition("Apple copy producer epoch is not ready/pinned"));
  if (!state.terminal.ok()) return std::unexpected(state.terminal);
  if (source.port_ >= state.values->arrays.size())
    return std::unexpected(Status::Internal("Apple copy producer output is absent"));
  const auto &value = state.values->arrays[source.port_];
  const auto &spec = state.program->outputs[source.port_];
  if (value.status() != mx::array::available || value.shape() != spec.shape || value.dtype() != spec.dtype)
    return std::unexpected(Status::Internal("Apple copy output differs from its prepared carrier"));
  return value;
}
Result<cd::CompletionDependency> detail::InvocationAccess::source_dependency(
    const InputSource &source,const std::shared_ptr<Runtime> &runtime) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto epoch,source_epoch(source,runtime));(void)epoch;
  return cd::CompletionDependency{&detail::completion_vtable,source.state_,&source.state_->terminal_hint};
}
Result<mx::array> detail::InvocationAccess::device_source_output(const InputSource &source,
    const std::shared_ptr<Runtime> &runtime,std::uint64_t expected_epoch,DeviceCheckInputs &checks) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto epoch,source_epoch(source,runtime));(void)epoch;
  if(runtime->owner_id.load(std::memory_order_acquire)!=platform::this_thread::get_id())
    return std::unexpected(Status::FailedPrecondition("Apple device source resolution requires its original FIFO owner"));
  auto &state=*source.state_;
  {
    std::lock_guard lock(state.binding_mutex);
    if(!expected_epoch || state.submitted_epoch.load(std::memory_order_acquire)!=expected_epoch || !state.source_readers)
      return std::unexpected(Status::FailedPrecondition("Apple device source lost its original pinned epoch"));
    if(state.done.load(std::memory_order_acquire))HYLE_RETURN_IF_ERROR(state.terminal);
    else if(!state.native_deferred || !state.native_values_ready)
      return std::unexpected(Status::FailedPrecondition("Apple device source has no original encoded values"));
    HYLE_RETURN_IF_ERROR(state.native_terminal);
    if(source.port_>=state.values->arrays.size())return std::unexpected(Status::Internal("Apple device source output is absent"));
  }
  HYLE_RETURN_IF_ERROR(detail::invocation_device_checks(state,checks));
  const auto &value=state.values->arrays[source.port_];const auto &spec=state.program->outputs[source.port_];
  if(value.shape()!=spec.shape || value.dtype()!=spec.dtype)
    return std::unexpected(Status::Internal("Apple device source differs from its original prepared carrier"));
  HYLE_ASSIGN_OR_RETURN(auto storage,detail::MemoryAccess::submitted_storage_info(value));(void)storage;
  return value;
}
void detail::InvocationAccess::release_source(const InputSource &source) noexcept {
  HYLE_FATAL_ASSERT(source.state_ != nullptr);
  std::lock_guard lock(source.state_->binding_mutex);
  source.state_->release_source_reader();
}

Status detail::InvocationAccess::prepare_inputs(Invocation &invocation,
    std::span<const InputBinding> inputs) noexcept {
  auto *state = invocation.state_;
  if (!state) return Status::FailedPrecondition("empty Apple native invocation");
  try {
    std::lock_guard lock(state->binding_mutex);
    if (!state->done.load(std::memory_order_acquire) || state->retired.load(std::memory_order_acquire))
      return Status::Unavailable("Apple native slot in flight or failed");
    if (state->source_readers != 0)
      return Status::Unavailable("Apple outputs are retained by a pending native consumer");
    HYLE_ASSIGN_OR_RETURN(auto writer, state->result_slot.begin());
    if (inputs.size() != state->program->inputs.size())
      return Status::InvalidArgument("Apple source input arity differs");
    std::lock_guard source_lock(state->program->runtime->queue_mutex);
    if (!state->program->runtime->accepting)
      return Status::FailedPrecondition("Apple Context is closed");
    if (!state->program->runtime->failure.ok()) return state->program->runtime->failure;
    std::vector<InvocationState::PreparedInput> candidate;
    candidate.reserve(inputs.size());
    std::vector<InvocationState *> pending;
    pending.reserve(inputs.size());
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      const auto &input = inputs[port];
      if (input.value.has_value() == input.source.valid())
        return Status::InvalidArgument("Apple input requires exactly one ready value or producer");
      const auto &spec = state->program->inputs[port];
      if (input.value) {
        if (input.value->status() != mx::array::available ||
            input.value->shape() != spec.shape || input.value->dtype() != spec.dtype)
          return Status::InvalidArgument("Apple ready input differs from its prepared carrier");
      } else {
        auto *producer = input.source.state_;
        if (producer->program->runtime != state->program->runtime)
          return Status::Unimplemented("Apple late inputs require the same native Runtime");
        if (input.source.port_ >= producer->program->outputs.size())
          return Status::OutOfRange("Apple late input output port");
        const auto &output = producer->program->outputs[input.source.port_];
        if (output.shape != spec.shape || output.dtype != spec.dtype)
          return Status::InvalidArgument("Apple late input carrier differs");
        pending.push_back(producer);
      }
      InvocationState::PreparedInput prepared{input};
      if (input.source.valid())
        prepared.consumed_epoch = input.source.state_->submitted_epoch.load(std::memory_order_acquire);
      if (port < state->prepared_inputs.size()) {
        const auto &previous = state->prepared_inputs[port];
        if (input.source.valid() && input.source.state_ == previous.binding.source.state_ &&
            input.source.port_ == previous.binding.source.port_)
          prepared.consumed_epoch = std::max(prepared.consumed_epoch, previous.consumed_epoch);
      }
      candidate.push_back(std::move(prepared));
    }
    container::FlatHashSet<InvocationState *> visited;
    while (!pending.empty()) {
      auto *producer = pending.back(); pending.pop_back();
      if (producer == state) return Status::InvalidArgument("Apple late input graph contains a cycle");
      const auto inserted = visited.try_insert(producer);
      if (inserted == decltype(visited)::InsertResult::kOutOfMemory)
        return Status::ResourceExhausted("Apple source DAG metadata");
      if (inserted == decltype(visited)::InsertResult::kAlreadyPresent) continue;
      for (const auto &input : producer->prepared_inputs)
        if (input.binding.source.valid()) pending.push_back(input.binding.source.state_);
    }
    state->prepared_inputs = std::move(candidate);
    state->source_bindings = true;
    state->inputs.clear();
    state->bound = true;
    return Status::Ok();
  } catch (...) { return native_exception(); }
}

Status detail::InvocationAccess::release_inputs(Invocation &invocation) noexcept {
  return PagedBatchAccess::release_inputs(invocation);
}
Status detail::InvocationAccess::can_detach_inputs(const Invocation &invocation,
    const cd::ResultStorageOwner &result) noexcept {
  return PagedBatchAccess::can_detach_inputs(invocation, result);
}
void detail::InvocationAccess::detach_inputs(Invocation &invocation) noexcept {
  PagedBatchAccess::detach_inputs(invocation);
}
Status detail::InvocationAccess::reuse_gate(Invocation &invocation) noexcept {
  return PagedBatchAccess::reuse_gate(invocation);
}
Result<OwnedResult<Results>> detail::InvocationAccess::submit_ordered(
    Invocation &invocation, LeaseUseBatch uses,
    Status (*preflight)(const void *) noexcept, const void *source) noexcept {
  return invocation.submit_guarded(std::move(uses), preflight, source);
}
Status detail::InvocationAccess::reserve_dependencies(Invocation &invocation, std::size_t capacity) noexcept {
  auto *state = invocation.state_;
  if (!state) return Status::FailedPrecondition("empty Apple native invocation");
  if (capacity > (SIZE_MAX - 8) / 2) return Status::OutOfRange("Apple dependency capacity overflows");
  std::lock_guard lock(state->binding_mutex);
  if (!state->done.load(std::memory_order_acquire) || state->retired.load(std::memory_order_acquire) ||
      state->source_readers || !state->dependencies.empty())
    return Status::Unavailable("Apple dependency preparation requires an idle original slot");
  try {
    HYLE_RETURN_IF_ERROR(state->dependencies.reserve(capacity));
    for (auto &region : state->recurrent)
      if (region) HYLE_RETURN_IF_ERROR(region->program->reserve_waits(8 + 2 * capacity));
    return Status::Ok();
  } catch (...) { return native_exception(); }
}
Result<OwnedResult<Results>> detail::InvocationAccess::submit_ordered(
    Invocation &invocation, LeaseUseBatch uses,
    Status (*preflight)(const void *) noexcept, const void *source,
    std::span<const cd::CompletionDependency> dependencies) noexcept {
  return invocation.submit_guarded(std::move(uses), preflight, source, dependencies);
}
Result<mx::array> detail::InvocationAccess::output(const Invocation &invocation,
    std::size_t port) noexcept {
  auto *state = invocation.state_;
  if (!state) return std::unexpected(Status::FailedPrecondition("empty Apple native invocation"));
  std::lock_guard lock(state->binding_mutex);
  if (state->done.load(std::memory_order_acquire) == 0 ||
      state->submitted_epoch.load(std::memory_order_acquire) == 0)
    return std::unexpected(Status::FailedPrecondition("Apple output requires a completed submission"));
  if (!state->terminal.ok()) return std::unexpected(state->terminal);
  if (port >= state->values->arrays.size())
    return std::unexpected(Status::OutOfRange("Apple native output port"));
  return state->values->arrays[port];
}
std::shared_ptr<detail::Runtime> detail::InvocationAccess::runtime(
    const Invocation &invocation) noexcept {
  return invocation.state_ ? invocation.state_->program->runtime : nullptr;
}

Status detail::PagedBatchAccess::reuse_gate(Invocation&invocation)noexcept{
  auto*state=invocation.state_;if(!state)return Status::FailedPrecondition("empty Apple batch native slot");
  std::lock_guard lock(state->binding_mutex);
  if(!state->done.load(std::memory_order_acquire) || state->retired.load(std::memory_order_acquire))return Status::Unavailable("Apple batch native slot in flight/failed");
  if(state->source_readers)return Status::Unavailable("Apple outputs retained by pending native consumer");
  HYLE_ASSIGN_OR_RETURN(auto writer,state->result_slot.begin());return Status::Ok();
}
bool detail::PagedBatchAccess::empty_cache_reads(const Invocation&invocation)noexcept{
  auto*state=invocation.state_;
  if(!state || state->execute!=execute_batch_invocation)return false;
  std::lock_guard lock(state->binding_mutex);
  if(!state->done.load(std::memory_order_acquire) || state->retired.load(std::memory_order_acquire) ||
      state->program->inputs.size()!=1)return false;
  const auto&layout=static_cast<const RuntimeBatchInvocationState*>(state)->layout;
  const auto*function=state->program->compiled.target<RuntimeBatchFunction>();
  return function && function->initial.origin_schema().owns(layout) &&
      layout.origin_schema().capacity().storage==BatchStorage::kPaged &&
      !layout.facts().query_rows && !layout.facts().page_refs;
}
Status detail::PagedBatchAccess::release_inputs(Invocation&invocation)noexcept{
  if(!invocation.state_)return Status::FailedPrecondition("empty Apple native slot");
  auto&state=*invocation.state_;
  std::lock_guard lock(state.binding_mutex);
  if(!state.done.load(std::memory_order_acquire) || state.retired.load(std::memory_order_acquire))return Status::Unavailable("Apple native slot in flight/failed");
  if(state.source_readers)return Status::Unavailable("Apple outputs retained by pending native consumer");
  HYLE_ASSIGN_OR_RETURN(auto writer,state.result_slot.begin());
  std::lock_guard source_lock(state.program->runtime->queue_mutex);
  state.prepared_inputs.clear();state.source_bindings=false;
  state.bound=false;state.inputs.clear();state.values->arrays.clear();
  for(auto&value:state.native_values)value.reset();
  state.arguments.clear();state.region_outputs.clear();return Status::Ok();
}
Status detail::PagedBatchAccess::can_detach_inputs(const Invocation &invocation,
    const cd::ResultStorageOwner &result) noexcept {
  auto *state = invocation.state_;
  if (!state) return Status::FailedPrecondition("empty Apple native slot");
  std::lock_guard lock(state->binding_mutex);
  if (!state->done.load(std::memory_order_acquire))
    return Status::Unavailable("Apple native inputs are still in use");
  // done is published only after native/source/page uses retire. Failed or
  // closed slots may be cleaned; this does not reopen execution admission.
  // Outgoing readers retain these original outputs, not permission to reuse
  // the producer. detach_inputs preserves their arrays until those readers
  // retire; bind/submit/reuse still reject every outstanding source reader.
  return cd::can_detach_retired(state->result_slot, result);
}
void detail::PagedBatchAccess::detach_inputs(Invocation &invocation) noexcept {
  auto &state = *invocation.state_;
  std::lock_guard lock(state.binding_mutex);
  cd::discard_retired_payload(state.result_slot);
  std::lock_guard source_lock(state.program->runtime->queue_mutex);
  state.prepared_inputs.clear(); state.source_bindings = false;
  state.bound = false; state.inputs.clear();
  state.outputs_detached = true;
  if (!state.source_readers) state.values->arrays.clear();
  for (auto &value : state.native_values) value.reset();
  state.arguments.clear(); state.region_outputs.clear();
  state.block_arguments.clear(); state.block_outputs.clear(); state.output_blocks.clear();
  for (auto &recurrent : state.recurrent)
    if (recurrent) recurrent->detach_inputs();
}
Result<Program> detail::PagedBatchAccess::prepare_runtime_program(const Context& context,
    BatchLayoutVersion initial,std::span<const InputSpec> inputs,RuntimeFunction build,
    const WorkloadContext& workload,bool causal,std::int32_t window_left) noexcept {
  return prepare_runtime_program(ContextAccess::runtime(context),std::move(initial),inputs,
      std::move(build),workload,causal,window_left);
}
Result<Program> detail::PagedBatchAccess::prepare_runtime_program(std::shared_ptr<Runtime> runtime,
    BatchLayoutVersion initial,std::span<const InputSpec> inputs,RuntimeFunction build,
    const WorkloadContext& workload,bool causal,std::int32_t window_left) noexcept {
  try {
    HYLE_ASSIGN_OR_RETURN(auto program,ContextAccess::compile(std::move(runtime),
        [initial,build](const std::vector<mx::array>& args,mx::Stream stream){return build(initial,args,stream);},
        inputs,workload));
    auto& compiled=program.state_->compiled;
    compiled=RuntimeBatchFunction{std::move(initial),std::move(build),causal,window_left,std::move(compiled)};
    return program;
  } catch(...) {return std::unexpected(native_exception());}
}
Status detail::PagedBatchAccess::update_layout(Invocation& invocation,
    const BatchLayoutVersion& layout) noexcept {
  HYLE_RETURN_IF_ERROR(reuse_gate(invocation));auto& state=*invocation.state_;
  if(state.execute!=execute_batch_invocation)
    return Status::InvalidArgument("Apple runtime page layout requires a paged invocation");
  const auto& initial=state.program->compiled.target<RuntimeBatchFunction>()->initial;
  const auto a=initial.host(),b=layout.host();
  const auto empty_ports=[](const BatchLayoutVersion& v){return v.origin_schema().capacity().storage==BatchStorage::kPaged && v.facts().query_rows==0 && v.facts().page_refs==0;};
  if(!initial.origin_schema().owns(layout) || a.query_indptr.size()!=b.query_indptr.size() ||
      !std::equal(a.query_indptr.begin(),a.query_indptr.end(),b.query_indptr.begin()) ||
      empty_ports(initial)!=empty_ports(layout))
    return Status::InvalidArgument("Apple runtime page Q topology differs");
  std::lock_guard lock(state.binding_mutex);
  static_cast<RuntimeBatchInvocationState&>(state).layout=layout;return Status::Ok();
}
Status detail::PagedBatchAccess::update_ragged(Invocation& invocation,
    const BatchLayoutVersion& layout) noexcept {
  auto* state=invocation.state_;
  if(!state || state->execute!=execute_batch_invocation)
    return Status::InvalidArgument("Apple ragged update requires a ragged Program");
  const auto& fn=*state->program->compiled.target<RuntimeBatchFunction>();
  const auto a=fn.initial.host(),b=layout.host();
  if(fn.initial.origin_schema().capacity().storage!=BatchStorage::kRagged ||
      !fn.initial.origin_schema().owns(layout) || a.query_indptr.size()!=b.query_indptr.size() ||
      !std::equal(a.query_indptr.begin(),a.query_indptr.end(),b.query_indptr.begin()))
    return Status::InvalidArgument("Apple ragged update schema/Q topology differs");
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,fn.causal,fn.window_left));
  for(std::size_t r=0;r<layout.facts().requests;++r)
    if(b.query_indptr[r+1]>b.query_indptr[r] && b.kv_indptr[r+1]==b.kv_indptr[r])
      return Status::InvalidArgument("Apple nonempty ragged query requires KV");
  {
    std::lock_guard lock(state->program->runtime->queue_mutex);
    if(!state->program->runtime->accepting)return Status::FailedPrecondition("Apple ragged Context is closed");
    if(!state->program->runtime->failure.ok())return state->program->runtime->failure;
  }
  HYLE_RETURN_IF_ERROR(release_inputs(invocation));
  std::lock_guard lock(state->binding_mutex);
  static_cast<RuntimeBatchInvocationState&>(*state).layout=layout;return Status::Ok();
}
Status update_ragged_batch(Invocation& invocation,const BatchLayoutVersion& layout) noexcept {
  return detail::PagedBatchAccess::update_ragged(invocation,layout);
}
Result<OwnedResult<Results>> Invocation::submit() noexcept {
  return submit_guarded({});
}

Result<OwnedResult<Results>> Invocation::submit_guarded(LeaseUseBatch uses,
    Status (*preflight)(const void *) noexcept, const void *source) noexcept {
  return submit_guarded(std::move(uses), preflight, source, {});
}
Result<OwnedResult<Results>> Invocation::submit_guarded(LeaseUseBatch uses,
    Status (*preflight)(const void *) noexcept, const void *source,
    std::span<const cd::CompletionDependency> dependencies) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.apple.submit");
  if (!state_)
    return std::unexpected(
        Status::FailedPrecondition("invalid Apple Invocation"));
  if (bool(preflight) != bool(source))
    return std::unexpected(Status::InvalidArgument("Apple source preflight/payload differs"));
  std::lock_guard lock(state_->binding_mutex);
  if (!state_->bound || state_->retired.load(std::memory_order_acquire))
    return std::unexpected(
        Status::FailedPrecondition("Apple Invocation is unbound or failed"));
  if (state_->source_readers != 0)
    return std::unexpected(Status::Unavailable("Apple outputs retained by pending native consumer"));
  if (state_->submitted_epoch.load(std::memory_order_relaxed) == UINT64_MAX)
    return std::unexpected(Status::ResourceExhausted("Apple native submission epoch exhausted"));
  auto writer = state_->result_slot.begin();
  if (!writer)
    return std::unexpected(std::move(writer).error());
  struct SourcePins final {
    detail::InvocationState &state;
    bool admitted = false;
    ~SourcePins() { if (!admitted) state.release_source_pins(); }
  } pins{*state_};
  HYLE_RETURN_IF_ERROR(state_->snapshot_sources());
  HYLE_RETURN_IF_ERROR(state_->dependencies.capture(state_->program->runtime, dependencies, state_));
  state_->outputs_detached = false;
  state_->values->arrays.clear();
  const Status populated = writer->emplace(Results(state_->values));
  if (!populated.ok())
    return std::unexpected(populated);
  state_->terminal = Status::Ok();
  state_->terminal_hint = Status::Ok();
  state_->cancel.store(false, std::memory_order_relaxed);
  state_->done.store(0, std::memory_order_release);
  state_->native_uses = std::move(uses);
  detail::retain(state_); // Completion reference, released by its native vtable
  auto completion =
      cd::CompletionAccess::make(&detail::completion_vtable, state_);
  auto result = writer->publish(std::move(completion));
  if (!result) {
    detail::finish(*state_, result.error());
    return std::unexpected(std::move(result).error());
  }
  const auto previous_epoch = state_->submitted_epoch.fetch_add(1, std::memory_order_release);
  state_->source_preflight = preflight;
  state_->pending_source = source;
  detail::retain(state_); // worker reference exists before queue admission
  const Status admitted = state_->program->runtime->enqueue(*state_);
  if (!admitted.ok()) {
    state_->source_preflight = nullptr;
    state_->pending_source = nullptr;
    state_->submitted_epoch.store(previous_epoch, std::memory_order_release);
    detail::finish(*state_, admitted);
    detail::release(state_);
    return std::unexpected(admitted);
  }
  for (auto &input : state_->prepared_inputs)
    if (input.binding.source.valid()) input.consumed_epoch = input.expected_epoch;
  pins.admitted = true; // The original Job now releases pins after its drain.
  return result;
}

Status Invocation::wait_native() noexcept {
  if (!state_) return Status::FailedPrecondition("empty Apple native invocation");
  return detail::completion_vtable.wait(state_);
}
} // namespace hyle::compute::apple
