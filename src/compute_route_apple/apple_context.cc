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
#include <mlx/device.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/transforms.h>

#include "apple_checks.h"
#include "apple_internal.h"
#include "apple_recurrent.h"
#include "completion_internal.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/generated_result.h"
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
struct Job {
  void (*execute)(Job *) noexcept = nullptr;
  Job *next = nullptr;
};

struct Runtime {
  std::mutex queue_mutex;
  std::mutex close_mutex;
  platform::Parker wake;
  platform::Thread thread;
  std::atomic<std::uint64_t> owner_id{0};
  Job *head = nullptr;
  Job *tail = nullptr;
  bool accepting = true;
  Status failure;
  Environment environment;
  std::optional<mx::Stream> stream;

  Status enqueue(Job &job) noexcept {
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

  void poison(Status terminal) noexcept {
    std::lock_guard lock(queue_mutex);
    if (failure.ok())
      failure = std::move(terminal);
  }

  void run() noexcept {
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
        } else if (!accepting) {
          break;
        }
      }
      if (job != nullptr)
        job->execute(job);
      else
        wake.park();
    }
  }

  Status close() noexcept {
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
};

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

// Compile each maximal MLX region once. Native region boundaries preserve its
// private fusion and do not expose its internal node/value table to replay.
std::vector<mx::array> prepare_mlx(MlxProgram &program, RegionFunction region,
                                  std::span<const InputSpec> inputs,
                                  mx::Stream stream) {
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
  for (const auto &spec : inputs)
    isolated.push_back(mx::zeros(spec.shape, spec.dtype, stream));
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

Status evaluate_mlx(const MlxProgram &program,
                    const std::vector<mx::array> &inputs,
                    std::vector<mx::array> &outputs, mx::Stream stream) {
  outputs = program.compiled(inputs);
  mx::async_eval(outputs);
  mx::synchronize(stream);
  for (auto &output : outputs)
    output.wait();
  if (!program.checks->present)
    return Status::Ok();
  const auto code = outputs.back().data<std::int32_t>()[0];
  outputs.pop_back();
  return algorithm_status(code);
}

struct PreparedPlan {
  RegionPlan description;
  std::vector<std::optional<MlxProgram>> mlx;
  std::vector<std::size_t> last_use;
};

struct ProgramState : MlxProgram {
  std::shared_ptr<Runtime> runtime;
  WorkloadContext workload;
  std::vector<InputSpec> inputs;
  std::unique_ptr<PreparedPlan> native;
};

struct ResultValues {
  std::vector<mx::array> arrays;
};

struct InvocationState : Job {
  std::atomic<std::uint32_t> refs{1};
  std::shared_ptr<ProgramState> program;
  std::shared_ptr<ResultValues> values;
  cd::ResultSlot<Results> result_slot;
  std::vector<mx::array> inputs;
  std::vector<std::unique_ptr<Recurrent>> recurrent;
  std::vector<std::optional<mx::array>> native_values;
  std::vector<mx::array> arguments;
  std::vector<mx::array> region_outputs;
  std::mutex binding_mutex;
  std::mutex completion_mutex;
  std::atomic<std::uint32_t> done{1};
  std::atomic<bool> cancel{false};
  std::atomic<bool> retired{false};
  bool bound = false;
  Status terminal;
  cd::CompletionLifetime abandoned;
  Status terminal_hint;

  InvocationState(std::shared_ptr<ProgramState> p, cd::ResultSlot<Results> slot)
      : program(std::move(p)), values(std::make_shared<ResultValues>()),
        result_slot(std::move(slot)) {
    inputs.reserve(program->inputs.size());
  }
};

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
      HYLE_RETURN_IF_ERROR(evaluate_mlx(*prepared.mlx[i], state.arguments,
                                        state.region_outputs, stream));
    } else {
      HYLE_ASSIGN_OR_RETURN(state.region_outputs,
                            state.recurrent[i]->run(state.arguments, stream));
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
  return Status::Ok();
}

void retain(InvocationState *state) noexcept {
  state->refs.fetch_add(1, std::memory_order_relaxed);
}
void release(InvocationState *state) noexcept {
  if (state != nullptr &&
      state->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
    delete state;
}

void finish(InvocationState &state, Status terminal) noexcept {
  cd::CompletionLifetime abandoned;
  Status hint;
  {
    std::lock_guard lock(state.completion_mutex);
    state.terminal = terminal;
    if (!terminal.ok())
      state.retired.store(true, std::memory_order_release);
    abandoned = std::exchange(state.abandoned, {});
    hint = std::move(state.terminal_hint);
    state.done.store(1, std::memory_order_release);
  }
  platform::detail::wake_all(&state.done);
  deliver(abandoned, std::move(terminal), hint);
}

void execute_invocation(Job *job) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.apple.execute");
  auto &state = *static_cast<InvocationState *>(job);
  auto &runtime = *state.program->runtime;
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
      terminal = state.program->native
                     ? execute_plan(state, *runtime.stream)
                     : evaluate_mlx(*state.program, state.inputs,
                                    state.values->arrays, *runtime.stream);
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
    if (terminal.ok() && state.cancel.load(std::memory_order_acquire))
      terminal =
          Status::Cancelled("Apple submission cancelled after native drain");
  }
  finish(state, std::move(terminal));
  release(&state); // admitted worker reference
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
          state.cancel.store(true, std::memory_order_release);
          return Status::Ok();
        }};
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
      if (!mx::is_available(mx::Device(mx::Device::gpu)))
        return Status::Unavailable("MLX Metal GPU is unavailable");
      runtime->stream = mx::new_stream(mx::Device::gpu);
      runtime->environment = detail::read_environment();
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
  HYLE_TRACE_ZONE_NAMED("hyle.apple.prepare");
  if (!state_ || !region)
    return std::unexpected(
        Status::InvalidArgument("invalid Apple Context or region"));
  if (!workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple workload"));
  try {
    auto program = std::make_shared<detail::ProgramState>();
    program->runtime = state_;
    program->workload = workload;
    program->inputs.assign(inputs.begin(), inputs.end());
    auto prepare = std::make_shared<detail::ColdJob>([&]() -> Status {
      const auto stream = *state_->stream;
      try {
        (void)detail::prepare_mlx(*program, std::move(region),
                                   program->inputs, stream);
        return Status::Ok();
      } catch (...) {
        const Status failure = native_exception();
        try {
          mx::synchronize(stream);
        } catch (...) {
          state_->poison(native_exception());
        }
        return failure;
      }
    });
    const Status status = detail::run_cold(*state_, prepare);
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

Result<Program> detail::ContextAccess::compile(const Context &context,
                                               RegionPlan plan) noexcept {
  if (!context.state_ || !plan.workload.valid() || plan.regions.empty() ||
      plan.outputs.empty() || plan.constants.size() != plan.constant_values.size())
    return std::unexpected(Status::InvalidArgument("invalid Apple native region plan"));
  try {
    auto program = std::make_shared<ProgramState>();
    program->runtime = context.state_;
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
      } else if (region.kind == RegionKind::kGgml) {
        const auto traits = operation_traits(region.operation);
        if (!traits.inputs.accepts(region.inputs.size()) ||
            region.outputs.size() != traits.outputs ||
            region.native_inputs.size() != region.inputs.size())
          return std::unexpected(
              Status::InvalidArgument("Apple ggml region arity mismatch"));
        HYLE_RETURN_IF_ERROR(ggml_metal::NativeProgram::validate(
            region.operation, region.native_inputs));
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
                Status::InvalidArgument("Apple ggml input schema mismatch"));
        for (std::size_t port = 0; port < outputs.size(); ++port)
          if (!match(outputs[port], region.outputs[port]))
            return std::unexpected(
                Status::InvalidArgument("Apple ggml output schema mismatch"));
      } else {
        return std::unexpected(Status::InvalidArgument("unknown Apple native region kind"));
      }
    }
    for (auto id : plan.outputs) {
      if (id >= defined.size() || !defined[id])
        return std::unexpected(Status::InvalidArgument("Apple native output has no definition"));
      prepared.last_use[id] = plan.regions.size();
    }
    prepared.description = std::move(plan);
    prepared.mlx.resize(prepared.description.regions.size());
    auto prepare = std::make_shared<ColdJob>([&]() -> Status {
      const auto stream = *context.state_->stream;
      try {
        for (std::size_t i = 0; i < prepared.description.regions.size(); ++i) {
          auto &region = prepared.description.regions[i];
          if (region.kind == RegionKind::kGgml) {
            // Qualify original pipeline/resources in preparation. Each actual
            // invocation subsequently owns an independent backend/metadata.
            HYLE_ASSIGN_OR_RETURN(
                auto probe,
                Recurrent::create(region.operation, region.native_inputs));
            continue;
          }
          std::vector<InputSpec> specs;
          for (auto id : region.inputs)
            specs.push_back(prepared.description.values[id]);
          auto &mlx = prepared.mlx[i].emplace();
          auto outputs = prepare_mlx(mlx, std::move(region.mlx), specs, stream);
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
          context.state_->poison(native_exception());
        }
        return failure;
      }
    });
    HYLE_RETURN_IF_ERROR(run_cold(*context.state_, prepare));
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
    auto state = std::make_unique<detail::InvocationState>(state_, std::move(*slot));
    state->execute = detail::execute_invocation;
    if (state_->native) {
      const auto &plan = state_->native->description;
      state->native_values.resize(plan.values.size());
      state->recurrent.resize(plan.regions.size());
      state->values->arrays.reserve(plan.outputs.size());
      std::size_t max_inputs = 0;
      for (const auto &region : plan.regions)
        max_inputs = (std::max)(max_inputs, region.inputs.size());
      state->arguments.reserve(max_inputs);
      auto prepare = std::make_shared<detail::ColdJob>([&]() -> Status {
        for (std::size_t i = 0; i < plan.regions.size(); ++i)
          if (plan.regions[i].kind == detail::RegionKind::kGgml) {
            HYLE_ASSIGN_OR_RETURN(state->recurrent[i],
                detail::Recurrent::create(plan.regions[i].operation,
                                          plan.regions[i].native_inputs));
          }
        return Status::Ok();
      });
      HYLE_RETURN_IF_ERROR(detail::run_cold(*state_->runtime, prepare));
    }
    return Invocation(state.release());
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

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
  state_->inputs.assign(inputs.begin(), inputs.end());
  state_->bound = true;
  return Status::Ok();
}

Result<OwnedResult<Results>> Invocation::submit() noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.apple.submit");
  if (!state_)
    return std::unexpected(
        Status::FailedPrecondition("invalid Apple Invocation"));
  std::lock_guard lock(state_->binding_mutex);
  if (!state_->bound || state_->retired.load(std::memory_order_acquire))
    return std::unexpected(
        Status::FailedPrecondition("Apple Invocation is unbound or failed"));
  auto writer = state_->result_slot.begin();
  if (!writer)
    return std::unexpected(std::move(writer).error());
  state_->values->arrays.clear();
  const Status populated = writer->emplace(Results(state_->values));
  if (!populated.ok())
    return std::unexpected(populated);
  state_->terminal = Status::Ok();
  state_->cancel.store(false, std::memory_order_relaxed);
  state_->done.store(0, std::memory_order_release);
  detail::retain(state_); // Completion reference, released by its native vtable
  auto completion =
      cd::CompletionAccess::make(&detail::completion_vtable, state_);
  auto result = writer->publish(std::move(completion));
  if (!result) {
    detail::finish(*state_, result.error());
    return std::unexpected(std::move(result).error());
  }
  detail::retain(state_); // worker reference exists before queue admission
  const Status admitted = state_->program->runtime->enqueue(*state_);
  if (!admitted.ok()) {
    detail::finish(*state_, admitted);
    detail::release(state_);
    return std::unexpected(admitted);
  }
  return result;
}
} // namespace hyle::compute::apple
