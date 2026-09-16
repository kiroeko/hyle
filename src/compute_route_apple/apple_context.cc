#include "hyle/compute/apple.h"

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

#include "completion_internal.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/container/flat_hash_set.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace cd = compute::detail;

namespace {
Status native_exception() noexcept {
  try {
    throw;
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

// Cold only. Native primitives remain in one MLX GPU stream; the complete
// compiled region is one hot invocation. Reuse the existing fallible hash set.
void validate_region(const std::vector<mx::array> &outputs, mx::Stream stream) {
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
    if (owner_id.load(std::memory_order_acquire) == platform::this_thread::get_id())
      return Status::FailedPrecondition("Apple Context cannot close on its native owner");
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
  std::function<Status()> body;
  Status terminal;
  std::atomic<std::uint32_t> done{0};
  explicit ColdJob(std::function<Status()> fn) : body(std::move(fn)) {
    execute = [](Job *job) noexcept {
      auto &self = *static_cast<ColdJob *>(job);
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

Status run_cold(Runtime &runtime, ColdJob &job) noexcept {
  if (runtime.owner_id.load(std::memory_order_acquire) == platform::this_thread::get_id())
    return Status::FailedPrecondition("Apple cold preparation cannot reenter its native owner");
  const Status admitted = runtime.enqueue(job);
  if (!admitted.ok())
    return admitted;
  while (job.done.load(std::memory_order_acquire) == 0)
    platform::detail::wait_on_address(&job.done, 0);
  return job.terminal;
}

struct ArraySpec {
  mx::Shape shape;
  mx::Dtype type;
};

struct ProgramState {
  std::shared_ptr<Runtime> runtime;
  std::vector<ArraySpec> inputs;
  std::function<std::vector<mx::array>(const std::vector<mx::array> &)> compiled;
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

void retain(InvocationState *state) noexcept {
  state->refs.fetch_add(1, std::memory_order_relaxed);
}
void release(InvocationState *state) noexcept {
  if (state != nullptr && state->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
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
  auto &state = *static_cast<InvocationState *>(job);
  auto &runtime = *state.program->runtime;
  Status terminal;
  {
    std::lock_guard lock(runtime.queue_mutex);
    terminal = runtime.failure;
  }
  if (terminal.ok() && state.cancel.load(std::memory_order_acquire))
    terminal = Status::Cancelled("Apple submission cancelled before native execution");
  if (terminal.ok()) {
    try {
      state.values->arrays = state.program->compiled(state.inputs);
      mx::async_eval(state.values->arrays);
      mx::synchronize(*runtime.stream);
      // After checked stream completion, mark native outputs ready before
      // publishing the release/acquire terminal word to arbitrary waiters.
      for (auto &array : state.values->arrays)
        array.wait();
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
      terminal = Status::Cancelled("Apple submission cancelled after native drain");
  }
  finish(state, std::move(terminal));
  release(&state); // admitted worker reference
}

const cd::CompletionVTable completion_vtable{
    .destroy = [](void *pointer, cd::CompletionLifetime lifetime, Status hint) noexcept {
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
    .wait = [](void *pointer) noexcept {
      const auto &state = *static_cast<InvocationState *>(pointer);
      while (state.done.load(std::memory_order_acquire) == 0)
        platform::detail::wait_on_address(&state.done, 0);
      return state.terminal;
    },
    .request_cancel = [](void *pointer) noexcept {
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
    detail::ColdJob initialize([&]() -> Status {
      if (!mx::is_available(mx::Device(mx::Device::gpu)))
        return Status::Unavailable("MLX Metal GPU is unavailable");
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

Result<Program> Context::compile(RegionFunction region,
                                std::span<const mx::array> examples) const noexcept {
  if (!state_ || !region)
    return std::unexpected(Status::InvalidArgument("invalid Apple Context or region"));
  try {
    auto program = std::make_shared<detail::ProgramState>();
    program->runtime = state_;
    for (const auto &array : examples) {
      if (array.status() != mx::array::available)
        return std::unexpected(Status::FailedPrecondition("Apple examples must be ready arrays"));
      program->inputs.push_back({array.shape(), array.dtype()});
    }
    detail::ColdJob prepare([&]() -> Status {
      const auto stream = *state_->stream;
      try {
        program->compiled = mx::compile(
            [fn = std::move(region), stream](const std::vector<mx::array> &inputs) {
              auto outputs = fn(inputs, stream);
              validate_region(outputs, stream);
              return outputs;
            }, false);
        std::vector<mx::array> isolated;
        isolated.reserve(program->inputs.size());
        for (const auto &spec : program->inputs)
          isolated.push_back(mx::zeros(spec.shape, spec.type, stream));
        mx::async_eval(isolated);
        mx::synchronize(stream);
        for (auto &input : isolated)
          input.wait();
        auto outputs = program->compiled(isolated);
        mx::async_eval(outputs);
        mx::synchronize(stream);
        for (auto &output : outputs)
          output.wait();
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

Result<Invocation> Program::create_invocation(mem::AllocatorRef allocator) const noexcept {
  if (!state_)
    return std::unexpected(Status::FailedPrecondition("invalid Apple Program"));
  auto slot = cd::ResultSlot<Results>::create(allocator);
  if (!slot)
    return std::unexpected(std::move(slot).error());
  try {
    auto *state = new detail::InvocationState(state_, std::move(*slot));
    state->execute = detail::execute_invocation;
    return Invocation(state);
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
    return Status::FailedPrecondition("Apple Invocation is in flight or failed");
  if (inputs.size() != state_->program->inputs.size())
    return Status::InvalidArgument("Apple input count differs from prepared region");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &spec = state_->program->inputs[i];
    if (inputs[i].dtype() != spec.type || inputs[i].shape() != spec.shape)
      return Status::InvalidArgument("Apple input shape/dtype differs from prepared region");
    if (inputs[i].status() != mx::array::available)
      return Status::FailedPrecondition("Apple bind requires successfully evaluated inputs");
  }
  // Capacity was reserved at slot creation. array copies retain native owners.
  state_->inputs.assign(inputs.begin(), inputs.end());
  state_->bound = true;
  return Status::Ok();
}

Result<OwnedResult<Results>> Invocation::submit() noexcept {
  if (!state_)
    return std::unexpected(Status::FailedPrecondition("invalid Apple Invocation"));
  std::lock_guard lock(state_->binding_mutex);
  if (!state_->bound || state_->retired.load(std::memory_order_acquire))
    return std::unexpected(Status::FailedPrecondition("Apple Invocation is unbound or failed"));
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
  auto completion = cd::CompletionAccess::make(&detail::completion_vtable, state_);
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
