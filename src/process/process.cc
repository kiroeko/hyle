#include "hyle/process/process.h"

#include <algorithm>
#include <mutex>
#include <utility>

#include "hyle/core/assert.h"
#include "process_internal.h"

namespace hyle::process {
namespace detail {
namespace {

bool contains_nul(std::string_view text) noexcept {
  return text.find('\0') != std::string_view::npos;
}

bool valid_stdio_kind(ProcessStdioKind kind) noexcept {
  switch (kind) {
    case ProcessStdioKind::kInherit:
    case ProcessStdioKind::kNull:
    case ProcessStdioKind::kFile:
    case ProcessStdioKind::kPipe:
    case ProcessStdioKind::kMergeStdout:
      return true;
  }
  return false;
}

Status validate_stdio(ProcessStdio value, bool stderr_slot) noexcept {
  if (!valid_stdio_kind(value.kind()))
    return Status::InvalidArgument("process stdio kind is invalid");
  if (value.kind() == ProcessStdioKind::kMergeStdout && !stderr_slot)
    return Status::InvalidArgument(
        "merge_stdout is valid only for child stderr");
  if (value.kind() == ProcessStdioKind::kFile &&
      value.file_handle() == platform::FileHandle::kInvalid) {
    return Status::InvalidArgument("process stdio file handle is not open");
  }
  return Status{};
}

Status validate_start_options(const ProcessStartOptions& options) noexcept {
  if (options.executable.empty() || contains_nul(options.executable))
    return Status::InvalidArgument(
        "process executable must be non-empty and contain no NUL");
  if (options.executable_search != ExecutableSearch::kExact &&
      options.executable_search != ExecutableSearch::kPath) {
    return Status::InvalidArgument("process executable search mode is invalid");
  }
  if (options.environment_mode != EnvironmentMode::kInherit &&
      options.environment_mode != EnvironmentMode::kReplace) {
    return Status::InvalidArgument("process environment mode is invalid");
  }
  if (options.working_directory.has_value() &&
      (options.working_directory->empty() ||
       contains_nul(*options.working_directory))) {
    return Status::InvalidArgument(
        "process working directory must be non-empty and contain no NUL");
  }
  for (std::string_view argument : options.arguments) {
    if (contains_nul(argument))
      return Status::InvalidArgument("process argument contains NUL");
  }
  for (std::size_t i = 0; i < options.environment_changes.size(); ++i) {
    const EnvironmentChange& change = options.environment_changes[i];
    if (change.name.empty() || contains_nul(change.name) ||
        change.name.find('=') != std::string_view::npos) {
      return Status::InvalidArgument(
          "environment variable name must be non-empty and contain no '=' or NUL");
    }
    if (change.value.has_value() && contains_nul(*change.value))
      return Status::InvalidArgument("environment variable value contains NUL");
    if (options.environment_mode == EnvironmentMode::kReplace &&
        !change.value.has_value()) {
      return Status::InvalidArgument(
          "replacement environment cannot contain unset entries");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (options.environment_changes[j].name == change.name)
        return Status::InvalidArgument(
            "process environment contains duplicate variable names");
    }
  }
  Status status = validate_stdio(options.stdin_stream, false);
  if (!status.ok()) return status;
  status = validate_stdio(options.stdout_stream, false);
  if (!status.ok()) return status;
  status = validate_stdio(options.stderr_stream, true);
  if (!status.ok()) return status;

  const bool has_pipe =
      options.stdin_stream.kind() == ProcessStdioKind::kPipe ||
      options.stdout_stream.kind() == ProcessStdioKind::kPipe ||
      options.stderr_stream.kind() == ProcessStdioKind::kPipe;
  if (has_pipe) {
    if (options.pipe_io == nullptr || !options.pipe_io->valid())
      return Status::FailedPrecondition(
          "process pipe stdio requires a valid IoContext");
    if (options.pipe_owner >= options.pipe_io->num_workers())
      return Status::OutOfRange(
          "process pipe owner is outside IoContext");
  }
  return Status{};
}

Status close_pipes_locked(ProcessState* state) noexcept {
  Status result;
  result.Update(state->stdin_stream.close());
  result.Update(state->stdout_stream.close());
  result.Update(state->stderr_stream.close());
  return result;
}

Result<ProcessExit> completed_result_locked(const ProcessState* state) noexcept {
  HYLE_FATAL_ASSERT(state->exit_published);
  if (!state->wait_status.ok())
    return make_unexpected(state->wait_status);
  return state->exit;
}

struct CancelWake {
  ProcessState* state = nullptr;

  static void notify(void* raw) noexcept {
    auto* self = static_cast<CancelWake*>(raw);
    ProcessRuntimeState* runtime = self->state->runtime;
    std::lock_guard<util::SpinLock> lock(runtime->monitor);
    self->state->completion.notify_all();
  }
};

Result<ProcessExit> wait_locked(
    ProcessState* state, const ProcessWaitOptions& options,
    std::unique_lock<util::SpinLock>& lock) noexcept {
  for (;;) {
    if (state->exit_published) return completed_result_locked(state);
    if (options.cancel != nullptr && options.cancel->is_cancelled())
      return make_unexpected(Status::Cancelled("process wait cancelled"));
    if (options.deadline.has_value()) {
      if (*options.deadline <= time::monotonic_now())
        return make_unexpected(
            Status::DeadlineExceeded("process wait deadline exceeded"));
      (void)state->completion.wait_until(lock, *options.deadline);
    } else {
      state->completion.wait(lock);
    }
  }
}

void finish_spawn(ProcessRuntimeState* runtime) noexcept {
  std::lock_guard<util::SpinLock> lock(runtime->monitor);
  HYLE_FATAL_ASSERT(runtime->spawning != 0);
  --runtime->spawning;
  runtime->changed.notify_all();
}

}  // namespace

void retain_runtime(ProcessRuntimeState* state) noexcept {
  state->refs.fetch_add(1, std::memory_order_relaxed);
}

void release_runtime(ProcessRuntimeState* state) noexcept {
  if (state == nullptr ||
      state->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) {
    return;
  }
  HYLE_FATAL_ASSERT(state->stopped);
  HYLE_FATAL_ASSERT(state->active.empty());
  HYLE_FATAL_ASSERT(!state->carrier_thread.joinable());
  const mem::AllocatorRef allocator = state->allocator;
  allocator.destroy(state);
}

void retain_process(ProcessState* state) noexcept {
  state->refs.fetch_add(1, std::memory_order_relaxed);
}

void release_process(ProcessState* state) noexcept {
  if (state == nullptr ||
      state->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) {
    return;
  }
  ProcessRuntimeState* runtime = state->runtime;
  const mem::AllocatorRef allocator = runtime->allocator;
  allocator.destroy(state);
  release_runtime(runtime);
}

void publish_exit(ProcessState* state, NativeExit result) noexcept {
  ProcessRuntimeState* runtime = state->runtime;
  {
    std::lock_guard<util::SpinLock> lock(runtime->monitor);
    if (state->exit_published) return;
    state->wait_status = std::move(result.status);
    state->exit.exit_code = result.exit_code;
    state->exit.termination_signal = result.signal;
    state->exit.kind = state->force_requested
                           ? ProcessExitKind::kForced
                           : (result.signaled ? ProcessExitKind::kSignaled
                                              : ProcessExitKind::kExited);
    state->exit_published = true;
    HYLE_FATAL_ASSERT(state->active_hook.linked());
    runtime->active.remove(state);
    HYLE_FATAL_ASSERT(runtime->active_count != 0);
    --runtime->active_count;
    state->completion.notify_all();
    runtime->changed.notify_all();
  }
  native_release_after_exit(state);
  release_process(state);  // registry/reaper reference
}

}  // namespace detail

ProcessRuntime::~ProcessRuntime() {
  if (state_ == nullptr) return;
  shutdown().IgnoreError();
  detail::release_runtime(std::exchange(state_, nullptr));
}

ProcessRuntime::ProcessRuntime(ProcessRuntime&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

ProcessRuntime& ProcessRuntime::operator=(ProcessRuntime&& other) noexcept {
  if (this == &other) return *this;
  if (state_ != nullptr) {
    shutdown().IgnoreError();
    detail::release_runtime(state_);
  }
  state_ = std::exchange(other.state_, nullptr);
  return *this;
}

Result<ProcessRuntime>
ProcessRuntime::create(mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid())
    return make_unexpected(
        Status::InvalidArgument("process runtime allocator is invalid"));
  auto* state = allocator.construct<detail::ProcessRuntimeState>(allocator);
  if (state == nullptr)
    return make_unexpected(
        Status::ResourceExhausted("process runtime allocation failed"));
  Status setup = detail::native_runtime_start(state);
  if (!setup.ok()) {
    state->stopped = true;
    allocator.destroy(state);
    return make_unexpected(std::move(setup));
  }
  platform::ThreadOpts thread_options;
  thread_options.name = "hyle-process";
  auto thread = platform::Thread::spawn(
      &detail::native_reaper_entry, state, thread_options);
  if (!thread.has_value()) {
    detail::native_runtime_close(state);
    state->stopped = true;
    allocator.destroy(state);
    return make_unexpected(
        std::move(thread).error().WithContext("start process reaper"));
  }
  state->carrier_thread = std::move(*thread);
  return ProcessRuntime(state);
}

bool ProcessRuntime::accepting() const noexcept {
  if (state_ == nullptr) return false;
  std::lock_guard<util::SpinLock> lock(state_->monitor);
  return state_->accepting;
}

std::size_t ProcessRuntime::active_processes() const noexcept {
  if (state_ == nullptr) return 0;
  std::lock_guard<util::SpinLock> lock(state_->monitor);
  return state_->active_count;
}

Result<ChildProcess>
ProcessRuntime::spawn(const ProcessStartOptions& options) noexcept {
  if (state_ == nullptr)
    return make_unexpected(
        Status::FailedPrecondition("process runtime is not valid"));
  Status validation = detail::validate_start_options(options);
  if (!validation.ok()) return make_unexpected(std::move(validation));

  {
    std::lock_guard<util::SpinLock> lock(state_->monitor);
    if (!state_->accepting)
      return make_unexpected(
          Status::FailedPrecondition("process runtime is shutting down"));
    ++state_->spawning;
  }

  detail::ProcessState* process =
      state_->allocator.construct<detail::ProcessState>(state_);
  if (process == nullptr) {
    detail::finish_spawn(state_);
    return make_unexpected(
        Status::ResourceExhausted("child process state allocation failed"));
  }
  detail::retain_runtime(state_);  // held by ProcessState until final release

  Status spawned = detail::native_spawn(state_, process, options);
  if (!spawned.ok()) {
    detail::finish_spawn(state_);
    detail::release_process(process);
    return make_unexpected(std::move(spawned));
  }

  Status registered;
  {
    std::lock_guard<util::SpinLock> lock(state_->monitor);
    detail::retain_process(process);  // runtime registry/reaper owner
    state_->active.push_back(process);
    ++state_->active_count;
    registered = detail::native_register(state_, process);
    if (!registered.ok()) {
      state_->active.remove(process);
      --state_->active_count;
    }
    HYLE_FATAL_ASSERT(state_->spawning != 0);
    --state_->spawning;
    state_->changed.notify_all();
  }
  if (!registered.ok()) {
    detail::native_abort_and_reap(process);
    detail::release_process(process);  // registry reference
    detail::release_process(process);  // prospective ChildProcess reference
    return make_unexpected(std::move(registered).WithContext(
        "register child process exit notification"));
  }
  return ChildProcess(process);
}

void ProcessRuntime::begin_shutdown() noexcept {
  if (state_ == nullptr) return;
  std::lock_guard<util::SpinLock> lock(state_->monitor);
  state_->accepting = false;
  state_->changed.notify_all();
}

Status ProcessRuntime::shutdown() noexcept {
  if (state_ == nullptr) return Status{};
  std::unique_lock<util::SpinLock> lock(state_->monitor);
  state_->accepting = false;
  if (state_->stopped) return Status{};
  if (state_->stopping) {
    state_->changed.wait(lock, [&] { return state_->stopped; });
    return Status{};
  }
  state_->stopping = true;
  state_->changed.wait(lock, [&] { return state_->spawning == 0; });

  Status result;
  state_->active.for_each([&](detail::ProcessState* process) noexcept {
    result.Update(detail::close_pipes_locked(process));
    if (!process->exit_published) {
      Status killed = detail::native_force_kill_tree(process);
      if (killed.ok()) process->force_requested = true;
      result.Update(std::move(killed));
    }
  });
  state_->changed.wait(lock, [&] { return state_->active_count == 0; });
  state_->carrier_stop.store(true, std::memory_order_release);
  detail::native_runtime_wake(state_);
  lock.unlock();

  state_->carrier_thread.join();
  detail::native_runtime_close(state_);

  lock.lock();
  state_->stopped = true;
  state_->changed.notify_all();
  return result;
}

ChildProcess::~ChildProcess() { reset_(true); }

ChildProcess::ChildProcess(ChildProcess&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

ChildProcess& ChildProcess::operator=(ChildProcess&& other) noexcept {
  if (this != &other) {
    reset_(true);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}

std::uint64_t ChildProcess::id() const noexcept {
  return state_ != nullptr ? state_->process_id : 0;
}

Result<std::optional<ProcessExit>> ChildProcess::try_wait() const noexcept {
  if (state_ == nullptr)
    return make_unexpected(
        Status::FailedPrecondition("child process is not valid"));
  std::lock_guard<util::SpinLock> lock(state_->runtime->monitor);
  if (!state_->exit_published) return std::optional<ProcessExit>{};
  if (!state_->wait_status.ok())
    return make_unexpected(state_->wait_status);
  return std::optional<ProcessExit>{state_->exit};
}

Result<ProcessExit>
ChildProcess::wait(ProcessWaitOptions options) const noexcept {
  if (state_ == nullptr)
    return make_unexpected(
        Status::FailedPrecondition("child process is not valid"));
  if (options.cancel != nullptr && !options.cancel->valid())
    return make_unexpected(
        Status::InvalidArgument("process wait cancel token is invalid"));

  detail::CancelWake wake{state_};
  if (options.cancel != nullptr) {
    sched::CancelCallback callback(
        *options.cancel, &detail::CancelWake::notify, &wake);
    std::unique_lock<util::SpinLock> lock(state_->runtime->monitor);
    return detail::wait_locked(state_, options, lock);
  }
  std::unique_lock<util::SpinLock> lock(state_->runtime->monitor);
  return detail::wait_locked(state_, options, lock);
}

io::LocalStream* ChildProcess::stdin_pipe() noexcept {
  return state_ != nullptr && state_->stdin_stream.valid()
             ? &state_->stdin_stream
             : nullptr;
}

io::LocalStream* ChildProcess::stdout_pipe() noexcept {
  return state_ != nullptr && state_->stdout_stream.valid()
             ? &state_->stdout_stream
             : nullptr;
}

io::LocalStream* ChildProcess::stderr_pipe() noexcept {
  return state_ != nullptr && state_->stderr_stream.valid()
             ? &state_->stderr_stream
             : nullptr;
}

Status ChildProcess::close_stdin() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("child process is not valid");
  std::lock_guard<util::SpinLock> lock(state_->runtime->monitor);
  if (!state_->stdin_stream.valid())
    return Status::FailedPrecondition("child process has no open stdin pipe");
  return state_->stdin_stream.close();
}

Status ChildProcess::request_termination() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("child process is not valid");
  std::lock_guard<util::SpinLock> lock(state_->runtime->monitor);
  if (state_->exit_published) return Status{};
  return detail::native_request_termination(state_);
}

Status ChildProcess::force_kill_tree() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("child process is not valid");
  std::lock_guard<util::SpinLock> lock(state_->runtime->monitor);
  if (state_->exit_published) return Status{};
  Status status = detail::native_force_kill_tree(state_);
  if (status.ok()) state_->force_requested = true;
  return status;
}

Status ChildProcess::release_to_reaper() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("child process is not valid");
  detail::ProcessState* state = std::exchange(state_, nullptr);
  Status status;
  {
    std::lock_guard<util::SpinLock> lock(state->runtime->monitor);
    status = detail::close_pipes_locked(state);
  }
  detail::release_process(state);
  return status;
}

void ChildProcess::reset_(bool terminate) noexcept {
  detail::ProcessState* state = std::exchange(state_, nullptr);
  if (state == nullptr) return;
  {
    std::lock_guard<util::SpinLock> lock(state->runtime->monitor);
    detail::close_pipes_locked(state).IgnoreError();
    if (terminate && !state->exit_published) {
      Status status = detail::native_force_kill_tree(state);
      if (status.ok()) state->force_requested = true;
      status.IgnoreError();
    }
  }
  detail::release_process(state);
}

}  // namespace hyle::process
