#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif

#include "process_internal.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <spawn.h>
#include <string_view>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "hyle/container/small_vector.h"
#include "hyle/core/environment.h"
#include "hyle/error/system_error.h"

#if HYLE_OS_LINUX
#  include <sys/epoll.h>
#  include <sys/eventfd.h>
#  include <sys/syscall.h>
#elif HYLE_OS_MACOS
#  include <sys/event.h>
#  include <sys/time.h>
#endif

extern char** environ;

namespace hyle::process::detail {
namespace {

template <class T, std::size_t Inline>
using Scratch = container::SmallVector<T, Inline, mem::AllocatorRef>;

Status errno_status(int error, const char* operation) noexcept {
  return status_from_errno(error, operation);
}

bool append_bytes(Scratch<char, 512>& output, std::string_view text,
                  bool nul = false) noexcept {
  const std::size_t extra = text.size() + (nul ? 1u : 0u);
  if (extra > SIZE_MAX - output.size() ||
      !output.try_reserve(output.size() + extra)) {
    return false;
  }
  for (char byte : text) output.push_back(byte);
  if (nul) output.push_back('\0');
  return true;
}

struct PackedStrings {
  explicit PackedStrings(mem::AllocatorRef allocator) noexcept
      : bytes(allocator), offsets(allocator), values(allocator) {}

  bool add(std::string_view text) noexcept {
    if (!offsets.try_push_back(bytes.size())) return false;
    return append_bytes(bytes, text, true);
  }

  bool finish() noexcept {
    if (!values.try_reserve(offsets.size() + 1)) return false;
    for (std::size_t offset : offsets) values.push_back(bytes.data() + offset);
    values.push_back(nullptr);
    return true;
  }

  Scratch<char, 512> bytes;
  Scratch<std::size_t, 16> offsets;
  Scratch<char*, 17> values;
};

bool environment_name_matches(const char* entry,
                              std::string_view name) noexcept {
  const char* separator = std::strchr(entry, '=');
  if (separator == nullptr ||
      static_cast<std::size_t>(separator - entry) != name.size()) {
    return false;
  }
  return std::memcmp(entry, name.data(), name.size()) == 0;
}

bool environment_entry_replaced(
    const char* entry,
    std::span<const EnvironmentChange> changes) noexcept {
  for (const EnvironmentChange& change : changes) {
    if (environment_name_matches(entry, change.name)) return true;
  }
  return false;
}

Result<PackedStrings> build_environment(
    mem::AllocatorRef allocator, const ProcessStartOptions& options) noexcept {
  PackedStrings result(allocator);
  if (options.environment_mode == EnvironmentMode::kInherit) {
    for (char** current = environ; current != nullptr && *current != nullptr;
         ++current) {
      if (!environment_entry_replaced(*current, options.environment_changes) &&
          !result.add(*current)) {
        return make_unexpected(Status::ResourceExhausted(
            "process environment allocation failed"));
      }
    }
  }
  for (const EnvironmentChange& change : options.environment_changes) {
    if (!change.value.has_value()) continue;
    if (!result.offsets.try_push_back(result.bytes.size()) ||
        !append_bytes(result.bytes, change.name) ||
        !append_bytes(result.bytes, "=") ||
        !append_bytes(result.bytes, *change.value, true)) {
      return make_unexpected(Status::ResourceExhausted(
          "process environment allocation failed"));
    }
  }
  return result;
}

Result<PackedStrings> build_arguments(
    mem::AllocatorRef allocator, const ProcessStartOptions& options) noexcept {
  PackedStrings result(allocator);
  if (!result.add(options.executable))
    return make_unexpected(
        Status::ResourceExhausted("process argv allocation failed"));
  for (std::string_view argument : options.arguments) {
    if (!result.add(argument))
      return make_unexpected(
          Status::ResourceExhausted("process argv allocation failed"));
  }
  return result;
}

Result<Scratch<char, 512>> copy_c_string(mem::AllocatorRef allocator,
                                         std::string_view text) noexcept {
  Scratch<char, 512> result(allocator);
  if (!append_bytes(result, text, true))
    return make_unexpected(
        Status::ResourceExhausted("process path allocation failed"));
  return result;
}

int move_above_stdio(int descriptor) noexcept {
  if (descriptor < 0 || descriptor > STDERR_FILENO) return descriptor;
#if defined(F_DUPFD_CLOEXEC)
  const int moved = ::fcntl(descriptor, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
#else
  const int moved = ::fcntl(descriptor, F_DUPFD, STDERR_FILENO + 1);
  if (moved >= 0) (void)::fcntl(moved, F_SETFD, FD_CLOEXEC);
#endif
  const int saved = errno;
  (void)::close(descriptor);
  errno = saved;
  return moved;
}

Status set_cloexec(int descriptor) noexcept {
  const int flags = ::fcntl(descriptor, F_GETFD, 0);
  if (flags < 0 || ::fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0)
    return errno_status(errno, "set child stdio close-on-exec");
  return Status{};
}

Status set_nonblocking(int descriptor) noexcept {
  const int flags = ::fcntl(descriptor, F_GETFL, 0);
  if (flags < 0 || ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0)
    return errno_status(errno, "set parent process pipe nonblocking");
  return Status{};
}

struct StdioSetup {
  explicit StdioSetup(ProcessState* value) noexcept : state(value) {}
  ~StdioSetup() {
    for (std::size_t i = 0; i < temporary_count; ++i)
      if (temporary[i] >= 0) (void)::close(temporary[i]);
  }

  bool remember(int descriptor) noexcept {
    for (std::size_t i = 0; i < temporary_count; ++i)
      if (temporary[i] == descriptor) return true;
    if (temporary_count == std::size(temporary)) return false;
    temporary[temporary_count++] = descriptor;
    return true;
  }

  Status make_pipe(int slot, const ProcessStartOptions& options) noexcept {
    int pair[2] = {-1, -1};
#if HYLE_OS_LINUX && defined(SOCK_CLOEXEC)
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) != 0)
#else
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0)
#endif
      return errno_status(errno, "create child stdio socketpair");
    pair[0] = move_above_stdio(pair[0]);
    if (pair[0] < 0) {
      const int error = errno;
      (void)::close(pair[1]);
      return errno_status(error, "normalize parent process pipe descriptor");
    }
    pair[1] = move_above_stdio(pair[1]);
    if (pair[1] < 0) {
      const int error = errno;
      (void)::close(pair[0]);
      return errno_status(error, "normalize child process pipe descriptor");
    }
    Status status = set_cloexec(pair[0]);
    if (status.ok()) status = set_cloexec(pair[1]);
    if (status.ok()) status = set_nonblocking(pair[0]);
    if (!status.ok()) {
      (void)::close(pair[0]);
      (void)::close(pair[1]);
      return status;
    }

    auto stream = io::LocalStream::adopt_anonymous(
        *options.pipe_io, options.pipe_owner, pair[0]);
    pair[0] = -1;  // adopt owns success and failure
    if (!stream.has_value()) {
      (void)::close(pair[1]);
      return std::move(stream).error();
    }
    if (!remember(pair[1])) {
      (void)::close(pair[1]);
      return Status::ResourceExhausted("too many child stdio descriptors");
    }
    source[slot] = pair[1];
    if (slot == STDIN_FILENO)
      state->stdin_stream = std::move(*stream);
    else if (slot == STDOUT_FILENO)
      state->stdout_stream = std::move(*stream);
    else
      state->stderr_stream = std::move(*stream);
    return Status{};
  }

  Status prepare_slot(int slot, ProcessStdio config,
                      const ProcessStartOptions& options) noexcept {
    switch (config.kind()) {
      case ProcessStdioKind::kInherit:
        source[slot] = slot;
        return Status{};
      case ProcessStdioKind::kNull: {
        int descriptor = ::open("/dev/null",
                                slot == STDIN_FILENO ? O_RDONLY : O_WRONLY);
        if (descriptor < 0)
          return errno_status(errno, "open /dev/null for child stdio");
        descriptor = move_above_stdio(descriptor);
        if (descriptor < 0)
          return errno_status(errno, "normalize child null descriptor");
        Status status = set_cloexec(descriptor);
        if (!status.ok()) {
          (void)::close(descriptor);
          return status;
        }
        if (!remember(descriptor)) {
          (void)::close(descriptor);
          return Status::ResourceExhausted("too many child stdio descriptors");
        }
        source[slot] = descriptor;
        return Status{};
      }
      case ProcessStdioKind::kFile: {
        const std::intptr_t raw = config.file_handle();
        if (raw < 0 || raw > INT_MAX)
          return Status::OutOfRange("child stdio file descriptor is invalid");
        source[slot] = static_cast<int>(raw);
        return Status{};
      }
      case ProcessStdioKind::kPipe:
        return make_pipe(slot, options);
      case ProcessStdioKind::kMergeStdout:
        source[slot] = source[STDOUT_FILENO];
        return Status{};
    }
    return Status::InvalidArgument("process stdio kind is invalid");
  }

  Status prepare(const ProcessStartOptions& options) noexcept {
    Status status = prepare_slot(STDIN_FILENO, options.stdin_stream, options);
    if (!status.ok()) return status;
    status = prepare_slot(STDOUT_FILENO, options.stdout_stream, options);
    if (!status.ok()) return status;
    return prepare_slot(STDERR_FILENO, options.stderr_stream, options);
  }

  Status add_actions(posix_spawn_file_actions_t* actions) noexcept {
    for (int slot = STDIN_FILENO; slot <= STDERR_FILENO; ++slot) {
      if (source[slot] == slot) {
#if HYLE_OS_MACOS
        const int inherit_rc =
            ::posix_spawn_file_actions_addinherit_np(actions, slot);
        if (inherit_rc != 0)
          return errno_status(inherit_rc,
                              "preserve inherited child stdio");
#endif
        continue;
      }
      const int rc = ::posix_spawn_file_actions_adddup2(
          actions, source[slot], slot);
      if (rc != 0) return errno_status(rc, "configure child stdio dup2");
    }
    for (std::size_t i = 0; i < temporary_count; ++i) {
      const int rc =
          ::posix_spawn_file_actions_addclose(actions, temporary[i]);
      if (rc != 0)
        return errno_status(rc, "configure child stdio close");
    }
#if HYLE_OS_LINUX
    const int close_from_rc =
        ::posix_spawn_file_actions_addclosefrom_np(actions, STDERR_FILENO + 1);
    if (close_from_rc != 0)
      return errno_status(close_from_rc,
                          "close unrelated child descriptors");
#endif
    return Status{};
  }

  ProcessState* state;
  int source[3] = {STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO};
  int temporary[6] = {-1, -1, -1, -1, -1, -1};
  std::size_t temporary_count = 0;
};

NativeExit collect_exit(ProcessState* state) noexcept {
  // The root is still a zombie here, so its PID/process-group identifier cannot
  // be reused. Close the managed tree while that identity is pinned, then reap
  // the exact root. Killing after waitpid would create a tiny opportunity to
  // signal an unrelated, newly reused process group.
  if (state->posix_pid > 0)
    (void)::kill(-static_cast<pid_t>(state->posix_pid), SIGKILL);
  int native_status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(static_cast<pid_t>(state->posix_pid), &native_status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0) {
    return NativeExit{errno_status(errno, "waitpid managed child")};
  }
  if (WIFEXITED(native_status)) {
    NativeExit result;
    result.exit_code = static_cast<std::uint32_t>(WEXITSTATUS(native_status));
    return result;
  }
  if (WIFSIGNALED(native_status)) {
    NativeExit result;
    result.signal = WTERMSIG(native_status);
    result.signaled = true;
    return result;
  }
  return NativeExit{Status::Internal(
      "managed child produced an unsupported wait status")};
}

}  // namespace

Status native_runtime_start(ProcessRuntimeState* state) noexcept {
#if HYLE_OS_LINUX
  const int carrier = ::epoll_create1(EPOLL_CLOEXEC);
  if (carrier < 0) return errno_status(errno, "create process epoll carrier");
  const int wake = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (wake < 0) {
    const int error = errno;
    (void)::close(carrier);
    return errno_status(error, "create process carrier wake event");
  }
  epoll_event event{};
  event.events = EPOLLIN;
  event.data.ptr = nullptr;
  if (::epoll_ctl(carrier, EPOLL_CTL_ADD, wake, &event) != 0) {
    const int error = errno;
    (void)::close(wake);
    (void)::close(carrier);
    return errno_status(error, "register process carrier wake event");
  }
  state->native_carrier = carrier;
  state->native_wake = wake;
#elif HYLE_OS_MACOS
  const int carrier = ::kqueue();
  if (carrier < 0) return errno_status(errno, "create process kqueue carrier");
  struct kevent change;
  EV_SET(&change, 1, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0, nullptr);
  if (::kevent(carrier, &change, 1, nullptr, 0, nullptr) != 0) {
    const int error = errno;
    (void)::close(carrier);
    return errno_status(error, "register process kqueue wake event");
  }
  state->native_carrier = carrier;
#endif
  return Status{};
}

void native_runtime_wake(ProcessRuntimeState* state) noexcept {
#if HYLE_OS_LINUX
  const std::uint64_t one = 1;
  const ssize_t written =
      ::write(static_cast<int>(state->native_wake), &one, sizeof(one));
  (void)written;
#elif HYLE_OS_MACOS
  struct kevent change;
  EV_SET(&change, 1, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
  (void)::kevent(static_cast<int>(state->native_carrier), &change, 1,
                 nullptr, 0, nullptr);
#endif
}

void native_runtime_close(ProcessRuntimeState* state) noexcept {
#if HYLE_OS_LINUX
  if (state->native_wake >= 0)
    (void)::close(static_cast<int>(std::exchange(state->native_wake, -1)));
#endif
  if (state->native_carrier >= 0)
    (void)::close(static_cast<int>(std::exchange(state->native_carrier, -1)));
}

void native_reaper_entry(void* raw) noexcept {
  auto* runtime = static_cast<ProcessRuntimeState*>(raw);
#if HYLE_OS_LINUX
  epoll_event events[32]{};
  for (;;) {
    const int count = ::epoll_wait(static_cast<int>(runtime->native_carrier),
                                   events, static_cast<int>(std::size(events)),
                                   -1);
    if (count < 0) {
      if (errno == EINTR) continue;
      HYLE_FATAL_ASSERT_MSG(false, "process epoll carrier failed");
    }
    for (int i = 0; i < count; ++i) {
      auto* state = static_cast<ProcessState*>(events[i].data.ptr);
      if (state == nullptr) {
        std::uint64_t value = 0;
        while (::read(static_cast<int>(runtime->native_wake), &value,
                      sizeof(value)) < 0 && errno == EINTR) {}
      } else {
        publish_exit(state, collect_exit(state));
      }
    }
    if (runtime->carrier_stop.load(std::memory_order_acquire)) return;
  }
#elif HYLE_OS_MACOS
  struct kevent events[32]{};
  for (;;) {
    const int count = ::kevent(static_cast<int>(runtime->native_carrier),
                               nullptr, 0, events,
                               static_cast<int>(std::size(events)), nullptr);
    if (count < 0) {
      if (errno == EINTR) continue;
      HYLE_FATAL_ASSERT_MSG(false, "process kqueue carrier failed");
    }
    for (int i = 0; i < count; ++i) {
      auto* state = static_cast<ProcessState*>(events[i].udata);
      if (state != nullptr) publish_exit(state, collect_exit(state));
    }
    if (runtime->carrier_stop.load(std::memory_order_acquire)) return;
  }
#endif
}

Status native_spawn(ProcessRuntimeState* runtime, ProcessState* state,
                    const ProcessStartOptions& options) noexcept {
  auto arguments = build_arguments(runtime->allocator, options);
  if (!arguments.has_value()) return std::move(arguments).error();
  if (!arguments->finish())
    return Status::ResourceExhausted(
        "process argv pointer allocation failed");

  PackedStrings environment(runtime->allocator);
  char** environment_pointer = environ;
  if (options.environment_mode == EnvironmentMode::kReplace ||
      !options.environment_changes.empty()) {
    auto built = build_environment(runtime->allocator, options);
    if (!built.has_value()) return std::move(built).error();
    environment = std::move(*built);
    if (!environment.finish())
      return Status::ResourceExhausted(
          "process environment pointer allocation failed");
    environment_pointer = environment.values.data();
  }

  StdioSetup stdio(state);
  Status status = stdio.prepare(options);
  if (!status.ok()) return status;

  posix_spawn_file_actions_t actions;
  int rc = ::posix_spawn_file_actions_init(&actions);
  if (rc != 0) return errno_status(rc, "initialize process spawn actions");
  status = stdio.add_actions(&actions);
  Scratch<char, 512> working_directory(runtime->allocator);
  if (status.ok() && options.working_directory.has_value()) {
    auto copied = copy_c_string(runtime->allocator,
                                *options.working_directory);
    if (!copied.has_value()) {
      status = std::move(copied).error();
    } else {
      working_directory = std::move(*copied);
      rc = ::posix_spawn_file_actions_addchdir_np(
          &actions, working_directory.data());
      if (rc != 0)
        status = errno_status(rc, "configure child working directory");
    }
  }
  if (!status.ok()) {
    (void)::posix_spawn_file_actions_destroy(&actions);
    return status;
  }

  posix_spawnattr_t attributes;
  rc = ::posix_spawnattr_init(&attributes);
  if (rc != 0) {
    (void)::posix_spawn_file_actions_destroy(&actions);
    return errno_status(rc, "initialize process spawn attributes");
  }
  short flags = POSIX_SPAWN_SETPGROUP;
#if HYLE_OS_MACOS
  flags = static_cast<short>(flags | POSIX_SPAWN_CLOEXEC_DEFAULT);
#endif
  rc = ::posix_spawnattr_setflags(&attributes, flags);
  if (rc == 0) rc = ::posix_spawnattr_setpgroup(&attributes, 0);
  if (rc != 0) {
    (void)::posix_spawnattr_destroy(&attributes);
    (void)::posix_spawn_file_actions_destroy(&actions);
    return errno_status(rc, "configure child process group");
  }

  pid_t pid = -1;
  if (options.executable_search == ExecutableSearch::kPath) {
    rc = ::posix_spawnp(&pid, arguments->values[0], &actions, &attributes,
                        arguments->values.data(), environment_pointer);
  } else {
    rc = ::posix_spawn(&pid, arguments->values[0], &actions, &attributes,
                       arguments->values.data(), environment_pointer);
  }
  (void)::posix_spawnattr_destroy(&attributes);
  (void)::posix_spawn_file_actions_destroy(&actions);
  if (rc != 0) return errno_status(rc, "spawn managed child process");

  state->posix_pid = static_cast<int>(pid);
  state->process_id = static_cast<std::uint64_t>(pid);
#if HYLE_OS_LINUX
  const int pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
  if (pidfd < 0) {
    const int error = errno;
    (void)::kill(-pid, SIGKILL);
    while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    state->posix_pid = -1;
    return error == ENOSYS
               ? Status::Unimplemented(
                     "managed process requires Linux pidfd support")
               : errno_status(error, "open managed child pidfd");
  }
  state->native_exit = pidfd;
#endif
  return Status{};
}

Status native_register(ProcessRuntimeState* runtime,
                       ProcessState* state) noexcept {
#if HYLE_OS_LINUX
  epoll_event event{};
  event.events = EPOLLIN | EPOLLONESHOT;
  event.data.ptr = state;
  if (::epoll_ctl(static_cast<int>(runtime->native_carrier), EPOLL_CTL_ADD,
                  static_cast<int>(state->native_exit), &event) != 0) {
    return errno_status(errno, "register managed child pidfd");
  }
#elif HYLE_OS_MACOS
  struct kevent change;
  EV_SET(&change, static_cast<uintptr_t>(state->posix_pid), EVFILT_PROC,
         EV_ADD | EV_ENABLE | EV_ONESHOT, NOTE_EXIT, 0, state);
  if (::kevent(static_cast<int>(runtime->native_carrier), &change, 1,
               nullptr, 0, nullptr) != 0) {
    return errno_status(errno, "register managed child process filter");
  }
#endif
  return Status{};
}

void native_abort_and_reap(ProcessState* state) noexcept {
  if (state->posix_pid > 0) {
    (void)::kill(-static_cast<pid_t>(state->posix_pid), SIGKILL);
    while (::waitpid(static_cast<pid_t>(state->posix_pid), nullptr, 0) < 0 &&
           errno == EINTR) {}
    state->posix_pid = -1;
  }
  if (state->native_exit >= 0)
    (void)::close(static_cast<int>(std::exchange(state->native_exit, -1)));
}

Status native_request_termination(ProcessState* state) noexcept {
  if (::kill(-static_cast<pid_t>(state->posix_pid), SIGTERM) == 0 ||
      errno == ESRCH) {
    return Status{};
  }
  return errno_status(errno, "request managed process-group termination");
}

Status native_force_kill_tree(ProcessState* state) noexcept {
  if (::kill(-static_cast<pid_t>(state->posix_pid), SIGKILL) == 0 ||
      errno == ESRCH) {
    return Status{};
  }
  return errno_status(errno, "force kill managed process group");
}

void native_release_after_exit(ProcessState* state) noexcept {
  // collect_exit closed the process group before reaping the root, while its
  // PID still pinned the group identity.
  if (state->native_exit >= 0)
    (void)::close(static_cast<int>(std::exchange(state->native_exit, -1)));
  state->posix_pid = -1;
}

}  // namespace hyle::process::detail
