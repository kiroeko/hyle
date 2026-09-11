#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "hyle/error/result.h"
#include "hyle/io/local.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/cancel.h"
#include "hyle/time/clock.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PROCESS_API
#elif defined(HYLE_PROCESS_BUILD)
#  define HYLE_PROCESS_API HYLE_EXPORT
#else
#  define HYLE_PROCESS_API HYLE_IMPORT
#endif

namespace hyle::process {

namespace detail {
struct ProcessRuntimeState;
struct ProcessState;
}

// Executable path lookup is never implicit. kExact passes the named executable
// directly to the OS; kPath asks the host's PATH search mechanism to resolve a
// basename. Neither route invokes a shell.
enum class ExecutableSearch : std::uint8_t {
  kExact,
  kPath,
};

enum class EnvironmentMode : std::uint8_t {
  kInherit,
  kReplace,
};

// A borrowed environment edit. value==nullopt removes the variable in inherit
// mode and is rejected in replacement mode. Names and values are copied before
// spawn returns and therefore need not outlive that call.
struct EnvironmentChange {
  std::string_view name;
  std::optional<std::string_view> value;
};

enum class ProcessStdioKind : std::uint8_t {
  kInherit,
  kNull,
  kFile,
  kPipe,
  kMergeStdout,
};

// One child-side stdio binding. file() snapshots the native value; that native
// file must remain open until spawn returns. The child receives an independent
// duplicate/dup2 and the caller keeps ownership. merge_stdout() is valid only
// for stderr.
class ProcessStdio {
 public:
  constexpr ProcessStdio() noexcept = default;
  static constexpr ProcessStdio inherit() noexcept { return {}; }
  static constexpr ProcessStdio null() noexcept {
    return ProcessStdio(ProcessStdioKind::kNull,
                        platform::FileHandle::kInvalid);
  }
  static constexpr ProcessStdio pipe() noexcept {
    return ProcessStdio(ProcessStdioKind::kPipe,
                        platform::FileHandle::kInvalid);
  }
  static constexpr ProcessStdio file(
      const platform::FileHandle& handle) noexcept {
    return ProcessStdio(ProcessStdioKind::kFile, handle.get());
  }
  static constexpr ProcessStdio merge_stdout() noexcept {
    return ProcessStdio(ProcessStdioKind::kMergeStdout,
                        platform::FileHandle::kInvalid);
  }

  constexpr ProcessStdioKind kind() const noexcept { return kind_; }
  constexpr std::intptr_t file_handle() const noexcept {
    return file_;
  }

 private:
  constexpr ProcessStdio(ProcessStdioKind kind,
                         std::intptr_t file) noexcept
      : kind_(kind), file_(file) {}

  ProcessStdioKind kind_ = ProcessStdioKind::kInherit;
  std::intptr_t file_ = platform::FileHandle::kInvalid;
};

struct ProcessStartOptions {
  // arguments excludes argv[0]; HYLE inserts executable as argv[0]. Empty
  // argument values are preserved. All views are borrowed through spawn only.
  std::string_view executable;
  std::span<const std::string_view> arguments;
  ExecutableSearch executable_search = ExecutableSearch::kExact;
  std::optional<std::string_view> working_directory;
  EnvironmentMode environment_mode = EnvironmentMode::kInherit;
  std::span<const EnvironmentChange> environment_changes;

  ProcessStdio stdin_stream = ProcessStdio::inherit();
  ProcessStdio stdout_stream = ProcessStdio::inherit();
  ProcessStdio stderr_stream = ProcessStdio::inherit();

  // Required only when at least one stdio binding is pipe(). The created parent
  // endpoint is bound to this shard and follows LocalStream's normal rule that
  // operations run on its owner worker.
  io::IoContext* pipe_io = nullptr;
  unsigned pipe_owner = 0;
};

struct ProcessWaitOptions {
  std::optional<time::MonotonicTimePoint> deadline;
  const sched::CancelToken* cancel = nullptr;
};

enum class ProcessExitKind : std::uint8_t {
  kExited,
  kSignaled,
  kForced,
};

struct ProcessExit {
  ProcessExitKind kind = ProcessExitKind::kExited;
  std::uint32_t exit_code = 0;
  int termination_signal = 0;
};

class ChildProcess;

// Explicit admission/reap owner. One runtime owns one native, non-polling exit
// carrier shared by every child. The object is thread-safe. shutdown() rejects
// new spawn, force-terminates remaining trees and reaps them; destruction calls
// the same idempotent path. The injected allocator must outlive this object and
// every ChildProcess produced by it.
class ProcessRuntime {
 public:
  ProcessRuntime() noexcept = default;
  HYLE_PROCESS_API ~ProcessRuntime();
  HYLE_PROCESS_API ProcessRuntime(ProcessRuntime&& other) noexcept;
  HYLE_PROCESS_API ProcessRuntime& operator=(ProcessRuntime&& other) noexcept;
  ProcessRuntime(const ProcessRuntime&) = delete;
  ProcessRuntime& operator=(const ProcessRuntime&) = delete;

  HYLE_PROCESS_API static Result<ProcessRuntime> create(
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_PROCESS_API bool accepting() const noexcept;
  HYLE_PROCESS_API std::size_t active_processes() const noexcept;

  HYLE_PROCESS_API Result<ChildProcess>
  spawn(const ProcessStartOptions& options) noexcept;

  HYLE_PROCESS_API void begin_shutdown() noexcept;
  HYLE_PROCESS_API Status shutdown() noexcept;

 private:
  explicit ProcessRuntime(detail::ProcessRuntimeState* state) noexcept
      : state_(state) {}

  detail::ProcessRuntimeState* state_ = nullptr;
};

// Move-only operation owner for one child tree. Status/wait/termination methods
// are thread-safe; each returned LocalStream retains its own single-read/
// single-write in-flight contract and must be quiescent before runtime shutdown.
// Dropping a live child closes parent pipes, force-terminates its tree and hands
// reap to ProcessRuntime without blocking the destructor.
class ChildProcess {
 public:
  ChildProcess() noexcept = default;
  HYLE_PROCESS_API ~ChildProcess();
  HYLE_PROCESS_API ChildProcess(ChildProcess&& other) noexcept;
  HYLE_PROCESS_API ChildProcess& operator=(ChildProcess&& other) noexcept;
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_PROCESS_API std::uint64_t id() const noexcept;

  HYLE_PROCESS_API Result<std::optional<ProcessExit>> try_wait() const noexcept;
  // Works from a HYLE fiber (suspends it) or an ordinary thread (parks it).
  // Cancellation/deadline ends only this wait; the child keeps running and may
  // be waited again.
  HYLE_PROCESS_API Result<ProcessExit>
  wait(ProcessWaitOptions options = {}) const noexcept;

  HYLE_PROCESS_API io::LocalStream* stdin_pipe() noexcept;
  HYLE_PROCESS_API io::LocalStream* stdout_pipe() noexcept;
  HYLE_PROCESS_API io::LocalStream* stderr_pipe() noexcept;
  HYLE_PROCESS_API Status close_stdin() noexcept;

  // POSIX sends SIGTERM to the process group. Windows has no portable
  // cooperative equivalent and reports Unimplemented; use stdin/application
  // protocol there. This never masquerades force termination as graceful.
  HYLE_PROCESS_API Status request_termination() noexcept;
  HYLE_PROCESS_API Status force_kill_tree() noexcept;

  // Close parent pipes and invalidate this owner while allowing the child to
  // continue. ProcessRuntime still observes and reaps it, and shutdown still
  // terminates it. This is not an unmanaged native detach.
  HYLE_PROCESS_API Status release_to_reaper() noexcept;

 private:
  friend class ProcessRuntime;
  explicit ChildProcess(detail::ProcessState* state) noexcept : state_(state) {}
  HYLE_PROCESS_API void reset_(bool terminate) noexcept;

  detail::ProcessState* state_ = nullptr;
};

}  // namespace hyle::process
