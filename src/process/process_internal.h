#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "hyle/container/intrusive_list.h"
#include "hyle/process/process.h"
#include "hyle/platform/thread.h"
#include "hyle/sched/sync.h"
#include "hyle/util/spin_lock.h"

namespace hyle::process::detail {

struct ProcessRuntimeState;

struct ProcessState {
  explicit ProcessState(ProcessRuntimeState* owner) noexcept : runtime(owner) {}

  std::atomic<std::size_t> refs{1};  // initial public ChildProcess owner
  ProcessRuntimeState* runtime = nullptr;
  container::SafeIntrusiveListHook active_hook;
  sched::ConditionVariable completion;

  io::LocalStream stdin_stream;
  io::LocalStream stdout_stream;
  io::LocalStream stderr_stream;

  std::uint64_t process_id = 0;
  std::intptr_t native_process = -1;
  std::intptr_t native_tree = -1;
  std::intptr_t native_wait = -1;
  std::intptr_t native_exit = -1;
  int posix_pid = -1;

  Status wait_status{};
  ProcessExit exit{};
  bool exit_published = false;
  bool force_requested = false;
};

using ActiveProcesses = container::SafeIntrusiveList<
    ProcessState, &ProcessState::active_hook>;

struct ProcessRuntimeState {
  explicit ProcessRuntimeState(mem::AllocatorRef value) noexcept
      : allocator(value) {}

  std::atomic<std::size_t> refs{1};  // public ProcessRuntime owner
  mem::AllocatorRef allocator;
  mutable util::SpinLock monitor;
  sched::ConditionVariable changed;
  ActiveProcesses active;
  platform::Thread carrier_thread;

  std::intptr_t native_carrier = -1;
  std::intptr_t native_wake = -1;
  std::size_t active_count = 0;
  std::size_t spawning = 0;
  bool accepting = true;
  bool stopping = false;
  bool stopped = false;
  std::atomic<bool> carrier_stop{false};
};

struct NativeExit {
  Status status{};
  std::uint32_t exit_code = 0;
  int signal = 0;
  bool signaled = false;
};

void retain_runtime(ProcessRuntimeState* state) noexcept;
void release_runtime(ProcessRuntimeState* state) noexcept;
void retain_process(ProcessState* state) noexcept;
void release_process(ProcessState* state) noexcept;

void publish_exit(ProcessState* state, NativeExit result) noexcept;

Status native_runtime_start(ProcessRuntimeState* state) noexcept;
void native_runtime_wake(ProcessRuntimeState* state) noexcept;
void native_runtime_close(ProcessRuntimeState* state) noexcept;
void native_reaper_entry(void* raw) noexcept;

Status native_spawn(ProcessRuntimeState* runtime, ProcessState* state,
                    const ProcessStartOptions& options) noexcept;
Status native_register(ProcessRuntimeState* runtime,
                       ProcessState* state) noexcept;
void native_abort_and_reap(ProcessState* state) noexcept;
Status native_request_termination(ProcessState* state) noexcept;
Status native_force_kill_tree(ProcessState* state) noexcept;
void native_release_after_exit(ProcessState* state) noexcept;

}  // namespace hyle::process::detail
