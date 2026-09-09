#include "watch_internal.h"

#include "hyle/core/environment.h"

#if HYLE_OS_LINUX

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sys/inotify.h>
#include <unistd.h>

#include "hyle/error/system_error.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/detail/net_cancel.h"
#include "hyle/io/detail/proactor.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/coro.h"

namespace hyle::io::detail {
namespace {

struct InotifyReadAwaitable {
  FileWatcherState* state = nullptr;
  Proactor* proactor = nullptr;
  bool* first = nullptr;
  IoOperation operation{};
  CancelOpCtx cancel{};

  bool await_ready() noexcept {
    if (state->native_cancel.is_cancelled()) {
      operation.result = kCancelled;
      return true;
    }
    return false;
  }

  bool await_suspend(std::coroutine_handle<>) noexcept {
    operation.waker = sched::current_waker();
    cancel.p = proactor;
    cancel.token = to_token(&operation);
    const auto bytes = static_cast<std::uint32_t>(
        state->shared_native_buffer.size() * sizeof(std::uint32_t));
    if (!proactor->submit_read(state->native_stream,
                               state->shared_native_buffer.data(), bytes, 0,
                               cancel.token)) {
      operation.result = kSubmitFailed;
      return false;
    }
    if (*first) {
      *first = false;
      state->report_armed();
    }
    sched::CancelCallback callback(state->native_cancel,
                                   &CancelOpCtx::on_cancel, &cancel);
    sched::suspend();
    return false;
  }

  std::int64_t await_resume() noexcept {
    if (operation.result < 0 && state->native_cancel.is_cancelled())
      return kCancelled;
    return operation.result;
  }
};

FileWatchChange translate_mask(std::uint32_t mask) noexcept {
  FileWatchChange changes = FileWatchChange::kNone;
  if ((mask & (IN_MODIFY | IN_CLOSE_WRITE)) != 0)
    changes |= FileWatchChange::kContent;
  if ((mask & IN_ATTRIB) != 0) changes |= FileWatchChange::kMetadata;
  if ((mask & IN_CREATE) != 0) changes |= FileWatchChange::kCreated;
  if ((mask & IN_DELETE) != 0) changes |= FileWatchChange::kRemoved;
  if ((mask & IN_MOVED_FROM) != 0) changes |= FileWatchChange::kMovedFrom;
  if ((mask & IN_MOVED_TO) != 0) changes |= FileWatchChange::kMovedTo;
  return changes;
}

bool root_invalidated(std::uint32_t mask) noexcept {
  return (mask & (IN_DELETE_SELF | IN_MOVE_SELF | IN_IGNORED | IN_UNMOUNT)) !=
         0;
}

bool publish_inotify_buffer(FileWatcherState& state,
                            std::size_t byte_count) noexcept {
  const auto* bytes = reinterpret_cast<const std::byte*>(
      state.shared_native_buffer.data());
  std::size_t offset = 0;
  bool changed = false;
  bool lost_last_root = false;
  std::lock_guard<util::SpinLock> lock(state.monitor);
  while (offset < byte_count) {
    if (byte_count - offset < sizeof(inotify_event)) {
      changed = state.require_rescan_locked() || changed;
      break;
    }
    const auto* event =
        reinterpret_cast<const inotify_event*>(bytes + offset);
    const std::size_t record_size = sizeof(inotify_event) + event->len;
    if (record_size > byte_count - offset) {
      changed = state.require_rescan_locked() || changed;
      break;
    }
    if ((event->mask & IN_Q_OVERFLOW) != 0) {
      changed = state.require_rescan_locked() || changed;
      offset += record_size;
      continue;
    }

    const std::size_t* root_index =
        state.native_root_by_watch.find(event->wd);
    if (root_index == nullptr) {
      changed = state.require_rescan_locked() || changed;
      offset += record_size;
      continue;
    }
    NativeWatchRoot& root = state.roots[*root_index];
    if (root_invalidated(event->mask)) {
      root.active = false;
      changed = state.invalidate_root_locked(*root_index) || changed;
      bool any_active = false;
      for (const NativeWatchRoot& candidate : state.roots)
        any_active = any_active || candidate.active;
      lost_last_root = lost_last_root || !any_active;
    }

    std::size_t name_size = 0;
    while (name_size < event->len && event->name[name_size] != '\0')
      ++name_size;
    const FileWatchChange changes = translate_mask(event->mask);
    if (name_size != 0 && changes != FileWatchChange::kNone) {
      changed = state.emit_for_root_locked(
                    *root_index, std::string_view(event->name, name_size),
                    changes, event->cookie) ||
                changed;
    }
    offset += record_size;
  }
  state.publish_locked(changed);
  if (lost_last_root && state.failure.ok()) {
    state.failure = Status::NotFound("all FileWatcher roots were invalidated");
    state.changed.notify_all();
  }
  return changed;
}

sched::Task<void> run_inotify_pump(mem::AllocatorRef allocator,
                                   FileWatcherState* state) noexcept {
  (void)allocator;  // Selected by Task's allocator-aware promise new.
  Proactor& proactor = IoContextAccess::proactor(*state->io, state->owner);
  bool first = true;
  for (;;) {
    const std::int64_t result =
        co_await InotifyReadAwaitable{state, &proactor, &first};
    if (result == kCancelled && state->native_cancel.is_cancelled()) co_return;
    if (result == kSubmitFailed) {
      const Status error = Status::ResourceExhausted(
          "FileWatcher inotify read could not enter the IO shard");
      if (first)
        state->report_start_failure(error);
      else
        state->fail(error);
      co_return;
    }
    if (result == -EAGAIN || result == -EWOULDBLOCK) continue;
    if (result <= 0) {
      state->fail(result == 0
                      ? Status::DataLoss("FileWatcher inotify stream ended")
                      : status_from_native_result(
                            result, NativeOperationDomain::kPosix,
                            "FileWatcher inotify read"));
      co_return;
    }
    (void)publish_inotify_buffer(*state, static_cast<std::size_t>(result));
  }
}

}  // namespace

Status initialize_native_watch(FileWatcherState& state) noexcept {
  const int descriptor = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (descriptor < 0)
    return status_from_errno(errno, "create FileWatcher inotify descriptor");
  state.native_stream = descriptor;

  if (!state.native_root_by_watch.try_reserve(state.roots.size()))
    return Status::ResourceExhausted(
        "FileWatcher descriptor lookup allocation failed");
  constexpr std::uint32_t kMask =
      IN_ATTRIB | IN_CLOSE_WRITE | IN_MODIFY | IN_CREATE | IN_DELETE |
      IN_MOVED_FROM | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF |
      IN_UNMOUNT;
  for (std::size_t index = 0; index < state.roots.size(); ++index) {
    const int watch =
        ::inotify_add_watch(descriptor, state.root_path(index).data(), kMask);
    if (watch < 0)
      return status_from_errno(errno, "add FileWatcher inotify root");
    NativeWatchRoot& root = state.roots[index];
    root.native_watch = watch;
    root.active = true;
    auto [entry, result] =
        state.native_root_by_watch.try_emplace(watch, index);
    if (result == decltype(state.native_root_by_watch)::InsertResult::kOutOfMemory)
      return Status::ResourceExhausted(
          "FileWatcher descriptor lookup allocation failed");
    if (result == decltype(state.native_root_by_watch)::InsertResult::kAlreadyPresent)
      return Status::Internal("inotify reused an active watch descriptor");
    (void)entry;
  }

  const std::size_t words = (state.options.native_buffer_bytes + 3) / 4;
  if (!state.shared_native_buffer.try_resize(words))
    return Status::ResourceExhausted(
        "FileWatcher native buffer allocation failed");
  state.expected_arms = 1;
  sched::Task<void> task = run_inotify_pump(state.allocator, &state);
  if (!task.valid())
    return Status::ResourceExhausted(
        "FileWatcher pump coroutine allocation failed");
  state.shared_pump = sched::spawn_task_on(
      *state.scheduler, state.owner, state.allocator, std::move(task),
      kWatchPumpStackSize);
  if (state.shared_pump.is_finished()) {
    Result<void> started = state.shared_pump.join();
    if (!started.has_value()) return std::move(started).error();
    return Status::Internal("FileWatcher pump exited during startup");
  }
  return Status::Ok();
}

Status shutdown_native_watch(FileWatcherState& state) noexcept {
  Status status;
  if (state.shared_pump.valid()) {
    Result<void> joined = state.shared_pump.join();
    if (!joined.has_value()) status.Update(std::move(joined).error());
  }
  if (state.native_stream != -1) {
    const int descriptor = static_cast<int>(state.native_stream);
    for (NativeWatchRoot& root : state.roots) {
      if (root.native_watch >= 0) {
        // EINVAL is ordinary after IN_IGNORED removed an invalidated watch.
        if (::inotify_rm_watch(descriptor, root.native_watch) != 0 &&
            errno != EINVAL)
          status.Update(status_from_errno(errno,
                                          "remove FileWatcher inotify root"));
        root.native_watch = -1;
      }
      root.active = false;
    }
    state.native_stream = -1;
    if (::close(descriptor) != 0)
      status.Update(status_from_errno(errno,
                                      "close FileWatcher inotify descriptor"));
  }
  return status;
}

}  // namespace hyle::io::detail

#endif  // HYLE_OS_LINUX
