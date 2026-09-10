#include "watch_internal.h"

#include "hyle/core/environment.h"

#if HYLE_OS_LINUX

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string_view>
#include <sys/inotify.h>
#include <unistd.h>

#include "hyle/error/system_error.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/detail/net_cancel.h"
#include "hyle/io/detail/proactor.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/coro.h"

namespace hyle::io::detail {
namespace {

constexpr std::uint32_t kWatchMask =
    IN_ATTRIB | IN_CLOSE_WRITE | IN_MODIFY | IN_CREATE | IN_DELETE |
    IN_MOVED_FROM | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT |
    IN_ONLYDIR | IN_DONT_FOLLOW;

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

bool watch_invalidated(std::uint32_t mask) noexcept {
  return (mask & (IN_DELETE_SELF | IN_MOVE_SELF | IN_IGNORED | IN_UNMOUNT)) !=
         0;
}

bool directory_topology_changed(std::uint32_t mask) noexcept {
  return (mask & IN_ISDIR) != 0 &&
         (mask & (IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO)) != 0;
}

Result<std::size_t> add_native_directory(FileWatcherState& state,
                                         int descriptor,
                                         std::string_view path,
                                         bool recursive,
                                         std::size_t* scan_head,
                                         std::size_t* scan_tail) noexcept {
  if (std::size_t* existing = state.native_directory_by_path.find(path)) {
    NativeWatchDirectory& directory = state.native_directories[*existing];
    if (recursive && !directory.recursive) {
      directory.recursive = true;
      if (*scan_tail == kNoWatchIndex)
        *scan_head = *existing;
      else
        state.native_directories[*scan_tail].next_scan = *existing;
      *scan_tail = *existing;
    }
    return *existing;
  }
  if (state.native_directories.size() ==
      state.options.max_watched_directories)
    return make_unexpected(Status::ResourceExhausted(
        "FileWatcher reached max_watched_directories"));

  NativeWatchDirectory directory(state.allocator);
  if (!directory.path.try_resize(path.size() + 1))
    return make_unexpected(Status::ResourceExhausted(
        "FileWatcher directory path allocation failed"));
  if (!path.empty())
    std::memcpy(directory.path.data(), path.data(), path.size());
  directory.path[path.size()] = '\0';
  directory.native_watch =
      ::inotify_add_watch(descriptor, directory.path.data(), kWatchMask);
  if (directory.native_watch < 0)
    return make_unexpected(
        status_from_errno(errno, "add FileWatcher inotify directory"));
  directory.recursive = recursive;

  if (!state.native_directories.try_push_back(std::move(directory)))
    return make_unexpected(Status::ResourceExhausted(
        "FileWatcher directory table allocation failed"));
  const std::size_t index = state.native_directories.size() - 1;
  const std::string_view stored = state.native_directory_path(index);
  auto [path_entry, path_result] =
      state.native_directory_by_path.try_emplace(stored, index);
  if (path_result ==
      decltype(state.native_directory_by_path)::InsertResult::kOutOfMemory)
    return make_unexpected(Status::ResourceExhausted(
        "FileWatcher directory lookup allocation failed"));
  (void)path_entry;

  auto [watch_entry, watch_result] = state.native_root_by_watch.try_emplace(
      state.native_directories[index].native_watch, index);
  if (watch_result ==
      decltype(state.native_root_by_watch)::InsertResult::kOutOfMemory)
    return make_unexpected(Status::ResourceExhausted(
        "FileWatcher descriptor lookup allocation failed"));
  if (watch_result ==
      decltype(state.native_root_by_watch)::InsertResult::kAlreadyPresent)
    return make_unexpected(Status::FailedPrecondition(
        "FileWatcher roots alias one native directory identity"));
  (void)watch_entry;
  if (recursive) {
    if (*scan_tail == kNoWatchIndex)
      *scan_head = index;
    else
      state.native_directories[*scan_tail].next_scan = index;
    *scan_tail = index;
  }
  return index;
}

struct DirectoryScanContext {
  FileWatcherState* state = nullptr;
  int descriptor = -1;
  std::size_t prefix_size = 0;
  std::size_t* scan_head = nullptr;
  std::size_t* scan_tail = nullptr;
  Status status{};
};

bool add_scanned_directory(const char* name, platform::FileType type,
                           void* opaque) noexcept {
  if (type != platform::FileType::kDirectory &&
      type != platform::FileType::kUnknown)
    return true;
  auto& context = *static_cast<DirectoryScanContext*>(opaque);
  auto& scratch = context.state->shared_path_buffer;
  const std::size_t name_size = std::strlen(name);
  if (context.prefix_size >= static_cast<std::size_t>(PATH_MAX) ||
      name_size > static_cast<std::size_t>(PATH_MAX) - context.prefix_size - 1) {
    context.status =
        Status::OutOfRange("FileWatcher descendant path exceeds PATH_MAX");
    return false;
  }
  const std::size_t size = context.prefix_size + name_size;
  std::memcpy(scratch.data() + context.prefix_size, name, name_size);
  scratch[size] = '\0';
  if (type == platform::FileType::kUnknown) {
    Result<platform::FileInfo> info = platform::stat(scratch.data());
    if (!info.has_value()) {
      if (info.error().IsNotFound()) return true;
      context.status = std::move(info).error().WithContext(
          "inspect FileWatcher descendant");
      return false;
    }
    if (!info->is_directory() || info->is_symlink) return true;
  }
  Result<std::size_t> added = add_native_directory(
      *context.state, context.descriptor,
      std::string_view(scratch.data(), size), true, context.scan_head,
      context.scan_tail);
  if (!added.has_value()) {
    context.status = std::move(added).error();
    return false;
  }
  return true;
}

Status populate_native_directories(FileWatcherState& state, int descriptor,
                                   bool initial) noexcept {
  if (!state.native_root_by_watch.try_reserve(state.roots.size()) ||
      !state.native_directory_by_path.try_reserve(state.roots.size()))
    return Status::ResourceExhausted(
        "FileWatcher native lookup allocation failed");

  std::size_t scan_head = kNoWatchIndex;
  std::size_t scan_tail = kNoWatchIndex;
  for (std::size_t root_index = 0; root_index < state.roots.size();
       ++root_index) {
    NativeWatchRoot& root = state.roots[root_index];
    if (!initial && !root.active) continue;
    Result<std::size_t> added = add_native_directory(
        state, descriptor, state.root_path(root_index),
        root.first_tree != kNoWatchIndex, &scan_head, &scan_tail);
    if (!added.has_value()) {
      Status error = std::move(added).error();
      if (!initial && error.IsNotFound()) {
        std::lock_guard<util::SpinLock> lock(state.monitor);
        root.active = false;
        const bool changed = state.invalidate_root_locked(root_index);
        state.publish_locked(changed);
        continue;
      }
      return error.WithContext("FileWatcher root");
    }
    root.native_watch = state.native_directories[*added].native_watch;
    root.active = true;
  }

  // A FIFO of table indices is deliberately embedded in the directory table:
  // target order may insert a direct child root before its recursive parent.
  // Scanning the parent then upgrades that already-passed child to recursive.
  // A monotonic vector loop would miss the child's descendants; this worklist
  // enqueues every false->true transition exactly once without a second O(D)
  // allocation or an O(D^2) restart scan.
  while (scan_head != kNoWatchIndex) {
    const std::size_t index = scan_head;
    scan_head = state.native_directories[index].next_scan;
    state.native_directories[index].next_scan = kNoWatchIndex;
    if (scan_head == kNoWatchIndex) scan_tail = kNoWatchIndex;
    const std::string_view parent = state.native_directory_path(index);
    if (parent.size() >= static_cast<std::size_t>(PATH_MAX))
      return Status::OutOfRange("FileWatcher directory exceeds PATH_MAX");
    std::memcpy(state.shared_path_buffer.data(), parent.data(), parent.size());
    std::size_t prefix = parent.size();
    if (prefix == 0 || state.shared_path_buffer[prefix - 1] != '/')
      state.shared_path_buffer[prefix++] = '/';
    state.shared_path_buffer[prefix] = '\0';
    DirectoryScanContext context{&state, descriptor, prefix, &scan_head,
                                 &scan_tail};
    Result<bool> walked = platform::directory_iterate(
        state.native_directories[index].path.data(), add_scanned_directory,
        &context);
    if (!context.status.ok()) return context.status;
    if (!walked.has_value()) {
      Status error = std::move(walked).error();
      if (error.IsNotFound()) continue;
      return error.WithContext("enumerate FileWatcher directory tree");
    }
  }

  bool any_active = false;
  for (const NativeWatchRoot& root : state.roots)
    any_active = any_active || root.active;
  return any_active
             ? Status::Ok()
             : Status::NotFound("all FileWatcher roots were invalidated");
}

void clear_native_directories(FileWatcherState& state) noexcept {
  state.native_directory_by_path.clear();
  state.native_root_by_watch.clear();
  state.native_directories.clear();
  for (NativeWatchRoot& root : state.roots) root.native_watch = -1;
}

Status rebuild_native_directories(FileWatcherState& state) noexcept {
  if (state.native_stream != -1) {
    const int old_descriptor = static_cast<int>(state.native_stream);
    state.native_stream = -1;
    if (::close(old_descriptor) != 0)
      return status_from_errno(errno,
                               "close stale FileWatcher inotify descriptor");
  }
  clear_native_directories(state);
  const int descriptor = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (descriptor < 0)
    return status_from_errno(errno, "recreate FileWatcher inotify descriptor");
  state.native_stream = descriptor;
  Status populated = populate_native_directories(state, descriptor, false);
  if (!populated.ok()) {
    state.native_stream = -1;
    (void)::close(descriptor);
    clear_native_directories(state);
  }
  return populated;
}

bool publish_inotify_buffer(FileWatcherState& state,
                            std::size_t byte_count) noexcept {
  const auto* bytes = reinterpret_cast<const std::byte*>(
      state.shared_native_buffer.data());
  std::size_t offset = 0;
  bool changed = false;
  bool rebuild = false;
  bool lost_last_root = false;
  const bool recursive = state.has_tree_targets;
  std::lock_guard<util::SpinLock> lock(state.monitor);
  while (offset < byte_count) {
    if (byte_count - offset < sizeof(inotify_event)) {
      changed = state.require_rescan_locked() || changed;
      rebuild = recursive;
      break;
    }
    const auto* event =
        reinterpret_cast<const inotify_event*>(bytes + offset);
    const std::size_t record_size = sizeof(inotify_event) + event->len;
    if (record_size > byte_count - offset) {
      changed = state.require_rescan_locked() || changed;
      rebuild = recursive;
      break;
    }
    if ((event->mask & IN_Q_OVERFLOW) != 0) {
      changed = state.require_rescan_locked() || changed;
      rebuild = recursive;
      offset += record_size;
      continue;
    }

    const std::size_t* directory_index =
        state.native_root_by_watch.find(event->wd);
    if (directory_index == nullptr) {
      changed = state.require_rescan_locked() || changed;
      rebuild = recursive;
      offset += record_size;
      continue;
    }
    if (!recursive) {
      NativeWatchRoot& root = state.roots[*directory_index];
      if (watch_invalidated(event->mask)) {
        root.active = false;
        changed = state.invalidate_root_locked(*directory_index) || changed;
        changed = state.require_rescan_locked() || changed;
        bool any_active = false;
        for (const NativeWatchRoot& candidate : state.roots)
          any_active = any_active || candidate.active;
        lost_last_root = lost_last_root || !any_active;
      }
      std::size_t name_size = 0;
      while (name_size < event->len && event->name[name_size] != '\0')
        ++name_size;
      const FileWatchChange changes = translate_mask(event->mask);
      if (name_size != 0 && changes != FileWatchChange::kNone)
        changed = state.emit_for_root_locked(
                      *directory_index,
                      std::string_view(event->name, name_size), changes,
                      event->cookie) ||
                  changed;
      offset += record_size;
      continue;
    }
    if (watch_invalidated(event->mask)) {
      const std::string_view directory_path =
          state.native_directory_path(*directory_index);
      if (const std::size_t* root_index =
              state.root_by_path.find(directory_path)) {
        NativeWatchRoot& root = state.roots[*root_index];
        root.active = false;
        changed = state.invalidate_root_locked(*root_index) || changed;
      }
      rebuild = true;
      changed = state.require_rescan_locked() || changed;
      bool any_active = false;
      for (const NativeWatchRoot& root : state.roots)
        any_active = any_active || root.active;
      lost_last_root = lost_last_root || !any_active;
    }

    std::size_t name_size = 0;
    while (name_size < event->len && event->name[name_size] != '\0')
      ++name_size;
    const FileWatchChange changes = translate_mask(event->mask);
    if (name_size != 0 && changes != FileWatchChange::kNone) {
      const std::string_view parent =
          state.native_directory_path(*directory_index);
      const std::size_t separator =
          parent.empty() || parent.back() == '/' ? 0 : 1;
      if (parent.size() + separator + name_size >=
          state.shared_path_buffer.size()) {
        changed = state.require_rescan_locked() || changed;
        rebuild = true;
      } else {
        std::memcpy(state.shared_path_buffer.data(), parent.data(),
                    parent.size());
        std::size_t size = parent.size();
        if (separator != 0) state.shared_path_buffer[size++] = '/';
        std::memcpy(state.shared_path_buffer.data() + size, event->name,
                    name_size);
        size += name_size;
        changed = state.emit_absolute_path_locked(
                      std::string_view(state.shared_path_buffer.data(), size),
                      changes, event->cookie) ||
                  changed;
      }
    }
    if (directory_topology_changed(event->mask)) {
      changed = state.require_rescan_locked() || changed;
      rebuild = true;
    }
    offset += record_size;
  }
  state.publish_locked(changed);
  if (lost_last_root && state.failure.ok()) {
    state.failure = Status::NotFound("all FileWatcher roots were invalidated");
    state.changed.notify_all();
  }
  return rebuild && !lost_last_root;
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
    if (publish_inotify_buffer(*state, static_cast<std::size_t>(result))) {
      Status rebuilt = rebuild_native_directories(*state);
      if (!rebuilt.ok()) {
        state->fail(std::move(rebuilt));
        co_return;
      }
    }
  }
}

}  // namespace

Status initialize_native_watch(FileWatcherState& state) noexcept {
  const std::size_t words = (state.options.native_buffer_bytes + 3) / 4;
  if (!state.shared_native_buffer.try_resize(words))
    return Status::ResourceExhausted(
        "FileWatcher native buffer allocation failed");

  const int descriptor = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (descriptor < 0)
    return status_from_errno(errno, "create FileWatcher inotify descriptor");
  state.native_stream = descriptor;
  if (state.has_tree_targets) {
    if (!state.shared_path_buffer.try_resize(
            static_cast<std::size_t>(PATH_MAX) + 1))
      return Status::ResourceExhausted(
          "FileWatcher path scratch allocation failed");
    Status populated = populate_native_directories(state, descriptor, true);
    if (!populated.ok()) return populated;
  } else {
    if (!state.native_root_by_watch.try_reserve(state.roots.size()))
      return Status::ResourceExhausted(
          "FileWatcher descriptor lookup allocation failed");
    for (std::size_t index = 0; index < state.roots.size(); ++index) {
      const int watch = ::inotify_add_watch(
          descriptor, state.root_path(index).data(), kWatchMask);
      if (watch < 0)
        return status_from_errno(errno, "add FileWatcher inotify root");
      NativeWatchRoot& root = state.roots[index];
      root.native_watch = watch;
      root.active = true;
      auto [entry, result] =
          state.native_root_by_watch.try_emplace(watch, index);
      if (result ==
          decltype(state.native_root_by_watch)::InsertResult::kOutOfMemory)
        return Status::ResourceExhausted(
            "FileWatcher descriptor lookup allocation failed");
      if (result ==
          decltype(state.native_root_by_watch)::InsertResult::kAlreadyPresent)
        return Status::Internal("inotify reused an active watch descriptor");
      (void)entry;
    }
  }

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
    state.native_stream = -1;
    if (::close(descriptor) != 0)
      status.Update(status_from_errno(
          errno, "close FileWatcher inotify descriptor"));
  }
  clear_native_directories(state);
  for (NativeWatchRoot& root : state.roots) root.active = false;
  return status;
}

}  // namespace hyle::io::detail

#endif  // HYLE_OS_LINUX
