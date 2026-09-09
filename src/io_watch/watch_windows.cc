#include "watch_internal.h"

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

#include "hyle/error/system_error.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/detail/net_cancel.h"
#include "hyle/io/detail/proactor.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/coro.h"

namespace hyle::io::detail {
namespace {

Status encode_native_root(std::string_view input, mem::AllocatorRef allocator,
                          container::PodVector<wchar_t>* output) noexcept {
  if (input.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    return Status::OutOfRange("file watch path exceeds Win32 limits");
  const int decoded = ::MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, input.data(), static_cast<int>(input.size()),
      nullptr, 0);
  if (decoded <= 0)
    return status_from_win32(::GetLastError(),
                             "decode file watch root path");

  container::PodVector<wchar_t> plain(allocator);
  if (!plain.try_resize(static_cast<std::size_t>(decoded) + 1))
    return Status::ResourceExhausted(
        "file watch native root allocation failed");
  if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.data(),
                            static_cast<int>(input.size()), plain.data(),
                            decoded) != decoded)
    return status_from_win32(::GetLastError(),
                             "decode file watch root path");
  plain[static_cast<std::size_t>(decoded)] = L'\0';
  for (int i = 0; i < decoded; ++i) {
    if (plain[static_cast<std::size_t>(i)] == L'/')
      plain[static_cast<std::size_t>(i)] = L'\\';
  }

  const bool unc = decoded >= 2 && plain[0] == L'\\' && plain[1] == L'\\';
  constexpr wchar_t kDevicePrefix[] = L"\\\\?\\";
  constexpr wchar_t kUncPrefix[] = L"\\\\?\\UNC\\";
  const std::size_t prefix = unc ? std::size(kUncPrefix) - 1
                                 : std::size(kDevicePrefix) - 1;
  const std::size_t skipped = unc ? 2 : 0;
  const std::size_t tail = static_cast<std::size_t>(decoded) - skipped;
  if (!output->try_resize(prefix + tail + 1))
    return Status::ResourceExhausted(
        "file watch extended root allocation failed");
  std::memcpy(output->data(), unc ? kUncPrefix : kDevicePrefix,
              prefix * sizeof(wchar_t));
  std::memcpy(output->data() + prefix, plain.data() + skipped,
              tail * sizeof(wchar_t));
  (*output)[prefix + tail] = L'\0';
  return Status::Ok();
}

struct DirectoryReadAwaitable {
  FileWatcherState* state = nullptr;
  NativeWatchRoot* root = nullptr;
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
        root->native_buffer.size() * sizeof(std::uint32_t));
    if (!proactor->submit_directory_changes(
            root->native_handle, root->native_buffer.data(), bytes,
            cancel.token)) {
      operation.result = kSubmitFailed;
      return false;
    }
    if (*first) {
      *first = false;
      root->active = true;
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

FileWatchChange translate_action(DWORD action) noexcept {
  switch (action) {
    case FILE_ACTION_ADDED:
      return FileWatchChange::kCreated;
    case FILE_ACTION_REMOVED:
      return FileWatchChange::kRemoved;
    case FILE_ACTION_MODIFIED:
      // ReadDirectoryChangesW reports one coarse MODIFIED action for the
      // selected content/size/metadata filters. Preserve that uncertainty.
      return FileWatchChange::kContent | FileWatchChange::kMetadata;
    case FILE_ACTION_RENAMED_OLD_NAME:
      return FileWatchChange::kMovedFrom;
    case FILE_ACTION_RENAMED_NEW_NAME:
      return FileWatchChange::kMovedTo;
    default:
      return FileWatchChange::kNone;
  }
}

bool root_failure_code(std::int64_t result) noexcept {
  if (result >= 0) return false;
  const auto code = static_cast<DWORD>(-result);
  return code == ERROR_ACCESS_DENIED || code == ERROR_FILE_NOT_FOUND ||
         code == ERROR_PATH_NOT_FOUND || code == ERROR_INVALID_HANDLE;
}

void publish_root_loss(FileWatcherState& state, std::size_t root_index,
                       std::int64_t native_result) noexcept {
  bool no_roots = true;
  {
    std::lock_guard<util::SpinLock> lock(state.monitor);
    NativeWatchRoot& root = state.roots[root_index];
    root.active = false;
    const bool changed = state.invalidate_root_locked(root_index);
    for (const NativeWatchRoot& candidate : state.roots)
      no_roots = no_roots && !candidate.active;
    state.publish_locked(changed);
  }
  if (no_roots) {
    state.fail(status_from_native_result(
        native_result, NativeOperationDomain::kWin32,
        "all FileWatcher roots were invalidated"));
  }
}

bool publish_directory_buffer(FileWatcherState& state, std::size_t root_index,
                              std::size_t byte_count) noexcept {
  NativeWatchRoot& root = state.roots[root_index];
  const auto* bytes = reinterpret_cast<const std::byte*>(
      root.native_buffer.data());
  std::size_t offset = 0;
  bool changed = false;
  std::lock_guard<util::SpinLock> lock(state.monitor);
  while (offset < byte_count) {
    if (byte_count - offset < offsetof(FILE_NOTIFY_INFORMATION, FileName)) {
      changed = state.require_rescan_locked() || changed;
      break;
    }
    const auto* record = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(
        bytes + offset);
    const std::size_t name_bytes = record->FileNameLength;
    const std::size_t header = offsetof(FILE_NOTIFY_INFORMATION, FileName);
    if ((name_bytes & 1u) != 0 || name_bytes > byte_count - offset - header) {
      changed = state.require_rescan_locked() || changed;
      break;
    }
    const int name_units = static_cast<int>(name_bytes / sizeof(wchar_t));
    char name[1024];
    int name_size = 0;
    if (name_units != 0) {
      name_size = ::WideCharToMultiByte(
          CP_UTF8, WC_ERR_INVALID_CHARS, record->FileName, name_units, name,
          static_cast<int>(sizeof(name)), nullptr, nullptr);
      if (name_size <= 0) {
        changed = state.require_rescan_locked() || changed;
      }
    }
    const FileWatchChange action = translate_action(record->Action);
    if (name_size > 0 && action != FileWatchChange::kNone) {
      std::uint64_t correlation = 0;
      if (record->Action == FILE_ACTION_RENAMED_OLD_NAME) {
        correlation = state.next_rename_correlation++;
        if (correlation == 0) correlation = state.next_rename_correlation++;
        root.pending_rename = correlation;
      } else if (record->Action == FILE_ACTION_RENAMED_NEW_NAME) {
        correlation = root.pending_rename;
        root.pending_rename = 0;
      }
      changed = state.emit_for_root_locked(
                    root_index,
                    std::string_view(name, static_cast<std::size_t>(name_size)),
                    action, correlation) ||
                changed;
    }

    if (record->NextEntryOffset == 0) break;
    if (record->NextEntryOffset < header ||
        record->NextEntryOffset > byte_count - offset) {
      changed = state.require_rescan_locked() || changed;
      break;
    }
    offset += record->NextEntryOffset;
  }
  state.publish_locked(changed);
  return changed;
}

sched::Task<void> run_directory_pump(mem::AllocatorRef allocator,
                                     FileWatcherState* state,
                                     std::size_t root_index) noexcept {
  Proactor& proactor = IoContextAccess::proactor(*state->io, state->owner);
  NativeWatchRoot& root = state->roots[root_index];
  bool first = true;
  for (;;) {
    const std::int64_t result = co_await DirectoryReadAwaitable{
        state, &root, &proactor, &first};
    if (result == kCancelled && state->native_cancel.is_cancelled()) co_return;
    if (result == kSubmitFailed) {
      const Status error = Status::ResourceExhausted(
          "FileWatcher directory read could not enter the IO shard");
      if (first)
        state->report_start_failure(error);
      else
        state->fail(error);
      co_return;
    }
    if (result == -static_cast<std::int64_t>(ERROR_NOTIFY_ENUM_DIR) ||
        result == 0) {
      std::lock_guard<util::SpinLock> lock(state->monitor);
      root.pending_rename = 0;
      const bool changed = state->require_rescan_locked();
      state->publish_locked(changed);
      continue;
    }
    if (result < 0) {
      if (root_failure_code(result))
        publish_root_loss(*state, root_index, result);
      else
        state->fail(status_from_native_result(
            result, NativeOperationDomain::kWin32,
            "FileWatcher ReadDirectoryChangesW"));
      co_return;
    }
    (void)publish_directory_buffer(*state, root_index,
                                   static_cast<std::size_t>(result));
  }
}

}  // namespace

Status initialize_native_watch(FileWatcherState& state) noexcept {
  state.expected_arms = state.roots.size();
  const std::size_t words = (state.options.native_buffer_bytes + 3) / 4;
  for (std::size_t index = 0; index < state.roots.size(); ++index) {
    NativeWatchRoot& root = state.roots[index];
    if (!root.native_buffer.try_resize(words))
      return Status::ResourceExhausted(
          "FileWatcher native buffer allocation failed");
    container::PodVector<wchar_t> path(state.allocator);
    Status encoded = encode_native_root(state.root_path(index), state.allocator,
                                        &path);
    if (!encoded.ok()) return encoded;
    HANDLE handle = ::CreateFileW(
        path.data(), FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
      return status_from_win32(::GetLastError(),
                               "open FileWatcher directory root");
    root.native_handle = reinterpret_cast<std::intptr_t>(handle);
    if (!IoContextAccess::bind_handle(*state.io, state.owner,
                                      root.native_handle))
      return status_from_win32(::GetLastError(),
                               "associate FileWatcher root with IO shard");
  }

  for (std::size_t index = 0; index < state.roots.size(); ++index) {
    sched::Task<void> task =
        run_directory_pump(state.allocator, &state, index);
    if (!task.valid())
      return Status::ResourceExhausted(
          "FileWatcher pump coroutine allocation failed");
    state.roots[index].pump = sched::spawn_task_on(
        *state.scheduler, state.owner, state.allocator, std::move(task),
        mem::kFiberStackSmall);
    if (state.roots[index].pump.is_finished()) {
      Result<void> started = state.roots[index].pump.join();
      if (!started.has_value()) return std::move(started).error();
      return Status::Internal("FileWatcher pump exited during startup");
    }
  }
  return Status::Ok();
}

Status shutdown_native_watch(FileWatcherState& state) noexcept {
  Status status;
  for (NativeWatchRoot& root : state.roots) {
    if (root.pump.valid()) {
      Result<void> joined = root.pump.join();
      if (!joined.has_value()) status.Update(std::move(joined).error());
    }
  }
  for (NativeWatchRoot& root : state.roots) {
    if (root.native_handle != -1) {
      HANDLE handle = reinterpret_cast<HANDLE>(root.native_handle);
      root.native_handle = -1;
      if (!::CloseHandle(handle))
        status.Update(status_from_win32(::GetLastError(),
                                       "close FileWatcher directory root"));
    }
    root.active = false;
  }
  return status;
}

}  // namespace hyle::io::detail

#endif  // HYLE_OS_WINDOWS
