#include "process_internal.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cwchar>
#include <mutex>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/core/environment.h"
#include "hyle/error/system_error.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace hyle::process::detail {
namespace {

template <class T, std::size_t Inline>
using Scratch = container::SmallVector<T, Inline, mem::AllocatorRef>;

constexpr DWORD kForcedExitCode = 0xC000013Au;

HANDLE as_handle(std::intptr_t value) noexcept {
  return reinterpret_cast<HANDLE>(value);
}

std::intptr_t as_native(HANDLE value) noexcept {
  return reinterpret_cast<std::intptr_t>(value);
}

Status win32_status(DWORD error, const char* operation) noexcept {
  return status_from_win32(error, operation);
}

template <std::size_t Inline>
bool append_wide(Scratch<wchar_t, Inline>& output, const wchar_t* source,
                 std::size_t size, bool nul = false) noexcept {
  const std::size_t extra = size + (nul ? 1u : 0u);
  if (extra > SIZE_MAX - output.size() ||
      !output.try_reserve(output.size() + extra)) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) output.push_back(source[i]);
  if (nul) output.push_back(L'\0');
  return true;
}

Status append_utf8(Scratch<wchar_t, 512>& output, std::string_view text,
                   bool nul = false) noexcept {
  if (text.size() > static_cast<std::size_t>(INT_MAX))
    return Status::OutOfRange("process UTF-8 value exceeds Win32 limit");
  int required = 0;
  if (!text.empty()) {
    required = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), nullptr, 0);
    if (required <= 0)
      return win32_status(::GetLastError(), "decode process UTF-8 value");
  }
  const std::size_t extra = static_cast<std::size_t>(required) +
                            (nul ? 1u : 0u);
  if (extra > SIZE_MAX - output.size() ||
      !output.try_resize(output.size() + extra)) {
    return Status::ResourceExhausted(
        "process UTF-16 scratch allocation failed");
  }
  const std::size_t offset = output.size() - extra;
  if (required != 0 &&
      ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                            static_cast<int>(text.size()),
                            output.data() + offset, required) != required) {
    return win32_status(::GetLastError(), "decode process UTF-8 value");
  }
  if (nul) output.back() = L'\0';
  return Status{};
}

Result<Scratch<wchar_t, 512>> wide_c_string(
    mem::AllocatorRef allocator, std::string_view text) noexcept {
  Scratch<wchar_t, 512> result(allocator);
  Status status = append_utf8(result, text, true);
  if (!status.ok()) return make_unexpected(std::move(status));
  return result;
}

Status append_quoted_argument(mem::AllocatorRef allocator,
                              Scratch<wchar_t, 512>& command,
                              std::string_view argument) noexcept {
  Scratch<wchar_t, 512> wide(allocator);
  Status status = append_utf8(wide, argument);
  if (!status.ok()) return status;
  if (!command.empty() && !command.try_push_back(L' '))
    return Status::ResourceExhausted(
        "process command-line allocation failed");
  if (!command.try_push_back(L'\"'))
    return Status::ResourceExhausted(
        "process command-line allocation failed");
  std::size_t backslashes = 0;
  for (wchar_t value : wide) {
    if (value == L'\\') {
      ++backslashes;
      continue;
    }
    if (value == L'\"') {
      if (!command.try_reserve(command.size() + backslashes * 2 + 2))
        return Status::ResourceExhausted(
            "process command-line allocation failed");
      for (std::size_t i = 0; i < backslashes * 2 + 1; ++i)
        command.push_back(L'\\');
      command.push_back(L'\"');
    } else {
      if (!command.try_reserve(command.size() + backslashes + 1))
        return Status::ResourceExhausted(
            "process command-line allocation failed");
      for (std::size_t i = 0; i < backslashes; ++i)
        command.push_back(L'\\');
      command.push_back(value);
    }
    backslashes = 0;
  }
  if (!command.try_reserve(command.size() + backslashes * 2 + 2))
    return Status::ResourceExhausted(
        "process command-line allocation failed");
  for (std::size_t i = 0; i < backslashes * 2; ++i)
    command.push_back(L'\\');
  command.push_back(L'\"');
  return Status{};
}

Result<Scratch<wchar_t, 512>> build_command_line(
    mem::AllocatorRef allocator, const ProcessStartOptions& options) noexcept {
  Scratch<wchar_t, 512> result(allocator);
  Status status = append_quoted_argument(allocator, result, options.executable);
  for (std::string_view argument : options.arguments) {
    if (!status.ok()) break;
    status = append_quoted_argument(allocator, result, argument);
  }
  if (status.ok() && !result.try_push_back(L'\0'))
    status = Status::ResourceExhausted(
        "process command-line allocation failed");
  if (!status.ok()) return make_unexpected(std::move(status));
  return result;
}

bool equal_name(const wchar_t* left, std::size_t left_size,
                const wchar_t* right, std::size_t right_size) noexcept {
  if (left_size != right_size) return false;
  return ::CompareStringOrdinal(left, static_cast<int>(left_size), right,
                                static_cast<int>(right_size), TRUE) ==
         CSTR_EQUAL;
}

std::size_t environment_name_size(const wchar_t* text,
                                  std::size_t size) noexcept {
  std::size_t cursor = text[0] == L'=' ? 1u : 0u;
  while (cursor < size && text[cursor] != L'=') ++cursor;
  return cursor;
}

struct WideChange {
  std::size_t offset = 0;
  std::size_t name_size = 0;
  std::size_t text_size = 0;
  bool unset = false;
};

struct EnvironmentItem {
  const wchar_t* inherited = nullptr;
  std::size_t change_offset = 0;
  std::size_t size = 0;
  bool changed = false;
};

class EnvironmentBlock {
 public:
  explicit EnvironmentBlock(mem::AllocatorRef allocator) noexcept
      : changes_(allocator), change_text_(allocator), items_(allocator),
        output_(allocator) {}
  ~EnvironmentBlock() {
    if (inherited_ != nullptr) ::FreeEnvironmentStringsW(inherited_);
  }
  EnvironmentBlock(const EnvironmentBlock&) = delete;
  EnvironmentBlock& operator=(const EnvironmentBlock&) = delete;

  Status build(const ProcessStartOptions& options) noexcept {
    for (const EnvironmentChange& change : options.environment_changes) {
      const std::size_t offset = change_text_.size();
      Status status = append_utf8(change_text_, change.name);
      if (!status.ok()) return status;
      const std::size_t name_size = change_text_.size() - offset;
      for (const WideChange& existing : changes_) {
        if (equal_name(change_text_.data() + existing.offset,
                       existing.name_size, change_text_.data() + offset,
                       name_size)) {
          return Status::InvalidArgument(
              "Windows process environment contains duplicate names");
        }
      }
      if (change.value.has_value()) {
        if (!change_text_.try_push_back(L'='))
          return Status::ResourceExhausted(
              "process environment allocation failed");
        status = append_utf8(change_text_, *change.value, true);
        if (!status.ok()) return status;
      }
      if (!changes_.try_push_back(WideChange{
              offset, name_size, change_text_.size() - offset,
              !change.value.has_value()})) {
        return Status::ResourceExhausted(
            "process environment index allocation failed");
      }
    }

    if (options.environment_mode == EnvironmentMode::kInherit) {
      inherited_ = ::GetEnvironmentStringsW();
      if (inherited_ == nullptr)
        return win32_status(::GetLastError(), "read process environment");
      const wchar_t* current = inherited_;
      while (*current != L'\0') {
        const std::size_t size = std::wcslen(current);
        const std::size_t name_size = environment_name_size(current, size);
        bool replaced = false;
        for (const WideChange& change : changes_) {
          if (equal_name(current, name_size,
                         change_text_.data() + change.offset,
                         change.name_size)) {
            replaced = true;
            break;
          }
        }
        if (!replaced && !items_.try_push_back(
                             EnvironmentItem{current, 0, size, false})) {
          return Status::ResourceExhausted(
              "process environment index allocation failed");
        }
        current += size + 1;
      }
    }

    for (const WideChange& change : changes_) {
      if (!change.unset &&
          !items_.try_push_back(EnvironmentItem{
              nullptr, change.offset, change.text_size - 1, true})) {
        return Status::ResourceExhausted(
            "process environment index allocation failed");
      }
    }
    std::sort(items_.begin(), items_.end(), [&](const EnvironmentItem& left,
                                                const EnvironmentItem& right) {
      const wchar_t* a = left.changed
                             ? change_text_.data() + left.change_offset
                             : left.inherited;
      const wchar_t* b = right.changed
                             ? change_text_.data() + right.change_offset
                             : right.inherited;
      return ::CompareStringOrdinal(a, static_cast<int>(left.size), b,
                                    static_cast<int>(right.size), TRUE) ==
             CSTR_LESS_THAN;
    });
    for (const EnvironmentItem& item : items_) {
      const wchar_t* text = item.changed
                                ? change_text_.data() + item.change_offset
                                : item.inherited;
      if (!append_wide(output_, text, item.size, true))
        return Status::ResourceExhausted(
            "process environment block allocation failed");
    }
    if (output_.empty() && !output_.try_push_back(L'\0'))
      return Status::ResourceExhausted(
          "process environment block allocation failed");
    if (!output_.try_push_back(L'\0'))
      return Status::ResourceExhausted(
          "process environment block allocation failed");
    return Status{};
  }

  wchar_t* data() noexcept { return output_.data(); }

 private:
  Scratch<WideChange, 16> changes_;
  Scratch<wchar_t, 512> change_text_;
  Scratch<EnvironmentItem, 64> items_;
  Scratch<wchar_t, 1024> output_;
  LPWCH inherited_ = nullptr;
};

Result<HANDLE> duplicate_inheritable(HANDLE source,
                                     const char* operation) noexcept {
  HANDLE duplicate = INVALID_HANDLE_VALUE;
  if (source == nullptr || source == INVALID_HANDLE_VALUE ||
      !::DuplicateHandle(::GetCurrentProcess(), source, ::GetCurrentProcess(),
                         &duplicate, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
    return make_unexpected(win32_status(::GetLastError(), operation));
  }
  return duplicate;
}

Result<HANDLE> open_null(bool input) noexcept {
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  HANDLE handle = ::CreateFileW(
      L"NUL", input ? GENERIC_READ : GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    return make_unexpected(win32_status(::GetLastError(),
                                        "open NUL for child stdio"));
  return handle;
}

Result<std::pair<HANDLE, HANDLE>> make_pipe_pair(bool parent_writes) noexcept {
  static std::atomic<unsigned long long> sequence{0};
  wchar_t name[128]{};
  const unsigned long long id =
      sequence.fetch_add(1, std::memory_order_relaxed);
  const int written = _snwprintf_s(
      name, std::size(name), _TRUNCATE, L"\\\\.\\pipe\\hyle-process-%lu-%llu",
      static_cast<unsigned long>(::GetCurrentProcessId()), id);
  if (written <= 0)
    return make_unexpected(
        Status::Internal("format child named-pipe path failed"));

  const DWORD parent_access =
      (parent_writes ? PIPE_ACCESS_OUTBOUND : PIPE_ACCESS_INBOUND) |
      FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE;
  HANDLE parent = ::CreateNamedPipeW(
      name, parent_access,
      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT |
          PIPE_REJECT_REMOTE_CLIENTS,
      1, 64 * 1024, 64 * 1024, 0, nullptr);
  if (parent == INVALID_HANDLE_VALUE)
    return make_unexpected(
        win32_status(::GetLastError(), "create child named pipe"));

  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  HANDLE child = ::CreateFileW(
      name, parent_writes ? GENERIC_READ : GENERIC_WRITE, 0, &security,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (child == INVALID_HANDLE_VALUE) {
    const DWORD error = ::GetLastError();
    (void)::CloseHandle(parent);
    return make_unexpected(win32_status(error, "connect child named pipe"));
  }
  if (!::SetHandleInformation(parent, HANDLE_FLAG_INHERIT, 0)) {
    const DWORD error = ::GetLastError();
    (void)::CloseHandle(child);
    (void)::CloseHandle(parent);
    return make_unexpected(
        win32_status(error, "make parent process pipe non-inheritable"));
  }
  return std::pair<HANDLE, HANDLE>{parent, child};
}

struct StdioSetup {
  explicit StdioSetup(ProcessState* value) noexcept : state(value) {}
  ~StdioSetup() {
    for (std::size_t i = 0; i < owned_count; ++i)
      if (owned[i] != nullptr && owned[i] != INVALID_HANDLE_VALUE)
        (void)::CloseHandle(owned[i]);
  }

  bool remember(HANDLE handle) noexcept {
    for (std::size_t i = 0; i < owned_count; ++i)
      if (owned[i] == handle) return true;
    if (owned_count == std::size(owned)) return false;
    owned[owned_count++] = handle;
    return true;
  }

  Status prepare_pipe(int slot, const ProcessStartOptions& options) noexcept {
    auto pair = make_pipe_pair(slot == 0);
    if (!pair.has_value()) return std::move(pair).error();
    auto parent = io::LocalStream::adopt_anonymous(
        *options.pipe_io, options.pipe_owner, as_native(pair->first));
    pair->first = INVALID_HANDLE_VALUE;  // adopt owns success and failure
    if (!parent.has_value()) {
      (void)::CloseHandle(pair->second);
      return std::move(parent).error();
    }
    if (!remember(pair->second)) {
      (void)::CloseHandle(pair->second);
      return Status::ResourceExhausted("too many child stdio handles");
    }
    child[slot] = pair->second;
    if (slot == 0)
      state->stdin_stream = std::move(*parent);
    else if (slot == 1)
      state->stdout_stream = std::move(*parent);
    else
      state->stderr_stream = std::move(*parent);
    return Status{};
  }

  Status prepare_slot(int slot, ProcessStdio config,
                      const ProcessStartOptions& options) noexcept {
    Result<HANDLE> handle = make_unexpected(Status::Internal(
        "child stdio setup did not select a handle"));
    switch (config.kind()) {
      case ProcessStdioKind::kInherit: {
        const DWORD standard = slot == 0 ? STD_INPUT_HANDLE
                               : slot == 1 ? STD_OUTPUT_HANDLE
                                           : STD_ERROR_HANDLE;
        HANDLE source = ::GetStdHandle(standard);
        if (source == nullptr || source == INVALID_HANDLE_VALUE)
          handle = open_null(slot == 0);
        else
          handle = duplicate_inheritable(source,
                                         "duplicate inherited child stdio");
        break;
      }
      case ProcessStdioKind::kNull:
        handle = open_null(slot == 0);
        break;
      case ProcessStdioKind::kFile:
        handle = duplicate_inheritable(
            as_handle(config.file_handle()),
            "duplicate child stdio file handle");
        break;
      case ProcessStdioKind::kPipe:
        return prepare_pipe(slot, options);
      case ProcessStdioKind::kMergeStdout:
        child[slot] = child[1];
        return Status{};
    }
    if (!handle.has_value()) return std::move(handle).error();
    if (!remember(*handle)) {
      (void)::CloseHandle(*handle);
      return Status::ResourceExhausted("too many child stdio handles");
    }
    child[slot] = *handle;
    return Status{};
  }

  Status prepare(const ProcessStartOptions& options) noexcept {
    Status status = prepare_slot(0, options.stdin_stream, options);
    if (!status.ok()) return status;
    status = prepare_slot(1, options.stdout_stream, options);
    if (!status.ok()) return status;
    return prepare_slot(2, options.stderr_stream, options);
  }

  Status build_handle_list(Scratch<HANDLE, 6>& handles) noexcept {
    for (HANDLE value : child) {
      bool duplicate = false;
      for (HANDLE existing : handles)
        if (existing == value) duplicate = true;
      if (!duplicate && !handles.try_push_back(value))
        return Status::ResourceExhausted(
            "child inherited-handle list allocation failed");
    }
    return Status{};
  }

  ProcessState* state;
  HANDLE child[3] = {INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE,
                     INVALID_HANDLE_VALUE};
  HANDLE owned[6] = {INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE,
                     INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE,
                     INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE};
  std::size_t owned_count = 0;
};

VOID CALLBACK process_wait_callback(PVOID raw, BOOLEAN) noexcept {
  auto* state = static_cast<ProcessState*>(raw);
  const BOOL posted = ::PostQueuedCompletionStatus(
      as_handle(state->runtime->native_carrier), 0,
      reinterpret_cast<ULONG_PTR>(state), nullptr);
  HYLE_FATAL_ASSERT_MSG(posted != FALSE,
                        "post managed process completion failed");
}

NativeExit collect_exit(ProcessState* state) noexcept {
  DWORD code = STILL_ACTIVE;
  if (!::GetExitCodeProcess(as_handle(state->native_process), &code))
    return NativeExit{
        win32_status(::GetLastError(), "read managed child exit code")};
  if (code == STILL_ACTIVE)
    return NativeExit{Status::Internal(
        "managed child wait fired before process exit")};
  NativeExit result;
  result.exit_code = code;
  return result;
}

void close_native_process(ProcessState* state) noexcept {
  if (state->native_process >= 0) {
    (void)::CloseHandle(as_handle(state->native_process));
    state->native_process = -1;
  }
  if (state->native_tree >= 0) {
    // KILL_ON_JOB_CLOSE makes a root exit close the whole managed tree and
    // prevents inherited pipe writers from surviving the ProcessExit value.
    (void)::CloseHandle(as_handle(state->native_tree));
    state->native_tree = -1;
  }
}

}  // namespace

Status native_runtime_start(ProcessRuntimeState* state) noexcept {
  HANDLE carrier = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
  if (carrier == nullptr)
    return win32_status(::GetLastError(),
                        "create process completion carrier");
  state->native_carrier = as_native(carrier);
  return Status{};
}

void native_runtime_wake(ProcessRuntimeState* state) noexcept {
  (void)::PostQueuedCompletionStatus(as_handle(state->native_carrier), 0, 0,
                                     nullptr);
}

void native_runtime_close(ProcessRuntimeState* state) noexcept {
  if (state->native_carrier >= 0) {
    (void)::CloseHandle(as_handle(state->native_carrier));
    state->native_carrier = -1;
  }
}

void native_reaper_entry(void* raw) noexcept {
  auto* runtime = static_cast<ProcessRuntimeState*>(raw);
  for (;;) {
    DWORD bytes = 0;
    ULONG_PTR key = 0;
    OVERLAPPED* overlapped = nullptr;
    const BOOL ok = ::GetQueuedCompletionStatus(
        as_handle(runtime->native_carrier), &bytes, &key, &overlapped, INFINITE);
    if (!ok && overlapped == nullptr)
      HYLE_FATAL_ASSERT_MSG(false, "process completion carrier failed");
    if (key == 0) {
      if (runtime->carrier_stop.load(std::memory_order_acquire)) return;
      continue;
    }
    auto* state = reinterpret_cast<ProcessState*>(key);
    // native_register publishes the wait handle while the runtime monitor is
    // held by spawn(). A very short-lived child may queue its completion before
    // RegisterWaitForSingleObject returns, so acquire the same monitor before
    // consuming the handle. Besides removing the data race, this guarantees we
    // unregister the actual wait rather than observing its initial sentinel.
    HANDLE wait = nullptr;
    {
      std::unique_lock lock(runtime->monitor);
      if (state->native_wait >= 0)
        wait = as_handle(std::exchange(state->native_wait, -1));
    }
    if (wait != nullptr) {
      const BOOL unregistered = ::UnregisterWaitEx(wait, INVALID_HANDLE_VALUE);
      HYLE_FATAL_ASSERT_MSG(unregistered != FALSE ||
                                ::GetLastError() == ERROR_IO_PENDING,
                            "unregister process wait failed");
    }
    publish_exit(state, collect_exit(state));
  }
}

Status native_spawn(ProcessRuntimeState* runtime, ProcessState* state,
                    const ProcessStartOptions& options) noexcept {
  auto executable = wide_c_string(runtime->allocator, options.executable);
  if (!executable.has_value()) return std::move(executable).error();
  auto command = build_command_line(runtime->allocator, options);
  if (!command.has_value()) return std::move(command).error();

  Scratch<wchar_t, 512> working_directory(runtime->allocator);
  wchar_t* working_directory_pointer = nullptr;
  if (options.working_directory.has_value()) {
    auto copied = wide_c_string(runtime->allocator,
                                *options.working_directory);
    if (!copied.has_value()) return std::move(copied).error();
    working_directory = std::move(*copied);
    working_directory_pointer = working_directory.data();
  }

  EnvironmentBlock environment(runtime->allocator);
  wchar_t* environment_pointer = nullptr;
  if (options.environment_mode == EnvironmentMode::kReplace ||
      !options.environment_changes.empty()) {
    Status status = environment.build(options);
    if (!status.ok()) return status;
    environment_pointer = environment.data();
  }

  StdioSetup stdio(state);
  Status status = stdio.prepare(options);
  if (!status.ok()) return status;
  Scratch<HANDLE, 6> inherited(runtime->allocator);
  status = stdio.build_handle_list(inherited);
  if (!status.ok()) return status;

  SIZE_T attribute_bytes = 0;
  (void)::InitializeProcThreadAttributeList(nullptr, 1, 0,
                                            &attribute_bytes);
  Scratch<std::byte, 256> attribute_storage(runtime->allocator);
  if (!attribute_storage.try_resize(attribute_bytes))
    return Status::ResourceExhausted(
        "process startup attribute allocation failed");
  auto* attributes = reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(
      attribute_storage.data());
  if (!::InitializeProcThreadAttributeList(attributes, 1, 0,
                                            &attribute_bytes)) {
    return win32_status(::GetLastError(),
                        "initialize process startup attributes");
  }
  if (!::UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
          inherited.data(), inherited.size() * sizeof(HANDLE), nullptr,
          nullptr)) {
    const DWORD error = ::GetLastError();
    ::DeleteProcThreadAttributeList(attributes);
    return win32_status(error, "restrict child inherited handles");
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = stdio.child[0];
  startup.StartupInfo.hStdOutput = stdio.child[1];
  startup.StartupInfo.hStdError = stdio.child[2];
  startup.lpAttributeList = attributes;

  HANDLE job = ::CreateJobObjectW(nullptr, nullptr);
  if (job == nullptr) {
    const DWORD error = ::GetLastError();
    ::DeleteProcThreadAttributeList(attributes);
    return win32_status(error, "create managed child job");
  }
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!::SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits))) {
    const DWORD error = ::GetLastError();
    (void)::CloseHandle(job);
    ::DeleteProcThreadAttributeList(attributes);
    return win32_status(error, "configure managed child job");
  }

  PROCESS_INFORMATION info{};
  wchar_t* application =
      options.executable_search == ExecutableSearch::kExact
          ? executable->data()
          : nullptr;
  const DWORD flags = EXTENDED_STARTUPINFO_PRESENT |
                      CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED;
  const BOOL created = ::CreateProcessW(
      application, command->data(), nullptr, nullptr, TRUE, flags,
      environment_pointer, working_directory_pointer, &startup.StartupInfo,
      &info);
  const DWORD create_error = created ? ERROR_SUCCESS : ::GetLastError();
  ::DeleteProcThreadAttributeList(attributes);
  if (!created) {
    (void)::CloseHandle(job);
    return win32_status(create_error, "spawn managed child process");
  }
  if (!::AssignProcessToJobObject(job, info.hProcess)) {
    const DWORD error = ::GetLastError();
    (void)::TerminateProcess(info.hProcess, kForcedExitCode);
    (void)::WaitForSingleObject(info.hProcess, INFINITE);
    (void)::CloseHandle(info.hThread);
    (void)::CloseHandle(info.hProcess);
    (void)::CloseHandle(job);
    return win32_status(error, "assign managed child to job");
  }
  if (::ResumeThread(info.hThread) == static_cast<DWORD>(-1)) {
    const DWORD error = ::GetLastError();
    (void)::TerminateJobObject(job, kForcedExitCode);
    (void)::WaitForSingleObject(info.hProcess, INFINITE);
    (void)::CloseHandle(info.hThread);
    (void)::CloseHandle(info.hProcess);
    (void)::CloseHandle(job);
    return win32_status(error, "resume managed child process");
  }
  (void)::CloseHandle(info.hThread);
  state->process_id = info.dwProcessId;
  state->native_process = as_native(info.hProcess);
  state->native_tree = as_native(job);
  return Status{};
}

Status native_register(ProcessRuntimeState*, ProcessState* state) noexcept {
  HANDLE wait = nullptr;
  if (!::RegisterWaitForSingleObject(
          &wait, as_handle(state->native_process), &process_wait_callback,
          state, INFINITE, WT_EXECUTEONLYONCE)) {
    return win32_status(::GetLastError(),
                        "register managed child wait");
  }
  state->native_wait = as_native(wait);
  return Status{};
}

void native_abort_and_reap(ProcessState* state) noexcept {
  if (state->native_tree >= 0)
    (void)::TerminateJobObject(as_handle(state->native_tree), kForcedExitCode);
  else if (state->native_process >= 0)
    (void)::TerminateProcess(as_handle(state->native_process), kForcedExitCode);
  if (state->native_process >= 0)
    (void)::WaitForSingleObject(as_handle(state->native_process), INFINITE);
  if (state->native_wait >= 0) {
    (void)::UnregisterWaitEx(as_handle(state->native_wait),
                             INVALID_HANDLE_VALUE);
    state->native_wait = -1;
  }
  close_native_process(state);
}

Status native_request_termination(ProcessState*) noexcept {
  return Status::Unimplemented(
      "Windows has no portable cooperative process termination; use stdin or application protocol");
}

Status native_force_kill_tree(ProcessState* state) noexcept {
  if (state->native_tree < 0)
    return Status::FailedPrecondition("managed child job is closed");
  if (!::TerminateJobObject(as_handle(state->native_tree), kForcedExitCode)) {
    const DWORD error = ::GetLastError();
    if (error != ERROR_ACCESS_DENIED)
      return win32_status(error, "force kill managed child job");
    DWORD code = STILL_ACTIVE;
    if (!::GetExitCodeProcess(as_handle(state->native_process), &code) ||
        code == STILL_ACTIVE) {
      return win32_status(error, "force kill managed child job");
    }
  }
  return Status{};
}

void native_release_after_exit(ProcessState* state) noexcept {
  close_native_process(state);
}

}  // namespace hyle::process::detail
