#include "hyle/platform/thread.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

#include "hyle/platform.h"

// Verified on three platforms: Windows MSVC, WSL Linux (GCC14), and macOS arm64
// (Apple clang) -- spawn/join, name, affinity, stack size, and priority all
// exercised by the real-machine test runs.

namespace hyle::platform {
namespace {

// Heap-passed trampoline payload: the user entry + ctx, plus a NUL-terminated
// name copy (string_view is not NUL-guaranteed -- we honor the C-API boundary
// rule by copying) and optional affinity. Applied from inside the new thread so
// one code path works across OSes (Linux/mac name+affinity APIs act on self).
//
// Thread-name buffer capacity (incl. NUL). 64 comfortably covers every OS limit
// (Linux pthread_setname_np caps at 16; macOS/Windows are looser); used as the
// SINGLE source for both the byte buffer and the wide-conversion count so they
// can never drift apart into an overflow.
inline constexpr int kThreadNameCap = 64;

struct ThreadStart {
  void (*entry)(void*);
  void* ctx;
  char name[kThreadNameCap];
  bool has_name;
  std::size_t affinity_count;
  bool has_priority;
  ThreadPriority priority;
  bool require_config;
  bool run_entry;
  // Used only by required setup, where creator and new thread rendezvous. The
  // best-effort fast path keeps the original single-owner delete (no ref RMW).
  std::atomic<std::uint32_t> refs{2};
  std::atomic<std::uint32_t> setup_state{0};
  Status setup_status;
};

unsigned* affinity_cpus(ThreadStart* start) noexcept {
  return reinterpret_cast<unsigned*>(start + 1);
}

const unsigned* affinity_cpus(const ThreadStart* start) noexcept {
  return reinterpret_cast<const unsigned*>(start + 1);
}

void destroy_start(ThreadStart* start) noexcept {
  start->~ThreadStart();
  ::operator delete(start);
}

ThreadStart* make_start(void (*entry)(void*), void* ctx, const ThreadOpts& o) {
  if (o.affinity.size() >
      (std::numeric_limits<std::size_t>::max() - sizeof(ThreadStart)) /
          sizeof(unsigned)) {
    return nullptr;
  }
  const std::size_t bytes =
      sizeof(ThreadStart) + o.affinity.size() * sizeof(unsigned);
  void* storage = ::operator new(bytes, std::nothrow);
  auto* s = storage != nullptr ? ::new (storage) ThreadStart{} : nullptr;
  if (s == nullptr) return nullptr;
  s->entry = entry;
  s->ctx = ctx;
  s->has_name = !o.name.empty();
  if (s->has_name) {
    std::size_t n = o.name.size() < sizeof(s->name) - 1 ? o.name.size()
                                                        : sizeof(s->name) - 1;
    std::memcpy(s->name, o.name.data(), n);
    s->name[n] = '\0';
  }
  s->affinity_count = o.affinity.size();
  for (std::size_t i = 0; i < s->affinity_count; ++i) {
    affinity_cpus(s)[i] = o.affinity[i];
  }
  s->has_priority = o.priority.has_value();
  s->priority = o.priority.value_or(ThreadPriority::kNormal);
  s->require_config = o.config_policy == ThreadConfigPolicy::kRequired;
  s->run_entry = true;
  return s;
}

bool valid_config_policy(ThreadConfigPolicy policy) noexcept {
  return policy == ThreadConfigPolicy::kBestEffort ||
         policy == ThreadConfigPolicy::kRequired;
}

void release_start(ThreadStart* s) noexcept {
  if (s->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy_start(s);
}

void publish_required_setup(ThreadStart* s, Status status) noexcept {
  if (!s->require_config) {
    status.IgnoreError();
    return;
  }
  s->run_entry = status.ok();
  s->setup_status = std::move(status);
  s->setup_state.store(1, std::memory_order_release);
  detail::wake_one(&s->setup_state);
  while (s->setup_state.load(std::memory_order_acquire) != 2) {
    detail::wait_on_address(&s->setup_state, 1);
  }
}

Status await_required_setup(ThreadStart* s) noexcept {
  while (s->setup_state.load(std::memory_order_acquire) != 1) {
    detail::wait_on_address(&s->setup_state, 0);
  }
  Status status = std::move(s->setup_status);
  s->setup_state.store(2, std::memory_order_release);
  detail::wake_one(&s->setup_state);
  return status;
}

}  // namespace
}  // namespace hyle::platform

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>

namespace hyle::platform {
namespace {

const char* win32_error_name(unsigned long code) noexcept {
  switch (code) {
    case ERROR_NOT_ENOUGH_MEMORY: return "ERROR_NOT_ENOUGH_MEMORY";
    case ERROR_OUTOFMEMORY: return "ERROR_OUTOFMEMORY";
    case ERROR_MAX_THRDS_REACHED: return "ERROR_MAX_THRDS_REACHED";
    case ERROR_ACCESS_DENIED: return "ERROR_ACCESS_DENIED";
    case ERROR_INVALID_PARAMETER: return "ERROR_INVALID_PARAMETER";
    default: return "WIN32_ERROR";
  }
}

Status thread_os_error(const char* operation, unsigned long code) noexcept {
  char description[192]{};
  DWORD length = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
      static_cast<DWORD>(code), 0, description,
      static_cast<DWORD>(sizeof(description)), nullptr);
  while (length != 0 &&
         (description[length - 1] == '\r' || description[length - 1] == '\n')) {
    description[--length] = '\0';
  }

  char message[320];
  std::snprintf(message, sizeof(message),
                "windows %s failed: %s (%lu)%s%s", operation,
                win32_error_name(code), code, length == 0 ? "" : ": ",
                length == 0 ? "" : description);
  if (code == ERROR_NOT_ENOUGH_MEMORY || code == ERROR_OUTOFMEMORY ||
      code == ERROR_MAX_THRDS_REACHED) {
    return Status::FromNative(StatusCode::ResourceExhausted,
                              {NativeErrorDomain::Win32, code}, message);
  }
  if (code == ERROR_ACCESS_DENIED)
    return Status::FromNative(StatusCode::PermissionDenied,
                              {NativeErrorDomain::Win32, code}, message);
  if (code == ERROR_INVALID_PARAMETER)
    return Status::FromNative(StatusCode::InvalidArgument,
                              {NativeErrorDomain::Win32, code}, message);
  return Status::FromNative(StatusCode::Internal,
                            {NativeErrorDomain::Win32, code}, message);
}

Status set_name_impl(std::string_view name,
                     HANDLE thread = ::GetCurrentThread()) noexcept {
  char utf8[kThreadNameCap];
  std::size_t n = name.size() < sizeof(utf8) - 1 ? name.size()
                                                  : sizeof(utf8) - 1;
  std::memcpy(utf8, name.data(), n);
  utf8[n] = '\0';
  wchar_t wide[kThreadNameCap];
  int count = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1,
                                    wide, kThreadNameCap);
  if (count == 0) {
    return thread_os_error("MultiByteToWideChar(thread name)",
                           ::GetLastError());
  }
  const HRESULT result = ::SetThreadDescription(thread, wide);
  if (FAILED(result)) {
    char message[128];
    std::snprintf(message, sizeof(message),
                  "windows SetThreadDescription failed: HRESULT 0x%08lx",
                  static_cast<unsigned long>(result));
    return Status::Internal(message);
  }
  return Status::Ok();
}

Status set_affinity_impl(const unsigned* cpus, std::size_t cpu_count,
                         HANDLE thread = ::GetCurrentThread()) noexcept {
  if (cpu_count == 0) {
    return Status::InvalidArgument("thread affinity CPU set is empty");
  }
  const WORD groups = ::GetActiveProcessorGroupCount();
  GROUP_AFFINITY affinity{};
  bool have_group = false;
  for (std::size_t i = 0; i < cpu_count; ++i) {
    unsigned cpu = cpus[i];
    bool found = false;
    for (WORD group = 0; group < groups; ++group) {
      const DWORD count = ::GetActiveProcessorCount(group);
      if (cpu < count) {
        if (cpu >= sizeof(KAFFINITY) * 8u) {
          return Status::OutOfRange("thread affinity CPU exceeds group mask");
        }
        if (have_group && affinity.Group != group) {
          return Status::InvalidArgument(
              "thread affinity CPUs cross Windows processor groups");
        }
        affinity.Group = group;
        affinity.Mask |= static_cast<KAFFINITY>(1) << cpu;
        have_group = true;
        found = true;
        break;
      }
      cpu -= count;
    }
    if (!found) return Status::OutOfRange("thread affinity CPU is not online");
  }
  if (!::SetThreadGroupAffinity(thread, &affinity, nullptr)) {
    return thread_os_error("SetThreadGroupAffinity", ::GetLastError());
  }
  return Status::Ok();
}

Status set_priority_impl(ThreadPriority priority,
                         HANDLE thread = ::GetCurrentThread()) noexcept {
  int native = THREAD_PRIORITY_NORMAL;
  switch (priority) {
    case ThreadPriority::kLowest:  native = THREAD_PRIORITY_LOWEST; break;
    case ThreadPriority::kLow:     native = THREAD_PRIORITY_BELOW_NORMAL; break;
    case ThreadPriority::kNormal:  native = THREAD_PRIORITY_NORMAL; break;
    case ThreadPriority::kHigh:    native = THREAD_PRIORITY_ABOVE_NORMAL; break;
    case ThreadPriority::kHighest: native = THREAD_PRIORITY_HIGHEST; break;
    default: return Status::InvalidArgument("thread priority is invalid");
  }
  if (!::SetThreadPriority(thread, native)) {
    return thread_os_error("SetThreadPriority", ::GetLastError());
  }
  return Status::Ok();
}

Status apply_start_config(const ThreadStart& start,
                          HANDLE thread = ::GetCurrentThread()) noexcept {
  Status status;
  if (start.has_name) status.Update(set_name_impl(start.name, thread));
  if (start.affinity_count != 0) {
    status.Update(set_affinity_impl(affinity_cpus(&start),
                                    start.affinity_count, thread));
  }
  if (start.has_priority)
    status.Update(set_priority_impl(start.priority, thread));
  return status;
}

DWORD WINAPI thread_trampoline(LPVOID arg) noexcept {
  auto* s = static_cast<ThreadStart*>(arg);
  bool run_entry = true;
  if (s->require_config) {
    // The OS thread is fully created (including sanitizer/runtime TLS init), but
    // user code cannot start until the creator applies config and opens gate 1.
    while (s->setup_state.load(std::memory_order_acquire) == 0) {
      detail::wait_on_address(&s->setup_state, 0);
    }
    run_entry = s->setup_state.load(std::memory_order_relaxed) == 1;
  } else {
    apply_start_config(*s).IgnoreError();
  }
  void (*entry)(void*) = s->entry;
  void* ctx = s->ctx;
  if (s->require_config)
    release_start(s);
  else
    destroy_start(s);
  if (run_entry) entry(ctx);
  return 0;
}

}  // namespace

Result<Thread> Thread::spawn(void (*entry)(void*), void* ctx, ThreadOpts opts) {
  if (entry == nullptr) {
    return std::unexpected(Status::InvalidArgument("thread entry is null"));
  }
  if (!valid_config_policy(opts.config_policy)) {
    return std::unexpected(
        Status::InvalidArgument("thread config policy is invalid"));
  }
  ThreadStart* s = make_start(entry, ctx, opts);
  if (s == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("thread start allocation failed"));
  }
  SIZE_T stack = static_cast<SIZE_T>(opts.stack_size.value_or(0));
  const bool required = opts.config_policy == ThreadConfigPolicy::kRequired;
  const DWORD flags = stack ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0;
  HANDLE h =
      ::CreateThread(nullptr, stack, &thread_trampoline, s, flags, nullptr);
  if (h == nullptr) {
    const DWORD code = ::GetLastError();
    destroy_start(s);
    return std::unexpected(thread_os_error("CreateThread", code));
  }
  if (required) {
    Status setup = apply_start_config(*s, h);
    s->setup_state.store(setup.ok() ? 1u : 2u, std::memory_order_release);
    detail::wake_one(&s->setup_state);
    release_start(s);
    if (!setup.ok()) {
      HYLE_FATAL_ASSERT(::WaitForSingleObject(h, INFINITE) == WAIT_OBJECT_0);
      HYLE_FATAL_ASSERT(::CloseHandle(h) != 0);
      return std::unexpected(std::move(setup).WithContext("thread setup"));
    }
  }
  Thread t;
  t.handle_ = reinterpret_cast<std::uintptr_t>(h);
  return t;
}

void Thread::join() {
  HYLE_FATAL_ASSERT(joinable());  // join on a non-joinable Thread is a contract bug
  HANDLE h = reinterpret_cast<HANDLE>(handle_);
  HYLE_FATAL_ASSERT(::WaitForSingleObject(h, INFINITE) == WAIT_OBJECT_0);
  HYLE_FATAL_ASSERT(::CloseHandle(h) != 0);
  handle_ = 0;
}

unsigned hardware_concurrency() noexcept {
  const DWORD all = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  if (all != 0) return static_cast<unsigned>(all);
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  return si.dwNumberOfProcessors ? si.dwNumberOfProcessors : 1u;
}

namespace this_thread {
std::uint64_t get_id() noexcept { return ::GetCurrentThreadId(); }
void yield() noexcept { ::SwitchToThread(); }
Status set_name(std::string_view name) noexcept { return set_name_impl(name); }
Status set_affinity(CpuAffinity affinity) noexcept {
  if (affinity.empty()) {
    return Status::InvalidArgument("thread affinity CPU set is empty");
  }
  return set_affinity_impl(affinity.data(), affinity.size());
}
Status set_priority(ThreadPriority priority) noexcept {
  return set_priority_impl(priority);
}
}  // namespace this_thread

}  // namespace hyle::platform

#else  // ---- POSIX (Linux / macOS) -- verified on WSL Linux + macOS arm64 ------
#  include <pthread.h>
#  include <sched.h>
#  include <sys/resource.h>  // setpriority (best-effort thread nice)
#  include <unistd.h>
#  if defined(HYLE_OS_LINUX)
#    include <sys/syscall.h>
#  elif defined(HYLE_OS_MACOS)
#    include <pthread/qos.h>
#  endif

namespace hyle::platform {
namespace {

const char* posix_error_name(int code) noexcept {
  switch (code) {
    case EAGAIN: return "EAGAIN";
    case ENOMEM: return "ENOMEM";
    case EINVAL: return "EINVAL";
    case EPERM: return "EPERM";
    case EACCES: return "EACCES";
    default: return "ERRNO_UNKNOWN";
  }
}

[[maybe_unused]] const char* strerror_text(int result, char* buffer) noexcept {
  return result == 0 ? buffer : nullptr;
}

[[maybe_unused]] const char* strerror_text(char* result, char*) noexcept {
  return result;
}

Status thread_os_error(const char* operation, int code) noexcept {
  char description[192]{};
  const char* text = strerror_text(
      ::strerror_r(code, description, sizeof(description)), description);
  char message[320];
  std::snprintf(message, sizeof(message), "posix %s failed: %s (%d)%s%s",
                operation, posix_error_name(code), code,
                text == nullptr || text[0] == '\0' ? "" : ": ",
                text == nullptr ? "" : text);
  if (code == EAGAIN || code == ENOMEM) {
    return Status::FromNative(StatusCode::ResourceExhausted,
                              {NativeErrorDomain::PosixErrno, code}, message);
  }
  if (code == EINVAL)
    return Status::FromNative(StatusCode::InvalidArgument,
                              {NativeErrorDomain::PosixErrno, code}, message);
  if (code == EPERM || code == EACCES) {
    return Status::FromNative(StatusCode::PermissionDenied,
                              {NativeErrorDomain::PosixErrno, code}, message);
  }
  return Status::FromNative(StatusCode::Internal,
                            {NativeErrorDomain::PosixErrno, code}, message);
}

Status set_name_impl(std::string_view name) noexcept {
  char buffer[kThreadNameCap];
#  if defined(HYLE_OS_LINUX)
  // Linux includes the NUL in its 16-byte limit. Truncate deliberately instead
  // of passing a longer name and silently losing the entire request with ERANGE.
  constexpr std::size_t effective_cap = 16;
#  else
  constexpr std::size_t effective_cap = kThreadNameCap;
#  endif
  std::size_t n = name.size() < effective_cap - 1 ? name.size()
                                                    : effective_cap - 1;
  std::memcpy(buffer, name.data(), n);
  buffer[n] = '\0';
#  if defined(HYLE_OS_MACOS)
  const int rc = ::pthread_setname_np(buffer);
#  else
  const int rc = ::pthread_setname_np(::pthread_self(), buffer);
#  endif
  return rc == 0 ? Status::Ok() : thread_os_error("pthread_setname_np", rc);
}

Status set_affinity_impl(const unsigned* cpus, std::size_t cpu_count) noexcept {
  if (cpu_count == 0) {
    return Status::InvalidArgument("thread affinity CPU set is empty");
  }
#  if defined(HYLE_OS_LINUX)
  // Match available_cpu_ids()'s explicit dynamic-mask ceiling. This prevents a
  // bogus UINT_MAX request from allocating hundreds of MiB without assuming
  // configured CPU ids are dense (hotplug systems may have holes).
  unsigned max_cpu = 0;
  for (std::size_t i = 0; i < cpu_count; ++i) {
    if (cpus[i] >= (std::size_t{1} << 20)) {
      return Status::OutOfRange(
          "thread affinity CPU exceeds supported id space");
    }
    if (cpus[i] > max_cpu) max_cpu = cpus[i];
  }
#    if defined(CPU_ALLOC)
  const std::size_t count = static_cast<std::size_t>(max_cpu) + 1;
  cpu_set_t* set = CPU_ALLOC(count);
  if (set == nullptr) {
    return Status::ResourceExhausted("thread affinity mask allocation failed");
  }
  const std::size_t bytes = CPU_ALLOC_SIZE(count);
  CPU_ZERO_S(bytes, set);
  for (std::size_t i = 0; i < cpu_count; ++i) {
    CPU_SET_S(cpus[i], bytes, set);
  }
  const int rc = ::pthread_setaffinity_np(::pthread_self(), bytes, set);
  CPU_FREE(set);
#    else
  if (max_cpu >= CPU_SETSIZE) {
    return Status::OutOfRange("thread affinity CPU exceeds cpu_set_t");
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (std::size_t i = 0; i < cpu_count; ++i) CPU_SET(cpus[i], &set);
  const int rc =
      ::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set);
#    endif
  return rc == 0 ? Status::Ok()
                 : thread_os_error("pthread_setaffinity_np", rc);
#  else
  (void)cpus;
  return Status::Unimplemented(
      "hard thread affinity is unavailable on macOS");
#  endif
}

Status set_priority_impl(ThreadPriority priority) noexcept {
#  if defined(HYLE_OS_LINUX)
  int nice_value = 0;
  switch (priority) {
    case ThreadPriority::kLowest:  nice_value = 10; break;
    case ThreadPriority::kLow:     nice_value = 5; break;
    case ThreadPriority::kNormal:  nice_value = 0; break;
    case ThreadPriority::kHigh:    nice_value = -5; break;
    case ThreadPriority::kHighest: nice_value = -10; break;
    default: return Status::InvalidArgument("thread priority is invalid");
  }
  errno = 0;
  if (::setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
    return thread_os_error("setpriority", errno);
  }
  return Status::Ok();
#  else
  qos_class_t qos = QOS_CLASS_DEFAULT;
  switch (priority) {
    case ThreadPriority::kLowest:  qos = QOS_CLASS_BACKGROUND; break;
    case ThreadPriority::kLow:     qos = QOS_CLASS_UTILITY; break;
    case ThreadPriority::kNormal:  qos = QOS_CLASS_DEFAULT; break;
    case ThreadPriority::kHigh:    qos = QOS_CLASS_USER_INITIATED; break;
    case ThreadPriority::kHighest: qos = QOS_CLASS_USER_INTERACTIVE; break;
    default: return Status::InvalidArgument("thread priority is invalid");
  }
  const int rc = ::pthread_set_qos_class_self_np(qos, 0);
  return rc == 0 ? Status::Ok()
                 : thread_os_error("pthread_set_qos_class_self_np", rc);
#  endif
}

Status apply_start_config(const ThreadStart& start) noexcept {
  Status status;
  if (start.has_name) status.Update(set_name_impl(start.name));
  if (start.affinity_count != 0) {
    status.Update(
        set_affinity_impl(affinity_cpus(&start), start.affinity_count));
  }
  if (start.has_priority) status.Update(set_priority_impl(start.priority));
  return status;
}

void* thread_trampoline(void* arg) noexcept {
  auto* s = static_cast<ThreadStart*>(arg);
  publish_required_setup(s, apply_start_config(*s));
  void (*entry)(void*) = s->entry;
  void* ctx = s->ctx;
  const bool run_entry = s->run_entry;
  const bool required = s->require_config;
  if (required)
    release_start(s);
  else
    destroy_start(s);
  if (run_entry) entry(ctx);
  return nullptr;
}

}  // namespace

Result<Thread> Thread::spawn(void (*entry)(void*), void* ctx, ThreadOpts opts) {
  if (entry == nullptr) {
    return std::unexpected(Status::InvalidArgument("thread entry is null"));
  }
  if (!valid_config_policy(opts.config_policy)) {
    return std::unexpected(
        Status::InvalidArgument("thread config policy is invalid"));
  }
  ThreadStart* s = make_start(entry, ctx, opts);
  if (s == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("thread start allocation failed"));
  }
  pthread_attr_t attr;
  const bool required = opts.config_policy == ThreadConfigPolicy::kRequired;
  int rc = ::pthread_attr_init(&attr);
  if (rc != 0) {
    destroy_start(s);
    return std::unexpected(thread_os_error("pthread_attr_init", rc));
  }
  if (opts.stack_size) {
    rc = ::pthread_attr_setstacksize(&attr, *opts.stack_size);
    if (rc != 0) {
      ::pthread_attr_destroy(&attr);
      destroy_start(s);
      return std::unexpected(
          thread_os_error("pthread_attr_setstacksize", rc));
    }
  }
  pthread_t tid;
  rc = ::pthread_create(&tid, &attr, &thread_trampoline, s);
  ::pthread_attr_destroy(&attr);
  if (rc != 0) {
    destroy_start(s);
    return std::unexpected(thread_os_error("pthread_create", rc));
  }
  if (required) {
    Status setup = await_required_setup(s);
    release_start(s);
    if (!setup.ok()) {
      HYLE_FATAL_ASSERT(::pthread_join(tid, nullptr) == 0);
      return std::unexpected(std::move(setup).WithContext("thread setup"));
    }
  }
  Thread t;
  // pthread_t is an integer on glibc but an opaque pointer on macOS/BSD, so no single
  // cast is well-formed on both (static_cast rejects the pointer form, reinterpret_cast
  // rejects the integer form, and if constexpr still type-checks the untaken branch).
  // A byte copy sidesteps the representation difference; both are pointer-sized.
  static_assert(sizeof(pthread_t) <= sizeof(std::uintptr_t));
  t.handle_ = 0;
  std::memcpy(&t.handle_, &tid, sizeof(tid));
  return t;
}

void Thread::join() {
  HYLE_FATAL_ASSERT(joinable());  // join on a non-joinable Thread is a contract bug
  pthread_t tid{};
  std::memcpy(&tid, &handle_, sizeof(tid));
  HYLE_FATAL_ASSERT(::pthread_join(tid, nullptr) == 0);
  handle_ = 0;
}

unsigned hardware_concurrency() noexcept {
  long n = ::sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? static_cast<unsigned>(n) : 1u;
}

namespace this_thread {
std::uint64_t get_id() noexcept {
  // Cache the OS thread id in thread_local storage: it is INVARIANT for a thread's
  // lifetime, so re-querying the kernel every call is pure waste. folly (ThreadId.cpp
  // getOSThreadID) and abseil (sysinfo.cc GetCachedTID) both do this -- folly measured
  // ~275ns uncached -> ~2ns cached. 0 is never a valid tid, so it doubles as the
  // "not yet fetched" sentinel (lazy init on first call per thread).
  // FORK CAVEAT: a value cached in the parent would be stale in a fork() child (new
  // tid). HYLE ships NO fork/spawn (process mgmt is out of scope -- clay-asset-liability
  // §进程), so there is no atfork reset; if fork support is ever added, this needs a
  // fork epoch (folly's approach) or a pthread_atfork handler to clear the cache.
  static thread_local std::uint64_t cached = 0;
  if (cached != 0) return cached;
#  if defined(HYLE_OS_LINUX)
  cached = static_cast<std::uint64_t>(::syscall(SYS_gettid));
#  else
  std::uint64_t tid = 0;
  ::pthread_threadid_np(nullptr, &tid);
  cached = tid;
#  endif
  return cached;
}
void yield() noexcept { ::sched_yield(); }
Status set_name(std::string_view name) noexcept { return set_name_impl(name); }
Status set_affinity(CpuAffinity affinity) noexcept {
  if (affinity.empty()) {
    return Status::InvalidArgument("thread affinity CPU set is empty");
  }
  return set_affinity_impl(affinity.data(), affinity.size());
}
Status set_priority(ThreadPriority priority) noexcept {
  return set_priority_impl(priority);
}
}  // namespace this_thread

}  // namespace hyle::platform
#endif

// ---- shared (platform-agnostic) tail ----------------------------------------
namespace hyle::platform {

Thread& Thread::operator=(Thread&& other) noexcept {
  if (this != &other) {
    // Overwriting a still-joinable thread is a leak/bug -> fail-fast.
    HYLE_FATAL_ASSERT(!joinable());
    handle_ = other.handle_;
    other.handle_ = 0;
  }
  return *this;
}

Thread::~Thread() {
  // A joinable thread destroyed without join() is a bug; surface it loudly
  // (charter fail-fast, mirrors std::thread's terminate).
  HYLE_FATAL_ASSERT(!joinable());
}

}  // namespace hyle::platform
