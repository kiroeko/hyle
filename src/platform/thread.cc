#include "hyle/platform/thread.h"

#include <cstring>
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
  bool has_affinity;
  unsigned cpu;
  bool has_priority;
  ThreadPriority priority;
};

ThreadStart* make_start(void (*entry)(void*), void* ctx, const ThreadOpts& o) {
  auto* s = new ThreadStart{};
  s->entry = entry;
  s->ctx = ctx;
  s->has_name = !o.name.empty();
  if (s->has_name) {
    std::size_t n = o.name.size() < sizeof(s->name) - 1 ? o.name.size()
                                                        : sizeof(s->name) - 1;
    std::memcpy(s->name, o.name.data(), n);
    s->name[n] = '\0';
  }
  s->has_affinity = o.affinity_cpu.has_value();
  s->cpu = o.affinity_cpu.value_or(0);
  s->has_priority = o.priority.has_value();
  s->priority = o.priority.value_or(ThreadPriority::kNormal);
  return s;
}

}  // namespace
}  // namespace hyle::platform

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>

namespace hyle::platform {
namespace {

DWORD WINAPI thread_trampoline(LPVOID arg) noexcept {
  auto* s = static_cast<ThreadStart*>(arg);
  if (s->has_name) {
    wchar_t wname[kThreadNameCap];
    int wn = ::MultiByteToWideChar(CP_UTF8, 0, s->name, -1, wname,
                                   kThreadNameCap);
    if (wn > 0) ::SetThreadDescription(::GetCurrentThread(), wname);
  }
  // cpu < 64: a single-group affinity mask. cpu >= 64 would be a shift-count
  // overflow (1 << 64 is UB, x86 masks it to bit 0 -> silently pins CPU 0) and
  // anyway needs SetThreadGroupAffinity across a processor group (>64-core box) --
  // not wired here, so an out-of-range pin is a best-effort no-op.
  if (s->has_affinity && s->cpu < 64) {
    ::SetThreadAffinityMask(::GetCurrentThread(),
                            static_cast<DWORD_PTR>(1) << s->cpu);
  }
  if (s->has_priority) {
    int p = THREAD_PRIORITY_NORMAL;
    switch (s->priority) {
      case ThreadPriority::kLowest:  p = THREAD_PRIORITY_LOWEST; break;
      case ThreadPriority::kLow:     p = THREAD_PRIORITY_BELOW_NORMAL; break;
      case ThreadPriority::kNormal:  p = THREAD_PRIORITY_NORMAL; break;
      case ThreadPriority::kHigh:    p = THREAD_PRIORITY_ABOVE_NORMAL; break;
      case ThreadPriority::kHighest: p = THREAD_PRIORITY_HIGHEST; break;
    }
    ::SetThreadPriority(::GetCurrentThread(), p);  // best-effort
  }
  void (*entry)(void*) = s->entry;
  void* ctx = s->ctx;
  delete s;
  entry(ctx);
  return 0;
}

}  // namespace

Result<Thread> Thread::spawn(void (*entry)(void*), void* ctx, ThreadOpts opts) {
  ThreadStart* s = make_start(entry, ctx, opts);
  SIZE_T stack = static_cast<SIZE_T>(opts.stack_size.value_or(0));
  HANDLE h = ::CreateThread(nullptr, stack, &thread_trampoline, s,
                            stack ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0,
                            nullptr);
  if (h == nullptr) {
    delete s;
    return std::unexpected(Status::Internal("CreateThread failed"));
  }
  Thread t;
  t.handle_ = reinterpret_cast<std::uintptr_t>(h);
  return t;
}

void Thread::join() {
  HYLE_FATAL_ASSERT(joinable());  // join on a non-joinable Thread is a contract bug
  HANDLE h = reinterpret_cast<HANDLE>(handle_);
  ::WaitForSingleObject(h, INFINITE);
  ::CloseHandle(h);
  handle_ = 0;
}

unsigned hardware_concurrency() noexcept {
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  return si.dwNumberOfProcessors ? si.dwNumberOfProcessors : 1u;
}

namespace this_thread {
std::uint64_t get_id() noexcept { return ::GetCurrentThreadId(); }
void yield() noexcept { ::SwitchToThread(); }
void set_name(std::string_view name) noexcept {
  char buf[kThreadNameCap];
  std::size_t n = name.size() < sizeof(buf) - 1 ? name.size() : sizeof(buf) - 1;
  std::memcpy(buf, name.data(), n);
  buf[n] = '\0';
  wchar_t wname[kThreadNameCap];
  int wn = ::MultiByteToWideChar(CP_UTF8, 0, buf, -1, wname, kThreadNameCap);
  if (wn > 0) ::SetThreadDescription(::GetCurrentThread(), wname);
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
#  endif

namespace hyle::platform {
namespace {

void* thread_trampoline(void* arg) noexcept {
  auto* s = static_cast<ThreadStart*>(arg);
  if (s->has_name) {
#  if defined(HYLE_OS_MACOS)
    ::pthread_setname_np(s->name);            // self only, no tid arg
#  else
    ::pthread_setname_np(::pthread_self(), s->name);  // <=15 chars effective
#  endif
  }
#  if defined(HYLE_OS_LINUX)
  // cpu < CPU_SETSIZE (1024): fits a fixed cpu_set_t. Beyond it CPU_SET would
  // write out of bounds, so an out-of-range pin is a best-effort no-op (a
  // >1024-core box needs a CPU_ALLOC-sized dynamic set -- not wired here).
  if (s->has_affinity && s->cpu < CPU_SETSIZE) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(s->cpu, &set);
    ::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set);
  }
#  endif  // macOS: no hard pinning; affinity is a best-effort no-op here.
  if (s->has_priority) {
#  if defined(HYLE_OS_LINUX)
    // Linux nice is PER-THREAD (setpriority PRIO_PROCESS/who=0 acts on the calling
    // thread, not the whole process). Best-effort: RAISING priority (negative
    // nice) needs CAP_SYS_NICE / root, so it silently no-ops without it.
    int nice_val = 0;
    switch (s->priority) {
      case ThreadPriority::kLowest:  nice_val = 10; break;
      case ThreadPriority::kLow:     nice_val = 5; break;
      case ThreadPriority::kNormal:  nice_val = 0; break;
      case ThreadPriority::kHigh:    nice_val = -5; break;
      case ThreadPriority::kHighest: nice_val = -10; break;
    }
    (void)::setpriority(PRIO_PROCESS, 0, nice_val);  // best-effort; ignore EACCES
#  else
    // macOS/BSD: setpriority(PRIO_PROCESS) is PROCESS-scoped (a thread is not a
    // PID), so it would drag the WHOLE process -- wrong for a per-thread request.
    // Use pthread_setschedparam, which is genuinely per-thread. Map the ladder
    // across the SCHED_OTHER priority range; best-effort (needs privilege to
    // raise, no-ops otherwise).
    int lo = ::sched_get_priority_min(SCHED_OTHER);
    int hi = ::sched_get_priority_max(SCHED_OTHER);
    int mid = (lo + hi) / 2;
    int prio = mid;
    switch (s->priority) {
      case ThreadPriority::kLowest:  prio = lo; break;
      case ThreadPriority::kLow:     prio = (lo + mid) / 2; break;
      case ThreadPriority::kNormal:  prio = mid; break;
      case ThreadPriority::kHigh:    prio = (mid + hi) / 2; break;
      case ThreadPriority::kHighest: prio = hi; break;
    }
    sched_param param{};
    param.sched_priority = prio;
    (void)::pthread_setschedparam(::pthread_self(), SCHED_OTHER, &param);
#  endif
  }
  void (*entry)(void*) = s->entry;
  void* ctx = s->ctx;
  delete s;
  entry(ctx);
  return nullptr;
}

}  // namespace

Result<Thread> Thread::spawn(void (*entry)(void*), void* ctx, ThreadOpts opts) {
  ThreadStart* s = make_start(entry, ctx, opts);
  pthread_attr_t attr;
  ::pthread_attr_init(&attr);
  if (opts.stack_size) ::pthread_attr_setstacksize(&attr, *opts.stack_size);
  pthread_t tid;
  int rc = ::pthread_create(&tid, &attr, &thread_trampoline, s);
  ::pthread_attr_destroy(&attr);
  if (rc != 0) {
    delete s;
    return std::unexpected(Status::Internal("pthread_create failed"));
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
  ::pthread_join(tid, nullptr);
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
void set_name(std::string_view name) noexcept {
  char buf[kThreadNameCap];
  std::size_t n = name.size() < sizeof(buf) - 1 ? name.size() : sizeof(buf) - 1;
  std::memcpy(buf, name.data(), n);
  buf[n] = '\0';
#  if defined(HYLE_OS_MACOS)
  ::pthread_setname_np(buf);
#  else
  ::pthread_setname_np(::pthread_self(), buf);
#  endif
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
