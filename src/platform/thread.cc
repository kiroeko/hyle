#include "hyle/platform/thread.h"

#include <cstring>
#include <type_traits>

#include "hyle/platform.h"

// Only the Windows path is built and tested on this machine. The POSIX path is
// written to spec but NOT yet verified on real hardware/CI.

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
  if (s->has_affinity) {
    ::SetThreadAffinityMask(::GetCurrentThread(),
                            static_cast<DWORD_PTR>(1) << s->cpu);
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

#else  // ---- POSIX (Linux / macOS) -- written to spec, UNVERIFIED on this box -
#  include <pthread.h>
#  include <sched.h>
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
  if (s->has_affinity) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(s->cpu, &set);
    ::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set);
  }
#  endif  // macOS: no hard pinning; affinity is a best-effort no-op here.
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
