#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/platform/wait.h"  // HYLE_PLATFORM_THREAD_API

namespace hyle::platform {

// Options a raw std::thread cannot express but the work-stealing scheduler needs
// (sched-design §3.3 per-core pinning; std::thread can't even set stack size).
// This is the natural C++ surface. `name` is a view: spawn() copies it synchronously, so the
// caller only needs to keep the backing buffer alive through the call.
// Portable 5-level priority. OS priority models diverge wildly (Windows discrete
// levels vs POSIX nice vs SCHED_FIFO), so we expose a small portable ladder and
// map it per-OS. Best-effort: a level the OS won't grant without privilege is a
// soft no-op (the thread still runs at default), never a failure.
enum class ThreadPriority : std::int8_t {
  kLowest = -2, kLow = -1, kNormal = 0, kHigh = 1, kHighest = 2,
};

// What spawn() should do when an explicitly requested name, affinity or
// priority cannot be applied. Best-effort is the portable default (notably,
// macOS has no public hard CPU-pinning API). Required makes setup synchronous:
// spawn returns the setup Status and never calls entry on failure.
enum class ThreadConfigPolicy : std::uint8_t {
  kBestEffort = 0,
  kRequired = 1,
};

// Portable hard-affinity shape: no restriction, one logical CPU, or a set of
// logical CPUs on which the thread may migrate. CPU ids come from
// available_cpu_ids() and may be sparse. A multi-CPU affinity is a view;
// Thread::spawn() copies it before returning, so its backing storage only needs
// to remain alive through the call. Deliberately no initializer_list overload:
// storing a temporary list in ThreadOpts would dangle before spawn().
class CpuAffinity {
 public:
  CpuAffinity() noexcept = default;
  CpuAffinity(unsigned cpu) noexcept : single_cpu_(cpu), count_(1) {}
  CpuAffinity(std::span<const unsigned> cpus) noexcept
      : cpus_(cpus.data()), count_(cpus.size()) {}

  bool empty() const noexcept { return count_ == 0; }
  std::size_t size() const noexcept { return count_; }
  const unsigned* data() const noexcept {
    return cpus_ != nullptr ? cpus_ : (count_ != 0 ? &single_cpu_ : nullptr);
  }
  unsigned operator[](std::size_t index) const noexcept {
    return data()[index];
  }

 private:
  const unsigned* cpus_ = nullptr;
  unsigned single_cpu_ = 0;
  std::size_t count_ = 0;
};

// Current-thread execution state which is not a process-wide CPU fact. A zero
// byte width means the architecture/OS does not expose that state; it is not a
// guessed fixed width. Static feature presence remains in cpu_features().
struct ThreadIsaState {
  std::size_t scalable_vector_bytes = 0;  // SVE or RVV VL
  std::size_t streaming_vector_bytes = 0; // SME streaming VL
  bool amx_tile_permitted = false;        // current thread may use XTILEDATA
};

struct ThreadOpts {
  // Natural C++ configuration value. It has no cross-toolchain layout/version
  // contract and is not a foreign-runtime ABI record.
  std::string_view name;                     // <=15 bytes effective on Linux
  std::optional<std::size_t> stack_size;     // bytes; honored via native attrs
  // Empty means unrestricted. Windows CPU ids are flattened across processor
  // groups, but one affinity set must stay within one group. Best-effort on macOS.
  CpuAffinity affinity;
  std::optional<ThreadPriority> priority;    // scheduling priority (best-effort; needs privilege on POSIX)
  ThreadConfigPolicy config_policy = ThreadConfigPolicy::kBestEffort;
};

// Owning thread handle. Move-only. Stores the raw OS handle (no std::thread in
// the public representation), and lets spawn honor stack_size/affinity via
// native CreateThread / pthread_attr. ~Thread fails-fast if still joinable
// (charter: forgotten join is a bug, surface it -- mirrors std::thread).
class HYLE_PLATFORM_THREAD_API Thread {
 public:
  Thread() noexcept = default;
  Thread(Thread&& other) noexcept : handle_(other.handle_) { other.handle_ = 0; }
  Thread& operator=(Thread&& other) noexcept;
  Thread(const Thread&) = delete;
  Thread& operator=(const Thread&) = delete;
  ~Thread();

  // entry(ctx) runs on the new thread; ctx must outlive the thread's execution.
  // Function-pointer + void* (not std::function): allocation-minimal and no
  // type-erasure dispatch. This remains a C++ API, not a foreign-runtime ABI.
  // Returns a structured OS creation failure. With kRequired it also returns
  // configuration failures and guarantees entry was not called.
  static Result<Thread> spawn(void (*entry)(void*), void* ctx, ThreadOpts opts = {});

  // Owning C++ convenience: move a zero-argument callable into the new thread.
  // This is the normal shape for capturing and move-only lambdas. It pays one
  // extra cold-path allocation for the callable; the function-pointer overload
  // above remains the allocation-minimal core used by sched.
  // As with std::thread, an exception escaping the callable terminates the
  // process. A spawn/configuration failure destroys the callable without
  // invoking it and is returned through Result.
  template <class Fn>
    requires std::invocable<std::decay_t<Fn>>
  static Result<Thread> spawn(Fn&& fn, ThreadOpts opts = {}) {
    using Callable = std::decay_t<Fn>;
    using Start = CallableStart<Callable>;
    auto* start = new (std::nothrow) Start{std::forward<Fn>(fn)};
    if (start == nullptr) {
      return std::unexpected(
          Status::ResourceExhausted("thread callable allocation failed"));
    }
    auto result = spawn(&Start::invoke, start, opts);
    if (!result) delete start;
    return result;
  }

  bool joinable() const noexcept { return handle_ != 0; }
  void join();

 private:
  template <class Fn>
  struct CallableStart {
    Fn fn;

    static void invoke(void* ctx) {
      // Own before invoking so stack unwinding also releases the callable on
      // toolchains where exceptions are enabled; Thread's OS trampoline is the
      // uncaught-exception/fail-fast boundary.
      std::unique_ptr<CallableStart> self(
          static_cast<CallableStart*>(ctx));
      std::invoke(std::move(self->fn));
    }
  };

  // HANDLE on Windows, pthread_t on POSIX, packed into uintptr_t. 0 = empty.
  std::uintptr_t handle_ = 0;
};

// ---- small helpers ----------------------------------------------------------

namespace this_thread {
// Opaque OS thread id of the caller (for logging / assertions).
HYLE_PLATFORM_THREAD_API std::uint64_t get_id() noexcept;
// Hint to the OS scheduler to relinquish the rest of this time slice.
HYLE_PLATFORM_THREAD_API void yield() noexcept;
// Configure the calling thread. Unlike best-effort ThreadOpts, these return the
// native failure through the shared error model. Ignoring a failure must be
// explicit with Status::IgnoreError(). affinity uses available_cpu_ids().
HYLE_PLATFORM_THREAD_API Status set_name(std::string_view name) noexcept;
HYLE_PLATFORM_THREAD_API Status set_affinity(CpuAffinity affinity) noexcept;
HYLE_PLATFORM_THREAD_API Status set_priority(ThreadPriority priority) noexcept;

// Query the calling thread's hard-affinity mask using the same flattened CPU
// id space accepted by set_affinity(). Returns the full count and writes at
// most cap ids; dst may be null for a count-only query. macOS has no public
// hard-affinity API and reports Unimplemented.
HYLE_PLATFORM_THREAD_API Result<std::size_t> affinity(unsigned* dst,
                                               std::size_t cap) noexcept;

// Query dynamic ISA state for the calling thread. This does not select a SIMD
// backend or enter SME streaming mode; it only reports the native state needed
// by a dispatcher. Explicitly request Linux AMX tile permission before an AMX
// kernel; unsupported CPUs/OSes report Unimplemented without a fallback.
HYLE_PLATFORM_THREAD_API Result<ThreadIsaState> isa_state() noexcept;
HYLE_PLATFORM_THREAD_API Status request_amx_tile_permission() noexcept;
}  // namespace this_thread

}  // namespace hyle::platform
