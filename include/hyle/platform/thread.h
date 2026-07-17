#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "hyle/error/result.h"
#include "hyle/platform/wait.h"  // HYLE_PLATFORM_API

namespace hyle::platform {

// Options a raw std::thread cannot express but the work-stealing scheduler needs
// (sched-design §3.3 per-core pinning; std::thread can't even set stack size).
// All fields are layout-stable (view + trivially-copyable optionals) so this
// struct is ABI-clean across the DLL seam. `name` is a view: spawn() copies it
// synchronously, the caller owns the backing buffer.
// Portable 5-level priority. OS priority models diverge wildly (Windows discrete
// levels vs POSIX nice vs SCHED_FIFO), so we expose a small portable ladder and
// map it per-OS. Best-effort: a level the OS won't grant without privilege is a
// soft no-op (the thread still runs at default), never a failure.
enum class ThreadPriority : std::int8_t {
  kLowest = -2, kLow = -1, kNormal = 0, kHigh = 1, kHighest = 2,
};

struct ThreadOpts {
  std::string_view name;                     // <=15 bytes effective on Linux
  std::optional<std::size_t> stack_size;     // bytes; honored via native attrs
  std::optional<unsigned> affinity_cpu;      // pin to logical CPU (best-effort on macOS)
  std::optional<ThreadPriority> priority;    // scheduling priority (best-effort; needs privilege on POSIX)
};

// Owning thread handle. Move-only. Stores the raw OS handle (no std::thread in
// the public surface): ABI-clean and lets spawn honor stack_size/affinity via
// native CreateThread / pthread_attr. ~Thread fails-fast if still joinable
// (charter: forgotten join is a bug, surface it -- mirrors std::thread).
class HYLE_PLATFORM_API Thread {
 public:
  Thread() noexcept = default;
  Thread(Thread&& other) noexcept : handle_(other.handle_) { other.handle_ = 0; }
  Thread& operator=(Thread&& other) noexcept;
  HYLE_DISABLE_COPY(Thread);
  ~Thread();

  // entry(ctx) runs on the new thread; ctx must outlive the thread's execution.
  // Function-pointer + void* (not std::function): ABI-clean + zero type erasure.
  // Returns Internal on OS spawn failure.
  static Result<Thread> spawn(void (*entry)(void*), void* ctx, ThreadOpts opts = {});

  bool joinable() const noexcept { return handle_ != 0; }
  void join();

 private:
  // HANDLE on Windows, pthread_t on POSIX, packed into uintptr_t. 0 = empty.
  std::uintptr_t handle_ = 0;
};

// ---- small helpers ----------------------------------------------------------

// Logical core count (>=1; 0 from the OS is normalized to 1).
HYLE_PLATFORM_API unsigned hardware_concurrency() noexcept;

namespace this_thread {
// Opaque OS thread id of the caller (for logging / assertions).
HYLE_PLATFORM_API std::uint64_t get_id() noexcept;
// Hint to the OS scheduler to relinquish the rest of this time slice.
HYLE_PLATFORM_API void yield() noexcept;
// Set the calling thread's name (best-effort; see ThreadOpts::name limits).
HYLE_PLATFORM_API void set_name(std::string_view name) noexcept;
}  // namespace this_thread

}  // namespace hyle::platform
