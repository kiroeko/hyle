#pragma once

#include <atomic>
#include <cstdint>

#include "hyle/core/compiler.h"
#include "hyle/platform/thread_visibility.h"
#include "hyle/time/clock.h"

namespace hyle::platform {

// ---- low-level address wait (futex equivalent) ------------------------------
// Block the calling thread while *addr == expected, until another thread changes
// the word and wakes it (or a spurious wake occurs). This is the minimal common
// denominator across futex(Linux) / WaitOnAddress(Windows) / os_sync(macOS).
//
// Contract (matches futex):
//  - The compare-and-block is atomic w.r.t. wakes: a wake that happens after the
//    value is changed will not be missed by a parker about to sleep.
//  - Spurious wakeups ARE allowed; callers MUST re-check their condition in a
//    loop. (The Parker below does exactly that; prefer Parker for app code.)
namespace detail {
HYLE_PLATFORM_THREAD_API void wait_on_address(const std::atomic<std::uint32_t>* addr,
                                       std::uint32_t expected) noexcept;
// Timed variant: block until woken or `timeout_ns` elapses. Returns false iff it
// returned because the timeout expired (best-effort: the OS may wake slightly
// late, and a spurious wake still returns true -- callers loop and re-check).
// timeout_ns == 0 is a non-blocking poll. Backs an event loop's timer-queue
// deadline: the owner computes the next-expiry duration and hands it to
// futex/WaitOnAddress so the worker sleeps without periodic polling.
HYLE_PLATFORM_THREAD_API bool wait_on_address_for(
    const std::atomic<std::uint32_t>* addr, std::uint32_t expected,
    std::uint64_t timeout_ns) noexcept;
HYLE_PLATFORM_THREAD_API void wake_one(const std::atomic<std::uint32_t>* addr) noexcept;
HYLE_PLATFORM_THREAD_API void wake_all(const std::atomic<std::uint32_t>* addr) noexcept;
}  // namespace detail

// ---- Parker / Unparker (portable, built on wait_on_address) -----------------
// One Parker per worker thread. park() blocks until a matching unpark(); the
// three-state machine makes unpark-before-park safe (the token is remembered)
// and absorbs spurious wakeups so callers get clean "blocked until told"
// semantics. Model mirrors Rust std / tokio's parking. The portable layer is
// written ONCE here; only wait_on_address has per-OS backends.
//
// API macro is on the methods, not the class: park/unpark live in the DLL, but
// exporting the whole class would drag the std::atomic member across the seam
// (MSVC C4251). Method-level export keeps the layout client-side and clean.
class Parker {
 public:
  Parker() noexcept = default;
  Parker(const Parker&) = delete;
  Parker& operator=(const Parker&) = delete;
  Parker(Parker&&) = delete;
  Parker& operator=(Parker&&) = delete;

  // Block until a token is available, then consume it. Re-checks state in a
  // loop, so a spurious low-level wake just re-sleeps (never a false return).
  HYLE_PLATFORM_THREAD_API void park() noexcept;

  // Timed park: block until a token is available OR `timeout` elapses.
  // Returns true if a token was consumed, false otherwise (timeout OR a spurious
  // low-level wake). SINGLE-SHOT: unlike park(), it does NOT re-loop on a
  // spurious wake -- it may return false early, and the caller is expected to
  // re-check its own condition and re-park with a fresh deadline (mirrors Rust
  // std's park_timeout). A non-positive timeout is a non-blocking check. This is the
  // worker's "sleep until next timer" call: its timer queue supplies the
  // deadline, and the worker's loop re-checks + recomputes the bound each pass.
  HYLE_PLATFORM_THREAD_API bool park_for(time::Duration timeout) noexcept;

  // Make a token available; wake a parked thread if one is waiting. Idempotent:
  // repeated unparks without an intervening park collapse to a single token.
  HYLE_PLATFORM_THREAD_API void unpark() noexcept;

 private:
  // State values chosen for the proven futex-Parker scheme (Rust std):
  //   EMPTY(0): no token, not parked.  NOTIFIED(1): a token is available.
  //   PARKED(0xFFFFFFFF): a thread is asleep on this word.
  // park does fetch_sub(1): NOTIFIED->EMPTY (consume & return) or EMPTY->PARKED
  // (then wait while ==PARKED). unpark swaps in NOTIFIED and wakes if was PARKED.
  enum : std::uint32_t { kEmpty = 0, kNotified = 1, kParked = 0xFFFFFFFFu };
  std::atomic<std::uint32_t> state_{kEmpty};
};

// Handle that unparks a specific Parker. Cheap to copy/pass to the waker spine.
class Unparker {
 public:
  explicit Unparker(Parker* p) noexcept : parker_(p) {}
  void unpark() const noexcept { parker_->unpark(); }
  Parker* get() const noexcept { return parker_; }

 private:
  Parker* parker_;
};

}  // namespace hyle::platform
