#pragma once

#include <atomic>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_API
#elif defined(HYLE_PLATFORM_BUILD)
#  define HYLE_PLATFORM_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_API HYLE_IMPORT
#endif

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
HYLE_PLATFORM_API void wait_on_address(const std::atomic<std::uint32_t>* addr,
                                       std::uint32_t expected) noexcept;
HYLE_PLATFORM_API void wake_one(const std::atomic<std::uint32_t>* addr) noexcept;
HYLE_PLATFORM_API void wake_all(const std::atomic<std::uint32_t>* addr) noexcept;
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
  HYLE_DISABLE_COPY_AND_MOVE(Parker);

  // Block until a token is available, then consume it. Re-checks state in a
  // loop, so a spurious low-level wake just re-sleeps (never a false return).
  HYLE_PLATFORM_API void park() noexcept;

  // Make a token available; wake a parked thread if one is waiting. Idempotent:
  // repeated unparks without an intervening park collapse to a single token.
  HYLE_PLATFORM_API void unpark() noexcept;

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
