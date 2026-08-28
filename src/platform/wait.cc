#include "hyle/platform/wait.h"

#include <atomic>
#include <cstdint>

#include "hyle/platform.h"

// ---- low-level wait_on_address: one backend per OS --------------------------
// Each maps the (addr, expected) compare-and-block 1:1 onto the platform futex.
// All three backends are exercised on their native Windows/Linux/macOS hosts.

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
// WaitOnAddress / WakeByAddress* live in API set, import lib Synchronization.lib.

namespace hyle::platform::detail {

void wait_on_address(const std::atomic<std::uint32_t>* addr,
                     std::uint32_t expected) noexcept {
  // volatile cast: the OS reads the word directly; the atomic guarantees the
  // store side publishes before the wake.
  ::WaitOnAddress(
      const_cast<volatile VOID*>(reinterpret_cast<const volatile VOID*>(addr)),
      &expected, sizeof(expected), INFINITE);
}
bool wait_on_address_for(const std::atomic<std::uint32_t>* addr,
                         std::uint32_t expected,
                         std::uint64_t timeout_ns) noexcept {
  // A zero timeout is a poll, not a request to enter the kernel with a zero
  // deadline. WaitOnAddress(0) is microseconds slower than this atomic check.
  if (timeout_ns == 0) {
    return addr->load(std::memory_order_relaxed) != expected;
  }
  // WaitOnAddress takes a millisecond timeout. Round UP so a sub-ms deadline
  // never truncates to 0. Clamp below INFINITE (0xFFFFFFFF) so a huge ns never
  // accidentally means "forever".
  const std::uint64_t rounded =
      timeout_ns / 1'000'000ull +
      (timeout_ns % 1'000'000ull != 0 ? 1ull : 0ull);
  const DWORD ms =
      rounded >= INFINITE ? (INFINITE - 1) : static_cast<DWORD>(rounded);
  BOOL ok = ::WaitOnAddress(
      const_cast<volatile VOID*>(reinterpret_cast<const volatile VOID*>(addr)),
      &expected, sizeof(expected), ms);
  // FALSE + ERROR_TIMEOUT is the only "timed out" outcome; a value mismatch or
  // real wake returns TRUE (caller re-checks its condition).
  if (!ok && ::GetLastError() == ERROR_TIMEOUT) return false;
  return true;
}
void wake_one(const std::atomic<std::uint32_t>* addr) noexcept {
  ::WakeByAddressSingle(const_cast<PVOID>(reinterpret_cast<const void*>(addr)));
}
void wake_all(const std::atomic<std::uint32_t>* addr) noexcept {
  ::WakeByAddressAll(const_cast<PVOID>(reinterpret_cast<const void*>(addr)));
}

}  // namespace hyle::platform::detail

#elif defined(HYLE_OS_LINUX)
#  include <climits>
#  include <ctime>
#  include <cerrno>
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>

namespace hyle::platform::detail {

static long futex(const std::atomic<std::uint32_t>* addr, int op,
                  std::uint32_t val) noexcept {
  return ::syscall(SYS_futex, reinterpret_cast<const std::uint32_t*>(addr), op,
                   val, nullptr, nullptr, 0);
}
void wait_on_address(const std::atomic<std::uint32_t>* addr,
                     std::uint32_t expected) noexcept {
  futex(addr, FUTEX_WAIT_PRIVATE, expected);  // blocks while *addr == expected
}
bool wait_on_address_for(const std::atomic<std::uint32_t>* addr,
                         std::uint32_t expected,
                         std::uint64_t timeout_ns) noexcept {
  // FUTEX_WAIT with {0,0} is still a syscall (especially expensive under WSL).
  if (timeout_ns == 0) {
    return addr->load(std::memory_order_relaxed) != expected;
  }
  // FUTEX_WAIT interprets its timespec as a RELATIVE duration (unlike
  // FUTEX_WAIT_BITSET, which is absolute). A relative timeout is exactly what
  // the wheel wants: "sleep at most this long from now".
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(timeout_ns / 1'000'000'000ull);
  ts.tv_nsec = static_cast<long>(timeout_ns % 1'000'000'000ull);
  long r = ::syscall(SYS_futex, reinterpret_cast<const std::uint32_t*>(addr),
                     FUTEX_WAIT_PRIVATE, expected, &ts, nullptr, 0);
  // Only ETIMEDOUT means the deadline fired. -EAGAIN (value already changed),
  // -EINTR (signal), or success all mean "re-check your condition" -> true.
  if (r == -1 && errno == ETIMEDOUT) return false;
  return true;
}
void wake_one(const std::atomic<std::uint32_t>* addr) noexcept {
  futex(addr, FUTEX_WAKE_PRIVATE, 1);
}
void wake_all(const std::atomic<std::uint32_t>* addr) noexcept {
  futex(addr, FUTEX_WAKE_PRIVATE, INT_MAX);
}

}  // namespace hyle::platform::detail

#elif defined(HYLE_OS_MACOS)
// os_sync_wait_on_address is public API on macOS 14.4+. The project/package
// contract sets and validates that deployment floor; keep a source-level guard
// too so a non-CMake compilation cannot emit a binary with unresolved imports
// for an older declared target.
#  include <Availability.h>
#  if __MAC_OS_X_VERSION_MIN_REQUIRED < 140400
#    error "HYLE platform wait requires macOS deployment target 14.4 or newer"
#  endif
#  include <cerrno>
#  include <os/os_sync_wait_on_address.h>

namespace hyle::platform::detail {

void wait_on_address(const std::atomic<std::uint32_t>* addr,
                     std::uint32_t expected) noexcept {
  ::os_sync_wait_on_address(const_cast<void*>(reinterpret_cast<const void*>(addr)),
                            expected, sizeof(std::uint32_t),
                            OS_SYNC_WAIT_ON_ADDRESS_NONE);
}
bool wait_on_address_for(const std::atomic<std::uint32_t>* addr,
                         std::uint32_t expected,
                         std::uint64_t timeout_ns) noexcept {
  if (timeout_ns == 0) {
    return addr->load(std::memory_order_relaxed) != expected;
  }
  // os_sync_wait_on_address_with_timeout: OS_CLOCK_MACH_ABSOLUTE_TIME + a
  // RELATIVE deadline expressed in ns (macOS 14.4+). Matches the futex relative
  // model. Returns -1/errno==ETIMEDOUT on deadline.
  int r = ::os_sync_wait_on_address_with_timeout(
      const_cast<void*>(reinterpret_cast<const void*>(addr)), expected,
      sizeof(std::uint32_t), OS_SYNC_WAIT_ON_ADDRESS_NONE,
      OS_CLOCK_MACH_ABSOLUTE_TIME, timeout_ns);
  if (r == -1 && errno == ETIMEDOUT) return false;
  return true;
}
void wake_one(const std::atomic<std::uint32_t>* addr) noexcept {
  ::os_sync_wake_by_address_any(
      const_cast<void*>(reinterpret_cast<const void*>(addr)),
      sizeof(std::uint32_t), OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
void wake_all(const std::atomic<std::uint32_t>* addr) noexcept {
  ::os_sync_wake_by_address_all(
      const_cast<void*>(reinterpret_cast<const void*>(addr)),
      sizeof(std::uint32_t), OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

}  // namespace hyle::platform::detail

#else
#  error "HYLE platform/wait: unsupported OS (no futex-equivalent backend)"
#endif

// ---- portable Parker (one copy, over wait_on_address) -----------------------
// Ported from Rust std's futex thread parker (proven correct). The fetch_sub(1)
// trick encodes the EMPTY<->PARKED<->NOTIFIED transitions in one RMW.

namespace hyle::platform {

void Parker::park() noexcept {
  // Consume an already-delivered token: NOTIFIED(1) - 1 == EMPTY(0) -> return.
  // Otherwise EMPTY(0) - 1 wraps to PARKED(0xFFFFFFFF) and we go to sleep.
  if (state_.fetch_sub(1, std::memory_order_acquire) == kNotified) {
    return;
  }
  for (;;) {
    detail::wait_on_address(&state_, kParked);  // sleeps while state_ == PARKED
    // Woken: only a real unpark (which set NOTIFIED) lets us leave; anything
    // else is spurious, so re-sleep.
    std::uint32_t expected = kNotified;
    if (state_.compare_exchange_strong(expected, kEmpty,
                                       std::memory_order_acquire,
                                       std::memory_order_acquire)) {
      return;
    }
  }
}

void Parker::unpark() noexcept {
  // Publish the token. If a thread was PARKED, wake it. EMPTY/NOTIFIED prior
  // states just leave the token for a (future or already-returned) parker.
  if (state_.exchange(kNotified, std::memory_order_release) == kParked) {
    detail::wake_one(&state_);
  }
}

bool Parker::park_for(time::Duration timeout) noexcept {
  const std::uint64_t timeout_ns = time::duration_ns_saturated(timeout);
  // Polling must stay entirely in user space. Parker has one consumer, so the
  // no-token path needs only a load; publishing PARKED or issuing a CAS here is
  // needless work (and an uncontended CAS is notably expensive on Apple ARM).
  // A racing unpark after an empty load simply leaves its token for next time.
  if (timeout_ns == 0) {
    if (state_.load(std::memory_order_relaxed) != kNotified) return false;
    return state_.fetch_sub(1, std::memory_order_acquire) == kNotified;
  }
  // Same fetch_sub(1) token trick as park(), but SINGLE-SHOT: after one timed
  // wait we return regardless, because a spurious wake is indistinguishable from
  // a partial-timeout here and the caller re-checks its condition + re-parks
  // with a fresh deadline (this mirrors Rust std's park_timeout, which likewise
  // does not internally loop on the remaining time). Returns true iff a token
  // was actually consumed.
  if (state_.fetch_sub(1, std::memory_order_acquire) == kNotified) {
    return true;  // token was already waiting
  }
  // EMPTY(0)-1 wrapped to PARKED; wait at most timeout_ns for an unpark.
  detail::wait_on_address_for(&state_, kParked, timeout_ns);
  // Reset PARKED->EMPTY and report whether the token arrived. swap gives us the
  // prior value atomically: NOTIFIED means unpark won the race (consume it),
  // anything else means we timed out / spuriously woke with no token.
  return state_.exchange(kEmpty, std::memory_order_acquire) == kNotified;
}

}  // namespace hyle::platform
