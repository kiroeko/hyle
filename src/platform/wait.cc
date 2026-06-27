#include "hyle/platform/wait.h"

#include <atomic>
#include <cstdint>

#include "hyle/platform.h"

// ---- low-level wait_on_address: one backend per OS --------------------------
// Each maps the (addr, expected) compare-and-block 1:1 onto the platform futex.
// Only the Windows path is built and tested on this machine; the Linux/macOS
// paths are written to spec but NOT yet verified on real hardware/CI.

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
void wake_one(const std::atomic<std::uint32_t>* addr) noexcept {
  ::WakeByAddressSingle(const_cast<PVOID>(reinterpret_cast<const void*>(addr)));
}
void wake_all(const std::atomic<std::uint32_t>* addr) noexcept {
  ::WakeByAddressAll(const_cast<PVOID>(reinterpret_cast<const void*>(addr)));
}

}  // namespace hyle::platform::detail

#elif defined(HYLE_OS_LINUX)
#  include <climits>
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
void wake_one(const std::atomic<std::uint32_t>* addr) noexcept {
  futex(addr, FUTEX_WAKE_PRIVATE, 1);
}
void wake_all(const std::atomic<std::uint32_t>* addr) noexcept {
  futex(addr, FUTEX_WAKE_PRIVATE, INT_MAX);
}

}  // namespace hyle::platform::detail

#elif defined(HYLE_OS_MACOS)
// os_sync_wait_on_address is public API on macOS 14.4+. Older macOS needs a
// fallback (待调研: __ulock is private / dispatch_semaphore changes the model).
#  include <os/os_sync_wait_on_address.h>

namespace hyle::platform::detail {

void wait_on_address(const std::atomic<std::uint32_t>* addr,
                     std::uint32_t expected) noexcept {
  ::os_sync_wait_on_address(const_cast<void*>(reinterpret_cast<const void*>(addr)),
                            expected, sizeof(std::uint32_t),
                            OS_SYNC_WAIT_ON_ADDRESS_NONE);
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

}  // namespace hyle::platform
