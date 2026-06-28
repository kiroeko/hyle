#pragma once

#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_TIME_API
#elif defined(HYLE_TIME_BUILD)
#  define HYLE_TIME_API HYLE_EXPORT
#else
#  define HYLE_TIME_API HYLE_IMPORT
#endif

// Time primitives. Two distinct clocks (conflating them is a classic bug):
//   - monotonic: steady, never jumps/goes back, arbitrary zero. The ONLY clock
//     for deadlines/timeouts/elapsed (sched will key timers on it). clay lacked
//     this -- its time.h is wall-clock only, which silently breaks when the
//     admin/NTP steps the system clock.
//   - realtime: wall clock, ns since the Unix epoch. For timestamps/logging.
// Nanoseconds everywhere (clay used lossy float us/ms + ms ticks); ns is the
// modern unit (std::chrono / tokio Instant) and what the syscalls already give.

namespace hyle::time {

// Steady monotonic counter in ns from an unspecified zero. Non-decreasing across
// calls on the same machine. Linux/mac: clock_gettime(CLOCK_MONOTONIC); Windows:
// QueryPerformanceCounter scaled by the (fixed) frequency.
HYLE_TIME_API std::uint64_t monotonic_ns() noexcept;

// Wall-clock ns since 1970-01-01T00:00:00Z (UTC), not counting leap seconds.
// May jump if the system clock is adjusted -- never use for elapsed/deadlines.
HYLE_TIME_API std::uint64_t realtime_ns() noexcept;

// Sleep at least `ns` (the OS may round up / wake late). 0 yields.
HYLE_TIME_API void sleep_for_ns(std::uint64_t ns) noexcept;

inline void sleep_for_us(std::uint64_t us) noexcept {
  sleep_for_ns(us * 1'000ull);
}
inline void sleep_for_ms(std::uint64_t ms) noexcept {
  sleep_for_ns(ms * 1'000'000ull);
}

// A captured monotonic instant; the ergonomic way to measure elapsed time
// (tokio Instant). Header-inline -- no DLL hop for the arithmetic.
struct Instant {
  std::uint64_t ns = 0;

  static Instant now() noexcept { return Instant{monotonic_ns()}; }
  std::uint64_t elapsed_ns() const noexcept { return monotonic_ns() - ns; }
  // ns from this instant to `later` (0 if later is actually earlier).
  std::uint64_t until_ns(Instant later) const noexcept {
    return later.ns > ns ? later.ns - ns : 0;
  }
};

}  // namespace hyle::time
