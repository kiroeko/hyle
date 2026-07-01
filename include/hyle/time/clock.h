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
//
// This is the DEFAULT clock: correct everywhere, safe across sleep/suspend, and
// the only one valid for deadlines/timeouts (sched keys its timers on it). ~12ns.
HYLE_TIME_API std::uint64_t monotonic_ns() noexcept;

// Fast monotonic counter in ns, read straight from the CPU timestamp counter
// (rdtsc) instead of going through the OS primitive -- about 2x faster (~6ns vs
// ~12ns) because it skips the QPC/clock_gettime call shell. Opt-in: you pick it
// at the call site, the default monotonic_ns() pays nothing for its existence.
//
// TRADEOFFS (read before using -- this is why it is NOT the default):
//   - Requires an invariant TSC (constant rate, synced across cores). Probed via
//     CPUID; call fast_monotonic_available() to check. On hardware without it
//     (or non-x86), this transparently FALLS BACK to monotonic_ns() -- always
//     safe, just not faster there.
//   - MAY JUMP after the machine wakes from sleep/suspend or on VM migration:
//     the TSC can briefly read wrong right after resume (this is exactly why
//     abseil disables its cycle-counter clock by default). Do NOT use it for
//     deadlines, timeouts, or anything sched schedules on -- use monotonic_ns().
//   - NOT STRICTLY MONOTONIC across cores: even with an invariant TSC, a thread
//     migrated between cores may observe a slightly lower value than its prior
//     read (per-core TSC sync is approximate). There is no max() clamp (that
//     would cost a global atomic). So computing elapsed as (b - a) can underflow
//     to a huge value if the reads straddle a migration. Pin the thread, or only
//     diff reads taken close together on one core, or just use monotonic_ns().
//   - First call pays a one-time TSC calibration: an exact read from CPUID leaf
//     0x15 when the CPU enumerates it (no measurement), else a ~few-ms timing
//     calibration against the OS clock. Subsequent calls are a bare rdtsc + a
//     multiply/shift.
//
// USE IT FOR: high-frequency elapsed-time sampling / profiling hot-path spans
// where you want the lowest possible read cost and a rare post-resume glitch is
// tolerable. When in doubt, use monotonic_ns().
HYLE_TIME_API std::uint64_t fast_monotonic_ns() noexcept;

// True iff fast_monotonic_ns() uses the hardware TSC fast path on this machine
// (invariant TSC present and calibrated). When false, fast_monotonic_ns() is
// just a forwarder to monotonic_ns() and offers no speedup. Result is cached;
// the first call triggers CPUID probing + calibration.
HYLE_TIME_API bool fast_monotonic_available() noexcept;

// Wall-clock ns since 1970-01-01T00:00:00Z (UTC), not counting leap seconds.
// May jump if the system clock is adjusted -- never use for elapsed/deadlines.
HYLE_TIME_API std::uint64_t realtime_ns() noexcept;

// Fast wall-clock ns -- the realtime counterpart of fast_monotonic_ns(). Reads the
// OS's COARSE time source (Windows GetSystemTimeAsFileTime = KUSER_SHARED_DATA
// read; Linux CLOCK_REALTIME_COARSE = vDSO), skipping the high-precision path
// realtime_ns() uses. Measured ~7.6x faster on Windows (1.8 vs 13.5 ns) and ~4.2x
// on Linux (3.1 vs 13.0 ns).
//
// TRADEOFF (why it is NOT the default, and unconditionally "available" unlike the
// TSC path): the coarse clock's RESOLUTION is the OS timer tick, ~1-15 ms (measured
// ~4 ms here), NOT nanoseconds. The value is still real wall-clock ns since the
// epoch and never needs calibration -- it just updates only once per tick, so two
// calls within the same tick return the SAME value. Same jump-on-adjust caveat as
// realtime_ns() (never use for elapsed/deadlines -- use monotonic_ns()).
//
// USE IT FOR: high-frequency wall-clock TIMESTAMPS where ms resolution is fine and
// read cost matters -- log record stamps, trace events, sampling. When you need
// true sub-ms wall-clock precision (rare), use realtime_ns().
HYLE_TIME_API std::uint64_t fast_realtime_ns() noexcept;

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
