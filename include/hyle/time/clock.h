#pragma once

#include <chrono>
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

using Duration = std::chrono::nanoseconds;
using MonotonicClock = std::chrono::steady_clock;
using MonotonicTimePoint =
    std::chrono::time_point<MonotonicClock, Duration>;
using RealtimeClock = std::chrono::system_clock;
using RealtimeTimePoint = std::chrono::time_point<RealtimeClock, Duration>;

// Typed C++ clock views. Raw *_ns functions remain the compact hot-path/C ABI
// representation; normal C++ callers can use standard chrono types and avoid
// mixing a wall timestamp, duration and monotonic deadline by accident.
HYLE_FORCE_INLINE MonotonicTimePoint monotonic_now() noexcept {
  return std::chrono::time_point_cast<Duration>(MonotonicClock::now());
}

HYLE_FORCE_INLINE RealtimeTimePoint realtime_now() noexcept {
  return std::chrono::time_point_cast<Duration>(RealtimeClock::now());
}

// Steady monotonic counter in ns from an unspecified zero. The native C++ path
// is header-inline over std::chrono, avoiding an extra hyle_time shared-library
// hop before the standard library reaches CLOCK_MONOTONIC/QPC. The stable C ABI
// remains an exported wrapper. This is the only clock for deadlines/timeouts.
HYLE_FORCE_INLINE std::uint64_t monotonic_ns() noexcept {
  const auto value = monotonic_now().time_since_epoch().count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

// Fast monotonic counter in ns, read straight from the CPU timestamp counter
// (rdtsc) instead of going through the OS primitive. It can be materially
// cheaper because it skips the QPC/clock_gettime call shell; platform/build
// point values live in the performance ledger. Opt-in: you pick it at the call
// site, and the default monotonic_ns() pays nothing for its existence.
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
// Header-inline for the same one-hop reason as monotonic_ns(). May jump if the
// system clock is adjusted -- never use for elapsed/deadlines.
HYLE_FORCE_INLINE std::uint64_t realtime_ns() noexcept {
  const auto value = realtime_now().time_since_epoch().count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

// Fast wall-clock ns -- the realtime counterpart of fast_monotonic_ns(). Reads the
// OS's COARSE time source (Windows GetSystemTimeAsFileTime = KUSER_SHARED_DATA
// read; Linux CLOCK_REALTIME_COARSE = vDSO), skipping the high-precision path
// realtime_ns() uses. It is materially cheaper on systems where that coarse
// source exists; exact ratios are platform/build dependent and live in the
// benchmark ledger rather than the API contract.
//
// TRADEOFF (why it is NOT the default): when a native coarse clock is available,
// its RESOLUTION is an OS tick rather than one nanosecond. The value is still a
// wall-clock ns count since the epoch, but it may update only once per tick, so
// nearby calls can return the same value. On systems without a coarse source this
// entry falls back to realtime_ns(); query fast_realtime_available() when the
// distinction affects policy. The same jump-on-adjust caveat as realtime_ns()
// applies (never use it for elapsed/deadlines -- use monotonic_ns()).
//
// USE IT FOR: high-frequency wall-clock TIMESTAMPS where ms resolution is fine and
// read cost matters -- log record stamps, trace events, sampling. When you need
// true sub-ms wall-clock precision (rare), use realtime_ns().
HYLE_TIME_API std::uint64_t fast_realtime_ns() noexcept;

// True only when fast_realtime_ns() uses a genuinely coarser/faster native
// source. It is false on platforms where that entry transparently falls back
// to realtime_ns(), allowing a logging pipeline to make an informed choice.
HYLE_TIME_API bool fast_realtime_available() noexcept;

// Sleep at least `ns` (the OS may round up / wake late). 0 is an immediate
// no-op; yielding is deliberately left to the scheduling API.
HYLE_TIME_API void sleep_for_ns(std::uint64_t ns) noexcept;

// Sleep until an absolute monotonic deadline. A deadline at/in the past returns
// immediately. This is the overflow-safe primitive for callers that already
// carry an absolute deadline.
HYLE_TIME_API void sleep_until_ns(std::uint64_t deadline_ns) noexcept;

// Saturating deadline arithmetic shared by sched/io and FFI consumers. Keeping
// this here avoids every timeout caller open-coding `monotonic_ns() + timeout`
// and accidentally wrapping a very large duration into the past.
constexpr std::uint64_t saturating_add_ns(std::uint64_t base,
                                          std::uint64_t delta) noexcept {
  return delta > UINT64_MAX - base ? UINT64_MAX : base + delta;
}

// Convert any standard duration to positive nanoseconds, rounding upward so a
// requested wait/deadline never becomes shorter during unit conversion.
// Non-positive/NaN values become zero; huge/infinite values saturate.
template <class Rep, class Period>
std::uint64_t duration_ns_saturated(
    std::chrono::duration<Rep, Period> duration) noexcept {
  using FloatNs = std::chrono::duration<long double, std::nano>;
  const long double ns = FloatNs(duration).count();
  if (!(ns > 0.0L)) return 0;
  const long double max = static_cast<long double>(UINT64_MAX);
  if (ns >= max) return UINT64_MAX;
  const auto whole = static_cast<std::uint64_t>(ns);
  return static_cast<long double>(whole) < ns ? whole + 1 : whole;
}

inline std::uint64_t to_monotonic_ns(MonotonicTimePoint point) noexcept {
  const auto value = point.time_since_epoch().count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

inline std::uint64_t to_realtime_ns(RealtimeTimePoint point) noexcept {
  const auto value = point.time_since_epoch().count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

inline MonotonicTimePoint monotonic_time_point(
    std::uint64_t value_ns) noexcept {
  constexpr auto max = static_cast<std::uint64_t>(Duration::max().count());
  const auto clamped = value_ns > max ? Duration::max().count()
                                      : static_cast<Duration::rep>(value_ns);
  return MonotonicTimePoint(Duration(clamped));
}

inline RealtimeTimePoint realtime_time_point(std::uint64_t value_ns) noexcept {
  constexpr auto max = static_cast<std::uint64_t>(Duration::max().count());
  const auto clamped = value_ns > max ? Duration::max().count()
                                      : static_cast<Duration::rep>(value_ns);
  return RealtimeTimePoint(Duration(clamped));
}

inline std::uint64_t deadline_after_ns(std::uint64_t timeout_ns) noexcept {
  return saturating_add_ns(monotonic_ns(), timeout_ns);
}

inline std::uint64_t remaining_ns(std::uint64_t deadline_ns) noexcept {
  const std::uint64_t now = monotonic_ns();
  return deadline_ns > now ? deadline_ns - now : 0;
}

template <class Rep, class Period>
inline MonotonicTimePoint deadline_after(
    std::chrono::duration<Rep, Period> timeout) noexcept {
  return monotonic_time_point(
      deadline_after_ns(duration_ns_saturated(timeout)));
}

inline Duration remaining(MonotonicTimePoint deadline) noexcept {
  const MonotonicTimePoint now = monotonic_now();
  return deadline > now ? deadline - now : Duration::zero();
}

template <class Rep, class Period>
inline void sleep_for(std::chrono::duration<Rep, Period> duration) noexcept {
  sleep_for_ns(duration_ns_saturated(duration));
}

inline void sleep_until(MonotonicTimePoint deadline) noexcept {
  sleep_until_ns(to_monotonic_ns(deadline));
}

inline void sleep_for_us(std::uint64_t us) noexcept {
  sleep_for_ns(
      us > UINT64_MAX / 1'000ull ? UINT64_MAX : us * 1'000ull);
}
inline void sleep_for_ms(std::uint64_t ms) noexcept {
  sleep_for_ns(ms > UINT64_MAX / 1'000'000ull
                   ? UINT64_MAX
                   : ms * 1'000'000ull);
}

// A type-safe monotonic snapshot (oneTBB tick_count / Tokio Instant shape).
// The representation is hidden so it cannot be confused with realtime ns.
class Instant {
 public:
  Instant() noexcept = default;
  explicit Instant(MonotonicTimePoint point) noexcept : point_(point) {}

  static Instant now() noexcept { return Instant(monotonic_now()); }
  [[nodiscard]] MonotonicTimePoint time_point() const noexcept {
    return point_;
  }
  [[nodiscard]] Duration elapsed() const noexcept {
    const MonotonicTimePoint now = monotonic_now();
    return now > point_ ? now - point_ : Duration::zero();
  }
  [[nodiscard]] std::uint64_t elapsed_ns() const noexcept {
    return static_cast<std::uint64_t>(elapsed().count());
  }
  [[nodiscard]] Duration until(Instant later) const noexcept {
    return later.point_ > point_ ? later.point_ - point_ : Duration::zero();
  }
  [[nodiscard]] std::uint64_t until_ns(Instant later) const noexcept {
    return static_cast<std::uint64_t>(until(later).count());
  }

  friend Duration operator-(Instant later, Instant earlier) noexcept {
    // Match std::chrono time-point subtraction: direction is meaningful, so
    // reverse subtraction is negative rather than silently clamped to zero.
    // elapsed()/until() retain their explicitly non-negative convenience shape.
    return later.point_ - earlier.point_;
  }
  friend bool operator==(Instant, Instant) noexcept = default;
  friend auto operator<=>(Instant, Instant) noexcept = default;

 private:
  MonotonicTimePoint point_{};
};

}  // namespace hyle::time
