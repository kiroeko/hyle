#include "hyle/time/clock.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>

#  include <thread>
#endif

namespace hyle::time {

#if defined(_WIN32)

namespace {
// QPC frequency is fixed after boot; query once. Counts/sec -> ns scaling.
std::int64_t qpc_freq() noexcept {
  static const std::int64_t f = [] {
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);  // never fails on Win XP+
    return li.QuadPart;
  }();
  return f;
}
}  // namespace

std::uint64_t monotonic_ns() noexcept {
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  std::int64_t f = qpc_freq();
  // Split to avoid overflow: (ticks/f)*1e9 + (ticks%f)*1e9/f. ticks*1e9 alone
  // overflows 64-bit after ~9 seconds at a 1 GHz freq.
  std::int64_t t = c.QuadPart;
  std::uint64_t whole = static_cast<std::uint64_t>(t / f) * 1'000'000'000ull;
  std::uint64_t frac =
      (static_cast<std::uint64_t>(t % f) * 1'000'000'000ull) /
      static_cast<std::uint64_t>(f);
  return whole + frac;
}

std::uint64_t realtime_ns() noexcept {
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);  // 100-ns ticks since 1601-01-01 UTC
  std::uint64_t ticks =
      (static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
  // 1601->1970 offset in 100-ns ticks.
  constexpr std::uint64_t kEpochDiff = 116444736000000000ull;
  return (ticks - kEpochDiff) * 100ull;
}

void sleep_for_ns(std::uint64_t ns) noexcept {
  if (ns == 0) {
    SwitchToThread();
    return;
  }
  // Sleep takes ms (coarse, ~15ms granularity by default). Round up so we never
  // under-sleep. Sub-ms precision waiting is a sched concern, not this floor.
  DWORD ms = static_cast<DWORD>((ns + 999'999ull) / 1'000'000ull);
  Sleep(ms);
}

#else  // POSIX (Linux / macOS)

std::uint64_t monotonic_ns() noexcept {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

std::uint64_t realtime_ns() noexcept {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

void sleep_for_ns(std::uint64_t ns) noexcept {
  if (ns == 0) {
    std::this_thread::yield();
    return;
  }
  struct timespec req;
  req.tv_sec = static_cast<std::time_t>(ns / 1'000'000'000ull);
  req.tv_nsec = static_cast<long>(ns % 1'000'000'000ull);
  // Resume after EINTR so a signal can't cut the sleep short.
  struct timespec rem;
  while (nanosleep(&req, &rem) != 0) req = rem;
}

#endif

}  // namespace hyle::time
