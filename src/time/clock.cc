#include "hyle/time/clock.h"

#if defined(HYLE_TESTING)
#  include <cstdlib>
#endif

#include "hyle/macros.h"
#include "hyle/platform.h"  // HYLE_OS_* (project convention over raw _WIN32)

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <cerrno>
#  include <time.h>
#endif

// x86-64 profiling path reads the TSC directly. Pull in the
// CPUID + rdtsc intrinsics; on MSVC they live in <intrin.h>, on GCC/Clang in
// <x86intrin.h>/<cpuid.h>. Guarded so non-x86 builds don't reference them.
#if defined(__x86_64__) || defined(_M_X64)
#  define HYLE_TIME_HAS_TSC 1
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#    include <x86intrin.h>
#  endif
#else
#  define HYLE_TIME_HAS_TSC 0
#endif

namespace hyle::time {

namespace {

constexpr std::uint64_t scale_add_saturated(std::uint64_t whole,
                                             std::uint64_t scale,
                                             std::uint64_t frac) noexcept {
  if (whole > (UINT64_MAX - frac) / scale) return UINT64_MAX;
  return whole * scale + frac;
}

#if defined(HYLE_TESTING) && HYLE_TIME_HAS_TSC
bool force_tsc_timing_calibration() noexcept {
#  if defined(HYLE_OS_WINDOWS)
  char value[2] = {};
  const DWORD length = ::GetEnvironmentVariableA(
      "HYLE_TIME_FORCE_TSC_TIMING_CALIBRATION", value,
      static_cast<DWORD>(sizeof(value)));
  return length == 1 && value[0] == '1';
#  else
  const char* value =
      std::getenv("HYLE_TIME_FORCE_TSC_TIMING_CALIBRATION");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
#  endif
}
#endif

}  // namespace

#if defined(HYLE_OS_WINDOWS)

std::uint64_t coarse_realtime_ns() noexcept {
  // Coarse: reads KUSER_SHARED_DATA without the precise-clock path, at the cost
  // of OS-tick resolution. Platform/build point values live in the perf ledger.
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  std::uint64_t ticks =
      (static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
  constexpr std::uint64_t kEpochDiff = 116444736000000000ull;
  if (ticks <= kEpochDiff) return 0;
  const std::uint64_t unix_ticks = ticks - kEpochDiff;
  return scale_add_saturated(unix_ticks, 100ull, 0);
}

namespace {

void coarse_sleep_for_ns(std::uint64_t ns) noexcept {
  // Sleep takes ms (coarse, ~15ms granularity by default). Round up so we never
  // under-sleep. DWORD_MAX is INFINITE, so cap each finite chunk one below it.
  // Looping also avoids truncating durations longer than ~49.7 days.
  constexpr std::uint64_t kNsPerMs = 1'000'000ull;
  constexpr DWORD kMaxFiniteMs = INFINITE - 1;
  constexpr std::uint64_t kMaxChunkNs =
      static_cast<std::uint64_t>(kMaxFiniteMs) * kNsPerMs;
  while (ns > kMaxChunkNs) {
    Sleep(kMaxFiniteMs);
    ns -= kMaxChunkNs;
  }
  const DWORD ms = static_cast<DWORD>(
      ns / kNsPerMs + (ns % kNsPerMs != 0 ? 1ull : 0ull));
  Sleep(ms);
}

class ThreadWaitableTimer {
 public:
  ThreadWaitableTimer() noexcept {
    // Windows 10 1803+ accepts flag 0x2 for high-resolution waitable timers.
    // Keep the numeric value local so building against an older SDK does not
    // remove the runtime capability probe.
    constexpr DWORD kHighResolution = 0x00000002;
    handle_ = ::CreateWaitableTimerExW(nullptr, nullptr, kHighResolution,
                                       TIMER_ALL_ACCESS);
    if (handle_ == nullptr) {
      handle_ = ::CreateWaitableTimerExW(nullptr, nullptr, 0,
                                         TIMER_ALL_ACCESS);
    }
  }
  ~ThreadWaitableTimer() {
    if (handle_ != nullptr) ::CloseHandle(handle_);
  }
  ThreadWaitableTimer(const ThreadWaitableTimer&) = delete;
  ThreadWaitableTimer& operator=(const ThreadWaitableTimer&) = delete;

  [[nodiscard]] HANDLE get() const noexcept { return handle_; }

 private:
  HANDLE handle_ = nullptr;
};

HANDLE thread_waitable_timer() noexcept {
  thread_local ThreadWaitableTimer timer;
  return timer.get();
}

}  // namespace

void sleep_for_ns(std::uint64_t ns) noexcept {
  // Match std/Abseil duration semantics: a zero/non-positive duration is an
  // immediate no-op. Thread yielding is a scheduling primitive, not a sleep.
  if (ns == 0) return;

  HANDLE timer = thread_waitable_timer();
  if (timer == nullptr) {
    coarse_sleep_for_ns(ns);
    return;
  }

  // Relative waitable-timer due times are negative 100ns units. Round upward
  // to preserve the at-least contract. UINT64_MAX ns still fits in LONGLONG
  // after division by 100.
  const std::uint64_t ticks = ns / 100ull + (ns % 100ull != 0 ? 1ull : 0ull);
  LARGE_INTEGER due;
  due.QuadPart = -static_cast<LONGLONG>(ticks);
  if (!::SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0) ||
      ::WaitForSingleObject(timer, INFINITE) != WAIT_OBJECT_0) {
    // A restricted/old host may reject waitable timers. Sleeping coarsely is a
    // correctness-preserving fallback; never turn a recoverable OS limitation
    // into an early wake or process abort.
    coarse_sleep_for_ns(ns);
  }
}

#else  // POSIX (Linux / macOS)

std::uint64_t coarse_realtime_ns() noexcept {
  // Linux CLOCK_REALTIME_COARSE is a vDSO read with kernel-tick resolution.
  // macOS/other POSIX lacking it fall back to CLOCK_REALTIME (still correct,
  // just not faster there). Point values live in the perf ledger.
  struct timespec ts;
#if defined(CLOCK_REALTIME_COARSE)
  HYLE_FATAL_ASSERT(clock_gettime(CLOCK_REALTIME_COARSE, &ts) == 0);
#else
  HYLE_FATAL_ASSERT(clock_gettime(CLOCK_REALTIME, &ts) == 0);
#endif
  if (ts.tv_sec < 0) return 0;
  return scale_add_saturated(static_cast<std::uint64_t>(ts.tv_sec),
                             1'000'000'000ull,
                             static_cast<std::uint64_t>(ts.tv_nsec));
}

void sleep_for_ns(std::uint64_t ns) noexcept {
  if (ns == 0) return;
  struct timespec req;
  req.tv_sec = static_cast<std::time_t>(ns / 1'000'000'000ull);
  req.tv_nsec = static_cast<long>(ns % 1'000'000'000ull);
  // Resume after EINTR so a signal can't cut the sleep short.
  struct timespec rem;
  while (nanosleep(&req, &rem) != 0) {
    HYLE_FATAL_ASSERT(errno == EINTR);
    req = rem;
  }
}

#endif

bool coarse_realtime_available() noexcept {
#if defined(HYLE_OS_WINDOWS) || defined(CLOCK_REALTIME_COARSE)
  return true;
#else
  return false;
#endif
}

void sleep_until_ns(std::uint64_t deadline_ns) noexcept {
#if defined(HYLE_OS_LINUX)
  // Linux exposes the exact primitive this API means: an absolute
  // CLOCK_MONOTONIC sleep. It avoids a userspace now/read/subtract loop and is
  // immune to relative-wait drift across signals.
  struct timespec deadline;
  deadline.tv_sec = static_cast<std::time_t>(deadline_ns / 1'000'000'000ull);
  deadline.tv_nsec = static_cast<long>(deadline_ns % 1'000'000'000ull);
  int result;
  do {
    result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr);
  } while (result == EINTR);
  HYLE_FATAL_ASSERT(result == 0);
#else
  for (;;) {
    const std::uint64_t now = monotonic_ns();
    if (deadline_ns <= now) return;
    sleep_for_ns(deadline_ns - now);
  }
#endif
}

// ---------------------------------------------------------------------------
// profiling_clock_ns: TSC-backed profiling path (x86-64), Windows + POSIX.
// Reads rdtsc directly and scales cycles->ns with a precomputed fixed-point
// multiplier, skipping the OS clock call. Falls back to monotonic_ns() when the
// hardware lacks an invariant TSC or on non-x86 targets.
//
// Calibration prefers the EXACT CPUID leaf 0x15 (TSC/crystal ratio + crystal Hz)
// when the CPU enumerates it -- zero measurement, zero startup wait, exact
// (not 1%-approx) multiplier. When leaf 0x15 is masked (hypervisors: WSL/Hyper-V)
// or absent (older CPUs), it falls back to abseil's sysinfo.cc timing approach
// (read setup verified against the local abseil checkout): pair each TSC read with
// the OS clock and keep the lowest-latency pair (minimizes preemption skew), then
// grow the sample window until the measured frequency stabilizes within 1%.
// References: Intel SDM Vol.3 leaf 0x15/0x16; abseil GetTimeTscPair /
// MeasureTscFrequency; invariant-TSC probe per Tracy CheckHardwareSupportsInvariantTSC
// (CPUID leaf 1 EDX bit4 = rdtsc present, leaf 0x80000007 EDX bit8 = invariant TSC).
// ---------------------------------------------------------------------------

namespace {

#if HYLE_TIME_HAS_TSC

inline std::uint64_t read_tsc() noexcept {
#if defined(_MSC_VER)
  return __rdtsc();
#else
  return __builtin_ia32_rdtsc();
#endif
}

void cpuid(unsigned leaf, unsigned regs[4]) noexcept {
#if defined(_MSC_VER)
  int r[4];
  __cpuid(r, static_cast<int>(leaf));
  regs[0] = static_cast<unsigned>(r[0]);
  regs[1] = static_cast<unsigned>(r[1]);
  regs[2] = static_cast<unsigned>(r[2]);
  regs[3] = static_cast<unsigned>(r[3]);
#else
  __get_cpuid(leaf, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
}

bool hardware_has_invariant_tsc() noexcept {
  unsigned regs[4] = {0, 0, 0, 0};
  cpuid(1, regs);
  if (!(regs[3] & (1u << 4))) return false;  // EDX bit 4: rdtsc instruction
  // Need the extended leaf to even ask about invariant TSC.
  cpuid(0x80000000u, regs);
  if (regs[0] < 0x80000007u) return false;
  cpuid(0x80000007u, regs);
  return (regs[3] & (1u << 8)) != 0;  // EDX bit 8: invariant TSC
}

// One (OS-ns, tsc) sample taken with minimal gap between the two reads.
struct TimeTscPair {
  std::uint64_t ns;
  std::uint64_t tsc;
};

// Exact TSC frequency straight from CPUID leaf 0x15 (the "TSC/core crystal clock"
// enumeration), no measurement. TSC_Hz = crystal_Hz * numerator / denominator:
//   EAX = denominator of the ratio, EBX = numerator, ECX = core crystal Hz.
// Returns 0 (=> caller falls back to timing) whenever the leaf can't give a valid
// ratio. That covers THREE distinct cases, not just VMs -- the check is "did this
// CPU hand me a usable ratio right now", never "am I in a VM" (fragile):
//   1. Hypervisors (WSL2/Hyper-V/KVM) mask leaf 0x15 -> all zero. Bare-metal Linux
//      on the SAME i9 DOES enumerate it (this is not OS-dependent, it's whether a
//      hypervisor sits in between); bare-metal Windows here returns 2/156/38.4MHz
//      = 2.9952 GHz exactly.
//   2. Older Intel (pre-Skylake ~2015): invariant TSC present but leaf 0x15 not
//      enumerated -> timing path.
//   3. CPUs whose firmware/virtualization does not enumerate a usable ratio,
//      including many AMD generations, take the timing path. Capability probing
//      is authoritative; vendor-name guesses are not.
// When EBX/EAX are present but ECX (crystal) is 0, Intel's SDM says derive the
// crystal from leaf 0x16 (CPU base freq, MHz): a documented approximation for
// client parts that omit the crystal. Zero everywhere -> 0 (timing fallback).
double tsc_hz_from_cpuid() noexcept {
  unsigned regs[4] = {0, 0, 0, 0};
  cpuid(0, regs);
  if (regs[0] < 0x15u) return 0.0;  // leaf 0x15 not supported
  cpuid(0x15u, regs);
  const unsigned denom = regs[0], numer = regs[1];
  unsigned crystal = regs[2];
  if (denom == 0 || numer == 0) return 0.0;  // ratio not enumerated
  if (crystal == 0) {
    // Crystal not reported: derive from leaf 0x16 base MHz (SDM-sanctioned fallback).
    unsigned r2[4] = {0, 0, 0, 0};
    cpuid(0, r2);
    if (r2[0] < 0x16u) return 0.0;
    cpuid(0x16u, r2);
    const unsigned base_mhz = r2[0] & 0xffffu;
    if (base_mhz == 0) return 0.0;
    // base_MHz ~= crystal * numer/denom / 1e6  =>  crystal = base_MHz*1e6*denom/numer
    crystal = static_cast<unsigned>(
        static_cast<std::uint64_t>(base_mhz) * 1'000'000ull * denom / numer);
    if (crystal == 0) return 0.0;
  }
  return static_cast<double>(crystal) * static_cast<double>(numer) /
         static_cast<double>(denom);
}

// Read the pair several times, keep the one whose two clock reads were closest
// together -- that sample is least likely to have been preempted mid-read.
TimeTscPair best_time_tsc_pair() noexcept {
  TimeTscPair best{0, 0};
  std::uint64_t best_gap = ~std::uint64_t{0};
  for (int i = 0; i < 10; ++i) {
    std::uint64_t n0 = monotonic_ns();
    std::uint64_t tsc = read_tsc();
    std::uint64_t n1 = monotonic_ns();
    std::uint64_t gap = n1 - n0;
    if (gap < best_gap) {
      best_gap = gap;
      best.ns = n0;
      best.tsc = tsc;
    }
  }
  return best;
}

double measure_tsc_hz_once(std::uint64_t sleep_ns) noexcept {
  TimeTscPair a = best_time_tsc_pair();
  // Invariant TSC keeps counting while the thread sleeps. Measure the actual OS
  // elapsed interval after wake, so scheduler overshoot affects first-use wall
  // latency but not the frequency estimate. This follows Abseil's
  // MeasureTscFrequencyWithSleep shape and avoids burning an entire core during
  // the 1/2/4/...ms convergence windows.
  sleep_for_ns(sleep_ns);
  TimeTscPair b = best_time_tsc_pair();
  if (b.ns <= a.ns || b.tsc <= a.tsc) return 0.0;
  double elapsed_ticks = static_cast<double>(b.tsc - a.tsc);
  double elapsed_secs = static_cast<double>(b.ns - a.ns) * 1e-9;
  return elapsed_ticks / elapsed_secs;
}

// Grow the window (1ms, 2ms, ...) until two successive estimates agree to 1%.
double measure_tsc_hz() noexcept {
  double last = -1.0;
  std::uint64_t window_ns = 1'000'000;  // 1 ms
  for (int i = 0; i < 8; ++i) {
    double m = measure_tsc_hz_once(window_ns);
    if (m > 1.0) {
      if (last > 0.0 && m * 0.99 < last && last < m * 1.01) return m;
      last = m;
    }
    window_ns *= 2;
  }
  return last;
}

// Best available TSC frequency: prefer the EXACT CPUID leaf 0x15 value (no
// measurement, no startup wait, no 1% window tolerance), fall back to the
// timing method when the leaf is masked/absent (hypervisors, older CPUs).
// NOTE: this runs ONCE at first-call calibration (tsc_scale() Meyers singleton),
// NOT on the hot read path -- profiling_clock_ns() is rdtsc + one
// multiply-shift regardless of how mult was computed. leaf 0x15 improves STARTUP
// cost (skips the ms-scale calibration wait) and multiplier PRECISION (exact vs
// ~1%), it
// does NOT change per-call read latency.
double best_tsc_hz() noexcept {
#if defined(HYLE_TESTING)
  // Test-only process gate: exercise the timing fallback even on hosts whose
  // CPUID 0x15 normally makes it unreachable. This branch is compiled out of
  // shipped libraries and is used by a fresh-process cold-start CPU-time test.
  if (force_tsc_timing_calibration()) return measure_tsc_hz();
#endif
  double exact = tsc_hz_from_cpuid();
  if (exact > 1.0) return exact;
  return measure_tsc_hz();
}

// Fixed-point Q32 multiplier: ns = (tsc * kMult) >> 32. Computed once.
constexpr int kTscShift = 32;

struct TscScale {
  std::uint64_t mult;  // (1e9 / tsc_hz) * 2^32, 0 if TSC unusable
  bool available;
};

const TscScale& tsc_scale() noexcept {
  static const TscScale s = [] {
    if (!hardware_has_invariant_tsc()) return TscScale{0, false};
    double hz = best_tsc_hz();
    if (!(hz > 1.0)) return TscScale{0, false};  // calibration failed
    double m = (1e9 / hz) * static_cast<double>(std::uint64_t{1} << kTscShift);
    return TscScale{static_cast<std::uint64_t>(m), true};
  }();
  return s;
}

// (tsc * mult) >> 32 in full 128-bit precision, no overflow.
inline std::uint64_t scale_tsc_to_ns(std::uint64_t tsc, std::uint64_t mult) noexcept {
#if defined(_MSC_VER)
  std::uint64_t hi;
  std::uint64_t lo = _umul128(tsc, mult, &hi);
  return (hi << (64 - kTscShift)) | (lo >> kTscShift);
#else
  __extension__ using u128 = unsigned __int128;  // GCC/Clang ext; silence -Wpedantic
  u128 p = static_cast<u128>(tsc) * mult;
  return static_cast<std::uint64_t>(p >> kTscShift);
#endif
}

#endif  // HYLE_TIME_HAS_TSC

}  // namespace

bool profiling_clock_available() noexcept {
#if HYLE_TIME_HAS_TSC
  return tsc_scale().available;
#else
  return false;
#endif
}

std::uint64_t profiling_clock_ns() noexcept {
#if HYLE_TIME_HAS_TSC
  const TscScale& s = tsc_scale();
  if (s.available) return scale_tsc_to_ns(read_tsc(), s.mult);
#endif
  return monotonic_ns();  // no invariant TSC (or non-x86): safe fallback
}

}  // namespace hyle::time
