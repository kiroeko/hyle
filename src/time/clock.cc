#include "hyle/time/clock.h"

#include "clock_internal.h"

#if defined(HYLE_TESTING)
#  include <cstdlib>
#endif

#include "hyle/core/compiler.h"
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

// ARM64 exposes a system-wide architected virtual counter to EL0. Folly uses
// the same CNTVCT_EL0 path for its hardware timestamp; unlike a CPU cycle
// counter it runs at CNTFRQ_EL0 and is independent of core frequency. GCC and
// Clang support the required system-register asm on Linux/macOS. MSVC ARM64
// stays on the portable monotonic fallback until a reviewed intrinsic path is
// available instead of relying on unsupported inline asm.
#if defined(HYLE_ARCH_ARM64) && \
    (defined(HYLE_COMPILER_CLANG) || defined(HYLE_COMPILER_GCC))
#  define HYLE_TIME_HAS_ARM_COUNTER 1
#else
#  define HYLE_TIME_HAS_ARM_COUNTER 0
#endif

namespace hyle::time {

namespace {

constexpr std::uint64_t scale_add_saturated(std::uint64_t whole,
                                             std::uint64_t scale,
                                             std::uint64_t frac) noexcept {
  if (whole > (UINT64_MAX - frac) / scale) return UINT64_MAX;
  return whole * scale + frac;
}

#if HYLE_TIME_HAS_TSC || HYLE_TIME_HAS_ARM_COUNTER

constexpr int kHardwareCounterShift = 32;

// (counter * multiplier) >> 32 in full 128-bit precision. The common Apple
// ARM64 frequency is exactly 1 GHz and bypasses this helper entirely; the
// fixed-point path keeps other architected-counter frequencies division-free.
inline std::uint64_t scale_hardware_counter(std::uint64_t counter,
                                            std::uint64_t multiplier) noexcept {
#  if defined(_MSC_VER)
  std::uint64_t high;
  const std::uint64_t low = _umul128(counter, multiplier, &high);
  return (high << (64 - kHardwareCounterShift)) |
         (low >> kHardwareCounterShift);
#  else
  __extension__ using u128 = unsigned __int128;
  const u128 product = static_cast<u128>(counter) * multiplier;
  return static_cast<std::uint64_t>(product >> kHardwareCounterShift);
#  endif
}

#endif

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

namespace {

HYLE_FORCE_INLINE std::uint64_t coarse_realtime_ns_impl() noexcept {
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

HYLE_FORCE_INLINE std::uint64_t coarse_monotonic_ns_impl() noexcept {
  constexpr std::uint64_t kNsPerMs = 1'000'000ull;
  const std::uint64_t milliseconds = ::GetTickCount64();
  return milliseconds > UINT64_MAX / kNsPerMs
             ? UINT64_MAX
             : milliseconds * kNsPerMs;
}

std::uint64_t filetime_pair_ns(FILETIME kernel, FILETIME user) noexcept {
  const auto ticks = [](FILETIME value) noexcept {
    return (static_cast<std::uint64_t>(value.dwHighDateTime) << 32) |
           value.dwLowDateTime;
  };
  const std::uint64_t kernel_ticks = ticks(kernel);
  const std::uint64_t user_ticks = ticks(user);
  if (user_ticks > UINT64_MAX - kernel_ticks) return UINT64_MAX;
  const std::uint64_t total_ticks = kernel_ticks + user_ticks;
  return total_ticks > UINT64_MAX / 100u ? UINT64_MAX : total_ticks * 100u;
}

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

namespace {

HYLE_FORCE_INLINE std::uint64_t coarse_realtime_ns_impl() noexcept {
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

HYLE_FORCE_INLINE std::uint64_t coarse_monotonic_ns_impl() noexcept {
#if defined(CLOCK_MONOTONIC_COARSE)
  struct timespec ts;
  HYLE_FATAL_ASSERT(clock_gettime(CLOCK_MONOTONIC_COARSE, &ts) == 0);
  HYLE_FATAL_ASSERT(ts.tv_sec >= 0);
  return scale_add_saturated(static_cast<std::uint64_t>(ts.tv_sec),
                             1'000'000'000ull,
                             static_cast<std::uint64_t>(ts.tv_nsec));
#else
  return monotonic_ns();
#endif
}

}  // namespace

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

std::uint64_t coarse_realtime_ns() noexcept {
  return coarse_realtime_ns_impl();
}

bool coarse_realtime_available() noexcept {
#if defined(HYLE_OS_WINDOWS) || defined(CLOCK_REALTIME_COARSE)
  return true;
#else
  return false;
#endif
}

std::uint64_t coarse_monotonic_ns() noexcept {
  return coarse_monotonic_ns_impl();
}

bool coarse_monotonic_available() noexcept {
#if defined(HYLE_OS_WINDOWS) || defined(CLOCK_MONOTONIC_COARSE)
  return true;
#else
  return false;
#endif
}

std::uint64_t current_thread_cpu_ns() noexcept {
#if defined(HYLE_OS_WINDOWS)
  FILETIME created;
  FILETIME exited;
  FILETIME kernel;
  FILETIME user;
  HYLE_FATAL_ASSERT(::GetThreadTimes(::GetCurrentThread(), &created, &exited,
                                    &kernel, &user) != 0);
  return filetime_pair_ns(kernel, user);
#elif defined(CLOCK_THREAD_CPUTIME_ID)
  struct timespec value;
  HYLE_FATAL_ASSERT(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0);
  HYLE_FATAL_ASSERT(value.tv_sec >= 0);
  return scale_add_saturated(static_cast<std::uint64_t>(value.tv_sec),
                             1'000'000'000ull,
                             static_cast<std::uint64_t>(value.tv_nsec));
#else
#  error "HYLE current-thread CPU clock needs a supported native clock"
#endif
}

std::uint64_t process_cpu_ns() noexcept {
#if defined(HYLE_OS_WINDOWS)
  FILETIME created;
  FILETIME exited;
  FILETIME kernel;
  FILETIME user;
  HYLE_FATAL_ASSERT(::GetProcessTimes(::GetCurrentProcess(), &created, &exited,
                                     &kernel, &user) != 0);
  return filetime_pair_ns(kernel, user);
#elif defined(CLOCK_PROCESS_CPUTIME_ID)
  struct timespec value;
  HYLE_FATAL_ASSERT(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) == 0);
  HYLE_FATAL_ASSERT(value.tv_sec >= 0);
  return scale_add_saturated(static_cast<std::uint64_t>(value.tv_sec),
                             1'000'000'000ull,
                             static_cast<std::uint64_t>(value.tv_nsec));
#else
#  error "HYLE process CPU clock needs a supported native clock"
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
// profiling_clock_ns: direct hardware-counter profiling path. x86-64 uses an
// invariant TSC; GCC/Clang ARM64 uses the architected virtual counter. Both
// scale into ns without an OS clock call and fall back to monotonic_ns() when a
// reviewed hardware path is unavailable.
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

inline std::uint64_t read_tsc_measurement_start() noexcept {
#if defined(_MSC_VER)
  _ReadWriteBarrier();
  _mm_lfence();
  _ReadWriteBarrier();
  const std::uint64_t value = __rdtsc();
  _ReadWriteBarrier();
  _mm_lfence();
  _ReadWriteBarrier();
  return value;
#else
  std::uint32_t low;
  std::uint32_t high;
  __asm__ volatile("lfence\n\trdtsc\n\tlfence"
                   : "=a"(low), "=d"(high)
                   :
                   : "memory");
  return (static_cast<std::uint64_t>(high) << 32) | low;
#endif
}

inline std::uint64_t read_tsc_measurement_stop(bool has_rdtscp) noexcept {
  if (!has_rdtscp) return read_tsc_measurement_start();
#if defined(_MSC_VER)
  _ReadWriteBarrier();
  unsigned aux;
  const std::uint64_t value = __rdtscp(&aux);
  _ReadWriteBarrier();
  _mm_lfence();
  _ReadWriteBarrier();
  return value;
#else
  std::uint64_t value;
  __asm__ volatile("rdtscp\n\t"
                   "shl $32, %%rdx\n\t"
                   "or %%rdx, %%rax\n\t"
                   "lfence"
                   : "=a"(value)
                   :
                   : "rdx", "rcx", "cc", "memory");
  return value;
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

bool hardware_has_rdtscp() noexcept {
  unsigned regs[4] = {0, 0, 0, 0};
  cpuid(0x80000000u, regs);
  if (regs[0] < 0x80000001u) return false;
  cpuid(0x80000001u, regs);
  return (regs[3] & (1u << 27)) != 0;
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
  const unsigned crystal = regs[2];
  unsigned base_mhz = 0;
  if (denom != 0 && numer != 0 && crystal == 0) {
    // Crystal not reported: derive from leaf 0x16 base MHz (SDM-sanctioned fallback).
    unsigned r2[4] = {0, 0, 0, 0};
    cpuid(0, r2);
    if (r2[0] < 0x16u) return 0.0;
    cpuid(0x16u, r2);
    base_mhz = r2[0] & 0xffffu;
  }
  return detail::derive_tsc_hz({denom, numer, crystal, base_mhz});
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

// Grow the window until two successive estimates agree to 1%. Production
// starts at 1ms. The test-only forced path starts beyond one Windows scheduler
// accounting quantum so GetThreadTimes can distinguish sleeping from a busy
// calibration without turning a 15.625ms accounting tick into a false failure.
double measure_tsc_hz(std::uint64_t initial_window_ns = 1'000'000) noexcept {
  double last = -1.0;
  std::uint64_t window_ns = initial_window_ns;
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
  if (force_tsc_timing_calibration()) return measure_tsc_hz(32'000'000);
#endif
  double exact = tsc_hz_from_cpuid();
  if (exact > 1.0) return exact;
  return measure_tsc_hz();
}

struct TscScale {
  std::uint64_t mult;  // (1e9 / tsc_hz) * 2^32, 0 if TSC unusable
  bool available;
  bool rdtscp;
};

const TscScale& tsc_scale() noexcept {
  static const TscScale s = [] {
    if (!hardware_has_invariant_tsc()) return TscScale{0, false, false};
    double hz = best_tsc_hz();
    const std::uint64_t multiplier = detail::tsc_scale_multiplier(hz);
    if (multiplier == 0) {
      return TscScale{0, false, false};  // calibration failed
    }
    return TscScale{multiplier, true, hardware_has_rdtscp()};
  }();
  return s;
}

#endif  // HYLE_TIME_HAS_TSC

#if HYLE_TIME_HAS_ARM_COUNTER

inline std::uint64_t read_arm_virtual_counter() noexcept {
  std::uint64_t value;
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(value));
  return value;
}

inline std::uint64_t read_arm_virtual_counter_ordered() noexcept {
  std::uint64_t value;
  __asm__ volatile("isb\n\tmrs %0, cntvct_el0\n\tisb"
                   : "=r"(value)
                   :
                   : "memory");
  return value;
}

inline std::uint64_t read_arm_counter_frequency() noexcept {
  std::uint64_t value;
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value));
  return value;
}

struct ArmCounterScale {
  std::uint64_t multiplier;
  std::uint64_t frequency;
};

const ArmCounterScale& arm_counter_scale() noexcept {
  static const ArmCounterScale scale = [] {
    const std::uint64_t frequency = read_arm_counter_frequency();
    if (frequency == 0) return ArmCounterScale{0, 0};
    const long double multiplier =
        (1'000'000'000.0L / static_cast<long double>(frequency)) *
        static_cast<long double>(std::uint64_t{1}
                                 << kHardwareCounterShift);
    if (!(multiplier > 0.0L) ||
        multiplier >= static_cast<long double>(UINT64_MAX)) {
      return ArmCounterScale{0, 0};
    }
    return ArmCounterScale{static_cast<std::uint64_t>(multiplier),
                           frequency};
  }();
  return scale;
}

#endif  // HYLE_TIME_HAS_ARM_COUNTER

}  // namespace

bool profiling_clock_available() noexcept {
#if HYLE_TIME_HAS_TSC
  return tsc_scale().available;
#elif HYLE_TIME_HAS_ARM_COUNTER
  return arm_counter_scale().frequency != 0;
#else
  return false;
#endif
}

std::uint64_t profiling_clock_ns() noexcept {
#if HYLE_TIME_HAS_TSC
  const TscScale& s = tsc_scale();
  if (s.available) return scale_hardware_counter(read_tsc(), s.mult);
#elif HYLE_TIME_HAS_ARM_COUNTER
  const ArmCounterScale& scale = arm_counter_scale();
  if (scale.frequency == 1'000'000'000u) {
    // Apple Silicon exposes a 1 GHz virtual counter: its tick is already ns.
    return read_arm_virtual_counter();
  }
  if (scale.frequency != 0) {
    return scale_hardware_counter(read_arm_virtual_counter(),
                                  scale.multiplier);
  }
#endif
  return monotonic_ns();
}

namespace {

std::uint64_t profiling_measurement_ns(bool stop) noexcept {
#if HYLE_TIME_HAS_TSC
  const TscScale& scale = tsc_scale();
  if (scale.available) {
    const std::uint64_t counter =
        stop ? read_tsc_measurement_stop(scale.rdtscp)
             : read_tsc_measurement_start();
    return scale_hardware_counter(counter, scale.mult);
  }
#elif HYLE_TIME_HAS_ARM_COUNTER
  (void)stop;
  const ArmCounterScale& scale = arm_counter_scale();
  if (scale.frequency != 0) {
    const std::uint64_t counter = read_arm_virtual_counter_ordered();
    if (scale.frequency == 1'000'000'000u) return counter;
    return scale_hardware_counter(counter, scale.multiplier);
  }
#else
  (void)stop;
#endif
  return monotonic_ns();
}

}  // namespace

std::uint64_t profiling_measurement_start_ns() noexcept {
  return profiling_measurement_ns(false);
}

std::uint64_t profiling_measurement_stop_ns() noexcept {
  return profiling_measurement_ns(true);
}

}  // namespace hyle::time
