#include "hyle/time/clock.h"

#include "hyle/platform.h"  // HYLE_OS_* (project convention over raw _WIN32)

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>

#  include <thread>
#endif

// x86-64 fast path (fast_monotonic_ns) reads the TSC directly. Pull in the
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

#if defined(HYLE_OS_WINDOWS)

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
  std::int64_t t = c.QuadPart;
  std::int64_t f = qpc_freq();
  // The frequency is a runtime value, so a plain (t/f) is a 64-bit idiv (~20-40
  // cyc) on the hot read path -- and the split below pays it twice. Special-case
  // the frequencies hardware actually reports so the divisor is a compile-time
  // constant the compiler folds into a magic-number multiply (no idiv). This is
  // exactly what MSVC's steady_clock::now() does; matching it closes the Win gap.
  constexpr std::int64_t kTenMHz = 10'000'000;          // ubiquitous on x86/x64
  constexpr std::int64_t kTwentyFourMHz = 24'000'000;   // common on ARM64 / emulated x86
  if (f == kTenMHz) {
    // 1e9 / 1e7 == 100 exactly; t*100 can't overflow u64 until ~585 yr of uptime.
    return static_cast<std::uint64_t>(t) * 100ull;
  }
  // Split to avoid overflow: (t/f)*1e9 + (t%f)*1e9/f. t*1e9 alone overflows
  // 64-bit after ~9 seconds at a 1 GHz freq. With a constant divisor the
  // compiler emits the magic-number multiply instead of idiv.
  if (f == kTwentyFourMHz) {
    std::uint64_t whole =
        static_cast<std::uint64_t>(t / kTwentyFourMHz) * 1'000'000'000ull;
    std::uint64_t frac =
        (static_cast<std::uint64_t>(t % kTwentyFourMHz) * 1'000'000'000ull) /
        static_cast<std::uint64_t>(kTwentyFourMHz);
    return whole + frac;
  }
  // Uncommon frequency: runtime-divisor split (the two idivs we can't fold away).
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

std::uint64_t fast_realtime_ns() noexcept {
  // Coarse: reads KUSER_SHARED_DATA (no syscall), ~1.8 ns vs ~13.5 for the Precise
  // variant, at the cost of ~1-15 ms resolution (updates on the timer tick).
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  std::uint64_t ticks =
      (static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
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

std::uint64_t fast_realtime_ns() noexcept {
  // Linux CLOCK_REALTIME_COARSE is a vDSO read (~3 ns vs ~13 for CLOCK_REALTIME),
  // resolution = the kernel tick (~1-4 ms). macOS/other POSIX lacking it fall back
  // to CLOCK_REALTIME (still correct, just not faster there).
  struct timespec ts;
#if defined(CLOCK_REALTIME_COARSE)
  clock_gettime(CLOCK_REALTIME_COARSE, &ts);
#else
  clock_gettime(CLOCK_REALTIME, &ts);
#endif
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

// ---------------------------------------------------------------------------
// fast_monotonic_ns: TSC-backed fast path (x86-64), shared by Windows + POSIX.
// Reads rdtsc directly and scales cycles->ns with a precomputed fixed-point
// multiplier, skipping the OS clock call. Falls back to monotonic_ns() when the
// hardware lacks an invariant TSC or on non-x86 targets.
//
// Calibration prefers the EXACT CPUID leaf 0x15 (TSC/crystal ratio + crystal Hz)
// when the CPU enumerates it -- zero measurement, zero startup busy-spin, exact
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
//   3. AMD: has NO leaf 0x15 at all (it's an Intel-specific leaf) -> ALWAYS the
//      timing path. Not a missed optimization -- the leaf simply doesn't exist there.
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
  // Busy-spin rather than sleep: keeps the core (and its TSC) live, and we don't
  // depend on sleep granularity. The window is short (ms range).
  std::uint64_t target = a.ns + sleep_ns;
  while (monotonic_ns() < target) { /* spin */ }
  TimeTscPair b = best_time_tsc_pair();
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
    if (last > 0.0 && m * 0.99 < last && last < m * 1.01) return m;
    last = m;
    window_ns *= 2;
  }
  return last;
}

// Best available TSC frequency: prefer the EXACT CPUID leaf 0x15 value (no
// measurement, no startup busy-spin, no 1% window tolerance), fall back to the
// timing method when the leaf is masked/absent (hypervisors, older CPUs).
// NOTE: this runs ONCE at first-call calibration (tsc_scale() Meyers singleton),
// NOT on the hot read path -- fast_monotonic_ns() is always rdtsc + one
// multiply-shift regardless of how mult was computed. leaf 0x15 improves STARTUP
// cost (skips the ms-scale busy-spin) and multiplier PRECISION (exact vs ~1%), it
// does NOT change per-call read latency.
double best_tsc_hz() noexcept {
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

bool fast_monotonic_available() noexcept {
#if HYLE_TIME_HAS_TSC
  return tsc_scale().available;
#else
  return false;
#endif
}

std::uint64_t fast_monotonic_ns() noexcept {
#if HYLE_TIME_HAS_TSC
  const TscScale& s = tsc_scale();
  if (s.available) return scale_tsc_to_ns(read_tsc(), s.mult);
#endif
  return monotonic_ns();  // no invariant TSC (or non-x86): safe fallback
}

}  // namespace hyle::time
