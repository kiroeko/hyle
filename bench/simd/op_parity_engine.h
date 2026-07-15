// op_parity_engine -- shared measurement engine for the op_parity_* benches. Exists because the
// old per-op single-instance measurement was shown to carry two systematic errors on the reference
// machine (i9-13900K, MSVC; full evidence chain in docs/debug/op-parity-harness-artifacts-hunt.md):
//   1. scheduler noise: unpinned runs migrate between P/E cores -> per-run ratio swings of 10-30%
//      (an instruction-identical pair read 0.91 one build, 1.27..1.49 the next).
//   2. code-placement bias: two loops with IDENTICAL instructions at different code addresses in
//      the same binary differ stably by +-10..15% (tail ~30%) from front-end placement effects
//      alone (uop-cache/fetch-window geometry). Deterministic per binary, re-rolled per compile;
//      order-swap does NOT cancel it, small TUs do NOT reliably cancel it.
//   3. closure stack/heap 4K aliasing: kernels that re-load the lambda capture chain every
//      iteration stall when the closure's stack address aliases the store stream mod 4096 --
//      seeded by the LAUNCHER's environment-block size (same exe, bash vs powershell, 25-40%
//      one-sided). Countered in the bench TUs, not here: every kernel hoists `X.data()` into
//      non-escaping pointer locals so the hot loop never touches the closure.
// Countermeasures implemented here:
//   * pin_and_boost(): one P-core + high priority (Windows/Linux affinity; macOS QoS hint).
//   * replication: each kernel is a GENERIC lambda taking (auto rep); the engine instantiates it
//     R=4 times per side. Each instantiation is a distinct function at a distinct address, so the
//     placement factor is SAMPLED instead of drawn once. A globally-unique immediate is baked into
//     each instance (doNotOptimizeAway(kUniq)) so linker ICF cannot fold replicas back to one
//     address. Sides run interleaved (H0 G0 H1 G1 ...) to symmetrize any residual time drift.
//   * calibration anchors: ops whose HYLE and Highway backends are the same single instruction
//     (Neg/Sub/Add/Min/...) must aggregate to ~1.00; scripts/op-parity-report.py enforces this as
//     a gate before any verdicts are read.
// Aggregation contract (scripts/op-parity-report.py): per side take best-of-replicas (placement-
// ideal) and median-of-replicas; a LOSE verdict requires both estimators to agree, in every input
// run. Row labels are "HYLE#<r>" / "HWY#<r>".
#pragma once

#include <nanobench.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#elif defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace hyle_bench {

inline constexpr int kReplicas = 4;  // per side; report aggregates best+median across these

// Pin to one performance core and raise priority. Env HYLE_BENCH_CPU picks the logical CPU
// (default 2: a P-core on the 13900K reference box, away from core 0 interrupt traffic). macOS has
// no affinity API; USER_INTERACTIVE QoS keeps a busy thread on performance cores.
inline void pin_and_boost() {
  const char* env = std::getenv("HYLE_BENCH_CPU");
#if defined(_WIN32)
  const int cpu = env ? std::atoi(env) : 2;
  ::SetPriorityClass(::GetCurrentProcess(), HIGH_PRIORITY_CLASS);
  ::SetThreadAffinityMask(::GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
#elif defined(__linux__)
  const int cpu = env ? std::atoi(env) : 2;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  (void)sched_setaffinity(0, sizeof(set), &set);
#elif defined(__APPLE__)
  (void)env;
  (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
  (void)env;
#endif
}

// Order-swap control kept from the old harness: env HYLE_BENCH_HWYFIRST runs Highway before HYLE
// within every replica round. With interleaving + replication this is a redundancy check, not a
// correction; the report may be fed one run per order and requires verdict agreement.
inline bool hwy_first() {
  static const bool v = std::getenv("HYLE_BENCH_HWYFIRST") != nullptr;
  return v;
}

// Subset filter: env HYLE_BENCH_ONLY="Foo,Bar" runs only pairs whose title contains one of the
// comma-separated substrings, so a single-op fix reruns in seconds instead of the whole 200-op TU
// (the full sweep is long enough that macOS jetsam SIGKILLs it mid-run, discarding entire passes).
// Calibration anchors always run regardless -- the report's gate needs them present. Substring
// match on the title (e.g. "AESKeyGen" matches "AESKeyGenAssist u8"); empty/unset = run everything.
inline bool title_selected(const std::string& title) {
  static const char* filt = std::getenv("HYLE_BENCH_ONLY");
  if (!filt || !*filt) return true;
  // anchors: same names the report's ANCHORS set keys on -- keep them so CALIBRATION can pass.
  static const char* kAnchors[] = {"Neg i32", "Sub i32", "Mul i32", "Max i32", "ShiftLeft<3> i32",
                                    "Add f32", "MulAdd f32", "Min i32", "SaturatedAdd i16"};
  for (const char* a : kAnchors)
    if (title.find(a) != std::string::npos) return true;
  const std::string f(filt);
  std::size_t start = 0;
  while (start <= f.size()) {
    std::size_t comma = f.find(',', start);
    const std::size_t end = (comma == std::string::npos) ? f.size() : comma;
    if (end > start && title.find(f.substr(start, end - start)) != std::string::npos) return true;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return false;
}

// Epoch-length scale. Default epochs (~1-2ms at these ops' speeds) are fine on the pinned Windows
// and WSL legs (within-measurement err% 0.3-1.6%) but on macOS ambient app timers (ms-scale
// bursts, no affinity API to escape them) inflated err% to 4-28%; 8x longer epochs average the
// bursts out. Env HYLE_BENCH_EPOCH_SCALE overrides.
inline std::uint64_t epoch_scale() {
  if (const char* e = std::getenv("HYLE_BENCH_EPOCH_SCALE")) {
    const auto v = std::strtoull(e, nullptr, 10);
    return v ? v : 1;
  }
#if defined(__APPLE__)
  return 8;
#else
  return 1;
#endif
}

// rep tag passed to every kernel lambda. ::value is globally unique per instantiation (SITE is
// __COUNTER__ at the call site) -- kernels MUST end with `hyle_bench::mark(rep);` which bakes that
// value into the instance's code so no two measured functions are byte-identical (ICF defeat).
template <int SITE, int R>
using RepTag = std::integral_constant<int, SITE * 16 + R>;

template <class Tag>
inline void mark(Tag) {
  ankerl::nanobench::doNotOptimizeAway(static_cast<int>(Tag::value));
}

// Phase stagger -- kernels call this FIRST: `hyle_bench::stagger(rep);`. Replica k pads ~16*k
// bytes of straight-line code before the measured loop so the four replicas sample four distinct
// 64B code phases. Without it, same-sized replica instantiations get laid out at a constant
// stride; when that stride is a multiple of 64 the four copies share ONE phase (observed: all
// HYLE vpermd replicas at phase 32 while Highway's stride of 720 swept 48/0/16/32), and
// best-of-replicas then favors the sweeping side by up to ~10% even for byte-identical kernels.
// The volatile stores run once per op() call, outside the timed loop's steady state -- their cost
// is amortized over minEpochIterations and identical across sides.
// Must force-inline: an outlined stagger would be ONE shared function and shift nothing.
#if defined(_MSC_VER)
#define HYLE_BENCH_FORCEINLINE __forceinline
#else
#define HYLE_BENCH_FORCEINLINE inline __attribute__((always_inline))
#endif
template <class Tag>
HYLE_BENCH_FORCEINLINE void stagger(Tag) {
  constexpr unsigned k = static_cast<unsigned>(Tag::value) & 3u;
  volatile int s = 100;  // baseline store for every replica; k adds 2 more each (~16B/step)
  if constexpr (k >= 1) { s = 101; s = 102; }
  if constexpr (k >= 2) { s = 103; s = 104; }
  if constexpr (k >= 3) { s = 105; s = 106; }
  (void)s;
}

namespace detail {
template <int SITE, int R, class FH, class FG>
inline void run_round(ankerl::nanobench::Bench& b, FH& h, FG& g) {
  static constexpr char kH[][8] = {"HYLE#0", "HYLE#1", "HYLE#2", "HYLE#3"};
  static constexpr char kG[][8] = {"HWY#0", "HWY#1", "HWY#2", "HWY#3"};
  static_assert(R < 4, "label tables sized for kReplicas==4");
  if (hwy_first()) {
    b.run(kG[R], [&] { g(RepTag<SITE, R>{}); });
    b.run(kH[R], [&] { h(RepTag<SITE, R + 8>{}); });
  } else {
    b.run(kH[R], [&] { h(RepTag<SITE, R>{}); });
    b.run(kG[R], [&] { g(RepTag<SITE, R + 8>{}); });
  }
}
}  // namespace detail

// h/g: generic lambdas `[&](auto rep){ <loop>; hyle_bench::mark(rep); }`. Instantiated kReplicas
// times each, interleaved.
template <int SITE, class FH, class FG>
inline void run_pair_replicated(ankerl::nanobench::Bench& b, FH&& h, FG&& g) {
  if (!title_selected(b.title())) return;  // HYLE_BENCH_ONLY subset filter
  b.minEpochIterations(b.minEpochIterations() * epoch_scale());
  detail::run_round<SITE, 0>(b, h, g);
  detail::run_round<SITE, 1>(b, h, g);
  detail::run_round<SITE, 2>(b, h, g);
  detail::run_round<SITE, 3>(b, h, g);
}

// Informational third row (xsimd / store+reload baseline): single instance, label passthrough.
// No verdict is derived from these rows.
template <class F>
inline void run_third(ankerl::nanobench::Bench& b, const char* label, F&& f) {
  b.run(label, std::forward<F>(f));
}

}  // namespace hyle_bench

// Call-site macro: gives each pair a compile-time-unique SITE id for the ICF-defeating marks.
#define HYLE_RUN_PAIR(bench_, ...) ::hyle_bench::run_pair_replicated<__COUNTER__>((bench_), __VA_ARGS__)
