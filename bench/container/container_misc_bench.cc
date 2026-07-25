// container misc bench (task #19 breadth + gap-fill #34/#35): the claim-bearing containers.
// performance.md 桶A: SmallVector "小 N 省 malloc（大赢）";
// CArray = the dynamic-array (std::vector equivalent, C-ABI header-before-data).
//
// SOTA rivals (not just std -- "只赢 std 不算价值"):
//   * SmallVector  -> absl::InlinedVector (THE small-vector SOTA), gated -DHYLE_BENCH_ABSL
//   * CArray       -> std::vector (the drop-in equivalent, always on)
// std equivalents always run as the floor. Win or lose, printed.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/container/carray.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/allocator.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/container/inlined_vector.h"
#endif

using hyle::container::CArray;
using hyle::container::SmallVector;
using hyle::mem::system_allocator;

namespace {
namespace nb = ankerl::nanobench;
constexpr int kSmall = 8;      // fits inline (no spill) -- the SmallVector design point
constexpr int kRounds = 1000;  // construct+fill+destroy churn per epoch
constexpr int kBig = 4096;     // CArray/vector growth element count
}  // namespace

int main() {
  // ---- SmallVector vs absl::InlinedVector vs std::vector: small-N churn -------
  // Each round: build fresh, push kSmall ints, destroy. SmallVector<int,16> and
  // absl::InlinedVector<int,16> both stay inline (zero heap); std::vector mallocs.
  // absl::InlinedVector is the SOTA small-vector (same inline-storage design) ->
  // the apples-to-apples rival; std::vector is the floor.
  {
    nb::Bench b;
    b.title("small-N churn: build+fill 8 ints+destroy")
        .relative(true)
        .batch(kRounds * kSmall)
        .minEpochIterations(200);
    b.run("HYLE SmallVector<int,16> (inline)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        SmallVector<int, 16> v;
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::InlinedVector<int,16> (SOTA)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        absl::InlinedVector<int, 16> v;
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#endif
    b.run("std::vector<int> (heap)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        std::vector<int> v;
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
    b.run("std::vector<int> reserve(8) (heap, 1 malloc)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        std::vector<int> v;
        v.reserve(kSmall);
        for (int i = 0; i < kSmall; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
  }

  // ---- SmallVector SPILL path: push kSpill > inline cap (heap grow) ----------
  // The small-N block above stays inline (the no-malloc design point). This block
  // exercises the OTHER half: push 64 ints into SmallVector<int,16> so it spills
  // inline->heap and then grows. absl::InlinedVector<int,16> spills the same way
  // (the SOTA rival); std::vector mallocs from the first push (the floor). Tests
  // whether HYLE's spill/grow path is competitive once inline storage is exceeded.
  {
    constexpr int kSpill = 64;  // > 16 inline cap -> forces spill + ~2 grows
    nb::Bench b;
    b.title("spill churn: build+fill 64 ints+destroy (>inline cap)")
        .relative(true)
        .batch(kRounds * kSpill)
        .minEpochIterations(100);
    b.run("HYLE SmallVector<int,16> (spill->heap)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        SmallVector<int, 16> v;
        for (int i = 0; i < kSpill; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#if defined(HYLE_BENCH_ABSL)
    b.run("absl::InlinedVector<int,16> (SOTA, spill->heap)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        absl::InlinedVector<int, 16> v;
        for (int i = 0; i < kSpill; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
#endif
    b.run("std::vector<int> (heap from push 1)", [&] {
      for (int r = 0; r < kRounds; ++r) {
        std::vector<int> v;
        for (int i = 0; i < kSpill; ++i) v.push_back(i);
        nb::doNotOptimizeAway(v.data());
      }
    });
  }

  // ---- CArray vs std::vector: large-N push_back growth + indexed read --------
  // CArray is the dynamic-array (std::vector equivalent) with a C-ABI header-
  // before-data layout: arr[i] is one load, the element ptr hands straight to a C
  // API. For trivially-copyable T both memcpy-grow, so this measures whether the
  // type-erased core + header layout costs anything vs std::vector's inline growth.
  // std::vector IS the drop-in SOTA here (an excellent, heavily-tuned dynamic array).
  {
    nb::Bench b;
    b.title("dynamic-array growth: push_back 4096 ints (no reserve)")
        .relative(true)
        .batch(kBig)
        .minEpochIterations(5000);
    b.run("HYLE CArray<int> (header-before-data)", [&] {
      CArray<int> a(system_allocator());
      for (int i = 0; i < kBig; ++i) a.push_back(i);
      nb::doNotOptimizeAway(a.data());
    });
    b.run("std::vector<int> (SOTA drop-in)", [&] {
      std::vector<int> v;
      for (int i = 0; i < kBig; ++i) v.push_back(i);
      nb::doNotOptimizeAway(v.data());
    });
  }
  {
    // Indexed read hot path: sequential sum over a prebuilt array (the arr[i]
    // single-load claim). Build once, sum many times.
    CArray<int> ha(system_allocator());
    for (int i = 0; i < kBig; ++i) ha.push_back(i);
    std::vector<int> hv;
    for (int i = 0; i < kBig; ++i) hv.push_back(i);
    nb::Bench b;
    b.title("dynamic-array sequential read sum (4096 ints)")
        .relative(true)
        .batch(kBig)
        .minEpochIterations(2000);
    b.run("HYLE CArray<int>", [&] {
      std::int64_t sum = 0;
      for (int i = 0; i < kBig; ++i) sum += ha[static_cast<std::uint64_t>(i)];
      nb::doNotOptimizeAway(sum);
    });
    b.run("std::vector<int>", [&] {
      std::int64_t sum = 0;
      for (int i = 0; i < kBig; ++i) sum += hv[static_cast<std::size_t>(i)];
      nb::doNotOptimizeAway(sum);
    });
  }

  return 0;
}
