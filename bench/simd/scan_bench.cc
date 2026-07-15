// scan_bench -- scalar O(n) chain vs HYLE inclusive_scan (P1-1).
// On Linux x64 HYLE scan is pure scalar by policy (see scan_dispatch.cc); expect ~TIE with
// the local scalar chain. On Windows HYLE is SIMD and should beat MSVC scalar substantially.
// Avoids per-iter full vector copy; re-seed in place with a cheap fill.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/simd/scan.h"

namespace {
// 1M u32: serial chain wall is clear; working set ~4MB (outside L2 on typical cores).
constexpr std::size_t kN = 1 << 20;

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void scalar_inclusive_u32(std::uint32_t* data, std::size_t n) noexcept {
  if (n == 0) return;
  // True loop-carried dependence — cannot legally vectorize across iterations.
  for (std::size_t i = 1; i < n; ++i) data[i] += data[i - 1];
}

void fill_ones(std::uint32_t* data, std::size_t n) noexcept {
  // Not the measured path; just re-seed. memset is fine for u32 all-0x01 pattern? No —
  // need value 1 per dword. Use a tight fill (autovec OK here).
  for (std::size_t i = 0; i < n; ++i) data[i] = 1u;
}
}  // namespace

int main() {
  std::vector<std::uint32_t> buf(kN);

  {
    ankerl::nanobench::Bench b;
    b.title("inclusive_scan u32 (1M ones)").relative(true).batch(kN).minEpochIterations(50);
    b.run("scalar chain", [&] {
      fill_ones(buf.data(), kN);
      scalar_inclusive_u32(buf.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(buf.back());
    });
    b.run("HYLE inclusive_scan", [&] {
      fill_ones(buf.data(), kN);
      hyle::simd::inclusive_scan(buf.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(buf.back());
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("exclusive_scan u32 (1M ones)").relative(true).batch(kN).minEpochIterations(50);
    b.run("HYLE exclusive_scan", [&] {
      fill_ones(buf.data(), kN);
      hyle::simd::exclusive_scan(buf.data(), kN);
      ankerl::nanobench::doNotOptimizeAway(buf.back());
    });
  }
  return 0;
}
