// gather_bench -- the "measure, don't assume" bench for simd-09 #66: native gather/compress
// intrinsics vs the store+reload emulation, AT THE SAME ISA. The backend picks native for the
// common widths (avx2 vpgatherd for 32-bit, avx512 vpcompress/vpexpand for 32/64-bit); this bench
// races that native path against GatherIndexEmulated / CompressEmulated (the store+reload form
// kept accessible in the backend). Compiled per ISA variant (HYLE_SIMD_TARGET_*), so run the avx2
// exe on an AVX2 host and the avx512 exe under Intel SDE. Prints native-relative-to-emulated:
// >100% native faster, <100% native SLOWER (the AVX2-vpgather-is-slow case -- then the "native"
// pick is a loss and this bench is the evidence to reconsider it).
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/vec.h"

namespace hs = hyle::simd;
namespace nb = ankerl::nanobench;

namespace {
constexpr std::size_t kTable = 4096;  // gather source table (fits L1)
}  // namespace

int main() {
  const hs::ScalableTag<std::int32_t> d;
  const std::size_t n = hs::Lanes(d);

  // Gather source + random in-range indices.
  std::mt19937_64 rng(12345);
  std::vector<std::int32_t> table(kTable);
  for (std::size_t i = 0; i < kTable; ++i) table[i] = static_cast<std::int32_t>(i * 3 + 1);
  std::uniform_int_distribution<int> di(0, static_cast<int>(kTable) - 1);
  alignas(64) std::int32_t idxbuf[64];
  for (std::size_t i = 0; i < n; ++i) idxbuf[i] = di(rng);
  auto vidx = hs::Load(d, idxbuf);

  std::printf("gather/compress native-vs-emulated (i32, %zu lanes)\n", n);

  {
    nb::Bench b;
    b.title("GatherIndex i32").relative(true).batch(n).minEpochIterations(20000);
    b.run("emulated (store+reload)", [&] {
      auto r = hs::GatherIndexEmulated(d, table.data(), vidx);
      nb::doNotOptimizeAway(r);
    });
    b.run("native (vpgather/vpcompress per ISA)", [&] {
      auto r = hs::GatherIndex(d, table.data(), vidx);
      nb::doNotOptimizeAway(r);
    });
  }

  // Compress: half the lanes set (typical filter density).
  alignas(64) std::int32_t vbuf[64], mbuf[64];
  for (std::size_t i = 0; i < n; ++i) { vbuf[i] = static_cast<std::int32_t>(i + 1); mbuf[i] = (i % 2); }
  auto vv = hs::Load(d, vbuf);
  auto mask = hs::Eq(hs::Load(d, mbuf), hs::Set(d, 1));
  {
    nb::Bench b;
    b.title("Compress i32 (50% dense)").relative(true).batch(n).minEpochIterations(20000);
    b.run("emulated (store+reload)", [&] {
      auto r = hs::CompressEmulated(vv, mask);
      nb::doNotOptimizeAway(r);
    });
    b.run("native (vpcompress per ISA)", [&] {
      auto r = hs::Compress(vv, mask);
      nb::doNotOptimizeAway(r);
    });
  }
  return 0;
}
