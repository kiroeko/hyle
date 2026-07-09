// op_parity_bench -- the "is our self-made op face as fast as Highway's?" bench. The worry it
// answers: HYLE hand-wrote ~217 ops per backend (copying Highway's per-op intrinsic sequences, not
// vendoring Highway's ops/). If any of those is SLOWER than Highway's same op, self-making it was
// a loss and we should just call Highway (it's vendored in third_party, always available).
//
// APPLES-TO-APPLES (the inlining-asymmetry lesson, feedback_benchmark_inlining_asymmetry): BOTH
// HYLE (vec.h) and Highway (hn::) are header-only and included HERE, so both inline into this TU
// at the SAME ISA -- no .so boundary, no export macro on either side. A gap is then a real codegen
// difference, not a call/inline artifact. Compiled per-ISA (HYLE_SIMD_TARGET_* + matching -march),
// like gather_bench: run the avx2 exe on an AVX2 host, the avx512 exe under Intel SDE.
//
// Reads: >100% => HYLE faster than Highway; <100% => HYLE SLOWER (the case that would justify
// switching that op to a Highway call). ~100% => tie (self-made costs nothing, keep it for the
// dtype/scalable control it buys).
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "hyle/simd/vec.h"

// Highway at the SAME ISA. HWY_COMPILE_ONLY_* not needed: we include the static header and pin the
// target via the compile flags (same -march this TU gets), so hn:: == HWY_NAMESPACE resolves to
// the matching target and inlines.
#include "hwy/highway.h"
namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;
namespace nb = ankerl::nanobench;

namespace {
constexpr std::size_t kN = 4096;  // L1-resident; isolates compute throughput

std::vector<float> genf(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}
std::vector<std::int32_t> geni(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> d(-1000, 1000);
  std::vector<std::int32_t> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}
}  // namespace

int main() {
  const auto a = genf(kN, 1), b = genf(kN, 2), c = genf(kN, 3);
  std::vector<float> out(kN);
  const auto ia = geni(kN, 4);
  std::vector<std::int32_t> iout(kN);

  const hs::ScalableTag<float> hd;
  const hn::ScalableTag<float> gd;
  const hs::ScalableTag<std::int32_t> hdi;
  const hn::ScalableTag<std::int32_t> gdi;
  const std::size_t hw = hs::Lanes(hd);
  const std::size_t gw = hn::Lanes(gd);

  std::printf("op parity: HYLE vs Highway, same ISA (HYLE lanes=%zu, hwy lanes=%zu)\n", hw, gw);

  // ---- 1. Add (pure passthrough: MUST tie -- same vaddps) ----
  {
    nb::Bench bench; bench.title("Add f32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    bench.run("HYLE Add", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::Add(hs::LoadU(hd, a.data() + i), hs::LoadU(hd, b.data() + i)), hd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
    bench.run("Highway Add", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::Add(hn::LoadU(gd, a.data() + i), hn::LoadU(gd, b.data() + i)), gd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
  }

  // ---- 2. MulAdd / FMA (passthrough: MUST tie -- same vfmadd) ----
  {
    nb::Bench bench; bench.title("MulAdd f32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    bench.run("HYLE MulAdd", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::MulAdd(hs::LoadU(hd, a.data() + i), hs::LoadU(hd, b.data() + i), hs::LoadU(hd, c.data() + i)), hd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
    bench.run("Highway MulAdd", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::MulAdd(hn::LoadU(gd, a.data() + i), hn::LoadU(gd, b.data() + i), hn::LoadU(gd, c.data() + i)), gd, out.data() + i);
      nb::doNotOptimizeAway(out.data());
    });
  }

  // ---- 3. ReduceSum (HYLE self-made horizontal reduce vs Highway ReduceSum) ----
  {
    nb::Bench bench; bench.title("ReduceSum f32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    bench.run("HYLE ReduceSum", [&] {
      float acc = 0;
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        acc += hs::ReduceSum(hd, hs::LoadU(hd, a.data() + i));
      nb::doNotOptimizeAway(acc);
    });
    bench.run("Highway ReduceSum", [&] {
      float acc = 0;
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        acc += hn::ReduceSum(gd, hn::LoadU(gd, a.data() + i));
      nb::doNotOptimizeAway(acc);
    });
  }

  // ---- 4. Reverse (HYLE swizzle vs Highway) ----
  {
    nb::Bench bench; bench.title("Reverse i32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    bench.run("HYLE Reverse", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::Reverse(hdi, hs::LoadU(hdi, ia.data() + i)), hdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway Reverse", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::Reverse(gdi, hn::LoadU(gdi, ia.data() + i)), gdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ---- 5. Min (int -- passthrough vpminsd, should tie) ----
  {
    const auto ib = geni(kN, 5);
    nb::Bench bench; bench.title("Min i32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    bench.run("HYLE Min", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::Min(hs::LoadU(hdi, ia.data() + i), hs::LoadU(hdi, ib.data() + i)), hdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway Min", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::Min(hn::LoadU(gdi, ia.data() + i), hn::LoadU(gdi, ib.data() + i)), gdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ================= EMULATION-CLASS ops (the ones that CAN differ) =================
  // These are where HYLE self-implemented a store+reload / scalar-fallback / composed form and
  // Highway may have a native instruction or a smarter sequence. THIS is the real test of "did
  // self-making cost us". Use bench-stat.py -n 15 on the output; a gap only counts if > combined
  // spread (single-run % is noise).

  // ---- 6. PopulationCount u32 (HYLE store+reload+std::popcount vs Highway shuffle-LUT) ----
  // Highway PopulationCount is unsigned-only; use u32 tags both sides for apples-to-apples.
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const std::size_t hwu = hs::Lanes(hdu), gwu = hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN), uo(kN);
    { std::mt19937_64 rng(6); std::uniform_int_distribution<std::uint32_t> d(0, 0xFFFFFFFFu); for (auto& e : ua) e = d(rng); }
    nb::Bench bench; bench.title("PopulationCount u32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    bench.run("HYLE PopulationCount", [&] {
      for (std::size_t i = 0; i + hwu <= kN; i += hwu)
        hs::StoreU(hs::PopulationCount(hs::LoadU(hdu, ua.data() + i)), hdu, uo.data() + i);
      nb::doNotOptimizeAway(uo.data());
    });
    bench.run("Highway PopulationCount", [&] {
      for (std::size_t i = 0; i + gwu <= kN; i += gwu)
        hn::StoreU(hn::PopulationCount(hn::LoadU(gdu, ua.data() + i)), gdu, uo.data() + i);
      nb::doNotOptimizeAway(uo.data());
    });
  }

  // ---- 7. Compress i32 (HYLE store+reload vs Highway; AVX2 has no VBMI2 so both emulate) ----
  {
    const auto imask = geni(kN, 7);
    nb::Bench bench; bench.title("Compress i32 (4096, 50% dense)").relative(true).batch(kN).minEpochIterations(5000);
    bench.run("HYLE Compress", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw) {
        auto v = hs::LoadU(hdi, ia.data() + i);
        auto m = hs::Gt(hs::LoadU(hdi, imask.data() + i), hs::Set(hdi, 0));
        hs::StoreU(hs::Compress(v, m), hdi, iout.data() + i);
      }
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway Compress", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw) {
        auto v = hn::LoadU(gdi, ia.data() + i);
        auto m = hn::Gt(hn::LoadU(gdi, imask.data() + i), hn::Set(gdi, 0));
        hn::StoreU(hn::Compress(v, m), gdi, iout.data() + i);
      }
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ---- 8. GatherIndex i32 (HYLE native _mm256_i32gather vs Highway GatherIndex) ----
  {
    std::vector<std::int32_t> table(kN);
    for (std::size_t i = 0; i < kN; ++i) table[i] = static_cast<std::int32_t>(i * 3 + 1);
    std::vector<std::int32_t> idxbuf(kN);
    { std::mt19937_64 rng(9); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : idxbuf) e = d(rng); }
    nb::Bench bench; bench.title("GatherIndex i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    bench.run("HYLE GatherIndex", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::GatherIndex(hdi, table.data(), hs::LoadU(hdi, idxbuf.data() + i)), hdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway GatherIndex", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::GatherIndex(gdi, table.data(), hn::LoadU(gdi, idxbuf.data() + i)), gdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ---- 9. SaturatedAdd i16 (HYLE vs Highway -- both should be native vpaddsw) ----
  {
    const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
    const std::size_t hw16 = hs::Lanes(hd16), gw16 = hn::Lanes(gd16);
    std::vector<std::int16_t> sa(kN), sb(kN), so(kN);
    { std::mt19937_64 rng(11); std::uniform_int_distribution<int> d(-30000, 30000); for (std::size_t i=0;i<kN;++i){sa[i]=(std::int16_t)d(rng); sb[i]=(std::int16_t)d(rng);} }
    nb::Bench bench; bench.title("SaturatedAdd i16 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE SaturatedAdd", [&] {
      for (std::size_t i = 0; i + hw16 <= kN; i += hw16)
        hs::StoreU(hs::SaturatedAdd(hs::LoadU(hd16, sa.data() + i), hs::LoadU(hd16, sb.data() + i)), hd16, so.data() + i);
      nb::doNotOptimizeAway(so.data());
    });
    bench.run("Highway SaturatedAdd", [&] {
      for (std::size_t i = 0; i + gw16 <= kN; i += gw16)
        hn::StoreU(hn::SaturatedAdd(hn::LoadU(gd16, sa.data() + i), hn::LoadU(gd16, sb.data() + i)), gd16, so.data() + i);
      nb::doNotOptimizeAway(so.data());
    });
  }

  // ---- 10. TableLookupBytes u8 (HYLE vs Highway -- native vpshufb both) ----
  {
    const hs::ScalableTag<std::uint8_t> hd8; const hn::ScalableTag<std::uint8_t> gd8;
    const std::size_t hw8 = hs::Lanes(hd8), gw8 = hn::Lanes(gd8);
    std::vector<std::uint8_t> tb(kN), ib8(kN), ob(kN);
    { std::mt19937_64 rng(13); std::uniform_int_distribution<int> d(0,15); for (std::size_t i=0;i<kN;++i){tb[i]=(std::uint8_t)(i*7); ib8[i]=(std::uint8_t)d(rng);} }
    nb::Bench bench; bench.title("TableLookupBytes u8 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    bench.run("HYLE TableLookupBytes", [&] {
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8)
        hs::StoreU(hs::TableLookupBytes(hs::LoadU(hd8, tb.data() + i), hs::LoadU(hd8, ib8.data() + i)), hd8, ob.data() + i);
      nb::doNotOptimizeAway(ob.data());
    });
    bench.run("Highway TableLookupBytes", [&] {
      for (std::size_t i = 0; i + gw8 <= kN; i += gw8)
        hn::StoreU(hn::TableLookupBytes(hn::LoadU(gd8, tb.data() + i), hn::LoadU(gd8, ib8.data() + i)), gd8, ob.data() + i);
      nb::doNotOptimizeAway(ob.data());
    });
  }

  return 0;
}
