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
#include "xsimd/xsimd.hpp"  // xsimd's op face is narrower (math-focused): only a few structural
                            // ops overlap (compress / reduce_add / zip_lo). Those get a 3rd row.
namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;
namespace nb = ankerl::nanobench;
using xb_i32 = xsimd::batch<std::int32_t>;
using xb_f32 = xsimd::batch<float>;

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

  // ---- 1. Add (pure passthrough: MUST tie -- same vaddps). 3-way: also the public add_f32
  //         batch's primitive (simd-add补齐, compute-07 vecadd 的底层核). xsimd 第三方对照. ----
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
    bench.run("xsimd add", [&] {
      constexpr std::size_t xw = xb_f32::size;
      for (std::size_t i = 0; i + xw <= kN; i += xw)
        (xb_f32::load_unaligned(a.data() + i) + xb_f32::load_unaligned(b.data() + i)).store_unaligned(out.data() + i);
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
    constexpr std::size_t xw = xb_f32::size;
    bench.run("xsimd reduce_add", [&] {
      float acc = 0;
      for (std::size_t i = 0; i + xw <= kN; i += xw)
        acc += xsimd::reduce_add(xb_f32::load_unaligned(a.data() + i));
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
    constexpr std::size_t xw = xb_i32::size;
    bench.run("xsimd compress", [&] {
      for (std::size_t i = 0; i + xw <= kN; i += xw) {
        auto v = xb_i32::load_unaligned(ia.data() + i);
        auto m = xb_i32::load_unaligned(imask.data() + i) > xb_i32(0);  // mask>0
        xsimd::compress(v, m).store_unaligned(iout.data() + i);
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

  // ---- 11. LeadingZeroCount u32 (HYLE store+reload+std vs Highway) ----
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const std::size_t hwu = hs::Lanes(hdu), gwu = hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN), uo(kN);
    { std::mt19937_64 rng(21); std::uniform_int_distribution<std::uint32_t> d(1, 0xFFFFFFFFu); for (auto& e : ua) e = d(rng); }
    nb::Bench bench; bench.title("LeadingZeroCount u32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    bench.run("HYLE LeadingZeroCount", [&] {
      for (std::size_t i = 0; i + hwu <= kN; i += hwu)
        hs::StoreU(hs::LeadingZeroCount(hs::LoadU(hdu, ua.data() + i)), hdu, uo.data() + i);
      nb::doNotOptimizeAway(uo.data());
    });
    bench.run("Highway LeadingZeroCount", [&] {
      for (std::size_t i = 0; i + gwu <= kN; i += gwu)
        hn::StoreU(hn::LeadingZeroCount(hn::LoadU(gdu, ua.data() + i)), gdu, uo.data() + i);
      nb::doNotOptimizeAway(uo.data());
    });
  }

  // ---- 12. Expand i32 (HYLE store+reload vs Highway; inverse of Compress) ----
  {
    const auto imask = geni(kN, 22);
    nb::Bench bench; bench.title("Expand i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
    bench.run("HYLE Expand", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw) {
        auto v = hs::LoadU(hdi, ia.data() + i);
        auto m = hs::Gt(hs::LoadU(hdi, imask.data() + i), hs::Set(hdi, 0));
        hs::StoreU(hs::Expand(v, m), hdi, iout.data() + i);
      }
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway Expand", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw) {
        auto v = hn::LoadU(gdi, ia.data() + i);
        auto m = hn::Gt(hn::LoadU(gdi, imask.data() + i), hn::Set(gdi, 0));
        hn::StoreU(hn::Expand(v, m), gdi, iout.data() + i);
      }
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ---- 13. ConcatEven i32 (HYLE store+reload vs Highway ConcatEven) ----
  {
    const auto ib = geni(kN, 23);
    nb::Bench bench; bench.title("ConcatEven i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE ConcatEven", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::ConcatEven(hs::LoadU(hdi, ia.data() + i), hs::LoadU(hdi, ib.data() + i)), hdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway ConcatEven", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::ConcatEven(gdi, hn::LoadU(gdi, ia.data() + i), hn::LoadU(gdi, ib.data() + i)), gdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ---- 14. InterleaveLower i32 (HYLE native unpack+blend vs Highway) ----
  {
    const auto ib = geni(kN, 24);
    nb::Bench bench; bench.title("InterleaveLower i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE InterleaveLower", [&] {
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::InterleaveLower(hs::LoadU(hdi, ia.data() + i), hs::LoadU(hdi, ib.data() + i)), hdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
    bench.run("Highway InterleaveLower", [&] {
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::InterleaveLower(hn::LoadU(gdi, ia.data() + i), hn::LoadU(gdi, ib.data() + i)), gdi, iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
    constexpr std::size_t xw = xb_i32::size;
    bench.run("xsimd zip_lo", [&] {
      for (std::size_t i = 0; i + xw <= kN; i += xw)
        xsimd::zip_lo(xb_i32::load_unaligned(ia.data() + i), xb_i32::load_unaligned(ib.data() + i)).store_unaligned(iout.data() + i);
      nb::doNotOptimizeAway(iout.data());
    });
  }

  // ---- 15. ReverseLaneBytes u32 (HYLE store+reload vs Highway vpshufb) ----
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const std::size_t hwu = hs::Lanes(hdu), gwu = hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN), uo(kN);
    { std::mt19937_64 rng(25); std::uniform_int_distribution<std::uint32_t> d(0, 0xFFFFFFFFu); for (auto& e : ua) e = d(rng); }
    nb::Bench bench; bench.title("ReverseLaneBytes u32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE ReverseLaneBytes", [&] {
      for (std::size_t i = 0; i + hwu <= kN; i += hwu)
        hs::StoreU(hs::ReverseLaneBytes(hs::LoadU(hdu, ua.data() + i)), hdu, uo.data() + i);
      nb::doNotOptimizeAway(uo.data());
    });
    bench.run("Highway ReverseLaneBytes", [&] {
      for (std::size_t i = 0; i + gwu <= kN; i += gwu)
        hn::StoreU(hn::ReverseLaneBytes(hn::LoadU(gdu, ua.data() + i)), gdu, uo.data() + i);
      nb::doNotOptimizeAway(uo.data());
    });
  }

  // ---- 16. PromoteLowerTo i16->i32 (HYLE vs Highway) ----
  {
    const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
    const hs::ScalableTag<std::int32_t> hd32; const hn::ScalableTag<std::int32_t> gd32;
    const std::size_t hw16 = hs::Lanes(hd16), gw16 = hn::Lanes(gd16);
    std::vector<std::int16_t> s16(kN); std::vector<std::int32_t> o32(kN);
    { std::mt19937_64 rng(26); std::uniform_int_distribution<int> d(-30000,30000); for (auto& e : s16) e = (std::int16_t)d(rng); }
    nb::Bench bench; bench.title("PromoteLowerTo i16->i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE PromoteLowerTo", [&] {
      for (std::size_t i = 0; i + hw16 <= kN; i += hw16)
        hs::StoreU(hs::PromoteLowerTo(hd32, hs::LoadU(hd16, s16.data() + i)), hd32, o32.data() + i);
      nb::doNotOptimizeAway(o32.data());
    });
    bench.run("Highway PromoteLowerTo", [&] {
      for (std::size_t i = 0; i + gw16 <= kN; i += gw16)
        hn::StoreU(hn::PromoteLowerTo(gd32, hn::LoadU(gd16, s16.data() + i)), gd32, o32.data() + i);
      nb::doNotOptimizeAway(o32.data());
    });
  }

  // ---- 17. DemoteToNearestInt f64->i32 (simd-18 @opus). THREE rows:
  //  * HYLE (safe): FULL emu128-spec sign-based saturation (Abs+2 blends -> NaN yields sign-based
  //    MIN/MAX). ~15% slower than Highway's x86 native, but that is a CORRECTNESS difference not a
  //    loss -- Highway native does only Min(v,MAX)+cvtpd2dq (NaN -> INT32_MIN, diverging from its own
  //    emu128 ref). The extra work buys spec-faithful NaN handling.
  //  * HYLE (in-range fast): DemoteToNearestIntInRange = plain RNE + cvtt (2 instrs: vroundpd+
  //    vcvttpd2dq). Caller guarantees finite & in-[MIN,MAX]. This is the InRange-family opt-in
  //    fast path -- should MATCH or beat Highway since it drops even Highway's Min clamp.
  //  * Highway native. ----
  {
    const hs::ScalableTag<double> hdd; const hn::ScalableTag<double> gdd;
    const hs::FixedTag<std::int32_t, hs::kNativeVectorBytes / sizeof(double)> hdi32{};
    const hn::Rebind<std::int32_t, hn::ScalableTag<double>> gdi32{};
    const std::size_t hwd = hs::Lanes(hdd), gwd = hn::Lanes(gdd);
    std::vector<double> da(kN); std::vector<std::int32_t> o32(kN);
    { std::mt19937_64 rng(27); std::uniform_real_distribution<double> d(-1e6, 1e6); for (auto& e : da) e = d(rng); }
    nb::Bench bench; bench.title("DemoteToNearestInt f64->i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE DemoteToNearestInt (safe)", [&] {
      for (std::size_t i = 0; i + hwd <= kN; i += hwd)
        hs::StoreU(hs::DemoteToNearestInt(hdi32, hs::LoadU(hdd, da.data() + i)), hdi32, o32.data() + i);
      nb::doNotOptimizeAway(o32.data());
    });
    bench.run("HYLE DemoteToNearestIntInRange (fast)", [&] {
      for (std::size_t i = 0; i + hwd <= kN; i += hwd)
        hs::StoreU(hs::DemoteToNearestIntInRange(hdi32, hs::LoadU(hdd, da.data() + i)), hdi32, o32.data() + i);
      nb::doNotOptimizeAway(o32.data());
    });
    bench.run("Highway DemoteToNearestInt", [&] {
      for (std::size_t i = 0; i + gwd <= kN; i += gwd)
        hn::StoreU(hn::DemoteToNearestInt(gdi32, hn::LoadU(gdd, da.data() + i)), gdi32, o32.data() + i);
      nb::doNotOptimizeAway(o32.data());
    });
  }

  // ---- 18. DemoteMaskTo i32->i16 (simd-18 @opus). HYLE now stays in the vector domain on vector-mask
  // backends (TruncateTo the mask vector -> native pack) instead of the old bitfield round-trip: that
  // cut the gap from ~6x to ~2x. Residual ~2x is a cold-op register shuffle (vpackusdw + a mov vs
  // Highway's fused pack+movemask), NOT the algorithm. On AVX-512 HYLE keeps the bitfield form which is
  // OPTIMAL (masks ARE the k-register bitfield, zero data movement). Mask width-bridging is a COLD op
  // (sub-ns, not inner-loop), so the residual is accepted; row documents it. ----
  {
    const hs::ScalableTag<std::int32_t> hd32; const hn::ScalableTag<std::int32_t> gd32;
    using HD16 = hs::Rebind<std::int16_t, hs::ScalableTag<std::int32_t>>;
    using GD16 = hn::Rebind<std::int16_t, hn::ScalableTag<std::int32_t>>;
    const HD16 hd16{}; const GD16 gd16{};
    const std::size_t hw32 = hs::Lanes(hd32), gw32 = hn::Lanes(gd32);
    std::vector<std::int32_t> s32(kN);
    { std::mt19937_64 rng(28); std::uniform_int_distribution<int> d(-1000,1000); for (auto& e : s32) e = d(rng); }
    std::uint64_t sink = 0;
    nb::Bench bench; bench.title("DemoteMaskTo i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    bench.run("HYLE DemoteMaskTo", [&] {
      for (std::size_t i = 0; i + hw32 <= kN; i += hw32) {
        auto m = hs::Gt(hs::LoadU(hd32, s32.data() + i), hs::Zero(hd32));
        sink ^= hs::BitsFromMask(hd16, hs::DemoteMaskTo(hd16, hd32, m));
      }
    });
    bench.run("Highway DemoteMaskTo", [&] {
      for (std::size_t i = 0; i + gw32 <= kN; i += gw32) {
        auto m = hn::Gt(hn::LoadU(gd32, s32.data() + i), hn::Zero(gd32));
        sink ^= hn::BitsFromMask(gd16, hn::DemoteMaskTo(gd16, gd32, m));
      }
    });
    nb::doNotOptimizeAway(sink);
  }

  // ---- 19. Interleaved load/store (AoS<->SoA). HYLE now uses Highway's in-register transpose recipe
  // (block transpose via Concat*/Shuffle128 + per-block shuffle_ps/unpack) instead of the old
  // store-to-stack + scalar-gather. Three rows: HYLE (register transpose) vs Highway (same recipe) vs
  // a "store+reload baseline" = the OLD HYLE impl, to quantify what the port bought. float3 (xyz) and
  // float4 (RGBA) are the common consumers. store+reload should be the clear loser (memory round-trip). ----
  {
    std::vector<float> aos3(kN * 3), aos4(kN * 4);
    for (std::size_t i = 0; i < kN * 3; ++i) aos3[i] = static_cast<float>(i);
    for (std::size_t i = 0; i < kN * 4; ++i) aos4[i] = static_cast<float>(i);
    float fsink = 0;
    {
      nb::Bench bench; bench.title("LoadInterleaved3 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE register-transpose", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          hs::VFromD<decltype(hd)> v0, v1, v2;
          hs::LoadInterleaved3(hd, aos3.data() + i * 3, v0, v1, v2);
          fsink += hs::GetLane(hs::Add(hs::Add(v0, v1), v2));
        }
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          hn::VFromD<decltype(gd)> v0, v1, v2;
          hn::LoadInterleaved3(gd, aos3.data() + i * 3, v0, v1, v2);
          fsink += hn::GetLane(hn::Add(hn::Add(v0, v1), v2));
        }
      });
      bench.run("store+reload baseline", [&] {  // the OLD HYLE deinterleave
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          alignas(64) float t0[16], t1[16], t2[16];
          const float* p = aos3.data() + i * 3;
          for (std::size_t j = 0; j < hw; ++j) { t0[j] = p[3*j]; t1[j] = p[3*j+1]; t2[j] = p[3*j+2]; }
          fsink += hs::GetLane(hs::Add(hs::Add(hs::LoadU(hd, t0), hs::LoadU(hd, t1)), hs::LoadU(hd, t2)));
        }
      });
    }
    {
      const auto v0 = hs::LoadU(hd, a.data()), v1 = hs::LoadU(hd, b.data()),
                 v2 = hs::LoadU(hd, c.data()), v3 = hs::LoadU(hd, a.data());
      const auto g0 = hn::LoadU(gd, a.data()), g1 = hn::LoadU(gd, b.data()),
                 g2 = hn::LoadU(gd, c.data()), g3 = hn::LoadU(gd, a.data());
      nb::Bench bench; bench.title("StoreInterleaved4 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE register-transpose", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) hs::StoreInterleaved4(v0, v1, v2, v3, hd, aos4.data() + i * 4);
        nb::doNotOptimizeAway(aos4.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) hn::StoreInterleaved4(g0, g1, g2, g3, gd, aos4.data() + i * 4);
        nb::doNotOptimizeAway(aos4.data());
      });
      bench.run("store+reload baseline", [&] {  // the OLD HYLE interleave
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          alignas(64) float t0[16], t1[16], t2[16], t3[16];
          hs::StoreU(v0, hd, t0); hs::StoreU(v1, hd, t1); hs::StoreU(v2, hd, t2); hs::StoreU(v3, hd, t3);
          float* p = aos4.data() + i * 4;
          for (std::size_t j = 0; j < hw; ++j) { p[4*j]=t0[j]; p[4*j+1]=t1[j]; p[4*j+2]=t2[j]; p[4*j+3]=t3[j]; }
        }
        nb::doNotOptimizeAway(aos4.data());
      });
    }
    nb::doNotOptimizeAway(fsink);
  }

  // ---- 20. Interleaved u8/u16 (byte-table gather path). Same three-way comparison for the 8/16-bit
  // widths that now use Highway's TableLookupBytesOr0 recipe instead of store+reload. LoadInterleaved4
  // u8 is the RGBA-bytes case (image pixels). ----
  {
    const hs::ScalableTag<std::uint8_t> hd8; const hn::ScalableTag<std::uint8_t> gd8;
    const std::size_t hw8 = hs::Lanes(hd8), gw8 = hn::Lanes(gd8);
    std::vector<std::uint8_t> aos(kN * 4);
    for (std::size_t i = 0; i < kN * 4; ++i) aos[i] = static_cast<std::uint8_t>(i * 7 + 1);
    std::uint64_t bsink = 0;
    nb::Bench bench; bench.title("LoadInterleaved4 u8 (4096, RGBA)").relative(true).batch(kN).minEpochIterations(2000);
    bench.run("HYLE register-transpose", [&] {
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8) {
        hs::VFromD<decltype(hd8)> v0,v1,v2,v3; hs::LoadInterleaved4(hd8, aos.data()+i*4, v0,v1,v2,v3);
        bsink ^= hs::GetLane(hs::Add(hs::Add(v0,v1),hs::Add(v2,v3)));
      }
    });
    bench.run("Highway", [&] {
      for (std::size_t i = 0; i + gw8 <= kN; i += gw8) {
        hn::VFromD<decltype(gd8)> v0,v1,v2,v3; hn::LoadInterleaved4(gd8, aos.data()+i*4, v0,v1,v2,v3);
        bsink ^= hn::GetLane(hn::Add(hn::Add(v0,v1),hn::Add(v2,v3)));
      }
    });
    bench.run("store+reload baseline", [&] {  // the OLD HYLE deinterleave
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8) {
        alignas(64) std::uint8_t t0[64],t1[64],t2[64],t3[64];
        const std::uint8_t* p = aos.data()+i*4;
        for (std::size_t j=0;j<hw8;++j){t0[j]=p[4*j];t1[j]=p[4*j+1];t2[j]=p[4*j+2];t3[j]=p[4*j+3];}
        bsink ^= hs::GetLane(hs::Add(hs::Add(hs::LoadU(hd8,t0),hs::LoadU(hd8,t1)),hs::Add(hs::LoadU(hd8,t2),hs::LoadU(hd8,t3))));
      }
    });
    nb::doNotOptimizeAway(bsink);
  }

  // ---- 21. TableLookupLanes / TwoTablesLookupLanes / SetTableIndices (cross-lane permute) ----
  {
    std::vector<std::int32_t> idx(kN), ib(kN);
    { std::mt19937_64 rng(31); std::uniform_int_distribution<int> d(0, static_cast<int>(hw) - 1);
      for (std::size_t i = 0; i < kN; ++i) idx[i] = d(rng);
      d = std::uniform_int_distribution<int>(0, static_cast<int>(2 * hw) - 1);
      for (std::size_t i = 0; i < kN; ++i) ib[i] = d(rng); }
    const auto table2 = geni(kN, 32);
    {
      nb::Bench bench; bench.title("TableLookupLanes i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto v = hs::LoadU(hdi, ia.data() + i);
          auto ind = hs::SetTableIndices(hdi, idx.data() + i);
          hs::StoreU(hs::TableLookupLanes(v, ind), hdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto v = hn::LoadU(gdi, ia.data() + i);
          auto ind = hn::SetTableIndices(gdi, idx.data() + i);
          hn::StoreU(hn::TableLookupLanes(v, ind), gdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
    }
    {
      nb::Bench bench; bench.title("TwoTablesLookupLanes i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto a0 = hs::LoadU(hdi, ia.data() + i), a1 = hs::LoadU(hdi, table2.data() + i);
          auto ind = hs::SetTableIndices(hdi, ib.data() + i);
          hs::StoreU(hs::TwoTablesLookupLanes(hdi, a0, a1, ind), hdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto a0 = hn::LoadU(gdi, ia.data() + i), a1 = hn::LoadU(gdi, table2.data() + i);
          auto ind = hn::SetTableIndices(gdi, ib.data() + i);
          hn::StoreU(hn::TwoTablesLookupLanes(gdi, a0, a1, ind), gdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
    }
  }

  // ---- 22. ScatterIndex / CompressStore / MaskedLoad / MaskedLoadOr / BlendedStore / LoadN / StoreN ----
  {
    std::vector<std::int32_t> table(kN, 0), idxbuf(kN);
    { std::mt19937_64 rng(33); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : idxbuf) e = d(rng); }
    const auto imask = geni(kN, 34);
    {
      nb::Bench bench; bench.title("ScatterIndex i32 (4096)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw)
          hs::ScatterIndex(hs::LoadU(hdi, ia.data() + i), hdi, table.data(), hs::LoadU(hdi, idxbuf.data() + i));
        nb::doNotOptimizeAway(table.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw)
          hn::ScatterIndex(hn::LoadU(gdi, ia.data() + i), gdi, table.data(), hn::LoadU(gdi, idxbuf.data() + i));
        nb::doNotOptimizeAway(table.data());
      });
    }
    {
      std::size_t written = 0;
      nb::Bench bench; bench.title("CompressStore i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&] {
        std::size_t pos = 0;
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto v = hs::LoadU(hdi, ia.data() + i);
          auto m = hs::Gt(hs::LoadU(hdi, imask.data() + i), hs::Set(hdi, 0));
          pos += hs::CompressStore(v, m, hdi, iout.data() + pos);
          if (pos + hw > kN) pos = 0;
        }
        written = pos;
      });
      bench.run("Highway", [&] {
        std::size_t pos = 0;
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto v = hn::LoadU(gdi, ia.data() + i);
          auto m = hn::Gt(hn::LoadU(gdi, imask.data() + i), hn::Set(gdi, 0));
          pos += hn::CompressStore(v, m, gdi, iout.data() + pos);
          if (pos + gw > kN) pos = 0;
        }
        written = pos;
      });
      nb::doNotOptimizeAway(written);
    }
    {
      nb::Bench bench; bench.title("MaskedLoad i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data() + i), hs::Set(hdi, 0));
          hs::StoreU(hs::MaskedLoad(m, hdi, ia.data() + i), hdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data() + i), hn::Set(gdi, 0));
          hn::StoreU(hn::MaskedLoad(m, gdi, ia.data() + i), gdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
    }
    {
      nb::Bench bench; bench.title("MaskedLoadOr i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data() + i), hs::Set(hdi, 0));
          hs::StoreU(hs::MaskedLoadOr(hs::Set(hdi, -77), m, hdi, ia.data() + i), hdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data() + i), hn::Set(gdi, 0));
          hn::StoreU(hn::MaskedLoadOr(hn::Set(gdi, -77), m, gdi, ia.data() + i), gdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
    }
    {
      nb::Bench bench; bench.title("BlendedStore i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data() + i), hs::Set(hdi, 0));
          hs::BlendedStore(hs::LoadU(hdi, ia.data() + i), m, hdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data() + i), hn::Set(gdi, 0));
          hn::BlendedStore(hn::LoadU(gdi, ia.data() + i), m, gdi, iout.data() + i);
        }
        nb::doNotOptimizeAway(iout.data());
      });
    }
    {
      const std::size_t k = hw / 2 == 0 ? 1 : hw / 2;
      nb::Bench bench; bench.title("LoadN/StoreN i32 (half lanes)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&] {
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto v = hs::LoadN(hdi, ia.data() + i, k);
          hs::StoreN(v, hdi, iout.data() + i, k);
        }
        nb::doNotOptimizeAway(iout.data());
      });
      bench.run("Highway", [&] {
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto v = hn::LoadN(gdi, ia.data() + i, k);
          hn::StoreN(v, gdi, iout.data() + i, k);
        }
        nb::doNotOptimizeAway(iout.data());
      });
    }
  }

  // ---- 23. Bit ops: MulHigh i16/u16, MulHigh32, Reverse8, BitShuffle, MultiRotateRight ----
  {
    const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
    const hs::ScalableTag<std::uint16_t> hdu16; const hn::ScalableTag<std::uint16_t> gdu16;
    const std::size_t hw16 = hs::Lanes(hd16), gw16 = hn::Lanes(gd16);
    std::vector<std::int16_t> sa(kN), sb(kN), so(kN);
    std::vector<std::uint16_t> ua(kN), ub(kN), uo(kN);
    { std::mt19937_64 rng(35); std::uniform_int_distribution<int> d(-30000,30000);
      for (std::size_t i=0;i<kN;++i){ sa[i]=(std::int16_t)d(rng); sb[i]=(std::int16_t)d(rng);
        ua[i]=(std::uint16_t)(d(rng)&0xFFFF); ub[i]=(std::uint16_t)(d(rng)&0xFFFF); } }
    {
      nb::Bench bench; bench.title("MulHigh i16 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw16<=kN;i+=hw16) hs::StoreU(hs::MulHigh(hs::LoadU(hd16,sa.data()+i),hs::LoadU(hd16,sb.data()+i)),hd16,so.data()+i); nb::doNotOptimizeAway(so.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw16<=kN;i+=gw16) hn::StoreU(hn::MulHigh(hn::LoadU(gd16,sa.data()+i),hn::LoadU(gd16,sb.data()+i)),gd16,so.data()+i); nb::doNotOptimizeAway(so.data()); });
    }
    {
      nb::Bench bench; bench.title("MulHigh u16 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw16<=kN;i+=hw16) hs::StoreU(hs::MulHigh(hs::LoadU(hdu16,ua.data()+i),hs::LoadU(hdu16,ub.data()+i)),hdu16,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw16<=kN;i+=gw16) hn::StoreU(hn::MulHigh(hn::LoadU(gdu16,ua.data()+i),hn::LoadU(gdu16,ub.data()+i)),gdu16,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    }
  }
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const std::size_t hwu = hs::Lanes(hdu), gwu = hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN), ub(kN), uo(kN);
    { std::mt19937_64 rng(36); for (auto& e : ua) e = static_cast<std::uint32_t>(rng()); for (auto& e : ub) e = static_cast<std::uint32_t>(rng()); }
    nb::Bench bench; bench.title("MulHigh32 u32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hwu<=kN;i+=hwu) hs::StoreU(hs::MulHigh32(hs::LoadU(hdu,ua.data()+i),hs::LoadU(hdu,ub.data()+i)),hdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gwu<=kN;i+=gwu) hn::StoreU(hn::MulHigh(hn::LoadU(gdu,ua.data()+i),hn::LoadU(gdu,ub.data()+i)),gdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
  }
  {
    nb::Bench bench; bench.title("Reverse8 i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::Reverse8(hdi,hs::LoadU(hdi,ia.data()+i)),hdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::Reverse8(gdi,hn::LoadU(gdi,ia.data()+i)),gdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
  }
  // BitShuffle / MultiRotateRight: u64 data + u8 index (8 per data lane)
  {
    const hs::ScalableTag<std::uint64_t> hd64; const hn::ScalableTag<std::uint64_t> gd64;
    const hs::ScalableTag<std::uint8_t> hd8; const hn::ScalableTag<std::uint8_t> gd8;
    const std::size_t hw64 = hs::Lanes(hd64), gw64 = hn::Lanes(gd64);
    const std::size_t hw8 = hs::Lanes(hd8);
    std::vector<std::uint64_t> u64a(kN), u64o(kN);
    std::vector<std::uint8_t> idx8(kN * 8);
    { std::mt19937_64 rng(37); for (auto& e : u64a) e = rng();
      std::uniform_int_distribution<int> d(0, 63); for (auto& e : idx8) e = static_cast<std::uint8_t>(d(rng)); }
    {
      nb::Bench bench; bench.title("BitShuffle u64 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE", [&]{
        for (std::size_t i = 0; i + hw64 <= kN; i += hw64)
          hs::StoreU(hs::BitShuffle(hs::LoadU(hd64, u64a.data()+i), hs::LoadU(hd8, idx8.data()+i*8)), hd64, u64o.data()+i);
        nb::doNotOptimizeAway(u64o.data());
      });
      bench.run("Highway", [&]{
        for (std::size_t i = 0; i + gw64 <= kN; i += gw64)
          hn::StoreU(hn::BitShuffle(hn::LoadU(gd64, u64a.data()+i), hn::LoadU(gd8, idx8.data()+i*8)), gd64, u64o.data()+i);
        nb::doNotOptimizeAway(u64o.data());
      });
    }
    {
      nb::Bench bench; bench.title("MultiRotateRight u64 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE", [&]{
        for (std::size_t i = 0; i + hw64 <= kN; i += hw64)
          hs::StoreU(hs::MultiRotateRight(hs::LoadU(hd64, u64a.data()+i), hs::LoadU(hd8, idx8.data()+i*8)), hd64, u64o.data()+i);
        nb::doNotOptimizeAway(u64o.data());
      });
      bench.run("Highway", [&]{
        for (std::size_t i = 0; i + gw64 <= kN; i += gw64)
          hn::StoreU(hn::MultiRotateRight(hn::LoadU(gd64, u64a.data()+i), hn::LoadU(gd8, idx8.data()+i*8)), gd64, u64o.data()+i);
        nb::doNotOptimizeAway(u64o.data());
      });
    }
    (void)hw8;
  }

  // ---- 24. Interleave2 + StoreInterleaved3 (domain gap) + PromoteMaskTo ----
  {
    std::vector<float> aos2(kN * 2), aos3(kN * 3);
    for (std::size_t i = 0; i < kN * 2; ++i) aos2[i] = static_cast<float>(i);
    for (std::size_t i = 0; i < kN * 3; ++i) aos3[i] = static_cast<float>(i);
    float fsink = 0;
    {
      nb::Bench bench; bench.title("LoadInterleaved2 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw){ hs::VFromD<decltype(hd)> v0,v1; hs::LoadInterleaved2(hd,aos2.data()+i*2,v0,v1); fsink+=hs::GetLane(hs::Add(v0,v1)); }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw){ hn::VFromD<decltype(gd)> v0,v1; hn::LoadInterleaved2(gd,aos2.data()+i*2,v0,v1); fsink+=hn::GetLane(hn::Add(v0,v1)); }
      });
    }
    {
      const auto v0=hs::LoadU(hd,a.data()), v1=hs::LoadU(hd,b.data()), v2=hs::LoadU(hd,c.data());
      const auto g0=hn::LoadU(gd,a.data()), g1=hn::LoadU(gd,b.data()), g2=hn::LoadU(gd,c.data());
      nb::Bench bench; bench.title("StoreInterleaved2 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreInterleaved2(v0,v1,hd,aos2.data()+i*2); nb::doNotOptimizeAway(aos2.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreInterleaved2(g0,g1,gd,aos2.data()+i*2); nb::doNotOptimizeAway(aos2.data()); });
    }
    {
      const auto v0=hs::LoadU(hd,a.data()), v1=hs::LoadU(hd,b.data()), v2=hs::LoadU(hd,c.data());
      const auto g0=hn::LoadU(gd,a.data()), g1=hn::LoadU(gd,b.data()), g2=hn::LoadU(gd,c.data());
      nb::Bench bench; bench.title("StoreInterleaved3 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreInterleaved3(v0,v1,v2,hd,aos3.data()+i*3); nb::doNotOptimizeAway(aos3.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreInterleaved3(g0,g1,g2,gd,aos3.data()+i*3); nb::doNotOptimizeAway(aos3.data()); });
    }
    nb::doNotOptimizeAway(fsink);
  }
  // ---- 25. Mask domain (DemoteMaskTo already §18): Promote / OrderedDemote2 / Combine / Half /
  // Slide / Rebind / Set / Dup128MaskFromMaskBits. Sink via BitsFromMask so the mask isn't DCE'd. ----
  {
    std::uint64_t sink = 0;
    const auto imask = geni(kN, 39);
    // PromoteMaskTo: Rebind i16 (same lane count as i32) -> ScalableTag i32
    {
      using HD16 = hs::Rebind<std::int16_t, hs::ScalableTag<std::int32_t>>;
      using GD16 = hn::Rebind<std::int16_t, hn::ScalableTag<std::int32_t>>;
      const HD16 hd16r{}; const GD16 gd16r{};
      std::vector<std::int16_t> s16(kN);
      { std::mt19937_64 rng(40); std::uniform_int_distribution<int> d(-1000,1000); for (auto& e : s16) e = (std::int16_t)d(rng); }
      nb::Bench bench; bench.title("PromoteMaskTo i16->i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hd16r, s16.data()+i), hs::Set(hd16r, std::int16_t{0}));
          sink ^= hs::BitsFromMask(hdi, hs::PromoteMaskTo(hdi, hd16r, m));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gd16r, s16.data()+i), hn::Set(gd16r, std::int16_t{0}));
          sink ^= hn::BitsFromMask(gdi, hn::PromoteMaskTo(gdi, gd16r, m));
        }
      });
    }
    // OrderedDemote2MasksTo i32+i32 -> i16 full
    {
      using HD16 = hs::RepartitionToNarrow<hs::ScalableTag<std::int32_t>>;
      using GD16 = hn::RepartitionToNarrow<hn::ScalableTag<std::int32_t>>;
      // Actually OrderedDemote2MasksTo wants DTo with 2N lanes of half width = full narrow ScalableTag i16
      const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
      nb::Bench bench; bench.title("OrderedDemote2MasksTo i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+2*hw<=kN;i+=2*hw) {
          auto ma = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          auto mb = hs::Gt(hs::LoadU(hdi, imask.data()+i+hw), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hd16, hs::OrderedDemote2MasksTo(hd16, hdi, ma, mb));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+2*gw<=kN;i+=2*gw) {
          auto ma = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          auto mb = hn::Gt(hn::LoadU(gdi, imask.data()+i+gw), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gd16, hn::OrderedDemote2MasksTo(gd16, gdi, ma, mb));
        }
      });
    }
    // CombineMasks / LowerHalfOfMask / UpperHalfOfMask
    {
      const hs::Half<hs::ScalableTag<std::int32_t>> hdh; const hn::Half<hn::ScalableTag<std::int32_t>> gdh;
      const std::size_t hwh = hs::Lanes(hdh), gwh = hn::Lanes(gdh);
      nb::Bench bench; bench.title("CombineMasks i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto full = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          auto lo = hs::LowerHalfOfMask(hdh, full);
          auto hi = hs::UpperHalfOfMask(hdh, full);
          sink ^= hs::BitsFromMask(hdi, hs::CombineMasks(hdi, hi, lo));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto full = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          auto lo = hn::LowerHalfOfMask(gdh, full);
          auto hi = hn::UpperHalfOfMask(gdh, full);
          sink ^= hn::BitsFromMask(gdi, hn::CombineMasks(gdi, hi, lo));
        }
      });
      (void)hwh; (void)gwh;
    }
    // SlideMask1Up/Down + SlideMaskUp/DownLanes
    {
      nb::Bench bench; bench.title("SlideMask1Up i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMask1Up(hdi, m));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMask1Up(gdi, m));
        }
      });
    }
    {
      nb::Bench bench; bench.title("SlideMask1Down i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMask1Down(hdi, m));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMask1Down(gdi, m));
        }
      });
    }
    {
      nb::Bench bench; bench.title("SlideMaskUpLanes i32 amt=2 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMaskUpLanes(hdi, m, 2));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMaskUpLanes(gdi, m, 2));
        }
      });
    }
    {
      nb::Bench bench; bench.title("SlideMaskDownLanes i32 amt=2 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMaskDownLanes(hdi, m, 2));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMaskDownLanes(gdi, m, 2));
        }
      });
    }
    // RebindMask i32 -> f32, SetMask, Dup128MaskFromMaskBits
    {
      nb::Bench bench; bench.title("RebindMask i32->f32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, imask.data()+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hd, hs::RebindMask(hd, m));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, imask.data()+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gd, hn::RebindMask(gd, m));
        }
      });
    }
    {
      nb::Bench bench; bench.title("SetMask i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink ^= hs::BitsFromMask(hdi, hs::SetMask(hdi, (imask[i] & 1) != 0));
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink ^= hn::BitsFromMask(gdi, hn::SetMask(gdi, (imask[i] & 1) != 0));
      });
    }
    {
      nb::Bench bench; bench.title("Dup128MaskFromMaskBits i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink ^= hs::BitsFromMask(hdi, hs::Dup128MaskFromMaskBits(hdi, static_cast<unsigned>(imask[i] & 0xF)));
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink ^= hn::BitsFromMask(gdi, hn::Dup128MaskFromMaskBits(gdi, static_cast<unsigned>(imask[i] & 0xF)));
      });
    }
    nb::doNotOptimizeAway(sink);
  }

  // ---- 26. Single-vector PromoteTo/DemoteTo/ShiftRightAndDemoteTo (Highway Rebind form) ----
  // Avoid Store of partial-tag results (fixed-width Store may write a full register). Sink via GetLane.
  {
    using HD16 = hs::Rebind<std::int16_t, hs::ScalableTag<std::int32_t>>;
    using GD16 = hn::Rebind<std::int16_t, hn::ScalableTag<std::int32_t>>;
    const HD16 hd16r{}; const GD16 gd16r{};
    const hs::ScalableTag<std::int16_t> hd16full; const hn::ScalableTag<std::int16_t> gd16full;
    std::vector<std::int16_t> s16(kN);
    { std::mt19937_64 rng(41); std::uniform_int_distribution<int> d(-30000,30000); for (auto& e : s16) e = (std::int16_t)d(rng); }
    std::int64_t sink = 0;
    {
      nb::Bench bench; bench.title("PromoteTo i16->i32 1-arg (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto half = hs::LowerHalf(hs::LoadU(hd16full, s16.data()+i));
          sink += hs::GetLane(hs::PromoteTo(hdi, half));
        }
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto half = hn::LowerHalf(hn::Half<decltype(gd16full)>(), hn::LoadU(gd16full, s16.data()+i));
          sink += hn::GetLane(hn::PromoteTo(gdi, half));
        }
      });
    }
    {
      nb::Bench bench; bench.title("DemoteTo i32->i16 1-arg (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink += hs::GetLane(hs::DemoteTo(hd16r, hs::LoadU(hdi, ia.data()+i)));
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink += hn::GetLane(hn::DemoteTo(gd16r, hn::LoadU(gdi, ia.data()+i)));
      });
    }
    {
      nb::Bench bench; bench.title("ShiftRightAndDemoteTo<3> i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink += hs::GetLane(hs::ShiftRightAndDemoteTo<3>(hd16r, hs::LoadU(hdi, ia.data()+i)));
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink += hn::GetLane(hn::ShiftRightAndDemoteTo<3>(gd16r, hn::LoadU(gdi, ia.data()+i)));
      });
    }
    {
      nb::Bench bench; bench.title("RoundingShiftRightAndDemoteTo<3> i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink += hs::GetLane(hs::RoundingShiftRightAndDemoteTo<3>(hd16r, hs::LoadU(hdi, ia.data()+i)));
      });
      bench.run("Highway", [&]{
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink += hn::GetLane(hn::RoundingShiftRightAndDemoteTo<3>(gd16r, hn::LoadU(gdi, ia.data()+i)));
      });
    }
    nb::doNotOptimizeAway(sink);
  }

  // ---- 27. GatherOffset / ScatterOffset (BYTE-offset gather/scatter, generic store+reload) ----
  {
    std::vector<std::int32_t> table(kN), off(kN), sink_tbl(kN, 0);
    for (std::size_t i = 0; i < kN; ++i) table[i] = static_cast<std::int32_t>(i * 5 + 2);
    { std::mt19937_64 rng(71); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : off) e = d(rng) * static_cast<int>(sizeof(std::int32_t)); }  // byte offsets
    {
      nb::Bench bench; bench.title("GatherOffset i32 (4096)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::GatherOffset(hdi,table.data(),hs::LoadU(hdi,off.data()+i)),hdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::GatherOffset(gdi,table.data(),hn::LoadU(gdi,off.data()+i)),gdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
    }
    {
      nb::Bench bench; bench.title("ScatterOffset i32 (4096)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::ScatterOffset(hs::LoadU(hdi,ia.data()+i),hdi,sink_tbl.data(),hs::LoadU(hdi,off.data()+i)); nb::doNotOptimizeAway(sink_tbl.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::ScatterOffset(hn::LoadU(gdi,ia.data()+i),gdi,sink_tbl.data(),hn::LoadU(gdi,off.data()+i)); nb::doNotOptimizeAway(sink_tbl.data()); });
    }
  }

  // ---- 28. CompressBits / CompressNot / CompressBlendedStore / LoadExpand ----
  {
    const auto imask = geni(kN, 72);
    // CompressBits: mask packed as bytes (1 bit/lane). Build per-block byte-mask for both.
    {
      nb::Bench bench; bench.title("CompressNot i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,imask.data()+i),hs::Set(hdi,0)); hs::StoreU(hs::CompressNot(hs::LoadU(hdi,ia.data()+i),m),hdi,iout.data()+i);} nb::doNotOptimizeAway(iout.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,imask.data()+i),hn::Set(gdi,0)); hn::StoreU(hn::CompressNot(hn::LoadU(gdi,ia.data()+i),m),gdi,iout.data()+i);} nb::doNotOptimizeAway(iout.data()); });
    }
    {
      std::size_t w=0;
      nb::Bench bench; bench.title("CompressBlendedStore i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ std::size_t pos=0; for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,imask.data()+i),hs::Set(hdi,0)); pos+=hs::CompressBlendedStore(hs::LoadU(hdi,ia.data()+i),m,hdi,iout.data()+(pos+hw<=kN?pos:0));} w=pos; });
      bench.run("Highway", [&]{ std::size_t pos=0; for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,imask.data()+i),hn::Set(gdi,0)); pos+=hn::CompressBlendedStore(hn::LoadU(gdi,ia.data()+i),m,gdi,iout.data()+(pos+gw<=kN?pos:0));} w=pos; });
      nb::doNotOptimizeAway(w);
    }
    {
      nb::Bench bench; bench.title("LoadExpand i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,imask.data()+i),hs::Set(hdi,0)); hs::StoreU(hs::LoadExpand(m,hdi,ia.data()+i),hdi,iout.data()+i);} nb::doNotOptimizeAway(iout.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,imask.data()+i),hn::Set(gdi,0)); hn::StoreU(hn::LoadExpand(m,gdi,ia.data()+i),gdi,iout.data()+i);} nb::doNotOptimizeAway(iout.data()); });
    }
  }

  // ---- 29. N-variant gather/scatter/load (partial, runtime n; index-based) ----
  {
    std::vector<std::int32_t> table(kN), idxb(kN), stbl(kN, 0);
    for (std::size_t i = 0; i < kN; ++i) table[i] = static_cast<std::int32_t>(i * 7 + 3);
    { std::mt19937_64 rng(73); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : idxb) e = d(rng); }
    const std::size_t kNlan = hw / 2 ? hw / 2 : 1;
    {
      nb::Bench bench; bench.title("GatherIndexN i32 (half n)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::GatherIndexN(hdi,table.data(),hs::LoadU(hdi,idxb.data()+i),kNlan),hdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::GatherIndexN(gdi,table.data(),hn::LoadU(gdi,idxb.data()+i),kNlan),gdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
    }
    {
      nb::Bench bench; bench.title("GatherIndexNOr i32 (half n)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::GatherIndexNOr(hs::Set(hdi,-1),hdi,table.data(),hs::LoadU(hdi,idxb.data()+i),kNlan),hdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::GatherIndexNOr(hn::Set(gdi,-1),gdi,table.data(),hn::LoadU(gdi,idxb.data()+i),kNlan),gdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
    }
    {
      nb::Bench bench; bench.title("ScatterIndexN i32 (half n)").relative(true).batch(kN).minEpochIterations(3000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::ScatterIndexN(hs::LoadU(hdi,ia.data()+i),hdi,stbl.data(),hs::LoadU(hdi,idxb.data()+i),kNlan); nb::doNotOptimizeAway(stbl.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::ScatterIndexN(hn::LoadU(gdi,ia.data()+i),gdi,stbl.data(),hn::LoadU(gdi,idxb.data()+i),kNlan); nb::doNotOptimizeAway(stbl.data()); });
    }
    {
      nb::Bench bench; bench.title("LoadNOr i32 (half n)").relative(true).batch(kN).minEpochIterations(5000);
      bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::LoadNOr(hs::Set(hdi,-1),hdi,ia.data()+i,kNlan),hdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
      bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::LoadNOr(hn::Set(gdi,-1),gdi,ia.data()+i,kNlan),gdi,iout.data()+i); nb::doNotOptimizeAway(iout.data()); });
    }
  }

  return 0;
}
