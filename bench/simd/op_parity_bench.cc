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
// Reads (via scripts/op-parity-report.py v4 ONLY -- raw rows carry per-replica placement bias, see
// docs/debug/op-parity-harness-artifacts-hunt.md): ratio >1 => HYLE faster; <1 => HYLE SLOWER (the case
// that would justify switching that op to a Highway call); ~1 => tie (self-made costs nothing,
// keep it for the dtype/scalable control it buys).
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "hyle/simd/vec.h"

// Highway at the SAME ISA. HWY_COMPILE_ONLY_* not needed: we include the static header and pin the
// target via the compile flags (same -march this TU gets), so hn:: == HWY_NAMESPACE resolves to
// the matching target and inlines.
#include "hwy/highway.h"
#include "xsimd/xsimd.hpp"  // xsimd's op face is narrower (math-focused): only a few structural
                            // ops overlap (compress / reduce_add / zip_lo). Those get a 3rd row.
#include "op_parity_engine.h"
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
  hyle_bench::pin_and_boost();
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
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_a = a.data(); auto* p_b = b.data(); auto* p_out = out.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::Add(hs::LoadU(hd, p_a + i), hs::LoadU(hd, p_b + i)), hd, p_out + i);
      nb::doNotOptimizeAway(p_out);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_a = a.data(); auto* p_b = b.data(); auto* p_out = out.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::Add(hn::LoadU(gd, p_a + i), hn::LoadU(gd, p_b + i)), gd, p_out + i);
      nb::doNotOptimizeAway(p_out);
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "xsimd add", [&] { auto* p_a = a.data(); auto* p_b = b.data(); auto* p_out = out.data();
      constexpr std::size_t xw = xb_f32::size;
      for (std::size_t i = 0; i + xw <= kN; i += xw)
        (xb_f32::load_unaligned(p_a + i) + xb_f32::load_unaligned(p_b + i)).store_unaligned(p_out + i);
      nb::doNotOptimizeAway(p_out);
    });
  }

  // ---- 2. MulAdd / FMA (passthrough: MUST tie -- same vfmadd) ----
  {
    nb::Bench bench; bench.title("MulAdd f32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_a = a.data(); auto* p_b = b.data(); auto* p_c = c.data(); auto* p_out = out.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::MulAdd(hs::LoadU(hd, p_a + i), hs::LoadU(hd, p_b + i), hs::LoadU(hd, p_c + i)), hd, p_out + i);
      nb::doNotOptimizeAway(p_out);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_a = a.data(); auto* p_b = b.data(); auto* p_c = c.data(); auto* p_out = out.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::MulAdd(hn::LoadU(gd, p_a + i), hn::LoadU(gd, p_b + i), hn::LoadU(gd, p_c + i)), gd, p_out + i);
      nb::doNotOptimizeAway(p_out);
     hyle_bench::mark(rep); });
  }

  // ---- 3. ReduceSum (HYLE self-made horizontal reduce vs Highway ReduceSum) ----
  {
    nb::Bench bench; bench.title("ReduceSum f32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_a = a.data();
      float acc = 0;
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        acc += hs::ReduceSum(hd, hs::LoadU(hd, p_a + i));
      nb::doNotOptimizeAway(acc);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_a = a.data();
      float acc = 0;
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        acc += hn::ReduceSum(gd, hn::LoadU(gd, p_a + i));
      nb::doNotOptimizeAway(acc);
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "xsimd reduce_add", [&] { auto* p_a = a.data();
      constexpr std::size_t xw = xb_f32::size;
      float acc = 0;
      for (std::size_t i = 0; i + xw <= kN; i += xw)
        acc += xsimd::reduce_add(xb_f32::load_unaligned(p_a + i));
      nb::doNotOptimizeAway(acc);
    });
  }

  // ---- 4. Reverse (HYLE swizzle vs Highway) ----
  {
    nb::Bench bench; bench.title("Reverse i32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::Reverse(hdi, hs::LoadU(hdi, p_ia + i)), hdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::Reverse(gdi, hn::LoadU(gdi, p_ia + i)), gdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
  }

  // ---- 5. Min (int -- passthrough vpminsd, should tie) ----
  {
    const auto ib = geni(kN, 5);
    nb::Bench bench; bench.title("Min i32 (4096)").relative(true).batch(kN).minEpochIterations(20000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::Min(hs::LoadU(hdi, p_ia + i), hs::LoadU(hdi, p_ib + i)), hdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::Min(hn::LoadU(gdi, p_ia + i), hn::LoadU(gdi, p_ib + i)), gdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
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
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ua = ua.data(); auto* p_uo = uo.data();
      for (std::size_t i = 0; i + hwu <= kN; i += hwu)
        hs::StoreU(hs::PopulationCount(hs::LoadU(hdu, p_ua + i)), hdu, p_uo + i);
      nb::doNotOptimizeAway(p_uo);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ua = ua.data(); auto* p_uo = uo.data();
      for (std::size_t i = 0; i + gwu <= kN; i += gwu)
        hn::StoreU(hn::PopulationCount(hn::LoadU(gdu, p_ua + i)), gdu, p_uo + i);
      nb::doNotOptimizeAway(p_uo);
     hyle_bench::mark(rep); });
  }

  // ---- 7. Compress i32 (HYLE store+reload vs Highway; AVX2 has no VBMI2 so both emulate) ----
  {
    const auto imask = geni(kN, 7);
    nb::Bench bench; bench.title("Compress i32 (4096, 50% dense)").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw) {
        auto v = hs::LoadU(hdi, p_ia + i);
        auto m = hs::Gt(hs::LoadU(hdi, p_imask + i), hs::Set(hdi, 0));
        hs::StoreU(hs::Compress(v, m), hdi, p_iout + i);
      }
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw) {
        auto v = hn::LoadU(gdi, p_ia + i);
        auto m = hn::Gt(hn::LoadU(gdi, p_imask + i), hn::Set(gdi, 0));
        hn::StoreU(hn::Compress(v, m), gdi, p_iout + i);
      }
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "xsimd compress", [&] { auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
      constexpr std::size_t xw = xb_i32::size;
      for (std::size_t i = 0; i + xw <= kN; i += xw) {
        auto v = xb_i32::load_unaligned(p_ia + i);
        auto m = xb_i32::load_unaligned(p_imask + i) > xb_i32(0);  // mask>0
        xsimd::compress(v, m).store_unaligned(p_iout + i);
      }
      nb::doNotOptimizeAway(p_iout);
    });
  }

  // ---- 8. GatherIndex i32 (HYLE native _mm256_i32gather vs Highway GatherIndex) ----
  {
    std::vector<std::int32_t> table(kN);
    for (std::size_t i = 0; i < kN; ++i) table[i] = static_cast<std::int32_t>(i * 3 + 1);
    std::vector<std::int32_t> idxbuf(kN);
    { std::mt19937_64 rng(9); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : idxbuf) e = d(rng); }
    nb::Bench bench; bench.title("GatherIndex i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_idxbuf = idxbuf.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::GatherIndex(hdi, p_table, hs::LoadU(hdi, p_idxbuf + i)), hdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_idxbuf = idxbuf.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::GatherIndex(gdi, p_table, hn::LoadU(gdi, p_idxbuf + i)), gdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
  }

  // ---- coverage re-audit (2026-07-14): masked gather/scatter (HYLE scalar fallback vs Highway native) ----
  {
    std::vector<std::int32_t> table(kN);
    for (std::size_t i = 0; i < kN; ++i) table[i] = static_cast<std::int32_t>(i * 3 + 1);
    std::vector<std::int32_t> idxbuf(kN);
    { std::mt19937_64 rng(19); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : idxbuf) e = d(rng); }
    {
      nb::Bench bench; bench.title("MaskedGatherIndex i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_t=table.data(); auto* p_ix=idxbuf.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw){ auto ix=hs::LoadU(hdi,p_ix+i); hs::StoreU(hs::MaskedGatherIndex(hs::Gt(ix,hs::Set(hdi,(std::int32_t)(kN/2))),hdi,p_t,ix),hdi,p_o+i);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_t=table.data(); auto* p_ix=idxbuf.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw){ auto ix=hn::LoadU(gdi,p_ix+i); hn::StoreU(hn::MaskedGatherIndex(hn::Gt(ix,hn::Set(gdi,(std::int32_t)(kN/2))),gdi,p_t,ix),gdi,p_o+i);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("MaskedGatherIndexOr i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_t=table.data(); auto* p_ix=idxbuf.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw){ auto ix=hs::LoadU(hdi,p_ix+i); hs::StoreU(hs::MaskedGatherIndexOr(hs::Set(hdi,-1),hs::Gt(ix,hs::Set(hdi,(std::int32_t)(kN/2))),hdi,p_t,ix),hdi,p_o+i);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_t=table.data(); auto* p_ix=idxbuf.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw){ auto ix=hn::LoadU(gdi,p_ix+i); hn::StoreU(hn::MaskedGatherIndexOr(hn::Set(gdi,-1),hn::Gt(ix,hn::Set(gdi,(std::int32_t)(kN/2))),gdi,p_t,ix),gdi,p_o+i);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("MaskedScatterIndex i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); auto* p_ix=idxbuf.data(); auto* p_t=table.data(); for(std::size_t i=0;i+hw<=kN;i+=hw){ auto ix=hs::LoadU(hdi,p_ix+i); hs::MaskedScatterIndex(hs::LoadU(hdi,p_ia+i),hs::Gt(ix,hs::Set(hdi,(std::int32_t)(kN/2))),hdi,p_t,ix);} nb::doNotOptimizeAway(p_t);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); auto* p_ix=idxbuf.data(); auto* p_t=table.data(); for(std::size_t i=0;i+gw<=kN;i+=gw){ auto ix=hn::LoadU(gdi,p_ix+i); hn::MaskedScatterIndex(hn::LoadU(gdi,p_ia+i),hn::Gt(ix,hn::Set(gdi,(std::int32_t)(kN/2))),gdi,p_t,ix);} nb::doNotOptimizeAway(p_t);  hyle_bench::mark(rep); });
    }
  }

  // ---- 9. SaturatedAdd i16 (HYLE vs Highway -- both should be native vpaddsw) ----
  {
    const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
    const std::size_t hw16 = hs::Lanes(hd16), gw16 = hn::Lanes(gd16);
    std::vector<std::int16_t> sa(kN), sb(kN), so(kN);
    { std::mt19937_64 rng(11); std::uniform_int_distribution<int> d(-30000, 30000); for (std::size_t i=0;i<kN;++i){sa[i]=(std::int16_t)d(rng); sb[i]=(std::int16_t)d(rng);} }
    nb::Bench bench; bench.title("SaturatedAdd i16 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_sa = sa.data(); auto* p_sb = sb.data(); auto* p_so = so.data();
      for (std::size_t i = 0; i + hw16 <= kN; i += hw16)
        hs::StoreU(hs::SaturatedAdd(hs::LoadU(hd16, p_sa + i), hs::LoadU(hd16, p_sb + i)), hd16, p_so + i);
      nb::doNotOptimizeAway(p_so);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_sa = sa.data(); auto* p_sb = sb.data(); auto* p_so = so.data();
      for (std::size_t i = 0; i + gw16 <= kN; i += gw16)
        hn::StoreU(hn::SaturatedAdd(hn::LoadU(gd16, p_sa + i), hn::LoadU(gd16, p_sb + i)), gd16, p_so + i);
      nb::doNotOptimizeAway(p_so);
     hyle_bench::mark(rep); });
  }

  // ---- 10. TableLookupBytes u8. The PAIR compares contract-equivalent ops: HYLE
  // TableLookupBytesInRange (caller guarantees idx in [0,15]) vs Highway TableLookupBytes (same
  // precondition family) -- raw vpshufb both, MUST tie. HYLE's default TableLookupBytes promises
  // &15 for ANY index byte (stronger than Highway; oracle-fixed semantics) at +1 vpand; it rides
  // as the informational third row so its price stays visible. ----
  {
    const hs::ScalableTag<std::uint8_t> hd8; const hn::ScalableTag<std::uint8_t> gd8;
    const std::size_t hw8 = hs::Lanes(hd8), gw8 = hn::Lanes(gd8);
    std::vector<std::uint8_t> tb(kN), ib8(kN), ob(kN);
    { std::mt19937_64 rng(13); std::uniform_int_distribution<int> d(0,15); for (std::size_t i=0;i<kN;++i){tb[i]=(std::uint8_t)(i*7); ib8[i]=(std::uint8_t)d(rng);} }
    nb::Bench bench; bench.title("TableLookupBytes u8 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_tb = tb.data(); auto* p_ib8 = ib8.data(); auto* p_ob = ob.data();
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8)
        hs::StoreU(hs::TableLookupBytesInRange(hs::LoadU(hd8, p_tb + i), hs::LoadU(hd8, p_ib8 + i)), hd8, p_ob + i);
      nb::doNotOptimizeAway(p_ob);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_tb = tb.data(); auto* p_ib8 = ib8.data(); auto* p_ob = ob.data();
      for (std::size_t i = 0; i + gw8 <= kN; i += gw8)
        hn::StoreU(hn::TableLookupBytes(hn::LoadU(gd8, p_tb + i), hn::LoadU(gd8, p_ib8 + i)), gd8, p_ob + i);
      nb::doNotOptimizeAway(p_ob);
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "HYLE TableLookupBytes (&15 semantics)", [&] { auto* p_tb = tb.data(); auto* p_ib8 = ib8.data(); auto* p_ob = ob.data();
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8)
        hs::StoreU(hs::TableLookupBytes(hs::LoadU(hd8, p_tb + i), hs::LoadU(hd8, p_ib8 + i)), hd8, p_ob + i);
      nb::doNotOptimizeAway(p_ob);
    });
  }

  // ---- 11. LeadingZeroCount u32 (HYLE store+reload+std vs Highway) ----
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const std::size_t hwu = hs::Lanes(hdu), gwu = hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN), uo(kN);
    { std::mt19937_64 rng(21); std::uniform_int_distribution<std::uint32_t> d(1, 0xFFFFFFFFu); for (auto& e : ua) e = d(rng); }
    nb::Bench bench; bench.title("LeadingZeroCount u32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ua = ua.data(); auto* p_uo = uo.data();
      for (std::size_t i = 0; i + hwu <= kN; i += hwu)
        hs::StoreU(hs::LeadingZeroCount(hs::LoadU(hdu, p_ua + i)), hdu, p_uo + i);
      nb::doNotOptimizeAway(p_uo);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ua = ua.data(); auto* p_uo = uo.data();
      for (std::size_t i = 0; i + gwu <= kN; i += gwu)
        hn::StoreU(hn::LeadingZeroCount(hn::LoadU(gdu, p_ua + i)), gdu, p_uo + i);
      nb::doNotOptimizeAway(p_uo);
     hyle_bench::mark(rep); });
  }

  // ---- 12. Expand i32 (HYLE store+reload vs Highway; inverse of Compress) ----
  {
    const auto imask = geni(kN, 22);
    nb::Bench bench; bench.title("Expand i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw) {
        auto v = hs::LoadU(hdi, p_ia + i);
        auto m = hs::Gt(hs::LoadU(hdi, p_imask + i), hs::Set(hdi, 0));
        hs::StoreU(hs::Expand(v, m), hdi, p_iout + i);
      }
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw) {
        auto v = hn::LoadU(gdi, p_ia + i);
        auto m = hn::Gt(hn::LoadU(gdi, p_imask + i), hn::Set(gdi, 0));
        hn::StoreU(hn::Expand(v, m), gdi, p_iout + i);
      }
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
  }

  // ---- 13. ConcatEven i32 (HYLE store+reload vs Highway ConcatEven) ----
  {
    const auto ib = geni(kN, 23);
    nb::Bench bench; bench.title("ConcatEven i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::ConcatEven(hs::LoadU(hdi, p_ia + i), hs::LoadU(hdi, p_ib + i)), hdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::ConcatEven(gdi, hn::LoadU(gdi, p_ia + i), hn::LoadU(gdi, p_ib + i)), gdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
  }

  // ---- 14. InterleaveLower i32 (HYLE native unpack+blend vs Highway) ----
  {
    const auto ib = geni(kN, 24);
    nb::Bench bench; bench.title("InterleaveLower i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + hw <= kN; i += hw)
        hs::StoreU(hs::InterleaveLower(hs::LoadU(hdi, p_ia + i), hs::LoadU(hdi, p_ib + i)), hdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      for (std::size_t i = 0; i + gw <= kN; i += gw)
        hn::StoreU(hn::InterleaveLower(hn::LoadU(gdi, p_ia + i), hn::LoadU(gdi, p_ib + i)), gdi, p_iout + i);
      nb::doNotOptimizeAway(p_iout);
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "xsimd zip_lo", [&] { auto* p_ia = ia.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
      constexpr std::size_t xw = xb_i32::size;
      for (std::size_t i = 0; i + xw <= kN; i += xw)
        xsimd::zip_lo(xb_i32::load_unaligned(p_ia + i), xb_i32::load_unaligned(p_ib + i)).store_unaligned(p_iout + i);
      nb::doNotOptimizeAway(p_iout);
    });
  }

  // ---- 15. ReverseLaneBytes u32 (HYLE store+reload vs Highway vpshufb) ----
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const std::size_t hwu = hs::Lanes(hdu), gwu = hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN), uo(kN);
    { std::mt19937_64 rng(25); std::uniform_int_distribution<std::uint32_t> d(0, 0xFFFFFFFFu); for (auto& e : ua) e = d(rng); }
    nb::Bench bench; bench.title("ReverseLaneBytes u32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ua = ua.data(); auto* p_uo = uo.data();
      for (std::size_t i = 0; i + hwu <= kN; i += hwu)
        hs::StoreU(hs::ReverseLaneBytes(hs::LoadU(hdu, p_ua + i)), hdu, p_uo + i);
      nb::doNotOptimizeAway(p_uo);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_ua = ua.data(); auto* p_uo = uo.data();
      for (std::size_t i = 0; i + gwu <= kN; i += gwu)
        hn::StoreU(hn::ReverseLaneBytes(hn::LoadU(gdu, p_ua + i)), gdu, p_uo + i);
      nb::doNotOptimizeAway(p_uo);
     hyle_bench::mark(rep); });
  }

  // ---- 16. PromoteLowerTo i16->i32 (HYLE vs Highway) ----
  {
    const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
    const hs::ScalableTag<std::int32_t> hd32; const hn::ScalableTag<std::int32_t> gd32;
    const std::size_t hw16 = hs::Lanes(hd16), gw16 = hn::Lanes(gd16);
    std::vector<std::int16_t> s16(kN); std::vector<std::int32_t> o32(kN);
    { std::mt19937_64 rng(26); std::uniform_int_distribution<int> d(-30000,30000); for (auto& e : s16) e = (std::int16_t)d(rng); }
    nb::Bench bench; bench.title("PromoteLowerTo i16->i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_s16 = s16.data(); auto* p_o32 = o32.data();
      for (std::size_t i = 0; i + hw16 <= kN; i += hw16)
        hs::StoreU(hs::PromoteLowerTo(hd32, hs::LoadU(hd16, p_s16 + i)), hd32, p_o32 + i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_s16 = s16.data(); auto* p_o32 = o32.data();
      for (std::size_t i = 0; i + gw16 <= kN; i += gw16)
        hn::StoreU(hn::PromoteLowerTo(gd32, hn::LoadU(gd16, p_s16 + i)), gd32, p_o32 + i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
  }

  // ---- 17. DemoteToNearestInt f64->i32 (simd-18 @opus; semantics decision 2026-07-14). Rows:
  //  * HYLE default: Highway's x86 sequence verbatim (Min(v,MAX)+cvtpd2dq; NaN->MAX, saturating)
  //    -- MUST tie. The old emu128 sign-based-NaN version (2.5x cost for a semantic Highway itself
  //    honors on no hardware target) was dropped; see vec.h decision record.
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
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_da = da.data(); auto* p_o32 = o32.data();
      for (std::size_t i = 0; i + hwd <= kN; i += hwd)
        hs::StoreU(hs::DemoteToNearestInt(hdi32, hs::LoadU(hdd, p_da + i)), hdi32, p_o32 + i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_da = da.data(); auto* p_o32 = o32.data();
      for (std::size_t i = 0; i + gwd <= kN; i += gwd)
        hn::StoreU(hn::DemoteToNearestInt(gdi32, hn::LoadU(gdd, p_da + i)), gdi32, p_o32 + i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "HYLE DemoteToNearestIntInRange (fast)", [&] { auto* p_da = da.data(); auto* p_o32 = o32.data();
      for (std::size_t i = 0; i + hwd <= kN; i += hwd)
        hs::StoreU(hs::DemoteToNearestIntInRange(hdi32, hs::LoadU(hdd, p_da + i)), hdi32, p_o32 + i);
      nb::doNotOptimizeAway(p_o32);
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
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_s32 = s32.data();
      for (std::size_t i = 0; i + hw32 <= kN; i += hw32) {
        auto m = hs::Gt(hs::LoadU(hd32, p_s32 + i), hs::Zero(hd32));
        sink ^= hs::BitsFromMask(hd16, hs::DemoteMaskTo(hd16, hd32, m));
      }
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_s32 = s32.data();
      for (std::size_t i = 0; i + gw32 <= kN; i += gw32) {
        auto m = hn::Gt(hn::LoadU(gd32, p_s32 + i), hn::Zero(gd32));
        sink ^= hn::BitsFromMask(gd16, hn::DemoteMaskTo(gd16, gd32, m));
      }
     hyle_bench::mark(rep); });
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
      HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_aos3 = aos3.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          hs::VFromD<decltype(hd)> v0, v1, v2;
          hs::LoadInterleaved3(hd, p_aos3 + i * 3, v0, v1, v2);
          fsink += hs::GetLane(hs::Add(hs::Add(v0, v1), v2));
        }
       hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_aos3 = aos3.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          hn::VFromD<decltype(gd)> v0, v1, v2;
          hn::LoadInterleaved3(gd, p_aos3 + i * 3, v0, v1, v2);
          fsink += hn::GetLane(hn::Add(hn::Add(v0, v1), v2));
        }
       hyle_bench::mark(rep); });
      hyle_bench::run_third(bench, "store+reload baseline", [&] { auto* p_aos3 = aos3.data();  // the OLD HYLE deinterleave
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          alignas(64) float t0[16], t1[16], t2[16];
          const float* p = p_aos3 + i * 3;
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
      HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_aos4 = aos4.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) hs::StoreInterleaved4(v0, v1, v2, v3, hd, p_aos4 + i * 4);
        nb::doNotOptimizeAway(p_aos4);
       hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_aos4 = aos4.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) hn::StoreInterleaved4(g0, g1, g2, g3, gd, p_aos4 + i * 4);
        nb::doNotOptimizeAway(p_aos4);
       hyle_bench::mark(rep); });
      hyle_bench::run_third(bench, "store+reload baseline", [&] { auto* p_aos4 = aos4.data();  // the OLD HYLE interleave
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          alignas(64) float t0[16], t1[16], t2[16], t3[16];
          hs::StoreU(v0, hd, t0); hs::StoreU(v1, hd, t1); hs::StoreU(v2, hd, t2); hs::StoreU(v3, hd, t3);
          float* p = p_aos4 + i * 4;
          for (std::size_t j = 0; j < hw; ++j) { p[4*j]=t0[j]; p[4*j+1]=t1[j]; p[4*j+2]=t2[j]; p[4*j+3]=t3[j]; }
        }
        nb::doNotOptimizeAway(p_aos4);
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
    HYLE_RUN_PAIR(bench, [&](auto rep) { hyle_bench::stagger(rep); auto* p_aos = aos.data();
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8) {
        hs::VFromD<decltype(hd8)> v0,v1,v2,v3; hs::LoadInterleaved4(hd8, p_aos+i*4, v0,v1,v2,v3);
        bsink ^= hs::GetLane(hs::Add(hs::Add(v0,v1),hs::Add(v2,v3)));
      }
     hyle_bench::mark(rep); }, [&](auto rep) { hyle_bench::stagger(rep); auto* p_aos = aos.data();
      for (std::size_t i = 0; i + gw8 <= kN; i += gw8) {
        hn::VFromD<decltype(gd8)> v0,v1,v2,v3; hn::LoadInterleaved4(gd8, p_aos+i*4, v0,v1,v2,v3);
        bsink ^= hn::GetLane(hn::Add(hn::Add(v0,v1),hn::Add(v2,v3)));
      }
     hyle_bench::mark(rep); });
    hyle_bench::run_third(bench, "store+reload baseline", [&] { auto* p_aos = aos.data();  // the OLD HYLE deinterleave
      for (std::size_t i = 0; i + hw8 <= kN; i += hw8) {
        alignas(64) std::uint8_t t0[64],t1[64],t2[64],t3[64];
        const std::uint8_t* p = p_aos+i*4;
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
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_idx = idx.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto v = hs::LoadU(hdi, p_ia + i);
          auto ind = hs::SetTableIndices(hdi, p_idx + i);
          hs::StoreU(hs::TableLookupLanes(v, ind), hdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_idx = idx.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto v = hn::LoadU(gdi, p_ia + i);
          auto ind = hn::SetTableIndices(gdi, p_idx + i);
          hn::StoreU(hn::TableLookupLanes(v, ind), gdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("TwoTablesLookupLanes i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_table2 = table2.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto a0 = hs::LoadU(hdi, p_ia + i), a1 = hs::LoadU(hdi, p_table2 + i);
          auto ind = hs::SetTableIndices(hdi, p_ib + i);
          hs::StoreU(hs::TwoTablesLookupLanes(hdi, a0, a1, ind), hdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_table2 = table2.data(); auto* p_ib = ib.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto a0 = hn::LoadU(gdi, p_ia + i), a1 = hn::LoadU(gdi, p_table2 + i);
          auto ind = hn::SetTableIndices(gdi, p_ib + i);
          hn::StoreU(hn::TwoTablesLookupLanes(gdi, a0, a1, ind), gdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); });
    }
  }

  // ---- 22. ScatterIndex / CompressStore / MaskedLoad / MaskedLoadOr / BlendedStore / LoadN / StoreN ----
  {
    std::vector<std::int32_t> table(kN, 0), idxbuf(kN);
    { std::mt19937_64 rng(33); std::uniform_int_distribution<int> d(0, kN - 1); for (auto& e : idxbuf) e = d(rng); }
    const auto imask = geni(kN, 34);
    {
      nb::Bench bench; bench.title("ScatterIndex i32 (4096)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_table = table.data(); auto* p_idxbuf = idxbuf.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw)
          hs::ScatterIndex(hs::LoadU(hdi, p_ia + i), hdi, p_table, hs::LoadU(hdi, p_idxbuf + i));
        nb::doNotOptimizeAway(p_table);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_table = table.data(); auto* p_idxbuf = idxbuf.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw)
          hn::ScatterIndex(hn::LoadU(gdi, p_ia + i), gdi, p_table, hn::LoadU(gdi, p_idxbuf + i));
        nb::doNotOptimizeAway(p_table);
       hyle_bench::mark(rep); });
    }
    {
      std::size_t written = 0;
      nb::Bench bench; bench.title("CompressStore i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
        std::size_t pos = 0;
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto v = hs::LoadU(hdi, p_ia + i);
          auto m = hs::Gt(hs::LoadU(hdi, p_imask + i), hs::Set(hdi, 0));
          pos += hs::CompressStore(v, m, hdi, p_iout + pos);
          if (pos + hw > kN) pos = 0;
        }
        written = pos;
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_imask = imask.data(); auto* p_iout = iout.data();
        std::size_t pos = 0;
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto v = hn::LoadU(gdi, p_ia + i);
          auto m = hn::Gt(hn::LoadU(gdi, p_imask + i), hn::Set(gdi, 0));
          pos += hn::CompressStore(v, m, gdi, p_iout + pos);
          if (pos + gw > kN) pos = 0;
        }
        written = pos;
       hyle_bench::mark(rep); });
      nb::doNotOptimizeAway(written);
    }
    {
      nb::Bench bench; bench.title("MaskedLoad i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask + i), hs::Set(hdi, 0));
          hs::StoreU(hs::MaskedLoad(m, hdi, p_ia + i), hdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask + i), hn::Set(gdi, 0));
          hn::StoreU(hn::MaskedLoad(m, gdi, p_ia + i), gdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("MaskedLoadOr i32 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask + i), hs::Set(hdi, 0));
          hs::StoreU(hs::MaskedLoadOr(hs::Set(hdi, -77), m, hdi, p_ia + i), hdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask + i), hn::Set(gdi, 0));
          hn::StoreU(hn::MaskedLoadOr(hn::Set(gdi, -77), m, gdi, p_ia + i), gdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("BlendedStore i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask + i), hs::Set(hdi, 0));
          hs::BlendedStore(hs::LoadU(hdi, p_ia + i), m, hdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask + i), hn::Set(gdi, 0));
          hn::BlendedStore(hn::LoadU(gdi, p_ia + i), m, gdi, p_iout + i);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); });
    }
    {
      const std::size_t k = hw / 2 == 0 ? 1 : hw / 2;
      nb::Bench bench; bench.title("LoadN/StoreN i32 (half lanes)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + hw <= kN; i += hw) {
          auto v = hs::LoadN(hdi, p_ia + i, k);
          hs::StoreN(v, hdi, p_iout + i, k);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data();
        for (std::size_t i = 0; i + gw <= kN; i += gw) {
          auto v = hn::LoadN(gdi, p_ia + i, k);
          hn::StoreN(v, gdi, p_iout + i, k);
        }
        nb::doNotOptimizeAway(p_iout);
       hyle_bench::mark(rep); });
    }
  }

  // ---- 23. Bit ops: all-width MulHigh + fused MulHighLow, Reverse8, BitShuffle,
  // MultiRotateRight. C1 deliberately benchmarks every lane width because x86/NEON use
  // different constructions (native i16, widened i8/i32, four-limb i64).
  auto bench_mul_high = []<typename T>(const char* title, std::uint64_t seed) {
    const hs::ScalableTag<T> hd;
    const hn::ScalableTag<T> gd;
    const std::size_t hw = hs::Lanes(hd), gw = hn::Lanes(gd);
    std::vector<T> a(kN), b(kN), out(kN);
    std::mt19937_64 rng(seed);
    for (auto& e : a) e = static_cast<T>(rng());
    for (auto& e : b) e = static_cast<T>(rng());
    nb::Bench bench;
    bench.title(title).relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(
        bench,
        [&](auto rep) {
          hyle_bench::stagger(rep);
          for (std::size_t i = 0; i + hw <= kN; i += hw)
            hs::StoreU(hs::MulHigh(hs::LoadU(hd, a.data() + i),
                                   hs::LoadU(hd, b.data() + i)),
                       hd, out.data() + i);
          nb::doNotOptimizeAway(out.data());
          hyle_bench::mark(rep);
        },
        [&](auto rep) {
          hyle_bench::stagger(rep);
          for (std::size_t i = 0; i + gw <= kN; i += gw)
            hn::StoreU(hn::MulHigh(hn::LoadU(gd, a.data() + i),
                                   hn::LoadU(gd, b.data() + i)),
                       gd, out.data() + i);
          nb::doNotOptimizeAway(out.data());
          hyle_bench::mark(rep);
        });
  };
#if !defined(HYLE_SIMD_TARGET_SSE4)
  // Highway's SSE4 face only supplies the native i/u16 rows without an Abort fallback.
  bench_mul_high.template operator()<std::int8_t>("MulHigh i8 (4096)", 35);
  bench_mul_high.template operator()<std::uint8_t>("MulHigh u8 (4096)", 36);
#endif
  bench_mul_high.template operator()<std::int16_t>("MulHigh i16 (4096)", 37);
  bench_mul_high.template operator()<std::uint16_t>("MulHigh u16 (4096)", 38);
#if !defined(HYLE_SIMD_TARGET_SSE4)
  bench_mul_high.template operator()<std::int32_t>("MulHigh i32 (4096)", 39);
  bench_mul_high.template operator()<std::uint32_t>("MulHigh u32 (4096)", 40);
  bench_mul_high.template operator()<std::int64_t>("MulHigh i64 (4096)", 41);
  bench_mul_high.template operator()<std::uint64_t>("MulHigh u64 (4096)", 42);
#endif

  // Highway's SSE4 target deliberately maps u64 low multiply to Abort (no baseline vector
  // implementation), so there is no same-contract vendor pair for the fused row there. AVX2,
  // AVX-512 and NEON do have both low and high operations.
#if !defined(HYLE_SIMD_TARGET_SSE4)
  {
    const hs::ScalableTag<std::uint64_t> hd;
    const hn::ScalableTag<std::uint64_t> gd;
    const std::size_t hw = hs::Lanes(hd), gw = hn::Lanes(gd);
    std::vector<std::uint64_t> a(kN), b(kN), lo(kN), hi(kN);
    std::mt19937_64 rng(43);
    for (auto& e : a) e = rng();
    for (auto& e : b) e = rng();
    nb::Bench bench;
    bench.title("MulHighLow u64 (4096)").relative(true).batch(kN).minEpochIterations(4000);
    HYLE_RUN_PAIR(
        bench,
        [&](auto rep) {
          hyle_bench::stagger(rep);
          for (std::size_t i = 0; i + hw <= kN; i += hw) {
            auto high = hs::Zero(hd);
            const auto low = hs::MulHighLow(hs::LoadU(hd, a.data() + i),
                                             hs::LoadU(hd, b.data() + i), high);
            hs::StoreU(low, hd, lo.data() + i);
            hs::StoreU(high, hd, hi.data() + i);
          }
          nb::doNotOptimizeAway(lo.data());
          nb::doNotOptimizeAway(hi.data());
          hyle_bench::mark(rep);
        },
        [&](auto rep) {
          hyle_bench::stagger(rep);
          for (std::size_t i = 0; i + gw <= kN; i += gw) {
            const auto va = hn::LoadU(gd, a.data() + i);
            const auto vb = hn::LoadU(gd, b.data() + i);
            hn::StoreU(hn::Mul(va, vb), gd, lo.data() + i);
            hn::StoreU(hn::MulHigh(va, vb), gd, hi.data() + i);
          }
          nb::doNotOptimizeAway(lo.data());
          nb::doNotOptimizeAway(hi.data());
          hyle_bench::mark(rep);
        });
  }
#endif
  // Reverse8 / BitShuffle / MultiRotateRight are x86-shaped (need >=8 i32 lanes / 512-bit u64
  // path). Highway NEON hard-asserts them (arm_neon-inl.h:6535, i32 NEON = 4 lanes), so guard the
  // whole cluster off NEON -- HYLE's own NEON backend has the correct narrow-vector semantics but
  // there is no Highway NEON counterpart to pair against here.
#if !defined(HYLE_SIMD_TARGET_NEON)
  {
    nb::Bench bench; bench.title("Reverse8 i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::Reverse8(hdi,hs::LoadU(hdi,p_ia+i)),hdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::Reverse8(gdi,hn::LoadU(gdi,p_ia+i)),gdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
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
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u64a = u64a.data(); auto* p_idx8 = idx8.data(); auto* p_u64o = u64o.data();
        for (std::size_t i = 0; i + hw64 <= kN; i += hw64)
          hs::StoreU(hs::BitShuffle(hs::LoadU(hd64, p_u64a+i), hs::LoadU(hd8, p_idx8+i*8)), hd64, p_u64o+i);
        nb::doNotOptimizeAway(p_u64o);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u64a = u64a.data(); auto* p_idx8 = idx8.data(); auto* p_u64o = u64o.data();
        for (std::size_t i = 0; i + gw64 <= kN; i += gw64)
          hn::StoreU(hn::BitShuffle(hn::LoadU(gd64, p_u64a+i), hn::LoadU(gd8, p_idx8+i*8)), gd64, p_u64o+i);
        nb::doNotOptimizeAway(p_u64o);
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("MultiRotateRight u64 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u64a = u64a.data(); auto* p_idx8 = idx8.data(); auto* p_u64o = u64o.data();
        for (std::size_t i = 0; i + hw64 <= kN; i += hw64)
          hs::StoreU(hs::MultiRotateRight(hs::LoadU(hd64, p_u64a+i), hs::LoadU(hd8, p_idx8+i*8)), hd64, p_u64o+i);
        nb::doNotOptimizeAway(p_u64o);
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u64a = u64a.data(); auto* p_idx8 = idx8.data(); auto* p_u64o = u64o.data();
        for (std::size_t i = 0; i + gw64 <= kN; i += gw64)
          hn::StoreU(hn::MultiRotateRight(hn::LoadU(gd64, p_u64a+i), hn::LoadU(gd8, p_idx8+i*8)), gd64, p_u64o+i);
        nb::doNotOptimizeAway(p_u64o);
       hyle_bench::mark(rep); });
    }
    (void)hw8;
  }
#endif  // !HYLE_SIMD_TARGET_NEON

  // ---- 24. Interleave2 + StoreInterleaved3 (domain gap) + PromoteMaskTo ----
  {
    std::vector<float> aos2(kN * 2), aos3(kN * 3);
    for (std::size_t i = 0; i < kN * 2; ++i) aos2[i] = static_cast<float>(i);
    for (std::size_t i = 0; i < kN * 3; ++i) aos3[i] = static_cast<float>(i);
    float fsink = 0;
    {
      nb::Bench bench; bench.title("LoadInterleaved2 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_aos2 = aos2.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw){ hs::VFromD<decltype(hd)> v0,v1; hs::LoadInterleaved2(hd,p_aos2+i*2,v0,v1); fsink+=hs::GetLane(hs::Add(v0,v1)); }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_aos2 = aos2.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw){ hn::VFromD<decltype(gd)> v0,v1; hn::LoadInterleaved2(gd,p_aos2+i*2,v0,v1); fsink+=hn::GetLane(hn::Add(v0,v1)); }
       hyle_bench::mark(rep); });
    }
    {
      const auto v0=hs::LoadU(hd,a.data()), v1=hs::LoadU(hd,b.data()), v2=hs::LoadU(hd,c.data());
      const auto g0=hn::LoadU(gd,a.data()), g1=hn::LoadU(gd,b.data()), g2=hn::LoadU(gd,c.data());
      nb::Bench bench; bench.title("StoreInterleaved2 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_aos2 = aos2.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreInterleaved2(v0,v1,hd,p_aos2+i*2); nb::doNotOptimizeAway(p_aos2);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_aos2 = aos2.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreInterleaved2(g0,g1,gd,p_aos2+i*2); nb::doNotOptimizeAway(p_aos2);  hyle_bench::mark(rep); });
    }
    {
      const auto v0=hs::LoadU(hd,a.data()), v1=hs::LoadU(hd,b.data()), v2=hs::LoadU(hd,c.data());
      const auto g0=hn::LoadU(gd,a.data()), g1=hn::LoadU(gd,b.data()), g2=hn::LoadU(gd,c.data());
      nb::Bench bench; bench.title("StoreInterleaved3 f32 (4096)").relative(true).batch(kN).minEpochIterations(2000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_aos3 = aos3.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreInterleaved3(v0,v1,v2,hd,p_aos3+i*3); nb::doNotOptimizeAway(p_aos3);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_aos3 = aos3.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreInterleaved3(g0,g1,g2,gd,p_aos3+i*3); nb::doNotOptimizeAway(p_aos3);  hyle_bench::mark(rep); });
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
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_s16 = s16.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hd16r, p_s16+i), hs::Set(hd16r, std::int16_t{0}));
          sink ^= hs::BitsFromMask(hdi, hs::PromoteMaskTo(hdi, hd16r, m));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_s16 = s16.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gd16r, p_s16+i), hn::Set(gd16r, std::int16_t{0}));
          sink ^= hn::BitsFromMask(gdi, hn::PromoteMaskTo(gdi, gd16r, m));
        }
       hyle_bench::mark(rep); });
    }
    // OrderedDemote2MasksTo i32+i32 -> i16 full
    {
      using HD16 = hs::RepartitionToNarrow<hs::ScalableTag<std::int32_t>>;
      using GD16 = hn::RepartitionToNarrow<hn::ScalableTag<std::int32_t>>;
      // Actually OrderedDemote2MasksTo wants DTo with 2N lanes of half width = full narrow ScalableTag i16
      const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
      nb::Bench bench; bench.title("OrderedDemote2MasksTo i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+2*hw<=kN;i+=2*hw) {
          auto ma = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          auto mb = hs::Gt(hs::LoadU(hdi, p_imask+i+hw), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hd16, hs::OrderedDemote2MasksTo(hd16, hdi, ma, mb));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+2*gw<=kN;i+=2*gw) {
          auto ma = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          auto mb = hn::Gt(hn::LoadU(gdi, p_imask+i+gw), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gd16, hn::OrderedDemote2MasksTo(gd16, gdi, ma, mb));
        }
       hyle_bench::mark(rep); });
    }
    // CombineMasks / LowerHalfOfMask / UpperHalfOfMask
    {
      const hs::Half<hs::ScalableTag<std::int32_t>> hdh; const hn::Half<hn::ScalableTag<std::int32_t>> gdh;
      const std::size_t hwh = hs::Lanes(hdh), gwh = hn::Lanes(gdh);
      nb::Bench bench; bench.title("CombineMasks i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto full = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          auto lo = hs::LowerHalfOfMask(hdh, full);
          auto hi = hs::UpperHalfOfMask(hdh, full);
          sink ^= hs::BitsFromMask(hdi, hs::CombineMasks(hdi, hi, lo));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto full = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          auto lo = hn::LowerHalfOfMask(gdh, full);
          auto hi = hn::UpperHalfOfMask(gdh, full);
          sink ^= hn::BitsFromMask(gdi, hn::CombineMasks(gdi, hi, lo));
        }
       hyle_bench::mark(rep); });
      (void)hwh; (void)gwh;
    }
    // SlideMask1Up/Down + SlideMaskUp/DownLanes
    {
      nb::Bench bench; bench.title("SlideMask1Up i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMask1Up(hdi, m));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMask1Up(gdi, m));
        }
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("SlideMask1Down i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMask1Down(hdi, m));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMask1Down(gdi, m));
        }
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("SlideMaskUpLanes i32 amt=2 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMaskUpLanes(hdi, m, 2));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMaskUpLanes(gdi, m, 2));
        }
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("SlideMaskDownLanes i32 amt=2 (4096)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hdi, hs::SlideMaskDownLanes(hdi, m, 2));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gdi, hn::SlideMaskDownLanes(gdi, m, 2));
        }
       hyle_bench::mark(rep); });
    }
    // RebindMask i32 -> f32, SetMask, Dup128MaskFromMaskBits
    {
      nb::Bench bench; bench.title("RebindMask i32->f32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto m = hs::Gt(hs::LoadU(hdi, p_imask+i), hs::Zero(hdi));
          sink ^= hs::BitsFromMask(hd, hs::RebindMask(hd, m));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto m = hn::Gt(hn::LoadU(gdi, p_imask+i), hn::Zero(gdi));
          sink ^= hn::BitsFromMask(gd, hn::RebindMask(gd, m));
        }
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("SetMask i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep);
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink ^= hs::BitsFromMask(hdi, hs::SetMask(hdi, (imask[i] & 1) != 0));
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep);
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink ^= hn::BitsFromMask(gdi, hn::SetMask(gdi, (imask[i] & 1) != 0));
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("Dup128MaskFromMaskBits i32 (4096)").relative(true).batch(kN).minEpochIterations(10000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep);
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink ^= hs::BitsFromMask(hdi, hs::Dup128MaskFromMaskBits(hdi, static_cast<unsigned>(imask[i] & 0xF)));
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep);
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink ^= hn::BitsFromMask(gdi, hn::Dup128MaskFromMaskBits(gdi, static_cast<unsigned>(imask[i] & 0xF)));
       hyle_bench::mark(rep); });
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
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_s16 = s16.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw) {
          auto half = hs::LowerHalf(hs::LoadU(hd16full, p_s16+i));
          sink += hs::GetLane(hs::PromoteTo(hdi, half));
        }
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_s16 = s16.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw) {
          auto half = hn::LowerHalf(hn::Half<decltype(gd16full)>(), hn::LoadU(gd16full, p_s16+i));
          sink += hn::GetLane(hn::PromoteTo(gdi, half));
        }
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("DemoteTo i32->i16 1-arg (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink += hs::GetLane(hs::DemoteTo(hd16r, hs::LoadU(hdi, p_ia+i)));
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink += hn::GetLane(hn::DemoteTo(gd16r, hn::LoadU(gdi, p_ia+i)));
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("ShiftRightAndDemoteTo<3> i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink += hs::GetLane(hs::ShiftRightAndDemoteTo<3>(hd16r, hs::LoadU(hdi, p_ia+i)));
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink += hn::GetLane(hn::ShiftRightAndDemoteTo<3>(gd16r, hn::LoadU(gdi, p_ia+i)));
       hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("RoundingShiftRightAndDemoteTo<3> i32->i16 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data();
        for (std::size_t i=0;i+hw<=kN;i+=hw)
          sink += hs::GetLane(hs::RoundingShiftRightAndDemoteTo<3>(hd16r, hs::LoadU(hdi, p_ia+i)));
       hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data();
        for (std::size_t i=0;i+gw<=kN;i+=gw)
          sink += hn::GetLane(hn::RoundingShiftRightAndDemoteTo<3>(gd16r, hn::LoadU(gdi, p_ia+i)));
       hyle_bench::mark(rep); });
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
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_off = off.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::GatherOffset(hdi,p_table,hs::LoadU(hdi,p_off+i)),hdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_off = off.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::GatherOffset(gdi,p_table,hn::LoadU(gdi,p_off+i)),gdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("ScatterOffset i32 (4096)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_sink_tbl = sink_tbl.data(); auto* p_off = off.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::ScatterOffset(hs::LoadU(hdi,p_ia+i),hdi,p_sink_tbl,hs::LoadU(hdi,p_off+i)); nb::doNotOptimizeAway(p_sink_tbl);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_sink_tbl = sink_tbl.data(); auto* p_off = off.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::ScatterOffset(hn::LoadU(gdi,p_ia+i),gdi,p_sink_tbl,hn::LoadU(gdi,p_off+i)); nb::doNotOptimizeAway(p_sink_tbl);  hyle_bench::mark(rep); });
    }
  }

  // ---- 28. CompressBits / CompressNot / CompressBlendedStore / LoadExpand ----
  {
    const auto imask = geni(kN, 72);
    // CompressBits: mask packed as bytes (1 bit/lane). Build per-block byte-mask for both.
    {
      nb::Bench bench; bench.title("CompressNot i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,p_imask+i),hs::Set(hdi,0)); hs::StoreU(hs::CompressNot(hs::LoadU(hdi,p_ia+i),m),hdi,p_iout+i);} nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,p_imask+i),hn::Set(gdi,0)); hn::StoreU(hn::CompressNot(hn::LoadU(gdi,p_ia+i),m),gdi,p_iout+i);} nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
    }
    {
      std::vector<std::uint64_t> u64a(kN), u64mask(kN), u64out(kN);
      for (std::size_t i = 0; i < kN; ++i) {
        u64a[i] = static_cast<std::uint64_t>(i * 0x9E3779B1u + 17u);
        u64mask[i] = static_cast<std::uint64_t>(i & 1u);
      }
      const hs::ScalableTag<std::uint64_t> hdu64;
      const hn::ScalableTag<std::uint64_t> gdu64;
      const std::size_t hw64 = hs::Lanes(hdu64);
      const std::size_t gw64 = hn::Lanes(gdu64);
      nb::Bench bench; bench.title("CompressNot u64 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_mask = u64mask.data(); auto* p_a = u64a.data(); auto* p_out = u64out.data(); for(std::size_t i=0;i+hw64<=kN;i+=hw64){ auto m=hs::Gt(hs::LoadU(hdu64,p_mask+i),hs::Set(hdu64,0)); hs::StoreU(hs::CompressNot(hs::LoadU(hdu64,p_a+i),m),hdu64,p_out+i);} nb::doNotOptimizeAway(p_out); hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_mask = u64mask.data(); auto* p_a = u64a.data(); auto* p_out = u64out.data(); for(std::size_t i=0;i+gw64<=kN;i+=gw64){ auto m=hn::Gt(hn::LoadU(gdu64,p_mask+i),hn::Set(gdu64,0)); hn::StoreU(hn::CompressNot(hn::LoadU(gdu64,p_a+i),m),gdu64,p_out+i);} nb::doNotOptimizeAway(p_out); hyle_bench::mark(rep); });
    }
    {
      std::size_t w=0;
      nb::Bench bench; bench.title("CompressBlendedStore i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data(); std::size_t pos=0; for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,p_imask+i),hs::Set(hdi,0)); pos+=hs::CompressBlendedStore(hs::LoadU(hdi,p_ia+i),m,hdi,p_iout+(pos+hw<=kN?pos:0));} w=pos;  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data(); std::size_t pos=0; for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,p_imask+i),hn::Set(gdi,0)); pos+=hn::CompressBlendedStore(hn::LoadU(gdi,p_ia+i),m,gdi,p_iout+(pos+gw<=kN?pos:0));} w=pos;  hyle_bench::mark(rep); });
      nb::doNotOptimizeAway(w);
    }
    {
      nb::Bench bench; bench.title("LoadExpand i32 (4096, 50%)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,p_imask+i),hs::Set(hdi,0)); hs::StoreU(hs::LoadExpand(m,hdi,p_ia+i),hdi,p_iout+i);} nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_imask = imask.data(); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,p_imask+i),hn::Set(gdi,0)); hn::StoreU(hn::LoadExpand(m,gdi,p_ia+i),gdi,p_iout+i);} nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
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
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_idxb = idxb.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::GatherIndexN(hdi,p_table,hs::LoadU(hdi,p_idxb+i),kNlan),hdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_idxb = idxb.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::GatherIndexN(gdi,p_table,hn::LoadU(gdi,p_idxb+i),kNlan),gdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("GatherIndexNOr i32 (half n)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_idxb = idxb.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::GatherIndexNOr(hs::Set(hdi,-1),hdi,p_table,hs::LoadU(hdi,p_idxb+i),kNlan),hdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_table = table.data(); auto* p_idxb = idxb.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::GatherIndexNOr(hn::Set(gdi,-1),gdi,p_table,hn::LoadU(gdi,p_idxb+i),kNlan),gdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("ScatterIndexN i32 (half n)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_stbl = stbl.data(); auto* p_idxb = idxb.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::ScatterIndexN(hs::LoadU(hdi,p_ia+i),hdi,p_stbl,hs::LoadU(hdi,p_idxb+i),kNlan); nb::doNotOptimizeAway(p_stbl);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_stbl = stbl.data(); auto* p_idxb = idxb.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::ScatterIndexN(hn::LoadU(gdi,p_ia+i),gdi,p_stbl,hn::LoadU(gdi,p_idxb+i),kNlan); nb::doNotOptimizeAway(p_stbl);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("ScatterIndexN i32 (one lane)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_stbl = stbl.data(); auto* p_idxb = idxb.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::ScatterIndexN(hs::LoadU(hdi,p_ia+i),hdi,p_stbl,hs::LoadU(hdi,p_idxb+i),1); nb::doNotOptimizeAway(p_stbl);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_stbl = stbl.data(); auto* p_idxb = idxb.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::ScatterIndexN(hn::LoadU(gdi,p_ia+i),gdi,p_stbl,hn::LoadU(gdi,p_idxb+i),1); nb::doNotOptimizeAway(p_stbl);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("ScatterIndexN i32 (full n)").relative(true).batch(kN).minEpochIterations(3000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_stbl = stbl.data(); auto* p_idxb = idxb.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::ScatterIndexN(hs::LoadU(hdi,p_ia+i),hdi,p_stbl,hs::LoadU(hdi,p_idxb+i),hw); nb::doNotOptimizeAway(p_stbl);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_stbl = stbl.data(); auto* p_idxb = idxb.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::ScatterIndexN(hn::LoadU(gdi,p_ia+i),gdi,p_stbl,hn::LoadU(gdi,p_idxb+i),gw); nb::doNotOptimizeAway(p_stbl);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("LoadNOr i32 (half n)").relative(true).batch(kN).minEpochIterations(5000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::LoadNOr(hs::Set(hdi,-1),hdi,p_ia+i,kNlan),hdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia = ia.data(); auto* p_iout = iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::LoadNOr(hn::Set(gdi,-1),gdi,p_ia+i,kNlan),gdi,p_iout+i); nb::doNotOptimizeAway(p_iout);  hyle_bench::mark(rep); });
    }
  }

  // ---- 30. Previously-unmeasured native mask/bitfield seams. Keep these separate from synthetic
  // helpers so a slow mask transition cannot hide inside a larger consumer. ----
  {
    std::uint64_t sink = 0;
    std::vector<std::uint8_t> mask_bytes(kN);
    for (std::size_t i = 0; i < kN; ++i) mask_bytes[i] = static_cast<std::uint8_t>(i * 37u);
    {
      nb::Bench bench; bench.title("BitsFromMask i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) sink^=hs::BitsFromMask(hdi,hs::Gt(hs::LoadU(hdi,p_ia+i),hs::Zero(hdi))); nb::doNotOptimizeAway(sink); hyle_bench::mark(rep); },
                           [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) sink^=hn::BitsFromMask(gdi,hn::Gt(hn::LoadU(gdi,p_ia+i),hn::Zero(gdi))); nb::doNotOptimizeAway(sink); hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("MaskFromVec i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) sink^=hs::BitsFromMask(hdi,hs::MaskFromVec(hs::LoadU(hdi,p_ia+i))); nb::doNotOptimizeAway(sink); hyle_bench::mark(rep); },
                           [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) sink^=hn::BitsFromMask(gdi,hn::MaskFromVec(hn::LoadU(gdi,p_ia+i))); nb::doNotOptimizeAway(sink); hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("VecFromMask i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,p_ia+i),hs::Zero(hdi)); hs::StoreU(hs::VecFromMask(hdi,m),hdi,p_o+i);} nb::doNotOptimizeAway(p_o); hyle_bench::mark(rep); },
                           [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,p_ia+i),hn::Zero(gdi)); hn::StoreU(hn::VecFromMask(gdi,m),gdi,p_o+i);} nb::doNotOptimizeAway(p_o); hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("LoadMaskBits i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_bits=mask_bytes.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::VecFromMask(hdi,hs::LoadMaskBits(hdi,static_cast<std::uint64_t>(p_bits[i]))),hdi,p_o+i); nb::doNotOptimizeAway(p_o); hyle_bench::mark(rep); },
                           [&](auto rep){ hyle_bench::stagger(rep); auto* p_bits=mask_bytes.data(); auto* p_o=iout.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::VecFromMask(gdi,hn::LoadMaskBits(gdi,p_bits+i)),gdi,p_o+i); nb::doNotOptimizeAway(p_o); hyle_bench::mark(rep); });
    }
    {
      nb::Bench bench; bench.title("StoreMaskBits i32 (4096)").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); auto* p_bits=mask_bytes.data(); for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::StoreMaskBits(hdi,hs::Gt(hs::LoadU(hdi,p_ia+i),hs::Zero(hdi)),p_bits+i); nb::doNotOptimizeAway(p_bits); nb::doNotOptimizeAway(sink); hyle_bench::mark(rep); },
                           [&](auto rep){ hyle_bench::stagger(rep); auto* p_ia=ia.data(); auto* p_bits=mask_bytes.data(); for(std::size_t i=0;i+gw<=kN;i+=gw) sink+=hn::StoreMaskBits(gdi,hn::Gt(hn::LoadU(gdi,p_ia+i),hn::Zero(gdi)),p_bits+i); nb::doNotOptimizeAway(p_bits); nb::doNotOptimizeAway(sink); hyle_bench::mark(rep); });
    }
  }

  // Raw high-bit-zeroing byte shuffle. This is distinct from the stronger HYLE default
  // TableLookupBytes contract already measured in section 10.
  {
    std::vector<std::uint8_t> bytes(kN), indices(kN), shuffled(kN);
    for(std::size_t i=0;i<kN;++i){ bytes[i]=static_cast<std::uint8_t>(i); indices[i]=(i&1)?static_cast<std::uint8_t>(0x80):static_cast<std::uint8_t>(i&15); }
    const hs::ScalableTag<std::uint8_t> hdu8; const hn::ScalableTag<std::uint8_t> gdu8;
    const std::size_t hwu8=hs::Lanes(hdu8), gwu8=hn::Lanes(gdu8);
    nb::Bench bench; bench.title("TableLookupBytesOr0 u8 (4096)").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_b=bytes.data(); auto* p_i=indices.data(); auto* p_o=shuffled.data(); for(std::size_t i=0;i+hwu8<=kN;i+=hwu8) hs::StoreU(hs::TableLookupBytesOr0(hs::LoadU(hdu8,p_b+i),hs::LoadU(hdu8,p_i+i)),hdu8,p_o+i); nb::doNotOptimizeAway(p_o); hyle_bench::mark(rep); },
                         [&](auto rep){ hyle_bench::stagger(rep); auto* p_b=bytes.data(); auto* p_i=indices.data(); auto* p_o=shuffled.data(); for(std::size_t i=0;i+gwu8<=kN;i+=gwu8) hn::StoreU(hn::TableLookupBytesOr0(hn::LoadU(gdu8,p_b+i),hn::LoadU(gdu8,p_i+i)),gdu8,p_o+i); nb::doNotOptimizeAway(p_o); hyle_bench::mark(rep); });
  }

  // ---- crypto AES/CLMul: moved here from crypto_bench (which used uncalibrated nb::run and let
  // Highway's loop hoist -> a fake ~10x; disasm proved HYLE's op-graph is identical, 2x vaesenc).
  // Calibrated load->op->store per lane-group (distinct memory + store sink defeats CSE/hoist). ----
  {
    std::vector<std::uint8_t> cs(kN), ck(kN), cout(kN);
    { std::mt19937_64 r(51); std::uniform_int_distribution<int> d(0,255); for(std::size_t i=0;i<kN;++i){ cs[i]=(std::uint8_t)d(r); ck[i]=(std::uint8_t)d(r);} }
    const hs::ScalableTag<std::uint8_t> hc8; const hn::ScalableTag<std::uint8_t> gc8;
    const std::size_t hcw=hs::Lanes(hc8), gcw=hn::Lanes(gc8);
    const auto aes_binary = [&](const char* title, auto hop, auto gop){
      nb::Bench b; b.title(title).relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* ps=cs.data(); auto* pk=ck.data(); auto* po=cout.data(); for(std::size_t i=0;i+hcw<=kN;i+=hcw) hs::StoreU(hop(hs::LoadU(hc8,ps+i),hs::LoadU(hc8,pk+i)),hc8,po+i); nb::doNotOptimizeAway(po);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* ps=cs.data(); auto* pk=ck.data(); auto* po=cout.data(); for(std::size_t i=0;i+gcw<=kN;i+=gcw) hn::StoreU(gop(hn::LoadU(gc8,ps+i),hn::LoadU(gc8,pk+i)),gc8,po+i); nb::doNotOptimizeAway(po);  hyle_bench::mark(rep); });
    };
    const auto aes_unary = [&](const char* title, auto hop, auto gop){
      nb::Bench b; b.title(title).relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* ps=cs.data(); auto* po=cout.data(); for(std::size_t i=0;i+hcw<=kN;i+=hcw) hs::StoreU(hop(hs::LoadU(hc8,ps+i)),hc8,po+i); nb::doNotOptimizeAway(po);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* ps=cs.data(); auto* po=cout.data(); for(std::size_t i=0;i+gcw<=kN;i+=gcw) hn::StoreU(gop(hn::LoadU(gc8,ps+i)),gc8,po+i); nb::doNotOptimizeAway(po);  hyle_bench::mark(rep); });
    };
    aes_binary("AESRound u8", [&](auto s,auto k){return hs::AESRound(s,k);}, [&](auto s,auto k){return hn::AESRound(s,k);});
    aes_binary("AESLastRound u8", [&](auto s,auto k){return hs::AESLastRound(s,k);}, [&](auto s,auto k){return hn::AESLastRound(s,k);});
    aes_binary("AESRoundInv u8", [&](auto s,auto k){return hs::AESRoundInv(s,k);}, [&](auto s,auto k){return hn::AESRoundInv(s,k);});
    aes_binary("AESLastRoundInv u8", [&](auto s,auto k){return hs::AESLastRoundInv(s,k);}, [&](auto s,auto k){return hn::AESLastRoundInv(s,k);});
    aes_unary("AESInvMixColumns u8", [&](auto s){return hs::AESInvMixColumns(s);}, [&](auto s){return hn::AESInvMixColumns(s);});
    aes_unary("AESKeyGenAssist u8", [&](auto s){return hs::AESKeyGenAssist<0x01>(s);}, [&](auto s){return hn::AESKeyGenAssist<0x01>(s);});
    // CLMul: u64 lanes
    std::vector<std::uint64_t> ca(kN), cb(kN), cou(kN);
    { std::mt19937_64 r(52); for(std::size_t i=0;i<kN;++i){ ca[i]=r(); cb[i]=r(); } }
    const hs::ScalableTag<std::uint64_t> hc64; const hn::ScalableTag<std::uint64_t> gc64;
    const std::size_t hcw64=hs::Lanes(hc64), gcw64=hn::Lanes(gc64);
    const auto clmul=[&](const char* title, auto hop, auto gop){
      nb::Bench b; b.title(title).relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* pa=ca.data(); auto* pb=cb.data(); auto* po=cou.data(); for(std::size_t i=0;i+hcw64<=kN;i+=hcw64) hs::StoreU(hop(hs::LoadU(hc64,pa+i),hs::LoadU(hc64,pb+i)),hc64,po+i); nb::doNotOptimizeAway(po);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* pa=ca.data(); auto* pb=cb.data(); auto* po=cou.data(); for(std::size_t i=0;i+gcw64<=kN;i+=gcw64) hn::StoreU(gop(hn::LoadU(gc64,pa+i),hn::LoadU(gc64,pb+i)),gc64,po+i); nb::doNotOptimizeAway(po);  hyle_bench::mark(rep); });
    };
    clmul("CLMulLower u64", [&](auto a,auto b){return hs::CLMulLower(a,b);}, [&](auto a,auto b){return hn::CLMulLower(a,b);});
    clmul("CLMulUpper u64", [&](auto a,auto b){return hs::CLMulUpper(a,b);}, [&](auto a,auto b){return hn::CLMulUpper(a,b);});
  }

  return 0;
}
