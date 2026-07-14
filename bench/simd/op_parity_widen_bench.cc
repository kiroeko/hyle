// op_parity_widen -- the bespoke-harness sweep: HYLE self-made width-CHANGING ops (promote/demote/
// truncate/convert/SumsOf/ReorderDemote2To/WidenMulPairwiseAdd) vs Highway. These change lane
// count/width so they need explicit dst tags, not the uniform loop of op_parity_full. They are the
// prime store+reload suspects. Same TU / same ISA / both inline.
// Measurement: op_parity_engine.h (pinned core, replicated kernels); read via
// scripts/op-parity-report.py only.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include "hyle/simd/vec.h"
#include "hwy/highway.h"
#include "op_parity_engine.h"
namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;
namespace nb = ankerl::nanobench;

namespace {
constexpr std::size_t kN = 4096;
template <class T> std::vector<T> gen(std::uint64_t s, T lo, T hi){ std::mt19937_64 r(s); std::uniform_int_distribution<long long> d(lo,hi); std::vector<T> v(kN); for(auto&e:v)e=(T)d(r); return v; }
}

int main(){
  hyle_bench::pin_and_boost();
  auto i16a = gen<std::int16_t>(1, -30000, 30000);
  auto i16b = gen<std::int16_t>(2, -30000, 30000);
  auto i32a = gen<std::int32_t>(3, -1000000, 1000000);
  auto i32b = gen<std::int32_t>(4, -1000000, 1000000);
  auto u8a  = gen<std::uint8_t>(5, 0, 255);
  std::vector<std::int32_t> o32(kN); std::vector<std::int16_t> o16(kN);
  std::vector<std::uint8_t> o8(kN); std::vector<std::uint64_t> o64(kN);

  // tags
  const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
  const hs::ScalableTag<std::int32_t> hd32; const hn::ScalableTag<std::int32_t> gd32;
  const hs::ScalableTag<std::uint8_t> hd8;  const hn::ScalableTag<std::uint8_t> gd8;
  const hs::ScalableTag<std::uint32_t> hd32u; const hn::ScalableTag<std::uint32_t> gd32u;
  const hs::ScalableTag<std::uint64_t> hd64; const hn::ScalableTag<std::uint64_t> gd64;
  const std::size_t w16h=hs::Lanes(hd16), w16g=hn::Lanes(gd16);
  const std::size_t w32h=hs::Lanes(hd32), w32g=hn::Lanes(gd32);
  const std::size_t w8h=hs::Lanes(hd8), w8g=hn::Lanes(gd8);

  // PromoteUpperTo i16->i32 (HYLE half-at-a-time dst = ScalableTag<i32>; Highway = Rebind<i32> on Half)
  {
    nb::Bench b; b.title("PromoteUpperTo i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_o32 = o32.data(); for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::PromoteUpperTo(hd32, hs::LoadU(hd16,p_i16a+i)), hd32, p_o32+i); nb::doNotOptimizeAway(p_o32);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_o32 = o32.data(); const hn::Rebind<std::int32_t, hn::Half<decltype(gd16)>> gdn; for(std::size_t i=0;i+w16g<=kN;i+=w16g){ auto up = hn::UpperHalf(hn::Half<decltype(gd16)>(), hn::LoadU(gd16,p_i16a+i)); hn::StoreU(hn::PromoteTo(gdn, up), gdn, p_o32+i);} nb::doNotOptimizeAway(p_o32);  hyle_bench::mark(rep); });
  }
  // DemoteTo i32->i16 (HYLE two wide -> one narrow; Highway ReorderDemote2To or DemoteTo per half)
  {
    nb::Bench b; b.title("DemoteTo i32->i16 (sat)").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::DemoteTo(hd16, hs::LoadU(hd32,p_i32a+i), hs::LoadU(hd32,p_i32a+i+w32h)), hd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); const hn::Rebind<std::int16_t, decltype(gd32)> gdn; for(std::size_t i=0;i+w32g<=kN;i+=w32g){ auto d = hn::DemoteTo(gdn, hn::LoadU(gd32,p_i32a+i)); hn::StoreU(d, gdn, p_o16+i);} nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  // ReorderDemote2To i32->i16
  {
    nb::Bench b; b.title("ReorderDemote2To i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderDemote2To(hd16, hs::LoadU(hd32,p_i32a+i), hs::LoadU(hd32,p_i32a+i+w32h)), hd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderDemote2To(gd16, hn::LoadU(gd32,p_i32a+i), hn::LoadU(gd32,p_i32a+i+w32g)), gd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  // WidenMulPairwiseAdd i16->i32
  {
    nb::Bench b; b.title("WidenMulPairwiseAdd i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data(); for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::WidenMulPairwiseAdd(hd32, hs::LoadU(hd16,p_i16a+i), hs::LoadU(hd16,p_i16b+i)), hd32, p_o32+i); nb::doNotOptimizeAway(p_o32);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data(); for(std::size_t i=0;i+w16g<=kN;i+=w16g) hn::StoreU(hn::WidenMulPairwiseAdd(gd32, hn::LoadU(gd16,p_i16a+i), hn::LoadU(gd16,p_i16b+i)), gd32, p_o32+i); nb::doNotOptimizeAway(p_o32);  hyle_bench::mark(rep); });
  }
  // MulFixedPoint15 i16
  {
    nb::Bench b; b.title("MulFixedPoint15 i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::MulFixedPoint15(hs::LoadU(hd16,p_i16a+i), hs::LoadU(hd16,p_i16b+i)), hd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+w16g<=kN;i+=w16g) hn::StoreU(hn::MulFixedPoint15(hn::LoadU(gd16,p_i16a+i), hn::LoadU(gd16,p_i16b+i)), gd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  // SumsOf8 u8->u64
  {
    nb::Bench b; b.title("SumsOf8 u8->u64").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_o64 = o64.data(); for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf8(hs::LoadU(hd8,p_u8a+i)), hd64, p_o64+i/8); nb::doNotOptimizeAway(p_o64);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_o64 = o64.data(); for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf8(hn::LoadU(gd8,p_u8a+i)), gd64, p_o64+i/8); nb::doNotOptimizeAway(p_o64);  hyle_bench::mark(rep); });
  }
  // GetExponent f32 (HYLE store+reload+std::logb -- prime suspect)
  {
    auto fa = [](){ std::mt19937_64 r(9); std::uniform_real_distribution<float> d(0.5f,1e6f); std::vector<float> v(kN); for(auto&e:v)e=d(r); return v; }();
    std::vector<float> fo(kN);
    const hs::ScalableTag<float> hdf; const hn::ScalableTag<float> gdf;
    const std::size_t wfh=hs::Lanes(hdf), wfg=hn::Lanes(gdf);
    nb::Bench b; b.title("GetExponent f32").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_fa = fa.data(); auto* p_fo = fo.data(); for(std::size_t i=0;i+wfh<=kN;i+=wfh) hs::StoreU(hs::GetExponent(hs::LoadU(hdf,p_fa+i)), hdf, p_fo+i); nb::doNotOptimizeAway(p_fo);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_fa = fa.data(); auto* p_fo = fo.data(); for(std::size_t i=0;i+wfg<=kN;i+=wfg) hn::StoreU(hn::GetExponent(hn::LoadU(gdf,p_fa+i)), gdf, p_fo+i); nb::doNotOptimizeAway(p_fo);  hyle_bench::mark(rep); });
  }

  // ---- Tier1: widen MAC family ----
  // ReorderWidenMulAccumulate i16->i32
  {
    nb::Bench b; b.title("ReorderWidenMulAccumulate i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto sum1 = hs::Zero(hd32);
        auto sum0 = hs::ReorderWidenMulAccumulate(hd32, hs::LoadU(hd16,p_i16a+i), hs::LoadU(hd16,p_i16b+i), hs::Zero(hd32), sum1);
        hs::StoreU(hs::RearrangeToOddPlusEven(sum0, sum1), hd32, p_o32+i);
      }
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16g<=kN;i+=w16g) {
        auto sum1 = hn::Zero(gd32);
        auto sum0 = hn::ReorderWidenMulAccumulate(gd32, hn::LoadU(gd16,p_i16a+i), hn::LoadU(gd16,p_i16b+i), hn::Zero(gd32), sum1);
        hn::StoreU(hn::RearrangeToOddPlusEven(sum0, sum1), gd32, p_o32+i);
      }
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
  }
  // SatWidenMulPairwiseAdd u8*i8 -> i16
  {
    auto i8b = gen<std::int8_t>(11, -100, 100);
    std::vector<std::int16_t> o16s(kN);
    const hs::ScalableTag<std::int8_t> hi8; const hn::ScalableTag<std::int8_t> gi8;
    nb::Bench b; b.title("SatWidenMulPairwiseAdd u8*i8->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_i8b = i8b.data(); auto* p_o16s = o16s.data();
      for (std::size_t i=0;i+w8h<=kN;i+=w8h)
        hs::StoreU(hs::SatWidenMulPairwiseAdd(hd16, hs::LoadU(hd8,p_u8a+i), hs::LoadU(hi8,p_i8b+i)), hd16, p_o16s+i);
      nb::doNotOptimizeAway(p_o16s);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_i8b = i8b.data(); auto* p_o16s = o16s.data();
      for (std::size_t i=0;i+w8g<=kN;i+=w8g)
        hn::StoreU(hn::SatWidenMulPairwiseAdd(gd16, hn::LoadU(gd8,p_u8a+i), hn::LoadU(gi8,p_i8b+i)), gd16, p_o16s+i);
      nb::doNotOptimizeAway(p_o16s);
     hyle_bench::mark(rep); });
  }
  // SumOfMulQuadAccumulate i8->i32
  {
    auto i8a = gen<std::int8_t>(12, -50, 50);
    auto i8b = gen<std::int8_t>(13, -50, 50);
    const hs::ScalableTag<std::int8_t> hi8; const hn::ScalableTag<std::int8_t> gi8;
    const std::size_t w8ih=hs::Lanes(hi8), w8ig=hn::Lanes(gi8);
    nb::Bench b; b.title("SumOfMulQuadAccumulate i8->i32").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i8a = i8a.data(); auto* p_i8b = i8b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w8ih<=kN;i+=w8ih)
        hs::StoreU(hs::SumOfMulQuadAccumulate(hd32, hs::LoadU(hi8,p_i8a+i), hs::LoadU(hi8,p_i8b+i), hs::Zero(hd32)), hd32, p_o32+i/4);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i8a = i8a.data(); auto* p_i8b = i8b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w8ig<=kN;i+=w8ig)
        hn::StoreU(hn::SumOfMulQuadAccumulate(gd32, hn::LoadU(gi8,p_i8a+i), hn::LoadU(gi8,p_i8b+i), hn::Zero(gd32)), gd32, p_o32+i/4);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
  }
  // WidenMulAccumulate i16->i32 (half-at-a-time dual acc)
  {
    nb::Bench b; b.title("WidenMulAccumulate i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto high = hs::Zero(hd32);
        auto low = hs::WidenMulAccumulate(hd32, hs::LoadU(hd16,p_i16a+i), hs::LoadU(hd16,p_i16b+i), hs::Zero(hd32), high);
        hs::StoreU(hs::Add(low, high), hd32, p_o32+i);
      }
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16g<=kN;i+=w16g) {
        auto high = hn::Zero(gd32);
        auto low = hn::WidenMulAccumulate(gd32, hn::LoadU(gd16,p_i16a+i), hn::LoadU(gd16,p_i16b+i), hn::Zero(gd32), high);
        hn::StoreU(hn::Add(low, high), gd32, p_o32+i);
      }
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
  }
  // SatWidenMulPairwiseAccumulate i16->i32
  {
    nb::Bench b; b.title("SatWidenMulPairwiseAccumulate i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16h<=kN;i+=w16h)
        hs::StoreU(hs::SatWidenMulPairwiseAccumulate(hd32, hs::LoadU(hd16,p_i16a+i), hs::LoadU(hd16,p_i16b+i), hs::Zero(hd32)), hd32, p_o32+i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16g<=kN;i+=w16g)
        hn::StoreU(hn::SatWidenMulPairwiseAccumulate(gd32, hn::LoadU(gd16,p_i16a+i), hn::LoadU(gd16,p_i16b+i), hn::Zero(gd32)), gd32, p_o32+i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
  }
  // SatWidenMulAccumFixedPoint: Highway takes Rebind<i16,di32> (same lane count as i32 = half-register).
  // HYLE accepts full V16; ZeroExtendVector(LowerHalf) matches the half-register input width without a
  // stack copy (InterleaveWholeLower only consumes the lower half).
  {
    const hn::Rebind<std::int16_t, decltype(gd32)> g16r;
    nb::Bench b; b.title("SatWidenMulAccumFixedPoint i16->i32").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto a = hs::ZeroExtendVector(hd16, hs::LowerHalf(hs::LoadU(hd16, p_i16a+i)));
        auto b2 = hs::ZeroExtendVector(hd16, hs::LowerHalf(hs::LoadU(hd16, p_i16b+i)));
        hs::StoreU(hs::SatWidenMulAccumFixedPoint(hd32, a, b2, hs::Zero(hd32)), hd32, p_o32+i);
      }
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_i16b = i16b.data(); auto* p_o32 = o32.data();
      for (std::size_t i=0;i+w32g<=kN;i+=w32g)
        hn::StoreU(hn::SatWidenMulAccumFixedPoint(gd32, hn::LoadU(g16r,p_i16a+i), hn::LoadU(g16r,p_i16b+i), hn::Zero(gd32)), gd32, p_o32+i);
      nb::doNotOptimizeAway(p_o32);
     hyle_bench::mark(rep); });
  }
  // SumsOfAdjQuadAbsDiff<0,0> u8->u16 (template offsets match Highway mpsadbw defaults)
  {
    auto u8b = gen<std::uint8_t>(14, 0, 255);
    std::vector<std::uint16_t> o16u(kN);
    const hs::ScalableTag<std::uint16_t> hdu16; const hn::ScalableTag<std::uint16_t> gdu16;
    nb::Bench b; b.title("SumsOfAdjQuadAbsDiff u8->u16").relative(true).batch(kN).minEpochIterations(5000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_u8b = u8b.data(); auto* p_o16u = o16u.data();
      for (std::size_t i=0;i+w8h<=kN;i+=w8h)
        hs::StoreU(hs::SumsOfAdjQuadAbsDiff<0,0>(hs::LoadU(hd8,p_u8a+i), hs::LoadU(hd8,p_u8b+i)), hdu16, p_o16u+i/2);
      nb::doNotOptimizeAway(p_o16u);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_u8b = u8b.data(); auto* p_o16u = o16u.data();
      for (std::size_t i=0;i+w8g<=kN;i+=w8g)
        hn::StoreU(hn::SumsOfAdjQuadAbsDiff<0,0>(hn::LoadU(gd8,p_u8a+i), hn::LoadU(gd8,p_u8b+i)), gdu16, p_o16u+i/2);
      nb::doNotOptimizeAway(p_o16u);
     hyle_bench::mark(rep); });
  }
  // OrderedDemote2To / ZeroExtendVector (single-vector Promote/Demote already thin wraps of Lower/PromoteLower)
  {
    nb::Bench b; b.title("OrderedDemote2To i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::OrderedDemote2To(hd16, hs::LoadU(hd32,p_i32a+i), hs::LoadU(hd32,p_i32a+i+w32h)), hd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::OrderedDemote2To(gd16, hn::LoadU(gd32,p_i32a+i), hn::LoadU(gd32,p_i32a+i+w32g)), gd16, p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  {
    nb::Bench b; b.title("ZeroExtendVector i16 half->full").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_o16 = o16.data();
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto half = hs::LowerHalf(hs::LoadU(hd16, p_i16a+i));
        hs::StoreU(hs::ZeroExtendVector(hd16, half), hd16, p_o16+i);
      }
      nb::doNotOptimizeAway(p_o16);
     hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i16a = i16a.data(); auto* p_o16 = o16.data();
      for (std::size_t i=0;i+w16g<=kN;i+=w16g) {
        auto half = hn::LowerHalf(hn::Half<decltype(gd16)>(), hn::LoadU(gd16, p_i16a+i));
        hn::StoreU(hn::ZeroExtendVector(gd16, half), gd16, p_o16+i);
      }
      nb::doNotOptimizeAway(p_o16);
     hyle_bench::mark(rep); });
  }

  // ---- PromoteEvenTo / PromoteOddTo i8->i16 直测 (刚向量化的承重 op; sink via GetLane 避 partial store) ----
  {
    auto i8s = gen<std::int8_t>(21, -120, 120);
    const hs::ScalableTag<std::int8_t> hi8; const hn::ScalableTag<std::int8_t> gi8;
    const hs::ScalableTag<std::int16_t> hd16b; const hn::ScalableTag<std::int16_t> gd16b;
    const std::size_t w8=hs::Lanes(hi8), w8g2=hn::Lanes(gi8);
    std::int64_t sink=0;
    {
      nb::Bench b; b.title("PromoteEvenTo i8->i16").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i8s = i8s.data(); for(std::size_t i=0;i+w8<=kN;i+=w8) sink+=hs::GetLane(hs::PromoteEvenTo(hd16b,hs::LoadU(hi8,p_i8s+i)));  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i8s = i8s.data(); for(std::size_t i=0;i+w8g2<=kN;i+=w8g2) sink+=hn::GetLane(hn::PromoteEvenTo(gd16b,hn::LoadU(gi8,p_i8s+i)));  hyle_bench::mark(rep); });
    }
    {
      nb::Bench b; b.title("PromoteOddTo i8->i16").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i8s = i8s.data(); for(std::size_t i=0;i+w8<=kN;i+=w8) sink+=hs::GetLane(hs::PromoteOddTo(hd16b,hs::LoadU(hi8,p_i8s+i)));  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i8s = i8s.data(); for(std::size_t i=0;i+w8g2<=kN;i+=w8g2) sink+=hn::GetLane(hn::PromoteOddTo(gd16b,hn::LoadU(gi8,p_i8s+i)));  hyle_bench::mark(rep); });
    }
    nb::doNotOptimizeAway(sink);
  }

  // ---- OrderedShiftRightAndDemote2To / ReorderShiftRightAndDemote2To i32->i16 (shift+narrow 融合) ----
  {
    nb::Bench b; b.title("OrderedShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::OrderedShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,p_i32a+i),hs::LoadU(hd32,p_i32a+i+w32h)),hd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::OrderedShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,p_i32a+i),hn::LoadU(gd32,p_i32a+i+w32g)),gd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  {
    nb::Bench b; b.title("ReorderShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,p_i32a+i),hs::LoadU(hd32,p_i32a+i+w32h)),hd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,p_i32a+i),hn::LoadU(gd32,p_i32a+i+w32g)),gd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  {
    nb::Bench b; b.title("OrderedRoundingShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::OrderedRoundingShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,p_i32a+i),hs::LoadU(hd32,p_i32a+i+w32h)),hd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::OrderedRoundingShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,p_i32a+i),hn::LoadU(gd32,p_i32a+i+w32g)),gd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }
  {
    nb::Bench b; b.title("ReorderRoundingShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderRoundingShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,p_i32a+i),hs::LoadU(hd32,p_i32a+i+w32h)),hd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_i32a = i32a.data(); auto* p_o16 = o16.data(); for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderRoundingShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,p_i32a+i),hn::LoadU(gd32,p_i32a+i+w32g)),gd16,p_o16+i); nb::doNotOptimizeAway(p_o16);  hyle_bench::mark(rep); });
  }

  // ---- directive2 覆盖补齐: SAD/pairwise-sum 族 (SumsOf8 已有; 补 2/4/8AbsDiff) ----
  {
    const hs::ScalableTag<std::uint16_t> hd16u; const hn::ScalableTag<std::uint16_t> gd16u;
    std::vector<std::uint16_t> o16u(kN);
    nb::Bench b; b.title("SumsOf2 u8->u16").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_o16u = o16u.data(); for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf2(hs::LoadU(hd8,p_u8a+i)), hd16u, p_o16u+i/2); nb::doNotOptimizeAway(p_o16u);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_o16u = o16u.data(); for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf2(hn::LoadU(gd8,p_u8a+i)), gd16u, p_o16u+i/2); nb::doNotOptimizeAway(p_o16u);  hyle_bench::mark(rep); });
  }
  {
    std::vector<std::uint32_t> o32u(kN);
    nb::Bench b; b.title("SumsOf4 u8->u32").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_o32u = o32u.data(); for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf4(hs::LoadU(hd8,p_u8a+i)), hd32u, p_o32u+i/4); nb::doNotOptimizeAway(p_o32u);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_o32u = o32u.data(); for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf4(hn::LoadU(gd8,p_u8a+i)), gd32u, p_o32u+i/4); nb::doNotOptimizeAway(p_o32u);  hyle_bench::mark(rep); });
  }
  {
    auto u8b = gen<std::uint8_t>(6, 0, 255);
    nb::Bench b; b.title("SumsOf8AbsDiff u8->u64").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_u8b = u8b.data(); auto* p_o64 = o64.data(); for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf8AbsDiff(hs::LoadU(hd8,p_u8a+i),hs::LoadU(hd8,p_u8b+i)), hd64, p_o64+i/8); nb::doNotOptimizeAway(p_o64);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_u8a = u8a.data(); auto* p_u8b = u8b.data(); auto* p_o64 = o64.data(); for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf8AbsDiff(hn::LoadU(gd8,p_u8a+i),hn::LoadU(gd8,p_u8b+i)), gd64, p_o64+i/8); nb::doNotOptimizeAway(p_o64);  hyle_bench::mark(rep); });
  }

  // ---- coverage re-audit (2026-07-14): bf16/dtype ops the old doc falsely claimed "covered by dtype_bench" ----
  // bf16 = uint16 IEEE carrier (2N lanes) -> f32 (N lanes). MulEvenAdd/MulOddAdd are the bf16 dot-product FMA.
  {
    const hs::ScalableTag<std::uint16_t> hdbf; const hn::ScalableTag<std::uint16_t> gdbf;
    const hs::ScalableTag<float> hdf; const hn::ScalableTag<float> gdf;
    const std::size_t wbfh=hs::Lanes(hdbf), wbfg=hn::Lanes(gdbf);
    // Highway's MulEvenAdd takes a bfloat16_t-typed vector (Repartition of the f32 tag); bit-cast the
    // uint16 carrier load into it. HYLE's takes the uint16 carrier directly -- same bits, equal work.
    const hn::Repartition<hwy::bfloat16_t, decltype(gdf)> gdbf16;
    std::mt19937_64 rr(77); std::uniform_int_distribution<int> dd(0x3c00, 0x4200);  // benign bf16 exps
    std::vector<std::uint16_t> bfa(kN), bfb(kN); for(std::size_t i=0;i<kN;++i){ bfa[i]=(std::uint16_t)dd(rr); bfb[i]=(std::uint16_t)dd(rr);}
    std::vector<float> fcc(kN, 1.0f), foo(kN);
    {
      nb::Bench b; b.title("MulEvenAdd bf16->f32").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a=bfa.data(); auto* p_b=bfb.data(); auto* p_c=fcc.data(); auto* p_o=foo.data(); for(std::size_t i=0;i+wbfh<=kN;i+=wbfh){ hs::StoreU(hs::MulEvenAdd(hdf, hs::LoadU(hdbf,p_a+i), hs::LoadU(hdbf,p_b+i), hs::LoadU(hdf,p_c+i/2)), hdf, p_o+i/2);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a=bfa.data(); auto* p_b=bfb.data(); auto* p_c=fcc.data(); auto* p_o=foo.data(); for(std::size_t i=0;i+wbfg<=kN;i+=wbfg){ hn::StoreU(hn::MulEvenAdd(gdf, hn::BitCast(gdbf16,hn::LoadU(gdbf,p_a+i)), hn::BitCast(gdbf16,hn::LoadU(gdbf,p_b+i)), hn::LoadU(gdf,p_c+i/2)), gdf, p_o+i/2);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); });
    }
    {
      nb::Bench b; b.title("MulOddAdd bf16->f32").relative(true).batch(kN).minEpochIterations(8000);
      HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a=bfa.data(); auto* p_b=bfb.data(); auto* p_c=fcc.data(); auto* p_o=foo.data(); for(std::size_t i=0;i+wbfh<=kN;i+=wbfh){ hs::StoreU(hs::MulOddAdd(hdf, hs::LoadU(hdbf,p_a+i), hs::LoadU(hdbf,p_b+i), hs::LoadU(hdf,p_c+i/2)), hdf, p_o+i/2);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a=bfa.data(); auto* p_b=bfb.data(); auto* p_c=fcc.data(); auto* p_o=foo.data(); for(std::size_t i=0;i+wbfg<=kN;i+=wbfg){ hn::StoreU(hn::MulOddAdd(gdf, hn::BitCast(gdbf16,hn::LoadU(gdbf,p_a+i)), hn::BitCast(gdbf16,hn::LoadU(gdbf,p_b+i)), hn::LoadU(gdf,p_c+i/2)), gdf, p_o+i/2);} nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); });
    }
  }
  // U8FromU32: u32 (N lanes) -> u8 (N lanes packed low). HYLE VFromD<Simd<u8,N,0>>, Highway Vec128<u8,8>.
  {
    auto u32a = gen<std::uint32_t>(21, 0, 255);
    const hs::CappedTag<std::uint8_t, 8> hd8c; const hn::CappedTag<std::uint8_t, 8> gd8c;
    nb::Bench b; b.title("U8FromU32 u32->u8").relative(true).batch(kN).minEpochIterations(8000);
    HYLE_RUN_PAIR(b, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a=u32a.data(); auto* p_o=o8.data(); for(std::size_t i=0;i+w32h<=kN;i+=w32h) hs::StoreU(hs::U8FromU32(hs::LoadU(hd32u,p_a+i)), hd8c, p_o+i); nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); }, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a=u32a.data(); auto* p_o=o8.data(); for(std::size_t i=0;i+w32g<=kN;i+=w32g) hn::StoreU(hn::U8FromU32(hn::LoadU(gd32u,p_a+i)), gd8c, p_o+i); nb::doNotOptimizeAway(p_o);  hyle_bench::mark(rep); });
  }

  return 0;
}
