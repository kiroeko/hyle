// op_parity_widen -- the bespoke-harness sweep: HYLE self-made width-CHANGING ops (promote/demote/
// truncate/convert/SumsOf/ReorderDemote2To/WidenMulPairwiseAdd) vs Highway. These change lane
// count/width so they need explicit dst tags, not the uniform loop of op_parity_full. They are the
// prime store+reload suspects. Same TU / same ISA / both inline. -n15, read median ns/op.
#include <nanobench.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include "hyle/simd/vec.h"
#include "hwy/highway.h"
namespace hn = hwy::HWY_NAMESPACE;
namespace hs = hyle::simd;
namespace nb = ankerl::nanobench;

namespace {
constexpr std::size_t kN = 4096;
template <class T> std::vector<T> gen(std::uint64_t s, T lo, T hi){ std::mt19937_64 r(s); std::uniform_int_distribution<long long> d(lo,hi); std::vector<T> v(kN); for(auto&e:v)e=(T)d(r); return v; }
// Order-swap control (see op_parity_full_bench.cc): env HYLE_BENCH_HWYFIRST runs Highway first.
inline bool bench_hwy_first(){ static const bool v = std::getenv("HYLE_BENCH_HWYFIRST") != nullptr; return v; }
template <class FH, class FG>
inline void run_pair(nb::Bench& b, FH&& h, FG&& g){
  if(bench_hwy_first()){ b.run("Highway", g); b.run("HYLE", h); }
  else { b.run("HYLE", h); b.run("Highway", g); }
}
}

int main(){
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
    run_pair(b, [&]{ for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::PromoteUpperTo(hd32, hs::LoadU(hd16,i16a.data()+i)), hd32, o32.data()+i); nb::doNotOptimizeAway(o32.data()); }, [&]{ const hn::Rebind<std::int32_t, hn::Half<decltype(gd16)>> gdn; for(std::size_t i=0;i+w16g<=kN;i+=w16g){ auto up = hn::UpperHalf(hn::Half<decltype(gd16)>(), hn::LoadU(gd16,i16a.data()+i)); hn::StoreU(hn::PromoteTo(gdn, up), gdn, o32.data()+i);} nb::doNotOptimizeAway(o32.data()); });
  }
  // DemoteTo i32->i16 (HYLE two wide -> one narrow; Highway ReorderDemote2To or DemoteTo per half)
  {
    nb::Bench b; b.title("DemoteTo i32->i16 (sat)").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::DemoteTo(hd16, hs::LoadU(hd32,i32a.data()+i), hs::LoadU(hd32,i32a.data()+i+w32h)), hd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ const hn::Rebind<std::int16_t, decltype(gd32)> gdn; for(std::size_t i=0;i+w32g<=kN;i+=w32g){ auto d = hn::DemoteTo(gdn, hn::LoadU(gd32,i32a.data()+i)); hn::StoreU(d, gdn, o16.data()+i);} nb::doNotOptimizeAway(o16.data()); });
  }
  // ReorderDemote2To i32->i16
  {
    nb::Bench b; b.title("ReorderDemote2To i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderDemote2To(hd16, hs::LoadU(hd32,i32a.data()+i), hs::LoadU(hd32,i32a.data()+i+w32h)), hd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderDemote2To(gd16, hn::LoadU(gd32,i32a.data()+i), hn::LoadU(gd32,i32a.data()+i+w32g)), gd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  // WidenMulPairwiseAdd i16->i32
  {
    nb::Bench b; b.title("WidenMulPairwiseAdd i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::WidenMulPairwiseAdd(hd32, hs::LoadU(hd16,i16a.data()+i), hs::LoadU(hd16,i16b.data()+i)), hd32, o32.data()+i); nb::doNotOptimizeAway(o32.data()); }, [&]{ for(std::size_t i=0;i+w16g<=kN;i+=w16g) hn::StoreU(hn::WidenMulPairwiseAdd(gd32, hn::LoadU(gd16,i16a.data()+i), hn::LoadU(gd16,i16b.data()+i)), gd32, o32.data()+i); nb::doNotOptimizeAway(o32.data()); });
  }
  // MulFixedPoint15 i16
  {
    nb::Bench b; b.title("MulFixedPoint15 i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::MulFixedPoint15(hs::LoadU(hd16,i16a.data()+i), hs::LoadU(hd16,i16b.data()+i)), hd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+w16g<=kN;i+=w16g) hn::StoreU(hn::MulFixedPoint15(hn::LoadU(gd16,i16a.data()+i), hn::LoadU(gd16,i16b.data()+i)), gd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  // SumsOf8 u8->u64
  {
    nb::Bench b; b.title("SumsOf8 u8->u64").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf8(hs::LoadU(hd8,u8a.data()+i)), hd64, o64.data()+i/8); nb::doNotOptimizeAway(o64.data()); }, [&]{ for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf8(hn::LoadU(gd8,u8a.data()+i)), gd64, o64.data()+i/8); nb::doNotOptimizeAway(o64.data()); });
  }
  // GetExponent f32 (HYLE store+reload+std::logb -- prime suspect)
  {
    auto fa = [](){ std::mt19937_64 r(9); std::uniform_real_distribution<float> d(0.5f,1e6f); std::vector<float> v(kN); for(auto&e:v)e=d(r); return v; }();
    std::vector<float> fo(kN);
    const hs::ScalableTag<float> hdf; const hn::ScalableTag<float> gdf;
    const std::size_t wfh=hs::Lanes(hdf), wfg=hn::Lanes(gdf);
    nb::Bench b; b.title("GetExponent f32").relative(true).batch(kN).minEpochIterations(5000);
    run_pair(b, [&]{ for(std::size_t i=0;i+wfh<=kN;i+=wfh) hs::StoreU(hs::GetExponent(hs::LoadU(hdf,fa.data()+i)), hdf, fo.data()+i); nb::doNotOptimizeAway(fo.data()); }, [&]{ for(std::size_t i=0;i+wfg<=kN;i+=wfg) hn::StoreU(hn::GetExponent(hn::LoadU(gdf,fa.data()+i)), gdf, fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }

  // ---- Tier1: widen MAC family ----
  // ReorderWidenMulAccumulate i16->i32
  {
    nb::Bench b; b.title("ReorderWidenMulAccumulate i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto sum1 = hs::Zero(hd32);
        auto sum0 = hs::ReorderWidenMulAccumulate(hd32, hs::LoadU(hd16,i16a.data()+i), hs::LoadU(hd16,i16b.data()+i), hs::Zero(hd32), sum1);
        hs::StoreU(hs::RearrangeToOddPlusEven(sum0, sum1), hd32, o32.data()+i);
      }
      nb::doNotOptimizeAway(o32.data());
    }, [&]{
      for (std::size_t i=0;i+w16g<=kN;i+=w16g) {
        auto sum1 = hn::Zero(gd32);
        auto sum0 = hn::ReorderWidenMulAccumulate(gd32, hn::LoadU(gd16,i16a.data()+i), hn::LoadU(gd16,i16b.data()+i), hn::Zero(gd32), sum1);
        hn::StoreU(hn::RearrangeToOddPlusEven(sum0, sum1), gd32, o32.data()+i);
      }
      nb::doNotOptimizeAway(o32.data());
    });
  }
  // SatWidenMulPairwiseAdd u8*i8 -> i16
  {
    auto i8b = gen<std::int8_t>(11, -100, 100);
    std::vector<std::int16_t> o16s(kN);
    const hs::ScalableTag<std::int8_t> hi8; const hn::ScalableTag<std::int8_t> gi8;
    nb::Bench b; b.title("SatWidenMulPairwiseAdd u8*i8->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w8h<=kN;i+=w8h)
        hs::StoreU(hs::SatWidenMulPairwiseAdd(hd16, hs::LoadU(hd8,u8a.data()+i), hs::LoadU(hi8,i8b.data()+i)), hd16, o16s.data()+i);
      nb::doNotOptimizeAway(o16s.data());
    }, [&]{
      for (std::size_t i=0;i+w8g<=kN;i+=w8g)
        hn::StoreU(hn::SatWidenMulPairwiseAdd(gd16, hn::LoadU(gd8,u8a.data()+i), hn::LoadU(gi8,i8b.data()+i)), gd16, o16s.data()+i);
      nb::doNotOptimizeAway(o16s.data());
    });
  }
  // SumOfMulQuadAccumulate i8->i32
  {
    auto i8a = gen<std::int8_t>(12, -50, 50);
    auto i8b = gen<std::int8_t>(13, -50, 50);
    const hs::ScalableTag<std::int8_t> hi8; const hn::ScalableTag<std::int8_t> gi8;
    const std::size_t w8ih=hs::Lanes(hi8), w8ig=hn::Lanes(gi8);
    nb::Bench b; b.title("SumOfMulQuadAccumulate i8->i32").relative(true).batch(kN).minEpochIterations(5000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w8ih<=kN;i+=w8ih)
        hs::StoreU(hs::SumOfMulQuadAccumulate(hd32, hs::LoadU(hi8,i8a.data()+i), hs::LoadU(hi8,i8b.data()+i), hs::Zero(hd32)), hd32, o32.data()+i/4);
      nb::doNotOptimizeAway(o32.data());
    }, [&]{
      for (std::size_t i=0;i+w8ig<=kN;i+=w8ig)
        hn::StoreU(hn::SumOfMulQuadAccumulate(gd32, hn::LoadU(gi8,i8a.data()+i), hn::LoadU(gi8,i8b.data()+i), hn::Zero(gd32)), gd32, o32.data()+i/4);
      nb::doNotOptimizeAway(o32.data());
    });
  }
  // WidenMulAccumulate i16->i32 (half-at-a-time dual acc)
  {
    nb::Bench b; b.title("WidenMulAccumulate i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto high = hs::Zero(hd32);
        auto low = hs::WidenMulAccumulate(hd32, hs::LoadU(hd16,i16a.data()+i), hs::LoadU(hd16,i16b.data()+i), hs::Zero(hd32), high);
        hs::StoreU(hs::Add(low, high), hd32, o32.data()+i);
      }
      nb::doNotOptimizeAway(o32.data());
    }, [&]{
      for (std::size_t i=0;i+w16g<=kN;i+=w16g) {
        auto high = hn::Zero(gd32);
        auto low = hn::WidenMulAccumulate(gd32, hn::LoadU(gd16,i16a.data()+i), hn::LoadU(gd16,i16b.data()+i), hn::Zero(gd32), high);
        hn::StoreU(hn::Add(low, high), gd32, o32.data()+i);
      }
      nb::doNotOptimizeAway(o32.data());
    });
  }
  // SatWidenMulPairwiseAccumulate i16->i32
  {
    nb::Bench b; b.title("SatWidenMulPairwiseAccumulate i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w16h<=kN;i+=w16h)
        hs::StoreU(hs::SatWidenMulPairwiseAccumulate(hd32, hs::LoadU(hd16,i16a.data()+i), hs::LoadU(hd16,i16b.data()+i), hs::Zero(hd32)), hd32, o32.data()+i);
      nb::doNotOptimizeAway(o32.data());
    }, [&]{
      for (std::size_t i=0;i+w16g<=kN;i+=w16g)
        hn::StoreU(hn::SatWidenMulPairwiseAccumulate(gd32, hn::LoadU(gd16,i16a.data()+i), hn::LoadU(gd16,i16b.data()+i), hn::Zero(gd32)), gd32, o32.data()+i);
      nb::doNotOptimizeAway(o32.data());
    });
  }
  // SatWidenMulAccumFixedPoint: Highway takes Rebind<i16,di32> (same lane count as i32 = half-register).
  // HYLE accepts full V16; ZeroExtendVector(LowerHalf) matches the half-register input width without a
  // stack copy (InterleaveWholeLower only consumes the lower half).
  {
    const hn::Rebind<std::int16_t, decltype(gd32)> g16r;
    nb::Bench b; b.title("SatWidenMulAccumFixedPoint i16->i32").relative(true).batch(kN).minEpochIterations(5000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto a = hs::ZeroExtendVector(hd16, hs::LowerHalf(hs::LoadU(hd16, i16a.data()+i)));
        auto b2 = hs::ZeroExtendVector(hd16, hs::LowerHalf(hs::LoadU(hd16, i16b.data()+i)));
        hs::StoreU(hs::SatWidenMulAccumFixedPoint(hd32, a, b2, hs::Zero(hd32)), hd32, o32.data()+i);
      }
      nb::doNotOptimizeAway(o32.data());
    }, [&]{
      for (std::size_t i=0;i+w32g<=kN;i+=w32g)
        hn::StoreU(hn::SatWidenMulAccumFixedPoint(gd32, hn::LoadU(g16r,i16a.data()+i), hn::LoadU(g16r,i16b.data()+i), hn::Zero(gd32)), gd32, o32.data()+i);
      nb::doNotOptimizeAway(o32.data());
    });
  }
  // SumsOfAdjQuadAbsDiff<0,0> u8->u16 (template offsets match Highway mpsadbw defaults)
  {
    auto u8b = gen<std::uint8_t>(14, 0, 255);
    std::vector<std::uint16_t> o16u(kN);
    const hs::ScalableTag<std::uint16_t> hdu16; const hn::ScalableTag<std::uint16_t> gdu16;
    nb::Bench b; b.title("SumsOfAdjQuadAbsDiff u8->u16").relative(true).batch(kN).minEpochIterations(5000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w8h<=kN;i+=w8h)
        hs::StoreU(hs::SumsOfAdjQuadAbsDiff<0,0>(hs::LoadU(hd8,u8a.data()+i), hs::LoadU(hd8,u8b.data()+i)), hdu16, o16u.data()+i/2);
      nb::doNotOptimizeAway(o16u.data());
    }, [&]{
      for (std::size_t i=0;i+w8g<=kN;i+=w8g)
        hn::StoreU(hn::SumsOfAdjQuadAbsDiff<0,0>(hn::LoadU(gd8,u8a.data()+i), hn::LoadU(gd8,u8b.data()+i)), gdu16, o16u.data()+i/2);
      nb::doNotOptimizeAway(o16u.data());
    });
  }
  // OrderedDemote2To / ZeroExtendVector (single-vector Promote/Demote already thin wraps of Lower/PromoteLower)
  {
    nb::Bench b; b.title("OrderedDemote2To i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::OrderedDemote2To(hd16, hs::LoadU(hd32,i32a.data()+i), hs::LoadU(hd32,i32a.data()+i+w32h)), hd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::OrderedDemote2To(gd16, hn::LoadU(gd32,i32a.data()+i), hn::LoadU(gd32,i32a.data()+i+w32g)), gd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  {
    nb::Bench b; b.title("ZeroExtendVector i16 half->full").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{
      for (std::size_t i=0;i+w16h<=kN;i+=w16h) {
        auto half = hs::LowerHalf(hs::LoadU(hd16, i16a.data()+i));
        hs::StoreU(hs::ZeroExtendVector(hd16, half), hd16, o16.data()+i);
      }
      nb::doNotOptimizeAway(o16.data());
    }, [&]{
      for (std::size_t i=0;i+w16g<=kN;i+=w16g) {
        auto half = hn::LowerHalf(hn::Half<decltype(gd16)>(), hn::LoadU(gd16, i16a.data()+i));
        hn::StoreU(hn::ZeroExtendVector(gd16, half), gd16, o16.data()+i);
      }
      nb::doNotOptimizeAway(o16.data());
    });
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
      run_pair(b, [&]{ for(std::size_t i=0;i+w8<=kN;i+=w8) sink+=hs::GetLane(hs::PromoteEvenTo(hd16b,hs::LoadU(hi8,i8s.data()+i))); }, [&]{ for(std::size_t i=0;i+w8g2<=kN;i+=w8g2) sink+=hn::GetLane(hn::PromoteEvenTo(gd16b,hn::LoadU(gi8,i8s.data()+i))); });
    }
    {
      nb::Bench b; b.title("PromoteOddTo i8->i16").relative(true).batch(kN).minEpochIterations(8000);
      run_pair(b, [&]{ for(std::size_t i=0;i+w8<=kN;i+=w8) sink+=hs::GetLane(hs::PromoteOddTo(hd16b,hs::LoadU(hi8,i8s.data()+i))); }, [&]{ for(std::size_t i=0;i+w8g2<=kN;i+=w8g2) sink+=hn::GetLane(hn::PromoteOddTo(gd16b,hn::LoadU(gi8,i8s.data()+i))); });
    }
    nb::doNotOptimizeAway(sink);
  }

  // ---- OrderedShiftRightAndDemote2To / ReorderShiftRightAndDemote2To i32->i16 (shift+narrow 融合) ----
  {
    nb::Bench b; b.title("OrderedShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::OrderedShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,i32a.data()+i),hs::LoadU(hd32,i32a.data()+i+w32h)),hd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::OrderedShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,i32a.data()+i),hn::LoadU(gd32,i32a.data()+i+w32g)),gd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  {
    nb::Bench b; b.title("ReorderShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,i32a.data()+i),hs::LoadU(hd32,i32a.data()+i+w32h)),hd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,i32a.data()+i),hn::LoadU(gd32,i32a.data()+i+w32g)),gd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  {
    nb::Bench b; b.title("OrderedRoundingShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::OrderedRoundingShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,i32a.data()+i),hs::LoadU(hd32,i32a.data()+i+w32h)),hd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::OrderedRoundingShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,i32a.data()+i),hn::LoadU(gd32,i32a.data()+i+w32g)),gd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  {
    nb::Bench b; b.title("ReorderRoundingShiftRightAndDemote2To<2> i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderRoundingShiftRightAndDemote2To<2>(hd16,hs::LoadU(hd32,i32a.data()+i),hs::LoadU(hd32,i32a.data()+i+w32h)),hd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); }, [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderRoundingShiftRightAndDemote2To<2>(gd16,hn::LoadU(gd32,i32a.data()+i),hn::LoadU(gd32,i32a.data()+i+w32g)),gd16,o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }

  // ---- directive2 覆盖补齐: SAD/pairwise-sum 族 (SumsOf8 已有; 补 2/4/8AbsDiff) ----
  {
    const hs::ScalableTag<std::uint16_t> hd16u; const hn::ScalableTag<std::uint16_t> gd16u;
    std::vector<std::uint16_t> o16u(kN);
    nb::Bench b; b.title("SumsOf2 u8->u16").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf2(hs::LoadU(hd8,u8a.data()+i)), hd16u, o16u.data()+i/2); nb::doNotOptimizeAway(o16u.data()); }, [&]{ for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf2(hn::LoadU(gd8,u8a.data()+i)), gd16u, o16u.data()+i/2); nb::doNotOptimizeAway(o16u.data()); });
  }
  {
    std::vector<std::uint32_t> o32u(kN);
    nb::Bench b; b.title("SumsOf4 u8->u32").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf4(hs::LoadU(hd8,u8a.data()+i)), hd32u, o32u.data()+i/4); nb::doNotOptimizeAway(o32u.data()); }, [&]{ for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf4(hn::LoadU(gd8,u8a.data()+i)), gd32u, o32u.data()+i/4); nb::doNotOptimizeAway(o32u.data()); });
  }
  {
    auto u8b = gen<std::uint8_t>(6, 0, 255);
    nb::Bench b; b.title("SumsOf8AbsDiff u8->u64").relative(true).batch(kN).minEpochIterations(8000);
    run_pair(b, [&]{ for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf8AbsDiff(hs::LoadU(hd8,u8a.data()+i),hs::LoadU(hd8,u8b.data()+i)), hd64, o64.data()+i/8); nb::doNotOptimizeAway(o64.data()); }, [&]{ for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf8AbsDiff(hn::LoadU(gd8,u8a.data()+i),hn::LoadU(gd8,u8b.data()+i)), gd64, o64.data()+i/8); nb::doNotOptimizeAway(o64.data()); });
  }

  return 0;
}
