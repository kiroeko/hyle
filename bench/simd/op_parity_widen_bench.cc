// op_parity_widen -- the bespoke-harness sweep: HYLE self-made width-CHANGING ops (promote/demote/
// truncate/convert/SumsOf/ReorderDemote2To/WidenMulPairwiseAdd) vs Highway. These change lane
// count/width so they need explicit dst tags, not the uniform loop of op_parity_full. They are the
// prime store+reload suspects. Same TU / same ISA / both inline. -n15, read median ns/op.
#include <nanobench.h>
#include <cstddef>
#include <cstdint>
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
    b.run("HYLE", [&]{ for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::PromoteUpperTo(hd32, hs::LoadU(hd16,i16a.data()+i)), hd32, o32.data()+i); nb::doNotOptimizeAway(o32.data()); });
    b.run("Highway", [&]{ const hn::Rebind<std::int32_t, hn::Half<decltype(gd16)>> gdn; for(std::size_t i=0;i+w16g<=kN;i+=w16g){ auto up = hn::UpperHalf(hn::Half<decltype(gd16)>(), hn::LoadU(gd16,i16a.data()+i)); hn::StoreU(hn::PromoteTo(gdn, up), gdn, o32.data()+i);} nb::doNotOptimizeAway(o32.data()); });
  }
  // DemoteTo i32->i16 (HYLE two wide -> one narrow; Highway ReorderDemote2To or DemoteTo per half)
  {
    nb::Bench b; b.title("DemoteTo i32->i16 (sat)").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::DemoteTo(hd16, hs::LoadU(hd32,i32a.data()+i), hs::LoadU(hd32,i32a.data()+i+w32h)), hd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
    b.run("Highway", [&]{ const hn::Rebind<std::int16_t, decltype(gd32)> gdn; for(std::size_t i=0;i+w32g<=kN;i+=w32g){ auto d = hn::DemoteTo(gdn, hn::LoadU(gd32,i32a.data()+i)); hn::StoreU(d, gdn, o16.data()+i);} nb::doNotOptimizeAway(o16.data()); });
  }
  // ReorderDemote2To i32->i16
  {
    nb::Bench b; b.title("ReorderDemote2To i32->i16").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+2*w32h<=kN;i+=2*w32h) hs::StoreU(hs::ReorderDemote2To(hd16, hs::LoadU(hd32,i32a.data()+i), hs::LoadU(hd32,i32a.data()+i+w32h)), hd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+2*w32g<=kN;i+=2*w32g) hn::StoreU(hn::ReorderDemote2To(gd16, hn::LoadU(gd32,i32a.data()+i), hn::LoadU(gd32,i32a.data()+i+w32g)), gd16, o16.data()+i); nb::doNotOptimizeAway(o16.data()); });
  }
  // WidenMulPairwiseAdd i16->i32
  {
    nb::Bench b; b.title("WidenMulPairwiseAdd i16->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+w16h<=kN;i+=w16h) hs::StoreU(hs::WidenMulPairwiseAdd(hd32, hs::LoadU(hd16,i16a.data()+i), hs::LoadU(hd16,i16b.data()+i)), hd32, o32.data()+i); nb::doNotOptimizeAway(o32.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+w16g<=kN;i+=w16g) hn::StoreU(hn::WidenMulPairwiseAdd(gd32, hn::LoadU(gd16,i16a.data()+i), hn::LoadU(gd16,i16b.data()+i)), gd32, o32.data()+i); nb::doNotOptimizeAway(o32.data()); });
  }
  // SumsOf8 u8->u64
  {
    nb::Bench b; b.title("SumsOf8 u8->u64").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+w8h<=kN;i+=w8h) hs::StoreU(hs::SumsOf8(hs::LoadU(hd8,u8a.data()+i)), hd64, o64.data()+i/8); nb::doNotOptimizeAway(o64.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+w8g<=kN;i+=w8g) hn::StoreU(hn::SumsOf8(hn::LoadU(gd8,u8a.data()+i)), gd64, o64.data()+i/8); nb::doNotOptimizeAway(o64.data()); });
  }
  return 0;
}
