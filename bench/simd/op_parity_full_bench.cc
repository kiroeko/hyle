// op_parity_full -- FULL sweep of HYLE self-made ops vs Highway, same TU / same ISA / both inline.
// Auto-covers the clean unary/binary/ternary/compare shapes on i32 (or f32 for float-only ops).
// Ops needing bespoke harness (convert/widen/interleaved-load/crypto/gather) are in op_parity_bench.cc
// or documented as skipped. Run with bench-stat.py -n15; read MEDIAN ns/op (sub-ns kernels: ratio
// spread is noisy, median ratio is the signal). >100% HYLE faster, <100% HYLE slower.
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
std::vector<std::int32_t> geni(std::uint64_t s){ std::mt19937_64 r(s); std::uniform_int_distribution<int> d(-1000,1000); std::vector<std::int32_t> v(kN); for(auto&e:v)e=d(r); return v; }
std::vector<float> genf(std::uint64_t s){ std::mt19937_64 r(s); std::uniform_real_distribution<float> d(0.5f,4.0f); std::vector<float> v(kN); for(auto&e:v)e=d(r); return v; }
}

// UNARY int:  out[i] = OP(a[i])
#define UNARY_I(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(HYOP(hs::LoadU(hdi,a.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(HWYOP(hn::LoadU(gdi,a.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
} while(0)
// BINARY int: out[i] = OP(a[i], b2[i])
#define BINARY_I(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(HYOP(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(HWYOP(hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
} while(0)
// UNARY float
#define UNARY_F(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(HYOP(hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(HWYOP(hn::LoadU(gdf,fa.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
} while(0)
// COMPARE int -> mask -> VecFromMask (store the mask as vec so the op isn't DCE'd)
#define CMP_I(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::VecFromMask(hdi,HYOP(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i))),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::VecFromMask(gdi,HWYOP(hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i))),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
} while(0)

int main(){
  auto a=geni(1); auto b2=geni(2); std::vector<std::int32_t> o(kN);
  auto fa=genf(3); std::vector<float> fo(kN);
  const hs::ScalableTag<std::int32_t> hdi; const hn::ScalableTag<std::int32_t> gdi;
  const hs::ScalableTag<float> hdf; const hn::ScalableTag<float> gdf;
  const std::size_t hw=hs::Lanes(hdi), gw=hn::Lanes(gdi), hwf=hs::Lanes(hdf), gwf=hn::Lanes(gdf);

  // unary int
  UNARY_I("Abs i32", hs::Abs, hn::Abs);
  UNARY_I("Neg i32", hs::Neg, hn::Neg);
  UNARY_I("BroadcastSignBit i32", hs::BroadcastSignBit, hn::BroadcastSignBit);
  UNARY_I("DupEven i32", hs::DupEven, hn::DupEven);
  UNARY_I("DupOdd i32", hs::DupOdd, hn::DupOdd);
  UNARY_I("ReverseBits i32", hs::ReverseBits, hn::ReverseBits);
  UNARY_I("TrailingZeroCount i32", hs::TrailingZeroCount, hn::TrailingZeroCount);
  UNARY_I("ShiftLeft<3> i32", hs::ShiftLeft<3>, hn::ShiftLeft<3>);
  UNARY_I("ShiftRight<3> i32", hs::ShiftRight<3>, hn::ShiftRight<3>);
  // binary int
  BINARY_I("Sub i32", hs::Sub, hn::Sub);
  BINARY_I("Mul i32", hs::Mul, hn::Mul);
  BINARY_I("Max i32", hs::Max, hn::Max);
  BINARY_I("AbsDiff i32", hs::AbsDiff, hn::AbsDiff);
  BINARY_I("OddEven i32", hs::OddEven, hn::OddEven);
  BINARY_I("InterleaveUpper i32", [&](auto x,auto y){return hs::InterleaveUpper(x,y);}, [&](auto x,auto y){return hn::InterleaveUpper(gdi,x,y);});
  BINARY_I("ConcatOdd i32", [&](auto x,auto y){return hs::ConcatOdd(x,y);}, [&](auto x,auto y){return hn::ConcatOdd(gdi,x,y);});
  BINARY_I("Shl i32", hs::Shl, hn::Shl);
  // unary float
  UNARY_F("Sqrt f32", hs::Sqrt, hn::Sqrt);
  UNARY_F("ApproximateReciprocal f32", hs::ApproximateReciprocal, hn::ApproximateReciprocal);
  UNARY_F("ApproximateReciprocalSqrt f32", hs::ApproximateReciprocalSqrt, hn::ApproximateReciprocalSqrt);
  // compare
  CMP_I("Eq i32", hs::Eq, hn::Eq);
  CMP_I("Lt i32", hs::Lt, hn::Lt);
  CMP_I("Gt i32", hs::Gt, hn::Gt);
  return 0;
}
