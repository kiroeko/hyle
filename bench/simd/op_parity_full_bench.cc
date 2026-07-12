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
#define HWY_HEADER_ONLY
#include "hwy/highway.h"
#include "hwy/contrib/intdiv/intdiv-inl.h"
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
// UNARY/BINARY/TERNARY float
#define UNARY_F(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(HYOP(hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(HWYOP(hn::LoadU(gdf,fa.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
} while(0)
#define BINARY_F(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(HYOP(hs::LoadU(hdf,fa.data()+i),hs::LoadU(hdf,fb.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(HWYOP(hn::LoadU(gdf,fa.data()+i),hn::LoadU(gdf,fb.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
} while(0)
#define TERNARY_F(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(HYOP(hs::LoadU(hdf,fa.data()+i),hs::LoadU(hdf,fb.data()+i),hs::LoadU(hdf,fc.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(HWYOP(hn::LoadU(gdf,fa.data()+i),hn::LoadU(gdf,fb.data()+i),hn::LoadU(gdf,fc.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); }); \
} while(0)
// COMPARE int -> mask -> VecFromMask (store the mask as vec so the op isn't DCE'd)
#define CMP_I(TITLE, HYOP, HWYOP) do { \
  nb::Bench b; b.title(TITLE).relative(true).batch(kN).minEpochIterations(8000); \
  b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::VecFromMask(hdi,HYOP(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i))),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
  b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::VecFromMask(gdi,HWYOP(hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i))),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); }); \
} while(0)

int main(){
  auto a=geni(1); auto b2=geni(2); std::vector<std::int32_t> o(kN);
  auto fa=genf(3); auto fb=genf(4); auto fc=genf(5); std::vector<float> fo(kN);
  const hs::ScalableTag<std::int32_t> hdi; const hn::ScalableTag<std::int32_t> gdi;
  const hs::ScalableTag<float> hdf; const hn::ScalableTag<float> gdf;
  const std::size_t hw=hs::Lanes(hdi), gw=hn::Lanes(gdi), hwf=hs::Lanes(hdf), gwf=hn::Lanes(gdf);
  volatile std::int32_t runtime_divisor=7;
  const std::int32_t divisor=runtime_divisor;  // runtime invariant, read once outside timing
  const hs::Divisor<std::int32_t> hdiv(divisor);
  const auto gdiv=hn::ComputeDivisorParams(divisor);

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

  // P4 invariant divide: parameter precompute is outside both timed loops. Highway's contrib
  // IntDiv is the matching reusable-divisor API; both remainder paths are `a - q * divisor`.
  {
    nb::Bench bench; bench.title("Mod i32 invariant").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) { const auto x=hs::LoadU(hdi,a.data()+i); const auto r=hs::Mod(x,hdiv); nb::doNotOptimizeAway(r); } });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) { const auto x=hn::LoadU(gdi,a.data()+i); const auto q=hn::IntDiv(gdi,x,gdiv); const auto r=hn::Sub(x,hn::Mul(q,hn::Set(gdi,gdiv.divisor))); nb::doNotOptimizeAway(r); } });
  }
  {
    nb::Bench bench; bench.title("MaskedModOr i32 invariant").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) { const auto x=hs::LoadU(hdi,a.data()+i); const auto r=hs::MaskedModOr(hs::Gt(x,hs::Zero(hdi)),hs::Set(hdi,-77),x,hdiv); nb::doNotOptimizeAway(r); } });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) { const auto x=hn::LoadU(gdi,a.data()+i); const auto q=hn::IntDiv(gdi,x,gdiv); const auto r=hn::Sub(x,hn::Mul(q,hn::Set(gdi,gdiv.divisor))); const auto masked=hn::IfThenElse(hn::Gt(x,hn::Zero(gdi)),r,hn::Set(gdi,-77)); nb::doNotOptimizeAway(masked); } });
  }
  {
    nb::Bench bench; bench.title("MulEven i32-i64").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) { const auto r=hs::MulEven(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)); nb::doNotOptimizeAway(r); } });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) { const auto r=hn::MulEven(hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)); nb::doNotOptimizeAway(r); } });
  }
  {
    nb::Bench bench; bench.title("MulOdd i32-i64").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) { const auto r=hs::MulOdd(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)); nb::doNotOptimizeAway(r); } });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) { const auto r=hn::MulOdd(hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)); nb::doNotOptimizeAway(r); } });
  }

  // unary float
  UNARY_F("Sqrt f32", hs::Sqrt, hn::Sqrt);
  UNARY_F("ApproximateReciprocal f32", hs::ApproximateReciprocal, hn::ApproximateReciprocal);
  UNARY_F("ApproximateReciprocalSqrt f32", hs::ApproximateReciprocalSqrt, hn::ApproximateReciprocalSqrt);
  // compare
  CMP_I("Eq i32", hs::Eq, hn::Eq);
  CMP_I("Lt i32", hs::Lt, hn::Lt);
  CMP_I("Gt i32", hs::Gt, hn::Gt);

  // P3 Masked* sweep. All 24 HYLE names are measured; the two HYLE-only zeroing complex
  // conveniences use the equivalent Highway IfThenElseZero composition as the baseline.
  for (std::size_t i = 0; i < kN; ++i) b2[i] = static_cast<std::int32_t>(i & 3u);
  BINARY_I("MaskedShl i32", [&](auto x,auto s){return hs::MaskedShl(hs::Gt(x,hs::Zero(hdi)),x,s);}, [&](auto x,auto s){return hn::MaskedShl(hn::Gt(x,hn::Zero(gdi)),x,s);});
  BINARY_I("MaskedShlOr i32", [&](auto x,auto s){return hs::MaskedShlOr(hs::Gt(x,hs::Zero(hdi)),hs::Set(hdi,-77),x,s);}, [&](auto x,auto s){return hn::MaskedShlOr(hn::Set(gdi,-77),hn::Gt(x,hn::Zero(gdi)),x,s);});
  BINARY_I("MaskedShr i32", [&](auto x,auto s){return hs::MaskedShr(hs::Gt(x,hs::Zero(hdi)),x,s);}, [&](auto x,auto s){return hn::MaskedShr(hn::Gt(x,hn::Zero(gdi)),x,s);});
  BINARY_I("MaskedShrOr i32", [&](auto x,auto s){return hs::MaskedShrOr(hs::Gt(x,hs::Zero(hdi)),hs::Set(hdi,-77),x,s);}, [&](auto x,auto s){return hn::MaskedShrOr(hn::Set(gdi,-77),hn::Gt(x,hn::Zero(gdi)),x,s);});
  BINARY_I("MaskedShiftLeft<3> i32", [&](auto x,auto){return hs::MaskedShiftLeft<3>(hs::Gt(x,hs::Zero(hdi)),x);}, [&](auto x,auto){return hn::MaskedShiftLeft<3>(hn::Gt(x,hn::Zero(gdi)),x);});
  BINARY_I("MaskedShiftLeftOr<3> i32", [&](auto x,auto){return hs::MaskedShiftLeftOr<3>(hs::Gt(x,hs::Zero(hdi)),hs::Set(hdi,-77),x);}, [&](auto x,auto){return hn::MaskedShiftLeftOr<3>(hn::Set(gdi,-77),hn::Gt(x,hn::Zero(gdi)),x);});
  BINARY_I("MaskedShiftRight<3> i32", [&](auto x,auto){return hs::MaskedShiftRight<3>(hs::Gt(x,hs::Zero(hdi)),x);}, [&](auto x,auto){return hn::MaskedShiftRight<3>(hn::Gt(x,hn::Zero(gdi)),x);});
  BINARY_I("MaskedShiftRightOr<3> i32", [&](auto x,auto){return hs::MaskedShiftRightOr<3>(hs::Gt(x,hs::Zero(hdi)),hs::Set(hdi,-77),x);}, [&](auto x,auto){return hn::MaskedShiftRightOr<3>(hn::Set(gdi,-77),hn::Gt(x,hn::Zero(gdi)),x);});

  const auto run_reduce = [&](const char* title, auto hop, auto gop) {
    nb::Bench bench; bench.title(title).relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw) { const auto v=hs::LoadU(hdi,a.data()+i); sink+=hop(hs::Gt(v,hs::Zero(hdi)),v); } nb::doNotOptimizeAway(sink); });
    bench.run("Highway", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+gw<=kN;i+=gw) { const auto v=hn::LoadU(gdi,a.data()+i); sink+=gop(hn::Gt(v,hn::Zero(gdi)),v); } nb::doNotOptimizeAway(sink); });
  };
  run_reduce("MaskedReduceSum i32", [&](auto m,auto v){return hs::MaskedReduceSum(hdi,m,v);}, [&](auto m,auto v){return hn::MaskedReduceSum(gdi,m,v);});
  run_reduce("MaskedReduceMin i32", [&](auto m,auto v){return hs::MaskedReduceMin(hdi,m,v);}, [&](auto m,auto v){return hn::MaskedReduceMin(gdi,m,v);});
  run_reduce("MaskedReduceMax i32", [&](auto m,auto v){return hs::MaskedReduceMax(hdi,m,v);}, [&](auto m,auto v){return hn::MaskedReduceMax(gdi,m,v);});

  UNARY_F("MaskedApproxReciprocal f32", [&](auto v){return hs::MaskedApproximateReciprocal(hs::Gt(v,hs::Set(hdf,2.0f)),v);}, [&](auto v){return hn::MaskedApproximateReciprocal(hn::Gt(v,hn::Set(gdf,2.0f)),v);});
  UNARY_F("MaskedApproxReciprocalSqrt f32", [&](auto v){return hs::MaskedApproximateReciprocalSqrt(hs::Gt(v,hs::Set(hdf,2.0f)),v);}, [&](auto v){return hn::MaskedApproximateReciprocalSqrt(hn::Gt(v,hn::Set(gdf,2.0f)),v);});
  UNARY_I("MaskedLeadingZeroCount i32", [&](auto v){return hs::MaskedLeadingZeroCount(hs::Gt(v,hs::Zero(hdi)),v);}, [&](auto v){return hn::MaskedLeadingZeroCount(hn::Gt(v,hn::Zero(gdi)),v);});

  {
    nb::Bench bench; bench.title("MaskedConvertTo f32-i32").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) { const auto v=hs::LoadU(hdf,fa.data()+i); const auto m=hs::Gt(hs::ConvertTo(hdi,v),hs::Set(hdi,2)); hs::StoreU(hs::MaskedConvertTo(m,hdi,v),hdi,o.data()+i); } nb::doNotOptimizeAway(o.data()); });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) { const auto v=hn::LoadU(gdf,fa.data()+i); const auto m=hn::Gt(hn::ConvertTo(gdi,v),hn::Set(gdi,2)); hn::StoreU(hn::MaskedConvertTo(m,gdi,v),gdi,o.data()+i); } nb::doNotOptimizeAway(o.data()); });
  }

  const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
  const std::size_t hw16=hs::Lanes(hd16), gw16=hn::Lanes(gd16);
  std::vector<std::int16_t> a16(kN*2), b16(kN*2), o16(kN*2);
  for (std::size_t i=0;i<a16.size();++i) { const int j=static_cast<int>(i); a16[i]=static_cast<std::int16_t>(j*17-12000); b16[i]=static_cast<std::int16_t>(9000-j*11); }
  {
    nb::Bench bench; bench.title("MaskedMulFixedPoint15 i16").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw16<=kN;i+=hw16) { const auto x=hs::LoadU(hd16,a16.data()+i), y=hs::LoadU(hd16,b16.data()+i); hs::StoreU(hs::MaskedMulFixedPoint15(hs::Gt(x,hs::Zero(hd16)),x,y),hd16,o16.data()+i); } nb::doNotOptimizeAway(o16.data()); });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw16<=kN;i+=gw16) { const auto x=hn::LoadU(gd16,a16.data()+i), y=hn::LoadU(gd16,b16.data()+i); hn::StoreU(hn::MaskedMulFixedPoint15(hn::Gt(x,hn::Zero(gd16)),x,y),gd16,o16.data()+i); } nb::doNotOptimizeAway(o16.data()); });
  }
  {
    nb::Bench bench; bench.title("MaskedWidenMulPairwiseAdd i16-i32").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t si=0,oi=0;si+hw16<=kN*2&&oi+hw<=kN;si+=hw16,oi+=hw) { const auto x=hs::LoadU(hd16,a16.data()+si), y=hs::LoadU(hd16,b16.data()+si); const auto ref=hs::WidenMulPairwiseAdd(hdi,x,y); hs::StoreU(hs::MaskedWidenMulPairwiseAdd(hdi,hs::Gt(ref,hs::Zero(hdi)),x,y),hdi,o.data()+oi); } nb::doNotOptimizeAway(o.data()); });
    bench.run("Highway", [&]{ for(std::size_t si=0,oi=0;si+gw16<=kN*2&&oi+gw<=kN;si+=gw16,oi+=gw) { const auto x=hn::LoadU(gd16,a16.data()+si), y=hn::LoadU(gd16,b16.data()+si); const auto ref=hn::WidenMulPairwiseAdd(gdi,x,y); hn::StoreU(hn::MaskedWidenMulPairwiseAdd(gdi,hn::Gt(ref,hn::Zero(gdi)),x,y),gdi,o.data()+oi); } nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench bench; bench.title("MaskedSatAddOr i16").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw16<=kN;i+=hw16) { const auto x=hs::LoadU(hd16,a16.data()+i), y=hs::LoadU(hd16,b16.data()+i); hs::StoreU(hs::MaskedSatAddOr(hs::Gt(x,hs::Zero(hd16)),hs::Set(hd16,-77),x,y),hd16,o16.data()+i); } nb::doNotOptimizeAway(o16.data()); });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw16<=kN;i+=gw16) { const auto x=hn::LoadU(gd16,a16.data()+i), y=hn::LoadU(gd16,b16.data()+i); hn::StoreU(hn::MaskedSatAddOr(hn::Set(gd16,-77),hn::Gt(x,hn::Zero(gd16)),x,y),gd16,o16.data()+i); } nb::doNotOptimizeAway(o16.data()); });
  }
  {
    nb::Bench bench; bench.title("MaskedSatSubOr i16").relative(true).batch(kN).minEpochIterations(8000);
    bench.run("HYLE", [&]{ for(std::size_t i=0;i+hw16<=kN;i+=hw16) { const auto x=hs::LoadU(hd16,a16.data()+i), y=hs::LoadU(hd16,b16.data()+i); hs::StoreU(hs::MaskedSatSubOr(hs::Gt(x,hs::Zero(hd16)),hs::Set(hd16,-77),x,y),hd16,o16.data()+i); } nb::doNotOptimizeAway(o16.data()); });
    bench.run("Highway", [&]{ for(std::size_t i=0;i+gw16<=kN;i+=gw16) { const auto x=hn::LoadU(gd16,a16.data()+i), y=hn::LoadU(gd16,b16.data()+i); hn::StoreU(hn::MaskedSatSubOr(hn::Set(gd16,-77),hn::Gt(x,hn::Zero(gd16)),x,y),gd16,o16.data()+i); } nb::doNotOptimizeAway(o16.data()); });
  }

  BINARY_F("MaskedMulComplex f32", [&](auto x,auto y){return hs::MaskedMulComplex(hs::Gt(x,hs::Set(hdf,2.0f)),x,y);}, [&](auto x,auto y){const auto m=hn::Gt(x,hn::Set(gdf,2.0f));return hn::IfThenElseZero(m,hn::MulComplex(x,y));});
  BINARY_F("MaskedMulComplexOr f32", [&](auto x,auto y){return hs::MaskedMulComplexOr(hs::Gt(x,hs::Set(hdf,2.0f)),hs::Set(hdf,-77.0f),x,y);}, [&](auto x,auto y){const auto m=hn::Gt(x,hn::Set(gdf,2.0f));return hn::MaskedMulComplexOr(hn::Set(gdf,-77.0f),m,x,y);});
  BINARY_F("MaskedMulComplexConj f32", [&](auto x,auto y){return hs::MaskedMulComplexConj(hs::Gt(x,hs::Set(hdf,2.0f)),x,y);}, [&](auto x,auto y){return hn::MaskedMulComplexConj(hn::Gt(x,hn::Set(gdf,2.0f)),x,y);});
  TERNARY_F("MaskedMulComplexAdd f32", [&](auto x,auto y,auto z){return hs::MaskedMulComplexAdd(hs::Gt(x,hs::Set(hdf,2.0f)),x,y,z);}, [&](auto x,auto y,auto z){const auto m=hn::Gt(x,hn::Set(gdf,2.0f));return hn::IfThenElseZero(m,hn::MulComplexAdd(x,y,z));});
  TERNARY_F("MaskedMulComplexConjAdd f32", [&](auto x,auto y,auto z){return hs::MaskedMulComplexConjAdd(hs::Gt(x,hs::Set(hdf,2.0f)),x,y,z);}, [&](auto x,auto y,auto z){return hn::MaskedMulComplexConjAdd(hn::Gt(x,hn::Set(gdf,2.0f)),x,y,z);});

  // ---- Tier1: bare complex (hot unmasked forms; only Masked* were covered above) ----
  UNARY_F("ComplexConj f32", hs::ComplexConj, hn::ComplexConj);
  BINARY_F("MulComplex f32", hs::MulComplex, hn::MulComplex);
  BINARY_F("MulComplexConj f32", hs::MulComplexConj, hn::MulComplexConj);
  TERNARY_F("MulComplexAdd f32", hs::MulComplexAdd, hn::MulComplexAdd);
  TERNARY_F("MulComplexConjAdd f32", hs::MulComplexConjAdd, hn::MulComplexConjAdd);

  // ---- Tier1: reductions (only ReduceSum + Masked* existed) ----
  {
    const auto run_full_reduce = [&](const char* title, auto hop, auto gop) {
      nb::Bench bench; bench.title(title).relative(true).batch(kN).minEpochIterations(8000);
      bench.run("HYLE", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hop(hs::LoadU(hdi,a.data()+i)); nb::doNotOptimizeAway(sink); });
      bench.run("Highway", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+gw<=kN;i+=gw) sink+=gop(hn::LoadU(gdi,a.data()+i)); nb::doNotOptimizeAway(sink); });
    };
    run_full_reduce("ReduceMin i32", [&](auto v){return hs::ReduceMin(hdi,v);}, [&](auto v){return hn::ReduceMin(gdi,v);});
    run_full_reduce("ReduceMax i32", [&](auto v){return hs::ReduceMax(hdi,v);}, [&](auto v){return hn::ReduceMax(gdi,v);});
    // ReduceMul/And/Or: Highway has no same-name scalar reduce (only Sum/Min/Max). Tier3 vs scalar fold.
    {
      nb::Bench b; b.title("ReduceMul i32 (vs scalar fold)").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::ReduceMul(hdi,hs::LoadU(hdi,a.data()+i)); nb::doNotOptimizeAway(sink); });
      b.run("scalar fold", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw){ std::int32_t p=1; for(std::size_t j=0;j<hw;++j) p*=a[i+j]; sink+=p; } nb::doNotOptimizeAway(sink); });
    }
    {
      nb::Bench b; b.title("ReduceAnd i32 (vs scalar fold)").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::ReduceAnd(hdi,hs::LoadU(hdi,a.data()+i)); nb::doNotOptimizeAway(sink); });
      b.run("scalar fold", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw){ std::int32_t p=~0; for(std::size_t j=0;j<hw;++j) p&=a[i+j]; sink+=p; } nb::doNotOptimizeAway(sink); });
    }
    {
      nb::Bench b; b.title("ReduceOr i32 (vs scalar fold)").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::ReduceOr(hdi,hs::LoadU(hdi,a.data()+i)); nb::doNotOptimizeAway(sink); });
      b.run("scalar fold", [&]{ std::int64_t sink=0; for(std::size_t i=0;i+hw<=kN;i+=hw){ std::int32_t p=0; for(std::size_t j=0;j<hw;++j) p|=a[i+j]; sink+=p; } nb::doNotOptimizeAway(sink); });
    }
  }

  // ---- Tier1: 128-bit pair cmp/min/max (u64 pairs; HYLE synthesizes from Reverse2/DupOdd) ----
  {
    const hs::ScalableTag<std::uint64_t> hdu; const hn::ScalableTag<std::uint64_t> gdu;
    const std::size_t hwu=hs::Lanes(hdu), gwu=hn::Lanes(gdu);
    std::vector<std::uint64_t> ua(kN), ub(kN), uo(kN);
    { std::mt19937_64 r(41); for(auto&e:ua)e=r(); for(auto&e:ub)e=r(); }
    auto cmp128 = [&](const char* title, auto hop, auto gop) {
      nb::Bench b; b.title(title).relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hwu<=kN;i+=hwu) hs::StoreU(hs::VecFromMask(hdu,hop(hs::LoadU(hdu,ua.data()+i),hs::LoadU(hdu,ub.data()+i))),hdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gwu<=kN;i+=gwu) hn::StoreU(hn::VecFromMask(gdu,gop(hn::LoadU(gdu,ua.data()+i),hn::LoadU(gdu,ub.data()+i))),gdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    };
    cmp128("Lt128 u64", [&](auto x,auto y){return hs::Lt128(hdu,x,y);}, [&](auto x,auto y){return hn::Lt128(gdu,x,y);});
    cmp128("Lt128Upper u64", [&](auto x,auto y){return hs::Lt128Upper(hdu,x,y);}, [&](auto x,auto y){return hn::Lt128Upper(gdu,x,y);});
    cmp128("Eq128 u64", [&](auto x,auto y){return hs::Eq128(hdu,x,y);}, [&](auto x,auto y){return hn::Eq128(gdu,x,y);});
    cmp128("Ne128 u64", [&](auto x,auto y){return hs::Ne128(hdu,x,y);}, [&](auto x,auto y){return hn::Ne128(gdu,x,y);});
    {
      nb::Bench b; b.title("Min128 u64").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hwu<=kN;i+=hwu) hs::StoreU(hs::Min128(hdu,hs::LoadU(hdu,ua.data()+i),hs::LoadU(hdu,ub.data()+i)),hdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gwu<=kN;i+=gwu) hn::StoreU(hn::Min128(gdu,hn::LoadU(gdu,ua.data()+i),hn::LoadU(gdu,ub.data()+i)),gdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    }
    {
      nb::Bench b; b.title("Max128 u64").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hwu<=kN;i+=hwu) hs::StoreU(hs::Max128(hdu,hs::LoadU(hdu,ua.data()+i),hs::LoadU(hdu,ub.data()+i)),hdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gwu<=kN;i+=gwu) hn::StoreU(hn::Max128(gdu,hn::LoadU(gdu,ua.data()+i),hn::LoadU(gdu,ub.data()+i)),gdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    }
  }

  // ---- Tier2: multi-insn but simple, vendor same-op ----
  UNARY_I("SaturatedAbs i32", hs::SaturatedAbs, hn::SaturatedAbs);
  UNARY_I("SaturatedNeg i32", hs::SaturatedNeg, hn::SaturatedNeg);
  UNARY_I("Reverse2 i32", [&](auto v){return hs::Reverse2(hdi,v);}, [&](auto v){return hn::Reverse2(gdi,v);});
  UNARY_I("Reverse4 i32", [&](auto v){return hs::Reverse4(hdi,v);}, [&](auto v){return hn::Reverse4(gdi,v);});
  UNARY_I("Slide1Up i32", [&](auto v){return hs::Slide1Up(hdi,v);}, [&](auto v){return hn::Slide1Up(gdi,v);});
  UNARY_I("Slide1Down i32", [&](auto v){return hs::Slide1Down(hdi,v);}, [&](auto v){return hn::Slide1Down(gdi,v);});
  UNARY_I("RotateLeft<3> i32", hs::RotateLeft<3>, hn::RotateLeft<3>);
  UNARY_I("RotateRight<3> i32", hs::RotateRight<3>, hn::RotateRight<3>);
  UNARY_I("Broadcast<0> i32", hs::Broadcast<0>, hn::Broadcast<0>);
  BINARY_I("AddSub i32", hs::AddSub, hn::AddSub);
  BINARY_I("Clamp i32", [&](auto x,auto y){return hs::Clamp(x,hs::Min(x,y),hs::Max(x,y));}, [&](auto x,auto y){return hn::Clamp(x,hn::Min(x,y),hn::Max(x,y));});
  BINARY_I("RoundingShr i32", hs::RoundingShr, hn::RoundingShr);
  {
    const hs::ScalableTag<std::uint16_t> hdu16; const hn::ScalableTag<std::uint16_t> gdu16;
    const std::size_t hwu16=hs::Lanes(hdu16), gwu16=hn::Lanes(gdu16);
    std::vector<std::uint16_t> u16a(kN), u16b(kN), u16o(kN);
    { std::mt19937_64 r(51); std::uniform_int_distribution<int> d(0,65535); for(auto&e:u16a)e=(std::uint16_t)d(r); for(auto&e:u16b)e=(std::uint16_t)d(r); }
    nb::Bench b; b.title("AverageRound u16").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwu16<=kN;i+=hwu16) hs::StoreU(hs::AverageRound(hs::LoadU(hdu16,u16a.data()+i),hs::LoadU(hdu16,u16b.data()+i)),hdu16,u16o.data()+i); nb::doNotOptimizeAway(u16o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwu16<=kN;i+=gwu16) hn::StoreU(hn::AverageRound(hn::LoadU(gdu16,u16a.data()+i),hn::LoadU(gdu16,u16b.data()+i)),gdu16,u16o.data()+i); nb::doNotOptimizeAway(u16o.data()); });
  }
  {
    // MulByPow2 / MulByFloorPow2: float * 2^exp
    std::vector<std::int32_t> exp(kN);
    for (std::size_t i=0;i<kN;++i) exp[i] = static_cast<std::int32_t>((i % 7) - 3);
    nb::Bench b; b.title("MulByPow2 f32").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::MulByPow2(hs::LoadU(hdf,fa.data()+i),hs::LoadU(hdi,exp.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::MulByPow2(hn::LoadU(gdf,fa.data()+i),hn::LoadU(gdi,exp.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }
  {
    nb::Bench b; b.title("MulByFloorPow2 f32").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::MulByFloorPow2(hs::LoadU(hdf,fa.data()+i),hs::LoadU(hdf,fb.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::MulByFloorPow2(hn::LoadU(gdf,fa.data()+i),hn::LoadU(gdf,fb.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }
  {
    nb::Bench b; b.title("GetBiasedExponent f32").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) { auto r=hs::GetBiasedExponent(hs::LoadU(hdf,fa.data()+i)); nb::doNotOptimizeAway(r); } });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) { auto r=hn::GetBiasedExponent(hn::LoadU(gdf,fa.data()+i)); nb::doNotOptimizeAway(r); } });
  }

  // ---- Tier3: HYLE-only (no vendor same op) — vs scalar Div / raw estimate only ----
  {
    nb::Bench b; b.title("ReciprocalNewton f32 (vs Div / ApproxRecip)").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE ReciprocalNewton", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::ReciprocalNewton(hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("HYLE ApproximateReciprocal", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::ApproximateReciprocal(hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("HYLE Div(1,x)", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::Div(hs::Set(hdf,1.0f),hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }
  {
    nb::Bench b; b.title("ReciprocalSqrtNewton f32 (vs Sqrt-Div / ApproxRSqrt)").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE ReciprocalSqrtNewton", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::ReciprocalSqrtNewton(hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("HYLE ApproximateReciprocalSqrt", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::ApproximateReciprocalSqrt(hs::LoadU(hdf,fa.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("HYLE Div(1,Sqrt)", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::Div(hs::Set(hdf,1.0f),hs::Sqrt(hs::LoadU(hdf,fa.data()+i))),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }

  // ---- Low-priority Tier2 rest: Shuffle*/Per4Lane/Pairwise/SlideLanes/MulRound/CombineShift ----
  UNARY_I("Shuffle0123 i32", hs::Shuffle0123, hn::Shuffle0123);
  UNARY_I("Shuffle1032 i32", hs::Shuffle1032, hn::Shuffle1032);
  UNARY_I("Shuffle2301 i32", hs::Shuffle2301, hn::Shuffle2301);
  UNARY_I("Shuffle0321 i32", hs::Shuffle0321, hn::Shuffle0321);
  UNARY_I("Shuffle2103 i32", hs::Shuffle2103, hn::Shuffle2103);
  // Macro-hostile template commas: write explicit benches (UNARY_I splits on ',').
  {
    nb::Bench b; b.title("Per4LaneBlockShuffle 3210 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::Per4LaneBlockShuffle<3,2,1,0>(hs::LoadU(hdi,a.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::Per4LaneBlockShuffle<3,2,1,0>(hn::LoadU(gdi,a.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("PairwiseAdd i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::PairwiseAdd(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::PairwiseAdd(gdi,hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("PairwiseSub i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::PairwiseSub(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::PairwiseSub(gdi,hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("PairwiseAdd128 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::PairwiseAdd128(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::PairwiseAdd128(gdi,hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("PairwiseSub128 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::PairwiseSub128(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::PairwiseSub128(gdi,hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("SlideUpLanes amt=2 i32").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::SlideUpLanes(hdi,hs::LoadU(hdi,a.data()+i),2),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::SlideUpLanes(gdi,hn::LoadU(gdi,a.data()+i),2),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("SlideDownLanes amt=2 i32").relative(true).batch(kN).minEpochIterations(5000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::SlideDownLanes(hdi,hs::LoadU(hdi,a.data()+i),2),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::SlideDownLanes(gdi,hn::LoadU(gdi,a.data()+i),2),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  BINARY_F("MulRound f32", hs::MulRound, hn::MulRound);
  {
    nb::Bench b; b.title("CombineShiftRightBytes4 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::CombineShiftRightBytes<4>(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::CombineShiftRightBytes<4>(gdi,hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("CombineShiftRightLanes1 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::CombineShiftRightLanes<1>(hs::LoadU(hdi,a.data()+i),hs::LoadU(hdi,b2.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::CombineShiftRightLanes<1>(gdi,hn::LoadU(gdi,a.data()+i),hn::LoadU(gdi,b2.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  // TruncateTo u32->u16 (same lane count partial)
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    const hs::Simd<std::uint16_t, hs::ScalableTag<std::uint32_t>::kMaxLanes, 0> hdn;
    const hn::Rebind<std::uint16_t, hn::ScalableTag<std::uint32_t>> gdn;
    const std::size_t hwu=hs::Lanes(hdu), gwu=hn::Lanes(gdu);
    std::vector<std::uint32_t> ua(kN); std::vector<std::uint16_t> uo(kN);
    { std::mt19937_64 r(52); for(auto&e:ua)e=static_cast<std::uint32_t>(r()); }
    nb::Bench b; b.title("TruncateTo u32->u16").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{
      for(std::size_t i=0;i+hwu<=kN;i+=hwu) {
        auto r = hs::TruncateTo(hdn, hs::LoadU(hdu, ua.data()+i));
        hs::StoreU(r, hs::DFromV<decltype(r)>{}, uo.data()+i);
      }
      nb::doNotOptimizeAway(uo.data());
    });
    b.run("Highway", [&]{
      for(std::size_t i=0;i+gwu<=kN;i+=gwu) {
        auto r = hn::TruncateTo(gdn, hn::LoadU(gdu, ua.data()+i));
        hn::StoreU(r, hn::DFromV<decltype(r)>{}, uo.data()+i);
      }
      nb::doNotOptimizeAway(uo.data());
    });
  }

  // ---- 全量直通单指令（禁止抽样：用户要求每 op 都有 parity 行）----
  // 已在 op_parity_bench/full 出现的：Add/MulAdd/Min(i32)/Max(i32)/Eq/Lt/Gt/Sub/Mul/Abs/Neg
  // 此处补齐此前"跳过"清单的全部剩余项。
  BINARY_I("And i32", hs::And, hn::And);
  BINARY_I("Or i32", hs::Or, hn::Or);
  BINARY_I("Xor i32", hs::Xor, hn::Xor);
  BINARY_I("AndNot i32", hs::AndNot, hn::AndNot);
  UNARY_I("Not i32", hs::Not, hn::Not);
  BINARY_F("Min f32", hs::Min, hn::Min);
  BINARY_F("Max f32", hs::Max, hn::Max);
  BINARY_F("Div f32", hs::Div, hn::Div);
  BINARY_F("CopySign f32", hs::CopySign, hn::CopySign);
  UNARY_F("Floor f32", hs::Floor, hn::Floor);
  UNARY_F("Ceil f32", hs::Ceil, hn::Ceil);
  UNARY_F("Round f32", hs::Round, hn::Round);
  UNARY_F("Trunc f32", hs::Trunc, hn::Trunc);
  CMP_I("Ne i32", hs::Ne, hn::Ne);
  CMP_I("Le i32", hs::Le, hn::Le);
  CMP_I("Ge i32", hs::Ge, hn::Ge);
  // IfThenElse / IfThenElseZero / IfThenZeroElse
  BINARY_I("IfThenElse i32",
           [&](auto x,auto y){return hs::IfThenElse(hs::Gt(x,hs::Zero(hdi)),x,y);},
           [&](auto x,auto y){return hn::IfThenElse(hn::Gt(x,hn::Zero(gdi)),x,y);});
  UNARY_I("IfThenElseZero i32",
          [&](auto v){return hs::IfThenElseZero(hs::Gt(v,hs::Zero(hdi)),v);},
          [&](auto v){return hn::IfThenElseZero(hn::Gt(v,hn::Zero(gdi)),v);});
  UNARY_I("IfThenZeroElse i32",
          [&](auto v){return hs::IfThenZeroElse(hs::Gt(v,hs::Zero(hdi)),v);},
          [&](auto v){return hn::IfThenZeroElse(hn::Gt(v,hn::Zero(gdi)),v);});
  // ShiftLeftSame / ShiftRightSame (runtime shift amount)
  {
    volatile int bits = 3;
    nb::Bench b; b.title("ShiftLeftSame i32 bits=3").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::ShiftLeftSame(hs::LoadU(hdi,a.data()+i),bits),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::ShiftLeftSame(hn::LoadU(gdi,a.data()+i),bits),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    volatile int bits = 3;
    nb::Bench b; b.title("ShiftRightSame i32 bits=3").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::ShiftRightSame(hs::LoadU(hdi,a.data()+i),bits),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::ShiftRightSame(hn::LoadU(gdi,a.data()+i),bits),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  // RoundToInt / NearestInt f32->i32
  {
    nb::Bench b; b.title("RoundToInt f32->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::RoundToInt(hdi,hs::LoadU(hdf,fa.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::NearestInt(hn::LoadU(gdf,fa.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  // ConvertTo f32<->i32
  {
    nb::Bench b; b.title("ConvertTo f32->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::ConvertTo(hdi,hs::LoadU(hdf,fa.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::ConvertTo(gdi,hn::LoadU(gdf,fa.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("ConvertTo i32->f32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::ConvertTo(hdf,hs::LoadU(hdi,a.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::ConvertTo(gdf,hn::LoadU(gdi,a.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }
  // BitCast i32 <-> f32
  {
    nb::Bench b; b.title("BitCast i32->f32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::BitCast(hdf,hs::LoadU(hdi,a.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::BitCast(gdf,hn::LoadU(gdi,a.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }
  {
    nb::Bench b; b.title("BitCast f32->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::BitCast(hdi,hs::LoadU(hdf,fa.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::BitCast(gdi,hn::LoadU(gdf,fa.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  // Set / Zero / Load / Store (construction + memory)
  {
    nb::Bench b; b.title("Set i32").relative(true).batch(kN).minEpochIterations(10000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::Set(hdi,static_cast<std::int32_t>(i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::Set(gdi,static_cast<std::int32_t>(i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("Zero i32").relative(true).batch(kN).minEpochIterations(10000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::Zero(hdi),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::Zero(gdi),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("LoadU/StoreU i32").relative(true).batch(kN).minEpochIterations(10000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::LoadU(hdi,a.data()+i),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::LoadU(gdi,a.data()+i),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    // Aligned Load/Store requires 32-byte alignment; std::vector is not guaranteed.
    alignas(64) std::int32_t aa[4096], oo[4096];
    for (std::size_t i=0;i<kN;++i) aa[i]=a[i];
    nb::Bench b; b.title("Load/Store i32 aligned").relative(true).batch(kN).minEpochIterations(10000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::Store(hs::Load(hdi,aa+i),hdi,oo+i); nb::doNotOptimizeAway(oo); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::Store(hn::Load(gdi,aa+i),gdi,oo+i); nb::doNotOptimizeAway(oo); });
  }
  // Mask bitwise And/Or/Xor/AndNot/Not (sink via BitsFromMask)
  {
    std::uint64_t sink = 0;
    nb::Bench b; b.title("Mask And i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{
      for(std::size_t i=0;i+hw<=kN;i+=hw) {
        auto m0=hs::Gt(hs::LoadU(hdi,a.data()+i),hs::Zero(hdi));
        auto m1=hs::Gt(hs::LoadU(hdi,b2.data()+i),hs::Zero(hdi));
        sink ^= hs::BitsFromMask(hdi, hs::And(m0,m1));
      }
    });
    b.run("Highway", [&]{
      for(std::size_t i=0;i+gw<=kN;i+=gw) {
        auto m0=hn::Gt(hn::LoadU(gdi,a.data()+i),hn::Zero(gdi));
        auto m1=hn::Gt(hn::LoadU(gdi,b2.data()+i),hn::Zero(gdi));
        sink ^= hn::BitsFromMask(gdi, hn::And(m0,m1));
      }
    });
    nb::doNotOptimizeAway(sink);
  }
  {
    std::uint64_t sink = 0;
    nb::Bench b; b.title("Mask Or i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{
      for(std::size_t i=0;i+hw<=kN;i+=hw) {
        auto m0=hs::Gt(hs::LoadU(hdi,a.data()+i),hs::Zero(hdi));
        auto m1=hs::Gt(hs::LoadU(hdi,b2.data()+i),hs::Zero(hdi));
        sink ^= hs::BitsFromMask(hdi, hs::Or(m0,m1));
      }
    });
    b.run("Highway", [&]{
      for(std::size_t i=0;i+gw<=kN;i+=gw) {
        auto m0=hn::Gt(hn::LoadU(gdi,a.data()+i),hn::Zero(gdi));
        auto m1=hn::Gt(hn::LoadU(gdi,b2.data()+i),hn::Zero(gdi));
        sink ^= hn::BitsFromMask(gdi, hn::Or(m0,m1));
      }
    });
    nb::doNotOptimizeAway(sink);
  }
  {
    std::uint64_t sink = 0;
    nb::Bench b; b.title("Mask Xor i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{
      for(std::size_t i=0;i+hw<=kN;i+=hw) {
        auto m0=hs::Gt(hs::LoadU(hdi,a.data()+i),hs::Zero(hdi));
        auto m1=hs::Gt(hs::LoadU(hdi,b2.data()+i),hs::Zero(hdi));
        sink ^= hs::BitsFromMask(hdi, hs::Xor(m0,m1));
      }
    });
    b.run("Highway", [&]{
      for(std::size_t i=0;i+gw<=kN;i+=gw) {
        auto m0=hn::Gt(hn::LoadU(gdi,a.data()+i),hn::Zero(gdi));
        auto m1=hn::Gt(hn::LoadU(gdi,b2.data()+i),hn::Zero(gdi));
        sink ^= hn::BitsFromMask(gdi, hn::Xor(m0,m1));
      }
    });
    nb::doNotOptimizeAway(sink);
  }
  {
    std::uint64_t sink = 0;
    nb::Bench b; b.title("Mask AndNot i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{
      for(std::size_t i=0;i+hw<=kN;i+=hw) {
        auto m0=hs::Gt(hs::LoadU(hdi,a.data()+i),hs::Zero(hdi));
        auto m1=hs::Gt(hs::LoadU(hdi,b2.data()+i),hs::Zero(hdi));
        sink ^= hs::BitsFromMask(hdi, hs::AndNot(m0,m1));
      }
    });
    b.run("Highway", [&]{
      for(std::size_t i=0;i+gw<=kN;i+=gw) {
        auto m0=hn::Gt(hn::LoadU(gdi,a.data()+i),hn::Zero(gdi));
        auto m1=hn::Gt(hn::LoadU(gdi,b2.data()+i),hn::Zero(gdi));
        sink ^= hn::BitsFromMask(gdi, hn::AndNot(m0,m1));
      }
    });
    nb::doNotOptimizeAway(sink);
  }
  {
    std::uint64_t sink = 0;
    nb::Bench b; b.title("Mask Not i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{
      for(std::size_t i=0;i+hw<=kN;i+=hw) {
        auto m0=hs::Gt(hs::LoadU(hdi,a.data()+i),hs::Zero(hdi));
        sink ^= hs::BitsFromMask(hdi, hs::Not(m0));
      }
    });
    b.run("Highway", [&]{
      for(std::size_t i=0;i+gw<=kN;i+=gw) {
        auto m0=hn::Gt(hn::LoadU(gdi,a.data()+i),hn::Zero(gdi));
        sink ^= hn::BitsFromMask(gdi, hn::Not(m0));
      }
    });
    nb::doNotOptimizeAway(sink);
  }
  // Add f32 (also in op_parity_bench; keep here so full sweep is self-contained)
  BINARY_F("Add f32", hs::Add, hn::Add);
  BINARY_F("Sub f32", hs::Sub, hn::Sub);
  BINARY_F("Mul f32", hs::Mul, hn::Mul);

  // ---- FMA 变体族 (AI 热路径; 全走 vfmadd/vfmsub 系列, 应 MUST-tie) ----
  TERNARY_F("MulSub f32", hs::MulSub, hn::MulSub);              // a*b - c
  TERNARY_F("NegMulAdd f32", hs::NegMulAdd, hn::NegMulAdd);     // -(a*b) + c
  TERNARY_F("NegMulSub f32", hs::NegMulSub, hn::NegMulSub);     // -(a*b) - c
  TERNARY_F("MulAddSub f32", hs::MulAddSub, hn::MulAddSub);     // a*b -/+ c (even/odd)
  TERNARY_F("MulSubAdd f32", hs::MulSubAdd, hn::MulSubAdd);     // a*b +/- c (even/odd)

  // ---- round-to-int 族 (fused round + convert) ----
  {
    nb::Bench b; b.title("NearestInt f32->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::NearestInt(hs::LoadU(hdf,fa.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::NearestInt(hn::LoadU(gdf,fa.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("CeilInt f32->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::CeilInt(hs::LoadU(hdf,fa.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::CeilInt(hn::LoadU(gdf,fa.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("FloorInt f32->i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::FloorInt(hs::LoadU(hdf,fa.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::FloorInt(hn::LoadU(gdf,fa.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }

  // ---- NaN-aware min/max + magnitude (IEEE minNum/maxNum + |·|-compare) ----
  BINARY_F("MinNumber f32", hs::MinNumber, hn::MinNumber);
  BINARY_F("MaxNumber f32", hs::MaxNumber, hn::MaxNumber);
  BINARY_F("MinMagnitude f32", hs::MinMagnitude, hn::MinMagnitude);
  BINARY_F("MaxMagnitude f32", hs::MaxMagnitude, hn::MaxMagnitude);

  // ---- IfNegative 族 (sign-bit select, 无分支) ----
  {
    nb::Bench b; b.title("IfNegativeThenElse f32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) hs::StoreU(hs::IfNegativeThenElse(hs::LoadU(hdf,fa.data()+i),hs::LoadU(hdf,fb.data()+i),hs::LoadU(hdf,fc.data()+i)),hdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) hn::StoreU(hn::IfNegativeThenElse(hn::LoadU(gdf,fa.data()+i),hn::LoadU(gdf,fb.data()+i),hn::LoadU(gdf,fc.data()+i)),gdf,fo.data()+i); nb::doNotOptimizeAway(fo.data()); });
  }
  UNARY_F("IfNegativeThenElseZero f32", [&](auto v){return hs::IfNegativeThenElseZero(v,hs::Set(hdf,3.0f));}, [&](auto v){return hn::IfNegativeThenElseZero(v,hn::Set(gdf,3.0f));});
  UNARY_F("IfNegativeThenZeroElse f32", [&](auto v){return hs::IfNegativeThenZeroElse(v,hs::Set(hdf,3.0f));}, [&](auto v){return hn::IfNegativeThenZeroElse(v,hn::Set(gdf,3.0f));});
  UNARY_F("ZeroIfNegative f32", hs::ZeroIfNegative, hn::ZeroIfNegative);

  // ---- 谓词/掩码生成 (IsNaN/IsInf/IsFinite/IsNegative; sink mask bits) ----
  {
    auto fspecial = fa; fspecial[0]=std::numeric_limits<float>::quiet_NaN(); fspecial[1]=std::numeric_limits<float>::infinity();
    std::uint64_t sink=0;
    auto pred=[&](const char* t, auto hop, auto gop){
      nb::Bench b; b.title(t).relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) sink^=hs::BitsFromMask(hdf,hop(hs::LoadU(hdf,fspecial.data()+i))); });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) sink^=hn::BitsFromMask(gdf,gop(hn::LoadU(gdf,fspecial.data()+i))); });
    };
    pred("IsNaN f32", [&](auto v){return hs::IsNaN(v);}, [&](auto v){return hn::IsNaN(v);});
    pred("IsInf f32", [&](auto v){return hs::IsInf(v);}, [&](auto v){return hn::IsInf(v);});
    pred("IsFinite f32", [&](auto v){return hs::IsFinite(v);}, [&](auto v){return hn::IsFinite(v);});
    nb::doNotOptimizeAway(sink);
  }
  {
    std::uint64_t sink=0;
    nb::Bench b; b.title("IsNegative i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) sink^=hs::BitsFromMask(hdi,hs::IsNegative(hs::LoadU(hdi,a.data()+i))); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) sink^=hn::BitsFromMask(gdi,hn::IsNegative(hn::LoadU(gdi,a.data()+i))); });
    nb::doNotOptimizeAway(sink);
  }

  // ---- mask reductions/predicates (AllTrue/AllFalse/CountTrue/FindFirst/LastTrue) ----
  {
    const auto im = geni(60);
    std::uint64_t sink=0;
    auto mred=[&](const char* t, auto hop, auto gop){
      nb::Bench b; b.title(t).relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,im.data()+i),hs::Zero(hdi)); sink+=(std::uint64_t)hop(hdi,m);} });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,im.data()+i),hn::Zero(gdi)); sink+=(std::uint64_t)gop(gdi,m);} });
    };
    mred("CountTrue i32", [&](auto d,auto m){return hs::CountTrue(d,m);}, [&](auto d,auto m){return hn::CountTrue(d,m);});
    mred("AllTrue i32", [&](auto d,auto m){return (std::size_t)hs::AllTrue(d,m);}, [&](auto d,auto m){return (std::size_t)hn::AllTrue(d,m);});
    mred("AllFalse i32", [&](auto d,auto m){return (std::size_t)hs::AllFalse(d,m);}, [&](auto d,auto m){return (std::size_t)hn::AllFalse(d,m);});
    mred("FindFirstTrue i32", [&](auto d,auto m){return (std::size_t)(hs::FindFirstTrue(d,m)+1);}, [&](auto d,auto m){return (std::size_t)(hn::FindFirstTrue(d,m)+1);});
    mred("FindLastTrue i32", [&](auto d,auto m){return (std::size_t)(hs::FindLastTrue(d,m)+1);}, [&](auto d,auto m){return (std::size_t)(hn::FindLastTrue(d,m)+1);});
    nb::doNotOptimizeAway(sink);
  }

  // ---- InterleaveEven/Odd + InterleaveWholeLower/Upper (Highway 带 d tag) ----
  BINARY_I("InterleaveEven i32", [&](auto x,auto y){return hs::InterleaveEven(x,y);}, [&](auto x,auto y){return hn::InterleaveEven(gdi,x,y);});
  BINARY_I("InterleaveOdd i32", [&](auto x,auto y){return hs::InterleaveOdd(x,y);}, [&](auto x,auto y){return hn::InterleaveOdd(gdi,x,y);});
  BINARY_I("InterleaveWholeLower i32", [&](auto x,auto y){return hs::InterleaveWholeLower(hdi,x,y);}, [&](auto x,auto y){return hn::InterleaveWholeLower(gdi,x,y);});
  BINARY_I("InterleaveWholeUpper i32", [&](auto x,auto y){return hs::InterleaveWholeUpper(hdi,x,y);}, [&](auto x,auto y){return hn::InterleaveWholeUpper(gdi,x,y);});

  // ---- Rotate*Same (runtime bit count) ----
  {
    volatile int bits=5;
    nb::Bench b; b.title("RotateLeftSame u32 bits=5").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ const hs::ScalableTag<std::uint32_t> hdu; for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::RotateLeftSame(hs::BitCast(hdu,hs::LoadU(hdi,a.data()+i)),bits),hdu,reinterpret_cast<std::uint32_t*>(o.data())+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ const hn::ScalableTag<std::uint32_t> gdu; for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::RotateLeftSame(hn::BitCast(gdu,hn::LoadU(gdi,a.data()+i)),bits),gdu,reinterpret_cast<std::uint32_t*>(o.data())+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    volatile int bits=5;
    nb::Bench b; b.title("RotateRightSame u32 bits=5").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ const hs::ScalableTag<std::uint32_t> hdu; for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::RotateRightSame(hs::BitCast(hdu,hs::LoadU(hdi,a.data()+i)),bits),hdu,reinterpret_cast<std::uint32_t*>(o.data())+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ const hn::ScalableTag<std::uint32_t> gdu; for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::RotateRightSame(hn::BitCast(gdu,hn::LoadU(gdi,a.data()+i)),bits),gdu,reinterpret_cast<std::uint32_t*>(o.data())+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    volatile int bits=5;
    nb::Bench b; b.title("RoundingShiftRightSame i32 bits=5").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::RoundingShiftRightSame(hs::LoadU(hdi,a.data()+i),bits),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::RoundingShiftRightSame(hn::LoadU(gdi,a.data()+i),bits),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  {
    nb::Bench b; b.title("RoundingShiftRight<2> i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::RoundingShiftRight<2>(hs::LoadU(hdi,a.data()+i)),hdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::RoundingShiftRight<2>(hn::LoadU(gdi,a.data()+i)),gdi,o.data()+i); nb::doNotOptimizeAway(o.data()); });
  }
  // ---- SaturatedSub i16 (native vpsubsw) ----
  {
    const hs::ScalableTag<std::int16_t> hd16; const hn::ScalableTag<std::int16_t> gd16;
    const std::size_t hw16=hs::Lanes(hd16), gw16=hn::Lanes(gd16);
    std::vector<std::int16_t> s16a(kN), s16b(kN), s16o(kN);
    { std::mt19937_64 r(61); std::uniform_int_distribution<int> dd(-30000,30000); for(auto&e:s16a)e=(std::int16_t)dd(r); for(auto&e:s16b)e=(std::int16_t)dd(r); }
    nb::Bench b; b.title("SaturatedSub i16").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw16<=kN;i+=hw16) hs::StoreU(hs::SaturatedSub(hs::LoadU(hd16,s16a.data()+i),hs::LoadU(hd16,s16b.data()+i)),hd16,s16o.data()+i); nb::doNotOptimizeAway(s16o.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw16<=kN;i+=gw16) hn::StoreU(hn::SaturatedSub(hn::LoadU(gd16,s16a.data()+i),hn::LoadU(gd16,s16b.data()+i)),gd16,s16o.data()+i); nb::doNotOptimizeAway(s16o.data()); });
  }

  // ---- SumOfLanes/MinOfLanes/MaxOfLanes (broadcast reduce), Shr(var), TestBit, BroadcastLane ----
  UNARY_I("SumOfLanes i32", [&](auto v){return hs::SumOfLanes(hdi,v);}, [&](auto v){return hn::SumOfLanes(gdi,v);});
  UNARY_I("MinOfLanes i32", [&](auto v){return hs::MinOfLanes(hdi,v);}, [&](auto v){return hn::MinOfLanes(gdi,v);});
  UNARY_I("MaxOfLanes i32", [&](auto v){return hs::MaxOfLanes(hdi,v);}, [&](auto v){return hn::MaxOfLanes(gdi,v);});
  UNARY_I("BroadcastLane<3> i32", hs::BroadcastLane<3>, hn::BroadcastLane<3>);
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    std::vector<std::uint32_t> ua(kN), ub(kN), uo(kN);
    { std::mt19937_64 r(62); for(auto&e:ua)e=(std::uint32_t)r(); for(auto&e:ub)e=r()&31u; }
    nb::Bench b; b.title("Shr u32 (variable)").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::Shr(hs::LoadU(hdu,ua.data()+i),hs::LoadU(hdu,ub.data()+i)),hdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::Shr(hn::LoadU(gdu,ua.data()+i),hn::LoadU(gdu,ub.data()+i)),gdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
  }
  {
    std::uint64_t sink=0;
    nb::Bench b; b.title("TestBit i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) sink^=hs::BitsFromMask(hdi,hs::TestBit(hs::LoadU(hdi,a.data()+i),hs::Set(hdi,4))); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) sink^=hn::BitsFromMask(gdi,hn::TestBit(hn::LoadU(gdi,a.data()+i),hn::Set(gdi,4))); });
    nb::doNotOptimizeAway(sink);
  }
  // ---- AllBits0/1, HighestSetBitIndex, MinMagnitude (bit-level utilities) ----
  {
    std::uint64_t sink=0;
    nb::Bench b; b.title("AllBits1 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::AllBits1(hs::LoadU(hdi,a.data()+i)); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto v=hn::LoadU(gdi,a.data()+i); sink+=hn::AllTrue(gdi,hn::Eq(v,hn::Set(gdi,-1)));} });
    nb::doNotOptimizeAway(sink);
  }
  {
    const hs::ScalableTag<std::uint32_t> hdu; const hn::ScalableTag<std::uint32_t> gdu;
    std::vector<std::uint32_t> ua(kN), uo(kN);
    { std::mt19937_64 r(63); for(auto&e:ua)e=(std::uint32_t)r(); }
    nb::Bench b; b.title("HighestSetBitIndex u32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) hs::StoreU(hs::HighestSetBitIndex(hs::LoadU(hdu,ua.data()+i)),hdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) hn::StoreU(hn::HighestSetBitIndex(hn::LoadU(gdu,ua.data()+i)),gdu,uo.data()+i); nb::doNotOptimizeAway(uo.data()); });
  }
  // ---- mask 生成谓词 SetOnlyFirst/SetBeforeFirst/SetAtOrBeforeFirst/SetAtOrAfterFirst/ExclusiveNeither/IsEitherNaN ----
  {
    const auto im=geni(64); std::uint64_t sink=0;
    auto mgen=[&](const char* t, auto hop, auto gop){
      nb::Bench b; b.title(t).relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw){ auto m=hs::Gt(hs::LoadU(hdi,im.data()+i),hs::Zero(hdi)); sink^=hs::BitsFromMask(hdi,hop(m));} });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto m=hn::Gt(hn::LoadU(gdi,im.data()+i),hn::Zero(gdi)); sink^=hn::BitsFromMask(gdi,gop(m));} });
    };
    mgen("SetOnlyFirst i32", [&](auto m){return hs::SetOnlyFirst(m);}, [&](auto m){return hn::SetOnlyFirst(m);});
    mgen("SetBeforeFirst i32", [&](auto m){return hs::SetBeforeFirst(m);}, [&](auto m){return hn::SetBeforeFirst(m);});
    mgen("SetAtOrBeforeFirst i32", [&](auto m){return hs::SetAtOrBeforeFirst(m);}, [&](auto m){return hn::SetAtOrBeforeFirst(m);});
    mgen("SetAtOrAfterFirst i32", [&](auto m){return hs::SetAtOrAfterFirst(m);}, [&](auto m){return hn::SetAtOrAfterFirst(m);});
    nb::doNotOptimizeAway(sink);
  }
  {
    const auto im=geni(65), im2=geni(66); std::uint64_t sink=0;
    nb::Bench b; b.title("ExclusiveNeither i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw){ auto a0=hs::Gt(hs::LoadU(hdi,im.data()+i),hs::Zero(hdi)); auto b0=hs::Gt(hs::LoadU(hdi,im2.data()+i),hs::Zero(hdi)); sink^=hs::BitsFromMask(hdi,hs::ExclusiveNeither(a0,b0));} });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto a0=hn::Gt(hn::LoadU(gdi,im.data()+i),hn::Zero(gdi)); auto b0=hn::Gt(hn::LoadU(gdi,im2.data()+i),hn::Zero(gdi)); sink^=hn::BitsFromMask(gdi,hn::ExclusiveNeither(a0,b0));} });
    nb::doNotOptimizeAway(sink);
  }
  {
    auto fspec=fa; fspec[0]=std::numeric_limits<float>::quiet_NaN(); std::uint64_t sink=0;
    nb::Bench b; b.title("IsEitherNaN f32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hwf<=kN;i+=hwf) sink^=hs::BitsFromMask(hdf,hs::IsEitherNaN(hs::LoadU(hdf,fspec.data()+i),hs::LoadU(hdf,fb.data()+i))); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gwf<=kN;i+=gwf) sink^=hn::BitsFromMask(gdf,hn::IsEitherNaN(hn::LoadU(gdf,fspec.data()+i),hn::LoadU(gdf,fb.data()+i))); });
    nb::doNotOptimizeAway(sink);
  }
  {
    std::uint64_t sink=0;
    nb::Bench b; b.title("AllBits0 i32").relative(true).batch(kN).minEpochIterations(8000);
    b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::AllBits0(hs::LoadU(hdi,a.data()+i)); });
    b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw){ auto v=hn::LoadU(gdi,a.data()+i); sink+=hn::AllTrue(gdi,hn::Eq(v,hn::Zero(gdi)));} });
    nb::doNotOptimizeAway(sink);
  }
  // ---- FindKnownFirst/LastTrue (known-nonzero mask variants; caller guarantees ≥1 true) ----
  {
    const auto im=geni(67); std::uint64_t sink=0;
    // ensure lane 0 true so the "known" precondition holds
    auto mk=[&](std::size_t i){ auto v=hs::LoadU(hdi,im.data()+i); return hs::Or(hs::Gt(v,hs::Zero(hdi)), hs::FirstN(hdi,1)); };
    auto gk=[&](std::size_t i){ auto v=hn::LoadU(gdi,im.data()+i); return hn::Or(hn::Gt(v,hn::Zero(gdi)), hn::FirstN(gdi,1)); };
    { nb::Bench b; b.title("FindKnownFirstTrue i32").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::FindKnownFirstTrue(hdi,mk(i)); });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) sink+=hn::FindKnownFirstTrue(gdi,gk(i)); }); }
    { nb::Bench b; b.title("FindKnownLastTrue i32").relative(true).batch(kN).minEpochIterations(8000);
      b.run("HYLE", [&]{ for(std::size_t i=0;i+hw<=kN;i+=hw) sink+=hs::FindKnownLastTrue(hdi,mk(i)); });
      b.run("Highway", [&]{ for(std::size_t i=0;i+gw<=kN;i+=gw) sink+=hn::FindKnownLastTrue(gdi,gk(i)); }); }
    nb::doNotOptimizeAway(sink);
  }
  return 0;
}
