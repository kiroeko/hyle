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
  return 0;
}
