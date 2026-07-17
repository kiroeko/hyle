// Compile-instantiation probe for the scalable product math body. This is linked into the SVE and
// RVV backend tests but is intentionally not called: its job is to make every template body cross
// the real target compiler, catching backend operations that shallow header tests do not reach.

#include "src/simd/math/math_hyle-inl.h"
#include "src/simd/math/math_xsimd_hyle-inl.h"
#include "src/simd/math/table_f64.h"

namespace hyle::simd::test {

using namespace hyle::simd;
using namespace hyle::simd::detail;

template <class D>
void instantiate_math(D d, TFromD<D>* p) {
  auto x = LoadU(d, p);
  auto y = x;

#define HYLE_INSTANTIATE_UNARY(fn) y = fn(d, y)
  HYLE_INSTANTIATE_UNARY(Sin);
  HYLE_INSTANTIATE_UNARY(Cos);
  HYLE_INSTANTIATE_UNARY(Exp);
  HYLE_INSTANTIATE_UNARY(Exp2);
  HYLE_INSTANTIATE_UNARY(Expm1);
  HYLE_INSTANTIATE_UNARY(Log);
  HYLE_INSTANTIATE_UNARY(Log2);
  HYLE_INSTANTIATE_UNARY(Log10);
  HYLE_INSTANTIATE_UNARY(Log1p);
  HYLE_INSTANTIATE_UNARY(Tanh);
  HYLE_INSTANTIATE_UNARY(Erf);
  HYLE_INSTANTIATE_UNARY(Atan);
  HYLE_INSTANTIATE_UNARY(Cbrt);
  HYLE_INSTANTIATE_UNARY(Asinh);
  HYLE_INSTANTIATE_UNARY(Acosh);
  HYLE_INSTANTIATE_UNARY(Acos);
  HYLE_INSTANTIATE_UNARY(Asin);
  HYLE_INSTANTIATE_UNARY(Sinh);
  HYLE_INSTANTIATE_UNARY(Cosh);
  HYLE_INSTANTIATE_UNARY(Tan);
  HYLE_INSTANTIATE_UNARY(Atanh);
#undef HYLE_INSTANTIATE_UNARY

  y = Atan2(d, y, x);
  y = Pow(d, y, x);
  y = Hypot(d, y, x);
  y = xsimd_hyle::Erfc(d, y);
  y = xsimd_hyle::Exp10(d, y);
  y = xsimd_hyle::Fmod(d, y, x);
  y = xsimd_hyle::Fdim(d, y, x);
  y = xsimd_hyle::Tgamma(d, y);
  y = xsimd_hyle::HypotInRange(d, y, x);
  StoreU(y, d, p);
}

void scalable_math_compile_probe(float* f32, double* f64) {
  instantiate_math(ScalableTag<float>{}, f32);
  instantiate_math(ScalableTag<double>{}, f64);

  const ScalableTag<double> d;
  const auto x = LoadU(d, f64);
  StoreU(table_f64::LogNormal(d, table_f64::ExpNormal(d, x)), d, f64);

  // Rebind<double -> i32> is a logical half tag on scalable backends. The in-range primitive's
  // public result has the same lane count and therefore uses an explicit full i32 tag.
  const Simd<std::int32_t, ScalableTag<double>::kMaxLanes, 0> di;
  (void)DemoteInRangeTo(di, x);
}

}  // namespace hyle::simd::test
