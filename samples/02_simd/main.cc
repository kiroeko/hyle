// Sample 02 -- simd: two levels of the SIMD pillar.
//
//  (A) DISPATCHED OPS (ops.h) -- the easy, portable path. dot_f32()/saxpy_f32() pick the
//      best ISA kernel at RUNTIME from the host's CPUID (scalar/SSE4.2/AVX2). One binary
//      runs fast everywhere; an old CPU never executes an instruction it lacks. This is
//      what most consumers want. active_variant() tells you which kernel got selected.
//
//  (B) DIRECT vec<T,N> KERNEL (vec.h) -- write your own portable kernel against the D-tag
//      abstraction (ScalableTag + Lanes), no intrinsics. The backend + your -march decide
//      the width. Compiled at baseline here = SSE2; add -march=native (see CMakeLists) to
//      widen. Use this when you need a custom kernel the dispatched ops don't provide.
//
// Build: see ../README.md.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "hyle/simd/dispatch.h"  // best_isa / isa_name / active_variant
#include "hyle/simd/ops.h"       // dot_f32 / saxpy_f32 (runtime-dispatched)
#include "hyle/simd/vec.h"       // vec<T,N> D-tag, for the direct kernel

// (B) A direct portable kernel: sum of squares = dot(x, x). Written once against the
// D-tag contract; vectorizes to whatever backend this TU compiled for.
static float sum_of_squares(const float* x, std::size_t n) {
  namespace hs = hyle::simd;
  const hs::ScalableTag<float> d;
  const std::size_t w = hs::Lanes(d);
  auto acc = hs::Zero(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto v = hs::LoadU(d, x + i);
    acc = hs::MulAdd(v, v, acc);  // acc += v*v
  }
  float s = hs::ReduceSum(d, acc);
  for (; i < n; ++i) s += x[i] * x[i];  // tail
  return s;
}

int main() {
  // What ISA did the runtime dispatcher choose on THIS machine?
  std::printf("host best ISA: %s | dispatched kernels run as: %s\n",
              hyle::simd::isa_name(hyle::simd::best_isa()),
              hyle::simd::isa_name(hyle::simd::active_variant()));

  constexpr std::size_t n = 1000;
  std::vector<float> x(n), y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = 1.0f + 0.001f * static_cast<float>(i);
    y[i] = 2.0f - 0.0005f * static_cast<float>(i);
  }

  // (A) dispatched ops
  float dot = hyle::simd::dot_f32(x.data(), y.data(), n);
  std::printf("dot(x,y)            = %.4f  (dispatched)\n", dot);

  // saxpy mutates y in place: y = a*x + y
  hyle::simd::saxpy_f32(0.5f, x.data(), y.data(), n);
  std::printf("y[0] after saxpy    = %.4f  (dispatched)\n", y[0]);

  // (B) direct vec kernel
  std::printf("sum_of_squares(x)   = %.4f  (direct vec<T,N> kernel)\n",
              sum_of_squares(x.data(), n));
  return 0;
}
