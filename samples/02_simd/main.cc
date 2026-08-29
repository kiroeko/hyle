// Sample 02 -- simd: two levels of the SIMD pillar.
//
//  (A) DISPATCHED OPS (ops.h) -- the easy, portable path. dot()/saxpy() pick the
//      best ISA kernel at RUNTIME from the host's CPUID (scalar/SSE4.2/AVX2). One binary
//      runs fast everywhere; an old CPU never executes an instruction it lacks. This is
//      what most consumers want. runtime().isa tells you which kernel got selected.
//
//  (B) EXACT-TARGET KERNEL -- source users can bind this TU to one ISA and use the centrally
//      locked Highway register API directly. There is no CPUID or indirect call; the resulting
//      binary requires that deployment ISA. Installed SDK users stay on (A).
//
// Build: see ../README.md.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "hyle/simd/dispatch.h"  // best_isa / isa_name
#include "hyle/simd/ops.h"       // dot / saxpy (runtime-dispatched)
#include "hyle/simd/runtime.h"

#if defined(HYLE_SIMD_SOURCE_EXPERT)
#include "hwy/highway.h"

// (B) A direct source-expert kernel: sum of squares = dot(x, x). The target-bound TU uses
// Highway itself; HYLE does not duplicate its register surface or expose vectors across a boundary.
static float sum_of_squares(const float* x, std::size_t n) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const std::size_t w = hn::Lanes(d);
  auto acc = hn::Zero(d);
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto v = hn::LoadU(d, x + i);
    acc = hn::MulAdd(v, v, acc);
  }
  float s = hn::ReduceSum(d, acc);
  for (; i < n; ++i) s += x[i] * x[i];
  return s;
}
#endif

int main() {
  // What ISA did the runtime dispatcher choose on THIS machine?
  std::printf("host best ISA: %s | dispatched kernels run as: %s\n",
              hyle::simd::isa_name(hyle::simd::best_isa()),
              hyle::simd::isa_name(hyle::simd::runtime().isa));

  constexpr std::size_t n = 1000;
  std::vector<float> x(n), y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = 1.0f + 0.001f * static_cast<float>(i);
    y[i] = 2.0f - 0.0005f * static_cast<float>(i);
  }

  // (A) dispatched ops
  float dot = hyle::simd::dot(x.data(), y.data(), n);
  std::printf("dot(x,y)            = %.4f  (dispatched)\n", dot);

  // saxpy mutates y in place: y = a*x + y
  hyle::simd::saxpy(0.5f, x.data(), y.data(), n);
  std::printf("y[0] after saxpy    = %.4f  (dispatched)\n", y[0]);

#if defined(HYLE_SIMD_SOURCE_EXPERT)
  // (B) direct locked-Highway kernel, enabled only by HYLE_SAMPLE_SIMD_ISA.
  std::printf("sum_of_squares(x)   = %.4f  (source expert: direct Highway)\n",
              sum_of_squares(x.data(), n));
#endif
  return 0;
}
