// Sample 02 -- simd: two levels of the SIMD pillar.
//
//  (A) DISPATCHED OPS (ops.h) -- the easy, portable path. dot()/saxpy() pick the
//      best ISA kernel at RUNTIME from the host's CPUID (scalar/SSE4.2/AVX2). One binary
//      runs fast everywhere; an old CPU never executes an instruction it lacks. This is
//      what most consumers want. selected_target().isa reports the selected kernel tier.
//
//  (B) EXACT-TARGET KERNEL -- users can bind this TU to one ISA and use the centrally
//      locked Highway register API directly. There is no CPUID or indirect call; the resulting
//      binary requires that deployment ISA. This is identical for source/static/shared delivery.
//
//  (C) CUSTOM FAT KERNEL -- users can compile one direct-Highway template body once per
//      canonical ISA. HYLE selects and caches one whole-kernel pointer; the vector loop has no
//      feature branch, callback, wrapper register type, or second Highway dispatch.
//
// Build: see ../README.md.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "hyle/platform/cpu.h"   // raw host CPU facts
#include "hyle/simd/ops.h"       // dot / saxpy (runtime-dispatched)
#include "hyle/simd/runtime.h"

#if defined(HYLE_SAMPLE_SIMD_FAT)
#include "hyle/simd/fat_kernel.h"

using BiasReluFn = void (*)(float*, std::size_t, float) noexcept;
void hyle_sample_bias_relu_scalar(float*, std::size_t, float) noexcept;
#if defined(HYLE_ARCH_X64)
void hyle_sample_bias_relu_sse4(float*, std::size_t, float) noexcept;
void hyle_sample_bias_relu_avx2(float*, std::size_t, float) noexcept;
void hyle_sample_bias_relu_avx512(float*, std::size_t, float) noexcept;
#elif defined(HYLE_ARCH_ARM64)
void hyle_sample_bias_relu_neon(float*, std::size_t, float) noexcept;
void hyle_sample_bias_relu_sve(float*, std::size_t, float) noexcept;
#elif defined(HYLE_ARCH_RISCV64)
void hyle_sample_bias_relu_rvv(float*, std::size_t, float) noexcept;
#endif
#endif

#if defined(HYLE_SIMD_EXPERT)
#include "hwy/highway.h"

#if defined(HYLE_SIMD_EXACT_TARGET_SCALAR) && HWY_STATIC_TARGET != HWY_SCALAR
#error "hyle_target_simd(SCALAR) did not select Highway scalar"
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4) && HWY_STATIC_TARGET != HWY_SSE4
#error "hyle_target_simd(SSE4) did not select Highway SSE4"
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2) && HWY_STATIC_TARGET != HWY_AVX2
#error "hyle_target_simd(AVX2) did not select Highway AVX2"
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512) && HWY_STATIC_TARGET != HWY_AVX3
#error "hyle_target_simd(AVX512) did not select Highway AVX3"
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON) && \
    HWY_STATIC_TARGET != HWY_NEON && \
    HWY_STATIC_TARGET != HWY_NEON_WITHOUT_AES
#error "hyle_target_simd(NEON) did not select Highway NEON"
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE) && HWY_STATIC_TARGET != HWY_SVE
#error "hyle_target_simd(SVE) did not select Highway SVE"
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV) && HWY_STATIC_TARGET != HWY_RVV
#error "hyle_target_simd(RVV) did not select Highway RVV"
#endif

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
              hyle::platform::isa_name(hyle::platform::best_isa()),
              hyle::platform::isa_name(hyle::simd::selected_target().isa));

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

#if defined(HYLE_SIMD_EXPERT)
  // (B) direct locked-Highway kernel, enabled only by HYLE_SAMPLE_SIMD_ISA.
  std::printf("sum_of_squares(x)   = %.4f  (expert: direct Highway)\n",
              sum_of_squares(x.data(), n));
#endif

#if defined(HYLE_SAMPLE_SIMD_FAT)
  // (C) Select once at the pointer/count boundary, then call the same direct-Highway body.
#if defined(HYLE_ARCH_X64)
  static const BiasReluFn bias_relu = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_sample_bias_relu_scalar, &hyle_sample_bias_relu_sse4,
      &hyle_sample_bias_relu_avx2, &hyle_sample_bias_relu_avx512);
#elif defined(HYLE_ARCH_ARM64)
  static const BiasReluFn bias_relu = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_sample_bias_relu_scalar, &hyle_sample_bias_relu_neon,
      &hyle_sample_bias_relu_sve);
#elif defined(HYLE_ARCH_RISCV64)
  static const BiasReluFn bias_relu = HYLE_SIMD_SELECT_FAT_KERNEL(
      &hyle_sample_bias_relu_scalar, &hyle_sample_bias_relu_rvv);
#else
  static const BiasReluFn bias_relu =
      HYLE_SIMD_SELECT_FAT_KERNEL(&hyle_sample_bias_relu_scalar);
#endif
  bias_relu(x.data(), x.size(), -1.25f);
  std::printf("x[0] after bias_relu = %.4f  (source fat: direct Highway)\n", x[0]);
#endif
  return 0;
}
