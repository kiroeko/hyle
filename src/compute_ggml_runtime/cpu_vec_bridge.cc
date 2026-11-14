// Keep the locked CPU vector unit and its original scalar fallback together.
// AArch64 Advanced SIMD is baseline; no optional BF16 ISA is required here.
#define ggml_vec_dot_bf16 ggml_hyle_scalar_vec_dot_bf16
#include HYLE_GGML_CPU_VEC_SOURCE
#undef ggml_vec_dot_bf16

#if defined(_MSC_VER) && !defined(__clang__)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

extern "C" decltype(ggml_hyle_scalar_vec_dot_bf16) ggml_vec_dot_bf16;

extern "C" void ggml_vec_dot_bf16(
    int n, float *GGML_RESTRICT s, size_t bs, ggml_bf16_t *GGML_RESTRICT x,
    size_t bx, ggml_bf16_t *GGML_RESTRICT y, size_t by, int nrc) {
  assert(nrc == 1);
  GGML_UNUSED(nrc);
  GGML_UNUSED(bs);
  GGML_UNUSED(bx);
  GGML_UNUSED(by);
  // Preserve the scalar implementation's F32 products and F64 accumulation.
  // Widen bits directly, without narrowing the source or making an F32 bank.
  float64x2_t sums[4] = {vdupq_n_f64(0), vdupq_n_f64(0),
                         vdupq_n_f64(0), vdupq_n_f64(0)};
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const auto a = vld1q_u16(reinterpret_cast<const uint16_t *>(x + i));
    const auto b = vld1q_u16(reinterpret_cast<const uint16_t *>(y + i));
    const auto a0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(a)), 16));
    const auto a1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(a), 16));
    const auto b0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(b)), 16));
    const auto b1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(b), 16));
    const auto p0 = vmulq_f32(a0, b0);
    const auto p1 = vmulq_f32(a1, b1);
    sums[0] = vaddq_f64(sums[0], vcvt_f64_f32(vget_low_f32(p0)));
    sums[1] = vaddq_f64(sums[1], vcvt_high_f64_f32(p0));
    sums[2] = vaddq_f64(sums[2], vcvt_f64_f32(vget_low_f32(p1)));
    sums[3] = vaddq_f64(sums[3], vcvt_high_f64_f32(p1));
  }
  ggml_float sum = vaddvq_f64(vaddq_f64(vaddq_f64(sums[0], sums[1]),
                                      vaddq_f64(sums[2], sums[3])));
  for (; i < n; ++i)
    sum += static_cast<ggml_float>(GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i]));
  *s = static_cast<float>(sum);
}
