#include "hyle/simd/dtype_ops.h"

#include <cstddef>

#include "hyle/core/environment.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/dtype/conversion_kernels.h"
#include "src/simd/dtype/kernels.h"
#include "src/simd/runtime_internal.h"

// Baseline dispatcher for the complete storage conversion graph. The common
// f32/f16/bf16 routes are selected once in Runtime and enter a generated direct-
// Highway target body. fp8 and precise x86 feature gaps remain local below.

namespace hyle::simd {
namespace {

template <typename Src, typename Dst, class Decode, class Encode>
void convert_through_f32(const Src* src, Dst* dst, std::size_t n,
                         Decode decode, Encode encode) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const float value = decode(src[i]);
    dst[i] = encode(value);
  }
}

}  // namespace

#if defined(HYLE_SIMD_EXACT)
#define HYLE_DTYPE_CONVERT_CALL(name, ...) \
  HYLE_SIMD_EXACT_SYMBOL(name)(__VA_ARGS__)
#else
#define HYLE_DTYPE_CONVERT_CALL(name, ...) \
  runtime().dtype_convert.name(__VA_ARGS__)
#endif

void convert(const float16_t* src, float* dst, std::size_t n) noexcept {
  HYLE_DTYPE_CONVERT_CALL(f16_to_f32, src, dst, n);
}

void convert(const float* src, float16_t* dst, std::size_t n) noexcept {
  HYLE_DTYPE_CONVERT_CALL(f32_to_f16, src, dst, n);
}

void convert(const bfloat16_t* src, float* dst, std::size_t n) noexcept {
  HYLE_DTYPE_CONVERT_CALL(bf16_to_f32, src, dst, n);
}

void convert(const float* src, bfloat16_t* dst, std::size_t n) noexcept {
  HYLE_DTYPE_CONVERT_CALL(f32_to_bf16, src, dst, n);
}

// fp8 -> f32: SCALAR only (no hardware fp8->f32 convert; the AVX10.2 fp8 instructions are the
// f32/f16 -> fp8 direction only). The scalar bit-math core is the complete implementation.
void convert(const float8_e4m3_t* src, float* dst, std::size_t n) noexcept {
  detail::f8e4m3_to_f32_scalar(src, dst, n);
}
void convert(const float8_e5m2_t* src, float* dst, std::size_t n) noexcept {
  detail::f8e5m2_to_f32_scalar(src, dst, n);
}

// AVX10.2 gate: build has -mavx10.2-512 (clang-20) AND the CPU advertises avx10_2 (+ZMM).
#if HYLE_ARCH_X64 && defined(HYLE_HAVE_AVX10_2)
bool use_avx10_2() noexcept {
  return (runtime().dtype.features & Runtime::kFeatureAvx10_2) != 0;
}
#endif

// f32 -> fp8: AVX10.2 hardware kernel iff available. There is NO direct f32->fp8 instruction, so
// the kernel goes f32 --(round-to-odd)--> f16 --(hw)--> fp8, which is BIT-IDENTICAL to the direct
// scalar oracle (round-to-odd defeats the double-rounding; verified over 486.6M f32). Else scalar.
void convert(const float* src, float8_e4m3_t* dst, std::size_t n) noexcept {
#if HYLE_ARCH_X64 && defined(HYLE_HAVE_AVX10_2)
  static const bool hw = use_avx10_2();
  if (hw) { detail::f32_to_f8e4m3_avx10_2(src, dst, n); return; }
#endif
  detail::f32_to_f8e4m3_scalar(src, dst, n);
}
void convert(const float* src, float8_e5m2_t* dst, std::size_t n) noexcept {
#if HYLE_ARCH_X64 && defined(HYLE_HAVE_AVX10_2)
  static const bool hw = use_avx10_2();
  if (hw) { detail::f32_to_f8e5m2_avx10_2(src, dst, n); return; }
#endif
  detail::f32_to_f8e5m2_scalar(src, dst, n);
}
void convert(const float16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept {
#if HYLE_ARCH_X64 && defined(HYLE_HAVE_AVX10_2)
  static const bool hw = use_avx10_2();
  if (hw) { detail::f16_to_f8e4m3_avx10_2(src, dst, n); return; }
#endif
  detail::f16_to_f8e4m3_scalar(src, dst, n);
}
void convert(const float16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept {
#if HYLE_ARCH_X64 && defined(HYLE_HAVE_AVX10_2)
  static const bool hw = use_avx10_2();
  if (hw) { detail::f16_to_f8e5m2_avx10_2(src, dst, n); return; }
#endif
  detail::f16_to_f8e5m2_scalar(src, dst, n);
}

void convert(const float16_t* src, bfloat16_t* dst, std::size_t n) noexcept {
  HYLE_DTYPE_CONVERT_CALL(f16_to_bf16, src, dst, n);
}

void convert(const bfloat16_t* src, float16_t* dst, std::size_t n) noexcept {
  HYLE_DTYPE_CONVERT_CALL(bf16_to_f16, src, dst, n);
}

void convert(const bfloat16_t* src, float8_e4m3_t* dst, std::size_t n) noexcept {
  convert_through_f32(src, dst, n, bf16_to_f32, f32_to_f8e4m3);
}

void convert(const bfloat16_t* src, float8_e5m2_t* dst, std::size_t n) noexcept {
  convert_through_f32(src, dst, n, bf16_to_f32, f32_to_f8e5m2);
}

void convert(const float8_e4m3_t* src, float16_t* dst, std::size_t n) noexcept {
  convert_through_f32(src, dst, n, f8e4m3_to_f32, f32_to_f16);
}

void convert(const float8_e4m3_t* src, bfloat16_t* dst, std::size_t n) noexcept {
  convert_through_f32(src, dst, n, f8e4m3_to_f32, f32_to_bf16);
}

void convert(const float8_e4m3_t* src, float8_e5m2_t* dst,
             std::size_t n) noexcept {
  convert_through_f32(src, dst, n, f8e4m3_to_f32, f32_to_f8e5m2);
}

void convert(const float8_e5m2_t* src, float16_t* dst, std::size_t n) noexcept {
  convert_through_f32(src, dst, n, f8e5m2_to_f32, f32_to_f16);
}

void convert(const float8_e5m2_t* src, bfloat16_t* dst, std::size_t n) noexcept {
  convert_through_f32(src, dst, n, f8e5m2_to_f32, f32_to_bf16);
}

void convert(const float8_e5m2_t* src, float8_e4m3_t* dst,
             std::size_t n) noexcept {
  convert_through_f32(src, dst, n, f8e5m2_to_f32, f32_to_f8e4m3);
}

#undef HYLE_DTYPE_CONVERT_CALL

}  // namespace hyle::simd
