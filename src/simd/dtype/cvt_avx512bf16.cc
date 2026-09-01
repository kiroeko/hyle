// AVX-512-BF16 f32->bf16 conversion (compiled -mavx512f -mavx512bw -mavx512vl -mavx512dq
// -mavx512bf16 / MSVC /arch:AVX512). Only REACHED when cpu_features().avx512bf16 is true
// (the dispatcher gates it, and avx512bf16 is XGETBV/ZMM-gated in dispatch.cc), so no
// AVX-512 instruction executes on a CPU lacking it.
//
// Why this exists: the common path is direct Highway at the product tier, while
// AVX-512-BF16 folds f32 -> bf16 rounding into VCVTNEPS2BF16. This precise
// feature specialization is selected only for a capable host in a fat build.
//
// VCVTNEPS2BF16 is bit-identical to the scalar oracle for finite values. Hardware may choose
// another quiet-NaN payload; those cold exceptional lanes are restored after the vector store.
#include "hyle/core/environment.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if HYLE_ARCH_X64

#include "src/simd/dtype/kernels.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dtype.h"

namespace hyle::simd::detail {

// f32 -> bf16, 16-wide, hardware RNE via VCVTNEPS2BF16. Tail = scalar oracle (exact).
void f32_to_bf16_avx512bf16(const float* src, bfloat16_t* dst, std::size_t n) noexcept {
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 v = _mm512_loadu_ps(src + i);
    const __m256bh b = _mm512_cvtneps_pbh(v);  // 16x f32 -> 16x bf16, RNE
    // C-style cast (not reinterpret_cast): __m256bh is a vector type on GCC/Clang but a
    // union on MSVC; the C-style cast is the form both toolchains accept (probe-verified).
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), (__m256i)b);
    const __mmask16 nan_mask = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
    if (nan_mask != 0) {
      for (std::size_t lane = 0; lane < 16; ++lane) {
        if ((nan_mask & (__mmask16{1} << lane)) != 0)
          dst[i + lane] = f32_to_bf16(src[i + lane]);
      }
    }
  }
  for (; i < n; ++i) dst[i] = f32_to_bf16(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
