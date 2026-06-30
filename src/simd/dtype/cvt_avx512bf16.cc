// AVX-512-BF16 f32->bf16 conversion (compiled -mavx512f -mavx512bw -mavx512vl -mavx512dq
// -mavx512bf16 / MSVC /arch:AVX512). Only REACHED when cpu_features().avx512bf16 is true
// (the dispatcher gates it, and avx512bf16 is XGETBV/ZMM-gated in dispatch.cc), so no
// AVX-512 instruction executes on a CPU lacking it.
//
// Why this exists (dtype's one un-pulled-ahead convert): the AVX2 bf16 path (cvt_avx2.cc)
// open-codes RNE as ~7 ymm integer ops (bias + shift + packus + permute) and only ties
// Highway (Win 1.09x / Lin 平). AVX-512-BF16 folds that whole RNE sequence into ONE
// instruction -- VCVTNEPS2BF16 (_mm512_cvtneps_pbh): 16 f32 lanes -> 16 bf16 lanes with
// hardware round-to-nearest-even. This is item ① of the perf "遗留优化方向" table.
//
// STATUS (honest, same bar as backend/avx512.h + kernels_avx512.cc): written + compiled
// (both GCC14 -mavx512bf16 objdump-verified to emit `vcvtneps2bf16`, and MSVC /arch:AVX512)
// + NOT executed -- this dev box is AVX2-only (no AVX-512-BF16 hardware, no Intel SDE here),
// so execution-correctness is PENDING (marked 存疑 until an AVX-512-BF16 machine or SDE runs
// it). The dispatcher (dtype_dispatch.cc) DOES route an AVX-512-BF16 host here by default,
// so a real such CPU runs this unvalidated path -- flagged in docs/architecture/simd-design.md.
//
// CORRECTNESS NOTE (why the test treats NaN equal-by-class, not bit-exact): for FINITE
// inputs VCVTNEPS2BF16 is IEEE round-to-nearest-even, identical to the scalar oracle's RNE,
// so finite outputs are bit-exact. For NaN the HARDWARE quiets to its own QNaN pattern,
// which need not match the scalar oracle's exact mantissa bits -- so the cross-check uses
// equal-by-class for NaN (same stance as the F16C fp16 path, unlike the AVX2 bf16 path
// which IS bit-identical because it open-codes the same integer math as the oracle).
#include "hyle/platform.h"  // HYLE_ARCH_X64 -- MUST precede the guard below

#if defined(HYLE_ARCH_X64)

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
  }
  for (; i < n; ++i) dst[i] = f32_to_bf16(src[i]);  // tail = scalar oracle
}

}  // namespace hyle::simd::detail

#endif  // HYLE_ARCH_X64
