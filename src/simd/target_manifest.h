#pragma once

// CMake builds generate the authoritative target manifest from the exact/fat source list.
// A few architecture validation harnesses intentionally compile individual TUs without CMake;
// keep those useful by deriving the same canonical list from their already-declared target macros.
#if __has_include("simd-dispatch/target_manifest.h")
#include "simd-dispatch/target_manifest.h"
#else

#include "hyle/platform.h"

#if defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
#define HYLE_SIMD_COMPILED_SCALAR 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Scalar, scalar, Scalar)
#define HYLE_SIMD_EXACT_VARIANT scalar
#define HYLE_SIMD_EXACT_RUNTIME kScalarRuntime
#define HYLE_SIMD_EXACT_ISA_TOKEN Scalar
#elif defined(HYLE_SIMD_EXACT_TARGET_SSE4)
#define HYLE_SIMD_COMPILED_SSE4 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Sse4, sse4, Sse42)
#define HYLE_SIMD_EXACT_VARIANT sse4
#define HYLE_SIMD_EXACT_RUNTIME kSse4Runtime
#define HYLE_SIMD_EXACT_ISA_TOKEN Sse42
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX2)
#define HYLE_SIMD_COMPILED_AVX2 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Avx2, avx2, Avx2)
#define HYLE_SIMD_EXACT_VARIANT avx2
#define HYLE_SIMD_EXACT_RUNTIME kAvx2Runtime
#define HYLE_SIMD_EXACT_ISA_TOKEN Avx2
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512)
#define HYLE_SIMD_COMPILED_AVX512 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Avx512, avx512, Avx512)
#define HYLE_SIMD_EXACT_VARIANT avx512
#define HYLE_SIMD_EXACT_RUNTIME kAvx512Runtime
#define HYLE_SIMD_EXACT_ISA_TOKEN Avx512
#elif defined(HYLE_SIMD_EXACT_TARGET_NEON)
#define HYLE_SIMD_COMPILED_NEON 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Neon, neon, Neon)
#define HYLE_SIMD_EXACT_VARIANT neon
#define HYLE_SIMD_EXACT_RUNTIME kNeonRuntime
#define HYLE_SIMD_EXACT_ISA_TOKEN Neon
#elif defined(HYLE_SIMD_EXACT_TARGET_SVE)
#define HYLE_SIMD_COMPILED_SVE 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Sve, sve, Sve)
#define HYLE_SIMD_EXACT_VARIANT sve
#define HYLE_SIMD_EXACT_RUNTIME kSveRuntime
#define HYLE_SIMD_EXACT_ISA_TOKEN Sve
#elif defined(HYLE_SIMD_EXACT_TARGET_RVV)
#define HYLE_SIMD_COMPILED_RVV 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Rvv, rvv, Rvv)
#define HYLE_SIMD_EXACT_VARIANT rvv
#define HYLE_SIMD_EXACT_RUNTIME kRvvRuntime
#define HYLE_SIMD_EXACT_ISA_TOKEN Rvv
#elif defined(HYLE_ARCH_X64)
#define HYLE_SIMD_COMPILED_SCALAR 1
#define HYLE_SIMD_COMPILED_SSE4 1
#define HYLE_SIMD_COMPILED_AVX2 1
#define HYLE_SIMD_COMPILED_AVX512 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) \
  M(Scalar, scalar, Scalar)                    \
  M(Sse4, sse4, Sse42)                        \
  M(Avx2, avx2, Avx2)                         \
  M(Avx512, avx512, Avx512)
#elif defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_SIMD_COMPILED_SCALAR 1
#define HYLE_SIMD_COMPILED_NEON 1
#define HYLE_SIMD_COMPILED_SVE 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) \
  M(Scalar, scalar, Scalar)                    \
  M(Neon, neon, Neon)                         \
  M(Sve, sve, Sve)
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SIMD_COMPILED_SCALAR 1
#define HYLE_SIMD_COMPILED_NEON 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) \
  M(Scalar, scalar, Scalar)                    \
  M(Neon, neon, Neon)
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_SIMD_COMPILED_SCALAR 1
#define HYLE_SIMD_COMPILED_RVV 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) \
  M(Scalar, scalar, Scalar)                    \
  M(Rvv, rvv, Rvv)
#else
#define HYLE_SIMD_COMPILED_SCALAR 1
#define HYLE_SIMD_FOR_EACH_COMPILED_TARGET(M) M(Scalar, scalar, Scalar)
#endif

#endif
