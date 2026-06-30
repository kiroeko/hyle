// AVX-512 variant of the dispatched kernels (compiled -mavx512f -mavx512bw -mavx512vl
// -mavx512dq / MSVC /arch:AVX512). Only ENTERED when best_isa() >= kAvx512, so the 512-bit
// zmm codegen never runs on a non-AVX512 CPU. Backend = backend/avx512.h (k-register mask
// model); the SHARED kernel body in kernel_impl.h compiles 16-wide (512-bit / 4-byte f32)
// here with zero source change -- the whole point of the D-tag abstraction.
//
// STATUS (honest, same bar as backend/avx512.h): written + compiled + objdump-verified to
// emit real zmm instructions (33 zmm ops incl. vfmadd213ps zmm). NOT executed -- this dev
// box is AVX2-only (no AVX-512 hardware, no Intel SDE here), so execution-correctness is
// PENDING (marked 存疑 until an AVX-512 machine or SDE runs it). The dispatcher (dispatch_
// ops.cc) DOES route an AVX-512 host here by default (user directive), so a real AVX-512 CPU
// runs this unvalidated path -- flagged in docs/architecture/simd-design.md.
#define HYLE_SIMD_TARGET_AVX512
#define HYLE_KERNEL_SUFFIX avx512
#include "src/simd/ops/kernel_impl.h"
