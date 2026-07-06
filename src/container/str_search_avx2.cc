// AVX2 variant (compiled -mavx2 -mfma / MSVC /arch:AVX2). Entered only when best_isa() >= kAvx2,
// so the 256-bit codegen never runs on a non-AVX2 CPU. Replaces the old hand-rolled
// string_ref_avx2.cc (same 32B/iter SAD count + backward rfind, now via the vec.h abstraction).
#define HYLE_SIMD_TARGET_AVX2
#define HYLE_STRSEARCH_SUFFIX avx2
#include "src/container/str_search_impl.h"
