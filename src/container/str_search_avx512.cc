// AVX-512 variant (compiled -mavx512f -mavx512bw -mavx512dq -mavx512vl). Entered only when
// best_isa() >= kAvx512, so the 512-bit codegen never runs on a non-AVX-512 CPU. 64B/iter
// SAD count (native _mm512_sad_epu8 via SumsOf8) + backward rfind (k-mask + FindLastTrue).
#define HYLE_SIMD_TARGET_AVX512
#define HYLE_STRSEARCH_SUFFIX avx512
#include "src/container/str_search_impl.h"
