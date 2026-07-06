// NEON variant (aarch64 baseline, no extra -march). NEW: the old count/rfind cores were
// x86-only (string_ref_avx2.cc gated HYLE_ARCH_X64 + an SSE2 baseline), so aarch64 fell to a
// scalar byte loop. This gives Apple-Silicon/ARM the 16B/iter vectorized count (vpaddlq SAD
// chain) + backward rfind (vceqq + FindLastTrue) for the first time.
#define HYLE_SIMD_TARGET_NEON
#define HYLE_STRSEARCH_SUFFIX neon
#include "src/container/str_search_impl.h"
