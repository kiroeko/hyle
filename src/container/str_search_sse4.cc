// SSE4.2 variant (compiled -msse4.2 / MSVC baseline). Entered only when best_isa() >= kSse42.
#define HYLE_SIMD_TARGET_SSE4
#define HYLE_STRSEARCH_SUFFIX sse4
#include "src/container/str_search_impl.h"
