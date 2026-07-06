// Scalar variant of the dispatched string count/rfind kernels (portable floor, no -march).
// Always available on every arch; the dispatcher's fallback.
#define HYLE_STRSEARCH_SUFFIX scalar
#include "src/container/str_search_impl.h"
