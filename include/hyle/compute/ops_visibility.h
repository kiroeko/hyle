#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_OPS_API
#elif defined(HYLE_COMPUTE_OPS_BUILD)
#  define HYLE_COMPUTE_OPS_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_OPS_API HYLE_IMPORT
#endif
