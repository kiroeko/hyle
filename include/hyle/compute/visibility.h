#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif
