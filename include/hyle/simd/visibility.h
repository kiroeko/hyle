#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_SIMD_API
#elif defined(HYLE_SIMD_BUILD)
#  define HYLE_SIMD_API HYLE_EXPORT
#else
#  define HYLE_SIMD_API HYLE_IMPORT
#endif
