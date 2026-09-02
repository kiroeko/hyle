#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_MEM_API
#elif defined(HYLE_MEM_BUILD)
#  define HYLE_MEM_API HYLE_EXPORT
#else
#  define HYLE_MEM_API HYLE_IMPORT
#endif
