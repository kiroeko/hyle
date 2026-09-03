#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_FS_API
#elif defined(HYLE_PLATFORM_FS_BUILD)
#  define HYLE_PLATFORM_FS_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_FS_API HYLE_IMPORT
#endif
