#pragma once

#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_GRAPH_API
#elif defined(HYLE_GRAPH_BUILD)
#  define HYLE_GRAPH_API HYLE_EXPORT
#else
#  define HYLE_GRAPH_API HYLE_IMPORT
#endif
