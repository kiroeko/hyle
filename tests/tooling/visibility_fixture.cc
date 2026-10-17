#include "hyle/visibility.h"

#define HYLE_VISIBILITY_STRING_IMPL(value) #value
#define HYLE_VISIBILITY_STRING(value) HYLE_VISIBILITY_STRING_IMPL(value)

const char *visibility_compute = HYLE_VISIBILITY_STRING(HYLE_COMPUTE_API);
const char *visibility_graph = HYLE_VISIBILITY_STRING(HYLE_GRAPH_API);
const char *visibility_io = HYLE_VISIBILITY_STRING(HYLE_IO_API);
const char *visibility_container = HYLE_VISIBILITY_STRING(HYLE_CONTAINER_API);
const char *visibility_concurrent = HYLE_VISIBILITY_STRING(HYLE_CONTAINER_CONCURRENT_API);
const char *visibility_queue = HYLE_VISIBILITY_STRING(HYLE_CONTAINER_QUEUE_API);
