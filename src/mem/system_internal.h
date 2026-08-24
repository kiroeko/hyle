#pragma once

#include "hyle/mem/allocator.h"

#if defined(HYLE_TESTING)
namespace hyle::mem {

// TEST-ONLY: dependency gate that observes the exact mimalloc object folded into
// hyle_mem instead of linking a second allocator copy into the test executable.
HYLE_MEM_API int system_backend_version_for_testing() noexcept;

}  // namespace hyle::mem
#endif
