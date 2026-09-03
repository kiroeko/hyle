#pragma once

#include "hyle/mem/allocator.h"

namespace hyle::mem::detail {

// Intra-module system backing seam. Product TUs use these direct hidden calls
// instead of bouncing through the shared-consumer import seam or the runtime
// AllocatorRef vtable. They preserve the exact mimalloc/alignment contract owned
// by system.cc and are never installed as public headers.
void* system_local_allocate(std::size_t size,
                            std::size_t alignment) noexcept;
void system_local_deallocate(void* pointer, std::size_t size,
                             std::size_t alignment) noexcept;
void* system_local_reallocate(void* pointer, std::size_t new_size,
                              std::size_t alignment) noexcept;

}  // namespace hyle::mem::detail

#if defined(HYLE_TESTING)
namespace hyle::mem {

// TEST-ONLY: dependency gate that observes the exact mimalloc object folded into
// hyle_mem instead of linking a second allocator copy into the test executable.
HYLE_MEM_API int system_backend_version_for_testing() noexcept;

}  // namespace hyle::mem
#endif
