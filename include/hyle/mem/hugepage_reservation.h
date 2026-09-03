#pragma once

#include <cstddef>

#include "hyle/error/status.h"
#include "hyle/mem/visibility.h"
#include "hyle/platform/numa.h"

namespace hyle::mem {

// Reserve contiguous 1 GiB OS pages for the shipping mimalloc heap. This is
// allocator-process configuration, not a HugeMapping owner operation, so it
// remains in the base Mem factor.
HYLE_MEM_API Status reserve_huge_1g_pages(
    std::size_t pages, int numa_node = platform::kNumaAny,
    unsigned timeout_ms = 0) noexcept;

}  // namespace hyle::mem
