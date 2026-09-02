#pragma once

// Public arena family. These owners share bulk/rewind lifetime semantics but
// deliberately keep storage ownership and backing policy in distinct types.
#include "hyle/mem/region.h"          // Arena, ArenaOptions, FixedArena
#include "hyle/mem/shared_pool.h"     // PooledArena
#include "hyle/mem/stack_allocator.h" // StackArena
