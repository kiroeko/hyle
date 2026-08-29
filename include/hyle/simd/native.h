#pragma once

#include "hyle/simd/target.h"
#include "hyle/simd/fat_kernel.h"

// Umbrella for target-bound SIMD code. The selected backend is fixed for this translation unit;
// use the ordinary batch APIs when shipping one binary to an unknown CPU. Use
// selected_target() only for diagnostics, not to dispatch individual operations.

#include "hyle/simd/native/algorithm.h"
#include "hyle/simd/native/ops.h"
#include "hyle/simd/native/random.h"
#include "hyle/simd/native/scan.h"
#include "hyle/simd/native/search.h"
