#pragma once

#include "hyle/simd/target.h"
#include "hyle/simd/fat_kernel.h"

// Umbrella for target-bound SIMD code. The selected backend is fixed for this translation unit;
// use the ordinary batch APIs or runtime() when shipping one binary to an unknown CPU.

#include "hyle/simd/native/ops.h"
#include "hyle/simd/native/random.h"
#include "hyle/simd/native/scan.h"
