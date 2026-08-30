/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

// Source-only finite HYLE provider exceptions.  Include this only in a target
// bound TU created by hyle_target_simd()/HYLE's per-ISA build.  Standard SIMD
// primitives remain owned by locked Highway; this header contains no fallback
// and deliberately fails for targets without a qualified HYLE winner.

#include "hyle/simd/provider/route.h"

#if defined(HYLE_SIMD_TARGET_AVX2)
#include "hyle/simd/provider/hyle_winners_avx2-inl.h"
#elif defined(HYLE_SIMD_TARGET_NEON)
#include "hyle/simd/provider/hyle_winners_neon-inl.h"
#else
#error "HYLE finite winner bodies are available only for qualified AVX2 or NEON targets"
#endif

