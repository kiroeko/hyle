#pragma once

// SIMD export and target type boundary. Raw CPU/OS facts belong to
// hyle::platform and are intentionally not re-exported from this namespace.
// Users that need detection call platform directly; users that need the target
// actually selected by HYLE call selected_target() from runtime.h.

#include "hyle/platform/cpu.h"
#include "hyle/visibility.h"

// simd module export macro. Defined here (not in platform/cpu.h) because it gates
// simd's OWN exported ops (ops.h/math.h/dtype_ops.h include this header for it).
// The detector's own export macro is HYLE_PLATFORM_API, over in platform/cpu.h.
#if defined(HYLE_STATIC)
#  define HYLE_SIMD_API
#elif defined(HYLE_SIMD_BUILD)
#  define HYLE_SIMD_API HYLE_EXPORT
#else
#  define HYLE_SIMD_API HYLE_IMPORT
#endif

namespace hyle::simd {

using Isa = ::hyle::platform::Isa;

}  // namespace hyle::simd
