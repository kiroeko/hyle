#pragma once

// Runtime ISA dispatch types/queries. These MOVED to platform/ (conventions §10 /
// simd-design "下沉 CPUID 到 platform"): "what can this CPU run" is a platform
// capability, so both simd/ and container/ can query it WITHOUT depending on the
// simd module. This header now forwards the platform names into hyle::simd so the
// existing simd call sites (dispatch_ops.cc, kernels, dtype/math) stay unchanged.
//
// The detector implementation (CPUID + XGETBV, LTO-isolated) lives in
// platform/cpu.{h,cc}.

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

using CpuFeatures = ::hyle::platform::CpuFeatures;
using Isa = ::hyle::platform::Isa;

using ::hyle::platform::best_isa;
using ::hyle::platform::cpu_features;
using ::hyle::platform::isa_name;
using ::hyle::platform::supports;

}  // namespace hyle::simd
