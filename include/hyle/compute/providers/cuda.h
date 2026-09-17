#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_CUDA_API
#elif defined(HYLE_COMPUTE_PROVIDER_CUDA_BUILD)
#define HYLE_COMPUTE_PROVIDER_CUDA_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_CUDA_API HYLE_IMPORT
#endif

namespace hyle::compute::cuda_native {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x435544414b455201ULL};
inline constexpr ProviderId kSglangProviderId{0x48594c4500000001ULL,
                                              0x53474c4143540001ULL};

// Expert composition factory for HYLE's native CUDA tensor/gated primitives.
// Plain View uses the schema's exact storage alias without launching a kernel.
// Copy/Fill/Select preserve 1/2/4/8-byte scalar storage; Cast and Unary/Binary
// support FP32/FP16/BF16. Numeric tensor output is dense; rank <= 8, rank-zero
// broadcasting and arbitrary nonnegative byte strides are supported. Integer
// and FP8 arithmetic/Cast are not implemented. Elementwise arithmetic allows
// only exact input/output aliases with identical descriptions; Copy/Cast do not
// overlap. These primitives require no native workspace or per-slot state.
//
// RMSNorm/LayerNorm use dense FP32/FP16/BF16, FP32 accumulation on the last
// axis, and independently typed scale/bias (input type or FP32 for half input).
// LayerNorm uses two-pass centered variance. Only exact input0/output aliasing
// is allowed. One kernel per nonempty norm, no caller workspace or per-slot
// device state; ordinary prepare binds the compiled SDK collective once.
//
// Reduce Sum/Mean/Min/Max and Softmax/LogSoftmax accept plain FP32/FP16/BF16,
// arbitrary axes/byte strides and dense output. Accumulation is FP32 and must
// be explicit for half Reduce. Empty reductions write their scalar identities;
// Min/Max propagate any NaN. Reduce never aliases input. Softmax supports
// scale, broadcast Bool/same-type additive masks and exact input0 aliases; an
// invalid normalization row yields NaN. One native kernel, no caller workspace
// or slot state. Softmax caches <=8192 columns in <=32 KiB of FP32 shared
// memory; longer rows read their input three times. Input/output count <=
// INT32_MAX.
//
// Gated activation:
// Supports SiLU/GELU/GELU-tanh, dense FP16/BF16/FP32 separate inputs or [...,
// gate | value] packed halves. Arithmetic is FP32 with one final conversion to
// the output type. Exact shapes are prepared once; output element count <=
// INT32_MAX. Empty outputs require no device payload allocation or kernel
// launch. No tensor aliasing, per-invocation native storage, or caller
// workspace. Payload owners must cover the real Completion.
// Kernel/configuration binding is cold; the shared CUDA stream also permits
// native Graph capture.
#if defined(HYLE_COMPUTE_CUDA)
HYLE_COMPUTE_PROVIDER_CUDA_API Provider provider() noexcept;

// Pinned precise SGLang implementation under the same native ownership and
// completion contract and three activation modes. Packed halves only, 16-byte
// tensor alignment, FP16/BF16 hidden dimension divisible by 8 or FP32 by 4.
// FP32 intermediates and one final output conversion; no fast-math, PDL,
// filtering, or input reuse. The ordinary CUDA route selects it in this domain
// during preparation.
HYLE_COMPUTE_PROVIDER_CUDA_API Provider sglang_provider() noexcept;
#endif

} // namespace hyle::compute::cuda_native
