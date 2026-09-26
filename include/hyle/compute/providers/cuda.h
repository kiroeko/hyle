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
inline constexpr ProviderId kCcclProviderId{0x48594c4500000001ULL,
                                            0x4343434c00000001ULL};
inline constexpr ProviderId kDeepSeekProviderId{0x48594c4500000001ULL,
                                                0x445345454b000001ULL};

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
// Concat accepts variadic plain 1/2/4/8-byte scalar inputs and dense output,
// arbitrary axis/byte strides, and preserves storage bits exactly. No input/
// output overlap. Up to four inputs per native kernel (one for the common two
// inputs); empty groups do not launch or access pointers. Cold metadata scales
// with arity, no caller workspace or slot state. Output count <= INT32_MAX.
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
// Also supplies dense same-type FP16/BF16 RMSNorm with a weight: widths
// 64/128/256, multiples of 256 through 8192, and multiples of 512 through
// 16384; FP32 accumulation and one final rounding, exact input0 alias allowed.
// Unary SiLU/GELU/GELU-tanh accepts dense FP32/FP16/BF16 complete 16-byte
// vectors and exact input0 aliases, with stable tails and -Inf -> +0.
// These added domains require 16-byte bindings, count <= INT32_MAX, no caller
// workspace or slot state. Other norms/layouts remain separate candidates.
// Complete model_ops::SqrtSoftplusRouting uses the pinned original MoE gate
// program: F32 [T,256/384] logits and [E] bias, top6 F32 weights/I32 IDs.
// It preserves unbiased positive-sum normalization and output scaling, with
// native tie selection rather than generic stable TopK. No workspace/aliases;
// four-byte bindings, count <= INT32_MAX, cold launch binding and capture.
HYLE_COMPUTE_PROVIDER_CUDA_API Provider sglang_provider() noexcept;

// Original DeepSeek block128 weight decode for cold model preparation.
// E4M3FN codes and native F32 scales become dense BF16/F32 [N,K]. One native
// launch, no workspace, no aliases; input alignment4, output alignment2/4,
// signed-I32 native offsets and CUDA Y-grid bounds. Qualified AOT domain is
// SM89 / Driver API13040. Caller owns the resulting cache and its reuse.
// Also binds the original DeepSeek V4 BF16 quantization roundtrips:
// E4M3FN/group64 and E2M1/group32, with upward power-of-two scales. These
// require distinct 16-byte-aligned input/output and caller workspace of four
// bytes per group (4-byte aligned), kept alive until Completion. One launch;
// no private invocation allocation or Python/TileLang runtime dependency.
HYLE_COMPUTE_PROVIDER_CUDA_API Provider deepseek_provider() noexcept;

// Original CCCL3.3.4 device collectives, qualified with CUDA13.3
// toolkit/runtime on SM89. Other versions/architectures decline during cold
// preparation. Scan accepts F32/I32; integer overflow, including the exclusive
// final total, fails without writing output. TopK accepts F32/F16/BF16/I32,
// preserves source bits, puts NaNs last and retains original index order for
// equal keys. Both accept plain rank<=8 byte strides, dense outputs,
// count<=INT32_MAX, and no aliases. Allocate workspace_requirement() bytes per
// invocation; ordinary Graph plans that workspace. Execution is managed, and
// checked I32 Scan waits on a two-scalar GPU validation summary. No external
// capture. Gather preserves plain 1/2/4/8-byte storage with I32/I64 indices and
// the same rank/axis/stride/alias rules. Checked indices use original GPU
// min/max and a two-scalar host wait before writing any output; unchecked skips
// validation. Output and index counts <=INT32_MAX, storage spans <=PTRDIFF_MAX.
HYLE_COMPUTE_PROVIDER_CUDA_API Provider cccl_provider() noexcept;
#endif

} // namespace hyle::compute::cuda_native
