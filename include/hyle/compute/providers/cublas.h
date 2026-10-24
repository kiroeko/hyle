#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::cublas {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4355424c41530001ULL};

// Expert composition factory for native CUDA matrix projections.
// MatMul accepts matching plain FP16/BF16/FP32 inputs whose final two axes
// have row/column-major storage, with the ordinary inferred dense output.
// Positive leading batch extents use the ordinary
// right-aligned broadcast contract; batch strides must contain whole elements
// and output matrix spans must not overlap. Storage bases are 16-byte aligned.
// BFloatLinearF32 additionally accepts dense BF16 [T,K] activations and
// stored BF16 [E,K] weights, producing dense F32 [T,E] scores. Its weight
// orientation is a native descriptor; no transpose or dtype copy is required.
// Accumulation is FP32, without TF32 input
// conversion. Affine batch runs, descriptors and algorithms are fixed in
// prepare; rank-2 uses one run. Each Invocation owns
// a separate vendor handle and borrows its caller-supplied CUDA workspace.
// Context/artifacts retain the backend; payload and workspace owners must cover
// every real Completion. Vendor-private storage has an unknown byte bound.
#if defined(HYLE_COMPUTE_CUDA)
HYLE_COMPUTE_PROVIDER_CUBLAS_API Provider provider() noexcept;
#endif

} // namespace hyle::compute::cublas
