#pragma once

#include <cstddef>

#include "hyle/compute/device.h"
#include "hyle/compute/providers/marlin.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_ROUTE_CUDA_API
#elif defined(HYLE_COMPUTE_ROUTE_CUDA_BUILD)
#define HYLE_COMPUTE_ROUTE_CUDA_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_ROUTE_CUDA_API HYLE_IMPORT
#endif

namespace hyle::compute::cuda {

struct DefaultContextOptions final {
  std::size_t device_ordinal = 0;
  // Borrowed through every derived Queue, artifact, invocation and Completion.
  mem::AllocatorRef metadata_allocator{};
};

#if defined(HYLE_COMPUTE_CUDA)
// The route's persistent signed group128 Q4 carrier and cold packing owner.
// FP16 activation/output, K % 128 == 0, N % 64 == 0, M in [1,64], SM80+.
using QuantizedWeights = marlin::QuantizedWeights;

// Fixed development rules: dense FP16/FP32 MatMul uses cuBLASLt; the exact Q4
// carrier uses Marlin. Packed FP16/FP32 SwiGLU uses pinned precise SGLang for
// complete 16-byte vectors (hidden % 8/4 == 0), with 16-byte tensor alignment.
// Separate inputs and non-vector tails use HYLE CUDA kernels.
// Prepare binds these implementations once. Unsupported domains are errors;
// no timing search, CPU fallback, or caller-supplied Provider order is needed.
// Native MemoryResource and stream interop live in compute/backends/cuda.h.
HYLE_COMPUTE_ROUTE_CUDA_API Result<Context>
create_context(DefaultContextOptions options = {}) noexcept;
#endif

} // namespace hyle::compute::cuda
