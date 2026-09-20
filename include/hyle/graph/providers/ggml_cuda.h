#pragma once

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_PROVIDER_GGML_CUDA_API
#elif defined(HYLE_GRAPH_PROVIDER_GGML_CUDA_BUILD)
#define HYLE_GRAPH_PROVIDER_GGML_CUDA_API HYLE_EXPORT
#else
#define HYLE_GRAPH_PROVIDER_GGML_CUDA_API HYLE_IMPORT
#endif

namespace hyle::graph::ggml_cuda {

// The Compute provider plus Graph region discovery/lowering. Original ggml
// owns fusion and native execution inside each admitted region. Resources,
// completion and failure retirement are identical to the Direct path.
#if defined(HYLE_COMPUTE_GGML_CUDA)
HYLE_GRAPH_PROVIDER_GGML_CUDA_API compute::Provider provider() noexcept;
#endif

} // namespace hyle::graph::ggml_cuda
