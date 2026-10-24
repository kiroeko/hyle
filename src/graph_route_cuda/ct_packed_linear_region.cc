#include "ct_packed_linear_region.h"

#if defined(HYLE_COMPUTE_TORCH_CUDA)
#include "../graph/artifact_packed_region.h"
#include "../graph/static_fp8_region.h"
#include "../graph/mapped_pattern_region.h"
#include "hyle/compute/providers/torch_cuda.h"

namespace hyle::graph::cuda::internal {
namespace {
const compute::detail::ProviderVTable &original() noexcept {
  return *compute::detail::ProviderAccess::vtable(compute::torch_cuda::provider());
}
}
compute::Provider ct_packed_linear_provider() noexcept {
  return graph::detail::MappedPatternRegions<&original,
      graph::detail::ArtifactPackedPattern, graph::detail::StaticFloat8Pattern>::decorate(compute::torch_cuda::provider());
}
} // namespace hyle::graph::cuda::internal
#else
namespace hyle::graph::cuda::internal {
compute::Provider ct_packed_linear_provider() noexcept { return {}; }
} // namespace hyle::graph::cuda::internal
#endif
