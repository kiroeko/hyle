#include "hyle/graph/cpu.h"
#include "../compute_route_cpu/default_policy.h"
#include "hyle/compute/providers/native.h"
#include "ct_packed_linear_region.h"

#if defined(HYLE_COMPUTE_GGML_CPU)
#include "ggml_region.h"
#endif
#if defined(HYLE_GRAPH_ONEDNN)
#include "hyle/graph/providers/onednn.h"
#endif

namespace hyle::graph::cpu {

Result<compute::Context>
create_context(compute::cpu::DefaultContextOptions options) noexcept {
  compute::Provider ggml;
#if defined(HYLE_COMPUTE_GGML_CPU)
  ggml = internal::ggml_regions({.executor = options.executor,
                                 .native_threads = options.native_threads});
#endif
  compute::ProviderBinding region;
#if defined(HYLE_GRAPH_ONEDNN)
  region = {onednn::provider({.executor = options.executor}), 150};
#endif
  return compute::cpu::internal::create_context(
      options, compute::cpu::internal::DefaultScope::kGraph,
      std::move(ggml), std::move(region),
      internal::ct_packed_regions(compute::native::provider(
          {.executor = options.executor})));
}

} // namespace hyle::graph::cpu
