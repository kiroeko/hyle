#include "hyle/graph/cpu.h"

#include <array>

#if defined(HYLE_COMPUTE_GGML_CPU)
#include "ggml_region.h"
#endif
#include "hyle/compute/providers/native.h"
#if defined(HYLE_COMPUTE_ONEDNN)
#include "hyle/compute/providers/onednn.h"
#endif
#if defined(HYLE_GRAPH_ONEDNN)
#include "hyle/graph/providers/onednn.h"
#endif

namespace hyle::graph::cpu {

Result<compute::Context>
create_context(compute::cpu::DefaultContextOptions options) noexcept {
  const std::array providers{
#if defined(HYLE_COMPUTE_GGML_CPU)
      internal::ggml_regions({.executor = options.executor,
                              .native_threads = options.native_threads}),
#endif
#if defined(HYLE_GRAPH_ONEDNN)
      onednn::provider({.executor = options.executor}),
#endif
#if defined(HYLE_COMPUTE_ONEDNN)
      compute::onednn::provider({.executor = options.executor}),
#endif
#if defined(HYLE_COMPUTE_GGML_CPU)
      compute::ggml_cpu::provider({.executor = options.executor,
                                   .native_threads = options.native_threads}),
#endif
      compute::native::provider({.executor = options.executor})};
  return compute::Context::create(
      compute::cpu::device(), {.metadata_allocator = options.metadata_allocator,
                               .providers = providers});
}

} // namespace hyle::graph::cpu
