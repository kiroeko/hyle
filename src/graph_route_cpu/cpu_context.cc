#include "hyle/graph/cpu.h"
#include "../compute_route_cpu/default_policy.h"

#include "../compute_provider_native/default_rules.h"
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
  using compute::cpu::internal::DefaultProvider;
  const bool single_thread = options.native_threads == 1 &&
                             options.executor.concurrency() <= 1;
  const auto configured = [single_thread](compute::Provider provider, DefaultProvider family) {
    return compute::cpu::internal::with_defaults(
        std::move(provider), family, compute::cpu::internal::DefaultScope::kGraph,
        single_thread);
  };
  std::array providers{
#if defined(HYLE_COMPUTE_GGML_CPU)
      compute::ProviderBinding{
          internal::ggml_regions({.executor = options.executor,
                                  .native_threads = options.native_threads}),
          100},
#endif
#if defined(HYLE_GRAPH_ONEDNN)
      compute::ProviderBinding{onednn::provider({.executor = options.executor}),
                               150},
#endif
#if defined(HYLE_COMPUTE_ONEDNN)
      compute::ProviderBinding{
          configured(compute::onednn::provider({.executor = options.executor}),
                     DefaultProvider::kOneDnn), 900},
#endif
#if defined(HYLE_COMPUTE_GGML_CPU)
      compute::ProviderBinding{configured(compute::ggml_cpu::provider(
                                   {.executor = options.executor,
                                    .native_threads = options.native_threads}),
                                   DefaultProvider::kGgml), 900},
#endif
      compute::ProviderBinding{configured(compute::aten_cpu::provider(),
                                          DefaultProvider::kAten), 900},
      compute::ProviderBinding{
          compute::native::internal::default_provider(
              {.executor = options.executor},
              compute::native::internal::DefaultRuleScope::kGraph,
              options.native_threads == 1),
          1000}};
  auto count = providers.size();
  if (!providers[count - 2].provider.valid()) {
    providers[count - 2] = std::move(providers[count - 1]);
    --count;
  }
  return compute::Context::create(
      compute::cpu::device(), {.metadata_allocator = options.metadata_allocator,
                               .providers = std::span(providers).first(count)});
}

} // namespace hyle::graph::cpu
