#include "hyle/compute/cpu.h"

#include "../compute_provider_native/default_rules.h"
#include <array>

#include "hyle/compute/providers/native.h"
#if defined(HYLE_COMPUTE_GGML_CPU)
#include "hyle/compute/providers/ggml_cpu.h"
#endif
#if defined(HYLE_COMPUTE_ONEDNN)
#include "hyle/compute/providers/onednn.h"
#endif

namespace hyle::compute::cpu {

Result<Context> create_context(DefaultContextOptions options) noexcept {
  std::array providers{
#if defined(HYLE_COMPUTE_ONEDNN)
      ProviderBinding{onednn::provider({.executor = options.executor}), 100},
#endif
#if defined(HYLE_COMPUTE_GGML_CPU)
      ProviderBinding{
          ggml_cpu::provider({.executor = options.executor,
                              .native_threads = options.native_threads}),
          200},
#endif
      ProviderBinding{aten_cpu::provider(), 350},
      ProviderBinding{native::internal::default_provider(
                          {.executor = options.executor},
                          native::internal::DefaultRuleScope::kDirect,
                          options.native_threads == 1),
                      1000}};
  auto count = providers.size();
  if (!providers[count - 2].provider.valid()) {
    providers[count - 2] = std::move(providers[count - 1]);
    --count;
  }
  return Context::create(device(),
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = std::span(providers).first(count)});
}

} // namespace hyle::compute::cpu
