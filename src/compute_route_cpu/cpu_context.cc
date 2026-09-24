#include "hyle/compute/cpu.h"

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
  const std::array providers{
#if defined(HYLE_COMPUTE_ONEDNN)
      ProviderBinding{onednn::provider({.executor = options.executor}), 100},
#endif
#if defined(HYLE_COMPUTE_GGML_CPU)
      ProviderBinding{
          ggml_cpu::provider({.executor = options.executor,
                              .native_threads = options.native_threads}),
          200},
#endif
      ProviderBinding{native::provider({.executor = options.executor}), 1000}};
  return Context::create(device(),
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = providers});
}

} // namespace hyle::compute::cpu
