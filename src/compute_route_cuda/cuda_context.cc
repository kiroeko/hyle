#include "default_context.h"
#include "hyle/compute/cuda.h"

#include <array>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/flashinfer.h"
#include "hyle/compute/providers/ggml_cuda.h"

namespace hyle::compute::cuda {

Result<Context> create_context(DefaultContextOptions options) noexcept {
  return internal::create_context(options, ggml_cuda::provider());
}

Result<Context> internal::create_context(DefaultContextOptions options,
                                         Provider ggml) noexcept {
  auto selected = device(options.device_ordinal);
  if (!selected)
    return std::unexpected(std::move(selected).error());
  const std::array providers{
#if defined(HYLE_COMPUTE_FLASHINFER)
      ProviderBinding{flashinfer::provider(), 100},
#endif
      ProviderBinding{ggml, 200},
      ProviderBinding{cublas::provider(), 100},
      ProviderBinding{cuda_native::sglang_provider(), 100},
      ProviderBinding{cuda_native::deepseek_provider(), 100},
      ProviderBinding{cuda_native::cccl_provider(), 100},
      ProviderBinding{cuda_native::provider(), 1000}};
  return Context::create(*selected,
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = providers});
}

} // namespace hyle::compute::cuda
