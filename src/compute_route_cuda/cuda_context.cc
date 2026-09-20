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
      flashinfer::provider(),
#endif
      ggml,
      cublas::provider(),
      marlin::provider(),
      cuda_native::sglang_provider(),
      cuda_native::cccl_provider(),
      cuda_native::provider()};
  return Context::create(*selected,
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = providers});
}

} // namespace hyle::compute::cuda
