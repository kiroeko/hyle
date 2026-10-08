#include "default_context.h"
#include "default_policy.h"
#include "best_of.h"
#include "hyle/compute/cuda.h"

#include <array>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/flashinfer.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/providers/torch_cuda.h"

namespace hyle::compute::cuda {

Result<Context> create_context(DefaultContextOptions options) noexcept {
  return internal::create_context(options, ggml_cuda::provider());
}

Result<Context> internal::create_context(DefaultContextOptions options,
                                         Provider ggml, DefaultScope scope) noexcept {
  auto selected = device(options.device_ordinal);
  if (!selected)
    return std::unexpected(std::move(selected).error());
  using internal::DefaultProvider;
  const auto configured = [scope](Provider provider, DefaultProvider family) {
    return internal::with_defaults(std::move(provider), family, scope);
  };
  const auto native = cuda_native::provider();
  const auto qualified = ggml.identity().version ==
          "456172ec+iq1-46cbf0e9-resource-f99368c68335266b"
      ? internal::with_qualified_defaults(native, scope) : Provider{};
  const std::array providers{
      ProviderBinding{qualified, 900},
#if defined(HYLE_COMPUTE_FLASHINFER)
      ProviderBinding{configured(flashinfer::provider(), DefaultProvider::kFlashInfer), 900},
#endif
      ProviderBinding{configured(ggml, DefaultProvider::kGgml), 900},
#if defined(HYLE_COMPUTE_TORCH_CUDA)
      ProviderBinding{configured(torch_cuda::provider(), DefaultProvider::kTorch), 900},
#endif
      ProviderBinding{configured(cublas::provider(), DefaultProvider::kCublas), 900},
      ProviderBinding{configured(cuda_native::sglang_provider(), DefaultProvider::kSglang), 900},
      ProviderBinding{configured(cuda_native::deepseek_provider(), DefaultProvider::kDeepSeek), 900},
      ProviderBinding{configured(cuda_native::cccl_provider(), DefaultProvider::kCccl), 900},
      ProviderBinding{configured(native, DefaultProvider::kNative), 1000}};
  return Context::create(*selected,
                         {.metadata_allocator = options.metadata_allocator,
                          .providers = std::span(providers).subspan(qualified.valid() ? 0 : 1)});
}

} // namespace hyle::compute::cuda
