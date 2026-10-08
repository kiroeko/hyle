#pragma once

#include "hyle/compute/provider.h"
#include "hyle/compute/cuda.h"
#include "default_context.h"

namespace hyle::compute::cuda::internal {

enum class DefaultProvider : std::uint8_t {
  kCublas = 1, kGgml, kTorch, kSglang, kDeepSeek, kCccl, kFlashInfer, kNative
};

// Same-build cold decoration only. Original capability/resource admission and
// all prepared execution tables remain in their route-native providers.
HYLE_COMPUTE_ROUTE_CUDA_API Provider
with_defaults(Provider, DefaultProvider, DefaultScope) noexcept;

} // namespace hyle::compute::cuda::internal
