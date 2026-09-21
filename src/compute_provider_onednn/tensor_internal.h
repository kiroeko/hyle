#pragma once

#include <span>

#include "hyle/compute/visibility.h"
#include "provider_internal.h"

namespace hyle::compute::onednn::internal {

struct TensorRequest final {
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  TensorDesc output{};
};

HYLE_COMPUTE_PROVIDER_ONEDNN_API Status validate_tensor(
    const TensorRequest &request, ParallelExecutorRef executor) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API ProviderPrepareOutcome prepare_tensor(
    const TensorRequest &request, ParallelExecutorRef executor,
    mem::AllocatorRef allocator, detail::ProviderPrepared *prepared) noexcept;

} // namespace hyle::compute::onednn::internal
