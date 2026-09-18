#pragma once

#include <span>

#include "hyle/compute/visibility.h"
#include "provider_internal.h"

namespace hyle::compute::onednn::internal {

inline constexpr std::size_t kNormMaxPostOps = 4;
inline constexpr std::size_t kNormMaxInputs = 3 + kNormMaxPostOps;
inline constexpr std::size_t kNormWorkspaceAlignment = 64;
inline constexpr ProviderRuntimeProperties kNormRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

struct NormPostOp final {
  BinaryKind kind = BinaryKind::kAdd;
  std::size_t input = 0;
};

// Cold, uninstalled Compute/Graph seam. No vendor object or borrowed request
// escapes prepare. The first norm_inputs entries belong to the norm itself.
struct NormRequest final {
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  TensorDesc output{};
  std::size_t norm_inputs = 0;
  std::span<const NormPostOp> post_ops{};
};

HYLE_COMPUTE_PROVIDER_ONEDNN_API Status validate_normalization(
    const NormRequest &request, ParallelExecutorRef executor) noexcept;

HYLE_COMPUTE_PROVIDER_ONEDNN_API ProviderPrepareOutcome prepare_normalization(
    const NormRequest &request, ParallelExecutorRef executor,
    mem::AllocatorRef allocator, detail::ProviderPrepared *prepared) noexcept;

} // namespace hyle::compute::onednn::internal
