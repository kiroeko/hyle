#pragma once

#include <cstdint>
#include <vector>

#include "hyle/compute/providers/ggml_cuda.h"
#include "provider_internal.h"

namespace hyle::compute::ggml_cuda::internal {

inline constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kProvider,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false};
inline constexpr detail::ResourceBindingRequirement kBinding{
    .required_properties = mem::MemoryProperty::kDeviceAccessible,
    .minimum_alignment = 256};

inline std::size_t boundary_alignment(Operation operation,
                                      DataType type) noexcept {
  // Original scalar copy/conversion kernels accept typed subviews. Other
  // native programs keep the alignment of their qualified vectorized paths.
  return operation.get_if<Copy>() || operation.get_if<Cast>() ||
                 operation.get_if<Concat>()
             ? data_type_size(type)
             : kBinding.minimum_alignment;
}

// Private value identities are recipe-local. No Graph types, borrowed graph
// spans or caller tensor addresses are retained by the immutable program.
struct Node final {
  Operation operation;
  std::vector<std::uint32_t> inputs;
  std::uint32_t output = 0;
};
struct Recipe final {
  std::vector<TensorDesc> values;
  std::vector<Node> nodes;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};

HYLE_COMPUTE_PROVIDER_GGML_CUDA_API bool
supports(void *context, Operation, std::span<const TensorDesc>,
         std::span<const TensorDesc>) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API ProviderPrepareOutcome
compile(void *context, Recipe recipe, mem::AllocatorRef,
        detail::ProviderPrepared *) noexcept;
// Same-build Graph adapter supplies its static extension table. It reuses
// the identical Compute context, compiler and per-invocation lifecycle.
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Provider
with_extension(ProviderExtensionId id, const void *table) noexcept;

} // namespace hyle::compute::ggml_cuda::internal
