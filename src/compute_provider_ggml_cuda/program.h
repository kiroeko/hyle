#pragma once

#include <cstdint>
#include <vector>

#include "hyle/compute/providers/ggml_cuda.h"
#include "provider_internal.h"

namespace hyle::compute::ggml_cuda::internal {

inline constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    // Proposal upper bound; compile refines this with the original native
    // graph compatibility check for the actual lowered program.
    .capture_compatible = true};
inline constexpr detail::ResourceBindingRequirement kBinding{
    .required_properties = mem::MemoryProperty::kDeviceAccessible,
    .minimum_alignment = 256};

inline std::size_t boundary_alignment(Operation operation,
                                      DataType type) noexcept {
  // F32 Add uses typed scalar loads/stores, including native multi-Add and
  // fused bias readers. A dense row slice need not begin on an allocation
  // boundary. Other readers of the same value retain their own requirements.
  if (const auto *binary = operation.get_if<Binary>();
      binary && binary->kind == BinaryKind::kAdd && type == DataType::kFloat32)
    return sizeof(float);
  // Original scalar copy/conversion kernels accept typed subviews. Other
  // native programs keep the alignment of their qualified vectorized paths.
  // GDN's F32 scalar loads also accept per-token gate subviews (a token's
  // scalar decay/beta is often only four bytes past the preceding token).
  return operation.get_if<Copy>() || operation.get_if<Cast>() ||
                 operation.get_if<Concat>() || operation.get_if<GatedDeltaNet>()
             ? data_type_size(type)
             : kBinding.minimum_alignment;
}

// Private value identities are recipe-local. No Graph types, borrowed graph
// spans or caller tensor addresses are retained by the immutable program.
struct Node final {
  Operation operation;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  // Trusted cold Graph lowering only. GDN replaces its final dense state by
  // the original pool and private source/destination IDs. CausalConv uses
  // [chunk, weights, history pool, source IDs, destination IDs], retaining
  // original sample precision for history and F32 convolution mathematics.
  // Destination IDs are batch-major [B*K]; only Y is a tensor output. Actual
  // pool permissions belong to the outer Graph effect, never these integers.
  bool indexed_state = false;
};
struct Recipe final {
  struct Constant final {
    std::uint32_t value;
    ConstTensorView binding;
  };
  std::vector<TensorDesc> values;
  std::vector<Node> nodes;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  // Graph's immutable artifact owns these bindings through region teardown.
  // Ordinary Compute inputs remain rebindable and never enter this list.
  std::vector<Constant> constants;
  // Native values whose original external effects must execute even when no
  // tensor escapes. These are not published outputs or extra execution work.
  std::vector<std::uint32_t> effect_roots;
};

// Cold constant preparation records into the caller's original sequence. The
// returned owner retains native preparation state through actual retirement,
// including partial preparation failures. The caller retains source,
// destination, allocator and context until it releases that owner.
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Status
pack_matrix_constant(void *context, ConstTensorView source,
                     TensorView destination, mem::AllocatorRef,
                     detail::QueueSequence &,
                     detail::CompletionLifetime *retained_native) noexcept;
// A nonempty cold program has only immutable ready constants, no rebindable
// inputs. Destinations match recipe.outputs in order. The caller owns all
// bindings/resources/context through retained_native's actual retirement.
// Internally these fixed roots use the original prepared input boundary,
// including checked-index validation; their binding span is episode-owned.
// An already-ready region with no mathematical work needs no native program.
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Status
evaluate_constant_recipe(void *context, Recipe recipe,
                         std::span<const TensorView> destinations,
                         mem::AllocatorRef, detail::QueueSequence &,
                         detail::CompletionLifetime *retained_native) noexcept;

HYLE_COMPUTE_PROVIDER_GGML_CUDA_API bool
supports_bfloat_paired_projection(void *context, std::uint64_t inner,
                                 std::uint64_t rows) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API bool
supports(void *context, Operation, std::span<const TensorDesc>,
         std::span<const TensorDesc>) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API bool
supports_indexed_state(void *context, const Node &,
                       std::span<const TensorDesc> values) noexcept;
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API ProviderPrepareOutcome
compile(void *context, Recipe recipe, mem::AllocatorRef,
        detail::ProviderPrepared *) noexcept;
// Same-build Graph adapter supplies its static extension table. It reuses
// the identical Compute context, compiler and per-invocation lifecycle.
HYLE_COMPUTE_PROVIDER_GGML_CUDA_API Provider
with_extension(ProviderExtensionId id, const void *table) noexcept;

} // namespace hyle::compute::ggml_cuda::internal
