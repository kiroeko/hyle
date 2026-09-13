#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/detail/resource_binding.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"

namespace hyle::compute::detail {

struct ContextState;

struct ProviderOperationProposal final {
  const ContextState *context = nullptr;
  ProviderIdentity provider{};
  std::size_t provider_index = 0;
  std::uint64_t route = 0;
  ProviderRuntimeProperties runtime{};
  ResourceBindingRequirement workspace_resource{};
  std::optional<WorkspaceRequirement> workspace{};
  std::optional<std::size_t> persistent_bytes{};
};

struct ProviderPrepareRequest final {
  Operation operation{};
  std::span<const TensorDesc> inputs{};
  std::span<const TensorDesc> outputs{};
  std::span<const mem::MemoryResourceRef> input_resources{};
  std::span<const mem::MemoryResourceRef> output_resources{};
  mem::MemoryResourceRef workspace_resource{};
  std::uint64_t route = 0;
};

struct ProviderPreparedVTable final {
  void (*destroy_prepared)(void *state,
                           mem::AllocatorRef allocator) noexcept = nullptr;
  AliasSupport (*alias_support)(void *prepared, std::size_t output_index,
                                std::size_t input_index) noexcept = nullptr;
  Status (*create_invocation)(void *prepared, mem::AllocatorRef allocator,
                              mem::BufferView workspace,
                              void **invocation) noexcept = nullptr;
  void (*destroy_invocation)(void *invocation,
                             mem::AllocatorRef allocator) noexcept = nullptr;
  // Exactly one submit callback is present, selected by runtime.mode.
  // shared_submit records into an already-open HYLE Queue sequence and never
  // creates a Completion. delegated_submit owns its independent endpoint and
  // returns its one real Completion; it must consume dependencies without a
  // hidden host wait or return Unimplemented before accepting work.
  EnqueueResult (*shared_submit)(
      void *prepared, void *invocation, Queue &queue,
      std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) noexcept = nullptr;
  Result<Completion> (*delegated_submit)(
      void *prepared, void *invocation,
      std::span<const CompletionDependency> dependencies,
      std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) noexcept = nullptr;
  // Delegated-only cold endpoint contract. Graph calls endpoint() after
  // invocation creation and asks can_import_endpoint() for every dependency
  // edge before replay. A true answer commits delegated_submit() to consuming
  // the matching CompletionDependency without a hidden host wait.
  CompletionEndpoint (*endpoint)(void *prepared,
                                 void *invocation) noexcept = nullptr;
  bool (*can_import_endpoint)(void *prepared, void *invocation,
                              CompletionEndpoint source) noexcept = nullptr;
};

struct ProviderPrepared final {
  const ProviderPreparedVTable *vtable = nullptr;
  void *state = nullptr;
  WorkspaceRequirement workspace{};
  ProviderRuntimeProperties runtime{};
  std::optional<std::size_t> persistent_bytes{};
};

struct ProviderVTable final {
  ProviderIdentity identity{};
  Status (*create_context)(const Provider &provider, Device device,
                           void *backend_context, mem::AllocatorRef allocator,
                           void **context) noexcept = nullptr;
  void (*destroy_context)(void *context,
                          mem::AllocatorRef allocator) noexcept = nullptr;
  // Cold, allocation-free capability/resource proposal. Caller supplies one
  // requirement slot per input/output. Unimplemented declines this route.
  Status (*propose)(void *context, Operation operation,
                    std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    std::span<ResourceBindingRequirement> input_requirements,
                    std::span<ResourceBindingRequirement> output_requirements,
                    ProviderOperationProposal *proposal) noexcept = nullptr;
  // Compile only an already-selected exact resource route.
  Status (*prepare)(void *context, const ProviderPrepareRequest &request,
                    mem::AllocatorRef allocator,
                    ProviderPrepared *prepared) noexcept = nullptr;
  // Provider-owned work accepted before this call must reach terminal state.
  // It is required even for shared providers, where an O(1) no-op is normal.
  Status (*drain_context)(void *context) noexcept = nullptr;
  // Optional object-local extension lookup. A successful query returns a
  // static interface table for the exact requested ID/version; nullptr means
  // unsupported. The returned table must remain valid while this provider
  // implementation is loaded.
  const void *(*query_extension)(void *context,
                                 ProviderExtensionId id) noexcept = nullptr;
};

// Cross-factor cold preparation seam used by Graph. It exposes proposals for
// one configured provider without leaking provider state or retaining caller
// spans. Exact preparation revalidates the proposal identity and resources.
struct OperationPreparationAccess final {
  // Allocation-free planner predicate for a proposal's provider-neutral
  // resource requirement. Backend accessibility remains a separate
  // backend accessibility check; exact prepare revalidates both.
  HYLE_COMPUTE_API static Status
  match_resource(const ResourceBindingRequirement &requirement,
                 mem::MemoryResourceRef resource) noexcept;

  HYLE_COMPUTE_API static Result<ProviderOperationProposal>
  propose(const Context &context, std::size_t provider_index,
          Operation operation, std::span<const TensorDesc> inputs,
          std::span<const TensorDesc> outputs,
          std::span<ResourceBindingRequirement> input_requirements,
          std::span<ResourceBindingRequirement> output_requirements) noexcept;

  HYLE_COMPUTE_API static Result<PreparedOperation>
  prepare(const Context &context, const ProviderOperationProposal &proposal,
          Operation operation, std::span<const TensorDesc> inputs,
          std::span<const TensorDesc> outputs,
          std::span<const ResourceBindingRequirement> input_requirements,
          std::span<const ResourceBindingRequirement> output_requirements,
          OperationResourceBindings resources) noexcept;
};

struct ProviderAccess final {
  template <class Payload>
    requires(std::is_trivially_copyable_v<Payload> &&
             sizeof(Payload) <= Provider::kInlineBytes &&
             alignof(Payload) <= alignof(void *))
  static Provider make(const ProviderVTable *vtable,
                       const Payload &payload) noexcept {
    Provider result;
    result.vtable_ = vtable;
    std::memcpy(result.payload_.data(), &payload, sizeof(Payload));
    return result;
  }

  template <class Payload>
    requires(std::is_trivially_copyable_v<Payload> &&
             sizeof(Payload) <= Provider::kInlineBytes &&
             alignof(Payload) <= alignof(void *))
  static Payload payload(const Provider &provider) noexcept {
    Payload result{};
    std::memcpy(&result, provider.payload_.data(), sizeof(Payload));
    return result;
  }

  static const ProviderVTable *vtable(const Provider &provider) noexcept {
    return static_cast<const ProviderVTable *>(provider.vtable_);
  }
};

} // namespace hyle::compute::detail
