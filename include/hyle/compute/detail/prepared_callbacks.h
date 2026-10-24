#pragma once

#include <cstddef>
#include <span>

#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"

namespace hyle::compute::detail {

// Same-build erasure used by typed custom kernels and built-in providers.
// Consumers implement C++ kernel types, not this table. No context creation,
// capability discovery, partition or vendor-provider SPI is exposed here.
struct ProviderPreparedVTable final {
  void (*destroy_prepared)(void *state,
                           mem::AllocatorRef allocator) noexcept = nullptr;
  AliasSupport (*alias_support)(void *prepared, std::size_t output_index,
                                std::size_t input_index) noexcept = nullptr;
  // On failure return nullptr or a partial state destroyable with the same
  // allocator. The caller cleans a non-null partial before propagating error.
  Status (*create_invocation)(void *prepared, mem::AllocatorRef allocator,
                              mem::BufferView workspace,
                              void **invocation) noexcept = nullptr;
  void (*destroy_invocation)(void *invocation,
                             mem::AllocatorRef allocator) noexcept = nullptr;
  // Shared callbacks record inside the original admitted sequence, never create another
  // Completion. Delegated callbacks instead consume dependencies and return
  // the independent runtime's one real Completion. Exactly one is present.
  EnqueueResult (*shared_submit)(void *prepared, void *invocation, QueueSequence &sequence,
                                 std::span<const ConstTensorView> inputs,
                                 std::span<const TensorView> outputs) noexcept =
      nullptr;
  Result<Completion> (*delegated_submit)(
      void *prepared, void *invocation,
      std::span<const CompletionDependency> dependencies,
      std::span<const ConstTensorView> inputs,
      std::span<const TensorView> outputs) noexcept = nullptr;
  // Delegated-only cold contract. A true import answer commits submit to
  // consuming that endpoint's dependencies without an undeclared host wait.
  CompletionEndpoint (*endpoint)(void *prepared,
                                 void *invocation) noexcept = nullptr;
  bool (*can_import_endpoint)(void *prepared, void *invocation,
                              CompletionEndpoint source) noexcept = nullptr;
  // Optional cold binding/resource preparation for the original invocation.
  // It may allocate owned resources, but never reads/writes tensor payloads,
  // accepts device work, or creates a Completion. Spans are borrowed only for
  // this call. Failure leaves the slot retryable; repeated calls for the same
  // bindings must not duplicate retained owners or invalidate a prior success.
  Status (*prepare_bindings)(void *prepared, void *invocation,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept = nullptr;
};

} // namespace hyle::compute::detail
