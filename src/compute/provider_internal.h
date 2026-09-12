#pragma once

#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"

namespace hyle::compute::detail {

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
  // Success returns a Completion for the complete ordered Queue prefix through
  // this command.  Destroying an earlier handle cannot cancel that prefix or
  // prevent a later prefix Completion from reporting an earlier terminal
  // backend error.  A synchronous Result error means this command accepted no
  // work.
  Result<Completion> (*submit)(void *prepared, void *invocation, Queue &queue,
                               std::span<const ConstTensorView> inputs,
                               std::span<const TensorView> outputs) noexcept =
      nullptr;
};

struct ProviderPrepared final {
  const ProviderPreparedVTable *vtable = nullptr;
  void *state = nullptr;
  WorkspaceRequirement workspace{};
};

struct ProviderVTable final {
  ProviderIdentity identity{};
  Status (*create_context)(const Provider &provider, Device device,
                           void *backend_context, mem::AllocatorRef allocator,
                           void **context) noexcept = nullptr;
  void (*destroy_context)(void *context,
                          mem::AllocatorRef allocator) noexcept = nullptr;
  Status (*prepare)(void *context, Operation operation,
                    std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    mem::AllocatorRef allocator,
                    ProviderPrepared *prepared) noexcept = nullptr;
  // Optional object-local extension lookup. A successful query returns a
  // static interface table for the exact requested ID/version; nullptr means
  // unsupported. The returned table must remain valid while this provider
  // implementation is loaded.
  const void *(*query_extension)(void *context,
                                 ProviderExtensionId id) noexcept = nullptr;
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
