#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <span>

#include "context_state.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/util/ref_counted.h"
#include "provider_internal.h"

namespace hyle::compute::detail {

struct PreparedState final : util::RefCounted<PreparedState> {
  PreparedState(ContextState *context_value, Operation operation_value,
                ProviderIdentity provider_value,
                std::size_t allocation_size_value,
                std::size_t descriptor_offset_value,
                std::size_t resource_offset_value,
                std::size_t input_count_value,
                std::size_t output_count_value,
                mem::MemoryResourceRef workspace_resource_value) noexcept
      : allocator(context_value->allocator), context(context_value),
        operation(operation_value), provider(provider_value),
        allocation_size(allocation_size_value),
        descriptor_offset(descriptor_offset_value),
        resource_offset(resource_offset_value),
        input_count(input_count_value), output_count(output_count_value) {
    workspace_resource = workspace_resource_value;
    detail::retain(context);
  }

  TensorDesc *inputs() noexcept {
    auto *bytes = reinterpret_cast<std::byte *>(this);
    return reinterpret_cast<TensorDesc *>(bytes + descriptor_offset);
  }
  const TensorDesc *inputs() const noexcept {
    auto *bytes = reinterpret_cast<const std::byte *>(this);
    return reinterpret_cast<const TensorDesc *>(bytes + descriptor_offset);
  }
  TensorDesc *outputs() noexcept { return inputs() + input_count; }
  const TensorDesc *outputs() const noexcept { return inputs() + input_count; }
  mem::MemoryResourceRef *input_resources() noexcept {
    auto *bytes = reinterpret_cast<std::byte *>(this);
    return reinterpret_cast<mem::MemoryResourceRef *>(bytes + resource_offset);
  }
  const mem::MemoryResourceRef *input_resources() const noexcept {
    auto *bytes = reinterpret_cast<const std::byte *>(this);
    return reinterpret_cast<const mem::MemoryResourceRef *>(bytes +
                                                            resource_offset);
  }
  mem::MemoryResourceRef *output_resources() noexcept {
    return input_resources() + input_count;
  }
  const mem::MemoryResourceRef *output_resources() const noexcept {
    return input_resources() + input_count;
  }

  void destroy() noexcept {
    if (provider_prepared.vtable != nullptr &&
        provider_prepared.vtable->destroy_prepared != nullptr) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 allocator);
    }
    for (std::size_t index = output_count; index != 0; --index)
      std::destroy_at(outputs() + index - 1);
    for (std::size_t index = input_count; index != 0; --index)
      std::destroy_at(inputs() + index - 1);
    ContextState *retained_context = context;
    const mem::AllocatorRef owner = allocator;
    const std::size_t bytes = allocation_size;
    this->~PreparedState();
    detail::release(retained_context);
    owner.deallocate(this, bytes, alignof(PreparedState));
  }

  mem::AllocatorRef allocator{};
  ContextState *context = nullptr;
  Operation operation{};
  ProviderIdentity provider{};
  ProviderInstanceId provider_instance{};
  std::size_t allocation_size = 0;
  std::size_t descriptor_offset = 0;
  std::size_t resource_offset = 0;
  std::size_t input_count = 0;
  std::size_t output_count = 0;
  mem::MemoryResourceRef workspace_resource{};
  ProviderPrepared provider_prepared{};
};

inline void retain(PreparedState *state) noexcept {
  if (state != nullptr)
    state->incref();
}
inline void release(PreparedState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

struct InvocationState final : util::RefCounted<InvocationState> {
  InvocationState(PreparedState *prepared_value,
                  void *provider_state_value) noexcept
      : prepared(prepared_value), provider_state(provider_state_value) {
    detail::retain(prepared);
  }

  void destroy() noexcept {
    PreparedState *owner_prepared = prepared;
    const mem::AllocatorRef allocator = owner_prepared->allocator;
    if (owner_prepared->provider_prepared.vtable != nullptr &&
        owner_prepared->provider_prepared.vtable->destroy_invocation !=
            nullptr) {
      owner_prepared->provider_prepared.vtable->destroy_invocation(
          provider_state, allocator);
    }
    this->~InvocationState();
    allocator.deallocate(this, sizeof(InvocationState),
                         alignof(InvocationState));
    detail::release(owner_prepared);
  }

  PreparedState *prepared = nullptr;
  void *provider_state = nullptr;
  CompletionLifetime completion_lifetime{};
  std::atomic<bool> in_flight{false};
};

inline void retain(InvocationState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(InvocationState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

} // namespace hyle::compute::detail
