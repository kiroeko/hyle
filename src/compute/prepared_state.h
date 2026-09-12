#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "context_state.h"
#include "hyle/util/ref_counted.h"
#include "native_route.h"
#include "provider_internal.h"

namespace hyle::compute::detail {

struct PreparedState final : util::RefCounted<PreparedState> {
  PreparedState(ContextState *context_value, Operation operation_value,
                std::span<const TensorDesc> input_values,
                TensorDesc output_value, ProviderKind provider_value,
                bool fallback_value) noexcept
      : allocator(context_value->allocator), context(context_value),
        operation(operation_value), output(output_value),
        provider(provider_value), fallback(fallback_value),
        input_count(static_cast<std::uint8_t>(input_values.size())) {
    for (std::size_t index = 0; index < input_values.size(); ++index)
      inputs[index] = input_values[index];
    detail::retain(context);
  }

  void destroy() noexcept {
    if (provider_prepared.vtable != nullptr &&
        provider_prepared.vtable->destroy_prepared != nullptr) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 allocator);
    }
    ContextState *retained_context = context;
    const mem::AllocatorRef owner = allocator;
    this->~PreparedState();
    detail::release(retained_context);
    owner.deallocate(this, sizeof(PreparedState), alignof(PreparedState));
  }

  mem::AllocatorRef allocator{};
  ContextState *context = nullptr;
  Operation operation{RmsNorm{}};
  std::array<TensorDesc, 2> inputs{};
  TensorDesc output{};
  NativeParameters parameters{};
  float scalar = 0.0F;
  ProviderKind provider = ProviderKind::kUnknown;
  bool fallback = false;
  std::uint8_t input_count = 0;
  NativeRoute native_route = nullptr;
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

} // namespace hyle::compute::detail
