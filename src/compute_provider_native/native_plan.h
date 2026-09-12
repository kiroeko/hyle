#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "../compute/native_route.h"
#include "hyle/compute/prepared_operation.h"

namespace hyle::compute::native::internal {

struct NativePlan;
using SubmitRoute = Result<Completion> (*)(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept;

// Cold prepared metadata shared by the native operation families. A family
// selects one submit route during prepare; submit never searches by kind or
// grows metadata. The compact scalar/parameter slots hold only route-specific
// immutable values, not benchmark-driven per-tensor state.
struct NativePlan final {
  detail::NativeParameters parameters{};
  float scalar = 0.0F;
  std::uint64_t scalar_bits = 0;
  detail::NativeRoute route = nullptr;
  SubmitRoute submit_route = nullptr;
};

inline void preserve_first_error(Status &terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok())
    terminal = std::move(candidate);
}

inline Result<Completion> completed(Status status) noexcept {
  return Completion::completed(std::move(status));
}

inline bool aligned_for_storage(const void *address,
                                std::size_t alignment) noexcept {
  return address == nullptr ||
         (alignment != 0 &&
          (reinterpret_cast<std::uintptr_t>(address) & (alignment - 1)) == 0);
}

Status prepare_transform(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept;

} // namespace hyle::compute::native::internal
