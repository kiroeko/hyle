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
using LinearTransformRoute = void (*)(const void *source, void *destination,
                                      std::size_t count) noexcept;
using ScalarTransformRoute = void (*)(const std::byte *source,
                                      std::byte *destination) noexcept;

// Cold prepared metadata shared by the native operation families. A family
// selects one submit route during prepare; submit never searches by kind or
// grows metadata. The compact scalar/parameter slots hold only route-specific
// immutable values, not benchmark-driven per-tensor state.
struct NativePlan final {
  detail::NativeParameters parameters{};
  float scalar = 0.0F;
  std::uint64_t scalar_bits = 0;
  detail::NativeRoute route = nullptr;
  LinearTransformRoute linear_transform_route = nullptr;
  ScalarTransformRoute scalar_transform_route = nullptr;
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

// TensorDesc construction already proved that every non-negative stride and
// the complete byte extent fit size_t. This bounded-rank helper therefore
// needs no checked arithmetic in the steady reference loop.
inline std::size_t logical_byte_offset(const TensorDesc &description,
                                       std::size_t linear_index) noexcept {
  std::size_t offset = 0;
  const auto dimensions = description.dimensions();
  const auto strides = description.byte_strides();
  for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
    const std::size_t axis = dimensions.size() - 1 - reverse;
    const std::size_t extent = static_cast<std::size_t>(dimensions[axis]);
    const std::size_t coordinate = linear_index % extent;
    linear_index /= extent;
    offset += coordinate * static_cast<std::size_t>(strides[axis]);
  }
  return offset;
}

Status prepare_transform(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept;
Status prepare_cast(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    NativePlan &plan) noexcept;

} // namespace hyle::compute::native::internal
