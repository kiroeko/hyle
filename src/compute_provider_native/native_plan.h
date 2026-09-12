#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "../compute/native_route.h"
#include "hyle/compute/prepared_operation.h"

namespace hyle::compute::native::internal {

struct NativePlan;
enum class ReductionRoute : std::uint8_t {
  kGeneral = 0,
  kAllDense,
  kLastAxisDense,
};
using SubmitRoute = Result<Completion> (*)(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept;
using LinearElementwiseRoute = void (*)(const void *first, const void *second,
                                        const void *third, void *output,
                                        std::size_t count) noexcept;
using ScalarElementwiseRoute = void (*)(const std::byte *first,
                                        const std::byte *second,
                                        const std::byte *third,
                                        std::byte *output) noexcept;

// Cold prepared metadata shared by the native operation families. A family
// selects one submit route during prepare; submit never searches by kind or
// grows metadata. The compact scalar/parameter slots hold only route-specific
// immutable values, not benchmark-driven per-tensor state.
struct NativePlan final {
  detail::NativeParameters parameters{};
  float scalar = 0.0F;
  std::uint64_t scalar_bits = 0;
  ReductionKind reduction_kind = ReductionKind::kSum;
  ReductionRoute reduction_route = ReductionRoute::kGeneral;
  std::uint8_t reduction_axes_mask = 0;
  detail::NativeRoute route = nullptr;
  LinearElementwiseRoute linear_elementwise_route = nullptr;
  ScalarElementwiseRoute scalar_elementwise_route = nullptr;
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

// Map one dense output coordinate back into a broadcast-compatible input.
// The schema has already proved rank and dimension compatibility.
inline std::size_t
logical_broadcast_byte_offset(const TensorDesc &input,
                              std::span<const std::int64_t> output_dimensions,
                              std::size_t linear_index) noexcept {
  std::size_t offset = 0;
  const auto input_dimensions = input.dimensions();
  const auto input_strides = input.byte_strides();
  for (std::size_t reverse = 0; reverse < output_dimensions.size(); ++reverse) {
    const std::size_t output_axis = output_dimensions.size() - 1 - reverse;
    const std::size_t output_extent =
        static_cast<std::size_t>(output_dimensions[output_axis]);
    const std::size_t coordinate = linear_index % output_extent;
    linear_index /= output_extent;
    if (reverse >= input_dimensions.size())
      continue;
    const std::size_t input_axis = input_dimensions.size() - 1 - reverse;
    if (input_dimensions[input_axis] != 1) {
      offset +=
          coordinate * static_cast<std::size_t>(input_strides[input_axis]);
    }
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
Status prepare_elementwise(Operation operation,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           NativePlan &plan) noexcept;
Status prepare_reduction(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept;
Status prepare_layer_norm(Operation operation,
                          std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs,
                          NativePlan &plan) noexcept;

} // namespace hyle::compute::native::internal
