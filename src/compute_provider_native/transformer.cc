#include "native_plan.h"
#include "numeric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include "../compute/rope_kernels.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/fat_kernel.h"
#include "hyle/simd/math.h"

namespace hyle::compute::native::internal {
namespace {

Status require_plain_f32(const TensorDesc &description,
                         const char *message) noexcept {
  if (description.data_type() != DataType::kFloat32 ||
      !description.is_plain()) {
    return Status::Unimplemented(message);
  }
  return Status::Ok();
}

Status invalidate_for_read(mem::ConstBufferView storage) noexcept {
  if (storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    return Status::Ok();
  }
  return storage.invalidate();
}

bool same_storage(mem::ConstBufferView left, mem::BufferView right) noexcept {
  return left.resource() == right.resource() &&
         left.native_handle() == right.native_handle() &&
         left.offset() == right.offset() && left.size() == right.size();
}

std::size_t product(std::span<const std::int64_t> dimensions, std::size_t first,
                    std::size_t last) noexcept {
  std::size_t result = 1;
  for (std::size_t axis = first; axis < last; ++axis)
    result *= static_cast<std::size_t>(dimensions[axis]);
  return result;
}

void coordinates(std::span<const std::int64_t> dimensions, std::size_t linear,
                 std::array<std::size_t, kMaxTensorRank> &result) noexcept {
  result.fill(0);
  for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
    const std::size_t axis = dimensions.size() - 1 - reverse;
    const std::size_t extent = static_cast<std::size_t>(dimensions[axis]);
    result[axis] = linear % extent;
    linear /= extent;
  }
}

float load_f32(const std::byte *data, std::size_t offset) noexcept {
  float value = 0.0F;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

void store_f32(std::byte *data, std::size_t offset, float value) noexcept {
  std::memcpy(data + offset, &value, sizeof(value));
}

std::int64_t load_position(const std::byte *data, const TensorDesc &description,
                           std::size_t linear) noexcept {
  const std::size_t offset = logical_byte_offset(description, linear);
  std::int32_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

RopeKernelRoute select_rope_kernel(RopeLayout layout) noexcept {
#if HYLE_ARCH_X64
  if (layout == RopeLayout::kInterleaved) {
    return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_interleaved_scalar,
                                       &detail::run_rope_interleaved_sse4,
                                       &detail::run_rope_interleaved_avx2,
                                       &detail::run_rope_interleaved_avx512);
  }
  return HYLE_SIMD_SELECT_FAT_KERNEL(
      &detail::run_rope_half_split_scalar, &detail::run_rope_half_split_sse4,
      &detail::run_rope_half_split_avx2, &detail::run_rope_half_split_avx512);
#elif HYLE_ARCH_ARM64
  if (layout == RopeLayout::kInterleaved) {
    return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_interleaved_scalar,
                                       &detail::run_rope_interleaved_neon,
                                       &detail::run_rope_interleaved_sve);
  }
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_half_split_scalar,
                                     &detail::run_rope_half_split_neon,
                                     &detail::run_rope_half_split_sve);
#elif HYLE_ARCH_RISCV64
  if (layout == RopeLayout::kInterleaved) {
    return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_interleaved_scalar,
                                       &detail::run_rope_interleaved_rvv);
  }
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_half_split_scalar,
                                     &detail::run_rope_half_split_rvv);
#else
  return layout == RopeLayout::kInterleaved
             ? HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_interleaved_scalar)
             : HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_rope_half_split_scalar);
#endif
}

detail::EnqueueResult
submit_rope(const NativePlan &plan, std::span<const ConstTensorView> inputs,
            std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 3 || outputs.size() != 1)
    return completed(Status::Internal("native RoPE plan is invalid"));

  const TensorDesc &data_desc = inputs[0].description();
  const TensorDesc &positions_desc = inputs[1].description();
  const TensorDesc &frequencies_desc = inputs[2].description();
  const TensorDesc &output_desc = outputs[0].description();
  auto elements = output_desc.element_count();
  if (!elements)
    return completed(std::move(elements).error());
  if (*elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView data_storage = inputs[0].storage();
  const mem::ConstBufferView positions_storage = inputs[1].storage();
  const mem::ConstBufferView frequencies_storage = inputs[2].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place = same_storage(data_storage, output_storage);
  Status status = invalidate_for_read(data_storage);
  if (!status.ok())
    return completed(std::move(status));
  status = invalidate_for_read(positions_storage);
  if (!status.ok())
    return completed(std::move(status));
  status = invalidate_for_read(frequencies_storage);
  if (!status.ok())
    return completed(std::move(status));

  auto positions_result = positions_storage.map_host();
  if (!positions_result)
    return completed(std::move(positions_result).error());
  mem::HostMapping positions_mapping = std::move(*positions_result);
  auto frequencies_result = frequencies_storage.map_host();
  if (!frequencies_result) {
    positions_mapping.close().IgnoreError();
    return completed(std::move(frequencies_result).error());
  }
  mem::HostMapping frequencies_mapping = std::move(*frequencies_result);

  mem::HostMapping data_mapping;
  const std::byte *data = nullptr;
  if (!in_place) {
    auto data_result = data_storage.map_host();
    if (!data_result) {
      frequencies_mapping.close().IgnoreError();
      positions_mapping.close().IgnoreError();
      return completed(std::move(data_result).error());
    }
    data_mapping = std::move(*data_result);
    data = data_mapping.bytes().data();
  }

  auto output_result = output_storage.map_host(
      in_place ? mem::HostAccess::kReadWrite : mem::HostAccess::kWrite);
  if (!output_result) {
    data_mapping.close().IgnoreError();
    frequencies_mapping.close().IgnoreError();
    positions_mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    data_mapping.close().IgnoreError();
    frequencies_mapping.close().IgnoreError();
    positions_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  std::byte *output = writable->data();
  if (in_place)
    data = output;
  const std::byte *positions = positions_mapping.bytes().data();
  const std::byte *frequencies = frequencies_mapping.bytes().data();
  if (!aligned_for_storage(data, data_desc.required_alignment()) ||
      !aligned_for_storage(positions, positions_desc.required_alignment()) ||
      !aligned_for_storage(frequencies,
                           frequencies_desc.required_alignment()) ||
      !aligned_for_storage(output, output_desc.required_alignment())) {
    output_mapping.close().IgnoreError();
    data_mapping.close().IgnoreError();
    frequencies_mapping.close().IgnoreError();
    positions_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU RoPE mapping does not satisfy tensor alignment"));
  }

  const auto dimensions = data_desc.dimensions();
  const std::size_t depth = static_cast<std::size_t>(dimensions.back());
  const std::size_t rows = *elements / depth;
  const std::size_t pairs = plan.rotary_dimensions / 2;
  std::array<std::size_t, kMaxTensorRank> coordinate{};
  for (std::size_t row = 0; row < rows; ++row) {
    coordinates(dimensions.first(dimensions.size() - 1), row, coordinate);
    const std::size_t sequence = coordinate[plan.sequence_axis];
    std::size_t position_linear = sequence;
    if (positions_desc.rank() > 1) {
      position_linear = 0;
      for (std::size_t axis = 0; axis < positions_desc.rank(); ++axis) {
        position_linear =
            position_linear *
                static_cast<std::size_t>(positions_desc.dimensions()[axis]) +
            coordinate[axis];
      }
    }
    const std::int64_t position =
        load_position(positions, positions_desc, position_linear);
    const std::size_t row_linear = row * depth;
    if (plan.rope_kernel_route != nullptr) {
      const std::size_t frequency_linear =
          frequencies_desc.rank() == 1 ? 0 : sequence * pairs;
      const auto *row_input =
          reinterpret_cast<const float *>(data) + row_linear;
      const auto *row_frequencies =
          reinterpret_cast<const float *>(frequencies) + frequency_linear;
      auto *row_output = reinterpret_cast<float *>(output) + row_linear;
      plan.rope_kernel_route(row_input, row_frequencies, row_output, pairs,
                             static_cast<float>(position));
      if (row_input != row_output && plan.rotary_dimensions < depth) {
        std::copy_n(row_input + plan.rotary_dimensions,
                    depth - plan.rotary_dimensions,
                    row_output + plan.rotary_dimensions);
      }
      continue;
    }
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const std::size_t frequency_linear =
          frequencies_desc.rank() == 1 ? pair : sequence * pairs + pair;
      const float frequency = load_f32(
          frequencies, logical_byte_offset(frequencies_desc, frequency_linear));
      const float angle = static_cast<float>(position) * frequency;
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const std::size_t first =
          plan.rope_layout == RopeLayout::kInterleaved ? pair * 2 : pair;
      const std::size_t second = plan.rope_layout == RopeLayout::kInterleaved
                                     ? first + 1
                                     : pair + pairs;
      const float left =
          load_f32(data, logical_byte_offset(data_desc, row_linear + first));
      const float right =
          load_f32(data, logical_byte_offset(data_desc, row_linear + second));
      store_f32(output, (row_linear + first) * sizeof(float),
                left * cosine - right * sine);
      store_f32(output, (row_linear + second) * sizeof(float),
                left * sine + right * cosine);
    }
    for (std::size_t column = plan.rotary_dimensions; column < depth;
         ++column) {
      store_f32(
          output, (row_linear + column) * sizeof(float),
          load_f32(data, logical_byte_offset(data_desc, row_linear + column)));
    }
  }

  Status terminal = Status::Ok();
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, data_mapping.close());
  preserve_first_error(terminal, frequencies_mapping.close());
  preserve_first_error(terminal, positions_mapping.close());
  return completed(std::move(terminal));
}

template <class Value = float>
float masked_softmax_value(const NativePlan &plan, const std::byte *input,
                           const TensorDesc &input_desc, const std::byte *mask,
                           const TensorDesc *mask_desc,
                           std::size_t logical) noexcept {
  float value =
      numeric::load<Value>(input + logical_byte_offset(input_desc, logical));
  value *= plan.scalar;
  if (plan.mask_kind == MaskKind::kNone)
    return value;
  const std::size_t mask_offset = logical_broadcast_byte_offset(
      *mask_desc, input_desc.dimensions(), logical);
  if (plan.mask_kind == MaskKind::kBoolean)
    return mask[mask_offset] == std::byte{0}
               ? -std::numeric_limits<float>::infinity()
               : value;
  return value + numeric::load<Value>(mask + mask_offset);
}

void invalid_softmax_row(std::byte *output, std::size_t outer,
                         std::size_t inner_index, std::size_t axis_extent,
                         std::size_t inner) noexcept {
  const float invalid = std::numeric_limits<float>::quiet_NaN();
  for (std::size_t lane = 0; lane < axis_extent; ++lane) {
    const std::size_t logical =
        (outer * axis_extent + lane) * inner + inner_index;
    store_f32(output, logical * sizeof(float), invalid);
  }
}

template <class Value>
detail::EnqueueResult submit_softmax_impl(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs, mem::BufferView workspace) noexcept {
  if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1)
    return completed(Status::Internal("native Softmax plan is invalid"));
  const TensorDesc &input_desc = inputs[0].description();
  const TensorDesc &output_desc = outputs[0].description();
  const TensorDesc *mask_desc =
      inputs.size() == 2 ? &inputs[1].description() : nullptr;
  auto elements = output_desc.element_count();
  if (!elements)
    return completed(std::move(elements).error());
  if (*elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView input_storage = inputs[0].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place = same_storage(input_storage, output_storage);
  Status status = invalidate_for_read(input_storage);
  if (!status.ok())
    return completed(std::move(status));
  mem::ConstBufferView mask_storage;
  if (inputs.size() == 2) {
    mask_storage = inputs[1].storage();
    status = invalidate_for_read(mask_storage);
    if (!status.ok())
      return completed(std::move(status));
  }

  mem::HostMapping input_mapping;
  const std::byte *input = nullptr;
  if (!in_place) {
    auto input_result = input_storage.map_host();
    if (!input_result)
      return completed(std::move(input_result).error());
    input_mapping = std::move(*input_result);
    input = input_mapping.bytes().data();
  }
  mem::HostMapping mask_mapping;
  const std::byte *mask = nullptr;
  if (inputs.size() == 2) {
    auto mask_result = mask_storage.map_host();
    if (!mask_result) {
      input_mapping.close().IgnoreError();
      return completed(std::move(mask_result).error());
    }
    mask_mapping = std::move(*mask_result);
    mask = mask_mapping.bytes().data();
  }
  auto output_result = output_storage.map_host(
      in_place ? mem::HostAccess::kReadWrite : mem::HostAccess::kWrite);
  if (!output_result) {
    mask_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(std::move(output_result).error());
  }
  mem::HostMapping output_mapping = std::move(*output_result);
  auto writable = output_mapping.writable_bytes();
  if (!writable) {
    output_mapping.close().IgnoreError();
    mask_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(std::move(writable).error());
  }
  std::byte *output = writable->data();
  if (in_place)
    input = output;
  if (!aligned_for_storage(input, input_desc.required_alignment()) ||
      !aligned_for_storage(output, output_desc.required_alignment()) ||
      (mask_desc != nullptr &&
       !aligned_for_storage(mask, mask_desc->required_alignment()))) {
    output_mapping.close().IgnoreError();
    mask_mapping.close().IgnoreError();
    input_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU Softmax mapping does not satisfy tensor alignment"));
  }

  const auto dimensions = input_desc.dimensions();
  const std::size_t axis_extent =
      static_cast<std::size_t>(dimensions[plan.axis]);
  const std::size_t outer = product(dimensions, 0, plan.axis);
  const std::size_t inner =
      product(dimensions, plan.axis + 1, dimensions.size());
  Status terminal = Status::Ok();
  if constexpr (!std::is_same_v<Value, float>) {
    auto scratch_result = workspace.map_host(mem::HostAccess::kWrite);
    if (!scratch_result) {
      output_mapping.close().IgnoreError();
      mask_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed(std::move(scratch_result).error());
    }
    mem::HostMapping scratch_mapping = std::move(*scratch_result);
    auto scratch_bytes = scratch_mapping.writable_bytes();
    if (!scratch_bytes ||
        !aligned_for_storage(scratch_bytes->data(), alignof(float))) {
      scratch_mapping.close().IgnoreError();
      output_mapping.close().IgnoreError();
      mask_mapping.close().IgnoreError();
      input_mapping.close().IgnoreError();
      return completed(Status::FailedPrecondition(
          "CPU Softmax workspace mapping is invalid"));
    }
    auto *row_scratch = reinterpret_cast<float *>(scratch_bytes->data());
    for (std::size_t row = 0; row < outer; ++row) {
      for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
        if (input_desc.is_dense() && inner == 1 && mask_desc == nullptr &&
            plan.scalar == 1.0F) {
          numeric::load_tile<Value>(input, row * axis_extent, row_scratch,
                                    axis_extent);
        } else {
          for (std::size_t lane = 0; lane < axis_extent; ++lane) {
            const std::size_t logical =
                (row * axis_extent + lane) * inner + inner_index;
            row_scratch[lane] = masked_softmax_value<Value>(
                plan, input, input_desc, mask, mask_desc, logical);
          }
        }
        if (plan.softmax_kind == SoftmaxKind::kSoftmax)
          simd::softmax_f32(row_scratch, axis_extent);
        else
          simd::log_softmax_f32(row_scratch, axis_extent);
        if (inner == 1) {
          numeric::store_tile<Value>(row_scratch, output, row * axis_extent,
                                     axis_extent);
        } else {
          for (std::size_t lane = 0; lane < axis_extent; ++lane) {
            const std::size_t logical =
                (row * axis_extent + lane) * inner + inner_index;
            numeric::store<Value>(output + logical * sizeof(Value),
                                  row_scratch[lane]);
          }
        }
      }
    }
    preserve_first_error(terminal, scratch_mapping.close());
  } else {
    const bool contiguous = input_desc.is_dense() && output_desc.is_dense() &&
                            plan.axis + 1 == dimensions.size();
    if (contiguous) {
      for (std::size_t row = 0; row < outer; ++row) {
        const std::size_t base = row * axis_extent;
        auto *destination = reinterpret_cast<float *>(output) + base;
        const auto *source = reinterpret_cast<const float *>(input) + base;
        if (mask_desc == nullptr && plan.scalar == 1.0F) {
          if (plan.softmax_kind == SoftmaxKind::kSoftmax)
            simd::softmax_f32(source, destination, axis_extent);
          else
            simd::log_softmax_f32(source, destination, axis_extent);
          continue;
        }
        for (std::size_t lane = 0; lane < axis_extent; ++lane) {
          destination[lane] = masked_softmax_value(
              plan, input, input_desc, mask, mask_desc, base + lane);
        }
        if (plan.softmax_kind == SoftmaxKind::kSoftmax)
          simd::softmax_f32(destination, axis_extent);
        else
          simd::log_softmax_f32(destination, axis_extent);
      }
    } else {
      for (std::size_t row = 0; row < outer; ++row) {
        for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
          float maximum = -std::numeric_limits<float>::infinity();
          bool valid = false;
          for (std::size_t lane = 0; lane < axis_extent; ++lane) {
            const std::size_t logical =
                (row * axis_extent + lane) * inner + inner_index;
            const float value = masked_softmax_value(plan, input, input_desc,
                                                     mask, mask_desc, logical);
            if (std::isnan(value) ||
                value == std::numeric_limits<float>::infinity()) {
              valid = false;
              maximum = std::numeric_limits<float>::quiet_NaN();
              break;
            }
            if (std::isfinite(value)) {
              maximum = (std::max)(maximum, value);
              valid = true;
            }
          }
          if (!valid) {
            invalid_softmax_row(output, row, inner_index, axis_extent, inner);
            continue;
          }
          double sum = 0.0;
          for (std::size_t lane = 0; lane < axis_extent; ++lane) {
            const std::size_t logical =
                (row * axis_extent + lane) * inner + inner_index;
            const float value = masked_softmax_value(plan, input, input_desc,
                                                     mask, mask_desc, logical);
            const float exponential = std::exp(value - maximum);
            sum += static_cast<double>(exponential);
            store_f32(output, logical * sizeof(float),
                      plan.softmax_kind == SoftmaxKind::kSoftmax ? exponential
                                                                 : value);
          }
          if (plan.softmax_kind == SoftmaxKind::kSoftmax) {
            const float inverse = static_cast<float>(1.0 / sum);
            for (std::size_t lane = 0; lane < axis_extent; ++lane) {
              const std::size_t logical =
                  (row * axis_extent + lane) * inner + inner_index;
              store_f32(output, logical * sizeof(float),
                        load_f32(output, logical * sizeof(float)) * inverse);
            }
          } else {
            const float shift = maximum + static_cast<float>(std::log(sum));
            for (std::size_t lane = 0; lane < axis_extent; ++lane) {
              const std::size_t logical =
                  (row * axis_extent + lane) * inner + inner_index;
              store_f32(output, logical * sizeof(float),
                        load_f32(output, logical * sizeof(float)) - shift);
            }
          }
        }
      }
    }
  }
  if (!output_storage.resource().properties().has(
          mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output_mapping.flush());
  }
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, mask_mapping.close());
  preserve_first_error(terminal, input_mapping.close());
  return completed(std::move(terminal));
}

detail::EnqueueResult
submit_softmax(const NativePlan &plan, std::span<const ConstTensorView> inputs,
               std::span<const TensorView> outputs) noexcept {
  return submit_softmax_impl<float>(plan, inputs, outputs, {});
}

Result<std::size_t> normalized_axis(std::int32_t requested,
                                    std::size_t rank) noexcept {
  std::int64_t axis = requested;
  if (axis < 0)
    axis += static_cast<std::int64_t>(rank);
  if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) {
    return std::unexpected(
        Status::OutOfRange("native transformer axis is out of range"));
  }
  return static_cast<std::size_t>(axis);
}

} // namespace

Status prepare_transformer(Operation operation,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           NativePlan &plan) noexcept {
  switch (operation.kind()) {
  case OperationKind::kRoPE: {
    if (inputs.size() != 3 || outputs.size() != 1)
      return Status::Internal("native RoPE arity is invalid");
    Status status = require_plain_f32(
        inputs[0], "native RoPE currently requires plain float32 data");
    if (!status.ok())
      return status;
    status = require_plain_f32(
        inputs[2], "native RoPE currently requires plain float32 frequencies");
    if (!status.ok())
      return status;
    if (!inputs[1].is_plain() || inputs[1].data_type() != DataType::kInt32) {
      return Status::Unimplemented(
          "native RoPE retains only plain i32 positions; i64 uses mature math");
    }
    status = require_plain_f32(
        outputs[0], "native RoPE currently requires plain float32 output");
    if (!status.ok() || !outputs[0].is_dense())
      return status.ok()
                 ? Status::Unimplemented("native RoPE requires dense output")
                 : status;
    const RoPE *rope = operation.get_if<RoPE>();
    auto sequence_axis = normalized_axis(rope->sequence_axis, inputs[0].rank());
    if (!sequence_axis)
      return std::move(sequence_axis).error();
    plan.rotary_dimensions = rope->rotary_dimensions;
    plan.sequence_axis = *sequence_axis;
    plan.rope_layout = rope->layout;
    if (inputs[0].is_dense() && inputs[2].is_dense())
      plan.rope_kernel_route = select_rope_kernel(rope->layout);
    if (inputs[0] == outputs[0])
      plan.alias_input_mask = 1;
    plan.submit_route = &submit_rope;
    return Status::Ok();
  }
  case OperationKind::kSoftmax: {
    if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1)
      return Status::Internal("native Softmax arity is invalid");
    const DataType type = inputs[0].data_type();
    if ((type != DataType::kFloat32 && type != DataType::kFloat16 &&
         type != DataType::kBFloat16) ||
        !inputs[0].is_plain() || outputs[0].data_type() != type ||
        !outputs[0].is_plain() || !outputs[0].is_dense())
      return Status::Unimplemented("native Softmax requires plain "
                                   "float32/float16/bfloat16 and dense output");
    const Softmax *softmax = operation.get_if<Softmax>();
    if (softmax->accumulation_type != DataType::kUnknown &&
        softmax->accumulation_type != DataType::kFloat32) {
      return Status::Unimplemented(
          "native Softmax currently supports float32 accumulation");
    }
    if (type != DataType::kFloat32 &&
        softmax->accumulation_type != DataType::kFloat32)
      return Status::Unimplemented(
          "native half Softmax requires float32 accumulation");
    if (softmax->mask != MaskKind::kNone) {
      if (!inputs[1].is_plain() || (softmax->mask == MaskKind::kAdditive &&
                                    inputs[1].data_type() != type)) {
        return Status::Unimplemented(
            "native Softmax requires plain Bool or matching floating masks");
      }
    }
    auto axis = normalized_axis(softmax->axis, inputs[0].rank());
    if (!axis)
      return std::move(axis).error();
    plan.axis = *axis;
    plan.scalar = softmax->scale;
    plan.softmax_kind = softmax->kind;
    plan.mask_kind = softmax->mask;
    if (inputs[0] == outputs[0])
      plan.alias_input_mask = 1;
    if (type == DataType::kFloat32) {
      plan.submit_route = &submit_softmax;
    } else {
      const auto extent =
          math::try_integer_cast<std::size_t>(inputs[0].dimensions()[*axis]);
      if (!extent)
        return Status::OutOfRange("native Softmax row exceeds size_t");
      HYLE_ASSIGN_OR_RETURN(const auto output_count,
                            outputs[0].element_count());
      const auto scratch =
          math::try_multiply(output_count == 0 ? 0 : *extent, sizeof(float));
      if (!scratch)
        return Status::OutOfRange("native Softmax row workspace overflows");
      plan.workspace = {.size = *scratch, .alignment = alignof(float)};
      plan.workspace_submit_route =
          type == DataType::kFloat16 ? &submit_softmax_impl<simd::float16_t>
                                     : &submit_softmax_impl<simd::bfloat16_t>;
    }
    return Status::Ok();
  }
  default:
    return Status::Unimplemented("operation is not a native transformer op");
  }
}

} // namespace hyle::compute::native::internal
