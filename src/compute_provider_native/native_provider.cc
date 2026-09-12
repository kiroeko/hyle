#include "hyle/compute/providers/native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <new>
#include <span>
#include <utility>

#include "../compute/matmul_kernels.h"
#include "../compute/native_route.h"
#include "../compute/provider_internal.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/fat_kernel.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"
#include "native_plan.h"

namespace hyle::compute::native {
namespace {

using Parameters = detail::NativeParameters;
using Route = detail::NativeRoute;
using internal::aligned_for_storage;
using internal::completed;
using internal::NativePlan;
using internal::preserve_first_error;

struct NativeContext final {
  ParallelExecutorRef executor{};
};

void run_rms_norm(const float *input, const float *weight, float *output,
                  const Parameters &parameters, float epsilon) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t hidden = parameters[1];
  if (hidden == 0)
    return;
  for (std::size_t row = 0; row < rows; ++row) {
    const float *row_input = input + row * hidden;
    float *row_output = output + row * hidden;
    if (weight == nullptr)
      simd::rms_norm_f32(row_input, row_output, hidden, epsilon);
    else
      simd::rms_norm_f32(row_input, weight, row_output, hidden, epsilon);
  }
}

void run_silu_and_mul(const float *input, const float *, float *output,
                      const Parameters &parameters, float) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t hidden = parameters[1];
  if (hidden == 0)
    return;
  for (std::size_t row = 0; row < rows; ++row) {
    const float *gate = input + row * hidden * 2;
    const float *value = gate + hidden;
    simd::swiglu_f32_batch(gate, value, output + row * hidden, hidden);
  }
}

void run_matmul_transposed_weight(const float *left, const float *right,
                                  float *output, const Parameters &parameters,
                                  float) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t inner = parameters[1];
  const std::size_t columns = parameters[2];
  if (rows == 0 || columns == 0)
    return;
  if (inner == 0) {
    std::fill_n(output, rows * columns, 0.0F);
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const float *left_row = left + row * inner;
    float *output_row = output + row * columns;
    for (std::size_t column = 0; column < columns; ++column)
      output_row[column] = simd::dot(left_row, right + column * inner, inner);
  }
}

Route select_row_major_matmul() noexcept {
#if HYLE_ARCH_X64
  return HYLE_SIMD_SELECT_FAT_KERNEL(
      &detail::run_matmul_row_major_scalar, &detail::run_matmul_row_major_sse4,
      &detail::run_matmul_row_major_avx2, &detail::run_matmul_row_major_avx512);
#elif HYLE_ARCH_ARM64
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_matmul_row_major_scalar,
                                     &detail::run_matmul_row_major_neon,
                                     &detail::run_matmul_row_major_sve);
#elif HYLE_ARCH_RISCV64
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_matmul_row_major_scalar,
                                     &detail::run_matmul_row_major_rvv);
#else
  return HYLE_SIMD_SELECT_FAT_KERNEL(&detail::run_matmul_row_major_scalar);
#endif
}

Status validate_logical_f32(const TensorDesc &description) noexcept {
  if (!description.valid())
    return Status::InvalidArgument("operation tensor description is invalid");
  if (description.data_type() != DataType::kFloat32) {
    return Status::Unimplemented(
        "HYLE native route currently requires float32 tensors");
  }
  return Status::Ok();
}

Status validate_dense_f32(const TensorDesc &description) noexcept {
  Status status = validate_logical_f32(description);
  if (!status.ok())
    return status;
  if (!description.is_plain() || !description.is_dense()) {
    return Status::Unimplemented(
        "HYLE native route requires dense plain tensor storage");
  }
  return Status::Ok();
}

bool is_transposed_dense_matrix(const TensorDesc &description) noexcept {
  if (!description.valid() || !description.is_plain() ||
      description.rank() != 2)
    return false;
  const auto dimensions = description.dimensions();
  const auto strides = description.byte_strides();
  const auto row_bytes = math::try_multiply(
      dimensions[0], static_cast<std::int64_t>(sizeof(float)));
  return row_bytes.has_value() &&
         strides[0] == static_cast<std::int64_t>(sizeof(float)) &&
         strides[1] == *row_bytes;
}

Result<std::size_t> outer_count(const TensorDesc &description) noexcept {
  std::size_t result = 1;
  const auto dimensions = description.dimensions();
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis) {
    auto value = math::try_integer_cast<std::size_t>(dimensions[axis]);
    if (!value)
      return std::unexpected(
          Status::OutOfRange("tensor dimension exceeds size_t"));
    auto next = math::try_multiply(result, *value);
    if (!next)
      return std::unexpected(Status::OutOfRange("tensor row count overflows"));
    result = *next;
  }
  return result;
}

Result<std::size_t> dimension(const TensorDesc &description,
                              std::size_t axis) noexcept {
  auto result =
      math::try_integer_cast<std::size_t>(description.dimensions()[axis]);
  if (!result)
    return std::unexpected(
        Status::OutOfRange("tensor dimension exceeds size_t"));
  return *result;
}

Status create_context(const Provider &provider_value, Device device, void *,
                      mem::AllocatorRef allocator,
                      void **context_output) noexcept {
  if (context_output == nullptr)
    return Status::InvalidArgument("native context output is null");
  *context_output = nullptr;
  if (device.device_class() != DeviceClass::kCpu)
    return Status::Unimplemented("HYLE native provider requires a CPU device");
  void *storage =
      allocator.allocate(sizeof(NativeContext), alignof(NativeContext));
  if (storage == nullptr)
    return Status::ResourceExhausted(
        "native provider context allocation failed");
  const Options options =
      detail::ProviderAccess::payload<Options>(provider_value);
  *context_output = ::new (storage) NativeContext{.executor = options.executor};
  return Status::Ok();
}

void destroy_context(void *context_value,
                     mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<NativeContext *>(context_value);
  if (context == nullptr)
    return;
  context->~NativeContext();
  allocator.deallocate(context, sizeof(NativeContext), alignof(NativeContext));
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<NativePlan *>(state);
  if (plan == nullptr)
    return;
  plan->~NativePlan();
  allocator.deallocate(plan, sizeof(NativePlan), alignof(NativePlan));
}

Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("native invocation output is null");
  *output = nullptr;
  return Status::Ok();
}

void destroy_invocation(void *, mem::AllocatorRef) noexcept {}

Result<Completion> submit_float(const NativePlan &plan,
                                std::span<const ConstTensorView> inputs,
                                std::span<const TensorView> outputs) noexcept {
  if (plan.route == nullptr || inputs.empty() || inputs.size() > 2 ||
      outputs.size() != 1) {
    return completed(Status::Internal("native float plan is invalid"));
  }

  const mem::ConstBufferView first_storage = inputs[0].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place =
      first_storage.resource() == output_storage.resource() &&
      first_storage.native_handle() == output_storage.native_handle() &&
      first_storage.offset() == output_storage.offset() &&
      first_storage.size() == output_storage.size();
  mem::ConstBufferView second_storage;
  bool shared_inputs = false;
  if (inputs.size() == 2) {
    second_storage = inputs[1].storage();
    shared_inputs =
        first_storage.resource() == second_storage.resource() &&
        first_storage.native_handle() == second_storage.native_handle() &&
        first_storage.offset() == second_storage.offset() &&
        first_storage.size() == second_storage.size();
  }

  const mem::MemoryProperties first_properties =
      first_storage.resource().properties();
  const mem::MemoryProperties output_properties =
      output_storage.resource().properties();
  mem::MemoryProperties second_properties{};
  if (inputs.size() == 2)
    second_properties = second_storage.resource().properties();

  if (!first_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = first_storage.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }
  if (inputs.size() == 2 && !shared_inputs &&
      !second_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = second_storage.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }

  mem::HostMapping first_mapping;
  mem::HostMapping second_mapping;
  mem::HostMapping output_mapping;
  const float *first_data = nullptr;
  const float *second_data = nullptr;
  float *output_data = nullptr;

  if (in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kReadWrite);
    if (!mapping)
      return completed(std::move(mapping).error());
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable)
      return completed(std::move(writable).error());
    output_data = reinterpret_cast<float *>(writable->data());
    first_data = output_data;
  } else {
    auto mapping = first_storage.map_host();
    if (!mapping)
      return completed(std::move(mapping).error());
    first_mapping = std::move(*mapping);
    first_data = reinterpret_cast<const float *>(first_mapping.bytes().data());
  }

  if (inputs.size() == 2) {
    if (shared_inputs && !in_place) {
      second_data = first_data;
    } else {
      auto mapping = second_storage.map_host();
      if (!mapping) {
        first_mapping.close().IgnoreError();
        output_mapping.close().IgnoreError();
        return completed(std::move(mapping).error());
      }
      second_mapping = std::move(*mapping);
      second_data =
          reinterpret_cast<const float *>(second_mapping.bytes().data());
    }
  }

  if (!in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kWrite);
    if (!mapping) {
      second_mapping.close().IgnoreError();
      first_mapping.close().IgnoreError();
      return completed(std::move(mapping).error());
    }
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) {
      output_mapping.close().IgnoreError();
      second_mapping.close().IgnoreError();
      first_mapping.close().IgnoreError();
      return completed(std::move(writable).error());
    }
    output_data = reinterpret_cast<float *>(writable->data());
  }

  const bool first_aligned = aligned_for_storage(
      first_data, inputs[0].description().required_alignment());
  const bool second_aligned =
      inputs.size() != 2 ||
      aligned_for_storage(second_data,
                          inputs[1].description().required_alignment());
  const bool output_aligned = aligned_for_storage(
      output_data, outputs[0].description().required_alignment());
  if (!first_aligned || !second_aligned || !output_aligned) {
    output_mapping.close().IgnoreError();
    second_mapping.close().IgnoreError();
    first_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU operation tensor mapping does not satisfy its format alignment"));
  }

  plan.route(first_data, second_data, output_data, plan.parameters,
             plan.scalar);
  Status terminal = Status::Ok();
  if (!output_properties.has(mem::MemoryProperty::kHostDeviceCoherent))
    preserve_first_error(terminal, output_mapping.flush());
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, second_mapping.close());
  preserve_first_error(terminal, first_mapping.close());
  return completed(std::move(terminal));
}

Result<Completion> submit(void *prepared_value, void *, Queue &,
                          std::span<const ConstTensorView> inputs,
                          std::span<const TensorView> outputs) noexcept {
  auto *plan = static_cast<NativePlan *>(prepared_value);
  if (plan == nullptr || plan->submit_route == nullptr) {
    return completed(Status::Internal("native execution state is invalid"));
  }
  return plan->submit_route(*plan, inputs, outputs);
}

const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .submit = &submit,
};

Status prepare(void *, Operation operation, std::span<const TensorDesc> inputs,
               std::span<const TensorDesc> outputs, mem::AllocatorRef allocator,
               detail::ProviderPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return Status::InvalidArgument("native prepared output is null");
  *prepared_output = {};
  if (outputs.size() != 1)
    return Status::Unimplemented("native route requires one output");

  NativePlan plan{};
  switch (operation.kind()) {
  case OperationKind::kView:
  case OperationKind::kCopy:
  case OperationKind::kFill: {
    Status status =
        internal::prepare_transform(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kCast: {
    Status status = internal::prepare_cast(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kUnary:
  case OperationKind::kBinary:
  case OperationKind::kSelect: {
    Status status =
        internal::prepare_elementwise(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kReduce: {
    Status status =
        internal::prepare_reduction(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kLayerNorm: {
    Status status =
        internal::prepare_layer_norm(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kRmsNorm: {
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    if (inputs.size() == 2) {
      status = validate_dense_f32(inputs[1]);
      if (!status.ok())
        return status;
    }
    status = validate_dense_f32(outputs[0]);
    if (!status.ok())
      return status;
    auto rows = outer_count(inputs[0]);
    auto hidden = dimension(inputs[0], inputs[0].rank() - 1);
    if (!rows)
      return std::move(rows).error();
    if (!hidden)
      return std::move(hidden).error();
    plan.parameters = {*rows, *hidden, 0};
    plan.scalar = operation.get_if<RmsNorm>()->epsilon;
    plan.route = &run_rms_norm;
    plan.submit_route = &submit_float;
    break;
  }
  case OperationKind::kMatMul: {
    if (inputs.size() != 2 || inputs[0].rank() != 2 || inputs[1].rank() != 2)
      return Status::Unimplemented("native MatMul currently requires rank 2");
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    status = validate_logical_f32(inputs[1]);
    if (!status.ok())
      return status;
    if (!inputs[1].is_plain() ||
        (!inputs[1].is_dense() && !is_transposed_dense_matrix(inputs[1]))) {
      return Status::Unimplemented(
          "native MatMul right operand layout is unsupported");
    }
    status = validate_dense_f32(outputs[0]);
    if (!status.ok())
      return status;
    auto rows = dimension(inputs[0], 0);
    auto inner = dimension(inputs[0], 1);
    auto columns = dimension(inputs[1], 1);
    if (!rows)
      return std::move(rows).error();
    if (!inner)
      return std::move(inner).error();
    if (!columns)
      return std::move(columns).error();
    plan.parameters = {*rows, *inner, *columns};
    plan.route = inputs[1].is_dense() ? select_row_major_matmul()
                                      : &run_matmul_transposed_weight;
    plan.submit_route = &submit_float;
    break;
  }
  case OperationKind::kGatedActivation: {
    const GatedActivation *gated = operation.get_if<GatedActivation>();
    if (gated == nullptr || gated->activation != GatedActivationKind::kSilu ||
        gated->inputs != GatedInputKind::kPackedHalves || inputs.size() != 1) {
      return Status::Unimplemented(
          "native gated route currently supports packed SiLU only");
    }
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    status = validate_dense_f32(outputs[0]);
    if (!status.ok())
      return status;
    auto rows = outer_count(inputs[0]);
    auto packed = dimension(inputs[0], inputs[0].rank() - 1);
    if (!rows)
      return std::move(rows).error();
    if (!packed)
      return std::move(packed).error();
    plan.parameters = {*rows, *packed / 2, 0};
    plan.route = &run_silu_and_mul;
    plan.submit_route = &submit_float;
    break;
  }
  default:
    return Status::Unimplemented(
        "operation has no current HYLE native executable route");
  }

  void *storage = allocator.allocate(sizeof(NativePlan), alignof(NativePlan));
  if (storage == nullptr)
    return Status::ResourceExhausted("native plan allocation failed");
  auto *prepared = ::new (storage) NativePlan(plan);
  prepared_output->vtable = &kPreparedVTable;
  prepared_output->state = prepared;
  prepared_output->workspace = {};
  return Status::Ok();
}

const detail::ProviderVTable kProviderVTable{
    .identity = ProviderIdentity{.id = kProviderId,
                                 .name = "hyle-native",
                                 .version = "1"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .prepare = &prepare,
};

} // namespace

Provider provider(Options options) noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, options);
}

} // namespace hyle::compute::native
