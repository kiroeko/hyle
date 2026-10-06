#include "hyle/compute/providers/native.h"

#include "../compute/provider_internal.h"

#if defined(HYLE_COMPUTE_TORCH_CPU)
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/Exception.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numbers>
#include <string_view>

#include "hyle/mem/buffer.h"

namespace hyle::compute::aten_cpu {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kProvider,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = true,
    .supports_concurrent_invocations = true,
    .capture_compatible = false};

struct Program final {
  Operation operation{};
  std::span<TensorDesc> inputs{};
  TensorDesc output{};
  std::array<std::byte, 8> scalar{};
  std::size_t allocation_bytes = 0;
  std::size_t axis = 0;
  std::size_t index_count = 0;
  std::array<std::int64_t, kMaxTensorRank> reduction_axes{};
  std::size_t reduction_axis_count = 0;
  bool empty_reduction = false;
};
struct alignas(std::max_align_t) Invocation final {
  std::span<mem::HostMapping> mappings{};
  std::span<at::Tensor> tensors{};
  std::span<at::Tensor> numeric_inputs{};
  at::Tensor condition{}, index_bytes{}, index_destination{}, index_values{};
  std::size_t allocation_bytes = 0;
};
static_assert(alignof(TensorDesc) <= alignof(Program));
static_assert(alignof(mem::HostMapping) <= alignof(Invocation));
static_assert(alignof(at::Tensor) <= alignof(mem::HostMapping));

Status exception_status(const c10::Error &error) noexcept {
  const std::string_view message = error.what_without_backtrace();
  if (message.find("DefaultCPUAllocator: not enough memory:") != message.npos ||
      message.find("DefaultCPUAllocator: can't allocate memory:") !=
          message.npos)
    return Status::ResourceExhausted(message);
  return Status::Internal(message);
}

Result<std::size_t> allocation_size(std::size_t header, std::size_t count,
                                    std::size_t item) noexcept {
  if (count > ((std::numeric_limits<std::size_t>::max)() - header) / item)
    return std::unexpected(
        Status::OutOfRange("ATen CPU metadata capacity overflows"));
  return header + count * item;
}

bool numeric_program(Operation operation) noexcept {
  if (const auto *scatter = operation.get_if<Scatter>())
    return scatter->kind == ScatterKind::kAdd;
  if (const auto *unary = operation.get_if<Unary>())
    return unary->kind == UnaryKind::kSigmoid ||
           unary->kind == UnaryKind::kSilu || unary->kind == UnaryKind::kGelu ||
           unary->kind == UnaryKind::kGeluTanh ||
           unary->kind == UnaryKind::kSoftplus;
  if (const auto *binary = operation.get_if<Binary>())
    return binary->kind == BinaryKind::kMinimum ||
           binary->kind == BinaryKind::kMaximum;
  return operation.get_if<Cast>() || operation.get_if<GatedActivation>() ||
         operation.get_if<Reduce>() || operation.get_if<MatMul>() ||
         operation.get_if<Softmax>() || operation.get_if<RoPE>() ||
         operation.get_if<ScaledDotProductAttention>() ||
         operation.get_if<RmsNorm>() || operation.get_if<LayerNorm>();
}
bool floating(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}
bool cast_storage(DataType type) noexcept {
  return floating(type) || type == DataType::kFloat8E4M3FN ||
         type == DataType::kFloat8E5M2;
}
at::ScalarType native_type(DataType type) noexcept {
  switch (type) {
  case DataType::kBool:
    return at::kByte;
  case DataType::kInt32:
    return at::kInt;
  case DataType::kInt64:
    return at::kLong;
  case DataType::kFloat16:
    return at::kHalf;
  case DataType::kBFloat16:
    return at::kBFloat16;
  case DataType::kFloat8E4M3FN:
    return at::kFloat8_e4m3fn;
  case DataType::kFloat8E5M2:
    return at::kFloat8_e5m2;
  default:
    return at::kFloat;
  }
}

Status select(Operation operation, std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  const bool supported =
      operation.get_if<Select>() || operation.get_if<Gather>() ||
      operation.get_if<Fill>() || operation.get_if<Copy>() ||
      operation.get_if<Concat>() ||
      (operation.get_if<Scatter>() &&
       operation.get_if<Scatter>()->kind == ScatterKind::kAssign) ||
      numeric_program(operation);
  if (!supported)
    return Status::Unimplemented("ATen CPU program is not admitted");
  std::array<TensorDesc, 1> inferred{};
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0] ||
      !outputs[0].is_dense())
    return Status::Unimplemented("ATen CPU requires the inferred dense output");
  const auto validate = [](const TensorDesc &desc) -> Status {
    const auto width = data_type_size(desc.data_type());
    if (!desc.is_plain() || width == 0 || width > 8)
      return Status::Unimplemented(
          "ATen CPU storage requires known plain scalar types");
    HYLE_ASSIGN_OR_RETURN(auto bytes, desc.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto elements, desc.element_count());
    if (bytes > static_cast<std::uint64_t>(INT64_MAX) ||
        elements > static_cast<std::uint64_t>(INT64_MAX) / width)
      return Status::Unimplemented(
          "ATen CPU byte carrier exceeds native int64 capacity");
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(validate(outputs[0]));
  for (const auto &input : inputs)
    HYLE_RETURN_IF_ERROR(validate(input));
  if (numeric_program(operation)) {
    const auto admitted = operation.get_if<Cast>() ? &cast_storage : &floating;
    if (!admitted(outputs[0].data_type()))
      return Status::Unimplemented(
          "ATen CPU numerical storage is not admitted");
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto type = inputs[i].data_type();
      const bool rope_index =
          operation.get_if<RoPE>() && i == 1 &&
          (type == DataType::kInt32 || type == DataType::kInt64);
      const bool scatter_index =
          operation.get_if<Scatter>() && i == 1 &&
          (type == DataType::kInt32 || type == DataType::kInt64);
      const bool bool_mask =
          type == DataType::kBool &&
          ((operation.get_if<Softmax>() && i == 1 &&
            operation.get_if<Softmax>()->mask == MaskKind::kBoolean) ||
           (operation.get_if<ScaledDotProductAttention>() && i == 3 &&
            operation.get_if<ScaledDotProductAttention>()->mask ==
                MaskKind::kBoolean));
      if (!admitted(type) && !rope_index && !scatter_index && !bool_mask)
        return Status::Unimplemented(
            "ATen CPU numerical input is not admitted");
    }
    if (operation.get_if<RoPE>() &&
        (inputs[0].data_type() != DataType::kFloat32 ||
         inputs[2].data_type() != DataType::kFloat32))
      return Status::Unimplemented("ATen CPU RoPE requires F32 data/frequency");
    auto accumulation = DataType::kFloat32;
    if (const auto *op = operation.get_if<MatMul>())
      accumulation = op->accumulation_type;
    if (const auto *op = operation.get_if<Softmax>())
      accumulation = op->accumulation_type;
    if (const auto *op = operation.get_if<ScaledDotProductAttention>())
      accumulation = op->accumulation_type;
    if (const auto *op = operation.get_if<RmsNorm>())
      accumulation = op->accumulation_type;
    if (const auto *op = operation.get_if<LayerNorm>())
      accumulation = op->accumulation_type;
    if (accumulation != DataType::kFloat32 &&
        !(accumulation == DataType::kUnknown &&
          inputs[0].data_type() == DataType::kFloat32))
      return Status::Unimplemented(
          "ATen CPU numerical program requires F32 accumulation");
    if (const auto *attention = operation.get_if<ScaledDotProductAttention>()) {
      const auto sequence =
          attention->layout == AttentionLayout::kBatchSequenceHeadDepth ? 1U
                                                                        : 2U;
      const auto heads = sequence == 1 ? 2U : 1U;
      std::uint64_t count = 1;
      for (const auto value :
           {inputs[0].dimensions()[0], inputs[0].dimensions()[heads],
            inputs[0].dimensions()[sequence],
            inputs[1].dimensions()[sequence]}) {
        if (value != 0 && count > static_cast<std::uint64_t>(INT64_MAX) /
                                      static_cast<std::uint64_t>(value))
          return Status::Unimplemented(
              "ATen CPU score tensor exceeds native int64 capacity");
        count *= static_cast<std::uint64_t>(value);
      }
    }
    if (const auto *reduce = operation.get_if<Reduce>())
      if (reduce->accumulation_type != DataType::kFloat32 &&
          !(reduce->accumulation_type == DataType::kUnknown &&
            inputs[0].data_type() == DataType::kFloat32))
        return Status::Unimplemented(
            "ATen CPU Reduce requires F32 accumulation");
  }
  auto program_bytes =
      allocation_size(sizeof(Program), inputs.size(), sizeof(TensorDesc));
  if (!program_bytes)
    return std::move(program_bytes).error();
  auto invocation_bytes =
      allocation_size(sizeof(Invocation), inputs.size(),
                      sizeof(mem::HostMapping) + 2 * sizeof(at::Tensor));
  if (!invocation_bytes)
    return std::move(invocation_bytes).error();
  return Status::Ok();
}

struct ByteShape final {
  std::array<std::int64_t, kMaxTensorRank + 1> sizes{}, strides{};
  std::size_t rank = 0;
};
ByteShape byte_shape(const TensorDesc &desc) noexcept {
  ByteShape shape;
  shape.rank = desc.rank() + 1;
  for (std::size_t i = 0; i < desc.rank(); ++i) {
    shape.sizes[i] = desc.dimensions()[i];
    shape.strides[i] = desc.byte_strides()[i];
  }
  shape.sizes[desc.rank()] =
      static_cast<std::int64_t>(data_type_size(desc.data_type()));
  shape.strides[desc.rank()] = 1;
  return shape;
}
at::Tensor byte_tensor(const TensorDesc &desc, void *data) {
  const auto shape = byte_shape(desc);
  return at::from_blob(data, at::IntArrayRef(shape.sizes.data(), shape.rank),
                       at::IntArrayRef(shape.strides.data(), shape.rank),
                       at::kByte);
}

void preserve_error(Status &status, Status other) noexcept {
  if (status.ok() && !other.ok())
    status = std::move(other);
}
void destroy_program(void *state, mem::AllocatorRef allocator) noexcept {
  auto *p = static_cast<Program *>(state);
  if (!p)
    return;
  const auto bytes = p->allocation_bytes;
  for (auto &desc : p->inputs)
    std::destroy_at(&desc);
  std::destroy_at(p);
  allocator.deallocate(p, bytes, alignof(Program));
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *inv = static_cast<Invocation *>(state);
  if (!inv)
    return;
  const auto bytes = inv->allocation_bytes;
  for (auto &tensor : inv->tensors)
    std::destroy_at(&tensor);
  for (auto &tensor : inv->numeric_inputs)
    std::destroy_at(&tensor);
  for (auto &mapping : inv->mappings)
    std::destroy_at(&mapping);
  std::destroy_at(inv);
  allocator.deallocate(inv, bytes, alignof(Invocation));
}
Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView, void **output) noexcept {
  if (!output)
    return Status::InvalidArgument("ATen CPU invocation output is null");
  *output = nullptr;
  const auto &p = *static_cast<const Program *>(state);
  HYLE_ASSIGN_OR_RETURN(
      auto bytes,
      allocation_size(sizeof(Invocation), p.inputs.size(),
                      sizeof(mem::HostMapping) + 2 * sizeof(at::Tensor)));
  auto *storage =
      static_cast<std::byte *>(allocator.allocate(bytes, alignof(Invocation)));
  if (!storage)
    return Status::ResourceExhausted("ATen CPU invocation allocation failed");
  auto *inv = ::new (storage) Invocation{.allocation_bytes = bytes};
  auto *maps =
      reinterpret_cast<mem::HostMapping *>(storage + sizeof(Invocation));
  auto *tensors = reinterpret_cast<at::Tensor *>(storage + sizeof(Invocation) +
                                                 p.inputs.size() *
                                                     sizeof(mem::HostMapping));
  for (std::size_t i = 0; i < p.inputs.size(); ++i) {
    std::construct_at(maps + i);
    std::construct_at(tensors + i);
    std::construct_at(tensors + p.inputs.size() + i);
  }
  inv->mappings = {maps, p.inputs.size()};
  inv->tensors = {tensors, p.inputs.size()};
  inv->numeric_inputs = {tensors + p.inputs.size(), p.inputs.size()};
  Status status;
  try {
    at::NoGradGuard guard;
    if (numeric_program(p.operation))
      for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        const auto dimensions = p.inputs[i].dimensions();
        inv->numeric_inputs[i] =
            at::empty(at::IntArrayRef(dimensions.data(), dimensions.size()),
                      native_type(p.inputs[i].data_type()));
      }
    if (p.operation.get_if<Select>()) {
      const auto shape = byte_shape(p.inputs[0]);
      inv->condition =
          at::empty(at::IntArrayRef(shape.sizes.data(), shape.rank), at::kBool);
    }
    if (p.operation.get_if<Gather>() ||
        (p.operation.get_if<Scatter>() &&
         p.operation.get_if<Scatter>()->kind == ScatterKind::kAssign)) {
      const auto width =
          static_cast<std::int64_t>(data_type_size(p.inputs[1].data_type()));
      // Own an aligned native buffer even when the public index bytes are
      // unaligned.
      inv->index_bytes = at::empty(
          {static_cast<std::int64_t>(p.index_count) * width}, at::kByte);
      const auto shape = byte_shape(p.inputs[1]);
      inv->index_destination = inv->index_bytes.view(
          at::IntArrayRef(shape.sizes.data(), shape.rank));
      inv->index_values =
          inv->index_bytes.view(width == 4 ? at::kInt : at::kLong);
    }
  } catch (const std::bad_alloc &) {
    status = Status::ResourceExhausted(
        "ATen CPU private invocation allocation failed");
  } catch (const c10::Error &error) {
    status = exception_status(error);
  } catch (const std::exception &error) {
    status = Status::Internal(error.what());
  }
  if (!status.ok()) {
    destroy_invocation(inv, allocator);
    return status;
  }
  *output = inv;
  return Status::Ok();
}

// Boundary validation, not payload gather/assign math. Complete before any
// output write, including a late invalid index and odd byte strides.
Status check_indices(const Program &p, const std::byte *data) noexcept {
  const auto &desc = p.inputs[1];
  const auto extent = p.inputs[0].dimensions()[p.axis];
  for (std::size_t linear = 0; linear < p.index_count; ++linear) {
    auto remaining = linear;
    std::size_t offset = 0;
    for (std::size_t reverse = 0; reverse < desc.rank(); ++reverse) {
      const auto axis = desc.rank() - 1 - reverse;
      const auto dimension = static_cast<std::size_t>(desc.dimensions()[axis]);
      offset += remaining % dimension *
                static_cast<std::size_t>(desc.byte_strides()[axis]);
      remaining /= dimension;
    }
    std::int64_t index = 0;
    if (desc.data_type() == DataType::kInt32) {
      std::int32_t value = 0;
      std::memcpy(&value, data + offset, sizeof(value));
      index = value;
    } else
      std::memcpy(&index, data + offset, sizeof(index));
    if (index < 0 || index >= extent)
      return Status::OutOfRange("ATen CPU index is outside the selected axis");
  }
  return Status::Ok();
}

// These are original ATen Tensor programs. The private native inputs are
// aligned and packed by original Byte copy, never a truncated element stride.
at::Tensor numeric_input(const Program &p, Invocation &inv, std::size_t index) {
  const auto shape = byte_shape(p.inputs[index]);
  auto bytes = inv.numeric_inputs[index].flatten().view(at::kByte).view(
      at::IntArrayRef(shape.sizes.data(), shape.rank));
  bytes.copy_(inv.tensors[index]);
  return inv.numeric_inputs[index];
}
// Original ATen conversion followed by public NaN/finite-overflow semantics.
// All payload transformations below are original Tensor kernels, not HYLE
// loops.
at::Tensor storage_cast(const at::Tensor &input, at::ScalarType target) {
  if (input.scalar_type() == target)
    return input;
  auto value = input.to(at::kFloat);
  if (target == at::kFloat) {
    // Restore the declared widening payload bits, including signalling NaNs.
    // Native arithmetic conversions can quiet them before a storage write.
    const bool half = input.scalar_type() == at::kHalf;
    const bool bf16 = input.scalar_type() == at::kBFloat16;
    const bool e4 = input.scalar_type() == at::kFloat8_e4m3fn;
    auto bits = input.flatten()
                    .view(half || bf16 ? at::kShort : at::kByte)
                    .to(at::kInt);
    auto sign = bits.bitwise_and(half || bf16 ? 0x8000 : 0x80)
                    .bitwise_left_shift(half || bf16 ? 16 : 24);
    at::Tensor nan, payload;
    if (half) {
      nan = bits.bitwise_and(0x7c00).eq(0x7c00).logical_and(
          bits.bitwise_and(0x3ff).ne(0));
      payload = bits.bitwise_and(0x3ff)
                    .bitwise_left_shift(13)
                    .bitwise_or(0x7f800000)
                    .bitwise_or(sign);
    } else if (bf16) {
      nan = bits.bitwise_and(0x7f80).eq(0x7f80).logical_and(
          bits.bitwise_and(0x7f).ne(0));
      payload = bits.bitwise_and(0xffff).bitwise_left_shift(16);
    } else if (e4) {
      nan = bits.bitwise_and(0x7f).eq(0x7f);
      payload = sign.bitwise_or(0x7fc00000);
    } else {
      nan = bits.bitwise_and(0x7c).eq(0x7c).logical_and(
          bits.bitwise_and(3).ne(0));
      payload = bits.bitwise_and(3)
                    .bitwise_left_shift(21)
                    .bitwise_or(0x7f800000)
                    .bitwise_or(sign);
    }
    auto output = at::where(nan, payload, value.flatten().view(at::kInt));
    return output.view(at::kFloat).view(input.sizes());
  }
  const bool e4 = target == at::kFloat8_e4m3fn;
  const bool e5 = target == at::kFloat8_e5m2;
  auto converted = value;
  if (e4 || e5) {
    const double maximum = e4 ? 448 : 57344;
    converted = value.clamp(-maximum, maximum);
    if (e5)
      converted = at::where(value.isinf(), value, converted);
  }
  auto output = converted.to(target);
  const auto storage = e4 || e5 ? at::kByte : at::kShort;
  auto bits = output.flatten().view(storage);
  const int positive = e4                    ? 0x7f
                       : e5                  ? 0x7e
                       : target == at::kHalf ? 0x7fff
                                             : 0x7fc0;
  const int negative = e4 ? 0xff : e5 ? 0xfe : target == at::kHalf ? -1 : -64;
  auto nan = at::where(value.flatten().signbit(), at::full_like(bits, negative),
                       at::full_like(bits, positive));
  bits = at::where(value.flatten().isnan(), nan, bits);
  return bits.view(target).view(input.sizes());
}

at::Tensor activation(const at::Tensor &input, GatedActivationKind kind) {
  auto wide = input.to(at::kDouble);
  at::Tensor result;
  switch (kind) {
  case GatedActivationKind::kSilu:
    result = at::silu(wide);
    break;
  case GatedActivationKind::kGelu:
    // The equivalent erfc form avoids cancellation in the negative tail.
    result = wide.mul(0.5).mul(at::erfc(wide.mul(-std::numbers::sqrt2 / 2)));
    break;
  case GatedActivationKind::kGeluTanh: {
    auto inner = wide.add(wide.pow(3).mul(0.044715))
                     .mul(2 * std::sqrt(2 / std::numbers::pi));
    result = wide.mul(at::sigmoid(inner));
    break;
  }
  }
  return at::where(wide.eq(-(std::numeric_limits<double>::infinity)()),
                   at::zeros_like(result), result)
      .to(at::kFloat);
}
at::Tensor number_minmax(const at::Tensor &left, const at::Tensor &right,
                         bool minimum) {
  auto result = minimum ? at::fmin(left, right) : at::fmax(left, right);
  auto negative = minimum ? at::signbit(left).logical_or(at::signbit(right))
                          : at::signbit(left).logical_and(at::signbit(right));
  auto zero =
      at::where(negative, at::full_like(result, -0.0), at::zeros_like(result));
  return at::where(left.eq(0).logical_and(right.eq(0)), zero, result);
}
at::Tensor reduction(const Program &p, const at::Tensor &input) {
  const auto &op = *p.operation.get_if<Reduce>();
  if (p.reduction_axis_count == 0)
    return input;
  if (p.empty_reduction) {
    const auto dimensions = p.output.dimensions();
    const auto infinity = (std::numeric_limits<double>::infinity)();
    const double identity = op.kind == ReductionKind::kSum ? 0.0
                            : op.kind == ReductionKind::kMean
                                ? (std::numeric_limits<double>::quiet_NaN)()
                            : op.kind == ReductionKind::kMinimum ? infinity
                                                                 : -infinity;
    return at::full(at::IntArrayRef(dimensions.data(), dimensions.size()),
                    identity, at::kFloat);
  }
  const auto axes =
      at::IntArrayRef(p.reduction_axes.data(), p.reduction_axis_count);
  at::Tensor result;
  switch (op.kind) {
  case ReductionKind::kSum:
    return at::sum(input, axes, op.keep_dimensions);
  case ReductionKind::kMean:
    return at::mean(input, axes, op.keep_dimensions);
  case ReductionKind::kMinimum:
    result = at::amin(input, axes, op.keep_dimensions);
    break;
  case ReductionKind::kMaximum:
    result = at::amax(input, axes, op.keep_dimensions);
    break;
  }
  // Original reductions propagate NaN, but their equal-zero choice is not the
  // public number min/max rule. Repair only zero results with original kernels.
  auto negative = op.kind == ReductionKind::kMinimum
                      ? at::any(at::signbit(input).logical_and(input.eq(0)),
                                axes, op.keep_dimensions)
                      : at::all(at::signbit(input).logical_or(input.ne(0)),
                                axes, op.keep_dimensions);
  auto zero =
      at::where(negative, at::full_like(result, -0.0), at::zeros_like(result));
  return at::where(result.eq(0), zero, result);
}
at::Tensor transformer_softmax(const Program &p, Invocation &inv,
                               const at::Tensor &first) {
  const auto &op = *p.operation.get_if<Softmax>();
  auto scores = first.mul(op.scale);
  if (op.mask == MaskKind::kBoolean)
    scores = scores.masked_fill(numeric_input(p, inv, 1).eq(0),
                                -(std::numeric_limits<float>::infinity)());
  else if (op.mask == MaskKind::kAdditive)
    scores = scores.add(numeric_input(p, inv, 1).to(at::kFloat));
  return op.kind == SoftmaxKind::kLogSoftmax
             ? at::log_softmax(scores, static_cast<std::int64_t>(p.axis),
                               at::kFloat)
             : at::softmax(scores, static_cast<std::int64_t>(p.axis),
                           at::kFloat);
}
at::Tensor transformer_rope(const Program &p, Invocation &inv,
                            const at::Tensor &first) {
  const auto &op = *p.operation.get_if<RoPE>();
  const auto positions = numeric_input(p, inv, 1);
  const auto frequency = numeric_input(p, inv, 2);
  std::array<std::int64_t, kMaxTensorRank> ps{}, fs{};
  ps.fill(1);
  fs.fill(1);
  if (positions.dim() == 1)
    ps[p.axis] = positions.size(0);
  else
    for (std::int64_t i = 0; i < positions.dim(); ++i)
      ps[static_cast<std::size_t>(i)] = positions.size(i);
  const auto rank = p.inputs[0].rank();
  const auto rotary = static_cast<std::int64_t>(op.rotary_dimensions);
  fs[rank - 1] = rotary / 2;
  if (frequency.dim() == 2)
    fs[p.axis] = frequency.size(0);
  auto angles =
      positions.to(at::kDouble)
          .view(at::IntArrayRef(ps.data(), rank))
          .mul(
              frequency.to(at::kDouble).view(at::IntArrayRef(fs.data(), rank)));
  auto cosine = angles.cos(), sine = angles.sin();
  auto data = first.to(at::kDouble);
  const bool split = op.layout == RopeLayout::kHalfSplit;
  auto left =
      split ? data.slice(-1, 0, rotary / 2) : data.slice(-1, 0, rotary, 2);
  auto right =
      split ? data.slice(-1, rotary / 2, rotary) : data.slice(-1, 1, rotary, 2);
  auto a = left.mul(cosine).sub(right.mul(sine));
  auto b = left.mul(sine).add(right.mul(cosine));
  auto rotated =
      split ? at::cat({a, b}, -1) : at::stack({a, b}, -1).flatten(-2);
  auto result = first.clone();
  result.slice(-1, 0, rotary).copy_(rotated);
  return result;
}
at::Tensor transformer_attention(const Program &p, Invocation &inv,
                                 const at::Tensor &first) {
  const auto &op = *p.operation.get_if<ScaledDotProductAttention>();
  auto q = first, k = numeric_input(p, inv, 1).to(at::kFloat),
       v = numeric_input(p, inv, 2).to(at::kFloat);
  const bool bshd = op.layout == AttentionLayout::kBatchSequenceHeadDepth;
  if (bshd) {
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);
  }
  const auto groups = q.size(1) / k.size(1);
  if (groups != 1) {
    k = k.repeat_interleave(groups, 1);
    v = v.repeat_interleave(groups, 1);
  }
  auto scores = at::matmul(q, k.transpose(-1, -2)).mul(op.scale);
  if (op.mask == MaskKind::kBoolean)
    scores = scores.masked_fill(numeric_input(p, inv, 3).eq(0),
                                -(std::numeric_limits<float>::infinity)());
  else if (op.mask == MaskKind::kAdditive)
    scores = scores.add(numeric_input(p, inv, 3).to(at::kFloat));
  if (op.causal) {
    auto row = at::arange(q.size(2), at::kLong).view({q.size(2), 1});
    auto column = at::arange(k.size(2), at::kLong).view({1, k.size(2)});
    auto allowed = column.le(row.add(k.size(2) - q.size(2)));
    scores = scores.masked_fill(allowed.logical_not(),
                                -(std::numeric_limits<float>::infinity)());
  }
  auto result = at::matmul(at::softmax(scores, -1, at::kFloat), v);
  if (k.size(2) == 0)
    result = at::full_like(result, (std::numeric_limits<float>::quiet_NaN)());
  return bshd ? result.transpose(1, 2) : result;
}
at::Tensor transformer_norm(const Program &p, Invocation &inv,
                            const at::Tensor &first) {
  const std::array<std::int64_t, 1> shape{first.size(-1)};
  std::optional<at::Tensor> gamma, beta;
  if (p.inputs.size() > 1)
    gamma = numeric_input(p, inv, 1).to(at::kFloat);
  if (p.inputs.size() > 2)
    beta = numeric_input(p, inv, 2).to(at::kFloat);
  if (const auto *op = p.operation.get_if<RmsNorm>())
    return at::rms_norm(first, shape, gamma, op->epsilon);
  return at::layer_norm(first, shape, gamma, beta,
                        p.operation.get_if<LayerNorm>()->epsilon);
}
void execute_numeric(const Program &p, Invocation &inv, at::Tensor &output) {
  auto first = numeric_input(p, inv, 0);
  at::Tensor result;
  if (p.operation.get_if<Cast>()) {
    result = storage_cast(first, native_type(p.output.data_type()));
  } else {
    first = first.to(at::kFloat);
    if (const auto *unary = p.operation.get_if<Unary>()) {
      if (unary->kind == UnaryKind::kSigmoid)
        result = at::sigmoid(first.to(at::kDouble)).to(at::kFloat);
      else if (unary->kind == UnaryKind::kSoftplus)
        result = at::softplus(first, 1, 20);
      else
        result = activation(first, unary->kind == UnaryKind::kSilu
                                       ? GatedActivationKind::kSilu
                                   : unary->kind == UnaryKind::kGelu
                                       ? GatedActivationKind::kGelu
                                       : GatedActivationKind::kGeluTanh);
    } else if (const auto *binary = p.operation.get_if<Binary>()) {
      result = number_minmax(first, numeric_input(p, inv, 1).to(at::kFloat),
                             binary->kind == BinaryKind::kMinimum);
    } else if (const auto *gate = p.operation.get_if<GatedActivation>()) {
      at::Tensor values;
      if (gate->inputs == GatedInputKind::kSeparate)
        values = numeric_input(p, inv, 1).to(at::kFloat);
      else {
        const auto axis = static_cast<std::int64_t>(p.output.rank() - 1);
        const auto hidden = p.output.dimensions().back();
        values = first.slice(axis, hidden);
        first = first.slice(axis, 0, hidden);
      }
      // Keep an F32 activation and multiplication, with one storage rounding.
      result = activation(first, gate->activation).mul(values);
    } else if (p.operation.get_if<Scatter>()) {
      auto indices = numeric_input(p, inv, 1).flatten();
      auto updates = numeric_input(p, inv, 2).to(at::kFloat);
      const auto axis = static_cast<std::int64_t>(p.axis);
      updates =
          p.inputs[1].rank() == 0
              ? updates.unsqueeze(axis)
              : updates.flatten(
                    axis,
                    axis + static_cast<std::int64_t>(p.inputs[1].rank()) - 1);
      // Public Add allows wider accumulation and reassociation. Retain all
      // updates, use the original numeric kernel, and round to storage once.
      result = first.clone();
      result.index_add_(axis, indices, updates);
    } else if (p.operation.get_if<MatMul>())
      result = at::matmul(first, numeric_input(p, inv, 1).to(at::kFloat));
    else if (p.operation.get_if<Softmax>())
      result = transformer_softmax(p, inv, first);
    else if (p.operation.get_if<RoPE>())
      result = transformer_rope(p, inv, first);
    else if (p.operation.get_if<ScaledDotProductAttention>())
      result = transformer_attention(p, inv, first);
    else if (p.operation.get_if<RmsNorm>() || p.operation.get_if<LayerNorm>())
      result = transformer_norm(p, inv, first);
    else
      result = reduction(p, first);
  }
  const auto shape = byte_shape(p.output);
  auto bytes = result.to(native_type(p.output.data_type()))
                   .contiguous()
                   .flatten()
                   .view(at::kByte)
                   .view(at::IntArrayRef(shape.sizes.data(), shape.rank));
  output.copy_(bytes);
}

Status execute(const Program &p, Invocation &inv,
               std::byte *destination) noexcept {
  try {
    at::NoGradGuard guard;
    const auto &input = inv.tensors;
    auto out = byte_tensor(p.output, destination);
    if (numeric_program(p.operation)) {
      if (p.operation.get_if<Scatter>() && p.index_count == 0) {
        out.copy_(input[0]);
        return Status::Ok();
      }
      execute_numeric(p, inv, out);
      return Status::Ok();
    }
    switch (p.operation.kind()) {
    case OperationKind::kSelect:
      at::ne_out(inv.condition, input[0], 0);
      at::where_out(out, inv.condition, input[1], input[2]);
      break;
    case OperationKind::kCopy:
      out.copy_(input[0]);
      break;
    case OperationKind::kFill: {
      auto scalar = at::from_blob(
          const_cast<std::byte *>(p.scalar.data()),
          {static_cast<std::int64_t>(data_type_size(p.output.data_type()))},
          at::kByte);
      out.copy_(scalar);
      break;
    }
    case OperationKind::kConcat:
      at::cat_out(out, at::TensorList(input.data(), input.size()),
                  static_cast<std::int64_t>(p.axis));
      break;
    case OperationKind::kGather: {
      inv.index_destination.copy_(input[1]);
      // Multi-rank/scalar public indices have the same dense logical order
      // as the single index_select axis; preserve the actual public output.
      auto shape = byte_shape(p.inputs[0]);
      shape.sizes[p.axis] = static_cast<std::int64_t>(p.index_count);
      auto flat_out = at::from_blob(
          destination, at::IntArrayRef(shape.sizes.data(), shape.rank),
          at::kByte);
      at::index_select_out(flat_out, input[0],
                           static_cast<std::int64_t>(p.axis), inv.index_values);
      break;
    }
    case OperationKind::kScatter: {
      if (p.index_count == 0) {
        out.copy_(input[0]);
        break;
      }
      inv.index_destination.copy_(input[1]);
      const auto [sorted, order] =
          at::sort(inv.index_values, std::optional<bool>(true), 0, false);
      const auto count = static_cast<std::int64_t>(p.index_count);
      auto last = at::cat({sorted.slice(0, 1).ne(sorted.slice(0, 0, count - 1)),
                           at::ones({1}, at::kBool)});
      auto winners = order.masked_select(last);
      auto unique = sorted.masked_select(last).to(at::kLong);
      const auto axis = static_cast<std::int64_t>(p.axis);
      auto updates =
          p.inputs[1].rank() == 0
              ? input[2].unsqueeze(axis)
              : input[2].flatten(
                    axis,
                    axis + static_cast<std::int64_t>(p.inputs[1].rank()) - 1);
      auto selected = updates.index_select(axis, winners);
      out.copy_(input[0]);
      out.index_copy_(axis, unique, selected);
      break;
    }
    default:
      return Status::Internal("ATen CPU prepared program kind is invalid");
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ATen CPU execution allocation failed");
  } catch (const c10::Error &error) {
    return exception_status(error);
  } catch (const std::exception &error) {
    return Status::Internal(error.what());
  }
}

detail::EnqueueResult submit(void *state, void *invocation, Queue &,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const Program *>(state);
  auto &inv = *static_cast<Invocation *>(invocation);
  const auto completed = [](Status status) {
    return detail::EnqueueResult::accepted_command(std::move(status));
  };
  if (inputs.size() != p.inputs.size() || outputs.size() != 1)
    return completed(Status::Internal("ATen CPU binding count differs"));
  if (p.output.element_count().value() == 0)
    return completed(Status::Ok());
  Status status;
  try {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto buffer = inputs[i].storage();
      if (!buffer.resource().properties().has(
              mem::MemoryProperty::kHostDeviceCoherent))
        status = buffer.invalidate();
      if (!status.ok())
        break;
      auto mapping = buffer.map_host();
      if (!mapping) {
        status = std::move(mapping).error();
        break;
      }
      inv.mappings[i] = std::move(*mapping);
      inv.tensors[i] = byte_tensor(
          p.inputs[i], const_cast<std::byte *>(inv.mappings[i].bytes().data()));
    }
    if (status.ok() &&
        (p.operation.get_if<Gather>() || p.operation.get_if<Scatter>()))
      status = check_indices(p, inv.mappings[1].bytes().data());
  } catch (const std::bad_alloc &) {
    status = Status::ResourceExhausted(
        "ATen CPU borrowed tensor metadata allocation failed");
  } catch (const c10::Error &error) {
    status = exception_status(error);
  } catch (const std::exception &error) {
    status = Status::Internal(error.what());
  }
  mem::HostMapping output_mapping;
  if (status.ok()) {
    auto mapping = outputs[0].storage().map_host(mem::HostAccess::kWrite);
    if (!mapping)
      status = std::move(mapping).error();
    else {
      output_mapping = std::move(*mapping);
      auto writable = output_mapping.writable_bytes();
      if (!writable)
        status = std::move(writable).error();
      else
        status = execute(p, inv, writable->data());
      if (!outputs[0].storage().resource().properties().has(
              mem::MemoryProperty::kHostDeviceCoherent))
        preserve_error(status, output_mapping.flush());
    }
  }
  // Drop every borrowed native tensor before its host mapping closes. Owned
  // condition/index buffers remain invocation-private for the next replay.
  for (auto &tensor : inv.tensors)
    tensor = at::Tensor{};
  preserve_error(status, output_mapping.close());
  for (auto &mapping : inv.mappings)
    preserve_error(status, mapping.close());
  return completed(std::move(status));
}

AliasSupport alias(void *state, std::size_t output,
                   std::size_t input) noexcept {
  const auto &p = *static_cast<const Program *>(state);
  const bool allowed =
      (input == 0 &&
       (p.operation.get_if<Scatter>() || p.operation.get_if<Unary>() ||
        p.operation.get_if<Softmax>() || p.operation.get_if<RoPE>() ||
        p.operation.get_if<RmsNorm>() || p.operation.get_if<LayerNorm>())) ||
      p.operation.get_if<Binary>();
  return {.supported = allowed && output == 0 && input < p.inputs.size() &&
                       p.inputs[input] == p.output};
}
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_program,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
detail::ResourceBindingRequirement host() noexcept {
  return {.required_properties = mem::MemoryProperty::kHostAccessible,
          .minimum_alignment = 1};
}
Status create_context(const Provider &, Device device, void *,
                      mem::AllocatorRef, void **out) noexcept {
  if (!out)
    return Status::InvalidArgument("ATen CPU context output is null");
  *out = nullptr;
  return device.device_class() == DeviceClass::kCpu
             ? Status::Ok()
             : Status::Unimplemented("ATen CPU provider requires a CPU device");
}
void destroy_context(void *, mem::AllocatorRef) noexcept {}
Status drain(void *) noexcept { return Status::Ok(); }
Status propose(void *, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *out) noexcept {
  if (!out || inputs.size() != request.inputs.size() ||
      outputs.size() != request.outputs.size())
    return Status::InvalidArgument(
        "ATen CPU proposal binding count is invalid");
  *out = {};
  if (!request.shape.exact())
    return Status::Unimplemented("ATen CPU requires exact shapes");
  HYLE_RETURN_IF_ERROR(
      select(request.operation, request.inputs, request.outputs));
  for (auto &item : inputs)
    item = host();
  outputs[0] = host();
  *out = {
      .route = static_cast<std::uint64_t>(request.operation.kind()),
      .runtime = kRuntime,
      .workspace_resource = host(),
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes =
                   sizeof(Program) + request.inputs.size() * sizeof(TensorDesc),
               .copy_bytes = std::nullopt,
               .repack_bytes = std::nullopt,
               .synchronization_count = 0,
               .dispatch_count = std::nullopt,
               .compilation_storage = {.resident_bytes = 0, .peak_bytes = 0}},
      .implementation = ProviderImplementationId{
          0x4154454e00000100ULL +
          static_cast<std::uint64_t>(request.operation.kind())}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *out) noexcept {
  if (!out)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("ATen CPU prepared output is null"));
  *out = {};
  if (!request.shape.exact() ||
      request.route != static_cast<std::uint64_t>(request.operation.kind()))
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("ATen CPU selected route is unavailable"));
  auto status = select(request.operation, request.inputs, request.outputs);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(status)
               : ProviderPrepareOutcome::error(status);
  const auto bytes =
      sizeof(Program) + request.inputs.size() * sizeof(TensorDesc);
  auto *storage =
      static_cast<std::byte *>(allocator.allocate(bytes, alignof(Program)));
  if (!storage)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("ATen CPU program allocation failed"));
  auto *p = ::new (storage) Program{.operation = request.operation,
                                    .output = request.outputs[0],
                                    .allocation_bytes = bytes};
  auto *descriptions =
      reinterpret_cast<TensorDesc *>(storage + sizeof(Program));
  for (std::size_t i = 0; i < request.inputs.size(); ++i)
    std::construct_at(descriptions + i, request.inputs[i]);
  p->inputs = {descriptions, request.inputs.size()};
  std::int32_t requested_axis = 0;
  if (const auto *op = request.operation.get_if<Gather>())
    requested_axis = op->axis;
  if (const auto *op = request.operation.get_if<Scatter>())
    requested_axis = op->axis;
  if (const auto *op = request.operation.get_if<Concat>())
    requested_axis = op->axis;
  if (const auto *op = request.operation.get_if<Softmax>())
    requested_axis = op->axis;
  if (const auto *op = request.operation.get_if<RoPE>())
    requested_axis = op->sequence_axis;
  if (!p->inputs.empty())
    p->axis = static_cast<std::size_t>(
        requested_axis < 0
            ? static_cast<std::int64_t>(requested_axis) + p->inputs[0].rank()
            : requested_axis);
  if (request.operation.get_if<Gather>() || request.operation.get_if<Scatter>())
    p->index_count = p->inputs[1].element_count().value();
  if (const auto *reduce = request.operation.get_if<Reduce>()) {
    std::array<bool, kMaxTensorRank> selected{};
    if (reduce->axes.is_all())
      for (std::size_t i = 0; i < p->inputs[0].rank(); ++i)
        selected[i] = true;
    else
      for (auto axis : reduce->axes.values())
        selected[static_cast<std::size_t>(
            axis < 0 ? axis + static_cast<std::int64_t>(p->inputs[0].rank())
                     : axis)] = true;
    for (std::size_t i = 0; i < p->inputs[0].rank(); ++i)
      if (selected[i]) {
        p->reduction_axes[p->reduction_axis_count++] =
            static_cast<std::int64_t>(i);
        p->empty_reduction =
            p->empty_reduction || p->inputs[0].dimensions()[i] == 0;
      }
  }
  if (const auto *fill = request.operation.get_if<Fill>()) {
    const auto bits = fill->value.bits();
    switch (data_type_size(p->output.data_type())) {
    case 1: {
      const auto value = static_cast<std::uint8_t>(bits);
      std::memcpy(p->scalar.data(), &value, 1);
      break;
    }
    case 2: {
      const auto value = static_cast<std::uint16_t>(bits);
      std::memcpy(p->scalar.data(), &value, 2);
      break;
    }
    case 4: {
      const auto value = static_cast<std::uint32_t>(bits);
      std::memcpy(p->scalar.data(), &value, 4);
      break;
    }
    default:
      std::memcpy(p->scalar.data(), &bits, 8);
      break;
    }
  }
  *out = {.vtable = &kPreparedVTable,
          .state = p,
          .runtime = kRuntime,
          .persistent_bytes = bytes,
          .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = kProviderId,
                 .name = "ATen CPU programs",
                 .version =
                     "libtorch-2.8.0-cpu+programs-5+compilation-bounds-1"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain};
} // namespace
Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}
} // namespace hyle::compute::aten_cpu
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#else
namespace hyle::compute::aten_cpu {
Provider provider() noexcept { return {}; }
} // namespace hyle::compute::aten_cpu
#endif
