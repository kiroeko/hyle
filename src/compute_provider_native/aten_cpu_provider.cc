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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
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
};
struct alignas(std::max_align_t) Invocation final {
  std::span<mem::HostMapping> mappings{};
  std::span<at::Tensor> tensors{};
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

Status select(Operation operation, std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  const bool supported =
      operation.get_if<Select>() || operation.get_if<Gather>() ||
      operation.get_if<Fill>() || operation.get_if<Copy>() ||
      operation.get_if<Concat>() ||
      (operation.get_if<Scatter>() &&
       operation.get_if<Scatter>()->kind == ScatterKind::kAssign);
  if (!supported)
    return Status::Unimplemented("ATen CPU storage program is not admitted");
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
  auto program_bytes =
      allocation_size(sizeof(Program), inputs.size(), sizeof(TensorDesc));
  if (!program_bytes)
    return std::move(program_bytes).error();
  auto invocation_bytes =
      allocation_size(sizeof(Invocation), inputs.size(),
                      sizeof(mem::HostMapping) + sizeof(at::Tensor));
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
                      sizeof(mem::HostMapping) + sizeof(at::Tensor)));
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
  }
  inv->mappings = {maps, p.inputs.size()};
  inv->tensors = {tensors, p.inputs.size()};
  Status status;
  try {
    at::NoGradGuard guard;
    if (p.operation.get_if<Select>()) {
      const auto shape = byte_shape(p.inputs[0]);
      inv->condition =
          at::empty(at::IntArrayRef(shape.sizes.data(), shape.rank), at::kBool);
    }
    if (p.operation.get_if<Gather>() || p.operation.get_if<Scatter>()) {
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

Status execute(const Program &p, Invocation &inv,
               std::byte *destination) noexcept {
  try {
    at::NoGradGuard guard;
    const auto &input = inv.tensors;
    auto out = byte_tensor(p.output, destination);
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
  return {.supported =
              output == 0 && input == 0 && !p.inputs.empty() &&
              p.inputs[0] == p.output &&
              (p.operation.get_if<Copy>() || p.operation.get_if<Scatter>())};
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
  *out = {.route = static_cast<std::uint64_t>(request.operation.kind()),
          .runtime = kRuntime,
          .workspace_resource = host(),
          .cost = {.workspace = WorkspaceRequirement{},
                   .persistent_bytes = sizeof(Program) + request.inputs.size() *
                                                             sizeof(TensorDesc),
                   .copy_bytes = std::nullopt,
                   .repack_bytes = std::nullopt,
                   .synchronization_count = 0,
                   .dispatch_count = std::nullopt},
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
  if (!p->inputs.empty())
    p->axis = static_cast<std::size_t>(
        requested_axis < 0
            ? static_cast<std::int64_t>(requested_axis) + p->inputs[0].rank()
            : requested_axis);
  if (request.operation.get_if<Gather>() || request.operation.get_if<Scatter>())
    p->index_count = p->inputs[1].element_count().value();
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
                 .name = "ATen CPU storage/index",
                 .version = "libtorch-2.8.0-cpu+storage-1"},
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
