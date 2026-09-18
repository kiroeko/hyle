#include "region.h"
#include "native_graph.h"

#include <algorithm>
#include <limits>
#include <new>

#include "ggml-backend.h"
#include "ggml.h"

namespace hyle::compute::ggml_vulkan::detail {
namespace {
bool tensor_domain(const TensorDesc &description) noexcept {
  return description.valid() && description.is_plain() &&
         description.is_dense() &&
         description.data_type() == DataType::kFloat32 &&
         description.rank() > 0 && description.rank() <= GGML_MAX_DIMS &&
         description.element_count() && *description.element_count() > 0 &&
         *description.element_count() <= INT32_MAX;
}
bool same_shape(const TensorDesc &a, const TensorDesc &b) noexcept {
  return a.rank() == b.rank() &&
         std::ranges::equal(a.dimensions(), b.dimensions());
}
struct Imports final {
  NativeDevice device;
  std::vector<ggml_backend_buffer *> buffers;
  ~Imports() {
    auto active = device.activate();
    for (auto *buffer : buffers)
      ggml_backend_buffer_free(buffer);
  }
};
} // namespace

bool region_supports(const Operation &operation,
                     std::span<const TensorDesc> inputs,
                     const TensorDesc &output) noexcept {
  if (!tensor_domain(output) || !std::ranges::all_of(inputs, tensor_domain))
    return false;
  if (const auto *norm = operation.get_if<RmsNorm>())
    return norm->accumulation_type == DataType::kFloat32;
  if (const auto *unary = operation.get_if<Unary>())
    return unary->kind == UnaryKind::kSilu;
  if (const auto *binary = operation.get_if<Binary>())
    return inputs.size() == 2 &&
           (binary->kind == BinaryKind::kAdd ||
            binary->kind == BinaryKind::kMultiply) &&
           (same_shape(inputs[0], output) || same_shape(inputs[1], output));
  if (const auto *gate = operation.get_if<GatedActivation>())
    return gate->activation == GatedActivationKind::kSilu &&
           gate->inputs == GatedInputKind::kSeparate;
  return false;
}

struct Region::State final {
  NativeDevice device;
  std::shared_ptr<Imports> imports;
  std::vector<ggml_tensor *> ports;
  NativeGraph graph;
  NativeWork work; // Retires before graph/imports/device.
  bool bound = false;
  bool pending = false;
};

Result<Region> Region::create(NativeDevice device,
                              const RegionRecipe &recipe) noexcept {
  if (!device || recipe.nodes.empty() || recipe.outputs.empty() ||
      recipe.values.size() > INT32_MAX / 8 ||
      recipe.nodes.size() > INT32_MAX / 8)
    return std::unexpected(
        Status::InvalidArgument("ggml Vulkan region recipe"));
  try {
    std::vector<bool> available(recipe.values.size(), false);
    for (const auto input : recipe.inputs) {
      if (input >= available.size() || available[input] ||
          !tensor_domain(recipe.values[input]))
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region input"));
      available[input] = true;
    }
    for (const auto &node : recipe.nodes) {
      if (node.output >= available.size() || available[node.output])
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region SSA"));
      std::vector<TensorDesc> inputs;
      for (const auto input : node.inputs) {
        if (input >= available.size() || !available[input])
          return std::unexpected(
              Status::InvalidArgument("ggml Vulkan region topology"));
        inputs.push_back(recipe.values[input]);
      }
      HYLE_ASSIGN_OR_RETURN(auto inferred,
                            infer_single_output(node.operation, inputs));
      if (inferred != recipe.values[node.output] ||
          !region_supports(node.operation, inputs, inferred))
        return std::unexpected(
            Status::Unimplemented("ggml Vulkan region semantic domain"));
      available[node.output] = true;
    }
    for (const auto output : recipe.outputs)
      if (output >= available.size() || !available[output] ||
          std::ranges::find(recipe.inputs, output) != recipe.inputs.end())
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region output"));
    const auto capacity = recipe.inputs.size() + 3 * recipe.nodes.size() + 16;
    const auto graph_bytes = ggml_graph_overhead_custom(capacity, false);
    if (capacity > (SIZE_MAX - graph_bytes) / ggml_tensor_overhead())
      return std::unexpected(
          Status::ResourceExhausted("ggml Vulkan region metadata capacity"));
    HYLE_ASSIGN_OR_RETURN(
        auto arena,
        MetadataArena::create(capacity * ggml_tensor_overhead() + graph_bytes,
                              mem::system_allocator()));
    Region result;
    result.state_ = std::make_shared<State>();
    auto &state = *result.state_;
    state.device = std::move(device);
    state.imports = std::make_shared<Imports>();
    state.imports->device = state.device;
    const auto port_count = recipe.inputs.size() + recipe.outputs.size();
    state.ports.reserve(port_count);
    state.imports->buffers.reserve(port_count);
    auto active = state.device.activate();
    auto *context = arena.get();
    std::vector<ggml_tensor *> tensors(recipe.values.size());
    for (const auto input : recipe.inputs) {
      std::int64_t shape[GGML_MAX_DIMS]{1, 1, 1, 1};
      const auto dimensions = recipe.values[input].dimensions();
      std::ranges::reverse_copy(dimensions, shape);
      tensors[input] = ggml_new_tensor(
          context, GGML_TYPE_F32, static_cast<int>(dimensions.size()), shape);
      ggml_set_input(tensors[input]);
      state.ports.push_back(tensors[input]);
    }
    for (const auto &node : recipe.nodes) {
      auto *left = tensors[node.inputs[0]];
      ggml_tensor *output = nullptr;
      if (const auto *norm = node.operation.get_if<RmsNorm>()) {
        output = ggml_rms_norm(context, left, norm->epsilon);
        if (node.inputs.size() > 1)
          output = ggml_mul(context, output, tensors[node.inputs[1]]);
      } else if (node.operation.get_if<Unary>()) {
        output = ggml_silu(context, left);
      } else if (node.operation.get_if<GatedActivation>()) {
        output = ggml_swiglu_split(context, left, tensors[node.inputs[1]]);
      } else {
        auto *right = tensors[node.inputs[1]];
        if (!same_shape(recipe.values[node.inputs[0]],
                        recipe.values[node.output]))
          std::swap(left, right);
        output = node.operation.get_if<Binary>()->kind == BinaryKind::kAdd
                     ? ggml_add(context, left, right)
                     : ggml_mul(context, left, right);
      }
      tensors[node.output] = output;
    }
    auto *graph = ggml_new_graph_custom(context, capacity, false);
    for (const auto output : recipe.outputs) {
      ggml_set_output(tensors[output]);
      ggml_build_forward_expand(graph, tensors[output]);
      state.ports.push_back(tensors[output]);
    }
    for (auto *tensor : state.ports) {
      HYLE_ASSIGN_OR_RETURN(auto buffer,
                            state.device.binding_buffer(ggml_nbytes(tensor)));
      state.imports->buffers.push_back(buffer);
      if (ggml_backend_tensor_alloc(buffer, tensor,
                                    ggml_backend_buffer_get_base(buffer)) !=
          GGML_STATUS_SUCCESS)
        return std::unexpected(
            Status::Internal("ggml Vulkan region port storage"));
    }
    HYLE_ASSIGN_OR_RETURN(state.graph,
                          NativeGraph::allocate(state.device, std::move(arena),
                                                graph, state.imports));
    HYLE_ASSIGN_OR_RETURN(state.work, state.device.create_work());
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("ggml Vulkan typed region"));
  }
}

Status
Region::validate(std::span<const RegionBinding> bindings) const noexcept {
  if (!state_ || state_->pending || bindings.size() != state_->ports.size())
    return Status::FailedPrecondition("ggml Vulkan region binding state");
  auto active = state_->device.activate();
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto &port = bindings[i];
    if (port.handle == VK_NULL_HANDLE && !port.owner)
      continue;
    const auto base = reinterpret_cast<std::uintptr_t>(
        ggml_backend_buffer_get_base(state_->imports->buffers[i]));
    if (port.handle == VK_NULL_HANDLE || !port.owner ||
        port.offset % sizeof(float) ||
        port.capacity < ggml_nbytes(state_->ports[i]) ||
        port.offset > SIZE_MAX - port.capacity ||
        port.offset > UINTPTR_MAX - base)
      return Status::InvalidArgument("ggml Vulkan region native buffer range");
  }
  return Status::Ok();
}

Status Region::bind(std::span<const RegionBinding> bindings) noexcept {
  HYLE_RETURN_IF_ERROR(validate(bindings));
  auto active = state_->device.activate();
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto &port = bindings[i];
    auto *buffer = state_->imports->buffers[i];
    const auto size = port.handle ? port.offset + port.capacity
                                  : ggml_nbytes(state_->ports[i]);
    state_->device.bind_buffer(buffer, port.handle, size, port.owner);
    state_->ports[i]->data = reinterpret_cast<void *>(
        reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(buffer)) +
        port.offset);
  }
  state_->bound = std::ranges::all_of(
      bindings, [](const auto &port) { return port.handle != VK_NULL_HANDLE; });
  return Status::Ok();
}
WorkState Region::submit() noexcept {
  if (!state_ || !state_->bound)
    return {Status::FailedPrecondition("ggml Vulkan region ports are unbound")};
  const auto result =
      state_->work.submit(state_->graph.get(), state_->graph.lease());
  state_->pending = result.pending;
  return result;
}
WorkState Region::poll() noexcept {
  if (!state_)
    return {Status::FailedPrecondition("empty ggml Vulkan region")};
  const auto result = state_->work.poll();
  state_->pending = result.pending;
  return result;
}
WorkState Region::wait() noexcept {
  if (!state_)
    return {Status::FailedPrecondition("empty ggml Vulkan region")};
  const auto result = state_->work.wait();
  state_->pending = result.pending;
  return result;
}
VkFence Region::fence() const noexcept {
  return state_ ? state_->work.fence() : VK_NULL_HANDLE;
}
std::size_t Region::storage_bytes() const noexcept {
  return state_ ? state_->graph.storage_bytes() : 0;
}
} // namespace hyle::compute::ggml_vulkan::detail
