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
         (description.data_type() == DataType::kFloat32 ||
          description.data_type() == DataType::kInt32) &&
         description.rank() <= GGML_MAX_DIMS && description.element_count() &&
         *description.element_count() > 0 &&
         *description.element_count() <= INT32_MAX;
}
bool same_shape(const TensorDesc &a, const TensorDesc &b) noexcept {
  return a.rank() == b.rank() &&
         std::ranges::equal(a.dimensions(), b.dimensions());
}
ggml_unary_op unary_operation(UnaryKind kind) noexcept {
  switch (kind) {
  case UnaryKind::kAbs:
    return GGML_UNARY_OP_ABS;
  case UnaryKind::kNeg:
    return GGML_UNARY_OP_NEG;
  case UnaryKind::kExp:
    return GGML_UNARY_OP_EXP;
  case UnaryKind::kTanh:
    return GGML_UNARY_OP_TANH;
  case UnaryKind::kSigmoid:
    return GGML_UNARY_OP_SIGMOID;
  case UnaryKind::kSilu:
    return GGML_UNARY_OP_SILU;
  default:
    return GGML_UNARY_OP_COUNT;
  }
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
                     std::span<const TensorDesc> outputs) noexcept {
  if (!std::ranges::all_of(outputs, tensor_domain) ||
      !std::ranges::all_of(inputs, tensor_domain))
    return false;
  if (const auto *topk = operation.get_if<TopK>())
    return inputs.size() == 1 && outputs.size() == 2 && inputs[0].rank() > 0 &&
           inputs[0].data_type() == DataType::kFloat32 &&
           outputs[0].data_type() == DataType::kFloat32 &&
           outputs[1].data_type() == DataType::kInt32 &&
           (topk->axis == -1 ||
            topk->axis == static_cast<std::int32_t>(inputs[0].rank()) - 1);
  if (const auto *gather = operation.get_if<Gather>())
    return inputs.size() == 2 && outputs.size() == 1 &&
           gather->bounds == IndexBoundsPolicy::kUnchecked &&
           (gather->axis == 0 || gather->axis == -2) &&
           inputs[0].data_type() == DataType::kInt32 && inputs[0].rank() == 2 &&
           inputs[1].data_type() == DataType::kInt32 && inputs[1].rank() == 1 &&
           outputs[0].data_type() == DataType::kInt32;
  if (outputs.size() != 1 || outputs[0].data_type() != DataType::kFloat32 ||
      !std::ranges::all_of(inputs, [](const auto &input) {
        return input.data_type() == DataType::kFloat32;
      }))
    return false;
  const auto &output = outputs[0];
  if (const auto *norm = operation.get_if<RmsNorm>())
    return norm->accumulation_type == DataType::kFloat32;
  if (const auto *norm = operation.get_if<LayerNorm>())
    return norm->accumulation_type == DataType::kFloat32;
  if (const auto *reduce = operation.get_if<Reduce>()) {
    const auto axes = reduce->axes.values();
    return inputs.size() == 1 && inputs[0].rank() > 0 &&
           reduce->keep_dimensions &&
           (reduce->kind == ReductionKind::kSum ||
            reduce->kind == ReductionKind::kMean) &&
           (reduce->accumulation_type == DataType::kFloat32 ||
            reduce->accumulation_type == DataType::kUnknown) &&
           ((reduce->axes.is_all() && inputs[0].rank() == 1) ||
            (!reduce->axes.is_all() && axes.size() == 1 &&
             (axes[0] == -1 ||
              axes[0] == static_cast<int>(inputs[0].rank()) - 1)));
  }
  if (const auto *softmax = operation.get_if<Softmax>())
    return inputs[0].rank() > 0 && softmax->kind == SoftmaxKind::kSoftmax &&
           (softmax->accumulation_type == DataType::kFloat32 ||
            softmax->accumulation_type == DataType::kUnknown) &&
           (softmax->axis == -1 ||
            softmax->axis == static_cast<int>(inputs[0].rank()) - 1) &&
           ((softmax->mask == MaskKind::kNone && inputs.size() == 1) ||
            (softmax->mask == MaskKind::kAdditive && inputs.size() == 2 &&
             softmax->scale == 1.0F && same_shape(inputs[0], inputs[1])));
  if (const auto *scan = operation.get_if<Scan>())
    return inputs.size() == 1 && inputs[0].rank() > 0 && !scan->exclusive &&
           (scan->axis == -1 ||
            scan->axis == static_cast<std::int32_t>(inputs[0].rank()) - 1);
  if (const auto *unary = operation.get_if<Unary>())
    return unary_operation(unary->kind) != GGML_UNARY_OP_COUNT;
  if (const auto *binary = operation.get_if<Binary>())
    return inputs.size() == 2 &&
           (binary->kind == BinaryKind::kAdd ||
            binary->kind == BinaryKind::kSubtract ||
            binary->kind == BinaryKind::kMultiply ||
            binary->kind == BinaryKind::kDivide) &&
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
  std::vector<ggml_tensor *> views;
  NativeGraph graph;
  NativeWork work; // Retires before graph/imports/device.
  bool bound = false;
  bool pending = false;
};

Result<Region> Region::create(NativeDevice device,
                              const RegionRecipe &recipe) noexcept {
  if (!device || recipe.nodes.empty() || recipe.outputs.empty() ||
      recipe.values.size() > INT32_MAX / 8 ||
      recipe.nodes.size() > INT32_MAX / 16)
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
      if (node.outputs.size() != operation_traits(node.operation).outputs)
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region SSA"));
      std::vector<TensorDesc> inputs;
      for (const auto input : node.inputs) {
        if (input >= available.size() || !available[input])
          return std::unexpected(
              Status::InvalidArgument("ggml Vulkan region topology"));
        inputs.push_back(recipe.values[input]);
      }
      std::vector<TensorDesc> inferred(node.outputs.size());
      HYLE_RETURN_IF_ERROR(
          infer_output_descriptions(node.operation, inputs, inferred));
      if (!region_supports(node.operation, inputs, inferred))
        return std::unexpected(
            Status::Unimplemented("ggml Vulkan region semantic domain"));
      for (std::size_t port = 0; port < node.outputs.size(); ++port) {
        const auto output = node.outputs[port];
        if (output >= available.size() || available[output] ||
            inferred[port] != recipe.values[output])
          return std::unexpected(
              Status::InvalidArgument("ggml Vulkan region output SSA/schema"));
        available[output] = true;
      }
    }
    std::vector<bool> published(recipe.values.size(), false);
    for (const auto output : recipe.outputs) {
      if (output >= available.size() || !available[output] ||
          published[output] ||
          std::ranges::find(recipe.inputs, output) != recipe.inputs.end())
        return std::unexpected(
            Status::InvalidArgument("ggml Vulkan region output"));
      published[output] = true;
    }
    // TopK/Scan: at most 10/16 nodes including reshape/view/cont adapters.
    // LayerNorm: input reshape, GroupNorm, restored shape, optional Mul/Add.
    // Row reduction: up to three exact row pieces, concat and shape restore.
    // Other operations need at most three (broadcast plus binary, or RMSNorm).
    auto capacity = recipe.inputs.size() + 16;
    for (const auto &node : recipe.nodes)
      capacity += node.operation.get_if<Scan>()        ? 16
                  : node.operation.get_if<Reduce>()    ? 12
                  : node.operation.get_if<TopK>()      ? 10
                  : node.operation.get_if<LayerNorm>() ? 5
                                                       : 3;
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
          context,
          recipe.values[input].data_type() == DataType::kInt32 ? GGML_TYPE_I32
                                                               : GGML_TYPE_F32,
          (std::max)(1, static_cast<int>(dimensions.size())), shape);
      ggml_set_input(tensors[input]);
      state.ports.push_back(tensors[input]);
    }
    for (const auto &node : recipe.nodes) {
      auto *left = tensors[node.inputs[0]];
      ggml_tensor *output = nullptr;
      if (const auto *topk = node.operation.get_if<TopK>()) {
        const auto width = left->ne[0];
        const auto rows = ggml_nelements(left) / width;
        const auto count = static_cast<std::int64_t>(topk->count);
        auto *matrix = ggml_reshape_2d(context, left, width, rows);
        auto *sorted = ggml_argsort(context, matrix,
                                    topk->largest ? GGML_SORT_ORDER_DESC
                                                  : GGML_SORT_ORDER_ASC);
        auto *selected =
            ggml_view_2d(context, sorted, count, rows, sorted->nb[1], 0);
        std::int64_t shape[GGML_MAX_DIMS]{1, 1, 1, 1};
        std::ranges::reverse_copy(recipe.values[node.outputs[0]].dimensions(),
                                  shape);
        auto *indices = ggml_cont_4d(context, selected, shape[0], shape[1],
                                     shape[2], shape[3]);
        auto *row_indices = ggml_reshape_2d(context, indices, count, rows);
        // Reinterpret 32-bit storage, never numerically convert scores to I32.
        // Original GET_ROWS(I32) uses uint loads/stores, retaining NaN
        // payloads, signed zeros and subnormals. Contiguous published tensors
        // own ports.
        auto *raw = ggml_reshape_2d(context, matrix, width, rows);
        raw->type = GGML_TYPE_I32;
        auto *scalars = ggml_reshape_3d(context, raw, 1, width, rows);
        auto *gathered = ggml_get_rows(context, scalars, row_indices);
        auto *floats = ggml_reshape_4d(context, gathered, 1, count, rows, 1);
        floats->type = GGML_TYPE_F32;
        output = ggml_cont_4d(context, floats, shape[0], shape[1], shape[2],
                              shape[3]);
        tensors[node.outputs[1]] = indices;
      } else if (node.operation.get_if<Gather>()) {
        output = ggml_get_rows(context, left, tensors[node.inputs[1]]);
      } else if (node.operation.get_if<Scan>()) {
        const auto width = left->ne[0];
        const auto rows = ggml_nelements(left) / width;
        const auto tile = rows > 262144 ? 262144 : 512;
        if (rows > 512 && rows % tile != 0) {
          // Original single-pass CUMSUM rounds the XYZ row dispatch and has
          // no row guard. At most three native pieces cover it exactly.
          auto *matrix = ggml_reshape_2d(context, left, width, rows);
          std::int64_t offset = 0;
          while (offset < rows) {
            const auto remaining = rows - offset;
            const auto block = remaining > 262144 ? 262144
                               : remaining > 512  ? 512
                                                  : 1;
            const auto count = remaining / block * block;
            auto *part = ggml_view_2d(context, matrix, width, count,
                                      matrix->nb[1], offset * matrix->nb[1]);
            // The multi-pass variant requires descriptor-aligned storage.
            if (offset != 0)
              part = ggml_cont(context, part);
            part = ggml_cumsum(context, part);
            output = output ? ggml_concat(context, output, part, 1) : part;
            offset += count;
          }
          if (left->ne[2] != 1 || left->ne[3] != 1)
            output = ggml_cont_4d(context, output, left->ne[0], left->ne[1],
                                  left->ne[2], left->ne[3]);
        } else {
          output = ggml_cumsum(context, left);
        }
      } else if (const auto *reduce = node.operation.get_if<Reduce>()) {
        const auto row_reduce = [&](ggml_tensor *value) {
          return reduce->kind == ReductionKind::kMean
                     ? ggml_mean(context, value)
                     : ggml_sum_rows(context, value);
        };
        const auto rows = ggml_nelements(left) / left->ne[0];
        const auto tile = rows > 262144 ? 262144 : 512;
        if (rows > 512 && rows % tile != 0) {
          // SUM_ROWS has no padded-row guard. Preserve the upstream kernel,
          // dispatching at most three exact XYZ row tiles, as for CUMSUM.
          auto *matrix = ggml_reshape_2d(context, left, left->ne[0], rows);
          std::int64_t offset = 0;
          while (offset < rows) {
            const auto remaining = rows - offset;
            const auto block = remaining > 262144 ? 262144
                               : remaining > 512  ? 512
                                                  : 1;
            const auto count = remaining / block * block;
            auto *part = ggml_view_2d(context, matrix, left->ne[0], count,
                                      matrix->nb[1], offset * matrix->nb[1]);
            part = row_reduce(part);
            output = output ? ggml_concat(context, output, part, 1) : part;
            offset += count;
          }
          if (left->ne[2] != 1 || left->ne[3] != 1)
            output = ggml_cont_4d(context, output, 1, left->ne[1], left->ne[2],
                                  left->ne[3]);
        } else {
          output = row_reduce(left);
        }
      } else if (const auto *softmax = node.operation.get_if<Softmax>()) {
        output = ggml_soft_max_ext(
            context, left,
            node.inputs.size() == 2 ? tensors[node.inputs[1]] : nullptr,
            softmax->scale, 0.0F);
      } else if (const auto *norm = node.operation.get_if<RmsNorm>()) {
        output = ggml_rms_norm(context, left, norm->epsilon);
        if (node.inputs.size() > 1)
          output = ggml_mul(context, output, tensors[node.inputs[1]]);
      } else if (const auto *layer = node.operation.get_if<LayerNorm>()) {
        // Original GroupNorm computes centered variance. One group per batch
        // makes each last-axis row a group without changing the shader.
        const auto rows = ggml_nelements(left) / left->ne[0];
        auto *grouped = ggml_reshape_4d(context, left, left->ne[0], 1, 1, rows);
        output = ggml_group_norm(context, grouped, 1, layer->epsilon);
        if (node.inputs.size() == 1) {
          // A published port must own storage, not alias an internal result.
          output = ggml_cont_4d(context, output, left->ne[0], left->ne[1],
                                left->ne[2], left->ne[3]);
        } else {
          output = ggml_reshape_4d(context, output, left->ne[0], left->ne[1],
                                   left->ne[2], left->ne[3]);
          output = ggml_mul(context, output, tensors[node.inputs[1]]);
          if (node.inputs.size() > 2)
            output = ggml_add(context, output, tensors[node.inputs[2]]);
        }
      } else if (const auto *unary = node.operation.get_if<Unary>()) {
        output = ggml_unary(context, left, unary_operation(unary->kind));
      } else if (node.operation.get_if<GatedActivation>()) {
        output = ggml_swiglu_split(context, left, tensors[node.inputs[1]]);
      } else {
        auto *right = tensors[node.inputs[1]];
        const auto kind = node.operation.get_if<Binary>()->kind;
        if (!same_shape(recipe.values[node.inputs[0]],
                        recipe.values[node.outputs[0]])) {
          if (kind == BinaryKind::kAdd || kind == BinaryKind::kMultiply)
            std::swap(left, right);
          else
            // ggml broadcasts only RHS. Preserve noncommutative order by
            // materializing LHS in the vendor graph and allocator.
            left = ggml_repeat(context, left, right);
        }
        switch (kind) {
        case BinaryKind::kAdd:
          output = ggml_add(context, left, right);
          break;
        case BinaryKind::kSubtract:
          output = ggml_sub(context, left, right);
          break;
        case BinaryKind::kMultiply:
          output = ggml_mul(context, left, right);
          break;
        case BinaryKind::kDivide:
          output = ggml_div(context, left, right);
          break;
        default:
          return std::unexpected(
              Status::Internal("ggml Vulkan unqualified binary recipe"));
        }
      }
      tensors[node.outputs[0]] = output;
    }
    auto *graph = ggml_new_graph_custom(context, capacity, false);
    for (const auto output : recipe.outputs) {
      ggml_set_output(tensors[output]);
      ggml_build_forward_expand(graph, tensors[output]);
      state.ports.push_back(tensors[output]);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
      auto *node = ggml_graph_node(graph, i);
      if (node->view_src)
        state.views.push_back(node);
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
  // ggml flattens view_src to the allocation root. Refresh the already frozen
  // views after rebinding a port with a different native byte origin.
  for (auto *view : state_->views)
    view->data = static_cast<char *>(view->view_src->data) + view->view_offs;
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
