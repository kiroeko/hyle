#include "hyle/graph/executable.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/resource_budget_access.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/detail/input_source.h"
#include "hyle/graph/detail/region_adapter.h"

#include "hyle/obs/trace.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "../compute/provider_internal.h"
#include "../compute_ops/view_alias.h"
#include "compile_internal.h"
#include "graph_state.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/transfer.h"
#include "hyle/core/assert.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/container/hash.h"
#include "hyle/util/scope_guard.h"
#include "plan_state.h"
#include "view_domain.h"
#include "submission_internal.h"

namespace hyle::graph {
using detail::delegated_step_index;
using detail::enqueue_segment_steps;
using detail::shared_lane;

namespace {

using detail::add_compiled_storage;

Status plan_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph plan metadata allocation failed");
}

Status invocation_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph invocation allocation failed");
}

std::uint64_t hash_word(std::uint64_t hash, std::uint64_t word) noexcept {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  for (std::size_t byte = 0; byte < sizeof(word); ++byte) {
    hash ^= word & 0xffU;
    hash *= kPrime;
    word >>= 8;
  }
  return hash;
}

Result<compute::TensorDesc>
resolve_spec(const detail::GraphState &graph, const TensorSpec &spec,
             std::span<const std::int64_t> values) noexcept {
  if (spec.kind() == TensorSpecKind::kConcrete)
    return spec.concrete_description();

  std::array<std::int64_t, compute::kMaxTensorRank> extents{};
  for (std::size_t index = 0; index < spec.rank(); ++index) {
    const Dimension dimension = spec.dimensions()[index];
    if (dimension.is_fixed()) {
      extents[index] = dimension.fixed_value();
      continue;
    }
    if (!detail::owns(graph, dimension.symbol())) {
      return std::unexpected(
          Status::Internal("graph tensor spec contains a foreign dimension"));
    }
    extents[index] = values[dimension.symbol().ordinal()];
  }
  const std::span<const std::int64_t> shape(extents.data(), spec.rank());
  if (spec.kind() == TensorSpecKind::kDense)
    return compute::TensorDesc::dense(spec.data_type(), shape);
  if (spec.kind() == TensorSpecKind::kBlockedDense) {
    return compute::TensorDesc::blocked_dense(spec.data_type(), shape,
                                              spec.storage_format());
  }
  return std::unexpected(Status::Internal("graph tensor spec kind is invalid"));
}

bool has_dynamic_axes(const detail::GraphState &graph,
                      const TensorSpec &spec) noexcept {
  // Fixed extents do not imply fixed strides: a preceding dynamic View can
  // change layout without changing this tensor's logical shape.
  if (spec.kind() == TensorSpecKind::kInferred &&
      !compute::detail::ShapeProgramView{graph.dimensions, graph.shape_guards}
           .exact())
    return true;
  for (const Dimension axis : spec.dimensions())
    if (!axis.is_fixed() &&
        !graph.dimensions[axis.symbol().ordinal()].bounds.exact())
      return true;
  return false;
}

bool has_dynamic_shape(const detail::GraphState &graph,
                       const detail::NodeRecord &node) noexcept {
  if (const auto *view = detail::view_spec(graph, node)) {
    const auto dynamic = [&](Dimension value) noexcept {
      return value.is_symbol() &&
             !graph.dimensions[value.symbol().ordinal()].bounds.exact();
    };
    for (const auto dimension : view->dimensions())
      if (dynamic(dimension))
        return true;
    if (view->kind() == compute::ViewKind::kSlice &&
        (dynamic(view->start()) || dynamic(view->count())))
      return true;
  }
  for (std::size_t port = 0; port < node.input_count; ++port)
    if (has_dynamic_axes(
            graph, graph.values[graph.edges[node.input_begin + port]].spec))
      return true;
  for (std::size_t port = 0; port < node.output_count; ++port)
    if (has_dynamic_axes(graph, graph.values[node.output_begin + port].spec))
      return true;
  return false;
}

// Borrowed only during one proposal/prepare call. Core copies the domain into
// the prepared root; fixed nodes keep the ordinary exact path even if some
// unrelated node uses dynamic dimensions.
class NodeShapeScratch final {
public:
  explicit NodeShapeScratch(mem::AllocatorRef allocator) noexcept
      : inputs_(allocator), outputs_(allocator) {}

  Result<compute::detail::ProviderShapeDomain>
  bind(const detail::PlanState &plan, std::size_t ordinal) noexcept {
    const auto *graph = detail::GraphAccess::state(plan.domain_graph);
    if (graph == nullptr || !has_dynamic_shape(*graph, graph->nodes[ordinal]))
      return compute::detail::ProviderShapeDomain{};
    const auto &node = graph->nodes[ordinal];
    if (!inputs_.try_resize(node.input_count) ||
        !outputs_.try_resize(node.output_count))
      return std::unexpected(plan_allocation_failure());
    const auto axes = [&](std::size_t value) noexcept {
      const auto &spec = graph->values[value].spec;
      compute::detail::ProviderTensorAxes result;
      for (std::size_t axis = 0; axis < spec.dimensions().size(); ++axis)
        if (!spec.dimensions()[axis].is_fixed())
          result.values[axis] = spec.dimensions()[axis].symbol().ordinal();
      if (detail::has_strided_view_layout(*graph, value)) {
        result.layout = compute::detail::ProviderTensorLayout::kBoundedStrided;
        result.maximum_bytes =
            plan.descriptions[value].required_bytes().value();
      }
      return result;
    };
    for (std::size_t port = 0; port < node.input_count; ++port)
      inputs_[port] = axes(graph->edges[node.input_begin + port]);
    for (std::size_t port = 0; port < node.output_count; ++port)
      outputs_[port] = axes(node.output_begin + port);
    return compute::detail::ProviderShapeDomain{.dimensions = graph->dimensions,
                                                .inputs = inputs_,
                                                .outputs = outputs_,
                                                .guards = graph->shape_guards};
  }

private:
  container::SmallVector<compute::detail::ProviderTensorAxes, 4,
                         mem::AllocatorRef>
      inputs_, outputs_;
};

Status
validate_specialized_operations(const detail::GraphState &graph,
                                std::span<compute::TensorDesc> descriptions,
                                mem::AllocatorRef allocator,
                                std::span<const std::int64_t> values,
                                bool fixed_nodes_only = false) noexcept {
  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions inputs(allocator);
  Descriptions inferred(allocator);
  for (const detail::NodeRecord &node : graph.nodes) {
    if (fixed_nodes_only && has_dynamic_shape(graph, node))
      continue;
    if (!inputs.try_resize(node.input_count) ||
        !inferred.try_resize(node.output_count)) {
      return plan_allocation_failure();
    }
    for (std::size_t index = 0; index < node.input_count; ++index) {
      inputs[index] = descriptions[graph.edges[node.input_begin + index]];
    }
    HYLE_ASSIGN_OR_RETURN(const auto operation,
                          detail::bind_operation(graph, node, values));
    Status status =
        compute::infer_output_descriptions(operation, inputs, inferred);
    if (!status.ok())
      return status.WithContext("graph specialization operation inference");
    for (std::size_t index = 0; index < node.output_count; ++index) {
      const auto ordinal = node.output_begin + index;
      const auto &spec = graph.values[ordinal].spec;
      if (spec.kind() == TensorSpecKind::kInferred) {
        const auto &actual = inferred[index];
        if (actual.data_type() != spec.data_type() ||
            actual.storage_format() != spec.storage_format() ||
            actual.rank() != spec.rank())
          return Status::InvalidArgument(
              "inferred graph output type/format/rank differs from constraint");
        for (std::size_t axis = 0; axis < spec.rank(); ++axis) {
          const auto dimension = spec.dimensions()[axis];
          const auto expected = dimension.is_fixed()
                                    ? dimension.fixed_value()
                                    : values[dimension.symbol().ordinal()];
          if (actual.dimensions()[axis] != expected)
            return Status::InvalidArgument(
                "inferred graph output shape differs from constraint");
        }
        descriptions[ordinal] = actual;
        continue;
      }
      if (inferred[index] != descriptions[node.output_begin + index]) {
        return Status::InvalidArgument(
            "graph specialization output spec does not match operation "
            "inference");
      }
    }
  }
  return Status::Ok();
}

using ShapeValues = container::SmallVector<std::int64_t, 4, mem::AllocatorRef>;

Status bind_shape_values(const detail::GraphState &graph,
                         std::span<const ShapeBinding> bindings,
                         mem::AllocatorRef allocator,
                         ShapeValues &values) noexcept {
  using Flags = container::SmallVector<std::uint8_t, 4, mem::AllocatorRef>;
  Flags seen(allocator);
  if (!values.try_resize(graph.dimensions.size()) ||
      !seen.try_resize(graph.dimensions.size())) {
    return plan_allocation_failure();
  }
  const compute::detail::ShapeProgramView shape{graph.dimensions,
                                                graph.shape_guards};
  if (bindings.size() != shape.input_count()) {
    return Status::InvalidArgument("shape requires exactly one binding for "
                                   "every independent graph dimension");
  }
  for (std::size_t index = 0; index < seen.size(); ++index)
    seen[index] = 0;
  for (const ShapeBinding binding : bindings) {
    if (!detail::owns(graph, binding.dimension)) {
      return Status::InvalidArgument(
          "shape binding does not belong to this graph");
    }
    const std::size_t ordinal = binding.dimension.ordinal();
    if (graph.dimensions[ordinal].kind !=
        compute::detail::ShapeDimensionKind::kInput) {
      return Status::InvalidArgument(
          "derived graph dimensions cannot be bound by the caller");
    }
    if (seen[ordinal] != 0) {
      return Status::InvalidArgument("shape binding is duplicated");
    }
    values[ordinal] = binding.value;
    seen[ordinal] = 1;
  }
  // Equal counts, valid input IDs, and no duplicates prove every input slot is
  // filled. Evaluate derived values and guards before consulting any Provider.
  return compute::detail::evaluate_shape_program(shape, values);
}

Result<SpecializationSignature> make_shape_signature(
    const detail::GraphState &graph, std::span<const std::int64_t> values,
    mem::AllocatorRef allocator) noexcept;

template <std::size_t InlineCount>
Result<SpecializationSignature> specialize_bound_descriptions(
    const detail::GraphState &graph, std::span<const std::int64_t> values,
    mem::AllocatorRef allocator,
    container::SmallVector<compute::TensorDesc, InlineCount, mem::AllocatorRef>
        &descriptions) noexcept {
  if (!descriptions.try_resize(graph.values.size()))
    return std::unexpected(plan_allocation_failure());

  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    if (graph.values[index].spec.kind() == TensorSpecKind::kInferred) {
      descriptions[index] = {};
      continue;
    }
    auto description = resolve_spec(graph, graph.values[index].spec, values);
    if (!description)
      return std::unexpected(std::move(description).error());
    descriptions[index] = *description;
  }
  Status semantic = validate_specialized_operations(
      graph,
      std::span<compute::TensorDesc>(descriptions.data(), descriptions.size()),
      allocator, values);
  if (!semantic.ok())
    return std::unexpected(std::move(semantic));

  return make_shape_signature(graph, values, allocator);
}

Result<SpecializationSignature> make_shape_signature(
    const detail::GraphState &graph, std::span<const std::int64_t> values,
    mem::AllocatorRef allocator) noexcept {
  void *storage = allocator.allocate(sizeof(detail::SignatureState),
                                     alignof(detail::SignatureState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *state =
      ::new (storage) detail::SignatureState(allocator, graph.generation);
  if (!state->bindings.try_resize(graph.dimensions.size())) {
    detail::release(state);
    return std::unexpected(plan_allocation_failure());
  }
  std::uint64_t hash = 14695981039346656037ULL;
  hash = hash_word(hash, graph.generation);
  for (std::size_t index = 0; index < graph.dimensions.size(); ++index) {
    state->bindings[index] = ShapeBinding{
        .dimension = detail::GraphIdentity::dimension(
            graph.generation, static_cast<std::uint32_t>(index)),
        .value = values[index],
    };
    hash = hash_word(hash, index);
    hash = hash_word(hash, static_cast<std::uint64_t>(values[index]));
  }
  state->hash = static_cast<std::size_t>(hash);
  return detail::SpecializationSignatureAccess::adopt(state);
}

std::optional<std::size_t>
find_exact_variant(const detail::PlanState &plan,
                   std::span<const std::int64_t> values) noexcept {
  const auto bindings = [&](std::uint32_t index) noexcept {
    const auto *child =
        detail::ExecutableAccess::state(plan.exact_variants[index]);
    return std::span<const ShapeBinding>(
        detail::SpecializationSignatureAccess::state(child->specialization)
            ->bindings);
  };
  const auto position = std::lower_bound(
      plan.variant_order.begin(), plan.variant_order.end(), values,
      [&](std::uint32_t index, std::span<const std::int64_t> point) noexcept {
        const auto candidate = bindings(index);
        for (std::size_t axis = 0; axis < point.size(); ++axis) {
          if (candidate[axis].value != point[axis])
            return candidate[axis].value < point[axis];
        }
        return false;
      });
  if (position == plan.variant_order.end())
    return std::nullopt;
  const auto candidate = bindings(*position);
  for (std::size_t axis = 0; axis < values.size(); ++axis)
    if (candidate[axis].value != values[axis])
      return std::nullopt;
  return *position;
}

Status describe_domain(const detail::GraphState &graph,
                       mem::AllocatorRef allocator,
                       detail::TensorDescriptions &descriptions,
                       ShapeValues &envelope) noexcept {
  HYLE_RETURN_IF_ERROR(compute::detail::validate_shape_program(
      {graph.dimensions, graph.shape_guards}));
  for (auto value : graph.constants)
    if (has_dynamic_axes(graph, graph.values[value].spec))
      return Status::InvalidArgument("domain constants must have fixed shapes; "
                                     "use an input for dynamic values");
  if (!envelope.try_resize(graph.dimensions.size()) ||
      !descriptions.try_resize(graph.values.size()))
    return plan_allocation_failure();
  // This box need not be an admitted point. Do not evaluate guards/derived
  // expressions or infer dynamic operations at its corner.
  for (std::size_t index = 0; index < envelope.size(); ++index)
    envelope[index] = graph.dimensions[index].bounds.last();
  const compute::detail::ShapeProgramView program{graph.dimensions,
                                                  graph.shape_guards};
  if (program.exact()) {
    HYLE_RETURN_IF_ERROR(
        compute::detail::evaluate_shape_program(program, envelope));
  }
  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    if (graph.values[index].spec.kind() == TensorSpecKind::kInferred) {
      descriptions[index] = {};
      continue;
    }
    HYLE_ASSIGN_OR_RETURN(
        descriptions[index],
        resolve_spec(graph, graph.values[index].spec, envelope));
  }
  if (!program.exact())
    HYLE_RETURN_IF_ERROR(
        detail::prove_view_domains(graph, descriptions, allocator));
  return validate_specialized_operations(graph, descriptions, allocator,
                                         envelope, !program.exact());
}

const compute::TensorDesc &
logical_description(const detail::InvocationState &invocation,
                    std::size_t value) noexcept {
  return invocation.plan->domain_graph.valid()
             ? invocation.concrete_descriptions[value]
             : invocation.plan->descriptions[value];
}

const compute::TensorDesc &
physical_description(const detail::InvocationState &invocation,
                     std::size_t instance) noexcept {
  return invocation.plan->domain_graph.valid()
             ? invocation.concrete_descriptions[invocation.plan
                                                    ->instance_values[instance]]
             : invocation.plan->instance_descriptions[instance];
}

Result<std::size_t>
bound_alias_offset(const detail::InvocationState &invocation,
                   std::size_t instance) noexcept {
  const auto &plan = *invocation.plan;
  const auto *graph = detail::GraphAccess::state(plan.domain_graph);
  if (graph == nullptr ||
      plan.alias_sources[instance] == detail::kNoInPlaceSource)
    return plan.alias_offsets[instance];
  const auto producer = graph->values[plan.instance_values[instance]].producer;
  if (producer == detail::kNoNode ||
      graph->nodes[producer].kind() != compute::OperationKind::kView)
    return plan.alias_offsets[instance];
  const auto &node = graph->nodes[producer];
  const auto *symbolic = detail::view_spec(*graph, node);
  compute::View view;
  if (symbolic) {
    if (symbolic->kind() != compute::ViewKind::kSlice)
      return std::size_t{0};
    const auto &bindings =
        detail::SpecializationSignatureAccess::state(invocation.specialization)
            ->bindings;
    const auto value = [&](Dimension dimension) noexcept {
      return dimension.is_fixed()
                 ? dimension.fixed_value()
                 : bindings[dimension.symbol().ordinal()].value;
    };
    HYLE_ASSIGN_OR_RETURN(
        view, compute::View::slice(symbolic->axis(), value(symbolic->start()),
                                   value(symbolic->count()), symbolic->step()));
  } else {
    view = *node.operation().get_if<compute::View>();
  }
  return compute::detail::view_alias_offset(
      view, physical_description(invocation, plan.alias_sources[instance]),
      physical_description(invocation, instance));
}

Result<mem::BufferView>
workspace_tensor_storage(detail::InvocationState &invocation,
                         std::uint32_t instance) noexcept {
  const auto &plan = *invocation.plan;
  if (!plan.domain_graph.valid())
    return invocation.workspace.view(plan.workspace_requests[instance]);
  const auto source = plan.alias_sources[instance];
  if (source == detail::kNoInPlaceSource)
    return invocation.workspace.view(plan.workspace_requests[instance]);
  // Physical instances are created producer-first. Reuse the already-bound
  // parent instead of walking the complete alias chain for every descendant.
  if (source >= instance || !invocation.writable[source].valid())
    return std::unexpected(
        Status::Internal("graph workspace alias parent is unavailable"));
  HYLE_ASSIGN_OR_RETURN(const auto offset,
                        bound_alias_offset(invocation, instance));
  HYLE_ASSIGN_OR_RETURN(
      const auto bytes,
      physical_description(invocation, instance).required_bytes());
  return invocation.writable[source].storage().subview(offset, bytes);
}

Result<Graph>
make_specialized_graph(const detail::GraphState &source,
                       std::span<const compute::TensorDesc> descriptions,
                       mem::AllocatorRef allocator,
                       std::span<const std::int64_t> values,
                       bool retain_shape_program = true) noexcept {
  detail::BuilderState builder(source.generation, allocator);
  if (!builder.dimensions.try_resize(retain_shape_program ? source.dimensions.size() : 0) ||
      !builder.shape_guards.try_resize(retain_shape_program ? source.shape_guards.size() : 0) ||
      !builder.values.try_resize(source.values.size()) ||
      !builder.nodes.try_resize(source.nodes.size()) ||
      !builder.edges.try_resize(source.edges.size()) ||
      !builder.resource_edges.try_resize(source.resource_edges.size()) ||
      !builder.inputs.try_resize(source.inputs.size()) ||
      !builder.constants.try_resize(source.constants.size()) ||
      !builder.outputs.try_resize(source.outputs.size()) ||
      !builder.dependencies.try_resize(source.dependencies.size())) {
    return std::unexpected(plan_allocation_failure());
  }
  for (std::size_t index = 0; index < builder.dimensions.size(); ++index)
    builder.dimensions[index] = source.dimensions[index];
  for (std::size_t index = 0; index < builder.shape_guards.size(); ++index)
    builder.shape_guards[index] = source.shape_guards[index];
  for (std::size_t index = 0; index < source.values.size(); ++index) {
    auto spec = TensorSpec::concrete(descriptions[index]);
    if (!spec)
      return std::unexpected(std::move(spec).error());
    builder.values[index] = source.values[index];
    builder.values[index].spec = *spec;
    builder.values[index].description = descriptions[index];
  }
  for (std::size_t index = 0; index < source.nodes.size(); ++index) {
    builder.nodes[index] = source.nodes[index];
    HYLE_ASSIGN_OR_RETURN(
        builder.nodes[index].descriptor,
        detail::bind_operation(source, source.nodes[index], values));
  }
  for (std::size_t index = 0; index < source.edges.size(); ++index)
    builder.edges[index] = source.edges[index];
  builder.effect_resource_count = source.effect_resource_count;
  for (std::size_t index = 0; index < source.resource_edges.size(); ++index)
    builder.resource_edges[index] = source.resource_edges[index];
  for (std::size_t index = 0; index < source.inputs.size(); ++index)
    builder.inputs[index] = source.inputs[index];
  for (std::size_t index = 0; index < source.constants.size(); ++index)
    builder.constants[index] = source.constants[index];
  for (std::size_t index = 0; index < source.outputs.size(); ++index)
    builder.outputs[index] = source.outputs[index];
  for (std::size_t index = 0; index < source.dependencies.size(); ++index)
    builder.dependencies[index] = source.dependencies[index];

  void *storage = allocator.allocate(sizeof(detail::GraphState),
                                     alignof(detail::GraphState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *state = ::new (storage) detail::GraphState(builder);
  return detail::GraphAccess::adopt(state);
}

bool owns(const detail::PlanState &plan, ValueId value) noexcept {
  if (!value.valid() ||
      detail::GraphIdentity::generation(value) != plan.generation ||
      value.ordinal() >= plan.descriptions.size()) {
    return false;
  }
  const std::uint8_t role = plan.binding_roles[value.ordinal()];
  const ValueKind expected =
      (role & detail::kExternalInput) != 0
          ? ValueKind::kInput
          : ((role & detail::kConstant) != 0 ? ValueKind::kConstant
                                             : ValueKind::kNodeOutput);
  return value.kind() == expected;
}

struct ByteRange final {
  mem::MemoryResourceRef resource{};
  void *handle = nullptr;
  std::size_t offset = 0;
  std::size_t size = 0;
};

ByteRange byte_range(mem::ConstBufferView view) noexcept {
  return {view.resource(), view.native_handle(), view.offset(), view.size()};
}

ByteRange byte_range(mem::BufferView view) noexcept {
  return {view.resource(), view.native_handle(), view.offset(), view.size()};
}

bool same_allocation(const ByteRange &left, const ByteRange &right) noexcept {
  return left.resource == right.resource && left.handle == right.handle;
}

bool equal_range(const ByteRange &left, const ByteRange &right) noexcept {
  return same_allocation(left, right) && left.offset == right.offset &&
         left.size == right.size;
}

bool overlaps(const ByteRange &left, const ByteRange &right) noexcept {
  if (!same_allocation(left, right) || left.size == 0 || right.size == 0)
    return false;
  if (left.offset <= right.offset)
    return right.offset - left.offset < left.size;
  return left.offset - right.offset < right.size;
}

bool overlaps(EffectResourceFootprint left,
              EffectResourceFootprint right) noexcept {
  if (left.is_memory() != right.is_memory())
    return false;
  if (left.is_memory())
    return overlaps(byte_range(left.memory_range()),
                    byte_range(right.memory_range()));
  return left.object_identity() == right.object_identity();
}

Status validate_effect_tensor_alias(const detail::PlanState &plan,
                                    ByteRange tensor) noexcept {
  for (const auto &binding : plan.effect_bindings) {
    if (binding.footprint.is_memory() &&
        overlaps(byte_range(binding.footprint.memory_range()), tensor)) {
      return Status::InvalidArgument(
          "effect resource memory must not alias graph tensor storage");
    }
  }
  return Status::Ok();
}

using EffectBindings = detail::EffectBindings;

Status collect_effect_bindings(
    const detail::GraphState &graph, std::span<const EffectResourceBinding> bindings,
    std::span<const ConstantBinding> constants, mem::AllocatorRef allocator, EffectBindings &result) noexcept {
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> used(
      allocator);
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> bound(
      allocator);
  if (!used.try_resize(graph.effect_resource_count) ||
      !bound.try_resize(graph.effect_resource_count))
    return plan_allocation_failure();
  std::fill(used.begin(), used.end(), std::uint8_t{0});
  std::fill(bound.begin(), bound.end(), std::uint8_t{0});
  std::size_t used_count = 0;
  for (const auto &node : graph.nodes) {
    const auto count = node.traits().resource_count;
    for (std::size_t port = 0; port < count; ++port) {
      const auto resource = graph.resource_edges[node.resource_begin + port];
      if (!used[resource]) {
        used[resource] = 1;
        ++used_count;
      }
    }
  }
  if (!result.try_reserve(used_count))
    return plan_allocation_failure();
  for (const auto &binding : bindings) {
    if (!detail::owns(graph, binding.resource) || !binding.footprint.valid())
      return Status::InvalidArgument(
          "invalid effect resource footprint binding");
    const auto ordinal = binding.resource.ordinal();
    if (bound[ordinal])
      return Status::InvalidArgument(
          "effect resource was bound more than once");
    bound[ordinal] = 1;
    if (!used[ordinal])
      continue;
    for (const auto &previous : result) {
      if (overlaps(previous.footprint, binding.footprint))
        return Status::InvalidArgument("different effect resource IDs alias; "
                                       "use one ID or disjoint ranges");
    }
    if (!result.try_push_back(binding))
      return plan_allocation_failure();
  }
  for (std::size_t resource = 0; resource < used.size(); ++resource) {
    if (used[resource] && !bound[resource])
      return Status::InvalidArgument(
          "graph effect resource has no backing footprint");
  }
  for (const auto &binding : constants)
    HYLE_RETURN_IF_ERROR(detail::validate_effect_tensor_alias(result, binding.tensor.storage()));
  return Status::Ok();
}

Status bind_effect_resources(detail::PlanState &plan,
                             const detail::GraphState &graph,
                             CompileOptions options) noexcept {
  return collect_effect_bindings(graph, options.effect_bindings, options.constants,
                                 plan.allocator, plan.effect_bindings);
}

bool matches_alias_range(ByteRange source, ByteRange output,
                         std::size_t byte_offset) noexcept {
  return same_allocation(source, output) && output.offset >= source.offset &&
         output.offset - source.offset == byte_offset &&
         byte_offset <= source.size && output.size <= source.size - byte_offset;
}

Result<bool> matches_planned_alias(const detail::InvocationState &invocation,
                                   std::size_t left_value, ByteRange left,
                                   std::size_t right_value,
                                   ByteRange right) noexcept {
  const auto &plan = *invocation.plan;
  const std::uint32_t left_ordinal = plan.external_instances[left_value];
  const std::uint32_t right_ordinal = plan.external_instances[right_value];
  if (left_ordinal == detail::kNoWorkspaceRequest ||
      right_ordinal == detail::kNoWorkspaceRequest) {
    return false;
  }
  if (plan.alias_sources[left_ordinal] == right_ordinal) {
    HYLE_ASSIGN_OR_RETURN(const auto offset,
                          bound_alias_offset(invocation, left_ordinal));
    return matches_alias_range(right, left, offset);
  }
  if (plan.alias_sources[right_ordinal] == left_ordinal) {
    HYLE_ASSIGN_OR_RETURN(const auto offset,
                          bound_alias_offset(invocation, right_ordinal));
    return matches_alias_range(left, right, offset);
  }
  return false;
}

ByteRange bound_range(const detail::InvocationState &invocation,
                      std::size_t logical_ordinal) noexcept {
  const std::uint32_t ordinal =
      invocation.plan->external_instances[logical_ordinal];
  const std::uint8_t role = invocation.plan->binding_roles[logical_ordinal];
  if ((role & detail::kConstant) != 0)
    return byte_range(invocation.readable[ordinal].storage());
  if ((role & detail::kExternalOutput) != 0)
    return byte_range(invocation.writable[ordinal].storage());
  return byte_range(invocation.readable[ordinal].storage());
}

ValueId external_alias_value(const detail::PlanState &plan,
                            std::uint32_t entry) noexcept {
  if (entry < plan.input_ids.size())
    return plan.input_ids[entry];
  return plan.output_ids[entry - plan.input_ids.size()];
}

Status create_external_alias_index(detail::InvocationState &invocation) noexcept {
  const auto &plan = *invocation.plan;
  const auto inputs = plan.input_ids.size(), outputs = plan.output_ids.size();
  if (outputs > SIZE_MAX - inputs || inputs + outputs >= detail::kNoNode)
    return invocation_allocation_failure();
  const auto count = inputs + outputs;
  if (count == 0)
    return Status::Ok();
  std::size_t buckets = 1;
  while (buckets < count) {
    if (buckets > SIZE_MAX / 2)
      return invocation_allocation_failure();
    buckets *= 2;
  }
  if (buckets > SIZE_MAX / 2 / sizeof(std::uint32_t) ||
      count > SIZE_MAX / sizeof(detail::ExternalAliasEntry) ||
      plan.instance_descriptions.size() > SIZE_MAX / sizeof(std::uint32_t))
    return invocation_allocation_failure();
  if (!invocation.external_alias_heads.try_resize(2 * buckets) ||
      !invocation.external_alias_entries.try_resize(count) ||
      !invocation.external_alias_instances.try_resize(
          plan.instance_descriptions.size()))
    return invocation_allocation_failure();
  std::fill(invocation.external_alias_heads.begin(),
            invocation.external_alias_heads.end(), detail::kNoNode);
  std::fill(invocation.external_alias_instances.begin(),
            invocation.external_alias_instances.end(), detail::kNoNode);
  for (std::size_t index = count; index-- != 0;) {
    const auto value = external_alias_value(plan, static_cast<std::uint32_t>(index));
    const auto instance = plan.external_instances[value.ordinal()];
    if (instance != detail::kNoWorkspaceRequest) {
      auto &entry = invocation.external_alias_entries[index];
      entry.same_instance_next = invocation.external_alias_instances[instance];
      invocation.external_alias_instances[instance] = static_cast<std::uint32_t>(index);
    }
  }
  return Status::Ok();
}

std::size_t external_alias_bucket(const detail::InvocationState &invocation,
                                 ByteRange range, bool output) noexcept {
  // Hash opaque identities without interpreting either as a host address.
  const auto hash = container::Hash<std::pair<const mem::MemoryResource *, void *>>{}(
      {&range.resource.resource(), range.handle});
  const auto count = invocation.external_alias_heads.size() / 2;
  return (static_cast<std::size_t>(hash) & (count - 1)) + (output ? count : 0);
}

void rebuild_external_alias_index(detail::InvocationState &invocation) noexcept {
  std::fill(invocation.external_alias_heads.begin(),
            invocation.external_alias_heads.end(), detail::kNoNode);
  // Reverse insertion preserves the original per-list order in every bucket.
  for (std::size_t index = invocation.external_alias_entries.size(); index-- != 0;) {
    auto &entry = invocation.external_alias_entries[index];
    entry.bucket = SIZE_MAX;
    entry.next = entry.previous = detail::kNoNode;
    const auto value = external_alias_value(*invocation.plan,
                                           static_cast<std::uint32_t>(index));
    const auto instance = invocation.plan->external_instances[value.ordinal()];
    if (instance == detail::kNoWorkspaceRequest || !invocation.bound[instance])
      continue;
    const auto range = bound_range(invocation, value.ordinal());
    if (range.size == 0)
      continue; // Empty ranges cannot overlap, including opaque zero-byte views.
    entry.bucket = external_alias_bucket(invocation, range,
        index >= invocation.plan->input_ids.size());
    auto &head = invocation.external_alias_heads[entry.bucket];
    entry.next = head;
    if (head != detail::kNoNode)
      invocation.external_alias_entries[head].previous = static_cast<std::uint32_t>(index);
    head = static_cast<std::uint32_t>(index);
  }
  invocation.external_alias_dirty = false;
}

void commit_external_alias_binding(detail::InvocationState &invocation,
                                   std::size_t ordinal) noexcept {
  if (invocation.external_alias_dirty) {
    // A generated/projected external range changed since the previous bind.
    // Rebuild once from committed bindings, never once per ordinary bind.
    rebuild_external_alias_index(invocation);
    return;
  }
  const auto instance = invocation.plan->external_instances[ordinal];
  for (auto index = invocation.external_alias_instances[instance];
       index != detail::kNoNode;
       index = invocation.external_alias_entries[index].same_instance_next) {
    auto &entry = invocation.external_alias_entries[index];
    if (entry.bucket != SIZE_MAX) {
      if (entry.previous == detail::kNoNode)
        invocation.external_alias_heads[entry.bucket] = entry.next;
      else
        invocation.external_alias_entries[entry.previous].next = entry.next;
      if (entry.next != detail::kNoNode)
        invocation.external_alias_entries[entry.next].previous = entry.previous;
    }
    entry.bucket = SIZE_MAX;
    entry.next = entry.previous = detail::kNoNode;
    const auto value = external_alias_value(*invocation.plan, index);
    const auto range = bound_range(invocation, value.ordinal());
    if (range.size == 0)
      continue;
    entry.bucket = external_alias_bucket(invocation, range,
        index >= invocation.plan->input_ids.size());
    auto next = invocation.external_alias_heads[entry.bucket];
    auto previous = detail::kNoNode;
    while (next != detail::kNoNode && next < index) {
      previous = next;
      next = invocation.external_alias_entries[next].next;
    }
    entry.previous = previous;
    entry.next = next;
    if (previous == detail::kNoNode)
      invocation.external_alias_heads[entry.bucket] = index;
    else
      invocation.external_alias_entries[previous].next = index;
    if (next != detail::kNoNode)
      invocation.external_alias_entries[next].previous = index;
  }
}

void note_external_alias_projection(detail::InvocationState &invocation,
                                    std::uint32_t instance,
                                    compute::ConstTensorView tensor,
                                    bool bound) noexcept {
  // Membership comes from the actual logical input/output lists, including
  // published generated and source-aliased outputs, not a role-bit shortcut.
  if (!invocation.external_alias_instances.empty() &&
      invocation.external_alias_instances[instance] != detail::kNoNode &&
      (bool(invocation.bound[instance]) != bound ||
       !equal_range(byte_range(invocation.readable[instance].storage()),
                    byte_range(tensor.storage()))))
    invocation.external_alias_dirty = true;
}

// The index contains positions in the existing immutable physical binding
// arrays, not another payload/permission owner. Values below constant_ids.size()
// are C entries; the remaining positions are P entries in prepared_values.
ByteRange immutable_alias_range(const detail::PlanState &plan,
                               std::size_t index) noexcept {
  if (index < plan.constant_ids.size())
    return byte_range(plan.constant_bindings[index].storage());
  return byte_range(
      plan.prepared_values[index - plan.constant_ids.size()].binding.storage());
}

// Opaque handles are only equality/order keys here, never interpreted as CPU
// addresses. std::less supplies a total pointer order across allocations.
bool allocation_less(ByteRange left, ByteRange right) noexcept {
  const auto *a = &left.resource.resource();
  const auto *b = &right.resource.resource();
  if (a != b)
    return std::less<const mem::MemoryResource *>{}(a, b);
  return std::less<void *>{}(left.handle, right.handle);
}

Status build_immutable_alias_index(detail::PlanState &plan) noexcept {
  auto &order = plan.immutable_alias_order;
  order.clear();
  const auto constants = plan.constant_ids.size();
  if (plan.prepared_values.size() > SIZE_MAX - constants ||
      !order.try_reserve(constants + plan.prepared_values.size()))
    return plan_allocation_failure();
  for (std::size_t index = 0; index < constants; ++index) {
    // Match the old C loop's physical-instance eligibility. Its remaining
    // same-value/bound/planned-alias checks still run at bind time below.
    const auto instance =
        plan.external_instances[plan.constant_ids[index].ordinal()];
    if (instance != detail::kNoWorkspaceRequest &&
        plan.constant_bindings[index].storage().size() != 0 &&
        !order.try_push_back(index))
      return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < plan.prepared_values.size(); ++index)
    if (plan.prepared_values[index].binding.storage().size() != 0 &&
        !order.try_push_back(constants + index))
      return plan_allocation_failure();
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    const auto a = immutable_alias_range(plan, left);
    const auto b = immutable_alias_range(plan, right);
    if (allocation_less(a, b)) return true;
    if (allocation_less(b, a)) return false;
    // Preserve the original C-before-P and per-array order in each group.
    return left < right;
  });
  return Status::Ok();
}

Status build_external_input_topology(detail::PlanState &plan) noexcept {
  auto &aliases = plan.external_input_aliases;
  auto &consumers = plan.external_input_consumers;
  for (std::size_t instance = 0; instance < plan.alias_sources.size(); ++instance) {
    const auto parent = plan.alias_sources[instance];
    if (parent == detail::kNoInPlaceSource || parent == instance)
      continue;
    const auto value = plan.instance_values[parent];
    // Every descendant chain contains a direct child of its physical root.
    if ((plan.binding_roles[value] & detail::kExternalInput) != 0 &&
        plan.external_instances[value] == parent &&
        !aliases.try_push_back(value))
      return plan_allocation_failure();
  }
  if (!aliases.empty()) {
    std::sort(aliases.begin(), aliases.end());
    aliases.erase(std::unique(aliases.begin(), aliases.end()), aliases.end());
  }
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const auto &step = plan.steps[index];
    for (std::uint32_t port = 0; port < step.input_count; ++port) {
      const auto instance = plan.node_inputs[step.input_begin + port];
      const auto value = plan.instance_values[instance];
      // Include other-storage replicas and copies as well as the external
      // root. A projected binding must still reject any unqualified reader.
      if ((plan.binding_roles[value] & detail::kExternalInput) != 0 &&
          !consumers.try_push_back(detail::ExternalInputConsumer{
              value, instance, static_cast<std::uint32_t>(index), port}))
        return plan_allocation_failure();
    }
  }
  if (!consumers.empty())
    std::sort(consumers.begin(), consumers.end(),
              [](const auto &left, const auto &right) noexcept {
                if (left.value != right.value) return left.value < right.value;
                if (left.step != right.step) return left.step < right.step;
                return left.port < right.port;
              });
  return Status::Ok();
}

Status validate_external_alias(detail::InvocationState &invocation,
                               std::size_t ordinal,
                               ByteRange candidate,
                               std::span<const detail::InputSourceBinding> incoming = {}) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  if (invocation.external_alias_dirty)
    rebuild_external_alias_index(invocation);
  const bool candidate_is_output =
      (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0;
  const auto instance = plan.external_instances[ordinal];
  // Every later binding is checked against the already-bound outputs. An
  // unchanged output range therefore remains valid, even after input rebinds;
  // do not rescan all model constants on each repeated output bind. This does
  // not skip ordinary binding/replay invalidation or native input validation.
  if (incoming.empty() && candidate_is_output && instance != detail::kNoWorkspaceRequest &&
      invocation.bound[instance] && invocation.writable[instance].valid() &&
      equal_range(candidate, bound_range(invocation, ordinal)))
    return Status::Ok();
  HYLE_RETURN_IF_ERROR(validate_effect_tensor_alias(plan, candidate));

  auto check = [&](ValueId other_value) noexcept -> Status {
    const std::size_t other = other_value.ordinal();
    const std::uint32_t other_instance = plan.external_instances[other];
    const auto replacement = std::find_if(incoming.begin(), incoming.end(),
        [&](const auto &item) { return item.value == other_value; });
    if (other == ordinal || other_instance == detail::kNoWorkspaceRequest ||
        (replacement == incoming.end() && invocation.bound[other_instance] == 0)) {
      return Status::Ok();
    }
    const bool other_is_output =
        (plan.binding_roles[other] & detail::kExternalOutput) != 0;
    if (!candidate_is_output && !other_is_output)
      return Status::Ok();
    const ByteRange existing = replacement == incoming.end()
        ? bound_range(invocation, other) : byte_range(replacement->tensor.storage());
    if (!overlaps(candidate, existing))
      return Status::Ok();
    const auto immutable_input = [&](std::size_t value) noexcept {
      for (const auto input : plan.input_ids)
        if (input.ordinal() == value)
          return detail::has_input_source(invocation, input);
      return false;
    };
    if ((!candidate_is_output && immutable_input(ordinal)) ||
        (!other_is_output && immutable_input(other)))
      return Status::InvalidArgument("graph output overlaps an immutable retained input");
    HYLE_ASSIGN_OR_RETURN(
        const auto planned,
        matches_planned_alias(invocation, ordinal, candidate, other, existing));
    if (planned) {
      return Status::Ok();
    }
    return Status::InvalidArgument(
        "external graph tensor bindings overlap unsafely");
  };

  auto check_dynamic = [&](bool output) noexcept -> Status {
    if (candidate.size == 0 || invocation.external_alias_heads.empty())
      return Status::Ok();
    const auto bucket = external_alias_bucket(invocation, candidate, output);
    for (auto index = invocation.external_alias_heads[bucket];
         index != detail::kNoNode;
         index = invocation.external_alias_entries[index].next) {
      const auto value = external_alias_value(plan, index);
      // Hash collisions only widen candidate enumeration. The original exact
      // allocation/range test and all permission/planned-alias gates remain.
      if (same_allocation(candidate, bound_range(invocation, value.ordinal())))
        HYLE_RETURN_IF_ERROR(check(value));
    }
    return Status::Ok();
  };

  if (candidate_is_output) {
    HYLE_RETURN_IF_ERROR(check_dynamic(false));
    // Immutable allocations keep their separate compiled index. Dynamic
    // candidates above/below still use the original binding permission gates.
    const auto &order = plan.immutable_alias_order;
    auto current = std::lower_bound(order.begin(), order.end(), candidate,
        [&](std::size_t index, ByteRange range) {
          return allocation_less(immutable_alias_range(plan, index), range);
        });
    for (; current != order.end(); ++current) {
      const auto immutable = immutable_alias_range(plan, *current);
      if (!same_allocation(candidate, immutable))
        break;
      if (*current < plan.constant_ids.size()) {
        // Keep all old C semantics, including same-value/unbound skips and
        // matches_planned_alias; a constant is not unconditionally disjoint.
        HYLE_RETURN_IF_ERROR(check(plan.constant_ids[*current]));
      } else if (overlaps(candidate, immutable)) {
        return Status::InvalidArgument(
            "external graph output overlaps an immutable prepared value");
      }
    }
  }
  HYLE_RETURN_IF_ERROR(check_dynamic(true));
  // The old allocation index cannot enumerate a newly moved input range.
  // Scan only this bounded cold bundle using the same permission/alias gate.
  for (const auto &item : incoming) HYLE_RETURN_IF_ERROR(check(item.value));
  return Status::Ok();
}

struct PreparationScope final {
  detail::InvocationState &state;
  ~PreparationScope() { detail::release_preparation(state); }
};

void reset_capture_cache(detail::InvocationState &invocation) noexcept {
  invocation.captured_ranges.clear();
  invocation.capture_rebuilds.clear();
  for (auto &span : invocation.capture_spans)
    span = {};
  invocation.replay_prepared = false;
  invocation.replay_route = ReplayRoute::kUnprepared;
  invocation.replay_report = {};
}

void invalidate_binding_cache(detail::InvocationState &invocation,
                              bool physical_change = true) noexcept {
  if (physical_change)
    detail::discard_configuration_spans(invocation);
  invocation.bindings_prepared = false;
  invocation.transfers_prepared = false;
  invocation.dependencies_prepared = false;
  invocation.outputs_ready.store(false, std::memory_order_release);
  // Retained commands are dormant until a complete cold preparation proves
  // their current arguments. Permission/source owners never live in this cache.
  invocation.replay_prepared = false;
  invocation.replay_route = ReplayRoute::kUnprepared;
  invocation.replay_report = {};
}

void invalidate_unproven_bindings(detail::InvocationState &invocation) noexcept {
  // An opaque token can change carriers when its owner is withdrawn. Destroy
  // unqualified recordings first, without scanning every step on every bind.
  for (std::size_t index = 0; index < invocation.captured_ranges.size();) {
    const auto &range = invocation.captured_ranges[index];
    if (range.commands.tensor_bindings_have_stable_identity()) {
      ++index;
      continue;
    }
    const std::size_t begin = range.step_begin;
    const std::size_t end = begin + range.step_count;
    for (std::size_t step = begin; step < end; ++step)
      if (invocation.plan->steps[step].emits_command())
        detail::invalidate_step_recording(invocation,
                                          static_cast<std::uint32_t>(step));
  }
}

template <class Tensor>
bool same_physical_tensor(const Tensor &left, const Tensor &right) noexcept {
  if (left.valid() != right.valid() ||
      left.description() != right.description())
    return false;
  const auto a = left.storage(), b = right.storage();
  return a.resource() == b.resource() &&
         a.native_handle() == b.native_handle() &&
         a.offset() == b.offset() && a.size() == b.size();
}

detail::CaptureRebuildRange *capture_storage(detail::InvocationState &invocation,
                                             std::uint32_t begin,
                                             std::uint32_t count) noexcept {
  for (auto &saved : invocation.capture_rebuilds)
    if (saved.step_begin == begin && saved.step_count == count)
      return &saved;
  return nullptr;
}

void discard_capture_variants(detail::CaptureRebuildRange &saved) noexcept {
  for (auto &variant : saved.variants) {
    variant.commands.reset();
    variant.bindings.clear();
    variant.key_valid = false;
  }
  saved.active_variant = 2;
  saved.qualification_valid = false;
  saved.preserves_private_arguments = false;
}

Status qualify_capture_variants(detail::InvocationState &invocation,
                                detail::CaptureRebuildRange &saved) noexcept {
  if (saved.qualification_valid)
    return Status::Ok();
  bool preserves = true;
  std::size_t ports = 0;
  for (std::size_t index = saved.step_begin;
       index < std::size_t{saved.step_begin} + saved.step_count; ++index) {
    const auto &step = invocation.plan->steps[index];
    if (!step.emits_command())
      continue;
    if (invocation.step_bindings[index].native_inputs) {
      preserves = false;
      break;
    }
    if (step.region.valid()) {
      preserves = detail::RegionAccess::preserves_capture_bindings(
          invocation.invocations[index].region);
    } else if (step.kind == detail::PlanStepKind::kOperation) {
      auto inspected = compute::detail::OperationPreparationAccess::preserves_capture_bindings(
          invocation.invocations[index].operation);
      if (!inspected)
        return std::move(inspected).error();
      preserves = *inspected;
    } else {
      preserves = false;
    }
    if (!preserves)
      break;
    if (ports > SIZE_MAX - step.input_count ||
        ports + step.input_count > SIZE_MAX - step.output_count)
      return Status::ResourceExhausted("graph capture binding key is too large");
    ports += std::size_t{step.input_count} + step.output_count;
  }
  if (preserves)
    for (auto &variant : saved.variants)
      if (!variant.bindings.try_reserve(ports))
        return Status::ResourceExhausted("graph capture binding key allocation failed");
  saved.preserves_private_arguments = preserves;
  saved.qualification_valid = true;
  return Status::Ok();
}

bool capture_key_matches(const detail::InvocationState &invocation,
                         const detail::CaptureRebuildRange &saved,
                         const detail::CaptureBindingVariant &variant) noexcept {
  if (!variant.key_valid)
    return false;
  std::size_t port = 0;
  for (std::size_t index = saved.step_begin;
       index < std::size_t{saved.step_begin} + saved.step_count; ++index) {
    const auto &step = invocation.plan->steps[index];
    if (!step.emits_command())
      continue;
    for (std::size_t input = 0; input < step.input_count; ++input)
      if (port == variant.bindings.size() ||
          !same_physical_tensor(variant.bindings[port++],
              invocation.submission_inputs[step.input_begin + input]))
        return false;
    for (std::size_t output = 0; output < step.output_count; ++output)
      if (port == variant.bindings.size() ||
          !same_physical_tensor(variant.bindings[port++],
              invocation.submission_outputs[step.output_begin + output].as_const()))
        return false;
  }
  return port == variant.bindings.size();
}

void remember_capture_key(const detail::InvocationState &invocation,
                          detail::CaptureRebuildRange &saved,
                          std::uint8_t selected) noexcept {
  auto &variant = saved.variants[selected];
  variant.bindings.clear();
  for (std::size_t index = saved.step_begin;
       index < std::size_t{saved.step_begin} + saved.step_count; ++index) {
    const auto &step = invocation.plan->steps[index];
    if (!step.emits_command())
      continue;
    for (std::size_t input = 0; input < step.input_count; ++input) {
      const bool appended = variant.bindings.try_push_back(
          invocation.submission_inputs[step.input_begin + input]);
      HYLE_FATAL_ASSERT(appended);
    }
    for (std::size_t output = 0; output < step.output_count; ++output) {
      const bool appended = variant.bindings.try_push_back(
          invocation.submission_outputs[step.output_begin + output].as_const());
      HYLE_FATAL_ASSERT(appended);
    }
  }
  variant.key_valid = true;
  saved.active_variant = selected;
  saved.last_used = selected;
}

void append_capture_range(detail::InvocationState &invocation,
                           std::uint32_t begin, std::uint32_t count,
                           compute::detail::CapturedCommands commands) noexcept {
  const bool appended = invocation.captured_ranges.try_push_back(
      detail::CapturedStepRange{begin, count, std::move(commands)});
  HYLE_FATAL_ASSERT(appended);
  for (std::size_t index = invocation.captured_ranges.size() - 1;
       index != 0 && invocation.captured_ranges[index - 1].step_begin >
                         invocation.captured_ranges[index].step_begin; --index)
    std::swap(invocation.captured_ranges[index - 1], invocation.captured_ranges[index]);
}

void rebuild_capture_spans(detail::InvocationState &invocation) noexcept {
  std::size_t index = 0;
  for (std::size_t segment_index = 0;
       segment_index < invocation.plan->segments.size(); ++segment_index) {
    auto &span = invocation.capture_spans[segment_index];
    span.begin = index;
    const auto &segment = invocation.plan->segments[segment_index];
    const auto end = segment.step_begin + segment.step_count;
    while (index < invocation.captured_ranges.size() &&
           invocation.captured_ranges[index].step_begin < end)
      ++index;
    span.count = index - span.begin;
  }
}

std::uint32_t alias_root(const detail::PlanState &plan,
                         std::uint32_t value) noexcept;

Status validate_known_binding_alignment(const detail::InvocationState &invocation,
    std::uint32_t instance, mem::ConstBufferView view) noexcept {
  if (!view.valid() || view.empty())
    return Status::Ok();
  const auto &plan = *invocation.plan;
  const auto needed = plan.instance_alignments[instance];
  if (needed <= 1)
    return Status::Ok();
  auto base = view.resource().properties().minimum_alignment;
  if (base < needed) {
    const auto root = alias_root(plan, instance);
    if (plan.workspace_requests[root] != detail::kNoWorkspaceRequest) {
      // The LifetimePlan allocated this backing at the root's propagated
      // alignment. This is a real allocation contract, not a handle cast.
      base = (std::max)(base, plan.instance_alignments[root]);
    } else if ((plan.instance_roles[root] & detail::kConstant) != 0) {
      const auto *owner = plan.owned_instance_constant(root);
      if (owner && owner->resource() == view.resource() && owner->native_handle() == view.native_handle())
        base = (std::max)(base, owner->alignment());
    }
  }
  // Unknown opaque caller alignment stays the typed provider's contract.
  // Reject only an actual offset proven invalid by a known base guarantee.
  if (base >= needed && base % needed == 0 && view.offset() % needed != 0)
    return Status::InvalidArgument("graph binding offset does not satisfy provider port alignment");
  return Status::Ok();
}

Status project_generated_output(detail::InvocationState &invocation,
    std::size_t step_index, std::size_t port) noexcept {
  const auto &plan = *invocation.plan;
  const auto &step = plan.steps[step_index];
  const auto instance = plan.node_outputs[step.output_begin + port];
  const auto contract = step.region.output_contract(port);
  if (contract.kind == RegionOutputKind::kDeferredGenerated) {
    RegionValue value;
    value.description = physical_description(invocation, instance);
    value.reference = invocation.invocations[step_index].region.output_reference(port);
    invocation.native_values[instance] = value;
    note_external_alias_projection(invocation, instance, {}, true);
    invocation.readable[instance] = {};
    invocation.bound[instance] = 1;
    return Status::Ok();
  }
  HYLE_ASSIGN_OR_RETURN(auto value,
      invocation.invocations[step_index].region.output_value(port));
  if (value.description != physical_description(invocation, instance) ||
      value.reference.valid() ||
      (bool(value.native_type) != bool(value.native_handle)) ||
      (contract.projection == RegionOutputProjection::kTensorView && !value.tensor.valid()) ||
      (!value.tensor.valid() && value.native_handle == nullptr))
    return Status::Internal("generated region output has an invalid physical representation");
  if (value.tensor.valid()) {
    const auto &source = plan.targets[step.target.ordinal()].source;
    const auto nominated = plan.storage_domains[plan.instance_storage[instance]].resource;
    if (value.tensor.description() != value.description ||
        !detail::ExecutionSourceAccess::matches_read_resource(source, nominated,
            value.tensor.storage().resource()))
      return Status::Internal("generated region output does not match its admitted storage");
    HYLE_RETURN_IF_ERROR(source.validate_resource(value.tensor.storage().resource()));
    HYLE_RETURN_IF_ERROR(validate_known_binding_alignment(invocation, instance, value.tensor.storage()));
  }
  invocation.native_values[instance] = value;
  note_external_alias_projection(invocation, instance, value.tensor, true);
  invocation.readable[instance] = value.tensor;
  invocation.bound[instance] = 1;
  return Status::Ok();
}

Status seal_invocation_bindings(detail::InvocationState &invocation) noexcept {
  // Seals also validate managed slots whose ports are ordinary TensorViews.
  // Stable addresses cannot preserve a previous episode's preparation fact.
  for (const auto index : invocation.plan->seal_binding_steps) {
    const auto status = invocation.invocations[index].region.seal_bindings();
    if (!status.ok()) {
      invalidate_binding_cache(invocation);
      return status;
    }
  }
  return Status::Ok();
}

Status prepare_command_bindings(detail::InvocationState &invocation,
    std::size_t step_index,
    std::optional<std::span<const std::uint32_t>> &lease_candidates) noexcept {
  const auto &plan = *invocation.plan;
  const auto &step = plan.steps[step_index];
  auto &cached = invocation.step_bindings[step_index];
  // Without a positive actual-backend identity proof, every new binding
  // episode re-runs the cold hook even if no old graph exists (Commands or
  // failed capture) and the opaque raw tuple compares equal.
  bool changed = !cached.stable_binding_identity ||
                 cached.prepared_revision != cached.revision;
  // Native references and owned imports can replace private carriers without
  // changing a raw address. Only ordinary caller TensorView arguments are
  // eligible for equality-based cold binding reuse.
  bool native_inputs = false;
  const bool borrowed_tensor_carrier = step.region.valid() &&
      detail::RegionAccess::borrows_tensor_views_only(step.region);
  for (std::size_t index = 0; index < step.input_count; ++index) {
    const auto instance = plan.node_inputs[step.input_begin + index];
    const compute::ConstTensorView &binding =
        invocation.readable[instance];
    const bool native = step.region.valid() && step.region.prepares_bindings();
    if (!binding.valid() && !(native &&
        !invocation.native_values.empty() &&
        (invocation.native_values[instance].native_handle != nullptr ||
         invocation.native_values[instance].reference.valid()))) {
      return Status::Internal(
          "materializing graph step has no readable input storage");
    }
    if (binding.valid())
      HYLE_RETURN_IF_ERROR(validate_known_binding_alignment(invocation, instance, binding.storage()));
    auto &submitted = invocation.submission_inputs[step.input_begin + index];
    const bool port_changed = !same_physical_tensor(submitted, binding);
    changed |= port_changed;
    if (port_changed)
      submitted = binding;
    if (native) {
      auto value = invocation.native_values[instance];
      value.description = physical_description(invocation, instance);
      value.tensor = binding;
      // This exact artifact consumes TensorViews only. Its unused lease field
      // must not search every current page guard for each native input; the
      // resource episode still admits all actual permissions at submission.
      value.lease = nullptr;
      if (!borrowed_tensor_carrier) {
        // TensorView-only artifacts never use this table. Build it once, at
        // the first actual native lease lookup, while the same preparation
        // guard still excludes mutation of the resource binding table.
        if (!lease_candidates) {
          HYLE_ASSIGN_OR_RETURN(auto candidates,
                                detail::collect_input_lease_candidates(invocation));
          lease_candidates = candidates;
        }
        value.lease = detail::input_lease(invocation, binding, *lease_candidates);
      }
      native_inputs |= !borrowed_tensor_carrier &&
          (bool(value.native_type) || value.native_handle != nullptr || value.reference.valid() ||
           (value.lease != nullptr && value.lease->valid()));
      invocation.native_inputs[index] = value;
    }
  }
  for (std::size_t index = 0; index < step.output_count; ++index) {
    const compute::TensorView &binding =
        invocation.writable[plan.node_outputs[step.output_begin + index]];
    if (!binding.valid() &&
        (plan.instance_roles[plan.node_outputs[step.output_begin + index]] &
            detail::kGeneratedOutput) == 0) {
      return Status::Internal(
          "materializing graph step has no writable output storage");
    }
    if (binding.valid())
      HYLE_RETURN_IF_ERROR(validate_known_binding_alignment(invocation,
          plan.node_outputs[step.output_begin + index], binding.storage()));
    auto &submitted = invocation.submission_outputs[step.output_begin + index];
    const bool port_changed = !same_physical_tensor(submitted, binding);
    changed |= port_changed;
    if (step.region.valid())
      native_inputs |= step.region.output_kind(index) != RegionOutputKind::kCaller;
    if (port_changed)
      submitted = binding;
  }
  changed |= native_inputs || cached.native_inputs;
  if (changed) {
    if (cached.stable_binding_identity && !native_inputs && !cached.native_inputs &&
        cached.prepared_revision == cached.revision)
      detail::recycle_step_recording(invocation, static_cast<std::uint32_t>(step_index));
    else
      detail::invalidate_step_recording(invocation, static_cast<std::uint32_t>(step_index));
  }
  cached.native_inputs = native_inputs;
  if (step.region.valid() && step.region.prepares_bindings()) {
    if (changed)
      HYLE_RETURN_IF_ERROR(invocation.invocations[step_index].region.prepare_bindings(
          std::span<const RegionValue>(invocation.native_inputs).first(step.input_count),
          std::span<const compute::TensorView>(invocation.submission_outputs)
              .subspan(step.output_begin, step.output_count)));
    for (std::size_t port = 0; port < step.output_count; ++port) {
      const auto kind = step.region.output_kind(port);
      if (kind != RegionOutputKind::kCaller)
        HYLE_RETURN_IF_ERROR(project_generated_output(invocation, step_index, port));
    }
  } else if (changed && step.kind == detail::PlanStepKind::kOperation) {
    HYLE_RETURN_IF_ERROR(compute::detail::OperationInvocationAccess::prepare_bindings(
        invocation.invocations[step_index].operation,
        std::span<const compute::ConstTensorView>(invocation.submission_inputs)
            .subspan(step.input_begin, step.input_count),
        std::span<const compute::TensorView>(invocation.submission_outputs)
            .subspan(step.output_begin, step.output_count)));
  }
  cached.prepared_revision = cached.revision;
  return Status::Ok();
}

Status
prepare_invocation_bindings(detail::InvocationState &invocation) noexcept {
  if (invocation.bindings_prepared) {
    // Native backing owners may escape without changing any address. Reuse
    // still checks the selected slot's original guard; no native rebind or
    // new reference baseline is needed for unchanged physical bindings.
    return seal_invocation_bindings(invocation);
  }
  // Any failed preparation abandons the optional physical shortcut; retrying
  // still has the original complete path and current permission checks.
  util::ScopeGuard discard_configuration_spans{[&]() noexcept {
    detail::discard_configuration_spans(invocation);
  }};
  const detail::PlanState &plan = *invocation.plan;
  for (ValueId value : plan.input_ids) {
    const std::uint32_t instance = plan.external_instances[value.ordinal()];
    if (instance == detail::kNoWorkspaceRequest ||
        invocation.bound[instance] == 0) {
      return Status::FailedPrecondition(
          "invocation input bindings are incomplete");
    }
  }
  for (ValueId value : plan.output_ids) {
    const std::size_t ordinal = value.ordinal();
    if ((plan.binding_roles[ordinal] & (detail::kInvocationOwnedOutput |
                                        detail::kSourceAliasedOutput)) == 0 &&
        (plan.external_instances[ordinal] == detail::kNoWorkspaceRequest ||
         invocation.bound[plan.external_instances[ordinal]] == 0)) {
      return Status::FailedPrecondition(
          "invocation output bindings are incomplete");
    }
  }

  std::optional<std::span<const std::uint32_t>> lease_candidates;
  util::ScopeGuard clear_lease_candidates{[&]() noexcept {
    detail::clear_input_lease_candidates(invocation);
  }};

  // A configuration-only reset may preserve physical TensorView spans, never
  // the new episode's readiness or permissions. Unproven steps still run their
  // original cold hook; only actual stable binding proofs skip those hooks.
  const bool reuse_spans = detail::reusable_configuration_spans(invocation);
  if (reuse_spans) {
    for (std::size_t index = 0; index < plan.steps.size(); ++index) {
      if (!plan.steps[index].emits_command()) continue;
      const auto &cached = invocation.step_bindings[index];
      if (!cached.stable_binding_identity || cached.prepared_revision != cached.revision)
        HYLE_RETURN_IF_ERROR(prepare_command_bindings(invocation, index, lease_candidates));
    }
    HYLE_RETURN_IF_ERROR(seal_invocation_bindings(invocation));
    invocation.bindings_prepared = true;
    return Status::Ok();
  }

  // Resolve aliases and construct stable spans before any Queue accepts work.
  // Cache them until a binding changes so steady replay does not rebuild all
  // per-step views or rescan external values.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (step.kind == detail::PlanStepKind::kView) {
      for (std::size_t output_index = 0; output_index < step.output_count;
           ++output_index) {
        const std::uint32_t output =
            plan.node_outputs[step.output_begin + output_index];
        const std::uint32_t source = plan.alias_sources[output];
        if (source == detail::kNoInPlaceSource ||
            !invocation.readable[source].valid()) {
          return Status::Internal(
              "non-materializing graph output has no readable source");
        }
        const auto &description = physical_description(invocation, output);
        const auto &previous_projection = invocation.readable[output];
        // TensorView factories clip storage to the descriptor's exact extent.
        // Reuse only that immutable metadata, never a lease or readiness fact.
        const bool same_description = previous_projection.valid() &&
            previous_projection.description() == description;
        auto bytes = same_description
            ? Result<std::size_t>{previous_projection.storage().size()}
            : description.required_bytes();
        if (!bytes)
          return std::move(bytes).error();
        HYLE_ASSIGN_OR_RETURN(const auto offset,
                              bound_alias_offset(invocation, output));
        auto storage =
            invocation.readable[source].storage().subview(offset, *bytes);
        if (!storage)
          return std::move(storage).error();
        const auto previous_storage = previous_projection.storage();
        const bool same_projection = same_description &&
            previous_storage.resource() == storage->resource() &&
            previous_storage.native_handle() == storage->native_handle() &&
            previous_storage.offset() == storage->offset() &&
            previous_storage.size() == storage->size();
        auto derived = same_projection
            ? Result<compute::ConstTensorView>{previous_projection}
            : compute::ConstTensorView::create(description, *storage);
        if (!derived)
          return std::move(derived).error();
        HYLE_RETURN_IF_ERROR(validate_known_binding_alignment(invocation, output, derived->storage()));
        const std::uint32_t logical_output = plan.instance_values[output];
        if ((plan.binding_roles[logical_output] &
             detail::kSourceAliasedOutput) != 0) {
          note_external_alias_projection(invocation, output, *derived,
                                          invocation.bound[output] != 0);
          invocation.readable[output] = *derived;
        } else if ((plan.instance_roles[output] & detail::kExternalOutput) !=
                   0) {
          if (!equal_range(byte_range(derived->storage()),
                           byte_range(invocation.writable[output].storage()))) {
            return Status::InvalidArgument(
                "published graph view does not bind its inferred storage");
          }
        } else {
          note_external_alias_projection(invocation, output, *derived,
                                          invocation.bound[output] != 0);
          invocation.readable[output] = *derived;
        }
      }
      continue;
    }
    // Ready-only regions retain their logical/control position, while all
    // physical outputs were bound from owned artifacts during compilation.
    // They have neither View aliases to resolve nor a provider invocation.
    if (!step.emits_command())
      continue;
    HYLE_RETURN_IF_ERROR(prepare_command_bindings(invocation, step_index, lease_candidates));
  }
  HYLE_RETURN_IF_ERROR(seal_invocation_bindings(invocation));
  invocation.bindings_prepared = true;
  return Status::Ok();
}

compute::Queue *candidate_queue(std::span<const QueueBinding> queues,
                                ExecutionLaneId lane) noexcept {
  for (const QueueBinding binding : queues) {
    if (binding.lane == lane)
      return binding.queue;
  }
  return nullptr;
}

bool delegated_can_import_endpoint(
    detail::InvocationState &invocation, const detail::PlanSegment &segment,
    compute::detail::CompletionEndpoint source) noexcept {
  auto selected = delegated_step_index(*invocation.plan, segment);
  if (!selected)
    return false;
  const detail::PlanStep &step = invocation.plan->steps[*selected];
  if (step.region.valid()) {
    return invocation.invocations[*selected].region.can_import_endpoint(source);
  }
  if (step.kind == detail::PlanStepKind::kOperation) {
    return compute::detail::OperationInvocationAccess::can_import_endpoint(
        invocation.invocations[*selected].operation, source);
  }
  return false;
}

Status validate_transfer_routes(detail::InvocationState &invocation,
                                std::span<const QueueBinding> queues,
                                ReplayReport &report) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  for (std::size_t index = 0; index < plan.segments.size(); ++index) {
    invocation.candidate_segment_blocks_host[index] =
        plan.segments[index].blocks_host;
  }
  for (const auto step_index : plan.copy_steps) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (step.region.valid()) {
      // Exact native resource/range validation already ran in cold binding.
      // This selected transfer has its own original delegated Completion.
      ++report.queued_transfer_count;
      continue;
    }
    compute::Queue *queue = candidate_queue(queues, step.lane);
    if (!step.lane.valid() || queue == nullptr || step.input_count != 1 ||
        step.output_count != 1) {
      return Status::Internal("compiled graph transfer route is invalid");
    }
    const std::uint32_t input = plan.node_inputs[step.input_begin];
    const std::uint32_t output = plan.node_outputs[step.output_begin];
    auto properties = compute::transfer_properties(
        *queue, invocation.writable[output].storage(),
        invocation.readable[input].storage());
    if (!properties) {
      return std::move(properties)
          .error()
          .WithContext("validating compiled graph transfer route");
    }
    if (properties->mode != compute::TransferMode::kImmediate &&
        properties->mode != compute::TransferMode::kQueuedDirect) {
      return Status::Internal(
          "compute backend returned an invalid graph transfer mode");
    }
    if (properties->mode == compute::TransferMode::kImmediate)
      ++report.immediate_transfer_count;
    else
      ++report.queued_transfer_count;
    report.peer_transfer_count += properties->peer ? 1u : 0u;
    const auto source = invocation.readable[input].storage();
    const auto destination = invocation.writable[output].storage();
    const bool no_op = destination.size() == 0 ||
        (destination.resource() == source.resource() &&
         destination.native_handle() == source.native_handle() &&
         destination.offset() == source.offset());
    if (!no_op && properties->mode == compute::TransferMode::kImmediate)
      invocation.candidate_segment_blocks_host[plan.step_segments[step_index]] = 1;
  }
  return Status::Ok();
}

Result<ReplayReport>
prepare_dependency_routes(detail::InvocationState &invocation,
                          HostWaitPolicy host_wait_policy) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  if (invocation.candidate_dependency_routes.size() !=
      plan.segment_dependencies.size()) {
    return std::unexpected(
        Status::Internal("graph dependency route storage is invalid"));
  }

  ReplayReport report{};
  for (std::size_t segment_index = 0; segment_index < plan.segments.size();
       ++segment_index) {
    const detail::PlanSegment &segment = plan.segments[segment_index];
    const bool destination_shared = shared_lane(plan, segment.lane);
    const auto &destination =
        invocation.candidate_queues[segment.lane.ordinal()];
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;
    for (std::size_t dependency_index = segment.dependency_begin;
         dependency_index < dependency_end; ++dependency_index) {
      const std::uint32_t source_segment =
          plan.segment_dependencies[dependency_index];
      if (source_segment >= segment_index ||
          (destination_shared && !destination.valid())) {
        return std::unexpected(
            Status::Internal("compiled graph dependency is invalid"));
      }
      const detail::PlanSegment &source_record = plan.segments[source_segment];
      const bool source_shared = shared_lane(plan, source_record.lane);
      const auto &source =
          invocation.candidate_queues[source_record.lane.ordinal()];
      if (source_shared && !source.valid()) {
        return std::unexpected(
            Status::Internal("compiled graph dependency source is invalid"));
      }

      ReplayDependencyRoute route = ReplayDependencyRoute::kOrderedQueue;
      const compute::detail::CompletionEndpoint source_endpoint =
          invocation.candidate_lane_endpoints[source_record.lane.ordinal()];
      const compute::detail::CompletionEndpoint destination_endpoint =
          invocation.candidate_lane_endpoints[segment.lane.ordinal()];
      if ((source_shared && destination_shared && source.same_owner(destination)) ||
          (source_endpoint.valid() &&
           source_endpoint == destination_endpoint)) {
        ++report.ordered_dependency_count;
      } else if (!source_endpoint.valid()) {
        route = ReplayDependencyRoute::kSynchronousSource;
        ++report.synchronous_source_dependency_count;
      } else {
        bool non_blocking = false;
        if (destination_shared) {
          auto accepted = compute::detail::QueueSequence::can_import_endpoint(
              destination, source_endpoint);
          if (!accepted) {
            return std::unexpected(std::move(accepted).error().WithContext(
                "selecting graph completion dependency route"));
          }
          non_blocking = *accepted;
        } else {
          non_blocking = delegated_can_import_endpoint(invocation, segment,
                                                       source_endpoint);
        }
        if (non_blocking) {
          route = ReplayDependencyRoute::kDeviceWait;
          ++report.device_wait_dependency_count;
        } else {
          if (host_wait_policy == HostWaitPolicy::kReject) {
            return std::unexpected(Status::Unimplemented(
                "graph execution topology requires a rejected host wait"));
          }
          route = ReplayDependencyRoute::kHostWait;
          ++report.host_wait_dependency_count;
        }
      }
      invocation.candidate_dependency_routes[dependency_index] = route;
    }
  }
  return report;
}

bool segment_capture_compatible(const detail::PlanState &plan,
                                const detail::PlanSegment &segment) noexcept {
  if (segment.step_count == 0)
    return false;
  const std::size_t end = segment.step_begin + segment.step_count;
  for (std::size_t step = segment.step_begin; step < end; ++step) {
    if (!plan.steps[step].capture_compatible())
      return false;
  }
  return true;
}

Status validate_external_resource(const detail::PlanState &plan,
                                  std::size_t logical_ordinal,
                                  mem::MemoryResourceRef resource) noexcept {
  const std::uint32_t storage = plan.external_storage[logical_ordinal];
  if (storage >= plan.storage_domains.size())
    return Status::Internal("graph external storage domain is invalid");
  if (plan.storage_domains[storage].resource != resource) {
    return Status::InvalidArgument(
        "tensor binding does not match its compiled memory resource");
  }
  return Status::Ok();
}

std::uint32_t alias_root(const detail::PlanState &plan,
                         std::uint32_t value) noexcept;

Status validate_configured_read_resource(const detail::PlanState &plan,
    mem::MemoryResourceRef nominated, mem::MemoryResourceRef resource) noexcept {
  for (const auto &target : plan.targets) {
    if (detail::ExecutionSourceAccess::matches_read_resource(
            target.source, nominated, resource)) {
      HYLE_RETURN_IF_ERROR(target.source.validate_resource(resource));
      return Status::Ok();
    }
  }
  return Status::InvalidArgument(
      "readonly memory resource has no configured source authority");
}

Status validate_external_read_resource(const detail::PlanState &plan,
    std::size_t logical_ordinal, mem::MemoryResourceRef resource) noexcept {
  const auto storage = plan.external_storage[logical_ordinal];
  if (storage >= plan.storage_domains.size())
    return Status::Internal("graph external storage domain is invalid");
  const auto nominated = plan.storage_domains[storage].resource;
  if (nominated == resource)
    return Status::Ok();
  const auto instance = plan.external_instances[logical_ordinal];
  if (instance == detail::kNoWorkspaceRequest)
    return Status::Internal("readonly graph source has no external physical instance");
  bool consumed = false;
  for (const auto &step : plan.steps) {
    if (!step.emits_command())
      continue;
    bool reads_input = false;
    for (std::size_t port = 0; port < step.input_count; ++port) {
      if (alias_root(plan, plan.node_inputs[step.input_begin + port]) == instance) {
        reads_input = true;
        break;
      }
    }
    if (!reads_input)
      continue;
    consumed = true;
    const auto &source = plan.targets[step.target.ordinal()].source;
    if (step.kind != detail::PlanStepKind::kRegion ||
        !detail::ExecutionSourceAccess::matches_read_resource(source, nominated, resource))
      return Status::InvalidArgument(
          "readonly memory resource is not accepted by its compiled readers");
    HYLE_RETURN_IF_ERROR(source.validate_resource(resource));
  }
  if (consumed)
    return Status::Ok();
  // A published source or metadata-only View has no device reader to query.
  // It still needs an actual source's explicit readonly-family authority.
  return validate_configured_read_resource(plan, nominated, resource);
}

Status initialize_plan_metadata(
    detail::PlanState &plan, const detail::GraphState &graph,
    std::span<const compute::TensorDesc> descriptions) noexcept {
  if (!plan.steps.try_reserve(graph.nodes.size()) ||
      !plan.input_ids.try_reserve(graph.inputs.size()) ||
      !plan.constant_ids.try_reserve(graph.constants.size()) ||
      !plan.constant_bindings.try_resize(graph.constants.size()) ||
      !plan.output_ids.try_reserve(graph.outputs.size()) ||
      !plan.node_inputs.try_reserve(graph.edges.size()) ||
      !plan.node_outputs.try_reserve(graph.values.size()) ||
      !plan.descriptions.try_resize(graph.values.size()) ||
      !plan.instance_descriptions.try_resize(graph.values.size()) ||
      !plan.instance_values.try_resize(graph.values.size()) ||
      !plan.instance_storage.try_resize(graph.values.size()) ||
      !plan.instance_output_contracts.try_resize(graph.values.size()) ||
      !plan.instance_producer_targets.try_resize(graph.values.size()) ||
      !plan.instance_constant_owners.try_resize(graph.values.size()) ||
      !plan.instance_roles.try_resize(graph.values.size()) ||
      !plan.external_instances.try_resize(graph.values.size()) ||
      !plan.external_storage.try_resize(graph.values.size()) ||
      !plan.workspace_requests.try_resize(graph.values.size()) ||
      !plan.node_steps.try_resize(graph.nodes.size()) ||
      !plan.value_materialized.try_resize(graph.values.size()) ||
      !plan.alias_sources.try_resize(graph.values.size()) ||
      !plan.alias_offsets.try_resize(graph.values.size()) ||
      !plan.binding_roles.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }

  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    plan.descriptions[index] = descriptions[index];
    plan.instance_descriptions[index] = descriptions[index];
    plan.instance_values[index] = static_cast<std::uint32_t>(index);
    plan.instance_storage[index] = 0;
    plan.instance_output_contracts[index] = {};
    plan.instance_producer_targets[index] = detail::kNoWorkspaceRequest;
    plan.instance_constant_owners[index] = detail::kNoWorkspaceRequest;
    plan.instance_roles[index] = 0;
    plan.external_instances[index] = detail::kNoWorkspaceRequest;
    plan.external_storage[index] = 0;
    plan.workspace_requests[index] = detail::kNoWorkspaceRequest;
    plan.value_materialized[index] = 0;
    plan.alias_sources[index] = detail::kNoInPlaceSource;
    plan.alias_offsets[index] = 0;
    plan.binding_roles[index] = 0;
  }
  for (std::size_t index = 0; index < plan.node_steps.size(); ++index)
    plan.node_steps[index] = detail::kNoNode;
  plan.logical_node_count = graph.nodes.size();
  plan.input_ids.clear();
  plan.constant_ids.clear();
  plan.output_ids.clear();
  for (std::uint32_t ordinal : graph.inputs) {
    if (!plan.input_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kExternalInput;
  }
  for (std::uint32_t ordinal : graph.constants) {
    if (!plan.constant_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kConstant;
  }
  for (std::uint32_t ordinal : graph.outputs) {
    if (!plan.output_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kExternalOutput;
  }
  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    plan.instance_roles[index] = plan.binding_roles[index];
    if ((plan.binding_roles[index] &
         (detail::kExternalInput | detail::kExternalOutput |
          detail::kConstant)) != 0) {
      plan.external_instances[index] = static_cast<std::uint32_t>(index);
    }
  }
  return Status::Ok();
}

Status initialize_plan_lanes(detail::PlanState &plan,
                             CompileEnvironment environment) noexcept {
  plan.lanes.clear();
  if (environment.lanes.empty()) {
    for (std::size_t target = 0; target < environment.targets.size();
         ++target) {
      if (!plan.lanes.try_push_back(detail::ExecutionLaneRecord{
              .target = ExecutionTargetId{static_cast<std::uint32_t>(target)},
              .role = ExecutionLaneRole::kCompute}))
        return plan_allocation_failure();
    }
  } else {
    for (const auto &lane : environment.lanes) {
      if (!plan.lanes.try_push_back(detail::ExecutionLaneRecord{
              .target = lane.target, .role = lane.role}))
        return plan_allocation_failure();
    }
  }
  return Status::Ok();
}

// Immutable original sources, borrowed only during this compilation. Each
// candidate publishes its own ready destination views into PlanState.
struct ConstantPreparation final {
  explicit ConstantPreparation(mem::AllocatorRef allocator) noexcept
      : bindings(allocator) {}
  container::SmallVector<ConstantBinding, 0, mem::AllocatorRef> bindings;
};

std::size_t constant_slot(const detail::PlanState &plan,
                           std::uint32_t value) noexcept {
  const auto found = std::lower_bound(plan.constant_ids.begin(), plan.constant_ids.end(), value,
      [](ValueId constant, std::uint32_t ordinal) { return constant.ordinal() < ordinal; });
  return found != plan.constant_ids.end() && found->ordinal() == value
      ? static_cast<std::size_t>(found - plan.constant_ids.begin()) : SIZE_MAX;
}

Status propagate_alias_alignments(detail::PlanState &plan,
                                  bool check_offsets = true) noexcept {
  if (std::ranges::all_of(plan.alias_sources,
          [](auto source) { return source == detail::kNoInPlaceSource; }))
    return Status::Ok();
  detail::ValueOrdinals roots(plan.allocator);
  container::SmallVector<std::size_t, 0, mem::AllocatorRef> offsets(plan.allocator);
  if (!roots.try_resize(plan.instance_values.size()) ||
      !offsets.try_resize(plan.instance_values.size()))
    return plan_allocation_failure();
  for (std::size_t ordinal = 0; ordinal < plan.instance_values.size(); ++ordinal) {
    const auto source = plan.alias_sources[ordinal];
    if (source == detail::kNoInPlaceSource) {
      roots[ordinal] = static_cast<std::uint32_t>(ordinal);
      offsets[ordinal] = 0;
      continue;
    }
    if (source >= ordinal || plan.alias_offsets[ordinal] > SIZE_MAX - offsets[source])
      return Status::Internal("graph physical alias chain is invalid");
    roots[ordinal] = roots[source];
    offsets[ordinal] = offsets[source] + plan.alias_offsets[ordinal];
    const auto alignment = plan.instance_alignments[ordinal];
    if (check_offsets && offsets[ordinal] % alignment != 0)
      return Status::InvalidArgument(
          "graph alias offset cannot satisfy provider port alignment");
    // Only the allocation root inherits this requirement. An intermediate
    // View may have a different legal alignment than a later slice's port.
    auto &root_alignment = plan.instance_alignments[roots[ordinal]];
    root_alignment = (std::max)(root_alignment, alignment);
  }
  return Status::Ok();
}

Result<bool> accepts_input_alias_alignment(const detail::PlanState &plan,
    std::uint32_t instance, std::size_t alignment) noexcept {
  // Domain View offsets belong to the concrete Invocation, not its envelope.
  if (plan.domain_graph.valid() || alignment <= 1)
    return true;
  auto root = instance;
  std::size_t offset = 0;
  for (;;) {
    if (root >= plan.alias_sources.size())
      return std::unexpected(Status::Internal("graph input alias is invalid"));
    const auto parent = plan.alias_sources[root];
    if (parent == detail::kNoInPlaceSource)
      break;
    if (parent >= root || plan.alias_offsets[root] > SIZE_MAX - offset)
      return std::unexpected(Status::Internal("graph input alias chain is invalid"));
    offset += plan.alias_offsets[root];
    root = parent;
  }
  if (offset % alignment == 0)
    return true;
  // Only Graph-owned caller-buffer roots have the zero-aligned allocation
  // origin enforced below. External/constant bindings and generated carriers
  // retain their own address contracts; an unknown base is not a bad offset.
  if ((plan.instance_roles[root] & (detail::kExternalInput |
          detail::kExternalOutput | detail::kConstant)) != 0 ||
      plan.instance_output_contracts[root].kind != RegionOutputKind::kCaller)
    return true;
  return false;
}

Status prepare_constant_alignments(detail::PlanState &plan) noexcept {
  if (plan.constant_ids.empty())
    return Status::Ok();
  // Metadata Views already have an immutable physical source. Their reader
  // requirements must reach a constant before its bytes are allocated.
  for (const auto &step : plan.steps) {
    if (step.kind != detail::PlanStepKind::kView)
      continue;
    if (step.input_count != 1 || step.output_count != 1)
      return Status::Internal("metadata View has invalid physical ports");
    const auto output = plan.node_outputs[step.output_begin];
    plan.alias_sources[output] = plan.node_inputs[step.input_begin];
    plan.alias_offsets[output] = step.view_offset;
  }
  // A domain's offset is resolved from its concrete Invocation. Propagating a
  // necessary base alignment here does not prove a symbolic offset is aligned.
  return propagate_alias_alignments(plan, !plan.domain_graph.valid());
}

Status bind_constants(detail::PlanState &plan, const detail::GraphState &graph,
                      CompileOptions options,
                      ConstantPreparation &constants) noexcept {
  if (options.constants.size() != graph.constants.size()) {
    return Status::InvalidArgument(
        "compile requires exactly one binding for every graph constant");
  }
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Flags = container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;
  Ordinals slots(plan.allocator);
  Flags seen(plan.allocator);
  if (!slots.try_resize(graph.values.size()) ||
      !seen.try_resize(graph.constants.size())) {
    return plan_allocation_failure();
  }
  if (!constants.bindings.try_resize(graph.constants.size()) ||
      !plan.constant_owners.try_resize(graph.constants.size()) ||
      !plan.constant_leases.try_resize(graph.constants.size()))
    return plan_allocation_failure();
  for (std::size_t index = 0; index < slots.size(); ++index)
    slots[index] = detail::kNoWorkspaceRequest;
  for (std::size_t index = 0; index < graph.constants.size(); ++index) {
    slots[graph.constants[index]] = static_cast<std::uint32_t>(index);
    seen[index] = 0;
  }

  for (const ConstantBinding &binding : options.constants) {
    if (binding.policy != ConstantPolicy::kBorrow &&
        binding.policy != ConstantPolicy::kMaterialize)
      return Status::InvalidArgument("constant source policy is invalid");
    if (!detail::owns(graph, binding.value) ||
        binding.value.kind() != ValueKind::kConstant) {
      return Status::InvalidArgument(
          "constant binding does not belong to this graph");
    }
    const std::uint32_t slot = slots[binding.value.ordinal()];
    if (slot == detail::kNoWorkspaceRequest || seen[slot] != 0)
      return Status::InvalidArgument("constant binding is duplicated");
    if (!binding.tensor.valid() ||
        binding.tensor.description() !=
            plan.descriptions[binding.value.ordinal()]) {
      return Status::InvalidArgument(
          "constant binding does not match its graph description");
    }
    plan.constant_bindings[slot] = binding.tensor;
    if (binding.owner && !binding.lease)
      return Status::InvalidArgument("constant storage owner requires a ready lease");
    if (binding.lease) {
      HYLE_ASSIGN_OR_RETURN(auto retained, binding.lease.view());
      const auto backing = compute::detail::LeaseAccess::metadata_backing_allocator(binding.lease);
      const auto system = mem::system_allocator();
      const bool process_allocator = backing.valid() &&
          backing.alloc_fn() == system.alloc_fn() &&
          backing.free_fn() == system.free_fn() &&
          backing.realloc_fn() == system.realloc_fn() &&
          backing.state() == system.state();
      // A CPU ready lease already owns its Buffer. Its original resource and
      // metadata allocator need no separate owner only when both are the
      // exact process-lifetime system objects, not a host-accessible wrapper.
      if (!binding.owner &&
          (retained.resource() != mem::system_resource() || !process_allocator))
        return Status::InvalidArgument(
            "ownerless constant lease requires the process system resource and allocator");
      if (!binding.owner.owns_allocator(backing) && !process_allocator)
        return Status::InvalidArgument(
            "constant storage owner does not retain its lease metadata allocator");
      const auto actual = binding.tensor.storage();
      if ((binding.owner && binding.owner.ref() != retained.resource()) ||
          retained.resource() != actual.resource() ||
          retained.native_handle() != actual.native_handle() ||
          actual.offset() < retained.offset() ||
          actual.offset() - retained.offset() > retained.size() ||
          actual.size() > retained.size() - (actual.offset() - retained.offset()))
        return Status::InvalidArgument("constant view is outside its owned ready lease");
      if (binding.policy == ConstantPolicy::kBorrow) {
        plan.constant_owners[slot] = binding.owner;
        plan.constant_leases[slot] = binding.lease;
      }
    }
    constants.bindings[slot] = binding;
    seen[slot] = 1;
  }
  for (std::uint8_t present : seen) {
    if (present == 0)
      return Status::InvalidArgument("graph constant binding is missing");
  }
  return Status::Ok();
}

bool needs_canonical_constant(const detail::PlanState &plan,
                              const ConstantPreparation &constants,
                              std::size_t instance) noexcept {
  if ((plan.instance_roles[instance] & detail::kConstant) == 0 ||
      plan.alias_sources[instance] != detail::kNoInPlaceSource)
    return false;
  const auto slot = constant_slot(plan, plan.instance_values[instance]);
  return slot != SIZE_MAX &&
      (plan.external_instances[plan.instance_values[instance]] != instance ||
       constants.bindings[slot].policy == ConstantPolicy::kMaterialize);
}

// A retry numbers physical instances anew. Retain ready owners by their real
// immutable allocation keys, never by the previous instance ordinal. A sorted
// cold index keeps this O(C log C + I log C), without a second payload cache.
Status retain_constant_owners(detail::PlanState &plan,
                               const ConstantPreparation &constants) noexcept {
  if (plan.canonical_constants.empty())
    return Status::Ok();
  detail::ValueOrdinals order(plan.allocator), remap(plan.allocator);
  if (!order.try_reserve(plan.canonical_constants.size()) ||
      !remap.try_resize(plan.canonical_constants.size()))
    return plan_allocation_failure();
  std::fill(remap.begin(), remap.end(), detail::kNoWorkspaceRequest);
  for (std::size_t index = 0; index < plan.canonical_constants.size(); ++index)
    if (plan.canonical_constants[index] && plan.canonical_constants[index]->ready &&
        !order.try_push_back(static_cast<std::uint32_t>(index)))
      return plan_allocation_failure();
  const auto less = [](const detail::CanonicalConstantKey &left,
                       const detail::CanonicalConstantKey &right) noexcept {
    if (left.constant != right.constant)
      return left.constant < right.constant;
    const auto *l = &left.resource.resource(), *r = &right.resource.resource();
    if (l != r)
      return std::less<const mem::MemoryResource *>{}(l, r);
    return left.alignment < right.alignment;
  };
  std::sort(order.begin(), order.end(), [&](auto left, auto right) noexcept {
    return less(plan.canonical_constants[left]->key, plan.canonical_constants[right]->key);
  });
  for (std::size_t instance = 0; instance < plan.instance_values.size(); ++instance) {
    if (!needs_canonical_constant(plan, constants, instance))
      continue;
    const auto logical = plan.instance_values[instance];
    const auto resource = plan.storage_domains[plan.instance_storage[instance]].resource;
    const auto &description = plan.instance_descriptions[instance];
    const detail::CanonicalConstantKey key{logical, description, resource,
        (std::max)({description.required_alignment(), plan.instance_alignments[instance],
                    resource.properties().minimum_alignment})};
    const auto found = std::lower_bound(order.begin(), order.end(), key,
        [&](auto index, const auto &wanted) noexcept {
          return less(plan.canonical_constants[index]->key, wanted);
        });
    if (found == order.end())
      continue;
    const auto &owner = plan.canonical_constants[*found];
    if (owner->key.constant != logical || owner->key.description != description ||
        owner->buffer.resource() != resource || owner->buffer.alignment() < key.alignment)
      continue;
    plan.instance_constant_owners[instance] = *found;
    remap[*found] = 0; // Mark before compacting the unique owner arena.
  }
  std::uint32_t retained = 0;
  for (std::size_t index = 0; index < plan.canonical_constants.size(); ++index) {
    if (remap[index] == detail::kNoWorkspaceRequest)
      continue;
    remap[index] = retained;
    if (index != retained)
      plan.canonical_constants[retained] = std::move(plan.canonical_constants[index]);
    ++retained;
  }
  plan.canonical_constants.try_resize(retained);
  for (auto &index : plan.instance_constant_owners)
    if (index != detail::kNoWorkspaceRequest)
      index = remap[index];
  return Status::Ok();
}

// Exact queued-copy roots, independently retained by one original target
// prefix. No Plan/Completion cycle or unrelated model constants are retained.
struct ConstantCopyBatch final : util::RefCounted<ConstantCopyBatch> {
  struct Root final {
    compute::StorageOwner source_owner;
    compute::ReadLease source;
    util::Ref<detail::CanonicalConstantState> destination;
  };
  ConstantCopyBatch(mem::AllocatorRef selected, compute::StorageOwner metadata,
                    compute::StorageOwner target, ExecutionSource endpoint) noexcept
      : allocator(selected), metadata_owner(std::move(metadata)),
        target_owner(std::move(target)), source(std::move(endpoint)),
        roots(selected) {}
  void destroy() noexcept {
    const auto selected = allocator;
    const auto retained = metadata_owner;
    this->~ConstantCopyBatch();
    selected.deallocate(this, sizeof(ConstantCopyBatch), alignof(ConstantCopyBatch));
  }
  void protect(compute::Completion &completion) noexcept {
    incref();
    retention.own = {this, +[](void *, const Status &) noexcept {},
        +[](void *value) noexcept { static_cast<ConstantCopyBatch *>(value)->decref(); }};
    const auto attached = compute::detail::CompletionAccess::attach_retention(completion, retention);
    HYLE_FATAL_ASSERT(attached.ok());
  }
  mem::AllocatorRef allocator;
  compute::StorageOwner metadata_owner;
  // The source endpoint may use this allocator; release it before the target
  // domain/Context allocator, even when the outer compile has already failed.
  compute::StorageOwner target_owner;
  ExecutionSource source;
  container::SmallVector<Root, 0, mem::AllocatorRef> roots;
  compute::detail::CompletionRetention retention{};
};

Status materialize_constants(detail::PlanState &plan,
                             CompileEnvironment environment,
                             ConstantPreparation &constants,
                             detail::ConstantArtifactCache &packing,
                             bool &route_declined) noexcept {
  route_declined = false;
  if (constants.bindings.empty())
    return Status::Ok();
  std::size_t canonical_count = 0;
  for (std::size_t instance = 0; instance < plan.instance_values.size(); ++instance)
    canonical_count += needs_canonical_constant(plan, constants, instance) ? 1U : 0U;
  if (canonical_count == 0)
    return Status::Ok();
  container::SmallVector<compute::Queue, 2, mem::AllocatorRef> queues(
      plan.allocator);
  struct PendingCopy final {
    detail::ConstantCopy transfer;
    std::uint32_t target;
    bool native;
    compute::TransferMode mode = compute::TransferMode::kImmediate;
    std::size_t slot = 0;
    util::Ref<detail::CanonicalConstantState> canonical{};
  };
  container::SmallVector<PendingCopy, 0, mem::AllocatorRef> pending(plan.allocator);
  auto select_copy = [&](detail::ConstantCopy transfer, std::uint32_t storage)
      noexcept -> Result<PendingCopy> {
    if (!queues.try_resize(environment.targets.size()))
      return std::unexpected(plan_allocation_failure());
    // Prefer the target owning the destination domain, then other configured
    // targets. Querying unsupported routes is cold; an actual copy error is
    // never permission to resubmit potentially accepted bytes elsewhere.
    for (const bool preferred : {true, false}) {
      for (std::size_t target = 0; target < environment.targets.size();
           ++target) {
        if ((environment.targets[target].default_storage.ordinal() ==
             storage) != preferred)
          continue;
        const auto &source = plan.targets[target].source;
        auto retained = detail::ExecutionSourceAccess::compute(source);
        if (!retained) {
          if (!retained.error().IsUnimplemented())
            return std::unexpected(std::move(retained).error());
          HYLE_ASSIGN_OR_RETURN(const bool supported,
              detail::ExecutionSourceAccess::can_copy_constant(source, transfer));
          if (supported)
            return PendingCopy{transfer, static_cast<std::uint32_t>(target), true};
          continue;
        }
        bool compatible = true;
        for (const auto resource :
             {transfer.destination.storage().resource(), transfer.source.storage().resource()}) {
          auto status = source.validate_resource(resource);
          if (status.ok())
            continue;
          if (!status.IsFailedPrecondition() && !status.IsUnimplemented())
            return std::unexpected(std::move(status));
          compatible = false;
          break;
        }
        if (!compatible)
          continue;
        if (!queues[target]) {
          auto queue = (*retained)->create_queue();
          if (!queue)
            return std::unexpected(std::move(queue).error());
          queues[target] = std::move(*queue);
        }
        auto properties = compute::transfer_properties(queues[target],
            transfer.destination.storage(), transfer.source.storage());
        if (!properties) {
          auto status = std::move(properties).error();
          if (status.IsUnimplemented())
            continue;
          return std::unexpected(std::move(status));
        }
        return PendingCopy{transfer, static_cast<std::uint32_t>(target), false,
                           properties->mode};
      }
    }
    route_declined = true; // Pure qualification, before any transfer accepts work.
    return std::unexpected(Status::Unimplemented(
        "no configured target can materialize the graph constant"));
  };
  if (!pending.try_reserve(canonical_count) ||
      !plan.prepared_values.try_reserve(canonical_count))
    return plan_allocation_failure();
  for (std::size_t instance = 0; instance < plan.instance_values.size(); ++instance) {
    if (!needs_canonical_constant(plan, constants, instance))
      continue;
    const auto slot = constant_slot(plan, plan.instance_values[instance]);
    const auto &binding = constants.bindings[slot];
    const auto storage = plan.instance_storage[instance];
    if (storage >= plan.storage_domains.size())
      return Status::Internal("constant has no selected storage domain");
    const auto &description = binding.tensor.description();
    if (instance >= plan.instance_alignments.size())
      return Status::Internal("constant has no physical alignment contract");
    HYLE_ASSIGN_OR_RETURN(auto acquired, packing.acquire_canonical(
        plan, slot, static_cast<std::uint32_t>(instance), plan.storage_domains[storage].resource,
        (std::max)(description.required_alignment(), plan.instance_alignments[instance])));
    HYLE_ASSIGN_OR_RETURN(auto destination, compute::TensorView::create(
        description, acquired.owner->buffer.view()));
    if (acquired.created && !destination.storage().empty()) {
      HYLE_ASSIGN_OR_RETURN(auto selected,
          select_copy({destination, binding.tensor}, storage));
      selected.slot = slot;
      selected.canonical = acquired.owner;
      if (selected.mode == compute::TransferMode::kQueuedDirect) {
        const auto endpoint = detail::ExecutionSourceAccess::compute(
            plan.targets[selected.target].source);
        // bind_constants already proved a system lease's complete process
        // resource/allocator identity and its exact view. Keep that real lease
        // in ConstantCopyBatch through native retirement just like owned bytes.
        const bool retained_source = binding.lease &&
            (binding.owner || binding.tensor.storage().resource() == mem::system_resource());
        if (!retained_source || !acquired.owner->storage_owner ||
            !plan.metadata_owner || !endpoint || !*endpoint ||
            acquired.owner->storage_owner.context_identity() != (*endpoint)->identity_key())
          return Status::FailedPrecondition(
              "queued constant materialization requires a ready lease and storage owner for source, destination and metadata");
      }
      if (!pending.try_push_back(selected))
        return plan_allocation_failure();
    }
  }
  // Graph-owned allocations, charges and route queries precede accepted bytes.
  // A native endpoint sees one batch so it can reuse its real staging and
  // command resources across weights. These resources never enter replay.
  container::SmallVector<detail::ConstantCopy, 0, mem::AllocatorRef> batch(plan.allocator);
  const auto native_count = static_cast<std::size_t>(std::count_if(
      pending.begin(), pending.end(), [](const PendingCopy &copy) { return copy.native; }));
  if (!batch.try_reserve(native_count))
    return plan_allocation_failure();
  // All ownership/metadata admission precedes the first copy on any target.
  // Reserve one owner per queued target, not one allocation/event per weight.
  container::SmallVector<util::Ref<ConstantCopyBatch>, 2, mem::AllocatorRef>
      retained(plan.allocator);
  if (!retained.try_resize(environment.targets.size()))
    return plan_allocation_failure();
  for (std::size_t target = 0; target < environment.targets.size(); ++target) {
    const auto count = static_cast<std::size_t>(std::count_if(pending.begin(), pending.end(),
        [&](const auto &copy) { return copy.target == target &&
            copy.mode == compute::TransferMode::kQueuedDirect; }));
    if (count == 0) continue;
    const auto first = std::ranges::find_if(pending, [&](const auto &copy) {
      return copy.target == target && copy.mode == compute::TransferMode::kQueuedDirect;
    });
    auto *memory = plan.allocator.allocate(sizeof(ConstantCopyBatch), alignof(ConstantCopyBatch));
    if (!memory) return plan_allocation_failure();
    retained[target] = util::adopt_ref(::new (memory) ConstantCopyBatch(
        plan.allocator, plan.metadata_owner, first->canonical->storage_owner,
        plan.targets[target].source));
    if (!retained[target]->roots.try_reserve(count))
      return plan_allocation_failure();
    for (const auto &copy : pending) {
      if (copy.target != target || copy.mode != compute::TransferMode::kQueuedDirect) continue;
      const auto &binding = constants.bindings[copy.slot];
      if (!retained[target]->roots.try_push_back({binding.owner, binding.lease, copy.canonical}))
        return plan_allocation_failure();
    }
  }
  for (std::size_t target = 0; target < environment.targets.size(); ++target) {
    batch.clear();
    for (const auto &copy : pending)
      if (copy.target == target && copy.native && !batch.try_push_back(copy.transfer))
        return plan_allocation_failure();
    if (!batch.empty()) {
      detail::StorageCharge scratch;
      if (plan.compilation_storage) {
        HYLE_ASSIGN_OR_RETURN(const auto bytes,
            detail::ExecutionSourceAccess::constant_copy_scratch_bytes(
                plan.targets[target].source, batch));
        HYLE_ASSIGN_OR_RETURN(scratch, plan.compilation_storage->reserve_private(bytes));
      }
      HYLE_RETURN_IF_ERROR(detail::ExecutionSourceAccess::copy_constants(
          plan.targets[target].source, batch, plan.allocator));
      continue;
    }
    if (std::ranges::none_of(pending, [&](const auto &copy) {
          return copy.target == target && !copy.native;
        }))
      continue;
    auto sequence = compute::detail::QueueSequence::begin(queues[target]);
    if (!sequence) return std::move(sequence).error();
    for (const auto &copy : pending) {
      if (copy.target != target)
        continue;
      sequence->record(compute::detail::TransferAccess::enqueue_copy(*sequence,
          copy.transfer.destination.storage(), copy.transfer.source.storage()));
      if (!sequence->terminal().ok()) break;
    }
    auto completion = sequence->finish();
    if (!completion) return std::move(completion).error();
    if (retained[target]) retained[target]->protect(*completion);
    // A failed observation hands the owner to the existing backend retirement
    // mechanism. Neither failure nor abandonment publishes canonical bytes.
    HYLE_RETURN_IF_ERROR(completion->wait());
    retained[target] = {};
  }
  // Only successful original terminal copies make new cache entries reusable.
  // All sources remain the caller's original immutable bindings through here.
  for (const auto &owner : plan.canonical_constants)
    owner->ready = true;
  for (std::size_t instance = 0; instance < plan.instance_values.size(); ++instance) {
    const auto *owner = plan.owned_instance_constant(instance);
    if (owner == nullptr)
      continue;
    HYLE_ASSIGN_OR_RETURN(auto ready, compute::ConstTensorView::create(
        plan.instance_descriptions[instance], owner->view()));
    const auto logical = plan.instance_values[instance];
    if (plan.external_instances[logical] == instance) {
      plan.constant_bindings[constant_slot(plan, logical)] = ready;
    } else if (!plan.prepared_values.try_push_back(
                   {static_cast<std::uint32_t>(instance), ready})) {
      return plan_allocation_failure();
    }
  }
  return Status::Ok();
}

bool valid_strength(PlacementStrength strength) noexcept {
  return strength == PlacementStrength::kPreferred ||
         strength == PlacementStrength::kRequired;
}

// Cold-only policy index. No storage is added to Plan or Invocation. Sorting
// the small family lists avoids rescanning them per node/candidate; an opted-in
// node table gives constant-time matching and duplicate detection.
class ProviderPolicy final {
public:
  explicit ProviderPolicy(mem::AllocatorRef allocator) noexcept
      : allowed_(allocator), denied_(allocator), nodes_(allocator) {}

  Status initialize(const detail::GraphState &graph,
                    CompileOptions options) noexcept {
    HYLE_RETURN_IF_ERROR(copy_families(options.allowed_providers, allowed_));
    HYLE_RETURN_IF_ERROR(copy_families(options.denied_providers, denied_));
    if (options.node_providers.empty())
      return Status::Ok();
    if (!nodes_.try_resize(graph.nodes.size()))
      return plan_allocation_failure();
    std::fill(nodes_.begin(), nodes_.end(), nullptr);
    for (const NodeProviderConstraint &constraint : options.node_providers) {
      if (!detail::owns(graph, constraint.node) || !constraint.provider.valid())
        return Status::InvalidArgument(
            "graph node provider constraint is invalid");
      auto &slot = nodes_[constraint.node.ordinal()];
      if (slot != nullptr)
        return Status::InvalidArgument(
            "graph node provider constraint is duplicated");
      slot = &constraint;
    }
    return Status::Ok();
  }

  Result<bool> allows_family(const compute::detail::ContextRef &context,
                             std::size_t index) const noexcept {
    if (allowed_.empty() && denied_.empty())
      return true;
    auto identity = context.provider_identity(index);
    if (!identity)
      return std::unexpected(std::move(identity).error());
    return allows_family(identity->id);
  }

  Result<bool> allows_operation(const compute::detail::ContextRef &context,
                                std::size_t index,
                                std::size_t node) const noexcept {
    const auto *constraint = nodes_.empty() ? nullptr : nodes_[node];
    if (constraint == nullptr)
      return allows_family(context, index);
    auto identity = context.provider_identity(index);
    if (!identity)
      return std::unexpected(std::move(identity).error());
    if (identity->id != constraint->provider || !allows_family(identity->id))
      return false;
    if (!constraint->instance.valid())
      return true;
    auto instance = context.provider_instance_id(index);
    if (!instance)
      return std::unexpected(std::move(instance).error());
    return *instance == constraint->instance;
  }

  bool allows_node(std::size_t node, compute::ProviderId provider,
                   compute::ProviderInstanceId instance) const noexcept {
    const auto *constraint = nodes_.empty() ? nullptr : nodes_[node];
    return constraint == nullptr || (constraint->provider == provider &&
                                     (!constraint->instance.valid() ||
                                      constraint->instance == instance));
  }

  bool allows_native() const noexcept { return allowed_.empty(); }

private:
  using Families =
      container::SmallVector<compute::ProviderId, 4, mem::AllocatorRef>;
  static bool less(compute::ProviderId left,
                   compute::ProviderId right) noexcept {
    return left.namespace_id != right.namespace_id
               ? left.namespace_id < right.namespace_id
               : left.provider_id < right.provider_id;
  }
  static Status copy_families(std::span<const compute::ProviderId> source,
                              Families &destination) noexcept {
    for (const auto id : source) {
      if (!id.valid())
        return Status::InvalidArgument(
            "graph provider filter contains an invalid family ID");
    }
    if (!destination.try_resize(source.size()))
      return plan_allocation_failure();
    std::copy(source.begin(), source.end(), destination.begin());
    std::sort(destination.begin(), destination.end(), less);
    return Status::Ok();
  }
  bool allows_family(compute::ProviderId provider) const noexcept {
    return (allowed_.empty() ||
            std::binary_search(allowed_.begin(), allowed_.end(), provider,
                               less)) &&
           !std::binary_search(denied_.begin(), denied_.end(), provider, less);
  }

  Families allowed_;
  Families denied_;
  container::SmallVector<const NodeProviderConstraint *, 16, mem::AllocatorRef>
      nodes_;
};

struct RegionExtension final {
  detail::GraphRegionSource source;
  std::uint32_t provider_index = UINT32_MAX;
  ExecutionTargetId target{};
  const detail::RegionAdapter *adapter = nullptr;

  Status call(compute::detail::ProviderExtensionCallFn function,
              void *context) const noexcept {
    struct Call final {
      const detail::RegionAdapter *adapter;
      compute::detail::ProviderExtensionCallFn function;
      void *context;
    } call{adapter, function, context};
    return source.call(
        +[](void *native_context, const void *native_interface,
            void *opaque) noexcept -> Status {
          const auto &call = *static_cast<Call *>(opaque);
          return call.function(
              call.adapter ? call.adapter->context : native_context,
              call.adapter ? call.adapter->interface : native_interface,
              call.context);
        }, &call);
  }
};

struct RegionCandidate final {
  compute::ProviderRouteCostFacts cost{};
  std::uint64_t route = 0;
  compute::ProviderReplayClass replay_class =
      compute::ProviderReplayClass::kUnknown;
  std::uint32_t node_offset = 0;
  std::uint32_t node_count = 0;
  std::uint32_t extension_index = 0;
  std::uint32_t provider_index = 0;
  std::uint32_t proposal_order = 0;
  ExecutionTargetId target{};
  compute::ProviderId provider_family{};
  compute::ProviderImplementationId implementation{};
  compute::RoutePolicyId policy{};
  std::uint32_t preference = 0;
};

struct RouteDiagnostic final {
  compute::ProviderImplementationId implementation{};
  compute::RoutePolicyId policy{};
  std::uint32_t preference = 0;
  std::optional<std::uint32_t> dispatch_count{};
};

constexpr bool replay_allowed(ProviderReplayPolicy policy,
                              compute::ProviderReplayClass replay) noexcept {
  return policy == ProviderReplayPolicy::kAllowAny ||
         replay == compute::ProviderReplayClass::kPrepared;
}

using RegionExtensions =
    container::SmallVector<RegionExtension, 4, mem::AllocatorRef>;
using RegionCandidates =
    container::SmallVector<RegionCandidate, 16, mem::AllocatorRef>;
using RegionNodeIds = container::SmallVector<NodeId, 32, mem::AllocatorRef>;
using RegionValues = container::SmallVector<ValueId, 8, mem::AllocatorRef>;

struct RegionResourceSkeleton final {
  RegionResourceProposal proposal{};
  std::size_t port_begin = 0;
  std::size_t input_reference_begin = 0;
  std::size_t output_contract_begin = 0;
  std::size_t input_count = 0;
  std::size_t output_count = 0;
  std::size_t effect_begin = 0;
  std::size_t effect_count = 0;
  bool has_unknown_effects = false;
};

struct OperationCandidate final {
  compute::detail::ProviderOperationProposal proposal{};
  std::size_t requirements_begin = 0;
  bool rejected = false;
  // A node-cover alternative, not a synthetic Provider. Its facts are exact
  // zero and it never enters propose/prepare or a Provider budget.
  bool metadata_view = false;
};

struct OperationCandidateRange final {
  std::size_t begin = 0;
  std::size_t count = 0;
};

// Typed leaf proposals share the compile-local discovery lifetime with region
// proposals. These contain no prepared artifacts, only validated cold facts.
struct OperationDiscovery final {
  explicit OperationDiscovery(mem::AllocatorRef allocator) noexcept
      : candidates(allocator), requirements(allocator), ranges(allocator) {}

  std::span<const OperationCandidate>
  for_node(std::size_t node, std::size_t target,
           std::size_t target_count) const noexcept {
    if (ranges.empty())
      return {};
    const auto range = ranges[node * target_count + target];
    return std::span<const OperationCandidate>(candidates)
        .subspan(range.begin, range.count);
  }

  container::SmallVector<OperationCandidate, 4, mem::AllocatorRef> candidates;
  container::SmallVector<compute::ResourceRequirement, 16, mem::AllocatorRef>
      requirements;
  container::SmallVector<OperationCandidateRange, 16, mem::AllocatorRef> ranges;
};

// A resumable node-cover search over both leaves and regions. This owns only
// cold indices; proposed facts and their owners stay in the discovery pool.
// Every descent covers the smallest uncovered node, so depth is at most N.
class RouteSearch final {
public:
  static constexpr std::size_t kNoRoute = SIZE_MAX;
  struct Route final {
    std::size_t proposal = 0;
    std::uint32_t first = 0;
    std::uint32_t count = 1;
    std::uint32_t target = 0;
    std::uint32_t provider = 0;
    compute::ProviderId provider_family{};
    compute::ProviderImplementationId implementation{};
    compute::RoutePolicyId policy{};
    std::uint32_t preference_misses = 0;
    std::uint32_t policy_preference = 0;
    std::optional<std::uint32_t> dispatches{};
    bool region = false;
  };
  struct Frame final {
    std::uint32_t node = 0;
    std::size_t next = 0;
    std::size_t chosen = kNoRoute;
  };

  explicit RouteSearch(mem::AllocatorRef allocator) noexcept
      : routes_(allocator), begins_(allocator), frames_(allocator),
        occupied_(allocator), incidence_begins_(allocator),
        incidence_(allocator), blockers_(allocator), available_(allocator),
        excluded_(allocator) {}

  Status initialize(std::size_t node_count, std::size_t target_count,
                    const OperationDiscovery &operations,
                    std::span<const RegionCandidate> regions,
                    std::span<const NodeId> region_nodes,
                    std::span<const std::uint8_t> region_rejected,
                    std::span<const std::uint32_t> preferred) noexcept {
    operations_ = operations.candidates;
    regions_ = regions;
    region_nodes_ = region_nodes;
    region_rejected_ = region_rejected;
    if (!begins_.try_resize(node_count + 1) ||
        !occupied_.try_resize(node_count) || !frames_.try_reserve(node_count))
      return plan_allocation_failure();
    std::fill(begins_.begin(), begins_.end(), std::size_t{0});
    std::fill(occupied_.begin(), occupied_.end(), std::uint8_t{0});
    for (std::uint32_t node = 0; node < node_count; ++node) {
      for (std::uint32_t target = 0; target < target_count; ++target) {
        for (const auto &proposal :
             operations.for_node(node, target, target_count)) {
          if (!routes_.try_push_back(Route{
                  .proposal =
                      static_cast<std::size_t>(&proposal - operations_.data()),
                  .first = node,
                  .target = target,
                  .provider = static_cast<std::uint32_t>(
                      proposal.proposal.provider_index),
                  .provider_family = proposal.proposal.provider.id,
                  .implementation = proposal.proposal.implementation,
                  .policy = proposal.proposal.policy,
                  .preference_misses = preferred[node] != detail::kNoNode &&
                                               preferred[node] != target
                                           ? 1u
                                           : 0u,
                  .policy_preference = proposal.proposal.preference,
                  .dispatches = proposal.proposal.cost.dispatch_count}))
            return plan_allocation_failure();
        }
      }
    }
    for (std::size_t index = 0; index < regions.size(); ++index) {
      const auto &candidate = regions[index];
      if (region_rejected[candidate.proposal_order] != 0)
        continue;
      const auto nodes =
          region_nodes.subspan(candidate.node_offset, candidate.node_count);
      std::uint32_t misses = 0;
      for (const auto node : nodes)
        misses +=
            preferred[node.ordinal()] != detail::kNoNode &&
                    preferred[node.ordinal()] != candidate.target.ordinal()
                ? 1u
                : 0u;
      if (!routes_.try_push_back(
              Route{.proposal = index,
                    .first = nodes.front().ordinal(),
                    .count = candidate.node_count,
                    .target = candidate.target.ordinal(),
                    .provider = candidate.provider_index,
                    .provider_family = candidate.provider_family,
                    .implementation = candidate.implementation,
                    .policy = candidate.policy,
                    .preference_misses = misses,
                    .policy_preference = candidate.preference,
                    .dispatches = candidate.cost.dispatch_count,
                    .region = true}))
        return plan_allocation_failure();
    }
    const bool compare_dispatches = std::all_of(
        routes_.begin(), routes_.end(), [](const Route &route) noexcept {
          return route.dispatches.has_value();
        });
    std::sort(
        routes_.begin(), routes_.end(),
        [&](const Route &left, const Route &right) noexcept {
          if (left.first != right.first)
            return left.first < right.first;
          if (left.preference_misses != right.preference_misses)
            return left.preference_misses < right.preference_misses;
          if (left.policy_preference != right.policy_preference)
            return left.policy_preference < right.policy_preference;
          if (compare_dispatches && left.dispatches != right.dispatches)
            return left.dispatches < right.dispatches;
          if (left.region != right.region)
            return left.region;
          if (left.policy.namespace_id != right.policy.namespace_id)
            return left.policy.namespace_id < right.policy.namespace_id;
          if (left.policy.policy_id != right.policy.policy_id)
            return left.policy.policy_id < right.policy.policy_id;
          if (left.policy.version != right.policy.version)
            return left.policy.version < right.policy.version;
          if (left.provider_family.namespace_id !=
              right.provider_family.namespace_id) {
            return left.provider_family.namespace_id <
                   right.provider_family.namespace_id;
          }
          if (left.provider_family.provider_id !=
              right.provider_family.provider_id) {
            return left.provider_family.provider_id <
                   right.provider_family.provider_id;
          }
          if (left.implementation.value() != right.implementation.value()) {
            return left.implementation.value() < right.implementation.value();
          }
          if (left.target != right.target)
            return left.target < right.target;
          if (left.provider != right.provider)
            return left.provider < right.provider;
          if (left.count != right.count)
            return left.count > right.count;
          return left.proposal < right.proposal;
        });
    for (const auto &route : routes_)
      ++begins_[route.first + 1];
    for (std::size_t node = 1; node < begins_.size(); ++node)
      begins_[node] += begins_[node - 1];
    HYLE_RETURN_IF_ERROR(initialize_coverage(node_count));
    if (node_count != 0 && !frames_.try_push_back(Frame{.next = begins_[0]}))
      return plan_allocation_failure();
    return Status::Ok();
  }

  Result<bool> next(std::size_t budget, CompileSearchReport &report,
                    bool &limited) noexcept {
    limited = false;
    refresh_rejections();
    if (occupied_.empty()) {
      if (yielded_)
        return false;
      yielded_ = true;
      return true;
    }
    if (yielded_) {
      // A selected-route decline invalidates that entire subtree, not just
      // its last leaf. Other physical-plan failures backtrack normally.
      std::size_t keep = frames_.size();
      for (std::size_t index = 0; index < frames_.size(); ++index) {
        if (rejected(routes_[frames_[index].chosen])) {
          keep = index + 1;
          break;
        }
      }
      while (frames_.size() > keep) {
        undo(frames_.back());
        frames_.pop_back();
      }
      undo(frames_.back());
      yielded_ = false;
    }
    while (!frames_.empty()) {
      auto &frame = frames_.back();
      // A hole in the union of compatible candidates cannot be repaired by
      // enumerating unrelated prefix alternatives. Backtrack at its cause.
      if (unsupported_ != 0 || frame.next == begins_[frame.node + 1]) {
        frames_.pop_back();
        if (!frames_.empty())
          undo(frames_.back());
        continue;
      }
      if (report.route_search_states == budget) {
        limited = true;
        report.route_search_limit_reached = true;
        return std::unexpected(Status::ResourceExhausted(
            "graph route search state budget is exhausted"));
      }
      ++report.route_search_states;
      const auto index = frame.next++;
      if (excluded_[index] || blockers_[index] != 0)
        continue;
      frame.chosen = index;
      select(index);
      if (unsupported_ != 0) {
        undo(frame);
        continue;
      }
      if (covered_ == occupied_.size()) {
        yielded_ = true;
        return true;
      }
      auto next_node = frame.node + 1;
      while (occupied_[next_node] != 0)
        ++next_node;
      if (!frames_.try_push_back(Frame{next_node, begins_[next_node]}))
        return std::unexpected(plan_allocation_failure());
    }
    return false;
  }

  std::span<const Frame> frames() const noexcept { return frames_; }
  const Route &route(std::size_t index) const noexcept {
    return routes_[index];
  }
  std::size_t uncovered_node() const noexcept {
    for (std::size_t node = 0; node < available_.size(); ++node)
      if (!occupied_[node] && available_[node] == 0)
        return node;
    return kNoRoute;
  }

  template <class Visitor>
  void for_each_node(const Route &route, Visitor &&visit) const noexcept {
    if (!route.region) {
      visit(route.first);
      return;
    }
    const auto &region = regions_[route.proposal];
    for (const auto node :
         region_nodes_.subspan(region.node_offset, region.node_count))
      visit(node.ordinal());
  }

private:
  // Cold incidence index, not another routing table. Membership comes only
  // from actual leaf/region proposals. Each region contributes every member,
  // including nodes with no starting candidate of their own.
  Status initialize_coverage(std::size_t node_count) noexcept {
    if (!incidence_begins_.try_resize(node_count + 1) ||
        !available_.try_resize(node_count) ||
        !blockers_.try_resize(routes_.size()) ||
        !excluded_.try_resize(routes_.size()))
      return plan_allocation_failure();
    std::fill(incidence_begins_.begin(), incidence_begins_.end(), std::size_t{0});
    std::fill(blockers_.begin(), blockers_.end(), std::size_t{0});
    std::size_t count = 0;
    for (const auto &route : routes_) {
      if (route.count > SIZE_MAX - count)
        return Status::ResourceExhausted("graph route membership overflows");
      count += route.count;
      for_each_node(route, [&](std::uint32_t node) noexcept {
        ++incidence_begins_[node + 1];
      });
    }
    if (!incidence_.try_resize(count))
      return plan_allocation_failure();
    for (std::size_t node = 0; node < node_count; ++node) {
      incidence_begins_[node + 1] += incidence_begins_[node];
      available_[node] = incidence_begins_[node]; // temporary write head
    }
    for (std::size_t index = 0; index < routes_.size(); ++index) {
      excluded_[index] = rejected(routes_[index]);
      for_each_node(routes_[index], [&](std::uint32_t node) noexcept {
        incidence_[available_[node]++] = index;
      });
    }
    std::fill(available_.begin(), available_.end(), std::size_t{0});
    for (std::size_t index = 0; index < routes_.size(); ++index)
      if (!excluded_[index])
        for_each_node(routes_[index], [&](std::uint32_t node) noexcept {
          ++available_[node];
        });
    unsupported_ = static_cast<std::size_t>(
        std::count(available_.begin(), available_.end(), std::size_t{0}));
    return Status::Ok();
  }

  void remove_support(std::uint32_t node) noexcept {
    HYLE_FATAL_ASSERT(available_[node] != 0);
    if (--available_[node] == 0 && !occupied_[node])
      ++unsupported_;
  }
  void add_support(std::uint32_t node) noexcept {
    if (available_[node]++ == 0 && !occupied_[node])
      --unsupported_;
  }
  void refresh_rejections() noexcept {
    // Preparation only removes candidates. Do not resurrect a declined
    // implementation when undo makes an overlapping route compatible again.
    for (std::size_t index = 0; index < routes_.size(); ++index) {
      if (excluded_[index] || !rejected(routes_[index]))
        continue;
      excluded_[index] = 1;
      if (blockers_[index] == 0)
        for_each_node(routes_[index],
                      [&](std::uint32_t node) noexcept { remove_support(node); });
    }
  }
  void select(std::size_t index) noexcept {
    const auto &route = routes_[index];
    for_each_node(route, [&](std::uint32_t node) noexcept {
      if (available_[node] == 0)
        --unsupported_;
      occupied_[node] = 1;
    });
    for_each_node(route, [&](std::uint32_t node) noexcept {
      for (auto at = incidence_begins_[node]; at < incidence_begins_[node + 1]; ++at) {
        const auto other = incidence_[at];
        if (blockers_[other]++ == 0 && !excluded_[other])
          for_each_node(routes_[other], [&](std::uint32_t member) noexcept {
            remove_support(member);
          });
      }
    });
    covered_ += route.count;
  }

  bool rejected(const Route &route) const noexcept {
    return route.region
               ? region_rejected_[regions_[route.proposal].proposal_order] != 0
               : operations_[route.proposal].rejected;
  }
  void undo(Frame &frame) noexcept {
    const auto &route = routes_[frame.chosen];
    for_each_node(route, [&](std::uint32_t node) noexcept {
      for (auto at = incidence_begins_[node]; at < incidence_begins_[node + 1]; ++at) {
        const auto other = incidence_[at];
        HYLE_FATAL_ASSERT(blockers_[other] != 0);
        if (--blockers_[other] == 0 && !excluded_[other])
          for_each_node(routes_[other], [&](std::uint32_t member) noexcept {
            add_support(member);
          });
      }
    });
    for_each_node(route, [&](std::uint32_t node) noexcept {
      occupied_[node] = 0;
      if (available_[node] == 0)
        ++unsupported_;
    });
    covered_ -= route.count;
    frame.chosen = kNoRoute;
  }
  container::SmallVector<Route, 16, mem::AllocatorRef> routes_;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> begins_;
  container::SmallVector<Frame, 16, mem::AllocatorRef> frames_;
  container::SmallVector<std::uint8_t, 32, mem::AllocatorRef> occupied_;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> incidence_begins_;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> incidence_;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> blockers_;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> available_;
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> excluded_;
  std::span<const OperationCandidate> operations_{};
  std::span<const RegionCandidate> regions_{};
  std::span<const NodeId> region_nodes_{};
  std::span<const std::uint8_t> region_rejected_{};
  std::size_t covered_ = 0;
  std::size_t unsupported_ = 0;
  bool yielded_ = false;
};

// A dependent choice path, not a materialized Cartesian product. Replaying a
// prefix visits the same decisions; advancing a parent discards its suffix.
// Only genuine branches spend state budget or occupy this compile-only tape.
class ResourceSearch final {
public:
  explicit ResourceSearch(mem::AllocatorRef allocator) noexcept
      : path_(allocator) {}

  void reset() noexcept {
    path_.clear();
    cursor_ = 0;
  }
  bool branched() const noexcept { return !path_.empty(); }
  std::size_t prefix() const noexcept { return cursor_; }
  void reject_suffix(std::size_t prefix) noexcept {
    HYLE_FATAL_ASSERT(prefix <= cursor_);
    cursor_ = prefix;
    while (path_.size() > cursor_)
      path_.pop_back();
  }

  Result<bool> next(std::size_t budget, CompileSearchReport &report,
                    bool &limited) noexcept {
    while (path_.size() > cursor_)
      path_.pop_back();
    while (!path_.empty()) {
      auto &decision = path_.back();
      if (decision.selected + 1 < decision.count) {
        HYLE_RETURN_IF_ERROR(charge(budget, report, limited));
        ++decision.selected;
        cursor_ = 0;
        return true;
      }
      path_.pop_back();
    }
    cursor_ = 0;
    return false;
  }

  Result<std::uint32_t> choose(std::span<const std::uint32_t> choices,
                               std::size_t budget, CompileSearchReport &report,
                               bool &limited) noexcept {
    if (choices.empty())
      return std::unexpected(Status::Internal("empty graph resource choice"));
    if (choices.size() == 1)
      return choices.front();
    if (cursor_ == path_.size()) {
      HYLE_RETURN_IF_ERROR(charge(budget, report, limited));
      if (!path_.try_push_back(Decision{0, choices.size()}))
        return std::unexpected(plan_allocation_failure());
    }
    const auto &decision = path_[cursor_++];
    if (decision.count != choices.size() || decision.selected >= choices.size())
      return std::unexpected(Status::Internal(
          "graph resource choices changed within an identical prefix"));
    return choices[decision.selected];
  }

private:
  struct Decision final {
    std::size_t selected = 0;
    std::size_t count = 0;
  };
  static Status charge(std::size_t budget, CompileSearchReport &report,
                       bool &limited) noexcept {
    if (report.route_search_states == budget) {
      limited = true;
      report.route_search_limit_reached = true;
      return Status::ResourceExhausted(
          "graph resource search state budget is exhausted");
    }
    ++report.route_search_states;
    return Status::Ok();
  }
  container::SmallVector<Decision, 16, mem::AllocatorRef> path_;
  std::size_t cursor_ = 0;
};

Status discover_operation_candidates(
    const detail::PlanState &plan, const detail::GraphState &graph,
    CompileEnvironment environment, CompileOptions options,
    const ProviderPolicy &policy,
    std::span<const std::uint32_t> required_targets,
    OperationDiscovery &discovery, CompileSearchReport &search) noexcept {
  const auto target_count = environment.targets.size();
  if (graph.nodes.size() != 0 && target_count > SIZE_MAX / graph.nodes.size())
    return Status::ResourceExhausted("operation proposal index size overflows");
  if (!discovery.ranges.try_resize(graph.nodes.size() * target_count))
    return plan_allocation_failure();
  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  using Requirements = container::SmallVector<compute::ResourceRequirement, 4,
                                              mem::AllocatorRef>;
  Descriptions inputs(plan.allocator), outputs(plan.allocator);
  Requirements input_requirements(plan.allocator),
      output_requirements(plan.allocator);
  NodeShapeScratch shape_scratch(plan.allocator);
  for (std::size_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    const auto &node = graph.nodes[node_index];
    const bool metadata_view = node.kind() == compute::OperationKind::kView;
    if (!metadata_view &&
        options.operation_fallback == OperationFallbackPolicy::kReject)
      continue;
    if (!inputs.try_resize(node.input_count) ||
        !outputs.try_resize(node.output_count) ||
        !input_requirements.try_resize(node.input_count) ||
        !output_requirements.try_resize(node.output_count))
      return plan_allocation_failure();
    for (std::size_t index = 0; index < node.input_count; ++index)
      inputs[index] = plan.descriptions[graph.edges[node.input_begin + index]];
    for (std::size_t index = 0; index < node.output_count; ++index)
      outputs[index] = plan.descriptions[node.output_begin + index];
    HYLE_ASSIGN_OR_RETURN(auto shape, shape_scratch.bind(plan, node_index));
    for (std::size_t target = 0; target < target_count; ++target) {
      auto &range = discovery.ranges[node_index * target_count + target];
      range.begin = discovery.candidates.size();
      if (required_targets[node_index] != detail::kNoWorkspaceRequest &&
          required_targets[node_index] != target)
        continue;
      if (metadata_view) {
        // A per-node Provider constraint can only be satisfied by a real
        // region covering this View, never by an unowned metadata alternative.
        if (!policy.allows_node(node_index, {}, {}))
          continue;
        compute::detail::ProviderOperationProposal facts;
        facts.runtime = {
            .mode = compute::ProviderRuntimeMode::kSharedQueue,
            .workspace_owner = compute::ProviderWorkspaceOwner::kCaller,
            .replay_class = compute::ProviderReplayClass::kPrepared,
            .capture_compatible = true};
        facts.cost = {
            .workspace = compute::WorkspaceRequirement{},
            .persistent_bytes = 0,
            .copy_bytes = 0,
            .repack_bytes = 0,
            .synchronization_count = 0,
            .dispatch_count = 0,
            .compilation_storage = {.resident_bytes = 0, .peak_bytes = 0}};
        facts.preference = 0;
        if (!discovery.candidates.try_push_back(
                {.proposal = facts, .metadata_view = true}))
          return plan_allocation_failure();
        ++range.count;
        continue;
      }
      auto retained = detail::ExecutionSourceAccess::compute(
          environment.targets[target].source);
      if (!retained) {
        if (retained.error().IsUnimplemented())
          continue;
        return std::move(retained).error();
      }
      const auto &context = **retained;
      for (std::size_t provider = 0; provider < context.provider_count();
           ++provider) {
        HYLE_ASSIGN_OR_RETURN(auto allowed, policy.allows_operation(
                                                context, provider, node_index));
        if (!allowed)
          continue;
        if (search.operation_proposals == options.max_operation_proposals) {
          search.operation_proposal_limit_reached = true;
          continue;
        }
        auto proposal = compute::detail::OperationPreparationAccess::propose(
            context, provider, node.operation(), inputs, outputs,
            input_requirements, output_requirements, shape, plan.allocator,
            options.workload);
        if (!proposal) {
          if (proposal.error().IsUnimplemented())
            continue;
          return std::move(proposal).error();
        }
        if (!replay_allowed(options.provider_replay,
                            proposal->runtime.replay_class))
          continue;
        const std::size_t requirement_count = inputs.size() + outputs.size();
        const std::size_t begin = discovery.requirements.size();
        if (requirement_count > SIZE_MAX - begin)
          return Status::ResourceExhausted(
              "operation proposal requirement storage overflows");
        if (!discovery.requirements.try_resize(begin + requirement_count) ||
            !discovery.candidates.try_push_back(
                OperationCandidate{*proposal, begin}))
          return plan_allocation_failure();
        std::copy(input_requirements.begin(), input_requirements.end(),
                  discovery.requirements.begin() + begin);
        std::copy(output_requirements.begin(), output_requirements.end(),
                  discovery.requirements.begin() + begin + inputs.size());
        ++range.count;
        ++search.operation_proposals;
      }
    }
  }
  return Status::Ok();
}

// Immutable placement input, validated once for this compilation. Actual
// resource authority and the selected physical instances remain attempt-local.
struct PlacementFacts final {
  using Ordinals =
      container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Flags = container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;

  explicit PlacementFacts(mem::AllocatorRef allocator) noexcept
      : node_required(allocator), node_preferred(allocator),
        value_required(allocator), value_preferred(allocator),
        external_seen(allocator) {}

  Ordinals node_required;
  Ordinals node_preferred;
  Ordinals value_required;
  Ordinals value_preferred;
  Flags external_seen;
};

enum class RegionCoverage : std::uint8_t { kNone, kEligible, kRejected };

// One discovery per compile, shared by the bounded construction attempts.
// Proposal ordinals remain stable while the eligible partition changes.
struct RegionDiscovery final {
  explicit RegionDiscovery(mem::AllocatorRef allocator) noexcept
      : placement(allocator), extensions(allocator), candidates(allocator),
        nodes(allocator), rejected(allocator), resources(allocator),
        values(allocator), requirements(allocator), input_references(allocator),
        output_contracts(allocator), effects(allocator), operations(allocator),
        region_coverage(allocator), routes(allocator), physical(allocator) {}

  PlacementFacts placement;
  RegionExtensions extensions;
  RegionCandidates candidates;
  RegionNodeIds nodes;
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> rejected;
  container::SmallVector<RegionResourceSkeleton, 4, mem::AllocatorRef>
      resources;
  RegionValues values;
  container::SmallVector<compute::ResourceRequirement, 16, mem::AllocatorRef>
      requirements;
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> input_references;
  container::SmallVector<RegionOutputContract, 16, mem::AllocatorRef>
      output_contracts;
  container::SmallVector<RegionEffectAccess, 4, mem::AllocatorRef> effects;
  OperationDiscovery operations;
  container::SmallVector<RegionCoverage, 32, mem::AllocatorRef> region_coverage;
  RouteSearch routes;
  ResourceSearch physical;
  Status last_rejection = Status::Ok();
  bool ready = false;
  bool has_route = false;
  std::span<const detail::RegionAdapter> adapters;
};

struct PlacementAttemptOutcome final {
  std::optional<std::uint32_t> rejected_region{};
  std::optional<std::size_t> rejected_operation{};
  bool rejected_plan = false;
  bool no_feasible_route = false;
  bool search_limit = false;
  bool prepare_limit = false;
};

// Cold compile scratch. Keep the original adjacency once and sort a proposed
// partition in O(V + E + U log U), where U is the number of contracted units.
// Duplicate data/control edges need no deduplication: each increment has one
// matching decrement. The same checker serves selection and final emission.
class RegionContraction final {
public:
  explicit RegionContraction(mem::AllocatorRef allocator) noexcept
      : begins_(allocator), successors_(allocator), heads_(allocator),
        next_(allocator), indegree_(allocator), ready_(allocator),
        order_(allocator) {}

  Status initialize(const detail::GraphState &graph) noexcept {
    const std::size_t count = graph.nodes.size();
    if (count >= detail::kNoNode || graph.edges.size() > detail::kNoNode ||
        graph.dependencies.size() > detail::kNoNode - graph.edges.size()) {
      return Status::ResourceExhausted(
          "graph contracted DAG exceeds its ordinal capacity");
    }
    if (!begins_.try_resize(count + 1) || !heads_.try_resize(count) ||
        !next_.try_resize(count) || !indegree_.try_resize(count) ||
        !ready_.try_reserve(count) || !order_.try_reserve(count)) {
      return plan_allocation_failure();
    }
    std::fill(begins_.begin(), begins_.end(), std::uint32_t{0});
    auto visit_edges = [&](auto &&visit) noexcept {
      for (std::uint32_t destination = 0; destination < count; ++destination) {
        const detail::NodeRecord &node = graph.nodes[destination];
        for (std::size_t index = 0; index < node.input_count; ++index) {
          const auto value = graph.edges[node.input_begin + index];
          const auto source = graph.values[value].producer;
          if (source != detail::kNoNode)
            visit(source, destination);
        }
        for (std::size_t index = 0; index < node.dependency_count; ++index)
          visit(graph.dependencies[node.dependency_begin + index], destination);
      }
    };
    visit_edges([&](std::uint32_t source, std::uint32_t) noexcept {
      ++begins_[source + 1];
    });
    for (std::size_t index = 1; index < begins_.size(); ++index)
      begins_[index] += begins_[index - 1];
    if (!successors_.try_resize(begins_.back()))
      return plan_allocation_failure();
    std::copy_n(begins_.begin(), count, indegree_.begin());
    visit_edges([&](std::uint32_t source, std::uint32_t destination) noexcept {
      successors_[indegree_[source]++] = destination;
    });
    return Status::Ok();
  }

  // Each node maps to the smallest node ordinal in its selected region, or
  // to itself when it is a leaf. false means the proposed contraction cycles.
  Result<bool> sort(std::span<const std::uint32_t> owners) noexcept {
    if (owners.size() != heads_.size())
      return std::unexpected(Status::Internal("graph partition size mismatch"));
    std::fill(heads_.begin(), heads_.end(), detail::kNoNode);
    std::fill(indegree_.begin(), indegree_.end(), std::uint32_t{0});
    ready_.clear();
    order_.clear();
    for (std::uint32_t node = 0; node < owners.size(); ++node) {
      const std::uint32_t owner = owners[node];
      if (owner >= owners.size() || owners[owner] != owner)
        return std::unexpected(
            Status::Internal("graph partition owner is invalid"));
      next_[node] = heads_[owner];
      heads_[owner] = node;
      for (std::size_t index = begins_[node]; index < begins_[node + 1];
           ++index) {
        const std::uint32_t destination = owners[successors_[index]];
        if (destination != owner)
          ++indegree_[destination];
      }
    }
    const auto later = [](std::uint32_t left, std::uint32_t right) noexcept {
      return left > right;
    };
    std::size_t unit_count = 0;
    for (std::uint32_t unit = 0; unit < owners.size(); ++unit) {
      if (heads_[unit] == detail::kNoNode)
        continue;
      ++unit_count;
      if (indegree_[unit] == 0) {
        if (!ready_.try_push_back(unit))
          return std::unexpected(plan_allocation_failure());
        std::push_heap(ready_.begin(), ready_.end(), later);
      }
    }
    while (!ready_.empty()) {
      std::pop_heap(ready_.begin(), ready_.end(), later);
      const std::uint32_t unit = ready_.back();
      ready_.pop_back();
      if (!order_.try_push_back(unit))
        return std::unexpected(plan_allocation_failure());
      for (std::uint32_t node = heads_[unit]; node != detail::kNoNode;
           node = next_[node]) {
        for (std::size_t index = begins_[node]; index < begins_[node + 1];
             ++index) {
          const std::uint32_t destination = owners[successors_[index]];
          if (destination == unit)
            continue;
          if (indegree_[destination] == 0)
            return std::unexpected(
                Status::Internal("graph partition degree underflow"));
          if (--indegree_[destination] == 0) {
            if (!ready_.try_push_back(destination))
              return std::unexpected(plan_allocation_failure());
            std::push_heap(ready_.begin(), ready_.end(), later);
          }
        }
      }
    }
    return order_.size() == unit_count;
  }

  std::span<const std::uint32_t> order() const noexcept { return order_; }

private:
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  Ordinals begins_;
  Ordinals successors_;
  Ordinals heads_;
  Ordinals next_;
  Ordinals indegree_;
  Ordinals ready_;
  Ordinals order_;
};

std::span<const NodeId>
candidate_nodes(const RegionCandidate &candidate,
                std::span<const NodeId> storage) noexcept {
  return storage.subspan(candidate.node_offset, candidate.node_count);
}

bool nodes_are_contiguous(std::span<const NodeId> nodes) noexcept {
  if (nodes.empty())
    return false;
  for (std::size_t index = 1; index < nodes.size(); ++index) {
    if (nodes[index].ordinal() != nodes[index - 1].ordinal() + 1)
      return false;
  }
  return true;
}

Result<bool> validate_contractible_region(
    const detail::GraphState &graph, std::span<const NodeId> nodes,
    container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> &selected,
    container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>
        &reachable_outside) noexcept {
  if (nodes.empty())
    return std::unexpected(
        Status::InvalidArgument("graph region node set is empty"));
  if (nodes_are_contiguous(nodes)) {
    for (NodeId node : nodes) {
      if (!detail::owns(graph, node))
        return std::unexpected(Status::InvalidArgument(
            "provider proposed a graph region node from another graph"));
    }
    return true;
  }
  std::uint32_t first = detail::kNoNode;
  std::uint32_t last = 0;
  for (NodeId node : nodes) {
    if (!detail::owns(graph, node)) {
      return std::unexpected(Status::InvalidArgument(
          "provider proposed a graph region node from another graph"));
    }
    first = (std::min)(first, node.ordinal());
    last = (std::max)(last, node.ordinal());
  }
  const auto span = static_cast<std::size_t>(last) - first + 1;
  if (!selected.try_resize(span) || !reachable_outside.try_resize(span))
    return std::unexpected(plan_allocation_failure());
  std::fill(selected.begin(), selected.end(), std::uint8_t{0});
  std::fill(reachable_outside.begin(), reachable_outside.end(),
            std::uint8_t{0});
  for (NodeId node : nodes) {
    const auto index = node.ordinal() - first;
    if (selected[index] != 0)
      return std::unexpected(
          Status::InvalidArgument("graph region node set has duplicates"));
    selected[index] = 1;
  }
  // In a canonical DAG every path has increasing ordinals, so an interval
  // (including a singleton) cannot leave the set and then re-enter it.
  if (span == nodes.size())
    return true;

  auto predecessor_state = [&](std::uint32_t predecessor,
                               bool current_selected) noexcept {
    // Before the first selected node no path can have left this region.
    // Later predecessors cannot occur in the canonical DAG.
    if (predecessor == detail::kNoNode || predecessor < first ||
        predecessor > last)
      return false;
    const auto index = predecessor - first;
    if (current_selected)
      return selected[index] == 0 && reachable_outside[index] != 0;
    return selected[index] != 0 || reachable_outside[index] != 0;
  };
  for (std::uint32_t node_index = first; node_index <= last; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    const auto index = node_index - first;
    const bool current_selected = selected[index] != 0;
    bool reached = false;
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t value = graph.edges[node.input_begin + input_index];
      reached = reached || predecessor_state(graph.values[value].producer,
                                             current_selected);
    }
    for (std::size_t dependency_index = 0;
         dependency_index < node.dependency_count; ++dependency_index) {
      reached =
          reached ||
          predecessor_state(
              graph.dependencies[node.dependency_begin + dependency_index],
              current_selected);
    }
    if (current_selected && reached) {
      return false;
    }
    if (!current_selected)
      reachable_outside[index] = reached ? 1 : 0;
  }
  return true;
}

// One graph-wide use count, then only the nodes/ports of each candidate.
// Stamps avoid clearing a whole-graph bitmap for every proposed region.
class RegionBoundaryIndex final {
public:
  explicit RegionBoundaryIndex(mem::AllocatorRef allocator) noexcept
      : node_marks_(allocator), value_marks_(allocator), uses_(allocator),
        internal_uses_(allocator) {}

  Status initialize(const detail::GraphState &graph) noexcept {
    if (!node_marks_.try_resize(graph.nodes.size()) ||
        !value_marks_.try_resize(graph.values.size()) ||
        !uses_.try_resize(graph.values.size()) ||
        !internal_uses_.try_resize(graph.values.size()))
      return plan_allocation_failure();
    std::fill(node_marks_.begin(), node_marks_.end(), std::uint64_t{0});
    std::fill(value_marks_.begin(), value_marks_.end(), std::uint64_t{0});
    std::fill(uses_.begin(), uses_.end(), std::size_t{0});
    for (const auto value : graph.edges)
      ++uses_[value];
    return Status::Ok();
  }

  Status collect(const detail::PlanState &plan, const detail::GraphState &graph,
                 std::span<const NodeId> nodes, std::uint64_t marker,
                 RegionValues &inputs, RegionValues &outputs) noexcept {
    inputs.clear();
    outputs.clear();
    for (const auto node : nodes)
      node_marks_[node.ordinal()] = marker;
    for (const auto node_id : nodes) {
      const auto &node = graph.nodes[node_id.ordinal()];
      for (std::size_t port = 0; port < node.input_count; ++port) {
        const auto value = graph.edges[node.input_begin + port];
        const bool first_use = value_marks_[value] != marker;
        if (first_use) {
          value_marks_[value] = marker;
          internal_uses_[value] = 0;
        }
        ++internal_uses_[value];
        const auto producer = graph.values[value].producer;
        if (first_use &&
            (producer == detail::kNoNode || node_marks_[producer] != marker) &&
            !inputs.try_push_back(detail::value_id(graph, value)))
          return plan_allocation_failure();
      }
    }
    for (const auto node_id : nodes) {
      const auto &node = graph.nodes[node_id.ordinal()];
      for (std::size_t port = 0; port < node.output_count; ++port) {
        const auto value = node.output_begin + port;
        const auto inside =
            value_marks_[value] == marker ? internal_uses_[value] : 0;
        const bool published =
            (plan.binding_roles[value] & detail::kExternalOutput) != 0;
        if ((published || uses_[value] > inside) &&
            !outputs.try_push_back(
                detail::value_id(graph, static_cast<std::uint32_t>(value))))
          return plan_allocation_failure();
      }
    }
    return Status::Ok();
  }

private:
  container::SmallVector<std::uint64_t, 16, mem::AllocatorRef> node_marks_,
      value_marks_;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> uses_,
      internal_uses_;
};

Status propose_region_resources(
    const RegionExtension &extension, const RegionCompileRequest &request,
    mem::AllocatorRef allocator,
    std::span<RegionInputRequirement> input_requirements,
    std::span<RegionOutputRequirement> output_requirements,
    RegionResourceProposal &resource_proposal) noexcept {
  for (auto &requirement : input_requirements)
    requirement = {};
  for (auto &requirement : output_requirements)
    requirement = {};
  resource_proposal = {};
  struct ProposalEpisode final {
    const RegionCompileRequest *request = nullptr;
    mem::AllocatorRef allocator{};
    std::span<RegionInputRequirement> input_requirements{};
    std::span<RegionOutputRequirement> output_requirements{};
    RegionResourceProposal *proposal = nullptr;
  } proposal_episode{&request, allocator, input_requirements,
                     output_requirements, &resource_proposal};
  Status proposed = extension.call(
      +[](void *provider_context, const void *interface_value,
          void *opaque) noexcept -> Status {
        const auto *interface =
            static_cast<const RegionProviderVTable *>(interface_value);
        auto &call = *static_cast<ProposalEpisode *>(opaque);
        if (interface == nullptr || interface->propose_resources == nullptr) {
          return Status::Internal(
              "graph region resource proposal callback is missing");
        }
        return interface->propose_resources(
            provider_context, *call.request, call.allocator,
            call.input_requirements, call.output_requirements, call.proposal);
      },
      &proposal_episode);
  if (!proposed.ok())
    return proposed;
  if (!resource_proposal.runtime.valid()) {
    return Status::Internal(
        "graph region provider proposed invalid runtime properties");
  }
  if (!resource_proposal.compilation_storage.valid())
    return Status::Internal(
        "graph region provider proposed invalid compilation storage");
  HYLE_RETURN_IF_ERROR(
      compute::detail::OperationPreparationAccess::validate_requirement(
          resource_proposal.workspace_resource));
  for (const auto &requirement : input_requirements)
    HYLE_RETURN_IF_ERROR(
        compute::detail::OperationPreparationAccess::validate_requirement(requirement.resource));
  for (const auto &requirement : output_requirements) {
    HYLE_RETURN_IF_ERROR(
        compute::detail::OperationPreparationAccess::validate_requirement(requirement.resource));
    if (!requirement.value.valid())
      return Status::Internal("graph region proposed an invalid output carrier");
  }
  return Status::Ok();
}

Status
discover_region_resources(const Graph &published_graph,
                          const detail::GraphState &graph,
                          detail::PlanState &plan, CompileEnvironment environment,
                          CompileOptions options,
                          std::span<const std::uint32_t> required_targets,
                          RegionDiscovery &discovery) noexcept {
  std::size_t eligible = 0;
  for (const auto &candidate : discovery.candidates) {
    const auto nodes = candidate_nodes(candidate, discovery.nodes);
    if (!std::all_of(nodes.begin(), nodes.end(), [&](NodeId node) noexcept {
          return required_targets[node.ordinal()] ==
                     detail::kNoWorkspaceRequest ||
                 required_targets[node.ordinal()] == candidate.target.ordinal();
        }))
      continue;
    discovery.candidates[eligible++] = candidate;
  }
  discovery.candidates.try_resize(eligible);
  if (discovery.candidates.empty())
    return Status::Ok();
  if (!discovery.resources.try_resize(plan.search.region_proposals))
    return plan_allocation_failure();
  RegionBoundaryIndex boundary(plan.allocator);
  HYLE_RETURN_IF_ERROR(boundary.initialize(graph));
  RegionValues inputs(plan.allocator), outputs(plan.allocator);
  container::SmallVector<RegionInputRequirement, 8, mem::AllocatorRef>
      input_requirements(plan.allocator);
  container::SmallVector<RegionOutputRequirement, 8, mem::AllocatorRef>
      output_requirements(plan.allocator);
  container::SmallVector<RegionEffectAccess, 4, mem::AllocatorRef> effects(
      plan.allocator);
  for (const auto &candidate : discovery.candidates) {
    const auto nodes = candidate_nodes(candidate, discovery.nodes);
    HYLE_RETURN_IF_ERROR(boundary.collect(
        plan, graph, nodes,
        static_cast<std::uint64_t>(candidate.proposal_order) + 1, inputs,
        outputs));
    if (!input_requirements.try_resize(inputs.size()) ||
        !output_requirements.try_resize(outputs.size()))
      return plan_allocation_failure();
    effects.clear();
    bool unknown_effects = false;
    for (const auto node_id : nodes) {
      const auto &node = graph.nodes[node_id.ordinal()];
      const auto traits = node.traits();
      unknown_effects |=
          traits.effect == compute::EffectKind::kExternalSideEffect;
      for (std::size_t port = 0; port < traits.resource_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto access, compute::operation_resource_access(
                                               node.operation(), port));
        if (!effects.try_push_back(RegionEffectAccess{
                detail::GraphIdentity::effect_resource(
                    graph.generation,
                    graph.resource_edges[node.resource_begin + port]),
                access}))
          return plan_allocation_failure();
      }
    }
    std::sort(effects.begin(), effects.end(),
              [](const auto &left, const auto &right) noexcept {
                return left.resource.ordinal() < right.resource.ordinal();
              });
    std::size_t effect_count = 0;
    for (const auto access : effects) {
      if (effect_count != 0 &&
          effects[effect_count - 1].resource == access.resource) {
        if (effects[effect_count - 1].access != access.access)
          effects[effect_count - 1].access =
              compute::ResourceAccess::kReadWrite;
      } else {
        effects[effect_count++] = access;
      }
    }
    effects.try_resize(effect_count);
    const RegionCompileRequest request{.graph = &published_graph,
                                       .nodes = nodes,
                                       .inputs = inputs,
                                       .outputs = outputs,
                                       .constants = options.constants,
                                       .route = candidate.route,
                                       .effect_accesses = effects,
                                       .effect_bindings = plan.effect_bindings,
                                       .has_unknown_effects = unknown_effects,
                                       .workload = options.workload,
                                       .resource_budget = options.resource_budget,
                                       .available_storage_domains = environment.storage_domains};
    RegionResourceProposal proposal;
    Status status = propose_region_resources(
        discovery.extensions[candidate.extension_index], request,
        plan.allocator, input_requirements, output_requirements, proposal);
    if (status.ok() &&
        candidate.replay_class != compute::ProviderReplayClass::kUnknown &&
        candidate.replay_class != proposal.runtime.replay_class)
      return Status::Internal(
          "region provider changed its declared replay class");
    if (status.ok() && (unknown_effects || !effects.empty()) &&
        !proposal.preserves_effects)
      status = Status::Unimplemented(
          "region provider does not preserve requested resource effects");
    if (!status.ok()) {
      if (!status.IsUnimplemented())
        return status;
      discovery.rejected[candidate.proposal_order] = 1;
      ++plan.search.region_route_rejections;
      discovery.last_rejection = std::move(status);
      continue;
    }
    if (inputs.size() > SIZE_MAX - outputs.size())
      return Status::ResourceExhausted("region boundary size overflows");
    const auto port_count = inputs.size() + outputs.size();
    const auto begin = discovery.values.size();
    const auto input_reference_begin = discovery.input_references.size();
    const auto output_contract_begin = discovery.output_contracts.size();
    const auto effect_begin = discovery.effects.size();
    if (port_count > SIZE_MAX - begin ||
        inputs.size() > SIZE_MAX - input_reference_begin ||
        outputs.size() > SIZE_MAX - output_contract_begin ||
        effects.size() > SIZE_MAX - effect_begin)
      return Status::ResourceExhausted(
          "region resource skeleton storage overflows");
    if (!discovery.values.try_resize(begin + port_count) ||
        !discovery.requirements.try_resize(begin + port_count) ||
        !discovery.input_references.try_resize(input_reference_begin + inputs.size()) ||
        !discovery.output_contracts.try_resize(output_contract_begin + outputs.size()) ||
        !discovery.effects.try_resize(effect_begin + effects.size()))
      return plan_allocation_failure();
    std::copy(inputs.begin(), inputs.end(), discovery.values.begin() + begin);
    std::copy(outputs.begin(), outputs.end(),
              discovery.values.begin() + begin + inputs.size());
    for (std::size_t port = 0; port < input_requirements.size(); ++port) {
      discovery.requirements[begin + port] = input_requirements[port].resource;
      discovery.input_references[input_reference_begin + port] =
          input_requirements[port].accepts_reference;
    }
    for (std::size_t port = 0; port < output_requirements.size(); ++port) {
      discovery.requirements[begin + inputs.size() + port] = output_requirements[port].resource;
      discovery.output_contracts[output_contract_begin + port] = output_requirements[port].value;
    }
    std::copy(effects.begin(), effects.end(),
              discovery.effects.begin() + effect_begin);
    discovery.resources[candidate.proposal_order] =
        RegionResourceSkeleton{.proposal = proposal,
                               .port_begin = begin,
                               .input_reference_begin = input_reference_begin,
                               .output_contract_begin = output_contract_begin,
                               .input_count = inputs.size(),
                               .output_count = outputs.size(),
                               .effect_begin = effect_begin,
                               .effect_count = effects.size(),
                               .has_unknown_effects = unknown_effects};
  }
  return Status::Ok();
}

struct RegionPrepareAttempt final {
  detail::PreparedRegion prepared{};
  detail::PreparedConstantOutputs outputs{};
  compute::detail::ProviderPrepareDecline decline =
      compute::detail::ProviderPrepareDecline::kNone;
  Status reason = Status::Ok();
  bool compilation_budget_decline = false;
};

Result<RegionPrepareAttempt> compile_region_artifact(
    detail::PlanState &plan, const RegionExtension &extension,
    RegionCompileRequest request,
    const RegionResourceProposal &resource_proposal,
    std::span<const std::uint8_t> input_references,
    std::span<const RegionOutputContract> output_contracts, CompileOptions options,
    detail::ConstantArtifactCache &packing) noexcept {
  if (input_references.size() != request.inputs.size() ||
      output_contracts.size() != request.outputs.size())
    return std::unexpected(Status::Internal("graph region carrier arity differs"));
  detail::ConstantArtifactScope scope{packing, plan, extension.source,
      extension.target, options, request.constants, request.outputs, request.output_resources,
      detail::PreparedConstantOutputs(plan.allocator)};
  scope.constant_origins = request.constant_origins;
  request.constant_artifacts = scope.access();
  ProviderRegionPrepared provider_prepared{};
  struct Episode final {
    const RegionCompileRequest *request = nullptr;
    mem::AllocatorRef allocator{};
    ProviderRegionPrepared *prepared = nullptr;
    compute::ProviderPrepareOutcome outcome{};
  } episode{&request, plan.allocator, &provider_prepared};
  const Status compiled = extension.call(
      +[](void *provider_context, const void *interface_value,
          void *opaque) noexcept -> Status {
        const auto *interface =
            static_cast<const RegionProviderVTable *>(interface_value);
        auto &call = *static_cast<Episode *>(opaque);
        if (interface == nullptr || interface->compile == nullptr)
          return Status::Internal("graph region compile callback is missing");
        call.outcome = interface->compile(provider_context, *call.request,
                                          call.allocator, call.prepared);
        return Status::Ok();
      },
      &episode);
  if (!compiled.ok()) {
    if (provider_prepared.vtable != nullptr &&
        provider_prepared.vtable->destroy_prepared != nullptr) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 plan.allocator);
    }
    return std::unexpected(compiled);
  }
  const ProviderRegionPreparedVTable *table = provider_prepared.vtable;
  auto &outcome = episode.outcome;
  if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared ||
      !outcome.valid()) {
    if (provider_prepared.state != nullptr &&
        (table == nullptr || table->destroy_prepared == nullptr)) {
      return std::unexpected(Status::Internal(
          "graph region provider returned an unowned partial artifact"));
    }
    if (table != nullptr && table->destroy_prepared != nullptr)
      table->destroy_prepared(provider_prepared.state, plan.allocator);
    if (!outcome.valid()) {
      return std::unexpected(Status::Internal(
          "graph region provider returned an invalid prepare outcome"));
    }
    if (outcome.kind == compute::ProviderPrepareOutcomeKind::kError)
      return std::unexpected(std::move(outcome.status));
    return RegionPrepareAttempt{
        .decline =
            outcome.kind ==
                    compute::ProviderPrepareOutcomeKind::kDeclinedUnsupported
                ? compute::detail::ProviderPrepareDecline::kUnsupported
                : compute::detail::ProviderPrepareDecline::kResourceBudget,
        .reason = std::move(outcome.status),
        .compilation_budget_decline = scope.budget_declined};
  }
  const compute::WorkspaceRequirement workspace =
      provider_prepared.properties.workspace;
  const compute::ProviderRuntimeProperties runtime =
      provider_prepared.properties.runtime;
  const bool shared =
      runtime.mode == compute::ProviderRuntimeMode::kSharedQueue;
  const auto &properties = provider_prepared.properties;
  // Zero dispatch is a cold, immutable result, never a substitute for an
  // outputless effect or a provider episode hidden from Queue scheduling.
  // bind_prepared_output has already checked each binding's exact resource,
  // descriptor, acquired owner and unpublished output identity.
  const bool ready_only = properties.dispatch_count == 0 && shared &&
      extension.source.valid() && !extension.source.is_native() &&
      !request.outputs.empty() &&
      scope.outputs.size() == request.outputs.size() &&
      std::ranges::all_of(request.outputs, [&](ValueId output) {
        return std::ranges::count_if(scope.outputs, [&](const auto &entry) {
          return entry.value == output;
        }) == 1;
      }) &&
      request.effect_accesses.empty() && !request.has_unknown_effects &&
      workspace.size == 0 && properties.copy_bytes == 0 &&
      properties.packed_constant_count == 0 &&
      properties.packed_constant_bytes == 0 &&
      properties.synchronization_count == 0 &&
      properties.invocation_private_bytes == std::optional<std::size_t>{0};
  if (table == nullptr || table->destroy_prepared == nullptr ||
      table->alias_support == nullptr || table->create_invocation == nullptr ||
      table->destroy_invocation == nullptr ||
      (shared &&
       (table->shared_submit == nullptr || table->delegated_submit != nullptr ||
        table->endpoint != nullptr || table->can_import_endpoint != nullptr)) ||
      (!shared &&
       (table->shared_submit != nullptr || table->delegated_submit == nullptr ||
        table->endpoint == nullptr || table->can_import_endpoint == nullptr)) ||
      !runtime.valid() ||
      !compute::detail::prepared_runtime_matches(runtime, resource_proposal.runtime) ||
      (std::ranges::any_of(input_references, [](auto accepts) { return accepts != 0; }) &&
       table->prepare_bindings == nullptr) ||
      (extension.source.is_native() && shared) ||
      workspace.alignment == 0 ||
      (workspace.alignment & (workspace.alignment - 1)) != 0 ||
      (runtime.workspace_owner == compute::ProviderWorkspaceOwner::kCaller &&
       workspace.size != 0 && !request.workspace_resource.valid()) ||
      (runtime.workspace_owner == compute::ProviderWorkspaceOwner::kProvider &&
       request.workspace_resource.valid()) ||
      (properties.dispatch_count == 0 && !ready_only) ||
      provider_prepared.properties.preserves_effects !=
          resource_proposal.preserves_effects ||
      provider_prepared.properties.packed_constant_count >
          options.constants.size()) {
    if (table != nullptr && table->destroy_prepared != nullptr)
      table->destroy_prepared(provider_prepared.state, plan.allocator);
    return std::unexpected(Status::Internal(
        "graph region provider returned an invalid compiled artifact"));
  }
  std::size_t live_port = 0;
  for (std::size_t port = 0; port < request.outputs.size(); ++port) {
    // Immutable prepared values are ready owned bindings, not runtime native
    // ports. Match by logical identity before advancing the live port index.
    if (std::ranges::any_of(scope.outputs,
            [&](const auto &entry) { return entry.value == request.outputs[port]; }))
      continue;
    const auto actual = table->output_contract
        ? table->output_contract(provider_prepared.state, live_port)
        : RegionOutputContract{};
    ++live_port;
    if (!actual.valid() || actual != output_contracts[port]) {
      table->destroy_prepared(provider_prepared.state, plan.allocator);
      return std::unexpected(Status::Internal(
          "graph region changed its proposed output carrier"));
    }
    const auto kind = actual.kind;
    if (kind == RegionOutputKind::kCaller) continue;
    if (shared || table->output_value == nullptr ||
        (kind == RegionOutputKind::kBoundGenerated && table->prepare_bindings == nullptr)) {
      table->destroy_prepared(provider_prepared.state, plan.allocator);
      return std::unexpected(Status::Internal("generated region output contract is incomplete"));
    }
  }
  return RegionPrepareAttempt{
      .prepared = detail::PreparedRegion(extension.source, provider_prepared,
                                         plan.allocator),
      .outputs = std::move(scope.outputs)};
}

Status discover_region_candidates(
    const Graph &published_graph, const detail::GraphState &graph,
    const ExecutionSource &source, CompileOptions options,
    const ProviderPolicy &provider_policy, mem::AllocatorRef allocator,
    RegionExtensions &extensions, RegionCandidates &candidates,
    RegionNodeIds &proposal_nodes, std::size_t &proposal_count,
    std::size_t &proposal_node_count, CompileSearchReport &search,
    ExecutionTargetId target, std::span<const std::uint32_t> node_targets = {},
    const detail::GraphState *domain = nullptr,
    std::span<const detail::RegionAdapter> adapters = {},
    const OperationDiscovery *operations = nullptr) noexcept;

struct CompilationReservation final {
  detail::StorageCharge resident;
  detail::StorageCharge transient;
};

Result<CompilationReservation> reserve_compilation_storage(
    detail::PlanState &plan, CompileOptions options,
    compute::ProviderCompilationStorage storage, bool &budget_declined) noexcept {
  budget_declined = false;
  if (!plan.compilation_storage)
    return CompilationReservation{};
  const auto limits = *options.compilation_storage;
  const bool unknown = !storage.resident_bytes || !storage.peak_bytes;
  if (unknown && limits.unknown == UnknownStoragePolicy::kReject &&
      (limits.max_resident_bytes != SIZE_MAX ||
       limits.max_peak_bytes != SIZE_MAX)) {
    budget_declined = true;
    return std::unexpected(Status::ResourceExhausted(
        "graph provider compilation storage upper bound is unknown"));
  }
  const auto resident = storage.resident_bytes.value_or(0);
  const auto peak = (std::max)(resident, storage.peak_bytes.value_or(0));
  auto retained = plan.compilation_storage->reserve_private(storage.resident_bytes, &budget_declined);
  if (!retained)
    return std::unexpected(std::move(retained).error());
  auto transient = plan.compilation_storage->reserve_private(
      storage.peak_bytes ? std::optional<std::size_t>{peak - resident} : std::nullopt,
      &budget_declined);
  if (!transient)
    return std::unexpected(std::move(transient).error());
  if (unknown)
    plan.compilation_storage->note_unknown();
  return CompilationReservation{std::move(*retained), std::move(*transient)};
}

Status initialize_placement_facts(const detail::PlanState &plan,
                                  const detail::GraphState &graph,
                                  CompileOptions options,
                                  PlacementFacts &facts) noexcept {
  constexpr std::uint32_t kNone = detail::kNoWorkspaceRequest;
  const std::size_t target_count = plan.targets.size();
  const std::size_t storage_count = plan.storage_domains.size();
  const std::size_t value_count = graph.values.size();
  const std::size_t node_count = graph.nodes.size();
  if (options.max_route_search_states == 0)
    return Status::InvalidArgument("graph route search budget is zero");
  if (value_count != 0 && storage_count > SIZE_MAX / value_count) {
    return Status::ResourceExhausted(
        "graph physical-instance matrix size overflows");
  }
  if (node_count != 0 && target_count > SIZE_MAX / node_count) {
    return Status::ResourceExhausted(
        "graph placement capability matrix size overflows");
  }

  auto &node_required = facts.node_required;
  auto &node_preferred = facts.node_preferred;
  auto &value_required = facts.value_required;
  auto &value_preferred = facts.value_preferred;
  auto &external_seen = facts.external_seen;
  if (!node_required.try_resize(node_count) ||
      !node_preferred.try_resize(node_count) ||
      !value_required.try_resize(value_count) ||
      !value_preferred.try_resize(value_count) ||
      !external_seen.try_resize(value_count)) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < node_count; ++index) {
    node_required[index] = kNone;
    node_preferred[index] = kNone;
  }
  for (std::size_t index = 0; index < value_count; ++index) {
    value_required[index] = kNone;
    value_preferred[index] = kNone;
    external_seen[index] = 0;
  }

  for (const NodePlacement placement : options.node_placements) {
    if (!detail::owns(graph, placement.node) || !placement.target.valid() ||
        placement.target.ordinal() >= target_count ||
        !valid_strength(placement.strength)) {
      return Status::InvalidArgument("graph node placement is invalid");
    }
    std::uint32_t &slot = placement.strength == PlacementStrength::kRequired
                              ? node_required[placement.node.ordinal()]
                              : node_preferred[placement.node.ordinal()];
    if (slot != kNone)
      return Status::InvalidArgument("graph node placement is duplicated");
    slot = placement.target.ordinal();
  }
  for (const ValuePlacement placement : options.value_placements) {
    if (!detail::owns(graph, placement.value) || !placement.storage.valid() ||
        placement.storage.ordinal() >= storage_count ||
        !valid_strength(placement.strength)) {
      return Status::InvalidArgument("graph value placement is invalid");
    }
    std::uint32_t &slot = placement.strength == PlacementStrength::kRequired
                              ? value_required[placement.value.ordinal()]
                              : value_preferred[placement.value.ordinal()];
    if (slot != kNone)
      return Status::InvalidArgument("graph value placement is duplicated");
    slot = placement.storage.ordinal();
  }
  for (const ExternalResourceContract contract : options.external_resources) {
    if (!detail::owns(graph, contract.value) || !contract.storage.valid() ||
        contract.storage.ordinal() >= storage_count) {
      return Status::InvalidArgument(
          "graph external resource contract is invalid");
    }
    const std::size_t value = contract.value.ordinal();
    if ((plan.binding_roles[value] &
         (detail::kExternalInput | detail::kExternalOutput |
          detail::kConstant)) == 0) {
      return Status::InvalidArgument(
          "external resource contract names an internal value");
    }
    if (external_seen[value] != 0)
      return Status::InvalidArgument(
          "graph external resource contract is duplicated");
    external_seen[value] = 1;
  }
  return Status::Ok();
}

Status prepare_placed_steps_attempt(
    detail::PlanState &plan, const Graph &published_graph,
    const detail::GraphState &graph, CompileEnvironment environment,
    CompileOptions options, const ProviderPolicy &provider_policy,
    RegionDiscovery &discovery, ConstantPreparation &constants,
    PlacementAttemptOutcome &outcome,
    detail::ConstantArtifactCache &packing) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  struct OperationPlacement final {
    const OperationCandidate *route = nullptr;
    std::uint32_t workspace_storage = detail::kNoWorkspaceRequest;
  };
  constexpr std::uint32_t kNone = detail::kNoWorkspaceRequest;
  const std::size_t target_count = plan.targets.size();
  const std::size_t storage_count = plan.storage_domains.size();
  const std::size_t value_count = graph.values.size();
  const std::size_t node_count = graph.nodes.size();
  const auto &node_required = discovery.placement.node_required;
  const auto &node_preferred = discovery.placement.node_preferred;
  const auto &value_required = discovery.placement.value_required;
  const auto &value_preferred = discovery.placement.value_preferred;
  const auto &external_seen = discovery.placement.external_seen;

  // Output locality can change with the next complete candidate. Restore the
  // original contracts without reparsing or retaining a second baseline array.
  std::fill(plan.external_storage.begin(), plan.external_storage.end(), 0U);
  for (const ExternalResourceContract contract : options.external_resources)
    plan.external_storage[contract.value.ordinal()] = contract.storage.ordinal();

  // A required placement on an external source supplies its locality when no
  // stronger exact resource contract was given.
  for (std::uint32_t value : graph.inputs) {
    if (external_seen[value] == 0) {
      if (value_required[value] != kNone)
        plan.external_storage[value] = value_required[value];
      else if (value_preferred[value] != kNone)
        plan.external_storage[value] = value_preferred[value];
    }
  }
  for (std::uint32_t value : graph.constants) {
    if (external_seen[value] == 0) {
      if (value_required[value] != kNone)
        plan.external_storage[value] = value_required[value];
      else if (value_preferred[value] != kNone)
        plan.external_storage[value] = value_preferred[value];
    }
  }

  // A borrowed native readonly import can have its own resource owner. Its
  // source must prove that family now; selected readers are checked again
  // after physical aliases exist. Materialized constants select their actual
  // destination later, before the cold copy batch is prepared.
  for (std::size_t index = 0; index < graph.constants.size(); ++index) {
    const bool materialize = !constants.bindings.empty() &&
        constants.bindings[index].policy == ConstantPolicy::kMaterialize;
    if (materialize)
      continue;
    const std::uint32_t value = graph.constants[index];
    const mem::MemoryResourceRef actual =
        plan.constant_bindings[index].storage().resource();
    const std::uint32_t storage = plan.external_storage[value];
    if (storage >= storage_count) {
      return Status::InvalidArgument(
          "constant binding does not match its external resource contract");
    }
    const auto nominated = plan.storage_domains[storage].resource;
    if (nominated != actual) {
      HYLE_RETURN_IF_ERROR(validate_configured_read_resource(plan, nominated, actual));
    }
  }

  auto &extensions = discovery.extensions;
  auto &all_region_candidates = discovery.candidates;
  auto &all_region_nodes = discovery.nodes;
  if (!discovery.ready) {
    HYLE_RETURN_IF_ERROR(discover_operation_candidates(
        plan, graph, environment, options, provider_policy, node_required,
        discovery.operations, plan.search));
    if (options.search_policy == CompileSearchPolicy::kRequireUnexhausted &&
        plan.search.operation_proposal_limit_reached)
      return Status::ResourceExhausted("graph operation discovery reached a "
                                       "search limit under strict policy");
    std::size_t region_proposal_count = 0;
    std::size_t region_proposal_node_count = 0;
    for (std::size_t target = 0; target < target_count; ++target) {
      Status discovered = discover_region_candidates(
          published_graph, graph, environment.targets[target].source, options,
          provider_policy, plan.allocator, extensions, all_region_candidates,
          all_region_nodes, region_proposal_count, region_proposal_node_count,
          plan.search, ExecutionTargetId{static_cast<std::uint32_t>(target)},
          node_required, detail::GraphAccess::state(plan.domain_graph),
          discovery.adapters, &discovery.operations);
      if (!discovered.ok())
        return discovered;
    }
    plan.search.region_proposals = region_proposal_count;
    plan.search.region_proposal_nodes = region_proposal_node_count;
    if (options.search_policy == CompileSearchPolicy::kRequireUnexhausted &&
        plan.search.optimization_limited()) {
      return Status::ResourceExhausted("graph candidate discovery reached a "
                                       "search limit under strict policy");
    }
    if (!all_region_nodes.try_resize(region_proposal_node_count) ||
        !discovery.rejected.try_resize(region_proposal_count))
      return plan_allocation_failure();
    std::fill(discovery.rejected.begin(), discovery.rejected.end(),
              std::uint8_t{0});
    HYLE_RETURN_IF_ERROR(discover_region_resources(
        published_graph, graph, plan, environment, options, node_required, discovery));
    if (!discovery.region_coverage.try_resize(node_count))
      return plan_allocation_failure();
    std::fill(discovery.region_coverage.begin(), discovery.region_coverage.end(),
              RegionCoverage::kNone);
    for (const RegionCandidate &candidate : all_region_candidates) {
      for (NodeId node : candidate_nodes(candidate, all_region_nodes))
        discovery.region_coverage[node.ordinal()] = RegionCoverage::kEligible;
    }
    for (const RegionCandidate &candidate : all_region_candidates) {
      if (discovery.rejected[candidate.proposal_order] != 0) {
        for (NodeId node : candidate_nodes(candidate, all_region_nodes))
          discovery.region_coverage[node.ordinal()] = RegionCoverage::kRejected;
      }
    }
    HYLE_RETURN_IF_ERROR(discovery.routes.initialize(
        node_count, target_count, discovery.operations, all_region_candidates,
        all_region_nodes, discovery.rejected, node_preferred));
    discovery.ready = true;
  }
  bool next_physical = false;
  if (discovery.has_route) {
    HYLE_ASSIGN_OR_RETURN(
        next_physical,
        discovery.physical.next(options.max_route_search_states, plan.search,
                                outcome.search_limit));
  }
  if (!next_physical) {
    HYLE_ASSIGN_OR_RETURN(discovery.has_route,
                          discovery.routes.next(options.max_route_search_states,
                                                plan.search,
                                                outcome.search_limit));
    if (!discovery.has_route) {
      outcome.no_feasible_route = true;
      const auto missing = discovery.routes.uncovered_node();
      if (missing != RouteSearch::kNoRoute) {
        const bool limited = plan.search.operation_proposal_limit_reached ||
                             plan.search.region_proposal_limit_reached ||
                             plan.search.per_provider_proposal_limit_reached ||
                             plan.search.region_node_limit_reached;
        Status status = limited
                            ? Status::ResourceExhausted(
                                  "candidate discovery limit left a graph node uncovered")
                            : Status::Unimplemented(
                                  "no available operation or region implementation covers this node");
        if (plan.search.operation_proposal_limit_reached)
          status = std::move(status).WithContext("operation_proposal_limit");
        if (plan.search.region_proposal_limit_reached)
          status = std::move(status).WithContext("region_proposal_limit");
        if (plan.search.per_provider_proposal_limit_reached)
          status = std::move(status).WithContext("per_provider_proposal_limit");
        if (plan.search.region_node_limit_reached)
          status = std::move(status).WithContext("region_node_limit");
        std::array<char, 32> location{'g', 'r', 'a', 'p', 'h', ' ', 'n', 'o', 'd', 'e', ' '};
        const auto number = std::to_chars(location.data() + 11,
                                           location.data() + location.size(), missing);
        HYLE_FATAL_ASSERT(number.ec == std::errc{});
        const auto &node = graph.nodes[missing];
        const auto name = node.view_index()
                              ? compute::operation_name(node.kind())
                              : compute::operation_name(node.operation());
        return std::move(status).WithContext(name).WithContext(
            std::string_view(location.data(), number.ptr - location.data()));
      }
      return Status::Unimplemented(
          "no complete graph route satisfies the available candidates");
    }
    discovery.physical.reset();
  }

  Ordinals producer_storage(plan.allocator);
  Ordinals node_targets(plan.allocator);
  Ordinals node_workspace_storage(plan.allocator);
  Ordinals selected_edge_storage(plan.allocator);
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> preparation_prefix(
      plan.allocator);
  container::SmallVector<const OperationCandidate *, 8, mem::AllocatorRef>
      operation_routes(plan.allocator);
  if (!producer_storage.try_resize(value_count) ||
      !node_targets.try_resize(node_count) ||
      !node_workspace_storage.try_resize(node_count) ||
      !preparation_prefix.try_resize(node_count) ||
      !selected_edge_storage.try_resize(graph.edges.size()) ||
      !operation_routes.try_resize(node_count)) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < value_count; ++index)
    producer_storage[index] = kNone;
  for (std::uint32_t &storage : node_workspace_storage)
    storage = kNone;
  for (std::uint32_t &storage : selected_edge_storage)
    storage = kNone;
  for (std::uint32_t value : graph.inputs)
    producer_storage[value] = plan.external_storage[value];
  for (std::uint32_t value : graph.constants)
    producer_storage[value] = plan.external_storage[value];

  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  using Resources =
      container::SmallVector<mem::MemoryResourceRef, 4, mem::AllocatorRef>;
  Descriptions inputs(plan.allocator);
  Descriptions outputs(plan.allocator);
  Resources input_resources(plan.allocator);
  Resources output_resources(plan.allocator);
  Ordinals selected_inputs(plan.allocator);
  Ordinals selected_outputs(plan.allocator);
  Ordinals resource_choices(plan.allocator);
  std::size_t &placement_attempts = plan.search.placement_attempts;

  auto select_storage =
      [&](std::uint32_t target,
          const compute::detail::ResourceBindingRequirement &requirement,
          std::uint32_t required, std::uint32_t first,
          std::uint32_t second) noexcept -> Result<std::uint32_t> {
    auto compatible = [&](std::uint32_t storage) noexcept -> Result<bool> {
      if (storage >= storage_count)
        return false;
      const mem::MemoryResourceRef resource =
          plan.storage_domains[storage].resource;
      Status matched =
          compute::detail::OperationPreparationAccess::match_resource(
              requirement, resource);
      if (!matched.ok()) {
        if (!matched.IsUnimplemented())
          return std::unexpected(std::move(matched));
        return false;
      }
      Status accessible =
          plan.targets[target].source.validate_resource(resource);
      if (!accessible.ok()) {
        if (!accessible.IsFailedPrecondition() && !accessible.IsUnimplemented())
          return std::unexpected(std::move(accessible));
        return false;
      }
      return true;
    };

    if (required != kNone) {
      auto accepted = compatible(required);
      if (!accepted)
        return std::unexpected(std::move(accepted).error());
      if (*accepted)
        return required;
      return std::unexpected(Status::Unimplemented(
          "required graph storage does not satisfy provider constraints"));
    }
    resource_choices.clear();
    auto offer = [&](std::uint32_t storage) noexcept -> Status {
      HYLE_ASSIGN_OR_RETURN(const bool accepted, compatible(storage));
      if (accepted && !resource_choices.try_push_back(storage))
        return plan_allocation_failure();
      return Status::Ok();
    };
    HYLE_RETURN_IF_ERROR(offer(first));
    if (second != first)
      HYLE_RETURN_IF_ERROR(offer(second));
    for (std::uint32_t storage = 0; storage < storage_count; ++storage) {
      if (storage == first || storage == second)
        continue;
      HYLE_RETURN_IF_ERROR(offer(storage));
    }
    if (resource_choices.empty())
      return std::unexpected(Status::Unimplemented(
          "no graph storage satisfies provider resource constraints"));
    return discovery.physical.choose(resource_choices,
                                     options.max_route_search_states,
                                     plan.search, outcome.search_limit);
  };

  auto select_operation_resources =
      [&](std::size_t node_index, std::uint32_t candidate,
          const OperationCandidate &route) noexcept
      -> Result<OperationPlacement> {
    const detail::NodeRecord &node = graph.nodes[node_index];
    const std::uint32_t default_storage =
        plan.targets[candidate].default_storage.ordinal();
    if (!selected_inputs.try_resize(node.input_count) ||
        !selected_outputs.try_resize(node.output_count)) {
      return std::unexpected(plan_allocation_failure());
    }

    if (route.metadata_view) {
      if (node.input_count != 1 || node.output_count != 1)
        return std::unexpected(Status::Internal("View arity is invalid"));
      const auto required = value_required[node.output_begin];
      const auto storage =
          required != kNone ? required
                            : producer_storage[graph.edges[node.input_begin]];
      if (storage >= storage_count)
        return std::unexpected(Status::Internal("View source has no storage"));
      // Merely describing an alias does not require host mapping or kernel
      // access by this target. Any actual relocation is a checked Copy below.
      selected_inputs[0] = storage;
      selected_outputs[0] = storage;
      return OperationPlacement{&route, kNone};
    }

    const auto requirements =
        std::span<const compute::ResourceRequirement>(
            discovery.operations.requirements)
            .subspan(route.requirements_begin,
                     static_cast<std::size_t>(node.input_count) +
                         node.output_count);
    const auto leaf_inputs = requirements.first(node.input_count);
    const auto leaf_outputs = requirements.subspan(node.input_count);
    const auto traits = node.traits();
    const bool must_alias_input =
        traits.alias == compute::AliasPolicy::kMustAliasInput0;
    if (must_alias_input && node.input_count == 0)
      return std::unexpected(Status::Internal(
          "operation requires an input alias but has no inputs"));
    std::uint32_t alias_storage = kNone;
    if (must_alias_input) {
      for (std::size_t index = 0; index < node.output_count; ++index) {
        const auto required = value_required[node.output_begin + index];
        if (required == kNone)
          continue;
        if (alias_storage != kNone && alias_storage != required)
          return std::unexpected(Status::Unimplemented(
              "required output storages contradict operation aliasing"));
        alias_storage = required;
      }
    }
    for (std::size_t index = 0; index < node.input_count; ++index) {
      const std::uint32_t value = graph.edges[node.input_begin + index];
      HYLE_ASSIGN_OR_RETURN(selected_inputs[index],
                            select_storage(candidate, leaf_inputs[index],
                                           index == 0 ? alias_storage : kNone,
                                           producer_storage[value],
                                           default_storage));
    }
    for (std::size_t index = 0; index < node.output_count; ++index) {
      const auto value = node.output_begin + index;
      if (must_alias_input && value_required[value] != kNone &&
          value_required[value] != selected_inputs[0])
        return std::unexpected(Status::Unimplemented(
            "required output storage contradicts operation aliasing"));
      const std::uint32_t required_storage =
          must_alias_input ? selected_inputs[0] : value_required[value];
      const std::uint32_t preferred_storage = value_preferred[value] != kNone
                                                  ? value_preferred[value]
                                                  : default_storage;
      HYLE_ASSIGN_OR_RETURN(selected_outputs[index],
                            select_storage(candidate, leaf_outputs[index],
                                           required_storage, preferred_storage,
                                           default_storage));
    }
    std::uint32_t workspace_storage = kNone;
    if (route.proposal.runtime.workspace_owner ==
        compute::ProviderWorkspaceOwner::kCaller) {
      HYLE_ASSIGN_OR_RETURN(
          workspace_storage,
          select_storage(
              candidate, route.proposal.workspace_resource, kNone,
              default_storage,
              node.output_count != 0
                  ? selected_outputs[0]
                  : (node.input_count != 0 ? selected_inputs[0] : kNone)));
    }
    return OperationPlacement{&route, workspace_storage};
  };

  auto publish_operation_placement =
      [&](std::size_t node_index, OperationPlacement placement) noexcept {
        const auto &node = graph.nodes[node_index];
        for (std::size_t index = 0; index < node.input_count; ++index)
          selected_edge_storage[node.input_begin + index] =
              selected_inputs[index];
        for (std::size_t index = 0; index < node.output_count; ++index)
          producer_storage[node.output_begin + index] = selected_outputs[index];
        node_workspace_storage[node_index] = placement.workspace_storage;
        operation_routes[node_index] = placement.route;
      };

  RegionCandidates selected_regions(plan.allocator);
  for (const auto &frame : discovery.routes.frames()) {
    const auto &route = discovery.routes.route(frame.chosen);
    const auto default_storage =
        plan.targets[route.target].default_storage.ordinal();
    discovery.routes.for_each_node(
        route, [&](std::uint32_t node_index) noexcept {
          node_targets[node_index] = route.target;
          const auto &node = graph.nodes[node_index];
          for (std::size_t index = 0; index < node.output_count; ++index) {
            const auto value = node.output_begin + index;
            producer_storage[value] =
                value_required[value] != kNone    ? value_required[value]
                : value_preferred[value] != kNone ? value_preferred[value]
                                                  : default_storage;
          }
        });
    if (route.region) {
      if (!selected_regions.try_push_back(
              all_region_candidates[route.proposal]))
        return plan_allocation_failure();
    } else {
      operation_routes[route.first] =
          &discovery.operations.candidates[route.proposal];
    }
  }

  const auto &region_coverage = discovery.region_coverage;

  // With no explicit output storage, keep a published node result where its
  // producer already resides. This avoids manufacturing a boundary copy just
  // because storage zero appeared first in the environment.
  for (std::uint32_t value : graph.outputs) {
    if (external_seen[value] == 0 &&
        graph.values[value].producer != detail::kNoNode) {
      plan.external_storage[value] = producer_storage[value];
    }
  }

  const std::size_t matrix_instances = value_count * storage_count;
  // One original generated carrier may coexist with its concrete caller
  // replica in the same storage domain. The matrix caches the latter.
  if (value_count > SIZE_MAX - matrix_instances)
    return Status::ResourceExhausted("graph carrier-instance capacity overflows");
  const std::size_t maximum_instances = matrix_instances + value_count;
  if (node_count > SIZE_MAX - maximum_instances) {
    return Status::ResourceExhausted("graph plan capacity overflows");
  }
  Ordinals instances(plan.allocator);
  Ordinals producer_instances(plan.allocator);
  if (!instances.try_resize(matrix_instances) ||
      !producer_instances.try_resize(value_count)) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &instance : instances)
    instance = kNone;
  for (std::uint32_t &instance : producer_instances)
    instance = kNone;

  plan.steps.clear();
  plan.artifact_charges.clear();
  plan.node_inputs.clear();
  plan.node_outputs.clear();
  plan.step_nodes.clear();
  plan.instance_descriptions.clear();
  plan.instance_values.clear();
  plan.instance_storage.clear();
  plan.instance_output_contracts.clear();
  plan.instance_producer_targets.clear();
  plan.instance_constant_owners.clear();
  plan.instance_alignments.clear();
  plan.instance_roles.clear();
  plan.workspace_requests.clear();
  plan.value_materialized.clear();
  plan.alias_sources.clear();
  plan.alias_offsets.clear();
  for (std::size_t index = 0; index < plan.external_instances.size(); ++index)
    plan.external_instances[index] = kNone;
  if (graph.edges.size() > (SIZE_MAX - graph.outputs.size()) / std::size_t{2}) {
    return Status::ResourceExhausted("graph physical edge capacity overflows");
  }
  const std::size_t edge_capacity =
      graph.edges.size() * 2 + graph.outputs.size();
  if (!plan.steps.try_reserve(node_count + maximum_instances) ||
      !plan.node_inputs.try_reserve(edge_capacity) ||
      !plan.node_outputs.try_reserve(node_count + maximum_instances) ||
      !plan.step_nodes.try_reserve(node_count) ||
      !plan.instance_descriptions.try_reserve(maximum_instances) ||
      !plan.instance_values.try_reserve(maximum_instances) ||
      !plan.instance_storage.try_reserve(maximum_instances) ||
      !plan.instance_output_contracts.try_reserve(maximum_instances) ||
      !plan.instance_producer_targets.try_reserve(maximum_instances) ||
      !plan.instance_constant_owners.try_reserve(maximum_instances) ||
      !plan.instance_alignments.try_reserve(maximum_instances) ||
      !plan.instance_roles.try_reserve(maximum_instances) ||
      !plan.workspace_requests.try_reserve(maximum_instances) ||
      !plan.value_materialized.try_reserve(maximum_instances) ||
      !plan.alias_sources.try_reserve(maximum_instances) ||
      !plan.alias_offsets.try_reserve(maximum_instances)) {
    return plan_allocation_failure();
  }

  auto add_instance =
      [&](std::uint32_t value, std::uint32_t storage,
          std::uint8_t roles) noexcept -> Result<std::uint32_t> {
    if (plan.instance_values.size() >= kNone)
      return std::unexpected(Status::ResourceExhausted(
          "graph physical-instance identity space is exhausted"));
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(plan.instance_values.size());
    if (!plan.instance_descriptions.try_push_back(plan.descriptions[value]) ||
        !plan.instance_values.try_push_back(value) ||
        !plan.instance_storage.try_push_back(storage) ||
        !plan.instance_output_contracts.try_push_back(RegionOutputContract{}) ||
        !plan.instance_producer_targets.try_push_back(kNone) ||
        !plan.instance_constant_owners.try_push_back(kNone) ||
        !plan.instance_alignments.try_push_back(1) ||
        !plan.instance_roles.try_push_back(roles) ||
        !plan.workspace_requests.try_push_back(kNone) ||
        !plan.value_materialized.try_push_back(0) ||
        !plan.alias_sources.try_push_back(detail::kNoInPlaceSource) ||
        !plan.alias_offsets.try_push_back(0)) {
      return std::unexpected(plan_allocation_failure());
    }
    instances[static_cast<std::size_t>(value) * storage_count + storage] =
        ordinal;
    return ordinal;
  };

  auto append_copy_step = [&](std::uint32_t source, std::uint32_t destination,
                              std::uint32_t target) noexcept -> Status {
    const std::uint32_t input_begin =
        static_cast<std::uint32_t>(plan.node_inputs.size());
    const std::uint32_t output_begin =
        static_cast<std::uint32_t>(plan.node_outputs.size());
    if (!plan.node_inputs.try_push_back(source) ||
        !plan.node_outputs.try_push_back(destination)) {
      return plan_allocation_failure();
    }
    auto bytes = plan.instance_descriptions[source].required_bytes();
    if (!bytes)
      return std::move(bytes).error();
    const std::uint32_t source_storage = plan.instance_storage[source];
    const std::uint32_t destination_storage =
        plan.instance_storage[destination];
    if (!plan.steps.try_push_back(detail::PlanStep::copy(
            input_begin, output_begin, *bytes, StorageDomainId{source_storage},
            StorageDomainId{destination_storage}, ExecutionTargetId{target})))
      return plan_allocation_failure();
    plan.value_materialized[destination] = 1;
    plan.instance_output_contracts[destination] = {};
    plan.instance_producer_targets[destination] = target;
    return Status::Ok();
  };

  auto copy_source = [&](std::uint32_t instance) noexcept {
    const auto producer = plan.instance_producer_targets[instance];
    return detail::CopySourceDescription{
        .description = plan.instance_descriptions[instance],
        .resource = plan.storage_domains[plan.instance_storage[instance]].resource,
        .value = plan.instance_output_contracts[instance],
        .producer = producer < plan.targets.size() ? plan.targets[producer].source : ExecutionSource{}};
  };

  auto supports_copy = [&](std::uint32_t target, const detail::CopySourceDescription &source,
                           std::uint32_t destination) noexcept -> Result<bool> {
    // Shared Queue and native transfer capabilities each prove their own
    // actual resources. Accessibility alone grants neither execution path.
    auto retained = detail::ExecutionSourceAccess::compute(
        plan.targets[target].source);
    if (!retained) {
      if (retained.error().IsUnimplemented())
        return detail::ExecutionSourceAccess::can_copy(
            plan.targets[target].source, source,
            plan.storage_domains[destination].resource);
      return std::unexpected(std::move(retained).error());
    }
    if (source.value.projection != RegionOutputProjection::kTensorView ||
        source.value.kind == RegionOutputKind::kDeferredGenerated)
      return false;
    for (const auto resource : {source.resource, plan.storage_domains[destination].resource}) {
      auto status = plan.targets[target].source.validate_resource(
          resource);
      if (status.ok())
        continue;
      if (status.IsFailedPrecondition() || status.IsUnimplemented())
        return false;
      return std::unexpected(std::move(status));
    }
    return true;
  };

  auto copy_targets = [&](const detail::CopySourceDescription &source,
                          std::uint32_t destination) noexcept -> Status {
    resource_choices.clear();
    for (unsigned preference = 0; preference != 3; ++preference) {
      for (std::size_t index = 0; index < target_count; ++index) {
        const auto storage = plan.targets[index].default_storage.ordinal();
        const unsigned rank = storage == destination ? 0
                              : plan.storage_domains[storage].resource == source.resource ? 1
                                                     : 2;
        if (rank != preference)
          continue;
        HYLE_ASSIGN_OR_RETURN(
            const bool supports,
            supports_copy(static_cast<std::uint32_t>(index), source, destination));
        if (supports &&
            !resource_choices.try_push_back(static_cast<std::uint32_t>(index)))
          return plan_allocation_failure();
      }
    }
    return Status::Ok();
  };

  auto select_copy_target = [&]() noexcept -> Result<std::uint32_t> {
    return discovery.physical.choose(resource_choices,
                                     options.max_route_search_states,
                                     plan.search, outcome.search_limit);
  };

  auto append_copy = [&](std::uint32_t source,
                         std::uint32_t destination) noexcept -> Status {
    const auto source_storage = plan.instance_storage[source];
    const auto destination_storage = plan.instance_storage[destination];
    const auto description = copy_source(source);
    HYLE_RETURN_IF_ERROR(copy_targets(description, destination_storage));
    if (!resource_choices.empty()) {
      HYLE_ASSIGN_OR_RETURN(const auto target, select_copy_target());
      return append_copy_step(source, destination, target);
    }

    // Search the configured host-visible intermediates using both actual
    // transfer capabilities. No backend name or Device-count gate is needed.
    for (std::uint32_t storage = 0; storage < storage_count; ++storage) {
      if (storage == source_storage || storage == destination_storage ||
          !plan.storage_domains[storage].resource.properties().has(
              mem::MemoryProperty::kHostAccessible))
        continue;
      bool from = false, to = false;
      const detail::CopySourceDescription intermediate{
          .description = description.description,
          .resource = plan.storage_domains[storage].resource};
      for (std::uint32_t target = 0; target < target_count; ++target) {
        HYLE_ASSIGN_OR_RETURN(const bool accepts_source,
                              supports_copy(target, description, storage));
        HYLE_ASSIGN_OR_RETURN(const bool accepts_destination,
                              supports_copy(target, intermediate, destination_storage));
        from |= accepts_source;
        to |= accepts_destination;
      }
      if (from && to && !resource_choices.try_push_back(storage))
        return plan_allocation_failure();
    }
    if (resource_choices.empty()) {
      outcome.rejected_plan = true;
      return Status::Unimplemented("storage boundary has no configured transfer bridge");
    }
    HYLE_ASSIGN_OR_RETURN(const auto storage, select_copy_target());
    const auto value = plan.instance_values[source];
    auto staged = instances[static_cast<std::size_t>(value) * storage_count + storage];
    if (staged == kNone) {
      HYLE_ASSIGN_OR_RETURN(staged, add_instance(value, storage, 0));
    }
    if (!plan.value_materialized[staged]) {
      HYLE_RETURN_IF_ERROR(copy_targets(description, storage));
      HYLE_ASSIGN_OR_RETURN(const auto from, select_copy_target());
      HYLE_RETURN_IF_ERROR(append_copy_step(source, staged, from));
    }
    HYLE_RETURN_IF_ERROR(copy_targets(copy_source(staged), destination_storage));
    HYLE_ASSIGN_OR_RETURN(const auto to, select_copy_target());
    return append_copy_step(staged, destination, to);
  };

  auto ensure_local_instance =
      [&](auto &&self, std::uint32_t value,
          std::uint32_t storage, bool needs_tensor) noexcept -> Result<std::uint32_t> {
    const std::uint32_t source = producer_instances[value];
    if (source == kNone) {
      return std::unexpected(
          Status::Internal("graph value has no physical producer"));
    }
    // A reference-capable consumer keeps the original producer even when a
    // concrete replica has subsequently occupied the same matrix entry.
    if (!needs_tensor && plan.instance_storage[source] == storage)
      return source;
    std::uint32_t instance =
        instances[static_cast<std::size_t>(value) * storage_count + storage];
    if (instance != kNone) {
      const auto contract = plan.instance_output_contracts[instance];
      if (!needs_tensor || (contract.projection == RegionOutputProjection::kTensorView &&
                            contract.kind != RegionOutputKind::kDeferredGenerated))
        return instance;
    }
    const bool immutable = (plan.instance_roles[source] & detail::kConstant) != 0;
    const auto parent = plan.alias_sources[source];
    if (immutable && parent != detail::kNoInPlaceSource) {
      HYLE_ASSIGN_OR_RETURN(const auto local,
          self(self, plan.instance_values[parent], storage, needs_tensor));
      HYLE_ASSIGN_OR_RETURN(const auto created, add_instance(value, storage, detail::kConstant));
      plan.alias_sources[created] = local;
      plan.alias_offsets[created] = plan.alias_offsets[source];
      plan.instance_output_contracts[created] = plan.instance_output_contracts[local];
      plan.instance_producer_targets[created] = plan.instance_producer_targets[local];
      return created;
    }
    // Only logical immutable roots can acquire canonical replicas here.
    // Provider-prepared outputs keep their original artifact provenance.
    const bool canonical = immutable && constant_slot(plan, value) != SIZE_MAX;
    auto created = add_instance(value, storage, canonical ? detail::kConstant : 0);
    if (!created)
      return std::unexpected(std::move(created).error());
    if (canonical)
      return *created;
    Status copied = append_copy(source, *created);
    if (!copied.ok())
      return std::unexpected(std::move(copied));
    return *created;
  };

  auto needs_tensor = [&](std::uint32_t value, std::uint32_t target,
                          bool accepts_reference) noexcept {
    const auto instance = producer_instances[value];
    if (!accepts_reference || instance == kNone) return true;
    const auto producer = plan.instance_producer_targets[instance];
    return producer >= plan.targets.size() ||
        !plan.targets[producer].source.same_owner(plan.targets[target].source);
  };

  for (std::uint32_t value : graph.inputs) {
    auto instance = add_instance(value, plan.external_storage[value],
                                 plan.binding_roles[value]);
    if (!instance)
      return std::move(instance).error();
    producer_instances[value] = *instance;
    plan.external_instances[value] = *instance;
  }
  for (std::uint32_t value : graph.constants) {
    auto instance = add_instance(value, plan.external_storage[value],
                                 plan.binding_roles[value]);
    if (!instance)
      return std::move(instance).error();
    producer_instances[value] = *instance;
    plan.external_instances[value] = *instance;
  }

  Ordinals contracted_owners(plan.allocator);
  Ordinals region_by_node(plan.allocator);
  if (!contracted_owners.try_resize(node_count) ||
      !region_by_node.try_resize(node_count)) {
    return plan_allocation_failure();
  }
  for (std::uint32_t node = 0; node < node_count; ++node) {
    contracted_owners[node] = node;
    region_by_node[node] = kNone;
  }
  for (std::size_t index = 0; index < selected_regions.size(); ++index) {
    const auto nodes =
        candidate_nodes(selected_regions[index], all_region_nodes);
    for (NodeId node : nodes) {
      if (region_by_node[node.ordinal()] != kNone)
        return Status::Internal("selected graph regions overlap");
      region_by_node[node.ordinal()] = static_cast<std::uint32_t>(index);
      contracted_owners[node.ordinal()] = nodes.front().ordinal();
    }
  }
  RegionContraction contraction(plan.allocator);
  Ordinals interval_order(plan.allocator);
  std::span<const std::uint32_t> unit_order;
  const bool intervals_only =
      std::all_of(selected_regions.begin(), selected_regions.end(),
                  [&](const RegionCandidate &candidate) noexcept {
                    return nodes_are_contiguous(
                        candidate_nodes(candidate, all_region_nodes));
                  });
  if (intervals_only) {
    // The original topological order is already valid after contracting
    // disjoint intervals. Native-only and contiguous-provider plans need no
    // second adjacency table or topological sort.
    if (!interval_order.try_reserve(node_count))
      return plan_allocation_failure();
    for (std::uint32_t node = 0; node < node_count; ++node) {
      if (contracted_owners[node] == node &&
          !interval_order.try_push_back(node))
        return plan_allocation_failure();
    }
    unit_order = interval_order;
  } else {
    Status initialized = contraction.initialize(graph);
    if (!initialized.ok())
      return initialized;
    auto ordered = contraction.sort(contracted_owners);
    if (!ordered)
      return std::move(ordered).error();
    if (!*ordered) {
      outcome.rejected_plan = true;
      return Status::Unimplemented(
          "candidate graph regions form a cyclic contraction");
    }
    unit_order = contraction.order();
  }

  Ordinals physical_inputs(plan.allocator);
  for (std::uint32_t unit_node : unit_order) {
    if (region_by_node[unit_node] != kNone) {
      const RegionCandidate &candidate =
          selected_regions[region_by_node[unit_node]];
      const std::span<const NodeId> region_nodes =
          candidate_nodes(candidate, all_region_nodes);
      const std::uint32_t target = candidate.target.ordinal();
      const std::uint32_t default_storage =
          plan.targets[target].default_storage.ordinal();
      const auto &skeleton = discovery.resources[candidate.proposal_order];
      const auto values =
          std::span<const ValueId>(discovery.values)
              .subspan(skeleton.port_begin,
                       skeleton.input_count + skeleton.output_count);
      const auto logical_inputs = values.first(skeleton.input_count);
      const auto logical_outputs = values.subspan(skeleton.input_count);
      const auto requirements =
          std::span<const compute::ResourceRequirement>(discovery.requirements)
              .subspan(skeleton.port_begin, values.size());
      const auto input_requirements = requirements.first(skeleton.input_count);
      const auto input_references = std::span<const std::uint8_t>(discovery.input_references)
          .subspan(skeleton.input_reference_begin, skeleton.input_count);
      const auto output_requirements =
          requirements.subspan(skeleton.input_count);
      const auto output_contracts = std::span<const RegionOutputContract>(discovery.output_contracts)
          .subspan(skeleton.output_contract_begin, skeleton.output_count);
      if (!selected_inputs.try_resize(logical_inputs.size()) ||
          !selected_outputs.try_resize(logical_outputs.size())) {
        return plan_allocation_failure();
      }
      const auto &resource_proposal = skeleton.proposal;
      Status region_status = Status::Ok();
      bool route_supported = true;

      for (std::size_t index = 0;
           route_supported && index < logical_inputs.size(); ++index) {
        const std::uint32_t value = logical_inputs[index].ordinal();
        auto storage = select_storage(target, input_requirements[index], kNone,
                                      producer_storage[value], default_storage);
        if (!storage) {
          region_status = std::move(storage).error();
          if (!region_status.IsUnimplemented())
            return region_status;
          route_supported = false;
          break;
        }
        selected_inputs[index] = *storage;
      }
      for (std::size_t index = 0;
           route_supported && index < logical_outputs.size(); ++index) {
        const std::uint32_t value = logical_outputs[index].ordinal();
        const std::uint32_t preferred = value_preferred[value] != kNone
                                            ? value_preferred[value]
                                            : default_storage;
        auto storage =
            select_storage(target, output_requirements[index],
                           value_required[value], preferred, default_storage);
        if (!storage) {
          region_status = std::move(storage).error();
          if (!region_status.IsUnimplemented())
            return region_status;
          route_supported = false;
          break;
        }
        selected_outputs[index] = *storage;
      }
      std::uint32_t workspace_storage = kNone;
      if (route_supported && resource_proposal.runtime.workspace_owner ==
                                 compute::ProviderWorkspaceOwner::kCaller) {
        const std::uint32_t adjacent_storage =
            !selected_outputs.empty()
                ? selected_outputs[0]
                : (!selected_inputs.empty() ? selected_inputs[0] : kNone);
        auto storage =
            select_storage(target, resource_proposal.workspace_resource, kNone,
                           default_storage, adjacent_storage);
        if (!storage) {
          region_status = std::move(storage).error();
          if (!region_status.IsUnimplemented())
            return region_status;
          route_supported = false;
        } else {
          workspace_storage = *storage;
        }
      }

      if (!route_supported) {
        // This is an explicit cold rejection, not an arbitrary error code.
        // Discard the construction before reconsidering its partition/target.
        // Resource matcher's actual errors returned above never reach here.
        outcome.rejected_region = candidate.proposal_order;
        return region_status;
      }
      preparation_prefix[region_nodes.front().ordinal()] =
          discovery.physical.prefix();
      physical_inputs.clear();
      if (!physical_inputs.try_reserve(logical_inputs.size()))
        return plan_allocation_failure();
      for (std::size_t index = 0; index < logical_inputs.size(); ++index) {
        auto instance = ensure_local_instance(ensure_local_instance, logical_inputs[index].ordinal(),
                                              selected_inputs[index],
                                              needs_tensor(logical_inputs[index].ordinal(), target,
                                                  input_references[index] != 0));
        if (!instance)
          return std::move(instance).error();
        HYLE_ASSIGN_OR_RETURN(const bool aligned,
            accepts_input_alias_alignment(plan, *instance,
                input_requirements[index].minimum_alignment));
        if (!aligned) {
          discovery.physical.reject_suffix(
              preparation_prefix[region_nodes.front().ordinal()]);
          outcome.rejected_region = candidate.proposal_order;
          return Status::Unimplemented(
              "graph input alias offset cannot satisfy region port alignment");
        }
        plan.instance_alignments[*instance] =
            (std::max)(plan.instance_alignments[*instance],
                       input_requirements[index].minimum_alignment);
        if (!physical_inputs.try_push_back(*instance))
          return plan_allocation_failure();
      }
      const std::uint32_t input_begin =
          static_cast<std::uint32_t>(plan.node_inputs.size());
      for (std::uint32_t instance : physical_inputs) {
        if (!plan.node_inputs.try_push_back(instance))
          return plan_allocation_failure();
      }
      const std::uint32_t output_begin =
          static_cast<std::uint32_t>(plan.node_outputs.size());
      for (std::size_t index = 0; index < logical_outputs.size(); ++index) {
        const std::uint32_t value = logical_outputs[index].ordinal();
        const std::uint32_t storage = selected_outputs[index];
        if ((plan.binding_roles[value] & detail::kExternalOutput) != 0 &&
            external_seen[value] == 0) {
          plan.external_storage[value] = storage;
        }
        std::uint8_t roles = 0;
        if ((plan.binding_roles[value] & detail::kExternalOutput) != 0 &&
            plan.external_storage[value] == storage) {
          roles |= detail::kExternalOutput;
        }
        auto instance = add_instance(value, storage, roles);
        if (!instance)
          return std::move(instance).error();
        plan.instance_alignments[*instance] =
            (std::max)(plan.instance_alignments[*instance],
                       output_requirements[index].minimum_alignment);
        plan.instance_output_contracts[*instance] = output_contracts[index];
        plan.instance_producer_targets[*instance] = target;
        producer_instances[value] = *instance;
        producer_storage[value] = storage;
        if ((roles & detail::kExternalOutput) != 0)
          plan.external_instances[value] = *instance;
        if (!plan.node_outputs.try_push_back(*instance))
          return plan_allocation_failure();
        plan.value_materialized[*instance] = 1;
      }
      const std::uint32_t step_index =
          static_cast<std::uint32_t>(plan.steps.size());
      const std::uint32_t step_node_begin =
          static_cast<std::uint32_t>(plan.step_nodes.size());
      for (NodeId node : region_nodes) {
        if (!plan.step_nodes.try_push_back(node.ordinal()))
          return plan_allocation_failure();
      }
      if (!plan.steps.try_emplace_back(
              detail::PreparedRegion{}, region_nodes.front().ordinal(),
              step_node_begin, candidate.node_count,
              nodes_are_contiguous(region_nodes), input_begin,
              static_cast<std::uint32_t>(logical_inputs.size()), output_begin,
              static_cast<std::uint32_t>(logical_outputs.size()),
              ExecutionTargetId{target},
              workspace_storage == kNone
                  ? StorageDomainId{}
                  : StorageDomainId{workspace_storage})) {
        return plan_allocation_failure();
      }
      for (NodeId node : region_nodes)
        plan.node_steps[node.ordinal()] = step_index;
      ++plan.region_count;
      continue;
    }

    const std::uint32_t node_index = unit_node;
    const detail::NodeRecord &node = graph.nodes[node_index];
    const std::uint32_t target = node_targets[node_index];
    auto placement = select_operation_resources(node_index, target,
                                                *operation_routes[node_index]);
    if (!placement) {
      Status status = std::move(placement).error();
      outcome.rejected_plan = status.IsUnimplemented();
      return status;
    }
    publish_operation_placement(node_index, *placement);
    preparation_prefix[node_index] = discovery.physical.prefix();
    const std::uint32_t workspace_storage = node_workspace_storage[node_index];
    std::span<const compute::ResourceRequirement> port_requirements;
    if (!placement->route->metadata_view) {
      port_requirements =
          std::span<const compute::ResourceRequirement>(
              discovery.operations.requirements)
              .subspan(placement->route->requirements_begin,
                       static_cast<std::size_t>(node.input_count) +
                           node.output_count);
    }
    physical_inputs.clear();
    if (!physical_inputs.try_reserve(node.input_count))
      return plan_allocation_failure();
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t value = graph.edges[node.input_begin + input_index];
      const std::uint32_t input_storage =
          selected_edge_storage[node.input_begin + input_index];
      if (input_storage == kNone)
        return Status::Internal("prepared graph input has no selected storage");
      auto instance = ensure_local_instance(ensure_local_instance, value, input_storage, true);
      if (!instance)
        return std::move(instance).error();
      if (!port_requirements.empty()) {
        HYLE_ASSIGN_OR_RETURN(const bool aligned,
            accepts_input_alias_alignment(plan, *instance,
                port_requirements[input_index].minimum_alignment));
        if (!aligned) {
          discovery.physical.reject_suffix(preparation_prefix[node_index]);
          outcome.rejected_operation = static_cast<std::size_t>(
              placement->route - discovery.operations.candidates.data());
          return Status::Unimplemented(
              "graph input alias offset cannot satisfy operation port alignment");
        }
      }
      if (!port_requirements.empty())
        plan.instance_alignments[*instance] =
            (std::max)(plan.instance_alignments[*instance],
                       port_requirements[input_index].minimum_alignment);
      if (!physical_inputs.try_push_back(*instance))
        return plan_allocation_failure();
    }
    const std::uint32_t input_begin =
        static_cast<std::uint32_t>(plan.node_inputs.size());
    for (std::uint32_t instance : physical_inputs) {
      if (!plan.node_inputs.try_push_back(instance))
        return plan_allocation_failure();
    }
    const std::uint32_t output_begin =
        static_cast<std::uint32_t>(plan.node_outputs.size());
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t value =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      const std::uint32_t output_storage = producer_storage[value];
      if ((plan.binding_roles[value] & detail::kExternalOutput) != 0 &&
          external_seen[value] == 0) {
        plan.external_storage[value] = output_storage;
      }
      std::uint8_t roles = 0;
      if ((plan.binding_roles[value] & detail::kExternalOutput) != 0 &&
          plan.external_storage[value] == output_storage) {
        roles |= detail::kExternalOutput;
      }
      auto instance = add_instance(value, output_storage, roles);
      if (!instance)
        return std::move(instance).error();
      if (!port_requirements.empty())
        plan.instance_alignments[*instance] =
            (std::max)(plan.instance_alignments[*instance],
                       port_requirements[node.input_count + output_index]
                           .minimum_alignment);
      producer_instances[value] = *instance;
      plan.instance_producer_targets[*instance] = target;
      if ((roles & detail::kExternalOutput) != 0)
        plan.external_instances[value] = *instance;
      if (!plan.node_outputs.try_push_back(*instance))
        return plan_allocation_failure();
      plan.value_materialized[*instance] = 1;
    }
    const bool materializes = node.traits().materializes;
    const CompileFallbackReason reason =
        options.region_selection == RegionSelectionPolicy::kDisabled
            ? CompileFallbackReason::kRegionSelectionDisabled
        : region_coverage[node_index] == RegionCoverage::kRejected
            ? CompileFallbackReason::kRegionDeclined
            : (region_coverage[node_index] == RegionCoverage::kNone
                   ? CompileFallbackReason::kNoEligibleRegion
                   : CompileFallbackReason::kRegionNotSelected);
    const std::uint32_t step_index =
        static_cast<std::uint32_t>(plan.steps.size());
    const std::uint32_t step_node_begin =
        static_cast<std::uint32_t>(plan.step_nodes.size());
    if (!plan.step_nodes.try_push_back(node_index))
      return plan_allocation_failure();
    if (!plan.steps.try_emplace_back(
            compute::PreparedOperation{},
            static_cast<std::uint32_t>(node_index), step_node_begin,
            input_begin, node.input_count, output_begin, node.output_count,
            materializes, reason, ExecutionTargetId{target},
            workspace_storage == kNone ? StorageDomainId{}
                                       : StorageDomainId{workspace_storage})) {
      return plan_allocation_failure();
    }
    if (node.kind() == compute::OperationKind::kView) {
      auto &step = plan.steps.back();
      step.kind = detail::PlanStepKind::kView;
      step.fallback_reason = CompileFallbackReason::kNone;
      if (plan.domain_graph.valid()) {
        // The envelope may violate guards. Domain View offsets are resolved
        // from the already-checked concrete signature, never this corner.
        step.view_offset = 0;
      } else {
        HYLE_ASSIGN_OR_RETURN(
            step.view_offset,
            compute::detail::view_alias_offset(
                *node.operation().get_if<compute::View>(),
                plan.instance_descriptions[plan.node_inputs[input_begin]],
                plan.instance_descriptions[plan.node_outputs[output_begin]]));
      }
      const auto input = plan.node_inputs[input_begin];
      const auto output = plan.node_outputs[output_begin];
      plan.alias_sources[output] = input;
      plan.alias_offsets[output] = step.view_offset;
      plan.instance_output_contracts[output] = plan.instance_output_contracts[input];
      plan.instance_producer_targets[output] = plan.instance_producer_targets[input];
      if ((plan.instance_roles[input] & detail::kConstant) != 0) {
        plan.instance_roles[output] |= detail::kConstant;
        plan.value_materialized[output] = 0;
      }
    }
    plan.node_steps[node_index] = step_index;
  }
  for (std::uint32_t node_step : plan.node_steps) {
    if (node_step == kNone)
      return Status::Internal("graph node has no execution step");
  }

  for (std::uint32_t value : graph.outputs) {
    if (plan.external_instances[value] != kNone)
      continue;
    const std::uint32_t source = producer_instances[value];
    if (source == kNone)
      return Status::Internal("published graph value has no physical source");
    const std::uint32_t storage = plan.external_storage[value];
    std::uint32_t destination =
        instances[static_cast<std::size_t>(value) * storage_count + storage];
    if (destination == kNone) {
      HYLE_ASSIGN_OR_RETURN(destination,
          ensure_local_instance(ensure_local_instance, value, storage, false));
      plan.instance_roles[destination] |= detail::kExternalOutput;
    } else {
      plan.instance_roles[destination] |= detail::kExternalOutput;
    }
    plan.external_instances[value] = destination;
  }

  // The entire physical route, including output copies, exists before any
  // heavyweight artifact. A prepare decline invalidates this construction;
  // it never changes the selected provider/target/storage in this loop.
  auto aligned = prepare_constant_alignments(plan);
  if (!aligned.ok()) {
    outcome.rejected_plan = aligned.IsInvalidArgument();
    return aligned;
  }
  HYLE_RETURN_IF_ERROR(retain_constant_owners(plan, constants));
  HYLE_RETURN_IF_ERROR(packing.begin_candidate());
  bool constant_route_declined = false;
  const auto materialized = materialize_constants(plan, environment, constants, packing,
                                                 constant_route_declined);
  if (!materialized.ok()) {
    outcome.rejected_plan = constant_route_declined;
    return materialized;
  }
  if (plan.compilation_storage &&
      !plan.artifact_charges.try_resize(plan.steps.size()))
    return plan_allocation_failure();
  NodeShapeScratch shape_scratch(plan.allocator);
  for (auto &step : plan.steps) {
    if (step.kind == detail::PlanStepKind::kCopy) {
      const auto &source = plan.targets[step.target.ordinal()].source;
      auto retained = detail::ExecutionSourceAccess::compute(source);
      if (retained)
        continue; // The original shared Queue copy has no prepared artifact.
      if (!retained.error().IsUnimplemented())
        return std::move(retained).error();
      auto charge = reserve_compilation_storage(plan, options, {}, outcome.rejected_plan);
      if (!charge) {
        return std::move(charge).error();
      }
      HYLE_ASSIGN_OR_RETURN(auto capability, detail::ExecutionSourceAccess::region(source));
      ProviderRegionPrepared prepared{};
      const auto input = plan.node_inputs[step.input_begin];
      const auto output = plan.node_outputs[step.output_begin];
      const auto status = detail::ExecutionSourceAccess::prepare_copy(source,
          copy_source(input),
          plan.storage_domains[plan.instance_storage[output]].resource,
          plan.allocator, &prepared);
      // Unsupported geometry is a cold route decline, including when no
      // artifact was created. Actual preparation errors remain fatal.
      if (!status.ok())
        outcome.rejected_plan = status.IsUnimplemented();
      const auto *table = prepared.vtable;
      if (table == nullptr || table->destroy_prepared == nullptr) {
        if (prepared.state != nullptr || status.ok())
          return Status::Internal("native transfer returned an unowned artifact");
        return status;
      }
      detail::PreparedRegion artifact(std::move(capability), prepared, plan.allocator);
      if (!status.ok()) {
        return status;
      }
      const auto runtime = prepared.properties.runtime;
      if (!runtime.valid() || runtime.mode != compute::ProviderRuntimeMode::kDelegated ||
          runtime.workspace_owner != compute::ProviderWorkspaceOwner::kProvider ||
          table->alias_support == nullptr || table->create_invocation == nullptr ||
          table->destroy_invocation == nullptr || table->shared_submit != nullptr ||
          table->delegated_submit == nullptr || table->endpoint == nullptr ||
          table->can_import_endpoint == nullptr || table->prepare_bindings == nullptr ||
          table->seal_bindings == nullptr || prepared.properties.workspace.size != 0 ||
          (table->output_contract && table->output_contract(prepared.state, 0) != RegionOutputContract{}))
        return Status::Internal("native transfer returned an invalid execution contract");
      if (!replay_allowed(options.provider_replay, runtime.replay_class)) {
        outcome.rejected_plan = true;
        return Status::Unimplemented("native transfer does not satisfy the replay policy");
      }
      step.region = std::move(artifact);
      if (plan.compilation_storage)
        plan.artifact_charges[static_cast<std::size_t>(&step - plan.steps.data())] = std::move(charge->resident);
      continue;
    }
    if (step.kind == detail::PlanStepKind::kView)
      continue;
    if (!inputs.try_resize(step.input_count) ||
        !outputs.try_resize(step.output_count) ||
        !input_resources.try_resize(step.input_count) ||
        !output_resources.try_resize(step.output_count))
      return plan_allocation_failure();
    for (std::size_t port = 0; port < step.input_count; ++port) {
      const auto instance = plan.node_inputs[step.input_begin + port];
      inputs[port] = plan.instance_descriptions[instance];
      input_resources[port] =
          plan.storage_domains[plan.instance_storage[instance]].resource;
    }
    for (std::size_t port = 0; port < step.output_count; ++port) {
      const auto instance = plan.node_outputs[step.output_begin + port];
      outputs[port] = plan.instance_descriptions[instance];
      output_resources[port] =
          plan.storage_domains[plan.instance_storage[instance]].resource;
    }
    const auto workspace_resource =
        step.workspace_storage.valid()
            ? plan.storage_domains[step.workspace_storage.ordinal()].resource
            : mem::MemoryResourceRef{};
    if (placement_attempts == options.max_placement_attempts) {
      outcome.prepare_limit = true;
      plan.search.placement_limit_reached = true;
      return Status::ResourceExhausted(
          "graph placement preparation budget is exhausted");
    }
    ++placement_attempts;
    if (step.kind == detail::PlanStepKind::kOperation) {
      const auto *route = operation_routes[step.node_begin];
      if (route == nullptr || route->rejected)
        return Status::Internal(
            "planned graph operation has no eligible route");
      const auto requirements =
          std::span<const compute::ResourceRequirement>(
              discovery.operations.requirements)
              .subspan(route->requirements_begin,
                       static_cast<std::size_t>(step.input_count) +
                           step.output_count);
      HYLE_ASSIGN_OR_RETURN(auto shape,
                            shape_scratch.bind(plan, step.node_begin));
      auto charge = reserve_compilation_storage(
          plan, options, route->proposal.cost.compilation_storage, outcome.rejected_plan);
      if (!charge) {
        return std::move(charge).error();
      }
      HYLE_ASSIGN_OR_RETURN(auto retained,
          detail::ExecutionSourceAccess::compute(
              environment.targets[step.target.ordinal()].source));
      auto attempt =
          compute::detail::OperationPreparationAccess::prepare_candidate(
              *retained,
              route->proposal, graph.nodes[step.node_begin].operation(), inputs,
              outputs, requirements.first(step.input_count),
              requirements.subspan(step.input_count),
              compute::OperationResourceBindings{.inputs = input_resources,
                                                 .outputs = output_resources,
                                                 .workspace_resource =
                                                     workspace_resource},
              shape, plan.allocator);
      if (!attempt)
        return std::move(attempt).error();
      if (!attempt->succeeded()) {
        discovery.physical.reject_suffix(preparation_prefix[step.node_begin]);
        outcome.rejected_operation = static_cast<std::size_t>(
            route - discovery.operations.candidates.data());
        return std::move(attempt->reason);
      }
      step.operation = std::move(attempt->prepared);
      if (plan.compilation_storage)
        plan.artifact_charges[static_cast<std::size_t>(
            &step - plan.steps.data())] = std::move(charge->resident);
    } else {
      const auto &candidate = selected_regions[region_by_node[step.node_begin]];
      const auto &skeleton = discovery.resources[candidate.proposal_order];
      const auto values =
          std::span<const ValueId>(discovery.values)
              .subspan(skeleton.port_begin,
                       skeleton.input_count + skeleton.output_count);
      container::SmallVector<ConstantBinding, 0, mem::AllocatorRef>
          region_constants(plan.allocator);
      container::SmallVector<RegionConstantOrigin, 0, mem::AllocatorRef>
          region_constant_origins(plan.allocator);
      for (std::size_t port = 0; port < step.input_count; ++port) {
        const auto value = values[port];
        const auto instance = plan.node_inputs[step.input_begin + port];
        if (instance >= plan.instance_roles.size() ||
            instance >= plan.alias_sources.size())
          return Status::Internal("region constant physical instance is invalid");
        if ((plan.instance_roles[instance] & detail::kConstant) == 0)
          continue;
        // A declared domain resolves View offsets from each invocation's
        // concrete signature. Its envelope is not a cold immutable binding.
        if (plan.domain_graph.valid() &&
            plan.alias_sources[instance] != detail::kNoInPlaceSource)
          continue;
        auto root = instance;
        std::size_t offset = 0;
        while (plan.alias_sources[root] != detail::kNoInPlaceSource) {
          const auto parent = plan.alias_sources[root];
          if (parent >= root || parent >= plan.alias_sources.size() ||
              root >= plan.alias_offsets.size() ||
              plan.alias_offsets[root] > SIZE_MAX - offset)
            return Status::Internal("region constant physical alias chain is invalid");
          offset += plan.alias_offsets[root];
          root = parent;
        }
        if (root >= plan.instance_values.size() ||
            root >= plan.instance_descriptions.size() ||
            root >= plan.instance_storage.size() ||
            instance >= plan.instance_descriptions.size() ||
            instance >= plan.instance_storage.size() ||
            plan.instance_storage[root] != plan.instance_storage[instance])
          return Status::Internal("region constant physical root is invalid");
        const auto logical_root = plan.instance_values[root];
        const auto slot = constant_slot(plan, logical_root);
        // Prepared region outputs have their own artifact provenance. This
        // interface currently exposes only original logical constant roots.
        if (slot == SIZE_MAX)
          continue;
        const auto *owned = plan.owned_instance_constant(root);
        auto binding = plan.constant_bindings[slot];
        if (owned != nullptr) {
          HYLE_ASSIGN_OR_RETURN(binding, compute::ConstTensorView::create(
              plan.instance_descriptions[root], owned->view()));
        }
        HYLE_ASSIGN_OR_RETURN(const auto bytes,
            plan.instance_descriptions[instance].required_bytes());
        HYLE_ASSIGN_OR_RETURN(const auto storage,
            binding.storage().subview(offset, bytes));
        HYLE_ASSIGN_OR_RETURN(binding, compute::ConstTensorView::create(
            plan.instance_descriptions[instance], storage));
        if (owned && plan.instance_storage[root] >= plan.storage_domains.size())
          return Status::Internal("region constant physical storage is invalid");
        if (!region_constants.try_push_back({value, binding,
                owned ? ConstantPolicy::kMaterialize : ConstantPolicy::kBorrow,
                owned ? plan.storage_domains[plan.instance_storage[root]].owner
                      : plan.constant_owners[slot],
                owned ? compute::ReadLease{} : plan.constant_leases[slot]}) ||
            !region_constant_origins.try_push_back(
                {value, plan.constant_ids[slot], offset}))
          return plan_allocation_failure();
      }
      RegionCompileRequest request{
          .graph = &published_graph,
          .nodes = candidate_nodes(candidate, all_region_nodes),
          .inputs = values.first(skeleton.input_count),
          .outputs = values.subspan(skeleton.input_count),
          .constants = region_constants,
          .input_resources = input_resources,
          .output_resources = output_resources,
          .workspace_resource = workspace_resource,
          .route = candidate.route,
          .effect_accesses =
              std::span<const RegionEffectAccess>(discovery.effects)
                  .subspan(skeleton.effect_begin, skeleton.effect_count),
          .effect_bindings = plan.effect_bindings,
          .has_unknown_effects = skeleton.has_unknown_effects,
          .workload = options.workload,
          .resource_budget = options.resource_budget,
          .available_storage_domains = environment.storage_domains};
      request.constant_origins = region_constant_origins;
      auto charge = reserve_compilation_storage(
          plan, options, skeleton.proposal.compilation_storage, outcome.rejected_plan);
      if (!charge) {
        return std::move(charge).error();
      }
      auto attempt =
          compile_region_artifact(plan, extensions[candidate.extension_index],
                                  request, skeleton.proposal,
                                  std::span<const std::uint8_t>(discovery.input_references)
                                      .subspan(skeleton.input_reference_begin, skeleton.input_count),
                                  std::span<const RegionOutputContract>(discovery.output_contracts)
                                      .subspan(skeleton.output_contract_begin, skeleton.output_count),
                                  options, packing);
      if (!attempt)
        return std::move(attempt).error();
      if (!attempt->prepared.valid()) {
        // Shared packing admission depends on this complete candidate's live
        // prefix, not on whether the provider itself is supported.
        if (attempt->compilation_budget_decline) {
          outcome.rejected_plan = true;
          return std::move(attempt->reason);
        }
        discovery.physical.reject_suffix(preparation_prefix[step.node_begin]);
        outcome.rejected_region = candidate.proposal_order;
        return std::move(attempt->reason);
      }
      step.region = std::move(attempt->prepared);
      // Native output ports omit immutable prepared values, but their logical
      // node identities remain in this region for all data/control ordering.
      // Consumers (including metadata Views) keep the same physical instance.
      std::uint32_t live_outputs = 0;
      for (std::uint32_t port = 0; port < step.output_count; ++port) {
        const auto instance = plan.node_outputs[step.output_begin + port];
        const auto logical = plan.instance_values[instance];
        const auto prepared = std::ranges::find_if(attempt->outputs,
            [&](const auto &entry) { return entry.value.ordinal() == logical; });
        if (prepared == attempt->outputs.end()) {
          plan.node_outputs[step.output_begin + live_outputs++] = instance;
          continue;
        }
        if (!plan.prepared_values.try_push_back({instance, prepared->binding}))
          return plan_allocation_failure();
        plan.instance_roles[instance] |= detail::kConstant;
        plan.instance_output_contracts[instance] = {};
        plan.instance_producer_targets[instance] = kNone;
        plan.value_materialized[instance] = 0;
      }
      step.output_count = live_outputs;
      if (plan.compilation_storage)
        plan.artifact_charges[static_cast<std::size_t>(
            &step - plan.steps.data())] = std::move(charge->resident);
    }
  }

  return Status::Ok();
}

void reset_workspace_decisions(detail::PlanState &plan) noexcept;
Status finalize_placed_steps(detail::PlanState &plan,
                             const detail::GraphState &graph,
                             ResourceSearch &physical, CompileOptions options,
                             bool &search_limit, bool &rejected_plan) noexcept;

Result<CompileReport>
build_report(const detail::PlanState &plan, const detail::GraphState &graph,
             std::span<const RouteDiagnostic> routes) noexcept;

Result<CompiledStorageReport>
compiled_storage(const detail::PlanState &plan) noexcept {
  CompiledStorageReport total;
  for (const auto &artifact : plan.constant_artifacts)
    HYLE_RETURN_IF_ERROR(add_compiled_storage(total, artifact->storage()));
  for (const auto &owner : plan.canonical_constants) {
    if (owner && owner->ready)
      HYLE_RETURN_IF_ERROR(add_compiled_storage(
          total, owner->storage()));
  }
  for (const auto &step : plan.steps) {
    if (!step.region.valid() && step.kind != detail::PlanStepKind::kOperation)
      continue;
    const auto persistent = step.region.valid()
                                ? step.region.properties().persistent_bytes
                                : step.operation.persistent_bytes();
    HYLE_RETURN_IF_ERROR(add_compiled_storage(
        total, {.provider_persistent_bytes = persistent.value_or(0),
                .packed_constant_bytes =
                    step.region.valid()
                        ? step.region.properties().packed_constant_bytes
                        : 0,
                .unknown_provider_count = persistent ? 0u : 1u}));
  }
  return total;
}

Status check_compiled_storage(CompiledStorageReport storage,
                              CompiledStorageLimits limits,
                              CompiledStorageReport shared) noexcept {
  if (limits.max_bytes == SIZE_MAX)
    return Status::Ok();
  // Shared bytes describe actual prior selected owners, including canonical
  // allocations. A child still reports all storage it independently retains.
  HYLE_RETURN_IF_ERROR(detail::subtract_compiled_storage(storage, shared));
  if (storage.known_bytes() > limits.max_bytes)
    return Status::ResourceExhausted(
        "graph compiled storage budget is exceeded");
  if (storage.unknown_provider_count != 0 &&
      limits.unknown == UnknownStoragePolicy::kReject)
    return Status::FailedPrecondition(
        "graph compiled storage budget cannot admit unknown provider storage");
  return Status::Ok();
}

struct InvocationStorageRequirement final {
  std::size_t bytes = 0;
  bool unknown = false;
};

Result<InvocationStorageRequirement>
invocation_storage_requirement(const detail::PlanState &plan) noexcept {
  InvocationStorageRequirement result{plan.workspace_bytes, false};
  for (const auto &step : plan.steps) {
    if (!step.emits_command())
      continue;
    std::optional<std::size_t> bytes;
    if (step.kind == detail::PlanStepKind::kOperation)
      bytes = step.operation.invocation_private_bytes();
    else if (step.region.valid())
      bytes = step.region.properties().invocation_private_bytes;
    else
      continue; // Backend transfers have no provider invocation allocation.
    if (!bytes) {
      result.unknown = true;
    } else {
      if (*bytes > SIZE_MAX - result.bytes)
        return std::unexpected(Status::ResourceExhausted(
            "graph invocation storage size overflows"));
      result.bytes += *bytes;
    }
  }
  return result;
}

Status check_invocation_storage(const detail::PlanState &plan,
                                InvocationStorageLimits limits) noexcept {
  HYLE_ASSIGN_OR_RETURN(const auto requirement,
                        invocation_storage_requirement(plan));
  if (requirement.bytes > limits.max_bytes)
    return Status::ResourceExhausted(
        "graph invocation workspace/private storage exceeds budget");
  if (limits.max_bytes != SIZE_MAX && requirement.unknown &&
      limits.unknown == UnknownStoragePolicy::kReject)
    return Status::ResourceExhausted(
        "graph invocation private storage has no declared upper bound");
  return Status::Ok();
}

// One complete candidate transaction. Common Context/resource/constant owners,
// specialization and aggregate search counters stay in plan. No Invocation
// exists yet; moving these vectors cannot invalidate a submitted command.
// The retained snapshot is destroyed before plan's canonical constants.
Status prepare_placed_steps(
    detail::PlanState &plan, const Graph &published_graph,
    const detail::GraphState &graph, CompileEnvironment environment,
    CompileOptions options, const ProviderPolicy &provider_policy,
    ConstantPreparation &constants, std::span<RouteDiagnostic> routes,
    CompileReport &selected_report,
    detail::ConstantArtifactCache &packing,
    std::span<const detail::RegionAdapter> adapters) noexcept {
  RegionDiscovery discovery(plan.allocator);
  discovery.adapters = adapters;
  HYLE_RETURN_IF_ERROR(
      initialize_placement_facts(plan, graph, options, discovery.placement));
  const std::size_t configured_lane_count = plan.lanes.size();
  bool attempted = false;
  for (;;) {
    // No Invocation or user work exists during this transaction. Clear all
    // artifacts before calling any provider for the next construction; the
    // attempt resets its physical metadata and keeps aggregate budgets. Final
    // scheduling/alias/workspace choices also belong to this one candidate.
    plan.steps.clear();
    plan.artifact_charges.clear();
    plan.immutable_alias_order.clear();
    plan.prepared_values.clear();
    plan.constant_artifacts.clear();
    for (std::size_t slot = 0; slot < constants.bindings.size(); ++slot) {
      plan.constant_bindings[slot] = constants.bindings[slot].tensor;
    }
    plan.shared_artifact_storage = {};
    plan.region_count = 0;
    if (attempted) {
      while (plan.lanes.size() > configured_lane_count)
        plan.lanes.pop_back();
      plan.serialization_edges.clear();
      reset_workspace_decisions(plan);
    }
    std::fill(plan.node_steps.begin(), plan.node_steps.end(), detail::kNoNode);
    PlacementAttemptOutcome outcome;
    Status status = prepare_placed_steps_attempt(
        plan, published_graph, graph, environment, options, provider_policy,
        discovery, constants, outcome, packing);
    attempted = true;
    if (status.ok())
      status =
          finalize_placed_steps(plan, graph, discovery.physical, options,
                                outcome.search_limit, outcome.rejected_plan);
    if (status.ok()) {
      for (std::size_t slot = 0; slot < plan.constant_ids.size(); ++slot) {
        if (!constants.bindings.empty() &&
            constants.bindings[slot].policy == ConstantPolicy::kMaterialize)
          continue;
        status = validate_external_read_resource(
            plan, plan.constant_ids[slot].ordinal(),
            plan.constant_bindings[slot].storage().resource());
        if (!status.ok()) {
          outcome.rejected_plan = true;
          break;
        }
      }
    }
    if (status.ok())
      HYLE_RETURN_IF_ERROR(build_immutable_alias_index(plan));
    CompiledStorageReport storage_report;
    if (status.ok()) {
      auto storage = compiled_storage(plan);
      if (!storage) {
        status = std::move(storage).error();
      } else {
        storage_report = *storage;
        status =
            check_compiled_storage(storage_report, options.compiled_storage,
                                   plan.shared_artifact_storage);
      }
      // No user work has been accepted. As with the final workspace gate,
      // reject this complete route, not the provider or the whole search.
      outcome.rejected_plan = !status.ok();
    }
    if (status.ok() && options.invocation_storage) {
      status = check_invocation_storage(plan, *options.invocation_storage);
      outcome.rejected_plan = !status.ok();
    }
    if (status.ok() && options.require_concurrent_invocations &&
        !plan.supports_concurrent_invocations) {
      outcome.rejected_plan = true;
      status = Status::FailedPrecondition(
          "graph route cannot satisfy requested concurrent invocations");
    }
    if (outcome.rejected_plan) {
      if (status.ok() || outcome.rejected_region || outcome.rejected_operation)
        return Status::Internal("graph plan rejection has conflicting outcome");
      ++plan.search.route_plan_rejections;
      discovery.last_rejection = std::move(status);
      continue;
    }
    if (!outcome.rejected_region && !outcome.rejected_operation) {
      if (outcome.no_feasible_route && !discovery.last_rejection.ok())
        return discovery.last_rejection;
      if (status.ok()) {
        // No retry can use these facts now. Empty moves release their spill
        // storage without allocating before the report's metadata is built.
        discovery.placement = PlacementFacts(plan.allocator);
        discovery.region_coverage =
            decltype(discovery.region_coverage)(plan.allocator);
        for (const auto &frame : discovery.routes.frames()) {
          const auto &route = discovery.routes.route(frame.chosen);
          const auto diagnostic =
              route.region
                  ? RouteDiagnostic{discovery.candidates[route.proposal]
                                        .implementation,
                                    discovery.candidates[route.proposal].policy,
                                    discovery.candidates[route.proposal]
                                        .preference,
                                    route.dispatches}
                  : RouteDiagnostic{
                        discovery.operations.candidates[route.proposal]
                            .proposal.implementation,
                        discovery.operations.candidates[route.proposal]
                            .proposal.policy,
                        discovery.operations.candidates[route.proposal]
                            .proposal.preference,
                        route.dispatches};
          discovery.routes.for_each_node(
              route,
              [&](std::uint32_t node) noexcept { routes[node] = diagnostic; });
        }
        ++plan.search.feasible_plans;
        HYLE_ASSIGN_OR_RETURN(auto report, build_report(plan, graph, routes));
        auto &report_state = *detail::CompileReportAccess::state(report);
        report_state.compiled_storage = storage_report;
        report_state.total_compiled_storage = storage_report;
        report_state.search = plan.search;
        selected_report = std::move(report);
        return Status::Ok();
      }
      if ((outcome.search_limit || outcome.prepare_limit) &&
          !discovery.last_rejection.ok())
        return std::move(status)
            .WithContext(discovery.last_rejection.message())
            .WithContext("graph search exhausted after a rejected candidate");
      return status;
    }
    if (status.ok() || (outcome.rejected_region && outcome.rejected_operation))
      return Status::Internal("graph retry has no unique declined route");
    if (outcome.rejected_operation) {
      const auto rejected = *outcome.rejected_operation;
      if (rejected >= discovery.operations.candidates.size() ||
          discovery.operations.candidates[rejected].rejected)
        return Status::Internal(
            "graph operation retry did not exclude a candidate");
      if (!discovery.physical.branched())
        discovery.operations.candidates[rejected].rejected = true;
      ++plan.search.operation_route_rejections;
    } else {
      const auto rejected = *outcome.rejected_region;
      if (rejected >= discovery.rejected.size() ||
          discovery.rejected[rejected] != 0)
        return Status::Internal(
            "graph region retry did not exclude a candidate");
      if (!discovery.physical.branched()) {
        discovery.rejected[rejected] = 1;
        // Resource filtering can leave proposal ordinals sparse. Update only
        // this fully rejected candidate, not a physical-suffix decline.
        for (const auto &candidate : discovery.candidates) {
          if (candidate.proposal_order != rejected)
            continue;
          for (NodeId node : candidate_nodes(candidate, discovery.nodes))
            discovery.region_coverage[node.ordinal()] = RegionCoverage::kRejected;
          break;
        }
      }
      ++plan.search.region_route_rejections;
    }
    discovery.last_rejection = std::move(status);
  }
}

Status discover_region_candidates(
    const Graph &published_graph, const detail::GraphState &graph,
    const ExecutionSource &source, CompileOptions options,
    const ProviderPolicy &provider_policy, mem::AllocatorRef allocator,
    RegionExtensions &extensions, RegionCandidates &candidates,
    RegionNodeIds &proposal_nodes, std::size_t &proposal_count,
    std::size_t &proposal_node_count, CompileSearchReport &search,
    ExecutionTargetId target, std::span<const std::uint32_t> node_targets,
    const detail::GraphState *domain,
    std::span<const detail::RegionAdapter> adapters,
    const OperationDiscovery *operations) noexcept {
  if (options.region_selection == RegionSelectionPolicy::kDisabled ||
      graph.nodes.empty()) {
    return Status::Ok();
  }
  if (options.max_region_proposals_per_provider == 0) {
    return Status::InvalidArgument(
        "enabled graph region selection requires a non-zero per-provider "
        "proposal budget");
  }
  if (options.max_region_proposals >
          (std::numeric_limits<std::uint32_t>::max)() ||
      options.max_region_proposals_per_provider >
          (std::numeric_limits<std::uint32_t>::max)() ||
      options.max_region_proposal_nodes >
          (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::OutOfRange(
        "graph region proposal budget exceeds the supported identity space");
  }
  auto retained = detail::ExecutionSourceAccess::compute(source);
  if (!retained && !retained.error().IsUnimplemented())
    return std::move(retained).error();
  const auto *context = retained ? *retained : nullptr;
  const auto provider_count = context ? context->provider_count() : 0;
  if (provider_count > (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::ResourceExhausted(
        "compute provider identity space is exhausted");
  }
  const std::size_t extension_begin = extensions.size();
  for (std::size_t index = 0; index < provider_count; ++index) {
    auto allowed = provider_policy.allows_family(*context, index);
    if (!allowed)
      return std::move(allowed).error();
    if (!*allowed)
      continue;
    auto extension = compute::detail::ProviderExtensionAccess::query(
        *context, index, kRegionProviderExtensionId);
    if (!extension) {
      if (extension.error().IsUnimplemented())
        continue;
      return std::move(extension).error();
    }
    HYLE_ASSIGN_OR_RETURN(auto capability,
        detail::GraphRegionSource::shared(source, std::move(*extension)));
    if (!extensions.try_push_back(
            RegionExtension{std::move(capability),
                            static_cast<std::uint32_t>(index), target})) {
      return plan_allocation_failure();
    }
  }
  if (!context && provider_policy.allows_native()) {
    auto capability = detail::ExecutionSourceAccess::region(source);
    if (capability) {
      if (!extensions.try_push_back(RegionExtension{
              std::move(*capability), UINT32_MAX, target}))
        return plan_allocation_failure();
    } else if (!capability.error().IsUnimplemented()) {
      return std::move(capability).error();
    }
  }
  for (const auto &adapter : adapters) {
    if (adapter.target != target)
      continue;
    bool allowed = provider_policy.allows_native();
    if (context && adapter.provider_index != UINT32_MAX) {
      HYLE_ASSIGN_OR_RETURN(allowed,
          provider_policy.allows_family(*context, adapter.provider_index));
    }
    if (!allowed)
      continue;
    if (!extensions.try_push_back(RegionExtension{
            adapter.source, adapter.provider_index, target, &adapter}))
      return plan_allocation_failure();
  }
  if (extensions.size() == extension_begin)
    return Status::Ok();

  container::SmallVector<RegionProposal, 32, mem::AllocatorRef>
      proposal_storage(allocator);
  if (proposal_count > options.max_region_proposals ||
      proposal_node_count > options.max_region_proposal_nodes ||
      proposal_node_count > proposal_nodes.size()) {
    return Status::Internal("graph region proposal accounting overflowed");
  }
  const std::size_t remaining_budget =
      options.max_region_proposals - proposal_count;
  if (!proposal_storage.try_resize(remaining_budget) ||
      !candidates.try_reserve(options.max_region_proposals)) {
    return plan_allocation_failure();
  }

  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> selected(
      allocator);
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> reachable_outside(
      allocator);
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> allowed_nodes(allocator);
  container::SmallVector<RegionLeafPreference, 16, mem::AllocatorRef>
      leaf_preferences(allocator);
  if ((!node_targets.empty() && node_targets.size() != graph.nodes.size()) ||
      (domain != nullptr && domain->nodes.size() != graph.nodes.size()))
    return Status::Internal("region discovery node eligibility has a different graph");
  if (!allowed_nodes.try_resize(graph.nodes.size()))
    return plan_allocation_failure();
  const std::size_t leaf_target_count = operations
      ? operations->ranges.size() / graph.nodes.size() : 0;
  if (context && operations &&
      (operations->ranges.size() % graph.nodes.size() != 0 ||
       target.ordinal() >= leaf_target_count))
    return Status::Internal("region leaf preference target differs");
  if (context && operations &&
      !leaf_preferences.try_resize(graph.nodes.size()))
    return plan_allocation_failure();
  // One original provider can expose multiple native capabilities. Adapting
  // them does not multiply its existing hard discovery allowance.
  container::SmallVector<std::size_t, 8, mem::AllocatorRef> provider_proposals(
      allocator);
  // Native capabilities on this one actual source share one allowance. They
  // have no Compute provider index; the extra counter is not a provider slot.
  if (!provider_proposals.try_resize(provider_count + 1))
    return plan_allocation_failure();
  std::fill(provider_proposals.begin(), provider_proposals.end(), 0);
  for (std::size_t extension_index = extension_begin;
       extension_index < extensions.size(); ++extension_index) {
    const auto &extension = extensions[extension_index].source;
    if (!leaf_preferences.empty()) {
      for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        auto &facts = leaf_preferences[node];
        facts = {};
        for (const auto &leaf : operations->for_node(
                 node, target.ordinal(), leaf_target_count)) {
          // The zero-copy metadata alternative has no actual provider owner.
          if (leaf.metadata_view)
            continue;
          const auto rank = leaf.proposal.preference;
          if (!facts.best || rank < *facts.best)
            facts.best = rank;
          if (leaf.proposal.provider_index ==
                  extensions[extension_index].provider_index &&
              (!facts.provider || rank < *facts.provider))
            facts.provider = rank;
        }
      }
    }
    bool any_allowed = false, all_allowed = true;
    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
      const bool allowed =
          (node_targets.empty() || node_targets[node] == detail::kNoWorkspaceRequest ||
           node_targets[node] == target.ordinal()) &&
          provider_policy.allows_node(node, extension.provider().id,
                                      extension.provider_instance_id()) &&
          (domain == nullptr || !has_dynamic_shape(*domain, domain->nodes[node]));
      allowed_nodes[node] = allowed ? 1 : 0;
      any_allowed |= allowed;
      all_allowed &= allowed;
    }
    if (!any_allowed)
      continue;
    const std::size_t remaining = options.max_region_proposals - proposal_count;
    const auto provider_index = extensions[extension_index].provider_index != UINT32_MAX
        ? extensions[extension_index].provider_index : provider_count;
    const std::size_t provider_remaining =
        options.max_region_proposals_per_provider - provider_proposals[provider_index];
    const std::size_t provider_capacity =
        (std::min)(remaining, provider_remaining);
    const std::size_t remaining_nodes =
        options.max_region_proposal_nodes - proposal_node_count;
    if (remaining == 0 || remaining_nodes == 0) {
      // There is still an unvisited region provider. No zero-capacity callback
      // is needed, and no later target/provider can restore a global budget.
      search.region_proposal_limit_reached |= remaining == 0;
      search.region_node_limit_reached |= remaining_nodes == 0;
      break;
    }
    if (provider_remaining == 0) {
      search.per_provider_proposal_limit_reached = true;
      continue;
    }
    // A unique proposal cannot contain more nodes than the graph. Reserve
    // only this provider's possible contribution, bounded by the global cap.
    const std::size_t node_capacity =
        provider_capacity > remaining_nodes / graph.nodes.size()
            ? remaining_nodes
            : provider_capacity * graph.nodes.size();
    if (!proposal_nodes.try_resize(proposal_node_count + node_capacity))
      return plan_allocation_failure();
    RegionProposalWriter writer(
        std::span<RegionProposal>(proposal_storage).first(provider_capacity),
        std::span<NodeId>(proposal_nodes)
            .subspan(proposal_node_count, node_capacity));
    struct Episode final {
      RegionDiscoveryRequest request;
      mem::AllocatorRef allocator{};
      RegionProposalWriter *writer = nullptr;
    } episode{{published_graph, all_allowed ? std::span<const std::uint8_t>{}
                                           : std::span<const std::uint8_t>(allowed_nodes),
               std::span<const RegionLeafPreference>(leaf_preferences)},
              allocator, &writer};
    const Status proposed = extensions[extension_index].call(
        +[](void *provider_context, const void *interface_value,
            void *opaque) noexcept -> Status {
          const auto *interface =
              static_cast<const RegionProviderVTable *>(interface_value);
          auto &call = *static_cast<Episode *>(opaque);
          if (interface == nullptr || interface->propose == nullptr ||
              interface->compile == nullptr) {
            return Status::Internal(
                "graph region provider interface is incomplete");
          }
          return interface->propose(provider_context, call.request,
                                    call.allocator, *call.writer);
        },
        &episode);
    if (!proposed.ok())
      return proposed;
    provider_proposals[provider_index] += writer.proposals().size();

    if (writer.proposal_limit_reached()) {
      search.region_proposal_limit_reached |= provider_capacity == remaining;
      search.per_provider_proposal_limit_reached |=
          provider_capacity == provider_remaining;
    }
    search.region_node_limit_reached |= writer.node_limit_reached();

    std::uint32_t provider_proposal_order = 0;
    for (const RegionProposal &proposal : writer.proposals()) {
      const std::uint32_t proposal_order = static_cast<std::uint32_t>(
          proposal_count + provider_proposal_order++);
      auto valid = validate_contractible_region(graph, proposal.nodes, selected,
                                                reachable_outside);
      if (!valid)
        return std::move(valid).error();
      if (!*valid)
        return Status::InvalidArgument(
            "graph region leaves its node set and later re-enters it");
      if (!compute::ProviderRuntimeProperties{.replay_class =
                                                  proposal.replay_class}
               .valid()) {
        return Status::Internal(
            "region provider declared an invalid replay class");
      }
      if (!proposal.cost.valid()) {
        return Status::Internal(
            "region provider declared invalid structural cost facts");
      }
      if ((!proposal.policy.empty() && !proposal.policy.valid()) ||
          (proposal.policy.valid() && !proposal.implementation.valid())) {
        return Status::Internal(
            "region provider declared an invalid route identity");
      }
      if (!replay_allowed(options.provider_replay, proposal.replay_class))
        continue;
      // The request exposes these same target/provider/exact-domain facts
      // before structural discovery. A proposal still proves no broader
      // mathematical domain, and never overrides an excluded member.
      if (std::any_of(proposal.nodes.begin(), proposal.nodes.end(),
                      [&](NodeId node) { return allowed_nodes[node.ordinal()] == 0; }))
        continue;
      const std::ptrdiff_t node_offset =
          proposal.nodes.data() - proposal_nodes.data();
      if (node_offset < 0 ||
          static_cast<std::size_t>(node_offset) >
              options.max_region_proposal_nodes - proposal.nodes.size()) {
        return Status::Internal(
            "graph region proposal node storage is not writer-owned");
      }
      std::uint32_t binding_preference = compute::kUnrankedProviderPreference;
      if (context && extensions[extension_index].provider_index != UINT32_MAX) {
        HYLE_ASSIGN_OR_RETURN(binding_preference, context->provider_preference(
            extensions[extension_index].provider_index));
      }
      const std::uint32_t preference =
          proposal.preference == (std::numeric_limits<std::uint32_t>::max)()
              ? binding_preference
              : proposal.preference;
      if (!candidates.try_push_back(RegionCandidate{
              .cost = proposal.cost,
              .route = proposal.route,
              .replay_class = proposal.replay_class,
              .node_offset = static_cast<std::uint32_t>(node_offset),
              .node_count = static_cast<std::uint32_t>(proposal.nodes.size()),
              .extension_index = static_cast<std::uint32_t>(extension_index),
              .provider_index = extensions[extension_index].provider_index,
              .proposal_order = proposal_order,
              .target = target,
              .provider_family = extension.provider().id,
              .implementation = proposal.implementation,
              .policy = proposal.policy,
              .preference = preference,
          })) {
        return plan_allocation_failure();
      }
    }
    proposal_count += writer.size();
    proposal_node_count += writer.node_size();
  }
  return Status::Ok();
}

template <class Unsigned>
Unsigned saturating_add(Unsigned left, Unsigned right) noexcept {
  const Unsigned limit = (std::numeric_limits<Unsigned>::max)();
  return right > limit - left ? limit : left + right;
}

template <class Unsigned>
Unsigned saturating_multiply(Unsigned left, Unsigned right) noexcept {
  const Unsigned limit = (std::numeric_limits<Unsigned>::max)();
  return right != 0 && left > limit / right ? limit : left * right;
}

// Immutable values may have no native write port, but still carry their
// logical producer's incoming control dependencies to ordinary data consumers.
// Resolve the logical node through node_steps each time: scheduling can reorder
// physical steps without changing the prepared value's identity.
Status add_immutable_value_producers(
    const detail::PlanState &plan, const detail::GraphState &graph,
    std::span<std::uint32_t> producers) noexcept {
  for (std::size_t instance = 0; instance < plan.instance_values.size(); ++instance) {
    if ((plan.instance_roles[instance] & detail::kConstant) == 0)
      continue;
    const auto logical = plan.instance_values[instance];
    if (logical >= graph.values.size())
      return Status::Internal("graph prepared output value is invalid");
    const auto node = graph.values[logical].producer;
    if (node == detail::kNoNode && graph.values[logical].kind == ValueKind::kConstant)
      continue;
    if (node >= plan.node_steps.size() ||
        plan.node_steps[node] >= plan.steps.size())
      return Status::Internal("graph prepared output producer is invalid");
    const auto step = plan.node_steps[node];
    if (producers[instance] != detail::kNoNode &&
        producers[instance] != step)
      return Status::Internal("graph prepared output producer conflicts");
    producers[instance] = step;
  }
  return Status::Ok();
}

// Reorder already-prepared plan steps without changing the logical Graph.
// Kahn scheduling keeps a ready FIFO per execution lane and continues the
// current lane while possible. This deterministically removes avoidable
// Queue segment boundaries from interleaved independent branches; providers,
// transfer routes and bindings remain fully resolved before the warm path.
Status schedule_steps(detail::PlanState &plan, const detail::GraphState &graph,
                      ResourceSearch &physical, CompileOptions options,
                      bool &search_limit, bool &rejected_plan) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Edges = container::SmallVector<std::uint64_t, 32, mem::AllocatorRef>;
  constexpr std::uint32_t kNone = detail::kNoNode;
  Ordinals lane_choices(plan.allocator);
  const std::size_t step_count = plan.steps.size();
  if (step_count >= kNone) {
    return Status::ResourceExhausted(
        "graph scheduled-step identity space is exhausted");
  }
  for (std::size_t step_index = 0; step_index < step_count; ++step_index) {
    detail::PlanStep &step = plan.steps[step_index];
    if (step.schedule_identity == kNone)
      step.schedule_identity = static_cast<std::uint32_t>(step_index);
    if (!step.target.valid() || step.target.ordinal() >= plan.targets.size())
      return Status::Internal("prepared graph step has an invalid target");
    if (step.runtime_properties().mode ==
        compute::ProviderRuntimeMode::kDelegated) {
      if (step.lane.valid() && step.lane.ordinal() < plan.lanes.size() &&
          plan.lanes[step.lane.ordinal()].runtime_mode ==
              compute::ProviderRuntimeMode::kDelegated) {
        continue;
      }
      if (plan.lanes.size() >= detail::kNoNode ||
          !plan.lanes.try_push_back(detail::ExecutionLaneRecord{
              .target = step.target,
              .role = step.kind == detail::PlanStepKind::kCopy
                          ? ExecutionLaneRole::kTransfer : ExecutionLaneRole::kCompute,
              .runtime_mode = compute::ProviderRuntimeMode::kDelegated,
          })) {
        return plan.lanes.size() >= detail::kNoNode
                   ? Status::ResourceExhausted(
                         "graph delegated lane identity space is exhausted")
                   : plan_allocation_failure();
      }
      step.lane =
          ExecutionLaneId{static_cast<std::uint32_t>(plan.lanes.size() - 1)};
      continue;
    }
    // A workspace-serialization pass changes order, not this candidate's lane.
    if (step.lane.valid())
      continue;
    const ExecutionLaneRole preferred_role =
        step.kind == detail::PlanStepKind::kCopy ? ExecutionLaneRole::kTransfer
                                                 : ExecutionLaneRole::kCompute;
    lane_choices.clear();
    for (std::size_t lane = 0; lane < plan.lanes.size(); ++lane) {
      if (plan.lanes[lane].runtime_mode ==
              compute::ProviderRuntimeMode::kSharedQueue &&
          plan.lanes[lane].target == step.target &&
          plan.lanes[lane].role == preferred_role) {
        if (!lane_choices.try_push_back(static_cast<std::uint32_t>(lane)))
          return plan_allocation_failure();
      }
    }
    if (preferred_role != ExecutionLaneRole::kCompute) {
      for (std::size_t lane = 0; lane < plan.lanes.size(); ++lane) {
        if (plan.lanes[lane].runtime_mode ==
                compute::ProviderRuntimeMode::kSharedQueue &&
            plan.lanes[lane].target == step.target &&
            plan.lanes[lane].role == ExecutionLaneRole::kCompute) {
          if (!lane_choices.try_push_back(static_cast<std::uint32_t>(lane)))
            return plan_allocation_failure();
        }
      }
    }
    if (lane_choices.empty()) {
      rejected_plan = true;
      return Status::InvalidArgument(
          "compile environment has no compatible lane for a graph step");
    }
    HYLE_ASSIGN_OR_RETURN(const auto selected,
                          physical.choose(lane_choices,
                                          options.max_route_search_states,
                                          plan.search, search_limit));
    step.lane = ExecutionLaneId{selected};
  }
  if (step_count <= 1 || plan.lanes.size() <= 1)
    return Status::Ok();
  if (plan.node_inputs.size() > SIZE_MAX - graph.dependencies.size() ||
      plan.node_inputs.size() + graph.dependencies.size() >
          SIZE_MAX - plan.serialization_edges.size()) {
    return Status::ResourceExhausted(
        "graph scheduling dependency capacity overflows");
  }

  Ordinals producers(plan.allocator);
  Ordinals identity_steps(plan.allocator);
  Edges edges(plan.allocator);
  if (!producers.try_resize(plan.instance_values.size()) ||
      !identity_steps.try_resize(step_count) ||
      !edges.try_reserve(plan.node_inputs.size() + graph.dependencies.size() +
                         plan.serialization_edges.size())) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &producer : producers)
    producer = kNone;
  for (std::uint32_t &step : identity_steps)
    step = kNone;
  for (std::size_t step_index = 0; step_index < step_count; ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (step.schedule_identity >= step_count ||
        identity_steps[step.schedule_identity] != kNone) {
      return Status::Internal("graph scheduled-step identity is invalid");
    }
    identity_steps[step.schedule_identity] =
        static_cast<std::uint32_t>(step_index);
    for (std::size_t output_index = 0; output_index < step.output_count;
         ++output_index) {
      producers[plan.node_outputs[step.output_begin + output_index]] =
          static_cast<std::uint32_t>(step_index);
    }
  }
  HYLE_RETURN_IF_ERROR(add_immutable_value_producers(plan, graph, producers));

  auto add_edge = [&](std::uint32_t source,
                      std::uint32_t destination) noexcept -> Status {
    if (source == kNone || source == destination)
      return Status::Ok();
    const std::uint64_t key =
        (static_cast<std::uint64_t>(source) << 32) | destination;
    return edges.try_push_back(key) ? Status::Ok() : plan_allocation_failure();
  };
  for (std::size_t step_index = 0; step_index < step_count; ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    for (std::size_t input_index = 0; input_index < step.input_count;
         ++input_index) {
      Status status =
          add_edge(producers[plan.node_inputs[step.input_begin + input_index]],
                   static_cast<std::uint32_t>(step_index));
      if (!status.ok())
        return status;
    }
    if (step.kind == detail::PlanStepKind::kCopy) {
      continue;
    }
    const std::span<const std::uint32_t> logical_nodes =
        std::span<const std::uint32_t>(plan.step_nodes)
            .subspan(step.node_set_begin, step.node_count);
    for (std::uint32_t node : logical_nodes) {
      const detail::NodeRecord &record = graph.nodes[node];
      for (std::size_t index = 0; index < record.dependency_count; ++index) {
        const std::uint32_t prerequisite =
            graph.dependencies[record.dependency_begin + index];
        Status status = add_edge(plan.node_steps[prerequisite],
                                 static_cast<std::uint32_t>(step_index));
        if (!status.ok())
          return status;
      }
    }
  }
  for (const std::uint64_t stable_edge : plan.serialization_edges) {
    const std::uint32_t source_identity =
        static_cast<std::uint32_t>(stable_edge >> 32);
    const std::uint32_t destination_identity =
        static_cast<std::uint32_t>(stable_edge);
    if (source_identity >= identity_steps.size() ||
        destination_identity >= identity_steps.size() ||
        identity_steps[source_identity] == kNone ||
        identity_steps[destination_identity] == kNone) {
      return Status::Internal("graph serialization edge identity is invalid");
    }
    Status status = add_edge(identity_steps[source_identity],
                             identity_steps[destination_identity]);
    if (!status.ok())
      return status;
  }
  std::sort(edges.begin(), edges.end());
  auto unique_end = std::unique(edges.begin(), edges.end());
  if (!edges.try_resize(static_cast<std::size_t>(unique_end - edges.begin())))
    return plan_allocation_failure();
  if (edges.size() >= kNone) {
    return Status::ResourceExhausted(
        "graph scheduling dependency identity space is exhausted");
  }

  Ordinals indegree(plan.allocator);
  Ordinals outgoing_begin(plan.allocator);
  Ordinals ready_head(plan.allocator);
  Ordinals ready_tail(plan.allocator);
  Ordinals ready_next(plan.allocator);
  if (!indegree.try_resize(step_count) ||
      !outgoing_begin.try_resize(step_count + 1) ||
      !ready_head.try_resize(plan.lanes.size()) ||
      !ready_tail.try_resize(plan.lanes.size()) ||
      !ready_next.try_resize(step_count)) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &value : indegree)
    value = 0;
  for (std::uint32_t &value : outgoing_begin)
    value = 0;
  for (std::uint32_t &value : ready_head)
    value = kNone;
  for (std::uint32_t &value : ready_tail)
    value = kNone;
  for (std::uint32_t &value : ready_next)
    value = kNone;
  for (const std::uint64_t edge : edges) {
    const std::uint32_t source = static_cast<std::uint32_t>(edge >> 32);
    const std::uint32_t destination = static_cast<std::uint32_t>(edge);
    if (source >= step_count || destination >= step_count ||
        source == destination) {
      return Status::Internal("prepared graph step dependency is invalid");
    }
    ++outgoing_begin[source + 1];
    ++indegree[destination];
  }
  for (std::size_t index = 1; index < outgoing_begin.size(); ++index)
    outgoing_begin[index] += outgoing_begin[index - 1];

  auto enqueue_ready = [&](std::uint32_t step) noexcept -> Status {
    const ExecutionLaneId lane = plan.steps[step].lane;
    if (!lane.valid() || lane.ordinal() >= plan.lanes.size())
      return Status::Internal("prepared graph step has an invalid lane");
    std::uint32_t &head = ready_head[lane.ordinal()];
    std::uint32_t &tail = ready_tail[lane.ordinal()];
    if (tail == kNone)
      head = step;
    else
      ready_next[tail] = step;
    tail = step;
    return Status::Ok();
  };
  for (std::size_t step = 0; step < step_count; ++step) {
    if (indegree[step] == 0) {
      Status status = enqueue_ready(static_cast<std::uint32_t>(step));
      if (!status.ok())
        return status;
    }
  }

  detail::PlanSteps scheduled(plan.allocator);
  if (!scheduled.try_reserve(step_count))
    return plan_allocation_failure();
  std::uint32_t current_lane = kNone;
  while (scheduled.size() < step_count) {
    if (current_lane == kNone || ready_head[current_lane] == kNone) {
      current_lane = kNone;
      std::uint32_t earliest = kNone;
      for (std::size_t lane = 0; lane < ready_head.size(); ++lane) {
        if (ready_head[lane] != kNone &&
            (earliest == kNone || ready_head[lane] < earliest)) {
          earliest = ready_head[lane];
          current_lane = static_cast<std::uint32_t>(lane);
        }
      }
      if (current_lane == kNone)
        return Status::Internal("prepared graph step dependencies are cyclic");
    }
    const std::uint32_t selected = ready_head[current_lane];
    ready_head[current_lane] = ready_next[selected];
    if (ready_head[current_lane] == kNone)
      ready_tail[current_lane] = kNone;
    ready_next[selected] = kNone;

    const std::size_t begin = outgoing_begin[selected];
    const std::size_t end = outgoing_begin[selected + 1];
    for (std::size_t edge_index = begin; edge_index < end; ++edge_index) {
      const std::uint32_t destination =
          static_cast<std::uint32_t>(edges[edge_index]);
      if (indegree[destination] == 0)
        return Status::Internal("graph scheduling indegree underflow");
      if (--indegree[destination] == 0) {
        Status status = enqueue_ready(destination);
        if (!status.ok())
          return status;
      }
    }
    if (!scheduled.try_push_back(std::move(plan.steps[selected])))
      return plan_allocation_failure();
  }

  plan.steps = std::move(scheduled);
  for (std::uint32_t &step : plan.node_steps)
    step = kNone;
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (step.kind == detail::PlanStepKind::kCopy) {
      continue;
    }
    const std::span<const std::uint32_t> logical_nodes =
        std::span<const std::uint32_t>(plan.step_nodes)
            .subspan(step.node_set_begin, step.node_count);
    for (std::uint32_t node : logical_nodes)
      plan.node_steps[node] = static_cast<std::uint32_t>(step_index);
  }
  return Status::Ok();
}

Status build_segments(detail::PlanState &plan,
                      const detail::GraphState &graph) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Edges = container::SmallVector<std::uint64_t, 32, mem::AllocatorRef>;
  if (plan.steps.size() >= detail::kNoNode) {
    return Status::ResourceExhausted(
        "graph segment step identity space is exhausted");
  }
  plan.segments.clear();
  plan.capture_ranges.clear();
  plan.seal_binding_steps.clear();
  plan.copy_steps.clear();
  plan.command_count = 0;
  plan.capture_command_count = 0;
  plan.segment_dependencies.clear();
  plan.segment_successors.clear();
  plan.has_parallel_targets = false;
  plan.terminal_segments.clear();
  plan.completion_segments.clear();
  plan.required_queue_count = 0;
  Ordinals producers(plan.allocator);
  Ordinals identity_steps(plan.allocator);
  Ordinals last_lane_segment(plan.allocator);
  Edges edges(plan.allocator);
  if (!plan.segments.try_reserve(plan.steps.size()) ||
      !plan.step_segments.try_resize(plan.steps.size()) ||
      !producers.try_resize(plan.instance_values.size()) ||
      !identity_steps.try_resize(plan.steps.size()) ||
      !last_lane_segment.try_resize(plan.lanes.size()) ||
      !plan.lane_segment_counts.try_resize(plan.lanes.size())) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &segment : plan.step_segments)
    segment = detail::kNoNode;
  for (std::uint32_t &producer : producers)
    producer = detail::kNoNode;
  for (std::uint32_t &step : identity_steps)
    step = detail::kNoNode;
  for (std::uint32_t &segment : last_lane_segment)
    segment = detail::kNoNode;
  for (std::uint32_t &count : plan.lane_segment_counts)
    count = 0;

  for (std::size_t step = 0; step < plan.steps.size(); ++step) {
    const auto &scheduled = plan.steps[step];
    const std::uint32_t identity = scheduled.schedule_identity;
    if (identity >= identity_steps.size() ||
        identity_steps[identity] != detail::kNoNode) {
      return Status::Internal("graph scheduled-step identity is invalid");
    }
    identity_steps[identity] = static_cast<std::uint32_t>(step);
    if (scheduled.kind == detail::PlanStepKind::kCopy &&
        !plan.copy_steps.try_push_back(static_cast<std::uint32_t>(step)))
      return plan_allocation_failure();
    if (scheduled.emits_command() && scheduled.region.valid() &&
        scheduled.region.seals_bindings() &&
        !plan.seal_binding_steps.try_push_back(static_cast<std::uint32_t>(step)))
      return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const detail::PlanStep &step = plan.steps[index];
    for (std::size_t output_index = 0; output_index < step.output_count;
         ++output_index) {
      producers[plan.node_outputs[step.output_begin + output_index]] =
          static_cast<std::uint32_t>(index);
    }
  }
  HYLE_RETURN_IF_ERROR(add_immutable_value_producers(plan, graph, producers));

  // Index the existing memory-reuse edges by destination once. Together with
  // tensor/control producers these are entry prerequisites, not removable
  // apparent waits. This is cold O(steps + edges) metadata only.
  Ordinals serialization_begin(plan.allocator), serialization_cursor(plan.allocator);
  Ordinals serialization_sources(plan.allocator), entry_stamp(plan.allocator);
  if (!serialization_begin.try_resize(plan.steps.size() + 1) ||
      !serialization_cursor.try_resize(plan.steps.size()) ||
      !serialization_sources.try_resize(plan.serialization_edges.size()) ||
      !entry_stamp.try_resize(plan.steps.size())) return plan_allocation_failure();
  std::fill(serialization_begin.begin(), serialization_begin.end(), 0u);
  std::fill(entry_stamp.begin(), entry_stamp.end(), detail::kNoNode);
  for (const auto edge : plan.serialization_edges) {
    const auto source = static_cast<std::uint32_t>(edge >> 32);
    const auto destination = static_cast<std::uint32_t>(edge);
    if (source >= identity_steps.size() || destination >= identity_steps.size())
      return Status::Internal("graph serialization edge identity is invalid");
    ++serialization_begin[identity_steps[destination] + 1];
  }
  for (std::size_t i = 1; i < serialization_begin.size(); ++i)
    serialization_begin[i] += serialization_begin[i - 1];
  std::copy_n(serialization_begin.begin(), plan.steps.size(), serialization_cursor.begin());
  for (const auto edge : plan.serialization_edges) {
    const auto source = identity_steps[static_cast<std::uint32_t>(edge >> 32)];
    const auto destination = identity_steps[static_cast<std::uint32_t>(edge)];
    serialization_sources[serialization_cursor[destination]++] = source;
  }
  auto visit_sources = [&](std::size_t index, auto &&visit) noexcept -> Status {
    const auto &step = plan.steps[index];
    for (std::size_t port = 0; port < step.input_count; ++port)
      visit(producers[plan.node_inputs[step.input_begin + port]]);
    if (step.kind != detail::PlanStepKind::kCopy) {
      const auto nodes = std::span<const std::uint32_t>(plan.step_nodes)
          .subspan(step.node_set_begin, step.node_count);
      for (const auto node : nodes) {
        const auto &record = graph.nodes[node];
        for (std::size_t edge = 0; edge < record.dependency_count; ++edge)
          visit(plan.node_steps[graph.dependencies[record.dependency_begin + edge]]);
      }
    }
    for (auto edge = serialization_begin[index]; edge < serialization_begin[index + 1]; ++edge)
      visit(serialization_sources[edge]);
    return Status::Ok();
  };

  // Views and ready-only regions do not enter a Queue. Keep their logical
  // dependencies inside the surrounding report interval, but
  // start/split runtime segments only for actual commands. This avoids an
  // empty sequence/event for View-only graphs and lets consecutive commands
  // on one lane remain one uninterrupted Queue prefix.
  std::uint32_t active_segment = detail::kNoNode;
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    if (!plan.steps[index].emits_command()) {
      plan.step_segments[index] = active_segment;
      continue;
    }
    const ExecutionLaneId lane = plan.steps[index].lane;
    if (!lane.valid() || lane.ordinal() >= plan.lanes.size())
      return Status::Internal("graph plan step has an invalid lane");
    bool split = active_segment == detail::kNoNode ||
                 plan.segments[active_segment].lane != lane;
    if (!split) {
      HYLE_RETURN_IF_ERROR(visit_sources(index, [&](std::uint32_t source) noexcept {
        if (source == detail::kNoNode || source >= index) return;
        const auto prerequisite = plan.step_segments[source];
        if (prerequisite != detail::kNoNode && prerequisite != active_segment &&
            entry_stamp[prerequisite] != active_segment) split = true;
      }));
    }
    if (split) {
      if (active_segment != detail::kNoNode) {
        detail::PlanSegment &previous = plan.segments[active_segment];
        previous.step_count =
            static_cast<std::uint32_t>(index - previous.step_begin);
      }
      if (!plan.segments.try_push_back(detail::PlanSegment{
              .lane = lane,
              .step_begin = active_segment == detail::kNoNode
                                ? 0u
                                : static_cast<std::uint32_t>(index),
          })) {
        return plan_allocation_failure();
      }
      active_segment = static_cast<std::uint32_t>(plan.segments.size() - 1);
      HYLE_RETURN_IF_ERROR(visit_sources(index, [&](std::uint32_t source) noexcept {
        if (source == detail::kNoNode || source >= index) return;
        const auto prerequisite = plan.step_segments[source];
        if (prerequisite != detail::kNoNode && prerequisite != active_segment)
          entry_stamp[prerequisite] = active_segment;
      }));
    }
    plan.step_segments[index] = active_segment;
  }
  if (active_segment != detail::kNoNode) {
    detail::PlanSegment &last = plan.segments[active_segment];
    last.step_count =
        static_cast<std::uint32_t>(plan.steps.size() - last.step_begin);
  }
  for (std::size_t segment_index = 0; segment_index < plan.segments.size();
       ++segment_index) {
    const detail::PlanSegment &segment = plan.segments[segment_index];
    const std::size_t end = segment.step_begin + segment.step_count;
    for (std::size_t step = segment.step_begin; step < end; ++step)
      plan.step_segments[step] = static_cast<std::uint32_t>(segment_index);
    ++plan.lane_segment_counts[segment.lane.ordinal()];
  }
  for (std::size_t lane = 0; lane < plan.lane_segment_counts.size(); ++lane) {
    if (plan.lane_segment_counts[lane] != 0 &&
        plan.lanes[lane].runtime_mode ==
            compute::ProviderRuntimeMode::kSharedQueue) {
      ++plan.required_queue_count;
    }
  }


  auto add_edge = [&](std::uint32_t segment,
                      std::uint32_t prerequisite) noexcept -> Status {
    if (segment == prerequisite)
      return Status::Ok();
    const std::uint64_t key =
        (static_cast<std::uint64_t>(segment) << 32) | prerequisite;
    return edges.try_push_back(key) ? Status::Ok() : plan_allocation_failure();
  };
  for (std::size_t segment_index = 0; segment_index < plan.segments.size();
       ++segment_index) {
    const detail::PlanSegment &segment = plan.segments[segment_index];
    const std::uint32_t lane = segment.lane.ordinal();
    if (last_lane_segment[lane] != detail::kNoNode) {
      Status status = add_edge(static_cast<std::uint32_t>(segment_index),
                               last_lane_segment[lane]);
      if (!status.ok())
        return status;
    }
    last_lane_segment[lane] = static_cast<std::uint32_t>(segment_index);
    const std::size_t end = segment.step_begin + segment.step_count;
    for (std::size_t step_index = segment.step_begin; step_index < end;
         ++step_index) {
      const detail::PlanStep &step = plan.steps[step_index];
      for (std::size_t input_index = 0; input_index < step.input_count;
           ++input_index) {
        const std::uint32_t producer =
            producers[plan.node_inputs[step.input_begin + input_index]];
        if (producer == detail::kNoNode)
          continue;
        Status status = add_edge(static_cast<std::uint32_t>(segment_index),
                                 plan.step_segments[producer]);
        if (!status.ok())
          return status;
      }
      if (step.kind == detail::PlanStepKind::kCopy) {
        continue;
      }
      const std::span<const std::uint32_t> logical_nodes =
          std::span<const std::uint32_t>(plan.step_nodes)
              .subspan(step.node_set_begin, step.node_count);
      for (std::uint32_t node : logical_nodes) {
        const detail::NodeRecord &record = graph.nodes[node];
        for (std::size_t dependency_index = 0;
             dependency_index < record.dependency_count; ++dependency_index) {
          const std::uint32_t dependency =
              graph.dependencies[record.dependency_begin + dependency_index];
          const std::uint32_t dependency_step = plan.node_steps[dependency];
          if (dependency_step == detail::kNoNode)
            return Status::Internal(
                "graph control dependency has no compiled step");
          Status status = add_edge(static_cast<std::uint32_t>(segment_index),
                                   plan.step_segments[dependency_step]);
          if (!status.ok())
            return status;
        }
      }
    }
  }
  for (const std::uint64_t stable_edge : plan.serialization_edges) {
    const std::uint32_t source_identity =
        static_cast<std::uint32_t>(stable_edge >> 32);
    const std::uint32_t destination_identity =
        static_cast<std::uint32_t>(stable_edge);
    if (source_identity >= identity_steps.size() ||
        destination_identity >= identity_steps.size()) {
      return Status::Internal("graph serialization edge identity is invalid");
    }
    const std::uint32_t source_step = identity_steps[source_identity];
    const std::uint32_t destination_step = identity_steps[destination_identity];
    if (source_step == detail::kNoNode || destination_step == detail::kNoNode) {
      return Status::Internal("graph serialization edge step is unavailable");
    }
    Status status = add_edge(plan.step_segments[destination_step],
                             plan.step_segments[source_step]);
    if (!status.ok())
      return status;
  }

  std::sort(edges.begin(), edges.end());
  auto unique_end = std::unique(edges.begin(), edges.end());
  if (!edges.try_resize(static_cast<std::size_t>(unique_end - edges.begin()))) {
    return plan_allocation_failure();
  }
  if (edges.size() > UINT32_MAX)
    return Status::ResourceExhausted(
        "graph segment dependency index space is exhausted");
  Ordinals outgoing(plan.allocator);
  if (!outgoing.try_resize(plan.segments.size()))
    return plan_allocation_failure();
  for (std::uint32_t &count : outgoing)
    count = 0;
  for (std::uint64_t edge : edges)
    ++outgoing[static_cast<std::uint32_t>(edge)];
  for (std::size_t index = 0; index < outgoing.size(); ++index) {
    if (outgoing[index] == 0 && !plan.terminal_segments.try_push_back(
                                    static_cast<std::uint32_t>(index))) {
      return plan_allocation_failure();
    }
  }
  for (std::uint32_t last : last_lane_segment) {
    if (last != detail::kNoNode &&
        !plan.completion_segments.try_push_back(last)) {
      return plan_allocation_failure();
    }
  }

  if (!plan.segment_dependencies.try_resize(edges.size()) ||
      !plan.segment_successors.try_resize(edges.size()))
    return plan_allocation_failure();
  std::size_t write = 0;
  for (std::size_t segment_index = 0; segment_index < plan.segments.size();
       ++segment_index) {
    detail::PlanSegment &segment = plan.segments[segment_index];
    segment.dependency_begin = static_cast<std::uint32_t>(write);
    while (write < edges.size() &&
           static_cast<std::uint32_t>(edges[write] >> 32) == segment_index) {
      plan.segment_dependencies[write] =
          static_cast<std::uint32_t>(edges[write]);
      ++write;
    }
    segment.dependency_count =
        static_cast<std::uint32_t>(write - segment.dependency_begin);
  }

  // Reuse the outgoing counts as CSR insertion cursors. Successors index only
  // the final dependencies above, including original lane and workspace edges.
  write = 0;
  for (std::size_t index = 0; index < plan.segments.size(); ++index) {
    auto &segment = plan.segments[index];
    segment.successor_begin = static_cast<std::uint32_t>(write);
    segment.successor_count = outgoing[index];
    outgoing[index] = segment.successor_begin;
    write += segment.successor_count;
  }
  ExecutionTargetId first_target{};
  bool different_targets = false;
  bool has_unordered_segments = false;
  for (std::size_t destination = 0; destination < plan.segments.size();
       ++destination) {
    const auto &segment = plan.segments[destination];
    const auto target = plan.lanes[segment.lane.ordinal()].target;
    if (!first_target.valid())
      first_target = target;
    else
      different_targets |= target != first_target;
    bool follows_previous = false;
    const std::size_t end =
        segment.dependency_begin + segment.dependency_count;
    for (std::size_t edge = segment.dependency_begin; edge < end; ++edge) {
      const auto source = plan.segment_dependencies[edge];
      plan.segment_successors[outgoing[source]++] =
          static_cast<std::uint32_t>(destination);
      follows_previous |= destination != 0 && source == destination - 1;
    }
    has_unordered_segments |= destination != 0 && !follows_previous;
  }
  // A fully ordered cross-target graph retains the original serial fast path.
  // Any missing adjacent prerequisite can expose ready work behind a wait.
  plan.has_parallel_targets = different_targets && has_unordered_segments;
  // Preserve the exact compatible-run boundaries, including metadata between
  // commands and trailing Views, once after final scheduling. Queue support,
  // addresses, permissions and capture artifacts remain Invocation facts.
  for (std::size_t segment_index = 0; segment_index < plan.segments.size();
       ++segment_index) {
    auto &segment = plan.segments[segment_index];
    const std::size_t end = segment.step_begin + segment.step_count;
    for (std::size_t index = segment.step_begin; index < end; ++index) {
      const auto &step = plan.steps[index];
      if (!step.emits_command())
        continue;
      ++plan.command_count;
      segment.blocks_host |= step.runtime_properties().blocks_host;
    }
    if (!shared_lane(plan, segment.lane))
      continue;
    std::size_t cursor = segment.step_begin;
    while (cursor < end) {
      while (cursor < end && (!plan.steps[cursor].emits_command() ||
                             !plan.steps[cursor].capture_compatible()))
        ++cursor;
      if (cursor == end)
        break;
      const auto begin = cursor;
      std::uint32_t commands = 0;
      while (cursor < end) {
        const auto &step = plan.steps[cursor];
        if (step.emits_command() && !step.capture_compatible())
          break;
        commands += step.emits_command();
        ++cursor;
      }
      if (!plan.capture_ranges.try_push_back(detail::PlanCaptureRange{
              static_cast<std::uint32_t>(segment_index),
              static_cast<std::uint32_t>(begin),
              static_cast<std::uint32_t>(cursor - begin), commands}))
        return plan_allocation_failure();
      plan.capture_command_count += commands;
    }
  }
  return Status::Ok();
}

std::uint32_t alias_root(const detail::PlanState &plan,
                         std::uint32_t value) noexcept {
  while (plan.alias_sources[value] != detail::kNoInPlaceSource)
    value = plan.alias_sources[value];
  return value;
}

bool has_external_storage_root(const detail::PlanState &plan,
                               std::uint32_t value) noexcept {
  const std::uint32_t root = alias_root(plan, value);
  const std::uint8_t role = plan.instance_roles[root];
  return (role & (detail::kExternalInput | detail::kExternalOutput |
                  detail::kConstant | detail::kGeneratedOutput)) != 0;
}

void reset_workspace_decisions(detail::PlanState &plan) noexcept {
  plan.workspace_plan = {};
  plan.workspace_bytes = 0;
  plan.workspace_backing_count = 0;
  plan.supports_concurrent_invocations = true;
  plan.serialization_edge_count = plan.serialization_edges.size();
  for (std::size_t index = 0; index < plan.alias_sources.size(); ++index) {
    // Cold replicas of immutable Views retain their true parent/offset even
    // when no repeated physical View step is needed in that storage domain.
    if ((plan.instance_roles[index] & detail::kConstant) == 0) {
      plan.alias_sources[index] = detail::kNoInPlaceSource;
      plan.alias_offsets[index] = 0;
    }
    plan.instance_roles[index] &= static_cast<std::uint8_t>(
        ~(detail::kInvocationOwnedOutput | detail::kRetainedOutputRoot |
          detail::kGeneratedOutput));
  }
  for (std::uint8_t &role : plan.binding_roles) {
    role &= static_cast<std::uint8_t>(
        ~(detail::kInvocationOwnedOutput | detail::kSourceAliasedOutput |
          detail::kGeneratedOutput));
  }
  for (const auto value : plan.output_ids) {
    const auto instance = plan.external_instances[value.ordinal()];
    if (instance != detail::kNoWorkspaceRequest &&
        (plan.instance_roles[instance] & detail::kConstant) != 0)
      plan.binding_roles[value.ordinal()] |= detail::kSourceAliasedOutput;
  }
  for (std::uint32_t &request : plan.workspace_requests)
    request = detail::kNoWorkspaceRequest;
  plan.step_workspace_requests.clear();
  plan.storage_workspace_slots.clear();
  plan.workspace_resources.clear();
  plan.workspace_storage.clear();
}

Status plan_workspace(detail::PlanState &plan, const detail::GraphState &graph,
                      bool &reschedule_requested,
                      bool &rejected_plan) noexcept {
  using LastUses = container::SmallVector<std::uint64_t, 16, mem::AllocatorRef>;
  using Requests =
      container::SmallVector<mem::LifetimeRequest, 8, mem::AllocatorRef>;

  reschedule_requested = false;

  LastUses last_uses(plan.allocator);
  detail::ValueOrdinals producer_steps(plan.allocator);
  Requests requests(plan.allocator);
  if (!last_uses.try_resize(plan.instance_values.size()) ||
      !producer_steps.try_resize(plan.instance_values.size()) ||
      !plan.step_workspace_requests.try_resize(plan.steps.size()) ||
      !requests.try_reserve(plan.instance_values.size() + plan.steps.size())) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < last_uses.size(); ++index) {
    last_uses[index] = 0;
    producer_steps[index] = detail::kNoNode;
  }
  for (std::size_t index = 0; index < plan.step_workspace_requests.size();
       ++index) {
    plan.step_workspace_requests[index] = detail::kNoWorkspaceRequest;
  }

  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    const std::uint64_t terminal = step_index + 1;
    for (std::size_t output_index = 0; output_index < step.output_count;
         ++output_index) {
      const std::uint32_t output =
          plan.node_outputs[step.output_begin + output_index];
      last_uses[output] = (std::max)(last_uses[output], terminal);
      producer_steps[output] = static_cast<std::uint32_t>(step_index);
      if (step.kind == detail::PlanStepKind::kRegion) {
        const auto kind = step.region.output_kind(output_index);
        if (kind != RegionOutputKind::kCaller) {
          if (kind != RegionOutputKind::kStableGenerated &&
              kind != RegionOutputKind::kBoundGenerated &&
              kind != RegionOutputKind::kDeferredGenerated)
            return Status::Internal("region declared an invalid output storage kind");
          plan.instance_roles[output] |= detail::kGeneratedOutput;
          if ((plan.instance_roles[output] & detail::kExternalOutput) != 0) {
            plan.instance_roles[output] |= detail::kInvocationOwnedOutput;
            plan.binding_roles[plan.instance_values[output]] |=
                detail::kGeneratedOutput | detail::kInvocationOwnedOutput;
          }
        }
      }
    }
    for (std::size_t input_index = 0; input_index < step.input_count;
         ++input_index) {
      const std::uint32_t input =
          plan.node_inputs[step.input_begin + input_index];
      last_uses[input] = (std::max)(last_uses[input], terminal);
    }
  }

  const std::size_t segment_count = plan.segments.size();
  const std::size_t lane_count = plan.lanes.size();
  constexpr std::size_t kMaximumVectorClockCells = 4u * 1024u * 1024u;
  const bool precise_happens_before =
      lane_count != 0 && segment_count <= kMaximumVectorClockCells / lane_count;
  detail::ValueOrdinals segment_sequence(plan.allocator);
  container::SmallVector<std::uint64_t, 32, mem::AllocatorRef> clocks(plan.allocator);
  detail::ValueOrdinals lane_sequence(plan.allocator);
  if (!segment_sequence.try_resize(segment_count) ||
      !lane_sequence.try_resize(lane_count) ||
      (precise_happens_before &&
       !clocks.try_resize(segment_count * lane_count))) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &sequence : lane_sequence)
    sequence = 0;
  if (precise_happens_before) {
    for (std::uint64_t &clock : clocks)
      clock = 0;
    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      std::uint64_t *row = clocks.data() + segment * lane_count;
      const detail::PlanSegment &record = plan.segments[segment];
      const std::size_t dependency_end =
          record.dependency_begin + record.dependency_count;
      for (std::size_t dependency = record.dependency_begin;
           dependency < dependency_end; ++dependency) {
        const std::uint32_t source = plan.segment_dependencies[dependency];
        if (source >= segment)
          return Status::Internal("graph segment order is not topological");
        const std::uint64_t *source_row =
            clocks.data() + static_cast<std::size_t>(source) * lane_count;
        for (std::size_t lane = 0; lane < lane_count; ++lane)
          row[lane] = (std::max)(row[lane], source_row[lane]);
      }
      const std::uint32_t lane = record.lane.ordinal();
      segment_sequence[segment] = ++lane_sequence[lane];
      row[lane] = segment_sequence[segment];
    }
  } else {
    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      const std::uint32_t lane = plan.segments[segment].lane.ordinal();
      segment_sequence[segment] = ++lane_sequence[lane];
    }
  }

  const auto segment_before = [&](std::uint32_t source,
                                  std::uint32_t destination) noexcept {
    if (source == destination)
      return true;
    if (source >= segment_count || destination >= segment_count)
      return false;
    const auto source_lane = plan.segments[source].lane.ordinal();
    const auto destination_lane = plan.segments[destination].lane.ordinal();
    if (source_lane == destination_lane)
      return segment_sequence[source] < segment_sequence[destination];
    return precise_happens_before && segment_sequence[source] <=
        clocks[static_cast<std::size_t>(destination) * lane_count + source_lane];
  };

  // Establish all required aliases before choosing destructive aliases. A
  // later View is another reader of the same storage even when its own node
  // has not been visited at the candidate write's position.
  for (const auto &step : plan.steps) {
    if (step.kind != detail::PlanStepKind::kOperation &&
        step.kind != detail::PlanStepKind::kView)
      continue;
    const auto &node = graph.nodes[step.node_begin];
    if (node.traits().alias != compute::AliasPolicy::kMustAliasInput0 ||
        node.input_count == 0)
      continue;
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const auto output = plan.node_outputs[step.output_begin + output_index];
      const auto support = step.kind == detail::PlanStepKind::kView
          ? Result<compute::AliasSupport>(compute::AliasSupport{
                .supported = true, .byte_offset = step.view_offset})
          : step.operation.alias_support(output_index, 0);
      if (!support || !support->supported)
        return Status::Internal("prepared provider omitted a required graph alias");
      const auto input = plan.node_inputs[step.input_begin];
      plan.alias_sources[output] = input;
      plan.alias_offsets[output] = support->byte_offset;
      if ((plan.instance_roles[output] & detail::kExternalOutput) != 0) {
        const auto logical_output = plan.instance_values[output];
        if (has_external_storage_root(plan, input)) {
          plan.binding_roles[logical_output] |= detail::kSourceAliasedOutput;
        } else {
          plan.binding_roles[logical_output] |= detail::kInvocationOwnedOutput;
          plan.instance_roles[output] |= detail::kInvocationOwnedOutput;
          plan.instance_roles[alias_root(plan, input)] |= detail::kRetainedOutputRoot;
        }
      }
    }
  }

  // Liveness belongs to shared physical storage, not just one SSA value.
  // Required aliases are producer-first; optional aliases merge these groups
  // as they are selected. Path compression keeps the cold scratch near-linear
  // without changing the plan's immediate parent/offset or warm path.
  detail::ValueOrdinals alias_groups(plan.allocator);
  if (!alias_groups.try_resize(plan.instance_values.size()))
    return plan_allocation_failure();
  for (std::uint32_t instance = 0; instance < plan.instance_values.size(); ++instance) {
    const auto source = plan.alias_sources[instance];
    if (source != detail::kNoInPlaceSource && source >= instance)
      return Status::Internal("graph physical alias chain is invalid");
    const auto root = source == detail::kNoInPlaceSource
        ? instance : alias_groups[source];
    alias_groups[instance] = root;
    // A published value is read after commands finish, including when only a
    // View of a private workspace root is published.
    if ((plan.instance_roles[instance] & detail::kExternalOutput) != 0)
      last_uses[instance] = plan.steps.size() + 1;
    last_uses[root] = (std::max)(last_uses[root], last_uses[instance]);
  }
  const auto group_root = [&](std::uint32_t instance) noexcept {
    auto root = instance;
    while (alias_groups[root] != root)
      root = alias_groups[root];
    while (alias_groups[instance] != instance) {
      const auto parent = alias_groups[instance];
      alias_groups[instance] = root;
      instance = parent;
    }
    return root;
  };

  // A scalar last use is sufficient on one ordered lane. With several
  // lanes, earlier independent readers must also precede the destructive
  // write in the actual segment DAG. Reuse its existing bounded clocks.
  // Compact read lists exist only in cold multi-lane planning. On a successful
  // merge the old input readers are consumed, so a long in-place chain does
  // not repeatedly visit its complete history.
  detail::ValueOrdinals alias_read_heads(plan.allocator);
  detail::ValueOrdinals alias_read_next(plan.allocator);
  detail::ValueOrdinals alias_read_segments(plan.allocator);
  if (lane_count > 1) {
    if (!alias_read_heads.try_resize(plan.instance_values.size()) ||
        !alias_read_next.try_resize(plan.node_inputs.size()) ||
        !alias_read_segments.try_resize(plan.node_inputs.size()))
      return plan_allocation_failure();
    std::fill(alias_read_heads.begin(), alias_read_heads.end(), detail::kNoNode);
    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
      const auto &step = plan.steps[step_index];
      if (!step.emits_command())
        continue;
      const auto segment = plan.step_segments[step_index];
      if (segment >= segment_count)
        return Status::Internal("graph alias reader has no segment");
      for (std::size_t port = 0; port < step.input_count; ++port) {
        const auto use = step.input_begin + port;
        if (use >= detail::kNoNode)
          return Status::ResourceExhausted("graph alias reader identity space is exhausted");
        const auto root = group_root(plan.node_inputs[use]);
        alias_read_next[use] = alias_read_heads[root];
        alias_read_segments[use] = segment;
        alias_read_heads[root] = static_cast<std::uint32_t>(use);
      }
    }
  }
  const auto readers_before_write = [&](std::uint32_t root,
                                        std::size_t step) noexcept {
    if (lane_count <= 1)
      return true;
    for (auto use = alias_read_heads[root]; use != detail::kNoNode;
         use = alias_read_next[use])
      if (!segment_before(alias_read_segments[use], plan.step_segments[step]))
        return false;
    return true;
  };

  // The schema grants broad permission; the selected provider narrows it.
  // An optional alias may consume storage only after every alias reader.
  for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
    const auto &step = plan.steps[step_index];
    // Region aliases need a complete boundary-port contract before selection.
    if (step.kind != detail::PlanStepKind::kOperation)
      continue;
    const auto &node = graph.nodes[step.node_begin];
    const auto alias = node.traits().alias;
    if (alias == compute::AliasPolicy::kNone ||
        alias == compute::AliasPolicy::kMustAliasInput0 || node.input_count == 0)
      continue;
    for (std::size_t output_index = 0; output_index < node.output_count; ++output_index) {
      const auto output = plan.node_outputs[step.output_begin + output_index];
      const auto output_root = group_root(output);
      const bool output_is_external =
          (plan.instance_roles[output] & detail::kExternalOutput) != 0;
      const auto candidate_count = alias == compute::AliasPolicy::kMayAliasAnyInput
          ? node.input_count : std::size_t{1};
      for (std::size_t input_index = 0; input_index < candidate_count; ++input_index) {
        const auto input = plan.node_inputs[step.input_begin + input_index];
        if (plan.instance_storage[input] != plan.instance_storage[output])
          continue;
        const auto input_root = group_root(input);
        if (input_root == output_root || last_uses[input_root] != step_index + 1)
          continue;
        // Constants and their metadata Views stay immutable across replays.
        if ((plan.instance_roles[alias_root(plan, input)] & detail::kConstant) != 0)
          continue;
        HYLE_ASSIGN_OR_RETURN(auto support,
                              step.operation.alias_support(output_index, input_index));
        if (!support.supported || support.byte_offset != 0)
          continue;
        const bool input_is_external = (plan.instance_roles[input] &
            (detail::kExternalInput | detail::kConstant)) != 0;
        const bool internal_pair = !output_is_external && !input_is_external &&
            (plan.instance_roles[input] & detail::kExternalOutput) == 0 &&
            !has_external_storage_root(plan, input);
        const bool external_pair = output_is_external && input_is_external;
        if ((internal_pair || external_pair) &&
            plan.instance_descriptions[input] == plan.instance_descriptions[output] &&
            readers_before_write(input_root, step_index)) {
          plan.alias_sources[output] = input;
          plan.alias_offsets[output] = 0;
          alias_groups[output_root] = input_root;
          last_uses[input_root] = (std::max)(last_uses[input_root], last_uses[output_root]);
          if (lane_count > 1) {
            alias_read_heads[input_root] = alias_read_heads[output_root];
            alias_read_heads[output_root] = detail::kNoNode;
          }
          break;
        }
      }
    }
  }

  // The selected provider's port requirement applies to the physical bytes,
  // including when a metadata View exposes those bytes under another shape.
  // Propagate each port's requirement to its allocation root, checking the
  // complete root-relative offset rather than every intermediate View edge.
  HYLE_RETURN_IF_ERROR(propagate_alias_alignments(plan, !plan.domain_graph.valid()));

  for (std::size_t ordinal = 0; ordinal < plan.instance_values.size();
       ++ordinal) {
    if ((plan.instance_roles[ordinal] & detail::kGeneratedOutput) != 0 ||
        plan.value_materialized[ordinal] == 0 ||
        ((plan.instance_roles[ordinal] & detail::kExternalOutput) != 0 &&
         (plan.instance_roles[ordinal] & detail::kInvocationOwnedOutput) ==
             0)) {
      continue;
    }
    const compute::TensorDesc &description =
        plan.instance_descriptions[ordinal];
    auto size = description.required_bytes();
    if (!size)
      return std::move(size).error();
    if (requests.size() >= detail::kNoWorkspaceRequest) {
      return Status::ResourceExhausted(
          "graph workspace identity space is exhausted");
    }
    const std::uint32_t request = static_cast<std::uint32_t>(requests.size());
    const std::uint32_t producer_step = producer_steps[ordinal];
    if (producer_step == detail::kNoNode)
      return Status::Internal("graph value has no compiled producer step");
    const std::uint64_t begin = producer_step;
    const std::uint64_t end = (std::max)(begin + 1, last_uses[ordinal]);
    const std::uint32_t alias_source = plan.alias_sources[ordinal];
    const std::uint32_t alias_request =
        alias_source == detail::kNoInPlaceSource
            ? mem::kNoLifetimeAlias
            : plan.workspace_requests[alias_source];
    if (alias_source != detail::kNoInPlaceSource &&
        alias_request == detail::kNoWorkspaceRequest) {
      const bool materializes = plan.steps[producer_step].materializes;
      if (!materializes &&
          has_external_storage_root(plan, static_cast<std::uint32_t>(ordinal)))
        continue;
      return Status::Internal("graph internal alias source has no workspace");
    }
    const bool domain_alias =
        plan.domain_graph.valid() && alias_request != mem::kNoLifetimeAlias;
    if (!requests.try_push_back(mem::LifetimeRequest{
            .resource_slot = plan.instance_storage[ordinal],
            // Preserve root lifetime, not a fictitious joint offset/span
            // maximum. Actual subviews are bound and checked per Invocation.
            .size = domain_alias ? 0 : *size,
            // A derived view only promises its tensor's required alignment;
            // forcing the root's preferred 64-byte kernel alignment onto a
            // non-zero slice offset would reject otherwise valid aliases.
            .alignment = domain_alias ? 1
                         : alias_request == mem::kNoLifetimeAlias
                             ? (std::max)({std::size_t{64},
                                           description.required_alignment(),
                                           plan.instance_alignments[ordinal]})
                             : (std::max)(description.required_alignment(),
                                          plan.instance_alignments[ordinal]),
            .begin = begin,
            .end = end,
            .alias_of = alias_request,
            .alias_offset = domain_alias ? 0 : plan.alias_offsets[ordinal],
            .flags = (plan.instance_roles[ordinal] &
                      detail::kRetainedOutputRoot) != 0
                         ? mem::LifetimeFlag::kOutput
                         : mem::LifetimeFlag::kNone,
        })) {
      return plan_allocation_failure();
    }
    plan.workspace_requests[ordinal] = request;
  }

  // Provider scratchpads are per Invocation, not shared prepared state. Step
  // execution is ordered, so disjoint [step,step+1) requests may reuse one
  // backing while every operation/region invocation retains a stable view.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    if ((plan.steps[step_index].region.valid() ||
         plan.steps[step_index].kind == detail::PlanStepKind::kOperation) &&
        !plan.steps[step_index]
             .runtime_properties()
             .supports_concurrent_invocations) {
      plan.supports_concurrent_invocations = false;
    }
    if (!plan.steps[step_index].emits_command())
      continue;
    const compute::WorkspaceRequirement requirement =
        plan.steps[step_index].workspace_requirement();
    if (requirement.size == 0)
      continue;
    if (requests.size() >= detail::kNoWorkspaceRequest) {
      return Status::ResourceExhausted(
          "graph workspace identity space is exhausted");
    }
    const std::uint32_t request = static_cast<std::uint32_t>(requests.size());
    if (!requests.try_push_back(mem::LifetimeRequest{
            .resource_slot = plan.steps[step_index].workspace_storage.ordinal(),
            .size = requirement.size,
            .alignment = requirement.alignment,
            .begin = step_index,
            .end = step_index + 1,
        })) {
      return plan_allocation_failure();
    }
    plan.step_workspace_requests[step_index] = request;
  }

  // LifetimePlanner consumes scalar intervals. Different lanes instead form
  // a partial order, so first partition alias groups into comparable chains.
  // Requests on different chains get distinct planner resource slots even
  // when they use the same MemoryResource; this prevents an arbitrary
  // topological ordering from making unordered device work alias.
  using Words = container::SmallVector<std::uint64_t, 32, mem::AllocatorRef>;
  using U64s = container::SmallVector<std::uint64_t, 16, mem::AllocatorRef>;
  detail::ValueOrdinals request_roots(plan.allocator);
  detail::ValueOrdinals group_storage(plan.allocator);
  detail::ValueOrdinals group_chain(plan.allocator);
  detail::ValueOrdinals group_start_segment(plan.allocator);
  detail::ValueOrdinals chain_head(plan.allocator);
  detail::ValueOrdinals chain_tail(plan.allocator);
  detail::ValueOrdinals chain_storage(plan.allocator);
  detail::BindingRoles group_present(plan.allocator);
  U64s group_begin(plan.allocator);
  U64s group_end(plan.allocator);
  U64s use_pairs(plan.allocator);
  const std::size_t request_count = requests.size();
  if (!request_roots.try_resize(request_count) ||
      !group_storage.try_resize(request_count) ||
      !group_chain.try_resize(request_count) ||
      !group_start_segment.try_resize(request_count) ||
      !group_present.try_resize(request_count) ||
      !group_begin.try_resize(request_count) ||
      !group_end.try_resize(request_count) ||
      !use_pairs.try_reserve(plan.node_inputs.size() + request_count)) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < request_count; ++index) {
    group_storage[index] = detail::kNoWorkspaceRequest;
    group_chain[index] = detail::kNoWorkspaceRequest;
    group_start_segment[index] = detail::kNoNode;
    group_present[index] = 0;
    group_begin[index] = UINT64_MAX;
    group_end[index] = 0;
  }
  for (std::uint32_t index = 0; index < request_count; ++index) {
    const auto parent = requests[index].alias_of;
    if (parent != mem::kNoLifetimeAlias && parent >= index)
      return Status::Internal("graph workspace alias graph is invalid");
    // Requests are emitted producer-first, so each parent's root is ready.
    const auto root = parent == mem::kNoLifetimeAlias ? index : request_roots[parent];
    request_roots[index] = root;
    if (group_present[root] == 0) {
      group_present[root] = 1;
      group_storage[root] = requests[index].resource_slot;
    } else if (group_storage[root] != requests[index].resource_slot) {
      return Status::Internal("graph workspace alias crosses storage domains");
    }
    group_begin[root] = (std::min)(group_begin[root], requests[index].begin);
    group_end[root] = (std::max)(group_end[root], requests[index].end);
  }

  auto add_use = [&](std::uint32_t request,
                     std::uint32_t segment) noexcept -> Status {
    if (request >= request_count || segment >= plan.segments.size())
      return Status::Internal("graph workspace use is out of range");
    const std::uint64_t pair =
        (static_cast<std::uint64_t>(request_roots[request]) << 32) | segment;
    return use_pairs.try_push_back(pair) ? Status::Ok()
                                         : plan_allocation_failure();
  };
  for (std::uint32_t request = 0; request < request_count; ++request) {
    const std::uint64_t begin = requests[request].begin;
    if (begin >= plan.step_segments.size() ||
        plan.step_segments[static_cast<std::size_t>(begin)] ==
            detail::kNoNode) {
      return Status::Internal("graph workspace producer has no segment");
    }
    const std::uint32_t root = request_roots[request];
    const std::uint32_t segment =
        plan.step_segments[static_cast<std::size_t>(begin)];
    if (group_start_segment[root] == detail::kNoNode ||
        begin == group_begin[root]) {
      group_start_segment[root] = segment;
    }
    Status added = add_use(request, segment);
    if (!added.ok())
      return added;
  }
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const std::uint32_t segment = plan.step_segments[step_index];
    if (segment == detail::kNoNode)
      continue;
    const detail::PlanStep &step = plan.steps[step_index];
    for (std::size_t input_index = 0; input_index < step.input_count;
         ++input_index) {
      const std::uint32_t instance =
          plan.node_inputs[step.input_begin + input_index];
      const std::uint32_t request = plan.workspace_requests[instance];
      if (request == detail::kNoWorkspaceRequest)
        continue;
      Status added = add_use(request, segment);
      if (!added.ok())
        return added;
    }
  }
  std::sort(use_pairs.begin(), use_pairs.end());
  auto use_end = std::unique(use_pairs.begin(), use_pairs.end());
  if (!use_pairs.try_resize(
          static_cast<std::size_t>(use_end - use_pairs.begin()))) {
    return plan_allocation_failure();
  }

  auto group_uses = [&](std::uint32_t root) noexcept {
    const std::uint64_t first = static_cast<std::uint64_t>(root) << 32;
    const std::uint64_t last = first | UINT32_MAX;
    return std::pair{
        std::lower_bound(use_pairs.begin(), use_pairs.end(), first),
        std::upper_bound(use_pairs.begin(), use_pairs.end(), last)};
  };
  auto group_before = [&](std::uint32_t left, std::uint32_t right) noexcept {
    if (group_end[left] > group_begin[right])
      return false;
    const std::uint32_t destination = group_start_segment[right];
    if (destination == detail::kNoNode)
      return false;
    const std::uint32_t destination_lane =
        plan.segments[destination].lane.ordinal();
    const auto [begin, end] = group_uses(left);
    for (auto use = begin; use != end; ++use) {
      const std::uint32_t source = static_cast<std::uint32_t>(*use);
      if (source == destination)
        continue;
      const std::uint32_t source_lane = plan.segments[source].lane.ordinal();
      if (source_lane == destination_lane &&
          segment_sequence[source] < segment_sequence[destination]) {
        continue;
      }
      if (!precise_happens_before ||
          segment_sequence[source] >
              clocks[static_cast<std::size_t>(destination) * lane_count +
                     source_lane]) {
        return false;
      }
    }
    return true;
  };

  detail::ValueOrdinals groups(plan.allocator);
  if (!groups.try_reserve(request_count) ||
      !chain_head.try_reserve(request_count) ||
      !chain_tail.try_reserve(request_count) ||
      !chain_storage.try_reserve(request_count)) {
    return plan_allocation_failure();
  }
  for (std::uint32_t root = 0; root < request_count; ++root) {
    if (group_present[root] != 0 && !groups.try_push_back(root))
      return plan_allocation_failure();
  }
  std::sort(groups.begin(), groups.end(),
            [&](std::uint32_t left, std::uint32_t right) noexcept {
              if (group_begin[left] != group_begin[right])
                return group_begin[left] < group_begin[right];
              return left < right;
            });

  plan.workspace_resources.clear();
  plan.workspace_storage.clear();
  if (!plan.storage_workspace_slots.try_resize(plan.storage_domains.size()) ||
      !plan.workspace_resources.try_reserve(groups.size()) ||
      !plan.workspace_storage.try_reserve(groups.size())) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &slot : plan.storage_workspace_slots)
    slot = detail::kNoWorkspaceRequest;
  for (std::uint32_t root : groups) {
    std::uint32_t selected = detail::kNoWorkspaceRequest;
    for (std::size_t chain = 0; chain < chain_tail.size(); ++chain) {
      if (chain_storage[chain] == group_storage[root] &&
          group_before(chain_tail[chain], root)) {
        selected = static_cast<std::uint32_t>(chain);
        break;
      }
    }
    if (selected == detail::kNoWorkspaceRequest) {
      if (chain_tail.size() >= detail::kNoWorkspaceRequest ||
          !chain_head.try_push_back(root) || !chain_tail.try_push_back(root) ||
          !chain_storage.try_push_back(group_storage[root]) ||
          !plan.workspace_resources.try_push_back(
              plan.storage_domains[group_storage[root]].resource) ||
          !plan.workspace_storage.try_push_back(group_storage[root])) {
        return plan_allocation_failure();
      }
      selected = static_cast<std::uint32_t>(chain_tail.size() - 1);
      if (plan.storage_workspace_slots[group_storage[root]] ==
          detail::kNoWorkspaceRequest) {
        plan.storage_workspace_slots[group_storage[root]] = selected;
      }
    } else {
      chain_tail[selected] = root;
    }
    group_chain[root] = selected;
  }
  for (std::size_t index = 0; index < request_count; ++index)
    requests[index].resource_slot = group_chain[request_roots[index]];

  const std::span<const mem::MemoryResourceRef> selected_resources(
      plan.workspace_resources.data(), plan.workspace_resources.size());
  auto lifetime_plan = mem::LifetimePlanner::build(
      std::span<const mem::LifetimeRequest>(requests.data(), requests.size()),
      selected_resources, plan.allocator);
  if (!lifetime_plan)
    return std::move(lifetime_plan).error();

  plan.workspace_plan = std::move(*lifetime_plan);
  plan.workspace_bytes = 0;
  plan.workspace_backing_count = 0;
  U64s storage_peaks(plan.allocator);
  U64s chain_peaks(plan.allocator);
  if (!storage_peaks.try_resize(plan.storage_domains.size()) ||
      !chain_peaks.try_resize(chain_tail.size()))
    return plan_allocation_failure();
  for (std::uint64_t &peak : storage_peaks)
    peak = 0;
  for (std::uint64_t &peak : chain_peaks)
    peak = 0;
  if (!requests.empty()) {
    for (std::size_t index = 0; index < plan.workspace_plan.resource_count();
         ++index) {
      auto resource = plan.workspace_plan.resource_plan(index);
      if (!resource)
        return std::move(resource).error();
      plan.workspace_bytes =
          saturating_add(plan.workspace_bytes, resource->peak_bytes);
      plan.workspace_backing_count =
          saturating_add(plan.workspace_backing_count, resource->backing_count);
      chain_peaks[index] = resource->peak_bytes;
      const std::uint32_t storage = plan.workspace_storage[index];
      storage_peaks[storage] =
          saturating_add(storage_peaks[storage],
                         static_cast<std::uint64_t>(resource->peak_bytes));
    }
  }
  for (std::size_t storage = 0; storage < plan.storage_domains.size();
       ++storage) {
    if (storage_peaks[storage] <=
        plan.storage_domains[storage].workspace_budget_bytes)
      continue;
    if (plan.workspace_reuse != WorkspaceReusePolicy::kAllowSerialization) {
      rejected_plan = true;
      return Status::ResourceExhausted(
          "graph workspace budget is exceeded by the existing order");
    }
    if (!precise_happens_before) {
      rejected_plan = true;
      return Status::ResourceExhausted(
          "graph workspace serialization exceeds the bounded happens-before "
          "analysis");
    }

    auto command_identity = [&](std::uint32_t segment,
                                bool first) noexcept -> Result<std::uint32_t> {
      if (segment >= plan.segments.size()) {
        return std::unexpected(
            Status::Internal("graph serialization segment is out of range"));
      }
      const detail::PlanSegment &record = plan.segments[segment];
      const std::size_t end = record.step_begin + record.step_count;
      if (first) {
        for (std::size_t step = record.step_begin; step < end; ++step) {
          if (plan.steps[step].emits_command())
            return plan.steps[step].schedule_identity;
        }
      } else {
        for (std::size_t step = end; step > record.step_begin; --step) {
          if (plan.steps[step - 1].emits_command())
            return plan.steps[step - 1].schedule_identity;
        }
      }
      return std::unexpected(
          Status::Internal("graph serialization segment has no command"));
    };

    Words proposed(plan.allocator);
    Words best_edges(plan.allocator);
    auto propose_chain_order =
        [&](std::size_t source_chain,
            std::size_t destination_chain) noexcept -> Result<bool> {
      proposed.clear();
      const std::uint32_t source_root = chain_tail[source_chain];
      const std::uint32_t destination_root = chain_head[destination_chain];
      const std::uint32_t destination = group_start_segment[destination_root];
      auto destination_identity = command_identity(destination, true);
      if (!destination_identity)
        return std::unexpected(std::move(destination_identity).error());
      const auto [use_begin, use_end] = group_uses(source_root);
      for (auto use = use_begin; use != use_end; ++use) {
        const std::uint32_t source = static_cast<std::uint32_t>(*use);
        if (source == destination || segment_before(source, destination))
          continue;
        if (segment_before(destination, source))
          return false;
        auto source_identity = command_identity(source, false);
        if (!source_identity)
          return std::unexpected(std::move(source_identity).error());
        const std::uint64_t edge =
            (static_cast<std::uint64_t>(*source_identity) << 32) |
            *destination_identity;
        if (std::find(plan.serialization_edges.begin(),
                      plan.serialization_edges.end(),
                      edge) != plan.serialization_edges.end() ||
            std::find(proposed.begin(), proposed.end(), edge) !=
                proposed.end()) {
          continue;
        }
        if (!proposed.try_push_back(edge))
          return std::unexpected(plan_allocation_failure());
      }
      return !proposed.empty();
    };

    std::uint64_t best_saving = 0;
    for (std::size_t left = 0; left < chain_tail.size(); ++left) {
      if (chain_storage[left] != storage)
        continue;
      for (std::size_t right = left + 1; right < chain_tail.size(); ++right) {
        if (chain_storage[right] != storage)
          continue;
        const std::uint64_t saving =
            (std::min)(chain_peaks[left], chain_peaks[right]);
        if (saving < best_saving)
          continue;
        auto feasible = propose_chain_order(left, right);
        if (!feasible)
          return std::move(feasible).error();
        if (!*feasible) {
          feasible = propose_chain_order(right, left);
          if (!feasible)
            return std::move(feasible).error();
        }
        if (!*feasible)
          continue;
        if (saving == best_saving && !best_edges.empty())
          continue;
        if (!best_edges.try_resize(proposed.size()))
          return plan_allocation_failure();
        std::copy(proposed.begin(), proposed.end(), best_edges.begin());
        best_saving = saving;
      }
    }
    if (best_edges.empty()) {
      rejected_plan = true;
      return Status::ResourceExhausted(
          "graph workspace budget cannot be satisfied by safe bounded "
          "serialization");
    }
    if (plan.serialization_edges.size() > SIZE_MAX - best_edges.size() ||
        !plan.serialization_edges.try_reserve(plan.serialization_edges.size() +
                                              best_edges.size())) {
      return plan_allocation_failure();
    }
    for (const std::uint64_t edge : best_edges) {
      if (!plan.serialization_edges.try_push_back(edge))
        return plan_allocation_failure();
    }
    std::sort(plan.serialization_edges.begin(), plan.serialization_edges.end());
    plan.serialization_edge_count = plan.serialization_edges.size();
    reschedule_requested = true;
    return Status::Ok();
  }
  return Status::Ok();
}

// Only explicit local hard-constraint failures authorize another complete
// route. An allocation/provider/alias error with the same Status code does not.
Status finalize_placed_steps(detail::PlanState &plan,
                             const detail::GraphState &graph,
                             ResourceSearch &physical, CompileOptions options,
                             bool &search_limit, bool &rejected_plan) noexcept {
  rejected_plan = false;
  HYLE_RETURN_IF_ERROR(schedule_steps(plan, graph, physical, options,
                                      search_limit, rejected_plan));
  HYLE_RETURN_IF_ERROR(build_segments(plan, graph));
  if (plan.instance_values.size() > SIZE_MAX - plan.steps.size())
    return Status::ResourceExhausted(
        "graph workspace serialization round budget overflows");
  const std::size_t maximum_serialization_rounds =
      plan.instance_values.size() + plan.steps.size();
  std::size_t serialization_round = 0;
  for (;;) {
    reset_workspace_decisions(plan);
    bool reschedule_requested = false;
    HYLE_RETURN_IF_ERROR(
        plan_workspace(plan, graph, reschedule_requested, rejected_plan));
    if (!reschedule_requested)
      return Status::Ok();
    if (serialization_round++ >= maximum_serialization_rounds) {
      rejected_plan = true;
      return Status::ResourceExhausted(
          "graph workspace serialization exhausted its deterministic "
          "round budget");
    }
    HYLE_RETURN_IF_ERROR(schedule_steps(plan, graph, physical, options,
                                        search_limit, rejected_plan));
    HYLE_RETURN_IF_ERROR(build_segments(plan, graph));
  }
}

std::size_t step_constant_count(const detail::PlanState &plan,
                                const detail::PlanStep &step) noexcept {
  std::size_t count = 0;
  for (std::size_t index = 0; index < step.input_count; ++index) {
    const std::uint32_t ordinal = plan.node_inputs[step.input_begin + index];
    if ((plan.instance_roles[ordinal] & detail::kConstant) != 0)
      ++count;
  }
  return count;
}

Result<CompileReport>
build_report(const detail::PlanState &plan, const detail::GraphState &graph,
             std::span<const RouteDiagnostic> routes) noexcept {
  using ProviderTexts =
      container::FlatHashMap<std::uint64_t, compute::ProviderIdentity,
                             container::Hash<std::uint64_t>, std::equal_to<>,
                             mem::AllocatorRef>;
  ProviderTexts providers(plan.allocator);
  std::size_t allocation_bytes = sizeof(detail::ReportState);
  for (const detail::PlanStep &step : plan.steps) {
    const auto identity = step.provider();
    if (!identity.valid())
      continue;
    const auto instance = step.provider_instance_id();
    if (!instance.valid())
      return std::unexpected(
          Status::Internal("graph report provider has no instance identity"));
    auto [entry, result] = providers.try_emplace(instance.value(), identity);
    if (entry == nullptr)
      return std::unexpected(plan_allocation_failure());
    if (result != ProviderTexts::InsertResult::kInserted)
      continue;
    for (const auto text : {entry->name, entry->version}) {
      if (text.size() > SIZE_MAX - allocation_bytes)
        return std::unexpected(Status::ResourceExhausted(
            "graph report provider text size overflows"));
      allocation_bytes += text.size();
    }
  }
  void *storage =
      plan.allocator.allocate(allocation_bytes, alignof(detail::ReportState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *state =
      ::new (storage) detail::ReportState(plan.allocator, allocation_bytes);
  auto *text_cursor = reinterpret_cast<char *>(state + 1);
  providers.for_each([&](std::uint64_t,
                         compute::ProviderIdentity &identity) noexcept {
    auto copy_text = [&](std::string_view text) noexcept -> std::string_view {
      if (text.empty())
        return {};
      std::memcpy(text_cursor, text.data(), text.size());
      const std::string_view copy(text_cursor, text.size());
      text_cursor += text.size();
      return copy;
    };
    identity.name = copy_text(identity.name);
    identity.version = copy_text(identity.version);
  });
  state->search = plan.search;
  if (!state->nodes.try_resize(graph.nodes.size()) ||
      !state->steps.try_resize(plan.steps.size()) ||
      !state->targets.try_resize(plan.targets.size()) ||
      !state->storage_domains.try_resize(plan.storage_domains.size()) ||
      !state->lanes.try_resize(plan.lanes.size()) ||
      !state->segments.try_resize(plan.segments.size()) ||
      !state->physical_values.try_resize(plan.instance_values.size()) ||
      !state->constants.try_resize(plan.constant_ids.size()) ||
      !state->step_inputs.try_resize(plan.node_inputs.size()) ||
      !state->step_outputs.try_resize(plan.node_outputs.size()) ||
      !state->step_nodes.try_resize(plan.step_nodes.size()) ||
      !state->step_node_begins.try_resize(plan.steps.size()) ||
      !state->segment_dependencies.try_resize(
          plan.segment_dependencies.size())) {
    detail::release(state);
    return std::unexpected(plan_allocation_failure());
  }
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    const auto &node = graph.nodes[index];
    state->nodes[index] = CompileNodeReport{
        .ordinal = static_cast<std::uint32_t>(index),
        .kind = node.kind(),
        .traits = node.traits(),
    };
  }
  std::copy(plan.node_inputs.begin(), plan.node_inputs.end(),
            state->step_inputs.begin());
  std::copy(plan.node_outputs.begin(), plan.node_outputs.end(),
            state->step_outputs.begin());
  for (std::size_t index = 0; index < plan.step_nodes.size(); ++index)
    state->step_nodes[index] =
        detail::GraphIdentity::node(graph.generation, plan.step_nodes[index]);
  std::copy(plan.segment_dependencies.begin(), plan.segment_dependencies.end(),
            state->segment_dependencies.begin());
  for (std::size_t index = 0; index < plan.targets.size(); ++index) {
    const auto &source = plan.targets[index].source;
    state->targets[index] = CompileTargetReport{
        .target = ExecutionTargetId{static_cast<std::uint32_t>(index)},
        .backend = source.backend_id(),
        .device_class = source.device_class(),
        .default_storage = plan.targets[index].default_storage,
    };
  }
  for (std::size_t index = 0; index < plan.storage_domains.size(); ++index) {
    state->storage_domains[index] = CompileStorageReport{
        .storage = StorageDomainId{static_cast<std::uint32_t>(index)},
        .properties = plan.storage_domains[index].resource.properties(),
        .workspace_budget_bytes =
            plan.storage_domains[index].workspace_budget_bytes,
    };
    for (std::size_t workspace_slot = 0;
         workspace_slot < plan.workspace_storage.size(); ++workspace_slot) {
      if (plan.workspace_storage[workspace_slot] != index)
        continue;
      auto resource = plan.workspace_plan.resource_plan(workspace_slot);
      if (!resource) {
        detail::release(state);
        return std::unexpected(std::move(resource).error());
      }
      state->storage_domains[index].workspace_bytes = saturating_add(
          state->storage_domains[index].workspace_bytes, resource->peak_bytes);
      state->storage_domains[index].workspace_backing_count =
          saturating_add(state->storage_domains[index].workspace_backing_count,
                         resource->backing_count);
    }
  }
  for (std::size_t index = 0; index < plan.lanes.size(); ++index) {
    const detail::ExecutionLaneRecord &lane = plan.lanes[index];
    const auto &source = plan.targets[lane.target.ordinal()].source;
    auto properties = source.device_properties();
    if (!properties) {
      detail::release(state);
      return std::unexpected(std::move(properties).error());
    }
    Result<bool> capture = false;
    auto retained = detail::ExecutionSourceAccess::compute(source);
    if (retained) {
      capture = (*retained)->has_backend_extension(
          compute::detail::kCommandCaptureExtensionId);
    } else if (!retained.error().IsUnimplemented()) {
      capture = std::unexpected(std::move(retained).error());
    }
    if (!capture) {
      detail::release(state);
      return std::unexpected(std::move(capture).error());
    }
    state->lanes[index] = CompileLaneReport{
        .lane = ExecutionLaneId{static_cast<std::uint32_t>(index)},
        .target = lane.target,
        .role = lane.role,
        .runtime_mode = lane.runtime_mode,
        .completion_slots_per_invocation =
            lane.runtime_mode == compute::ProviderRuntimeMode::kSharedQueue
                ? plan.lane_segment_counts[index]
                : 0,
        .queue_required =
            lane.runtime_mode == compute::ProviderRuntimeMode::kSharedQueue &&
            plan.lane_segment_counts[index] != 0,
        .asynchronous_queue =
            lane.runtime_mode == compute::ProviderRuntimeMode::kSharedQueue &&
            properties->asynchronous_queues,
        .capture_supported =
            lane.runtime_mode == compute::ProviderRuntimeMode::kSharedQueue &&
            *capture,
    };
    if (plan.lane_segment_counts[index] == 0) {
      continue;
    }
    if (!state->backend.valid() && !state->mixed_backends) {
      state->backend = source.backend_id();
    } else if (!state->mixed_backends &&
               state->backend != source.backend_id()) {
      state->backend = {};
      state->mixed_backends = true;
    }
  }
  state->workspace_bytes = plan.workspace_bytes;
  state->workspace_backing_count = plan.workspace_backing_count;
  state->specialization = plan.specialization;
  state->is_shape_domain = plan.domain_graph.valid();
  state->finite_only = plan.finite_only;
  state->node_count = plan.logical_node_count;
  state->segment_count = plan.segments.size();
  state->segment_dependency_count = plan.segment_dependencies.size();
  state->serialization_edge_count = plan.serialization_edge_count;
  state->region_count = plan.region_count;
  state->constant_count = plan.constant_ids.size();
  state->supports_concurrent_invocations = plan.supports_concurrent_invocations;
  for (std::size_t slot = 0; slot < plan.constant_ids.size(); ++slot) {
    const auto value = plan.constant_ids[slot];
    const auto *buffer = plan.owned_constant(slot);
    const bool owned = buffer != nullptr;
    state->constants[slot] = CompileConstantReport{
        .value = value,
        .policy =
            owned ? ConstantPolicy::kMaterialize : ConstantPolicy::kBorrow,
        .storage = StorageDomainId{plan.external_storage[value.ordinal()]},
        .description = plan.constant_bindings[slot].description()};
    state->borrowed_constant_count += owned ? 0u : 1u;
    state->materialized_constant_count += owned ? 1u : 0u;
  }
  for (const auto &owner : plan.canonical_constants) {
    if (!owner || !owner->ready)
      continue;
    const auto slot = constant_slot(plan, owner->key.constant);
    if (slot == SIZE_MAX) {
      detail::release(state);
      return std::unexpected(Status::Internal("canonical owner has no logical constant"));
    }
    auto &bytes = state->constants[slot].materialized_bytes;
    bytes = saturating_add(bytes, owner->buffer.size());
    state->materialized_constant_bytes = saturating_add(
        state->materialized_constant_bytes, owner->buffer.size());
  }

  for (std::size_t index = 0; index < plan.instance_values.size(); ++index) {
    const std::uint32_t logical_value = plan.instance_values[index];
    const std::uint8_t roles = plan.instance_roles[index];
    CompilePhysicalValueReport physical{
        .value = detail::value_id(graph, logical_value),
        .storage = StorageDomainId{plan.instance_storage[index]},
        .description = plan.instance_descriptions[index],
        .external_input = (roles & detail::kExternalInput) != 0,
        .external_output = (roles & detail::kExternalOutput) != 0,
        .constant = (roles & detail::kConstant) != 0,
        .invocation_owned_output =
            (roles & detail::kInvocationOwnedOutput) != 0,
        .workspace_backed =
            plan.workspace_requests[index] != detail::kNoWorkspaceRequest,
    };
    if (plan.alias_sources[index] != detail::kNoInPlaceSource)
      physical.alias_source = plan.alias_sources[index];
    state->physical_values[index] = physical;
  }

  for (std::size_t index = 0; index < plan.segments.size(); ++index) {
    const detail::PlanSegment &segment = plan.segments[index];
    bool contains_commands = false;
    const std::size_t step_end = segment.step_begin + segment.step_count;
    for (std::size_t step = segment.step_begin; step < step_end; ++step)
      contains_commands = contains_commands || plan.steps[step].emits_command();
    const bool capture_compatible =
        contains_commands && segment_capture_compatible(plan, segment);
    const bool capture_supported =
        capture_compatible &&
        state->lanes[segment.lane.ordinal()].capture_supported;
    state->segments[index] = CompileSegmentReport{
        .lane = segment.lane,
        .runtime_mode = plan.lanes[segment.lane.ordinal()].runtime_mode,
        .step_begin = segment.step_begin,
        .step_count = segment.step_count,
        .dependency_begin = segment.dependency_begin,
        .dependency_count = segment.dependency_count,
        .terminal = std::binary_search(plan.terminal_segments.begin(),
                                       plan.terminal_segments.end(),
                                       static_cast<std::uint32_t>(index)),
        .contains_commands = contains_commands,
        .capture_compatible = capture_compatible,
        .capture_supported = capture_supported,
    };
    state->command_segment_count += contains_commands ? 1u : 0u;
    state->delegated_segment_count +=
        contains_commands && plan.lanes[segment.lane.ordinal()].runtime_mode ==
                                 compute::ProviderRuntimeMode::kDelegated
            ? 1u
            : 0u;
    state->capturable_segment_count += capture_supported ? 1u : 0u;
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;
    for (std::size_t dependency = segment.dependency_begin;
         dependency < dependency_end; ++dependency) {
      const std::uint32_t source = plan.segment_dependencies[dependency];
      const std::uint32_t source_lane = plan.segments[source].lane.ordinal();
      if (source_lane != segment.lane.ordinal() &&
          (state->lanes[source_lane].runtime_mode ==
               compute::ProviderRuntimeMode::kDelegated ||
           state->lanes[source_lane].asynchronous_queue)) {
        state->synchronization_count =
            saturating_add(state->synchronization_count, std::uint32_t{1});
      }
    }
  }

  for (ValueId output : plan.output_ids) {
    if ((plan.binding_roles[output.ordinal()] &
         detail::kInvocationOwnedOutput) != 0) {
      ++state->invocation_owned_output_count;
    }
  }

  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const detail::PlanStep &step = plan.steps[index];
    state->step_node_begins[index] = step.node_set_begin;
    CompileStepReport step_report{
        .kind = step.kind == detail::PlanStepKind::kRegion
                    ? CompileStepKind::kProviderRegion
                    : CompileStepKind::kOperationFallback,
        .fallback_reason = step.fallback_reason,
        .provider = step.provider(),
        .provider_instance = step.provider_instance_id(),
        .target = step.target,
        .lane = step.lane,
        .workspace_storage = step.workspace_storage,
        .source_storage = step.source_storage,
        .destination_storage = step.destination_storage,
        .node_begin = step.node_begin,
        .node_count = step.node_count,
        .nodes_contiguous = step.nodes_contiguous,
        .workspace = step.reported_workspace_requirement(),
        .borrowed_constant_count = step_constant_count(plan, step),
        .dispatch_count = step.emits_command() ? 1u : 0u,
        .input_begin = step.input_begin,
        .output_begin = step.output_begin,
        .input_count = step.input_count,
        .output_count = step.output_count,
        .runtime = step.runtime_properties(),
        .implementation = step.kind == detail::PlanStepKind::kCopy
                              ? compute::ProviderImplementationId{}
                              : routes[step.node_begin].implementation,
        .policy = step.kind == detail::PlanStepKind::kCopy
                      ? compute::RoutePolicyId{}
                      : routes[step.node_begin].policy,
        .preference = step.kind == detail::PlanStepKind::kCopy
                          ? 0
                          : routes[step.node_begin].preference,
    };
    if (step_report.provider.valid()) {
      const auto *owned = providers.find(step_report.provider_instance.value());
      if (owned == nullptr) {
        detail::release(state);
        return std::unexpected(
            Status::Internal("graph report provider text is missing"));
      }
      step_report.provider = *owned;
    }
    if (step.kind == detail::PlanStepKind::kCopy) {
      step_report.kind = CompileStepKind::kCopy;
      step_report.persistent_bytes = step.region.valid()
          ? step.region.properties().persistent_bytes : std::optional<std::size_t>{0};
      step_report.copy_bytes = step.copy_bytes;
      state->copy_bytes = saturating_add(state->copy_bytes, step.copy_bytes);
    }
    if (step.kind == detail::PlanStepKind::kView) {
      step_report.kind = CompileStepKind::kView;
      step_report.persistent_bytes = 0;
      step_report.implementation = {};
    }
    if (step.kind == detail::PlanStepKind::kRegion) {
      const RegionArtifactProperties properties = step.region.properties();
      if (properties.packed_constant_count >
          step_report.borrowed_constant_count) {
        detail::release(state);
        return std::unexpected(Status::Internal(
            "graph region packed more constants than its boundary contains"));
      }
      step_report.persistent_bytes = properties.persistent_bytes;
      step_report.packed_constant_count = properties.packed_constant_count;
      step_report.packed_constant_bytes = properties.packed_constant_bytes;
      step_report.copy_bytes = properties.copy_bytes;
      step_report.synchronization_count = properties.synchronization_count;
      step_report.dispatch_count = properties.dispatch_count;
      state->packed_constant_count = saturating_add(
          state->packed_constant_count, step_report.packed_constant_count);
      state->packed_constant_bytes = saturating_add(
          state->packed_constant_bytes, step_report.packed_constant_bytes);
      state->copy_bytes =
          saturating_add(state->copy_bytes, step_report.copy_bytes);
      state->synchronization_count = saturating_add(
          state->synchronization_count, step_report.synchronization_count);
    } else if (step.kind == detail::PlanStepKind::kOperation) {
      ++state->fallback_operation_count;
      step_report.persistent_bytes = step.operation.persistent_bytes();
      step_report.dispatch_count = routes[step.node_begin].dispatch_count.value_or(
          step_report.dispatch_count);
    }
    state->steps[index] = step_report;

    if (!state->mixed_providers && step_report.provider.valid()) {
      if (!state->provider.valid()) {
        state->provider = step_report.provider;
      } else if (state->provider.id != step_report.provider.id ||
                 state->provider.version != step_report.provider.version) {
        state->provider = {};
        state->mixed_providers = true;
      }
    }
  }
  for (const auto &artifact : plan.constant_artifacts) {
    state->packed_constant_count =
        saturating_add(state->packed_constant_count, std::size_t{1});
    state->packed_constant_bytes =
        saturating_add(state->packed_constant_bytes, artifact->packed_bytes);
  }
  return detail::CompileReportAccess::adopt(state);
}

} // namespace

namespace {
void invalidate_recording(detail::InvocationState &invocation,
                          std::uint32_t selected_step, bool recycle) noexcept {
  HYLE_FATAL_ASSERT(selected_step < invocation.step_bindings.size());
  auto &binding = invocation.step_bindings[selected_step];
  // Saturation cannot preserve readiness: every changed binding must be
  // prepared again, even when a retired physical recording can stay dormant.
  if (binding.revision != UINT64_MAX)
    ++binding.revision;
  binding.prepared_revision = 0;
  // Explicit native/configuration/Queue invalidation also drops dormant
  // allocations before the original provider releases its mutable resources.
  if (!recycle)
    for (auto &saved : invocation.capture_rebuilds)
      if (selected_step >= saved.step_begin && selected_step - saved.step_begin < saved.step_count) {
        saved.storage.reset();
        discard_capture_variants(saved);
      }
  for (std::size_t index = 0; index < invocation.captured_ranges.size();) {
    const auto &range = invocation.captured_ranges[index];
    if (selected_step < range.step_begin ||
        selected_step - range.step_begin >= range.step_count) {
      ++index;
      continue;
    }
    bool ordinary = recycle;
    for (std::size_t step = range.step_begin;
         ordinary && step < range.step_begin + range.step_count; ++step) {
      const auto &planned = invocation.plan->steps[step];
      if (!planned.emits_command()) continue;
      const auto &cached = invocation.step_bindings[step];
      ordinary = cached.stable_binding_identity && !cached.native_inputs &&
          (!planned.region.valid() || detail::RegionAccess::borrows_tensor_views_only(planned.region));
    }
    if (auto *saved = capture_storage(invocation, range.step_begin, range.step_count)) {
      if (ordinary && saved->preserves_private_arguments && saved->active_variant < 2 &&
          saved->variants[saved->active_variant].key_valid) {
        auto &variant = saved->variants[saved->active_variant];
        HYLE_FATAL_ASSERT(!variant.commands.valid());
        variant.commands = std::move(invocation.captured_ranges[index].commands);
        saved->active_variant = 2;
      } else if (ordinary) {
        saved->storage = compute::detail::CommandCaptureAccess::recycle_after_retirement(
            std::move(invocation.captured_ranges[index].commands));
        saved->active_variant = 2;
      } else {
        saved->storage.reset();
        discard_capture_variants(*saved);
      }
    }
    // A positive private-resource contract keeps only retired commands and
    // borrowed keys dormant. Other artifacts are destroyed or permanently
    // stripped of launch authority before mutable provider resources change.
    for (std::size_t next = index + 1;
         next < invocation.captured_ranges.size(); ++next)
      invocation.captured_ranges[next - 1] =
          std::move(invocation.captured_ranges[next]);
    invocation.captured_ranges.pop_back();
  }
  for (auto &span : invocation.capture_spans)
    span = {};
  invalidate_binding_cache(invocation);
}
} // namespace

void detail::invalidate_step_recording(InvocationState &invocation,
                                      std::uint32_t selected_step) noexcept {
  invalidate_recording(invocation, selected_step, false);
}

void detail::recycle_step_recording(InvocationState &invocation,
                                   std::uint32_t selected_step) noexcept {
  invalidate_recording(invocation, selected_step, true);
}

Status detail::collect_effect_bindings(
    const Graph &graph, std::span<const EffectResourceBinding> bindings,
    std::span<const ConstantBinding> constants, mem::AllocatorRef allocator, EffectBindings &output) noexcept {
  const auto *state = GraphAccess::state(graph);
  if (state == nullptr || !allocator.valid())
    return Status::InvalidArgument("effect graph or allocator is invalid");
  return graph::collect_effect_bindings(*state, bindings, constants, allocator, output);
}

Status detail::validate_effect_tensor_alias(std::span<const EffectResourceBinding> bindings,
                                            mem::ConstBufferView tensor) noexcept {
  for (const auto &binding : bindings)
    if (binding.footprint.is_memory() && overlaps(byte_range(binding.footprint.memory_range()), byte_range(tensor)))
      return Status::InvalidArgument("effect resource memory must not alias graph tensor storage");
  return Status::Ok();
}

Result<ExactGraph> specialize_exact(const Graph &graph,
                                    std::span<const ShapeBinding> shape,
                                    mem::AllocatorRef allocator) noexcept {
  const auto *logical = detail::GraphAccess::state(graph);
  if (logical == nullptr || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("exact graph or allocator is invalid"));
  ShapeValues values(allocator);
  HYLE_RETURN_IF_ERROR(bind_shape_values(*logical, shape, allocator, values));
  container::SmallVector<compute::TensorDesc, 8, mem::AllocatorRef> descriptions(allocator);
  HYLE_ASSIGN_OR_RETURN(auto signature,
                        specialize_bound_descriptions(*logical, values, allocator, descriptions));
  HYLE_ASSIGN_OR_RETURN(auto concrete,
                        make_specialized_graph(*logical, descriptions, allocator, values, false));
  return ExactGraph{std::move(concrete), std::move(signature)};
}

Result<bool> is_region_contractible(const Graph &graph,
                                    std::span<const NodeId> nodes,
                                    mem::AllocatorRef allocator) noexcept {
  const detail::GraphState *state = detail::GraphAccess::state(graph);
  if (state == nullptr || !allocator.valid())
    return std::unexpected(
        Status::InvalidArgument("region query owner is invalid"));
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> selected(
      allocator);
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> reached(
      allocator);
  return validate_contractible_region(*state, nodes, selected, reached);
}

SpecializationSignature::SpecializationSignature(
    const SpecializationSignature &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

SpecializationSignature &SpecializationSignature::operator=(
    const SpecializationSignature &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

SpecializationSignature::SpecializationSignature(
    SpecializationSignature &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

SpecializationSignature &
SpecializationSignature::operator=(SpecializationSignature &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

SpecializationSignature::~SpecializationSignature() { detail::release(state_); }

std::size_t SpecializationSignature::binding_count() const noexcept {
  return state_ != nullptr ? state_->bindings.size() : 0;
}

Result<ShapeBinding>
SpecializationSignature::binding(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("specialization signature is invalid"));
  }
  if (index >= state_->bindings.size()) {
    return std::unexpected(Status::OutOfRange(
        "specialization signature binding index is out of range"));
  }
  return state_->bindings[index];
}

std::size_t SpecializationSignature::hash() const noexcept {
  return state_ != nullptr ? state_->hash : 0;
}

bool SpecializationSignature::equals(
    const SpecializationSignature &other) const noexcept {
  if (state_ == other.state_)
    return true;
  if (state_ == nullptr || other.state_ == nullptr ||
      state_->generation != other.state_->generation ||
      state_->bindings.size() != other.state_->bindings.size()) {
    return false;
  }
  for (std::size_t index = 0; index < state_->bindings.size(); ++index) {
    if (state_->bindings[index] != other.state_->bindings[index])
      return false;
  }
  return true;
}

CompileReport::CompileReport(const CompileReport &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

CompileReport &CompileReport::operator=(const CompileReport &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

CompileReport::CompileReport(CompileReport &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

CompileReport &CompileReport::operator=(CompileReport &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

CompileReport::~CompileReport() { detail::release(state_); }

compute::BackendId CompileReport::backend() const noexcept {
  return state_ != nullptr ? state_->backend : compute::BackendId{};
}

compute::ProviderIdentity CompileReport::provider() const noexcept {
  return state_ != nullptr ? state_->provider : compute::ProviderIdentity{};
}

CompileSearchReport CompileReport::search() const noexcept {
  return state_ != nullptr ? state_->search : CompileSearchReport{};
}

CompileSearchReport CompileReport::total_search() const noexcept {
  return state_ != nullptr ? state_->total_search : CompileSearchReport{};
}

std::size_t CompileReport::exact_variant_count() const noexcept {
  return state_ != nullptr ? state_->exact_variants.size() : 0;
}

Result<CompileReport>
CompileReport::exact_variant(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  if (index >= state_->exact_variants.size())
    return std::unexpected(
        Status::OutOfRange("exact variant index is out of range"));
  return state_->exact_variants[index];
}

std::size_t CompileReport::workspace_bytes() const noexcept {
  return state_ != nullptr ? state_->workspace_bytes : 0;
}

std::size_t CompileReport::workspace_backing_count() const noexcept {
  return state_ != nullptr ? state_->workspace_backing_count : 0;
}

std::size_t CompileReport::node_count() const noexcept {
  return state_ != nullptr ? state_->node_count : 0;
}

Result<CompileNodeReport>
CompileReport::node(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  if (index >= state_->nodes.size())
    return std::unexpected(Status::OutOfRange("compile report node index"));
  return state_->nodes[index];
}

std::size_t CompileReport::target_count() const noexcept {
  return state_ != nullptr ? state_->targets.size() : 0;
}

std::size_t CompileReport::storage_domain_count() const noexcept {
  return state_ != nullptr ? state_->storage_domains.size() : 0;
}

std::size_t CompileReport::lane_count() const noexcept {
  return state_ != nullptr ? state_->lanes.size() : 0;
}

std::size_t CompileReport::step_count() const noexcept {
  return state_ != nullptr ? state_->steps.size() : 0;
}

std::size_t CompileReport::segment_count() const noexcept {
  return state_ != nullptr ? state_->segment_count : 0;
}

std::size_t CompileReport::segment_dependency_count() const noexcept {
  return state_ != nullptr ? state_->segment_dependency_count : 0;
}

std::size_t CompileReport::command_segment_count() const noexcept {
  return state_ != nullptr ? state_->command_segment_count : 0;
}

std::size_t CompileReport::delegated_segment_count() const noexcept {
  return state_ != nullptr ? state_->delegated_segment_count : 0;
}

std::size_t CompileReport::serialization_edge_count() const noexcept {
  return state_ != nullptr ? state_->serialization_edge_count : 0;
}

std::size_t CompileReport::capturable_segment_count() const noexcept {
  return state_ != nullptr ? state_->capturable_segment_count : 0;
}

std::size_t CompileReport::physical_value_count() const noexcept {
  return state_ != nullptr ? state_->physical_values.size() : 0;
}

std::size_t CompileReport::region_count() const noexcept {
  return state_ != nullptr ? state_->region_count : 0;
}

std::size_t CompileReport::fallback_operation_count() const noexcept {
  return state_ != nullptr ? state_->fallback_operation_count : 0;
}

std::size_t CompileReport::constant_count() const noexcept {
  return state_ != nullptr ? state_->constant_count : 0;
}

std::size_t CompileReport::borrowed_constant_count() const noexcept {
  return state_ != nullptr ? state_->borrowed_constant_count : 0;
}

std::size_t CompileReport::materialized_constant_count() const noexcept {
  return state_ != nullptr ? state_->materialized_constant_count : 0;
}

std::size_t CompileReport::materialized_constant_bytes() const noexcept {
  return state_ != nullptr ? state_->materialized_constant_bytes : 0;
}

CompiledStorageReport CompileReport::compiled_storage() const noexcept {
  return state_ != nullptr ? state_->compiled_storage : CompiledStorageReport{};
}

CompiledStorageReport CompileReport::total_compiled_storage() const noexcept {
  return state_ != nullptr ? state_->total_compiled_storage
                           : CompiledStorageReport{};
}

std::optional<CompilationStorageReport>
CompileReport::compilation_storage() const noexcept {
  return state_ != nullptr ? state_->compilation_storage : std::nullopt;
}

Result<CompileConstantReport>
CompileReport::constant(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  if (index >= state_->constants.size())
    return std::unexpected(
        Status::OutOfRange("graph constant report index is out of range"));
  return state_->constants[index];
}

std::size_t CompileReport::packed_constant_count() const noexcept {
  return state_ != nullptr ? state_->packed_constant_count : 0;
}

std::size_t CompileReport::packed_constant_bytes() const noexcept {
  return state_ != nullptr ? state_->packed_constant_bytes : 0;
}

std::size_t CompileReport::invocation_owned_output_count() const noexcept {
  return state_ != nullptr ? state_->invocation_owned_output_count : 0;
}

std::size_t CompileReport::copy_bytes() const noexcept {
  return state_ != nullptr ? state_->copy_bytes : 0;
}

std::uint32_t CompileReport::synchronization_count() const noexcept {
  return state_ != nullptr ? state_->synchronization_count : 0;
}

bool CompileReport::supports_concurrent_invocations() const noexcept {
  return state_ != nullptr && state_->supports_concurrent_invocations;
}

bool CompileReport::mixed_providers() const noexcept {
  return state_ != nullptr && state_->mixed_providers;
}

bool CompileReport::mixed_backends() const noexcept {
  return state_ != nullptr && state_->mixed_backends;
}

bool CompileReport::is_shape_domain() const noexcept {
  return state_ != nullptr && state_->is_shape_domain;
}

bool CompileReport::has_execution_route() const noexcept {
  return state_ != nullptr && !state_->finite_only;
}

SpecializationSignature CompileReport::specialization() const noexcept {
  return state_ != nullptr ? state_->specialization : SpecializationSignature{};
}

Result<CompileTargetReport>
CompileReport::target(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->targets.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report target index is out of range"));
  }
  return state_->targets[index];
}

Result<CompileStorageReport>
CompileReport::storage_domain(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->storage_domains.size()) {
    return std::unexpected(Status::OutOfRange(
        "compile report storage domain index is out of range"));
  }
  return state_->storage_domains[index];
}

Result<CompileLaneReport>
CompileReport::lane(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->lanes.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report lane index is out of range"));
  }
  return state_->lanes[index];
}

Result<CompileSegmentReport>
CompileReport::segment(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->segments.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report segment index is out of range"));
  }
  return state_->segments[index];
}

Result<CompileSegmentDependencyReport>
CompileReport::segment_dependency(std::size_t segment_index,
                                  std::size_t dependency_index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (segment_index >= state_->segments.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report segment index is out of range"));
  }
  const CompileSegmentReport &segment = state_->segments[segment_index];
  if (dependency_index >= segment.dependency_count) {
    return std::unexpected(Status::OutOfRange(
        "compile report segment dependency index is out of range"));
  }
  const std::uint32_t source =
      state_->segment_dependencies[segment.dependency_begin + dependency_index];
  const ExecutionLaneId source_lane = state_->segments[source].lane;
  CompileDependencyRoute route = CompileDependencyRoute::kOrderedQueue;
  if (source_lane != segment.lane) {
    const CompileLaneReport &source_report =
        state_->lanes[source_lane.ordinal()];
    route = (source_report.runtime_mode ==
                 compute::ProviderRuntimeMode::kDelegated ||
             source_report.asynchronous_queue)
                ? CompileDependencyRoute::kQueueBindingDependent
                : CompileDependencyRoute::kSourceCompletesInline;
  }
  return CompileSegmentDependencyReport{
      .source_segment = source,
      .destination_segment = static_cast<std::uint32_t>(segment_index),
      .route = route,
  };
}

Result<CompilePhysicalValueReport>
CompileReport::physical_value(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->physical_values.size()) {
    return std::unexpected(Status::OutOfRange(
        "compile report physical value index is out of range"));
  }
  return state_->physical_values[index];
}

Result<std::uint32_t>
CompileReport::step_input(std::size_t step_index,
                          std::size_t input_index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (step_index >= state_->steps.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report step index is out of range"));
  }
  const CompileStepReport &step = state_->steps[step_index];
  if (input_index >= step.input_count) {
    return std::unexpected(
        Status::OutOfRange("compile report step input index is out of range"));
  }
  return state_->step_inputs[step.input_begin + input_index];
}

Result<std::uint32_t>
CompileReport::step_output(std::size_t step_index,
                           std::size_t output_index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (step_index >= state_->steps.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report step index is out of range"));
  }
  const CompileStepReport &step = state_->steps[step_index];
  if (output_index >= step.output_count) {
    return std::unexpected(
        Status::OutOfRange("compile report step output index is out of range"));
  }
  return state_->step_outputs[step.output_begin + output_index];
}

Result<NodeId> CompileReport::step_node(std::size_t step_index,
                                        std::size_t node_index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (step_index >= state_->steps.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report step index is out of range"));
  }
  const CompileStepReport &step = state_->steps[step_index];
  if (node_index >= step.node_count) {
    return std::unexpected(
        Status::OutOfRange("compile report step node index is out of range"));
  }
  return state_->step_nodes[state_->step_node_begins[step_index] + node_index];
}

Result<CompileStepReport>
CompileReport::step(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->steps.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report step index is out of range"));
  }
  return state_->steps[index];
}

Executable::Executable(const Executable &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

Executable &Executable::operator=(const Executable &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

Executable::Executable(Executable &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Executable &Executable::operator=(Executable &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Executable::~Executable() { detail::release(state_); }

compute::WorkloadContext Executable::workload() const noexcept {
  return state_ != nullptr ? state_->workload : compute::WorkloadContext{};
}

std::size_t Executable::input_count() const noexcept {
  return state_ != nullptr ? state_->input_ids.size() : 0;
}

std::size_t Executable::output_count() const noexcept {
  return state_ != nullptr ? state_->output_ids.size() : 0;
}

std::size_t Executable::node_count() const noexcept {
  return state_ != nullptr ? state_->logical_node_count : 0;
}

std::size_t Executable::target_count() const noexcept {
  return state_ != nullptr ? state_->targets.size() : 0;
}

std::size_t Executable::storage_domain_count() const noexcept {
  return state_ != nullptr ? state_->storage_domains.size() : 0;
}

std::size_t Executable::lane_count() const noexcept {
  return state_ != nullptr ? state_->lanes.size() : 0;
}

std::size_t Executable::step_count() const noexcept {
  return state_ != nullptr ? state_->steps.size() : 0;
}

std::size_t Executable::segment_count() const noexcept {
  return state_ != nullptr ? state_->segments.size() : 0;
}

std::size_t Executable::segment_dependency_count() const noexcept {
  return state_ != nullptr ? state_->segment_dependencies.size() : 0;
}

std::size_t Executable::required_queue_count() const noexcept {
  return state_ != nullptr ? state_->required_queue_count : 0;
}

Result<bool> Executable::requires_queue(ExecutionLaneId lane) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (!lane.valid() || lane.ordinal() >= state_->lanes.size()) {
    return std::unexpected(
        Status::OutOfRange("execution lane is out of range"));
  }
  return state_->lane_segment_counts[lane.ordinal()] != 0 &&
         state_->lanes[lane.ordinal()].runtime_mode ==
             compute::ProviderRuntimeMode::kSharedQueue;
}

std::size_t Executable::region_count() const noexcept {
  return state_ != nullptr ? state_->region_count : 0;
}

std::size_t Executable::workspace_bytes() const noexcept {
  return state_ != nullptr ? state_->workspace_bytes : 0;
}

std::size_t Executable::workspace_backing_count() const noexcept {
  return state_ != nullptr ? state_->workspace_backing_count : 0;
}

bool Executable::supports_concurrent_invocations() const noexcept {
  return state_ != nullptr && state_->supports_concurrent_invocations;
}

bool Executable::is_shape_domain() const noexcept {
  return state_ != nullptr && state_->domain_graph.valid();
}

bool Executable::has_execution_route() const noexcept {
  return state_ != nullptr && !state_->finite_only;
}

SpecializationSignature Executable::specialization() const noexcept {
  return state_ != nullptr ? state_->specialization : SpecializationSignature{};
}

Result<ValueId> Executable::input_id(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (index >= state_->input_ids.size()) {
    return std::unexpected(
        Status::OutOfRange("executable input index is out of range"));
  }
  return state_->input_ids[index];
}

Result<ValueId> Executable::output_id(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (index >= state_->output_ids.size()) {
    return std::unexpected(
        Status::OutOfRange("executable output index is out of range"));
  }
  return state_->output_ids[index];
}

Result<OutputOwnership>
Executable::output_ownership(std::size_t index) const noexcept {
  auto value = output_id(index);
  if (!value)
    return std::unexpected(std::move(value).error());
  if (state_->finite_only)
    return std::unexpected(Status::FailedPrecondition(
        "finite-exact output ownership requires a selected route"));
  const std::uint8_t role = state_->binding_roles[value->ordinal()];
  if ((role & detail::kInvocationOwnedOutput) != 0)
    return OutputOwnership::kInvocation;
  if ((role & (detail::kSourceAliasedOutput | detail::kExternalInput |
               detail::kConstant)) != 0) {
    if (!state_->canonical_constants.empty()) {
      const auto instance = state_->external_instances[value->ordinal()];
      if (state_->owned_instance_constant(alias_root(*state_, instance)) != nullptr)
        return OutputOwnership::kExecutable;
    }
    return OutputOwnership::kSourceAlias;
  }
  return OutputOwnership::kCaller;
}

std::size_t Executable::exact_variant_count() const noexcept {
  return state_ != nullptr ? state_->exact_variants.size() : 0;
}

std::optional<InvocationStorageUsage>
Executable::invocation_storage_usage() const noexcept {
  if (state_ == nullptr || !state_->invocation_storage)
    return std::nullopt;
  return state_->invocation_storage->usage();
}

Result<Executable> Executable::exact_variant(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  if (index >= state_->exact_variants.size())
    return std::unexpected(
        Status::OutOfRange("exact variant index is out of range"));
  return state_->exact_variants[index];
}

Result<Invocation> Executable::create_invocation(
    InvocationOptions options) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }

  auto *budget = state_->invocation_storage.get();
  const auto allocator =
      budget != nullptr ? budget->allocator() : state_->runtime_allocator();
  detail::ConcreteDescriptions concrete(allocator);
  SpecializationSignature signature;
  ShapeValues values(allocator);
  const auto *domain = detail::GraphAccess::state(state_->domain_graph);
  if (domain != nullptr) {
    HYLE_RETURN_IF_ERROR(bind_shape_values(*domain, options.shape, allocator, values));
    if (const auto variant = find_exact_variant(*state_, values)) {
      values = ShapeValues(allocator);
      return state_->exact_variants[*variant].create_invocation(
          {.resources = std::move(options.resources)});
    }
    if (state_->finite_only)
      return std::unexpected(Status::Unimplemented(
          "shape is outside the prepared finite-exact variants"));
    HYLE_ASSIGN_OR_RETURN(signature, specialize_bound_descriptions(
                                         *domain, values, allocator, concrete));
  } else if (!options.shape.empty()) {
    return std::unexpected(
        Status::InvalidArgument("exact executable already has a fixed shape"));
  }

  detail::InvocationState::Reservation reservation(allocator);
  if (budget != nullptr) {
    HYLE_ASSIGN_OR_RETURN(const auto requirement,
                          invocation_storage_requirement(*state_));
    auto reserved = budget->reserve(requirement.bytes);
    if (!reserved)
      return std::unexpected(Status::ResourceExhausted(
          "graph shared invocation storage budget is exhausted"));
    reservation.aggregate = std::move(*reserved);
    const auto &resources = budget->resource_budget();
    if (resources) {
      if (!reservation.resources.try_reserve(
              state_->workspace_resources.size() + state_->steps.size()))
        return std::unexpected(invocation_allocation_failure());
      for (std::size_t slot = 0; slot < state_->workspace_resources.size(); ++slot) {
        HYLE_ASSIGN_OR_RETURN(auto report, state_->workspace_plan.resource_plan(slot));
        HYLE_ASSIGN_OR_RETURN(auto charge,
            resources.reserve(state_->workspace_resources[slot], report.peak_bytes,
                              compute::ResourceBudgetCategory::kWorkspace));
        if (!reservation.resources.try_push_back(std::move(charge)))
          return std::unexpected(invocation_allocation_failure());
      }
      for (const auto &step : state_->steps) {
        if (!step.emits_command() ||
            (step.kind != detail::PlanStepKind::kOperation && !step.region.valid()))
          continue;
        const auto private_bytes = step.kind == detail::PlanStepKind::kOperation
            ? step.operation.invocation_private_bytes()
            : step.region.properties().invocation_private_bytes;
        HYLE_ASSIGN_OR_RETURN(auto charge, resources.reserve_unattributed(private_bytes));
        if (!reservation.resources.try_push_back(std::move(charge)))
          return std::unexpected(invocation_allocation_failure());
      }
    }
    if (requirement.unknown)
      budget->note_unknown();
  }

  const std::span<const mem::MemoryResourceRef> selected_resources(
      state_->workspace_resources.data(), state_->workspace_resources.size());
  auto workspace =
      state_->workspace_plan.instantiate(selected_resources, allocator);
  if (!workspace)
    return std::unexpected(std::move(workspace).error());

  void *storage = allocator.allocate(
      detail::InvocationState::allocation_bytes(budget != nullptr),
      alignof(detail::InvocationState));
  if (storage == nullptr) {
    return std::unexpected(invocation_allocation_failure());
  }
  auto *invocation = ::new (storage)
      detail::InvocationState(state_, std::move(*workspace), allocator);
  if (budget != nullptr)
    ::new (invocation->storage_reservation())
        detail::InvocationState::Reservation(std::move(reservation));
  auto observation = compute::detail::RetainedCompletion::create(allocator);
  if (!observation) {
    auto error = std::move(observation).error();
    detail::release(invocation);
    return std::unexpected(std::move(error));
  }
  invocation->observation = std::move(*observation);
  invocation->concrete_descriptions = std::move(concrete);
  invocation->specialization = std::move(signature);
  const auto alias_index = create_external_alias_index(*invocation);
  if (!alias_index.ok()) {
    detail::release(invocation);
    return std::unexpected(alias_index);
  }
  const std::size_t value_count = state_->instance_descriptions.size();
  bool native_values = false;
  std::size_t native_input_count = 0;
  for (const auto &step : state_->steps) {
    if (!step.emits_command() || !step.region.valid()) continue;
    native_values |= step.region.prepares_bindings();
    native_input_count = (std::max)(native_input_count, std::size_t{step.input_count});
    for (std::size_t port = 0; port < step.output_count; ++port)
      native_values |= step.region.output_kind(port) != RegionOutputKind::kCaller;
  }
  if (!invocation->readable.try_resize(value_count) ||
      !invocation->writable.try_resize(value_count) ||
      !invocation->bound.try_resize(value_count) ||
      !invocation->invocations.try_reserve(state_->steps.size()) ||
      !invocation->submission_inputs.try_resize(state_->node_inputs.size()) ||
      !invocation->submission_outputs.try_resize(state_->node_outputs.size()) ||
      !invocation->step_bindings.try_resize(state_->steps.size()) ||
      !invocation->queues.try_resize(state_->lanes.size()) ||
      !invocation->candidate_queues.try_resize(state_->lanes.size()) ||
      !invocation->queue_seen.try_resize(state_->lanes.size()) ||
      !invocation->segment_completions.try_resize(state_->segments.size()) ||
      !invocation->capture_spans.try_resize(state_->segments.size()) ||
      !invocation->segment_progress.try_resize(state_->segments.size()) ||
      !invocation->lane_progress.try_resize(state_->lanes.size()) ||
      !invocation->segment_blocks_host.try_resize(state_->segments.size()) ||
      !invocation->candidate_segment_blocks_host.try_resize(state_->segments.size()) ||
      !invocation->dependency_routes.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->candidate_dependency_routes.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->lane_endpoints.try_resize(state_->lanes.size()) ||
      !invocation->candidate_lane_endpoints.try_resize(state_->lanes.size()) ||
      !invocation->completion_dependencies.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->aggregate_segments.try_resize(state_->segments.size()) ||
      !invocation->aggregate_children.try_resize(state_->segments.size())) {
    detail::release(invocation);
    return std::unexpected(invocation_allocation_failure());
  }
  if (native_values &&
      (!invocation->native_values.try_resize(value_count) ||
       !invocation->native_inputs.try_resize(native_input_count))) {
    detail::release(invocation);
    return std::unexpected(invocation_allocation_failure());
  }
  for (std::size_t ordinal = 0; ordinal < value_count; ++ordinal) {
    invocation->bound[ordinal] = 0;
    const std::uint32_t request = state_->workspace_requests[ordinal];
    if (request == detail::kNoWorkspaceRequest)
      continue;
    auto storage_view = workspace_tensor_storage(
        *invocation, static_cast<std::uint32_t>(ordinal));
    if (!storage_view) {
      Status error = std::move(storage_view).error();
      detail::release(invocation);
      return std::unexpected(std::move(error));
    }
    auto tensor = compute::TensorView::create(
        physical_description(*invocation, ordinal), *storage_view);
    if (!tensor) {
      Status error = std::move(tensor).error();
      detail::release(invocation);
      return std::unexpected(std::move(error));
    }
    invocation->writable[ordinal] = *tensor;
    invocation->readable[ordinal] = tensor->as_const();
  }
  invocation->aggregate_completion.completions =
      invocation->segment_completions.data();
  invocation->aggregate_completion.completion_segments =
      invocation->aggregate_segments.data();
  invocation->aggregate_completion.children =
      invocation->aggregate_children.data();
  invocation->aggregate_completion.completion_count = 0;
  for (std::size_t index = 0; index < invocation->queues.size(); ++index) {
    invocation->queues[index].reset();
    invocation->candidate_queues[index].reset();
    invocation->queue_seen[index] = 0;
    invocation->lane_endpoints[index] = {};
    invocation->candidate_lane_endpoints[index] = {};
  }
  for (std::size_t index = 0; index < state_->constant_ids.size(); ++index) {
    const std::size_t logical = state_->constant_ids[index].ordinal();
    const std::uint32_t ordinal = state_->external_instances[logical];
    invocation->readable[ordinal] = state_->constant_bindings[index];
    invocation->bound[ordinal] = 1;
  }
  for (const auto &value : state_->prepared_values) {
    invocation->readable[value.instance] = value.binding;
    invocation->bound[value.instance] = 1;
  }
  // Constant metadata aliases share the complete cold root. Bind each exact
  // child using this Invocation's concrete shape/offset, without copying a
  // slice or retaining a second canonical allocation under an ambiguous key.
  for (std::size_t instance = 0; instance < state_->instance_values.size(); ++instance) {
    const auto parent = state_->alias_sources[instance];
    if ((state_->instance_roles[instance] & detail::kConstant) == 0 ||
        parent == detail::kNoInPlaceSource)
      continue;
    auto bind_alias = [&]() noexcept -> Status {
      if (parent >= instance || !invocation->readable[parent].valid())
        return Status::Internal("immutable View has no ready physical parent");
      HYLE_ASSIGN_OR_RETURN(const auto offset, bound_alias_offset(*invocation, instance));
      HYLE_ASSIGN_OR_RETURN(const auto bytes,
          physical_description(*invocation, instance).required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto storage, invocation->readable[parent].storage().subview(offset, bytes));
      HYLE_ASSIGN_OR_RETURN(invocation->readable[instance],
          compute::ConstTensorView::create(physical_description(*invocation, instance), storage));
      // Provider port alignment remains part of prepare_replay, alongside
      // all other bindings. This merely resolves the immutable byte range.
      invocation->bound[instance] = 1;
      return Status::Ok();
    };
    const auto status = bind_alias();
    if (!status.ok()) {
      detail::release(invocation);
      return std::unexpected(status);
    }
  }
  for (std::size_t step_index = 0; step_index < state_->steps.size();
       ++step_index) {
    mem::BufferView step_workspace;
    const std::uint32_t request = state_->step_workspace_requests[step_index];
    if (request != detail::kNoWorkspaceRequest) {
      auto view = invocation->workspace.view(request);
      if (!view) {
        Status error = std::move(view).error();
        detail::release(invocation);
        return std::unexpected(std::move(error));
      }
      step_workspace = *view;
    }
    detail::StepInvocation created_step;
    const detail::PlanStep &step = state_->steps[step_index];
    if (step.emits_command() &&
        (step.kind == detail::PlanStepKind::kOperation || step.region.valid())) {
      if (step.region.valid()) {
        auto created = step.region.create_invocation(step_workspace,
                                                     invocation->allocator);
        if (!created) {
          Status error = std::move(created).error();
          detail::release(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.region = std::move(*created);
      } else {
        compute::PreparedOperation instance;
        const compute::PreparedOperation *operation = &step.operation;
        if (domain != nullptr &&
            has_dynamic_shape(*domain, domain->nodes[step.node_begin])) {
          detail::TensorDescriptions bound_inputs(allocator),
              bound_outputs(allocator);
          if (!bound_inputs.try_resize(step.input_count) ||
              !bound_outputs.try_resize(step.output_count)) {
            detail::release(invocation);
            return std::unexpected(invocation_allocation_failure());
          }
          for (std::size_t port = 0; port < step.input_count; ++port)
            bound_inputs[port] = physical_description(
                *invocation, state_->node_inputs[step.input_begin + port]);
          for (std::size_t port = 0; port < step.output_count; ++port)
            bound_outputs[port] = physical_description(
                *invocation, state_->node_outputs[step.output_begin + port]);
          auto bound = compute::detail::OperationPreparationAccess::instantiate(
              step.operation, values, bound_inputs, bound_outputs,
              invocation->allocator);
          if (!bound) {
            Status error = std::move(bound).error();
            detail::release(invocation);
            return std::unexpected(std::move(error));
          }
          instance = std::move(*bound);
          operation = &instance;
        }
        auto created =
            compute::detail::OperationPreparationAccess::create_invocation(
                *operation, step_workspace, invocation->allocator);
        if (!created) {
          Status error = std::move(created).error();
          detail::release(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.operation = std::move(*created);
      }
    }
    if (!invocation->invocations.try_push_back(std::move(created_step))) {
      detail::release(invocation);
      return std::unexpected(invocation_allocation_failure());
    }
    if (step.region.valid()) {
      for (std::size_t port = 0; port < step.output_count; ++port) {
        const auto kind = step.region.output_kind(port);
        if (kind != RegionOutputKind::kStableGenerated &&
            kind != RegionOutputKind::kDeferredGenerated) continue;
        const auto projected = project_generated_output(*invocation, step_index, port);
        if (!projected.ok()) {
          detail::release(invocation);
          return std::unexpected(projected);
        }
      }
    }
  }
  // A delegated endpoint is an invocation-local ordered execution identity.
  // Resolve it once, after every provider invocation exists, so replay
  // preparation can qualify all cross-island edges without a warm lookup.
  for (std::size_t step_index = 0; step_index < state_->steps.size();
       ++step_index) {
    const detail::PlanStep &step = state_->steps[step_index];
    if (!step.emits_command() || step.runtime_properties().mode !=
                                     compute::ProviderRuntimeMode::kDelegated) {
      continue;
    }
    if (!step.lane.valid() || step.lane.ordinal() >= state_->lanes.size() ||
        invocation->lane_endpoints[step.lane.ordinal()].valid()) {
      detail::release(invocation);
      return std::unexpected(Status::Internal(
          "graph delegated invocation has an invalid execution lane"));
    }
    compute::detail::CompletionEndpoint endpoint{};
    if (step.region.valid()) {
      endpoint =
          invocation->invocations[step_index].region.delegated_endpoint();
    } else if (step.kind == detail::PlanStepKind::kOperation) {
      endpoint = compute::detail::OperationInvocationAccess::delegated_endpoint(
          invocation->invocations[step_index].operation);
    }
    if (!endpoint.valid()) {
      detail::release(invocation);
      return std::unexpected(Status::Internal(
          "delegated provider returned an invalid execution endpoint"));
    }
    invocation->lane_endpoints[step.lane.ordinal()] = endpoint;
    invocation->candidate_lane_endpoints[step.lane.ordinal()] = endpoint;
  }
  rebuild_external_alias_index(*invocation);
  auto result = detail::InvocationAccess::adopt(invocation);
  if (options.resources)
    HYLE_RETURN_IF_ERROR(result.prepare_resources(std::move(*options.resources)));
  return result;
}

Invocation::Invocation(Invocation &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Invocation &Invocation::operator=(Invocation &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr)
      state_->observation.reset();
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Invocation::~Invocation() {
  // The observation retains the submitted Completion, which may in turn
  // retain this original state. Drop this root before the public owner.
  if (state_ != nullptr)
    state_->observation.reset();
  detail::release(state_);
}

Status detail::validate_read_binding(InvocationState &state, ValueId value,
    compute::ConstTensorView tensor, std::span<const InputSourceBinding> incoming) noexcept {
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(&state));
  HYLE_RETURN_IF_ERROR(detail::check_binding_replacement(state, value));
  if (!tensor.valid())
    return Status::InvalidArgument("tensor binding is invalid");
  const detail::PlanState &plan = *state.plan;
  if (!::hyle::graph::owns(plan, value)) {
    return Status::InvalidArgument("value does not belong to this executable");
  }
  const std::size_t ordinal = value.ordinal();
  if ((plan.binding_roles[ordinal] & detail::kConstant) != 0)
    return Status::InvalidArgument("graph constants are bound at compile time");
  if ((plan.binding_roles[ordinal] &
       (detail::kInvocationOwnedOutput | detail::kSourceAliasedOutput)) != 0) {
    return Status::FailedPrecondition(
        "graph output storage is fixed by its compiled ownership policy");
  }
  if ((plan.binding_roles[ordinal] & detail::kExternalInput) == 0) {
    if ((plan.binding_roles[ordinal] & detail::kExternalOutput) != 0) {
      return Status::InvalidArgument(
          "graph output requires mutable tensor storage");
    }
    return Status::InvalidArgument("internal graph values cannot be bound");
  }
  if (tensor.description() != logical_description(state, ordinal)) {
    return Status::InvalidArgument(
        "input tensor does not match the executable description");
  }
  Status resource =
      validate_external_read_resource(plan, ordinal, tensor.storage().resource());
  if (!resource.ok())
    return resource;
  Status alias =
      validate_external_alias(state, ordinal, byte_range(tensor.storage()), incoming);
  if (!alias.ok())
    return alias;
  const std::uint32_t instance = plan.external_instances[ordinal];
  if (instance == detail::kNoWorkspaceRequest)
    return Status::Internal("graph input has no external physical instance");
  return Status::Ok();
}
void detail::commit_read_binding(InvocationState &state, ValueId value,
    compute::ConstTensorView tensor) noexcept {
  const auto ordinal = value.ordinal();
  const auto instance = state.plan->external_instances[ordinal];
  invalidate_unproven_bindings(state);
  invalidate_binding_cache(state);
  state.readable[instance] = tensor;
  state.bound[instance] = 1;
  commit_external_alias_binding(state, ordinal);
  detail::note_borrowed_binding(state, value);
}
Status Invocation::bind(ValueId value,
                        compute::ConstTensorView tensor) noexcept {
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state_));
  HYLE_RETURN_IF_ERROR(detail::validate_read_binding(*state_, value, tensor));
  detail::commit_read_binding(*state_, value, tensor);
  return Status::Ok();
}

Status Invocation::bind(ValueId value, compute::TensorView tensor) noexcept {
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state_));
  HYLE_RETURN_IF_ERROR(detail::check_binding_replacement(*state_, value));
  if (!tensor.valid())
    return Status::InvalidArgument("tensor binding is invalid");
  const detail::PlanState &plan = *state_->plan;
  if (!owns(plan, value)) {
    return Status::InvalidArgument("value does not belong to this executable");
  }
  const std::size_t ordinal = value.ordinal();
  if ((plan.binding_roles[ordinal] & detail::kConstant) != 0)
    return Status::InvalidArgument("graph constants are bound at compile time");
  if ((plan.binding_roles[ordinal] &
       (detail::kExternalInput | detail::kExternalOutput)) ==
      (detail::kExternalInput | detail::kExternalOutput)) {
    return bind(value, tensor.as_const());
  }
  if ((plan.binding_roles[ordinal] & detail::kExternalOutput) != 0) {
    if ((plan.binding_roles[ordinal] & (detail::kInvocationOwnedOutput |
                                        detail::kSourceAliasedOutput)) != 0) {
      return Status::FailedPrecondition(
          "graph output storage is fixed by its compiled ownership policy");
    }
    if (tensor.description() != logical_description(*state_, ordinal)) {
      return Status::InvalidArgument(
          "output tensor does not match the executable description");
    }
    Status resource =
        validate_external_resource(plan, ordinal, tensor.storage().resource());
    if (!resource.ok())
      return resource;
    Status alias =
        validate_external_alias(*state_, ordinal, byte_range(tensor.storage()));
    if (!alias.ok())
      return alias;
    const std::uint32_t instance = plan.external_instances[ordinal];
    if (instance == detail::kNoWorkspaceRequest) {
      return Status::Internal("graph output has no external physical instance");
    }
    invalidate_unproven_bindings(*state_);
    invalidate_binding_cache(*state_);
    state_->writable[instance] = tensor;
    state_->readable[instance] = tensor.as_const();
    state_->bound[instance] = 1;
    commit_external_alias_binding(*state_, ordinal);
    detail::note_borrowed_binding(*state_, value);
    return Status::Ok();
  }
  if ((plan.binding_roles[ordinal] & detail::kExternalInput) != 0) {
    return bind(value, tensor.as_const());
  }
  return Status::InvalidArgument("internal graph values cannot be bound");
}

SpecializationSignature Invocation::specialization() const noexcept {
  if (state_ == nullptr)
    return {};
  return state_->plan->domain_graph.valid() ? state_->specialization
                                            : state_->plan->specialization;
}

Executable Invocation::executable() const noexcept {
  if (state_ == nullptr)
    return {};
  detail::retain(state_->plan);
  return detail::ExecutableAccess::adopt(state_->plan);
}

std::optional<std::size_t> Invocation::exact_variant_index() const noexcept {
  return state_ != nullptr ? state_->plan->exact_variant_index : std::nullopt;
}

bool Invocation::in_flight() const noexcept {
  return state_ != nullptr && state_->phase.load(std::memory_order_acquire) ==
                                  detail::InvocationPhase::kRunning;
}

Status detail::RegionAccess::revoke_preparation(Invocation &invocation) noexcept {
  auto *state = detail::InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state));
  invalidate_unproven_bindings(*state);
  invalidate_binding_cache(*state, false);
  return Status::Ok();
}

Status detail::RegionAccess::visit_bound(Invocation &invocation,
                                        detail::RegionAccessFn callback,
                                        void *user) noexcept {
  auto *state = detail::InvocationAccess::state(invocation);
  if (state == nullptr || callback == nullptr)
    return Status::InvalidArgument("bound region access requires an invocation and callback");
  HYLE_RETURN_IF_ERROR(detail::begin_preparation(*state));
  PreparationScope preparation{*state};
  HYLE_RETURN_IF_ERROR(prepare_invocation_bindings(*state));
  const auto &plan = *state->plan;
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const auto &step = plan.steps[index];
    if (step.kind != detail::PlanStepKind::kRegion)
      continue;
    const auto &slot = state->invocations[index].region;
    const detail::RegionAccessView view{
        static_cast<std::uint32_t>(index), plan.generation,
        std::span<const std::uint32_t>(plan.step_nodes)
            .subspan(step.node_set_begin, step.node_count),
        step.region.source_.shared_extension(), step.region.prepared_.state,
        slot.state_, step.lane,
        std::span<const compute::ConstTensorView>(state->submission_inputs)
            .subspan(step.input_begin, step.input_count),
        std::span<const compute::TensorView>(state->submission_outputs)
            .subspan(step.output_begin, step.output_count),
        step.region.prepared_.vtable};
    HYLE_RETURN_IF_ERROR(callback(user, view));
  }
  return Status::Ok();
}

Status Invocation::reset() noexcept {
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state_));
  // The payload first rejects retained source controls without disturbing the
  // old episode. Its reset revokes readiness before changing native resources.
  HYLE_RETURN_IF_ERROR(detail::reset_resource_payload(*this));
  invalidate_unproven_bindings(*state_);
  invalidate_binding_cache(*state_);
  state_->outputs_ready.store(false, std::memory_order_release);
  for (ValueId value : state_->plan->input_ids) {
    const std::uint32_t ordinal =
        state_->plan->external_instances[value.ordinal()];
    state_->readable[ordinal] = {};
    state_->bound[ordinal] = 0;
  }
  for (ValueId value : state_->plan->output_ids) {
    const std::size_t logical = value.ordinal();
    const std::uint32_t ordinal = state_->plan->external_instances[logical];
    if ((state_->plan->instance_roles[ordinal] & detail::kConstant) != 0)
      continue;
    state_->readable[ordinal] = {};
    state_->writable[ordinal] = {};
    state_->bound[ordinal] = 0;
  }
  detail::reset_resource_bindings(*state_);
  rebuild_external_alias_index(*state_);
  return Status::Ok();
}

Status detail::RegionAccess::reset_configuration(Invocation &invocation) noexcept {
  auto *state = detail::InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(detail::resource_control_ready(state));
  // Seed before callbacks so any successful ordinary bind or command-resource
  // replacement inside reset can revoke the physical candidate permanently.
  detail::remember_configuration_spans(*state);
  const auto reset = detail::reset_resource_payload(invocation);
  if (!reset.ok()) {
    detail::discard_configuration_spans(*state);
    return reset;
  }
  invalidate_unproven_bindings(*state);
  invalidate_binding_cache(*state, false);
  detail::reset_configuration_bindings(*state);
  rebuild_external_alias_index(*state);
  return Status::Ok();
}

Status Invocation::prepare_replay(std::span<const QueueBinding> queues,
                                  ReplayOptions options) noexcept {
  if (!state_)
    return Status::FailedPrecondition("graph invocation is empty");
  HYLE_RETURN_IF_ERROR(detail::begin_preparation(*state_));
  PreparationScope preparation{*state_};
  if (options.mode != ReplayMode::kCommands &&
      options.mode != ReplayMode::kPreferCapture &&
      options.mode != ReplayMode::kRequireCapture) {
    return Status::InvalidArgument("graph replay mode is invalid");
  }
  if (options.host_wait != HostWaitPolicy::kAllow &&
      options.host_wait != HostWaitPolicy::kReject) {
    return Status::InvalidArgument("graph host-wait policy is invalid");
  }
  // Backend capture/private driver allocation has no declared hard bound yet.
  // Do not mutate an existing replay or accept work before rejecting it.
  if (options.mode != ReplayMode::kCommands &&
      state_->plan->invocation_storage &&
      state_->plan->invocation_storage->rejects_unknown()) {
    if (options.mode == ReplayMode::kRequireCapture)
      return Status::ResourceExhausted(
          "graph command capture storage has no declared upper bound");
    options.mode = ReplayMode::kCommands;
  }
  if (queues.size() != state_->plan->required_queue_count) {
    return Status::InvalidArgument("replay preparation requires one queue for "
                                   "every active execution lane");
  }
  struct CandidateQueuesReset final {
    detail::InvocationState &state;
    ~CandidateQueuesReset() {
      for (auto &queue : state.candidate_queues)
        queue.reset();
    }
  } candidate_reset{*state_};
  for (std::size_t index = 0; index < state_->queues.size(); ++index) {
    state_->queue_seen[index] = 0;
    state_->candidate_queues[index].reset();
  }
  for (const QueueBinding binding : queues) {
    if (!binding.lane.valid() ||
        binding.lane.ordinal() >= state_->plan->lanes.size() ||
        binding.queue == nullptr || !binding.queue->valid() ||
        !binding.queue->is_open()) {
      return Status::InvalidArgument("graph replay queue binding is invalid");
    }
    const std::size_t lane = binding.lane.ordinal();
    if (state_->plan->lane_segment_counts[lane] == 0 ||
        !shared_lane(*state_->plan, binding.lane)) {
      return Status::InvalidArgument(
          "graph replay queue names a lane that does not accept a queue");
    }
    if (state_->queue_seen[lane] != 0)
      return Status::InvalidArgument(
          "graph replay queue binding is duplicated");
    HYLE_ASSIGN_OR_RETURN(auto retained,
        compute::detail::QueueRef::acquire(*binding.queue));
    const ExecutionTargetId target = state_->plan->lanes[lane].target;
    if (!detail::ExecutionSourceAccess::owns(
            state_->plan->targets[target.ordinal()].source, retained)) {
      return Status::InvalidArgument(
          "graph replay queue belongs to a different compute context");
    }
    state_->candidate_queues[lane] = std::move(retained);
    state_->queue_seen[lane] = 1;
  }
  for (std::size_t lane = 0; lane < state_->queue_seen.size(); ++lane) {
    if (state_->plan->lane_segment_counts[lane] != 0 &&
        shared_lane(*state_->plan,
                    ExecutionLaneId{static_cast<std::uint32_t>(lane)}) &&
        state_->queue_seen[lane] == 0) {
      return Status::InvalidArgument("graph replay queue binding is missing");
    }
  }

  for (std::size_t lane = 0; lane < state_->plan->lanes.size(); ++lane) {
    const ExecutionLaneId lane_id{static_cast<std::uint32_t>(lane)};
    if (state_->plan->lane_segment_counts[lane] == 0) {
      state_->candidate_lane_endpoints[lane] = {};
      continue;
    }
    if (!shared_lane(*state_->plan, lane_id)) {
      if (!state_->lane_endpoints[lane].valid()) {
        return Status::Internal(
            "graph delegated execution endpoint is unavailable");
      }
      state_->candidate_lane_endpoints[lane] = state_->lane_endpoints[lane];
      continue;
    }
    const auto &queue = state_->candidate_queues[lane];
    if (!queue.valid())
      return Status::Internal("graph replay queue binding is unavailable");
    auto endpoint = compute::detail::QueueSequence::completion_endpoint(queue);
    if (!endpoint) {
      return std::move(endpoint).error().WithContext(
          "querying graph Queue completion endpoint");
    }
    state_->candidate_lane_endpoints[lane] = *endpoint;
  }

  // Commands will enter ordinary provider encoders on the next submit. Old
  // captures must release their native pool addresses before any cold rebind.
  // Invalid modes/Queues above have not disturbed a previously prepared route.
  if (options.mode == ReplayMode::kCommands)
    reset_capture_cache(*state_);
  // A recording belongs to its actual Queue owner, not a borrowed Queue
  // wrapper or logical lane. Destroy foreign recordings before any cold hook.
  for (std::size_t lane = 0; lane < state_->queues.size(); ++lane) {
    if (!shared_lane(*state_->plan,
                    ExecutionLaneId{static_cast<std::uint32_t>(lane)}) ||
        state_->queues[lane].same_owner(state_->candidate_queues[lane]))
      continue;
    for (const auto &segment : state_->plan->segments) {
      if (segment.lane.ordinal() != lane)
        continue;
      const std::size_t end = segment.step_begin + segment.step_count;
      for (std::size_t index = segment.step_begin; index < end; ++index) {
        if (!state_->plan->steps[index].emits_command())
          continue;
        state_->step_bindings[index].stable_binding_identity = false;
        detail::invalidate_step_recording(*state_, static_cast<std::uint32_t>(index));
      }
    }
  }
  Status bindings = prepare_invocation_bindings(*state_);
  if (!bindings.ok())
    return std::move(bindings).WithContext("preparing graph replay bindings");
  ReplayReport transfer_report{};
  Status transfers = validate_transfer_routes(*state_, queues, transfer_report);
  if (!transfers.ok())
    return transfers;
  auto dependency_report =
      prepare_dependency_routes(*state_, options.host_wait);
  if (!dependency_report)
    return std::move(dependency_report).error();
  HYLE_RETURN_IF_ERROR(detail::prepare_input_sources(*state_, queues, options.host_wait));
  dependency_report->immediate_transfer_count =
      transfer_report.immediate_transfer_count;
  dependency_report->queued_transfer_count =
      transfer_report.queued_transfer_count;
  dependency_report->peer_transfer_count = transfer_report.peer_transfer_count;

  // A deployment may deliberately bind several logical lanes to one ordered
  // Queue. Reserve that Queue for the sum of all its emitted segments, once.
  // Per-lane reservations would understate ring capacity because the
  // backend sees one physical Queue.
  for (std::size_t index = 0; index < queues.size(); ++index) {
    const auto &queue =
        state_->candidate_queues[queues[index].lane.ordinal()];
    bool seen = false;
    for (std::size_t earlier = 0; earlier < index; ++earlier)
      seen = seen || queue.same_owner(
                         state_->candidate_queues[queues[earlier].lane.ordinal()]);
    if (seen)
      continue;
    std::size_t count = 0;
    for (const QueueBinding binding : queues) {
      if (!queue.same_owner(state_->candidate_queues[binding.lane.ordinal()]))
        continue;
      const std::size_t lane_count =
          state_->plan->lane_segment_counts[binding.lane.ordinal()];
      if (lane_count > SIZE_MAX - count) {
        return Status::ResourceExhausted(
            "graph replay completion capacity overflows");
      }
      count += lane_count;
    }
    Status reserved = queue.reserve_completions(count);
    if (!reserved.ok()) {
      return std::move(reserved).WithContext(
          "reserving graph replay completion capacity");
    }
  }

  // Reserve only the bounded number of selected compatible runs. This is
  // cold metadata, before replacing a usable route or recording any device work.
  if (options.mode != ReplayMode::kCommands) {
    if (options.mode == ReplayMode::kRequireCapture &&
        state_->plan->capture_command_count != state_->plan->command_count)
      return Status::Unimplemented("a selected graph command is not capture-safe");
    if (!state_->captured_ranges.try_reserve(state_->plan->capture_ranges.size()))
      return Status::ResourceExhausted("graph capture range allocation failed");
    if (!state_->capture_rebuilds.try_reserve(state_->plan->capture_ranges.size()))
      return Status::ResourceExhausted("graph capture rebuild allocation failed");
    while (state_->capture_rebuilds.size() < state_->plan->capture_ranges.size()) {
      const auto &range = state_->plan->capture_ranges[state_->capture_rebuilds.size()];
      if (!state_->capture_rebuilds.try_push_back(
              detail::CaptureRebuildRange{range.step_begin, range.step_count, state_->allocator}))
        return Status::ResourceExhausted("graph capture rebuild slot allocation failed");
    }
    for (auto &saved : state_->capture_rebuilds)
      HYLE_RETURN_IF_ERROR(qualify_capture_variants(*state_, saved));
  }

  // Publish the replacement Queue/route only after validation. Any selected
  // resources already changed above have revoked their old replay first.
  for (std::size_t lane = 0; lane < state_->queues.size(); ++lane)
    state_->queues[lane] = std::move(state_->candidate_queues[lane]);
  // Publish producer identity only at successful preparation, including capture.
  // Delegated lanes have no Queue and retain their own producer group.
  auto commit_producer_groups = [&]() noexcept {
    detail::commit_input_sources(*state_);
    std::copy(state_->candidate_segment_blocks_host.begin(),
              state_->candidate_segment_blocks_host.end(),
              state_->segment_blocks_host.begin());
    for (std::size_t lane = 0; lane < state_->lane_progress.size(); ++lane) {
      auto &progress = state_->lane_progress[lane];
      progress.producer_group = static_cast<std::uint32_t>(lane);
      if (!shared_lane(*state_->plan,
                       ExecutionLaneId{static_cast<std::uint32_t>(lane)}))
        continue;
      for (std::size_t earlier = 0; earlier < lane; ++earlier) {
        if (state_->queues[lane].same_owner(state_->queues[earlier])) {
          progress.producer_group = static_cast<std::uint32_t>(earlier);
          break;
        }
      }
    }
  };
  std::copy(state_->candidate_dependency_routes.begin(),
            state_->candidate_dependency_routes.end(),
            state_->dependency_routes.begin());
  std::copy(state_->candidate_lane_endpoints.begin(),
            state_->candidate_lane_endpoints.end(),
            state_->lane_endpoints.begin());
  state_->transfers_prepared = true;
  state_->dependencies_prepared = true;
  state_->replay_report = *dependency_report;

  state_->replay_prepared = false;
  state_->replay_route = ReplayRoute::kUnprepared;
  state_->replay_report.route = ReplayRoute::kUnprepared;
  if (options.mode == ReplayMode::kCommands) {
    rebuild_capture_spans(*state_);
    commit_producer_groups();
    state_->replay_route = ReplayRoute::kCommands;
    state_->replay_report.route = ReplayRoute::kCommands;
    state_->replay_prepared = true;
    return Status::Ok();
  }

  auto unprepare_captures = [&]() noexcept {
    // Unchanged and newly completed recordings may remain dormant after a
    // later cold rejection. No route can launch them until preparation commits.
    for (auto &span : state_->capture_spans)
      span = {};
    state_->replay_prepared = false;
    state_->replay_route = ReplayRoute::kUnprepared;
    state_->replay_report = {};
    // A failed encoder may have invalidated private provider state. No old
    // variant can bypass that failure on a later same-tuple preparation.
    for (auto &saved : state_->capture_rebuilds) {
      saved.storage.reset();
      discard_capture_variants(saved);
    }
  };
  auto finish_preparation =
      [&](compute::detail::QueueSequence &sequence) noexcept -> Status {
    auto completion = sequence.finish();
    if (!completion)
      return std::move(completion).error();
    HYLE_RETURN_IF_ERROR(detail::retain_preparation_completion(*state_, *completion));
    return completion->wait();
  };

  std::size_t captured_command_count = 0;
  std::size_t recorded_range_count = 0;
  std::size_t reused_range_count = 0;
  for (const auto &candidate : state_->plan->capture_ranges) {
    const auto &segment = state_->plan->segments[candidate.segment];
    detail::PlanSegment range = segment;
    range.step_begin = candidate.step_begin;
    range.step_count = candidate.step_count;
    const auto retained = std::lower_bound(
        state_->captured_ranges.begin(), state_->captured_ranges.end(),
        range.step_begin, [](const auto &value, std::uint32_t begin) noexcept {
          return value.step_begin < begin;
        });
    if (retained != state_->captured_ranges.end() &&
        retained->step_begin == range.step_begin &&
        retained->step_count == range.step_count) {
      ++reused_range_count;
      captured_command_count += candidate.command_count;
      continue;
    }
    auto *saved = capture_storage(*state_, range.step_begin, range.step_count);
    std::uint8_t selected_variant = 2;
    if (saved != nullptr && saved->preserves_private_arguments) {
      for (std::uint8_t index = 0; index < 2; ++index) {
        auto &variant = saved->variants[index];
        if (!variant.commands.valid() || !capture_key_matches(*state_, *saved, variant))
          continue;
        append_capture_range(*state_, range.step_begin, range.step_count,
                             std::move(variant.commands));
        saved->active_variant = index;
        saved->last_used = index;
        selected_variant = index;
        break;
      }
      if (selected_variant < 2) {
        ++reused_range_count;
        captured_command_count += candidate.command_count;
        continue;
      }
      selected_variant = saved->variants[0].key_valid
          ? saved->variants[1].key_valid ? static_cast<std::uint8_t>(1 - saved->last_used) : std::uint8_t{1}
          : std::uint8_t{0};
      auto &evicted = saved->variants[selected_variant];
      if (evicted.commands.valid())
        saved->storage = compute::detail::CommandCaptureAccess::recycle_after_retirement(
            std::move(evicted.commands));
      evicted.key_valid = false;
      evicted.bindings.clear();
    }
    compute::Queue *queue = candidate_queue(queues, segment.lane);
    if (queue == nullptr) {
      unprepare_captures();
      return Status::Internal("graph capture Queue binding is unavailable");
    }
    auto extension = compute::detail::CommandCaptureAccess::query(*queue);
    if (!extension) {
      Status unavailable = std::move(extension).error();
      if (options.mode == ReplayMode::kPreferCapture && unavailable.IsUnimplemented()) {
        if (saved != nullptr) {
          saved->storage.reset();
          discard_capture_variants(*saved);
        }
        continue;
      }
      unprepare_captures();
      return std::move(unavailable).WithContext("querying graph command capture support");
    }
    auto sequence = compute::detail::QueueSequence::begin(
        state_->queues[segment.lane.ordinal()]);
    if (!sequence) {
      unprepare_captures();
      return std::move(sequence).error().WithContext("beginning graph command capture sequence");
    }
    if (state_->plan->invocation_storage)
      state_->plan->invocation_storage->note_unknown();
    Status begun = compute::detail::CommandCaptureAccess::begin(*extension, *sequence);
    if (!begun.ok()) {
      finish_preparation(*sequence).IgnoreError();
      if (options.mode == ReplayMode::kPreferCapture && begun.IsUnimplemented()) {
        if (saved != nullptr) {
          saved->storage.reset();
          discard_capture_variants(*saved);
        }
        continue;
      }
      unprepare_captures();
      return std::move(begun).WithContext("beginning graph command capture");
    }
    Status recorded = enqueue_segment_steps(*state_, range, *sequence);
    if (!recorded.ok()) {
      const Status aborted = compute::detail::CommandCaptureAccess::abort(*extension, *sequence);
      const Status finished = finish_preparation(*sequence);
      (void)aborted;
      (void)finished;
      unprepare_captures();
      return std::move(recorded).WithContext("recording a capture-compatible graph range");
    }
    compute::detail::CaptureRebuildStorage rebuild;
    if (saved != nullptr)
      rebuild = std::move(saved->storage);
    auto captured = compute::detail::CommandCaptureAccess::end(
        *extension, *sequence, state_->allocator, std::move(rebuild));
    Status finished = finish_preparation(*sequence);
    if (!captured) {
      Status capture_error = std::move(captured).error();
      finished.IgnoreError();
      // A failed retirement can still own the original Invocation. It cannot
      // be converted to a usable command fallback while the prefix is live.
      if (options.mode == ReplayMode::kPreferCapture &&
          capture_error.IsUnimplemented() && finished.ok()) {
        if (saved != nullptr)
          discard_capture_variants(*saved);
        continue;
      }
      unprepare_captures();
      return std::move(capture_error).WithContext("finalizing graph command capture");
    }
    if (!finished.ok()) {
      unprepare_captures();
      return std::move(finished).WithContext("retiring graph command capture sequence");
    }
    // This fact comes from the selected Queue's actual capture backend, not
    // from the generic allocation token. Keep it after Commands destroys the
    // artifact so proven pointer-backed bindings need no redundant rebind.
    const bool stable_identity = captured->tensor_bindings_have_stable_identity();
    for (std::size_t index = range.step_begin;
         index < std::size_t{range.step_begin} + range.step_count; ++index)
      state_->step_bindings[index].stable_binding_identity = stable_identity;
    if (stable_identity && saved != nullptr && selected_variant < 2)
      remember_capture_key(*state_, *saved, selected_variant);
    append_capture_range(*state_, range.step_begin, range.step_count, std::move(*captured));
    ++recorded_range_count;
    captured_command_count += candidate.command_count;
  }
  state_->replay_route = captured_command_count == 0 ? ReplayRoute::kCommands
      : captured_command_count == state_->plan->command_count ? ReplayRoute::kCaptured
                                               : ReplayRoute::kHybrid;
  state_->replay_report.route = state_->replay_route;
  state_->replay_report.recorded_capture_range_count = recorded_range_count;
  state_->replay_report.reused_capture_range_count = reused_range_count;
  rebuild_capture_spans(*state_);
  commit_producer_groups();
  state_->replay_prepared = true;
  return Status::Ok();
}

ReplayRoute Invocation::replay_route() const noexcept {
  return state_ != nullptr ? state_->replay_route : ReplayRoute::kUnprepared;
}

Result<ReplayReport> Invocation::replay_report() const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  if (!state_->replay_prepared) {
    return std::unexpected(
        Status::FailedPrecondition("invocation replay is not prepared"));
  }
  return state_->replay_report;
}

Result<ReplayDependencyRoute>
Invocation::dependency_route(std::size_t segment_index,
                             std::size_t dependency_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  if (!state_->replay_prepared) {
    return std::unexpected(
        Status::FailedPrecondition("invocation replay is not prepared"));
  }
  if (segment_index >= state_->plan->segments.size()) {
    return std::unexpected(
        Status::OutOfRange("graph replay segment index is out of range"));
  }
  const detail::PlanSegment &segment = state_->plan->segments[segment_index];
  if (dependency_index >= segment.dependency_count) {
    return std::unexpected(Status::OutOfRange(
        "graph replay segment dependency index is out of range"));
  }
  return state_->dependency_routes[segment.dependency_begin + dependency_index];
}

Result<compute::ConstTensorView>
Invocation::output(ValueId value) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  if (state_->phase.load(std::memory_order_acquire) !=
          detail::InvocationPhase::kIdle ||
      !state_->outputs_ready.load(std::memory_order_acquire)) {
    return std::unexpected(Status::FailedPrecondition(
        "graph outputs require a successful completed submission"));
  }
  const detail::PlanState &plan = *state_->plan;
  if (!owns(plan, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this executable"));
  }
  const std::size_t ordinal = value.ordinal();
  if ((plan.binding_roles[ordinal] & detail::kExternalOutput) == 0) {
    return std::unexpected(
        Status::InvalidArgument("value is not a published graph output"));
  }
  const std::uint32_t instance = plan.external_instances[ordinal];
  if (instance == detail::kNoWorkspaceRequest) {
    return std::unexpected(
        Status::Internal("published graph output has no readable storage"));
  }
  return detail::output_tensor(*state_, instance);
}


Result<CompiledGraph> compile(const Graph &graph,
                              const compute::Context &context,
                              CompileOptions options) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto source, ExecutionSource::acquire(context));
  const std::array storage_domains{StorageDomain{
      .resource = mem::system_resource(),
  }};
  const std::array targets{ExecutionTarget{
      .source = std::move(source),
      .default_storage = StorageDomainId{0},
  }};
  const std::array lanes{ExecutionLane{
      .target = ExecutionTargetId{0},
      .role = ExecutionLaneRole::kCompute,
  }};
  return compile(graph,
                 CompileEnvironment{
                     .storage_domains = storage_domains,
                     .targets = targets,
                     .lanes = lanes,
                 },
                 options);
}

namespace {

Result<CompiledGraph> compile_plan(
    const Graph &graph, CompileEnvironment environment, CompileOptions options,
    detail::ConstantArtifactCache &packing,
    bool inherited_budget = false,
    util::Ref<detail::StorageBudgetState> invocation_storage = {},
    util::Ref<detail::StorageBudgetState> compilation_storage = {},
    std::span<const detail::RegionAdapter> adapters = {},
    bool finite_shell = false) noexcept {
  if (!options.workload.valid())
    return std::unexpected(Status::InvalidArgument("graph workload context"));
  if (!graph.valid()) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }
  if (environment.storage_domains.empty()) {
    return std::unexpected(
        Status::InvalidArgument("compile environment has no storage domains"));
  }
  if (environment.targets.empty()) {
    return std::unexpected(Status::InvalidArgument(
        "compile environment has no execution targets"));
  }
  if (environment.storage_domains.size() >
          (std::numeric_limits<std::uint32_t>::max)() ||
      environment.targets.size() >
          (std::numeric_limits<std::uint32_t>::max)() ||
      environment.lanes.size() > (std::numeric_limits<std::uint32_t>::max)()) {
    return std::unexpected(Status::ResourceExhausted(
        "compile environment identity space is exhausted"));
  }
  for (std::size_t index = 0; index < environment.storage_domains.size();
       ++index) {
    const StorageDomain &storage = environment.storage_domains[index];
    if (!storage.resource.valid()) {
      return std::unexpected(Status::InvalidArgument(
          "compile environment storage resource is invalid"));
    }
    if (storage.owner && storage.owner.ref() != storage.resource)
      return std::unexpected(Status::InvalidArgument(
          "compile storage owner belongs to another resource"));
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (environment.storage_domains[previous].resource == storage.resource) {
        return std::unexpected(Status::InvalidArgument(
            "compile environment duplicates a storage resource"));
      }
    }
  }
  for (std::size_t index = 0; index < environment.targets.size(); ++index) {
    const ExecutionTarget &target = environment.targets[index];
    if (!target.source.valid() || !target.source.is_open()) {
      return std::unexpected(Status::FailedPrecondition(
          "compile environment execution source is invalid or closed"));
    }
    if (!target.default_storage.valid() ||
        target.default_storage.ordinal() >=
            environment.storage_domains.size()) {
      return std::unexpected(Status::InvalidArgument(
          "compile environment target default storage is invalid"));
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (environment.targets[previous].source.same_owner(target.source)) {
        return std::unexpected(Status::InvalidArgument(
            "compile environment duplicates an execution source"));
      }
    }
    Status compatible = target.source.validate_resource(
        environment.storage_domains[target.default_storage.ordinal()].resource);
    if (!compatible.ok())
      return std::unexpected(std::move(compatible));
  }
  for (const auto &adapter : adapters) {
    if (!adapter.target.valid() ||
        adapter.target.ordinal() >= environment.targets.size() ||
        !adapter.source.valid() || adapter.interface == nullptr ||
        adapter.interface->propose == nullptr ||
        adapter.interface->propose_resources == nullptr ||
        adapter.interface->compile == nullptr)
      return std::unexpected(Status::InvalidArgument(
          "graph capability adapter is incomplete or has a foreign target"));
    const auto &source = environment.targets[adapter.target.ordinal()].source;
    if (!source.same_owner(adapter.source.source()))
      return std::unexpected(Status::InvalidArgument(
          "graph capability adapter belongs to another execution source"));
    if (adapter.source.is_shared_queue()) {
      if (adapter.provider_index != UINT32_MAX || adapter.source.native_interface()!=adapter.interface)
        return std::unexpected(Status::InvalidArgument(
            "shared Queue transfer cannot have a numerical provider slot"));
      HYLE_ASSIGN_OR_RETURN(auto retained,
          detail::ExecutionSourceAccess::compute(source));
      (void)retained;
      continue;
    }
    if (adapter.source.is_native()) {
      if (adapter.provider_index != UINT32_MAX)
        return std::unexpected(Status::InvalidArgument(
            "native region capability cannot have a Compute provider slot"));
      HYLE_ASSIGN_OR_RETURN(auto actual,
          detail::ExecutionSourceAccess::region(source));
      if (adapter.source.native_interface() != actual.native_interface())
        return std::unexpected(Status::InvalidArgument(
            "native region adapter does not retain its actual source capability"));
      continue;
    }
    HYLE_ASSIGN_OR_RETURN(auto retained,
        detail::ExecutionSourceAccess::compute(source));
    HYLE_ASSIGN_OR_RETURN(auto identity,
        retained->provider_identity(adapter.provider_index));
    HYLE_ASSIGN_OR_RETURN(auto instance,
        retained->provider_instance_id(adapter.provider_index));
    const auto *extension = adapter.source.shared_extension();
    if (extension == nullptr || !retained->owns(*extension) ||
        adapter.source.provider() != identity ||
        adapter.source.provider_instance_id() != instance)
      return std::unexpected(Status::InvalidArgument(
          "graph capability adapter does not belong to its exact Context/provider"));
  }
  for (const ExecutionLane &lane : environment.lanes) {
    if (!lane.target.valid() ||
        lane.target.ordinal() >= environment.targets.size() ||
        (lane.role != ExecutionLaneRole::kCompute &&
         lane.role != ExecutionLaneRole::kTransfer &&
         lane.role != ExecutionLaneRole::kCommunication)) {
      return std::unexpected(
          Status::InvalidArgument("compile environment lane is invalid"));
    }
  }
  const auto &source = environment.targets[0].source;
  if (options.search_policy != CompileSearchPolicy::kAllowLimited &&
      options.search_policy != CompileSearchPolicy::kRequireUnexhausted) {
    return std::unexpected(
        Status::InvalidArgument("graph compile search policy is invalid"));
  }
  if (options.compiled_storage.unknown != UnknownStoragePolicy::kReject &&
      options.compiled_storage.unknown != UnknownStoragePolicy::kAllow)
    return std::unexpected(Status::InvalidArgument(
        "graph compiled storage unknown policy is invalid"));
  if (options.invocation_storage &&
      options.invocation_storage->unknown != UnknownStoragePolicy::kReject &&
      options.invocation_storage->unknown != UnknownStoragePolicy::kAllow)
    return std::unexpected(Status::InvalidArgument(
        "graph invocation storage unknown policy is invalid"));
  if (options.compilation_storage &&
      options.compilation_storage->unknown != UnknownStoragePolicy::kReject &&
      options.compilation_storage->unknown != UnknownStoragePolicy::kAllow)
    return std::unexpected(Status::InvalidArgument(
        "graph compilation storage unknown policy is invalid"));
  if (options.region_selection != RegionSelectionPolicy::kDisabled &&
      options.region_selection !=
          RegionSelectionPolicy::kAllowProviderRegions) {
    return std::unexpected(
        Status::InvalidArgument("graph region selection policy is invalid"));
  }
  if (options.region_selection != RegionSelectionPolicy::kDisabled &&
      (options.max_region_proposals_per_provider == 0 ||
       (!inherited_budget && (options.max_region_proposals == 0 ||
                              options.max_region_proposal_nodes == 0)))) {
    return std::unexpected(Status::InvalidArgument(
        "enabled graph region selection requires non-zero hard budgets"));
  }
  if (options.max_region_proposals >
          (std::numeric_limits<std::uint32_t>::max)() ||
      options.max_region_proposals_per_provider >
          (std::numeric_limits<std::uint32_t>::max)() ||
      options.max_region_proposal_nodes >
          (std::numeric_limits<std::uint32_t>::max)()) {
    return std::unexpected(Status::OutOfRange(
        "graph region proposal budget exceeds the supported identity space"));
  }
  if (options.operation_fallback != OperationFallbackPolicy::kAllow &&
      options.operation_fallback != OperationFallbackPolicy::kReject) {
    return std::unexpected(
        Status::InvalidArgument("graph operation fallback policy is invalid"));
  }
  if (options.operation_fallback == OperationFallbackPolicy::kAllow &&
      options.max_operation_proposals == 0 && !inherited_budget)
    return std::unexpected(
        Status::InvalidArgument("graph operation proposal budget is zero"));
  if (options.provider_replay != ProviderReplayPolicy::kAllowAny &&
      options.provider_replay != ProviderReplayPolicy::kRequirePrepared) {
    return std::unexpected(
        Status::InvalidArgument("graph provider replay policy is invalid"));
  }
  if (options.workspace_reuse != WorkspaceReusePolicy::kExistingOrderOnly &&
      options.workspace_reuse != WorkspaceReusePolicy::kAllowSerialization) {
    return std::unexpected(
        Status::InvalidArgument("graph workspace reuse policy is invalid"));
  }
  const detail::GraphState *logical = detail::GraphAccess::state(graph);
  if (logical == nullptr) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }

  auto backing = source.metadata_allocator();
  if (options.resource_budget) {
    HYLE_ASSIGN_OR_RETURN(backing, options.resource_budget.metadata_allocator());
  }
  if (!backing.valid()) {
    return std::unexpected(Status::FailedPrecondition(
        "execution source has no metadata allocator"));
  }
  compute::StorageOwner metadata_owner;
  const auto actual_backing = options.resource_budget
      ? compute::detail::ResourceBudgetAccess::backing_allocator(options.resource_budget)
      : backing;
  for (const auto &domain : environment.storage_domains)
    if (domain.owner.owns_allocator(actual_backing)) {
      metadata_owner = domain.owner;
      break;
    }
  if (options.compilation_storage && !compilation_storage) {
    HYLE_ASSIGN_OR_RETURN(
        auto *created,
        detail::StorageBudgetState::create(
            backing,
            {.max_bytes = options.compilation_storage->max_peak_bytes,
             .unknown = options.compilation_storage->unknown},
            {}, true, options.resource_budget, metadata_owner));
    compilation_storage = util::adopt_ref(created);
  }
  const auto allocator =
      compilation_storage ? compilation_storage->allocator() : backing;
  ProviderPolicy provider_policy(allocator);
  HYLE_RETURN_IF_ERROR(provider_policy.initialize(*logical, options));
  detail::TensorDescriptions descriptions(allocator);
  ShapeValues shape_values(allocator);
  SpecializationSignature signature;
  if (options.shape_policy == CompileShapePolicy::kExact) {
    HYLE_RETURN_IF_ERROR(
        bind_shape_values(*logical, options.shape, allocator, shape_values));
    HYLE_ASSIGN_OR_RETURN(
        signature, specialize_bound_descriptions(*logical, shape_values,
                                                 allocator, descriptions));
  } else if (options.shape_policy == CompileShapePolicy::kDeclaredDomain) {
    if (!options.shape.empty())
      return std::unexpected(Status::InvalidArgument(
          "declared-domain compilation does not accept exact shape bindings"));
    HYLE_RETURN_IF_ERROR(
        describe_domain(*logical, allocator, descriptions, shape_values));
  } else {
    return std::unexpected(
        Status::InvalidArgument("compile shape policy is invalid"));
  }
  auto specialized =
      make_specialized_graph(*logical,
                             std::span<const compute::TensorDesc>(
                                 descriptions.data(), descriptions.size()),
                             allocator, shape_values,
                             options.shape_policy ==
                                 CompileShapePolicy::kDeclaredDomain);
  if (!specialized)
    return std::unexpected(std::move(specialized).error());
  const detail::GraphState *concrete = detail::GraphAccess::state(*specialized);
  if (concrete == nullptr) {
    return std::unexpected(
        Status::Internal("specialized graph state is invalid"));
  }

  void *storage =
      allocator.allocate(sizeof(detail::PlanState), alignof(detail::PlanState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *plan =
      ::new (storage) detail::PlanState(allocator, concrete->generation);
  plan->workload = options.workload;
  plan->metadata_owner = metadata_owner;
  plan->compilation_storage = std::move(compilation_storage);
  if (options.invocation_storage && !invocation_storage) {
    auto created = detail::StorageBudgetState::create(
        backing, *options.invocation_storage, allocator, false,
        options.resource_budget, metadata_owner);
    if (!created) {
      detail::release(plan);
      return std::unexpected(std::move(created).error());
    }
    invocation_storage = util::adopt_ref(*created);
  }
  plan->invocation_storage = std::move(invocation_storage);
  if (!finite_shell)
    plan->specialization = std::move(signature);
  plan->finite_only = finite_shell;
  if (finite_shell || options.shape_policy == CompileShapePolicy::kDeclaredDomain)
    plan->domain_graph = graph;
  plan->workspace_reuse = options.workspace_reuse;
  if (!plan->storage_domains.try_reserve(environment.storage_domains.size()) ||
      !plan->targets.try_reserve(environment.targets.size()) ||
      !plan->lanes.try_reserve(environment.lanes.empty()
                                   ? environment.targets.size()
                                   : environment.lanes.size())) {
    detail::release(plan);
    return std::unexpected(plan_allocation_failure());
  }
  for (const StorageDomain &storage_domain : environment.storage_domains) {
    if (!plan->storage_domains.try_push_back(detail::StorageDomainRecord{
            .resource = storage_domain.resource,
            .workspace_budget_bytes = storage_domain.workspace_budget_bytes,
            .owner = storage_domain.owner,
        })) {
      detail::release(plan);
      return std::unexpected(plan_allocation_failure());
    }
  }
  for (const ExecutionTarget &target : environment.targets) {
    if (!plan->targets.try_push_back(detail::ExecutionTargetRecord{
            .source = target.source,
            .default_storage = target.default_storage,
        })) {
      detail::release(plan);
      return std::unexpected(plan_allocation_failure());
    }
  }
  Status lanes = initialize_plan_lanes(*plan, environment);
  if (!lanes.ok()) {
    detail::release(plan);
    return std::unexpected(std::move(lanes));
  }

  if (finite_shell) {
    // Only semantic ports and endpoint metadata live on this cold index.
    // Concrete descriptions above validate its first declared point; no
    // physical values, constants, Provider artifacts or workspace are kept.
    plan->logical_node_count = logical->nodes.size();
    if (!plan->input_ids.try_reserve(logical->inputs.size()) ||
        !plan->output_ids.try_reserve(logical->outputs.size()) ||
        !plan->lane_segment_counts.try_resize(plan->lanes.size())) {
      detail::release(plan);
      return std::unexpected(plan_allocation_failure());
    }
    for (const auto input : logical->inputs) {
      if (!plan->input_ids.try_push_back(detail::value_id(*logical, input))) {
        detail::release(plan);
        return std::unexpected(plan_allocation_failure());
      }
    }
    for (const auto output : logical->outputs) {
      if (!plan->output_ids.try_push_back(detail::value_id(*logical, output))) {
        detail::release(plan);
        return std::unexpected(plan_allocation_failure());
      }
    }
    auto report = build_report(*plan, *logical, {});
    if (!report) {
      detail::release(plan);
      return std::unexpected(std::move(report).error());
    }
    return CompiledGraph{detail::ExecutableAccess::adopt(plan), std::move(*report)};
  }

  container::SmallVector<RouteDiagnostic, 16, mem::AllocatorRef> routes(
      allocator);
  if (!routes.try_resize(concrete->nodes.size())) {
    detail::release(plan);
    return std::unexpected(plan_allocation_failure());
  }
  ConstantPreparation constants(allocator);
  CompileReport report;
  Status status =
      initialize_plan_metadata(*plan, *concrete,
                               std::span<const compute::TensorDesc>(
                                   descriptions.data(), descriptions.size()));
  if (status.ok())
    status = bind_constants(*plan, *concrete, options, constants);
  if (status.ok())
    status = bind_effect_resources(*plan, *concrete, options);
  if (status.ok())
    status = prepare_placed_steps(*plan, *specialized, *concrete, environment,
                                  options, provider_policy, constants, routes,
                                  report, packing, adapters);
  if (status.ok())
    status = build_external_input_topology(*plan);
  if (!status.ok()) {
    detail::release(plan);
    return std::unexpected(std::move(status));
  }
  CompiledGraph result;
  auto *report_state = detail::CompileReportAccess::state(report);
  report_state->total_search = plan->search;
  result.report = std::move(report);
  result.executable = detail::ExecutableAccess::adopt(plan);
  return result;
}

// The table is compile input, not a cache. Validate all domain points and
// duplicates before preparing any exact artifacts. Temporary values die before
// child compilation; the final index references the children's owned
// signatures.
Status
index_exact_variants(detail::PlanState &plan,
                     std::span<const ExactShapeVariant> variants,
                     std::span<const ConstantBinding> constants) noexcept {
  const auto &graph = *detail::GraphAccess::state(plan.domain_graph);
  if (constants.size() != graph.constants.size())
    return Status::InvalidArgument(
        "compile requires exactly one binding for every graph constant");
  const std::size_t dimensions = graph.dimensions.size();
  if (dimensions != 0 && variants.size() > SIZE_MAX / dimensions)
    return Status::ResourceExhausted("exact variant dimension table overflows");
  container::SmallVector<std::int64_t, 0, mem::AllocatorRef> points(
      plan.allocator);
  ShapeValues values(plan.allocator);
  detail::TensorDescriptions descriptions(plan.allocator);
  container::SmallVector<std::uint8_t, 0, mem::AllocatorRef> seen(plan.allocator);
  if (!points.try_resize(variants.size() * dimensions) ||
      !plan.variant_order.try_resize(variants.size()) ||
      !seen.try_resize(graph.values.size()))
    return plan_allocation_failure();
  for (const auto &binding : constants) {
    if (!detail::owns(graph, binding.value) ||
        binding.value.kind() != ValueKind::kConstant ||
        !binding.tensor.valid() ||
        (binding.policy != ConstantPolicy::kBorrow &&
         binding.policy != ConstantPolicy::kMaterialize))
      return Status::InvalidArgument("exact variant constant binding is invalid");
    if (seen[binding.value.ordinal()] != 0)
      return Status::InvalidArgument("constant binding is duplicated");
    seen[binding.value.ordinal()] = 1;
  }
  for (std::size_t index = 0; index < variants.size(); ++index) {
    HYLE_RETURN_IF_ERROR(bind_shape_values(graph, variants[index].shape,
                                           plan.allocator, values));
    auto signature = specialize_bound_descriptions(
        graph, values, plan.allocator, descriptions);
    if (!signature)
      return std::move(signature).error();
    for (const auto &binding : constants)
      if (binding.tensor.description() != descriptions[binding.value.ordinal()])
        return Status::InvalidArgument(
            "constant binding does not match an exact variant description");
    for (std::size_t axis = 0; axis < dimensions; ++axis)
      points[index * dimensions + axis] = values[axis];
    plan.variant_order[index] = static_cast<std::uint32_t>(index);
  }
  const auto less = [&](std::uint32_t left, std::uint32_t right) noexcept {
    for (std::size_t axis = 0; axis < dimensions; ++axis) {
      const auto a = points[static_cast<std::size_t>(left) * dimensions + axis];
      const auto b =
          points[static_cast<std::size_t>(right) * dimensions + axis];
      if (a != b)
        return a < b;
    }
    return false;
  };
  std::sort(plan.variant_order.begin(), plan.variant_order.end(), less);
  for (std::size_t index = 1; index < plan.variant_order.size(); ++index)
    if (!less(plan.variant_order[index - 1], plan.variant_order[index]))
      return Status::InvalidArgument("exact variant shape is duplicated");
  return Status::Ok();
}

Status accumulate_search(CompileSearchReport &total,
                         const CompileSearchReport &part) noexcept {
  constexpr std::array counts{&CompileSearchReport::operation_proposals,
                              &CompileSearchReport::region_proposals,
                              &CompileSearchReport::region_proposal_nodes,
                              &CompileSearchReport::placement_attempts,
                              &CompileSearchReport::region_route_rejections,
                              &CompileSearchReport::operation_route_rejections,
                              &CompileSearchReport::route_search_states,
                              &CompileSearchReport::route_plan_rejections,
                              &CompileSearchReport::feasible_plans};
  for (const auto field : counts) {
    if (part.*field > SIZE_MAX - total.*field)
      return Status::ResourceExhausted(
          "aggregate graph compile counters overflow");
    total.*field += part.*field;
  }
  constexpr std::array limits{
      &CompileSearchReport::region_proposal_limit_reached,
      &CompileSearchReport::per_provider_proposal_limit_reached,
      &CompileSearchReport::region_node_limit_reached,
      &CompileSearchReport::operation_proposal_limit_reached,
      &CompileSearchReport::route_search_limit_reached,
      &CompileSearchReport::placement_limit_reached};
  for (const auto field : limits)
    total.*field |= part.*field;
  return Status::Ok();
}

Result<CompileOptions>
remaining_compile_budget(CompileOptions options,
                         const CompileSearchReport &spent,
                         CompiledStorageReport storage) noexcept {
  struct Counter final {
    std::size_t CompileOptions::*cap;
    std::size_t CompileSearchReport::*count;
  };
  constexpr std::array counters{
      Counter{&CompileOptions::max_placement_attempts,
              &CompileSearchReport::placement_attempts},
      Counter{&CompileOptions::max_route_search_states,
              &CompileSearchReport::route_search_states},
      Counter{&CompileOptions::max_operation_proposals,
              &CompileSearchReport::operation_proposals},
      Counter{&CompileOptions::max_region_proposals,
              &CompileSearchReport::region_proposals},
      Counter{&CompileOptions::max_region_proposal_nodes,
              &CompileSearchReport::region_proposal_nodes}};
  for (const auto counter : counters) {
    if (spent.*(counter.count) > options.*(counter.cap))
      return std::unexpected(
          Status::Internal("graph compile exceeded its budget"));
    options.*(counter.cap) -= spent.*(counter.count);
  }
  if (options.max_route_search_states == 0)
    return std::unexpected(Status::ResourceExhausted(
        "graph exact variant has no remaining search budget"));
  if (options.compiled_storage.max_bytes != SIZE_MAX) {
    if (storage.known_bytes() > options.compiled_storage.max_bytes)
      return std::unexpected(
          Status::Internal("graph compile exceeded its storage budget"));
    options.compiled_storage.max_bytes -= storage.known_bytes();
  }
  return options;
}

Result<CompiledGraph> compile_family(const Graph &graph,
                                     CompileEnvironment environment,
                                     CompileOptions options,
                                     std::span<const detail::RegionAdapter> adapters = {}) noexcept {
  const auto variants = options.exact_variants;
  const bool finite = options.shape_policy == CompileShapePolicy::kFiniteExact;
  if (!variants.empty() &&
      options.shape_policy != CompileShapePolicy::kDeclaredDomain && !finite)
    return std::unexpected(Status::InvalidArgument(
        "exact variants require declared-domain or finite-exact compilation"));
  if (finite && (variants.empty() || !options.shape.empty()))
    return std::unexpected(Status::InvalidArgument(
        "finite-exact compilation requires a nonempty table and no single shape"));
  if (variants.size() > (std::numeric_limits<std::uint32_t>::max)())
    return std::unexpected(
        Status::ResourceExhausted("exact variant index space is exhausted"));
  options.exact_variants = {};
  detail::ConstantArtifactCache packing;
  auto parent_options = options;
  if (finite) {
    parent_options.shape_policy = CompileShapePolicy::kExact;
    parent_options.shape = variants.front().shape;
  }
  HYLE_ASSIGN_OR_RETURN(auto result,
                        compile_plan(graph, environment, parent_options, packing,
                                     false, {}, {}, adapters, finite));
  packing.commit_plan();
  if (variants.empty())
    return result;

  auto &plan = *detail::ExecutableAccess::state(result.executable);
  auto &report = *detail::CompileReportAccess::state(result.report);
  HYLE_RETURN_IF_ERROR(index_exact_variants(plan, variants, options.constants));
  if (!plan.exact_variants.try_reserve(variants.size()) ||
      !report.exact_variants.try_reserve(variants.size()))
    return std::unexpected(plan_allocation_failure());
  for (std::size_t index = 0; index < variants.size(); ++index) {
    HYLE_ASSIGN_OR_RETURN(
        auto child_options,
        remaining_compile_budget(options, report.total_search,
                                 report.total_compiled_storage));
    child_options.shape_policy = CompileShapePolicy::kExact;
    child_options.shape = variants[index].shape;
    auto child_result = compile_plan(graph, environment, child_options,
                                     packing, true, plan.invocation_storage,
                                     plan.compilation_storage, adapters);
    if (!child_result) {
      const std::string_view prefix = finite ? "finite exact variant index "
                                             : "exact variant index ";
      std::array<char, 64> location{};
      std::copy(prefix.begin(), prefix.end(), location.begin());
      const auto number = std::to_chars(location.data() + prefix.size(),
                                       location.data() + location.size(), index);
      HYLE_FATAL_ASSERT(number.ec == std::errc{});
      return std::unexpected(std::move(child_result).error().WithContext(
          std::string_view(location.data(), number.ptr - location.data())));
    }
    auto child = std::move(*child_result);
    packing.commit_plan();
    auto &child_plan = *detail::ExecutableAccess::state(child.executable);
    child_plan.exact_variant_index = index;
    HYLE_RETURN_IF_ERROR(
        accumulate_search(report.total_search, child.report.search()));
    HYLE_RETURN_IF_ERROR(add_compiled_storage(
        report.total_compiled_storage, child.report.compiled_storage(),
        child_plan.shared_artifact_storage));
    if (finite) {
      plan.supports_concurrent_invocations &=
          child_plan.supports_concurrent_invocations;
      report.supports_concurrent_invocations = plan.supports_concurrent_invocations;
    }
    if (!plan.exact_variants.try_push_back(std::move(child.executable)) ||
        !report.exact_variants.try_push_back(std::move(child.report)))
      return std::unexpected(plan_allocation_failure());
  }
  return result;
}

} // namespace

namespace {

Result<CompiledGraph> compile_admitted(
    const Graph &graph, CompileEnvironment environment, CompileOptions options,
    std::span<const detail::RegionAdapter> adapters = {}) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.graph.compile");
  if (options.resource_budget) {
    const auto unknown = options.resource_budget.allows_unattributed()
        ? UnknownStoragePolicy::kAllow : UnknownStoragePolicy::kReject;
    if (!options.compilation_storage)
      options.compilation_storage = CompilationStorageLimits{SIZE_MAX, SIZE_MAX, unknown};
    if (!options.invocation_storage)
      options.invocation_storage = InvocationStorageLimits{SIZE_MAX, unknown};
  }
  HYLE_ASSIGN_OR_RETURN(auto result,
                        compile_family(graph, environment, options,
                                       adapters));
  const auto &plan = *detail::ExecutableAccess::state(result.executable);
  if (plan.compilation_storage) {
    // All family compilation scratch has died. Retained reports, signatures,
    // plans and the shared budget owner are included; borrowed inputs are not.
    const auto usage = plan.compilation_storage->usage();
    if (usage.live_bytes > options.compilation_storage->max_resident_bytes)
      return std::unexpected(Status::ResourceExhausted(
          "graph family exceeds compilation resident storage budget"));
    const CompilationStorageReport snapshot{usage.live_bytes, usage.peak_bytes,
                                            usage.has_unknown_storage};
    auto &report = *detail::CompileReportAccess::state(result.report);
    report.compilation_storage = snapshot;
    for (const auto &child : report.exact_variants)
      detail::CompileReportAccess::state(child)->compilation_storage = snapshot;
  }
  return result;
}

} // namespace

Result<CompiledGraph> compile(const Graph &graph,
                              CompileEnvironment environment,
                              CompileOptions options) noexcept {
  return compile_admitted(graph, environment, options);
}

Result<CompiledGraph> detail::compile_with_adapters(
    const Graph &graph, CompileEnvironment environment, CompileOptions options,
    std::span<const RegionAdapter> adapters) noexcept {
  return compile_admitted(graph, environment, options, adapters);
}

} // namespace hyle::graph
