#include "hyle/graph/graph.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "graph_state.h"

namespace hyle::graph {
namespace {

std::uint64_t next_generation() noexcept {
  static std::atomic<std::uint64_t> next{1};
  std::uint64_t generation = next.fetch_add(1, std::memory_order_relaxed);
  // Zero is the invalid/moved-from sentinel. Wraparound would require more
  // builders than a process can physically create, but keep the invariant
  // explicit instead of ever publishing a forgeable invalid identity.
  while (generation == 0) {
    generation = next.fetch_add(1, std::memory_order_relaxed);
  }
  return generation;
}

Status invalid_graph() noexcept {
  return Status::FailedPrecondition("graph is invalid");
}

Status invalid_builder() noexcept {
  return Status::FailedPrecondition("graph builder is invalid");
}

Status allocation_failure() noexcept {
  return Status::ResourceExhausted("graph metadata allocation failed");
}

Status symbolic_description() noexcept {
  return Status::FailedPrecondition(
      "graph value is symbolic and requires a concrete specialization");
}

Status validate_bounds(DimensionBounds bounds) noexcept {
  const compute::detail::ProviderShapeRange range{
      bounds.minimum, bounds.maximum, bounds.multiple_of};
  return range.valid()
             ? Status::Ok()
             : Status::InvalidArgument("invalid graph dimension bounds");
}

template <class State>
Status validate_spec(const State &state, const TensorSpec &spec) noexcept {
  if (!spec.valid())
    return Status::InvalidArgument("graph tensor spec is invalid");
  for (Dimension dimension : spec.dimensions()) {
    if (!dimension.valid()) {
      return Status::InvalidArgument(
          "graph tensor spec contains an invalid extent");
    }
    if (dimension.is_symbol() && !detail::owns(state, dimension.symbol())) {
      return Status::InvalidArgument(
          "tensor dimension belongs to another graph builder");
    }
  }
  return Status::Ok();
}

Status canonicalize_dependencies(detail::BuilderState &state) noexcept {
  using Counts = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  Counts counts(state.allocator);
  Counts cursors(state.allocator);
  Counts seen(state.allocator);
  if (!counts.try_resize(state.nodes.size()) ||
      !cursors.try_resize(state.nodes.size()) ||
      !seen.try_resize(state.nodes.size()) ||
      !state.dependencies.try_resize(state.control_edges.size())) {
    return allocation_failure();
  }
  for (std::size_t index = 0; index < state.nodes.size(); ++index) {
    counts[index] = 0;
    seen[index] = detail::kNoNode;
  }
  for (const detail::ControlEdgeRecord edge : state.control_edges) {
    if (counts[edge.node] == (std::numeric_limits<std::uint32_t>::max)()) {
      return Status::ResourceExhausted(
          "graph control dependency count is exhausted");
    }
    ++counts[edge.node];
  }

  std::uint32_t begin = 0;
  for (std::size_t index = 0; index < state.nodes.size(); ++index) {
    state.nodes[index].dependency_begin = begin;
    state.nodes[index].dependency_count = counts[index];
    cursors[index] = begin;
    begin += counts[index];
  }
  for (const detail::ControlEdgeRecord edge : state.control_edges)
    state.dependencies[cursors[edge.node]++] = edge.prerequisite;

  std::size_t write = 0;
  for (std::size_t node_index = 0; node_index < state.nodes.size();
       ++node_index) {
    detail::NodeRecord &node = state.nodes[node_index];
    const std::uint32_t old_begin = node.dependency_begin;
    const std::uint32_t old_count = node.dependency_count;
    node.dependency_begin = static_cast<std::uint32_t>(write);
    const std::uint32_t marker = static_cast<std::uint32_t>(node_index);
    for (std::size_t index = 0; index < old_count; ++index) {
      const std::uint32_t prerequisite = state.dependencies[old_begin + index];
      if (seen[prerequisite] == marker)
        continue;
      seen[prerequisite] = marker;
      state.dependencies[write++] = prerequisite;
    }
    node.dependency_count =
        static_cast<std::uint32_t>(write - node.dependency_begin);
  }
  state.dependencies.try_resize(write);
  return Status::Ok();
}

constexpr bool fits_value_ordinal(std::size_t size) noexcept {
  return size <= (std::numeric_limits<std::uint32_t>::max)();
}

constexpr bool fits_node_ordinal(std::size_t size) noexcept {
  // UINT32_MAX is reserved as ValueRecord::producer's "no producer" value.
  return size < (std::numeric_limits<std::uint32_t>::max)();
}

constexpr bool fits_dimension_ordinal(std::size_t size) noexcept {
  return size < compute::detail::kFixedShapeAxis;
}

Result<compute::detail::ShapeOperand>
shape_operand(const detail::BuilderState &state, Dimension value) noexcept {
  if (!value.valid() ||
      (value.is_symbol() && !detail::owns(state, value.symbol()))) {
    return std::unexpected(Status::InvalidArgument(
        "shape operand must be a constant or a dimension from this builder"));
  }
  return value.is_fixed()
             ? compute::detail::ShapeOperand::fixed(value.fixed_value())
             : compute::detail::ShapeOperand::value(value.symbol().ordinal());
}

compute::detail::ProviderShapeRange
operand_bounds(const detail::BuilderState &state,
               compute::detail::ShapeOperand value) noexcept {
  if (!value.is_fixed())
    return state.dimensions[value.ordinal].bounds;
  return {value.constant, value.constant,
          value.constant == 0 ? 1 : value.constant};
}

Result<DimensionId> derived_dimension(detail::BuilderState *state,
                                      compute::detail::ShapeDimensionKind kind,
                                      Dimension left,
                                      Dimension right) noexcept {
  if (state == nullptr)
    return std::unexpected(invalid_builder());
  auto lhs = shape_operand(*state, left);
  if (!lhs)
    return std::unexpected(std::move(lhs).error());
  auto rhs = shape_operand(*state, right);
  if (!rhs)
    return std::unexpected(std::move(rhs).error());
  auto bounds = compute::detail::infer_shape_range(
      kind, operand_bounds(*state, *lhs), operand_bounds(*state, *rhs));
  if (!bounds)
    return std::unexpected(std::move(bounds).error());
  if (!fits_dimension_ordinal(state->dimensions.size())) {
    return std::unexpected(Status::ResourceExhausted(
        "graph dimension identity space is exhausted"));
  }
  const auto ordinal = static_cast<std::uint32_t>(state->dimensions.size());
  if (!state->dimensions.try_push_back(
          detail::DimensionRecord{*bounds, kind, *lhs, *rhs}))
    return std::unexpected(allocation_failure());
  return detail::GraphIdentity::dimension(state->generation, ordinal);
}

Status shape_guard(detail::BuilderState *state,
                   compute::detail::ShapeGuardKind kind, Dimension left,
                   Dimension right) noexcept {
  if (state == nullptr)
    return invalid_builder();
  auto lhs = shape_operand(*state, left);
  if (!lhs)
    return std::move(lhs).error();
  auto rhs = shape_operand(*state, right);
  if (!rhs)
    return std::move(rhs).error();
  if (!state->shape_guards.try_push_back(
          compute::detail::ShapeGuard{kind, *lhs, *rhs}))
    return allocation_failure();
  return Status::Ok();
}

static_assert(fits_value_ordinal(
    static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)())));
static_assert(!fits_node_ordinal(
    static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)())));

Status validate_dimensions(std::span<const Dimension> dimensions) noexcept {
  if (dimensions.size() > compute::kMaxTensorRank)
    return Status::InvalidArgument("graph metadata rank exceeds limit");
  for (const auto dimension : dimensions)
    if (!dimension.valid())
      return Status::InvalidArgument("graph metadata dimension is invalid");
  return Status::Ok();
}

Result<std::size_t> view_axis(std::int32_t axis, std::size_t rank) noexcept {
  const auto normalized =
      axis < 0 ? axis + static_cast<std::int32_t>(rank) : axis;
  if (normalized < 0 || static_cast<std::size_t>(normalized) >= rank)
    return std::unexpected(
        Status::InvalidArgument("graph view axis is outside its input rank"));
  return static_cast<std::size_t>(normalized);
}

// All fallible preparation precedes publication, including resolving fixed
// output descriptions. Failed construction may retain capacity, not content.
Result<NodeId>
append_node(detail::BuilderState &state,
            std::variant<compute::Operation, ViewSpec> descriptor,
            std::span<const ValueId> inputs,
            std::span<const TensorSpec> outputs) noexcept {
  if (!fits_node_ordinal(state.nodes.size()) ||
      inputs.size() > (std::numeric_limits<std::uint16_t>::max)() ||
      outputs.size() > (std::numeric_limits<std::uint8_t>::max)() ||
      (!outputs.empty() &&
       state.values.size() > (std::numeric_limits<std::uint32_t>::max)() -
                                 (outputs.size() - 1)) ||
      state.edges.size() >
          (std::numeric_limits<std::uint32_t>::max)() - inputs.size())
    return std::unexpected(
        Status::ResourceExhausted("graph identity space is exhausted"));
  container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef> resolved(
      state.allocator);
  if (!resolved.try_resize(outputs.size()))
    return std::unexpected(allocation_failure());
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    auto value = outputs[index].concrete_description();
    if (value)
      resolved[index] = *value;
    else if (!value.error().IsFailedPrecondition())
      return std::unexpected(std::move(value).error());
  }
  const auto *view = std::get_if<ViewSpec>(&descriptor);
  if (view != nullptr && (!fits_value_ordinal(state.views.size()) ||
                          !state.views.try_reserve(state.views.size() + 1)))
    return std::unexpected(allocation_failure());
  if (!state.edges.try_reserve(state.edges.size() + inputs.size()) ||
      !state.nodes.try_reserve(state.nodes.size() + 1) ||
      !state.values.try_reserve(state.values.size() + outputs.size()))
    return std::unexpected(allocation_failure());
  const auto ordinal = static_cast<std::uint32_t>(state.nodes.size());
  const auto edge_begin = static_cast<std::uint32_t>(state.edges.size());
  const auto output_begin = static_cast<std::uint32_t>(state.values.size());
  std::variant<compute::Operation, std::uint32_t> stored;
  if (view != nullptr) {
    stored = static_cast<std::uint32_t>(state.views.size());
    const bool added = state.views.try_push_back(*view);
    HYLE_FATAL_ASSERT(added);
  } else {
    stored = std::move(std::get<compute::Operation>(descriptor));
  }
  for (const auto input : inputs) {
    const bool added = state.edges.try_push_back(input.ordinal());
    HYLE_FATAL_ASSERT(added);
  }
  const bool node_added = state.nodes.try_push_back(detail::NodeRecord{
      .descriptor = std::move(stored),
      .input_begin = edge_begin,
      .output_begin = output_begin,
      .input_count = static_cast<std::uint16_t>(inputs.size()),
      .output_count = static_cast<std::uint8_t>(outputs.size()),
  });
  HYLE_FATAL_ASSERT(node_added);
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const bool added = state.values.try_push_back(detail::ValueRecord{
        .spec = outputs[index],
        .description = resolved[index],
        .kind = ValueKind::kNodeOutput,
        .producer = ordinal,
    });
    HYLE_FATAL_ASSERT(added);
  }
  return detail::GraphIdentity::node(state.generation, ordinal);
}

} // namespace

Result<ViewSpec>
ViewSpec::reshape(std::span<const Dimension> dimensions) noexcept {
  HYLE_RETURN_IF_ERROR(validate_dimensions(dimensions));
  ViewSpec result;
  std::copy(dimensions.begin(), dimensions.end(), result.dimensions_.begin());
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.valid_ = true;
  return result;
}

Result<ViewSpec> ViewSpec::slice(std::int32_t axis, Dimension start,
                                 Dimension count, std::int64_t step) noexcept {
  if (!start.valid() || !count.valid())
    return std::unexpected(
        Status::InvalidArgument("graph slice start/count is invalid"));
  // Reuse the concrete schema's fixed axis and step contract.
  auto checked = compute::View::slice(axis, 0, 0, step);
  if (!checked)
    return std::unexpected(std::move(checked).error());
  ViewSpec result;
  result.kind_ = compute::ViewKind::kSlice;
  result.axis_ = axis;
  result.start_ = start;
  result.count_ = count;
  result.step_ = step;
  result.valid_ = true;
  return result;
}

Result<ViewSpec>
ViewSpec::permute(std::span<const std::int32_t> axes) noexcept {
  auto checked = compute::View::permute(axes);
  if (!checked)
    return std::unexpected(std::move(checked).error());
  ViewSpec result;
  result.kind_ = compute::ViewKind::kPermute;
  std::copy(axes.begin(), axes.end(), result.permutation_.begin());
  result.rank_ = static_cast<std::uint8_t>(axes.size());
  result.valid_ = true;
  return result;
}

Result<ViewSpec>
ViewSpec::broadcast_to(std::span<const Dimension> dimensions) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto result, reshape(dimensions));
  result.kind_ = compute::ViewKind::kBroadcast;
  return result;
}

bool ViewSpec::is_symbolic() const noexcept {
  if (!valid())
    return false;
  if (kind_ == compute::ViewKind::kSlice)
    return start_.is_symbol() || count_.is_symbol();
  for (const auto dimension : dimensions())
    if (dimension.is_symbol())
      return true;
  return false;
}

Result<compute::View>
detail::bind_view(const ViewSpec &view,
                  std::span<const std::int64_t> values) noexcept {
  if (!view.valid())
    return std::unexpected(
        Status::InvalidArgument("graph ViewSpec is invalid"));
  const auto value = [&](Dimension dimension) -> Result<std::int64_t> {
    if (dimension.is_fixed())
      return dimension.fixed_value();
    if (dimension.symbol().ordinal() >= values.size())
      return std::unexpected(Status::FailedPrecondition(
          "graph ViewSpec needs concrete dimension bindings"));
    return values[dimension.symbol().ordinal()];
  };
  if (view.kind() == compute::ViewKind::kSlice) {
    HYLE_ASSIGN_OR_RETURN(const auto start, value(view.start()));
    HYLE_ASSIGN_OR_RETURN(const auto count, value(view.count()));
    return compute::View::slice(view.axis(), start, count, view.step());
  }
  if (view.kind() == compute::ViewKind::kPermute)
    return compute::View::permute(view.permutation());
  std::array<std::int64_t, compute::kMaxTensorRank> dimensions{};
  for (std::size_t axis = 0; axis < view.dimensions().size(); ++axis) {
    HYLE_ASSIGN_OR_RETURN(dimensions[axis], value(view.dimensions()[axis]));
  }
  HYLE_ASSIGN_OR_RETURN(auto shape,
                        compute::TensorShape::create(
                            {dimensions.data(), view.dimensions().size()}));
  return view.kind() == compute::ViewKind::kReshape
             ? compute::View::reshape(shape)
             : compute::View::broadcast_to(shape);
}

Result<compute::Operation>
detail::bind_operation(const GraphState &graph, const NodeRecord &node,
                       std::span<const std::int64_t> values) noexcept {
  if (const auto *view = view_spec(graph, node)) {
    HYLE_ASSIGN_OR_RETURN(auto bound, bind_view(*view, values));
    return compute::Operation(bound);
  }
  return node.operation();
}

Result<TensorSpec>
TensorSpec::concrete(compute::TensorDesc description) noexcept {
  if (!description.valid()) {
    return std::unexpected(Status::InvalidArgument(
        "concrete graph tensor description is invalid"));
  }
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());

  TensorSpec result;
  result.concrete_ = description;
  result.format_ = description.storage_format();
  result.type_ = description.data_type();
  result.rank_ = static_cast<std::uint8_t>(description.rank());
  result.kind_ = TensorSpecKind::kConcrete;
  for (std::size_t index = 0; index < description.rank(); ++index) {
    // Fixed and symbolic extents share TensorDesc's nonnegative size domain.
    result.dimensions_[index] =
        Dimension::fixed(description.dimensions()[index]);
  }
  return result;
}

Result<TensorSpec>
TensorSpec::dense(compute::DataType type,
                  std::span<const Dimension> dimensions) noexcept {
  if (type == compute::DataType::kUnknown ||
      dimensions.size() > compute::kMaxTensorRank) {
    return std::unexpected(Status::InvalidArgument(
        "symbolic dense tensor type or rank is invalid"));
  }
  TensorSpec result;
  result.type_ = type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.kind_ = TensorSpecKind::kDense;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (!dimensions[index].valid()) {
      return std::unexpected(Status::InvalidArgument(
          "symbolic dense tensor extent must be positive or a valid symbol"));
    }
    result.dimensions_[index] = dimensions[index];
  }
  if (!result.is_symbolic()) {
    auto concrete = result.concrete_description();
    if (!concrete)
      return std::unexpected(std::move(concrete).error());
  }
  return result;
}

Result<TensorSpec>
TensorSpec::blocked_dense(compute::DataType logical_type,
                          std::span<const Dimension> dimensions,
                          compute::StorageFormat format) noexcept {
  if (logical_type == compute::DataType::kUnknown || dimensions.empty() ||
      dimensions.size() > compute::kMaxTensorRank ||
      format.kind() != compute::StorageFormatKind::kBlocked) {
    return std::unexpected(Status::InvalidArgument(
        "symbolic blocked tensor type, rank, or format is invalid"));
  }
  TensorSpec result;
  result.format_ = format;
  result.type_ = logical_type;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.kind_ = TensorSpecKind::kBlockedDense;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (!dimensions[index].valid()) {
      return std::unexpected(Status::InvalidArgument(
          "symbolic blocked tensor extent must be positive or a valid symbol"));
    }
    result.dimensions_[index] = dimensions[index];
  }
  if (!result.is_symbolic()) {
    auto concrete = result.concrete_description();
    if (!concrete)
      return std::unexpected(std::move(concrete).error());
  }
  return result;
}

bool TensorSpec::is_symbolic() const noexcept {
  if (!valid() || kind_ == TensorSpecKind::kConcrete)
    return false;
  if (kind_ == TensorSpecKind::kInferred)
    return true;
  for (Dimension dimension : dimensions()) {
    if (dimension.is_symbol())
      return true;
  }
  return false;
}

Result<TensorSpec>
TensorSpec::inferred(compute::DataType type,
                     std::span<const Dimension> dimensions,
                     compute::StorageFormat format) noexcept {
  HYLE_RETURN_IF_ERROR(validate_dimensions(dimensions));
  if (compute::data_type_size(type) == 0)
    return std::unexpected(
        Status::InvalidArgument("inferred graph tensor type is invalid"));
  TensorSpec result;
  result.type_ = type;
  result.format_ = format;
  result.rank_ = static_cast<std::uint8_t>(dimensions.size());
  result.kind_ = TensorSpecKind::kInferred;
  std::copy(dimensions.begin(), dimensions.end(), result.dimensions_.begin());
  return result;
}

Result<compute::TensorDesc> TensorSpec::concrete_description() const noexcept {
  if (!valid())
    return std::unexpected(
        Status::FailedPrecondition("tensor spec is invalid"));
  if (kind_ == TensorSpecKind::kConcrete)
    return concrete_;
  if (is_symbolic())
    return std::unexpected(symbolic_description());

  std::array<std::int64_t, compute::kMaxTensorRank> extents{};
  for (std::size_t index = 0; index < rank_; ++index)
    extents[index] = dimensions_[index].fixed_value();
  const std::span<const std::int64_t> shape(extents.data(), rank_);
  if (kind_ == TensorSpecKind::kDense)
    return compute::TensorDesc::dense(type_, shape);
  if (kind_ == TensorSpecKind::kBlockedDense)
    return compute::TensorDesc::blocked_dense(type_, shape, format_);
  return std::unexpected(Status::Internal("tensor spec kind is invalid"));
}

Graph::Graph(const Graph &other) noexcept : state_(other.state_) {
  detail::retain(state_);
}

Graph &Graph::operator=(const Graph &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

Graph::Graph(Graph &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Graph &Graph::operator=(Graph &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Graph::~Graph() { detail::release(state_); }

std::size_t Graph::value_count() const noexcept {
  return state_ != nullptr ? state_->values.size() : 0;
}

std::size_t Graph::dimension_count() const noexcept {
  return state_ != nullptr ? state_->dimensions.size() : 0;
}

std::size_t Graph::independent_dimension_count() const noexcept {
  return state_ != nullptr
             ? compute::detail::ShapeProgramView{state_->dimensions,
                                                 state_->shape_guards}
                   .input_count()
             : 0;
}

std::size_t Graph::input_count() const noexcept {
  return state_ != nullptr ? state_->inputs.size() : 0;
}

std::size_t Graph::constant_count() const noexcept {
  return state_ != nullptr ? state_->constants.size() : 0;
}

std::size_t Graph::node_count() const noexcept {
  return state_ != nullptr ? state_->nodes.size() : 0;
}

std::size_t Graph::output_count() const noexcept {
  return state_ != nullptr ? state_->outputs.size() : 0;
}

std::size_t Graph::effect_resource_count() const noexcept {
  return state_ != nullptr ? state_->effect_resource_count : 0;
}

Result<EffectResourceId>
Graph::effect_resource_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->effect_resource_count)
    return std::unexpected(
        Status::OutOfRange("graph effect resource index is out of range"));
  return detail::GraphIdentity::effect_resource(
      state_->generation, static_cast<std::uint32_t>(index));
}

Result<DimensionId> Graph::dimension_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->dimensions.size()) {
    return std::unexpected(
        Status::OutOfRange("graph dimension index is out of range"));
  }
  return detail::GraphIdentity::dimension(state_->generation,
                                          static_cast<std::uint32_t>(index));
}

Result<DimensionBounds>
Graph::dimension_bounds(DimensionId dimension) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, dimension)) {
    return std::unexpected(
        Status::InvalidArgument("dimension does not belong to this graph"));
  }
  const auto bounds = state_->dimensions[dimension.ordinal()].bounds;
  return DimensionBounds{bounds.minimum, bounds.maximum, bounds.multiple_of};
}

Result<bool>
Graph::dimension_is_independent(DimensionId dimension) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, dimension)) {
    return std::unexpected(
        Status::InvalidArgument("dimension does not belong to this graph"));
  }
  return state_->dimensions[dimension.ordinal()].kind ==
         compute::detail::ShapeDimensionKind::kInput;
}

Result<ValueId> Graph::input_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->inputs.size()) {
    return std::unexpected(
        Status::OutOfRange("graph input index is out of range"));
  }
  return detail::value_id(*state_, state_->inputs[index]);
}

Result<const compute::TensorDesc &>
Graph::input_description(std::size_t index) const noexcept {
  auto id = input_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  const compute::TensorDesc &description =
      state_->values[id->ordinal()].description;
  if (!description.valid())
    return std::unexpected(symbolic_description());
  return description;
}

Result<const TensorSpec &> Graph::input_spec(std::size_t index) const noexcept {
  auto id = input_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  return state_->values[id->ordinal()].spec;
}

Result<ValueId> Graph::constant_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->constants.size()) {
    return std::unexpected(
        Status::OutOfRange("graph constant index is out of range"));
  }
  return detail::value_id(*state_, state_->constants[index]);
}

Result<const compute::TensorDesc &>
Graph::constant_description(std::size_t index) const noexcept {
  auto id = constant_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  const compute::TensorDesc &description =
      state_->values[id->ordinal()].description;
  if (!description.valid())
    return std::unexpected(symbolic_description());
  return description;
}

Result<const TensorSpec &>
Graph::constant_spec(std::size_t index) const noexcept {
  auto id = constant_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  return state_->values[id->ordinal()].spec;
}

Result<NodeId> Graph::node_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->nodes.size()) {
    return std::unexpected(
        Status::OutOfRange("graph node index is out of range"));
  }
  return detail::GraphIdentity::node(state_->generation,
                                     static_cast<std::uint32_t>(index));
}

Result<const compute::Operation &>
Graph::operation(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  if (state_->nodes[node.ordinal()].view_index())
    return std::unexpected(Status::FailedPrecondition(
        "logical ViewSpec requires specialization; inspect view_spec()"));
  return state_->nodes[node.ordinal()].operation();
}

Result<compute::OperationKind>
Graph::operation_kind(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node))
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  return state_->nodes[node.ordinal()].kind();
}

Result<const ViewSpec &> Graph::view_spec(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node))
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  const auto *view = detail::view_spec(*state_, state_->nodes[node.ordinal()]);
  if (view == nullptr)
    return std::unexpected(Status::FailedPrecondition(
        "node holds a concrete Operation, not a ViewSpec"));
  return *view;
}

Result<std::size_t> Graph::node_input_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].input_count;
}

Result<ValueId> Graph::node_input(NodeId node,
                                  std::size_t input_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (input_index >= record.input_count) {
    return std::unexpected(
        Status::OutOfRange("graph node input index is out of range"));
  }
  return detail::value_id(*state_,
                          state_->edges[record.input_begin + input_index]);
}

Result<std::size_t> Graph::node_dependency_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].dependency_count;
}

Result<NodeId>
Graph::node_dependency(NodeId node,
                       std::size_t dependency_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (dependency_index >= record.dependency_count) {
    return std::unexpected(
        Status::OutOfRange("graph node dependency index is out of range"));
  }
  return detail::GraphIdentity::node(
      state_->generation,
      state_->dependencies[record.dependency_begin + dependency_index]);
}

Result<std::size_t> Graph::node_resource_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node))
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  return state_->nodes[node.ordinal()].traits().resource_count;
}

Result<EffectResourceId> Graph::node_resource(NodeId node,
                                              std::size_t port) const noexcept {
  auto count = node_resource_count(node);
  if (!count)
    return std::unexpected(std::move(count).error());
  if (port >= *count)
    return std::unexpected(
        Status::OutOfRange("graph node resource port is out of range"));
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  return detail::GraphIdentity::effect_resource(
      state_->generation, state_->resource_edges[record.resource_begin + port]);
}

Result<std::size_t> Graph::node_output_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].output_count;
}

Result<ValueId> Graph::result(NodeId node,
                              std::size_t output_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (output_index >= record.output_count) {
    return std::unexpected(
        Status::OutOfRange("graph node output index is out of range"));
  }
  return detail::value_id(
      *state_, record.output_begin + static_cast<std::uint32_t>(output_index));
}

Result<NodeId> Graph::producer(ValueId value) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this graph"));
  }
  const std::uint32_t producer = state_->values[value.ordinal()].producer;
  if (producer == detail::kNoNode) {
    return std::unexpected(Status::FailedPrecondition(
        "external graph value has no producer node"));
  }
  return detail::GraphIdentity::node(state_->generation, producer);
}

Result<ValueId> Graph::output_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->outputs.size()) {
    return std::unexpected(
        Status::OutOfRange("graph output index is out of range"));
  }
  return detail::value_id(*state_, state_->outputs[index]);
}

Result<const compute::TensorDesc &>
Graph::output_description(std::size_t index) const noexcept {
  auto id = output_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  const compute::TensorDesc &description =
      state_->values[id->ordinal()].description;
  if (!description.valid())
    return std::unexpected(symbolic_description());
  return description;
}

Result<const TensorSpec &>
Graph::output_spec(std::size_t index) const noexcept {
  auto id = output_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  return state_->values[id->ordinal()].spec;
}

Result<const compute::TensorDesc &>
Graph::value_description(ValueId value) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this graph"));
  }
  const compute::TensorDesc &description =
      state_->values[value.ordinal()].description;
  if (!description.valid())
    return std::unexpected(symbolic_description());
  return description;
}

Result<const TensorSpec &> Graph::value_spec(ValueId value) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this graph"));
  }
  return state_->values[value.ordinal()].spec;
}

GraphBuilder::GraphBuilder(GraphBuilder &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

GraphBuilder &GraphBuilder::operator=(GraphBuilder &&other) noexcept {
  if (this != &other) {
    detail::destroy(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

GraphBuilder::~GraphBuilder() { detail::destroy(state_); }

Result<GraphBuilder>
GraphBuilder::create(GraphBuilderOptions options) noexcept {
  mem::AllocatorRef allocator = options.metadata_allocator;
  if (!allocator.valid())
    allocator = mem::system_allocator();
  void *storage = allocator.allocate(sizeof(detail::BuilderState),
                                     alignof(detail::BuilderState));
  if (storage == nullptr)
    return std::unexpected(allocation_failure());
  auto *state =
      ::new (storage) detail::BuilderState(next_generation(), allocator);
  return GraphBuilder(state);
}

Result<EffectResourceId> GraphBuilder::effect_resource() noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (state_->effect_resource_count == UINT32_MAX)
    return std::unexpected(Status::ResourceExhausted(
        "graph effect resource identity space is exhausted"));
  return detail::GraphIdentity::effect_resource(
      state_->generation, state_->effect_resource_count++);
}

Status GraphBuilder::bind_resources(
    NodeId node, std::span<const EffectResourceId> resources) noexcept {
  if (state_ == nullptr)
    return invalid_builder();
  if (!detail::owns(*state_, node))
    return Status::InvalidArgument("node does not belong to this builder");
  auto &record = state_->nodes[node.ordinal()];
  const auto traits = record.traits();
  if (resources.size() != traits.resource_count)
    return Status::InvalidArgument(
        "graph resource binding count does not match operation schema");
  for (EffectResourceId resource : resources) {
    if (!detail::owns(*state_, resource))
      return Status::InvalidArgument(
          "effect resource belongs to another builder");
  }
  if (resources.empty())
    return Status::Ok();
  if (record.resource_begin == detail::kNoNode) {
    const std::size_t begin = state_->resource_edges.size();
    if (resources.size() > UINT32_MAX - begin)
      return Status::ResourceExhausted(
          "graph resource edge space is exhausted");
    if (!state_->resource_edges.try_resize(begin + resources.size()))
      return allocation_failure();
    record.resource_begin = static_cast<std::uint32_t>(begin);
  }
  for (std::size_t port = 0; port < resources.size(); ++port)
    state_->resource_edges[record.resource_begin + port] =
        resources[port].ordinal();
  return Status::Ok();
}

Result<DimensionId> GraphBuilder::dimension(DimensionBounds bounds) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  Status validation = validate_bounds(bounds);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  if (!fits_dimension_ordinal(state_->dimensions.size())) {
    return std::unexpected(Status::ResourceExhausted(
        "graph dimension identity space is exhausted"));
  }
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(state_->dimensions.size());
  if (!state_->dimensions.try_push_back(detail::DimensionRecord{
          .bounds = {bounds.minimum, bounds.maximum, bounds.multiple_of}})) {
    return std::unexpected(allocation_failure());
  }
  return detail::GraphIdentity::dimension(state_->generation, ordinal);
}

Result<DimensionId> GraphBuilder::dimension_sum(Dimension left,
                                                Dimension right) noexcept {
  return derived_dimension(state_, compute::detail::ShapeDimensionKind::kAdd,
                           left, right);
}
Result<DimensionId> GraphBuilder::dimension_product(Dimension left,
                                                    Dimension right) noexcept {
  return derived_dimension(
      state_, compute::detail::ShapeDimensionKind::kMultiply, left, right);
}
Result<DimensionId> GraphBuilder::dimension_quotient(Dimension left,
                                                     Dimension right) noexcept {
  return derived_dimension(
      state_, compute::detail::ShapeDimensionKind::kExactDivide, left, right);
}
Result<DimensionId>
GraphBuilder::dimension_broadcast(Dimension left, Dimension right) noexcept {
  return derived_dimension(
      state_, compute::detail::ShapeDimensionKind::kBroadcast, left, right);
}
Status GraphBuilder::require_dimension_equal(Dimension left,
                                             Dimension right) noexcept {
  return shape_guard(state_, compute::detail::ShapeGuardKind::kEqual, left,
                     right);
}
Status GraphBuilder::require_dimension_less_equal(Dimension left,
                                                  Dimension right) noexcept {
  return shape_guard(state_, compute::detail::ShapeGuardKind::kLessEqual, left,
                     right);
}

Result<ValueId> GraphBuilder::input(compute::TensorDesc description) noexcept {
  auto spec = TensorSpec::concrete(description);
  if (!spec)
    return std::unexpected(std::move(spec).error());
  return input(*spec);
}

Result<ValueId> GraphBuilder::input(TensorSpec spec) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  Status validation = validate_spec(*state_, spec);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  if (spec.kind() == TensorSpecKind::kInferred)
    return std::unexpected(Status::InvalidArgument(
        "input tensor spec requires an explicit storage layout"));
  compute::TensorDesc description{};
  auto resolved = spec.concrete_description();
  if (resolved)
    description = *resolved;
  else if (!resolved.error().IsFailedPrecondition())
    return std::unexpected(std::move(resolved).error());
  if (!fits_value_ordinal(state_->values.size())) {
    return std::unexpected(
        Status::ResourceExhausted("graph value identity space is exhausted"));
  }
  if (!state_->values.try_reserve(state_->values.size() + 1) ||
      !state_->inputs.try_reserve(state_->inputs.size() + 1)) {
    return std::unexpected(allocation_failure());
  }

  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(state_->values.size());
  const bool value_added = state_->values.try_push_back(detail::ValueRecord{
      .spec = spec,
      .description = description,
      .kind = ValueKind::kInput,
      .producer = detail::kNoNode,
  });
  const bool input_added = state_->inputs.try_push_back(ordinal);
  if (!value_added || !input_added) {
    return std::unexpected(allocation_failure());
  }
  return detail::value_id(*state_, ordinal);
}

Result<ValueId>
GraphBuilder::constant(compute::TensorDesc description) noexcept {
  auto spec = TensorSpec::concrete(description);
  if (!spec)
    return std::unexpected(std::move(spec).error());
  return constant(*spec);
}

Result<ValueId> GraphBuilder::constant(TensorSpec spec) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  Status validation = validate_spec(*state_, spec);
  if (!validation.ok())
    return std::unexpected(std::move(validation));
  if (spec.kind() == TensorSpecKind::kInferred)
    return std::unexpected(Status::InvalidArgument(
        "constant tensor spec requires an explicit storage layout"));
  compute::TensorDesc description{};
  auto resolved = spec.concrete_description();
  if (resolved)
    description = *resolved;
  else if (!resolved.error().IsFailedPrecondition())
    return std::unexpected(std::move(resolved).error());
  if (!fits_value_ordinal(state_->values.size())) {
    return std::unexpected(
        Status::ResourceExhausted("graph value identity space is exhausted"));
  }
  if (!state_->values.try_reserve(state_->values.size() + 1) ||
      !state_->constants.try_reserve(state_->constants.size() + 1)) {
    return std::unexpected(allocation_failure());
  }

  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(state_->values.size());
  const bool value_added = state_->values.try_push_back(detail::ValueRecord{
      .spec = spec,
      .description = description,
      .kind = ValueKind::kConstant,
      .producer = detail::kNoNode,
  });
  const bool constant_added = state_->constants.try_push_back(ordinal);
  if (!value_added || !constant_added)
    return std::unexpected(allocation_failure());
  return detail::value_id(*state_, ordinal);
}

Result<NodeId>
GraphBuilder::add_node(compute::Operation operation,
                       std::span<const ValueId> inputs) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  const compute::OperationTraits traits = compute::operation_traits(operation);
  if (!operation.valid() || !traits.inputs.accepts(inputs.size())) {
    return std::unexpected(
        Status::InvalidArgument("graph operation input count is invalid"));
  }

  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  using Specs = container::SmallVector<TensorSpec, 4, mem::AllocatorRef>;
  Descriptions descriptions(state_->allocator);
  Descriptions inferred_outputs(state_->allocator);
  Specs output_specs(state_->allocator);
  if (!descriptions.try_resize(inputs.size()) ||
      !inferred_outputs.try_resize(traits.outputs) ||
      !output_specs.try_resize(traits.outputs)) {
    return std::unexpected(allocation_failure());
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!detail::owns(*state_, inputs[index])) {
      return std::unexpected(Status::InvalidArgument(
          "operation input belongs to another builder"));
    }
    const compute::TensorDesc &description =
        state_->values[inputs[index].ordinal()].description;
    if (!description.valid()) {
      return std::unexpected(Status::FailedPrecondition(
          "symbolic operation requires explicit output tensor specs"));
    }
    descriptions[index] = description;
  }
  Status inferred = compute::infer_output_descriptions(operation, descriptions,
                                                       inferred_outputs);
  if (!inferred.ok())
    return std::unexpected(std::move(inferred));
  for (std::size_t index = 0; index < traits.outputs; ++index) {
    auto spec = TensorSpec::concrete(inferred_outputs[index]);
    if (!spec)
      return std::unexpected(std::move(spec).error());
    output_specs[index] = *spec;
  }
  return add_node(operation, inputs, output_specs);
}

Result<NodeId>
GraphBuilder::add_node(compute::Operation operation,
                       std::span<const ValueId> inputs,
                       std::span<const TensorSpec> outputs) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  Status effects = compute::validate_operation_effects(operation);
  if (!effects.ok())
    return std::unexpected(std::move(effects));
  const compute::OperationTraits traits = compute::operation_traits(operation);
  if (!operation.valid() || !traits.inputs.accepts(inputs.size())) {
    return std::unexpected(
        Status::InvalidArgument("graph operation input count is invalid"));
  }
  if (outputs.size() != traits.outputs) {
    return std::unexpected(Status::InvalidArgument(
        "graph operation output spec count is invalid"));
  }
  if (inputs.size() > (std::numeric_limits<std::uint16_t>::max)())
    return std::unexpected(Status::ResourceExhausted(
        "graph operation input identity space is exhausted"));
  const std::size_t output_count = outputs.size();
  if (!fits_node_ordinal(state_->nodes.size()) ||
      (output_count != 0 &&
       state_->values.size() >
           (std::numeric_limits<std::uint32_t>::max)() - (output_count - 1)) ||
      state_->edges.size() >
          (std::numeric_limits<std::uint32_t>::max)() - inputs.size()) {
    return std::unexpected(
        Status::ResourceExhausted("graph identity space is exhausted"));
  }

  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions descriptions(state_->allocator);
  Descriptions inferred_outputs(state_->allocator);
  if (!descriptions.try_resize(inputs.size()) ||
      !inferred_outputs.try_resize(output_count)) {
    return std::unexpected(allocation_failure());
  }
  bool all_concrete = true;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!detail::owns(*state_, inputs[index])) {
      return std::unexpected(Status::InvalidArgument(
          "operation input belongs to another builder"));
    }
    descriptions[index] = state_->values[inputs[index].ordinal()].description;
    all_concrete = all_concrete && descriptions[index].valid();
  }
  for (const TensorSpec &output : outputs) {
    Status validation = validate_spec(*state_, output);
    if (!validation.ok())
      return std::unexpected(std::move(validation));
    all_concrete = all_concrete && !output.is_symbolic();
  }
  if (all_concrete) {
    Status inferred = compute::infer_output_descriptions(
        operation, descriptions, inferred_outputs);
    if (!inferred.ok())
      return std::unexpected(std::move(inferred));
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      auto declared = outputs[index].concrete_description();
      if (!declared)
        return std::unexpected(std::move(declared).error());
      if (*declared != inferred_outputs[index]) {
        return std::unexpected(Status::InvalidArgument(
            "explicit graph output spec does not match operation inference"));
      }
    }
  }

  return append_node(*state_, std::move(operation), inputs, outputs);
}

Result<NodeId> GraphBuilder::add_node(ViewSpec view, ValueId input) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (!view.valid() || !detail::owns(*state_, input))
    return std::unexpected(
        Status::InvalidArgument("graph ViewSpec or input identity is invalid"));
  for (const auto dimension : view.dimensions()) {
    auto checked = shape_operand(*state_, dimension);
    if (!checked)
      return std::unexpected(std::move(checked).error());
  }
  if (view.kind() == compute::ViewKind::kSlice) {
    auto start = shape_operand(*state_, view.start());
    if (!start)
      return std::unexpected(std::move(start).error());
    auto count = shape_operand(*state_, view.count());
    if (!count)
      return std::unexpected(std::move(count).error());
  }
  const auto &source = state_->values[input.ordinal()];
  const std::array inputs{input};
  if (source.description.valid() && !view.is_symbolic()) {
    HYLE_ASSIGN_OR_RETURN(auto concrete, detail::bind_view(view, {}));
    return add_node(compute::Operation(concrete), inputs);
  }

  std::array<Dimension, compute::kMaxTensorRank> dimensions{};
  std::size_t rank = source.spec.rank();
  if (view.kind() == compute::ViewKind::kReshape ||
      view.kind() == compute::ViewKind::kBroadcast) {
    if (view.kind() == compute::ViewKind::kBroadcast &&
        view.dimensions().size() < rank)
      return std::unexpected(Status::InvalidArgument(
          "graph broadcast target rank is smaller than input rank"));
    rank = view.dimensions().size();
    std::copy(view.dimensions().begin(), view.dimensions().end(),
              dimensions.begin());
  } else if (view.kind() == compute::ViewKind::kSlice) {
    HYLE_ASSIGN_OR_RETURN(const auto axis, view_axis(view.axis(), rank));
    std::copy(source.spec.dimensions().begin(), source.spec.dimensions().end(),
              dimensions.begin());
    dimensions[axis] = view.count();
  } else {
    if (view.permutation().size() != rank)
      return std::unexpected(Status::InvalidArgument(
          "graph permutation rank differs from input rank"));
    std::array<bool, compute::kMaxTensorRank> seen{};
    for (std::size_t axis = 0; axis < rank; ++axis) {
      HYLE_ASSIGN_OR_RETURN(const auto from,
                            view_axis(view.permutation()[axis], rank));
      if (seen[from])
        return std::unexpected(
            Status::InvalidArgument("graph permutation repeats an axis"));
      seen[from] = true;
      dimensions[axis] = source.spec.dimensions()[from];
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto output,
                        TensorSpec::inferred(source.spec.data_type(),
                                             {dimensions.data(), rank},
                                             source.spec.storage_format()));
  const std::array outputs{output};
  return append_node(*state_, std::move(view), inputs, outputs);
}

Result<ValueId> GraphBuilder::add(ViewSpec view, ValueId input) noexcept {
  HYLE_ASSIGN_OR_RETURN(const auto node, add_node(std::move(view), input));
  return result(node);
}

Result<ValueId> GraphBuilder::result(NodeId node,
                                     std::size_t output_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this builder"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (output_index >= record.output_count) {
    return std::unexpected(
        Status::OutOfRange("graph node output index is out of range"));
  }
  return detail::value_id(
      *state_, record.output_begin + static_cast<std::uint32_t>(output_index));
}

Status GraphBuilder::depends_on(NodeId node, NodeId prerequisite) noexcept {
  if (state_ == nullptr)
    return invalid_builder();
  if (!detail::owns(*state_, node) || !detail::owns(*state_, prerequisite)) {
    return Status::InvalidArgument(
        "control dependency node belongs to another builder");
  }
  if (prerequisite.ordinal() >= node.ordinal()) {
    return Status::InvalidArgument(
        "control dependency must point to an earlier node");
  }
  if (state_->control_edges.size() >=
      (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::ResourceExhausted(
        "graph control dependency identity space is exhausted");
  }
  if (!state_->control_edges.try_push_back(detail::ControlEdgeRecord{
          .node = node.ordinal(), .prerequisite = prerequisite.ordinal()})) {
    return allocation_failure();
  }
  return Status::Ok();
}

Status GraphBuilder::publish(ValueId value) noexcept {
  if (state_ == nullptr)
    return invalid_builder();
  if (!detail::owns(*state_, value)) {
    return Status::InvalidArgument("graph output belongs to another builder");
  }
  for (std::uint32_t published : state_->outputs) {
    if (published == value.ordinal()) {
      return Status::AlreadyExists("graph output is already published");
    }
  }
  if (!state_->outputs.try_push_back(value.ordinal())) {
    return allocation_failure();
  }
  return Status::Ok();
}

Result<Graph> GraphBuilder::finish() && noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (state_->nodes.empty() && state_->outputs.empty()) {
    return std::unexpected(
        Status::FailedPrecondition("graph requires a node or published value"));
  }
  Status dependencies = canonicalize_dependencies(*state_);
  Status shape = compute::detail::validate_shape_program(
      {state_->dimensions, state_->shape_guards});
  if (!shape.ok())
    return std::unexpected(std::move(shape));
  if (!dependencies.ok())
    return std::unexpected(std::move(dependencies));
  Status effects = detail::validate_effect_order(*state_);
  if (!effects.ok())
    return std::unexpected(std::move(effects));

  const mem::AllocatorRef allocator = state_->allocator;
  void *storage = allocator.allocate(sizeof(detail::GraphState),
                                     alignof(detail::GraphState));
  if (storage == nullptr)
    return std::unexpected(allocation_failure());
  auto *published = ::new (storage) detail::GraphState(*state_);
  detail::destroy(state_);
  state_ = nullptr;
  return detail::GraphAccess::adopt(published);
}

} // namespace hyle::graph
