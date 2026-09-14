#include "hyle/graph/executable.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <string_view>
#include <utility>

#include "../compute/completion_internal.h"
#include "../compute/provider_internal.h"
#include "graph_state.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/transfer.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/container/hash.h"
#include "plan_state.h"

namespace hyle::graph {
namespace {

Status plan_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph plan metadata allocation failed");
}

Status invocation_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph invocation allocation failed");
}

void merge_aggregate_terminal(detail::AggregateCompletionState &aggregate,
                              const Status &terminal) noexcept {
  std::lock_guard lock(aggregate.lock);
  aggregate.terminal.Update(terminal);
}

void release_aggregate_episode(
    detail::AggregateCompletionState *aggregate) noexcept {
  if (aggregate->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  compute::detail::CompletionLifetime outer;
  Status terminal;
  {
    std::lock_guard lock(aggregate->lock);
    outer = aggregate->outer_lifetime;
    aggregate->outer_lifetime = {};
    terminal = aggregate->terminal;
  }
  if (outer.valid()) {
    outer.terminal(outer.state, terminal);
    // release() may destroy the InvocationState containing aggregate. Do not
    // touch aggregate after this call.
    outer.release(outer.state);
  }
}

void aggregate_child_terminal(void *opaque, const Status &terminal) noexcept {
  auto *child = static_cast<detail::AggregateChildState *>(opaque);
  merge_aggregate_terminal(*child->aggregate, terminal);
}

void aggregate_child_release(void *opaque) noexcept {
  auto *child = static_cast<detail::AggregateChildState *>(opaque);
  release_aggregate_episode(child->aggregate);
}

compute::Completion &
aggregate_terminal_completion(detail::AggregateCompletionState &aggregate,
                              std::size_t index) noexcept {
  return aggregate.completions[aggregate.completion_segments[index]];
}

Result<compute::CompletionPoll> poll_aggregate(void *opaque) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  Status terminal = Status::Ok();
  bool pending = false;
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid()) {
      terminal.Update(Status::Internal("graph terminal completion is invalid"));
      continue;
    }
    auto result = completion.poll();
    if (!result)
      terminal.Update(std::move(result).error());
    else
      pending = pending || *result == compute::CompletionPoll::kPending;
  }
  merge_aggregate_terminal(aggregate, terminal);
  if (pending)
    return compute::CompletionPoll::kPending;
  {
    std::lock_guard lock(aggregate.lock);
    terminal = aggregate.terminal;
  }
  return terminal.ok()
             ? Result<compute::CompletionPoll>(
                   compute::CompletionPoll::kComplete)
             : Result<compute::CompletionPoll>(std::unexpected(terminal));
}

Status wait_aggregate(void *opaque) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  Status terminal = Status::Ok();
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid()) {
      terminal.Update(Status::Internal("graph terminal completion is invalid"));
      continue;
    }
    terminal.Update(completion.wait());
  }
  merge_aggregate_terminal(aggregate, terminal);
  std::lock_guard lock(aggregate.lock);
  return aggregate.terminal;
}

Status cancel_aggregate(void *opaque) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  Status terminal = Status::Ok();
  bool unsupported = false;
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid())
      continue;
    Status result = completion.request_cancel();
    if (result.IsUnimplemented())
      unsupported = true;
    else
      terminal.Update(std::move(result));
  }
  if (!terminal.ok())
    return terminal;
  return unsupported
             ? Status::Unimplemented(
                   "one or more graph terminal segments cannot be cancelled")
             : Status::Ok();
}

void destroy_aggregate(void *opaque,
                       compute::detail::CompletionLifetime lifetime,
                       Status terminal_hint) noexcept {
  auto &aggregate = *static_cast<detail::AggregateCompletionState *>(opaque);
  if (!lifetime.valid()) {
    for (std::size_t index = 0; index < aggregate.completion_count; ++index)
      aggregate_terminal_completion(aggregate, index) = {};
    return;
  }

  {
    std::lock_guard lock(aggregate.lock);
    aggregate.terminal = std::move(terminal_hint);
    aggregate.outer_lifetime = lifetime;
  }
  // The sentinel prevents an inline child retirement from releasing the
  // aggregate owner while the remaining child attachments are being built.
  aggregate.remaining.store(1, std::memory_order_release);
  for (std::size_t index = 0; index < aggregate.completion_count; ++index) {
    compute::Completion &completion =
        aggregate_terminal_completion(aggregate, index);
    if (!completion.valid()) {
      merge_aggregate_terminal(
          aggregate, Status::Internal("graph terminal completion is invalid"));
      continue;
    }
    auto polled = completion.poll();
    if (!polled) {
      merge_aggregate_terminal(aggregate, polled.error());
      completion = {};
      continue;
    }
    if (*polled == compute::CompletionPoll::kComplete) {
      completion = {};
      continue;
    }

    detail::AggregateChildState &child = aggregate.children[index];
    child.aggregate = &aggregate;
    child.lifetime = compute::detail::CompletionLifetime{
        .state = &child,
        .terminal = &aggregate_child_terminal,
        .release = &aggregate_child_release,
    };
    aggregate.remaining.fetch_add(1, std::memory_order_relaxed);
    // attach_lifetime consumes the counter episode on every return path.
    compute::detail::CompletionAccess::attach_lifetime(completion,
                                                       child.lifetime)
        .IgnoreError();
    completion = {};
  }
  release_aggregate_episode(&aggregate);
}

const compute::detail::CompletionVTable kAggregateCompletionVTable{
    .destroy = &destroy_aggregate,
    .poll = &poll_aggregate,
    .wait = &wait_aggregate,
    .request_cancel = &cancel_aggregate,
};

compute::Completion make_aggregate_completion(
    detail::AggregateCompletionState &aggregate) noexcept {
  aggregate.remaining.store(0, std::memory_order_relaxed);
  {
    std::lock_guard lock(aggregate.lock);
    aggregate.terminal = Status::Ok();
    aggregate.outer_lifetime = {};
  }
  return compute::detail::CompletionAccess::make(&kAggregateCompletionVTable,
                                                 &aggregate);
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

Status validate_specialized_operations(
    const detail::GraphState &graph,
    std::span<const compute::TensorDesc> descriptions,
    mem::AllocatorRef allocator) noexcept {
  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions inputs(allocator);
  Descriptions inferred(allocator);
  for (const detail::NodeRecord &node : graph.nodes) {
    if (!inputs.try_resize(node.input_count) ||
        !inferred.try_resize(node.output_count)) {
      return plan_allocation_failure();
    }
    for (std::size_t index = 0; index < node.input_count; ++index) {
      inputs[index] = descriptions[graph.edges[node.input_begin + index]];
    }
    Status status =
        compute::infer_output_descriptions(node.operation, inputs, inferred);
    if (!status.ok())
      return status.WithContext("graph specialization operation inference");
    for (std::size_t index = 0; index < node.output_count; ++index) {
      if (inferred[index] != descriptions[node.output_begin + index]) {
        return Status::InvalidArgument(
            "graph specialization output spec does not match operation "
            "inference");
      }
    }
  }
  return Status::Ok();
}

Result<SpecializationSignature>
specialize_descriptions(const detail::GraphState &graph,
                        std::span<const ShapeBinding> bindings,
                        mem::AllocatorRef allocator,
                        detail::TensorDescriptions &descriptions) noexcept {
  using Values = container::SmallVector<std::int64_t, 4, mem::AllocatorRef>;
  using Flags = container::SmallVector<std::uint8_t, 4, mem::AllocatorRef>;
  Values values(allocator);
  Flags seen(allocator);
  if (!values.try_resize(graph.dimensions.size()) ||
      !seen.try_resize(graph.dimensions.size()) ||
      !descriptions.try_resize(graph.values.size())) {
    return std::unexpected(plan_allocation_failure());
  }
  const compute::detail::ShapeProgramView shape{graph.dimensions,
                                                graph.shape_guards};
  if (bindings.size() != shape.input_count()) {
    return std::unexpected(
        Status::InvalidArgument("compile requires exactly one binding for "
                                "every independent graph dimension"));
  }
  for (std::size_t index = 0; index < seen.size(); ++index)
    seen[index] = 0;
  for (const ShapeBinding binding : bindings) {
    if (!detail::owns(graph, binding.dimension)) {
      return std::unexpected(Status::InvalidArgument(
          "shape binding does not belong to this graph"));
    }
    const std::size_t ordinal = binding.dimension.ordinal();
    if (graph.dimensions[ordinal].kind !=
        compute::detail::ShapeDimensionKind::kInput) {
      return std::unexpected(Status::InvalidArgument(
          "derived graph dimensions cannot be bound by the caller"));
    }
    if (seen[ordinal] != 0) {
      return std::unexpected(
          Status::InvalidArgument("shape binding is duplicated"));
    }
    values[ordinal] = binding.value;
    seen[ordinal] = 1;
  }
  // Equal counts, valid input IDs, and no duplicates prove every input slot is
  // filled. Evaluate derived values and guards before consulting any Provider.
  Status evaluated = compute::detail::evaluate_shape_program(shape, values);
  if (!evaluated.ok())
    return std::unexpected(std::move(evaluated));

  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    auto description = resolve_spec(graph, graph.values[index].spec, values);
    if (!description)
      return std::unexpected(std::move(description).error());
    descriptions[index] = *description;
  }
  Status semantic = validate_specialized_operations(
      graph,
      std::span<const compute::TensorDesc>(descriptions.data(),
                                           descriptions.size()),
      allocator);
  if (!semantic.ok())
    return std::unexpected(std::move(semantic));

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

Result<Graph>
make_specialized_graph(const detail::GraphState &source,
                       std::span<const compute::TensorDesc> descriptions,
                       mem::AllocatorRef allocator) noexcept {
  detail::BuilderState builder(source.generation, allocator);
  if (!builder.dimensions.try_resize(source.dimensions.size()) ||
      !builder.shape_guards.try_resize(source.shape_guards.size()) ||
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
  for (std::size_t index = 0; index < source.dimensions.size(); ++index)
    builder.dimensions[index] = source.dimensions[index];
  for (std::size_t index = 0; index < source.shape_guards.size(); ++index)
    builder.shape_guards[index] = source.shape_guards[index];
  for (std::size_t index = 0; index < source.values.size(); ++index) {
    auto spec = TensorSpec::concrete(descriptions[index]);
    if (!spec)
      return std::unexpected(std::move(spec).error());
    builder.values[index] = source.values[index];
    builder.values[index].spec = *spec;
    builder.values[index].description = descriptions[index];
  }
  for (std::size_t index = 0; index < source.nodes.size(); ++index)
    builder.nodes[index] = source.nodes[index];
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

void mark_invocation_terminal(void *opaque, const Status &terminal) noexcept {
  auto *state = static_cast<detail::InvocationState *>(opaque);
  state->outputs_ready.store(terminal.ok(), std::memory_order_release);
  if (!state->plan->supports_concurrent_invocations) {
    state->plan->exclusive_in_flight.store(false, std::memory_order_release);
  }
  state->in_flight.store(false, std::memory_order_release);
}

void release_invocation_lifetime(void *opaque) noexcept {
  detail::release(static_cast<detail::InvocationState *>(opaque));
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

Status bind_effect_resources(detail::PlanState &plan,
                             const detail::GraphState &graph,
                             CompileOptions options) noexcept {
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> used(
      plan.allocator);
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> bound(
      plan.allocator);
  if (!used.try_resize(graph.effect_resource_count) ||
      !bound.try_resize(graph.effect_resource_count))
    return plan_allocation_failure();
  std::fill(used.begin(), used.end(), std::uint8_t{0});
  std::fill(bound.begin(), bound.end(), std::uint8_t{0});
  std::size_t used_count = 0;
  for (const auto &node : graph.nodes) {
    const auto count = compute::operation_traits(node.operation).resource_count;
    for (std::size_t port = 0; port < count; ++port) {
      const auto resource = graph.resource_edges[node.resource_begin + port];
      if (!used[resource]) {
        used[resource] = 1;
        ++used_count;
      }
    }
  }
  if (!plan.effect_bindings.try_reserve(used_count))
    return plan_allocation_failure();
  for (const auto &binding : options.effect_bindings) {
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
    for (const auto &previous : plan.effect_bindings) {
      if (overlaps(previous.footprint, binding.footprint))
        return Status::InvalidArgument("different effect resource IDs alias; "
                                       "use one ID or disjoint ranges");
    }
    if (!plan.effect_bindings.try_push_back(binding))
      return plan_allocation_failure();
  }
  for (std::size_t resource = 0; resource < used.size(); ++resource) {
    if (used[resource] && !bound[resource])
      return Status::InvalidArgument(
          "graph effect resource has no backing footprint");
  }
  for (const auto &binding : options.constants)
    HYLE_RETURN_IF_ERROR(validate_effect_tensor_alias(
        plan, byte_range(binding.tensor.storage())));
  return Status::Ok();
}

bool matches_alias_range(ByteRange source, ByteRange output,
                         std::size_t byte_offset) noexcept {
  return same_allocation(source, output) && output.offset >= source.offset &&
         output.offset - source.offset == byte_offset &&
         byte_offset <= source.size && output.size <= source.size - byte_offset;
}

bool matches_planned_alias(const detail::PlanState &plan,
                           std::size_t left_value, ByteRange left,
                           std::size_t right_value, ByteRange right) noexcept {
  const std::uint32_t left_ordinal = plan.external_instances[left_value];
  const std::uint32_t right_ordinal = plan.external_instances[right_value];
  if (left_ordinal == detail::kNoWorkspaceRequest ||
      right_ordinal == detail::kNoWorkspaceRequest) {
    return false;
  }
  if (plan.alias_sources[left_ordinal] == right_ordinal) {
    return matches_alias_range(right, left, plan.alias_offsets[left_ordinal]);
  }
  if (plan.alias_sources[right_ordinal] == left_ordinal) {
    return matches_alias_range(left, right, plan.alias_offsets[right_ordinal]);
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

Status validate_external_alias(const detail::InvocationState &invocation,
                               std::size_t ordinal,
                               ByteRange candidate) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  HYLE_RETURN_IF_ERROR(validate_effect_tensor_alias(plan, candidate));
  const bool candidate_is_output =
      (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0;

  auto check = [&](ValueId other_value) noexcept -> Status {
    const std::size_t other = other_value.ordinal();
    const std::uint32_t other_instance = plan.external_instances[other];
    if (other == ordinal || other_instance == detail::kNoWorkspaceRequest ||
        invocation.bound[other_instance] == 0) {
      return Status::Ok();
    }
    const bool other_is_output =
        (plan.binding_roles[other] & detail::kExternalOutput) != 0;
    if (!candidate_is_output && !other_is_output)
      return Status::Ok();
    const ByteRange existing = bound_range(invocation, other);
    if (!overlaps(candidate, existing))
      return Status::Ok();
    if (matches_planned_alias(plan, ordinal, candidate, other, existing)) {
      return Status::Ok();
    }
    return Status::InvalidArgument(
        "external graph tensor bindings overlap unsafely");
  };

  if (candidate_is_output) {
    for (ValueId input : plan.input_ids) {
      Status status = check(input);
      if (!status.ok())
        return status;
    }
    for (ValueId constant : plan.constant_ids) {
      Status status = check(constant);
      if (!status.ok())
        return status;
    }
  }
  for (ValueId output : plan.output_ids) {
    Status status = check(output);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

void invalidate_binding_cache(detail::InvocationState &invocation) noexcept {
  invocation.bindings_prepared = false;
  invocation.transfers_prepared = false;
  invocation.dependencies_prepared = false;
  invocation.outputs_ready.store(false, std::memory_order_release);
  if (invocation.replay_route == ReplayRoute::kCaptured ||
      invocation.replay_route == ReplayRoute::kHybrid) {
    for (compute::detail::CapturedCommands &captured :
         invocation.captured_segments) {
      captured.reset();
    }
  }
  invocation.replay_prepared = false;
  invocation.replay_route = ReplayRoute::kUnprepared;
  invocation.replay_report = {};
}

Status
prepare_invocation_bindings(detail::InvocationState &invocation) noexcept {
  if (invocation.bindings_prepared)
    return Status::Ok();
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

  // Resolve aliases and construct stable spans before any Queue accepts work.
  // Cache them until a binding changes so steady replay does not rebuild all
  // per-step views or rescan external values.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (!step.emits_command()) {
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
        auto bytes = plan.instance_descriptions[output].required_bytes();
        if (!bytes)
          return std::move(bytes).error();
        auto storage = invocation.readable[source].storage().subview(
            plan.alias_offsets[output], *bytes);
        if (!storage)
          return std::move(storage).error();
        auto derived = compute::ConstTensorView::create(
            plan.instance_descriptions[output], *storage);
        if (!derived)
          return std::move(derived).error();
        const std::uint32_t logical_output = plan.instance_values[output];
        if ((plan.binding_roles[logical_output] &
             detail::kSourceAliasedOutput) != 0) {
          invocation.readable[output] = *derived;
        } else if ((plan.instance_roles[output] & detail::kExternalOutput) !=
                   0) {
          if (!equal_range(byte_range(derived->storage()),
                           byte_range(invocation.writable[output].storage()))) {
            return Status::InvalidArgument(
                "published graph view does not bind its inferred storage");
          }
        } else {
          invocation.readable[output] = *derived;
        }
      }
      continue;
    }
    for (std::size_t index = 0; index < step.input_count; ++index) {
      const compute::ConstTensorView binding =
          invocation.readable[plan.node_inputs[step.input_begin + index]];
      if (!binding.valid()) {
        return Status::Internal(
            "materializing graph step has no readable input storage");
      }
      invocation.submission_inputs[step.input_begin + index] = binding;
    }
    for (std::size_t index = 0; index < step.output_count; ++index) {
      const compute::TensorView binding =
          invocation.writable[plan.node_outputs[step.output_begin + index]];
      if (!binding.valid()) {
        return Status::Internal(
            "materializing graph step has no writable output storage");
      }
      invocation.submission_outputs[step.output_begin + index] = binding;
    }
  }
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

bool shared_lane(const detail::PlanState &plan, ExecutionLaneId lane) noexcept {
  return lane.valid() && lane.ordinal() < plan.lanes.size() &&
         plan.lanes[lane.ordinal()].runtime_mode ==
             compute::ProviderRuntimeMode::kSharedQueue;
}

Result<std::size_t>
delegated_step_index(const detail::PlanState &plan,
                     const detail::PlanSegment &segment) noexcept {
  if (!segment.lane.valid() || segment.lane.ordinal() >= plan.lanes.size() ||
      plan.lanes[segment.lane.ordinal()].runtime_mode !=
          compute::ProviderRuntimeMode::kDelegated) {
    return std::unexpected(
        Status::Internal("graph delegated segment has an invalid lane"));
  }
  std::size_t selected = SIZE_MAX;
  const std::size_t end = segment.step_begin + segment.step_count;
  for (std::size_t step_index = segment.step_begin; step_index < end;
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (!step.emits_command())
      continue;
    if (step.lane != segment.lane ||
        step.runtime_properties().mode !=
            compute::ProviderRuntimeMode::kDelegated ||
        selected != SIZE_MAX) {
      return std::unexpected(Status::Internal(
          "graph delegated segment does not own exactly one command"));
    }
    selected = step_index;
  }
  if (selected == SIZE_MAX) {
    return std::unexpected(
        Status::Internal("graph delegated segment has no command"));
  }
  return selected;
}

bool delegated_can_import_endpoint(
    detail::InvocationState &invocation, const detail::PlanSegment &segment,
    compute::detail::CompletionEndpoint source) noexcept {
  auto selected = delegated_step_index(*invocation.plan, segment);
  if (!selected)
    return false;
  const detail::PlanStep &step = invocation.plan->steps[*selected];
  if (step.kind == detail::PlanStepKind::kRegion) {
    return invocation.invocations[*selected].region.can_import_endpoint(source);
  }
  if (step.kind == detail::PlanStepKind::kOperation) {
    return compute::detail::OperationInvocationAccess::can_import_endpoint(
        invocation.invocations[*selected].operation, source);
  }
  return false;
}

Result<compute::Completion>
submit_delegated_segment(detail::InvocationState &invocation,
                         const detail::PlanSegment &segment,
                         std::span<const compute::detail::CompletionDependency>
                             dependencies) noexcept {
  auto selected = delegated_step_index(*invocation.plan, segment);
  if (!selected)
    return std::unexpected(std::move(selected).error());
  const detail::PlanStep &step = invocation.plan->steps[*selected];
  const auto all_inputs = std::span<const compute::ConstTensorView>(
      invocation.submission_inputs.data(), invocation.submission_inputs.size());
  const auto all_outputs = std::span<const compute::TensorView>(
      invocation.submission_outputs.data(),
      invocation.submission_outputs.size());
  const auto inputs = all_inputs.subspan(step.input_begin, step.input_count);
  const auto outputs =
      all_outputs.subspan(step.output_begin, step.output_count);
  if (step.kind == detail::PlanStepKind::kRegion) {
    return invocation.invocations[*selected].region.submit_delegated(
        dependencies, inputs, outputs);
  }
  if (step.kind == detail::PlanStepKind::kOperation) {
    return compute::detail::OperationInvocationAccess::
        submit_delegated_borrowed_prevalidated(
            invocation.invocations[*selected].operation, dependencies, inputs,
            outputs);
  }
  return std::unexpected(
      Status::Internal("graph delegated step kind is invalid"));
}

Status validate_transfer_routes(detail::InvocationState &invocation,
                                std::span<const QueueBinding> queues,
                                ReplayReport &report) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  for (const detail::PlanStep &step : plan.steps) {
    if (step.kind != detail::PlanStepKind::kCopy)
      continue;
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
  }
  return Status::Ok();
}

Result<ReplayReport>
prepare_dependency_routes(detail::InvocationState &invocation,
                          std::span<const QueueBinding> queues,
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
    compute::Queue *destination =
        destination_shared ? candidate_queue(queues, segment.lane) : nullptr;
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;
    for (std::size_t dependency_index = segment.dependency_begin;
         dependency_index < dependency_end; ++dependency_index) {
      const std::uint32_t source_segment =
          plan.segment_dependencies[dependency_index];
      if (source_segment >= segment_index ||
          (destination_shared && destination == nullptr)) {
        return std::unexpected(
            Status::Internal("compiled graph dependency is invalid"));
      }
      const detail::PlanSegment &source_record = plan.segments[source_segment];
      const bool source_shared = shared_lane(plan, source_record.lane);
      compute::Queue *source =
          source_shared ? candidate_queue(queues, source_record.lane) : nullptr;
      if (source_shared && source == nullptr) {
        return std::unexpected(
            Status::Internal("compiled graph dependency source is invalid"));
      }

      ReplayDependencyRoute route = ReplayDependencyRoute::kOrderedQueue;
      const compute::detail::CompletionEndpoint source_endpoint =
          invocation.candidate_lane_endpoints[source_record.lane.ordinal()];
      const compute::detail::CompletionEndpoint destination_endpoint =
          invocation.candidate_lane_endpoints[segment.lane.ordinal()];
      if ((source_shared && destination_shared && source == destination) ||
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
              *destination, source_endpoint);
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

Status
enqueue_segment_steps(detail::InvocationState &invocation,
                      const detail::PlanSegment &segment,
                      compute::detail::QueueSequence &sequence) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  const auto all_inputs = std::span<const compute::ConstTensorView>(
      invocation.submission_inputs.data(), invocation.submission_inputs.size());
  const auto all_outputs = std::span<const compute::TensorView>(
      invocation.submission_outputs.data(),
      invocation.submission_outputs.size());
  const std::size_t end = segment.step_begin + segment.step_count;
  for (std::size_t step_index = segment.step_begin; step_index < end;
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (!step.emits_command())
      continue;
    const auto step_inputs =
        all_inputs.subspan(step.input_begin, step.input_count);
    const auto step_outputs =
        all_outputs.subspan(step.output_begin, step.output_count);
    compute::detail::EnqueueResult enqueued;
    if (step.kind == detail::PlanStepKind::kRegion) {
      enqueued = invocation.invocations[step_index].region.enqueue(
          sequence, step_inputs, step_outputs);
    } else if (step.kind == detail::PlanStepKind::kOperation) {
      enqueued =
          compute::detail::OperationInvocationAccess::enqueue_prevalidated(
              invocation.invocations[step_index].operation, sequence,
              step_inputs, step_outputs);
    } else if (step.kind == detail::PlanStepKind::kCopy) {
      enqueued = compute::detail::TransferAccess::enqueue_prevalidated_copy(
          sequence, step_outputs[0].storage(), step_inputs[0].storage());
    } else {
      enqueued = compute::detail::EnqueueResult::rejected(
          Status::Internal("compiled graph step kind is invalid"));
    }
    const bool keep_recording = enqueued.status.ok();
    Status status = enqueued.status;
    sequence.record(std::move(enqueued));
    if (!keep_recording)
      return status;
  }
  return Status::Ok();
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

Status bind_constants(detail::PlanState &plan, const detail::GraphState &graph,
                      CompileOptions options) noexcept {
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
  for (std::size_t index = 0; index < slots.size(); ++index)
    slots[index] = detail::kNoWorkspaceRequest;
  for (std::size_t index = 0; index < graph.constants.size(); ++index) {
    slots[graph.constants[index]] = static_cast<std::uint32_t>(index);
    seen[index] = 0;
  }

  for (const ConstantBinding &binding : options.constants) {
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
    seen[slot] = 1;
  }
  for (std::uint8_t present : seen) {
    if (present == 0)
      return Status::InvalidArgument("graph constant binding is missing");
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

  Result<bool> allows_family(const compute::Context &context,
                             std::size_t index) const noexcept {
    if (allowed_.empty() && denied_.empty())
      return true;
    auto identity = context.provider_identity(index);
    if (!identity)
      return std::unexpected(std::move(identity).error());
    return allows_family(identity->id);
  }

  Result<bool> allows_operation(const compute::Context &context,
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
  compute::detail::ProviderExtensionRef extension{};
  std::uint32_t provider_index = 0;
  ExecutionTargetId target{};
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

// One discovery per compile, shared by the bounded construction attempts.
// Proposal ordinals remain stable while the eligible partition changes.
struct RegionDiscovery final {
  explicit RegionDiscovery(mem::AllocatorRef allocator) noexcept
      : extensions(allocator), candidates(allocator), nodes(allocator),
        rejected(allocator) {}

  RegionExtensions extensions;
  RegionCandidates candidates;
  RegionNodeIds nodes;
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> rejected;
  bool ready = false;
};

struct PlacementAttemptOutcome final {
  std::optional<std::uint32_t> rejected_region{};
  bool no_feasible_route = false;
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
  if (!selected.try_resize(graph.nodes.size()) ||
      !reachable_outside.try_resize(graph.nodes.size())) {
    return std::unexpected(plan_allocation_failure());
  }
  std::fill(selected.begin(), selected.end(), std::uint8_t{0});
  std::fill(reachable_outside.begin(), reachable_outside.end(),
            std::uint8_t{0});
  std::uint32_t first = detail::kNoNode;
  std::uint32_t last = 0;
  for (NodeId node : nodes) {
    if (!detail::owns(graph, node)) {
      return std::unexpected(Status::InvalidArgument(
          "provider proposed a graph region node from another graph"));
    }
    if (selected[node.ordinal()] != 0)
      return std::unexpected(
          Status::InvalidArgument("graph region node set has duplicates"));
    selected[node.ordinal()] = 1;
    first = (std::min)(first, node.ordinal());
    last = (std::max)(last, node.ordinal());
  }
  // In a canonical DAG every path has increasing ordinals, so an interval
  // (including a singleton) cannot leave the set and then re-enter it.
  if (static_cast<std::size_t>(last) - first + 1 == nodes.size())
    return true;

  auto predecessor_state = [&](std::uint32_t predecessor,
                               bool current_selected) noexcept {
    if (predecessor == detail::kNoNode)
      return false;
    if (current_selected)
      return selected[predecessor] == 0 && reachable_outside[predecessor] != 0;
    return selected[predecessor] != 0 || reachable_outside[predecessor] != 0;
  };
  for (std::uint32_t node_index = first; node_index <= last; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    const bool current_selected = selected[node_index] != 0;
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
      reachable_outside[node_index] = reached ? 1 : 0;
  }
  return true;
}

Status collect_region_boundary(
    const detail::PlanState &plan, const detail::GraphState &graph,
    std::span<const NodeId> nodes,
    container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>
        &node_membership,
    container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> &outside_uses,
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> &marks,
    std::uint32_t marker, RegionValues &inputs,
    RegionValues &outputs) noexcept {
  if (!node_membership.try_resize(graph.nodes.size()) ||
      !outside_uses.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }
  std::fill(node_membership.begin(), node_membership.end(), std::uint8_t{0});
  std::fill(outside_uses.begin(), outside_uses.end(), std::uint8_t{0});
  for (NodeId node : nodes)
    node_membership[node.ordinal()] = 1;
  for (std::uint32_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    if (node_membership[node_index] != 0)
      continue;
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      outside_uses[graph.edges[node.input_begin + input_index]] = 1;
    }
  }
  for (NodeId node_id : nodes) {
    const std::uint32_t node_index = node_id.ordinal();
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t ordinal = graph.edges[node.input_begin + input_index];
      const std::uint32_t producer = graph.values[ordinal].producer;
      if (producer != detail::kNoNode && node_membership[producer] != 0)
        continue;
      if (marks[ordinal] == marker)
        continue;
      marks[ordinal] = marker;
      if (!inputs.try_push_back(detail::value_id(graph, ordinal)))
        return plan_allocation_failure();
    }
  }
  for (NodeId node_id : nodes) {
    const std::uint32_t node_index = node_id.ordinal();
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t ordinal =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      const bool published =
          (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0;
      if ((published || outside_uses[ordinal] != 0) &&
          !outputs.try_push_back(detail::value_id(graph, ordinal))) {
        return plan_allocation_failure();
      }
    }
  }
  return Status::Ok();
}

Status propose_region_resources(
    const RegionExtension &extension, const RegionCompileRequest &request,
    std::span<compute::detail::ResourceBindingRequirement> input_requirements,
    std::span<compute::detail::ResourceBindingRequirement> output_requirements,
    RegionResourceProposal &resource_proposal) noexcept {
  for (auto &requirement : input_requirements)
    requirement = {};
  for (auto &requirement : output_requirements)
    requirement = {};
  resource_proposal = {};
  struct ProposalEpisode final {
    const RegionCompileRequest *request = nullptr;
    std::span<compute::detail::ResourceBindingRequirement> input_requirements{};
    std::span<compute::detail::ResourceBindingRequirement>
        output_requirements{};
    RegionResourceProposal *proposal = nullptr;
  } proposal_episode{&request, input_requirements, output_requirements,
                     &resource_proposal};
  Status proposed = compute::detail::ProviderExtensionAccess::call(
      extension.extension,
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
            provider_context, *call.request, call.input_requirements,
            call.output_requirements, call.proposal);
      },
      &proposal_episode);
  if (!proposed.ok())
    return proposed;
  if (!resource_proposal.runtime.valid()) {
    return Status::Internal(
        "graph region provider proposed invalid runtime properties");
  }
  return Status::Ok();
}

struct RegionPrepareAttempt final {
  detail::PreparedRegion prepared{};
  compute::detail::ProviderPrepareDecline decline =
      compute::detail::ProviderPrepareDecline::kNone;
  Status reason = Status::Ok();
};

Result<RegionPrepareAttempt>
compile_region_artifact(detail::PlanState &plan,
                        const RegionExtension &extension,
                        const RegionCompileRequest &request,
                        const RegionResourceProposal &resource_proposal,
                        CompileOptions options) noexcept {
  ProviderRegionPrepared provider_prepared{};
  struct Episode final {
    const RegionCompileRequest *request = nullptr;
    mem::AllocatorRef allocator{};
    ProviderRegionPrepared *prepared = nullptr;
    compute::ProviderPrepareOutcome outcome{};
  } episode{&request, plan.allocator, &provider_prepared};
  const Status compiled = compute::detail::ProviderExtensionAccess::call(
      extension.extension,
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
        .reason = std::move(outcome.status)};
  }
  const compute::WorkspaceRequirement workspace =
      provider_prepared.properties.workspace;
  const compute::ProviderRuntimeProperties runtime =
      provider_prepared.properties.runtime;
  const bool shared =
      runtime.mode == compute::ProviderRuntimeMode::kSharedQueue;
  if (table == nullptr || table->destroy_prepared == nullptr ||
      table->alias_support == nullptr || table->create_invocation == nullptr ||
      table->destroy_invocation == nullptr ||
      (shared &&
       (table->shared_submit == nullptr || table->delegated_submit != nullptr ||
        table->endpoint != nullptr || table->can_import_endpoint != nullptr)) ||
      (!shared &&
       (table->shared_submit != nullptr || table->delegated_submit == nullptr ||
        table->endpoint == nullptr || table->can_import_endpoint == nullptr)) ||
      !runtime.valid() || runtime != resource_proposal.runtime ||
      workspace.alignment == 0 ||
      (workspace.alignment & (workspace.alignment - 1)) != 0 ||
      (runtime.workspace_owner == compute::ProviderWorkspaceOwner::kCaller &&
       workspace.size != 0 && !request.workspace_resource.valid()) ||
      (runtime.workspace_owner == compute::ProviderWorkspaceOwner::kProvider &&
       request.workspace_resource.valid()) ||
      provider_prepared.properties.dispatch_count == 0 ||
      provider_prepared.properties.preserves_effects !=
          resource_proposal.preserves_effects ||
      provider_prepared.properties.packed_constant_count >
          options.constants.size()) {
    if (table != nullptr && table->destroy_prepared != nullptr)
      table->destroy_prepared(provider_prepared.state, plan.allocator);
    return std::unexpected(Status::Internal(
        "graph region provider returned an invalid compiled artifact"));
  }
  return RegionPrepareAttempt{
      .prepared = detail::PreparedRegion(extension.extension, provider_prepared,
                                         plan.allocator)};
}

Status discover_region_candidates(
    const Graph &published_graph, const detail::GraphState &graph,
    const compute::Context &context, CompileOptions options,
    const ProviderPolicy &provider_policy, mem::AllocatorRef allocator,
    RegionExtensions &extensions, RegionCandidates &candidates,
    RegionNodeIds &proposal_nodes, std::size_t &proposal_count,
    std::size_t &proposal_node_count, CompileSearchReport &search,
    ExecutionTargetId target,
    std::span<const std::uint32_t> node_targets = {}) noexcept;

Status select_regions(RegionCandidates &candidates, mem::AllocatorRef allocator,
                      std::span<const NodeId> proposal_nodes,
                      const detail::GraphState &graph,
                      RegionCandidates &selected) noexcept;

Status prepare_placed_steps_attempt(
    detail::PlanState &plan, const Graph &published_graph,
    const detail::GraphState &graph, CompileEnvironment environment,
    CompileOptions options, const ProviderPolicy &provider_policy,
    RegionDiscovery &discovery, PlacementAttemptOutcome &outcome) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Flags = container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;
  using Prepared =
      container::SmallVector<compute::PreparedOperation, 8, mem::AllocatorRef>;
  using Requirements =
      container::SmallVector<compute::detail::ResourceBindingRequirement, 4,
                             mem::AllocatorRef>;
  constexpr std::uint32_t kNone = detail::kNoWorkspaceRequest;
  const std::size_t target_count = plan.targets.size();
  const std::size_t storage_count = plan.storage_domains.size();
  const std::size_t value_count = graph.values.size();
  const std::size_t node_count = graph.nodes.size();
  if (options.max_placement_attempts == 0)
    return Status::InvalidArgument("graph placement budget is zero");
  if (value_count != 0 && storage_count > SIZE_MAX / value_count) {
    return Status::ResourceExhausted(
        "graph physical-instance matrix size overflows");
  }
  if (node_count != 0 && target_count > SIZE_MAX / node_count) {
    return Status::ResourceExhausted(
        "graph placement capability matrix size overflows");
  }

  Ordinals node_required(plan.allocator);
  Ordinals node_preferred(plan.allocator);
  Ordinals value_required(plan.allocator);
  Ordinals value_preferred(plan.allocator);
  Flags external_seen(plan.allocator);
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
    plan.external_storage[index] = 0;
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
    plan.external_storage[value] = contract.storage.ordinal();
  }

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

  // Compile-time constants already expose their exact resource. Validate it
  // against the selected exact storage resource now.
  for (std::size_t index = 0; index < graph.constants.size(); ++index) {
    const std::uint32_t value = graph.constants[index];
    const mem::MemoryResourceRef actual =
        plan.constant_bindings[index].storage().resource();
    const std::uint32_t storage = plan.external_storage[value];
    if (storage >= storage_count ||
        plan.storage_domains[storage].resource != actual) {
      return Status::InvalidArgument(
          "constant binding does not match its external resource contract");
    }
  }

  auto &extensions = discovery.extensions;
  auto &all_region_candidates = discovery.candidates;
  auto &all_region_nodes = discovery.nodes;
  if (!discovery.ready) {
    std::size_t region_proposal_count = 0;
    std::size_t region_proposal_node_count = 0;
    for (std::size_t target = 0; target < target_count; ++target) {
      Status discovered = discover_region_candidates(
          published_graph, graph, *environment.targets[target].context, options,
          provider_policy, plan.allocator, extensions, all_region_candidates,
          all_region_nodes, region_proposal_count, region_proposal_node_count,
          plan.search, ExecutionTargetId{static_cast<std::uint32_t>(target)});
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
    discovery.ready = true;
  }
  Flags region_support(plan.allocator);
  if (!region_support.try_resize(node_count * target_count))
    return plan_allocation_failure();
  for (std::uint8_t &supported : region_support)
    supported = 0;
  for (const RegionCandidate &candidate : all_region_candidates) {
    if (discovery.rejected[candidate.proposal_order] != 0)
      continue;
    for (NodeId node : candidate_nodes(candidate, all_region_nodes)) {
      region_support[static_cast<std::size_t>(node.ordinal()) * target_count +
                     candidate.target.ordinal()] = 1;
    }
  }

  Ordinals producer_storage(plan.allocator);
  Ordinals node_targets(plan.allocator);
  Ordinals node_workspace_storage(plan.allocator);
  Ordinals selected_edge_storage(plan.allocator);
  Ordinals target_candidates(plan.allocator);
  Prepared prepared(plan.allocator);
  if (!producer_storage.try_resize(value_count) ||
      !node_targets.try_resize(node_count) ||
      !node_workspace_storage.try_resize(node_count) ||
      !selected_edge_storage.try_resize(graph.edges.size()) ||
      !target_candidates.try_resize(target_count) ||
      !prepared.try_resize(node_count)) {
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
  Requirements input_requirements(plan.allocator);
  Requirements output_requirements(plan.allocator);
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
        if (matched.IsInternal())
          return std::unexpected(std::move(matched));
        return false;
      }
      Status accessible =
          plan.targets[target].context.validate_resource(resource);
      if (!accessible.ok()) {
        if (accessible.IsInternal())
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
    const std::array preferred{first, second};
    for (std::uint32_t storage : preferred) {
      if (storage == kNone)
        continue;
      auto accepted = compatible(storage);
      if (!accepted)
        return std::unexpected(std::move(accepted).error());
      if (*accepted)
        return storage;
    }
    for (std::uint32_t storage = 0; storage < storage_count; ++storage) {
      if (storage == first || storage == second)
        continue;
      auto accepted = compatible(storage);
      if (!accepted)
        return std::unexpected(std::move(accepted).error());
      if (*accepted)
        return storage;
    }
    return std::unexpected(Status::Unimplemented(
        "no graph storage satisfies provider resource constraints"));
  };

  auto prepare_operation = [&](std::size_t node_index,
                               std::uint32_t candidate) noexcept
      -> Result<compute::detail::ProviderPrepareAttempt> {
    const detail::NodeRecord &node = graph.nodes[node_index];
    const compute::Context &context = *environment.targets[candidate].context;
    const std::uint32_t default_storage =
        plan.targets[candidate].default_storage.ordinal();
    if (!selected_inputs.try_resize(node.input_count) ||
        !selected_outputs.try_resize(node.output_count) ||
        !input_requirements.try_resize(node.input_count) ||
        !output_requirements.try_resize(node.output_count)) {
      return std::unexpected(plan_allocation_failure());
    }

    Status unsupported = Status::Unimplemented(
        "no configured provider accepts the graph operation resources");
    auto decline = compute::detail::ProviderPrepareDecline::kUnsupported;
    for (std::size_t provider = 0; provider < context.provider_count();
         ++provider) {
      auto allowed =
          provider_policy.allows_operation(context, provider, node_index);
      if (!allowed)
        return std::unexpected(std::move(allowed).error());
      if (!*allowed)
        continue;
      decline = compute::detail::ProviderPrepareDecline::kUnsupported;
      auto proposal = compute::detail::OperationPreparationAccess::propose(
          context, provider, node.operation, inputs, outputs,
          input_requirements, output_requirements);
      if (!proposal) {
        unsupported = std::move(proposal).error();
        if (unsupported.IsUnimplemented())
          continue;
        return std::unexpected(std::move(unsupported));
      }

      if (!replay_allowed(options.provider_replay,
                          proposal->runtime.replay_class)) {
        unsupported = Status::Unimplemented(
            "operation provider does not declare required prepared replay");
        continue;
      }

      bool route_supported = true;
      for (std::size_t input_index = 0; input_index < node.input_count;
           ++input_index) {
        const std::uint32_t value = graph.edges[node.input_begin + input_index];
        auto storage =
            select_storage(candidate, input_requirements[input_index], kNone,
                           producer_storage[value], default_storage);
        if (!storage) {
          unsupported = std::move(storage).error();
          if (!unsupported.IsUnimplemented())
            return std::unexpected(std::move(unsupported));
          route_supported = false;
          break;
        }
        selected_inputs[input_index] = *storage;
        input_resources[input_index] = plan.storage_domains[*storage].resource;
      }
      if (!route_supported)
        continue;

      const compute::OperationTraits traits =
          compute::operation_traits(node.operation);
      if (traits.alias == compute::AliasPolicy::kMustAliasInput0 &&
          node.input_count == 0) {
        return std::unexpected(Status::Internal(
            "operation requires an input alias but has no inputs"));
      }
      for (std::size_t output_index = 0; output_index < node.output_count;
           ++output_index) {
        const std::uint32_t value =
            node.output_begin + static_cast<std::uint32_t>(output_index);
        const bool must_alias_input =
            traits.alias == compute::AliasPolicy::kMustAliasInput0;
        const std::uint32_t required_storage =
            must_alias_input
                ? (value_required[value] != kNone &&
                           value_required[value] != selected_inputs[0]
                       ? kNone
                       : selected_inputs[0])
                : value_required[value];
        if (must_alias_input && value_required[value] != kNone &&
            value_required[value] != selected_inputs[0]) {
          route_supported = false;
          unsupported = Status::Unimplemented(
              "required output storage contradicts operation aliasing");
          break;
        }
        const std::uint32_t preferred_storage = value_preferred[value] != kNone
                                                    ? value_preferred[value]
                                                    : default_storage;
        auto storage = select_storage(
            candidate, output_requirements[output_index], required_storage,
            preferred_storage, default_storage);
        if (!storage) {
          unsupported = std::move(storage).error();
          if (!unsupported.IsUnimplemented())
            return std::unexpected(std::move(unsupported));
          route_supported = false;
          break;
        }
        selected_outputs[output_index] = *storage;
        output_resources[output_index] =
            plan.storage_domains[*storage].resource;
      }
      if (!route_supported)
        continue;

      std::uint32_t workspace_storage = kNone;
      mem::MemoryResourceRef workspace_resource{};
      if (proposal->runtime.workspace_owner ==
          compute::ProviderWorkspaceOwner::kCaller) {
        auto selected = select_storage(
            candidate, proposal->workspace_resource, kNone, default_storage,
            node.output_count != 0
                ? selected_outputs[0]
                : (node.input_count != 0 ? selected_inputs[0] : kNone));
        if (!selected) {
          unsupported = std::move(selected).error();
          if (!unsupported.IsUnimplemented())
            return std::unexpected(std::move(unsupported));
          continue;
        }
        workspace_storage = *selected;
        workspace_resource = plan.storage_domains[*selected].resource;
      }
      if (placement_attempts == options.max_placement_attempts) {
        return std::unexpected(Status::ResourceExhausted(
            "graph placement preparation budget is exhausted"));
      }
      ++placement_attempts;
      auto result =
          compute::detail::OperationPreparationAccess::prepare_candidate(
              context, *proposal, node.operation, inputs, outputs,
              input_requirements, output_requirements,
              compute::OperationResourceBindings{
                  .inputs = input_resources,
                  .outputs = output_resources,
                  .workspace_resource = workspace_resource,
              });
      if (!result) {
        return std::unexpected(std::move(result).error());
      }
      if (!result->succeeded()) {
        decline = result->decline;
        unsupported = std::move(result->reason);
        continue;
      }

      for (std::size_t input_index = 0; input_index < node.input_count;
           ++input_index) {
        selected_edge_storage[node.input_begin + input_index] =
            selected_inputs[input_index];
      }
      for (std::size_t output_index = 0; output_index < node.output_count;
           ++output_index) {
        producer_storage[node.output_begin + output_index] =
            selected_outputs[output_index];
      }
      node_workspace_storage[node_index] = workspace_storage;
      return result;
    }
    return compute::detail::ProviderPrepareAttempt{
        .decline = decline, .reason = std::move(unsupported)};
  };

  for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    std::uint32_t required = node_required[node_index];
    std::uint32_t preferred = node_preferred[node_index];
    for (std::size_t index = 0; index < target_count; ++index)
      target_candidates[index] = static_cast<std::uint32_t>(index);
    std::sort(target_candidates.begin(), target_candidates.end(),
              [&](std::uint32_t left, std::uint32_t right) noexcept {
                if (required != kNone)
                  return left == required && right != required;
                const bool left_preferred = left == preferred;
                const bool right_preferred = right == preferred;
                if (left_preferred != right_preferred)
                  return left_preferred;
                const bool left_region =
                    region_support[node_index * target_count + left] != 0;
                const bool right_region =
                    region_support[node_index * target_count + right] != 0;
                if (left_region != right_region)
                  return left_region;
                std::size_t left_local = 0;
                std::size_t right_local = 0;
                for (std::size_t input_index = 0;
                     input_index < node.input_count; ++input_index) {
                  const std::uint32_t value =
                      graph.edges[node.input_begin + input_index];
                  left_local +=
                      producer_storage[value] ==
                              plan.targets[left].default_storage.ordinal()
                          ? 1u
                          : 0u;
                  right_local +=
                      producer_storage[value] ==
                              plan.targets[right].default_storage.ordinal()
                          ? 1u
                          : 0u;
                }
                if (left_local != right_local)
                  return left_local > right_local;
                return left < right;
              });
    if (!inputs.try_resize(node.input_count) ||
        !outputs.try_resize(node.output_count) ||
        !input_resources.try_resize(node.input_count) ||
        !output_resources.try_resize(node.output_count)) {
      return plan_allocation_failure();
    }
    for (std::size_t index = 0; index < node.input_count; ++index)
      inputs[index] = plan.descriptions[graph.edges[node.input_begin + index]];
    for (std::size_t index = 0; index < node.output_count; ++index)
      outputs[index] = plan.descriptions[node.output_begin + index];

    Status unsupported = Status::Unimplemented(
        "no execution target supports the graph operation");
    bool selected = false;
    for (std::uint32_t candidate : target_candidates) {
      if (required != kNone && candidate != required)
        continue;
      const std::uint32_t default_storage =
          plan.targets[candidate].default_storage.ordinal();
      if (region_support[node_index * target_count + candidate] != 0) {
        node_targets[node_index] = candidate;
        for (std::size_t output_index = 0; output_index < node.output_count;
             ++output_index) {
          const std::uint32_t value =
              node.output_begin + static_cast<std::uint32_t>(output_index);
          producer_storage[value] =
              value_required[value] != kNone    ? value_required[value]
              : value_preferred[value] != kNone ? value_preferred[value]
                                                : default_storage;
        }
        selected = true;
        break;
      }
      if (options.operation_fallback == OperationFallbackPolicy::kReject)
        continue;
      auto result = prepare_operation(node_index, candidate);
      if (!result) {
        return std::move(result).error().WithContext(
            "preparing graph operation placement candidate");
      }
      if (result->succeeded()) {
        prepared[node_index] = std::move(result->prepared);
        node_targets[node_index] = candidate;
        selected = true;
        break;
      }
      unsupported = std::move(result->reason);
    }
    if (!selected) {
      outcome.no_feasible_route = true;
      return unsupported;
    }
  }

  RegionCandidates candidates(plan.allocator);
  RegionCandidates selected_regions(plan.allocator);
  if (!candidates.try_reserve(all_region_candidates.size()))
    return plan_allocation_failure();
  for (const RegionCandidate &candidate : all_region_candidates) {
    if (discovery.rejected[candidate.proposal_order] != 0)
      continue;
    bool eligible = true;
    for (NodeId node : candidate_nodes(candidate, all_region_nodes)) {
      if (node_targets[node.ordinal()] != candidate.target.ordinal()) {
        eligible = false;
        break;
      }
    }
    if (eligible && !candidates.try_push_back(candidate))
      return plan_allocation_failure();
  }
  Status selected = select_regions(candidates, plan.allocator, all_region_nodes,
                                   graph, selected_regions);
  if (!selected.ok())
    return selected;
  if (options.operation_fallback == OperationFallbackPolicy::kReject) {
    std::size_t covered_nodes = 0;
    for (const RegionCandidate &candidate : selected_regions) {
      covered_nodes += candidate.node_count;
    }
    if (covered_nodes != node_count) {
      outcome.no_feasible_route = true;
      return Status::Unimplemented(
          "multi-domain graph specialization requires an operation fallback");
    }
  }
  Flags selected_nodes(plan.allocator);
  if (!selected_nodes.try_resize(node_count))
    return plan_allocation_failure();
  for (std::uint8_t &covered : selected_nodes)
    covered = 0;
  for (const RegionCandidate &candidate : selected_regions) {
    for (NodeId node : candidate_nodes(candidate, all_region_nodes)) {
      selected_nodes[node.ordinal()] = 1;
    }
  }
  for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
    if (selected_nodes[node_index] != 0 || prepared[node_index].valid())
      continue;
    const detail::NodeRecord &node = graph.nodes[node_index];
    const std::uint32_t required = node_required[node_index];
    const std::uint32_t preferred = node_preferred[node_index];
    const std::uint32_t originally_selected = node_targets[node_index];
    for (std::size_t index = 0; index < target_count; ++index)
      target_candidates[index] = static_cast<std::uint32_t>(index);
    std::sort(target_candidates.begin(), target_candidates.end(),
              [&](std::uint32_t left, std::uint32_t right) noexcept {
                if (required != kNone)
                  return left == required && right != required;
                const bool left_original = left == originally_selected;
                const bool right_original = right == originally_selected;
                if (left_original != right_original)
                  return left_original;
                const bool left_preferred = left == preferred;
                const bool right_preferred = right == preferred;
                if (left_preferred != right_preferred)
                  return left_preferred;
                std::size_t left_local = 0;
                std::size_t right_local = 0;
                for (std::size_t input_index = 0;
                     input_index < node.input_count; ++input_index) {
                  const std::uint32_t value =
                      graph.edges[node.input_begin + input_index];
                  left_local +=
                      producer_storage[value] ==
                              plan.targets[left].default_storage.ordinal()
                          ? 1u
                          : 0u;
                  right_local +=
                      producer_storage[value] ==
                              plan.targets[right].default_storage.ordinal()
                          ? 1u
                          : 0u;
                }
                if (left_local != right_local)
                  return left_local > right_local;
                return left < right;
              });
    if (!inputs.try_resize(node.input_count) ||
        !outputs.try_resize(node.output_count) ||
        !input_resources.try_resize(node.input_count) ||
        !output_resources.try_resize(node.output_count)) {
      return plan_allocation_failure();
    }
    for (std::size_t index = 0; index < node.input_count; ++index)
      inputs[index] = plan.descriptions[graph.edges[node.input_begin + index]];
    for (std::size_t index = 0; index < node.output_count; ++index)
      outputs[index] = plan.descriptions[node.output_begin + index];

    Status unsupported = Status::Unimplemented(
        "no execution target supports an unselected graph operation");
    bool fallback_selected = false;
    for (std::uint32_t candidate : target_candidates) {
      if (required != kNone && candidate != required)
        continue;
      auto fallback = prepare_operation(node_index, candidate);
      if (!fallback) {
        return std::move(fallback).error().WithContext(
            "preparing operation left outside selected graph regions");
      }
      if (fallback->succeeded()) {
        prepared[node_index] = std::move(fallback->prepared);
        node_targets[node_index] = candidate;
        fallback_selected = true;
        break;
      }
      unsupported = std::move(fallback->reason);
    }
    if (!fallback_selected) {
      outcome.no_feasible_route = true;
      return unsupported;
    }
  }

  enum class RegionCoverage : std::uint8_t { kNone, kEligible, kRejected };
  container::SmallVector<RegionCoverage, 32, mem::AllocatorRef> region_coverage(
      plan.allocator);
  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> marks(
      plan.allocator);
  Flags region_node_membership(plan.allocator);
  Flags region_outside_uses(plan.allocator);
  if (!region_coverage.try_resize(node_count) ||
      !marks.try_resize(value_count)) {
    return plan_allocation_failure();
  }
  std::fill(region_coverage.begin(), region_coverage.end(),
            RegionCoverage::kNone);
  for (const RegionCandidate &candidate : candidates) {
    for (NodeId node : candidate_nodes(candidate, all_region_nodes))
      region_coverage[node.ordinal()] = RegionCoverage::kEligible;
  }
  for (const RegionCandidate &candidate : all_region_candidates) {
    if (discovery.rejected[candidate.proposal_order] != 0) {
      for (NodeId node : candidate_nodes(candidate, all_region_nodes))
        region_coverage[node.ordinal()] = RegionCoverage::kRejected;
    }
  }
  for (std::uint32_t &mark : marks)
    mark = 0;

  // With no explicit output storage, keep a published node result where its
  // producer already resides. This avoids manufacturing a boundary copy just
  // because storage zero appeared first in the environment.
  for (std::uint32_t value : graph.outputs) {
    if (external_seen[value] == 0 &&
        graph.values[value].producer != detail::kNoNode) {
      plan.external_storage[value] = producer_storage[value];
    }
  }

  const std::size_t maximum_instances = value_count * storage_count;
  if (node_count > SIZE_MAX - maximum_instances) {
    return Status::ResourceExhausted("graph plan capacity overflows");
  }
  Ordinals instances(plan.allocator);
  Ordinals producer_instances(plan.allocator);
  if (!instances.try_resize(maximum_instances) ||
      !producer_instances.try_resize(value_count)) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &instance : instances)
    instance = kNone;
  for (std::uint32_t &instance : producer_instances)
    instance = kNone;

  plan.steps.clear();
  plan.node_inputs.clear();
  plan.node_outputs.clear();
  plan.step_nodes.clear();
  plan.instance_descriptions.clear();
  plan.instance_values.clear();
  plan.instance_storage.clear();
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

  auto append_copy = [&](std::uint32_t source,
                         std::uint32_t destination) noexcept -> Status {
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
    const mem::MemoryResourceRef source_resource =
        plan.storage_domains[source_storage].resource;
    const mem::MemoryResourceRef destination_resource =
        plan.storage_domains[destination_storage].resource;
    auto supports_route = [&](std::uint32_t target) noexcept {
      return plan.targets[target]
                 .context.validate_resource(source_resource)
                 .ok() &&
             plan.targets[target]
                 .context.validate_resource(destination_resource)
                 .ok();
    };
    std::uint32_t execution_target = kNone;
    for (std::size_t index = 0; index < target_count; ++index) {
      if (plan.targets[index].default_storage.ordinal() ==
              destination_storage &&
          supports_route(static_cast<std::uint32_t>(index))) {
        execution_target = static_cast<std::uint32_t>(index);
        break;
      }
    }
    if (execution_target == kNone) {
      for (std::size_t index = 0; index < target_count; ++index) {
        if (plan.targets[index].default_storage.ordinal() == source_storage &&
            supports_route(static_cast<std::uint32_t>(index))) {
          execution_target = static_cast<std::uint32_t>(index);
          break;
        }
      }
    }
    if (execution_target == kNone) {
      for (std::size_t index = 0; index < target_count; ++index) {
        if (supports_route(static_cast<std::uint32_t>(index))) {
          execution_target = static_cast<std::uint32_t>(index);
          break;
        }
      }
    }
    if (execution_target == kNone) {
      return Status::Unimplemented(
          "no execution target can transfer between the selected resources");
    }
    if (!plan.steps.try_push_back(detail::PlanStep::copy(
            input_begin, output_begin, *bytes, StorageDomainId{source_storage},
            StorageDomainId{destination_storage},
            ExecutionTargetId{execution_target}))) {
      return plan_allocation_failure();
    }
    plan.value_materialized[destination] = 1;
    return Status::Ok();
  };

  auto ensure_local_instance =
      [&](std::uint32_t value,
          std::uint32_t storage) noexcept -> Result<std::uint32_t> {
    std::uint32_t instance =
        instances[static_cast<std::size_t>(value) * storage_count + storage];
    if (instance != kNone)
      return instance;
    const std::uint32_t source = producer_instances[value];
    if (source == kNone) {
      return std::unexpected(
          Status::Internal("graph value has no physical producer"));
    }
    auto created = add_instance(value, storage, 0);
    if (!created)
      return std::unexpected(std::move(created).error());
    Status copied = append_copy(source, *created);
    if (!copied.ok())
      return std::unexpected(std::move(copied));
    return *created;
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
    if (!*ordered)
      return Status::Internal(
          "selected graph regions form a cyclic contraction");
    unit_order = contraction.order();
  }

  Ordinals physical_inputs(plan.allocator);
  std::uint32_t marker = 1;
  NodeId operation_node;
  std::span<const NodeId> operation_nodes;
  for (std::uint32_t unit_node : unit_order) {
    operation_nodes = {};
    if (region_by_node[unit_node] != kNone) {
      const RegionCandidate &candidate =
          selected_regions[region_by_node[unit_node]];
      const std::span<const NodeId> region_nodes =
          candidate_nodes(candidate, all_region_nodes);
      const std::uint32_t target = candidate.target.ordinal();
      const std::uint32_t default_storage =
          plan.targets[target].default_storage.ordinal();
      RegionValues logical_inputs(plan.allocator);
      RegionValues logical_outputs(plan.allocator);
      Status boundary = collect_region_boundary(
          plan, graph, region_nodes, region_node_membership,
          region_outside_uses, marks, marker++, logical_inputs,
          logical_outputs);
      if (!boundary.ok())
        return boundary;
      if (!selected_inputs.try_resize(logical_inputs.size()) ||
          !selected_outputs.try_resize(logical_outputs.size()) ||
          !input_resources.try_resize(logical_inputs.size()) ||
          !output_resources.try_resize(logical_outputs.size()) ||
          !input_requirements.try_resize(logical_inputs.size()) ||
          !output_requirements.try_resize(logical_outputs.size())) {
        return plan_allocation_failure();
      }
      container::SmallVector<RegionEffectAccess, 4, mem::AllocatorRef>
          effect_accesses(plan.allocator);
      bool unknown_effects = false;
      for (NodeId node_id : region_nodes) {
        const auto &node = graph.nodes[node_id.ordinal()];
        const auto traits = compute::operation_traits(node.operation);
        unknown_effects |=
            traits.effect == compute::EffectKind::kExternalSideEffect;
        for (std::size_t port = 0; port < traits.resource_count; ++port) {
          auto access =
              compute::operation_resource_access(node.operation, port);
          if (!access)
            return std::move(access).error();
          if (!effect_accesses.try_push_back(RegionEffectAccess{
                  detail::GraphIdentity::effect_resource(
                      graph.generation,
                      graph.resource_edges[node.resource_begin + port]),
                  *access}))
            return plan_allocation_failure();
        }
      }
      std::sort(effect_accesses.begin(), effect_accesses.end(),
                [](const auto &left, const auto &right) noexcept {
                  return left.resource.ordinal() < right.resource.ordinal();
                });
      std::size_t effect_count = 0;
      for (const auto access : effect_accesses) {
        if (effect_count != 0 &&
            effect_accesses[effect_count - 1].resource == access.resource) {
          if (effect_accesses[effect_count - 1].access != access.access)
            effect_accesses[effect_count - 1].access =
                compute::ResourceAccess::kReadWrite;
        } else {
          effect_accesses[effect_count++] = access;
        }
      }
      effect_accesses.try_resize(effect_count);
      RegionCompileRequest request{
          .graph = &published_graph,
          .nodes = region_nodes,
          .inputs = logical_inputs,
          .outputs = logical_outputs,
          .constants = options.constants,
          .route = candidate.route,
          .effect_accesses = effect_accesses,
          .effect_bindings = plan.effect_bindings,
          .has_unknown_effects = unknown_effects,
      };
      RegionResourceProposal resource_proposal{};
      Status region_status = propose_region_resources(
          extensions[candidate.extension_index], request, input_requirements,
          output_requirements, resource_proposal);
      bool route_supported = region_status.ok();
      if (!route_supported && !region_status.IsUnimplemented())
        return region_status;
      if (route_supported && (unknown_effects || !effect_accesses.empty()) &&
          !resource_proposal.preserves_effects) {
        region_status = Status::Unimplemented(
            "region provider does not preserve requested resource effects");
        route_supported = false;
      }
      if (route_supported &&
          candidate.replay_class != compute::ProviderReplayClass::kUnknown &&
          candidate.replay_class != resource_proposal.runtime.replay_class) {
        return Status::Internal(
            "region provider changed its declared replay class");
      }

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
        input_resources[index] = plan.storage_domains[*storage].resource;
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
        output_resources[index] = plan.storage_domains[*storage].resource;
      }
      std::uint32_t workspace_storage = kNone;
      mem::MemoryResourceRef workspace_resource{};
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
          workspace_resource = plan.storage_domains[*storage].resource;
        }
      }

      Result<detail::PreparedRegion> region =
          std::unexpected(Status::Unimplemented(
              "selected graph region has no compatible resource route"));
      if (route_supported) {
        request.input_resources = input_resources;
        request.output_resources = output_resources;
        request.workspace_resource = workspace_resource;
        if (placement_attempts == options.max_placement_attempts) {
          return Status::ResourceExhausted(
              "graph placement preparation budget is exhausted");
        }
        ++placement_attempts;
        auto attempt =
            compile_region_artifact(plan, extensions[candidate.extension_index],
                                    request, resource_proposal, options);
        if (!attempt)
          return std::move(attempt).error();
        if (attempt->prepared.valid()) {
          region = std::move(attempt->prepared);
        } else {
          region_status = std::move(attempt->reason);
          route_supported = false;
        }
      }
      if (!route_supported) {
        // This is an explicit cold rejection, not an arbitrary error code.
        // Discard the construction before reconsidering its partition/target.
        // Fatal prepare errors returned above never reach this branch.
        outcome.rejected_region = candidate.proposal_order;
        return region_status;
      } else {
        physical_inputs.clear();
        if (!physical_inputs.try_reserve(logical_inputs.size()))
          return plan_allocation_failure();
        for (std::size_t index = 0; index < logical_inputs.size(); ++index) {
          auto instance = ensure_local_instance(logical_inputs[index].ordinal(),
                                                selected_inputs[index]);
          if (!instance)
            return std::move(instance).error();
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
                std::move(*region), region_nodes.front().ordinal(),
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
    } else {
      operation_node = detail::GraphIdentity::node(graph.generation, unit_node);
      operation_nodes = std::span<const NodeId>(&operation_node, 1);
    }

    for (NodeId operation : operation_nodes) {
      const std::uint32_t node_index = operation.ordinal();
      const detail::NodeRecord &node = graph.nodes[node_index];
      if (!prepared[node_index].valid()) {
        if (!inputs.try_resize(node.input_count) ||
            !outputs.try_resize(node.output_count) ||
            !input_resources.try_resize(node.input_count) ||
            !output_resources.try_resize(node.output_count)) {
          return plan_allocation_failure();
        }
        for (std::size_t index = 0; index < node.input_count; ++index)
          inputs[index] =
              plan.descriptions[graph.edges[node.input_begin + index]];
        for (std::size_t index = 0; index < node.output_count; ++index)
          outputs[index] = plan.descriptions[node.output_begin + index];
        for (std::size_t index = 0; index < target_count; ++index)
          target_candidates[index] = static_cast<std::uint32_t>(index);
        const std::uint32_t required = node_required[node_index];
        const std::uint32_t preferred = node_preferred[node_index];
        const std::uint32_t original = node_targets[node_index];
        std::sort(target_candidates.begin(), target_candidates.end(),
                  [&](std::uint32_t left, std::uint32_t right) noexcept {
                    if (required != kNone)
                      return left == required && right != required;
                    const bool left_original = left == original;
                    const bool right_original = right == original;
                    if (left_original != right_original)
                      return left_original;
                    const bool left_preferred = left == preferred;
                    const bool right_preferred = right == preferred;
                    if (left_preferred != right_preferred)
                      return left_preferred;
                    return left < right;
                  });
        Status unsupported = Status::Unimplemented(
            "no operation fallback supports a declined graph region route");
        for (std::uint32_t candidate : target_candidates) {
          if (required != kNone && candidate != required)
            continue;
          auto fallback = prepare_operation(node_index, candidate);
          if (!fallback)
            return std::move(fallback).error();
          if (fallback->succeeded()) {
            prepared[node_index] = std::move(fallback->prepared);
            node_targets[node_index] = candidate;
            break;
          }
          unsupported = std::move(fallback->reason);
        }
        if (!prepared[node_index].valid())
          return unsupported;
      }
      const std::uint32_t target = node_targets[node_index];
      const std::uint32_t workspace_storage =
          node_workspace_storage[node_index];
      if (workspace_storage == kNone &&
          prepared[node_index].runtime_properties().workspace_owner ==
              compute::ProviderWorkspaceOwner::kCaller)
        return Status::Internal(
            "prepared graph operation has no workspace storage");
      physical_inputs.clear();
      if (!physical_inputs.try_reserve(node.input_count))
        return plan_allocation_failure();
      for (std::size_t input_index = 0; input_index < node.input_count;
           ++input_index) {
        const std::uint32_t value = graph.edges[node.input_begin + input_index];
        const std::uint32_t input_storage =
            selected_edge_storage[node.input_begin + input_index];
        if (input_storage == kNone)
          return Status::Internal(
              "prepared graph input has no selected storage");
        auto instance = ensure_local_instance(value, input_storage);
        if (!instance)
          return std::move(instance).error();
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
        producer_instances[value] = *instance;
        if ((roles & detail::kExternalOutput) != 0)
          plan.external_instances[value] = *instance;
        if (!plan.node_outputs.try_push_back(*instance))
          return plan_allocation_failure();
        plan.value_materialized[*instance] = 1;
      }
      const bool materializes =
          compute::operation_traits(node.operation).materializes;
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
              std::move(prepared[node_index]),
              static_cast<std::uint32_t>(node_index), step_node_begin,
              input_begin, node.input_count, output_begin, node.output_count,
              materializes, reason, ExecutionTargetId{target},
              workspace_storage == kNone
                  ? StorageDomainId{}
                  : StorageDomainId{workspace_storage})) {
        return plan_allocation_failure();
      }
      plan.node_steps[node_index] = step_index;
    }
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
      auto created = add_instance(value, storage, detail::kExternalOutput);
      if (!created)
        return std::move(created).error();
      destination = *created;
      Status copied = append_copy(source, destination);
      if (!copied.ok())
        return copied;
    } else {
      plan.instance_roles[destination] |= detail::kExternalOutput;
    }
    plan.external_instances[value] = destination;
  }

  return Status::Ok();
}

Status prepare_placed_steps(detail::PlanState &plan,
                            const Graph &published_graph,
                            const detail::GraphState &graph,
                            CompileEnvironment environment,
                            CompileOptions options,
                            const ProviderPolicy &provider_policy) noexcept {
  RegionDiscovery discovery(plan.allocator);
  Status last_rejection = Status::Ok();
  for (;;) {
    // No Invocation or user work exists during this transaction. Clear all
    // artifacts before calling any provider for the next construction; the
    // attempt resets its physical metadata and keeps aggregate budgets.
    plan.steps.clear();
    plan.region_count = 0;
    std::fill(plan.node_steps.begin(), plan.node_steps.end(), detail::kNoNode);
    PlacementAttemptOutcome outcome;
    Status status = prepare_placed_steps_attempt(
        plan, published_graph, graph, environment, options, provider_policy,
        discovery, outcome);
    if (!outcome.rejected_region) {
      if (outcome.no_feasible_route && !last_rejection.ok())
        return last_rejection;
      return status;
    }
    const std::uint32_t rejected = *outcome.rejected_region;
    if (status.ok() || rejected >= discovery.rejected.size() ||
        discovery.rejected[rejected] != 0) {
      return Status::Internal("graph region retry did not exclude a candidate");
    }
    discovery.rejected[rejected] = 1;
    ++plan.search.region_route_rejections;
    last_rejection = std::move(status);
  }
}

Status discover_region_candidates(
    const Graph &published_graph, const detail::GraphState &graph,
    const compute::Context &context, CompileOptions options,
    const ProviderPolicy &provider_policy, mem::AllocatorRef allocator,
    RegionExtensions &extensions, RegionCandidates &candidates,
    RegionNodeIds &proposal_nodes, std::size_t &proposal_count,
    std::size_t &proposal_node_count, CompileSearchReport &search,
    ExecutionTargetId target,
    std::span<const std::uint32_t> node_targets) noexcept {
  if (options.region_selection == RegionSelectionPolicy::kDisabled ||
      graph.nodes.empty()) {
    return Status::Ok();
  }
  if (options.max_region_proposals == 0) {
    return Status::InvalidArgument(
        "enabled graph region selection requires a non-zero proposal budget");
  }
  if (options.max_region_proposals_per_provider == 0) {
    return Status::InvalidArgument(
        "enabled graph region selection requires a non-zero per-provider "
        "proposal budget");
  }
  if (options.max_region_proposal_nodes == 0) {
    return Status::InvalidArgument(
        "enabled graph region selection requires a non-zero node budget");
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
  if (context.provider_count() > (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::ResourceExhausted(
        "compute provider identity space is exhausted");
  }
  const std::size_t extension_begin = extensions.size();
  for (std::size_t index = 0; index < context.provider_count(); ++index) {
    auto allowed = provider_policy.allows_family(context, index);
    if (!allowed)
      return std::move(allowed).error();
    if (!*allowed)
      continue;
    auto extension = compute::detail::ProviderExtensionAccess::query(
        context, index, kRegionProviderExtensionId);
    if (!extension) {
      if (extension.error().IsUnimplemented())
        continue;
      return std::move(extension).error();
    }
    if (!extensions.try_push_back(
            RegionExtension{std::move(*extension),
                            static_cast<std::uint32_t>(index), target})) {
      return plan_allocation_failure();
    }
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
  for (std::size_t extension_index = extension_begin;
       extension_index < extensions.size(); ++extension_index) {
    const std::size_t remaining = options.max_region_proposals - proposal_count;
    const std::size_t provider_capacity =
        (std::min)(remaining, options.max_region_proposals_per_provider);
    const std::size_t remaining_nodes =
        options.max_region_proposal_nodes - proposal_node_count;
    if (remaining == 0 || remaining_nodes == 0) {
      // There is still an unvisited region provider. No zero-capacity callback
      // is needed, and no later target/provider can restore a global budget.
      search.region_proposal_limit_reached |= remaining == 0;
      search.region_node_limit_reached |= remaining_nodes == 0;
      break;
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
      const Graph *graph = nullptr;
      RegionProposalWriter *writer = nullptr;
    } episode{&published_graph, &writer};
    const Status proposed = compute::detail::ProviderExtensionAccess::call(
        extensions[extension_index].extension,
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
          return interface->propose(provider_context, *call.graph,
                                    *call.writer);
        },
        &episode);
    if (!proposed.ok())
      return proposed;

    if (writer.proposal_limit_reached()) {
      search.region_proposal_limit_reached |= provider_capacity == remaining;
      search.per_provider_proposal_limit_reached |=
          provider_capacity == options.max_region_proposals_per_provider;
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
      if (!replay_allowed(options.provider_replay, proposal.replay_class))
        continue;
      const auto &extension = extensions[extension_index].extension;
      if (!std::all_of(proposal.nodes.begin(), proposal.nodes.end(),
                       [&](NodeId node) noexcept {
                         return provider_policy.allows_node(
                             node.ordinal(), extension.provider().id,
                             extension.provider_instance_id());
                       }))
        continue;
      bool target_eligible = true;
      if (!node_targets.empty()) {
        for (NodeId node : proposal.nodes) {
          if (node_targets[node.ordinal()] != target.ordinal()) {
            target_eligible = false;
            break;
          }
        }
      }
      if (!target_eligible)
        continue;
      const std::ptrdiff_t node_offset =
          proposal.nodes.data() - proposal_nodes.data();
      if (node_offset < 0 ||
          static_cast<std::size_t>(node_offset) >
              options.max_region_proposal_nodes - proposal.nodes.size()) {
        return Status::Internal(
            "graph region proposal node storage is not writer-owned");
      }
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
          })) {
        return plan_allocation_failure();
      }
    }
    proposal_count += writer.size();
    proposal_node_count += writer.node_size();
  }
  return Status::Ok();
}

struct RegionSelectionScore final {
  std::uint64_t covered_nodes = 0;
  std::uint64_t dispatches = 0;
  std::uint64_t provider_rank_cost = 0;
};

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

bool better_score(RegionSelectionScore left, RegionSelectionScore right,
                  bool compare_dispatches) noexcept {
  if (left.covered_nodes != right.covered_nodes)
    return left.covered_nodes > right.covered_nodes;
  if (compare_dispatches && left.dispatches != right.dispatches)
    return left.dispatches < right.dispatches;
  return left.provider_rank_cost < right.provider_rank_cost;
}

Status select_regions(RegionCandidates &candidates, mem::AllocatorRef allocator,
                      std::span<const NodeId> proposal_nodes,
                      const detail::GraphState &graph,
                      RegionCandidates &selected) noexcept {
  // A missing count is not zero, nor evidence against the candidate. Disable
  // this whole score coordinate when any candidate is unknown. Comparing it
  // only pairwise when known would also violate sort's transitivity.
  const bool compare_dispatches =
      std::all_of(candidates.begin(), candidates.end(),
                  [](const RegionCandidate &candidate) {
                    return candidate.cost.dispatch_count.has_value();
                  });
  const bool intervals_only = std::all_of(
      candidates.begin(), candidates.end(),
      [&](const RegionCandidate &candidate) noexcept {
        return nodes_are_contiguous(candidate_nodes(candidate, proposal_nodes));
      });
  if (!intervals_only) {
    std::sort(
        candidates.begin(), candidates.end(),
        [&](const RegionCandidate &left,
            const RegionCandidate &right) noexcept {
          if (left.node_count != right.node_count)
            return left.node_count > right.node_count;
          if (compare_dispatches &&
              left.cost.dispatch_count != right.cost.dispatch_count)
            return left.cost.dispatch_count < right.cost.dispatch_count;
          if (left.provider_index != right.provider_index)
            return left.provider_index < right.provider_index;
          const auto left_nodes = candidate_nodes(left, proposal_nodes);
          const auto right_nodes = candidate_nodes(right, proposal_nodes);
          if (left_nodes.front().ordinal() != right_nodes.front().ordinal()) {
            return left_nodes.front().ordinal() < right_nodes.front().ordinal();
          }
          if (left.route != right.route)
            return left.route < right.route;
          return left.proposal_order < right.proposal_order;
        });
    container::SmallVector<std::uint8_t, 32, mem::AllocatorRef> occupied(
        allocator);
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> owners(
        allocator);
    if (!occupied.try_resize(graph.nodes.size()) ||
        !owners.try_resize(graph.nodes.size()) ||
        !selected.try_reserve(candidates.size())) {
      return plan_allocation_failure();
    }
    std::fill(occupied.begin(), occupied.end(), std::uint8_t{0});
    for (std::uint32_t node = 0; node < owners.size(); ++node)
      owners[node] = node;
    RegionContraction contraction(allocator);
    Status initialized = contraction.initialize(graph);
    if (!initialized.ok())
      return initialized;
    for (const RegionCandidate &candidate : candidates) {
      const auto nodes = candidate_nodes(candidate, proposal_nodes);
      if (std::any_of(nodes.begin(), nodes.end(), [&](NodeId node) noexcept {
            return occupied[node.ordinal()] != 0;
          })) {
        continue;
      }
      for (NodeId node : nodes)
        owners[node.ordinal()] = nodes.front().ordinal();
      // Disjoint individually contractible sets can still form a cycle when
      // combined. Test the proposed partition, not its bounding intervals:
      // interleaved sets remain eligible whenever their joint DAG is valid.
      auto acyclic = contraction.sort(owners);
      if (!acyclic)
        return std::move(acyclic).error();
      if (!*acyclic) {
        for (NodeId node : nodes)
          owners[node.ordinal()] = node.ordinal();
        continue;
      }
      if (!selected.try_push_back(candidate))
        return plan_allocation_failure();
      for (NodeId node : nodes)
        occupied[node.ordinal()] = 1;
    }
    std::sort(
        selected.begin(), selected.end(),
        [&](const RegionCandidate &left,
            const RegionCandidate &right) noexcept {
          const auto left_nodes = candidate_nodes(left, proposal_nodes);
          const auto right_nodes = candidate_nodes(right, proposal_nodes);
          if (left_nodes.front().ordinal() != right_nodes.front().ordinal()) {
            return left_nodes.front().ordinal() < right_nodes.front().ordinal();
          }
          return left.proposal_order < right.proposal_order;
        });
    return Status::Ok();
  }

  std::sort(
      candidates.begin(), candidates.end(),
      [&](const RegionCandidate &left, const RegionCandidate &right) noexcept {
        const auto left_nodes = candidate_nodes(left, proposal_nodes);
        const auto right_nodes = candidate_nodes(right, proposal_nodes);
        const std::uint64_t left_end = left_nodes.back().ordinal() + 1ULL;
        const std::uint64_t right_end = right_nodes.back().ordinal() + 1ULL;
        if (left_end != right_end)
          return left_end < right_end;
        if (left_nodes.front().ordinal() != right_nodes.front().ordinal())
          return left_nodes.front().ordinal() < right_nodes.front().ordinal();
        if (left.extension_index != right.extension_index)
          return left.extension_index < right.extension_index;
        if (left.route != right.route)
          return left.route < right.route;
        return left.proposal_order < right.proposal_order;
      });
  using Indices = container::SmallVector<std::uint32_t, 32, mem::AllocatorRef>;
  using Scores =
      container::SmallVector<RegionSelectionScore, 32, mem::AllocatorRef>;
  using Decisions = container::SmallVector<std::uint8_t, 32, mem::AllocatorRef>;
  Indices predecessors(allocator);
  Scores scores(allocator);
  Decisions take(allocator);
  if (!predecessors.try_resize(candidates.size()) ||
      !scores.try_resize(candidates.size() + 1) ||
      !take.try_resize(candidates.size())) {
    return plan_allocation_failure();
  }
  scores[0] = {};
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto nodes = candidate_nodes(candidates[index], proposal_nodes);
    const std::uint64_t begin = nodes.front().ordinal();
    const auto previous_end = std::upper_bound(
        candidates.begin(), candidates.begin() + index, begin,
        [&](std::uint64_t value, const RegionCandidate &candidate) noexcept {
          return value <
                 candidate_nodes(candidate, proposal_nodes).back().ordinal() +
                     1ULL;
        });
    const std::size_t previous_count =
        static_cast<std::size_t>(previous_end - candidates.begin());
    predecessors[index] = static_cast<std::uint32_t>(previous_count);
    RegionSelectionScore include = scores[previous_count];
    include.covered_nodes = saturating_add(
        include.covered_nodes,
        static_cast<std::uint64_t>(candidates[index].node_count));
    if (compare_dispatches) {
      include.dispatches = saturating_add(
          include.dispatches,
          static_cast<std::uint64_t>(*candidates[index].cost.dispatch_count));
    }
    include.provider_rank_cost = saturating_add(
        include.provider_rank_cost,
        saturating_multiply(
            static_cast<std::uint64_t>(candidates[index].provider_index) + 1,
            static_cast<std::uint64_t>(candidates[index].node_count)));
    const RegionSelectionScore exclude = scores[index];
    if (better_score(include, exclude, compare_dispatches)) {
      scores[index + 1] = include;
      take[index] = 1;
    } else {
      scores[index + 1] = exclude;
      take[index] = 0;
    }
  }

  std::size_t cursor = candidates.size();
  while (cursor != 0) {
    const std::size_t index = cursor - 1;
    if (take[index] == 0) {
      --cursor;
      continue;
    }
    if (!selected.try_push_back(candidates[index]))
      return plan_allocation_failure();
    cursor = predecessors[index];
  }
  std::reverse(selected.begin(), selected.end());
  return Status::Ok();
}

// Reorder already-prepared plan steps without changing the logical Graph.
// Kahn scheduling keeps a ready FIFO per execution lane and continues the
// current lane while possible. This deterministically removes avoidable
// Queue segment boundaries from interleaved independent branches; providers,
// transfer routes and bindings remain fully resolved before the warm path.
Status schedule_steps(detail::PlanState &plan,
                      const detail::GraphState &graph) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Edges = container::SmallVector<std::uint64_t, 32, mem::AllocatorRef>;
  constexpr std::uint32_t kNone = detail::kNoNode;
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
              .role = ExecutionLaneRole::kCompute,
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
    const ExecutionLaneRole preferred_role =
        step.kind == detail::PlanStepKind::kCopy ? ExecutionLaneRole::kTransfer
                                                 : ExecutionLaneRole::kCompute;
    std::uint32_t selected = kNone;
    for (std::size_t lane = 0; lane < plan.lanes.size(); ++lane) {
      if (plan.lanes[lane].runtime_mode ==
              compute::ProviderRuntimeMode::kSharedQueue &&
          plan.lanes[lane].target == step.target &&
          plan.lanes[lane].role == preferred_role) {
        selected = static_cast<std::uint32_t>(lane);
        break;
      }
    }
    if (selected == kNone && preferred_role != ExecutionLaneRole::kCompute) {
      for (std::size_t lane = 0; lane < plan.lanes.size(); ++lane) {
        if (plan.lanes[lane].runtime_mode ==
                compute::ProviderRuntimeMode::kSharedQueue &&
            plan.lanes[lane].target == step.target &&
            plan.lanes[lane].role == ExecutionLaneRole::kCompute) {
          selected = static_cast<std::uint32_t>(lane);
          break;
        }
      }
    }
    if (selected == kNone) {
      return Status::InvalidArgument(
          "compile environment has no compatible lane for a graph step");
    }
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
    if (step.kind != detail::PlanStepKind::kOperation &&
        step.kind != detail::PlanStepKind::kRegion) {
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
    if (step.kind != detail::PlanStepKind::kOperation &&
        step.kind != detail::PlanStepKind::kRegion) {
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
  plan.segment_dependencies.clear();
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

  // Metadata-only View steps resolve aliases during binding preparation and
  // do not enter a Queue. Keep them inside the surrounding report interval, but
  // start/split runtime segments only for actual commands. This avoids an
  // empty sequence/event for View-only graphs and lets consecutive commands
  // on one lane remain one uninterrupted Queue prefix.
  std::uint32_t active_segment = detail::kNoNode;
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    if (!plan.steps[index].emits_command())
      continue;
    const ExecutionLaneId lane = plan.steps[index].lane;
    if (!lane.valid() || lane.ordinal() >= plan.lanes.size())
      return Status::Internal("graph plan step has an invalid lane");
    if (active_segment == detail::kNoNode ||
        plan.segments[active_segment].lane != lane) {
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
    }
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
  for (std::size_t step = 0; step < plan.steps.size(); ++step) {
    const std::uint32_t identity = plan.steps[step].schedule_identity;
    if (identity >= identity_steps.size() ||
        identity_steps[identity] != detail::kNoNode) {
      return Status::Internal("graph scheduled-step identity is invalid");
    }
    identity_steps[identity] = static_cast<std::uint32_t>(step);
  }
  for (std::size_t lane = 0; lane < plan.lane_segment_counts.size(); ++lane) {
    if (plan.lane_segment_counts[lane] != 0 &&
        plan.lanes[lane].runtime_mode ==
            compute::ProviderRuntimeMode::kSharedQueue) {
      ++plan.required_queue_count;
    }
  }

  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const detail::PlanStep &step = plan.steps[index];
    for (std::size_t output_index = 0; output_index < step.output_count;
         ++output_index) {
      producers[plan.node_outputs[step.output_begin + output_index]] =
          static_cast<std::uint32_t>(index);
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
      if (step.kind != detail::PlanStepKind::kOperation &&
          step.kind != detail::PlanStepKind::kRegion) {
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

  if (!plan.segment_dependencies.try_resize(edges.size()))
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
                  detail::kConstant)) != 0;
}

void reset_workspace_decisions(detail::PlanState &plan) noexcept {
  plan.workspace_plan = {};
  plan.workspace_bytes = 0;
  plan.workspace_backing_count = 0;
  plan.supports_concurrent_invocations = true;
  plan.serialization_edge_count = plan.serialization_edges.size();
  for (std::size_t index = 0; index < plan.alias_sources.size(); ++index) {
    plan.alias_sources[index] = detail::kNoInPlaceSource;
    plan.alias_offsets[index] = 0;
    plan.instance_roles[index] &= static_cast<std::uint8_t>(
        ~(detail::kInvocationOwnedOutput | detail::kRetainedOutputRoot));
  }
  for (std::uint8_t &role : plan.binding_roles) {
    role &= static_cast<std::uint8_t>(
        ~(detail::kInvocationOwnedOutput | detail::kSourceAliasedOutput));
  }
  for (std::uint32_t &request : plan.workspace_requests)
    request = detail::kNoWorkspaceRequest;
  plan.step_workspace_requests.clear();
  plan.storage_workspace_slots.clear();
  plan.workspace_resources.clear();
  plan.workspace_storage.clear();
}

Status plan_workspace(detail::PlanState &plan, const detail::GraphState &graph,
                      bool &reschedule_requested) noexcept {
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
    }
    for (std::size_t input_index = 0; input_index < step.input_count;
         ++input_index) {
      const std::uint32_t input =
          plan.node_inputs[step.input_begin + input_index];
      last_uses[input] = (std::max)(last_uses[input], terminal);
    }
  }

  // The schema supplies only a broad permission. A prepared provider narrows
  // it for the selected route and layout. Required view aliases preserve
  // storage without consuming the source; optional in-place aliases require
  // equal descriptions and the source's final read.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    // Region aliases are optional and are not selected until the provider can
    // report every affected boundary port. A region must therefore support
    // distinct boundary outputs even if it advertises optional aliases.
    if (step.kind != detail::PlanStepKind::kOperation)
      continue;
    const detail::NodeRecord &node = graph.nodes[step.node_begin];
    const compute::OperationTraits traits =
        compute::operation_traits(node.operation);
    const compute::AliasPolicy alias = traits.alias;
    if (alias == compute::AliasPolicy::kNone || node.input_count == 0)
      continue;
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t output =
          plan.node_outputs[step.output_begin + output_index];
      const std::uint32_t logical_output = plan.instance_values[output];
      if (alias == compute::AliasPolicy::kMustAliasInput0) {
        auto support = step.operation.alias_support(output_index, 0);
        if (!support || !support->supported) {
          return Status::Internal(
              "prepared provider omitted a required graph alias");
        }
        const std::uint32_t input = plan.node_inputs[step.input_begin];
        plan.alias_sources[output] = input;
        plan.alias_offsets[output] = support->byte_offset;
        if ((plan.instance_roles[output] & detail::kExternalOutput) != 0) {
          if (has_external_storage_root(plan, input)) {
            plan.binding_roles[logical_output] |= detail::kSourceAliasedOutput;
          } else {
            plan.binding_roles[logical_output] |=
                detail::kInvocationOwnedOutput;
            plan.instance_roles[output] |= detail::kInvocationOwnedOutput;
            plan.instance_roles[alias_root(plan, input)] |=
                detail::kRetainedOutputRoot;
          }
        }
        continue;
      }

      const bool output_is_external =
          (plan.instance_roles[output] & detail::kExternalOutput) != 0;
      const std::size_t candidate_count =
          alias == compute::AliasPolicy::kMayAliasAnyInput ? node.input_count
                                                           : std::size_t{1};
      for (std::size_t input_index = 0; input_index < candidate_count;
           ++input_index) {
        const std::uint32_t input =
            plan.node_inputs[step.input_begin + input_index];
        auto support = step.operation.alias_support(output_index, input_index);
        if (!support)
          return std::move(support).error();
        if (!support->supported || support->byte_offset != 0)
          continue;
        const bool input_is_external =
            (plan.instance_roles[input] &
             (detail::kExternalInput | detail::kConstant)) != 0;
        const bool internal_pair =
            !output_is_external && !input_is_external &&
            (plan.instance_roles[input] & detail::kExternalOutput) == 0 &&
            !has_external_storage_root(plan, input);
        const bool external_pair = output_is_external && input_is_external;
        if ((internal_pair || external_pair) &&
            last_uses[input] == step_index + 1 &&
            plan.instance_descriptions[input] ==
                plan.instance_descriptions[output]) {
          plan.alias_sources[output] = input;
          plan.alias_offsets[output] = 0;
          break;
        }
      }
    }
  }

  for (std::size_t ordinal = 0; ordinal < plan.instance_values.size();
       ++ordinal) {
    if (plan.value_materialized[ordinal] == 0 ||
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
    if (!requests.try_push_back(mem::LifetimeRequest{
            .resource_slot = plan.instance_storage[ordinal],
            .size = *size,
            // A derived view only promises its tensor's required alignment;
            // forcing the root's preferred 64-byte kernel alignment onto a
            // non-zero slice offset would reject otherwise valid aliases.
            .alignment = alias_request == mem::kNoLifetimeAlias
                             ? (std::max)(std::size_t{64},
                                          description.required_alignment())
                             : description.required_alignment(),
            .begin = begin,
            .end = end,
            .alias_of = alias_request,
            .alias_offset = plan.alias_offsets[ordinal],
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
    if ((plan.steps[step_index].kind == detail::PlanStepKind::kRegion ||
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
    std::uint32_t root = index;
    std::size_t depth = 0;
    while (requests[root].alias_of != mem::kNoLifetimeAlias) {
      root = requests[root].alias_of;
      if (root >= request_count || ++depth > request_count)
        return Status::Internal("graph workspace alias graph is invalid");
    }
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

  const std::size_t segment_count = plan.segments.size();
  const std::size_t lane_count = plan.lanes.size();
  constexpr std::size_t kMaximumVectorClockCells = 4u * 1024u * 1024u;
  const bool precise_happens_before =
      lane_count != 0 && segment_count <= kMaximumVectorClockCells / lane_count;
  detail::ValueOrdinals segment_sequence(plan.allocator);
  Words clocks(plan.allocator);
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
      selected_resources);
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
      return Status::ResourceExhausted(
          "graph workspace budget is exceeded by the existing order");
    }
    if (!precise_happens_before) {
      return Status::ResourceExhausted(
          "graph workspace serialization exceeds the bounded happens-before "
          "analysis");
    }

    auto segment_before = [&](std::uint32_t source,
                              std::uint32_t destination) noexcept {
      if (source == destination)
        return true;
      if (source >= segment_count || destination >= segment_count)
        return false;
      const std::uint32_t source_lane = plan.segments[source].lane.ordinal();
      const std::uint32_t destination_lane =
          plan.segments[destination].lane.ordinal();
      if (source_lane == destination_lane) {
        return segment_sequence[source] < segment_sequence[destination];
      }
      return segment_sequence[source] <=
             clocks[static_cast<std::size_t>(destination) * lane_count +
                    source_lane];
    };
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

Result<CompileReport> build_report(const detail::PlanState &plan,
                                   const detail::GraphState &graph) noexcept {
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
  if (!state->steps.try_resize(plan.steps.size()) ||
      !state->targets.try_resize(plan.targets.size()) ||
      !state->storage_domains.try_resize(plan.storage_domains.size()) ||
      !state->lanes.try_resize(plan.lanes.size()) ||
      !state->segments.try_resize(plan.segments.size()) ||
      !state->physical_values.try_resize(plan.instance_values.size()) ||
      !state->step_inputs.try_resize(plan.node_inputs.size()) ||
      !state->step_outputs.try_resize(plan.node_outputs.size()) ||
      !state->step_nodes.try_resize(plan.step_nodes.size()) ||
      !state->step_node_begins.try_resize(plan.steps.size()) ||
      !state->segment_dependencies.try_resize(
          plan.segment_dependencies.size())) {
    detail::release(state);
    return std::unexpected(plan_allocation_failure());
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
    const compute::Device device = plan.targets[index].context.device();
    state->targets[index] = CompileTargetReport{
        .target = ExecutionTargetId{static_cast<std::uint32_t>(index)},
        .device = device,
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
    const compute::Device device =
        plan.targets[lane.target.ordinal()].context.device();
    auto properties = device.properties();
    if (!properties) {
      detail::release(state);
      return std::unexpected(std::move(properties).error());
    }
    auto capture =
        plan.targets[lane.target.ordinal()].context.has_backend_extension(
            compute::detail::kCommandCaptureExtensionId);
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
      state->backend = device.backend_id();
    } else if (!state->mixed_backends &&
               state->backend != device.backend_id()) {
      state->backend = {};
      state->mixed_backends = true;
    }
  }
  state->workspace_bytes = plan.workspace_bytes;
  state->workspace_backing_count = plan.workspace_backing_count;
  state->specialization = plan.specialization;
  state->node_count = plan.logical_node_count;
  state->segment_count = plan.segments.size();
  state->segment_dependency_count = plan.segment_dependencies.size();
  state->serialization_edge_count = plan.serialization_edge_count;
  state->region_count = plan.region_count;
  state->constant_count = plan.constant_ids.size();
  state->supports_concurrent_invocations = plan.supports_concurrent_invocations;
  // PlanState conservatively keeps every compile-time binding because the
  // provider contract reports packed counts, not exact constant identities.
  state->borrowed_constant_count = plan.constant_ids.size();

  for (std::size_t index = 0; index < plan.instance_values.size(); ++index) {
    const std::uint32_t logical = plan.instance_values[index];
    const std::uint8_t roles = plan.instance_roles[index];
    CompilePhysicalValueReport physical{
        .value = detail::value_id(graph, logical),
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
      step_report.copy_bytes = step.copy_bytes;
      state->copy_bytes = saturating_add(state->copy_bytes, step.copy_bytes);
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
  return detail::CompileReportAccess::adopt(state);
}

} // namespace

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

std::size_t CompileReport::workspace_bytes() const noexcept {
  return state_ != nullptr ? state_->workspace_bytes : 0;
}

std::size_t CompileReport::workspace_backing_count() const noexcept {
  return state_ != nullptr ? state_->workspace_backing_count : 0;
}

std::size_t CompileReport::node_count() const noexcept {
  return state_ != nullptr ? state_->node_count : 0;
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
  const std::uint8_t role = state_->binding_roles[value->ordinal()];
  if ((role & detail::kInvocationOwnedOutput) != 0)
    return OutputOwnership::kInvocation;
  if ((role & (detail::kSourceAliasedOutput | detail::kExternalInput |
               detail::kConstant)) != 0) {
    return OutputOwnership::kSourceAlias;
  }
  return OutputOwnership::kCaller;
}

Result<Invocation> Executable::create_invocation() const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }

  const std::span<const mem::MemoryResourceRef> selected_resources(
      state_->workspace_resources.data(), state_->workspace_resources.size());
  auto workspace = state_->workspace_plan.instantiate(selected_resources);
  if (!workspace)
    return std::unexpected(std::move(workspace).error());

  void *storage = state_->allocator.allocate(sizeof(detail::InvocationState),
                                             alignof(detail::InvocationState));
  if (storage == nullptr) {
    return std::unexpected(invocation_allocation_failure());
  }
  auto *invocation =
      ::new (storage) detail::InvocationState(state_, std::move(*workspace));
  const std::size_t value_count = state_->instance_descriptions.size();
  if (!invocation->readable.try_resize(value_count) ||
      !invocation->writable.try_resize(value_count) ||
      !invocation->bound.try_resize(value_count) ||
      !invocation->invocations.try_reserve(state_->steps.size()) ||
      !invocation->submission_inputs.try_resize(state_->node_inputs.size()) ||
      !invocation->submission_outputs.try_resize(state_->node_outputs.size()) ||
      !invocation->queues.try_resize(state_->lanes.size()) ||
      !invocation->queue_seen.try_resize(state_->lanes.size()) ||
      !invocation->segment_completions.try_resize(state_->segments.size()) ||
      !invocation->captured_segments.try_resize(state_->segments.size()) ||
      !invocation->dependency_routes.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->candidate_dependency_routes.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->lane_endpoints.try_resize(state_->lanes.size()) ||
      !invocation->candidate_lane_endpoints.try_resize(state_->lanes.size()) ||
      !invocation->completion_dependencies.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->aggregate_segments.try_resize(state_->lanes.size()) ||
      !invocation->aggregate_children.try_resize(state_->lanes.size())) {
    detail::release(invocation);
    return std::unexpected(invocation_allocation_failure());
  }
  for (std::size_t ordinal = 0; ordinal < value_count; ++ordinal) {
    invocation->bound[ordinal] = 0;
    const std::uint32_t request = state_->workspace_requests[ordinal];
    if (request == detail::kNoWorkspaceRequest)
      continue;
    auto storage_view = invocation->workspace.view(request);
    if (!storage_view) {
      Status error = std::move(storage_view).error();
      detail::release(invocation);
      return std::unexpected(std::move(error));
    }
    auto tensor = compute::TensorView::create(
        state_->instance_descriptions[ordinal], *storage_view);
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
    invocation->queues[index] = nullptr;
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
        (step.kind == detail::PlanStepKind::kOperation ||
         step.kind == detail::PlanStepKind::kRegion)) {
      if (step.kind == detail::PlanStepKind::kRegion) {
        auto created = step.region.create_invocation(step_workspace);
        if (!created) {
          Status error = std::move(created).error();
          detail::release(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.region = std::move(*created);
      } else {
        auto created = step.operation.create_invocation(step_workspace);
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
    if (step.kind == detail::PlanStepKind::kRegion) {
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
  return detail::InvocationAccess::adopt(invocation);
}

Invocation::Invocation(Invocation &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Invocation &Invocation::operator=(Invocation &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Invocation::~Invocation() { detail::release(state_); }

Status Invocation::bind(ValueId value,
                        compute::ConstTensorView tensor) noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("invocation is invalid");
  }
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation bindings cannot change while work is in flight");
  }
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
  if (tensor.description() != plan.descriptions[ordinal]) {
    return Status::InvalidArgument(
        "input tensor does not match the executable description");
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
  if (instance == detail::kNoWorkspaceRequest)
    return Status::Internal("graph input has no external physical instance");
  invalidate_binding_cache(*state_);
  state_->readable[instance] = tensor;
  state_->bound[instance] = 1;
  return Status::Ok();
}

Status Invocation::bind(ValueId value, compute::TensorView tensor) noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("invocation is invalid");
  }
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation bindings cannot change while work is in flight");
  }
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
    if (tensor.description() != plan.descriptions[ordinal]) {
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
    invalidate_binding_cache(*state_);
    state_->writable[instance] = tensor;
    state_->readable[instance] = tensor.as_const();
    state_->bound[instance] = 1;
    return Status::Ok();
  }
  if ((plan.binding_roles[ordinal] & detail::kExternalInput) != 0) {
    return bind(value, tensor.as_const());
  }
  return Status::InvalidArgument("internal graph values cannot be bound");
}

bool Invocation::in_flight() const noexcept {
  return state_ != nullptr && state_->in_flight.load(std::memory_order_acquire);
}

Status Invocation::reset() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("invocation is invalid");
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation cannot reset while work is in flight");
  }
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
    if ((state_->plan->binding_roles[logical] & detail::kConstant) != 0)
      continue;
    const std::uint32_t ordinal = state_->plan->external_instances[logical];
    state_->readable[ordinal] = {};
    state_->writable[ordinal] = {};
    state_->bound[ordinal] = 0;
  }
  return Status::Ok();
}

Status Invocation::prepare_replay(std::span<const QueueBinding> queues,
                                  ReplayOptions options) noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("invocation is invalid");
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation replay cannot be prepared while work is in flight");
  }
  if (options.mode != ReplayMode::kCommands &&
      options.mode != ReplayMode::kPreferCapture &&
      options.mode != ReplayMode::kRequireCapture) {
    return Status::InvalidArgument("graph replay mode is invalid");
  }
  if (options.host_wait != HostWaitPolicy::kAllow &&
      options.host_wait != HostWaitPolicy::kReject) {
    return Status::InvalidArgument("graph host-wait policy is invalid");
  }
  if (queues.size() != state_->plan->required_queue_count) {
    return Status::InvalidArgument("replay preparation requires one queue for "
                                   "every active execution lane");
  }
  for (std::size_t index = 0; index < state_->queues.size(); ++index)
    state_->queue_seen[index] = 0;
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
    const ExecutionTargetId target = state_->plan->lanes[lane].target;
    if (!state_->plan->targets[target.ordinal()].context.owns(*binding.queue)) {
      return Status::InvalidArgument(
          "graph replay queue belongs to a different compute context");
    }
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
    compute::Queue *queue = candidate_queue(queues, lane_id);
    if (queue == nullptr)
      return Status::Internal("graph replay queue binding is unavailable");
    auto endpoint = compute::detail::QueueSequence::completion_endpoint(*queue);
    if (!endpoint) {
      return std::move(endpoint).error().WithContext(
          "querying graph Queue completion endpoint");
    }
    state_->candidate_lane_endpoints[lane] = *endpoint;
  }

  Status bindings = prepare_invocation_bindings(*state_);
  if (!bindings.ok())
    return std::move(bindings).WithContext("preparing graph replay bindings");
  ReplayReport transfer_report{};
  Status transfers = validate_transfer_routes(*state_, queues, transfer_report);
  if (!transfers.ok())
    return transfers;
  auto dependency_report =
      prepare_dependency_routes(*state_, queues, options.host_wait);
  if (!dependency_report)
    return std::move(dependency_report).error();
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
    bool seen = false;
    for (std::size_t earlier = 0; earlier < index; ++earlier)
      seen = seen || queues[earlier].queue == queues[index].queue;
    if (seen)
      continue;
    std::size_t count = 0;
    for (const QueueBinding binding : queues) {
      if (binding.queue != queues[index].queue)
        continue;
      const std::size_t lane_count =
          state_->plan->lane_segment_counts[binding.lane.ordinal()];
      if (lane_count > SIZE_MAX - count) {
        return Status::ResourceExhausted(
            "graph replay completion capacity overflows");
      }
      count += lane_count;
    }
    Status reserved = queues[index].queue->reserve_completions(count);
    if (!reserved.ok()) {
      return std::move(reserved).WithContext(
          "reserving graph replay completion capacity");
    }
  }

  // Do not disturb a previously usable replay route until every replacement
  // Queue, binding, transfer and cold completion capacity has been validated
  // successfully.
  for (const QueueBinding binding : queues)
    state_->queues[binding.lane.ordinal()] = binding.queue;
  std::copy(state_->candidate_dependency_routes.begin(),
            state_->candidate_dependency_routes.end(),
            state_->dependency_routes.begin());
  std::copy(state_->candidate_lane_endpoints.begin(),
            state_->candidate_lane_endpoints.end(),
            state_->lane_endpoints.begin());
  state_->transfers_prepared = true;
  state_->dependencies_prepared = true;
  state_->replay_report = *dependency_report;

  for (compute::detail::CapturedCommands &captured :
       state_->captured_segments) {
    captured.reset();
  }
  state_->replay_prepared = false;
  state_->replay_route = ReplayRoute::kUnprepared;
  state_->replay_report.route = ReplayRoute::kUnprepared;
  if (options.mode == ReplayMode::kCommands) {
    state_->replay_route = ReplayRoute::kCommands;
    state_->replay_report.route = ReplayRoute::kCommands;
    state_->replay_prepared = true;
    return Status::Ok();
  }

  auto clear_captures = [&]() noexcept {
    for (compute::detail::CapturedCommands &captured :
         state_->captured_segments) {
      captured.reset();
    }
    state_->replay_prepared = false;
    state_->replay_route = ReplayRoute::kUnprepared;
    state_->replay_report = {};
  };
  auto finish_preparation =
      [](compute::detail::QueueSequence &sequence) noexcept -> Status {
    auto completion = sequence.finish();
    if (!completion)
      return std::move(completion).error();
    return completion->wait();
  };

  std::size_t command_segment_count = 0;
  std::size_t captured_segment_count = 0;
  for (std::size_t segment_index = 0;
       segment_index < state_->plan->segments.size(); ++segment_index) {
    const detail::PlanSegment &segment = state_->plan->segments[segment_index];
    bool has_commands = false;
    const std::size_t step_end = segment.step_begin + segment.step_count;
    for (std::size_t step = segment.step_begin; step < step_end; ++step)
      has_commands = has_commands || state_->plan->steps[step].emits_command();
    if (!has_commands)
      continue;
    ++command_segment_count;
    if (!shared_lane(*state_->plan, segment.lane)) {
      if (options.mode == ReplayMode::kRequireCapture) {
        clear_captures();
        return Status::Unimplemented(
            "a delegated graph segment cannot use Queue command capture");
      }
      continue;
    }
    if (!segment_capture_compatible(*state_->plan, segment)) {
      if (options.mode == ReplayMode::kRequireCapture) {
        clear_captures();
        return Status::Unimplemented(
            "a graph segment contains a provider that is not capture-safe");
      }
      continue;
    }

    compute::Queue &queue = *state_->queues[segment.lane.ordinal()];
    auto extension = compute::detail::CommandCaptureAccess::query(queue);
    if (!extension) {
      Status unavailable = std::move(extension).error();
      if (options.mode == ReplayMode::kPreferCapture &&
          unavailable.IsUnimplemented())
        continue;
      clear_captures();
      return std::move(unavailable)
          .WithContext("querying graph command capture support");
    }
    auto sequence = compute::detail::QueueSequence::begin(queue);
    if (!sequence) {
      clear_captures();
      return std::move(sequence).error().WithContext(
          "beginning graph command capture sequence");
    }
    Status begun =
        compute::detail::CommandCaptureAccess::begin(*extension, *sequence);
    if (!begun.ok()) {
      // begin() rejected before any graph command was admitted. Retire the
      // reserved sequence slot, but preserve the causal capture error.
      finish_preparation(*sequence).IgnoreError();
      if (options.mode == ReplayMode::kPreferCapture && begun.IsUnimplemented())
        continue;
      clear_captures();
      return std::move(begun).WithContext("beginning graph command capture");
    }

    Status recorded = enqueue_segment_steps(*state_, segment, *sequence);
    if (!recorded.ok()) {
      Status aborted =
          compute::detail::CommandCaptureAccess::abort(*extension, *sequence);
      Status finished = finish_preparation(*sequence);
      (void)aborted;
      (void)finished;
      clear_captures();
      return std::move(recorded).WithContext(
          "recording a capture-compatible graph segment");
    }
    auto captured = compute::detail::CommandCaptureAccess::end(
        *extension, *sequence, state_->allocator);
    Status finished = finish_preparation(*sequence);
    if (!captured) {
      Status capture_error = std::move(captured).error();
      // end() owns capture termination even on failure. The empty preparation
      // prefix still has to retire, but cannot replace the causal error.
      finished.IgnoreError();
      if (options.mode == ReplayMode::kPreferCapture &&
          capture_error.IsUnimplemented()) {
        continue;
      }
      clear_captures();
      return std::move(capture_error)
          .WithContext("finalizing graph command capture");
    }
    if (!finished.ok()) {
      clear_captures();
      return std::move(finished).WithContext(
          "retiring graph command capture sequence");
    }
    state_->captured_segments[segment_index] = std::move(*captured);
    ++captured_segment_count;
  }

  if (captured_segment_count == 0) {
    state_->replay_route = ReplayRoute::kCommands;
  } else if (captured_segment_count == command_segment_count) {
    state_->replay_route = ReplayRoute::kCaptured;
  } else {
    state_->replay_route = ReplayRoute::kHybrid;
  }
  state_->replay_report.route = state_->replay_route;
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
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return std::unexpected(Status::FailedPrecondition(
        "graph outputs are unavailable while work is in flight"));
  }
  if (!state_->outputs_ready.load(std::memory_order_acquire)) {
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
  if (instance == detail::kNoWorkspaceRequest ||
      !state_->readable[instance].valid()) {
    return std::unexpected(
        Status::Internal("published graph output has no readable storage"));
  }
  return state_->readable[instance];
}

Result<compute::Completion> Invocation::submit(compute::Queue &queue) noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  if (state_->plan->required_queue_count != 1) {
    return std::unexpected(Status::InvalidArgument(
        "single-queue submit requires exactly one active execution lane"));
  }
  std::uint32_t active_lane = 0;
  while (state_->plan->lane_segment_counts[active_lane] == 0 ||
         !shared_lane(*state_->plan, ExecutionLaneId{active_lane}))
    ++active_lane;
  const std::array bindings{
      QueueBinding{.lane = ExecutionLaneId{active_lane}, .queue = &queue}};
  if (!state_->replay_prepared ||
      (state_->replay_route == ReplayRoute::kCommands &&
       state_->queues[active_lane] != &queue)) {
    Status prepared = prepare_replay(bindings);
    if (!prepared.ok())
      return std::unexpected(std::move(prepared));
  }
  return submit(bindings);
}

Result<compute::Completion> Invocation::submit() noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  if (state_->plan->required_queue_count != 0) {
    return std::unexpected(Status::InvalidArgument(
        "queue-free submit requires a graph with no caller Queue lanes"));
  }
  constexpr std::span<const QueueBinding> no_queues{};
  if (!state_->replay_prepared) {
    Status prepared = prepare_replay(no_queues);
    if (!prepared.ok())
      return std::unexpected(std::move(prepared));
  }
  return submit(no_queues);
}

Result<compute::Completion>
Invocation::submit(std::span<const QueueBinding> queues) noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  bool idle = false;
  if (!state_->in_flight.compare_exchange_strong(
          idle, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return std::unexpected(
        Status::FailedPrecondition("invocation already has work in flight"));
  }
  const detail::PlanState &plan = *state_->plan;
  bool exclusive = false;
  if (!plan.supports_concurrent_invocations &&
      !state_->plan->exclusive_in_flight.compare_exchange_strong(
          exclusive, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    state_->in_flight.store(false, std::memory_order_release);
    return std::unexpected(Status::FailedPrecondition(
        "a provider in this executable requires serialized invocations"));
  }
  state_->outputs_ready.store(false, std::memory_order_release);
  struct RunRollback final {
    detail::InvocationState *state = nullptr;
    bool plan_exclusive = false;
    bool armed = true;
    ~RunRollback() {
      if (armed) {
        if (plan_exclusive) {
          state->plan->exclusive_in_flight.store(false,
                                                 std::memory_order_release);
        }
        state->in_flight.store(false, std::memory_order_release);
      }
    }
    void disarm() noexcept { armed = false; }
  } rollback{state_, !plan.supports_concurrent_invocations};

  if (!state_->replay_prepared) {
    return std::unexpected(Status::FailedPrecondition(
        "multi-domain invocation requires prepare_replay()"));
  }
  if (queues.size() != plan.required_queue_count) {
    return std::unexpected(Status::InvalidArgument(
        "submit requires one queue for every active execution domain"));
  }
  for (std::size_t index = 0; index < state_->queue_seen.size(); ++index)
    state_->queue_seen[index] = 0;
  for (const QueueBinding binding : queues) {
    if (!binding.lane.valid() || binding.lane.ordinal() >= plan.lanes.size() ||
        binding.queue == nullptr) {
      return std::unexpected(
          Status::InvalidArgument("graph submit queue binding is invalid"));
    }
    const std::size_t lane = binding.lane.ordinal();
    if (plan.lane_segment_counts[lane] == 0 ||
        !shared_lane(plan, binding.lane) || state_->queue_seen[lane] != 0 ||
        state_->queues[lane] != binding.queue) {
      return std::unexpected(Status::InvalidArgument(
          "graph submit queues differ from prepared replay queues"));
    }
    state_->queue_seen[lane] = 1;
  }
  for (std::size_t lane = 0; lane < state_->queue_seen.size(); ++lane) {
    if (plan.lane_segment_counts[lane] != 0 &&
        shared_lane(plan, ExecutionLaneId{static_cast<std::uint32_t>(lane)}) &&
        state_->queue_seen[lane] == 0) {
      return std::unexpected(
          Status::InvalidArgument("graph submit queue binding is missing"));
    }
  }
  if (!state_->bindings_prepared || !state_->transfers_prepared ||
      !state_->dependencies_prepared) {
    return std::unexpected(Status::Internal(
        "prepared graph replay lost a binding, transfer, or dependency route"));
  }

  compute::Completion final_completion;
  Status tail_error = Status::Ok();
  for (compute::Completion &completion : state_->segment_completions)
    completion = {};
  for (std::size_t segment_index = 0; segment_index < plan.segments.size();
       ++segment_index) {
    const detail::PlanSegment &segment = plan.segments[segment_index];
    const bool is_shared = shared_lane(plan, segment.lane);
    compute::Queue *queue =
        is_shared ? state_->queues[segment.lane.ordinal()] : nullptr;
    bool continue_segment = true;
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;

    // A host fallback is a cold-selected, public replay fact. Complete it
    // before taking the destination Queue's single-producer admission so a
    // slow upstream device cannot monopolize an otherwise idle Queue.
    for (std::size_t dependency_index = segment.dependency_begin;
         dependency_index < dependency_end; ++dependency_index) {
      const std::uint32_t source_segment =
          plan.segment_dependencies[dependency_index];
      if (source_segment >= segment_index ||
          !state_->segment_completions[source_segment].valid()) {
        tail_error =
            Status::Internal("graph segment dependency is not available");
        continue_segment = false;
        break;
      }
      compute::Completion &source = state_->segment_completions[source_segment];
      const ReplayDependencyRoute route =
          state_->dependency_routes[dependency_index];
      if (route == ReplayDependencyRoute::kOrderedQueue ||
          route == ReplayDependencyRoute::kSynchronousSource)
        continue;
      if (route == ReplayDependencyRoute::kHostWait) {
        Status host = source.wait();
        if (!host.ok()) {
          tail_error = std::move(host);
          continue_segment = false;
          break;
        }
      } else if (route != ReplayDependencyRoute::kDeviceWait) {
        tail_error =
            Status::Internal("prepared graph dependency route is invalid");
        continue_segment = false;
        break;
      }
    }
    if (!continue_segment)
      break;

    if (is_shared) {
      if (queue == nullptr) {
        tail_error =
            Status::Internal("prepared graph shared lane has no Queue binding");
        break;
      }
      auto sequence = compute::detail::QueueSequence::begin(*queue);
      if (!sequence) {
        tail_error = std::move(sequence).error();
        break;
      }
      for (std::size_t dependency_index = segment.dependency_begin;
           dependency_index < dependency_end; ++dependency_index) {
        if (state_->dependency_routes[dependency_index] ==
            ReplayDependencyRoute::kDeviceWait) {
          const std::uint32_t source_segment =
              plan.segment_dependencies[dependency_index];
          compute::Completion &source =
              state_->segment_completions[source_segment];
          compute::detail::EnqueueResult waited =
              sequence->enqueue_wait(source);
          const bool wait_succeeded = waited.status.ok();
          if (!wait_succeeded) {
            tail_error =
                waited.status.IsUnimplemented()
                    ? Status::Internal(
                          "prepared graph completion-wait route was rejected")
                    : waited.status;
          }
          sequence->record(std::move(waited));
          if (!wait_succeeded) {
            continue_segment = false;
            break;
          }
        }
      }

      if (continue_segment &&
          state_->captured_segments[segment_index].valid()) {
        compute::detail::EnqueueResult launched =
            state_->captured_segments[segment_index].enqueue(*sequence);
        continue_segment = launched.status.ok();
        if (!continue_segment)
          tail_error = launched.status;
        sequence->record(std::move(launched));
      } else if (continue_segment) {
        Status enqueued = enqueue_segment_steps(*state_, segment, *sequence);
        if (!enqueued.ok()) {
          tail_error = std::move(enqueued);
          continue_segment = false;
        }
      }
      auto completion = sequence->finish();
      if (!completion) {
        tail_error = std::move(completion).error();
        break;
      }
      state_->segment_completions[segment_index] = std::move(*completion);
    } else {
      std::size_t imported_count = 0;
      for (std::size_t dependency_index = segment.dependency_begin;
           dependency_index < dependency_end; ++dependency_index) {
        if (state_->dependency_routes[dependency_index] !=
            ReplayDependencyRoute::kDeviceWait) {
          continue;
        }
        const std::uint32_t source_segment =
            plan.segment_dependencies[dependency_index];
        compute::Completion &source =
            state_->segment_completions[source_segment];
        const compute::detail::CompletionDependency dependency =
            compute::detail::CompletionAccess::dependency(source);
        if (!dependency.pending()) {
          if (dependency.terminal_hint != nullptr &&
              !dependency.terminal_hint->ok()) {
            tail_error = *dependency.terminal_hint;
            continue_segment = false;
            break;
          }
          continue;
        }
        state_->completion_dependencies[segment.dependency_begin +
                                        imported_count] = dependency;
        ++imported_count;
      }
      if (continue_segment) {
        const auto dependencies =
            imported_count == 0
                ? std::span<const compute::detail::CompletionDependency>{}
                : std::span<const compute::detail::CompletionDependency>(
                      state_->completion_dependencies.data() +
                          segment.dependency_begin,
                      imported_count);
        auto completion =
            submit_delegated_segment(*state_, segment, dependencies);
        if (!completion) {
          tail_error = std::move(completion).error();
          continue_segment = false;
        } else {
          state_->segment_completions[segment_index] = std::move(*completion);
        }
      }
    }
    if (!continue_segment)
      break;
  }

  auto select_latest_queue_prefixes =
      [&](std::span<const std::uint32_t> candidates) noexcept {
        std::size_t count = 0;
        for (const std::uint32_t candidate : candidates) {
          const detail::PlanSegment &candidate_segment =
              plan.segments[candidate];
          if (!shared_lane(plan, candidate_segment.lane)) {
            // A delegated Completion represents this exact provider episode,
            // not a HYLE ordered-Queue prefix. Even equal endpoint identities
            // therefore remain independently observable terminal work.
            state_->aggregate_segments[count++] = candidate;
            continue;
          }
          const compute::Queue *queue =
              state_->queues[candidate_segment.lane.ordinal()];
          std::size_t match = count;
          for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t selected = state_->aggregate_segments[index];
            if (shared_lane(plan, plan.segments[selected].lane) &&
                state_->queues[plan.segments[selected].lane.ordinal()] ==
                    queue) {
              match = index;
              break;
            }
          }
          if (match == count) {
            state_->aggregate_segments[count++] = candidate;
          } else if (candidate > state_->aggregate_segments[match]) {
            state_->aggregate_segments[match] = candidate;
          }
        }
        return std::span<const std::uint32_t>(state_->aggregate_segments.data(),
                                              count);
      };

  auto assemble_completion = [&](std::span<const std::uint32_t> segments,
                                 const Status &terminal) noexcept -> Status {
    if (segments.empty())
      return terminal;
    for (std::uint32_t segment : segments) {
      if (segment >= state_->segment_completions.size() ||
          !state_->segment_completions[segment].valid()) {
        return Status::Internal(
            "graph completion segment is unavailable after submission");
      }
    }
    if (segments.size() == 1) {
      final_completion =
          std::move(state_->segment_completions[segments.front()]);
      compute::detail::CompletionAccess::append_terminal(final_completion,
                                                         terminal);
      return Status::Ok();
    }
    for (std::size_t index = 0; index < segments.size(); ++index)
      state_->aggregate_segments[index] = segments[index];
    state_->aggregate_completion.completion_count = segments.size();
    final_completion = make_aggregate_completion(state_->aggregate_completion);
    if (!terminal.ok()) {
      merge_aggregate_terminal(state_->aggregate_completion, terminal);
      compute::detail::CompletionAccess::append_terminal(final_completion,
                                                         terminal);
    }
    return Status::Ok();
  };

  if (!tail_error.ok()) {
    // Preserve the latest accepted ordered prefix from every participating
    // lane. A synchronous tail rejection therefore remains asynchronous
    // when earlier independent branches are still running; no hidden host
    // drain is needed on the error path.
    for (std::size_t lane = 0; lane < state_->queue_seen.size(); ++lane)
      state_->queue_seen[lane] = 0;
    for (std::size_t index = 0; index < state_->segment_completions.size();
         ++index) {
      if (!state_->segment_completions[index].valid())
        continue;
      const std::uint32_t lane = plan.segments[index].lane.ordinal();
      state_->aggregate_segments[lane] = static_cast<std::uint32_t>(index);
      state_->queue_seen[lane] = 1;
    }
    std::size_t count = 0;
    for (std::size_t lane = 0; lane < state_->queue_seen.size(); ++lane) {
      if (state_->queue_seen[lane] != 0)
        state_->aggregate_segments[count++] = state_->aggregate_segments[lane];
    }
    if (count == 0)
      return std::unexpected(std::move(tail_error));
    const auto selected =
        select_latest_queue_prefixes(std::span<const std::uint32_t>(
            state_->aggregate_segments.data(), count));
    const Status assembled = assemble_completion(selected, tail_error);
    if (!assembled.ok())
      return std::unexpected(assembled);
    tail_error = Status::Ok();
  } else {
    const auto selected =
        select_latest_queue_prefixes(std::span<const std::uint32_t>(
            plan.completion_segments.data(), plan.completion_segments.size()));
    Status assembled = assemble_completion(selected, Status::Ok());
    if (!assembled.ok())
      return std::unexpected(std::move(assembled));
  }
  if (!final_completion.valid())
    final_completion = compute::Completion::completed();

  if (!compute::detail::CompletionAccess::has_async_state(final_completion)) {
    mark_invocation_terminal(state_, final_completion.wait());
    rollback.disarm();
    return final_completion;
  }

  detail::retain(state_);
  state_->completion_lifetime = compute::detail::CompletionLifetime{
      .state = state_,
      .terminal = &mark_invocation_terminal,
      .release = &release_invocation_lifetime,
  };
  Status attached = compute::detail::CompletionAccess::attach_lifetime(
      final_completion, state_->completion_lifetime);
  if (!attached.ok()) {
    // This is an invariant/error path after work may have been accepted.
    // Drain before exposing mutable Invocation storage again.
    final_completion.wait().IgnoreError();
    return std::unexpected(std::move(attached));
  }
  rollback.disarm();
  return final_completion;
}

Result<CompiledGraph> compile(const Graph &graph,
                              const compute::Context &context,
                              CompileOptions options) noexcept {
  const std::array storage_domains{StorageDomain{
      .resource = mem::system_resource(),
  }};
  const std::array targets{ExecutionTarget{
      .context = &context,
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

Result<CompiledGraph> compile(const Graph &graph,
                              CompileEnvironment environment,
                              CompileOptions options) noexcept {
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
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (environment.storage_domains[previous].resource == storage.resource) {
        return std::unexpected(Status::InvalidArgument(
            "compile environment duplicates a storage resource"));
      }
    }
  }
  for (std::size_t index = 0; index < environment.targets.size(); ++index) {
    const ExecutionTarget &target = environment.targets[index];
    if (target.context == nullptr || !target.context->valid() ||
        !target.context->is_open()) {
      return std::unexpected(Status::FailedPrecondition(
          "compile environment context is invalid or closed"));
    }
    if (!target.default_storage.valid() ||
        target.default_storage.ordinal() >=
            environment.storage_domains.size()) {
      return std::unexpected(Status::InvalidArgument(
          "compile environment target default storage is invalid"));
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (environment.targets[previous].context == target.context) {
        return std::unexpected(Status::InvalidArgument(
            "compile environment duplicates an execution context"));
      }
    }
    auto lease = compute::detail::ContextRef::acquire(*target.context);
    if (!lease)
      return std::unexpected(std::move(lease).error());
    Status compatible = lease->validate_resource(
        environment.storage_domains[target.default_storage.ordinal()].resource);
    if (!compatible.ok())
      return std::unexpected(std::move(compatible));
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
  const compute::Context &context = *environment.targets[0].context;
  if (options.search_policy != CompileSearchPolicy::kAllowLimited &&
      options.search_policy != CompileSearchPolicy::kRequireUnexhausted) {
    return std::unexpected(
        Status::InvalidArgument("graph compile search policy is invalid"));
  }
  if (!context.valid() || !context.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("compute context is invalid or closed"));
  }
  if (options.region_selection != RegionSelectionPolicy::kDisabled &&
      options.region_selection !=
          RegionSelectionPolicy::kPreferProviderRegions) {
    return std::unexpected(
        Status::InvalidArgument("graph region selection policy is invalid"));
  }
  if (options.region_selection != RegionSelectionPolicy::kDisabled &&
      (options.max_region_proposals == 0 ||
       options.max_region_proposals_per_provider == 0 ||
       options.max_region_proposal_nodes == 0)) {
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

  const mem::AllocatorRef allocator = context.metadata_allocator();
  if (!allocator.valid()) {
    return std::unexpected(Status::FailedPrecondition(
        "compute context has no metadata allocator"));
  }
  ProviderPolicy provider_policy(allocator);
  HYLE_RETURN_IF_ERROR(provider_policy.initialize(*logical, options));
  detail::TensorDescriptions descriptions(allocator);
  auto signature =
      specialize_descriptions(*logical, options.shape, allocator, descriptions);
  if (!signature)
    return std::unexpected(std::move(signature).error());
  auto specialized =
      make_specialized_graph(*logical,
                             std::span<const compute::TensorDesc>(
                                 descriptions.data(), descriptions.size()),
                             allocator);
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
  plan->specialization = *signature;
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
        })) {
      detail::release(plan);
      return std::unexpected(plan_allocation_failure());
    }
  }
  for (const ExecutionTarget &target : environment.targets) {
    auto lease = compute::detail::ContextRef::acquire(*target.context);
    if (!lease) {
      Status error = std::move(lease).error();
      detail::release(plan);
      return std::unexpected(std::move(error));
    }
    if (!plan->targets.try_push_back(detail::ExecutionTargetRecord{
            .context = std::move(*lease),
            .default_storage = target.default_storage,
        })) {
      detail::release(plan);
      return std::unexpected(plan_allocation_failure());
    }
  }
  if (environment.lanes.empty()) {
    for (std::size_t target = 0; target < environment.targets.size();
         ++target) {
      if (!plan->lanes.try_push_back(detail::ExecutionLaneRecord{
              .target = ExecutionTargetId{static_cast<std::uint32_t>(target)},
              .role = ExecutionLaneRole::kCompute,
          })) {
        detail::release(plan);
        return std::unexpected(plan_allocation_failure());
      }
    }
  } else {
    for (const ExecutionLane &lane : environment.lanes) {
      if (!plan->lanes.try_push_back(detail::ExecutionLaneRecord{
              .target = lane.target,
              .role = lane.role,
          })) {
        detail::release(plan);
        return std::unexpected(plan_allocation_failure());
      }
    }
  }

  Status status =
      initialize_plan_metadata(*plan, *concrete,
                               std::span<const compute::TensorDesc>(
                                   descriptions.data(), descriptions.size()));
  if (status.ok())
    status = bind_constants(*plan, *concrete, options);
  if (status.ok())
    status = bind_effect_resources(*plan, *concrete, options);
  if (status.ok())
    status = prepare_placed_steps(*plan, *specialized, *concrete, environment,
                                  options, provider_policy);
  if (status.ok())
    status = schedule_steps(*plan, *concrete);
  if (status.ok())
    status = build_segments(*plan, *concrete);
  if (status.ok()) {
    if (plan->instance_values.size() > SIZE_MAX - plan->steps.size()) {
      status = Status::ResourceExhausted(
          "graph workspace serialization round budget overflows");
    } else {
      const std::size_t maximum_serialization_rounds =
          plan->instance_values.size() + plan->steps.size();
      std::size_t serialization_round = 0;
      while (status.ok()) {
        reset_workspace_decisions(*plan);
        bool reschedule_requested = false;
        status = plan_workspace(*plan, *concrete, reschedule_requested);
        if (!status.ok() || !reschedule_requested)
          break;
        if (serialization_round++ >= maximum_serialization_rounds) {
          status = Status::ResourceExhausted(
              "graph workspace serialization exhausted its deterministic "
              "round budget");
          break;
        }
        status = schedule_steps(*plan, *concrete);
        if (status.ok())
          status = build_segments(*plan, *concrete);
      }
    }
  }
  if (!status.ok()) {
    detail::release(plan);
    return std::unexpected(std::move(status));
  }
  auto report = build_report(*plan, *concrete);
  if (!report) {
    detail::release(plan);
    return std::unexpected(std::move(report).error());
  }

  CompiledGraph result;
  result.report = std::move(*report);
  result.executable = detail::ExecutableAccess::adopt(plan);
  return result;
}

} // namespace hyle::graph
