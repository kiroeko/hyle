#include "hyle/graph/executable.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <utility>

#include "../compute/completion_internal.h"
#include "graph_state.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/transfer.h"
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
  if (bindings.size() != graph.dimensions.size()) {
    return std::unexpected(Status::InvalidArgument(
        "compile requires exactly one binding for every graph dimension"));
  }
  for (std::size_t index = 0; index < seen.size(); ++index)
    seen[index] = 0;
  for (const ShapeBinding binding : bindings) {
    if (!detail::owns(graph, binding.dimension)) {
      return std::unexpected(Status::InvalidArgument(
          "shape binding does not belong to this graph"));
    }
    const std::size_t ordinal = binding.dimension.ordinal();
    if (seen[ordinal] != 0) {
      return std::unexpected(
          Status::InvalidArgument("shape binding is duplicated"));
    }
    const DimensionBounds bounds = graph.dimensions[ordinal].bounds;
    if (binding.value < bounds.minimum || binding.value > bounds.maximum ||
        binding.value % bounds.multiple_of != 0) {
      return std::unexpected(Status::OutOfRange(
          "shape binding violates its bounded dimension contract"));
    }
    values[ordinal] = binding.value;
    seen[ordinal] = 1;
  }

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
      !builder.values.try_resize(source.values.size()) ||
      !builder.nodes.try_resize(source.nodes.size()) ||
      !builder.edges.try_resize(source.edges.size()) ||
      !builder.inputs.try_resize(source.inputs.size()) ||
      !builder.constants.try_resize(source.constants.size()) ||
      !builder.outputs.try_resize(source.outputs.size()) ||
      !builder.dependencies.try_resize(source.dependencies.size())) {
    return std::unexpected(plan_allocation_failure());
  }
  for (std::size_t index = 0; index < source.dimensions.size(); ++index)
    builder.dimensions[index] = source.dimensions[index];
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
                                ExecutionDomainId domain) noexcept {
  for (const QueueBinding binding : queues) {
    if (binding.domain == domain)
      return binding.queue;
  }
  return nullptr;
}

Status validate_transfer_routes(detail::InvocationState &invocation,
                                std::span<const QueueBinding> queues,
                                ReplayReport &report) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  for (const detail::PlanStep &step : plan.steps) {
    if (step.kind != detail::PlanStepKind::kCopy)
      continue;
    compute::Queue *queue = candidate_queue(queues, step.domain);
    if (!step.domain.valid() || queue == nullptr || step.input_count != 1 ||
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
    compute::Queue *destination = candidate_queue(queues, segment.domain);
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;
    for (std::size_t dependency_index = segment.dependency_begin;
         dependency_index < dependency_end; ++dependency_index) {
      const std::uint32_t source_segment =
          plan.segment_dependencies[dependency_index];
      if (source_segment >= segment_index || destination == nullptr) {
        return std::unexpected(
            Status::Internal("compiled graph dependency is invalid"));
      }
      compute::Queue *source =
          candidate_queue(queues, plan.segments[source_segment].domain);
      if (source == nullptr) {
        return std::unexpected(
            Status::Internal("compiled graph dependency source is invalid"));
      }

      ReplayDependencyRoute route = ReplayDependencyRoute::kOrderedQueue;
      if (source == destination) {
        ++report.ordered_dependency_count;
      } else {
        auto selected = compute::detail::QueueSequence::dependency_route(
            *destination, *source);
        if (!selected) {
          return std::unexpected(std::move(selected).error().WithContext(
              "selecting graph Queue dependency route"));
        }
        switch (*selected) {
        case compute::detail::QueueDependencyRoute::kNoWait:
          route = ReplayDependencyRoute::kSynchronousSource;
          ++report.synchronous_source_dependency_count;
          break;
        case compute::detail::QueueDependencyRoute::kDeviceWait:
          route = ReplayDependencyRoute::kDeviceWait;
          ++report.device_wait_dependency_count;
          break;
        case compute::detail::QueueDependencyRoute::kHostWait:
          if (host_wait_policy == HostWaitPolicy::kReject) {
            return std::unexpected(Status::Unimplemented(
                "graph Queue topology requires a rejected host wait"));
          }
          route = ReplayDependencyRoute::kHostWait;
          ++report.host_wait_dependency_count;
          break;
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
  const std::uint32_t domain = plan.external_domains[logical_ordinal];
  if (domain >= plan.domains.size())
    return Status::Internal("graph external domain is invalid");
  const mem::MemoryResourceRef expected =
      plan.external_resources[logical_ordinal];
  if (expected.valid() && expected != resource) {
    return Status::InvalidArgument(
        "tensor binding does not match its compiled memory resource");
  }
  Status compatible = plan.domains[domain].context.validate_resource(resource);
  if (!compatible.ok()) {
    return std::move(compatible)
        .WithContext("validating graph tensor binding resource");
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
      !plan.instance_domains.try_resize(graph.values.size()) ||
      !plan.instance_roles.try_resize(graph.values.size()) ||
      !plan.external_instances.try_resize(graph.values.size()) ||
      !plan.external_domains.try_resize(graph.values.size()) ||
      !plan.external_resources.try_resize(graph.values.size()) ||
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
    plan.instance_domains[index] = 0;
    plan.instance_roles[index] = 0;
    plan.external_instances[index] = detail::kNoWorkspaceRequest;
    plan.external_domains[index] = 0;
    plan.external_resources[index] = {};
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

struct RegionExtension final {
  compute::detail::ProviderExtensionRef extension{};
  std::uint32_t provider_index = 0;
  ExecutionDomainId domain{};
};

struct RegionCandidate final {
  RegionProposal proposal{};
  std::uint32_t extension_index = 0;
  std::uint32_t provider_index = 0;
  std::uint32_t proposal_order = 0;
  ExecutionDomainId domain{};
};

using RegionExtensions =
    container::SmallVector<RegionExtension, 4, mem::AllocatorRef>;
using RegionCandidates =
    container::SmallVector<RegionCandidate, 16, mem::AllocatorRef>;
using RegionValues = container::SmallVector<ValueId, 8, mem::AllocatorRef>;

Status collect_region_boundary(
    const detail::PlanState &plan, const detail::GraphState &graph,
    const RegionProposal &proposal,
    std::span<const std::uint32_t> last_consumers,
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> &marks,
    std::uint32_t marker, RegionValues &inputs,
    RegionValues &outputs) noexcept {
  const std::uint32_t begin = proposal.nodes.begin;
  const std::uint32_t end =
      begin + static_cast<std::uint32_t>(proposal.nodes.count);
  for (std::uint32_t node_index = begin; node_index < end; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t ordinal = graph.edges[node.input_begin + input_index];
      const std::uint32_t producer = graph.values[ordinal].producer;
      if (producer != detail::kNoNode && producer >= begin && producer < end)
        continue;
      if (marks[ordinal] == marker)
        continue;
      marks[ordinal] = marker;
      if (!inputs.try_push_back(detail::value_id(graph, ordinal)))
        return plan_allocation_failure();
    }
  }
  for (std::uint32_t node_index = begin; node_index < end; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t ordinal =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      const bool published =
          (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0;
      const bool consumed_after = last_consumers[ordinal] != detail::kNoNode &&
                                  last_consumers[ordinal] >= end;
      if ((published || consumed_after) &&
          !outputs.try_push_back(detail::value_id(graph, ordinal))) {
        return plan_allocation_failure();
      }
    }
  }
  return Status::Ok();
}

Result<detail::PreparedRegion> compile_region_artifact(
    detail::PlanState &plan, const Graph &published_graph,
    const RegionExtension &extension, const RegionProposal &proposal,
    std::span<const ValueId> inputs, std::span<const ValueId> outputs,
    CompileOptions options) noexcept {
  RegionCompileRequest request{
      .graph = &published_graph,
      .nodes = proposal.nodes,
      .inputs = inputs,
      .outputs = outputs,
      .constants = options.constants,
      .route = proposal.route,
  };
  ProviderRegionPrepared provider_prepared{};
  struct Episode final {
    const RegionCompileRequest *request = nullptr;
    mem::AllocatorRef allocator{};
    ProviderRegionPrepared *prepared = nullptr;
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
        return interface->compile(provider_context, *call.request,
                                  call.allocator, call.prepared);
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
  const compute::WorkspaceRequirement workspace =
      provider_prepared.properties.workspace;
  if (table == nullptr || table->destroy_prepared == nullptr ||
      table->alias_support == nullptr || table->create_invocation == nullptr ||
      table->destroy_invocation == nullptr || table->submit == nullptr ||
      workspace.alignment == 0 ||
      (workspace.alignment & (workspace.alignment - 1)) != 0 ||
      provider_prepared.properties.dispatch_count == 0 ||
      provider_prepared.properties.packed_constant_count >
          options.constants.size()) {
    if (table != nullptr && table->destroy_prepared != nullptr)
      table->destroy_prepared(provider_prepared.state, plan.allocator);
    return std::unexpected(Status::Internal(
        "graph region provider returned an invalid compiled artifact"));
  }
  return detail::PreparedRegion(extension.extension, provider_prepared,
                                plan.allocator);
}

Status discover_region_candidates(
    const Graph &published_graph, const detail::GraphState &graph,
    const compute::Context &context, CompileOptions options,
    mem::AllocatorRef allocator, RegionExtensions &extensions,
    RegionCandidates &candidates, ExecutionDomainId domain,
    std::span<const std::uint32_t> node_domains = {}) noexcept;

Status select_regions(RegionCandidates &candidates, mem::AllocatorRef allocator,
                      RegionSelectionPolicy policy,
                      RegionCandidates &selected) noexcept;

Status prepare_multi_domain_steps(detail::PlanState &plan,
                                  const Graph &published_graph,
                                  const detail::GraphState &graph,
                                  CompileEnvironment environment,
                                  CompileOptions options) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Flags = container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;
  using Prepared =
      container::SmallVector<compute::PreparedOperation, 8, mem::AllocatorRef>;
  constexpr std::uint32_t kNone = detail::kNoWorkspaceRequest;
  const std::size_t domain_count = plan.domains.size();
  const std::size_t value_count = graph.values.size();
  const std::size_t node_count = graph.nodes.size();
  if (options.max_placement_attempts == 0)
    return Status::InvalidArgument("graph placement budget is zero");
  if (value_count != 0 && domain_count > SIZE_MAX / value_count) {
    return Status::ResourceExhausted(
        "graph physical-instance matrix size overflows");
  }
  if (node_count != 0 && domain_count > SIZE_MAX / node_count) {
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
    plan.external_domains[index] = 0;
    plan.external_resources[index] = {};
  }

  for (const NodePlacement placement : options.node_placements) {
    if (!detail::owns(graph, placement.node) || !placement.domain.valid() ||
        placement.domain.ordinal() >= domain_count ||
        !valid_strength(placement.strength)) {
      return Status::InvalidArgument("graph node placement is invalid");
    }
    std::uint32_t &slot = placement.strength == PlacementStrength::kRequired
                              ? node_required[placement.node.ordinal()]
                              : node_preferred[placement.node.ordinal()];
    if (slot != kNone)
      return Status::InvalidArgument("graph node placement is duplicated");
    slot = placement.domain.ordinal();
  }
  for (const ValuePlacement placement : options.value_placements) {
    if (!detail::owns(graph, placement.value) || !placement.domain.valid() ||
        placement.domain.ordinal() >= domain_count ||
        !valid_strength(placement.strength)) {
      return Status::InvalidArgument("graph value placement is invalid");
    }
    std::uint32_t &slot = placement.strength == PlacementStrength::kRequired
                              ? value_required[placement.value.ordinal()]
                              : value_preferred[placement.value.ordinal()];
    if (slot != kNone)
      return Status::InvalidArgument("graph value placement is duplicated");
    slot = placement.domain.ordinal();
  }
  for (const ExternalResourceContract contract : options.external_resources) {
    if (!detail::owns(graph, contract.value) || !contract.domain.valid() ||
        contract.domain.ordinal() >= domain_count ||
        !contract.resource.valid()) {
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
    Status compatible =
        plan.domains[contract.domain.ordinal()].context.validate_resource(
            contract.resource);
    if (!compatible.ok())
      return std::move(compatible)
          .WithContext("validating graph external resource contract");
    external_seen[value] = 1;
    plan.external_domains[value] = contract.domain.ordinal();
    plan.external_resources[value] = contract.resource;
  }

  // A required placement on an external source supplies its locality when no
  // stronger exact resource contract was given.
  for (std::uint32_t value : graph.inputs) {
    if (external_seen[value] == 0) {
      if (value_required[value] != kNone)
        plan.external_domains[value] = value_required[value];
      else if (value_preferred[value] != kNone)
        plan.external_domains[value] = value_preferred[value];
    }
  }
  for (std::uint32_t value : graph.constants) {
    if (external_seen[value] == 0) {
      if (value_required[value] != kNone)
        plan.external_domains[value] = value_required[value];
      else if (value_preferred[value] != kNone)
        plan.external_domains[value] = value_preferred[value];
    }
  }

  // Compile-time constants already expose their exact resource. Validate it
  // against explicit contracts and the selected execution domain now.
  for (std::size_t index = 0; index < graph.constants.size(); ++index) {
    const std::uint32_t value = graph.constants[index];
    const mem::MemoryResourceRef actual =
        plan.constant_bindings[index].storage().resource();
    if (plan.external_resources[value].valid() &&
        plan.external_resources[value] != actual) {
      return Status::InvalidArgument(
          "constant binding does not match its external resource contract");
    }
    Status compatible =
        plan.domains[plan.external_domains[value]].context.validate_resource(
            actual);
    if (!compatible.ok())
      return std::move(compatible)
          .WithContext("validating graph constant placement");
    if (!plan.external_resources[value].valid())
      plan.external_resources[value] = actual;
  }

  RegionExtensions extensions(plan.allocator);
  RegionCandidates all_region_candidates(plan.allocator);
  for (std::size_t domain = 0; domain < domain_count; ++domain) {
    Status discovered = discover_region_candidates(
        published_graph, graph, *environment.domains[domain].context, options,
        plan.allocator, extensions, all_region_candidates,
        ExecutionDomainId{static_cast<std::uint32_t>(domain)});
    if (!discovered.ok())
      return discovered;
  }
  Flags region_support(plan.allocator);
  if (!region_support.try_resize(node_count * domain_count))
    return plan_allocation_failure();
  for (std::uint8_t &supported : region_support)
    supported = 0;
  for (const RegionCandidate &candidate : all_region_candidates) {
    for (std::uint32_t node = candidate.proposal.nodes.begin;
         node < candidate.proposal.nodes.end(); ++node) {
      region_support[static_cast<std::size_t>(node) * domain_count +
                     candidate.domain.ordinal()] = 1;
    }
  }

  Ordinals producer_domains(plan.allocator);
  Ordinals node_domains(plan.allocator);
  Ordinals domain_candidates(plan.allocator);
  Prepared prepared(plan.allocator);
  if (!producer_domains.try_resize(value_count) ||
      !node_domains.try_resize(node_count) ||
      !domain_candidates.try_resize(domain_count) ||
      !prepared.try_resize(node_count)) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < value_count; ++index)
    producer_domains[index] = kNone;
  for (std::uint32_t value : graph.inputs)
    producer_domains[value] = plan.external_domains[value];
  for (std::uint32_t value : graph.constants)
    producer_domains[value] = plan.external_domains[value];

  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions inputs(plan.allocator);
  Descriptions outputs(plan.allocator);
  std::size_t placement_attempts = 0;
  for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    std::uint32_t required = node_required[node_index];
    std::uint32_t preferred = node_preferred[node_index];
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t value =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      if (value_required[value] != kNone) {
        if (required != kNone && required != value_required[value]) {
          return Status::InvalidArgument(
              "required node and output placements conflict");
        }
        required = value_required[value];
      }
      if (preferred == kNone && value_preferred[value] != kNone)
        preferred = value_preferred[value];
    }
    for (std::size_t index = 0; index < domain_count; ++index)
      domain_candidates[index] = static_cast<std::uint32_t>(index);
    std::stable_sort(
        domain_candidates.begin(), domain_candidates.end(),
        [&](std::uint32_t left, std::uint32_t right) noexcept {
          if (required != kNone)
            return left == required && right != required;
          const bool left_preferred = left == preferred;
          const bool right_preferred = right == preferred;
          if (left_preferred != right_preferred)
            return left_preferred;
          const bool left_region =
              region_support[node_index * domain_count + left] != 0;
          const bool right_region =
              region_support[node_index * domain_count + right] != 0;
          if (left_region != right_region)
            return left_region;
          std::size_t left_local = 0;
          std::size_t right_local = 0;
          for (std::size_t input_index = 0; input_index < node.input_count;
               ++input_index) {
            const std::uint32_t value =
                graph.edges[node.input_begin + input_index];
            left_local += producer_domains[value] == left ? 1u : 0u;
            right_local += producer_domains[value] == right ? 1u : 0u;
          }
          if (left_local != right_local)
            return left_local > right_local;
          return left < right;
        });
    if (!inputs.try_resize(node.input_count) ||
        !outputs.try_resize(node.output_count)) {
      return plan_allocation_failure();
    }
    for (std::size_t index = 0; index < node.input_count; ++index)
      inputs[index] = plan.descriptions[graph.edges[node.input_begin + index]];
    for (std::size_t index = 0; index < node.output_count; ++index)
      outputs[index] = plan.descriptions[node.output_begin + index];

    Status unsupported = Status::Unimplemented(
        "no execution domain supports the graph operation");
    bool selected = false;
    for (std::uint32_t candidate : domain_candidates) {
      if (required != kNone && candidate != required)
        continue;
      if (region_support[node_index * domain_count + candidate] != 0) {
        node_domains[node_index] = candidate;
        for (std::size_t output_index = 0; output_index < node.output_count;
             ++output_index) {
          producer_domains[node.output_begin + output_index] = candidate;
        }
        selected = true;
        break;
      }
      if (placement_attempts == options.max_placement_attempts) {
        return Status::ResourceExhausted(
            "graph placement preparation budget is exhausted");
      }
      ++placement_attempts;
      auto result = compute::prepare(*environment.domains[candidate].context,
                                     node.operation, inputs, outputs);
      if (result) {
        prepared[node_index] = std::move(*result);
        node_domains[node_index] = candidate;
        for (std::size_t output_index = 0; output_index < node.output_count;
             ++output_index) {
          producer_domains[node.output_begin + output_index] = candidate;
        }
        selected = true;
        break;
      }
      unsupported = std::move(result).error();
      if (!unsupported.IsUnimplemented())
        return std::move(unsupported)
            .WithContext("preparing graph operation placement candidate");
    }
    if (!selected)
      return unsupported;
  }

  RegionCandidates candidates(plan.allocator);
  RegionCandidates selected_regions(plan.allocator);
  if (!candidates.try_reserve(all_region_candidates.size()))
    return plan_allocation_failure();
  for (const RegionCandidate &candidate : all_region_candidates) {
    bool eligible = true;
    for (std::uint32_t node = candidate.proposal.nodes.begin;
         node < candidate.proposal.nodes.end(); ++node) {
      if (node_domains[node] != candidate.domain.ordinal()) {
        eligible = false;
        break;
      }
    }
    if (eligible && !candidates.try_push_back(candidate))
      return plan_allocation_failure();
  }
  Status selected = select_regions(candidates, plan.allocator,
                                   options.region_selection, selected_regions);
  if (!selected.ok())
    return selected;
  if (options.operation_fallback == OperationFallbackPolicy::kReject) {
    std::size_t covered_nodes = 0;
    for (const RegionCandidate &candidate : selected_regions) {
      covered_nodes += candidate.proposal.nodes.count;
    }
    if (covered_nodes != node_count) {
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
    for (std::uint32_t node = candidate.proposal.nodes.begin;
         node < candidate.proposal.nodes.end(); ++node) {
      selected_nodes[node] = 1;
    }
  }
  for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
    if (selected_nodes[node_index] != 0 || prepared[node_index].valid())
      continue;
    const detail::NodeRecord &node = graph.nodes[node_index];
    std::uint32_t required = node_required[node_index];
    std::uint32_t preferred = node_preferred[node_index];
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t value =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      if (value_required[value] != kNone)
        required = value_required[value];
      if (preferred == kNone && value_preferred[value] != kNone)
        preferred = value_preferred[value];
    }
    const std::uint32_t originally_selected = node_domains[node_index];
    for (std::size_t index = 0; index < domain_count; ++index)
      domain_candidates[index] = static_cast<std::uint32_t>(index);
    std::stable_sort(domain_candidates.begin(), domain_candidates.end(),
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
                             producer_domains[value] == left ? 1u : 0u;
                         right_local +=
                             producer_domains[value] == right ? 1u : 0u;
                       }
                       if (left_local != right_local)
                         return left_local > right_local;
                       return left < right;
                     });
    if (!inputs.try_resize(node.input_count) ||
        !outputs.try_resize(node.output_count)) {
      return plan_allocation_failure();
    }
    for (std::size_t index = 0; index < node.input_count; ++index)
      inputs[index] = plan.descriptions[graph.edges[node.input_begin + index]];
    for (std::size_t index = 0; index < node.output_count; ++index)
      outputs[index] = plan.descriptions[node.output_begin + index];

    Status unsupported = Status::Unimplemented(
        "no execution domain supports an unselected graph operation");
    bool fallback_selected = false;
    for (std::uint32_t candidate : domain_candidates) {
      if (required != kNone && candidate != required)
        continue;
      if (placement_attempts == options.max_placement_attempts) {
        return Status::ResourceExhausted(
            "graph placement preparation budget is exhausted");
      }
      ++placement_attempts;
      auto fallback = compute::prepare(*environment.domains[candidate].context,
                                       node.operation, inputs, outputs);
      if (fallback) {
        prepared[node_index] = std::move(*fallback);
        node_domains[node_index] = candidate;
        for (std::size_t output_index = 0; output_index < node.output_count;
             ++output_index) {
          producer_domains[node.output_begin + output_index] = candidate;
        }
        fallback_selected = true;
        break;
      }
      unsupported = std::move(fallback).error();
      if (!unsupported.IsUnimplemented()) {
        return std::move(unsupported)
            .WithContext(
                "preparing operation left outside selected graph regions");
      }
    }
    if (!fallback_selected)
      return unsupported;
  }

  container::SmallVector<std::int64_t, 32, mem::AllocatorRef>
      eligible_region_coverage(plan.allocator);
  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> last_consumers(
      plan.allocator);
  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> marks(
      plan.allocator);
  if (!eligible_region_coverage.try_resize(node_count + 1) ||
      !last_consumers.try_resize(value_count) ||
      !marks.try_resize(value_count)) {
    return plan_allocation_failure();
  }
  for (std::int64_t &count : eligible_region_coverage)
    count = 0;
  for (const RegionCandidate &candidate : candidates) {
    ++eligible_region_coverage[candidate.proposal.nodes.begin];
    --eligible_region_coverage[static_cast<std::size_t>(
        candidate.proposal.nodes.end())];
  }
  for (std::size_t index = 1; index < eligible_region_coverage.size(); ++index)
    eligible_region_coverage[index] += eligible_region_coverage[index - 1];
  for (std::size_t index = 0; index < value_count; ++index) {
    last_consumers[index] = detail::kNoNode;
    marks[index] = 0;
  }
  for (std::uint32_t node_index = 0; node_index < node_count; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      last_consumers[graph.edges[node.input_begin + input_index]] = node_index;
    }
  }

  // With no explicit output resource, keep a published node result where its
  // producer already executes. This avoids manufacturing a boundary copy just
  // because domain zero appeared first in the environment.
  for (std::uint32_t value : graph.outputs) {
    if (external_seen[value] == 0 &&
        graph.values[value].producer != detail::kNoNode) {
      plan.external_domains[value] = producer_domains[value];
    }
  }

  const std::size_t maximum_instances = value_count * domain_count;
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
  plan.instance_descriptions.clear();
  plan.instance_values.clear();
  plan.instance_domains.clear();
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
      !plan.instance_descriptions.try_reserve(maximum_instances) ||
      !plan.instance_values.try_reserve(maximum_instances) ||
      !plan.instance_domains.try_reserve(maximum_instances) ||
      !plan.instance_roles.try_reserve(maximum_instances) ||
      !plan.workspace_requests.try_reserve(maximum_instances) ||
      !plan.value_materialized.try_reserve(maximum_instances) ||
      !plan.alias_sources.try_reserve(maximum_instances) ||
      !plan.alias_offsets.try_reserve(maximum_instances)) {
    return plan_allocation_failure();
  }

  auto add_instance =
      [&](std::uint32_t value, std::uint32_t domain,
          std::uint8_t roles) noexcept -> Result<std::uint32_t> {
    if (plan.instance_values.size() >= kNone)
      return std::unexpected(Status::ResourceExhausted(
          "graph physical-instance identity space is exhausted"));
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(plan.instance_values.size());
    if (!plan.instance_descriptions.try_push_back(plan.descriptions[value]) ||
        !plan.instance_values.try_push_back(value) ||
        !plan.instance_domains.try_push_back(domain) ||
        !plan.instance_roles.try_push_back(roles) ||
        !plan.workspace_requests.try_push_back(kNone) ||
        !plan.value_materialized.try_push_back(0) ||
        !plan.alias_sources.try_push_back(detail::kNoInPlaceSource) ||
        !plan.alias_offsets.try_push_back(0)) {
      return std::unexpected(plan_allocation_failure());
    }
    instances[static_cast<std::size_t>(value) * domain_count + domain] =
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
    auto instance_resource = [&](std::uint32_t instance) noexcept {
      const std::uint32_t logical = plan.instance_values[instance];
      const mem::MemoryResourceRef external = plan.external_resources[logical];
      if ((plan.instance_roles[instance] &
           (detail::kExternalInput | detail::kExternalOutput |
            detail::kConstant)) != 0 &&
          external.valid()) {
        return external;
      }
      return plan.domains[plan.instance_domains[instance]].workspace_resource;
    };
    const mem::MemoryResourceRef source_resource = instance_resource(source);
    const mem::MemoryResourceRef destination_resource =
        instance_resource(destination);
    const std::uint32_t source_domain = plan.instance_domains[source];
    const std::uint32_t destination_domain = plan.instance_domains[destination];
    const auto requires_exact_external_resource =
        [&](std::uint32_t instance) noexcept {
          const std::uint8_t roles = plan.instance_roles[instance];
          const std::uint32_t logical = plan.instance_values[instance];
          return (roles & (detail::kExternalInput | detail::kExternalOutput |
                           detail::kConstant)) != 0 &&
                 !plan.external_resources[logical].valid();
        };
    if (source_domain != destination_domain &&
        (requires_exact_external_resource(source) ||
         requires_exact_external_resource(destination))) {
      return Status::InvalidArgument(
          "a cross-domain external value requires an exact resource contract");
    }
    std::uint32_t execution_domain = kNone;
    auto supports_route = [&](std::uint32_t domain) noexcept {
      return plan.domains[domain]
                 .context.validate_resource(source_resource)
                 .ok() &&
             plan.domains[domain]
                 .context.validate_resource(destination_resource)
                 .ok();
    };
    if (supports_route(destination_domain)) {
      execution_domain = destination_domain;
    } else if (source_domain != destination_domain &&
               supports_route(source_domain)) {
      execution_domain = source_domain;
    } else {
      for (std::size_t index = 0; index < domain_count; ++index) {
        if (index != source_domain && index != destination_domain &&
            supports_route(static_cast<std::uint32_t>(index))) {
          execution_domain = static_cast<std::uint32_t>(index);
          break;
        }
      }
    }
    if (execution_domain == kNone) {
      return Status::Unimplemented(
          "no execution domain can transfer between the selected resources");
    }
    if (!plan.steps.try_push_back(detail::PlanStep::copy(
            input_begin, output_begin, *bytes, ExecutionDomainId{source_domain},
            ExecutionDomainId{destination_domain},
            ExecutionDomainId{execution_domain}))) {
      return plan_allocation_failure();
    }
    plan.value_materialized[destination] = 1;
    return Status::Ok();
  };

  auto ensure_local_instance =
      [&](std::uint32_t value,
          std::uint32_t domain) noexcept -> Result<std::uint32_t> {
    std::uint32_t instance =
        instances[static_cast<std::size_t>(value) * domain_count + domain];
    if (instance != kNone)
      return instance;
    const std::uint32_t source = producer_instances[value];
    if (source == kNone) {
      return std::unexpected(
          Status::Internal("graph value has no physical producer"));
    }
    auto created = add_instance(value, domain, 0);
    if (!created)
      return std::unexpected(std::move(created).error());
    Status copied = append_copy(source, *created);
    if (!copied.ok())
      return std::unexpected(std::move(copied));
    return *created;
  };

  for (std::uint32_t value : graph.inputs) {
    auto instance = add_instance(value, plan.external_domains[value],
                                 plan.binding_roles[value]);
    if (!instance)
      return std::move(instance).error();
    producer_instances[value] = *instance;
    plan.external_instances[value] = *instance;
  }
  for (std::uint32_t value : graph.constants) {
    auto instance = add_instance(value, plan.external_domains[value],
                                 plan.binding_roles[value]);
    if (!instance)
      return std::move(instance).error();
    producer_instances[value] = *instance;
    plan.external_instances[value] = *instance;
  }

  Ordinals physical_inputs(plan.allocator);
  std::size_t selection_index = 0;
  std::uint32_t node_index = 0;
  std::uint32_t marker = 1;
  while (node_index < node_count) {
    if (selection_index < selected_regions.size() &&
        selected_regions[selection_index].proposal.nodes.begin == node_index) {
      const RegionCandidate &candidate = selected_regions[selection_index];
      const std::uint32_t domain = candidate.domain.ordinal();
      RegionValues logical_inputs(plan.allocator);
      RegionValues logical_outputs(plan.allocator);
      Status boundary = collect_region_boundary(
          plan, graph, candidate.proposal, last_consumers, marks, marker++,
          logical_inputs, logical_outputs);
      if (!boundary.ok())
        return boundary;
      auto region = compile_region_artifact(
          plan, published_graph, extensions[candidate.extension_index],
          candidate.proposal, logical_inputs, logical_outputs, options);
      if (!region)
        return std::move(region).error();

      physical_inputs.clear();
      if (!physical_inputs.try_reserve(logical_inputs.size()))
        return plan_allocation_failure();
      for (ValueId input : logical_inputs) {
        auto instance = ensure_local_instance(input.ordinal(), domain);
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
      for (ValueId output : logical_outputs) {
        const std::uint32_t value = output.ordinal();
        std::uint8_t roles = 0;
        if ((plan.binding_roles[value] & detail::kExternalOutput) != 0 &&
            plan.external_domains[value] == domain) {
          roles |= detail::kExternalOutput;
        }
        auto instance = add_instance(value, domain, roles);
        if (!instance)
          return std::move(instance).error();
        producer_instances[value] = *instance;
        if ((roles & detail::kExternalOutput) != 0)
          plan.external_instances[value] = *instance;
        if (!plan.node_outputs.try_push_back(*instance))
          return plan_allocation_failure();
        plan.value_materialized[*instance] = 1;
      }
      const std::uint32_t step_index =
          static_cast<std::uint32_t>(plan.steps.size());
      if (!plan.steps.try_emplace_back(
              std::move(*region), node_index, candidate.proposal.nodes.count,
              input_begin, static_cast<std::uint32_t>(logical_inputs.size()),
              output_begin, static_cast<std::uint32_t>(logical_outputs.size()),
              ExecutionDomainId{domain})) {
        return plan_allocation_failure();
      }
      const std::uint32_t end = node_index + candidate.proposal.nodes.count;
      for (std::uint32_t covered = node_index; covered < end; ++covered)
        plan.node_steps[covered] = step_index;
      ++plan.region_count;
      node_index = end;
      ++selection_index;
      continue;
    }

    const detail::NodeRecord &node = graph.nodes[node_index];
    const std::uint32_t domain = node_domains[node_index];
    physical_inputs.clear();
    if (!physical_inputs.try_reserve(node.input_count))
      return plan_allocation_failure();
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t value = graph.edges[node.input_begin + input_index];
      auto instance = ensure_local_instance(value, domain);
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
      std::uint8_t roles = 0;
      if ((plan.binding_roles[value] & detail::kExternalOutput) != 0 &&
          plan.external_domains[value] == domain) {
        roles |= detail::kExternalOutput;
      }
      auto instance = add_instance(value, domain, roles);
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
            : (eligible_region_coverage[node_index] == 0
                   ? CompileFallbackReason::kNoEligibleRegion
                   : CompileFallbackReason::kRegionNotSelected);
    const std::uint32_t step_index =
        static_cast<std::uint32_t>(plan.steps.size());
    if (!plan.steps.try_emplace_back(
            std::move(prepared[node_index]),
            static_cast<std::uint32_t>(node_index), input_begin,
            node.input_count, output_begin, node.output_count, materializes,
            reason, ExecutionDomainId{domain})) {
      return plan_allocation_failure();
    }
    plan.node_steps[node_index] = step_index;
    ++node_index;
  }

  for (std::uint32_t value : graph.outputs) {
    if (plan.external_instances[value] != kNone)
      continue;
    const std::uint32_t source = producer_instances[value];
    if (source == kNone)
      return Status::Internal("published graph value has no physical source");
    const std::uint32_t domain = plan.external_domains[value];
    std::uint32_t destination =
        instances[static_cast<std::size_t>(value) * domain_count + domain];
    if (destination == kNone) {
      auto created = add_instance(value, domain, detail::kExternalOutput);
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

Status discover_region_candidates(
    const Graph &published_graph, const detail::GraphState &graph,
    const compute::Context &context, CompileOptions options,
    mem::AllocatorRef allocator, RegionExtensions &extensions,
    RegionCandidates &candidates, ExecutionDomainId domain,
    std::span<const std::uint32_t> node_domains) noexcept {
  if (options.region_selection == RegionSelectionPolicy::kDisabled ||
      graph.nodes.size() < 2) {
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
  if (options.max_region_proposals >
          (std::numeric_limits<std::uint32_t>::max)() ||
      options.max_region_proposals_per_provider >
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
    auto extension = compute::detail::ProviderExtensionAccess::query(
        context, index, kRegionProviderExtensionId);
    if (!extension) {
      if (extension.error().IsUnimplemented())
        continue;
      return std::move(extension).error();
    }
    if (!extensions.try_push_back(
            RegionExtension{std::move(*extension),
                            static_cast<std::uint32_t>(index), domain})) {
      return plan_allocation_failure();
    }
  }
  if (extensions.size() == extension_begin)
    return Status::Ok();

  container::SmallVector<RegionProposal, 32, mem::AllocatorRef> storage(
      allocator);
  if (candidates.size() > options.max_region_proposals) {
    return Status::Internal("graph region proposal accounting overflowed");
  }
  const std::size_t remaining_budget =
      options.max_region_proposals - candidates.size();
  if (!storage.try_resize(remaining_budget) ||
      !candidates.try_reserve(options.max_region_proposals)) {
    return plan_allocation_failure();
  }

  std::size_t used = 0;
  for (std::size_t extension_index = extension_begin;
       extension_index < extensions.size(); ++extension_index) {
    const std::size_t remaining = storage.size() - used;
    const std::size_t provider_capacity =
        (std::min)(remaining, options.max_region_proposals_per_provider);
    RegionProposalWriter writer(
        std::span<RegionProposal>(storage).subspan(used, provider_capacity));
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

    for (const RegionProposal &proposal : writer.proposals()) {
      if (proposal.nodes.count < 2 ||
          proposal.nodes.end() > graph.nodes.size()) {
        return Status::InvalidArgument(
            "provider proposed an invalid graph region interval");
      }
      const std::optional<std::uint64_t> estimate =
          proposal.cost.estimated_net_savings_ns;
      if ((estimate.has_value() && *estimate == 0) ||
          (options.region_selection ==
               RegionSelectionPolicy::kRequireEstimatedBenefit &&
           !estimate.has_value())) {
        continue;
      }
      if (proposal.cost.dispatch_count == 0) {
        return Status::InvalidArgument(
            "provider proposed a graph region with no dispatch");
      }
      bool domain_eligible = true;
      if (!node_domains.empty()) {
        for (std::uint32_t node = proposal.nodes.begin;
             node < proposal.nodes.end(); ++node) {
          if (node_domains[node] != domain.ordinal()) {
            domain_eligible = false;
            break;
          }
        }
      }
      if (!domain_eligible)
        continue;
      if (!candidates.try_push_back(RegionCandidate{
              proposal, static_cast<std::uint32_t>(extension_index),
              extensions[extension_index].provider_index,
              static_cast<std::uint32_t>(candidates.size()), domain})) {
        return plan_allocation_failure();
      }
    }
    used += writer.size();
  }
  return Status::Ok();
}

struct RegionSelectionScore final {
  std::uint64_t estimated_savings = 0;
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
                  RegionSelectionPolicy policy) noexcept {
  if (policy == RegionSelectionPolicy::kRequireEstimatedBenefit &&
      left.estimated_savings != right.estimated_savings) {
    return left.estimated_savings > right.estimated_savings;
  }
  if (left.covered_nodes != right.covered_nodes)
    return left.covered_nodes > right.covered_nodes;
  if (left.dispatches != right.dispatches)
    return left.dispatches < right.dispatches;
  return left.provider_rank_cost < right.provider_rank_cost;
}

Status select_regions(RegionCandidates &candidates, mem::AllocatorRef allocator,
                      RegionSelectionPolicy policy,
                      RegionCandidates &selected) noexcept {
  std::sort(
      candidates.begin(), candidates.end(),
      [](const RegionCandidate &left, const RegionCandidate &right) noexcept {
        if (left.proposal.nodes.end() != right.proposal.nodes.end())
          return left.proposal.nodes.end() < right.proposal.nodes.end();
        if (left.proposal.nodes.begin != right.proposal.nodes.begin)
          return left.proposal.nodes.begin < right.proposal.nodes.begin;
        if (left.extension_index != right.extension_index)
          return left.extension_index < right.extension_index;
        if (left.proposal.route != right.proposal.route)
          return left.proposal.route < right.proposal.route;
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
    const std::uint64_t begin = candidates[index].proposal.nodes.begin;
    const auto previous_end = std::upper_bound(
        candidates.begin(), candidates.begin() + index, begin,
        [](std::uint64_t value, const RegionCandidate &candidate) noexcept {
          return value < candidate.proposal.nodes.end();
        });
    const std::size_t previous_count =
        static_cast<std::size_t>(previous_end - candidates.begin());
    predecessors[index] = static_cast<std::uint32_t>(previous_count);
    RegionSelectionScore include = scores[previous_count];
    include.estimated_savings = saturating_add(
        include.estimated_savings,
        candidates[index].proposal.cost.estimated_net_savings_ns.value_or(0));
    include.covered_nodes = saturating_add(
        include.covered_nodes,
        static_cast<std::uint64_t>(candidates[index].proposal.nodes.count));
    include.dispatches =
        saturating_add(include.dispatches,
                       static_cast<std::uint64_t>(
                           candidates[index].proposal.cost.dispatch_count));
    include.provider_rank_cost = saturating_add(
        include.provider_rank_cost,
        saturating_multiply(
            static_cast<std::uint64_t>(candidates[index].provider_index) + 1,
            static_cast<std::uint64_t>(
                candidates[index].proposal.nodes.count)));
    const RegionSelectionScore exclude = scores[index];
    if (better_score(include, exclude, policy)) {
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

Status append_operation_step(detail::PlanState &plan,
                             const detail::GraphState &graph,
                             const compute::Context &context,
                             std::uint32_t node_index,
                             CompileFallbackReason fallback_reason) noexcept {
  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions input_descriptions(plan.allocator);
  Descriptions output_descriptions(plan.allocator);
  const detail::NodeRecord &node = graph.nodes[node_index];
  if (!input_descriptions.try_resize(node.input_count) ||
      !output_descriptions.try_resize(node.output_count)) {
    return plan_allocation_failure();
  }
  const std::uint32_t plan_input_begin =
      static_cast<std::uint32_t>(plan.node_inputs.size());
  const std::uint32_t plan_output_begin =
      static_cast<std::uint32_t>(plan.node_outputs.size());
  for (std::size_t index = 0; index < node.input_count; ++index) {
    const std::uint32_t ordinal = graph.edges[node.input_begin + index];
    input_descriptions[index] = plan.descriptions[ordinal];
    if (!plan.node_inputs.try_push_back(ordinal))
      return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < node.output_count; ++index) {
    const std::uint32_t ordinal =
        node.output_begin + static_cast<std::uint32_t>(index);
    output_descriptions[index] = plan.descriptions[ordinal];
    if (!plan.node_outputs.try_push_back(ordinal))
      return plan_allocation_failure();
    plan.value_materialized[ordinal] = 1;
  }
  auto prepared = compute::prepare(context, node.operation, input_descriptions,
                                   output_descriptions);
  if (!prepared)
    return std::move(prepared).error();
  const bool materializes =
      compute::operation_traits(node.operation).materializes;
  const std::uint32_t step_index =
      static_cast<std::uint32_t>(plan.steps.size());
  if (!plan.steps.try_emplace_back(std::move(*prepared), node_index,
                                   plan_input_begin, node.input_count,
                                   plan_output_begin, node.output_count,
                                   materializes, fallback_reason)) {
    return plan_allocation_failure();
  }
  plan.node_steps[node_index] = step_index;
  return Status::Ok();
}

Status append_region_step(
    detail::PlanState &plan, const Graph &published_graph,
    const detail::GraphState &graph, const RegionExtension &extension,
    const RegionProposal &proposal,
    std::span<const std::uint32_t> last_consumers,
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> &marks,
    std::uint32_t marker, CompileOptions options) noexcept {
  RegionValues inputs(plan.allocator);
  RegionValues outputs(plan.allocator);
  const std::uint32_t begin = proposal.nodes.begin;
  const std::uint32_t end =
      begin + static_cast<std::uint32_t>(proposal.nodes.count);
  Status boundary = collect_region_boundary(
      plan, graph, proposal, last_consumers, marks, marker, inputs, outputs);
  if (!boundary.ok())
    return boundary;
  auto prepared = compile_region_artifact(plan, published_graph, extension,
                                          proposal, inputs, outputs, options);
  if (!prepared)
    return std::move(prepared).error();

  const std::uint32_t plan_input_begin =
      static_cast<std::uint32_t>(plan.node_inputs.size());
  const std::uint32_t plan_output_begin =
      static_cast<std::uint32_t>(plan.node_outputs.size());
  for (ValueId input : inputs) {
    if (!plan.node_inputs.try_push_back(input.ordinal()))
      return plan_allocation_failure();
  }
  for (ValueId output : outputs) {
    if (!plan.node_outputs.try_push_back(output.ordinal()))
      return plan_allocation_failure();
    plan.value_materialized[output.ordinal()] = 1;
  }

  const std::uint32_t step_index =
      static_cast<std::uint32_t>(plan.steps.size());
  if (!plan.steps.try_emplace_back(
          std::move(*prepared), begin, proposal.nodes.count, plan_input_begin,
          static_cast<std::uint32_t>(inputs.size()), plan_output_begin,
          static_cast<std::uint32_t>(outputs.size()))) {
    return plan_allocation_failure();
  }
  for (std::uint32_t node_index = begin; node_index < end; ++node_index)
    plan.node_steps[node_index] = step_index;
  ++plan.region_count;
  return Status::Ok();
}

Status prepare_steps(detail::PlanState &plan, const Graph &published_graph,
                     const detail::GraphState &graph,
                     const compute::Context &context,
                     CompileOptions options) noexcept {
  RegionExtensions extensions(plan.allocator);
  RegionCandidates candidates(plan.allocator);
  RegionCandidates selected(plan.allocator);
  Status status = discover_region_candidates(
      published_graph, graph, context, options, plan.allocator, extensions,
      candidates, ExecutionDomainId{0});
  if (!status.ok())
    return status;
  status = select_regions(candidates, plan.allocator, options.region_selection,
                          selected);
  if (!status.ok())
    return status;
  if (options.operation_fallback == OperationFallbackPolicy::kReject) {
    std::size_t covered_nodes = 0;
    for (const RegionCandidate &candidate : selected) {
      covered_nodes = saturating_add(
          covered_nodes,
          static_cast<std::size_t>(candidate.proposal.nodes.count));
    }
    if (covered_nodes != graph.nodes.size()) {
      return Status::Unimplemented(
          "graph specialization requires a single-operation fallback");
    }
  }

  container::SmallVector<std::int64_t, 32, mem::AllocatorRef>
      eligible_region_coverage(plan.allocator);
  if (!eligible_region_coverage.try_resize(graph.nodes.size() + 1))
    return plan_allocation_failure();
  for (std::int64_t &count : eligible_region_coverage)
    count = 0;
  for (const RegionCandidate &candidate : candidates) {
    ++eligible_region_coverage[candidate.proposal.nodes.begin];
    --eligible_region_coverage[static_cast<std::size_t>(
        candidate.proposal.nodes.end())];
  }
  for (std::size_t index = 1; index < eligible_region_coverage.size();
       ++index) {
    eligible_region_coverage[index] += eligible_region_coverage[index - 1];
  }

  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> last_consumers(
      plan.allocator);
  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> marks(
      plan.allocator);
  if (!last_consumers.try_resize(graph.values.size()) ||
      !marks.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    last_consumers[index] = detail::kNoNode;
    marks[index] = 0;
  }
  for (std::uint32_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      last_consumers[graph.edges[node.input_begin + input_index]] = node_index;
    }
  }

  std::size_t selection_index = 0;
  std::uint32_t node_index = 0;
  std::uint32_t marker = 1;
  while (node_index < graph.nodes.size()) {
    if (selection_index < selected.size() &&
        selected[selection_index].proposal.nodes.begin == node_index) {
      const RegionCandidate &candidate = selected[selection_index];
      status = append_region_step(
          plan, published_graph, graph, extensions[candidate.extension_index],
          candidate.proposal, last_consumers, marks, marker++, options);
      if (!status.ok())
        return status;
      node_index += candidate.proposal.nodes.count;
      ++selection_index;
      continue;
    }
    const CompileFallbackReason reason =
        options.region_selection == RegionSelectionPolicy::kDisabled
            ? CompileFallbackReason::kRegionSelectionDisabled
            : (eligible_region_coverage[node_index] == 0
                   ? CompileFallbackReason::kNoEligibleRegion
                   : CompileFallbackReason::kRegionNotSelected);
    status = append_operation_step(plan, graph, context, node_index, reason);
    if (!status.ok())
      return status;
    ++node_index;
  }
  return Status::Ok();
}

// Reorder already-prepared plan steps without changing the logical Graph.
// Kahn scheduling keeps a ready FIFO per execution domain and continues the
// current domain while possible. This deterministically removes avoidable
// Queue segment boundaries from interleaved independent branches; providers,
// transfer routes and bindings remain fully resolved before the warm path.
Status schedule_steps(detail::PlanState &plan,
                      const detail::GraphState &graph) noexcept {
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Edges = container::SmallVector<std::uint64_t, 32, mem::AllocatorRef>;
  constexpr std::uint32_t kNone = detail::kNoNode;
  const std::size_t step_count = plan.steps.size();
  if (step_count <= 1 || plan.domains.size() <= 1)
    return Status::Ok();
  if (step_count >= kNone) {
    return Status::ResourceExhausted(
        "graph scheduled-step identity space is exhausted");
  }
  if (plan.node_inputs.size() > SIZE_MAX - graph.dependencies.size()) {
    return Status::ResourceExhausted(
        "graph scheduling dependency capacity overflows");
  }

  Ordinals producers(plan.allocator);
  Edges edges(plan.allocator);
  if (!producers.try_resize(plan.instance_values.size()) ||
      !edges.try_reserve(plan.node_inputs.size() + graph.dependencies.size())) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &producer : producers)
    producer = kNone;
  for (std::size_t step_index = 0; step_index < step_count; ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
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
    const std::uint32_t node_end = step.node_begin + step.node_count;
    for (std::uint32_t node = step.node_begin; node < node_end; ++node) {
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
      !ready_head.try_resize(plan.domains.size()) ||
      !ready_tail.try_resize(plan.domains.size()) ||
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
    if (source >= destination || destination >= step_count) {
      return Status::Internal(
          "prepared graph steps are not in topological order");
    }
    ++outgoing_begin[source + 1];
    ++indegree[destination];
  }
  for (std::size_t index = 1; index < outgoing_begin.size(); ++index)
    outgoing_begin[index] += outgoing_begin[index - 1];

  auto enqueue_ready = [&](std::uint32_t step) noexcept -> Status {
    const ExecutionDomainId domain = plan.steps[step].domain;
    if (!domain.valid() || domain.ordinal() >= plan.domains.size())
      return Status::Internal("prepared graph step has an invalid domain");
    std::uint32_t &head = ready_head[domain.ordinal()];
    std::uint32_t &tail = ready_tail[domain.ordinal()];
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
  std::uint32_t current_domain = kNone;
  while (scheduled.size() < step_count) {
    if (current_domain == kNone || ready_head[current_domain] == kNone) {
      current_domain = kNone;
      std::uint32_t earliest = kNone;
      for (std::size_t domain = 0; domain < ready_head.size(); ++domain) {
        if (ready_head[domain] != kNone &&
            (earliest == kNone || ready_head[domain] < earliest)) {
          earliest = ready_head[domain];
          current_domain = static_cast<std::uint32_t>(domain);
        }
      }
      if (current_domain == kNone)
        return Status::Internal("prepared graph step dependencies are cyclic");
    }
    const std::uint32_t selected = ready_head[current_domain];
    ready_head[current_domain] = ready_next[selected];
    if (ready_head[current_domain] == kNone)
      ready_tail[current_domain] = kNone;
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
    const std::uint32_t end = step.node_begin + step.node_count;
    for (std::uint32_t node = step.node_begin; node < end; ++node)
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
  Ordinals step_segments(plan.allocator);
  Ordinals producers(plan.allocator);
  Ordinals last_domain_segment(plan.allocator);
  Edges edges(plan.allocator);
  if (!plan.segments.try_reserve(plan.steps.size()) ||
      !step_segments.try_resize(plan.steps.size()) ||
      !producers.try_resize(plan.instance_values.size()) ||
      !last_domain_segment.try_resize(plan.domains.size()) ||
      !plan.domain_segment_counts.try_resize(plan.domains.size())) {
    return plan_allocation_failure();
  }
  for (std::uint32_t &segment : step_segments)
    segment = detail::kNoNode;
  for (std::uint32_t &producer : producers)
    producer = detail::kNoNode;
  for (std::uint32_t &segment : last_domain_segment)
    segment = detail::kNoNode;
  for (std::uint32_t &count : plan.domain_segment_counts)
    count = 0;

  // Metadata-only View steps resolve aliases during binding preparation and
  // do not enter a Queue. Keep them inside the surrounding report interval, but
  // start/split runtime segments only for actual commands. This avoids an
  // empty sequence/event for View-only graphs and lets consecutive commands
  // on one domain remain one uninterrupted Queue prefix.
  std::uint32_t active_segment = detail::kNoNode;
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    if (!plan.steps[index].emits_command())
      continue;
    const ExecutionDomainId domain = plan.steps[index].domain;
    if (!domain.valid() || domain.ordinal() >= plan.domains.size())
      return Status::Internal("graph plan step has an invalid domain");
    if (active_segment == detail::kNoNode ||
        plan.segments[active_segment].domain != domain) {
      if (active_segment != detail::kNoNode) {
        detail::PlanSegment &previous = plan.segments[active_segment];
        previous.step_count =
            static_cast<std::uint32_t>(index - previous.step_begin);
      }
      if (!plan.segments.try_push_back(detail::PlanSegment{
              .domain = domain,
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
      step_segments[step] = static_cast<std::uint32_t>(segment_index);
    ++plan.domain_segment_counts[segment.domain.ordinal()];
  }
  for (std::uint32_t count : plan.domain_segment_counts)
    plan.required_queue_count += count != 0 ? 1u : 0u;

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
    const std::uint32_t domain = segment.domain.ordinal();
    if (last_domain_segment[domain] != detail::kNoNode) {
      Status status = add_edge(static_cast<std::uint32_t>(segment_index),
                               last_domain_segment[domain]);
      if (!status.ok())
        return status;
    }
    last_domain_segment[domain] = static_cast<std::uint32_t>(segment_index);
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
                                 step_segments[producer]);
        if (!status.ok())
          return status;
      }
      if (step.kind != detail::PlanStepKind::kOperation &&
          step.kind != detail::PlanStepKind::kRegion) {
        continue;
      }
      const std::uint32_t node_end = step.node_begin + step.node_count;
      for (std::uint32_t node = step.node_begin; node < node_end; ++node) {
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
                                   step_segments[dependency_step]);
          if (!status.ok())
            return status;
        }
      }
    }
  }

  // LifetimePlanner reasons in the canonical step order, while different
  // domain Queues may execute concurrently. If a step on domain B reads an
  // Invocation-owned value stored in domain A, the next A segment must not
  // reuse that backing until B's read is complete. Add the corresponding
  // cross-domain anti-dependency in one reverse pass; the normal per-domain
  // chain then protects every later A segment as well.
  Ordinals next_domain_segment(plan.allocator);
  if (!next_domain_segment.try_resize(plan.domains.size()))
    return plan_allocation_failure();
  for (std::uint32_t &next : next_domain_segment)
    next = detail::kNoNode;
  for (std::size_t reverse = plan.segments.size(); reverse != 0; --reverse) {
    const std::uint32_t segment_index = static_cast<std::uint32_t>(reverse - 1);
    const detail::PlanSegment &segment = plan.segments[segment_index];
    const std::size_t end = segment.step_begin + segment.step_count;
    for (std::size_t step_index = segment.step_begin; step_index < end;
         ++step_index) {
      const detail::PlanStep &step = plan.steps[step_index];
      if (!step.emits_command())
        continue;
      for (std::size_t input_index = 0; input_index < step.input_count;
           ++input_index) {
        const std::uint32_t input =
            plan.node_inputs[step.input_begin + input_index];
        const std::uint32_t source_domain = plan.instance_domains[input];
        if (source_domain == segment.domain.ordinal() ||
            plan.workspace_requests[input] == detail::kNoWorkspaceRequest) {
          continue;
        }
        const std::uint32_t next = next_domain_segment[source_domain];
        if (next != detail::kNoNode) {
          Status status = add_edge(next, segment_index);
          if (!status.ok())
            return status;
        }
      }
    }
    next_domain_segment[segment.domain.ordinal()] = segment_index;
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
  for (std::uint32_t last : last_domain_segment) {
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

Status plan_workspace(detail::PlanState &plan,
                      const detail::GraphState &graph) noexcept {
  using LastUses = container::SmallVector<std::uint64_t, 16, mem::AllocatorRef>;
  using Requests =
      container::SmallVector<mem::LifetimeRequest, 8, mem::AllocatorRef>;

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
            .resource_slot = plan.instance_domains[ordinal],
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
    if (plan.steps[step_index].kind == detail::PlanStepKind::kRegion &&
        !plan.steps[step_index]
             .region.properties()
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
            .resource_slot = plan.steps[step_index].domain.ordinal(),
            .size = requirement.size,
            .alignment = requirement.alignment,
            .begin = step_index,
            .end = step_index + 1,
        })) {
      return plan_allocation_failure();
    }
    plan.step_workspace_requests[step_index] = request;
  }

  plan.workspace_resources.clear();
  if (!plan.domain_workspace_slots.try_resize(plan.domains.size()) ||
      !plan.workspace_resources.try_reserve(plan.domains.size()))
    return plan_allocation_failure();
  for (std::uint32_t &slot : plan.domain_workspace_slots)
    slot = detail::kNoWorkspaceRequest;
  for (mem::LifetimeRequest &request : requests) {
    const std::uint32_t domain = request.resource_slot;
    if (domain >= plan.domain_workspace_slots.size())
      return Status::Internal("graph workspace domain is out of range");
    std::uint32_t &slot = plan.domain_workspace_slots[domain];
    if (slot == detail::kNoWorkspaceRequest) {
      if (plan.workspace_resources.size() >= detail::kNoWorkspaceRequest ||
          !plan.workspace_resources.try_push_back(
              plan.domains[domain].workspace_resource)) {
        return plan_allocation_failure();
      }
      slot = static_cast<std::uint32_t>(plan.workspace_resources.size() - 1);
    }
    request.resource_slot = slot;
  }
  const std::span<const mem::MemoryResourceRef> selected_resources(
      plan.workspace_resources.data(), plan.workspace_resources.size());
  auto lifetime_plan = mem::LifetimePlanner::build(
      std::span<const mem::LifetimeRequest>(requests.data(), requests.size()),
      selected_resources);
  if (!lifetime_plan)
    return std::move(lifetime_plan).error();

  plan.workspace_plan = std::move(*lifetime_plan);
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
    }
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
  void *storage = plan.allocator.allocate(sizeof(detail::ReportState),
                                          alignof(detail::ReportState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *state = ::new (storage) detail::ReportState(plan.allocator);
  if (!state->steps.try_resize(plan.steps.size()) ||
      !state->domains.try_resize(plan.domains.size()) ||
      !state->segments.try_resize(plan.segments.size()) ||
      !state->physical_values.try_resize(plan.instance_values.size()) ||
      !state->step_inputs.try_resize(plan.node_inputs.size()) ||
      !state->step_outputs.try_resize(plan.node_outputs.size()) ||
      !state->segment_dependencies.try_resize(
          plan.segment_dependencies.size())) {
    detail::release(state);
    return std::unexpected(plan_allocation_failure());
  }
  std::copy(plan.node_inputs.begin(), plan.node_inputs.end(),
            state->step_inputs.begin());
  std::copy(plan.node_outputs.begin(), plan.node_outputs.end(),
            state->step_outputs.begin());
  std::copy(plan.segment_dependencies.begin(), plan.segment_dependencies.end(),
            state->segment_dependencies.begin());
  for (std::size_t index = 0; index < plan.domains.size(); ++index) {
    const compute::Device device = plan.domains[index].context.device();
    auto properties = device.properties();
    if (!properties) {
      detail::release(state);
      return std::unexpected(std::move(properties).error());
    }
    auto capture = plan.domains[index].context.has_backend_extension(
        compute::detail::kCommandCaptureExtensionId);
    if (!capture) {
      detail::release(state);
      return std::unexpected(std::move(capture).error());
    }
    state->domains[index] = CompileDomainReport{
        .domain = ExecutionDomainId{static_cast<std::uint32_t>(index)},
        .device = device,
        .completion_slots_per_invocation = plan.domain_segment_counts[index],
        .queue_required = plan.domain_segment_counts[index] != 0,
        .asynchronous_queue = properties->asynchronous_queues,
        .capture_supported = *capture,
    };
    const std::uint32_t workspace_slot = plan.domain_workspace_slots[index];
    if (workspace_slot != detail::kNoWorkspaceRequest) {
      auto resource = plan.workspace_plan.resource_plan(workspace_slot);
      if (!resource) {
        detail::release(state);
        return std::unexpected(std::move(resource).error());
      }
      state->domains[index].workspace_bytes = resource->peak_bytes;
      state->domains[index].workspace_backing_count = resource->backing_count;
    }
    if (plan.domain_segment_counts[index] == 0) {
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
        .domain = ExecutionDomainId{plan.instance_domains[index]},
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
        state->domains[segment.domain.ordinal()].capture_supported;
    state->segments[index] = CompileSegmentReport{
        .domain = segment.domain,
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
    state->capturable_segment_count += capture_supported ? 1u : 0u;
    const std::size_t dependency_end =
        segment.dependency_begin + segment.dependency_count;
    for (std::size_t dependency = segment.dependency_begin;
         dependency < dependency_end; ++dependency) {
      const std::uint32_t source = plan.segment_dependencies[dependency];
      const std::uint32_t source_domain =
          plan.segments[source].domain.ordinal();
      if (source_domain != segment.domain.ordinal() &&
          state->domains[source_domain].asynchronous_queue) {
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
    CompileStepReport step_report{
        .kind = step.kind == detail::PlanStepKind::kRegion
                    ? CompileStepKind::kProviderRegion
                    : CompileStepKind::kOperationFallback,
        .fallback_reason = step.fallback_reason,
        .provider = step.provider(),
        .domain = step.domain,
        .source_domain = step.source_domain,
        .destination_domain = step.destination_domain,
        .node_begin = step.node_begin,
        .node_count = step.node_count,
        .workspace = step.workspace_requirement(),
        .borrowed_constant_count = step_constant_count(plan, step),
        .dispatch_count = step.emits_command() ? 1u : 0u,
        .input_begin = step.input_begin,
        .output_begin = step.output_begin,
        .input_count = step.input_count,
        .output_count = step.output_count,
        .capture_compatible = step.capture_compatible(),
    };
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
      step_report.blocks_host = properties.blocks_host;
      step_report.uses_private_threads = properties.uses_private_threads;
      step_report.supports_concurrent_invocations =
          properties.supports_concurrent_invocations;
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

std::size_t CompileReport::workspace_bytes() const noexcept {
  return state_ != nullptr ? state_->workspace_bytes : 0;
}

std::size_t CompileReport::workspace_backing_count() const noexcept {
  return state_ != nullptr ? state_->workspace_backing_count : 0;
}

std::size_t CompileReport::node_count() const noexcept {
  return state_ != nullptr ? state_->node_count : 0;
}

std::size_t CompileReport::domain_count() const noexcept {
  return state_ != nullptr ? state_->domains.size() : 0;
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

Result<CompileDomainReport>
CompileReport::domain(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->domains.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report domain index is out of range"));
  }
  return state_->domains[index];
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
  const ExecutionDomainId source_domain = state_->segments[source].domain;
  CompileDependencyRoute route = CompileDependencyRoute::kOrderedQueue;
  if (source_domain != segment.domain) {
    route = state_->domains[source_domain.ordinal()].asynchronous_queue
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

std::size_t Executable::domain_count() const noexcept {
  return state_ != nullptr ? state_->domains.size() : 0;
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

Result<bool>
Executable::requires_queue(ExecutionDomainId domain) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (!domain.valid() || domain.ordinal() >= state_->domains.size()) {
    return std::unexpected(
        Status::OutOfRange("execution domain is out of range"));
  }
  return state_->domain_segment_counts[domain.ordinal()] != 0;
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
      !invocation->queues.try_resize(state_->domains.size()) ||
      !invocation->queue_seen.try_resize(state_->domains.size()) ||
      !invocation->segment_completions.try_resize(state_->segments.size()) ||
      !invocation->captured_segments.try_resize(state_->segments.size()) ||
      !invocation->dependency_routes.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->candidate_dependency_routes.try_resize(
          state_->segment_dependencies.size()) ||
      !invocation->aggregate_segments.try_resize(state_->domains.size()) ||
      !invocation->aggregate_children.try_resize(state_->domains.size())) {
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
                                   "every active execution domain");
  }
  for (std::size_t index = 0; index < state_->queues.size(); ++index)
    state_->queue_seen[index] = 0;
  for (const QueueBinding binding : queues) {
    if (!binding.domain.valid() ||
        binding.domain.ordinal() >= state_->plan->domains.size() ||
        binding.queue == nullptr || !binding.queue->valid() ||
        !binding.queue->is_open()) {
      return Status::InvalidArgument("graph replay queue binding is invalid");
    }
    const std::size_t domain = binding.domain.ordinal();
    if (state_->plan->domain_segment_counts[domain] == 0) {
      return Status::InvalidArgument(
          "graph replay queue names an unused execution domain");
    }
    if (state_->queue_seen[domain] != 0)
      return Status::InvalidArgument(
          "graph replay queue binding is duplicated");
    if (!state_->plan->domains[domain].context.owns(*binding.queue)) {
      return Status::InvalidArgument(
          "graph replay queue belongs to a different compute context");
    }
    state_->queue_seen[domain] = 1;
  }
  for (std::size_t domain = 0; domain < state_->queue_seen.size(); ++domain) {
    if (state_->plan->domain_segment_counts[domain] != 0 &&
        state_->queue_seen[domain] == 0) {
      return Status::InvalidArgument("graph replay queue binding is missing");
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
      prepare_dependency_routes(*state_, queues, options.host_wait);
  if (!dependency_report)
    return std::move(dependency_report).error();
  dependency_report->immediate_transfer_count =
      transfer_report.immediate_transfer_count;
  dependency_report->queued_transfer_count =
      transfer_report.queued_transfer_count;
  dependency_report->peer_transfer_count = transfer_report.peer_transfer_count;

  // A deployment may deliberately bind several logical domains to one
  // ordered Queue (for example, distinct workspace resources accepted by one
  // Context). Reserve that Queue for the sum of all its emitted segments,
  // once. Per-domain reservations would understate ring capacity because the
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
      const std::size_t domain_count =
          state_->plan->domain_segment_counts[binding.domain.ordinal()];
      if (domain_count > SIZE_MAX - count) {
        return Status::ResourceExhausted(
            "graph replay completion capacity overflows");
      }
      count += domain_count;
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
    state_->queues[binding.domain.ordinal()] = binding.queue;
  std::copy(state_->candidate_dependency_routes.begin(),
            state_->candidate_dependency_routes.end(),
            state_->dependency_routes.begin());
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
    if (!segment_capture_compatible(*state_->plan, segment)) {
      if (options.mode == ReplayMode::kRequireCapture) {
        clear_captures();
        return Status::Unimplemented(
            "a graph segment contains a provider that is not capture-safe");
      }
      continue;
    }

    compute::Queue &queue = *state_->queues[segment.domain.ordinal()];
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
        "single-queue submit requires exactly one active execution domain"));
  }
  std::uint32_t active_domain = 0;
  while (state_->plan->domain_segment_counts[active_domain] == 0)
    ++active_domain;
  const std::array bindings{QueueBinding{
      .domain = ExecutionDomainId{active_domain}, .queue = &queue}};
  if (!state_->replay_prepared ||
      (state_->replay_route == ReplayRoute::kCommands &&
       state_->queues[active_domain] != &queue)) {
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
        "queue-free submit requires a graph with no runtime commands"));
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
    if (!binding.domain.valid() ||
        binding.domain.ordinal() >= plan.domains.size() ||
        binding.queue == nullptr) {
      return std::unexpected(
          Status::InvalidArgument("graph submit queue binding is invalid"));
    }
    const std::size_t domain = binding.domain.ordinal();
    if (plan.domain_segment_counts[domain] == 0 ||
        state_->queue_seen[domain] != 0 ||
        state_->queues[domain] != binding.queue) {
      return std::unexpected(Status::InvalidArgument(
          "graph submit queues differ from prepared replay queues"));
    }
    state_->queue_seen[domain] = 1;
  }
  for (std::size_t domain = 0; domain < state_->queue_seen.size(); ++domain) {
    if (plan.domain_segment_counts[domain] != 0 &&
        state_->queue_seen[domain] == 0) {
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
    compute::Queue &queue = *state_->queues[segment.domain.ordinal()];
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

    auto sequence = compute::detail::QueueSequence::begin(queue);
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
        compute::detail::EnqueueResult waited = sequence->enqueue_wait(source);
        const bool wait_succeeded = waited.status.ok();
        if (!wait_succeeded) {
          tail_error =
              waited.status.IsUnimplemented()
                  ? Status::Internal(
                        "prepared graph device-wait route was rejected")
                  : waited.status;
        }
        sequence->record(std::move(waited));
        if (!wait_succeeded) {
          continue_segment = false;
          break;
        }
      }
    }

    if (continue_segment && state_->captured_segments[segment_index].valid()) {
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
    if (!continue_segment)
      break;
  }

  auto select_latest_queue_prefixes =
      [&](std::span<const std::uint32_t> candidates) noexcept {
        std::size_t count = 0;
        for (const std::uint32_t candidate : candidates) {
          const compute::Queue *queue =
              state_->queues[plan.segments[candidate].domain.ordinal()];
          std::size_t match = count;
          for (std::size_t index = 0; index < count; ++index) {
            const std::uint32_t selected = state_->aggregate_segments[index];
            if (state_->queues[plan.segments[selected].domain.ordinal()] ==
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
    // domain. A synchronous tail rejection therefore remains asynchronous
    // when earlier independent branches are still running; no hidden host
    // drain is needed on the error path.
    for (std::size_t domain = 0; domain < state_->queue_seen.size(); ++domain)
      state_->queue_seen[domain] = 0;
    for (std::size_t index = 0; index < state_->segment_completions.size();
         ++index) {
      if (!state_->segment_completions[index].valid())
        continue;
      const std::uint32_t domain = plan.segments[index].domain.ordinal();
      state_->aggregate_segments[domain] = static_cast<std::uint32_t>(index);
      state_->queue_seen[domain] = 1;
    }
    std::size_t count = 0;
    for (std::size_t domain = 0; domain < state_->queue_seen.size(); ++domain) {
      if (state_->queue_seen[domain] != 0)
        state_->aggregate_segments[count++] =
            state_->aggregate_segments[domain];
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
  const std::array domains{ExecutionDomain{
      .context = &context,
      .workspace_resource = mem::system_resource(),
  }};
  return compile(graph, CompileEnvironment{.domains = domains}, options);
}

Result<CompiledGraph> compile(const Graph &graph,
                              CompileEnvironment environment,
                              CompileOptions options) noexcept {
  if (!graph.valid()) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }
  if (environment.domains.empty()) {
    return std::unexpected(
        Status::InvalidArgument("compile environment has no domains"));
  }
  if (environment.domains.size() >
      (std::numeric_limits<std::uint32_t>::max)()) {
    return std::unexpected(Status::ResourceExhausted(
        "compile environment domain identity space is exhausted"));
  }
  for (const ExecutionDomain &domain : environment.domains) {
    if (domain.context == nullptr || !domain.context->valid() ||
        !domain.context->is_open()) {
      return std::unexpected(Status::FailedPrecondition(
          "compile environment context is invalid or closed"));
    }
    if (!domain.workspace_resource.valid()) {
      return std::unexpected(Status::InvalidArgument(
          "compile environment workspace resource is invalid"));
    }
    auto lease = compute::detail::ContextRef::acquire(*domain.context);
    if (!lease)
      return std::unexpected(std::move(lease).error());
    Status compatible = lease->validate_resource(domain.workspace_resource);
    if (!compatible.ok())
      return std::unexpected(std::move(compatible));
  }
  const compute::Context &context = *environment.domains[0].context;
  if (!context.valid() || !context.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("compute context is invalid or closed"));
  }
  if (options.region_selection != RegionSelectionPolicy::kDisabled &&
      options.region_selection !=
          RegionSelectionPolicy::kPreferProviderRegions &&
      options.region_selection !=
          RegionSelectionPolicy::kRequireEstimatedBenefit) {
    return std::unexpected(
        Status::InvalidArgument("graph region selection policy is invalid"));
  }
  if (options.operation_fallback != OperationFallbackPolicy::kAllow &&
      options.operation_fallback != OperationFallbackPolicy::kReject) {
    return std::unexpected(
        Status::InvalidArgument("graph operation fallback policy is invalid"));
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
  if (!plan->domains.try_reserve(environment.domains.size())) {
    detail::release(plan);
    return std::unexpected(plan_allocation_failure());
  }
  for (const ExecutionDomain &domain : environment.domains) {
    auto lease = compute::detail::ContextRef::acquire(*domain.context);
    if (!lease) {
      Status error = std::move(lease).error();
      detail::release(plan);
      return std::unexpected(std::move(error));
    }
    if (!plan->domains.try_push_back(detail::ExecutionDomainRecord{
            .context = std::move(*lease),
            .workspace_resource = domain.workspace_resource,
        })) {
      detail::release(plan);
      return std::unexpected(plan_allocation_failure());
    }
  }

  Status status =
      initialize_plan_metadata(*plan, *concrete,
                               std::span<const compute::TensorDesc>(
                                   descriptions.data(), descriptions.size()));
  if (status.ok())
    status = bind_constants(*plan, *concrete, options);
  const bool explicit_placement =
      environment.domains.size() != 1 || !options.node_placements.empty() ||
      !options.value_placements.empty() || !options.external_resources.empty();
  if (status.ok()) {
    status =
        explicit_placement
            ? prepare_multi_domain_steps(*plan, *specialized, *concrete,
                                         environment, options)
            : prepare_steps(*plan, *specialized, *concrete, context, options);
  }
  if (status.ok())
    status = schedule_steps(*plan, *concrete);
  if (status.ok())
    status = plan_workspace(*plan, *concrete);
  if (status.ok())
    status = build_segments(*plan, *concrete);
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
