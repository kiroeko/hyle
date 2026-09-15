#include "route_profile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <type_traits>

#include "graph_state.h"
#include "hyle/container/hash.h"
#include "plan_state.h"

namespace hyle::graph::detail {
namespace {

// Versioned, length-delimited scalar serialization through Container's byte
// hash. No padding, native pointers, transient IDs or native-endian POD loads.
// A changed hash implementation invalidates hyle_build; this is not a file
// format or a cryptographic attestation.
class Fingerprint final {
public:
  explicit Fingerprint(std::uint64_t domain) noexcept { add(domain); }
  template <class T>
    requires(std::is_integral_v<T> || std::is_enum_v<T>)
  void add(T input) noexcept {
    auto value = static_cast<std::uint64_t>(input);
    std::array<std::uint8_t, 8> bytes{};
    for (auto &byte : bytes) {
      byte = static_cast<std::uint8_t>(value);
      value >>= 8;
    }
    for (auto &word : value_.words)
      word = container::hash_bytes(bytes.data(), bytes.size(), word);
  }
  void add(float value) noexcept { add(std::bit_cast<std::uint32_t>(value)); }
  void add(compute::ProfileFingerprint value) noexcept {
    for (auto word : value.words)
      add(word);
  }
  template <class T> void optional(std::optional<T> value) noexcept {
    add(value.has_value());
    if (value)
      add(*value);
  }
  template <class Range> void words(const Range &range) noexcept {
    add(range.size());
    for (auto word : range)
      add(word);
  }
  void format(const compute::StorageFormat &format) noexcept {
    add(format.kind());
    add(format.id().namespace_id);
    add(format.id().format_id);
    add(format.elements_per_block());
    add(format.bytes_per_block());
    add(format.required_alignment());
  }
  Status tensor(const compute::TensorDesc &tensor) noexcept {
    add(tensor.data_type());
    words(tensor.dimensions());
    words(tensor.byte_strides());
    format(tensor.storage_format());
    optional(tensor.block_axis());
    HYLE_ASSIGN_OR_RETURN(const auto bytes, tensor.required_bytes());
    add(bytes);
    return Status::Ok();
  }
  void operand(compute::detail::ShapeOperand value) noexcept {
    add(value.ordinal);
    add(value.constant);
  }
  compute::ProfileFingerprint finish() const noexcept { return value_; }

private:
  compute::ProfileFingerprint value_{
      {0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL,
       0x082efa98ec4e6c89ULL}};
};

// Explicit fields are deliberately enumerated. Extending OperationKind must
// extend this encoder; unknown descriptors never become hashable via memcpy.
Status operation_identity(Fingerprint &key,
                          const compute::Operation &operation) noexcept {
  using namespace compute;
  key.add(operation.kind());
  const auto traits = operation_traits(operation);
  key.add(traits.inputs.minimum);
  key.add(traits.inputs.maximum);
  key.add(traits.outputs);
  key.add(traits.alias);
  key.add(traits.effect);
  key.add(traits.precision);
  key.add(traits.accumulation_type);
  key.add(traits.materializes);
  key.add(traits.resource_count);
  for (std::size_t port = 0; port < traits.resource_count; ++port) {
    HYLE_ASSIGN_OR_RETURN(const auto access,
                          operation_resource_access(operation, port));
    key.add(access);
  }
  switch (operation.kind()) {
  case OperationKind::kView: {
    const auto &v = *operation.get_if<View>();
    key.add(v.kind());
    key.add(v.shape().valid());
    key.words(v.shape().dimensions());
    key.add(v.axis());
    key.add(v.start());
    key.add(v.count());
    key.add(v.step());
    key.words(v.permutation());
    break;
  }
  case OperationKind::kCopy:
  case OperationKind::kSelect:
    break;
  case OperationKind::kFill: {
    const auto &v = *operation.get_if<Fill>();
    key.words(v.output_shape.dimensions());
    key.add(v.output_type);
    key.add(v.value.data_type());
    key.add(v.value.bits());
    break;
  }
  case OperationKind::kCast:
    key.add(operation.get_if<Cast>()->output_type);
    break;
  case OperationKind::kUnary:
    key.add(operation.get_if<Unary>()->kind);
    break;
  case OperationKind::kBinary:
    key.add(operation.get_if<Binary>()->kind);
    break;
  case OperationKind::kReduce: {
    const auto &v = *operation.get_if<Reduce>();
    key.add(v.kind);
    key.add(v.axes.is_all());
    key.words(v.axes.values());
    key.add(v.keep_dimensions);
    key.add(v.accumulation_type);
    break;
  }
  case OperationKind::kRmsNorm: {
    const auto &v = *operation.get_if<RmsNorm>();
    key.add(v.epsilon);
    key.add(v.accumulation_type);
    break;
  }
  case OperationKind::kLayerNorm: {
    const auto &v = *operation.get_if<LayerNorm>();
    key.add(v.epsilon);
    key.add(v.accumulation_type);
    break;
  }
  case OperationKind::kMatMul:
    key.add(operation.get_if<MatMul>()->accumulation_type);
    break;
  case OperationKind::kGatedActivation: {
    const auto &v = *operation.get_if<GatedActivation>();
    key.add(v.activation);
    key.add(v.inputs);
    break;
  }
  case OperationKind::kGather: {
    const auto &v = *operation.get_if<Gather>();
    key.add(v.axis);
    key.add(v.bounds);
    break;
  }
  case OperationKind::kScatter: {
    const auto &v = *operation.get_if<Scatter>();
    key.add(v.axis);
    key.add(v.kind);
    key.add(v.bounds);
    break;
  }
  case OperationKind::kConcat:
    key.add(operation.get_if<Concat>()->axis);
    break;
  case OperationKind::kRoPE: {
    const auto &v = *operation.get_if<RoPE>();
    key.add(v.rotary_dimensions);
    key.add(v.sequence_axis);
    key.add(v.layout);
    break;
  }
  case OperationKind::kSoftmax: {
    const auto &v = *operation.get_if<Softmax>();
    key.add(v.axis);
    key.add(v.scale);
    key.add(v.kind);
    key.add(v.mask);
    key.add(v.accumulation_type);
    break;
  }
  case OperationKind::kTopK: {
    const auto &v = *operation.get_if<TopK>();
    key.add(v.count);
    key.add(v.axis);
    key.add(v.largest);
    key.add(v.sorted);
    key.add(v.stable);
    break;
  }
  case OperationKind::kScaledDotProductAttention: {
    const auto &v = *operation.get_if<ScaledDotProductAttention>();
    key.add(v.layout);
    key.add(v.mask);
    key.add(v.scale);
    key.add(v.causal);
    key.add(v.accumulation_type);
    break;
  }
  case OperationKind::kCustom:
    key.add(operation.custom_id()->namespace_id);
    key.add(operation.custom_id()->operation_id);
    // Instance attributes belong to CompileProfileContext::semantics.
    break;
  default:
    return Status::Unimplemented("operation has no stable profile encoding");
  }
  return Status::Ok();
}

Status allocation_failure() noexcept {
  return Status::ResourceExhausted("graph profile metadata allocation failed");
}

} // namespace

Status validate_profile_context(CompileEnvironment environment,
                                CompileOptions options) noexcept {
  if (options.profile_context == nullptr) {
    return options.profiles.empty()
               ? Status::Ok()
               : Status::InvalidArgument(
                     "graph profiles require a current context");
  }
  const auto &context = *options.profile_context;
  if (!context.workload.valid() ||
      !std::isfinite(context.minimum_relative_improvement) ||
      context.minimum_relative_improvement < 0 ||
      context.minimum_relative_improvement >= 1)
    return Status::InvalidArgument("graph profile workload/policy is invalid");
  // No provider determinism capability contract exists yet. Do not pretend
  // that a timing-record flag can prove this hard execution requirement.
  if (context.workload.require_determinism)
    return Status::Unimplemented(
        "graph deterministic provider qualification is unavailable");
  if ((!context.targets.empty() &&
       context.targets.size() != environment.targets.size()) ||
      (!context.storage_domains.empty() &&
       context.storage_domains.size() != environment.storage_domains.size()))
    return Status::InvalidArgument(
        "graph profile environment dimensions differ");
  for (std::size_t i = 0; i < context.targets.size(); ++i) {
    if (context.targets[i].backend.valid() &&
        context.targets[i].backend !=
            environment.targets[i].context->device().backend_id())
      return Status::InvalidArgument(
          "graph profile backend does not match its target");
  }
  for (std::size_t i = 0; i < context.providers.size(); ++i) {
    const auto &binding = context.providers[i];
    if (!binding.target.valid() ||
        binding.target.ordinal() >= environment.targets.size())
      return Status::InvalidArgument(
          "graph profile provider target is invalid");
    HYLE_ASSIGN_OR_RETURN(
        const auto instance,
        environment.targets[binding.target.ordinal()]
            .context->provider_instance_id(binding.provider_index));
    if (instance != binding.instance)
      return Status::InvalidArgument("graph profile provider binding is stale");
    for (std::size_t previous = 0; previous < i; ++previous) {
      if (context.providers[previous].target == binding.target &&
          context.providers[previous].provider_index == binding.provider_index)
        return Status::InvalidArgument(
            "graph profile provider binding is duplicated");
    }
  }
  return Status::Ok();
}

Status build_profile_identity(ReportState &report, const PlanState &plan,
                              const GraphState &logical,
                              const CompileProfileContext &context) noexcept {
  auto &storage = report.profile;
  auto &identity = storage.identity;
  identity.hyle_build = context.hyle_build;
  identity.workload_data = context.workload_data;
  identity.workload = context.workload;
  identity.completion = compute::ProfileCompletionBoundary::kGraphTerminal;
  if (!storage.targets.try_resize(context.targets.size()) ||
      !storage.inputs.try_resize(logical.inputs.size() +
                                 logical.constants.size()) ||
      !storage.outputs.try_resize(logical.outputs.size()) ||
      !storage.providers.try_reserve(report.steps.size()))
    return allocation_failure();
  std::copy(context.targets.begin(), context.targets.end(),
            storage.targets.begin());
  std::size_t input = 0;
  for (auto value : logical.inputs)
    storage.inputs[input++] = plan.descriptions[value];
  for (auto value : logical.constants)
    storage.inputs[input++] = plan.descriptions[value];
  for (std::size_t i = 0; i < logical.outputs.size(); ++i)
    storage.outputs[i] = plan.descriptions[logical.outputs[i]];

  Fingerprint semantics(0x4752415048530002ULL);
  semantics.add(context.semantics);
  semantics.add(logical.nodes.size());
  for (const auto &node : logical.nodes) {
    const auto *view = view_spec(logical, node);
    semantics.add(view != nullptr);
    if (view) {
      const auto dimension = [&](Dimension value) noexcept {
        semantics.add(value.is_fixed());
        semantics.add(value.is_fixed()
                          ? static_cast<std::uint64_t>(value.fixed_value())
                          : value.symbol().ordinal());
      };
      semantics.add(compute::OperationKind::kView);
      semantics.add(view->kind());
      semantics.add(view->dimensions().size());
      for (const auto axis : view->dimensions())
        dimension(axis);
      semantics.add(view->axis());
      dimension(view->start());
      dimension(view->count());
      semantics.add(view->step());
      semantics.words(view->permutation());
    } else {
      HYLE_RETURN_IF_ERROR(operation_identity(semantics, node.operation()));
    }
    semantics.add(node.input_begin);
    semantics.add(node.input_count);
    semantics.add(node.output_begin);
    semantics.add(node.output_count);
    semantics.add(node.dependency_begin);
    semantics.add(node.dependency_count);
    semantics.add(node.resource_begin);
  }
  semantics.words(logical.edges);
  semantics.words(logical.dependencies);
  semantics.words(logical.resource_edges);
  semantics.add(logical.effect_resource_count);
  semantics.words(logical.inputs);
  semantics.words(logical.constants);
  semantics.words(logical.outputs);
  // Shape bindings may change View attributes without changing any tensor
  // shape/stride (e.g. two equally sized slices at different starts). The
  // concrete semantic identity must include those values, not only the AST.
  const auto *signature =
      SpecializationSignatureAccess::state(plan.specialization);
  semantics.add(signature != nullptr);
  if (signature) {
    semantics.add(signature->bindings.size());
    for (const auto binding : signature->bindings)
      semantics.add(binding.value);
  }
  semantics.add(plan.descriptions.size());
  for (const auto &description : plan.descriptions)
    HYLE_RETURN_IF_ERROR(semantics.tensor(description));
  if (context.semantics.valid())
    identity.semantics = semantics.finish();

  Fingerprint shape(0x4752415048440001ULL);
  // An envelope-equal exact point must not qualify a complete-domain route.
  shape.add(plan.domain_graph.valid());
  shape.add(logical.dimensions.size());
  for (const auto &dimension : logical.dimensions) {
    shape.add(dimension.kind);
    shape.add(dimension.bounds.minimum);
    shape.add(dimension.bounds.maximum);
    shape.add(dimension.bounds.multiple_of);
    shape.operand(dimension.left);
    shape.operand(dimension.right);
  }
  shape.add(logical.shape_guards.size());
  for (const auto &guard : logical.shape_guards) {
    shape.add(guard.kind);
    shape.operand(guard.left);
    shape.operand(guard.right);
  }
  shape.add(logical.values.size());
  for (const auto &value : logical.values) {
    shape.add(value.spec.kind());
    shape.add(value.spec.data_type());
    shape.format(value.spec.storage_format());
    shape.add(value.spec.rank());
    for (auto axis : value.spec.dimensions()) {
      shape.add(axis.is_fixed());
      shape.add(axis.is_fixed() ? static_cast<std::uint64_t>(axis.fixed_value())
                                : axis.symbol().ordinal());
    }
  }
  identity.shape_domain = shape.finish();

  Fingerprint route(0x4752415048520002ULL);
  bool complete = context.replay_configuration.valid() &&
                  context.storage_domains.size() == plan.storage_domains.size();
  route.add(context.replay_configuration);
  route.add(report.targets.size());
  for (const auto &target : report.targets)
    route.add(target.default_storage.ordinal());
  route.add(report.storage_domains.size());
  for (std::size_t i = 0; i < report.storage_domains.size(); ++i) {
    const auto &domain = report.storage_domains[i];
    if (i < context.storage_domains.size()) {
      route.add(context.storage_domains[i]);
      complete &= context.storage_domains[i].valid();
    }
    route.add(domain.properties.flags);
    route.add(domain.properties.minimum_alignment);
    route.add(domain.properties.maximum_allocation);
    route.add(domain.workspace_bytes);
    route.add(domain.workspace_backing_count);
  }
  route.add(report.lanes.size());
  for (const auto &lane : report.lanes) {
    route.add(lane.target.ordinal());
    route.add(lane.role);
    route.add(lane.runtime_mode);
    route.add(lane.completion_slots_per_invocation);
    route.add(lane.queue_required);
    route.add(lane.asynchronous_queue);
    route.add(lane.capture_supported);
  }
  route.add(report.steps.size());
  for (const auto &step : report.steps) {
    route.add(step.kind);
    route.add(step.target.ordinal());
    route.add(step.lane.ordinal());
    route.add(step.workspace_storage.ordinal());
    route.add(step.source_storage.ordinal());
    route.add(step.destination_storage.ordinal());
    route.add(step.node_begin);
    route.add(step.node_count);
    route.add(step.nodes_contiguous);
    route.add(step.input_begin);
    route.add(step.input_count);
    route.add(step.output_begin);
    route.add(step.output_count);
    route.add(step.workspace.size);
    route.add(step.workspace.alignment);
    route.optional(step.persistent_bytes);
    route.add(step.borrowed_constant_count);
    route.add(step.packed_constant_count);
    route.add(step.packed_constant_bytes);
    route.add(step.copy_bytes);
    route.add(step.synchronization_count);
    route.add(step.dispatch_count);
    route.add(step.runtime.mode);
    route.add(step.runtime.workspace_owner);
    route.add(step.runtime.replay_class);
    route.add(step.runtime.blocks_host);
    route.add(step.runtime.uses_private_threads);
    route.add(step.runtime.supports_concurrent_invocations);
    route.add(step.runtime.capture_compatible);
    if (step.provider.valid()) {
      compute::ProfileProviderIdentity provider{
          .provider = step.provider.id, .implementation = step.profile_key};
      for (const auto &binding : context.providers) {
        if (binding.target == step.target &&
            binding.instance == step.provider_instance) {
          provider.build = binding.build;
          provider.configuration = binding.configuration;
          break;
        }
      }
      if (!storage.providers.try_push_back(provider))
        return allocation_failure();
    }
  }
  route.words(report.step_inputs);
  route.words(report.step_outputs);
  route.words(report.step_node_begins);
  route.add(report.step_nodes.size());
  for (auto node : report.step_nodes)
    route.add(node.ordinal());
  route.add(report.physical_values.size());
  for (const auto &value : report.physical_values) {
    route.add(value.value.ordinal());
    route.add(value.storage.ordinal());
    HYLE_RETURN_IF_ERROR(route.tensor(value.description));
    route.add(value.external_input);
    route.add(value.external_output);
    route.add(value.constant);
    route.add(value.invocation_owned_output);
    route.add(value.workspace_backed);
    route.optional(value.alias_source);
  }
  route.words(plan.alias_offsets);
  route.words(plan.binding_roles);
  route.words(plan.workspace_requests);
  route.words(plan.step_workspace_requests);
  route.words(plan.workspace_storage);
  route.add(plan.workspace_plan.request_count());
  for (std::size_t i = 0; i < plan.workspace_plan.request_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(const auto placement,
                          plan.workspace_plan.placement(i));
    route.add(placement.resource_slot);
    route.add(placement.backing_index);
    route.add(placement.offset);
    route.add(placement.size);
    route.add(placement.alignment);
  }
  route.add(report.constants.size());
  for (const auto &constant : report.constants) {
    route.add(constant.value.ordinal());
    route.add(constant.policy);
    route.add(constant.storage.ordinal());
    route.add(constant.materialized_bytes);
  }
  route.add(plan.constant_artifacts.size());
  route.add(plan.shared_packed_bytes);
  for (const auto &artifact : plan.constant_artifacts) {
    route.add(artifact->key.constant);
    route.add(artifact->key.storage);
    route.add(artifact->key.packing.format);
    route.add(artifact->key.packing.configuration);
    route.add(artifact->packed_bytes);
    route.optional(artifact->private_storage.resident_bytes);
    route.optional(artifact->private_storage.peak_bytes);
    HYLE_RETURN_IF_ERROR(route.tensor(artifact->key.description));
    // Use the qualified external provider identity, never a process-local ID.
    bool qualified = false;
    for (const auto &binding : context.providers) {
      if (binding.target == artifact->target &&
          binding.instance == artifact->key.provider) {
        route.add(binding.build);
        route.add(binding.configuration);
        qualified = binding.build.valid() && binding.configuration.valid();
        break;
      }
    }
    complete &= qualified;
  }
  route.add(report.segments.size());
  for (const auto &segment : report.segments) {
    route.add(segment.lane.ordinal());
    route.add(segment.runtime_mode);
    route.add(segment.step_begin);
    route.add(segment.step_count);
    route.add(segment.dependency_begin);
    route.add(segment.dependency_count);
    route.add(segment.terminal);
    route.add(segment.contains_commands);
    route.add(segment.capture_compatible);
    route.add(segment.capture_supported);
  }
  route.words(report.segment_dependencies);
  route.words(plan.serialization_edges);
  route.words(plan.completion_segments);
  if (complete)
    identity.route = route.finish();
  return Status::Ok();
}

} // namespace hyle::graph::detail
