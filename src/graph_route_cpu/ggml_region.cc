#include "ggml_region.h"

#include <algorithm>
#include <array>
#include <exception>
#include <new>
#include <vector>

#include "hyle/graph/region_provider.h"
#include "hyle/graph/detail/region_access.h"
#include "program.h"
#include "../graph_ggml_shared/constant_packing.h"
#include "../graph/region_discovery.h"

namespace hyle::graph::cpu::internal {
namespace native = compute::ggml_cpu::internal;
namespace {

inline constexpr compute::RoutePolicyId kGgmlCpuRegionPolicy{
    0x48594c4547474350ULL, 1, 1};
inline constexpr std::uint32_t kGgmlCpuRegionPreference = 100;

Result<std::size_t>
matmul_repack_bytes(const Graph &graph,
                    std::span<const NodeId> nodes,
                    std::span<const ValueId> packed_constants = {}) noexcept {
  std::size_t total = 0;
  for (const auto node : nodes) {
    auto operation = graph.operation(node);
    if (!operation)
      return std::unexpected(std::move(operation).error());
    if (operation->get_if<compute::MatMul>() == nullptr)
      continue;
    auto left_id = graph.node_input(node, 0);
    if (!left_id)
      return std::unexpected(std::move(left_id).error());
    auto left = graph.value_description(*left_id);
    if (!left)
      return std::unexpected(std::move(left).error());
    if (left->dimensions().back() == 0)
      continue; // Original zero identity requires neither operand repack.
    auto input = graph.node_input(node, 1);
    if (!input)
      return std::unexpected(std::move(input).error());
    auto description = graph.value_description(*input);
    if (!description)
      return std::unexpected(std::move(description).error());
    auto physical = description->transposed();
    if ((!physical || !physical->is_dense()) &&
        std::ranges::find(packed_constants, *input) == packed_constants.end()) {
      auto bytes = description->required_bytes();
      if (!bytes)
        return std::unexpected(std::move(bytes).error());
      if (*bytes > SIZE_MAX - total)
        return std::unexpected(
            Status::OutOfRange("ggml CPU region repack bytes overflow"));
      total += *bytes;
    }
    auto result_id = graph.result(node);
    if (!result_id)
      return std::unexpected(std::move(result_id).error());
    auto result = graph.value_description(*result_id);
    if (!result)
      return std::unexpected(std::move(result).error());
    const auto expanded = result->element_count().value() /
        static_cast<std::size_t>(result->dimensions().back()) *
        static_cast<std::size_t>(left->dimensions().back());
    if (expanded != left->element_count().value()) {
      const auto element_size = compute::data_type_size(left->data_type());
      if (expanded > (SIZE_MAX - total) / element_size)
        return std::unexpected(
            Status::OutOfRange("ggml CPU region broadcast bytes overflow"));
      total += expanded * element_size;
    }
  }
  return total;
}

Result<bool> supported_node(void *context, const Graph &graph,
                            NodeId node) noexcept {
  auto operation = graph.operation(node);
  if (!operation || !native::region_operation(*operation))
    return false;
  auto input_count = graph.node_input_count(node);
  auto output_count = graph.node_output_count(node);
  auto effects = graph.node_resource_count(node);
  if (!input_count || !output_count || !effects)
    return std::unexpected(
        Status::InvalidArgument("ggml CPU region node is invalid"));
  if (*output_count == 0 || *effects != 0)
    return false;
  try {
    std::vector<compute::TensorDesc> inputs;
    for (std::size_t i = 0; i < *input_count; ++i) {
      auto id = graph.node_input(node, i);
      if (!id)
        return std::unexpected(std::move(id).error());
      auto description = graph.value_description(*id);
      if (!description)
        return std::unexpected(std::move(description).error());
      inputs.push_back(*description);
    }
    std::vector<compute::TensorDesc> outputs;
    for (std::size_t i = 0; i < *output_count; ++i) {
      auto id = graph.result(node, i);
      if (!id)
        return std::unexpected(std::move(id).error());
      auto description = graph.value_description(*id);
      if (!description)
        return std::unexpected(std::move(description).error());
      if (description->element_count().value() == 0)
        return false; // Empty outputs do not bind a nonempty native backing.
      outputs.push_back(*description);
    }
    return native::supports(context, *operation, inputs, outputs);
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("ggml CPU region schema allocation failed"));
  }
}

// A singleton is useful when this adapter can prepare an immutable expert
// bank once. Ordinary dynamic operations keep their existing Direct route.
bool single_constant_expert(const Graph &graph, NodeId node) noexcept {
  const auto operation = graph.operation(node);
  if (!operation || !operation->custom_as<compute::model_ops::GroupedMatMul>())
    return false;
  const auto weight = graph.node_input(node, 0);
  if (!weight) return false;
  const auto description = graph.value_description(*weight);
  if (!description || !description->is_plain() || !description->is_dense() ||
      description->rank() != 3 ||
      description->data_type() != compute::DataType::kBFloat16)
    return false;
  auto value = *weight;
  for (std::size_t depth = 0; depth <= graph.node_count(); ++depth) {
    if (value.kind() == ValueKind::kConstant) return true;
    if (value.kind() != ValueKind::kNodeOutput) return false;
    const auto producer = graph.producer(value);
    if (!producer) return false;
    const auto kind = graph.operation_kind(*producer);
    const auto inputs = graph.node_input_count(*producer);
    if (!kind || *kind != compute::OperationKind::kView || !inputs || *inputs != 1)
      return false;
    const auto input = graph.node_input(*producer, 0);
    if (!input) return false;
    value = *input;
  }
  return false;
}

Status propose(void *context, const RegionDiscoveryRequest &request, mem::AllocatorRef allocator,
               RegionProposalWriter &writer) noexcept {
  const auto &graph = request.graph;
  if (!request.allowed_nodes.empty() && request.allowed_nodes.size() != graph.node_count())
    return Status::InvalidArgument("ggml CPU discovery mask size differs");
  if (graph.node_count() == 0)
    return Status::Ok();
  // Constant-only metadata Views stay original Graph aliases. They may bridge
  // discovery because they cannot depend on work in a proposed native region,
  // but they are never native members or copied/packed by this adapter.
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>
      published_views(allocator), constant_views(allocator);
  container::SmallVector<NodeId, 16, mem::AllocatorRef> members(allocator);
  if (!published_views.try_resize(graph.node_count()) ||
      !constant_views.try_resize(graph.node_count()) ||
      !members.try_resize(graph.node_count()))
    return Status::ResourceExhausted("ggml CPU constant View discovery metadata");
  std::fill(published_views.begin(), published_views.end(), std::uint8_t{0});
  std::fill(constant_views.begin(), constant_views.end(), std::uint8_t{0});
  for (std::size_t port = 0; port < graph.output_count(); ++port) {
    HYLE_ASSIGN_OR_RETURN(auto value, graph.output_id(port));
    if (value.kind() != ValueKind::kNodeOutput) continue;
    HYLE_ASSIGN_OR_RETURN(auto producer, graph.producer(value));
    HYLE_ASSIGN_OR_RETURN(auto kind, graph.operation_kind(producer));
    if (kind == compute::OperationKind::kView)
      published_views[producer.ordinal()] = 1;
  }
  // Propagate the publication fence through View ancestors once, backwards.
  for (std::size_t i = graph.node_count(); i != 0; --i) {
    if (!published_views[i - 1]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i - 1));
    HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
    if (inputs != 1) continue;
    HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, 0));
    if (value.kind() != ValueKind::kNodeOutput) continue;
    HYLE_ASSIGN_OR_RETURN(auto producer, graph.producer(value));
    HYLE_ASSIGN_OR_RETURN(auto kind, graph.operation_kind(producer));
    if (kind == compute::OperationKind::kView)
      published_views[producer.ordinal()] = 1;
  }
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    if (published_views[i] ||
        (!request.allowed_nodes.empty() && !request.allowed_nodes[i]))
      continue;
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    auto operation = graph.operation(node);
    if (!operation || !operation->get_if<compute::View>()) continue;
    HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
    HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
    HYLE_ASSIGN_OR_RETURN(auto effects, graph.node_resource_count(node));
    HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(node));
    if (inputs != 1 || outputs != 1 || effects != 0 || dependencies != 0)
      continue;
    HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, 0));
    if (value.kind() == ValueKind::kConstant) {
      constant_views[i] = 1;
    } else if (value.kind() == ValueKind::kNodeOutput) {
      HYLE_ASSIGN_OR_RETURN(auto producer, graph.producer(value));
      constant_views[i] = constant_views[producer.ordinal()];
    }
  }
  const auto emit = [&](std::span<const NodeId> nodes) -> Result<bool> {
    std::size_t count = 0;
    for (const auto node : nodes)
      if (!constant_views[node.ordinal()]) members[count++] = node;
    nodes = std::span(members).first(count);
    if (nodes.empty() || (nodes.size() == 1 && !single_constant_expert(graph, nodes[0])))
      return true;
    auto repack = matmul_repack_bytes(graph, nodes);
    if (!repack)
      return std::unexpected(std::move(repack).error());
    return writer.try_add(
        {.nodes = nodes,
         .cost = {.workspace = compute::WorkspaceRequirement{},
                  .persistent_bytes = std::nullopt,
                  .copy_bytes = 0,
                  .repack_bytes = *repack,
                  .synchronization_count = 0,
                  .dispatch_count = 1},
         .route = 1,
         .replay_class = compute::ProviderReplayClass::kPrepared,
         .implementation = compute::ProviderImplementationId{2},
         .policy = kGgmlCpuRegionPolicy,
         .preference = kGgmlCpuRegionPreference});
  };
  return detail::discover_regions(graph, allocator,
      [&](NodeId node) -> Result<bool> {
        if (!request.allowed_nodes.empty() && !request.allowed_nodes[node.ordinal()])
          return false;
        if (constant_views[node.ordinal()]) return true;
        return supported_node(context, graph, node);
      }, emit);
}

Status qualify(void *context, const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.route != 1 || request.nodes.empty() ||
      request.outputs.empty() || request.has_unknown_effects ||
      !request.effect_accesses.empty())
    return Status::Unimplemented("ggml CPU region boundary is not admitted");
  if (request.nodes.size() == 1 &&
      !single_constant_expert(*request.graph, request.nodes.front()))
    return Status::Unimplemented("ggml CPU singleton has no immutable preparation");
  for (const auto node : request.nodes) {
    auto supported = supported_node(context, *request.graph, node);
    if (!supported)
      return std::move(supported).error();
    if (!*supported)
      return Status::Unimplemented(
          "ggml CPU region contains an unsupported node");
  }
  return Status::Ok();
}
Status resources(void *context, const RegionCompileRequest &request,
                 mem::AllocatorRef,
                 std::span<RegionInputRequirement> inputs,
                 std::span<RegionOutputRequirement> outputs,
                 RegionResourceProposal *proposal) noexcept {
  *proposal = {};
  HYLE_RETURN_IF_ERROR(qualify(context, request));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = {.resource = {
                     .required_properties = mem::MemoryProperty::kHostAccessible,
                     .minimum_alignment =
                         request.graph->value_description(request.inputs[i])
                             .value()
                             .required_alignment()}};
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = {.resource = {
                      .required_properties = mem::MemoryProperty::kHostAccessible,
                      .minimum_alignment =
                          request.graph->value_description(request.outputs[i])
                              .value()
                              .required_alignment()}};
  *proposal = {
      .runtime = native::runtime(context),
      .workspace_resource = {.required_properties =
                                 mem::MemoryProperty::kHostAccessible}};
  return Status::Ok();
}

struct Artifact final {
  compute::detail::ProviderPrepared native;
};
void destroy(void *value, mem::AllocatorRef allocator) noexcept {
  auto *artifact = static_cast<Artifact *>(value);
  if (!artifact)
    return;
  artifact->native.vtable->destroy_prepared(artifact->native.state, allocator);
  artifact->~Artifact();
  allocator.deallocate(artifact, sizeof(Artifact), alignof(Artifact));
}
compute::AliasSupport alias(void *, std::size_t, std::size_t) noexcept {
  return {};
}
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  const auto &artifact = static_cast<Artifact *>(value)->native;
  return artifact.vtable->create_invocation(artifact.state, allocator,
                                            workspace, output);
}
// Both Direct and Graph use the same invocation callbacks; this table keeps
// only an immutable pointer to that table, never a second execution program.
struct Invocation final {
  const compute::detail::ProviderPreparedVTable *vtable = nullptr;
  void *state = nullptr;
};
Status create_region_invocation(void *value, mem::AllocatorRef allocator,
                                mem::BufferView workspace,
                                void **output) noexcept {
  *output = nullptr;
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return Status::ResourceExhausted(
        "ggml CPU region invocation allocation failed");
  const auto &artifact = static_cast<Artifact *>(value)->native;
  auto *invocation = ::new (storage) Invocation{artifact.vtable, nullptr};
  *output = invocation;
  return create_invocation(value, allocator, workspace, &invocation->state);
}
void destroy_region_invocation(void *value,
                               mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(value);
  if (!invocation)
    return;
  invocation->vtable->destroy_invocation(invocation->state, allocator);
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
compute::detail::EnqueueResult
submit(void *value, void *lane, compute::detail::QueueSequence &sequence,
       std::span<const compute::ConstTensorView> inputs,
       std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = static_cast<Artifact *>(value)->native;
  return artifact.vtable->shared_submit(artifact.state,
                                        static_cast<Invocation *>(lane)->state,
                                        sequence, inputs, outputs);
}
const void *query_extension(void *, compute::ProviderExtensionId) noexcept;
const ProviderRegionPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = &submit,
    .query_extension = &query_extension};

compute::ProviderPrepareOutcome
compile(void *context, const RegionCompileRequest &request,
        mem::AllocatorRef allocator, ProviderRegionPrepared *output) noexcept {
  *output = {};
  auto status = qualify(context, request);
  if (!status.ok())
    return compute::ProviderPrepareOutcome::unsupported(std::move(status));
  try {
    auto repack = matmul_repack_bytes(*request.graph, request.nodes);
    if (!repack)
      return compute::ProviderPrepareOutcome::error(std::move(repack).error());
    native::Recipe recipe;
    std::vector<ValueId> values(request.inputs.begin(), request.inputs.end());
    for (std::uint32_t i = 0; i < values.size(); ++i)
      recipe.inputs.push_back(i);
    for (const auto node : request.nodes)
      for (std::size_t port = 0;
           port < request.graph->node_output_count(node).value(); ++port)
        values.push_back(request.graph->result(node, port).value());
    const auto local = [&](ValueId value) -> std::uint32_t {
      return static_cast<std::uint32_t>(std::ranges::find(values, value) -
                                        values.begin());
    };
    for (const auto id : values)
      recipe.values.push_back(request.graph->value_description(id).value());
    for (const auto node : request.nodes) {
      native::Node lowered{request.graph->operation(node).value(), {}, {}};
      for (std::size_t port = 0;
           port < request.graph->node_output_count(node).value(); ++port)
        lowered.outputs.push_back(
            local(request.graph->result(node, port).value()));
      for (std::size_t i = 0; i < request.graph->node_input_count(node).value();
           ++i) {
        const auto id = local(request.graph->node_input(node, i).value());
        if (id >= values.size())
          return compute::ProviderPrepareOutcome::error(Status::InvalidArgument(
              "ggml CPU region misses an input boundary"));
        lowered.inputs.push_back(id);
      }
      recipe.nodes.push_back(std::move(lowered));
    }
    for (const auto id : request.outputs)
      recipe.outputs.push_back(local(id));
    std::vector<ValueId> packed_constants;
    const auto casts = ggml_shared::prepare_constant_casts(context, request, values, recipe,
        &native::widen_bfloat_constant);
    if (casts.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
      return casts;
    const auto packing = ggml_shared::pack_constants(context, request, values, recipe,
        &native::widen_bfloat_constant, &native::pack_matrix_constant, &packed_constants);
    if (packing.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
      return packing;
    repack = matmul_repack_bytes(*request.graph, request.nodes, packed_constants);
    if (!repack)
      return compute::ProviderPrepareOutcome::error(std::move(repack).error());
    void *storage = allocator.allocate(sizeof(Artifact), alignof(Artifact));
    if (!storage)
      return compute::ProviderPrepareOutcome::error(
          Status::ResourceExhausted(
              "ggml CPU region artifact allocation failed"));
    auto *artifact = ::new (storage) Artifact;
    const auto outcome = native::compile(context, std::move(recipe), allocator,
                                         &artifact->native);
    if (!artifact->native.state) {
      artifact->~Artifact();
      allocator.deallocate(artifact, sizeof(Artifact), alignof(Artifact));
      return outcome;
    }
    *output = {.vtable = &kPrepared,
               .state = artifact,
               .properties = {.workspace = {},
                              .persistent_bytes = std::nullopt,
                              .copy_bytes = *repack,
                              .synchronization_count = 0,
                              .dispatch_count = 1,
                              .runtime = native::runtime(context),
                              .invocation_private_bytes = std::nullopt}};
    return outcome;
  } catch (const std::bad_alloc &) {
    return compute::ProviderPrepareOutcome::error(
        Status::ResourceExhausted("ggml CPU region recipe allocation failed"));
  } catch (const std::exception &error) {
    return compute::ProviderPrepareOutcome::error(
        Status::Internal(error.what()));
  }
}
const RegionProviderVTable kRegion{
    .propose = &propose, .propose_resources = &resources, .compile = &compile};
Status inspect_projected_read(const ProviderRegionPrepared &prepared, void *,
    std::uint32_t input_port, compute::detail::OperationPreparedAccess *access) noexcept {
  if (prepared.vtable != &kPrepared || !prepared.state)
    return Status::Unimplemented("selected CPU region is not this compiled carrier");
  return native::inspect_projected_read(
      static_cast<const Artifact *>(prepared.state)->native, input_port, access);
}
const void *query_extension(void *, compute::ProviderExtensionId id) noexcept {
  static constexpr detail::RegionProjectedReadVTable projected{&inspect_projected_read};
  return id == detail::kRegionProjectedReadExtensionId ? &projected : nullptr;
}
} // namespace

compute::Provider ggml_regions(compute::ggml_cpu::Options options) noexcept {
  static const native::Extension extension{kRegionProviderExtensionId,
                                           &kRegion};
  return native::with_extension(options, &extension);
}
} // namespace hyle::graph::cpu::internal
