#include "ggml_region.h"

#include <algorithm>
#include <array>
#include <exception>
#include <new>
#include <vector>

#include "hyle/graph/region_provider.h"
#include "program.h"

namespace hyle::graph::cpu::internal {
namespace native = compute::ggml_cpu::internal;
namespace {

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
  if (*output_count != 1 || *effects != 0)
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
      if (description->element_count().value() == 0)
        return false; // Zero-size leaves do not need a native execution
                      // episode.
      inputs.push_back(*description);
    }
    auto id = graph.result(node);
    if (!id)
      return std::unexpected(std::move(id).error());
    auto output = graph.value_description(*id);
    if (!output)
      return std::unexpected(std::move(output).error());
    return native::supports(context, *operation, inputs,
                            std::span(&*output, 1));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("ggml CPU region schema allocation failed"));
  }
}

Status propose(void *context, const Graph &graph, mem::AllocatorRef allocator,
               RegionProposalWriter &writer) noexcept {
  struct Scratch final {
    mem::AllocatorRef allocator;
    NodeId *data;
    std::size_t capacity;
    ~Scratch() { allocator.deallocate_n(data, capacity); }
  } scratch{allocator, nullptr, graph.node_count()};
  if (scratch.capacity < 2)
    return Status::Ok();
  scratch.data = allocator.allocate_n<NodeId>(scratch.capacity);
  if (!scratch.data)
    return Status::ResourceExhausted(
        "ggml CPU region discovery allocation failed");
  std::size_t size = 0;
  const auto emit = [&]() -> Result<bool> {
    if (size < 2)
      return true;
    const auto nodes = std::span(scratch.data, size);
    auto contractible = is_region_contractible(graph, nodes, allocator);
    if (!contractible)
      return std::unexpected(std::move(contractible).error());
    if (!*contractible)
      return true;
    return writer.try_add(
        {.nodes = nodes,
         .cost = {.workspace = compute::WorkspaceRequirement{},
                  .persistent_bytes = std::nullopt,
                  .copy_bytes = 0,
                  .repack_bytes = 0,
                  .synchronization_count = 0,
                  .dispatch_count = 1},
         .route = 1,
         .replay_class = compute::ProviderReplayClass::kPrepared,
         .profile_key = compute::ProviderRouteKey{2}});
  };
  // An independent node between two related ggml nodes does not
  // break their region. The DAG contraction check, not insertion adjacency,
  // decides whether the eligible set has a legal native boundary. A set that
  // leaves and re-enters retains the existing single-operation routes.
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    auto node = graph.node_id(i);
    if (!node)
      return std::move(node).error();
    auto supported = supported_node(context, graph, *node);
    if (!supported)
      return std::move(supported).error();
    if (*supported)
      scratch.data[size++] = *node;
  }
  auto accepted = emit();
  if (!accepted)
    return std::move(accepted).error();
  return Status::Ok();
}

Status qualify(void *context, const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.route != 1 || request.nodes.size() < 2 ||
      request.outputs.empty() || request.has_unknown_effects ||
      !request.effect_accesses.empty())
    return Status::Unimplemented("ggml CPU region boundary is not admitted");
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
                 std::span<compute::detail::ResourceBindingRequirement> inputs,
                 std::span<compute::detail::ResourceBindingRequirement> outputs,
                 RegionResourceProposal *proposal) noexcept {
  *proposal = {};
  HYLE_RETURN_IF_ERROR(qualify(context, request));
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = {.required_properties = mem::MemoryProperty::kHostAccessible,
                 .minimum_alignment =
                     request.graph->value_description(request.inputs[i])
                         .value()
                         .required_alignment()};
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = {.required_properties = mem::MemoryProperty::kHostAccessible,
                  .minimum_alignment =
                      request.graph->value_description(request.outputs[i])
                          .value()
                          .required_alignment()};
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
submit(void *value, void *lane, compute::Queue &queue,
       std::span<const compute::ConstTensorView> inputs,
       std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = static_cast<Artifact *>(value)->native;
  return artifact.vtable->shared_submit(artifact.state,
                                        static_cast<Invocation *>(lane)->state,
                                        queue, inputs, outputs);
}
const ProviderRegionPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = &submit};

compute::ProviderPrepareOutcome
compile(void *context, const RegionCompileRequest &request,
        mem::AllocatorRef allocator, ProviderRegionPrepared *output) noexcept {
  *output = {};
  auto status = qualify(context, request);
  if (!status.ok())
    return compute::ProviderPrepareOutcome::unsupported(std::move(status));
  try {
    native::Recipe recipe;
    std::vector<ValueId> values(request.inputs.begin(), request.inputs.end());
    for (std::uint32_t i = 0; i < values.size(); ++i)
      recipe.inputs.push_back(i);
    for (const auto node : request.nodes)
      values.push_back(request.graph->result(node).value());
    const auto local = [&](ValueId value) -> std::uint32_t {
      return static_cast<std::uint32_t>(std::ranges::find(values, value) -
                                        values.begin());
    };
    for (const auto id : values)
      recipe.values.push_back(request.graph->value_description(id).value());
    for (const auto node : request.nodes) {
      native::Node lowered{request.graph->operation(node).value(),
                           {},
                           local(request.graph->result(node).value())};
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
    void *storage = allocator.allocate(sizeof(Artifact), alignof(Artifact));
    if (!storage)
      return compute::ProviderPrepareOutcome::resource_declined(
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
                              .copy_bytes = 0,
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
} // namespace

compute::Provider ggml_regions(compute::ggml_cpu::Options options) noexcept {
  static const native::Extension extension{kRegionProviderExtensionId,
                                           &kRegion};
  return native::with_extension(options, &extension);
}
} // namespace hyle::graph::cpu::internal
