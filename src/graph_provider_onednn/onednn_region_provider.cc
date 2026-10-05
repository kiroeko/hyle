#include "hyle/graph/providers/onednn.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <utility>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/allocator.h"
#include "normalization_internal.h"
#include "provider_internal.h"

namespace hyle::graph::onednn {
namespace {

using compute::ConstTensorView;
using compute::DataType;
using compute::Operation;
using compute::ParallelExecutorRef;
using compute::Provider;
using compute::Queue;
using compute::TensorDesc;
using compute::TensorView;

inline constexpr std::uint64_t kNormalizationRoute = 2;
inline constexpr std::uint64_t kRoutePolicyNamespace = 0x48594c454f4e4544ULL;
inline constexpr compute::RoutePolicyId kNormalizationPolicy{
    kRoutePolicyNamespace, 2, 1};
inline constexpr std::uint32_t kNormalizationPreference = 150;
namespace norm = compute::onednn::internal;
struct RegionContext final {
  ParallelExecutorRef executor{};
};

struct ValueUse final {
  std::uint32_t consumers;
  bool published;
  std::uint32_t consumer;
};

struct UseTable final {
  mem::AllocatorRef allocator{};
  ValueUse *values = nullptr;
  std::size_t count = 0;

  UseTable() noexcept = default;
  UseTable(const UseTable &) = delete;
  UseTable &operator=(const UseTable &) = delete;
  ~UseTable() {
    if (values != nullptr)
      allocator.deallocate_n(values, count);
  }
};

Status build_use_table(const Graph &graph, mem::AllocatorRef allocator,
                       UseTable *table) noexcept {
  if (table == nullptr || !allocator.valid())
    return Status::InvalidArgument("oneDNN region use-table owner is invalid");
  table->allocator = allocator;
  table->count = graph.value_count();
  if (table->count == 0)
    return Status::Ok();
  table->values = allocator.allocate_zeroed<ValueUse>(table->count);
  if (table->values == nullptr) {
    return Status::ResourceExhausted(
        "oneDNN region value-use table allocation failed");
  }

  for (std::size_t node_index = 0; node_index < graph.node_count();
       ++node_index) {
    auto node = graph.node_id(node_index);
    if (!node)
      return std::move(node).error();
    auto count = graph.node_input_count(*node);
    if (!count)
      return std::move(count).error();
    for (std::size_t input_index = 0; input_index < *count; ++input_index) {
      auto input = graph.node_input(*node, input_index);
      if (!input)
        return std::move(input).error();
      if (input->ordinal() >= table->count)
        return Status::Internal("graph input ordinal exceeds value table");
      ValueUse &use = table->values[input->ordinal()];
      if (use.consumers == (std::numeric_limits<std::uint32_t>::max)()) {
        return Status::OutOfRange("graph value consumer count overflows");
      }
      ++use.consumers;
      use.consumer = node->ordinal();
    }
  }
  for (std::size_t output_index = 0; output_index < graph.output_count();
       ++output_index) {
    auto output = graph.output_id(output_index);
    if (!output)
      return std::move(output).error();
    if (output->ordinal() >= table->count)
      return Status::Internal("graph output ordinal exceeds value table");
    table->values[output->ordinal()].published = true;
  }
  return Status::Ok();
}

struct NormPattern final {
  std::array<NodeId, norm::kNormMaxPostOps + 1> nodes{};
  std::array<ValueId, norm::kNormMaxInputs> inputs{};
  std::array<TensorDesc, norm::kNormMaxInputs> descriptions{};
  std::array<norm::NormPostOp, norm::kNormMaxPostOps> post_ops{};
  Operation operation{};
  ValueId output{};
  TensorDesc output_description{};
  std::size_t node_count = 0;
  std::size_t norm_inputs = 0;

  std::size_t input_count() const noexcept {
    return norm_inputs + node_count - 1;
  }
  norm::NormRequest request() const noexcept {
    return {operation, std::span(descriptions).first(input_count()),
            output_description, norm_inputs,
            std::span(post_ops).first(node_count - 1)};
  }
};

Status match_norm(const Graph &graph, NodeId node,
                  std::span<const ValueUse> uses, ParallelExecutorRef executor,
                  NormPattern &pattern) noexcept {
  pattern = {};
  auto operation = graph.operation(node);
  if (!operation)
    return std::move(operation).error();
  if (operation->kind() != compute::OperationKind::kRmsNorm &&
      operation->kind() != compute::OperationKind::kLayerNorm)
    return Status::Ok();
  auto count = graph.node_input_count(node);
  auto output = graph.result(node);
  if (!count)
    return std::move(count).error();
  if (!output)
    return std::move(output).error();
  auto description = graph.value_description(*output);
  if (!description)
    return std::move(description).error();
  if (*count == 0 || *count > 3 ||
      description->data_type() != DataType::kFloat32)
    return Status::Ok();
  pattern.operation = *operation;
  pattern.norm_inputs = *count;
  pattern.nodes[0] = node;
  pattern.node_count = 1;
  pattern.output = *output;
  pattern.output_description = *description;
  for (std::size_t index = 0; index < *count; ++index) {
    auto input = graph.node_input(node, index);
    if (!input)
      return std::move(input).error();
    auto input_description = graph.value_description(*input);
    if (!input_description)
      return std::move(input_description).error();
    pattern.inputs[index] = *input;
    pattern.descriptions[index] = *input_description;
  }
  Status status = norm::validate_normalization(pattern.request(), executor);
  if (!status.ok()) {
    pattern.node_count = 0;
    return status.IsUnimplemented() ? Status::Ok() : std::move(status);
  }
  while (pattern.node_count < pattern.nodes.size()) {
    const ValueUse &use = uses[pattern.output.ordinal()];
    if (use.consumers != 1 || use.published)
      break;
    auto consumer = graph.node_id(use.consumer);
    if (!consumer)
      return std::move(consumer).error();
    auto next_operation = graph.operation(*consumer);
    if (!next_operation)
      return std::move(next_operation).error();
    const auto *binary = next_operation->get_if<compute::Binary>();
    if (binary == nullptr || (binary->kind != compute::BinaryKind::kAdd &&
                              binary->kind != compute::BinaryKind::kMultiply))
      break;
    auto left = graph.node_input(*consumer, 0);
    auto right = graph.node_input(*consumer, 1);
    auto next_output = graph.result(*consumer);
    if (!left)
      return std::move(left).error();
    if (!right)
      return std::move(right).error();
    if (!next_output)
      return std::move(next_output).error();
    const ValueId rhs = *left == pattern.output ? *right : *left;
    auto rhs_description = graph.value_description(rhs);
    auto next_description = graph.value_description(*next_output);
    if (!rhs_description)
      return std::move(rhs_description).error();
    if (!next_description)
      return std::move(next_description).error();
    if (*next_description != pattern.output_description)
      break;
    NormPattern candidate = pattern;
    const std::size_t input = candidate.input_count();
    candidate.inputs[input] = rhs;
    candidate.descriptions[input] = *rhs_description;
    candidate.post_ops[candidate.node_count - 1] = {binary->kind, input};
    candidate.nodes[candidate.node_count++] = *consumer;
    candidate.output = *next_output;
    status = norm::validate_normalization(candidate.request(), executor);
    if (status.IsUnimplemented())
      break;
    if (!status.ok())
      return status;
    pattern = candidate;
  }
  return Status::Ok();
}

Status propose_norms(const Graph &graph, std::span<const ValueUse> uses,
                     ParallelExecutorRef executor, mem::AllocatorRef allocator,
                     RegionProposalWriter &writer) noexcept {
  for (std::size_t index = 0; index < graph.node_count(); ++index) {
    auto node = graph.node_id(index);
    if (!node)
      return std::move(node).error();
    NormPattern pattern;
    Status status = match_norm(graph, *node, uses, executor, pattern);
    if (!status.ok())
      return status;
    if (pattern.node_count < 2)
      continue;
    const auto nodes = std::span(pattern.nodes).first(pattern.node_count);
    auto contractible = is_region_contractible(graph, nodes, allocator);
    if (!contractible)
      return std::move(contractible).error();
    if (!*contractible)
      continue;
    auto added = writer.try_add(
        {.nodes = nodes,
         .cost = {.workspace = std::nullopt,
                  .persistent_bytes = std::nullopt,
                  .copy_bytes = 0,
                  .repack_bytes = 0,
                  .synchronization_count = 0,
                  .dispatch_count = 1},
         .route = kNormalizationRoute,
         .replay_class = norm::kNormRuntime.replay_class,
         .implementation =
             compute::ProviderImplementationId{kNormalizationRoute},
         .policy = kNormalizationPolicy,
         .preference = kNormalizationPreference});
    if (!added)
      return std::move(added).error();
    if (!*added)
      break;
  }
  return Status::Ok();
}

Status propose_regions(void *context_value, const Graph &graph,
                       mem::AllocatorRef allocator,
                       RegionProposalWriter &writer) noexcept {
  if (context_value == nullptr || !allocator.valid())
    return Status::Internal("oneDNN region context is invalid");
  UseTable table;
  Status status = build_use_table(graph, allocator, &table);
  if (!status.ok())
    return status;
  const std::span<const ValueUse> uses(table.values, table.count);
  return propose_norms(graph, uses,
                       static_cast<RegionContext *>(context_value)->executor,
                       allocator, writer);
}

struct NormRegionPlan final {
  compute::detail::ProviderPrepared primitive{};
  std::array<std::size_t, norm::kNormMaxInputs> inputs{};
  std::size_t input_count = 0;
  std::size_t boundary_count = 0;
  ProviderRegionPreparedVTable vtable{};
};

void destroy_norm_region(void *value, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<NormRegionPlan *>(value);
  plan->primitive.vtable->destroy_prepared(plan->primitive.state, allocator);
  plan->~NormRegionPlan();
  allocator.deallocate(plan, sizeof(NormRegionPlan), alignof(NormRegionPlan));
}

compute::AliasSupport norm_region_alias(void *value, std::size_t output,
                                        std::size_t input) noexcept {
  const auto &plan = *static_cast<NormRegionPlan *>(value);
  return {.supported = output == 0 && input == plan.inputs[0]};
}

Status create_norm_invocation(void *value, mem::AllocatorRef allocator,
                              mem::BufferView workspace,
                              void **invocation) noexcept {
  auto &prepared = static_cast<NormRegionPlan *>(value)->primitive;
  return prepared.vtable->create_invocation(prepared.state, allocator,
                                            workspace, invocation);
}

// The primitive owns the invocation. Its destroy function is copied into the
// immutable region vtable; a region needs no additional invocation wrapper.
compute::detail::EnqueueResult
submit_norm(void *value, void *invocation, Queue &queue,
            std::span<const ConstTensorView> inputs,
            std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<NormRegionPlan *>(value);
  if (inputs.size() != plan.boundary_count || outputs.size() != 1)
    return compute::detail::EnqueueResult::accepted_command(
        Status::Internal("oneDNN norm region bindings are invalid"));
  std::array<ConstTensorView, norm::kNormMaxInputs> ordered{};
  for (std::size_t index = 0; index < plan.input_count; ++index)
    ordered[index] = inputs[plan.inputs[index]];
  return plan.primitive.vtable->shared_submit(
      plan.primitive.state, invocation, queue,
      std::span(ordered).first(plan.input_count), outputs);
}

compute::ProviderPrepareOutcome
compile_norm(RegionContext &context, const RegionCompileRequest &request,
             mem::AllocatorRef allocator,
             ProviderRegionPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm region output is null"));
  *prepared_output = {};
  if (request.graph == nullptr || request.nodes.size() < 2 ||
      request.nodes.size() > norm::kNormMaxPostOps + 1 ||
      request.outputs.size() != 1 ||
      request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != 1 ||
      !request.workspace_resource.valid())
    return compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm region request is invalid"));
  UseTable table;
  Status status = build_use_table(*request.graph, allocator, &table);
  if (!status.ok())
    return compute::ProviderPrepareOutcome::error(std::move(status));
  NormPattern pattern;
  status = match_norm(*request.graph, request.nodes.front(),
                      std::span(table.values, table.count), context.executor,
                      pattern);
  if (!status.ok())
    return compute::ProviderPrepareOutcome::error(std::move(status));
  if (pattern.node_count < 2 ||
      !std::ranges::equal(std::span(pattern.nodes).first(pattern.node_count),
                          request.nodes) ||
      pattern.output != request.outputs[0])
    return compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("oneDNN norm region boundaries do not match"));
  std::array<std::size_t, norm::kNormMaxInputs> bindings{};
  for (std::size_t index = 0; index < pattern.input_count(); ++index) {
    auto found = std::ranges::find(request.inputs, pattern.inputs[index]);
    if (found == request.inputs.end())
      return compute::ProviderPrepareOutcome::error(
          Status::InvalidArgument("oneDNN norm region input is missing"));
    bindings[index] = static_cast<std::size_t>(found - request.inputs.begin());
  }
  for (const auto input : request.inputs) {
    if (std::ranges::find(
            std::span(pattern.inputs).first(pattern.input_count()), input) ==
        std::span(pattern.inputs).first(pattern.input_count()).end())
      return compute::ProviderPrepareOutcome::error(
          Status::InvalidArgument("oneDNN norm region has an unrelated input"));
  }
  compute::detail::ProviderPrepared primitive;
  auto outcome = norm::prepare_normalization(
      pattern.request(), context.executor, allocator, &primitive);
  if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
    return outcome;
  void *storage =
      allocator.allocate(sizeof(NormRegionPlan), alignof(NormRegionPlan));
  if (storage == nullptr) {
    primitive.vtable->destroy_prepared(primitive.state, allocator);
    return compute::ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted(
            "oneDNN norm region wrapper allocation failed"));
  }
  auto *region = ::new (storage) NormRegionPlan{
      .primitive = primitive,
      .inputs = bindings,
      .input_count = pattern.input_count(),
      .boundary_count = request.inputs.size(),
      .vtable = {.destroy_prepared = &destroy_norm_region,
                 .alias_support = &norm_region_alias,
                 .create_invocation = &create_norm_invocation,
                 .destroy_invocation = primitive.vtable->destroy_invocation,
                 .shared_submit = &submit_norm}};
  *prepared_output = {.vtable = &region->vtable,
                      .state = region,
                      .properties = {.workspace = primitive.workspace,
                                     .persistent_bytes = std::nullopt,
                                     .dispatch_count = 1,
                                     .runtime = norm::kNormRuntime,
                                     .invocation_private_bytes = std::nullopt}};
  return compute::ProviderPrepareOutcome::prepared();
}

Status propose_region_resources(
    void *context_value, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<compute::detail::ResourceBindingRequirement> input_requirements,
    std::span<compute::detail::ResourceBindingRequirement> output_requirements,
    RegionResourceProposal *proposal) noexcept {
  if (request.route == kNormalizationRoute) {
    if (context_value == nullptr || request.graph == nullptr ||
        proposal == nullptr || request.nodes.size() < 2 ||
        request.nodes.size() > norm::kNormMaxPostOps + 1 ||
        request.outputs.size() != 1 ||
        input_requirements.size() != request.inputs.size() ||
        output_requirements.size() != 1)
      return Status::InvalidArgument(
          "oneDNN norm resource proposal is invalid");
    for (std::size_t index = 0; index < request.inputs.size(); ++index) {
      auto description =
          request.graph->value_description(request.inputs[index]);
      if (!description)
        return std::move(description).error();
      input_requirements[index] = {
          .required_properties = mem::MemoryProperty::kHostAccessible,
          .minimum_alignment = description->required_alignment()};
    }
    auto description = request.graph->value_description(request.outputs[0]);
    if (!description)
      return std::move(description).error();
    output_requirements[0] = {
        .required_properties = mem::MemoryProperty::kHostAccessible,
        .minimum_alignment = description->required_alignment()};
    *proposal = {
        .runtime = norm::kNormRuntime,
        .workspace_resource = {
            .required_properties = mem::MemoryProperty::kHostAccessible,
            .minimum_alignment = norm::kNormWorkspaceAlignment}};
    return Status::Ok();
  }
  return Status::Unimplemented("oneDNN region route is not qualified");
}

compute::ProviderPrepareOutcome
compile_region(void *context, const RegionCompileRequest &request,
               mem::AllocatorRef allocator,
               ProviderRegionPrepared *prepared) noexcept {
  if (request.route == kNormalizationRoute && context != nullptr)
    return compile_norm(*static_cast<RegionContext *>(context), request,
                        allocator, prepared);
  return compute::ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("oneDNN region route is not qualified"));
}

const RegionProviderVTable kRegionVTable{
    .propose = &propose_regions,
    .propose_resources = &propose_region_resources,
    .compile = &compile_region,
};

Status create_context(const Provider &provider_value, compute::Device device,
                      void *, mem::AllocatorRef allocator,
                      void **context_output) noexcept {
  if (context_output == nullptr)
    return Status::InvalidArgument("oneDNN region context output is null");
  *context_output = nullptr;
  if (device.device_class() != compute::DeviceClass::kCpu) {
    return Status::Unimplemented(
        "oneDNN graph region provider requires a CPU device");
  }
  if (!allocator.valid())
    return Status::FailedPrecondition("oneDNN region allocator is invalid");
  void *storage =
      allocator.allocate(sizeof(RegionContext), alignof(RegionContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("oneDNN region context allocation failed");
  const Options options =
      compute::detail::ProviderAccess::payload<Options>(provider_value);
  *context_output = ::new (storage) RegionContext{.executor = options.executor};
  return Status::Ok();
}

void destroy_context(void *context_value,
                     mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<RegionContext *>(context_value);
  if (context == nullptr)
    return;
  context->~RegionContext();
  allocator.deallocate(context, sizeof(RegionContext), alignof(RegionContext));
}

Status propose_single(void *, const compute::detail::ProviderOperationRequest &,
                      std::span<compute::detail::ResourceBindingRequirement>,
                      std::span<compute::detail::ResourceBindingRequirement>,
                      compute::detail::ProviderOperationProposal *) noexcept {
  return Status::Unimplemented(
      "oneDNN graph region provider has no single-operation route");
}

compute::ProviderPrepareOutcome
prepare_single(void *, const compute::detail::ProviderPrepareRequest &,
               mem::AllocatorRef,
               compute::detail::ProviderPrepared *) noexcept {
  return compute::ProviderPrepareOutcome::unsupported(Status::Unimplemented(
      "oneDNN graph region provider has no single-operation route"));
}

Status drain_context(void *) noexcept { return Status::Ok(); }

const void *query_extension(void *, compute::ProviderExtensionId id) noexcept {
  return id == kRegionProviderExtensionId ? &kRegionVTable : nullptr;
}

const compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = kProviderId,
                 .name = "onednn-graph-region",
                 .version = "3.13.2"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose_single,
    .prepare = &prepare_single,
    .drain_context = &drain_context,
    .query_extension = &query_extension,
};

} // namespace

compute::Provider provider(Options options) noexcept {
  return compute::detail::ProviderAccess::make(&kProviderVTable, options);
}

} // namespace hyle::graph::onednn
