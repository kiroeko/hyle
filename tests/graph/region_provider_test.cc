#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "backend_internal.h"
#include "completion_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/buffer.h"
#include "provider_internal.h"
#include "test_environment.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::compute::AliasSupport;
using hyle::compute::Completion;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::Operation;
using hyle::compute::Provider;
using hyle::compute::ProviderPrepareOutcome;
using hyle::compute::ProviderPrepareOutcomeKind;
using hyle::compute::ProviderReplayClass;
using hyle::compute::Queue;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::graph::CompileFallbackReason;
using hyle::graph::CompileOptions;
using hyle::graph::CompileReport;
using hyle::graph::CompileSearchPolicy;
using hyle::graph::CompileStepKind;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::NodeId;
using hyle::graph::OperationFallbackPolicy;
using hyle::graph::ProviderRegionPrepared;
using hyle::graph::ProviderReplayPolicy;
using hyle::graph::RegionCompileRequest;
using hyle::graph::RegionProposal;
using hyle::graph::RegionProposalWriter;
using hyle::graph::RegionSelectionPolicy;
using hyle::graph::ValueId;
using hyle::graph::test::Environment;
using hyle::graph::test::ExecutionSpec;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

constexpr hyle::compute::ProviderId kRegionProviderId{0x544553542d475241ULL,
                                                      0x50482d524547494fULL};

enum class ProposalMode : std::uint8_t {
  kPairs = 0,
  kOverlappingRoutes,
  kNonContiguous,
  kNonContiguousReordered,
  kSingleton,
  kInterleavedChains,
  kJointCycle,
  kDuplicateNode,
  kForeignNode,
  kOverflow,
  kCompileFailure,
  kInvocationFailure,
  kAsyncPairs,
  kAsyncRejectSecond,
};

struct RegionProbe final {
  ProposalMode mode = ProposalMode::kPairs;
  std::optional<std::uint32_t> proposal_dispatch_count = 1;
  std::size_t propose_count = 0;
  std::size_t compile_count = 0;
  std::size_t single_prepare_count = 0;
  std::size_t single_artifact_count = 0;
  std::size_t region_resource_count = 0;
  std::size_t context_destroy_count = 0;
  std::size_t create_invocation_count = 0;
  std::size_t destroy_invocation_count = 0;
  std::size_t submit_count = 0;
  std::size_t submit_attempt_count = 0;
  std::size_t completion_poll_count = 0;
  std::size_t completion_wait_count = 0;
  std::size_t completion_destroy_count = 0;
  std::size_t device_wait_count = 0;
  std::size_t reserve_completion_count = 0;
  std::size_t last_reserved_completions = 0;
  std::size_t transfer_query_count = 0;
  std::size_t capture_begin_count = 0;
  std::size_t capture_end_count = 0;
  std::size_t capture_abort_count = 0;
  std::size_t capture_launch_count = 0;
  std::size_t capture_destroy_count = 0;
  std::size_t destroy_prepared_count = 0;
  std::size_t effect_create_count = 0;
  std::size_t effect_submit_count = 0;
  std::size_t completed_prefix = 0;
  bool cancelled = false;
  bool terminal_failure = false;
  bool asynchronous_backend = false;
  bool delegated_single = false;
  bool transfer_supported = true;
  bool dependency_wait_supported = true;
  bool capture_supported = false;
  bool capture_active = false;
  bool decline_regions = false;
  hyle::mem::MemoryResourceRef required_resource{};
  hyle::mem::MemoryResourceRef region_input_resource{};
  hyle::mem::MemoryResourceRef region_output_resource{};
  hyle::mem::MemoryResourceRef region_workspace_resource{};
  hyle::mem::MemoryResourceRef compiled_input_resource{};
  hyle::mem::MemoryResourceRef compiled_output_resource{};
  hyle::mem::MemoryResourceRef compiled_workspace_resource{};
  bool invocation_workspace_valid = false;
  std::size_t invocation_workspace_size = 0;
  hyle::compute::detail::CompletionLifetime abandoned_lifetime{};
  Status abandoned_hint{};
  hyle::graph::RegionArtifactProperties artifact_properties{};
  std::optional<hyle::compute::ProviderReplayClass> proposed_replay_override{};
  ProviderReplayClass discovery_replay_class = ProviderReplayClass::kUnknown;
  std::optional<ProviderReplayClass> single_replay_override{};
  NodeId foreign_node{};
  std::uint32_t compiled_begin = (std::numeric_limits<std::uint32_t>::max)();
  std::uint32_t compiled_count = 0;
  Status discovery_failure{};
  std::optional<ProviderPrepareOutcome> region_outcome_override{};
  std::optional<ProviderPrepareOutcome> single_outcome_override{};
  std::size_t destroy_count_at_single_proposal = 0;
};

struct ExternalEffect final {
  static constexpr hyle::compute::OperationId operation_id{0x48594c45U,
                                                           0x45464654U};
  static constexpr std::string_view operation_name = "test-external-effect";

  hyle::compute::OperationTraits operation_traits() const noexcept {
    return {{0, 0},
            0,
            hyle::compute::AliasPolicy::kNone,
            hyle::compute::EffectKind::kExternalSideEffect,
            hyle::compute::PrecisionPolicy::kExact,
            DataType::kUnknown,
            false};
  }

  Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    return inputs.empty() && outputs.empty()
               ? Status::Ok()
               : Status::InvalidArgument("effect operation has no tensors");
  }
};

struct ProviderPayload final {
  RegionProbe *probe = nullptr;
};

struct RegionPlan final {
  RegionProbe *probe = nullptr;
  std::uint32_t negation_count = 0;
};

struct RegionRun final {
  RegionProbe *probe = nullptr;
};

Status async_terminal(const RegionProbe &probe) noexcept {
  if (probe.cancelled)
    return Status::Cancelled("region sequence was cancelled");
  return probe.terminal_failure
             ? Status::Internal("injected asynchronous terminal failure")
             : Status::Ok();
}

void finish_abandoned_prefix(RegionProbe &probe) {
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  probe.completed_prefix = probe.submit_count;
  Status terminal = async_terminal(probe);
  if (terminal.ok() && !probe.abandoned_hint.ok())
    terminal = probe.abandoned_hint;
  const auto lifetime = probe.abandoned_lifetime;
  probe.abandoned_lifetime = {};
  probe.abandoned_hint = {};
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

void destroy_async_completion(
    void *state, hyle::compute::detail::CompletionLifetime lifetime,
    Status terminal_hint) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  ++probe.completion_destroy_count;
  if (!lifetime.valid())
    return;
  if (probe.completed_prefix < probe.submit_count) {
    probe.abandoned_lifetime = lifetime;
    probe.abandoned_hint = std::move(terminal_hint);
    return;
  }
  Status terminal = async_terminal(probe);
  if (terminal.ok() && !terminal_hint.ok())
    terminal = std::move(terminal_hint);
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

Result<hyle::compute::CompletionPoll>
poll_async_completion(void *state) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  ++probe.completion_poll_count;
  if (probe.completed_prefix < probe.submit_count)
    return hyle::compute::CompletionPoll::kPending;
  const Status terminal = async_terminal(probe);
  if (!terminal.ok())
    return std::unexpected(terminal);
  return hyle::compute::CompletionPoll::kComplete;
}

Status wait_async_completion(void *state) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  ++probe.completion_wait_count;
  probe.completed_prefix = probe.submit_count;
  return async_terminal(probe);
}

Status cancel_async_completion(void *state) noexcept {
  auto &probe = *static_cast<RegionProbe *>(state);
  probe.cancelled = true;
  probe.completed_prefix = probe.submit_count;
  return Status::Ok();
}

const hyle::compute::detail::CompletionVTable kAsyncCompletionVTable{
    .destroy = &destroy_async_completion,
    .poll = &poll_async_completion,
    .wait = &wait_async_completion,
    .request_cancel = &cancel_async_completion,
};

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

class HostResource final : public hyle::mem::MemoryResource {
public:
  std::string_view name() const noexcept override {
    return "graph.region-test.host";
  }

  hyle::mem::MemoryProperties properties() const noexcept override {
    return hyle::mem::MemoryProperties{
        .flags = hyle::mem::MemoryProperty::kHostAccessible |
                 hyle::mem::MemoryProperty::kHostDeviceCoherent,
        .domain = hyle::mem::MemoryDomainId{this, 0},
    };
  }

  Result<hyle::mem::ResourceAllocation>
  allocate(std::size_t size, std::size_t alignment) noexcept override {
    return hyle::mem::system_resource().resource().allocate(size, alignment);
  }

  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    hyle::mem::system_resource().resource().deallocate(allocation);
  }

  Result<hyle::mem::ResourceHostMapping>
  map_host(void *native_handle, std::size_t offset, std::size_t size,
           hyle::mem::HostAccess) noexcept override {
    return hyle::mem::ResourceHostMapping{
        size == 0 ? nullptr
                  : static_cast<void *>(
                        static_cast<std::byte *>(native_handle) + offset),
        nullptr,
    };
  }
};

OwnedTensor make_tensor(TensorDesc description) {
  OwnedTensor result;
  result.buffer = Buffer::create(hyle::mem::system_resource(),
                                 description.required_bytes().value(), 64)
                      .value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
}

OwnedTensor make_tensor(TensorDesc description,
                        hyle::mem::MemoryResourceRef resource) {
  OwnedTensor result;
  result.buffer = Buffer::create(resource, description.required_bytes().value(),
                                 64)
                      .value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
}

void write(OwnedTensor &tensor, std::span<const float> values) {
  ASSERT_TRUE(tensor.buffer.view().write(std::as_bytes(values)).ok());
}

std::vector<float> read(const OwnedTensor &tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(tensor.buffer.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

bool is_negation(const Graph &graph, std::size_t node_index) {
  auto node = graph.node_id(node_index);
  if (!node)
    return false;
  auto operation = graph.operation(*node);
  if (!operation)
    return false;
  const Unary *unary = operation->get_if<Unary>();
  return unary != nullptr && unary->kind == UnaryKind::kNeg;
}

Result<bool> add_node_set_proposal(
    RegionProposalWriter &writer, std::span<const NodeId> nodes,
    std::optional<std::uint32_t> dispatch_count = 1,
    ProviderReplayClass replay_class = ProviderReplayClass::kUnknown) noexcept {
  return writer.try_add(RegionProposal{
      .nodes = nodes,
      .cost = {.dispatch_count = dispatch_count},
      .route =
          nodes.empty()
              ? 0
              : (static_cast<std::uint64_t>(nodes.front().ordinal()) << 32) |
                    nodes.size(),
      .replay_class = replay_class,
  });
}

Result<bool>
add_proposal(const Graph &graph, RegionProposalWriter &writer,
             std::uint32_t begin, std::uint32_t count,
             std::optional<std::uint32_t> dispatch_count = 1) noexcept {
  if (count != 2)
    return std::unexpected(
        Status::InvalidArgument("test region proposal count is not two"));
  auto first = graph.node_id(begin);
  auto second = graph.node_id(begin + 1);
  if (!first)
    return std::unexpected(std::move(first).error());
  if (!second)
    return std::unexpected(std::move(second).error());
  const std::array nodes{*first, *second};
  return add_node_set_proposal(writer, nodes, dispatch_count);
}

Result<bool> discover_test_regions(RegionProbe &probe, const Graph &graph,
                                   RegionProposalWriter &writer) noexcept {
  ++probe.propose_count;
  if (probe.mode == ProposalMode::kSingleton) {
    const std::array nodes{graph.node_id(0).value()};
    return add_node_set_proposal(writer, nodes, probe.proposal_dispatch_count,
                                 probe.discovery_replay_class);
  }
  if (probe.mode == ProposalMode::kInterleavedChains ||
      probe.mode == ProposalMode::kJointCycle) {
    const std::array first{graph.node_id(0).value(), graph.node_id(2).value()};
    const std::array second{graph.node_id(1).value(), graph.node_id(3).value()};
    auto added = add_node_set_proposal(writer, first);
    if (!added || !*added)
      return added;
    return add_node_set_proposal(writer, second);
  }
  if (probe.mode == ProposalMode::kOverflow) {
    auto first = add_proposal(graph, writer, 0, 2);
    if (!first || !*first)
      return first;
    return add_proposal(graph, writer, 0, 2);
  }
  if (probe.mode == ProposalMode::kOverlappingRoutes) {
    auto first = add_proposal(graph, writer, 0, 2, 2);
    if (!first || !*first)
      return first;
    return add_proposal(graph, writer, 1, 2, 1);
  }
  if (probe.mode == ProposalMode::kNonContiguous) {
    auto first = graph.node_id(0);
    auto last = graph.node_id(2);
    if (!first)
      return std::unexpected(std::move(first).error());
    if (!last)
      return std::unexpected(std::move(last).error());
    const std::array nodes{*last, *first};
    return add_node_set_proposal(writer, nodes, probe.proposal_dispatch_count);
  }
  if (probe.mode == ProposalMode::kNonContiguousReordered) {
    auto first = graph.node_id(0);
    auto last = graph.node_id(3);
    if (!first)
      return std::unexpected(std::move(first).error());
    if (!last)
      return std::unexpected(std::move(last).error());
    const std::array nodes{*first, *last};
    return add_node_set_proposal(writer, nodes, probe.proposal_dispatch_count);
  }
  if (probe.mode == ProposalMode::kDuplicateNode) {
    auto node = graph.node_id(0);
    if (!node)
      return std::unexpected(std::move(node).error());
    const std::array nodes{*node, *node};
    return add_node_set_proposal(writer, nodes);
  }
  if (probe.mode == ProposalMode::kForeignNode) {
    auto local = graph.node_id(1);
    if (!local)
      return std::unexpected(std::move(local).error());
    const std::array nodes{probe.foreign_node, *local};
    return add_node_set_proposal(writer, nodes);
  }
  for (std::size_t index = 0; index + 1 < graph.node_count(); ++index) {
    if (!is_negation(graph, index) || !is_negation(graph, index + 1))
      continue;
    auto added = add_proposal(graph, writer, static_cast<std::uint32_t>(index),
                              2, probe.proposal_dispatch_count);
    if (!added || !*added)
      return added;
    ++index;
  }
  return true;
}

Status propose_regions(void *context, const Graph &graph,
                       RegionProposalWriter &writer) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  auto discovered = discover_test_regions(probe, graph, writer);
  if (!discovered)
    return std::move(discovered).error();
  // A real callback error must not be swallowed just because the writer also
  // reached a capacity. Normal limited discovery returns Ok, not OOM.
  return probe.discovery_failure;
}

Status create_region_invocation(void *prepared, AllocatorRef allocator,
                                hyle::mem::BufferView workspace,
                                void **output) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  void *storage = allocator.allocate(sizeof(RegionRun), alignof(RegionRun));
  if (storage == nullptr)
    return Status::ResourceExhausted("region invocation allocation failed");
  *output = ::new (storage) RegionRun{plan.probe};
  ++plan.probe->create_invocation_count;
  plan.probe->invocation_workspace_valid = workspace.valid();
  plan.probe->invocation_workspace_size = workspace.size();
  if (plan.probe->mode == ProposalMode::kInvocationFailure) {
    return Status::Internal("injected region invocation failure");
  }
  return Status::Ok();
}

void destroy_region_invocation(void *invocation,
                               AllocatorRef allocator) noexcept {
  if (invocation == nullptr)
    return;
  auto *run = static_cast<RegionRun *>(invocation);
  ++run->probe->destroy_invocation_count;
  run->~RegionRun();
  allocator.deallocate(invocation, sizeof(RegionRun), alignof(RegionRun));
}

void destroy_region_prepared(void *prepared, AllocatorRef allocator) noexcept {
  auto *plan = static_cast<RegionPlan *>(prepared);
  if (plan == nullptr)
    return;
  ++plan->probe->destroy_prepared_count;
  plan->~RegionPlan();
  allocator.deallocate(plan, sizeof(RegionPlan), alignof(RegionPlan));
}

AliasSupport region_alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

Status execute_region(void *prepared, void *invocation,
                      std::span<const ConstTensorView> inputs,
                      std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  if (invocation == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return Status::Internal("region binding mismatch");
  auto input_mapping = inputs[0].storage().map_host();
  if (!input_mapping)
    return std::move(input_mapping).error();
  auto output_mapping =
      outputs[0].storage().map_host(hyle::mem::HostAccess::kWrite);
  if (!output_mapping) {
    input_mapping->close().IgnoreError();
    return std::move(output_mapping).error();
  }
  auto writable = output_mapping->writable_bytes();
  if (!writable) {
    Status error = std::move(writable).error();
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return error;
  }
  const std::span<const std::byte> source = input_mapping->bytes();
  if (source.size() != writable->size() || source.size() % sizeof(float) != 0) {
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return Status::Internal("region byte size mismatch");
  }
  const auto input_values =
      std::span(reinterpret_cast<const float *>(source.data()),
                source.size() / sizeof(float));
  const auto output_values =
      std::span(reinterpret_cast<float *>(writable->data()),
                writable->size() / sizeof(float));
  const float sign = (plan.negation_count & 1U) == 0 ? 1.0F : -1.0F;
  for (std::size_t index = 0; index < input_values.size(); ++index)
    output_values[index] = sign * input_values[index];
  Status terminal = output_mapping->close();
  const Status input_close = input_mapping->close();
  if (terminal.ok() && !input_close.ok())
    terminal = input_close;
  ++plan.probe->submit_count;
  return terminal;
}

hyle::compute::detail::EnqueueResult
submit_region(void *prepared, void *invocation, Queue &,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  if (plan.probe->mode == ProposalMode::kAsyncPairs ||
      plan.probe->mode == ProposalMode::kAsyncRejectSecond) {
    if (plan.probe->capture_active)
      return hyle::compute::detail::EnqueueResult::accepted_command();
    ++plan.probe->submit_attempt_count;
    if (plan.probe->mode == ProposalMode::kAsyncRejectSecond &&
        plan.probe->submit_attempt_count == 2) {
      return hyle::compute::detail::EnqueueResult::rejected(
          Status::Unavailable("injected second-region rejection"));
    }
  }
  return hyle::compute::detail::EnqueueResult::accepted_command(
      execute_region(prepared, invocation, inputs, outputs));
}

Result<Completion> submit_delegated_region(
    void *prepared, void *invocation,
    std::span<const hyle::compute::detail::CompletionDependency> dependencies,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  for (const auto dependency : dependencies) {
    if (!dependency.pending() ||
        dependency.vtable != &kAsyncCompletionVTable) {
      return std::unexpected(Status::Unimplemented(
          "delegated test region cannot import this completion"));
    }
    ++plan.probe->device_wait_count;
  }
  ++plan.probe->submit_attempt_count;
  Status terminal = execute_region(prepared, invocation, inputs, outputs);
  if (!terminal.ok())
    return Completion::completed(std::move(terminal));
  return hyle::compute::detail::CompletionAccess::make(
      &kAsyncCompletionVTable, plan.probe);
}

hyle::compute::detail::CompletionEndpoint
delegated_region_endpoint(void *, void *invocation) noexcept {
  auto *run = static_cast<RegionRun *>(invocation);
  return run == nullptr
             ? hyle::compute::detail::CompletionEndpoint{}
             : hyle::compute::detail::CompletionEndpoint{
                   .authority = &kAsyncCompletionVTable,
                   .identity = run,
               };
}

bool delegated_region_can_import(
    void *, void *invocation,
    hyle::compute::detail::CompletionEndpoint source) noexcept {
  auto *run = static_cast<RegionRun *>(invocation);
  return run != nullptr && run->probe->dependency_wait_supported &&
         source.authority == &kAsyncCompletionVTable &&
         source.identity != nullptr;
}

const hyle::graph::ProviderRegionPreparedVTable kPreparedRegionVTable{
    .destroy_prepared = &destroy_region_prepared,
    .alias_support = &region_alias_support,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = &submit_region,
};

const hyle::graph::ProviderRegionPreparedVTable kDelegatedRegionVTable{
    .destroy_prepared = &destroy_region_prepared,
    .alias_support = &region_alias_support,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = nullptr,
    .delegated_submit = &submit_delegated_region,
    .endpoint = &delegated_region_endpoint,
    .can_import_endpoint = &delegated_region_can_import,
};

Status propose_region_resources(
    void *context, const RegionCompileRequest &request,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        input_requirements,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        output_requirements,
    hyle::graph::RegionResourceProposal *proposal) noexcept {
  if (context == nullptr || request.graph == nullptr || proposal == nullptr ||
      input_requirements.size() != request.inputs.size() ||
      output_requirements.size() != request.outputs.size()) {
    return Status::InvalidArgument("region resource proposal is invalid");
  }
  for (auto &requirement : input_requirements)
    requirement = {};
  for (auto &requirement : output_requirements)
    requirement = {};
  auto &probe = *static_cast<RegionProbe *>(context);
  ++probe.region_resource_count;
  if (!input_requirements.empty() && probe.region_input_resource.valid())
    input_requirements[0].exact_resource = probe.region_input_resource;
  if (!output_requirements.empty() && probe.region_output_resource.valid())
    output_requirements[0].exact_resource = probe.region_output_resource;
  *proposal = hyle::graph::RegionResourceProposal{
      .runtime = probe.artifact_properties.runtime,
      .workspace_resource = {
          .exact_resource = probe.region_workspace_resource,
      },
  };
  if (probe.proposed_replay_override)
    proposal->runtime.replay_class = *probe.proposed_replay_override;
  return Status::Ok();
}

Status compile_region_checked(void *context,
                              const RegionCompileRequest &request,
                              AllocatorRef allocator,
                              ProviderRegionPrepared *output) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  const std::size_t expected_boundary_count =
      probe.mode == ProposalMode::kNonContiguousReordered ||
              probe.mode == ProposalMode::kJointCycle
          ? 2u
          : 1u;
  if (output == nullptr || request.graph == nullptr || request.nodes.empty() ||
      request.inputs.size() != expected_boundary_count ||
      request.outputs.size() != expected_boundary_count ||
      request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != request.outputs.size()) {
    return Status::InvalidArgument("region compile request is invalid");
  }
  if ((probe.region_input_resource.valid() &&
       request.input_resources[0] != probe.region_input_resource) ||
      (probe.region_output_resource.valid() &&
       request.output_resources[0] != probe.region_output_resource) ||
      (probe.artifact_properties.runtime.workspace_owner ==
               hyle::compute::ProviderWorkspaceOwner::kCaller
           ? (probe.region_workspace_resource.valid() &&
              request.workspace_resource != probe.region_workspace_resource)
           : request.workspace_resource.valid())) {
    return Status::InvalidArgument(
        "region compile resources do not match the proposed route");
  }
  for (NodeId node : request.nodes) {
    if (!is_negation(*request.graph, node.ordinal()))
      return Status::InvalidArgument("region contains a non-negation node");
  }
  void *storage = allocator.allocate(sizeof(RegionPlan), alignof(RegionPlan));
  if (storage == nullptr)
    return Status::ResourceExhausted("region plan allocation failed");
  auto *plan = ::new (storage) RegionPlan{
      .probe = &probe,
      .negation_count = static_cast<std::uint32_t>(request.nodes.size())};
  *output = ProviderRegionPrepared{
      .vtable = probe.artifact_properties.runtime.mode ==
                        hyle::compute::ProviderRuntimeMode::kDelegated
                    ? &kDelegatedRegionVTable
                    : &kPreparedRegionVTable,
      .state = plan,
      .properties = probe.artifact_properties,
  };
  ++probe.compile_count;
  probe.compiled_input_resource = request.input_resources[0];
  probe.compiled_output_resource = request.output_resources[0];
  probe.compiled_workspace_resource = request.workspace_resource;
  probe.compiled_begin = request.nodes.front().ordinal();
  probe.compiled_count = static_cast<std::uint32_t>(request.nodes.size());
  if (probe.mode == ProposalMode::kCompileFailure)
    return Status::Internal("injected region compile failure");
  return Status::Ok();
}

ProviderPrepareOutcome compile_region(void *context,
                                      const RegionCompileRequest &request,
                                      AllocatorRef allocator,
                                      ProviderRegionPrepared *output) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  if (probe.decline_regions)
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("injected pre-acceptance region decline"));
  Status status = compile_region_checked(context, request, allocator, output);
  if (!status.ok())
    return ProviderPrepareOutcome::error(std::move(status));
  return probe.region_outcome_override.value_or(
      ProviderPrepareOutcome::prepared());
}

const hyle::graph::RegionProviderVTable kRegionVTable{
    .propose = &propose_regions,
    .propose_resources = &propose_region_resources,
    .compile = &compile_region,
};

Status create_provider_context(const Provider &provider, hyle::compute::Device,
                               void *, AllocatorRef, void **output) noexcept {
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  *output = payload.probe;
  return Status::Ok();
}

void destroy_provider_context(void *context, AllocatorRef) noexcept {
  ++static_cast<RegionProbe *>(context)->context_destroy_count;
}

void destroy_effect_prepared(void *, AllocatorRef) noexcept {}

AliasSupport effect_alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

Status create_effect_invocation(void *prepared, AllocatorRef,
                                hyle::mem::BufferView,
                                void **invocation) noexcept {
  if (invocation == nullptr)
    return Status::InvalidArgument("effect invocation output is null");
  auto &probe = *static_cast<RegionProbe *>(prepared);
  ++probe.effect_create_count;
  *invocation = prepared;
  return Status::Ok();
}

void destroy_effect_invocation(void *, AllocatorRef) noexcept {}

hyle::compute::detail::EnqueueResult
submit_effect(void *, void *invocation, Queue &,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  if (invocation == nullptr || !inputs.empty() || !outputs.empty()) {
    return hyle::compute::detail::EnqueueResult::accepted_command(
        Status::Internal("effect invocation binding mismatch"));
  }
  ++static_cast<RegionProbe *>(invocation)->effect_submit_count;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

const hyle::compute::detail::ProviderPreparedVTable kEffectPreparedVTable{
    .destroy_prepared = &destroy_effect_prepared,
    .alias_support = &effect_alias_support,
    .create_invocation = &create_effect_invocation,
    .destroy_invocation = &destroy_effect_invocation,
    .shared_submit = &submit_effect,
    .delegated_submit = nullptr,
};

constexpr hyle::compute::ProviderRuntimeProperties kEffectRuntime{
    .mode = hyle::compute::ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

constexpr hyle::compute::ProviderRuntimeProperties kUnaryRuntime{
    .mode = hyle::compute::ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

hyle::compute::ProviderRuntimeProperties
single_runtime(const RegionProbe &probe, bool effect) noexcept {
  auto runtime =
      effect ? kEffectRuntime
             : (probe.delegated_single ? probe.artifact_properties.runtime
                                       : kUnaryRuntime);
  if (probe.single_replay_override)
    runtime.replay_class = *probe.single_replay_override;
  return runtime;
}

const hyle::compute::detail::ProviderPreparedVTable kUnaryPreparedVTable{
    .destroy_prepared = &destroy_region_prepared,
    .alias_support = &region_alias_support,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = &submit_region,
    .delegated_submit = nullptr,
};

const hyle::compute::detail::ProviderPreparedVTable
    kDelegatedUnaryPreparedVTable{
        .destroy_prepared = &destroy_region_prepared,
        .alias_support = &region_alias_support,
        .create_invocation = &create_region_invocation,
        .destroy_invocation = &destroy_region_invocation,
        .shared_submit = nullptr,
        .delegated_submit = &submit_delegated_region,
        .endpoint = &delegated_region_endpoint,
        .can_import_endpoint = &delegated_region_can_import,
    };

Status propose_single(
    void *context,
    const hyle::compute::detail::ProviderOperationRequest &request,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        input_requirements,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        output_requirements,
    hyle::compute::detail::ProviderOperationProposal *proposal) noexcept {
  const Operation operation = request.operation;
  const auto inputs = request.inputs;
  const auto outputs = request.outputs;
  auto &probe = *static_cast<RegionProbe *>(context);
  ++probe.single_prepare_count;
  probe.destroy_count_at_single_proposal = probe.destroy_prepared_count;
  const bool effect = operation.custom_as<ExternalEffect>().has_value();
  const bool constrained_unary =
      operation.kind() == hyle::compute::OperationKind::kUnary &&
      probe.required_resource.valid();
  if (!effect && !constrained_unary)
    return Status::Unimplemented("region provider has no single-op route");
  if (proposal == nullptr ||
      (effect && (!inputs.empty() || !outputs.empty() ||
                  !input_requirements.empty() ||
                  !output_requirements.empty())) ||
      (constrained_unary &&
       (inputs.size() != 1 || outputs.size() != 1 ||
        input_requirements.size() != 1 || output_requirements.size() != 1))) {
    return Status::InvalidArgument("effect preparation binding mismatch");
  }
  if (constrained_unary) {
    input_requirements[0].exact_resource = probe.required_resource;
    output_requirements[0].exact_resource = probe.required_resource;
  }
  *proposal = hyle::compute::detail::ProviderOperationProposal{
      .route = effect ? 1u : 2u,
      .runtime = single_runtime(probe, effect),
      .cost =
          hyle::compute::ProviderRouteCostFacts{
              .workspace = hyle::compute::WorkspaceRequirement{},
          },
  };
  return Status::Ok();
}

hyle::compute::ProviderPrepareOutcome
prepare_single(void *context,
               const hyle::compute::detail::ProviderPrepareRequest &request,
               AllocatorRef allocator,
               hyle::compute::detail::ProviderPrepared *prepared) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  ++probe.single_artifact_count;
  const bool effect = request.operation.custom_as<ExternalEffect>().has_value();
  const bool constrained_unary =
      request.operation.kind() == hyle::compute::OperationKind::kUnary &&
      probe.required_resource.valid();
  if (prepared == nullptr ||
      (effect && (!request.inputs.empty() || !request.outputs.empty() ||
                  request.route != 1)) ||
      (constrained_unary &&
       (request.inputs.size() != 1 || request.outputs.size() != 1 ||
        request.input_resources.size() != 1 ||
        request.output_resources.size() != 1 || request.route != 2 ||
        request.input_resources[0] != probe.required_resource ||
        request.output_resources[0] != probe.required_resource)) ||
      (!effect && !constrained_unary)) {
    return hyle::compute::ProviderPrepareOutcome::error(
        Status::InvalidArgument("effect exact preparation mismatch"));
  }
  if (constrained_unary) {
    void *storage = allocator.allocate(sizeof(RegionPlan), alignof(RegionPlan));
    if (storage == nullptr)
      return hyle::compute::ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("unary plan allocation failed"));
    auto *plan = ::new (storage) RegionPlan{&probe, 1};
    *prepared = hyle::compute::detail::ProviderPrepared{
        .vtable = probe.delegated_single ? &kDelegatedUnaryPreparedVTable
                                         : &kUnaryPreparedVTable,
        .state = plan,
        .workspace = {},
        .runtime = single_runtime(probe, false),
    };
    return probe.single_outcome_override.value_or(
        ProviderPrepareOutcome::prepared());
  }
  *prepared = hyle::compute::detail::ProviderPrepared{
      .vtable = &kEffectPreparedVTable,
      .state = &probe,
      .runtime = single_runtime(probe, true),
  };
  return probe.single_outcome_override.value_or(
      ProviderPrepareOutcome::prepared());
}

Status drain_provider_context(void *) noexcept { return Status::Ok(); }

const void *query_extension(void *,
                            hyle::compute::ProviderExtensionId id) noexcept {
  return id == hyle::graph::kRegionProviderExtensionId ? &kRegionVTable
                                                       : nullptr;
}

const hyle::compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = kRegionProviderId,
                 .name = "test-region",
                 .version = "1"},
    .create_context = &create_provider_context,
    .destroy_context = &destroy_provider_context,
    .propose = &propose_single,
    .prepare = &prepare_single,
    .drain_context = &drain_provider_context,
    .query_extension = &query_extension,
};

Provider region_provider(RegionProbe &probe) noexcept {
  return hyle::compute::detail::ProviderAccess::make(&kProviderVTable,
                                                     ProviderPayload{&probe});
}

struct AsyncBackendPayload final {
  RegionProbe *probe = nullptr;
};

constexpr hyle::compute::BackendId kAsyncBackendId{0x544553542d475241ULL,
                                                   0x50482d4153594e43ULL};

Result<hyle::compute::Device>
async_device(const hyle::compute::Backend &backend, std::size_t index) noexcept;

Result<std::size_t>
async_device_count(const hyle::compute::Backend &) noexcept {
  return std::size_t{1};
}

bool same_async_device(const hyle::compute::Device &left,
                       const hyle::compute::Device &right) noexcept {
  return hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(left)
             .probe ==
         hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(
             right)
             .probe;
}

Result<hyle::compute::DeviceProperties>
async_device_properties(const hyle::compute::Device &device) noexcept {
  if (hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(device)
          .probe == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("test graph backend device is invalid"));
  }
  return hyle::compute::DeviceProperties{
      .backend = kAsyncBackendId,
      .device_class = hyle::compute::DeviceClass::kCpu,
      .backend_name = "test-graph-async",
      .name = "test Graph async sequence backend",
      .unified_memory = true,
      .asynchronous_queues = true,
  };
}

Status validate_async_resource(const hyle::compute::Device &,
                                hyle::mem::MemoryResourceRef resource) noexcept {
  if (!resource.valid())
    return Status::InvalidArgument("test graph resource is invalid");
  if (!resource.properties().has(hyle::mem::MemoryProperty::kHostAccessible)) {
    return Status::FailedPrecondition(
        "test graph backend requires host-accessible storage");
  }
  return Status::Ok();
}

Status create_async_backend_context(const hyle::compute::Device &device,
                                    AllocatorRef, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("test graph context output is null");
  *output =
      hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(device)
          .probe;
  return *output != nullptr
             ? Status::Ok()
             : Status::InvalidArgument("test graph backend has no probe");
}

void destroy_async_backend_context(void *, AllocatorRef) noexcept {}

Status create_async_queue(void *context, hyle::compute::QueueOptions,
                          AllocatorRef, void **output) noexcept {
  if (context == nullptr || output == nullptr)
    return Status::InvalidArgument("test graph queue output is invalid");
  *output = context;
  return Status::Ok();
}

void destroy_async_queue(void *, AllocatorRef) noexcept {}

Status begin_async_sequence(void *queue, void **sequence) noexcept {
  if (queue == nullptr || sequence == nullptr)
    return Status::InvalidArgument("test graph sequence is invalid");
  *sequence = queue;
  return Status::Ok();
}

Completion finish_async_sequence(void *, void *sequence, bool accepted,
                                 Status terminal) noexcept {
  if (!accepted)
    return Completion::completed(std::move(terminal));
  Completion completion = hyle::compute::detail::CompletionAccess::make(
      &kAsyncCompletionVTable, sequence);
  hyle::compute::detail::CompletionAccess::append_terminal(completion,
                                                           std::move(terminal));
  return completion;
}

Status reserve_async_completions(void *queue, std::size_t minimum) noexcept {
  if (queue == nullptr)
    return Status::InvalidArgument("test graph queue is invalid");
  auto &probe = *static_cast<RegionProbe *>(queue);
  ++probe.reserve_completion_count;
  probe.last_reserved_completions = minimum;
  return Status::Ok();
}

Result<hyle::compute::TransferProperties>
async_transfer_properties(void *context, hyle::mem::BufferView,
                          hyle::mem::ConstBufferView) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe != nullptr)
    ++probe->transfer_query_count;
  if (probe == nullptr || !probe->transfer_supported) {
    return std::unexpected(
        Status::Unimplemented("test graph transfer route is unavailable"));
  }
  return hyle::compute::TransferProperties{
      .mode = hyle::compute::TransferMode::kImmediate};
}

hyle::compute::detail::EnqueueResult
async_copy(void *context, hyle::mem::BufferView destination,
           hyle::mem::ConstBufferView source) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe == nullptr || !probe->transfer_supported) {
    return hyle::compute::detail::EnqueueResult::rejected(
        Status::Unimplemented("test graph transfer route is unavailable"));
  }
  return hyle::compute::detail::EnqueueResult::accepted_command(
      destination.copy_from(source));
}

hyle::compute::detail::EnqueueResult
async_fill(void *, hyle::mem::BufferView destination,
           std::byte value) noexcept {
  return hyle::compute::detail::EnqueueResult::accepted_command(
      destination.fill(value));
}

hyle::compute::detail::EnqueueResult async_wait_completion(
    void *queue,
    const hyle::compute::detail::CompletionVTable *completion_vtable,
    void *completion_state) noexcept {
  if (queue == nullptr || completion_vtable != &kAsyncCompletionVTable ||
      completion_state == nullptr) {
    return hyle::compute::detail::EnqueueResult::rejected(
        Status::Unimplemented("test graph backend has no dependency route"));
  }
  ++static_cast<RegionProbe *>(queue)->device_wait_count;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

hyle::compute::detail::CompletionEndpoint
async_completion_endpoint(void *source_queue) noexcept {
  return source_queue == nullptr
             ? hyle::compute::detail::CompletionEndpoint{}
             : hyle::compute::detail::CompletionEndpoint{
                   .authority = &kAsyncCompletionVTable,
                   .identity = source_queue,
               };
}

bool can_async_wait_endpoint(
    void *destination_queue,
    hyle::compute::detail::CompletionEndpoint source) noexcept {
  auto *destination = static_cast<RegionProbe *>(destination_queue);
  return destination != nullptr &&
         source.authority == &kAsyncCompletionVTable &&
         source.identity != nullptr && destination->dependency_wait_supported;
}

struct MockCapturedCommands final {
  RegionProbe *probe = nullptr;
};

Status begin_mock_capture(void *context, void *queue) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe == nullptr || queue != context || !probe->capture_supported)
    return Status::Unimplemented("mock command capture is unavailable");
  if (probe->capture_active)
    return Status::FailedPrecondition("mock command capture is active");
  probe->capture_active = true;
  ++probe->capture_begin_count;
  return Status::Ok();
}

Status end_mock_capture(void *context, void *queue, AllocatorRef allocator,
                        void **output) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe == nullptr || queue != context || !probe->capture_active ||
      output == nullptr || !allocator.valid()) {
    return Status::FailedPrecondition("mock command capture is invalid");
  }
  *output = nullptr;
  void *storage = allocator.allocate(sizeof(MockCapturedCommands),
                                     alignof(MockCapturedCommands));
  probe->capture_active = false;
  if (storage == nullptr)
    return Status::ResourceExhausted("mock capture allocation failed");
  *output = ::new (storage) MockCapturedCommands{probe};
  ++probe->capture_end_count;
  return Status::Ok();
}

Status abort_mock_capture(void *context, void *queue) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe == nullptr || queue != context || !probe->capture_active)
    return Status::FailedPrecondition("mock command capture is not active");
  probe->capture_active = false;
  ++probe->capture_abort_count;
  return Status::Ok();
}

void destroy_mock_capture(void *state, AllocatorRef allocator) noexcept {
  auto *captured = static_cast<MockCapturedCommands *>(state);
  if (captured == nullptr)
    return;
  ++captured->probe->capture_destroy_count;
  captured->~MockCapturedCommands();
  allocator.deallocate(captured, sizeof(MockCapturedCommands),
                       alignof(MockCapturedCommands));
}

hyle::compute::detail::EnqueueResult
launch_mock_capture(void *context, void *queue, void *state) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  auto *captured = static_cast<MockCapturedCommands *>(state);
  if (probe == nullptr || queue != context || captured == nullptr ||
      captured->probe != probe) {
    return hyle::compute::detail::EnqueueResult::rejected(
        Status::InvalidArgument("mock captured command is invalid"));
  }
  ++probe->capture_launch_count;
  ++probe->submit_attempt_count;
  ++probe->submit_count;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

const hyle::compute::detail::CommandCaptureVTable kMockCaptureVTable{
    .begin = &begin_mock_capture,
    .end = &end_mock_capture,
    .abort = &abort_mock_capture,
    .destroy = &destroy_mock_capture,
    .launch = &launch_mock_capture,
};

const void *
query_async_extension(void *context,
                      hyle::compute::BackendExtensionId id) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe != nullptr && probe->capture_supported &&
      id == hyle::compute::detail::kCommandCaptureExtensionId) {
    return &kMockCaptureVTable;
  }
  return nullptr;
}

Status drain_async(void *state) noexcept {
  if (state != nullptr) {
    auto &probe = *static_cast<RegionProbe *>(state);
    probe.completed_prefix = probe.submit_count;
  }
  return Status::Ok();
}

const hyle::compute::detail::BackendVTable kAsyncBackendVTable{
    .id = kAsyncBackendId,
    .name = "test-graph-async",
    .device_class = hyle::compute::DeviceClass::kCpu,
    .device_count = &async_device_count,
    .device = &async_device,
    .same_device = &same_async_device,
    .properties = &async_device_properties,
    .validate_resource = &validate_async_resource,
    .create_context = &create_async_backend_context,
    .destroy_context = &destroy_async_backend_context,
    .create_queue = &create_async_queue,
    .destroy_queue = &destroy_async_queue,
    .begin_sequence = &begin_async_sequence,
    .finish_sequence = &finish_async_sequence,
    .reserve_completions = &reserve_async_completions,
    .transfer_properties = &async_transfer_properties,
    .copy_async = &async_copy,
    .fill_async = &async_fill,
    .wait_completion = &async_wait_completion,
    .completion_endpoint = &async_completion_endpoint,
    .can_wait_endpoint = &can_async_wait_endpoint,
    .query_extension = &query_async_extension,
    .drain_context = &drain_async,
    .drain_queue = &drain_async,
};

Result<hyle::compute::Device>
async_device(const hyle::compute::Backend &backend,
             std::size_t index) noexcept {
  if (index != 0) {
    return std::unexpected(
        Status::OutOfRange("test graph device index is out of range"));
  }
  return hyle::compute::detail::DeviceAccess::make(
      &kAsyncBackendVTable,
      hyle::compute::detail::BackendAccess::payload<AsyncBackendPayload>(
          backend));
}

hyle::compute::Device async_device(RegionProbe &probe) noexcept {
  const auto backend = hyle::compute::detail::BackendAccess::make(
      &kAsyncBackendVTable, AsyncBackendPayload{&probe});
  return backend.device(0).value();
}

struct NegationGraph final {
  Graph graph{};
  ValueId input{};
  ValueId output{};
};

NegationGraph make_negations(std::size_t count, bool finish_with_abs = false) {
  GraphBuilder builder = GraphBuilder::create().value();
  NegationGraph result;
  result.input = builder.input(row(8)).value();
  ValueId value = result.input;
  for (std::size_t index = 0; index < count; ++index)
    value = builder.add(Unary{UnaryKind::kNeg}, value).value();
  if (finish_with_abs)
    value = builder.add(Unary{UnaryKind::kAbs}, value).value();
  result.output = value;
  EXPECT_TRUE(builder.publish(value).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

struct NonContiguousNegationGraph final {
  Graph graph{};
  ValueId main_input{};
  ValueId side_input{};
  ValueId output{};
};

NonContiguousNegationGraph make_non_contiguous_negations() {
  GraphBuilder builder = GraphBuilder::create().value();
  NonContiguousNegationGraph result;
  result.main_input = builder.input(row(8)).value();
  const ValueId first =
      builder.add(Unary{UnaryKind::kNeg}, result.main_input).value();
  result.side_input = builder.input(row(8)).value();
  (void)builder.add(Unary{UnaryKind::kNeg}, result.side_input).value();
  result.output = builder.add(Unary{UnaryKind::kNeg}, first).value();
  EXPECT_TRUE(builder.publish(result.output).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

Graph make_reordered_non_contiguous_negations() {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId main_input = builder.input(row(8)).value();
  const ValueId selected_first =
      builder.add(Unary{UnaryKind::kNeg}, main_input).value();
  const ValueId outside_consumer =
      builder.add(Unary{UnaryKind::kNeg}, selected_first).value();
  const ValueId side_input = builder.input(row(8)).value();
  const ValueId outside_producer =
      builder.add(Unary{UnaryKind::kNeg}, side_input).value();
  const ValueId selected_last =
      builder.add(Unary{UnaryKind::kNeg}, outside_producer).value();
  EXPECT_TRUE(builder.publish(outside_consumer).ok());
  EXPECT_TRUE(builder.publish(selected_last).ok());
  return std::move(builder).finish().value();
}

Context make_context(RegionProbe &probe) {
  const std::array providers{region_provider(probe),
                             hyle::compute::native::provider()};
  const bool asynchronous = probe.asynchronous_backend ||
                            probe.mode == ProposalMode::kAsyncPairs ||
                            probe.mode == ProposalMode::kAsyncRejectSecond;
  return Context::create(asynchronous ? async_device(probe)
                                      : hyle::compute::cpu::device(),
                         {.providers = providers})
      .value();
}

Context make_context(RegionProbe &first, RegionProbe &second) {
  const std::array providers{region_provider(first), region_provider(second),
                             hyle::compute::native::provider()};
  return Context::create(hyle::compute::cpu::device(), {.providers = providers})
      .value();
}

void use_delegated_runtime(RegionProbe &probe) noexcept {
  probe.artifact_properties.runtime = {
      .mode = hyle::compute::ProviderRuntimeMode::kDelegated,
      .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
      .blocks_host = false,
      .uses_private_threads = true,
      .supports_concurrent_invocations = true,
      .capture_compatible = false,
  };
}

TEST(GraphRegions, ExecutesNonMaterializingExternalEffectSteps) {
  constexpr std::array<ValueId, 0> no_inputs{};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto first =
      builder.add_node(Operation::custom(ExternalEffect{}), no_inputs).value();
  const auto second =
      builder.add_node(Operation::custom(ExternalEffect{}), no_inputs).value();
  ASSERT_TRUE(builder.depends_on(second, first).ok());
  Graph graph = std::move(builder).finish().value();

  RegionProbe probe;
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(graph, context).value();
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.report.segment_count(), 1u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 1u);
  EXPECT_EQ(compiled.report.step(0)->dispatch_count, 1u);
  EXPECT_EQ(compiled.report.step(1)->dispatch_count, 1u);

  auto invocation = compiled.executable.create_invocation().value();
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.effect_create_count, 2u);
  EXPECT_EQ(probe.effect_submit_count, 2u);
}

TEST(GraphRegions, ReplayCostFactsSurviveRegionCompilation) {
  using hyle::compute::ProviderReplayClass;
  constexpr std::array classes{ProviderReplayClass::kUnknown,
                               ProviderReplayClass::kPrepared,
                               ProviderReplayClass::kManaged};
  const NegationGraph logical = make_negations(2);
  for (auto replay_class : classes) {
    RegionProbe probe;
    probe.artifact_properties.runtime.replay_class = replay_class;
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    ASSERT_EQ(compiled->report.region_count(), 1u);
    EXPECT_EQ(compiled->report.step(0)->runtime.replay_class, replay_class);
  }
}

TEST(GraphRegions, ReportKeepsInstanceFactsWithoutRetainingTheirContext) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  CompileReport retained_region;
  CompileReport retained_leaf;
  hyle::compute::ProviderInstanceId region_id;
  hyle::compute::ProviderInstanceId leaf_id;
  {
    Context context = make_context(probe);
    region_id = context.provider_instance_id(0).value();
    leaf_id = context.provider_instance_id(1).value();
    auto region = hyle::graph::compile(logical.graph, context).value();
    auto leaf = hyle::graph::compile(
                    logical.graph, context,
                    {.region_selection = RegionSelectionPolicy::kDisabled})
                    .value();
    EXPECT_EQ(region.report.step(0)->provider_instance, region_id);
    EXPECT_EQ(leaf.report.step(0)->provider_instance, leaf_id);
    EXPECT_EQ(leaf.report.step(1)->provider_instance, leaf_id);
    retained_region = region.report;
    retained_leaf = leaf.report;
  }
  EXPECT_EQ(probe.context_destroy_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
  EXPECT_EQ(retained_region.step(0)->provider_instance, region_id);
  EXPECT_EQ(retained_leaf.step(0)->provider_instance, leaf_id);
  Context recreated = make_context(probe);
  EXPECT_NE(recreated.provider_instance_id(0).value(), region_id);
  EXPECT_NE(recreated.provider_instance_id(1).value(), leaf_id);
}

TEST(GraphRegions,
     PreparedOnlyRejectsUnknownAndManagedBeforeRegionPreparation) {
  const auto logical = make_negations(1);
  for (auto replay :
       {ProviderReplayClass::kUnknown, ProviderReplayClass::kManaged}) {
    RegionProbe probe{.mode = ProposalMode::kSingleton};
    probe.discovery_replay_class = replay;
    // Even a later prepared declaration cannot rescue unknown discovery.
    probe.artifact_properties.runtime.replay_class =
        replay == ProviderReplayClass::kUnknown ? ProviderReplayClass::kPrepared
                                                : replay;
    Context context = make_context(probe);
    auto strict = hyle::graph::compile(
        logical.graph, context,
        {.provider_replay = ProviderReplayPolicy::kRequirePrepared});
    ASSERT_TRUE(strict.has_value()) << strict.error().message();
    EXPECT_EQ(strict->report.region_count(), 0u);
    EXPECT_EQ(strict->report.step(0)->runtime.replay_class,
              ProviderReplayClass::kPrepared);
    EXPECT_EQ(probe.region_resource_count, 0u);
    EXPECT_EQ(probe.compile_count, 0u);

    auto permissive = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(permissive.has_value()) << permissive.error().message();
    EXPECT_EQ(permissive->report.region_count(), 1u);
    EXPECT_EQ(permissive->report.step(0)->runtime.replay_class,
              probe.artifact_properties.runtime.replay_class);
    EXPECT_EQ(probe.compile_count, 1u);
  }
}

TEST(GraphRegions, PreparedReplayCanHaveDelegatedOwnership) {
  const auto logical = make_negations(1);
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  use_delegated_runtime(probe);
  probe.discovery_replay_class = ProviderReplayClass::kPrepared;
  probe.artifact_properties.runtime.replay_class =
      ProviderReplayClass::kPrepared;
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.provider_replay = ProviderReplayPolicy::kRequirePrepared});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->runtime.mode,
            hyle::compute::ProviderRuntimeMode::kDelegated);
  EXPECT_EQ(compiled->executable.required_queue_count(), 0u);

  auto input = make_tensor(row(8));
  auto output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit();
    ASSERT_TRUE(done.has_value()) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto result = read(output, values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
      EXPECT_EQ(result[index], -values[index]);
  }
  EXPECT_EQ(probe.propose_count, 1u);
  EXPECT_EQ(probe.region_resource_count, 1u);
  EXPECT_EQ(probe.compile_count, 1u);
}

TEST(GraphRegions, ReplayFilteringPrecedesTargetSelection) {
  const auto logical = make_negations(1);
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  probe.discovery_replay_class = ProviderReplayClass::kManaged;
  probe.artifact_properties.runtime.replay_class =
      ProviderReplayClass::kManaged;
  const std::array providers{region_provider(probe)};
  Context first =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  const std::array native_providers{hyle::compute::native::provider()};
  Context second = Context::create(hyle::compute::cpu::device(),
                                   {.providers = native_providers})
                       .value();
  const Environment environment{{&first, hyle::mem::system_resource()},
                                {&second, hyle::mem::system_resource()}};
  auto compiled = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.provider_replay = ProviderReplayPolicy::kRequirePrepared});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 0u);
  EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
  EXPECT_EQ(probe.region_resource_count, 0u);
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, KnownDiscoveryReplayCannotChangeBeforeArtifactCreation) {
  const auto logical = make_negations(1);
  for (auto declared :
       {ProviderReplayClass::kPrepared, ProviderReplayClass::kManaged}) {
    for (auto actual :
         {ProviderReplayClass::kUnknown, ProviderReplayClass::kPrepared,
          ProviderReplayClass::kManaged}) {
      if (declared == actual)
        continue;
      RegionProbe probe{.mode = ProposalMode::kSingleton};
      probe.discovery_replay_class = declared;
      probe.artifact_properties.runtime.replay_class = actual;
      Context context = make_context(probe);
      auto compiled = hyle::graph::compile(logical.graph, context);
      ASSERT_FALSE(compiled.has_value());
      EXPECT_TRUE(compiled.error().IsInternal());
      EXPECT_EQ(probe.region_resource_count, 1u);
      EXPECT_EQ(probe.compile_count, 0u);
      EXPECT_EQ(probe.submit_count, 0u);
    }
  }
}

TEST(GraphRegions, PreparedOnlyFiltersLeafBeforeArtifactCreation) {
  const auto logical = make_negations(1);
  for (auto replay :
       {ProviderReplayClass::kUnknown, ProviderReplayClass::kManaged,
        ProviderReplayClass::kPrepared}) {
    RegionProbe probe;
    probe.required_resource = hyle::mem::system_resource();
    probe.single_replay_override = replay;
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.region_selection = RegionSelectionPolicy::kDisabled,
         .provider_replay = ProviderReplayPolicy::kRequirePrepared});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.step(0)->runtime.replay_class,
              ProviderReplayClass::kPrepared);
    const bool selected = replay == ProviderReplayClass::kPrepared;
    EXPECT_EQ(probe.single_artifact_count, selected ? 1u : 0u);
    EXPECT_EQ(compiled->report.step(0)->provider.id == kRegionProviderId,
              selected);
    EXPECT_EQ(probe.propose_count, 0u);
  }
}

TEST(GraphRegions, PreparedOnlyFailsWhenNoEligibleExecutionRouteExists) {
  GraphBuilder builder = GraphBuilder::create().value();
  (void)builder
      .add_node(Operation::custom(ExternalEffect{}), std::array<ValueId, 0>{})
      .value();
  Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      graph, context,
      {.region_selection = RegionSelectionPolicy::kDisabled,
       .provider_replay = ProviderReplayPolicy::kRequirePrepared});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsUnimplemented());
  EXPECT_EQ(probe.single_artifact_count, 0u);
  EXPECT_EQ(probe.effect_create_count, 0u);
}

TEST(GraphRegions, ReplayPolicyDoesNotHideInvalidEnumsOrDiscoveryBudget) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton};
  RegionProbe second{.mode = ProposalMode::kSingleton};
  Context context = make_context(first, second);
  auto invalid_policy = hyle::graph::compile(
      logical.graph, context,
      {.provider_replay = static_cast<ProviderReplayPolicy>(255)});
  ASSERT_FALSE(invalid_policy.has_value());
  EXPECT_TRUE(invalid_policy.error().IsInvalidArgument());
  EXPECT_EQ(first.propose_count, 0u);

  first.discovery_replay_class = static_cast<ProviderReplayClass>(255);
  auto invalid_declaration = hyle::graph::compile(
      logical.graph, context,
      {.provider_replay = ProviderReplayPolicy::kRequirePrepared});
  ASSERT_FALSE(invalid_declaration.has_value());
  EXPECT_TRUE(invalid_declaration.error().IsInternal());

  first.discovery_replay_class = ProviderReplayClass::kUnknown;
  auto budget = hyle::graph::compile(
      logical.graph, context,
      {.max_region_proposals = 1,
       .provider_replay = ProviderReplayPolicy::kRequirePrepared});
  ASSERT_TRUE(budget.has_value()) << budget.error().message();
  EXPECT_TRUE(budget->report.search().region_proposal_limit_reached);
  EXPECT_EQ(budget->report.search().region_proposals, 1u);
  EXPECT_EQ(budget->report.region_count(), 0u);
  EXPECT_EQ(budget->report.fallback_operation_count(), 1u);
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 0u);
}

TEST(GraphRegions, RejectsInvalidOrChangedReplayCostBeforeWorkAcceptance) {
  using hyle::compute::ProviderReplayClass;
  const NegationGraph logical = make_negations(2);
  RegionProbe probe;
  probe.artifact_properties.runtime.replay_class = ProviderReplayClass::kManaged;
  probe.proposed_replay_override = static_cast<ProviderReplayClass>(255);
  Context context = make_context(probe);
  auto invalid_proposal = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(invalid_proposal.has_value());
  EXPECT_TRUE(invalid_proposal.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 0u);

  probe.proposed_replay_override = ProviderReplayClass::kPrepared;
  auto changed = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(changed.has_value());
  EXPECT_TRUE(changed.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);

  probe.artifact_properties.runtime.replay_class =
      static_cast<ProviderReplayClass>(255);
  auto invalid_artifact = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(invalid_artifact.has_value());
  EXPECT_TRUE(invalid_artifact.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 2u);
  EXPECT_EQ(probe.destroy_prepared_count, 2u);
  EXPECT_EQ(probe.submit_count, 0u);
}

TEST(GraphRegions, CompilesAndSubmitsOneArtifactWithoutInternalStorage) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.node_count(), 2u);
  EXPECT_EQ(compiled.report.step_count(), 1u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 0u);
  EXPECT_EQ(compiled.report.provider().id, kRegionProviderId);
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  EXPECT_EQ(probe.single_prepare_count, 0u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
  EXPECT_EQ(probe.submit_count, 1u);
}

TEST(GraphRegions,
     DelegatedRegionRunsWithoutAQueueAndRetainsTheWholeInvocation) {
  RegionProbe probe;
  use_delegated_runtime(probe);
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.delegated_segment_count(), 1u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 0u);
  ASSERT_EQ(compiled.report.lane_count(), 2u);
  EXPECT_FALSE(compiled.report.lane(0)->queue_required);
  EXPECT_EQ(compiled.report.lane(1)->runtime_mode,
            hyle::compute::ProviderRuntimeMode::kDelegated);
  EXPECT_FALSE(compiled.report.lane(1)->queue_required);
  EXPECT_FALSE(compiled.executable.requires_queue(
                   hyle::graph::ExecutionLaneId{1})
                   .value());

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  EXPECT_TRUE(invocation
                  .prepare_replay(
                      {}, {.mode = hyle::graph::ReplayMode::kRequireCapture})
                  .IsUnimplemented());
  ASSERT_TRUE(invocation.prepare_replay({}).ok());

  auto completion = invocation.submit();
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_EQ(completion->poll().value(),
            hyle::compute::CompletionPoll::kPending);
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));

  auto cancelled = invocation.submit().value();
  EXPECT_TRUE(cancelled.request_cancel().ok());
  EXPECT_TRUE(cancelled.wait().IsCancelled());
  EXPECT_FALSE(invocation.in_flight());
  probe.cancelled = false;

  probe.terminal_failure = true;
  EXPECT_TRUE(invocation.submit()->wait().IsInternal());
  EXPECT_FALSE(invocation.in_flight());
  probe.terminal_failure = false;

  Completion abandoned = invocation.submit().value();
  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 0u);
  abandoned = {};
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.destroy_invocation_count, 0u);
  finish_abandoned_prefix(probe);
  EXPECT_EQ(probe.destroy_invocation_count, 1u);
}

TEST(GraphRegions, DelegatedSingleOperationUsesTheGraphLifetimeOwner) {
  RegionProbe probe;
  use_delegated_runtime(probe);
  probe.delegated_single = true;
  probe.required_resource = hyle::mem::system_resource();
  const NegationGraph logical = make_negations(1);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 0u);
  ASSERT_EQ(compiled.report.step_count(), 1u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 0u);
  EXPECT_EQ(compiled.report.step(0)->runtime.mode,
            hyle::compute::ProviderRuntimeMode::kDelegated);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  const std::array<float, 8> expected{4, 3, 2, 1, -1, -2, -3, -4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto completion = invocation.submit();
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(read(output, expected.size()),
            std::vector<float>(expected.begin(), expected.end()));
}

TEST(GraphRegions, ImportsASharedQueuePrefixIntoADelegatedRegion) {
  RegionProbe probe{.mode = ProposalMode::kOverlappingRoutes};
  probe.asynchronous_backend = true;
  probe.required_resource = hyle::mem::system_resource();
  use_delegated_runtime(probe);
  const NegationGraph logical = make_negations(3);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(logical.graph, context, CompileOptions{}).value();
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.report.region_count(), 1u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  const std::array<float, 8> expected{4, 3, 2, 1, -1, -2, -3, -4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.prepare_replay(
                            std::array{hyle::graph::QueueBinding{
                                .lane = hyle::graph::ExecutionLaneId{0},
                                .queue = &queue}})
                  .ok());
  const auto replay = invocation.replay_report().value();
  EXPECT_EQ(replay.device_wait_dependency_count, 1u);
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(probe.device_wait_count, 1u);
  EXPECT_EQ(read(output, expected.size()),
            std::vector<float>(expected.begin(), expected.end()));
}

TEST(GraphRegions, ImportsADelegatedRegionIntoASharedQueue) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  use_delegated_runtime(probe);
  const NegationGraph logical = make_negations(2, true);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.report.region_count(), 1u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  const std::array<float, 8> expected{4, 3, 2, 1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(probe.device_wait_count, 1u);
  EXPECT_EQ(read(output, expected.size()),
            std::vector<float>(expected.begin(), expected.end()));
}

TEST(GraphRegions, ChainsDelegatedRegionsWithOneProviderDependency) {
  RegionProbe probe;
  use_delegated_runtime(probe);
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 2u);
  EXPECT_EQ(compiled.report.delegated_segment_count(), 2u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 0u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  ASSERT_TRUE(invocation.prepare_replay({}).ok());
  EXPECT_EQ(invocation.replay_report()->device_wait_dependency_count, 1u);
  auto completion = invocation.submit();
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(probe.device_wait_count, 1u);
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, ExposesADelegatedHostWaitBeforeAcceptingAnyWork) {
  RegionProbe probe;
  probe.dependency_wait_supported = false;
  use_delegated_runtime(probe);
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());

  EXPECT_TRUE(invocation
                  .prepare_replay(
                      {}, {.host_wait = hyle::graph::HostWaitPolicy::kReject})
                  .IsUnimplemented());
  EXPECT_EQ(probe.submit_count, 0u);
  ASSERT_TRUE(invocation.prepare_replay({}).ok());
  EXPECT_EQ(invocation.replay_report()->host_wait_dependency_count, 1u);
  EXPECT_EQ(invocation.replay_report()->device_wait_dependency_count, 0u);
  auto completion = invocation.submit();
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(probe.device_wait_count, 0u);
  EXPECT_GE(probe.completion_wait_count, 2u);
}

TEST(GraphRegions, KeepsIndependentDelegatedTerminalEpisodesDistinct) {
  RegionProbe probe;
  use_delegated_runtime(probe);
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(8)).value();
  const ValueId second_input = builder.input(row(8)).value();
  ValueId first_output = builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  first_output = builder.add(Unary{UnaryKind::kNeg}, first_output).value();
  ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  second_output = builder.add(Unary{UnaryKind::kNeg}, second_output).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 2u);
  ASSERT_EQ(compiled.report.segment_dependency_count(), 0u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 0u);
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor first_source = make_tensor(row(8));
  OwnedTensor second_source = make_tensor(row(8));
  OwnedTensor first_destination = make_tensor(row(8));
  OwnedTensor second_destination = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());

  auto completion = invocation.submit();
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(probe.submit_count, 2u);
  EXPECT_EQ(probe.completion_wait_count, 2u);
}

TEST(GraphRegions, SelectsExactBoundaryAndWorkspaceResourcesForARegion) {
  RegionProbe probe;
  HostResource first_storage;
  HostResource second_storage;
  probe.region_input_resource = hyle::mem::MemoryResourceRef(second_storage);
  probe.region_output_resource = hyle::mem::MemoryResourceRef(second_storage);
  probe.region_workspace_resource =
      hyle::mem::MemoryResourceRef(second_storage);
  probe.artifact_properties.workspace = {.size = 256, .alignment = 64};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  const Environment environment{{
      ExecutionSpec{.context = &context,
                    .storage = hyle::mem::MemoryResourceRef(first_storage)},
      ExecutionSpec{.context = &context,
                    .storage = hyle::mem::MemoryResourceRef(second_storage)},
  }};

  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment()).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  ASSERT_EQ(compiled.report.step_count(), 2u);
  const auto region = compiled.report.step(1).value();
  EXPECT_EQ(region.kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(region.workspace_storage, environment.storage(1));
  EXPECT_EQ(region.workspace.size, 256u);
  EXPECT_EQ(probe.compiled_input_resource,
            hyle::mem::MemoryResourceRef(second_storage));
  EXPECT_EQ(probe.compiled_output_resource,
            hyle::mem::MemoryResourceRef(second_storage));
  EXPECT_EQ(probe.compiled_workspace_resource,
            hyle::mem::MemoryResourceRef(second_storage));
  EXPECT_GT(compiled.report.copy_bytes(), 0u);
  EXPECT_EQ(compiled.report.storage_domain(0)->workspace_bytes, 0u);
  EXPECT_GE(compiled.report.storage_domain(1)->workspace_bytes, 256u);

  OwnedTensor input = make_tensor(
      row(8), hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor output = make_tensor(
      row(8), hyle::mem::MemoryResourceRef(second_storage));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
  EXPECT_TRUE(probe.invocation_workspace_valid);
  EXPECT_EQ(probe.invocation_workspace_size, 256u);
}

TEST(GraphRegions, KeepsProviderOwnedWorkspaceOutsideTheGraphPlan) {
  RegionProbe probe;
  probe.artifact_properties.workspace = {.size = 4096, .alignment = 64};
  probe.artifact_properties.runtime.workspace_owner =
      hyle::compute::ProviderWorkspaceOwner::kProvider;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);

  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  const auto step = compiled.report.step(0).value();
  EXPECT_EQ(step.workspace.size, 4096u);
  EXPECT_FALSE(step.workspace_storage.valid());

  auto invocation = compiled.executable.create_invocation().value();
  EXPECT_FALSE(probe.invocation_workspace_valid);
  EXPECT_EQ(probe.invocation_workspace_size, 0u);
}

TEST(GraphRegions, FallsBackWhenASelectedRegionHasNoResourceRoute) {
  RegionProbe probe;
  HostResource unavailable;
  probe.region_input_resource = hyle::mem::MemoryResourceRef(unavailable);
  probe.region_output_resource = hyle::mem::MemoryResourceRef(unavailable);
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);

  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.compile_count, 0u);
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    const auto step = compiled.report.step(index).value();
    EXPECT_EQ(step.kind, CompileStepKind::kOperationFallback);
    EXPECT_EQ(step.fallback_reason,
              CompileFallbackReason::kRegionNotSelected);
  }
}

TEST(GraphRegions, CapturesAndReplaysACommandSegmentWithoutRewalkingItsSteps) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  const Environment environment{{&context, hyle::mem::system_resource()}};
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.lane_count(), 1u);
  ASSERT_TRUE(compiled.report.lane(0).has_value());
  EXPECT_TRUE(compiled.report.lane(0)->capture_supported);
  EXPECT_EQ(compiled.report.command_segment_count(), 1u);
  EXPECT_EQ(compiled.report.capturable_segment_count(), 1u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      .lane = environment.lane(0), .queue = &queue}};

  ASSERT_TRUE(
      invocation
          .prepare_replay(queues,
                          {.mode = hyle::graph::ReplayMode::kRequireCapture})
          .ok());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kCaptured);
  EXPECT_EQ(probe.capture_begin_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 1u);
  EXPECT_EQ(probe.capture_abort_count, 0u);
  EXPECT_EQ(probe.submit_count, 0u);

  const std::array invalid_queues{hyle::graph::QueueBinding{
      .lane = environment.lane(0), .queue = nullptr}};
  EXPECT_TRUE(invocation.prepare_replay(invalid_queues).IsInvalidArgument());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kCaptured);
  EXPECT_EQ(probe.capture_destroy_count, 0u);

  for (int replay = 0; replay < 2; ++replay) {
    auto completion = invocation.submit(queues);
    ASSERT_TRUE(completion.has_value()) << completion.error().message();
    EXPECT_TRUE(completion->wait().ok());
  }
  EXPECT_EQ(probe.capture_launch_count, 2u);
  EXPECT_EQ(probe.submit_count, 2u);

  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_EQ(probe.capture_destroy_count, 1u);
}

TEST(GraphRegions, MixesACompiledRegionWithSingleOperationFallback) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2, true);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.node_count(), 3u);
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 1u);
  EXPECT_TRUE(compiled.report.mixed_providers());
  EXPECT_FALSE(compiled.report.provider().valid());
  EXPECT_EQ(probe.single_prepare_count, 1u);
  auto fallback = compiled.report.step(1);
  ASSERT_TRUE(fallback.has_value());
  EXPECT_EQ(fallback->kind, CompileStepKind::kOperationFallback);
  EXPECT_EQ(fallback->fallback_reason,
            CompileFallbackReason::kNoEligibleRegion);
  EXPECT_FALSE(fallback->persistent_bytes.has_value());

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  const std::array<float, 8> expected{4, 3, 2, 1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, expected.size()),
            std::vector<float>(expected.begin(), expected.end()));
}

TEST(GraphRegions,
     SelectsFewerEpisodesForSameIntervalCoverageDeterministically) {
  RegionProbe probe{.mode = ProposalMode::kOverlappingRoutes};
  const NegationGraph logical = make_negations(3);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(logical.graph, context, CompileOptions{}).value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compiled_begin, 1u);
  EXPECT_EQ(probe.compiled_count, 2u);
}

TEST(GraphRegions, CompilesAContractibleNonContiguousNodeSet) {
  RegionProbe probe{.mode = ProposalMode::kNonContiguous};
  const NonContiguousNegationGraph logical = make_non_contiguous_negations();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();

  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 1u);
  bool found_region = false;
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    const auto step = compiled.report.step(index).value();
    if (step.kind != CompileStepKind::kProviderRegion)
      continue;
    found_region = true;
    EXPECT_EQ(step.node_begin, 0u);
    EXPECT_EQ(step.node_count, 2u);
    EXPECT_FALSE(step.nodes_contiguous);
    EXPECT_EQ(compiled.report.step_node(index, 0)->ordinal(), 0u);
    EXPECT_EQ(compiled.report.step_node(index, 1)->ordinal(), 2u);
  }
  EXPECT_TRUE(found_region);
}

TEST(GraphRegions, SchedulesAContractedNodeSetAfterItsExternalPredecessor) {
  RegionProbe probe{.mode = ProposalMode::kNonContiguousReordered};
  const Graph logical = make_reordered_non_contiguous_negations();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical, context).value();

  ASSERT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.step(0)->kind, CompileStepKind::kOperationFallback);
  EXPECT_EQ(compiled.report.step_node(0, 0)->ordinal(), 2u);
  EXPECT_EQ(compiled.report.step(1)->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled.report.step_node(1, 0)->ordinal(), 0u);
  EXPECT_EQ(compiled.report.step_node(1, 1)->ordinal(), 3u);
  EXPECT_EQ(compiled.report.step(2)->kind, CompileStepKind::kOperationFallback);
  EXPECT_EQ(compiled.report.step_node(2, 0)->ordinal(), 1u);
}

TEST(GraphRegions, RejectsANodeSetThatLeavesAndReentersTheRegion) {
  RegionProbe probe{.mode = ProposalMode::kNonContiguous};
  const NegationGraph logical = make_negations(3);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInvalidArgument());
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, ExecutesInterleavedIndependentRegions) {
  RegionProbe probe{.mode = ProposalMode::kInterleavedChains};
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input_a = builder.input(row(8)).value();
  const ValueId input_b = builder.input(row(8)).value();
  const ValueId first_a = builder.add(Unary{UnaryKind::kNeg}, input_a).value();
  const ValueId first_b = builder.add(Unary{UnaryKind::kNeg}, input_b).value();
  const ValueId output_a = builder.add(Unary{UnaryKind::kNeg}, first_a).value();
  const ValueId output_b = builder.add(Unary{UnaryKind::kNeg}, first_b).value();
  ASSERT_TRUE(builder.publish(output_a).ok());
  ASSERT_TRUE(builder.publish(output_b).ok());
  const Graph graph = std::move(builder).finish().value();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 2u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 0u);
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor first = make_tensor(row(8));
  OwnedTensor second = make_tensor(row(8));
  OwnedTensor result_a = make_tensor(row(8));
  OwnedTensor result_b = make_tensor(row(8));
  const std::array<float, 8> a{1, -2, 3, -4, 5, -6, 7, -8};
  const std::array<float, 8> b{-3, 4, 5, 6, -7, 8, 9, 10};
  write(first, a);
  write(second, b);
  ASSERT_TRUE(invocation.bind(input_a, first.view).ok());
  ASSERT_TRUE(invocation.bind(input_b, second.view).ok());
  ASSERT_TRUE(invocation.bind(output_a, result_a.view).ok());
  ASSERT_TRUE(invocation.bind(output_b, result_b.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(result_a, a.size()), std::vector<float>(a.begin(), a.end()));
  EXPECT_EQ(read(result_b, b.size()), std::vector<float>(b.begin(), b.end()));
}

TEST(GraphRegions, ExcludesARegionThatWouldCycleWithAnAcceptedRegion) {
  RegionProbe probe{.mode = ProposalMode::kJointCycle};
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input_a = builder.input(row(8)).value();
  const ValueId input_b = builder.input(row(8)).value();
  const ValueId first_a = builder.add(Unary{UnaryKind::kNeg}, input_a).value();
  const ValueId first_b = builder.add(Unary{UnaryKind::kNeg}, input_b).value();
  const ValueId output_b = builder.add(Unary{UnaryKind::kNeg}, first_b).value();
  const ValueId output_a = builder.add(Unary{UnaryKind::kNeg}, first_a).value();
  ASSERT_TRUE(builder.publish(output_a).ok());
  ASSERT_TRUE(builder.publish(output_b).ok());
  const Graph graph = std::move(builder).finish().value();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(graph, context).value();
  // {0,2} and {1,3} are individually contractible, but the edges 0->3 and
  // 1->2 make their joint contraction cyclic. The second proposal stays leaf.
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.compile_count, 1u);
}

TEST(GraphRegions, DeclinedNonContiguousRegionExecutesLeafFallbacks) {
  RegionProbe probe{.mode = ProposalMode::kNonContiguous,
                    .decline_regions = true};
  const auto logical = make_non_contiguous_negations();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 3u);
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor side = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  write(side, values);
  ASSERT_TRUE(invocation.bind(logical.main_input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.side_input, side.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, AGraphHostingProviderCanOwnASingleNode) {
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  const auto logical = make_negations(1);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.step(0)->node_count, 1u);
  EXPECT_EQ(compiled.report.step_node(0, 0).value(),
            logical.graph.node_id(0).value());
  EXPECT_TRUE(compiled.report.step_node(0, 1).error().IsOutOfRange());
}

TEST(GraphRegions, RejectsDuplicateAndForeignProposalNodes) {
  const NegationGraph logical = make_negations(2);
  {
    RegionProbe probe{.mode = ProposalMode::kDuplicateNode};
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsInvalidArgument());
  }
  {
    const NegationGraph foreign = make_negations(2);
    RegionProbe probe{.mode = ProposalMode::kForeignNode,
                      .foreign_node = foreign.graph.node_id(0).value()};
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsInvalidArgument());
  }
}

TEST(GraphRegions, ProviderCanQueryContractibilityIncludingControlEdges) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(row(8)).value();
  const auto first =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{input})
          .value();
  const auto outside =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{input})
          .value();
  const auto last = builder
                        .add_node(Operation(Unary{UnaryKind::kNeg}),
                                  std::array{builder.result(first).value()})
                        .value();
  ASSERT_TRUE(builder.depends_on(outside, first).ok());
  ASSERT_TRUE(builder.depends_on(last, outside).ok());
  ASSERT_TRUE(builder.publish(builder.result(last).value()).ok());
  const Graph graph = std::move(builder).finish().value();
  EXPECT_FALSE(
      hyle::graph::is_region_contractible(graph, std::array{last, first})
          .value());
  EXPECT_TRUE(hyle::graph::is_region_contractible(
                  graph, std::array{last, first, outside})
                  .value());
  EXPECT_TRUE(
      hyle::graph::is_region_contractible(graph, std::array{first, first})
          .error()
          .IsInvalidArgument());
  EXPECT_TRUE(hyle::graph::is_region_contractible(graph, {})
                  .error()
                  .IsInvalidArgument());
  const auto foreign = make_negations(1);
  EXPECT_TRUE(hyle::graph::is_region_contractible(
                  graph, std::array{foreign.graph.node_id(0).value()})
                  .error()
                  .IsInvalidArgument());
}

TEST(GraphRegions, ProposalNodeLimitDoesNotAdmitAPartialRegion) {
  RegionProbe probe{.mode = ProposalMode::kNonContiguous};
  const NonContiguousNegationGraph logical = make_non_contiguous_negations();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context, CompileOptions{.max_region_proposal_nodes = 1});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_TRUE(compiled->report.search().region_node_limit_reached);
  EXPECT_EQ(compiled->report.search().region_proposals, 0u);
  EXPECT_EQ(compiled->report.search().region_proposal_nodes, 0u);
  EXPECT_EQ(compiled->report.region_count(), 0u);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 3u);
  EXPECT_EQ(probe.compile_count, 0u);

  auto no_coverage = hyle::graph::compile(
      logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject,
       .max_region_proposal_nodes = 1});
  ASSERT_FALSE(no_coverage.has_value());
  EXPECT_TRUE(no_coverage.error().IsUnimplemented());
}

TEST(GraphRegions, RelocatesAnUnselectedRegionNodeToAWorkingFallbackTarget) {
  RegionProbe probe{.mode = ProposalMode::kOverlappingRoutes};
  const NegationGraph logical = make_negations(3);
  const std::array region_only{region_provider(probe)};
  Context region_context =
      Context::create(hyle::compute::cpu::device(), {.providers = region_only})
          .value();
  const std::array native_only{hyle::compute::native::provider()};
  Context fallback_context =
      Context::create(hyle::compute::cpu::device(), {.providers = native_only})
          .value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{&region_context, host},
                                {&fallback_context, host}};
  const std::array resources{hyle::graph::ExternalResourceContract{
      .value = logical.input,
      .storage = environment.storage(1),
  }};
  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.external_resources = resources})
          .value();

  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 1u);
  EXPECT_EQ(probe.compiled_begin, 1u);
  EXPECT_EQ(probe.compiled_count, 2u);
  EXPECT_EQ(probe.single_prepare_count, 1u);
  bool fallback_on_second_target = false;
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    auto step = compiled.report.step(index);
    ASSERT_TRUE(step.has_value());
    fallback_on_second_target =
        fallback_on_second_target ||
        (step->kind == CompileStepKind::kOperationFallback &&
         step->target == environment.target(1));
  }
  EXPECT_TRUE(fallback_on_second_target);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue region_queue = region_context.create_queue().value();
  Queue fallback_queue = fallback_context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &region_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &fallback_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  std::array<float, 8> expected{};
  for (std::size_t index = 0; index < values.size(); ++index)
    expected[index] = -values[index];
  EXPECT_EQ(read(output, expected.size()),
            std::vector<float>(expected.begin(), expected.end()));
}

TEST(GraphRegions, IgnoresUnusedEnvironmentBackendsInPlanSummary) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  Context unused = make_context(probe);
  const std::array native_only{hyle::compute::native::provider()};
  Context active =
      Context::create(hyle::compute::cpu::device(), {.providers = native_only})
          .value();
  const NegationGraph logical = make_negations(1);
  const auto host = hyle::mem::system_resource();
  const Environment environment{{&unused, host}, {&active, host}};
  const std::array placements{hyle::graph::NodePlacement{
      .node = logical.graph.node_id().value(),
      .target = environment.target(1),
      .strength = hyle::graph::PlacementStrength::kRequired,
  }};
  const std::array resources{
      hyle::graph::ExternalResourceContract{
          .value = logical.input,
          .storage = environment.storage(1)},
      hyle::graph::ExternalResourceContract{
          .value = logical.output,
          .storage = environment.storage(1)},
  };
  const auto compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  EXPECT_FALSE(compiled.report.lane(environment.lane(0).ordinal())
                   ->queue_required);
  EXPECT_TRUE(compiled.report.lane(environment.lane(1).ordinal())
                  ->queue_required);
  EXPECT_EQ(compiled.report.backend(), hyle::compute::cpu::kBackendId);
  EXPECT_FALSE(compiled.report.mixed_backends());
}

TEST(GraphRegions, DefaultPolicyAcceptsUnknownCountsAsExplicitOptIn) {
  RegionProbe probe{.proposal_dispatch_count = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.single_prepare_count, 0u);
}

TEST(GraphRegions, MixedUnknownCountsUseATransitiveDeterministicFallback) {
  for (auto mode : {ProposalMode::kPairs, ProposalMode::kNonContiguous}) {
    Graph graph = mode == ProposalMode::kPairs
                      ? make_negations(2).graph
                      : make_non_contiguous_negations().graph;
    // Pairwise "compare when both known" would cycle: first < unknown by
    // provider order, unknown < third by order, third < first by count.
    RegionProbe first{.mode = mode, .proposal_dispatch_count = 3};
    RegionProbe unknown{.mode = mode, .proposal_dispatch_count = std::nullopt};
    RegionProbe third{.mode = mode, .proposal_dispatch_count = 1};
    const std::array providers{region_provider(first), region_provider(unknown),
                               region_provider(third),
                               hyle::compute::native::provider()};
    Context context =
        Context::create(hyle::compute::cpu::device(), {.providers = providers})
            .value();
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto compiled = hyle::graph::compile(graph, context);
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      EXPECT_EQ(compiled->report.region_count(), 1u);
    }
    EXPECT_EQ(first.compile_count, 3u);
    EXPECT_EQ(unknown.compile_count, 0u);
    EXPECT_EQ(third.compile_count, 0u);
  }
}

TEST(GraphRegions, ZeroDispatchCountIsMalformedNotZeroWork) {
  RegionProbe probe{.proposal_dispatch_count = 0};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, DefaultPolicyUsesEarlierProviderAsTheFinalTieBreak) {
  RegionProbe first;
  RegionProbe second;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
}

TEST(GraphRegions, DefaultPolicyPrefersFewerDispatchEpisodesBeforeProvider) {
  RegionProbe first{.proposal_dispatch_count = 2};
  RegionProbe second{.proposal_dispatch_count = 1};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 1u);
}

TEST(GraphRegions, RejectsAnUnknownRegionSelectionPolicy) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      CompileOptions{.region_selection =
                         static_cast<RegionSelectionPolicy>(0xff)});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInvalidArgument());
  EXPECT_EQ(probe.propose_count, 0u);
}

TEST(GraphRegions, DisabledPolicyUsesOnlyExplicitSingleOperationFallback) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          CompileOptions{.region_selection = RegionSelectionPolicy::kDisabled})
          .value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.propose_count, 0u);
  EXPECT_EQ(probe.single_prepare_count, 2u);
  auto fallback = compiled.report.step(0);
  ASSERT_TRUE(fallback.has_value());
  EXPECT_EQ(fallback->fallback_reason,
            CompileFallbackReason::kRegionSelectionDisabled);
}

TEST(GraphRegions, ProposalLimitReturnsAndReportsAValidatedExecutable) {
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto compiled = hyle::graph::compile(
        logical.graph, context, CompileOptions{.max_region_proposals = 1});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    const auto search = compiled->report.search();
    EXPECT_TRUE(search.optimization_limited());
    EXPECT_TRUE(search.region_proposal_limit_reached);
    EXPECT_FALSE(search.per_provider_proposal_limit_reached);
    EXPECT_FALSE(search.region_node_limit_reached);
    EXPECT_EQ(search.region_proposals, 1u);
    EXPECT_EQ(search.region_proposal_nodes, 2u);
    EXPECT_EQ(search.placement_attempts, 1u);
    EXPECT_EQ(compiled->report.region_count(), 1u);
    EXPECT_EQ(compiled->report.fallback_operation_count(), 0u);
    OwnedTensor input = make_tensor(row(8));
    OwnedTensor output = make_tensor(row(8));
    const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
    write(input, values);
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
    ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
    EXPECT_EQ(read(output, values.size()),
              std::vector<float>(values.begin(), values.end()));
  }
  EXPECT_EQ(probe.compile_count, 3u);
  EXPECT_EQ(probe.single_prepare_count, 0u);
}

TEST(GraphRegions, PerProviderLimitStillAllowsDiscoveryFromTheNextProvider) {
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  RegionProbe second;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      CompileOptions{.max_region_proposals = 4,
                     .max_region_proposals_per_provider = 1});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_TRUE(compiled->report.search().per_provider_proposal_limit_reached);
  EXPECT_FALSE(compiled->report.search().region_proposal_limit_reached);
  EXPECT_EQ(compiled->report.search().region_proposals, 2u);
  EXPECT_EQ(compiled->report.search().region_proposal_nodes, 4u);
  EXPECT_EQ(second.propose_count, 1u);
}

TEST(GraphRegions, StrictSearchRejectsLimitedDiscoveryBeforePreparation) {
  const auto logical = make_negations(2);
  for (unsigned limit = 0; limit != 3; ++limit) {
    RegionProbe probe{.mode = ProposalMode::kOverflow};
    Context context = make_context(probe);
    CompileOptions options{.search_policy =
                               CompileSearchPolicy::kRequireUnexhausted};
    if (limit == 0)
      options.max_region_proposals = 1;
    else if (limit == 1)
      options.max_region_proposals_per_provider = 1;
    else
      options.max_region_proposal_nodes = 1;
    auto compiled = hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsResourceExhausted());
    EXPECT_EQ(probe.compile_count, 0u);
    EXPECT_EQ(probe.single_prepare_count, 0u);
  }
}

TEST(GraphRegions, ExactCapacityWithoutOmissionsIsNotLimitedSearch) {
  RegionProbe probe;
  const auto logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.max_region_proposals = 1,
       .max_region_proposals_per_provider = 1,
       .max_region_proposal_nodes = 2,
       .search_policy = CompileSearchPolicy::kRequireUnexhausted});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_FALSE(compiled->report.search().optimization_limited());
  EXPECT_EQ(compiled->report.search().region_proposals, 1u);
  EXPECT_EQ(compiled->report.search().region_proposal_nodes, 2u);
  EXPECT_FALSE(CompileReport{}.search().optimization_limited());
}

TEST(GraphRegions, GlobalLimitSkipsLaterTargetsButPreservesRequiredPlacement) {
  RegionProbe first;
  RegionProbe second;
  const auto logical = make_negations(2);
  Context first_context = make_context(first);
  Context second_context = make_context(second);
  const auto host = hyle::mem::system_resource();
  const Environment environment{{&first_context, host},
                                {&second_context, host}};
  const std::array placements{hyle::graph::NodePlacement{
      .node = logical.graph.node_id(0).value(),
      .target = environment.target(1),
      .strength = hyle::graph::PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.node_placements = placements, .max_region_proposals = 1});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_TRUE(compiled->report.search().region_proposal_limit_reached);
  EXPECT_EQ(compiled->report.search().region_proposals, 1u);
  EXPECT_EQ(second.propose_count, 0u);
  EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
}

TEST(GraphRegions, LimitedDiscoveryNeverHidesRealProviderErrors) {
  const auto logical = make_negations(2);
  for (const Status failure :
       {Status::ResourceExhausted("provider scratch allocation failed"),
        Status::Internal("provider consistency check failed")}) {
    RegionProbe probe{.mode = ProposalMode::kOverflow};
    probe.discovery_failure = failure;
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context,
                                         {.max_region_proposals = 1});
    ASSERT_FALSE(compiled.has_value());
    EXPECT_EQ(compiled.error(), failure);
    EXPECT_EQ(probe.compile_count, 0u);
    EXPECT_EQ(probe.single_prepare_count, 0u);
  }
}

TEST(GraphRegions, LimitedSearchCannotWaiveWorkspaceOrPreparationLimits) {
  const auto logical = make_negations(2);
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  probe.artifact_properties.workspace = {.size = 128, .alignment = 64};
  Context context = make_context(probe);
  const std::array storage{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 0}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  auto no_space = hyle::graph::compile(
      logical.graph, {.storage_domains = storage, .targets = targets},
      {.max_region_proposals = 1});
  ASSERT_FALSE(no_space.has_value());
  EXPECT_TRUE(no_space.error().IsResourceExhausted());
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);

  auto incomplete = hyle::graph::compile(
      logical.graph, context,
      {.max_placement_attempts = 1, .max_region_proposal_nodes = 1});
  ASSERT_FALSE(incomplete.has_value());
  EXPECT_TRUE(incomplete.error().IsResourceExhausted());
}

TEST(GraphRegions, UnknownSearchPolicyIsRejectedBeforeDiscovery) {
  RegionProbe probe;
  const auto logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.search_policy = static_cast<CompileSearchPolicy>(255)});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInvalidArgument());
  EXPECT_EQ(probe.propose_count, 0u);
}

TEST(GraphRegions, WriterSeparatesCapacityRefusalFromValidationErrors) {
  const auto logical = make_negations(2);
  std::array<RegionProposal, 1> proposals;
  std::array<NodeId, 2> nodes;
  RegionProposalWriter writer(proposals, nodes);
  const std::array requested{logical.graph.node_id(1).value(),
                             logical.graph.node_id(0).value()};
  ASSERT_TRUE(writer.try_add({.nodes = requested}).value());
  EXPECT_EQ(writer.proposals()[0].nodes[0], requested[1]);
  EXPECT_FALSE(writer.try_add({.nodes = requested}).value());
  EXPECT_TRUE(writer.proposal_limit_reached());
  EXPECT_EQ(writer.size(), 1u);
  EXPECT_EQ(writer.node_size(), 2u);
  auto invalid = writer.try_add({});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
}

TEST(GraphRegions, RejectFallbackPolicyRequiresCompleteRegionCoverage) {
  RegionProbe probe;
  Context context = make_context(probe);
  const NegationGraph complete = make_negations(2);
  EXPECT_TRUE(
      hyle::graph::compile(complete.graph, context,
                           CompileOptions{.operation_fallback =
                                              OperationFallbackPolicy::kReject})
          .has_value());

  const NegationGraph partial = make_negations(2, true);
  auto rejected = hyle::graph::compile(
      partial.graph, context,
      CompileOptions{.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsUnimplemented());
}

TEST(GraphRegions, DetailedReportOwnsAndEnumeratesTheCompiledPlan) {
  RegionProbe probe;
  probe.artifact_properties = {
      .persistent_bytes = 128,
      .copy_bytes = 64,
      .synchronization_count = 2,
      .dispatch_count = 3,
      .runtime = {
          .blocks_host = true,
          .uses_private_threads = true,
          .supports_concurrent_invocations = false,
      },
  };
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  CompileReport retained = compiled.report;
  compiled.report = {};
  compiled.executable = {};
  EXPECT_EQ(probe.destroy_prepared_count, 1u);

  ASSERT_TRUE(retained.valid());
  ASSERT_EQ(retained.step_count(), 1u);
  EXPECT_EQ(retained.copy_bytes(), 64u);
  EXPECT_EQ(retained.synchronization_count(), 2u);
  auto step = retained.step(0);
  ASSERT_TRUE(step.has_value()) << step.error().message();
  EXPECT_EQ(step->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(step->provider.id, kRegionProviderId);
  EXPECT_EQ(step->node_begin, 0u);
  EXPECT_EQ(step->node_count, 2u);
  ASSERT_TRUE(step->persistent_bytes.has_value());
  EXPECT_EQ(*step->persistent_bytes, 128u);
  EXPECT_EQ(step->copy_bytes, 64u);
  EXPECT_EQ(step->synchronization_count, 2u);
  EXPECT_EQ(step->dispatch_count, 3u);
  EXPECT_TRUE(step->runtime.blocks_host);
  EXPECT_TRUE(step->runtime.uses_private_threads);
  EXPECT_FALSE(step->runtime.supports_concurrent_invocations);
  EXPECT_TRUE(retained.step(1).error().IsOutOfRange());

  retained = {};
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
}

TEST(GraphRegions, CostFactsPreserveUnknownAndExactZero) {
  const NegationGraph logical = make_negations(2);

  RegionProbe unknown_probe;
  Context unknown_context = make_context(unknown_probe);
  auto unknown = hyle::graph::compile(logical.graph, unknown_context).value();
  auto unknown_step = unknown.report.step(0);
  ASSERT_TRUE(unknown_step.has_value()) << unknown_step.error().message();
  EXPECT_FALSE(unknown_step->persistent_bytes.has_value());

  RegionProbe zero_probe;
  zero_probe.artifact_properties.persistent_bytes = 0;
  Context zero_context = make_context(zero_probe);
  auto zero = hyle::graph::compile(logical.graph, zero_context).value();
  auto zero_step = zero.report.step(0);
  ASSERT_TRUE(zero_step.has_value()) << zero_step.error().message();
  ASSERT_TRUE(zero_step->persistent_bytes.has_value());
  EXPECT_EQ(*zero_step->persistent_bytes, 0u);
}

TEST(GraphRegions, SerializesArtifactsThatRejectConcurrentInvocations) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.artifact_properties.runtime.supports_concurrent_invocations = false;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_FALSE(compiled.report.supports_concurrent_invocations());
  EXPECT_FALSE(compiled.executable.supports_concurrent_invocations());

  OwnedTensor first_input = make_tensor(row(8));
  OwnedTensor first_output = make_tensor(row(8));
  OwnedTensor second_input = make_tensor(row(8));
  OwnedTensor second_output = make_tensor(row(8));
  auto first = compiled.executable.create_invocation().value();
  auto second = compiled.executable.create_invocation().value();
  ASSERT_TRUE(first.bind(logical.input, first_input.view).ok());
  ASSERT_TRUE(first.bind(logical.output, first_output.view).ok());
  ASSERT_TRUE(second.bind(logical.input, second_input.view).ok());
  ASSERT_TRUE(second.bind(logical.output, second_output.view).ok());
  Queue queue = context.create_queue().value();

  auto first_completion = first.submit(queue);
  ASSERT_TRUE(first_completion.has_value())
      << first_completion.error().message();
  EXPECT_TRUE(first.in_flight());
  auto rejected = second.submit(queue);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_FALSE(second.in_flight());

  ASSERT_TRUE(first_completion->wait().ok());
  auto second_completion = second.submit(queue);
  ASSERT_TRUE(second_completion.has_value())
      << second_completion.error().message();
  EXPECT_TRUE(second_completion->wait().ok());
}

TEST(GraphRegions, RegionExtensionVersionDoesNotAliasTheV1Table) {
  constexpr hyle::compute::ProviderExtensionId version_one{
      0x48594c4547524150ULL, 0x48524547494f4e31ULL, 1};
  constexpr hyle::compute::ProviderExtensionId version_two{
      0x48594c4547524150ULL, 0x48524547494f4e31ULL, 2};
  EXPECT_EQ(query_extension(nullptr, version_one), nullptr);
  EXPECT_EQ(query_extension(nullptr, version_two), nullptr);
  EXPECT_EQ(query_extension(nullptr, hyle::graph::kRegionProviderExtensionId),
            &kRegionVTable);
}

TEST(GraphRegions,
     ExplicitRegionDeclinesReleasePartialArtifactsBeforeFallback) {
  const auto logical = make_negations(2);
  const std::array declines{
      ProviderPrepareOutcome::unsupported(
          Status::Unimplemented("safe decline")),
      ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("candidate-local budget"))};
  for (const auto &decline : declines) {
    RegionProbe probe;
    probe.region_outcome_override = decline;
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(probe.compile_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
    EXPECT_EQ(probe.destroy_count_at_single_proposal, 1u);
    EXPECT_EQ(compiled->report.region_count(), 0u);
    EXPECT_EQ(compiled->report.fallback_operation_count(), 2u);
    OwnedTensor input = make_tensor(row(8));
    OwnedTensor output = make_tensor(row(8));
    const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
    write(input, values);
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
    ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
    EXPECT_EQ(read(output, values.size()),
              std::vector<float>(values.begin(), values.end()));

    auto forbidden = hyle::graph::compile(
        logical.graph, context,
        {.operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_FALSE(forbidden.has_value());
    EXPECT_EQ(forbidden.error(), decline.status);
    EXPECT_EQ(probe.destroy_prepared_count, 2u);
  }
}

TEST(GraphRegions, RegionErrorCodesDoNotGrantPermissionToRetry) {
  const auto logical = make_negations(2);
  for (const Status error :
       {Status::Unimplemented("fatal region compile failure"),
        Status::ResourceExhausted("real region allocation failure")}) {
    RegionProbe probe;
    probe.region_outcome_override = ProviderPrepareOutcome::error(error);
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_EQ(compiled.error(), error);
    EXPECT_EQ(probe.compile_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
    EXPECT_EQ(probe.single_prepare_count, 0u);
  }
}

TEST(GraphRegions, InvalidRegionOutcomeIsFatalAndReclaimsItsArtifact) {
  const auto logical = make_negations(2);
  const std::array invalid{
      ProviderPrepareOutcome{static_cast<ProviderPrepareOutcomeKind>(255),
                             Status::Unimplemented("invalid kind")},
      ProviderPrepareOutcome{ProviderPrepareOutcomeKind::kPrepared,
                             Status::Unimplemented("prepared with error")},
      ProviderPrepareOutcome::unsupported(
          Status::ResourceExhausted("wrong code")),
      ProviderPrepareOutcome::error(Status::Ok())};
  for (const auto &outcome : invalid) {
    EXPECT_FALSE(outcome.valid());
    RegionProbe probe;
    probe.region_outcome_override = outcome;
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsInternal());
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
    EXPECT_EQ(probe.single_prepare_count, 0u);
  }
}

TEST(GraphRegions, ExplicitLeafDeclinesCanSelectAnotherTarget) {
  const auto logical = make_negations(1);
  const std::array declines{
      ProviderPrepareOutcome::unsupported(
          Status::Unimplemented("safe leaf decline")),
      ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("leaf candidate-local budget"))};
  for (const auto &decline : declines) {
    RegionProbe probe;
    probe.required_resource = hyle::mem::system_resource();
    probe.single_outcome_override = decline;
    const std::array providers{region_provider(probe)};
    auto first =
        Context::create(hyle::compute::cpu::device(), {.providers = providers})
            .value();
    auto second = Context::create(hyle::compute::cpu::device(),
                                  hyle::compute::native::provider())
                      .value();
    const Environment environment{{&first, probe.required_resource},
                                  {&second, probe.required_resource}};
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.region_selection = RegionSelectionPolicy::kDisabled});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(probe.single_artifact_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
    EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
    EXPECT_EQ(compiled->report.search().placement_attempts, 2u);
  }
}

TEST(GraphRegions, LeafErrorDoesNotBecomeARetryAcrossProvidersOrTargets) {
  const auto logical = make_negations(1);
  for (const Status error :
       {Status::Unimplemented("fatal leaf compile failure"),
        Status::ResourceExhausted("real leaf allocation failure")}) {
    RegionProbe probe;
    probe.required_resource = hyle::mem::system_resource();
    probe.single_outcome_override = ProviderPrepareOutcome::error(error);
    auto first = make_context(probe);
    auto second = Context::create(hyle::compute::cpu::device(),
                                  hyle::compute::native::provider())
                      .value();
    const Environment environment{{&first, probe.required_resource},
                                  {&second, probe.required_resource}};
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.region_selection = RegionSelectionPolicy::kDisabled});
    ASSERT_FALSE(compiled.has_value());
    EXPECT_EQ(compiled.error().code(), error.code());
    EXPECT_NE(compiled.error().message().find(error.message()),
              std::string_view::npos);
    EXPECT_EQ(probe.single_artifact_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
  }
}

TEST(GraphRegions, LeafErrorsRemainFatalDuringBothRegionFallbackStages) {
  for (const ProposalMode mode :
       {ProposalMode::kOverlappingRoutes, ProposalMode::kPairs}) {
    const auto logical = make_negations(mode == ProposalMode::kPairs ? 2 : 3);
    RegionProbe probe{.mode = mode};
    probe.required_resource = hyle::mem::system_resource();
    probe.single_outcome_override = ProviderPrepareOutcome::error(
        Status::Unimplemented("fatal leaf fallback failure"));
    if (mode == ProposalMode::kPairs) {
      probe.region_outcome_override = ProviderPrepareOutcome::unsupported(
          Status::Unimplemented("safe region decline"));
    }
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsUnimplemented());
    EXPECT_NE(compiled.error().message().find("fatal leaf fallback failure"),
              std::string_view::npos);
    EXPECT_EQ(probe.single_artifact_count, 1u);
    const std::size_t regions_prepared = mode == ProposalMode::kPairs ? 1u : 0u;
    EXPECT_EQ(probe.compile_count, regions_prepared);
    EXPECT_EQ(probe.destroy_prepared_count, regions_prepared + 1);
  }
}

TEST(GraphRegions, RollsBackPartiallyCreatedProviderState) {
  {
    RegionProbe probe{.mode = ProposalMode::kCompileFailure};
    const NegationGraph logical = make_negations(2);
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_TRUE(compiled.error().IsInternal());
    EXPECT_EQ(probe.compile_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
  }
  {
    RegionProbe probe{.mode = ProposalMode::kInvocationFailure};
    const NegationGraph logical = make_negations(2);
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context).value();
    auto invocation = compiled.executable.create_invocation();
    ASSERT_FALSE(invocation.has_value());
    EXPECT_TRUE(invocation.error().IsInternal());
    EXPECT_EQ(probe.create_invocation_count, 1u);
    EXPECT_EQ(probe.destroy_invocation_count, 1u);
  }
}

TEST(GraphRegions, InvocationRetainsItsCompiledRegionAndWorkspaceOwner) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  Queue queue = context.create_queue().value();
  hyle::graph::Invocation invocation;
  {
    auto compiled = hyle::graph::compile(logical.graph, context).value();
    invocation = compiled.executable.create_invocation().value();
  }
  EXPECT_EQ(probe.destroy_prepared_count, 0u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));

  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
}

TEST(GraphRegions,
     OrderedSequenceDoesNotHostWaitAndRetainsInvocationUntilTerminal) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.report.region_count(), 2u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();

  auto submitted = invocation.submit(queue);
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message();
  EXPECT_EQ(probe.submit_count, 2u);
  EXPECT_EQ(probe.completion_wait_count, 0u);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(invocation.reset().IsFailedPrecondition());
  EXPECT_TRUE(
      invocation.bind(logical.input, input.view).IsFailedPrecondition());
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
  auto pending = submitted->poll();
  ASSERT_TRUE(pending.has_value());
  EXPECT_EQ(*pending, hyle::compute::CompletionPoll::kPending);
  EXPECT_TRUE(submitted->wait().ok());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_EQ(probe.completion_destroy_count, 1u);
  ASSERT_TRUE(invocation.reset().ok());

  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Completion abandoned = invocation.submit(queue).value();
  EXPECT_EQ(probe.submit_count, 4u);
  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 0u);
  abandoned = {};
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.destroy_invocation_count, 0u);

  probe.completed_prefix = probe.submit_count;
  Status terminal = async_terminal(probe);
  if (terminal.ok() && !probe.abandoned_hint.ok())
    terminal = probe.abandoned_hint;
  probe.abandoned_lifetime.terminal(probe.abandoned_lifetime.state, terminal);
  probe.abandoned_lifetime.release(probe.abandoned_lifetime.state);
  probe.abandoned_lifetime = {};
  EXPECT_EQ(probe.destroy_invocation_count, 2u);
}

TEST(GraphRegions, RejectedTailIsReportedAfterTheAcceptedAsyncPrefix) {
  RegionProbe probe{.mode = ProposalMode::kAsyncRejectSecond};
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();

  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_EQ(probe.submit_attempt_count, 2u);
  EXPECT_EQ(probe.submit_count, 1u);
  EXPECT_EQ(probe.completion_wait_count, 0u);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(completion->wait().IsUnavailable());
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_FALSE(invocation.in_flight());
}

TEST(GraphRegions,
     CrossDomainWorkspaceReuseWaitsAndEveryDomainReportsLateFailure) {
  RegionProbe first_probe{.mode = ProposalMode::kAsyncPairs};
  RegionProbe second_probe{.mode = ProposalMode::kAsyncPairs};
  second_probe.terminal_failure = true;
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(8)).value();
  const ValueId second_input = builder.input(row(8)).value();
  ValueId crossing = builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  crossing = builder.add(Unary{UnaryKind::kNeg}, crossing).value();
  const auto remote_head =
      builder.add_node(Operation{Unary{UnaryKind::kNeg}}, std::array{crossing})
          .value();
  ValueId remote = builder.result(remote_head).value();
  const auto remote_tail =
      builder.add_node(Operation{Unary{UnaryKind::kNeg}}, std::array{remote})
          .value();
  remote = builder.result(remote_tail).value();
  const auto local_head =
      builder
          .add_node(Operation{Unary{UnaryKind::kNeg}}, std::array{second_input})
          .value();
  ASSERT_TRUE(builder.depends_on(local_head, remote_tail).ok());
  ValueId local = builder.result(local_head).value();
  local = builder.add(Unary{UnaryKind::kNeg}, local).value();
  ASSERT_TRUE(builder.publish(remote).ok());
  ASSERT_TRUE(builder.publish(local).ok());
  Graph graph = std::move(builder).finish().value();

  const Environment environment{
      {&first, hyle::mem::system_resource()},
      {&second, hyle::mem::system_resource()},
  };
  std::array<hyle::graph::NodePlacement, 6> placements{};
  for (std::size_t index = 0; index < placements.size(); ++index) {
    placements[index] = hyle::graph::NodePlacement{
        .node = graph.node_id(index).value(),
        .target = environment.target(index >= 2 && index < 4 ? 1u : 0u),
        .strength = hyle::graph::PlacementStrength::kRequired,
    };
  }
  auto compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements})
          .value();
  ASSERT_EQ(compiled.report.segment_count(), 3u);
  EXPECT_TRUE(compiled.report.segment(2)->terminal);
  EXPECT_FALSE(compiled.report.segment(1)->terminal);
  bool waits_for_remote_read = false;
  for (std::size_t index = 0;
       index < compiled.report.segment(2)->dependency_count; ++index) {
    waits_for_remote_read =
        waits_for_remote_read ||
        compiled.report.segment_dependency(2, index)->source_segment == 1;
  }
  EXPECT_TRUE(waits_for_remote_read);

  OwnedTensor first_source = make_tensor(row(8));
  OwnedTensor second_source = make_tensor(row(8));
  OwnedTensor remote_output = make_tensor(row(8));
  OwnedTensor local_output = make_tensor(row(8));
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(remote, remote_output.view).ok());
  ASSERT_TRUE(invocation.bind(local, local_output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &first_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto replay = invocation.replay_report().value();
  EXPECT_EQ(replay.ordered_dependency_count, 1u);
  EXPECT_EQ(replay.device_wait_dependency_count, 2u);
  EXPECT_EQ(replay.ordered_dependency_count +
                replay.device_wait_dependency_count,
            compiled.report.segment_dependency_count());
  EXPECT_EQ(replay.host_wait_dependency_count, 0u);
  auto completion = invocation.submit(queues);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_EQ(first_probe.device_wait_count, 1u);
  EXPECT_EQ(second_probe.device_wait_count, 1u);
  EXPECT_TRUE(completion->wait().IsInternal());
  EXPECT_EQ(first_probe.completion_wait_count, 1u);
  EXPECT_EQ(second_probe.completion_wait_count, 1u);
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_TRUE(invocation.output(remote).error().IsFailedPrecondition());
}

TEST(GraphRegions, MultiTargetIndependentSinksUseANonblockingAggregate) {
  RegionProbe first_probe{.mode = ProposalMode::kAsyncPairs};
  RegionProbe second_probe{.mode = ProposalMode::kAsyncPairs};
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(8)).value();
  const ValueId second_input = builder.input(row(8)).value();
  const ValueId first_middle =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_middle).value();
  const ValueId second_middle =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  const ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_middle).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  const Environment environment{
      {&first, hyle::mem::system_resource()},
      {&second, hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = graph.node_id(0).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(1).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(2).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(3).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  const std::array resources{
      hyle::graph::ExternalResourceContract{
          .value = first_input,
          .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{
          .value = second_input,
          .storage = environment.storage(1)},
      hyle::graph::ExternalResourceContract{
          .value = first_output,
          .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{
          .value = second_output,
          .storage = environment.storage(1)},
  };
  auto compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.segment_count(), 2u);
  EXPECT_EQ(compiled.report.segment_dependency_count(), 0u);
  EXPECT_EQ(compiled.report.command_segment_count(), 2u);
  EXPECT_EQ(compiled.report.capturable_segment_count(), 0u);
  EXPECT_TRUE(compiled.report.segment(0)->terminal);
  EXPECT_TRUE(compiled.report.segment(1)->terminal);
  EXPECT_FALSE(compiled.report.mixed_backends());

  OwnedTensor first_source = make_tensor(row(8));
  OwnedTensor second_source = make_tensor(row(8));
  OwnedTensor first_destination = make_tensor(row(8));
  OwnedTensor second_destination = make_tensor(row(8));
  const std::array<float, 8> first_values{-4, -3, -2, -1, 1, 2, 3, 4};
  const std::array<float, 8> second_values{-8, -7, -6, -5, 5, 6, 7, 8};
  write(first_source, first_values);
  write(second_source, second_values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &first_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &second_queue},
  };
  EXPECT_TRUE(
      invocation
          .prepare_replay(queues,
                          {.mode = hyle::graph::ReplayMode::kRequireCapture})
          .IsUnimplemented());
  ASSERT_TRUE(invocation
                  .prepare_replay(
                      queues, {.mode = hyle::graph::ReplayMode::kPreferCapture})
                  .ok());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kCommands);
  auto completion = invocation.submit(queues);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_EQ(first_probe.device_wait_count, 0u);
  EXPECT_EQ(second_probe.device_wait_count, 0u);
  EXPECT_EQ(first_probe.completion_wait_count, 0u);
  EXPECT_EQ(second_probe.completion_wait_count, 0u);
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(first_probe.completion_wait_count, 1u);
  EXPECT_EQ(second_probe.completion_wait_count, 1u);
}

TEST(GraphRegions, AbandonedAggregateRetainsEveryIndependentSink) {
  RegionProbe first_probe{.mode = ProposalMode::kAsyncPairs};
  RegionProbe second_probe{.mode = ProposalMode::kAsyncPairs};
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(8)).value();
  const ValueId second_input = builder.input(row(8)).value();
  const ValueId first_middle =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_middle).value();
  const ValueId second_middle =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  const ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_middle).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  const Environment environment{
      {&first, hyle::mem::system_resource()},
      {&second, hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = graph.node_id(0).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(1).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(2).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(3).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  const std::array resources{
      hyle::graph::ExternalResourceContract{
          .value = first_input,
          .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{
          .value = second_input,
          .storage = environment.storage(1)},
      hyle::graph::ExternalResourceContract{
          .value = first_output,
          .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{
          .value = second_output,
          .storage = environment.storage(1)},
  };
  auto compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor first_source = make_tensor(row(8));
  OwnedTensor second_source = make_tensor(row(8));
  OwnedTensor first_destination = make_tensor(row(8));
  OwnedTensor second_destination = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &first_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  auto completion = invocation.submit(queues);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(invocation.in_flight());

  *completion = {};
  EXPECT_EQ(first_probe.completion_wait_count, 0u);
  EXPECT_EQ(second_probe.completion_wait_count, 0u);
  EXPECT_TRUE(first_probe.abandoned_lifetime.valid());
  EXPECT_TRUE(second_probe.abandoned_lifetime.valid());
  EXPECT_TRUE(invocation.in_flight());

  finish_abandoned_prefix(first_probe);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(invocation.output(first_output).error().IsFailedPrecondition());
  finish_abandoned_prefix(second_probe);
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_TRUE(invocation.output(first_output).has_value());
  EXPECT_TRUE(invocation.output(second_output).has_value());
}

TEST(GraphRegions, MultiTargetPlacementKeepsEligibleProviderRegions) {
  RegionProbe first_probe;
  RegionProbe second_probe;
  const NegationGraph logical = make_negations(4);
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);
  const Environment environment{
      {&first, hyle::mem::system_resource()},
      {&second, hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(2).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(3).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled_result =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements});
  ASSERT_TRUE(compiled_result.has_value())
      << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  EXPECT_EQ(compiled.report.region_count(), 2u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 0u);
  ASSERT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.step(0)->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled.report.step(0)->target, environment.target(0));
  EXPECT_EQ(compiled.report.step(1)->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled.report.step(1)->target, environment.target(1));
  EXPECT_EQ(first_probe.compile_count, 1u);
  EXPECT_EQ(second_probe.compile_count, 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &first_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, RejectsSurplusBindingsForASingleActiveLane) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  const Environment environment{{&context, hyle::mem::system_resource()}};
  auto compiled = hyle::graph::compile(logical.graph, environment.environment())
                      .value();
  ASSERT_EQ(compiled.report.segment_count(), 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  auto invocation = compiled.executable.create_invocation().value();
  Queue queue = context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &queue},
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &queue},
  };
  EXPECT_TRUE(invocation.prepare_replay(queues).IsInvalidArgument());
  EXPECT_EQ(probe.reserve_completion_count, 0u);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  EXPECT_TRUE(invocation.prepare_replay(queues).IsInvalidArgument());
  EXPECT_EQ(probe.reserve_completion_count, 0u);
}

TEST(GraphRegions, RejectsAnUnavailableTransferBeforeAnySegmentIsSubmitted) {
  RegionProbe first_probe{.mode = ProposalMode::kAsyncPairs};
  RegionProbe second_probe{.mode = ProposalMode::kAsyncPairs};
  second_probe.transfer_supported = false;
  const NegationGraph logical = make_negations(2);
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);
  HostResource first_storage;
  HostResource second_storage;
  first_probe.required_resource = hyle::mem::MemoryResourceRef(first_storage);
  second_probe.required_resource = hyle::mem::MemoryResourceRef(second_storage);
  const Environment environment{{
      ExecutionSpec{.context = &first,
                    .storage = hyle::mem::MemoryResourceRef(first_storage)},
      ExecutionSpec{.context = &second,
                    .storage = hyle::mem::MemoryResourceRef(second_storage)},
  }};
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled_result =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements});
  ASSERT_TRUE(compiled_result.has_value())
      << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  OwnedTensor input = make_tensor(
      row(8), hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor output = make_tensor(
      row(8), hyle::mem::MemoryResourceRef(second_storage));
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &first_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &second_queue},
  };
  EXPECT_TRUE(invocation.prepare_replay(queues).IsUnimplemented());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(first_probe.submit_count, 0u);
  EXPECT_EQ(second_probe.submit_count, 0u);
  EXPECT_EQ(second_probe.transfer_query_count, 1u);
}

TEST(GraphRegions, ExposesAndCanRejectAHostDependencyBeforeSubmission) {
  RegionProbe first_probe{.mode = ProposalMode::kAsyncPairs};
  RegionProbe second_probe{.mode = ProposalMode::kAsyncPairs};
  second_probe.dependency_wait_supported = false;
  const NegationGraph logical = make_negations(2);
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);
  HostResource first_storage;
  HostResource second_storage;
  first_probe.required_resource = hyle::mem::MemoryResourceRef(first_storage);
  second_probe.required_resource = hyle::mem::MemoryResourceRef(second_storage);
  const Environment environment{{
      ExecutionSpec{.context = &first,
                    .storage = hyle::mem::MemoryResourceRef(first_storage)},
      ExecutionSpec{.context = &second,
                    .storage = hyle::mem::MemoryResourceRef(second_storage)},
  }};
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .target = environment.target(0),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .target = environment.target(1),
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements})
          .value();
  EXPECT_EQ(compiled.report.synchronization_count(), 1u);

  OwnedTensor input = make_tensor(
      row(8), hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor output = make_tensor(
      row(8), hyle::mem::MemoryResourceRef(second_storage));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0),
                                 .queue = &first_queue},
      hyle::graph::QueueBinding{.lane = environment.lane(1),
                                 .queue = &second_queue},
  };
  EXPECT_TRUE(
      invocation
          .prepare_replay(queues,
                          {.host_wait = hyle::graph::HostWaitPolicy::kReject})
          .IsUnimplemented());
  EXPECT_TRUE(invocation.replay_report().error().IsFailedPrecondition());
  EXPECT_EQ(first_probe.reserve_completion_count, 0u);
  EXPECT_EQ(second_probe.reserve_completion_count, 0u);

  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto replay = invocation.replay_report().value();
  EXPECT_EQ(replay.route, hyle::graph::ReplayRoute::kCommands);
  EXPECT_EQ(replay.immediate_transfer_count + replay.queued_transfer_count, 1u);
  EXPECT_EQ(replay.device_wait_dependency_count, 0u);
  EXPECT_EQ(replay.host_wait_dependency_count, 1u);
  EXPECT_EQ(invocation.dependency_route(1, 0).value(),
            hyle::graph::ReplayDependencyRoute::kHostWait);
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  EXPECT_GE(first_probe.completion_wait_count, 1u);
  EXPECT_EQ(second_probe.device_wait_count, 0u);
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
}

} // namespace
