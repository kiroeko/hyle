#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../compute/workload_fixture.h"
#include "backend_internal.h"
#include "completion_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/route_policy.h"
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
using hyle::compute::ResourceAccess;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::graph::CompileFallbackReason;
using hyle::graph::CompileOptions;
using hyle::graph::CompileReport;
using hyle::graph::CompileSearchPolicy;
using hyle::graph::CompileStepKind;
using hyle::graph::EffectResourceBinding;
using hyle::graph::EffectResourceFootprint;
using hyle::graph::EffectResourceId;
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
  kResourceSingletons,
  kResourcePairs,
  kBoundaryWindows,
  kEquivalentRoutes,
};

struct PackingProbe;

struct RegionProbe final {
  // Physical identity is captured by the device factory before Context
  // creation.
  std::uint64_t device_identity = 0;
  ProposalMode mode = ProposalMode::kPairs;
  const hyle::compute::WorkloadContext *required_workload = nullptr;
  hyle::compute::WorkloadContext resource_workload{};
  hyle::compute::WorkloadContext compiled_workload{};
  std::size_t singleton_node = 0;
  std::optional<std::uint32_t> proposal_dispatch_count = 1;
  std::size_t propose_count = 0;
  std::size_t extension_query_count = 0;
  std::size_t compile_count = 0;
  std::size_t single_proposal_count = 0;
  std::size_t single_artifact_count = 0;
  std::size_t region_resource_count = 0;
  std::size_t boundary_checks = 0;
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
  mutable std::atomic<std::size_t> resource_validation_count{0};
  std::size_t copy_count = 0;
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
  hyle::mem::MemoryResourceRef declined_workspace_resource{};
  bool workspace_decline_is_error = false;
  hyle::mem::MemoryResourceRef compiled_input_resource{};
  hyle::mem::MemoryResourceRef compiled_output_resource{};
  hyle::mem::MemoryResourceRef compiled_workspace_resource{};
  bool invocation_workspace_valid = false;
  std::size_t invocation_workspace_size = 0;
  hyle::compute::detail::CompletionLifetime abandoned_lifetime{};
  Status abandoned_hint{};
  hyle::graph::RegionArtifactProperties artifact_properties{};
  hyle::compute::ProviderCompilationStorage compilation_storage{};
  std::size_t discovery_scratch_bytes = 0;
  std::size_t resource_scratch_bytes = 0;
  std::optional<bool> proposed_effects_override{};
  std::size_t observed_effect_count = 0;
  hyle::compute::ResourceAccess observed_effect_access{};
  bool observed_unknown_effects = false;
  std::int32_t last_resource_read = 0;
  std::optional<hyle::compute::ProviderReplayClass> proposed_replay_override{};
  ProviderReplayClass discovery_replay_class = ProviderReplayClass::kUnknown;
  std::optional<ProviderReplayClass> single_replay_override{};
  NodeId foreign_node{};
  std::uint32_t compiled_begin = (std::numeric_limits<std::uint32_t>::max)();
  std::uint32_t compiled_count = 0;
  Status discovery_failure{};
  Status resource_query_failure{};
  std::optional<hyle::compute::ResourceRequirement> requirement_override{};
  std::uint32_t requirement_port = 0; // input, output, workspace
  std::optional<ProviderPrepareOutcome> region_outcome_override{};
  std::optional<ProviderPrepareOutcome> single_outcome_override{};
  std::size_t destroy_count_at_single_preparation = 0;
  std::optional<std::uint64_t> region_outcome_route{};
  const RegionProbe *observe_cleanup = nullptr;
  std::size_t observed_cleanup_at_compile = 0;
  hyle::mem::MemoryResourceRef backend_only_resource{};
  hyle::mem::MemoryResourceRef validation_failure_resource{};
  Status validation_failure{};
  hyle::compute::ProviderImplementationId implementation{};
  std::optional<ConstTensorView> compiled_constant{};
  const std::size_t *constant_live_allocations = nullptr;
  std::size_t constant_allocations_at_destroy = 0;
  PackingProbe *packing = nullptr;
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

struct ResourceTouch final {
  static constexpr hyle::compute::OperationId operation_id{0x48594c45U,
                                                           0x52455306U};
  static constexpr std::string_view operation_name = "test-resource-touch";
  std::int32_t *state = nullptr;
  hyle::compute::ResourceAccess access =
      hyle::compute::ResourceAccess::kReadWrite;
  std::int32_t value = 1;
  hyle::compute::OperationTraits operation_traits() const noexcept {
    return {.effect = hyle::compute::EffectKind::kResourceAccess,
            .materializes = false,
            .resource_count = 1};
  }
  hyle::compute::ResourceAccess resource_access(std::size_t) const noexcept {
    return access;
  }
  Status infer_output_descriptions(std::span<const TensorDesc>,
                                   std::span<TensorDesc>) const noexcept {
    return state != nullptr ? Status::Ok()
                            : Status::InvalidArgument("resource state is null");
  }
};

struct EffectPlan final {
  RegionProbe *probe = nullptr;
  std::array<ResourceTouch, 2> touches{};
  std::size_t count = 0;
  bool region = false;
};

extern const hyle::graph::ProviderRegionPreparedVTable kResourceRegionVTable;

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

struct PackingProbe final {
  hyle::graph::ConstantPackingKey key{1, 0};
  hyle::mem::MemoryResourceRef storage{};
  hyle::mem::MemoryResourceRef second_storage{};
  hyle::compute::ProviderCompilationStorage bounds{16, 16};
  std::size_t bytes = 16;
  std::size_t extra_private_bytes = 0;
  bool unknown_persistent = false;
  bool key_per_node = false;
  bool inconsistent_second_request = false;
  bool invalid_constant = false;
  bool decline_after_pack = false;
  Status factory_failure{};
  std::atomic<std::size_t> factories{0}, destroys{0}, executions{0};
};

struct PackedWeights final {
  PackingProbe *probe = nullptr;
  Buffer buffer;
  Buffer private_metadata;
  std::atomic<std::size_t> borrowers{0};
};

struct PackedRegion final {
  RegionProbe *probe = nullptr;
  const PackedWeights *weights = nullptr;
  float epsilon = 0;
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

// A real mutable external buffer, not an opaque identity standing in for
// addressable bytes. Its mapping and owner outlive each compiled invocation.
struct ResourceStorage final {
  Buffer buffer =
      Buffer::create(hyle::mem::system_resource(), sizeof(std::int32_t), 64)
          .value();
  hyle::mem::HostMapping mapping = buffer.view().map_host().value();

  explicit ResourceStorage(std::int32_t initial = 0) {
    ::new (value()) std::int32_t(initial);
  }
  std::int32_t *value() {
    return reinterpret_cast<std::int32_t *>(mapping.writable_bytes()->data());
  }
  EffectResourceFootprint footprint() const {
    return EffectResourceFootprint::memory(buffer.view());
  }
};

NodeId add_resource_touch(GraphBuilder &builder, EffectResourceId resource,
                          ResourceStorage &storage, ResourceAccess access,
                          std::int32_t value = 1) {
  auto node = builder
                  .add_node(Operation::custom(
                                ResourceTouch{storage.value(), access, value}),
                            std::array<ValueId, 0>{})
                  .value();
  EXPECT_TRUE(builder.bind_resources(node, std::array{resource}).ok());
  return node;
}

class HostResource final : public hyle::mem::MemoryResource {
public:
  std::size_t allocation_count = 0;
  std::size_t live_allocations = 0;

  explicit HostResource(bool host_accessible = true) noexcept
      : host_accessible_(host_accessible) {}

  std::string_view name() const noexcept override {
    return "graph.region-test.host";
  }

  hyle::mem::MemoryProperties properties() const noexcept override {
    return hyle::mem::MemoryProperties{
        .flags = host_accessible_
                     ? hyle::mem::MemoryProperty::kHostAccessible |
                           hyle::mem::MemoryProperty::kHostDeviceCoherent
                     : hyle::mem::MemoryProperty::kNone,
        .domain = hyle::mem::MemoryDomainId{this, 0},
    };
  }

  Result<hyle::mem::ResourceAllocation>
  allocate(std::size_t size, std::size_t alignment) noexcept override {
    auto allocation =
        hyle::mem::system_resource().resource().allocate(size, alignment);
    if (allocation) {
      ++allocation_count;
      ++live_allocations;
    }
    return allocation;
  }

  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    --live_allocations;
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

private:
  bool host_accessible_;
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
  result.buffer =
      Buffer::create(resource, description.required_bytes().value(), 64)
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
    ProviderReplayClass replay_class = ProviderReplayClass::kUnknown,
    hyle::compute::ProviderImplementationId implementation = {}) noexcept {
  return writer.try_add(RegionProposal{
      .nodes = nodes,
      .cost = {.dispatch_count = dispatch_count},
      .route =
          nodes.empty()
              ? 0
              : (static_cast<std::uint64_t>(nodes.front().ordinal()) << 32) |
                    nodes.size(),
      .replay_class = replay_class,
      .implementation = implementation,
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
  if (probe.packing != nullptr) {
    for (std::size_t index = 0; index < graph.node_count(); ++index) {
      const std::array nodes{graph.node_id(index).value()};
      auto added = add_node_set_proposal(
          writer, nodes, 1, probe.discovery_replay_class, probe.implementation);
      if (!added || !*added)
        return added;
    }
    return true;
  }
  if (probe.mode == ProposalMode::kBoundaryWindows) {
    for (std::size_t begin = 0; begin < graph.node_count(); ++begin) {
      for (std::size_t count = 1;
           count <= 3 && begin + count <= graph.node_count(); ++count) {
        std::array<NodeId, 3> nodes{};
        for (std::size_t index = 0; index < count; ++index)
          nodes[index] = graph.node_id(begin + index).value();
        auto added =
            add_node_set_proposal(writer, std::span(nodes).first(count));
        if (!added || !*added)
          return added;
      }
    }
    return true;
  }
  if (probe.mode == ProposalMode::kResourceSingletons ||
      probe.mode == ProposalMode::kResourcePairs) {
    const std::size_t width =
        probe.mode == ProposalMode::kResourcePairs ? 2 : 1;
    for (std::size_t index = 0; index < graph.node_count(); index += width) {
      std::array<NodeId, 2> nodes{};
      const auto count = (std::min)(width, graph.node_count() - index);
      for (std::size_t offset = 0; offset < count; ++offset)
        nodes[offset] = graph.node_id(index + offset).value();
      auto added = add_node_set_proposal(writer, std::span(nodes).first(count));
      if (!added || !*added)
        return added;
    }
    return true;
  }
  if (probe.mode == ProposalMode::kSingleton) {
    const std::array nodes{graph.node_id(probe.singleton_node).value()};
    return add_node_set_proposal(writer, nodes, probe.proposal_dispatch_count,
                                 probe.discovery_replay_class,
                                 probe.implementation);
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
  if (probe.mode == ProposalMode::kEquivalentRoutes) {
    auto first = add_proposal(graph, writer, 0, 2, 2);
    if (!first || !*first)
      return first;
    const std::array nodes{graph.node_id(0).value(), graph.node_id(1).value()};
    return writer.try_add(RegionProposal{
        .nodes = nodes, .cost = {.dispatch_count = 1}, .route = 3});
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
    return add_node_set_proposal(writer, nodes, probe.proposal_dispatch_count,
                                 ProviderReplayClass::kUnknown,
                                 probe.implementation);
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
                       AllocatorRef allocator,
                       RegionProposalWriter &writer) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  if (probe.discovery_scratch_bytes != 0) {
    void *scratch = allocator.allocate(probe.discovery_scratch_bytes,
                                       alignof(std::max_align_t));
    if (scratch == nullptr)
      return Status::ResourceExhausted(
          "region discovery scratch allocation failed");
    allocator.deallocate(scratch, probe.discovery_scratch_bytes,
                         alignof(std::max_align_t));
  }
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
  if (plan->probe->constant_live_allocations != nullptr)
    plan->probe->constant_allocations_at_destroy =
        *plan->probe->constant_live_allocations;
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
    if (!dependency.pending() || dependency.vtable != &kAsyncCompletionVTable) {
      return std::unexpected(Status::Unimplemented(
          "delegated test region cannot import this completion"));
    }
    ++plan.probe->device_wait_count;
  }
  ++plan.probe->submit_attempt_count;
  Status terminal = execute_region(prepared, invocation, inputs, outputs);
  if (!terminal.ok())
    return Completion::completed(std::move(terminal));
  return hyle::compute::detail::CompletionAccess::make(&kAsyncCompletionVTable,
                                                       plan.probe);
}

hyle::compute::detail::CompletionEndpoint
delegated_region_endpoint(void *, void *invocation) noexcept {
  auto *run = static_cast<RegionRun *>(invocation);
  return run == nullptr ? hyle::compute::detail::CompletionEndpoint{}
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

Status pack_weights(void *context, ConstTensorView source,
                    hyle::mem::MemoryResourceRef resource,
                    AllocatorRef allocator,
                    hyle::graph::ConstantArtifactPayload *output) noexcept {
  auto &probe = *static_cast<PackingProbe *>(context);
  ++probe.factories;
  void *storage =
      allocator.allocate(sizeof(PackedWeights), alignof(PackedWeights));
  if (storage == nullptr)
    return Status::ResourceExhausted("packed weights owner allocation failed");
  auto *packed = ::new (storage)
      PackedWeights{.probe = &probe, .buffer = {}, .private_metadata = {}};
  *output = {packed, +[](void *opaque, AllocatorRef owner) noexcept {
               auto *weights = static_cast<PackedWeights *>(opaque);
               EXPECT_EQ(weights->borrowers.load(), 0U);
               ++weights->probe->destroys;
               weights->~PackedWeights();
               owner.deallocate(weights, sizeof(PackedWeights),
                                alignof(PackedWeights));
             }};
  HYLE_ASSIGN_OR_RETURN(packed->buffer, Buffer::create(resource, 16, 64));
  if (probe.extra_private_bytes != 0) {
    HYLE_ASSIGN_OR_RETURN(
        packed->private_metadata,
        Buffer::create(resource, probe.extra_private_bytes, 64));
  }
  std::array<float, 4> values{};
  HYLE_RETURN_IF_ERROR(
      source.storage().read(std::as_writable_bytes(std::span(values))));
  std::reverse(values.begin(), values.end());
  HYLE_RETURN_IF_ERROR(
      packed->buffer.view().write(std::as_bytes(std::span(values))));
  return probe.factory_failure;
}

const hyle::graph::ProviderRegionPreparedVTable kPackedRegionVTable{
    .destroy_prepared =
        +[](void *opaque, AllocatorRef allocator) noexcept {
          auto *plan = static_cast<PackedRegion *>(opaque);
          if (plan == nullptr)
            return;
          ++plan->probe->destroy_prepared_count;
          const_cast<PackedWeights *>(plan->weights)->borrowers.fetch_sub(1);
          plan->~PackedRegion();
          allocator.deallocate(plan, sizeof(PackedRegion),
                               alignof(PackedRegion));
        },
    .alias_support = &region_alias_support,
    .create_invocation =
        +[](void *prepared, AllocatorRef, hyle::mem::BufferView,
            void **output) noexcept {
          *output =
              prepared; // immutable; this fixture has no mutable run state
          return Status::Ok();
        },
    .destroy_invocation = +[](void *, AllocatorRef) noexcept {},
    .shared_submit =
        +[](void *opaque, void *, Queue &,
            std::span<const ConstTensorView> inputs,
            std::span<const TensorView> outputs) noexcept {
          const auto &plan = *static_cast<PackedRegion *>(opaque);
          const auto execute = [&]() noexcept -> Status {
            if (inputs.size() != 2 || outputs.size() != 1)
              return Status::Internal("packed RMS boundary mismatch");
            std::array<float, 4> gamma{};
            HYLE_RETURN_IF_ERROR(plan.weights->buffer.view().read(
                std::as_writable_bytes(std::span(gamma))));
            HYLE_ASSIGN_OR_RETURN(auto input, inputs[0].storage().map_host());
            HYLE_ASSIGN_OR_RETURN(auto output, outputs[0].storage().map_host());
            HYLE_ASSIGN_OR_RETURN(auto bytes, output.writable_bytes());
            const auto *x =
                reinterpret_cast<const float *>(input.bytes().data());
            auto *y = reinterpret_cast<float *>(bytes.data());
            const auto count = bytes.size() / sizeof(float);
            for (std::size_t base = 0; base < count; base += 4) {
              float sum = 0;
              for (std::size_t i = 0; i < 4; ++i)
                sum += x[base + i] * x[base + i];
              const float scale = 1 / std::sqrt(sum / 4 + plan.epsilon);
              for (std::size_t i = 0; i < 4; ++i)
                y[base + i] = x[base + i] * scale * gamma[3 - i];
            }
            ++plan.weights->probe->executions;
            HYLE_RETURN_IF_ERROR(output.close());
            return input.close();
          };
          return hyle::compute::detail::EnqueueResult::accepted_command(
              execute());
        },
};

ProviderPrepareOutcome
compile_packed_region(RegionProbe &probe, const RegionCompileRequest &request,
                      AllocatorRef allocator,
                      ProviderRegionPrepared *output) noexcept {
  auto &packing = *probe.packing;
  if (request.nodes.size() != 1 || request.inputs.size() != 2)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("packing RMS request"));
  const auto operation = request.graph->operation(request.nodes[0]).value();
  const auto *rms = operation.get_if<hyle::compute::RmsNorm>();
  if (rms == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("packing needs RMS"));
  auto key = packing.key;
  if (packing.key_per_node)
    key.configuration += request.nodes[0].ordinal();
  auto resource =
      packing.storage.valid() ? packing.storage : request.input_resources[1];
  if (packing.second_storage.valid() && request.nodes[0].ordinal() != 0)
    resource = packing.second_storage;
  hyle::graph::ConstantArtifactDeclaration declaration{
      .key = key,
      .storage = resource,
      .packed_bytes = packing.bytes,
      .persistent_bytes =
          packing.unknown_persistent
              ? std::nullopt
              : std::optional<std::size_t>{sizeof(PackedWeights) +
                                           packing.extra_private_bytes},
      .private_storage = packing.bounds};
  hyle::graph::ConstantArtifactView first, second;
  const auto constant =
      packing.invalid_constant ? request.inputs[0] : request.inputs[1];
  auto prepared = request.constant_artifacts.get_or_create(
      constant, declaration, &pack_weights, &packing, &first);
  if (prepared.kind != ProviderPrepareOutcomeKind::kPrepared)
    return prepared;
  if (packing.inconsistent_second_request)
    ++declaration.packed_bytes;
  prepared = request.constant_artifacts.get_or_create(
      constant, declaration, &pack_weights, &packing, &second);
  if (prepared.kind != ProviderPrepareOutcomeKind::kPrepared)
    return prepared;
  if (first.state != second.state)
    return ProviderPrepareOutcome::error(
        Status::Internal("same packing was duplicated"));
  void *memory =
      allocator.allocate(sizeof(PackedRegion), alignof(PackedRegion));
  if (memory == nullptr)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("packed region allocation failed"));
  auto *weights = static_cast<const PackedWeights *>(first.state);
  const_cast<PackedWeights *>(weights)->borrowers.fetch_add(1);
  *output = {.vtable = &kPackedRegionVTable,
             .state =
                 ::new (memory) PackedRegion{&probe, weights, rms->epsilon},
             .properties = probe.artifact_properties};
  ++probe.compile_count;
  return packing.decline_after_pack
             ? ProviderPrepareOutcome::unsupported(
                   Status::Unimplemented("packing route declined"))
             : ProviderPrepareOutcome::prepared();
}

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

// Deliberately independent, slow whole-graph oracle. Product boundary discovery
// uses incidence counts/stamps; the test searches every outside consumer.
bool boundary_matches_full_scan(const RegionCompileRequest &request) {
  const auto &graph = *request.graph;
  const auto contains = [](const auto &values, auto value) {
    return std::find(values.begin(), values.end(), value) != values.end();
  };
  std::vector<ValueId> inputs, outputs;
  for (auto node : request.nodes) {
    for (std::size_t port = 0; port < graph.node_input_count(node).value();
         ++port) {
      const auto value = graph.node_input(node, port).value();
      const auto producer = graph.producer(value);
      if ((!producer || !contains(request.nodes, *producer)) &&
          !contains(inputs, value))
        inputs.push_back(value);
    }
    for (std::size_t port = 0; port < graph.node_output_count(node).value();
         ++port) {
      const auto value = graph.result(node, port).value();
      bool external = false;
      for (std::size_t index = 0; index < graph.output_count(); ++index)
        external |= graph.output_id(index).value() == value;
      for (std::size_t index = 0; index < graph.node_count(); ++index) {
        const auto consumer = graph.node_id(index).value();
        if (contains(request.nodes, consumer))
          continue;
        for (std::size_t input = 0;
             input < graph.node_input_count(consumer).value(); ++input)
          external |= graph.node_input(consumer, input).value() == value;
      }
      if (external)
        outputs.push_back(value);
    }
  }
  return std::equal(inputs.begin(), inputs.end(), request.inputs.begin(),
                    request.inputs.end()) &&
         std::equal(outputs.begin(), outputs.end(), request.outputs.begin(),
                    request.outputs.end());
}

Status propose_region_resources(
    void *context, const RegionCompileRequest &request, AllocatorRef allocator,
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
  probe.resource_workload = request.workload;
  if (probe.required_workload != nullptr) {
    const hyle::compute::RouteSelector selector{
        .platform = hyle::compute::kBuildRoutePlatform,
        .route = hyle::compute::NativeRoute::kCpu,
        .profile = hyle::compute::exact_profile_selector(
            probe.required_workload->profile.description()),
        .phase = probe.required_workload->phase,
        .region = probe.required_workload->region};
    const hyle::compute::RouteMatchInput facts{
        .platform = hyle::compute::kBuildRoutePlatform,
        .route = hyle::compute::NativeRoute::kCpu,
        .profile = request.workload.profile.description(),
        .phase = request.workload.phase,
        .region = request.workload.region};
    if (!hyle::compute::match_route_selector(selector, facts))
      return Status::Unimplemented("model-specific region is not qualified");
  }
  EXPECT_FALSE(request.constant_artifacts.available());
  if (probe.resource_scratch_bytes != 0) {
    void *scratch = allocator.allocate(probe.resource_scratch_bytes,
                                       alignof(std::max_align_t));
    if (scratch == nullptr)
      return Status::ResourceExhausted(
          "region resource query scratch allocation failed");
    allocator.deallocate(scratch, probe.resource_scratch_bytes,
                         alignof(std::max_align_t));
  }
  ++probe.boundary_checks;
  if (!boundary_matches_full_scan(request))
    return Status::Internal("region boundary differs from full-scan oracle");
  if (!probe.resource_query_failure.ok())
    return probe.resource_query_failure;
  probe.observed_effect_count = request.effect_accesses.size();
  if (!request.effect_accesses.empty())
    probe.observed_effect_access = request.effect_accesses.front().access;
  probe.observed_unknown_effects = request.has_unknown_effects;
  if (!input_requirements.empty() && probe.region_input_resource.valid())
    input_requirements[0].exact_resource = probe.region_input_resource;
  if (!output_requirements.empty() && probe.region_output_resource.valid())
    output_requirements[0].exact_resource = probe.region_output_resource;
  *proposal = hyle::graph::RegionResourceProposal{
      .runtime = probe.artifact_properties.runtime,
      .workspace_resource =
          {
              .exact_resource = probe.region_workspace_resource,
          },
      .preserves_effects = probe.proposed_effects_override.value_or(
          probe.artifact_properties.preserves_effects),
      .compilation_storage = probe.compilation_storage,
  };
  if (probe.proposed_replay_override)
    proposal->runtime.replay_class = *probe.proposed_replay_override;
  if (probe.requirement_override) {
    if (probe.requirement_port == 0 && !input_requirements.empty())
      input_requirements[0] = *probe.requirement_override;
    else if (probe.requirement_port == 1 && !output_requirements.empty())
      output_requirements[0] = *probe.requirement_override;
    else
      proposal->workspace_resource = *probe.requirement_override;
  }
  return Status::Ok();
}

Status compile_region_checked(void *context,
                              const RegionCompileRequest &request,
                              AllocatorRef allocator,
                              ProviderRegionPrepared *output) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  if (probe.mode == ProposalMode::kResourceSingletons ||
      probe.mode == ProposalMode::kResourcePairs) {
    if (request.nodes.empty() || request.nodes.size() > 2 ||
        !request.inputs.empty() || !request.outputs.empty() ||
        request.effect_accesses.empty())
      return Status::InvalidArgument("invalid resource region request");
    EffectPlan candidate{
        .probe = &probe, .count = request.nodes.size(), .region = true};
    for (std::size_t index = 0; index < request.nodes.size(); ++index) {
      auto op = request.graph->operation(request.nodes[index]);
      if (!op)
        return std::move(op).error();
      auto touch = op->custom_as<ResourceTouch>();
      if (!touch)
        return Status::InvalidArgument(
            "resource region contains another operation");
      candidate.touches[index] = *touch;
    }
    void *storage = allocator.allocate(sizeof(EffectPlan), alignof(EffectPlan));
    if (storage == nullptr)
      return Status::ResourceExhausted("effect region allocation failed");
    auto *plan = ::new (storage) EffectPlan(candidate);
    *output = {.vtable = &kResourceRegionVTable,
               .state = plan,
               .properties = probe.artifact_properties};
    ++probe.compile_count;
    return Status::Ok();
  }
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
  if (!request.constants.empty())
    probe.compiled_constant = request.constants.front().tensor;
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
  probe.compiled_workload = request.workload;
  if (probe.required_workload != nullptr) {
    EXPECT_EQ(request.workload, *probe.required_workload);
  }
  if (probe.observe_cleanup != nullptr)
    probe.observed_cleanup_at_compile =
        probe.observe_cleanup->destroy_prepared_count;
  if (probe.decline_regions)
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("injected pre-acceptance region decline"));
  if (probe.packing != nullptr)
    return compile_packed_region(probe, request, allocator, output);
  Status status = compile_region_checked(context, request, allocator, output);
  if (!status.ok())
    return ProviderPrepareOutcome::error(std::move(status));
  if (probe.declined_workspace_resource.valid() &&
      request.workspace_resource == probe.declined_workspace_resource) {
    const auto failure = Status::ResourceExhausted("workspace-local rejection");
    return probe.workspace_decline_is_error
               ? ProviderPrepareOutcome::error(failure)
               : ProviderPrepareOutcome::resource_declined(failure);
  }
  if (probe.region_outcome_route &&
      *probe.region_outcome_route != request.route)
    return ProviderPrepareOutcome::prepared();
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

void destroy_effect_prepared(void *opaque, AllocatorRef allocator) noexcept {
  auto *plan = static_cast<EffectPlan *>(opaque);
  if (plan->region)
    ++plan->probe->destroy_prepared_count;
  plan->~EffectPlan();
  allocator.deallocate(plan, sizeof(EffectPlan), alignof(EffectPlan));
}

AliasSupport effect_alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

Status create_effect_invocation(void *prepared, AllocatorRef,
                                hyle::mem::BufferView,
                                void **invocation) noexcept {
  if (invocation == nullptr)
    return Status::InvalidArgument("effect invocation output is null");
  auto &probe = *static_cast<EffectPlan *>(prepared)->probe;
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
  auto &plan = *static_cast<EffectPlan *>(invocation);
  ++plan.probe->effect_submit_count;
  for (std::size_t index = 0; index < plan.count; ++index) {
    const auto &touch = plan.touches[index];
    switch (touch.access) {
    case hyle::compute::ResourceAccess::kRead:
      plan.probe->last_resource_read = *touch.state;
      break;
    case hyle::compute::ResourceAccess::kWrite:
      *touch.state = touch.value;
      break;
    case hyle::compute::ResourceAccess::kReadWrite:
      *touch.state += touch.value;
      break;
    }
  }
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

const hyle::graph::ProviderRegionPreparedVTable kResourceRegionVTable{
    .destroy_prepared = &destroy_effect_prepared,
    .alias_support = &effect_alias_support,
    .create_invocation = &create_effect_invocation,
    .destroy_invocation = &destroy_effect_invocation,
    .shared_submit = &submit_effect,
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
  ++probe.single_proposal_count;
  const bool effect = operation.custom_as<ExternalEffect>().has_value() ||
                      operation.custom_as<ResourceTouch>().has_value();
  const bool constrained_unary =
      operation.kind() == hyle::compute::OperationKind::kUnary &&
      probe.required_resource.valid();
  if (!effect && !constrained_unary)
    return Status::Unimplemented("region provider has no single-op route");
  if (proposal == nullptr ||
      (effect &&
       (!inputs.empty() || !outputs.empty() || !input_requirements.empty() ||
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
  probe.destroy_count_at_single_preparation = probe.destroy_prepared_count;
  const auto touch = request.operation.custom_as<ResourceTouch>();
  const bool effect =
      request.operation.custom_as<ExternalEffect>().has_value() ||
      touch.has_value();
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
  void *storage = allocator.allocate(sizeof(EffectPlan), alignof(EffectPlan));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("effect plan allocation failed"));
  auto *effect_plan = ::new (storage) EffectPlan{.probe = &probe};
  if (touch) {
    effect_plan->touches[0] = *touch;
    effect_plan->count = 1;
  }
  *prepared = hyle::compute::detail::ProviderPrepared{
      .vtable = &kEffectPreparedVTable,
      .state = effect_plan,
      .runtime = single_runtime(probe, true),
  };
  return probe.single_outcome_override.value_or(
      ProviderPrepareOutcome::prepared());
}

Status drain_provider_context(void *) noexcept { return Status::Ok(); }

const void *query_extension(void *context,
                            hyle::compute::ProviderExtensionId id) noexcept {
  if (context != nullptr)
    ++static_cast<RegionProbe *>(context)->extension_query_count;
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
  std::uint64_t identity = 0;
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
             .identity ==
         hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(
             right)
             .identity;
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

Status validate_async_resource(const hyle::compute::Device &device,
                               hyle::mem::MemoryResourceRef resource) noexcept {
  if (!resource.valid())
    return Status::InvalidArgument("test graph resource is invalid");
  const auto *probe =
      hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(device)
          .probe;
  probe->resource_validation_count.fetch_add(1, std::memory_order_relaxed);
  if (probe->validation_failure_resource == resource)
    return probe->validation_failure;
  if (probe->backend_only_resource.valid() &&
      probe->backend_only_resource != resource)
    return Status::FailedPrecondition(
        "test backend cannot access this storage");
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
  ++probe->copy_count;
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
  return source_queue == nullptr ? hyle::compute::detail::CompletionEndpoint{}
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
      &kAsyncBackendVTable, AsyncBackendPayload{&probe, probe.device_identity});
  return backend.device(0).value();
}

struct NegationGraph final {
  Graph graph{};
  ValueId input{};
  ValueId output{};
};

NegationGraph make_negations(std::size_t count, bool finish_with_abs = false,
                            UnaryKind tail = UnaryKind::kAbs) {
  GraphBuilder builder = GraphBuilder::create().value();
  NegationGraph result;
  result.input = builder.input(row(8)).value();
  ValueId value = result.input;
  for (std::size_t index = 0; index < count; ++index)
    value = builder.add(Unary{UnaryKind::kNeg}, value).value();
  if (finish_with_abs)
    value = builder.add(Unary{tail}, value).value();
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
  const bool asynchronous = probe.asynchronous_backend ||
                            probe.mode == ProposalMode::kAsyncPairs ||
                            probe.mode == ProposalMode::kAsyncRejectSecond;
  if (asynchronous) {
    const std::array providers{
        hyle::compute::ProviderBinding{region_provider(probe), 100},
        hyle::compute::ProviderBinding{hyle::compute::native::provider(), 200}};
    return Context::create(async_device(probe), {.providers = providers})
        .value();
  }
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(probe), 100},
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 200},
      hyle::compute::ProviderBinding{hyle::compute::native::provider(), 300}};
  return Context::create(hyle::compute::cpu::device(),
                         {.providers = providers})
      .value();
}

Context make_context(RegionProbe &first, RegionProbe &second) {
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(first), 100},
      hyle::compute::ProviderBinding{region_provider(second), 200},
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 300},
      hyle::compute::ProviderBinding{hyle::compute::native::provider(), 400}};
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

void expect_negation_replay(const hyle::graph::Executable &executable,
                            const NegationGraph &logical,
                            const Context &context) {
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  write(input, std::array{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F});
  auto run = executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  auto completion =
      executable.required_queue_count() == 0 ? run.submit() : run.submit(queue);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(read(output, 8),
            (std::vector<float>{-1.F, 2.F, -3.F, 4.F, -5.F, 6.F, -7.F, 8.F}));
}

TEST(GraphRules, ModelQualificationIsColdExactAndKeepsTheGenericFallback) {
  using namespace hyle::compute;
  const auto required = workload_test::known(WorkloadPhase::kDecode);
  std::array<WorkloadContext, 9> requests;
  requests.fill(required);
  requests[1] = {};
  auto changed = workload_test::kModel;
  changed.model_publisher = "unknown";
  changed.model_repository = "unknown/unknown-model";
  changed.model_family = "unknown";
  changed.model_name = "unknown-model";
  requests[2].profile = ModelArtifactProfile::create(changed).value();
  changed = workload_test::kModel;
  changed.model_family = "other-family";
  requests[3].profile = ModelArtifactProfile::create(changed).value();
  changed = workload_test::kModel;
  changed.model_name = "other-model";
  requests[4].profile = ModelArtifactProfile::create(changed).value();
  changed = workload_test::kModel;
  changed.model_revision = "other-revision";
  requests[5].profile = ModelArtifactProfile::create(changed).value();
  changed = workload_test::kModel;
  changed.configuration_digest =
      "1000000000000000000000000000000000000000000000000000000000000000";
  requests[6].profile = ModelArtifactProfile::create(changed).value();
  requests[7].phase = WorkloadPhase::kPrefill;
  ++requests[8].region.version;
  const auto logical = make_negations(1);
  for (std::size_t index = 0; index < requests.size(); ++index) {
    SCOPED_TRACE(index);
    RegionProbe specialized{.mode = ProposalMode::kSingleton},
        generic{.mode = ProposalMode::kSingleton};
    specialized.required_workload = &required;
    auto context = make_context(specialized, generic);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.operation_fallback = OperationFallbackPolicy::kReject,
         .workload = requests[index]});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->executable.workload(), requests[index]);
    EXPECT_EQ(specialized.resource_workload, requests[index]);
    auto &selected = index == 0 ? specialized : generic;
    EXPECT_EQ(selected.compiled_workload, requests[index]);
    EXPECT_EQ(specialized.compile_count, index == 0 ? 1U : 0U);
    EXPECT_EQ(generic.compile_count, index == 0 ? 0U : 1U);
    const auto discoveries = specialized.propose_count + generic.propose_count;
    const auto qualifications =
        specialized.region_resource_count + generic.region_resource_count;
    for (int repeat = 0; repeat < 3; ++repeat)
      expect_negation_replay(compiled->executable, logical, context);
    EXPECT_EQ(specialized.propose_count + generic.propose_count, discoveries);
    EXPECT_EQ(specialized.region_resource_count + generic.region_resource_count,
              qualifications);
  }
}

TEST(GraphRules, FirstCompletePlanBindsWithoutPreparingAlternatives) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(0)->provider_instance,
            context.provider_instance_id(0).value());
  EXPECT_EQ(compiled->report.search().feasible_plans, 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
  EXPECT_EQ(first.submit_count + second.submit_count, 0u);
  const auto discoveries = first.propose_count + second.propose_count;
  for (int i = 0; i != 3; ++i)
    expect_negation_replay(compiled->executable, logical, context);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
  EXPECT_EQ(first.propose_count + second.propose_count, discoveries);
}

TEST(GraphRouteBoundary, SameDeviceContextsAndHostStorageRemainValid) {
  const auto logical = make_negations(1);
  for (bool different_device : {false, true}) {
    RegionProbe first{.mode = ProposalMode::kSingleton},
        second{.mode = ProposalMode::kSingleton};
    auto first_context = make_context(first);
    second.asynchronous_backend = different_device;
    auto second_context = make_context(second);
    EXPECT_EQ(first_context.device() == second_context.device(),
              !different_device);
    HostResource alternate;
    const Environment environment{
        {&first_context, hyle::mem::system_resource()},
        {&second_context, hyle::mem::MemoryResourceRef(alternate)}};
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.operation_fallback = OperationFallbackPolicy::kReject});
    if (different_device) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsInvalidArgument());
      EXPECT_EQ(first.propose_count + second.propose_count, 0u);
      EXPECT_EQ(first.compile_count + second.compile_count, 0u);
    } else {
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->report.target_count(), 2u);
      EXPECT_EQ(compiled->report.storage_domain_count(), 2u);
      expect_negation_replay(compiled->executable, logical, first_context);
    }
    EXPECT_EQ(alternate.live_allocations, 0u);
  }
}

TEST(GraphRouteBoundary, DifferentDevicesOfOneBackendAreRejected) {
  RegionProbe first{.device_identity = 1, .mode = ProposalMode::kAsyncPairs},
      second{.device_identity = 2, .mode = ProposalMode::kAsyncPairs};
  auto first_context = make_context(first);
  auto second_context = make_context(second);
  ASSERT_EQ(first_context.device().backend_id(),
            second_context.device().backend_id());
  ASSERT_NE(first_context.device(), second_context.device());
  const auto logical = make_negations(1);
  const Environment environment{
      {&first_context, hyle::mem::system_resource()},
      {&second_context, hyle::mem::system_resource()}};
  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment());
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsInvalidArgument());
  EXPECT_EQ(first.propose_count + second.propose_count, 0u);
}

TEST(GraphRules, FirstFeasiblePlanDoesNotSpendBudgetOnUnusedAlternatives) {
  const auto logical = make_negations(1);
  for (bool strict : {false, true}) {
    RegionProbe first{.mode = ProposalMode::kSingleton},
        second{.mode = ProposalMode::kSingleton};
    auto context = make_context(first, second);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.max_placement_attempts = 1,
         .max_route_search_states = 1,
         .operation_fallback = OperationFallbackPolicy::kReject,
         .search_policy = strict ? CompileSearchPolicy::kRequireUnexhausted
                                 : CompileSearchPolicy::kAllowLimited});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_FALSE(compiled->report.search().optimization_limited());
    EXPECT_EQ(compiled->report.search().feasible_plans, 1u);
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(second.compile_count, 0u);
    expect_negation_replay(compiled->executable, logical, context);
  }
}

TEST(GraphRules, ColdDeclineRetriesButRealErrorStopsPreparation) {
  const auto logical = make_negations(1);
  for (bool real_error : {false, true}) {
    RegionProbe first{.mode = ProposalMode::kSingleton},
        second{.mode = ProposalMode::kSingleton};
    const auto failure =
        Status::ResourceExhausted("provider preparation failed");
    first.region_outcome_override =
        real_error ? ProviderPrepareOutcome::error(failure)
                   : ProviderPrepareOutcome::resource_declined(failure);
    auto context = make_context(first, second);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.operation_fallback = OperationFallbackPolicy::kReject});
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    if (real_error) {
      ASSERT_FALSE(compiled);
      EXPECT_EQ(compiled.error(), failure);
      EXPECT_EQ(second.compile_count, 0u);
    } else {
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
      EXPECT_EQ(second.compile_count, 1u);
      expect_negation_replay(compiled->executable, logical, context);
    }
  }
}

TEST(GraphRules, RequiredProviderKeepsItsOwnLaneAndWorkspaceMetadata) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  use_delegated_runtime(first);
  first.artifact_properties.workspace = {.size = 128, .alignment = 64};
  auto context = make_context(first, second);
  for (std::size_t provider : {0u, 1u}) {
    const std::array required{hyle::graph::NodeProviderConstraint{
        logical.graph.node_id().value(), kRegionProviderId,
        context.provider_instance_id(provider).value()}};
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.node_providers = required,
         .operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.workspace_bytes(), provider == 1 ? 0u : 128u);
    EXPECT_EQ(compiled->report.lane_count(), provider == 1 ? 1u : 2u);
    EXPECT_EQ(compiled->report.delegated_segment_count(),
              provider == 1 ? 0u : 1u);
    expect_negation_replay(compiled->executable, logical, context);
  }
}

TEST(GraphResources, SameRegionCanUseAnotherWorkspaceWithoutChangingProvider) {
  const auto logical = make_negations(1);
  HostResource other;
  const auto host = hyle::mem::system_resource();
  const hyle::mem::MemoryResourceRef alternate(other);
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  probe.region_input_resource = probe.region_output_resource = host;
  probe.artifact_properties.workspace = {.size = 128, .alignment = 64};
  auto context = make_context(probe);
  const std::array domains{
      hyle::graph::StorageDomain{.resource = host, .workspace_budget_bytes = 0},
      hyle::graph::StorageDomain{.resource = alternate,
                                 .workspace_budget_bytes = 128}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  const hyle::graph::CompileEnvironment environment{.storage_domains = domains,
                                                    .targets = targets};
  auto result = hyle::graph::compile(
      logical.graph, environment,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(result->report.region_count(), 1u);
  EXPECT_EQ(result->report.step(0)->workspace_storage,
            hyle::graph::StorageDomainId{1});
  EXPECT_EQ(result->report.search().route_plan_rejections, 1u);
  EXPECT_EQ(result->report.search().region_route_rejections, 0u);
  EXPECT_EQ(probe.region_resource_count, 1u);
  EXPECT_EQ(probe.compile_count, 2u);
  expect_negation_replay(result->executable, logical, context);
  EXPECT_EQ(other.live_allocations, 0u);

  // An exact Provider resource contract remains hard, even with another
  // otherwise compatible resource in the environment.
  probe.region_workspace_resource = host;
  auto required = hyle::graph::compile(
      logical.graph, environment,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(required);
  EXPECT_TRUE(required.error().IsResourceExhausted());
}

TEST(GraphResources, ResourceLocalDeclineDoesNotPoisonOtherBindings) {
  const auto logical = make_negations(1);
  HostResource other;
  const auto host = hyle::mem::system_resource();
  for (bool real_error : {false, true}) {
    RegionProbe probe{.mode = ProposalMode::kSingleton};
    probe.region_input_resource = probe.region_output_resource = host;
    probe.declined_workspace_resource = host;
    probe.workspace_decline_is_error = real_error;
    probe.artifact_properties.workspace = {.size = 128, .alignment = 64};
    auto context = make_context(probe);
    const Environment environment{
        {&context, host}, {&context, hyle::mem::MemoryResourceRef(other)}};
    auto result = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.operation_fallback = OperationFallbackPolicy::kReject});
    if (real_error) {
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsResourceExhausted());
      EXPECT_EQ(probe.compile_count, 1u);
      EXPECT_EQ(probe.destroy_prepared_count, 1u);
    } else {
      ASSERT_TRUE(result) << result.error().message();
      EXPECT_EQ(result->report.search().region_route_rejections, 1u);
      EXPECT_EQ(probe.compile_count, 2u);
      EXPECT_EQ(probe.destroy_prepared_count, 1u);
      EXPECT_EQ(result->report.step(0)->workspace_storage,
                environment.storage(1));
      expect_negation_replay(result->executable, logical, context);
    }
  }
  EXPECT_EQ(other.live_allocations, 0u);
}

TEST(GraphResources, WorkspaceBudgetSelectsFeasibleStorageAndLane) {
  const auto logical = make_negations(1);
  HostResource other;
  const auto host = hyle::mem::system_resource();
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  probe.region_input_resource = probe.region_output_resource = host;
  probe.artifact_properties.workspace = {.size = 128, .alignment = 64};
  auto context = make_context(probe);
  std::array domains{hyle::graph::StorageDomain{.resource = host},
                     hyle::graph::StorageDomain{
                         .resource = hyle::mem::MemoryResourceRef(other)}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  const std::array lanes{
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0}},
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0}}};
  const hyle::graph::CompileEnvironment environment{
      .storage_domains = domains, .targets = targets, .lanes = lanes};
  for (bool alternate : {false, true}) {
    domains[0].workspace_budget_bytes = alternate ? 0 : SIZE_MAX;
    auto result = hyle::graph::compile(
        logical.graph, environment,
        {.operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(result->report.step(0)->workspace_storage,
              hyle::graph::StorageDomainId{alternate ? 1u : 0u});
    EXPECT_EQ(result->report.step(0)->lane, hyle::graph::ExecutionLaneId{0});
    EXPECT_EQ(result->report.search().feasible_plans, 1u);
    EXPECT_FALSE(result->report.search().optimization_limited());
    expect_negation_replay(result->executable, logical, context);
  }
  EXPECT_EQ(other.live_allocations, 0u);
}

TEST(GraphResources, LaneSearchSharesStateAndPrepareLimitsWithNodeCover) {
  const auto logical = make_negations(1);
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  probe.implementation = hyle::compute::ProviderImplementationId{41};
  auto context = make_context(probe);
  const Environment base{{&context, hyle::mem::system_resource()}};
  const std::array lanes{
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0}},
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0}}};
  auto environment = base.environment();
  environment.lanes = lanes;
  for (bool state_limit : {false, true}) {
    for (bool strict : {false, true}) {
      auto result = hyle::graph::compile(
          logical.graph, environment,
          {.max_placement_attempts = state_limit ? 4096u : 1u,
           .max_route_search_states = state_limit ? 2u : 65536u,
           .operation_fallback = OperationFallbackPolicy::kReject,
           .search_policy = strict ? CompileSearchPolicy::kRequireUnexhausted
                                   : CompileSearchPolicy::kAllowLimited});
      ASSERT_TRUE(result) << result.error().message();
      EXPECT_FALSE(result->report.search().optimization_limited());
      EXPECT_EQ(result->report.search().feasible_plans, 1u);
      expect_negation_replay(result->executable, logical, context);
    }
  }
  auto incomplete = hyle::graph::compile(
      logical.graph, environment,
      {.max_route_search_states = 1,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(incomplete);
  EXPECT_TRUE(incomplete.error().IsResourceExhausted());
}

TEST(GraphResources,
     MatureIntermediateMovesToFeasibleStorageAndPreservesAliasRules) {
  const auto logical = make_negations(2);
  HostResource other;
  const auto host = hyle::mem::system_resource();
  const std::array providers{
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 100}};
  auto context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  const std::array domains{
      hyle::graph::StorageDomain{.resource = host, .workspace_budget_bytes = 0},
      hyle::graph::StorageDomain{.resource =
                                     hyle::mem::MemoryResourceRef(other),
                                 .workspace_budget_bytes = 64}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  const hyle::graph::CompileEnvironment environment{.storage_domains = domains,
                                                    .targets = targets};
  const std::array external{
      hyle::graph::ExternalResourceContract{logical.input,
                                            hyle::graph::StorageDomainId{0}},
      hyle::graph::ExternalResourceContract{logical.output,
                                            hyle::graph::StorageDomainId{0}}};
  auto result = hyle::graph::compile(logical.graph, environment,
                                     {.external_resources = external});
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_GT(result->report.search().route_plan_rejections, 0u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto run = result->executable.create_invocation().value();
  EXPECT_GT(other.allocation_count, 0u);
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, 8), std::vector<float>(values.begin(), values.end()));
  const auto middle = logical.graph.result(logical.graph.node_id(0).value());
  ASSERT_TRUE(middle);
  const std::array required{
      hyle::graph::ValuePlacement{*middle, hyle::graph::StorageDomainId{0},
                                  hyle::graph::PlacementStrength::kRequired}};
  auto impossible = hyle::graph::compile(
      logical.graph, environment,
      {.value_placements = required, .external_resources = external});
  ASSERT_FALSE(impossible);
  EXPECT_TRUE(impossible.error().IsResourceExhausted());
}

TEST(GraphRules, ExplicitLeafConstraintPreservesCompleteExecution) {
  const auto logical = make_negations(1);
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  auto context = make_context(probe);
  for (bool leaf : {false, true}) {
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.region_selection =
             leaf ? RegionSelectionPolicy::kDisabled
                  : RegionSelectionPolicy::kAllowProviderRegions});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), leaf ? 0u : 1u);
    EXPECT_EQ(compiled->report.fallback_operation_count(), leaf ? 1u : 0u);
    expect_negation_replay(compiled->executable, logical, context);
  }
}

TEST(GraphRules, ConcurrencyAndRequiredProviderRemainHardConstraints) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  first.implementation = hyle::compute::ProviderImplementationId{41};
  second.implementation = hyle::compute::ProviderImplementationId{42};
  first.artifact_properties.runtime.supports_concurrent_invocations = false;
  auto context = make_context(first, second);
  const Environment environment{{&context, hyle::mem::system_resource()}};
  const std::array required{hyle::graph::NodeProviderConstraint{
      logical.graph.node_id().value(), kRegionProviderId,
      context.provider_instance_id(0).value()}};
  auto fixed = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.node_providers = required,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(fixed) << fixed.error().message();
  EXPECT_EQ(fixed->report.step(0)->provider_instance, required[0].instance);
  auto parallel = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.operation_fallback = OperationFallbackPolicy::kReject,
       .require_concurrent_invocations = true});
  ASSERT_TRUE(parallel) << parallel.error().message();
  EXPECT_EQ(parallel->report.step(0)->provider_instance,
            context.provider_instance_id(1).value());
  EXPECT_EQ(parallel->report.search().route_plan_rejections, 1u);
  auto impossible = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.node_providers = required,
       .operation_fallback = OperationFallbackPolicy::kReject,
       .require_concurrent_invocations = true});
  ASSERT_FALSE(impossible);
  EXPECT_TRUE(impossible.error().IsFailedPrecondition());
}

TEST(GraphRules, RejectedAndAcceptedPlansShareOneMaterializedConstant) {
  HostResource resource;
  const hyle::mem::MemoryResourceRef owned(resource);
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  first.implementation = hyle::compute::ProviderImplementationId{41};
  second.implementation = hyle::compute::ProviderImplementationId{42};
  first.constant_live_allocations = &resource.live_allocations;
  second.constant_live_allocations = &resource.live_allocations;
  auto context = make_context(first, second);
  const Environment environment{{&context, owned}};
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(8)).value();
  const auto output_id = builder.add(Unary{UnaryKind::kNeg}, constant).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(8));
  write(source, std::array{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F});
  const std::array constants{
      hyle::graph::ConstantBinding{constant, source.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  ASSERT_EQ(resource.live_allocations, 0u);
  resource.allocation_count = 0;
  first.artifact_properties.runtime.supports_concurrent_invocations = false;
  auto compiled = hyle::graph::compile(
      graph, environment.environment(),
      {.constants = constants,
       .operation_fallback = OperationFallbackPolicy::kReject,
       .require_concurrent_invocations = true});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.search().route_plan_rejections, 1u);
  EXPECT_EQ(resource.allocation_count, 1u);
  EXPECT_EQ(resource.live_allocations, 1u);
  EXPECT_EQ(first.constant_allocations_at_destroy, 1u);
  ASSERT_TRUE(first.compiled_constant);
  ASSERT_TRUE(second.compiled_constant);
  EXPECT_EQ(first.compiled_constant->storage().native_handle(),
            second.compiled_constant->storage().native_handle());
  ASSERT_TRUE(source.buffer.reset().ok());
  auto output = make_tensor(row(8), owned);
  auto invocation = compiled->executable.create_invocation().value();
  compiled->executable = {};
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(read(output, 8),
            (std::vector<float>{-1.F, 2.F, -3.F, 4.F, -5.F, 6.F, -7.F, 8.F}));
  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(output.buffer.reset().ok());
  invocation = {};
  EXPECT_EQ(second.constant_allocations_at_destroy, 1u);
  EXPECT_EQ(resource.live_allocations, 0u);
  EXPECT_EQ(compiled->report.materialized_constant_bytes(), 32u);
}

TEST(GraphRules, EveryMetadataAllocationFailureIsFatalAndReclaimsPlans) {
  struct Allocations {
    AllocatorRef backing = hyle::mem::system_allocator();
    std::size_t attempts = 0, fail_at = SIZE_MAX, live = 0;
  } allocations;
  const AllocatorRef allocator{
      +[](void *opaque, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        auto &a = *static_cast<Allocations *>(opaque);
        if (a.attempts++ == a.fail_at)
          return nullptr;
        auto *result = a.backing.allocate(bytes, alignment);
        a.live += result != nullptr;
        return result;
      },
      +[](void *opaque, void *pointer, std::size_t bytes,
          std::size_t alignment) noexcept {
        auto &a = *static_cast<Allocations *>(opaque);
        a.live -= pointer != nullptr;
        a.backing.deallocate(pointer, bytes, alignment);
      },
      +[](void *opaque, void *pointer, std::size_t old_bytes,
          std::size_t new_bytes, std::size_t alignment) noexcept -> void * {
        auto &a = *static_cast<Allocations *>(opaque);
        if (new_bytes != 0 && a.attempts++ == a.fail_at)
          return nullptr;
        auto *result =
            a.backing.reallocate(pointer, old_bytes, new_bytes, alignment);
        if (pointer == nullptr && result != nullptr)
          ++a.live;
        if (pointer != nullptr && new_bytes == 0)
          --a.live;
        return result;
      },
      &allocations};
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  first.implementation = hyle::compute::ProviderImplementationId{41};
  second.implementation = hyle::compute::ProviderImplementationId{42};
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(first), 100},
      hyle::compute::ProviderBinding{region_provider(second), 200}};
  auto context =
      Context::create(hyle::compute::cpu::device(),
                      {.metadata_allocator = allocator, .providers = providers})
          .value();
  const auto logical = make_negations(1);
  const Environment base_environment{{&context, hyle::mem::system_resource()}};
  const std::array lanes{
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0}},
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0}}};
  auto environment = base_environment.environment();
  environment.lanes = lanes;
  const CompileOptions options{.operation_fallback =
                                   OperationFallbackPolicy::kReject,
                               .require_concurrent_invocations = true};
  first.artifact_properties.runtime.supports_concurrent_invocations = false;
  const auto baseline_live = allocations.live;
  allocations.attempts = 0;
  {
    auto good = hyle::graph::compile(logical.graph, environment, options);
    ASSERT_TRUE(good) << good.error().message();
    // Each of the two available lanes rejects the serial-only first artifact.
    EXPECT_EQ(good->report.search().route_plan_rejections, 2u);
  }
  const auto count = allocations.attempts;
  ASSERT_GT(count, 0u);
  EXPECT_EQ(allocations.live, baseline_live);
  for (std::size_t fail_at = 0; fail_at < count; ++fail_at) {
    SCOPED_TRACE(fail_at);
    allocations.fail_at = fail_at;
    allocations.attempts = 0;
    {
      auto failure = hyle::graph::compile(logical.graph, environment, options);
      ASSERT_FALSE(failure);
      EXPECT_TRUE(failure.error().IsResourceExhausted());
    }
    EXPECT_EQ(allocations.live, baseline_live);
    allocations.fail_at = SIZE_MAX;
  }
  auto recovered = hyle::graph::compile(logical.graph, environment, options);
  ASSERT_TRUE(recovered) << recovered.error().message();
  expect_negation_replay(recovered->executable, logical, context);
  recovered = std::unexpected(Status::Cancelled("release test artifact"));
  context = {};
  EXPECT_EQ(allocations.live, 0u);
}

TEST(GraphRegions, DiscoversResourcesForLaterCandidatesWithoutPreparingThem) {
  const auto logical = make_negations(2);
  RegionProbe first, second;
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.region_resource_count, 1u);
  EXPECT_EQ(second.region_resource_count, 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
  EXPECT_EQ(compiled->report.region_count(), 1u);
}

TEST(GraphRegions, LaterResourceQueryFailurePrecedesAnyArtifactPreparation) {
  const auto logical = make_negations(2);
  RegionProbe first, second;
  second.resource_query_failure = Status::DataLoss("invalid provider metadata");
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(compiled);
  EXPECT_EQ(compiled.error(), second.resource_query_failure);
  EXPECT_EQ(first.region_resource_count, 1u);
  EXPECT_EQ(second.region_resource_count, 1u);
  EXPECT_EQ(first.compile_count + second.compile_count, 0u);
  EXPECT_EQ(first.single_artifact_count + second.single_artifact_count, 0u);
}

TEST(GraphRegions,
     ValidatesEveryCandidateResourceRequirementBeforePreparation) {
  const auto logical = make_negations(2);
  using hyle::compute::ResourceRequirement;
  using hyle::mem::MemoryProperty;
  const std::array malformed{
      ResourceRequirement{.minimum_alignment = 3},
      ResourceRequirement{
          .required_properties = MemoryProperty::kHostAccessible,
          .forbidden_properties = MemoryProperty::kHostAccessible},
      ResourceRequirement{.requires_exact_domain = true}};
  for (std::uint32_t port = 0; port < 3; ++port) {
    SCOPED_TRACE(port);
    for (const auto &requirement : malformed) {
      RegionProbe first, second;
      second.requirement_override = requirement;
      second.requirement_port = port;
      auto context = make_context(first, second);
      auto compiled = hyle::graph::compile(logical.graph, context);
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsInternal());
      EXPECT_EQ(first.region_resource_count, 1u);
      EXPECT_EQ(second.region_resource_count, 1u);
      EXPECT_EQ(first.compile_count + second.compile_count, 0u);
      EXPECT_EQ(first.single_artifact_count + second.single_artifact_count, 0u);
    }
  }
}

TEST(GraphRegions, RequiredTargetFiltersCandidatesBeforeResourceQuery) {
  const auto logical = make_negations(2);
  RegionProbe first, second;
  second.resource_query_failure =
      Status::DataLoss("excluded target must not query");
  auto first_context = make_context(first),
       second_context = make_context(second);
  const auto host = hyle::mem::system_resource();
  const Environment environment{{&first_context, host},
                                {&second_context, host}};
  const std::array required{
      hyle::graph::NodePlacement{logical.graph.node_id(0).value(),
                                 environment.target(0),
                                 hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{logical.graph.node_id(1).value(),
                                 environment.target(0),
                                 hyle::graph::PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(logical.graph, environment.environment(),
                                       {.node_placements = required});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.region_resource_count, 1u);
  EXPECT_EQ(second.region_resource_count, 0u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
  EXPECT_EQ(compiled->report.step(0)->target, environment.target(0));
}

TEST(GraphRegions, UnsupportedResourceQueryRejectsOnlyThatCandidate) {
  const auto logical = make_negations(2);
  RegionProbe first, second;
  first.resource_query_failure =
      Status::Unimplemented("no matching storage contract");
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.region_resource_count, 1u);
  EXPECT_EQ(second.region_resource_count, 1u);
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 1u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
}

TEST(GraphRegions,
     CandidateBoundariesMatchFullScanAcrossFanoutAndRepeatedPorts) {
  for (std::uint32_t seed = 1; seed <= 32; ++seed) {
    SCOPED_TRACE(seed);
    auto builder = GraphBuilder::create().value();
    std::vector<ValueId> values{builder.input(row(8)).value(),
                                builder.input(row(8)).value()};
    std::vector<float> expected{1.F, 2.F};
    std::uint32_t random = seed;
    for (std::size_t index = 0; index < 12; ++index) {
      random = random * 1664525u + 1013904223u;
      const auto lhs = random % values.size();
      random = random * 1664525u + 1013904223u;
      const auto rhs = index % 2 == 0 ? lhs : random % values.size();
      if (index % 3 == 0) {
        values.push_back(
            builder.add(Unary{UnaryKind::kNeg}, values[lhs]).value());
        expected.push_back(-expected[lhs]);
      } else {
        values.push_back(
            builder
                .add(hyle::compute::Binary{hyle::compute::BinaryKind::kAdd},
                     values[lhs], values[rhs])
                .value());
        expected.push_back(expected[lhs] + expected[rhs]);
      }
    }
    // Published intermediate results remain outputs even when also consumed
    // inside a region. Other edges exercise outside fanout and repeated ports.
    const std::array published{std::size_t{3}, std::size_t{6},
                               seed % 2 == 0 ? std::size_t{9}
                                             : values.size() - 1};
    for (auto index : published)
      ASSERT_TRUE(builder.publish(values[index]).ok());
    const auto graph = std::move(builder).finish().value();
    RegionProbe probe{.mode = ProposalMode::kBoundaryWindows};
    probe.decline_regions = true;
    auto context = make_context(probe);
    auto compiled = hyle::graph::compile(graph, context);
    ASSERT_TRUE(compiled) << compiled.error().message();
    const auto candidates = 3 * graph.node_count() - 3;
    EXPECT_EQ(probe.region_resource_count, candidates);
    EXPECT_EQ(probe.boundary_checks, candidates);
    EXPECT_EQ(probe.propose_count, 1u);
    EXPECT_EQ(compiled->report.region_count(), 0u);

    auto input_a = make_tensor(row(8)), input_b = make_tensor(row(8));
    std::array<float, 8> input_values{};
    input_values.fill(1.F);
    write(input_a, input_values);
    input_values.fill(2.F);
    write(input_b, input_values);
    auto outputs = std::array{make_tensor(row(8)), make_tensor(row(8)),
                              make_tensor(row(8))};
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(values[0], input_a.view).ok());
    ASSERT_TRUE(invocation.bind(values[1], input_b.view).ok());
    for (std::size_t index = 0; index < published.size(); ++index)
      ASSERT_TRUE(
          invocation.bind(values[published[index]], outputs[index].view).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
    for (std::size_t index = 0; index < published.size(); ++index)
      EXPECT_EQ(read(outputs[index], 8),
                std::vector<float>(8, expected[published[index]]));
  }
}

TEST(GraphRegions, GlobalProviderFiltersSkipDiscoveryAndLeafCallbacks) {
  const auto logical = make_negations(2);
  for (int mode = 0; mode < 3; ++mode) {
    RegionProbe probe;
    auto context = make_context(probe);
    const auto queries = probe.extension_query_count;
    const std::array native{context.provider_identity(1)->id};
    const std::array region{kRegionProviderId};
    CompileOptions options;
    if (mode == 0)
      options.allowed_providers = native;
    else
      options.denied_providers = region;
    if (mode == 2)
      options.allowed_providers = region; // deny overrides allow
    auto compiled = hyle::graph::compile(logical.graph, context, options);
    EXPECT_EQ(probe.extension_query_count, queries);
    EXPECT_EQ(probe.propose_count, 0u);
    EXPECT_EQ(probe.single_proposal_count, 0u);
    EXPECT_EQ(probe.compile_count, 0u);
    if (mode == 2) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsUnimplemented());
    } else {
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->report.region_count(), 0u);
      EXPECT_EQ(compiled->report.step(0)->provider.id, native[0]);
    }
  }
}

TEST(GraphRegions, LeafProposalLimitCanRetainACompleteRegionPlan) {
  const auto logical = make_negations(2);
  for (auto policy : {CompileSearchPolicy::kAllowLimited,
                      CompileSearchPolicy::kRequireUnexhausted}) {
    RegionProbe probe;
    auto context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context,
                                         {.max_placement_attempts = 1,
                                          .max_operation_proposals = 1,
                                          .search_policy = policy});
    if (policy == CompileSearchPolicy::kRequireUnexhausted) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsResourceExhausted());
      EXPECT_EQ(probe.compile_count, 0u);
      EXPECT_EQ(probe.region_resource_count, 0u);
      continue;
    }
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), 1u);
    EXPECT_EQ(compiled->report.search().operation_proposals, 1u);
    EXPECT_EQ(compiled->report.search().placement_attempts, 1u);
    EXPECT_TRUE(compiled->report.search().operation_proposal_limit_reached);
    EXPECT_EQ(probe.single_artifact_count, 0u);
  }
  RegionProbe probe;
  auto context = make_context(probe);
  auto regions_only = hyle::graph::compile(
      logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject,
       .max_operation_proposals = 0});
  ASSERT_TRUE(regions_only);
  EXPECT_EQ(probe.single_proposal_count, 0u);
  EXPECT_EQ(regions_only->report.search().operation_proposals, 0u);
  EXPECT_FALSE(regions_only->report.search().operation_proposal_limit_reached);
}

TEST(GraphRegions, RebuildingAPartitionReusesTheOriginalLeafProposals) {
  const auto logical = make_negations(3);
  RegionProbe probe;
  probe.region_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("fused route declined"));
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  // The initial plan places the unpaired tail but does not prepare it before
  // the region. After the pair declines, all three leaves are prepared once.
  EXPECT_EQ(probe.single_proposal_count, 3u);
  EXPECT_EQ(compiled->report.search().operation_proposals, 3u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 4u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
  EXPECT_EQ(probe.propose_count, 1u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue).value().wait().ok());
  auto expected = std::vector<float>(values.begin(), values.end());
  for (float &value : expected)
    value = -value;
  EXPECT_EQ(read(output, values.size()), expected);
  EXPECT_EQ(probe.single_proposal_count, 3u);
}

TEST(GraphRegions, ImpossibleLaterLeafStorageDoesNotPrepareEarlierLeaves) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  probe.required_resource = hyle::mem::system_resource();
  auto context = make_context(probe);
  HostResource other;
  const std::array storage{
      hyle::graph::StorageDomain{.resource = hyle::mem::system_resource()},
      hyle::graph::StorageDomain{.resource =
                                     hyle::mem::MemoryResourceRef(other)}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  const auto instance = context.provider_instance_id(0).value();
  const std::array providers{
      hyle::graph::NodeProviderConstraint{logical.graph.node_id(0).value(),
                                          kRegionProviderId, instance},
      hyle::graph::NodeProviderConstraint{logical.graph.node_id(1).value(),
                                          kRegionProviderId, instance}};
  const std::array placements{hyle::graph::ValuePlacement{
      .value = logical.output,
      .storage = hyle::graph::StorageDomainId{1},
      .strength = hyle::graph::PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(
      logical.graph, {.storage_domains = storage, .targets = targets},
      {.value_placements = placements,
       .node_providers = providers,
       .region_selection = RegionSelectionPolicy::kDisabled});
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsUnimplemented());
  EXPECT_EQ(probe.single_proposal_count, 2u);
  EXPECT_EQ(probe.single_artifact_count, 0u);
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, RegionMustMatchEveryCoveredNodeProvider) {
  const auto logical = make_negations(4);
  RegionProbe probe;
  auto context = make_context(probe);
  const auto native = context.provider_identity(1)->id;
  const std::array constraints{
      hyle::graph::NodeProviderConstraint{logical.graph.node_id(1).value(),
                                          native},
      hyle::graph::NodeProviderConstraint{logical.graph.node_id(3).value(),
                                          kRegionProviderId}};
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       {.node_providers = constraints});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(probe.region_resource_count, 1u);
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.compiled_begin, 2u);
  EXPECT_EQ(probe.compiled_count, 2u);
  EXPECT_EQ(compiled->report.step(1)->provider.id, native);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue).value().wait().ok());
  EXPECT_EQ(read(output, 8), std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, ConflictingInstancesPreventFusionWithoutRelaxingNodes) {
  const auto logical = make_negations(2);
  for (auto mode : {ProposalMode::kPairs, ProposalMode::kSingleton}) {
    RegionProbe first{.mode = mode}, second{.mode = mode};
    second.singleton_node = 1;
    auto context = make_context(first, second);
    const std::array constraints{
        hyle::graph::NodeProviderConstraint{
            logical.graph.node_id(0).value(), kRegionProviderId,
            context.provider_instance_id(0).value()},
        hyle::graph::NodeProviderConstraint{
            logical.graph.node_id(1).value(), kRegionProviderId,
            context.provider_instance_id(1).value()}};
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.node_providers = constraints,
         .operation_fallback = OperationFallbackPolicy::kReject});
    if (mode == ProposalMode::kPairs) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsUnimplemented());
      EXPECT_EQ(first.region_resource_count + second.region_resource_count, 0u);
      EXPECT_EQ(first.compile_count + second.compile_count, 0u);
    } else {
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->report.region_count(), 2u);
      EXPECT_EQ(compiled->report.step(0)->provider_instance,
                constraints[0].instance);
      EXPECT_EQ(compiled->report.step(1)->provider_instance,
                constraints[1].instance);
    }
    EXPECT_EQ(first.single_proposal_count + second.single_proposal_count, 0u);
  }
}

TEST(GraphRegions, RequiredInstanceSelectsTargetAndCannotOverridePlacement) {
  const auto logical = make_negations(2);
  RegionProbe first, second;
  auto a = make_context(first), b = make_context(second);
  const std::array storage{
      hyle::graph::StorageDomain{hyle::mem::system_resource()}};
  const std::array targets{
      hyle::graph::ExecutionTarget{&a, hyle::graph::StorageDomainId{0}},
      hyle::graph::ExecutionTarget{&b, hyle::graph::StorageDomainId{0}}};
  const std::array constraints{hyle::graph::NodeProviderConstraint{
      logical.graph.node_id(0).value(), kRegionProviderId,
      b.provider_instance_id(0).value()}};
  auto compiled = hyle::graph::compile(
      logical.graph, {.storage_domains = storage, .targets = targets},
      {.node_providers = constraints});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.step(0)->provider_instance,
            constraints[0].instance);
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 1u);
  const std::array placements{hyle::graph::NodePlacement{
      constraints[0].node, hyle::graph::ExecutionTargetId{0},
      hyle::graph::PlacementStrength::kRequired}};
  auto conflict = hyle::graph::compile(
      logical.graph, {.storage_domains = storage, .targets = targets},
      {.node_placements = placements, .node_providers = constraints});
  ASSERT_FALSE(conflict);
  EXPECT_TRUE(conflict.error().IsUnimplemented());
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 1u);
}

TEST(GraphRegions, RegionRetryAndLeafFallbackCannotEscapeRequiredInstance) {
  const auto logical = make_negations(2);
  for (auto fallback :
       {OperationFallbackPolicy::kAllow, OperationFallbackPolicy::kReject}) {
    RegionProbe first, second;
    first.region_outcome_override = ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("required route lacks resources"));
    auto context = make_context(first, second);
    const std::array constraints{hyle::graph::NodeProviderConstraint{
        logical.graph.node_id(0).value(), kRegionProviderId,
        context.provider_instance_id(0).value()}};
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.node_providers = constraints, .operation_fallback = fallback});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsResourceExhausted());
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    EXPECT_EQ(second.compile_count, 0u);
    EXPECT_EQ(second.region_resource_count, 0u);
    EXPECT_EQ(second.single_proposal_count,
              fallback == OperationFallbackPolicy::kAllow ? 1u : 0u);
    EXPECT_EQ(second.single_artifact_count, 0u);
  }
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

TEST(GraphRegions, NamedResourceExecutesAndInvocationRetainsItsPreparedOwner) {
  ResourceStorage storage{7};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  const auto write_node = add_resource_touch(builder, resource, storage,
                                             ResourceAccess::kWrite, 41);
  const auto read_node =
      add_resource_touch(builder, resource, storage, ResourceAccess::kRead);
  ASSERT_TRUE(builder.depends_on(read_node, write_node).ok());
  Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  hyle::graph::Invocation invocation;
  {
    const std::array bindings{
        EffectResourceBinding{resource, storage.footprint()}};
    auto compiled =
        hyle::graph::compile(graph, context, {.effect_bindings = bindings})
            .value();
    EXPECT_EQ(compiled.report.region_count(), 0u);
    invocation = compiled.executable.create_invocation().value();
  }
  graph = {};
  Queue queue = context.create_queue().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    *storage.value() = -1;
    ASSERT_TRUE(invocation.submit(queue)->wait().ok());
    EXPECT_EQ(*storage.value(), 41);
    EXPECT_EQ(probe.last_resource_read, 41);
  }
  EXPECT_EQ(probe.effect_create_count, 2u);
  EXPECT_EQ(probe.effect_submit_count, 4u);
}

TEST(GraphRegions, ValidatesBackingBeforePreparingAnyResourceOperation) {
  ResourceStorage storage;
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  const auto unused = builder.effect_resource().value();
  add_resource_touch(builder, resource, storage, ResourceAccess::kReadWrite);
  const Graph graph = std::move(builder).finish().value();
  GraphBuilder foreign = GraphBuilder::create().value();
  const auto other = foreign.effect_resource().value();
  const EffectResourceBinding valid{resource, storage.footprint()};
  const std::array<std::vector<EffectResourceBinding>, 7> invalid{{
      {},
      {{resource, {}}},
      {{other, storage.footprint()}},
      {{EffectResourceId{}, storage.footprint()}},
      {valid, valid},
      {{unused, storage.footprint()}},
      {{resource, EffectResourceFootprint::object(nullptr)}},
  }};
  RegionProbe probe;
  Context context = make_context(probe);
  for (const auto &bindings : invalid) {
    auto result =
        hyle::graph::compile(graph, context, {.effect_bindings = bindings});
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsInvalidArgument()) << result.error().message();
    EXPECT_EQ(probe.single_artifact_count, 0u);
  }
  // Unused identities need no backing. Extra valid declarations are harmless
  // and do not constrain the graph's live access set.
  const std::array bindings{valid,
                            EffectResourceBinding{unused, storage.footprint()}};
  ASSERT_TRUE(
      hyle::graph::compile(graph, context, {.effect_bindings = bindings})
          .has_value());
}

TEST(GraphRegions, DifferentResourceIdsMustNotHideTheSameMemoryOrObject) {
  ResourceStorage storage;
  GraphBuilder builder = GraphBuilder::create().value();
  const auto first = builder.effect_resource().value();
  const auto second = builder.effect_resource().value();
  add_resource_touch(builder, first, storage, ResourceAccess::kRead);
  add_resource_touch(builder, second, storage, ResourceAccess::kRead);
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  // Object identities model an external runtime owner. Both invalid input
  // cases are rejected before the implementation could access any resource.
  for (auto footprint :
       {storage.footprint(), EffectResourceFootprint::object(&probe)}) {
    const std::array bindings{EffectResourceBinding{first, footprint},
                              EffectResourceBinding{second, footprint}};
    auto result =
        hyle::graph::compile(graph, context, {.effect_bindings = bindings});
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsInvalidArgument());
    EXPECT_EQ(probe.single_artifact_count, 0u);
  }
}

TEST(GraphRegions, DisjointSubrangesOfOneBackingRemainIndependent) {
  Buffer buffer = Buffer::create(hyle::mem::system_resource(), 16, 64).value();
  auto mapping = buffer.view().map_host().value();
  auto *first_value = ::new (mapping.writable_bytes()->data()) std::int32_t{0};
  auto *second_value =
      ::new (mapping.writable_bytes()->data() + 8) std::int32_t{0};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto first = builder.effect_resource().value();
  const auto second = builder.effect_resource().value();
  const auto first_node =
      builder
          .add_node(Operation::custom(
                        ResourceTouch{first_value, ResourceAccess::kWrite, 17}),
                    std::array<ValueId, 0>{})
          .value();
  const auto second_node =
      builder
          .add_node(Operation::custom(ResourceTouch{
                        second_value, ResourceAccess::kWrite, 29}),
                    std::array<ValueId, 0>{})
          .value();
  ASSERT_TRUE(builder.bind_resources(first_node, std::array{first}).ok());
  ASSERT_TRUE(builder.bind_resources(second_node, std::array{second}).ok());
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  std::array bindings{
      EffectResourceBinding{first, EffectResourceFootprint::memory(
                                       buffer.view().subview(0, 8).value())},
      EffectResourceBinding{second, EffectResourceFootprint::memory(
                                        buffer.view().subview(8, 8).value())}};
  auto compiled =
      hyle::graph::compile(graph, context, {.effect_bindings = bindings})
          .value();
  auto invocation = compiled.executable.create_invocation().value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(*first_value, 17);
  EXPECT_EQ(*second_value, 29);
  // One-byte overlap is still an alias, even though these particular test
  // writes happen not to touch that byte.
  bindings[0].footprint =
      EffectResourceFootprint::memory(buffer.view().subview(0, 9).value());
  auto overlap =
      hyle::graph::compile(graph, context, {.effect_bindings = bindings});
  ASSERT_FALSE(overlap.has_value());
  EXPECT_TRUE(overlap.error().IsInvalidArgument());
}

TEST(GraphRegions,
     ResourceRegionsPreserveOrderingWithoutSerializingIndependentDomains) {
  // Independent writes, ordered write/read, and unordered same-domain reads.
  for (int scenario : {0, 1, 2}) {
    SCOPED_TRACE(scenario);
    const bool same_resource = scenario != 0;
    const bool dependent = scenario == 1;
    ResourceStorage first_storage, second_storage;
    GraphBuilder builder = GraphBuilder::create().value();
    const auto first = builder.effect_resource().value();
    const auto second =
        same_resource ? first : builder.effect_resource().value();
    const auto first_node = add_resource_touch(
        builder, first, first_storage,
        scenario == 2 ? ResourceAccess::kRead : ResourceAccess::kWrite, 37);
    const auto second_node = add_resource_touch(
        builder, second, same_resource ? first_storage : second_storage,
        same_resource ? ResourceAccess::kRead : ResourceAccess::kWrite, 53);
    if (dependent) {
      ASSERT_TRUE(builder.depends_on(second_node, first_node).ok());
    }
    const Graph graph = std::move(builder).finish().value();
    RegionProbe first_probe{.mode = ProposalMode::kResourceSingletons};
    RegionProbe second_probe{.mode = ProposalMode::kResourceSingletons};
    for (auto *probe : {&first_probe, &second_probe}) {
      probe->artifact_properties.preserves_effects = true;
      probe->artifact_properties.runtime = kEffectRuntime;
    }
    Context first_context = make_context(first_probe);
    Context second_context = make_context(second_probe);
    const Environment environment{
        {&first_context, hyle::mem::system_resource()},
        {&second_context, hyle::mem::system_resource()}};
    const std::array placements{
        hyle::graph::NodePlacement{first_node, environment.target(0),
                                   hyle::graph::PlacementStrength::kRequired},
        hyle::graph::NodePlacement{second_node, environment.target(1),
                                   hyle::graph::PlacementStrength::kRequired}};
    const std::array bindings{
        EffectResourceBinding{first, first_storage.footprint()},
        EffectResourceBinding{second, second_storage.footprint()}};
    auto result = hyle::graph::compile(
        graph, environment.environment(),
        {.effect_bindings = std::span(bindings).first(same_resource ? 1 : 2),
         .node_placements = placements,
         .operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto compiled = std::move(*result);
    EXPECT_EQ(compiled.report.region_count(), 2u);
    EXPECT_EQ(compiled.report.segment_count(), 2u);
    EXPECT_EQ(compiled.report.segment_dependency_count(), dependent ? 1u : 0u);
    EXPECT_EQ(first_probe.observed_effect_count, 1u);
    EXPECT_EQ(second_probe.observed_effect_count, 1u);
    auto invocation = compiled.executable.create_invocation().value();
    auto first_queue = first_context.create_queue().value();
    auto second_queue = second_context.create_queue().value();
    const std::array queues{
        hyle::graph::QueueBinding{environment.lane(0), &first_queue},
        hyle::graph::QueueBinding{environment.lane(1), &second_queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    ASSERT_TRUE(invocation.submit(queues)->wait().ok());
    EXPECT_EQ(*first_storage.value(), scenario == 2 ? 0 : 37);
    if (same_resource) {
      EXPECT_EQ(second_probe.last_resource_read, scenario == 2 ? 0 : 37);
    } else {
      EXPECT_EQ(*second_storage.value(), 53);
    }
  }
}

TEST(GraphRegions, RegionMustPreserveTheUnionOfItsResourceAccesses) {
  for (bool preserves : {false, true}) {
    ResourceStorage storage{5};
    GraphBuilder builder = GraphBuilder::create().value();
    const auto resource = builder.effect_resource().value();
    const auto read_node =
        add_resource_touch(builder, resource, storage, ResourceAccess::kRead);
    const auto update_node = add_resource_touch(builder, resource, storage,
                                                ResourceAccess::kWrite, 19);
    ASSERT_TRUE(builder.depends_on(update_node, read_node).ok());
    const Graph graph = std::move(builder).finish().value();
    RegionProbe probe{.mode = ProposalMode::kResourcePairs};
    probe.artifact_properties.preserves_effects = preserves;
    probe.artifact_properties.runtime = kEffectRuntime;
    Context context = make_context(probe);
    const std::array bindings{
        EffectResourceBinding{resource, storage.footprint()}};
    auto compiled =
        hyle::graph::compile(graph, context, {.effect_bindings = bindings})
            .value();
    EXPECT_EQ(probe.observed_effect_count, 1u);
    EXPECT_EQ(probe.observed_effect_access, ResourceAccess::kReadWrite);
    EXPECT_FALSE(probe.observed_unknown_effects);
    EXPECT_EQ(probe.compile_count, preserves ? 1u : 0u);
    EXPECT_EQ(compiled.report.region_count(), preserves ? 1u : 0u);
    auto invocation = compiled.executable.create_invocation().value();
    auto queue = context.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue)->wait().ok());
    EXPECT_EQ(probe.last_resource_read, 5);
    EXPECT_EQ(*storage.value(), 19);
  }
}

TEST(GraphRegions, ChangingEffectPreservationIsFatalAndReclaimsTheArtifact) {
  ResourceStorage storage;
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  add_resource_touch(builder, resource, storage, ResourceAccess::kReadWrite);
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe{.mode = ProposalMode::kResourceSingletons};
  probe.proposed_effects_override = true;
  probe.artifact_properties.runtime = kEffectRuntime;
  Context context = make_context(probe);
  const std::array bindings{
      EffectResourceBinding{resource, storage.footprint()}};
  auto compiled =
      hyle::graph::compile(graph, context, {.effect_bindings = bindings});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
  EXPECT_EQ(probe.single_artifact_count, 0u);
  EXPECT_EQ(*storage.value(), 0);
}

TEST(GraphRegions, EffectOrderingCannotLeaveAndReenterAContractedRegion) {
  ResourceStorage storage;
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  std::array<NodeId, 3> nodes;
  for (auto &node : nodes)
    node = add_resource_touch(builder, resource, storage,
                              ResourceAccess::kReadWrite);
  ASSERT_TRUE(builder.depends_on(nodes[1], nodes[0]).ok());
  ASSERT_TRUE(builder.depends_on(nodes[2], nodes[1]).ok());
  const Graph graph = std::move(builder).finish().value();
  EXPECT_FALSE(
      hyle::graph::is_region_contractible(graph, std::array{nodes[0], nodes[2]})
          .value());
  EXPECT_TRUE(hyle::graph::is_region_contractible(graph, nodes).value());
}

TEST(GraphRegions,
     UnknownEffectsStillRequireAnExplicitRegionPreservationPromise) {
  GraphBuilder builder = GraphBuilder::create().value();
  ASSERT_TRUE(builder
                  .add_node(Operation::custom(ExternalEffect{}),
                            std::array<ValueId, 0>{})
                  .has_value());
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe{.mode = ProposalMode::kResourceSingletons};
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(graph, context).value();
  EXPECT_TRUE(probe.observed_unknown_effects);
  EXPECT_EQ(probe.observed_effect_count, 0u);
  EXPECT_EQ(probe.compile_count, 0u);
  EXPECT_EQ(compiled.report.region_count(), 0u);
  auto invocation = compiled.executable.create_invocation().value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.effect_submit_count, 1u);
}

TEST(GraphRegions, TensorBindingCannotHideAnEffectAliasInEitherBindingOrder) {
  ResourceStorage storage{0};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  add_resource_touch(builder, resource, storage, ResourceAccess::kReadWrite);
  const auto input_id = builder.input(row(1)).value();
  const auto output_id = builder.add(Unary{UnaryKind::kNeg}, input_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  const std::array bindings{
      EffectResourceBinding{resource, storage.footprint()}};
  auto compiled =
      hyle::graph::compile(graph, context, {.effect_bindings = bindings})
          .value();
  OwnedTensor input = make_tensor(row(1));
  OwnedTensor output = make_tensor(row(1));
  write(input, std::array{3.0F});
  const auto aliased =
      TensorView::create(row(1), storage.buffer.view()).value();
  auto queue = context.create_queue().value();
  for (bool input_first : {false, true}) {
    auto invocation = compiled.executable.create_invocation().value();
    const auto first = input_first ? input_id : output_id;
    const auto second = input_first ? output_id : input_id;
    EXPECT_TRUE(invocation.bind(first, aliased).IsInvalidArgument());
    ASSERT_TRUE(
        invocation.bind(first, input_first ? input.view : output.view).ok());
    EXPECT_TRUE(invocation.bind(second, aliased).IsInvalidArgument());
    ASSERT_TRUE(
        invocation.bind(second, input_first ? output.view : input.view).ok());
    ASSERT_TRUE(invocation.submit(queue)->wait().ok());
    EXPECT_EQ(read(output, 1), std::vector<float>{-3.0F});
  }
  EXPECT_EQ(*storage.value(), 2);
}

TEST(GraphRegions, ConstantsCannotHideAnEffectAlias) {
  ResourceStorage storage;
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  add_resource_touch(builder, resource, storage, ResourceAccess::kWrite);
  const auto constant = builder.constant(row(1)).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, constant).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  const auto alias = TensorView::create(row(1), storage.buffer.view()).value();
  std::array constants{
      hyle::graph::ConstantBinding{constant, alias.as_const()}};
  const std::array resources{
      EffectResourceBinding{resource, storage.footprint()}};
  for (const auto policy : {hyle::graph::ConstantPolicy::kBorrow,
                            hyle::graph::ConstantPolicy::kMaterialize}) {
    constants[0].policy = policy;
    auto result = hyle::graph::compile(
        graph, context, {.constants = constants, .effect_bindings = resources});
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsInvalidArgument());
    EXPECT_EQ(probe.single_artifact_count, 0u);
  }
  OwnedTensor independent = make_tensor(row(1));
  constants[0].tensor = independent.view.as_const();
  ASSERT_TRUE(hyle::graph::compile(
                  graph, context,
                  {.constants = constants, .effect_bindings = resources})
                  .has_value());
}

TEST(GraphRegions, ExactShapeSpecializationKeepsResourceIdentityAndBinding) {
  ResourceStorage storage{0};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  add_resource_touch(builder, resource, storage, ResourceAccess::kReadWrite, 3);
  const auto width = builder.dimension({.minimum = 1, .maximum = 4}).value();
  const auto spec =
      hyle::graph::TensorSpec::dense(DataType::kFloat32,
                                     std::array{hyle::graph::Dimension(width)})
          .value();
  const auto input_id = builder.input(spec).value();
  const auto output_id =
      builder.add(Unary{UnaryKind::kNeg}, spec, input_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe;
  Context context = make_context(probe);
  const std::array resources{
      EffectResourceBinding{resource, storage.footprint()}};
  const std::array shape{hyle::graph::ShapeBinding{width, 4}};
  auto compiled =
      hyle::graph::compile(graph, context,
                           {.shape = shape, .effect_bindings = resources})
          .value();
  OwnedTensor input = make_tensor(row(4));
  OwnedTensor output = make_tensor(row(4));
  write(input, std::array{1.0F, 2.0F, 3.0F, 4.0F});
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(*storage.value(), 3);
  EXPECT_EQ(read(output, 4), (std::vector<float>{-1, -2, -3, -4}));
}

TEST(GraphRegions, ResourceCompileAllocationFailuresRollBackAndCanRetry) {
  struct AllocationState final {
    AllocatorRef backing = hyle::mem::system_allocator();
    std::size_t attempts = 0, fail_at = SIZE_MAX, live = 0;
  } state;
  const AllocatorRef allocator{
      +[](void *opaque, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        auto &s = *static_cast<AllocationState *>(opaque);
        if (s.attempts++ == s.fail_at)
          return nullptr;
        void *p = s.backing.allocate(bytes, alignment);
        s.live += p != nullptr ? 1u : 0u;
        return p;
      },
      +[](void *opaque, void *pointer, std::size_t bytes,
          std::size_t alignment) noexcept {
        auto &s = *static_cast<AllocationState *>(opaque);
        s.live -= pointer != nullptr ? 1u : 0u;
        s.backing.deallocate(pointer, bytes, alignment);
      },
      +[](void *opaque, void *pointer, std::size_t old_bytes,
          std::size_t new_bytes, std::size_t alignment) noexcept -> void * {
        auto &s = *static_cast<AllocationState *>(opaque);
        if (new_bytes == 0) {
          s.live -= pointer != nullptr ? 1u : 0u;
          return s.backing.reallocate(pointer, old_bytes, 0, alignment);
        }
        if (s.attempts++ == s.fail_at)
          return nullptr;
        void *p =
            s.backing.reallocate(pointer, old_bytes, new_bytes, alignment);
        if (pointer == nullptr && p != nullptr)
          ++s.live;
        return p;
      },
      &state};
  std::array<ResourceStorage, 18> storage;
  std::array<EffectResourceBinding, 18> bindings;
  GraphBuilder builder = GraphBuilder::create().value();
  for (std::size_t index = 0; index < storage.size(); ++index) {
    const auto resource = builder.effect_resource().value();
    bindings[index] = {resource, storage[index].footprint()};
    add_resource_touch(builder, resource, storage[index],
                       ResourceAccess::kReadWrite);
  }
  const Graph graph = std::move(builder).finish().value();
  RegionProbe probe{.mode = ProposalMode::kResourcePairs};
  probe.artifact_properties.preserves_effects = true;
  probe.artifact_properties.runtime = kEffectRuntime;
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(probe), 100}};
  auto context =
      Context::create(hyle::compute::cpu::device(),
                      {.metadata_allocator = allocator, .providers = providers})
          .value();
  const auto live = state.live;
  const CompileOptions options{.effect_bindings = bindings,
                               .operation_fallback =
                                   OperationFallbackPolicy::kReject};
  state.attempts = 0;
  {
    auto compiled = hyle::graph::compile(graph, context, options);
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), 9u);
  }
  const auto count = state.attempts;
  ASSERT_GT(count, 0u);
  EXPECT_EQ(state.live, live);
  for (std::size_t failure = 0; failure < count; ++failure) {
    SCOPED_TRACE(failure);
    state.attempts = 0;
    state.fail_at = failure;
    {
      auto failed = hyle::graph::compile(graph, context, options);
      ASSERT_FALSE(failed.has_value());
      EXPECT_TRUE(failed.error().IsResourceExhausted())
          << failed.error().message();
    }
    EXPECT_EQ(state.live, live);
    state.fail_at = SIZE_MAX;
    {
      auto recovered = hyle::graph::compile(graph, context, options);
      ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
    }
    EXPECT_EQ(state.live, live);
  }
  context = {};
  EXPECT_EQ(state.live, 0u);
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
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(probe), 100}};
  Context first =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  const std::array native_providers{
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 100}};
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
  probe.artifact_properties.runtime.replay_class =
      ProviderReplayClass::kManaged;
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
  EXPECT_EQ(probe.single_proposal_count, 2u);
  EXPECT_EQ(probe.single_artifact_count, 0u);

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
  EXPECT_FALSE(
      compiled.executable.requires_queue(hyle::graph::ExecutionLaneId{1})
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
  EXPECT_TRUE(invocation.submit().error().IsFailedPrecondition());
  auto recreate = [&] {
    invocation = compiled.executable.create_invocation().value();
    EXPECT_TRUE(invocation.bind(logical.input, input.view).ok());
    EXPECT_TRUE(invocation.bind(logical.output, output.view).ok());
  };
  recreate();

  probe.terminal_failure = true;
  EXPECT_TRUE(invocation.submit()->wait().IsInternal());
  EXPECT_FALSE(invocation.in_flight());
  probe.terminal_failure = false;
  EXPECT_TRUE(invocation.submit().error().IsFailedPrecondition());
  EXPECT_TRUE(invocation.reset().IsFailedPrecondition());
  recreate();

  Completion abandoned = invocation.submit().value();
  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 2u);
  abandoned = {};
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.destroy_invocation_count, 2u);
  finish_abandoned_prefix(probe);
  EXPECT_EQ(probe.destroy_invocation_count, 3u);
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
  ASSERT_TRUE(
      invocation
          .prepare_replay(std::array{hyle::graph::QueueBinding{
              .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}})
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
  const NegationGraph logical = make_negations(2, true, UnaryKind::kExp);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.report.region_count(), 1u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_EQ(probe.device_wait_count, 1u);
  const auto actual = read(output, values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    EXPECT_NEAR(actual[index], std::exp(values[index]),
                1.0e-5F * std::exp(values[index]));
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
  ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
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

  OwnedTensor input =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor output =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(second_storage));
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
    EXPECT_EQ(step.fallback_reason, CompileFallbackReason::kRegionDeclined);
  }
}

TEST(GraphRegions, LaterRegionResourceDeclineDoesNotPrepareAnEarlierLeaf) {
  const auto logical = make_negations(3);
  HostResource unavailable;
  RegionProbe probe{.mode = ProposalMode::kSingleton, .singleton_node = 2};
  probe.required_resource = hyle::mem::system_resource();
  probe.region_output_resource = hyle::mem::MemoryResourceRef(unavailable);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 0u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
  EXPECT_EQ(probe.compile_count, 0u);
  // The abandoned region partition prepares neither of its two prefix leaves.
  // Each leaf is prepared exactly once, in the final all-leaf construction.
  EXPECT_EQ(probe.single_artifact_count, 3u);
  EXPECT_EQ(probe.destroy_prepared_count, 0u);
  EXPECT_EQ(probe.single_proposal_count, 3u);
  EXPECT_EQ(probe.region_resource_count, 1u);
}

TEST(GraphRegions, UnavailableFinalOutputCopyFailsBeforeAnyArtifact) {
  const auto logical = make_negations(2);
  HostResource inaccessible{false};
  RegionProbe probe;
  probe.required_resource = hyle::mem::system_resource();
  auto context = make_context(probe);
  const Environment environment{
      {&context, probe.required_resource},
      {&context, hyle::mem::MemoryResourceRef(inaccessible)}};
  const std::array external{hyle::graph::ExternalResourceContract{
      logical.output, environment.storage(1)}};
  for (const auto selection : {RegionSelectionPolicy::kDisabled,
                               RegionSelectionPolicy::kAllowProviderRegions}) {
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.external_resources = external, .region_selection = selection});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsUnimplemented());
    EXPECT_EQ(probe.compile_count, 0u);
    EXPECT_EQ(probe.single_artifact_count, 0u);
    EXPECT_EQ(probe.destroy_prepared_count, 0u);
  }
}

TEST(GraphRegions, PreparedRegionReplayKeepsResourceValidationCold) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  auto invocation = compiled.executable.create_invocation().value();
  auto input = make_tensor(row(8));
  auto output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto cold_checks =
      probe.resource_validation_count.load(std::memory_order_relaxed);
  ASSERT_GT(cold_checks, 0u);
  for (float first : {1.F, 9.F}) {
    const std::array values{first, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
    write(input, values);
    auto completion = invocation.submit(queue);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    EXPECT_EQ(read(output, 8),
              std::vector<float>(values.begin(), values.end()));
  }
  EXPECT_EQ(probe.resource_validation_count.load(std::memory_order_relaxed),
            cold_checks);
  EXPECT_EQ(probe.submit_count, 2u);

  HostResource other_resource;
  auto foreign =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(other_resource));
  EXPECT_TRUE(invocation.bind(logical.input, foreign.view).IsInvalidArgument());
}

TEST(GraphRegions, PreparedReplayChecksReplacedQueueOwnerAndClosedContext) {
  RegionProbe probe;
  RegionProbe other_probe;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto other_context = make_context(other_probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto invocation = compiled.executable.create_invocation().value();
  auto input = make_tensor(row(8));
  auto output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());

  queue = other_context.create_queue().value();
  auto rejected = invocation.submit(queue);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(probe.submit_count + other_probe.submit_count, 0u);

  queue = context.create_queue().value();
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_EQ(read(output, 8), std::vector<float>(values.begin(), values.end()));
  EXPECT_EQ(probe.submit_count, 1u);
  ASSERT_TRUE(context.close().ok());
  auto closed = invocation.submit(queue);
  ASSERT_FALSE(closed);
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
  EXPECT_EQ(probe.submit_count, 1u);
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
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0), .queue = &queue}};

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

  const std::array invalid_queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0), .queue = nullptr}};
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
  EXPECT_EQ(probe.single_proposal_count, 3u);
  EXPECT_EQ(probe.single_artifact_count, 0u);
  auto fallback = compiled.report.step(1);
  ASSERT_TRUE(fallback.has_value());
  EXPECT_EQ(fallback->kind, CompileStepKind::kOperationFallback);
  EXPECT_EQ(fallback->fallback_reason,
            CompileFallbackReason::kNoEligibleRegion);
  EXPECT_FALSE(fallback->persistent_bytes.has_value());
  EXPECT_EQ(fallback->provider.id, hyle::compute::ggml_cpu::kProviderId);

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

TEST(GraphRegions, FailedCommandAndCapturedSlotsRetireAtTheActualTerminal) {
  for (bool capture : {false, true}) {
    for (bool abandon : {false, true}) {
      SCOPED_TRACE(capture);
      SCOPED_TRACE(abandon);
      RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
      probe.capture_supported = capture;
      probe.artifact_properties.runtime.capture_compatible = true;
      const auto logical = make_negations(2);
      auto context = make_context(probe);
      auto compiled = hyle::graph::compile(logical.graph, context).value();
      auto input = make_tensor(row(8)), output = make_tensor(row(8));
      const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
      write(input, values);
      auto queue = context.create_queue().value();
      const std::array queues{hyle::graph::QueueBinding{
          .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}};
      const hyle::graph::ReplayOptions options{
          .mode = capture ? hyle::graph::ReplayMode::kRequireCapture
                          : hyle::graph::ReplayMode::kCommands};
      auto make_slot = [&] {
        auto result = compiled.executable.create_invocation().value();
        EXPECT_TRUE(result.bind(logical.input, input.view).ok());
        EXPECT_TRUE(result.bind(logical.output, output.view).ok());
        EXPECT_TRUE(result.prepare_replay(queues, options).ok());
        return result;
      };
      auto failed_slot = make_slot();
      auto independent_slot = make_slot();
      auto completion = failed_slot.submit(queues).value();
      EXPECT_TRUE(failed_slot.in_flight());
      probe.terminal_failure = true;
      if (abandon) {
        completion = {};
        EXPECT_TRUE(failed_slot.in_flight());
        finish_abandoned_prefix(probe);
      } else {
        EXPECT_TRUE(completion.wait().IsInternal());
      }
      EXPECT_FALSE(failed_slot.in_flight());
      probe.terminal_failure = false;
      const auto submits = probe.submit_attempt_count;
      const auto captures = probe.capture_launch_count;
      EXPECT_TRUE(failed_slot.reset().IsFailedPrecondition());
      EXPECT_TRUE(failed_slot.bind(logical.input, input.view.as_const())
                      .IsFailedPrecondition());
      EXPECT_TRUE(
          failed_slot.bind(logical.output, output.view).IsFailedPrecondition());
      EXPECT_TRUE(
          failed_slot.prepare_replay(queues, options).IsFailedPrecondition());
      EXPECT_TRUE(failed_slot.submit(queues).error().IsFailedPrecondition());
      EXPECT_TRUE(
          failed_slot.output(logical.output).error().IsFailedPrecondition());
      EXPECT_EQ(probe.submit_attempt_count, submits);
      EXPECT_EQ(probe.capture_launch_count, captures);
      ASSERT_TRUE(independent_slot.submit(queues)->wait().ok());
      ASSERT_TRUE(independent_slot.submit(queues)->wait().ok());
      auto fresh = make_slot();
      ASSERT_TRUE(fresh.submit(queues)->wait().ok());
    }
  }
}

TEST(GraphRegions, ConfiguredProviderOrderPrecedesSuffixDispatchEstimate) {
  RegionProbe probe{.mode = ProposalMode::kOverlappingRoutes};
  const NegationGraph logical = make_negations(3);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(logical.graph, context, CompileOptions{}).value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compiled_begin, 0u);
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
  probe.implementation = hyle::compute::ProviderImplementationId{88};
  const Graph logical = make_reordered_non_contiguous_negations();
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical, context).value();

  ASSERT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.step(0)->kind, CompileStepKind::kOperationFallback);
  EXPECT_EQ(compiled.report.step_node(0, 0)->ordinal(), 2u);
  EXPECT_EQ(compiled.report.step(1)->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled.report.step_node(1, 0)->ordinal(), 0u);
  EXPECT_EQ(compiled.report.step_node(1, 1)->ordinal(), 3u);
  EXPECT_EQ(compiled.report.step(1)->implementation, probe.implementation);
  EXPECT_NE(compiled.report.step(0)->implementation, probe.implementation);
  EXPECT_NE(compiled.report.step(2)->implementation, probe.implementation);
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
  EXPECT_EQ(compiled.report.search().route_plan_rejections, 1u);
  EXPECT_EQ(compiled.report.search().region_route_rejections, 0u);
  EXPECT_EQ(compiled.report.search().placement_attempts, 3u);
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
  const std::array region_only{
      hyle::compute::ProviderBinding{region_provider(probe), 100}};
  Context region_context =
      Context::create(hyle::compute::cpu::device(), {.providers = region_only})
          .value();
  const std::array native_only{
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 100}};
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
  EXPECT_EQ(probe.compiled_begin, 0u);
  EXPECT_EQ(probe.compiled_count, 2u);
  EXPECT_EQ(probe.single_proposal_count, 3u);
  EXPECT_EQ(probe.single_artifact_count, 0u);
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

TEST(GraphRegions, IgnoresUnusedSameDeviceContextsInPlanSummary) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  Context unused =
      Context::create(hyle::compute::cpu::device(), region_provider(probe))
          .value();
  const std::array native_only{
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 100}};
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
      hyle::graph::ExternalResourceContract{.value = logical.input,
                                            .storage = environment.storage(1)},
      hyle::graph::ExternalResourceContract{.value = logical.output,
                                            .storage = environment.storage(1)},
  };
  const auto compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  EXPECT_FALSE(
      compiled.report.lane(environment.lane(0).ordinal())->queue_required);
  EXPECT_TRUE(
      compiled.report.lane(environment.lane(1).ordinal())->queue_required);
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
  EXPECT_EQ(probe.single_proposal_count, 2u);
  EXPECT_EQ(probe.single_artifact_count, 0u);
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
    const std::array providers{
        hyle::compute::ProviderBinding{region_provider(first), 100},
        hyle::compute::ProviderBinding{region_provider(unknown), 200},
        hyle::compute::ProviderBinding{region_provider(third), 300},
        hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 400}};
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

TEST(GraphRegions, DefaultPolicyUsesExplicitProviderPreference) {
  RegionProbe first;
  RegionProbe second;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
}

TEST(GraphRegions, DispatchCountCannotOverrideProviderPreference) {
  RegionProbe first{.proposal_dispatch_count = 2};
  RegionProbe second{.proposal_dispatch_count = 1};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
}

TEST(GraphRegions, LeafAndRegionProvidersShareExplicitPreference) {
  const auto logical = make_negations(2);
  for (const bool region_first : {false, true}) {
    RegionProbe probe;
    std::array providers{
        hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(),
                                       region_first ? 200u : 100u},
        hyle::compute::ProviderBinding{region_provider(probe),
                                       region_first ? 100u : 200u}};
    auto context =
        Context::create(hyle::compute::cpu::device(), {.providers = providers})
            .value();
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), region_first ? 1u : 0u);
    EXPECT_EQ(probe.compile_count, region_first ? 1u : 0u);
    EXPECT_EQ(probe.propose_count, 1u);
    EXPECT_EQ(probe.region_resource_count, 1u);
    EXPECT_EQ(compiled->report.search().route_search_states,
              region_first ? 1u : 2u);
    EXPECT_EQ(compiled->report.search().route_plan_rejections, 0u);
    auto input = make_tensor(row(8)), output = make_tensor(row(8));
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
}

TEST(GraphRegions, EqualPreferenceUsesPlanFactsBeforeRegistrationOrder) {
  const auto logical = make_negations(2);
  for (const bool leaf_registered_first : {false, true}) {
    RegionProbe probe;
    const auto region =
        hyle::compute::ProviderBinding{region_provider(probe), 100};
    const auto leaf =
        hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 100};
    Context context;
    if (leaf_registered_first) {
      const std::array providers{leaf, region};
      context = Context::create(hyle::compute::cpu::device(),
                                {.providers = providers})
                    .value();
    } else {
      const std::array providers{region, leaf};
      context = Context::create(hyle::compute::cpu::device(),
                                {.providers = providers})
                    .value();
    }
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), 1u);
    EXPECT_EQ(compiled->report.step_count(), 1u);
    EXPECT_EQ(probe.compile_count, 1u);
  }
}

TEST(GraphRegions, DispatchBreaksOtherwiseEquivalentSameProviderRouteTies) {
  RegionProbe probe{.mode = ProposalMode::kEquivalentRoutes};
  // If route 2 were prepared at all this would be fatal, not a safe retry.
  probe.region_outcome_route = 2;
  probe.region_outcome_override = ProviderPrepareOutcome::error(
      Status::Internal("higher dispatch route was prepared"));
  auto context = make_context(probe);
  const auto logical = make_negations(2);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(compiled->report.search().route_search_states, 1u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 0u);
}

TEST(GraphRegions, SearchBudgetIsCumulativeAcrossTransactionalDeclines) {
  const auto logical = make_negations(2);
  for (const std::size_t budget : {0u, 1u, 2u}) {
    RegionProbe first, second;
    first.region_outcome_override = ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("first route declined"));
    auto context = make_context(first, second);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.max_placement_attempts = 2,
         .max_route_search_states = budget,
         .operation_fallback = OperationFallbackPolicy::kReject,
         .search_policy = CompileSearchPolicy::kRequireUnexhausted});
    if (budget == 0) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsInvalidArgument());
      EXPECT_EQ(first.propose_count + second.propose_count, 0u);
    } else if (budget == 1) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsResourceExhausted());
      EXPECT_NE(compiled.error().message().find("search state budget"),
                std::string_view::npos);
      EXPECT_EQ(first.compile_count, 1u);
      EXPECT_EQ(second.compile_count, 0u);
    } else {
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      EXPECT_EQ(compiled->report.search().route_search_states, 2u);
      EXPECT_EQ(compiled->report.search().placement_attempts, 2u);
      EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
      EXPECT_FALSE(compiled->report.search().optimization_limited());
      EXPECT_EQ(first.compile_count, 1u);
      EXPECT_EQ(second.compile_count, 1u);
      EXPECT_EQ(first.propose_count, 1u);
      EXPECT_EQ(second.propose_count, 1u);
    }
  }
}

TEST(GraphRegions, SearchBudgetNeverPreparesAPartiallyCoveredGraph) {
  RegionProbe probe;
  probe.required_resource = hyle::mem::system_resource();
  auto context = make_context(probe);
  const auto logical = make_negations(3);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.region_selection = RegionSelectionPolicy::kDisabled,
       .max_route_search_states = 2});
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(probe.single_artifact_count, 0u);
  EXPECT_EQ(probe.single_proposal_count, 3u);
}

TEST(GraphRegions, PhysicalCopyFailureBacktracksWithoutPreparingArtifacts) {
  const auto logical = make_negations(1);
  for (const bool fatal : {false, true}) {
    HostResource first_storage, second_storage;
    RegionProbe first, second;
    first.required_resource = first.backend_only_resource =
        hyle::mem::MemoryResourceRef(first_storage);
    second.required_resource = second.backend_only_resource =
        hyle::mem::MemoryResourceRef(second_storage);
    if (fatal) {
      first.validation_failure_resource = second.required_resource;
      first.validation_failure =
          Status::Internal("device resource validation failed");
    }
    const std::array first_providers{
        hyle::compute::ProviderBinding{region_provider(first), 100}};
    const std::array second_providers{
        hyle::compute::ProviderBinding{region_provider(second), 100}};
    auto first_context =
        Context::create(async_device(first), {.providers = first_providers})
            .value();
    auto second_context =
        Context::create(async_device(second), {.providers = second_providers})
            .value();
    const Environment environment{{&first_context, first.required_resource},
                                  {&second_context, second.required_resource}};
    const std::array external{hyle::graph::ExternalResourceContract{
        logical.input, environment.storage(1)}};
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.external_resources = external,
         .max_placement_attempts = 1,
         .region_selection = RegionSelectionPolicy::kDisabled,
         .max_route_search_states = 2});
    EXPECT_EQ(first.single_artifact_count, 0u);
    if (fatal) {
      ASSERT_FALSE(compiled);
      EXPECT_EQ(compiled.error(), first.validation_failure);
      EXPECT_EQ(second.single_artifact_count, 0u);
      continue;
    }
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.search().route_plan_rejections, 1u);
    EXPECT_EQ(compiled->report.search().route_search_states, 2u);
    EXPECT_EQ(compiled->report.search().placement_attempts, 1u);
    EXPECT_EQ(compiled->report.search().operation_route_rejections, 0u);
    EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
    EXPECT_EQ(compiled->report.copy_bytes(), 0u);
    EXPECT_EQ(second.single_artifact_count, 1u);
    auto input = make_tensor(row(8), second.required_resource),
         output = make_tensor(row(8), second.required_resource);
    const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
    write(input, values);
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
    ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
    auto queue = second_context.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
    std::vector<float> expected(values.begin(), values.end());
    for (auto &value : expected)
      value = -value;
    EXPECT_EQ(read(output, values.size()), expected);
  }
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
  EXPECT_EQ(probe.single_proposal_count, 2u);
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
  EXPECT_EQ(probe.single_proposal_count, 6u);
  EXPECT_EQ(probe.single_artifact_count, 0u);
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
    EXPECT_EQ(probe.single_proposal_count, 0u);
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
  for (const Status &failure :
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
    EXPECT_EQ(probe.single_proposal_count, 0u);
  }
}

TEST(GraphRegions, FinalWorkspaceRejectionTriesTheSmallerRegion) {
  const auto logical = make_negations(1);
  for (const auto reuse :
       {hyle::graph::WorkspaceReusePolicy::kExistingOrderOnly,
        hyle::graph::WorkspaceReusePolicy::kAllowSerialization}) {
    RegionProbe first{.mode = ProposalMode::kSingleton};
    RegionProbe second{.mode = ProposalMode::kSingleton};
    use_delegated_runtime(first);
    first.artifact_properties.workspace = {.size = 128, .alignment = 64};
    first.implementation = hyle::compute::ProviderImplementationId{41};
    second.implementation = hyle::compute::ProviderImplementationId{42};
    second.observe_cleanup = &first;
    auto context = make_context(first, second);
    const std::array storage{hyle::graph::StorageDomain{
        .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 0}};
    const std::array targets{hyle::graph::ExecutionTarget{
        .context = &context,
        .default_storage = hyle::graph::StorageDomainId{0}}};
    auto compiled = hyle::graph::compile(
        logical.graph, {.storage_domains = storage, .targets = targets},
        {.workspace_reuse = reuse,
         .operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    EXPECT_EQ(second.observed_cleanup_at_compile, 1u);
    EXPECT_EQ(first.propose_count, 1u);
    EXPECT_EQ(second.propose_count, 1u);
    EXPECT_EQ(compiled->report.search().route_plan_rejections, 1u);
    EXPECT_EQ(compiled->report.search().region_route_rejections, 0u);
    EXPECT_EQ(compiled->report.search().placement_attempts, 2u);
    EXPECT_EQ(compiled->report.lane_count(), 1u);
    EXPECT_EQ(compiled->report.step(0)->implementation, second.implementation);
    EXPECT_EQ(compiled->report.step(0)->provider_instance,
              context.provider_instance_id(1).value());
    auto input = make_tensor(row(8)), output = make_tensor(row(8));
    write(input, std::array{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F});
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(run.bind(logical.input, input.view).ok());
    ASSERT_TRUE(run.bind(logical.output, output.view).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(run.submit(queue).value().wait().ok());
    EXPECT_EQ(read(output, 8),
              (std::vector<float>{-1.F, 2.F, -3.F, 4.F, -5.F, 6.F, -7.F, 8.F}));
  }
}

TEST(GraphRegions, MaterializedConstantsSurviveRouteRetryAndArtifactTeardown) {
  HostResource resource;
  const hyle::mem::MemoryResourceRef owned_resource(resource);
  RegionProbe first{.mode = ProposalMode::kSingleton};
  RegionProbe second{.mode = ProposalMode::kSingleton};
  use_delegated_runtime(first);
  first.artifact_properties.workspace = {.size = 128, .alignment = 64};
  first.constant_live_allocations = &resource.live_allocations;
  second.constant_live_allocations = &resource.live_allocations;
  auto context = make_context(first, second);
  GraphBuilder builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(8)).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, constant).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(8));
  write(source, std::array{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{constant, source.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array storage{hyle::graph::StorageDomain{
      .resource = owned_resource, .workspace_budget_bytes = 0}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  CompileReport report;
  {
    auto compiled = hyle::graph::compile(
        graph, {.storage_domains = storage, .targets = targets},
        {.constants = bindings,
         .operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    report = compiled->report;
    EXPECT_EQ(report.search().route_plan_rejections, 1u);
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(second.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    EXPECT_EQ(first.constant_allocations_at_destroy, 1u);
    EXPECT_EQ(resource.allocation_count, 1u);
    ASSERT_TRUE(first.compiled_constant);
    ASSERT_TRUE(second.compiled_constant);
    EXPECT_EQ(first.compiled_constant->storage().resource(), owned_resource);
    EXPECT_EQ(first.compiled_constant->storage().native_handle(),
              second.compiled_constant->storage().native_handle());
    EXPECT_NE(first.compiled_constant->storage().native_handle(),
              source.view.storage().native_handle());
    ASSERT_TRUE(source.buffer.reset().ok());
    auto destination = make_tensor(row(8), owned_resource);
    auto run = compiled->executable.create_invocation().value();
    compiled->executable = {};
    ASSERT_TRUE(run.bind(output, destination.view).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(run.submit(queue)->wait().ok());
    EXPECT_EQ(read(destination, 8),
              (std::vector<float>{-1.F, 2.F, -3.F, 4.F, -5.F, 6.F, -7.F, 8.F}));
    // The external output is not needed by artifact teardown. Removing it
    // makes the remaining allocation unambiguously the canonical constant.
    ASSERT_TRUE(run.reset().ok());
    ASSERT_TRUE(destination.buffer.reset().ok());
    EXPECT_EQ(resource.live_allocations, 1u);
    EXPECT_EQ(second.destroy_prepared_count, 0u);
  }
  EXPECT_EQ(second.destroy_prepared_count, 1u);
  EXPECT_EQ(second.constant_allocations_at_destroy, 1u);
  EXPECT_EQ(resource.live_allocations, 0u);
  EXPECT_EQ(report.materialized_constant_bytes(), 8 * sizeof(float));
}

TEST(GraphRegions, ConstantCopiesWaitForAcceptedFailuresWithoutRetryingThem) {
  HostResource resource;
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  probe.asynchronous_backend = true;
  auto context = make_context(probe);
  const Environment environment{
      {&context, hyle::mem::MemoryResourceRef(resource)}};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(4)).value();
  ASSERT_TRUE(builder.publish(constant).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{constant, source.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  probe.transfer_supported = false;
  auto unsupported = hyle::graph::compile(graph, environment.environment(),
                                          {.constants = bindings});
  ASSERT_FALSE(unsupported);
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  EXPECT_EQ(probe.copy_count, 0u);
  EXPECT_EQ(probe.completion_wait_count, 0u);
  EXPECT_EQ(resource.live_allocations, 0u);
  probe.transfer_supported = true;
  probe.terminal_failure = true;
  auto failed = hyle::graph::compile(graph, environment.environment(),
                                     {.constants = bindings});
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsInternal());
  EXPECT_EQ(probe.copy_count, 1u);
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_EQ(resource.live_allocations, 0u);
  EXPECT_EQ(read(source, 4), (std::vector<float>{1.F, 2.F, 3.F, 4.F}));
  probe.terminal_failure = false;
  {
    auto retry = hyle::graph::compile(graph, environment.environment(),
                                      {.constants = bindings});
    ASSERT_TRUE(retry) << retry.error().message();
    EXPECT_EQ(resource.live_allocations, 1u);
    EXPECT_EQ(probe.copy_count, 2u);
    EXPECT_EQ(probe.completion_wait_count, 2u);
  }
  EXPECT_EQ(resource.live_allocations, 0u);
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, FinalWorkspaceRejectionCanUseALeafPlan) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  probe.artifact_properties.workspace = {.size = 128, .alignment = 64};
  auto context = make_context(probe);
  const std::array storage{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 64}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  auto compiled = hyle::graph::compile(
      logical.graph, {.storage_domains = storage, .targets = targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
  EXPECT_EQ(compiled->report.region_count(), 0u);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 2u);
  EXPECT_EQ(compiled->report.search().route_plan_rejections, 1u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 3u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue).value().wait().ok());
  EXPECT_EQ(read(output, 8), std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, FinalWorkspaceRetryDiscardsFailedSerializationEdges) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto a = builder.input(row(8)).value();
  const auto b = builder.input(row(8)).value();
  const auto x = builder.add(Unary{UnaryKind::kNeg}, a).value();
  const auto y = builder.add(Unary{UnaryKind::kNeg}, b).value();
  ASSERT_TRUE(builder.publish(x).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  const auto graph = std::move(builder).finish().value();
  RegionProbe large{.mode = ProposalMode::kSingleton};
  RegionProbe small{.mode = ProposalMode::kSingleton};
  RegionProbe side{.mode = ProposalMode::kSingleton, .singleton_node = 1};
  large.artifact_properties.workspace = {.size = 128, .alignment = 32};
  small.artifact_properties.workspace = {.size = 32, .alignment = 32};
  side.artifact_properties.workspace = {.size = 32, .alignment = 32};
  auto first = make_context(large, small), second = make_context(side);
  const std::array storage{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 64}};
  const std::array targets{
      hyle::graph::ExecutionTarget{&first, hyle::graph::StorageDomainId{0}},
      hyle::graph::ExecutionTarget{&second, hyle::graph::StorageDomainId{0}}};
  const std::array placements{
      hyle::graph::NodePlacement{graph.node_id(0).value(),
                                 hyle::graph::ExecutionTargetId{0},
                                 hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{graph.node_id(1).value(),
                                 hyle::graph::ExecutionTargetId{1},
                                 hyle::graph::PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(
      graph, {.storage_domains = storage, .targets = targets},
      {.node_placements = placements,
       .workspace_reuse =
           hyle::graph::WorkspaceReusePolicy::kAllowSerialization,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(large.compile_count, 1u);
  EXPECT_EQ(large.destroy_prepared_count, 1u);
  EXPECT_EQ(side.compile_count, 2u);
  EXPECT_EQ(side.destroy_prepared_count, 1u);
  EXPECT_EQ(compiled->report.search().route_plan_rejections, 1u);
  EXPECT_EQ(compiled->report.serialization_edge_count(), 0u);
  EXPECT_EQ(compiled->report.workspace_bytes(), 64u);
  EXPECT_EQ(compiled->report.segment(0)->dependency_count, 0u);
  EXPECT_EQ(compiled->report.segment(1)->dependency_count, 0u);
}

TEST(GraphRegions, FinalWorkspaceRetryKeepsHardConstraintsAndBudgets) {
  const auto logical = make_negations(1);
  for (const int scenario : {0, 1, 2, 3}) {
    SCOPED_TRACE(scenario);
    RegionProbe first{.mode = ProposalMode::kSingleton};
    RegionProbe second{.mode = ProposalMode::kSingleton};
    first.artifact_properties.workspace = {.size = 128, .alignment = 64};
    if (scenario == 3)
      second.artifact_properties.workspace =
          first.artifact_properties.workspace;
    auto context = make_context(first, second);
    const std::array storage{hyle::graph::StorageDomain{
        .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 0}};
    const std::array targets{hyle::graph::ExecutionTarget{
        .context = &context,
        .default_storage = hyle::graph::StorageDomainId{0}}};
    const std::array constraints{hyle::graph::NodeProviderConstraint{
        logical.graph.node_id(0).value(), kRegionProviderId,
        context.provider_instance_id(0).value()}};
    CompileOptions options;
    options.operation_fallback = OperationFallbackPolicy::kReject;
    if (scenario == 0)
      options.max_placement_attempts = 1;
    if (scenario == 1)
      options.max_route_search_states = 1;
    if (scenario == 2)
      options.node_providers = constraints;
    auto compiled = hyle::graph::compile(
        logical.graph, {.storage_domains = storage, .targets = targets},
        options);
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsResourceExhausted());
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    EXPECT_EQ(second.compile_count, scenario == 3 ? 1u : 0u);
    EXPECT_EQ(second.destroy_prepared_count, second.compile_count);
    EXPECT_EQ(first.single_artifact_count + second.single_artifact_count, 0u);
  }
}

TEST(GraphRegions, FinalWorkspaceRetryDoesNotHideRealProviderError) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton};
  RegionProbe second{.mode = ProposalMode::kSingleton};
  first.artifact_properties.workspace = {.size = 128, .alignment = 64};
  const Status failure =
      Status::ResourceExhausted("real prepare allocation failure");
  second.region_outcome_override = ProviderPrepareOutcome::error(failure);
  auto context = make_context(first, second);
  const std::array storage{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 0}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .context = &context, .default_storage = hyle::graph::StorageDomainId{0}}};
  auto compiled = hyle::graph::compile(
      logical.graph, {.storage_domains = storage, .targets = targets},
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(compiled);
  EXPECT_EQ(compiled.error(), failure);
  EXPECT_EQ(first.destroy_prepared_count, 1u);
  EXPECT_EQ(second.destroy_prepared_count, 1u);
}

TEST(GraphRegions, MissingLaneSelectsAnotherTargetWithoutWaivingPlacement) {
  const auto logical = make_negations(1);
  for (const bool required : {false, true}) {
    RegionProbe first{.mode = ProposalMode::kSingleton};
    RegionProbe second{.mode = ProposalMode::kSingleton};
    auto a = make_context(first), b = make_context(second);
    const std::array storage{
        hyle::graph::StorageDomain{hyle::mem::system_resource()}};
    const std::array targets{
        hyle::graph::ExecutionTarget{&a, hyle::graph::StorageDomainId{0}},
        hyle::graph::ExecutionTarget{&b, hyle::graph::StorageDomainId{0}}};
    const std::array lanes{
        hyle::graph::ExecutionLane{hyle::graph::ExecutionTargetId{1}}};
    const std::array placements{hyle::graph::NodePlacement{
        logical.graph.node_id(0).value(), hyle::graph::ExecutionTargetId{0},
        hyle::graph::PlacementStrength::kRequired}};
    CompileOptions options;
    options.operation_fallback = OperationFallbackPolicy::kReject;
    if (required)
      options.node_placements = placements;
    auto compiled = hyle::graph::compile(
        logical.graph,
        {.storage_domains = storage, .targets = targets, .lanes = lanes},
        options);
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    if (required) {
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsInvalidArgument());
      EXPECT_EQ(second.compile_count, 0u);
      continue;
    }
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(second.compile_count, 1u);
    EXPECT_EQ(compiled->report.search().route_plan_rejections, 1u);
    EXPECT_EQ(compiled->report.step(0)->target,
              hyle::graph::ExecutionTargetId{1});
    auto input = make_tensor(row(8)), output = make_tensor(row(8));
    write(input, std::array{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F});
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(run.bind(logical.input, input.view).ok());
    ASSERT_TRUE(run.bind(logical.output, output.view).ok());
    auto queue = b.create_queue().value();
    ASSERT_TRUE(run.submit(queue).value().wait().ok());
    EXPECT_EQ(read(output, 8),
              (std::vector<float>{-1.F, 2.F, -3.F, 4.F, -5.F, 6.F, -7.F, 8.F}));
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
      .runtime =
          {
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

TEST(GraphCompilationStorage,
     RegionDiscoveryAndResourceScratchShareCompileQuota) {
  const auto logical = make_negations(2);
  for (bool resource_query : {false, true}) {
    SCOPED_TRACE(resource_query ? "resource query" : "discovery");
    RegionProbe probe;
    probe.compilation_storage = {.resident_bytes = 0, .peak_bytes = 0};
    auto context = make_context(probe);
    CompileOptions options{
        .compilation_storage = hyle::graph::CompilationStorageLimits{},
        .operation_fallback = OperationFallbackPolicy::kReject};
    auto plain = hyle::graph::compile(logical.graph, context, options).value();
    const auto base = plain.report.compilation_storage().value();
    plain = {};
    const auto scratch_bytes = base.peak_bytes * 2;
    if (resource_query)
      probe.resource_scratch_bytes = scratch_bytes;
    else
      probe.discovery_scratch_bytes = scratch_bytes;
    const auto prepared = probe.compile_count;
    options.compilation_storage->max_peak_bytes = base.peak_bytes;
    auto rejected = hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsResourceExhausted());
    EXPECT_EQ(rejected.error().message(),
              resource_query ? "region resource query scratch allocation failed"
                             : "region discovery scratch allocation failed");
    EXPECT_EQ(probe.compile_count, prepared);
    options.compilation_storage->max_peak_bytes = SIZE_MAX;
    auto measured = hyle::graph::compile(logical.graph, context, options);
    ASSERT_TRUE(measured) << measured.error().message();
    const auto usage = measured->report.compilation_storage().value();
    EXPECT_GE(usage.peak_bytes, scratch_bytes);
    EXPECT_EQ(usage.resident_bytes, base.resident_bytes);
    EXPECT_FALSE(usage.has_unknown_storage);
    *measured = {};
    options.compilation_storage->max_resident_bytes = usage.resident_bytes;
    options.compilation_storage->max_peak_bytes = usage.peak_bytes;
    auto bounded = hyle::graph::compile(logical.graph, context, options);
    ASSERT_TRUE(bounded) << bounded.error().message();
    EXPECT_EQ(bounded->report.compilation_storage()->peak_bytes,
              usage.peak_bytes);
  }
}

TEST(GraphCompilationStorage,
     RegionUnknownAndInvalidBoundsPrecedeArtifactCreation) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  auto context = make_context(probe);
  CompileOptions options{
      .compilation_storage =
          hyle::graph::CompilationStorageLimits{.max_resident_bytes = 1U << 20},
      .operation_fallback = OperationFallbackPolicy::kReject};
  auto unknown = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(unknown);
  EXPECT_EQ(probe.compile_count, 0U);
  EXPECT_GT(probe.region_resource_count, 0U);
  options.compilation_storage->unknown =
      hyle::graph::UnknownStoragePolicy::kAllow;
  auto partial = hyle::graph::compile(logical.graph, context, options);
  ASSERT_TRUE(partial) << partial.error().message();
  EXPECT_TRUE(partial->report.compilation_storage()->has_unknown_storage);
  *partial = {};
  const auto count = probe.compile_count;
  probe.compilation_storage = {.resident_bytes = 2, .peak_bytes = 1};
  auto invalid = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInternal());
  EXPECT_EQ(probe.compile_count, count);
}

TEST(GraphCompilationStorage, RegionPreparePeakIsTransientNotRetainedPerStep) {
  const auto logical = make_negations(4);
  RegionProbe probe;
  probe.compilation_storage = {.resident_bytes = 0, .peak_bytes = 0};
  auto context = make_context(probe);
  CompileOptions options{
      .compilation_storage = hyle::graph::CompilationStorageLimits{},
      .operation_fallback = OperationFallbackPolicy::kReject};
  auto plain = hyle::graph::compile(logical.graph, context, options).value();
  ASSERT_EQ(plain.report.region_count(), 2U);
  const auto base = plain.report.compilation_storage().value();
  plain = {};
  const auto private_peak = base.peak_bytes * 2;
  probe.compilation_storage = {.resident_bytes = 4096,
                               .peak_bytes = private_peak};
  options.compilation_storage->max_peak_bytes =
      base.peak_bytes + private_peak + 8192;
  auto charged = hyle::graph::compile(logical.graph, context, options);
  ASSERT_TRUE(charged) << charged.error().message();
  const auto usage = charged->report.compilation_storage().value();
  EXPECT_EQ(usage.resident_bytes, base.resident_bytes + 8192);
  EXPECT_GE(usage.peak_bytes, private_peak);
  EXPECT_LT(usage.peak_bytes, 2 * private_peak);
  EXPECT_FALSE(usage.has_unknown_storage);
}

TEST(GraphCompilationStorage,
     RegionDeclineCleansPrivateChargesBeforeReselection) {
  const auto logical = make_negations(2);
  RegionProbe first, second;
  first.compilation_storage = {.resident_bytes = 1U << 18,
                               .peak_bytes = 1U << 19};
  first.region_outcome_override = ProviderPrepareOutcome::resource_declined(
      Status::ResourceExhausted("partial region declined"));
  second.compilation_storage = {.resident_bytes = 0, .peak_bytes = 0};
  second.observe_cleanup = &first;
  auto context = make_context(first, second);
  CompileOptions options{
      .compilation_storage = hyle::graph::CompilationStorageLimits{},
      .operation_fallback = OperationFallbackPolicy::kReject};
  auto measured = hyle::graph::compile(logical.graph, context, options).value();
  const auto usage = measured.report.compilation_storage().value();
  measured = {};
  const auto attempts = first.compile_count;
  const auto destroyed = first.destroy_prepared_count;
  options.compilation_storage->max_resident_bytes = usage.resident_bytes;
  options.compilation_storage->max_peak_bytes = usage.peak_bytes;
  auto compiled = hyle::graph::compile(logical.graph, context, options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.compile_count, attempts + 1);
  EXPECT_EQ(first.destroy_prepared_count, destroyed + 1);
  EXPECT_EQ(second.observed_cleanup_at_compile, destroyed + 1);
  EXPECT_GE(compiled->report.compilation_storage()->peak_bytes, 1U << 19);
  EXPECT_LT(compiled->report.compilation_storage()->resident_bytes, 1U << 17);
}

TEST(GraphStorage, UnknownProviderStorageRequiresExplicitFiniteBudgetOptIn) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  auto context = make_context(probe);
  CompileOptions options{.compiled_storage = {.max_bytes = 0},
                         .operation_fallback =
                             OperationFallbackPolicy::kReject};
  auto rejected = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsFailedPrecondition())
      << rejected.error().message();
  EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
  options.compiled_storage.unknown = hyle::graph::UnknownStoragePolicy::kAllow;
  auto partial = hyle::graph::compile(logical.graph, context, options);
  ASSERT_TRUE(partial) << partial.error().message();
  const auto report = partial->report.compiled_storage();
  EXPECT_EQ(report.known_bytes(), 0U);
  EXPECT_EQ(report.unknown_provider_count, 1U);
  EXPECT_FALSE(report.total_bytes());
  EXPECT_FALSE(partial->report.total_compiled_storage().total_bytes());
  // A default unlimited compile is not a finite-budget promise.
  auto unlimited = hyle::graph::compile(logical.graph, context).value();
  EXPECT_FALSE(unlimited.report.compiled_storage().total_bytes());
}

TEST(GraphStorage, StorageLimitCanReselectACompleteRegionBeforePublishing) {
  const auto logical = make_negations(2);
  for (bool unknown : {false, true}) {
    RegionProbe first, second;
    first.artifact_properties.persistent_bytes =
        unknown ? std::nullopt : std::optional<std::size_t>{129};
    second.artifact_properties.persistent_bytes = 128;
    second.observe_cleanup = &first;
    auto context = make_context(first, second);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.compiled_storage = {.max_bytes = 128},
         .operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(first.compile_count, 1U);
    EXPECT_EQ(first.destroy_prepared_count, 1U);
    EXPECT_EQ(second.observed_cleanup_at_compile, 1U);
    EXPECT_EQ(first.propose_count, 1U);
    EXPECT_EQ(second.propose_count, 1U);
    EXPECT_EQ(compiled->report.compiled_storage().total_bytes(), 128U);
    EXPECT_EQ(compiled->report.search().route_plan_rejections, 1U);
    EXPECT_EQ(compiled->report.step(0)->provider_instance,
              context.provider_instance_id(1).value());
  }
}

TEST(GraphStorage, MatureLeafFallbackCannotTurnUnknownStorageIntoZero) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  probe.artifact_properties.persistent_bytes = SIZE_MAX;
  auto context = make_context(probe);
  auto baseline = hyle::graph::compile(
                      logical.graph, context,
                      {.region_selection = RegionSelectionPolicy::kDisabled})
                      .value();
  const auto required = baseline.report.compiled_storage().total_bytes();
  EXPECT_FALSE(required);
  const auto known = baseline.report.compiled_storage().known_bytes();
  auto strict = hyle::graph::compile(
      logical.graph, context, {.compiled_storage = {.max_bytes = known}});
  ASSERT_FALSE(strict);
  EXPECT_TRUE(strict.error().IsFailedPrecondition() ||
              strict.error().IsResourceExhausted());
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.compiled_storage = {.max_bytes = known,
                            .unknown = hyle::graph::UnknownStoragePolicy::kAllow}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 0U);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 2U);
  EXPECT_FALSE(compiled->report.compiled_storage().total_bytes());
  EXPECT_EQ(compiled->report.compiled_storage().known_bytes(), known);
  EXPECT_EQ(probe.destroy_prepared_count, probe.compile_count);
  auto failed = hyle::graph::compile(
      logical.graph, context,
      {.compiled_storage = {.max_bytes = 0},
       .search_policy = CompileSearchPolicy::kAllowLimited});
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted() ||
              failed.error().IsFailedPrecondition())
      << failed.error().message();
}

TEST(GraphStorage, CompiledStorageOverflowRollsBackInsteadOfSaturating) {
  const auto logical = make_negations(4);
  RegionProbe probe;
  probe.artifact_properties.persistent_bytes = SIZE_MAX;
  auto context = make_context(probe);
  auto failed = hyle::graph::compile(
      logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted()) << failed.error().message();
  EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
  EXPECT_GT(probe.compile_count, 0U);
  probe.artifact_properties.persistent_bytes = 0;
  auto zero = hyle::graph::compile(
      logical.graph, context,
      {.compiled_storage = {.max_bytes = 0},
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(zero) << zero.error().message();
  EXPECT_EQ(zero->report.compiled_storage().total_bytes(), 0U);
}

TEST(GraphStorage,
     PackedConstantsAreAdditionalToCanonicalAndPersistentStorage) {
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(8)).value();
  const auto middle = builder.add(Unary{UnaryKind::kNeg}, constant).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, middle).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto tensor = make_tensor(row(8));
  const std::array bindings{
      hyle::graph::ConstantBinding{constant, tensor.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  RegionProbe probe;
  probe.artifact_properties.persistent_bytes = 128;
  probe.artifact_properties.packed_constant_count = 1;
  probe.artifact_properties.packed_constant_bytes = 64;
  auto context = make_context(probe);
  CompileOptions options{.constants = bindings,
                         .compiled_storage = {.max_bytes = 224},
                         .operation_fallback =
                             OperationFallbackPolicy::kReject};
  auto compiled = hyle::graph::compile(graph, context, options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto report = compiled->report.compiled_storage();
  EXPECT_EQ(report.materialized_constant_bytes, 32U);
  EXPECT_EQ(report.provider_persistent_bytes, 128U);
  EXPECT_EQ(report.packed_constant_bytes, 64U);
  EXPECT_EQ(report.total_bytes(), 224U);
  --options.compiled_storage.max_bytes;
  auto over = hyle::graph::compile(graph, context, options);
  ASSERT_FALSE(over);
  EXPECT_TRUE(over.error().IsResourceExhausted()) << over.error().message();
  probe.artifact_properties.persistent_bytes.reset();
  options.compiled_storage.max_bytes = 96;
  options.compiled_storage.unknown = hyle::graph::UnknownStoragePolicy::kAllow;
  auto partial = hyle::graph::compile(graph, context, options);
  ASSERT_TRUE(partial) << partial.error().message();
  EXPECT_EQ(partial->report.compiled_storage().known_bytes(), 96U);
  EXPECT_FALSE(partial->report.compiled_storage().total_bytes());
  --options.compiled_storage.max_bytes;
  EXPECT_FALSE(hyle::graph::compile(graph, context, options));
}

TEST(GraphStorage, UnknownStoragePolicyIsValidatedBeforeProviderDiscovery) {
  RegionProbe probe;
  auto context = make_context(probe);
  auto logical = make_negations(2);
  auto result = hyle::graph::compile(
      logical.graph, context,
      {.compiled_storage = {
           .unknown = static_cast<hyle::graph::UnknownStoragePolicy>(99)}});
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsInvalidArgument());
  EXPECT_EQ(probe.propose_count, 0U);
  EXPECT_EQ(probe.compile_count, 0U);
}

TEST(GraphStorage, EveryRetainedVariantSpendsTheSameStorageBudget) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  probe.artifact_properties.persistent_bytes = 128;
  auto context = make_context(probe);
  const std::array<hyle::graph::ExactShapeVariant, 1> variants{};
  CompileOptions options{
      .shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain,
      .exact_variants = variants,
      .compiled_storage = {.max_bytes = 256},
      .operation_fallback = OperationFallbackPolicy::kReject};
  {
    auto compiled = hyle::graph::compile(logical.graph, context, options);
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.compiled_storage().total_bytes(), 128U);
    EXPECT_EQ(
        compiled->report.exact_variant(0)->compiled_storage().total_bytes(),
        128U);
    EXPECT_EQ(compiled->report.total_compiled_storage().total_bytes(), 256U);
  }
  --options.compiled_storage.max_bytes;
  auto failed = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted()) << failed.error().message();
  EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
  // The complete family sum must be representable even with no finite cap.
  probe.artifact_properties.persistent_bytes = SIZE_MAX;
  options.compiled_storage.max_bytes = SIZE_MAX;
  failed = hyle::graph::compile(logical.graph, context, options);
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted()) << failed.error().message();
  EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
  probe.artifact_properties.persistent_bytes.reset();
  options.compiled_storage.max_bytes = 0;
  options.compiled_storage.unknown = hyle::graph::UnknownStoragePolicy::kAllow;
  auto partial = hyle::graph::compile(logical.graph, context, options);
  ASSERT_TRUE(partial) << partial.error().message();
  EXPECT_EQ(partial->report.total_compiled_storage().unknown_provider_count,
            2U);
  EXPECT_FALSE(partial->report.total_compiled_storage().total_bytes());
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

TEST(GraphRegions, FiniteVariantsShareRegionDiscoveryAndNodeBudgets) {
  for (bool node_budget : {false, true}) {
    RegionProbe probe{.mode = ProposalMode::kSingleton};
    auto context = make_context(probe);
    auto builder = GraphBuilder::create().value();
    const auto rows = builder.dimension({1, 3, 1}).value();
    const auto fixed_input = builder.input(row(8)).value();
    const auto fixed_output =
        builder.add(Unary{UnaryKind::kNeg}, fixed_input).value();
    const auto spec =
        hyle::graph::TensorSpec::dense(
            DataType::kFloat32, std::array{hyle::graph::Dimension(rows),
                                           hyle::graph::Dimension::fixed(8)})
            .value();
    const auto input = builder.input(spec).value();
    const auto output =
        builder.add(hyle::compute::RmsNorm{}, spec, input).value();
    ASSERT_TRUE(builder.publish(fixed_output).ok());
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    const std::array point{hyle::graph::ShapeBinding{rows, 1}};
    const std::array variants{hyle::graph::ExactShapeVariant{point}};
    CompileOptions options{.shape_policy =
                               hyle::graph::CompileShapePolicy::kDeclaredDomain,
                           .exact_variants = variants};
    if (node_budget)
      options.max_region_proposal_nodes = 1;
    else
      options.max_region_proposals = 1;
    auto compiled = hyle::graph::compile(graph, context, options);
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.region_count(), 1U);
    const auto child = compiled->report.exact_variant(0).value();
    EXPECT_EQ(child.region_count(), 0U);
    EXPECT_TRUE(child.search().optimization_limited());
    EXPECT_EQ(child.search().region_node_limit_reached, node_budget);
    EXPECT_EQ(child.search().region_proposal_limit_reached, !node_budget);
    EXPECT_EQ(compiled->report.total_search().region_proposals, 1U);
    EXPECT_EQ(compiled->report.total_search().region_proposal_nodes, 1U);
    EXPECT_EQ(probe.propose_count, 1U);
    EXPECT_EQ(probe.compile_count, 1U);
    auto run = compiled->executable.create_invocation(point).value();
    auto fixed_in = make_tensor(row(8)), fixed_out = make_tensor(row(8));
    const auto actual_desc =
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{1, 8})
            .value();
    auto in = make_tensor(actual_desc), out = make_tensor(actual_desc);
    write(fixed_in, std::vector<float>(8, 2.0F));
    write(in, std::vector<float>(8, 1.0F));
    ASSERT_TRUE(run.bind(fixed_input, fixed_in.view).ok());
    ASSERT_TRUE(run.bind(fixed_output, fixed_out.view).ok());
    ASSERT_TRUE(run.bind(input, in.view).ok());
    ASSERT_TRUE(run.bind(output, out.view).ok());
    auto queue = context.create_queue().value();
    ASSERT_TRUE(run.submit(queue)->wait().ok());
    EXPECT_EQ(read(fixed_out, 8), std::vector<float>(8, -2.0F));
    for (float value : read(out, 8))
      EXPECT_NEAR(value, 1.0F, 1.0e-5F);
    EXPECT_EQ(probe.propose_count, 1U);
    EXPECT_EQ(probe.compile_count, 1U);
    options.search_policy = CompileSearchPolicy::kRequireUnexhausted;
    auto strict = hyle::graph::compile(graph, context, options);
    ASSERT_FALSE(strict);
    EXPECT_TRUE(strict.error().IsResourceExhausted())
        << strict.error().message();
  }
}

TEST(GraphRegions, ExactOnlyRegionCannotClaimDynamicEnvelope) {
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      hyle::graph::TensorSpec::dense(
          DataType::kFloat32, std::array{hyle::graph::Dimension(rows),
                                         hyle::graph::Dimension::fixed(8)})
          .value();
  const auto input = builder.input(spec).value();
  const auto output =
      builder.add(hyle::compute::RmsNorm{}, spec, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(
      graph, context,
      {.shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(probe.propose_count, 1U);
  EXPECT_EQ(probe.region_resource_count, 0U);
  EXPECT_EQ(probe.compile_count, 0U);
  EXPECT_EQ(compiled->report.region_count(), 0U);
  const std::array shape{hyle::graph::ShapeBinding{rows, 2}};
  auto run = compiled->executable.create_invocation(shape);
  ASSERT_TRUE(run.has_value()) << run.error().message();
  // None of the cold discovery/compilation callbacks are revisited.
  EXPECT_EQ(probe.propose_count, 1U);
  EXPECT_EQ(probe.region_resource_count, 0U);
  EXPECT_EQ(probe.compile_count, 0U);
}

TEST(GraphRegions, DomainInvocationsShareTheOriginalNonconcurrentArtifactGate) {
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({1, 4, 1}).value();
  const auto spec =
      hyle::graph::TensorSpec::dense(
          DataType::kFloat32, std::array{hyle::graph::Dimension(rows),
                                         hyle::graph::Dimension::fixed(8)})
          .value();
  const auto dynamic_input = builder.input(spec).value();
  const auto input = builder.input(row(8)).value();
  const auto middle = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, middle).value();
  ASSERT_TRUE(builder.publish(dynamic_input).ok());
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.artifact_properties.runtime.supports_concurrent_invocations = false;
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(
      graph, context,
      {.shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1U);
  EXPECT_FALSE(compiled->executable.supports_concurrent_invocations());
  const std::array small{hyle::graph::ShapeBinding{rows, 1}};
  const std::array large{hyle::graph::ShapeBinding{rows, 4}};
  auto first = compiled->executable.create_invocation(small).value();
  auto second = compiled->executable.create_invocation(large).value();
  EXPECT_NE(first.specialization(), second.specialization());
  auto small_input = make_tensor(
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{1, 8})
          .value());
  auto large_input = make_tensor(
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{4, 8})
          .value());
  auto first_input = make_tensor(row(8)), first_output = make_tensor(row(8));
  auto second_input = make_tensor(row(8)), second_output = make_tensor(row(8));
  write(first_input, std::array{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F});
  write(second_input, std::array{2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F});
  ASSERT_TRUE(first.bind(dynamic_input, small_input.view).ok());
  ASSERT_TRUE(second.bind(dynamic_input, large_input.view).ok());
  ASSERT_TRUE(first.bind(input, first_input.view).ok());
  ASSERT_TRUE(second.bind(input, second_input.view).ok());
  ASSERT_TRUE(first.bind(output, first_output.view).ok());
  ASSERT_TRUE(second.bind(output, second_output.view).ok());
  auto queue = context.create_queue().value();
  auto pending = first.submit(queue);
  ASSERT_TRUE(pending.has_value()) << pending.error().message();
  EXPECT_TRUE(first.in_flight());
  const auto submissions = probe.submit_count;
  auto rejected = second.submit(queue);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(probe.submit_count, submissions);
  ASSERT_TRUE(pending->wait().ok());
  EXPECT_EQ(first.output(dynamic_input)->description(),
            small_input.view.description());
  ASSERT_TRUE(second.submit(queue)->wait().ok());
  EXPECT_EQ(second.output(dynamic_input)->description(),
            large_input.view.description());
  EXPECT_EQ(probe.compile_count, 1U);
}

TEST(GraphRegions, RegionExtensionVersionDoesNotAliasTheV1Table) {
  constexpr hyle::compute::ProviderExtensionId version_one{
      0x48594c4547524150ULL, 0x48524547494f4e31ULL, 1};
  constexpr hyle::compute::ProviderExtensionId version_two{
      0x48594c4547524150ULL, 0x48524547494f4e31ULL, 2};
  EXPECT_EQ(query_extension(nullptr, version_one), nullptr);
  EXPECT_EQ(query_extension(nullptr, version_two), nullptr);
  auto previous = hyle::graph::kRegionProviderExtensionId;
  --previous.version;
  EXPECT_EQ(query_extension(nullptr, previous), nullptr);
  EXPECT_EQ(query_extension(nullptr, hyle::graph::kRegionProviderExtensionId),
            &kRegionVTable);
}

TEST(GraphRegions, ReselectionReportsTheStableKeyNotThePrepareToken) {
  const auto logical = make_negations(1);
  for (const auto key : {hyle::compute::ProviderImplementationId{},
                         hyle::compute::ProviderImplementationId{91}}) {
    RegionProbe first{.mode = ProposalMode::kSingleton},
        second{.mode = ProposalMode::kSingleton};
    first.implementation = hyle::compute::ProviderImplementationId{41};
    first.region_outcome_override = ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("first region declined"));
    second.implementation = key;
    auto context = make_context(first, second);
    auto compiled = hyle::graph::compile(
        logical.graph, context,
        {.operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.step(0)->implementation, key);
    EXPECT_EQ(compiled->report.step(0)->provider_instance,
              context.provider_instance_id(1).value());
    EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(first.destroy_prepared_count, 1u);
    EXPECT_EQ(second.compile_count, 1u);
  }
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
    probe.required_resource = hyle::mem::system_resource();
    probe.region_outcome_override = decline;
    Context context = make_context(probe);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(probe.compile_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
    EXPECT_EQ(probe.destroy_count_at_single_preparation, 1u);
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
  for (const Status &error :
       {Status::Unimplemented("fatal region compile failure"),
        Status::ResourceExhausted("real region allocation failure")}) {
    RegionProbe probe;
    probe.region_outcome_override = ProviderPrepareOutcome::error(error);
    RegionProbe alternative;
    Context context = make_context(probe, alternative);
    auto compiled = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(compiled.has_value());
    EXPECT_EQ(compiled.error(), error);
    EXPECT_EQ(probe.compile_count, 1u);
    EXPECT_EQ(probe.destroy_prepared_count, 1u);
    EXPECT_EQ(probe.single_proposal_count, 2u);
    EXPECT_EQ(probe.single_artifact_count, 0u);
    EXPECT_EQ(alternative.compile_count, 0u);
  }
}

TEST(GraphRegions, RetriesAnotherRegionProviderWithoutRediscovery) {
  const auto logical = make_negations(2);
  const std::array declines{
      ProviderPrepareOutcome::unsupported(
          Status::Unimplemented("unsupported route")),
      ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("candidate-local budget"))};
  for (const auto &decline : declines) {
    for (auto fallback :
         {OperationFallbackPolicy::kAllow, OperationFallbackPolicy::kReject}) {
      RegionProbe first;
      first.region_outcome_override = decline;
      RegionProbe second;
      second.observe_cleanup = &first;
      Context context = make_context(first, second);
      auto compiled = hyle::graph::compile(
          logical.graph, context,
          {.max_placement_attempts = 2,
           .operation_fallback = fallback,
           .max_region_proposals = 2,
           .max_region_proposals_per_provider = 1,
           .max_region_proposal_nodes = 4,
           .search_policy = CompileSearchPolicy::kRequireUnexhausted});
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      EXPECT_EQ(first.propose_count, 1u);
      EXPECT_EQ(second.propose_count, 1u);
      EXPECT_EQ(first.compile_count, 1u);
      EXPECT_EQ(second.compile_count, 1u);
      EXPECT_EQ(second.observed_cleanup_at_compile, 1u);
      EXPECT_EQ(first.single_proposal_count + second.single_proposal_count,
                fallback == OperationFallbackPolicy::kAllow ? 4u : 0u);
      EXPECT_EQ(first.single_artifact_count + second.single_artifact_count, 0u);
      EXPECT_EQ(compiled->report.region_count(), 1u);
      EXPECT_EQ(compiled->report.fallback_operation_count(), 0u);
      EXPECT_EQ(compiled->report.step(0)->provider_instance,
                context.provider_instance_id(1).value());
      const auto search = compiled->report.search();
      EXPECT_EQ(search.region_proposals, 2u);
      EXPECT_EQ(search.region_proposal_nodes, 4u);
      EXPECT_EQ(search.placement_attempts, 2u);
      EXPECT_EQ(search.region_route_rejections, 1u);
      EXPECT_FALSE(search.optimization_limited());

      auto input = make_tensor(row(8));
      auto output = make_tensor(row(8));
      const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
      write(input, values);
      auto invocation = compiled->executable.create_invocation().value();
      ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
      ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
      auto queue = context.create_queue().value();
      ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
      EXPECT_EQ(read(output, values.size()),
                std::vector<float>(values.begin(), values.end()));
      EXPECT_EQ(first.submit_count, 0u);
      EXPECT_EQ(second.submit_count, 1u);
    }
  }
}

TEST(GraphRegions, RepartitionsAfterOneRouteOfTheSameProviderDeclines) {
  const auto logical = make_negations(3);
  RegionProbe probe{.mode = ProposalMode::kOverlappingRoutes};
  probe.region_outcome_route = 2;
  probe.region_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("declined prefix fusion"));
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(probe.propose_count, 1u);
  EXPECT_EQ(probe.compile_count, 2u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
  EXPECT_EQ(probe.compiled_begin, 1u);
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 1u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
  EXPECT_EQ(compiled->report.step(0)->fallback_reason,
            CompileFallbackReason::kRegionDeclined);

  auto input = make_tensor(row(8));
  auto output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
  std::vector<float> expected;
  for (float value : values)
    expected.push_back(-value);
  EXPECT_EQ(read(output, values.size()), expected);
}

TEST(GraphRegions, RetryDestroysEarlierArtifactsBeforeRebuildingThePartition) {
  const auto logical = make_negations(4);
  RegionProbe first;
  first.region_outcome_route = (std::uint64_t{2} << 32) | 2;
  first.region_outcome_override = ProviderPrepareOutcome::resource_declined(
      Status::ResourceExhausted("second region exceeds private budget"));
  RegionProbe second;
  second.observe_cleanup = &first;
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.max_placement_attempts = 4,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(first.propose_count, 1u);
  EXPECT_EQ(second.propose_count, 1u);
  EXPECT_EQ(first.region_resource_count, 2u);
  EXPECT_EQ(second.region_resource_count, 2u);
  EXPECT_EQ(first.compile_count, 3u);
  EXPECT_EQ(second.compile_count, 1u);
  // Both the rejected partial artifact and the earlier successful artifact
  // of the abandoned partition are gone, before the alternative is prepared.
  EXPECT_EQ(second.observed_cleanup_at_compile, 2u);
  EXPECT_EQ(compiled->report.region_count(), 2u);
  EXPECT_EQ(compiled->report.step_count(), 2u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 4u);
  EXPECT_EQ(compiled->report.search().region_proposals, 4u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);

  auto input = make_tensor(row(8));
  auto output = make_tensor(row(8));
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

TEST(GraphRegions, RegionRetryCanChangeTargetButHonorsRequiredPlacement) {
  const auto logical = make_negations(2);
  RegionProbe first;
  first.region_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("first target declines"));
  RegionProbe second;
  Context first_context = make_context(first);
  Context second_context = make_context(second);
  const auto resource = hyle::mem::system_resource();
  const Environment environment{{&first_context, resource},
                                {&second_context, resource}};
  auto compiled = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
  EXPECT_EQ(second.compile_count, 1u);

  const std::array required{
      hyle::graph::NodePlacement{logical.graph.node_id(0).value(),
                                 environment.target(0),
                                 hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{logical.graph.node_id(1).value(),
                                 environment.target(0),
                                 hyle::graph::PlacementStrength::kRequired}};
  auto unavailable = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.node_placements = required,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(unavailable.has_value());
  EXPECT_EQ(unavailable.error(), first.region_outcome_override->status);
  EXPECT_EQ(second.compile_count, 1u);
  EXPECT_EQ(first.single_proposal_count + second.single_proposal_count, 0u);
}

TEST(GraphRegions, RegionRetriesDoNotRechargePreparationBudget) {
  const auto logical = make_negations(2);
  RegionProbe first;
  first.region_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("first route declines"));
  RegionProbe second;
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.max_placement_attempts = 1,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(first.propose_count, 1u);
  EXPECT_EQ(second.propose_count, 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
  EXPECT_EQ(first.destroy_prepared_count, 1u);
}

TEST(GraphRegions, AllRejectedRegionsTerminateWithoutRediscoveryOrLeafWork) {
  const auto logical = make_negations(2);
  RegionProbe first;
  first.region_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("first route declines"));
  RegionProbe second;
  second.region_outcome_override = ProviderPrepareOutcome::resource_declined(
      Status::ResourceExhausted("second route declines"));
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.max_placement_attempts = 2,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_EQ(compiled.error(), second.region_outcome_override->status);
  EXPECT_EQ(first.propose_count, 1u);
  EXPECT_EQ(second.propose_count, 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 1u);
  EXPECT_EQ(first.destroy_prepared_count, 1u);
  EXPECT_EQ(second.destroy_prepared_count, 1u);
  EXPECT_EQ(first.single_proposal_count + second.single_proposal_count, 0u);
}

TEST(GraphRegions, UnmatchedRegionStorageCanSelectAnotherCompleteRegion) {
  const auto logical = make_negations(2);
  HostResource unavailable;
  RegionProbe first;
  first.region_input_resource = hyle::mem::MemoryResourceRef(unavailable);
  RegionProbe second;
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(first.propose_count, 1u);
  EXPECT_EQ(second.propose_count, 1u);
  EXPECT_EQ(first.region_resource_count, 1u);
  EXPECT_EQ(first.compile_count, 0u);
  EXPECT_EQ(second.compile_count, 1u);
  EXPECT_EQ(compiled->report.region_count(), 1u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 1u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
}

TEST(GraphRegions, EveryAllocationFailureInRegionReselectionRollsBack) {
  for (const bool final_workspace_rejection : {false, true}) {
    SCOPED_TRACE(final_workspace_rejection);
    struct AllocationState final {
      AllocatorRef backing = hyle::mem::system_allocator();
      std::size_t attempts = 0;
      std::size_t fail_at = SIZE_MAX;
      std::size_t live = 0;
    } state;
    const AllocatorRef allocator{
        +[](void *opaque, std::size_t bytes,
            std::size_t alignment) noexcept -> void * {
          auto &state = *static_cast<AllocationState *>(opaque);
          if (state.attempts++ == state.fail_at)
            return nullptr;
          void *result = state.backing.allocate(bytes, alignment);
          state.live += result != nullptr ? 1u : 0u;
          return result;
        },
        +[](void *opaque, void *pointer, std::size_t bytes,
            std::size_t alignment) noexcept {
          auto &state = *static_cast<AllocationState *>(opaque);
          state.live -= pointer != nullptr ? 1u : 0u;
          state.backing.deallocate(pointer, bytes, alignment);
        },
        +[](void *opaque, void *pointer, std::size_t old_bytes,
            std::size_t new_bytes, std::size_t alignment) noexcept -> void * {
          auto &state = *static_cast<AllocationState *>(opaque);
          if (new_bytes == 0) {
            state.live -= pointer != nullptr ? 1u : 0u;
            return state.backing.reallocate(pointer, old_bytes, 0, alignment);
          }
          if (state.attempts++ == state.fail_at)
            return nullptr;
          void *result = state.backing.reallocate(pointer, old_bytes, new_bytes,
                                                  alignment);
          if (pointer == nullptr && result != nullptr)
            ++state.live;
          return result;
        },
        &state};
    RegionProbe first;
    first.region_outcome_route = (std::uint64_t{2} << 32) | 2;
    first.region_outcome_override = ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("reselect after the second region"));
    if (final_workspace_rejection) {
      first.region_outcome_override.reset();
      first.artifact_properties.workspace = {.size = 128, .alignment = 64};
      use_delegated_runtime(first);
    }
    RegionProbe second;
    const std::array providers{
        hyle::compute::ProviderBinding{region_provider(first), 100},
        hyle::compute::ProviderBinding{region_provider(second), 200}};
    auto context = Context::create(hyle::compute::cpu::device(),
                                   {.metadata_allocator = allocator,
                                    .providers = providers})
                       .value();
    // Cross all inline capacities in the cold region boundary/resource cache,
    // including flat port requirements, before injecting each allocation
    // failure.
    const auto logical = make_negations(final_workspace_rejection ? 2 : 18);
    const std::array storage{hyle::graph::StorageDomain{
        .resource = hyle::mem::system_resource(),
        .workspace_budget_bytes = final_workspace_rejection ? 0u : SIZE_MAX}};
    const std::array targets{hyle::graph::ExecutionTarget{
        .context = &context,
        .default_storage = hyle::graph::StorageDomainId{0}}};
    const hyle::graph::CompileEnvironment environment{
        .storage_domains = storage, .targets = targets};
    const CompileOptions options{.operation_fallback =
                                     OperationFallbackPolicy::kReject};
    const std::size_t context_allocations = state.live;
    state.attempts = 0;
    {
      auto compiled = hyle::graph::compile(logical.graph, environment, options);
      ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
      EXPECT_EQ(compiled->report.search().region_route_rejections,
                final_workspace_rejection ? 0u : 1u);
      EXPECT_EQ(compiled->report.search().route_plan_rejections,
                final_workspace_rejection ? 1u : 0u);
    }
    const std::size_t attempts = state.attempts;
    ASSERT_GT(attempts, 0u);
    EXPECT_EQ(state.live, context_allocations);
    for (std::size_t failure = 0; failure < attempts; ++failure) {
      SCOPED_TRACE(failure);
      state.attempts = 0;
      state.fail_at = failure;
      {
        auto failed = hyle::graph::compile(logical.graph, environment, options);
        ASSERT_FALSE(failed.has_value());
        EXPECT_TRUE(failed.error().IsResourceExhausted());
      }
      EXPECT_EQ(state.live, context_allocations);
      state.fail_at = SIZE_MAX;
      {
        auto recovered =
            hyle::graph::compile(logical.graph, environment, options);
        ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
      }
      EXPECT_EQ(state.live, context_allocations);
    }
    context = {};
    EXPECT_EQ(state.live, 0u);
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
    EXPECT_EQ(probe.single_proposal_count, 2u);
    EXPECT_EQ(probe.single_artifact_count, 0u);
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
    HostResource first_storage, second_storage;
    RegionProbe probe;
    probe.required_resource = hyle::mem::MemoryResourceRef(first_storage);
    probe.single_outcome_override = decline;
    const std::array providers{
        hyle::compute::ProviderBinding{region_provider(probe), 100}};
    auto first =
        Context::create(hyle::compute::cpu::device(), {.providers = providers})
            .value();
    auto second = Context::create(hyle::compute::cpu::device(),
                                  hyle::compute::ggml_cpu::provider())
                      .value();
    const Environment environment{
        {&first, probe.required_resource},
        {&second, hyle::mem::MemoryResourceRef(second_storage)}};
    const std::array external{hyle::graph::ExternalResourceContract{
        logical.output, environment.storage(1)}};
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.external_resources = external,
         .region_selection = RegionSelectionPolicy::kDisabled});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    // Two caller-workspace bindings are distinct legal prepare requests.
    // Changing only a later copy carrier must NOT prepare either one again.
    EXPECT_EQ(probe.single_artifact_count, 2u);
    EXPECT_EQ(probe.destroy_prepared_count, 2u);
    EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
    EXPECT_EQ(compiled->report.search().placement_attempts, 3u);
    EXPECT_EQ(compiled->report.search().operation_route_rejections, 2u);
    EXPECT_EQ(probe.single_proposal_count, 1u);
    EXPECT_EQ(compiled->report.step_count(), 1u);
    // The first route would copy its output from storage 0 to storage 1;
    // the replacement produces there directly. No obsolete copy survives.
    EXPECT_EQ(compiled->report.copy_bytes(), 0u);
    auto input = make_tensor(row(8), probe.required_resource);
    auto output =
        make_tensor(row(8), hyle::mem::MemoryResourceRef(second_storage));
    const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
    write(input, values);
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
    ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
    auto queue = second.create_queue().value();
    ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
    auto expected = std::vector<float>(values.begin(), values.end());
    for (auto &value : expected)
      value = -value;
    EXPECT_EQ(read(output, values.size()), expected);
  }
}

TEST(GraphRegions, LeafDeclineReleasesTheRegionPrefixBeforeReplanning) {
  const auto logical = make_negations(3);
  RegionProbe probe;
  probe.observe_cleanup = &probe;
  probe.required_resource = hyle::mem::system_resource();
  probe.single_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("declined tail leaf"));
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       {.max_placement_attempts = 4});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(probe.compile_count, 2u);
  EXPECT_EQ(probe.single_artifact_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 2u); // rejected leaf + old prefix
  EXPECT_EQ(probe.observed_cleanup_at_compile, 2u);
  EXPECT_EQ(probe.single_proposal_count, 3u);
  EXPECT_EQ(probe.region_resource_count, 1u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 4u);
  EXPECT_EQ(compiled->report.search().operation_route_rejections, 1u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 0u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
  auto expected = std::vector<float>(values.begin(), values.end());
  for (auto &value : expected)
    value = -value;
  EXPECT_EQ(read(output, values.size()), expected);
}

TEST(GraphRegions,
     RegionAndLeafDeclinesShareDiscoveryAndCumulativePrepareBudget) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  probe.required_resource = hyle::mem::system_resource();
  probe.region_outcome_override = ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("declined region"));
  probe.single_outcome_override = ProviderPrepareOutcome::resource_declined(
      Status::ResourceExhausted("declined leaf budget"));
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       {.max_placement_attempts = 6});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(probe.propose_count, 1u);
  EXPECT_EQ(probe.region_resource_count, 1u);
  EXPECT_EQ(probe.single_proposal_count, 2u);
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.single_artifact_count, 2u);
  EXPECT_EQ(compiled->report.search().placement_attempts, 6u);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1u);
  EXPECT_EQ(compiled->report.search().operation_route_rejections, 2u);
  EXPECT_EQ(compiled->report.region_count(), 0u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue).value().wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
  auto insufficient = hyle::graph::compile(logical.graph, context,
                                           {.max_placement_attempts = 5});
  ASSERT_FALSE(insufficient);
  EXPECT_TRUE(insufficient.error().IsResourceExhausted());
}

TEST(GraphRegions, LeafErrorDoesNotBecomeARetryAcrossProvidersOrTargets) {
  const auto logical = make_negations(1);
  for (const Status &error :
       {Status::Unimplemented("fatal leaf compile failure"),
        Status::ResourceExhausted("real leaf allocation failure")}) {
    RegionProbe probe;
    probe.required_resource = hyle::mem::system_resource();
    probe.single_outcome_override = ProviderPrepareOutcome::error(error);
    auto first = make_context(probe);
    auto second = Context::create(hyle::compute::cpu::device(),
                                  hyle::compute::ggml_cpu::provider())
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
    const std::size_t regions_prepared = 1;
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

TEST(GraphInvocationStorage, RegionBudgetSurvivesAnAbandonedCompletion) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.artifact_properties.invocation_private_bytes = 0;
  const auto logical = make_negations(4);
  auto context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          {.invocation_storage =
               hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 20}})
          .value();
  ASSERT_EQ(compiled.report.region_count(), 2U);
  auto invocation = compiled.executable.create_invocation().value();
  const auto bytes = compiled.executable.invocation_storage_usage()->live_bytes;
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  auto completion = invocation.submit(queue).value();
  invocation = {};
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, bytes);
  completion = {};
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, bytes);
  EXPECT_EQ(probe.destroy_invocation_count, 0U);
  probe.completed_prefix = probe.submit_count;
  probe.abandoned_lifetime.terminal(probe.abandoned_lifetime.state,
                                    Status::Ok());
  probe.abandoned_lifetime.release(probe.abandoned_lifetime.state);
  probe.abandoned_lifetime = {};
  EXPECT_EQ(probe.destroy_invocation_count, 2U);
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
}

TEST(GraphInvocationStorage, PartialRegionCreationReturnsTheSameOwnersCharge) {
  RegionProbe probe{.mode = ProposalMode::kInvocationFailure};
  probe.artifact_properties.invocation_private_bytes = 128;
  auto context = make_context(probe);
  auto logical = make_negations(2);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          {.invocation_storage =
               hyle::graph::InvocationStorageLimits{.max_bytes = 1U << 20}})
          .value();
  ASSERT_EQ(compiled.report.region_count(), 1U);
  auto failed = compiled.executable.create_invocation();
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsInternal());
  EXPECT_EQ(probe.create_invocation_count, 1U);
  EXPECT_EQ(probe.destroy_invocation_count, 1U);
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
  probe.mode = ProposalMode::kPairs;
  EXPECT_TRUE(compiled.executable.create_invocation());
  EXPECT_EQ(compiled.executable.invocation_storage_usage()->live_bytes, 0U);
}

TEST(GraphInvocationStorage, CaptureCannotSilentlyBypassAHardCompleteBudget) {
  for (bool allow_unknown : {false, true}) {
    RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
    probe.capture_supported = true;
    probe.artifact_properties.runtime.capture_compatible = true;
    probe.artifact_properties.invocation_private_bytes = 0;
    auto context = make_context(probe);
    auto logical = make_negations(2);
    auto compiled =
        hyle::graph::compile(
            logical.graph, context,
            {.invocation_storage =
                 hyle::graph::InvocationStorageLimits{
                     .max_bytes = 1U << 20,
                     .unknown =
                         allow_unknown
                             ? hyle::graph::UnknownStoragePolicy::kAllow
                             : hyle::graph::UnknownStoragePolicy::kReject}})
            .value();
    auto invocation = compiled.executable.create_invocation().value();
    auto input = make_tensor(row(8)), output = make_tensor(row(8));
    ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
    ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
    auto queue = context.create_queue().value();
    const std::array queues{hyle::graph::QueueBinding{
        .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}};
    auto result = invocation.prepare_replay(
        queues, {.mode = hyle::graph::ReplayMode::kRequireCapture});
    if (allow_unknown) {
      ASSERT_TRUE(result.ok()) << result.message();
      EXPECT_EQ(probe.capture_begin_count, 1U);
      EXPECT_TRUE(
          compiled.executable.invocation_storage_usage()->has_unknown_storage);
    } else {
      EXPECT_TRUE(result.IsResourceExhausted());
      EXPECT_EQ(probe.capture_begin_count, 0U);
      EXPECT_FALSE(
          compiled.executable.invocation_storage_usage()->has_unknown_storage);
      ASSERT_TRUE(
          invocation
              .prepare_replay(queues,
                              {.mode = hyle::graph::ReplayMode::kPreferCapture})
              .ok());
      EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kCommands);
      EXPECT_EQ(probe.capture_begin_count, 0U);
    }
  }
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
      hyle::graph::ExternalResourceContract{.value = first_input,
                                            .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{.value = second_input,
                                            .storage = environment.storage(1)},
      hyle::graph::ExternalResourceContract{.value = first_output,
                                            .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{.value = second_output,
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
      hyle::graph::ExternalResourceContract{.value = first_input,
                                            .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{.value = second_input,
                                            .storage = environment.storage(1)},
      hyle::graph::ExternalResourceContract{.value = first_output,
                                            .storage = environment.storage(0)},
      hyle::graph::ExternalResourceContract{.value = second_output,
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
  ASSERT_TRUE(compiled_result.has_value()) << compiled_result.error().message();
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
  auto compiled =
      hyle::graph::compile(logical.graph, environment.environment()).value();
  ASSERT_EQ(compiled.report.segment_count(), 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  auto invocation = compiled.executable.create_invocation().value();
  Queue queue = context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.lane = environment.lane(0), .queue = &queue},
      hyle::graph::QueueBinding{.lane = environment.lane(0), .queue = &queue},
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
  ASSERT_TRUE(compiled_result.has_value()) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  OwnedTensor input =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor output =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(second_storage));
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

  OwnedTensor input =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor output =
      make_tensor(row(8), hyle::mem::MemoryResourceRef(second_storage));
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

struct PackingGraph final {
  Graph graph;
  ValueId input, weight, output;
  hyle::graph::DimensionId rows;
};

PackingGraph make_packing_graph(bool dynamic = false, std::size_t nodes = 1) {
  auto builder = GraphBuilder::create().value();
  PackingGraph result;
  auto first = hyle::graph::Dimension::fixed(2);
  if (dynamic) {
    result.rows = builder.dimension({1, 3, 1}).value();
    first = hyle::graph::Dimension(result.rows);
  }
  const auto spec = hyle::graph::TensorSpec::dense(
                        DataType::kFloat32,
                        std::array{first, hyle::graph::Dimension::fixed(4)})
                        .value();
  result.input = builder.input(spec).value();
  result.weight = builder.constant(row(4)).value();
  result.output = result.input;
  for (std::size_t i = 0; i < nodes; ++i)
    result.output =
        builder
            .add(hyle::compute::RmsNorm{0}, spec, result.output, result.weight)
            .value();
  EXPECT_TRUE(builder.publish(result.output).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

void configure_packing(RegionProbe &probe, PackingProbe &packing) {
  probe.packing = &packing;
  probe.implementation = hyle::compute::ProviderImplementationId{901};
  probe.compilation_storage = {0, 0};
  probe.artifact_properties.persistent_bytes = sizeof(PackedRegion);
  probe.artifact_properties.invocation_private_bytes = 0;
}

void expect_packing_replay(const hyle::graph::Executable &executable,
                           const PackingGraph &graph, const Context &context,
                           std::size_t rows, float gamma_scale = 1) {
  const auto desc = TensorDesc::dense(DataType::kFloat32,
                                      std::array<std::int64_t, 2>{
                                          static_cast<std::int64_t>(rows), 4})
                        .value();
  auto input = make_tensor(desc), output = make_tensor(desc);
  write(input, std::vector<float>(rows * 4, 1));
  auto run = executable.create_invocation().value();
  ASSERT_TRUE(run.bind(graph.input, input.view).ok());
  ASSERT_TRUE(run.bind(graph.output, output.view).ok());
  auto queue = context.create_queue().value();
  auto completion = run.submit(queue);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  const auto values = read(output, rows * 4);
  for (std::size_t i = 0; i < values.size(); ++i)
    EXPECT_FLOAT_EQ(values[i], static_cast<float>(i % 4 + 1) * gamma_scale);
}

TEST(GraphPacking, ExactVariantsShareWeightsAndSurviveParentAndSource) {
  PackingProbe packing;
  RegionProbe probe;
  configure_packing(probe, packing);
  auto context = make_context(probe);
  auto graph = make_packing_graph(true);
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{graph.weight, source.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array shape_a{hyle::graph::ShapeBinding{graph.rows, 2}};
  const std::array shape_b{hyle::graph::ShapeBinding{graph.rows, 3}};
  const std::array variants{hyle::graph::ExactShapeVariant{shape_a},
                            hyle::graph::ExactShapeVariant{shape_b}};
  auto compiled = hyle::graph::compile(
      graph.graph, context,
      {.shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain,
       .exact_variants = variants,
       .constants = bindings,
       .compilation_storage = {{.max_resident_bytes = 1U << 20,
                                .max_peak_bytes = 1U << 20}},
       .max_region_proposals = 8,
       .max_region_proposals_per_provider = 4,
       .max_region_proposal_nodes = 16});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(),
            0U); // real bounded Native fallback
  EXPECT_EQ(compiled->report.exact_variant(0)->region_count(), 1U);
  EXPECT_EQ(compiled->report.exact_variant(1)->region_count(), 1U);
  EXPECT_EQ(packing.factories.load(), 1U);
  EXPECT_EQ(compiled->report.total_compiled_storage().packed_constant_bytes,
            16U);
  EXPECT_EQ(
      compiled->report.total_compiled_storage().materialized_constant_bytes,
      16U);
  EXPECT_EQ(compiled->report.exact_variant(0)->packed_constant_count(), 1U);
  EXPECT_EQ(compiled->report.exact_variant(1)
                ->compiled_storage()
                .packed_constant_bytes,
            16U);
  EXPECT_FALSE(compiled->report.compilation_storage()->has_unknown_storage);
  auto first = compiled->executable.exact_variant(0).value();
  auto second = compiled->executable.exact_variant(1).value();
  source = {};
  *compiled = hyle::graph::CompiledGraph{};
  EXPECT_EQ(packing.destroys.load(), 0U);
  std::thread a([&] { expect_packing_replay(first, graph, context, 2); });
  std::thread b([&] { expect_packing_replay(second, graph, context, 3); });
  a.join();
  b.join();
  EXPECT_EQ(packing.factories.load(), 1U);
  EXPECT_EQ(packing.executions.load(), 2U);
  first = {};
  EXPECT_EQ(packing.destroys.load(), 0U);
  second = {};
  EXPECT_EQ(packing.destroys.load(), 1U);
  EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
}

TEST(GraphPacking, IdentitySeparatesFormatStorageAndProviderInstances) {
  for (int scenario = 0; scenario != 4; ++scenario) {
    SCOPED_TRACE(scenario);
    HostResource other;
    PackingProbe packing;
    RegionProbe first, second;
    configure_packing(first, packing);
    configure_packing(second, packing);
    packing.key_per_node = scenario == 1;
    if (scenario == 2)
      packing.second_storage = hyle::mem::MemoryResourceRef(other);
    auto context = make_context(first, second);
    const auto graph = make_packing_graph(false, 2);
    const Environment environment{
        {&context, hyle::mem::system_resource()},
        {&context, hyle::mem::MemoryResourceRef(other)}};
    std::array<hyle::graph::NodeProviderConstraint, 2> constraints;
    for (std::size_t i = 0; i < constraints.size(); ++i)
      constraints[i] = {
          graph.graph.node_id(i).value(), kRegionProviderId,
          context.provider_instance_id(scenario == 3 ? i : 0).value()};
    auto source = make_tensor(row(4));
    write(source, std::array{1.F, 2.F, 3.F, 4.F});
    const std::array bindings{
        hyle::graph::ConstantBinding{graph.weight, source.view.as_const()}};
    {
      auto compiled = hyle::graph::compile(
          graph.graph, environment.environment(),
          {.constants = bindings,
           .node_providers = constraints,
           .operation_fallback = OperationFallbackPolicy::kReject});
      ASSERT_TRUE(compiled) << compiled.error().message();
      const auto count = scenario == 0 ? 1U : 2U;
      EXPECT_EQ(compiled->report.region_count(), 2U);
      EXPECT_EQ(compiled->report.packed_constant_count(), count);
      EXPECT_EQ(compiled->report.packed_constant_bytes(), 16U * count);
      EXPECT_EQ(compiled->report.compiled_storage().packed_constant_bytes,
                16U * count);
      EXPECT_EQ(packing.factories.load(), count);
    }
    EXPECT_EQ(packing.factories.load(), packing.destroys.load());
    EXPECT_EQ(other.live_allocations, 0U);
  }
}

TEST(GraphPacking, NoGlobalCacheAcrossContentsOrPackingConfigurations) {
  PackingProbe packing;
  RegionProbe probe;
  configure_packing(probe, packing);
  auto context = make_context(probe);
  const auto graph = make_packing_graph();
  const Environment environment{{&context, hyle::mem::system_resource()}};
  auto source = make_tensor(row(4));
  const std::array bindings{
      hyle::graph::ConstantBinding{graph.weight, source.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  CompileOptions options{.constants = bindings};
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  auto first =
      hyle::graph::compile(graph.graph, environment.environment(), options)
          .value();
  // Same addresses, different family binding/content: must prepare again.
  write(source, std::array{2.F, 4.F, 6.F, 8.F});
  auto second =
      hyle::graph::compile(graph.graph, environment.environment(), options)
          .value();
  EXPECT_EQ(packing.factories.load(), 2U);
  expect_packing_replay(first.executable, graph, context, 2);
  expect_packing_replay(second.executable, graph, context, 2, 2);
  ++packing.key.configuration;
  auto third =
      hyle::graph::compile(graph.graph, environment.environment(), options)
          .value();
  EXPECT_EQ(packing.factories.load(), 3U);
  expect_packing_replay(third.executable, graph, context, 2, 2);
}

TEST(GraphPacking, BoundsAndPartialFactoryFailuresAreTransactional) {
  const auto graph = make_packing_graph();
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{graph.weight, source.view.as_const()}};
  for (int scenario = 0; scenario != 7; ++scenario) {
    SCOPED_TRACE(scenario);
    PackingProbe packing;
    RegionProbe probe;
    configure_packing(probe, packing);
    auto context = make_context(probe);
    CompileOptions options{
        .constants = bindings,
        .compilation_storage = {{.max_resident_bytes = 1U << 20,
                                 .max_peak_bytes = 1U << 20}},
        .operation_fallback = OperationFallbackPolicy::kReject,
        .max_region_proposals = 8,
        .max_region_proposals_per_provider = 4,
        .max_region_proposal_nodes = 16};
    if (scenario == 0 || scenario == 1)
      packing.bounds = {};
    if (scenario == 1)
      options.compilation_storage->unknown =
          hyle::graph::UnknownStoragePolicy::kAllow;
    if (scenario == 2)
      packing.bounds = {SIZE_MAX, SIZE_MAX};
    if (scenario == 3)
      packing.factory_failure =
          Status::ResourceExhausted("real factory failure");
    if (scenario == 4)
      packing.inconsistent_second_request = true;
    if (scenario == 5)
      packing.invalid_constant = true;
    if (scenario == 6) {
      packing.bytes = SIZE_MAX;
      packing.bounds = {16, 16};
    }
    {
      auto compiled = hyle::graph::compile(graph.graph, context, options);
      if (scenario == 1) {
        ASSERT_TRUE(compiled) << compiled.error().message();
        EXPECT_TRUE(
            compiled->report.compilation_storage()->has_unknown_storage);
      } else {
        ASSERT_FALSE(compiled);
        if (scenario == 3) {
          EXPECT_EQ(compiled.error(), packing.factory_failure);
        }
        if (scenario == 4 || scenario == 5) {
          EXPECT_TRUE(compiled.error().IsInvalidArgument())
              << compiled.error().message();
        }
        if (scenario == 0 || scenario == 2 || scenario == 5) {
          EXPECT_EQ(packing.factories.load(), 0U);
        }
      }
    }
    EXPECT_EQ(packing.factories.load(), packing.destroys.load());
    EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
    EXPECT_GT(probe.region_resource_count, 0U);
  }
  hyle::graph::ConstantArtifactView view{&source, 4};
  auto unavailable = hyle::graph::ConstantArtifactAccess{}.get_or_create(
      graph.weight, {}, &pack_weights, nullptr, &view);
  EXPECT_EQ(unavailable.kind, ProviderPrepareOutcomeKind::kError);
  EXPECT_EQ(view.state, nullptr);
}

TEST(GraphPacking,
     RejectedPackingDiesBeforeFallbackAndSelectedPackingSurvives) {
  for (bool first_selected : {false, true}) {
    PackingProbe first_packing, second_packing;
    RegionProbe first, second;
    configure_packing(first, first_packing);
    configure_packing(second, second_packing);
    second.implementation = hyle::compute::ProviderImplementationId{902};
    auto context = make_context(first, second);
    const auto graph = make_packing_graph();
    const Environment environment{{&context, hyle::mem::system_resource()}};
    auto source = make_tensor(row(4));
    write(source, std::array{1.F, 2.F, 3.F, 4.F});
    const std::array bindings{
        hyle::graph::ConstantBinding{graph.weight, source.view.as_const()}};
    CompileOptions options{.constants = bindings,
                           .operation_fallback =
                               OperationFallbackPolicy::kReject};
    first_packing.decline_after_pack = !first_selected;
    {
      auto compiled =
          hyle::graph::compile(graph.graph, environment.environment(), options);
      ASSERT_TRUE(compiled) << compiled.error().message();
      EXPECT_EQ(compiled->report.packed_constant_count(), 1U);
      EXPECT_EQ(compiled->report.step(0)->provider_instance,
                context.provider_instance_id(first_selected ? 0 : 1).value());
      auto &rejected = first_selected ? second_packing : first_packing;
      EXPECT_EQ(rejected.factories.load(), rejected.destroys.load());
      expect_packing_replay(compiled->executable, graph, context, 2);
    }
    EXPECT_EQ(first_packing.factories.load(), first_packing.destroys.load());
    EXPECT_EQ(second_packing.factories.load(), second_packing.destroys.load());
  }
}

TEST(GraphPacking, OpaqueMetadataCannotDisappearFromCompiledStorage) {
  PackingProbe packing;
  packing.extra_private_bytes = 4096;
  packing.bounds = {4112, 4112};
  RegionProbe probe;
  configure_packing(probe, packing);
  auto context = make_context(probe);
  const auto graph = make_packing_graph();
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{graph.weight, source.view.as_const()}};
  CompileOptions options{
      .constants = bindings,
      .compiled_storage = {.max_bytes = sizeof(PackedRegion) + 16},
      .operation_fallback = OperationFallbackPolicy::kReject};
  EXPECT_FALSE(hyle::graph::compile(graph.graph, context, options));
  EXPECT_GT(packing.factories.load(), 0U);
  options.compiled_storage.max_bytes += sizeof(PackedWeights) + 4096;
  {
    auto exact = hyle::graph::compile(graph.graph, context, options);
    ASSERT_TRUE(exact) << exact.error().message();
    EXPECT_EQ(exact->report.compiled_storage().total_bytes(),
              options.compiled_storage.max_bytes);
  }
  packing.unknown_persistent = true;
  auto unknown = hyle::graph::compile(graph.graph, context, options);
  ASSERT_FALSE(unknown);
  EXPECT_TRUE(unknown.error().IsFailedPrecondition())
      << unknown.error().message();
  options.compiled_storage.unknown = hyle::graph::UnknownStoragePolicy::kAllow;
  {
    auto partial = hyle::graph::compile(graph.graph, context, options);
    ASSERT_TRUE(partial) << partial.error().message();
    EXPECT_EQ(partial->report.compiled_storage().unknown_provider_count, 1U);
    EXPECT_FALSE(partial->report.compiled_storage().total_bytes());
  }
  EXPECT_EQ(packing.factories.load(), packing.destroys.load());
}

TEST(GraphPacking, FamilyCapsChargeSharedOpaqueAndPackedStorageOnlyOnce) {
  PackingProbe packing;
  packing.extra_private_bytes = 4096;
  packing.bounds = {4112, 4112};
  RegionProbe probe;
  configure_packing(probe, packing);
  auto context = make_context(probe);
  auto graph = make_packing_graph(true);
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{graph.weight, source.view.as_const(),
                                   hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array a{hyle::graph::ShapeBinding{graph.rows, 2}};
  const std::array b{hyle::graph::ShapeBinding{graph.rows, 3}};
  const std::array variants{hyle::graph::ExactShapeVariant{a},
                            hyle::graph::ExactShapeVariant{b}};
  CompileOptions options{
      .shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain,
      .exact_variants = variants,
      .constants = bindings,
      .compilation_storage = hyle::graph::CompilationStorageLimits{},
      .max_region_proposals = 8,
      .max_region_proposals_per_provider = 4,
      .max_region_proposal_nodes = 16};
  {
    auto measured = hyle::graph::compile(graph.graph, context, options);
    ASSERT_TRUE(measured) << measured.error().message();
    const auto total = measured->report.total_compiled_storage();
    EXPECT_EQ(total.provider_persistent_bytes,
              measured->report.compiled_storage().provider_persistent_bytes +
                  2 * sizeof(PackedRegion) + sizeof(PackedWeights) + 4096);
    EXPECT_EQ(total.packed_constant_bytes, 16U);
    options.compiled_storage.max_bytes = total.known_bytes();
    const auto usage = *measured->report.compilation_storage();
    options.compilation_storage->max_resident_bytes = usage.resident_bytes;
    options.compilation_storage->max_peak_bytes = usage.peak_bytes;
  }
  {
    auto exact = hyle::graph::compile(graph.graph, context, options);
    ASSERT_TRUE(exact) << exact.error().message();
    EXPECT_EQ(exact->report.total_compiled_storage().total_bytes(),
              options.compiled_storage.max_bytes);
  }
  --options.compiled_storage.max_bytes;
  auto rejected = hyle::graph::compile(graph.graph, context, options);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted())
      << rejected.error().message();
  EXPECT_EQ(packing.factories.load(), packing.destroys.load());
}

TEST(GraphPacking, EveryMetadataAllocationFailureReleasesPackingAndCanRetry) {
  struct State {
    AllocatorRef backing = hyle::mem::system_allocator();
    std::size_t attempts = 0, fail_at = SIZE_MAX, live = 0;
  } state;
  const AllocatorRef allocator{
      +[](void *opaque, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        auto &s = *static_cast<State *>(opaque);
        if (s.attempts++ == s.fail_at)
          return nullptr;
        void *p = s.backing.allocate(bytes, alignment);
        s.live += p != nullptr;
        return p;
      },
      +[](void *opaque, void *p, std::size_t bytes,
          std::size_t alignment) noexcept {
        auto &s = *static_cast<State *>(opaque);
        s.live -= p != nullptr;
        s.backing.deallocate(p, bytes, alignment);
      },
      +[](void *opaque, void *p, std::size_t old_bytes, std::size_t bytes,
          std::size_t alignment) noexcept -> void * {
        auto &s = *static_cast<State *>(opaque);
        if (bytes == 0) {
          s.live -= p != nullptr;
          return s.backing.reallocate(p, old_bytes, 0, alignment);
        }
        if (s.attempts++ == s.fail_at)
          return nullptr;
        void *result = s.backing.reallocate(p, old_bytes, bytes, alignment);
        s.live += p == nullptr && result != nullptr;
        return result;
      },
      &state};
  PackingProbe packing;
  RegionProbe probe;
  configure_packing(probe, packing);
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(probe), 100}};
  auto context =
      Context::create(hyle::compute::cpu::device(),
                      {.metadata_allocator = allocator, .providers = providers})
          .value();
  const auto graph = make_packing_graph(false, 2);
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array bindings{
      hyle::graph::ConstantBinding{graph.weight, source.view.as_const()}};
  CompileOptions options{
      .constants = bindings,
      .compilation_storage = {{.max_resident_bytes = 1U << 20,
                               .max_peak_bytes = 1U << 20}},
      .operation_fallback = OperationFallbackPolicy::kReject,
      .max_region_proposals = 8,
      .max_region_proposals_per_provider = 4,
      .max_region_proposal_nodes = 16};
  const auto live = state.live;
  state.attempts = 0;
  {
    auto result = hyle::graph::compile(graph.graph, context, options);
    ASSERT_TRUE(result) << result.error().message();
  }
  const auto allocations = state.attempts;
  EXPECT_EQ(state.live, live);
  for (std::size_t i = 0; i < allocations; ++i) {
    SCOPED_TRACE(i);
    state.attempts = 0;
    state.fail_at = i;
    {
      auto failed = hyle::graph::compile(graph.graph, context, options);
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
    }
    EXPECT_EQ(state.live, live);
    state.fail_at = SIZE_MAX;
    {
      auto retry = hyle::graph::compile(graph.graph, context, options);
      ASSERT_TRUE(retry) << retry.error().message();
    }
    EXPECT_EQ(state.live, live);
    EXPECT_EQ(probe.compile_count, probe.destroy_prepared_count);
  }
  context = {};
  EXPECT_EQ(state.live, 0U);
}

} // namespace
