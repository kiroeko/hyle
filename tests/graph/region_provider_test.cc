#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/region_provider.h"
#include "hyle/graph/result.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/detail/region_adapter.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/stats.h"
#include "provider_internal.h"
#include "plan_state.h"
#include "region_discovery.h"
#include "submission_internal.h"
#include "submission_lifetime.h"
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
struct ReadyConstantProbe;
struct RegionProbe;
class TransientHostResource;

struct AsyncQueuePrefix final {
  RegionProbe *probe = nullptr;
  std::size_t prefix = 0;
};

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
  std::size_t binding_prepare_count = 0;
  std::size_t binding_seal_count = 0;
  Status binding_seal_failure{};
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
  std::array<hyle::mem::BufferView, 2> queued_copy_destinations{};
  std::size_t capture_begin_count = 0;
  std::size_t capture_end_count = 0;
  std::size_t capture_abort_count = 0;
  std::size_t capture_launch_count = 0;
  std::size_t capture_destroy_count = 0;
  std::size_t capture_recycle_count = 0;
  std::size_t capture_rebuild_count = 0;
  Status capture_end_failure{};
  std::size_t destroy_prepared_count = 0;
  std::size_t effect_create_count = 0;
  std::size_t effect_submit_count = 0;
  std::size_t completed_prefix = 0;
  // Opt-in sequence snapshots let the frontier tests distinguish an old Queue
  // prefix from the actual later accepted prefix on that same Queue.
  bool fixed_prefix_completions = false;
  std::size_t fixed_prefix_count = 0;
  std::size_t abandoned_prefix = 0;
  std::array<AsyncQueuePrefix, 4> fixed_prefixes{};
  bool cancelled = false;
  bool terminal_failure = false;
  bool unknown_completion_observation = false;
  bool asynchronous_backend = false;
  bool delegated_single = false;
  bool bounded_unary_domain = false;
  bool deferred_outputs = false;
  bool deferred_owned_output = false;
  bool deferred_transient_export = false;
  std::size_t deferred_lease_exports = 0;
  std::size_t deferred_projections = 0;
  std::size_t deferred_references = 0;
  std::size_t deferred_owned_inputs = 0;
  Status (*deferred_binding_observer)(
      void *, const hyle::graph::RegionValue &) noexcept = nullptr;
  void *deferred_binding_observer_context = nullptr;
  bool transfer_supported = true;
  bool queued_transfers = false;
  bool dependency_wait_supported = true;
  bool capture_supported = false;
  bool capture_preserves_binding_identity = false;
  bool capture_recycling_supported = false;
  bool capture_private_arguments_stable = false;
  bool borrows_tensor_bindings = false;
  hyle::mem::ConstBufferView recording_input{}, recording_output{};
  hyle::mem::ConstBufferView launched_input{}, launched_output{};
  bool capture_active = false;
  bool capture_control_only = false;
  bool require_cold_bindings = false;
  bool cold_bindings_capture_compatible = false;
  std::optional<std::size_t> required_capture_destroys_before_submit{};
  bool configurable_region = false;
  std::size_t configuration_count = 0;
  std::size_t capture_destroys_at_configuration = 0;
  Status configuration_failure{};
  std::optional<std::uint32_t> uncapturable_region_begin{};
  Status binding_prepare_failure{};
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
  bool observe_leaf_preferences = false;
  std::array<hyle::graph::RegionLeafPreference, 8> observed_leaf_preferences{};
  std::size_t observed_leaf_count = 0;
  std::size_t leaf_proposals_at_discovery = 0;
  std::size_t resource_scratch_bytes = 0;
  std::optional<bool> proposed_effects_override{};
  std::size_t observed_effect_count = 0;
  hyle::compute::ResourceAccess observed_effect_access{};
  bool observed_unknown_effects = false;
  std::int32_t last_resource_read = 0;
  std::optional<hyle::compute::ProviderReplayClass> proposed_replay_override{};
  std::optional<hyle::graph::RegionOutputContract> proposed_output_contract{};
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
  ReadyConstantProbe *ready_constants = nullptr;
};

struct ReadyConstantProbe final {
  struct Input final {
    hyle::graph::RegionConstantOrigin origin;
    ConstTensorView tensor;
    hyle::mem::MemoryResourceRef owner_resource;
    hyle::graph::ConstantPolicy policy;
    bool has_lease;
    bool can_prepare_queued;
  };
  std::vector<NodeId> nodes;
  std::vector<Input> inputs;
  std::vector<std::array<float, 4>> source_values;
  // Borrowed observer of the fixture's actual mutable payload owner.
  hyle::mem::BufferView prepared_backing{};
  std::size_t output_limit = SIZE_MAX;
  std::size_t factories = 0;
  std::size_t destroys = 0;
  std::size_t compile_attempts = 0;
  bool require_queued_eligibility = false;
};

struct ReadyConstantPayload final {
  ReadyConstantProbe *probe = nullptr;
  Buffer buffer;
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
  bool bounded_unary = false;
};

struct RegionRun final {
  RegionProbe *probe = nullptr;
  hyle::mem::ConstBufferView cold_input{};
  hyle::mem::BufferView cold_output{};
  std::uint32_t configuration_revision = 0;
};

struct PackingProbe final {
  hyle::graph::ConstantPackingKey key{1, 0};
  hyle::mem::MemoryResourceRef storage{};
  hyle::mem::MemoryResourceRef second_storage{};
  hyle::mem::MemoryResourceRef expected_source_resource{};
  std::optional<std::array<float, 4>> expected_source_values{};
  void *observed_source_handle = nullptr;
  hyle::compute::ProviderCompilationStorage bounds{16, 16};
  std::size_t bytes = 16;
  std::size_t extra_private_bytes = 0;
  bool unknown_persistent = false;
  bool key_per_node = false;
  bool inconsistent_second_request = false;
  bool invalid_constant = false;
  bool decline_after_pack = false;
  bool queue_preparation = false;
  bool retry_failed_preparation = false;
  RegionProbe *preparation_queue = nullptr;
  const hyle::compute::ResourceBudget *preparation_budget = nullptr;
  Status factory_failure{};
  std::atomic<std::size_t> factories{0}, destroys{0}, executions{0};
  std::atomic<std::size_t> preparation_retires{0};
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
  probe.completed_prefix = (std::max)(probe.submit_count, probe.abandoned_prefix);
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
  if (probe.unknown_completion_observation)
    return std::unexpected(
        Status::Unavailable("injected unknown preparation observation"));
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
  if (probe.unknown_completion_observation)
    return Status::Unavailable("injected unknown preparation observation");
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

void destroy_fixed_prefix(void *state,
    hyle::compute::detail::CompletionLifetime lifetime, Status hint) noexcept {
  const auto &prefix = *static_cast<AsyncQueuePrefix *>(state);
  auto &probe = *prefix.probe;
  ++probe.completion_destroy_count;
  if (!lifetime.valid()) return;
  if (probe.completed_prefix < prefix.prefix) {
    probe.abandoned_lifetime = lifetime;
    probe.abandoned_hint = std::move(hint);
    probe.abandoned_prefix = prefix.prefix;
    return;
  }
  auto terminal = async_terminal(probe);
  terminal.Update(std::move(hint));
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

Result<hyle::compute::CompletionPoll> poll_fixed_prefix(void *state) noexcept {
  const auto &prefix = *static_cast<AsyncQueuePrefix *>(state);
  auto &probe = *prefix.probe;
  ++probe.completion_poll_count;
  if (probe.completed_prefix < prefix.prefix)
    return hyle::compute::CompletionPoll::kPending;
  const auto terminal = async_terminal(probe);
  if (!terminal.ok()) return std::unexpected(terminal);
  return hyle::compute::CompletionPoll::kComplete;
}

Status wait_fixed_prefix(void *state) noexcept {
  const auto &prefix = *static_cast<AsyncQueuePrefix *>(state);
  auto &probe = *prefix.probe;
  ++probe.completion_wait_count;
  probe.completed_prefix = (std::max)(probe.completed_prefix, prefix.prefix);
  return async_terminal(probe);
}

Status cancel_fixed_prefix(void *state) noexcept {
  auto &prefix = *static_cast<AsyncQueuePrefix *>(state);
  prefix.probe->cancelled = true;
  prefix.probe->completed_prefix =
      (std::max)(prefix.probe->completed_prefix, prefix.prefix);
  return Status::Ok();
}

const hyle::compute::detail::CompletionVTable kFixedPrefixCompletionVTable{
    .destroy = &destroy_fixed_prefix,
    .poll = &poll_fixed_prefix,
    .wait = &wait_fixed_prefix,
    .request_cancel = &cancel_fixed_prefix,
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

// Each Buffer token owns the same original host Root, while every exported
// ReadLease adopts a fresh Domain. The slot never caches an exported Domain.
class TransientHostResource final : public hyle::mem::MemoryResource {
  struct Root final {
    Buffer backing;
    std::size_t *live;
    Root(Buffer selected, std::size_t &count) noexcept : backing(std::move(selected)),live(&count) { ++*live; }
    ~Root() { backing={};--*live; }
  };
  struct Token final { std::shared_ptr<Root> root; };
public:
  std::size_t live_roots=0,live_tokens=0,exports=0;
  hyle::mem::MemoryResourceRef ref() noexcept { return hyle::mem::MemoryResourceRef(*this); }
  std::string_view name() const noexcept override { return "graph.transient-export.host"; }
  hyle::mem::MemoryProperties properties() const noexcept override {
    return {.flags=hyle::mem::MemoryProperty::kHostAccessible |
                   hyle::mem::MemoryProperty::kHostDeviceCoherent |
                   hyle::mem::MemoryProperty::kHostAddressIdentity,
            .domain=hyle::mem::MemoryDomainId{this,0}};
  }
  Result<hyle::mem::ResourceAllocation> allocate(std::size_t bytes,std::size_t alignment) noexcept override {
    HYLE_ASSIGN_OR_RETURN(auto backing,Buffer::create(hyle::mem::system_resource(),bytes,alignment));
    std::shared_ptr<Root> root;
    try { root=std::make_shared<Root>(std::move(backing),live_roots); }
    catch(const std::bad_alloc &) { return std::unexpected(Status::ResourceExhausted("transient host root")); }
    auto *token=new(std::nothrow) Token{std::move(root)};
    if (!token) return std::unexpected(Status::ResourceExhausted("transient host token"));
    ++live_tokens;return hyle::mem::ResourceAllocation{token,bytes,alignment};
  }
  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    delete static_cast<Token *>(allocation.native_handle);--live_tokens;
  }
  Result<Buffer> export_alias(hyle::mem::BufferView original) noexcept {
    if (original.resource()!=ref() || !original.native_handle() || original.offset())
      return std::unexpected(Status::InvalidArgument("transient export source differs"));
    const auto &root=static_cast<const Token *>(original.native_handle())->root;
    if (original.size()!=root->backing.size())
      return std::unexpected(Status::InvalidArgument("transient export is not the whole original root"));
    auto *token=new(std::nothrow) Token{root};
    if (!token) return std::unexpected(Status::ResourceExhausted("transient export token"));
    ++live_tokens;
    auto result=Buffer::adopt(ref(),{token,root->backing.size(),root->backing.alignment()});
    if (!result) { delete token;--live_tokens; }
    else ++exports;
    return result;
  }
  Result<hyle::mem::ResourceHostMapping> map_host(void *handle,std::size_t offset,
      std::size_t bytes,hyle::mem::HostAccess access) noexcept override {
    const auto &root=static_cast<const Token *>(handle)->root;
    if (offset>root->backing.size() || bytes>root->backing.size()-offset)
      return std::unexpected(Status::OutOfRange("transient root mapping range"));
    return hyle::mem::system_resource().resource().map_host(root->backing.native_handle(),offset,bytes,access);
  }
  Status unmap_host(void *handle,void *token,std::size_t offset,std::size_t bytes,
      hyle::mem::HostAccess access) noexcept override {
    return hyle::mem::system_resource().resource().unmap_host(
        static_cast<const Token *>(handle)->root->backing.native_handle(),token,offset,bytes,access);
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
  if (probe.ready_constants != nullptr)
    return add_node_set_proposal(writer, probe.ready_constants->nodes,
                                 1, probe.discovery_replay_class,
                                 probe.implementation);
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

Status propose_regions(void *context, const hyle::graph::RegionDiscoveryRequest &request,
                       AllocatorRef allocator,
                       RegionProposalWriter &writer) noexcept {
  const auto &graph = request.graph;
  auto &probe = *static_cast<RegionProbe *>(context);
  if (probe.observe_leaf_preferences) {
    if (request.leaf_preferences.size() > probe.observed_leaf_preferences.size())
      return Status::InvalidArgument("test leaf preference observation capacity");
    probe.observed_leaf_count = request.leaf_preferences.size();
    std::ranges::copy(request.leaf_preferences, probe.observed_leaf_preferences.begin());
    probe.leaf_proposals_at_discovery = probe.single_proposal_count;
  }
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

Status prepare_unary_bindings(void *prepared, void *invocation,
                              std::span<const ConstTensorView> inputs,
                              std::span<const TensorView> outputs) noexcept {
  auto &probe = *static_cast<RegionPlan *>(prepared)->probe;
  if (!probe.require_cold_bindings)
    return Status::Ok();
  ++probe.binding_prepare_count;
  if (probe.capture_active || inputs.size() != 1 || outputs.size() != 1)
    return Status::Internal("ordinary bindings were prepared during execution");
  if (!probe.binding_prepare_failure.ok())
    return probe.binding_prepare_failure;
  auto &run = *static_cast<RegionRun *>(invocation);
  run.cold_input = inputs[0].storage();
  run.cold_output = outputs[0].storage();
  return Status::Ok();
}

Status execute_region(void *prepared, void *invocation,
                      std::span<const ConstTensorView> inputs,
                      std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  if (invocation == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return Status::Internal("region binding mismatch");
  if (plan.probe->require_cold_bindings) {
    const auto &run = *static_cast<RegionRun *>(invocation);
    const auto same = [](auto a, auto b) noexcept {
      return a.resource() == b.resource() &&
             a.native_handle() == b.native_handle() &&
             a.offset() == b.offset() && a.size() == b.size();
    };
    if (!same(run.cold_input, inputs[0].storage()) ||
        !same(run.cold_output, outputs[0].storage()))
      return Status::Internal("ordinary submit did not use its cold bindings");
  }
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
  if (plan.bounded_unary) {
    const auto &input = inputs[0].description();
    const auto &output = outputs[0].description();
    if (!input.is_plain() || !output.is_plain() ||
        input.data_type() != DataType::kFloat32 || output.data_type() != DataType::kFloat32 ||
        input.rank() != 1 || output.rank() != 1 || !output.is_dense() ||
        input.dimensions()[0] != output.dimensions()[0] || input.byte_strides()[0] < 0 ||
        writable->size() % sizeof(float) != 0 ||
        writable->size() / sizeof(float) != static_cast<std::size_t>(input.dimensions()[0])) {
      output_mapping->close().IgnoreError();
      input_mapping->close().IgnoreError();
      return Status::Internal("bounded unary binding differs from its domain");
    }
    const auto count = static_cast<std::size_t>(input.dimensions()[0]);
    const auto stride = static_cast<std::size_t>(input.byte_strides()[0]);
    if (count && (source.size() < sizeof(float) ||
        (stride && count - 1 > (source.size() - sizeof(float)) / stride))) {
      output_mapping->close().IgnoreError();
      input_mapping->close().IgnoreError();
      return Status::Internal("bounded unary input exceeds its checked storage");
    }
    for (std::size_t i = 0; i < count; ++i) {
      float value = 0;
      std::memcpy(&value, source.data() + i * stride, sizeof(value));
      value = -value;
      std::memcpy(writable->data() + i * sizeof(value), &value, sizeof(value));
    }
  } else {
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
  }
  Status terminal = output_mapping->close();
  const Status input_close = input_mapping->close();
  if (terminal.ok() && !input_close.ok())
    terminal = input_close;
  ++plan.probe->submit_count;
  return terminal;
}

hyle::compute::detail::EnqueueResult
submit_region(void *prepared, void *invocation, hyle::compute::detail::QueueSequence &,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  if (!plan.probe->capture_active &&
      plan.probe->required_capture_destroys_before_submit &&
      plan.probe->capture_destroy_count !=
          *plan.probe->required_capture_destroys_before_submit)
    return hyle::compute::detail::EnqueueResult::rejected(
        Status::Internal("ordinary encoder retained an old capture"));
  if (plan.probe->mode == ProposalMode::kAsyncPairs ||
      plan.probe->mode == ProposalMode::kAsyncRejectSecond) {
    if (plan.probe->capture_active) {
      if (inputs.size() == 1 && outputs.size() == 1) {
        plan.probe->recording_input = inputs[0].storage();
        plan.probe->recording_output = outputs[0].storage();
      }
      return hyle::compute::detail::EnqueueResult::accepted_command();
    }
    ++plan.probe->submit_attempt_count;
    if (plan.probe->mode == ProposalMode::kAsyncRejectSecond &&
        plan.probe->submit_attempt_count == 2) {
      return hyle::compute::detail::EnqueueResult::rejected(
          Status::Unavailable("injected second-region rejection"));
    }
  }
  if (plan.probe->capture_control_only) {
    ++plan.probe->submit_count;
    return hyle::compute::detail::EnqueueResult::accepted_command();
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

constexpr hyle::compute::ProviderExtensionId kTestConfiguration{
    0x54455354434f4e46ULL, 0x494755524154494fULL, 1};

Status inspect_region_configuration(
    const ProviderRegionPrepared &prepared, void *, std::uint32_t,
    hyle::compute::ProviderExtensionId capability,
    hyle::graph::detail::RegionNodeAccess *access) noexcept {
  const auto &probe = *static_cast<RegionPlan *>(prepared.state)->probe;
  if (!probe.configurable_region || capability != kTestConfiguration || !access)
    return Status::Unimplemented("test region has no requested configuration");
  access->access.requires_configuration = true;
  return Status::Ok();
}

Result<bool> configure_region_revision(
    const ProviderRegionPrepared &prepared, void *invocation, std::uint32_t,
    hyle::compute::ProviderExtensionId capability,
    const void *configuration) noexcept {
  auto &probe = *static_cast<RegionPlan *>(prepared.state)->probe;
  if (!probe.configurable_region || capability != kTestConfiguration ||
      !invocation || !configuration)
    return std::unexpected(Status::InvalidArgument("invalid test configuration"));
  ++probe.configuration_count;
  probe.capture_destroys_at_configuration = probe.capture_destroy_count;
  // Deliberately mutate the selected slot before a possible cold failure.
  static_cast<RegionRun *>(invocation)->configuration_revision =
      *static_cast<const std::uint32_t *>(configuration);
  if (!probe.configuration_failure.ok())
    return std::unexpected(probe.configuration_failure);
  return true;
}

const hyle::graph::detail::RegionProjectedReadVTable kConfigurationVTable{
    .inspect_node = &inspect_region_configuration,
    .configure_node = &configure_region_revision,
};

const void *query_region_configuration(
    void *prepared, hyle::compute::ProviderExtensionId id) noexcept {
  const auto &probe = *static_cast<RegionPlan *>(prepared)->probe;
  if (probe.capture_private_arguments_stable &&
      id == hyle::graph::detail::kRegionCaptureBindingsExtensionId) {
    static constexpr hyle::graph::detail::RegionCaptureBindingsVTable capture{
        [](void *value, void *slot) noexcept -> bool {
          const auto *plan = static_cast<const RegionPlan *>(value);
          const auto *run = static_cast<const RegionRun *>(slot);
          return plan && run && run->probe == plan->probe &&
              plan->probe->capture_private_arguments_stable;
        }};
    return &capture;
  }
  if (probe.borrows_tensor_bindings &&
      id == hyle::graph::detail::kRegionBindingReleaseExtensionId) {
    // This opt-in fixture passes only borrowed TensorViews and owns no native
    // input imports. Withdrawal therefore has no retained input owner to drop.
    static constexpr hyle::graph::detail::RegionBindingReleaseVTable release{
        .validate = [](void *, void *) noexcept { return Status::Ok(); },
        .release = [](void *, void *) noexcept {},
        .borrows_tensor_views_only = true};
    return &release;
  }
  return probe.configurable_region &&
          id == hyle::graph::detail::kRegionProjectedReadExtensionId
      ? &kConfigurationVTable : nullptr;
}

const hyle::graph::ProviderRegionPreparedVTable kPreparedRegionVTable{
    .destroy_prepared = &destroy_region_prepared,
    .alias_support = &region_alias_support,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .shared_submit = &submit_region,
    .seal_bindings = [](void *prepared, void *) noexcept -> Status {
      auto &probe = *static_cast<RegionPlan *>(prepared)->probe;
      ++probe.binding_seal_count;
      return probe.binding_seal_failure;
    },
    .query_extension = &query_region_configuration,
};
const hyle::graph::ProviderRegionPreparedVTable kPrivateStableRegionVTable = [] {
  auto table = kPreparedRegionVTable;
  table.prepare_bindings = +[](void *prepared, void *slot,
      std::span<const hyle::graph::RegionValue> inputs,
      std::span<const TensorView> outputs) noexcept -> Status {
    if (inputs.size() != 1)
      return Status::InvalidArgument("private stable mock expects one input");
    const std::array tensors{inputs[0].tensor};
    return prepare_unary_bindings(prepared, slot, tensors, outputs);
  };
  return table;
}();

ProviderPrepareOutcome compile_ready_constant_region(
    RegionProbe &probe, const RegionCompileRequest &request,
    AllocatorRef allocator, ProviderRegionPrepared *output) noexcept {
  auto &ready = *probe.ready_constants;
  ++ready.compile_attempts;
  if (request.outputs.empty() || request.output_resources.empty())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("ready constant test has no output"));
  EXPECT_EQ(request.constants.size(), request.constant_origins.size());
  for (std::size_t index = 0; index < request.constants.size(); ++index) {
    const auto &binding = request.constants[index];
    if (index >= request.constant_origins.size())
      return ProviderPrepareOutcome::error(Status::Internal("missing test constant origin"));
    EXPECT_EQ(request.constant_origins[index].value, binding.value);
    ready.inputs.push_back({request.constant_origins[index], binding.tensor,
        binding.owner.ref(), binding.policy, binding.lease.valid(),
        request.constant_artifacts.can_prepare_queued(binding.value,
                                                       request.output_resources[0])});
  }
  if (request.constants.empty())
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("test boundary has no cold constant"));
  if (request.constants.size() != 1 || request.inputs.size() != 1 ||
      request.outputs.empty())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("ready constant test boundary"));
  if (ready.require_queued_eligibility && !ready.inputs.back().can_prepare_queued)
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("test boundary has no owned queued preparation"));
  struct FactoryContext final {
    ReadyConstantProbe *probe;
    std::size_t output_count;
  } factory_context{&ready, request.outputs.size()};
  const auto bytes = request.outputs.size() * 4 * sizeof(float);
  const auto factory = +[](void *opaque,
      const hyle::graph::ConstantArtifactPrepareRequest &prepare,
      hyle::graph::ConstantArtifactPayload *payload) noexcept -> Status {
    const auto &context = *static_cast<FactoryContext *>(opaque);
    ++context.probe->factories;
    auto *data = prepare.allocator.construct<ReadyConstantPayload>();
    if (data == nullptr)
      return Status::ResourceExhausted("ready constant payload metadata");
    data->probe = context.probe;
    *payload = {data, +[](void *state, AllocatorRef owner) noexcept {
      auto *data = static_cast<ReadyConstantPayload *>(state);
      ++data->probe->destroys;
      owner.destroy(data);
    }};
    HYLE_ASSIGN_OR_RETURN(data->buffer, Buffer::create(prepare.destination,
        context.output_count * 4 * sizeof(float), 64));
    std::array<float, 4> values{};
    HYLE_RETURN_IF_ERROR(prepare.source.storage().read(
        std::as_writable_bytes(std::span(values))));
    context.probe->source_values.push_back(values);
    for (float &value : values)
      value = -value;
    for (std::size_t port = 0; port < context.output_count; ++port)
      HYLE_RETURN_IF_ERROR(data->buffer.write(port * sizeof(values),
          std::as_bytes(std::span(values))));
    context.probe->prepared_backing = data->buffer.view();
    return Status::Ok();
  };
  hyle::graph::ConstantArtifactView artifact;
  const auto acquired = request.constant_artifacts.get_or_create(
      request.constants[0].value,
      {.key = {2, request.outputs.size()},
       .storage = request.output_resources[0], .packed_bytes = bytes,
       .persistent_bytes = sizeof(ReadyConstantPayload),
       .private_storage = {0, 0}},
      factory, &factory_context, &artifact);
  if (acquired.kind != ProviderPrepareOutcomeKind::kPrepared)
    return acquired;
  const auto &data = *static_cast<const ReadyConstantPayload *>(artifact.state);
  for (std::size_t port = 0;
       port < (std::min)(request.outputs.size(), ready.output_limit); ++port) {
    auto description = request.graph->value_description(request.outputs[port]);
    if (!description)
      return ProviderPrepareOutcome::error(std::move(description).error());
    auto storage = data.buffer.view().subview(port * 4 * sizeof(float),
                                             4 * sizeof(float));
    if (!storage)
      return ProviderPrepareOutcome::error(std::move(storage).error());
    auto binding = ConstTensorView::create(*description, *storage);
    if (!binding)
      return ProviderPrepareOutcome::error(std::move(binding).error());
    const auto bound = request.constant_artifacts.bind_prepared_output(
        request.outputs[port], artifact, *binding);
    if (!bound.ok())
      return ProviderPrepareOutcome::error(bound);
  }
  auto *plan = allocator.construct<RegionPlan>();
  if (plan == nullptr)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("ready constant region metadata"));
  plan->probe = &probe;
  *output = {&kPreparedRegionVTable, plan, probe.artifact_properties};
  ++probe.compile_count;
  return ProviderPrepareOutcome::prepared();
}

Status pack_weights(void *context,
                    const hyle::graph::ConstantArtifactPrepareRequest &request,
                    hyle::graph::ConstantArtifactPayload *output) noexcept {
  const auto source = request.source;
  const auto resource = request.destination;
  const auto allocator = request.allocator;
  auto &probe = *static_cast<PackingProbe *>(context);
  ++probe.factories;
  if (probe.expected_source_resource.valid() &&
      source.storage().resource() != probe.expected_source_resource)
    return Status::Internal("packing received another storage's constant");
  probe.observed_source_handle = source.storage().native_handle();
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
  if (probe.expected_source_values && values != *probe.expected_source_values)
    return Status::Internal("packing received incorrect local constant bytes");
  std::reverse(values.begin(), values.end());
  HYLE_RETURN_IF_ERROR(
      packed->buffer.view().write(std::as_bytes(std::span(values))));
  return probe.factory_failure;
}

Status pack_queued_weights(
    void *context,
    const hyle::graph::ConstantArtifactPrepareRequest &request,
    hyle::graph::ConstantArtifactPayload *output) noexcept {
  HYLE_RETURN_IF_ERROR(pack_weights(context, request, output));
  auto &probe = *static_cast<PackingProbe *>(context);
  if (!request.sequence || !probe.preparation_queue)
    return Status::Internal("queued packing has no original sequence");
  output->retire_preparation = +[](void *opaque) noexcept {
    auto &weights = *static_cast<PackedWeights *>(opaque);
    if (weights.probe->preparation_budget)
      EXPECT_EQ(weights.probe->preparation_budget->unattributed_usage()
                    .private_upper_bound_bytes,
                weights.probe->bounds.peak_bytes.value());
    ++weights.probe->preparation_retires;
    weights.private_metadata = {};
  };
  // The original mock Queue produces a real asynchronous Completion. Only
  // its explicit native-retirement callback may release the borrowed state.
  ++probe.preparation_queue->submit_count;
  request.sequence->record(
      hyle::compute::detail::EnqueueResult::accepted_command());
  return Status::Ok();
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
        +[](void *opaque, void *, hyle::compute::detail::QueueSequence &,
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
      .private_storage = packing.bounds,
      .queue_preparation = packing.queue_preparation};
  hyle::graph::ConstantArtifactView first, second;
  const auto constant =
      packing.invalid_constant ? request.inputs[0] : request.inputs[1];
  const auto factory = packing.queue_preparation ? &pack_queued_weights
                                                : &pack_weights;
  if (packing.retry_failed_preparation)
    first = {&packing, SIZE_MAX}; // Failure must actively clear a stale view.
  auto prepared = request.constant_artifacts.get_or_create(
      constant, declaration, factory, &packing, &first);
  if (prepared.kind != ProviderPrepareOutcomeKind::kPrepared) {
    if (packing.retry_failed_preparation &&
        prepared.kind == ProviderPrepareOutcomeKind::kError &&
        packing.preparation_queue &&
        packing.preparation_queue->abandoned_lifetime.valid()) {
      EXPECT_EQ(first.state, nullptr);
      EXPECT_EQ(first.packed_bytes, 0U);
      EXPECT_EQ(output->state, nullptr);
      EXPECT_EQ(output->vtable, nullptr);
      EXPECT_EQ(probe.compile_count, 0U);
      EXPECT_EQ(packing.factories.load(), 1U);
      finish_abandoned_prefix(*packing.preparation_queue);
      EXPECT_EQ(packing.preparation_retires.load(), 1U);
      EXPECT_EQ(packing.destroys.load(), 1U);
      // Retry within this exact cache scope, using the same constant and key.
      prepared = request.constant_artifacts.get_or_create(
          constant, declaration, factory, &packing, &first);
      EXPECT_EQ(first.state, nullptr);
      EXPECT_EQ(first.packed_bytes, 0U);
      EXPECT_EQ(output->state, nullptr);
      EXPECT_EQ(output->vtable, nullptr);
    }
    return prepared;
  }
  if (packing.inconsistent_second_request)
    ++declaration.packed_bytes;
  prepared = request.constant_artifacts.get_or_create(
      constant, declaration, factory, &packing, &second);
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

// A generated port has no ready representation during cold binding. This
// small carrier resolves original port references when it executes; the real
// Apple consumer separately tests asynchronous native Job ordering.
struct DeferredPlan final {
  RegionPlan base;
  TensorDesc description;
};
struct DeferredRun final {
  RegionRun base;
  Buffer backing;
  hyle::compute::ReadLease output_owner;
  bool immutable_generation = false;
  ConstTensorView input;
  hyle::graph::RegionOutputReference source;
  hyle::compute::ReadLease input_owner;
  Status terminal = Status::Unavailable("deferred output has not executed");
};
extern const hyle::graph::ProviderRegionPreparedVTable kDeferredRegionVTable;
const hyle::graph::ProviderRegionPreparedVTable kDeferredRegionVTable{
    .destroy_prepared = [](void *state, AllocatorRef allocator) noexcept {
      auto *plan = static_cast<DeferredPlan *>(state);
      ++plan->base.probe->destroy_prepared_count;
      allocator.destroy(plan);
    },
    .alias_support = &region_alias_support,
    .create_invocation = [](void *prepared, AllocatorRef allocator,
        hyle::mem::BufferView, void **output) noexcept -> Status {
      const auto &plan = *static_cast<DeferredPlan *>(prepared);
      auto *run = allocator.construct<DeferredRun>();
      if (!run) return Status::ResourceExhausted("deferred test slot");
      *output = run;
      run->base.probe = plan.base.probe;
      HYLE_ASSIGN_OR_RETURN(auto bytes, plan.description.required_bytes());
      const auto resource = plan.base.probe->deferred_transient_export
          ? plan.base.probe->region_output_resource : hyle::mem::system_resource();
      HYLE_ASSIGN_OR_RETURN(run->backing, Buffer::create(resource, bytes));
      ++plan.base.probe->create_invocation_count;
      return Status::Ok();
    },
    .destroy_invocation = [](void *state, AllocatorRef allocator) noexcept {
      auto *run = static_cast<DeferredRun *>(state);
      ++run->base.probe->destroy_invocation_count;
      allocator.destroy(run);
    },
    .delegated_submit = [](void *prepared, void *invocation,
        std::span<const hyle::compute::detail::CompletionDependency>,
        std::span<const ConstTensorView>, std::span<const TensorView>) noexcept -> Result<Completion> {
      auto &plan = *static_cast<DeferredPlan *>(prepared);
      auto &run = *static_cast<DeferredRun *>(invocation);
      if (run.immutable_generation)
        return std::unexpected(Status::Unavailable("owned deferred generation is immutable"));
      auto input = run.input;
      if (run.source.valid()) {
        auto value = run.source.vtable->output_value(
            run.source.prepared, run.source.invocation, run.source.port);
        if (!value) {
          run.terminal = value.error();
          return Completion::completed(run.terminal);
        }
        input = value->tensor;
      }
      HYLE_ASSIGN_OR_RETURN(auto output, TensorView::create(plan.description, run.backing.view()));
      const std::array inputs{input};
      const std::array outputs{output};
      run.terminal = execute_region(&plan.base, &run.base, inputs, outputs);
      run.immutable_generation = run.terminal.ok() &&
          (plan.base.probe->deferred_owned_output || plan.base.probe->deferred_transient_export);
      if (run.terminal.ok() && plan.base.probe->deferred_owned_output) {
        HYLE_ASSIGN_OR_RETURN(run.output_owner,
            hyle::compute::ReadLease::adopt(std::move(run.backing)));
      }
      return Completion::completed(run.terminal);
    },
    .endpoint = [](void *prepared, void *) noexcept {
      return hyle::compute::detail::CompletionEndpoint{
          &kDeferredRegionVTable, static_cast<DeferredPlan *>(prepared)->base.probe};
    },
    .can_import_endpoint = [](void *prepared, void *,
        hyle::compute::detail::CompletionEndpoint source) noexcept {
      return source.authority == &kDeferredRegionVTable &&
          source.identity == static_cast<DeferredPlan *>(prepared)->base.probe;
    },
    .output_contract = [](void *, std::size_t) noexcept {
      return hyle::graph::RegionOutputContract{
          .kind = hyle::graph::RegionOutputKind::kDeferredGenerated};
    },
    .output_value = [](void *prepared, void *invocation, std::size_t port) noexcept
        -> Result<hyle::graph::RegionValue> {
      auto &plan = *static_cast<DeferredPlan *>(prepared);
      auto &run = *static_cast<DeferredRun *>(invocation);
      ++plan.base.probe->deferred_projections;
      if (port != 0) return std::unexpected(Status::OutOfRange("deferred output port"));
      if (!run.terminal.ok()) return std::unexpected(run.terminal);
      if (run.output_owner.valid()) {
        HYLE_ASSIGN_OR_RETURN(auto tensor, run.output_owner.tensor(plan.description));
        return hyle::graph::RegionValue{.description = plan.description, .tensor = tensor};
      }
      HYLE_ASSIGN_OR_RETURN(auto tensor, ConstTensorView::create(plan.description, run.backing.view()));
      return hyle::graph::RegionValue{.description = plan.description, .tensor = tensor};
    },
    .prepare_bindings = [](void *prepared, void *invocation,
        std::span<const hyle::graph::RegionValue> inputs,
        std::span<const TensorView> outputs) noexcept -> Status {
      auto &plan = *static_cast<DeferredPlan *>(prepared);
      auto &run = *static_cast<DeferredRun *>(invocation);
      // This opt-in fixture publishes one immutable generation, never a new
      // writer over an exported backing. Existing reusable deferred slots are unchanged.
      if (run.immutable_generation)
        return Status::Unavailable("owned deferred generation is immutable");
      if (inputs.size() != 1 || outputs.size() != 1 || outputs[0].valid())
        return Status::InvalidArgument("deferred test binding arity/storage");
      if (plan.base.probe->deferred_binding_observer)
        HYLE_RETURN_IF_ERROR(plan.base.probe->deferred_binding_observer(
            plan.base.probe->deferred_binding_observer_context, inputs[0]));
      run.input = inputs[0].tensor;
      run.source = inputs[0].reference;
      if (run.source.valid()) {
        if (run.source.vtable != &kDeferredRegionVTable || run.source.port != 0 ||
            static_cast<DeferredPlan *>(run.source.prepared)->base.probe != plan.base.probe)
          return Status::InvalidArgument("deferred test source owner differs");
        ++plan.base.probe->deferred_references;
      }
      run.input_owner = inputs[0].lease ? *inputs[0].lease : hyle::compute::ReadLease{};
      if (run.input_owner.valid()) ++plan.base.probe->deferred_owned_inputs;
      return Status::Ok();
    },
    .output_lease = [](void *prepared, void *invocation, std::size_t port) noexcept
        -> Result<hyle::compute::ReadLease> {
      auto &plan = *static_cast<DeferredPlan *>(prepared);
      auto &run = *static_cast<DeferredRun *>(invocation);
      if (!plan.base.probe->deferred_owned_output && !plan.base.probe->deferred_transient_export)
        return std::unexpected(Status::Unimplemented("deferred fixture has no owned lease"));
      if (port != 0) return std::unexpected(Status::OutOfRange("deferred output lease port"));
      if (!run.terminal.ok()) return std::unexpected(run.terminal);
      if (plan.base.probe->deferred_transient_export) {
        auto *resource = dynamic_cast<TransientHostResource *>(
            &plan.base.probe->region_output_resource.resource());
        if (!resource)
          return std::unexpected(Status::InvalidArgument("transient fixture output resource differs"));
        HYLE_ASSIGN_OR_RETURN(auto backing, resource->export_alias(run.backing.view()));
        ++plan.base.probe->deferred_lease_exports;
        return hyle::compute::ReadLease::adopt(std::move(backing));
      }
      if (!run.output_owner.valid())
        return std::unexpected(Status::FailedPrecondition("deferred output lease is not ready"));
      ++plan.base.probe->deferred_lease_exports;
      return run.output_owner;
    },
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
    std::span<hyle::graph::RegionInputRequirement> input_requirements,
    std::span<hyle::graph::RegionOutputRequirement> output_requirements,
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
  if (probe.deferred_outputs) {
    for (auto &requirement : input_requirements)
      requirement.accepts_reference = true;
    for (auto &requirement : output_requirements)
      requirement.value.kind = hyle::graph::RegionOutputKind::kDeferredGenerated;
  }
  if (probe.proposed_output_contract)
    for (auto &requirement : output_requirements)
      requirement.value = *probe.proposed_output_contract;
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
    input_requirements[0].resource.exact_resource = probe.region_input_resource;
  if (probe.packing != nullptr &&
      probe.packing->expected_source_resource.valid() &&
      input_requirements.size() == 2)
    input_requirements[1].resource.exact_resource =
        probe.packing->expected_source_resource;
  if (!output_requirements.empty() && probe.region_output_resource.valid())
    output_requirements[0].resource.exact_resource = probe.region_output_resource;
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
      input_requirements[0].resource = *probe.requirement_override;
    else if (probe.requirement_port == 1 && !output_requirements.empty())
      output_requirements[0].resource = *probe.requirement_override;
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
                    : probe.capture_private_arguments_stable
                        ? &kPrivateStableRegionVTable : &kPreparedRegionVTable,
      .state = plan,
      .properties = probe.artifact_properties,
  };
  if (probe.uncapturable_region_begin == request.nodes.front().ordinal())
    output->properties.runtime.capture_compatible = false;
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
  if (probe.ready_constants != nullptr)
    return compile_ready_constant_region(probe, request, allocator, output);
  if (probe.deferred_outputs) {
    if (request.nodes.empty() || request.inputs.size() != 1 || request.outputs.size() != 1)
      return ProviderPrepareOutcome::error(Status::InvalidArgument("deferred test boundary"));
    auto *plan = allocator.construct<DeferredPlan>();
    if (!plan) return ProviderPrepareOutcome::error(Status::ResourceExhausted("deferred test artifact"));
    plan->base = {&probe, static_cast<std::uint32_t>(request.nodes.size())};
    plan->description = request.graph->value_description(request.outputs[0]).value();
    *output = {&kDeferredRegionVTable, plan, probe.artifact_properties};
    ++probe.compile_count;
    return ProviderPrepareOutcome::prepared();
  }
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
submit_effect(void *, void *invocation, hyle::compute::detail::QueueSequence &,
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
  if (probe.cold_bindings_capture_compatible)
    runtime.capture_compatible = true;
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
    .prepare_bindings = &prepare_unary_bindings,
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

Status validate_bounded_unary(Operation operation,
    std::span<const TensorDesc> inputs, std::span<const TensorDesc> outputs,
    hyle::compute::detail::ProviderShapeDomain shape) noexcept {
  const auto *unary = operation.get_if<Unary>();
  if (!unary || unary->kind != UnaryKind::kNeg || inputs.size() != 1 || outputs.size() != 1 ||
      !inputs[0].is_plain() || !outputs[0].is_plain() ||
      inputs[0].data_type() != DataType::kFloat32 || outputs[0].data_type() != DataType::kFloat32 ||
      inputs[0].rank() != 1 || outputs[0].rank() != 1 || !outputs[0].is_dense() ||
      inputs[0].dimensions()[0] != outputs[0].dimensions()[0] || inputs[0].byte_strides()[0] < 0)
    return Status::Unimplemented("bounded test unary requires plain rank-one F32 Neg");
  if (!shape.empty() && (shape.inputs.size() != 1 || shape.outputs.size() != 1 ||
      shape.inputs[0].values[0] != hyle::compute::detail::kFixedShapeAxis ||
      shape.outputs[0].values[0] != hyle::compute::detail::kFixedShapeAxis ||
      shape.outputs[0].layout != hyle::compute::detail::ProviderTensorLayout::kDerived))
    return Status::Unimplemented("bounded test unary requires the same fixed extent and dense output");
  return Status::Ok();
}

Status instantiate_bounded_unary(void *prepared,
    std::span<const TensorDesc> inputs, std::span<const TensorDesc> outputs,
    AllocatorRef allocator, hyle::compute::detail::ProviderPrepared *instance) noexcept {
  if (!prepared || !instance)
    return Status::InvalidArgument("bounded unary cold instance is invalid");
  *instance = {};
  const auto &root = *static_cast<const RegionPlan *>(prepared);
  if (!root.bounded_unary)
    return Status::InvalidArgument("bounded unary root is not a domain artifact");
  // Compute already checked these descriptors against the root's fixed axes.
  // This callback only instantiates the same scalar code/configuration.
  HYLE_RETURN_IF_ERROR(validate_bounded_unary(Operation(Unary{UnaryKind::kNeg}),
                                             inputs, outputs, {}));
  auto *plan = allocator.construct<RegionPlan>(root);
  if (!plan) return Status::ResourceExhausted("bounded unary cold instance metadata");
  *instance = {.vtable = &kUnaryPreparedVTable, .state = plan,
               .workspace = {}, .runtime = kUnaryRuntime};
  return Status::Ok();
}

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
    if (probe.bounded_unary_domain) {
      if (probe.delegated_single || probe.single_replay_override)
        return Status::Unimplemented("bounded test unary uses its original shared runtime");
      HYLE_RETURN_IF_ERROR(validate_bounded_unary(operation, inputs, outputs, request.shape));
      if (probe.requirement_override && probe.requirement_port == 0)
        input_requirements[0].minimum_alignment = probe.requirement_override->minimum_alignment;
    }
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
      .shape_coverage = constrained_unary && probe.bounded_unary_domain
          ? hyle::compute::detail::ProviderShapeCoverage::kEntireDomain
          : hyle::compute::detail::ProviderShapeCoverage::kExactPoint,
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
    if (probe.bounded_unary_domain) {
      if (probe.delegated_single || probe.single_replay_override)
        return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
            "bounded test unary uses its original shared runtime"));
      const auto status = validate_bounded_unary(request.operation, request.inputs,
                                                request.outputs, request.shape);
      if (!status.ok()) return ProviderPrepareOutcome::unsupported(status);
    }
    void *storage = allocator.allocate(sizeof(RegionPlan), alignof(RegionPlan));
    if (storage == nullptr)
      return hyle::compute::ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("unary plan allocation failed"));
    auto *plan = ::new (storage) RegionPlan{&probe, 1};
    plan->bounded_unary = probe.bounded_unary_domain;
    *prepared = hyle::compute::detail::ProviderPrepared{
        .vtable = probe.delegated_single ? &kDelegatedUnaryPreparedVTable
                                         : &kUnaryPreparedVTable,
        .state = plan,
        .workspace = {},
        .runtime = single_runtime(probe, false),
        .instantiate = probe.bounded_unary_domain && !request.shape.exact()
            ? &instantiate_bounded_unary : nullptr,
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
  auto &probe = *static_cast<RegionProbe *>(sequence);
  if (probe.fixed_prefix_completions) {
    if (probe.fixed_prefix_count == probe.fixed_prefixes.size())
      return Completion::completed(Status::ResourceExhausted(
          "test fixed Queue prefix capacity exhausted"));
    auto &prefix = probe.fixed_prefixes[probe.fixed_prefix_count];
    prefix = {&probe, ++probe.fixed_prefix_count};
    auto completion = hyle::compute::detail::CompletionAccess::make(
        &kFixedPrefixCompletionVTable, &prefix);
    hyle::compute::detail::CompletionAccess::append_terminal(
        completion, std::move(terminal));
    return completion;
  }
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
      .mode = probe->queued_transfers || probe->unknown_completion_observation
                  ? hyle::compute::TransferMode::kQueuedDirect
                  : hyle::compute::TransferMode::kImmediate};
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
  if (probe->queued_transfers || probe->unknown_completion_observation) {
    if (probe->submit_count < probe->queued_copy_destinations.size())
      probe->queued_copy_destinations[probe->submit_count] = destination;
    ++probe->submit_count;
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
  bool recycled = false;
  hyle::mem::ConstBufferView input{}, output{};
};

Status begin_mock_capture(void *context, void *queue) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe == nullptr || queue != context || !probe->capture_supported)
    return Status::Unimplemented("mock command capture is unavailable");
  if (probe->capture_active)
    return Status::FailedPrecondition("mock command capture is active");
  probe->capture_active = true;
  probe->recording_input = {};
  probe->recording_output = {};
  ++probe->capture_begin_count;
  return Status::Ok();
}

Status end_mock_capture(void *context, void *queue, AllocatorRef allocator,
                        void *rebuild_storage, void **output) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe == nullptr || queue != context || !probe->capture_active ||
      output == nullptr || !allocator.valid()) {
    return Status::FailedPrecondition("mock command capture is invalid");
  }
  *output = nullptr;
  probe->capture_active = false;
  auto *rebuild = static_cast<MockCapturedCommands *>(rebuild_storage);
  if (rebuild != nullptr &&
      (rebuild->probe != probe || !rebuild->recycled))
    return Status::InvalidArgument("mock capture rebuild storage is invalid");
  if (!probe->capture_end_failure.ok())
    return probe->capture_end_failure;
  void *storage = allocator.allocate(sizeof(MockCapturedCommands),
                                     alignof(MockCapturedCommands));
  if (storage == nullptr)
    return Status::ResourceExhausted("mock capture allocation failed");
  *output = ::new (storage) MockCapturedCommands{
      probe, false, probe->recording_input, probe->recording_output};
  if (rebuild != nullptr)
    ++probe->capture_rebuild_count;
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
      captured->probe != probe || captured->recycled) {
    return hyle::compute::detail::EnqueueResult::rejected(
        Status::InvalidArgument("mock captured command is invalid"));
  }
  ++probe->capture_launch_count;
  probe->launched_input = captured->input;
  probe->launched_output = captured->output;
  ++probe->submit_attempt_count;
  ++probe->submit_count;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

bool recycle_mock_capture(void *state) noexcept {
  auto *captured = static_cast<MockCapturedCommands *>(state);
  if (captured == nullptr || captured->recycled ||
      !captured->probe->capture_recycling_supported)
    return false;
  captured->recycled = true;
  ++captured->probe->capture_recycle_count;
  return true;
}

const hyle::compute::detail::CommandCaptureVTable kMockCaptureVTable{
    .begin = &begin_mock_capture,
    .end = &end_mock_capture,
    .abort = &abort_mock_capture,
    .destroy = &destroy_mock_capture,
    .launch = &launch_mock_capture,
    .recycle_after_retirement = &recycle_mock_capture,
};

const hyle::compute::detail::CommandCaptureVTable kStableMockCaptureVTable = [] {
  auto table = kMockCaptureVTable;
  // Test-only opt-in to exercise generic range retention. This mock proves
  // ordering/counts; it does not certify any real backend's memory resource.
  table.tensor_bindings_have_stable_identity = true;
  return table;
}();

const void *
query_async_extension(void *context,
                      hyle::compute::BackendExtensionId id) noexcept {
  auto *probe = static_cast<RegionProbe *>(context);
  if (probe != nullptr && probe->capture_supported &&
      id == hyle::compute::detail::kCommandCaptureExtensionId) {
    return probe->capture_preserves_binding_identity
        ? &kStableMockCaptureVTable : &kMockCaptureVTable;
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

Context make_context(RegionProbe &probe,
                     hyle::mem::AllocatorRef metadata_allocator = {}) {
  const bool asynchronous = probe.asynchronous_backend ||
                            probe.mode == ProposalMode::kAsyncPairs ||
                            probe.mode == ProposalMode::kAsyncRejectSecond;
  if (asynchronous) {
    const std::array providers{
        hyle::compute::ProviderBinding{region_provider(probe), 100},
        hyle::compute::ProviderBinding{hyle::compute::native::provider(), 200}};
    return Context::create(async_device(probe),
        {.metadata_allocator = metadata_allocator, .providers = providers})
        .value();
  }
  const std::array providers{
      hyle::compute::ProviderBinding{region_provider(probe), 100},
      hyle::compute::ProviderBinding{hyle::compute::ggml_cpu::provider(), 200},
      hyle::compute::ProviderBinding{hyle::compute::native::provider(), 300}};
  return Context::create(hyle::compute::cpu::device(),
                         {.metadata_allocator = metadata_allocator, .providers = providers})
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

TEST(GraphRegions, DiscoveryReceivesTheExactProviderAdmittedLeafPreferences) {
  const auto logical = make_negations(2);
  for (const bool allow_leaves : {true, false}) {
    SCOPED_TRACE(allow_leaves);
    RegionProbe first, second;
    first.observe_leaf_preferences = second.observe_leaf_preferences = true;
    first.required_resource = second.required_resource = hyle::mem::system_resource();
    auto context = make_context(first, second);
    auto compiled = hyle::graph::compile(logical.graph, context,
        {.operation_fallback = allow_leaves ? OperationFallbackPolicy::kAllow
                                          : OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(first.observed_leaf_count, logical.graph.node_count());
    ASSERT_EQ(second.observed_leaf_count, logical.graph.node_count());
    EXPECT_EQ(first.leaf_proposals_at_discovery, allow_leaves ? 2u : 0u);
    EXPECT_EQ(second.leaf_proposals_at_discovery, allow_leaves ? 2u : 0u);
    for (std::size_t node = 0; node < logical.graph.node_count(); ++node) {
      const auto &a = first.observed_leaf_preferences[node];
      const auto &b = second.observed_leaf_preferences[node];
      if (allow_leaves) {
        EXPECT_EQ(a.provider, std::optional<std::uint32_t>{100});
        EXPECT_EQ(b.provider, std::optional<std::uint32_t>{200});
        EXPECT_EQ(a.best, std::optional<std::uint32_t>{100});
        EXPECT_EQ(b.best, a.best);
      } else {
        EXPECT_FALSE(a.provider);
        EXPECT_FALSE(a.best);
        EXPECT_FALSE(b.provider);
        EXPECT_FALSE(b.best);
      }
    }
    // Both instances have the same ProviderId; the distinct ranks above
    // therefore prove that discovery keeps their actual instance identity.
    EXPECT_EQ(first.compile_count, 1u);
    EXPECT_EQ(second.compile_count, 0u);
    EXPECT_EQ(compiled->report.region_count(), 1u);
  }
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

TEST(GraphRouteBoundary, CapabilityAdapterKeepsOriginalRegionsAndOnePlan) {
  const auto logical = make_negations(3);
  RegionProbe original{.mode = ProposalMode::kPairs};
  RegionProbe adapted{.mode = ProposalMode::kSingleton};
  adapted.singleton_node = 2;
  auto context = make_context(original);
  const Environment environment{{&context, hyle::mem::system_resource()}};
  auto source = hyle::compute::detail::ProviderExtensionAccess::query(
      context, 0, hyle::graph::kRegionProviderExtensionId);
  ASSERT_TRUE(source) << source.error().message();
  std::array adapters{hyle::graph::detail::RegionAdapter{
      .source = hyle::graph::detail::GraphRegionSource::shared(
          environment.environment().targets[environment.target(0).ordinal()].source,
          std::move(*source)).value(),
      .provider_index = 0,
      .target = environment.target(0),
      .interface = &kRegionVTable,
      .context = &adapted}};
  auto compiled = hyle::graph::detail::compile_with_adapters(
      logical.graph, environment.environment(),
      {.operation_fallback = OperationFallbackPolicy::kReject}, adapters);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(original.compile_count, 1u);
  EXPECT_EQ(adapted.compile_count, 1u);
  EXPECT_EQ(original.propose_count, 1u);
  EXPECT_EQ(adapted.propose_count, 1u);
  // Compile-local adapter state is not a runtime owner or dispatch table.
  adapters = {};
  for (int i = 0; i != 2; ++i)
    expect_negation_replay(compiled->executable, logical, context);
  EXPECT_EQ(original.compile_count, 1u);
  EXPECT_EQ(adapted.compile_count, 1u);
  EXPECT_EQ(original.propose_count, 1u);
  EXPECT_EQ(adapted.propose_count, 1u);
}

TEST(GraphRouteBoundary, CapabilityAdaptersShareTheOriginalProviderProposalCap) {
  const auto logical = make_negations(3);
  RegionProbe original{.mode = ProposalMode::kPairs};
  RegionProbe adapted{.mode = ProposalMode::kSingleton};
  adapted.singleton_node = 2;
  auto context = make_context(original);
  const Environment environment{{&context, hyle::mem::system_resource()}};
  const std::array adapters{hyle::graph::detail::RegionAdapter{
      .source = hyle::graph::detail::GraphRegionSource::shared(
          environment.environment().targets[environment.target(0).ordinal()].source,
          hyle::compute::detail::ProviderExtensionAccess::query(
              context, 0, hyle::graph::kRegionProviderExtensionId).value()).value(),
      .provider_index = 0,
      .target = environment.target(0),
      .interface = &kRegionVTable,
      .context = &adapted}};
  auto compiled = hyle::graph::detail::compile_with_adapters(
      logical.graph, environment.environment(),
      {.operation_fallback = OperationFallbackPolicy::kReject,
       .max_region_proposals_per_provider = 1,
       .search_policy = CompileSearchPolicy::kRequireUnexhausted}, adapters);
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(original.propose_count, 1u);
  EXPECT_EQ(adapted.propose_count, 0u);
  EXPECT_EQ(original.compile_count + adapted.compile_count, 0u);
}

TEST(GraphRouteBoundary, CapabilityAdapterRequiresExactContextAndInstance) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton};
  RegionProbe second{.mode = ProposalMode::kSingleton};
  auto context = make_context(first, second);
  RegionProbe other{.mode = ProposalMode::kSingleton};
  auto other_context = make_context(other);
  const Environment environment{{&context, hyle::mem::system_resource()}};
  auto wrong_context = hyle::compute::detail::ProviderExtensionAccess::query(
      other_context, 0, hyle::graph::kRegionProviderExtensionId).value();
  auto wrong_instance = hyle::compute::detail::ProviderExtensionAccess::query(
      context, 1, hyle::graph::kRegionProviderExtensionId).value();
  const auto wrong_context_source = hyle::graph::detail::GraphRegionSource::shared(
      hyle::graph::ExecutionSource::acquire(other_context).value(),
      std::move(wrong_context)).value();
  const auto wrong_instance_source = hyle::graph::detail::GraphRegionSource::shared(
      environment.environment().targets[environment.target(0).ordinal()].source,
      std::move(wrong_instance)).value();
  for (const auto &source : {wrong_context_source, wrong_instance_source}) {
    const std::array adapters{hyle::graph::detail::RegionAdapter{
        .source = source,
        .provider_index = 0,
        .target = environment.target(0),
        .interface = &kRegionVTable,
        .context = &first}};
    auto compiled = hyle::graph::detail::compile_with_adapters(
        logical.graph, environment.environment(), {}, adapters);
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsInvalidArgument());
    EXPECT_EQ(first.propose_count + second.propose_count + other.propose_count, 0u);
    EXPECT_EQ(first.compile_count + second.compile_count + other.compile_count, 0u);
  }
}

TEST(GraphResources, FixedBindingIndexKeepsRebindGenerationAndResultLeaseGuards) {
  constexpr std::size_t count = 16;
  const std::array providers{hyle::compute::ProviderBinding{
      hyle::compute::ggml_cpu::provider(), 100}};
  auto context_result = Context::create(hyle::compute::cpu::device(),
                                        {.providers = providers});
  ASSERT_TRUE(context_result) << context_result.error().message();
  auto context = std::move(*context_result);
  auto queue_result = context.create_queue();
  ASSERT_TRUE(queue_result) << queue_result.error().message();
  auto queue = std::move(*queue_result);
  auto builder = GraphBuilder::create().value();
  std::array<ValueId, count> inputs{}, outputs{};
  for (std::size_t i = 0; i < count; ++i) {
    inputs[i] = builder.input(row(8)).value();
    outputs[i] = builder.add(Unary{UnaryKind::kNeg}, inputs[i]).value();
    ASSERT_TRUE(builder.publish(outputs[i]).ok());
  }
  ASSERT_TRUE(builder.publish(inputs[0]).ok()); // Input/output identity is one fixed record.
  auto graph_result = std::move(builder).finish();
  ASSERT_TRUE(graph_result) << graph_result.error().message();
  auto graph = std::move(*graph_result);
  auto foreign_builder = GraphBuilder::create().value();
  const auto foreign_first = foreign_builder.input(row(8)).value();
  (void)foreign_builder.add(Unary{UnaryKind::kNeg}, foreign_first).value();
  const auto foreign = foreign_builder.input(row(8)).value();
  ASSERT_EQ(foreign.ordinal(), inputs[1].ordinal());

  std::array<OwnedTensor, count> output_storage;
  std::array<hyle::compute::LeasePool, count> first_pools, second_pools;
  std::array<hyle::compute::ReadLease, count> first, second;
  const std::array<float, 8> low{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<float, 8> high{11, 12, 13, 14, 15, 16, 17, 18};
  auto borrowed = make_tensor(row(8));
  write(borrowed, low);
  for (std::size_t i = 0; i < count; ++i) {
    output_storage[i] = make_tensor(row(8));
    first_pools[i] = hyle::compute::LeasePool::create(
        hyle::mem::system_resource(), 32).value();
    second_pools[i] = hyle::compute::LeasePool::create(
        hyle::mem::system_resource(), 32).value();
    auto one = first_pools[i].allocate(32, 64).value();
    auto two = second_pools[i].allocate(32, 64).value();
    ASSERT_TRUE(one.view()->write(std::as_bytes(std::span(low))).ok());
    ASSERT_TRUE(two.view()->write(std::as_bytes(std::span(high))).ok());
    first[i] = one.publish().value();
    second[i] = two.publish().value();
  }
  auto compiled_result = hyle::graph::compile(graph, context,
      {.region_selection = RegionSelectionPolicy::kDisabled});
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  auto invocation_result = compiled.executable.create_invocation(
      {.resources = hyle::graph::ResourceOptions{}});
  ASSERT_TRUE(invocation_result) << invocation_result.error().message();
  auto run = std::move(*invocation_result);
  for (unsigned round = 0; round < 4; ++round) {
    SCOPED_TRACE(round);
    if (round != 0) ASSERT_TRUE(run.reset().ok());
    auto &selected = round % 2 ? second : first;
    const auto &expected = round % 2 ? high : low;
    for (std::size_t remaining = count; remaining != 0; --remaining) {
      const auto i = remaining - 1;
      ASSERT_TRUE(run.bind(inputs[i], row(8), selected[i]).ok());
      ASSERT_TRUE(run.bind(outputs[i], output_storage[i].view).ok());
    }
    ASSERT_TRUE(hyle::graph::detail::RegionAccess::bind_configuration(
        run, inputs[1], row(8), selected[1]).ok());
    // The replacement guard runs before the ordinary foreign-ValueId check.
    // An ordinal-only lookup would incorrectly report FailedPrecondition.
    EXPECT_TRUE(run.bind(foreign, borrowed.view.as_const()).IsInvalidArgument());
    EXPECT_TRUE(run.bind(inputs[1], borrowed.view.as_const()).IsFailedPrecondition());
    ASSERT_TRUE(run.submit(queue)->wait().ok());
    for (const auto &output : output_storage) {
      auto actual = read(output, 8);
      for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], -expected[i]);
    }
    auto result_status = run.result(inputs[0]);
    ASSERT_TRUE(result_status) << result_status.error().message();
    auto result = std::move(*result_status);
    auto old_result = result;
    auto exported_status = result.storage_lease();
    ASSERT_TRUE(exported_status) << exported_status.error().message();
    auto exported = std::move(*exported_status);
    EXPECT_TRUE(run.bind(inputs[0], borrowed.view.as_const()).IsUnavailable());
    EXPECT_TRUE(run.reset().IsUnavailable());
    result.reset();
    EXPECT_TRUE(run.reset().IsUnavailable());
    old_result.reset();
    ASSERT_TRUE(run.bind(inputs[0], borrowed.view.as_const()).ok());
    // Rebinding releases the old fixed permission; the exported lease still
    // independently protects its backing until explicitly released.
    EXPECT_FALSE(selected[0].try_write());
    exported.reset();
    auto writable = selected[0].try_write();
    ASSERT_TRUE(writable) << writable.error().message();
    selected[0] = writable->publish().value();
  }
}

TEST(GraphRouteBoundary, ProjectedRegionRequiresItsSelectedArtifactAndPhysicalPort) {
  const auto logical = make_negations(1);
  RegionProbe probe{.mode = ProposalMode::kSingleton};
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject}).value();
  auto metadata = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  const hyle::graph::ResourceOptions resources{.allocator = metadata.get()};
  auto run = compiled.executable.create_invocation({.resources = resources}).value();
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::reserve_reads(run, 4, resources).ok());
  const auto cold_allocations = metadata.snapshot().total_allocs;
  auto pool = hyle::compute::LeasePool::create(hyle::mem::system_resource(),
      1024, {.max_blocks = 2}).value();
  auto writer = pool.allocate(32).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{0}).ok());
  auto input = writer.publish().value();
  auto tensor = input.tensor(row(8)).value();
  std::array<hyle::compute::ReadLease, 3> pages{
      input.subview(0, 8).value(), input.subview(8, 8).value(),
      input.subview(16, 16).value()};
  hyle::graph::detail::ProjectedReader selected;
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::visit(run,
      [](void *opaque, const hyle::graph::detail::RegionAccessView &view) noexcept {
        *static_cast<hyle::graph::detail::ProjectedReader *>(opaque) = {
            view.step, view.prepared_vtable, view.prepared, 0};
        return Status::Ok();
      }, &selected).ok());
  for (int change = 0; change != 3; ++change) {
    auto wrong = selected;
    if (change == 0) wrong.implementation = &kResourceRegionVTable;
    if (change == 1) wrong.prepared = &probe;
    if (change == 2) wrong.input_port = 1;
    const std::array readers{wrong};
    const auto status = hyle::graph::detail::RegionAccess::bind_projected(
        run, logical.input, tensor, pages, readers);
    EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
  }
  const std::array duplicated{selected, selected};
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::bind_projected(
      run, logical.input, tensor, pages, duplicated).IsInvalidArgument());
  const std::array readers{selected};
  // An addressable tensor and exact artifact identity do not prove that this
  // actual negation reads no bytes. The selected implementation must expose
  // and qualify an empty footprint before core can omit its ready permission.
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::bind_projected(
      run, logical.input, tensor, {}, readers).IsUnimplemented());
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::bind_projected(
      run, logical.input, tensor, pages, readers).ok());
  auto output = pool.allocate(32).value();
  ASSERT_TRUE(run.bind(logical.output, row(8), std::move(output)).ok());
  auto replacement = make_tensor(row(8));
  const std::array<float, 8> changed{1, 2, 3, 4, 5, 6, 7, 8};
  write(replacement, changed);
  // Actual anonymous suffix guards exist here. Ordinary rebinding must erase
  // only this input's pages while keeping the fixed output record in place.
  ASSERT_TRUE(run.bind(logical.input, replacement.view.as_const()).ok());
  pages = {};
  auto restored = input.try_write();
  ASSERT_TRUE(restored) << restored.error().message();
  input = restored->publish().value();
  tensor = input.tensor(row(8)).value();
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::reserve_reads(run, 4, resources).ok());
  EXPECT_EQ(metadata.snapshot().total_allocs, cold_allocations);
  auto queue = context.create_queue().value();
  auto completion = run.submit(queue);
  ASSERT_TRUE(completion) << completion.error().message();
  EXPECT_TRUE(completion->wait().ok());
  {
    auto result = run.result(logical.output).value();
    std::array<float, 8> observed{};
    ASSERT_TRUE(result.tensor()->storage().read(
        std::as_writable_bytes(std::span(observed))).ok());
    for (std::size_t i = 0; i < observed.size(); ++i) EXPECT_EQ(observed[i], -changed[i]);
  }
  *completion = {};
  ASSERT_TRUE(run.reset().ok());
  pages = {input.subview(0, 8).value(), input.subview(8, 8).value(),
           input.subview(16, 16).value()};
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::bind_projected(
      run, logical.input, tensor, pages, readers).ok());
  pages = {};
  // Reset must also release a nonempty suffix without moving fixed slots.
  ASSERT_TRUE(run.reset().ok());
  auto after_reset = input.try_write();
  ASSERT_TRUE(after_reset) << after_reset.error().message();
}

TEST(GraphRouteBoundary, ProjectedInputRetainsAllReadersAndLivePageGuards) {
  auto builder = GraphBuilder::create().value();
  const auto bank = builder.input(row(8)).value();
  const auto side = builder.input(row(8)).value();
  const auto first_output = builder.add(Unary{UnaryKind::kNeg}, bank).value();
  auto tail = side;
  for (std::size_t i = 0; i != 64; ++i)
    tail = builder.add(Unary{UnaryKind::kNeg}, tail).value();
  const auto second_output = builder.add(Unary{UnaryKind::kNeg}, bank).value();
  for (const auto output : {first_output, tail, second_output})
    ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton, .singleton_node = 65};
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 2U);
  auto run_status = compiled->executable.create_invocation(
      {.resources = hyle::graph::ResourceOptions{}});
  ASSERT_TRUE(run_status) << run_status.error().message();
  auto run = std::move(*run_status);
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::reserve_reads(run, 2, {}).ok());
  auto pool = hyle::compute::LeasePool::create(hyle::mem::system_resource(),
      1024, {.max_blocks = 2}).value();
  auto writer = pool.allocate(32).value();
  const std::array<float, 8> values{1, -2, 3, -4, 5, -6, 7, -8};
  ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());
  auto input = writer.publish().value();
  const auto tensor = input.tensor(row(8)).value();
  const std::array pages{input.subview(0, 16).value(), input.subview(16, 16).value()};
  struct Readers final {
    std::array<hyle::graph::detail::ProjectedReader, 2> values{};
    std::size_t size = 0;
  } readers;
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::visit(run,
      [](void *opaque, const hyle::graph::detail::RegionAccessView &view) noexcept {
        auto &selected = *static_cast<Readers *>(opaque);
        if (selected.size == selected.values.size())
          return Status::Internal("unexpected projected test reader");
        selected.values[selected.size++] = {
            view.step, view.prepared_vtable, view.prepared, 0};
        return Status::Ok();
      }, &readers).ok());
  ASSERT_EQ(readers.size, 2U);
  const auto incomplete = hyle::graph::detail::RegionAccess::bind_projected(
      run, bank, tensor, pages, std::span(readers.values).first(1));
  EXPECT_TRUE(incomplete.IsInvalidArgument()) << incomplete.message();
  EXPECT_EQ(incomplete.message(), "projected capacity has an unqualified physical reader");
  auto foreign = pool.allocate(32).value().publish().value();
  const std::array foreign_pages{foreign};
  const auto wrong_page = hyle::graph::detail::RegionAccess::bind_projected(
      run, bank, tensor, foreign_pages, readers.values);
  EXPECT_TRUE(wrong_page.IsInvalidArgument()) << wrong_page.message();
  EXPECT_EQ(wrong_page.message(), "projected page permission is outside its native capacity");
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::bind_projected(
      run, bank, tensor, pages, readers.values).ok());
  auto side_tensor = make_tensor(row(8));
  write(side_tensor, values);
  std::array outputs{make_tensor(row(8)), make_tensor(row(8)), make_tensor(row(8))};
  ASSERT_TRUE(run.bind(side, side_tensor.view.as_const()).ok());
  ASSERT_TRUE(run.bind(first_output, outputs[0].view).ok());
  ASSERT_TRUE(run.bind(tail, outputs[1].view).ok());
  ASSERT_TRUE(run.bind(second_output, outputs[2].view).ok());
  auto queue = context.create_queue().value();
  auto completed = run.submit(queue);
  ASSERT_TRUE(completed) << completed.error().message();
  ASSERT_TRUE(completed->wait().ok());
  for (std::size_t output = 0; output != outputs.size(); ++output) {
    const auto observed = read(outputs[output], values.size());
    for (std::size_t i = 0; i != values.size(); ++i)
      EXPECT_EQ(observed[i], output == 1 ? values[i] : -values[i]);
  }

  // A real boundary transfer is another reader of the same logical bank.
  // Selecting only the root's region cannot authorize that copied replica.
  HostResource replica_storage;
  RegionProbe root_reader{.mode = ProposalMode::kSingleton},
      replica_reader{.mode = ProposalMode::kSingleton, .singleton_node = 65};
  root_reader.region_input_resource = root_reader.region_output_resource =
      hyle::mem::system_resource();
  replica_reader.region_input_resource = replica_reader.region_output_resource =
      hyle::mem::MemoryResourceRef(replica_storage);
  auto copy_context = make_context(root_reader, replica_reader);
  const Environment environment{
      {&copy_context, hyle::mem::system_resource()},
      {&copy_context, hyle::mem::MemoryResourceRef(replica_storage)}};
  const std::array contracts{
      hyle::graph::ExternalResourceContract{bank, environment.storage(0)},
      hyle::graph::ExternalResourceContract{second_output, environment.storage(1)}};
  auto copied = hyle::graph::compile(graph, environment.environment(),
      {.external_resources = contracts});
  ASSERT_TRUE(copied) << copied.error().message();
  ASSERT_EQ(copied->report.region_count(), 2U);
  ASSERT_GT(copied->report.copy_bytes(), 0U);
  auto copied_status = copied->executable.create_invocation(
      {.resources = hyle::graph::ResourceOptions{}});
  ASSERT_TRUE(copied_status) << copied_status.error().message();
  auto copied_run = std::move(*copied_status);
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::reserve_reads(copied_run, 2, {}).ok());
  Readers copied_readers;
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::visit(copied_run,
      [](void *opaque, const hyle::graph::detail::RegionAccessView &view) noexcept {
        auto &selected = *static_cast<Readers *>(opaque);
        if (view.node_ordinals.size() == 1 && view.node_ordinals[0] == 0)
          selected.values[selected.size++] = {
              view.step, view.prepared_vtable, view.prepared, 0};
        return Status::Ok();
      }, &copied_readers).ok());
  ASSERT_EQ(copied_readers.size, 1U);
  const auto copied_rejection = hyle::graph::detail::RegionAccess::bind_projected(
      copied_run, bank, tensor, pages, std::span(copied_readers.values).first(1));
  EXPECT_TRUE(copied_rejection.IsInvalidArgument()) << copied_rejection.message();
  EXPECT_EQ(copied_rejection.message(), "projected capacity has an unqualified physical reader");
}

TEST(GraphRouteBoundary, ContextRoutesKeepExactOwnersAndResourceChecks) {
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
    const std::array placement{hyle::graph::NodePlacement{
        logical.graph.node_id(0).value(), environment.target(0),
        hyle::graph::PlacementStrength::kRequired}};
    auto compiled = hyle::graph::compile(
        logical.graph, environment.environment(),
        {.node_placements = placement,
         .operation_fallback = OperationFallbackPolicy::kReject});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->report.target_count(), 2u);
    EXPECT_EQ(compiled->report.storage_domain_count(), 2u);
    expect_negation_replay(compiled->executable, logical, first_context);

    auto queue = first_context.create_queue().value();
    auto foreign_queue = second_context.create_queue().value();
    EXPECT_TRUE(hyle::graph::detail::RegionAccess::validate_queue(
        compiled->executable, environment.lane(0), queue).ok());
    EXPECT_TRUE(hyle::graph::detail::RegionAccess::validate_queue(
        compiled->executable, environment.lane(0), foreign_queue)
                    .IsInvalidArgument());

    const auto discoveries = first.propose_count + second.propose_count;
    const auto preparations = first.compile_count + second.compile_count;
    const auto source =
        hyle::graph::ExecutionSource::acquire(first_context).value();
    const std::array duplicate_targets{
        hyle::graph::ExecutionTarget{source, environment.storage(0)},
        hyle::graph::ExecutionTarget{source, environment.storage(0)}};
    auto duplicate_environment = environment.environment();
    duplicate_environment.targets = duplicate_targets;
    auto duplicate = hyle::graph::compile(logical.graph, duplicate_environment);
    ASSERT_FALSE(duplicate);
    EXPECT_TRUE(duplicate.error().IsInvalidArgument());

    HostResource inaccessible(false);
    const Environment invalid_resource{
        {&first_context, hyle::mem::system_resource()},
        {&second_context, hyle::mem::MemoryResourceRef(inaccessible)}};
    EXPECT_FALSE(hyle::graph::compile(logical.graph,
                                      invalid_resource.environment()));
    EXPECT_EQ(first.propose_count + second.propose_count, discoveries);
    EXPECT_EQ(first.compile_count + second.compile_count, preparations);
    EXPECT_EQ(alternate.live_allocations, 0u);
  }
}

TEST(GraphRouteBoundary, DifferentDevicesUseTheSelectedSourceCapabilities) {
  RegionProbe first{.device_identity = 1, .mode = ProposalMode::kSingleton,
                    .asynchronous_backend = true},
      second{.device_identity = 2, .mode = ProposalMode::kSingleton,
             .asynchronous_backend = true};
  auto first_context = make_context(first);
  auto second_context = make_context(second);
  ASSERT_EQ(first_context.device().backend_id(),
            second_context.device().backend_id());
  ASSERT_NE(first_context.device(), second_context.device());
  const auto logical = make_negations(1);
  const Environment environment{
      {&first_context, hyle::mem::system_resource()},
      {&second_context, hyle::mem::system_resource()}};
  const std::array placement{hyle::graph::NodePlacement{
      logical.graph.node_id(0).value(), environment.target(1),
      hyle::graph::PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(
      logical.graph, environment.environment(),
      {.node_placements = placement,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->target, environment.target(1));
  expect_negation_replay(compiled->executable, logical, second_context);
  EXPECT_EQ(first.submit_count, 0u);
  EXPECT_EQ(second.submit_count, 1u);

  auto queue = second_context.create_queue().value();
  auto foreign_queue = first_context.create_queue().value();
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::validate_queue(
      compiled->executable, environment.lane(1), queue).ok());
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::validate_queue(
      compiled->executable, environment.lane(1), foreign_queue)
                  .IsInvalidArgument());

  const auto discoveries = first.propose_count + second.propose_count;
  second.backend_only_resource = hyle::mem::system_resource();
  HostResource foreign_resource;
  const Environment inaccessible{
      {&first_context, hyle::mem::system_resource()},
      {&second_context, hyle::mem::MemoryResourceRef(foreign_resource)}};
  auto rejected = hyle::graph::compile(logical.graph, inaccessible.environment());
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(rejected.error().message(), "test backend cannot access this storage");
  EXPECT_EQ(first.propose_count + second.propose_count, discoveries);
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
  {
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
  } // Release retained execution sources before the public Context.
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

TEST(GraphRegions, ColdRegionAccessUsesActualLaneAndExactContextQueue) {
  const auto logical = make_negations(4);
  RegionProbe first, second;
  auto first_context = make_context(first), second_context = make_context(second);
  auto first_queue = first_context.create_queue().value();
  auto second_queue = second_context.create_queue().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{&first_context, host}, {&second_context, host}};
  const std::array placements{
      hyle::graph::NodePlacement{logical.graph.node_id(0).value(), environment.target(0), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{logical.graph.node_id(1).value(), environment.target(0), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{logical.graph.node_id(2).value(), environment.target(1), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{logical.graph.node_id(3).value(), environment.target(1), hyle::graph::PlacementStrength::kRequired}};
  auto compiled = hyle::graph::compile(logical.graph, environment.environment(),
      {.node_placements = placements, .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->executable.required_queue_count(), 2u);
  using Access = hyle::graph::detail::RegionAccess;
  EXPECT_TRUE(Access::validate_queue(compiled->executable, environment.lane(0), first_queue).ok());
  EXPECT_TRUE(Access::validate_queue(compiled->executable, environment.lane(1), second_queue).ok());
  EXPECT_TRUE(Access::validate_queue(compiled->executable, environment.lane(0), second_queue).IsInvalidArgument());
  EXPECT_TRUE(Access::validate_queue(compiled->executable, hyle::graph::ExecutionLaneId{2}, first_queue).IsInvalidArgument());
  struct Probe { const CompileReport* report; std::array<bool, 2> lanes{}; std::size_t count=0; } probe{&compiled->report};
  const auto visit = +[](void* opaque, const hyle::graph::detail::RegionAccessView& view) noexcept -> Status {
    auto& probe = *static_cast<Probe*>(opaque);
    HYLE_ASSIGN_OR_RETURN(auto step, probe.report->step(view.step));
    if (view.lane != step.lane || !view.lane.valid() || view.lane.ordinal() >= probe.lanes.size())
      return Status::Internal("region visitor lost its original execution lane");
    probe.lanes[view.lane.ordinal()] = true;
    ++probe.count;
    return Status::Ok();
  };
  ASSERT_TRUE(Access::visit(compiled->executable, visit, &probe).ok());
  EXPECT_EQ(probe.count, 2u);
  EXPECT_TRUE(probe.lanes[0] && probe.lanes[1]);
  auto invocation = compiled->executable.create_invocation().value();
  probe.count = 0;
  probe.lanes = {};
  ASSERT_TRUE(Access::visit(invocation, visit, &probe).ok());
  EXPECT_EQ(probe.count, 2u);
  EXPECT_TRUE(probe.lanes[0] && probe.lanes[1]);
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

struct SliceAlignmentGraph final {
  Graph graph;
  ValueId input, output;
};

SliceAlignmentGraph make_slice_alignment_graph() {
  auto builder = GraphBuilder::create().value();
  const auto input = builder.input(row(85)).value();
  const auto root = builder.add(hyle::compute::Binary{}, input, input).value();
  // A real materialized F32 root with the 68-byte row offset of [5,17].
  const auto selected = builder.add(
      hyle::compute::View::slice(0, 17, 17).value(), root).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, selected).value();
  EXPECT_TRUE(builder.publish(output).ok());
  return {std::move(builder).finish().value(), input, output};
}

TEST(GraphRegions, InputSliceAlignmentDeclinesRegionAndLeafBeforePreparation) {
  const auto logical = make_slice_alignment_graph();
  for (const bool leaf : {false, true}) {
    SCOPED_TRACE(leaf);
    RegionProbe strict{.mode = ProposalMode::kSingleton, .singleton_node = 2};
    RegionProbe compatible{.mode = ProposalMode::kSingleton, .singleton_node = 2};
    for (auto *probe : {&strict, &compatible}) {
      probe->required_resource = hyle::mem::system_resource();
      probe->bounded_unary_domain = true;
    }
    strict.requirement_override = hyle::compute::ResourceRequirement{
        .minimum_alignment = 256};
    compatible.requirement_override = hyle::compute::ResourceRequirement{
        .minimum_alignment = alignof(float)};
    auto context = make_context(strict, compatible);
    CompileOptions options{.region_selection = leaf
        ? RegionSelectionPolicy::kDisabled
        : RegionSelectionPolicy::kAllowProviderRegions};
    auto compiled = hyle::graph::compile(logical.graph, context, options);
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(strict.compile_count, 0u);
    EXPECT_EQ(strict.single_artifact_count, 0u);
    EXPECT_EQ(compiled->report.region_count(), leaf ? 0u : 1u);
    EXPECT_GT(leaf ? compiled->report.search().operation_route_rejections
                   : compiled->report.search().region_route_rejections, 0u);
    bool selected_compatible = false;
    for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
      const auto step = compiled->report.step(i).value();
      if (step.node_begin == 2) {
        EXPECT_EQ(step.provider_instance, context.provider_instance_id(1).value());
        selected_compatible = true;
      }
    }
    EXPECT_TRUE(selected_compatible);
    auto input = make_tensor(row(85)), output = make_tensor(row(17));
    std::vector<float> values(85), expected(17);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i + 1);
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = -2 * values[i + 17];
    write(input, values);
    auto run = compiled->executable.create_invocation().value();
    ASSERT_TRUE(run.bind(logical.input, input.view).ok());
    ASSERT_TRUE(run.bind(logical.output, output.view).ok());
    auto queue = context.create_queue().value();
    auto done = run.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    EXPECT_EQ(read(output, 17), expected);

    // A required instance is a hard constraint, not permission to use the
    // lower-alignment alternate after the selected candidate is declined.
    const std::array required{hyle::graph::NodeProviderConstraint{
        logical.graph.node_id(2).value(), kRegionProviderId,
        context.provider_instance_id(0).value()}};
    options.node_providers = required;
    auto rejected = hyle::graph::compile(logical.graph, context, options);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented()) << rejected.error().message();
    EXPECT_EQ(strict.compile_count, 0u);
    EXPECT_EQ(strict.single_artifact_count, 0u);
  }
}

TEST(GraphRegions, CopiedSliceGetsItsOwnAlignedInputBeforeCandidateQualification) {
  const auto logical = make_slice_alignment_graph();
  HostResource source, destination;
  RegionProbe strict{.mode = ProposalMode::kSingleton, .singleton_node = 2};
  strict.requirement_override = hyle::compute::ResourceRequirement{
      .minimum_alignment = 256,
      .exact_resource = hyle::mem::MemoryResourceRef(destination)};
  strict.region_output_resource = hyle::mem::MemoryResourceRef(destination);
  auto context = make_context(strict);
  const Environment environment{{&context, hyle::mem::MemoryResourceRef(source)},
                               {&context, hyle::mem::MemoryResourceRef(destination)}};
  const std::array external{
      hyle::graph::ExternalResourceContract{logical.input, environment.storage(0)},
      hyle::graph::ExternalResourceContract{logical.output, environment.storage(1)}};
  auto compiled = hyle::graph::compile(logical.graph, environment.environment(),
                                      {.external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(strict.compile_count, 1u);
  EXPECT_EQ(strict.compiled_input_resource,
            hyle::mem::MemoryResourceRef(destination));
  EXPECT_GT(compiled->report.copy_bytes(), 0u);
  auto input = make_tensor(row(85), hyle::mem::MemoryResourceRef(source));
  auto output = make_tensor(row(17), hyle::mem::MemoryResourceRef(destination));
  std::vector<float> values(85), expected(17);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i + 1);
  for (std::size_t i = 0; i < expected.size(); ++i)
    expected[i] = -2 * values[i + 17];
  write(input, values);
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  auto done = run.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  EXPECT_EQ(read(output, 17), expected);
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
      hyle::graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(a).value(), hyle::graph::StorageDomainId{0}},
      hyle::graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(b).value(), hyle::graph::StorageDomainId{0}}};
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

TEST(GraphRegions, RejectsChangedOutputCarrierBeforeSlotCreation) {
  RegionProbe probe;
  probe.proposed_output_contract = hyle::graph::RegionOutputContract{
      .kind = hyle::graph::RegionOutputKind::kStableGenerated};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto changed = hyle::graph::compile(logical.graph, context);
  ASSERT_FALSE(changed.has_value());
  EXPECT_TRUE(changed.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.destroy_prepared_count, 1u);
  EXPECT_EQ(probe.create_invocation_count, 0u);
  EXPECT_EQ(probe.submit_count, 0u);

  // A missing callback means Caller/TensorView, matching the default proposal.
  probe.proposed_output_contract.reset();
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1u);
  auto invocation = compiled->executable.create_invocation().value();
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

TEST(GraphRegions, BorrowedCaptureLaunchUsesOriginalAdmissionAndRejectsForeignContext) {
  namespace cd=hyle::compute::detail;
  RegionProbe probe{.mode=ProposalMode::kAsyncPairs};probe.capture_supported=true;
  Context context=make_context(probe);Queue queue=context.create_queue().value();
  auto extension=cd::CommandCaptureAccess::query(queue);ASSERT_TRUE(extension);
  auto recording=cd::QueueSequence::begin(queue);ASSERT_TRUE(recording);
  EXPECT_FALSE(recording->recording());
  ASSERT_TRUE(cd::CommandCaptureAccess::begin(*extension,*recording).ok());
  EXPECT_TRUE(recording->recording());
  auto premature = recording->finish(); ASSERT_FALSE(premature);
  EXPECT_TRUE(premature.error().IsFailedPrecondition());
  EXPECT_TRUE(recording->valid());
  auto moved = std::move(*recording);
  EXPECT_FALSE(recording->recording());
  EXPECT_TRUE(moved.recording());
  auto captured=cd::CommandCaptureAccess::end(*extension,moved,hyle::mem::system_allocator());
  EXPECT_FALSE(moved.recording());
  ASSERT_TRUE(captured);auto recorded=moved.finish();ASSERT_TRUE(recorded);
  ASSERT_TRUE(recorded->wait().ok());
  RegionProbe foreign_probe{.mode=ProposalMode::kAsyncPairs};foreign_probe.capture_supported=true;
  Context foreign=make_context(foreign_probe);Queue foreign_queue=foreign.create_queue().value();
  auto foreign_sequence=cd::QueueSequence::begin(foreign_queue);ASSERT_TRUE(foreign_sequence);
  const auto rejected=captured->enqueue(*foreign_sequence);
  EXPECT_TRUE(rejected.status.IsInvalidArgument());EXPECT_EQ(probe.capture_launch_count,0u);
  auto foreign_done=foreign_sequence->finish();ASSERT_TRUE(foreign_done);ASSERT_TRUE(foreign_done->wait().ok());
  auto admitted=cd::QueueSequence::begin(queue);ASSERT_TRUE(admitted);
  admitted->record(captured->enqueue(*admitted));
  auto done=admitted->finish();ASSERT_TRUE(done);ASSERT_TRUE(done->wait().ok());
  EXPECT_EQ(probe.capture_launch_count,1u);captured->reset();EXPECT_EQ(probe.capture_destroy_count,1u);
  {
    auto abandoned = cd::QueueSequence::begin(queue); ASSERT_TRUE(abandoned);
    ASSERT_TRUE(cd::CommandCaptureAccess::begin(*extension, *abandoned).ok());
  }
  EXPECT_FALSE(probe.capture_active);
  EXPECT_EQ(probe.capture_abort_count, 1u);
}

TEST(GraphRegions, OrdinaryColdBindingsRetryWithoutExecutingAndTrackRebinds) {
  RegionProbe probe;
  probe.required_resource = hyle::mem::system_resource();
  probe.decline_regions = true;
  probe.require_cold_bindings = true;
  const auto logical = make_negations(1);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 0u);
  auto run = compiled.executable.create_invocation().value();
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  auto next_input = make_tensor(row(8)), next_output = make_tensor(row(8));
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  const std::array next_values{8.F, -7.F, 6.F, -5.F, 4.F, -3.F, 2.F, -1.F};
  const std::array sentinel{19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F};
  write(input, values); write(next_input, next_values); write(output, sentinel);
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  probe.binding_prepare_failure = Status::ResourceExhausted("cold binding OOM");
  const auto rejected = run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kPreferCapture});
  EXPECT_TRUE(rejected.IsResourceExhausted());
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_EQ(probe.capture_begin_count, 0u);
  EXPECT_EQ(probe.submit_count, 0u);
  EXPECT_EQ(read(output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));
  probe.binding_prepare_failure = Status::Ok();
  ASSERT_TRUE(run.prepare_replay(queues).ok());
  EXPECT_EQ(probe.binding_prepare_count, 2u);
  ASSERT_TRUE(run.prepare_replay(queues).ok());
  EXPECT_EQ(probe.binding_prepare_count, 2u);
  ASSERT_TRUE(run.submit(queues)->wait().ok());
  EXPECT_EQ(read(output, 8), (std::vector<float>{-1, 2, -3, 4, -5, 6, -7, 8}));
  auto held = run.result(logical.output).value();
  EXPECT_TRUE(run.bind(logical.input, next_input.view).IsUnavailable());
  held.reset();
  ASSERT_TRUE(run.bind(logical.input, next_input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, next_output.view).ok());
  ASSERT_TRUE(run.prepare_replay(queues).ok());
  EXPECT_EQ(probe.binding_prepare_count, 3u);
  ASSERT_TRUE(run.submit(queues)->wait().ok());
  EXPECT_EQ(read(next_output, 8), (std::vector<float>{-8, 7, -6, 5, -4, 3, -2, 1}));
  EXPECT_EQ(read(output, 8), (std::vector<float>{-1, 2, -3, 4, -5, 6, -7, 8}));
  EXPECT_EQ(probe.binding_prepare_count, 3u);
  ASSERT_TRUE(run.reset().ok());
}

TEST(GraphRegions, OrdinaryBindingsAreColdBeforeOriginalCommandCapture) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.required_resource = hyle::mem::system_resource();
  probe.decline_regions = true;
  probe.require_cold_bindings = true;
  probe.cold_bindings_capture_compatible = true;
  probe.capture_supported = true;
  auto context = make_context(probe);
  const auto logical = make_negations(1);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  auto run = compiled.executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.binding_prepare_count, 1u);
  EXPECT_EQ(probe.submit_count, 0u);
  EXPECT_EQ(probe.capture_begin_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 1u);
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kCaptured);
  ASSERT_TRUE(run.submit(queues)->wait().ok());
  EXPECT_EQ(probe.capture_launch_count, 1u);
  EXPECT_EQ(probe.binding_prepare_count, 1u);
  // Commands must retire the recording before invoking the ordinary encoder.
  // Unlike the mock captured replay above, this path executes real CPU math.
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  const std::array sentinel{19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F};
  write(input, values);
  write(output, sentinel);
  probe.required_capture_destroys_before_submit = 1;
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kCommands}).ok());
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
  EXPECT_EQ(read(output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));
  ASSERT_TRUE(run.submit(queues)->wait().ok());
  EXPECT_EQ(read(output, 8), (std::vector<float>{-1, 2, -3, 4, -5, 6, -7, 8}));
  EXPECT_EQ(probe.capture_launch_count, 1u);
}

TEST(GraphRegions, CapturesCompatibleRangesAroundAnUncapturableCommandInOneSequence) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  // Mock captures and the command gap are control-only: no synthetic replay
  // pretends to produce the preceding range's tensor payload.
  probe.capture_control_only = true;
  // The exact middle program narrows its cold proposal's capture upper bound.
  probe.uncapturable_region_begin = 2;
  auto context = make_context(probe);
  const auto logical = make_negations(6);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 3u);
  ASSERT_EQ(compiled.report.command_segment_count(), 1u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  auto run = compiled.executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  EXPECT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).IsUnimplemented());
  EXPECT_EQ(probe.capture_begin_count, 0u);
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kPreferCapture}).ok());
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kHybrid);
  EXPECT_EQ(probe.capture_begin_count, 2u);
  EXPECT_EQ(probe.capture_end_count, 2u);
  EXPECT_EQ(probe.submit_count, 0u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 2u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
  // This backend's mock captures prove ordering/coverage, not native math.
  const auto waits = probe.completion_wait_count;
  auto done = run.submit(queues); ASSERT_TRUE(done);
  EXPECT_EQ(probe.capture_launch_count, 2u);
  EXPECT_EQ(probe.submit_attempt_count, 3u);
  EXPECT_EQ(probe.submit_count, 3u);
  EXPECT_EQ(probe.completion_wait_count, waits);
  EXPECT_EQ(probe.last_reserved_completions, 1u);
  ASSERT_TRUE(done->wait().ok());
  *done = {};
  const auto prepare_again = [&] {
    const auto status = run.prepare_replay(
        queues, {.mode = hyle::graph::ReplayMode::kPreferCapture});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  prepare_again();
  EXPECT_EQ(probe.capture_begin_count, 2u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 2u);
  // Reset revokes public bindings. Dormant commands cannot execute until a
  // complete preparation proves these same physical arguments again.
  ASSERT_TRUE(run.reset().ok());
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  prepare_again();
  EXPECT_EQ(probe.capture_destroy_count, 0u);
  EXPECT_EQ(probe.capture_begin_count, 2u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 2u);

  auto next_input = make_tensor(row(8)), next_output = make_tensor(row(8));
  ASSERT_TRUE(run.bind(logical.input, next_input.view).ok());
  prepare_again();
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  EXPECT_EQ(probe.capture_begin_count, 3u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  ASSERT_TRUE(run.bind(logical.output, next_output.view).ok());
  prepare_again();
  EXPECT_EQ(probe.capture_destroy_count, 2u);
  EXPECT_EQ(probe.capture_begin_count, 4u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kHybrid);
  ASSERT_TRUE(run.reset().ok());
}

TEST(GraphRegions, CapturesAndReplaysACommandSegmentWithoutRewalkingItsSteps) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
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
  struct BoundPorts { const OwnedTensor* input; const OwnedTensor* output; unsigned count=0; } bound{&input,&output};
  const auto visitor=+[](void* opaque,const hyle::graph::detail::RegionAccessView& view) noexcept -> Status {
    auto& bound=*static_cast<BoundPorts*>(opaque);
    if(view.inputs.size()!=1 || view.outputs.size()!=1 ||
       view.inputs[0].storage().native_handle()!=bound.input->view.storage().native_handle() ||
       view.outputs[0].storage().native_handle()!=bound.output->view.storage().native_handle())
      return Status::Internal("bound visitor lost actual region ports");
    ++bound.count;return Status::Ok();
  };
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::visit_bound(invocation,visitor,&bound).ok());
  EXPECT_EQ(bound.count,1u);EXPECT_EQ(probe.submit_count,0u);
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

  ASSERT_TRUE(hyle::graph::detail::RegionAccess::revoke_preparation(invocation).ok());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_EQ(probe.capture_destroy_count, 0u);
  const auto dormant = invocation.submit(queues);
  ASSERT_FALSE(dormant);
  EXPECT_TRUE(dormant.error().IsFailedPrecondition());
  EXPECT_EQ(probe.capture_launch_count, 2u);
  ASSERT_TRUE(invocation.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_begin_count, 1u);
  EXPECT_EQ(invocation.replay_report()->reused_capture_range_count, 1u);

  // This region has only ordinary TensorViews, with no native import array.
  // Repeated preparation must still check its current managed-slot seal.
  const auto seals = probe.binding_seal_count;
  probe.binding_seal_failure = Status::FailedPrecondition("current slot is not prepared");
  EXPECT_TRUE(invocation.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).IsFailedPrecondition());
  EXPECT_EQ(probe.binding_seal_count, seals + 1);
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_EQ(probe.capture_destroy_count, 0u);
  EXPECT_FALSE(invocation.submit(queues));
  probe.binding_seal_failure = Status::Ok();
  ASSERT_TRUE(invocation.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_begin_count, 1u);
  EXPECT_EQ(invocation.replay_report()->reused_capture_range_count, 1u);

  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  EXPECT_EQ(invocation.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_EQ(probe.capture_destroy_count, 0u);
  ASSERT_TRUE(invocation.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_begin_count, 1u);
  EXPECT_EQ(invocation.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(invocation.replay_report()->reused_capture_range_count, 1u);
}

TEST(GraphRegions, RetiredCaptureRebuildsNewBindingsWithoutPublishingFailure) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
  probe.capture_recycling_supported = true;
  probe.borrows_tensor_bindings = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto run = compiled.executable.create_invocation().value();
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  auto next_input = make_tensor(row(8)), next_output = make_tensor(row(8));
  auto third_input = make_tensor(row(8)), third_output = make_tensor(row(8));
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  const auto prepare = [&] {
    return run.prepare_replay(
        queues, {.mode = hyle::graph::ReplayMode::kRequireCapture});
  };
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  ASSERT_TRUE(prepare().ok());
  auto first = run.submit(queues);
  ASSERT_TRUE(first);
  ASSERT_TRUE(first->wait().ok());
  *first = {};

  ASSERT_TRUE(run.bind(logical.input, next_input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, next_output.view).ok());
  auto stale = run.submit(queues);
  ASSERT_FALSE(stale);
  EXPECT_TRUE(stale.error().IsFailedPrecondition());
  EXPECT_EQ(probe.capture_launch_count, 1u);
  ASSERT_TRUE(prepare().ok());
  EXPECT_EQ(probe.capture_recycle_count, 1u);
  EXPECT_EQ(probe.capture_rebuild_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 2u);
  // end returns a distinct live artifact; the consumed nonlaunchable seed dies.
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
  struct Ports { const OwnedTensor *input; const OwnedTensor *output; } ports{
      &next_input, &next_output};
  const auto visit = +[](void *raw,
      const hyle::graph::detail::RegionAccessView &view) noexcept -> Status {
    const auto &ports = *static_cast<Ports *>(raw);
    if (view.inputs.size() != 1 || view.outputs.size() != 1 ||
        view.inputs[0].storage().native_handle() !=
            ports.input->view.storage().native_handle() ||
        view.outputs[0].storage().native_handle() !=
            ports.output->view.storage().native_handle())
      return Status::Internal("rebuild retained an old tensor binding");
    return Status::Ok();
  };
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::visit_bound(
      run, visit, &ports).ok());
  auto second = run.submit(queues);
  ASSERT_TRUE(second);
  ASSERT_TRUE(second->wait().ok());
  *second = {};

  ASSERT_TRUE(run.bind(logical.input, third_input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, third_output.view).ok());
  probe.capture_end_failure = Status::ResourceExhausted("replacement rejected");
  EXPECT_TRUE(prepare().IsResourceExhausted());
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_FALSE(run.submit(queues));
  EXPECT_EQ(probe.capture_recycle_count, 2u);
  EXPECT_EQ(probe.capture_rebuild_count, 1u);
  EXPECT_EQ(probe.capture_launch_count, 2u);
  EXPECT_FALSE(probe.capture_active);
  probe.capture_end_failure = Status::Ok();
  ASSERT_TRUE(prepare().ok());
  EXPECT_EQ(probe.capture_end_count, 3u);
  EXPECT_EQ(probe.capture_rebuild_count, 1u);
  auto retry = run.submit(queues);
  ASSERT_TRUE(retry);
  EXPECT_TRUE(retry->wait().ok());
}

TEST(GraphRegions, NativeCommandInvalidationKeepsOnlyRebuildSeedAndQueueStillDiscards) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
  probe.capture_recycling_supported = true;
  probe.borrows_tensor_bindings = true;
  probe.configurable_region = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto run = compiled.executable.create_invocation().value();
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  auto next_input = make_tensor(row(8)), third_input = make_tensor(row(8));
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  auto initial = run.submit(queues);
  ASSERT_TRUE(initial);
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::invalidate_commands(run, 0).IsUnavailable());
  EXPECT_EQ(probe.capture_recycle_count, 0u);
  ASSERT_TRUE(initial->wait().ok());
  *initial = {};
  const std::uint32_t revision = 7;
  auto configured = hyle::graph::detail::RegionAccess::configure_node(
      run, logical.graph.node_id(0).value(), kTestConfiguration, &revision);
  ASSERT_TRUE(configured);
  EXPECT_TRUE(*configured);
  EXPECT_EQ(probe.capture_destroys_at_configuration, 1u);
  EXPECT_EQ(probe.capture_recycle_count, 0u);
  EXPECT_FALSE(run.submit(queues));
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());

  // A seal rejection happens after physical rebinding can recycle the old
  // artifact, but before end can consume that nonlaunchable storage.
  ASSERT_TRUE(run.bind(logical.input, next_input.view).ok());
  probe.binding_seal_failure = Status::FailedPrecondition("new slot not sealed");
  EXPECT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).IsFailedPrecondition());
  EXPECT_EQ(probe.capture_recycle_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 2u);
  EXPECT_FALSE(run.submit(queues));
  // Native argument changes revoke old launch/binding authority before the
  // provider mutates resources, but may retain its authorized nonlaunchable seed.
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::invalidate_commands(run, 0).ok());
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  EXPECT_EQ(probe.capture_recycle_count, 1u);
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::invalidate_commands(run, 0).ok());
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  EXPECT_EQ(probe.capture_recycle_count, 1u);
  const auto *native_invalidated = hyle::graph::detail::InvocationAccess::state(run);
  ASSERT_NE(native_invalidated, nullptr);
  ASSERT_EQ(native_invalidated->capture_rebuilds.size(), 1u);
  const auto &invalidated_range = native_invalidated->capture_rebuilds[0];
  EXPECT_TRUE(invalidated_range.storage.valid());
  EXPECT_FALSE(invalidated_range.qualification_valid);
  EXPECT_FALSE(invalidated_range.preserves_private_arguments);
  EXPECT_TRUE(native_invalidated->captured_ranges.empty());
  for (const auto &variant : invalidated_range.variants) {
    EXPECT_FALSE(variant.key_valid);
    EXPECT_FALSE(variant.commands.valid());
    EXPECT_TRUE(variant.bindings.empty());
  }
  EXPECT_FALSE(run.submit(queues));
  probe.binding_seal_failure = Status::Ok();
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_rebuild_count, 1u);

  ASSERT_TRUE(run.bind(logical.input, third_input.view).ok());
  probe.binding_seal_failure = Status::FailedPrecondition("new slot not sealed");
  EXPECT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).IsFailedPrecondition());
  EXPECT_EQ(probe.capture_recycle_count, 2u);
  EXPECT_EQ(probe.capture_end_count, 3u);
  EXPECT_FALSE(run.submit(queues));
  probe.binding_seal_failure = Status::Ok();
  auto next_queue = context.create_queue().value();
  const std::array next_queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &next_queue}};
  ASSERT_TRUE(run.prepare_replay(
      next_queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_destroy_count, 3u);
  EXPECT_EQ(probe.capture_end_count, 4u);
  EXPECT_EQ(probe.capture_recycle_count, 2u);
  EXPECT_EQ(probe.capture_rebuild_count, 1u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  auto done = run.submit(next_queues);
  ASSERT_TRUE(done);
  EXPECT_TRUE(done->wait().ok());
}

TEST(GraphRegions, StablePrivateCaptureVariantsReuseBothBoundPortsAfterPreparation) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
  probe.capture_private_arguments_stable = true;
  probe.borrows_tensor_bindings = true;
  probe.require_cold_bindings = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  auto run = compiled.executable.create_invocation().value();
  auto input_a = make_tensor(row(8)), output_a = make_tensor(row(8));
  auto input_b = make_tensor(row(8)), output_b = make_tensor(row(8));
  const std::array input_ports{&input_a, &input_b};
  const std::array output_ports{&output_a, &output_b};
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  const auto same = [](auto left, auto right) {
    return left.resource() == right.resource() &&
        left.native_handle() == right.native_handle() &&
        left.offset() == right.offset() && left.size() == right.size();
  };
  std::size_t episode = 0;
  for (const std::size_t index : {0u, 1u, 0u, 1u}) {
    ASSERT_TRUE(run.bind(logical.input, input_ports[index]->view).ok());
    ASSERT_TRUE(run.bind(logical.output, output_ports[index]->view).ok());
    EXPECT_FALSE(run.submit(queues));
    ASSERT_TRUE(run.prepare_replay(
        queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
    EXPECT_EQ(probe.binding_prepare_count, episode + 1);
    EXPECT_EQ(probe.binding_seal_count, episode + 1);
    EXPECT_EQ(run.replay_report()->recorded_capture_range_count, episode < 2 ? 1u : 0u);
    EXPECT_EQ(run.replay_report()->reused_capture_range_count, episode < 2 ? 0u : 1u);
    auto completed = run.submit(queues);
    ASSERT_TRUE(completed);
    ASSERT_TRUE(completed->wait().ok());
    *completed = {};
    EXPECT_TRUE(same(probe.launched_input, input_ports[index]->view.storage()));
    EXPECT_TRUE(same(probe.launched_output, output_ports[index]->view.storage()));
    ++episode;
  }
  EXPECT_EQ(probe.capture_end_count, 2u);
  EXPECT_EQ(probe.capture_launch_count, 4u);
  EXPECT_EQ(probe.capture_destroy_count, 0u);
  EXPECT_EQ(probe.create_invocation_count, 1u);
  EXPECT_EQ(probe.capture_recycle_count, 0u);
}

TEST(GraphRegions, CaptureVariantsUseBoundedLruAndClearBeforeForcedChanges) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
  probe.capture_private_arguments_stable = true;
  probe.capture_recycling_supported = true;
  probe.borrows_tensor_bindings = true;
  probe.require_cold_bindings = true;
  probe.configurable_region = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto run = compiled.executable.create_invocation().value();
  auto input_a = make_tensor(row(8)), output_a = make_tensor(row(8));
  auto input_b = make_tensor(row(8)), output_b = make_tensor(row(8));
  auto output_c = make_tensor(row(8));
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  const auto prepare = [&](const OwnedTensor &input, const OwnedTensor &output) {
    auto status = run.bind(logical.input, input.view);
    if (!status.ok()) return status;
    status = run.bind(logical.output, output.view);
    if (!status.ok()) return status;
    return run.prepare_replay(queues, {.mode = hyle::graph::ReplayMode::kRequireCapture});
  };
  ASSERT_TRUE(prepare(input_a, output_a).ok());
  ASSERT_TRUE(prepare(input_b, output_b).ok());
  ASSERT_TRUE(prepare(input_a, output_a).ok());
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  // C shares A's input: an output-only change must still miss the complete key.
  ASSERT_TRUE(prepare(input_a, output_c).ok());
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(probe.capture_recycle_count, 1u);
  EXPECT_EQ(probe.capture_rebuild_count, 1u);
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  ASSERT_TRUE(prepare(input_a, output_a).ok());
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 3u); // A survived; least-recent B did not.

  auto destroyed = probe.capture_destroy_count;
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::invalidate_commands(run, 0).ok());
  EXPECT_EQ(probe.capture_destroy_count, destroyed + 2);
  EXPECT_FALSE(run.submit(queues));
  ASSERT_TRUE(prepare(input_a, output_a).ok());
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  ASSERT_TRUE(prepare(input_b, output_b).ok());

  destroyed = probe.capture_destroy_count;
  const std::uint32_t revision = 9;
  probe.configuration_failure = Status::ResourceExhausted("configuration changed before rejection");
  auto configured = hyle::graph::detail::RegionAccess::configure_node(
      run, logical.graph.node_id(0).value(), kTestConfiguration, &revision);
  ASSERT_FALSE(configured);
  EXPECT_TRUE(configured.error().IsResourceExhausted());
  EXPECT_EQ(probe.capture_destroys_at_configuration, destroyed + 2);
  EXPECT_FALSE(run.submit(queues));
  probe.configuration_failure = Status::Ok();
  ASSERT_TRUE(prepare(input_a, output_a).ok());
  ASSERT_TRUE(prepare(input_b, output_b).ok());

  destroyed = probe.capture_destroy_count;
  auto replacement_queue = context.create_queue().value();
  const std::array replacement_queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &replacement_queue}};
  ASSERT_TRUE(run.prepare_replay(replacement_queues,
      {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_destroy_count, destroyed + 2);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  ASSERT_TRUE(run.bind(logical.input, input_a.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output_a.view).ok());
  ASSERT_TRUE(run.prepare_replay(replacement_queues,
      {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  destroyed = probe.capture_destroy_count;
  ASSERT_TRUE(run.prepare_replay(replacement_queues,
      {.mode = hyle::graph::ReplayMode::kCommands}).ok());
  EXPECT_EQ(probe.capture_destroy_count, destroyed + 2);
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input_a, values);
  probe.required_capture_destroys_before_submit = destroyed + 2;
  auto completed = run.submit(replacement_queues);
  ASSERT_TRUE(completed);
  ASSERT_TRUE(completed->wait().ok());
  EXPECT_EQ(read(output_a, 8), std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, CaptureVariantEndAndSealFailuresCannotPublishAnOldRoute) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.capture_preserves_binding_identity = true;
  probe.capture_private_arguments_stable = true;
  probe.capture_recycling_supported = true;
  probe.borrows_tensor_bindings = true;
  probe.require_cold_bindings = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto run = compiled.executable.create_invocation().value();
  auto input_a = make_tensor(row(8)), output_a = make_tensor(row(8));
  auto input_b = make_tensor(row(8)), output_b = make_tensor(row(8));
  auto input_c = make_tensor(row(8)), output_c = make_tensor(row(8));
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  const auto bind = [&](const OwnedTensor &input, const OwnedTensor &output) {
    auto status = run.bind(logical.input, input.view);
    return status.ok() ? run.bind(logical.output, output.view) : status;
  };
  const auto prepare = [&] {
    return run.prepare_replay(queues, {.mode = hyle::graph::ReplayMode::kRequireCapture});
  };
  ASSERT_TRUE(bind(input_a, output_a).ok()); ASSERT_TRUE(prepare().ok());
  ASSERT_TRUE(bind(input_b, output_b).ok()); ASSERT_TRUE(prepare().ok());
  ASSERT_TRUE(bind(input_c, output_c).ok());
  probe.capture_end_failure = Status::ResourceExhausted("new C capture rejected");
  EXPECT_TRUE(prepare().IsResourceExhausted());
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_FALSE(run.submit(queues));
  EXPECT_EQ(probe.capture_launch_count, 0u);
  EXPECT_EQ(probe.capture_end_count, 2u);
  EXPECT_EQ(probe.capture_recycle_count, 1u);
  EXPECT_EQ(probe.capture_rebuild_count, 0u);
  EXPECT_FALSE(probe.capture_active);
  probe.capture_end_failure = Status::Ok();
  ASSERT_TRUE(prepare().ok());
  EXPECT_EQ(probe.capture_end_count, 3u);
  EXPECT_EQ(probe.capture_rebuild_count, 0u); // Failed end consumed its idle seed.

  // A failed capture discarded every old variant. Explicitly rebuild B, then
  // return to C so B is demonstrably dormant before testing its fresh seal.
  ASSERT_TRUE(bind(input_b, output_b).ok());
  ASSERT_TRUE(prepare().ok());
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 4u);
  ASSERT_TRUE(bind(input_c, output_c).ok());
  ASSERT_TRUE(prepare().ok());
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 4u);
  ASSERT_TRUE(bind(input_b, output_b).ok());
  const auto preparations = probe.binding_prepare_count;
  const auto seals = probe.binding_seal_count;
  probe.binding_seal_failure = Status::FailedPrecondition("current episode is unsealed");
  EXPECT_TRUE(prepare().IsFailedPrecondition());
  EXPECT_EQ(probe.binding_prepare_count, preparations + 1);
  EXPECT_EQ(probe.binding_seal_count, seals + 1);
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_FALSE(run.submit(queues));
  EXPECT_EQ(probe.capture_end_count, 4u);
  probe.binding_seal_failure = Status::Ok();
  ASSERT_TRUE(prepare().ok());
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  EXPECT_EQ(probe.capture_end_count, 4u);
  auto completed = run.submit(queues);
  ASSERT_TRUE(completed);
  ASSERT_TRUE(completed->wait().ok());
  EXPECT_EQ(probe.launched_input.native_handle(), input_b.view.storage().native_handle());
  EXPECT_EQ(probe.launched_output.native_handle(), output_b.view.storage().native_handle());
}

TEST(GraphRegions, UnknownCaptureIdentityRepreparesAfterSameTupleBindingEpisode) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.required_resource = hyle::mem::system_resource();
  probe.capture_supported = true;
  probe.decline_regions = true;
  probe.require_cold_bindings = true;
  probe.cold_bindings_capture_compatible = true;
  // The default mock contract does not prove a canonical physical address
  // across binding episodes. Equal opaque handles must not infer that proof.
  ASSERT_FALSE(probe.capture_preserves_binding_identity);
  auto context = make_context(probe);
  const auto logical = make_negations(1);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array sentinel{19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F};
  write(output, sentinel);
  auto run = compiled.executable.create_invocation().value();
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  const auto bind = [&] {
    ASSERT_TRUE(run.bind(logical.input, input.view).ok());
    ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  };
  const auto prepare = [&] {
    const auto status = run.prepare_replay(
        queues, {.mode = hyle::graph::ReplayMode::kRequireCapture});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  bind();
  prepare();
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_EQ(probe.binding_prepare_count, 1u);
  EXPECT_EQ(probe.capture_begin_count, 1u);
  prepare();
  EXPECT_EQ(probe.binding_prepare_count, 1u);
  EXPECT_EQ(probe.capture_begin_count, 1u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 1u);
  ASSERT_TRUE(run.reset().ok());
  EXPECT_EQ(probe.capture_destroy_count, 1u);
  bind();
  prepare();
  EXPECT_EQ(probe.binding_prepare_count, 2u);
  EXPECT_EQ(probe.capture_begin_count, 2u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
  // Successful ordinary rebinding also starts a new binding episode, even
  // without reset and even though every TensorDesc/range bit is unchanged.
  bind();
  EXPECT_EQ(probe.capture_destroy_count, 2u);
  prepare();
  EXPECT_EQ(probe.binding_prepare_count, 3u);
  EXPECT_EQ(probe.capture_begin_count, 3u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
  EXPECT_EQ(probe.submit_count, 0u);
  EXPECT_EQ(read(output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kCommands}).ok());
  EXPECT_EQ(probe.capture_destroy_count, 3u);
  EXPECT_EQ(probe.binding_prepare_count, 3u);
  // The same safety rule also applies when no graph remains: an unknown
  // backend's cached ordinary bindings cannot cross a successful new bind.
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  write(input, values);
  bind();
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kCommands}).ok());
  EXPECT_EQ(probe.binding_prepare_count, 4u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
  EXPECT_EQ(read(output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));
  ASSERT_TRUE(run.submit(queues)->wait().ok());
  EXPECT_EQ(read(output, 8), (std::vector<float>{-1, 2, -3, 4, -5, 6, -7, 8}));
}

TEST(GraphRegions, FailedSelectedConfigurationInvalidatesBeforeItsCallback) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.configurable_region = true;
  probe.artifact_properties.runtime.capture_compatible = true;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.region_count(), 1u);
  auto input = make_tensor(row(8)), output = make_tensor(row(8));
  const std::array sentinel{19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F, 19.F};
  write(output, sentinel);
  auto run = compiled.executable.create_invocation().value();
  ASSERT_TRUE(run.bind(logical.input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  ASSERT_EQ(probe.capture_end_count, 1u);
  probe.configuration_failure = Status::ResourceExhausted("configured revision OOM");
  const std::uint32_t revision = 7;
  const auto configured = hyle::graph::detail::RegionAccess::configure_node(
      run, logical.graph.node_id(0).value(), kTestConfiguration, &revision);
  ASSERT_FALSE(configured);
  EXPECT_TRUE(configured.error().IsResourceExhausted());
  EXPECT_EQ(configured.error().message(), "configured revision OOM");
  EXPECT_EQ(probe.configuration_count, 1u);
  EXPECT_EQ(probe.capture_destroys_at_configuration, 1u);
  EXPECT_EQ(run.replay_route(), hyle::graph::ReplayRoute::kUnprepared);
  EXPECT_EQ(probe.submit_count, 0u);
  EXPECT_EQ(read(output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));
  probe.configuration_failure = Status::Ok();
  ASSERT_TRUE(hyle::graph::detail::RegionAccess::configure_node(
      run, logical.graph.node_id(0).value(), kTestConfiguration, &revision).value());
  ASSERT_TRUE(run.prepare_replay(
      queues, {.mode = hyle::graph::ReplayMode::kRequireCapture}).ok());
  EXPECT_EQ(probe.capture_begin_count, 2u);
  EXPECT_EQ(run.replay_report()->recorded_capture_range_count, 1u);
  EXPECT_EQ(run.replay_report()->reused_capture_range_count, 0u);
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
  // Nodes 2/3 have neither a starting region nor a leaf alternative. They
  // are covered by earlier-starting {0,2}/{1,3}, not by begins_[node] alone.
  auto result = hyle::graph::compile(
      graph, context, {.max_route_search_states = 2,
                       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(result) << result.error().message();
  auto compiled = std::move(*result);
  EXPECT_EQ(compiled.report.region_count(), 2u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 0u);
  EXPECT_EQ(compiled.report.search().route_search_states, 2u);
  EXPECT_EQ(probe.single_proposal_count, 0u);
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

TEST(GraphRegions, ProviderContractibilityUsesOnlyCandidateSpan) {
  auto builder = GraphBuilder::create().value();
  const auto input = builder.input(row(8)).value();
  auto prefix = input;
  for (std::size_t i = 0; i < 64; ++i) {
    const auto node =
        builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{prefix})
            .value();
    prefix = builder.result(node).value();
  }
  const auto first =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{prefix})
          .value();
  const auto outside =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{prefix})
          .value();
  const auto last =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}),
                       std::array{builder.result(first).value()})
          .value();
  ASSERT_TRUE(builder.depends_on(last, outside).ok());
  const auto control_outside =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{prefix})
          .value();
  const auto control_last =
      builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{prefix})
          .value();
  ASSERT_TRUE(builder.depends_on(control_outside, last).ok());
  ASSERT_TRUE(builder.depends_on(control_last, control_outside).ok());
  ASSERT_TRUE(builder.publish(builder.result(control_last).value()).ok());
  const auto graph = std::move(builder).finish().value();
  ASSERT_GT(first.ordinal(), 16U);

  auto metadata =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  const auto original_allocations = metadata.snapshot().total_allocs;
  auto shared_predecessor = hyle::graph::is_region_contractible(
      graph, std::array{last, first}, metadata.get());
  ASSERT_TRUE(shared_predecessor) << shared_predecessor.error().message();
  EXPECT_TRUE(*shared_predecessor);
  auto control_reentry = hyle::graph::is_region_contractible(
      graph, std::array{last, control_last}, metadata.get());
  ASSERT_TRUE(control_reentry) << control_reentry.error().message();
  EXPECT_FALSE(*control_reentry);
  auto data_and_control_reentry = hyle::graph::is_region_contractible(
      graph, std::array{first, control_last}, metadata.get());
  ASSERT_TRUE(data_and_control_reentry)
      << data_and_control_reentry.error().message();
  EXPECT_FALSE(*data_and_control_reentry);
  EXPECT_EQ(metadata.snapshot().total_allocs, original_allocations);
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
  // Discovery stopped at its node budget, so absent coverage is not proof
  // that the graph is unsupported. The refused region remains unadmitted.
  EXPECT_TRUE(no_coverage.error().IsResourceExhausted());
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

struct ReadyConstantGraph final {
  Graph graph;
  ValueId constant;
  std::array<ValueId, 2> outputs;
  ValueId alias;
};

ReadyConstantGraph make_ready_constant_graph(ReadyConstantProbe &ready) {
  auto builder = GraphBuilder::create().value();
  ReadyConstantGraph result;
  result.constant = builder.constant(row(4)).value();
  std::array<ValueId, 2> prepared;
  for (std::size_t port = 0; port < prepared.size(); ++port) {
    const auto node = builder.add_node(Operation(Unary{UnaryKind::kNeg}),
                                       std::array{result.constant}).value();
    ready.nodes.push_back(node);
    prepared[port] = builder.result(node).value();
  }
  for (std::size_t port = 0; port < prepared.size(); ++port) {
    result.outputs[port] = builder.add(Unary{UnaryKind::kNeg}, prepared[port]).value();
    EXPECT_TRUE(builder.publish(result.outputs[port]).ok());
  }
  const auto slice = hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(0),
      hyle::graph::Dimension::fixed(4)).value();
  result.alias = builder.add(slice, prepared[1]).value();
  EXPECT_TRUE(builder.publish(result.alias).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

void configure_ready_constants(RegionProbe &probe, ReadyConstantProbe &ready) {
  probe.ready_constants = &ready;
  probe.required_resource = hyle::mem::system_resource();
  probe.artifact_properties.dispatch_count = 0;
  probe.artifact_properties.invocation_private_bytes = 0;
  probe.artifact_properties.runtime.replay_class = ProviderReplayClass::kPrepared;
}

ValueId add_ready_constant_node(GraphBuilder &builder, ReadyConstantProbe &ready,
                                ValueId input) {
  const auto node = builder.add_node(Operation(Unary{UnaryKind::kNeg}),
                                     std::array{input}).value();
  ready.nodes.push_back(node);
  return builder.result(node).value();
}

TEST(GraphRegions, ImmutableConstantViewsPreserveDenseReshapeAndSliceBindings) {
  ReadyConstantProbe ready;
  RegionProbe probe;
  configure_ready_constants(probe, ready);
  auto context = make_context(probe);
  auto builder = GraphBuilder::create().value();
  const auto before = builder.add_node(Operation::custom(ExternalEffect{}),
                                       std::array<ValueId, 0>{}).value();
  const auto root = builder.constant(row(12)).value();
  const auto matrix = builder.add(hyle::graph::ViewSpec::reshape(std::array{
      hyle::graph::Dimension::fixed(3), hyle::graph::Dimension::fixed(4)}).value(), root).value();
  const auto slice = builder.add(hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(1), hyle::graph::Dimension::fixed(1)).value(), matrix).value();
  const auto selected = builder.add(hyle::graph::ViewSpec::reshape(std::array{
      hyle::graph::Dimension::fixed(4)}).value(), slice).value();
  const auto prepared = add_ready_constant_node(builder, ready, selected);
  ASSERT_TRUE(builder.depends_on(ready.nodes[0], before).ok());
  const auto output = builder.add(Unary{UnaryKind::kNeg}, prepared).value();
  ASSERT_TRUE(builder.publish(selected).ok());
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(12));
  write(source, std::array{1.F, 2.F, 3.F, 4.F, -5.F, 6.F,
                          -7.F, 8.F, 9.F, 10.F, 11.F, 12.F});
  const std::array constants{hyle::graph::ConstantBinding{root, source.view.as_const()}};
  auto compiled = hyle::graph::compile(graph, context, {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(ready.inputs.size(), 1U);
  EXPECT_EQ(ready.inputs[0].origin.value, selected);
  EXPECT_EQ(ready.inputs[0].origin.root, root);
  EXPECT_EQ(ready.inputs[0].origin.byte_offset, 4 * sizeof(float));
  EXPECT_EQ(ready.inputs[0].tensor.description(), row(4));
  EXPECT_EQ(ready.inputs[0].tensor.storage().native_handle(), source.view.storage().native_handle());
  EXPECT_EQ(ready.inputs[0].tensor.storage().offset(), 4 * sizeof(float));
  ASSERT_EQ(ready.source_values.size(), 1U);
  EXPECT_EQ(ready.source_values[0], (std::array{-5.F, 6.F, -7.F, 8.F}));
  auto run = compiled->executable.create_invocation().value();
  auto destination = make_tensor(row(4));
  ASSERT_TRUE(run.bind(output, destination.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(run.submit(queue)->wait().ok());
  EXPECT_EQ(probe.effect_submit_count, 1U);
  EXPECT_EQ(read(destination, 4), (std::vector<float>{-5.F, 6.F, -7.F, 8.F}));
  const auto published = run.output(selected);
  ASSERT_TRUE(published) << published.error().message();
  EXPECT_EQ(published->storage().native_handle(), source.view.storage().native_handle());
  EXPECT_EQ(published->storage().offset(), 4 * sizeof(float));
}

TEST(GraphRegions, ConstantViewOffsetsSeparateFiniteVariantArtifacts) {
  ReadyConstantProbe ready;
  RegionProbe probe;
  configure_ready_constants(probe, ready);
  auto context = make_context(probe);
  auto builder = GraphBuilder::create().value();
  const auto start = builder.dimension({0, 4, 1}).value();
  const auto root = builder.constant(row(8)).value();
  const auto selected = builder.add(hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension(start), hyle::graph::Dimension::fixed(4)).value(), root).value();
  const auto spec = hyle::graph::TensorSpec::concrete(row(4)).value();
  const auto node = builder.add_node(Operation(Unary{UnaryKind::kNeg}),
      std::array{selected}, std::array{spec}).value();
  ready.nodes.push_back(node);
  const auto prepared = builder.result(node).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, spec, prepared).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(8));
  write(source, std::array{1.F, 2.F, 3.F, 4.F, -5.F, -6.F, -7.F, -8.F});
  const std::array constants{hyle::graph::ConstantBinding{
      root, source.view.as_const(), hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array first{hyle::graph::ShapeBinding{start, 0}};
  const std::array second{hyle::graph::ShapeBinding{start, 4}};
  const std::array variants{hyle::graph::ExactShapeVariant{first},
                            hyle::graph::ExactShapeVariant{second}};
  auto compiled = hyle::graph::compile(graph, context,
      {.shape_policy = hyle::graph::CompileShapePolicy::kFiniteExact,
       .exact_variants = variants, .constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(ready.inputs.size(), 2U);
  EXPECT_EQ(ready.inputs[0].origin.value, ready.inputs[1].origin.value);
  EXPECT_EQ(ready.inputs[0].origin.root, root);
  EXPECT_EQ(ready.inputs[1].origin.root, root);
  EXPECT_EQ(ready.inputs[0].tensor.description(), ready.inputs[1].tensor.description());
  EXPECT_EQ(ready.inputs[0].origin.byte_offset, 0U);
  EXPECT_EQ(ready.inputs[1].origin.byte_offset, 4 * sizeof(float));
  EXPECT_EQ(ready.factories, 2U);
  ASSERT_EQ(ready.source_values.size(), 2U);
  EXPECT_EQ(ready.source_values[0], (std::array{1.F, 2.F, 3.F, 4.F}));
  EXPECT_EQ(ready.source_values[1], (std::array{-5.F, -6.F, -7.F, -8.F}));
  ASSERT_TRUE(source.buffer.reset().ok());
  auto queue = context.create_queue().value();
  for (std::size_t variant = 0; variant < variants.size(); ++variant) {
    auto child = compiled->executable.exact_variant(variant).value();
    auto run = child.create_invocation().value();
    auto destination = make_tensor(row(4));
    ASSERT_TRUE(run.bind(output, destination.view).ok());
    ASSERT_TRUE(run.submit(queue)->wait().ok());
    const std::vector<float> expected = variant == 0
        ? std::vector<float>{1.F, 2.F, 3.F, 4.F}
        : std::vector<float>{-5.F, -6.F, -7.F, -8.F};
    EXPECT_EQ(read(destination, 4), expected);
  }
}

TEST(GraphRegions, ConstantViewPreparationKeepsEachPhysicalRootOwner) {
  struct Resource final {
    std::shared_ptr<HostResource> storage;
    hyle::mem::MemoryResourceRef ref() const noexcept {
      return hyle::mem::MemoryResourceRef(*storage);
    }
  };
  ReadyConstantProbe first_ready, second_ready;
  RegionProbe first_probe, second_probe;
  configure_ready_constants(first_probe, first_ready);
  configure_ready_constants(second_probe, second_ready);
  first_ready.require_queued_eligibility = second_ready.require_queued_eligibility = true;
  auto first_context = make_context(first_probe);
  auto second_context = make_context(second_probe);
  auto first_owner = hyle::compute::StorageOwner::adopt(first_context,
      Resource{std::make_shared<HostResource>()}).value();
  auto second_owner = hyle::compute::StorageOwner::adopt(second_context,
      Resource{std::make_shared<HostResource>()}).value();
  first_probe.required_resource = first_probe.region_input_resource =
      first_probe.region_output_resource = first_owner.ref();
  second_probe.required_resource = second_probe.region_input_resource =
      second_probe.region_output_resource = second_owner.ref();
  const Environment environment{{&first_context, first_owner.ref()},
                                {&second_context, second_owner.ref()}};
  const std::array domains{
      hyle::graph::StorageDomain{.resource = first_owner.ref(), .owner = first_owner},
      hyle::graph::StorageDomain{.resource = second_owner.ref(), .owner = second_owner}};
  auto actual_environment = environment.environment();
  actual_environment.storage_domains = domains;
  auto builder = GraphBuilder::create().value();
  const auto root = builder.constant(row(12)).value();
  const auto selected = builder.add(hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(4), hyle::graph::Dimension::fixed(4)).value(), root).value();
  const std::array prepared{
      add_ready_constant_node(builder, first_ready, selected),
      add_ready_constant_node(builder, second_ready, selected)};
  const std::array outputs{
      builder.add(Unary{UnaryKind::kNeg}, prepared[0]).value(),
      builder.add(Unary{UnaryKind::kNeg}, prepared[1]).value()};
  for (const auto output : outputs)
    ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(selected).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array placements{
      hyle::graph::NodePlacement{first_ready.nodes[0], environment.target(0), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{second_ready.nodes[0], environment.target(1), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{graph.producer(outputs[0]).value(), environment.target(0), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{graph.producer(outputs[1]).value(), environment.target(1), hyle::graph::PlacementStrength::kRequired}};
  const std::array external{
      hyle::graph::ExternalResourceContract{root, environment.storage(0)},
      hyle::graph::ExternalResourceContract{selected, environment.storage(0)},
      hyle::graph::ExternalResourceContract{outputs[0], environment.storage(0)},
      hyle::graph::ExternalResourceContract{outputs[1], environment.storage(1)}};
  auto buffer = Buffer::create(first_owner.ref(), 12 * sizeof(float), 64).value();
  const std::array values{1.F, 2.F, 3.F, 4.F, -5.F, 6.F,
                          -7.F, 8.F, 9.F, 10.F, 11.F, 12.F};
  ASSERT_TRUE(buffer.write(0, std::as_bytes(std::span(values))).ok());
  auto source = first_owner.adopt_buffer(std::move(buffer)).value();
  const std::array constants{hyle::graph::ConstantBinding{
      .value = root, .tensor = source.tensor(row(12)).value(),
      .owner = first_owner, .lease = source}};
  auto compiled = hyle::graph::compile(graph, actual_environment,
      {.constants = constants, .node_placements = placements, .external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(first_ready.inputs.size(), 1U);
  ASSERT_EQ(second_ready.inputs.size(), 1U);
  const auto &first = first_ready.inputs[0];
  const auto &second = second_ready.inputs[0];
  for (const auto *input : {&first, &second}) {
    EXPECT_EQ(input->origin.value, selected);
    EXPECT_EQ(input->origin.root, root);
    EXPECT_EQ(input->origin.byte_offset, 4 * sizeof(float));
    EXPECT_TRUE(input->can_prepare_queued);
  }
  EXPECT_EQ(first.tensor.storage().resource(), first_owner.ref());
  EXPECT_EQ(first.owner_resource, first_owner.ref());
  EXPECT_EQ(first.policy, hyle::graph::ConstantPolicy::kBorrow);
  EXPECT_TRUE(first.has_lease);
  EXPECT_EQ(second.tensor.storage().resource(), second_owner.ref());
  EXPECT_EQ(second.owner_resource, second_owner.ref());
  EXPECT_EQ(second.policy, hyle::graph::ConstantPolicy::kMaterialize);
  EXPECT_FALSE(second.has_lease);
  EXPECT_NE(first.tensor.storage().native_handle(), second.tensor.storage().native_handle());
  EXPECT_EQ(first_ready.factories, 1U);
  EXPECT_EQ(second_ready.factories, 1U);
  auto run = compiled->executable.create_invocation().value();
  auto first_output = make_tensor(row(4), first_owner.ref());
  auto second_output = make_tensor(row(4), second_owner.ref());
  ASSERT_TRUE(run.bind(outputs[0], first_output.view).ok());
  ASSERT_TRUE(run.bind(outputs[1], second_output.view).ok());
  auto first_queue = first_context.create_queue().value();
  auto second_queue = second_context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{environment.lane(0), &first_queue},
      hyle::graph::QueueBinding{environment.lane(1), &second_queue}};
  const auto replay = run.prepare_replay(queues);
  ASSERT_TRUE(replay.ok()) << replay.message();
  auto completion = run.submit(queues);
  ASSERT_TRUE(completion) << completion.error().message();
  const auto terminal = completion->wait();
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  const std::vector<float> expected{-5.F, 6.F, -7.F, 8.F};
  EXPECT_EQ(read(first_output, 4), expected);
  EXPECT_EQ(read(second_output, 4), expected);
}

TEST(GraphRegions, DynamicAndUnownedViewsDoNotAcquireQueuedArtifacts) {
  for (const bool dynamic : {false, true}) {
    SCOPED_TRACE(dynamic);
    ReadyConstantProbe ready;
    ready.require_queued_eligibility = true;
    RegionProbe probe;
    configure_ready_constants(probe, ready);
    auto context = make_context(probe);
    auto builder = GraphBuilder::create().value();
    const auto root = dynamic ? builder.input(row(8)).value()
                              : builder.constant(row(8)).value();
    const auto selected = builder.add(hyle::graph::ViewSpec::slice(0,
        hyle::graph::Dimension::fixed(2), hyle::graph::Dimension::fixed(4)).value(), root).value();
    const auto prepared = add_ready_constant_node(builder, ready, selected);
    const auto output = builder.add(Unary{UnaryKind::kNeg}, prepared).value();
    ASSERT_TRUE(builder.publish(output).ok());
    const auto graph = std::move(builder).finish().value();
    auto source = make_tensor(row(8));
    write(source, std::array{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F});
    const std::array constants{hyle::graph::ConstantBinding{root, source.view.as_const()}};
    auto compiled = hyle::graph::compile(graph, context,
        {.constants = dynamic ? std::span<const hyle::graph::ConstantBinding>{}
                               : std::span<const hyle::graph::ConstantBinding>(constants)});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_GT(ready.compile_attempts, 0U);
    EXPECT_EQ(ready.inputs.size(), dynamic ? 0U : 1U);
    if (!dynamic) {
      EXPECT_EQ(ready.inputs[0].origin.root, root);
      EXPECT_EQ(ready.inputs[0].origin.byte_offset, 2 * sizeof(float));
      EXPECT_FALSE(ready.inputs[0].owner_resource.valid());
      EXPECT_FALSE(ready.inputs[0].has_lease);
      EXPECT_FALSE(ready.inputs[0].can_prepare_queued);
    }
    EXPECT_EQ(ready.factories, 0U);
    EXPECT_EQ(compiled->report.region_count(), 0U);
    auto queue = context.create_queue().value();
    auto run = compiled->executable.create_invocation().value();
    auto destination = make_tensor(row(4));
    for (int replay = 0; replay < 2; ++replay) {
      if (replay != 0) {
        ASSERT_TRUE(run.reset().ok());
        if (dynamic)
          write(source, std::array{-1.F, -2.F, -3.F, -4.F, -5.F, -6.F, -7.F, -8.F});
      }
      if (dynamic)
        ASSERT_TRUE(run.bind(root, source.view).ok());
      ASSERT_TRUE(run.bind(output, destination.view).ok());
      ASSERT_TRUE(run.submit(queue)->wait().ok());
      const float sign = dynamic && replay != 0 ? -1.F : 1.F;
      EXPECT_EQ(read(destination, 4), (std::vector<float>{3 * sign, 4 * sign, 5 * sign, 6 * sign}));
      EXPECT_EQ(ready.factories, 0U);
    }
  }
}

TEST(GraphRegions, ReadyOnlyConstantsShareOwnedOutputsAcrossSlotsAndOldResults) {
  ReadyConstantProbe ready;
  RegionProbe probe;
  configure_ready_constants(probe, ready);
  auto context = make_context(probe);
  const auto logical = make_ready_constant_graph(ready);
  auto source = make_tensor(row(4));
  const std::array values{1.F, -2.F, 3.F, -4.F};
  write(source, values);
  const std::array constants{hyle::graph::ConstantBinding{
      logical.constant, source.view.as_const(),
      hyle::graph::ConstantPolicy::kMaterialize}};
  auto compiled = hyle::graph::compile(logical.graph, context,
                                      {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1U);
  ASSERT_EQ(compiled->report.step_count(), 4U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 1U);
  const auto *plan = hyle::graph::detail::ExecutableAccess::state(compiled->executable);
  ASSERT_NE(plan, nullptr);
  std::size_t ready_regions = 0;
  for (const auto &step : plan->steps) {
    if (!step.region.valid())
      continue;
    ++ready_regions;
    EXPECT_EQ(step.kind, hyle::graph::detail::PlanStepKind::kRegion);
    EXPECT_EQ(step.node_count, 2U);
    EXPECT_EQ(step.output_count, 0U);
    EXPECT_FALSE(step.emits_command());
  }
  EXPECT_EQ(ready_regions, 1U);
  EXPECT_EQ(ready.factories, 1U);
  ASSERT_TRUE(ready.prepared_backing.valid());
  ASSERT_EQ(ready.prepared_backing.size(), 8 * sizeof(float));
  const auto reject_prepared_alias = [&](hyle::graph::Invocation &slot) {
    for (const auto offset : {std::size_t{0}, sizeof(float), 4 * sizeof(float)}) {
      const auto overlap = TensorView::create(row(4),
          ready.prepared_backing.subview(offset, 4 * sizeof(float)).value()).value();
      EXPECT_TRUE(slot.bind(logical.outputs[0], overlap).IsInvalidArgument());
    }
    std::array<float, 8> retained{};
    EXPECT_TRUE(ready.prepared_backing.read(
        std::as_writable_bytes(std::span(retained))).ok());
    EXPECT_EQ(retained, (std::array{-1.F, 2.F, -3.F, 4.F, -1.F, 2.F, -3.F, 4.F}));
  };
  ASSERT_TRUE(source.buffer.reset().ok());
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  std::array<std::array<OwnedTensor, 2>, 2> outputs;
  auto queue = context.create_queue().value();
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    reject_prepared_alias(slots[slot]);
    for (std::size_t port = 0; port < logical.outputs.size(); ++port) {
      outputs[slot][port] = make_tensor(row(4));
      ASSERT_TRUE(slots[slot].bind(logical.outputs[port], outputs[slot][port].view).ok());
    }
  }
  const std::array queues{hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(slots[0].prepare_replay(queues).ok());
  ASSERT_TRUE(slots[0].submit(queue)->wait().ok());
  auto old = slots[0].result(logical.alias).value();
  auto old_output = slots[0].result(logical.outputs[0]).value();
  EXPECT_TRUE(slots[0].reset().IsUnavailable());
  for (int replay = 0; replay < 2; ++replay) {
    if (replay != 0) {
      ASSERT_TRUE(slots[1].reset().ok());
      reject_prepared_alias(slots[1]);
      for (std::size_t port = 0; port < logical.outputs.size(); ++port)
        ASSERT_TRUE(slots[1].bind(logical.outputs[port], outputs[1][port].view).ok());
    }
    ASSERT_TRUE(slots[1].prepare_replay(queues).ok());
    ASSERT_TRUE(slots[1].submit(queue)->wait().ok());
    for (const auto &output : outputs[1])
      EXPECT_EQ(read(output, 4), std::vector<float>(values.begin(), values.end()));
    std::array<float, 4> retained{};
    const auto old_tensor = old.tensor();
    ASSERT_TRUE(old_tensor) << old_tensor.error().message();
    ASSERT_TRUE(old_tensor->storage().read(
        std::as_writable_bytes(std::span(retained))).ok());
    EXPECT_EQ(retained, (std::array{-1.F, 2.F, -3.F, 4.F}));
    EXPECT_EQ(read(outputs[0][0], 4), std::vector<float>(values.begin(), values.end()));
  }
  EXPECT_EQ(probe.create_invocation_count, 4U); // Only the two consumers per slot.
  EXPECT_EQ(probe.submit_count, 6U);
  compiled->executable = {};
  for (auto &slot : slots)
    slot = {};
  EXPECT_EQ(ready.destroys, 0U);
  ASSERT_TRUE(old.tensor());
  ASSERT_TRUE(old_output.tensor());
  old.reset();
  EXPECT_EQ(ready.destroys, 0U);
  old_output.reset();
  EXPECT_EQ(ready.destroys, 1U);
}

TEST(GraphRegions, ReadyOnlyRegionCarriesControlDependenciesWithoutItsOwnQueue) {
  ReadyConstantProbe ready;
  RegionProbe before_probe, ready_probe, after_probe;
  configure_ready_constants(ready_probe, ready);
  // Preserve the exact proposed runtime contract even though this selected
  // artifact has no invocation, host blocking or private-thread episode.
  ready_probe.artifact_properties.runtime.blocks_host = true;
  ready_probe.artifact_properties.runtime.uses_private_threads = true;
  after_probe.required_resource = hyle::mem::system_resource();
  auto before_context = make_context(before_probe);
  auto ready_context = make_context(ready_probe);
  auto after_context = make_context(after_probe);
  const auto resource = hyle::mem::system_resource();
  const Environment environment{{&before_context, resource},
                                {&ready_context, resource},
                                {&after_context, resource}};
  ResourceStorage state;
  auto builder = GraphBuilder::create().value();
  const auto effect = builder.effect_resource().value();
  const auto before = add_resource_touch(builder, effect, state,
                                          ResourceAccess::kWrite, 43);
  const auto constant = builder.constant(row(4)).value();
  const auto prepared = builder.add_node(Operation(Unary{UnaryKind::kNeg}),
                                         std::array{constant}).value();
  ready.nodes.push_back(prepared);
  ASSERT_TRUE(builder.depends_on(prepared, before).ok());
  // The consumer has no direct resource/control edge to before. Its required
  // ordering must travel through the ready value's logical producer.
  const auto output = builder.add(Unary{UnaryKind::kNeg},
                                  builder.result(prepared).value()).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array placements{
      hyle::graph::NodePlacement{before, environment.target(0), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{prepared, environment.target(1), hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{graph.producer(output).value(), environment.target(2), hyle::graph::PlacementStrength::kRequired}};
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, 2.F, 3.F, 4.F});
  const std::array constants{hyle::graph::ConstantBinding{constant, source.view.as_const()}};
  const std::array effects{EffectResourceBinding{effect, state.footprint()}};
  auto compiled = hyle::graph::compile(graph, environment.environment(),
      {.constants = constants, .effect_bindings = effects, .node_placements = placements});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1U);
  ASSERT_EQ(compiled->report.segment_count(), 2U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 2U);
  ASSERT_EQ(compiled->report.segment(1)->dependency_count, 1U);
  EXPECT_EQ(compiled->report.segment_dependency(1, 0)->source_segment, 0U);
  EXPECT_EQ(compiled->report.segment(0)->lane, environment.lane(0));
  EXPECT_EQ(compiled->report.segment(1)->lane, environment.lane(2));
  auto run = compiled->executable.create_invocation().value();
  auto destination = make_tensor(row(4));
  ASSERT_TRUE(run.bind(output, destination.view).ok());
  auto before_queue = before_context.create_queue().value();
  auto after_queue = after_context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{environment.lane(0), &before_queue},
      hyle::graph::QueueBinding{environment.lane(2), &after_queue}};
  ASSERT_TRUE(run.prepare_replay(queues).ok());
  ASSERT_TRUE(run.submit(queues)->wait().ok());
  EXPECT_EQ(before_probe.effect_submit_count, 1U);
  EXPECT_EQ(after_probe.effect_submit_count, 0U);
  EXPECT_EQ(after_probe.submit_count, 1U);
  EXPECT_EQ(*state.value(), 43);
  EXPECT_EQ(read(destination, 4), (std::vector<float>{1.F, 2.F, 3.F, 4.F}));
  EXPECT_EQ(ready_probe.create_invocation_count, 0U);
  EXPECT_EQ(ready_probe.submit_count, 0U);
}

TEST(GraphRegions, ReadyOnlyRegionAndViewsExecuteWithoutAnyQueue) {
  ReadyConstantProbe ready;
  RegionProbe probe;
  configure_ready_constants(probe, ready);
  auto context = make_context(probe);
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(4)).value();
  const auto prepared = builder.add_node(Operation(Unary{UnaryKind::kNeg}),
                                         std::array{constant}).value();
  ready.nodes.push_back(prepared);
  const auto slice = hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(1),
      hyle::graph::Dimension::fixed(2)).value();
  const auto output = builder.add(slice, builder.result(prepared).value()).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(4));
  write(source, std::array{1.F, -2.F, 3.F, -4.F});
  const std::array constants{hyle::graph::ConstantBinding{
      constant, source.view.as_const(), hyle::graph::ConstantPolicy::kMaterialize}};
  auto compiled = hyle::graph::compile(graph, context, {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1U);
  EXPECT_EQ(compiled->report.segment_count(), 0U);
  EXPECT_EQ(compiled->report.workspace_bytes(), 0U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 0U);
  for (std::size_t step = 0; step < compiled->report.step_count(); ++step)
    EXPECT_EQ(compiled->report.step(step)->dispatch_count, 0U);
  auto run = compiled->executable.create_invocation().value();
  compiled->executable = {};
  ASSERT_TRUE(source.buffer.reset().ok());
  for (int replay = 0; replay < 3; ++replay) {
    if (replay != 0)
      ASSERT_TRUE(run.reset().ok());
    ASSERT_TRUE(run.prepare_replay({}).ok());
    auto completion = run.submit();
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    auto result = run.result(output).value();
    std::array<float, 2> values{};
    const auto tensor = result.tensor();
    ASSERT_TRUE(tensor) << tensor.error().message();
    ASSERT_TRUE(tensor->storage().read(
        std::as_writable_bytes(std::span(values))).ok());
    EXPECT_EQ(values, (std::array{2.F, -3.F}));
  }
  EXPECT_EQ(ready.factories, 1U);
  EXPECT_EQ(probe.create_invocation_count, 0U);
  EXPECT_EQ(probe.submit_count, 0U);
}

TEST(GraphRegions, ZeroDispatchRejectsIncompletePreparedOutputs) {
  for (const std::size_t limit : {0U, 1U}) {
    SCOPED_TRACE(limit);
    ReadyConstantProbe ready;
    ready.output_limit = limit;
    RegionProbe probe;
    configure_ready_constants(probe, ready);
    auto context = make_context(probe);
    const auto logical = make_ready_constant_graph(ready);
    auto source = make_tensor(row(4));
    write(source, std::array{1.F, 2.F, 3.F, 4.F});
    const std::array constants{hyle::graph::ConstantBinding{logical.constant, source.view.as_const()}};
    auto compiled = hyle::graph::compile(logical.graph, context, {.constants = constants});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsInternal());
    EXPECT_EQ(probe.compile_count, 1U);
    EXPECT_EQ(probe.destroy_prepared_count, 1U);
    EXPECT_EQ(probe.create_invocation_count, 0U);
    EXPECT_EQ(ready.factories, 1U);
    EXPECT_EQ(ready.destroys, 1U);
  }
}

TEST(GraphRegions, ZeroDispatchRejectsEffectsEvenWithEveryOutputPrepared) {
  for (const bool named : {false, true}) {
    SCOPED_TRACE(named);
    ReadyConstantProbe ready;
    RegionProbe probe;
    configure_ready_constants(probe, ready);
    probe.artifact_properties.preserves_effects = true;
    auto context = make_context(probe);
    ResourceStorage state;
    auto builder = GraphBuilder::create().value();
    const auto resource = builder.effect_resource().value();
    ready.nodes.push_back(named
        ? add_resource_touch(builder, resource, state, ResourceAccess::kReadWrite)
        : builder.add_node(Operation::custom(ExternalEffect{}), std::array<ValueId, 0>{}).value());
    const auto constant = builder.constant(row(4)).value();
    const auto prepared = builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{constant}).value();
    ready.nodes.push_back(prepared);
    const auto output = builder.add(Unary{UnaryKind::kNeg}, builder.result(prepared).value()).value();
    ASSERT_TRUE(builder.publish(output).ok());
    const auto graph = std::move(builder).finish().value();
    auto source = make_tensor(row(4));
    write(source, std::array{1.F, 2.F, 3.F, 4.F});
    const std::array constants{hyle::graph::ConstantBinding{constant, source.view.as_const()}};
    const std::array effects{EffectResourceBinding{resource, state.footprint()}};
    auto compiled = hyle::graph::compile(graph, context,
        {.constants = constants,
         .effect_bindings = named ? std::span<const EffectResourceBinding>(effects)
                                  : std::span<const EffectResourceBinding>{}});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsInternal());
    EXPECT_EQ(probe.observed_effect_count, named ? 1U : 0U);
    EXPECT_EQ(probe.observed_unknown_effects, !named);
    EXPECT_EQ(probe.compile_count, 1U);
    EXPECT_EQ(probe.destroy_prepared_count, 1U);
    EXPECT_EQ(probe.effect_submit_count, 0U);
    EXPECT_EQ(ready.destroys, 1U);
    EXPECT_EQ(*state.value(), 0);
  }
}

TEST(GraphRegions, ZeroDispatchNeverElidesAnOutputlessResourceEffect) {
  ResourceStorage state;
  auto builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  add_resource_touch(builder, resource, state, ResourceAccess::kWrite, 47);
  const auto graph = std::move(builder).finish().value();
  RegionProbe probe{.mode = ProposalMode::kResourceSingletons};
  probe.artifact_properties.runtime = kEffectRuntime;
  probe.artifact_properties.preserves_effects = true;
  probe.artifact_properties.dispatch_count = 0;
  probe.artifact_properties.invocation_private_bytes = 0;
  auto context = make_context(probe);
  const std::array effects{EffectResourceBinding{resource, state.footprint()}};
  auto compiled = hyle::graph::compile(graph, context, {.effect_bindings = effects});
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsInternal());
  EXPECT_EQ(probe.compile_count, 1U);
  EXPECT_EQ(probe.destroy_prepared_count, 1U);
  EXPECT_EQ(probe.effect_submit_count, 0U);
  EXPECT_EQ(*state.value(), 0);
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

TEST(GraphRegions, LongAlternativePrefixCannotHideAnUncoveredTail) {
  // Sixteen independent pair choices have two provider alternatives each.
  // The final Abs is covered by neither provider; exploring those 2^16
  // prefixes cannot repair it and must not consume the route-search budget.
  const auto logical = make_negations(32, true);
  RegionProbe first, second;
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      {.max_route_search_states = 64,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_FALSE(compiled);
  EXPECT_TRUE(compiled.error().IsUnimplemented()) << compiled.error().message();
  EXPECT_EQ(first.propose_count, 1u);
  EXPECT_EQ(second.propose_count, 1u);
  EXPECT_LE(first.region_resource_count, 16u);
  EXPECT_LE(second.region_resource_count, 16u);
  EXPECT_EQ(first.compile_count + second.compile_count, 0u);
  EXPECT_EQ(first.single_proposal_count + second.single_proposal_count, 0u);
  EXPECT_EQ(first.single_artifact_count + second.single_artifact_count, 0u);
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
        .source = hyle::graph::ExecutionSource::acquire(context).value(),
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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

struct ColdCopyResourceLifetime final {
  hyle::compute::ResourceBudget *budget = nullptr;
  std::size_t destroys = 0;
  std::size_t allocations_at_destroy = 0;
  std::size_t constant_charge_at_destroy = SIZE_MAX;
  std::size_t external_charge_at_destroy = SIZE_MAX;
};

struct ColdCopyResourceDeleter final {
  ColdCopyResourceLifetime *lifetime;
  void operator()(HostResource *resource) const noexcept {
    EXPECT_EQ(resource->live_allocations, 0U);
    lifetime->allocations_at_destroy = resource->allocation_count;
    if (lifetime->budget) {
      const auto usage = lifetime->budget->usage(
          hyle::mem::MemoryResourceRef(*resource));
      EXPECT_TRUE(usage);
      if (usage) {
        const auto &charges = usage->live_by_category;
        lifetime->constant_charge_at_destroy = charges[static_cast<std::size_t>(
            hyle::compute::ResourceBudgetCategory::kConstant)];
        lifetime->external_charge_at_destroy = charges[static_cast<std::size_t>(
            hyle::compute::ResourceBudgetCategory::kExternal)];
      }
    }
    ++lifetime->destroys;
    delete resource;
  }
};

struct ColdCopyResource final {
  std::unique_ptr<HostResource, ColdCopyResourceDeleter> resource;
  hyle::mem::MemoryResourceRef ref() const noexcept {
    return resource ? hyle::mem::MemoryResourceRef(*resource)
                    : hyle::mem::MemoryResourceRef{};
  }
};

TEST(GraphRegions,
     UnknownMaterializationObservationRetainsOwnersUntilNativeRetirement) {
  const std::array values{1.F, 2.F, 3.F, 4.F};
  const std::array second_values{-3.F, 7.F, -11.F, 13.F};
  hyle::compute::ResourceBudget budget;
  ColdCopyResourceLifetime lifetime{.budget = &budget};
  HostResource *tracked = nullptr;
  RegionProbe probe;
  probe.asynchronous_backend = true;
  probe.unknown_completion_observation = true;
  struct RetireOnExit final {
    RegionProbe &probe;
    ~RetireOnExit() {
      if (probe.abandoned_lifetime.valid())
        finish_abandoned_prefix(probe);
    }
  } cleanup{probe};
  {
    auto context = make_context(probe);
    ColdCopyResource resource{
        {new HostResource, ColdCopyResourceDeleter{&lifetime}}};
    tracked = resource.resource.get();
    auto adopted = hyle::compute::StorageOwner::adopt(context, std::move(resource));
    ASSERT_TRUE(adopted) << adopted.error().message();
    auto owner = std::move(*adopted);
    const std::array limits{
        hyle::compute::ResourceBudgetLimit{hyle::mem::system_resource(), SIZE_MAX},
        hyle::compute::ResourceBudgetLimit{owner.ref(), SIZE_MAX}};
    budget = hyle::compute::ResourceBudget::create({.limits = limits}).value();
    auto backing = Buffer::create(owner.ref(), 16, 64).value();
    ASSERT_TRUE(backing.write(0, std::as_bytes(std::span(values))).ok());
    auto ready = owner.adopt_buffer(std::move(backing), budget).value();
    auto second_backing = Buffer::create(owner.ref(), 16, 64).value();
    ASSERT_TRUE(second_backing.write(
        0, std::as_bytes(std::span(second_values))).ok());
    auto second_ready = owner.adopt_buffer(std::move(second_backing), budget).value();
    auto builder = GraphBuilder::create().value();
    const auto constant = builder.constant(row(4)).value();
    const auto second_constant = builder.constant(row(4)).value();
    ASSERT_TRUE(builder.publish(constant).ok());
    ASSERT_TRUE(builder.publish(second_constant).ok());
    const auto graph = std::move(builder).finish().value();
    const std::array bindings{hyle::graph::ConstantBinding{
        .value = constant,
        .tensor = ready.tensor(row(4)).value(),
        .policy = hyle::graph::ConstantPolicy::kMaterialize,
        .owner = owner,
        .lease = ready}, hyle::graph::ConstantBinding{
        .value = second_constant,
        .tensor = second_ready.tensor(row(4)).value(),
        .policy = hyle::graph::ConstantPolicy::kMaterialize,
        .owner = owner,
        .lease = second_ready}};
    const std::array storage{hyle::graph::StorageDomain{
        .resource = owner.ref(), .owner = owner}};
    const std::array targets{hyle::graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(),
        .default_storage = hyle::graph::StorageDomainId{0}}};
    auto compiled = hyle::graph::compile(graph,
        {.storage_domains = storage, .targets = targets},
        {.constants = bindings,
         .compilation_storage = {{.max_resident_bytes = 1U << 20,
                                  .max_peak_bytes = 1U << 20}},
         .resource_budget = budget});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsUnavailable());
    EXPECT_EQ(compiled.error().message(),
              "injected unknown preparation observation");
  } // Every caller-owned Context, resource owner, ready lease and plan is gone.
  EXPECT_EQ(probe.copy_count, 2U);
  EXPECT_EQ(probe.submit_count, 2U);
  EXPECT_EQ(probe.completion_wait_count, 1U);
  EXPECT_EQ(probe.completion_destroy_count, 1U);
  EXPECT_EQ(probe.completed_prefix, 0U);
  EXPECT_EQ(probe.compile_count, 0U);
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.context_destroy_count, 0U);
  ASSERT_EQ(lifetime.destroys, 0U);
  ASSERT_EQ(tracked->live_allocations, 4U); // two ready sources and two canonicals
  const auto usage = budget.usage(hyle::mem::MemoryResourceRef(*tracked));
  ASSERT_TRUE(usage);
  EXPECT_EQ(usage->live_by_category[static_cast<std::size_t>(
                hyle::compute::ResourceBudgetCategory::kConstant)], 32U);
  EXPECT_EQ(usage->live_by_category[static_cast<std::size_t>(
                hyle::compute::ResourceBudgetCategory::kExternal)], 32U);
  EXPECT_NE(probe.queued_copy_destinations[0].native_handle(),
            probe.queued_copy_destinations[1].native_handle());
  std::array<std::array<float, 4>, 2> actual{};
  for (std::size_t index = 0; index < actual.size(); ++index)
    ASSERT_TRUE(probe.queued_copy_destinations[index].read(
        std::as_writable_bytes(std::span(actual[index]))).ok());
  EXPECT_TRUE((actual[0] == values && actual[1] == second_values) ||
              (actual[1] == values && actual[0] == second_values));

  finish_abandoned_prefix(probe);
  EXPECT_FALSE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.completed_prefix, 2U);
  EXPECT_EQ(probe.copy_count, 2U); // Accepted bytes are never resubmitted.
  EXPECT_EQ(probe.context_destroy_count, 1U);
  EXPECT_EQ(lifetime.destroys, 1U);
  EXPECT_EQ(lifetime.allocations_at_destroy, 4U);
  EXPECT_EQ(lifetime.constant_charge_at_destroy, 0U);
  EXPECT_EQ(lifetime.external_charge_at_destroy, 0U);
}

TEST(GraphRegions,
     OwnerlessSystemMaterializationRetainsLeaseUntilNativeRetirement) {
  const std::array values{1.F, 2.F, 3.F, 4.F};
  hyle::compute::ResourceBudget budget;
  ColdCopyResourceLifetime lifetime{.budget = &budget};
  RegionProbe probe;
  probe.asynchronous_backend = true;
  probe.unknown_completion_observation = true;
  struct RetireOnExit final {
    RegionProbe &probe;
    ~RetireOnExit() {
      if (probe.abandoned_lifetime.valid())
        finish_abandoned_prefix(probe);
    }
  } cleanup{probe};
  {
    auto context = make_context(probe);
    ColdCopyResource resource{
        {new HostResource, ColdCopyResourceDeleter{&lifetime}}};
    auto adopted = hyle::compute::StorageOwner::adopt(context, std::move(resource));
    ASSERT_TRUE(adopted) << adopted.error().message();
    auto owner = std::move(*adopted);
    const std::array limits{
        hyle::compute::ResourceBudgetLimit{hyle::mem::system_resource(), SIZE_MAX},
        hyle::compute::ResourceBudgetLimit{owner.ref(), SIZE_MAX}};
    budget = hyle::compute::ResourceBudget::create({.limits = limits}).value();
    auto backing = Buffer::create(hyle::mem::system_resource(), 16, 64).value();
    ASSERT_TRUE(backing.write(0, std::as_bytes(std::span(values))).ok());
    auto ready = hyle::compute::ReadLease::adopt(std::move(backing),
        {.resource_budget = budget}).value();
    auto builder = GraphBuilder::create().value();
    const auto constant = builder.constant(row(4)).value();
    ASSERT_TRUE(builder.publish(constant).ok());
    const auto graph = std::move(builder).finish().value();
    const std::array bindings{hyle::graph::ConstantBinding{
        .value = constant,
        .tensor = ready.tensor(row(4)).value(),
        .policy = hyle::graph::ConstantPolicy::kMaterialize,
        .lease = ready}};
    const std::array storage{hyle::graph::StorageDomain{
        .resource = owner.ref(), .owner = owner}};
    const std::array targets{hyle::graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(),
        .default_storage = hyle::graph::StorageDomainId{0}}};
    auto compiled = hyle::graph::compile(graph,
        {.storage_domains = storage, .targets = targets},
        {.constants = bindings,
         .compilation_storage = {{.max_resident_bytes = 1U << 20,
                                  .max_peak_bytes = 1U << 20}},
         .resource_budget = budget});
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().IsUnavailable());
    EXPECT_EQ(compiled.error().message(),
              "injected unknown preparation observation");
  } // Source handles, destination owner and failed compile result are gone.
  EXPECT_EQ(probe.copy_count, 1U);
  EXPECT_EQ(probe.submit_count, 1U);
  EXPECT_EQ(probe.completion_wait_count, 1U);
  ASSERT_TRUE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.context_destroy_count, 0U);
  EXPECT_EQ(lifetime.destroys, 0U);
  const auto external_bytes = [&] {
    return budget.usage(hyle::mem::system_resource())
        ->live_by_category[static_cast<std::size_t>(
            hyle::compute::ResourceBudgetCategory::kExternal)];
  };
  EXPECT_EQ(external_bytes(), 16U);
  std::array<float, 4> actual{};
  ASSERT_TRUE(probe.queued_copy_destinations[0].read(
      std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, values);
  finish_abandoned_prefix(probe);
  EXPECT_FALSE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(probe.completed_prefix, 1U);
  EXPECT_EQ(probe.copy_count, 1U);
  EXPECT_EQ(external_bytes(), 0U);
  EXPECT_EQ(probe.context_destroy_count, 1U);
  EXPECT_EQ(lifetime.destroys, 1U);
  EXPECT_EQ(lifetime.constant_charge_at_destroy, 0U);
}

TEST(GraphRegions, QueuedMaterializationRejectsRawSourceBeforeCopy) {
  RegionProbe probe;
  probe.asynchronous_backend = true;
  probe.queued_transfers = true; // Queue capability is independent of wait faults.
  auto context = make_context(probe);
  ColdCopyResourceLifetime lifetime;
  ColdCopyResource resource{
      {new HostResource, ColdCopyResourceDeleter{&lifetime}}};
  auto *tracked = resource.resource.get();
  auto adopted = hyle::compute::StorageOwner::adopt(context, std::move(resource));
  ASSERT_TRUE(adopted) << adopted.error().message();
  auto owner = std::move(*adopted);
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(4)).value();
  ASSERT_TRUE(builder.publish(constant).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(4));
  const std::array values{1.F, 2.F, 3.F, 4.F};
  write(source, values);
  const std::array bindings{hyle::graph::ConstantBinding{
      .value = constant, .tensor = source.view.as_const(),
      .policy = hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array storage{hyle::graph::StorageDomain{
      .resource = owner.ref(), .owner = owner}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .source = hyle::graph::ExecutionSource::acquire(context).value(),
      .default_storage = hyle::graph::StorageDomainId{0}}};
  auto compiled = hyle::graph::compile(graph,
      {.storage_domains = storage, .targets = targets}, {.constants = bindings});
  ASSERT_FALSE(compiled); // Materialize must not silently become Borrow.
  EXPECT_TRUE(compiled.error().IsFailedPrecondition());
  EXPECT_NE(compiled.error().message().find("ready"), std::string_view::npos);
  EXPECT_NE(compiled.error().message().find("lease"), std::string_view::npos);
  EXPECT_NE(compiled.error().message().find("storage owner"), std::string_view::npos);
  EXPECT_GT(probe.transfer_query_count, 0U);
  EXPECT_EQ(probe.copy_count, 0U);
  EXPECT_EQ(probe.submit_count, 0U);
  EXPECT_EQ(probe.completion_wait_count, 0U);
  EXPECT_FALSE(probe.abandoned_lifetime.valid());
  EXPECT_EQ(tracked->live_allocations, 0U);
  EXPECT_EQ(read(source, 4), std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, FinalWorkspaceRejectionCanUseALeafPlan) {
  const auto logical = make_negations(2);
  RegionProbe probe;
  probe.artifact_properties.workspace = {.size = 128, .alignment = 64};
  auto context = make_context(probe);
  const std::array storage{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(), .workspace_budget_bytes = 64}};
  const std::array targets{hyle::graph::ExecutionTarget{
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
      hyle::graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(first).value(), hyle::graph::StorageDomainId{0}},
      hyle::graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(second).value(), hyle::graph::StorageDomainId{0}}};
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
        .source = hyle::graph::ExecutionSource::acquire(context).value(),
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
        hyle::graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(a).value(), hyle::graph::StorageDomainId{0}},
        hyle::graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(b).value(), hyle::graph::StorageDomainId{0}}};
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
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = hyle::graph::StorageDomainId{0}}};
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
  auto metadata = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  auto context = make_context(probe, metadata.get());
  const auto context_metadata = metadata.snapshot().live_bytes;
  CompileOptions options{
      .compilation_storage = hyle::graph::CompilationStorageLimits{},
      .operation_fallback = OperationFallbackPolicy::kReject};
  auto plain = hyle::graph::compile(logical.graph, context, options).value();
  ASSERT_EQ(plain.report.region_count(), 2U);
  const auto base = plain.report.compilation_storage().value();
  const auto base_metadata = metadata.snapshot().live_bytes;
  plain = {};
  EXPECT_EQ(metadata.snapshot().live_bytes, context_metadata);
  const auto private_peak = base.peak_bytes * 2;
  probe.compilation_storage = {.resident_bytes = 4096,
                               .peak_bytes = private_peak};
  options.compilation_storage->max_peak_bytes =
      base.peak_bytes + private_peak + 8192;
  auto charged = hyle::graph::compile(logical.graph, context, options);
  ASSERT_TRUE(charged) << charged.error().message();
  const auto usage = charged->report.compilation_storage().value();
  const auto charged_metadata = metadata.snapshot().live_bytes;
  ASSERT_GE(charged_metadata, base_metadata);
  EXPECT_EQ(usage.resident_bytes,
            base.resident_bytes + 8192 + charged_metadata - base_metadata);
  EXPECT_GE(usage.peak_bytes, private_peak);
  EXPECT_LT(usage.peak_bytes, 2 * private_peak);
  EXPECT_FALSE(usage.has_unknown_storage);
  *charged = {};
  EXPECT_EQ(metadata.snapshot().live_bytes, context_metadata);
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
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  const auto submitted = probe.submit_count;
  EXPECT_TRUE(second.prepare_replay(queues).IsFailedPrecondition());
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::visit_bound(second,
      +[](void *, const hyle::graph::detail::RegionAccessView &) noexcept {
        return Status::Ok();
      }, nullptr).IsFailedPrecondition());
  EXPECT_EQ(probe.submit_count, submitted);
  auto rejected = second.submit(queue);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_FALSE(second.in_flight());

  ASSERT_TRUE(first_completion->wait().ok());
  ASSERT_TRUE(second.prepare_replay(queues).ok());
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
    auto run = compiled->executable.create_invocation({.shape = point}).value();
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
  // Exact-only region discovery has no eligible node in this dynamic graph.
  EXPECT_EQ(probe.propose_count, 0U);
  EXPECT_EQ(probe.region_resource_count, 0U);
  EXPECT_EQ(probe.compile_count, 0U);
  EXPECT_EQ(compiled->report.region_count(), 0U);
  const std::array shape{hyle::graph::ShapeBinding{rows, 2}};
  auto run = compiled->executable.create_invocation({.shape = shape});
  ASSERT_TRUE(run.has_value()) << run.error().message();
  // None of the cold discovery/compilation callbacks are revisited.
  EXPECT_EQ(probe.propose_count, 0U);
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
  auto first = compiled->executable.create_invocation({.shape = small}).value();
  auto second = compiled->executable.create_invocation({.shape = large}).value();
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

TEST(GraphRegions, RetryRebuildsPhysicalInstanceAlignmentsWithoutRejectedRequirements) {
  const auto logical = make_negations(1);
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  first.requirement_override = hyle::compute::ResourceRequirement{
      .minimum_alignment = 256};
  first.requirement_port = 0;
  first.region_outcome_override = ProviderPrepareOutcome::resource_declined(
      Status::ResourceExhausted("strong-alignment candidate declines"));
  second.requirement_override = hyle::compute::ResourceRequirement{
      .minimum_alignment = alignof(float)};
  second.requirement_port = 0;
  auto context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(first.compile_count, 1U);
  EXPECT_EQ(first.destroy_prepared_count, 1U);
  EXPECT_EQ(second.compile_count, 1U);
  EXPECT_EQ(compiled->report.search().region_route_rejections, 1U);

  // These are parallel physical-instance arrays. A rejected partition must not
  // leave either extra entries or its stricter port requirement in the winner.
  const auto *plan = hyle::graph::detail::ExecutableAccess::state(compiled->executable);
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->instance_alignments.size(), plan->instance_values.size());
  EXPECT_EQ(plan->instance_alignments.size(), plan->instance_descriptions.size());
  EXPECT_EQ(plan->instance_alignments.size(), plan->instance_storage.size());
  for (std::size_t i = 0; i < (std::min)(plan->instance_values.size(),
                                        plan->instance_alignments.size()); ++i) {
    const auto expected = plan->instance_values[i] == logical.input.ordinal()
        ? alignof(float) : std::size_t{1};
    EXPECT_EQ(plan->instance_alignments[i], expected) << i;
  }
  expect_negation_replay(compiled->executable, logical, context);
}

TEST(GraphRegions, DomainConstantViewChecksActualPortAlignmentBeforeEnqueue) {
  auto made_builder = GraphBuilder::create();
  ASSERT_TRUE(made_builder) << made_builder.error().message();
  auto builder = std::move(*made_builder);
  const auto made_start = builder.dimension({0, 1, 1});
  ASSERT_TRUE(made_start) << made_start.error().message();
  const auto start = *made_start;
  const auto made_constant = builder.constant(row(8));
  ASSERT_TRUE(made_constant) << made_constant.error().message();
  const auto constant = *made_constant;
  const auto slice = hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension(start), hyle::graph::Dimension::fixed(4));
  ASSERT_TRUE(slice) << slice.error().message();
  const auto made_selected = builder.add(*slice, constant);
  ASSERT_TRUE(made_selected) << made_selected.error().message();
  // ViewSpec keeps its symbolic origin even when its result shape is fixed.
  // The ordinary builder therefore requires the known output spec explicitly.
  const auto output_spec = hyle::graph::TensorSpec::concrete(row(4));
  ASSERT_TRUE(output_spec) << output_spec.error().message();
  const auto made_output = builder.add(Unary{UnaryKind::kNeg}, *output_spec, *made_selected);
  ASSERT_TRUE(made_output) << made_output.error().message();
  const auto output = *made_output;
  ASSERT_TRUE(builder.publish(output).ok());
  auto finished = std::move(builder).finish();
  ASSERT_TRUE(finished) << finished.error().message();
  auto graph = std::move(*finished);
  RegionProbe probe;
  probe.bounded_unary_domain = true;
  probe.required_resource = hyle::mem::system_resource();
  probe.requirement_override = hyle::compute::ResourceRequirement{
      .minimum_alignment = 64};
  probe.requirement_port = 0;
  auto context = make_context(probe);
  auto source = make_tensor(row(8));
  write(source, std::array{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F});
  const std::array constants{hyle::graph::ConstantBinding{
      constant, source.view.as_const(), hyle::graph::ConstantPolicy::kMaterialize}};
  const auto fixture_instance = context.provider_instance_id(0);
  ASSERT_TRUE(fixture_instance) << fixture_instance.error().message();
  const auto unary_node = graph.node_id(1);
  ASSERT_TRUE(unary_node) << unary_node.error().message();
  const std::array unary_provider{hyle::graph::NodeProviderConstraint{
      *unary_node, kRegionProviderId, *fixture_instance}};
  auto compiled = hyle::graph::compile(graph, context,
      {.shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain,
       .constants = constants,
       .node_providers = unary_provider});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->executable.region_count(), 0U);
  ASSERT_EQ(compiled->report.fallback_operation_count(), 1U);
  ASSERT_EQ(probe.single_artifact_count, 1U);
  const auto *plan = hyle::graph::detail::ExecutableAccess::state(compiled->executable);
  ASSERT_NE(plan, nullptr);
  const auto *canonical = plan->owned_constant(0);
  ASSERT_NE(canonical, nullptr);
  ASSERT_GE(canonical->alignment(), 64U);
  auto made_queue = context.create_queue();
  ASSERT_TRUE(made_queue) << made_queue.error().message();
  auto queue = std::move(*made_queue);
  const std::array queues{hyle::graph::QueueBinding{
      .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue}};
  for (const auto actual_start : {0, 1}) {
    SCOPED_TRACE(actual_start);
    const std::array shape{hyle::graph::ShapeBinding{start, actual_start}};
    auto created = compiled->executable.create_invocation({.shape = shape});
    ASSERT_TRUE(created) << created.error().message();
    auto run = std::move(*created);
    auto destination = make_tensor(row(4));
    write(destination, std::array{99.F, 99.F, 99.F, 99.F});
    ASSERT_TRUE(run.bind(output, destination.view).ok());
    const auto submitted = probe.submit_count;
    const auto copied = probe.copy_count;
    const auto cold = run.prepare_replay(queues);
    if (actual_start == 0) {
      ASSERT_TRUE(cold.ok()) << cold.message();
      auto accepted = run.submit(queues);
      ASSERT_TRUE(accepted) << accepted.error().message();
      ASSERT_TRUE(accepted->wait().ok());
      EXPECT_EQ(read(destination, 4), (std::vector<float>{-1.F, -2.F, -3.F, -4.F}));
    } else {
      // The real offset is 4B even though the cold envelope's placeholder is
      // zero. The provider requires 64B and must never receive this binding.
      EXPECT_TRUE(cold.IsInvalidArgument()) << cold.message();
      auto denied = run.submit(queues);
      EXPECT_FALSE(denied);
      EXPECT_EQ(probe.submit_count, submitted);
      EXPECT_EQ(probe.copy_count, copied);
      EXPECT_EQ(read(destination, 4), (std::vector<float>{99.F, 99.F, 99.F, 99.F}));
    }
  }

  // Two individually 4B offsets produce a legal 8B root-relative origin.
  // The intermediate metadata-only view has no 8B consumer of its own.
  auto made_chain_builder = GraphBuilder::create();
  ASSERT_TRUE(made_chain_builder) << made_chain_builder.error().message();
  auto chain_builder = std::move(*made_chain_builder);
  const auto chain_start = chain_builder.dimension({0, 1, 1});
  ASSERT_TRUE(chain_start) << chain_start.error().message();
  const auto chain_constant = chain_builder.constant(row(8));
  ASSERT_TRUE(chain_constant) << chain_constant.error().message();
  const auto first_slice = hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension(*chain_start), hyle::graph::Dimension::fixed(6));
  ASSERT_TRUE(first_slice) << first_slice.error().message();
  const auto middle = chain_builder.add(*first_slice, *chain_constant);
  ASSERT_TRUE(middle) << middle.error().message();
  const auto second_slice = hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(1), hyle::graph::Dimension::fixed(4));
  ASSERT_TRUE(second_slice) << second_slice.error().message();
  const auto last_view = chain_builder.add(*second_slice, *middle);
  ASSERT_TRUE(last_view) << last_view.error().message();
  const auto chain_output = chain_builder.add(Unary{UnaryKind::kNeg}, *output_spec, *last_view);
  ASSERT_TRUE(chain_output) << chain_output.error().message();
  ASSERT_TRUE(chain_builder.publish(*chain_output).ok());
  auto chain_graph = std::move(chain_builder).finish();
  ASSERT_TRUE(chain_graph) << chain_graph.error().message();
  probe.requirement_override = hyle::compute::ResourceRequirement{
      .minimum_alignment = 8};
  const std::array chain_constants{hyle::graph::ConstantBinding{
      *chain_constant, source.view.as_const(), hyle::graph::ConstantPolicy::kMaterialize}};
  const auto chain_unary_node = chain_graph->node_id(2);
  ASSERT_TRUE(chain_unary_node) << chain_unary_node.error().message();
  const std::array chain_provider{hyle::graph::NodeProviderConstraint{
      *chain_unary_node, kRegionProviderId, *fixture_instance}};
  auto chain = hyle::graph::compile(*chain_graph, context,
      {.shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain,
       .constants = chain_constants,
       .node_providers = chain_provider});
  ASSERT_TRUE(chain) << chain.error().message();
  ASSERT_EQ(chain->executable.region_count(), 0U);
  ASSERT_EQ(chain->report.fallback_operation_count(), 1U);
  ASSERT_EQ(probe.single_artifact_count, 2U);
  const std::array chain_shape{hyle::graph::ShapeBinding{*chain_start, 1}};
  auto made_chain = chain->executable.create_invocation({.shape = chain_shape});
  ASSERT_TRUE(made_chain) << made_chain.error().message();
  auto chain_run = std::move(*made_chain);
  auto chain_destination = make_tensor(row(4));
  ASSERT_TRUE(chain_run.bind(*chain_output, chain_destination.view).ok());
  const auto prepared = chain_run.prepare_replay(queues);
  ASSERT_TRUE(prepared.ok()) << prepared.message();
  auto completed = chain_run.submit(queues);
  ASSERT_TRUE(completed) << completed.error().message();
  ASSERT_TRUE(completed->wait().ok());
  EXPECT_EQ(read(chain_destination, 4), (std::vector<float>{-3.F, -4.F, -5.F, -6.F}));
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
    {
      const std::array storage{hyle::graph::StorageDomain{
          .resource = hyle::mem::system_resource(),
          .workspace_budget_bytes = final_workspace_rejection ? 0u : SIZE_MAX}};
      const std::array targets{hyle::graph::ExecutionTarget{
          .source = hyle::graph::ExecutionSource::acquire(context).value(),
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
    } // Release retained execution sources before the public Context.
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

TEST(GraphRegions, RawBindingsAndOwnedResultsUseTheSameInvocationGate) {
  RegionProbe probe;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  Queue queue = context.create_queue().value();
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  const std::array<float, 8> values{-4, -3, -2, -1, 1, 2, 3, 4};
  write(input, values);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  auto result = invocation.result(logical.output).value();
  auto copy = result;
  EXPECT_TRUE(invocation.bind(logical.input, input.view).IsUnavailable());
  EXPECT_TRUE(invocation.bind(logical.output, output.view).IsUnavailable());
  EXPECT_TRUE(invocation.reset().IsUnavailable());
  EXPECT_TRUE(invocation.prepare_resources().IsUnavailable());
  EXPECT_TRUE(invocation.submit(queue).error().IsUnavailable());
  const std::array queues{hyle::graph::QueueBinding{
      hyle::graph::ExecutionLaneId{0}, &queue}};
  EXPECT_TRUE(invocation.prepare_replay(queues).IsUnavailable());
  EXPECT_TRUE(hyle::graph::detail::RegionAccess::visit(invocation,
      +[](void *, const hyle::graph::detail::RegionAccessView &) noexcept {
        return Status::Ok();
      }, nullptr).IsUnavailable());
  result.reset();
  EXPECT_TRUE(invocation.reset().IsUnavailable());
  copy.reset();
  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  result = invocation.result(logical.output).value();
  ASSERT_TRUE(invocation.close().ok());
  EXPECT_TRUE(invocation.bind(logical.input, input.view).IsFailedPrecondition());
  EXPECT_TRUE(invocation.drain().ok());
  compiled = {};
  invocation = {};
  EXPECT_EQ(probe.destroy_invocation_count, 0u);
  ASSERT_TRUE(result.tensor().has_value());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
  result.reset();
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
  EXPECT_TRUE(invocation.reset().IsUnavailable());
  EXPECT_TRUE(
      invocation.bind(logical.input, input.view).IsUnavailable());
  EXPECT_TRUE(invocation.submit(queue).error().IsUnavailable());
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

TEST(GraphPacking, UnknownPreparationObservationRetainsOwnersUntilNativeRetirement) {
  struct ResourceDeleter final {
    std::size_t *destroys;
    void operator()(HostResource *resource) const noexcept {
      EXPECT_EQ(resource->live_allocations, 0U);
      ++*destroys;
      delete resource;
    }
  };
  struct UniqueResource final {
    std::unique_ptr<HostResource, ResourceDeleter> resource;
    hyle::mem::MemoryResourceRef ref() const noexcept {
      return resource ? hyle::mem::MemoryResourceRef(*resource)
                      : hyle::mem::MemoryResourceRef{};
    }
  };
  for (const bool retry : {false, true}) {
    SCOPED_TRACE(retry);
    const std::size_t attempts = retry ? 2U : 1U;
    std::size_t resource_destroys = 0;
    HostResource *tracked = nullptr;
    hyle::compute::ResourceBudget budget;
    RegionProbe probe;
    probe.asynchronous_backend = true;
    probe.unknown_completion_observation = true;
    PackingProbe packing;
    configure_packing(probe, packing);
    packing.queue_preparation = true;
    packing.retry_failed_preparation = retry;
    packing.preparation_queue = &probe;
    packing.preparation_budget = &budget;
    packing.bounds = {16, 48};
    packing.extra_private_bytes = 32;
    // Also retire on an assertion exit, while every callback probe is alive.
    struct RetireOnExit final {
      RegionProbe &probe;
      ~RetireOnExit() {
        if (probe.abandoned_lifetime.valid())
          finish_abandoned_prefix(probe);
      }
    } cleanup{probe};
    {
      auto context = make_context(probe);
      UniqueResource resource{
          {new HostResource, ResourceDeleter{&resource_destroys}}};
      tracked = resource.resource.get();
      auto adopted = hyle::compute::StorageOwner::adopt(context, std::move(resource));
      ASSERT_TRUE(adopted) << adopted.error().message();
      auto owner = std::move(*adopted);
      const std::array limits{
          hyle::compute::ResourceBudgetLimit{hyle::mem::system_resource(), SIZE_MAX},
          hyle::compute::ResourceBudgetLimit{owner.ref(), SIZE_MAX}};
      budget = hyle::compute::ResourceBudget::create(
          {.limits = limits, .allow_unattributed = true}).value();
      auto backing = Buffer::create(owner.ref(), 16, 64).value();
      const std::array values{1.F, 2.F, 3.F, 4.F};
      ASSERT_TRUE(backing.write(0, std::as_bytes(std::span(values))).ok());
      auto ready = owner.adopt_buffer(std::move(backing), budget).value();
      const auto graph = make_packing_graph();
      const std::array constants{hyle::graph::ConstantBinding{
          .value = graph.weight,
          .tensor = ready.tensor(row(4)).value(),
          .owner = owner,
          .lease = ready}};
      packing.storage = owner.ref();
      packing.expected_source_resource = owner.ref();
      const std::array storage{hyle::graph::StorageDomain{
          .resource = owner.ref(), .owner = owner}};
      const std::array targets{hyle::graph::ExecutionTarget{
          .source = hyle::graph::ExecutionSource::acquire(context).value(),
          .default_storage = hyle::graph::StorageDomainId{0}}};
      auto compiled = hyle::graph::compile(graph.graph,
          {.storage_domains = storage, .targets = targets},
          {.constants = constants,
           .compilation_storage = {{.max_resident_bytes = 1U << 20,
                                    .max_peak_bytes = 1U << 20}},
           .resource_budget = budget,
           .operation_fallback = OperationFallbackPolicy::kReject,
           .max_region_proposals = 8,
           .max_region_proposals_per_provider = 4,
           .max_region_proposal_nodes = 16});
      ASSERT_FALSE(compiled);
      EXPECT_TRUE(compiled.error().IsUnavailable());
      EXPECT_EQ(compiled.error().message(),
                "injected unknown preparation observation");
    } // Drop caller Context, owner, ready lease, environment and failed compile.
    EXPECT_EQ(packing.factories.load(), attempts);
    EXPECT_EQ(packing.executions.load(), 0U);
    EXPECT_EQ(probe.compile_count, 0U); // No unfinished payload was published.
    EXPECT_EQ(probe.completion_wait_count, attempts);
    EXPECT_EQ(probe.completion_destroy_count, attempts);
    EXPECT_EQ(probe.completed_prefix, attempts - 1U);
    EXPECT_EQ(probe.submit_count, attempts);
    ASSERT_TRUE(probe.abandoned_lifetime.valid());
    EXPECT_EQ(packing.preparation_retires.load(), attempts - 1U);
    EXPECT_EQ(packing.destroys.load(), attempts - 1U);
    EXPECT_EQ(probe.context_destroy_count, 0U);
    ASSERT_EQ(resource_destroys, 0U);
    EXPECT_EQ(tracked->live_allocations, 3U); // ready source, payload, temporary
    EXPECT_EQ(budget.unattributed_usage().private_upper_bound_bytes, 48U);

    finish_abandoned_prefix(probe);
    EXPECT_FALSE(probe.abandoned_lifetime.valid());
    EXPECT_EQ(packing.preparation_retires.load(), attempts);
    EXPECT_EQ(packing.destroys.load(), attempts);
    EXPECT_EQ(probe.completed_prefix, attempts);
    EXPECT_EQ(probe.context_destroy_count, 1U);
    EXPECT_EQ(resource_destroys, 1U);
    EXPECT_EQ(budget.unattributed_usage().private_upper_bound_bytes, 0U);
  }
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

TEST(GraphRegions, DeferredPortsBindOriginalSourcesAndProjectOnlyAfterSuccess) {
  RegionProbe probe;
  probe.deferred_outputs = true;
  probe.artifact_properties.runtime = {
      .mode = hyle::compute::ProviderRuntimeMode::kDelegated,
      .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kProvider,
      .replay_class = ProviderReplayClass::kManaged,
      .blocks_host = true,
      .supports_concurrent_invocations = true};
  auto context = make_context(probe);
  const auto logical = make_negations(4);
  auto compiled = hyle::graph::compile(logical.graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 2U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 0U);
  EXPECT_EQ(compiled->executable.workspace_bytes(), 0U);
  for (const bool owned : {false, true}) {
    const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
    auto input = make_tensor(row(8));
    write(input, values);
    auto run = compiled->executable.create_invocation().value();
    const auto projections = probe.deferred_projections;
    if (owned) {
      ASSERT_TRUE(run.prepare_resources().ok());
      auto lease = hyle::compute::ReadLease::adopt(std::move(input.buffer)).value();
      ASSERT_TRUE(run.bind(logical.input, row(8), std::move(lease)).ok());
    } else {
      ASSERT_TRUE(run.bind(logical.input, input.view).ok());
    }
    ASSERT_TRUE(run.prepare_replay(std::span<const hyle::graph::QueueBinding>{}).ok());
    EXPECT_EQ(probe.deferred_projections, projections);
    ASSERT_FALSE(run.result(logical.output));
    for (int phase = 0; phase != 2; ++phase) {
      auto completion = run.submit(std::span<const hyle::graph::QueueBinding>{});
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      auto output = run.result(logical.output);
      ASSERT_TRUE(output) << output.error().message();
      auto tensor = output->tensor(); ASSERT_TRUE(tensor) << tensor.error().message();
      std::array<float, 8> actual{};
      ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, values);
      EXPECT_TRUE(run.prepare_replay(std::span<const hyle::graph::QueueBinding>{}).IsUnavailable());
      output->reset();
      ASSERT_TRUE(run.prepare_replay(std::span<const hyle::graph::QueueBinding>{}).ok());
    }
    auto output = run.result(logical.output); ASSERT_TRUE(output);
    ASSERT_TRUE(run.close().ok());
    run = {};
    auto tensor = output->tensor(); ASSERT_TRUE(tensor);
    std::array<float, 8> actual{};
    ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, values);
  }
  EXPECT_EQ(probe.deferred_references, 2U);
  EXPECT_EQ(probe.deferred_owned_inputs, 1U);
  EXPECT_EQ(probe.submit_count, 8U);
  EXPECT_EQ(probe.create_invocation_count, probe.destroy_invocation_count);
}

TEST(GraphRegions, OwnedNativeInputsKeepFirstCoveringPermissionAcrossRetryAndRebind) {
  RegionProbe probe;
  probe.deferred_outputs = true;
  probe.artifact_properties.runtime = {
      .mode = hyle::compute::ProviderRuntimeMode::kDelegated,
      .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kProvider,
      .replay_class = ProviderReplayClass::kManaged,
      .blocks_host = true,
      .supports_concurrent_invocations = true};
  auto context = make_context(probe);
  auto builder = GraphBuilder::create().value();
  std::array<ValueId, 3> inputs{}, outputs{};
  for (auto &input : inputs) input = builder.input(row(8)).value();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto first = builder.add(Unary{UnaryKind::kNeg}, inputs[i]).value();
    outputs[i] = builder.add(Unary{UnaryKind::kNeg}, first).value();
    ASSERT_TRUE(builder.publish(outputs[i]).ok());
  }
  auto graph = std::move(builder).finish().value();
  auto compiled = hyle::graph::compile(graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 3U);
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.prepare_resources().ok());
  auto pool = hyle::compute::LeasePool::create(hyle::mem::system_resource(), 128).value();
  const std::array<float, 16> values{1, 2, 3, 4, 5, 6, 7, 8,
                                    9, 10, 11, 12, 13, 14, 15, 16};
  auto writer = pool.allocate(sizeof(values), 64).value();
  ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());
  auto broad = writer.publish().value();
  auto narrow = broad.subview(4 * sizeof(float), 8 * sizeof(float)).value();
  auto replacement = make_tensor(row(16));
  const std::array<float, 16> changed{21, 22, 23, 24, 25, 26, 27, 28,
                                     29, 30, 31, 32, 33, 34, 35, 36};
  write(replacement, changed);
  auto next = hyle::compute::ReadLease::adopt(std::move(replacement.buffer)).value();
  // Binding order must not replace the original fixed-record first match.
  ASSERT_TRUE(run.bind(inputs[2], narrow.tensor(row(8)).value()).ok());
  ASSERT_TRUE(run.bind(inputs[1], row(8), narrow).ok());
  ASSERT_TRUE(run.bind(inputs[0], row(8), broad).ok());

  struct Observed final {
    hyle::graph::Invocation *run = nullptr;
    ValueId input{};
    ConstTensorView replacement;
    std::array<hyle::mem::ConstBufferView, 3> permissions{};
    std::array<bool, 3> has_permission{};
    std::size_t count = 0;
    bool reject_once = true;
  } observed{.run = &run, .input = inputs[0],
             .replacement = next.tensor(row(8)).value()};
  probe.deferred_binding_observer_context = &observed;
  probe.deferred_binding_observer = +[](void *opaque,
      const hyle::graph::RegionValue &input) noexcept -> Status {
    auto &seen = *static_cast<Observed *>(opaque);
    // Native hooks borrow the episode; reentrant public mutation must fail
    // before it can replace a lease while another port is being prepared.
    EXPECT_TRUE(seen.run->bind(seen.input, seen.replacement).IsUnavailable());
    EXPECT_TRUE(seen.run->reset().IsUnavailable());
    EXPECT_TRUE(seen.run->prepare_resources().IsUnavailable());
    if (seen.count >= seen.permissions.size())
      return Status::Internal("unexpected deferred test port count");
    const auto index = seen.count++;
    seen.has_permission[index] = input.lease && input.lease->valid();
    if (seen.has_permission[index]) {
      HYLE_ASSIGN_OR_RETURN(seen.permissions[index], input.lease->view());
    }
    if (std::exchange(seen.reject_once, false))
      return Status::Unavailable("injected cold binding failure");
    return Status::Ok();
  };
  const auto no_queues = std::span<const hyle::graph::QueueBinding>{};
  EXPECT_TRUE(run.prepare_replay(no_queues).IsUnavailable());
  ASSERT_EQ(observed.count, 1U);
  for (unsigned phase = 0; phase != 3; ++phase) {
    SCOPED_TRACE(phase);
    if (phase == 1)
      ASSERT_TRUE(run.bind(inputs[0], broad.tensor(row(8)).value()).ok());
    if (phase == 2)
      ASSERT_TRUE(run.bind(inputs[0], row(8), next).ok());
    observed.count = 0;
    observed.permissions = {};
    observed.has_permission = {};
    ASSERT_TRUE(run.prepare_replay(no_queues).ok());
    ASSERT_EQ(observed.count, 3U);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const bool owned = phase != 1 || i != 0;
      EXPECT_EQ(observed.has_permission[i], owned);
      if (!owned) continue;
      const auto expected = phase == 0 ? broad.view().value()
          : i == 0 ? next.view().value() : narrow.view().value();
      const auto actual = observed.permissions[i];
      EXPECT_EQ(actual.resource(), expected.resource());
      EXPECT_EQ(actual.native_handle(), expected.native_handle());
      EXPECT_EQ(actual.offset(), expected.offset());
      EXPECT_EQ(actual.size(), expected.size());
    }
    auto completion = run.submit(no_queues);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto result = run.result(outputs[i]);
      ASSERT_TRUE(result) << result.error().message();
      std::array<float, 8> actual{};
      ASSERT_TRUE(result->tensor()->storage().read(
          std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t j = 0; j < actual.size(); ++j)
        EXPECT_EQ(actual[j], i == 0 ? (phase != 2 ? values[j] : changed[j])
                                    : values[j + 4]);
    }
  }
}

TEST(GraphRegions, EmptyOwnedNativeInputKeepsItsValidZeroBytePermission) {
  RegionProbe probe;
  probe.deferred_outputs = true;
  probe.artifact_properties.runtime = {
      .mode = hyle::compute::ProviderRuntimeMode::kDelegated,
      .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kProvider,
      .replay_class = ProviderReplayClass::kManaged,
      .blocks_host = true,
      .supports_concurrent_invocations = true};
  auto context = make_context(probe);
  auto builder = GraphBuilder::create().value();
  const auto input = builder.input(row(0)).value();
  const auto first = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const auto output = builder.add(Unary{UnaryKind::kNeg}, first).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = hyle::graph::compile(graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 1U);
  auto pool = hyle::compute::LeasePool::create(hyle::mem::system_resource(), 32).value();
  auto writer = pool.allocate(32, 64).value();
  auto owner = writer.publish().value();
  const auto empty = owner.subview(owner.size(), 0).value();
  const auto expected = empty.view().value();
  struct Observed final {
    bool called = false;
    bool owned = false;
    hyle::mem::ConstBufferView permission;
  } observed;
  probe.deferred_binding_observer_context = &observed;
  probe.deferred_binding_observer = +[](void *opaque,
      const hyle::graph::RegionValue &input) noexcept -> Status {
    auto &seen = *static_cast<Observed *>(opaque);
    seen.called = true;
    seen.owned = input.lease && input.lease->valid();
    if (seen.owned) {
      HYLE_ASSIGN_OR_RETURN(seen.permission, input.lease->view());
    }
    return Status::Ok();
  };
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.prepare_resources().ok());
  ASSERT_TRUE(run.bind(input, row(0), empty).ok());
  const auto no_queues = std::span<const hyle::graph::QueueBinding>{};
  ASSERT_TRUE(run.prepare_replay(no_queues).ok());
  ASSERT_TRUE(observed.called);
  ASSERT_TRUE(observed.owned);
  EXPECT_EQ(observed.permission.resource(), expected.resource());
  EXPECT_EQ(observed.permission.native_handle(), expected.native_handle());
  EXPECT_EQ(observed.permission.offset(), expected.offset());
  EXPECT_EQ(observed.permission.size(), 0U);
  EXPECT_EQ(probe.deferred_owned_inputs, 1U);
}

TEST(GraphRegions, ExclusiveDeferredInputDoesNotCaptureItsUpgradeableReader) {
  RegionProbe probe;
  probe.deferred_outputs = true;
  probe.artifact_properties.runtime = {
      .mode = hyle::compute::ProviderRuntimeMode::kDelegated,
      .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kProvider,
      .replay_class = ProviderReplayClass::kManaged,
      .blocks_host = true,
      .supports_concurrent_invocations = true};
  auto context = make_context(probe);
  auto builder = GraphBuilder::create().value();
  auto fixed = builder.input(row(8)).value();
  auto activation = builder.input(row(8)).value();
  const auto pair = [&](ValueId input) {
    auto first = builder.add(Unary{UnaryKind::kNeg}, input).value();
    return builder.add(Unary{UnaryKind::kNeg}, first).value();
  };
  auto fixed_output = pair(fixed);
  auto activation_output = pair(activation);
  ASSERT_TRUE(builder.publish(fixed_output).ok());
  ASSERT_TRUE(builder.publish(activation_output).ok());
  auto graph = std::move(builder).finish().value();
  auto compiled = hyle::graph::compile(graph, context,
      {.operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.region_count(), 2U);
  auto pool = hyle::compute::LeasePool::create(hyle::mem::system_resource(), 64).value();
  auto writer = pool.allocate(8 * sizeof(float), 64).value();
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F, 7.F, -8.F};
  ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());
  auto other = make_tensor(row(8));
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.prepare_resources().ok());
  ASSERT_TRUE(run.bind(fixed, row(8), std::move(writer)).ok());
  for (int phase = 0; phase != 2; ++phase) {
    SCOPED_TRACE(phase);
    write(other, std::vector<float>(8, float(phase + 1)));
    // Even though fixed's binding is unchanged, rebind the other input to
    // prepare native references again after fixed's writer was published.
    ASSERT_TRUE(run.bind(activation, other.view).ok());
    ASSERT_TRUE(run.prepare_replay(std::span<const hyle::graph::QueueBinding>{}).ok());
    EXPECT_EQ(probe.deferred_owned_inputs, 0U);
    auto submitted = run.submit(std::span<const hyle::graph::QueueBinding>{});
    ASSERT_TRUE(submitted) << submitted.error().message();
    ASSERT_TRUE(submitted->wait().ok());
    auto output = run.result(fixed_output);
    ASSERT_TRUE(output);
    auto tensor = output->tensor(); ASSERT_TRUE(tensor);
    std::array<float, 8> actual{};
    ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, values);
  }
}

TEST(GraphRegions, DiscoveryPreservesTheExistingExternalEntryFrontier) {
  enum class Edges { kNewData, kNewControl, kIndependent, kSharedData,
                     kSharedControl, kReenterData, kReenterControl };
  for (const auto edges : {Edges::kNewData, Edges::kNewControl, Edges::kIndependent,
                           Edges::kSharedData, Edges::kSharedControl,
                           Edges::kReenterData, Edges::kReenterControl}) {
    SCOPED_TRACE(static_cast<int>(edges));
    auto builder = GraphBuilder::create().value();
    const auto input = builder.input(row(4)).value();
    const auto add = [&](ValueId source) {
      return builder.add_node(Operation(Unary{UnaryKind::kNeg}), std::array{source}).value();
    };
    const auto outside = add(input);
    const auto a = add(edges == Edges::kSharedData ? builder.result(outside).value() : input);
    const auto b = add(edges == Edges::kReenterData ? builder.result(a).value() : input);
    const auto c = add(edges == Edges::kNewData || edges == Edges::kReenterData
        ? builder.result(b).value()
        : edges == Edges::kSharedData ? builder.result(outside).value() : input);
    if (edges == Edges::kNewControl)
      ASSERT_TRUE(builder.depends_on(c, b).ok());
    if (edges == Edges::kSharedControl) {
      ASSERT_TRUE(builder.depends_on(a, outside).ok());
      ASSERT_TRUE(builder.depends_on(c, outside).ok());
    }
    if (edges == Edges::kReenterControl) {
      ASSERT_TRUE(builder.depends_on(b, a).ok());
      ASSERT_TRUE(builder.depends_on(c, b).ok());
    }
    for (const auto node : {outside, a, b, c})
      ASSERT_TRUE(builder.publish(builder.result(node).value()).ok());
    auto finished = std::move(builder).finish();
    ASSERT_TRUE(finished) << finished.error().message();
    const auto &graph = *finished;
    std::vector<std::vector<NodeId>> regions;
    const auto status = hyle::graph::detail::discover_regions(
        graph, hyle::mem::system_allocator(),
        [&](NodeId node) -> Result<bool> { return node == a || node == c; },
        [&](std::span<const NodeId> nodes) -> Result<bool> {
          regions.emplace_back(nodes.begin(), nodes.end());
          return true;
        });
    ASSERT_TRUE(status.ok()) << status.message();
    const bool split = edges == Edges::kNewData || edges == Edges::kNewControl ||
        edges == Edges::kReenterData || edges == Edges::kReenterControl;
    if (split)
      EXPECT_EQ(regions, (std::vector<std::vector<NodeId>>{{a}, {c}}));
    else
      EXPECT_EQ(regions, (std::vector<std::vector<NodeId>>{{a, c}}));
    for (const auto &region : regions) {
      auto contractible = hyle::graph::is_region_contractible(graph, region);
      ASSERT_TRUE(contractible) << contractible.error().message();
      EXPECT_TRUE(*contractible);
    }
  }
}

TEST(GraphConstants, BorrowedReplicasKeepViewOffsetsAndNeverCopyAtReplay) {
  HostResource primary_storage, replica_storage;
  const hyle::mem::MemoryResourceRef primary(primary_storage), replica(replica_storage);
  RegionProbe first{.mode = ProposalMode::kSingleton},
      second{.mode = ProposalMode::kSingleton};
  first.asynchronous_backend = second.asynchronous_backend = true;
  first.singleton_node = 1;
  second.singleton_node = 3;
  first.region_input_resource = first.region_output_resource = primary;
  second.region_input_resource = second.region_output_resource = replica;
  auto first_context = make_context(first), second_context = make_context(second);
  const Environment environment{{&first_context, primary}, {&second_context, replica}};
  auto builder = GraphBuilder::create().value();
  const auto constant = builder.constant(row(12)).value();
  const auto output_spec = hyle::graph::TensorSpec::concrete(row(4)).value();
  const auto first_view = builder.add(hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(1), hyle::graph::Dimension::fixed(4)).value(),
      constant).value();
  const auto first_output = builder.add(Unary{UnaryKind::kNeg}, output_spec, first_view).value();
  const auto second_view = builder.add(hyle::graph::ViewSpec::slice(0,
      hyle::graph::Dimension::fixed(5), hyle::graph::Dimension::fixed(4)).value(),
      constant).value();
  const auto second_output = builder.add(Unary{UnaryKind::kNeg}, output_spec, second_view).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  ASSERT_TRUE(builder.publish(second_view).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(12), primary);
  const std::array values{1.F, -2.F, 3.F, -4.F, 5.F, -6.F,
                          7.F, -8.F, 9.F, -10.F, 11.F, -12.F};
  write(source, values);
  const std::array bindings{hyle::graph::ConstantBinding{
      constant, source.view.as_const(), hyle::graph::ConstantPolicy::kBorrow}};
  const std::array placements{
      hyle::graph::NodePlacement{graph.node_id(1).value(), environment.target(0),
                                hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{graph.node_id(3).value(), environment.target(1),
                                hyle::graph::PlacementStrength::kRequired}};
  const std::array external{
      hyle::graph::ExternalResourceContract{constant, environment.storage(0)},
      hyle::graph::ExternalResourceContract{first_output, environment.storage(0)},
      hyle::graph::ExternalResourceContract{second_output, environment.storage(1)},
      hyle::graph::ExternalResourceContract{second_view, environment.storage(1)}};
  const auto primary_allocations = primary_storage.allocation_count;
  auto compiled = hyle::graph::compile(graph, environment.environment(),
      {.constants = bindings, .node_placements = placements,
       .external_resources = external,
       .operation_fallback = OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 2U);
  EXPECT_EQ(compiled->report.copy_bytes(), 0U);
  EXPECT_EQ(compiled->report.materialized_constant_bytes(), sizeof(values));
  EXPECT_EQ(first.copy_count + second.copy_count, 1U);
  EXPECT_EQ(primary_storage.allocation_count, primary_allocations);
  EXPECT_EQ(replica_storage.allocation_count, 1U);
  const auto *plan = hyle::graph::detail::ExecutableAccess::state(compiled->executable);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(plan->constant_bindings.size(), 1U);
  EXPECT_EQ(plan->constant_bindings[0].storage().resource(), primary);
  EXPECT_EQ(plan->constant_bindings[0].storage().native_handle(),
            source.view.storage().native_handle());
  EXPECT_EQ(plan->owned_constant(0), nullptr);
  std::size_t replica_roots = 0;
  for (std::size_t i = 0; i < plan->instance_values.size(); ++i) {
    if (plan->instance_values[i] != constant.ordinal() ||
        plan->instance_storage[i] != environment.storage(1).ordinal())
      continue;
    ++replica_roots;
    const auto *owner = plan->owned_instance_constant(i);
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(owner->view().resource(), replica);
    EXPECT_NE(owner->view().native_handle(), source.view.storage().native_handle());
    EXPECT_EQ(owner->view().size(), sizeof(values));
    std::array<float, 12> copied{};
    ASSERT_TRUE(owner->view().read(std::as_writable_bytes(std::span(copied))).ok());
    EXPECT_EQ(copied, values);
  }
  EXPECT_EQ(replica_roots, 1U);
  for (const auto &step : plan->steps)
    EXPECT_NE(step.kind, hyle::graph::detail::PlanStepKind::kCopy);

  auto first_queue = first_context.create_queue().value();
  auto second_queue = second_context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{environment.lane(0), &first_queue},
      hyle::graph::QueueBinding{environment.lane(1), &second_queue}};
  std::array<std::array<OwnedTensor, 2>, 2> outputs;
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  for (std::size_t slot = 0; slot != slots.size(); ++slot) {
    outputs[slot][0] = make_tensor(row(4), primary);
    outputs[slot][1] = make_tensor(row(4), replica);
    ASSERT_TRUE(slots[slot].bind(first_output, outputs[slot][0].view).ok());
    ASSERT_TRUE(slots[slot].bind(second_output, outputs[slot][1].view).ok());
    ASSERT_TRUE(slots[slot].prepare_replay(queues).ok());
  }
  for (int phase = 0; phase != 3; ++phase) {
    for (std::size_t slot = 0; slot != slots.size(); ++slot) {
      SCOPED_TRACE(::testing::Message() << "phase=" << phase << " slot=" << slot);
      if (phase == 2) {
        // Reset ordinary outputs, but the published immutable View must stay
        // bound to its destination-domain root without a new cold copy.
        ASSERT_TRUE(slots[slot].reset().ok());
        ASSERT_TRUE(slots[slot].bind(first_output, outputs[slot][0].view).ok());
        ASSERT_TRUE(slots[slot].bind(second_output, outputs[slot][1].view).ok());
        ASSERT_TRUE(slots[slot].prepare_replay(queues).ok());
      }
      for (auto &output : outputs[slot])
        write(output, std::array{99.F, 99.F, 99.F, 99.F});
      auto completion = slots[slot].submit(queues);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      EXPECT_EQ(read(outputs[slot][0], 4), (std::vector<float>{2.F, -3.F, 4.F, -5.F}));
      EXPECT_EQ(read(outputs[slot][1], 4), (std::vector<float>{6.F, -7.F, 8.F, -9.F}));
      auto alias = slots[slot].result(second_view);
      ASSERT_TRUE(alias) << alias.error().message();
      const auto tensor = alias->tensor();
      ASSERT_TRUE(tensor) << tensor.error().message();
      EXPECT_EQ(tensor->storage().resource(), replica);
      EXPECT_EQ(tensor->storage().offset(), 5 * sizeof(float));
      std::array<float, 4> selected{};
      ASSERT_TRUE(tensor->storage().read(std::as_writable_bytes(std::span(selected))).ok());
      EXPECT_EQ(selected, (std::array{-6.F, 7.F, -8.F, 9.F}));
      EXPECT_EQ(first.copy_count + second.copy_count, 1U);
    }
  }
  EXPECT_EQ(first.submit_count, 6U);
  EXPECT_EQ(second.submit_count, 6U);
}

TEST(GraphPacking, FiniteReplicasPackLocalWeightsAndChargeEachStorageOnce) {
  HostResource first_storage, second_storage;
  const hyle::mem::MemoryResourceRef first_resource(first_storage), second_resource(second_storage);
  PackingProbe first_packing, second_packing;
  RegionProbe first, second;
  configure_packing(first, first_packing);
  configure_packing(second, second_packing);
  first.asynchronous_backend = second.asynchronous_backend = true;
  first.region_input_resource = first.region_output_resource = first_resource;
  second.region_input_resource = second.region_output_resource = second_resource;
  const std::array gamma{1.25F, -2.5F, .75F, 4.F};
  first_packing.expected_source_resource = first_resource;
  second_packing.expected_source_resource = second_resource;
  first_packing.expected_source_values = second_packing.expected_source_values = gamma;
  auto first_context = make_context(first), second_context = make_context(second);
  const Environment environment{{&first_context, first_resource},
                                 {&second_context, second_resource}};
  auto builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({2, 3, 1}).value();
  const auto spec = hyle::graph::TensorSpec::dense(DataType::kFloat32,
      std::array{hyle::graph::Dimension(rows), hyle::graph::Dimension::fixed(4)}).value();
  const auto first_input = builder.input(spec).value();
  const auto second_input = builder.input(spec).value();
  const auto weight = builder.constant(row(4)).value();
  const auto first_output = builder.add(hyle::compute::RmsNorm{0}, spec,
                                        first_input, weight).value();
  const auto second_output = builder.add(hyle::compute::RmsNorm{0}, spec,
                                         second_input, weight).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  const auto graph = std::move(builder).finish().value();
  auto source = make_tensor(row(4), first_resource);
  write(source, gamma);
  const auto source_handle = source.view.storage().native_handle();
  const std::array bindings{hyle::graph::ConstantBinding{
      weight, source.view.as_const(), hyle::graph::ConstantPolicy::kMaterialize}};
  const std::array a{hyle::graph::ShapeBinding{rows, 2}}, b{hyle::graph::ShapeBinding{rows, 3}};
  const std::array variants{hyle::graph::ExactShapeVariant{a}, hyle::graph::ExactShapeVariant{b}};
  const std::array placements{
      hyle::graph::NodePlacement{graph.node_id(0).value(), environment.target(0),
                                hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{graph.node_id(1).value(), environment.target(1),
                                hyle::graph::PlacementStrength::kRequired}};
  const std::array external{
      hyle::graph::ExternalResourceContract{weight, environment.storage(0)},
      hyle::graph::ExternalResourceContract{first_input, environment.storage(0)},
      hyle::graph::ExternalResourceContract{second_input, environment.storage(1)},
      hyle::graph::ExternalResourceContract{first_output, environment.storage(0)},
      hyle::graph::ExternalResourceContract{second_output, environment.storage(1)}};
  CompileOptions options{
      .shape_policy = hyle::graph::CompileShapePolicy::kFiniteExact,
      .exact_variants = variants, .constants = bindings,
      .node_placements = placements, .external_resources = external,
      .operation_fallback = OperationFallbackPolicy::kReject};
  std::size_t family_bytes = 0;
  {
    auto measured = hyle::graph::compile(graph, environment.environment(), options);
    ASSERT_TRUE(measured) << measured.error().message();
    EXPECT_EQ(first.copy_count + second.copy_count, 2U);
    EXPECT_EQ(first_packing.factories.load(), 1U);
    EXPECT_EQ(second_packing.factories.load(), 1U);
    EXPECT_NE(first_packing.observed_source_handle, source_handle);
    EXPECT_NE(second_packing.observed_source_handle, source_handle);
    EXPECT_NE(first_packing.observed_source_handle, second_packing.observed_source_handle);
    const auto total = measured->report.total_compiled_storage();
    EXPECT_EQ(total.materialized_constant_bytes, 2 * sizeof(gamma));
    EXPECT_EQ(total.packed_constant_bytes, 2 * sizeof(gamma));
    for (std::size_t i = 0; i != variants.size(); ++i) {
      const auto report = measured->report.exact_variant(i);
      ASSERT_TRUE(report) << report.error().message();
      EXPECT_EQ(report->region_count(), 2U);
      EXPECT_EQ(report->copy_bytes(), 0U);
    }
    family_bytes = total.known_bytes();
  }
  ASSERT_GT(family_bytes, 0U);
  options.compiled_storage.max_bytes = family_bytes - 1;
  auto rejected = hyle::graph::compile(graph, environment.environment(), options);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted()) << rejected.error().message();
  options.compiled_storage.max_bytes = family_bytes;
  auto compiled = hyle::graph::compile(graph, environment.environment(), options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->report.total_compiled_storage().known_bytes(), family_bytes);
  const auto cold_copies = first.copy_count + second.copy_count;
  const auto cold_factories = first_packing.factories.load() + second_packing.factories.load();
  std::array children{compiled->executable.exact_variant(0).value(),
                      compiled->executable.exact_variant(1).value()};
  // Each exact child must retain both cold replicas and packed artifacts.
  *compiled = hyle::graph::CompiledGraph{};
  source = {};
  auto first_queue = first_context.create_queue().value();
  auto second_queue = second_context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{environment.lane(0), &first_queue},
      hyle::graph::QueueBinding{environment.lane(1), &second_queue}};
  std::array<std::array<OwnedTensor, 4>, 2> tensors;
  std::array slots{children[0].create_invocation().value(),
                   children[1].create_invocation().value()};
  for (std::size_t slot = 0; slot != slots.size(); ++slot) {
    const auto description = TensorDesc::dense(DataType::kFloat32,
        std::array<std::int64_t, 2>{static_cast<std::int64_t>(slot + 2), 4}).value();
    for (std::size_t i = 0; i != tensors[slot].size(); ++i)
      tensors[slot][i] = make_tensor(description, i % 2 == 0 ? first_resource : second_resource);
    ASSERT_TRUE(slots[slot].bind(first_input, tensors[slot][0].view).ok());
    ASSERT_TRUE(slots[slot].bind(second_input, tensors[slot][1].view).ok());
    ASSERT_TRUE(slots[slot].bind(first_output, tensors[slot][2].view).ok());
    ASSERT_TRUE(slots[slot].bind(second_output, tensors[slot][3].view).ok());
    ASSERT_TRUE(slots[slot].prepare_replay(queues).ok());
  }
  for (int phase = 0; phase != 2; ++phase) {
    for (std::size_t slot = 0; slot != slots.size(); ++slot) {
      const auto count = (slot + 2) * 4;
      const float sign = phase == 0 ? 1.F : -1.F;
      write(tensors[slot][0], std::vector<float>(count, sign));
      write(tensors[slot][1], std::vector<float>(count, -sign));
      auto completion = slots[slot].submit(queues);
      ASSERT_TRUE(completion) << completion.error().message();
      ASSERT_TRUE(completion->wait().ok());
      for (std::size_t target = 0; target != 2; ++target) {
        const auto actual = read(tensors[slot][target + 2], count);
        for (std::size_t i = 0; i != count; ++i)
          EXPECT_FLOAT_EQ(actual[i], (target == 0 ? sign : -sign) * gamma[i % 4]);
      }
      EXPECT_EQ(first.copy_count + second.copy_count, cold_copies);
      EXPECT_EQ(first_packing.factories.load() + second_packing.factories.load(), cold_factories);
    }
  }
}

TEST(SubmissionFrontier, PendingHostDependencyDoesNotHideAnIndependentLane) {
  auto metadata = hyle::mem::StatsAllocator::create(
      hyle::mem::system_allocator()).value();
  RegionProbe first_probe, dependent_probe, independent_probe;
  const auto storage = hyle::mem::system_resource();
  for (auto *probe : std::array{&first_probe, &dependent_probe,
                               &independent_probe}) {
    probe->asynchronous_backend = true;
    probe->required_resource = storage;
  }
  dependent_probe.dependency_wait_supported = false;
  auto first = make_context(first_probe, metadata.get());
  auto dependent = make_context(dependent_probe, metadata.get());
  auto independent = make_context(independent_probe, metadata.get());
  const Environment environment{{&first, storage}, {&dependent, storage},
                                {&independent, storage}};
  auto builder = GraphBuilder::create().value();
  const auto first_input = builder.input(row(8)).value();
  const auto independent_input = builder.input(row(8)).value();
  const auto produced = builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const auto dependent_output = builder.add(Unary{UnaryKind::kNeg}, produced).value();
  const auto independent_output =
      builder.add(Unary{UnaryKind::kNeg}, independent_input).value();
  ASSERT_TRUE(builder.publish(dependent_output).ok());
  ASSERT_TRUE(builder.publish(independent_output).ok());
  const auto graph = std::move(builder).finish().value();
  std::array<hyle::graph::NodePlacement, 3> placements;
  for (std::size_t i = 0; i != placements.size(); ++i)
    placements[i] = {graph.node_id(i).value(), environment.target(i),
                     hyle::graph::PlacementStrength::kRequired};
  CompileOptions options;
  options.node_placements = placements;
  options.region_selection = RegionSelectionPolicy::kDisabled;
  auto compiled = hyle::graph::compile(graph, environment.environment(), options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.segment_count(), 3u);
  ASSERT_EQ(compiled->report.segment(1)->dependency_count, 1u);
  ASSERT_EQ(compiled->report.segment(2)->dependency_count, 0u);
  auto run = compiled->executable.create_invocation().value();
  auto input = make_tensor(row(8)), side = make_tensor(row(8));
  auto output = make_tensor(row(8)), side_output = make_tensor(row(8));
  ASSERT_TRUE(run.bind(first_input, input.view).ok());
  ASSERT_TRUE(run.bind(independent_input, side.view).ok());
  ASSERT_TRUE(run.bind(dependent_output, output.view).ok());
  ASSERT_TRUE(run.bind(independent_output, side_output.view).ok());
  auto first_queue = first.create_queue().value();
  auto dependent_queue = dependent.create_queue().value();
  auto independent_queue = independent.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{environment.lane(0), &first_queue},
      hyle::graph::QueueBinding{environment.lane(1), &dependent_queue},
      hyle::graph::QueueBinding{environment.lane(2), &independent_queue}};
  ASSERT_TRUE(run.prepare_replay(queues).ok());
  ASSERT_EQ(run.replay_report()->host_wait_dependency_count, 1u);
  const auto cold_allocations = metadata.snapshot().total_allocs;
  const std::array<float, 8> sentinel{77, 77, 77, 77, 77, 77, 77, 77};
  for (int epoch = 0; epoch != 2; ++epoch) {
    SCOPED_TRACE(epoch);
    std::array<float, 8> values{1, -2, 3, -4, 5, -6, 7, -8};
    if (epoch != 0) for (auto &value : values) value = -value;
    write(input, values);
    write(side, values);
    write(output, sentinel);
    write(side_output, sentinel);
    auto started = hyle::graph::detail::SubmissionCursor::begin(run, queues);
    ASSERT_TRUE(started) << started.error().message();
    auto cursor = std::move(*started);
    auto advanced = cursor.advance();
    ASSERT_TRUE(advanced) << advanced.error().message();
    EXPECT_TRUE(*advanced);
    EXPECT_EQ(first_probe.submit_count, std::size_t(epoch + 1));
    advanced = cursor.advance();
    ASSERT_TRUE(advanced) << advanced.error().message();
    EXPECT_TRUE(*advanced); // B is pending, but C must already be accepted.
    EXPECT_EQ(dependent_probe.submit_count, std::size_t(epoch));
    EXPECT_EQ(independent_probe.submit_count, std::size_t(epoch + 1));
    EXPECT_EQ(read(output, 8),
              std::vector<float>(sentinel.begin(), sentinel.end()));
    EXPECT_TRUE(run.output(independent_output).error().IsFailedPrecondition());
    advanced = cursor.advance();
    ASSERT_TRUE(advanced) << advanced.error().message();
    EXPECT_FALSE(*advanced);
    EXPECT_NE(cursor.pending_dependency(), nullptr);
    EXPECT_EQ(first_probe.completion_wait_count, 0u);
    first_probe.completed_prefix = first_probe.submit_count;
    advanced = cursor.advance();
    ASSERT_TRUE(advanced) << advanced.error().message();
    EXPECT_TRUE(*advanced);
    ASSERT_TRUE(cursor.done());
    auto completed = cursor.finish();
    ASSERT_TRUE(completed) << completed.error().message();
    EXPECT_TRUE(run.in_flight());
    ASSERT_TRUE(completed->wait().ok());
    EXPECT_FALSE(run.in_flight());
    EXPECT_TRUE(run.output(dependent_output));
    EXPECT_TRUE(run.output(independent_output));
    EXPECT_EQ(read(output, 8), std::vector<float>(values.begin(), values.end()));
    for (auto &value : values) value = -value;
    EXPECT_EQ(read(side_output, 8),
              std::vector<float>(values.begin(), values.end()));
    EXPECT_EQ(metadata.snapshot().total_allocs, cold_allocations);
  }
}

struct CopyFrontierGraph final {
  Graph graph;
  ValueId direct_input, remote_input, direct_output, remote_output;
};

CopyFrontierGraph make_copy_frontier_graph() {
  auto builder = GraphBuilder::create().value();
  CopyFrontierGraph result;
  result.direct_input = builder.input(row(8)).value();
  result.remote_input = builder.input(row(8)).value();
  result.direct_output =
      builder.add(Unary{UnaryKind::kNeg}, result.direct_input).value();
  result.remote_output =
      builder.add(Unary{UnaryKind::kNeg}, result.remote_input).value();
  EXPECT_TRUE(builder.publish(result.direct_output).ok());
  EXPECT_TRUE(builder.publish(result.remote_output).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

TEST(SubmissionFrontier, SameQueueKeepsTheLatestAcceptedPrefixAfterReordering) {
  RegionProbe probe;
  probe.asynchronous_backend = true;
  probe.fixed_prefix_completions = true;
  probe.required_resource = hyle::mem::system_resource();
  auto context = make_context(probe);
  HostResource remote_storage;
  const std::array storages{
      hyle::graph::StorageDomain{hyle::mem::system_resource()},
      hyle::graph::StorageDomain{hyle::mem::MemoryResourceRef(remote_storage)}};
  const std::array targets{hyle::graph::ExecutionTarget{
      hyle::graph::ExecutionSource::acquire(context).value(),
      hyle::graph::StorageDomainId{0}}};
  const std::array lanes{
      hyle::graph::ExecutionLane{hyle::graph::ExecutionTargetId{0},
                                 hyle::graph::ExecutionLaneRole::kCompute},
      hyle::graph::ExecutionLane{hyle::graph::ExecutionTargetId{0},
                                 hyle::graph::ExecutionLaneRole::kTransfer}};
  const auto logical = make_copy_frontier_graph();
  const std::array resources{hyle::graph::ExternalResourceContract{
      logical.remote_input, hyle::graph::StorageDomainId{1}}};
  CompileOptions options;
  options.external_resources = resources;
  options.region_selection = RegionSelectionPolicy::kDisabled;
  auto compiled = hyle::graph::compile(logical.graph,
      {.storage_domains = storages, .targets = targets, .lanes = lanes}, options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.segment_count(), 3u);
  ASSERT_EQ(compiled->report.step(1)->kind, CompileStepKind::kCopy);
  ASSERT_EQ(compiled->report.segment(1)->dependency_count, 0u);
  auto run = compiled->executable.create_invocation().value();
  auto input = make_tensor(row(8));
  auto remote = make_tensor(row(8), hyle::mem::MemoryResourceRef(remote_storage));
  auto output = make_tensor(row(8)), remote_output = make_tensor(row(8));
  const std::array<float, 8> values{1, -2, 3, -4, 5, -6, 7, -8};
  const std::array<float, 8> sentinel{77, 77, 77, 77, 77, 77, 77, 77};
  write(input, values);
  write(remote, values);
  write(output, sentinel);
  write(remote_output, sentinel);
  ASSERT_TRUE(run.bind(logical.direct_input, input.view).ok());
  ASSERT_TRUE(run.bind(logical.remote_input, remote.view).ok());
  ASSERT_TRUE(run.bind(logical.direct_output, output.view).ok());
  ASSERT_TRUE(run.bind(logical.remote_output, remote_output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{0}, &queue},
      hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{1}, &queue}};
  ASSERT_TRUE(run.prepare_replay(queues).ok());
  auto started = hyle::graph::detail::SubmissionCursor::begin(run, queues);
  ASSERT_TRUE(started) << started.error().message();
  auto cursor = std::move(*started);
  const auto first = cursor.claim_ready();
  ASSERT_EQ(first, 0u);
  cursor.defer_segment(first);
  const auto copy = cursor.claim_ready();
  ASSERT_EQ(copy, 1u);
  auto copied = cursor.submit_segment(copy);
  ASSERT_TRUE(copied.status.ok()) << copied.status.message();
  ASSERT_TRUE(cursor.commit_segment(copy, std::move(copied)).ok());
  const auto later = cursor.claim_ready();
  ASSERT_EQ(later, first);
  auto executed = cursor.submit_segment(later);
  ASSERT_TRUE(executed.status.ok()) << executed.status.message();
  ASSERT_TRUE(cursor.commit_segment(later, std::move(executed)).ok());
  ASSERT_EQ(probe.fixed_prefix_count, 2u);
  ASSERT_EQ(probe.submit_count, 1u);
  ASSERT_EQ(probe.copy_count, 1u);
  probe.completed_prefix = 1; // The higher ordinal's older prefix has retired.
  cursor.reject_successors(Status::Unavailable("injected owner tail rejection"));
  auto completed = cursor.finish();
  ASSERT_TRUE(completed) << completed.error().message();
  const auto pending = completed->poll();
  ASSERT_TRUE(pending) << pending.error().message();
  EXPECT_EQ(*pending, hyle::compute::CompletionPoll::kPending);
  EXPECT_TRUE(run.in_flight());
  EXPECT_TRUE(run.output(logical.direct_output).error().IsFailedPrecondition());
  EXPECT_TRUE(completed->wait().IsUnavailable());
  EXPECT_EQ(probe.completed_prefix, 2u);
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_FALSE(run.in_flight());
  std::array<float, 8> negative = values;
  for (auto &value : negative) value = -value;
  EXPECT_EQ(read(output, 8), std::vector<float>(negative.begin(), negative.end()));
  EXPECT_EQ(read(remote_output, 8),
            std::vector<float>(sentinel.begin(), sentinel.end()));
  EXPECT_TRUE(run.output(logical.direct_output).error().IsFailedPrecondition());
}

TEST(SubmissionFrontier, ClaimedQueueGroupWaitsForOutcomeAndCancellationCollectsIt) {
  RegionProbe probe;
  probe.asynchronous_backend = true;
  probe.required_resource = hyle::mem::system_resource();
  auto context = make_context(probe);
  HostResource remote_storage;
  const std::array storages{
      hyle::graph::StorageDomain{hyle::mem::system_resource()},
      hyle::graph::StorageDomain{hyle::mem::MemoryResourceRef(remote_storage)}};
  const std::array targets{hyle::graph::ExecutionTarget{
      hyle::graph::ExecutionSource::acquire(context).value(),
      hyle::graph::StorageDomainId{0}}};
  const std::array lanes{
      hyle::graph::ExecutionLane{hyle::graph::ExecutionTargetId{0},
                                 hyle::graph::ExecutionLaneRole::kCompute},
      hyle::graph::ExecutionLane{hyle::graph::ExecutionTargetId{0},
                                 hyle::graph::ExecutionLaneRole::kTransfer}};
  const auto logical = make_copy_frontier_graph();
  const std::array resources{hyle::graph::ExternalResourceContract{
      logical.remote_input, hyle::graph::StorageDomainId{1}}};
  CompileOptions options;
  options.external_resources = resources;
  options.region_selection = RegionSelectionPolicy::kDisabled;
  auto compiled = hyle::graph::compile(logical.graph,
      {.storage_domains = storages, .targets = targets, .lanes = lanes}, options);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.segment_count(), 3u);
  ASSERT_EQ(compiled->report.segment(1)->dependency_count, 0u);
  auto queue = context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{0}, &queue},
      hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{1}, &queue}};
  const std::array<float, 8> values{1, -2, 3, -4, 5, -6, 7, -8};
  const std::array<float, 8> sentinel{77, 77, 77, 77, 77, 77, 77, 77};
  for (const bool cancel_before_commit : {false, true}) {
    SCOPED_TRACE(cancel_before_commit);
    auto run = compiled->executable.create_invocation().value();
    auto input = make_tensor(row(8));
    auto remote = make_tensor(row(8), hyle::mem::MemoryResourceRef(remote_storage));
    auto output = make_tensor(row(8)), remote_output = make_tensor(row(8));
    write(input, values);
    write(remote, values);
    write(output, sentinel);
    write(remote_output, sentinel);
    ASSERT_TRUE(run.bind(logical.direct_input, input.view).ok());
    ASSERT_TRUE(run.bind(logical.remote_input, remote.view).ok());
    ASSERT_TRUE(run.bind(logical.direct_output, output.view).ok());
    ASSERT_TRUE(run.bind(logical.remote_output, remote_output.view).ok());
    ASSERT_TRUE(run.prepare_replay(queues).ok());
    auto started = hyle::graph::detail::SubmissionCursor::begin(run, queues);
    ASSERT_TRUE(started) << started.error().message();
    auto cursor = std::move(*started);
    const auto claimed = cursor.claim_ready();
    ASSERT_EQ(claimed, 0u);
    EXPECT_EQ(cursor.claim_ready(), SIZE_MAX); // Independent copy, same Queue.
    auto outcome = cursor.submit_segment(claimed);
    ASSERT_TRUE(outcome.status.ok()) << outcome.status.message();
    EXPECT_EQ(cursor.claim_ready(), SIZE_MAX); // Returning math is not commit.
    auto premature = cursor.finish();
    ASSERT_FALSE(premature);
    EXPECT_TRUE(premature.error().IsFailedPrecondition());
    if (cancel_before_commit) {
      cursor.request_cancel();
      premature = cursor.finish();
      ASSERT_FALSE(premature);
      EXPECT_TRUE(premature.error().IsFailedPrecondition());
    }
    ASSERT_TRUE(cursor.commit_segment(claimed, std::move(outcome)).ok());
    if (!cancel_before_commit) {
      const auto released = cursor.claim_ready();
      ASSERT_EQ(released, 1u);
      cursor.defer_segment(released); // Do not accept the independent copy.
      cursor.request_cancel();
    }
    EXPECT_EQ(cursor.claim_ready(), SIZE_MAX);
    auto completed = cursor.finish();
    ASSERT_TRUE(completed) << completed.error().message();
    const auto pending = completed->poll();
    ASSERT_TRUE(pending) << pending.error().message();
    EXPECT_EQ(*pending, hyle::compute::CompletionPoll::kPending);
    EXPECT_TRUE(run.in_flight());
    EXPECT_TRUE(run.output(logical.direct_output).error().IsFailedPrecondition());
    EXPECT_TRUE(completed->wait().IsCancelled());
    EXPECT_FALSE(run.in_flight());
    EXPECT_EQ(probe.copy_count, 0u);
    EXPECT_EQ(read(remote_output, 8),
              std::vector<float>(sentinel.begin(), sentinel.end()));
    std::array<float, 8> negative = values;
    for (auto &value : negative) value = -value;
    EXPECT_EQ(read(output, 8),
              std::vector<float>(negative.begin(), negative.end()));
  }
}

TEST(SubmissionLifetime, UnacceptedFailureKeepsAdmissionUntilTheOriginalOwnerExits) {
  namespace gd = hyle::graph::detail;
  RegionProbe probe{.mode = ProposalMode::kPairs};
  probe.artifact_properties.runtime.supports_concurrent_invocations = false;
  const auto logical = make_negations(2);
  auto context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_FALSE(compiled->executable.supports_concurrent_invocations());
  auto first = compiled->executable.create_invocation().value();
  auto second = compiled->executable.create_invocation().value();
  auto *state = gd::InvocationAccess::state(first);
  ASSERT_FALSE(state->resources); // Retirement does not require lease bindings.
  ASSERT_TRUE(gd::prepare_submission_lifetime(*state, hyle::mem::system_allocator()).ok());
  EXPECT_FALSE(state->resources);
  EXPECT_TRUE(state->submission_lifetime);
  auto first_input = make_tensor(row(8)), first_output = make_tensor(row(8));
  auto second_input = make_tensor(row(8)), second_output = make_tensor(row(8));
  const std::array<float, 8> values{1, -2, 3, -4, 5, -6, 7, -8};
  const std::array<float, 8> sentinel{99, 99, 99, 99, 99, 99, 99, 99};
  write(first_input, values);
  write(second_input, values);
  write(first_output, sentinel);
  write(second_output, sentinel);
  ASSERT_TRUE(first.bind(logical.input, first_input.view).ok());
  ASSERT_TRUE(first.bind(logical.output, first_output.view).ok());
  ASSERT_TRUE(second.bind(logical.input, second_input.view).ok());
  ASSERT_TRUE(second.bind(logical.output, second_output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(first.prepare_replay(queues).ok());
  ASSERT_TRUE(second.prepare_replay(queues).ok());
  auto started = gd::SubmissionCursor::begin(first, queues);
  ASSERT_TRUE(started) << started.error().message();
  auto cursor = std::move(*started);
  const auto admitted = gd::begin_submission_lifetime(*state, {});
  ASSERT_TRUE(admitted.ok()) << admitted.message();
  const auto rejection = Status::Unavailable("driver rejected before any native acceptance");
  cursor.reject_successors(rejection);
  const auto rejected = cursor.finish(true);
  EXPECT_FALSE(rejected);
  if (!rejected) EXPECT_TRUE(rejected.error().IsUnavailable());
  EXPECT_TRUE(first.in_flight());
  EXPECT_EQ(state->phase.load(), gd::InvocationPhase::kRunning);
  EXPECT_FALSE(state->outputs_ready.load());
  EXPECT_TRUE(state->plan->exclusive_in_flight.load());
  EXPECT_TRUE(first.reset().IsUnavailable());
  EXPECT_FALSE(first.result(logical.output));
  auto blocked = second.submit(queues);
  EXPECT_FALSE(blocked);
  if (!blocked) EXPECT_TRUE(blocked.error().IsFailedPrecondition());
  else blocked->wait().IgnoreError();
  EXPECT_EQ(probe.submit_count, 0u);
  EXPECT_EQ(read(first_output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));
  EXPECT_EQ(read(second_output, 8), std::vector<float>(sentinel.begin(), sentinel.end()));

  // No native guard exists, but the unique owner's last slot access is still
  // an admission guard. Only returning it may publish reusable/no-output state.
  gd::finish_submission_lifetime(*state, rejection);
  EXPECT_FALSE(first.in_flight());
  EXPECT_EQ(state->phase.load(), gd::InvocationPhase::kIdle);
  EXPECT_FALSE(state->outputs_ready.load());
  EXPECT_FALSE(state->plan->exclusive_in_flight.load());
  EXPECT_TRUE(state->observation.available());
  EXPECT_TRUE(gd::submission_lifetime_status(*state).IsUnavailable());
  EXPECT_FALSE(first.result(logical.output));
  auto retried = first.submit(queues);
  ASSERT_TRUE(retried) << retried.error().message();
  ASSERT_TRUE(retried->wait().ok());
  auto independent = second.submit(queues);
  ASSERT_TRUE(independent) << independent.error().message();
  ASSERT_TRUE(independent->wait().ok());
  EXPECT_EQ(read(first_output, 8), std::vector<float>(values.begin(), values.end()));
  EXPECT_EQ(read(second_output, 8), std::vector<float>(values.begin(), values.end()));
}


namespace retention_scopes {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
c::ResourceBudget scope() {
  const std::array limits{c::ResourceBudgetLimit{m::system_resource()}};
  const std::array selected{m::system_resource()};
  return c::ResourceBudget::create({.limits=limits},selected,1u<<20).value();
}
std::size_t external(const c::ResourceBudget &budget) {
  return budget.usage(m::system_resource())->live_by_category[
      static_cast<std::size_t>(c::ResourceBudgetCategory::kExternal)];
}
void owned_deferred(RegionProbe &probe) {
  probe.deferred_outputs=true;probe.deferred_owned_output=true;
  probe.artifact_properties.runtime={
      .mode=c::ProviderRuntimeMode::kDelegated,
      .workspace_owner=c::ProviderWorkspaceOwner::kProvider,
      .replay_class=ProviderReplayClass::kManaged,
      .blocks_host=true,.supports_concurrent_invocations=true};
}
const std::array<float,8> values{1.F,-2.F,3.F,-4.F,5.F,-6.F,7.F,-8.F};
void expect_values(const c::ReadLease &lease) {
  std::array<float,8> actual{};
  ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual,values);
}

TEST(GraphResultRetentionScopes, GeneratedOwnerUsesOneExportAndDeduplicatesBothScopes) {
  auto local=scope(),global=scope();c::ReadLease escaped;
  {
    RegionProbe probe;owned_deferred(probe);auto context=make_context(probe);
    const auto logical=make_negations(2);
    auto compiled=g::compile(logical.graph,context,{.operation_fallback=OperationFallbackPolicy::kReject}).value();
    ASSERT_EQ(compiled.report.region_count(),1u);
    auto run=compiled.executable.create_invocation().value();
    auto input=make_tensor(row(8));write(input,values);
    ASSERT_TRUE(run.bind(logical.input,input.view).ok());
    ASSERT_TRUE(run.prepare_replay({}).ok());ASSERT_TRUE(run.submit()->wait().ok());
    auto result=run.result(logical.output).value();
    auto unbudgeted=result.storage_lease();ASSERT_TRUE(unbudgeted);expect_values(*unbudgeted);
    EXPECT_EQ(external(local),0u);EXPECT_EQ(external(global),0u);
    const auto before=probe.deferred_lease_exports;
    auto additional_only=result.storage_lease({},global);ASSERT_TRUE(additional_only);
    EXPECT_EQ(probe.deferred_lease_exports,before+1);
    EXPECT_EQ(external(global),32u);
    auto same=result.storage_lease(global,global);ASSERT_TRUE(same);
    EXPECT_EQ(external(global),32u);
    const auto exports=probe.deferred_lease_exports;
    auto pair=result.storage_lease(local,global);ASSERT_TRUE(pair);
    EXPECT_EQ(probe.deferred_lease_exports,exports+1); // no second native owner export
    EXPECT_EQ(external(local),32u);EXPECT_EQ(external(global),32u);
    EXPECT_EQ(pair->generation(),same->generation());
    auto piece=pair->subview(8,8).value();EXPECT_EQ(piece.size(),8u);
    EXPECT_EQ(external(local),32u);EXPECT_EQ(external(global),32u);
    EXPECT_FALSE(pair->try_write());
    escaped=std::move(*pair);
  }
  // The original slot/artifact/context no longer owns the backing; the one
  // exported immutable reader retains both completed-result charges.
  EXPECT_EQ(external(local),32u);EXPECT_EQ(external(global),32u);expect_values(escaped);
  escaped.reset();EXPECT_EQ(external(local),0u);EXPECT_EQ(external(global),0u);
}

TEST(GraphResultRetentionScopes, RejectedSecondScopePreservesFirstChargeAndResultForRetry) {
  auto local=scope(),global=scope();
  RegionProbe probe;owned_deferred(probe);auto context=make_context(probe);
  const auto logical=make_negations(2);
  auto compiled=g::compile(logical.graph,context,{.operation_fallback=OperationFallbackPolicy::kReject}).value();
  auto run=compiled.executable.create_invocation().value();
  auto input=make_tensor(row(8));write(input,values);
  ASSERT_TRUE(run.bind(logical.input,input.view).ok());
  ASSERT_TRUE(run.prepare_replay({}).ok());ASSERT_TRUE(run.submit()->wait().ok());
  auto result=run.result(logical.output).value();
  const auto usage=*global.shared_usage();
  auto credit=global.reserve(m::system_resource(),usage.limit_bytes-usage.live_bytes-1).value();
  const auto blocked_live=global.shared_usage()->live_bytes,blocked_external=external(global);
  const auto count=probe.deferred_lease_exports;
  auto rejected=result.storage_lease(local,global);ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(probe.deferred_lease_exports,count+1);
  EXPECT_EQ(external(local),32u);EXPECT_EQ(external(global),blocked_external);
  EXPECT_EQ(global.shared_usage()->live_bytes,blocked_live);
  std::array<float,8> observed{};
  ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed,values);
  credit.reset();
  auto retried=result.storage_lease(local,global);ASSERT_TRUE(retried);expect_values(*retried);
  EXPECT_EQ(external(local),32u);EXPECT_EQ(external(global),32u);
  result.reset();run={};compiled={};context={};
  EXPECT_EQ(external(local),32u);EXPECT_EQ(external(global),32u);
  retried->reset();EXPECT_EQ(external(local),0u);EXPECT_EQ(external(global),0u);
}

TEST(GraphResultRetentionScopes, SourceAliasSubviewKeepsTheSameDualAdmittedBacking) {
  auto local=scope(),global=scope();c::ReadLease escaped;
  {
    RegionProbe probe;auto context=make_context(probe);
    auto pool=c::LeasePool::create(m::system_resource(),512,
        {.max_blocks=1,.resource_budget=local},global).value();
    auto writer=pool.allocate(32,16).value();
    ASSERT_TRUE(writer.view()->write(std::as_bytes(std::span(values))).ok());
    auto ready=writer.publish().value();
    auto builder=GraphBuilder::create().value();const auto source=builder.input(row(8)).value();
    const auto alias=builder.add(c::View::slice(0,2,4).value(),source).value();
    ASSERT_TRUE(builder.publish(alias).ok());auto graph=std::move(builder).finish().value();
    auto compiled=g::compile(graph,context).value();auto run=compiled.executable.create_invocation().value();
    ASSERT_TRUE(run.prepare_resources().ok());ASSERT_TRUE(run.bind(source,row(8),ready).ok());
    ASSERT_TRUE(run.prepare_replay({}).ok());ASSERT_TRUE(run.submit()->wait().ok());
    auto result=run.result(alias).value();auto retained=result.storage_lease(local,global);ASSERT_TRUE(retained);
    EXPECT_EQ(retained->size(),16u);EXPECT_EQ(external(local),512u);EXPECT_EQ(external(global),512u);
    EXPECT_EQ(retained->generation(),ready.generation());escaped=std::move(*retained);
  }
  std::array<float,4> actual{};ASSERT_TRUE(escaped.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual,(std::array<float,4>{3.F,-4.F,5.F,-6.F}));
  EXPECT_EQ(external(local),512u);EXPECT_EQ(external(global),512u);
  escaped.reset();EXPECT_EQ(external(local),0u);EXPECT_EQ(external(global),0u);
}

TEST(GraphResultRetentionScopes, TransientFailureCanReleaseFirstFeeWithoutInvalidatingOriginalResult) {
  TransientHostResource resource;
  const auto make_scope=[&] {
    const std::array limits{c::ResourceBudgetLimit{resource.ref()},c::ResourceBudgetLimit{m::system_resource()}};
    const std::array selected{resource.ref(),m::system_resource()};
    return c::ResourceBudget::create({.limits=limits},selected,1u<<20).value();
  };
  const auto bytes=[&](const c::ResourceBudget &budget) {
    return budget.usage(resource.ref())->live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kExternal)];
  };
  auto local=make_scope(),global=make_scope();c::ReadLease escaped,uncharged;
  {
    RegionProbe probe;owned_deferred(probe);probe.deferred_owned_output=false;probe.deferred_transient_export=true;
    probe.region_output_resource=resource.ref();auto context=make_context(probe);
    const auto logical=make_negations(2);
    const std::array domains{g::StorageDomain{m::system_resource()},g::StorageDomain{resource.ref()}};
    const std::array targets{g::ExecutionTarget{g::ExecutionSource::acquire(context).value(),g::StorageDomainId{0}}};
    auto compiled=g::compile(logical.graph,{.storage_domains=domains,.targets=targets},
        {.operation_fallback=OperationFallbackPolicy::kReject}).value();
    auto run=compiled.executable.create_invocation().value();auto input=make_tensor(row(8));write(input,values);
    ASSERT_TRUE(run.bind(logical.input,input.view).ok());
    ASSERT_TRUE(run.prepare_replay({}).ok());ASSERT_TRUE(run.submit()->wait().ok());
    auto result=run.result(logical.output).value();
    uncharged=result.storage_lease().value();expect_values(uncharged);
    EXPECT_EQ(bytes(local),0u);EXPECT_EQ(bytes(global),0u);
    const auto usage=*global.shared_usage();
    auto credit=global.reserve(m::system_resource(),usage.limit_bytes-usage.live_bytes-1).value();
    const auto blocked=global.shared_usage()->live_bytes;
    auto rejected=result.storage_lease(local,global);ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsResourceExhausted());
    // The fresh failed export Domain has died. Its first fee is correctly
    // released, although the uncharged slot/root and earlier export still live.
    EXPECT_EQ(bytes(local),0u);EXPECT_EQ(bytes(global),0u);
    EXPECT_EQ(global.shared_usage()->live_bytes,blocked);
    EXPECT_EQ(resource.live_roots,1u);EXPECT_EQ(resource.live_tokens,2u);
    std::array<float,8> observed{};
    ASSERT_TRUE(result.tensor()->storage().read(std::as_writable_bytes(std::span(observed))).ok());
    EXPECT_EQ(observed,values);expect_values(uncharged);
    credit.reset();
    auto admitted=result.storage_lease(local,global);ASSERT_TRUE(admitted);expect_values(*admitted);
    EXPECT_EQ(bytes(local),32u);EXPECT_EQ(bytes(global),32u);
    escaped=std::move(*admitted);
    auto denied=make_scope();const auto denied_usage=*denied.shared_usage();
    auto denied_credit=denied.reserve(m::system_resource(),denied_usage.limit_bytes-denied_usage.live_bytes-1).value();
    auto failed_sibling=result.storage_lease(local,denied);ASSERT_FALSE(failed_sibling);
    EXPECT_TRUE(failed_sibling.error().IsResourceExhausted());
    // Releasing the transient failed sibling never clears an existing
    // successful charged export's same-scope canonical identity/fee.
    EXPECT_EQ(bytes(local),32u);EXPECT_EQ(bytes(global),32u);EXPECT_EQ(bytes(denied),0u);
    expect_values(escaped);expect_values(uncharged);
  }
  EXPECT_EQ(resource.live_roots,1u);EXPECT_EQ(resource.live_tokens,2u);
  EXPECT_EQ(bytes(local),32u);EXPECT_EQ(bytes(global),32u);expect_values(escaped);expect_values(uncharged);
  uncharged.reset();EXPECT_EQ(resource.live_tokens,1u);
  EXPECT_EQ(bytes(local),32u);EXPECT_EQ(bytes(global),32u);
  escaped.reset();EXPECT_EQ(resource.live_tokens,0u);EXPECT_EQ(resource.live_roots,0u);
  EXPECT_EQ(bytes(local),0u);EXPECT_EQ(bytes(global),0u);
}

} // namespace retention_scopes

} // namespace
