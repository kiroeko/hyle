#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
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
using hyle::compute::Queue;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::graph::CompileFallbackReason;
using hyle::graph::CompileOptions;
using hyle::graph::CompileReport;
using hyle::graph::CompileStepKind;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::OperationFallbackPolicy;
using hyle::graph::ProviderRegionPrepared;
using hyle::graph::RegionCompileRequest;
using hyle::graph::RegionNodeRange;
using hyle::graph::RegionProposal;
using hyle::graph::RegionProposalWriter;
using hyle::graph::RegionSelectionPolicy;
using hyle::graph::ValueId;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

constexpr hyle::compute::ProviderId kRegionProviderId{0x544553542d475241ULL,
                                                      0x50482d524547494fULL};

enum class ProposalMode : std::uint8_t {
  kPairs = 0,
  kWeightedOverlap,
  kOverflow,
  kCompileFailure,
  kInvocationFailure,
  kAsyncPairs,
  kAsyncRejectSecond,
};

struct RegionProbe final {
  ProposalMode mode = ProposalMode::kPairs;
  std::optional<std::uint64_t> proposal_savings = 10;
  std::uint32_t proposal_dispatch_count = 1;
  std::size_t propose_count = 0;
  std::size_t compile_count = 0;
  std::size_t single_prepare_count = 0;
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
  bool transfer_supported = true;
  bool dependency_wait_supported = true;
  bool capture_supported = false;
  bool capture_active = false;
  hyle::compute::detail::CompletionLifetime abandoned_lifetime{};
  Status abandoned_hint{};
  hyle::graph::RegionArtifactProperties artifact_properties{};
  std::uint32_t compiled_begin = (std::numeric_limits<std::uint32_t>::max)();
  std::uint32_t compiled_count = 0;
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

OwnedTensor make_tensor(TensorDesc description) {
  OwnedTensor result;
  result.buffer = Buffer::create(hyle::mem::system_resource(),
                                 description.required_bytes().value(), 64)
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

Status add_proposal(RegionProposalWriter &writer, std::uint32_t begin,
                    std::uint32_t count, std::optional<std::uint64_t> savings,
                    std::uint32_t dispatch_count = 1) noexcept {
  return writer.add(RegionProposal{
      .nodes = RegionNodeRange{begin, count},
      .cost = {.estimated_net_savings_ns = savings,
               .dispatch_count = dispatch_count},
      .route = (static_cast<std::uint64_t>(begin) << 32) | count,
  });
}

Status propose_regions(void *context, const Graph &graph,
                       RegionProposalWriter &writer) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  ++probe.propose_count;
  if (probe.mode == ProposalMode::kOverflow) {
    Status first = add_proposal(writer, 0, 2, 10);
    if (!first.ok())
      return first;
    return add_proposal(writer, 0, 2, 11);
  }
  if (probe.mode == ProposalMode::kWeightedOverlap) {
    Status first = add_proposal(writer, 0, 2, 10);
    if (!first.ok())
      return first;
    return add_proposal(writer, 1, 2, 20);
  }
  for (std::size_t index = 0; index + 1 < graph.node_count(); ++index) {
    if (!is_negation(graph, index) || !is_negation(graph, index + 1))
      continue;
    Status status =
        add_proposal(writer, static_cast<std::uint32_t>(index), 2,
                     probe.proposal_savings, probe.proposal_dispatch_count);
    if (!status.ok())
      return status;
    ++index;
  }
  return Status::Ok();
}

Status create_region_invocation(void *prepared, AllocatorRef allocator,
                                hyle::mem::BufferView, void **output) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  void *storage = allocator.allocate(sizeof(RegionRun), alignof(RegionRun));
  if (storage == nullptr)
    return Status::ResourceExhausted("region invocation allocation failed");
  *output = ::new (storage) RegionRun{plan.probe};
  ++plan.probe->create_invocation_count;
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

hyle::compute::detail::EnqueueResult
submit_region(void *prepared, void *invocation, Queue &,
              std::span<const ConstTensorView> inputs,
              std::span<const TensorView> outputs) noexcept {
  auto &plan = *static_cast<RegionPlan *>(prepared);
  if (invocation == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return hyle::compute::detail::EnqueueResult::accepted_command(
        Status::Internal("region binding mismatch"));
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
    ++plan.probe->submit_count;
    return hyle::compute::detail::EnqueueResult::accepted_command();
  }
  auto input_mapping = inputs[0].storage().map_host();
  if (!input_mapping)
    return hyle::compute::detail::EnqueueResult::accepted_command(
        std::move(input_mapping).error());
  auto output_mapping =
      outputs[0].storage().map_host(hyle::mem::HostAccess::kWrite);
  if (!output_mapping) {
    input_mapping->close().IgnoreError();
    return hyle::compute::detail::EnqueueResult::accepted_command(
        std::move(output_mapping).error());
  }
  auto writable = output_mapping->writable_bytes();
  if (!writable) {
    Status error = std::move(writable).error();
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return hyle::compute::detail::EnqueueResult::accepted_command(
        std::move(error));
  }
  const std::span<const std::byte> source = input_mapping->bytes();
  if (source.size() != writable->size() || source.size() % sizeof(float) != 0) {
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return hyle::compute::detail::EnqueueResult::accepted_command(
        Status::Internal("region byte size mismatch"));
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
  return hyle::compute::detail::EnqueueResult::accepted_command(
      std::move(terminal));
}

const hyle::graph::ProviderRegionPreparedVTable kPreparedRegionVTable{
    .destroy_prepared = &destroy_region_prepared,
    .alias_support = &region_alias_support,
    .create_invocation = &create_region_invocation,
    .destroy_invocation = &destroy_region_invocation,
    .submit = &submit_region,
};

Status compile_region(void *context, const RegionCompileRequest &request,
                      AllocatorRef allocator,
                      ProviderRegionPrepared *output) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  if (output == nullptr || request.graph == nullptr ||
      request.nodes.count < 2 || request.inputs.size() != 1 ||
      request.outputs.size() != 1) {
    return Status::InvalidArgument("region compile request is invalid");
  }
  for (std::uint32_t offset = 0; offset < request.nodes.count; ++offset) {
    if (!is_negation(*request.graph, request.nodes.begin + offset))
      return Status::InvalidArgument("region contains a non-negation node");
  }
  void *storage = allocator.allocate(sizeof(RegionPlan), alignof(RegionPlan));
  if (storage == nullptr)
    return Status::ResourceExhausted("region plan allocation failed");
  auto *plan = ::new (storage)
      RegionPlan{.probe = &probe, .negation_count = request.nodes.count};
  *output = ProviderRegionPrepared{
      .vtable = &kPreparedRegionVTable,
      .state = plan,
      .properties = probe.artifact_properties,
  };
  ++probe.compile_count;
  probe.compiled_begin = request.nodes.begin;
  probe.compiled_count = request.nodes.count;
  if (probe.mode == ProposalMode::kCompileFailure)
    return Status::Internal("injected region compile failure");
  return Status::Ok();
}

const hyle::graph::RegionProviderVTable kRegionVTable{
    .propose = &propose_regions,
    .compile = &compile_region,
};

Status create_provider_context(const Provider &provider, hyle::compute::Device,
                               void *, AllocatorRef, void **output) noexcept {
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  *output = payload.probe;
  return Status::Ok();
}

void destroy_provider_context(void *, AllocatorRef) noexcept {}

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

Status propose_single(
    void *context, Operation operation, std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        input_requirements,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        output_requirements,
    hyle::compute::detail::ProviderOperationProposal *proposal) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  ++probe.single_prepare_count;
  if (!operation.custom_as<ExternalEffect>().has_value())
    return Status::Unimplemented("region provider has no single-op route");
  if (!inputs.empty() || !outputs.empty() || !input_requirements.empty() ||
      !output_requirements.empty() || proposal == nullptr) {
    return Status::InvalidArgument("effect preparation binding mismatch");
  }
  *proposal = hyle::compute::detail::ProviderOperationProposal{
      .route = 1,
      .runtime = kEffectRuntime,
      .workspace = hyle::compute::WorkspaceRequirement{},
  };
  return Status::Ok();
}

Status prepare_single(
    void *context,
    const hyle::compute::detail::ProviderPrepareRequest &request,
    AllocatorRef,
    hyle::compute::detail::ProviderPrepared *prepared) noexcept {
  auto &probe = *static_cast<RegionProbe *>(context);
  if (!request.operation.custom_as<ExternalEffect>().has_value() ||
      !request.inputs.empty() || !request.outputs.empty() ||
      request.route != 1 || prepared == nullptr) {
    return Status::InvalidArgument("effect exact preparation mismatch");
  }
  *prepared = hyle::compute::detail::ProviderPrepared{
      .vtable = &kEffectPreparedVTable,
      .state = &probe,
      .runtime = kEffectRuntime,
  };
  return Status::Ok();
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

bool can_async_wait_queue(void *destination_queue,
                          hyle::compute::BackendId source_backend,
                          void *source_queue) noexcept {
  auto *destination = static_cast<RegionProbe *>(destination_queue);
  return destination != nullptr && source_backend == kAsyncBackendId &&
         source_queue != nullptr && destination->dependency_wait_supported;
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
    .can_wait_queue = &can_async_wait_queue,
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

Context make_context(RegionProbe &probe) {
  const std::array providers{region_provider(probe),
                             hyle::compute::native::provider()};
  const bool asynchronous = probe.mode == ProposalMode::kAsyncPairs ||
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

TEST(GraphRegions, CapturesAndReplaysACommandSegmentWithoutRewalkingItsSteps) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  probe.capture_supported = true;
  probe.artifact_properties.capture_compatible = true;
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  ASSERT_EQ(compiled.report.domain_count(), 1u);
  ASSERT_TRUE(compiled.report.domain(0).has_value());
  EXPECT_TRUE(compiled.report.domain(0)->capture_supported);
  EXPECT_EQ(compiled.report.command_segment_count(), 1u);
  EXPECT_EQ(compiled.report.capturable_segment_count(), 1u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{
      .domain = hyle::graph::ExecutionDomainId{0}, .queue = &queue}};

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
      .domain = hyle::graph::ExecutionDomainId{0}, .queue = nullptr}};
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
     SelectsTheHigherBenefitOverlappingIntervalDeterministically) {
  RegionProbe probe{.mode = ProposalMode::kWeightedOverlap};
  const NegationGraph logical = make_negations(3);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          CompileOptions{.region_selection =
                             RegionSelectionPolicy::kRequireEstimatedBenefit})
          .value();
  EXPECT_EQ(compiled.report.step_count(), 2u);
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compiled_begin, 1u);
  EXPECT_EQ(probe.compiled_count, 2u);
}

TEST(GraphRegions, RelocatesAnUnselectedRegionNodeToAWorkingFallbackDomain) {
  RegionProbe probe{.mode = ProposalMode::kWeightedOverlap};
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
  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &region_context,
                                   .workspace_resource = host},
      hyle::graph::ExecutionDomain{.context = &fallback_context,
                                   .workspace_resource = host},
  };
  const std::array resources{hyle::graph::ExternalResourceContract{
      .value = logical.input,
      .domain = hyle::graph::ExecutionDomainId{1},
      .resource = host,
  }};
  auto compiled =
      hyle::graph::compile(
          logical.graph, hyle::graph::CompileEnvironment{.domains = domains},
          CompileOptions{.external_resources = resources,
                         .region_selection =
                             RegionSelectionPolicy::kRequireEstimatedBenefit})
          .value();

  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 1u);
  EXPECT_EQ(probe.compiled_begin, 1u);
  EXPECT_EQ(probe.compiled_count, 2u);
  EXPECT_EQ(probe.single_prepare_count, 1u);
  bool fallback_on_second_domain = false;
  for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
    auto step = compiled.report.step(index);
    ASSERT_TRUE(step.has_value());
    fallback_on_second_domain =
        fallback_on_second_domain ||
        (step->kind == CompileStepKind::kOperationFallback &&
         step->domain == hyle::graph::ExecutionDomainId{1});
  }
  EXPECT_TRUE(fallback_on_second_domain);

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
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &region_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
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
  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &unused,
                                   .workspace_resource = host},
      hyle::graph::ExecutionDomain{.context = &active,
                                   .workspace_resource = host},
  };
  const std::array placements{hyle::graph::NodePlacement{
      .node = logical.graph.node_id().value(),
      .domain = hyle::graph::ExecutionDomainId{1},
      .strength = hyle::graph::PlacementStrength::kRequired,
  }};
  const std::array resources{
      hyle::graph::ExternalResourceContract{
          .value = logical.input,
          .domain = hyle::graph::ExecutionDomainId{1},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = logical.output,
          .domain = hyle::graph::ExecutionDomainId{1},
          .resource = host},
  };
  const auto compiled =
      hyle::graph::compile(logical.graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  EXPECT_FALSE(compiled.report.domain(0)->queue_required);
  EXPECT_TRUE(compiled.report.domain(1)->queue_required);
  EXPECT_EQ(compiled.report.backend(), hyle::compute::cpu::kBackendId);
  EXPECT_FALSE(compiled.report.mixed_backends());
}

TEST(GraphRegions, DefaultPolicyAcceptsUnknownEstimateAsExplicitOptIn) {
  RegionProbe probe{.proposal_savings = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(probe.compile_count, 1u);
  EXPECT_EQ(probe.single_prepare_count, 0u);
}

TEST(GraphRegions, StrictPolicyRejectsUnknownEstimate) {
  RegionProbe probe{.proposal_savings = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled =
      hyle::graph::compile(
          logical.graph, context,
          CompileOptions{.region_selection =
                             RegionSelectionPolicy::kRequireEstimatedBenefit})
          .value();
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.compile_count, 0u);
  EXPECT_EQ(probe.single_prepare_count, 2u);
}

TEST(GraphRegions, ExplicitZeroEstimateIsNotSelectedByDefault) {
  RegionProbe probe{.proposal_savings = 0};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 0u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 2u);
  EXPECT_EQ(probe.compile_count, 0u);
}

TEST(GraphRegions, DefaultPolicyUsesEarlierProviderAsTheFinalTieBreak) {
  RegionProbe first{.proposal_savings = std::nullopt};
  RegionProbe second{.proposal_savings = std::nullopt};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(first, second);
  auto compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.region_count(), 1u);
  EXPECT_EQ(first.compile_count, 1u);
  EXPECT_EQ(second.compile_count, 0u);
}

TEST(GraphRegions, DefaultPolicyPrefersFewerDispatchEpisodesBeforeProvider) {
  RegionProbe first{.proposal_savings = std::nullopt,
                    .proposal_dispatch_count = 2};
  RegionProbe second{.proposal_savings = std::nullopt,
                     .proposal_dispatch_count = 1};
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

TEST(GraphRegions, ProposalBudgetExhaustionFailsInsteadOfChangingThePlan) {
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context, CompileOptions{.max_region_proposals = 1});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(probe.compile_count, 0u);
  EXPECT_EQ(probe.single_prepare_count, 0u);
}

TEST(GraphRegions, PerProviderProposalBudgetIsAlsoAHardLimit) {
  RegionProbe probe{.mode = ProposalMode::kOverflow};
  const NegationGraph logical = make_negations(2);
  Context context = make_context(probe);
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      CompileOptions{.max_region_proposals = 4,
                     .max_region_proposals_per_provider = 1});
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsResourceExhausted());
  EXPECT_EQ(probe.compile_count, 0u);
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
      .blocks_host = true,
      .uses_private_threads = true,
      .supports_concurrent_invocations = false,
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
  EXPECT_TRUE(step->blocks_host);
  EXPECT_TRUE(step->uses_private_threads);
  EXPECT_FALSE(step->supports_concurrent_invocations);
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
  probe.artifact_properties.supports_concurrent_invocations = false;
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

  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &first,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
      hyle::graph::ExecutionDomain{.context = &second,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
  };
  std::array<hyle::graph::NodePlacement, 6> placements{};
  for (std::size_t index = 0; index < placements.size(); ++index) {
    placements[index] = hyle::graph::NodePlacement{
        .node = graph.node_id(index).value(),
        .domain =
            hyle::graph::ExecutionDomainId{index >= 2 && index < 4 ? 1u : 0u},
        .strength = hyle::graph::PlacementStrength::kRequired,
    };
  }
  auto compiled =
      hyle::graph::compile(graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
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
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &first_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
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

TEST(GraphRegions, MultiDomainIndependentSinksUseANonblockingAggregate) {
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

  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &first,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
      hyle::graph::ExecutionDomain{.context = &second,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = graph.node_id(0).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(1).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(2).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(3).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  const auto host = hyle::mem::system_resource();
  const std::array resources{
      hyle::graph::ExternalResourceContract{
          .value = first_input,
          .domain = hyle::graph::ExecutionDomainId{0},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = second_input,
          .domain = hyle::graph::ExecutionDomainId{1},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = first_output,
          .domain = hyle::graph::ExecutionDomainId{0},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = second_output,
          .domain = hyle::graph::ExecutionDomainId{1},
          .resource = host},
  };
  auto compiled =
      hyle::graph::compile(graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
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
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &first_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
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

  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &first,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
      hyle::graph::ExecutionDomain{.context = &second,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = graph.node_id(0).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(1).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(2).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = graph.node_id(3).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  const auto host = hyle::mem::system_resource();
  const std::array resources{
      hyle::graph::ExternalResourceContract{
          .value = first_input,
          .domain = hyle::graph::ExecutionDomainId{0},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = second_input,
          .domain = hyle::graph::ExecutionDomainId{1},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = first_output,
          .domain = hyle::graph::ExecutionDomainId{0},
          .resource = host},
      hyle::graph::ExternalResourceContract{
          .value = second_output,
          .domain = hyle::graph::ExecutionDomainId{1},
          .resource = host},
  };
  auto compiled =
      hyle::graph::compile(graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
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
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &first_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
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

TEST(GraphRegions, MultiDomainPlacementKeepsEligibleProviderRegions) {
  RegionProbe first_probe;
  RegionProbe second_probe;
  const NegationGraph logical = make_negations(4);
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);
  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &first,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
      hyle::graph::ExecutionDomain{.context = &second,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(2).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(3).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled =
      hyle::graph::compile(logical.graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
                           CompileOptions{.node_placements = placements})
          .value();
  EXPECT_EQ(compiled.report.region_count(), 2u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 0u);
  ASSERT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.step(0)->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled.report.step(0)->domain, hyle::graph::ExecutionDomainId{0});
  EXPECT_EQ(compiled.report.step(1)->kind, CompileStepKind::kCopy);
  EXPECT_EQ(compiled.report.step(2)->kind, CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled.report.step(2)->domain, hyle::graph::ExecutionDomainId{1});
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
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &first_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
                                .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  EXPECT_EQ(read(output, values.size()),
            std::vector<float>(values.begin(), values.end()));
}

TEST(GraphRegions, SharedQueueReservesTheCombinedDomainPrefixWithoutAWait) {
  RegionProbe probe{.mode = ProposalMode::kAsyncPairs};
  const NegationGraph logical = make_negations(4);
  Context context = make_context(probe);
  const auto host = hyle::mem::system_resource();
  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &context,
                                   .workspace_resource = host},
      hyle::graph::ExecutionDomain{.context = &context,
                                   .workspace_resource = host},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(2).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(3).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled =
      hyle::graph::compile(logical.graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
                           CompileOptions{.node_placements = placements})
          .value();
  ASSERT_EQ(compiled.report.segment_count(), 2u);
  EXPECT_EQ(compiled.report.synchronization_count(), 1u);

  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  auto invocation = compiled.executable.create_invocation().value();
  Queue queue = context.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
                                .queue = &queue},
  };
  EXPECT_TRUE(invocation.prepare_replay(queues).IsFailedPrecondition());
  EXPECT_EQ(probe.reserve_completion_count, 0u);
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto replay = invocation.replay_report().value();
  EXPECT_EQ(replay.ordered_dependency_count, 1u);
  EXPECT_EQ(replay.device_wait_dependency_count, 0u);
  EXPECT_EQ(replay.host_wait_dependency_count, 0u);
  EXPECT_EQ(invocation.dependency_route(1, 0).value(),
            hyle::graph::ReplayDependencyRoute::kOrderedQueue);
  EXPECT_EQ(probe.reserve_completion_count, 1u);
  EXPECT_EQ(probe.last_reserved_completions, 2u);
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  EXPECT_EQ(probe.device_wait_count, 0u);
  EXPECT_EQ(probe.completion_wait_count, 1u);
  EXPECT_EQ(probe.transfer_query_count, 1u);
}

TEST(GraphRegions, RejectsAnUnavailableTransferBeforeAnySegmentIsSubmitted) {
  RegionProbe first_probe{.mode = ProposalMode::kAsyncPairs};
  RegionProbe second_probe{.mode = ProposalMode::kAsyncPairs};
  second_probe.transfer_supported = false;
  const NegationGraph logical = make_negations(2);
  Context first = make_context(first_probe);
  Context second = make_context(second_probe);
  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &first,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
      hyle::graph::ExecutionDomain{.context = &second,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled =
      hyle::graph::compile(logical.graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
                           CompileOptions{.node_placements = placements})
          .value();
  OwnedTensor input = make_tensor(row(8));
  OwnedTensor output = make_tensor(row(8));
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.input, input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &first_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
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
  const std::array domains{
      hyle::graph::ExecutionDomain{.context = &first,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
      hyle::graph::ExecutionDomain{.context = &second,
                                   .workspace_resource =
                                       hyle::mem::system_resource()},
  };
  const std::array placements{
      hyle::graph::NodePlacement{.node = logical.graph.node_id(0).value(),
                                 .domain = hyle::graph::ExecutionDomainId{0},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{.node = logical.graph.node_id(1).value(),
                                 .domain = hyle::graph::ExecutionDomainId{1},
                                 .strength =
                                     hyle::graph::PlacementStrength::kRequired},
  };
  auto compiled =
      hyle::graph::compile(logical.graph,
                           hyle::graph::CompileEnvironment{.domains = domains},
                           CompileOptions{.node_placements = placements})
          .value();
  EXPECT_EQ(compiled.report.synchronization_count(), 1u);

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
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{0},
                                .queue = &first_queue},
      hyle::graph::QueueBinding{.domain = hyle::graph::ExecutionDomainId{1},
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
