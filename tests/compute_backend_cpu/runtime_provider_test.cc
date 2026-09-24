#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <thread>
#include <utility>

#include "backend_internal.h"
#include "completion_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/delegated_operation.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/quota_allocator.h"
#include "provider_internal.h"

namespace {

struct ProviderProbe final {
  std::array<std::size_t, 16> created{};
  std::array<std::size_t, 16> destroyed{};
  std::array<std::size_t, 16> prepared{};
  std::size_t create_count = 0;
  std::size_t destroy_count = 0;
  std::size_t prepare_count = 0;
  std::size_t fail_index = static_cast<std::size_t>(-1);
};

struct ProviderPayload final {
  ProviderProbe *probe = nullptr;
  std::size_t index = 0;
};

constexpr hyle::compute::ProviderExtensionId kProbeExtensionId{
    0x5445535400000001ULL, 0x455854454e53494fULL, 1};

struct ProbeExtensionInterface final {
  std::uint32_t marker = 0;
};

constexpr ProbeExtensionInterface kProbeExtension{0x48594c45U};

constexpr hyle::compute::BackendExtensionId kBackendProbeExtensionId{
    0x5445535400000002ULL, 0x4241434b454e4401ULL, 1};
constexpr ProbeExtensionInterface kBackendProbeExtension{0x43554441U};

hyle::Status create_provider_context(const hyle::compute::Provider &provider,
                                     hyle::compute::Device, void *,
                                     hyle::mem::AllocatorRef,
                                     void **context) noexcept {
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->created[payload.probe->create_count++] = payload.index;
  if (payload.index == payload.probe->fail_index) {
    *context = nullptr;
    return hyle::Status::InvalidArgument("injected provider failure");
  }
  *context = const_cast<hyle::compute::Provider *>(&provider);
  return hyle::Status::Ok();
}

void destroy_provider_context(void *context, hyle::mem::AllocatorRef) noexcept {
  const auto &provider = *static_cast<hyle::compute::Provider *>(context);
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->destroyed[payload.probe->destroy_count++] = payload.index;
}

hyle::Status
propose_provider(void *context,
                 const hyle::compute::detail::ProviderOperationRequest &,
                 std::span<hyle::compute::detail::ResourceBindingRequirement>,
                 std::span<hyle::compute::detail::ResourceBindingRequirement>,
                 hyle::compute::detail::ProviderOperationProposal *) noexcept {
  const auto &provider = *static_cast<hyle::compute::Provider *>(context);
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->prepared[payload.probe->prepare_count++] = payload.index;
  return hyle::Status::Unimplemented("probe provider declines operation");
}

hyle::compute::ProviderPrepareOutcome
prepare_provider(void *, const hyle::compute::detail::ProviderPrepareRequest &,
                 hyle::mem::AllocatorRef,
                 hyle::compute::detail::ProviderPrepared *) noexcept {
  return hyle::compute::ProviderPrepareOutcome::unsupported(
      hyle::Status::Unimplemented("probe provider declines operation"));
}

hyle::Status drain_provider_context(void *) noexcept {
  return hyle::Status::Ok();
}

const void *
query_provider_extension(void *,
                         hyle::compute::ProviderExtensionId id) noexcept {
  return id == kProbeExtensionId ? &kProbeExtension : nullptr;
}

const hyle::compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = {0x5445535400000001ULL, 0x50524f4245000001ULL},
                 .name = "test-probe",
                 .version = "1"},
    .create_context = &create_provider_context,
    .destroy_context = &destroy_provider_context,
    .propose = &propose_provider,
    .prepare = &prepare_provider,
    .drain_context = &drain_provider_context,
    .query_extension = &query_provider_extension,
};

const hyle::compute::detail::ProviderVTable kIncompleteProviderVTable{
    .identity = {.id = {0x5445535400000001ULL, 0x494e434f4d504c45ULL},
                 .name = "test-incomplete",
                 .version = "1"},
    .create_context = &create_provider_context,
    .destroy_context = &destroy_provider_context,
    .propose = &propose_provider,
    .prepare = &prepare_provider,
    // Deliberately missing drain_context: provider runtime ownership cannot
    // be accepted without a terminal teardown path.
};

hyle::compute::Provider make_provider(ProviderProbe &probe,
                                      std::size_t index) noexcept {
  return hyle::compute::detail::ProviderAccess::make(
      &kProviderVTable, ProviderPayload{&probe, index});
}

enum class PrepareDisposition : std::uint8_t {
  kPrepared = 0,
  kUnsupported,
  kResourceBudget,
  kError,
  kInvalidClassification,
  kUnownedPartial,
};

struct AsyncOperationProbe final {
  bool complete = false;
  bool cancelled = false;
  std::size_t submit_count = 0;
  std::size_t sequence_begin_count = 0;
  std::size_t sequence_finish_count = 0;
  std::size_t reserve_count = 0;
  std::size_t wait_enqueue_count = 0;
  std::size_t completion_destroy_count = 0;
  std::size_t invocation_destroy_count = 0;
  std::size_t prepare_count = 0;
  std::size_t prepared_destroy_count = 0;
  hyle::compute::detail::CompletionLifetime abandoned{};
  hyle::Status abandoned_hint{};
  PrepareDisposition prepare_disposition = PrepareDisposition::kPrepared;
  hyle::Status prepare_error =
      hyle::Status::DataLoss("injected provider failure");
  bool return_partial_artifact = false;
  bool fail_invocation = false;
  std::size_t proposal_count = 0;
  std::size_t shape_dimension_count = 0;
  bool shape_shares_input_output_axis = false;
  hyle::compute::detail::ProviderShapeRange observed_shape_range{};
  hyle::compute::detail::ProviderShapeCoverage shape_coverage =
      hyle::compute::detail::ProviderShapeCoverage::kExactPoint;
  hyle::compute::ProviderImplementationId implementation{};
  hyle::compute::RoutePolicyId policy{};
  hyle::compute::ProviderReplayClass proposed_replay =
      hyle::compute::ProviderReplayClass::kUnknown;
  hyle::compute::ProviderReplayClass prepared_replay =
      hyle::compute::ProviderReplayClass::kUnknown;
};

hyle::Status create_async_context(const hyle::compute::Provider &provider,
                                  hyle::compute::Device, void *,
                                  hyle::mem::AllocatorRef,
                                  void **context) noexcept {
  *context =
      hyle::compute::detail::ProviderAccess::payload<AsyncOperationProbe *>(
          provider);
  return hyle::Status::Ok();
}

void destroy_async_context(void *, hyle::mem::AllocatorRef) noexcept {}

void destroy_async_prepared(void *value, hyle::mem::AllocatorRef) noexcept {
  if (value != nullptr)
    ++static_cast<AsyncOperationProbe *>(value)->prepared_destroy_count;
}

hyle::compute::AliasSupport async_alias_support(void *, std::size_t,
                                                std::size_t) noexcept {
  return {};
}

hyle::Status create_async_invocation(void *prepared, hyle::mem::AllocatorRef,
                                     hyle::mem::BufferView,
                                     void **invocation) noexcept {
  *invocation = prepared;
  return static_cast<AsyncOperationProbe *>(prepared)->fail_invocation
             ? hyle::Status::DataLoss("injected partial invocation")
             : hyle::Status::Ok();
}

void destroy_async_invocation(void *invocation,
                              hyle::mem::AllocatorRef) noexcept {
  ++static_cast<AsyncOperationProbe *>(invocation)->invocation_destroy_count;
}

void destroy_async_operation_completion(
    void *state, hyle::compute::detail::CompletionLifetime lifetime,
    hyle::Status terminal_hint) noexcept {
  auto &probe = *static_cast<AsyncOperationProbe *>(state);
  ++probe.completion_destroy_count;
  if (!lifetime.valid())
    return;
  if (!probe.complete) {
    probe.abandoned = lifetime;
    probe.abandoned_hint = std::move(terminal_hint);
    return;
  }
  hyle::Status terminal =
      probe.cancelled ? hyle::Status::Cancelled("async operation cancelled")
                      : hyle::Status::Ok();
  if (terminal.ok() && !terminal_hint.ok())
    terminal = std::move(terminal_hint);
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

hyle::Result<hyle::compute::CompletionPoll>
poll_async_operation(void *state) noexcept {
  auto &probe = *static_cast<AsyncOperationProbe *>(state);
  if (!probe.complete)
    return hyle::compute::CompletionPoll::kPending;
  if (probe.cancelled) {
    return std::unexpected(
        hyle::Status::Cancelled("async operation cancelled"));
  }
  return hyle::compute::CompletionPoll::kComplete;
}

hyle::Status wait_async_operation(void *state) noexcept {
  auto &probe = *static_cast<AsyncOperationProbe *>(state);
  probe.complete = true;
  return probe.cancelled ? hyle::Status::Cancelled("async operation cancelled")
                         : hyle::Status::Ok();
}

hyle::Status cancel_async_operation(void *state) noexcept {
  auto &probe = *static_cast<AsyncOperationProbe *>(state);
  probe.cancelled = true;
  probe.complete = true;
  return hyle::Status::Ok();
}

const hyle::compute::detail::CompletionVTable kAsyncOperationCompletionVTable{
    .destroy = &destroy_async_operation_completion,
    .poll = &poll_async_operation,
    .wait = &wait_async_operation,
    .request_cancel = &cancel_async_operation,
};

hyle::compute::detail::EnqueueResult
submit_async_operation(void *, void *invocation, hyle::compute::Queue &,
                       std::span<const hyle::compute::ConstTensorView>,
                       std::span<const hyle::compute::TensorView>) noexcept {
  auto &probe = *static_cast<AsyncOperationProbe *>(invocation);
  ++probe.submit_count;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

const hyle::compute::detail::ProviderPreparedVTable kAsyncPreparedVTable{
    .destroy_prepared = &destroy_async_prepared,
    .alias_support = &async_alias_support,
    .create_invocation = &create_async_invocation,
    .destroy_invocation = &destroy_async_invocation,
    .shared_submit = &submit_async_operation,
    .delegated_submit = nullptr,
};

hyle::Status propose_async_operation(
    void *context,
    const hyle::compute::detail::ProviderOperationRequest &request,
    std::span<hyle::compute::detail::ResourceBindingRequirement>,
    std::span<hyle::compute::detail::ResourceBindingRequirement>,
    hyle::compute::detail::ProviderOperationProposal *proposal) noexcept {
  auto &probe = *static_cast<AsyncOperationProbe *>(context);
  ++probe.proposal_count;
  probe.shape_dimension_count = request.shape.dimensions.size();
  if (!request.shape.dimensions.empty())
    probe.observed_shape_range = request.shape.dimensions.front().bounds;
  probe.shape_shares_input_output_axis =
      !request.shape.inputs.empty() && !request.shape.outputs.empty() &&
      request.shape.inputs[0].values[0] == request.shape.outputs[0].values[0];
  *proposal = hyle::compute::detail::ProviderOperationProposal{
      .route = 1,
      .runtime =
          hyle::compute::ProviderRuntimeProperties{
              .mode = hyle::compute::ProviderRuntimeMode::kSharedQueue,
              .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
              .replay_class = probe.proposed_replay,
              .blocks_host = false,
              .uses_private_threads = false,
              .supports_concurrent_invocations = true,
              .capture_compatible = false,
          },
      .cost =
          hyle::compute::ProviderRouteCostFacts{
              .workspace = hyle::compute::WorkspaceRequirement{},
          },
      .shape_coverage = probe.shape_coverage,
      .implementation = probe.implementation,
      .policy = probe.policy,
  };
  return hyle::Status::Ok();
}

hyle::compute::ProviderPrepareOutcome prepare_async_operation(
    void *context, const hyle::compute::detail::ProviderPrepareRequest &request,
    hyle::mem::AllocatorRef,
    hyle::compute::detail::ProviderPrepared *prepared) noexcept {
  if (request.route != 1)
    return hyle::compute::ProviderPrepareOutcome::error(
        hyle::Status::InvalidArgument("test async route is invalid"));
  auto &probe = *static_cast<AsyncOperationProbe *>(context);
  ++probe.prepare_count;
  const auto artifact = hyle::compute::detail::ProviderPrepared{
      .vtable = &kAsyncPreparedVTable,
      .state = context,
      .runtime =
          hyle::compute::ProviderRuntimeProperties{
              .mode = hyle::compute::ProviderRuntimeMode::kSharedQueue,
              .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
              .replay_class = probe.prepared_replay,
              .blocks_host = false,
              .uses_private_threads = false,
              .supports_concurrent_invocations = true,
              .capture_compatible = false,
          },
  };
  if (probe.return_partial_artifact)
    *prepared = artifact;
  switch (probe.prepare_disposition) {
  case PrepareDisposition::kPrepared:
    *prepared = artifact;
    return hyle::compute::ProviderPrepareOutcome::prepared();
  case PrepareDisposition::kUnsupported:
    return hyle::compute::ProviderPrepareOutcome::unsupported(
        hyle::Status::Unimplemented("injected unsupported route"));
  case PrepareDisposition::kResourceBudget:
    return hyle::compute::ProviderPrepareOutcome::resource_declined(
        hyle::Status::ResourceExhausted("injected route budget rejection"));
  case PrepareDisposition::kError:
    return hyle::compute::ProviderPrepareOutcome::error(probe.prepare_error);
  case PrepareDisposition::kInvalidClassification:
    return hyle::compute::ProviderPrepareOutcome::unsupported(
        hyle::Status::InvalidArgument("mismatched decline classification"));
  case PrepareDisposition::kUnownedPartial:
    *prepared = hyle::compute::detail::ProviderPrepared{.state = context};
    return hyle::compute::ProviderPrepareOutcome::unsupported(
        hyle::Status::Unimplemented("unowned partial artifact"));
  }
  return hyle::compute::ProviderPrepareOutcome::error(
      hyle::Status::Internal("unknown test prepare disposition"));
}

const hyle::compute::detail::ProviderVTable kAsyncProviderVTable{
    .identity = {.id = {0x5445535400000002ULL, 0x4153594e43000001ULL},
                 .name = "test-async-operation",
                 .version = "1"},
    .create_context = &create_async_context,
    .destroy_context = &destroy_async_context,
    .propose = &propose_async_operation,
    .prepare = &prepare_async_operation,
    .drain_context = &drain_provider_context,
};

hyle::compute::Provider
make_async_provider(AsyncOperationProbe &probe) noexcept {
  return hyle::compute::detail::ProviderAccess::make(&kAsyncProviderVTable,
                                                     &probe);
}

struct AsyncBackendPayload final {
  AsyncOperationProbe *probe = nullptr;
};

constexpr hyle::compute::BackendId kAsyncBackendId{0x5445535400000003ULL,
                                                   0x4153594e43424531ULL};

hyle::Result<hyle::compute::Device>
async_device(const hyle::compute::Backend &backend, std::size_t index) noexcept;

hyle::Result<std::size_t>
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

hyle::Result<hyle::compute::DeviceProperties>
async_properties(const hyle::compute::Device &device) noexcept {
  if (hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(device)
          .probe == nullptr) {
    return std::unexpected(hyle::Status::InvalidArgument(
        "test async device has no completion probe"));
  }
  return hyle::compute::DeviceProperties{
      .backend = kAsyncBackendId,
      .device_class = hyle::compute::DeviceClass::kAccelerator,
      .backend_name = "test-async",
      .name = "test async sequence backend",
      .unified_memory = true,
      .asynchronous_queues = true,
  };
}

hyle::Status
validate_async_resource(const hyle::compute::Device &,
                        hyle::mem::MemoryResourceRef resource) noexcept {
  return resource.valid()
             ? hyle::Status::Ok()
             : hyle::Status::InvalidArgument("test resource is invalid");
}

hyle::Status create_async_backend_context(const hyle::compute::Device &device,
                                          hyle::mem::AllocatorRef,
                                          void **output) noexcept {
  if (output == nullptr)
    return hyle::Status::InvalidArgument("test context output is null");
  *output =
      hyle::compute::detail::DeviceAccess::payload<AsyncBackendPayload>(device)
          .probe;
  return *output != nullptr
             ? hyle::Status::Ok()
             : hyle::Status::InvalidArgument("test async device is invalid");
}

void destroy_async_backend_context(void *, hyle::mem::AllocatorRef) noexcept {}

hyle::Status create_async_queue(void *context, hyle::compute::QueueOptions,
                                hyle::mem::AllocatorRef,
                                void **output) noexcept {
  if (output == nullptr)
    return hyle::Status::InvalidArgument("test queue output is null");
  *output = context;
  return hyle::Status::Ok();
}

void destroy_async_queue(void *, hyle::mem::AllocatorRef) noexcept {}

hyle::Status begin_async_sequence(void *queue, void **sequence) noexcept {
  if (queue == nullptr || sequence == nullptr)
    return hyle::Status::InvalidArgument("test sequence is invalid");
  *sequence = queue;
  ++static_cast<AsyncOperationProbe *>(queue)->sequence_begin_count;
  return hyle::Status::Ok();
}

hyle::compute::Completion
finish_async_sequence(void *, void *sequence, bool accepted,
                      hyle::Status terminal) noexcept {
  ++static_cast<AsyncOperationProbe *>(sequence)->sequence_finish_count;
  if (!accepted)
    return hyle::compute::Completion::completed(std::move(terminal));
  auto completion = hyle::compute::detail::CompletionAccess::make(
      &kAsyncOperationCompletionVTable, sequence);
  hyle::compute::detail::CompletionAccess::append_terminal(completion,
                                                           std::move(terminal));
  return completion;
}

hyle::Status reserve_async_completions(void *queue, std::size_t) noexcept {
  ++static_cast<AsyncOperationProbe *>(queue)->reserve_count;
  return hyle::Status::Ok();
}

hyle::Result<hyle::compute::TransferProperties>
async_transfer_properties(void *, hyle::mem::BufferView,
                          hyle::mem::ConstBufferView) noexcept {
  return hyle::compute::TransferProperties{
      .mode = hyle::compute::TransferMode::kImmediate};
}

hyle::compute::detail::EnqueueResult
async_copy(void *, hyle::mem::BufferView destination,
           hyle::mem::ConstBufferView source) noexcept {
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
    void *) noexcept {
  if (completion_vtable != &kAsyncOperationCompletionVTable) {
    return hyle::compute::detail::EnqueueResult::rejected(
        hyle::Status::Unimplemented(
            "test backend cannot consume this completion type"));
  }
  ++static_cast<AsyncOperationProbe *>(queue)->wait_enqueue_count;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

hyle::compute::detail::CompletionEndpoint
async_completion_endpoint(void *source_queue) noexcept {
  return source_queue == nullptr
             ? hyle::compute::detail::CompletionEndpoint{}
             : hyle::compute::detail::CompletionEndpoint{
                   .authority = &kAsyncOperationCompletionVTable,
                   .identity = source_queue,
               };
}

bool can_async_wait_endpoint(
    void *destination_queue,
    hyle::compute::detail::CompletionEndpoint source) noexcept {
  return destination_queue != nullptr &&
         source.authority == &kAsyncOperationCompletionVTable &&
         source.identity != nullptr;
}

const void *
query_async_extension(void *, hyle::compute::BackendExtensionId id) noexcept {
  return id == kBackendProbeExtensionId ? &kBackendProbeExtension : nullptr;
}

hyle::Status drain_async(void *state) noexcept {
  if (state != nullptr)
    static_cast<AsyncOperationProbe *>(state)->complete = true;
  return hyle::Status::Ok();
}

const hyle::compute::detail::BackendVTable kAsyncBackendVTable{
    .id = kAsyncBackendId,
    .name = "test-async",
    .device_class = hyle::compute::DeviceClass::kAccelerator,
    .device_count = &async_device_count,
    .device = &async_device,
    .same_device = &same_async_device,
    .properties = &async_properties,
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

hyle::Result<hyle::compute::Device>
async_device(const hyle::compute::Backend &backend,
             std::size_t index) noexcept {
  if (index != 0)
    return std::unexpected(
        hyle::Status::OutOfRange("test async device index is out of range"));
  return hyle::compute::detail::DeviceAccess::make(
      &kAsyncBackendVTable,
      hyle::compute::detail::BackendAccess::payload<AsyncBackendPayload>(
          backend));
}

hyle::compute::Device async_device(AsyncOperationProbe &probe) noexcept {
  const auto backend = hyle::compute::detail::BackendAccess::make(
      &kAsyncBackendVTable, AsyncBackendPayload{&probe});
  return backend.device(0).value();
}

struct DelegatedProbe;

struct DelegatedContext final {
  DelegatedProbe *probe = nullptr;
  hyle::mem::AllocatorRef allocator{};
};

struct DelegatedInvocation final {
  DelegatedContext *context = nullptr;
  hyle::mem::AllocatorRef allocator{};
};

struct DelegatedCompletionState final {
  DelegatedProbe *probe = nullptr;
  hyle::mem::AllocatorRef allocator{};
  hyle::compute::detail::CompletionLifetime abandoned_lifetime{};
  hyle::Status abandoned_hint{};
  bool complete = false;
  bool cancelled = false;
  bool abandoned = false;
};

struct DelegatedProbe final {
  DelegatedCompletionState *pending = nullptr;
  std::size_t submit_count = 0;
  std::size_t drain_count = 0;
  std::size_t invocation_destroy_count = 0;
  bool fail = false;
  bool return_invalid_artifact = false;
  bool attach_owner = false;
  std::size_t owner_terminal_count = 0;
  std::size_t owner_release_count = 0;
  hyle::compute::detail::CompletionLifetime provider_lifetime{};
};

constexpr int kDelegatedEndpointAuthority = 0;

hyle::Status
delegated_terminal(const DelegatedCompletionState &state) noexcept {
  if (state.cancelled)
    return hyle::Status::Cancelled("delegated operation cancelled");
  if (state.probe->fail)
    return hyle::Status::DataLoss("delegated operation failed");
  return hyle::Status::Ok();
}

void release_delegated_completion(DelegatedCompletionState *state) noexcept {
  DelegatedProbe *probe = state->probe;
  if (probe->pending == state)
    probe->pending = nullptr;
  const hyle::mem::AllocatorRef allocator = state->allocator;
  state->~DelegatedCompletionState();
  allocator.deallocate(state, sizeof(DelegatedCompletionState),
                       alignof(DelegatedCompletionState));
}

void settle_abandoned_delegated(DelegatedCompletionState *state) noexcept {
  if (state == nullptr || !state->abandoned || !state->complete)
    return;
  hyle::Status terminal = delegated_terminal(*state);
  if (terminal.ok() && !state->abandoned_hint.ok())
    terminal = std::move(state->abandoned_hint);
  const hyle::compute::detail::CompletionLifetime lifetime =
      state->abandoned_lifetime;
  release_delegated_completion(state);
  lifetime.terminal(lifetime.state, terminal);
  lifetime.release(lifetime.state);
}

void destroy_delegated_completion(
    void *value, hyle::compute::detail::CompletionLifetime lifetime,
    hyle::Status terminal_hint) noexcept {
  auto *state = static_cast<DelegatedCompletionState *>(value);
  if (state == nullptr)
    return;
  if (!state->complete && lifetime.valid()) {
    state->abandoned = true;
    state->abandoned_lifetime = lifetime;
    state->abandoned_hint = std::move(terminal_hint);
    return;
  }
  hyle::Status terminal = delegated_terminal(*state);
  if (terminal.ok() && !terminal_hint.ok())
    terminal = std::move(terminal_hint);
  release_delegated_completion(state);
  if (lifetime.valid()) {
    lifetime.terminal(lifetime.state, terminal);
    lifetime.release(lifetime.state);
  }
}

hyle::Result<hyle::compute::CompletionPoll>
poll_delegated_completion(void *value) noexcept {
  auto &state = *static_cast<DelegatedCompletionState *>(value);
  if (!state.complete)
    return hyle::compute::CompletionPoll::kPending;
  hyle::Status terminal = delegated_terminal(state);
  if (!terminal.ok())
    return std::unexpected(std::move(terminal));
  return hyle::compute::CompletionPoll::kComplete;
}

hyle::Status wait_delegated_completion(void *value) noexcept {
  auto &state = *static_cast<DelegatedCompletionState *>(value);
  state.complete = true;
  return delegated_terminal(state);
}

hyle::Status cancel_delegated_completion(void *value) noexcept {
  auto &state = *static_cast<DelegatedCompletionState *>(value);
  state.cancelled = true;
  state.complete = true;
  return hyle::Status::Ok();
}

const hyle::compute::detail::CompletionVTable kDelegatedCompletionVTable{
    .destroy = &destroy_delegated_completion,
    .poll = &poll_delegated_completion,
    .wait = &wait_delegated_completion,
    .request_cancel = &cancel_delegated_completion,
};

hyle::Status create_delegated_context(const hyle::compute::Provider &provider,
                                      hyle::compute::Device, void *,
                                      hyle::mem::AllocatorRef allocator,
                                      void **output) noexcept {
  if (output == nullptr)
    return hyle::Status::InvalidArgument("delegated context output is null");
  *output = nullptr;
  void *storage =
      allocator.allocate(sizeof(DelegatedContext), alignof(DelegatedContext));
  if (storage == nullptr)
    return hyle::Status::ResourceExhausted(
        "delegated context allocation failed");
  DelegatedProbe *probe =
      hyle::compute::detail::ProviderAccess::payload<DelegatedProbe *>(
          provider);
  *output = ::new (storage) DelegatedContext{probe, allocator};
  return hyle::Status::Ok();
}

void destroy_delegated_context(void *value,
                               hyle::mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<DelegatedContext *>(value);
  if (context == nullptr)
    return;
  context->~DelegatedContext();
  allocator.deallocate(context, sizeof(DelegatedContext),
                       alignof(DelegatedContext));
}

void destroy_delegated_prepared(void *, hyle::mem::AllocatorRef) noexcept {}

hyle::compute::AliasSupport delegated_alias_support(void *, std::size_t,
                                                    std::size_t) noexcept {
  return {};
}

hyle::Status create_delegated_invocation(void *prepared,
                                         hyle::mem::AllocatorRef allocator,
                                         hyle::mem::BufferView,
                                         void **output) noexcept {
  if (output == nullptr)
    return hyle::Status::InvalidArgument("delegated invocation output is null");
  *output = nullptr;
  void *storage = allocator.allocate(sizeof(DelegatedInvocation),
                                     alignof(DelegatedInvocation));
  if (storage == nullptr)
    return hyle::Status::ResourceExhausted(
        "delegated invocation allocation failed");
  *output = ::new (storage)
      DelegatedInvocation{static_cast<DelegatedContext *>(prepared), allocator};
  return hyle::Status::Ok();
}

void destroy_delegated_invocation(void *value,
                                  hyle::mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<DelegatedInvocation *>(value);
  if (invocation == nullptr)
    return;
  ++invocation->context->probe->invocation_destroy_count;
  invocation->~DelegatedInvocation();
  allocator.deallocate(invocation, sizeof(DelegatedInvocation),
                       alignof(DelegatedInvocation));
}

hyle::Result<hyle::compute::Completion> submit_delegated(
    void *, void *invocation_value,
    std::span<const hyle::compute::detail::CompletionDependency> dependencies,
    std::span<const hyle::compute::ConstTensorView> inputs,
    std::span<const hyle::compute::TensorView> outputs) noexcept {
  auto *invocation = static_cast<DelegatedInvocation *>(invocation_value);
  if (invocation == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return std::unexpected(
        hyle::Status::Internal("delegated bindings are invalid"));
  DelegatedProbe &probe = *invocation->context->probe;
  for (const auto &dependency : dependencies) {
    if (dependency.terminal_hint != nullptr && !dependency.terminal_hint->ok())
      return std::unexpected(*dependency.terminal_hint);
    if (dependency.pending() &&
        dependency.vtable != &kDelegatedCompletionVTable)
      return std::unexpected(hyle::Status::Unimplemented(
          "test runtime cannot import this dependency"));
  }
  if (probe.pending != nullptr) {
    return std::unexpected(
        hyle::Status::FailedPrecondition("delegated runtime is occupied"));
  }
  void *storage = invocation->allocator.allocate(
      sizeof(DelegatedCompletionState), alignof(DelegatedCompletionState));
  if (storage == nullptr) {
    return std::unexpected(hyle::Status::ResourceExhausted(
        "delegated completion allocation failed"));
  }
  auto *state = ::new (storage) DelegatedCompletionState{
      .probe = &probe, .allocator = invocation->allocator};
  probe.pending = state;
  ++probe.submit_count;
  auto completion = hyle::compute::detail::CompletionAccess::make(
      &kDelegatedCompletionVTable, state);
  if (!completion.valid()) {
    release_delegated_completion(state);
    return std::unexpected(
        hyle::Status::Internal("delegated completion creation failed"));
  }
  if (probe.attach_owner) {
    probe.provider_lifetime = {
        &probe,
        [](void *value, const hyle::Status &) noexcept {
          ++static_cast<DelegatedProbe *>(value)->owner_terminal_count;
        },
        [](void *value) noexcept {
          ++static_cast<DelegatedProbe *>(value)->owner_release_count;
        }};
    const auto attached =
        hyle::compute::detail::CompletionAccess::attach_lifetime(
            completion, probe.provider_lifetime);
    HYLE_FATAL_ASSERT(attached.ok());
  }
  return completion;
}

hyle::compute::detail::CompletionEndpoint
delegated_endpoint(void *, void *invocation) noexcept {
  return {&kDelegatedEndpointAuthority, invocation};
}

bool can_import_delegated_endpoint(
    void *, void *, hyle::compute::detail::CompletionEndpoint source) noexcept {
  return source.valid() && source.authority == &kDelegatedEndpointAuthority;
}

const hyle::compute::detail::ProviderPreparedVTable kDelegatedPreparedVTable{
    .destroy_prepared = &destroy_delegated_prepared,
    .alias_support = &delegated_alias_support,
    .create_invocation = &create_delegated_invocation,
    .destroy_invocation = &destroy_delegated_invocation,
    .shared_submit = nullptr,
    .delegated_submit = &submit_delegated,
    .endpoint = &delegated_endpoint,
    .can_import_endpoint = &can_import_delegated_endpoint,
};

hyle::compute::detail::EnqueueResult
invalid_shared_submit(void *, void *, hyle::compute::Queue &,
                      std::span<const hyle::compute::ConstTensorView>,
                      std::span<const hyle::compute::TensorView>) noexcept {
  return hyle::compute::detail::EnqueueResult::rejected(
      hyle::Status::Internal("invalid mixed runtime artifact executed"));
}

const hyle::compute::detail::ProviderPreparedVTable
    kInvalidDelegatedPreparedVTable{
        .destroy_prepared = &destroy_delegated_prepared,
        .alias_support = &delegated_alias_support,
        .create_invocation = &create_delegated_invocation,
        .destroy_invocation = &destroy_delegated_invocation,
        .shared_submit = &invalid_shared_submit,
        .delegated_submit = &submit_delegated,
        .endpoint = &delegated_endpoint,
        .can_import_endpoint = &can_import_delegated_endpoint,
    };

constexpr hyle::compute::ProviderRuntimeProperties kDelegatedRuntime{
    .mode = hyle::compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kProvider,
    .blocks_host = false,
    .uses_private_threads = true,
    .supports_concurrent_invocations = false,
    .capture_compatible = false,
};

hyle::Status propose_delegated(
    void *, const hyle::compute::detail::ProviderOperationRequest &request,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        input_requirements,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        output_requirements,
    hyle::compute::detail::ProviderOperationProposal *proposal) noexcept {
  const auto operation = request.operation;
  const auto inputs = request.inputs;
  const auto outputs = request.outputs;
  if (proposal == nullptr ||
      operation.kind() != hyle::compute::OperationKind::kUnary ||
      inputs.size() != 1 || outputs.size() != 1 ||
      input_requirements.size() != 1 || output_requirements.size() != 1) {
    return hyle::Status::Unimplemented(
        "delegated test provider supports unary operations only");
  }
  input_requirements[0].required_properties =
      hyle::mem::MemoryProperty::kHostAccessible;
  output_requirements[0].required_properties =
      hyle::mem::MemoryProperty::kHostAccessible;
  *proposal = hyle::compute::detail::ProviderOperationProposal{
      .route = 1,
      .runtime = kDelegatedRuntime,
      .cost =
          hyle::compute::ProviderRouteCostFacts{
              .workspace = hyle::compute::WorkspaceRequirement{},
          },
  };
  return hyle::Status::Ok();
}

hyle::compute::ProviderPrepareOutcome
prepare_delegated(void *context,
                  const hyle::compute::detail::ProviderPrepareRequest &request,
                  hyle::mem::AllocatorRef,
                  hyle::compute::detail::ProviderPrepared *prepared) noexcept {
  if (context == nullptr || prepared == nullptr || request.route != 1 ||
      request.inputs.size() != 1 || request.outputs.size() != 1) {
    return hyle::compute::ProviderPrepareOutcome::error(
        hyle::Status::InvalidArgument(
            "delegated exact prepare request is invalid"));
  }
  *prepared = hyle::compute::detail::ProviderPrepared{
      .vtable = static_cast<DelegatedContext *>(context)
                        ->probe->return_invalid_artifact
                    ? &kInvalidDelegatedPreparedVTable
                    : &kDelegatedPreparedVTable,
      .state = context,
      .runtime = kDelegatedRuntime,
  };
  return hyle::compute::ProviderPrepareOutcome::prepared();
}

hyle::Status drain_delegated_context(void *value) noexcept {
  auto &context = *static_cast<DelegatedContext *>(value);
  ++context.probe->drain_count;
  DelegatedCompletionState *pending = context.probe->pending;
  if (pending != nullptr) {
    pending->complete = true;
    settle_abandoned_delegated(pending);
  }
  return hyle::Status::Ok();
}

const hyle::compute::detail::ProviderVTable kDelegatedProviderVTable{
    .identity = {.id = {0x5445535400000004ULL, 0x44454c4547415445ULL},
                 .name = "test-delegated-operation",
                 .version = "1"},
    .create_context = &create_delegated_context,
    .destroy_context = &destroy_delegated_context,
    .propose = &propose_delegated,
    .prepare = &prepare_delegated,
    .drain_context = &drain_delegated_context,
};

hyle::compute::Provider
make_delegated_provider(DelegatedProbe &probe) noexcept {
  return hyle::compute::detail::ProviderAccess::make(&kDelegatedProviderVTable,
                                                     &probe);
}

hyle::Status observe_backend_extension(void *, const void *interface_value,
                                       void *call_context) noexcept {
  const auto &interface =
      *static_cast<const ProbeExtensionInterface *>(interface_value);
  *static_cast<std::uint32_t *>(call_context) = interface.marker;
  return hyle::Status::Ok();
}

hyle::compute::detail::EnqueueResult
submit_backend_extension(void *backend_context, const void *interface_value,
                         void *backend_queue, void *call_context) noexcept {
  if (backend_context != backend_queue) {
    return hyle::compute::detail::EnqueueResult::rejected(
        hyle::Status::InvalidArgument("test backend queue owner differs"));
  }
  const auto &interface =
      *static_cast<const ProbeExtensionInterface *>(interface_value);
  *static_cast<std::uint32_t *>(call_context) = interface.marker;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

hyle::Status observe_probe_extension(void *, const void *interface_value,
                                     void *call_context) noexcept {
  const auto &interface =
      *static_cast<const ProbeExtensionInterface *>(interface_value);
  *static_cast<std::uint32_t *>(call_context) = interface.marker;
  return hyle::Status::Ok();
}

hyle::compute::detail::EnqueueResult
submit_probe_extension(void *, const void *interface_value, void *call_context,
                       hyle::compute::Queue &) noexcept {
  const auto &interface =
      *static_cast<const ProbeExtensionInterface *>(interface_value);
  *static_cast<std::uint32_t *>(call_context) = interface.marker;
  return hyle::compute::detail::EnqueueResult::accepted_command();
}

TEST(RuntimeProviders, ExtensionIsVersionedContextLocalAndRetainsItsOwner) {
  ProviderProbe probe;
  hyle::compute::detail::ProviderExtensionRef retained;
  {
    const auto configured = make_provider(probe, 3);
    auto created = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                  configured);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    EXPECT_EQ(created->provider_count(), 1u);
    ASSERT_TRUE(created->provider_identity(0).has_value());
    EXPECT_EQ(created->provider_identity(0)->id, kProviderVTable.identity.id);
    EXPECT_TRUE(created->provider_identity(1).error().IsOutOfRange());
    EXPECT_TRUE(
        hyle::compute::detail::ProviderExtensionAccess::query(*created, 0, {})
            .error()
            .IsInvalidArgument());
    EXPECT_TRUE(hyle::compute::detail::ProviderExtensionAccess::query(
                    *created, 0,
                    hyle::compute::ProviderExtensionId{
                        kProbeExtensionId.namespace_id,
                        kProbeExtensionId.extension_id, 2})
                    .error()
                    .IsUnimplemented());
    retained = hyle::compute::detail::ProviderExtensionAccess::query(
                   *created, 0, kProbeExtensionId)
                   .value();
    EXPECT_EQ(retained.id(), kProbeExtensionId);
    EXPECT_EQ(retained.provider().id, kProviderVTable.identity.id);

    std::uint32_t observed = 0;
    EXPECT_TRUE(hyle::compute::detail::ProviderExtensionAccess::call(
                    retained, &observe_probe_extension, &observed)
                    .ok());
    EXPECT_EQ(observed, kProbeExtension.marker);
  }
  EXPECT_EQ(probe.destroy_count, 0u);
  retained = {};
  EXPECT_EQ(probe.destroy_count, 1u);
}

TEST(RuntimeProviders, ExtensionSubmissionRequiresItsExactOpenContext) {
  ProviderProbe first_probe;
  ProviderProbe second_probe;
  const auto first_provider = make_provider(first_probe, 0);
  const auto second_provider = make_provider(second_probe, 1);
  auto first = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                              first_provider)
                   .value();
  auto second = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                               second_provider)
                    .value();
  auto extension = hyle::compute::detail::ProviderExtensionAccess::query(
                       first, 0, kProbeExtensionId)
                       .value();
  auto first_queue = first.create_queue().value();
  auto second_queue = second.create_queue().value();
  std::uint32_t observed = 0;

  auto wrong = hyle::compute::detail::ProviderExtensionAccess::submit(
      extension, second_queue, {}, {}, &submit_probe_extension, &observed);
  ASSERT_FALSE(wrong.has_value());
  EXPECT_TRUE(wrong.error().IsInvalidArgument());

  auto accepted = hyle::compute::detail::ProviderExtensionAccess::submit(
      extension, first_queue, {}, {}, &submit_probe_extension, &observed);
  ASSERT_TRUE(accepted.has_value()) << accepted.error().message();
  EXPECT_TRUE(accepted->wait().ok());
  EXPECT_EQ(observed, kProbeExtension.marker);

  ASSERT_TRUE(first.close().ok());
  EXPECT_TRUE(hyle::compute::detail::ProviderExtensionAccess::call(
                  extension, &observe_probe_extension, &observed)
                  .IsFailedPrecondition());
  auto closed = hyle::compute::detail::ProviderExtensionAccess::submit(
      extension, first_queue, {}, {}, &submit_probe_extension, &observed);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
}

TEST(RuntimeProviders, BackendExtensionIsVersionedAndContextLocal) {
  AsyncOperationProbe first_probe;
  AsyncOperationProbe second_probe;
  auto first =
      hyle::compute::Context::create(async_device(first_probe)).value();
  auto second =
      hyle::compute::Context::create(async_device(second_probe)).value();
  auto first_queue = first.create_queue().value();
  auto second_queue = second.create_queue().value();

  EXPECT_TRUE(hyle::compute::detail::BackendExtensionAccess::query(first, {})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(hyle::compute::detail::BackendExtensionAccess::query(
                  first,
                  hyle::compute::BackendExtensionId{
                      kBackendProbeExtensionId.namespace_id,
                      kBackendProbeExtensionId.extension_id, 2})
                  .error()
                  .IsUnimplemented());
  auto extension = hyle::compute::detail::BackendExtensionAccess::query(
                       first_queue, kBackendProbeExtensionId)
                       .value();
  EXPECT_EQ(extension.id(), kBackendProbeExtensionId);
  EXPECT_EQ(extension.backend(), kAsyncBackendId);

  std::uint32_t observed = 0;
  EXPECT_TRUE(hyle::compute::detail::BackendExtensionAccess::call(
                  extension, &observe_backend_extension, &observed)
                  .ok());
  EXPECT_EQ(observed, kBackendProbeExtension.marker);

  auto wrong = hyle::compute::detail::BackendExtensionAccess::submit(
      extension, second_queue, &submit_backend_extension, &observed);
  ASSERT_FALSE(wrong.has_value());
  EXPECT_TRUE(wrong.error().IsInvalidArgument());

  observed = 0;
  auto accepted = hyle::compute::detail::BackendExtensionAccess::submit(
      extension, first_queue, &submit_backend_extension, &observed);
  ASSERT_TRUE(accepted.has_value()) << accepted.error().message();
  EXPECT_TRUE(accepted->wait().ok());
  EXPECT_EQ(observed, kBackendProbeExtension.marker);

  EXPECT_TRUE(first.close().ok());
  EXPECT_TRUE(hyle::compute::detail::BackendExtensionAccess::call(
                  extension, &observe_backend_extension, &observed)
                  .IsFailedPrecondition());
}

TEST(RuntimeProviders, ContextCopiesExplicitProviderBindings) {
  ProviderProbe probe;
  std::array<hyle::compute::ProviderBinding, 6> providers{};
  for (std::size_t index = 0; index < providers.size(); ++index) {
    providers[index] = {make_provider(probe, index),
                        static_cast<std::uint32_t>(index)};
  }

  {
    auto created = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                  {.providers = providers});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    providers.fill({});

    const std::array<std::int64_t, 1> shape{8};
    auto description = hyle::compute::TensorDesc::dense(
        hyle::compute::DataType::kFloat32, shape);
    ASSERT_TRUE(description.has_value());
    auto prepared = hyle::compute::prepare(*created, hyle::compute::RmsNorm{},
                                           *description);
    ASSERT_FALSE(prepared.has_value());
    EXPECT_TRUE(prepared.error().IsUnimplemented());
    ASSERT_EQ(probe.prepare_count, providers.size());
    for (std::size_t index = 0; index < providers.size(); ++index)
      EXPECT_EQ(probe.prepared[index], index);
  }

  ASSERT_EQ(probe.destroy_count, providers.size());
  for (std::size_t index = 0; index < providers.size(); ++index)
    EXPECT_EQ(probe.destroyed[index], providers.size() - index - 1);
}

TEST(RuntimeProviders, InstanceIdsDistinguishSlotsAndSurviveHandleMoves) {
  using hyle::compute::Context;
  using hyle::compute::ProviderInstanceId;
  ProviderProbe probe;
  const std::array providers{
      hyle::compute::ProviderBinding{make_provider(probe, 0), 100},
      hyle::compute::ProviderBinding{make_provider(probe, 1), 200}};
  ProviderInstanceId retained;
  {
    auto context =
        Context::create(hyle::compute::cpu::device(), {.providers = providers})
            .value();
    const auto first = context.provider_instance_id(0).value();
    const auto second = context.provider_instance_id(1).value();
    ASSERT_TRUE(first.valid());
    ASSERT_TRUE(second.valid());
    EXPECT_NE(first, second);
    EXPECT_EQ(context.provider_identity(0).value(),
              context.provider_identity(1).value());
    EXPECT_TRUE(context.provider_instance_id(2).error().IsOutOfRange());
    Context moved = std::move(context);
    EXPECT_TRUE(context.provider_instance_id(0).error().IsFailedPrecondition());
    EXPECT_EQ(moved.provider_instance_id(0).value(), first);
    EXPECT_EQ(moved.provider_instance_id(1).value(), second);
    ASSERT_TRUE(moved.close().ok());
    EXPECT_EQ(moved.provider_instance_id(0).value(), first);
    EXPECT_FALSE(moved.is_open());
    retained = first;
  }
  EXPECT_EQ(probe.destroy_count, 2u);
  EXPECT_TRUE(retained.valid()); // diagnostic identity, not a lifetime handle
  auto recreated =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  EXPECT_NE(recreated.provider_instance_id(0).value(), retained);
  EXPECT_FALSE(ProviderInstanceId{}.valid());
  EXPECT_EQ(ProviderInstanceId{}, ProviderInstanceId{});
  EXPECT_FALSE(
      hyle::compute::PreparedOperation{}.provider_instance_id().valid());
  EXPECT_FALSE(
      hyle::compute::OperationInvocation{}.provider_instance_id().valid());
}

TEST(RuntimeProviders, ConcurrentContextCreationDoesNotReuseInstanceIds) {
  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kContexts = 8;
  std::array<hyle::compute::ProviderInstanceId, kThreads * kContexts>
      identities;
  std::array<std::thread, kThreads> workers;
  for (std::size_t worker = 0; worker < kThreads; ++worker) {
    workers[worker] = std::thread([&, worker] {
      ProviderProbe probe;
      for (std::size_t index = 0; index < kContexts; ++index) {
        auto context = hyle::compute::Context::create(
            hyle::compute::cpu::device(), make_provider(probe, index));
        ASSERT_TRUE(context.has_value()) << context.error().message();
        identities[worker * kContexts + index] =
            context->provider_instance_id(0).value();
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (std::size_t first = 0; first < identities.size(); ++first) {
    ASSERT_TRUE(identities[first].valid());
    for (std::size_t second = first + 1; second < identities.size(); ++second)
      EXPECT_NE(identities[first], identities[second]);
  }
}

TEST(RuntimeProviders, ProposalCannotBeSplicedIntoAnotherSameFamilySlot) {
  using hyle::compute::detail::OperationPreparationAccess;
  AsyncOperationProbe first;
  AsyncOperationProbe second;
  const std::array providers{
      hyle::compute::ProviderBinding{make_async_provider(first), 100},
      hyle::compute::ProviderBinding{make_async_provider(second), 200}};
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                {.providers = providers})
                     .value();
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32,
                                       std::array<std::int64_t, 1>{8})
          .value();
  const hyle::compute::Operation operation(
      hyle::compute::Unary{hyle::compute::UnaryKind::kNeg});
  const std::array descriptions{description};
  std::array<hyle::compute::detail::ResourceBindingRequirement, 1> inputs;
  std::array<hyle::compute::detail::ResourceBindingRequirement, 1> outputs;
  const auto proposal =
      OperationPreparationAccess::propose(context, 0, operation, descriptions,
                                          descriptions, inputs, outputs)
          .value();
  const auto expected = context.provider_instance_id(0).value();
  EXPECT_EQ(proposal.instance, expected);
  auto spliced = proposal;
  spliced.provider_index = 1;
  ASSERT_EQ(proposal.provider, context.provider_identity(1).value());
  const std::array resources{hyle::mem::system_resource()};
  const hyle::compute::OperationResourceBindings bindings{
      .inputs = resources,
      .outputs = resources,
      .workspace_resource = hyle::mem::system_resource()};
  auto rejected = OperationPreparationAccess::prepare_candidate(
      context, spliced, operation, descriptions, descriptions, inputs, outputs,
      bindings);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_EQ(first.prepare_count, 0u);
  EXPECT_EQ(second.prepare_count, 0u);
  auto original = OperationPreparationAccess::prepare_candidate(
                      context, proposal, operation, descriptions, descriptions,
                      inputs, outputs, bindings)
                      .value();
  ASSERT_TRUE(original.succeeded());
  EXPECT_EQ(original.prepared.provider_instance_id(), expected);
  auto copied = original.prepared;
  EXPECT_EQ(copied.provider_instance_id(), expected);
  auto invocation = copied.create_invocation().value();
  EXPECT_EQ(invocation.provider_instance_id(), expected);
  auto moved = std::move(copied);
  EXPECT_FALSE(copied.provider_instance_id().valid());
  EXPECT_EQ(moved.provider_instance_id(), expected);
}

TEST(RuntimeProviders, ProviderExtensionRetainsItsExactInstanceIdentity) {
  using hyle::compute::detail::ProviderExtensionAccess;
  using hyle::compute::detail::ProviderExtensionRef;
  ProviderProbe probe;
  const std::array providers{
      hyle::compute::ProviderBinding{make_provider(probe, 0), 100},
      hyle::compute::ProviderBinding{make_provider(probe, 1), 200}};
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                {.providers = providers})
                     .value();
  const auto expected = context.provider_instance_id(1).value();
  auto extension =
      ProviderExtensionAccess::query(context, 1, kProbeExtensionId).value();
  EXPECT_EQ(extension.provider_instance_id(), expected);
  auto copied = extension;
  auto moved = std::move(copied);
  EXPECT_FALSE(copied.provider_instance_id().valid());
  EXPECT_EQ(moved.provider_instance_id(), expected);
  ProviderExtensionRef assigned;
  assigned = moved;
  EXPECT_EQ(assigned.provider_instance_id(), expected);
  assigned = std::move(moved);
  EXPECT_FALSE(moved.provider_instance_id().valid());
  EXPECT_EQ(assigned.provider_instance_id(), expected);
  ASSERT_TRUE(context.close().ok());
  EXPECT_EQ(extension.provider_instance_id(), expected);
  EXPECT_TRUE(ProviderExtensionAccess::query(context, 1, kProbeExtensionId)
                  .error()
                  .IsFailedPrecondition());
}

TEST(RuntimeProviders,
     PreparationAttemptClassifiesDeclinesAndOwnsPartialArtifacts) {
  AsyncOperationProbe probe;
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                make_async_provider(probe))
                     .value();
  const std::array<std::int64_t, 1> shape{8};
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32, shape)
          .value();
  const hyle::compute::Operation operation(
      hyle::compute::Unary{hyle::compute::UnaryKind::kNeg});
  const std::array inputs{description};
  const std::array outputs{description};
  std::array<hyle::compute::detail::ResourceBindingRequirement, 1>
      input_requirements{};
  std::array<hyle::compute::detail::ResourceBindingRequirement, 1>
      output_requirements{};
  const auto proposal =
      hyle::compute::detail::OperationPreparationAccess::propose(
          context, 0, operation, inputs, outputs, input_requirements,
          output_requirements)
          .value();
  const std::array input_resources{hyle::mem::system_resource()};
  const std::array output_resources{hyle::mem::system_resource()};
  const hyle::compute::OperationResourceBindings resources{
      .inputs = input_resources,
      .outputs = output_resources,
      .workspace_resource = hyle::mem::system_resource(),
  };
  const auto attempt = [&]() {
    return hyle::compute::detail::OperationPreparationAccess::prepare_candidate(
        context, proposal, operation, inputs, outputs, input_requirements,
        output_requirements, resources);
  };

  probe.prepare_disposition = PrepareDisposition::kUnsupported;
  auto unsupported = attempt();
  ASSERT_TRUE(unsupported.has_value()) << unsupported.error().message();
  EXPECT_FALSE(unsupported->succeeded());
  EXPECT_EQ(unsupported->decline,
            hyle::compute::detail::ProviderPrepareDecline::kUnsupported);
  EXPECT_TRUE(unsupported->reason.IsUnimplemented());

  probe.prepare_disposition = PrepareDisposition::kResourceBudget;
  probe.return_partial_artifact = true;
  auto resource = attempt();
  ASSERT_TRUE(resource.has_value()) << resource.error().message();
  EXPECT_FALSE(resource->succeeded());
  EXPECT_EQ(resource->decline,
            hyle::compute::detail::ProviderPrepareDecline::kResourceBudget);
  EXPECT_TRUE(resource->reason.IsResourceExhausted());
  EXPECT_EQ(probe.prepared_destroy_count, 1u);

  probe.prepare_disposition = PrepareDisposition::kError;
  probe.return_partial_artifact = false;
  auto failed = attempt();
  ASSERT_FALSE(failed.has_value());
  EXPECT_TRUE(failed.error().IsDataLoss());

  probe.prepare_disposition = PrepareDisposition::kInvalidClassification;
  auto invalid_classification = attempt();
  ASSERT_FALSE(invalid_classification.has_value());
  EXPECT_TRUE(invalid_classification.error().IsInternal());

  probe.prepare_disposition = PrepareDisposition::kUnownedPartial;
  auto unowned = attempt();
  ASSERT_FALSE(unowned.has_value());
  EXPECT_TRUE(unowned.error().IsInternal());
}

TEST(RuntimeProviders, ReplayCostIsPreservedAndCannotChangeDuringPreparation) {
  using namespace hyle::compute;
  AsyncOperationProbe probe;
  auto context =
      Context::create(cpu::device(), make_async_provider(probe)).value();
  const std::array<std::int64_t, 1> shape{8};
  const auto description = TensorDesc::dense(DataType::kFloat32, shape).value();
  const auto make_prepared = [&]() {
    return prepare(context, Unary{UnaryKind::kNeg}, description);
  };
  constexpr std::array classes{ProviderReplayClass::kUnknown,
                               ProviderReplayClass::kPrepared,
                               ProviderReplayClass::kManaged};
  for (auto replay_class : classes) {
    probe.proposed_replay = probe.prepared_replay = replay_class;
    auto prepared = make_prepared();
    ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
    EXPECT_EQ(prepared->runtime_properties().replay_class, replay_class);
  }
  EXPECT_EQ(probe.prepare_count, 3u);
  EXPECT_EQ(probe.prepared_destroy_count, 3u);

  probe.proposed_replay = ProviderReplayClass::kPrepared;
  probe.prepared_replay = ProviderReplayClass::kManaged;
  auto changed = make_prepared();
  ASSERT_FALSE(changed.has_value());
  EXPECT_TRUE(changed.error().IsInternal());
  EXPECT_EQ(probe.prepared_destroy_count, 4u);

  probe.prepared_replay = static_cast<ProviderReplayClass>(255);
  auto invalid_artifact = make_prepared();
  ASSERT_FALSE(invalid_artifact.has_value());
  EXPECT_TRUE(invalid_artifact.error().IsInternal());
  EXPECT_EQ(probe.prepared_destroy_count, 5u);
  probe.proposed_replay = static_cast<ProviderReplayClass>(255);
  auto invalid_proposal = make_prepared();
  ASSERT_FALSE(invalid_proposal.has_value());
  EXPECT_TRUE(invalid_proposal.error().IsInternal());
  EXPECT_EQ(probe.prepare_count, 5u);
  EXPECT_EQ(probe.prepared_destroy_count, 5u);
  EXPECT_EQ(probe.submit_count, 0u);
}

TEST(RuntimeProviders, DirectPreparationRetriesDeclaredSafeDeclines) {
  const std::array<std::int64_t, 1> shape{8};
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32, shape)
          .value();

  AsyncOperationProbe unsupported_probe;
  AsyncOperationProbe fallback_probe;
  unsupported_probe.prepare_disposition = PrepareDisposition::kUnsupported;
  const std::array fallback_providers{
      hyle::compute::ProviderBinding{make_async_provider(unsupported_probe),
                                     100},
      hyle::compute::ProviderBinding{make_async_provider(fallback_probe), 200}};
  auto fallback_context =
      hyle::compute::Context::create(hyle::compute::cpu::device(),
                                     {.providers = fallback_providers})
          .value();
  auto prepared = hyle::compute::prepare(
      fallback_context, hyle::compute::Unary{hyle::compute::UnaryKind::kNeg},
      description);
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(unsupported_probe.prepare_count, 1u);
  EXPECT_EQ(fallback_probe.prepare_count, 1u);

  AsyncOperationProbe resource_probe;
  AsyncOperationProbe forbidden_fallback_probe;
  resource_probe.prepare_disposition = PrepareDisposition::kResourceBudget;
  resource_probe.return_partial_artifact = true;
  const std::array resource_providers{
      hyle::compute::ProviderBinding{make_async_provider(resource_probe), 100},
      hyle::compute::ProviderBinding{
          make_async_provider(forbidden_fallback_probe), 200}};
  auto resource_context =
      hyle::compute::Context::create(hyle::compute::cpu::device(),
                                     {.providers = resource_providers})
          .value();
  auto resource_fallback = hyle::compute::prepare(
      resource_context, hyle::compute::Unary{hyle::compute::UnaryKind::kNeg},
      description);
  ASSERT_TRUE(resource_fallback.has_value())
      << resource_fallback.error().message();
  EXPECT_EQ(resource_probe.prepare_count, 1u);
  EXPECT_EQ(resource_probe.prepared_destroy_count, 1u);
  EXPECT_EQ(forbidden_fallback_probe.prepare_count, 1u);
}

TEST(RuntimeProviders, DirectErrorCodesDoNotBecomeSafeDeclines) {
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32,
                                       std::array<std::int64_t, 1>{8})
          .value();
  for (const hyle::Status &error :
       {hyle::Status::Unimplemented("fatal exact preparation failure"),
        hyle::Status::ResourceExhausted("real exact preparation OOM")}) {
    AsyncOperationProbe first;
    AsyncOperationProbe fallback;
    first.prepare_disposition = PrepareDisposition::kError;
    first.prepare_error = error;
    first.return_partial_artifact = true;
    const std::array providers{
        hyle::compute::ProviderBinding{make_async_provider(first), 100},
        hyle::compute::ProviderBinding{make_async_provider(fallback), 200}};
    auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                  {.providers = providers})
                       .value();
    auto prepared = hyle::compute::prepare(
        context, hyle::compute::Unary{hyle::compute::UnaryKind::kNeg},
        description);
    ASSERT_FALSE(prepared.has_value());
    EXPECT_EQ(prepared.error(), error);
    EXPECT_EQ(first.prepare_count, 1u);
    EXPECT_EQ(first.prepared_destroy_count, 1u);
    EXPECT_EQ(fallback.proposal_count, 1u);
    EXPECT_EQ(fallback.prepare_count, 0u);
  }
}

TEST(RuntimeProviders, DirectBindingUsesPreferenceAndRejectsTies) {
  using namespace hyle::compute;
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{8})
          .value();

  AsyncOperationProbe lower_priority;
  AsyncOperationProbe preferred;
  preferred.implementation = ProviderImplementationId{17};
  preferred.policy = {0x54455354524f5554ULL, 3, 1};
  const std::array ranked{
      ProviderBinding{make_async_provider(lower_priority), 200},
      ProviderBinding{make_async_provider(preferred), 100}};
  auto context = Context::create(cpu::device(), {.providers = ranked}).value();
  auto prepared = prepare(context, Unary{UnaryKind::kNeg}, description);
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(lower_priority.proposal_count, 1u);
  EXPECT_EQ(preferred.proposal_count, 1u);
  EXPECT_EQ(lower_priority.prepare_count, 0u);
  EXPECT_EQ(preferred.prepare_count, 1u);
  EXPECT_EQ(prepared->implementation(), preferred.implementation);
  EXPECT_EQ(prepared->route_policy(), preferred.policy);
  EXPECT_EQ(prepared->route_preference(), 100u);

  AsyncOperationProbe first;
  AsyncOperationProbe second;
  const std::array ambiguous{ProviderBinding{make_async_provider(first), 100},
                             ProviderBinding{make_async_provider(second), 100}};
  auto ambiguous_context =
      Context::create(cpu::device(), {.providers = ambiguous}).value();
  auto rejected =
      prepare(ambiguous_context, Unary{UnaryKind::kNeg}, description);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_EQ(first.proposal_count, 1u);
  EXPECT_EQ(second.proposal_count, 1u);
  EXPECT_EQ(first.prepare_count, 0u);
  EXPECT_EQ(second.prepare_count, 0u);
}

TEST(RuntimeProviders, ContextCreationRollsBackInReverseOrder) {
  ProviderProbe probe;
  probe.fail_index = 2;
  const std::array providers{
      hyle::compute::ProviderBinding{make_provider(probe, 0), 100},
      hyle::compute::ProviderBinding{make_provider(probe, 1), 200},
      hyle::compute::ProviderBinding{make_provider(probe, 2), 300},
      hyle::compute::ProviderBinding{make_provider(probe, 3), 400}};

  auto created = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                {.providers = providers});
  ASSERT_FALSE(created.has_value());
  EXPECT_TRUE(created.error().IsInvalidArgument());
  ASSERT_EQ(probe.create_count, 3u);
  EXPECT_EQ(probe.created[0], 0u);
  EXPECT_EQ(probe.created[1], 1u);
  EXPECT_EQ(probe.created[2], 2u);
  ASSERT_EQ(probe.destroy_count, 2u);
  EXPECT_EQ(probe.destroyed[0], 1u);
  EXPECT_EQ(probe.destroyed[1], 0u);
}

TEST(RuntimeProviders, RejectsProviderWithoutRequiredDrainContract) {
  ProviderProbe probe;
  const auto incomplete = hyle::compute::detail::ProviderAccess::make(
      &kIncompleteProviderVTable, ProviderPayload{&probe, 0});
  auto created =
      hyle::compute::Context::create(hyle::compute::cpu::device(), incomplete);
  ASSERT_FALSE(created.has_value());
  EXPECT_TRUE(created.error().IsInvalidArgument());
  EXPECT_EQ(probe.create_count, 0u);
}

TEST(RuntimeProviders,
     RejectsPrivateContractMismatchAndUnwindsEarlierContexts) {
  ProviderProbe probe;
  auto incompatible_table = kProviderVTable;
  incompatible_table.contract_version =
      hyle::compute::detail::kProviderContractVersion + 1;
  const auto incompatible = hyle::compute::detail::ProviderAccess::make(
      &incompatible_table, ProviderPayload{&probe, 1});
  const std::array providers{
      hyle::compute::ProviderBinding{make_provider(probe, 0), 100},
      hyle::compute::ProviderBinding{incompatible, 200}};
  auto context = hyle::compute::Context::create(
      hyle::compute::cpu::device(),
      hyle::compute::ContextOptions{.providers = providers});
  ASSERT_FALSE(context.has_value());
  EXPECT_TRUE(context.error().IsInvalidArgument());
  ASSERT_EQ(probe.create_count, 1u);
  ASSERT_EQ(probe.destroy_count, 1u);
  EXPECT_EQ(probe.created[0], 0u);
  EXPECT_EQ(probe.destroyed[0], 0u);
}

TEST(RuntimeProviders, ShapeRequestsNeedExplicitWholeDomainCoverage) {
  using namespace hyle::compute;
  using namespace hyle::compute::detail;
  AsyncOperationProbe probe;
  auto context =
      Context::create(cpu::device(), make_async_provider(probe)).value();
  const std::array<std::int64_t, 1> upper_shape{8};
  const std::array descriptions{
      TensorDesc::dense(DataType::kFloat32, upper_shape).value()};
  std::array dimensions{ShapeDimension{.bounds = {1, 8, 2}}};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].values[0] = 0;
  const ProviderShapeDomain domain{dimensions, axes, axes};
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  const auto propose = [&]() {
    return OperationPreparationAccess::propose(
        context, 0, Operation(Unary{UnaryKind::kNeg}), descriptions,
        descriptions, inputs, outputs, domain);
  };
  auto exact_only = propose();
  ASSERT_FALSE(exact_only.has_value());
  EXPECT_TRUE(exact_only.error().IsUnimplemented());
  EXPECT_EQ(probe.proposal_count, 1u);
  EXPECT_EQ(probe.shape_dimension_count, 1u);
  EXPECT_EQ(probe.observed_shape_range, dimensions[0].bounds);
  EXPECT_TRUE(probe.shape_shares_input_output_axis);

  // This probe verifies the capability seam, not a real bounded runtime.
  probe.shape_coverage = ProviderShapeCoverage::kEntireDomain;
  auto entire = propose();
  ASSERT_TRUE(entire.has_value()) << entire.error().message();
  EXPECT_EQ(entire->shape_coverage, ProviderShapeCoverage::kEntireDomain);
  probe.shape_coverage = static_cast<ProviderShapeCoverage>(255);
  auto invalid = propose();
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInternal());
  EXPECT_EQ(probe.proposal_count, 3u);

  dimensions[0].bounds.maximum = 7;
  auto malformed = propose();
  ASSERT_FALSE(malformed.has_value());
  EXPECT_TRUE(malformed.error().IsInvalidArgument());
  EXPECT_EQ(probe.proposal_count, 3u);

  dimensions[0].bounds = ProviderShapeRange{7, 9, 8};
  probe.shape_coverage = ProviderShapeCoverage::kExactPoint;
  auto singleton = propose();
  ASSERT_TRUE(singleton.has_value()) << singleton.error().message();
  EXPECT_EQ(singleton->shape_coverage, ProviderShapeCoverage::kExactPoint);
  EXPECT_EQ(probe.prepare_count, 0u);
  EXPECT_EQ(probe.submit_count, 0u);
}

struct MalformedEffectSchema final {
  static constexpr hyle::compute::OperationId operation_id{0x48594c45U,
                                                           0x52455305U};
  static constexpr std::string_view operation_name = "bad-effect-schema";
  hyle::compute::OperationTraits operation_traits() const noexcept {
    return {.inputs = {1, 1},
            .outputs = 1,
            .effect = hyle::compute::EffectKind::kResourceAccess};
  }
  hyle::Status infer_output_descriptions(
      std::span<const hyle::compute::TensorDesc> inputs,
      std::span<hyle::compute::TensorDesc> outputs) const noexcept {
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};

TEST(RuntimeProviders, RejectsMalformedEffectsBeforeExactOrBoundedProposal) {
  using namespace hyle::compute;
  using namespace hyle::compute::detail;
  AsyncOperationProbe probe;
  probe.shape_coverage = ProviderShapeCoverage::kEntireDomain;
  auto context =
      Context::create(cpu::device(), make_async_provider(probe)).value();
  const std::array descriptions{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{8})
          .value()};
  const std::array dimensions{ShapeDimension{.bounds = {1, 8, 1}}};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].values[0] = 0;
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  const auto operation = Operation::custom(MalformedEffectSchema{});
  for (const ProviderShapeDomain domain :
       {ProviderShapeDomain{}, ProviderShapeDomain{dimensions, axes, axes}}) {
    auto result = OperationPreparationAccess::propose(
        context, 0, operation, descriptions, descriptions, inputs, outputs,
        domain);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsInvalidArgument());
  }
  EXPECT_EQ(probe.proposal_count, 0u);
  EXPECT_EQ(probe.prepare_count, 0u);
  EXPECT_EQ(probe.submit_count, 0u);
}

TEST(RuntimeProviders, GuardedEnvelopeIsNotUsedForConcreteSchemaInference) {
  using namespace hyle::compute;
  using namespace hyle::compute::detail;
  AsyncOperationProbe probe;
  probe.shape_coverage = ProviderShapeCoverage::kEntireDomain;
  auto context =
      Context::create(cpu::device(), make_async_provider(probe)).value();
  const std::array dimensions{ShapeDimension{.bounds = {1, 8, 1}},
                              ShapeDimension{.bounds = {1, 6, 1}}};
  const std::array guards{ShapeGuard{
      ShapeGuardKind::kEqual, ShapeOperand::value(0), ShapeOperand::value(1)}};
  std::array<ProviderTensorAxes, 1> input_axes{}, output_axes{};
  input_axes[0].values[0] = 0;
  output_axes[0].values[0] = 1;
  const ProviderShapeDomain domain{dimensions, input_axes, output_axes, guards};
  const std::array input_desc{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{8})
          .value()};
  const std::array output_desc{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{6})
          .value()};
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  // The equality guard admits (1,1)..(6,6), never the envelope tuple (8,6).
  // This is still only an SPI probe; no real bounded artifact is claimed.
  auto proposed = OperationPreparationAccess::propose(
      context, 0, Operation(Unary{UnaryKind::kNeg}), input_desc, output_desc,
      inputs, outputs, domain);
  ASSERT_TRUE(proposed.has_value()) << proposed.error().message();
  EXPECT_EQ(probe.proposal_count, 1u);
  auto exact = OperationPreparationAccess::propose(
      context, 0, Operation(Unary{UnaryKind::kNeg}), input_desc, output_desc,
      inputs, outputs);
  EXPECT_TRUE(exact.error().IsInvalidArgument());
  EXPECT_EQ(probe.proposal_count, 1u);
}

TEST(RuntimeProviders, OrderedQueueAdmitsOnlyOneRecordingSequenceAtATime) {
  ProviderProbe probe;
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                make_provider(probe, 0))
                     .value();
  auto queue = context.create_queue().value();
  EXPECT_TRUE(queue.reserve_completions(3).ok());
  {
    auto first = hyle::compute::detail::QueueSequence::begin(queue);
    ASSERT_TRUE(first.has_value()) << first.error().message();
    auto overlapping = hyle::compute::detail::QueueSequence::begin(queue);
    ASSERT_FALSE(overlapping.has_value());
    EXPECT_TRUE(overlapping.error().IsFailedPrecondition());
  }
  EXPECT_TRUE(hyle::compute::detail::QueueSequence::begin(queue).has_value());
}

TEST(RuntimeProviders, QueueSequenceEnqueuesOnlyCompatibleCompletionWaits) {
  AsyncOperationProbe probe;
  auto context = hyle::compute::Context::create(async_device(probe)).value();
  auto source_queue = context.create_queue().value();
  auto destination_queue = context.create_queue().value();

  EXPECT_EQ(hyle::compute::detail::QueueSequence::dependency_route(
                destination_queue, source_queue)
                .value(),
            hyle::compute::detail::QueueDependencyRoute::kDeviceWait);
  EXPECT_EQ(hyle::compute::detail::QueueSequence::dependency_route(source_queue,
                                                                   source_queue)
                .value(),
            hyle::compute::detail::QueueDependencyRoute::kNoWait);

  auto source_sequence =
      hyle::compute::detail::QueueSequence::begin(source_queue).value();
  source_sequence.record(
      hyle::compute::detail::EnqueueResult::accepted_command());
  auto source = source_sequence.finish().value();

  auto destination_sequence =
      hyle::compute::detail::QueueSequence::begin(destination_queue).value();
  auto wait = destination_sequence.enqueue_wait(source);
  EXPECT_TRUE(wait.accepted);
  EXPECT_TRUE(wait.status.ok()) << wait.status.message();
  destination_sequence.record(std::move(wait));
  auto destination = destination_sequence.finish().value();
  EXPECT_EQ(probe.wait_enqueue_count, 1u);

  auto cpu_context =
      hyle::compute::Context::create(hyle::compute::cpu::device()).value();
  auto cpu_queue = cpu_context.create_queue().value();
  EXPECT_EQ(hyle::compute::detail::QueueSequence::dependency_route(cpu_queue,
                                                                   source_queue)
                .value(),
            hyle::compute::detail::QueueDependencyRoute::kHostWait);
  EXPECT_EQ(hyle::compute::detail::QueueSequence::dependency_route(
                destination_queue, cpu_queue)
                .value(),
            hyle::compute::detail::QueueDependencyRoute::kNoWait);
  auto incompatible =
      hyle::compute::detail::QueueSequence::begin(cpu_queue).value();
  auto rejected = incompatible.enqueue_wait(source);
  EXPECT_FALSE(rejected.accepted);
  EXPECT_TRUE(rejected.status.IsUnimplemented());
  incompatible.record(std::move(rejected));
  EXPECT_TRUE(incompatible.finish().error().IsUnimplemented());

  auto failed = hyle::compute::Completion::completed(
      hyle::Status::DataLoss("source prefix failed"));
  auto immediate =
      hyle::compute::detail::QueueSequence::begin(cpu_queue).value();
  auto propagated = immediate.enqueue_wait(failed);
  EXPECT_FALSE(propagated.accepted);
  EXPECT_TRUE(propagated.status.IsDataLoss());
  immediate.record(std::move(propagated));
  EXPECT_TRUE(immediate.finish().error().IsDataLoss());

  EXPECT_TRUE(destination.wait().ok());
  EXPECT_TRUE(source.wait().ok());
}

TEST(RuntimeProviders,
     DirectAsyncCompletionRetainsAndSerializesItsOperationInvocation) {
  AsyncOperationProbe probe;
  auto context = hyle::compute::Context::create(async_device(probe),
                                                make_async_provider(probe))
                     .value();
  auto queue = context.create_queue().value();
  EXPECT_TRUE(queue.reserve_completions(3).ok());
  EXPECT_EQ(probe.reserve_count, 1u);
  const std::array<std::int64_t, 1> shape{4};
  auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32, shape)
          .value();
  auto prepared =
      hyle::compute::prepare(
          context, hyle::compute::Unary{hyle::compute::UnaryKind::kNeg},
          description)
          .value();
  auto input_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4 * sizeof(float))
          .value();
  auto output_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4 * sizeof(float))
          .value();
  auto input =
      hyle::compute::ConstTensorView::create(description, input_buffer.view())
          .value();
  auto output =
      hyle::compute::TensorView::create(description, output_buffer.view())
          .value();
  auto invocation = prepared.create_invocation().value();

  auto completion = invocation.submit(queue, input, output);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_EQ(probe.sequence_begin_count, 1u);
  EXPECT_EQ(probe.sequence_finish_count, 1u);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(
      invocation.submit(queue, input, output).error().IsFailedPrecondition());
  invocation = {};
  EXPECT_EQ(probe.invocation_destroy_count, 0u);
  *completion = {};
  EXPECT_TRUE(probe.abandoned.valid());
  EXPECT_EQ(probe.invocation_destroy_count, 0u);

  probe.complete = true;
  probe.abandoned.terminal(probe.abandoned.state, hyle::Status::Ok());
  probe.abandoned.release(probe.abandoned.state);
  probe.abandoned = {};
  EXPECT_EQ(probe.invocation_destroy_count, 1u);
}

struct WideMetadataSchema final {
  static constexpr hyle::compute::OperationId operation_id{0x4d455441ULL, 9};
  static constexpr std::string_view operation_name = "wide-metadata-fixture";
  hyle::compute::OperationTraits operation_traits() const noexcept {
    return {.inputs = {9, 9}, .outputs = 9};
  }
  hyle::Status infer_output_descriptions(
      std::span<const hyle::compute::TensorDesc> inputs,
      std::span<hyle::compute::TensorDesc> outputs) const noexcept {
    if (inputs.size() != 9 || outputs.size() != 9)
      return hyle::Status::InvalidArgument("wide schema arity");
    for (std::size_t index = 0; index < outputs.size(); ++index)
      outputs[index] = inputs[index];
    return hyle::Status::Ok();
  }
};

TEST(RuntimeProviders, InferenceScratchDoesNotBypassTheSelectedAllocator) {
  using namespace hyle::compute;
  AsyncOperationProbe probe;
  auto context =
      Context::create(cpu::device(), make_async_provider(probe)).value();
  const Operation operation = Operation::custom(WideMetadataSchema{});
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4})
          .value();
  std::array<TensorDesc, 9> descriptions;
  descriptions.fill(description);
  std::array<detail::ResourceBindingRequirement, 9> inputs{}, outputs{};
  hyle::mem::QuotaAllocator denied(hyle::mem::system_allocator(), 0);
  auto rejected = detail::OperationPreparationAccess::propose(
      context, 0, operation, descriptions, descriptions, inputs, outputs, {},
      denied.get());
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(probe.proposal_count, 0u);
  auto proposal =
      detail::OperationPreparationAccess::propose(
          context, 0, operation, descriptions, descriptions, inputs, outputs)
          .value();
  std::array<hyle::mem::MemoryResourceRef, 9> resources;
  resources.fill(hyle::mem::system_resource());
  auto failed = detail::OperationPreparationAccess::prepare_candidate(
      context, proposal, operation, descriptions, descriptions, inputs, outputs,
      {resources, resources}, {}, denied.get());
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  EXPECT_EQ(probe.prepare_count, 0u);
  hyle::mem::QuotaAllocator allowed(hyle::mem::system_allocator(), 65536);
  {
    auto prepared = detail::OperationPreparationAccess::prepare_candidate(
        context, proposal, operation, descriptions, descriptions, inputs,
        outputs, {resources, resources}, {}, allowed.get());
    ASSERT_TRUE(prepared) << prepared.error().message();
    ASSERT_TRUE(prepared->succeeded());
    EXPECT_GT(allowed.used(), 0u);
  }
  EXPECT_EQ(allowed.used(), 0u);
  EXPECT_EQ(probe.prepared_destroy_count, 1u);
}

TEST(RuntimeProviders, MetadataOverrideRetainsStorageUntilRealCompletion) {
  using namespace hyle::compute;
  hyle::mem::QuotaAllocator run_owner(hyle::mem::system_allocator(), 4096);
  AsyncOperationProbe probe;
  auto context =
      Context::create(async_device(probe), make_async_provider(probe)).value();
  auto queue = context.create_queue().value();
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4})
          .value();
  auto prepared = prepare(context, Unary{UnaryKind::kNeg}, description).value();
  auto input =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16).value();
  auto output =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16).value();
  auto invocation = detail::OperationPreparationAccess::create_invocation(
                        prepared, {}, run_owner.get())
                        .value();
  const auto bytes = run_owner.used();
  ASSERT_GT(bytes, 0u);
  auto completion = invocation.submit(
      queue, ConstTensorView::create(description, input.view()).value(),
      TensorView::create(description, output.view()).value());
  ASSERT_TRUE(completion) << completion.error().message();
  EXPECT_EQ(run_owner.used(), bytes);
  prepared = {};
  invocation = {};
  *completion = {};
  ASSERT_TRUE(probe.abandoned.valid());
  EXPECT_EQ(run_owner.used(), bytes);
  EXPECT_EQ(probe.invocation_destroy_count, 0u);
  EXPECT_EQ(probe.prepared_destroy_count, 0u);
  probe.complete = true;
  const auto lifetime = std::exchange(probe.abandoned, {});
  lifetime.terminal(lifetime.state, hyle::Status::Ok());
  lifetime.release(lifetime.state);
  EXPECT_EQ(run_owner.used(), 0u);
  EXPECT_EQ(probe.invocation_destroy_count, 1u);
  EXPECT_EQ(probe.prepared_destroy_count, 1u);
}

TEST(RuntimeProviders, InvocationFailureDestroysPartialWithItsSelectedOwner) {
  using namespace hyle::compute;
  AsyncOperationProbe probe;
  auto context =
      Context::create(async_device(probe), make_async_provider(probe)).value();
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4})
          .value();
  auto prepared = prepare(context, Unary{UnaryKind::kNeg}, description).value();
  hyle::mem::QuotaAllocator no_storage(hyle::mem::system_allocator(), 0);
  probe.fail_invocation = true;
  auto rejected = detail::OperationPreparationAccess::create_invocation(
      prepared, {}, no_storage.get());
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsDataLoss());
  EXPECT_EQ(probe.invocation_destroy_count, 1u);
  EXPECT_EQ(no_storage.used(), 0u);
  probe.fail_invocation = false;
  rejected = detail::OperationPreparationAccess::create_invocation(
      prepared, {}, no_storage.get());
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(probe.invocation_destroy_count, 2u);
  hyle::mem::QuotaAllocator sufficient(hyle::mem::system_allocator(), 4096);
  {
    auto retry = detail::OperationPreparationAccess::create_invocation(
        prepared, {}, sufficient.get());
    ASSERT_TRUE(retry) << retry.error().message();
    EXPECT_GT(sufficient.used(), 0u);
  }
  EXPECT_EQ(probe.invocation_destroy_count, 3u);
  EXPECT_EQ(sufficient.used(), 0u);
}

TEST(RuntimeProviders,
     PublicDelegatedDirectChecksBindingsAndRetainsBothOwners) {
  DelegatedProbe probe;
  probe.attach_owner = true;
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                make_delegated_provider(probe))
                     .value();
  const std::array<std::int64_t, 1> shape{4};
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32, shape)
          .value();
  auto input_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16).value();
  auto output_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16).value();
  const std::array inputs{
      hyle::compute::ConstTensorView::create(description, input_buffer.view())
          .value()};
  const std::array outputs{
      hyle::compute::TensorView::create(description, output_buffer.view())
          .value()};
  const hyle::compute::Operation operation(
      hyle::compute::Unary{hyle::compute::UnaryKind::kNeg});
  auto prepared =
      hyle::compute::prepare_delegated(context, operation, inputs, outputs);
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->runtime_properties().mode,
            hyle::compute::ProviderRuntimeMode::kDelegated);
  auto invocation = prepared->create_invocation({.dependency_capacity = 1});
  ASSERT_TRUE(invocation.has_value());
  EXPECT_FALSE(invocation->submit({}, outputs).has_value());
  hyle::compute::Completion invalid;
  const std::array invalid_dependency{&invalid};
  EXPECT_FALSE(invocation->submit_after(invalid_dependency, inputs, outputs)
                   .has_value());
  auto ready = hyle::compute::Completion::completed();
  const std::array too_many{&ready, &ready};
  auto overflow = invocation->submit_after(too_many, inputs, outputs);
  ASSERT_FALSE(overflow.has_value());
  EXPECT_TRUE(overflow.error().IsResourceExhausted());
  EXPECT_EQ(probe.submit_count, 0u);
  const std::array dependencies{&ready};
  auto source_failure = hyle::compute::Completion::completed(
      hyle::Status::Unavailable("source runtime failed"));
  const std::array failed_dependencies{&source_failure};
  auto dependency_failure =
      invocation->submit_after(failed_dependencies, inputs, outputs);
  ASSERT_FALSE(dependency_failure.has_value());
  EXPECT_TRUE(dependency_failure.error().IsUnavailable());
  EXPECT_EQ(probe.submit_count, 0u);
  auto completion = invocation->submit_after(dependencies, inputs, outputs);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(invocation->in_flight());
  ASSERT_NE(probe.pending, nullptr);
  EXPECT_FALSE(probe.pending->complete); // submit did not hide a wait
  EXPECT_FALSE(invocation->submit(inputs, outputs).has_value());
  EXPECT_TRUE(invocation->in_flight());
  EXPECT_EQ(completion->poll().value(),
            hyle::compute::CompletionPoll::kPending);
  ASSERT_TRUE(completion->wait().ok());
  EXPECT_FALSE(invocation->in_flight());
  EXPECT_EQ(probe.owner_terminal_count, 1u);
  EXPECT_EQ(probe.owner_release_count, 1u);

  probe.fail = true;
  auto failed = invocation->submit(inputs, outputs);
  ASSERT_TRUE(failed.has_value());
  EXPECT_TRUE(failed->wait().IsDataLoss());
  probe.fail = false;
  auto retired = invocation->submit(inputs, outputs);
  ASSERT_FALSE(retired.has_value());
  EXPECT_TRUE(retired.error().IsFailedPrecondition());
  EXPECT_FALSE(invocation->in_flight());
  *invocation = prepared->create_invocation({.dependency_capacity = 1}).value();
  auto cancelled = invocation->submit(inputs, outputs);
  ASSERT_TRUE(cancelled.has_value());
  ASSERT_TRUE(cancelled->request_cancel().ok());
  EXPECT_TRUE(cancelled->wait().IsCancelled());
  EXPECT_EQ(probe.owner_release_count, 3u);
  EXPECT_FALSE(invocation->submit(inputs, outputs).has_value());
  *invocation = prepared->create_invocation({.dependency_capacity = 1}).value();

  std::array<hyle::compute::Completion, 2> competing;
  std::barrier start(3);
  auto submit = [&](std::size_t index) {
    start.arrive_and_wait();
    auto attempt = invocation->submit(inputs, outputs);
    if (attempt)
      competing[index] = std::move(*attempt);
    else
      EXPECT_TRUE(attempt.error().IsFailedPrecondition());
  };
  std::thread first(submit, 0), second(submit, 1);
  start.arrive_and_wait();
  first.join();
  second.join();
  EXPECT_NE(competing[0].valid(), competing[1].valid());
  *invocation = {};
  *prepared = {};
  competing = {};
  EXPECT_EQ(probe.invocation_destroy_count, 2u);
  EXPECT_EQ(probe.owner_release_count, 3u);
  ASSERT_TRUE(context.drain().ok());
  EXPECT_EQ(probe.invocation_destroy_count, 3u);
  EXPECT_EQ(probe.owner_release_count, 4u);
}

TEST(RuntimeProviders,
     DelegatedDirectRejectsClosedContextAndInvalidPreparation) {
  DelegatedProbe probe;
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                make_delegated_provider(probe))
                     .value();
  const std::array<std::int64_t, 1> shape{1};
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32, shape)
          .value();
  const std::array descriptions{description};
  const std::array resources{hyle::mem::system_resource()};
  const hyle::compute::Operation operation(
      hyle::compute::Unary{hyle::compute::UnaryKind::kNeg});
  auto prepared = hyle::compute::prepare_delegated(
      context, operation, descriptions, descriptions,
      {.inputs = resources, .outputs = resources});
  ASSERT_TRUE(prepared.has_value());
  auto invocation = prepared->create_invocation();
  ASSERT_TRUE(invocation.has_value());
  auto input_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4).value();
  auto output_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4).value();
  const std::array inputs{
      hyle::compute::ConstTensorView::create(description, input_buffer.view())
          .value()};
  const std::array outputs{
      hyle::compute::TensorView::create(description, output_buffer.view())
          .value()};
  auto invalid =
      hyle::compute::prepare_delegated({}, operation, inputs, outputs);
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsFailedPrecondition());
  ASSERT_TRUE(context.close().ok());
  auto rejected = invocation->submit(inputs, outputs);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsFailedPrecondition());
  EXPECT_EQ(probe.submit_count, 0u);
}

TEST(RuntimeProviders, AbandonedDelegatedFailureRetiresOnlyItsOwnSlot) {
  DelegatedProbe probe;
  probe.attach_owner = true;
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                make_delegated_provider(probe))
                     .value();
  const auto desc =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32,
                                       std::array<std::int64_t, 1>{4})
          .value();
  auto input =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16).value();
  auto output =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 16).value();
  const std::array inputs{
      hyle::compute::ConstTensorView::create(desc, input.view()).value()};
  const std::array outputs{
      hyle::compute::TensorView::create(desc, output.view()).value()};
  auto prepared = hyle::compute::prepare_delegated(
                      context,
                      hyle::compute::Operation(
                          hyle::compute::Unary{hyle::compute::UnaryKind::kNeg}),
                      inputs, outputs)
                      .value();
  auto invocation = prepared.create_invocation().value();
  auto independent = prepared.create_invocation().value();
  auto abandoned = invocation.submit(inputs, outputs).value();
  probe.fail = true;
  abandoned = {};
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_EQ(probe.owner_release_count, 0u);
  ASSERT_TRUE(context.drain().ok());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(probe.owner_release_count, 1u);
  probe.fail = false;
  auto retired = invocation.submit(inputs, outputs);
  ASSERT_FALSE(retired.has_value());
  EXPECT_TRUE(retired.error().IsFailedPrecondition());
  EXPECT_EQ(probe.submit_count, 1u);
  ASSERT_TRUE(independent.submit(inputs, outputs)->wait().ok());
}

TEST(RuntimeProviders,
     DelegatedCompletionOwnsEndpointCancellationDrainAndAbandonment) {
  DelegatedProbe probe;
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                make_delegated_provider(probe))
                     .value();
  const std::array<std::int64_t, 1> shape{4};
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32, shape)
          .value();
  const hyle::compute::Operation operation(
      hyle::compute::Unary{hyle::compute::UnaryKind::kNeg});
  auto direct = hyle::compute::prepare(
      context, hyle::compute::Unary{hyle::compute::UnaryKind::kNeg},
      description);
  ASSERT_FALSE(direct.has_value());
  EXPECT_TRUE(direct.error().IsUnimplemented());

  std::array<hyle::compute::detail::ResourceBindingRequirement, 1>
      input_requirements{};
  std::array<hyle::compute::detail::ResourceBindingRequirement, 1>
      output_requirements{};
  const std::array input_descriptions{description};
  const std::array output_descriptions{description};
  auto proposal = hyle::compute::detail::OperationPreparationAccess::propose(
      context, 0, operation, input_descriptions, output_descriptions,
      input_requirements, output_requirements);
  ASSERT_TRUE(proposal.has_value()) << proposal.error().message();
  const std::array input_resources{hyle::mem::system_resource()};
  const std::array output_resources{hyle::mem::system_resource()};
  const hyle::compute::OperationResourceBindings resource_bindings{
      .inputs = input_resources,
      .outputs = output_resources,
  };

  probe.return_invalid_artifact = true;
  auto invalid_artifact =
      hyle::compute::detail::OperationPreparationAccess::prepare_candidate(
          context, *proposal, operation, input_descriptions,
          output_descriptions, input_requirements, output_requirements,
          resource_bindings);
  ASSERT_FALSE(invalid_artifact.has_value());
  EXPECT_TRUE(invalid_artifact.error().IsInternal());
  probe.return_invalid_artifact = false;
  auto attempt =
      hyle::compute::detail::OperationPreparationAccess::prepare_candidate(
          context, *proposal, operation, input_descriptions,
          output_descriptions, input_requirements, output_requirements,
          resource_bindings)
          .value();
  ASSERT_TRUE(attempt.succeeded());
  auto prepared = std::move(attempt.prepared);
  EXPECT_EQ(prepared.runtime_properties().mode,
            hyle::compute::ProviderRuntimeMode::kDelegated);
  EXPECT_EQ(prepared.runtime_properties().workspace_owner,
            hyle::compute::ProviderWorkspaceOwner::kProvider);

  auto input_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4 * sizeof(float))
          .value();
  auto output_buffer =
      hyle::mem::Buffer::create(hyle::mem::system_resource(), 4 * sizeof(float))
          .value();
  const auto input =
      hyle::compute::ConstTensorView::create(description, input_buffer.view())
          .value();
  const auto output =
      hyle::compute::TensorView::create(description, output_buffer.view())
          .value();
  auto invocation = prepared.create_invocation().value();

  const auto endpoint =
      hyle::compute::detail::OperationInvocationAccess::delegated_endpoint(
          invocation);
  EXPECT_TRUE(endpoint.valid());
  EXPECT_TRUE(
      hyle::compute::detail::OperationInvocationAccess::can_import_endpoint(
          invocation, endpoint));
  EXPECT_FALSE(
      hyle::compute::detail::OperationInvocationAccess::can_import_endpoint(
          invocation, {}));
  EXPECT_EQ(probe.submit_count, 0u);

  const std::array inputs{input};
  const std::array outputs{output};
  auto completion = hyle::compute::detail::OperationInvocationAccess::
      submit_delegated_prevalidated(invocation, {}, inputs, outputs);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_TRUE(hyle::compute::detail::OperationInvocationAccess::
                  submit_delegated_prevalidated(invocation, {}, inputs, outputs)
                      .error()
                      .IsFailedPrecondition());
  EXPECT_EQ(completion->poll().value(),
            hyle::compute::CompletionPoll::kPending);
  EXPECT_TRUE(completion->wait().ok());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(probe.pending, nullptr);

  completion = hyle::compute::detail::OperationInvocationAccess::
      submit_delegated_prevalidated(invocation, {}, inputs, outputs);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->request_cancel().ok());
  EXPECT_TRUE(completion->wait().IsCancelled());
  EXPECT_FALSE(invocation.in_flight());

  EXPECT_TRUE(hyle::compute::detail::OperationInvocationAccess::
                  submit_delegated_prevalidated(invocation, {}, inputs, outputs)
                      .error()
                      .IsFailedPrecondition());
  invocation = prepared.create_invocation().value();
  probe.fail = true;
  completion = hyle::compute::detail::OperationInvocationAccess::
      submit_delegated_prevalidated(invocation, {}, inputs, outputs);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsDataLoss());
  probe.fail = false;

  invocation = prepared.create_invocation().value();
  completion = hyle::compute::detail::OperationInvocationAccess::
      submit_delegated_prevalidated(invocation, {}, inputs, outputs);
  ASSERT_TRUE(completion.has_value());
  invocation = {};
  EXPECT_EQ(probe.invocation_destroy_count, 2u);
  *completion = {};
  ASSERT_NE(probe.pending, nullptr);
  EXPECT_TRUE(probe.pending->abandoned);
  EXPECT_TRUE(context.drain().ok());
  EXPECT_EQ(probe.drain_count, 1u);
  EXPECT_EQ(probe.pending, nullptr);
  EXPECT_EQ(probe.invocation_destroy_count, 3u);
}

} // namespace
