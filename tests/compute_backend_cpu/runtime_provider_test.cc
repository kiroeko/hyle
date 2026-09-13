#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "backend_internal.h"
#include "completion_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
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

hyle::Status propose_provider(
    void *context, hyle::compute::Operation,
    std::span<const hyle::compute::TensorDesc>,
    std::span<const hyle::compute::TensorDesc>,
    std::span<hyle::compute::detail::ResourceBindingRequirement>,
    std::span<hyle::compute::detail::ResourceBindingRequirement>,
    hyle::compute::detail::ProviderOperationProposal *) noexcept {
  const auto &provider = *static_cast<hyle::compute::Provider *>(context);
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->prepared[payload.probe->prepare_count++] = payload.index;
  return hyle::Status::Unimplemented("probe provider declines operation");
}

hyle::Status prepare_provider(
    void *, const hyle::compute::detail::ProviderPrepareRequest &,
    hyle::mem::AllocatorRef,
    hyle::compute::detail::ProviderPrepared *) noexcept {
  return hyle::Status::Unimplemented("probe provider declines operation");
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
  hyle::compute::detail::CompletionLifetime abandoned{};
  hyle::Status abandoned_hint{};
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

void destroy_async_prepared(void *, hyle::mem::AllocatorRef) noexcept {}

hyle::compute::AliasSupport async_alias_support(void *, std::size_t,
                                                std::size_t) noexcept {
  return {};
}

hyle::Status create_async_invocation(void *prepared, hyle::mem::AllocatorRef,
                                     hyle::mem::BufferView,
                                     void **invocation) noexcept {
  *invocation = prepared;
  return hyle::Status::Ok();
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
    void *, hyle::compute::Operation,
    std::span<const hyle::compute::TensorDesc>,
    std::span<const hyle::compute::TensorDesc>,
    std::span<hyle::compute::detail::ResourceBindingRequirement>,
    std::span<hyle::compute::detail::ResourceBindingRequirement>,
    hyle::compute::detail::ProviderOperationProposal *proposal) noexcept {
  *proposal = hyle::compute::detail::ProviderOperationProposal{
      .route = 1,
      .runtime = hyle::compute::ProviderRuntimeProperties{
          .mode = hyle::compute::ProviderRuntimeMode::kSharedQueue,
          .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
          .blocks_host = false,
          .uses_private_threads = false,
          .supports_concurrent_invocations = true,
          .capture_compatible = false,
      },
      .workspace = hyle::compute::WorkspaceRequirement{},
  };
  return hyle::Status::Ok();
}

hyle::Status prepare_async_operation(
    void *context, const hyle::compute::detail::ProviderPrepareRequest &request,
    hyle::mem::AllocatorRef,
    hyle::compute::detail::ProviderPrepared *prepared) noexcept {
  if (request.route != 1)
    return hyle::Status::InvalidArgument("test async route is invalid");
  *prepared = hyle::compute::detail::ProviderPrepared{
      .vtable = &kAsyncPreparedVTable,
      .state = context,
      .runtime = hyle::compute::ProviderRuntimeProperties{
          .mode = hyle::compute::ProviderRuntimeMode::kSharedQueue,
          .workspace_owner = hyle::compute::ProviderWorkspaceOwner::kCaller,
          .blocks_host = false,
          .uses_private_threads = false,
          .supports_concurrent_invocations = true,
          .capture_compatible = false,
      },
  };
  return hyle::Status::Ok();
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
};

constexpr int kDelegatedEndpointAuthority = 0;

hyle::Status delegated_terminal(
    const DelegatedCompletionState &state) noexcept {
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

hyle::Status create_delegated_context(
    const hyle::compute::Provider &provider, hyle::compute::Device,
    void *, hyle::mem::AllocatorRef allocator, void **output) noexcept {
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

hyle::Status create_delegated_invocation(
    void *prepared, hyle::mem::AllocatorRef allocator, hyle::mem::BufferView,
    void **output) noexcept {
  if (output == nullptr)
    return hyle::Status::InvalidArgument(
        "delegated invocation output is null");
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

void destroy_delegated_invocation(
    void *value, hyle::mem::AllocatorRef allocator) noexcept {
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
    std::span<const hyle::compute::detail::CompletionDependency>,
    std::span<const hyle::compute::ConstTensorView> inputs,
    std::span<const hyle::compute::TensorView> outputs) noexcept {
  auto *invocation = static_cast<DelegatedInvocation *>(invocation_value);
  if (invocation == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return std::unexpected(
        hyle::Status::Internal("delegated bindings are invalid"));
  DelegatedProbe &probe = *invocation->context->probe;
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
  auto *state = ::new (storage)
      DelegatedCompletionState{.probe = &probe,
                               .allocator = invocation->allocator};
  probe.pending = state;
  ++probe.submit_count;
  auto completion = hyle::compute::detail::CompletionAccess::make(
      &kDelegatedCompletionVTable, state);
  if (!completion.valid()) {
    release_delegated_completion(state);
    return std::unexpected(
        hyle::Status::Internal("delegated completion creation failed"));
  }
  return completion;
}

hyle::compute::detail::CompletionEndpoint delegated_endpoint(
    void *, void *invocation) noexcept {
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

hyle::compute::detail::EnqueueResult invalid_shared_submit(
    void *, void *, hyle::compute::Queue &,
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
    void *, hyle::compute::Operation operation,
    std::span<const hyle::compute::TensorDesc> inputs,
    std::span<const hyle::compute::TensorDesc> outputs,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        input_requirements,
    std::span<hyle::compute::detail::ResourceBindingRequirement>
        output_requirements,
    hyle::compute::detail::ProviderOperationProposal *proposal) noexcept {
  if (proposal == nullptr || operation.kind() !=
                                 hyle::compute::OperationKind::kUnary ||
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
      .workspace = hyle::compute::WorkspaceRequirement{},
  };
  return hyle::Status::Ok();
}

hyle::Status prepare_delegated(
    void *context,
    const hyle::compute::detail::ProviderPrepareRequest &request,
    hyle::mem::AllocatorRef,
    hyle::compute::detail::ProviderPrepared *prepared) noexcept {
  if (context == nullptr || prepared == nullptr || request.route != 1 ||
      request.inputs.size() != 1 || request.outputs.size() != 1) {
    return hyle::Status::InvalidArgument(
        "delegated exact prepare request is invalid");
  }
  *prepared = hyle::compute::detail::ProviderPrepared{
      .vtable = static_cast<DelegatedContext *>(context)
                        ->probe->return_invalid_artifact
                    ? &kInvalidDelegatedPreparedVTable
                    : &kDelegatedPreparedVTable,
      .state = context,
      .runtime = kDelegatedRuntime,
  };
  return hyle::Status::Ok();
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

hyle::compute::Provider make_delegated_provider(
    DelegatedProbe &probe) noexcept {
  return hyle::compute::detail::ProviderAccess::make(
      &kDelegatedProviderVTable, &probe);
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

TEST(RuntimeProviders, ContextCopiesAnArbitraryOrderedSequence) {
  ProviderProbe probe;
  std::array<hyle::compute::Provider, 6> providers{};
  for (std::size_t index = 0; index < providers.size(); ++index)
    providers[index] = make_provider(probe, index);

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
                                           *description, *description);
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

TEST(RuntimeProviders, ContextCreationRollsBackInReverseOrder) {
  ProviderProbe probe;
  probe.fail_index = 2;
  const std::array providers{make_provider(probe, 0), make_provider(probe, 1),
                             make_provider(probe, 2), make_provider(probe, 3)};

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
  auto created = hyle::compute::Context::create(
      hyle::compute::cpu::device(), incomplete);
  ASSERT_FALSE(created.has_value());
  EXPECT_TRUE(created.error().IsInvalidArgument());
  EXPECT_EQ(probe.create_count, 0u);
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

TEST(RuntimeProviders,
     DelegatedCompletionOwnsEndpointCancellationDrainAndAbandonment) {
  DelegatedProbe probe;
  auto context = hyle::compute::Context::create(
                     hyle::compute::cpu::device(),
                     make_delegated_provider(probe))
                     .value();
  const std::array<std::int64_t, 1> shape{4};
  const auto description =
      hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32,
                                       shape)
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
      hyle::compute::detail::OperationPreparationAccess::prepare(
          context, *proposal, operation, input_descriptions,
          output_descriptions, input_requirements, output_requirements,
          resource_bindings);
  ASSERT_FALSE(invalid_artifact.has_value());
  EXPECT_TRUE(invalid_artifact.error().IsInternal());
  probe.return_invalid_artifact = false;
  auto prepared = hyle::compute::detail::OperationPreparationAccess::prepare(
                      context, *proposal, operation, input_descriptions,
                      output_descriptions, input_requirements,
                      output_requirements, resource_bindings)
                      .value();
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
  const auto input = hyle::compute::ConstTensorView::create(
                         description, input_buffer.view())
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
                        submit_delegated_prevalidated(invocation, {}, inputs,
                                                      outputs);
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
                   submit_delegated_prevalidated(invocation, {}, inputs,
                                                 outputs);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->request_cancel().ok());
  EXPECT_TRUE(completion->wait().IsCancelled());
  EXPECT_FALSE(invocation.in_flight());

  probe.fail = true;
  completion = hyle::compute::detail::OperationInvocationAccess::
                   submit_delegated_prevalidated(invocation, {}, inputs,
                                                 outputs);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().IsDataLoss());
  probe.fail = false;

  completion = hyle::compute::detail::OperationInvocationAccess::
                   submit_delegated_prevalidated(invocation, {}, inputs,
                                                 outputs);
  ASSERT_TRUE(completion.has_value());
  invocation = {};
  EXPECT_EQ(probe.invocation_destroy_count, 0u);
  *completion = {};
  ASSERT_NE(probe.pending, nullptr);
  EXPECT_TRUE(probe.pending->abandoned);
  EXPECT_TRUE(context.drain().ok());
  EXPECT_EQ(probe.drain_count, 1u);
  EXPECT_EQ(probe.pending, nullptr);
  EXPECT_EQ(probe.invocation_destroy_count, 1u);
}

} // namespace
