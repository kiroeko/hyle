#include "hyle/compute/device.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "backend_internal.h"
#include "context_state.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "provider_internal.h"

namespace hyle::compute::detail {
namespace {

constexpr std::size_t align_up(std::size_t value,
                               std::size_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

void ContextState::destroy() noexcept {
  ProviderSlot *slots = providers();
  for (std::size_t index = provider_count; index != 0; --index) {
    ProviderSlot &slot = slots[index - 1];
    const ProviderVTable *table = ProviderAccess::vtable(slot.provider);
    if (table != nullptr && table->destroy_context != nullptr)
      table->destroy_context(slot.state, allocator);
    std::destroy_at(&slot);
  }
  if (backend_initialized && backend_vtable != nullptr &&
      backend_vtable->destroy_context != nullptr)
    backend_vtable->destroy_context(backend_state, allocator);
  const mem::AllocatorRef owner = allocator;
  const std::size_t bytes = allocation_size;
  this->~ContextState();
  owner.deallocate(this, bytes, alignof(ContextState));
}

void QueueState::destroy() noexcept {
  admission.close();
  ContextState *owner_context = context;
  const mem::AllocatorRef allocator = owner_context->allocator;
  const BackendVTable *table = owner_context->backend_vtable;
  if (table != nullptr && table->destroy_queue != nullptr)
    table->destroy_queue(backend_state, allocator);
  this->~QueueState();
  allocator.deallocate(this, sizeof(QueueState), alignof(QueueState));
  detail::release(owner_context);
}

Result<ContextState *>
create_context_state(Device device, ContextOptions options,
                     const BackendVTable *backend_vtable) noexcept {
  mem::AllocatorRef allocator = options.metadata_allocator;
  if (!allocator.valid())
    allocator = mem::system_allocator();

  constexpr std::size_t slot_alignment = alignof(ContextState::ProviderSlot);
  static_assert((slot_alignment & (slot_alignment - 1)) == 0);
  static_assert(alignof(ContextState) >= slot_alignment);
  const std::size_t slot_offset =
      align_up(sizeof(ContextState), slot_alignment);
  if (options.providers.size() >
      ((std::numeric_limits<std::size_t>::max)() - slot_offset) /
          sizeof(ContextState::ProviderSlot)) {
    return std::unexpected(
        Status::OutOfRange("compute provider metadata size overflows"));
  }
  const std::size_t allocation_size =
      slot_offset +
      options.providers.size() * sizeof(ContextState::ProviderSlot);
  void *storage = allocator.allocate(allocation_size, alignof(ContextState));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("compute context allocation failed"));
  }
  auto *state = ::new (storage)
      ContextState(device, allocator, allocation_size, slot_offset,
                   options.providers.size(), backend_vtable);

  Status status =
      backend_vtable->create_context(device, allocator, &state->backend_state);
  if (!status.ok()) {
    detail::release(state);
    return std::unexpected(std::move(status));
  }
  state->backend_initialized = true;

  ContextState::ProviderSlot *slots = state->providers();
  for (const Provider &provider : options.providers) {
    const ProviderVTable *table = ProviderAccess::vtable(provider);
    if (table == nullptr || !table->identity.valid() ||
        table->create_context == nullptr || table->destroy_context == nullptr ||
        table->prepare == nullptr) {
      detail::release(state);
      return std::unexpected(
          Status::InvalidArgument("compute provider is incomplete"));
    }
    ContextState::ProviderSlot &slot = slots[state->provider_count];
    std::construct_at(&slot, ContextState::ProviderSlot{provider, nullptr});
    status = table->create_context(slot.provider, device, state->backend_state,
                                   allocator, &slot.state);
    if (!status.ok()) {
      std::destroy_at(&slot);
      detail::release(state);
      return std::unexpected(std::move(status));
    }
    ++state->provider_count;
  }
  return state;
}

} // namespace hyle::compute::detail

namespace hyle::compute {

void detail::retain(ContextState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

void detail::release(ContextState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

void detail::retain(QueueState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

void detail::release(QueueState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

detail::BackendExtensionRef::BackendExtensionRef(detail::ContextState *owner,
                                                 const void *interface_value,
                                                 BackendExtensionId id,
                                                 BackendId backend) noexcept
    : owner_(owner), interface_(interface_value), id_(id), backend_(backend) {
  detail::retain(owner_);
}

detail::BackendExtensionRef::BackendExtensionRef(
    const detail::BackendExtensionRef &other) noexcept
    : owner_(other.owner_), interface_(other.interface_), id_(other.id_),
      backend_(other.backend_) {
  detail::retain(owner_);
}

detail::BackendExtensionRef &detail::BackendExtensionRef::operator=(
    const detail::BackendExtensionRef &other) noexcept {
  if (this != &other) {
    detail::retain(other.owner_);
    detail::release(owner_);
    owner_ = other.owner_;
    interface_ = other.interface_;
    id_ = other.id_;
    backend_ = other.backend_;
  }
  return *this;
}

detail::BackendExtensionRef::BackendExtensionRef(
    detail::BackendExtensionRef &&other) noexcept
    : owner_(other.owner_), interface_(other.interface_), id_(other.id_),
      backend_(other.backend_) {
  other.owner_ = nullptr;
  other.interface_ = nullptr;
  other.id_ = {};
  other.backend_ = {};
}

detail::BackendExtensionRef &detail::BackendExtensionRef::operator=(
    detail::BackendExtensionRef &&other) noexcept {
  if (this != &other) {
    detail::release(owner_);
    owner_ = other.owner_;
    interface_ = other.interface_;
    id_ = other.id_;
    backend_ = other.backend_;
    other.owner_ = nullptr;
    other.interface_ = nullptr;
    other.id_ = {};
    other.backend_ = {};
  }
  return *this;
}

detail::BackendExtensionRef::~BackendExtensionRef() { detail::release(owner_); }

Result<detail::BackendExtensionRef>
detail::BackendExtensionAccess::query(const Context &context,
                                      BackendExtensionId id) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (!id.valid()) {
    return std::unexpected(
        Status::InvalidArgument("compute backend extension id is invalid"));
  }
  if (!state->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  const void *interface_value =
      state->backend_vtable->query_extension(state->backend_state, id);
  Result<BackendExtensionRef> result =
      interface_value != nullptr
          ? Result<BackendExtensionRef>(BackendExtensionRef(
                state, interface_value, id, state->backend_vtable->id))
          : Result<BackendExtensionRef>(std::unexpected(Status::Unimplemented(
                "compute backend does not implement this extension")));
  state->admission.leave();
  return result;
}

Result<detail::BackendExtensionRef>
detail::BackendExtensionAccess::query(const Queue &queue,
                                      BackendExtensionId id) noexcept {
  detail::QueueState *state = detail::ContextAccess::state(queue);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("queue is invalid"));
  if (!id.valid()) {
    return std::unexpected(
        Status::InvalidArgument("compute backend extension id is invalid"));
  }
  if (!state->context->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  if (!state->admission.try_enter()) {
    state->context->admission.leave();
    return std::unexpected(Status::FailedPrecondition("queue is closed"));
  }
  const void *interface_value = state->context->backend_vtable->query_extension(
      state->context->backend_state, id);
  Result<BackendExtensionRef> result =
      interface_value != nullptr
          ? Result<BackendExtensionRef>(
                BackendExtensionRef(state->context, interface_value, id,
                                    state->context->backend_vtable->id))
          : Result<BackendExtensionRef>(std::unexpected(Status::Unimplemented(
                "compute backend does not implement this extension")));
  state->admission.leave();
  state->context->admission.leave();
  return result;
}

Status
detail::BackendExtensionAccess::call(const BackendExtensionRef &extension,
                                     BackendExtensionCallFn function,
                                     void *call_context) noexcept {
  if (!extension.valid() || function == nullptr)
    return Status::InvalidArgument("compute backend extension call is invalid");
  if (!extension.owner_->admission.try_enter())
    return Status::FailedPrecondition("compute context is closed");
  Status result = function(extension.owner_->backend_state,
                           extension.interface_, call_context);
  extension.owner_->admission.leave();
  return result;
}

Result<Completion> detail::BackendExtensionAccess::submit(
    const BackendExtensionRef &extension, Queue &queue,
    BackendExtensionEnqueueFn function, void *call_context) noexcept {
  auto sequence = QueueSequence::begin(queue);
  if (!sequence)
    return std::unexpected(std::move(sequence).error());
  EnqueueResult enqueued =
      enqueue_admitted(extension, *sequence, function, call_context);
  sequence->record(std::move(enqueued));
  return sequence->finish();
}

detail::EnqueueResult detail::BackendExtensionAccess::enqueue_admitted(
    const BackendExtensionRef &extension, QueueSequence &sequence,
    BackendExtensionEnqueueFn function, void *call_context) noexcept {
  if (!extension.valid() || function == nullptr) {
    return EnqueueResult::rejected(
        Status::InvalidArgument("compute backend extension submit is invalid"));
  }
  if (!sequence.valid()) {
    return EnqueueResult::rejected(
        Status::FailedPrecondition("compute queue sequence is invalid"));
  }
  Queue &queue = sequence.queue();
  detail::QueueState *state = detail::ContextAccess::state(queue);
  if (state == nullptr || state->context != extension.owner_) {
    return EnqueueResult::rejected(Status::InvalidArgument(
        "queue and backend extension belong to different contexts"));
  }
  return function(extension.owner_->backend_state, extension.interface_,
                  state->backend_state, call_context);
}

detail::ProviderExtensionRef::ProviderExtensionRef(
    detail::ContextState *owner, void *provider_context,
    const void *interface_value, ProviderExtensionId id,
    ProviderIdentity provider) noexcept
    : owner_(owner), provider_context_(provider_context),
      interface_(interface_value), id_(id), provider_(provider) {
  detail::retain(owner_);
}

detail::ProviderExtensionRef::ProviderExtensionRef(
    const detail::ProviderExtensionRef &other) noexcept
    : owner_(other.owner_), provider_context_(other.provider_context_),
      interface_(other.interface_), id_(other.id_), provider_(other.provider_) {
  detail::retain(owner_);
}

detail::ProviderExtensionRef &detail::ProviderExtensionRef::operator=(
    const detail::ProviderExtensionRef &other) noexcept {
  if (this != &other) {
    detail::retain(other.owner_);
    detail::release(owner_);
    owner_ = other.owner_;
    provider_context_ = other.provider_context_;
    interface_ = other.interface_;
    id_ = other.id_;
    provider_ = other.provider_;
  }
  return *this;
}

detail::ProviderExtensionRef::ProviderExtensionRef(
    detail::ProviderExtensionRef &&other) noexcept
    : owner_(other.owner_), provider_context_(other.provider_context_),
      interface_(other.interface_), id_(other.id_), provider_(other.provider_) {
  other.owner_ = nullptr;
  other.provider_context_ = nullptr;
  other.interface_ = nullptr;
  other.id_ = {};
  other.provider_ = {};
}

detail::ProviderExtensionRef &detail::ProviderExtensionRef::operator=(
    detail::ProviderExtensionRef &&other) noexcept {
  if (this != &other) {
    detail::release(owner_);
    owner_ = other.owner_;
    provider_context_ = other.provider_context_;
    interface_ = other.interface_;
    id_ = other.id_;
    provider_ = other.provider_;
    other.owner_ = nullptr;
    other.provider_context_ = nullptr;
    other.interface_ = nullptr;
    other.id_ = {};
    other.provider_ = {};
  }
  return *this;
}

detail::ProviderExtensionRef::~ProviderExtensionRef() {
  detail::release(owner_);
}

Result<detail::ProviderExtensionRef>
detail::ProviderExtensionAccess::query(const Context &context,
                                       std::size_t provider_index,
                                       ProviderExtensionId id) noexcept {
  detail::ContextState *state = detail::ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (!id.valid()) {
    return std::unexpected(
        Status::InvalidArgument("compute provider extension id is invalid"));
  }
  if (provider_index >= state->provider_count) {
    return std::unexpected(
        Status::OutOfRange("compute provider index is out of range"));
  }
  if (!state->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  const detail::ContextState::ProviderSlot &slot =
      state->providers()[provider_index];
  const detail::ProviderVTable *table =
      detail::ProviderAccess::vtable(slot.provider);
  const void *interface_value = table->query_extension != nullptr
                                    ? table->query_extension(slot.state, id)
                                    : nullptr;
  Result<detail::ProviderExtensionRef> result =
      interface_value != nullptr
          ? Result<detail::ProviderExtensionRef>(detail::ProviderExtensionRef(
                state, slot.state, interface_value, id, table->identity))
          : Result<detail::ProviderExtensionRef>(
                std::unexpected(Status::Unimplemented(
                    "compute provider does not implement this extension")));
  state->admission.leave();
  return result;
}

Status detail::ProviderExtensionAccess::call(
    const detail::ProviderExtensionRef &extension,
    ProviderExtensionCallFn function, void *call_context) noexcept {
  if (!extension.valid() || function == nullptr)
    return Status::InvalidArgument(
        "compute provider extension call is invalid");
  if (!extension.owner_->admission.try_enter())
    return Status::FailedPrecondition("compute context is closed");
  const Status result =
      function(extension.provider_context_, extension.interface_, call_context);
  extension.owner_->admission.leave();
  return result;
}

Result<Completion> detail::ProviderExtensionAccess::submit(
    const detail::ProviderExtensionRef &extension, Queue &queue,
    std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs, ProviderExtensionSubmitFn function,
    void *call_context) noexcept {
  auto sequence = detail::QueueSequence::begin(queue);
  if (!sequence)
    return std::unexpected(std::move(sequence).error());
  EnqueueResult enqueued = enqueue_admitted(extension, *sequence, inputs,
                                            outputs, function, call_context);
  sequence->record(std::move(enqueued));
  return sequence->finish();
}

detail::EnqueueResult detail::ProviderExtensionAccess::enqueue_admitted(
    const detail::ProviderExtensionRef &extension,
    detail::QueueSequence &sequence, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs, ProviderExtensionSubmitFn function,
    void *call_context) noexcept {
  if (!extension.valid() || function == nullptr) {
    return EnqueueResult::rejected(Status::InvalidArgument(
        "compute provider extension submit is invalid"));
  }
  if (!sequence.valid()) {
    return EnqueueResult::rejected(
        Status::FailedPrecondition("compute queue sequence is invalid"));
  }
  Queue &queue = sequence.queue();
  detail::QueueState *queue_state = detail::ContextAccess::state(queue);
  if (queue_state == nullptr || queue_state->context != extension.owner_) {
    return EnqueueResult::rejected(Status::InvalidArgument(
        "queue and provider extension belong to different contexts"));
  }
  for (const ConstTensorView &input : inputs) {
    if (!input.valid()) {
      return EnqueueResult::rejected(
          Status::InvalidArgument("provider extension input is invalid"));
    }
    Status status = extension.owner_->backend_vtable->validate_resource(
        extension.owner_->device, input.storage().resource());
    if (!status.ok())
      return EnqueueResult::rejected(std::move(status));
  }
  for (const TensorView &output : outputs) {
    if (!output.valid()) {
      return EnqueueResult::rejected(
          Status::InvalidArgument("provider extension output is invalid"));
    }
    Status status = extension.owner_->backend_vtable->validate_resource(
        extension.owner_->device, output.storage().resource());
    if (!status.ok())
      return EnqueueResult::rejected(std::move(status));
  }
  return function(extension.provider_context_, extension.interface_,
                  call_context, queue);
}

BackendId Backend::id() const noexcept {
  return vtable_ != nullptr ? vtable_->id : BackendId{};
}

std::string_view Backend::name() const noexcept {
  return vtable_ != nullptr ? vtable_->name : std::string_view{"unknown"};
}

DeviceClass Backend::device_class() const noexcept {
  return vtable_ != nullptr ? vtable_->device_class : DeviceClass::kUnknown;
}

Result<std::size_t> Backend::device_count() const noexcept {
  if (vtable_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("backend is invalid"));
  if (vtable_->device_count == nullptr)
    return std::unexpected(
        Status::Internal("backend device-count callback is missing"));
  return vtable_->device_count(*this);
}

Result<Device> Backend::device(std::size_t index) const noexcept {
  if (vtable_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("backend is invalid"));
  if (vtable_->device == nullptr)
    return std::unexpected(
        Status::Internal("backend device callback is missing"));
  return vtable_->device(*this, index);
}

BackendId Device::backend_id() const noexcept {
  return vtable_ != nullptr ? vtable_->id : BackendId{};
}

std::string_view Device::backend_name() const noexcept {
  return vtable_ != nullptr ? vtable_->name : std::string_view{"unknown"};
}

DeviceClass Device::device_class() const noexcept {
  return vtable_ != nullptr ? vtable_->device_class : DeviceClass::kUnknown;
}

Result<DeviceProperties> Device::properties() const noexcept {
  if (vtable_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("device is invalid"));
  if (vtable_->properties == nullptr)
    return std::unexpected(
        Status::Internal("backend properties callback is missing"));
  return vtable_->properties(*this);
}

bool operator==(const Device &left, const Device &right) noexcept {
  if (left.vtable_ == nullptr || right.vtable_ == nullptr)
    return left.vtable_ == right.vtable_;
  if (left.vtable_ != right.vtable_ || left.vtable_->id != right.vtable_->id)
    return false;
  return left.vtable_->same_device != nullptr &&
         left.vtable_->same_device(left, right);
}

Context::Context(Context &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr) {
      state_->admission.close();
      detail::release(state_);
    }
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Context::~Context() {
  if (state_ != nullptr) {
    state_->admission.close();
    detail::release(state_);
  }
}

Result<Context> Context::create(Device device,
                                ContextOptions options) noexcept {
  if (!device.valid())
    return std::unexpected(Status::InvalidArgument("device is invalid"));
  const detail::BackendVTable *table = detail::DeviceAccess::vtable(device);
  if (table == nullptr || !table->id.valid() || table->name.empty() ||
      table->same_device == nullptr || table->properties == nullptr ||
      table->validate_resource == nullptr || table->create_context == nullptr ||
      table->destroy_context == nullptr || table->create_queue == nullptr ||
      table->destroy_queue == nullptr || table->begin_sequence == nullptr ||
      table->finish_sequence == nullptr ||
      table->reserve_completions == nullptr ||
      table->transfer_properties == nullptr || table->copy_async == nullptr ||
      table->fill_async == nullptr || table->wait_completion == nullptr ||
      table->can_wait_queue == nullptr || table->query_extension == nullptr ||
      table->drain_context == nullptr || table->drain_queue == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "device backend has an incomplete runtime contract"));
  }
  auto state = detail::create_context_state(device, options, table);
  if (!state)
    return std::unexpected(std::move(state).error());
  return Context(*state);
}

Result<Context> Context::create(Device device, Provider provider) noexcept {
  if (!provider.valid())
    return std::unexpected(
        Status::InvalidArgument("compute provider is invalid"));
  const Provider providers[]{provider};
  return create(device, ContextOptions{.providers = providers});
}

bool Context::is_open() const noexcept {
  return state_ != nullptr && state_->admission.is_open();
}

Device Context::device() const noexcept {
  return state_ != nullptr ? state_->device : Device{};
}

mem::AllocatorRef Context::metadata_allocator() const noexcept {
  return state_ != nullptr ? state_->allocator : mem::AllocatorRef{};
}

std::size_t Context::provider_count() const noexcept {
  return state_ != nullptr ? state_->provider_count : 0;
}

Result<ProviderIdentity>
Context::provider_identity(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (index >= state_->provider_count) {
    return std::unexpected(
        Status::OutOfRange("compute provider index is out of range"));
  }
  const detail::ProviderVTable *table =
      detail::ProviderAccess::vtable(state_->providers()[index].provider);
  return table->identity;
}

detail::ContextRef::ContextRef(const ContextRef &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

detail::ContextRef &
detail::ContextRef::operator=(const ContextRef &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

detail::ContextRef::ContextRef(ContextRef &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

detail::ContextRef &detail::ContextRef::operator=(ContextRef &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

detail::ContextRef::~ContextRef() { detail::release(state_); }

Result<detail::ContextRef>
detail::ContextRef::acquire(const Context &context) noexcept {
  ContextState *state = ContextAccess::state(context);
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (!state->admission.is_open())
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  detail::retain(state);
  return ContextRef(state);
}

Device detail::ContextRef::device() const noexcept {
  return state_ != nullptr ? state_->device : Device{};
}

bool detail::ContextRef::is_open() const noexcept {
  return state_ != nullptr && state_->admission.is_open();
}

bool detail::ContextRef::owns(const Queue &queue) const noexcept {
  QueueState *queue_state = ContextAccess::state(queue);
  return state_ != nullptr && queue_state != nullptr &&
         queue_state->context == state_;
}

Result<bool> detail::ContextRef::has_backend_extension(
    BackendExtensionId id) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(
        Status::FailedPrecondition("context lease is invalid"));
  if (!id.valid()) {
    return std::unexpected(
        Status::InvalidArgument("compute backend extension id is invalid"));
  }
  if (!state_->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  const bool supported = state_->backend_vtable->query_extension(
                             state_->backend_state, id) != nullptr;
  state_->admission.leave();
  return supported;
}

Status detail::ContextRef::validate_resource(
    mem::MemoryResourceRef resource) const noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("context lease is invalid");
  if (!resource.valid())
    return Status::InvalidArgument("memory resource is invalid");
  return state_->backend_vtable->validate_resource(state_->device, resource);
}

Result<Queue> Context::create_queue(QueueOptions options) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (!state_->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));

  void *storage = state_->allocator.allocate(sizeof(detail::QueueState),
                                             alignof(detail::QueueState));
  if (storage == nullptr) {
    state_->admission.leave();
    return std::unexpected(
        Status::ResourceExhausted("compute queue allocation failed"));
  }
  void *backend_queue = nullptr;
  Status status = state_->backend_vtable->create_queue(
      state_->backend_state, options, state_->allocator, &backend_queue);
  if (!status.ok()) {
    state_->allocator.deallocate(storage, sizeof(detail::QueueState),
                                 alignof(detail::QueueState));
    state_->admission.leave();
    return std::unexpected(std::move(status));
  }
  auto *queue = ::new (storage) detail::QueueState(state_, backend_queue);
  state_->admission.leave();
  return Queue(queue);
}

Result<Queue> detail::ContextAccess::create_queue(const Context &context,
                                                  BackendQueueCreateFn create,
                                                  void *call_context) noexcept {
  ContextState *state = context.state_;
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  if (create == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("backend queue factory is null"));
  }
  if (!state->admission.try_enter())
    return std::unexpected(Status::FailedPrecondition("context is closed"));

  void *storage =
      state->allocator.allocate(sizeof(QueueState), alignof(QueueState));
  if (storage == nullptr) {
    state->admission.leave();
    return std::unexpected(
        Status::ResourceExhausted("compute queue allocation failed"));
  }
  void *backend_queue = nullptr;
  Status status = create(state->backend_state, state->allocator, call_context,
                         &backend_queue);
  if (!status.ok()) {
    state->allocator.deallocate(storage, sizeof(QueueState),
                                alignof(QueueState));
    state->admission.leave();
    return std::unexpected(std::move(status));
  }
  auto *queue = ::new (storage) QueueState(state, backend_queue);
  state->admission.leave();
  return Queue(queue);
}

Status Context::close() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("context is invalid");
  state_->admission.close();
  return Status::Ok();
}

Status Context::drain() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("context is invalid");
  state_->admission.wait_idle();
  return state_->backend_vtable->drain_context(state_->backend_state);
}

Queue::Queue(Queue &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Queue &Queue::operator=(Queue &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr) {
      state_->admission.close();
      detail::release(state_);
    }
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Queue::~Queue() {
  if (state_ != nullptr) {
    state_->admission.close();
    detail::release(state_);
  }
}

bool Queue::is_open() const noexcept {
  return state_ != nullptr && state_->admission.is_open() &&
         state_->context->admission.is_open();
}

Device Queue::device() const noexcept {
  return state_ != nullptr ? state_->context->device : Device{};
}

Status Queue::close() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("queue is invalid");
  state_->admission.close();
  return Status::Ok();
}

Status Queue::drain() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("queue is invalid");
  state_->admission.wait_idle();
  return state_->context->backend_vtable->drain_queue(state_->backend_state);
}

Status Queue::reserve_completions(std::size_t minimum) noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("queue is invalid");
  if (!state_->context->admission.try_enter())
    return Status::FailedPrecondition("context is closed");
  if (!state_->admission.try_enter()) {
    state_->context->admission.leave();
    return Status::FailedPrecondition("queue is closed");
  }
  const Status result = state_->context->backend_vtable->reserve_completions(
      state_->backend_state, minimum);
  state_->admission.leave();
  state_->context->admission.leave();
  return result;
}

Status Queue::begin_submission() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("queue is invalid");
  if (!state_->context->admission.try_enter())
    return Status::FailedPrecondition("context is closed");
  if (!state_->admission.try_enter()) {
    state_->context->admission.leave();
    return Status::FailedPrecondition("queue is closed");
  }
  if (state_->submitting.test_and_set(std::memory_order_acquire)) {
    state_->admission.leave();
    state_->context->admission.leave();
    return Status::FailedPrecondition(
        "ordered queue already has an active submission sequence");
  }
  return Status::Ok();
}

void Queue::end_submission() noexcept {
  state_->submitting.clear(std::memory_order_release);
  state_->admission.leave();
  state_->context->admission.leave();
}

detail::QueueSequence::QueueSequence(QueueSequence &&other) noexcept
    : queue_(other.queue_), backend_sequence_(other.backend_sequence_),
      terminal_(std::move(other.terminal_)), accepted_(other.accepted_) {
  other.queue_ = nullptr;
  other.backend_sequence_ = nullptr;
  other.terminal_ = Status::Ok();
  other.accepted_ = false;
}

detail::QueueSequence &
detail::QueueSequence::operator=(QueueSequence &&other) noexcept {
  if (this != &other) {
    reset();
    queue_ = other.queue_;
    backend_sequence_ = other.backend_sequence_;
    terminal_ = std::move(other.terminal_);
    accepted_ = other.accepted_;
    other.queue_ = nullptr;
    other.backend_sequence_ = nullptr;
    other.terminal_ = Status::Ok();
    other.accepted_ = false;
  }
  return *this;
}

detail::QueueSequence::~QueueSequence() { reset(); }

Result<detail::QueueSequence>
detail::QueueSequence::begin(Queue &queue) noexcept {
  Status status = queue.begin_submission();
  if (!status.ok())
    return std::unexpected(std::move(status));
  detail::QueueState *state = detail::ContextAccess::state(queue);
  void *backend_sequence = nullptr;
  status = state->context->backend_vtable->begin_sequence(state->backend_state,
                                                          &backend_sequence);
  if (!status.ok()) {
    queue.end_submission();
    return std::unexpected(std::move(status));
  }
  return QueueSequence(&queue, backend_sequence);
}

Result<detail::QueueDependencyRoute>
detail::QueueSequence::dependency_route(const Queue &destination,
                                        const Queue &source) noexcept {
  detail::QueueState *destination_state =
      detail::ContextAccess::state(destination);
  detail::QueueState *source_state = detail::ContextAccess::state(source);
  if (destination_state == nullptr || source_state == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compute dependency queue is invalid"));
  }
  if (!destination.is_open() || !source.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("compute dependency queue is closed"));
  }
  if (destination_state == source_state)
    return detail::QueueDependencyRoute::kNoWait;
  auto source_properties = source_state->context->device.properties();
  if (!source_properties)
    return std::unexpected(std::move(source_properties).error());
  if (!source_properties->asynchronous_queues)
    return detail::QueueDependencyRoute::kNoWait;
  const bool compatible =
      destination_state->context->backend_vtable->can_wait_queue(
          destination_state->backend_state,
          source_state->context->backend_vtable->id,
          source_state->backend_state);
  return compatible ? detail::QueueDependencyRoute::kDeviceWait
                    : detail::QueueDependencyRoute::kHostWait;
}

void detail::QueueSequence::record(EnqueueResult result) noexcept {
  if (queue_ == nullptr)
    return;
  accepted_ = accepted_ || result.accepted;
  if (terminal_.ok() && !result.status.ok())
    terminal_ = std::move(result.status);
}

detail::EnqueueResult
detail::QueueSequence::enqueue_wait(const Completion &source) noexcept {
  if (queue_ == nullptr) {
    return EnqueueResult::rejected(
        Status::FailedPrecondition("compute queue sequence is invalid"));
  }
  if (!source.valid()) {
    return EnqueueResult::rejected(
        Status::FailedPrecondition("source completion is invalid"));
  }
  CompletionDependency dependency = CompletionAccess::dependency(source);
  if (!dependency.pending()) {
    return dependency.terminal_hint == nullptr || dependency.terminal_hint->ok()
               ? EnqueueResult::no_op()
               : EnqueueResult::rejected(*dependency.terminal_hint);
  }
  detail::QueueState *state = detail::ContextAccess::state(*queue_);
  EnqueueResult result = state->context->backend_vtable->wait_completion(
      state->backend_state, dependency.vtable, dependency.state);
  if (result.accepted && result.status.ok() &&
      dependency.terminal_hint != nullptr && !dependency.terminal_hint->ok()) {
    result.status = *dependency.terminal_hint;
  }
  return result;
}

Result<Completion> detail::QueueSequence::finish() noexcept {
  if (queue_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compute queue sequence is invalid"));
  }
  Queue *queue = queue_;
  detail::QueueState *state = detail::ContextAccess::state(*queue);
  const bool accepted = accepted_;
  Status terminal = std::move(terminal_);
  Completion completion = state->context->backend_vtable->finish_sequence(
      state->backend_state, backend_sequence_, accepted, terminal);
  queue_ = nullptr;
  backend_sequence_ = nullptr;
  accepted_ = false;
  terminal_ = Status::Ok();
  queue->end_submission();
  if (!completion.valid()) {
    return std::unexpected(
        Status::Internal("compute backend returned an invalid completion"));
  }
  if (!accepted && !terminal.ok()) {
    return std::unexpected(std::move(terminal));
  }
  return completion;
}

void detail::QueueSequence::reset() noexcept {
  if (queue_ != nullptr) {
    if (accepted_ && terminal_.ok()) {
      terminal_ = Status::Internal(
          "accepted compute queue sequence was abandoned before finish");
    }
    auto completion = finish();
    // Destroying a valid asynchronous Completion is the backend's explicit
    // non-blocking abandonment path. Empty sequences merely release a cold
    // reservation. The destructor itself never waits.
    (void)completion;
  }
}

} // namespace hyle::compute
