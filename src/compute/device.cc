#include "hyle/compute/device.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "backend_internal.h"
#include "context_state.h"
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
      table->destroy_queue == nullptr || table->drain_context == nullptr ||
      table->drain_queue == nullptr) {
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

Status Queue::begin_submission() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("queue is invalid");
  if (!state_->context->admission.try_enter())
    return Status::FailedPrecondition("context is closed");
  if (!state_->admission.try_enter()) {
    state_->context->admission.leave();
    return Status::FailedPrecondition("queue is closed");
  }
  return Status::Ok();
}

void Queue::end_submission() noexcept {
  state_->admission.leave();
  state_->context->admission.leave();
}

} // namespace hyle::compute
