#include "hyle/compute/device.h"

#include <cstddef>
#include <new>
#include <utility>

#include "context_state.h"
#include "provider_internal.h"

namespace hyle::compute::detail {

void ContextState::destroy() noexcept {
  for (std::size_t index = provider_count; index != 0; --index) {
    ProviderSlot &slot = providers[index - 1];
    const ProviderVTable *table = ProviderAccess::vtable(slot.provider);
    if (table != nullptr && table->destroy_context != nullptr)
      table->destroy_context(slot.state, allocator);
  }
  const mem::AllocatorRef owner = allocator;
  this->~ContextState();
  owner.deallocate(this, sizeof(ContextState), alignof(ContextState));
}

} // namespace hyle::compute::detail

namespace hyle::compute {

std::string_view backend_name(BackendKind backend) noexcept {
  switch (backend) {
    case BackendKind::kCpu:
      return "cpu";
    case BackendKind::kCuda:
      return "cuda";
    case BackendKind::kMetal:
      return "metal";
    case BackendKind::kVulkan:
      return "vulkan";
    case BackendKind::kHip:
      return "hip";
    case BackendKind::kUnknown:
      break;
  }
  return "unknown";
}

Result<std::size_t> enumerate_devices(std::span<Device> output) noexcept {
  if (!output.empty()) output.front() = Device::cpu();
  return std::size_t{1};
}

Result<DeviceProperties> query_device(Device device) noexcept {
  if (!device.valid()) {
    return std::unexpected(Status::InvalidArgument("device is invalid"));
  }
  if (device != Device::cpu()) {
    return std::unexpected(Status::NotFound("device is not available"));
  }
  return DeviceProperties{
      .device = device,
      .name = "HYLE host CPU",
      .unified_memory = true,
      .asynchronous_queues = false,
  };
}

Context::Context(Context&& other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
  if (this != &other) {
    if (state_ != nullptr) {
      state_->open = false;
      detail::release(state_);
    }
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Context::~Context() {
  if (state_ != nullptr) {
    state_->open = false;
    detail::release(state_);
  }
}

Result<Context> Context::create(Device device, ContextOptions options) noexcept {
  if (!device.valid()) {
    return std::unexpected(Status::InvalidArgument("device is invalid"));
  }
  if (device != Device::cpu()) {
    return std::unexpected(
        Status::Unimplemented("device backend is not built"));
  }

  mem::AllocatorRef allocator = options.metadata_allocator;
  if (!allocator.valid()) allocator = mem::system_allocator();
  void* storage = allocator.allocate(sizeof(detail::ContextState),
                                     alignof(detail::ContextState));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("compute context allocation failed"));
  }
  auto* state = ::new (storage) detail::ContextState(device, allocator);
  for (const Provider &provider : options.providers.providers()) {
    const detail::ProviderVTable *table =
        detail::ProviderAccess::vtable(provider);
    if (table == nullptr || table->create_context == nullptr ||
        table->destroy_context == nullptr || table->prepare == nullptr) {
      detail::release(state);
      return std::unexpected(
          Status::InvalidArgument("compute provider is incomplete"));
    }
    void *provider_state = nullptr;
    Status status =
        table->create_context(provider, device, allocator, &provider_state);
    if (!status.ok()) {
      detail::release(state);
      return std::unexpected(std::move(status));
    }
    detail::ContextState::ProviderSlot &slot =
        state->providers[state->provider_count++];
    slot.provider = provider;
    slot.state = provider_state;
  }
  return Context(state);
}

Result<Context> Context::create(Device device, Provider provider) noexcept {
  ContextOptions options;
  if (!options.providers.add(provider)) {
    return std::unexpected(
        Status::InvalidArgument("compute provider is invalid"));
  }
  return create(device, options);
}

bool Context::is_open() const noexcept {
  return state_ != nullptr && state_->open;
}

Device Context::device() const noexcept {
  return state_ != nullptr ? state_->device : Device{};
}

mem::AllocatorRef Context::metadata_allocator() const noexcept {
  return state_ != nullptr ? state_->allocator : mem::AllocatorRef{};
}

Result<Queue> Context::create_queue(QueueOptions) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("context is invalid"));
  }
  if (!state_->open) {
    return std::unexpected(Status::FailedPrecondition("context is closed"));
  }
  detail::retain(state_);
  return Queue(state_);
}

Status Context::close() noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("context is invalid");
  }
  state_->open = false;
  return Status::Ok();
}

Queue::Queue(Queue&& other) noexcept
    : state_(other.state_), open_(other.open_) {
  other.state_ = nullptr;
  other.open_ = false;
}

Queue& Queue::operator=(Queue&& other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    open_ = other.open_;
    other.state_ = nullptr;
    other.open_ = false;
  }
  return *this;
}

Queue::~Queue() { detail::release(state_); }

bool Queue::is_open() const noexcept {
  return state_ != nullptr && open_ && state_->open;
}

Device Queue::device() const noexcept {
  return state_ != nullptr ? state_->device : Device{};
}

Status Queue::close() noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("queue is invalid");
  }
  open_ = false;
  return Status::Ok();
}

}  // namespace hyle::compute
