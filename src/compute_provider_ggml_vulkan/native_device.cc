#include "native_device.h"

#include <new>
#include <utility>

#include "hyle/platform/dynlib.h"

namespace hyle::compute::ggml_vulkan::detail {
namespace {
Status failure(bridge::Failure error) noexcept {
  switch (error.error) {
  case bridge::Error::None:
    return Status::Ok();
  case bridge::Error::InvalidArgument:
    return Status::InvalidArgument("ggml Vulkan resident region");
  case bridge::Error::InvalidState:
    return Status::FailedPrecondition("ggml Vulkan region slot is busy or failed");
  case bridge::Error::OutOfRange:
    return Status::OutOfRange("ggml Vulkan physical device index");
  case bridge::Error::Exhausted:
    return Status::ResourceExhausted("ggml Vulkan native resources");
  case bridge::Error::Unavailable:
    return Status::Unavailable("ggml Vulkan device or loader capability");
  case bridge::Error::Native: {
    auto code = StatusCode::Unknown;
    if (error.native == VK_ERROR_OUT_OF_HOST_MEMORY ||
        error.native == VK_ERROR_OUT_OF_DEVICE_MEMORY)
      code = StatusCode::ResourceExhausted;
    else if (error.native == VK_ERROR_DEVICE_LOST)
      code = StatusCode::DataLoss;
    else if (error.native == VK_ERROR_LAYER_NOT_PRESENT ||
             error.native == VK_ERROR_EXTENSION_NOT_PRESENT ||
             error.native == VK_ERROR_FEATURE_NOT_PRESENT ||
             error.native == VK_ERROR_INITIALIZATION_FAILED)
      code = StatusCode::Unavailable;
    return Status::FromNative(code, {NativeErrorDomain::Vulkan, error.native},
                              "ggml Vulkan native operation");
  }
  case bridge::Error::Internal:
    return Status::Internal("ggml Vulkan adapter exception");
  }
  return Status::Internal("ggml Vulkan failure code");
}

bool reserve(void *owner, std::size_t bytes, void **token) noexcept {
  auto charge = static_cast<mem::QuotaAllocator *>(owner)->try_reserve_bytes(bytes);
  if (!charge)
    return false;
  *token = new (std::nothrow) mem::QuotaAllocator::Reservation(std::move(*charge));
  return *token != nullptr;
}
void release(void *token) noexcept {
  delete static_cast<mem::QuotaAllocator::Reservation *>(token);
}
const char *default_loader() noexcept {
#if defined(_WIN32)
  return "vulkan-1.dll";
#elif defined(__APPLE__)
  return "libvulkan.1.dylib";
#else
  return "libvulkan.so.1";
#endif
}
} // namespace

struct NativeDevice::State final {
  platform::DynLib loader;
  bridge::Runtime *runtime = nullptr;
  bridge::Native native;
  ~State() { bridge::destroy(runtime); }
};

Result<NativeDevice> NativeDevice::create(NativeOptions options) noexcept {
  try {
    auto state = std::make_shared<State>();
    HYLE_ASSIGN_OR_RETURN(state->loader, platform::DynLib::open(
        options.loader_path ? options.loader_path : default_loader()));
    HYLE_ASSIGN_OR_RETURN(auto get, state->loader.symbol_as<PFN_vkGetInstanceProcAddr>(
        "vkGetInstanceProcAddr"));
    bridge::Options configuration{get, options.device_index,
        options.allow_software, options.validation, options.quota,
        options.quota ? reserve : nullptr, options.quota ? release : nullptr};
    HYLE_RETURN_IF_ERROR(failure(bridge::create(
        configuration, &state->runtime, &state->native)));
    NativeDevice result;
    result.state_ = std::move(state);
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("ggml Vulkan owner"));
  }
}

const bridge::Native &NativeDevice::native() const noexcept {
  static const bridge::Native empty;
  return state_ ? state_->native : empty;
}
NativeDevice::Scope NativeDevice::activate() const noexcept {
  return Scope(bridge::enter(state_ ? state_->runtime : nullptr));
}
NativeDevice::Scope::~Scope() { bridge::leave(previous_); }

Result<ggml_backend *> NativeDevice::create_backend() const noexcept {
  if (!state_)
    return std::unexpected(Status::InvalidArgument("empty ggml Vulkan device"));
  ggml_backend *result = nullptr;
  HYLE_RETURN_IF_ERROR(failure(bridge::create_backend(state_->runtime, &result)));
  return result;
}
Result<NativeWork> NativeDevice::create_work() const noexcept {
  if (!state_)
    return std::unexpected(Status::InvalidArgument("empty ggml Vulkan device"));
  NativeWork result;
  result.device_ = *this;
  HYLE_RETURN_IF_ERROR(failure(bridge::create_work(state_->runtime, &result.work_)));
  return result;
}
Result<ggml_backend_buffer *> NativeDevice::borrow(
    VkBuffer handle, std::size_t bytes,
    std::shared_ptr<const void> owner) const noexcept {
  if (!state_ || !handle || !bytes || !owner)
    return std::unexpected(Status::InvalidArgument("ggml Vulkan buffer owner"));
  ggml_backend_buffer *result = nullptr;
  HYLE_RETURN_IF_ERROR(failure(bridge::borrow_buffer(
      state_->runtime, handle, bytes, std::move(owner), &result)));
  return result;
}
Result<ggml_backend_buffer *>
NativeDevice::allocate_context(ggml_context *context) const noexcept {
  if (!state_ || !context)
    return std::unexpected(Status::InvalidArgument("ggml Vulkan tensor context"));
  ggml_backend_buffer *result = nullptr;
  HYLE_RETURN_IF_ERROR(failure(bridge::allocate_context(state_->runtime, context, &result)));
  return result;
}
Result<ggml_backend_buffer *>
NativeDevice::allocate_buffer(std::size_t bytes) const noexcept {
  if (!state_ || !bytes)
    return std::unexpected(Status::InvalidArgument("ggml Vulkan buffer size"));
  ggml_backend_buffer *result = nullptr;
  HYLE_RETURN_IF_ERROR(failure(bridge::allocate_buffer(state_->runtime, bytes, &result)));
  return result;
}
VkBuffer NativeDevice::buffer_handle(ggml_backend_buffer *buffer) const noexcept {
  return state_ ? bridge::buffer_handle(buffer) : VK_NULL_HANDLE;
}

Status NativeDevice::with_queue(
    const std::function<Status(VkQueue)> &function) const noexcept {
  if (!state_ || !function)
    return Status::InvalidArgument("ggml Vulkan native queue callback");
  struct Call {
    const std::function<Status(VkQueue)> &function;
    Status result;
  } call{function, Status::Ok()};
  const auto result = bridge::with_queue(state_->runtime,
      [](void *opaque, VkQueue queue) -> VkResult {
        auto &call = *static_cast<Call *>(opaque);
        call.result = call.function(queue);
        return VK_SUCCESS;
      }, &call);
  HYLE_RETURN_IF_ERROR(failure(result));
  return call.result;
}

NativeWork::NativeWork(NativeWork &&other) noexcept
    : device_(std::move(other.device_)), work_(std::exchange(other.work_, nullptr)),
      lease_(std::move(other.lease_)), pending_(std::exchange(other.pending_, false)) {}
NativeWork &NativeWork::operator=(NativeWork &&other) noexcept {
  if (this != &other) {
    reset();
    device_ = std::move(other.device_);
    work_ = std::exchange(other.work_, nullptr);
    lease_ = std::move(other.lease_);
    pending_ = std::exchange(other.pending_, false);
  }
  return *this;
}
NativeWork::~NativeWork() { reset(); }
void NativeWork::reset() noexcept {
  // The bridge proves retirement before freeing backend commands; the lease
  // and loader remain alive through its exceptional blocking drain.
  bridge::destroy_work(std::exchange(work_, nullptr));
  lease_.reset();
  pending_ = false;
  device_ = {};
}
WorkState NativeWork::submit(
    ggml_cgraph *graph, std::shared_ptr<const void> resource_lease) noexcept {
  if (!work_ || !graph || !resource_lease)
    return {Status::InvalidArgument("ggml Vulkan graph resource lease"), pending_};
  if (pending_)
    return {Status::FailedPrecondition("ggml Vulkan region is pending"), true};
  lease_ = std::move(resource_lease);
  const auto result = bridge::submit_work(work_, graph);
  pending_ = result.pending;
  if (!pending_)
    lease_.reset();
  return {failure(result.failure), result.pending, result.device_lost};
}
WorkState NativeWork::observe(bool wait) noexcept {
  if (!work_)
    return {Status::InvalidArgument("empty ggml Vulkan region slot")};
  const auto result = bridge::observe_work(work_, wait);
  pending_ = result.pending;
  if (!pending_)
    lease_.reset();
  return {failure(result.failure), result.pending, result.device_lost};
}
WorkState NativeWork::poll() noexcept { return observe(false); }
WorkState NativeWork::wait() noexcept { return observe(true); }
} // namespace hyle::compute::ggml_vulkan::detail
