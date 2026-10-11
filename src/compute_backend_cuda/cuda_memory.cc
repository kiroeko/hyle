#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/lease_access.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "cuda_internal.h"
#include "hyle/core/assert.h"

namespace hyle::compute::cuda::detail {

CudaResourceBase::CudaResourceBase(compute::detail::ContextState *owner,
                                   ContextState *context,
                                   MemoryKind kind) noexcept
    : owner_(owner), context_(context), kind_(kind) {
  compute::detail::retain(owner_);
}

CudaResourceBase::~CudaResourceBase() { compute::detail::release(owner_); }

mem::MemoryProperties CudaResourceBase::properties() const noexcept {
  mem::MemoryProperty flags = mem::MemoryProperty::kNone;
  switch (kind_) {
  case MemoryKind::kDevice:
    flags = mem::MemoryProperty::kDeviceAccessible;
    break;
  case MemoryKind::kPinnedHost:
    flags = mem::MemoryProperty::kHostAccessible |
            mem::MemoryProperty::kHostPinned;
    break;
  case MemoryKind::kManaged:
    flags = mem::MemoryProperty::kHostAccessible |
            mem::MemoryProperty::kDeviceAccessible |
            mem::MemoryProperty::kManaged;
    break;
  }
  const auto domain_value =
      (static_cast<std::uintptr_t>(static_cast<unsigned>(context_->ordinal))
       << 8) |
      (static_cast<std::uintptr_t>(kind_) + 1);
  return mem::MemoryProperties{
      .flags = flags,
      .minimum_alignment = kCudaAllocationAlignment,
      .maximum_allocation = SIZE_MAX,
      .domain = mem::MemoryDomainId{&kMemoryDomainAuthority, domain_value},
  };
}

Result<mem::ResourceHostMapping> CudaResourceBase::map_host(
    void *native_handle, std::size_t offset, std::size_t size,
    mem::HostAccess) noexcept {
  if (kind_ == MemoryKind::kDevice) {
    return std::unexpected(
        Status::Unimplemented("CUDA device memory is not host-mappable"));
  }
  void *data = size == 0
                   ? nullptr
                   : static_cast<void *>(
                         static_cast<std::byte *>(native_handle) + offset);
  return mem::ResourceHostMapping{.data = data, .token = nullptr};
}

Status CudaResourceBase::unmap_host(void *, void *, std::size_t, std::size_t,
                                    mem::HostAccess) noexcept {
  return Status::Ok();
}

Status CudaResourceBase::flush(void *, std::size_t, std::size_t) noexcept {
  return kind_ == MemoryKind::kDevice
             ? Status::Unimplemented("CUDA device memory is not host-mappable")
             : Status::Ok();
}

Status CudaResourceBase::invalidate(void *, std::size_t,
                                    std::size_t) noexcept {
  return kind_ == MemoryKind::kDevice
             ? Status::Unimplemented("CUDA device memory is not host-mappable")
             : Status::Ok();
}

namespace {

CUdeviceptr device_address(void *pointer, std::size_t offset) noexcept {
  return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer) +
                                  offset);
}

std::byte *host_address(void *pointer, std::size_t offset) noexcept {
  return static_cast<std::byte *>(pointer) + offset;
}

Status finish_default_transfer(CUresult submitted) noexcept {
  // Driver "synchronous" H2D/D2D/memset may return before device completion.
  // Pair with the same TU's default-stream mode, not a Context/device barrier.
  // Also observe the prefix when a native call reports an asynchronous error.
  Status result = driver_status(submitted);
  result.Update(driver_status(cuStreamSynchronize(nullptr)));
  return result;
}

} // namespace

Status CudaResourceBase::read(void *native_handle, std::size_t offset,
                              std::span<std::byte> destination) noexcept {
  if (destination.empty())
    return Status::Ok();
  if (kind_ != MemoryKind::kDevice) {
    std::memmove(destination.data(), host_address(native_handle, offset),
                 destination.size());
    return Status::Ok();
  }
  ScopedPrimaryContext scope(context_->primary);
  if (!scope)
    return scope.status();
  return driver_status(cuMemcpyDtoH(destination.data(),
                                    device_address(native_handle, offset),
                                    destination.size()));
}

Status CudaResourceBase::write(void *native_handle, std::size_t offset,
                               std::span<const std::byte> source) noexcept {
  if (source.empty())
    return Status::Ok();
  if (kind_ != MemoryKind::kDevice) {
    std::memmove(host_address(native_handle, offset), source.data(),
                 source.size());
    return Status::Ok();
  }
  ScopedPrimaryContext scope(context_->primary);
  if (!scope)
    return scope.status();
  return finish_default_transfer(
      cuMemcpyHtoD(device_address(native_handle, offset), source.data(),
                   source.size()));
}

Status CudaResourceBase::fill(void *native_handle, std::size_t offset,
                              std::size_t size, std::byte value) noexcept {
  if (size == 0)
    return Status::Ok();
  if (kind_ != MemoryKind::kDevice) {
    std::memset(host_address(native_handle, offset),
                std::to_integer<unsigned char>(value), size);
    return Status::Ok();
  }
  ScopedPrimaryContext scope(context_->primary);
  if (!scope)
    return scope.status();
  return finish_default_transfer(
      cuMemsetD8(device_address(native_handle, offset),
                 std::to_integer<unsigned char>(value), size));
}

Status CudaResourceBase::copy(void *destination_handle,
                              std::size_t destination_offset,
                              mem::MemoryResource &source_resource,
                              void *source_handle,
                              std::size_t source_offset,
                              std::size_t size) noexcept {
  if (size == 0)
    return Status::Ok();
  const mem::MemoryProperties source_properties = source_resource.properties();
  const CudaResourceBase *source_cuda = nullptr;
  if (source_properties.domain.authority == &kMemoryDomainAuthority)
    source_cuda = static_cast<const CudaResourceBase *>(&source_resource);

  if (source_cuda != nullptr) {
    if (source_cuda->context() != context_) {
      return Status::Unimplemented(
          "synchronous CUDA copy across contexts is unsupported");
    }
    ScopedPrimaryContext scope(context_->primary);
    if (!scope)
      return scope.status();
    return finish_default_transfer(cuMemcpy(
        device_address(destination_handle, destination_offset),
        device_address(source_handle, source_offset), size));
  }

  if (kind_ != MemoryKind::kDevice) {
    return mem::MemoryResource::copy(
        destination_handle, destination_offset, source_resource,
        source_handle, source_offset, size);
  }

  HYLE_RETURN_IF_ERROR(
      source_resource.invalidate(source_handle, source_offset, size));
  auto source = source_resource.map_host(source_handle, source_offset, size,
                                         mem::HostAccess::kRead);
  if (!source)
    return std::move(source).error();
  if (source->data == nullptr) {
    source_resource
        .unmap_host(source_handle, source->token, source_offset, size,
                    mem::HostAccess::kRead)
        .IgnoreError();
    return Status::Internal("source resource returned a null host mapping");
  }
  ScopedPrimaryContext scope(context_->primary);
  Status result = scope
                      ? finish_default_transfer(cuMemcpyHtoD(
                            device_address(destination_handle,
                                           destination_offset),
                            source->data, size))
                      : scope.status();
  result.Update(source_resource.unmap_host(
      source_handle, source->token, source_offset, size,
      mem::HostAccess::kRead));
  return result;
}

CudaResourceBase *resource(mem::MemoryResourceRef value) noexcept {
  if (!value.valid() ||
      value.properties().domain.authority != &kMemoryDomainAuthority)
    return nullptr;
  return static_cast<CudaResourceBase *>(&value.resource());
}

const CudaResourceBase *resource_const(mem::MemoryResourceRef value) noexcept {
  return resource(value);
}

void *byte_pointer(mem::BufferView view) noexcept {
  return reinterpret_cast<void *>(
      reinterpret_cast<std::uintptr_t>(view.native_handle()) + view.offset());
}

const void *byte_pointer(mem::ConstBufferView view) noexcept {
  return reinterpret_cast<const void *>(
      reinterpret_cast<std::uintptr_t>(view.native_handle()) + view.offset());
}

namespace {

class ResourceImpl final : public CudaResourceBase {
public:
  ResourceImpl(compute::detail::ContextState *owner, ContextState *context,
               MemoryKind kind, mem::AllocatorRef allocator) noexcept
      : CudaResourceBase(owner, context, kind), allocator_(allocator) {}

  ~ResourceImpl() override { HYLE_FATAL_ASSERT(live_.load() == 0); }

  std::string_view name() const noexcept override {
    switch (kind_) {
    case MemoryKind::kDevice:
      return "hyle.cuda.device";
    case MemoryKind::kPinnedHost:
      return "hyle.cuda.pinned_host";
    case MemoryKind::kManaged:
      return "hyle.cuda.managed";
    }
    return "hyle.cuda.unknown";
  }

  Result<mem::ResourceAllocation>
  allocate(std::size_t size, std::size_t alignment) noexcept override {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        alignment > kCudaAllocationAlignment) {
      return std::unexpected(Status::InvalidArgument(
          "CUDA allocation alignment must be a power of two no greater than 256"));
    }
    if (size == 0)
      return mem::ResourceAllocation{nullptr, 0, alignment};

    ScopedPrimaryContext scope(context_->primary);
    if (!scope)
      return std::unexpected(scope.status());
    void *pointer = nullptr;
    Status result = Status::Ok();
    if (kind_ == MemoryKind::kPinnedHost) {
      result = driver_status(cuMemHostAlloc(&pointer, size,
                                            CU_MEMHOSTALLOC_PORTABLE));
    } else {
      CUdeviceptr device_pointer = 0;
      const CUresult allocated =
          kind_ == MemoryKind::kManaged
              ? cuMemAllocManaged(&device_pointer, size, CU_MEM_ATTACH_GLOBAL)
              : cuMemAlloc(&device_pointer, size);
      result = driver_status(allocated);
      pointer = reinterpret_cast<void *>(
          static_cast<std::uintptr_t>(device_pointer));
    }
    if (!result.ok())
      return std::unexpected(std::move(result));

    live_.fetch_add(1, std::memory_order_relaxed);
    const std::size_t now =
        used_.fetch_add(size, std::memory_order_relaxed) + size;
    std::size_t peak = peak_.load(std::memory_order_relaxed);
    while (peak < now && !peak_.compare_exchange_weak(
                             peak, now, std::memory_order_relaxed,
                             std::memory_order_relaxed)) {
    }
    return mem::ResourceAllocation{pointer, size, alignment};
  }

  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    if (allocation.native_handle == nullptr) {
      HYLE_FATAL_ASSERT(allocation.size == 0);
      return;
    }
    ScopedPrimaryContext scope(context_->primary);
    HYLE_FATAL_ASSERT(scope.status().ok());
    const CUresult released =
        kind_ == MemoryKind::kPinnedHost
            ? cuMemFreeHost(allocation.native_handle)
            : cuMemFree(static_cast<CUdeviceptr>(
                  reinterpret_cast<std::uintptr_t>(allocation.native_handle)));
    HYLE_FATAL_ASSERT(released == CUDA_SUCCESS);
    used_.fetch_sub(allocation.size, std::memory_order_relaxed);
    live_.fetch_sub(1, std::memory_order_relaxed);
  }

  Result<mem::ResourceStats> stats() const noexcept override {
    const std::size_t used = used_.load(std::memory_order_relaxed);
    return mem::ResourceStats{
        .capacity_bytes = SIZE_MAX,
        .used_bytes = used,
        .reserved_bytes = used,
        .peak_used_bytes = peak_.load(std::memory_order_relaxed),
    };
  }

  Status trim() noexcept override {
    return Status::Unimplemented(
        "CUDA direct allocation resource has no retained pool to trim");
  }

  std::size_t live() const noexcept {
    return live_.load(std::memory_order_acquire);
  }

  void destroy_owner() noexcept {
    HYLE_FATAL_ASSERT(live() == 0);
    const mem::AllocatorRef allocator = allocator_;
    this->~ResourceImpl();
    allocator.deallocate(this, sizeof(ResourceImpl), alignof(ResourceImpl));
  }

private:
  mem::AllocatorRef allocator_{};
  std::atomic<std::size_t> live_{0};
  std::atomic<std::size_t> used_{0};
  std::atomic<std::size_t> peak_{0};
};

class ExternalResourceImpl final : public CudaResourceBase {
public:
  ExternalResourceImpl(compute::detail::ContextState *owner,
                       ContextState *context, void *expected) noexcept
      : CudaResourceBase(owner, context, MemoryKind::kDevice),
        expected_(expected) {}

  // Called only after Buffer::adopt has accepted the complete allocation.
  // This avoids destroying/reconstructing a resource object while a
  // MemoryResourceRef already points at it, and preserves the rule that a
  // failed wrapper construction never steals the caller's allocation.
  void activate_ownership() noexcept { owning_ = true; }
  bool owns_payload() const noexcept { return owning_; }

  std::string_view name() const noexcept override {
    return owning_ ? "hyle.cuda.external_adopted"
                   : "hyle.cuda.external_borrowed";
  }
  Result<mem::ResourceAllocation>
  allocate(std::size_t, std::size_t) noexcept override {
    return std::unexpected(Status::Unimplemented(
        "external CUDA storage cannot allocate another buffer"));
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    HYLE_FATAL_ASSERT(allocation.native_handle == expected_);
    if (!owning_ || allocation.native_handle == nullptr)
      return;
    ScopedPrimaryContext scope(context_->primary);
    HYLE_FATAL_ASSERT(scope.status().ok());
    const CUresult released = cuMemFree(static_cast<CUdeviceptr>(
        reinterpret_cast<std::uintptr_t>(allocation.native_handle)));
    HYLE_FATAL_ASSERT(released == CUDA_SUCCESS);
    expected_ = nullptr;
  }

private:
  bool owning_ = false;
  void *expected_ = nullptr;
};

struct ExternalBufferState final {
  ExternalBufferState(compute::detail::ContextState *owner,
                      ContextState *context, void *pointer) noexcept
      : allocator(owner->allocator), resource(owner, context, pointer) {}

  void destroy() noexcept {
    const mem::AllocatorRef selected = allocator;
    this->~ExternalBufferState();
    selected.deallocate(this, sizeof(ExternalBufferState),
                        alignof(ExternalBufferState));
  }

  mem::AllocatorRef allocator{};
  ExternalResourceImpl resource;
  mem::Buffer buffer{};
};

} // namespace

Result<ExternalBuffer> make_external(const Context &context, void *pointer,
                                     std::size_t size, bool owning) noexcept {
  if (size != 0 && pointer == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "non-empty external CUDA buffer has a null pointer"));
  }
  compute::detail::ContextState *owner =
      compute::detail::ContextAccess::state(context);
  if (owner == nullptr || !context.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("CUDA context is invalid or closed"));
  }
  auto ordinal = device_ordinal(context.device());
  if (!ordinal)
    return std::unexpected(std::move(ordinal).error());
  auto *cuda_context = static_cast<ContextState *>(owner->backend_state);

  if (size != 0) {
    ScopedPrimaryContext scope(cuda_context->primary);
    if (!scope)
      return std::unexpected(scope.status());
    const CUdeviceptr address = static_cast<CUdeviceptr>(
        reinterpret_cast<std::uintptr_t>(pointer));
    int pointer_device = -1;
    CUmemorytype memory_type = CU_MEMORYTYPE_HOST;
    CUcontext pointer_context = nullptr;
    HYLE_RETURN_IF_ERROR(driver_status(cuPointerGetAttribute(
        &pointer_device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address)));
    HYLE_RETURN_IF_ERROR(driver_status(cuPointerGetAttribute(
        &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address)));
    HYLE_RETURN_IF_ERROR(driver_status(cuPointerGetAttribute(
        &pointer_context, CU_POINTER_ATTRIBUTE_CONTEXT, address)));
    if (pointer_device != *ordinal || pointer_context != cuda_context->primary) {
      return std::unexpected(Status::InvalidArgument(
          "external CUDA pointer belongs to a different device or context"));
    }
    if (memory_type != CU_MEMORYTYPE_DEVICE &&
        memory_type != CU_MEMORYTYPE_UNIFIED) {
      return std::unexpected(Status::InvalidArgument(
          "external CUDA pointer must be device or managed storage"));
    }
  }

  void *storage = owner->allocator.allocate(sizeof(ExternalBufferState),
                                            alignof(ExternalBufferState));
  if (storage == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "external CUDA buffer-state allocation failed"));
  }
  // Ownership is activated only after Buffer::adopt accepts the complete
  // resource token, so a validation/allocation failure never steals input.
  auto *state =
      ::new (storage) ExternalBufferState(owner, cuda_context, pointer);
  auto buffer = mem::Buffer::adopt(
      mem::MemoryResourceRef(state->resource),
      mem::ResourceAllocation{pointer, size, kCudaAllocationAlignment});
  if (!buffer) {
    state->destroy();
    return std::unexpected(std::move(buffer).error());
  }
  if (owning)
    state->resource.activate_ownership();
  state->buffer = std::move(*buffer);
  return ExternalBuffer(state);
}

} // namespace hyle::compute::cuda::detail

namespace hyle::compute::cuda {

MemoryResource::MemoryResource(MemoryResource &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

MemoryResource &MemoryResource::operator=(MemoryResource &&other) noexcept {
  if (this != &other) {
    reset();
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

MemoryResource::~MemoryResource() { reset(); }

Result<MemoryResource> MemoryResource::create(const Context &context,
                                              MemoryKind kind) noexcept {
  compute::detail::ContextState *owner =
      compute::detail::ContextAccess::state(context);
  if (owner == nullptr || !context.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("CUDA context is invalid or closed"));
  }
  auto ordinal = device_ordinal(context.device());
  if (!ordinal)
    return std::unexpected(std::move(ordinal).error());
  if (kind != MemoryKind::kDevice && kind != MemoryKind::kPinnedHost &&
      kind != MemoryKind::kManaged) {
    return std::unexpected(
        Status::InvalidArgument("CUDA memory kind is invalid"));
  }
  void *storage = owner->allocator.allocate(sizeof(detail::ResourceImpl),
                                            alignof(detail::ResourceImpl));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("CUDA memory-resource allocation failed"));
  }
  auto *state = ::new (storage) detail::ResourceImpl(
      owner, static_cast<detail::ContextState *>(owner->backend_state), kind,
      owner->allocator);
  return MemoryResource(state);
}

mem::MemoryResourceRef MemoryResource::ref() const noexcept {
  return state_ != nullptr
             ? mem::MemoryResourceRef(
                   *static_cast<detail::ResourceImpl *>(state_))
             : mem::MemoryResourceRef{};
}

MemoryKind MemoryResource::kind() const noexcept {
  return state_ != nullptr
             ? static_cast<detail::ResourceImpl *>(state_)->kind()
             : MemoryKind::kDevice;
}

Device MemoryResource::device() const noexcept {
  return state_ != nullptr
             ? static_cast<detail::ResourceImpl *>(state_)->device()
             : Device{};
}

Status MemoryResource::trim() noexcept {
  return state_ != nullptr
             ? static_cast<detail::ResourceImpl *>(state_)->trim()
             : Status::FailedPrecondition("CUDA memory resource is invalid");
}

void MemoryResource::reset() noexcept {
  if (state_ != nullptr) {
    static_cast<detail::ResourceImpl *>(state_)->destroy_owner();
    state_ = nullptr;
  }
}

namespace detail {

Result<void *> mutable_pointer(mem::BufferView view, std::size_t alignment,
                               std::size_t element_size) noexcept {
  if (!view.valid())
    return std::unexpected(Status::InvalidArgument("CUDA buffer view is invalid"));
  if (resource(view.resource()) == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("buffer view is not CUDA storage"));
  }
  if (element_size == 0 || view.size() < element_size) {
    return std::unexpected(
        Status::OutOfRange("CUDA buffer view is smaller than the requested type"));
  }
  void *pointer = byte_pointer(view);
  if (alignment == 0 ||
      (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) != 0) {
    return std::unexpected(Status::InvalidArgument(
        "CUDA buffer offset does not satisfy the requested type alignment"));
  }
  return pointer;
}

Result<const void *> const_pointer(mem::ConstBufferView view,
                                   std::size_t alignment,
                                   std::size_t element_size) noexcept {
  if (!view.valid())
    return std::unexpected(Status::InvalidArgument("CUDA buffer view is invalid"));
  if (resource_const(view.resource()) == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("buffer view is not CUDA storage"));
  }
  if (element_size == 0 || view.size() < element_size) {
    return std::unexpected(
        Status::OutOfRange("CUDA buffer view is smaller than the requested type"));
  }
  const void *pointer = byte_pointer(view);
  if (alignment == 0 ||
      (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) != 0) {
    return std::unexpected(Status::InvalidArgument(
        "CUDA buffer offset does not satisfy the requested type alignment"));
  }
  return pointer;
}

} // namespace detail

ExternalBuffer::ExternalBuffer(ExternalBuffer &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

ExternalBuffer &ExternalBuffer::operator=(ExternalBuffer &&other) noexcept {
  if (this != &other) {
    reset();
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

ExternalBuffer::~ExternalBuffer() { reset(); }

Result<ExternalBuffer> ExternalBuffer::borrow(const Context &context,
                                              void *pointer,
                                              std::size_t size) noexcept {
  return detail::make_external(context, pointer, size, false);
}

Result<ExternalBuffer> ExternalBuffer::adopt(const Context &context,
                                             void *pointer,
                                             std::size_t size) noexcept {
  return detail::make_external(context, pointer, size, true);
}

mem::BufferView ExternalBuffer::view() noexcept {
  return state_ != nullptr
             ? static_cast<detail::ExternalBufferState *>(state_)->buffer.view()
             : mem::BufferView{};
}

mem::ConstBufferView ExternalBuffer::view() const noexcept {
  return state_ != nullptr
             ? static_cast<const detail::ExternalBufferState *>(state_)
                   ->buffer.view()
             : mem::ConstBufferView{};
}

std::size_t ExternalBuffer::size() const noexcept {
  return state_ != nullptr
             ? static_cast<const detail::ExternalBufferState *>(state_)
                   ->buffer.size()
             : 0;
}

Device ExternalBuffer::device() const noexcept {
  return state_ != nullptr
             ? static_cast<const detail::ExternalBufferState *>(state_)
                   ->resource.device()
             : Device{};
}

void ExternalBuffer::reset() noexcept {
  if (state_ != nullptr) {
    static_cast<detail::ExternalBufferState *>(state_)->destroy();
    state_ = nullptr;
  }
}

Result<ReadLease> ExternalBuffer::into_read_lease(LeaseImportOptions options) && noexcept {
  auto *state = static_cast<detail::ExternalBufferState *>(state_);
  if (!state || !state->resource.owns_payload())
    return std::unexpected(Status::FailedPrecondition("CUDA lease requires an adopted native payload"));
  HYLE_ASSIGN_OR_RETURN(auto lease, ReadLease::adopt(std::move(state->buffer), options));
  compute::detail::LeaseAccess::retain_resource_owner(lease, state,
      [](void *owner) noexcept { static_cast<detail::ExternalBufferState *>(owner)->destroy(); });
  state_ = nullptr;
  return lease;
}

Result<LeasePool> ExternalBuffer::into_lease_pool(LeasePoolOptions options) && noexcept {
  auto *state = static_cast<detail::ExternalBufferState *>(state_);
  if (!state || !state->resource.owns_payload())
    return std::unexpected(Status::FailedPrecondition("CUDA lease pool requires an adopted native payload"));
  HYLE_ASSIGN_OR_RETURN(auto pool, LeasePool::adopt(std::move(state->buffer), options));
  compute::detail::LeaseAccess::retain_resource_owner(pool, state,
      [](void *owner) noexcept { static_cast<detail::ExternalBufferState *>(owner)->destroy(); });
  state_ = nullptr;
  return pool;
}

} // namespace hyle::compute::cuda
