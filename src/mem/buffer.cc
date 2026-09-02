#include "hyle/mem/buffer.h"

#include <utility>

namespace hyle::mem {
namespace {

bool valid_alignment(std::size_t alignment) noexcept {
  return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

}  // namespace

Buffer::Buffer(Buffer&& other) noexcept
    : resource_(other.resource_), native_handle_(other.native_handle_),
      size_(other.size_), alignment_(other.alignment_) {
  other.resource_ = {};
  other.native_handle_ = nullptr;
  other.size_ = 0;
  other.alignment_ = alignof(std::max_align_t);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    reset().IgnoreError();
    resource_ = other.resource_;
    native_handle_ = other.native_handle_;
    size_ = other.size_;
    alignment_ = other.alignment_;
    other.resource_ = {};
    other.native_handle_ = nullptr;
    other.size_ = 0;
    other.alignment_ = alignof(std::max_align_t);
  }
  return *this;
}

Buffer::~Buffer() { reset().IgnoreError(); }

Result<Buffer> Buffer::create(MemoryResourceRef resource, std::size_t size,
                              std::size_t alignment) noexcept {
  if (!resource.valid()) {
    return std::unexpected(
        Status::InvalidArgument("buffer resource is invalid"));
  }
  if (!valid_alignment(alignment)) {
    return std::unexpected(Status::InvalidArgument(
        "buffer alignment must be a non-zero power of two"));
  }
  const MemoryProperties properties = resource.properties();
  if (!valid_alignment(properties.minimum_alignment)) {
    return std::unexpected(Status::Internal(
        "memory resource reports an invalid minimum alignment"));
  }
  if (alignment < properties.minimum_alignment) {
    alignment = properties.minimum_alignment;
  }
  if (size > properties.maximum_allocation) {
    return std::unexpected(
        Status::OutOfRange("buffer exceeds resource maximum allocation"));
  }

  auto allocation = resource.resource().allocate(size, alignment);
  if (!allocation) return std::unexpected(std::move(allocation).error());
  if (allocation->size != size || allocation->alignment != alignment ||
      (size != 0 && allocation->native_handle == nullptr)) {
    resource.resource().deallocate(*allocation);
    return std::unexpected(Status::Internal(
        "memory resource returned an invalid allocation token"));
  }
  return Buffer(resource, *allocation);
}

Result<Buffer> Buffer::adopt(MemoryResourceRef resource,
                             ResourceAllocation allocation) noexcept {
  if (!resource.valid()) {
    return std::unexpected(
        Status::InvalidArgument("buffer resource is invalid"));
  }
  if (!valid_alignment(allocation.alignment)) {
    return std::unexpected(Status::InvalidArgument(
        "imported buffer alignment must be a non-zero power of two"));
  }
  if (allocation.size != 0 && allocation.native_handle == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "non-empty imported buffer has a null native handle"));
  }
  const MemoryProperties properties = resource.properties();
  if (!valid_alignment(properties.minimum_alignment)) {
    return std::unexpected(Status::Internal(
        "memory resource reports an invalid minimum alignment"));
  }
  if (allocation.alignment < properties.minimum_alignment) {
    return std::unexpected(Status::InvalidArgument(
        "imported buffer does not meet resource minimum alignment"));
  }
  if (allocation.size > properties.maximum_allocation) {
    return std::unexpected(Status::OutOfRange(
        "imported buffer exceeds resource maximum allocation"));
  }
  return Buffer(resource, allocation);
}

Status Buffer::reset() noexcept {
  if (resource_.valid()) {
    resource_.resource().deallocate(
        ResourceAllocation{native_handle_, size_, alignment_});
  }
  resource_ = {};
  native_handle_ = nullptr;
  size_ = 0;
  alignment_ = alignof(std::max_align_t);
  return Status::Ok();
}

HostMapping::HostMapping(HostMapping&& other) noexcept
    : resource_(other.resource_), native_handle_(other.native_handle_),
      token_(other.token_), data_(other.data_), offset_(other.offset_),
      size_(other.size_), access_(other.access_) {
  other.resource_ = {};
  other.native_handle_ = nullptr;
  other.token_ = nullptr;
  other.data_ = nullptr;
  other.offset_ = 0;
  other.size_ = 0;
  other.access_ = HostAccess::kRead;
}

HostMapping& HostMapping::operator=(HostMapping&& other) noexcept {
  if (this != &other) {
    close().IgnoreError();
    resource_ = other.resource_;
    native_handle_ = other.native_handle_;
    token_ = other.token_;
    data_ = other.data_;
    offset_ = other.offset_;
    size_ = other.size_;
    access_ = other.access_;
    other.resource_ = {};
    other.native_handle_ = nullptr;
    other.token_ = nullptr;
    other.data_ = nullptr;
    other.offset_ = 0;
    other.size_ = 0;
    other.access_ = HostAccess::kRead;
  }
  return *this;
}

HostMapping::~HostMapping() { close().IgnoreError(); }

Result<std::span<std::byte>> HostMapping::writable_bytes() noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("host mapping is invalid"));
  }
  if (access_ == HostAccess::kRead) {
    return std::unexpected(
        Status::PermissionDenied("host mapping is read-only"));
  }
  return std::span<std::byte>(static_cast<std::byte*>(data_), size_);
}

Status HostMapping::flush() noexcept {
  if (!valid()) return Status::FailedPrecondition("host mapping is invalid");
  if (access_ == HostAccess::kRead) {
    return Status::FailedPrecondition("read-only mapping cannot be flushed");
  }
  return resource_.resource().flush(native_handle_, offset_, size_);
}

Status HostMapping::invalidate() noexcept {
  if (!valid()) return Status::FailedPrecondition("host mapping is invalid");
  if (access_ == HostAccess::kWrite) {
    return Status::FailedPrecondition(
        "write-only mapping cannot be invalidated for reading");
  }
  return resource_.resource().invalidate(native_handle_, offset_, size_);
}

Status HostMapping::close() noexcept {
  if (!resource_.valid()) return Status::Ok();
  Status result = resource_.resource().unmap_host(
      native_handle_, token_, offset_, size_, access_);
  resource_ = {};
  native_handle_ = nullptr;
  token_ = nullptr;
  data_ = nullptr;
  offset_ = 0;
  size_ = 0;
  access_ = HostAccess::kRead;
  return result;
}

}  // namespace hyle::mem
