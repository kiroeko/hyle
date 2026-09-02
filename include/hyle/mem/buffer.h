#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/mem/resource.h"
#include "hyle/mem/visibility.h"

namespace hyle::mem {

class BufferView;
class ConstBufferView;
class HostMapping;
class FileMapping;
class SharedMapping;

// One provider allocation. Buffer is move-only and keeps its complete wrapper
// state inline: no HYLE pimpl/control-block allocation. The provider and its
// backend state must outlive this owner and every in-flight operation.
class Buffer {
 public:
  constexpr Buffer() noexcept = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  HYLE_MEM_API Buffer(Buffer&& other) noexcept;
  HYLE_MEM_API Buffer& operator=(Buffer&& other) noexcept;
  HYLE_MEM_API ~Buffer();

  HYLE_MEM_API static Result<Buffer> create(
      MemoryResourceRef resource, std::size_t size,
      std::size_t alignment = alignof(std::max_align_t)) noexcept;

  // Provider-author seam for imported/external storage. The allocation must
  // already belong to `resource`; Buffer assumes ownership only on success and
  // will return the exact token to that provider. Typed Compute/IO adapters,
  // not ordinary callers, normally create ResourceAllocation values.
  HYLE_MEM_API static Result<Buffer> adopt(
      MemoryResourceRef resource, ResourceAllocation allocation) noexcept;

  constexpr bool valid() const noexcept { return resource_.valid(); }
  constexpr explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return size_; }
  std::size_t alignment() const noexcept { return alignment_; }
  MemoryResourceRef resource() const noexcept { return resource_; }
  MemoryProperties properties() const noexcept {
    return resource_.properties();
  }

  // Opaque provider handle. Only a typed adapter that has checked resource and
  // domain identity may interpret it; it is never a generally dereferenceable
  // pointer and callers must not perform pointer arithmetic on it.
  void* native_handle() const noexcept { return native_handle_; }

  BufferView view() noexcept;
  ConstBufferView view() const noexcept;

  Status write(std::size_t offset,
               std::span<const std::byte> source) noexcept;
  Status read(std::size_t offset,
              std::span<std::byte> destination) const noexcept;
  Status fill(std::size_t offset, std::size_t size,
              std::byte value) noexcept;
  HYLE_MEM_API Status reset() noexcept;

 private:
  Buffer(MemoryResourceRef resource, ResourceAllocation allocation) noexcept
      : resource_(resource), native_handle_(allocation.native_handle),
        size_(allocation.size), alignment_(allocation.alignment) {}

  MemoryResourceRef resource_{};
  void* native_handle_ = nullptr;
  std::size_t size_ = 0;
  std::size_t alignment_ = alignof(std::max_align_t);
};

// Mutable non-owning logical range. Exactly four words: provider identity,
// opaque allocation handle, byte offset and byte length. Owner reset/destruction
// invalidates the view. Moving the owner does not: the provider allocation token
// remains stable.
class BufferView {
 public:
  constexpr BufferView() noexcept = default;

  constexpr bool valid() const noexcept { return resource_.valid(); }
  constexpr explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  std::size_t offset() const noexcept { return offset_; }
  MemoryResourceRef resource() const noexcept { return resource_; }
  void* native_handle() const noexcept { return native_handle_; }

  Result<BufferView> subview(
      std::size_t offset,
      std::size_t size = std::dynamic_extent) const noexcept;
  Status write(std::span<const std::byte> source) const noexcept;
  Status read(std::span<std::byte> destination) const noexcept;
  Status fill(std::byte value) const noexcept;
  Status copy_from(ConstBufferView source) const noexcept;
  Result<HostMapping> map_host(
      HostAccess access = HostAccess::kReadWrite) const noexcept;
  Status flush() const noexcept;
  Status invalidate() const noexcept;

 private:
  friend class Buffer;
  friend class ConstBufferView;
  friend class FileMapping;
  friend class SharedMapping;
  constexpr BufferView(MemoryResourceRef resource, void* native_handle,
                       std::size_t offset, std::size_t size) noexcept
      : resource_(resource), native_handle_(native_handle), offset_(offset),
        size_(size) {}

  MemoryResourceRef resource_{};
  void* native_handle_ = nullptr;
  std::size_t offset_ = 0;
  std::size_t size_ = 0;
};

class ConstBufferView {
 public:
  constexpr ConstBufferView() noexcept = default;
  constexpr ConstBufferView(BufferView view) noexcept
      : resource_(view.resource_), native_handle_(view.native_handle_),
        offset_(view.offset_), size_(view.size_) {}

  constexpr bool valid() const noexcept { return resource_.valid(); }
  constexpr explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  std::size_t offset() const noexcept { return offset_; }
  MemoryResourceRef resource() const noexcept { return resource_; }
  void* native_handle() const noexcept { return native_handle_; }

  Result<ConstBufferView> subview(
      std::size_t offset,
      std::size_t size = std::dynamic_extent) const noexcept;
  Status read(std::span<std::byte> destination) const noexcept;
  Result<HostMapping> map_host() const noexcept;
  Status invalidate() const noexcept;

 private:
  friend class Buffer;
  friend class FileMapping;
  friend class SharedMapping;
  constexpr ConstBufferView(MemoryResourceRef resource, void* native_handle,
                            std::size_t offset, std::size_t size) noexcept
      : resource_(resource), native_handle_(native_handle), offset_(offset),
        size_(size) {}

  MemoryResourceRef resource_{};
  void* native_handle_ = nullptr;
  std::size_t offset_ = 0;
  std::size_t size_ = 0;
};

// Scoped CPU mapping of one view. Destruction unmaps but cannot report cleanup
// failure; call close() when the provider's unmap failure must be observed.
// Coherence remains explicit: invalidate before reading device writes and flush
// after host writes when the resource is not host-device coherent.
class HostMapping {
 public:
  constexpr HostMapping() noexcept = default;
  HostMapping(const HostMapping&) = delete;
  HostMapping& operator=(const HostMapping&) = delete;
  HYLE_MEM_API HostMapping(HostMapping&& other) noexcept;
  HYLE_MEM_API HostMapping& operator=(HostMapping&& other) noexcept;
  HYLE_MEM_API ~HostMapping();

  constexpr bool valid() const noexcept { return resource_.valid(); }
  constexpr explicit operator bool() const noexcept { return valid(); }
  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(data_), size_};
  }
  HYLE_MEM_API Result<std::span<std::byte>> writable_bytes() noexcept;
  HYLE_MEM_API Status flush() noexcept;
  HYLE_MEM_API Status invalidate() noexcept;
  HYLE_MEM_API Status close() noexcept;

 private:
  friend class BufferView;
  friend class ConstBufferView;
  HostMapping(MemoryResourceRef resource, void* native_handle, void* token,
              void* data, std::size_t offset, std::size_t size,
              HostAccess access) noexcept
      : resource_(resource), native_handle_(native_handle), token_(token),
        data_(data), offset_(offset), size_(size), access_(access) {}

  MemoryResourceRef resource_{};
  void* native_handle_ = nullptr;
  void* token_ = nullptr;
  void* data_ = nullptr;
  std::size_t offset_ = 0;
  std::size_t size_ = 0;
  HostAccess access_ = HostAccess::kRead;
};

static_assert(sizeof(Buffer) <= sizeof(void*) * 4);
static_assert(sizeof(BufferView) == sizeof(void*) * 4);
static_assert(sizeof(ConstBufferView) == sizeof(void*) * 4);
static_assert(std::is_trivially_copyable_v<BufferView>);
static_assert(std::is_trivially_copyable_v<ConstBufferView>);

// Hot borrowed-view operations stay header-inline. A source/static consumer
// can inline them completely; a shared consumer still performs only the one
// requested provider virtual dispatch, never an extra hyle_mem DLL hop.
inline BufferView Buffer::view() noexcept {
  return BufferView(resource_, native_handle_, 0, size_);
}

inline ConstBufferView Buffer::view() const noexcept {
  return ConstBufferView(resource_, native_handle_, 0, size_);
}

inline Status Buffer::write(std::size_t offset,
                            std::span<const std::byte> source) noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer is invalid");
  if (offset > size_ || source.size() > size_ - offset) {
    return Status::OutOfRange("buffer range exceeds its allocation");
  }
  return resource_.resource().write(native_handle_, offset, source);
}

inline Status Buffer::read(
    std::size_t offset, std::span<std::byte> destination) const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer is invalid");
  if (offset > size_ || destination.size() > size_ - offset) {
    return Status::OutOfRange("buffer range exceeds its allocation");
  }
  return resource_.resource().read(native_handle_, offset, destination);
}

inline Status Buffer::fill(std::size_t offset, std::size_t size,
                           std::byte value) noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer is invalid");
  if (offset > size_ || size > size_ - offset) {
    return Status::OutOfRange("buffer range exceeds its allocation");
  }
  return resource_.resource().fill(native_handle_, offset, size, value);
}

inline Result<BufferView> BufferView::subview(
    std::size_t offset, std::size_t size) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("buffer view is invalid"));
  }
  if (offset > size_) {
    return std::unexpected(
        Status::OutOfRange("buffer subview offset exceeds its range"));
  }
  const std::size_t available = size_ - offset;
  if (size == std::dynamic_extent) size = available;
  if (size > available || offset_ > SIZE_MAX - offset) {
    return std::unexpected(
        Status::OutOfRange("buffer subview size exceeds its range"));
  }
  return BufferView(resource_, native_handle_, offset_ + offset, size);
}

inline Status BufferView::write(
    std::span<const std::byte> source) const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer view is invalid");
  if (source.size() != size_) {
    return Status::InvalidArgument("write size must equal buffer view size");
  }
  return resource_.resource().write(native_handle_, offset_, source);
}

inline Status BufferView::read(
    std::span<std::byte> destination) const noexcept {
  return ConstBufferView(*this).read(destination);
}

inline Status BufferView::fill(std::byte value) const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer view is invalid");
  return resource_.resource().fill(native_handle_, offset_, size_, value);
}

inline Status BufferView::copy_from(ConstBufferView source) const noexcept {
  if (!valid() || !source.valid()) {
    return Status::FailedPrecondition("buffer view is invalid");
  }
  if (source.size() != size_) {
    return Status::InvalidArgument("copy ranges must have equal size");
  }
  return resource_.resource().copy(native_handle_, offset_,
                                   source.resource().resource(),
                                   source.native_handle(), source.offset(),
                                   size_);
}

inline Result<HostMapping> BufferView::map_host(HostAccess access) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("buffer view is invalid"));
  }
  if (!resource_.properties().has(MemoryProperty::kHostAccessible)) {
    return std::unexpected(Status::FailedPrecondition(
        "buffer resource is not host accessible"));
  }
  auto mapping =
      resource_.resource().map_host(native_handle_, offset_, size_, access);
  if (!mapping) return std::unexpected(std::move(mapping).error());
  if (size_ != 0 && mapping->data == nullptr) {
    resource_.resource()
        .unmap_host(native_handle_, mapping->token, offset_, size_, access)
        .IgnoreError();
    return std::unexpected(Status::Internal(
        "memory resource returned a null host mapping"));
  }
  return HostMapping(resource_, native_handle_, mapping->token, mapping->data,
                     offset_, size_, access);
}

inline Status BufferView::flush() const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer view is invalid");
  return resource_.resource().flush(native_handle_, offset_, size_);
}

inline Status BufferView::invalidate() const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer view is invalid");
  return resource_.resource().invalidate(native_handle_, offset_, size_);
}

inline Result<ConstBufferView> ConstBufferView::subview(
    std::size_t offset, std::size_t size) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("buffer view is invalid"));
  }
  if (offset > size_) {
    return std::unexpected(
        Status::OutOfRange("buffer subview offset exceeds its range"));
  }
  const std::size_t available = size_ - offset;
  if (size == std::dynamic_extent) size = available;
  if (size > available || offset_ > SIZE_MAX - offset) {
    return std::unexpected(
        Status::OutOfRange("buffer subview size exceeds its range"));
  }
  return ConstBufferView(resource_, native_handle_, offset_ + offset, size);
}

inline Status ConstBufferView::read(
    std::span<std::byte> destination) const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer view is invalid");
  if (destination.size() != size_) {
    return Status::InvalidArgument("read size must equal buffer view size");
  }
  return resource_.resource().read(native_handle_, offset_, destination);
}

inline Result<HostMapping> ConstBufferView::map_host() const noexcept {
  return BufferView(resource_, native_handle_, offset_, size_)
      .map_host(HostAccess::kRead);
}

inline Status ConstBufferView::invalidate() const noexcept {
  if (!valid()) return Status::FailedPrecondition("buffer view is invalid");
  return resource_.resource().invalidate(native_handle_, offset_, size_);
}

}  // namespace hyle::mem
