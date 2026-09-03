#include "hyle/mem/resource.h"

#include <cstring>
#include "system_internal.h"

#include "hyle/mem/stats.h"
#include "hyle/obs/trace.h"

namespace hyle::mem {
namespace {

bool host_coherent(const MemoryResource& resource) noexcept {
  return resource.properties().has(MemoryProperty::kHostDeviceCoherent);
}

class SystemMemoryResource final : public MemoryResource {
 public:
  std::string_view name() const noexcept override { return "hyle.system"; }

  MemoryProperties properties() const noexcept override {
    return MemoryProperties{
        .flags = MemoryProperty::kHostAccessible |
                 MemoryProperty::kHostDeviceCoherent,
        .minimum_alignment = kDefaultAlignment,
        .maximum_allocation = SIZE_MAX,
        .domain = MemoryDomainId{this, 0},
    };
  }

  Result<ResourceAllocation> allocate(std::size_t size,
                                      std::size_t alignment) noexcept override {
    if (size == 0) {
      return ResourceAllocation{nullptr, 0, alignment};
    }
    void* memory = detail::system_local_allocate(size, alignment);
    if (memory == nullptr) {
      return std::unexpected(
          Status::ResourceExhausted("system memory allocation failed"));
    }
    HYLE_TRACE_ALLOC_NAMED(memory, size, "hyle.system");
    return ResourceAllocation{memory, size, alignment};
  }

  void deallocate(ResourceAllocation allocation) noexcept override {
#if defined(HYLE_TRACE)
    if (allocation.native_handle != nullptr) {
      HYLE_TRACE_FREE_NAMED(allocation.native_handle, "hyle.system");
    }
#endif
    detail::system_local_deallocate(allocation.native_handle, allocation.size,
                                    allocation.alignment);
  }

  Status read(void* native_handle, std::size_t offset,
              std::span<std::byte> destination) noexcept override {
    if (!destination.empty()) {
      std::memmove(destination.data(),
                   static_cast<const std::byte*>(native_handle) + offset,
                   destination.size());
    }
    return Status::Ok();
  }

  Status write(void* native_handle, std::size_t offset,
               std::span<const std::byte> source) noexcept override {
    if (!source.empty()) {
      std::memmove(static_cast<std::byte*>(native_handle) + offset,
                   source.data(), source.size());
    }
    return Status::Ok();
  }

  Status fill(void* native_handle, std::size_t offset, std::size_t size,
              std::byte value) noexcept override {
    if (size != 0) {
      std::memset(static_cast<std::byte*>(native_handle) + offset,
                  std::to_integer<unsigned char>(value), size);
    }
    return Status::Ok();
  }

  Status copy(void* destination_handle, std::size_t destination_offset,
              MemoryResource& source_resource, void* source_handle,
              std::size_t source_offset, std::size_t size) noexcept override {
    if (&source_resource != this) {
      return MemoryResource::copy(destination_handle, destination_offset,
                                  source_resource, source_handle,
                                  source_offset, size);
    }
    if (size != 0) {
      std::memmove(static_cast<std::byte*>(destination_handle) +
                       destination_offset,
                   static_cast<const std::byte*>(source_handle) + source_offset,
                   size);
    }
    return Status::Ok();
  }

  Result<ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      HostAccess /*access*/) noexcept override {
    void* data = size == 0
                     ? nullptr
                     : static_cast<void*>(
                           static_cast<std::byte*>(native_handle) + offset);
    return ResourceHostMapping{data, nullptr};
  }

  Status unmap_host(void*, void*, std::size_t, std::size_t,
                    HostAccess) noexcept override {
    return Status::Ok();
  }
  Status flush(void*, std::size_t, std::size_t) noexcept override {
    return Status::Ok();
  }
  Status invalidate(void*, std::size_t, std::size_t) noexcept override {
    return Status::Ok();
  }

  Result<ResourceStats> stats() const noexcept override {
    auto snapshot = system_memory_snapshot();
    if (!snapshot) return std::unexpected(std::move(snapshot).error());
    return ResourceStats{
        .capacity_bytes = SIZE_MAX,
        .used_bytes = static_cast<std::size_t>(
            snapshot->allocator_committed_bytes),
        .reserved_bytes = static_cast<std::size_t>(
            snapshot->allocator_reserved_bytes),
        .peak_used_bytes = static_cast<std::size_t>(
            snapshot->allocator_peak_committed_bytes),
    };
  }

  Status trim() noexcept override {
    system_collect(false);
    return Status::Ok();
  }
};

}  // namespace

MemoryResource::~MemoryResource() = default;

Result<ResourceHostMapping> MemoryResource::map_host(
    void*, std::size_t, std::size_t, HostAccess) noexcept {
  return std::unexpected(
      Status::Unimplemented("memory resource is not host mappable"));
}

Status MemoryResource::unmap_host(void*, void*, std::size_t, std::size_t,
                                  HostAccess) noexcept {
  return Status::Ok();
}

Status MemoryResource::flush(void*, std::size_t, std::size_t) noexcept {
  if (host_coherent(*this)) return Status::Ok();
  return Status::Unimplemented(
      "memory resource does not implement host-write flush");
}

Status MemoryResource::invalidate(void*, std::size_t, std::size_t) noexcept {
  if (host_coherent(*this)) return Status::Ok();
  return Status::Unimplemented(
      "memory resource does not implement host-read invalidate");
}

Status MemoryResource::read(void* native_handle, std::size_t offset,
                            std::span<std::byte> destination) noexcept {
  if (destination.empty()) return Status::Ok();
  HYLE_RETURN_IF_ERROR(invalidate(native_handle, offset, destination.size()));
  auto mapping = map_host(native_handle, offset, destination.size(),
                          HostAccess::kRead);
  if (!mapping) return std::move(mapping).error();
  if (mapping->data == nullptr) {
    unmap_host(native_handle, mapping->token, offset, destination.size(),
               HostAccess::kRead)
        .IgnoreError();
    return Status::Internal("memory resource returned a null host mapping");
  }
  std::memcpy(destination.data(), mapping->data, destination.size());
  return unmap_host(native_handle, mapping->token, offset, destination.size(),
                    HostAccess::kRead);
}

Status MemoryResource::write(void* native_handle, std::size_t offset,
                             std::span<const std::byte> source) noexcept {
  if (source.empty()) return Status::Ok();
  auto mapping =
      map_host(native_handle, offset, source.size(), HostAccess::kWrite);
  if (!mapping) return std::move(mapping).error();
  if (mapping->data == nullptr) {
    unmap_host(native_handle, mapping->token, offset, source.size(),
               HostAccess::kWrite)
        .IgnoreError();
    return Status::Internal("memory resource returned a null host mapping");
  }
  std::memcpy(mapping->data, source.data(), source.size());
  Status result = flush(native_handle, offset, source.size());
  result.Update(unmap_host(native_handle, mapping->token, offset, source.size(),
                           HostAccess::kWrite));
  return result;
}

Status MemoryResource::fill(void* native_handle, std::size_t offset,
                            std::size_t size, std::byte value) noexcept {
  if (size == 0) return Status::Ok();
  auto mapping = map_host(native_handle, offset, size, HostAccess::kWrite);
  if (!mapping) return std::move(mapping).error();
  if (mapping->data == nullptr) {
    unmap_host(native_handle, mapping->token, offset, size, HostAccess::kWrite)
        .IgnoreError();
    return Status::Internal("memory resource returned a null host mapping");
  }
  std::memset(mapping->data, std::to_integer<unsigned char>(value), size);
  Status result = flush(native_handle, offset, size);
  result.Update(unmap_host(native_handle, mapping->token, offset, size,
                           HostAccess::kWrite));
  return result;
}

Status MemoryResource::copy(void* destination_handle,
                            std::size_t destination_offset,
                            MemoryResource& source_resource,
                            void* source_handle, std::size_t source_offset,
                            std::size_t size) noexcept {
  if (size == 0) return Status::Ok();

  HYLE_RETURN_IF_ERROR(
      source_resource.invalidate(source_handle, source_offset, size));
  auto source = source_resource.map_host(source_handle, source_offset, size,
                                         HostAccess::kRead);
  if (!source) return std::move(source).error();
  if (source->data == nullptr) {
    source_resource
        .unmap_host(source_handle, source->token, source_offset, size,
                    HostAccess::kRead)
        .IgnoreError();
    return Status::Internal("source resource returned a null host mapping");
  }

  auto destination = map_host(destination_handle, destination_offset, size,
                              HostAccess::kWrite);
  if (!destination) {
    source_resource
        .unmap_host(source_handle, source->token, source_offset, size,
                    HostAccess::kRead)
        .IgnoreError();
    return std::move(destination).error();
  }
  if (destination->data == nullptr) {
    unmap_host(destination_handle, destination->token, destination_offset,
               size, HostAccess::kWrite)
        .IgnoreError();
    source_resource
        .unmap_host(source_handle, source->token, source_offset, size,
                    HostAccess::kRead)
        .IgnoreError();
    return Status::Internal(
        "destination resource returned a null host mapping");
  }

  std::memmove(destination->data, source->data, size);
  Status result = flush(destination_handle, destination_offset, size);
  result.Update(unmap_host(destination_handle, destination->token,
                           destination_offset, size, HostAccess::kWrite));
  result.Update(source_resource.unmap_host(
      source_handle, source->token, source_offset, size, HostAccess::kRead));
  return result;
}

Result<ResourceStats> MemoryResource::stats() const noexcept {
  return std::unexpected(
      Status::Unimplemented("memory resource statistics are unavailable"));
}

Status MemoryResource::trim() noexcept {
  return Status::Unimplemented("memory resource cannot be trimmed");
}

MemoryResourceRef system_resource() noexcept {
  static SystemMemoryResource resource;
  return MemoryResourceRef(resource);
}

}  // namespace hyle::mem
