#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/mem/visibility.h"

namespace hyle::mem {

// Stable, universal facts about a memory resource. These are independent bits,
// not a closed host/device kind: managed or pinned storage can be accessible by
// more than one agent. New provider-specific facts belong in that provider's
// typed adapter rather than in a backend enum here.
enum class MemoryProperty : std::uint32_t {
  kNone = 0,
  kHostAccessible = 1u << 0,
  kDeviceAccessible = 1u << 1,
  kHostDeviceCoherent = 1u << 2,
  kHostPinned = 1u << 3,
  kManaged = 1u << 4,
  kExportable = 1u << 5,
  kPersistent = 1u << 6,
};

constexpr MemoryProperty operator|(MemoryProperty left,
                                   MemoryProperty right) noexcept {
  return static_cast<MemoryProperty>(static_cast<std::uint32_t>(left) |
                                     static_cast<std::uint32_t>(right));
}
constexpr MemoryProperty operator&(MemoryProperty left,
                                   MemoryProperty right) noexcept {
  return static_cast<MemoryProperty>(static_cast<std::uint32_t>(left) &
                                     static_cast<std::uint32_t>(right));
}
constexpr MemoryProperty& operator|=(MemoryProperty& left,
                                    MemoryProperty right) noexcept {
  left = left | right;
  return left;
}

// Process-local identity for an accessibility/coherence domain. `authority`
// namespaces `value`, so independently implemented providers cannot collide.
// This is deliberately not a serialized or cross-process identifier.
struct MemoryDomainId {
  const void* authority = nullptr;
  std::uintptr_t value = 0;

  friend constexpr bool operator==(MemoryDomainId,
                                   MemoryDomainId) noexcept = default;
};

struct MemoryProperties {
  MemoryProperty flags = MemoryProperty::kNone;
  std::size_t minimum_alignment = alignof(std::max_align_t);
  std::size_t maximum_allocation = SIZE_MAX;
  MemoryDomainId domain{};

  constexpr bool has(MemoryProperty property) const noexcept {
    return (flags & property) == property;
  }
};

// A cold resource snapshot. SIZE_MAX means the provider cannot report that
// field. `used_bytes` is provider-owned live/committed storage, not necessarily
// the sum of user-requested payload bytes.
struct ResourceStats {
  std::size_t capacity_bytes = SIZE_MAX;
  std::size_t used_bytes = SIZE_MAX;
  std::size_t reserved_bytes = SIZE_MAX;
  std::size_t peak_used_bytes = SIZE_MAX;
};

// Provider allocation token. `native_handle` is opaque and only the same
// provider (or its typed upper-layer adapter) may interpret it. For a non-empty
// successful allocation it is non-null. Size/alignment are the original
// logical request and must be returned unchanged on deallocation.
struct ResourceAllocation {
  void* native_handle = nullptr;
  std::size_t size = 0;
  std::size_t alignment = alignof(std::max_align_t);
};

enum class HostAccess : std::uint8_t {
  kRead = 0,
  kWrite = 1,
  kReadWrite = 2,
};

// Result of a provider host mapping. `data` addresses the exact requested
// range; `token` is returned unchanged to unmap_host and may be null for a
// persistent mapping.
struct ResourceHostMapping {
  void* data = nullptr;
  void* token = nullptr;
};

// Open, provider-neutral storage mechanism. Providers are long-lived objects;
// the resource, its backend/device state and every allocation must outlive all
// MemoryResourceRef/Buffer/View and in-flight use. Independent allocations may
// be used concurrently. Mutating/map/control operations on one allocation are
// externally synchronized unless a typed provider promises more.
class HYLE_MEM_API MemoryResource {
 public:
  MemoryResource() noexcept = default;
  MemoryResource(const MemoryResource&) = delete;
  MemoryResource& operator=(const MemoryResource&) = delete;
  virtual ~MemoryResource();

  virtual std::string_view name() const noexcept = 0;
  virtual MemoryProperties properties() const noexcept = 0;
  virtual Result<ResourceAllocation> allocate(
      std::size_t size, std::size_t alignment) noexcept = 0;
  virtual void deallocate(ResourceAllocation allocation) noexcept = 0;

  // Synchronous range operations. Providers can override native routes. The
  // base implementation uses a direct host mapping when available and never
  // allocates a staging buffer.
  virtual Status read(void* native_handle, std::size_t offset,
                      std::span<std::byte> destination) noexcept;
  virtual Status write(void* native_handle, std::size_t offset,
                       std::span<const std::byte> source) noexcept;
  virtual Status fill(void* native_handle, std::size_t offset,
                      std::size_t size, std::byte value) noexcept;
  virtual Status copy(
      void* destination_handle, std::size_t destination_offset,
      MemoryResource& source_resource, void* source_handle,
      std::size_t source_offset, std::size_t size) noexcept;

  virtual Result<ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      HostAccess access) noexcept;
  virtual Status unmap_host(void* native_handle, void* token,
                            std::size_t offset, std::size_t size,
                            HostAccess access) noexcept;
  virtual Status flush(void* native_handle, std::size_t offset,
                       std::size_t size) noexcept;
  virtual Status invalidate(void* native_handle, std::size_t offset,
                            std::size_t size) noexcept;

  virtual Result<ResourceStats> stats() const noexcept;
  virtual Status trim() noexcept;
};

// One-word, non-owning resource reference. Default construction is an explicit
// invalid sentinel for arrays/options and moved-from owners; normal allocation
// APIs reject it. Binding a temporary provider is forbidden.
class MemoryResourceRef {
 public:
  constexpr MemoryResourceRef() noexcept = default;
  constexpr explicit MemoryResourceRef(
      MemoryResource& resource HYLE_LIFETIME_BOUND) noexcept
      : resource_(&resource) {}
  MemoryResourceRef(MemoryResource&&) = delete;

  constexpr bool valid() const noexcept { return resource_ != nullptr; }
  constexpr explicit operator bool() const noexcept { return valid(); }

  MemoryResource& resource() const noexcept {
    HYLE_FATAL_ASSERT(resource_ != nullptr);
    return *resource_;
  }
  std::string_view name() const noexcept { return resource().name(); }
  MemoryProperties properties() const noexcept {
    return resource().properties();
  }
  MemoryDomainId domain() const noexcept { return properties().domain; }

  friend constexpr bool operator==(MemoryResourceRef,
                                   MemoryResourceRef) noexcept = default;

 private:
  MemoryResource* resource_ = nullptr;
};

static_assert(sizeof(MemoryResourceRef) == sizeof(void*));
static_assert(std::is_trivially_copyable_v<MemoryResourceRef>);

// Process-lifetime host-RAM resource used by generic Buffer owners. This is
// distinct from system_allocator() (raw host allocation) and the standard PMR
// adapter: use the narrowest semantic layer a consumer needs.
HYLE_MEM_API MemoryResourceRef system_resource() noexcept;

}  // namespace hyle::mem
