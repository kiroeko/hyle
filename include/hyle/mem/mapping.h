#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/visibility.h"
#include "hyle/platform/fs.h"
#include "hyle/platform/virtual_memory.h"

namespace hyle::mem {

// Mem is the storage-product facade; Platform remains the sole anonymous-VM
// mechanism and type authority. These aliases add no wrapper, dispatch or
// second enum contract.
using VirtualMapping = platform::VmRegion;
using MappingProtection = platform::Protect;
using MappingAdvice = platform::VmAdvice;
using ResidencyMode = platform::VmLockMode;
using ResidencyLimitPolicy = platform::VmLockLimitPolicy;
using ResidencyOptions = platform::VmLockOptions;

enum class MappingAccess : std::uint8_t {
  kReadOnly = 0,
  kReadWrite = 1,
  kCopyOnWrite = 2,
};

struct MappingRange {
  std::size_t offset = 0;
  std::size_t size = 0;
};

namespace detail {
struct FileMappingFragment;
}

// Move-only file storage for model weights, datasets and random-access files.
// The logical [0,size()) extent is stable, while POSIX release_range(s) may
// remove complete native pages from it. Accessors are checked so a hole can
// never be represented as one contiguous span/view by accident.
class FileMapping {
 public:
  HYLE_MEM_VM_API static Result<FileMapping> open(
      const char* path, MappingAccess access = MappingAccess::kReadOnly,
      std::size_t offset = 0, std::size_t length = 0) noexcept;
  HYLE_MEM_VM_API static Result<FileMapping> map(
      platform::FileHandle file,
      MappingAccess access = MappingAccess::kReadOnly,
      std::size_t offset = 0, std::size_t length = 0) noexcept;

  FileMapping() noexcept = default;
  HYLE_MEM_VM_API ~FileMapping();
  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping&) = delete;
  HYLE_MEM_VM_API FileMapping(FileMapping&&) noexcept;
  HYLE_MEM_VM_API FileMapping& operator=(FileMapping&&) noexcept;

  bool valid() const noexcept { return size_ != 0; }
  explicit operator bool() const noexcept { return valid(); }
  std::size_t size() const noexcept { return size_; }
  std::size_t mapped_size() const noexcept { return mapped_size_; }
  HYLE_MEM_VM_API std::size_t fragment_count() const noexcept;
  MappingAccess access() const noexcept { return access_; }

  HYLE_MEM_VM_API bool contains(
      std::size_t offset,
      std::size_t length = std::dynamic_extent) const noexcept;
  HYLE_MEM_VM_API Result<const void*> address(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) const noexcept;
  HYLE_MEM_VM_API Result<void*> writable_address(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) noexcept;
  HYLE_MEM_VM_API Result<std::span<const std::byte>> bytes(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) const noexcept;
  HYLE_MEM_VM_API Result<std::span<std::byte>> writable_bytes(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) noexcept;
  HYLE_MEM_VM_API Result<ConstBufferView> view(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) const noexcept;
  HYLE_MEM_VM_API Result<BufferView> writable_view(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) noexcept;

  HYLE_MEM_VM_API Status flush(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) noexcept;
  HYLE_MEM_VM_API Status advise(
      MappingAdvice advice, std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) noexcept;
  Status prefetch(std::size_t offset = 0,
                  std::size_t length = std::dynamic_extent) noexcept {
    return advise(MappingAdvice::kWillNeed, offset, length);
  }
  HYLE_MEM_VM_API Status populate(
      std::size_t offset = 0,
      std::size_t length = std::dynamic_extent) noexcept;

  HYLE_MEM_VM_API Result<std::size_t> release_range(
      std::size_t offset, std::size_t length) noexcept;
  HYLE_MEM_VM_API Result<std::size_t> release_ranges(
      std::span<const MappingRange> ranges) noexcept;

  HYLE_MEM_VM_API Status lock_prefix(
      std::size_t length, ResidencyOptions options = {}) noexcept;
  HYLE_MEM_VM_API Status lock(ResidencyOptions options = {}) noexcept;
  HYLE_MEM_VM_API Status unlock() noexcept;
  std::size_t locked_size() const noexcept { return locked_prefix_size_; }
  bool locked() const noexcept { return locked_prefix_size_ != 0; }

  HYLE_MEM_VM_API Status reset() noexcept;

 private:
  void* addr_ = nullptr;
  std::size_t size_ = 0;
  void* map_base_ = nullptr;
  std::size_t map_len_ = 0;
  std::intptr_t h0_ = -1;
  std::intptr_t h1_ = -1;
  MappingAccess access_ = MappingAccess::kReadOnly;
  std::size_t locked_prefix_size_ = 0;
  ResidencyMode locked_mode_ = ResidencyMode::kEager;
  detail::FileMappingFragment* fragments_ = nullptr;
  std::size_t fragment_count_ = 0;
  std::size_t fragment_capacity_ = 0;
  std::size_t mapped_size_ = 0;
};

// Named process-shared storage. Size and access are explicit external protocol
// facts; HYLE adds no hidden payload header and therefore preserves native page
// alignment. HYLE checks the native capacity reported by the OS: Linux shm
// preserves the requested ftruncate size, while Darwin and Win32 round capacity
// to a native page boundary. Those rounded platforms cannot recover an exact
// logical payload size through the mapping API, so peers must agree on it.
// Publication/readiness and interprocess synchronization remain the caller's
// protocol.
class SharedMapping {
 public:
  HYLE_MEM_VM_API static Result<SharedMapping> create(
      const char* name, std::size_t size) noexcept;
  HYLE_MEM_VM_API static Result<SharedMapping> open(
      const char* name, std::size_t size,
      MappingAccess access = MappingAccess::kReadOnly) noexcept;
  HYLE_MEM_VM_API static Status unlink(const char* name) noexcept;

  SharedMapping() noexcept = default;
  HYLE_MEM_VM_API ~SharedMapping();
  SharedMapping(const SharedMapping&) = delete;
  SharedMapping& operator=(const SharedMapping&) = delete;
  HYLE_MEM_VM_API SharedMapping(SharedMapping&&) noexcept;
  HYLE_MEM_VM_API SharedMapping& operator=(SharedMapping&&) noexcept;

  bool valid() const noexcept { return mapping_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  const void* data() const noexcept { return mapping_; }
  HYLE_MEM_VM_API Result<void*> writable_data() noexcept;
  std::size_t size() const noexcept { return size_; }
  MappingAccess access() const noexcept { return access_; }
  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(data()), size()};
  }
  HYLE_MEM_VM_API Result<std::span<std::byte>> writable_bytes() noexcept;
  HYLE_MEM_VM_API ConstBufferView view() const noexcept;
  HYLE_MEM_VM_API Result<BufferView> writable_view() noexcept;
  HYLE_MEM_VM_API Status flush() noexcept;
  HYLE_MEM_VM_API Status reset() noexcept;

 private:
  void* mapping_ = nullptr;
  std::size_t size_ = 0;
  std::intptr_t handle_ = -1;
  MappingAccess access_ = MappingAccess::kReadOnly;
};

}  // namespace hyle::mem
