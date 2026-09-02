#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/visibility.h"
#include "hyle/platform/mapped_file.h"
#include "hyle/platform/shared_memory.h"
#include "hyle/platform/virtual_memory.h"

namespace hyle::mem {

enum class MappingAccess : std::uint8_t {
  kReadOnly = 0,
  kReadWrite = 1,
  kCopyOnWrite = 2,
};

enum class MappingProtection : std::uint8_t {
  kNone = 0,
  kRead = 1,
  kReadWrite = 2,
};

enum class MappingAdvice : std::uint8_t {
  kNormal = 0,
  kRandom = 1,
  kSequential = 2,
  kWillNeed = 3,
  kDontNeed = 4,
  kHugePage = 5,
  kNoHugePage = 6,
};

enum class ResidencyMode : std::uint8_t {
  kEager = 0,
  kOnFault = 1,
};

enum class ResidencyLimitPolicy : std::uint8_t {
  kPreserve = 0,
  kAdjustIfNeeded = 1,
};

struct ResidencyOptions {
  ResidencyMode mode = ResidencyMode::kEager;
  ResidencyLimitPolicy limit_policy = ResidencyLimitPolicy::kPreserve;
};

// Anonymous virtual-address owner. reserve() creates inaccessible address
// space; allocate_read_write() creates an immediately usable mapping. Control
// ranges are page-granular and externally synchronized. No operation performs
// an implicit payload copy or hidden fallback allocation.
class VirtualMapping {
 public:
  HYLE_MEM_API static Result<VirtualMapping> reserve(
      std::size_t size) noexcept;
  HYLE_MEM_API static Result<VirtualMapping> allocate_read_write(
      std::size_t size) noexcept;
  HYLE_MEM_API static Result<VirtualMapping> reserve_numa(
      std::size_t size, int node) noexcept;
  HYLE_MEM_API static Result<VirtualMapping> allocate_numa_interleaved(
      std::size_t size, std::span<const unsigned> nodes = {},
      std::size_t bytes_per_chunk = 0) noexcept;

  VirtualMapping() noexcept = default;
  VirtualMapping(const VirtualMapping&) = delete;
  VirtualMapping& operator=(const VirtualMapping&) = delete;
  VirtualMapping(VirtualMapping&&) noexcept = default;
  VirtualMapping& operator=(VirtualMapping&&) noexcept = default;

  bool valid() const noexcept { return region_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  void* data() const noexcept { return region_.data(); }
  std::size_t size() const noexcept { return region_.size(); }

  HYLE_MEM_API Status commit(std::size_t offset,
                             std::size_t size) noexcept;
  HYLE_MEM_API Status decommit(std::size_t offset,
                               std::size_t size) noexcept;
  HYLE_MEM_API Status protect(std::size_t offset, std::size_t size,
                              MappingProtection protection) noexcept;
  HYLE_MEM_API Status advise(MappingAdvice advice, std::size_t offset = 0,
                             std::size_t size = 0) noexcept;
  HYLE_MEM_API Status lock(std::size_t offset = 0, std::size_t size = 0,
                           ResidencyOptions options = {}) noexcept;
  HYLE_MEM_API Status unlock(std::size_t offset = 0,
                             std::size_t size = 0) noexcept;
  HYLE_MEM_API Status release() noexcept;

 private:
  explicit VirtualMapping(platform::VmRegion region) noexcept
      : region_(std::move(region)) {}

  platform::VmRegion region_;
};

// File-backed mapping owner for model weights, datasets and random-access
// files. The file is never silently grown; length==0 maps through current EOF.
// Read-only access is represented by ConstBufferView, while writable_view()
// checks the mapping mode before exposing mutation.
class FileMapping {
 public:
  HYLE_MEM_API static Result<FileMapping> open(
      const char* path, MappingAccess access = MappingAccess::kReadOnly,
      std::size_t offset = 0, std::size_t length = 0) noexcept;
  HYLE_MEM_API static Result<FileMapping> map(
      platform::FileHandle file,
      MappingAccess access = MappingAccess::kReadOnly,
      std::size_t offset = 0, std::size_t length = 0) noexcept;

  FileMapping() noexcept = default;
  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping&) = delete;
  FileMapping(FileMapping&&) noexcept = default;
  FileMapping& operator=(FileMapping&&) noexcept = default;

  bool valid() const noexcept { return mapping_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  const void* data() const noexcept { return mapping_.data(); }
  std::size_t size() const noexcept { return mapping_.size(); }
  MappingAccess access() const noexcept { return access_; }
  std::span<const std::byte> bytes() const noexcept { return mapping_.bytes(); }

  HYLE_MEM_API ConstBufferView view() const noexcept;
  HYLE_MEM_API Result<std::span<std::byte>> writable_bytes() noexcept;
  HYLE_MEM_API Status flush(std::size_t offset = 0,
                            std::size_t length = 0) noexcept;
  HYLE_MEM_API Status advise(MappingAdvice advice, std::size_t offset = 0,
                             std::size_t length = 0) noexcept;
  HYLE_MEM_API Status lock_prefix(
      std::size_t length, ResidencyOptions options = {}) noexcept;
  HYLE_MEM_API Status lock(ResidencyOptions options = {}) noexcept;
  HYLE_MEM_API Status unlock() noexcept;
  std::size_t locked_size() const noexcept { return mapping_.locked_size(); }
  HYLE_MEM_API void reset() noexcept;

 private:
  FileMapping(platform::MappedFile mapping, MappingAccess access) noexcept
      : mapping_(std::move(mapping)), access_(access) {}

  platform::MappedFile mapping_;
  MappingAccess access_ = MappingAccess::kReadOnly;
};

// Named process-shared storage. The name/OS handle is a Platform mechanism;
// this Mem owner exposes the bytes through the same BufferView vocabulary as
// other host-accessible storage. Interprocess synchronization is deliberately
// not invented here and remains the caller/IPC protocol's responsibility.
class SharedMapping {
 public:
  HYLE_MEM_API static Result<SharedMapping> create(
      const char* name, std::size_t size) noexcept;
  HYLE_MEM_API static Result<SharedMapping> open(const char* name) noexcept;
  HYLE_MEM_API static Status unlink(const char* name) noexcept;

  SharedMapping() noexcept = default;
  SharedMapping(const SharedMapping&) = delete;
  SharedMapping& operator=(const SharedMapping&) = delete;
  SharedMapping(SharedMapping&&) noexcept = default;
  SharedMapping& operator=(SharedMapping&&) noexcept = default;

  bool valid() const noexcept { return mapping_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  void* data() noexcept { return mapping_.data(); }
  const void* data() const noexcept { return mapping_.data(); }
  std::size_t size() const noexcept { return mapping_.size(); }
  std::span<std::byte> bytes() noexcept {
    return {static_cast<std::byte*>(data()), size()};
  }
  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(data()), size()};
  }
  HYLE_MEM_API BufferView view() noexcept;
  HYLE_MEM_API ConstBufferView view() const noexcept;
  HYLE_MEM_API Status flush() noexcept;
  HYLE_MEM_API Status reset() noexcept;

 private:
  explicit SharedMapping(platform::SharedMemoryRegion mapping) noexcept
      : mapping_(std::move(mapping)) {}

  platform::SharedMemoryRegion mapping_;
};

}  // namespace hyle::mem
