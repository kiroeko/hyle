#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_SHARED_MEMORY_API
#elif defined(HYLE_PLATFORM_BUILD)
#  define HYLE_PLATFORM_SHARED_MEMORY_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_SHARED_MEMORY_API HYLE_IMPORT
#endif

namespace hyle::platform {

// OS mechanism for a named, process-shared, read-write memory object. Product
// ownership and Buffer views live in Mem; IO can compose namespace/handle
// transfer protocols without becoming the storage owner.
class SharedMemoryRegion {
 public:
  HYLE_PLATFORM_SHARED_MEMORY_API static Result<SharedMemoryRegion> create(
      const char* name, std::size_t size) noexcept;
  HYLE_PLATFORM_SHARED_MEMORY_API static Result<SharedMemoryRegion> open(
      const char* name) noexcept;
  HYLE_PLATFORM_SHARED_MEMORY_API static Status unlink(
      const char* name) noexcept;

  SharedMemoryRegion() noexcept = default;
  SharedMemoryRegion(const SharedMemoryRegion&) = delete;
  SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;
  HYLE_PLATFORM_SHARED_MEMORY_API SharedMemoryRegion(
      SharedMemoryRegion&& other) noexcept;
  HYLE_PLATFORM_SHARED_MEMORY_API SharedMemoryRegion& operator=(
      SharedMemoryRegion&& other) noexcept;
  HYLE_PLATFORM_SHARED_MEMORY_API ~SharedMemoryRegion();

  bool valid() const noexcept { return data_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

  HYLE_PLATFORM_SHARED_MEMORY_API Status flush() noexcept;
  HYLE_PLATFORM_SHARED_MEMORY_API Status reset() noexcept;

 private:
  void* mapping_ = nullptr;
  void* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t mapping_size_ = 0;
  std::intptr_t handle_ = -1;
};

}  // namespace hyle::platform
