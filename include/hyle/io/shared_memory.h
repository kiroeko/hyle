#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

// Named, process-shared memory mapping. The name is UTF-8 and identifies the
// same kernel object across processes. create() is exclusive; open() attaches
// to an existing object and maps its complete size.
class SharedMemory {
 public:
  SharedMemory() noexcept = default;
  HYLE_IO_API ~SharedMemory();
  HYLE_IO_API SharedMemory(SharedMemory&&) noexcept;
  HYLE_IO_API SharedMemory& operator=(SharedMemory&&) noexcept;
  HYLE_DISABLE_COPY(SharedMemory);

  HYLE_IO_API static SharedMemory create(const char* name,
                                         std::size_t size) noexcept;
  HYLE_IO_API static SharedMemory open(const char* name) noexcept;
  // POSIX removes the name while existing mappings remain alive. Windows named
  // mappings disappear automatically after the final handle closes, so unlink
  // is a successful no-op there.
  HYLE_IO_API static bool unlink(const char* name) noexcept;

  bool valid() const noexcept { return data_ != nullptr; }
  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  HYLE_IO_API bool flush() noexcept;
  HYLE_IO_API void reset() noexcept;

 private:
  void* mapping_ = nullptr;
  void* data_ = nullptr;
  std::size_t size_ = 0;
  std::intptr_t handle_ = -1;
};

}  // namespace hyle::io
