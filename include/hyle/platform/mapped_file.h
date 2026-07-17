#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_API
#elif defined(HYLE_PLATFORM_BUILD)
#  define HYLE_PLATFORM_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_API HYLE_IMPORT
#endif

// Memory-mapped file (host, single-process). Maps a file's bytes into the address
// space so it can be read/written like memory -- the OS pages it in lazily and
// manages the page cache. Real consumers: zero-copy load of large read-only files
// (weights, datasets, resources), or random-access edits without buffering.
//
// This is the SYNC host-memory sibling of virtual_memory's VmRegion (anonymous VM)
// -- both are platform-layer, depend only on error. Cross-PROCESS shared memory
// (shm_open / named mappings) is a separate, io-layer concern (io-05) and is NOT
// here. Async file read/write over a completion model lives in io/ (proactor).
//
// Design mirrors Rust memmap2 (Mmap / MmapMut / MmapCopy) and folly MemoryMapping:
// a move-only RAII handle; the mapping is torn down on destruction.

namespace hyle::platform {

class MappedFile {
 public:
  enum class Mode : std::uint8_t {
    kReadOnly,    // PROT_READ / PAGE_READONLY -- writes fault (memmap2 Mmap)
    kReadWrite,   // MAP_SHARED -- writes go back to the file (memmap2 MmapMut)
    kCopyOnWrite, // MAP_PRIVATE / PAGE_WRITECOPY -- writes stay private, never
                  // hit the file; the file may even be opened read-only (MmapCopy)
  };

  // Map [offset, offset+length) of `path` (length==0 => to end of file). An
  // explicit length must NOT extend past the file's end -- mapping past EOF is
  // rejected (!valid()), so a read-write mapping never silently grows the file
  // and no access ever faults past EOF. offset is aligned to the OS mapping
  // granularity internally (any value works; data() points at the exact offset).
  // On failure the returned MappedFile is !valid() (see platform vm_last_error()).
  // kReadWrite needs a writable file; kReadOnly / kCopyOnWrite need only read.
  HYLE_PLATFORM_API static MappedFile open(const char* path,
                                           Mode mode = Mode::kReadOnly,
                                           std::size_t offset = 0,
                                           std::size_t length = 0) noexcept;

  MappedFile() noexcept = default;
  HYLE_PLATFORM_API ~MappedFile();
  HYLE_PLATFORM_API MappedFile(MappedFile&& o) noexcept;
  HYLE_PLATFORM_API MappedFile& operator=(MappedFile&& o) noexcept;
  HYLE_DISABLE_COPY(MappedFile);

  bool valid() const noexcept { return addr_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // First mapped byte (the requested offset), or nullptr if !valid().
  void* data() const noexcept { return addr_; }
  // Mapped length in bytes.
  std::size_t size() const noexcept { return size_; }

  // Flush dirty pages back to the file (kReadWrite only; no-op that returns true
  // for kReadOnly / kCopyOnWrite). msync(MS_SYNC) / FlushViewOfFile.
  HYLE_PLATFORM_API bool flush() noexcept;

  // Unmap early (also done by the destructor). Safe on an invalid mapping.
  HYLE_PLATFORM_API void reset() noexcept;

 private:
  void* addr_ = nullptr;       // user-visible base (requested offset)
  std::size_t size_ = 0;       // user-visible length
  // Real mapping base/len: offset is rounded DOWN to the OS mapping granularity
  // (Windows dwAllocationGranularity / POSIX page), so the actual mapping may
  // start before addr_. Torn down via these, not addr_/size_.
  void* map_base_ = nullptr;
  std::size_t map_len_ = 0;
  // OS teardown state (kept opaque as intptr_t so the header pulls in no OS
  // headers): POSIX leaves both -1 (fd closed after mmap). Windows keeps
  // h0_ = file HANDLE and h1_ = mapping HANDLE for unmap+close.
  std::intptr_t h0_ = -1;
  std::intptr_t h1_ = -1;
  Mode mode_ = Mode::kReadOnly;
};

}  // namespace hyle::platform
