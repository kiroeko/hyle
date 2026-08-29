#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
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

class FileHandle;

class MappedFile {
 public:
  enum class Mode : std::uint8_t {
    kReadOnly,    // PROT_READ / PAGE_READONLY -- writes fault (memmap2 Mmap)
    kReadWrite,   // MAP_SHARED -- writes go back to the file (memmap2 MmapMut)
    kCopyOnWrite, // MAP_PRIVATE / PAGE_WRITECOPY -- writes stay private, never
                  // hit the file; the file may even be opened read-only (MmapCopy)
  };

  // Kernel access-pattern hints for a mapped range. These are performance
  // hints, never correctness requirements. POSIX maps them to madvise. Windows
  // can apply kWillNeed through PrefetchVirtualMemory; unsupported hints return
  // Unimplemented instead of silently claiming that the request was applied.
  enum class Advice : std::uint8_t {
    kNormal = 0,
    kRandom = 1,
    kSequential = 2,
    kWillNeed = 3,
    kDontNeed = 4,
  };

  // How newly requested pages enter the OS locked set. kOnFault avoids
  // synchronously faulting every page before the call returns, but is a Linux
  // mlock2(MLOCK_ONFAULT) capability; unsupported backends return
  // Unimplemented instead of silently changing the requested latency shape.
  enum class LockMode : std::uint8_t {
    kEager = 0,
    kOnFault = 1,
  };

  // Page locking is charged against a process-wide OS limit. Preserving that
  // limit is the unsurprising default. kAdjustIfNeeded explicitly permits one
  // best-effort soft-limit/working-set increase after a quota failure and one
  // retry; a successful increase is process-global and is not rolled back by
  // unlock or destruction.
  enum class LockLimitPolicy : std::uint8_t {
    kPreserve = 0,
    kAdjustIfNeeded = 1,
  };

  struct LockOptions {
    LockMode mode = LockMode::kEager;
    LockLimitPolicy limit_policy = LockLimitPolicy::kPreserve;
  };

  // Map [offset, offset+length) of `path` (length==0 => to end of file). An
  // explicit length must NOT extend past the file's end -- mapping past EOF is
  // rejected with OutOfRange, so a read-write mapping never silently grows the
  // file and no access ever faults past EOF. offset is aligned to the OS mapping
  // granularity internally (any value works; data() points at the exact offset).
  // Failures carry operation and native OS diagnostics in the returned Status.
  // kReadWrite needs a writable file; kReadOnly / kCopyOnWrite need only read.
  HYLE_PLATFORM_API static Result<MappedFile> open(
      const char* path, Mode mode = Mode::kReadOnly, std::size_t offset = 0,
      std::size_t length = 0) noexcept;

  // Consume an already-open file and map the same range. This is the natural
  // shape when a caller has opened/validated a file once and must avoid a
  // second path lookup. Ownership is consumed on success and failure. The
  // handle must be readable, and kReadWrite additionally requires write access.
  HYLE_PLATFORM_API static Result<MappedFile> map(
      FileHandle file, Mode mode = Mode::kReadOnly, std::size_t offset = 0,
      std::size_t length = 0) noexcept;

  MappedFile() noexcept = default;
  HYLE_PLATFORM_API ~MappedFile();
  HYLE_PLATFORM_API MappedFile(MappedFile&& o) noexcept;
  HYLE_PLATFORM_API MappedFile& operator=(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  bool valid() const noexcept { return addr_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // First mapped byte (the requested offset), or nullptr if !valid(). Read-only
  // access is always available; mutation requires writable_bytes(), which
  // checks the mapping mode instead of exposing writable memory through const.
  const void* data() const noexcept { return addr_; }
  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(addr_), size_};
  }
  HYLE_PLATFORM_API Result<std::span<std::byte>> writable_bytes() noexcept;
  // Mapped length in bytes.
  std::size_t size() const noexcept { return size_; }
  Mode mode() const noexcept { return mode_; }

  // Flush dirty pages in [offset, offset+length) back to the file (kReadWrite
  // only; OK no-op for kReadOnly / kCopyOnWrite). length==0 means through the
  // end. Empty end ranges are successful no-ops. msync(MS_SYNC) /
  // FlushViewOfFile + FlushFileBuffers.
  HYLE_PLATFORM_API Status flush(std::size_t offset = 0,
                                 std::size_t length = 0) noexcept;

  // Advise the kernel about [offset, offset+length). length==0 means through
  // the end of this mapping. Empty end ranges are successful no-ops. The
  // common path performs one native call and no allocation.
  HYLE_PLATFORM_API Status advise(Advice advice, std::size_t offset = 0,
                                  std::size_t length = 0) noexcept;

  // Convenience for the common model/dataset warm-up request.
  Status prefetch(std::size_t offset = 0, std::size_t length = 0) noexcept {
    return advise(Advice::kWillNeed, offset, length);
  }

  // Keep the full mapping resident. This is the common one-shot convenience
  // over lock_prefix(size(), options).
  Status lock() noexcept { return lock(LockOptions{}); }
  HYLE_PLATFORM_API Status lock(LockOptions options) noexcept;

  // Monotonically grow the locked public prefix to cover at least `length`
  // bytes. OS page rounding may make locked_size() larger than requested.
  // Growth locks only the newly uncovered native pages; failure preserves the
  // previously applied prefix. A non-empty prefix cannot change LockMode until
  // unlock(). length==0 is a successful no-op; length>size() is OutOfRange.
  // Control operations are externally synchronized, like reset/flush.
  Status lock_prefix(std::size_t length) noexcept {
    return lock_prefix(length, LockOptions{});
  }
  HYLE_PLATFORM_API Status lock_prefix(std::size_t length,
                                       LockOptions options) noexcept;

  // Release the entire applied prefix. Repeated unlock calls are idempotent;
  // unmapping also releases the OS lock.
  HYLE_PLATFORM_API Status unlock() noexcept;
  bool locked() const noexcept { return locked_native_size_ != 0; }

  // Public prefix currently covered by the page-granular native lock, clipped
  // to size(). Zero means no prefix is tracked as locked.
  std::size_t locked_size() const noexcept {
    if (locked_native_size_ == 0 || addr_ == nullptr || map_base_ == nullptr) {
      return 0;
    }
    const std::uintptr_t visible = reinterpret_cast<std::uintptr_t>(addr_);
    const std::uintptr_t native = reinterpret_cast<std::uintptr_t>(map_base_);
    const std::size_t leading = static_cast<std::size_t>(visible - native);
    if (locked_native_size_ <= leading) return 0;
    const std::size_t covered = locked_native_size_ - leading;
    return covered < size_ ? covered : size_;
  }

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
  std::size_t locked_native_size_ = 0;
  LockMode locked_mode_ = LockMode::kEager;
};

}  // namespace hyle::platform
