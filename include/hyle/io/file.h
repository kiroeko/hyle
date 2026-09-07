#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/error/result.h"
#include "hyle/io/buffer.h"
#include "hyle/io/detail/net_budget.h"
#include "hyle/io/io_context.h"
#include "hyle/io/operation.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#define HYLE_IO_API HYLE_EXPORT
#else
#define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

enum class FileAccess : std::uint8_t {
  kRead,
  kWrite,
  kReadWrite,
};

enum class FileCacheMode : std::uint8_t {
  kBuffered,
  // The open must establish a real OS direct/non-buffered contract or fail.
  // HYLE never silently falls back to buffered IO.
  kDirect,
};

enum class FileSyncMode : std::uint8_t {
  kDataAndMetadata,
  kDataOnly,
};

struct FileOpenOptions {
  FileAccess access = FileAccess::kRead;
  bool create = false;
  bool truncate = false;
  FileCacheMode cache = FileCacheMode::kBuffered;
  std::uint32_t create_permissions = 0644;
};

// Direct IO has independent address, offset, and length constraints. Values
// are powers of two and are 1 for an unconstrained dimension. Allocate through
// HYLE Mem with memory_alignment and validate every request with accepts().
struct DirectIoAlignment {
  std::size_t memory_alignment = 1;
  std::size_t offset_alignment = 1;
  std::size_t length_alignment = 1;

  [[nodiscard]] bool accepts(const void *buffer, std::uint64_t offset,
                             std::size_t length) const noexcept {
    if (length == 0)
      return true;
    const auto address = reinterpret_cast<std::uintptr_t>(buffer);
    return buffer != nullptr && address % memory_alignment == 0 &&
           offset % offset_alignment == 0 && length % length_alignment == 0;
  }
};

class File {
public:
  File() noexcept = default;
  HYLE_IO_API ~File();
  HYLE_IO_API File(File &&other) noexcept;
  HYLE_IO_API File &operator=(File &&other) noexcept;
  File(const File &) = delete;
  File &operator=(const File &) = delete;

  // Adopt the sole owner produced by Platform. POSIX io_uring can submit an
  // ordinary descriptor directly. A Windows FileHandle was not necessarily
  // opened for OVERLAPPED IO, so adoption honestly uses the blocking fallback;
  // open_file() remains the native asynchronous path there.
  HYLE_IO_API static Result<File> adopt(IoContext &io, unsigned owner_worker,
                                        platform::FileHandle &&handle) noexcept;

  bool valid() const noexcept { return handle_.valid(); }
  explicit operator bool() const noexcept { return valid(); }
  std::intptr_t native_handle() const noexcept { return handle_.get(); }
  unsigned owner_worker() const noexcept { return owner_; }
  FileCacheMode cache_mode() const noexcept { return cache_; }
  DirectIoAlignment direct_alignment() const noexcept { return alignment_; }

  // Transfer ownership back to Platform. There must be no operation in flight;
  // scheduler affinity and direct-IO metadata are intentionally discarded.
  HYLE_IO_API platform::FileHandle release() noexcept;

  struct ReadSomeAtAwaitable {
    File *self = nullptr;
    std::span<std::byte> buffer;
    std::uint64_t offset = 0;
    OperationOptions options{};
    detail::BudgetState state{};
    TransferResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API TransferResult await_resume() noexcept;
  };

  struct WriteSomeAtAwaitable {
    File *self = nullptr;
    std::span<const std::byte> buffer;
    std::uint64_t offset = 0;
    OperationOptions options{};
    detail::BudgetState state{};
    TransferResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API TransferResult await_resume() noexcept;
  };

  struct ReadBuffersAtAwaitable {
    File *self = nullptr;
    detail::BufferSequenceStorage<MutableBuffer> buffers;
    std::uint64_t offset = 0;
    OperationOptions options{};
    detail::BudgetState state{};
    detail::StagingBuffer staging{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  struct WriteBuffersAtAwaitable {
    File *self = nullptr;
    detail::BufferSequenceStorage<ConstBuffer> buffers;
    std::uint64_t offset = 0;
    OperationOptions options{};
    detail::BudgetState state{};
    detail::StagingBuffer staging{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  ReadSomeAtAwaitable read_some_at(std::uint64_t offset,
                                   std::span<std::byte> buffer,
                                   OperationOptions options = {}) noexcept {
    return ReadSomeAtAwaitable{this, buffer, offset, options};
  }
  WriteSomeAtAwaitable write_some_at(std::uint64_t offset,
                                     std::span<const std::byte> buffer,
                                     OperationOptions options = {}) noexcept {
    return WriteSomeAtAwaitable{this, buffer, offset, options};
  }
  ReadBuffersAtAwaitable read_some_at(
      std::uint64_t offset, std::span<const MutableBuffer> buffers,
      OperationOptions options = {},
      BufferSequenceCursor cursor = {}) noexcept {
    return ReadBuffersAtAwaitable{
        this, detail::BufferSequenceStorage<MutableBuffer>(buffers, cursor),
        offset, options};
  }
  WriteBuffersAtAwaitable write_some_at(
      std::uint64_t offset, std::span<const ConstBuffer> buffers,
      OperationOptions options = {},
      BufferSequenceCursor cursor = {}) noexcept {
    return WriteBuffersAtAwaitable{
        this, detail::BufferSequenceStorage<ConstBuffer>(buffers, cursor),
        offset, options};
  }

  struct InfoAwaitable {
    File *self = nullptr;
    detail::Proactor::FileStat native{};
    std::int64_t result = 0;
    Status preparation{};
    bool await_ready() noexcept { return false; }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Result<platform::FileInfo> await_resume() noexcept;
  };

  struct SyncAwaitable {
    File *self = nullptr;
    FileSyncMode mode = FileSyncMode::kDataAndMetadata;
    detail::IoOperation operation{};
    bool submitted = false;
    Status preparation{};
    bool await_ready() noexcept { return false; }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Status await_resume() noexcept;
  };

  struct PreallocateAwaitable {
    File *self = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    detail::IoOperation operation{};
    bool submitted = false;
    Status preparation{};
    bool await_ready() noexcept { return false; }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Status await_resume() noexcept;
  };

  struct CloseAwaitable {
    File *self = nullptr;
    detail::IoOperation operation{};
    bool submitted = false;
    bool await_ready() noexcept { return self == nullptr || !self->valid(); }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Status await_resume() noexcept;
  };

  InfoAwaitable info() noexcept { return InfoAwaitable{this}; }
  SyncAwaitable
  sync(FileSyncMode mode = FileSyncMode::kDataAndMetadata) noexcept {
    return SyncAwaitable{this, mode};
  }
  PreallocateAwaitable preallocate(std::uint64_t offset,
                                   std::uint64_t length) noexcept {
    return PreallocateAwaitable{this, offset, length};
  }
  CloseAwaitable close() noexcept { return CloseAwaitable{this}; }

private:
  HYLE_IO_API File(IoContext &io, platform::FileHandle &&handle,
                   unsigned owner_worker, bool native_async,
                   FileCacheMode cache, DirectIoAlignment alignment) noexcept;

  IoContext *io_ = nullptr;
  platform::FileHandle handle_;
  unsigned owner_ = 0;
  bool native_async_ = false;
  FileCacheMode cache_ = FileCacheMode::kBuffered;
  DirectIoAlignment alignment_{};

  friend struct OpenFileAwaitable;
};

// Owns the UTF-8 path across asynchronous open. The common short path remains
// inline; long paths grow through IoContext's injected allocator rather than a
// hidden global allocation. Open uses the blocking pool where the OS has no
// asynchronous path, never a scheduler worker.
struct OpenFileAwaitable {
  HYLE_IO_API OpenFileAwaitable(IoContext &io, unsigned owner_worker,
                                std::string_view path,
                                FileOpenOptions options) noexcept;
  HYLE_IO_API ~OpenFileAwaitable();
  HYLE_IO_API OpenFileAwaitable(OpenFileAwaitable &&) noexcept;
  HYLE_IO_API OpenFileAwaitable &operator=(OpenFileAwaitable &&) noexcept;
  OpenFileAwaitable(const OpenFileAwaitable &) = delete;
  OpenFileAwaitable &operator=(const OpenFileAwaitable &) = delete;

  HYLE_IO_API bool await_ready() noexcept;
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  HYLE_IO_API Result<File> await_resume() noexcept;

  IoContext *io = nullptr;
  unsigned owner = 0;
  FileOpenOptions options{};
  container::SmallVector<char, 256, mem::AllocatorRef> path;
  Status preparation{};
  std::int64_t native_result = detail::kSubmitFailed;
  std::intptr_t native_handle = platform::FileHandle::kInvalid;
  DirectIoAlignment alignment{};
  bool native_async = false;
};

inline OpenFileAwaitable open_file(IoContext &io, unsigned owner_worker,
                                   std::string_view path,
                                   FileOpenOptions options = {}) noexcept {
  return OpenFileAwaitable(io, owner_worker, path, options);
}

// Complete positional transfers preserve partial progress and reuse one
// absolute deadline/cancellation budget across every native chunk.
HYLE_IO_API sched::Task<TransferResult>
read_exact_at(mem::AllocatorRef allocator, File &file, std::uint64_t offset,
              std::span<std::byte> buffer,
              OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<TransferResult>
write_all_at(mem::AllocatorRef allocator, File &file, std::uint64_t offset,
             std::span<const std::byte> buffer,
             OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<BufferSequenceResult>
read_exact_at(mem::AllocatorRef allocator, File &file, std::uint64_t offset,
              std::span<const MutableBuffer> buffers,
              OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<BufferSequenceResult>
write_all_at(mem::AllocatorRef allocator, File &file, std::uint64_t offset,
             std::span<const ConstBuffer> buffers,
             OperationOptions options = {}) noexcept;

} // namespace hyle::io
