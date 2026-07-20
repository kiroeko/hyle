#pragma once

#include <cstdint>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"  // IoOperation, to_token, sched bridge
#include "hyle/io/proactor.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-05 file io (capability C5). Async file read/write/stat/sync over the proactor,
// co_await style like TcpStream. On io_uring these are REAL async ops (openat /
// read / write / fsync / fallocate use their native SQE; regular-file read/write is
// the one place io_uring genuinely beats epoll, which cannot async a regular file);
// on IOCP read/write are real OVERLAPPED operations; epoll/kqueue cannot async a
// regular file. The high-level File API therefore chooses per operation: native
// proactor when the backend supports it, otherwise IoContext's blocking pool. No
// synchronous disk syscall runs on a scheduler worker. The low-level Proactor
// remains available for users who deliberately want exact backend mechanics.
//
// Shard-per-core like the socket wrappers: a File is bound to an owner worker's
// ring, and its co_await ops MUST run on the fiber pinned there (Scheduler::
// spawn_on), so submit + reap + resume stay single-threaded on one core.
//
// Independent of C2 (timeout io): file ops don't take deadlines here (open/stat/
// fsync rarely need them; a slow disk is not a dead peer). Timeout/cancel is io-06.
//
// LIFETIME INVARIANT (shared with net.h's socket ops): an in-flight op parks its
// IoOperation slot on the awaiting fiber's frozen stack and hands its address to
// the kernel as the completion token. The fiber MUST NOT be destroyed while an op
// is in flight -- the completion would write into freed memory. co_await always
// waits for the completion, so ordinary use is safe; forcibly destroying a
// suspended fiber mid-op is what cooperative cancellation (io-06/C3) will address.

namespace hyle::io {

// A co_await-able that submits ONE proactor file op and resumes with its result.
// Distinct from net.h's socket ops only in which submit_* it calls; the suspend/
// wake plumbing is identical (an IoOperation slot on the frozen fiber stack whose
// address is the completion token). Result is >= 0 (fd / bytes / 0) or -errno.

// co_await FileReadOp -> bytes read (>=0, 0 = EOF) or negative errno. Reads at
// file `offset` (byte position), unlike a stream recv.
struct FileReadOp {
  IoContext* io;
  std::intptr_t fd;
  void* buf;
  std::uint32_t len;
  std::uint64_t offset;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await FileWriteOp -> bytes written (>=0) or negative errno, at file `offset`.
struct FileWriteOp {
  IoContext* io;
  std::intptr_t fd;
  const void* buf;
  std::uint32_t len;
  std::uint64_t offset;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await FileStatOp -> 0 on success (fills *out) or negative errno.
struct FileStatOp {
  IoContext* io;
  std::intptr_t fd;
  Proactor::FileStat* out;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await FileSyncOp -> 0 on success or negative errno. datasync = fdatasync.
struct FileSyncOp {
  IoContext* io;
  std::intptr_t fd;
  bool datasync;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  std::int64_t await_resume() noexcept { return op_.result; }
};

// co_await FileAllocOp -> 0 on success or negative errno (preallocate/extend).
struct FileAllocOp {
  IoContext* io;
  std::intptr_t fd;
  std::uint32_t mode;
  std::uint64_t offset;
  std::uint64_t len;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  std::int64_t await_resume() noexcept { return op_.result; }
};

// An open regular file. Move-only; co_await close() to release (the destructor
// also closes synchronously if you drop it without awaiting close -- but prefer
// co_await close() so the close itself is async on io_uring). read/write take an
// explicit file offset; stat/sync/allocate are metadata ops. All co_await methods
// MUST be awaited from the fiber pinned to owner_worker().
class File {
 public:
  File() noexcept = default;
  HYLE_IO_API File(IoContext& io, std::intptr_t fd, unsigned owner_worker) noexcept;
  HYLE_IO_API ~File();
  HYLE_IO_API File(File&& o) noexcept;
  HYLE_IO_API File& operator=(File&& o) noexcept;
  HYLE_DISABLE_COPY(File);

  bool valid() const noexcept { return fd_ >= 0; }
  std::intptr_t fd() const noexcept { return fd_; }
  unsigned owner_worker() const noexcept { return owner_; }

  FileReadOp read(void* buf, std::uint32_t len, std::uint64_t offset) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buf != nullptr || len == 0);
    return FileReadOp{io_, fd_, buf, len, offset};
  }
  FileWriteOp write(const void* buf, std::uint32_t len,
                    std::uint64_t offset) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(buf != nullptr || len == 0);
    return FileWriteOp{io_, fd_, buf, len, offset};
  }
  FileStatOp stat(Proactor::FileStat* out) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT(out != nullptr);
    return FileStatOp{io_, fd_, out};
  }
  FileSyncOp sync(bool datasync = false) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return FileSyncOp{io_, fd_, datasync};
  }
  FileAllocOp allocate(std::uint64_t offset, std::uint64_t len,
                       std::uint32_t mode = 0) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return FileAllocOp{io_, fd_, mode, offset, len};
  }

  // co_await file.close() -> 0 or negative errno. Releases the fd (async on
  // io_uring). After this the File is invalid; the destructor won't double-close.
  struct CloseAwaitable {
    File* self;
    IoOperation op_{};
    bool submitted_ = false;
    bool await_ready() noexcept { return false; }
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    std::int64_t await_resume() noexcept { return op_.result; }
  };
  CloseAwaitable close() noexcept { return CloseAwaitable{this}; }

 private:
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

// A co_await-able that opens `path` on `owner_worker`'s ring and wraps the fd in a
// File. co_await it from the fiber pinned to owner_worker. On failure (negative
// result) returns an invalid File (valid() == false). `path` must outlive the
// co_await (keep it alive on the coroutine frame -- io_uring reads it when the op
// runs). `flags` is a Proactor::OpenFlags bitmask; `mode` is the POSIX permission
// for a newly created file (ignored on Windows).
struct OpenAwaitable {
  IoContext* io;
  unsigned owner;
  const char* path;
  std::uint32_t flags;
  std::uint32_t mode;

  IoOperation op_{};
  bool submitted_ = false;

  bool await_ready() noexcept { return false; }
  HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
  File await_resume() noexcept {
    if (op_.result < 0) return File{};
    return File{*io, static_cast<std::intptr_t>(op_.result), owner};
  }
};

// co_await open_file(io, worker, "path", flags) -> File. Opens on `owner_worker`'s
// ring; the returned File's fibers must be spawn_on(owner_worker) to service it on
// that core. Default mode 0644 for created files.
inline OpenAwaitable open_file(IoContext& io, unsigned owner_worker,
                               const char* path, std::uint32_t flags,
                               std::uint32_t mode = 0644) noexcept {
  return OpenAwaitable{&io, owner_worker, path, flags, mode};
}

}  // namespace hyle::io
