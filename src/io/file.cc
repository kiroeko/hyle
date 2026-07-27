#include "hyle/io/file.h"

#include <cerrno>
#include <limits>
#include <utility>

#include "hyle/platform.h"  // HYLE_OS_*
#include "hyle/sched/blocking.h"
#include "hyle/time/unix_timestamp.h"

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  if defined(HYLE_OS_MACOS)
#    include <sys/fcntl.h>
#  endif
#endif

// io-05 file convenience layer implementation: RAII fd/handle lifetime. The
// co_await methods are inline in file.h (they build proactor-op awaitables); this
// file owns the fd's lifetime. Prefer `co_await file.close()` (async on io_uring)
// over dropping the File -- but the destructor closes SYNCHRONOUSLY as a safety net
// so a dropped-without-close File never leaks a descriptor.

namespace hyle::io {
namespace {

std::int64_t os_error() noexcept {
#if defined(HYLE_OS_WINDOWS)
  return -static_cast<std::int64_t>(::GetLastError());
#else
  return -static_cast<std::int64_t>(errno);
#endif
}

#if defined(HYLE_OS_WINDOWS)
std::int64_t file_read_sync(std::intptr_t fd, void* buf, std::uint32_t len,
                            std::uint64_t offset) noexcept {
  HANDLE event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (event == nullptr) return os_error();
  OVERLAPPED ov{};
  ov.Offset = static_cast<DWORD>(offset);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  // A low-bit-tagged event tells an IOCP-associated handle not to enqueue this
  // stack OVERLAPPED to the completion port. This helper waits synchronously and
  // the OVERLAPPED dies on return.
  ov.hEvent = reinterpret_cast<HANDLE>(
      reinterpret_cast<ULONG_PTR>(event) | static_cast<ULONG_PTR>(1));
  DWORD transferred = 0;
  BOOL ok = ::ReadFile(reinterpret_cast<HANDLE>(fd), buf, len, &transferred, &ov);
  if (!ok && ::GetLastError() == ERROR_IO_PENDING)
    ok = ::GetOverlappedResult(reinterpret_cast<HANDLE>(fd), &ov, &transferred,
                               TRUE);
  const std::int64_t result = ok ? static_cast<std::int64_t>(transferred)
                                 : os_error();
  ::CloseHandle(event);
  return result;
}

std::int64_t file_write_sync(std::intptr_t fd, const void* buf,
                             std::uint32_t len,
                             std::uint64_t offset) noexcept {
  HANDLE event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (event == nullptr) return os_error();
  OVERLAPPED ov{};
  ov.Offset = static_cast<DWORD>(offset);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  ov.hEvent = reinterpret_cast<HANDLE>(
      reinterpret_cast<ULONG_PTR>(event) | static_cast<ULONG_PTR>(1));
  DWORD transferred = 0;
  BOOL ok = ::WriteFile(reinterpret_cast<HANDLE>(fd), buf, len, &transferred, &ov);
  if (!ok && ::GetLastError() == ERROR_IO_PENDING)
    ok = ::GetOverlappedResult(reinterpret_cast<HANDLE>(fd), &ov, &transferred,
                               TRUE);
  const std::int64_t result = ok ? static_cast<std::int64_t>(transferred)
                                 : os_error();
  ::CloseHandle(event);
  return result;
}

std::int64_t file_open_sync(const char* path, std::uint32_t flags,
                            std::uint32_t) noexcept {
  DWORD access = 0;
  if (flags & Proactor::kRead) access |= GENERIC_READ;
  if (flags & Proactor::kWrite) access |= GENERIC_WRITE;
  if (flags & Proactor::kAppend) access |= FILE_APPEND_DATA;
  const DWORD disposition =
      (flags & Proactor::kCreate)
          ? ((flags & Proactor::kTruncate) ? CREATE_ALWAYS : OPEN_ALWAYS)
          : ((flags & Proactor::kTruncate) ? TRUNCATE_EXISTING : OPEN_EXISTING);
  wchar_t wpath[1024];
  if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wpath,
                            1024) == 0)
    return os_error();
  HANDLE h = ::CreateFileW(wpath, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, disposition, FILE_FLAG_OVERLAPPED, nullptr);
  return h == INVALID_HANDLE_VALUE
             ? os_error()
             : static_cast<std::int64_t>(
                   reinterpret_cast<std::intptr_t>(h));
}

std::int64_t file_stat_sync(std::intptr_t fd,
                            Proactor::FileStat* out) noexcept {
  BY_HANDLE_FILE_INFORMATION info;
  if (::GetFileInformationByHandle(reinterpret_cast<HANDLE>(fd), &info) == 0)
    return os_error();
  *out = Proactor::FileStat{};
  out->size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) |
              info.nFileSizeLow;
  const std::uint64_t ft =
      (static_cast<std::uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
      info.ftLastWriteTime.dwLowDateTime;
  out->modified_time =
      time::unix_timestamp_from_windows_filetime_ticks(ft);
  out->is_dir = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  return 0;
}

std::int64_t file_sync_sync(std::intptr_t fd, bool) noexcept {
  return ::FlushFileBuffers(reinterpret_cast<HANDLE>(fd)) ? 0 : os_error();
}

std::int64_t file_allocate_sync(std::intptr_t fd, std::uint32_t mode,
                                std::uint64_t offset,
                                std::uint64_t len) noexcept {
  if (mode != 0)
    return -static_cast<std::int64_t>(ERROR_NOT_SUPPORTED);
  constexpr auto kMaxOffset =
      static_cast<std::uint64_t>((std::numeric_limits<LONGLONG>::max)());
  if (len > kMaxOffset || offset > kMaxOffset - len) {
    ::SetLastError(ERROR_ARITHMETIC_OVERFLOW);
    return os_error();
  }
  HANDLE h = reinterpret_cast<HANDLE>(fd);
  LARGE_INTEGER current{};
  LARGE_INTEGER zero{};
  if (!::SetFilePointerEx(h, zero, &current, FILE_CURRENT)) return os_error();
  LARGE_INTEGER end{};
  end.QuadPart = static_cast<LONGLONG>(offset + len);
  if (!::SetFilePointerEx(h, end, nullptr, FILE_BEGIN) || !::SetEndOfFile(h)) {
    const auto error = os_error();
    (void)::SetFilePointerEx(h, current, nullptr, FILE_BEGIN);
    return error;
  }
  (void)::SetFilePointerEx(h, current, nullptr, FILE_BEGIN);
  return 0;
}

std::int64_t file_close_sync(std::intptr_t fd) noexcept {
  return ::CloseHandle(reinterpret_cast<HANDLE>(fd)) ? 0 : os_error();
}
#else
int open_flags_to_posix(std::uint32_t flags) noexcept {
  int out = O_CLOEXEC;
  const bool read = (flags & Proactor::kRead) != 0;
  const bool write = (flags & Proactor::kWrite) != 0;
  out |= read && write ? O_RDWR : (write ? O_WRONLY : O_RDONLY);
  if (flags & Proactor::kCreate) out |= O_CREAT;
  if (flags & Proactor::kTruncate) out |= O_TRUNC;
  if (flags & Proactor::kAppend) out |= O_APPEND;
  return out;
}

std::int64_t file_read_sync(std::intptr_t fd, void* buf, std::uint32_t len,
                            std::uint64_t offset) noexcept {
  if (offset >
      static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)()))
    return -EOVERFLOW;
  const ssize_t n =
      ::pread(static_cast<int>(fd), buf, len, static_cast<off_t>(offset));
  return n < 0 ? os_error() : static_cast<std::int64_t>(n);
}

std::int64_t file_write_sync(std::intptr_t fd, const void* buf,
                             std::uint32_t len,
                             std::uint64_t offset) noexcept {
  if (offset >
      static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)()))
    return -EOVERFLOW;
  const ssize_t n =
      ::pwrite(static_cast<int>(fd), buf, len, static_cast<off_t>(offset));
  return n < 0 ? os_error() : static_cast<std::int64_t>(n);
}

std::int64_t file_open_sync(const char* path, std::uint32_t flags,
                            std::uint32_t mode) noexcept {
  const int fd =
      ::open(path, open_flags_to_posix(flags), static_cast<mode_t>(mode));
  return fd < 0 ? os_error() : static_cast<std::int64_t>(fd);
}

std::int64_t file_stat_sync(std::intptr_t fd,
                            Proactor::FileStat* out) noexcept {
  struct stat st {};
  if (::fstat(static_cast<int>(fd), &st) != 0) return os_error();
  *out = Proactor::FileStat{};
  out->size = static_cast<std::uint64_t>(st.st_size);
#  if defined(HYLE_OS_MACOS)
  out->modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(st.st_mtimespec.tv_sec),
      static_cast<std::uint32_t>(st.st_mtimespec.tv_nsec));
#  else
  out->modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(st.st_mtim.tv_sec),
      static_cast<std::uint32_t>(st.st_mtim.tv_nsec));
#  endif
  out->mode = static_cast<std::uint32_t>(st.st_mode);
  out->is_dir = S_ISDIR(st.st_mode);
  return 0;
}

std::int64_t file_sync_sync(std::intptr_t fd, bool datasync) noexcept {
#  if defined(HYLE_OS_MACOS)
  (void)datasync;
  const int result = ::fsync(static_cast<int>(fd));
#  else
  const int result = datasync ? ::fdatasync(static_cast<int>(fd))
                              : ::fsync(static_cast<int>(fd));
#  endif
  return result == 0 ? 0 : os_error();
}

std::int64_t file_allocate_sync(std::intptr_t fd, std::uint32_t mode,
                                std::uint64_t offset,
                                std::uint64_t len) noexcept {
  if (mode != 0) return -ENOTSUP;
  if (len > static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)()) ||
      offset >
          static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)()) -
              len) {
    return -EOVERFLOW;
  }
#  if defined(HYLE_OS_MACOS)
  const off_t want = static_cast<off_t>(offset + len);
  fstore_t store{};
  store.fst_flags = F_ALLOCATECONTIG;
  store.fst_posmode = F_PEOFPOSMODE;
  store.fst_length = want;
  int result = ::fcntl(static_cast<int>(fd), F_PREALLOCATE, &store);
  if (result < 0) {
    store.fst_flags = F_ALLOCATEALL;
    result = ::fcntl(static_cast<int>(fd), F_PREALLOCATE, &store);
  }
  if (result < 0) return os_error();
  return ::ftruncate(static_cast<int>(fd), want) == 0 ? 0 : os_error();
#  else
  const int result =
      ::posix_fallocate(static_cast<int>(fd), static_cast<off_t>(offset),
                        static_cast<off_t>(len));
  return result == 0 ? 0 : -static_cast<std::int64_t>(result);
#  endif
}

std::int64_t file_close_sync(std::intptr_t fd) noexcept {
  return ::close(static_cast<int>(fd)) == 0 ? 0 : os_error();
}
#endif

template <class Fn>
std::int64_t offload(IoContext& io, Fn fn) noexcept {
  auto handle =
      sched::spawn_blocking(io.blocking_pool(), mem::system_allocator(),
                            std::move(fn));
  auto result = handle.join();
  return result ? *result : -1;
}

template <class Submit>
bool submit_and_suspend(IoOperation& op, bool& submitted,
                        Submit submit) noexcept {
  op.waker = sched::current_waker();
  submitted = submit(to_token(&op));
  if (!submitted) {
    op.result = -1;
    return false;
  }
  sched::suspend();
  return false;
}

void close_fd_sync(std::intptr_t fd) noexcept {
  if (fd < 0) return;
#if defined(HYLE_OS_WINDOWS)
  ::CloseHandle(reinterpret_cast<HANDLE>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}
}  // namespace

bool FileReadOp::await_suspend(std::coroutine_handle<>) noexcept {
  Proactor& p = io->proactor_for_current();
  if (p.has_async_file_data()) {
    return submit_and_suspend(op_, submitted_, [&](std::uint64_t token) {
      return p.submit_read(fd, buf, len, offset, token);
    });
  }
  op_.result = offload(*io, [fd = fd, buf = buf, len = len,
                             offset = offset]() noexcept {
    return file_read_sync(fd, buf, len, offset);
  });
  return false;
}

bool FileWriteOp::await_suspend(std::coroutine_handle<>) noexcept {
  Proactor& p = io->proactor_for_current();
  if (p.has_async_file_data()) {
    return submit_and_suspend(op_, submitted_, [&](std::uint64_t token) {
      return p.submit_write(fd, buf, len, offset, token);
    });
  }
  op_.result = offload(*io, [fd = fd, buf = buf, len = len,
                             offset = offset]() noexcept {
    return file_write_sync(fd, buf, len, offset);
  });
  return false;
}

bool FileStatOp::await_suspend(std::coroutine_handle<>) noexcept {
  (void)io->proactor_for_current();  // assert owner affinity
  op_.result = offload(
      *io, [fd = fd, out = out]() noexcept { return file_stat_sync(fd, out); });
  return false;
}

bool FileSyncOp::await_suspend(std::coroutine_handle<>) noexcept {
  Proactor& p = io->proactor_for_current();
  if (p.backend_kind() == HYLE_IO_BACKEND_IO_URING) {
    return submit_and_suspend(op_, submitted_, [&](std::uint64_t token) {
      return p.submit_fsync(fd, datasync, token);
    });
  }
  op_.result = offload(*io, [fd = fd, datasync = datasync]() noexcept {
    return file_sync_sync(fd, datasync);
  });
  return false;
}

bool FileAllocOp::await_suspend(std::coroutine_handle<>) noexcept {
  Proactor& p = io->proactor_for_current();
  if (p.backend_kind() == HYLE_IO_BACKEND_IO_URING) {
    return submit_and_suspend(op_, submitted_, [&](std::uint64_t token) {
      return p.submit_fallocate(fd, mode, offset, len, token);
    });
  }
  op_.result = offload(*io, [fd = fd, mode = mode, offset = offset,
                             len = len]() noexcept {
    return file_allocate_sync(fd, mode, offset, len);
  });
  return false;
}

bool File::CloseAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  HYLE_FATAL_ASSERT(self->valid() &&
                    sched::current_worker_index() ==
                        static_cast<int>(self->owner_));
  Proactor& p = self->io_->proactor_for_current();
  const std::intptr_t fd = self->fd_;
  self->fd_ = -1;
  if (p.backend_kind() == HYLE_IO_BACKEND_IO_URING) {
    op_.waker = sched::current_waker();
    submitted_ = p.submit_close(fd, to_token(&op_));
    if (!submitted_) {
      // A full/unavailable SQ must not turn ownership transfer above into a
      // descriptor leak. Fall back to the blocking pool and report close(2).
      op_.result =
          offload(*self->io_, [fd]() noexcept { return file_close_sync(fd); });
      return false;
    }
    sched::suspend();
    return false;
  }
  op_.result =
      offload(*self->io_, [fd]() noexcept { return file_close_sync(fd); });
  return false;
}

bool OpenAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  HYLE_FATAL_ASSERT(owner < io->num_workers() &&
                    sched::current_worker_index() == static_cast<int>(owner));
  Proactor& p = io->proactor(owner);
  if (p.backend_kind() == HYLE_IO_BACKEND_IO_URING) {
    return submit_and_suspend(op_, submitted_, [&](std::uint64_t token) {
      return p.submit_open(path, flags, mode, token);
    });
  }
  op_.result = offload(*io, [path = path, flags = flags,
                             mode = mode]() noexcept {
    return file_open_sync(path, flags, mode);
  });
  if (op_.result >= 0 &&
      !p.register_handle(static_cast<std::intptr_t>(op_.result))) {
    (void)file_close_sync(static_cast<std::intptr_t>(op_.result));
    op_.result = -1;
  }
  return false;
}

File::File(IoContext& io, std::intptr_t fd, unsigned owner) noexcept
    : io_(&io), fd_(fd), owner_(owner) {}

File::~File() { close_fd_sync(fd_); }

File::File(File&& o) noexcept
    : io_(o.io_), fd_(std::exchange(o.fd_, -1)), owner_(o.owner_) {}

File& File::operator=(File&& o) noexcept {
  if (this != &o) {
    close_fd_sync(fd_);
    io_ = o.io_;
    fd_ = std::exchange(o.fd_, -1);
    owner_ = o.owner_;
  }
  return *this;
}

}  // namespace hyle::io
