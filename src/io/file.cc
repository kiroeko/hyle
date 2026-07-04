#include "hyle/io/file.h"

#include <utility>

#include "hyle/platform.h"  // HYLE_OS_*

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <unistd.h>
#endif

// io-05 file convenience layer implementation: RAII fd/handle lifetime. The
// co_await methods are inline in file.h (they build proactor-op awaitables); this
// file owns the fd's lifetime. Prefer `co_await file.close()` (async on io_uring)
// over dropping the File -- but the destructor closes SYNCHRONOUSLY as a safety net
// so a dropped-without-close File never leaks a descriptor.

namespace hyle::io {
namespace {
void close_fd_sync(std::intptr_t fd) noexcept {
  if (fd < 0) return;
#if defined(HYLE_OS_WINDOWS)
  ::CloseHandle(reinterpret_cast<HANDLE>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}
}  // namespace

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
