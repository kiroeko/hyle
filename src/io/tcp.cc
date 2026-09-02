#include "hyle/io/tcp.h"

#include <cstring>
#include <utility>

#include "hyle/core/environment.h"  // HYLE_OS_* (must precede the per-OS socket headers)

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

// io-04 net convenience layer implementation: RAII fd management + cross-platform
// TCP socket setup. The co_await methods themselves are inline in tcp.h (they
// just build net.h awaitables); this file owns fd lifetime and the socket/bind/
// listen syscalls behind tcp_listen / tcp_socket.

namespace hyle::io {
namespace {

// Close a socket fd on either platform.
void close_fd(std::intptr_t fd) noexcept {
  if (fd < 0) return;
#if HYLE_OS_WINDOWS
  ::closesocket(static_cast<SOCKET>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}

// Create an overlapped/nonblocking TCP socket, or -1 on failure.
int native_family(AddressFamily family) noexcept {
  return family == AddressFamily::kIpv6 ? AF_INET6 : AF_INET;
}

std::intptr_t make_tcp_socket(AddressFamily family) noexcept {
  const int af = native_family(family);
#if HYLE_OS_WINDOWS
  SOCKET s = ::WSASocketW(af, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  return s == INVALID_SOCKET ? -1 : static_cast<std::intptr_t>(s);
#else
  int s = ::socket(af, SOCK_STREAM, 0);
  if (s < 0) return -1;
#  if HYLE_OS_MACOS
  // macOS has no MSG_NOSIGNAL. Without SO_NOSIGPIPE a normal peer-close race
  // during TLS close_notify can terminate the entire process instead of
  // completing the pending write with EPIPE.
  const int no_sigpipe = 1;
  if (::setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe,
                   sizeof(no_sigpipe)) != 0) {
    ::close(s);
    return -1;
  }
#  endif
  const int flags = ::fcntl(s, F_GETFL, 0);
  if (flags < 0 || ::fcntl(s, F_SETFL, flags | O_NONBLOCK) != 0) {
    ::close(s);
    return -1;
  }
  return static_cast<std::intptr_t>(s);
#endif
}

}  // namespace

// ---- TcpStream -------------------------------------------------------------

TcpStream::TcpStream(IoContext& io, std::intptr_t fd, unsigned owner) noexcept
    : io_(&io), fd_(fd), owner_(owner) {}

TcpStream::~TcpStream() { close_fd(fd_); }

TcpStream::TcpStream(TcpStream&& o) noexcept
    : io_(o.io_), fd_(std::exchange(o.fd_, -1)), owner_(o.owner_) {}

TcpStream& TcpStream::operator=(TcpStream&& o) noexcept {
  if (this != &o) {
    close_fd(fd_);
    io_ = o.io_;
    fd_ = std::exchange(o.fd_, -1);
    owner_ = o.owner_;
  }
  return *this;
}

bool TcpStream::set_nodelay(bool enabled) noexcept {
  if (!valid()) return false;
  const int value = enabled ? 1 : 0;
#if HYLE_OS_WINDOWS
  return ::setsockopt(static_cast<SOCKET>(fd_), IPPROTO_TCP, TCP_NODELAY,
                      reinterpret_cast<const char*>(&value),
                      sizeof(value)) == 0;
#else
  return ::setsockopt(static_cast<int>(fd_), IPPROTO_TCP, TCP_NODELAY, &value,
                      sizeof(value)) == 0;
#endif
}

bool TcpStream::set_keepalive(bool enabled) noexcept {
  if (!valid()) return false;
  const int value = enabled ? 1 : 0;
#if HYLE_OS_WINDOWS
  return ::setsockopt(static_cast<SOCKET>(fd_), SOL_SOCKET, SO_KEEPALIVE,
                      reinterpret_cast<const char*>(&value),
                      sizeof(value)) == 0;
#else
  return ::setsockopt(static_cast<int>(fd_), SOL_SOCKET, SO_KEEPALIVE, &value,
                      sizeof(value)) == 0;
#endif
}

bool TcpStream::shutdown_write() noexcept {
  if (!valid()) return false;
#if HYLE_OS_WINDOWS
  return ::shutdown(static_cast<SOCKET>(fd_), SD_SEND) == 0;
#else
  return ::shutdown(static_cast<int>(fd_), SHUT_WR) == 0;
#endif
}

// ---- TcpListener -----------------------------------------------------------

TcpListener::TcpListener(IoContext& io, std::intptr_t fd, unsigned owner) noexcept
    : io_(&io), fd_(fd), owner_(owner) {}

TcpListener::~TcpListener() { close_fd(fd_); }

TcpListener::TcpListener(TcpListener&& o) noexcept
    : io_(o.io_), fd_(std::exchange(o.fd_, -1)), owner_(o.owner_) {}

TcpListener& TcpListener::operator=(TcpListener&& o) noexcept {
  if (this != &o) {
    close_fd(fd_);
    io_ = o.io_;
    fd_ = std::exchange(o.fd_, -1);
    owner_ = o.owner_;
  }
  return *this;
}

TcpStream TcpListener::AcceptAwaitable::await_resume() noexcept {
  return self->finish_accept(inner.await_resume(), assign_owner);
}

TcpStream TcpListener::TimedAcceptAwaitable::await_resume() noexcept {
  return self->finish_accept(inner.await_resume(), assign_owner);
}

TcpStream TcpListener::CancelAcceptAwaitable::await_resume() noexcept {
  return self->finish_accept(inner.await_resume(), assign_owner);
}

TcpStream TcpListener::finish_accept(std::int64_t result,
                                     unsigned assign_owner) noexcept {
  if (result < 0) return TcpStream{};
  std::intptr_t new_fd = static_cast<std::intptr_t>(result);
#if HYLE_OS_MACOS
  const int no_sigpipe = 1;
  if (::setsockopt(static_cast<int>(new_fd), SOL_SOCKET, SO_NOSIGPIPE,
                   &no_sigpipe, sizeof(no_sigpipe)) != 0) {
    close_fd(new_fd);
    return TcpStream{};
  }
#endif
  // Bind the accepted fd to its assigned owner's ring (IOCP association; no-op on
  // io_uring/epoll). If binding fails, drop the fd and return invalid.
  if (!io_->bind_fd(assign_owner, new_fd)) {
    close_fd(new_fd);
    return TcpStream{};
  }
  return TcpStream{*io_, new_fd, assign_owner};
}

// ---- socket setup ----------------------------------------------------------

bool resolve_addr(const char* host, std::uint16_t port, void* out_ss,
                  std::uint32_t* out_len) noexcept {
  if (host == nullptr || out_ss == nullptr || out_len == nullptr) return false;
  auto* storage = static_cast<sockaddr_storage*>(out_ss);
  std::memset(storage, 0, sizeof(*storage));
  auto* v4 = reinterpret_cast<sockaddr_in*>(storage);
  v4->sin_family = AF_INET;
  v4->sin_port = htons(port);
  if (::inet_pton(AF_INET, host, &v4->sin_addr) == 1) {
    *out_len = sizeof(sockaddr_in);
    return true;
  }
  std::memset(storage, 0, sizeof(*storage));
  auto* v6 = reinterpret_cast<sockaddr_in6*>(storage);
  v6->sin6_family = AF_INET6;
  v6->sin6_port = htons(port);
  if (::inet_pton(AF_INET6, host, &v6->sin6_addr) == 1) {
    *out_len = sizeof(sockaddr_in6);
    return true;
  }
  *out_len = 0;
  return false;
}

TcpListener tcp_listen(IoContext& io, unsigned owner_worker, const char* host,
                       std::uint16_t port, int backlog) noexcept {
  sockaddr_storage sa{};
  std::uint32_t len = 0;
  if (!resolve_addr(host, port, &sa, &len)) return TcpListener{};
  const AddressFamily family =
      sa.ss_family == AF_INET6 ? AddressFamily::kIpv6 : AddressFamily::kIpv4;
  std::intptr_t fd = make_tcp_socket(family);
  if (fd < 0) return TcpListener{};

  int one = 1;
#if HYLE_OS_WINDOWS
  ::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&one), sizeof(one));
#else
  ::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif

#if HYLE_OS_WINDOWS
  if (::bind(static_cast<SOCKET>(fd), reinterpret_cast<sockaddr*>(&sa),
             static_cast<int>(len)) != 0 ||
      ::listen(static_cast<SOCKET>(fd), backlog) != 0) {
    close_fd(fd);
    return TcpListener{};
  }
#else
  if (::bind(static_cast<int>(fd), reinterpret_cast<sockaddr*>(&sa), len) != 0 ||
      ::listen(static_cast<int>(fd), backlog) != 0) {
    close_fd(fd);
    return TcpListener{};
  }
#endif

  if (!io.bind_fd(owner_worker, fd)) {  // associate with the owner's ring (IOCP)
    close_fd(fd);
    return TcpListener{};
  }
  return TcpListener{io, fd, owner_worker};
}

TcpStream tcp_socket(IoContext& io, unsigned owner_worker,
                     AddressFamily family) noexcept {
  std::intptr_t fd = make_tcp_socket(family);
  if (fd < 0) return TcpStream{};
  if (!io.bind_fd(owner_worker, fd)) {
    close_fd(fd);
    return TcpStream{};
  }
  return TcpStream{io, fd, owner_worker};
}

TcpStream tcp_socket(IoContext& io, unsigned owner_worker) noexcept {
  return tcp_socket(io, owner_worker, AddressFamily::kIpv4);
}

sched::Task<std::int64_t> tcp_read_exact(mem::AllocatorRef, TcpStream& stream,
                                         void* buffer,
                                         std::uint32_t len) noexcept {
  HYLE_FATAL_ASSERT(buffer != nullptr || len == 0);
  auto* bytes = static_cast<unsigned char*>(buffer);
  std::uint32_t done = 0;
  while (done < len) {
    const std::int64_t n = co_await stream.read(bytes + done, len - done);
    if (n <= 0) co_return n < 0 ? n : static_cast<std::int64_t>(done);
    done += static_cast<std::uint32_t>(n);
  }
  co_return static_cast<std::int64_t>(done);
}

sched::Task<std::int64_t> tcp_write_all(mem::AllocatorRef, TcpStream& stream,
                                        const void* buffer,
                                        std::uint32_t len) noexcept {
  HYLE_FATAL_ASSERT(buffer != nullptr || len == 0);
  const auto* bytes = static_cast<const unsigned char*>(buffer);
  std::uint32_t done = 0;
  while (done < len) {
    const std::int64_t n = co_await stream.write(bytes + done, len - done);
    if (n <= 0) co_return n < 0 ? n : static_cast<std::int64_t>(done);
    done += static_cast<std::uint32_t>(n);
  }
  co_return static_cast<std::int64_t>(done);
}

sched::Task<std::int64_t> tcp_read_exact_until(
    mem::AllocatorRef alloc, TcpStream& stream, void* buffer, std::uint32_t len,
    std::uint64_t deadline_ns) noexcept {
  HYLE_FATAL_ASSERT(buffer != nullptr || len == 0);
  auto* bytes = static_cast<unsigned char*>(buffer);
  std::uint32_t done = 0;
  while (done < len) {
    const std::int64_t n = co_await stream.read_until(
        bytes + done, len - done, alloc, deadline_ns);
    if (n <= 0) co_return n < 0 ? n : static_cast<std::int64_t>(done);
    done += static_cast<std::uint32_t>(n);
  }
  co_return static_cast<std::int64_t>(done);
}

sched::Task<std::int64_t> tcp_write_all_until(
    mem::AllocatorRef alloc, TcpStream& stream, const void* buffer,
    std::uint32_t len, std::uint64_t deadline_ns) noexcept {
  HYLE_FATAL_ASSERT(buffer != nullptr || len == 0);
  const auto* bytes = static_cast<const unsigned char*>(buffer);
  std::uint32_t done = 0;
  while (done < len) {
    const std::int64_t n = co_await stream.write_until(
        bytes + done, len - done, alloc, deadline_ns);
    if (n <= 0) co_return n < 0 ? n : static_cast<std::int64_t>(done);
    done += static_cast<std::uint32_t>(n);
  }
  co_return static_cast<std::int64_t>(done);
}

sched::Task<std::int64_t> tcp_read_exact_cancel(
    mem::AllocatorRef, TcpStream& stream, void* buffer, std::uint32_t len,
    const sched::CancelToken* token) noexcept {
  HYLE_FATAL_ASSERT(buffer != nullptr || len == 0);
  auto* bytes = static_cast<unsigned char*>(buffer);
  std::uint32_t done = 0;
  while (done < len) {
    const std::int64_t n =
        co_await stream.read_cancel(bytes + done, len - done, token);
    if (n <= 0) co_return n < 0 ? n : static_cast<std::int64_t>(done);
    done += static_cast<std::uint32_t>(n);
  }
  co_return static_cast<std::int64_t>(done);
}

sched::Task<std::int64_t> tcp_write_all_cancel(
    mem::AllocatorRef, TcpStream& stream, const void* buffer, std::uint32_t len,
    const sched::CancelToken* token) noexcept {
  HYLE_FATAL_ASSERT(buffer != nullptr || len == 0);
  const auto* bytes = static_cast<const unsigned char*>(buffer);
  std::uint32_t done = 0;
  while (done < len) {
    const std::int64_t n =
        co_await stream.write_cancel(bytes + done, len - done, token);
    if (n <= 0) co_return n < 0 ? n : static_cast<std::int64_t>(done);
    done += static_cast<std::uint32_t>(n);
  }
  co_return static_cast<std::int64_t>(done);
}

}  // namespace hyle::io
