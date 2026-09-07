#include "hyle/io/tcp.h"

#include <cerrno>
#include <cstring>
#include <utility>

#include "hyle/core/environment.h"  // HYLE_OS_* (must precede the per-OS socket headers)
#include "hyle/error/system_error.h"
#include "endpoint_native.h"

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

Status socket_error(const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_winsock(::WSAGetLastError(), operation);
#else
  return status_from_errno(errno, operation);
#endif
}

Status handle_error(const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_win32(::GetLastError(), operation);
#else
  return status_from_errno(errno, operation);
#endif
}

Result<Endpoint> query_endpoint(std::intptr_t fd, bool peer) noexcept {
  sockaddr_storage address{};
#if HYLE_OS_WINDOWS
  int size = sizeof(address);
  const int result = peer ? ::getpeername(static_cast<SOCKET>(fd),
                                          reinterpret_cast<sockaddr*>(&address),
                                          &size)
                          : ::getsockname(static_cast<SOCKET>(fd),
                                         reinterpret_cast<sockaddr*>(&address),
                                         &size);
#else
  socklen_t size = sizeof(address);
  const int result = peer ? ::getpeername(static_cast<int>(fd),
                                          reinterpret_cast<sockaddr*>(&address),
                                          &size)
                          : ::getsockname(static_cast<int>(fd),
                                         reinterpret_cast<sockaddr*>(&address),
                                         &size);
#endif
  if (result != 0)
    return make_unexpected(socket_error(peer ? "getpeername" : "getsockname"));
  return EndpointNativeAccess::decode(&address,
                                      static_cast<std::uint32_t>(size));
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

Result<TcpStream> TcpStream::open(IoContext& io, unsigned owner_worker,
                                  AddressFamily family) noexcept {
  if (owner_worker >= io.num_workers())
    return make_unexpected(Status::OutOfRange("TCP owner worker is out of range"));
  std::intptr_t fd = make_tcp_socket(family);
  if (fd < 0) return make_unexpected(socket_error("create TCP socket"));
  if (!io.bind_fd(owner_worker, fd)) {
    Status status = handle_error("associate TCP socket with IO shard");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  return TcpStream{io, fd, owner_worker};
}

Result<TcpStream> TcpStream::adopt(IoContext& io, unsigned owner_worker,
                                    std::intptr_t native_socket) noexcept {
  if (native_socket < 0)
    return make_unexpected(Status::InvalidArgument("invalid native TCP socket"));
  if (owner_worker >= io.num_workers()) {
    close_fd(native_socket);
    return make_unexpected(Status::OutOfRange("TCP owner worker is out of range"));
  }
  if (!io.bind_fd(owner_worker, native_socket)) {
    Status status = handle_error("associate adopted TCP socket with IO shard");
    close_fd(native_socket);
    return make_unexpected(std::move(status));
  }
  return TcpStream{io, native_socket, owner_worker};
}

std::intptr_t TcpStream::release_native_handle() noexcept {
  io_ = nullptr;
  return std::exchange(fd_, -1);
}

Status TcpStream::set_nodelay(bool enabled) noexcept {
  if (!valid()) return Status::FailedPrecondition("TCP stream is not open");
  const int value = enabled ? 1 : 0;
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd_), IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&value), sizeof(value)) != 0)
    return socket_error("set TCP_NODELAY");
#else
  if (::setsockopt(static_cast<int>(fd_), IPPROTO_TCP, TCP_NODELAY, &value,
                   sizeof(value)) != 0)
    return socket_error("set TCP_NODELAY");
#endif
  return Status{};
}

Status TcpStream::set_keepalive(bool enabled) noexcept {
  if (!valid()) return Status::FailedPrecondition("TCP stream is not open");
  const int value = enabled ? 1 : 0;
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd_), SOL_SOCKET, SO_KEEPALIVE,
                   reinterpret_cast<const char*>(&value), sizeof(value)) != 0)
    return socket_error("set SO_KEEPALIVE");
#else
  if (::setsockopt(static_cast<int>(fd_), SOL_SOCKET, SO_KEEPALIVE, &value,
                   sizeof(value)) != 0)
    return socket_error("set SO_KEEPALIVE");
#endif
  return Status{};
}

Status TcpStream::shutdown_write() noexcept {
  if (!valid()) return Status::FailedPrecondition("TCP stream is not open");
#if HYLE_OS_WINDOWS
  if (::shutdown(static_cast<SOCKET>(fd_), SD_SEND) != 0)
    return socket_error("shutdown TCP write side");
#else
  if (::shutdown(static_cast<int>(fd_), SHUT_WR) != 0)
    return socket_error("shutdown TCP write side");
#endif
  return Status{};
}

Result<Endpoint> TcpStream::local_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("TCP stream is not open"));
  return query_endpoint(fd_, false);
}

Result<Endpoint> TcpStream::peer_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("TCP stream is not open"));
  return query_endpoint(fd_, true);
}

bool TcpStream::ConnectAwaitable::await_ready() noexcept {
  auto native = EndpointNativeAccess::encode(endpoint);
  if (!native.has_value()) {
    preparation = native.error();
    return true;
  }
  std::memcpy(native_address.data(), native->storage, native->size);
  native_length = native->size;
  inner.proactor = &self->io_->proactor_for_current();
  inner.fd = self->fd_;
  inner.address = native_address.data();
  inner.address_length = native_length;
  inner.allocator = self->io_->operation_allocator();
  inner.options = options;
  return inner.await_ready();
}

Status TcpStream::ConnectAwaitable::await_resume() noexcept {
  if (!preparation.ok()) return preparation;
  return detail::transfer_from_native(inner.await_resume(),
                                      detail::kNetworkNativeDomain,
                                      "TCP connect", false)
      .status;
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

Result<TcpListener> TcpListener::bind(IoContext& io, unsigned owner_worker,
                                      const Endpoint& endpoint,
                                      const TcpListenOptions& options) noexcept {
  if (owner_worker >= io.num_workers())
    return make_unexpected(Status::OutOfRange("TCP owner worker is out of range"));
  if (options.backlog < 0)
    return make_unexpected(Status::InvalidArgument("TCP backlog cannot be negative"));
  auto native = EndpointNativeAccess::encode(endpoint);
  if (!native.has_value()) return make_unexpected(native.error());
  std::intptr_t fd = make_tcp_socket(endpoint.family());
  if (fd < 0) return make_unexpected(socket_error("create TCP listener"));

  const int reuse = options.reuse_address ? 1 : 0;
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse)) != 0) {
#else
  if (::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_REUSEADDR, &reuse,
                   sizeof(reuse)) != 0) {
#endif
    Status status = socket_error("set SO_REUSEADDR");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }

  if (endpoint.family() == AddressFamily::kIpv6) {
    const int ipv6_only = options.ipv6_only ? 1 : 0;
#if HYLE_OS_WINDOWS
    if (::setsockopt(static_cast<SOCKET>(fd), IPPROTO_IPV6, IPV6_V6ONLY,
                     reinterpret_cast<const char*>(&ipv6_only),
                     sizeof(ipv6_only)) != 0) {
#else
    if (::setsockopt(static_cast<int>(fd), IPPROTO_IPV6, IPV6_V6ONLY,
                     &ipv6_only, sizeof(ipv6_only)) != 0) {
#endif
      Status status = socket_error("set IPV6_V6ONLY");
      close_fd(fd);
      return make_unexpected(std::move(status));
    }
  }

  if (options.reuse_port) {
#if defined(SO_REUSEPORT)
    const int one = 1;
#  if HYLE_OS_WINDOWS
    const int result = ::setsockopt(
        static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEPORT,
        reinterpret_cast<const char*>(&one), sizeof(one));
#  else
    const int result = ::setsockopt(static_cast<int>(fd), SOL_SOCKET,
                                    SO_REUSEPORT, &one, sizeof(one));
#  endif
    if (result != 0) {
      Status status = socket_error("set SO_REUSEPORT");
      close_fd(fd);
      return make_unexpected(std::move(status));
    }
#else
    close_fd(fd);
    return make_unexpected(Status::Unimplemented(
        "reuse-port is not supported by this operating system"));
#endif
  }

#if HYLE_OS_WINDOWS
  if (::bind(static_cast<SOCKET>(fd),
             reinterpret_cast<const sockaddr*>(native->storage),
             static_cast<int>(native->size)) != 0 ||
      ::listen(static_cast<SOCKET>(fd), options.backlog) != 0) {
#else
  if (::bind(static_cast<int>(fd),
             reinterpret_cast<const sockaddr*>(native->storage),
             static_cast<socklen_t>(native->size)) != 0 ||
      ::listen(static_cast<int>(fd), options.backlog) != 0) {
#endif
    Status status = socket_error("bind/listen TCP socket");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  if (!io.bind_fd(owner_worker, fd)) {
    Status status = handle_error("associate TCP listener with IO shard");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  return TcpListener{io, fd, owner_worker};
}

Result<Endpoint> TcpListener::local_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("TCP listener is not open"));
  return query_endpoint(fd_, false);
}

Result<TcpStream> TcpListener::AcceptAwaitable::await_resume() noexcept {
  return self->finish_accept(inner.await_resume(), assign_owner);
}

Result<TcpStream> TcpListener::finish_accept(std::int64_t result,
                                             unsigned assign_owner) noexcept {
  if (result < 0) {
    return make_unexpected(
        detail::transfer_from_native(result, detail::kNetworkNativeDomain,
                                     "TCP accept", false)
            .status);
  }
  std::intptr_t new_fd = static_cast<std::intptr_t>(result);
#if HYLE_OS_MACOS
  const int no_sigpipe = 1;
  if (::setsockopt(static_cast<int>(new_fd), SOL_SOCKET, SO_NOSIGPIPE,
                   &no_sigpipe, sizeof(no_sigpipe)) != 0) {
    close_fd(new_fd);
    return make_unexpected(
        socket_error("set SO_NOSIGPIPE on accepted TCP socket"));
  }
#endif
  // Bind the accepted fd to its assigned owner's ring (IOCP association; no-op on
  // io_uring/epoll). If binding fails, drop the fd and return invalid.
  if (!io_->bind_fd(assign_owner, new_fd)) {
    Status status = handle_error("associate accepted TCP socket with IO shard");
    close_fd(new_fd);
    return make_unexpected(std::move(status));
  }
  return TcpStream{*io_, new_fd, assign_owner};
}

sched::Task<TransferResult> read_exact(mem::AllocatorRef, TcpStream& stream,
                                      std::span<std::byte> buffer,
                                      OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    TransferResult part =
        co_await stream.read_some(buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok() || part.eof) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0) {
      co_return TransferResult{
          done, Status::Unavailable("TCP read made no forward progress"), false};
    }
  }
  co_return TransferResult{done, Status{}, false};
}

sched::Task<TransferResult> write_all(mem::AllocatorRef, TcpStream& stream,
                                     std::span<const std::byte> buffer,
                                     OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    TransferResult part =
        co_await stream.write_some(buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok()) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0) {
      co_return TransferResult{
          done, Status::Unavailable("TCP write made no forward progress"), false};
    }
  }
  co_return TransferResult{done, Status{}, false};
}

}  // namespace hyle::io
