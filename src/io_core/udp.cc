#include "hyle/io/udp.h"

#include <cstring>
#include <utility>

#include "hyle/core/environment.h"   // HYLE_OS_* (must precede the per-OS socket headers)
#include "hyle/error/system_error.h"
#include "endpoint_native.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

// io-04 net datagram convenience layer implementation: RAII fd management + UDP
// socket setup. The co_await methods are inline in udp.h (they build net.h
// awaitables); this file owns fd lifetime and the socket/bind syscalls.

namespace hyle::io {
namespace {

void close_fd(std::intptr_t fd) noexcept {
  if (fd < 0) return;
#if HYLE_OS_WINDOWS
  ::closesocket(static_cast<SOCKET>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}

// Create an overlapped/nonblocking UDP socket, or -1 on failure.
int native_family(AddressFamily family) noexcept {
  return family == AddressFamily::kIpv6 ? AF_INET6 : AF_INET;
}

std::intptr_t make_udp_socket(AddressFamily family) noexcept {
  const int af = native_family(family);
#if HYLE_OS_WINDOWS
  SOCKET s = ::WSASocketW(af, SOCK_DGRAM, IPPROTO_UDP, nullptr, 0,
                          WSA_FLAG_OVERLAPPED);
  return s == INVALID_SOCKET ? -1 : static_cast<std::intptr_t>(s);
#else
  int s = ::socket(af, SOCK_DGRAM, 0);
  if (s < 0) return -1;
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

Result<Endpoint> query_local_endpoint(std::intptr_t fd) noexcept {
  sockaddr_storage address{};
#if HYLE_OS_WINDOWS
  int length = sizeof(address);
  if (::getsockname(static_cast<SOCKET>(fd),
                    reinterpret_cast<sockaddr*>(&address), &length) != 0)
    return make_unexpected(socket_error("get UDP local endpoint"));
#else
  socklen_t length = sizeof(address);
  if (::getsockname(static_cast<int>(fd),
                    reinterpret_cast<sockaddr*>(&address), &length) != 0)
    return make_unexpected(socket_error("get UDP local endpoint"));
#endif
  return EndpointNativeAccess::decode(&address,
                                      static_cast<std::uint32_t>(length));
}

Status configure_socket(std::intptr_t fd, AddressFamily family,
                        const UdpSocketOptions& options) noexcept {
  const int reuse = options.reuse_address ? 1 : 0;
#if HYLE_OS_WINDOWS
  if (::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse)) != 0)
#else
  if (::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_REUSEADDR, &reuse,
                   sizeof(reuse)) != 0)
#endif
    return socket_error("set UDP SO_REUSEADDR");

  if (options.broadcast) {
    const int one = 1;
#if HYLE_OS_WINDOWS
    if (::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_BROADCAST,
                     reinterpret_cast<const char*>(&one), sizeof(one)) != 0)
#else
    if (::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_BROADCAST, &one,
                     sizeof(one)) != 0)
#endif
      return socket_error("set UDP SO_BROADCAST");
  }
  if (family == AddressFamily::kIpv6) {
    const int ipv6_only = options.ipv6_only ? 1 : 0;
#if HYLE_OS_WINDOWS
    if (::setsockopt(static_cast<SOCKET>(fd), IPPROTO_IPV6, IPV6_V6ONLY,
                     reinterpret_cast<const char*>(&ipv6_only),
                     sizeof(ipv6_only)) != 0)
#else
    if (::setsockopt(static_cast<int>(fd), IPPROTO_IPV6, IPV6_V6ONLY,
                     &ipv6_only, sizeof(ipv6_only)) != 0)
#endif
      return socket_error("set UDP IPV6_V6ONLY");
  }
  if (options.reuse_port) {
#if defined(SO_REUSEPORT)
    const int one = 1;
#  if HYLE_OS_WINDOWS
    if (::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEPORT,
                     reinterpret_cast<const char*>(&one), sizeof(one)) != 0)
#  else
    if (::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_REUSEPORT, &one,
                     sizeof(one)) != 0)
#  endif
      return socket_error("set UDP SO_REUSEPORT");
#else
    return Status::Unimplemented(
        "UDP reuse-port is not supported by this operating system");
#endif
  }
  return Status{};
}

}  // namespace

// ---- UdpSocket -------------------------------------------------------------

UdpSocket::UdpSocket(IoContext& io, std::intptr_t fd, unsigned owner) noexcept
    : io_(&io), fd_(fd), owner_(owner) {}

UdpSocket::~UdpSocket() { close_fd(fd_); }

UdpSocket::UdpSocket(UdpSocket&& o) noexcept
    : io_(o.io_), fd_(std::exchange(o.fd_, -1)), owner_(o.owner_) {}

UdpSocket& UdpSocket::operator=(UdpSocket&& o) noexcept {
  if (this != &o) {
    close_fd(fd_);
    io_ = o.io_;
    fd_ = std::exchange(o.fd_, -1);
    owner_ = o.owner_;
  }
  return *this;
}

Result<UdpSocket> UdpSocket::open(IoContext& io, unsigned owner_worker,
                                  AddressFamily family,
                                  const UdpSocketOptions& options) noexcept {
  if (owner_worker >= io.num_workers())
    return make_unexpected(Status::OutOfRange("UDP owner worker is out of range"));
  std::intptr_t fd = make_udp_socket(family);
  if (fd < 0) return make_unexpected(socket_error("create UDP socket"));
  Status configured = configure_socket(fd, family, options);
  if (!configured.ok()) {
    close_fd(fd);
    return make_unexpected(std::move(configured));
  }
  if (!io.bind_fd(owner_worker, fd)) {
    Status status = handle_error("associate UDP socket with IO shard");
    close_fd(fd);
    return make_unexpected(std::move(status));
  }
  return UdpSocket{io, fd, owner_worker};
}

Result<UdpSocket> UdpSocket::bind(IoContext& io, unsigned owner_worker,
                                  const Endpoint& endpoint,
                                  const UdpSocketOptions& options) noexcept {
  auto native = EndpointNativeAccess::encode(endpoint);
  if (!native.has_value()) return make_unexpected(native.error());
  auto opened = open(io, owner_worker, endpoint.family(), options);
  if (!opened.has_value()) return make_unexpected(opened.error());
  UdpSocket socket = std::move(*opened);
#if HYLE_OS_WINDOWS
  const int result = ::bind(
      static_cast<SOCKET>(socket.fd_),
      reinterpret_cast<const sockaddr*>(native->storage),
      static_cast<int>(native->size));
#else
  const int result = ::bind(
      static_cast<int>(socket.fd_),
      reinterpret_cast<const sockaddr*>(native->storage),
      static_cast<socklen_t>(native->size));
#endif
  if (result != 0)
    return make_unexpected(socket_error("bind UDP socket"));
  return socket;
}

Result<UdpSocket> UdpSocket::adopt(IoContext& io, unsigned owner_worker,
                                   std::intptr_t native_socket) noexcept {
  if (native_socket < 0)
    return make_unexpected(Status::InvalidArgument("invalid native UDP socket"));
  if (owner_worker >= io.num_workers()) {
    close_fd(native_socket);
    return make_unexpected(Status::OutOfRange("UDP owner worker is out of range"));
  }
  if (!io.bind_fd(owner_worker, native_socket)) {
    Status status = handle_error("associate adopted UDP socket with IO shard");
    close_fd(native_socket);
    return make_unexpected(std::move(status));
  }
  return UdpSocket{io, native_socket, owner_worker};
}

std::intptr_t UdpSocket::release_native_handle() noexcept {
  io_ = nullptr;
  return std::exchange(fd_, -1);
}

Result<Endpoint> UdpSocket::local_endpoint() const noexcept {
  if (!valid())
    return make_unexpected(Status::FailedPrecondition("UDP socket is not open"));
  return query_local_endpoint(fd_);
}

bool UdpSocket::ReceiveAwaitable::await_ready() noexcept {
  inner.proactor = &self->io_->proactor_for_current();
  inner.fd = self->fd_;
  inner.buffer = buffer.data();
  inner.length = static_cast<std::uint32_t>(buffer.size());
  inner.address = native_peer.data();
  inner.address_capacity = static_cast<std::uint32_t>(native_peer.size());
  inner.allocator = self->io_->operation_allocator();
  inner.options = options;
  return inner.await_ready();
}

DatagramResult UdpSocket::ReceiveAwaitable::await_resume() noexcept {
  TransferResult transfer = detail::transfer_from_native(
      inner.await_resume(), detail::kNetworkNativeDomain, "UDP receive", false);
  if (!transfer.ok())
    return DatagramResult{transfer.bytes, Endpoint{}, std::move(transfer.status)};
  auto peer = EndpointNativeAccess::decode(native_peer.data(),
                                           static_cast<std::uint32_t>(
                                               native_peer.size()));
  if (!peer.has_value())
    return DatagramResult{transfer.bytes, Endpoint{}, peer.error()};
  return DatagramResult{transfer.bytes, *peer, Status{}};
}

bool UdpSocket::SendAwaitable::await_ready() noexcept {
  auto native = EndpointNativeAccess::encode(destination);
  if (!native.has_value()) {
    preparation = native.error();
    return true;
  }
  std::memcpy(native_destination.data(), native->storage, native->size);
  native_length = native->size;
  inner.proactor = &self->io_->proactor_for_current();
  inner.fd = self->fd_;
  inner.buffer = buffer.data();
  inner.length = static_cast<std::uint32_t>(buffer.size());
  inner.address = native_destination.data();
  inner.address_length = native_length;
  inner.allocator = self->io_->operation_allocator();
  inner.options = options;
  return inner.await_ready();
}

TransferResult UdpSocket::SendAwaitable::await_resume() noexcept {
  if (!preparation.ok()) return TransferResult{0, preparation, false};
  return detail::transfer_from_native(inner.await_resume(),
                                      detail::kNetworkNativeDomain,
                                      "UDP send", false);
}

}  // namespace hyle::io
