#include "hyle/io/udp.h"

#include <cstring>
#include <utility>

#include "hyle/io/tcp.h"     // resolve_addr (shared IPv4 literal -> sockaddr_in)
#include "hyle/platform.h"   // HYLE_OS_* (must precede the per-OS socket headers)

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
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
#if defined(HYLE_OS_WINDOWS)
  ::closesocket(static_cast<SOCKET>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}

// Create an overlapped/nonblocking UDP socket, or -1 on failure.
std::intptr_t make_udp_socket() noexcept {
#if defined(HYLE_OS_WINDOWS)
  SOCKET s = ::WSASocketW(AF_INET, SOCK_DGRAM, IPPROTO_UDP, nullptr, 0,
                          WSA_FLAG_OVERLAPPED);
  return s == INVALID_SOCKET ? -1 : static_cast<std::intptr_t>(s);
#else
  int s = ::socket(AF_INET, SOCK_DGRAM, 0);
  return s < 0 ? -1 : static_cast<std::intptr_t>(s);
#endif
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

// ---- socket setup ----------------------------------------------------------

UdpSocket udp_bind(IoContext& io, unsigned owner_worker, const char* host,
                   std::uint16_t port) noexcept {
  std::intptr_t fd = make_udp_socket();
  if (fd < 0) return UdpSocket{};

  int one = 1;
#if defined(HYLE_OS_WINDOWS)
  ::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&one), sizeof(one));
#else
  ::setsockopt(static_cast<int>(fd), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif

  sockaddr_in sa;
  std::uint32_t len = 0;
  if (!resolve_addr(host, port, &sa, &len)) {
    close_fd(fd);
    return UdpSocket{};
  }
#if defined(HYLE_OS_WINDOWS)
  if (::bind(static_cast<SOCKET>(fd), reinterpret_cast<sockaddr*>(&sa),
             static_cast<int>(len)) != 0) {
    close_fd(fd);
    return UdpSocket{};
  }
#else
  if (::bind(static_cast<int>(fd), reinterpret_cast<sockaddr*>(&sa), len) != 0) {
    close_fd(fd);
    return UdpSocket{};
  }
#endif

  if (!io.bind_fd(owner_worker, fd)) {  // associate with the owner's ring (IOCP)
    close_fd(fd);
    return UdpSocket{};
  }
  return UdpSocket{io, fd, owner_worker};
}

UdpSocket udp_socket(IoContext& io, unsigned owner_worker) noexcept {
  std::intptr_t fd = make_udp_socket();
  if (fd < 0) return UdpSocket{};
  if (!io.bind_fd(owner_worker, fd)) {
    close_fd(fd);
    return UdpSocket{};
  }
  return UdpSocket{io, fd, owner_worker};
}

}  // namespace hyle::io
