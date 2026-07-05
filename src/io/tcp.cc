#include "hyle/io/tcp.h"

#include <cstring>
#include <utility>

#include "hyle/platform.h"  // HYLE_OS_* (must precede the per-OS socket headers)

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
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
#if defined(HYLE_OS_WINDOWS)
  ::closesocket(static_cast<SOCKET>(fd));
#else
  ::close(static_cast<int>(fd));
#endif
}

// Create an overlapped/nonblocking TCP socket, or -1 on failure.
std::intptr_t make_tcp_socket() noexcept {
#if defined(HYLE_OS_WINDOWS)
  SOCKET s = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  return s == INVALID_SOCKET ? -1 : static_cast<std::intptr_t>(s);
#else
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  return s < 0 ? -1 : static_cast<std::intptr_t>(s);
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
  std::int64_t r = inner.await_resume();
  if (r < 0) return TcpStream{};  // accept failed -> invalid stream
  std::intptr_t new_fd = static_cast<std::intptr_t>(r);
  // Bind the accepted fd to its assigned owner's ring (IOCP association; no-op on
  // io_uring/epoll). If binding fails, drop the fd and return invalid.
  if (!self->io_->bind_fd(assign_owner, new_fd)) {
    close_fd(new_fd);
    return TcpStream{};
  }
  return TcpStream{*self->io_, new_fd, assign_owner};
}

// ---- socket setup ----------------------------------------------------------

bool resolve_addr(const char* host, std::uint16_t port, void* out_ss,
                  std::uint32_t* out_len) noexcept {
  auto* sa = static_cast<sockaddr_in*>(out_ss);
  std::memset(sa, 0, sizeof(sockaddr_in));
  sa->sin_family = AF_INET;
  sa->sin_port = htons(port);  // htons is a MACRO on macOS/BSD -- no :: qualifier (that's
                               // ill-formed on a macro); unqualified resolves the fn on Linux too.
  if (::inet_pton(AF_INET, host, &sa->sin_addr) != 1) return false;
  *out_len = sizeof(sockaddr_in);
  return true;
}

TcpListener tcp_listen(IoContext& io, unsigned owner_worker, const char* host,
                       std::uint16_t port, int backlog) noexcept {
  std::intptr_t fd = make_tcp_socket();
  if (fd < 0) return TcpListener{};

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
    return TcpListener{};
  }
#if defined(HYLE_OS_WINDOWS)
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

TcpStream tcp_socket(IoContext& io, unsigned owner_worker) noexcept {
  std::intptr_t fd = make_tcp_socket();
  if (fd < 0) return TcpStream{};
  if (!io.bind_fd(owner_worker, fd)) {
    close_fd(fd);
    return TcpStream{};
  }
  return TcpStream{io, fd, owner_worker};
}

}  // namespace hyle::io
