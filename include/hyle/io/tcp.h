#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/io/net_multishot.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-04 net convenience layer: RAII TCP socket wrappers over the raw co_await ops
// in net.h. TcpStream owns a connected socket fd and exposes co_await read/write;
// TcpListener binds/listens and exposes co_await accept -> TcpStream. Both are
// shard-per-core aware: a socket is bound to an owner worker (its ring), and the
// fiber servicing it must be pinned there (Scheduler::spawn_on) so submit + reap
// stay on one core. The wrappers carry the owner index so accept() can hand a new
// connection to the right core.
//
// These are thin: fd ownership (RAII close), the co_await methods delegate to
// net.h's RecvOp/SendOp/AcceptOp/ConnectOp, and a couple of cross-platform socket
// setup helpers (tcp_listen / tcp_connect) live in tcp.cc.

namespace hyle::io {

// A connected TCP socket. Move-only; closes the fd on destruction. read()/write()
// are co_await-able and MUST be awaited from the fiber pinned to owner_worker().
class TcpStream {
 public:
  TcpStream() noexcept = default;
  HYLE_IO_API TcpStream(IoContext& io, std::intptr_t fd,
                        unsigned owner_worker) noexcept;
  HYLE_IO_API ~TcpStream();
  HYLE_IO_API TcpStream(TcpStream&& o) noexcept;
  HYLE_IO_API TcpStream& operator=(TcpStream&& o) noexcept;
  HYLE_DISABLE_COPY(TcpStream);

  bool valid() const noexcept { return fd_ >= 0; }
  std::intptr_t fd() const noexcept { return fd_; }
  unsigned owner_worker() const noexcept { return owner_; }

  // co_await stream.read(buf, len) -> bytes read (>=0; 0 = peer closed) or
  // negative errno. Awaited from the fiber pinned to owner_worker().
  RecvOp read(void* buf, std::uint32_t len) noexcept {
    return RecvOp{&io_->proactor_for_current(), fd_, buf, len, 0};
  }
  // co_await stream.write(buf, len) -> bytes written or negative errno.
  SendOp write(const void* buf, std::uint32_t len) noexcept {
    return SendOp{&io_->proactor_for_current(), fd_, buf, len, 0};
  }

  // Zero-copy multishot receive (io_uring only): drive with `co_await
  // stream.read_multishot(bgid, alloc).next()`. The kernel picks buffers from group
  // `bgid`'s provided-buffer ring (register it once via the proactor's
  // register_buf_ring before calling). One arm delivers a stream of RecvBuf until the
  // peer closes. `alloc` backs the stream's heap slot (use the connection's
  // allocator). Awaited from the fiber pinned to owner_worker(). NOT on IOCP/epoll
  // (no provided-buffer ring there) -- register_buf_ring returns false, so the arm
  // fails and next() yields an error chunk; callers targeting those backends use
  // read() instead.
  MultishotRecvStream read_multishot(int bgid, mem::Allocator alloc) noexcept {
    return MultishotRecvStream{&io_->proactor_for_current(), fd_, bgid, alloc};
  }

  // co_await stream.connect(&ss, len) -> 0 on success or negative errno, where
  // (ss, len) is a sockaddr filled by resolve_addr(). Awaited from the fiber
  // pinned to owner_worker(). The sockaddr must outlive the co_await (keep it on
  // the coroutine frame). Use for a client socket from tcp_socket().
  ConnectOp connect(const void* addr, std::uint32_t addrlen) noexcept {
    return ConnectOp{&io_->proactor_for_current(), fd_, addr, addrlen};
  }

 private:
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

// A listening TCP socket. co_await accept() yields the next connection as a
// TcpStream bound to `accept_owner` (round-robin across workers by default so new
// connections shard across cores). Move-only; closes on destruction.
class TcpListener {
 public:
  TcpListener() noexcept = default;
  HYLE_IO_API TcpListener(IoContext& io, std::intptr_t fd,
                          unsigned owner_worker) noexcept;
  HYLE_IO_API ~TcpListener();
  HYLE_IO_API TcpListener(TcpListener&& o) noexcept;
  HYLE_IO_API TcpListener& operator=(TcpListener&& o) noexcept;
  HYLE_DISABLE_COPY(TcpListener);

  bool valid() const noexcept { return fd_ >= 0; }
  std::intptr_t fd() const noexcept { return fd_; }
  unsigned owner_worker() const noexcept { return owner_; }

  // An awaitable that co_awaits AcceptOp then wraps the accepted fd (binding it to
  // `assign_owner`'s ring) into a TcpStream. Returned by accept(); co_await it.
  struct AcceptAwaitable {
    TcpListener* self;
    unsigned assign_owner;
    AcceptOp inner;  // constructed in accept() with the listener's ring + fd

    bool await_ready() noexcept { return inner.await_ready(); }
    bool await_suspend(std::coroutine_handle<> h) noexcept {
      return inner.await_suspend(h);
    }
    // Bind the accepted fd to its owner's ring and wrap it. On failure (negative
    // result) returns an invalid TcpStream (valid() == false).
    HYLE_IO_API TcpStream await_resume() noexcept;
  };

  // co_await listener.accept() -> TcpStream. The new connection is bound to
  // `assign_owner` (its fibers must be spawn_on(assign_owner) to service it on
  // that core). accept() itself is awaited from the fiber pinned to the
  // listener's owner_worker().
  AcceptAwaitable accept(unsigned assign_owner) noexcept {
    return AcceptAwaitable{this, assign_owner,
                           AcceptOp{&io_->proactor_for_current(), fd_}};
  }

 private:
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

// ---- cross-platform socket setup (tcp.cc) ----------------------------------
// Create a listening TCP socket bound to `host`:`port` (host e.g. "127.0.0.1" or
// "0.0.0.0"), associate it with `owner_worker`'s ring, and return a TcpListener.
// Returns an invalid TcpListener (valid()==false) on failure. `backlog` is the
// listen queue depth.
HYLE_IO_API TcpListener tcp_listen(IoContext& io, unsigned owner_worker,
                                   const char* host, std::uint16_t port,
                                   int backlog = 128) noexcept;

// Create a client TCP socket bound to `owner_worker`'s ring (unconnected). The
// caller resolves the target with resolve_addr() then `co_await stream.connect(
// &ss, len)` from the fiber pinned to owner_worker. Returns an invalid TcpStream
// on failure.
HYLE_IO_API TcpStream tcp_socket(IoContext& io, unsigned owner_worker) noexcept;

// Fill `out_ss` (a sockaddr_in-sized buffer) + `out_len` from `host`:`port`
// (host e.g. "127.0.0.1"). Returns false if the host isn't a valid IPv4 literal.
// Keep `out_ss` alive across the co_await connect (it's on the coroutine frame).
HYLE_IO_API bool resolve_addr(const char* host, std::uint16_t port, void* out_ss,
                              std::uint32_t* out_len) noexcept;

}  // namespace hyle::io
