#pragma once

#include <cstdint>

#include "hyle/core/assert.h"
#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/io/net_cancel.h"
#include "hyle/io/net_timeout.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-04 net, datagram half: RAII UDP socket wrapper over the raw co_await ops in
// net.h. Unlike TCP there is no connection/accept -- a single UdpSocket both
// receives (recv_from, learning each datagram's sender) and sends (send_to, to an
// explicit destination). Shard-per-core aware exactly like TcpStream: the socket
// is bound to an owner worker's ring, and the fiber servicing it must be pinned
// there (Scheduler::spawn_on) so submit + reap stay on one core.
//
// Thin: fd ownership (RAII close) + the co_await methods delegate to net.h's
// RecvFromOp/SendToOp; udp_bind / udp_socket (socket+bind setup) live in udp.cc.

namespace hyle::io {

// A bound (or unbound-but-sendable) UDP socket. Move-only; closes the fd on
// destruction. recv_from()/send_to() are co_await-able and MUST be awaited from
// the fiber pinned to owner_worker().
class UdpSocket {
 public:
  UdpSocket() noexcept = default;
  HYLE_IO_API UdpSocket(IoContext& io, std::intptr_t fd,
                        unsigned owner_worker) noexcept;
  HYLE_IO_API ~UdpSocket();
  HYLE_IO_API UdpSocket(UdpSocket&& o) noexcept;
  HYLE_IO_API UdpSocket& operator=(UdpSocket&& o) noexcept;
  UdpSocket(const UdpSocket&) = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;

  bool valid() const noexcept { return fd_ >= 0; }
  std::intptr_t fd() const noexcept { return fd_; }
  unsigned owner_worker() const noexcept { return owner_; }

  // co_await sock.recv_from(buf, len, &ss) -> bytes received (>=0) or negative
  // errno; the sender's address is written into `peer` (a sockaddr_storage-sized
  // buffer). Both `buf` and `peer` must outlive the co_await (put them on the
  // coroutine frame). Read the peer length from peer->ss_family after resume.
  RecvFromOp recv_from(void* buf, std::uint32_t len, void* peer) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return RecvFromOp{&io_->proactor_for_current(), fd_, buf, len, peer,
                      kSockaddrCap};
  }
  TimedRecvFromOp recv_from_until(void* buf, std::uint32_t len, void* peer,
                                  mem::AllocatorRef alloc,
                                  std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT((buf != nullptr || len == 0) && peer != nullptr);
    return TimedRecvFromOp{&io_->proactor_for_current(), fd_, buf, len, peer,
                           kSockaddrCap, alloc, deadline_ns};
  }
  CancelRecvFromOp recv_from_cancel(
      void* buf, std::uint32_t len, void* peer,
      const sched::CancelToken* token) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT((buf != nullptr || len == 0) && peer != nullptr);
    return CancelRecvFromOp{&io_->proactor_for_current(), fd_, buf, len, peer,
                            kSockaddrCap, token};
  }

  // co_await sock.send_to(buf, len, &ss, sslen) -> bytes sent (>=0) or negative
  // errno, to the destination (ss, sslen) filled by resolve_addr(). `buf` and `ss`
  // must outlive the co_await.
  SendToOp send_to(const void* buf, std::uint32_t len, const void* dest,
                   std::uint32_t destlen) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    return SendToOp{&io_->proactor_for_current(), fd_, buf, len, dest, destlen};
  }
  TimedSendToOp send_to_until(const void* buf, std::uint32_t len,
                              const void* dest, std::uint32_t destlen,
                              mem::AllocatorRef alloc,
                              std::uint64_t deadline_ns) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT((buf != nullptr || len == 0) && dest != nullptr &&
                      destlen != 0);
    return TimedSendToOp{&io_->proactor_for_current(), fd_, buf, len, dest,
                         destlen, alloc, deadline_ns};
  }
  CancelSendToOp send_to_cancel(
      const void* buf, std::uint32_t len, const void* dest,
      std::uint32_t destlen, const sched::CancelToken* token) noexcept {
    HYLE_FATAL_ASSERT(valid() &&
                      sched::current_worker_index() == static_cast<int>(owner_));
    HYLE_FATAL_ASSERT((buf != nullptr || len == 0) && dest != nullptr &&
                      destlen != 0);
    return CancelSendToOp{&io_->proactor_for_current(), fd_, buf, len, dest,
                          destlen, token};
  }

  // Size a caller sockaddr buffer to this so recv_from can hold any address it
  // sees (sockaddr_storage is 128 bytes on Linux/Windows; matches proactor use).
  static constexpr std::uint32_t kSockaddrCap = 128;

 private:
  IoContext* io_ = nullptr;
  std::intptr_t fd_ = -1;
  unsigned owner_ = 0;
};

// ---- cross-platform socket setup (udp.cc) ----------------------------------
// Create a UDP socket bound to `host`:`port` (host e.g. "127.0.0.1"), associate
// it with `owner_worker`'s ring, and return a UdpSocket. Returns an invalid
// UdpSocket (valid()==false) on failure. Use for a receiver/server.
HYLE_IO_API UdpSocket udp_bind(IoContext& io, unsigned owner_worker,
                               const char* host, std::uint16_t port) noexcept;

// Create an unbound UDP socket associated with `owner_worker`'s ring (the OS
// assigns an ephemeral local port on first send). Use for a client that only
// send_to()s and recv_from()s replies. Returns invalid on failure.
HYLE_IO_API UdpSocket udp_socket(IoContext& io,
                                 unsigned owner_worker) noexcept;
HYLE_IO_API UdpSocket udp_socket(IoContext& io, unsigned owner_worker,
                                 AddressFamily family) noexcept;

}  // namespace hyle::io
