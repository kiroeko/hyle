#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"  // resolve_addr (shared IPv4 literal -> sockaddr helper)
#include "hyle/io/udp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#else
#  include <fcntl.h>
#endif

// io-04 net datagram test: a real UDP echo over loopback using the RAII UdpSocket
// wrapper + co_await recv_from/send_to. The server binds a port and recv_from's
// one datagram (learning the client's address), then send_to's it straight back.
// The client send_to's the server, then recv_from's the echo. Both fibers pinned
// to one worker (shard-per-core) so each socket's io stays on its core.
//
// The server socket is BOUND before either fiber runs (udp_bind is synchronous at
// setup), so there is no listen/accept race: the port is ready when the client
// sends. The client's source address is ephemeral (assigned by the OS on first
// send); the server echoes to exactly the address recv_from reported.

using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::udp_bind;
using hyle::io::udp_socket;
using hyle::io::UdpSocket;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::uint16_t kPort = 54355;

struct UdpResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<std::int64_t> server_recv{0};   // bytes the server received
  std::atomic<std::int64_t> server_sent{0};   // bytes the server echoed
  std::atomic<std::int64_t> client_recv{0};   // bytes the client got back
  std::atomic<bool> server_nonblocking{true};
  std::atomic<bool> client_nonblocking{true};
  char client_got[64] = {};
};

// Server: recv one datagram (learn sender addr), echo it back to that addr.
Task<int> server_coro(AllocatorRef, UdpSocket* sock, UdpResult* r) {
#if !defined(_WIN32)
  r->server_nonblocking.store(
      (::fcntl(static_cast<int>(sock->fd()), F_GETFL, 0) & O_NONBLOCK) != 0,
      std::memory_order_release);
#endif
  char buf[64] = {};
  alignas(16) unsigned char peer[UdpSocket::kSockaddrCap];
  std::int64_t n = co_await sock->recv_from(buf, sizeof(buf), &peer);
  r->server_recv.store(n, std::memory_order_release);
  if (n > 0) {
    // Echo back to the exact source. The peer length: IPv4 sockaddr_in = 16.
    std::int64_t w =
        co_await sock->send_to(buf, static_cast<std::uint32_t>(n), &peer, 16);
    r->server_sent.store(w, std::memory_order_release);
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

// Client: send a datagram to the server, then recv the echo and verify.
Task<int> client_coro(AllocatorRef, UdpSocket* sock, UdpResult* r) {
#if !defined(_WIN32)
  r->client_nonblocking.store(
      (::fcntl(static_cast<int>(sock->fd()), F_GETFL, 0) & O_NONBLOCK) != 0,
      std::memory_order_release);
#endif
  alignas(16) unsigned char dest[128];
  std::uint32_t dlen = 0;
  if (!resolve_addr("127.0.0.1", kPort, &dest, &dlen)) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  static const char kMsg[] = "udp-wrapper-echo";
  co_await sock->send_to(kMsg, sizeof(kMsg), &dest, dlen);
  alignas(16) unsigned char from[UdpSocket::kSockaddrCap];
  r->client_recv.store(
      co_await sock->recv_from(r->client_got, sizeof(kMsg), &from),
      std::memory_order_release);
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

}  // namespace

TEST(IoUdp, RecvFromSendToEchoRoundTrip) {
  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  UdpSocket srv = udp_bind(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(srv.valid()) << "udp_bind failed (port in use?)";
  UdpSocket cli = udp_socket(io, kOwner);
  ASSERT_TRUE(cli.valid());

  auto r = std::make_unique<UdpResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = server_coro(system_allocator(), &srv, r.get());
  dc->task = client_coro(system_allocator(), &cli, r.get());
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());

  // Both pinned to the owner core (shard-per-core); server spawned first so it is
  // waiting in recv_from before the client sends.
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->server_done.load());
  EXPECT_TRUE(r->client_done.load());
  EXPECT_TRUE(r->server_nonblocking.load());
  EXPECT_TRUE(r->client_nonblocking.load());
  constexpr std::int64_t kLen = static_cast<std::int64_t>(sizeof("udp-wrapper-echo"));
  EXPECT_EQ(r->server_recv.load(), kLen);
  EXPECT_EQ(r->server_sent.load(), kLen);
  EXPECT_EQ(r->client_recv.load(), kLen);
  EXPECT_STREQ(r->client_got, "udp-wrapper-echo");
}
