#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#endif

// io-04 net convenience layer test: a real TCP echo over loopback using the RAII
// TcpListener / TcpStream wrappers + co_await read/write/accept/connect. A server
// fiber listens, accepts one connection, and echoes; a client fiber connects,
// sends, and reads the echo back. Everything is pinned to one worker
// (shard-per-core) so the connection's io stays on its core.

using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tcp_listen;
using hyle::io::tcp_socket;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;  // single-core connection for this test
constexpr std::uint16_t kPort = 54344;

struct EchoResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<std::int64_t> server_echoed{0};
  char client_got[64] = {};
  std::atomic<std::int64_t> client_recv{0};
  std::atomic<bool> connected{false};
};

// Server: accept one connection, read a message, echo it back, close.
Task<int> server_coro(Allocator, IoContext* io, TcpListener* ls, EchoResult* r) {
  TcpStream conn = co_await ls->accept(kOwner);
  if (!conn.valid()) {
    r->server_done.store(true, std::memory_order_release);
    co_return 0;
  }
  char buf[64] = {};
  std::int64_t n = co_await conn.read(buf, sizeof(buf));
  if (n > 0) {
    std::int64_t w = co_await conn.write(buf, static_cast<std::uint32_t>(n));
    r->server_echoed.store(w, std::memory_order_release);
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

// Client: connect, send a message, read the echo, verify.
Task<int> client_coro(Allocator, IoContext* io, EchoResult* r) {
  TcpStream s = tcp_socket(*io, kOwner);
  if (!s.valid()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  // A sockaddr_storage-sized, suitably-aligned buffer; resolve_addr fills a
  // sockaddr_in into it. Lives on the coroutine frame so it outlives the connect.
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  if (!resolve_addr("127.0.0.1", kPort, &ss, &len)) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  std::int64_t cc = co_await s.connect(&ss, len);
  if (cc != 0) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->connected.store(true, std::memory_order_release);
  static const char kMsg[] = "tcp-wrapper-echo";
  co_await s.write(kMsg, sizeof(kMsg));
  r->client_recv.store(
      co_await s.read(r->client_got, sizeof(kMsg)), std::memory_order_release);
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

}  // namespace

TEST(IoTcp, ListenerAcceptEchoRoundTrip) {
  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid()) << "tcp_listen failed (port in use?)";

  auto r = std::make_unique<EchoResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = server_coro(system_allocator(), &io, &ls, r.get());
  dc->task = client_coro(system_allocator(), &io, r.get());
  Fiber fs(system_allocator(), kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(system_allocator(), kDefaultStackSize, &drive_entry, dc.get());

  // Both connection fibers pinned to the owner core (shard-per-core).
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->server_done.load());
  EXPECT_TRUE(r->client_done.load());
  EXPECT_TRUE(r->connected.load());
  EXPECT_EQ(r->server_echoed.load(), static_cast<std::int64_t>(sizeof("tcp-wrapper-echo")));
  EXPECT_EQ(r->client_recv.load(), static_cast<std::int64_t>(sizeof("tcp-wrapper-echo")));
  EXPECT_STREQ(r->client_got, "tcp-wrapper-echo");
}
