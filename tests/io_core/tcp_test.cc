#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>

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
#else
#  include <fcntl.h>
#endif

// io-04 net convenience layer test: a real TCP echo over loopback using the RAII
// TcpListener / TcpStream wrappers + co_await read/write/accept/connect. A server
// fiber listens, accepts one connection, and echoes; a client fiber connects,
// sends, and reads the echo back. Everything is pinned to one worker
// (shard-per-core) so the connection's io stays on its core.

using hyle::io::IoContext;
using hyle::io::Endpoint;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::read_exact;
using hyle::io::write_all;
using hyle::mem::AllocatorRef;
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
  std::atomic<bool> accepted_nonblocking{true};
  std::atomic<bool> client_nonblocking{true};
  std::atomic<bool> socket_options{false};
  std::atomic<bool> half_closed{false};
};

// Server: accept one connection, read a message, echo it back, close.
Task<int> server_coro(AllocatorRef, IoContext*, TcpListener* ls, EchoResult* r) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value()) {
    r->server_done.store(true, std::memory_order_release);
    co_return 0;
  }
  TcpStream conn = std::move(*accepted);
#if !defined(_WIN32)
  r->accepted_nonblocking.store(
      (::fcntl(static_cast<int>(conn.native_handle()), F_GETFL, 0) & O_NONBLOCK) != 0,
      std::memory_order_release);
#endif
  char buf[64] = {};
  auto read = co_await read_exact(
      system_allocator(), conn,
      std::as_writable_bytes(
          std::span{buf, sizeof("tcp-wrapper-echo")}));
  const std::int64_t n =
      read.ok() ? static_cast<std::int64_t>(read.bytes) : -1;
  if (n > 0) {
    auto written = co_await write_all(
        system_allocator(), conn,
        std::as_bytes(std::span{buf, static_cast<std::size_t>(n)}));
    const std::int64_t w =
        written.ok() ? static_cast<std::int64_t>(written.bytes) : -1;
    r->server_echoed.store(w, std::memory_order_release);
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

// Client: connect, send a message, read the echo, verify.
Task<int> client_coro(AllocatorRef, IoContext* io, EchoResult* r) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  TcpStream s = std::move(*opened);
#if !defined(_WIN32)
  r->client_nonblocking.store(
      (::fcntl(static_cast<int>(s.native_handle()), F_GETFL, 0) & O_NONBLOCK) != 0,
      std::memory_order_release);
#endif
  auto endpoint = Endpoint::parse("127.0.0.1", kPort);
  if (!endpoint.has_value()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  auto connected = co_await s.connect(*endpoint);
  if (!connected.ok()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->connected.store(true, std::memory_order_release);
  r->socket_options.store(s.set_nodelay().ok() && s.set_keepalive().ok(),
                          std::memory_order_release);
  static const char kMsg[] = "tcp-wrapper-echo";
  auto written =
      co_await write_all(system_allocator(), s, std::as_bytes(std::span{kMsg}));
  if (!written.ok()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->half_closed.store(s.shutdown_write().ok(), std::memory_order_release);
  r->client_recv.store(
      static_cast<std::int64_t>((co_await read_exact(
          system_allocator(), s,
          std::as_writable_bytes(std::span{r->client_got, sizeof(kMsg)}))).bytes),
      std::memory_order_release);
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

}  // namespace

TEST(IoTcp, ResolvesIpv4AndIpv6Literals) {
  EXPECT_TRUE(Endpoint::parse("127.0.0.1", 80).has_value());
  EXPECT_TRUE(Endpoint::parse("::1", 443).has_value());
  EXPECT_FALSE(Endpoint::parse("localhost", 80).has_value());
}

TEST(IoTcp, ListenerAcceptEchoRoundTrip) {
  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());

  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(hyle::io::AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value()) << "TCP bind failed (port in use?)";
  TcpListener ls = std::move(*listener);

  auto r = std::make_unique<EchoResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = server_coro(system_allocator(), &io, &ls, r.get());
  dc->task = client_coro(system_allocator(), &io, r.get());
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());

  // Both connection fibers pinned to the owner core (shard-per-core).
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->server_done.load());
  EXPECT_TRUE(r->client_done.load());
  EXPECT_TRUE(r->connected.load());
  EXPECT_TRUE(r->accepted_nonblocking.load());
  EXPECT_TRUE(r->client_nonblocking.load());
  EXPECT_TRUE(r->socket_options.load());
  EXPECT_TRUE(r->half_closed.load());
  EXPECT_EQ(r->server_echoed.load(), static_cast<std::int64_t>(sizeof("tcp-wrapper-echo")));
  EXPECT_EQ(r->client_recv.load(), static_cast<std::int64_t>(sizeof("tcp-wrapper-echo")));
  EXPECT_STREQ(r->client_got, "tcp-wrapper-echo");
}
