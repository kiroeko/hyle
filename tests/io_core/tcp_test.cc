#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>

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
using hyle::io::TcpConnectOptions;
using hyle::io::TcpKeepaliveOptions;
using hyle::io::TcpListenOptions;
using hyle::io::connect_tcp;
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
  std::atomic<bool> server_endpoint_inspection{false};
  std::atomic<bool> client_endpoint_inspection{false};
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
  auto local = conn.local_endpoint();
  auto peer = conn.peer_endpoint();
  r->server_endpoint_inspection.store(local.has_value() && peer.has_value(),
                                      std::memory_order_release);
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
  TcpKeepaliveOptions keepalive;
  keepalive.idle = std::chrono::seconds(30);
  keepalive.interval = std::chrono::seconds(5);
  keepalive.probe_count = 3;
  auto local = s.local_endpoint();
  auto peer = s.peer_endpoint();
  r->client_endpoint_inspection.store(
      local.has_value() && peer.has_value() && peer->port() == kPort,
      std::memory_order_release);
  r->socket_options.store(
      s.set_nodelay().ok() && s.set_keepalive(keepalive).ok(),
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

struct ConnectDrive {
  Task<hyle::Result<TcpStream>> task;
};
void connect_drive_entry(void* p) noexcept {
  static_cast<ConnectDrive*>(p)->task.resume();
}

Task<int> accept_once(AllocatorRef, TcpListener* listener,
                      std::atomic<bool>* accepted) {
  auto stream = co_await listener->accept(kOwner);
  accepted->store(stream.has_value(), std::memory_order_release);
  co_return 0;
}

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
  EXPECT_TRUE(r->server_endpoint_inspection.load());
  EXPECT_TRUE(r->client_endpoint_inspection.load());
  EXPECT_TRUE(r->half_closed.load());
  EXPECT_EQ(r->server_echoed.load(), static_cast<std::int64_t>(sizeof("tcp-wrapper-echo")));
  EXPECT_EQ(r->client_recv.load(), static_cast<std::int64_t>(sizeof("tcp-wrapper-echo")));
  EXPECT_STREQ(r->client_got, "tcp-wrapper-echo");
}

TEST(IoTcp, HostnameConnectOwnsInputAndFallsBackAcrossAddresses) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  auto listener = TcpListener::bind(
      io, kOwner,
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0));
  ASSERT_TRUE(listener.has_value());
  TcpListener server = std::move(*listener);
  auto local = server.local_endpoint();
  ASSERT_TRUE(local.has_value());

  std::atomic<bool> accepted{false};
  Drive accept_driver{
      accept_once(system_allocator(), &server, &accepted)};
  Fiber accept_fiber(kDefaultStackSize, &drive_entry, &accept_driver);

  ConnectDrive connect_driver;
  {
    std::string temporary_host = "localhost";
    connect_driver.task = connect_tcp(system_allocator(), io, kOwner,
                                      temporary_host, local->port());
  }
  ASSERT_TRUE(connect_driver.task.valid());
  Fiber connect_fiber(kDefaultStackSize, &connect_drive_entry,
                      &connect_driver);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &accept_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &connect_fiber));
  scheduler.drain();
  scheduler.shutdown();

  auto connected = connect_driver.task.result();
  EXPECT_TRUE(connected.has_value())
      << (connected.has_value() ? "" : connected.error().message());
  EXPECT_TRUE(accepted.load(std::memory_order_acquire));
}

TEST(IoTcp, PromptFailureAcceleratesNextEndpoint) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  auto listener = TcpListener::bind(
      io, kOwner,
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0));
  ASSERT_TRUE(listener.has_value());
  TcpListener server = std::move(*listener);
  auto good = server.local_endpoint();
  ASSERT_TRUE(good.has_value());

  const Endpoint candidates[] = {
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0), *good};
  TcpConnectOptions options;
  options.attempt_delay = std::chrono::seconds(2);
  options.max_parallel_attempts = 2;

  std::atomic<bool> accepted{false};
  Drive accept_driver{
      accept_once(system_allocator(), &server, &accepted)};
  ConnectDrive connect_driver{
      connect_tcp(system_allocator(), io, kOwner, candidates, options)};
  ASSERT_TRUE(connect_driver.task.valid());
  Fiber accept_fiber(kDefaultStackSize, &drive_entry, &accept_driver);
  Fiber connect_fiber(kDefaultStackSize, &connect_drive_entry,
                      &connect_driver);
  const auto started = std::chrono::steady_clock::now();
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &accept_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &connect_fiber));
  scheduler.drain();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  scheduler.shutdown();

  auto connected = connect_driver.task.result();
  EXPECT_TRUE(connected.has_value())
      << (connected.has_value() ? "" : connected.error().message());
  EXPECT_TRUE(accepted.load(std::memory_order_acquire));
  EXPECT_LT(elapsed, std::chrono::seconds(1));
}

TEST(IoTcp, AllFailedAttemptsAreDrainedBeforeReturn) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  const Endpoint candidates[] = {
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0),
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0),
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0)};
  TcpConnectOptions options;
  options.operation.deadline =
      hyle::time::deadline_after(std::chrono::seconds(1));
  options.attempt_delay = std::chrono::milliseconds(1);
  options.max_parallel_attempts = 3;
  ConnectDrive driver{
      connect_tcp(system_allocator(), io, kOwner, candidates, options)};
  ASSERT_TRUE(driver.task.valid());
  Fiber fiber(kDefaultStackSize, &connect_drive_entry, &driver);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();

  auto result = driver.task.result();
  EXPECT_FALSE(result.has_value());
}

TEST(IoTcp, ReusePortIsEitherHonoredOrExplicitlyUnsupported) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  TcpListenOptions options;
  options.reuse_port = true;
  auto first = TcpListener::bind(
      io, kOwner,
      Endpoint::loopback(hyle::io::AddressFamily::kIpv4, 0), options);
  if (!first.has_value()) {
    EXPECT_TRUE(first.error().IsUnimplemented()) << first.error().message();
    scheduler.shutdown();
    return;
  }
  auto endpoint = first->local_endpoint();
  ASSERT_TRUE(endpoint.has_value());
  auto second = TcpListener::bind(io, kOwner, *endpoint, options);
  EXPECT_TRUE(second.has_value())
      << (second.has_value() ? "" : second.error().message());
  scheduler.shutdown();
}
