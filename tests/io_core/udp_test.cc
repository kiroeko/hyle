#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>

#include "hyle/io/io_context.h"
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
using hyle::io::AddressFamily;
using hyle::io::ConstBuffer;
using hyle::io::Endpoint;
using hyle::io::MutableBuffer;
using hyle::io::UdpSocket;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

static_assert(!std::is_constructible_v<UdpSocket, IoContext&, std::intptr_t,
                                       unsigned>);
static_assert(std::is_constructible_v<bool, UdpSocket>);

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
      (::fcntl(static_cast<int>(sock->native_handle()), F_GETFL, 0) & O_NONBLOCK) != 0,
      std::memory_order_release);
#endif
  char buf[64] = {};
  const MutableBuffer input[] = {{buf, 7}, {buf + 7, sizeof(buf) - 7}};
  auto received =
      co_await sock->receive_from(std::span<const MutableBuffer>{input});
  const std::int64_t n = received.ok()
                             ? static_cast<std::int64_t>(received.progress.bytes)
                             : -1;
  r->server_recv.store(n, std::memory_order_release);
  if (n > 0) {
    const std::size_t first = std::min<std::size_t>(7, n);
    const ConstBuffer output[] = {
        {buf, first}, {buf + first, static_cast<std::size_t>(n) - first}};
    auto sent = co_await sock->send_to(
        std::span<const ConstBuffer>{output}, received.peer);
    const std::int64_t w = sent.ok()
                               ? static_cast<std::int64_t>(sent.progress.bytes)
                               : -1;
    r->server_sent.store(w, std::memory_order_release);
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

// Client: send a datagram to the server, then recv the echo and verify.
Task<int> client_coro(AllocatorRef, UdpSocket* sock, UdpResult* r) {
#if !defined(_WIN32)
  r->client_nonblocking.store(
      (::fcntl(static_cast<int>(sock->native_handle()), F_GETFL, 0) & O_NONBLOCK) != 0,
      std::memory_order_release);
#endif
  static const char kMsg[] = "udp-wrapper-echo";
  auto sent = co_await sock->send_to(
      std::as_bytes(std::span{kMsg, sizeof(kMsg)}),
      Endpoint::loopback(AddressFamily::kIpv4, kPort));
  if (!sent.ok()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  auto received = co_await sock->receive_from(
      std::as_writable_bytes(std::span{r->client_got, sizeof(kMsg)}));
  r->client_recv.store(received.ok()
                           ? static_cast<std::int64_t>(received.bytes)
                           : -1,
                       std::memory_order_release);
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

struct TruncationResult {
  std::atomic<bool> receive_done{false};
  std::atomic<bool> send_done{false};
  std::atomic<bool> truncated{false};
  std::atomic<std::size_t> bytes{0};
};

Task<int> receive_truncated(AllocatorRef, UdpSocket* socket,
                            TruncationResult* result) {
  std::byte first[2]{};
  std::byte second[3]{};
  const MutableBuffer buffers[] = {
      {first, sizeof(first)}, {second, sizeof(second)}};
  auto received =
      co_await socket->receive_from(std::span<const MutableBuffer>{buffers});
  result->truncated.store(received.ok() && received.truncated,
                          std::memory_order_release);
  result->bytes.store(received.progress.bytes, std::memory_order_release);
  result->receive_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> send_oversized(AllocatorRef, UdpSocket* socket, Endpoint destination,
                         TruncationResult* result) {
  static constexpr char kMessage[] = "larger-than-five";
  const ConstBuffer buffers[] = {
      {kMessage, 6}, {kMessage + 6, sizeof(kMessage) - 6}};
  auto sent = co_await socket->send_to(
      std::span<const ConstBuffer>{buffers}, destination);
  result->send_done.store(sent.ok(), std::memory_order_release);
  co_return 0;
}

}  // namespace

TEST(IoUdp, InvalidContextReturnsTypedSetupFailure) {
  Scheduler scheduler(1, system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());

  hyle::io::IoContextOptions options;
  options.queue_depth = 0;
  IoContext invalid(scheduler, options);
  ASSERT_FALSE(invalid.valid());

  auto opened = UdpSocket::open(invalid, kOwner);
  ASSERT_FALSE(opened.has_value());
  EXPECT_TRUE(opened.error().IsFailedPrecondition());
  scheduler.shutdown();
}

TEST(IoUdp, ReleaseReadoptAndMoveKeepExactlyOneNativeOwner) {
  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  {
    auto opened = UdpSocket::open(io, kOwner);
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    EXPECT_TRUE(opened->valid());
    EXPECT_EQ(opened->owner_worker(), kOwner);

    const std::intptr_t released = opened->release_native_handle();
    ASSERT_GE(released, 0);
    EXPECT_FALSE(opened->valid());
    EXPECT_EQ(opened->native_handle(), -1);

    auto adopted = UdpSocket::adopt(io, kOwner, released);
    ASSERT_TRUE(adopted.has_value()) << adopted.error().message();
    UdpSocket owner;
    owner = std::move(*adopted);
    EXPECT_TRUE(owner.valid());
    EXPECT_FALSE(adopted->valid());
    EXPECT_TRUE(owner.local_endpoint().has_value());
  }
  scheduler.shutdown();
}

TEST(IoUdp, RecvFromSendToEchoRoundTrip) {
  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());

  auto server = UdpSocket::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(server.has_value()) << "UDP bind failed (port in use?)";
  UdpSocket srv = std::move(*server);
  auto client = UdpSocket::open(io, kOwner);
  ASSERT_TRUE(client.has_value());
  UdpSocket cli = std::move(*client);

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

TEST(IoUdp, VectoredReceiveReportsDiscardedDatagramTail) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto server = UdpSocket::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, 0));
  ASSERT_TRUE(server.has_value());
  auto endpoint = server->local_endpoint();
  ASSERT_TRUE(endpoint.has_value());
  auto client = UdpSocket::open(io, kOwner);
  ASSERT_TRUE(client.has_value());

  TruncationResult result;
  Drive receiver{receive_truncated(system_allocator(), &*server, &result)};
  Drive sender{
      send_oversized(system_allocator(), &*client, *endpoint, &result)};
  Fiber receive_fiber(kDefaultStackSize, &drive_entry, &receiver);
  Fiber send_fiber(kDefaultStackSize, &drive_entry, &sender);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &receive_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &send_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result.receive_done.load());
  EXPECT_TRUE(result.send_done.load());
  EXPECT_TRUE(result.truncated.load());
  EXPECT_EQ(result.bytes.load(), 5u);
}
