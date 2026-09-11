#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

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
#  include <ws2tcpip.h>
#else
#  include <cerrno>
#  include <fcntl.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <csignal>
#  endif
#endif

// io-04 net convenience layer test: a real TCP echo over loopback using the RAII
// TcpListener / TcpStream wrappers + co_await read/write/accept/connect. A server
// fiber listens, accepts one connection, and echoes; a client fiber connects,
// sends, and reads the echo back. Everything is pinned to one worker
// (shard-per-core) so the connection's io stays on its core.

using hyle::io::IoContext;
using hyle::io::AddressFamily;
using hyle::io::ConstBuffer;
using hyle::io::Endpoint;
using hyle::io::MutableBuffer;
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

static_assert(!std::is_constructible_v<TcpStream, IoContext&, std::intptr_t,
                                       unsigned>);
static_assert(!std::is_constructible_v<TcpListener, IoContext&, std::intptr_t,
                                       unsigned>);
static_assert(std::is_constructible_v<bool, TcpStream>);
static_assert(std::is_constructible_v<bool, TcpListener>);

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
  const ConstBuffer output[] = {{kMsg, 4}, {kMsg + 4, sizeof(kMsg) - 4}};
  auto written = co_await write_all(system_allocator(), s,
                                    std::span<const ConstBuffer>{output});
  if (!written.ok()) {
    r->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->half_closed.store(s.shutdown_write().ok(), std::memory_order_release);
  const MutableBuffer input[] = {
      {r->client_got, 5}, {r->client_got + 5, sizeof(kMsg) - 5}};
  r->client_recv.store(
      static_cast<std::int64_t>(
          (co_await read_exact(system_allocator(), s,
                               std::span<const MutableBuffer>{input}))
              .progress.bytes),
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

struct NativeSocketPair {
  std::intptr_t owned = -1;
  std::intptr_t peer = -1;
};

void close_native_socket(std::intptr_t socket) noexcept {
  if (socket < 0) return;
#if defined(_WIN32)
  ::closesocket(static_cast<SOCKET>(socket));
#else
  ::close(static_cast<int>(socket));
#endif
}

NativeSocketPair make_native_socket_pair() noexcept {
#if defined(_WIN32)
  NativeSocketPair pair;
  SOCKET listener = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                                 WSA_FLAG_OVERLAPPED);
  if (listener == INVALID_SOCKET) return pair;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(listener, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0 ||
      ::listen(listener, 1) != 0) {
    ::closesocket(listener);
    return pair;
  }
  int address_size = sizeof(address);
  if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                    &address_size) != 0) {
    ::closesocket(listener);
    return pair;
  }
  SOCKET client = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                               WSA_FLAG_OVERLAPPED);
  if (client == INVALID_SOCKET ||
      ::connect(client, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) != 0) {
    if (client != INVALID_SOCKET) ::closesocket(client);
    ::closesocket(listener);
    return pair;
  }
  SOCKET server = ::accept(listener, nullptr, nullptr);
  ::closesocket(listener);
  if (server == INVALID_SOCKET) {
    ::closesocket(client);
    return pair;
  }
  pair.owned = static_cast<std::intptr_t>(client);
  pair.peer = static_cast<std::intptr_t>(server);
  return pair;
#else
  int sockets[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) return {};
  return {sockets[0], sockets[1]};
#endif
}

bool native_send_exact(std::intptr_t socket, const void* data,
                       std::size_t size) noexcept {
#if defined(_WIN32)
  return ::send(static_cast<SOCKET>(socket),
                static_cast<const char*>(data), static_cast<int>(size), 0) ==
         static_cast<int>(size);
#else
  return ::send(static_cast<int>(socket), data, size, 0) ==
         static_cast<ssize_t>(size);
#endif
}

std::size_t native_receive(std::intptr_t socket, void* data,
                           std::size_t size) noexcept {
#if defined(_WIN32)
  const int received = ::recv(static_cast<SOCKET>(socket),
                              static_cast<char*>(data),
                              static_cast<int>(size), 0);
  return received > 0 ? static_cast<std::size_t>(received) : 0;
#else
  const ssize_t received = ::recv(static_cast<int>(socket), data, size, 0);
  return received > 0 ? static_cast<std::size_t>(received) : 0;
#endif
}

void set_small_send_buffer(std::intptr_t socket) noexcept {
  constexpr int kSmallBuffer = 4096;
#if defined(_WIN32)
  (void)::setsockopt(static_cast<SOCKET>(socket), SOL_SOCKET, SO_SNDBUF,
                     reinterpret_cast<const char*>(&kSmallBuffer),
                     sizeof(kSmallBuffer));
#else
  (void)::setsockopt(static_cast<int>(socket), SOL_SOCKET, SO_SNDBUF,
                     &kSmallBuffer, sizeof(kSmallBuffer));
#endif
}

struct ImmediateTransferState {
  TcpStream* ordinary = nullptr;
  std::intptr_t ordinary_peer = -1;
  TcpStream* pressured = nullptr;
  hyle::io::detail::ImmediateTransferState empty_read =
      hyle::io::detail::ImmediateTransferState::kReady;
  hyle::io::detail::ImmediateTransferState partial_read =
      hyle::io::detail::ImmediateTransferState::kWouldBlock;
  hyle::io::detail::ImmediateTransferState ready_write =
      hyle::io::detail::ImmediateTransferState::kWouldBlock;
  hyle::io::detail::ImmediateTransferState pressured_write =
      hyle::io::detail::ImmediateTransferState::kReady;
  hyle::io::detail::ImmediateTransferState eof_read =
      hyle::io::detail::ImmediateTransferState::kWouldBlock;
  hyle::io::TransferResult partial{};
  hyle::io::TransferResult written{};
  hyle::io::TransferResult eof{};
  std::array<std::byte, 8> input{};
  std::array<std::byte, 3> echoed{};
  bool peer_send_ok = false;
  std::size_t peer_received = 0;
  std::size_t pressure_bytes = 0;
};

void immediate_transfer_entry(void* opaque) noexcept {
  auto* state = static_cast<ImmediateTransferState*>(opaque);
  auto empty = hyle::io::detail::tcp_try_read_some(
      *state->ordinary, std::span<std::byte>{state->input});
  state->empty_read = empty.state;

  constexpr std::array<std::byte, 3> kInbound = {
      std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
  state->peer_send_ok = native_send_exact(
      state->ordinary_peer, kInbound.data(), kInbound.size());
  auto partial = hyle::io::detail::tcp_try_read_some(
      *state->ordinary, std::span<std::byte>{state->input});
  state->partial_read = partial.state;
  state->partial = std::move(partial.transfer);

  constexpr std::array<std::byte, 3> kOutbound = {
      std::byte{'x'}, std::byte{'y'}, std::byte{'z'}};
  auto write = hyle::io::detail::tcp_try_write_some(
      *state->ordinary, std::span<const std::byte>{kOutbound});
  state->ready_write = write.state;
  state->written = std::move(write.transfer);
  if (state->written.ok() && state->written.bytes != 0) {
    state->peer_received = native_receive(
        state->ordinary_peer, state->echoed.data(), state->echoed.size());
  }

  close_native_socket(state->ordinary_peer);
  state->ordinary_peer = -1;
  auto eof = hyle::io::detail::tcp_try_read_some(
      *state->ordinary, std::span<std::byte>{state->input});
  state->eof_read = eof.state;
  state->eof = std::move(eof.transfer);

  constexpr std::array<std::byte, 4096> kPressure{};
  constexpr std::size_t kMaxAttempts = 1024;
  for (std::size_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
    auto result = hyle::io::detail::tcp_try_write_some(
        *state->pressured, std::span<const std::byte>{kPressure});
    if (result.state ==
        hyle::io::detail::ImmediateTransferState::kWouldBlock) {
      state->pressured_write = result.state;
      break;
    }
    if (!result.transfer.ok() || result.transfer.bytes == 0) {
      state->pressured_write = result.state;
      break;
    }
    state->pressure_bytes += result.transfer.bytes;
  }
}

#if defined(__linux__)
volatile std::sig_atomic_t g_sigpipe_count = 0;

extern "C" void count_sigpipe(int) noexcept { g_sigpipe_count = 1; }

struct PeerCloseWriteState {
  TcpStream* scalar = nullptr;
  TcpStream* vectored = nullptr;
  hyle::io::TransferResult scalar_result{};
  hyle::io::BufferSequenceResult vectored_result{};
};

Task<int> peer_close_write_coro(AllocatorRef,
                                PeerCloseWriteState* state) noexcept {
  constexpr std::array<std::byte, 4> kScalar = {
      std::byte{'p'}, std::byte{'i'}, std::byte{'p'}, std::byte{'e'}};
  state->scalar_result = co_await state->scalar->write_some(kScalar);

  constexpr std::array<std::byte, 2> kFirst = {
      std::byte{'v'}, std::byte{'1'}};
  constexpr std::array<std::byte, 2> kSecond = {
      std::byte{'v'}, std::byte{'2'}};
  const ConstBuffer buffers[] = {
      {kFirst.data(), kFirst.size()}, {kSecond.data(), kSecond.size()}};
  state->vectored_result = co_await state->vectored->write_some(buffers);
  co_return 0;
}
#endif

}  // namespace

TEST(IoTcp, ResolvesIpv4AndIpv6Literals) {
  EXPECT_TRUE(Endpoint::parse("127.0.0.1", 80).has_value());
  EXPECT_TRUE(Endpoint::parse("::1", 443).has_value());
  EXPECT_FALSE(Endpoint::parse("localhost", 80).has_value());
}

TEST(IoTcp, AdoptMoveAndReleaseKeepExactlyOneNativeOwner) {
  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  NativeSocketPair pair = make_native_socket_pair();
  ASSERT_GE(pair.owned, 0);
  ASSERT_GE(pair.peer, 0);
  {
    auto adopted = TcpStream::adopt(io, kOwner, pair.owned);
    pair.owned = -1;
    ASSERT_TRUE(adopted.has_value()) << adopted.error().message();
    EXPECT_TRUE(adopted->valid());
    EXPECT_EQ(adopted->owner_worker(), kOwner);
    EXPECT_TRUE(adopted->set_keepalive(false).ok());

    TcpStream owner;
    owner = std::move(*adopted);
    EXPECT_TRUE(owner.valid());
    EXPECT_FALSE(adopted->valid());
    EXPECT_TRUE(owner.set_keepalive().ok());

    const std::intptr_t released = owner.release_native_handle();
    ASSERT_GE(released, 0);
    EXPECT_FALSE(owner.valid());
    EXPECT_EQ(owner.native_handle(), -1);

#if defined(_WIN32)
    // Windows binds a socket to one IOCP for its complete native lifetime.
    // Ownership release is valid, but the same socket cannot be associated a
    // second time even with the same port. Close the transferred owner here.
    close_native_socket(released);
#else
    auto readopted = TcpStream::adopt(io, kOwner, released);
    ASSERT_TRUE(readopted.has_value()) << readopted.error().message();
    EXPECT_TRUE(readopted->valid());
#endif
  }
  close_native_socket(pair.peer);
  scheduler.shutdown();
}

TEST(IoTcp, ListenerMoveAssignmentClosesTheReplacedOwner) {
  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  {
    auto first = TcpListener::bind(
        io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, 0));
    auto second = TcpListener::bind(
        io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, 0));
    ASSERT_TRUE(first.has_value()) << first.error().message();
    ASSERT_TRUE(second.has_value()) << second.error().message();
    TcpListener owner = std::move(*first);
    const auto replacement_endpoint = second->local_endpoint();
    ASSERT_TRUE(replacement_endpoint.has_value());
    owner = std::move(*second);
    EXPECT_FALSE(second->valid());
    EXPECT_TRUE(owner.valid());
    EXPECT_EQ(owner.owner_worker(), kOwner);
    auto actual_endpoint = owner.local_endpoint();
    ASSERT_TRUE(actual_endpoint.has_value());
    EXPECT_EQ(*actual_endpoint, *replacement_endpoint);
  }
  scheduler.shutdown();
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

TEST(IoTcp, ImmediateProtocolTransferPreservesBoundariesWithoutWaiting) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  NativeSocketPair ordinary_pair = make_native_socket_pair();
  ASSERT_GE(ordinary_pair.owned, 0);
  ASSERT_GE(ordinary_pair.peer, 0);
  NativeSocketPair pressure_pair = make_native_socket_pair();
  ASSERT_GE(pressure_pair.owned, 0);
  ASSERT_GE(pressure_pair.peer, 0);
  set_small_send_buffer(pressure_pair.owned);

  auto ordinary = TcpStream::adopt(io, kOwner, ordinary_pair.owned);
  ordinary_pair.owned = -1;
  ASSERT_TRUE(ordinary.has_value())
      << (ordinary.has_value() ? "" : ordinary.error().message());
  auto pressured = TcpStream::adopt(io, kOwner, pressure_pair.owned);
  pressure_pair.owned = -1;
  ASSERT_TRUE(pressured.has_value())
      << (pressured.has_value() ? "" : pressured.error().message());

  ImmediateTransferState state;
  state.ordinary = &*ordinary;
  state.ordinary_peer = ordinary_pair.peer;
  state.pressured = &*pressured;
  Fiber fiber(kDefaultStackSize, &immediate_transfer_entry, &state);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  ordinary_pair.peer = state.ordinary_peer;

  using hyle::io::detail::ImmediateTransferState;
  EXPECT_EQ(state.empty_read, ImmediateTransferState::kWouldBlock);
  EXPECT_TRUE(state.peer_send_ok);
  EXPECT_EQ(state.partial_read, ImmediateTransferState::kReady);
  EXPECT_TRUE(state.partial.ok()) << state.partial.status.message();
  EXPECT_FALSE(state.partial.eof);
  EXPECT_EQ(state.partial.bytes, 3u);
  EXPECT_EQ(state.input[0], std::byte{'a'});
  EXPECT_EQ(state.input[1], std::byte{'b'});
  EXPECT_EQ(state.input[2], std::byte{'c'});
  EXPECT_EQ(state.ready_write, ImmediateTransferState::kReady);
  EXPECT_TRUE(state.written.ok()) << state.written.status.message();
  EXPECT_EQ(state.written.bytes, 3u);
  EXPECT_EQ(state.peer_received, 3u);
  EXPECT_EQ(state.echoed[0], std::byte{'x'});
  EXPECT_EQ(state.echoed[1], std::byte{'y'});
  EXPECT_EQ(state.echoed[2], std::byte{'z'});
  EXPECT_EQ(state.eof_read, ImmediateTransferState::kReady);
  EXPECT_TRUE(state.eof.ok()) << state.eof.status.message();
  EXPECT_TRUE(state.eof.eof);
  EXPECT_EQ(state.eof.bytes, 0u);
  EXPECT_GT(state.pressure_bytes, 0u);
  EXPECT_EQ(state.pressured_write, ImmediateTransferState::kWouldBlock);

  close_native_socket(ordinary_pair.peer);
  close_native_socket(pressure_pair.peer);
}

#if defined(__linux__)
TEST(IoTcp, PeerCloseReturnsTypedErrorsWithoutSigpipe) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());

  NativeSocketPair scalar_pair = make_native_socket_pair();
  NativeSocketPair vectored_pair = make_native_socket_pair();
  ASSERT_GE(scalar_pair.owned, 0);
  ASSERT_GE(scalar_pair.peer, 0);
  ASSERT_GE(vectored_pair.owned, 0);
  ASSERT_GE(vectored_pair.peer, 0);
  auto scalar = TcpStream::adopt(io, kOwner, scalar_pair.owned);
  scalar_pair.owned = -1;
  auto vectored = TcpStream::adopt(io, kOwner, vectored_pair.owned);
  vectored_pair.owned = -1;
  ASSERT_TRUE(scalar.has_value());
  ASSERT_TRUE(vectored.has_value());
  close_native_socket(scalar_pair.peer);
  scalar_pair.peer = -1;
  close_native_socket(vectored_pair.peer);
  vectored_pair.peer = -1;

  struct sigaction replacement {};
  struct sigaction previous {};
  replacement.sa_handler = &count_sigpipe;
  ::sigemptyset(&replacement.sa_mask);
  ASSERT_EQ(::sigaction(SIGPIPE, &replacement, &previous), 0);
  g_sigpipe_count = 0;

  PeerCloseWriteState state{&*scalar, &*vectored};
  Drive driver{peer_close_write_coro(system_allocator(), &state)};
  Fiber fiber(kDefaultStackSize, &drive_entry, &driver);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_EQ(::sigaction(SIGPIPE, &previous, nullptr), 0);

  EXPECT_FALSE(state.scalar_result.ok());
  EXPECT_EQ(state.scalar_result.status.native_error().domain,
            hyle::NativeErrorDomain::PosixErrno);
  EXPECT_EQ(state.scalar_result.status.native_error().code, EPIPE);
  EXPECT_FALSE(state.vectored_result.ok());
  EXPECT_EQ(state.vectored_result.status.native_error().domain,
            hyle::NativeErrorDomain::PosixErrno);
  EXPECT_EQ(state.vectored_result.status.native_error().code, EPIPE);
  EXPECT_EQ(g_sigpipe_count, 0);
}
#endif
