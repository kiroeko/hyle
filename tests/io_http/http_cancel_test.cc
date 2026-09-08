#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

#include "hyle/error/status.h"
#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::HttpBodyChunk;
using hyle::io::HttpBodySource;
using hyle::io::HttpClientConnection;
using hyle::io::HttpIncomingResponse;
using hyle::io::HttpRequestHead;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::CancelToken;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

struct BudgetResult {
  std::atomic<bool> accepted{false};
  std::atomic<bool> connected{false};
  std::atomic<bool> done{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
  // Keep the accepted peer alive without sending a response, so the HTTP read
  // is settled only by its public operation budget rather than EOF/RST.
  TcpStream held;
};

Task<int> stalled_server(AllocatorRef, TcpListener *listener,
                         BudgetResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (accepted) {
    result->held = std::move(*accepted);
    result->accepted.store(true, std::memory_order_release);
  }
  co_return 0;
}

Task<int> budget_client(AllocatorRef allocator, IoContext *io,
                        BudgetResult *result, std::uint16_t port,
                        OperationOptions options) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened) {
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  result->connected.store(true, std::memory_order_release);

  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made) {
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  HttpClientConnection connection = std::move(*made);
  HttpRequestHead request;
  request.authority = "127.0.0.1";
  Result<HttpIncomingResponse> response = co_await connection.request(
      request, std::span<const std::byte>{}, options);
  result->status.store(response ? StatusCode::Ok : response.error().code(),
                       std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

void run_budget_case(std::uint16_t port, OperationOptions options,
                     BudgetResult &result, CancelToken *cancel) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = stalled_server(system_allocator(), &listener, &result);
  client->task = budget_client(system_allocator(), &io, &result, port, options);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));

  std::thread canceller;
  if (cancel != nullptr) {
    canceller = std::thread([&result, cancel] {
      while (!result.connected.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      cancel->cancel();
    });
  }

  scheduler.drain();
  if (canceller.joinable())
    canceller.join();
  scheduler.shutdown();
}

TEST(IoHttpBudget, AbsoluteDeadlineCoversWriteAndResponseRead) {
  constexpr std::uint16_t kPort = 54418;
  BudgetResult result;
  run_budget_case(kPort,
                  OperationOptions{.deadline = hyle::time::deadline_after(
                                       std::chrono::milliseconds(30))},
                  result, nullptr);

  EXPECT_TRUE(result.accepted.load(std::memory_order_acquire));
  EXPECT_TRUE(result.connected.load(std::memory_order_acquire));
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::DeadlineExceeded);
}

TEST(IoHttpBudget, CancellationCoversWriteAndResponseRead) {
  constexpr std::uint16_t kPort = 54419;
  CancelToken token = CancelToken::root(system_allocator());
  BudgetResult result;
  run_budget_case(kPort,
                  OperationOptions{.deadline = hyle::time::deadline_after(
                                       std::chrono::seconds(5)),
                                   .cancel = &token},
                  result, &token);

  EXPECT_TRUE(result.accepted.load(std::memory_order_acquire));
  EXPECT_TRUE(result.connected.load(std::memory_order_acquire));
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::Cancelled);
}

struct BackpressureResult {
  std::vector<std::byte> chunk =
      std::vector<std::byte>(256 * 1024, std::byte{0x54});
  std::atomic<std::size_t> source_calls{0};
  std::atomic<StatusCode> status{StatusCode::Unknown};
  std::atomic<bool> client_done{false};
};

struct BackpressureSource {
  BackpressureResult *result = nullptr;
  std::size_t index = 0;
  static constexpr std::size_t kChunkCount = 2048;

  Task<Result<HttpBodyChunk>> next(AllocatorRef allocator,
                                   OperationOptions options) noexcept {
    return next_impl(allocator, this, options);
  }

private:
  static Task<Result<HttpBodyChunk>>
  next_impl(AllocatorRef, BackpressureSource *self, OperationOptions) noexcept {
    ++self->index;
    self->result->source_calls.fetch_add(1, std::memory_order_relaxed);
    co_return HttpBodyChunk{
        self->result->chunk, {}, self->index == kChunkCount};
  }
};

Task<int> slow_reader_server(AllocatorRef allocator, TcpListener *listener,
                             BackpressureResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection connection = std::move(*made);
  auto request = co_await connection.read_request();
  if (!request || !request->has_value())
    co_return 0;
  if (!(co_await request->value().body.discard()).ok())
    co_return 0;

  BackpressureSource source{result};
  constexpr std::uint64_t kBodyBytes =
      BackpressureSource::kChunkCount * 256ull * 1024ull;
  const Status sent = co_await connection.send_response(
      HttpResponseHead{}, HttpBodySource::from(source, kBodyBytes),
      OperationOptions{.deadline = hyle::time::deadline_after(
                           std::chrono::milliseconds(40))});
  result->status.store(sent.code(), std::memory_order_release);
  co_return 0;
}

Task<int> slow_reader_client(AllocatorRef allocator, IoContext *io,
                             BackpressureResult *result, std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made)
    co_return 0;
  HttpClientConnection connection = std::move(*made);
  HttpRequestHead request;
  request.authority = "127.0.0.1";
  auto started = co_await connection.start_request(request, 0);
  if (!started)
    co_return 0;
  auto writer = std::move(*started);
  if (!(co_await writer.finish()).ok())
    co_return 0;

  // Keep the connection open but deliberately do not consume the response.
  // The server must stop pulling the source when the socket applies pressure;
  // it may not eagerly queue the entire 512 MiB logical body.
  hyle::sched::sleep_for(std::chrono::milliseconds(120));
  (void)co_await connection.close();
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttpBudget, SlowConsumerAppliesBoundedWriterBackpressure) {
  constexpr std::uint16_t kPort = 54446;
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  BackpressureResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = slow_reader_server(system_allocator(), &listener, &result);
  client->task = slow_reader_client(system_allocator(), &io, &result, kPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::DeadlineExceeded);
  const std::size_t calls = result.source_calls.load(std::memory_order_acquire);
  EXPECT_GT(calls, 0u);
  EXPECT_LT(calls, BackpressureSource::kChunkCount);
}

void reset_connection_on_close(TcpStream &stream) noexcept {
  linger reset{1, 0};
#if defined(_WIN32)
  (void)::setsockopt(static_cast<SOCKET>(stream.native_handle()), SOL_SOCKET,
                     SO_LINGER, reinterpret_cast<const char *>(&reset),
                     sizeof(reset));
#else
  (void)::setsockopt(static_cast<int>(stream.native_handle()), SOL_SOCKET,
                     SO_LINGER, &reset, sizeof(reset));
#endif
}

Task<int> disconnected_writer_server(AllocatorRef allocator,
                                     TcpListener *listener,
                                     BackpressureResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection connection = std::move(*made);
  auto request = co_await connection.read_request();
  if (!request || !request->has_value())
    co_return 0;
  if (!(co_await request->value().body.discard()).ok())
    co_return 0;
  hyle::sched::sleep_for(std::chrono::milliseconds(20));

  BackpressureSource source{result};
  constexpr std::uint64_t kBodyBytes =
      BackpressureSource::kChunkCount * 256ull * 1024ull;
  const Status sent = co_await connection.send_response(
      HttpResponseHead{}, HttpBodySource::from(source, kBodyBytes),
      OperationOptions{
          .deadline = hyle::time::deadline_after(std::chrono::seconds(1))});
  result->status.store(sent.code(), std::memory_order_release);
  co_return 0;
}

Task<int> disconnecting_client(AllocatorRef allocator, IoContext *io,
                               BackpressureResult *result, std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  reset_connection_on_close(stream);
  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made)
    co_return 0;
  HttpClientConnection connection = std::move(*made);
  HttpRequestHead request;
  request.authority = "127.0.0.1";
  auto started = co_await connection.start_request(request, 0);
  if (!started)
    co_return 0;
  auto writer = std::move(*started);
  if (!(co_await writer.finish()).ok())
    co_return 0;
  connection = HttpClientConnection{};
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttpBudget, PeerDisconnectStopsStreamingProducer) {
  constexpr std::uint16_t kPort = 54447;
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  BackpressureResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      disconnected_writer_server(system_allocator(), &listener, &result);
  client->task = disconnecting_client(system_allocator(), &io, &result, kPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  const StatusCode status = result.status.load(std::memory_order_acquire);
  EXPECT_NE(status, StatusCode::Ok);
  EXPECT_NE(status, StatusCode::DeadlineExceeded);
  EXPECT_LT(result.source_calls.load(std::memory_order_acquire),
            BackpressureSource::kChunkCount);
}

} // namespace
