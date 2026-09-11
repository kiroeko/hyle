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

#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

using hyle::Result;
using hyle::container::StringView;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::HttpClientConnection;
using hyle::io::HttpRequestHead;
using hyle::io::HttpServerConnection;
using hyle::io::HttpTransport;
using hyle::io::HttpUpgradeStream;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::uint16_t kServerHandoffPort = 54533;
constexpr std::uint16_t kClientHandoffPort = 54534;
constexpr std::uint16_t kRejectionPort = 54535;
constexpr StringView kProtocol = "hyle-control/1";

OperationOptions test_operation() noexcept {
  return {.deadline = hyle::time::deadline_after(std::chrono::seconds(5))};
}

std::span<const std::byte> bytes(StringView text) noexcept {
  return {reinterpret_cast<const std::byte *>(text.data()), text.size()};
}

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

Task<bool> write_text(AllocatorRef allocator, TcpStream &stream,
                      StringView text) {
  hyle::io::TransferResult sent = co_await hyle::io::write_all(
      allocator, stream, bytes(text), test_operation());
  co_return sent.ok() && sent.bytes == text.size();
}

Task<bool> read_head(AllocatorRef allocator, TcpStream &stream,
                     std::string &output) {
  std::array<std::byte, 512> buffer{};
  while (output.find("\r\n\r\n") == std::string::npos && output.size() < 4096) {
    hyle::io::TransferResult read =
        co_await stream.read_some(buffer, test_operation());
    if (!read.ok() || read.eof || read.bytes == 0)
      co_return false;
    output.append(reinterpret_cast<const char *>(buffer.data()), read.bytes);
  }
  co_return output.find("\r\n\r\n") != std::string::npos;
}

struct HandoffResult {
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
};

Task<int> server_handoff_server(AllocatorRef allocator, TcpListener *listener,
                                HandoffResult *result) {
  auto accepted = co_await listener->accept(kOwner, test_operation());
  if (!accepted)
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection http = std::move(*made);
  auto incoming_result = co_await http.read_request(test_operation());
  if (!incoming_result || !incoming_result->has_value())
    co_return 0;
  auto incoming = std::move(incoming_result->value());

  // The first attempt must not consume an unfinished request owner.
  auto early_task =
      http.accept_upgrade(std::move(incoming), kProtocol, {}, test_operation());
  if (!early_task.valid())
    co_return 0;
  Result<HttpUpgradeStream> early = co_await early_task;
  if (early || !http.valid() || !incoming.body.valid())
    co_return 0;
  auto body_result = co_await incoming.body.read_all(4, test_operation());
  if (!body_result || *body_result != "BODY")
    co_return 0;

  Result<HttpUpgradeStream> upgraded = co_await http.accept_upgrade(
      std::move(incoming), kProtocol, {}, test_operation());
  if (!upgraded || http.valid() || incoming.body.valid())
    co_return 0;
  HttpUpgradeStream stream = std::move(*upgraded);
  std::array<std::byte, 4> prefix{};
  hyle::io::TransferResult read =
      co_await stream.read_some(prefix, test_operation());
  if (!read.ok() || read.bytes != prefix.size() ||
      std::memcmp(prefix.data(), "NEXT", prefix.size()) != 0) {
    co_return 0;
  }
  result->server_ok.store(true, std::memory_order_release);
  (void)co_await stream.shutdown(test_operation());
  co_return 0;
}

Task<int> server_handoff_client(AllocatorRef allocator, IoContext *io,
                                HandoffResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kServerHandoffPort),
            test_operation()))
           .ok()) {
    co_return 0;
  }
  static constexpr StringView kPacket = "POST /custom HTTP/1.1\r\n"
                                        "Host: 127.0.0.1\r\n"
                                        "Connection: keep-alive, Upgrade\r\n"
                                        "Upgrade: hyle-control/1\r\n"
                                        "Content-Length: 4\r\n\r\n"
                                        "BODYNEXT";
  if (!(co_await write_text(allocator, stream, kPacket)))
    co_return 0;
  std::string response;
  if (!(co_await read_head(allocator, stream, response)))
    co_return 0;
  result->client_ok.store(
      response.find("HTTP/1.1 101 Switching Protocols\r\n") == 0 &&
          response.find("Upgrade: hyle-control/1\r\n") != std::string::npos,
      std::memory_order_release);
  co_return 0;
}

TEST(IoHttpUpgrade, ServerPreservesBodyFailureAndPrefetchedProtocolBytes) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kServerHandoffPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<HandoffResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      server_handoff_server(system_allocator(), &listener, result.get());
  client->task = server_handoff_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_ok.load(std::memory_order_acquire));
}

Task<int> client_handoff_server(AllocatorRef allocator, TcpListener *listener,
                                HandoffResult *result) {
  auto accepted = co_await listener->accept(kOwner, test_operation());
  if (!accepted)
    co_return 0;
  TcpStream stream = std::move(*accepted);
  std::string request;
  if (!(co_await read_head(allocator, stream, request)))
    co_return 0;
  if (request.find("Connection: Upgrade\r\n") == std::string::npos ||
      request.find("Upgrade: hyle-control/1\r\n") == std::string::npos) {
    co_return 0;
  }
  static constexpr StringView kPacket =
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: hyle-control/1\r\n"
      "Connection: keep-alive, Upgrade\r\n\r\n"
      "READY";
  if (!(co_await write_text(allocator, stream, kPacket)))
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> client_handoff_client(AllocatorRef allocator, IoContext *io,
                                HandoffResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kClientHandoffPort),
            test_operation()))
           .ok()) {
    co_return 0;
  }
  auto made =
      HttpClientConnection::create(allocator, HttpTransport(std::move(stream)));
  if (!made)
    co_return 0;
  HttpClientConnection http = std::move(*made);
  const HttpRequestHead head{
      .method = "GET", .target = "/custom", .authority = "127.0.0.1"};
  auto response_result =
      co_await http.request_upgrade(head, kProtocol, test_operation());
  if (!response_result || response_result->head.status() != 101)
    co_return 0;
  auto response = std::move(*response_result);
  Result<HttpUpgradeStream> upgraded =
      http.take_upgrade(std::move(response), kProtocol);
  if (!upgraded || http.valid() || response.body.valid())
    co_return 0;
  HttpUpgradeStream stream_owner = std::move(*upgraded);
  std::array<std::byte, 5> prefix{};
  hyle::io::TransferResult read =
      co_await stream_owner.read_some(prefix, test_operation());
  if (read.ok() && read.bytes == prefix.size() &&
      std::memcmp(prefix.data(), "READY", prefix.size()) == 0) {
    result->client_ok.store(true, std::memory_order_release);
  }
  (void)co_await stream_owner.shutdown(test_operation());
  co_return 0;
}

TEST(IoHttpUpgrade, ClientPreservesPrefetchedProtocolBytes) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kClientHandoffPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<HandoffResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      client_handoff_server(system_allocator(), &listener, result.get());
  client->task = client_handoff_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_ok.load(std::memory_order_acquire));
}

Task<int> rejection_server(AllocatorRef allocator, TcpListener *listener,
                           HandoffResult *result) {
  auto accepted = co_await listener->accept(kOwner, test_operation());
  if (!accepted)
    co_return 0;
  TcpStream stream = std::move(*accepted);
  std::string request;
  if (!(co_await read_head(allocator, stream, request)))
    co_return 0;
  static constexpr StringView kPacket = "HTTP/1.1 403 Forbidden\r\n"
                                        "Content-Length: 6\r\n"
                                        "Connection: close\r\n\r\n"
                                        "denied";
  if (!(co_await write_text(allocator, stream, kPacket)))
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> rejection_client(AllocatorRef allocator, IoContext *io,
                           HandoffResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kRejectionPort),
            test_operation()))
           .ok()) {
    co_return 0;
  }
  auto made =
      HttpClientConnection::create(allocator, HttpTransport(std::move(stream)));
  if (!made)
    co_return 0;
  HttpClientConnection http = std::move(*made);
  const HttpRequestHead head{
      .method = "GET", .target = "/custom", .authority = "127.0.0.1"};
  auto response_result =
      co_await http.request_upgrade(head, kProtocol, test_operation());
  if (!response_result || response_result->head.status() != 403)
    co_return 0;
  auto response = std::move(*response_result);
  Result<HttpUpgradeStream> rejected =
      http.take_upgrade(std::move(response), kProtocol);
  if (rejected || !http.valid() || !response.body.valid())
    co_return 0;
  auto body = co_await response.body.read_all(6, test_operation());
  if (body && *body == "denied")
    result->client_ok.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttpUpgrade, RejectionRemainsAnOrdinaryReadableResponse) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kRejectionPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<HandoffResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = rejection_server(system_allocator(), &listener, result.get());
  client->task = rejection_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_ok.load(std::memory_order_acquire));
}

} // namespace
