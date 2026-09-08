#include <gtest/gtest.h>

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#else
#  include <fcntl.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

#include "hyle/io/http2.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::Status;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::Http2ClientSession;
using hyle::io::Http2IncomingRequest;
using hyle::io::Http2IncomingResponse;
using hyle::io::Http2RequestHead;
using hyle::io::Http2ResponseHead;
using hyle::io::Http2ServerSession;
using hyle::io::Http2StreamId;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::Event;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::uint16_t kPort = 54428;

std::span<const std::byte> bytes(std::string_view text) noexcept {
  return {reinterpret_cast<const std::byte *>(text.data()), text.size()};
}

OperationOptions budget() noexcept {
  return OperationOptions{.deadline = hyle::time::deadline_after(
                              hyle::time::Duration(5'000'000'000))};
}

void *oom_allocate(void *, std::size_t, std::size_t) noexcept {
  return nullptr;
}

void oom_deallocate(void *, void *, std::size_t, std::size_t) noexcept {}

void *oom_reallocate(void *, void *, std::size_t, std::size_t,
                     std::size_t) noexcept {
  return nullptr;
}

Task<Status> read_body(AllocatorRef, hyle::io::Http2BodyReader &body,
                       std::string *output,
                       std::size_t max_bytes = 4096) noexcept {
  for (;;) {
    auto next = co_await body.read(max_bytes, budget());
    if (!next.has_value())
      co_return std::move(next).error();
    output->append(reinterpret_cast<const char *>(next->bytes.data()),
                   next->bytes.size());
    if (next->end_stream) {
      if (!body.done())
        co_return Status::Internal(
            "HTTP/2 reader did not expose final-chunk completion");
      co_return Status::Ok();
    }
  }
}

struct Drive {
  Task<int> task;
};

void drive(void *opaque) noexcept {
  static_cast<Drive *>(opaque)->task.resume();
}

struct MultiplexResult {
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
  Http2StreamId submitted[2]{};
  Http2StreamId completed[2]{};
  int statuses[2]{};
  std::string paths[2];
  std::string requests[2];
  std::string responses[2];
};

Task<int> multiplex_server(AllocatorRef allocator, TcpListener *listener,
                           MultiplexResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);

  Http2StreamId ids[2]{};
  for (std::size_t index = 0; index < 2; ++index) {
    auto next = co_await session.accept(budget());
    if (!next.has_value() || !next->has_value())
      co_return 0;
    Http2IncomingRequest request = std::move(**next);
    ids[index] = request.stream_id;
    result->paths[index].assign(request.head.path().data(),
                                request.head.path().size());
    Status read =
        co_await read_body(allocator, request.body, &result->requests[index]);
    if (!read.ok())
      co_return 0;
  }

  const Http2ResponseHead second{.status = 202};
  if (!(co_await session.send_response(ids[1], second, bytes("response-two"),
                                       budget()))
           .ok())
    co_return 0;
  const Http2ResponseHead first{.status = 201};
  if (!(co_await session.send_response(ids[0], first, bytes("response-one"),
                                       budget()))
           .ok())
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> multiplex_client(AllocatorRef allocator, IoContext *io,
                           MultiplexResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);

  Http2RequestHead one{.method = "POST",
                       .scheme = "http",
                       .authority = "127.0.0.1",
                       .path = "/one"};
  auto first =
      co_await session.send_request(one, bytes("request-one"), budget());
  if (!first.has_value())
    co_return 0;
  result->submitted[0] = *first;
  Http2RequestHead two = one;
  two.path = "/two";
  auto second =
      co_await session.send_request(two, bytes("request-two"), budget());
  if (!second.has_value())
    co_return 0;
  result->submitted[1] = *second;

  for (std::size_t index = 0; index < 2; ++index) {
    auto next = co_await session.next_response(budget());
    if (!next.has_value() || !next->has_value())
      co_return 0;
    Http2IncomingResponse response = std::move(**next);
    result->completed[index] = response.stream_id;
    result->statuses[index] = response.head.status();
    Status read =
        co_await read_body(allocator, response.body, &result->responses[index]);
    if (!read.ok())
      co_return 0;
  }
  result->client_ok.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, MultiplexesIncrementalMessagesInCompletionOrder) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<MultiplexResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = multiplex_server(system_allocator(), &listener, result.get());
  client->task = multiplex_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load());
  EXPECT_TRUE(result->client_ok.load());
  EXPECT_TRUE(result->submitted[0].valid());
  EXPECT_GT(result->submitted[1].value(), result->submitted[0].value());
  EXPECT_EQ(result->paths[0], "/one");
  EXPECT_EQ(result->paths[1], "/two");
  EXPECT_EQ(result->requests[0], "request-one");
  EXPECT_EQ(result->requests[1], "request-two");
  EXPECT_EQ(result->completed[0], result->submitted[1]);
  EXPECT_EQ(result->statuses[0], 202);
  EXPECT_EQ(result->responses[0], "response-two");
  EXPECT_EQ(result->completed[1], result->submitted[0]);
  EXPECT_EQ(result->statuses[1], 201);
  EXPECT_EQ(result->responses[1], "response-one");
}

struct FlowResult {
  std::string request = std::string(192 * 1024, 'q');
  std::string response = std::string(256 * 1024, 'r');
  std::atomic<std::size_t> server_bytes{0};
  std::atomic<std::size_t> client_bytes{0};
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
};

struct StringBodySource {
  const std::string *body = nullptr;
  std::size_t offset = 0;
  std::size_t chunk_bytes = 4093;

  Task<hyle::Result<hyle::io::HttpBodyChunk>>
  next(AllocatorRef allocator, OperationOptions options) noexcept {
    return next_impl(allocator, this, options);
  }

private:
  static Task<hyle::Result<hyle::io::HttpBodyChunk>>
  next_impl(AllocatorRef, StringBodySource *self, OperationOptions) noexcept {
    const std::size_t remaining = self->body->size() - self->offset;
    const std::size_t count = (std::min)(remaining, self->chunk_bytes);
    const auto chunk = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(self->body->data()) + self->offset,
        count);
    self->offset += count;
    co_return hyle::io::HttpBodyChunk{.bytes = chunk,
                                      .trailers = {},
                                      .end_stream =
                                          self->offset == self->body->size()};
  }
};

Task<int> flow_server(AllocatorRef allocator, TcpListener *listener,
                      FlowResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  auto next = co_await session.accept(budget());
  if (!next.has_value() || !next->has_value())
    co_return 0;
  Http2IncomingRequest request = std::move(**next);
  std::string received;
  Status read = co_await read_body(allocator, request.body, &received);
  if (!read.ok())
    co_return 0;
  result->server_bytes.store(received.size(), std::memory_order_release);
  StringBodySource source{.body = &result->response};
  Status sent = co_await session.send_response(
      request.stream_id, Http2ResponseHead{},
      hyle::io::HttpBodySource::from(source, result->response.size()),
      budget());
  result->server_ok.store(sent.ok(), std::memory_order_release);
  co_return 0;
}

Task<int> flow_client(AllocatorRef allocator, IoContext *io,
                      FlowResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 1), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  Http2RequestHead head{.method = "POST",
                        .scheme = "http",
                        .authority = "127.0.0.1",
                        .path = "/flow"};
  StringBodySource source{.body = &result->request};
  auto submitted = co_await session.send_request(
      head, hyle::io::HttpBodySource::from(source, result->request.size()),
      budget());
  if (!submitted.has_value())
    co_return 0;
  auto response = co_await session.read_response(*submitted, budget());
  if (!response.has_value() || response->head.status() != 200)
    co_return 0;
  std::string received;
  Status read = co_await read_body(allocator, response->body, &received);
  result->client_bytes.store(received.size(), std::memory_order_release);
  result->client_ok.store(read.ok(), std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, ConsumerProgressDrivesBothFlowControlWindows) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 1));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<FlowResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = flow_server(system_allocator(), &listener, result.get());
  client->task = flow_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load());
  EXPECT_TRUE(result->client_ok.load());
  EXPECT_EQ(result->server_bytes.load(), result->request.size());
  EXPECT_EQ(result->client_bytes.load(), result->response.size());
}

#if !HYLE_OS_WINDOWS
constexpr std::size_t kDuplexStreams = 4;

struct DuplexProgressResult {
  std::string request = std::string(256 * 1024, 'd');
  std::string response = std::string(256 * 1024, 'u');
  std::array<std::string, kDuplexStreams> requests;
  std::array<std::string, kDuplexStreams> responses;
  std::array<Http2StreamId, kDuplexStreams> submitted{};
  std::array<Http2StreamId, kDuplexStreams> completed{};
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
  Event client_consumed;
};

bool make_small_socket_pair(std::array<int, 2> &sockets) noexcept {
  sockets = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets.data()) != 0)
    return false;
  constexpr int kSocketBufferBytes = 4 * 1024;
  for (const int socket : sockets) {
    const int flags = ::fcntl(socket, F_GETFL, 0);
    if (flags < 0 || ::fcntl(socket, F_SETFL, flags | O_NONBLOCK) != 0 ||
        ::setsockopt(socket, SOL_SOCKET, SO_SNDBUF, &kSocketBufferBytes,
                     sizeof(kSocketBufferBytes)) != 0 ||
        ::setsockopt(socket, SOL_SOCKET, SO_RCVBUF, &kSocketBufferBytes,
                     sizeof(kSocketBufferBytes)) != 0) {
      for (int &owned : sockets) {
        if (owned >= 0)
          (void)::close(owned);
        owned = -1;
      }
      return false;
    }
  }
  return true;
}

Task<int> duplex_progress_server(AllocatorRef allocator, TcpStream transport,
                                 DuplexProgressResult *result) noexcept {
  hyle::io::Http2Limits limits;
  limits.connection_receive_window = 65'535;
  auto made =
      Http2ServerSession::create(allocator, std::move(transport), limits);
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  for (std::size_t base = 0; base < kDuplexStreams; base += 2) {
    std::array<Http2StreamId, 2> ids{};
    for (std::size_t offset = 0; offset < ids.size(); ++offset) {
      auto incoming = co_await session.accept(budget());
      if (!incoming.has_value() || !incoming->has_value())
        co_return 0;
      Http2IncomingRequest request = std::move(**incoming);
      ids[offset] = request.stream_id;
      Status consumed = co_await read_body(
          allocator, request.body, &result->requests[base + offset], 16 * 1024);
      if (!consumed.ok())
        co_return 0;
    }
    for (std::size_t offset = ids.size(); offset-- != 0;) {
      Status sent = co_await session.send_response(
          ids[offset], Http2ResponseHead{}, bytes(result->response), budget());
      if (!sent.ok())
        co_return 0;
    }
  }
  result->server_ok.store(true, std::memory_order_release);
  // The protocol-under-test is a persistent session. Keep it alive until the
  // peer has consumed the final DATA/END_STREAM; otherwise unread WINDOW_UPDATE
  // frames can turn socket destruction into an unrelated abortive-close test.
  (void)result->client_consumed.wait_for(hyle::time::Duration(5'000'000'000));
  co_return 0;
}

Task<int> duplex_progress_client(AllocatorRef allocator, TcpStream transport,
                                 DuplexProgressResult *result) noexcept {
  hyle::io::Http2Limits limits;
  limits.connection_receive_window = 65'535;
  auto made =
      Http2ClientSession::create(allocator, std::move(transport), limits);
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  Http2RequestHead head{.method = "POST",
                        .scheme = "http",
                        .authority = "localhost",
                        .path = "/duplex-progress"};
  for (std::size_t base = 0; base < kDuplexStreams; base += 2) {
    for (std::size_t offset = 0; offset < 2; ++offset) {
      auto sent = co_await session.send_request(head, bytes(result->request),
                                                budget());
      if (!sent.has_value())
        co_return 0;
      result->submitted[base + offset] = *sent;
    }
    for (std::size_t offset = 0; offset < 2; ++offset) {
      auto incoming = co_await session.next_response(budget());
      if (!incoming.has_value() || !incoming->has_value())
        co_return 0;
      Http2IncomingResponse response = std::move(**incoming);
      result->completed[base + offset] = response.stream_id;
      Status consumed = co_await read_body(
          allocator, response.body, &result->responses[base + offset],
          16 * 1024);
      if (!consumed.ok())
        co_return 0;
    }
  }
  result->client_ok.store(true, std::memory_order_release);
  result->client_consumed.signal();
  co_return 0;
}

TEST(IoHttp2, DuplexProgressSurvivesSmallSocketAndFlowControl) {
  std::array<int, 2> sockets{};
  ASSERT_TRUE(make_small_socket_pair(sockets));
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto client_transport = TcpStream::adopt(io, kOwner, sockets[0]);
  sockets[0] = -1;
  ASSERT_TRUE(client_transport.has_value());
  auto server_transport = TcpStream::adopt(io, kOwner, sockets[1]);
  sockets[1] = -1;
  ASSERT_TRUE(server_transport.has_value());

  auto result = std::make_unique<DuplexProgressResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = duplex_progress_server(system_allocator(),
                                         std::move(*server_transport),
                                         result.get());
  client->task = duplex_progress_client(system_allocator(),
                                         std::move(*client_transport),
                                         result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load());
  EXPECT_TRUE(result->client_ok.load());
  for (std::size_t index = 0; index < kDuplexStreams; ++index) {
    EXPECT_EQ(result->requests[index], result->request);
    EXPECT_EQ(result->responses[index], result->response);
  }
  for (std::size_t base = 0; base < kDuplexStreams; base += 2) {
    EXPECT_EQ(result->completed[base], result->submitted[base + 1]);
    EXPECT_EQ(result->completed[base + 1], result->submitted[base]);
  }
}
#endif

struct ReuseResult {
  std::atomic<std::size_t> server_count{0};
  std::atomic<std::size_t> client_count{0};
};

Task<int> reuse_server(AllocatorRef allocator, TcpListener *listener,
                       ReuseResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  hyle::io::Http2Limits limits;
  limits.max_concurrent_streams = 1;
  auto made =
      Http2ServerSession::create(allocator, std::move(*accepted), limits);
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  for (std::size_t index = 0; index < 8; ++index) {
    auto next = co_await session.accept(budget());
    if (!next.has_value() || !next->has_value())
      co_return 0;
    {
      Http2IncomingRequest request = std::move(**next);
      if (!(co_await request.body.discard(budget())).ok())
        co_return 0;
      if (!(co_await session.send_response(
                request.stream_id, Http2ResponseHead{.status = 204},
                std::span<const std::byte>{}, budget()))
               .ok())
        co_return 0;
    }
    result->server_count.fetch_add(1, std::memory_order_release);
  }
  co_return 0;
}

Task<int> reuse_client(AllocatorRef allocator, IoContext *io,
                       ReuseResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 2), budget()))
           .ok())
    co_return 0;
  hyle::io::Http2Limits limits;
  limits.max_concurrent_streams = 1;
  auto made =
      Http2ClientSession::create(allocator, std::move(transport), limits);
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  const Http2RequestHead head{.method = "GET",
                              .scheme = "http",
                              .authority = "127.0.0.1",
                              .path = "/reuse"};
  for (std::size_t index = 0; index < 8; ++index) {
    auto submitted = co_await session.send_request(
        head, std::span<const std::byte>{}, budget());
    if (!submitted.has_value())
      co_return 0;
    {
      auto response = co_await session.read_response(*submitted, budget());
      if (!response.has_value() || response->head.status() != 204 ||
          !(co_await response->body.discard(budget())).ok())
        co_return 0;
    }
    result->client_count.fetch_add(1, std::memory_order_release);
  }
  co_return 0;
}

TEST(IoHttp2, ConcurrentLimitDoesNotBecomeLifetimeRequestLimit) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 2));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<ReuseResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = reuse_server(system_allocator(), &listener, result.get());
  client->task = reuse_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_EQ(result->server_count.load(), 8u);
  EXPECT_EQ(result->client_count.load(), 8u);
}

struct TrailerResult {
  std::string request_body;
  std::string request_trailer;
  std::string response_body;
  std::string response_trailer;
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
};

Task<Status> read_body_and_trailer(AllocatorRef,
                                   hyle::io::Http2BodyReader &body,
                                   std::string *output,
                                   std::string *trailer) noexcept {
  for (;;) {
    auto chunk = co_await body.read(2, budget());
    if (!chunk.has_value())
      co_return std::move(chunk).error();
    output->append(reinterpret_cast<const char *>(chunk->bytes.data()),
                   chunk->bytes.size());
    if (chunk->end_stream) {
      if (!body.done())
        co_return Status::Internal(
            "HTTP/2 reader did not expose trailer completion");
      auto value = chunk->trailers.find("x-hyle-check");
      if (value.has_value())
        trailer->assign(value->data(), value->size());
      co_return Status::Ok();
    }
  }
}

Task<int> trailer_server(AllocatorRef allocator, TcpListener *listener,
                         TrailerResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  auto incoming = co_await session.accept(budget());
  if (!incoming.has_value() || !incoming->has_value())
    co_return 0;
  Http2IncomingRequest request = std::move(**incoming);
  if (!(co_await read_body_and_trailer(allocator, request.body,
                                       &result->request_body,
                                       &result->request_trailer))
           .ok())
    co_return 0;

  auto started = co_await session.start_response(
      request.stream_id, Http2ResponseHead{.status = 200}, 6, budget());
  if (!started.has_value() ||
      !(co_await started->write(bytes("ghi"), budget())).ok() ||
      !(co_await started->write(bytes("jkl"), budget())).ok())
    co_return 0;
  const hyle::io::HttpHeader trailer{"x-hyle-check", "response"};
  if (!(co_await started->finish(std::span(&trailer, 1), budget())).ok())
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> trailer_client(AllocatorRef allocator, IoContext *io,
                         TrailerResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 3), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  auto started =
      co_await session.start_request(Http2RequestHead{.method = "POST",
                                                      .scheme = "http",
                                                      .authority = "127.0.0.1",
                                                      .path = "/trailers"},
                                     6, budget());
  if (!started.has_value() ||
      !(co_await started->body.write(bytes("abc"), budget())).ok() ||
      !(co_await started->body.write(bytes("def"), budget())).ok())
    co_return 0;
  const hyle::io::HttpHeader trailer{"x-hyle-check", "request"};
  if (!(co_await started->body.finish(std::span(&trailer, 1), budget())).ok())
    co_return 0;
  auto response = co_await session.read_response(started->stream_id, budget());
  if (!response.has_value() || response->head.status() != 200 ||
      !(co_await read_body_and_trailer(allocator, response->body,
                                       &result->response_body,
                                       &result->response_trailer))
           .ok())
    co_return 0;
  result->client_ok.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, StreamsFragmentedBodiesAndPublishesTrailersAtEnd) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 3));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<TrailerResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = trailer_server(system_allocator(), &listener, result.get());
  client->task = trailer_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load());
  EXPECT_TRUE(result->client_ok.load());
  EXPECT_EQ(result->request_body, "abcdef");
  EXPECT_EQ(result->request_trailer, "request");
  EXPECT_EQ(result->response_body, "ghijkl");
  EXPECT_EQ(result->response_trailer, "response");
}

struct ResetResult {
  std::atomic<bool> cancelled_before_io{false};
  std::atomic<bool> deadline_before_io{false};
  std::atomic<bool> reset_observed{false};
  std::atomic<bool> session_reused{false};
  std::atomic<bool> server_ok{false};
};

Task<int> reset_server(AllocatorRef allocator, TcpListener *listener,
                       ResetResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  {
    auto incoming = co_await session.accept(budget());
    if (!incoming.has_value() || !incoming->has_value())
      co_return 0;
    Http2IncomingRequest request = std::move(**incoming);
    if (!(co_await request.body.discard(budget())).ok() ||
        !(co_await session.reset(request.stream_id,
                                 hyle::io::Http2Error::kCancel, budget()))
             .ok())
      co_return 0;
  }
  {
    auto incoming = co_await session.accept(budget());
    if (!incoming.has_value() || !incoming->has_value())
      co_return 0;
    Http2IncomingRequest request = std::move(**incoming);
    if (!(co_await request.body.discard(budget())).ok() ||
        !(co_await session.send_response(request.stream_id,
                                         Http2ResponseHead{.status = 200},
                                         bytes("after-reset"), budget()))
             .ok())
      co_return 0;
  }
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> reset_client(AllocatorRef allocator, IoContext *io,
                       ResetResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 4), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);

  auto cancel = hyle::sched::CancelToken::try_create_root(allocator);
  if (!cancel.has_value())
    co_return 0;
  cancel->cancel();
  auto cancelled = co_await session.next_response(
      OperationOptions{.deadline = {}, .cancel = &*cancel});
  result->cancelled_before_io.store(!cancelled.has_value() &&
                                        cancelled.error().IsCancelled(),
                                    std::memory_order_release);

  auto expired = co_await session.next_response(OperationOptions{
      .deadline = hyle::time::deadline_after(hyle::time::Duration::zero())});
  result->deadline_before_io.store(!expired.has_value() &&
                                       expired.error().IsDeadlineExceeded(),
                                   std::memory_order_release);

  const Http2RequestHead head{.method = "GET",
                              .scheme = "http",
                              .authority = "127.0.0.1",
                              .path = "/reset"};
  auto first = co_await session.send_request(head, std::span<const std::byte>{},
                                             budget());
  if (!first.has_value())
    co_return 0;
  {
    auto response = co_await session.read_response(*first, budget());
    if (!response.has_value() || !response->stream_error.has_value() ||
        *response->stream_error != hyle::io::Http2Error::kCancel)
      co_return 0;
    auto body = co_await response->body.read(0, budget());
    result->reset_observed.store(!body.has_value() &&
                                     body.error().IsCancelled(),
                                 std::memory_order_release);
  }

  auto second =
      co_await session.send_request(Http2RequestHead{.method = "GET",
                                                     .scheme = "http",
                                                     .authority = "127.0.0.1",
                                                     .path = "/after-reset"},
                                    std::span<const std::byte>{}, budget());
  if (!second.has_value())
    co_return 0;
  auto response = co_await session.read_response(*second, budget());
  if (!response.has_value() || response->head.status() != 200)
    co_return 0;
  std::string body;
  Status read = co_await read_body(allocator, response->body, &body);
  result->session_reused.store(read.ok() && body == "after-reset",
                               std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, OperationCancellationAndStreamResetDoNotPoisonSession) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 4));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<ResetResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = reset_server(system_allocator(), &listener, result.get());
  client->task = reset_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->cancelled_before_io.load());
  EXPECT_TRUE(result->deadline_before_io.load());
  EXPECT_TRUE(result->reset_observed.load());
  EXPECT_TRUE(result->session_reused.load());
  EXPECT_TRUE(result->server_ok.load());
}

struct GoawayResult {
  std::atomic<bool> server_drained{false};
  std::atomic<bool> client_saw_goaway{false};
  std::atomic<bool> client_rejected_new_stream{false};
};

Task<int> goaway_server(AllocatorRef allocator, TcpListener *listener,
                        GoawayResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  {
    auto incoming = co_await session.accept(budget());
    if (!incoming.has_value() || !incoming->has_value())
      co_return 0;
    Http2IncomingRequest request = std::move(**incoming);
    if (!(co_await request.body.discard(budget())).ok() ||
        !(co_await session.send_response(request.stream_id,
                                         Http2ResponseHead{.status = 200},
                                         bytes("done"), budget()))
             .ok())
      co_return 0;
  }
  Status drained =
      co_await session.drain(hyle::io::Http2Error::kNoError, budget());
  result->server_drained.store(drained.ok(), std::memory_order_release);
  co_return 0;
}

Task<int> goaway_client(AllocatorRef allocator, IoContext *io,
                        GoawayResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 5), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  const Http2RequestHead head{.method = "GET",
                              .scheme = "http",
                              .authority = "127.0.0.1",
                              .path = "/drain"};
  auto submitted = co_await session.send_request(
      head, std::span<const std::byte>{}, budget());
  if (!submitted.has_value())
    co_return 0;
  {
    auto response = co_await session.read_response(*submitted, budget());
    if (!response.has_value() ||
        !(co_await response->body.discard(budget())).ok())
      co_return 0;
  }
  auto end = co_await session.next_response(budget());
  const auto peer = session.peer_goaway();
  result->client_saw_goaway.store(
      end.has_value() && !end->has_value() && peer.has_value() &&
          peer->error == hyle::io::Http2Error::kNoError,
      std::memory_order_release);
  auto rejected = co_await session.send_request(
      head, std::span<const std::byte>{}, budget());
  result->client_rejected_new_stream.store(!rejected.has_value() &&
                                               rejected.error().IsUnavailable(),
                                           std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, GoawayDrainsExistingStreamAndRejectsNewStreams) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 5));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<GoawayResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = goaway_server(system_allocator(), &listener, result.get());
  client->task = goaway_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_drained.load());
  EXPECT_TRUE(result->client_saw_goaway.load());
  EXPECT_TRUE(result->client_rejected_new_stream.load());
}

struct ValidationResult {
  std::atomic<bool> oom_reported{false};
  std::atomic<bool> forbidden_header_reported{false};
  std::atomic<bool> header_limit_reported{false};
  std::atomic<bool> typed_content_length_required{false};
  std::atomic<bool> valid_token_case_accepted{false};
  std::atomic<bool> server_ok{false};
};

Task<int> validation_server(AllocatorRef allocator, TcpListener *listener,
                            ValidationResult *result) noexcept {
  {
    auto oom_connection = co_await listener->accept(kOwner);
    if (!oom_connection.has_value())
      co_return 0;
  }
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  auto incoming = co_await session.accept(budget());
  if (!incoming.has_value() || !incoming->has_value())
    co_return 0;
  Http2IncomingRequest request = std::move(**incoming);
  auto te = request.head.headers().find("te");
  if (!te.has_value() ||
      std::string_view(te->data(), te->size()) != "Trailers" ||
      !(co_await request.body.discard(budget())).ok() ||
      !(co_await session.send_response(request.stream_id,
                                       Http2ResponseHead{.status = 204},
                                       std::span<const std::byte>{}, budget()))
           .ok())
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> validation_client(AllocatorRef allocator, IoContext *io,
                            ValidationResult *result) noexcept {
  {
    auto opened = TcpStream::open(*io, kOwner);
    if (!opened.has_value())
      co_return 0;
    TcpStream transport = std::move(*opened);
    if (!(co_await transport.connect(
              Endpoint::loopback(AddressFamily::kIpv4, kPort + 6), budget()))
             .ok())
      co_return 0;
    AllocatorRef oom(&oom_allocate, &oom_deallocate, &oom_reallocate, nullptr);
    auto failed = Http2ClientSession::create(oom, std::move(transport));
    result->oom_reported.store(!failed.has_value() &&
                                   failed.error().IsResourceExhausted(),
                               std::memory_order_release);
  }

  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 6), budget()))
           .ok())
    co_return 0;
  hyle::io::Http2Limits limits;
  limits.max_header_count = 1;
  auto made =
      Http2ClientSession::create(allocator, std::move(transport), limits);
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  const Http2RequestHead base{.method = "GET",
                              .scheme = "http",
                              .authority = "127.0.0.1",
                              .path = "/validation"};

  const hyle::io::HttpHeader forbidden{"connection", "close"};
  Http2RequestHead forbidden_head = base;
  forbidden_head.headers = std::span(&forbidden, 1);
  auto forbidden_result = co_await session.send_request(
      forbidden_head, std::span<const std::byte>{}, budget());
  result->forbidden_header_reported.store(
      !forbidden_result.has_value() &&
          forbidden_result.error().IsInvalidArgument(),
      std::memory_order_release);

  const hyle::io::HttpHeader too_many[] = {{"x-one", "1"}, {"x-two", "2"}};
  Http2RequestHead limited_head = base;
  limited_head.headers = too_many;
  auto limited_result = co_await session.send_request(
      limited_head, std::span<const std::byte>{}, budget());
  result->header_limit_reported.store(
      !limited_result.has_value() &&
          limited_result.error().IsResourceExhausted(),
      std::memory_order_release);

  const hyle::io::HttpHeader raw_length{"content-length", "0"};
  Http2RequestHead raw_length_head = base;
  raw_length_head.headers = std::span(&raw_length, 1);
  auto raw_length_result =
      co_await session.start_request(raw_length_head, std::nullopt, budget());
  result->typed_content_length_required.store(
      !raw_length_result.has_value() &&
          raw_length_result.error().IsInvalidArgument(),
      std::memory_order_release);

  const hyle::io::HttpHeader te{"te", "Trailers"};
  Http2RequestHead valid_head = base;
  valid_head.headers = std::span(&te, 1);
  auto submitted = co_await session.send_request(
      valid_head, std::span<const std::byte>{}, budget());
  if (!submitted.has_value())
    co_return 0;
  auto response = co_await session.read_response(*submitted, budget());
  result->valid_token_case_accepted.store(
      response.has_value() && response->head.status() == 204 &&
          (co_await response->body.discard(budget())).ok(),
      std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, RejectsInvalidHeadersAndSurfacesAllocatorAndLimitFailures) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 6));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<ValidationResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = validation_server(system_allocator(), &listener, result.get());
  client->task = validation_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->oom_reported.load());
  EXPECT_TRUE(result->forbidden_header_reported.load());
  EXPECT_TRUE(result->header_limit_reported.load());
  EXPECT_TRUE(result->typed_content_length_required.load());
  EXPECT_TRUE(result->valid_token_case_accepted.load());
  EXPECT_TRUE(result->server_ok.load());
}

struct InvalidWireResult {
  std::atomic<bool> rejected{false};
  std::atomic<bool> client_wrote{false};
  std::atomic<int> server_result{-1};
};

Task<int> invalid_wire_server(AllocatorRef allocator, TcpListener *listener,
                              InvalidWireResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  auto made = Http2ServerSession::create(allocator, std::move(*accepted));
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  auto incoming = co_await session.accept(budget());
  result->server_result.store(incoming.has_value()
                                  ? (incoming->has_value() ? 100 : 0)
                                  : static_cast<int>(incoming.error().code()),
                              std::memory_order_release);
  result->rejected.store(!incoming.has_value() && incoming.error().IsDataLoss(),
                         std::memory_order_release);
  co_return 0;
}

Task<int> invalid_wire_client(AllocatorRef allocator, IoContext *io,
                              InvalidWireResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 7), budget()))
           .ok())
    co_return 0;
  constexpr std::string_view invalid_preface =
      "BAD * HTTP/2.0\r\n\r\nSM\r\n\r\n";
  auto written = co_await hyle::io::write_all(allocator, stream,
                                              bytes(invalid_preface), budget());
  result->client_wrote.store(written.ok() &&
                                 written.bytes == invalid_preface.size(),
                             std::memory_order_release);
  (void)stream.shutdown_write();
  std::byte incoming[4096];
  for (;;) {
    auto received = co_await stream.read_some(incoming, budget());
    if (!received.ok() || received.eof || received.bytes == 0)
      break;
  }
  co_return 0;
}

TEST(IoHttp2, RejectsInvalidClientPrefaceAsProtocolDataLoss) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 7));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<InvalidWireResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      invalid_wire_server(system_allocator(), &listener, result.get());
  client->task = invalid_wire_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->client_wrote.load());
  EXPECT_TRUE(result->rejected.load());
  EXPECT_EQ(result->server_result.load(),
            static_cast<int>(hyle::StatusCode::DataLoss));
}

struct IsolationResult {
  std::string abandoned_body = std::string(48 * 1024, 'a');
  std::string oversized_body = std::string(72 * 1024, 'b');
  std::atomic<bool> abandoned_reader_reset{false};
  std::atomic<bool> body_limit_reported{false};
  std::atomic<bool> session_reused{false};
  std::atomic<bool> server_ok{false};
  std::atomic<int> server_stage{0};
  std::atomic<int> server_error{-1};
  std::atomic<int> client_stage{0};
};

Task<int> isolation_server(AllocatorRef allocator, TcpListener *listener,
                           IsolationResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  hyle::io::Http2Limits limits;
  limits.max_body_bytes = 64 * 1024;
  limits.stream_receive_window = 4 * 1024;
  limits.connection_receive_window = 64 * 1024;
  limits.read_chunk_bytes = 1024;
  auto made =
      Http2ServerSession::create(allocator, std::move(*accepted), limits);
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);
  result->server_stage.store(1, std::memory_order_release);

  {
    auto incoming = co_await session.accept(budget());
    if (!incoming.has_value() || !incoming->has_value()) {
      if (!incoming.has_value()) {
        result->server_error.store(static_cast<int>(incoming.error().code()),
                                   std::memory_order_release);
      }
      co_return 0;
    }
    Http2IncomingRequest request = std::move(**incoming);
    auto first = co_await request.body.read(1024, budget());
    if (!first.has_value() || first->bytes.empty() || first->end_stream)
      co_return 0;
    result->server_stage.store(2, std::memory_order_release);
    // Dropping the unfinished reader must return buffered connection credit,
    // send RST_STREAM, and leave unrelated streams usable.
  }

  {
    auto incoming = co_await session.accept(budget());
    if (!incoming.has_value() || !incoming->has_value())
      co_return 0;
    result->server_stage.store(3, std::memory_order_release);
    Http2IncomingRequest request = std::move(**incoming);
    Status discarded = co_await request.body.discard(budget());
    if (!discarded.ok())
      result->server_error.store(static_cast<int>(discarded.code()),
                                 std::memory_order_release);
    result->body_limit_reported.store(discarded.IsResourceExhausted(),
                                      std::memory_order_release);
  }

  auto incoming = co_await session.accept(budget());
  if (!incoming.has_value() || !incoming->has_value())
    co_return 0;
  result->server_stage.store(4, std::memory_order_release);
  Http2IncomingRequest request = std::move(**incoming);
  if (!(co_await request.body.discard(budget())).ok() ||
      !(co_await session.send_response(request.stream_id,
                                       Http2ResponseHead{.status = 204},
                                       std::span<const std::byte>{}, budget()))
           .ok())
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  result->server_stage.store(5, std::memory_order_release);
  co_return 0;
}

Task<int> isolation_client(AllocatorRef allocator, IoContext *io,
                           IsolationResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 8), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  result->client_stage.store(1, std::memory_order_release);
  Http2RequestHead head{.method = "POST",
                        .scheme = "http",
                        .authority = "127.0.0.1",
                        .path = "/abandon"};

  StringBodySource abandoned{.body = &result->abandoned_body};
  auto first = co_await session.send_request(
      head,
      hyle::io::HttpBodySource::from(abandoned, result->abandoned_body.size()),
      budget());
  bool abandoned_reset = !first.has_value();
  if (first.has_value()) {
    auto response = co_await session.read_response(*first, budget());
    abandoned_reset = response.has_value() &&
                      response->stream_error.has_value() &&
                      *response->stream_error == hyle::io::Http2Error::kCancel;
  }
  result->abandoned_reader_reset.store(abandoned_reset,
                                       std::memory_order_release);
  result->client_stage.store(2, std::memory_order_release);

  head.path = "/too-large";
  StringBodySource oversized{.body = &result->oversized_body};
  auto second = co_await session.send_request(
      head,
      hyle::io::HttpBodySource::from(oversized, result->oversized_body.size()),
      budget());
  result->client_stage.store(3, std::memory_order_release);
  if (second.has_value()) {
    auto response = co_await session.read_response(*second, budget());
    if (response.has_value())
      (void)co_await response->body.discard(budget());
  }

  head.method = "GET";
  head.path = "/after-isolation";
  auto third = co_await session.send_request(head, std::span<const std::byte>{},
                                             budget());
  if (!third.has_value())
    co_return 0;
  result->client_stage.store(4, std::memory_order_release);
  auto response = co_await session.read_response(*third, budget());
  if (!response.has_value() || response->head.status() != 204 ||
      !(co_await response->body.discard(budget())).ok())
    co_return 0;
  result->session_reused.store(true, std::memory_order_release);
  result->client_stage.store(5, std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, AbandonAndBodyLimitIsolateStreamsWithoutPoisoningSession) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 8));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<IsolationResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = isolation_server(system_allocator(), &listener, result.get());
  client->task = isolation_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->abandoned_reader_reset.load());
  EXPECT_TRUE(result->body_limit_reported.load());
  EXPECT_TRUE(result->session_reused.load());
  EXPECT_TRUE(result->server_ok.load());
  EXPECT_EQ(result->server_stage.load(), 5);
  EXPECT_EQ(result->server_error.load(),
            static_cast<int>(hyle::StatusCode::ResourceExhausted));
  EXPECT_EQ(result->client_stage.load(), 5);
}

struct DeadlineResult {
  std::string stalled_body = std::string(512 * 1024, 'd');
  std::atomic<bool> blocked_write_timed_out{false};
  std::atomic<bool> second_request_sent{false};
  std::atomic<bool> second_request_accepted{false};
  std::atomic<bool> second_response_received{false};
  std::atomic<bool> session_reused{false};
  std::atomic<bool> server_ok{false};
};

Task<int> deadline_server(AllocatorRef allocator, TcpListener *listener,
                          DeadlineResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  hyle::io::Http2Limits limits;
  limits.max_body_bytes = 1024 * 1024;
  limits.stream_receive_window = 64 * 1024;
  limits.connection_receive_window = 64 * 1024;
  auto made =
      Http2ServerSession::create(allocator, std::move(*accepted), limits);
  if (!made.has_value())
    co_return 0;
  Http2ServerSession session = std::move(*made);

  auto first = co_await session.accept(budget());
  if (!first.has_value() || !first->has_value())
    co_return 0;
  Http2IncomingRequest stalled = std::move(**first);
  // Keep this reader alive without consuming DATA. The stream and connection
  // windows must eventually stop the producer until its deadline cancels it.

  auto second = co_await session.accept(budget());
  if (!second.has_value() || !second->has_value())
    co_return 0;
  result->second_request_accepted.store(true, std::memory_order_release);
  Http2IncomingRequest request = std::move(**second);
  Status discarded = co_await request.body.discard(budget());
  if (!discarded.ok())
    co_return 0;
  Status sent = co_await session.send_response(
      request.stream_id, Http2ResponseHead{.status = 204},
      std::span<const std::byte>{}, budget());
  if (!sent.ok())
    co_return 0;
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> deadline_client(AllocatorRef allocator, IoContext *io,
                          DeadlineResult *result) noexcept {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPort + 9), budget()))
           .ok())
    co_return 0;
  auto made = Http2ClientSession::create(allocator, std::move(transport));
  if (!made.has_value())
    co_return 0;
  Http2ClientSession session = std::move(*made);
  Http2RequestHead head{.method = "POST",
                        .scheme = "http",
                        .authority = "127.0.0.1",
                        .path = "/deadline"};
  StringBodySource stalled{.body = &result->stalled_body};
  auto timed_out = co_await session.send_request(
      head,
      hyle::io::HttpBodySource::from(stalled, result->stalled_body.size()),
      OperationOptions{.deadline = hyle::time::deadline_after(
                           hyle::time::Duration(20'000'000))});
  result->blocked_write_timed_out.store(
      !timed_out.has_value() && timed_out.error().IsDeadlineExceeded(),
      std::memory_order_release);

  head.method = "GET";
  head.path = "/after-deadline";
  auto next = co_await session.send_request(head, std::span<const std::byte>{},
                                            budget());
  if (!next.has_value())
    co_return 0;
  result->second_request_sent.store(true, std::memory_order_release);
  auto response = co_await session.read_response(*next, budget());
  if (!response.has_value() || response->head.status() != 204 ||
      !(co_await response->body.discard(budget())).ok())
    co_return 0;
  result->second_response_received.store(true, std::memory_order_release);
  result->session_reused.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttp2, DeadlineInterruptsFlowControlledWriterWithoutPoisoningSession) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort + 9));
  ASSERT_TRUE(listener_result.has_value());
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<DeadlineResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = deadline_server(system_allocator(), &listener, result.get());
  client->task = deadline_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, drive, server.get());
  Fiber client_fiber(kDefaultStackSize, drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->blocked_write_timed_out.load());
  EXPECT_TRUE(result->second_request_sent.load());
  EXPECT_TRUE(result->second_request_accepted.load());
  EXPECT_TRUE(result->second_response_received.load());
  EXPECT_TRUE(result->session_reused.load());
  EXPECT_TRUE(result->server_ok.load());
}

} // namespace
