#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
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
#include "hyle/sched/timer.h"

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;
using hyle::container::StringView;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::HttpBodyChunk;
using hyle::io::HttpBodyChunkView;
using hyle::io::HttpBodySource;
using hyle::io::HttpClientConnection;
using hyle::io::HttpHeader;
using hyle::io::HttpIncomingResponse;
using hyle::io::HttpLimits;
using hyle::io::HttpRequestHead;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
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
constexpr std::uint16_t kJourneyPort = 54417;

std::span<const std::byte> bytes(StringView text) noexcept {
  return std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(text.data()), text.size());
}

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

void copy_text(char *destination, std::size_t capacity,
               StringView source) noexcept {
  const std::size_t count = std::min<std::size_t>(capacity - 1, source.size());
  if (count != 0)
    std::memcpy(destination, source.data(), count);
  destination[count] = '\0';
}

void *fail_allocate(void *, std::size_t, std::size_t) noexcept {
  return nullptr;
}

void fail_deallocate(void *, void *, std::size_t, std::size_t) noexcept {}

void *fail_reallocate(void *, void *, std::size_t, std::size_t,
                      std::size_t) noexcept {
  return nullptr;
}

AllocatorRef failing_allocator() noexcept {
  return AllocatorRef(&fail_allocate, &fail_deallocate, &fail_reallocate,
                      nullptr);
}

struct ByteTrackingAllocator {
  std::atomic<std::size_t> live{0};
  std::atomic<std::size_t> peak{0};
};

void record_live(ByteTrackingAllocator &state, std::size_t value) noexcept {
  state.live.store(value, std::memory_order_relaxed);
  std::size_t peak = state.peak.load(std::memory_order_relaxed);
  while (peak < value && !state.peak.compare_exchange_weak(
                             peak, value, std::memory_order_relaxed,
                             std::memory_order_relaxed)) {
  }
}

void *tracking_allocate(void *raw, std::size_t size,
                        std::size_t alignment) noexcept {
  void *result = system_allocator().allocate(size, alignment);
  if (result != nullptr) {
    auto &state = *static_cast<ByteTrackingAllocator *>(raw);
    record_live(state, state.live.load(std::memory_order_relaxed) + size);
  }
  return result;
}

void tracking_deallocate(void *raw, void *pointer, std::size_t size,
                         std::size_t alignment) noexcept {
  if (pointer != nullptr) {
    auto &state = *static_cast<ByteTrackingAllocator *>(raw);
    record_live(state, state.live.load(std::memory_order_relaxed) - size);
  }
  system_allocator().deallocate(pointer, size, alignment);
}

void *tracking_reallocate(void *raw, void *pointer, std::size_t old_size,
                          std::size_t new_size,
                          std::size_t alignment) noexcept {
  void *result =
      system_allocator().reallocate(pointer, old_size, new_size, alignment);
  if (result != nullptr || new_size == 0) {
    auto &state = *static_cast<ByteTrackingAllocator *>(raw);
    const std::size_t live = state.live.load(std::memory_order_relaxed);
    record_live(state, live - old_size + new_size);
  }
  return result;
}

AllocatorRef tracking_allocator(ByteTrackingAllocator &state) noexcept {
  return AllocatorRef(&tracking_allocate, &tracking_deallocate,
                      &tracking_reallocate, &state);
}

struct JourneyResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<int> served{0};
  std::atomic<int> statuses[4]{};
  char bodies[4][128]{};
  char posted[128]{};
  std::atomic<bool> trailer_seen{false};
};

struct TokenSource {
  std::size_t index = 0;

  Task<Result<HttpBodyChunk>> next(AllocatorRef allocator,
                                   OperationOptions options) noexcept {
    return next_impl(allocator, this, options);
  }

private:
  static Task<Result<HttpBodyChunk>> next_impl(AllocatorRef, TokenSource *self,
                                               OperationOptions) noexcept {
    static constexpr char kPart0[] = "data: tok-A\n\n";
    static constexpr char kPart1[] = "data: tok-B\n\n";
    static constexpr char kPart2[] = "data: [DONE]\n\n";
    static constexpr HttpHeader kTrailers[] = {
        {StringView("X-Hyle-Final"), StringView("yes")}};
    static constexpr StringView kParts[] = {
        StringView(kPart0, sizeof(kPart0) - 1, true),
        StringView(kPart1, sizeof(kPart1) - 1, true),
        StringView(kPart2, sizeof(kPart2) - 1, true)};
    if (self->index >= std::size(kParts)) {
      co_return hyle::make_unexpected(
          Status::FailedPrecondition("token source read after completion"));
    }
    const StringView part = kParts[self->index++];
    const bool final = self->index == std::size(kParts);
    co_return HttpBodyChunk{bytes(part),
                            final ? std::span<const HttpHeader>(kTrailers)
                                  : std::span<const HttpHeader>{},
                            final};
  }
};

Task<int> journey_server(AllocatorRef allocator, TcpListener *listener,
                         JourneyResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted) {
    result->server_done.store(true, std::memory_order_release);
    co_return 0;
  }
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made) {
    result->server_done.store(true, std::memory_order_release);
    co_return 0;
  }
  HttpServerConnection connection = std::move(*made);

  for (int index = 0; index < 4; ++index) {
    auto incoming_result = co_await connection.read_request();
    if (!incoming_result || !incoming_result->has_value())
      break;
    auto incoming = std::move(incoming_result->value());
    auto body_result = co_await incoming.body.read_all(1024);
    if (!body_result)
      break;

    const int served = result->served.fetch_add(1, std::memory_order_relaxed);
    if (incoming.head.method() == "POST") {
      copy_text(result->posted, sizeof(result->posted), *body_result);
    }

    static constexpr HttpHeader kTextHeader[] = {
        {StringView("Content-Type"), StringView("text/plain")}};
    static constexpr HttpHeader kSseHeader[] = {
        {StringView("Content-Type"), StringView("text/event-stream")},
        {StringView("Trailer"), StringView("X-Hyle-Final")}};
    HttpResponseHead response;
    response.headers = incoming.head.path() == "/stream"
                           ? std::span<const HttpHeader>(kSseHeader)
                           : std::span<const HttpHeader>(kTextHeader);
    Status sent;
    if (incoming.head.path() == "/stream") {
      TokenSource source;
      sent = co_await connection.send_response(response,
                                               HttpBodySource::from(source));
    } else if (incoming.head.method() == "POST") {
      sent = co_await connection.send_response(response, bytes("posted-ok"));
    } else {
      sent = co_await connection.send_response(response, bytes("hello-hyle"));
    }
    if (!sent.ok())
      break;
    (void)served;
  }
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> journey_client(AllocatorRef allocator, IoContext *io,
                         JourneyResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened) {
    result->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kJourneyPort)))
           .ok()) {
    result->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made) {
    result->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  HttpClientConnection connection = std::move(*made);

  const StringView methods[] = {"GET", "POST", "GET", "GET"};
  const StringView targets[] = {"/", "/echo", "/stream", "/stream"};
  for (int index = 0; index < 4; ++index) {
    HttpRequestHead request;
    request.method = methods[index];
    request.target = targets[index];
    request.authority = "127.0.0.1";
    const StringView request_body =
        index == 1 ? StringView("the-post-body") : StringView{};
    auto response_result =
        co_await connection.request(request, bytes(request_body));
    if (!response_result)
      break;
    HttpIncomingResponse response = std::move(*response_result);
    result->statuses[index].store(response.head.status(),
                                  std::memory_order_release);

    if (index < 2) {
      auto body_result = co_await response.body.read_all(4096);
      if (!body_result)
        break;
      copy_text(result->bodies[index], sizeof(result->bodies[index]),
                *body_result);
      continue;
    }

    std::string collected;
    for (;;) {
      auto chunk_result = co_await response.body.read(5);
      if (!chunk_result) {
        collected.clear();
        break;
      }
      const HttpBodyChunkView &chunk = *chunk_result;
      collected.append(reinterpret_cast<const char *>(chunk.bytes.data()),
                       chunk.bytes.size());
      if (chunk.end_stream) {
        const auto final = chunk.trailers.find("x-hyle-final");
        result->trailer_seen.store(final.has_value() && *final == "yes",
                                   std::memory_order_release);
        break;
      }
    }
    copy_text(result->bodies[index], sizeof(result->bodies[index]),
              StringView(collected));
  }

  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttp, FixedStreamingTrailersAndKeepAlive) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kJourneyPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<JourneyResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = journey_server(system_allocator(), &listener, result.get());
  client->task = journey_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result->served.load(std::memory_order_acquire), 4);
  for (const auto &status : result->statuses)
    EXPECT_EQ(status.load(), 200);
  EXPECT_STREQ(result->bodies[0], "hello-hyle");
  EXPECT_STREQ(result->posted, "the-post-body");
  EXPECT_STREQ(result->bodies[1], "posted-ok");
  EXPECT_STREQ(result->bodies[2],
               "data: tok-A\n\ndata: tok-B\n\ndata: [DONE]\n\n");
  EXPECT_STREQ(result->bodies[3], result->bodies[2]);
  EXPECT_TRUE(result->trailer_seen.load(std::memory_order_acquire));
}

struct ContinueResult {
  std::atomic<bool> server_saw_expect{false};
  std::atomic<bool> client_writer_disarmed{false};
  std::atomic<int> status{0};
  char request_body[64]{};
};

Task<int> continue_server(AllocatorRef allocator, TcpListener *listener,
                          ContinueResult *result, bool reject,
                          bool send_continue) {
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
  auto incoming = std::move(request->value());
  result->server_saw_expect.store(incoming.head.expects_continue(),
                                  std::memory_order_release);

  HttpResponseHead response;
  if (reject) {
    incoming.body = {};
    response.status = 413;
    response.keep_alive = false;
    (void)co_await connection.send_response(response, bytes("too-large"));
    co_return 0;
  }

  if (send_continue && !(co_await connection.send_continue()).ok()) {
    co_return 0;
  }
  auto body = co_await incoming.body.read_all(sizeof(result->request_body) - 1);
  if (!body)
    co_return 0;
  copy_text(result->request_body, sizeof(result->request_body), *body);
  (void)co_await connection.send_response(response, bytes("accepted"));
  co_return 0;
}

Task<int> automatic_continue_client(AllocatorRef allocator, IoContext *io,
                                    ContinueResult *result, std::uint16_t port,
                                    HttpLimits limits) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  auto made =
      HttpClientConnection::create(allocator, std::move(stream), limits);
  if (!made)
    co_return 0;
  HttpClientConnection connection = std::move(*made);
  HttpRequestHead request;
  request.method = "POST";
  request.target = "/upload";
  request.authority = "127.0.0.1";
  request.expect_continue = true;
  auto response = co_await connection.request(request, bytes("model-bytes"));
  if (!response)
    co_return 0;
  result->status.store(response->head.status(), std::memory_order_release);
  (void)co_await response->body.discard();
  co_return 0;
}

Task<int> early_response_client(AllocatorRef allocator, IoContext *io,
                                ContinueResult *result, std::uint16_t port) {
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
  request.method = "POST";
  request.target = "/upload";
  request.authority = "127.0.0.1";
  request.expect_continue = true;
  auto started = co_await connection.start_request(request, 4096);
  if (!started)
    co_return 0;
  auto writer = std::move(*started);
  auto decision = co_await connection.wait_for_continue(writer);
  if (!decision || !decision->has_value())
    co_return 0;
  result->client_writer_disarmed.store(!writer.valid(),
                                       std::memory_order_release);
  result->status.store(decision->value().head.status(),
                       std::memory_order_release);
  (void)co_await decision->value().body.discard();
  co_return 0;
}

void run_continue_case(std::uint16_t port, bool reject, ContinueResult &result,
                       bool send_continue = true, HttpLimits limits = {}) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = continue_server(system_allocator(), &listener, &result, reject,
                                 send_continue);
  client->task =
      reject ? early_response_client(system_allocator(), &io, &result, port)
             : automatic_continue_client(system_allocator(), &io, &result, port,
                                         limits);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
}

TEST(IoHttp, ExpectContinueWaitsBeforeSendingBody) {
  ContinueResult result;
  run_continue_case(54420, false, result);
  EXPECT_TRUE(result.server_saw_expect.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire), 200);
  EXPECT_STREQ(result.request_body, "model-bytes");
}

TEST(IoHttp, ExpectContinueReturnsEarlyFinalWithoutSendingBody) {
  ContinueResult result;
  run_continue_case(54421, true, result);
  EXPECT_TRUE(result.server_saw_expect.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_writer_disarmed.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire), 413);
  EXPECT_STREQ(result.request_body, "");
}

TEST(IoHttp, ExpectContinueFallsBackWhenServerIgnoresIt) {
  ContinueResult result;
  HttpLimits limits;
  limits.expect_continue_wait = std::chrono::milliseconds(5);
  run_continue_case(54444, false, result, false, limits);
  EXPECT_TRUE(result.server_saw_expect.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire), 200);
  EXPECT_STREQ(result.request_body, "model-bytes");
}

struct SmuggleResult {
  std::atomic<bool> accepted{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
};

Task<int> smuggle_server(AllocatorRef allocator, TcpListener *listener,
                         SmuggleResult *result, HttpLimits limits = {}) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made =
      HttpServerConnection::create(allocator, std::move(*accepted), limits);
  if (!made) {
    result->status.store(made.error().code(), std::memory_order_release);
    co_return 0;
  }
  HttpServerConnection connection = std::move(*made);
  auto request = co_await connection.read_request();
  result->status.store(request ? StatusCode::Ok : request.error().code(),
                       std::memory_order_release);
  result->accepted.store(request.has_value() && request->has_value(),
                         std::memory_order_release);
  co_return 0;
}

Task<int> raw_request_client(AllocatorRef, IoContext *io, const char *raw,
                             std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  (void)(co_await stream.write_some(
      std::as_bytes(std::span(raw, std::strlen(raw)))));
  std::byte response[1];
  (void)(co_await stream.read_some(response));
  co_return 0;
}

Task<int> fragmented_request_client(AllocatorRef allocator, IoContext *io,
                                    const char *raw, std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  const std::size_t size = std::strlen(raw);
  constexpr std::size_t kFragment = 7;
  for (std::size_t offset = 0; offset < size; offset += kFragment) {
    const std::size_t count = std::min(kFragment, size - offset);
    auto written = co_await hyle::io::write_all(
        allocator, stream, std::as_bytes(std::span(raw + offset, count)));
    if (!written.ok())
      co_return 0;
    hyle::sched::sleep_for(std::chrono::milliseconds(1));
  }
  std::byte response[1];
  (void)(co_await stream.read_some(response));
  co_return 0;
}

StatusCode raw_request_status(const char *raw, std::uint16_t port,
                              HttpLimits limits = {}) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  if (!io.valid())
    return StatusCode::Internal;
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  if (!listener_result)
    return listener_result.error().code();
  TcpListener listener = std::move(*listener_result);
  SmuggleResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = smuggle_server(system_allocator(), &listener, &result, limits);
  client->task = raw_request_client(system_allocator(), &io, raw, port);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  (void)scheduler.spawn_on(kOwner, &server_fiber);
  (void)scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();
  return result.status.load(std::memory_order_acquire);
}

bool raw_request_accepted(const char *raw, std::uint16_t port) {
  return raw_request_status(raw, port) == StatusCode::Ok;
}

TEST(IoHttp, RejectsAmbiguousRequestFraming) {
  EXPECT_TRUE(raw_request_accepted("GET / HTTP/1.1\r\nHost: x\r\n\r\n", 54430));
  EXPECT_FALSE(
      raw_request_accepted("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
                           "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
                           54431));
  EXPECT_FALSE(
      raw_request_accepted("POST / HTTP/1.1\r\nHost: x\r\n"
                           "Transfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n",
                           54432));
  EXPECT_FALSE(
      raw_request_accepted("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
                           "Content-Length: 6\r\n\r\nhello",
                           54433));
  EXPECT_FALSE(raw_request_accepted(
      "GET / HTTP/1.1\r\nHost: x\r\nHost: y\r\n\r\n", 54434));
}

TEST(IoHttp, ParsesRequestHeadAcrossTransportFragments) {
  constexpr std::uint16_t kPort = 54442;
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  SmuggleResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = smuggle_server(system_allocator(), &listener, &result);
  client->task = fragmented_request_client(
      system_allocator(), &io,
      "GET /fragmented?q=1 HTTP/1.1\r\nHost: inference.local\r\n"
      "X-Request: token\r\n\r\n",
      kPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_EQ(result.status.load(std::memory_order_acquire), StatusCode::Ok);
}

TEST(IoHttp, EnforcesConfiguredRequestHeadAndBodyLimits) {
  HttpLimits small_head;
  small_head.max_head_bytes = 32;
  EXPECT_EQ(
      raw_request_status("GET / HTTP/1.1\r\nHost: inference.local\r\n\r\n",
                         54422, small_head),
      StatusCode::ResourceExhausted);

  HttpLimits one_header;
  one_header.max_header_count = 1;
  EXPECT_EQ(
      raw_request_status("GET / HTTP/1.1\r\nHost: x\r\nX-Extra: y\r\n\r\n",
                         54423, one_header),
      StatusCode::ResourceExhausted);

  HttpLimits tiny_body;
  tiny_body.max_body_bytes = 4;
  EXPECT_EQ(raw_request_status(
                "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\n"
                "model",
                54424, tiny_body),
            StatusCode::ResourceExhausted);
}

Task<int> body_probe_server(AllocatorRef allocator, TcpListener *listener,
                            SmuggleResult *result, HttpLimits limits) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made =
      HttpServerConnection::create(allocator, std::move(*accepted), limits);
  if (!made) {
    result->status.store(made.error().code(), std::memory_order_release);
    co_return 0;
  }
  HttpServerConnection connection = std::move(*made);
  auto request = co_await connection.read_request();
  if (!request || !request->has_value()) {
    result->status.store(request ? StatusCode::DataLoss
                                 : request.error().code(),
                         std::memory_order_release);
    co_return 0;
  }
  auto body = co_await request->value().body.read_all(64 * 1024);
  result->status.store(body ? StatusCode::Ok : body.error().code(),
                       std::memory_order_release);
  co_return 0;
}

StatusCode raw_request_body_status(const char *raw, std::uint16_t port,
                                   HttpLimits limits) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  if (!io.valid())
    return StatusCode::Internal;
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  if (!listener_result)
    return listener_result.error().code();
  TcpListener listener = std::move(*listener_result);
  SmuggleResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      body_probe_server(system_allocator(), &listener, &result, limits);
  client->task = raw_request_client(system_allocator(), &io, raw, port);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  (void)scheduler.spawn_on(kOwner, &server_fiber);
  (void)scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();
  return result.status.load(std::memory_order_acquire);
}

TEST(IoHttp, EnforcesChunkedBodyLimitAndSyntax) {
  HttpLimits tiny_body;
  tiny_body.max_body_bytes = 4;
  EXPECT_EQ(raw_request_body_status("POST / HTTP/1.1\r\nHost: x\r\n"
                                    "Transfer-Encoding: chunked\r\n\r\n"
                                    "5\r\nmodel\r\n0\r\n\r\n",
                                    54428, tiny_body),
            StatusCode::ResourceExhausted);
  EXPECT_EQ(raw_request_body_status("POST / HTTP/1.1\r\nHost: x\r\n"
                                    "Transfer-Encoding: chunked\r\n\r\n"
                                    "Z\r\ninvalid\r\n0\r\n\r\n",
                                    54429, {}),
            StatusCode::InvalidArgument);
}

struct OomResult {
  std::atomic<bool> accepted{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
  TcpStream held;
};

Task<int> oom_server(AllocatorRef, TcpListener *listener, OomResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (accepted) {
    result->held = std::move(*accepted);
    result->accepted.store(true, std::memory_order_release);
  }
  co_return 0;
}

Task<int> oom_client(AllocatorRef, IoContext *io, OomResult *result,
                     std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  auto made =
      HttpClientConnection::create(failing_allocator(), std::move(stream));
  result->status.store(made ? StatusCode::Ok : made.error().code(),
                       std::memory_order_release);
  co_return 0;
}

TEST(IoHttp, ConnectionConstructionReportsAllocatorExhaustion) {
  constexpr std::uint16_t kPort = 54425;
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  OomResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = oom_server(system_allocator(), &listener, &result);
  client->task = oom_client(system_allocator(), &io, &result, kPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(result.accepted.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::ResourceExhausted);
}

struct ResponseProbe {
  std::atomic<bool> accepted{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
};

Task<int> raw_response_server(AllocatorRef, TcpListener *listener,
                              const char *raw,
                              std::chrono::milliseconds delay = {}) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  TcpStream stream = std::move(*accepted);
  std::byte request[1024];
  (void)(co_await stream.read_some(request));
  const char *separator = std::strstr(raw, "\r\n\r\n");
  std::size_t head_size = std::strlen(raw);
  if (delay.count() != 0 && separator != nullptr) {
    head_size = static_cast<std::size_t>(separator - raw) + 4;
  }
  (void)(co_await hyle::io::write_all(
      system_allocator(), stream, std::as_bytes(std::span(raw, head_size))));
  if (delay.count() != 0) {
    hyle::sched::sleep_for(delay);
    const std::size_t total = std::strlen(raw);
    (void)(co_await hyle::io::write_all(
        system_allocator(), stream,
        std::as_bytes(std::span(raw + head_size, total - head_size))));
  }
  co_return 0;
}

Task<int> fragmented_response_server(AllocatorRef allocator,
                                     TcpListener *listener, const char *raw) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  TcpStream stream = std::move(*accepted);
  std::byte request[1024];
  (void)(co_await stream.read_some(request));
  const std::size_t size = std::strlen(raw);
  constexpr std::size_t kFragment = 7;
  for (std::size_t offset = 0; offset < size; offset += kFragment) {
    const std::size_t count = std::min(kFragment, size - offset);
    auto written = co_await hyle::io::write_all(
        allocator, stream, std::as_bytes(std::span(raw + offset, count)));
    if (!written.ok())
      co_return 0;
    hyle::sched::sleep_for(std::chrono::milliseconds(1));
  }
  co_return 0;
}

Task<int> raw_response_client(AllocatorRef allocator, IoContext *io,
                              ResponseProbe *result, std::uint16_t port,
                              std::size_t *body_size = nullptr,
                              HttpLimits limits = {}) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  auto made =
      HttpClientConnection::create(allocator, std::move(stream), limits);
  if (!made) {
    result->status.store(made.error().code(), std::memory_order_release);
    co_return 0;
  }
  HttpClientConnection connection = std::move(*made);
  HttpRequestHead request;
  request.authority = "127.0.0.1";
  auto response =
      co_await connection.request(request, std::span<const std::byte>{});
  if (!response) {
    result->status.store(response.error().code(), std::memory_order_release);
    co_return 0;
  }
  result->accepted.store(true, std::memory_order_release);
  if (body_size != nullptr) {
    auto body = co_await response->body.read_all(64 * 1024);
    if (!body) {
      result->status.store(body.error().code(), std::memory_order_release);
      co_return 0;
    }
    *body_size = body->size();
  }
  result->status.store(StatusCode::Ok, std::memory_order_release);
  co_return 0;
}

StatusCode raw_response_status(const char *raw, std::uint16_t port,
                               HttpLimits limits = {},
                               std::size_t *body_size = nullptr) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  if (!io.valid())
    return StatusCode::Internal;
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  if (!listener_result)
    return listener_result.error().code();
  TcpListener listener = std::move(*listener_result);
  ResponseProbe result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = raw_response_server(system_allocator(), &listener, raw);
  client->task = raw_response_client(system_allocator(), &io, &result, port,
                                     body_size, limits);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  (void)scheduler.spawn_on(kOwner, &server_fiber);
  (void)scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();
  return result.status.load(std::memory_order_acquire);
}

bool raw_response_accepted(const char *raw, std::uint16_t port) {
  return raw_response_status(raw, port) == StatusCode::Ok;
}

TEST(IoHttp, ClientRejectsAmbiguousResponseFraming) {
  EXPECT_TRUE(raw_response_accepted(
      "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 54435));
  EXPECT_FALSE(
      raw_response_accepted("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n"
                            "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
                            54436));
  EXPECT_FALSE(raw_response_accepted(
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"
      "0\r\n\r\n",
      54437));
  EXPECT_FALSE(raw_response_accepted("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                                     "Content-Length: 3\r\n\r\nok!",
                                     54438));
}

TEST(IoHttp, ReportsTruncatedAndOversizedResponseBodies) {
  std::size_t body_size = 0;
  EXPECT_EQ(
      raw_response_status("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort",
                          54426, {}, &body_size),
      StatusCode::DataLoss);

  HttpLimits tiny_body;
  tiny_body.max_body_bytes = 4;
  EXPECT_EQ(
      raw_response_status("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nmodel",
                          54427, tiny_body, &body_size),
      StatusCode::ResourceExhausted);
}

TEST(IoHttp, ParsesResponseHeadAndBodyAcrossTransportFragments) {
  constexpr std::uint16_t kPort = 54443;
  static constexpr char kResponse[] =
      "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n"
      "X-Model: qwen\r\n\r\nhello-hyle";
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  ResponseProbe result;
  std::size_t body_size = 0;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      fragmented_response_server(system_allocator(), &listener, kResponse);
  client->task =
      raw_response_client(system_allocator(), &io, &result, kPort, &body_size);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_EQ(result.status.load(std::memory_order_acquire), StatusCode::Ok);
  EXPECT_EQ(body_size, 10u);
}

TEST(IoHttp, ReadsFragmentedCloseDelimitedBody) {
  constexpr std::uint16_t kPort = 54440;
  const std::string raw =
      "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + std::string(10000, 'B');
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  ResponseProbe result;
  std::size_t body_size = 0;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = raw_response_server(system_allocator(), &listener, raw.c_str(),
                                     std::chrono::milliseconds(20));
  client->task =
      raw_response_client(system_allocator(), &io, &result, kPort, &body_size);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(result.accepted.load(std::memory_order_acquire));
  EXPECT_EQ(body_size, 10000u);
}

struct RetentionResult {
  ByteTrackingAllocator allocator;
  std::atomic<std::size_t> before_response{0};
  std::atomic<std::size_t> after_response{0};
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<bool> server_closed{false};
  std::atomic<bool> client_closed{false};
  std::string request_body = std::string(256 * 1024, 'M');
};

Task<int> retention_server(AllocatorRef allocator, TcpListener *listener,
                           RetentionResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  HttpLimits limits;
  limits.max_retained_body_bytes = 4096;
  auto made =
      HttpServerConnection::create(allocator, std::move(*accepted), limits);
  if (!made)
    co_return 0;
  HttpServerConnection connection = std::move(*made);
  auto request = co_await connection.read_request();
  if (!request || !request->has_value())
    co_return 0;
  auto body = co_await request->value().body.read_all(512 * 1024);
  if (!body || body->size() != result->request_body.size())
    co_return 0;
  result->before_response.store(
      result->allocator.live.load(std::memory_order_relaxed),
      std::memory_order_release);
  if (!(co_await connection.send_response({}, bytes("ok"))).ok())
    co_return 0;
  result->after_response.store(
      result->allocator.live.load(std::memory_order_relaxed),
      std::memory_order_release);
  const Status closed = co_await connection.close();
  const Status closed_again = co_await connection.close();
  result->server_closed.store(closed.ok() && closed_again.ok() &&
                                  !connection.valid(),
                              std::memory_order_release);
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> retention_client(AllocatorRef allocator, IoContext *io,
                           RetentionResult *result, std::uint16_t port) {
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
  HttpRequestHead head;
  head.method = "POST";
  head.target = "/model-metadata";
  head.authority = "127.0.0.1";
  auto response = co_await connection.request(
      head, std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                           result->request_body.data()),
                                       result->request_body.size()));
  if (!response || !(co_await response->body.discard()).ok())
    co_return 0;
  const Status closed = co_await connection.close();
  const Status closed_again = co_await connection.close();
  result->client_closed.store(closed.ok() && closed_again.ok() &&
                                  !connection.valid(),
                              std::memory_order_release);
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoHttp, ReleasesLargeReadAllBufferAtMessageBoundary) {
  constexpr std::uint16_t kPort = 54445;
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  RetentionResult result;
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = retention_server(tracking_allocator(result.allocator),
                                  &listener, &result);
  client->task = retention_client(system_allocator(), &io, &result, kPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  ASSERT_TRUE(result.server_done.load(std::memory_order_acquire));
  ASSERT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.server_closed.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_closed.load(std::memory_order_acquire));
  const std::size_t before =
      result.before_response.load(std::memory_order_acquire);
  const std::size_t after =
      result.after_response.load(std::memory_order_acquire);
  EXPECT_GT(before, after);
  EXPECT_GE(before - after, 128 * 1024u);
}

} // namespace
