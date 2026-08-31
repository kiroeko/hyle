#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "hyle/io/http2.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

using hyle::container::StringView;
using hyle::io::Http2ClientConn;
using hyle::io::Http2Header;
using hyle::io::Http2MessageView;
using hyle::io::Http2ServerConn;
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

constexpr unsigned kOwner = 0;
constexpr std::uint16_t kPort = 54428;

struct Result {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<int> server_requests{0};
  std::int32_t submitted[2]{};
  std::int32_t completed[2]{};
  int status[2]{};
  char request_path[2][32]{};
  char request_body[2][32]{};
  char response_body[2][32]{};
};

void copy_ref(char* out, std::size_t cap, StringView value) noexcept {
  const std::size_t n = value.size() < cap - 1 ? value.size() : cap - 1;
  if (n != 0) std::memcpy(out, value.data(), n);
  out[n] = '\0';
}

Task<int> server_task(Allocator alloc, TcpListener* listener,
                      Result* result) noexcept {
  TcpStream stream = co_await listener->accept(kOwner);
  if (!stream.valid()) co_return 0;
  Http2ServerConn conn{alloc, std::move(stream)};
  if (!conn.valid()) co_return 0;

  std::int32_t ids[2]{};
  for (int i = 0; i < 2; ++i) {
    Http2MessageView* request = co_await conn.next_request();
    if (request == nullptr) co_return 0;
    ids[i] = request->stream_id();
    copy_ref(result->request_path[i], sizeof(result->request_path[i]),
             request->path());
    copy_ref(result->request_body[i], sizeof(result->request_body[i]),
             request->body());
    if (request->method() == "POST" &&
        request->header("x-hyle-repeat") == "same-value")
      result->server_requests.fetch_add(1, std::memory_order_relaxed);
  }

  // Respond in reverse order. A serialized HTTP/1 connection cannot do this;
  // HTTP/2 stream IDs and the nghttp2 state machine preserve both responses.
  Http2Header header{{"content-type"}, {"text/plain"}};
  if (!(co_await conn.respond(ids[1], 202, &header, 1, "response-two")))
    co_return 0;
  if (!(co_await conn.respond(ids[0], 201, &header, 1, "response-one")))
    co_return 0;
  (void)(co_await conn.shutdown());
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> client_task(Allocator alloc, IoContext* io, Result* result) noexcept {
  TcpStream stream = tcp_socket(*io, kOwner);
  alignas(16) unsigned char address[128]{};
  std::uint32_t length = 0;
  if (!resolve_addr("127.0.0.1", kPort, address, &length) ||
      co_await stream.connect(address, length) != 0)
    co_return 0;

  Http2ClientConn conn{alloc, std::move(stream)};
  if (!conn.valid()) co_return 0;
  Http2Header header{{"x-hyle-repeat"}, {"same-value"}};
  result->submitted[0] =
      co_await conn.submit_request("POST", "http", "127.0.0.1", "/one",
                                   &header, 1, "request-one");
  result->submitted[1] =
      co_await conn.submit_request("POST", "http", "127.0.0.1", "/two",
                                   &header, 1, "request-two");

  for (int i = 0; i < 2; ++i) {
    Http2MessageView* response = co_await conn.next_response();
    if (response == nullptr) co_return 0;
    result->completed[i] = response->stream_id();
    result->status[i] = response->status();
    copy_ref(result->response_body[i], sizeof(result->response_body[i]),
             response->body());
  }
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive(void* ptr) { static_cast<Drive*>(ptr)->task.resume(); }

}  // namespace

TEST(IoHttp2, MultiplexedLoopbackRoundTrip) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler, 2);
  ASSERT_TRUE(io.valid());
  TcpListener listener = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(listener.valid());

  auto result = std::make_unique<Result>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = server_task(system_allocator(), &listener, result.get());
  client->task = client_task(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_done.load());
  EXPECT_TRUE(result->client_done.load());
  EXPECT_EQ(result->server_requests.load(), 2);
  EXPECT_GT(result->submitted[0], 0);
  EXPECT_GT(result->submitted[1], result->submitted[0]);
  EXPECT_STREQ(result->request_path[0], "/one");
  EXPECT_STREQ(result->request_path[1], "/two");
  EXPECT_STREQ(result->request_body[0], "request-one");
  EXPECT_STREQ(result->request_body[1], "request-two");

  EXPECT_EQ(result->completed[0], result->submitted[1]);
  EXPECT_EQ(result->status[0], 202);
  EXPECT_STREQ(result->response_body[0], "response-two");
  EXPECT_EQ(result->completed[1], result->submitted[0]);
  EXPECT_EQ(result->status[1], 201);
  EXPECT_STREQ(result->response_body[1], "response-one");
}

namespace {

struct FlowResult {
  std::string request = std::string(96 * 1024, 'q');
  std::string response = std::string(160 * 1024, 'r');
  std::atomic<std::size_t> server_size{0};
  std::atomic<std::size_t> client_size{0};
  std::atomic<bool> ok{false};
};

Task<int> flow_server(Allocator alloc, TcpListener* listener,
                      FlowResult* result) noexcept {
  TcpStream stream = co_await listener->accept(kOwner);
  Http2ServerConn conn{alloc, std::move(stream)};
  Http2MessageView* request = co_await conn.next_request();
  if (request == nullptr) co_return 0;
  result->server_size.store(request->body().size(), std::memory_order_release);
  const StringView response{result->response.data(),
                            static_cast<std::uint32_t>(result->response.size())};
  result->ok.store(
      co_await conn.respond(request->stream_id(), 200, nullptr, 0, response),
      std::memory_order_release);
  co_return 0;
}

Task<int> flow_client(Allocator alloc, IoContext* io, std::uint16_t port,
                      FlowResult* result) noexcept {
  TcpStream stream = tcp_socket(*io, kOwner);
  alignas(16) unsigned char address[128]{};
  std::uint32_t length = 0;
  if (!resolve_addr("127.0.0.1", port, address, &length) ||
      co_await stream.connect(address, length) != 0)
    co_return 0;
  Http2ClientConn conn{alloc, std::move(stream)};
  const StringView request{result->request.data(),
                           static_cast<std::uint32_t>(result->request.size())};
  if (co_await conn.submit_request("POST", "http", "127.0.0.1", "/flow",
                                   nullptr, 0, request) <= 0)
    co_return 0;
  Http2MessageView* response = co_await conn.next_response();
  if (response != nullptr && response->status() == 200)
    result->client_size.store(response->body().size(),
                              std::memory_order_release);
  co_return 0;
}

}  // namespace

TEST(IoHttp2, DrivesConnectionAndStreamFlowControl) {
  constexpr std::uint16_t port = kPort + 1;
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler, 2);
  ASSERT_TRUE(io.valid());
  TcpListener listener = tcp_listen(io, kOwner, "127.0.0.1", port);
  ASSERT_TRUE(listener.valid());

  auto result = std::make_unique<FlowResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = flow_server(system_allocator(), &listener, result.get());
  client->task = flow_client(system_allocator(), &io, port, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->ok.load());
  EXPECT_EQ(result->server_size.load(), result->request.size());
  EXPECT_EQ(result->client_size.load(), result->response.size());
}
