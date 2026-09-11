#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>

#if defined(HYLE_IO_TLS) && defined(HYLE_IO_HTTP2)
#include "../support/tls_test_material.h"
#include "hyle/container/string_view.h"
#include "hyle/io/http.h"
#include "hyle/io/http2.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/io/websocket.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/time/clock.h"
#endif

namespace {

#if defined(HYLE_IO_TLS) && defined(HYLE_IO_HTTP2)

using namespace std::chrono_literals;

constexpr unsigned kOwner = 0;

struct Drive {
  hyle::sched::Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

hyle::io::OperationOptions budget() noexcept {
  return {.deadline = hyle::time::deadline_after(5s)};
}

std::span<const std::byte> bytes(const std::string &value) noexcept {
  return {reinterpret_cast<const std::byte *>(value.data()), value.size()};
}

struct WebSocketJourneyResult {
  std::atomic<unsigned> server_stage{0};
  std::atomic<unsigned> client_stage{0};
  std::atomic<bool> client_waiting{false};
  std::atomic<bool> cancellation_observed{false};
  std::atomic<bool> clean_close_observed{false};
  std::atomic<bool> cancelled_peer_teardown_observed{false};
};

hyle::sched::Task<int>
websocket_server(hyle::mem::AllocatorRef allocator,
                 hyle::io::TcpListener *listener,
                 hyle::io::TlsContext *context,
                 WebSocketJourneyResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner, budget());
  if (!accepted.has_value())
    co_return 0;
  result->server_stage.store(1, std::memory_order_release);

  auto wrapped =
      hyle::io::TlsStream::accept(std::move(*accepted), *context);
  if (!wrapped.has_value())
    co_return 0;
  hyle::io::TlsStream tls = std::move(*wrapped);
  if (!(co_await hyle::io::tls_handshake(allocator, tls, budget())).ok() ||
      tls.negotiated_protocol() != "http/1.1") {
    co_return 0;
  }
  result->server_stage.store(2, std::memory_order_release);

  auto made = hyle::io::HttpServerConnection::create(
      allocator, std::move(tls));
  if (!made.has_value())
    co_return 0;
  hyle::io::HttpServerConnection http = std::move(*made);
  auto incoming = co_await http.read_request(budget());
  if (!incoming.has_value() || !incoming->has_value() ||
      incoming->value().head.path() != "/tokens") {
    co_return 0;
  }
  auto request = std::move(incoming->value());
  auto upgraded = co_await hyle::io::WebSocketConnection::accept(
      allocator, http, std::move(request), {}, budget());
  if (!upgraded.has_value())
    co_return 0;
  hyle::io::WebSocketConnection socket = std::move(*upgraded);
  result->server_stage.store(3, std::memory_order_release);

  std::array<std::byte, 32> input{};
  auto message = co_await socket.read_some(input, budget());
  if (!message.has_value() || message->close.has_value() ||
      message->type != hyle::io::WebSocketMessageType::kText ||
      !message->end_message || message->bytes != 6 ||
      std::memcmp(input.data(), "tokens", 6) != 0 ||
      !(co_await socket.send_text("tokens", budget())).ok()) {
    co_return 0;
  }
  result->server_stage.store(4, std::memory_order_release);

  auto close = co_await socket.read_some(input, budget());
  if (!close.has_value() || !close->close.has_value() ||
      close->close->code != std::optional<std::uint16_t>{1000}) {
    co_return 0;
  }
  result->clean_close_observed.store(true, std::memory_order_release);
  result->server_stage.store(5, std::memory_order_release);

  // Cancellation can happen after a partial frame has reached the transport.
  // WebSocket therefore makes that connection terminal instead of guessing a
  // reusable framing boundary. Prove this separately from the clean-close
  // connection above.
  auto cancel_accepted = co_await listener->accept(kOwner, budget());
  if (!cancel_accepted.has_value())
    co_return 0;
  auto cancel_wrapped =
      hyle::io::TlsStream::accept(std::move(*cancel_accepted), *context);
  if (!cancel_wrapped.has_value())
    co_return 0;
  hyle::io::TlsStream cancel_tls = std::move(*cancel_wrapped);
  if (!(co_await hyle::io::tls_handshake(allocator, cancel_tls, budget()))
           .ok() ||
      cancel_tls.negotiated_protocol() != "http/1.1") {
    co_return 0;
  }
  auto cancel_http_result = hyle::io::HttpServerConnection::create(
      allocator, std::move(cancel_tls));
  if (!cancel_http_result.has_value())
    co_return 0;
  hyle::io::HttpServerConnection cancel_http =
      std::move(*cancel_http_result);
  auto cancel_incoming = co_await cancel_http.read_request(budget());
  if (!cancel_incoming.has_value() || !cancel_incoming->has_value() ||
      cancel_incoming->value().head.path() != "/cancel") {
    co_return 0;
  }
  auto cancel_request = std::move(cancel_incoming->value());
  auto cancel_upgrade = co_await hyle::io::WebSocketConnection::accept(
      allocator, cancel_http, std::move(cancel_request), {}, budget());
  if (!cancel_upgrade.has_value())
    co_return 0;
  hyle::io::WebSocketConnection cancel_socket =
      std::move(*cancel_upgrade);
  result->server_stage.store(6, std::memory_order_release);
  auto peer_teardown = co_await cancel_socket.read_some(input, budget());
  if (peer_teardown.has_value() || cancel_socket.valid())
    co_return 0;
  result->cancelled_peer_teardown_observed.store(true,
                                                  std::memory_order_release);
  result->server_stage.store(7, std::memory_order_release);
  co_return 0;
}

hyle::sched::Task<int>
websocket_client(hyle::mem::AllocatorRef allocator, hyle::io::IoContext *io,
                 hyle::io::TlsContext *context, std::uint16_t port,
                 const hyle::sched::CancelToken *cancel,
                 WebSocketJourneyResult *result) noexcept {
  auto opened = hyle::io::TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  hyle::io::TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4,
                                         port),
            budget()))
           .ok()) {
    co_return 0;
  }

  auto wrapped =
      hyle::io::TlsStream::connect(std::move(transport), *context);
  if (!wrapped.has_value())
    co_return 0;
  hyle::io::TlsStream tls = std::move(*wrapped);
  if (!(co_await hyle::io::tls_handshake(allocator, tls, budget())).ok() ||
      tls.negotiated_protocol() != "http/1.1") {
    co_return 0;
  }
  result->client_stage.store(1, std::memory_order_release);

  hyle::io::WebSocketClientOptions options;
  options.authority = "localhost";
  options.target = "/tokens";
  auto connected = co_await hyle::io::WebSocketConnection::connect(
      allocator, hyle::io::HttpTransport(std::move(tls)), options, budget());
  if (!connected.has_value())
    co_return 0;
  hyle::io::WebSocketConnection socket = std::move(*connected);
  if (!(co_await socket.send_text("tokens", budget())).ok())
    co_return 0;
  result->client_stage.store(2, std::memory_order_release);

  std::array<std::byte, 32> reply{};
  auto echoed = co_await socket.read_some(reply, budget());
  if (!echoed.has_value() || echoed->close.has_value() ||
      echoed->type != hyle::io::WebSocketMessageType::kText ||
      !echoed->end_message || echoed->bytes != 6 ||
      std::memcmp(reply.data(), "tokens", 6) != 0) {
    co_return 0;
  }
  result->client_stage.store(3, std::memory_order_release);

  if (!(co_await socket.close({}, budget())).ok())
    co_return 0;

  auto cancel_opened = hyle::io::TcpStream::open(*io, kOwner);
  if (!cancel_opened.has_value())
    co_return 0;
  hyle::io::TcpStream cancel_transport = std::move(*cancel_opened);
  if (!(co_await cancel_transport.connect(
            hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4,
                                         port),
            budget()))
           .ok()) {
    co_return 0;
  }
  auto cancel_wrapped =
      hyle::io::TlsStream::connect(std::move(cancel_transport), *context);
  if (!cancel_wrapped.has_value())
    co_return 0;
  hyle::io::TlsStream cancel_tls = std::move(*cancel_wrapped);
  if (!(co_await hyle::io::tls_handshake(allocator, cancel_tls, budget()))
           .ok() ||
      cancel_tls.negotiated_protocol() != "http/1.1") {
    co_return 0;
  }
  hyle::io::WebSocketClientOptions cancel_options;
  cancel_options.authority = "localhost";
  cancel_options.target = "/cancel";
  auto cancel_connected =
      co_await hyle::io::WebSocketConnection::connect(
          allocator, hyle::io::HttpTransport(std::move(cancel_tls)),
          cancel_options, budget());
  if (!cancel_connected.has_value())
    co_return 0;
  hyle::io::WebSocketConnection cancel_socket =
      std::move(*cancel_connected);
  result->client_stage.store(4, std::memory_order_release);

  result->client_waiting.store(true, std::memory_order_release);
  auto cancelled = co_await cancel_socket.read_some(
      reply, {.deadline = hyle::time::deadline_after(5s), .cancel = cancel});
  if (cancelled.has_value() || !cancelled.error().IsCancelled() ||
      cancel_socket.valid()) {
    co_return 0;
  }
  result->cancellation_observed.store(true, std::memory_order_release);
  result->client_stage.store(5, std::memory_order_release);
  co_return 0;
}

struct Http2JourneyResult {
  std::string request = std::string(96 * 1024, 'q');
  std::string response = std::string(96 * 1024, 'r');
  std::atomic<unsigned> server_stage{0};
  std::atomic<unsigned> client_stage{0};
  std::atomic<std::size_t> request_bytes{0};
  std::atomic<std::size_t> response_bytes{0};
  hyle::sched::Event client_consumed;
  hyle::sched::Event server_closed;
};

struct CountingSink {
  char expected = 0;
  std::size_t bytes = 0;

  hyle::Status operator()(const hyle::io::HttpBodyChunkView &chunk) noexcept {
    for (std::byte value : chunk.bytes) {
      if (value != static_cast<std::byte>(expected))
        return hyle::Status::DataLoss("HTTP/2 body changed in transit");
    }
    bytes += chunk.bytes.size();
    return hyle::Status{};
  }
};

hyle::sched::Task<int>
http2_server(hyle::mem::AllocatorRef allocator,
             hyle::io::TcpListener *listener, hyle::io::TlsContext *context,
             Http2JourneyResult *result) noexcept {
  auto accepted = co_await listener->accept(kOwner, budget());
  if (!accepted.has_value())
    co_return 0;
  auto wrapped =
      hyle::io::TlsStream::accept(std::move(*accepted), *context);
  if (!wrapped.has_value())
    co_return 0;
  hyle::io::TlsStream tls = std::move(*wrapped);
  if (!(co_await hyle::io::tls_handshake(allocator, tls, budget())).ok() ||
      tls.negotiated_protocol() != "h2") {
    co_return 0;
  }
  result->server_stage.store(1, std::memory_order_release);

  hyle::io::Http2Limits limits;
  limits.max_body_bytes = 256 * 1024;
  limits.stream_receive_window = 4 * 1024;
  limits.connection_receive_window = 65'535;
  limits.read_chunk_bytes = 1024;
  auto made = hyle::io::Http2ServerSession::create(
      allocator, std::move(tls), limits);
  if (!made.has_value())
    co_return 0;
  hyle::io::Http2ServerSession session = std::move(*made);
  auto accepted_request = co_await session.accept(budget());
  if (!accepted_request.has_value() || !accepted_request->has_value())
    co_return 0;
  hyle::io::Http2IncomingRequest request =
      std::move(**accepted_request);
  if (request.head.method() != "POST" || request.head.path() != "/infer")
    co_return 0;

  CountingSink sink{.expected = 'q'};
  if (!(co_await request.body.consume(sink, 1024, budget())).ok() ||
      sink.bytes != result->request.size()) {
    co_return 0;
  }
  result->request_bytes.store(sink.bytes, std::memory_order_release);
  result->server_stage.store(2, std::memory_order_release);

  if (!(co_await session.send_response(
            request.stream_id, hyle::io::Http2ResponseHead{.status = 200},
            bytes(result->response), budget()))
           .ok()) {
    co_return 0;
  }
  result->server_stage.store(3, std::memory_order_release);
  if (!result->client_consumed.wait_for(5s))
    co_return 0;
  if (!(co_await session.close(budget())).ok())
    co_return 0;
  result->server_stage.store(4, std::memory_order_release);
  result->server_closed.signal();
  co_return 0;
}

hyle::sched::Task<int>
http2_client(hyle::mem::AllocatorRef allocator, hyle::io::IoContext *io,
             hyle::io::TlsContext *context, std::uint16_t port,
             Http2JourneyResult *result) noexcept {
  auto opened = hyle::io::TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  hyle::io::TcpStream transport = std::move(*opened);
  if (!(co_await transport.connect(
            hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4,
                                         port),
            budget()))
           .ok()) {
    co_return 0;
  }
  auto wrapped =
      hyle::io::TlsStream::connect(std::move(transport), *context);
  if (!wrapped.has_value())
    co_return 0;
  hyle::io::TlsStream tls = std::move(*wrapped);
  if (!(co_await hyle::io::tls_handshake(allocator, tls, budget())).ok() ||
      tls.negotiated_protocol() != "h2") {
    co_return 0;
  }
  result->client_stage.store(1, std::memory_order_release);

  hyle::io::Http2Limits limits;
  limits.max_body_bytes = 256 * 1024;
  limits.stream_receive_window = 4 * 1024;
  limits.connection_receive_window = 65'535;
  limits.read_chunk_bytes = 1024;
  auto made = hyle::io::Http2ClientSession::create(
      allocator, std::move(tls), limits);
  if (!made.has_value())
    co_return 0;
  hyle::io::Http2ClientSession session = std::move(*made);
  auto submitted = co_await session.send_request(
      {.method = "POST",
       .scheme = "https",
       .authority = "localhost",
       .path = "/infer"},
      bytes(result->request), budget());
  if (!submitted.has_value())
    co_return 0;
  result->client_stage.store(2, std::memory_order_release);

  auto response = co_await session.read_response(*submitted, budget());
  if (!response.has_value() || response->head.status() != 200)
    co_return 0;
  CountingSink sink{.expected = 'r'};
  if (!(co_await response->body.consume(sink, 1024, budget())).ok() ||
      sink.bytes != result->response.size()) {
    co_return 0;
  }
  result->response_bytes.store(sink.bytes, std::memory_order_release);
  result->client_stage.store(3, std::memory_order_release);

  result->client_consumed.signal();
  if (!result->server_closed.wait_for(5s))
    co_return 0;
  result->client_stage.store(4, std::memory_order_release);
  co_return 0;
}

#endif

TEST(CorrectnessJourney,
     ServiceConnectionComposesTlsHttp1Http2WebSocketBudgetsAndClose) {
#if !defined(HYLE_IO_TLS) || !defined(HYLE_IO_HTTP2)
  GTEST_SKIP() << "service journey requires HYLE_IO_TLS and HYLE_IO_HTTP2";
#else
  const std::array<hyle::container::StringView, 1> http1_protocol{
      "http/1.1"};
  hyle::io::TlsServerOptions ws_server_options;
  ws_server_options.certificate_chain = hyle::io::TlsCredentialSource::pem(
      hyle::test_support::kTlsCertificatePem);
  ws_server_options.private_key = hyle::io::TlsCredentialSource::pem(
      hyle::test_support::kTlsPrivateKeyPem);
  ws_server_options.alpn_protocols = http1_protocol;
  ws_server_options.require_alpn = true;
  auto ws_server_context =
      hyle::io::TlsContext::make_server(ws_server_options);
  ASSERT_TRUE(ws_server_context.has_value())
      << ws_server_context.error().message();

  hyle::io::TlsClientOptions ws_client_options;
  ws_client_options.verify = hyle::io::TlsVerifyMode::kNone;
  ws_client_options.alpn_protocols = http1_protocol;
  ws_client_options.require_alpn = true;
  auto ws_client_context =
      hyle::io::TlsContext::make_client(ws_client_options);
  ASSERT_TRUE(ws_client_context.has_value())
      << ws_client_context.error().message();

  WebSocketJourneyResult ws_result;
  auto cancel = hyle::sched::CancelToken::try_create_root(
      hyle::mem::system_allocator());
  ASSERT_TRUE(cancel.has_value()) << cancel.error().message();
  {
    hyle::sched::Scheduler scheduler(2, hyle::mem::system_allocator(), 256);
    hyle::io::IoContext io(scheduler);
    ASSERT_TRUE(io.valid());
    auto bound = hyle::io::TcpListener::bind(
        io, kOwner,
        hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4));
    ASSERT_TRUE(bound.has_value()) << bound.error().message();
    hyle::io::TcpListener listener = std::move(*bound);
    auto endpoint = listener.local_endpoint();
    ASSERT_TRUE(endpoint.has_value()) << endpoint.error().message();
    ASSERT_NE(endpoint->port(), 0);

    auto server = std::make_unique<Drive>();
    auto client = std::make_unique<Drive>();
    server->task = websocket_server(hyle::mem::system_allocator(), &listener,
                                    &*ws_server_context, &ws_result);
    client->task = websocket_client(
        hyle::mem::system_allocator(), &io, &*ws_client_context,
        endpoint->port(), &*cancel, &ws_result);
    hyle::sched::Fiber server_fiber(hyle::sched::kDefaultStackSize,
                                    &drive_entry, server.get());
    hyle::sched::Fiber client_fiber(hyle::sched::kDefaultStackSize,
                                    &drive_entry, client.get());
    ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
    ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));

    std::thread canceller([&] {
      const auto until = std::chrono::steady_clock::now() + 2s;
      while (!ws_result.client_waiting.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < until) {
        std::this_thread::yield();
      }
      cancel->cancel();
    });
    scheduler.drain();
    canceller.join();
    scheduler.shutdown();
  }
  EXPECT_EQ(ws_result.server_stage.load(std::memory_order_acquire), 7u);
  EXPECT_EQ(ws_result.client_stage.load(std::memory_order_acquire), 5u);
  EXPECT_TRUE(
      ws_result.cancellation_observed.load(std::memory_order_acquire));
  EXPECT_TRUE(ws_result.clean_close_observed.load(std::memory_order_acquire));
  EXPECT_TRUE(ws_result.cancelled_peer_teardown_observed.load(
      std::memory_order_acquire));

  hyle::io::TlsServerOptions h2_server_options;
  h2_server_options.certificate_chain = hyle::io::TlsCredentialSource::pem(
      hyle::test_support::kTlsCertificatePem);
  h2_server_options.private_key = hyle::io::TlsCredentialSource::pem(
      hyle::test_support::kTlsPrivateKeyPem);
  h2_server_options.alpn_protocols = hyle::io::tls_http2_protocols(false);
  h2_server_options.require_alpn = true;
  auto h2_server_context =
      hyle::io::TlsContext::make_server(h2_server_options);
  ASSERT_TRUE(h2_server_context.has_value())
      << h2_server_context.error().message();

  hyle::io::TlsClientOptions h2_client_options;
  h2_client_options.verify = hyle::io::TlsVerifyMode::kNone;
  h2_client_options.alpn_protocols = hyle::io::tls_http2_protocols(false);
  h2_client_options.require_alpn = true;
  auto h2_client_context =
      hyle::io::TlsContext::make_client(h2_client_options);
  ASSERT_TRUE(h2_client_context.has_value())
      << h2_client_context.error().message();

  Http2JourneyResult h2_result;
  {
    hyle::sched::Scheduler scheduler(2, hyle::mem::system_allocator(), 256);
    hyle::io::IoContext io(scheduler);
    ASSERT_TRUE(io.valid());
    auto bound = hyle::io::TcpListener::bind(
        io, kOwner,
        hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4));
    ASSERT_TRUE(bound.has_value()) << bound.error().message();
    hyle::io::TcpListener listener = std::move(*bound);
    auto endpoint = listener.local_endpoint();
    ASSERT_TRUE(endpoint.has_value()) << endpoint.error().message();
    ASSERT_NE(endpoint->port(), 0);

    auto server = std::make_unique<Drive>();
    auto client = std::make_unique<Drive>();
    server->task = http2_server(hyle::mem::system_allocator(), &listener,
                                &*h2_server_context, &h2_result);
    client->task = http2_client(
        hyle::mem::system_allocator(), &io, &*h2_client_context,
        endpoint->port(), &h2_result);
    hyle::sched::Fiber server_fiber(hyle::sched::kDefaultStackSize,
                                    &drive_entry, server.get());
    hyle::sched::Fiber client_fiber(hyle::sched::kDefaultStackSize,
                                    &drive_entry, client.get());
    ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
    ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
    scheduler.drain();
    scheduler.shutdown();
  }
  EXPECT_EQ(h2_result.server_stage.load(std::memory_order_acquire), 4u);
  EXPECT_EQ(h2_result.client_stage.load(std::memory_order_acquire), 4u);
  EXPECT_EQ(h2_result.request_bytes.load(std::memory_order_acquire),
            h2_result.request.size());
  EXPECT_EQ(h2_result.response_bytes.load(std::memory_order_acquire),
            h2_result.response.size());
#endif
}

} // namespace
