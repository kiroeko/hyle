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
#include <string_view>
#include <thread>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/websocket.h"
#if defined(HYLE_IO_TLS)
#include "../support/tls_test_material.h"
#include "hyle/io/tls.h"
#endif
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"
#include "websocket_mask_internal.h"

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;
using hyle::container::StringView;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::HttpServerConnection;
using hyle::io::IoContext;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::WebSocketClientOptions;
using hyle::io::WebSocketConnection;
using hyle::io::WebSocketControlType;
using hyle::io::WebSocketMessageType;
using hyle::io::WebSocketServerOptions;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::CancelToken;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::uint16_t kJourneyPort = 54531;
constexpr std::uint16_t kPrefetchPort = 54532;
constexpr std::uint16_t kHandshakeRejectionPort = 54536;
constexpr std::uint16_t kFrameRejectionPort = 54537;
constexpr std::uint16_t kDeadlinePort = 54539;
constexpr std::uint16_t kCancelPort = 54540;
constexpr std::uint16_t kAbandonPort = 54541;
#if defined(HYLE_IO_TLS)
constexpr std::uint16_t kTlsPort = 54538;
#endif

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

struct JourneyResult {
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
  std::atomic<bool> server_owner_ok{false};
  std::atomic<bool> client_owner_ok{false};
  std::atomic<unsigned> client_ping_callbacks{0};
  std::atomic<unsigned> server_pong_callbacks{0};
};

void control_callback(void *context, WebSocketControlType type,
                      std::span<const std::byte>) noexcept {
  auto *counter = static_cast<std::atomic<unsigned> *>(context);
  if ((counter == nullptr) || (type != WebSocketControlType::kPing &&
                               type != WebSocketControlType::kPong)) {
    return;
  }
  counter->fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t checksum(std::span<const std::byte> bytes) noexcept {
  std::uint64_t value = 0;
  for (std::byte byte : bytes)
    value = value * 131 + std::to_integer<unsigned char>(byte);
  return value;
}

TEST(IoWebSocket, MaskCorePreservesOffsetAcrossPartialReads) {
  const std::array<std::byte, 4> key{std::byte{0x37}, std::byte{0xfa},
                                     std::byte{0x21}, std::byte{0x3d}};
  std::array<std::byte, 37> payload{};
  for (std::size_t i = 0; i != payload.size(); ++i)
    payload[i] = static_cast<std::byte>(i * 19u + 7u);
  const auto original = payload;

  hyle::io::detail::websocket_apply_mask(std::span<std::byte>(payload).first(5),
                                         key);
  hyle::io::detail::websocket_apply_mask(
      std::span<std::byte>(payload).subspan(5, 24), key, 5);
  hyle::io::detail::websocket_apply_mask(
      std::span<std::byte>(payload).subspan(29), key, 29);
  for (std::size_t i = 0; i != payload.size(); ++i) {
    EXPECT_EQ(payload[i], original[i] ^ key[i & 3u]);
  }

  hyle::io::detail::websocket_apply_mask(payload, key);
  EXPECT_EQ(payload, original);
}

Task<int> journey_server(AllocatorRef allocator, TcpListener *listener,
                         JourneyResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  Result<HttpServerConnection> made =
      HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection http = std::move(*made);
  auto request_result = co_await http.read_request();
  if (!request_result || !request_result->has_value())
    co_return 0;
  auto request = std::move(request_result->value());
  if (request.head.path() != "/stream")
    co_return 0;

  WebSocketServerOptions options;
  options.subprotocol = "hyle.tokens.v1";
  options.limits.max_message_bytes = 128 * 1024;
  options.limits.max_frame_bytes = 128 * 1024;
  options.limits.write_fragment_bytes = 16 * 1024;
  options.control_context = &result->server_pong_callbacks;
  options.control_callback = &control_callback;
  Result<WebSocketConnection> upgraded = co_await WebSocketConnection::accept(
      allocator, http, std::move(request), options);
  if (!upgraded)
    co_return 0;
  WebSocketConnection socket = std::move(*upgraded);
  if (socket.selected_subprotocol() != "hyle.tokens.v1" ||
      socket.owner_worker() != kOwner || socket.is_client())
    co_return 0;
  result->server_owner_ok.store(true, std::memory_order_release);

  std::array<std::byte, 8192> input{};
  std::array<std::size_t, 3> expected_sizes{6, 126, 65536};
  std::array<WebSocketMessageType, 3> expected_types{
      WebSocketMessageType::kText, WebSocketMessageType::kBinary,
      WebSocketMessageType::kBinary};
  std::array<std::uint64_t, 3> observed_checksums{};
  for (std::size_t message = 0; message != expected_sizes.size(); ++message) {
    std::size_t total = 0;
    std::uint64_t observed = 0;
    bool finished = false;
    while (!finished) {
      Result<hyle::io::WebSocketReadResult> chunk =
          co_await socket.read_some(input);
      if (!chunk || chunk->close.has_value() ||
          chunk->type != expected_types[message]) {
        co_return 0;
      }
      total += chunk->bytes;
      for (std::size_t i = 0; i != chunk->bytes; ++i)
        observed = observed * 131 + std::to_integer<unsigned char>(input[i]);
      finished = chunk->end_message;
    }
    if (total != expected_sizes[message])
      co_return 0;
    observed_checksums[message] = observed;
  }

  static constexpr std::array<std::byte, 126> kSmall = [] {
    std::array<std::byte, 126> value{};
    for (std::size_t i = 0; i != value.size(); ++i)
      value[i] = static_cast<std::byte>(i);
    return value;
  }();
  std::vector<std::byte> large(65536);
  for (std::size_t i = 0; i != large.size(); ++i)
    large[i] = static_cast<std::byte>(i * 17u);
  const std::array<std::byte, 6> text{static_cast<std::byte>('h'),
                                      std::byte{0xc3},
                                      std::byte{0xa9},
                                      static_cast<std::byte>('l'),
                                      static_cast<std::byte>('l'),
                                      static_cast<std::byte>('o')};
  if (observed_checksums[0] != checksum(text) ||
      observed_checksums[1] != checksum(kSmall) ||
      observed_checksums[2] != checksum(large)) {
    co_return 0;
  }

  static constexpr std::array<std::byte, 1> kPing{static_cast<std::byte>('p')};
  if (!(co_await socket.ping(kPing)).ok() ||
      !(co_await socket.send_text("ok")).ok() ||
      !(co_await socket.close()).ok()) {
    co_return 0;
  }
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> journey_client(AllocatorRef allocator, IoContext *io,
                         JourneyResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kJourneyPort)))
           .ok()) {
    co_return 0;
  }

  static constexpr std::array<StringView, 2> kProtocols{
      StringView("hyle.tokens.v1"), StringView("fallback")};
  WebSocketClientOptions options;
  options.authority = "127.0.0.1";
  options.target = "/stream";
  options.origin = "https://example.test";
  options.subprotocols = kProtocols;
  options.limits.max_message_bytes = 128 * 1024;
  options.limits.max_frame_bytes = 128 * 1024;
  options.limits.write_fragment_bytes = 128 * 1024;
  options.control_context = &result->client_ping_callbacks;
  options.control_callback = &control_callback;
  Result<WebSocketConnection> connected = co_await WebSocketConnection::connect(
      allocator, hyle::io::HttpTransport(std::move(stream)), options);
  if (!connected)
    co_return 0;
  WebSocketConnection socket = std::move(*connected);
  if (!socket.is_client() ||
      socket.selected_subprotocol() != "hyle.tokens.v1" ||
      socket.owner_worker() != kOwner) {
    co_return 0;
  }
  result->client_owner_ok.store(true, std::memory_order_release);

  static constexpr std::array<std::byte, 1> kUnsolicitedPong{
      static_cast<std::byte>('u')};
  if (!(co_await socket.pong(kUnsolicitedPong)).ok())
    co_return 0;

  // Validation before a first frame is recoverable: convenience helpers must
  // not poison a healthy connection merely because caller input is too large.
  std::vector<std::byte> oversized(128 * 1024 + 1);
  Status oversized_status = co_await socket.send_binary(oversized);
  if (oversized_status.code() != StatusCode::OutOfRange || !socket.valid())
    co_return 0;
  Result<hyle::io::WebSocketMessageWriter> invalid_text =
      socket.start_message(WebSocketMessageType::kText);
  if (!invalid_text)
    co_return 0;
  std::array<std::byte, 1> invalid_utf8{std::byte{0xc0}};
  Status invalid_status = co_await invalid_text->write(invalid_utf8, true);
  if (invalid_status.code() != StatusCode::InvalidArgument ||
      invalid_text->valid() || !socket.valid()) {
    co_return 0;
  }

  const std::array<std::byte, 6> text{static_cast<std::byte>('h'),
                                      std::byte{0xc3},
                                      std::byte{0xa9},
                                      static_cast<std::byte>('l'),
                                      static_cast<std::byte>('l'),
                                      static_cast<std::byte>('o')};
  Result<hyle::io::WebSocketMessageWriter> writer_result =
      socket.start_message(WebSocketMessageType::kText);
  if (!writer_result)
    co_return 0;
  auto writer = std::move(*writer_result);
  *writer_result = std::move(writer);
  if (writer.valid() || !writer_result->valid())
    co_return 0;
  if (socket.start_message(WebSocketMessageType::kBinary))
    co_return 0;
  Status competing_ping = co_await socket.ping();
  if (competing_ping.code() != StatusCode::FailedPrecondition)
    co_return 0;
  if (!(co_await writer_result->write(
            std::span<const std::byte>(text).first(2), false))
           .ok() ||
      !(co_await writer_result->write(
            std::span<const std::byte>(text).subspan(2), true))
           .ok()) {
    co_return 0;
  }

  std::array<std::byte, 126> small{};
  for (std::size_t i = 0; i != small.size(); ++i)
    small[i] = static_cast<std::byte>(i);
  if (!(co_await socket.send_binary(small)).ok())
    co_return 0;
  std::vector<std::byte> large(65536);
  for (std::size_t i = 0; i != large.size(); ++i)
    large[i] = static_cast<std::byte>(i * 17u);
  if (!(co_await socket.send_binary(large)).ok())
    co_return 0;

  std::array<std::byte, 8> reply{};
  Result<hyle::io::WebSocketReadResult> read = co_await socket.read_some(reply);
  if (!read || read->type != WebSocketMessageType::kText ||
      !read->end_message || read->bytes != 2 ||
      std::memcmp(reply.data(), "ok", 2) != 0) {
    co_return 0;
  }
  if (!(co_await socket.close()).ok())
    co_return 0;
  result->client_ok.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoWebSocket, StreamingMaskLengthsControlAndClose) {
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

  EXPECT_TRUE(result->server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->server_owner_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_owner_ok.load(std::memory_order_acquire));
  EXPECT_EQ(result->client_ping_callbacks.load(std::memory_order_acquire), 1u);
  EXPECT_EQ(result->server_pong_callbacks.load(std::memory_order_acquire), 2u);
}

struct PrefetchResult {
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
  std::atomic<unsigned> ping_callbacks{0};
};

Task<int> prefetch_server(AllocatorRef allocator, TcpListener *listener,
                          PrefetchResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection http = std::move(*made);
  auto request = co_await http.read_request();
  if (!request || !request->has_value())
    co_return 0;
  auto incoming = std::move(request->value());
  WebSocketServerOptions options;
  options.control_context = &result->ping_callbacks;
  options.control_callback = &control_callback;
  Result<WebSocketConnection> accepted_ws =
      co_await WebSocketConnection::accept(allocator, http, std::move(incoming),
                                           options);
  if (!accepted_ws)
    co_return 0;
  WebSocketConnection socket = std::move(*accepted_ws);
  std::array<std::byte, 8> output{};
  std::size_t used = 0;
  bool finished = false;
  while (!finished && used != output.size()) {
    Result<hyle::io::WebSocketReadResult> message =
        co_await socket.read_some(std::span(output).subspan(used));
    if (!message || message->close.has_value() ||
        message->type != WebSocketMessageType::kText) {
      co_return 0;
    }
    used += message->bytes;
    finished = message->end_message;
  }
  if (finished && used == 2 && std::memcmp(output.data(), "hi", 2) == 0 &&
      result->ping_callbacks.load(std::memory_order_acquire) == 1) {
    result->server_ok.store(true, std::memory_order_release);
  }
  co_return 0;
}

Task<int> prefetch_client(AllocatorRef allocator, IoContext *io,
                          PrefetchResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kPrefetchPort)))
           .ok()) {
    co_return 0;
  }

  static constexpr char kRequest[] =
      "GET /prefetch HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "Upgrade: websocket\r\n"
      "Connection: keep-alive, Upgrade\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
  // Fragmented text with an interleaved Ping, all in the same transport write
  // as the HTTP head. This exercises both HTTP prefix handoff and RFC control
  // interleaving without relying on packet scheduling luck.
  std::array<std::byte, sizeof(kRequest) - 1 + 21> packet{};
  std::memcpy(packet.data(), kRequest, sizeof(kRequest) - 1);
  std::byte *frame = packet.data() + sizeof(kRequest) - 1;
  frame[0] = std::byte{0x01};
  frame[1] = std::byte{0x81};
  frame[2] = std::byte{1};
  frame[3] = std::byte{2};
  frame[4] = std::byte{3};
  frame[5] = std::byte{4};
  frame[6] = static_cast<std::byte>(static_cast<unsigned char>('h') ^ 1u);
  frame[7] = std::byte{0x89};
  frame[8] = std::byte{0x81};
  frame[9] = std::byte{5};
  frame[10] = std::byte{6};
  frame[11] = std::byte{7};
  frame[12] = std::byte{8};
  frame[13] = static_cast<std::byte>(static_cast<unsigned char>('p') ^ 5u);
  frame[14] = std::byte{0x80};
  frame[15] = std::byte{0x81};
  frame[16] = std::byte{9};
  frame[17] = std::byte{10};
  frame[18] = std::byte{11};
  frame[19] = std::byte{12};
  frame[20] = static_cast<std::byte>(static_cast<unsigned char>('i') ^ 9u);
  hyle::io::TransferResult sent =
      co_await hyle::io::write_all(allocator, stream, packet);
  if (!sent.ok())
    co_return 0;

  std::array<char, 512> response{};
  std::size_t used = 0;
  bool accepted_response = false;
  bool received_pong = false;
  while (used != response.size() && !received_pong) {
    hyle::io::TransferResult read = co_await stream.read_some(
        std::as_writable_bytes(std::span(response).subspan(used)));
    if (!read.ok() || read.eof || read.bytes == 0)
      break;
    used += read.bytes;
    const std::string_view received(response.data(), used);
    const std::size_t end = received.find("\r\n\r\n");
    if (end != std::string_view::npos) {
      accepted_response =
          received.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") !=
          std::string_view::npos;
      for (std::size_t i = end + 4; i + 2 < used; ++i) {
        if (static_cast<unsigned char>(response[i]) == 0x8a &&
            static_cast<unsigned char>(response[i + 1]) == 0x01 &&
            response[i + 2] == 'p') {
          received_pong = true;
          break;
        }
      }
    }
  }
  result->client_ok.store(accepted_response && received_pong,
                          std::memory_order_release);
  co_return 0;
}

TEST(IoWebSocket, RfcHandshakePreservesPrefetchedFirstFrame) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPrefetchPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  auto result = std::make_unique<PrefetchResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = prefetch_server(system_allocator(), &listener, result.get());
  client->task = prefetch_client(system_allocator(), &io, result.get());
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_ok.load(std::memory_order_acquire));
}

constexpr std::array<StringView, 7> kRejectedHandshakes{
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 12\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: invalid\r\n\r\n",
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    "Sec-WebSocket-Extensions: permessage-deflate\r\n\r\n",
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    "Sec-WebSocket-Protocol: offered\r\n\r\n",
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    "Sec-WebSocket-Protocol: invalid protocol\r\n\r\n",
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\nSec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
    "GET / HTTP/1.1\r\nHost: localhost\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"};

Task<int> handshake_rejection_server(AllocatorRef allocator,
                                     TcpListener *listener,
                                     std::atomic<unsigned> *rejected) {
  for (std::size_t i = 0; i != kRejectedHandshakes.size(); ++i) {
    auto accepted = co_await listener->accept(kOwner);
    if (!accepted)
      co_return 0;
    auto made = HttpServerConnection::create(allocator, std::move(*accepted));
    if (!made)
      co_return 0;
    HttpServerConnection http = std::move(*made);
    auto request_result = co_await http.read_request();
    if (!request_result || !request_result->has_value())
      co_return 0;
    auto request = std::move(request_result->value());
    WebSocketServerOptions options;
    if (i == 3)
      options.subprotocol = "not-offered";
    Result<WebSocketConnection> socket = co_await WebSocketConnection::accept(
        allocator, http, std::move(request), options);
    if (socket || !http.valid() || !request.body.valid())
      co_return 0;
    if (!(co_await http.send_response({.status = 400, .keep_alive = false},
                                      std::span<const std::byte>{}))
             .ok()) {
      co_return 0;
    }
    rejected->fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

Task<int> handshake_rejection_client(AllocatorRef allocator, IoContext *io,
                                     std::atomic<unsigned> *responses) {
  for (StringView request : kRejectedHandshakes) {
    auto opened = TcpStream::open(*io, kOwner);
    if (!opened)
      co_return 0;
    TcpStream stream = std::move(*opened);
    if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4,
                                                     kHandshakeRejectionPort)))
             .ok()) {
      co_return 0;
    }
    hyle::io::TransferResult sent = co_await hyle::io::write_all(
        allocator, stream,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(request.data()),
            request.size()));
    if (!sent.ok())
      co_return 0;
    std::array<std::byte, 512> response{};
    hyle::io::TransferResult read = co_await stream.read_some(response);
    if (!read.ok() || read.bytes < 12 ||
        std::memcmp(response.data(), "HTTP/1.1 400", 12) != 0) {
      co_return 0;
    }
    responses->fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

TEST(IoWebSocket, RejectsMalformedOpeningHandshakesWithoutConsumingHttp) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner,
      Endpoint::loopback(AddressFamily::kIpv4, kHandshakeRejectionPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  std::atomic<unsigned> rejected{0};
  std::atomic<unsigned> responses{0};
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      handshake_rejection_server(system_allocator(), &listener, &rejected);
  client->task =
      handshake_rejection_client(system_allocator(), &io, &responses);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_EQ(rejected.load(std::memory_order_acquire),
            kRejectedHandshakes.size());
  EXPECT_EQ(responses.load(std::memory_order_acquire),
            kRejectedHandshakes.size());
}

struct BadFrame {
  std::array<std::byte, 10> bytes{};
  std::size_t size = 0;
  hyle::StatusCode expected = hyle::StatusCode::DataLoss;
  std::uint64_t max_message_bytes = 1024;
};

constexpr std::array<BadFrame, 8> kBadFrames{
    BadFrame{{std::byte{0x81}, std::byte{0x02}, static_cast<std::byte>('h'),
              static_cast<std::byte>('i')},
             4},
    BadFrame{{std::byte{0x82}, std::byte{0xfe}, std::byte{0x00},
              std::byte{0x01}, std::byte{0x01}, std::byte{0x02},
              std::byte{0x03}, std::byte{0x04}, std::byte{0x78}},
             9},
    BadFrame{{std::byte{0x81}, std::byte{0x82}, std::byte{0x01},
              std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
              std::byte{0xc1}, std::byte{0xad}},
             8},
    BadFrame{{std::byte{0x82}, std::byte{0x82}, std::byte{0x01},
              std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
              std::byte{0x60}, std::byte{0x60}},
             8,
             hyle::StatusCode::ResourceExhausted,
             1},
    BadFrame{{std::byte{0x88}, std::byte{0x82}, std::byte{0x01},
              std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
              std::byte{0x02}, std::byte{0xef}},
             8},
    // A 64-bit payload length with its forbidden high bit set.
    BadFrame{{std::byte{0x82}, std::byte{0xff}, std::byte{0x80},
              std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
              std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
              std::byte{0x01}},
             10},
    // A non-canonical 64-bit representation of a one-byte payload.
    BadFrame{{std::byte{0x82}, std::byte{0xff}, std::byte{0x00},
              std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
              std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
              std::byte{0x01}},
             10},
    // A continuation frame is invalid when no fragmented message is active.
    BadFrame{{std::byte{0x80}, std::byte{0x80}, std::byte{0x01},
              std::byte{0x02}, std::byte{0x03}, std::byte{0x04}},
             6}};

Task<int> frame_rejection_server(AllocatorRef allocator, TcpListener *listener,
                                 std::atomic<unsigned> *rejected) {
  for (const BadFrame &bad : kBadFrames) {
    auto accepted = co_await listener->accept(kOwner);
    if (!accepted)
      co_return 0;
    auto made = HttpServerConnection::create(allocator, std::move(*accepted));
    if (!made)
      co_return 0;
    HttpServerConnection http = std::move(*made);
    auto request_result = co_await http.read_request();
    if (!request_result || !request_result->has_value())
      co_return 0;
    auto request = std::move(request_result->value());
    WebSocketServerOptions options;
    options.limits.max_message_bytes = bad.max_message_bytes;
    options.limits.max_frame_bytes = 1024;
    options.limits.write_fragment_bytes = 1024;
    Result<WebSocketConnection> accepted_ws =
        co_await WebSocketConnection::accept(allocator, http,
                                             std::move(request), options);
    if (!accepted_ws)
      co_return 0;
    WebSocketConnection socket = std::move(*accepted_ws);
    std::array<std::byte, 32> output{};
    Result<hyle::io::WebSocketReadResult> message =
        co_await socket.read_some(output);
    if (message || message.error().code() != bad.expected)
      co_return 0;
    rejected->fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

Task<int> frame_rejection_client(AllocatorRef allocator, IoContext *io,
                                 std::atomic<unsigned> *responses) {
  static constexpr StringView kRequest =
      "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
      "Upgrade: websocket\r\nConnection: Upgrade\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
  for (const BadFrame &bad : kBadFrames) {
    auto opened = TcpStream::open(*io, kOwner);
    if (!opened)
      co_return 0;
    TcpStream stream = std::move(*opened);
    if (!(co_await stream.connect(
              Endpoint::loopback(AddressFamily::kIpv4, kFrameRejectionPort)))
             .ok()) {
      co_return 0;
    }
    hyle::io::TransferResult sent = co_await hyle::io::write_all(
        allocator, stream,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(kRequest.data()),
            kRequest.size()));
    if (!sent.ok())
      co_return 0;
    std::array<std::byte, 512> response{};
    hyle::io::TransferResult head = co_await stream.read_some(response);
    if (!head.ok() || head.bytes < 12 ||
        std::memcmp(response.data(), "HTTP/1.1 101", 12) != 0) {
      co_return 0;
    }
    sent = co_await hyle::io::write_all(
        allocator, stream,
        std::span<const std::byte>(bad.bytes.data(), bad.size));
    if (!sent.ok())
      co_return 0;
    hyle::io::TransferResult close = co_await stream.read_some(response);
    if (!close.ok() || close.bytes < 4 || response[0] != std::byte{0x88})
      co_return 0;
    responses->fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

TEST(IoWebSocket, RejectsRoleLengthUtf8LimitAndCloseViolations) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner,
      Endpoint::loopback(AddressFamily::kIpv4, kFrameRejectionPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);

  std::atomic<unsigned> rejected{0};
  std::atomic<unsigned> responses{0};
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      frame_rejection_server(system_allocator(), &listener, &rejected);
  client->task = frame_rejection_client(system_allocator(), &io, &responses);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_EQ(rejected.load(std::memory_order_acquire), kBadFrames.size());
  EXPECT_EQ(responses.load(std::memory_order_acquire), kBadFrames.size());
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

TEST(IoWebSocket, ReportsCoroutineAllocationFailureWithoutTouchingTransport) {
  WebSocketClientOptions options;
  options.authority = "localhost";
  auto task = WebSocketConnection::connect(failing_allocator(),
                                           hyle::io::HttpTransport{}, options);
  EXPECT_FALSE(task.valid());
}

struct BudgetResult {
  std::atomic<bool> server_ready{false};
  std::atomic<bool> client_ready{false};
  std::atomic<bool> done{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
  bool abandon_writer = false;
  WebSocketConnection held_server;
};

Task<int> budget_server(AllocatorRef allocator, TcpListener *listener,
                        BudgetResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection http = std::move(*made);
  auto request = co_await http.read_request();
  if (!request || !request->has_value())
    co_return 0;
  auto incoming = std::move(request->value());
  Result<WebSocketConnection> socket = co_await WebSocketConnection::accept(
      allocator, http, std::move(incoming));
  if (!socket)
    co_return 0;
  result->held_server = std::move(*socket);
  result->server_ready.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> budget_client(AllocatorRef allocator, IoContext *io,
                        std::uint16_t port, hyle::io::OperationOptions options,
                        BudgetResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  WebSocketClientOptions client_options;
  client_options.authority = "127.0.0.1";
  Result<WebSocketConnection> connected = co_await WebSocketConnection::connect(
      allocator, hyle::io::HttpTransport(std::move(stream)), client_options);
  if (!connected)
    co_return 0;
  WebSocketConnection socket = std::move(*connected);
  result->client_ready.store(true, std::memory_order_release);
  if (result->abandon_writer) {
    {
      auto writer = socket.start_message(WebSocketMessageType::kBinary);
      if (!writer || !writer->valid())
        co_return 0;
    }
    result->status.store(socket.valid() ? StatusCode::Unknown
                                        : StatusCode::FailedPrecondition,
                         std::memory_order_release);
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  std::array<std::byte, 8> bytes{};
  Result<hyle::io::WebSocketReadResult> read =
      co_await socket.read_some(bytes, options);
  result->status.store(read ? StatusCode::Ok : read.error().code(),
                       std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

void run_budget_case(std::uint16_t port, hyle::io::OperationOptions options,
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
  server->task = budget_server(system_allocator(), &listener, &result);
  client->task = budget_client(system_allocator(), &io, port, options, &result);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));

  std::thread canceller;
  if (cancel != nullptr) {
    canceller = std::thread([&result, cancel] {
      while (!result.client_ready.load(std::memory_order_acquire))
        std::this_thread::yield();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      cancel->cancel();
    });
  }
  scheduler.drain();
  if (canceller.joinable())
    canceller.join();
  scheduler.shutdown();
}

TEST(IoWebSocket, ReadDeadlineSettlesAStalledPeer) {
  BudgetResult result;
  run_budget_case(
      kDeadlinePort,
      {.deadline = hyle::time::deadline_after(std::chrono::milliseconds(30))},
      result, nullptr);
  EXPECT_TRUE(result.server_ready.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_ready.load(std::memory_order_acquire));
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::DeadlineExceeded);
}

TEST(IoWebSocket, ReadCancellationSettlesAStalledPeer) {
  CancelToken cancel = CancelToken::root(system_allocator());
  BudgetResult result;
  run_budget_case(
      kCancelPort,
      {.deadline = hyle::time::deadline_after(std::chrono::seconds(5)),
       .cancel = &cancel},
      result, &cancel);
  EXPECT_TRUE(result.server_ready.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_ready.load(std::memory_order_acquire));
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::Cancelled);
}

TEST(IoWebSocket, DroppingUnfinishedWriterInvalidatesConnection) {
  BudgetResult result;
  result.abandon_writer = true;
  run_budget_case(kAbandonPort, {}, result, nullptr);
  EXPECT_TRUE(result.server_ready.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_ready.load(std::memory_order_acquire));
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::FailedPrecondition);
}

#if defined(HYLE_IO_TLS)

using hyle::io::tls_handshake;
using hyle::io::TlsClientOptions;
using hyle::io::TlsContext;
using hyle::io::TlsCredentialSource;
using hyle::io::TlsServerOptions;
using hyle::io::TlsStream;

Task<int> tls_websocket_server(AllocatorRef allocator, TcpListener *listener,
                               TlsContext *context, JourneyResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto wrapped = TlsStream::accept(std::move(*accepted), *context);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if (!(co_await tls_handshake(allocator, tls)).ok())
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(tls));
  if (!made)
    co_return 0;
  HttpServerConnection http = std::move(*made);
  auto request = co_await http.read_request();
  if (!request || !request->has_value())
    co_return 0;
  auto incoming = std::move(request->value());
  Result<WebSocketConnection> accepted_ws =
      co_await WebSocketConnection::accept(allocator, http,
                                           std::move(incoming));
  if (!accepted_ws)
    co_return 0;
  WebSocketConnection socket = std::move(*accepted_ws);
  std::array<std::byte, 16> input{};
  auto message = co_await socket.read_some(input);
  if (!message || message->type != WebSocketMessageType::kText ||
      !message->end_message || message->bytes != 6 ||
      std::memcmp(input.data(), "secure", 6) != 0 ||
      !(co_await socket.send_text("secure")).ok() ||
      !(co_await socket.close()).ok()) {
    co_return 0;
  }
  result->server_ok.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> tls_websocket_client(AllocatorRef allocator, IoContext *io,
                               TlsContext *context, JourneyResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(
            Endpoint::loopback(AddressFamily::kIpv4, kTlsPort)))
           .ok()) {
    co_return 0;
  }
  auto wrapped = TlsStream::connect(std::move(stream), *context);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if (!(co_await tls_handshake(allocator, tls)).ok())
    co_return 0;
  WebSocketClientOptions options;
  options.authority = "localhost";
  options.target = "/secure";
  Result<WebSocketConnection> connected = co_await WebSocketConnection::connect(
      allocator, hyle::io::HttpTransport(std::move(tls)), options);
  if (!connected)
    co_return 0;
  WebSocketConnection socket = std::move(*connected);
  if (!(co_await socket.send_text("secure")).ok())
    co_return 0;
  std::array<std::byte, 16> input{};
  auto message = co_await socket.read_some(input);
  if (!message || message->type != WebSocketMessageType::kText ||
      !message->end_message || message->bytes != 6 ||
      std::memcmp(input.data(), "secure", 6) != 0 ||
      !(co_await socket.close()).ok()) {
    co_return 0;
  }
  result->client_ok.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoWebSocket, UsesTheSameProtocolOwnerOverTls) {
  TlsServerOptions server_options;
  server_options.certificate_chain =
      TlsCredentialSource::pem(hyle::test_support::kTlsCertificatePem);
  server_options.private_key =
      TlsCredentialSource::pem(hyle::test_support::kTlsPrivateKeyPem);
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server) << made_server.error().message();
  TlsContext server_context = std::move(*made_server);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client) << made_client.error().message();
  TlsContext client_context = std::move(*made_client);

  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto listener_result = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kTlsPort));
  ASSERT_TRUE(listener_result);
  TcpListener listener = std::move(*listener_result);
  auto result = std::make_unique<JourneyResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = tls_websocket_server(system_allocator(), &listener,
                                      &server_context, result.get());
  client->task = tls_websocket_client(system_allocator(), &io, &client_context,
                                      result.get());
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result->client_ok.load(std::memory_order_acquire));
}

#endif

} // namespace
