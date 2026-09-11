#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/io/http.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_WEBSOCKET_API
#elif defined(HYLE_IO_WEBSOCKET_BUILD)
#define HYLE_IO_WEBSOCKET_API HYLE_EXPORT
#else
#define HYLE_IO_WEBSOCKET_API HYLE_IMPORT
#endif

namespace hyle::io {

enum class WebSocketMessageType : std::uint8_t {
  kText,
  kBinary,
};

enum class WebSocketControlType : std::uint8_t {
  kPing,
  kPong,
};

struct WebSocketClose {
  // RFC 6455 allows a Close frame without a status code. The default is the
  // normal closure code used by close(); a received code-less Close reports
  // std::nullopt rather than inventing the reserved wire value 1005.
  std::optional<std::uint16_t> code = std::uint16_t{1000};
  container::StringView reason{};
};

struct WebSocketReadResult {
  std::size_t bytes = 0;
  WebSocketMessageType type = WebSocketMessageType::kBinary;
  bool end_message = false;
  // Present only when the peer's Close frame completed the connection. A
  // clean empty message is therefore distinct from close even though both may
  // return zero application bytes.
  std::optional<WebSocketClose> close{};
};

struct WebSocketLimits {
  std::uint64_t max_message_bytes = 16ull * 1024 * 1024;
  std::uint64_t max_frame_bytes = 16ull * 1024 * 1024;
  // Outgoing messages are fragmented at this boundary. Client frames reuse a
  // scratch buffer of at most this size because RFC masking cannot modify a
  // borrowed const payload in place.
  std::size_t write_fragment_bytes = 16u * 1024u;
};

using WebSocketControlCallback =
    void (*)(void *context, WebSocketControlType type,
             std::span<const std::byte> payload) noexcept;

struct WebSocketServerOptions {
  // Empty selects no subprotocol. A non-empty token must occur in the client's
  // Sec-WebSocket-Protocol offer; route/auth/Origin policy is decided by the
  // caller from HttpIncomingRequest::head before accept().
  container::StringView subprotocol{};
  std::span<const HttpHeader> headers{};
  WebSocketLimits limits{};
  void *control_context = nullptr;
  WebSocketControlCallback control_callback = nullptr;
};

struct WebSocketClientOptions {
  container::StringView authority{};
  container::StringView target = "/";
  container::StringView origin{};
  std::span<const container::StringView> subprotocols{};
  std::span<const HttpHeader> headers{};
  HttpLimits http_limits{};
  WebSocketLimits limits{};
  void *control_context = nullptr;
  WebSocketControlCallback control_callback = nullptr;
};

namespace detail {
struct WebSocketWriteState;
}

class WebSocketMessageWriter;

// RFC 6455 message connection over HYLE's existing HTTP/TCP/TLS/Sched stack.
// The object is pinned to its transport owner worker. One read and one write
// may progress concurrently; a second operation in either direction fails
// instead of silently interleaving message state.
class WebSocketConnection {
public:
  WebSocketConnection() noexcept = default;
  HYLE_IO_WEBSOCKET_API ~WebSocketConnection();
  HYLE_IO_WEBSOCKET_API WebSocketConnection(WebSocketConnection &&) noexcept;
  HYLE_IO_WEBSOCKET_API WebSocketConnection &
  operator=(WebSocketConnection &&) noexcept;
  WebSocketConnection(const WebSocketConnection &) = delete;
  WebSocketConnection &operator=(const WebSocketConnection &) = delete;

  // The task borrows `http`, `request`, and option views until completion;
  // await it while those objects are alive. A validation/allocation failure
  // before the 101 write leaves the HTTP owner and request usable.
  HYLE_IO_WEBSOCKET_API static sched::Task<Result<WebSocketConnection>>
  accept(mem::AllocatorRef allocator, HttpServerConnection &http,
         HttpIncomingRequest &&request, WebSocketServerOptions options = {},
         OperationOptions operation = {}) noexcept;

  // `transport` must already be connected (and TLS-handshaken for wss). DNS,
  // proxy, reconnect and authentication policy remain with the application.
  HYLE_IO_WEBSOCKET_API static sched::Task<Result<WebSocketConnection>>
  connect(mem::AllocatorRef allocator, HttpTransport transport,
          WebSocketClientOptions options,
          OperationOptions operation = {}) noexcept;

  HYLE_IO_WEBSOCKET_API bool valid() const noexcept;
  HYLE_IO_WEBSOCKET_API bool is_client() const noexcept;
  HYLE_IO_WEBSOCKET_API unsigned owner_worker() const noexcept;
  HYLE_IO_WEBSOCKET_API container::StringView
  selected_subprotocol() const noexcept;

  // Caller-buffer message streaming. Frame boundaries are deliberately not a
  // public contract; end_message marks the semantic message boundary. The
  // returned close reason is connection-owned until the next operation.
  HYLE_IO_WEBSOCKET_API sched::Task<Result<WebSocketReadResult>>
  read_some(std::span<std::byte> output,
            OperationOptions operation = {}) noexcept;

  HYLE_IO_WEBSOCKET_API Result<WebSocketMessageWriter>
  start_message(WebSocketMessageType type) noexcept;
  HYLE_IO_WEBSOCKET_API sched::Task<Status>
  send_text(container::StringView text,
            OperationOptions operation = {}) noexcept;
  HYLE_IO_WEBSOCKET_API sched::Task<Status>
  send_binary(std::span<const std::byte> bytes,
              OperationOptions operation = {}) noexcept;
  HYLE_IO_WEBSOCKET_API sched::Task<Status>
  ping(std::span<const std::byte> payload = {},
       OperationOptions operation = {}) noexcept;
  HYLE_IO_WEBSOCKET_API sched::Task<Status>
  pong(std::span<const std::byte> payload = {},
       OperationOptions operation = {}) noexcept;

  // Performs the complete close handshake and then closes the byte transport.
  // It rejects another read already in flight; a read that receives peer Close
  // performs the reciprocal reply itself and returns a typed close result.
  HYLE_IO_WEBSOCKET_API sched::Task<Status>
  close(WebSocketClose close = {}, OperationOptions operation = {}) noexcept;

private:
  friend class WebSocketMessageWriter;
  struct Impl;
  WebSocketConnection(mem::AllocatorRef allocator, Impl *impl) noexcept
      : impl_(impl), allocator_(allocator) {}

  Impl *impl_ = nullptr;
  mem::AllocatorRef allocator_{};
};

// Move-only borrow over one outgoing message. Each write completes one or more
// bounded frames before returning; the payload must remain alive for that
// await. Dropping an unfinished writer invalidates the connection because the
// peer would otherwise observe a permanently incomplete fragmented message.
class WebSocketMessageWriter {
public:
  WebSocketMessageWriter() noexcept = default;
  HYLE_IO_WEBSOCKET_API ~WebSocketMessageWriter();
  HYLE_IO_WEBSOCKET_API
  WebSocketMessageWriter(WebSocketMessageWriter &&other) noexcept;
  HYLE_IO_WEBSOCKET_API WebSocketMessageWriter &
  operator=(WebSocketMessageWriter &&other) noexcept;
  WebSocketMessageWriter(const WebSocketMessageWriter &) = delete;
  WebSocketMessageWriter &operator=(const WebSocketMessageWriter &) = delete;

  HYLE_IO_WEBSOCKET_API bool valid() const noexcept;
  HYLE_IO_WEBSOCKET_API sched::Task<Status>
  write(std::span<const std::byte> bytes, bool end_message = false,
        OperationOptions operation = {}) noexcept;
  sched::Task<Status> finish(OperationOptions operation = {}) noexcept {
    return write({}, true, operation);
  }

private:
  friend class WebSocketConnection;
  WebSocketMessageWriter(detail::WebSocketWriteState *state,
                         std::uint64_t generation,
                         mem::AllocatorRef allocator) noexcept
      : state_(state), generation_(generation), allocator_(allocator) {}
  HYLE_IO_WEBSOCKET_API void abandon_() noexcept;

  detail::WebSocketWriteState *state_ = nullptr;
  std::uint64_t generation_ = 0;
  mem::AllocatorRef allocator_{};
};

} // namespace hyle::io
