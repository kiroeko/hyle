#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hyle/container/string_view.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/region.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_HTTP_API
#elif defined(HYLE_IO_HTTP_BUILD)
#  define HYLE_IO_HTTP_API HYLE_EXPORT
#else
#  define HYLE_IO_HTTP_API HYLE_IMPORT
#endif

// io-10 HTTP/1.1 skeleton over the co_await TcpStream. Per docs scope "HTTP 最终
// 结论": HYLE owns the io (TCP/proactor/fiber) + writes the INTEGRATION layer (read
// loop -> feed parser -> Request/Response -> write); protocol PARSING is borrowed
// from vendored picohttpparser (a pure zero-IO state machine). ROUTING / handlers /
// business endpoints (/v1/chat) stay in the LAYER ABOVE -- this is a skeleton, not a
// framework. The value + performance is "HTTP over our own io_uring/shard-per-core,
// no second runtime, no blocking thread pool": bytes never leave the fiber/proactor.
//
// Two API levels (both provided, per design):
//   * LOW  -- HttpServerConn::next_request() / send_response() primitives: full
//             control, the caller drives the keep-alive loop.
//   * HIGH -- http_serve(listener, handler): a thin wrapper that runs the loop and
//             calls a co_await handler(req) -> response for each request (llama.cpp
//             style). Routing lives inside the handler.
//
// Streaming responses are a FIRST-CLASS citizen from day one (not a bolt-on): LLM
// serving streams tokens as they generate (chunked transfer / SSE). A response
// either carries a whole body OR a generator callback that yields chunks until done.
//
// THREADING / shard-per-core: an HttpServerConn wraps one TcpStream pinned to its
// owner worker; all its co_awaits run on that fiber/core (same contract as
// TcpStream). One connection = one fiber = one core, serialized, lock-free.

struct phr_header;  // picohttpparser (impl detail; never leaks into this header)

namespace hyle::io {

using hyle::container::StringView;

// Cap on request-line + headers (DoS guard: the incremental parse buffer is capped
// here; a client that never sends the header terminator can't grow it unbounded).
inline constexpr std::uint32_t kHttpMaxHeaderBytes = 64 * 1024;
// Default cap on a buffered request body. A streaming/large upload path can raise
// it per-connection; unbounded Content-Length is a memory-exhaustion DoS.
inline constexpr std::uint64_t kHttpMaxBodyBytes = 8ull * 1024 * 1024;
// Header slots parsed per message (picohttpparser fills a fixed array).
inline constexpr std::size_t kHttpMaxHeaders = 64;

// Forward declarations: the co_await layers are FREE coroutines (AllocatorRef-first);
// they need friend access to the message types' private view fields, so declare
// them here before those types.
class HttpServerConn;
class HttpClientConn;
class HttpRequest;
class HttpResponse;
class HttpResponseView;
class HttpTransport;
HYLE_IO_HTTP_API sched::Task<std::int64_t> http_transport_read(
    mem::AllocatorRef, HttpTransport&, void*, std::uint32_t) noexcept;
HYLE_IO_HTTP_API sched::Task<std::int64_t> http_transport_write(
    mem::AllocatorRef, HttpTransport&, const void*, std::uint32_t) noexcept;
HYLE_IO_HTTP_API sched::Task<int> http_transport_shutdown(
    mem::AllocatorRef, HttpTransport&) noexcept;
sched::Task<HttpRequest*> http_server_next(mem::AllocatorRef, HttpServerConn&) noexcept;
sched::Task<bool> http_server_send(mem::AllocatorRef, HttpServerConn&,
                                   const HttpResponse&) noexcept;
sched::Task<HttpResponseView*> http_client_request(mem::AllocatorRef, HttpClientConn&,
                                                   StringView, StringView,
                                                   StringView, StringView) noexcept;

// A connected byte transport for HTTP. Plain TCP is always available; TLS is an
// additional leg when HYLE was built with OpenSSL. A TLS stream must already
// have completed tls_handshake() before it is handed to HTTP.
class HttpTransport {
 public:
  HttpTransport() noexcept = default;
  explicit HttpTransport(TcpStream stream) noexcept
      : tcp_(std::move(stream)) {}
#if defined(HYLE_IO_TLS)
  explicit HttpTransport(TlsStream stream) noexcept
      : tls_(std::move(stream)), is_tls_(true) {}
#endif
  HttpTransport(HttpTransport&&) noexcept = default;
  HttpTransport& operator=(HttpTransport&&) noexcept = default;
  HttpTransport(const HttpTransport&) = delete;
  HttpTransport& operator=(const HttpTransport&) = delete;

  bool valid() const noexcept {
#if defined(HYLE_IO_TLS)
    return is_tls_ ? tls_.valid() : tcp_.valid();
#else
    return tcp_.valid();
#endif
  }
  bool encrypted() const noexcept {
#if defined(HYLE_IO_TLS)
    return is_tls_;
#else
    return false;
#endif
  }

 private:
  friend HYLE_IO_HTTP_API sched::Task<std::int64_t> http_transport_read(
      mem::AllocatorRef, HttpTransport&, void*, std::uint32_t) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<std::int64_t> http_transport_write(
      mem::AllocatorRef, HttpTransport&, const void*, std::uint32_t) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<int> http_transport_shutdown(
      mem::AllocatorRef, HttpTransport&) noexcept;
  TcpStream tcp_{};
#if defined(HYLE_IO_TLS)
  TlsStream tls_{};
  bool is_tls_ = false;
#endif
};

// ---- HttpRequest -----------------------------------------------------------
// A parsed request. method/path/header views point INTO the connection's read
// buffer (zero-copy) and are valid only until the next next_request() on the same
// connection (which reuses the buffer). body() likewise views buffered bytes. The
// high-level handler receives it by const& and must copy anything it keeps.
class HttpRequest {
 public:
  StringView method() const noexcept { return method_; }
  StringView target() const noexcept { return target_; }  // raw request-target (path?query), undecoded
  StringView path() const noexcept { return path_; }      // target with any ?query stripped
  StringView query() const noexcept { return query_; }    // after '?', empty if none
  int minor_version() const noexcept { return minor_version_; }  // 0 => HTTP/1.0, 1 => HTTP/1.1

  std::size_t header_count() const noexcept { return header_count_; }
  // header name/value by index (0..header_count()). Names are case-insensitive per
  // HTTP; use header() for a case-insensitive lookup.
  StringView header_name(std::size_t i) const noexcept;
  StringView header_value(std::size_t i) const noexcept;
  // Case-insensitive header lookup; returns an empty StringView if absent.
  HYLE_IO_HTTP_API StringView header(StringView name) const noexcept;

  // The (buffered) request body. Empty for GET/DELETE and when there is no body.
  StringView body() const noexcept { return body_; }
  bool keep_alive() const noexcept { return keep_alive_; }

 private:
  friend sched::Task<HttpRequest*> http_server_next(mem::AllocatorRef,
                                                    HttpServerConn&) noexcept;
  StringView method_, target_, path_, query_, body_;
  // Header views are stored as offsets into the parser's phr_header array, exposed
  // via the accessors above. We keep the raw array pointer + count (set by the conn).
  const phr_header* headers_ = nullptr;
  std::size_t header_count_ = 0;
  int minor_version_ = 1;
  bool keep_alive_ = true;
};

// ---- HttpResponse ----------------------------------------------------------
// What a handler produces. Either a whole in-memory body (set_body) OR a streaming
// generator (set_stream) for chunked transfer -- the LLM-serving path. Headers are
// caller-provided views that must outlive the send (the handler's frame does).
//
// The streaming generator is a fiber-side pull callback: each time the writer needs
// the next chunk it co_awaits nothing itself -- instead the generator is a
// `sched::Task<StreamChunk>`-producing functor. To stay allocation-light and avoid
// std::function in a hot path, the generator is expressed as a type-erased raw
// fn-ptr + context (same pattern as Fiber::EntryFn / spawn_blocking), producing the
// next chunk (data + "more?") on each call. See http.cc for the write pump.
struct StreamChunk {
  const void* data = nullptr;
  std::uint32_t len = 0;
  bool more = true;  // false => this is the final chunk (may also be empty)
};

class HttpResponse {
 public:
  // A streaming body generator: called repeatedly on the connection's fiber; each
  // call returns the next chunk. When StreamChunk::more is false the stream ends.
  // `ctx` is caller state (its lifetime must cover the send). Non-owning, no alloc.
  using NextChunkFn = StreamChunk (*)(void* ctx) noexcept;

  HttpResponse() noexcept = default;

  // Status line. reason may be empty -> a default phrase is chosen for common codes.
  void set_status(int code, StringView reason = {}) noexcept {
    status_ = code;
    reason_ = reason;
  }
  // Append a header. name/value views must outlive the send.
  HYLE_IO_HTTP_API bool add_header(StringView name, StringView value) noexcept;
  // Whole in-memory body (sets Content-Length). Mutually exclusive with set_stream.
  void set_body(StringView body) noexcept {
    body_ = body;
    streaming_ = false;
  }
  // Streaming body via chunked transfer-encoding (Transfer-Encoding: chunked). The
  // generator yields chunks until more==false. Use for SSE / token streaming.
  void set_stream(NextChunkFn next, void* ctx) noexcept {
    next_ = next;
    stream_ctx_ = ctx;
    streaming_ = true;
  }

  int status() const noexcept { return status_; }
  bool streaming() const noexcept { return streaming_; }

 private:
  friend sched::Task<bool> http_server_send(mem::AllocatorRef, HttpServerConn&,
                                            const HttpResponse&) noexcept;
  static constexpr std::size_t kMaxRespHeaders = 32;
  StringView reason_;
  StringView body_;
  StringView h_names_[kMaxRespHeaders];
  StringView h_vals_[kMaxRespHeaders];
  std::size_t h_count_ = 0;
  NextChunkFn next_ = nullptr;
  void* stream_ctx_ = nullptr;
  int status_ = 200;
  bool streaming_ = false;
};

// ---- HttpServerConn (LOW-level primitive) ----------------------------------
// Wraps one accepted TcpStream. Drives HTTP/1.1 over it: parse a request (headers +
// body), let the caller build a response, write it, then loop for keep-alive. Owns
// an internal growable read buffer (allocated via the given AllocatorRef) that persists
// across requests on the same connection (pipelined leftover bytes are preserved).
//
// Usage (low-level):
//   HttpServerConn conn{alloc, std::move(stream)};
//   while (true) {
//     HttpRequest* req = co_await conn.next_request();   // nullptr => closed/error
//     if (req == nullptr) break;
//     HttpResponse resp; ... build ...
//     if (!(co_await conn.send_response(resp))) break;
//     if (!req->keep_alive()) break;
//   }
class HttpServerConn {
 public:
  HYLE_IO_HTTP_API HttpServerConn(mem::AllocatorRef alloc, TcpStream stream) noexcept;
#if defined(HYLE_IO_TLS)
  HYLE_IO_HTTP_API HttpServerConn(mem::AllocatorRef alloc, TlsStream stream) noexcept;
#endif
  HYLE_IO_HTTP_API ~HttpServerConn();
  HttpServerConn(HttpServerConn&&) noexcept;
  HttpServerConn& operator=(HttpServerConn&&) noexcept;
  HttpServerConn(const HttpServerConn&) = delete;
  HttpServerConn& operator=(const HttpServerConn&) = delete;

  bool valid() const noexcept { return impl_ != nullptr && transport_.valid(); }
  HttpTransport& transport() noexcept { return transport_; }

  // co_await conn.next_request() reads + parses the next request. Yields
  // HttpRequest* (points into internal state, valid until the next call) or nullptr
  // on clean close / parse error / body too large. Implemented by forwarding to a
  // free coroutine (AllocatorRef-first) -- a MEMBER coroutine can't be used because the
  // promise operator new would see `this` as its first arg, not the AllocatorRef (the
  // same trap TLS/dns/file hit; see tls.cc).
  HYLE_IO_HTTP_API sched::Task<HttpRequest*> next_request() noexcept;

  // co_await conn.send_response(resp) -> true on success, false on socket error.
  // Handles Content-Length vs chunked framing and the Connection header.
  HYLE_IO_HTTP_API sched::Task<bool> send_response(const HttpResponse& resp) noexcept;

 private:
  friend sched::Task<HttpRequest*> http_server_next(mem::AllocatorRef, HttpServerConn&) noexcept;
  friend sched::Task<bool> http_server_send(mem::AllocatorRef, HttpServerConn&,
                                            const HttpResponse&) noexcept;
  struct Impl;
  Impl* impl_ = nullptr;   // read buffer + parser scratch (heap, via AllocatorRef)
  HttpTransport transport_;
  mem::AllocatorRef alloc_{};
};

// ---- high-level serve ------------------------------------------------------
// Run the keep-alive loop on `conn`, calling `handler(req) -> Task<HttpResponse>`
// for each request and writing the result. Returns when the connection closes. The
// handler is a type-erased fn producing a Task; routing lives inside it. This is the
// thin llama.cpp-style wrapper built on the primitives above.
//
// The AllocatorRef passed to each handler is a request-scoped mem::Arena view. Its
// ordinary blocks are reused by that keep-alive connection up to the configured
// retention limit, while all logical allocations are cleared after the response has
// been sent and the handler Task/coroutine frame has been destroyed. The connection
// parser and transport keep using the long-lived `alloc` passed to http_serve. A
// handler may therefore allocate response headers, body views, stream state, nested
// Task frames, and other scratch directly from its AllocatorRef without individual
// frees or cross-request retention. Anything that must survive the request must use
// caller-owned storage instead.
//
// HttpResponse must be fully described by the time its Task completes. Body/header
// views and stream ctx may point into the request allocator because it remains alive
// through the subsequent send. AllocatorRef is FIRST (the coroutine-frame allocation
// idiom -- a handler that IS a Task coroutine needs it first; see coro.h).
using HttpHandlerFn = sched::Task<HttpResponse> (*)(mem::AllocatorRef alloc, void* ctx,
                                                    const HttpRequest& req) noexcept;

struct HttpServeOptions {
  // HTTP starts much smaller than Arena's general-purpose 64 KiB default: a
  // keep-alive server may have thousands of mostly idle connections. Ordinary
  // blocks grow geometrically to 64 KiB; >=64 KiB allocations are dedicated and
  // Arena::clear() returns them immediately.
  mem::ArenaOptions request_region = []() constexpr {
    mem::ArenaOptions options;
    options.initial_block_size = 4 * 1024;
    options.max_block_size = 64 * 1024;
    options.large_allocation_threshold = 64 * 1024;
    return options;
  }();

  // After large blocks have been cleared, release all retained ordinary blocks
  // when their backing-byte high-water mark exceeds this cap. Zero means retain
  // nothing between requests. This bounds one pathological request's effect on a
  // long-lived connection without taxing the normal small-request path.
  std::size_t request_region_retain_limit = 256 * 1024;
};

HYLE_IO_HTTP_API sched::Task<void> http_serve(mem::AllocatorRef alloc, TcpStream stream,
                                         HttpHandlerFn handler, void* ctx,
                                         HttpServeOptions options = {}) noexcept;
#if defined(HYLE_IO_TLS)
HYLE_IO_HTTP_API sched::Task<void> http_serve(mem::AllocatorRef alloc, TlsStream stream,
                                         HttpHandlerFn handler,
                                         void* ctx,
                                         HttpServeOptions options = {}) noexcept;
#endif

// ---- HttpClientConn --------------------------------------------------------
// Client side over a connected TcpStream. send_request writes a request; then
// read_response parses the status line + headers + body. Mirrors the server conn's
// buffer management (phr_parse_response). Body is buffered (Content-Length/chunked).
class HttpResponseView {
 public:
  int status() const noexcept { return status_; }
  int minor_version() const noexcept { return minor_version_; }
  StringView reason() const noexcept { return reason_; }
  std::size_t header_count() const noexcept { return header_count_; }
  StringView header_name(std::size_t i) const noexcept;
  StringView header_value(std::size_t i) const noexcept;
  HYLE_IO_HTTP_API StringView header(StringView name) const noexcept;
  StringView body() const noexcept { return body_; }

 private:
  friend sched::Task<HttpResponseView*> http_client_request(
      mem::AllocatorRef, HttpClientConn&, StringView, StringView, StringView,
      StringView) noexcept;
  StringView reason_, body_;
  const phr_header* headers_ = nullptr;
  std::size_t header_count_ = 0;
  int status_ = 0;
  int minor_version_ = 1;
};

class HttpClientConn {
 public:
  HYLE_IO_HTTP_API HttpClientConn(mem::AllocatorRef alloc, TcpStream stream) noexcept;
#if defined(HYLE_IO_TLS)
  HYLE_IO_HTTP_API HttpClientConn(mem::AllocatorRef alloc, TlsStream stream) noexcept;
#endif
  HYLE_IO_HTTP_API ~HttpClientConn();
  HttpClientConn(HttpClientConn&&) noexcept;
  HttpClientConn& operator=(HttpClientConn&&) noexcept;
  HttpClientConn(const HttpClientConn&) = delete;
  HttpClientConn& operator=(const HttpClientConn&) = delete;

  bool valid() const noexcept { return impl_ != nullptr && transport_.valid(); }

  // co_await conn.request(method, path, host, body) -> HttpResponseView* (valid
  // until the next request) or nullptr on error. Writes "METHOD path HTTP/1.1\r\n
  // Host: <host>\r\n" (+ Content-Length if body) + body, then parses the response.
  // Forwards to a free coroutine (AllocatorRef-first), same reason as the server conn.
  HYLE_IO_HTTP_API sched::Task<HttpResponseView*> request(StringView method, StringView path,
                                                     StringView host,
                                                     StringView body = {}) noexcept;

 private:
  friend sched::Task<HttpResponseView*> http_client_request(
      mem::AllocatorRef, HttpClientConn&, StringView, StringView, StringView,
      StringView) noexcept;
  struct Impl;
  Impl* impl_ = nullptr;
  HttpTransport transport_;
  mem::AllocatorRef alloc_{};
};

}  // namespace hyle::io
