#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>

#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/io/buffer.h"
#include "hyle/io/operation.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/time/clock.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_HTTP_API
#elif defined(HYLE_IO_HTTP_BUILD)
#define HYLE_IO_HTTP_API HYLE_EXPORT
#else
#define HYLE_IO_HTTP_API HYLE_IMPORT
#endif

namespace hyle::io {

using hyle::container::StringView;

// HTTP/1 is a bounded framing layer over a connected TCP or already-handshaken
// TLS stream. It deliberately does not own listeners, routing, authentication,
// model schemas, handler pools, or batching. One connection and every borrowed
// view/reader/writer made from it stay on the transport's owner fiber.

enum class HttpVersion : std::uint8_t {
  kHttp10,
  kHttp11,
};

struct HttpLimits {
  std::size_t max_head_bytes = 64 * 1024;
  std::uint64_t max_body_bytes = 8ull * 1024 * 1024;
  std::size_t max_header_count = 128;
  // Upper bound for one transport read while consuming a body. A caller may
  // request a smaller returned chunk without changing this connection limit.
  std::size_t read_chunk_bytes = 16 * 1024;
  // Release oversized parsing/header buffers at the next safe message
  // boundary. This bounds high-water retention after an unusually large head
  // without disabling reuse for ordinary requests.
  std::size_t max_retained_head_bytes = 16 * 1024;
  // A completed read_all() may have grown a connection-owned aggregate
  // buffer. Keep small buffers for ordinary JSON reuse, but release a larger
  // high-water mark before the next message so one upload cannot pin MiBs on
  // every keep-alive connection. Zero releases every aggregate allocation.
  std::size_t max_retained_body_bytes = 64 * 1024;
  // Maximum time a client waits for an interim 100 response before sending the
  // body anyway. RFC 9110 permits this fallback and many HTTP/1 servers ignore
  // Expect. The enclosing OperationOptions deadline still wins when earlier.
  time::Duration expect_continue_wait = time::Duration(1'000'000'000);
};

struct HttpHeader {
  StringView name;
  StringView value;
};

// Non-owning header range. It preserves order and duplicates; find() returns
// the first value and uses ASCII case-insensitive field-name comparison.
class HttpHeadersView {
public:
  constexpr HttpHeadersView() noexcept = default;
  constexpr explicit HttpHeadersView(
      std::span<const HttpHeader> headers) noexcept
      : headers_(headers) {}

  constexpr std::size_t size() const noexcept { return headers_.size(); }
  constexpr bool empty() const noexcept { return headers_.empty(); }
  constexpr const HttpHeader *begin() const noexcept { return headers_.data(); }
  constexpr const HttpHeader *end() const noexcept {
    return headers_.data() == nullptr ? nullptr
                                      : headers_.data() + headers_.size();
  }
  constexpr const HttpHeader &operator[](std::size_t index) const noexcept {
    return headers_[index];
  }
  constexpr std::span<const HttpHeader> entries() const noexcept {
    return headers_;
  }

  HYLE_IO_HTTP_API std::optional<StringView>
  find(StringView name) const noexcept;
  bool contains(StringView name) const noexcept {
    return find(name).has_value();
  }

private:
  std::span<const HttpHeader> headers_{};
};

class HttpRequestView {
public:
  constexpr HttpRequestView() noexcept = default;
  constexpr HttpRequestView(StringView method, StringView target,
                            StringView path, StringView query,
                            HttpVersion version, HttpHeadersView headers,
                            bool keep_alive, bool expect_continue) noexcept
      : method_(method), target_(target), path_(path), query_(query),
        headers_(headers), version_(version), keep_alive_(keep_alive),
        expect_continue_(expect_continue) {}

  constexpr StringView method() const noexcept { return method_; }
  constexpr StringView target() const noexcept { return target_; }
  constexpr StringView path() const noexcept { return path_; }
  constexpr StringView query() const noexcept { return query_; }
  constexpr HttpVersion version() const noexcept { return version_; }
  constexpr HttpHeadersView headers() const noexcept { return headers_; }
  constexpr bool keep_alive() const noexcept { return keep_alive_; }
  constexpr bool expects_continue() const noexcept { return expect_continue_; }

private:
  StringView method_{};
  StringView target_{};
  StringView path_{};
  StringView query_{};
  HttpHeadersView headers_{};
  HttpVersion version_ = HttpVersion::kHttp11;
  bool keep_alive_ = true;
  bool expect_continue_ = false;
};

class HttpResponseView {
public:
  constexpr HttpResponseView() noexcept = default;
  constexpr HttpResponseView(int status, StringView reason, HttpVersion version,
                             HttpHeadersView headers, bool keep_alive) noexcept
      : reason_(reason), headers_(headers), status_(status), version_(version),
        keep_alive_(keep_alive) {}

  constexpr int status() const noexcept { return status_; }
  constexpr StringView reason() const noexcept { return reason_; }
  constexpr HttpVersion version() const noexcept { return version_; }
  constexpr HttpHeadersView headers() const noexcept { return headers_; }
  constexpr bool keep_alive() const noexcept { return keep_alive_; }

private:
  StringView reason_{};
  HttpHeadersView headers_{};
  int status_ = 0;
  HttpVersion version_ = HttpVersion::kHttp11;
  bool keep_alive_ = false;
};

struct HttpBodyChunkView {
  // Bytes remain valid until the next operation on this reader or until its
  // connection is destroyed. A final chunk may be empty and carry trailers.
  std::span<const std::byte> bytes{};
  HttpHeadersView trailers{};
  bool end_stream = false;
};

struct HttpBodyChunk {
  // A source borrows bytes/trailers until the pending writer await completes.
  std::span<const std::byte> bytes{};
  std::span<const HttpHeader> trailers{};
  bool end_stream = false;
};

namespace detail {
struct HttpBodyState;
struct HttpWriteState;
class HttpConnectionAccess;
} // namespace detail

class HttpBodyReader;
class HttpBodyWriter;

HYLE_IO_HTTP_API sched::Task<Result<HttpBodyChunkView>>
http_body_read(mem::AllocatorRef, HttpBodyReader &, std::size_t,
               OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Result<StringView>>
http_body_read_all(mem::AllocatorRef, HttpBodyReader &, std::size_t,
                   OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status>
http_body_discard(mem::AllocatorRef, HttpBodyReader &,
                  OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status> http_body_write(mem::AllocatorRef,
                                                     HttpBodyWriter &,
                                                     std::span<const std::byte>,
                                                     OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status>
http_body_finish(mem::AllocatorRef, HttpBodyWriter &,
                 std::span<const HttpHeader>, OperationOptions) noexcept;

// Move-only borrow over the active message body. Dropping an unfinished reader
// makes the connection non-reusable; discard() explicitly drains it when a
// caller wants keep-alive after ignoring a body.
class HttpBodyReader {
public:
  constexpr HttpBodyReader() noexcept = default;
  HYLE_IO_HTTP_API ~HttpBodyReader();
  HYLE_IO_HTTP_API HttpBodyReader(HttpBodyReader &&other) noexcept;
  HYLE_IO_HTTP_API HttpBodyReader &operator=(HttpBodyReader &&other) noexcept;
  HttpBodyReader(const HttpBodyReader &) = delete;
  HttpBodyReader &operator=(const HttpBodyReader &) = delete;

  HYLE_IO_HTTP_API bool valid() const noexcept;
  HYLE_IO_HTTP_API bool done() const noexcept;
  HYLE_IO_HTTP_API std::optional<std::uint64_t> remaining() const noexcept;

  // max_bytes==0 selects HttpLimits::read_chunk_bytes. Only one read/discard
  // may be in flight. The limit applies to returned payload, not framing bytes.
  sched::Task<Result<HttpBodyChunkView>>
  read(std::size_t max_bytes = 0, OperationOptions options = {}) noexcept {
    return http_body_read(allocator_, *this, max_bytes, options);
  }
  // Bounded convenience for JSON and other small bodies. The returned view is
  // connection-owned and is invalidated by the next reader/connection
  // operation.
  sched::Task<Result<StringView>>
  read_all(std::size_t max_bytes, OperationOptions options = {}) noexcept {
    return http_body_read_all(allocator_, *this, max_bytes, options);
  }
  sched::Task<Status> discard(OperationOptions options = {}) noexcept {
    return http_body_discard(allocator_, *this, options);
  }

private:
  friend class detail::HttpConnectionAccess;
  friend HYLE_IO_HTTP_API sched::Task<Result<HttpBodyChunkView>>
  http_body_read(mem::AllocatorRef, HttpBodyReader &, std::size_t,
                 OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Result<StringView>>
  http_body_read_all(mem::AllocatorRef, HttpBodyReader &, std::size_t,
                     OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  http_body_discard(mem::AllocatorRef, HttpBodyReader &,
                    OperationOptions) noexcept;

  HttpBodyReader(detail::HttpBodyState *state, std::uint64_t generation,
                 mem::AllocatorRef allocator) noexcept
      : state_(state), generation_(generation), allocator_(allocator) {}
  HYLE_IO_HTTP_API void abandon_() noexcept;

  detail::HttpBodyState *state_ = nullptr;
  std::uint64_t generation_ = 0;
  mem::AllocatorRef allocator_{};
};

// Non-owning, type-erased asynchronous pull source. Source::next must return
// Task<Result<HttpBodyChunk>> and must keep yielded bytes alive through that
// await. The Task frame uses the connection allocator; no queue or
// std::function allocation is introduced by HTTP.
class HttpBodySource {
public:
  using NextFn = sched::Task<Result<HttpBodyChunk>> (*)(
      mem::AllocatorRef, void *, OperationOptions) noexcept;

  constexpr HttpBodySource() noexcept = default;

  template <class Source>
    requires requires(Source &source, mem::AllocatorRef allocator,
                      OperationOptions options) {
      {
        source.next(allocator, options)
      } noexcept -> std::same_as<sched::Task<Result<HttpBodyChunk>>>;
    }
  static HttpBodySource
  from(Source &source,
       std::optional<std::uint64_t> content_length = std::nullopt) noexcept {
    return HttpBodySource(
        &source,
        [](mem::AllocatorRef allocator, void *context,
           OperationOptions options) noexcept {
          return static_cast<Source *>(context)->next(allocator, options);
        },
        content_length);
  }

  static constexpr HttpBodySource from_callback(
      void *context, NextFn next,
      std::optional<std::uint64_t> content_length = std::nullopt) noexcept {
    return HttpBodySource(context, next, content_length);
  }

  constexpr bool valid() const noexcept { return next_ != nullptr; }
  constexpr std::optional<std::uint64_t> content_length() const noexcept {
    return content_length_;
  }

private:
  friend class detail::HttpConnectionAccess;
  constexpr HttpBodySource(void *context, NextFn next,
                           std::optional<std::uint64_t> length) noexcept
      : context_(context), next_(next), content_length_(length) {}

  void *context_ = nullptr;
  NextFn next_ = nullptr;
  std::optional<std::uint64_t> content_length_{};
};

// Move-only streaming writer. Each write waits for transport progress before
// returning, so producer speed is naturally bounded by the peer. finish() emits
// the terminating chunk/trailers when chunked and retires the connection state.
class HttpBodyWriter {
public:
  constexpr HttpBodyWriter() noexcept = default;
  HYLE_IO_HTTP_API ~HttpBodyWriter();
  HYLE_IO_HTTP_API HttpBodyWriter(HttpBodyWriter &&other) noexcept;
  HYLE_IO_HTTP_API HttpBodyWriter &operator=(HttpBodyWriter &&other) noexcept;
  HttpBodyWriter(const HttpBodyWriter &) = delete;
  HttpBodyWriter &operator=(const HttpBodyWriter &) = delete;

  HYLE_IO_HTTP_API bool valid() const noexcept;
  HYLE_IO_HTTP_API std::optional<std::uint64_t> remaining() const noexcept;
  sched::Task<Status> write(std::span<const std::byte> bytes,
                            OperationOptions options = {}) noexcept {
    return http_body_write(allocator_, *this, bytes, options);
  }
  sched::Task<Status> finish(std::span<const HttpHeader> trailers = {},
                             OperationOptions options = {}) noexcept {
    return http_body_finish(allocator_, *this, trailers, options);
  }

private:
  friend class detail::HttpConnectionAccess;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  http_body_write(mem::AllocatorRef, HttpBodyWriter &,
                  std::span<const std::byte>, OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  http_body_finish(mem::AllocatorRef, HttpBodyWriter &,
                   std::span<const HttpHeader>, OperationOptions) noexcept;

  HttpBodyWriter(detail::HttpWriteState *state, std::uint64_t generation,
                 mem::AllocatorRef allocator) noexcept
      : state_(state), generation_(generation), allocator_(allocator) {}
  HYLE_IO_HTTP_API void abandon_() noexcept;

  detail::HttpWriteState *state_ = nullptr;
  std::uint64_t generation_ = 0;
  mem::AllocatorRef allocator_{};
};

struct HttpIncomingRequest {
  HttpRequestView head;
  HttpBodyReader body;
};

struct HttpIncomingResponse {
  HttpResponseView head;
  HttpBodyReader body;
};

struct HttpRequestHead {
  StringView method = "GET";
  StringView target = "/";
  // Required for HTTP/1.1 and serialized as Host. Caller headers must not
  // duplicate framing, Host, Connection, or Expect fields owned by this type.
  StringView authority{};
  std::span<const HttpHeader> headers{};
  HttpVersion version = HttpVersion::kHttp11;
  bool keep_alive = true;
  bool expect_continue = false;
};

struct HttpResponseHead {
  int status = 200;
  // Empty selects the standard phrase for common status codes.
  StringView reason{};
  std::span<const HttpHeader> headers{};
  bool keep_alive = true;
};

class HttpTransport;

namespace detail {
HYLE_IO_HTTP_API sched::Task<TransferResult>
http_transport_read(mem::AllocatorRef, HttpTransport &, std::span<std::byte>,
                    OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<TransferResult>
http_transport_write(mem::AllocatorRef, HttpTransport &,
                     std::span<const std::byte>, OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<BufferSequenceResult>
http_transport_write_all(mem::AllocatorRef, HttpTransport &,
                         std::span<const ConstBuffer>,
                         OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status>
http_transport_shutdown(mem::AllocatorRef, HttpTransport &,
                        OperationOptions) noexcept;
} // namespace detail

// Owning type-erased byte transport shared by HTTP/1 and HTTP/2. std::variant
// keeps the object at max(TCP,TLS) rather than paying storage for both legs.
class HttpTransport {
public:
  HttpTransport() noexcept = default;
  explicit HttpTransport(TcpStream stream) noexcept
      : stream_(std::in_place_type<TcpStream>, std::move(stream)) {}
#if defined(HYLE_IO_TLS)
  explicit HttpTransport(TlsStream stream) noexcept
      : stream_(std::in_place_type<TlsStream>, std::move(stream)) {}
#endif
  HttpTransport(HttpTransport &&) noexcept = default;
  HttpTransport &operator=(HttpTransport &&) noexcept = default;
  HttpTransport(const HttpTransport &) = delete;
  HttpTransport &operator=(const HttpTransport &) = delete;

  HYLE_IO_HTTP_API bool valid() const noexcept;
  HYLE_IO_HTTP_API bool encrypted() const noexcept;
  HYLE_IO_HTTP_API unsigned owner_worker() const noexcept;

  sched::Task<TransferResult>
  read_some(mem::AllocatorRef allocator, std::span<std::byte> buffer,
            OperationOptions options = {}) noexcept {
    return detail::http_transport_read(allocator, *this, buffer, options);
  }
  sched::Task<TransferResult>
  write_some(mem::AllocatorRef allocator, std::span<const std::byte> buffer,
             OperationOptions options = {}) noexcept {
    return detail::http_transport_write(allocator, *this, buffer, options);
  }
  sched::Task<Status> shutdown(mem::AllocatorRef allocator,
                               OperationOptions options = {}) noexcept {
    return detail::http_transport_shutdown(allocator, *this, options);
  }

private:
  friend HYLE_IO_HTTP_API sched::Task<TransferResult>
  detail::http_transport_read(mem::AllocatorRef, HttpTransport &,
                              std::span<std::byte>, OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<TransferResult>
  detail::http_transport_write(mem::AllocatorRef, HttpTransport &,
                               std::span<const std::byte>,
                               OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<BufferSequenceResult>
  detail::http_transport_write_all(mem::AllocatorRef, HttpTransport &,
                                   std::span<const ConstBuffer>,
                                   OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  detail::http_transport_shutdown(mem::AllocatorRef, HttpTransport &,
                                  OperationOptions) noexcept;

#if defined(HYLE_IO_TLS)
  std::variant<std::monostate, TcpStream, TlsStream> stream_{};
#else
  std::variant<std::monostate, TcpStream> stream_{};
#endif
};

class HttpServerConnection;
class HttpClientConnection;

HYLE_IO_HTTP_API sched::Task<Result<std::optional<HttpIncomingRequest>>>
http_server_read_request(mem::AllocatorRef, HttpServerConnection &,
                         OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status>
http_server_send_continue(mem::AllocatorRef, HttpServerConnection &,
                          OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Result<HttpBodyWriter>> http_server_start_response(
    mem::AllocatorRef, HttpServerConnection &, const HttpResponseHead &,
    std::optional<std::uint64_t>, OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status>
http_server_send_fixed(mem::AllocatorRef, HttpServerConnection &,
                       const HttpResponseHead &, std::span<const std::byte>,
                       OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Status>
http_server_send_source(mem::AllocatorRef, HttpServerConnection &,
                        const HttpResponseHead &, HttpBodySource,
                        OperationOptions) noexcept;

class HttpServerConnection {
public:
  HttpServerConnection() noexcept = default;
  HYLE_IO_HTTP_API ~HttpServerConnection();
  HYLE_IO_HTTP_API HttpServerConnection(HttpServerConnection &&) noexcept;
  HYLE_IO_HTTP_API HttpServerConnection &
  operator=(HttpServerConnection &&) noexcept;
  HttpServerConnection(const HttpServerConnection &) = delete;
  HttpServerConnection &operator=(const HttpServerConnection &) = delete;

  HYLE_IO_HTTP_API static Result<HttpServerConnection>
  create(mem::AllocatorRef allocator, HttpTransport transport,
         HttpLimits limits = {}) noexcept;
  static Result<HttpServerConnection> create(mem::AllocatorRef allocator,
                                             TcpStream stream,
                                             HttpLimits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#if defined(HYLE_IO_TLS)
  static Result<HttpServerConnection> create(mem::AllocatorRef allocator,
                                             TlsStream stream,
                                             HttpLimits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#endif

  HYLE_IO_HTTP_API bool valid() const noexcept;
  HYLE_IO_HTTP_API unsigned owner_worker() const noexcept;
  const HttpLimits &limits() const noexcept { return limits_; }

  sched::Task<Result<std::optional<HttpIncomingRequest>>>
  read_request(OperationOptions options = {}) noexcept {
    return http_server_read_request(allocator_, *this, options);
  }
  sched::Task<Status> send_continue(OperationOptions options = {}) noexcept {
    return http_server_send_continue(allocator_, *this, options);
  }
  sched::Task<Result<HttpBodyWriter>>
  start_response(const HttpResponseHead &head,
                 std::optional<std::uint64_t> content_length = std::nullopt,
                 OperationOptions options = {}) noexcept {
    return http_server_start_response(allocator_, *this, head, content_length,
                                      options);
  }
  sched::Task<Status> send_response(const HttpResponseHead &head,
                                    std::span<const std::byte> body,
                                    OperationOptions options = {}) noexcept {
    return http_server_send_fixed(allocator_, *this, head, body, options);
  }
  sched::Task<Status> send_response(const HttpResponseHead &head,
                                    HttpBodySource body,
                                    OperationOptions options = {}) noexcept {
    return http_server_send_source(allocator_, *this, head, body, options);
  }
  HYLE_IO_HTTP_API sched::Task<Status>
  close(OperationOptions options = {}) noexcept;

private:
  friend class detail::HttpConnectionAccess;
  friend HYLE_IO_HTTP_API
      sched::Task<Result<std::optional<HttpIncomingRequest>>>
      http_server_read_request(mem::AllocatorRef, HttpServerConnection &,
                               OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  http_server_send_continue(mem::AllocatorRef, HttpServerConnection &,
                            OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Result<HttpBodyWriter>>
  http_server_start_response(mem::AllocatorRef, HttpServerConnection &,
                             const HttpResponseHead &,
                             std::optional<std::uint64_t>,
                             OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  http_server_send_fixed(mem::AllocatorRef, HttpServerConnection &,
                         const HttpResponseHead &, std::span<const std::byte>,
                         OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Status>
  http_server_send_source(mem::AllocatorRef, HttpServerConnection &,
                          const HttpResponseHead &, HttpBodySource,
                          OperationOptions) noexcept;

  struct Impl;
  HttpServerConnection(mem::AllocatorRef allocator, HttpLimits limits,
                       Impl *impl) noexcept
      : impl_(impl), allocator_(allocator), limits_(limits) {}

  Impl *impl_ = nullptr;
  mem::AllocatorRef allocator_{};
  HttpLimits limits_{};
};

HYLE_IO_HTTP_API sched::Task<Result<HttpBodyWriter>>
http_client_start_request(mem::AllocatorRef, HttpClientConnection &,
                          const HttpRequestHead &, std::optional<std::uint64_t>,
                          OperationOptions) noexcept;
// A missing response means the caller may send the body: either the peer sent
// 100 Continue or the bounded continue wait elapsed. A present response is an
// early final response and disarms the supplied writer.
HYLE_IO_HTTP_API
sched::Task<Result<std::optional<HttpIncomingResponse>>>
http_client_wait_for_continue(mem::AllocatorRef, HttpClientConnection &,
                              HttpBodyWriter &, OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Result<HttpIncomingResponse>>
http_client_read_response(mem::AllocatorRef, HttpClientConnection &,
                          OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Result<HttpIncomingResponse>>
http_client_request_fixed(mem::AllocatorRef, HttpClientConnection &,
                          const HttpRequestHead &, std::span<const std::byte>,
                          OperationOptions) noexcept;
HYLE_IO_HTTP_API sched::Task<Result<HttpIncomingResponse>>
http_client_request_source(mem::AllocatorRef, HttpClientConnection &,
                           const HttpRequestHead &, HttpBodySource,
                           OperationOptions) noexcept;

class HttpClientConnection {
public:
  HttpClientConnection() noexcept = default;
  HYLE_IO_HTTP_API ~HttpClientConnection();
  HYLE_IO_HTTP_API HttpClientConnection(HttpClientConnection &&) noexcept;
  HYLE_IO_HTTP_API HttpClientConnection &
  operator=(HttpClientConnection &&) noexcept;
  HttpClientConnection(const HttpClientConnection &) = delete;
  HttpClientConnection &operator=(const HttpClientConnection &) = delete;

  HYLE_IO_HTTP_API static Result<HttpClientConnection>
  create(mem::AllocatorRef allocator, HttpTransport transport,
         HttpLimits limits = {}) noexcept;
  static Result<HttpClientConnection> create(mem::AllocatorRef allocator,
                                             TcpStream stream,
                                             HttpLimits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#if defined(HYLE_IO_TLS)
  static Result<HttpClientConnection> create(mem::AllocatorRef allocator,
                                             TlsStream stream,
                                             HttpLimits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#endif

  HYLE_IO_HTTP_API bool valid() const noexcept;
  HYLE_IO_HTTP_API unsigned owner_worker() const noexcept;
  const HttpLimits &limits() const noexcept { return limits_; }

  sched::Task<Result<HttpBodyWriter>>
  start_request(const HttpRequestHead &head,
                std::optional<std::uint64_t> content_length = std::nullopt,
                OperationOptions options = {}) noexcept {
    return http_client_start_request(allocator_, *this, head, content_length,
                                     options);
  }
  // Call after start_request() when head.expect_continue was true and before
  // the first body write. std::nullopt means 100 Continue was received; a
  // value is the server's early final response (for example 413/417), in which
  // case writer is invalidated without transmitting the body. A null result
  // also covers the configured expect_continue_wait fallback when a server
  // ignores Expect. The fixed/source conveniences perform this handshake
  // automatically.
  sched::Task<Result<std::optional<HttpIncomingResponse>>>
  wait_for_continue(HttpBodyWriter &writer,
                    OperationOptions options = {}) noexcept {
    return http_client_wait_for_continue(allocator_, *this, writer, options);
  }
  sched::Task<Result<HttpIncomingResponse>>
  read_response(OperationOptions options = {}) noexcept {
    return http_client_read_response(allocator_, *this, options);
  }
  sched::Task<Result<HttpIncomingResponse>>
  request(const HttpRequestHead &head, std::span<const std::byte> body,
          OperationOptions options = {}) noexcept {
    return http_client_request_fixed(allocator_, *this, head, body, options);
  }
  sched::Task<Result<HttpIncomingResponse>>
  request(const HttpRequestHead &head, HttpBodySource body,
          OperationOptions options = {}) noexcept {
    return http_client_request_source(allocator_, *this, head, body, options);
  }
  HYLE_IO_HTTP_API sched::Task<Status>
  close(OperationOptions options = {}) noexcept;

private:
  friend class detail::HttpConnectionAccess;
  friend HYLE_IO_HTTP_API sched::Task<Result<HttpBodyWriter>>
  http_client_start_request(mem::AllocatorRef, HttpClientConnection &,
                            const HttpRequestHead &,
                            std::optional<std::uint64_t>,
                            OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API
      sched::Task<Result<std::optional<HttpIncomingResponse>>>
      http_client_wait_for_continue(mem::AllocatorRef, HttpClientConnection &,
                                    HttpBodyWriter &,
                                    OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Result<HttpIncomingResponse>>
  http_client_read_response(mem::AllocatorRef, HttpClientConnection &,
                            OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Result<HttpIncomingResponse>>
  http_client_request_fixed(mem::AllocatorRef, HttpClientConnection &,
                            const HttpRequestHead &, std::span<const std::byte>,
                            OperationOptions) noexcept;
  friend HYLE_IO_HTTP_API sched::Task<Result<HttpIncomingResponse>>
  http_client_request_source(mem::AllocatorRef, HttpClientConnection &,
                             const HttpRequestHead &, HttpBodySource,
                             OperationOptions) noexcept;

  struct Impl;
  HttpClientConnection(mem::AllocatorRef allocator, HttpLimits limits,
                       Impl *impl) noexcept
      : impl_(impl), allocator_(allocator), limits_(limits) {}

  Impl *impl_ = nullptr;
  mem::AllocatorRef allocator_{};
  HttpLimits limits_{};
};

static_assert(std::is_trivially_copyable_v<HttpHeader>);
static_assert(std::is_trivially_copyable_v<HttpHeadersView>);
static_assert(std::is_trivially_copyable_v<HttpRequestView>);
static_assert(std::is_trivially_copyable_v<HttpResponseView>);

} // namespace hyle::io
