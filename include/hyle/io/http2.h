#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "hyle/io/http.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_HTTP2_API
#elif defined(HYLE_IO_HTTP2_BUILD)
#define HYLE_IO_HTTP2_API HYLE_EXPORT
#else
#define HYLE_IO_HTTP2_API HYLE_IMPORT
#endif

namespace hyle::io {

HYLE_IO_HTTP2_API bool http2_available() noexcept;

} // namespace hyle::io

#if defined(HYLE_IO_HTTP2)

namespace hyle::io {

// nghttp2 owns framing, HPACK and protocol stream state. HYLE owns the
// allocator-backed lifecycle, operation budgets and byte transport. A session
// and every handle derived from it stay on the transport's owner worker.
struct Http2Limits {
  std::size_t max_header_count = 128;
  std::size_t max_header_bytes = 64 * 1024;
  std::uint64_t max_body_bytes = 64ull * 1024 * 1024;
  std::uint32_t max_concurrent_streams = 100;
  // A value below the protocol's initial 65,535-byte stream window takes
  // effect only after the peer processes SETTINGS. HYLE safely absorbs that
  // bounded in-flight allowance before enforcing the steady-state window.
  std::uint32_t stream_receive_window = 64 * 1024;
  // HTTP/2 has no SETTINGS field that can reduce the initial connection
  // window, so this value must be at least 65,535.
  std::uint32_t connection_receive_window = 1024 * 1024;
  std::size_t read_chunk_bytes = 16 * 1024;
};

enum class Http2Error : std::uint32_t {
  kNoError = 0,
  kProtocol = 1,
  kInternal = 2,
  kFlowControl = 3,
  kSettingsTimeout = 4,
  kStreamClosed = 5,
  kFrameSize = 6,
  kRefusedStream = 7,
  kCancel = 8,
  kCompression = 9,
  kConnect = 10,
  kEnhanceYourCalm = 11,
  kInadequateSecurity = 12,
  kHttp11Required = 13,
};

class Http2StreamId {
public:
  constexpr Http2StreamId() noexcept = default;
  static constexpr std::optional<Http2StreamId>
  from_value(std::int32_t value) noexcept {
    return value > 0 ? std::optional<Http2StreamId>(Http2StreamId(value))
                     : std::nullopt;
  }
  constexpr bool valid() const noexcept { return value_ > 0; }
  constexpr std::int32_t value() const noexcept { return value_; }
  friend constexpr bool operator==(Http2StreamId,
                                   Http2StreamId) noexcept = default;

private:
  explicit constexpr Http2StreamId(std::int32_t value) noexcept
      : value_(value) {}
  std::int32_t value_ = 0;
};

class Http2RequestView {
public:
  constexpr Http2RequestView() noexcept = default;
  constexpr Http2RequestView(StringView method, StringView scheme,
                             StringView authority, StringView path,
                             HttpHeadersView headers) noexcept
      : method_(method), scheme_(scheme), authority_(authority), path_(path),
        headers_(headers) {}
  constexpr StringView method() const noexcept { return method_; }
  constexpr StringView scheme() const noexcept { return scheme_; }
  constexpr StringView authority() const noexcept { return authority_; }
  constexpr StringView path() const noexcept { return path_; }
  constexpr HttpHeadersView headers() const noexcept { return headers_; }

private:
  StringView method_{};
  StringView scheme_{};
  StringView authority_{};
  StringView path_{};
  HttpHeadersView headers_{};
};

class Http2ResponseView {
public:
  constexpr Http2ResponseView() noexcept = default;
  constexpr Http2ResponseView(int status, HttpHeadersView headers) noexcept
      : headers_(headers), status_(status) {}
  constexpr int status() const noexcept { return status_; }
  constexpr HttpHeadersView headers() const noexcept { return headers_; }

private:
  HttpHeadersView headers_{};
  int status_ = 0;
};

struct Http2RequestHead {
  StringView method = "GET";
  StringView scheme = "https";
  StringView authority{};
  StringView path = "/";
  std::span<const HttpHeader> headers{};
};

struct Http2ResponseHead {
  int status = 200;
  std::span<const HttpHeader> headers{};
};

struct Http2PeerGoaway {
  Http2StreamId last_stream_id{};
  Http2Error error = Http2Error::kNoError;
  // Session-owned and invalidated by the next session operation.
  StringView debug_data{};
};

namespace detail {
struct Http2SessionState;
struct Http2StreamState;
class Http2SessionAccess;
} // namespace detail

class Http2BodyReader;
class Http2BodyWriter;
class Http2ServerSession;
class Http2ClientSession;

HYLE_IO_HTTP2_API sched::Task<Result<HttpBodyChunkView>>
http2_body_read(mem::AllocatorRef, Http2BodyReader &, std::size_t,
                OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Status>
http2_body_discard(mem::AllocatorRef, Http2BodyReader &,
                   OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Status>
http2_body_write(mem::AllocatorRef, Http2BodyWriter &,
                 std::span<const std::byte>, OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Status>
http2_body_finish(mem::AllocatorRef, Http2BodyWriter &,
                  std::span<const HttpHeader>, OperationOptions) noexcept;

class Http2BodyReader {
public:
  constexpr Http2BodyReader() noexcept = default;
  HYLE_IO_HTTP2_API ~Http2BodyReader();
  HYLE_IO_HTTP2_API Http2BodyReader(Http2BodyReader &&) noexcept;
  HYLE_IO_HTTP2_API Http2BodyReader &operator=(Http2BodyReader &&) noexcept;
  Http2BodyReader(const Http2BodyReader &) = delete;
  Http2BodyReader &operator=(const Http2BodyReader &) = delete;

  HYLE_IO_HTTP2_API bool valid() const noexcept;
  HYLE_IO_HTTP2_API bool done() const noexcept;
  HYLE_IO_HTTP2_API std::optional<Http2Error> stream_error() const noexcept;
  sched::Task<Result<HttpBodyChunkView>>
  read(std::size_t max_bytes = 0, OperationOptions options = {}) noexcept {
    return http2_body_read(allocator_, *this, max_bytes, options);
  }
  sched::Task<Status> discard(OperationOptions options = {}) noexcept {
    return http2_body_discard(allocator_, *this, options);
  }

private:
  friend class detail::Http2SessionAccess;
  friend HYLE_IO_HTTP2_API sched::Task<Result<HttpBodyChunkView>>
  http2_body_read(mem::AllocatorRef, Http2BodyReader &, std::size_t,
                  OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Status>
  http2_body_discard(mem::AllocatorRef, Http2BodyReader &,
                     OperationOptions) noexcept;
  Http2BodyReader(detail::Http2SessionState *session,
                  detail::Http2StreamState *stream, std::uint64_t generation,
                  mem::AllocatorRef allocator) noexcept
      : session_(session), stream_(stream), generation_(generation),
        allocator_(allocator) {}
  HYLE_IO_HTTP2_API void abandon_() noexcept;
  detail::Http2SessionState *session_ = nullptr;
  detail::Http2StreamState *stream_ = nullptr;
  std::uint64_t generation_ = 0;
  mem::AllocatorRef allocator_{};
};

class Http2BodyWriter {
public:
  constexpr Http2BodyWriter() noexcept = default;
  HYLE_IO_HTTP2_API ~Http2BodyWriter();
  HYLE_IO_HTTP2_API Http2BodyWriter(Http2BodyWriter &&) noexcept;
  HYLE_IO_HTTP2_API Http2BodyWriter &operator=(Http2BodyWriter &&) noexcept;
  Http2BodyWriter(const Http2BodyWriter &) = delete;
  Http2BodyWriter &operator=(const Http2BodyWriter &) = delete;

  HYLE_IO_HTTP2_API bool valid() const noexcept;
  HYLE_IO_HTTP2_API bool done() const noexcept;
  sched::Task<Status> write(std::span<const std::byte> bytes,
                            OperationOptions options = {}) noexcept {
    return http2_body_write(allocator_, *this, bytes, options);
  }
  sched::Task<Status> finish(std::span<const HttpHeader> trailers = {},
                             OperationOptions options = {}) noexcept {
    return http2_body_finish(allocator_, *this, trailers, options);
  }

private:
  friend class detail::Http2SessionAccess;
  friend HYLE_IO_HTTP2_API sched::Task<Status>
  http2_body_write(mem::AllocatorRef, Http2BodyWriter &,
                   std::span<const std::byte>, OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Status>
  http2_body_finish(mem::AllocatorRef, Http2BodyWriter &,
                    std::span<const HttpHeader>, OperationOptions) noexcept;
  Http2BodyWriter(detail::Http2SessionState *session,
                  detail::Http2StreamState *stream, std::uint64_t generation,
                  mem::AllocatorRef allocator) noexcept
      : session_(session), stream_(stream), generation_(generation),
        allocator_(allocator) {}
  HYLE_IO_HTTP2_API void abandon_() noexcept;
  detail::Http2SessionState *session_ = nullptr;
  detail::Http2StreamState *stream_ = nullptr;
  std::uint64_t generation_ = 0;
  mem::AllocatorRef allocator_{};
};

struct Http2IncomingRequest {
  Http2StreamId stream_id{};
  Http2RequestView head{};
  std::optional<Http2Error> stream_error{};
  Http2BodyReader body{};
};
struct Http2IncomingResponse {
  Http2StreamId stream_id{};
  Http2ResponseView head{};
  std::optional<Http2Error> stream_error{};
  Http2BodyReader body{};
};
struct Http2OutgoingRequest {
  Http2StreamId stream_id{};
  Http2BodyWriter body{};
};

HYLE_IO_HTTP2_API sched::Task<Result<std::optional<Http2IncomingRequest>>>
http2_server_accept(mem::AllocatorRef, Http2ServerSession &,
                    OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Result<Http2BodyWriter>>
http2_server_start_response(mem::AllocatorRef, Http2ServerSession &,
                            Http2StreamId, const Http2ResponseHead &,
                            std::optional<std::uint64_t>,
                            OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Status>
http2_server_send_fixed(mem::AllocatorRef, Http2ServerSession &, Http2StreamId,
                        const Http2ResponseHead &, std::span<const std::byte>,
                        OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Status>
http2_server_send_source(mem::AllocatorRef, Http2ServerSession &, Http2StreamId,
                         const Http2ResponseHead &, HttpBodySource,
                         OperationOptions) noexcept;

class Http2ServerSession {
public:
  Http2ServerSession() noexcept = default;
  HYLE_IO_HTTP2_API ~Http2ServerSession();
  HYLE_IO_HTTP2_API Http2ServerSession(Http2ServerSession &&) noexcept;
  HYLE_IO_HTTP2_API Http2ServerSession &
  operator=(Http2ServerSession &&) noexcept;
  Http2ServerSession(const Http2ServerSession &) = delete;
  Http2ServerSession &operator=(const Http2ServerSession &) = delete;

  HYLE_IO_HTTP2_API static Result<Http2ServerSession>
      create(mem::AllocatorRef, HttpTransport, Http2Limits = {}) noexcept;
  static Result<Http2ServerSession> create(mem::AllocatorRef allocator,
                                           TcpStream stream,
                                           Http2Limits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#if defined(HYLE_IO_TLS)
  static Result<Http2ServerSession> create(mem::AllocatorRef allocator,
                                           TlsStream stream,
                                           Http2Limits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#endif
  HYLE_IO_HTTP2_API bool valid() const noexcept;
  HYLE_IO_HTTP2_API unsigned owner_worker() const noexcept;
  HYLE_IO_HTTP2_API const Http2Limits &limits() const noexcept;
  HYLE_IO_HTTP2_API std::optional<Http2PeerGoaway> peer_goaway() const noexcept;
  sched::Task<Result<std::optional<Http2IncomingRequest>>>
  accept(OperationOptions options = {}) noexcept {
    return http2_server_accept(allocator_, *this, options);
  }
  sched::Task<Result<Http2BodyWriter>>
  start_response(Http2StreamId id, const Http2ResponseHead &head,
                 std::optional<std::uint64_t> content_length = std::nullopt,
                 OperationOptions options = {}) noexcept {
    return http2_server_start_response(allocator_, *this, id, head,
                                       content_length, options);
  }
  sched::Task<Status> send_response(Http2StreamId id,
                                    const Http2ResponseHead &head,
                                    std::span<const std::byte> body,
                                    OperationOptions options = {}) noexcept {
    return http2_server_send_fixed(allocator_, *this, id, head, body, options);
  }
  sched::Task<Status> send_response(Http2StreamId id,
                                    const Http2ResponseHead &head,
                                    HttpBodySource body,
                                    OperationOptions options = {}) noexcept {
    return http2_server_send_source(allocator_, *this, id, head, body, options);
  }
  HYLE_IO_HTTP2_API sched::Task<Status> reset(Http2StreamId,
                                              Http2Error = Http2Error::kCancel,
                                              OperationOptions = {}) noexcept;
  HYLE_IO_HTTP2_API sched::Task<Status> drain(Http2Error = Http2Error::kNoError,
                                              OperationOptions = {}) noexcept;
  HYLE_IO_HTTP2_API sched::Task<Status> close(OperationOptions = {}) noexcept;

private:
  friend class detail::Http2SessionAccess;
  friend HYLE_IO_HTTP2_API
      sched::Task<Result<std::optional<Http2IncomingRequest>>>
      http2_server_accept(mem::AllocatorRef, Http2ServerSession &,
                          OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Result<Http2BodyWriter>>
  http2_server_start_response(mem::AllocatorRef, Http2ServerSession &,
                              Http2StreamId, const Http2ResponseHead &,
                              std::optional<std::uint64_t>,
                              OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Status>
  http2_server_send_fixed(mem::AllocatorRef, Http2ServerSession &,
                          Http2StreamId, const Http2ResponseHead &,
                          std::span<const std::byte>,
                          OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Status>
  http2_server_send_source(mem::AllocatorRef, Http2ServerSession &,
                           Http2StreamId, const Http2ResponseHead &,
                           HttpBodySource, OperationOptions) noexcept;
  detail::Http2SessionState *state_ = nullptr;
  mem::AllocatorRef allocator_{};
};

HYLE_IO_HTTP2_API sched::Task<Result<Http2OutgoingRequest>>
http2_client_start_request(mem::AllocatorRef, Http2ClientSession &,
                           const Http2RequestHead &,
                           std::optional<std::uint64_t>,
                           OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Result<Http2StreamId>>
http2_client_send_fixed(mem::AllocatorRef, Http2ClientSession &,
                        const Http2RequestHead &, std::span<const std::byte>,
                        OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Result<Http2StreamId>>
http2_client_send_source(mem::AllocatorRef, Http2ClientSession &,
                         const Http2RequestHead &, HttpBodySource,
                         OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Result<std::optional<Http2IncomingResponse>>>
http2_client_next_response(mem::AllocatorRef, Http2ClientSession &,
                           OperationOptions) noexcept;
HYLE_IO_HTTP2_API sched::Task<Result<Http2IncomingResponse>>
http2_client_read_response(mem::AllocatorRef, Http2ClientSession &,
                           Http2StreamId, OperationOptions) noexcept;

class Http2ClientSession {
public:
  Http2ClientSession() noexcept = default;
  HYLE_IO_HTTP2_API ~Http2ClientSession();
  HYLE_IO_HTTP2_API Http2ClientSession(Http2ClientSession &&) noexcept;
  HYLE_IO_HTTP2_API Http2ClientSession &
  operator=(Http2ClientSession &&) noexcept;
  Http2ClientSession(const Http2ClientSession &) = delete;
  Http2ClientSession &operator=(const Http2ClientSession &) = delete;

  HYLE_IO_HTTP2_API static Result<Http2ClientSession>
      create(mem::AllocatorRef, HttpTransport, Http2Limits = {}) noexcept;
  static Result<Http2ClientSession> create(mem::AllocatorRef allocator,
                                           TcpStream stream,
                                           Http2Limits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#if defined(HYLE_IO_TLS)
  static Result<Http2ClientSession> create(mem::AllocatorRef allocator,
                                           TlsStream stream,
                                           Http2Limits limits = {}) noexcept {
    return create(allocator, HttpTransport(std::move(stream)), limits);
  }
#endif
  HYLE_IO_HTTP2_API bool valid() const noexcept;
  HYLE_IO_HTTP2_API unsigned owner_worker() const noexcept;
  HYLE_IO_HTTP2_API const Http2Limits &limits() const noexcept;
  HYLE_IO_HTTP2_API std::optional<Http2PeerGoaway> peer_goaway() const noexcept;
  sched::Task<Result<Http2OutgoingRequest>>
  start_request(const Http2RequestHead &head,
                std::optional<std::uint64_t> content_length = std::nullopt,
                OperationOptions options = {}) noexcept {
    return http2_client_start_request(allocator_, *this, head, content_length,
                                      options);
  }
  sched::Task<Result<Http2StreamId>>
  send_request(const Http2RequestHead &head, std::span<const std::byte> body,
               OperationOptions options = {}) noexcept {
    return http2_client_send_fixed(allocator_, *this, head, body, options);
  }
  sched::Task<Result<Http2StreamId>>
  send_request(const Http2RequestHead &head, HttpBodySource body,
               OperationOptions options = {}) noexcept {
    return http2_client_send_source(allocator_, *this, head, body, options);
  }
  sched::Task<Result<std::optional<Http2IncomingResponse>>>
  next_response(OperationOptions options = {}) noexcept {
    return http2_client_next_response(allocator_, *this, options);
  }
  sched::Task<Result<Http2IncomingResponse>>
  read_response(Http2StreamId id, OperationOptions options = {}) noexcept {
    return http2_client_read_response(allocator_, *this, id, options);
  }
  HYLE_IO_HTTP2_API sched::Task<Status> reset(Http2StreamId,
                                              Http2Error = Http2Error::kCancel,
                                              OperationOptions = {}) noexcept;
  HYLE_IO_HTTP2_API sched::Task<Status> drain(Http2Error = Http2Error::kNoError,
                                              OperationOptions = {}) noexcept;
  HYLE_IO_HTTP2_API sched::Task<Status> close(OperationOptions = {}) noexcept;

private:
  friend class detail::Http2SessionAccess;
  friend HYLE_IO_HTTP2_API sched::Task<Result<Http2OutgoingRequest>>
  http2_client_start_request(mem::AllocatorRef, Http2ClientSession &,
                             const Http2RequestHead &,
                             std::optional<std::uint64_t>,
                             OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Result<Http2StreamId>>
  http2_client_send_fixed(mem::AllocatorRef, Http2ClientSession &,
                          const Http2RequestHead &, std::span<const std::byte>,
                          OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Result<Http2StreamId>>
  http2_client_send_source(mem::AllocatorRef, Http2ClientSession &,
                           const Http2RequestHead &, HttpBodySource,
                           OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API
      sched::Task<Result<std::optional<Http2IncomingResponse>>>
      http2_client_next_response(mem::AllocatorRef, Http2ClientSession &,
                                 OperationOptions) noexcept;
  friend HYLE_IO_HTTP2_API sched::Task<Result<Http2IncomingResponse>>
  http2_client_read_response(mem::AllocatorRef, Http2ClientSession &,
                             Http2StreamId, OperationOptions) noexcept;
  detail::Http2SessionState *state_ = nullptr;
  mem::AllocatorRef allocator_{};
};

} // namespace hyle::io

#endif // HYLE_IO_HTTP2
