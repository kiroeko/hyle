#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/io/http.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-11 HTTP/2. nghttp2 is the protocol core (RFC 9113 framing, HPACK, stream
// state and flow control); HYLE owns the byte transport and drives that core from
// one shard-pinned fiber over TcpStream or an already-handshaken TlsStream.
//
// This API deliberately exposes messages and stream IDs, not nghttp2 types.
// Multiple requests may be submitted before responses are consumed, and frames
// from different streams may arrive interleaved. The session remains single-owner:
// every method on a connection must run on its transport's owner fiber.

#if defined(HYLE_IO_HTTP2)

namespace hyle::io {

inline constexpr std::size_t kHttp2MaxHeaders = 128;
inline constexpr std::uint32_t kHttp2MaxHeaderBytes = 64 * 1024;
inline constexpr std::uint64_t kHttp2MaxBodyBytes = 8ull * 1024 * 1024;
inline constexpr std::uint32_t kHttp2DefaultMaxConcurrentStreams = 100;

struct Http2Header {
  StringView name;
  StringView value;
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

class Http2ServerConn;
class Http2ClientConn;

// A completed request or response. Views point into connection-owned storage and
// remain valid until the next next_request()/next_response() call on that same
// connection. Header lookup is ASCII case-insensitive for regular fields.
class Http2MessageView {
 public:
  bool valid() const noexcept { return stream_ != nullptr; }
  HYLE_IO_API std::int32_t stream_id() const noexcept;
  HYLE_IO_API StringView method() const noexcept;
  HYLE_IO_API StringView scheme() const noexcept;
  HYLE_IO_API StringView authority() const noexcept;
  HYLE_IO_API StringView path() const noexcept;
  HYLE_IO_API int status() const noexcept;
  HYLE_IO_API Http2Error stream_error() const noexcept;
  HYLE_IO_API std::size_t header_count() const noexcept;
  HYLE_IO_API StringView header_name(std::size_t index) const noexcept;
  HYLE_IO_API StringView header_value(std::size_t index) const noexcept;
  HYLE_IO_API StringView header(StringView name) const noexcept;
  HYLE_IO_API StringView body() const noexcept;

 private:
  friend class Http2ServerConn;
  friend class Http2ClientConn;
  friend sched::Task<Http2MessageView*> http2_server_next(
      mem::AllocatorRef, Http2ServerConn&) noexcept;
  friend sched::Task<Http2MessageView*> http2_client_next(
      mem::AllocatorRef, Http2ClientConn&) noexcept;
  const void* stream_ = nullptr;
};

sched::Task<Http2MessageView*> http2_server_next(
    mem::AllocatorRef, Http2ServerConn&) noexcept;
sched::Task<bool> http2_server_respond(
    mem::AllocatorRef, Http2ServerConn&, std::int32_t, int, const Http2Header*,
    std::size_t, StringView) noexcept;
sched::Task<bool> http2_server_reset(mem::AllocatorRef, Http2ServerConn&,
                                     std::int32_t, Http2Error) noexcept;
sched::Task<bool> http2_server_shutdown(mem::AllocatorRef, Http2ServerConn&,
                                        Http2Error) noexcept;

// Server-side prior-knowledge HTTP/2 connection (h2c or TLS selected as "h2" by
// the caller's protocol negotiation). The constructor queues the initial SETTINGS.
class Http2ServerConn {
 public:
  HYLE_IO_API Http2ServerConn(mem::AllocatorRef alloc,
                              TcpStream stream) noexcept;
#if defined(HYLE_IO_TLS)
  HYLE_IO_API Http2ServerConn(mem::AllocatorRef alloc,
                              TlsStream stream) noexcept;
#endif
  HYLE_IO_API ~Http2ServerConn();
  HYLE_IO_API Http2ServerConn(Http2ServerConn&&) noexcept;
  HYLE_IO_API Http2ServerConn& operator=(Http2ServerConn&&) noexcept;
  Http2ServerConn(const Http2ServerConn&) = delete;
  Http2ServerConn& operator=(const Http2ServerConn&) = delete;

  bool valid() const noexcept { return impl_ != nullptr && transport_.valid(); }

  // Reads until any stream has a complete request. Interleaved streams are kept
  // independently; completed requests are returned in completion order.
  HYLE_IO_API sched::Task<Http2MessageView*> next_request() noexcept;

  // Queues HEADERS plus an optional fixed body and flushes all currently encoded
  // frames. nghttp2 applies peer flow-control windows and frame sizing.
  HYLE_IO_API sched::Task<bool> respond(
      std::int32_t stream_id, int status, const Http2Header* headers = nullptr,
      std::size_t header_count = 0, StringView body = {}) noexcept;
  HYLE_IO_API sched::Task<bool> reset(
      std::int32_t stream_id,
      Http2Error error = Http2Error::kCancel) noexcept;
  HYLE_IO_API sched::Task<bool> shutdown(
      Http2Error error = Http2Error::kNoError) noexcept;

 private:
  friend sched::Task<Http2MessageView*> http2_server_next(
      mem::AllocatorRef, Http2ServerConn&) noexcept;
  friend sched::Task<bool> http2_server_respond(
      mem::AllocatorRef, Http2ServerConn&, std::int32_t, int,
      const Http2Header*, std::size_t, StringView) noexcept;
  friend sched::Task<bool> http2_server_reset(
      mem::AllocatorRef, Http2ServerConn&, std::int32_t, Http2Error) noexcept;
  friend sched::Task<bool> http2_server_shutdown(
      mem::AllocatorRef, Http2ServerConn&, Http2Error) noexcept;
  struct Impl;
  Impl* impl_ = nullptr;
  HttpTransport transport_{};
  Http2MessageView current_{};
  mem::AllocatorRef alloc_{};
};

sched::Task<std::int32_t> http2_client_submit(
    mem::AllocatorRef, Http2ClientConn&, StringView, StringView, StringView,
    StringView, const Http2Header*, std::size_t, StringView) noexcept;
sched::Task<Http2MessageView*> http2_client_next(
    mem::AllocatorRef, Http2ClientConn&) noexcept;
sched::Task<bool> http2_client_reset(mem::AllocatorRef, Http2ClientConn&,
                                     std::int32_t, Http2Error) noexcept;
sched::Task<bool> http2_client_shutdown(mem::AllocatorRef, Http2ClientConn&,
                                        Http2Error) noexcept;

// Client connection. submit_request() may be called repeatedly before
// next_response(); returned positive stream IDs identify multiplexed requests.
class Http2ClientConn {
 public:
  HYLE_IO_API Http2ClientConn(mem::AllocatorRef alloc,
                              TcpStream stream) noexcept;
#if defined(HYLE_IO_TLS)
  HYLE_IO_API Http2ClientConn(mem::AllocatorRef alloc,
                              TlsStream stream) noexcept;
#endif
  HYLE_IO_API ~Http2ClientConn();
  HYLE_IO_API Http2ClientConn(Http2ClientConn&&) noexcept;
  HYLE_IO_API Http2ClientConn& operator=(Http2ClientConn&&) noexcept;
  Http2ClientConn(const Http2ClientConn&) = delete;
  Http2ClientConn& operator=(const Http2ClientConn&) = delete;

  bool valid() const noexcept { return impl_ != nullptr && transport_.valid(); }

  HYLE_IO_API sched::Task<std::int32_t> submit_request(
      StringView method, StringView scheme, StringView authority,
      StringView path, const Http2Header* headers = nullptr,
      std::size_t header_count = 0, StringView body = {}) noexcept;
  HYLE_IO_API sched::Task<Http2MessageView*> next_response() noexcept;
  HYLE_IO_API sched::Task<bool> reset(
      std::int32_t stream_id,
      Http2Error error = Http2Error::kCancel) noexcept;
  HYLE_IO_API sched::Task<bool> shutdown(
      Http2Error error = Http2Error::kNoError) noexcept;

 private:
  friend sched::Task<std::int32_t> http2_client_submit(
      mem::AllocatorRef, Http2ClientConn&, StringView, StringView, StringView,
      StringView, const Http2Header*, std::size_t, StringView) noexcept;
  friend sched::Task<Http2MessageView*> http2_client_next(
      mem::AllocatorRef, Http2ClientConn&) noexcept;
  friend sched::Task<bool> http2_client_reset(
      mem::AllocatorRef, Http2ClientConn&, std::int32_t, Http2Error) noexcept;
  friend sched::Task<bool> http2_client_shutdown(
      mem::AllocatorRef, Http2ClientConn&, Http2Error) noexcept;
  struct Impl;
  Impl* impl_ = nullptr;
  HttpTransport transport_{};
  Http2MessageView current_{};
  mem::AllocatorRef alloc_{};
};

}  // namespace hyle::io

#endif  // HYLE_IO_HTTP2
