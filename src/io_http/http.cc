#include "hyle/io/http.h"

#include <cstring>
#include <span>

#include "hyle/mem/region.h"
#include "picohttpparser.h"

// io-10 HTTP/1.1 integration layer. HYLE owns the io (co_await TcpStream over
// the proactor/fiber) + this integration; protocol PARSING is borrowed from
// vendored picohttpparser (phr_parse_request/response + phr_decode_chunked, a
// pure zero-IO state machine). Reference for the loop structure: seastar
// src/http/httpd.cc (read loop feeds the parser, body source separate from
// header parse, _done flag governs keep-alive) -- see docs scope "HTTP
// 最终结论".
//
// The co_await layers are FREE coroutines taking mem::AllocatorRef FIRST (a
// member coroutine's promise operator new would bind `this`, not the
// AllocatorRef -- the trap tls.cc/dns.cc/file.cc all hit). The public members
// forward to them.

namespace hyle::io {

sched::Task<std::int64_t> http_transport_read(mem::AllocatorRef alloc,
                                              HttpTransport &transport,
                                              void *buffer,
                                              std::uint32_t length) noexcept {
#if defined(HYLE_IO_TLS)
  if (transport.is_tls_) {
    TransferResult result = co_await tls_read(
        alloc, transport.tls_,
        std::span<std::byte>(static_cast<std::byte *>(buffer), length));
    co_return result.ok() ? static_cast<std::int64_t>(result.bytes) : -1;
  }
#else
  (void)alloc;
#endif
  auto result = co_await transport.tcp_.read_some(
      std::span{static_cast<std::byte *>(buffer), length});
  co_return result.ok() ? static_cast<std::int64_t>(result.bytes) : -1;
}

sched::Task<std::int64_t> http_transport_write(mem::AllocatorRef alloc,
                                               HttpTransport &transport,
                                               const void *buffer,
                                               std::uint32_t length) noexcept {
#if defined(HYLE_IO_TLS)
  if (transport.is_tls_) {
    TransferResult result =
        co_await tls_write(alloc, transport.tls_,
                           std::span<const std::byte>(
                               static_cast<const std::byte *>(buffer), length));
    co_return result.ok() ? static_cast<std::int64_t>(result.bytes) : -1;
  }
#else
  (void)alloc;
#endif
  auto result = co_await transport.tcp_.write_some(
      std::span{static_cast<const std::byte *>(buffer), length});
  co_return result.ok() ? static_cast<std::int64_t>(result.bytes) : -1;
}

sched::Task<int> http_transport_shutdown(mem::AllocatorRef alloc,
                                         HttpTransport &transport) noexcept {
#if defined(HYLE_IO_TLS)
  if (transport.is_tls_)
    co_return (co_await tls_shutdown(alloc, transport.tls_)).ok() ? 0 : -1;
#else
  (void)alloc;
#endif
  co_return transport.tcp_.shutdown_write().ok() ? 0 : -1;
}

namespace {

// ASCII case-insensitive byte compare (HTTP header names are case-insensitive).
bool iequal(StringView a, StringView b) noexcept {
  if (a.size() != b.size())
    return false;
  for (std::uint32_t i = 0; i < a.size(); ++i) {
    char ca = a[i], cb = b[i];
    if (ca >= 'A' && ca <= 'Z')
      ca = static_cast<char>(ca - 'A' + 'a');
    if (cb >= 'A' && cb <= 'Z')
      cb = static_cast<char>(cb - 'A' + 'a');
    if (ca != cb)
      return false;
  }
  return true;
}

bool contains_crlf(StringView s) noexcept {
  for (std::uint32_t i = 0; i < s.size(); ++i)
    if (s[i] == '\r' || s[i] == '\n')
      return true;
  return false;
}

bool valid_header_name(StringView s) noexcept {
  if (s.empty())
    return false;
  // RFC token: reject controls, separators, space and non-ASCII. This is
  // stricter than merely blocking CRLF and keeps emitted framing unambiguous.
  static constexpr const char *kSeparators = "()<>@,;:\\\"/[]?={} \t";
  for (std::uint32_t i = 0; i < s.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c <= 0x20 || c >= 0x7f || std::strchr(kSeparators, c) != nullptr)
      return false;
  }
  return true;
}

// Parse a non-negative decimal (Content-Length). Returns false on overflow / a
// non-digit / empty. Bounds the value to guard against absurd lengths.
bool parse_u64(StringView s, std::uint64_t &out) noexcept {
  if (s.empty())
    return false;
  std::uint64_t v = 0;
  for (std::uint32_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c < '0' || c > '9')
      return false;
    if (v > (0xFFFFFFFFFFFFFFFFull - 9) / 10)
      return false; // overflow guard
    v = v * 10 + static_cast<std::uint64_t>(c - '0');
  }
  out = v;
  return true;
}

// A small growable byte buffer allocated via mem::AllocatorRef. Holds the read
// data for one connection; persists across keep-alive requests (leftover
// pipelined bytes stay at the front after consume()). Not a general container
// -- just what the HTTP read loop needs (append at tail, consume from front).
struct GrowBuf {
  mem::AllocatorRef alloc;
  char *data = nullptr;
  std::uint32_t size = 0; // bytes of valid data
  std::uint32_t cap = 0;

  explicit GrowBuf(mem::AllocatorRef a) noexcept : alloc(a) {}
  ~GrowBuf() {
    if (data != nullptr)
      alloc.deallocate(data, cap, alignof(std::max_align_t));
  }
  GrowBuf(const GrowBuf &) = delete;
  GrowBuf &operator=(const GrowBuf &) = delete;

  // Ensure room for `extra` more bytes past size. Returns false on OOM.
  bool reserve(std::uint32_t want_total) noexcept {
    if (want_total <= cap)
      return true;
    std::uint32_t ncap = cap == 0 ? 4096 : cap;
    while (ncap < want_total) {
      if (ncap >= 0x80000000u) {
        ncap =
            want_total; // next double would overflow u32 -> clamp to exact need
        break;
      }
      ncap *= 2;
    }
    char *np = static_cast<char *>(
        alloc.reallocate(data, cap, ncap, alignof(std::max_align_t)));
    if (np == nullptr)
      return false;
    data = np;
    cap = ncap;
    return true;
  }

  // Drop the first `n` bytes (shift the tail down). Keeps unconsumed pipelined
  // bytes at the front for the next request.
  void consume(std::uint32_t n) noexcept {
    if (n >= size) {
      size = 0;
      return;
    }
    std::memmove(data, data + n, size - n);
    size -= n;
  }
};

} // namespace

// ---- HttpServerConn::Impl --------------------------------------------------
// Per-connection state that outlives individual requests (heap, via
// AllocatorRef): the growable read buffer + the parser scratch (header array +
// the last parsed request views). Kept in an Impl so http.h needn't include
// picohttpparser.
struct HttpServerConn::Impl {
  GrowBuf buf;  // connection read buffer (raw bytes off the socket)
  GrowBuf head; // stable COPY of the current request's head bytes (views point
                // here)
  GrowBuf body; // decoded/collected body storage (views point here)
  phr_header headers[kHttpMaxHeaders];
  HttpRequest req; // reused each next_request(); views point into head/body
  explicit Impl(mem::AllocatorRef a) noexcept : buf(a), head(a), body(a) {}
};

// ---- HttpRequest accessors -------------------------------------------------

StringView HttpRequest::header_name(std::size_t i) const noexcept {
  if (i >= header_count_)
    return {};
  const auto &h = headers_[i];
  return StringView{h.name, static_cast<std::uint32_t>(h.name_len)};
}
StringView HttpRequest::header_value(std::size_t i) const noexcept {
  if (i >= header_count_)
    return {};
  const auto &h = headers_[i];
  return StringView{h.value, static_cast<std::uint32_t>(h.value_len)};
}
StringView HttpRequest::header(StringView name) const noexcept {
  for (std::size_t i = 0; i < header_count_; ++i) {
    if (iequal(header_name(i), name))
      return header_value(i);
  }
  return {};
}

// ---- HttpResponse ----------------------------------------------------------

bool HttpResponse::add_header(StringView name, StringView value) noexcept {
  if (h_count_ >= kMaxRespHeaders)
    return false;
  h_names_[h_count_] = name;
  h_vals_[h_count_] = value;
  ++h_count_;
  return true;
}

// ---- HttpResponseView accessors --------------------------------------------

StringView HttpResponseView::header_name(std::size_t i) const noexcept {
  if (i >= header_count_)
    return {};
  const auto &h = headers_[i];
  return StringView{h.name, static_cast<std::uint32_t>(h.name_len)};
}
StringView HttpResponseView::header_value(std::size_t i) const noexcept {
  if (i >= header_count_)
    return {};
  const auto &h = headers_[i];
  return StringView{h.value, static_cast<std::uint32_t>(h.value_len)};
}
StringView HttpResponseView::header(StringView name) const noexcept {
  for (std::size_t i = 0; i < header_count_; ++i) {
    if (iequal(header_name(i), name))
      return header_value(i);
  }
  return {};
}

// ---- server: read helpers (free coroutines, AllocatorRef-first)
// ---------------
namespace {

// co_await one read into buf's tail; grow first. Returns bytes read (0 = peer
// closed, <0 = error). Appends to buf.size on success.
sched::Task<std::int64_t> read_more(mem::AllocatorRef alloc, GrowBuf &buf,
                                    HttpTransport &transport) noexcept {
  if (!buf.reserve(buf.size + 4096))
    co_return -1;
  std::int64_t n = co_await http_transport_read(
      alloc, transport, buf.data + buf.size, buf.cap - buf.size);
  if (n > 0)
    buf.size += static_cast<std::uint32_t>(n);
  co_return n;
}

// Given headers already parsed (header block consumed from the FRONT is the
// caller's job), read the remaining body per framing into buf so that the body
// bytes are contiguous starting at buf.data[0..body_len]. Returns body length,
// or UINT64_MAX on error / too large. `header_len` = bytes of the request head
// still at the front of buf (consumed after we know body size). For
// Content-Length we ensure buf has header_len+CL bytes; for chunked we decode
// in place. NOTE: on success buf's front `header_len` bytes are the head
// (caller consumes), and body occupies buf.data[header_len ..
// header_len+body_len).

// Read exactly `need` total body bytes (already have `have`) appended after the
// head. Returns false on eof/err.
sched::Task<bool> fill_body(mem::AllocatorRef alloc, GrowBuf &buf,
                            HttpTransport &transport, std::uint32_t body_off,
                            std::uint64_t need) noexcept {
  while (buf.size - body_off < need) {
    std::int64_t n = co_await read_more(alloc, buf, transport);
    if (n <= 0)
      co_return false;
  }
  co_return true;
}

// Decode a chunked body into `out` (a separate GrowBuf holding the DECODED
// bytes). On entry buf.data[0..buf.size) is raw chunked-encoded bytes read so
// far (the head was already consumed by the caller); more may need reading.
//
// phr_decode_chunked rewrites its input region IN PLACE (stripping chunk
// framing), keeping cross-call state in `dec`. The correct incremental use is
// to feed it the bytes we have, copy out the decoded prefix, and on -2
// (incomplete) reset the read buffer and read fresh -- the decoder itself
// remembers where it is. Feeding it the same already-compacted bytes twice was
// the earlier bug (Linux delivered the whole body in one read and hid it;
// Windows IOCP split it across reads and tripped it).
//
// Returns decoded byte count on success (leaving trailing pipelined bytes at
// the front of buf), or UINT64_MAX on error.
sched::Task<std::uint64_t> decode_chunked(mem::AllocatorRef alloc, GrowBuf &buf,
                                          GrowBuf &out,
                                          HttpTransport &transport) noexcept {
  phr_chunked_decoder dec;
  std::memset(&dec, 0, sizeof(dec));
  // Consume the trailer-part + the terminating CRLF of the chunked body.
  // WITHOUT this, phr_decode_chunked Completes right after the last-chunk's
  // "0\r\n" and returns the final "\r\n" (the CRLF after the empty trailer, RFC
  // 7230 §4.1) as "undecoded octets" -- which we would then mistake for the
  // NEXT pipelined message, corrupting the following keep-alive
  // request/response (phr_parse_* then chokes on the spurious leading "\r\n").
  // With consume_trailer=1 the decoder eats the trailer + final CRLF, so rv is
  // 0 for a lone response and the true byte count only when a real pipelined
  // message actually follows. (Trailers are discarded -- HYLE exposes none;
  // better than breaking keep-alive.) Root cause of the multi-chunk keep-alive
  // bug found 2026-07-05.
  dec.consume_trailer = 1;
  for (;;) {
    if (buf.size > 0) {
      std::size_t rsz = buf.size;
      ssize_t rv = phr_decode_chunked(&dec, buf.data, &rsz);
      // buf.data[0..rsz) is now decoded data; append it to out.
      if (rsz > 0) {
        // Apply the body cap BEFORE narrowing/adding. In particular, a complete
        // final chunk returns rv>=0 below, so a check only on the "need more"
        // branch lets an oversized body bypass the cap in one packet.
        if (rsz > kHttpMaxBodyBytes - out.size || rsz > UINT32_MAX - out.size)
          co_return UINT64_MAX;
        if (!out.reserve(out.size + static_cast<std::uint32_t>(rsz)))
          co_return UINT64_MAX;
        std::memcpy(out.data + out.size, buf.data, rsz);
        out.size += static_cast<std::uint32_t>(rsz);
      }
      if (rv >= 0) {
        // Done. `rv` trailing octets (next pipelined message) sit at
        // buf.data[rsz].
        if (rv > 0)
          std::memmove(buf.data, buf.data + rsz, static_cast<std::size_t>(rv));
        buf.size = static_cast<std::uint32_t>(rv);
        co_return out.size;
      }
      if (rv == -1)
        co_return UINT64_MAX; // malformed chunk
      // rv == -2: the decoder consumed everything we gave it; read fresh bytes.
      buf.size = 0;
      if (out.size > kHttpMaxBodyBytes)
        co_return UINT64_MAX;
    }
    std::int64_t n = co_await read_more(alloc, buf, transport);
    if (n <= 0)
      co_return UINT64_MAX; // eof mid-body
  }
}

} // namespace

// ---- HttpServerConn lifetime -----------------------------------------------

HttpServerConn::HttpServerConn(mem::AllocatorRef alloc,
                               TcpStream stream) noexcept
    : transport_(HttpTransport{std::move(stream)}), alloc_(alloc) {
  impl_ = alloc.allocate_n<Impl>(1);
  if (impl_ != nullptr)
    new (impl_) Impl(alloc);
}
#if defined(HYLE_IO_TLS)
HttpServerConn::HttpServerConn(mem::AllocatorRef alloc,
                               TlsStream stream) noexcept
    : transport_(HttpTransport{std::move(stream)}), alloc_(alloc) {
  impl_ = alloc.allocate_n<Impl>(1);
  if (impl_ != nullptr)
    new (impl_) Impl(alloc);
}
#endif
HttpServerConn::~HttpServerConn() {
  if (impl_ != nullptr) {
    impl_->~Impl();
    alloc_.deallocate_n(impl_, 1);
  }
}
HttpServerConn::HttpServerConn(HttpServerConn &&o) noexcept
    : impl_(std::exchange(o.impl_, nullptr)),
      transport_(std::move(o.transport_)), alloc_(o.alloc_) {}
HttpServerConn &HttpServerConn::operator=(HttpServerConn &&o) noexcept {
  if (this != &o) {
    if (impl_ != nullptr) {
      impl_->~Impl();
      alloc_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(o.impl_, nullptr);
    transport_ = std::move(o.transport_);
    alloc_ = o.alloc_;
  }
  return *this;
}

// ---- server: next_request --------------------------------------------------

sched::Task<HttpRequest *> http_server_next(mem::AllocatorRef alloc,
                                            HttpServerConn &c) noexcept {
  HttpServerConn::Impl &im = *c.impl_;
  GrowBuf &buf = im.buf;
  // Leftover pipelined bytes from a previous request are at the front of buf.
  const char *method;
  const char *path;
  std::size_t method_len, path_len, num_headers;
  int minor_version;
  int pret;
  // Phase 1: read until the request HEAD is complete.
  std::uint32_t last_len = 0;
  for (;;) {
    if (buf.size > 0) {
      num_headers = kHttpMaxHeaders;
      pret = phr_parse_request(buf.data, buf.size, &method, &method_len, &path,
                               &path_len, &minor_version, im.headers,
                               &num_headers, last_len);
      if (pret > 0)
        break; // full head; pret = head byte count
      if (pret == -1)
        co_return nullptr; // malformed
      last_len = buf.size; // pret == -2: incomplete
      if (buf.size >= kHttpMaxHeaderBytes)
        co_return nullptr; // DoS guard
    }
    std::int64_t n = co_await read_more(alloc, buf, c.transport_);
    if (n <= 0)
      co_return nullptr; // clean close or error before a full request
  }
  const std::uint32_t head_len = static_cast<std::uint32_t>(pret);

  // Copy the head bytes into stable storage, then relocate parser views by
  // offset. Re-parsing the same head here used to double the parser cost on
  // every message.
  im.head.size = 0;
  if (!im.head.reserve(head_len))
    co_return nullptr;
  std::memcpy(im.head.data, buf.data, head_len);
  im.head.size = head_len;
  method = im.head.data + (method - buf.data);
  path = im.head.data + (path - buf.data);
  for (std::size_t i = 0; i < num_headers; ++i) {
    im.headers[i].name = im.head.data + (im.headers[i].name - buf.data);
    im.headers[i].value = im.head.data + (im.headers[i].value - buf.data);
  }

  HttpRequest &r = im.req;
  r = HttpRequest{};
  r.method_ = StringView{method, static_cast<std::uint32_t>(method_len)};
  r.target_ = StringView{path, static_cast<std::uint32_t>(path_len)};
  r.minor_version_ = minor_version;
  r.headers_ = im.headers;
  r.header_count_ = num_headers;
  std::uint32_t q = r.target_.find('?');
  if (q == StringView::npos) {
    r.path_ = r.target_;
  } else {
    r.path_ = r.target_.substr(0, q);
    r.query_ = r.target_.substr(q + 1, r.target_.size() - q - 1);
  }
  StringView conn = r.header("connection");
  if (minor_version >= 1) {
    r.keep_alive_ = !iequal(conn, "close");
  } else {
    r.keep_alive_ = iequal(conn, "keep-alive");
  }

  // Body framing. chunked precedence over Content-Length (RFC). Body goes into
  // im.body (stable) so the body view is also safe against buf moves.
  StringView te = r.header("transfer-encoding");
  StringView cl = r.header("content-length");
  // Request-smuggling defense (RFC 7230 §3.3.3). A message that is ambiguous
  // about its body length is a desync/smuggling vector when HYLE sits behind or
  // in front of another HTTP actor that resolves the ambiguity differently.
  // Reject (close the connection -> nullptr) rather than guess:
  //  - BOTH Transfer-Encoding and Content-Length present: the classic TE.CL /
  //  CL.TE
  //    smuggling pair. RFC says TE overrides CL, but a CL-honoring hop would
  //    frame it differently -> refuse.
  //  - Transfer-Encoding present but not EXACTLY "chunked" (e.g. "gzip,
  //  chunked",
  //    "chunked, chunked", "identity", or an obfuscated "chunked "): we only
  //    decode a single chunked layer; anything else is unframeable here, and
  //    lax matching is exactly how smuggling slips through. Only bare "chunked"
  //    is accepted.
  //  - Duplicate Content-Length headers with CONFLICTING values (RFC 7230
  //  §3.3.3
  //    rule: multiple CL with different values is unrecoverable): scan all
  //    headers, reject if two "content-length" values differ.
  if (!te.empty()) {
    if (!cl.empty())
      co_return nullptr; // TE + CL both present -> reject
    if (!iequal(te, "chunked"))
      co_return nullptr; // non-bare-chunked TE -> reject
  } else {
    StringView seen_cl{};
    for (std::size_t i = 0; i < r.header_count_; ++i) {
      if (iequal(r.header_name(i), "content-length")) {
        StringView v = r.header_value(i);
        if (!seen_cl.empty() && !(seen_cl == v))
          co_return nullptr; // conflicting
        seen_cl = v;
      }
    }
  }
  buf.consume(head_len); // head is safely copied; drop it from the read buffer
  im.body.size = 0;
  if (!te.empty()) { // == chunked (guarded above)
    std::uint64_t dn =
        co_await decode_chunked(alloc, buf, im.body, c.transport_);
    if (dn == UINT64_MAX)
      co_return nullptr;
    r.body_ = StringView{im.body.data, static_cast<std::uint32_t>(dn)};
  } else if (!cl.empty()) {
    std::uint64_t body_len = 0;
    if (!parse_u64(cl, body_len))
      co_return nullptr;
    if (body_len > kHttpMaxBodyBytes)
      co_return nullptr;
    // Pull the body into buf (from index 0 now), then copy it into im.body so
    // the view survives the next request's buf reuse.
    if (!(co_await fill_body(alloc, buf, c.transport_, 0, body_len)))
      co_return nullptr;
    if (!im.body.reserve(static_cast<std::uint32_t>(body_len)))
      co_return nullptr;
    std::memcpy(im.body.data, buf.data, body_len);
    im.body.size = static_cast<std::uint32_t>(body_len);
    r.body_ = StringView{im.body.data, static_cast<std::uint32_t>(body_len)};
    buf.consume(
        static_cast<std::uint32_t>(body_len)); // leftover pipelined stays
  } else {
    r.body_ = StringView{}; // no body (GET/DELETE)
  }
  co_return &r;
}

sched::Task<HttpRequest *> HttpServerConn::next_request() noexcept {
  return http_server_next(alloc_, *this);
}

// ---- server: send_response -------------------------------------------------
namespace {

const char *reason_phrase(int code) noexcept {
  switch (code) {
  case 200:
    return "OK";
  case 201:
    return "Created";
  case 204:
    return "No Content";
  case 400:
    return "Bad Request";
  case 401:
    return "Unauthorized";
  case 403:
    return "Forbidden";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 413:
    return "Payload Too Large";
  case 429:
    return "Too Many Requests";
  case 500:
    return "Internal Server Error";
  case 503:
    return "Service Unavailable";
  default:
    return "OK";
  }
}

// Append raw bytes / a decimal / a StringView to a GrowBuf (out framing
// buffer).
bool put(GrowBuf &b, const char *s, std::uint32_t n) noexcept {
  if (!b.reserve(b.size + n))
    return false;
  std::memcpy(b.data + b.size, s, n);
  b.size += n;
  return true;
}
bool put(GrowBuf &b, StringView s) noexcept {
  return put(b, s.data(), s.size());
}
bool put_cstr(GrowBuf &b, const char *s) noexcept {
  return put(b, s, static_cast<std::uint32_t>(std::strlen(s)));
}
bool put_u64(GrowBuf &b, std::uint64_t v) noexcept {
  char tmp[20];
  int i = 20;
  if (v == 0)
    tmp[--i] = '0';
  while (v > 0) {
    tmp[--i] = static_cast<char>('0' + v % 10);
    v /= 10;
  }
  return put(b, tmp + i, static_cast<std::uint32_t>(20 - i));
}
bool put_hex(GrowBuf &b, std::uint64_t v) noexcept {
  char tmp[16];
  int i = 16;
  const char *hd = "0123456789abcdef";
  if (v == 0)
    tmp[--i] = '0';
  while (v > 0) {
    tmp[--i] = hd[v & 0xF];
    v >>= 4;
  }
  return put(b, tmp + i, static_cast<std::uint32_t>(16 - i));
}

// co_await-write the whole [p, p+n) to the stream (loop over partial writes).
sched::Task<bool> write_all(mem::AllocatorRef alloc, HttpTransport &transport,
                            const char *p, std::uint32_t n) noexcept {
  std::uint32_t off = 0;
  while (off < n) {
    std::int64_t w =
        co_await http_transport_write(alloc, transport, p + off, n - off);
    if (w <= 0)
      co_return false;
    off += static_cast<std::uint32_t>(w);
  }
  co_return true;
}

} // namespace

sched::Task<bool> http_server_send(mem::AllocatorRef alloc, HttpServerConn &c,
                                   const HttpResponse &resp) noexcept {
  if (resp.status() < 100 || resp.status() > 999 || contains_crlf(resp.reason_))
    co_return false;
  if (resp.streaming() && resp.next_ == nullptr)
    co_return false;
  for (std::size_t i = 0; i < resp.h_count_; ++i) {
    if (!valid_header_name(resp.h_names_[i]) || contains_crlf(resp.h_vals_[i]))
      co_return false;
    // Framing is owned by this function. Accepting caller-supplied framing and
    // then appending our own CL/TE creates ambiguous duplicate headers.
    if (iequal(resp.h_names_[i], "content-length") ||
        iequal(resp.h_names_[i], "transfer-encoding"))
      co_return false;
  }
  GrowBuf out(alloc);
  // Status line.
  if (!put_cstr(out, "HTTP/1.1 "))
    co_return false;
  if (!put_u64(out, static_cast<std::uint64_t>(resp.status())))
    co_return false;
  if (!put(out, " ", 1))
    co_return false;
  StringView reason = resp.reason_;
  if (!reason.empty()) {
    if (!put(out, reason))
      co_return false;
  } else {
    if (!put_cstr(out, reason_phrase(resp.status())))
      co_return false;
  }
  if (!put_cstr(out, "\r\n"))
    co_return false;
  // Caller headers.
  for (std::size_t i = 0; i < resp.h_count_; ++i) {
    if (!put(out, resp.h_names_[i]) || !put_cstr(out, ": ") ||
        !put(out, resp.h_vals_[i]) || !put_cstr(out, "\r\n"))
      co_return false;
  }
  // Framing header: Content-Length for a fixed body, chunked for a stream.
  if (resp.streaming()) {
    if (!put_cstr(out, "Transfer-Encoding: chunked\r\n"))
      co_return false;
  } else {
    if (!put_cstr(out, "Content-Length: "))
      co_return false;
    if (!put_u64(out, resp.body_.size()))
      co_return false;
    if (!put_cstr(out, "\r\n"))
      co_return false;
  }
  if (!put_cstr(out, "\r\n"))
    co_return false; // end of headers

  // Send head (+ fixed body appended for the common small-response case).
  if (!resp.streaming()) {
    if (!resp.body_.empty()) {
      if (!put(out, resp.body_))
        co_return false;
    }
    co_return co_await write_all(alloc, c.transport_, out.data, out.size);
  }

  // Streaming: send the head, then pump chunks until more==false. Reuse `out`
  // (its head bytes are already written) as the per-chunk frame buffer: reset
  // size=0 each iteration so the capacity grown for the head/first chunk is
  // kept and reused -- no per-chunk allocation (was `GrowBuf frame(alloc)`
  // inside the loop = one alloc+free every chunk, hot for token-stream/SSE
  // responses).
  if (!(co_await write_all(alloc, c.transport_, out.data, out.size)))
    co_return false;
  for (;;) {
    StreamChunk ch = resp.next_(resp.stream_ctx_);
    if (ch.len > 0) {
      // chunk = <hex len>\r\n<data>\r\n
      out.size =
          0; // reuse buffer + capacity; no realloc unless this chunk is bigger
      if (!put_hex(out, ch.len) || !put_cstr(out, "\r\n") ||
          !put(out, static_cast<const char *>(ch.data), ch.len) ||
          !put_cstr(out, "\r\n"))
        co_return false;
      if (!(co_await write_all(alloc, c.transport_, out.data, out.size)))
        co_return false;
    }
    if (!ch.more)
      break;
  }
  // Terminating zero-length chunk.
  co_return co_await write_all(alloc, c.transport_, "0\r\n\r\n", 5);
}

sched::Task<bool>
HttpServerConn::send_response(const HttpResponse &resp) noexcept {
  return http_server_send(alloc_, *this, resp);
}

// ---- high-level http_serve -------------------------------------------------

namespace {

template <class Stream>
sched::Task<void> http_serve_impl(mem::AllocatorRef alloc, Stream stream,
                                  HttpHandlerFn handler, void *ctx,
                                  HttpServeOptions options) noexcept {
  HttpServerConn conn{alloc, std::move(stream)};
  if (!conn.valid())
    co_return;
  // Connection parsing/storage stays on the long-lived backing allocator. Only
  // handler-owned, request-lifetime work uses this region. Retain ordinary
  // blocks across keep-alive requests so a hot connection pays the backing
  // allocation cost once; clear after the response and handler frame are both
  // gone.
  mem::Arena request_region{alloc, options.request_region};
  for (;;) {
    HttpRequest *req = co_await conn.next_request();
    if (req == nullptr)
      break;
    const bool keep = req->keep_alive();
    bool sent = false;
    {
      // Destruction order is part of the contract: response, then handler Task
      // (whose coroutine frame may own response views), then Arena::clear().
      // This keeps body/header views and streaming state alive through the
      // send.
      mem::AllocatorRef request_alloc = request_region.as_allocator();
      sched::Task<HttpResponse> task = handler(request_alloc, ctx, *req);
      if (!task.valid())
        break;
      HttpResponse response = co_await task;
      sent = co_await conn.send_response(response);
    }
    request_region.clear();
    if (request_region.reserved() > options.request_region_retain_limit)
      request_region.release();
    if (!sent || !keep)
      break;
  }
  co_return;
}

} // namespace

sched::Task<void> http_serve(mem::AllocatorRef alloc, TcpStream stream,
                             HttpHandlerFn handler, void *ctx,
                             HttpServeOptions options) noexcept {
  return http_serve_impl(alloc, std::move(stream), handler, ctx, options);
}

#if defined(HYLE_IO_TLS)
sched::Task<void> http_serve(mem::AllocatorRef alloc, TlsStream stream,
                             HttpHandlerFn handler, void *ctx,
                             HttpServeOptions options) noexcept {
  return http_serve_impl(alloc, std::move(stream), handler, ctx, options);
}
#endif

// ---- HttpClientConn --------------------------------------------------------

struct HttpClientConn::Impl {
  GrowBuf buf;  // connection read buffer
  GrowBuf head; // stable COPY of the response head (views point here)
  GrowBuf body; // collected body storage (views point here)
  phr_header headers[kHttpMaxHeaders];
  HttpResponseView resp;
  explicit Impl(mem::AllocatorRef a) noexcept : buf(a), head(a), body(a) {}
};

HttpClientConn::HttpClientConn(mem::AllocatorRef alloc,
                               TcpStream stream) noexcept
    : transport_(HttpTransport{std::move(stream)}), alloc_(alloc) {
  impl_ = alloc.allocate_n<Impl>(1);
  if (impl_ != nullptr)
    new (impl_) Impl(alloc);
}
#if defined(HYLE_IO_TLS)
HttpClientConn::HttpClientConn(mem::AllocatorRef alloc,
                               TlsStream stream) noexcept
    : transport_(HttpTransport{std::move(stream)}), alloc_(alloc) {
  impl_ = alloc.allocate_n<Impl>(1);
  if (impl_ != nullptr)
    new (impl_) Impl(alloc);
}
#endif
HttpClientConn::~HttpClientConn() {
  if (impl_ != nullptr) {
    impl_->~Impl();
    alloc_.deallocate_n(impl_, 1);
  }
}
HttpClientConn::HttpClientConn(HttpClientConn &&o) noexcept
    : impl_(std::exchange(o.impl_, nullptr)),
      transport_(std::move(o.transport_)), alloc_(o.alloc_) {}
HttpClientConn &HttpClientConn::operator=(HttpClientConn &&o) noexcept {
  if (this != &o) {
    if (impl_ != nullptr) {
      impl_->~Impl();
      alloc_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(o.impl_, nullptr);
    transport_ = std::move(o.transport_);
    alloc_ = o.alloc_;
  }
  return *this;
}

sched::Task<HttpResponseView *>
http_client_request(mem::AllocatorRef alloc, HttpClientConn &c,
                    StringView method, StringView path, StringView host,
                    StringView body) noexcept {
  // These fields are serialized directly into the request line/Host header.
  // Reject delimiter injection instead of emitting a second attacker-controlled
  // header or request on the keep-alive connection.
  if (method.empty() || path.empty() || host.empty() || contains_crlf(method) ||
      contains_crlf(path) || contains_crlf(host)) {
    co_return nullptr;
  }
  if (!valid_header_name(method))
    co_return nullptr;
  for (std::uint32_t i = 0; i < path.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(path[i]);
    if (ch <= 0x20 || ch == 0x7f)
      co_return nullptr;
  }
  for (std::uint32_t i = 0; i < host.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(host[i]);
    if (ch <= 0x20 || ch == 0x7f)
      co_return nullptr;
  }
  HttpClientConn::Impl &im = *c.impl_;
  // Build + send the request. "METHOD path HTTP/1.1\r\nHost: host\r\n
  // Content-Length: N\r\n\r\n" + body.
  {
    GrowBuf out(alloc);
    if (!put(out, method) || !put(out, " ", 1) || !put(out, path) ||
        !put_cstr(out, " HTTP/1.1\r\nHost: ") || !put(out, host) ||
        !put_cstr(out, "\r\n"))
      co_return nullptr;
    if (!body.empty()) {
      if (!put_cstr(out, "Content-Length: ") || !put_u64(out, body.size()) ||
          !put_cstr(out, "\r\n"))
        co_return nullptr;
    }
    if (!put_cstr(out, "\r\n"))
      co_return nullptr;
    if (!body.empty() && !put(out, body))
      co_return nullptr;
    if (!(co_await write_all(alloc, c.transport_, out.data, out.size)))
      co_return nullptr;
  }

  // Read + parse the response. Leftover pipelined bytes stay at the front of
  // buf.
  GrowBuf &buf = im.buf;
  std::uint32_t last_len = 0;
  int minor_version, status;
  const char *msg;
  std::size_t msg_len, num_headers;
  int pret;
  for (;;) {
    if (buf.size > 0) {
      num_headers = kHttpMaxHeaders;
      pret =
          phr_parse_response(buf.data, buf.size, &minor_version, &status, &msg,
                             &msg_len, im.headers, &num_headers, last_len);
      if (pret > 0)
        break;
      if (pret == -1)
        co_return nullptr;
      last_len = buf.size;
      if (buf.size >= kHttpMaxHeaderBytes)
        co_return nullptr;
    }
    std::int64_t n = co_await read_more(alloc, buf, c.transport_);
    if (n <= 0)
      co_return nullptr;
  }

  const std::uint32_t head_len = static_cast<std::uint32_t>(pret);
  // Copy the head into stable storage and relocate parser views. This preserves
  // response views across buf reuse without a second parse.
  im.head.size = 0;
  if (!im.head.reserve(head_len))
    co_return nullptr;
  std::memcpy(im.head.data, buf.data, head_len);
  im.head.size = head_len;
  msg = im.head.data + (msg - buf.data);
  for (std::size_t i = 0; i < num_headers; ++i) {
    im.headers[i].name = im.head.data + (im.headers[i].name - buf.data);
    im.headers[i].value = im.head.data + (im.headers[i].value - buf.data);
  }

  HttpResponseView &r = im.resp;
  r = HttpResponseView{};
  r.status_ = status;
  r.minor_version_ = minor_version;
  r.reason_ = StringView{msg, static_cast<std::uint32_t>(msg_len)};
  r.headers_ = im.headers;
  r.header_count_ = num_headers;

  // Body-suppressing responses (RFC 7230 §3.3.3 rule 1): a response to a HEAD
  // request, and any 1xx / 204 / 304 status, has NO message body EVEN IF it
  // carries Content-Length or Transfer-Encoding. Framing off those headers here
  // would make us read a body that never comes -- blocking until the peer
  // closes, or (on a keep-alive conn) consuming the NEXT response's bytes =
  // desync. So short-circuit to an empty body and leave any pipelined bytes in
  // buf for the next response.
  bool no_body = iequal(method, "HEAD") || status == 204 || status == 304 ||
                 (status >= 100 && status < 200);
  if (no_body) {
    buf.consume(head_len);
    im.body.size = 0;
    r.body_ = StringView{};
    co_return &r;
  }

  // Body framing (same rules as the server request path).
  StringView te = r.header("transfer-encoding");
  StringView cl = r.header("content-length");
  // Apply the same anti-smuggling framing policy on responses as requests.
  // A client that accepts TE+CL, an unsupported transfer coding, or conflicting
  // duplicate Content-Length values can desynchronize its keep-alive stream
  // from an intermediary that resolves the ambiguity differently.
  if (!te.empty()) {
    if (!cl.empty())
      co_return nullptr;
    if (!iequal(te, "chunked"))
      co_return nullptr;
  } else {
    StringView seen_cl{};
    for (std::size_t i = 0; i < r.header_count_; ++i) {
      if (!iequal(r.header_name(i), "content-length"))
        continue;
      StringView v = r.header_value(i);
      if (!seen_cl.empty() && !(seen_cl == v))
        co_return nullptr;
      seen_cl = v;
    }
  }
  buf.consume(head_len);
  im.body.size = 0;
  if (!te.empty()) { // exactly bare "chunked" (validated above)
    std::uint64_t dn =
        co_await decode_chunked(alloc, buf, im.body, c.transport_);
    if (dn == UINT64_MAX)
      co_return nullptr;
    r.body_ = StringView{im.body.data, static_cast<std::uint32_t>(dn)};
  } else if (!cl.empty()) {
    std::uint64_t body_len = 0;
    if (!parse_u64(cl, body_len))
      co_return nullptr;
    if (body_len > kHttpMaxBodyBytes)
      co_return nullptr;
    if (!(co_await fill_body(alloc, buf, c.transport_, 0, body_len)))
      co_return nullptr;
    if (!im.body.reserve(static_cast<std::uint32_t>(body_len)))
      co_return nullptr;
    std::memcpy(im.body.data, buf.data, body_len);
    im.body.size = static_cast<std::uint32_t>(body_len);
    r.body_ = StringView{im.body.data, static_cast<std::uint32_t>(body_len)};
    buf.consume(static_cast<std::uint32_t>(body_len));
  } else {
    // No Content-Length and not chunked: the body runs until the peer CLOSES
    // the connection (HTTP/1.0, or an HTTP/1.1 "Connection: close" streaming
    // response -- RFC 7230 §3.3.3 rule 7). Read until EOF, accumulating into
    // buf; buf already holds whatever leading body bytes arrived with the head.
    // read_more appends and returns 0 on a clean peer close (the body
    // terminator here) or <0 on error. Cap at kHttpMaxBodyBytes so a peer that
    // never closes can't grow us unboundedly. (Previously this returned only
    // the already-buffered bytes -> silent truncation of any close-delimited
    // body larger than the header packet; red-team U3.)
    for (;;) {
      if (buf.size > kHttpMaxBodyBytes)
        co_return nullptr; // DoS guard
      std::int64_t n = co_await read_more(alloc, buf, c.transport_);
      if (n == 0)
        break; // clean close = end of body
      if (n < 0)
        co_return nullptr; // read error before close
    }
    std::uint32_t avail = buf.size;
    if (avail > 0) {
      if (!im.body.reserve(avail))
        co_return nullptr;
      std::memcpy(im.body.data, buf.data, avail);
      im.body.size = avail;
    }
    r.body_ = StringView{im.body.data, avail};
    buf.consume(avail);
  }
  co_return &r;
}

sched::Task<HttpResponseView *>
HttpClientConn::request(StringView method, StringView path, StringView host,
                        StringView body) noexcept {
  return http_client_request(alloc_, *this, method, path, host, body);
}

} // namespace hyle::io
