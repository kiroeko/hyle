#include "hyle/io/http.h"

#include <cstring>

#include "picohttpparser.h"

// io-10 HTTP/1.1 integration layer. HYLE owns the io (co_await TcpStream over the
// proactor/fiber) + this integration; protocol PARSING is borrowed from vendored
// picohttpparser (phr_parse_request/response + phr_decode_chunked, a pure zero-IO
// state machine). Reference for the loop structure: seastar src/http/httpd.cc
// (read loop feeds the parser, body source separate from header parse, _done flag
// governs keep-alive) -- see docs scope "HTTP 最终结论".
//
// The co_await layers are FREE coroutines taking mem::Allocator FIRST (a member
// coroutine's promise operator new would bind `this`, not the Allocator -- the trap
// tls.cc/dns.cc/file.cc all hit). The public members forward to them.

namespace hyle::io {
namespace {

// ASCII case-insensitive byte compare (HTTP header names are case-insensitive).
bool iequal(string_ref a, string_ref b) noexcept {
  if (a.size != b.size) return false;
  for (std::uint32_t i = 0; i < a.size; ++i) {
    char ca = a.data[i], cb = b.data[i];
    if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
    if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
    if (ca != cb) return false;
  }
  return true;
}

// Parse a non-negative decimal (Content-Length). Returns false on overflow / a
// non-digit / empty. Bounds the value to guard against absurd lengths.
bool parse_u64(string_ref s, std::uint64_t& out) noexcept {
  if (s.size == 0) return false;
  std::uint64_t v = 0;
  for (std::uint32_t i = 0; i < s.size; ++i) {
    char c = s.data[i];
    if (c < '0' || c > '9') return false;
    if (v > (0xFFFFFFFFFFFFFFFFull - 9) / 10) return false;  // overflow guard
    v = v * 10 + static_cast<std::uint64_t>(c - '0');
  }
  out = v;
  return true;
}

// A small growable byte buffer allocated via mem::Allocator. Holds the read data
// for one connection; persists across keep-alive requests (leftover pipelined
// bytes stay at the front after consume()). Not a general container -- just what
// the HTTP read loop needs (append at tail, consume from front).
struct GrowBuf {
  mem::Allocator alloc;
  char* data = nullptr;
  std::uint32_t size = 0;   // bytes of valid data
  std::uint32_t cap = 0;

  explicit GrowBuf(mem::Allocator a) noexcept : alloc(a) {}
  ~GrowBuf() {
    if (data != nullptr) alloc.deallocate(data, cap, alignof(std::max_align_t));
  }
  GrowBuf(const GrowBuf&) = delete;
  GrowBuf& operator=(const GrowBuf&) = delete;

  // Ensure room for `extra` more bytes past size. Returns false on OOM.
  bool reserve(std::uint32_t want_total) noexcept {
    if (want_total <= cap) return true;
    std::uint32_t ncap = cap == 0 ? 4096 : cap;
    while (ncap < want_total) {
      if (ncap >= 0x80000000u) {
        ncap = want_total;  // next double would overflow u32 -> clamp to exact need
        break;
      }
      ncap *= 2;
    }
    char* np = static_cast<char*>(
        alloc.reallocate(data, cap, ncap, alignof(std::max_align_t)));
    if (np == nullptr) return false;
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

}  // namespace

// ---- HttpServerConn::Impl --------------------------------------------------
// Per-connection state that outlives individual requests (heap, via Allocator):
// the growable read buffer + the parser scratch (header array + the last parsed
// request views). Kept in an Impl so http.h needn't include picohttpparser.
struct HttpServerConn::Impl {
  GrowBuf buf;   // connection read buffer (raw bytes off the socket)
  GrowBuf head;  // stable COPY of the current request's head bytes (views point here)
  GrowBuf body;  // decoded/collected body storage (views point here)
  phr_header headers[kHttpMaxHeaders];
  HttpRequest req;  // reused each next_request(); views point into head/body
  explicit Impl(mem::Allocator a) noexcept : buf(a), head(a), body(a) {}
};

// ---- HttpRequest accessors -------------------------------------------------

string_ref HttpRequest::header_name(std::size_t i) const noexcept {
  if (i >= header_count_) return {};
  const auto& h = headers_[i];
  return string_ref{h.name, static_cast<std::uint32_t>(h.name_len)};
}
string_ref HttpRequest::header_value(std::size_t i) const noexcept {
  if (i >= header_count_) return {};
  const auto& h = headers_[i];
  return string_ref{h.value, static_cast<std::uint32_t>(h.value_len)};
}
string_ref HttpRequest::header(string_ref name) const noexcept {
  for (std::size_t i = 0; i < header_count_; ++i) {
    if (iequal(header_name(i), name)) return header_value(i);
  }
  return {};
}

// ---- HttpResponse ----------------------------------------------------------

void HttpResponse::add_header(string_ref name, string_ref value) noexcept {
  if (h_count_ >= kMaxRespHeaders) return;  // silently cap (skeleton; headers are few)
  h_names_[h_count_] = name;
  h_vals_[h_count_] = value;
  ++h_count_;
}

// ---- HttpResponseView accessors --------------------------------------------

string_ref HttpResponseView::header_name(std::size_t i) const noexcept {
  if (i >= header_count_) return {};
  const auto& h = headers_[i];
  return string_ref{h.name, static_cast<std::uint32_t>(h.name_len)};
}
string_ref HttpResponseView::header_value(std::size_t i) const noexcept {
  if (i >= header_count_) return {};
  const auto& h = headers_[i];
  return string_ref{h.value, static_cast<std::uint32_t>(h.value_len)};
}
string_ref HttpResponseView::header(string_ref name) const noexcept {
  for (std::size_t i = 0; i < header_count_; ++i) {
    if (iequal(header_name(i), name)) return header_value(i);
  }
  return {};
}

// ---- server: read helpers (free coroutines, Allocator-first) ---------------
namespace {

// co_await one read into buf's tail; grow first. Returns bytes read (0 = peer
// closed, <0 = error). Appends to buf.size on success.
sched::Task<std::int64_t> read_more(mem::Allocator, GrowBuf& buf,
                                    TcpStream& s) noexcept {
  if (!buf.reserve(buf.size + 4096)) co_return -1;
  std::int64_t n = co_await s.read(buf.data + buf.size, buf.cap - buf.size);
  if (n > 0) buf.size += static_cast<std::uint32_t>(n);
  co_return n;
}

// Given headers already parsed (header block consumed from the FRONT is the
// caller's job), read the remaining body per framing into buf so that the body
// bytes are contiguous starting at buf.data[0..body_len]. Returns body length, or
// UINT64_MAX on error / too large. `header_len` = bytes of the request head still
// at the front of buf (consumed after we know body size). For Content-Length we
// ensure buf has header_len+CL bytes; for chunked we decode in place.
// NOTE: on success buf's front `header_len` bytes are the head (caller consumes),
// and body occupies buf.data[header_len .. header_len+body_len).

// Read exactly `need` total body bytes (already have `have`) appended after the
// head. Returns false on eof/err.
sched::Task<bool> fill_body(mem::Allocator alloc, GrowBuf& buf, TcpStream& s,
                            std::uint32_t body_off, std::uint64_t need) noexcept {
  while (buf.size - body_off < need) {
    std::int64_t n = co_await read_more(alloc, buf, s);
    if (n <= 0) co_return false;
  }
  co_return true;
}

// Decode a chunked body into `out` (a separate GrowBuf holding the DECODED bytes).
// On entry buf.data[0..buf.size) is raw chunked-encoded bytes read so far (the head
// was already consumed by the caller); more may need reading.
//
// phr_decode_chunked rewrites its input region IN PLACE (stripping chunk framing),
// keeping cross-call state in `dec`. The correct incremental use is to feed it the
// bytes we have, copy out the decoded prefix, and on -2 (incomplete) reset the read
// buffer and read fresh -- the decoder itself remembers where it is. Feeding it the
// same already-compacted bytes twice was the earlier bug (Linux delivered the whole
// body in one read and hid it; Windows IOCP split it across reads and tripped it).
//
// Returns decoded byte count on success (leaving trailing pipelined bytes at the
// front of buf), or UINT64_MAX on error.
sched::Task<std::uint64_t> decode_chunked(mem::Allocator alloc, GrowBuf& buf,
                                          GrowBuf& out, TcpStream& s) noexcept {
  phr_chunked_decoder dec;
  std::memset(&dec, 0, sizeof(dec));
  // Consume the trailer-part + the terminating CRLF of the chunked body. WITHOUT
  // this, phr_decode_chunked Completes right after the last-chunk's "0\r\n" and
  // returns the final "\r\n" (the CRLF after the empty trailer, RFC 7230 §4.1) as
  // "undecoded octets" -- which we would then mistake for the NEXT pipelined
  // message, corrupting the following keep-alive request/response (phr_parse_* then
  // chokes on the spurious leading "\r\n"). With consume_trailer=1 the decoder eats
  // the trailer + final CRLF, so rv is 0 for a lone response and the true byte count
  // only when a real pipelined message actually follows. (Trailers are discarded --
  // HYLE exposes none; better than breaking keep-alive.) Root cause of the
  // multi-chunk keep-alive bug found 2026-07-05.
  dec.consume_trailer = 1;
  for (;;) {
    if (buf.size > 0) {
      std::size_t rsz = buf.size;
      ssize_t rv = phr_decode_chunked(&dec, buf.data, &rsz);
      // buf.data[0..rsz) is now decoded data; append it to out.
      if (rsz > 0) {
        if (!out.reserve(out.size + static_cast<std::uint32_t>(rsz)))
          co_return UINT64_MAX;
        std::memcpy(out.data + out.size, buf.data, rsz);
        out.size += static_cast<std::uint32_t>(rsz);
      }
      if (rv >= 0) {
        // Done. `rv` trailing octets (next pipelined message) sit at buf.data[rsz].
        if (rv > 0)
          std::memmove(buf.data, buf.data + rsz, static_cast<std::size_t>(rv));
        buf.size = static_cast<std::uint32_t>(rv);
        co_return out.size;
      }
      if (rv == -1) co_return UINT64_MAX;  // malformed chunk
      // rv == -2: the decoder consumed everything we gave it; read fresh bytes.
      buf.size = 0;
      if (out.size > kHttpMaxBodyBytes) co_return UINT64_MAX;
    }
    std::int64_t n = co_await read_more(alloc, buf, s);
    if (n <= 0) co_return UINT64_MAX;  // eof mid-body
  }
}

}  // namespace

// ---- HttpServerConn lifetime -----------------------------------------------

HttpServerConn::HttpServerConn(mem::Allocator alloc, TcpStream stream) noexcept
    : stream_(std::move(stream)), alloc_(alloc) {
  impl_ = alloc.allocate_n<Impl>(1);
  if (impl_ != nullptr) new (impl_) Impl(alloc);
}
HttpServerConn::~HttpServerConn() {
  if (impl_ != nullptr) {
    impl_->~Impl();
    alloc_.deallocate_n(impl_, 1);
  }
}
HttpServerConn::HttpServerConn(HttpServerConn&& o) noexcept
    : impl_(std::exchange(o.impl_, nullptr)),
      stream_(std::move(o.stream_)),
      alloc_(o.alloc_) {}
HttpServerConn& HttpServerConn::operator=(HttpServerConn&& o) noexcept {
  if (this != &o) {
    if (impl_ != nullptr) {
      impl_->~Impl();
      alloc_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(o.impl_, nullptr);
    stream_ = std::move(o.stream_);
    alloc_ = o.alloc_;
  }
  return *this;
}

// ---- server: next_request --------------------------------------------------

sched::Task<HttpRequest*> http_server_next(mem::Allocator alloc,
                                           HttpServerConn& c) noexcept {
  HttpServerConn::Impl& im = *c.impl_;
  GrowBuf& buf = im.buf;
  // Leftover pipelined bytes from a previous request are at the front of buf.
  const char* method;
  const char* path;
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
      if (pret > 0) break;                // full head; pret = head byte count
      if (pret == -1) co_return nullptr;  // malformed
      last_len = buf.size;                // pret == -2: incomplete
      if (buf.size >= kHttpMaxHeaderBytes) co_return nullptr;  // DoS guard
    }
    std::int64_t n = co_await read_more(alloc, buf, c.stream_);
    if (n <= 0) co_return nullptr;  // clean close or error before a full request
  }
  const std::uint32_t head_len = static_cast<std::uint32_t>(pret);

  // Copy the head bytes into a STABLE buffer, then RE-PARSE against that copy so all
  // views (method/path/header) point into memory that won't move or be overwritten
  // while the handler runs -- fixes the realloc-dangle + consume-overwrite bugs the
  // red-team found (buf gets consumed/realloc'd during body handling below).
  im.head.size = 0;
  if (!im.head.reserve(head_len)) co_return nullptr;
  std::memcpy(im.head.data, buf.data, head_len);
  im.head.size = head_len;
  num_headers = kHttpMaxHeaders;
  pret = phr_parse_request(im.head.data, head_len, &method, &method_len, &path,
                           &path_len, &minor_version, im.headers, &num_headers, 0);
  if (pret <= 0) co_return nullptr;  // should not happen (parsed once already)

  HttpRequest& r = im.req;
  r = HttpRequest{};
  r.method_ = string_ref{method, static_cast<std::uint32_t>(method_len)};
  r.target_ = string_ref{path, static_cast<std::uint32_t>(path_len)};
  r.minor_version_ = minor_version;
  r.headers_ = im.headers;
  r.header_count_ = num_headers;
  std::uint32_t q = r.target_.find('?');
  if (q == string_ref::npos) {
    r.path_ = r.target_;
  } else {
    r.path_ = r.target_.substr(0, q);
    r.query_ = r.target_.substr(q + 1, r.target_.size - q - 1);
  }
  string_ref conn = r.header("connection");
  if (minor_version >= 1) {
    r.keep_alive_ = !iequal(conn, "close");
  } else {
    r.keep_alive_ = iequal(conn, "keep-alive");
  }

  // Body framing. chunked precedence over Content-Length (RFC). Body goes into
  // im.body (stable) so the body view is also safe against buf moves.
  string_ref te = r.header("transfer-encoding");
  string_ref cl = r.header("content-length");
  // Request-smuggling defense (RFC 7230 §3.3.3). A message that is ambiguous about
  // its body length is a desync/smuggling vector when HYLE sits behind or in front
  // of another HTTP actor that resolves the ambiguity differently. Reject (close
  // the connection -> nullptr) rather than guess:
  //  - BOTH Transfer-Encoding and Content-Length present: the classic TE.CL / CL.TE
  //    smuggling pair. RFC says TE overrides CL, but a CL-honoring hop would frame
  //    it differently -> refuse.
  //  - Transfer-Encoding present but not EXACTLY "chunked" (e.g. "gzip, chunked",
  //    "chunked, chunked", "identity", or an obfuscated "chunked "): we only decode
  //    a single chunked layer; anything else is unframeable here, and lax matching
  //    is exactly how smuggling slips through. Only bare "chunked" is accepted.
  //  - Duplicate Content-Length headers with CONFLICTING values (RFC 7230 §3.3.3
  //    rule: multiple CL with different values is unrecoverable): scan all headers,
  //    reject if two "content-length" values differ.
  if (te.size > 0) {
    if (cl.size > 0) co_return nullptr;          // TE + CL both present -> reject
    if (!iequal(te, "chunked")) co_return nullptr;  // non-bare-chunked TE -> reject
  } else {
    string_ref seen_cl{};
    for (std::size_t i = 0; i < r.header_count_; ++i) {
      if (iequal(r.header_name(i), "content-length")) {
        string_ref v = r.header_value(i);
        if (seen_cl.size > 0 && !(seen_cl == v)) co_return nullptr;  // conflicting
        seen_cl = v;
      }
    }
  }
  buf.consume(head_len);  // head is safely copied; drop it from the read buffer
  im.body.size = 0;
  if (te.size > 0) {  // == chunked (guarded above)
    std::uint64_t dn = co_await decode_chunked(alloc, buf, im.body, c.stream_);
    if (dn == UINT64_MAX) co_return nullptr;
    r.body_ = string_ref{im.body.data, static_cast<std::uint32_t>(dn)};
  } else if (cl.size > 0) {
    std::uint64_t body_len = 0;
    if (!parse_u64(cl, body_len)) co_return nullptr;
    if (body_len > kHttpMaxBodyBytes) co_return nullptr;
    // Pull the body into buf (from index 0 now), then copy it into im.body so the
    // view survives the next request's buf reuse.
    if (!(co_await fill_body(alloc, buf, c.stream_, 0, body_len))) co_return nullptr;
    if (!im.body.reserve(static_cast<std::uint32_t>(body_len))) co_return nullptr;
    std::memcpy(im.body.data, buf.data, body_len);
    im.body.size = static_cast<std::uint32_t>(body_len);
    r.body_ = string_ref{im.body.data, static_cast<std::uint32_t>(body_len)};
    buf.consume(static_cast<std::uint32_t>(body_len));  // leftover pipelined stays
  } else {
    r.body_ = string_ref{};  // no body (GET/DELETE)
  }
  co_return &r;
}

sched::Task<HttpRequest*> HttpServerConn::next_request() noexcept {
  return http_server_next(alloc_, *this);
}

// ---- server: send_response -------------------------------------------------
namespace {

const char* reason_phrase(int code) noexcept {
  switch (code) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "OK";
  }
}

// Append raw bytes / a decimal / a string_ref to a GrowBuf (out framing buffer).
bool put(GrowBuf& b, const char* s, std::uint32_t n) noexcept {
  if (!b.reserve(b.size + n)) return false;
  std::memcpy(b.data + b.size, s, n);
  b.size += n;
  return true;
}
bool put(GrowBuf& b, string_ref s) noexcept { return put(b, s.data, s.size); }
bool put_cstr(GrowBuf& b, const char* s) noexcept {
  return put(b, s, static_cast<std::uint32_t>(std::strlen(s)));
}
bool put_u64(GrowBuf& b, std::uint64_t v) noexcept {
  char tmp[20];
  int i = 20;
  if (v == 0) tmp[--i] = '0';
  while (v > 0) {
    tmp[--i] = static_cast<char>('0' + v % 10);
    v /= 10;
  }
  return put(b, tmp + i, static_cast<std::uint32_t>(20 - i));
}
bool put_hex(GrowBuf& b, std::uint64_t v) noexcept {
  char tmp[16];
  int i = 16;
  const char* hd = "0123456789abcdef";
  if (v == 0) tmp[--i] = '0';
  while (v > 0) {
    tmp[--i] = hd[v & 0xF];
    v >>= 4;
  }
  return put(b, tmp + i, static_cast<std::uint32_t>(16 - i));
}

// co_await-write the whole [p, p+n) to the stream (loop over partial writes).
sched::Task<bool> write_all(mem::Allocator, TcpStream& s, const char* p,
                            std::uint32_t n) noexcept {
  std::uint32_t off = 0;
  while (off < n) {
    std::int64_t w = co_await s.write(p + off, n - off);
    if (w <= 0) co_return false;
    off += static_cast<std::uint32_t>(w);
  }
  co_return true;
}

}  // namespace

sched::Task<bool> http_server_send(mem::Allocator alloc, HttpServerConn& c,
                                   const HttpResponse& resp) noexcept {
  GrowBuf out(alloc);
  // Status line.
  if (!put_cstr(out, "HTTP/1.1 ")) co_return false;
  if (!put_u64(out, static_cast<std::uint64_t>(resp.status()))) co_return false;
  if (!put(out, " ", 1)) co_return false;
  string_ref reason = resp.reason_;
  if (reason.size > 0) {
    if (!put(out, reason)) co_return false;
  } else {
    if (!put_cstr(out, reason_phrase(resp.status()))) co_return false;
  }
  if (!put_cstr(out, "\r\n")) co_return false;
  // Caller headers.
  for (std::size_t i = 0; i < resp.h_count_; ++i) {
    if (!put(out, resp.h_names_[i]) || !put_cstr(out, ": ") ||
        !put(out, resp.h_vals_[i]) || !put_cstr(out, "\r\n"))
      co_return false;
  }
  // Framing header: Content-Length for a fixed body, chunked for a stream.
  if (resp.streaming()) {
    if (!put_cstr(out, "Transfer-Encoding: chunked\r\n")) co_return false;
  } else {
    if (!put_cstr(out, "Content-Length: ")) co_return false;
    if (!put_u64(out, resp.body_.size)) co_return false;
    if (!put_cstr(out, "\r\n")) co_return false;
  }
  if (!put_cstr(out, "\r\n")) co_return false;  // end of headers

  // Send head (+ fixed body appended for the common small-response case).
  if (!resp.streaming()) {
    if (resp.body_.size > 0) {
      if (!put(out, resp.body_)) co_return false;
    }
    co_return co_await write_all(alloc, c.stream_, out.data, out.size);
  }

  // Streaming: send the head, then pump chunks until more==false. Reuse `out`
  // (its head bytes are already written) as the per-chunk frame buffer: reset
  // size=0 each iteration so the capacity grown for the head/first chunk is kept
  // and reused -- no per-chunk allocation (was `GrowBuf frame(alloc)` inside the
  // loop = one alloc+free every chunk, hot for token-stream/SSE responses).
  if (!(co_await write_all(alloc, c.stream_, out.data, out.size))) co_return false;
  for (;;) {
    StreamChunk ch = resp.next_(resp.stream_ctx_);
    if (ch.len > 0) {
      // chunk = <hex len>\r\n<data>\r\n
      out.size = 0;  // reuse buffer + capacity; no realloc unless this chunk is bigger
      if (!put_hex(out, ch.len) || !put_cstr(out, "\r\n") ||
          !put(out, static_cast<const char*>(ch.data), ch.len) ||
          !put_cstr(out, "\r\n"))
        co_return false;
      if (!(co_await write_all(alloc, c.stream_, out.data, out.size)))
        co_return false;
    }
    if (!ch.more) break;
  }
  // Terminating zero-length chunk.
  co_return co_await write_all(alloc, c.stream_, "0\r\n\r\n", 5);
}

sched::Task<bool> HttpServerConn::send_response(const HttpResponse& resp) noexcept {
  return http_server_send(alloc_, *this, resp);
}

// ---- high-level http_serve -------------------------------------------------

sched::Task<void> http_serve(mem::Allocator alloc, TcpStream stream,
                             HttpHandlerFn handler, void* ctx) noexcept {
  HttpServerConn conn{alloc, std::move(stream)};
  if (!conn.valid()) co_return;
  for (;;) {
    HttpRequest* req = co_await conn.next_request();
    if (req == nullptr) break;
    bool keep = req->keep_alive();
    sched::Task<HttpResponse> ht = handler(alloc, ctx, *req);
    HttpResponse resp = co_await ht;
    if (!(co_await conn.send_response(resp))) break;
    if (!keep) break;
  }
  co_return;
}

// ---- HttpClientConn --------------------------------------------------------

struct HttpClientConn::Impl {
  GrowBuf buf;   // connection read buffer
  GrowBuf head;  // stable COPY of the response head (views point here)
  GrowBuf body;  // collected body storage (views point here)
  phr_header headers[kHttpMaxHeaders];
  HttpResponseView resp;
  explicit Impl(mem::Allocator a) noexcept : buf(a), head(a), body(a) {}
};

HttpClientConn::HttpClientConn(mem::Allocator alloc, TcpStream stream) noexcept
    : stream_(std::move(stream)), alloc_(alloc) {
  impl_ = alloc.allocate_n<Impl>(1);
  if (impl_ != nullptr) new (impl_) Impl(alloc);
}
HttpClientConn::~HttpClientConn() {
  if (impl_ != nullptr) {
    impl_->~Impl();
    alloc_.deallocate_n(impl_, 1);
  }
}
HttpClientConn::HttpClientConn(HttpClientConn&& o) noexcept
    : impl_(std::exchange(o.impl_, nullptr)),
      stream_(std::move(o.stream_)),
      alloc_(o.alloc_) {}
HttpClientConn& HttpClientConn::operator=(HttpClientConn&& o) noexcept {
  if (this != &o) {
    if (impl_ != nullptr) {
      impl_->~Impl();
      alloc_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(o.impl_, nullptr);
    stream_ = std::move(o.stream_);
    alloc_ = o.alloc_;
  }
  return *this;
}

sched::Task<HttpResponseView*> http_client_request(mem::Allocator alloc,
                                                   HttpClientConn& c,
                                                   string_ref method,
                                                   string_ref path, string_ref host,
                                                   string_ref body) noexcept {
  HttpClientConn::Impl& im = *c.impl_;
  // Build + send the request. "METHOD path HTTP/1.1\r\nHost: host\r\n
  // Content-Length: N\r\n\r\n" + body.
  {
    GrowBuf out(alloc);
    if (!put(out, method) || !put(out, " ", 1) || !put(out, path) ||
        !put_cstr(out, " HTTP/1.1\r\nHost: ") || !put(out, host) ||
        !put_cstr(out, "\r\n"))
      co_return nullptr;
    if (body.size > 0) {
      if (!put_cstr(out, "Content-Length: ") || !put_u64(out, body.size) ||
          !put_cstr(out, "\r\n"))
        co_return nullptr;
    }
    if (!put_cstr(out, "\r\n")) co_return nullptr;
    if (body.size > 0 && !put(out, body)) co_return nullptr;
    if (!(co_await write_all(alloc, c.stream_, out.data, out.size)))
      co_return nullptr;
  }

  // Read + parse the response. Leftover pipelined bytes stay at the front of buf.
  GrowBuf& buf = im.buf;
  std::uint32_t last_len = 0;
  int minor_version, status;
  const char* msg;
  std::size_t msg_len, num_headers;
  int pret;
  for (;;) {
    if (buf.size > 0) {
      num_headers = kHttpMaxHeaders;
      pret = phr_parse_response(buf.data, buf.size, &minor_version, &status, &msg,
                                &msg_len, im.headers, &num_headers, last_len);
      if (pret > 0) break;
      if (pret == -1) co_return nullptr;
      last_len = buf.size;
      if (buf.size >= kHttpMaxHeaderBytes) co_return nullptr;
    }
    std::int64_t n = co_await read_more(alloc, buf, c.stream_);
    if (n <= 0) co_return nullptr;
  }

  const std::uint32_t head_len = static_cast<std::uint32_t>(pret);
  // Copy the head into a stable buffer and re-parse, so the response views survive
  // the buf realloc/consume during body handling (same fix as the server path).
  im.head.size = 0;
  if (!im.head.reserve(head_len)) co_return nullptr;
  std::memcpy(im.head.data, buf.data, head_len);
  im.head.size = head_len;
  num_headers = kHttpMaxHeaders;
  pret = phr_parse_response(im.head.data, head_len, &minor_version, &status, &msg,
                            &msg_len, im.headers, &num_headers, 0);
  if (pret <= 0) co_return nullptr;

  HttpResponseView& r = im.resp;
  r = HttpResponseView{};
  r.status_ = status;
  r.minor_version_ = minor_version;
  r.reason_ = string_ref{msg, static_cast<std::uint32_t>(msg_len)};
  r.headers_ = im.headers;
  r.header_count_ = num_headers;

  // Body-suppressing responses (RFC 7230 §3.3.3 rule 1): a response to a HEAD
  // request, and any 1xx / 204 / 304 status, has NO message body EVEN IF it carries
  // Content-Length or Transfer-Encoding. Framing off those headers here would make
  // us read a body that never comes -- blocking until the peer closes, or (on a
  // keep-alive conn) consuming the NEXT response's bytes = desync. So short-circuit
  // to an empty body and leave any pipelined bytes in buf for the next response.
  bool no_body = iequal(method, "HEAD") || status == 204 || status == 304 ||
                 (status >= 100 && status < 200);
  if (no_body) {
    buf.consume(head_len);
    im.body.size = 0;
    r.body_ = string_ref{};
    co_return &r;
  }

  // Body framing (same rules as the server request path).
  string_ref te = r.header("transfer-encoding");
  string_ref cl = r.header("content-length");
  buf.consume(head_len);
  im.body.size = 0;
  if (iequal(te, "chunked")) {
    std::uint64_t dn = co_await decode_chunked(alloc, buf, im.body, c.stream_);
    if (dn == UINT64_MAX) co_return nullptr;
    r.body_ = string_ref{im.body.data, static_cast<std::uint32_t>(dn)};
  } else if (cl.size > 0) {
    std::uint64_t body_len = 0;
    if (!parse_u64(cl, body_len)) co_return nullptr;
    if (body_len > kHttpMaxBodyBytes) co_return nullptr;
    if (!(co_await fill_body(alloc, buf, c.stream_, 0, body_len))) co_return nullptr;
    if (!im.body.reserve(static_cast<std::uint32_t>(body_len))) co_return nullptr;
    std::memcpy(im.body.data, buf.data, body_len);
    im.body.size = static_cast<std::uint32_t>(body_len);
    r.body_ = string_ref{im.body.data, static_cast<std::uint32_t>(body_len)};
    buf.consume(static_cast<std::uint32_t>(body_len));
  } else {
    // No Content-Length and not chunked: the body runs until the peer CLOSES the
    // connection (HTTP/1.0, or an HTTP/1.1 "Connection: close" streaming response --
    // RFC 7230 §3.3.3 rule 7). Read until EOF, accumulating into buf; buf already
    // holds whatever leading body bytes arrived with the head. read_more appends and
    // returns 0 on a clean peer close (the body terminator here) or <0 on error. Cap
    // at kHttpMaxBodyBytes so a peer that never closes can't grow us unboundedly.
    // (Previously this returned only the already-buffered bytes -> silent truncation
    // of any close-delimited body larger than the header packet; red-team U3.)
    for (;;) {
      if (buf.size > kHttpMaxBodyBytes) co_return nullptr;  // DoS guard
      std::int64_t n = co_await read_more(alloc, buf, c.stream_);
      if (n == 0) break;             // clean close = end of body
      if (n < 0) co_return nullptr;  // read error before close
    }
    std::uint32_t avail = buf.size;
    if (avail > 0) {
      if (!im.body.reserve(avail)) co_return nullptr;
      std::memcpy(im.body.data, buf.data, avail);
      im.body.size = avail;
    }
    r.body_ = string_ref{im.body.data, avail};
    buf.consume(avail);
  }
  co_return &r;
}

sched::Task<HttpResponseView*> HttpClientConn::request(string_ref method,
                                                       string_ref path,
                                                       string_ref host,
                                                       string_ref body) noexcept {
  return http_client_request(alloc_, *this, method, path, host, body);
}

}  // namespace hyle::io
