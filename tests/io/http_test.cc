#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/timer.h"

// io-10 HTTP/1.1 test: a real HTTP server + client over loopback TCP, exercising
// the integration layer (co_await TcpStream -> picohttpparser -> Request/Response
// -> write). Covers: GET + fixed response, POST body round-trip, chunked streaming
// response (the LLM-serving path), and keep-alive (multiple requests on one conn).

using hyle::container::string_ref;
using hyle::io::HttpRequest;
using hyle::io::HttpResponse;
using hyle::io::HttpResponseView;
using hyle::io::HttpClientConn;
using hyle::io::HttpServerConn;
using hyle::io::http_serve;
using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::StreamChunk;
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
constexpr std::uint16_t kPort = 54417;

struct Result {
  std::atomic<int> client_status{0};
  std::atomic<int> client_status2{0};   // 2nd request (keep-alive)
  std::atomic<int> post_status{0};       // 3rd: POST
  std::atomic<int> stream_status{0};     // 4th: GET /stream (chunked)
  std::atomic<int> stream2_status{0};    // 5th: GET /stream again (chunked-after-chunked keep-alive)
  std::atomic<bool> client_done{false};
  std::atomic<bool> server_done{false};
  char body_got[256] = {};
  char body_got2[256] = {};
  char post_body_got[256] = {};          // response body the client saw for POST
  char stream_body_got[256] = {};        // decoded chunked stream body
  char stream2_body_got[256] = {};       // 5th req's decoded body (proves trailer fix)
  char post_seen[256] = {};              // body the server saw on the POST
};

// ---- server handler: routes on method+path ---------------------------------
// A Task<HttpResponse> handler (Allocator first, per the coroutine-frame rule).
// Routing lives here (skeleton gives the loop; business is the layer above).
Task<HttpResponse> handler(Allocator /*alloc*/, void* ctx,
                           const HttpRequest& req) noexcept {
  auto* r = static_cast<Result*>(ctx);
  HttpResponse resp;
  if (req.method() == "POST") {
    // Echo back a fixed OK; record what body the server received.
    std::uint32_t n = req.body().size < 255 ? req.body().size : 255;
    std::memcpy(r->post_seen, req.body().data, n);
    r->post_seen[n] = '\0';
    resp.set_status(200);
    resp.add_header("Content-Type", "text/plain");
    resp.set_body("posted-ok");
  } else if (req.path() == "/stream") {
    // Chunked streaming response (SSE-style token stream). The generator yields a
    // few fixed chunks then ends. ctx is a small counter kept in a static (test).
    resp.set_status(200);
    resp.add_header("Content-Type", "text/event-stream");
    struct Gen {
      int i = 0;
      static StreamChunk next(void* g) noexcept {
        auto* self = static_cast<Gen*>(g);
        static const char* parts[] = {"tok-A ", "tok-B ", "tok-C"};
        if (self->i >= 3) return StreamChunk{nullptr, 0, false};
        const char* p = parts[self->i++];
        return StreamChunk{p, static_cast<std::uint32_t>(std::strlen(p)),
                           self->i < 3};
      }
    };
    // Gen must outlive the send; keep it in a static (single-threaded test).
    static Gen g;
    g.i = 0;
    resp.set_stream(&Gen::next, &g);
  } else {
    resp.set_status(200);
    resp.add_header("Content-Type", "text/plain");
    resp.set_body("hello-hyle");
  }
  co_return resp;
}

Task<int> server_fiber(Allocator alloc, IoContext* io, TcpListener* ls,
                       Result* r) {
  // Accept up to 2 connections (one for the keep-alive test client, room for more).
  TcpStream conn = co_await ls->accept(kOwner);
  if (conn.valid()) {
    co_await http_serve(alloc, std::move(conn), &handler, r);
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> client_fiber(Allocator alloc, IoContext* io, Result* r) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", kPort, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) {
    r->client_done.store(true);
    co_return 0;
  }
  HttpClientConn conn{alloc, std::move(s)};
  // Request 1: GET / -> "hello-hyle" (keep-alive default).
  HttpResponseView* resp = co_await conn.request("GET", "/", "127.0.0.1");
  if (resp != nullptr) {
    r->client_status.store(resp->status(), std::memory_order_release);
    std::uint32_t n = resp->body().size < 255 ? resp->body().size : 255;
    std::memcpy(r->body_got, resp->body().data, n);
    r->body_got[n] = '\0';
  }
  // Request 2 on the SAME connection (keep-alive): GET / again.
  HttpResponseView* resp2 = co_await conn.request("GET", "/", "127.0.0.1");
  if (resp2 != nullptr) {
    r->client_status2.store(resp2->status(), std::memory_order_release);
    std::uint32_t n = resp2->body().size < 255 ? resp2->body().size : 255;
    std::memcpy(r->body_got2, resp2->body().data, n);
    r->body_got2[n] = '\0';
  }
  // Request 3 (keep-alive): POST with a body -> server records it + replies ok.
  HttpResponseView* resp3 =
      co_await conn.request("POST", "/echo", "127.0.0.1", "the-post-body");
  if (resp3 != nullptr) {
    r->post_status.store(resp3->status(), std::memory_order_release);
    std::uint32_t n = resp3->body().size < 255 ? resp3->body().size : 255;
    std::memcpy(r->post_body_got, resp3->body().data, n);
    r->post_body_got[n] = '\0';
  }
  // Request 4 (keep-alive): GET /stream -> chunked streaming response; the client
  // decodes the chunks back into a contiguous body ("tok-A tok-B tok-C").
  HttpResponseView* resp4 = co_await conn.request("GET", "/stream", "127.0.0.1");
  if (resp4 != nullptr) {
    r->stream_status.store(resp4->status(), std::memory_order_release);
    std::uint32_t n = resp4->body().size < 255 ? resp4->body().size : 255;
    std::memcpy(r->stream_body_got, resp4->body().data, n);
    r->stream_body_got[n] = '\0';
  }
  // Request 5 (keep-alive): GET /stream AGAIN -- a chunked response FOLLOWING a
  // chunked response on the same connection. This is the regression guard for the
  // multi-chunk keep-alive bug (2026-07-05): without consume_trailer=1, the prior
  // chunked body's terminating CRLF was left in the client buffer and mistaken for
  // this response's start, so phr_parse_response choked and request() returned null.
  HttpResponseView* resp5 = co_await conn.request("GET", "/stream", "127.0.0.1");
  if (resp5 != nullptr) {
    r->stream2_status.store(resp5->status(), std::memory_order_release);
    std::uint32_t n = resp5->body().size < 255 ? resp5->body().size : 255;
    std::memcpy(r->stream2_body_got, resp5->body().data, n);
    r->stream2_body_got[n] = '\0';
  }
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// ---- request-smuggling regression harness (red-team fix, 2026-07-05) ----------
// A raw-bytes client that sends a hand-crafted request the well-formed
// HttpClientConn could never produce (CL+TE both present, obfuscated TE, duplicate
// conflicting CL), so we can prove the server REJECTS it (closes without serving).
struct SmuggleResult {
  std::atomic<int> reqs_served{0};  // times the handler ran (0 = rejected)
};
Task<HttpResponse> smuggle_handler(Allocator, void* ctx,
                                   const HttpRequest&) noexcept {
  static_cast<SmuggleResult*>(ctx)->reqs_served.fetch_add(1, std::memory_order_relaxed);
  HttpResponse resp;
  resp.set_status(200);
  resp.set_body("ok");
  co_return resp;
}
Task<int> smuggle_server(Allocator alloc, IoContext* io, TcpListener* ls,
                         SmuggleResult* r) {
  TcpStream conn = co_await ls->accept(kOwner);
  if (conn.valid()) co_await http_serve(alloc, std::move(conn), &smuggle_handler, r);
  co_return 0;
}
Task<int> smuggle_client(Allocator, IoContext* io, const char* raw,
                         std::uint16_t port) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", port, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) co_return 0;
  co_await s.write(raw, static_cast<std::uint32_t>(std::strlen(raw)));
  char buf[512];
  (void)(co_await s.read(buf, sizeof(buf)));  // one read then close (see smuggle_probe rationale)
  co_return 0;
}
// Run one raw request against a fresh server; return how many times the handler
// ran (0 = the server rejected the request before dispatch).
int smuggle_served(const char* raw, std::uint16_t port) {
  Scheduler sched(2, system_allocator(), 256);
  IoContext io(sched, 2);
  if (!io.valid()) return -1;
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", port);
  if (!ls.valid()) return -1;
  SmuggleResult r;
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = smuggle_server(system_allocator(), &io, &ls, &r);
  dc->task = smuggle_client(system_allocator(), &io, raw, port);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();
  return r.reqs_served.load();
}

// Raw-response counterpart: prove the client applies the same framing policy and
// does not keep an ambiguously framed connection in its pool.
struct ResponseProbeResult {
  std::atomic<bool> accepted{false};
};
Task<int> raw_response_server(Allocator, TcpListener* ls, const char* raw) {
  TcpStream conn = co_await ls->accept(kOwner);
  if (!conn.valid()) co_return 0;
  char request[1024];
  (void)(co_await conn.read(request, sizeof(request)));
  std::uint32_t off = 0;
  const std::uint32_t len = static_cast<std::uint32_t>(std::strlen(raw));
  while (off < len) {
    std::int64_t n = co_await conn.write(raw + off, len - off);
    if (n <= 0) break;
    off += static_cast<std::uint32_t>(n);
  }
  co_return 0;
}
Task<int> raw_response_client(Allocator alloc, IoContext* io,
                              ResponseProbeResult* r, std::uint16_t port) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", port, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) co_return 0;
  HttpClientConn conn{alloc, std::move(s)};
  r->accepted.store(
      (co_await conn.request("GET", "/", "127.0.0.1")) != nullptr,
      std::memory_order_release);
  co_return 0;
}
bool response_accepted(const char* raw, std::uint16_t port) {
  Scheduler sched(2, system_allocator(), 256);
  IoContext io(sched, 2);
  if (!io.valid()) return false;
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", port);
  if (!ls.valid()) return false;
  ResponseProbeResult r;
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = raw_response_server(system_allocator(), &ls, raw);
  dc->task = raw_response_client(system_allocator(), &io, &r, port);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();
  return r.accepted.load(std::memory_order_acquire);
}

// ---- close-delimited response body harness (red-team U3 regression) -----------
// A raw server that sends the HEAD alone (no Content-Length / no chunked), PAUSES
// (so the client's first read gets only the head), then sends a body and closes.
// A HYLE HttpClientConn must read-until-EOF and return the FULL body -- pre-fix it
// returned only the already-buffered head packet (empty body = silent truncation).
struct CloseBodyResult {
  std::atomic<int> status{0};
  std::atomic<int> body_len{-1};
};
Task<int> closebody_server(Allocator, IoContext*, TcpListener* ls, int body_size) {
  TcpStream conn = co_await ls->accept(kOwner);
  if (!conn.valid()) co_return 0;
  char rbuf[2048];
  co_await conn.read(rbuf, sizeof(rbuf));  // read+discard the request
  std::string head = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n";
  std::uint32_t off = 0, t = static_cast<std::uint32_t>(head.size());
  while (off < t) {
    std::int64_t w = co_await conn.write(head.data() + off, t - off);
    if (w <= 0) co_return 0;
    off += static_cast<std::uint32_t>(w);
  }
  hyle::sched::sleep_for_ns(20'000'000);  // 20ms: client's first read gets only head
  std::string body(static_cast<std::size_t>(body_size), 'B');
  off = 0;
  t = static_cast<std::uint32_t>(body.size());
  while (off < t) {
    std::int64_t w = co_await conn.write(body.data() + off, t - off);
    if (w <= 0) break;
    off += static_cast<std::uint32_t>(w);
  }
  co_return 0;  // fiber returns -> conn closes -> client sees EOF (body terminator)
}
Task<int> closebody_client(Allocator alloc, IoContext* io, CloseBodyResult* r,
                           std::uint16_t port) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", port, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) co_return 0;
  HttpClientConn conn{alloc, std::move(s)};
  HttpResponseView* resp = co_await conn.request("GET", "/", "127.0.0.1");
  if (resp != nullptr) {
    r->status.store(resp->status(), std::memory_order_release);
    r->body_len.store(static_cast<int>(resp->body().size), std::memory_order_release);
  }
  co_return 0;
}

}  // namespace

TEST(IoHttp, GetKeepAlive) {
  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

  auto r = std::make_unique<Result>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = server_fiber(system_allocator(), &io, &ls, r.get());
  dc->task = client_fiber(system_allocator(), &io, r.get());
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->client_done.load());
  EXPECT_EQ(r->client_status.load(), 200);
  EXPECT_STREQ(r->body_got, "hello-hyle");
  // keep-alive: 2nd request on same connection succeeded.
  EXPECT_EQ(r->client_status2.load(), 200);
  EXPECT_STREQ(r->body_got2, "hello-hyle");
  // POST: server saw the request body; client got the reply body.
  EXPECT_EQ(r->post_status.load(), 200);
  EXPECT_STREQ(r->post_seen, "the-post-body");
  EXPECT_STREQ(r->post_body_got, "posted-ok");
  // chunked streaming: client decoded the 3 chunks back to a contiguous body.
  EXPECT_EQ(r->stream_status.load(), 200);
  EXPECT_STREQ(r->stream_body_got, "tok-A tok-B tok-C");
  // chunked-after-chunked keep-alive (regression guard for the trailer-CRLF bug):
  // the 5th request -- a chunked response following another chunked response on the
  // same connection -- must succeed and decode correctly.
  EXPECT_EQ(r->stream2_status.load(), 200);
  EXPECT_STREQ(r->stream2_body_got, "tok-A tok-B tok-C");
}

// Request-smuggling defense (RFC 7230 §3.3.3, red-team fix 2026-07-05). The server
// must REJECT (close without serving) ambiguous framing that would desync a proxy
// chain; a well-formed request still serves. Ports are distinct per case (avoid
// TIME_WAIT). Bidirectionally validated: pre-fix these payloads were all served.
TEST(IoHttp, RejectsSmuggling) {
  // Baseline: well-formed request IS served.
  EXPECT_GE(smuggle_served("GET / HTTP/1.1\r\nHost: x\r\n\r\n", 54430), 1);
  // CL + TE both present (classic CL.TE / TE.CL smuggling) -> rejected.
  EXPECT_EQ(smuggle_served("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
                           "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 54431), 0);
  // Obfuscated Transfer-Encoding (not bare "chunked") -> rejected.
  EXPECT_EQ(smuggle_served("POST / HTTP/1.1\r\nHost: x\r\n"
                           "Transfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n", 54432), 0);
  // Duplicate, conflicting Content-Length -> rejected.
  EXPECT_EQ(smuggle_served("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n"
                           "Content-Length: 6\r\n\r\nhello", 54433), 0);
  // A single valid chunked request is still served (no false positive).
  EXPECT_GE(smuggle_served("POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked"
                           "\r\n\r\n5\r\nhello\r\n0\r\n\r\n", 54434), 1);
}

TEST(IoHttp, ClientRejectsAmbiguousResponseFraming) {
  EXPECT_TRUE(response_accepted(
      "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 54435));
  EXPECT_FALSE(response_accepted(
      "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n"
      "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 54436));
  EXPECT_FALSE(response_accepted(
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"
      "0\r\n\r\n", 54437));
  EXPECT_FALSE(response_accepted(
      "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
      "Content-Length: 3\r\n\r\nok!", 54438));
}

// Close-delimited response body (RFC 7230 §3.3.3 rule 7, red-team U3 regression):
// a response with neither Content-Length nor Transfer-Encoding runs until the peer
// closes. The client must read-until-EOF and get the FULL body -- pre-fix it kept
// only the head-packet's buffered tail (silent truncation to empty when the body
// arrives in a later packet, forced here by the server's 20ms split).
TEST(IoHttp, ReadsCloseDelimitedBody) {
  constexpr unsigned kWorkers = 2;
  constexpr int kBodySize = 10000;
  constexpr std::uint16_t kCbPort = 54440;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kCbPort);
  ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

  CloseBodyResult r;
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = closebody_server(system_allocator(), &io, &ls, kBodySize);
  dc->task = closebody_client(system_allocator(), &io, &r, kCbPort);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_EQ(r.status.load(), 200);
  EXPECT_EQ(r.body_len.load(), kBodySize)
      << "close-delimited body truncated (U3): got " << r.body_len.load()
      << " of " << kBodySize;
}
