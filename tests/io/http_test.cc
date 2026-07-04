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
  std::atomic<bool> client_done{false};
  std::atomic<bool> server_done{false};
  char body_got[256] = {};
  char body_got2[256] = {};
  char post_body_got[256] = {};          // response body the client saw for POST
  char stream_body_got[256] = {};        // decoded chunked stream body
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
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

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
  Fiber fs(system_allocator(), kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(system_allocator(), kDefaultStackSize, &drive_entry, dc.get());
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
}
