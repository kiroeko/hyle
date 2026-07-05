// io-10 HTTP benchmark: the honest "what does the HTTP parse/framing layer cost?"
// comparison. HYLE's HTTP/1.1 request/response throughput has TWO cost components:
// (1) the io transport -- co_await TcpStream over the proactor/fiber stack, ALREADY
// measured by echo_bench (this is NOT what we're isolating here); (2) the HTTP layer
// ON TOP -- picohttpparser parse, GrowBuf head/body copies, Request/Response building,
// status-line + header formatting. A single req/s number can't separate them, so the
// baseline here is a RAW TCP echo over the SAME co_await TcpStream transport (same
// socketpair, same fibers, same proactor, NO HTTP) shuttling the IDENTICAL request +
// response byte volume. The ratio HYLE-HTTP / raw-TCP = pure HTTP-layer overhead,
// with the io transport cancelled out on both sides.
//
// Two measurements:
//   1. HYLE HTTP req/s at 1/2/4/8 workers -- shard scaling of the full HTTP path
//      (does the parse/framing layer preserve shard-per-core scaling?).
//   2. raw TCP echo (1 worker) moving the same bytes -- the transport floor; the
//      1-worker HTTP row over this = the framing overhead multiplier.
//
// Load: kConns keep-alive connections, each doing kReqRoundTrips request/response
// round-trips (a GET with a small fixed "hello-hyle" body), sharded round-robin
// across workers. Socketpair transport (like echo_bench) -- no accept, no TIME_WAIT.
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(__linux__)
#  include <sys/socket.h>
#  include <unistd.h>
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#endif

using hyle::container::string_ref;
using hyle::io::HttpClientConn;
using hyle::io::HttpRequest;
using hyle::io::HttpResponse;
using hyle::io::HttpResponseView;
using hyle::io::http_serve;
using hyle::io::IoContext;
using hyle::io::TcpStream;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr int kReqRoundTrips = 64;  // request/response round-trips per connection
constexpr int kConns = 64;          // concurrent keep-alive connections

// The EXACT bytes HYLE's client/server exchange for this workload (see below) -- the
// raw-TCP baseline shuttles these verbatim so the byte volume on the wire is identical
// and the ONLY difference is the HTTP CPU work (parse + build), not transport size.
//   client request : "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"  (no body -> no CL)
//   server response: 200 + Content-Type: text/plain + Content-Length: 10 + "hello-hyle"
const char kRawReq[] = "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
const char kRawResp[] =
    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 10\r\n\r\n"
    "hello-hyle";
constexpr std::uint32_t kRawReqLen = sizeof(kRawReq) - 1;
constexpr std::uint32_t kRawRespLen = sizeof(kRawResp) - 1;

// A connected fd pair (socketpair on Linux, loopback TCP on Windows) -- same as
// echo_bench: no accept, no TIME_WAIT, pure request/response timing.
struct Pair {
  std::intptr_t a = -1;  // client end
  std::intptr_t b = -1;  // server end
};

#if defined(__linux__)
Pair make_pair() {
  int sv[2];
  Pair p;
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0) {
    p.a = sv[0];
    p.b = sv[1];
  }
  return p;
}
void close_pair(const Pair& p) {
  if (p.a >= 0) ::close(static_cast<int>(p.a));
  if (p.b >= 0) ::close(static_cast<int>(p.b));
}
#endif
#if defined(_WIN32)
Pair make_pair() {
  Pair p;
  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (ls == INVALID_SOCKET) return p;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  ::listen(ls, 1);
  int slen = sizeof(sa);
  ::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen);
  SOCKET c = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  SOCKET s = ::accept(ls, nullptr, nullptr);
  ::closesocket(ls);
  p.a = static_cast<std::intptr_t>(c);
  p.b = static_cast<std::intptr_t>(s);
  return p;
}
void close_pair(const Pair& p) {
  if (p.a != -1) ::closesocket(static_cast<SOCKET>(p.a));
  if (p.b != -1) ::closesocket(static_cast<SOCKET>(p.b));
}
#endif

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// ---- HYLE HTTP path --------------------------------------------------------
// Server handler: GET / -> a small fixed "hello-hyle" body (the response whose
// bytes kRawResp mirrors). ctx unused (stateless fixed response).
Task<HttpResponse> bench_handler(Allocator, void*, const HttpRequest&) noexcept {
  HttpResponse resp;
  resp.set_status(200);
  resp.add_header("Content-Type", "text/plain");
  resp.set_body("hello-hyle");
  co_return resp;
}

// Server side of one connection: run the keep-alive serve loop until the client
// closes (its N requests done -> HttpClientConn drops -> socket close -> EOF).
Task<int> http_server(Allocator alloc, TcpStream server_end) {
  co_await http_serve(alloc, std::move(server_end), &bench_handler, nullptr);
  co_return 0;
}

// Client side: kReqRoundTrips GET / requests on ONE keep-alive connection, then drop
// (closing the socket, which ends the server's serve loop).
Task<int> http_client(Allocator alloc, TcpStream client_end) {
  HttpClientConn conn{alloc, std::move(client_end)};
  for (int i = 0; i < kReqRoundTrips; ++i) {
    HttpResponseView* resp = co_await conn.request("GET", "/", "127.0.0.1");
    if (resp == nullptr) break;
  }
  co_return 0;  // conn drops here -> client fd closed -> server sees EOF
}

// ---- raw TCP echo baseline (same transport, NO HTTP) -----------------------
// Server: kReqRoundTrips times, read the request bytes then write the response
// bytes. No parsing, no framing -- just shuttle the identical byte volume over the
// SAME co_await TcpStream the HTTP path uses. Small messages over loopback/socketpair
// arrive whole in one read (echo_bench relies on the same property).
Task<int> raw_server(Allocator, TcpStream server_end) {
  char buf[256];
  for (int i = 0; i < kReqRoundTrips; ++i) {
    std::int64_t r = co_await server_end.read(buf, sizeof(buf));
    if (r <= 0) break;
    std::int64_t w = co_await server_end.write(kRawResp, kRawRespLen);
    if (w != static_cast<std::int64_t>(kRawRespLen)) break;
  }
  co_return 0;
}
Task<int> raw_client(Allocator, TcpStream client_end) {
  char buf[256];
  for (int i = 0; i < kReqRoundTrips; ++i) {
    std::int64_t w = co_await client_end.write(kRawReq, kRawReqLen);
    if (w != static_cast<std::int64_t>(kRawReqLen)) break;
    std::int64_t r = co_await client_end.read(buf, sizeof(buf));
    if (r <= 0) break;
  }
  co_return 0;
}

// Build kConns connections on `workers` cores, sharded round-robin, and drive them
// to completion. `use_http` selects the HYLE HTTP path vs the raw-TCP baseline.
void run(unsigned workers, bool use_http) {
  Scheduler sched(workers, system_allocator(), 4096);
  IoContext io(sched, workers);

  std::vector<Pair> pairs(kConns);
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  drives.reserve(static_cast<std::size_t>(kConns) * 2);
  fibers.reserve(static_cast<std::size_t>(kConns) * 2);

  for (int i = 0; i < kConns; ++i) {
    pairs[i] = make_pair();
    unsigned owner = static_cast<unsigned>(i) % workers;
    io.bind_fd(owner, pairs[i].a);
    io.bind_fd(owner, pairs[i].b);
    // Wrap both ends in TcpStreams bound to the connection's owner core.
    TcpStream client_end(io, pairs[i].a, owner);
    TcpStream server_end(io, pairs[i].b, owner);

    auto ds = std::make_unique<Drive>();
    auto dc = std::make_unique<Drive>();
    if (use_http) {
      ds->task = http_server(system_allocator(), std::move(server_end));
      dc->task = http_client(system_allocator(), std::move(client_end));
    } else {
      ds->task = raw_server(system_allocator(), std::move(server_end));
      dc->task = raw_client(system_allocator(), std::move(client_end));
    }
    auto fs = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, ds.get());
    auto fc = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, dc.get());
    sched.spawn_on(owner, fs.get());
    sched.spawn_on(owner, fc.get());
    drives.push_back(std::move(ds));
    drives.push_back(std::move(dc));
    fibers.push_back(std::move(fs));
    fibers.push_back(std::move(fc));
  }
  sched.drain();
  sched.shutdown();
  for (auto& p : pairs) close_pair(p);
}

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
  const std::uint64_t batch =
      static_cast<std::uint64_t>(kConns) * kReqRoundTrips;  // req/resps per epoch

  // ---- HTTP req/s at 1/2/4/8 workers + raw-TCP transport floor ----
  // The HYLE HTTP rows vs the raw row (same bytes, same transport, no parse): the
  // ratio is the HTTP parse/framing overhead. The 1->8 worker trend is the shard
  // scaling of the full HTTP path.
  {
    ankerl::nanobench::Bench b;
    b.title("io HTTP: 64 conns x 64 req/resp round-trips (GET / -> hello-hyle)")
        .relative(true)
        .batch(batch)
        .unit("req")
        .minEpochIterations(30);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE HTTP (%u workers)", w);
      b.run(name, [&] { run(w, /*use_http=*/true); });
    }
    // Transport floor: raw TCP echo over the SAME co_await TcpStream (1 worker),
    // moving the identical request+response bytes with zero HTTP work.
    b.run("raw TCP echo (1 worker, same bytes, no HTTP)",
          [&] { run(1u, /*use_http=*/false); });
  }

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
