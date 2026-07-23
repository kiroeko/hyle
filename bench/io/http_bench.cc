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
//   3. HYLE_HTTP_REGION_FOOTPRINT_VARIANT={capped,capped_collect,unbounded}
//      selects an isolated long-connection Region/RSS certificate instead of
//      nanobench; scripts/http-region-footprint-stat.py drives fresh processes.
//
// Load: kConns keep-alive connections, each doing kReqRoundTrips request/response
// round-trips (a GET with a small fixed "hello-hyle" body), sharded round-robin
// across workers. Socketpair transport (like echo_bench) -- no accept, no TIME_WAIT.
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/timer.h"

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/resource.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <psapi.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#  include <pthread.h>
#endif

using hyle::container::string_ref;
using hyle::io::HttpClientConn;
using hyle::io::HttpRequest;
using hyle::io::HttpResponse;
using hyle::io::HttpResponseView;
using hyle::io::HttpServeOptions;
using hyle::io::http_serve;
using hyle::io::IoContext;
using hyle::io::TcpStream;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::mem::system_collect;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Latch;
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

#if !defined(_WIN32)
Pair make_pair() {
  int sv[2];
  Pair p;
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0) {
    for (int fd : sv) {
      const int flags = ::fcntl(fd, F_GETFL, 0);
      if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        ::close(sv[0]);
        ::close(sv[1]);
        return p;
      }
      ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    }
    p.a = sv[0];
    p.b = sv[1];
  }
  return p;
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
  if (::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) == SOCKET_ERROR ||
      ::listen(ls, 1) == SOCKET_ERROR) {
    ::closesocket(ls);
    return p;
  }
  int slen = sizeof(sa);
  if (::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen) ==
      SOCKET_ERROR) {
    ::closesocket(ls);
    return p;
  }
  SOCKET c = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (c == INVALID_SOCKET ||
      ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) ==
          SOCKET_ERROR) {
    if (c != INVALID_SOCKET) ::closesocket(c);
    ::closesocket(ls);
    return p;
  }
  SOCKET s = ::accept(ls, nullptr, nullptr);
  ::closesocket(ls);
  if (s == INVALID_SOCKET) {
    ::closesocket(c);
    return p;
  }
  // TcpStream owns and closes both endpoints. Mark this benchmark-only pair
  // abortive up front so process-level repetition does not fill Windows'
  // ephemeral range with TIME_WAIT sockets after ~10 runs.
  linger reset{1, 0};
  ::setsockopt(c, SOL_SOCKET, SO_LINGER,
               reinterpret_cast<const char*>(&reset), sizeof(reset));
  ::setsockopt(s, SOL_SOCKET, SO_LINGER,
               reinterpret_cast<const char*>(&reset), sizeof(reset));
  p.a = static_cast<std::intptr_t>(c);
  p.b = static_cast<std::intptr_t>(s);
  return p;
}
#endif

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

struct BenchState {
  std::atomic<int> completed{0};
};

struct ProcessMemory {
  std::size_t current_rss = 0;
  std::size_t peak_rss = 0;
};

ProcessMemory process_memory() noexcept {
  ProcessMemory result;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) !=
      0) {
    result.current_rss = static_cast<std::size_t>(counters.WorkingSetSize);
    result.peak_rss = static_cast<std::size_t>(counters.PeakWorkingSetSize);
  }
#elif defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
    result.current_rss = static_cast<std::size_t>(info.resident_size);
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    result.peak_rss = static_cast<std::size_t>(usage.ru_maxrss);
#elif defined(__linux__)
  long resident_pages = 0;
  if (FILE* statm = std::fopen("/proc/self/statm", "r")) {
    long total_pages = 0;
    const int fields =
        std::fscanf(statm, "%ld %ld", &total_pages, &resident_pages);
    std::fclose(statm);
    if (fields != 2) resident_pages = 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  rusage usage{};
  if (resident_pages > 0 && page_size > 0)
    result.current_rss = static_cast<std::size_t>(resident_pages) *
                         static_cast<std::size_t>(page_size);
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    result.peak_rss = static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
  return result;
}

void print_certificate_meta() noexcept {
#if defined(__clang__) && defined(_MSC_VER)
  std::printf("HYLE_HTTP_MEM_CERT meta format=1 compiler=clangcl-%d.%d.%d-msvc%d "
              "platform=windows\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__,
              _MSC_VER);
#elif defined(_MSC_VER)
  std::printf("HYLE_HTTP_MEM_CERT meta format=1 compiler=msvc-%d "
              "platform=windows\n",
              _MSC_VER);
#elif defined(__APPLE__) && defined(__clang__)
  std::printf("HYLE_HTTP_MEM_CERT meta format=1 compiler=appleclang-%d.%d.%d "
              "platform=macos\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__clang__)
  std::printf("HYLE_HTTP_MEM_CERT meta format=1 compiler=clang-%d.%d.%d "
              "platform=linux\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  std::printf("HYLE_HTTP_MEM_CERT meta format=1 compiler=gcc-%d.%d.%d "
              "platform=linux\n",
              __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  std::printf("HYLE_HTTP_MEM_CERT meta format=1 compiler=unknown "
              "platform=unknown\n");
#endif
}

void configure_footprint_thread() noexcept {
#if defined(__APPLE__)
  const char* requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char* name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "background") == 0) {
    qos = QOS_CLASS_BACKGROUND;
    name = "background";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "HYLE_BENCH_APPLE_QOS must be interactive, utility, or "
                 "background (got %s)\n",
                 requested);
    std::abort();
  }
  const int result = pthread_set_qos_class_self_np(qos, 0);
  std::printf("# HYLE_HTTP_MEM_BENCH apple_main_qos=%s qos_ok=%d "
              "affinity=unsupported\n",
              name, result == 0 ? 1 : 0);
  if (result != 0) std::abort();
#endif
}

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
Task<int> http_client(Allocator alloc, TcpStream client_end,
                      BenchState* state) {
  HttpClientConn conn{alloc, std::move(client_end)};
  for (int i = 0; i < kReqRoundTrips; ++i) {
    HttpResponseView* resp = co_await conn.request("GET", "/", "127.0.0.1");
    if (resp == nullptr) break;
    state->completed.fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;  // conn drops here -> client fd closed -> server sees EOF
}

// ---- request-Region footprint certificate --------------------------------
// This is a fresh-process mode of the same real HTTP path, selected through
// HYLE_HTTP_REGION_FOOTPRINT_VARIANT. It keeps many connections alive while a
// small request, a 1 MiB request-local spike, and another small request pass
// through each one. OS RSS and exact server-backing bytes are sampled only
// after every connection has completed a phase.
constexpr int kFootprintPhases = 3;

struct BackingStats {
  std::atomic<std::size_t> current{0};
  std::atomic<std::size_t> peak{0};
};

void update_peak(BackingStats* stats, std::size_t value) noexcept {
  std::size_t seen = stats->peak.load(std::memory_order_relaxed);
  while (seen < value &&
         !stats->peak.compare_exchange_weak(seen, value,
                                            std::memory_order_relaxed)) {
  }
}

void* tracked_allocate(void* state, std::size_t bytes,
                       std::size_t alignment) noexcept {
  void* pointer = system_allocator().allocate(bytes, alignment);
  if (pointer != nullptr) {
    auto* stats = static_cast<BackingStats*>(state);
    const std::size_t current =
        stats->current.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    update_peak(stats, current);
  }
  return pointer;
}

void tracked_deallocate(void* state, void* pointer, std::size_t bytes,
                        std::size_t alignment) noexcept {
  if (pointer != nullptr) {
    static_cast<BackingStats*>(state)->current.fetch_sub(
        bytes, std::memory_order_relaxed);
  }
  system_allocator().deallocate(pointer, bytes, alignment);
}

void* tracked_reallocate(void* state, void* pointer, std::size_t old_bytes,
                         std::size_t new_bytes,
                         std::size_t alignment) noexcept {
  void* replacement = system_allocator().reallocate(
      pointer, old_bytes, new_bytes, alignment);
  if (replacement == nullptr && new_bytes != 0) return nullptr;
  auto* stats = static_cast<BackingStats*>(state);
  std::size_t current = 0;
  if (new_bytes >= old_bytes) {
    current = stats->current.fetch_add(new_bytes - old_bytes,
                                       std::memory_order_relaxed) +
              (new_bytes - old_bytes);
  } else {
    current = stats->current.fetch_sub(old_bytes - new_bytes,
                                       std::memory_order_relaxed) -
              (old_bytes - new_bytes);
  }
  update_peak(stats, current);
  return replacement;
}

Allocator tracked_allocator(BackingStats* stats) noexcept {
  return Allocator(&tracked_allocate, &tracked_deallocate,
                   &tracked_reallocate, stats);
}

struct FootprintState {
  int connections = 0;
  std::size_t spike_bytes = 0;
  std::size_t max_block_bytes = 0;
  bool force_collect = false;
  std::size_t collect_ns = 0;
  BackingStats backing;
  std::array<std::unique_ptr<Latch>, kFootprintPhases> arrived;
  std::array<std::unique_ptr<Latch>, kFootprintPhases> released;
  std::array<ProcessMemory, kFootprintPhases> memory{};
  std::array<std::size_t, kFootprintPhases> exact_backing{};
};

void touch_pages(void* allocation, std::size_t bytes) noexcept {
  auto* data = static_cast<volatile std::byte*>(allocation);
  constexpr std::size_t kPage = 4096;
  for (std::size_t offset = 0; offset < bytes; offset += kPage)
    data[offset] = std::byte{0x5a};
  data[bytes - 1] = std::byte{0xa5};
}

Task<HttpResponse> footprint_handler(Allocator alloc, void* context,
                                     const HttpRequest& request) noexcept {
  auto* state = static_cast<FootprintState*>(context);
  HttpResponse response;
  if (request.path() == "/spike") {
    // 8 KiB resembles parser/graph scratch and packs several allocations per
    // 64 KiB ordinary block; 32 KiB would manufacture a near-2x size-class
    // crossing that tests fragmentation rather than retention policy.
    constexpr std::size_t kPiece = 8 * 1024;
    std::size_t allocated = 0;
    while (allocated < state->spike_bytes) {
      const std::size_t bytes =
          std::min(kPiece, state->spike_bytes - allocated);
      void* pointer = alloc.allocate(bytes, alignof(std::max_align_t));
      if (pointer == nullptr) {
        response.set_status(500);
        response.set_body("request-region-oom");
        co_return response;
      }
      touch_pages(pointer, bytes);
      allocated += bytes;
    }
  } else {
    void* pointer = alloc.allocate(64, alignof(std::max_align_t));
    if (pointer == nullptr) {
      response.set_status(500);
      response.set_body("request-region-oom");
      co_return response;
    }
    std::memset(pointer, 0x5a, 64);
  }
  response.set_body("ok");
  co_return response;
}

Task<int> footprint_server(Allocator alloc, TcpStream server_end,
                           FootprintState* state,
                           std::size_t retain_limit) {
  HttpServeOptions options;
  options.request_region.max_block_size = state->max_block_bytes;
  options.request_region_retain_limit = retain_limit;
  co_await http_serve(alloc, std::move(server_end), &footprint_handler, state,
                      options);
  co_return 0;
}

Task<int> footprint_client(Allocator alloc, TcpStream client_end,
                           FootprintState* state) {
  HttpClientConn connection{alloc, std::move(client_end)};
  constexpr const char* paths[kFootprintPhases] = {"/small", "/spike",
                                                    "/small"};
  for (int phase = 0; phase < kFootprintPhases; ++phase) {
    HttpResponseView* response =
        co_await connection.request("GET", paths[phase], "127.0.0.1");
    if (response == nullptr || response->status() != 200) std::abort();
    state->arrived[phase]->count_down();
    state->released[phase]->wait();
  }
  co_return 0;
}

Task<int> footprint_coordinator(Allocator, FootprintState* state) {
  for (int phase = 0; phase < kFootprintPhases; ++phase) {
    state->arrived[phase]->wait();
    // Client receive can win just before the corresponding server send
    // continuation. Parking (rather than runnable yielding) lets the worker poll
    // every remaining completion and clear every request Region.
    std::size_t previous = (std::numeric_limits<std::size_t>::max)();
    int stable_rounds = 0;
    for (int round = 0; round < 32 && stable_rounds < 4; ++round) {
      hyle::sched::sleep_for_ns(1'000'000);
      const std::size_t current =
          state->backing.current.load(std::memory_order_acquire);
      if (current == previous)
        ++stable_rounds;
      else
        stable_rounds = 0;
      previous = current;
    }
    state->exact_backing[phase] =
        state->backing.current.load(std::memory_order_acquire);
    if (phase == 1 && state->force_collect) {
      const auto begin = std::chrono::steady_clock::now();
      system_collect(true);
      const auto elapsed = std::chrono::steady_clock::now() - begin;
      state->collect_ns = static_cast<std::size_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    }
    // RSS is intentionally a settled idle observation, not the just-touched
    // process peak recorded separately. Backing deallocation is synchronous,
    // while mimalloc/OS page purge is allowed to lag briefly.
    hyle::sched::sleep_for_ns(25'000'000);
    state->memory[phase] = process_memory();
    state->released[phase]->count_down();
  }
  co_return 0;
}

std::string environment_value(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return {};
  std::string result(value);
  std::free(value);
  return result;
#else
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string{};
#endif
}

int env_positive_int(const char* name, int fallback) {
  const std::string value = environment_value(name);
  if (value.empty()) return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  return end != value.c_str() && *end == '\0' && parsed > 0
             ? static_cast<int>(parsed)
             : fallback;
}

int run_region_footprint(const char* variant) {
  configure_footprint_thread();
  const int connections =
      env_positive_int("HYLE_HTTP_REGION_CONNECTIONS", 128);
  const int spike_kib =
      env_positive_int("HYLE_HTTP_REGION_SPIKE_KIB", 1024);
  const int max_block_kib =
      env_positive_int("HYLE_HTTP_REGION_BLOCK_KIB", 64);
  const std::size_t retain_limit =
      std::strcmp(variant, "unbounded") == 0
          ? (std::numeric_limits<std::size_t>::max)()
          : 256u * 1024u;
  if (std::strcmp(variant, "capped") != 0 &&
      std::strcmp(variant, "capped_collect") != 0 &&
      std::strcmp(variant, "unbounded") != 0) {
    std::fputs("HYLE_HTTP_REGION_FOOTPRINT_VARIANT must be capped, "
               "capped_collect, or unbounded\n",
               stderr);
    return 2;
  }

  Scheduler scheduler(1, system_allocator(), 4096);
  IoContext io(scheduler, 1);
  FootprintState state;
  state.connections = connections;
  state.spike_bytes = static_cast<std::size_t>(spike_kib) * 1024;
  state.max_block_bytes = static_cast<std::size_t>(max_block_kib) * 1024;
  state.force_collect = std::strcmp(variant, "capped_collect") == 0;
  for (int phase = 0; phase < kFootprintPhases; ++phase) {
    state.arrived[phase] = std::make_unique<Latch>(connections);
    state.released[phase] = std::make_unique<Latch>(1);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  const ProcessMemory baseline = process_memory();

  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  drives.reserve(static_cast<std::size_t>(connections) * 2 + 1);
  fibers.reserve(static_cast<std::size_t>(connections) * 2 + 1);
  auto coordinator = std::make_unique<Drive>();
  coordinator->task = footprint_coordinator(system_allocator(), &state);
  auto coordinator_fiber = std::make_unique<Fiber>(
      kDefaultStackSize, &drive_entry, coordinator.get());
  scheduler.spawn_on(0, coordinator_fiber.get());
  drives.push_back(std::move(coordinator));
  fibers.push_back(std::move(coordinator_fiber));
  for (int i = 0; i < connections; ++i) {
    Pair pair = make_pair();
    if (pair.a == -1 || pair.b == -1) std::abort();
    io.bind_fd(0, pair.a);
    io.bind_fd(0, pair.b);
    TcpStream client_end(io, pair.a, 0);
    TcpStream server_end(io, pair.b, 0);
    auto server = std::make_unique<Drive>();
    auto client = std::make_unique<Drive>();
    server->task = footprint_server(tracked_allocator(&state.backing),
                                    std::move(server_end), &state,
                                    retain_limit);
    client->task = footprint_client(system_allocator(), std::move(client_end),
                                    &state);
    auto server_fiber =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, server.get());
    auto client_fiber =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, client.get());
    scheduler.spawn_on(0, server_fiber.get());
    scheduler.spawn_on(0, client_fiber.get());
    drives.push_back(std::move(server));
    drives.push_back(std::move(client));
    fibers.push_back(std::move(server_fiber));
    fibers.push_back(std::move(client_fiber));
  }
  scheduler.drain();
  scheduler.shutdown();
  fibers.clear();
  drives.clear();
  const std::size_t released_backing =
      state.backing.current.load(std::memory_order_acquire);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  const ProcessMemory released_memory = process_memory();

  print_certificate_meta();
  std::printf(
      "HYLE_HTTP_MEM_CERT footprint scenario=http_request_region "
      "variant=%s connections=%d spike_bytes_per_connection=%zu "
      "max_block_bytes=%zu retain_limit_bytes=%zu "
      "baseline_rss_bytes=%zu warm_rss_bytes=%zu "
      "spike_rss_bytes=%zu reuse_rss_bytes=%zu release_rss_bytes=%zu "
      "process_peak_rss_bytes=%zu warm_backing_bytes=%zu "
      "spike_backing_bytes=%zu reuse_backing_bytes=%zu "
      "peak_backing_bytes=%zu release_backing_bytes=%zu collect_ns=%zu\n",
      variant, connections, state.spike_bytes, state.max_block_bytes,
      retain_limit,
      baseline.current_rss, state.memory[0].current_rss,
      state.memory[1].current_rss, state.memory[2].current_rss,
      released_memory.current_rss, released_memory.peak_rss,
      state.exact_backing[0], state.exact_backing[1],
      state.exact_backing[2], state.backing.peak.load(std::memory_order_relaxed),
      released_backing, state.collect_ns);
  return released_backing == 0 ? 0 : 3;
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
Task<int> raw_client(Allocator, TcpStream client_end, BenchState* state) {
  char buf[256];
  for (int i = 0; i < kReqRoundTrips; ++i) {
    std::int64_t w = co_await client_end.write(kRawReq, kRawReqLen);
    if (w != static_cast<std::int64_t>(kRawReqLen)) break;
    std::int64_t r = co_await client_end.read(buf, sizeof(buf));
    if (r <= 0) break;
    state->completed.fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

// Build kConns connections on `workers` cores, sharded round-robin, and drive them
// to completion. `use_http` selects the HYLE HTTP path vs the raw-TCP baseline.
int run(unsigned workers, bool use_http) {
  Scheduler sched(workers, system_allocator(), 4096);
  IoContext io(sched, workers);

  std::vector<Pair> pairs(kConns);
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  BenchState state;
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
      dc->task =
          http_client(system_allocator(), std::move(client_end), &state);
    } else {
      ds->task = raw_server(system_allocator(), std::move(server_end));
      dc->task =
          raw_client(system_allocator(), std::move(client_end), &state);
    }
    auto fs =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, ds.get());
    auto fc =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, dc.get());
    sched.spawn_on(owner, fs.get());
    sched.spawn_on(owner, fc.get());
    drives.push_back(std::move(ds));
    drives.push_back(std::move(dc));
    fibers.push_back(std::move(fs));
    fibers.push_back(std::move(fc));
  }
  sched.drain();
  sched.shutdown();
  return state.completed.load(std::memory_order_relaxed);
}

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
  const std::string footprint_variant =
      environment_value("HYLE_HTTP_REGION_FOOTPRINT_VARIANT");
  if (!footprint_variant.empty()) {
    const int result = run_region_footprint(footprint_variant.c_str());
#if defined(_WIN32)
    ::WSACleanup();
#endif
    return result;
  }
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
        .minEpochIterations(1)
        .epochs(5);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE HTTP (%u workers)", w);
      b.run(name, [&] {
        if (run(w, /*use_http=*/true) != kConns * kReqRoundTrips)
          std::abort();
      });
    }
    // Transport floor: raw TCP echo over the SAME co_await TcpStream (1 worker),
    // moving the identical request+response bytes with zero HTTP work.
    b.run("raw TCP echo (1 worker, same bytes, no HTTP)",
          [&] {
            if (run(1u, /*use_http=*/false) != kConns * kReqRoundTrips)
              std::abort();
          });
  }

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
