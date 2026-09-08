// Product-shaped HTTP/1 evidence. Each HYLE row performs a bounded POST body
// read plus a chunked/SSE-like response over the shipping coroutine transport.
// The raw row moves the exact serialized bytes over the same TcpStream path and
// is only a transport-cost control; it is not a weaker-contract SOTA peer.

#include <nanobench.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hyle/container/string_view.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using hyle::Result;
using hyle::Status;
using hyle::container::StringView;
using hyle::io::HttpBodyChunk;
using hyle::io::HttpBodySource;
using hyle::io::HttpClientConnection;
using hyle::io::HttpHeader;
using hyle::io::HttpRequestHead;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

struct Pair {
  std::intptr_t client = -1;
  std::intptr_t server = -1;
};

#if !defined(_WIN32)
Pair make_pair() noexcept {
  int descriptors[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) != 0)
    return {};
  for (int descriptor : descriptors) {
    const int flags = ::fcntl(descriptor, F_GETFL, 0);
    if (flags < 0 || ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0) {
      ::close(descriptors[0]);
      ::close(descriptors[1]);
      return {};
    }
    (void)::fcntl(descriptor, F_SETFD, FD_CLOEXEC);
  }
  return Pair{descriptors[0], descriptors[1]};
}
#else
Pair make_pair() noexcept {
  SOCKET listener =
      ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (listener == INVALID_SOCKET)
    return {};
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(listener, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) == SOCKET_ERROR ||
      ::listen(listener, 1) == SOCKET_ERROR) {
    ::closesocket(listener);
    return {};
  }
  int address_size = sizeof(address);
  if (::getsockname(listener, reinterpret_cast<sockaddr *>(&address),
                    &address_size) == SOCKET_ERROR) {
    ::closesocket(listener);
    return {};
  }
  SOCKET client =
      ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (client == INVALID_SOCKET ||
      ::connect(client, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) == SOCKET_ERROR) {
    if (client != INVALID_SOCKET)
      ::closesocket(client);
    ::closesocket(listener);
    return {};
  }
  SOCKET server = ::accept(listener, nullptr, nullptr);
  ::closesocket(listener);
  if (server == INVALID_SOCKET) {
    ::closesocket(client);
    return {};
  }
  linger reset{1, 0};
  (void)::setsockopt(client, SOL_SOCKET, SO_LINGER,
                     reinterpret_cast<const char *>(&reset), sizeof(reset));
  (void)::setsockopt(server, SOL_SOCKET, SO_LINGER,
                     reinterpret_cast<const char *>(&reset), sizeof(reset));
  return Pair{static_cast<std::intptr_t>(client),
              static_cast<std::intptr_t>(server)};
}
#endif

std::span<const std::byte> as_bytes(std::string_view value) noexcept {
  return std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(value.data()), value.size());
}

struct Workload {
  const char *name = nullptr;
  int connections = 0;
  int rounds = 0;
  std::size_t request_bytes = 0;
  std::size_t response_bytes = 0;
  std::size_t response_chunk_bytes = 0;
  std::string request_body;
  std::string response_body;
  std::string raw_request;
  std::string raw_response;
};

void append_hex(std::string &output, std::size_t value) {
  char reverse[2 * sizeof(std::size_t)];
  std::size_t count = 0;
  do {
    const unsigned digit = static_cast<unsigned>(value & 0xfu);
    reverse[count++] =
        static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
    value >>= 4;
  } while (value != 0);
  while (count != 0)
    output.push_back(reverse[--count]);
}

Workload make_workload(const char *name, int connections, int rounds,
                       std::size_t request_bytes, std::size_t response_bytes,
                       std::size_t response_chunk_bytes) {
  Workload workload;
  workload.name = name;
  workload.connections = connections;
  workload.rounds = rounds;
  workload.request_bytes = request_bytes;
  workload.response_bytes = response_bytes;
  workload.response_chunk_bytes = response_chunk_bytes;
  workload.request_body.assign(request_bytes, 'Q');
  workload.response_body.assign(response_bytes, 'T');

  workload.raw_request = "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                         "Content-Type: application/json\r\nContent-Length: ";
  workload.raw_request += std::to_string(request_bytes);
  workload.raw_request += "\r\n\r\n";
  workload.raw_request += workload.request_body;

  workload.raw_response =
      "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
      "Transfer-Encoding: chunked\r\n\r\n";
  for (std::size_t offset = 0; offset < response_bytes;
       offset += response_chunk_bytes) {
    const std::size_t count =
        std::min(response_chunk_bytes, response_bytes - offset);
    append_hex(workload.raw_response, count);
    workload.raw_response += "\r\n";
    workload.raw_response.append(workload.response_body, offset, count);
    workload.raw_response += "\r\n";
  }
  workload.raw_response += "0\r\n\r\n";
  return workload;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

struct BenchState {
  std::atomic<int> completed{0};
};

struct ChunkSource {
  std::span<const std::byte> body;
  std::size_t chunk_bytes = 0;
  std::size_t offset = 0;

  Task<Result<HttpBodyChunk>> next(AllocatorRef allocator,
                                   OperationOptions options) noexcept {
    return next_impl(allocator, this, options);
  }

private:
  static Task<Result<HttpBodyChunk>> next_impl(AllocatorRef, ChunkSource *self,
                                               OperationOptions) noexcept {
    const std::size_t count =
        std::min(self->chunk_bytes, self->body.size() - self->offset);
    const auto part = self->body.subspan(self->offset, count);
    self->offset += count;
    co_return HttpBodyChunk{part, {}, self->offset == self->body.size()};
  }
};

Task<int> http_server(AllocatorRef allocator, TcpStream stream,
                      const Workload *workload) {
  auto made = HttpServerConnection::create(allocator, std::move(stream));
  if (!made)
    co_return 0;
  HttpServerConnection connection = std::move(*made);
  static constexpr HttpHeader kHeaders[] = {
      {StringView("Content-Type"), StringView("text/event-stream")}};
  for (int round = 0; round < workload->rounds; ++round) {
    auto request = co_await connection.read_request();
    if (!request || !request->has_value())
      break;
    auto body =
        co_await request->value().body.read_all(workload->request_bytes);
    if (!body || body->size() != workload->request_bytes)
      break;
    ChunkSource source{as_bytes(workload->response_body),
                       workload->response_chunk_bytes};
    HttpResponseHead response;
    response.headers = kHeaders;
    Status sent = co_await connection.send_response(
        response, HttpBodySource::from(source));
    if (!sent.ok())
      break;
  }
  co_return 0;
}

Task<int> http_client(AllocatorRef allocator, TcpStream stream,
                      const Workload *workload, BenchState *state) {
  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made)
    co_return 0;
  HttpClientConnection connection = std::move(*made);
  static constexpr HttpHeader kHeaders[] = {
      {StringView("Content-Type"), StringView("application/json")}};
  HttpRequestHead request;
  request.method = "POST";
  request.target = "/v1/completions";
  request.authority = "127.0.0.1";
  request.headers = kHeaders;
  for (int round = 0; round < workload->rounds; ++round) {
    auto response =
        co_await connection.request(request, as_bytes(workload->request_body));
    if (!response || response->head.status() != 200)
      break;
    std::size_t received = 0;
    for (;;) {
      auto chunk = co_await response->body.read();
      if (!chunk)
        co_return 0;
      received += chunk->bytes.size();
      if (chunk->end_stream)
        break;
    }
    if (received != workload->response_bytes)
      break;
    state->completed.fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

Task<int> raw_server(AllocatorRef allocator, TcpStream stream,
                     const Workload *workload) {
  std::vector<std::byte> request(workload->raw_request.size());
  for (int round = 0; round < workload->rounds; ++round) {
    auto read = co_await hyle::io::read_exact(allocator, stream, request);
    if (!read.ok() || read.bytes != request.size())
      break;
    auto written = co_await hyle::io::write_all(
        allocator, stream, as_bytes(workload->raw_response));
    if (!written.ok() || written.bytes != workload->raw_response.size())
      break;
  }
  co_return 0;
}

Task<int> raw_client(AllocatorRef allocator, TcpStream stream,
                     const Workload *workload, BenchState *state) {
  std::vector<std::byte> response(workload->raw_response.size());
  for (int round = 0; round < workload->rounds; ++round) {
    auto written = co_await hyle::io::write_all(
        allocator, stream, as_bytes(workload->raw_request));
    if (!written.ok() || written.bytes != workload->raw_request.size())
      break;
    auto read = co_await hyle::io::read_exact(allocator, stream, response);
    if (!read.ok() || read.bytes != response.size())
      break;
    state->completed.fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

int run(const Workload &workload, unsigned workers, bool http) {
  Scheduler scheduler(workers, system_allocator(), 4096);
  IoContext io(scheduler);
  if (!io.valid())
    return 0;
  BenchState state;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  drives.reserve(static_cast<std::size_t>(workload.connections) * 2);
  fibers.reserve(static_cast<std::size_t>(workload.connections) * 2);

  for (int index = 0; index < workload.connections; ++index) {
    const Pair pair = make_pair();
    if (pair.client == -1 || pair.server == -1)
      std::abort();
    const unsigned owner = static_cast<unsigned>(index) % workers;
    hyle::io::detail::IoContextAccess::bind_handle(io, owner, pair.client);
    hyle::io::detail::IoContextAccess::bind_handle(io, owner, pair.server);
    TcpStream client(io, pair.client, owner);
    TcpStream server(io, pair.server, owner);
    auto server_drive = std::make_unique<Drive>();
    auto client_drive = std::make_unique<Drive>();
    server_drive->task =
        http ? http_server(system_allocator(), std::move(server), &workload)
             : raw_server(system_allocator(), std::move(server), &workload);
    client_drive->task = http
                             ? http_client(system_allocator(),
                                           std::move(client), &workload, &state)
                             : raw_client(system_allocator(), std::move(client),
                                          &workload, &state);
    auto server_fiber = std::make_unique<Fiber>(kDefaultStackSize, &drive_entry,
                                                server_drive.get());
    auto client_fiber = std::make_unique<Fiber>(kDefaultStackSize, &drive_entry,
                                                client_drive.get());
    if (!scheduler.spawn_on(owner, server_fiber.get()) ||
        !scheduler.spawn_on(owner, client_fiber.get())) {
      std::abort();
    }
    drives.push_back(std::move(server_drive));
    drives.push_back(std::move(client_drive));
    fibers.push_back(std::move(server_fiber));
    fibers.push_back(std::move(client_fiber));
  }
  scheduler.drain();
  scheduler.shutdown();
  return state.completed.load(std::memory_order_relaxed);
}

void run_bench(const Workload &workload) {
  const std::uint64_t operations =
      static_cast<std::uint64_t>(workload.connections) * workload.rounds;
  ankerl::nanobench::Bench bench;
  bench.title(workload.name)
      .relative(true)
      .batch(operations)
      .unit("request")
      .minEpochIterations(1)
      .epochs(7);
  bench.run("raw transport control (same bytes)", [&] {
    if (run(workload, 1, false) != static_cast<int>(operations))
      std::abort();
  });
  bench.run("HYLE HTTP/1 (1 worker)", [&] {
    if (run(workload, 1, true) != static_cast<int>(operations))
      std::abort();
  });
  if (workload.connections >= 4) {
    bench.run("HYLE HTTP/1 (4 workers)", [&] {
      if (run(workload, 4, true) != static_cast<int>(operations))
        std::abort();
    });
  }
}

} // namespace

int main() {
#if defined(_WIN32)
  WSADATA data{};
  if (::WSAStartup(MAKEWORD(2, 2), &data) != 0)
    return 2;
#endif
  const Workload normal = make_workload(
      "HTTP/1 REST + streaming: 1 KiB each direction", 32, 32, 1024, 1024, 128);
  const Workload holdout =
      make_workload("HTTP/1 fragmented holdout: 64 KiB each direction", 8, 4,
                    64 * 1024, 64 * 1024, 4096);
  run_bench(normal);
  run_bench(holdout);
#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
