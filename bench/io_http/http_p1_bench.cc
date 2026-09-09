// Same-contract HTTP/1 P1 evidence for the inference-server journey.
//
// Both routes keep one asynchronous event worker alive. A charged episode
// creates one connected local stream, constructs client/server HTTP state,
// performs multiple sequential keep-alive POST requests, retains and validates
// each bounded request body, writes the same application-sized response chunks,
// incrementally validates the decoded response before retiring the connection.
// HYLE uses its public Scheduler/IoContext/HTTP API; the peer uses public
// Boost.Asio/Boost.Beast APIs. No raw parser or blocking-socket row can gate.

#include <nanobench.h>

#include <boost/asio.hpp>
#include <boost/asio/local/stream_protocol.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/http.hpp>

#include <algorithm>
#include <array>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include "hyle/container/string_view.h"
#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

using hyle::Result;
using hyle::Status;
using hyle::container::StringView;
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
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::size_t kPeerBodyBufferBytes = 8 * 1024;

void close_descriptor(int descriptor) noexcept {
  if (descriptor >= 0)
    (void)::close(descriptor);
}

class SocketPair {
public:
  SocketPair() noexcept = default;
  SocketPair(int client, int server) noexcept
      : client_(client), server_(server) {}
  ~SocketPair() {
    close_descriptor(client_);
    close_descriptor(server_);
  }
  SocketPair(SocketPair &&other) noexcept
      : client_(std::exchange(other.client_, -1)),
        server_(std::exchange(other.server_, -1)) {}
  SocketPair &operator=(SocketPair &&other) noexcept {
    if (this != &other) {
      close_descriptor(client_);
      close_descriptor(server_);
      client_ = std::exchange(other.client_, -1);
      server_ = std::exchange(other.server_, -1);
    }
    return *this;
  }
  SocketPair(const SocketPair &) = delete;
  SocketPair &operator=(const SocketPair &) = delete;

  bool valid() const noexcept { return client_ >= 0 && server_ >= 0; }
  int release_client() noexcept { return std::exchange(client_, -1); }
  int release_server() noexcept { return std::exchange(server_, -1); }

private:
  int client_ = -1;
  int server_ = -1;
};

SocketPair make_socket_pair() noexcept {
  int descriptors[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) != 0)
    return {};
  constexpr int kSocketBufferBytes = 4 * 1024;
  for (int descriptor : descriptors) {
    const int flags = ::fcntl(descriptor, F_GETFL, 0);
    if (flags < 0 ||
        ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0 ||
        ::fcntl(descriptor, F_SETFD, FD_CLOEXEC) != 0) {
      close_descriptor(descriptors[0]);
      close_descriptor(descriptors[1]);
      return {};
    }
    (void)::setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF,
                       &kSocketBufferBytes, sizeof(kSocketBufferBytes));
    (void)::setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF,
                       &kSocketBufferBytes, sizeof(kSocketBufferBytes));
  }
  return SocketPair(descriptors[0], descriptors[1]);
}

std::span<const std::byte> as_bytes(std::string_view value) noexcept {
  return {reinterpret_cast<const std::byte *>(value.data()), value.size()};
}

struct Workload {
  const char *title = nullptr;
  int rounds = 0;
  std::size_t request_bytes = 0;
  std::size_t response_bytes = 0;
  std::size_t response_chunk_bytes = 0;
  std::string request_body;
  std::string response_body;
};

Workload make_workload(const char *title, int rounds,
                       std::size_t request_bytes,
                       std::size_t response_bytes,
                       std::size_t response_chunk_bytes) {
  Workload workload;
  workload.title = title;
  workload.rounds = rounds;
  workload.request_bytes = request_bytes;
  workload.response_bytes = response_bytes;
  workload.response_chunk_bytes = response_chunk_bytes;
  workload.request_body.assign(request_bytes, 'Q');
  workload.response_body.assign(response_bytes, 'T');
  return workload;
}

Task<int> hyle_server(AllocatorRef allocator, TcpStream stream,
                      const Workload *workload) {
  auto made = HttpServerConnection::create(allocator, std::move(stream));
  if (!made)
    co_return -1;
  HttpServerConnection connection = std::move(*made);
  static constexpr HttpHeader kHeaders[] = {
      {StringView("Content-Type"), StringView("text/event-stream")}};

  for (int round = 0; round < workload->rounds; ++round) {
    auto incoming = co_await connection.read_request();
    if (!incoming || !incoming->has_value())
      co_return -1;
    if (incoming->value().head.method() != "POST" ||
        incoming->value().head.target() != "/v1/completions")
      co_return -1;
    auto body =
        co_await incoming->value().body.read_all(workload->request_bytes);
    if (!body || body->size() != workload->request_bytes ||
        std::memcmp(body->data(), workload->request_body.data(),
                    workload->request_bytes) != 0)
      co_return -1;

    HttpResponseHead response;
    response.headers = kHeaders;
    auto started = co_await connection.start_response(response);
    if (!started)
      co_return -1;
    auto writer = std::move(*started);
    const auto response_bytes = as_bytes(workload->response_body);
    for (std::size_t offset = 0; offset < response_bytes.size();
         offset += workload->response_chunk_bytes) {
      const std::size_t count =
          std::min(workload->response_chunk_bytes,
                   response_bytes.size() - offset);
      Status written = co_await writer.write(response_bytes.subspan(offset,
                                                                     count));
      if (!written.ok())
        co_return -1;
    }
    Status finished = co_await writer.finish();
    if (!finished.ok())
      co_return -1;
  }
  co_return workload->rounds;
}

Task<int> hyle_client(AllocatorRef allocator, TcpStream stream,
                      const Workload *workload) {
  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made)
    co_return -1;
  HttpClientConnection connection = std::move(*made);
  static constexpr HttpHeader kHeaders[] = {
      {StringView("Content-Type"), StringView("application/json")}};

  for (int round = 0; round < workload->rounds; ++round) {
    HttpRequestHead request;
    request.method = "POST";
    request.target = "/v1/completions";
    request.authority = "localhost";
    request.headers = kHeaders;
    auto response =
        co_await connection.request(request, as_bytes(workload->request_body));
    if (!response || response->head.status() != 200)
      co_return -1;
    std::size_t received = 0;
    for (;;) {
      auto chunk = co_await response->body.read(kPeerBodyBufferBytes);
      if (!chunk || received + chunk->bytes.size() >
                        workload->response_bytes ||
          std::memcmp(chunk->bytes.data(),
                      workload->response_body.data() + received,
                      chunk->bytes.size()) != 0)
        co_return -1;
      received += chunk->bytes.size();
      if (chunk->end_stream)
        break;
    }
    if (received != workload->response_bytes)
      co_return -1;
  }
  co_return workload->rounds;
}

class HyleHttpHarness {
public:
  HyleHttpHarness()
      : scheduler_(1, system_allocator(), 1024), io_(scheduler_) {
    if (!io_.valid())
      std::abort();
  }

  ~HyleHttpHarness() { scheduler_.shutdown(); }

  int run(const Workload &workload) {
    SocketPair pair = make_socket_pair();
    if (!pair.valid())
      return -1;
    auto adopted_client =
        TcpStream::adopt(io_, kOwner, pair.release_client());
    if (!adopted_client)
      return -1;
    auto adopted_server =
        TcpStream::adopt(io_, kOwner, pair.release_server());
    if (!adopted_server)
      return -1;
    TcpStream client = std::move(*adopted_client);
    TcpStream server = std::move(*adopted_server);

    auto server_join = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_server(system_allocator(), std::move(server), &workload));
    auto client_join = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_client(system_allocator(), std::move(client), &workload));
    auto client_result = client_join.join();
    auto server_result = server_join.join();
    if (!client_result || !server_result || *client_result != workload.rounds ||
        *server_result != workload.rounds)
      return -1;
    return workload.rounds;
  }

private:
  Scheduler scheduler_;
  IoContext io_;
};

using PeerSocket = asio::local::stream_protocol::socket;

struct BeastEpisode;

asio::awaitable<int> beast_server(std::shared_ptr<BeastEpisode> episode);
asio::awaitable<int> beast_client(std::shared_ptr<BeastEpisode> episode);

struct BeastEpisode : std::enable_shared_from_this<BeastEpisode> {
  BeastEpisode(asio::io_context &context, SocketPair pair,
               const Workload *selected)
      : server(context), client(context), workload(selected) {
    boost::system::error_code error;
    server.assign(asio::local::stream_protocol(), pair.release_server(),
                  error);
    if (error)
      std::abort();
    client.assign(asio::local::stream_protocol(), pair.release_client(),
                  error);
    if (error)
      std::abort();
  }

  void start() {
    auto self = shared_from_this();
    asio::co_spawn(
        server.get_executor(), beast_server(self),
        [self](std::exception_ptr error, int value) {
          self->complete(true, error, value);
        });
    asio::co_spawn(
        client.get_executor(), beast_client(self),
        [self](std::exception_ptr error, int value) {
          self->complete(false, error, value);
        });
  }

  void complete(bool server_side, const std::exception_ptr &error, int value) {
    if (error != nullptr)
      value = -1;
    if (server_side)
      server_result = value;
    else
      client_result = value;
    if (value < 0 && !failed) {
      failed = true;
      boost::system::error_code ignored;
      server.close(ignored);
      client.close(ignored);
    }
    if (++completions != 2)
      return;
    boost::system::error_code ignored;
    server.close(ignored);
    client.close(ignored);
    result.set_value(!failed && server_result == workload->rounds &&
                             client_result == workload->rounds
                         ? workload->rounds
                         : -1);
  }

  PeerSocket server;
  PeerSocket client;
  const Workload *workload = nullptr;
  std::promise<int> result;
  int server_result = -1;
  int client_result = -1;
  int completions = 0;
  bool failed = false;
};

asio::awaitable<int> beast_server(std::shared_ptr<BeastEpisode> episode) {
  beast::flat_buffer input(16 * 1024);
  const Workload &workload = *episode->workload;
  for (int round = 0; round < workload.rounds; ++round) {
    http::request_parser<http::string_body> parser;
    parser.header_limit(16 * 1024);
    parser.body_limit(workload.request_bytes);
    boost::system::error_code error;
    co_await http::async_read(episode->server, input, parser,
                              asio::redirect_error(asio::use_awaitable, error));
    if (error)
      co_return -1;
    const auto &request = parser.get();
    if (request.method() != http::verb::post ||
        request.target() != "/v1/completions" ||
        request.body() != workload.request_body || !request.keep_alive())
      co_return -1;

    http::response<http::empty_body> response{http::status::ok, 11};
    response.set(http::field::content_type, "text/event-stream");
    response.keep_alive(true);
    response.chunked(true);
    http::response_serializer<http::empty_body> serializer(response);
    co_await http::async_write_header(
        episode->server, serializer,
        asio::redirect_error(asio::use_awaitable, error));
    if (error)
      co_return -1;

    for (std::size_t offset = 0; offset < workload.response_bytes;
         offset += workload.response_chunk_bytes) {
      const std::size_t count =
          std::min(workload.response_chunk_bytes,
                   workload.response_bytes - offset);
      auto chunk = http::make_chunk(
          asio::buffer(workload.response_body.data() + offset, count));
      co_await asio::async_write(
          episode->server, chunk,
          asio::redirect_error(asio::use_awaitable, error));
      if (error)
        co_return -1;
    }
    auto last = http::make_chunk_last();
    co_await asio::async_write(
        episode->server, last,
        asio::redirect_error(asio::use_awaitable, error));
    if (error)
      co_return -1;
  }
  co_return workload.rounds;
}

asio::awaitable<int> beast_client(std::shared_ptr<BeastEpisode> episode) {
  beast::flat_buffer input(16 * 1024);
  const Workload &workload = *episode->workload;
  std::array<char, kPeerBodyBufferBytes> body_buffer{};
  for (int round = 0; round < workload.rounds; ++round) {
    http::request<http::buffer_body> request{http::verb::post,
                                             "/v1/completions", 11};
    request.set(http::field::host, "localhost");
    request.set(http::field::content_type, "application/json");
    request.keep_alive(true);
    request.content_length(workload.request_bytes);
    request.body().data =
        const_cast<char *>(workload.request_body.data()); // serializer borrows
    request.body().size = workload.request_bytes;
    request.body().more = false;
    http::request_serializer<http::buffer_body> serializer(request);
    boost::system::error_code error;
    co_await http::async_write(
        episode->client, serializer,
        asio::redirect_error(asio::use_awaitable, error));
    if (error)
      co_return -1;

    http::response_parser<http::buffer_body> parser;
    parser.header_limit(16 * 1024);
    parser.body_limit(workload.response_bytes);
    parser.eager(true);
    std::size_t received = 0;
    while (!parser.is_done()) {
      parser.get().body().data = body_buffer.data();
      parser.get().body().size = body_buffer.size();
      const std::size_t capacity = parser.get().body().size;
      error.clear();
      co_await http::async_read_some(
          episode->client, input, parser,
          asio::redirect_error(asio::use_awaitable, error));
      if (error && error != http::error::need_buffer)
        co_return -1;
      const std::size_t produced = capacity - parser.get().body().size;
      if (received + produced > workload.response_bytes ||
          std::memcmp(body_buffer.data(),
                      workload.response_body.data() + received, produced) != 0)
        co_return -1;
      received += produced;
    }
    if (parser.get().result() != http::status::ok ||
        !parser.get().chunked() || !parser.get().keep_alive() ||
        received != workload.response_bytes)
      co_return -1;
  }
  co_return workload.rounds;
}

class BeastHttpHarness {
public:
  BeastHttpHarness()
      : context_(1), work_(asio::make_work_guard(context_)),
        worker_([this] { context_.run(); }) {}

  ~BeastHttpHarness() {
    work_.reset();
    context_.stop();
    if (worker_.joinable())
      worker_.join();
  }

  int run(const Workload &workload) {
    SocketPair pair = make_socket_pair();
    if (!pair.valid())
      return -1;
    auto episode =
        std::make_shared<BeastEpisode>(context_, std::move(pair), &workload);
    auto result = episode->result.get_future();
    asio::post(context_, [episode] { episode->start(); });
    return result.get();
  }

private:
  asio::io_context context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_;
  std::thread worker_;
};

void run_workload(const Workload &workload, HyleHttpHarness &hyle,
                  BeastHttpHarness &peer, bool peer_first) {
  auto run_hyle = [&] {
    if (hyle.run(workload) != workload.rounds)
      std::abort();
  };
  auto run_peer = [&] {
    if (peer.run(workload) != workload.rounds)
      std::abort();
  };

  if (peer_first) {
    run_peer();
    run_hyle();
  } else {
    run_hyle();
    run_peer();
  }

  ankerl::nanobench::Bench bench;
  bench.title(workload.title)
      .relative(true)
      .batch(static_cast<std::uint64_t>(workload.rounds))
      .unit("request")
      .minEpochIterations(10)
      .epochs(11);
  if (peer_first) {
    bench.run("Boost.Beast HTTP/1 persistent runtime", run_peer);
    bench.run("HYLE HTTP/1 persistent runtime", run_hyle);
  } else {
    bench.run("HYLE HTTP/1 persistent runtime", run_hyle);
    bench.run("Boost.Beast HTTP/1 persistent runtime", run_peer);
  }
}

} // namespace

int main() {
  std::signal(SIGPIPE, SIG_IGN);
  const bool peer_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  const char *run_nonce = std::getenv("HYLE_PERF_RUN_NONCE");
  std::fprintf(stderr, "# HYLE_IO_HTTP_RUN execution_order=%s run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               run_nonce != nullptr ? run_nonce : "unmanaged");

  const Workload normal = make_workload(
      "HTTP/1 keep-alive REST + chunked 1KiB", 128, 1024, 1024, 128);
  const Workload holdout = make_workload(
      "HTTP/1 fragmented holdout 64KiB", 16, 64 * 1024, 64 * 1024, 4096);
  HyleHttpHarness hyle;
  BeastHttpHarness peer;
  run_workload(normal, hyle, peer, peer_first);
  run_workload(holdout, hyle, peer, peer_first);
  return 0;
}
