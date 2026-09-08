// Product-shaped io_transfer P1 holdout.
//
// Both routes fetch a 64 MiB identity-encoded artifact from a loopback HTTP/1
// server, interrupt after reaching 32 MiB, retain the sibling partial, resume
// with Range + If-Range, validate the 206 extent, sync and atomically publish.
// The peer is libcurl's public easy API, not a benchmark-local raw parser.

#if !defined(__linux__)
#error "io_transfer P1 currently has a Linux/WSL performance contract"
#endif

#include <nanobench.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <curl/curl.h>

#include "hyle/io/io_context.h"
#include "hyle/io/transfer.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

using hyle::Status;
using hyle::io::AtomicFileDestination;
using hyle::io::HttpTransferSource;
using hyle::io::IoContext;
using hyle::io::TransferDisposition;
using hyle::io::TransferExecution;
using hyle::io::TransferOptions;
using hyle::io::TransferPlan;
using hyle::io::TransferProgress;
using hyle::io::TransferProgressObserver;
using hyle::io::TransferReport;
using hyle::mem::system_allocator;
using hyle::sched::Scheduler;

constexpr unsigned kOwner = 0;
constexpr std::size_t kBufferBytes = 1024u * 1024u;
constexpr std::uint64_t kArtifactBytes = 64ull * 1024ull * 1024ull;
constexpr std::uint64_t kPrefixBytes = 32ull * 1024ull * 1024ull;
constexpr std::string_view kEtag = "\"hyle-io-transfer-p1\"";

[[noreturn]] void fail(const char *operation) noexcept {
  std::fprintf(stderr, "hyle_io_transfer_resume_p1_bench: %s failed (%d: %s)\n",
               operation, errno, std::strerror(errno));
  std::abort();
}

void require(bool condition, const char *operation) noexcept {
  if (!condition)
    fail(operation);
}

bool environment_present(const char *name) noexcept {
  return std::getenv(name) != nullptr;
}

void close_descriptor(int descriptor) noexcept {
  if (descriptor >= 0)
    (void)::close(descriptor);
}

bool send_all(int descriptor, const void *data, std::size_t size) noexcept {
  const auto *cursor = static_cast<const std::byte *>(data);
  while (size != 0) {
    const ssize_t result = ::send(descriptor, cursor, size, MSG_NOSIGNAL);
    if (result < 0 && errno == EINTR)
      continue;
    if (result <= 0)
      return false;
    cursor += result;
    size -= static_cast<std::size_t>(result);
  }
  return true;
}

void pwrite_all(int descriptor, const std::byte *data, std::size_t size,
                std::uint64_t offset) noexcept {
  while (size != 0) {
    const ssize_t result =
        ::pwrite(descriptor, data, size, static_cast<off_t>(offset));
    if (result < 0 && errno == EINTR)
      continue;
    require(result > 0, "pwrite peer partial");
    data += result;
    size -= static_cast<std::size_t>(result);
    offset += static_cast<std::uint64_t>(result);
  }
}

class RangeServer {
public:
  RangeServer() {
    listener_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    require(listener_ >= 0, "create range listener");
    int enabled = 1;
    require(::setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &enabled,
                         sizeof(enabled)) == 0,
            "set range listener reuseaddr");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    require(::bind(listener_, reinterpret_cast<sockaddr *>(&address),
                   sizeof(address)) == 0,
            "bind range listener");
    require(::listen(listener_, 16) == 0, "listen range server");
    socklen_t size = sizeof(address);
    require(::getsockname(listener_, reinterpret_cast<sockaddr *>(&address),
                          &size) == 0,
            "inspect range listener");
    port_ = ntohs(address.sin_port);
    worker_ = std::thread([this] { serve(); });
  }

  ~RangeServer() {
    stopping_.store(true, std::memory_order_release);
    (void)::shutdown(listener_, SHUT_RDWR);
    close_descriptor(listener_);
    listener_ = -1;
    if (worker_.joinable())
      worker_.join();
  }

  RangeServer(const RangeServer &) = delete;
  RangeServer &operator=(const RangeServer &) = delete;

  std::uint16_t port() const noexcept { return port_; }

private:
  void serve() noexcept {
    while (!stopping_.load(std::memory_order_acquire)) {
      const int client = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
      if (client < 0) {
        if (errno == EINTR)
          continue;
        if (stopping_.load(std::memory_order_acquire))
          return;
        fail("accept range request");
      }
      serve_one(client);
      close_descriptor(client);
    }
  }

  static void serve_one(int client) noexcept {
    std::string request;
    request.reserve(4096);
    std::array<char, 4096> input{};
    while (request.find("\r\n\r\n") == std::string::npos) {
      const ssize_t result = ::recv(client, input.data(), input.size(), 0);
      if (result < 0 && errno == EINTR)
        continue;
      if (result <= 0)
        return;
      request.append(input.data(), static_cast<std::size_t>(result));
      if (request.size() > 16 * 1024)
        return;
    }

    constexpr std::string_view range_prefix = "Range: bytes=";
    const std::string if_range = "If-Range: " + std::string(kEtag);
    const std::size_t range_position = request.find(range_prefix);
    std::uint64_t requested_first = 0;
    bool valid_range = false;
    if (range_position != std::string::npos) {
      const std::size_t line_end_position =
          request.find("\r\n", range_position);
      const char *first_digit =
          request.data() + range_position + range_prefix.size();
      if (line_end_position != std::string::npos) {
        const char *line_end = request.data() + line_end_position;
        auto parsed = std::from_chars(first_digit, line_end, requested_first);
        valid_range = parsed.ec == std::errc{} && parsed.ptr != first_digit &&
                      parsed.ptr != line_end && *parsed.ptr == '-' &&
                      requested_first < kArtifactBytes;
      }
    }
    const bool resumed =
        valid_range && request.find(if_range) != std::string::npos;
    const std::uint64_t first = resumed ? requested_first : 0;
    const std::uint64_t body_bytes = kArtifactBytes - first;
    std::array<char, 512> header{};
    const int count =
        resumed
            ? std::snprintf(header.data(), header.size(),
                            "HTTP/1.1 206 Partial Content\r\n"
                            "Content-Length: %llu\r\n"
                            "Content-Range: bytes %llu-%llu/%llu\r\n"
                            "ETag: %.*s\r\n"
                            "Accept-Ranges: bytes\r\n"
                            "Connection: close\r\n\r\n",
                            static_cast<unsigned long long>(body_bytes),
                            static_cast<unsigned long long>(first),
                            static_cast<unsigned long long>(kArtifactBytes - 1),
                            static_cast<unsigned long long>(kArtifactBytes),
                            static_cast<int>(kEtag.size()), kEtag.data())
            : std::snprintf(header.data(), header.size(),
                            "HTTP/1.1 200 OK\r\n"
                            "Content-Length: %llu\r\n"
                            "ETag: %.*s\r\n"
                            "Accept-Ranges: bytes\r\n"
                            "Connection: close\r\n\r\n",
                            static_cast<unsigned long long>(body_bytes),
                            static_cast<int>(kEtag.size()), kEtag.data());
    if (count <= 0 || static_cast<std::size_t>(count) >= header.size() ||
        !send_all(client, header.data(), static_cast<std::size_t>(count))) {
      return;
    }

    std::array<std::byte, kBufferBytes> zeros{};
    std::uint64_t remaining = body_bytes;
    while (remaining != 0) {
      const std::size_t chunk = static_cast<std::size_t>(
          std::min<std::uint64_t>(zeros.size(), remaining));
      if (!send_all(client, zeros.data(), chunk))
        return;
      remaining -= chunk;
    }
  }

  int listener_ = -1;
  std::uint16_t port_ = 0;
  std::atomic<bool> stopping_{false};
  std::thread worker_;
};

class Files {
public:
  Files() {
    std::array<char, 96> pattern{};
    std::snprintf(pattern.data(), pattern.size(),
                  "/tmp/hyle_io_transfer_resume_p1_%ld_XXXXXX",
                  static_cast<long>(::getpid()));
    char *made = ::mkdtemp(pattern.data());
    require(made != nullptr, "mkdtemp");
    directory_ = made;
    candidate_ = directory_ + "/candidate.gguf";
    peer_ = directory_ + "/peer.gguf";
    peer_partial_ = directory_ + "/peer.partial";
    write_old(candidate_);
    write_old(peer_);
  }

  ~Files() {
    (void)::unlink(peer_partial_.c_str());
    (void)::unlink(candidate_.c_str());
    (void)::unlink(peer_.c_str());
    (void)::rmdir(directory_.c_str());
  }

  const std::string &candidate() const noexcept { return candidate_; }
  const std::string &peer() const noexcept { return peer_; }
  const std::string &peer_partial() const noexcept { return peer_partial_; }

private:
  static void write_old(const std::string &path) noexcept {
    const int descriptor =
        ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
    require(descriptor >= 0, "open old resume destination");
    static constexpr std::string_view old = "old-model";
    require(::write(descriptor, old.data(), old.size()) ==
                static_cast<ssize_t>(old.size()),
            "write old resume destination");
    require(::fsync(descriptor) == 0, "fsync old resume destination");
    close_descriptor(descriptor);
  }

  std::string directory_;
  std::string candidate_;
  std::string peer_;
  std::string peer_partial_;
};

struct StopAfterPrefix {
  Status operator()(const TransferProgress &progress) noexcept {
    if (progress.artifact_bytes >= kPrefixBytes)
      return Status::Cancelled("P1 intentional interruption");
    return {};
  }
};

class HyleHarness {
public:
  HyleHarness() : scheduler_(1, system_allocator(), 1024), io_(scheduler_) {
    require(io_.valid(), "construct HYLE resume runtime");
  }

  ~HyleHarness() { scheduler_.shutdown(); }

  void run(const Files &files, std::uint16_t port) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port) + "/model.gguf";
    TransferOptions options;
    options.limits.buffer_bytes = kBufferBytes;
    options.limits.max_bytes = kArtifactBytes;
    auto initial =
        TransferPlan::http(system_allocator(), HttpTransferSource{url},
                           AtomicFileDestination{files.candidate()}, options);
    require(initial.has_value(), "construct initial HYLE HTTP transfer");
    StopAfterPrefix stop;
    TransferExecution interrupted;
    interrupted.owner_worker = kOwner;
    interrupted.progress = TransferProgressObserver::from(stop);
    TransferReport first = run_plan(std::move(*initial), interrupted);
    const std::uint64_t retained_bytes = first.checkpoint.partial_bytes();
    const bool retained =
        first.status.IsCancelled() &&
        first.disposition == TransferDisposition::kPartialRetained &&
        first.checkpoint.valid() && retained_bytes >= kPrefixBytes &&
        retained_bytes < kPrefixBytes + kBufferBytes;
    if (!retained) {
      const auto message = first.status.message();
      std::fprintf(
          stderr,
          "# HYLE_IO_TRANSFER_RESUME_FAILURE code=%u message=%.*s "
          "disposition=%u checkpoint=%s partial=%llu attempt=%llu\n",
          static_cast<unsigned>(first.status.code()),
          static_cast<int>(message.size()), message.data(),
          static_cast<unsigned>(first.disposition),
          first.checkpoint.valid() ? "valid" : "invalid",
          static_cast<unsigned long long>(first.checkpoint.partial_bytes()),
          static_cast<unsigned long long>(first.bytes_this_attempt));
      fail("retain exact HYLE HTTP prefix");
    }

    auto resumed = TransferPlan::resume(std::move(first.checkpoint));
    require(resumed.has_value(), "construct HYLE resume plan");
    TransferExecution continuation;
    continuation.owner_worker = kOwner;
    TransferReport second = run_plan(std::move(*resumed), continuation);
    require(second.published() && second.resumed && !second.restarted &&
                second.artifact_bytes == kArtifactBytes &&
                second.bytes_this_attempt == kArtifactBytes - retained_bytes,
            "complete HYLE HTTP range resume");
    nb::doNotOptimizeAway(second.artifact_bytes);
  }

private:
  TransferReport run_plan(TransferPlan plan, TransferExecution execution) {
    auto joined =
        hyle::sched::spawn_task(scheduler_, system_allocator(),
                                hyle::io::transfer(system_allocator(), io_,
                                                   std::move(plan), execution));
    auto result = joined.join();
    require(result.has_value(), "join HYLE HTTP transfer");
    return std::move(*result);
  }

  Scheduler scheduler_;
  IoContext io_;
};

class CurlGlobal {
public:
  CurlGlobal() {
    require(::curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK,
            "initialize libcurl");
  }
  ~CurlGlobal() { ::curl_global_cleanup(); }
  CurlGlobal(const CurlGlobal &) = delete;
  CurlGlobal &operator=(const CurlGlobal &) = delete;
};

struct CurlWriteState {
  int destination = -1;
  std::uint64_t offset = 0;
  std::uint64_t limit = 0;
  std::uint64_t written = 0;
};

std::size_t curl_write(char *data, std::size_t size, std::size_t count,
                       void *context) noexcept {
  auto *state = static_cast<CurlWriteState *>(context);
  if (size != 0 && count > (std::numeric_limits<std::size_t>::max)() / size)
    return 0;
  const std::size_t available = size * count;
  const std::size_t accepted = static_cast<std::size_t>(
      std::min<std::uint64_t>(available, state->limit - state->written));
  if (accepted != 0) {
    pwrite_all(state->destination, reinterpret_cast<const std::byte *>(data),
               accepted, state->offset + state->written);
    state->written += accepted;
  }
  // A short return is libcurl's public write-callback cancellation signal.
  return accepted;
}

std::size_t curl_header(char *data, std::size_t size, std::size_t count,
                        void *context) noexcept {
  if (size != 0 && count > (std::numeric_limits<std::size_t>::max)() / size)
    return 0;
  const std::size_t bytes = size * count;
  static_cast<std::string *>(context)->append(data, bytes);
  return bytes;
}

struct CurlResult {
  CURLcode code = CURLE_OK;
  long status = 0;
  std::string headers;
  std::uint64_t written = 0;
};

CurlResult curl_get(std::string_view url, int destination,
                    std::uint64_t destination_offset,
                    std::uint64_t bytes_to_write,
                    std::optional<std::uint64_t> resume_offset) {
  CURL *easy = ::curl_easy_init();
  require(easy != nullptr, "construct libcurl easy handle");
  CurlResult result;
  CurlWriteState write{.destination = destination,
                       .offset = destination_offset,
                       .limit = bytes_to_write};
  curl_slist *headers = nullptr;
  if (resume_offset.has_value()) {
    const std::string if_range = "If-Range: " + std::string(kEtag);
    headers = ::curl_slist_append(headers, if_range.c_str());
    require(headers != nullptr, "construct libcurl If-Range header");
  }
  auto set = [&](CURLoption option, auto value) noexcept {
    require(::curl_easy_setopt(easy, option, value) == CURLE_OK,
            "configure libcurl easy handle");
  };
  const std::string owned_url(url);
  set(CURLOPT_URL, owned_url.c_str());
  set(CURLOPT_HTTP_VERSION, static_cast<long>(CURL_HTTP_VERSION_1_1));
  set(CURLOPT_ACCEPT_ENCODING, "identity");
  set(CURLOPT_FOLLOWLOCATION, 1L);
  set(CURLOPT_MAXREDIRS, 5L);
  set(CURLOPT_NOSIGNAL, 1L);
  set(CURLOPT_CONNECTTIMEOUT_MS, 5000L);
  set(CURLOPT_TIMEOUT_MS, 60000L);
  set(CURLOPT_BUFFERSIZE, static_cast<long>(kBufferBytes));
  set(CURLOPT_WRITEFUNCTION, &curl_write);
  set(CURLOPT_WRITEDATA, &write);
  set(CURLOPT_HEADERFUNCTION, &curl_header);
  set(CURLOPT_HEADERDATA, &result.headers);
  if (headers != nullptr)
    set(CURLOPT_HTTPHEADER, headers);
  if (resume_offset.has_value()) {
    set(CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(*resume_offset));
  }
  result.code = ::curl_easy_perform(easy);
  require(::curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &result.status) ==
              CURLE_OK,
          "inspect libcurl response status");
  result.written = write.written;
  ::curl_easy_cleanup(easy);
  ::curl_slist_free_all(headers);
  return result;
}

void run_curl_reference(const Files &files, std::uint16_t port) {
  const int destination = ::open(files.peer_partial().c_str(),
                                 O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0600);
  require(destination >= 0, "open libcurl HTTP partial");
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.gguf";
  CurlResult first = curl_get(url, destination, 0, kPrefixBytes, std::nullopt);
  require(first.code == CURLE_WRITE_ERROR && first.status == 200 &&
              first.written == kPrefixBytes &&
              first.headers.find("ETag: " + std::string(kEtag)) !=
                  std::string::npos,
          "interrupt libcurl transfer at prefix");
  CurlResult second = curl_get(url, destination, kPrefixBytes,
                               kArtifactBytes - kPrefixBytes, kPrefixBytes);
  const std::string content_range =
      "Content-Range: bytes " + std::to_string(kPrefixBytes) + "-" +
      std::to_string(kArtifactBytes - 1) + "/" + std::to_string(kArtifactBytes);
  require(second.code == CURLE_OK && second.status == 206 &&
              second.written == kArtifactBytes - kPrefixBytes &&
              second.headers.find("ETag: " + std::string(kEtag)) !=
                  std::string::npos &&
              second.headers.find(content_range) != std::string::npos,
          "complete libcurl HTTP Range resume");
  require(::fsync(destination) == 0, "fsync libcurl HTTP partial");
  close_descriptor(destination);
  require(::rename(files.peer_partial().c_str(), files.peer().c_str()) == 0,
          "publish libcurl HTTP destination");
  nb::doNotOptimizeAway(port);
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first, Candidate &&candidate,
              Peer &&peer) {
  if (peer_first) {
    peer();
    candidate();
    bench.run("libcurl HTTP Range resume 64MiB", std::forward<Peer>(peer));
    bench.run("HYLE HTTP Range resume 64MiB",
              std::forward<Candidate>(candidate));
  } else {
    candidate();
    peer();
    bench.run("HYLE HTTP Range resume 64MiB",
              std::forward<Candidate>(candidate));
    bench.run("libcurl HTTP Range resume 64MiB", std::forward<Peer>(peer));
  }
}

} // namespace

int main() {
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_IO_TRANSFER_RESUME_RUN execution_order=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", affinity.topology,
               affinity.cpu, affinity.ok, hyle::bench::perf::run_nonce());

  CurlGlobal curl;
  RangeServer server;
  Files files;
  HyleHarness hyle;
  auto candidate = [&] { hyle.run(files, server.port()); };
  auto peer = [&] { run_curl_reference(files, server.port()); };

  nb::Bench bench;
  bench.title("interrupted HTTP Range model transfer 64MiB")
      .relative(true)
      .unit("transfer")
      .batch(1)
      .epochIterations(1)
      .epochs(3);
  run_pair(bench, peer_first, candidate, peer);
}
