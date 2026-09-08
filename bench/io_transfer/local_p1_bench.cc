// Product-shaped io_transfer P1 primary.
//
// The candidate and libcurl's public file:// route both copy one complete
// sparse model artifact
// through a bounded 1 MiB buffer, incrementally verify every byte, sync file
// data + metadata, close the temporary file and atomically replace an existing
// sibling destination. Direct POSIX was used only as a pre-candidate cost-floor
// diagnostic and is deliberately absent from the product verdict.

#if !defined(__linux__)
#error "io_transfer P1 currently has a Linux/WSL performance contract"
#endif

#include <nanobench.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
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
using hyle::io::IoContext;
using hyle::io::LocalTransferSource;
using hyle::io::TransferExecution;
using hyle::io::TransferOptions;
using hyle::io::TransferPlan;
using hyle::io::TransferReport;
using hyle::io::TransferVerifier;
using hyle::mem::system_allocator;
using hyle::sched::Scheduler;

constexpr unsigned kOwner = 0;
constexpr std::size_t kBufferBytes = 1024u * 1024u;
constexpr std::uint64_t kArtifactBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

[[noreturn]] void fail(const char *operation) noexcept {
  std::fprintf(stderr, "hyle_io_transfer_local_p1_bench: %s failed (%d: %s)\n",
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

void close_checked(int descriptor, const char *operation) noexcept {
  if (::close(descriptor) != 0)
    fail(operation);
}

void write_all(int descriptor, const void *data, std::size_t size) noexcept {
  const auto *cursor = static_cast<const std::byte *>(data);
  while (size != 0) {
    const ssize_t result = ::write(descriptor, cursor, size);
    if (result < 0 && errno == EINTR)
      continue;
    require(result > 0, "write");
    cursor += result;
    size -= static_cast<std::size_t>(result);
  }
}

void pwrite_all(int descriptor, const std::byte *data, std::size_t size,
                std::uint64_t offset) noexcept {
  while (size != 0) {
    const ssize_t result =
        ::pwrite(descriptor, data, size, static_cast<off_t>(offset));
    if (result < 0 && errno == EINTR)
      continue;
    require(result > 0, "pwrite");
    data += result;
    size -= static_cast<std::size_t>(result);
    offset += static_cast<std::uint64_t>(result);
  }
}

void update_fnv(std::uint64_t &hash,
                std::span<const std::byte> bytes) noexcept {
  for (const std::byte byte : bytes) {
    hash ^= std::to_integer<unsigned char>(byte);
    hash *= kFnvPrime;
  }
}

std::uint64_t zero_digest() noexcept {
  std::array<std::byte, kBufferBytes> zeros{};
  std::uint64_t hash = kFnvOffset;
  for (std::uint64_t offset = 0; offset < kArtifactBytes;
       offset += zeros.size()) {
    update_fnv(hash, zeros);
  }
  return hash;
}

struct FnvVerifier {
  std::uint64_t expected = 0;
  std::uint64_t hash = kFnvOffset;

  Status reset() noexcept {
    hash = kFnvOffset;
    return {};
  }

  Status update(std::span<const std::byte> bytes) noexcept {
    update_fnv(hash, bytes);
    return {};
  }

  Status finish(std::uint64_t size) noexcept {
    if (size != kArtifactBytes || hash != expected)
      return Status::DataLoss("P1 transfer checksum mismatch");
    return {};
  }
};

class Files {
public:
  Files() {
    std::array<char, 96> pattern{};
    std::snprintf(pattern.data(), pattern.size(),
                  "/tmp/hyle_io_transfer_local_p1_%ld_XXXXXX",
                  static_cast<long>(::getpid()));
    char *made = ::mkdtemp(pattern.data());
    require(made != nullptr, "mkdtemp");
    directory_ = made;
    source_ = directory_ + "/source.gguf";
    candidate_ = directory_ + "/candidate.gguf";
    peer_ = directory_ + "/peer.gguf";
    peer_partial_ = directory_ + "/peer.partial";

    const int source =
        ::open(source_.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    require(source >= 0, "open sparse source");
    require(::ftruncate(source, static_cast<off_t>(kArtifactBytes)) == 0,
            "ftruncate sparse source");
    require(::fsync(source) == 0, "fsync sparse source");
    close_checked(source, "close sparse source");
    write_old(candidate_);
    write_old(peer_);
    expected_ = zero_digest();
  }

  ~Files() {
    (void)::unlink(peer_partial_.c_str());
    (void)::unlink(candidate_.c_str());
    (void)::unlink(peer_.c_str());
    (void)::unlink(source_.c_str());
    (void)::rmdir(directory_.c_str());
  }

  Files(const Files &) = delete;
  Files &operator=(const Files &) = delete;

  const std::string &source() const noexcept { return source_; }
  const std::string &candidate() const noexcept { return candidate_; }
  const std::string &peer() const noexcept { return peer_; }
  const std::string &peer_partial() const noexcept { return peer_partial_; }
  std::uint64_t expected() const noexcept { return expected_; }

private:
  static void write_old(const std::string &path) noexcept {
    const int descriptor =
        ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
    require(descriptor >= 0, "open old destination");
    static constexpr std::string_view old = "old-model";
    write_all(descriptor, old.data(), old.size());
    require(::fsync(descriptor) == 0, "fsync old destination");
    close_checked(descriptor, "close old destination");
  }

  std::string directory_;
  std::string source_;
  std::string candidate_;
  std::string peer_;
  std::string peer_partial_;
  std::uint64_t expected_ = 0;
};

class HyleHarness {
public:
  HyleHarness() : scheduler_(1, system_allocator(), 1024), io_(scheduler_) {
    require(io_.valid(), "construct HYLE IO runtime");
  }

  ~HyleHarness() { scheduler_.shutdown(); }

  void run(const Files &files) {
    TransferOptions options;
    options.limits.buffer_bytes = kBufferBytes;
    options.limits.max_bytes = kArtifactBytes;
    auto plan = TransferPlan::local(
        system_allocator(), LocalTransferSource{files.source()},
        AtomicFileDestination{files.candidate()}, options);
    require(plan.has_value(), "construct HYLE transfer plan");

    FnvVerifier verifier{.expected = files.expected()};
    TransferExecution execution;
    execution.owner_worker = kOwner;
    execution.verifier = TransferVerifier::from(verifier);
    auto joined = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle::io::transfer(system_allocator(), io_, std::move(*plan),
                           execution));
    auto result = joined.join();
    require(result.has_value(), "join HYLE transfer");
    TransferReport report = std::move(*result);
    require(report.published() && report.artifact_bytes == kArtifactBytes,
            "complete HYLE transfer");
    nb::doNotOptimizeAway(report.artifact_bytes);
  }

private:
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

bool same_identity(const struct stat &before,
                   const struct stat &after) noexcept {
  return S_ISREG(before.st_mode) && S_ISREG(after.st_mode) &&
         before.st_size == after.st_size &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec;
}

struct CurlWriteState {
  int destination = -1;
  std::uint64_t offset = 0;
  std::uint64_t hash = kFnvOffset;
};

std::size_t curl_write(char *data, std::size_t size, std::size_t count,
                       void *context) noexcept {
  if (size != 0 && count > (std::numeric_limits<std::size_t>::max)() / size)
    return 0;
  const std::size_t bytes = size * count;
  auto *state = static_cast<CurlWriteState *>(context);
  const auto *input = reinterpret_cast<const std::byte *>(data);
  update_fnv(state->hash, std::span<const std::byte>(input, bytes));
  pwrite_all(state->destination, input, bytes, state->offset);
  state->offset += bytes;
  return bytes;
}

void run_curl_reference(const Files &files) {
  const int source = ::open(files.source().c_str(), O_RDONLY | O_CLOEXEC);
  require(source >= 0, "open libcurl source identity");
  struct stat before{};
  require(::fstat(source, &before) == 0 &&
              before.st_size == static_cast<off_t>(kArtifactBytes),
          "stat libcurl source identity");
  const int destination = ::open(files.peer_partial().c_str(),
                                 O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0600);
  require(destination >= 0, "open libcurl partial");

  CURL *easy = ::curl_easy_init();
  require(easy != nullptr, "construct libcurl easy handle");
  CurlWriteState write{.destination = destination};
  const std::string url = "file://" + files.source();
  auto set = [&](CURLoption option, auto value) noexcept {
    require(::curl_easy_setopt(easy, option, value) == CURLE_OK,
            "configure libcurl file transfer");
  };
  set(CURLOPT_URL, url.c_str());
  set(CURLOPT_PROTOCOLS_STR, "file");
  set(CURLOPT_NOSIGNAL, 1L);
  set(CURLOPT_BUFFERSIZE, static_cast<long>(kBufferBytes));
  set(CURLOPT_WRITEFUNCTION, &curl_write);
  set(CURLOPT_WRITEDATA, &write);
  require(::curl_easy_perform(easy) == CURLE_OK,
          "complete libcurl file transfer");
  ::curl_easy_cleanup(easy);

  struct stat after{};
  require(::fstat(source, &after) == 0 && same_identity(before, after),
          "recheck libcurl source identity");
  close_checked(source, "close libcurl source identity");
  require(write.offset == kArtifactBytes && write.hash == files.expected(),
          "verify libcurl checksum");
  require(::fsync(destination) == 0, "fsync libcurl partial");
  close_checked(destination, "close libcurl partial");
  require(::rename(files.peer_partial().c_str(), files.peer().c_str()) == 0,
          "publish libcurl destination");
  nb::doNotOptimizeAway(write.offset);
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first, Candidate &&candidate,
              Peer &&peer) {
  if (peer_first) {
    peer();
    candidate();
    bench.run("libcurl verified atomic transfer 256MiB",
              std::forward<Peer>(peer));
    bench.run("HYLE verified atomic transfer 256MiB",
              std::forward<Candidate>(candidate));
  } else {
    candidate();
    peer();
    bench.run("HYLE verified atomic transfer 256MiB",
              std::forward<Candidate>(candidate));
    bench.run("libcurl verified atomic transfer 256MiB",
              std::forward<Peer>(peer));
  }
}

} // namespace

int main() {
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_IO_TRANSFER_LOCAL_RUN execution_order=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first", affinity.topology,
               affinity.cpu, affinity.ok, hyle::bench::perf::run_nonce());

  CurlGlobal curl;
  Files files;
  HyleHarness hyle;
  auto candidate = [&] { hyle.run(files); };
  auto peer = [&] { run_curl_reference(files); };

  nb::Bench bench;
  bench.title("verified atomic model transfer 256MiB")
      .relative(true)
      .unit("transfer")
      .batch(1)
      .epochIterations(1)
      .epochs(3);
  run_pair(bench, peer_first, candidate, peer);
}
