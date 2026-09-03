#include <nanobench.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

#include "hyle/platform/fs.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kFileBytes = 4u * 1024u * 1024u;

[[noreturn]] void fail(const char* operation) {
  std::fprintf(stderr, "hyle_platform_fs_consumer_bench: %s failed\n",
               operation);
  std::abort();
}

void require(bool condition, const char* operation) {
  if (!condition) fail(operation);
}

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

std::uint64_t checksum(const std::vector<std::byte>& bytes) noexcept {
  std::uint64_t value = 0;
  for (std::byte byte : bytes) value += std::to_integer<unsigned char>(byte);
  return value;
}

class ModelFile {
 public:
  explicit ModelFile(bool peer_buffer_first) {
    path_ = (std::filesystem::temp_directory_path() /
             ("hyle_platform_fs_" +
              std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
              ".bin"))
                .string();
    std::vector<char> chunk(64u * 1024u);
    for (std::size_t index = 0; index < chunk.size(); ++index) {
      chunk[index] = static_cast<char>(index & 0x7f);
    }
    expected_ = 0;
    for (char value : chunk) {
      expected_ += static_cast<unsigned char>(value);
    }
    expected_ *= kFileBytes / chunk.size();

    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    require(output.good(), "create model file");
    for (std::size_t written = 0; written < kFileBytes;
         written += chunk.size()) {
      output.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    }
    require(output.good(), "write model file");
    output.close();

    if (peer_buffer_first) {
      peer_ = std::make_unique<std::vector<std::byte>>(kFileBytes);
      candidate_ = std::make_unique<std::vector<std::byte>>(kFileBytes);
    } else {
      candidate_ = std::make_unique<std::vector<std::byte>>(kFileBytes);
      peer_ = std::make_unique<std::vector<std::byte>>(kFileBytes);
    }
  }

  ~ModelFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void hyle_read() {
    auto file = hyle::platform::open_file(path_.c_str(),
                                         hyle::platform::kOpenRead);
    require(file.has_value(), "open_file");
    require(file->read_exact(candidate_->data(), candidate_->size()).ok(),
            "FileHandle::read_exact");
    require(file->close().ok(), "FileHandle::close");
    require(checksum(*candidate_) == expected_, "HYLE checksum");
    nb::doNotOptimizeAway(candidate_->data());
  }

  void raw_read() {
    std::size_t total = 0;
#if defined(_WIN32)
    wchar_t wide[1024];
    const int count = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                            path_.c_str(), -1, nullptr, 0);
    require(count > 0 && count <= static_cast<int>(std::size(wide)),
            "MultiByteToWideChar(size)");
    require(::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_.c_str(),
                                  -1, wide, count) == count,
            "MultiByteToWideChar(convert)");
    HANDLE file = ::CreateFileW(
        wide, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    require(file != INVALID_HANDLE_VALUE, "CreateFileW");
    while (total < peer_->size()) {
      DWORD got = 0;
      const auto remaining = static_cast<DWORD>(peer_->size() - total);
      require(::ReadFile(file, peer_->data() + total, remaining, &got,
                         nullptr) != 0,
              "ReadFile");
      if (got == 0) break;
      total += got;
    }
    require(::CloseHandle(file) != 0, "CloseHandle");
#else
    const int file = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    require(file >= 0, "open");
    while (total < peer_->size()) {
      const ssize_t got =
          ::read(file, peer_->data() + total, peer_->size() - total);
      if (got < 0 && errno == EINTR) continue;
      require(got >= 0, "read");
      if (got == 0) break;
      total += static_cast<std::size_t>(got);
    }
    require(::close(file) == 0, "close");
#endif
    require(total == peer_->size(), "raw complete read");
    require(checksum(*peer_) == expected_, "raw checksum");
    nb::doNotOptimizeAway(peer_->data());
  }

 private:
  std::string path_;
  std::unique_ptr<std::vector<std::byte>> candidate_;
  std::unique_ptr<std::vector<std::byte>> peer_;
  std::uint64_t expected_ = 0;
};

}  // namespace

int main() {
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(stderr,
               "# HYLE_PLATFORM_FS_RUN execution_order=%s "
               "allocation_order=%s affinity_topology=%s caller_cpu=%s "
               "affinity_ok=%s run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               peer_allocation_first ? "peer-first" : "candidate-first",
               affinity.topology, affinity.cpu, affinity.ok,
               hyle::bench::perf::run_nonce());

  ModelFile model(peer_allocation_first);
  model.hyle_read();
  model.raw_read();
  nb::Bench bench;
  bench.title("warmed model file complete sequential read")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(bench, peer_first, "HYLE FileHandle 4MiB", [&] { model.hyle_read(); },
           "direct OS file 4MiB", [&] { model.raw_read(); });
}
