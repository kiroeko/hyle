#if !defined(__linux__)
#  error "the partial-release replay has a Linux/WSL contract"
#endif

#include <nanobench.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hyle/mem/mapping.h"
#include "hyle/platform/virtual_memory.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kRangeCount = 64;
constexpr std::size_t kStridePages = 16;
constexpr std::size_t kReleasedPages = 8;
constexpr std::size_t kLeadingPages = 4;
constexpr std::size_t kFilePages = kRangeCount * kStridePages;

[[noreturn]] void fail(const char* operation) noexcept {
  std::fprintf(stderr, "hyle_mem_vm_model_loader_bench: %s failed (%d)\n",
               operation, errno);
  std::abort();
}

bool environment_present(const char* name) {
  return std::getenv(name) != nullptr;
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

class LlamaFragmentMapping {
 public:
  LlamaFragmentMapping(const char* path, std::size_t expected_size,
                       std::span<const hyle::mem::MappingRange> ranges)
      : size_(expected_size) {
    const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) fail("open(peer)");
    struct ::stat info {};
    if (::fstat(descriptor, &info) != 0 || info.st_size < 0 ||
        static_cast<std::uint64_t>(info.st_size) != expected_size) {
      ::close(descriptor);
      fail("fstat(peer)");
    }
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, descriptor, 0);
    const int map_error = errno;
    if (::close(descriptor) != 0) fail("close(peer)");
    if (data_ == MAP_FAILED) {
      errno = map_error;
      fail("mmap(peer)");
    }
    fragments_.emplace_back(0, size_);
    for (const auto range : ranges) {
      const int advice = ::posix_madvise(
          static_cast<std::byte*>(data_) + range.offset, range.size,
          POSIX_MADV_RANDOM);
      if (advice != 0) {
        errno = advice;
        fail("posix_madvise(peer)");
      }
    }
  }

  ~LlamaFragmentMapping() {
    for (const auto& fragment : fragments_) {
      if (::munmap(static_cast<std::byte*>(data_) + fragment.first,
                   fragment.second - fragment.first) != 0) {
        fail("munmap remaining(peer)");
      }
    }
  }

  LlamaFragmentMapping(const LlamaFragmentMapping&) = delete;
  LlamaFragmentMapping& operator=(const LlamaFragmentMapping&) = delete;

  void unmap_fragment(std::size_t first, std::size_t last) {
    const std::size_t page = hyle::platform::page_size();
    first = (first + page - 1) / page * page;
    last = last / page * page;
    if (last <= first) return;
    if (::munmap(static_cast<std::byte*>(data_) + first, last - first) != 0) {
      fail("munmap fragment(peer)");
    }

    std::vector<std::pair<std::size_t, std::size_t>> remaining;
    for (const auto& fragment : fragments_) {
      if (fragment.first < first && fragment.second > last) {
        remaining.emplace_back(fragment.first, first);
        remaining.emplace_back(last, fragment.second);
      } else if (fragment.first < first && fragment.second > first) {
        remaining.emplace_back(fragment.first, first);
      } else if (fragment.first < last && fragment.second > last) {
        remaining.emplace_back(last, fragment.second);
      } else if (!(fragment.first >= first && fragment.second <= last)) {
        remaining.push_back(fragment);
      }
    }
    fragments_ = std::move(remaining);
    released_ += last - first;
  }

  std::size_t released() const noexcept { return released_; }
  std::size_t fragment_count() const noexcept { return fragments_.size(); }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t released_ = 0;
  std::vector<std::pair<std::size_t, std::size_t>> fragments_;
};

class ModelFile {
 public:
  ModelFile() {
    const std::size_t page = hyle::platform::page_size();
    file_size_ = kFilePages * page;
    release_size_ = kReleasedPages * page;
    path_ = (std::filesystem::temp_directory_path() /
             ("hyle_mem_vm_model_loader_" +
              std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
              ".bin"))
                .string();
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output) std::abort();
    output.seekp(static_cast<std::streamoff>(file_size_ - 1));
    output.put('\0');
    output.close();
    if (!output) std::abort();

    for (std::size_t index = 0; index < ranges_.size(); ++index) {
      ranges_[index] = hyle::mem::MappingRange{
          (index * kStridePages + kLeadingPages) * page, release_size_};
    }
  }

  ~ModelFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void hyle_replay() const noexcept {
    auto mapping = hyle::mem::FileMapping::open(path_.c_str());
    if (!mapping || mapping->size() != file_size_) std::abort();
    for (const auto range : ranges_) {
      if (!mapping->advise(hyle::mem::MappingAdvice::kRandom, range.offset,
                           range.size)
               .ok()) {
        std::abort();
      }
    }
    auto released = mapping->release_ranges(ranges_);
    if (!released || *released != ranges_.size() * release_size_ ||
        mapping->mapped_size() != file_size_ - *released ||
        mapping->fragment_count() != ranges_.size() + 1) {
      std::abort();
    }
    if (!mapping->reset().ok()) std::abort();
    nb::doNotOptimizeAway(*released);
  }

  void llama_replay() const {
    LlamaFragmentMapping mapping(path_.c_str(), file_size_, ranges_);
    for (const auto range : ranges_) {
      mapping.unmap_fragment(range.offset, range.offset + range.size);
    }
    if (mapping.released() != ranges_.size() * release_size_ ||
        mapping.fragment_count() != ranges_.size() + 1) {
      std::abort();
    }
    nb::doNotOptimizeAway(mapping.released());
  }

 private:
  std::string path_;
  std::size_t file_size_ = 0;
  std::size_t release_size_ = 0;
  std::array<hyle::mem::MappingRange, kRangeCount> ranges_{};
};

}  // namespace

int main() {
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_MEM_VM_MODEL_LOADER_RUN execution_order=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               affinity.topology, affinity.cpu, affinity.ok,
               hyle::bench::perf::run_nonce());

  ModelFile model;
  model.hyle_replay();
  model.llama_replay();

  nb::Bench bench;
  bench.title("model loader lazy release aggregate N=64")
      .relative(true)
      .batch(kRangeCount)
      .minEpochIterations(512)
      .minEpochTime(std::chrono::milliseconds(50));
  run_pair(bench, peer_first, "HYLE FileMapping release_ranges N=64",
           [&] { model.hyle_replay(); },
           "llama fragment vector unmap N=64",
           [&] { model.llama_replay(); });
}
