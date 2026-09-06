#include <nanobench.h>

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

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "hyle/mem/mapping.h"

#include "../common/perf_process.h"

namespace {
namespace bip = boost::interprocess;
namespace nb = ankerl::nanobench;

constexpr std::size_t kBatch = 64;
constexpr std::size_t kPage = 4096;
constexpr std::size_t kPages = 4096;
constexpr std::size_t kFileBytes = kPage * kPages;

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

class ModelFile {
 public:
  ModelFile() {
    path_ = (std::filesystem::temp_directory_path() /
             ("hyle_mem_vm_consumer_" +
              std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
              ".bin"))
                .string();
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output) std::abort();
    std::vector<char> chunk(64u * 1024u);
    for (std::size_t index = 0; index < chunk.size(); ++index) {
      chunk[index] = static_cast<char>(index & 0x7f);
    }
    for (std::size_t written = 0; written < kFileBytes;
         written += chunk.size()) {
      output.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    }
    if (!output) std::abort();
    output.close();

    offsets_.reserve(kBatch);
    for (std::size_t index = 0; index < kBatch; ++index) {
      offsets_.push_back(((index * 4051u) & (kPages - 1u)) * kPage +
                         (index & 63u));
      expected_ += offsets_.back() & 0x7fu;
    }
  }

  ~ModelFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void hyle_replay() const noexcept {
    auto mapping = hyle::mem::FileMapping::open(path_.c_str());
    if (!mapping || mapping->size() != kFileBytes) std::abort();
    const auto bytes = mapping->bytes();
    if (!bytes) std::abort();
    std::uint64_t sum = 0;
    for (const std::size_t offset : offsets_) {
      sum += std::to_integer<unsigned char>((*bytes)[offset]);
    }
    if (sum != expected_) std::abort();
    nb::doNotOptimizeAway(sum);
  }

  void boost_replay() const {
    bip::file_mapping file(path_.c_str(), bip::read_only);
    bip::mapped_region mapping(file, bip::read_only);
    if (mapping.get_size() != kFileBytes) std::abort();
    const auto* bytes = static_cast<const unsigned char*>(mapping.get_address());
    std::uint64_t sum = 0;
    for (const std::size_t offset : offsets_) sum += bytes[offset];
    if (sum != expected_) std::abort();
    nb::doNotOptimizeAway(sum);
  }

 private:
  std::string path_;
  std::vector<std::size_t> offsets_;
  std::uint64_t expected_ = 0;
};

}  // namespace

int main() {
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_MEM_VM_MAPPING_RUN execution_order=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               affinity.topology, affinity.cpu, affinity.ok,
               hyle::bench::perf::run_nonce());

  // The only persistent allocation is the shared input file. Each measured
  // callback creates and destroys its own mapping owner, so there is no
  // candidate/peer allocation-order dimension to perturb honestly.
  ModelFile model;
  model.hyle_replay();
  model.boost_replay();
  nb::Bench bench;
  bench.title("model loader mapped probe aggregate N=64")
      .relative(true)
      .batch(kBatch)
      .minEpochIterations(128)
      .minEpochTime(std::chrono::milliseconds(20));
  run_pair(bench, peer_first, "HYLE FileMapping loader probe N=64",
           [&] { model.hyle_replay(); },
           "Boost mapped_region loader probe N=64",
           [&] { model.boost_replay(); });
}
