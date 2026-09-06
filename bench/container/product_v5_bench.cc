#include <nanobench.h>
#include <simdutf.h>

#include <absl/container/flat_hash_map.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../common/perf_process.h"
#include "hyle/container/hash.h"
#include "hyle/container/lru_cache.h"
#include "hyle/container/string_algo.h"
#include "hyle/container/string_view.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/std_allocator.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kBatch = 4096;
constexpr std::size_t kLruCapacity = 4096;
constexpr std::size_t kLruHotEntries = 1024;
constexpr std::size_t kMetadataRecords = 256;
constexpr std::size_t kMetadataBytes = 128;
constexpr std::size_t kUtfMeasurementUnitBytes = 8;
constexpr std::size_t kUtfMeasurementUnits =
    kMetadataRecords * kMetadataBytes / kUtfMeasurementUnitBytes;
static_assert(kMetadataRecords * kMetadataBytes % kUtfMeasurementUnitBytes ==
              0);

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

nb::Bench make_bench(const char* title, std::size_t batch) {
  nb::Bench bench;
  bench.title(title)
      .relative(true)
      .batch(batch)
      .minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

struct StableHash {
  std::uint64_t operator()(std::uint64_t value) const noexcept {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }
};

class PeerLru {
  struct Entry {
    std::uint64_t key;
    std::uint64_t value;
  };

  using ListAllocator = hyle::mem::SystemStdAllocator<Entry>;
  using List = std::list<Entry, ListAllocator>;
  using IndexValue = std::pair<const std::uint64_t, typename List::iterator>;
  using IndexAllocator = hyle::mem::SystemStdAllocator<IndexValue>;
  using Index = absl::flat_hash_map<std::uint64_t, typename List::iterator,
                                    StableHash, std::equal_to<>,
                                    IndexAllocator>;

 public:
  explicit PeerLru(std::size_t capacity)
      : entries_(ListAllocator{}),
        index_(0, StableHash{}, std::equal_to<>{}, IndexAllocator{}),
        capacity_(capacity) {
    index_.reserve(capacity);
  }

  std::uint64_t* get(std::uint64_t key) noexcept {
    const auto found = index_.find(key);
    if (found == index_.end()) return nullptr;
    entries_.splice(entries_.begin(), entries_, found->second);
    return &found->second->value;
  }

  template <class OnEvict>
  std::uint64_t* insert_or_assign(std::uint64_t key, std::uint64_t value,
                                  OnEvict&& on_evict) {
    const auto found = index_.find(key);
    if (found != index_.end()) {
      found->second->value = value;
      entries_.splice(entries_.begin(), entries_, found->second);
      return &found->second->value;
    }

    entries_.push_front({key, value});
    index_.emplace(key, entries_.begin());
    if (entries_.size() > capacity_) {
      const Entry& victim = entries_.back();
      std::invoke(on_evict, victim.key, victim.value);
      index_.erase(victim.key);
      entries_.pop_back();
    }
    return &entries_.front().value;
  }

  std::size_t size() const noexcept { return entries_.size(); }

 private:
  List entries_;
  Index index_;
  std::size_t capacity_;
};

using CandidateLru =
    hyle::container::LruCache<std::uint64_t, std::uint64_t, StableHash>;

class LruReplay {
 public:
  explicit LruReplay(bool peer_allocation_first) {
    if (peer_allocation_first) {
      peer_ = std::make_unique<PeerLru>(kLruCapacity);
      candidate_ = std::make_unique<CandidateLru>(
          hyle::container::LruCacheLimits{.max_entries = kLruCapacity});
    } else {
      candidate_ = std::make_unique<CandidateLru>(
          hyle::container::LruCacheLimits{.max_entries = kLruCapacity});
      peer_ = std::make_unique<PeerLru>(kLruCapacity);
    }
    for (std::uint64_t key = 0; key < kLruCapacity; ++key) {
      const std::uint64_t value = key * 17 + 3;
      if (!candidate_->try_insert_or_assign(key, value).stored())
        std::abort();
      (void)peer_->insert_or_assign(key, value,
                                    [](std::uint64_t, std::uint64_t) {});
    }
    // Establish the stable hot set before the first charged miss. Without
    // this setup-only pass, the numeric keys that will become hot can still
    // be the oldest initial entries and be evicted before their first access.
    for (std::uint64_t key = 0; key < kLruHotEntries; ++key) {
      if (candidate_->get(key) == nullptr || peer_->get(key) == nullptr)
        std::abort();
    }
  }

  HYLE_NOINLINE std::uint64_t candidate() {
    std::uint64_t checksum = 0;
    for (std::size_t operation = 0; operation < kBatch; ++operation) {
      if ((operation & 15U) != 15U) {
        const std::uint64_t key =
            (operation * 131) & (kLruHotEntries - 1);
        std::uint64_t* value = candidate_->get(key);
        if (value == nullptr) {
          std::fprintf(stderr, "candidate lru miss operation=%zu key=%llu next=%llu\n",
                       operation, static_cast<unsigned long long>(key),
                       static_cast<unsigned long long>(candidate_next_));
          std::abort();
        }
        checksum += *value;
      } else {
        const std::uint64_t key = candidate_next_++;
        const std::uint64_t value = key * 17 + 3;
        auto result = candidate_->try_insert_or_assign(
            key, value, 1,
            [&](std::uint64_t&& evicted_key, std::uint64_t&& evicted_value,
                std::size_t weight) noexcept {
              checksum += evicted_key ^ evicted_value ^ weight;
            });
        if (!result.stored() || result.evicted != 1) {
          std::fprintf(stderr,
                       "candidate lru insert operation=%zu key=%llu status=%u evicted=%zu\n",
                       operation, static_cast<unsigned long long>(key),
                       static_cast<unsigned>(result.status), result.evicted);
          std::abort();
        }
        checksum += *result.value;
      }
    }
    return checksum;
  }

  HYLE_NOINLINE std::uint64_t peer() {
    std::uint64_t checksum = 0;
    for (std::size_t operation = 0; operation < kBatch; ++operation) {
      if ((operation & 15U) != 15U) {
        const std::uint64_t key =
            (operation * 131) & (kLruHotEntries - 1);
        std::uint64_t* value = peer_->get(key);
        if (value == nullptr) std::abort();
        checksum += *value;
      } else {
        const std::uint64_t key = peer_next_++;
        const std::uint64_t value = key * 17 + 3;
        std::size_t evicted = 0;
        std::uint64_t* stored = peer_->insert_or_assign(
            key, value,
            [&](std::uint64_t evicted_key, std::uint64_t evicted_value) {
              checksum += evicted_key ^ evicted_value ^ 1U;
              ++evicted;
            });
        if (evicted != 1) std::abort();
        checksum += *stored;
      }
    }
    return checksum;
  }

  std::size_t candidate_size() const noexcept { return candidate_->size(); }
  std::size_t peer_size() const noexcept { return peer_->size(); }

 private:
  std::unique_ptr<CandidateLru> candidate_;
  std::unique_ptr<PeerLru> peer_;
  std::uint64_t candidate_next_ = kLruCapacity;
  std::uint64_t peer_next_ = kLruCapacity;
};

struct MetadataRecord {
  std::string input;
  std::vector<char16_t> output;
};

class UtfReplay {
 public:
  UtfReplay() {
    for (std::size_t record = 0; record < records_.size(); ++record) {
      std::string& input = records_[record].input;
      while (input.size() < kMetadataBytes) {
        if ((input.size() + record) % 29 == 0 &&
            input.size() + 6 <= kMetadataBytes) {
          input.append("\xE6\xA8\xA1\xE5\x9E\x8B");
        } else {
          input.push_back(static_cast<char>('a' + record % 26));
        }
      }
      const std::size_t required =
          simdutf::utf16_length_from_utf8(input.data(), input.size());
      records_[record].output.resize(required);
    }
  }

  HYLE_NOINLINE std::uint64_t candidate() noexcept {
    std::uint64_t checksum = 0;
    for (MetadataRecord& record : records_) {
      const hyle::container::UtfResult result = hyle::container::utf8_to_utf16(
          hyle::container::StringView{record.input}, record.output);
      if (!result || result.output_count != record.output.size()) std::abort();
      checksum += result.output_count + record.output.back();
    }
    return checksum;
  }

  HYLE_NOINLINE std::uint64_t peer() noexcept {
    std::uint64_t checksum = 0;
    for (MetadataRecord& record : records_) {
      const simdutf::result validation = simdutf::validate_utf8_with_errors(
          record.input.data(), record.input.size());
      if (validation.error != simdutf::SUCCESS) std::abort();
      const std::size_t required = simdutf::utf16_length_from_utf8(
          record.input.data(), record.input.size());
      if (required > record.output.size()) std::abort();
      const std::size_t produced = simdutf::convert_valid_utf8_to_utf16(
          record.input.data(), record.input.size(), record.output.data());
      if (produced != required) std::abort();
      checksum += produced + record.output.back();
    }
    return checksum;
  }

  std::size_t input_bytes() const noexcept {
    return kMetadataRecords * kMetadataBytes;
  }

 private:
  std::array<MetadataRecord, kMetadataRecords> records_{};
};

int validate_contract() {
  LruReplay lru(false);
  const std::uint64_t candidate_lru = lru.candidate();
  const std::uint64_t peer_lru = lru.peer();
  UtfReplay candidate_utf;
  UtfReplay peer_utf;
  const std::uint64_t candidate_text = candidate_utf.candidate();
  const std::uint64_t peer_text = peer_utf.peer();
  const bool passed = candidate_lru == peer_lru &&
                      candidate_text == peer_text &&
                      lru.candidate_size() == kLruCapacity &&
                      lru.peer_size() == kLruCapacity;
  std::printf(
      "# HYLE_CONTAINER_V5_CONTRACT schema=1 lru=%s text=%s "
      "lru_capacity=%zu lru_operations=%zu text_records=%zu "
      "text_input_bytes=%zu text_units=%zu text_unit_bytes=%zu verdict=%s\n",
      candidate_lru == peer_lru ? "PASS" : "FAIL",
      candidate_text == peer_text ? "PASS" : "FAIL", kLruCapacity, kBatch,
      kMetadataRecords, candidate_utf.input_bytes(), kUtfMeasurementUnits,
      kUtfMeasurementUnitBytes, passed ? "PASS" : "FAIL");
  return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract")
    return validate_contract();

  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_CONTAINER_V5_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s batch=%zu lru_capacity=%zu lru_hit_ratio=0.9375 "
      "text_records=%zu text_input_bytes=%zu text_units=%zu "
      "text_unit_bytes=%zu run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu, affinity.ok, kBatch,
      kLruCapacity, kMetadataRecords, kMetadataRecords * kMetadataBytes,
      kUtfMeasurementUnits, kUtfMeasurementUnitBytes,
      hyle::bench::perf::run_nonce());

  LruReplay lru(peer_allocation_first);
  nb::Bench lru_bench = make_bench(
      "container model-cache LRU mixed access N=4096", kBatch);
  run_pair(
      lru_bench, peer_first, "HYLE LruCache mixed access N=4096",
      [&] { nb::doNotOptimizeAway(lru.candidate()); },
      "Abseil+list LRU mixed access N=4096",
      [&] { nb::doNotOptimizeAway(lru.peer()); });

  std::unique_ptr<UtfReplay> candidate_utf;
  std::unique_ptr<UtfReplay> peer_utf;
  if (peer_allocation_first) {
    peer_utf = std::make_unique<UtfReplay>();
    candidate_utf = std::make_unique<UtfReplay>();
  } else {
    candidate_utf = std::make_unique<UtfReplay>();
    peer_utf = std::make_unique<UtfReplay>();
  }
  nb::Bench utf_bench = make_bench(
      "text model-metadata UTF-8 to UTF-16 8-byte units N=4096",
      kUtfMeasurementUnits);
  run_pair(
      utf_bench, peer_first, "HYLE UTF-8 to UTF-16 units N=4096",
      [&] { nb::doNotOptimizeAway(candidate_utf->candidate()); },
      "simdutf exact-contract UTF-8 to UTF-16 units N=4096",
      [&] { nb::doNotOptimizeAway(peer_utf->peer()); });
  return 0;
}
