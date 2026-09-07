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
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
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

bool environment_present(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

nb::Bench make_bench(const char *title, std::size_t batch) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(batch).minEpochTime(
      std::chrono::milliseconds(8));
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

template <class T, class RawAlloc> class ReplayStdAllocator {
  template <class, class> friend class ReplayStdAllocator;

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <class U> struct rebind {
    using other = ReplayStdAllocator<U, RawAlloc>;
  };

  ReplayStdAllocator()
    requires std::is_default_constructible_v<RawAlloc>
  = default;
  explicit ReplayStdAllocator(RawAlloc alloc) noexcept
      : alloc_(std::move(alloc)) {}

  template <class U>
  ReplayStdAllocator(const ReplayStdAllocator<U, RawAlloc> &other) noexcept
      : alloc_(other.raw_allocator()) {}

  [[nodiscard]] T *allocate(std::size_t count) {
    if (count > max_size())
      throw std::bad_array_new_length{};
    if (count == 0)
      return nullptr;
    void *storage = alloc_.allocate(count * sizeof(T), alignof(T));
    if (storage == nullptr)
      throw std::bad_alloc{};
    return static_cast<T *>(storage);
  }

  void deallocate(T *pointer, std::size_t count) noexcept {
    alloc_.deallocate(pointer, count * sizeof(T), alignof(T));
  }

  [[nodiscard]] static constexpr std::size_t max_size() noexcept {
    return (std::numeric_limits<std::size_t>::max)() / sizeof(T);
  }

  [[nodiscard]] const RawAlloc &raw_allocator() const noexcept {
    return alloc_;
  }

private:
  HYLE_NO_UNIQUE_ADDRESS RawAlloc alloc_{};
};

template <class T, class U, class RawAlloc>
bool operator==(const ReplayStdAllocator<T, RawAlloc> &left,
                const ReplayStdAllocator<U, RawAlloc> &right) noexcept {
  if constexpr (requires { left.raw_allocator() == right.raw_allocator(); }) {
    return left.raw_allocator() == right.raw_allocator();
  } else {
    return true;
  }
}

struct FailureBudget {
  std::size_t allocations_before_failure = 0;
};

struct FailingRawAllocator {
  FailureBudget *budget = nullptr;

  void *allocate(std::size_t size, std::size_t alignment) noexcept {
    if (budget == nullptr || budget->allocations_before_failure == 0)
      return nullptr;
    --budget->allocations_before_failure;
    return hyle::mem::SystemAllocator::allocate(size, alignment);
  }

  void deallocate(void *pointer, std::size_t size,
                  std::size_t alignment) noexcept {
    hyle::mem::SystemAllocator::deallocate(pointer, size, alignment);
  }

  friend bool operator==(FailingRawAllocator left,
                         FailingRawAllocator right) noexcept {
    return left.budget == right.budget;
  }
};

enum class PeerFailurePoint : std::uint8_t {
  kNone,
  kBeforeNode,
  kAfterNode,
};

template <class RawAlloc = hyle::mem::SystemAllocator,
          PeerFailurePoint FailurePoint = PeerFailurePoint::kNone>
class PeerLru {
  struct Entry {
    std::uint64_t key;
    std::uint64_t value;
    std::size_t weight;
  };

  using ListAllocator = ReplayStdAllocator<Entry, RawAlloc>;
  using List = std::list<Entry, ListAllocator>;
  using IndexValue = std::pair<const std::uint64_t, typename List::iterator>;
  using IndexAllocator = ReplayStdAllocator<IndexValue, RawAlloc>;
  using Index =
      absl::flat_hash_map<std::uint64_t, typename List::iterator, StableHash,
                          std::equal_to<>, IndexAllocator>;

public:
  explicit PeerLru(hyle::container::LruCacheLimits limits,
                   RawAlloc alloc = RawAlloc{})
      : entries_(ListAllocator{alloc}),
        index_(0, StableHash{}, std::equal_to<>{}, IndexAllocator{alloc}),
        limits_(limits) {}

  std::uint64_t *get(std::uint64_t key) noexcept {
    const auto found = index_.find(key);
    if (found == index_.end())
      return nullptr;
    entries_.splice(entries_.begin(), entries_, found->second);
    return &found->second->value;
  }

  template <class OnEvict>
    requires std::is_nothrow_invocable_v<OnEvict &, std::uint64_t,
                                         std::uint64_t, std::size_t>
  hyle::container::LruInsertResult<std::uint64_t>
  try_insert_or_assign(std::uint64_t key, std::uint64_t value,
                       std::size_t weight,
                       OnEvict &&on_evict) {
    using hyle::container::LruInsertStatus;
    if (limits_.max_entries == 0 || weight > limits_.max_weight)
      return {nullptr, LruInsertStatus::kRejected, 0};

    const auto found = index_.find(key);
    if (found != index_.end()) {
      const std::size_t weight_without_existing =
          total_weight_ - found->second->weight;
      if (weight > (std::numeric_limits<std::size_t>::max)() -
                       weight_without_existing)
        return {nullptr, LruInsertStatus::kRejected, 0};
      found->second->value = value;
      found->second->weight = weight;
      total_weight_ = weight_without_existing + weight;
      entries_.splice(entries_.begin(), entries_, found->second);
      const std::size_t evicted = trim(std::forward<OnEvict>(on_evict));
      return {&found->second->value, LruInsertStatus::kAssigned, evicted};
    }

    if (weight > (std::numeric_limits<std::size_t>::max)() - total_weight_)
      return {nullptr, LruInsertStatus::kRejected, 0};

    if constexpr (FailurePoint == PeerFailurePoint::kBeforeNode)
      return {nullptr, LruInsertStatus::kOutOfMemory, 0};

    try {
      entries_.push_front({key, value, weight});
    } catch (const std::bad_alloc &) {
      return {nullptr, LruInsertStatus::kOutOfMemory, 0};
    }
    if constexpr (FailurePoint == PeerFailurePoint::kAfterNode) {
      entries_.pop_front();
      return {nullptr, LruInsertStatus::kOutOfMemory, 0};
    }
    try {
      const auto [unused, inserted] = index_.emplace(key, entries_.begin());
      (void)unused;
      if (!inserted)
        std::abort();
    } catch (const std::bad_alloc &) {
      entries_.pop_front();
      return {nullptr, LruInsertStatus::kOutOfMemory, 0};
    }

    total_weight_ += weight;
    const std::size_t evicted = trim(std::forward<OnEvict>(on_evict));
    return {&entries_.front().value,
            LruInsertStatus::kInserted, evicted};
  }

  std::size_t size() const noexcept { return entries_.size(); }
  std::size_t total_weight() const noexcept { return total_weight_; }

private:
  template <class OnEvict> std::size_t trim(OnEvict &&on_evict) noexcept {
    std::size_t evicted = 0;
    while (entries_.size() > limits_.max_entries ||
           total_weight_ > limits_.max_weight) {
      const Entry &victim = entries_.back();
      const std::uint64_t key = victim.key;
      const std::uint64_t value = victim.value;
      const std::size_t weight = victim.weight;
      std::invoke(on_evict, key, value, weight);
      index_.erase(key);
      entries_.pop_back();
      total_weight_ -= weight;
      ++evicted;
    }
    return evicted;
  }

  List entries_;
  Index index_;
  hyle::container::LruCacheLimits limits_;
  std::size_t total_weight_ = 0;
};

template <class Alloc = hyle::mem::SystemAllocator>
using CandidateLruFor =
    hyle::container::LruCache<std::uint64_t, std::uint64_t, StableHash,
                              std::equal_to<>, Alloc>;
using CandidateLru = CandidateLruFor<>;

class LruReplay {
public:
  explicit LruReplay(bool peer_allocation_first) {
    constexpr hyle::container::LruCacheLimits limits{
        .max_entries = kLruCapacity, .max_weight = kLruCapacity};
    if (peer_allocation_first) {
      peer_ = std::make_unique<PeerLru<>>(limits);
      candidate_ = std::make_unique<CandidateLru>(limits);
    } else {
      candidate_ = std::make_unique<CandidateLru>(limits);
      peer_ = std::make_unique<PeerLru<>>(limits);
    }
    for (std::uint64_t key = 0; key < kLruCapacity; ++key) {
      const std::uint64_t value = key * 17 + 3;
      if (!candidate_->try_insert_or_assign(key, value).stored())
        std::abort();
      const auto result = peer_->try_insert_or_assign(
          key, value, 1,
          [](std::uint64_t, std::uint64_t, std::size_t) noexcept {});
      if (!result.stored())
        std::abort();
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
        const std::uint64_t key = (operation * 131) & (kLruHotEntries - 1);
        std::uint64_t *value = candidate_->get(key);
        if (value == nullptr) {
          std::fprintf(stderr,
                       "candidate lru miss operation=%zu key=%llu next=%llu\n",
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
            [&](std::uint64_t &&evicted_key, std::uint64_t &&evicted_value,
                std::size_t weight) noexcept {
              checksum += evicted_key ^ evicted_value ^ weight;
            });
        if (!result.stored() || result.evicted != 1) {
          std::fprintf(stderr,
                       "candidate lru insert operation=%zu key=%llu status=%u "
                       "evicted=%zu\n",
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
        const std::uint64_t key = (operation * 131) & (kLruHotEntries - 1);
        std::uint64_t *value = peer_->get(key);
        if (value == nullptr)
          std::abort();
        checksum += *value;
      } else {
        const std::uint64_t key = peer_next_++;
        const std::uint64_t value = key * 17 + 3;
        std::size_t evicted = 0;
        auto result = peer_->try_insert_or_assign(
            key, value, 1,
            [&](std::uint64_t evicted_key,
                std::uint64_t evicted_value,
                std::size_t weight) noexcept {
              checksum += evicted_key ^ evicted_value ^ weight;
              ++evicted;
            });
        if (!result.stored() || result.evicted != 1 || evicted != 1)
          std::abort();
        checksum += *result.value;
      }
    }
    return checksum;
  }

  std::size_t candidate_size() const noexcept { return candidate_->size(); }
  std::size_t peer_size() const noexcept { return peer_->size(); }
  std::size_t candidate_weight() const noexcept {
    return candidate_->total_weight();
  }
  std::size_t peer_weight() const noexcept { return peer_->total_weight(); }

private:
  std::unique_ptr<CandidateLru> candidate_;
  std::unique_ptr<PeerLru<>> peer_;
  std::uint64_t candidate_next_ = kLruCapacity;
  std::uint64_t peer_next_ = kLruCapacity;
};

bool validate_lru_failure_contract() {
  using Candidate = CandidateLruFor<FailingRawAllocator>;
  const hyle::container::LruCacheLimits limits{.max_entries = 4};

  FailureBudget candidate_node_failure{0};
  Candidate candidate_node(limits, FailingRawAllocator{&candidate_node_failure},
                           StableHash{});
  const auto candidate_node_result = candidate_node.try_insert_or_assign(1, 11);
  PeerLru<hyle::mem::SystemAllocator, PeerFailurePoint::kBeforeNode> peer_node(
      limits);
  const auto peer_node_result = peer_node.try_insert_or_assign(
      1, 11, 1,
      [](std::uint64_t, std::uint64_t, std::size_t) noexcept {});

  FailureBudget candidate_index_failure{1};
  Candidate candidate_index(
      limits, FailingRawAllocator{&candidate_index_failure}, StableHash{});
  const auto candidate_index_result =
      candidate_index.try_insert_or_assign(2, 22);

  PeerLru<hyle::mem::SystemAllocator, PeerFailurePoint::kAfterNode> peer_index(
      limits);
  const auto peer_index_result = peer_index.try_insert_or_assign(
      2, 22, 1,
      [](std::uint64_t, std::uint64_t, std::size_t) noexcept {});

  const auto out_of_memory = [](const auto &result) {
    return result.status == hyle::container::LruInsertStatus::kOutOfMemory &&
           result.value == nullptr && result.evicted == 0;
  };
  return out_of_memory(candidate_node_result) &&
         out_of_memory(peer_node_result) &&
         out_of_memory(candidate_index_result) &&
         out_of_memory(peer_index_result) && candidate_node.empty() &&
         peer_node.size() == 0 && candidate_index.empty() &&
         peer_index.size() == 0;
}

bool validate_lru_weight_contract() {
  using hyle::container::LruCacheLimits;
  using hyle::container::LruInsertStatus;
  constexpr LruCacheLimits limits{.max_entries = 4, .max_weight = 5};
  CandidateLru candidate(limits);
  PeerLru<> peer(limits);

  std::uint64_t candidate_evictions = 0;
  std::uint64_t peer_evictions = 0;
  auto candidate_evict = [&](std::uint64_t &&key, std::uint64_t &&value,
                             std::size_t weight) noexcept {
    candidate_evictions += key * 101 + value * 7 + weight;
  };
  auto peer_evict = [&](std::uint64_t key, std::uint64_t value,
                        std::size_t weight) noexcept {
    peer_evictions += key * 101 + value * 7 + weight;
  };

  const auto candidate_first =
      candidate.try_insert_or_assign(1, 11, 2, candidate_evict);
  const auto peer_first = peer.try_insert_or_assign(1, 11, 2, peer_evict);
  const auto candidate_second =
      candidate.try_insert_or_assign(2, 22, 2, candidate_evict);
  const auto peer_second = peer.try_insert_or_assign(2, 22, 2, peer_evict);
  const auto candidate_weight_evict =
      candidate.try_insert_or_assign(3, 33, 2, candidate_evict);
  const auto peer_weight_evict =
      peer.try_insert_or_assign(3, 33, 2, peer_evict);
  const auto candidate_assign =
      candidate.try_insert_or_assign(2, 222, 4, candidate_evict);
  const auto peer_assign =
      peer.try_insert_or_assign(2, 222, 4, peer_evict);
  const auto candidate_rejected =
      candidate.try_insert_or_assign(4, 44, 6, candidate_evict);
  const auto peer_rejected =
      peer.try_insert_or_assign(4, 44, 6, peer_evict);

  const auto same_result = [](const auto &left, const auto &right) {
    return left.status == right.status && left.evicted == right.evicted &&
           left.stored() == right.stored();
  };
  const std::uint64_t *candidate_value = candidate.get(2);
  const std::uint64_t *peer_value = peer.get(2);
  return same_result(candidate_first, peer_first) &&
         same_result(candidate_second, peer_second) &&
         same_result(candidate_weight_evict, peer_weight_evict) &&
         candidate_weight_evict.status == LruInsertStatus::kInserted &&
         candidate_weight_evict.evicted == 1 &&
         same_result(candidate_assign, peer_assign) &&
         candidate_assign.status == LruInsertStatus::kAssigned &&
         candidate_assign.evicted == 1 &&
         same_result(candidate_rejected, peer_rejected) &&
         candidate_rejected.status == LruInsertStatus::kRejected &&
         candidate.size() == 1 && peer.size() == 1 &&
         candidate.total_weight() == 4 && peer.total_weight() == 4 &&
         candidate_evictions == peer_evictions && candidate_value != nullptr &&
         peer_value != nullptr && *candidate_value == 222 && *peer_value == 222 &&
         !candidate.contains(1) && !candidate.contains(3) &&
         peer.get(1) == nullptr && peer.get(3) == nullptr;
}

hyle::container::UtfStatus peer_utf_status(simdutf::error_code error) noexcept {
  using hyle::container::UtfStatus;
  switch (error) {
  case simdutf::SUCCESS:
    return UtfStatus::kSuccess;
  case simdutf::HEADER_BITS:
    return UtfStatus::kHeaderBits;
  case simdutf::TOO_SHORT:
    return UtfStatus::kTooShort;
  case simdutf::TOO_LONG:
    return UtfStatus::kTooLong;
  case simdutf::OVERLONG:
    return UtfStatus::kOverlong;
  case simdutf::TOO_LARGE:
    return UtfStatus::kTooLarge;
  case simdutf::SURROGATE:
    return UtfStatus::kSurrogate;
  case simdutf::OUTPUT_BUFFER_TOO_SMALL:
    return UtfStatus::kOutputTooSmall;
  default:
    return UtfStatus::kOther;
  }
}

hyle::container::UtfResult
peer_utf8_to_utf16(hyle::container::StringView input,
                   std::span<char16_t> output) noexcept {
  using hyle::container::UtfResult;
  using hyle::container::UtfStatus;
  if (input.data() == nullptr) {
    return input.empty() ? UtfResult{}
                         : UtfResult{UtfStatus::kInvalidArgument, 0, 0};
  }
  if (output.data() == nullptr && !output.empty())
    return {UtfStatus::kInvalidArgument, 0, 0};

  const simdutf::implementation *provider =
      simdutf::get_active_implementation();
  const simdutf::result validation =
      provider->validate_utf8_with_errors(input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return {peer_utf_status(validation.error), validation.count, 0};

  const std::size_t required =
      provider->utf16_length_from_utf8(input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
#if SIMDUTF_IS_BIG_ENDIAN
    produced = provider->convert_valid_utf8_to_utf16be(
        input.data(), input.size(), output.data());
#else
    produced = provider->convert_valid_utf8_to_utf16le(
        input.data(), input.size(), output.data());
#endif
  }
  return {UtfStatus::kSuccess, input.size(), produced};
}

bool same_utf_result(hyle::container::UtfResult left,
                     hyle::container::UtfResult right) noexcept {
  return left.status == right.status &&
         left.input_offset == right.input_offset &&
         left.output_count == right.output_count;
}

struct MetadataRecord {
  std::string input;
  std::vector<char16_t> output;
};

class UtfReplay {
public:
  UtfReplay() {
    for (std::size_t record = 0; record < records_.size(); ++record) {
      std::string &input = records_[record].input;
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
    for (MetadataRecord &record : records_) {
      const hyle::container::UtfResult result = hyle::container::utf8_to_utf16(
          hyle::container::StringView{record.input}, record.output);
      if (!result || result.output_count != record.output.size())
        std::abort();
      checksum += result.output_count + record.output.back();
    }
    return checksum;
  }

  HYLE_NOINLINE std::uint64_t peer() noexcept {
    std::uint64_t checksum = 0;
    for (MetadataRecord &record : records_) {
      const hyle::container::UtfResult result = peer_utf8_to_utf16(
          hyle::container::StringView{record.input}, record.output);
      if (!result || result.output_count != record.output.size())
        std::abort();
      checksum += result.output_count + record.output.back();
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
  const bool lru_failure = validate_lru_failure_contract();
  const bool lru_weight = validate_lru_weight_contract();
  LruReplay lru(false);
  const std::uint64_t candidate_lru = lru.candidate();
  const std::uint64_t peer_lru = lru.peer();
  UtfReplay candidate_utf;
  UtfReplay peer_utf;
  const std::uint64_t candidate_unicode = candidate_utf.candidate();
  const std::uint64_t peer_unicode = peer_utf.peer();

  const std::string malformed{"\xF0\x28\x8C\x28", 4};
  std::array<char16_t, 8> malformed_candidate_output{};
  std::array<char16_t, 8> malformed_peer_output{};
  const auto candidate_malformed = hyle::container::utf8_to_utf16(
      hyle::container::StringView{malformed}, malformed_candidate_output);
  const auto peer_malformed = peer_utf8_to_utf16(
      hyle::container::StringView{malformed}, malformed_peer_output);
  const bool malformed_equal =
      !candidate_malformed && !peer_malformed &&
      candidate_malformed.status !=
          hyle::container::UtfStatus::kOutputTooSmall &&
      same_utf_result(candidate_malformed, peer_malformed) &&
      malformed_candidate_output == malformed_peer_output;

  const std::string valid_multibyte{"\xE6\xA8\xA1", 3};
  std::array<char16_t, 1> candidate_small_output{u'X'};
  std::array<char16_t, 1> peer_small_output{u'X'};
  const auto candidate_small = hyle::container::utf8_to_utf16(
      hyle::container::StringView{valid_multibyte},
      std::span<char16_t>{candidate_small_output}.first(0));
  const auto peer_small =
      peer_utf8_to_utf16(hyle::container::StringView{valid_multibyte},
                         std::span<char16_t>{peer_small_output}.first(0));
  const bool small_equal =
      candidate_small.status == hyle::container::UtfStatus::kOutputTooSmall &&
      peer_small.status == hyle::container::UtfStatus::kOutputTooSmall &&
      candidate_small.input_offset == valid_multibyte.size() &&
      candidate_small.output_count == 1 &&
      same_utf_result(candidate_small, peer_small) &&
      candidate_small_output == peer_small_output;

  const bool passed =
      candidate_lru == peer_lru && candidate_unicode == peer_unicode &&
      lru_failure && lru_weight && malformed_equal && small_equal &&
      lru.candidate_size() == kLruCapacity && lru.peer_size() == kLruCapacity &&
      lru.candidate_weight() == kLruCapacity &&
      lru.peer_weight() == kLruCapacity;
  std::printf("# HYLE_CONTAINER_V6_CONTRACT schema=3 lru=%s lru_oom=%s "
              "lru_weight=%s "
              "container_text=%s unicode_malformed=%s unicode_output_small=%s "
              "lru_capacity=%zu lru_operations=%zu unicode_records=%zu "
              "unicode_input_bytes=%zu unicode_units=%zu "
              "unicode_unit_bytes=%zu verdict=%s\n",
              candidate_lru == peer_lru ? "PASS" : "FAIL",
              lru_failure ? "PASS" : "FAIL",
              lru_weight ? "PASS" : "FAIL",
              candidate_unicode == peer_unicode ? "PASS" : "FAIL",
              malformed_equal ? "PASS" : "FAIL", small_equal ? "PASS" : "FAIL",
              kLruCapacity, kBatch, kMetadataRecords,
              candidate_utf.input_bytes(), kUtfMeasurementUnits,
              kUtfMeasurementUnitBytes, passed ? "PASS" : "FAIL");
  return passed ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract")
    return validate_contract();

  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(stderr,
               "# HYLE_CONTAINER_V6_RUN execution_order=%s allocation_order=%s "
               "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
               "affinity_ok=%s batch=%zu lru_capacity=%zu lru_hit_ratio=0.9375 "
               "unicode_records=%zu unicode_input_bytes=%zu unicode_units=%zu "
               "unicode_unit_bytes=%zu run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               peer_allocation_first ? "peer-first" : "candidate-first",
               qos.name, qos.ok, affinity.topology, affinity.cpu, affinity.ok,
               kBatch, kLruCapacity, kMetadataRecords,
               kMetadataRecords * kMetadataBytes, kUtfMeasurementUnits,
               kUtfMeasurementUnitBytes, hyle::bench::perf::run_nonce());

  LruReplay lru(peer_allocation_first);
  nb::Bench lru_bench =
      make_bench("container model-cache LRU mixed access N=4096", kBatch);
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
      "container_text model-metadata UTF-8 to UTF-16 8-byte units N=4096",
      kUtfMeasurementUnits);
  run_pair(
      utf_bench, peer_first, "HYLE UTF-8 to UTF-16 units N=4096",
      [&] { nb::doNotOptimizeAway(candidate_utf->candidate()); },
      "simdutf UtfResult-adapted UTF-8 to UTF-16 units N=4096",
      [&] { nb::doNotOptimizeAway(peer_utf->peer()); });
  return 0;
}
