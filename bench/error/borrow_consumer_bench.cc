// Product evidence for a normal fallible registry borrow.
//
// The registry algorithm, backing lifetime, request distribution and
// observable work are shared by HYLE and Abseil.  Only the result carrier is
// selected by the trait: hyle::Result<T&> or absl::StatusOr<T&>.
#include <nanobench.h>

#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"

#include "../common/perf_process.h"

#if defined(HYLE_BENCH_ABSL)
#  include "absl/status/status.h"
#  include "absl/status/statusor.h"
#endif

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kRegistrySize = 256;
constexpr std::size_t kLookupsPerRequest = 64;
constexpr std::size_t kExpectedHits = 60;
constexpr std::size_t kExpectedMisses = 4;

struct RegistryEntry {
  std::uint32_t key = 0;
  std::uint64_t value = 0;
};

struct LookupSummary {
  std::uint64_t checksum = 0;
  std::uint32_t hits = 0;
  std::uint32_t misses = 0;
};

struct HyleBorrowCarrier {
  template <class T>
  using Result = hyle::Result<T&>;

  template <class T>
  static Result<T> success(T& value) noexcept {
    return value;
  }

  template <class T>
  static Result<T> not_found() noexcept {
    return hyle::make_unexpected(hyle::Status::NotFound());
  }

  template <class T>
  static bool has_value(const Result<T>& result) noexcept {
    return result.has_value();
  }

  template <class T>
  static T& value(Result<T>& result) noexcept {
    return *result;
  }

  template <class T>
  static bool is_not_found(const Result<T>& result) noexcept {
    return !result && result.error().IsNotFound();
  }
};

#if defined(HYLE_BENCH_ABSL)
struct PeerBorrowCarrier {
  template <class T>
  using Result = absl::StatusOr<T&>;

  template <class T>
  static Result<T> success(T& value) noexcept {
    return value;
  }

  template <class T>
  static Result<T> not_found() noexcept {
    return absl::Status(absl::StatusCode::kNotFound, "",
                        absl::SourceLocation());
  }

  template <class T>
  static bool has_value(const Result<T>& result) noexcept {
    return result.ok();
  }

  template <class T>
  static T& value(Result<T>& result) noexcept {
    return *result;
  }

  template <class T>
  static bool is_not_found(const Result<T>& result) noexcept {
    return !result.ok() &&
           result.status().code() == absl::StatusCode::kNotFound;
  }
};
#endif

template <class Carrier>
HYLE_NOINLINE typename Carrier::template Result<RegistryEntry> registry_lookup(
    RegistryEntry* entries, std::size_t size, std::uint32_t key) noexcept {
  if (key >= size || entries[key].key != key) {
    return Carrier::template not_found<RegistryEntry>();
  }
  return Carrier::success(entries[key]);
}

template <class Carrier>
HYLE_NOINLINE LookupSummary registry_request(
    RegistryEntry* entries, std::size_t size,
    const std::uint32_t* keys) noexcept {
  LookupSummary summary;
  for (std::size_t index = 0; index < kLookupsPerRequest; ++index) {
    auto result = registry_lookup<Carrier>(entries, size, keys[index]);
    if (Carrier::template has_value<RegistryEntry>(result)) {
      const RegistryEntry& entry =
          Carrier::template value<RegistryEntry>(result);
      summary.checksum += entry.value ^ entry.key;
      ++summary.hits;
    } else {
      ++summary.misses;
    }
  }
  return summary;
}

void initialize_registry(
    std::array<RegistryEntry, kRegistrySize>& entries,
    std::array<std::uint32_t, kLookupsPerRequest>& keys) noexcept {
  for (std::size_t index = 0; index < entries.size(); ++index) {
    entries[index] = RegistryEntry{
        static_cast<std::uint32_t>(index),
        static_cast<std::uint64_t>(index + 11) * 0x9e3779b185ebca87ULL};
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    // One ordinary miss in every 16 lookups.  All other keys are stable hits
    // spread across the fixed table; neither carrier gets a friendlier branch
    // sequence or backing layout.
    keys[index] = index % 16 == 15
                      ? static_cast<std::uint32_t>(kRegistrySize + index)
                      : static_cast<std::uint32_t>((index * 37 + 19) %
                                                   kRegistrySize);
  }
}

bool rival_first_requested() {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, "HYLE_BENCH_RIVAL_FIRST") == 0 &&
      value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
#endif
}

template <class CandidateOp, class PeerOp>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, CandidateOp&& candidate,
              std::string_view peer_name, PeerOp&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<PeerOp>(peer));
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
  } else {
    bench.run(candidate_name, std::forward<CandidateOp>(candidate));
    bench.run(peer_name, std::forward<PeerOp>(peer));
  }
}

template <class Function>
std::uintptr_t function_address(Function function) noexcept {
  return reinterpret_cast<std::uintptr_t>(function);
}

int validate_contract() {
  std::array<RegistryEntry, kRegistrySize> entries{};
  std::array<std::uint32_t, kLookupsPerRequest> keys{};
  initialize_registry(entries, keys);

  const LookupSummary hyle = registry_request<HyleBorrowCarrier>(
      entries.data(), entries.size(), keys.data());
  auto hyle_borrow = registry_lookup<HyleBorrowCarrier>(
      entries.data(), entries.size(), keys[0]);
  auto hyle_missing = registry_lookup<HyleBorrowCarrier>(
      entries.data(), entries.size(),
      static_cast<std::uint32_t>(entries.size()));
  const bool hyle_ok = hyle.hits == kExpectedHits &&
                       hyle.misses == kExpectedMisses &&
                       HyleBorrowCarrier::has_value<RegistryEntry>(
                           hyle_borrow) &&
                       std::addressof(*hyle_borrow) ==
                           std::addressof(entries[keys[0]]) &&
                       HyleBorrowCarrier::is_not_found<RegistryEntry>(
                           hyle_missing);

  LookupSummary peer{};
  bool peer_ok = false;
#if defined(HYLE_BENCH_ABSL)
  peer = registry_request<PeerBorrowCarrier>(
      entries.data(), entries.size(), keys.data());
  auto peer_borrow = registry_lookup<PeerBorrowCarrier>(
      entries.data(), entries.size(), keys[0]);
  auto peer_missing = registry_lookup<PeerBorrowCarrier>(
      entries.data(), entries.size(),
      static_cast<std::uint32_t>(entries.size()));
  peer_ok = peer.hits == kExpectedHits && peer.misses == kExpectedMisses &&
            PeerBorrowCarrier::has_value<RegistryEntry>(peer_borrow) &&
            std::addressof(*peer_borrow) ==
                std::addressof(entries[keys[0]]) &&
            PeerBorrowCarrier::is_not_found<RegistryEntry>(peer_missing) &&
            peer.checksum == hyle.checksum &&
            peer_missing.status().GetSourceLocations().empty();
#endif

  const bool passed = hyle_ok && peer_ok;
  std::printf(
      "# HYLE_ERROR_BORROW_CONSUMER_CONTRACT schema=1 table=%zu lookups=%zu "
      "hyle_hits=%" PRIu32 " peer_hits=%" PRIu32 " hyle_misses=%" PRIu32
      " peer_misses=%" PRIu32 " hyle_checksum=%" PRIu64
      " peer_checksum=%" PRIu64 " borrow_identity=%s source_location=%s "
      "verdict=%s\n",
      kRegistrySize, kLookupsPerRequest, hyle.hits, peer.hits, hyle.misses,
      peer.misses, hyle.checksum, peer.checksum, hyle_ok ? "PASS" : "FAIL",
      peer_ok ? "PASS" : "FAIL", passed ? "PASS" : "FAIL");
  std::printf(
      "# HYLE_ERROR_BORROW_CONSUMER_LAYOUT hyle_lookup=%" PRIxPTR
      " hyle_request=%" PRIxPTR
#if defined(HYLE_BENCH_ABSL)
      " peer_lookup=%" PRIxPTR " peer_request=%" PRIxPTR
#endif
      "\n",
      function_address(registry_lookup<HyleBorrowCarrier>),
      function_address(registry_request<HyleBorrowCarrier>)
#if defined(HYLE_BENCH_ABSL)
          ,
      function_address(registry_lookup<PeerBorrowCarrier>),
      function_address(registry_request<PeerBorrowCarrier>)
#endif
  );
  return passed ? 0 : 2;
}

void run_registry_benchmark(bool peer_first) {
  std::array<RegistryEntry, kRegistrySize> entries{};
  std::array<std::uint32_t, kLookupsPerRequest> keys{};
  initialize_registry(entries, keys);

  nb::Bench bench;
  bench.title("Registry fixed-table borrow lookup 60 hit 4 miss")
      .relative(true)
      .batch(kLookupsPerRequest)
      .minEpochIterations(10000);
#if defined(HYLE_BENCH_ABSL)
  run_pair(
      bench, peer_first, "HYLE Result<T&> registry borrow",
      [&] {
        const auto summary = registry_request<HyleBorrowCarrier>(
            entries.data(), entries.size(), keys.data());
        nb::doNotOptimizeAway(summary);
      },
      "absl::StatusOr<T&> registry borrow", [&] {
        const auto summary = registry_request<PeerBorrowCarrier>(
            entries.data(), entries.size(), keys.data());
        nb::doNotOptimizeAway(summary);
      });
#else
  bench.run("HYLE Result<T&> registry borrow", [&] {
    const auto summary = registry_request<HyleBorrowCarrier>(
        entries.data(), entries.size(), keys.data());
    nb::doNotOptimizeAway(summary);
  });
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract") {
    return validate_contract();
  }

  const auto qos = hyle::bench::perf::configure_thread_qos();
  const bool peer_first = rival_first_requested();
  std::fprintf(
      stderr,
      "# HYLE_ERROR_BORROW_CONSUMER_RUN benchmark=error-borrow-consumer "
      "execution_order=%s apple_qos=%s qos_ok=%s affinity=external "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
      hyle::bench::perf::run_nonce());
  run_registry_benchmark(peer_first);
  return 0;
}
