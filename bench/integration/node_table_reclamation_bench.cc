#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <oneapi/tbb/concurrent_hash_map.h>

#include "hyle/container/concurrent_node_hash_map.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kEntries = 4096;
constexpr std::size_t kLookups = 4096;

using HyleMap =
    hyle::container::ConcurrentNodeHashMap<std::uint64_t, std::uint64_t>;
using ReclaimingHyleMap =
    hyle::container::ReclaimingTableConcurrentNodeHashMap<std::uint64_t,
                                                          std::uint64_t>;

struct TbbHashCompare {
  static std::size_t hash(std::uint64_t value) noexcept {
    return static_cast<std::size_t>(
        hyle::container::Hash<std::uint64_t>{}(value));
  }
  static bool equal(std::uint64_t left, std::uint64_t right) noexcept {
    return left == right;
  }
};

using TbbMap = oneapi::tbb::concurrent_hash_map<std::uint64_t, std::uint64_t,
                                                TbbHashCompare>;

template <class Map> void fill_hyle(Map &map) {
  typename Map::Accessor accessor;
  for (std::size_t index = 0; index != kEntries; ++index) {
    map.emplace(accessor, static_cast<std::uint64_t>(index),
                static_cast<std::uint64_t>(index * 3U + 1U));
    accessor.release();
  }
}

void fill_tbb(TbbMap &map) {
  for (std::size_t index = 0; index != kEntries; ++index) {
    TbbMap::accessor accessor;
    map.insert(accessor, static_cast<std::uint64_t>(index));
    accessor->second = static_cast<std::uint64_t>(index * 3U + 1U);
  }
}

template <class Map> std::uint64_t read_hyle(Map &map) noexcept {
  std::uint64_t checksum = 0;
  typename Map::ConstAccessor accessor;
  for (std::size_t index = 0; index != kLookups; ++index) {
    const std::uint64_t key = static_cast<std::uint64_t>(index * 17U) &
                              static_cast<std::uint64_t>(kEntries - 1U);
    map.find(accessor, key);
    checksum += *accessor;
    accessor.release();
  }
  return checksum;
}

std::uint64_t read_tbb(TbbMap &map) noexcept {
  std::uint64_t checksum = 0;
  TbbMap::const_accessor accessor;
  for (std::size_t index = 0; index != kLookups; ++index) {
    const std::uint64_t key = static_cast<std::uint64_t>(index * 17U) &
                              static_cast<std::uint64_t>(kEntries - 1U);
    map.find(accessor, key);
    checksum += accessor->second;
    accessor.release();
  }
  return checksum;
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first, Candidate &&candidate,
              Peer &&peer) {
  if (peer_first) {
    bench.run("oneTBB concurrent_hash_map const_accessor", peer);
    bench.run("HYLE ConcurrentNodeHashMap retained table", candidate);
  } else {
    bench.run("HYLE ConcurrentNodeHashMap retained table", candidate);
    bench.run("oneTBB concurrent_hash_map const_accessor", peer);
  }
}

} // namespace

int main() {
  HyleMap candidate(kEntries * 2U);
  ReclaimingHyleMap reclaiming(kEntries * 2U);
  TbbMap peer;
  peer.rehash(kEntries * 2U);
  fill_hyle(candidate);
  fill_hyle(reclaiming);
  fill_tbb(peer);

  nb::doNotOptimizeAway(read_hyle(candidate));
  nb::doNotOptimizeAway(read_hyle(reclaiming));
  nb::doNotOptimizeAway(read_tbb(peer));

  const bool peer_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  const bool formal_only =
      std::getenv("HYLE_NODE_TABLE_BENCH_FORMAL_ONLY") != nullptr;
  const char *run_nonce = std::getenv("HYLE_PERF_RUN_NONCE");
  std::fprintf(
      stderr,
      "# HYLE_NODE_TABLE_RUN execution_order=%s formal_only=%d run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", formal_only ? 1 : 0,
      run_nonce != nullptr ? run_nonce : "");

  nb::Bench primary;
  primary.title("node registry protected hit aggregate N=4096")
      .relative(true)
      .batch(kLookups)
      .minEpochIterations(100)
      .minEpochTime(std::chrono::milliseconds(20));
  run_pair(
      primary, peer_first, [&] { nb::doNotOptimizeAway(read_hyle(candidate)); },
      [&] { nb::doNotOptimizeAway(read_tbb(peer)); });

  if (!formal_only) {
    nb::Bench holdout;
    holdout.title("node registry reclamation lifetime holdout N=4096")
        .relative(true)
        .batch(kLookups)
        .minEpochIterations(100)
        .minEpochTime(std::chrono::milliseconds(20));
    holdout.run("HYLE retained table",
                [&] { nb::doNotOptimizeAway(read_hyle(candidate)); });
    holdout.run("HYLE reclaiming table",
                [&] { nb::doNotOptimizeAway(read_hyle(reclaiming)); });
  }
}
