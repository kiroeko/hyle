#include <nanobench.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#if __has_include(<flat_map>) && __has_include(<flat_set>)
#include <flat_map>
#include <flat_set>
#endif

#include "hyle/container/flat_sorted.h"

namespace {

namespace nb = ankerl::nanobench;
using HyleMap = hyle::container::FlatMap<std::uint64_t, std::uint64_t>;
using HyleSet = hyle::container::FlatSet<std::uint64_t>;

constexpr std::size_t kEntries = 1u << 15;
constexpr std::size_t kQueries = 1u << 14;
constexpr std::size_t kMutations = 64;

std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted_values() {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> values;
  values.reserve(kEntries);
  for (std::uint64_t key = 0; key < kEntries; ++key)
    values.emplace_back(key * 2, key * 3 + 1);
  return values;
}

std::vector<std::uint64_t> shuffled_queries(std::uint64_t seed) {
  std::vector<std::uint64_t> queries(kEntries);
  for (std::uint64_t key = 0; key < kEntries; ++key)
    queries[key] = key;
  std::mt19937_64 generator(seed);
  std::shuffle(queries.begin(), queries.end(), generator);
  queries.resize(kQueries);
  return queries;
}

// Portable same-layout floor used where the standard library has not shipped
// C++23 flat_map yet. It intentionally has no custom metadata or policy knob.
class VectorFlatMapPeer {
public:
  explicit VectorFlatMapPeer(
      const std::vector<std::pair<std::uint64_t, std::uint64_t>> &values)
      : values_(values) {}

  auto begin() const noexcept { return values_.begin(); }
  auto end() const noexcept { return values_.end(); }
  std::size_t size() const noexcept { return values_.size(); }

  auto lower_bound(std::uint64_t key) const noexcept {
    return std::lower_bound(values_.begin(), values_.end(), key,
                            [](const auto &entry, std::uint64_t value) {
                              return entry.first < value;
                            });
  }
  auto find(std::uint64_t key) const noexcept {
    const auto position = lower_bound(key);
    return position != values_.end() && position->first == key
               ? position
               : values_.end();
  }
  void insert_or_assign(std::uint64_t key, std::uint64_t value) {
    auto position = std::lower_bound(
        values_.begin(), values_.end(), key,
        [](const auto &entry, std::uint64_t input) {
          return entry.first < input;
        });
    if (position != values_.end() && position->first == key) {
      position->second = value;
      return;
    }
    values_.insert(position, {key, value});
  }
  bool erase(std::uint64_t key) {
    auto position = std::lower_bound(
        values_.begin(), values_.end(), key,
        [](const auto &entry, std::uint64_t input) {
          return entry.first < input;
        });
    if (position == values_.end() || position->first != key)
      return false;
    values_.erase(position);
    return true;
  }

private:
  std::vector<std::pair<std::uint64_t, std::uint64_t>> values_;
};

#if defined(__cpp_lib_flat_map) && __cpp_lib_flat_map >= 202207L
using FlatMapPeer = std::flat_map<std::uint64_t, std::uint64_t>;
using FlatSetPeer = std::flat_set<std::uint64_t>;
constexpr const char *kMapPeerName = "std::flat_map";
constexpr const char *kSetPeerName = "std::flat_set";

FlatMapPeer make_map_peer(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &values) {
  return FlatMapPeer(std::sorted_unique, values.begin(), values.end());
}
FlatSetPeer make_set_peer(const std::vector<std::uint64_t> &values) {
  return FlatSetPeer(std::sorted_unique, values.begin(), values.end());
}
#else
using FlatMapPeer = VectorFlatMapPeer;
using FlatSetPeer = std::vector<std::uint64_t>;
constexpr const char *kMapPeerName = "sorted std::vector peer";
constexpr const char *kSetPeerName = "sorted std::vector floor";

FlatMapPeer make_map_peer(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &values) {
  return FlatMapPeer(values);
}
FlatSetPeer make_set_peer(const std::vector<std::uint64_t> &values) {
  return values;
}
#endif

void find_hits(const HyleMap &map,
               const std::vector<std::uint64_t> &queries) {
  std::uint64_t sum = 0;
  for (const std::uint64_t query : queries)
    sum += map.find(query * 2)->value();
  nb::doNotOptimizeAway(sum);
}

template <class Map>
void find_peer_hits(const Map &map,
                    const std::vector<std::uint64_t> &queries) {
  std::uint64_t sum = 0;
  for (const std::uint64_t query : queries)
    sum += map.find(query * 2)->second;
  nb::doNotOptimizeAway(sum);
}

template <class Map>
void find_misses(const Map &map,
                 const std::vector<std::uint64_t> &queries) {
  std::size_t misses = 0;
  for (const std::uint64_t query : queries)
    misses += map.find(query * 2 + 1) == map.end();
  nb::doNotOptimizeAway(misses);
}

} // namespace

int main() {
  const auto values = sorted_values();
  const auto queries = shuffled_queries(0xF1A7u);
  const HyleMap hyle(hyle::container::sorted_unique, values.begin(),
                     values.end());
  const FlatMapPeer peer = make_map_peer(values);

  {
    nb::Bench bench;
    bench.title("flat ordered lookup hit, 32Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run(kMapPeerName, [&] { find_peer_hits(peer, queries); });
    bench.run("HYLE FlatMap", [&] { find_hits(hyle, queries); });
  }

  {
    nb::Bench bench;
    bench.title("flat ordered in-domain lookup miss, 32Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run(kMapPeerName, [&] { find_misses(peer, queries); });
    bench.run("HYLE FlatMap", [&] { find_misses(hyle, queries); });
  }

  {
    nb::Bench bench;
    bench.title("flat ordered deterministic full iteration")
        .relative(true)
        .batch(kEntries)
        .minEpochIterations(15);
    bench.run(kMapPeerName, [&] {
      std::uint64_t sum = 0;
      for (const auto &entry : peer)
        sum += entry.second;
      nb::doNotOptimizeAway(sum);
    });
    bench.run("HYLE FlatMap", [&] {
      std::uint64_t sum = 0;
      for (const auto &entry : hyle)
        sum += entry.value();
      nb::doNotOptimizeAway(sum);
    });
  }

  {
    nb::Bench bench;
    bench.title("flat ordered linear sorted-unique bulk build")
        .relative(true)
        .batch(kEntries)
        .minEpochIterations(500);
    bench.run(kMapPeerName, [&] {
      const FlatMapPeer built = make_map_peer(values);
      nb::doNotOptimizeAway(built.size());
    });
    bench.run("HYLE FlatMap", [&] {
      const HyleMap built(hyle::container::sorted_unique, values.begin(),
                          values.end());
      nb::doNotOptimizeAway(built.size());
    });
  }

  {
    nb::Bench bench;
    bench.title("flat ordered low mutation: 64 erase + append")
        .relative(true)
        .batch(kMutations * 2)
        .minEpochIterations(100);
    bench.run(kMapPeerName, [&] {
      auto changed = peer;
      for (std::uint64_t key = 0; key < kMutations; ++key)
        changed.erase(key * 2);
      for (std::uint64_t key = 0; key < kMutations; ++key)
        changed.insert_or_assign((kEntries + key) * 2, key);
      nb::doNotOptimizeAway(changed.size());
    });
    bench.run("HYLE FlatMap", [&] {
      auto changed = hyle;
      for (std::uint64_t key = 0; key < kMutations; ++key)
        changed.erase(key * 2);
      for (std::uint64_t key = 0; key < kMutations; ++key)
        changed.insert_or_assign((kEntries + key) * 2, key);
      nb::doNotOptimizeAway(changed.size());
    });
  }

  // Set is measured separately so map's mutable mapped-value shape does not
  // hide a set-specific storage regression.
  std::vector<std::uint64_t> set_values;
  set_values.reserve(kEntries);
  for (std::uint64_t key = 0; key < kEntries; ++key)
    set_values.push_back(key * 2);
  const HyleSet hyle_set(hyle::container::sorted_unique, set_values.begin(),
                         set_values.end());
  const FlatSetPeer set_peer = make_set_peer(set_values);
  {
    nb::Bench bench;
    bench.title("flat ordered set membership, 32Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run(kSetPeerName, [&] {
      std::size_t found = 0;
      for (const std::uint64_t query : queries)
#if defined(__cpp_lib_flat_map) && __cpp_lib_flat_map >= 202207L
        found += set_peer.contains(query * 2);
#else
        found += std::binary_search(set_peer.begin(), set_peer.end(), query * 2);
#endif
      nb::doNotOptimizeAway(found);
    });
    bench.run("HYLE FlatSet", [&] {
      std::size_t found = 0;
      for (const std::uint64_t query : queries)
        found += hyle_set.contains(query * 2);
      nb::doNotOptimizeAway(found);
    });
  }
}
