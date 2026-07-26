#include <nanobench.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "hyle/container/btree.h"

namespace {

namespace nb = ankerl::nanobench;
using HyleMap = hyle::container::BTreeMap<std::uint64_t, std::uint64_t>;
using AbslMap = absl::btree_map<std::uint64_t, std::uint64_t>;
using HyleSet = hyle::container::BTreeSet<std::uint64_t>;
using AbslSet = absl::btree_set<std::uint64_t>;

constexpr std::size_t kEntries = 1u << 16;
constexpr std::size_t kQueries = 1u << 15;
constexpr std::size_t kRangeWidth = 16;

std::vector<std::uint64_t> shuffled_keys(std::size_t count,
                                         std::uint64_t seed) {
  std::vector<std::uint64_t> keys(count);
  for (std::size_t index = 0; index < count; ++index)
    keys[index] = static_cast<std::uint64_t>(index);
  std::mt19937_64 generator(seed);
  std::shuffle(keys.begin(), keys.end(), generator);
  return keys;
}

template <class Map> Map filled(const std::vector<std::uint64_t> &keys) {
  Map map;
  for (std::uint64_t key : keys)
    map.insert_or_assign(key, key * 3 + 1);
  return map;
}

template <class Set> Set filled_set(const std::vector<std::uint64_t> &keys) {
  Set set;
  for (std::uint64_t key : keys)
    set.insert(key);
  return set;
}

void run_find(const std::vector<std::uint64_t> &queries, const AbslMap &map) {
  std::uint64_t sum = 0;
  for (std::uint64_t key : queries)
    sum += map.find(key)->second;
  nb::doNotOptimizeAway(sum);
}

void run_find(const std::vector<std::uint64_t> &queries, const HyleMap &map) {
  std::uint64_t sum = 0;
  for (std::uint64_t key : queries)
    sum += map.find(key)->value();
  nb::doNotOptimizeAway(sum);
}

template <class Map>
void run_find_miss_above_max(const std::vector<std::uint64_t> &queries,
                             const Map &map) {
  std::size_t misses = 0;
  for (std::uint64_t key : queries)
    misses += map.find(key + kEntries) == map.end();
  nb::doNotOptimizeAway(misses);
}

template <class Map>
void run_find_miss_between_keys(const std::vector<std::uint64_t> &queries,
                                const Map &map) {
  std::size_t misses = 0;
  for (std::uint64_t key : queries)
    misses += map.find(key * 2 + 1) == map.end();
  nb::doNotOptimizeAway(misses);
}

template <class Set>
void run_set_contains(const std::vector<std::uint64_t> &queries,
                      const Set &set) {
  std::size_t present = 0;
  for (std::uint64_t key : queries)
    present += set.contains(key);
  nb::doNotOptimizeAway(present);
}

void run_ranges(const std::vector<std::uint64_t> &queries, const AbslMap &map) {
  std::uint64_t sum = 0;
  for (std::uint64_t key : queries) {
    auto iterator = map.lower_bound(key);
    for (std::size_t offset = 0; offset < kRangeWidth && iterator != map.end();
         ++offset, ++iterator)
      sum += iterator->second;
  }
  nb::doNotOptimizeAway(sum);
}

void run_ranges(const std::vector<std::uint64_t> &queries, const HyleMap &map) {
  std::uint64_t sum = 0;
  for (std::uint64_t key : queries) {
    auto iterator = map.lower_bound(key);
    for (std::size_t offset = 0; offset < kRangeWidth && iterator != map.end();
         ++offset, ++iterator)
      sum += iterator->value();
  }
  nb::doNotOptimizeAway(sum);
}

} // namespace

int main() {
  const auto keys = shuffled_keys(kEntries, 0xB7EEu);
  const auto query_source = shuffled_keys(kEntries, 0xA11CEu);
  const std::vector<std::uint64_t> queries(query_source.begin(),
                                           query_source.begin() + kQueries);
  const HyleMap hyle = filled<HyleMap>(keys);
  const AbslMap absl = filled<AbslMap>(keys);
  const HyleSet hyle_set = filled_set<HyleSet>(keys);
  const AbslSet absl_set = filled_set<AbslSet>(keys);
  auto sparse_keys = keys;
  for (std::uint64_t &key : sparse_keys)
    key *= 2;
  const HyleMap hyle_sparse = filled<HyleMap>(sparse_keys);
  const AbslMap absl_sparse = filled<AbslMap>(sparse_keys);

  {
    nb::Bench bench;
    bench.title("ordered index find-hit, 64Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run("absl::btree_map", [&] { run_find(queries, absl); });
    bench.run("HYLE BTreeMap", [&] { run_find(queries, hyle); });
  }

  {
    nb::Bench bench;
    bench.title("ordered index above-maximum find-miss, 64Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run("absl::btree_map",
              [&] { run_find_miss_above_max(queries, absl); });
    bench.run("HYLE BTreeMap", [&] { run_find_miss_above_max(queries, hyle); });
  }

  {
    nb::Bench bench;
    bench.title("ordered index in-domain find-miss, 64Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run("absl::btree_map",
              [&] { run_find_miss_between_keys(queries, absl_sparse); });
    bench.run("HYLE BTreeMap",
              [&] { run_find_miss_between_keys(queries, hyle_sparse); });
  }

  {
    nb::Bench bench;
    bench.title("ordered set membership, 64Ki entries")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(15);
    bench.run("absl::btree_set", [&] { run_set_contains(queries, absl_set); });
    bench.run("HYLE BTreeSet", [&] { run_set_contains(queries, hyle_set); });
  }

  {
    nb::Bench bench;
    bench.title("range index lower_bound + scan 16")
        .relative(true)
        .batch(kQueries)
        .minEpochIterations(10);
    bench.run("absl::btree_map", [&] { run_ranges(queries, absl); });
    bench.run("HYLE BTreeMap", [&] { run_ranges(queries, hyle); });
  }

  {
    nb::Bench bench;
    bench.title("ordered index deterministic full iteration")
        .relative(true)
        .batch(kEntries)
        .minEpochIterations(15);
    bench.run("absl::btree_map", [&] {
      std::uint64_t sum = 0;
      for (const auto &entry : absl)
        sum += entry.second;
      nb::doNotOptimizeAway(sum);
    });
    bench.run("HYLE BTreeMap", [&] {
      std::uint64_t sum = 0;
      for (const auto &entry : hyle)
        sum += entry.value();
      nb::doNotOptimizeAway(sum);
    });
  }

  {
    nb::Bench bench;
    bench.title("ordered index build from unordered events")
        .relative(true)
        .batch(kEntries)
        .minEpochIterations(5);
    bench.run("absl::btree_map", [&] {
      auto map = filled<AbslMap>(keys);
      nb::doNotOptimizeAway(map.size());
    });
    bench.run("HYLE BTreeMap", [&] {
      auto map = filled<HyleMap>(keys);
      nb::doNotOptimizeAway(map.size());
    });
  }

  {
    nb::Bench bench;
    bench.title("ordered index copy existing index")
        .relative(true)
        .batch(kEntries)
        .minEpochIterations(5);
    bench.run("absl::btree_map", [&] {
      const AbslMap copy = absl;
      nb::doNotOptimizeAway(copy.size());
    });
    bench.run("HYLE BTreeMap", [&] {
      const HyleMap copy = hyle;
      nb::doNotOptimizeAway(copy.size());
    });
  }

  HyleMap hyle_mutable = hyle;
  AbslMap absl_mutable = absl;
  bool hyle_high_half = false;
  bool absl_high_half = false;
  constexpr std::size_t kChurn = kEntries / 4;
  {
    nb::Bench bench;
    bench.title("mutable ordered index erase + insert churn")
        .relative(true)
        .batch(kChurn)
        .minEpochIterations(5);
    bench.run("absl::btree_map", [&] {
      const std::uint64_t remove_base = absl_high_half ? kEntries : 0;
      const std::uint64_t insert_base = absl_high_half ? 0 : kEntries;
      for (std::size_t index = 0; index < kChurn; ++index) {
        absl_mutable.erase(remove_base + index);
        absl_mutable.insert_or_assign(insert_base + index, insert_base + index);
      }
      absl_high_half = !absl_high_half;
      nb::doNotOptimizeAway(absl_mutable.size());
    });
    bench.run("HYLE BTreeMap", [&] {
      const std::uint64_t remove_base = hyle_high_half ? kEntries : 0;
      const std::uint64_t insert_base = hyle_high_half ? 0 : kEntries;
      for (std::size_t index = 0; index < kChurn; ++index) {
        hyle_mutable.erase(remove_base + index);
        hyle_mutable.insert_or_assign(insert_base + index, insert_base + index);
      }
      hyle_high_half = !hyle_high_half;
      nb::doNotOptimizeAway(hyle_mutable.size());
    });
  }

  HyleMap hyle_expiry = hyle;
  AbslMap absl_expiry = absl;
  bool hyle_expiry_high_half = false;
  bool absl_expiry_high_half = false;
  {
    nb::Bench bench;
    bench.title("time index expire range + refill")
        .relative(true)
        .batch(kChurn)
        .minEpochIterations(5);
    bench.run("absl::btree_map", [&] {
      const std::uint64_t remove_base = absl_expiry_high_half ? kEntries : 0;
      const std::uint64_t insert_base = absl_expiry_high_half ? 0 : kEntries;
      absl_expiry.erase(absl_expiry.lower_bound(remove_base),
                        absl_expiry.lower_bound(remove_base + kChurn));
      for (std::size_t index = 0; index < kChurn; ++index)
        absl_expiry.insert_or_assign(insert_base + index, insert_base + index);
      absl_expiry_high_half = !absl_expiry_high_half;
      nb::doNotOptimizeAway(absl_expiry.size());
    });
    bench.run("HYLE BTreeMap", [&] {
      const std::uint64_t remove_base = hyle_expiry_high_half ? kEntries : 0;
      const std::uint64_t insert_base = hyle_expiry_high_half ? 0 : kEntries;
      hyle_expiry.erase(hyle_expiry.lower_bound(remove_base),
                        hyle_expiry.lower_bound(remove_base + kChurn));
      for (std::size_t index = 0; index < kChurn; ++index)
        hyle_expiry.insert_or_assign(insert_base + index, insert_base + index);
      hyle_expiry_high_half = !hyle_expiry_high_half;
      nb::doNotOptimizeAway(hyle_expiry.size());
    });
  }
  return 0;
}
