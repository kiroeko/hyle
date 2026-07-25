#define _CRT_SECURE_NO_WARNINGS 1

// FlatHashMap/FlatHashSet vs their Abseil peers -- the SOTA parity check (not
// just "beat the known-slow std::unordered_* family"). Abseil is the design
// HYLE's Swiss core is modeled on, so this is an honest reference comparison.
// Built only when -DHYLE_BENCH_ABSL=ON locates the cloned abseil. Whatever the
// numbers say, they print -- losing to absl is a real signal, not hidden.

#include <nanobench.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"

using hyle::container::SwissTable;
using hyle::container::FlatHashSet;
using hyle::mem::system_allocator;

namespace {
constexpr int kN = 100000;

std::vector<std::uint64_t> gen(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::uint64_t> v;
  v.reserve(n);
  for (int i = 0; i < n; ++i) v.push_back(rng());
  return v;
}
}  // namespace

int bench_swiss_vs_absl() {
  const std::vector<std::uint64_t> keys = gen(kN, 42);
  const std::vector<std::uint64_t> misses = gen(kN, 1337);

  SwissTable<std::uint64_t, std::uint64_t> hyle;
  absl::flat_hash_map<std::uint64_t, std::uint64_t> abslm;
  for (int i = 0; i < kN; ++i) {
    hyle.insert(keys[i], static_cast<std::uint64_t>(i));
    abslm[keys[i]] = static_cast<std::uint64_t>(i);
  }

  {
    ankerl::nanobench::Bench b;
    b.title("find-hit vs absl, 100k").relative(true).batch(kN).minEpochIterations(20);
    b.run("absl::flat_hash_map", [&] {
      std::uint64_t sum = 0;
      for (auto k : keys) {
        auto it = abslm.find(k);
        if (it != abslm.end()) sum += it->second;
      }
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
    b.run("SwissTable", [&] {
      std::uint64_t sum = 0;
      for (auto k : keys)
        if (auto* v = hyle.find(k)) sum += *v;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("find-miss vs absl, 100k").relative(true).batch(kN).minEpochIterations(20);
    b.run("absl::flat_hash_map", [&] {
      std::size_t h = 0;
      for (auto k : misses) h += (abslm.find(k) != abslm.end());
      ankerl::nanobench::doNotOptimizeAway(h);
    });
    b.run("SwissTable", [&] {
      std::size_t h = 0;
      for (auto k : misses) h += (hyle.find(k) != nullptr);
      ankerl::nanobench::doNotOptimizeAway(h);
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("insert vs absl, 100k (fresh)").relative(true).batch(kN).minEpochIterations(5);
    b.run("absl::flat_hash_map", [&] {
      absl::flat_hash_map<std::uint64_t, std::uint64_t> m;
      for (int i = 0; i < kN; ++i) m[keys[i]] = static_cast<std::uint64_t>(i);
      ankerl::nanobench::doNotOptimizeAway(m.size());
    });
    b.run("SwissTable", [&] {
      SwissTable<std::uint64_t, std::uint64_t> t;
      for (int i = 0; i < kN; ++i) t.insert(keys[i], static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(t.size());
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("erase vs absl, 100k (build+erase)").relative(true).batch(kN).minEpochIterations(5);
    b.run("absl::flat_hash_map", [&] {
      absl::flat_hash_map<std::uint64_t, std::uint64_t> m;
      for (int i = 0; i < kN; ++i) m[keys[i]] = static_cast<std::uint64_t>(i);
      for (auto k : keys) m.erase(k);
      ankerl::nanobench::doNotOptimizeAway(m.size());
    });
    b.run("SwissTable", [&] {
      SwissTable<std::uint64_t, std::uint64_t> t;
      for (int i = 0; i < kN; ++i) t.insert(keys[i], static_cast<std::uint64_t>(i));
      for (auto k : keys) t.erase(k);
      ankerl::nanobench::doNotOptimizeAway(t.size());
    });
  }
  // ---- iterate (visit every live entry; both pre-filled at top) ----
  // Fills the ledger gap "absl 未测此项": SwissTable.for_each vs absl range-for.
  // Both are flat (ctrl-byte scan + contiguous slots) so this is a same-design
  // check, not a walk-vs-pointer-chase blowout like vs std::unordered_map.
  {
    ankerl::nanobench::Bench b;
    b.title("iterate vs absl, 100k (visit all)").relative(true).batch(kN).minEpochIterations(20);
    b.run("absl::flat_hash_map", [&] {
      std::uint64_t sum = 0;
      for (const auto& kv : abslm) sum += kv.second;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
    b.run("SwissTable", [&] {
      std::uint64_t sum = 0;
      hyle.for_each([&](std::uint64_t, std::uint64_t& v) { sum += v; });
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
  }
  return 0;
}

int bench_set_vs_absl() {
  const std::vector<std::uint64_t> keys = gen(kN, 42);
  const std::vector<std::uint64_t> misses = gen(kN, 1337);
  FlatHashSet<std::uint64_t> hyle;
  absl::flat_hash_set<std::uint64_t> absl_set;
  for (std::uint64_t key : keys) {
    hyle.insert(key);
    absl_set.insert(key);
  }

  {
    ankerl::nanobench::Bench b;
    b.title("set find-hit vs absl, 100k")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    b.run("absl::flat_hash_set", [&] {
      std::size_t found = 0;
      for (std::uint64_t key : keys) found += absl_set.contains(key);
      ankerl::nanobench::doNotOptimizeAway(found);
    });
    b.run("HYLE FlatHashSet", [&] {
      std::size_t found = 0;
      for (std::uint64_t key : keys) found += hyle.contains(key);
      ankerl::nanobench::doNotOptimizeAway(found);
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("set find-miss vs absl, 100k")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    b.run("absl::flat_hash_set", [&] {
      std::size_t found = 0;
      for (std::uint64_t key : misses) found += absl_set.contains(key);
      ankerl::nanobench::doNotOptimizeAway(found);
    });
    b.run("HYLE FlatHashSet", [&] {
      std::size_t found = 0;
      for (std::uint64_t key : misses) found += hyle.contains(key);
      ankerl::nanobench::doNotOptimizeAway(found);
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("set insert vs absl, 100k (fresh)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(5);
    b.run("absl::flat_hash_set", [&] {
      absl::flat_hash_set<std::uint64_t> set;
      for (std::uint64_t key : keys) set.insert(key);
      ankerl::nanobench::doNotOptimizeAway(set.size());
    });
    b.run("HYLE FlatHashSet", [&] {
      FlatHashSet<std::uint64_t> set;
      for (std::uint64_t key : keys) set.insert(key);
      ankerl::nanobench::doNotOptimizeAway(set.size());
    });
  }
  {
    ankerl::nanobench::Bench b;
    b.title("set iterate vs absl, 100k")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    b.run("absl::flat_hash_set", [&] {
      std::uint64_t sum = 0;
      for (std::uint64_t key : absl_set) sum += key;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
    b.run("HYLE FlatHashSet", [&] {
      std::uint64_t sum = 0;
      for (std::uint64_t key : hyle) sum += key;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
  }
  return 0;
}

int main() {
  const char* filter = std::getenv("HYLE_ABSL_BENCH_FILTER");
  if (filter != nullptr && std::string_view(filter) == "map")
    return bench_swiss_vs_absl();
  if (filter != nullptr && std::string_view(filter) == "set")
    return bench_set_vs_absl();
  const int map_result = bench_swiss_vs_absl();
  return map_result == 0 ? bench_set_vs_absl() : map_result;
}
