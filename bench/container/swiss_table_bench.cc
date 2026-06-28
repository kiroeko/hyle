// SwissTable vs std::unordered_map (roadmap P3 gate: "hashmap 对拍 std + 基准").
// std::unordered_map is the baseline (relative=1.0); SwissTable's relative number
// is the speedup. We measure the four hot ops over 100k uint64 keys. Honest:
// whatever the numbers say, they print -- no cherry-picking.

#include <nanobench.h>

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"

using hyle::container::SwissTable;
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

int bench_swiss_table() {
  const std::vector<std::uint64_t> keys = gen(kN, 42);
  const std::vector<std::uint64_t> misses = gen(kN, 1337);  // disjoint set

  // Pre-filled maps for the read benchmarks.
  SwissTable<std::uint64_t, std::uint64_t> hyle(system_allocator());
  std::unordered_map<std::uint64_t, std::uint64_t> stdm;
  for (int i = 0; i < kN; ++i) {
    hyle.insert(keys[i], static_cast<std::uint64_t>(i));
    stdm[keys[i]] = static_cast<std::uint64_t>(i);
  }

  // ---- find (hit) ----
  {
    ankerl::nanobench::Bench b;
    b.title("find-hit, 100k uint64 keys").relative(true).batch(kN);
    b.run("std::unordered_map", [&] {
      std::uint64_t sum = 0;
      for (auto k : keys) {
        auto it = stdm.find(k);
        if (it != stdm.end()) sum += it->second;
      }
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
    b.run("SwissTable", [&] {
      std::uint64_t sum = 0;
      for (auto k : keys) {
        if (auto* v = hyle.find(k)) sum += *v;
      }
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
  }

  // ---- find (miss) ----
  {
    ankerl::nanobench::Bench b;
    b.title("find-miss, 100k absent keys").relative(true).batch(kN);
    b.run("std::unordered_map", [&] {
      std::size_t hits = 0;
      for (auto k : misses) hits += (stdm.find(k) != stdm.end());
      ankerl::nanobench::doNotOptimizeAway(hits);
    });
    b.run("SwissTable", [&] {
      std::size_t hits = 0;
      for (auto k : misses) hits += (hyle.find(k) != nullptr);
      ankerl::nanobench::doNotOptimizeAway(hits);
    });
  }

  // ---- insert (fresh map per epoch; both pay allocation) ----
  {
    ankerl::nanobench::Bench b;
    b.title("insert, 100k uint64 keys (fresh)").relative(true).batch(kN);
    b.run("std::unordered_map", [&] {
      std::unordered_map<std::uint64_t, std::uint64_t> m;
      for (int i = 0; i < kN; ++i) m[keys[i]] = static_cast<std::uint64_t>(i);
      ankerl::nanobench::doNotOptimizeAway(m.size());
    });
    b.run("SwissTable", [&] {
      SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
      for (int i = 0; i < kN; ++i)
        t.insert(keys[i], static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(t.size());
    });
  }

  // ---- erase (build then erase all; both pay build) ----
  {
    ankerl::nanobench::Bench b;
    b.title("erase, 100k keys (build+erase)").relative(true).batch(kN);
    b.run("std::unordered_map", [&] {
      std::unordered_map<std::uint64_t, std::uint64_t> m;
      for (int i = 0; i < kN; ++i) m[keys[i]] = static_cast<std::uint64_t>(i);
      for (auto k : keys) m.erase(k);
      ankerl::nanobench::doNotOptimizeAway(m.size());
    });
    b.run("SwissTable", [&] {
      SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
      for (int i = 0; i < kN; ++i)
        t.insert(keys[i], static_cast<std::uint64_t>(i));
      for (auto k : keys) t.erase(k);
      ankerl::nanobench::doNotOptimizeAway(t.size());
    });
  }

  return 0;
}
