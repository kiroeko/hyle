// SwissTable vs absl::flat_hash_map -- the SOTA parity check (not just "beat the
// known-slow std::unordered_map"). abseil is the design HYLE's SwissTable is
// modeled on, so this is the honest "did we actually match the reference?" bench.
// Built only when -DHYLE_BENCH_ABSL=ON locates the cloned abseil. Whatever the
// numbers say, they print -- losing to absl is a real signal, not hidden.

#include <nanobench.h>

#include <cstdint>
#include <random>
#include <vector>

#include "absl/container/flat_hash_map.h"
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

int bench_swiss_vs_absl() {
  const std::vector<std::uint64_t> keys = gen(kN, 42);
  const std::vector<std::uint64_t> misses = gen(kN, 1337);

  SwissTable<std::uint64_t, std::uint64_t> hyle(system_allocator());
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
      SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
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
      SwissTable<std::uint64_t, std::uint64_t> t(system_allocator());
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

int main() { return bench_swiss_vs_absl(); }
