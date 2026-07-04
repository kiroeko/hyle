// container-06c fuzzy path-match bench. Two things to PROVE (not just measure):
//
//  1. TIME COMPLEXITY. find_path_matches over N candidates must be O(N) in the
//     candidate count and O(L) in candidate length -- the whole point of the greedy
//     two-pass scorer (vs fzf's O(pattern*len) DP). We run N = 1k/10k/100k with a
//     FIXED per-candidate length and report ns-per-candidate: a flat line across the
//     three N proves linear scaling (a rising line would betray a hidden super-linear
//     factor -- e.g. an accidental rescore). This is the "prove the Big-O, not a
//     single-point ratio" bar the perf discipline demands.
//
//  2. THE COST OF THE BONUSES. path_match_score does class-based boundary/camel/
//     consecutive bonuses + a backward tighten pass. We race a std hand-rolled
//     subsequence check (the ALGORITHM floor: no scoring, just the O(L) membership
//     scan every fuzzy matcher must at least do). The ratio is the price of fzf-grade
//     ranking on top of bare matching -- an honest number, not a "free" claim.
//
// Run under scripts/bench-stat.py -n 15 for the tie/win verdict; a one-shot ratio is
// noise (see bench-stat.py). ns-per-candidate ACROSS the N rows is the scaling readout.

#include <nanobench.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "hyle/container/string_algo.h"
#include "hyle/container/string_ref.h"

using hyle::container::find_path_matches;
using hyle::container::path_match_score;
using hyle::container::string_ref;

namespace {
namespace nb = ankerl::nanobench;

// Build `n` synthetic path candidates of a realistic shape, e.g.
// "src/module_<i>/components/widget_<i>_view.cpp" -- long enough that the O(L) scan
// dominates. Stored so the string_ref views stay valid for the whole bench.
struct Corpus {
  std::vector<std::string> store;
  std::vector<string_ref> refs;
  explicit Corpus(std::uint32_t n) {
    store.reserve(n);
    refs.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      char buf[128];
      int len = std::snprintf(buf, sizeof(buf),
                              "src/module_%u/components/widget_%u_view.cpp", i % 997,
                              i);
      store.emplace_back(buf, static_cast<std::size_t>(len));
    }
    // refs point into `store`; build them only after `store` is fully populated
    // (reserve kept it put, but take addresses once nothing else reallocates).
    for (auto& s : store)
      refs.push_back(string_ref{s.data(), static_cast<std::uint32_t>(s.size())});
  }
};

// The algorithm floor: is `pat` an ordered case-insensitive subsequence of `cand`?
// The minimum work ANY fuzzy matcher must do (membership + order), with NO scoring.
bool std_subsequence(string_ref cand, string_ref pat) noexcept {
  if (pat.size == 0) return true;
  std::uint32_t pi = 0;
  for (std::uint32_t ci = 0; ci < cand.size; ++ci) {
    char a = cand.data[ci];
    char b = pat.data[pi];
    if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
    if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
    if (a == b && ++pi == pat.size) return true;
  }
  return false;
}

}  // namespace

int main() {
  // "wgtview" hits widget_..._view: a realistic query. (Explicit string_ref, no _sr
  // literal -- keep the bench free of the literals-namespace using.)
  const string_ref pattern{"wgtview", 7};

  // ---- 1. scaling: ns-per-candidate across N (flat == O(N)) ----
  {
    nb::Bench b;
    b.title("fuzzy find_path_matches: ns/candidate vs N (flat => O(N))")
        .unit("candidate");
    for (std::uint32_t n : {1000u, 10000u, 100000u}) {
      Corpus c(n);
      std::uint32_t idx[20];
      int score[20];
      char name[48];
      std::snprintf(name, sizeof(name), "find_path_matches N=%u", n);
      b.batch(n).run(name, [&] {
        std::uint32_t m = find_path_matches(c.refs.data(), n, pattern, idx, score, 20);
        nb::doNotOptimizeAway(m);
      });
    }
  }

  // ---- 2. per-call scorer vs the subsequence floor (cost of the bonuses) ----
  {
    Corpus c(10000);
    nb::Bench b;
    b.title("fuzzy per-candidate: HYLE scorer vs std subsequence floor")
        .relative(true)
        .unit("candidate")
        .minEpochIterations(50)  // 10k-candidate sweeps are ~0.5ms; 50 epochs stabilizes
        .batch(c.refs.size());
    b.run("HYLE path_match_score (scores)", [&] {
      long acc = 0;
      for (const auto& r : c.refs) acc += path_match_score(r, pattern);
      nb::doNotOptimizeAway(acc);
    });
    b.run("std subsequence check (no score)", [&] {
      long acc = 0;
      for (const auto& r : c.refs) acc += std_subsequence(r, pattern) ? 1 : 0;
      nb::doNotOptimizeAway(acc);
    });
  }

  return 0;
}
