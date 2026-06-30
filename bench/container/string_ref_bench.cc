// string_ref search/split API vs std::string_view (gap-fill #45). string_ref is
// HYLE's std::string_view replacement (conventions §7: same 16-byte POD + a
// null_terminated bit). Its whole search family (find/find_first_of/starts_with/
// split) had ZERO bench coverage. The find(char) path is a scalar loop documented
// to vectorize, while std::string_view::find(char) lowers to memchr (SIMD in libc)
// -- exactly where a naive scan can LOSE. Each op races the std::string_view
// equivalent (the drop-in SOTA), win or lose, std first as the reference.

#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "hyle/container/string_ref.h"

using hyle::container::string_ref;

namespace {
namespace nb = ankerl::nanobench;

// ~4 KiB of word-like tokens separated by spaces, deterministic (xorshift). Long
// enough that a full scan is measurable, small enough to stay in L1 (isolates the
// scan core, not memory bandwidth). Alphabet is a-z + space only, so digits and
// '~' are guaranteed ABSENT -> clean full-scan-miss workloads.
std::string gen_text() {
  std::string s;
  s.reserve(4096);
  std::uint64_t x = 0x9E3779B97F4A7C15ull;
  const char* alpha = "abcdefghijklmnopqrstuvwxyz";
  while (s.size() < 4000) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;  // xorshift64
    int wlen = 3 + static_cast<int>(x % 8);
    for (int i = 0; i < wlen; ++i) s.push_back(alpha[(x >> (i * 3)) % 26]);
    s.push_back(' ');
  }
  return s;
}
}  // namespace

int bench_string_ref() {
  const std::string text = gen_text();
  const std::string_view sv{text};
  const string_ref sr{text.data(), static_cast<std::uint32_t>(text.size())};

  // ---- find(char): full-scan MISS (purest memchr-vs-scalar signal) -----------
  // '~' is absent -> both scan the whole buffer. std::string_view::find lowers to
  // memchr (SIMD); HYLE's is a byte loop the compiler may auto-vectorize.
  // FAIRNESS: the search inputs are loop-invariant, so the compiler can HOIST the
  // whole call out of the bench loop (compute once) -- and it does this for std's
  // intrinsic but not always for HYLE's hand-loop, which would be a measurement
  // artifact, not a real gap. Defeat it SYMMETRICALLY: make `pos` opaque (via
  // doNotOptimizeAway) so neither side can precompute the result.
  {
    nb::Bench b;
    b.title("string_ref find(char) full-scan miss (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::find('~')", [&] {
      std::size_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.find('~', pos));
    });
    b.run("HYLE string_ref::find('~')", [&] {
      std::uint32_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.find('~', pos));
    });
  }

  // ---- find(needle): substring search, hit near the end ----------------------
  // 6-byte slice taken from near the end -> a hit after a long scan. Both are
  // naive O(n*m); std::string_view::find is the SOTA drop-in. Opaque `pos` defeats
  // hoisting symmetrically (see the find(char) note above).
  {
    const std::string needle_s = text.substr(text.size() - 24, 6);
    const std::string_view sv_needle{needle_s};
    const string_ref sr_needle{needle_s.data(),
                               static_cast<std::uint32_t>(needle_s.size())};
    nb::Bench b;
    b.title("string_ref find(needle) 6B substring (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::find(needle)", [&] {
      std::size_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.find(sv_needle, pos));
    });
    b.run("HYLE string_ref::find(needle)", [&] {
      std::uint32_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.find(sr_needle, pos));
    });
  }

  // ---- find_first_of(set): full-scan miss (4-char absent set) ----------------
  // digits absent from text -> both scan the whole buffer testing membership in a
  // 4-char set. HYLE's inner set.find is O(n*m); the std equivalent is the rival.
  // Opaque `pos` defeats hoisting symmetrically.
  {
    const std::string set_s = "0123";
    const std::string_view sv_set{set_s};
    const string_ref sr_set{set_s.data(),
                            static_cast<std::uint32_t>(set_s.size())};
    nb::Bench b;
    b.title("string_ref find_first_of(4-char set) miss (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::find_first_of(set)", [&] {
      std::size_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.find_first_of(sv_set, pos));
    });
    b.run("HYLE string_ref::find_first_of(set)", [&] {
      std::uint32_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.find_first_of(sr_set, pos));
    });
  }

  // ---- starts_with(prefix): 16-byte prefix match (both memcmp) ---------------
  {
    const std::string pre_s = text.substr(0, 16);
    const std::string_view sv_pre{pre_s};
    const string_ref sr_pre{pre_s.data(),
                            static_cast<std::uint32_t>(pre_s.size())};
    nb::Bench b;
    b.title("string_ref starts_with(16B prefix)").relative(true);
    b.run("std::string_view::starts_with",
          [&] { nb::doNotOptimizeAway(sv.starts_with(sv_pre)); });
    b.run("HYLE string_ref::starts_with",
          [&] { nb::doNotOptimizeAway(sr.starts_with(sr_pre)); });
  }

  // ---- split: tokenize the whole text by space (the no-alloc headline) -------
  // HYLE split(&s,c) returns sub-views, zero allocation. The std equivalent is a
  // string_view + find + remove_prefix loop (also zero allocation). Both walk the
  // whole buffer; measures the per-token split cost.
  {
    nb::Bench b;
    b.title("string_ref split whole text by ' ' (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view find+remove_prefix loop", [&] {
      std::string_view s = sv;
      std::uint32_t acc = 0;
      while (!s.empty()) {
        std::size_t at = s.find(' ');
        std::string_view tok = (at == std::string_view::npos) ? s : s.substr(0, at);
        acc += static_cast<std::uint32_t>(tok.size());
        if (at == std::string_view::npos)
          s = {};
        else
          s.remove_prefix(at + 1);
      }
      nb::doNotOptimizeAway(acc);
    });
    b.run("HYLE string_ref::split", [&] {
      string_ref s = sr;
      std::uint32_t acc = 0;
      while (!s.empty()) {
        string_ref tok = hyle::container::split(&s, ' ');
        acc += tok.size;
      }
      nb::doNotOptimizeAway(acc);
    });
  }

  return 0;
}
