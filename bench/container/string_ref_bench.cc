// string_ref search/split API vs std::string_view (gap-fill #45). string_ref is
// HYLE's std::string_view replacement (conventions §7: same 16-byte POD + a
// null_terminated bit). Its whole search family (find/find_first_of/starts_with/
// split) had ZERO bench coverage. The find(char) path is a scalar loop documented
// to vectorize, while std::string_view::find(char) lowers to memchr (SIMD in libc)
// -- exactly where a naive scan can LOSE. Each op races the std::string_view
// equivalent (the drop-in SOTA), win or lose, std first as the reference.

#include <nanobench.h>

#include <algorithm>
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

  // ---- rfind(char): backward scan miss (no memchr equivalent -> scalar loop) --
  // find(char) got memchr; rfind has no portable reverse-memchr, so it stays a
  // backward byte loop. '~' is absent -> full backward scan. std::string_view::
  // rfind is the rival (libc may have memrchr on Linux; MSVC scalarizes too).
  // Opaque pos defeats hoisting symmetrically (same fairness fix as find above).
  {
    nb::Bench b;
    b.title("string_ref rfind(char) full-scan miss (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::rfind('~')", [&] {
      std::size_t pos = std::string_view::npos;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.rfind('~', pos));
    });
    b.run("HYLE string_ref::rfind('~')", [&] {
      std::uint32_t pos = string_ref::npos;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.rfind('~', pos));
    });
  }

  // ---- find_first_not_of(set): 256-entry membership table scan ---------------
  // DIFFERENT runtime strategy than the benched find_first_of (memchr-min): this
  // builds a 256-bool table then scans O(n). The set is all lowercase letters +
  // space (everything in the text) so the scan runs to the end (worst case: no
  // char is "not in set" until... actually all are, returns npos -> full scan).
  {
    const std::string set_s = "abcdefghijklmnopqrstuvwxyz ";
    const std::string_view sv_set{set_s};
    const string_ref sr_set{set_s.data(), static_cast<std::uint32_t>(set_s.size())};
    nb::Bench b;
    b.title("string_ref find_first_not_of(full alphabet) miss (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::find_first_not_of", [&] {
      std::size_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.find_first_not_of(sv_set, pos));
    });
    b.run("HYLE string_ref::find_first_not_of", [&] {
      std::uint32_t pos = 0;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.find_first_not_of(sr_set, pos));
    });
  }

  // ---- find_last_of(set): backward membership-table scan ---------------------
  // The reverse cousin: 256-table + backward scan. Set is digits (absent) -> full
  // backward scan to npos. Rival std::string_view::find_last_of.
  {
    const std::string set_s = "0123456789";
    const std::string_view sv_set{set_s};
    const string_ref sr_set{set_s.data(), static_cast<std::uint32_t>(set_s.size())};
    nb::Bench b;
    b.title("string_ref find_last_of(digits) miss (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::find_last_of", [&] {
      std::size_t pos = std::string_view::npos;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.find_last_of(sv_set, pos));
    });
    b.run("HYLE string_ref::find_last_of", [&] {
      std::uint32_t pos = string_ref::npos;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.find_last_of(sr_set, pos));
    });
  }

  // ---- trim(): two-sided find_first/last_not_of (the common parse hot path) ---
  // trim composes find_first_not_of + find_last_not_of. Pad the text with leading/
  // trailing spaces so trim does real work on both ends. Rival: std::string_view
  // two-sided trim via find_first_not_of/find_last_not_of + substr.
  {
    std::string padded = "   \t  " + text + "  \t   ";
    const std::string_view sv_pad{padded};
    const string_ref sr_pad{padded.data(), static_cast<std::uint32_t>(padded.size())};
    const string_ref ws{" \t\r\n"};
    const std::string_view sv_ws{" \t\r\n"};
    nb::Bench b;
    b.title("string_ref trim (both ends, ~4KB)").relative(true);
    b.run("std::string_view two-sided trim", [&] {
      std::size_t b0 = sv_pad.find_first_not_of(sv_ws);
      std::size_t e0 = sv_pad.find_last_not_of(sv_ws);
      std::string_view t = (b0 == std::string_view::npos)
                               ? std::string_view{}
                               : sv_pad.substr(b0, e0 - b0 + 1);
      nb::doNotOptimizeAway(t.size());
    });
    b.run("HYLE string_ref::trim", [&] {
      nb::doNotOptimizeAway(sr_pad.trim(ws).size);
    });
  }

  // ---- rfind(needle): backward substring search (separate strategy from rfind
  // char -- backward region_eq window walk, no memchr). 6B needle taken from near
  // the START so the backward scan runs most of the buffer. Rival std::string_
  // view::rfind(sv). Opaque pos defeats hoisting.
  {
    const std::string needle_s = text.substr(8, 6);
    const std::string_view sv_needle{needle_s};
    const string_ref sr_needle{needle_s.data(), static_cast<std::uint32_t>(needle_s.size())};
    nb::Bench b;
    b.title("string_ref rfind(needle) 6B (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::string_view::rfind(needle)", [&] {
      std::size_t pos = std::string_view::npos;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sv.rfind(sv_needle, pos));
    });
    b.run("HYLE string_ref::rfind(needle)", [&] {
      std::uint32_t pos = string_ref::npos;
      nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(sr.rfind(sr_needle, pos));
    });
  }

  // ---- count(char): full-scan accumulate, no early exit (clay cstrchar). std
  // has no direct string_view::count -> std::count over the range is the rival.
  {
    nb::Bench b;
    b.title("string_ref count(' ') full-scan (~4KB)")
        .relative(true)
        .batch(text.size());
    b.run("std::count(begin,end,' ')", [&] {
      nb::doNotOptimizeAway(std::count(sv.begin(), sv.end(), ' '));
    });
    b.run("HYLE string_ref::count", [&] {
      nb::doNotOptimizeAway(sr.count(' '));
    });
  }

  // ---- count/rfind at SCALE: 4KB / 64KB / 1MB (the SIMD-width regression gate) --------
  // count/rfind were unified from hand-rolled SSE2/AVX2 onto the simd/vec.h abstraction
  // (2026-07). This section is the regression watch: at larger sizes the vector core
  // dominates (call overhead amortized), so any per-iter penalty vs the old hand core shows
  // here. Rival is std (std::count / std::string_view::rfind, both libc/STL SIMD). A random
  // byte buffer with a rare needle (present ~1/256) exercises the full scan + a real hit.
  for (std::size_t kb : {std::size_t{4}, std::size_t{64}, std::size_t{1024}}) {
    std::string big;
    big.resize(kb * 1024);
    std::uint64_t x = 0xD1B54A32D192ED03ull;
    for (char& ch : big) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      ch = static_cast<char>('a' + (x % 25));  // 'a'..'y' -> 'z' is the rare needle (absent!)
    }
    // sprinkle 'z' at ~1/512 so rfind finds a real hit and count is nonzero-but-sparse
    for (std::size_t i = 251; i < big.size(); i += 512) big[i] = 'z';
    const std::string_view bsv{big};
    const string_ref bsr{big.data(), static_cast<std::uint32_t>(big.size())};

    nb::Bench b;
    b.title("string_ref count('z') vs std (" + std::to_string(kb) + "KB)")
        .relative(true).batch(big.size());
    // Defeat hoisting: the buffer+needle are loop-invariant, so a pure count call would be
    // computed ONCE and cached (nanobench then measures ~0). Make the needle opaque per-iter
    // (doNotOptimizeAway forces it into a register the optimizer can't predict), so each call
    // must actually scan. Applied SYMMETRICALLY to std and HYLE.
    b.run("std::count", [&] {
      char needle = 'z'; nb::doNotOptimizeAway(needle);
      nb::doNotOptimizeAway(std::count(bsv.begin(), bsv.end(), needle));
    });
    b.run("HYLE count", [&] {
      char needle = 'z'; nb::doNotOptimizeAway(needle);
      nb::doNotOptimizeAway(bsr.count(needle));
    });

    nb::Bench br;
    br.title("string_ref rfind('Q' miss) vs std (" + std::to_string(kb) + "KB)")
        .relative(true).batch(big.size());
    br.run("std::string_view::rfind", [&] {
      char needle = 'Q'; nb::doNotOptimizeAway(needle);  // absent -> full backward scan
      std::size_t pos = std::string_view::npos; nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(bsv.rfind(needle, pos));
    });
    br.run("HYLE rfind", [&] {
      char needle = 'Q'; nb::doNotOptimizeAway(needle);
      std::uint32_t pos = string_ref::npos; nb::doNotOptimizeAway(pos);
      nb::doNotOptimizeAway(bsr.rfind(needle, pos));
    });
  }

  return 0;
}
