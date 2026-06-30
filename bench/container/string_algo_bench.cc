// string_algo utf8/hex bench (re-audit gap-fill #52). string_algo's UTF-8 ops
// (utf8_length / utf8_icmp / Utf8Decoder) are SCALAR byte loops -- the same shape
// that lost to SIMD in string_ref::find. The SOTA for UTF-8 counting/validation
// is simdutf (the SIMD library used by Node/Bun). This bench races HYLE's scalar
// utf8_length against simdutf::count_utf8 + validate_utf8 (gated -DHYLE_BENCH_
// SIMDUTF, since simdutf is an external clone) so the scalar-vs-SIMD gap is
// MEASURED, not assumed. hex_encode races a naive scalar reference (no single
// SOTA hex lib worth cloning; the point is to have a number). Win or lose, printed.

#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "hyle/container/string_algo.h"
#include "hyle/container/string_ref.h"

#if defined(HYLE_BENCH_SIMDUTF)
#  include "simdutf.h"
#endif

using hyle::container::hex_encode;
using hyle::container::string_ref;
using hyle::container::utf8_icmp;
using hyle::container::utf8_length;

namespace {
namespace nb = ankerl::nanobench;

// ~2 KB mixed ASCII + multibyte UTF-8 (Latin-1 accents + CJK), deterministic.
// Multibyte content is where a scalar decoder's per-byte branching hurts most vs
// simdutf's vectorized classification.
std::string gen_utf8() {
  std::string s;
  // Repeat a mixed block: ASCII word + 2-byte (é) + 3-byte (中) sequences.
  const char* block = "hello world ";       // ASCII
  const char* accent = "\xC3\xA9\xC3\xA8";  // é è (2-byte each)
  const char* cjk = "\xE4\xB8\xAD\xE6\x96\x87";  // 中文 (3-byte each)
  while (s.size() < 2000) {
    s += block;
    s += accent;
    s += cjk;
  }
  return s;
}
}  // namespace

int bench_string_algo() {
  const std::string u8 = gen_utf8();
  const string_ref sr{u8.data(), static_cast<std::uint32_t>(u8.size())};

  // ---- utf8_length: count code points (scalar decoder vs simdutf SIMD) -------
  // FAIRNESS: HYLE utf8_length does VALIDATE + COUNT in one pass (returns -1 on
  // malformed input). simdutf::count_utf8 counts ONLY (assumes valid) -- racing
  // against it alone is apples-to-oranges (HYLE does strictly more work). The
  // matched-contract rival is simdutf validate_utf8 + count_utf8 (same validate+
  // count semantics). Both shown: count-only marks the pure-SIMD ceiling, the
  // validate+count row is the honest same-contract comparison.
  {
    nb::Bench b;
    b.title("utf8 validate+count code points (~2KB mixed)")
        .relative(true)
        .batch(u8.size())
        .minEpochIterations(20000);
#if defined(HYLE_BENCH_SIMDUTF)
    b.run("simdutf validate_utf8+count_utf8 (same contract)", [&] {
      bool ok = simdutf::validate_utf8(u8.data(), u8.size());
      std::size_t n = ok ? simdutf::count_utf8(u8.data(), u8.size()) : 0;
      nb::doNotOptimizeAway(n);
    });
    b.run("simdutf::count_utf8 only (SIMD ceiling, no validate)", [&] {
      nb::doNotOptimizeAway(simdutf::count_utf8(u8.data(), u8.size()));
    });
#endif
    b.run("HYLE utf8_length (dispatched: scalar | simdutf-borrow when wired)", [&] {
      nb::doNotOptimizeAway(utf8_length(sr));
    });
  }

  // ---- utf8 validate only: HYLE has no validate-only entry, so this races -----
  // simdutf::validate_utf8 against HYLE utf8_length>=0 (which validates AND counts
  // -- still more work). Kept to show simdutf's pure-validate speed.
  {
    nb::Bench b;
    b.title("utf8 validate (~2KB mixed)")
        .relative(true)
        .batch(u8.size())
        .minEpochIterations(20000);
#if defined(HYLE_BENCH_SIMDUTF)
    b.run("simdutf::validate_utf8 (SIMD, validate only)", [&] {
      nb::doNotOptimizeAway(simdutf::validate_utf8(u8.data(), u8.size()));
    });
#endif
    b.run("HYLE utf8_length>=0 (dispatched validate+count)", [&] {
      nb::doNotOptimizeAway(utf8_length(sr) >= 0);
    });
  }

  // ---- utf8_icmp: case-insensitive compare two copies (whole-string scan) -----
  // No simdutf equivalent (it's a case-fold compare, not transcode); raced
  // against itself on equal inputs to measure the decode+fold loop throughput.
  {
    nb::Bench b;
    b.title("utf8_icmp equal ~2KB (decode+fold loop)")
        .relative(true)
        .batch(u8.size())
        .minEpochIterations(20000);
    b.run("HYLE utf8_icmp", [&] {
      nb::doNotOptimizeAway(utf8_icmp(sr, sr));
    });
  }

  // ---- hex_encode: 1 KB -> 2 KB hex (scalar table lookup) --------------------
  // No single SOTA hex lib worth cloning; race a naive byte-at-a-time reference
  // to confirm HYLE's table-lookup core isn't pathologically slow. Both scalar.
  {
    std::string bytes(1024, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<char>(i * 31 + 7);
    std::string out(bytes.size() * 2 + 1, '\0');
    nb::Bench b;
    b.title("hex_encode 1KB->2KB")
        .relative(true)
        .batch(bytes.size())
        .minEpochIterations(20000);
    b.run("naive byte-at-a-time hex (reference)", [&] {
      static const char* h = "0123456789abcdef";
      std::size_t w = 0;
      for (unsigned char c : bytes) {
        out[w++] = h[c >> 4];
        out[w++] = h[c & 0xF];
      }
      nb::doNotOptimizeAway(out.data());
    });
    b.run("HYLE hex_encode", [&] {
      hex_encode(bytes.data(), bytes.size(), out.data(), out.size());
      nb::doNotOptimizeAway(out.data());
    });
  }

  return 0;
}
