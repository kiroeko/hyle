// string_algo UTF-8/ASCII/hex bench. utf8_validate is the product's dispatched
// one-pass validation path; utf8_length is its validate+count path.
// ASCII case conversion and hex encode/decode are included specifically to decide
// whether another runtime SIMD kernel is justified; matched scalar references
// make that decision evidence-based.

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
using hyle::container::hex_decode;
using hyle::container::string_ref;
using hyle::container::to_lower;
using hyle::container::utf8_length;
using hyle::container::utf8_validate;

namespace {
namespace nb = ankerl::nanobench;

// Deterministic mixed ASCII + multibyte UTF-8 (Latin-1 accents + CJK).  Always
// append complete blocks: resizing through a code point would accidentally turn
// a valid workload into a truncated-tail workload.
std::string gen_mixed_utf8(std::size_t minimum_size) {
  std::string s;
  const char* block = "hello world ";            // ASCII
  const char* accent = "\xC3\xA9\xC3\xA8";       // é è (2-byte each)
  const char* cjk = "\xE4\xB8\xAD\xE6\x96\x87";  // 中文 (3-byte each)
  while (s.size() < minimum_size) {
    s += block;
    s += accent;
    s += cjk;
  }
  return s;
}

void bench_utf8_case(const char* label, const std::string& input) {
  const string_ref text{input.data(), static_cast<std::uint32_t>(input.size())};
  const std::uint64_t minimum_iterations =
      input.size() <= 64 ? 200'000 : 20'000;

  // utf8_length's product contract is validate + count, returning -1 for
  // malformed input.  The direct simdutf row implements that exact contract;
  // count_utf8 alone is intentionally absent because it assumes valid input.
  {
    nb::Bench bench;
    bench.title(std::string("utf8 validate+count: ") + label)
        .relative(true)
        .batch(input.size())
        .minEpochIterations(minimum_iterations);
#if defined(HYLE_BENCH_SIMDUTF)
    bench.run("simdutf validate+count -> -1 invalid (same contract)", [&] {
      const std::int64_t result =
          simdutf::validate_utf8(input.data(), input.size())
              ? static_cast<std::int64_t>(
                    simdutf::count_utf8(input.data(), input.size()))
              : std::int64_t{-1};
      nb::doNotOptimizeAway(result);
    });
#endif
    bench.run("HYLE utf8_length (simdutf 9.1 product path)",
              [&] { nb::doNotOptimizeAway(utf8_length(text)); });
  }

  // validate-only has a naturally identical direct simdutf contract.
  {
    nb::Bench bench;
    bench.title(std::string("utf8 validate-only: ") + label)
        .relative(true)
        .batch(input.size())
        .minEpochIterations(minimum_iterations);
#if defined(HYLE_BENCH_SIMDUTF)
    bench.run("simdutf validate_utf8 (same contract)", [&] {
      nb::doNotOptimizeAway(simdutf::validate_utf8(input.data(), input.size()));
    });
#endif
    bench.run("HYLE utf8_validate (simdutf 9.1 product path)",
              [&] { nb::doNotOptimizeAway(utf8_validate(text)); });
  }
}
}  // namespace

int bench_string_algo() {
  // Common consumer shapes: tiny protocol/path strings, long ASCII payloads,
  // mixed-language text, malformed input found after substantial scanning, and
  // a truncated multibyte sequence at the buffer tail.
  const std::string ascii_short = "GET /v1/models";
  const std::string ascii_long(4096, 'a');
  const std::string mixed_short = gen_mixed_utf8(24);
  const std::string mixed_long = gen_mixed_utf8(4096);
  std::string invalid_middle(4096, 'a');
  invalid_middle[invalid_middle.size() / 2] = static_cast<char>(0xFF);
  std::string truncated_tail(4096, 'a');
  truncated_tail.back() = static_cast<char>(0xE2);  // 3-byte lead, no payload.

  bench_utf8_case("ASCII short (14 B)", ascii_short);
  bench_utf8_case("ASCII long (4 KiB)", ascii_long);
  bench_utf8_case("mixed UTF-8 short", mixed_short);
  bench_utf8_case("mixed UTF-8 long (~4 KiB)", mixed_long);
  bench_utf8_case("invalid byte in middle (4 KiB)", invalid_middle);
  bench_utf8_case("truncated multibyte tail (4 KiB)", truncated_tail);

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

  // ---- hex_decode: 2 KB hex -> 1 KB bytes, including validation -------------
  {
    std::string bytes(1024, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<char>(i * 31 + 7);
    std::string hex(bytes.size() * 2 + 1, '\0');
    std::string out(bytes.size(), '\0');
    hex_encode(bytes.data(), bytes.size(), hex.data(), hex.size());
    const string_ref input{
        hex.data(), static_cast<std::uint32_t>(bytes.size() * 2)};
    nb::Bench b;
    b.title("hex_decode 2KB->1KB (validated)")
        .relative(true)
        .batch(bytes.size())
        .minEpochIterations(20000);
    b.run("naive validated hex (reference)", [&] {
      bool valid = true;
      for (std::size_t i = 0; i < bytes.size(); ++i) {
        const int high = hyle::container::hex_value(hex[i * 2]);
        const int low = hyle::container::hex_value(hex[i * 2 + 1]);
        valid &= high >= 0 && low >= 0;
        out[i] = static_cast<char>((high << 4) | low);
      }
      nb::doNotOptimizeAway(valid);
      nb::doNotOptimizeAway(out.data());
    });
    b.run("HYLE hex_decode", [&] {
      hex_decode(input, out.data(), out.size());
      nb::doNotOptimizeAway(out.data());
    });
  }

  // ---- ASCII lower: compiler auto-vectorization ceiling vs HYLE API --------
  {
    std::string input(4096, '\0');
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<char>('A' + (i % 26));
    std::string out(input.size() + 1, '\0');
    const string_ref text{input.data(),
                          static_cast<std::uint32_t>(input.size())};
    nb::Bench b;
    b.title("ASCII lowercase 4KB")
        .relative(true)
        .batch(input.size())
        .minEpochIterations(20000);
    b.run("scalar branch lower (reference)", [&] {
      for (std::size_t i = 0; i < input.size(); ++i) {
        const char c = input[i];
        out[i] = (c >= 'A' && c <= 'Z')
                     ? static_cast<char>(c + ('a' - 'A'))
                     : c;
      }
      nb::doNotOptimizeAway(out.data());
    });
    b.run("HYLE to_lower", [&] {
      to_lower(text, out.data(), static_cast<std::uint32_t>(out.size()));
      nb::doNotOptimizeAway(out.data());
    });
  }

  return 0;
}
