// hash bench (task #23): HYLE's runtime hasher had NO standalone benchmark -- it was only
// ever exercised inside SwissTable. README claims XXH3 as the default hasher; this races it.
// THREE+ groups per workload:
//   * HYLE hash_bytes / Hash<uint64> (mix64)   -- what HYLE ships
//   * raw XXH3_64bits + wyhash                  -- SOTA references (XXH3 is what HYLE wraps,
//                                                  so HYLE-vs-rawXXH3 = wrapper overhead;
//                                                  wyhash is the rival fast hash)
//   * std::hash                                 -- the std baseline
// xxhash is vendored (third_party/xxhash, XXH_INLINE_ALL). wyhash is a sibling clone, so it
// is gated behind HYLE_BENCH_WYHASH (header-only -> just an include path).
#include <nanobench.h>

#define XXH_INLINE_ALL
#include "xxhash.h"

#if defined(HYLE_BENCH_WYHASH)
#  include "wyhash.h"
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "hyle/container/hash.h"

namespace {
namespace nb = ankerl::nanobench;

#if defined(HYLE_BENCH_WYHASH)
// wyhash's published default secret (the 4 constants from wyhash.h's make_secret default).
const std::uint64_t kWySecret[4] = {0xa0761d6478bd642full, 0xe7037ed1a0b428dbull,
                                    0x8ebc6af09c88c6e3ull, 0x589965cc75374cc3ull};
#endif

std::string random_string(std::size_t len, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::string s(len, '\0');
  for (auto& c : s) c = static_cast<char>(rng() & 0xFF);
  return s;
}
}  // namespace

int main() {
  // ---- byte/string hashing across lengths (XXH3 is a byte hasher) ----
  for (std::size_t len : {8u, 16u, 64u, 256u}) {
    const std::string s = random_string(len, 0xABCDEF ^ len);
    const std::string_view sv(s);
    nb::Bench b;
    b.title("hash " + std::to_string(len) + "B")
        .relative(true)
        .batch(len)  // ns/op per byte; throughput visible
        .minEpochIterations(2000);

    b.run("HYLE hash_bytes (XXH3 wrapper)", [&] {
      nb::doNotOptimizeAway(hyle::container::hash_bytes(s.data(), s.size()));
    });
    b.run("raw XXH3_64bits (SOTA, what we wrap)", [&] {
      nb::doNotOptimizeAway(XXH3_64bits(s.data(), s.size()));
    });
#if defined(HYLE_BENCH_WYHASH)
    b.run("wyhash (SOTA rival)", [&] {
      nb::doNotOptimizeAway(wyhash(s.data(), s.size(), 0, kWySecret));
    });
#endif
    b.run("std::hash<string_view> (baseline)", [&] {
      nb::doNotOptimizeAway(std::hash<std::string_view>{}(sv));
    });
  }

  // ---- integer hashing (HYLE mix64: the find-miss-critical path) ----
  {
    std::mt19937_64 rng(42);
    std::vector<std::uint64_t> keys(4096);
    for (auto& k : keys) k = rng();
    nb::Bench b;
    b.title("hash uint64 (4096 keys)").relative(true).batch(keys.size()).minEpochIterations(2000);

    b.run("HYLE Hash<uint64> (mix64)", [&] {
      std::uint64_t acc = 0;
      hyle::container::Hash<std::uint64_t> h;
      for (auto k : keys) acc ^= h(k);
      nb::doNotOptimizeAway(acc);
    });
#if defined(HYLE_BENCH_WYHASH)
    b.run("wyhash64 (SOTA rival)", [&] {
      std::uint64_t acc = 0;
      for (auto k : keys) acc ^= wyhash64(k, 0x9E3779B97F4A7C15ull);
      nb::doNotOptimizeAway(acc);
    });
#endif
    b.run("std::hash<uint64> (baseline, often identity)", [&] {
      std::uint64_t acc = 0;
      std::hash<std::uint64_t> h;
      for (auto k : keys) acc ^= h(k);
      nb::doNotOptimizeAway(acc);
    });
  }

  // ---- small POD key (16B struct): typed Hash<T> inline path vs the old hash_bytes call.
  // This is the real map-key hot path. Hash<T> now inlines a mix for <=16B (absl/F14 style)
  // instead of calling the out-of-line XXH3 -- the row "HYLE Hash<Key16> (inline)" should
  // beat "explicit hash_bytes(&k,16)" (what the generic used to do).
  {
    struct Key16 { std::uint64_t a, b; };
    std::mt19937_64 rng(7);
    std::vector<Key16> keys(4096);
    for (auto& k : keys) { k.a = rng(); k.b = rng(); }
    nb::Bench b;
    b.title("hash 16B POD key (4096)").relative(true).batch(keys.size()).minEpochIterations(20000);
    b.run("HYLE Hash<Key16> (inline mix)", [&] {
      std::uint64_t acc = 0;
      hyle::container::Hash<Key16> h;
      for (const auto& k : keys) acc ^= h(k);
      nb::doNotOptimizeAway(acc);
    });
    b.run("explicit hash_bytes(&k,16) (old path, XXH3 call)", [&] {
      std::uint64_t acc = 0;
      for (const auto& k : keys) acc ^= hyle::container::hash_bytes(&k, sizeof(k));
      nb::doNotOptimizeAway(acc);
    });
#if defined(HYLE_BENCH_WYHASH)
    b.run("wyhash(&k,16) (SOTA rival)", [&] {
      std::uint64_t acc = 0;
      for (const auto& k : keys) acc ^= wyhash(&k, 16, 0, kWySecret);
      nb::doNotOptimizeAway(acc);
    });
#endif
  }

  return 0;
}
