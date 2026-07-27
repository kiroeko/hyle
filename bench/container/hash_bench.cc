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
// Upstream wyhash's rarely-used make_secret helpers are non-inline globals in
// the header. The product TU also includes that header, so give benchmark-local
// copies distinct names (ELF rejects the duplicate that COFF happens to fold).
#  define sprp hyle_bench_wyhash_sprp
#  define is_prime hyle_bench_wyhash_is_prime
#  include "wyhash.h"
#  undef is_prime
#  undef sprp
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "hyle/container/hash.h"
#include "hyle/platform/cpu.h"

#if defined(HYLE_HASH_PEER_AVX2)
std::uint64_t hyle_bench_xxh3_avx2(const void*, std::size_t,
                                   std::uint64_t) noexcept;
#endif
#if defined(HYLE_HASH_PEER_AVX512)
std::uint64_t hyle_bench_xxh3_avx512(const void*, std::size_t,
                                     std::uint64_t) noexcept;
#endif

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
#if defined(HYLE_HASH_PEER_AVX2)
  // Forced vector TUs must preserve XXH3's algorithm output. Keep this correctness
  // gate next to the benchmark that justifies product dispatch.
  if (hyle::platform::supports(hyle::platform::Isa::kAvx2)) {
    const std::string probe = random_string(4096, 0xA5A5A5A5u);
    if (hyle_bench_xxh3_avx2(probe.data(), probe.size(), 17) !=
        XXH3_64bits_withSeed(probe.data(), probe.size(), 17))
      return 2;
  }
#endif
#if defined(HYLE_HASH_PEER_AVX512)
  if (hyle::platform::supports(hyle::platform::Isa::kAvx512)) {
    const std::string probe = random_string(4096, 0x5A5A5A5Au);
    if (hyle_bench_xxh3_avx512(probe.data(), probe.size(), 17) !=
        XXH3_64bits_withSeed(probe.data(), probe.size(), 17))
      return 3;
  }
#endif
  // ---- byte/string hashing across lengths (XXH3 is a byte hasher) ----
  // Rotate through a POOL of distinct random strings per length (not one fixed
  // string): a single repeated input sits in L1 + trains the length-dispatch branch
  // perfectly, making tiny inputs read sub-cycle (unrealistic). smhasher-style varied
  // content keeps the measurement honest. We index the pool by a running counter.
  constexpr std::size_t kPool = 256;
  for (std::size_t len : {8u, 16u, 24u, 32u, 48u, 64u, 96u, 128u, 256u,
                          512u, 768u, 1024u, 4096u, 65536u}) {
    std::vector<std::string> pool;
    pool.reserve(kPool);
    for (std::size_t i = 0; i < kPool; ++i)
      pool.push_back(random_string(len, (0xABCDEF ^ len) * 1000003u + i));
    std::size_t it = 0;
    nb::Bench b;
    b.title("hash " + std::to_string(len) + "B")
        .relative(true)
        .batch(len)  // ns/op per byte; throughput visible. NOTE: multiply by `len`
                     // for per-hash cost -- small-input rows are fixed-overhead-bound,
                     // so per-byte is LOW there without meaning the algorithm is faster.
        .minEpochIterations(std::max<std::uint64_t>(
            4, 512000 / static_cast<std::uint64_t>(len)));

    // HYLE hash_bytes is best-of-breed by length: <=16B inline mix; medium
    // inputs use the shipping architecture's measured choice; longer inputs use
    // wyhash or the shipping platform's measured wide XXH3 crossover.
    const char* hyle_label =
        len <= 16 ? "HYLE hash_bytes (inline mix)"
                  : (len <= 128 ? "HYLE hash_bytes (runtime medium)"
                                : "HYLE hash_bytes (runtime long best-of)");
    b.run(hyle_label, [&] {
      const auto& s = pool[it++ & (kPool - 1)];
      nb::doNotOptimizeAway(hyle::container::hash_bytes(s.data(), s.size()));
    });
    b.run("raw XXH3_64bits (SOTA byte hasher)", [&] {
      const auto& s = pool[it++ & (kPool - 1)];
      nb::doNotOptimizeAway(XXH3_64bits(s.data(), s.size()));
    });
#if defined(HYLE_HASH_PEER_AVX2)
    if (len > 240 && hyle::platform::supports(hyle::platform::Isa::kAvx2)) {
      b.run("raw XXH3 AVX2 (runtime-safe peer)", [&] {
        const auto& s = pool[it++ & (kPool - 1)];
        nb::doNotOptimizeAway(
            hyle_bench_xxh3_avx2(s.data(), s.size(), 0));
      });
    }
#endif
#if defined(HYLE_HASH_PEER_AVX512)
    if (len > 240 && hyle::platform::supports(hyle::platform::Isa::kAvx512)) {
      b.run("raw XXH3 AVX-512 (runtime-safe peer)", [&] {
        const auto& s = pool[it++ & (kPool - 1)];
        nb::doNotOptimizeAway(
            hyle_bench_xxh3_avx512(s.data(), s.size(), 0));
      });
    }
#endif
#if defined(HYLE_BENCH_WYHASH)
    b.run("wyhash (SOTA rival)", [&] {
      const auto& s = pool[it++ & (kPool - 1)];
      nb::doNotOptimizeAway(wyhash(s.data(), s.size(), 0, kWySecret));
    });
#endif
    b.run("std::hash<string_view> (baseline)", [&] {
      const auto& s = pool[it++ & (kPool - 1)];
      nb::doNotOptimizeAway(std::hash<std::string_view>{}(std::string_view(s)));
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
  // This is the real map-key hot path. Hash<T> and a compile-time-sized
  // hash_bytes call both inline the <=16B mix (absl/F14 style). Their rows are
  // a facade-parity check: neither should retain the historical out-of-line
  // XXH3 seam, and neither is expected to beat the other materially.
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
    b.run("explicit hash_bytes(&k,16) (inline parity)", [&] {
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
