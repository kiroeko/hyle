// Standalone table-hash benchmark. THREE+ groups per workload:
//   * HYLE hash_bytes / Hash<uint64> (mix64)   -- what HYLE ships
//   * raw XXH3_64bits + wyhash                  -- SOTA algorithm references for the
//                                                  length-adaptive HYLE product path
//   * std::hash                                 -- the std baseline
// Both xxhash and wyhash are vendored product dependencies. Benchmark CMake may
// still gate the raw wyhash comparison row independently of product behavior.
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
#include "hyle/container/swiss_table.h"
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

  // ---- normal short-string policy cost. Keep the stateful hashers outside the
  // timed lambda: real tables construct/copy their policy once, so measuring
  // process-seed initialization on every key would be the wrong contract.
  {
    std::vector<std::string> keys;
    keys.reserve(4096);
    for (std::size_t i = 0; i < 4096; ++i) {
      keys.push_back(random_string(8 + (i % 25), 0x51A7u + i));
    }
    hyle::container::StringHash process_seeded;
    hyle::container::StringHash deterministic{0x9e3779b97f4a7c15ull};
    hyle::container::HardenedStringHash hardened{
        0x0123456789abcdefull, 0xfedcba9876543210ull};
    nb::Bench b;
    b.title("hash short string (4096 varied 8..32B keys)")
        .relative(true)
        .batch(keys.size())
        .minEpochIterations(2000);
    b.run("HYLE StringHash (cached process seed)", [&] {
      std::uint64_t acc = 0;
      for (const auto& key : keys) acc ^= process_seeded(key);
      nb::doNotOptimizeAway(acc);
    });
    b.run("HYLE StringHash (explicit deterministic seed)", [&] {
      std::uint64_t acc = 0;
      for (const auto& key : keys) acc ^= deterministic(key);
      nb::doNotOptimizeAway(acc);
    });
    b.run("HYLE HardenedStringHash (SipHash-2-4)", [&] {
      std::uint64_t acc = 0;
      for (const auto& key : keys) acc ^= hardened(key);
      nb::doNotOptimizeAway(acc);
    });
    b.run("std::hash<string_view> (baseline)", [&] {
      std::uint64_t acc = 0;
      const std::hash<std::string_view> hash;
      for (const auto& key : keys) acc ^= hash(key);
      nb::doNotOptimizeAway(acc);
    });
  }

  // ---- real heterogeneous FlatHashMap hit/miss lookup. Both rows use the
  // exact same table, allocation, equality, and query ownership; only the hash
  // policy changes. Half the string_view queries hit and half miss.
  {
    using FastMap = hyle::container::FlatHashMap<
        std::string, std::uint64_t, hyle::container::StringHash,
        hyle::container::StringEqual>;
    using HardenedMap = hyle::container::FlatHashMap<
        std::string, std::uint64_t, hyle::container::HardenedStringHash,
        hyle::container::StringEqual>;
    constexpr std::size_t kKeys = 4096;
    std::vector<std::string> keys;
    std::vector<std::string> misses;
    keys.reserve(kKeys);
    misses.reserve(kKeys);
    for (std::size_t i = 0; i < kKeys; ++i) {
      keys.push_back(random_string(8 + (i % 25), 0x7AB1Eu + i));
      misses.push_back(keys.back());
      misses.back().push_back(static_cast<char>(i));
    }

    FastMap fast;
    HardenedMap hardened(
        hyle::mem::SystemAllocator{},
        hyle::container::HardenedStringHash{0x0123456789abcdefull,
                                            0xfedcba9876543210ull},
        hyle::container::StringEqual{});
    fast.reserve(kKeys);
    hardened.reserve(kKeys);
    for (std::size_t i = 0; i < kKeys; ++i) {
      fast.insert_or_assign(keys[i], i);
      hardened.insert_or_assign(keys[i], i);
    }

    nb::Bench b;
    b.title("FlatHashMap<string> heterogeneous lookup (4096, 50% hit)")
        .relative(true)
        .batch(kKeys)
        .minEpochIterations(200);
    std::size_t iteration = 0;
    b.run("HYLE FlatHashMap + process-seeded StringHash", [&] {
      std::uint64_t acc = 0;
      for (std::size_t i = 0; i < kKeys; ++i) {
        const std::string& query = (i & 1u) == 0 ? keys[i] : misses[i];
        if (const auto* value = fast.find(std::string_view(query))) acc ^= *value;
      }
      nb::doNotOptimizeAway(acc + iteration++);
    });
    b.run("HYLE FlatHashMap + keyed SipHash-2-4", [&] {
      std::uint64_t acc = 0;
      for (std::size_t i = 0; i < kKeys; ++i) {
        const std::string& query = (i & 1u) == 0 ? keys[i] : misses[i];
        if (const auto* value = hardened.find(std::string_view(query))) {
          acc ^= *value;
        }
      }
      nb::doNotOptimizeAway(acc + iteration++);
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

  // ---- normal finite floating-point table keys. HYLE canonicalizes signed
  // zero then bit-casts into the same inline mixer as uint32/uint64; std::hash is
  // retained as the equality-compatible baseline.
  {
    std::vector<float> keys(4096);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = static_cast<float>(i) * 0.25f + 0.125f;
    }
    nb::Bench b;
    b.title("hash float (4096 finite keys)")
        .relative(true)
        .batch(keys.size())
        .minEpochIterations(50000);
    b.run("HYLE Hash<float> (canonical bits + mix64)", [&] {
      std::uint64_t acc = 0;
      hyle::container::Hash<float> h;
      for (float key : keys) acc ^= h(key);
      nb::doNotOptimizeAway(acc);
    });
    b.run("std::hash<float> (baseline)", [&] {
      std::uint64_t acc = 0;
      std::hash<float> h;
      for (float key : keys) acc ^= h(key);
      nb::doNotOptimizeAway(acc);
    });
  }
  {
    std::vector<double> keys(4096);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      keys[i] = static_cast<double>(i) * 0.25 + 0.125;
    }
    nb::Bench b;
    b.title("hash double (4096 finite keys)")
        .relative(true)
        .batch(keys.size())
        .minEpochIterations(50000);
    b.run("HYLE Hash<double> (canonical bits + mix64)", [&] {
      std::uint64_t acc = 0;
      hyle::container::Hash<double> h;
      for (double key : keys) acc ^= h(key);
      nb::doNotOptimizeAway(acc);
    });
    b.run("std::hash<double> (baseline)", [&] {
      std::uint64_t acc = 0;
      std::hash<double> h;
      for (double key : keys) acc ^= h(key);
      nb::doNotOptimizeAway(acc);
    });
  }

  // ---- explicitly bytewise 16B key vs a direct hash_bytes call. This models a
  // representation-equality map key whose owner deliberately opted in. Both
  // paths inline the <=16B mix (absl/F14 style); the rows are facade parity.
  {
    struct Key16 { std::uint64_t a, b; };
    std::mt19937_64 rng(7);
    std::vector<Key16> keys(4096);
    for (auto& k : keys) { k.a = rng(); k.b = rng(); }
    nb::Bench b;
    b.title("hash 16B POD key (4096)").relative(true).batch(keys.size()).minEpochIterations(20000);
    b.run("HYLE BytewiseHash<Key16> (explicit inline mix)", [&] {
      std::uint64_t acc = 0;
      hyle::container::BytewiseHash<Key16> h;
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
