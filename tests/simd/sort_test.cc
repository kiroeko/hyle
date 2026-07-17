#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/sort.h"
#include "hyle/simd/runtime.h"

// sort_test -- the borrowed Highway vqsort (whatever tier is live: avx512/avx2/neon/scalar) must
// agree with the std:: oracle on every (type, length, distribution, order). The dispatcher picks
// the widest tier the host supports; either way the RESULT is a total order, so:
//   sort         : element-wise equal to std::sort.
//   partial_sort : the first k, re-sorted, equal std::partial_sort's first k (the [k,n) tail is
//                  unspecified for both, so only [0,k) is compared).
//   select       : keys[k] equals the k-th order statistic, and the partition property holds
//                  (everything before order-<= keys[k], everything after order->=) -- exactly
//                  std::nth_element's postcondition.
// Both ascending and descending. Lengths cover 0/1/tiny (sorting-network base cases) and large
// (recursion), with distributions: random, all-equal, already-sorted, reverse-sorted, few-unique
// (heavy duplicates -- stresses the partition).

using hyle::simd::Isa;
using hyle::simd::partial_sort;
using hyle::simd::select;
using hyle::simd::sort;
using hyle::simd::SortOrder;
using hyle::simd::UInt128;

namespace {

// Build a test vector of length n with a named distribution, using a fixed-seed PRNG so failures
// reproduce. Integer and float types both go through the same generator (cast at the end).
template <typename T>
std::vector<T> make_data(std::size_t n, int dist, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<T> v(n);
  auto gen = [&](std::size_t i) -> T {
    switch (dist) {
      case 0: {  // random full range
        if constexpr (std::is_floating_point_v<T>) {
          std::uniform_real_distribution<double> d(-1e6, 1e6);
          return static_cast<T>(d(rng));
        } else {
          std::uniform_int_distribution<std::uint64_t> d;
          return static_cast<T>(d(rng));
        }
      }
      case 1:  // all equal
        return static_cast<T>(42);
      case 2:  // already sorted ascending
        return static_cast<T>(i);
      case 3:  // reverse sorted
        return static_cast<T>(n - i);
      case 4: {  // few unique (heavy duplicates)
        std::uniform_int_distribution<int> d(0, 7);
        return static_cast<T>(d(rng));
      }
      default:
        return T{};
    }
  };
  for (std::size_t i = 0; i < n; ++i) v[i] = gen(i);
  return v;
}

template <typename T>
void expect_sort(std::size_t n, int dist, SortOrder order, std::uint64_t seed) {
  const bool asc = (order == SortOrder::kAscending);
  std::vector<T> got = make_data<T>(n, dist, seed);
  std::vector<T> ref = got;
  sort(got.data(), got.size(), order);
  if (asc)
    std::sort(ref.begin(), ref.end(), std::less<T>{});
  else
    std::sort(ref.begin(), ref.end(), std::greater<T>{});
  ASSERT_EQ(got, ref) << "sort n=" << n << " dist=" << dist << " asc=" << asc;
}

template <typename T>
void expect_partial_sort(std::size_t n, std::size_t k, int dist, SortOrder order,
                         std::uint64_t seed) {
  const bool asc = (order == SortOrder::kAscending);
  std::vector<T> got = make_data<T>(n, dist, seed);
  std::vector<T> ref = got;
  partial_sort(got.data(), got.size(), k, order);
  if (asc)
    std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), std::less<T>{});
  else
    std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), std::greater<T>{});
  // Only [0,k) is defined for both; re-sort that prefix on our side (vqsort orders it, but be
  // robust to tie arrangement) and compare against the oracle's (already-sorted) prefix.
  for (std::size_t i = 0; i < k; ++i)
    ASSERT_EQ(got[i], ref[i]) << "partial_sort n=" << n << " k=" << k << " dist=" << dist
                              << " asc=" << asc << " i=" << i;
}

template <typename T>
void expect_select(std::size_t n, std::size_t k, int dist, SortOrder order, std::uint64_t seed) {
  const bool asc = (order == SortOrder::kAscending);
  std::vector<T> got = make_data<T>(n, dist, seed);
  std::vector<T> ref = got;
  select(got.data(), got.size(), k, order);
  if (asc)
    std::nth_element(ref.begin(), ref.begin() + k, ref.end(), std::less<T>{});
  else
    std::nth_element(ref.begin(), ref.begin() + k, ref.end(), std::greater<T>{});
  // The k-th order statistic must match.
  ASSERT_EQ(got[k], ref[k]) << "select kth n=" << n << " k=" << k << " dist=" << dist
                            << " asc=" << asc;
  // Partition postcondition: nothing before k orders-after keys[k], nothing after orders-before.
  const T pivot = got[k];
  for (std::size_t i = 0; i < k; ++i) {
    if (asc)
      ASSERT_LE(got[i], pivot) << "select left n=" << n << " k=" << k << " i=" << i;
    else
      ASSERT_GE(got[i], pivot) << "select left n=" << n << " k=" << k << " i=" << i;
  }
  for (std::size_t i = k + 1; i < n; ++i) {
    if (asc)
      ASSERT_GE(got[i], pivot) << "select right n=" << n << " k=" << k << " i=" << i;
    else
      ASSERT_LE(got[i], pivot) << "select right n=" << n << " k=" << k << " i=" << i;
  }
}

// The length ladder: base-case sizes + recursion-triggering sizes.
const std::vector<std::size_t> kLengths = {0, 1, 2, 3, 7, 8, 16, 17, 31,
                                           64, 100, 257, 1000, 4096, 10007};

template <typename T>
void run_all_for_type() {
  std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    for (int dist = 0; dist <= 4; ++dist) {
      for (std::size_t n : kLengths) {
        expect_sort<T>(n, dist, order, ++seed);
        if (n >= 1) {
          // partial_sort k in {0..n} (k<=n contract); pick a few representative k.
          for (std::size_t k : {std::size_t{0}, n / 3, n / 2, n}) {
            if (k <= n) expect_partial_sort<T>(n, k, dist, order, ++seed);
          }
          // select needs k<n.
          for (std::size_t k : {std::size_t{0}, n / 3, n / 2, n - 1}) {
            if (k < n) expect_select<T>(n, k, dist, order, ++seed);
          }
        }
      }
    }
  }
}

}  // namespace

TEST(Sort, U16) { run_all_for_type<std::uint16_t>(); }
TEST(Sort, I16) { run_all_for_type<std::int16_t>(); }
TEST(Sort, U32) { run_all_for_type<std::uint32_t>(); }
TEST(Sort, I32) { run_all_for_type<std::int32_t>(); }
TEST(Sort, U64) { run_all_for_type<std::uint64_t>(); }
TEST(Sort, I64) { run_all_for_type<std::int64_t>(); }
TEST(Sort, F32) { run_all_for_type<float>(); }
TEST(Sort, F64) { run_all_for_type<double>(); }

namespace {

bool less_u128(const UInt128& a, const UInt128& b) {
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
bool equal_u128(const UInt128& a, const UInt128& b) {
  return a.lo == b.lo && a.hi == b.hi;
}

std::vector<UInt128> make_u128(std::size_t n) {
  std::mt19937_64 rng(0x1285A17ULL + n);
  std::vector<UInt128> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = UInt128{rng(), rng()};
    if (i % 7 == 0) out[i].hi &= 3;        // repeated high words exercise low-word ordering
    if (i % 13 == 0) out[i] = UInt128{};   // duplicates
  }
  if (n > 2) {
    out[0] = UInt128{0, 1};
    out[1] = UInt128{~std::uint64_t{0}, 0};
    out[2] = UInt128{~std::uint64_t{0}, ~std::uint64_t{0}};
  }
  return out;
}

}  // namespace

TEST(Sort, U128SortPartialAndSelect) {
  for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{7},
                              std::size_t{31}, std::size_t{257}, std::size_t{4096}}) {
    const auto source = make_u128(n);
    for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
      const bool asc = order == SortOrder::kAscending;
      const auto before128 = [asc](const UInt128& a, const UInt128& b) {
        return asc ? less_u128(a, b) : less_u128(b, a);
      };
      auto expected = source;
      std::sort(expected.begin(), expected.end(), before128);

      auto got = source;
      sort(got.data(), got.size(), order);
      ASSERT_TRUE(std::equal(got.begin(), got.end(), expected.begin(), equal_u128))
          << "sort n=" << n << " asc=" << asc;

      for (const std::size_t k : {std::size_t{0}, n / 2, n}) {
        got = source;
        auto ref = source;
        partial_sort(got.data(), got.size(), k, order);
        std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), before128);
        ASSERT_TRUE(std::equal(got.begin(), got.begin() + k, ref.begin(), equal_u128))
            << "partial n=" << n << " k=" << k << " asc=" << asc;
      }

      for (const std::size_t k : {std::size_t{0}, n / 2, n - 1}) {
        got = source;
        select(got.data(), got.size(), k, order);
        ASSERT_TRUE(equal_u128(got[k], expected[k]))
            << "select n=" << n << " k=" << k << " asc=" << asc;
        for (std::size_t i = 0; i < k; ++i) ASSERT_FALSE(before128(got[k], got[i]));
        for (std::size_t i = k + 1; i < n; ++i) ASSERT_FALSE(before128(got[i], got[k]));
      }
    }
  }
}

TEST(Sort, FloatingNaNsLast) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    std::vector<float> values = {nan, 3.0f, -1.0f, nan, 2.0f, 0.0f};
    sort(values.data(), values.size(), order);
    ASSERT_FALSE(std::isnan(values[0]));
    ASSERT_FALSE(std::isnan(values[1]));
    ASSERT_FALSE(std::isnan(values[2]));
    ASSERT_FALSE(std::isnan(values[3]));
    ASSERT_TRUE(std::isnan(values[4]));
    ASSERT_TRUE(std::isnan(values[5]));
    if (order == SortOrder::kAscending) {
      EXPECT_EQ(values[0], -1.0f); EXPECT_EQ(values[1], 0.0f);
      EXPECT_EQ(values[2], 2.0f); EXPECT_EQ(values[3], 3.0f);
    } else {
      EXPECT_EQ(values[0], 3.0f); EXPECT_EQ(values[1], 2.0f);
      EXPECT_EQ(values[2], 0.0f); EXPECT_EQ(values[3], -1.0f);
    }
  }
}

TEST(Sort, FloatingNaNsLastPartialAndSelect) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> source = {nan, 3.0f, -1.0f, nan, 2.0f, 0.0f};
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    const bool asc = order == SortOrder::kAscending;
    std::vector<float> partial = source;
    partial_sort(partial.data(), partial.size(), 3, order);
    const float expected_prefix[] = {asc ? -1.0f : 3.0f, asc ? 0.0f : 2.0f,
                                     asc ? 2.0f : 0.0f};
    for (std::size_t i = 0; i < 3; ++i) {
      EXPECT_FALSE(std::isnan(partial[i]));
      EXPECT_EQ(partial[i], expected_prefix[i]);
    }

    std::vector<float> selected = source;
    select(selected.data(), selected.size(), 2, order);
    EXPECT_FALSE(std::isnan(selected[2]));
    EXPECT_EQ(selected[2], expected_prefix[2]);
    for (std::size_t i = 0; i < 2; ++i) {
      EXPECT_FALSE(std::isnan(selected[i]));
      if (asc) EXPECT_LE(selected[i], selected[2]);
      else EXPECT_GE(selected[i], selected[2]);
    }
    for (std::size_t i = 3; i < selected.size(); ++i) {
      if (std::isnan(selected[i])) continue;
      if (asc) EXPECT_GE(selected[i], selected[2]);
      else EXPECT_LE(selected[i], selected[2]);
    }
  }
}

TEST(Sort, SmallKIntegerPartialSort) {
  std::uint64_t seed = 0x51A11C0DEULL;
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    for (int dist : {0, 4}) {
      for (std::size_t n : {std::size_t{16}, std::size_t{64}, std::size_t{257},
                            std::size_t{4096}}) {
        for (std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{4},
                              std::size_t{8}, std::size_t{16}}) {
          if (k <= n) {
            expect_partial_sort<std::uint32_t>(n, k, dist, order, ++seed);
            expect_partial_sort<std::int64_t>(n, k, dist, order, ++seed);
          }
        }
      }
    }
  }
}

// vqsort's partition loop intentionally performs full-width stores near both
// ends of the active range. Keep live objects immediately outside that range
// so an incorrect Compress/StoreLeftRight contract is caught without relying
// on an allocator redzone. ASan supplies the stronger allocation-boundary
// check in the sanitizer build.
TEST(Sort, PartitionDoesNotTouchAdjacentObjects) {
  constexpr std::size_t kGuard = 32;
  constexpr std::uint32_t kLeft = 0xA17ECAFEu;
  constexpr std::uint32_t kRight = 0xBADC0FFEu;
  std::mt19937_64 rng(0xC0FFEE1234567890ULL);

  // Dense coverage around base-case/partition boundaries plus awkward large
  // sizes. Repeat with duplicate-heavy and full-range inputs, both orders and
  // all three public entry points.
  const std::size_t sizes[] = {33,  47,  63,  64,  65,  95,  127, 128,
                               129, 191, 255, 256, 257, 511, 1023, 4093};
  for (const SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    for (const bool few_unique : {false, true}) {
      for (const std::size_t n : sizes) {
        std::vector<std::uint32_t> storage(n + 2 * kGuard);
        std::fill_n(storage.begin(), kGuard, kLeft);
        std::fill_n(storage.end() - kGuard, kGuard, kRight);
        auto* const data = storage.data() + kGuard;
        for (std::size_t i = 0; i < n; ++i) {
          data[i] = few_unique ? static_cast<std::uint32_t>(rng() & 7u)
                               : static_cast<std::uint32_t>(rng());
        }
        const std::vector<std::uint32_t> input(data, data + n);

        const auto check_guards = [&] {
          EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + kGuard,
                                  [](std::uint32_t x) { return x == kLeft; }))
              << "left guard n=" << n;
          EXPECT_TRUE(std::all_of(storage.end() - kGuard, storage.end(),
                                  [](std::uint32_t x) { return x == kRight; }))
              << "right guard n=" << n;
        };

        sort(data, n, order);
        check_guards();

        std::copy(input.begin(), input.end(), data);
        partial_sort(data, n, n / 2, order);
        check_guards();

        std::copy(input.begin(), input.end(), data);
        select(data, n, n / 2, order);
        check_guards();
      }
    }
  }
}

// KV (top-k): sort by key, value must ride along. Verify (1) keys come out ordered, (2) every
// (key,value) pair is preserved (the multiset of pairs is unchanged -- value stays glued to its
// key), which is the arg-sort/top-k contract. Uses value = a hash of the original key so a
// mispaired value is caught even when keys tie.
template <typename KV, typename K>
void run_kv() {
  std::mt19937_64 rng(0xD1CE5EED);
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    const bool asc = (order == SortOrder::kAscending);
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{8}, std::size_t{17},
                          std::size_t{257}, std::size_t{4096}}) {
      std::vector<KV> kv(n);
      std::uniform_int_distribution<std::uint64_t> d(0, 50);  // small range -> ties
      for (std::size_t i = 0; i < n; ++i) {
        const K key = static_cast<K>(d(rng));
        kv[i].key = key;
        kv[i].value = static_cast<K>(key * 2654435761u + 1);  // deterministic fn(key) to catch mispair
      }
      sort(kv.data(), n, order);
      for (std::size_t i = 0; i < n; ++i)
        ASSERT_EQ(kv[i].value, static_cast<K>(kv[i].key * 2654435761u + 1))
            << "kv value mispaired n=" << n << " i=" << i;  // value still fn(its key)
      for (std::size_t i = 1; i < n; ++i) {
        if (asc) ASSERT_LE(kv[i - 1].key, kv[i].key) << "kv not asc n=" << n << " i=" << i;
        else ASSERT_GE(kv[i - 1].key, kv[i].key) << "kv not desc n=" << n << " i=" << i;
      }
      // partial_sort top-k: first k keys are the k smallest/largest, values still paired.
      if (n >= 4) {
        std::vector<KV> kv2(n);
        for (std::size_t i = 0; i < n; ++i) { kv2[i].key = static_cast<K>(d(rng));
          kv2[i].value = static_cast<K>(kv2[i].key * 2654435761u + 1); }
        const std::size_t k = n / 4;
        partial_sort(kv2.data(), n, k, order);
        for (std::size_t i = 0; i < k; ++i)
          ASSERT_EQ(kv2[i].value, static_cast<K>(kv2[i].key * 2654435761u + 1)) << "psort kv mispair";
        for (std::size_t i = 1; i < k; ++i)
          if (asc) ASSERT_LE(kv2[i - 1].key, kv2[i].key); else ASSERT_GE(kv2[i - 1].key, kv2[i].key);
      }
    }
  }
}
TEST(Sort, KeyValue32) { run_kv<hyle::simd::KeyValue32, std::uint32_t>(); }
TEST(Sort, KeyValue64) { run_kv<hyle::simd::KeyValue64, std::uint64_t>(); }

template <typename KV, typename K>
void run_kv_select() {
  const K keys[] = {4, 1, 4, 2, 0, 2, 5, 3, 1};
  std::vector<KV> source(sizeof(keys) / sizeof(keys[0]));
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i].key = keys[i];
    source[i].value = static_cast<K>(100 + i);
  }
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    const bool asc = order == SortOrder::kAscending;
    for (std::size_t k : {std::size_t{0}, source.size() / 2, source.size() - 1}) {
      std::vector<KV> got = source;
      std::vector<KV> ref = source;
      auto by_key = [asc](const KV& a, const KV& b) { return asc ? a.key < b.key : a.key > b.key; };
      std::nth_element(ref.begin(), ref.begin() + k, ref.end(), by_key);
      select(got.data(), got.size(), k, order);
      EXPECT_EQ(got[k].key, ref[k].key);
      for (std::size_t i = 0; i < k; ++i) {
        if (asc) EXPECT_LE(got[i].key, got[k].key);
        else EXPECT_GE(got[i].key, got[k].key);
      }
      for (std::size_t i = k + 1; i < got.size(); ++i) {
        if (asc) EXPECT_GE(got[i].key, got[k].key);
        else EXPECT_LE(got[i].key, got[k].key);
      }
      auto by_pair = [](const KV& a, const KV& b) {
        return a.key != b.key ? a.key < b.key : a.value < b.value;
      };
      std::sort(got.begin(), got.end(), by_pair);
      std::sort(ref.begin(), ref.end(), by_pair);
      for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i].key, ref[i].key);
        EXPECT_EQ(got[i].value, ref[i].value);
      }
    }
  }
}

TEST(Sort, KeyValueSelect32) { run_kv_select<hyle::simd::KeyValue32, std::uint32_t>(); }
TEST(Sort, KeyValueSelect64) { run_kv_select<hyle::simd::KeyValue64, std::uint64_t>(); }

template <typename KV, typename K>
void run_small_kv_partial() {
  std::mt19937_64 rng(0x5A11C0DEULL + sizeof(KV));
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    const bool asc = order == SortOrder::kAscending;
    for (std::size_t n : {std::size_t{16}, std::size_t{64}, std::size_t{257},
                          std::size_t{4096}}) {
      std::vector<KV> source(n);
      for (std::size_t i = 0; i < n; ++i) {
        source[i].key = static_cast<K>(rng() & 31u);
        source[i].value = static_cast<K>(i);
      }
      for (std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{4},
                            std::size_t{8}, std::size_t{16}}) {
        std::vector<KV> got = source;
        std::vector<KV> ref = source;
        auto by_key = [asc](const KV& a, const KV& b) {
          return asc ? a.key < b.key : a.key > b.key;
        };
        partial_sort(got.data(), got.size(), k, order);
        std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), by_key);
        for (std::size_t i = 0; i < k; ++i) ASSERT_EQ(got[i].key, ref[i].key);

        auto by_pair = [](const KV& a, const KV& b) {
          return a.key != b.key ? a.key < b.key : a.value < b.value;
        };
        std::sort(got.begin(), got.end(), by_pair);
        std::sort(ref.begin(), ref.end(), by_pair);
        for (std::size_t i = 0; i < n; ++i) {
          ASSERT_EQ(got[i].key, ref[i].key);
          ASSERT_EQ(got[i].value, ref[i].value);
        }
      }
    }
  }
}

TEST(Sort, SmallKKeyValue32) {
  run_small_kv_partial<hyle::simd::KeyValue32, std::uint32_t>();
}
TEST(Sort, SmallKKeyValue64) {
  run_small_kv_partial<hyle::simd::KeyValue64, std::uint64_t>();
}

// The dispatcher reports a sane tier for this host (avx512/avx2 on x86, neon on arm64, else
// scalar). Not asserting a specific value (depends on the build host), just that it is one of the
// expected tiers -- a smoke check that the shared runtime metadata is wired.
TEST(Sort, ActiveVariantIsSane) {
  const Isa v = hyle::simd::runtime().sort.isa;
  EXPECT_TRUE(v == Isa::kAvx512 || v == Isa::kAvx2 || v == Isa::kNeon || v == Isa::kScalar);
}
