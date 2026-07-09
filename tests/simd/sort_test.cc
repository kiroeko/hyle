#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "hyle/simd/dispatch.h"
#include "hyle/simd/sort.h"

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

using hyle::simd::active_sort_variant;
using hyle::simd::Isa;
using hyle::simd::partial_sort;
using hyle::simd::select;
using hyle::simd::sort;
using hyle::simd::SortOrder;

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

TEST(Sort, U32) { run_all_for_type<std::uint32_t>(); }
TEST(Sort, I32) { run_all_for_type<std::int32_t>(); }
TEST(Sort, U64) { run_all_for_type<std::uint64_t>(); }
TEST(Sort, I64) { run_all_for_type<std::int64_t>(); }
TEST(Sort, F32) { run_all_for_type<float>(); }
TEST(Sort, F64) { run_all_for_type<double>(); }

// The dispatcher reports a sane tier for this host (avx512/avx2 on x86, neon on arm64, else
// scalar). Not asserting a specific value (depends on the build host), just that it is one of the
// expected tiers -- a smoke check that active_sort_variant is wired.
TEST(Sort, ActiveVariantIsSane) {
  const Isa v = active_sort_variant();
  EXPECT_TRUE(v == Isa::kAvx512 || v == Isa::kAvx2 || v == Isa::kNeon || v == Isa::kScalar);
}
