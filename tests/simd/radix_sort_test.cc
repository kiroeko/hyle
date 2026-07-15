#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "hyle/simd/sort.h"

using hyle::simd::argsort;
using hyle::simd::radix_sort;
using hyle::simd::SortOrder;

namespace {

template <typename T>
std::vector<T> make_random(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<T> v(n);
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<float> d(-1e6f, 1e6f);
    for (auto& e : v) e = static_cast<T>(d(rng));
  } else {
    std::uniform_int_distribution<std::uint64_t> d;
    for (auto& e : v) e = static_cast<T>(d(rng));
  }
  return v;
}

template <typename T>
void check_radix(std::size_t n, SortOrder order, std::uint64_t seed) {
  auto a = make_random<T>(n, seed);
  auto b = a;
  radix_sort(a.data(), n, order);
  if (order == SortOrder::kAscending) std::sort(b.begin(), b.end());
  else std::sort(b.begin(), b.end(), std::greater<T>());
  for (std::size_t i = 0; i < n; ++i) {
    if constexpr (std::is_floating_point_v<T>)
      EXPECT_EQ(std::memcmp(&a[i], &b[i], sizeof(T)), 0) << "i=" << i << " n=" << n;
    else
      EXPECT_EQ(a[i], b[i]) << "i=" << i << " n=" << n;
  }
}

template <typename T>
void check_stable(std::size_t n) {
  // Many equal keys: stable means relative order of equals preserved via argsort on indices.
  std::vector<T> keys(n);
  for (std::size_t i = 0; i < n; ++i) keys[i] = static_cast<T>(i % 7);  // 7 unique
  std::vector<std::uint32_t> idx(n);
  argsort(keys.data(), idx.data(), n, SortOrder::kAscending);
  for (std::size_t i = 1; i < n; ++i) {
    EXPECT_LE(keys[idx[i - 1]], keys[idx[i]]);
    if (keys[idx[i - 1]] == keys[idx[i]]) {
      EXPECT_LT(idx[i - 1], idx[i]) << "stability broken at i=" << i;
    }
  }
}

}  // namespace

TEST(RadixSort, U32Sizes) {
  for (std::size_t n : {0u, 1u, 2u, 3u, 16u, 256u, 1000u, 4096u, 10000u}) {
    check_radix<std::uint32_t>(n, SortOrder::kAscending, 1 + n);
    check_radix<std::uint32_t>(n, SortOrder::kDescending, 2 + n);
  }
}

TEST(RadixSort, I32) {
  check_radix<std::int32_t>(5000, SortOrder::kAscending, 42);
  check_radix<std::int32_t>(5000, SortOrder::kDescending, 43);
}

TEST(RadixSort, U64) {
  check_radix<std::uint64_t>(3000, SortOrder::kAscending, 7);
  check_radix<std::uint64_t>(3000, SortOrder::kDescending, 8);
}

TEST(RadixSort, I64) {
  check_radix<std::int64_t>(3000, SortOrder::kAscending, 9);
  check_radix<std::int64_t>(3000, SortOrder::kDescending, 10);
}

TEST(RadixSort, F32Finite) {
  check_radix<float>(8000, SortOrder::kAscending, 11);
  check_radix<float>(8000, SortOrder::kDescending, 12);
}

TEST(RadixSort, StableArgsortU32) { check_stable<std::uint32_t>(5000); }
TEST(RadixSort, StableArgsortI32) { check_stable<std::int32_t>(5000); }

TEST(RadixSort, ArgsortMatchesSortedKeys) {
  auto keys = make_random<std::uint32_t>(2000, 99);
  std::vector<std::uint32_t> idx(keys.size());
  argsort(keys.data(), idx.data(), keys.size(), SortOrder::kAscending);
  std::vector<std::uint32_t> sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < keys.size(); ++i) EXPECT_EQ(keys[idx[i]], sorted[i]);
}

TEST(RadixSort, PublicSortUsesRadixForLargeU32) {
  // n >= kRadixSortMinN should still produce a correct total order.
  constexpr std::size_t n = hyle::simd::kRadixSortMinN;
  check_radix<std::uint32_t>(n, SortOrder::kAscending, 123);
  auto a = make_random<std::uint32_t>(n, 124);
  auto b = a;
  hyle::simd::sort(a.data(), n, SortOrder::kAscending);
  std::sort(b.begin(), b.end());
  EXPECT_EQ(a, b);
}
