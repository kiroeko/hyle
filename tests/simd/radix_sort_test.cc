#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <new>
#include <random>
#include <span>
#include <type_traits>
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

template <typename T>
void check_nonallocating(std::size_t n, SortOrder order, std::uint64_t seed) {
  auto values = make_random<T>(n, seed);
  auto expected = values;
  std::vector<T> scratch(hyle::simd::radix_sort_workspace_size(n));
  radix_sort(values.data(), values.size(), std::span<T>(scratch), order);
  if (order == SortOrder::kAscending)
    std::sort(expected.begin(), expected.end());
  else
    std::sort(expected.begin(), expected.end(), std::greater<T>());
  EXPECT_EQ(values, expected);

  auto keys = make_random<T>(n, seed + 1);
  std::vector<std::uint32_t> indices(n);
  std::vector<std::uint32_t> index_scratch(hyle::simd::argsort_workspace_size(n));
  argsort(keys.data(), indices.data(), n, std::span<std::uint32_t>(index_scratch), order);
  for (std::size_t i = 1; i < n; ++i) {
    if (order == SortOrder::kAscending)
      EXPECT_LE(keys[indices[i - 1]], keys[indices[i]]);
    else
      EXPECT_GE(keys[indices[i - 1]], keys[indices[i]]);
  }
}

void radix_exact_overlap() {
  std::uint32_t values[4] = {3, 2, 1, 0};
  radix_sort(values, 4, std::span<std::uint32_t>(values, 4));
}

void radix_partial_overlap() {
  std::uint32_t values[5] = {4, 3, 2, 1, 0};
  radix_sort(values, 4, std::span<std::uint32_t>(values + 1, 4));
}

void radix_short_scratch() {
  std::uint32_t values[4] = {3, 2, 1, 0};
  std::uint32_t scratch[3] = {};
  radix_sort(values, 4, std::span<std::uint32_t>(scratch, 3));
}

void argsort_keys_indices_overlap() {
  std::uint32_t keys[4] = {3, 2, 1, 0};
  std::uint32_t scratch[4] = {};
  argsort(keys, keys, 4, std::span<std::uint32_t>(scratch, 4));
}

void argsort_keys_scratch_overlap() {
  std::uint32_t keys_and_scratch[4] = {3, 2, 1, 0};
  std::uint32_t indices[4] = {};
  argsort(keys_and_scratch, indices, 4,
          std::span<std::uint32_t>(keys_and_scratch, 4));
}

void argsort_indices_scratch_overlap() {
  std::uint32_t keys[4] = {3, 2, 1, 0};
  std::uint32_t indices_and_scratch[4] = {};
  argsort(keys, indices_and_scratch, 4,
          std::span<std::uint32_t>(indices_and_scratch, 4));
}

void argsort_too_many_indices() {
  std::uint32_t key = 0;
  std::uint32_t index = 0;
  const std::size_t too_many =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 2;
  argsort(&key, &index, too_many,
          std::span<std::uint32_t>(&index, std::size_t{1}));
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

TEST(RadixSort, SixteenBitTypes) {
  check_radix<std::uint16_t>(5000, SortOrder::kAscending, 44);
  check_radix<std::uint16_t>(5000, SortOrder::kDescending, 45);
  check_radix<std::int16_t>(5000, SortOrder::kAscending, 46);
  check_radix<std::int16_t>(5000, SortOrder::kDescending, 47);
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

TEST(RadixSort, F64Finite) {
  check_radix<double>(8000, SortOrder::kAscending, 13);
  check_radix<double>(8000, SortOrder::kDescending, 14);
}

TEST(RadixSort, CallerOwnedScratchCoversEveryKeyWidth) {
  check_nonallocating<std::uint16_t>(4097, SortOrder::kAscending, 20);
  check_nonallocating<std::int16_t>(4097, SortOrder::kDescending, 21);
  check_nonallocating<std::uint32_t>(4097, SortOrder::kAscending, 22);
  check_nonallocating<std::int32_t>(4097, SortOrder::kDescending, 23);
  check_nonallocating<std::uint64_t>(4097, SortOrder::kAscending, 24);
  check_nonallocating<std::int64_t>(4097, SortOrder::kDescending, 25);
  check_nonallocating<float>(4097, SortOrder::kAscending, 26);
  check_nonallocating<double>(4097, SortOrder::kDescending, 27);
}

TEST(RadixSort, FloatingNaNsLastAndEquivalentValuesStable) {
  const float nan_a = std::bit_cast<float>(std::uint32_t{0x7fc00001u});
  const float nan_b = std::bit_cast<float>(std::uint32_t{0xffc00002u});
  const float keys[] = {nan_a, 0.0f, -0.0f, 3.0f, nan_b, -2.0f, 0.0f};
  constexpr std::size_t n = std::size(keys);
  std::uint32_t indices[n];
  std::uint32_t scratch[n];

  argsort(keys, indices, n, std::span<std::uint32_t>(scratch),
          SortOrder::kAscending);
  const std::uint32_t expected_ascending[] = {5, 1, 2, 6, 3, 0, 4};
  EXPECT_TRUE(std::equal(indices, indices + n, expected_ascending));

  argsort(keys, indices, n, std::span<std::uint32_t>(scratch),
          SortOrder::kDescending);
  const std::uint32_t expected_descending[] = {3, 1, 2, 6, 5, 0, 4};
  EXPECT_TRUE(std::equal(indices, indices + n, expected_descending));
}

TEST(RadixSort, FloatingValuePayloadsAndSignedZerosRemainStable) {
  const float nan_a = std::bit_cast<float>(std::uint32_t{0x7fc00001u});
  const float nan_b = std::bit_cast<float>(std::uint32_t{0xffc00002u});
  const std::vector<float> source = {nan_a, 0.0f, -0.0f, 1.0f,
                                     nan_b, -1.0f, 0.0f};
  const auto bits = [](float x) { return std::bit_cast<std::uint32_t>(x); };
  for (SortOrder order : {SortOrder::kAscending, SortOrder::kDescending}) {
    auto values = source;
    std::vector<float> scratch(values.size());
    radix_sort(values.data(), values.size(), std::span<float>(scratch), order);
    const std::uint32_t expected_ascending[] = {
        bits(-1.0f), bits(0.0f), bits(-0.0f), bits(0.0f),
        bits(1.0f), bits(nan_a), bits(nan_b)};
    const std::uint32_t expected_descending[] = {
        bits(1.0f), bits(0.0f), bits(-0.0f), bits(0.0f),
        bits(-1.0f), bits(nan_a), bits(nan_b)};
    const auto* expected = order == SortOrder::kAscending
                               ? expected_ascending
                               : expected_descending;
    for (std::size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(bits(values[i]), expected[i]);
    }
  }
}

TEST(RadixSort, WorkspaceAndLiveRangesMustBeDisjoint) {
  EXPECT_DEATH(radix_exact_overlap(), "HYLE fatal");
  EXPECT_DEATH(radix_partial_overlap(), "HYLE fatal");
  EXPECT_DEATH(radix_short_scratch(), "HYLE fatal");
}

TEST(RadixSort, ArgsortRejectsEveryOverlapPair) {
  EXPECT_DEATH(argsort_keys_indices_overlap(), "HYLE fatal");
  EXPECT_DEATH(argsort_keys_scratch_overlap(), "HYLE fatal");
  EXPECT_DEATH(argsort_indices_scratch_overlap(), "HYLE fatal");
}

TEST(RadixSort, AllocatingConveniencePropagatesAllocationFailure) {
  std::uint64_t key = 0;
  EXPECT_THROW(radix_sort(&key, std::numeric_limits<std::size_t>::max()),
               std::bad_alloc);
}

TEST(RadixSort, ArgsortRejectsIndicesBeyondUint32) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    EXPECT_DEATH(argsort_too_many_indices(), "HYLE fatal");
  }
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
